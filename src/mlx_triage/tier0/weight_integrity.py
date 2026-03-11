"""Check 0.3: Weight File Integrity.

Loads a sample of tensors from safetensors files and checks for numerical
anomalies that indicate corrupted or incomplete weight files.

Failure modes detected:
- NaN values (structurally valid but numerically garbage)
- Inf values
- All-zero layers (may indicate incomplete download)
"""

from __future__ import annotations

import glob
import json
import logging
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from mlx_triage.models import CheckStatus, DiagnosticResult

logger = logging.getLogger(__name__)

CHECK_ID = "0.3"
CHECK_NAME = "Weight File Integrity"

# Sampling limits to keep the check fast (<30s tier)
MAX_SHARDS = 3
MAX_TENSORS_PER_SHARD = 5

# Tensors with fewer elements than this are not flagged for all-zero
ALL_ZERO_MIN_ELEMENTS = 16

# Dtypes that numpy cannot load (require special handling)
_NUMPY_UNSUPPORTED_DTYPES = {"BF16"}


def _read_shard_dtypes(shard_path: str) -> set[str]:
    """Read the safetensors header and return the set of dtypes used."""
    try:
        with open(shard_path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                return set()
            header_len = struct.unpack("<Q", raw_len)[0]
            header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        return {
            meta.get("dtype", "")
            for key, meta in header.items()
            if key != "__metadata__" and isinstance(meta, dict)
        }
    except Exception:
        return set()


def _check_bf16_with_mlx(
    sampled_shards: list[str],
    total_shards: int,
) -> DiagnosticResult:
    """Inspect BF16 safetensors with MLX if available."""
    try:
        import mlx.core as mx
    except ImportError:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.INFO,
            detail=(
                "Weight files use BF16 format, which cannot be inspected with numpy, "
                "and MLX is not available for BF16 tensor checks."
            ),
            metadata={
                "shard_count": total_shards,
                "dtype": "BF16",
                "reason": "numpy_unsupported_dtype",
                "inspection_backend": "numpy",
            },
        )

    nan_tensors: list[str] = []
    inf_tensors: list[str] = []
    zero_tensors: list[str] = []
    load_errors: list[str] = []
    total_tensors_checked = 0

    def _to_bool(value: object) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)

    for shard_path in sampled_shards:
        shard_name = Path(shard_path).name
        try:
            tensors = mx.load(shard_path)
        except Exception as exc:
            logger.debug("Failed to load BF16 shard with MLX %s: %s", shard_name, exc)
            load_errors.append(f"{shard_name}: {exc}")
            continue

        tensor_names = list(tensors.keys())[:MAX_TENSORS_PER_SHARD]
        for name in tensor_names:
            arr = tensors[name]
            total_tensors_checked += 1

            try:
                has_mx_reductions = all(
                    hasattr(mx, fn) for fn in ("any", "isnan", "isinf", "all")
                )

                if has_mx_reductions:
                    has_nan = _to_bool(mx.any(mx.isnan(arr)))
                    has_inf = _to_bool(mx.any(mx.isinf(arr)))
                    arr_size = int(arr.size) if hasattr(arr, "size") else 0
                    is_all_zero = arr_size >= ALL_ZERO_MIN_ELEMENTS and _to_bool(
                        mx.all(arr == 0)
                    )
                else:
                    arr_np = np.asarray(arr)
                    has_nan = bool(np.any(np.isnan(arr_np)))
                    has_inf = bool(np.any(np.isinf(arr_np)))
                    arr_size = int(arr_np.size)
                    is_all_zero = arr_size >= ALL_ZERO_MIN_ELEMENTS and bool(
                        np.all(arr_np == 0)
                    )
            except Exception as exc:
                load_errors.append(f"{shard_name}::{name}: {exc}")
                continue

            if has_nan:
                nan_tensors.append(f"{shard_name}::{name}")

            if has_inf:
                inf_tensors.append(f"{shard_name}::{name}")

            if is_all_zero:
                zero_tensors.append(f"{shard_name}::{name}")

    metadata: dict[str, object] = {
        "shard_count": total_shards,
        "shards_sampled": len(sampled_shards),
        "tensor_count": total_tensors_checked,
        "dtype": "BF16",
        "inspection_backend": "mlx",
    }
    if load_errors:
        metadata["load_errors"] = load_errors

    if nan_tensors or inf_tensors:
        problems: list[str] = []
        if nan_tensors:
            problems.append(f"NaN detected in {len(nan_tensors)} tensor(s): {nan_tensors}")
            metadata["nan_tensors"] = nan_tensors
        if inf_tensors:
            problems.append(f"Inf detected in {len(inf_tensors)} tensor(s): {inf_tensors}")
            metadata["inf_tensors"] = inf_tensors
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.CRITICAL,
            detail=" | ".join(problems),
            remediation="Re-download the model weights. NaN/Inf values indicate corrupted "
            "weight files that will produce garbage output.",
            metadata=metadata,
        )

    if zero_tensors:
        metadata["zero_tensors"] = zero_tensors
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.WARNING,
            detail=f"All-zero tensors found in {len(zero_tensors)} layer(s): {zero_tensors}",
            remediation="Re-download the model. All-zero weight layers may indicate an "
            "incomplete or interrupted download.",
            metadata=metadata,
        )

    if load_errors and total_tensors_checked == 0:
        metadata["reason"] = "mlx_load_failed"
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.INFO,
            detail=f"Could not load BF16 safetensors shards with MLX: {load_errors}",
            metadata=metadata,
        )

    return DiagnosticResult(
        check_id=CHECK_ID,
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        detail=(
            f"Checked {total_tensors_checked} BF16 tensor(s) across "
            f"{len(sampled_shards)} shard(s) with MLX — no anomalies found."
        ),
        metadata=metadata,
    )


def check_weight_integrity(model_path: str) -> DiagnosticResult:
    """Check safetensors weight files for NaN, Inf, and all-zero tensors.

    Samples up to MAX_TENSORS_PER_SHARD tensors from up to MAX_SHARDS shard
    files.  Returns a DiagnosticResult with status:

    - CRITICAL: NaN or Inf values detected
    - WARNING: All-zero layers found (possible incomplete download)
    - INFO: No safetensors files found (check skipped)
    - PASS: Weights look clean

    Args:
        model_path: Path to the model directory containing safetensors files.

    Returns:
        A DiagnosticResult for check 0.3.
    """
    model_dir = Path(model_path)

    # Discover safetensors files
    shard_paths = sorted(glob.glob(str(model_dir / "*.safetensors")))

    if not shard_paths:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.INFO,
            detail="No safetensors files found; weight integrity check skipped.",
            metadata={"shard_count": 0, "tensor_count": 0},
        )

    # Cap to MAX_SHARDS
    sampled_shards = shard_paths[:MAX_SHARDS]

    # Check for BF16-only shards before attempting numpy load.
    # In that case, use MLX-based tensor loading when available.
    all_bf16 = True
    for sp in sampled_shards:
        dtypes = _read_shard_dtypes(sp)
        if not dtypes or not dtypes.issubset(_NUMPY_UNSUPPORTED_DTYPES):
            all_bf16 = False
            break

    if all_bf16:
        return _check_bf16_with_mlx(sampled_shards, total_shards=len(shard_paths))

    nan_tensors: list[str] = []
    inf_tensors: list[str] = []
    zero_tensors: list[str] = []
    total_tensors_checked = 0
    load_errors: list[str] = []

    for shard_path in sampled_shards:
        shard_name = Path(shard_path).name
        try:
            tensors = load_file(shard_path)
        except Exception as exc:
            logger.debug("Failed to load %s: %s", shard_name, exc)
            load_errors.append(f"{shard_name}: {exc}")
            continue

        # Sample up to MAX_TENSORS_PER_SHARD tensors
        tensor_names = list(tensors.keys())[:MAX_TENSORS_PER_SHARD]

        for name in tensor_names:
            arr = tensors[name]
            total_tensors_checked += 1

            # Check for NaN
            if np.any(np.isnan(arr)):
                nan_tensors.append(f"{shard_name}::{name}")

            # Check for Inf
            if np.any(np.isinf(arr)):
                inf_tensors.append(f"{shard_name}::{name}")

            # Check for all-zero (only flag if tensor is large enough)
            if arr.size >= ALL_ZERO_MIN_ELEMENTS and np.all(arr == 0):
                zero_tensors.append(f"{shard_name}::{name}")

    # Build metadata
    metadata: dict[str, object] = {
        "shard_count": len(shard_paths),
        "shards_sampled": len(sampled_shards),
        "tensor_count": total_tensors_checked,
    }

    if load_errors:
        metadata["load_errors"] = load_errors

    # Determine status — NaN/Inf are CRITICAL, all-zero is WARNING
    if nan_tensors or inf_tensors:
        problems: list[str] = []
        if nan_tensors:
            problems.append(f"NaN detected in {len(nan_tensors)} tensor(s): {nan_tensors}")
            metadata["nan_tensors"] = nan_tensors
        if inf_tensors:
            problems.append(f"Inf detected in {len(inf_tensors)} tensor(s): {inf_tensors}")
            metadata["inf_tensors"] = inf_tensors
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.CRITICAL,
            detail=" | ".join(problems),
            remediation="Re-download the model weights. NaN/Inf values indicate corrupted "
            "weight files that will produce garbage output.",
            metadata=metadata,
        )

    if zero_tensors:
        metadata["zero_tensors"] = zero_tensors
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.WARNING,
            detail=f"All-zero tensors found in {len(zero_tensors)} layer(s): {zero_tensors}",
            remediation="Re-download the model. All-zero weight layers may indicate an "
            "incomplete or interrupted download.",
            metadata=metadata,
        )

    if load_errors and total_tensors_checked == 0:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.INFO,
            detail=f"Could not load any safetensors shards: {load_errors}",
            metadata=metadata,
        )

    return DiagnosticResult(
        check_id=CHECK_ID,
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        detail=f"Checked {total_tensors_checked} tensor(s) across "
        f"{len(sampled_shards)} shard(s) — no anomalies found.",
        metadata=metadata,
    )
