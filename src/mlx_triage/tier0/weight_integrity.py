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
import logging
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
