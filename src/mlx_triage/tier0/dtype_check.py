"""Check 0.1: Dtype Compatibility Audit.

Reads config.json for the training dtype and safetensors file headers for
the stored dtype, then flags mismatches that are known to cause silent
inference failures on Apple Silicon / MLX.

Key failure modes (from research evidence):
- BF16 training -> FP16 stored = CRITICAL
    Gemma 3: activations overflow, FP16 max is 65504
- FP32 training -> FP16 stored = WARNING (precision loss)
- Quantized models (quantization_config present) = PASS (expected)
- Matching dtypes = PASS
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from mlx_triage.models import CheckStatus, DiagnosticResult

# Safetensors dtype codes -> canonical names
_SAFETENSORS_DTYPE_MAP: dict[str, str] = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool",
}

# Canonical torch_dtype strings -> normalised names
_TORCH_DTYPE_MAP: dict[str, str] = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
}

CHECK_ID = "0.1"
CHECK_NAME = "Dtype Compatibility"


def _read_safetensors_header(path: Path) -> dict:
    """Read the JSON header from a safetensors file without loading tensors.

    Format: 8-byte uint64 LE header length, then UTF-8 JSON of that length.
    """
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            msg = f"Safetensors file too short: {path}"
            raise ValueError(msg)
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
    return json.loads(header_bytes)


def _dominant_stored_dtype(model_path: Path) -> str | None:
    """Determine the dominant stored dtype across all safetensors files.

    Returns the most-common dtype (by tensor count) as a canonical name,
    or None if no safetensors files are found.
    """
    dtype_counts: dict[str, int] = {}

    for sf_path in sorted(model_path.glob("*.safetensors")):
        header = _read_safetensors_header(sf_path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            raw_dtype = meta.get("dtype", "")
            canonical = _SAFETENSORS_DTYPE_MAP.get(raw_dtype, raw_dtype.lower())
            dtype_counts[canonical] = dtype_counts.get(canonical, 0) + 1

    if not dtype_counts:
        return None

    # Return the most-common dtype
    return max(dtype_counts, key=lambda d: dtype_counts[d])


def check_dtype_compatibility(model_path: str) -> DiagnosticResult:
    """Run dtype compatibility audit on a model directory.

    Args:
        model_path: Path to the model directory containing config.json
                    and *.safetensors files.

    Returns:
        DiagnosticResult with status, detail, optional remediation, and metadata.
    """
    root = Path(model_path)
    config_path = root / "config.json"

    # --- Guard: config.json must exist ---
    if not config_path.exists():
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail="config.json not found in model directory.",
            metadata={"model_path": model_path},
        )

    with open(config_path) as f:
        config = json.load(f)

    # --- Determine training dtype ---
    raw_training = config.get("torch_dtype")
    training_dtype = _TORCH_DTYPE_MAP.get(raw_training, raw_training) if raw_training else None

    # --- Check for quantization ---
    is_quantized = "quantization_config" in config

    if is_quantized:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.PASS,
            detail=(
                f"Quantized model (quantization_config present). "
                f"Training dtype: {training_dtype or 'unknown'}. "
                f"Dtype mismatch check not applicable for quantized weights."
            ),
            metadata={
                "training_dtype": training_dtype,
                "stored_dtype": "quantized",
                "quantization_config": config["quantization_config"],
            },
        )

    # --- Determine stored dtype from safetensors ---
    stored_dtype = _dominant_stored_dtype(root)

    if stored_dtype is None:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail="No safetensors files found in model directory.",
            metadata={"model_path": model_path, "training_dtype": training_dtype},
        )

    if training_dtype is None:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.WARNING,
            detail=(
                f"torch_dtype not specified in config.json. "
                f"Stored dtype: {stored_dtype}. Cannot verify compatibility."
            ),
            metadata={"training_dtype": None, "stored_dtype": stored_dtype},
        )

    # --- Evaluate compatibility ---
    metadata = {
        "training_dtype": training_dtype,
        "stored_dtype": stored_dtype,
    }

    # CRITICAL: BF16 training -> FP16 stored
    # Gemma 3 failure mode: activations reach 800K after layer norm,
    # FP16 max is 65504 -> overflow -> NaN -> garbage output
    if training_dtype == "bfloat16" and stored_dtype == "float16":
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.CRITICAL,
            detail=(
                f"BF16 training dtype but FP16 stored weights detected. "
                f"BF16 values can exceed FP16 max (65504), causing overflow "
                f"and NaN propagation. This is the Gemma 3 failure mode."
            ),
            remediation=(
                "Re-download or re-convert the model with BF16 weights preserved. "
                "Do not cast BF16-trained models to FP16."
            ),
            metadata=metadata,
        )

    # WARNING: FP32 training -> FP16 stored (precision loss)
    if training_dtype == "float32" and stored_dtype == "float16":
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.WARNING,
            detail=(
                f"FP32 training dtype but FP16 stored weights. "
                f"Potential precision loss from float32 -> float16 downcast."
            ),
            remediation=(
                "Consider using BF16 or FP32 weights if quality issues are observed. "
                "FP16 has limited dynamic range compared to the training precision."
            ),
            metadata=metadata,
        )

    # PASS: matching or compatible dtypes
    return DiagnosticResult(
        check_id=CHECK_ID,
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        detail=(
            f"Training dtype {training_dtype}, stored as {stored_dtype}. Compatible."
        ),
        metadata=metadata,
    )
