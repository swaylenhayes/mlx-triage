"""Tests for Check 0.1: Dtype Compatibility Audit."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.dtype_check import check_dtype_compatibility


def _create_safetensors(path: Path, dtype: str = "BF16", num_tensors: int = 1) -> None:
    """Create a minimal safetensors file with a valid header.

    Duplicated from conftest for use in inline test helpers that build
    custom model directories (conftest fixtures cover the common cases).
    """
    dtype_sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1}
    elem_size = dtype_sizes.get(dtype, 4)
    shape = [2, 2]
    data_size = shape[0] * shape[1] * elem_size

    header: dict = {}
    offset = 0
    for i in range(num_tensors):
        header[f"model.layers.{i}.weight"] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + data_size],
        }
        offset += data_size

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * offset)


class TestDtypeCheckPass:
    """Cases that should produce PASS status."""

    def test_good_model_bf16_match(self, good_model: Path) -> None:
        """BF16 training dtype with BF16 stored weights should PASS."""
        result = check_dtype_compatibility(str(good_model))
        assert result.status == CheckStatus.PASS
        assert result.check_id == "0.1"
        assert "bfloat16" in result.detail.lower() or "bf16" in result.detail.lower()

    def test_quantized_model_skips_mismatch(self, quantized_model: Path) -> None:
        """Quantized models should PASS regardless of stored dtype."""
        result = check_dtype_compatibility(str(quantized_model))
        assert result.status == CheckStatus.PASS
        assert "quantiz" in result.detail.lower()

    def test_fp32_training_fp32_stored(self, tmp_path: Path) -> None:
        """FP32 training with FP32 stored weights should PASS."""
        d = tmp_path / "fp32-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "llama", "torch_dtype": "float32"})
        )
        _create_safetensors(d / "model.safetensors", dtype="F32")
        result = check_dtype_compatibility(str(d))
        assert result.status == CheckStatus.PASS


class TestDtypeCheckCritical:
    """Cases that should produce CRITICAL status."""

    def test_bf16_to_fp16_mismatch(self, bf16_to_fp16_model: Path) -> None:
        """BF16 training dtype but FP16 stored = CRITICAL (Gemma 3 failure mode)."""
        result = check_dtype_compatibility(str(bf16_to_fp16_model))
        assert result.status == CheckStatus.CRITICAL
        assert result.remediation is not None
        assert "overflow" in result.detail.lower() or "bf16" in result.detail.lower()


class TestDtypeCheckWarning:
    """Cases that should produce WARNING status."""

    def test_fp32_to_fp16_precision_loss(self, tmp_path: Path) -> None:
        """FP32 training dtype but FP16 stored = WARNING (precision loss)."""
        d = tmp_path / "fp32-to-fp16"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "llama", "torch_dtype": "float32"})
        )
        _create_safetensors(d / "model.safetensors", dtype="F16")
        result = check_dtype_compatibility(str(d))
        assert result.status == CheckStatus.WARNING
        assert result.remediation is not None


class TestDtypeCheckFail:
    """Cases that should produce FAIL status."""

    def test_missing_config_json(self, tmp_path: Path) -> None:
        """Missing config.json should produce FAIL."""
        d = tmp_path / "no-config"
        d.mkdir()
        _create_safetensors(d / "model.safetensors", dtype="BF16")
        result = check_dtype_compatibility(str(d))
        assert result.status == CheckStatus.FAIL
        assert "config.json" in result.detail.lower()


class TestDtypeCheckMetadata:
    """Verify metadata content in results."""

    def test_result_contains_metadata(self, good_model: Path) -> None:
        """Result metadata should include training and stored dtypes."""
        result = check_dtype_compatibility(str(good_model))
        assert "training_dtype" in result.metadata
        assert "stored_dtype" in result.metadata

    def test_critical_result_metadata(self, bf16_to_fp16_model: Path) -> None:
        """Critical result metadata should include both dtypes for debugging."""
        result = check_dtype_compatibility(str(bf16_to_fp16_model))
        assert "training_dtype" in result.metadata
        assert "stored_dtype" in result.metadata
        assert result.metadata["training_dtype"] == "bfloat16"
        assert result.metadata["stored_dtype"] == "float16"
