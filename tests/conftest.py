"""Shared test fixtures for mlx-triage tests."""

import json
import struct
from pathlib import Path

import pytest


def create_safetensors(path: Path, dtype: str = "BF16", num_tensors: int = 1) -> None:
    """Create a minimal safetensors file with a valid header.

    Args:
        path: File path to write.
        dtype: Safetensors dtype string (F16, BF16, F32).
        num_tensors: Number of tensor entries in header.
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


@pytest.fixture
def good_model(tmp_path: Path) -> Path:
    """Model directory that passes all Tier 0 checks."""
    d = tmp_path / "good-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"})
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
            }
        )
    )
    create_safetensors(d / "model.safetensors", dtype="BF16")
    return d


@pytest.fixture
def bf16_to_fp16_model(tmp_path: Path) -> Path:
    """Model with BF16 training dtype but FP16 stored weights (Gemma 3 failure mode)."""
    d = tmp_path / "bad-dtype"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "gemma", "torch_dtype": "bfloat16"})
    )
    create_safetensors(d / "model.safetensors", dtype="F16")
    return d


@pytest.fixture
def quantized_model(tmp_path: Path) -> Path:
    """A 4-bit quantized model (should pass dtype check)."""
    d = tmp_path / "quantized"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "torch_dtype": "bfloat16",
                "quantization_config": {"bits": 4, "quant_method": "mlx"},
            }
        )
    )
    create_safetensors(d / "model.safetensors", dtype="I8")
    return d
