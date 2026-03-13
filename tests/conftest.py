"""Shared test fixtures for mlx-triage tests."""

import json
import struct
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_triage.models import CheckStatus, DiagnosticResult


def _mock_bf16_weight_result(
    status: CheckStatus = CheckStatus.PASS,
    detail: str = "Mock BF16 inspection result.",
) -> DiagnosticResult:
    """Build a deterministic BF16 inspection result for hermetic tests."""
    return DiagnosticResult(
        check_id="0.3",
        name="Weight File Integrity",
        status=status,
        detail=detail,
        metadata={
            "shard_count": 1,
            "shards_sampled": 1,
            "tensor_count": 1,
            "dtype": "BF16",
            "inspection_backend": "mlx",
        },
    )


@contextmanager
def patch_bf16_weight_integrity(
    status: CheckStatus = CheckStatus.PASS,
    detail: str = "Mock BF16 inspection result.",
):
    """Patch BF16 MLX inspection to avoid importing real MLX in tests."""
    with patch(
        "mlx_triage.tier0.weight_integrity._check_bf16_with_mlx",
        return_value=_mock_bf16_weight_result(status=status, detail=detail),
    ):
        yield


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


@pytest.fixture
def thinking_model(tmp_path: Path) -> Path:
    """Model directory with thinking tokens in tokenizer vocabulary."""
    d = tmp_path / "thinking-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
            }
        )
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|endoftext|>",
                "bos_token": "<|startoftext|>",
                "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
                "added_tokens_decoder": {
                    "151648": {
                        "content": "<think>",
                        "special": True,
                    },
                    "151649": {
                        "content": "</think>",
                        "special": True,
                    },
                },
            }
        )
    )
    create_safetensors(d / "model.safetensors", dtype="BF16")
    return d


@pytest.fixture
def vlm_model(tmp_path: Path) -> Path:
    """Model directory with VLM architecture (vision_config present)."""
    d = tmp_path / "vlm-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 1024},
                "torch_dtype": "bfloat16",
            }
        )
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "</s>", "chat_template": "..."})
    )
    create_safetensors(d / "model.safetensors", dtype="BF16")
    return d
