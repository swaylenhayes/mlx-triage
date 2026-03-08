"""Tests for Check 0.5: Model architecture detection (VLM vs text-only)."""

import json
from pathlib import Path

import pytest

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.architecture_check import check_architecture


class TestArchitectureCheckPass:
    """Text-only models should PASS."""

    def test_text_only_causal_lm(self, tmp_path: Path) -> None:
        d = tmp_path / "text-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                }
            )
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.PASS
        assert result.check_id == "0.5"

    def test_text_only_no_architectures_key(self, tmp_path: Path) -> None:
        """Models without an architectures key are assumed text-only."""
        d = tmp_path / "no-arch"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "llama"})
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.PASS

    def test_text_only_moe(self, tmp_path: Path) -> None:
        """MoE text-only models should PASS."""
        d = tmp_path / "moe-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "model_type": "qwen3_moe",
                }
            )
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.PASS


class TestArchitectureCheckFail:
    """VLM models should FAIL (text-only loader cannot load them)."""

    def test_vlm_with_vision_config(self, tmp_path: Path) -> None:
        """Model with vision_config should FAIL."""
        d = tmp_path / "vlm-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "vision_config": {
                        "hidden_size": 1024,
                        "num_heads": 16,
                        "depth": 24,
                    },
                    "image_token_id": 248056,
                }
            )
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.FAIL
        assert "vision" in result.detail.lower() or "VLM" in result.detail
        assert result.remediation is not None

    def test_vlm_with_vision_config_only(self, tmp_path: Path) -> None:
        """vision_config alone is sufficient to detect VLM."""
        d = tmp_path / "vlm-minimal"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "some_vlm",
                    "vision_config": {"hidden_size": 768},
                }
            )
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.FAIL

    def test_vlm_with_vision_tower_config(self, tmp_path: Path) -> None:
        """Some models use vision_tower_config instead of vision_config."""
        d = tmp_path / "vlm-tower"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llava",
                    "vision_tower_config": {"hidden_size": 1024},
                }
            )
        )
        result = check_architecture(str(d))
        assert result.status == CheckStatus.FAIL


class TestArchitectureCheckMetadata:
    """Metadata should include architecture details."""

    def test_text_only_metadata(self, tmp_path: Path) -> None:
        d = tmp_path / "text-meta"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                }
            )
        )
        result = check_architecture(str(d))
        assert result.metadata["is_vlm"] is False
        assert result.metadata["architecture"] == "LlamaForCausalLM"

    def test_vlm_metadata(self, tmp_path: Path) -> None:
        d = tmp_path / "vlm-meta"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "vision_config": {"hidden_size": 1024},
                }
            )
        )
        result = check_architecture(str(d))
        assert result.metadata["is_vlm"] is True
        assert result.metadata["architecture"] == "Qwen3_5ForConditionalGeneration"
        assert "vision_config" in result.metadata["vlm_indicators"]


class TestArchitectureCheckGuards:
    """Guard clauses for missing/broken config."""

    def test_missing_config_json(self, tmp_path: Path) -> None:
        d = tmp_path / "no-config"
        d.mkdir()
        result = check_architecture(str(d))
        assert result.status == CheckStatus.FAIL
        assert "config.json" in result.detail

    def test_invalid_config_json(self, tmp_path: Path) -> None:
        d = tmp_path / "bad-config"
        d.mkdir()
        (d / "config.json").write_text("not valid json {{{")
        result = check_architecture(str(d))
        assert result.status == CheckStatus.FAIL
