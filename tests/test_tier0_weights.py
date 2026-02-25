"""Tests for Check 0.3: Weight File Integrity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.weight_integrity import check_weight_integrity


class TestWeightIntegrityPass:
    """Tests where weight integrity check should pass or be informational."""

    def test_good_model_returns_pass_or_info(self, good_model: Path) -> None:
        """good_model fixture uses BF16 which numpy can't load — returns INFO."""
        result = check_weight_integrity(str(good_model))
        assert result.status in (CheckStatus.PASS, CheckStatus.INFO)
        assert result.check_id == "0.3"
        assert result.name == "Weight File Integrity"

    def test_no_safetensors_returns_info(self, tmp_path: Path) -> None:
        """A directory with no safetensors files returns INFO."""
        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.INFO
        assert result.metadata["shard_count"] == 0
        assert "No safetensors files found" in result.detail

    def test_clean_weights_return_pass(self, tmp_path: Path) -> None:
        """Valid non-zero float32 weights should return PASS."""
        model_dir = tmp_path / "clean-model"
        model_dir.mkdir()
        tensors = {
            "model.layers.0.weight": np.random.randn(4, 4).astype(np.float32),
            "model.layers.1.weight": np.random.randn(4, 4).astype(np.float32),
        }
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.PASS
        assert result.metadata["tensor_count"] == 2
        assert result.metadata["shard_count"] == 1


class TestWeightIntegrityNaN:
    """Tests for NaN detection."""

    def test_nan_weights_return_critical(self, tmp_path: Path) -> None:
        """Tensors containing NaN values should return CRITICAL."""
        model_dir = tmp_path / "nan-model"
        model_dir.mkdir()
        bad_tensor = np.array([[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32)
        tensors = {"model.layers.0.weight": bad_tensor}
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.CRITICAL
        assert "NaN" in result.detail
        assert result.remediation is not None
        assert "nan_tensors" in result.metadata

    def test_all_nan_tensor_returns_critical(self, tmp_path: Path) -> None:
        """A fully NaN tensor should return CRITICAL."""
        model_dir = tmp_path / "all-nan-model"
        model_dir.mkdir()
        bad_tensor = np.full((4, 4), float("nan"), dtype=np.float32)
        tensors = {"model.layers.0.weight": bad_tensor}
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.CRITICAL
        assert "NaN" in result.detail


class TestWeightIntegrityInf:
    """Tests for Inf detection."""

    def test_inf_weights_return_critical(self, tmp_path: Path) -> None:
        """Tensors containing Inf values should return CRITICAL."""
        model_dir = tmp_path / "inf-model"
        model_dir.mkdir()
        bad_tensor = np.array([[1.0, float("inf")], [3.0, 4.0]], dtype=np.float32)
        tensors = {"model.layers.0.weight": bad_tensor}
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.CRITICAL
        assert "Inf" in result.detail
        assert result.remediation is not None
        assert "inf_tensors" in result.metadata

    def test_neg_inf_weights_return_critical(self, tmp_path: Path) -> None:
        """Negative Inf should also be detected."""
        model_dir = tmp_path / "neg-inf-model"
        model_dir.mkdir()
        bad_tensor = np.array([[float("-inf"), 2.0], [3.0, 4.0]], dtype=np.float32)
        tensors = {"model.layers.0.weight": bad_tensor}
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.CRITICAL
        assert "Inf" in result.detail

    def test_nan_and_inf_together_return_critical(self, tmp_path: Path) -> None:
        """Both NaN and Inf in the same file should report both."""
        model_dir = tmp_path / "nan-inf-model"
        model_dir.mkdir()
        tensors = {
            "model.layers.0.weight": np.array(
                [[float("nan"), 2.0], [3.0, 4.0]], dtype=np.float32
            ),
            "model.layers.1.weight": np.array(
                [[float("inf"), 2.0], [3.0, 4.0]], dtype=np.float32
            ),
        }
        save_file(tensors, str(model_dir / "model.safetensors"))
        result = check_weight_integrity(str(model_dir))
        assert result.status == CheckStatus.CRITICAL
        assert "NaN" in result.detail
        assert "Inf" in result.detail
