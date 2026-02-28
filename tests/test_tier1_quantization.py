# tests/test_tier1_quantization.py
"""Tests for quantization quality gate. All inference is mocked."""

from unittest.mock import patch, MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.quantization_quality import check_quantization_quality


def test_low_perplexity_passes():
    """Low perplexity -> PASS (model quality is good)."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.quantization_quality.load_model", return_value=(MagicMock(), MagicMock())):
            with patch("mlx_triage.tier1.quantization_quality._compute_perplexity", return_value=8.5):
                result = check_quantization_quality("/fake/model")

    assert result.check_id == "1.3"
    assert result.status == CheckStatus.PASS


def test_moderate_perplexity_info():
    """Moderate perplexity -> INFO (acceptable for aggressive quantization)."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.quantization_quality.load_model", return_value=(MagicMock(), MagicMock())):
            with patch("mlx_triage.tier1.quantization_quality._compute_perplexity", return_value=18.0):
                result = check_quantization_quality("/fake/model")

    assert result.status == CheckStatus.INFO


def test_high_perplexity_warning():
    """High perplexity -> WARNING."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.quantization_quality.load_model", return_value=(MagicMock(), MagicMock())):
            with patch("mlx_triage.tier1.quantization_quality._compute_perplexity", return_value=35.0):
                result = check_quantization_quality("/fake/model")

    assert result.status == CheckStatus.WARNING


def test_very_high_perplexity_fails():
    """Very high perplexity -> FAIL (model is broken or severely degraded)."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.quantization_quality.load_model", return_value=(MagicMock(), MagicMock())):
            with patch("mlx_triage.tier1.quantization_quality._compute_perplexity", return_value=75.0):
                result = check_quantization_quality("/fake/model")

    assert result.status == CheckStatus.FAIL


def test_mlx_not_available_skips():
    """When MLX is not installed -> SKIP."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=False):
        result = check_quantization_quality("/fake/model")
    assert result.status == CheckStatus.SKIP


def test_perplexity_error_returns_fail():
    """If perplexity computation raises -> FAIL with error detail."""
    with patch("mlx_triage.tier1.quantization_quality.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.quantization_quality.load_model", return_value=(MagicMock(), MagicMock())):
            with patch("mlx_triage.tier1.quantization_quality._compute_perplexity", side_effect=RuntimeError("OOM")):
                result = check_quantization_quality("/fake/model")

    assert result.status == CheckStatus.FAIL
    assert "OOM" in result.detail or "error" in result.detail.lower()
