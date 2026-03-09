# tests/test_tier1_quantization.py
"""Tests for quantization quality gate. All inference is mocked via backend."""

from unittest.mock import MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.quantization_quality import check_quantization_quality


def _make_backend(available=True, perplexity=None, perplexity_error=None):
    """Create a mock backend for quantization tests."""
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.load.return_value = (MagicMock(), MagicMock())
    if perplexity_error:
        backend.compute_perplexity.side_effect = perplexity_error
    elif perplexity is not None:
        backend.compute_perplexity.return_value = perplexity
    return backend


def test_low_perplexity_passes():
    """Low perplexity -> PASS (model quality is good)."""
    result = check_quantization_quality(
        "/fake/model", backend=_make_backend(perplexity=8.5)
    )
    assert result.check_id == "1.3"
    assert result.status == CheckStatus.PASS


def test_moderate_perplexity_info():
    """Moderate perplexity -> INFO."""
    result = check_quantization_quality(
        "/fake/model", backend=_make_backend(perplexity=18.0)
    )
    assert result.status == CheckStatus.INFO


def test_high_perplexity_warning():
    """High perplexity -> WARNING."""
    result = check_quantization_quality(
        "/fake/model", backend=_make_backend(perplexity=35.0)
    )
    assert result.status == CheckStatus.WARNING


def test_very_high_perplexity_fails():
    """Very high perplexity -> FAIL."""
    result = check_quantization_quality(
        "/fake/model", backend=_make_backend(perplexity=75.0)
    )
    assert result.status == CheckStatus.FAIL


def test_backend_not_available_skips():
    """When backend is not available -> SKIP."""
    result = check_quantization_quality(
        "/fake/model", backend=_make_backend(available=False)
    )
    assert result.status == CheckStatus.SKIP


def test_perplexity_error_returns_fail():
    """If perplexity computation raises -> FAIL with error detail."""
    result = check_quantization_quality(
        "/fake/model",
        backend=_make_backend(perplexity_error=RuntimeError("OOM")),
    )
    assert result.status == CheckStatus.FAIL
    assert "OOM" in result.detail or "error" in result.detail.lower()
