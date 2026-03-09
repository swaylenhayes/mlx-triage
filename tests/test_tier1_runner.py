# tests/test_tier1_runner.py
"""Tests for Tier 1 runner."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier1 import run_tier1


def _make_pass_result(check_id: str, name: str) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id, name=name, status=CheckStatus.PASS, detail="Mocked pass."
    )


def _make_backend(available=True):
    """Create a mock backend for tier1 runner tests."""
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.load.return_value = (MagicMock(), MagicMock())
    return backend


@contextmanager
def _patch_checks():
    """Patch all three check functions to return PASS results."""
    with patch(
        "mlx_triage.tier1.check_determinism",
        return_value=_make_pass_result("1.1", "Determinism"),
    ) as det, patch(
        "mlx_triage.tier1.check_reference_divergence",
        return_value=_make_pass_result("1.2", "Reference Divergence"),
    ) as ref, patch(
        "mlx_triage.tier1.check_quantization_quality",
        return_value=_make_pass_result("1.3", "Quantization Quality"),
    ) as quant:
        yield det, ref, quant


def test_run_tier1_returns_report():
    """run_tier1 returns a TierReport with tier=1."""
    with _patch_checks():
        report = run_tier1("/fake/model", backend=_make_backend())
    assert isinstance(report, TierReport)
    assert report.tier == 1
    assert len(report.checks) == 3


def test_run_tier1_all_pass():
    """All checks pass -> should_continue is True."""
    with _patch_checks():
        report = run_tier1("/fake/model", backend=_make_backend())
    assert report.should_continue is True


def test_run_tier1_backend_not_available():
    """When backend is not available, all checks should SKIP."""
    report = run_tier1("/fake/model", backend=_make_backend(available=False))
    assert report.tier == 1
    assert all(c.status == CheckStatus.SKIP for c in report.checks)


def test_run_tier1_check_ids():
    """Verify correct check IDs are present."""
    with _patch_checks():
        report = run_tier1("/fake/model", backend=_make_backend())
    check_ids = {c.check_id for c in report.checks}
    assert check_ids == {"1.1", "1.2", "1.3"}


def test_run_tier1_model_path_in_report():
    """Report should contain the model path."""
    with _patch_checks():
        report = run_tier1("/fake/model", backend=_make_backend())
    assert report.model == "/fake/model"


def test_run_tier1_loads_model_once():
    """Model should be loaded exactly once via the backend, not per-check."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    backend = _make_backend()
    backend.load.return_value = (mock_model, mock_tokenizer)

    with _patch_checks() as (mock_det, mock_ref, mock_quant):
        run_tier1("/fake/model", backend=backend)

    backend.load.assert_called_once_with("/fake/model")

    for mock_check in (mock_det, mock_ref, mock_quant):
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs["model"] is mock_model
        assert call_kwargs["tokenizer"] is mock_tokenizer
        assert call_kwargs["backend"] is backend


def test_run_tier1_auto_selects_backend(good_model):
    """When no backend provided, auto-selects via get_backend()."""
    mock_backend = _make_backend()
    with patch("mlx_triage.tier1.get_backend", return_value=mock_backend):
        with _patch_checks():
            report = run_tier1(str(good_model))
    assert isinstance(report, TierReport)
