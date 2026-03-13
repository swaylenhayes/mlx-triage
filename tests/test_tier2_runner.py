"""Tests for Tier 2 runner."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier2 import run_tier2


def _make_result(check_id: str, name: str) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id,
        name=name,
        status=CheckStatus.PASS,
        detail="Mocked pass.",
    )


def _make_backend(available: bool = True) -> MagicMock:
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.load.return_value = (MagicMock(), MagicMock())
    return backend


@contextmanager
def _patch_checks():
    with patch(
        "mlx_triage.tier2.check_batch_invariance",
        return_value=_make_result("2.1", "Batch Invariance"),
    ) as batch, patch(
        "mlx_triage.tier2.check_memory_pressure",
        return_value=_make_result("2.2", "Memory Pressure Sweep"),
    ) as memory, patch(
        "mlx_triage.tier2.check_context_length",
        return_value=_make_result("2.3", "Context Length Stress"),
    ) as context:
        yield batch, memory, context


def test_run_tier2_returns_report():
    with _patch_checks():
        report = run_tier2("/fake/model", backend=_make_backend())
    assert isinstance(report, TierReport)
    assert report.tier == 2
    assert len(report.checks) == 3


def test_run_tier2_backend_not_available():
    report = run_tier2("/fake/model", backend=_make_backend(available=False))
    assert report.tier == 2
    assert all(check.status == CheckStatus.SKIP for check in report.checks)


def test_run_tier2_loads_model_once():
    backend = _make_backend()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    backend.load.return_value = (mock_model, mock_tokenizer)

    with _patch_checks() as (mock_batch, mock_memory, mock_context):
        run_tier2("/fake/model", backend=backend)

    backend.load.assert_called_once_with("/fake/model")

    for mock_check in (mock_batch, mock_memory, mock_context):
        kwargs = mock_check.call_args[1]
        assert kwargs["model"] is mock_model
        assert kwargs["tokenizer"] is mock_tokenizer
        assert kwargs["backend"] is backend


def test_run_tier2_auto_selects_backend(good_model):
    mock_backend = _make_backend()
    with patch("mlx_triage.tier2.get_backend", return_value=mock_backend):
        with _patch_checks():
            report = run_tier2(str(good_model))
    assert isinstance(report, TierReport)
