# tests/test_tier1_runner.py
"""Tests for Tier 1 runner."""
from unittest.mock import MagicMock, patch

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier1 import run_tier1


def _make_pass_result(check_id: str, name: str) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id, name=name, status=CheckStatus.PASS, detail="Mocked pass."
    )


def _make_skip_result(check_id: str, name: str) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id, name=name, status=CheckStatus.SKIP, detail="Mocked skip."
    )


def test_run_tier1_returns_report():
    """run_tier1 returns a TierReport with tier=1."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.load_model", return_value=(MagicMock(), MagicMock())):
            with patch(
                "mlx_triage.tier1.check_determinism",
                return_value=_make_pass_result("1.1", "Determinism"),
            ):
                with patch(
                    "mlx_triage.tier1.check_reference_divergence",
                    return_value=_make_pass_result("1.2", "Reference Divergence"),
                ):
                    with patch(
                        "mlx_triage.tier1.check_quantization_quality",
                        return_value=_make_pass_result("1.3", "Quantization Quality"),
                    ):
                        report = run_tier1("/fake/model")

    assert isinstance(report, TierReport)
    assert report.tier == 1
    assert len(report.checks) == 3


def test_run_tier1_all_pass():
    """All checks pass -> should_continue is True."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.load_model", return_value=(MagicMock(), MagicMock())):
            with patch(
                "mlx_triage.tier1.check_determinism",
                return_value=_make_pass_result("1.1", "Determinism"),
            ):
                with patch(
                    "mlx_triage.tier1.check_reference_divergence",
                    return_value=_make_pass_result("1.2", "Reference Divergence"),
                ):
                    with patch(
                        "mlx_triage.tier1.check_quantization_quality",
                        return_value=_make_pass_result("1.3", "Quantization Quality"),
                    ):
                        report = run_tier1("/fake/model")

    assert report.should_continue is True


def test_run_tier1_mlx_not_available():
    """When MLX is not available, all checks should SKIP."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=False):
        report = run_tier1("/fake/model")

    assert report.tier == 1
    assert all(c.status == CheckStatus.SKIP for c in report.checks)


def test_run_tier1_check_ids():
    """Verify correct check IDs are present."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.load_model", return_value=(MagicMock(), MagicMock())):
            with patch(
                "mlx_triage.tier1.check_determinism",
                return_value=_make_pass_result("1.1", "Determinism"),
            ):
                with patch(
                    "mlx_triage.tier1.check_reference_divergence",
                    return_value=_make_pass_result("1.2", "Reference Divergence"),
                ):
                    with patch(
                        "mlx_triage.tier1.check_quantization_quality",
                        return_value=_make_pass_result("1.3", "Quantization Quality"),
                    ):
                        report = run_tier1("/fake/model")

    check_ids = {c.check_id for c in report.checks}
    assert check_ids == {"1.1", "1.2", "1.3"}


def test_run_tier1_model_path_in_report():
    """Report should contain the model path."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.load_model", return_value=(MagicMock(), MagicMock())):
            with patch(
                "mlx_triage.tier1.check_determinism",
                return_value=_make_pass_result("1.1", "Determinism"),
            ):
                with patch(
                    "mlx_triage.tier1.check_reference_divergence",
                    return_value=_make_pass_result("1.2", "Reference Divergence"),
                ):
                    with patch(
                        "mlx_triage.tier1.check_quantization_quality",
                        return_value=_make_pass_result("1.3", "Quantization Quality"),
                    ):
                        report = run_tier1("/fake/model")

    assert report.model == "/fake/model"


def test_run_tier1_loads_model_once():
    """Model should be loaded exactly once, not per-check."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("mlx_triage.tier1.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.load_model", return_value=(mock_model, mock_tokenizer)) as mock_load:
            with patch(
                "mlx_triage.tier1.check_determinism",
                return_value=_make_pass_result("1.1", "Determinism"),
            ) as mock_det:
                with patch(
                    "mlx_triage.tier1.check_reference_divergence",
                    return_value=_make_pass_result("1.2", "Reference Divergence"),
                ) as mock_ref:
                    with patch(
                        "mlx_triage.tier1.check_quantization_quality",
                        return_value=_make_pass_result("1.3", "Quantization Quality"),
                    ) as mock_quant:
                        run_tier1("/fake/model")

    # load_model called exactly once
    mock_load.assert_called_once_with("/fake/model")

    # Each check received model and tokenizer kwargs
    for mock_check in (mock_det, mock_ref, mock_quant):
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs["model"] is mock_model
        assert call_kwargs["tokenizer"] is mock_tokenizer
