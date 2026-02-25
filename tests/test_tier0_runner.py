from unittest.mock import patch

from mlx_triage.models import CheckStatus, TierReport
from mlx_triage.tier0 import run_tier0


def test_run_tier0_returns_report(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    assert isinstance(report, TierReport)
    assert report.tier == 0
    assert len(report.checks) == 4


def test_run_tier0_all_pass(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    statuses = [c.status for c in report.checks]
    assert CheckStatus.FAIL not in statuses
    assert CheckStatus.CRITICAL not in statuses
    assert report.should_continue is True


def test_run_tier0_catches_bad_dtype(bf16_to_fp16_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(bf16_to_fp16_model))
    dtype_check = next(c for c in report.checks if c.check_id == "0.1")
    assert dtype_check.status == CheckStatus.CRITICAL
    assert report.should_continue is False
