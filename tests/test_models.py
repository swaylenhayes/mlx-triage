from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport


def test_check_status_values():
    assert CheckStatus.PASS.value == "PASS"
    assert CheckStatus.CRITICAL.value == "CRITICAL"
    assert CheckStatus.WARNING.value == "WARNING"
    assert CheckStatus.FAIL.value == "FAIL"
    assert CheckStatus.INFO.value == "INFO"
    assert CheckStatus.SKIP.value == "SKIP"


def test_diagnostic_result_creation():
    result = DiagnosticResult(
        check_id="0.1",
        name="Dtype Compatibility",
        status=CheckStatus.PASS,
        detail="Compatible.",
    )
    assert result.check_id == "0.1"
    assert result.status == CheckStatus.PASS
    assert result.remediation is None
    assert result.metadata == {}


def test_diagnostic_result_with_remediation():
    result = DiagnosticResult(
        check_id="0.2",
        name="Tokenizer Config",
        status=CheckStatus.WARNING,
        detail="Missing stop token.",
        remediation="Add missing stop token to generation_config.json.",
    )
    assert result.remediation is not None


def test_tier_report_creation():
    checks = [
        DiagnosticResult(
            check_id="0.1",
            name="Dtype",
            status=CheckStatus.PASS,
            detail="OK",
        ),
    ]
    report = TierReport(
        tier=0,
        model="/path/to/model",
        timestamp="2026-02-25T14:30:00Z",
        checks=checks,
    )
    assert report.tier == 0
    assert len(report.checks) == 1
    assert report.verdict != ""
    assert isinstance(report.should_continue, bool)


def test_tier_report_verdict_all_pass():
    checks = [
        DiagnosticResult(check_id="0.1", name="A", status=CheckStatus.PASS, detail="OK"),
        DiagnosticResult(check_id="0.2", name="B", status=CheckStatus.PASS, detail="OK"),
    ]
    report = TierReport(tier=0, model="test", timestamp="now", checks=checks)
    assert "PASS" in report.verdict
    assert report.should_continue is True


def test_tier_report_verdict_critical():
    checks = [
        DiagnosticResult(check_id="0.1", name="A", status=CheckStatus.CRITICAL, detail="Bad"),
        DiagnosticResult(check_id="0.2", name="B", status=CheckStatus.PASS, detail="OK"),
    ]
    report = TierReport(tier=0, model="test", timestamp="now", checks=checks)
    assert "CRITICAL" in report.verdict
    assert report.should_continue is False
