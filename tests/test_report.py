import json

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.report import render_json, render_terminal


def _make_report() -> TierReport:
    return TierReport(
        tier=0,
        model="/path/to/model",
        timestamp="2026-02-25T14:30:00Z",
        checks=[
            DiagnosticResult(
                check_id="0.1",
                name="Dtype Compatibility",
                status=CheckStatus.PASS,
                detail="Training dtype bfloat16, stored as bfloat16. Compatible.",
            ),
            DiagnosticResult(
                check_id="0.2",
                name="Tokenizer Config",
                status=CheckStatus.WARNING,
                detail="No chat_template defined.",
                remediation="Add chat_template to tokenizer_config.json.",
            ),
        ],
    )


def test_render_json_valid():
    report = _make_report()
    output = render_json(report)
    parsed = json.loads(output)
    assert parsed["tier"] == 0
    assert parsed["model"] == "/path/to/model"
    assert len(parsed["checks"]) == 2
    assert parsed["checks"]["0.1"]["status"] == "PASS"
    assert parsed["verdict"] is not None
    assert "should_continue" in parsed


def test_render_json_roundtrip():
    report = _make_report()
    output = render_json(report)
    parsed = json.loads(output)
    assert parsed["checks"]["0.2"]["remediation"] == "Add chat_template to tokenizer_config.json."


def test_render_terminal_contains_key_info():
    report = _make_report()
    output = render_terminal(report)
    assert "model" in output.lower() or "/path/to/model" in output
    assert "Dtype" in output or "dtype" in output
    assert "PASS" in output
    assert "WARNING" in output
