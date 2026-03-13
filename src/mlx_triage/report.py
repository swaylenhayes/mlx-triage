"""Report generation for mlx-triage diagnostics."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlx_triage.models import CheckStatus, TierReport
from mlx_triage.traits import collect_traits

# Status → color mapping for Rich output
STATUS_COLORS = {
    CheckStatus.PASS: "green",
    CheckStatus.INFO: "blue",
    CheckStatus.SKIP: "dim",
    CheckStatus.WARNING: "yellow",
    CheckStatus.FAIL: "red",
    CheckStatus.CRITICAL: "bold red",
}


def _tier2_diagnostic_assessment(report: TierReport) -> dict | None:
    """Interpret Tier 2 outcomes as infra-like, model-like, or mixed."""
    if report.tier != 2:
        return None

    checks_by_id = {check.check_id: check for check in report.checks}
    required_ids = {"2.1", "2.2", "2.3"}
    if not required_ids.issubset(checks_by_id):
        return None

    batch_check = checks_by_id["2.1"]
    memory_check = checks_by_id["2.2"]
    context_check = checks_by_id["2.3"]
    checks = [batch_check, memory_check, context_check]

    if any(check.status == CheckStatus.SKIP for check in checks):
        return {
            "classification": "preflight-only",
            "detail": "Tier 2 is incomplete because one or more isolation checks were skipped.",
            "signals": [],
        }

    signals: list[str] = []
    if batch_check.status in (CheckStatus.WARNING, CheckStatus.FAIL, CheckStatus.CRITICAL):
        signals.append("batch_sensitive")
    if memory_check.status in (
        CheckStatus.WARNING,
        CheckStatus.FAIL,
        CheckStatus.CRITICAL,
    ):
        signals.append("memory_pressure_sensitive")
    if context_check.status in (
        CheckStatus.WARNING,
        CheckStatus.FAIL,
        CheckStatus.CRITICAL,
    ):
        signals.append("long_context_degradation")

    if not signals:
        return {
            "classification": "no-runtime-instability-detected",
            "detail": "Tier 2 did not surface runtime-instability signals across batching, memory pressure, or long-context stress.",
            "signals": [],
        }

    if any(check.status == CheckStatus.CRITICAL for check in checks):
        return {
            "classification": "infrastructure-likely",
            "detail": "Hard failures under Tier 2 stress suggest a runtime or infrastructure problem rather than a pure model-quality issue.",
            "signals": signals,
        }

    infra_signals = [signal for signal in signals if signal != "long_context_degradation"]
    context_only = signals == ["long_context_degradation"]

    if context_only:
        return {
            "classification": "model-likely",
            "detail": "Generation stayed operational, but quality degraded only under longer contexts, which points more strongly to model/context limitations than runtime failure.",
            "signals": signals,
        }

    if infra_signals and "long_context_degradation" in signals:
        return {
            "classification": "mixed-signals",
            "detail": "Tier 2 shows both runtime-sensitive behavior and long-context degradation, so the failure mode is mixed rather than cleanly attributable to one layer.",
            "signals": signals,
        }

    return {
        "classification": "infrastructure-likely",
        "detail": "Tier 2 shows batch- or memory-sensitive behavior, which is more consistent with inference/runtime instability than intrinsic model limits.",
        "signals": signals,
    }


def render_json(report: TierReport) -> str:
    """Render a TierReport as a JSON string."""
    checks_dict = {}
    for check in report.checks:
        entry: dict = {
            "status": check.status.value,
            "detail": check.detail,
        }
        if check.remediation:
            entry["remediation"] = check.remediation
        if check.metadata:
            entry["metadata"] = check.metadata
        checks_dict[check.check_id] = entry

    output = {
        "tier": report.tier,
        "model": report.model,
        "timestamp": report.timestamp,
        "checks": checks_dict,
        "verdict": report.verdict,
        "should_continue": report.should_continue,
        "claim_level": report.claim_level,
        "checks_executed": report.checks_executed,
        "checks_skipped": report.checks_skipped,
        "skipped_check_ids": report.skipped_check_ids,
    }
    if report.tier == 0:
        output["traits"] = collect_traits(report.checks)
    assessment = _tier2_diagnostic_assessment(report)
    if assessment is not None:
        output["diagnostic_assessment"] = assessment
    return json.dumps(output, indent=2)


def render_terminal(report: TierReport) -> str:
    """Render a TierReport as a Rich terminal string."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]Tier {report.tier} Diagnostic Report[/bold]\n"
            f"Model: {report.model}\n"
            f"Time: {report.timestamp}",
            title="mlx-triage",
            border_style="blue",
        )
    )

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Detail", min_width=40)

    for check in report.checks:
        color = STATUS_COLORS[check.status]
        status_text = Text(check.status.value, style=color)
        detail = check.detail
        if check.remediation:
            detail += f"\n[dim]Fix: {check.remediation}[/dim]"
        table.add_row(f"{check.check_id} {check.name}", status_text, detail)

    console.print(table)

    assessment = _tier2_diagnostic_assessment(report)
    if assessment is not None:
        console.print()
        console.print(
            Panel(
                f"[bold]{assessment['classification']}[/bold]\n{assessment['detail']}",
                title="Tier 2 Assessment",
                border_style="magenta",
            )
        )

    # Verdict
    worst = report.worst_status
    verdict_color = STATUS_COLORS[worst]
    console.print()
    console.print(
        Panel(
            f"[{verdict_color}]{report.verdict}[/{verdict_color}]"
            + (
                f"\n[dim]Proceed to Tier {report.tier + 1}: "
                f"{'Yes' if report.should_continue else 'No — fix issues first'}[/dim]"
                if report.tier < 3
                else ""
            ),
            title="Verdict",
            border_style=verdict_color.replace("bold ", ""),
        )
    )

    return buf.getvalue()


def write_reports(reports: list[TierReport], path: str, fmt: str = "json") -> None:
    """Write one or more reports to a file.

    Args:
        reports: List of TierReport objects (usually [tier0] or [tier0, tier1]).
        path: Output file path.
        fmt: Format — ``"json"`` or ``"terminal"``.
    """
    if fmt == "json":
        if len(reports) == 1:
            content = render_json(reports[0])
        else:
            all_reports = [json.loads(render_json(r)) for r in reports]
            content = json.dumps(all_reports, indent=2)
    else:
        content = "\n".join(render_terminal(r) for r in reports)

    with open(path, "w") as f:
        f.write(content)
