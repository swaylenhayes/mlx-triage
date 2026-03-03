"""Report generation for mlx-triage diagnostics."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlx_triage.models import CheckStatus, TierReport

# Status → color mapping for Rich output
STATUS_COLORS = {
    CheckStatus.PASS: "green",
    CheckStatus.INFO: "blue",
    CheckStatus.SKIP: "dim",
    CheckStatus.WARNING: "yellow",
    CheckStatus.FAIL: "red",
    CheckStatus.CRITICAL: "bold red",
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
    }
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
