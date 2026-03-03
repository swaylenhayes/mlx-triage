"""CLI entry point for mlx-triage."""

from __future__ import annotations

import click

from mlx_triage import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """MLX Inference Quality Diagnostic Toolkit.

    Run tiered diagnostic checks against MLX-served models to determine
    whether quality issues stem from the model or the inference infrastructure.
    """


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--tier",
    type=click.IntRange(0, 3),
    default=0,
    help="Maximum diagnostic tier to run (0-3).",
)
@click.option("--output", type=click.Path(), default=None, help="Save report to file.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "terminal"]),
    default="terminal",
    help="Output format.",
)
def check(model_path: str, tier: int, output: str | None, fmt: str) -> None:
    """Run diagnostic checks against an MLX model."""
    import json as json_mod

    from mlx_triage.models import CheckStatus
    from mlx_triage.report import render_json, render_terminal, write_reports
    from mlx_triage.tier0 import run_tier0

    reports = []

    # Run Tier 0 (always runs first)
    tier0_report = run_tier0(model_path)
    reports.append(tier0_report)

    # Run Tier 1 if requested
    if tier >= 1:
        if tier0_report.worst_status in (
            CheckStatus.CRITICAL,
            CheckStatus.FAIL,
        ):
            click.echo(
                "Tier 0 found critical issues. Fix those before running Tier 1.",
                err=True,
            )
        else:
            from mlx_triage.tier1 import run_tier1

            tier1_report = run_tier1(model_path)
            reports.append(tier1_report)

    # Output
    if output:
        write_reports(reports, output, fmt=fmt)
        click.echo(f"Report saved to {output}")
    elif fmt == "json":
        if len(reports) == 1:
            click.echo(render_json(reports[0]))
        else:
            all_reports = [json_mod.loads(render_json(r)) for r in reports]
            click.echo(json_mod.dumps(all_reports, indent=2))
    else:
        for report in reports:
            click.echo(render_terminal(report))


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), required=True, help="Output file path.")
def report(model_path: str, output: str) -> None:
    """Generate a full diagnostic report."""
    click.echo(f"Generating report for {model_path}...")
    # TODO: Wire to full multi-tier report generator


@cli.command()
@click.argument("model_a", type=click.Path(exists=True))
@click.argument("model_b", type=click.Path(exists=True))
def compare(model_a: str, model_b: str) -> None:
    """Compare diagnostics between two models."""
    click.echo(f"Comparing {model_a} vs {model_b}...")
    # TODO: Wire to comparison logic
