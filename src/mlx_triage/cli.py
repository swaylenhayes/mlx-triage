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
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero if any check is skipped.",
)
def check(model_path: str, tier: int, output: str | None, fmt: str, strict: bool) -> None:
    """Run diagnostic checks against an MLX model."""
    import json as json_mod

    from mlx_triage.models import CheckStatus
    from mlx_triage.report import render_json, render_terminal, write_reports
    from mlx_triage.tier0 import run_tier0

    reports = []

    # Run Tier 0 (always runs first)
    tier0_report = run_tier0(model_path)
    reports.append(tier0_report)

    higher_tiers_requested = tier >= 1
    tier0_blocks_higher_tiers = tier0_report.worst_status in (
        CheckStatus.CRITICAL,
        CheckStatus.FAIL,
    )

    if higher_tiers_requested and tier0_blocks_higher_tiers:
        click.echo(
            "Tier 0 found critical issues. Fix those before running higher tiers.",
            err=True,
        )
    else:
        # Run Tier 1 if requested
        if tier >= 1:
            from mlx_triage.tier1 import run_tier1

            tier1_report = run_tier1(model_path)
            reports.append(tier1_report)

        # Run Tier 2 if requested
        if tier >= 2:
            from mlx_triage.tier2 import run_tier2

            tier2_report = run_tier2(model_path)
            reports.append(tier2_report)

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

    if strict and any(report.checks_skipped > 0 for report in reports):
        click.echo(
            "Strict mode failed: one or more checks were skipped.",
            err=True,
        )
        raise click.exceptions.Exit(1)
