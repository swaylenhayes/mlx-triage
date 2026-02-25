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
    type=click.Choice(["json", "terminal", "markdown"]),
    default="terminal",
    help="Output format.",
)
def check(model_path: str, tier: int, output: str | None, fmt: str) -> None:
    """Run diagnostic checks against an MLX model."""
    from mlx_triage.report import render_json, render_terminal, write_report
    from mlx_triage.tier0 import run_tier0

    # Run Tier 0 (always runs first)
    report = run_tier0(model_path)

    # Output
    if output:
        write_report(report, output, fmt=fmt)
        click.echo(f"Report saved to {output}")
    elif fmt == "json":
        click.echo(render_json(report))
    else:
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
