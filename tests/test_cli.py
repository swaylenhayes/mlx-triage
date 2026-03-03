from click.testing import CliRunner

from mlx_triage.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MLX Inference Quality Diagnostic Toolkit" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_check_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "MODEL_PATH" in result.output


def test_removed_commands_not_exposed():
    """Stub commands (report, compare) should not be in the CLI for v0.1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert "report" not in result.output
    assert "compare" not in result.output
