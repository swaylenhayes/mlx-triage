import json
from unittest.mock import patch

from click.testing import CliRunner

from mlx_triage.cli import cli


def test_check_tier0_terminal(good_model):
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model)])
    assert result.exit_code == 0
    assert "Tier 0" in result.output or "PASS" in result.output


def test_check_tier0_json(good_model):
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["tier"] == 0
    assert "checks" in parsed


def test_check_tier0_output_file(good_model, tmp_path):
    runner = CliRunner()
    out_file = str(tmp_path / "report.json")
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(
            cli, ["check", str(good_model), "--output", out_file, "--format", "json"]
        )
    assert result.exit_code == 0
    with open(out_file) as f:
        parsed = json.load(f)
    assert parsed["tier"] == 0


def test_check_invalid_path():
    runner = CliRunner()
    result = runner.invoke(cli, ["check", "/nonexistent/path"])
    assert result.exit_code != 0
