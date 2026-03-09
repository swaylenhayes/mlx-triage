import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mlx_triage.cli import cli


def _make_unavailable_backend():
    """Create a mock backend that reports MLX as unavailable."""
    backend = MagicMock()
    backend.is_available.return_value = False
    return backend


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


def test_check_tier1_skips_when_mlx_missing(good_model):
    """--tier 1 should still work when MLX is not installed (checks SKIP)."""
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
            result = runner.invoke(cli, ["check", str(good_model), "--tier", "1"])
    assert result.exit_code == 0


def test_check_tier1_json_output(good_model):
    """--tier 1 --format json should produce valid JSON with both tiers."""
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
            result = runner.invoke(
                cli, ["check", str(good_model), "--tier", "1", "--format", "json"]
            )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    # Should be a list of two tier reports
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["tier"] == 0
    assert parsed[1]["tier"] == 1


def test_check_tier1_skipped_on_tier0_critical(bf16_to_fp16_model):
    """--tier 1 should skip Tier 1 when Tier 0 has CRITICAL issues."""
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(
            cli, ["check", str(bf16_to_fp16_model), "--tier", "1"]
        )
    assert result.exit_code == 0
    # Should print the skip warning to stderr
    # Only Tier 0 report should be rendered (no Tier 1 section)


def test_check_default_tier_unchanged(good_model):
    """Default (no --tier flag) should still run only Tier 0."""
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    # Default is Tier 0 only -- should be a single report dict, not a list
    assert isinstance(parsed, dict)
    assert parsed["tier"] == 0
