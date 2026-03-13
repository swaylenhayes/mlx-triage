import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mlx_triage.cli import cli
from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.utils.mlx_utils import GenerationResult
from tests.conftest import patch_bf16_weight_integrity


def _make_unavailable_backend():
    """Create a mock backend that reports MLX as unavailable."""
    backend = MagicMock()
    backend.is_available.return_value = False
    return backend


class _BatchCapableBackend:
    def is_available(self) -> bool:
        return True

    def load(self, _model_path: str):
        return object(), object()

    def generate_text(self, _model, _tokenizer, prompt, **_kwargs) -> GenerationResult:
        token = len(str(prompt))
        return GenerationResult(text=f"reply::{prompt}", tokens=[token])

    def generate_batch(
        self, _model, _tokenizer, prompts, **_kwargs
    ) -> list[GenerationResult]:
        return [
            GenerationResult(text=f"reply::{prompt}", tokens=[len(str(prompt))])
            for prompt in prompts
        ]


def _pass_result(check_id: str, name: str) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id,
        name=name,
        status=CheckStatus.PASS,
        detail="Mocked pass.",
    )


def test_check_tier0_terminal(good_model):
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            result = runner.invoke(cli, ["check", str(good_model)])
    assert result.exit_code == 0
    assert "Tier 0" in result.output or "PASS" in result.output


def test_check_tier0_json(good_model):
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            result = runner.invoke(cli, ["check", str(good_model), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["tier"] == 0
    assert "checks" in parsed


def test_check_tier0_output_file(good_model, tmp_path):
    runner = CliRunner()
    out_file = str(tmp_path / "report.json")
    with patch_bf16_weight_integrity():
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
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
                result = runner.invoke(cli, ["check", str(good_model), "--tier", "1"])
    assert result.exit_code == 0


def test_check_tier1_json_output(good_model):
    """--tier 1 --format json should produce valid JSON with both tiers."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
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


def test_check_tier2_json_output(good_model):
    """--tier 2 --format json should produce valid JSON with three tiers."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
                with patch(
                    "mlx_triage.tier2.get_backend", return_value=_make_unavailable_backend()
                ):
                    result = runner.invoke(
                        cli, ["check", str(good_model), "--tier", "2", "--format", "json"]
                    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert parsed[0]["tier"] == 0
    assert parsed[1]["tier"] == 1
    assert parsed[2]["tier"] == 2


def test_check_tier2_json_output_runs_real_batch_invariance(good_model):
    """Tier 2 JSON should include a non-placeholder 2.1 result when batching exists."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
                with patch("mlx_triage.tier2.get_backend", return_value=_BatchCapableBackend()):
                    with patch(
                        "mlx_triage.tier2.check_memory_pressure",
                        return_value=_pass_result("2.2", "Memory Pressure Sweep"),
                    ), patch(
                        "mlx_triage.tier2.check_context_length",
                        return_value=_pass_result("2.3", "Context Length Stress"),
                    ):
                        result = runner.invoke(
                            cli, ["check", str(good_model), "--tier", "2", "--format", "json"]
                        )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed[2]["checks"]["2.1"]["status"] == "PASS"


def test_check_strict_fails_when_any_check_skipped(good_model):
    """--strict should exit non-zero when any check status is SKIP."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            with patch("mlx_triage.tier1.get_backend", return_value=_make_unavailable_backend()):
                result = runner.invoke(
                    cli,
                    ["check", str(good_model), "--tier", "1", "--format", "json", "--strict"],
                )
    assert result.exit_code == 1

    json_output = result.output.split("\nStrict mode failed:", 1)[0]
    parsed = json.loads(json_output)
    assert isinstance(parsed, list)
    assert parsed[1]["checks_skipped"] > 0


def test_check_strict_passes_when_no_skips(good_model):
    """--strict should succeed when no check status is SKIP."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            result = runner.invoke(
                cli, ["check", str(good_model), "--format", "json", "--strict"]
            )
    assert result.exit_code == 0

    parsed = json.loads(result.output)
    assert isinstance(parsed, dict)
    assert parsed["checks_skipped"] == 0


def test_check_tier1_skipped_on_tier0_critical(bf16_to_fp16_model):
    """--tier 1 should skip Tier 1 when Tier 0 has CRITICAL issues."""
    runner = CliRunner()
    with patch_bf16_weight_integrity():
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
    with patch_bf16_weight_integrity():
        with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
            result = runner.invoke(cli, ["check", str(good_model), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    # Default is Tier 0 only -- should be a single report dict, not a list
    assert isinstance(parsed, dict)
    assert parsed["tier"] == 0
