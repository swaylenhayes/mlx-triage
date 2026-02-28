# tests/test_tier1_determinism.py
"""Tests for determinism check. All inference is mocked."""

from unittest.mock import patch, MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.determinism import check_determinism
from mlx_triage.utils.mlx_utils import GenerationResult


def _mock_generate(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens, logprobs=[], generation_tps=50.0)


def test_deterministic_model_passes():
    """Model producing identical output every run -> PASS."""

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("Hello world", [100, 200, 300])

    with patch("mlx_triage.tier1.determinism.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
            with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
                result = check_determinism("/fake/model", n_runs=5, n_prompts=3)

    assert result.check_id == "1.1"
    assert result.status == CheckStatus.PASS


def test_major_divergence_fails():
    """Completely different output each run -> FAIL."""
    call_count = [0]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        call_count[0] += 1
        return _mock_generate(f"output_{call_count[0]}", [call_count[0], call_count[0] + 1])

    with patch("mlx_triage.tier1.determinism.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
            with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
                result = check_determinism("/fake/model", n_runs=5, n_prompts=3)

    assert result.status in (CheckStatus.FAIL, CheckStatus.CRITICAL)


def test_empty_output_critical():
    """Empty output on any run -> CRITICAL."""

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("", [])

    with patch("mlx_triage.tier1.determinism.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
            with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
                result = check_determinism("/fake/model", n_runs=3, n_prompts=2)

    assert result.status == CheckStatus.CRITICAL


def test_mlx_not_available_skips():
    """When MLX is not installed, should SKIP."""
    with patch("mlx_triage.tier1.determinism.check_mlx_available", return_value=False):
        result = check_determinism("/fake/model")
    assert result.status == CheckStatus.SKIP
