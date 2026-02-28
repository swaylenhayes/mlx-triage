# tests/test_tier1_divergence.py
"""Tests for reference divergence check. All inference is mocked."""

from unittest.mock import patch, MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.reference_divergence import check_reference_divergence
from mlx_triage.utils.mlx_utils import GenerationResult


def _mock_mlx_generate(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens, logprobs=[], generation_tps=50.0)


def test_identical_outputs_pass():
    """MLX and reference produce identical tokens -> PASS."""
    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("Hello world", [100, 200, 300])

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return [100, 200, 300]

    with patch("mlx_triage.tier1.reference_divergence.generate_text", side_effect=fake_mlx_gen):
        with patch("mlx_triage.tier1.reference_divergence._generate_reference", side_effect=fake_ref_gen):
            with patch("mlx_triage.tier1.reference_divergence.load_model", return_value=(MagicMock(), MagicMock())):
                with patch("mlx_triage.tier1.reference_divergence.check_mlx_available", return_value=True):
                    with patch("mlx_triage.tier1.reference_divergence._check_reference_available", return_value=True):
                        result = check_reference_divergence("/fake/model", n_prompts=3)

    assert result.check_id == "1.2"
    assert result.status == CheckStatus.PASS


def test_high_divergence_fails():
    """Completely different tokens -> FAIL."""
    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("MLX output", [100, 200, 300])

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return [900, 800, 700]  # Completely different

    with patch("mlx_triage.tier1.reference_divergence.generate_text", side_effect=fake_mlx_gen):
        with patch("mlx_triage.tier1.reference_divergence._generate_reference", side_effect=fake_ref_gen):
            with patch("mlx_triage.tier1.reference_divergence.load_model", return_value=(MagicMock(), MagicMock())):
                with patch("mlx_triage.tier1.reference_divergence.check_mlx_available", return_value=True):
                    with patch("mlx_triage.tier1.reference_divergence._check_reference_available", return_value=True):
                        result = check_reference_divergence("/fake/model", n_prompts=3)

    assert result.status in (CheckStatus.FAIL, CheckStatus.WARNING)


def test_mlx_not_available_skips():
    """When MLX is not installed -> SKIP."""
    with patch("mlx_triage.tier1.reference_divergence.check_mlx_available", return_value=False):
        result = check_reference_divergence("/fake/model")
    assert result.status == CheckStatus.SKIP
    assert "mlx" in result.remediation.lower()


def test_reference_not_available_skips():
    """When transformers/torch not installed -> SKIP."""
    with patch("mlx_triage.tier1.reference_divergence.check_mlx_available", return_value=True):
        with patch("mlx_triage.tier1.reference_divergence._check_reference_available", return_value=False):
            result = check_reference_divergence("/fake/model")
    assert result.status == CheckStatus.SKIP
    assert "reference" in result.remediation.lower()


def test_partial_agreement_info():
    """~90% agreement -> INFO (expected cross-backend variance)."""
    tokens_mlx = list(range(100, 120))  # 20 tokens
    tokens_ref = list(range(100, 118)) + [999, 998]  # 18 match, 2 differ = 90%

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("output", tokens_mlx)

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return tokens_ref

    with patch("mlx_triage.tier1.reference_divergence.generate_text", side_effect=fake_mlx_gen):
        with patch("mlx_triage.tier1.reference_divergence._generate_reference", side_effect=fake_ref_gen):
            with patch("mlx_triage.tier1.reference_divergence.load_model", return_value=(MagicMock(), MagicMock())):
                with patch("mlx_triage.tier1.reference_divergence.check_mlx_available", return_value=True):
                    with patch("mlx_triage.tier1.reference_divergence._check_reference_available", return_value=True):
                        result = check_reference_divergence("/fake/model", n_prompts=2)

    assert result.status == CheckStatus.INFO
