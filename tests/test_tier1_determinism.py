# tests/test_tier1_determinism.py
"""Tests for determinism check. All inference is mocked via backend."""

from unittest.mock import MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.determinism import check_determinism
from mlx_triage.utils.mlx_utils import GenerationResult


def _mock_generate(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens, logprobs=[], generation_tps=50.0)


def _make_backend(available=True, generate_side_effect=None):
    """Create a mock backend for determinism tests."""
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.load.return_value = (MagicMock(), MagicMock())
    if generate_side_effect:
        backend.generate_text.side_effect = generate_side_effect
    return backend


def test_deterministic_model_passes():
    """Model producing identical output every run -> PASS."""

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("Hello world", [100, 200, 300])

    backend = _make_backend(generate_side_effect=fake_generate)
    result = check_determinism("/fake/model", n_runs=5, n_prompts=3, backend=backend)
    assert result.check_id == "1.1"
    assert result.status == CheckStatus.PASS


def test_major_divergence_fails():
    """Completely different output each run -> FAIL."""
    call_count = [0]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        call_count[0] += 1
        return _mock_generate(
            f"output_{call_count[0]}", [call_count[0], call_count[0] + 1]
        )

    backend = _make_backend(generate_side_effect=fake_generate)
    result = check_determinism("/fake/model", n_runs=5, n_prompts=3, backend=backend)
    assert result.status in (CheckStatus.FAIL, CheckStatus.CRITICAL)


def test_empty_output_critical():
    """Empty output on any run -> CRITICAL."""

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("", [])

    backend = _make_backend(generate_side_effect=fake_generate)
    result = check_determinism("/fake/model", n_runs=3, n_prompts=2, backend=backend)
    assert result.status == CheckStatus.CRITICAL


def test_backend_not_available_skips():
    """When backend is not available, should SKIP."""
    backend = _make_backend(available=False)
    result = check_determinism("/fake/model", backend=backend)
    assert result.status == CheckStatus.SKIP
