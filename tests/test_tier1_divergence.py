# tests/test_tier1_divergence.py
"""Tests for reference divergence check. All inference is mocked via backend."""

from unittest.mock import MagicMock, patch

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.reference_divergence import (
    ReferenceBackendError,
    check_reference_divergence,
)
from mlx_triage.utils.mlx_utils import GenerationResult


def _mock_mlx_generate(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens, logprobs=[], generation_tps=50.0)


def _make_backend(available=True, generate_side_effect=None):
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.load.return_value = (MagicMock(), MagicMock())
    if generate_side_effect:
        backend.generate_text.side_effect = generate_side_effect
    return backend


def test_identical_outputs_pass():
    """MLX and reference produce identical tokens -> PASS."""

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("Hello world", [100, 200, 300])

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return [100, 200, 300]

    backend = _make_backend(generate_side_effect=fake_mlx_gen)

    with patch(
        "mlx_triage.tier1.reference_divergence._generate_reference",
        side_effect=fake_ref_gen,
    ):
        with patch(
            "mlx_triage.tier1.reference_divergence._check_reference_available",
            return_value=True,
        ):
            result = check_reference_divergence(
                "/fake/model", n_prompts=3, backend=backend
            )

    assert result.check_id == "1.2"
    assert result.status == CheckStatus.PASS


def test_high_divergence_fails():
    """Completely different tokens -> FAIL."""

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("MLX output", [100, 200, 300])

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return [900, 800, 700]

    backend = _make_backend(generate_side_effect=fake_mlx_gen)

    with patch(
        "mlx_triage.tier1.reference_divergence._generate_reference",
        side_effect=fake_ref_gen,
    ):
        with patch(
            "mlx_triage.tier1.reference_divergence._check_reference_available",
            return_value=True,
        ):
            result = check_reference_divergence(
                "/fake/model", n_prompts=3, backend=backend
            )

    assert result.status in (CheckStatus.FAIL, CheckStatus.WARNING)


def test_backend_not_available_skips():
    """When backend is not available -> SKIP."""
    backend = _make_backend(available=False)
    result = check_reference_divergence("/fake/model", backend=backend)
    assert result.status == CheckStatus.SKIP
    assert "mlx" in result.remediation.lower()


def test_reference_not_available_skips():
    """When transformers/torch not installed -> SKIP."""
    backend = _make_backend()
    with patch(
        "mlx_triage.tier1.reference_divergence._check_reference_available",
        return_value=False,
    ):
        result = check_reference_divergence("/fake/model", backend=backend)
    assert result.status == CheckStatus.SKIP
    assert "reference" in result.remediation.lower()


def test_partial_agreement_info():
    """~90% agreement -> INFO (expected cross-backend variance)."""
    tokens_mlx = list(range(100, 120))
    tokens_ref = list(range(100, 118)) + [999, 998]

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("output", tokens_mlx)

    def fake_ref_gen(model_path, prompt, max_tokens, **kwargs):
        return tokens_ref

    backend = _make_backend(generate_side_effect=fake_mlx_gen)

    with patch(
        "mlx_triage.tier1.reference_divergence._generate_reference",
        side_effect=fake_ref_gen,
    ):
        with patch(
            "mlx_triage.tier1.reference_divergence._check_reference_available",
            return_value=True,
        ):
            result = check_reference_divergence(
                "/fake/model", n_prompts=2, backend=backend
            )

    assert result.status == CheckStatus.INFO


def test_reference_backend_error_skips():
    """Unsupported reference backend model format should SKIP, not crash."""

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("output", [1, 2, 3])

    backend = _make_backend(generate_side_effect=fake_mlx_gen)

    with patch(
        "mlx_triage.tier1.reference_divergence._generate_reference",
        side_effect=ReferenceBackendError("quantization_config is missing quant_method"),
    ):
        with patch(
            "mlx_triage.tier1.reference_divergence._check_reference_available",
            return_value=True,
        ):
            result = check_reference_divergence(
                "/fake/model", n_prompts=1, backend=backend
            )

    assert result.status == CheckStatus.SKIP
    assert "unavailable for this model" in result.detail


def test_reference_runtime_error_skips():
    """Unexpected reference runtime errors should SKIP with diagnostics."""

    def fake_mlx_gen(model, tokenizer, prompt, **kwargs):
        return _mock_mlx_generate("output", [1, 2, 3])

    backend = _make_backend(generate_side_effect=fake_mlx_gen)

    with patch(
        "mlx_triage.tier1.reference_divergence._generate_reference",
        side_effect=RuntimeError("reference backend exploded"),
    ):
        with patch(
            "mlx_triage.tier1.reference_divergence._check_reference_available",
            return_value=True,
        ):
            result = check_reference_divergence(
                "/fake/model", n_prompts=1, backend=backend
            )

    assert result.status == CheckStatus.SKIP
    assert "Reference generation failed" in result.detail
