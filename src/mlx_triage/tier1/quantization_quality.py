# src/mlx_triage/tier1/quantization_quality.py
"""Test 1.3: Quantization Quality Gate.

Measures model perplexity on a fixed evaluation corpus to verify that
quantization hasn't degraded model quality below acceptable thresholds.

Evidence: well-quantized 7B models typically achieve perplexity <10 on
standard text. 4-bit quantized models may be 10-15. Values >25 suggest
quality issues, and >50 indicates a broken model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import EVAL_CORPUS

if TYPE_CHECKING:
    from mlx_triage.utils.backends import ModelBackend

# Perplexity thresholds
GOOD_PPL = 15.0  # < 15 -> PASS (good quality)
ACCEPTABLE_PPL = 25.0  # < 25 -> INFO (acceptable for aggressive quant)
CONCERNING_PPL = 50.0  # < 50 -> WARNING (quality degradation)
# >= 50 -> FAIL (model is broken or severely degraded)


def check_quantization_quality(
    model_path: str,
    model: object | None = None,
    tokenizer: object | None = None,
    backend: ModelBackend | None = None,
) -> DiagnosticResult:
    """Run quantization quality gate using perplexity measurement.

    Args:
        model_path: Path to MLX model directory.
        model: Pre-loaded MLX model (optional, avoids redundant loading).
        tokenizer: Pre-loaded tokenizer (optional, avoids redundant loading).
        backend: ModelBackend instance (optional, defaults to MLXLMBackend).
    """
    if backend is None:
        from mlx_triage.utils.mlx_utils import MLXLMBackend

        backend = MLXLMBackend()

    if not backend.is_available():
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run quantization quality check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    if model is None or tokenizer is None:
        model, tokenizer = backend.load(model_path)

    try:
        ppl = backend.compute_perplexity(model, tokenizer, EVAL_CORPUS)
    except Exception as e:
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.FAIL,
            detail=f"Perplexity computation error: {e}",
            remediation="Check model integrity. The model may be corrupted or incompatible.",
            metadata={"error": str(e)},
        )

    if ppl < GOOD_PPL:
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.PASS,
            detail=f"Model perplexity is good ({ppl:.1f}). Quantization quality is acceptable.",
            metadata={"perplexity": ppl},
        )

    if ppl < ACCEPTABLE_PPL:
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.INFO,
            detail=f"Model perplexity is moderate ({ppl:.1f}). Acceptable for aggressive quantization.",
            metadata={"perplexity": ppl},
        )

    if ppl < CONCERNING_PPL:
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.WARNING,
            detail=f"Model perplexity is elevated ({ppl:.1f}). Quality may be degraded.",
            remediation="Consider using a less aggressive quantization (e.g., Q8 instead of Q4) or re-quantizing.",
            metadata={"perplexity": ppl},
        )

    return DiagnosticResult(
        check_id="1.3",
        name="Quantization Quality",
        status=CheckStatus.FAIL,
        detail=f"Model perplexity is very high ({ppl:.1f}). Model quality is severely degraded.",
        remediation="The model appears broken or severely mis-quantized. Re-download or use a different quantization.",
        metadata={"perplexity": ppl},
    )
