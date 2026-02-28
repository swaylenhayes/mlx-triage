# src/mlx_triage/tier1/quantization_quality.py
"""Test 1.3: Quantization Quality Gate.

Measures model perplexity on a fixed evaluation corpus to verify that
quantization hasn't degraded model quality below acceptable thresholds.

Evidence: well-quantized 7B models typically achieve perplexity <10 on
standard text. 4-bit quantized models may be 10-15. Values >25 suggest
quality issues, and >50 indicates a broken model.
"""

from __future__ import annotations

import math

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import EVAL_CORPUS
from mlx_triage.utils.mlx_utils import (
    check_mlx_available,
    load_model,
)

# Perplexity thresholds
GOOD_PPL = 15.0  # < 15 -> PASS (good quality)
ACCEPTABLE_PPL = 25.0  # < 25 -> INFO (acceptable for aggressive quant)
CONCERNING_PPL = 50.0  # < 50 -> WARNING (quality degradation)
# >= 50 -> FAIL (model is broken or severely degraded)


def _compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of text using the model.

    Tokenizes the text, feeds it through the model, and computes
    the average cross-entropy loss across all tokens. Returns
    exp(avg_loss) as the perplexity score.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        text: Evaluation text.

    Returns:
        Perplexity score (lower is better).
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Tokenize the evaluation corpus
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        raise ValueError("Evaluation text too short for perplexity computation")

    token_array = mx.array([tokens])  # Shape: (1, seq_len)

    # Forward pass to get logits
    logits = model(token_array)  # Shape: (1, seq_len, vocab_size)

    # Shift for next-token prediction: predict token[i+1] from logits[i]
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab_size)
    shift_labels = token_array[:, 1:]  # (1, seq_len-1)

    # Compute cross-entropy loss
    loss = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    avg_loss = mx.mean(loss).item()

    return math.exp(avg_loss)


def check_quantization_quality(
    model_path: str,
) -> DiagnosticResult:
    """Run quantization quality gate using perplexity measurement.

    Args:
        model_path: Path to MLX model directory.
    """
    if not check_mlx_available():
        return DiagnosticResult(
            check_id="1.3",
            name="Quantization Quality",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run quantization quality check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    model, tokenizer = load_model(model_path)

    try:
        ppl = _compute_perplexity(model, tokenizer, EVAL_CORPUS)
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
