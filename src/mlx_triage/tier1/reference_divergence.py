# src/mlx_triage/tier1/reference_divergence.py
"""Test 1.2: Reference Divergence.

Compares MLX model output against a Transformers (PyTorch) reference backend.
Identifies whether token-level differences between backends exceed expected
cross-implementation variance.

Requires optional dependency: mlx-triage[reference] (transformers, torch)
"""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import DIAGNOSTIC_PROMPTS
from mlx_triage.utils.comparison import divergence_point, token_agreement_rate
from mlx_triage.utils.mlx_utils import (
    check_mlx_available,
    generate_text,
    load_model,
)

# Thresholds for cross-backend agreement
HIGH_AGREEMENT = 0.95  # >95% -> PASS
MODERATE_AGREEMENT = 0.85  # >85% -> INFO (expected variance)
LOW_AGREEMENT = 0.70  # >70% -> WARNING


def _check_reference_available() -> bool:
    """Check if transformers and torch are installed."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _generate_reference(
    model_path: str,
    prompt: str,
    max_tokens: int = 256,
) -> list[int]:
    """Generate tokens using Transformers as reference backend.

    Args:
        model_path: Path to model directory (HuggingFace format).
        prompt: Text prompt.
        max_tokens: Maximum tokens to generate.

    Returns:
        List of token IDs from reference generation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ref_tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu"
    )

    inputs = ref_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = ref_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding
        )

    # Extract only generated tokens (exclude input)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:].tolist()

    return generated_ids


def check_reference_divergence(
    model_path: str,
    n_prompts: int = 5,
    seed: int = 42,
    model: object | None = None,
    tokenizer: object | None = None,
) -> DiagnosticResult:
    """Compare MLX output against Transformers reference.

    Args:
        model_path: Path to MLX model directory.
        n_prompts: Number of prompts to test.
        seed: Random seed for MLX generation.
        model: Pre-loaded MLX model (optional, avoids redundant loading).
        tokenizer: Pre-loaded tokenizer (optional, avoids redundant loading).
    """
    if not check_mlx_available():
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run reference divergence check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    if not _check_reference_available():
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.SKIP,
            detail="Transformers/torch not installed. Cannot run reference comparison.",
            remediation="Install reference dependencies: uv sync --extra reference",
        )

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)
    prompts = [p for p in DIAGNOSTIC_PROMPTS[:n_prompts] if isinstance(p["prompt"], str)]

    comparisons: list[dict] = []

    for prompt_spec in prompts:
        prompt_text = prompt_spec["prompt"]
        max_tokens = prompt_spec["max_tokens"]

        # MLX generation
        mlx_result = generate_text(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            temp=0.0,
            seed=seed,
        )

        # Reference generation
        ref_tokens = _generate_reference(model_path, prompt_text, max_tokens=max_tokens)

        agreement = token_agreement_rate(mlx_result.tokens, ref_tokens)
        div_point = divergence_point(mlx_result.tokens, ref_tokens)

        comparisons.append(
            {
                "prompt_id": prompt_spec["id"],
                "agreement": agreement,
                "divergence_point": div_point,
                "mlx_len": len(mlx_result.tokens),
                "ref_len": len(ref_tokens),
            }
        )

    if not comparisons:
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.SKIP,
            detail="No suitable prompts for reference comparison.",
        )

    avg_agreement = sum(c["agreement"] for c in comparisons) / len(comparisons)
    min_agreement = min(c["agreement"] for c in comparisons)

    if avg_agreement >= HIGH_AGREEMENT:
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.PASS,
            detail=f"MLX output closely matches reference. Avg agreement: {avg_agreement:.1%}.",
            metadata={"avg_agreement": avg_agreement, "comparisons": comparisons},
        )

    if avg_agreement >= MODERATE_AGREEMENT:
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.INFO,
            detail=f"Minor cross-backend variance detected. Avg agreement: {avg_agreement:.1%}.",
            metadata={
                "avg_agreement": avg_agreement,
                "min_agreement": min_agreement,
                "comparisons": comparisons,
            },
        )

    if avg_agreement >= LOW_AGREEMENT:
        return DiagnosticResult(
            check_id="1.2",
            name="Reference Divergence",
            status=CheckStatus.WARNING,
            detail=f"Notable divergence from reference. Avg agreement: {avg_agreement:.1%}.",
            remediation="Check model conversion quality. Re-download or re-convert the model.",
            metadata={
                "avg_agreement": avg_agreement,
                "min_agreement": min_agreement,
                "comparisons": comparisons,
            },
        )

    return DiagnosticResult(
        check_id="1.2",
        name="Reference Divergence",
        status=CheckStatus.FAIL,
        detail=f"Significant divergence from reference. Avg agreement: {avg_agreement:.1%}.",
        remediation=(
            "Model output significantly differs from reference. "
            "This may indicate a bad conversion, corrupted weights, "
            "or incompatible quantization."
        ),
        metadata={
            "avg_agreement": avg_agreement,
            "min_agreement": min_agreement,
            "comparisons": comparisons,
        },
    )
