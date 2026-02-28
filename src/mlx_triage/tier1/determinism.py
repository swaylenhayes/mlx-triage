# src/mlx_triage/tier1/determinism.py
"""Test 1.1: Determinism Check.

Verifies that the model produces identical output for identical inputs at
temperature=0. Non-determinism at temp=0 indicates infrastructure issues.

Evidence: quantized integer models (Q4, Q8) achieve perfect reproducibility
on MLX, while float16 may have minor variance. If a quantized model is
non-deterministic, something is very wrong.
"""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import DIAGNOSTIC_PROMPTS
from mlx_triage.utils.comparison import multi_run_consistency
from mlx_triage.utils.mlx_utils import (
    GenerationResult,
    check_mlx_available,
    generate_text,
    load_model,
)

# Thresholds from Token-DiFR evidence
MINOR_DIVERGENCE_THRESHOLD = 0.02  # < 2% = expected floating-point variance
MAJOR_DIVERGENCE_THRESHOLD = 0.05  # > 5% = infrastructure non-determinism


def check_determinism(
    model_path: str,
    n_runs: int = 10,
    n_prompts: int = 5,
    seed: int = 42,
) -> DiagnosticResult:
    """Run determinism check: same prompt N times at temp=0, compare outputs.

    Args:
        model_path: Path to MLX model directory.
        n_runs: Number of times to run each prompt.
        n_prompts: Number of prompts to test (from diagnostic suite).
        seed: Random seed for reproducibility.
    """
    if not check_mlx_available():
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run determinism check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    model, tokenizer = load_model(model_path)
    prompts = DIAGNOSTIC_PROMPTS[:n_prompts]

    all_consistencies: list[dict] = []
    empty_detected = False

    for prompt_spec in prompts:
        prompt = prompt_spec["prompt"]
        max_tokens = prompt_spec["max_tokens"]

        runs: list[list[int]] = []
        for _ in range(n_runs):
            result = generate_text(
                model, tokenizer, prompt,
                max_tokens=max_tokens, temp=0.0, seed=seed,
            )
            if not result.tokens:
                empty_detected = True
            runs.append(result.tokens)

        consistency = multi_run_consistency(runs)
        consistency["prompt_id"] = prompt_spec["id"]
        all_consistencies.append(consistency)

    if empty_detected:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.CRITICAL,
            detail="Empty output detected in one or more runs. Model may be broken.",
            remediation="Check model weights and configuration. Run Tier 0 checks first.",
            metadata={"consistencies": all_consistencies},
        )

    # Compute overall agreement
    avg_agreement = sum(c["agreement_rate"] for c in all_consistencies) / len(all_consistencies)
    all_consistent = all(c["consistent"] for c in all_consistencies)
    divergence_rate = 1.0 - avg_agreement

    if all_consistent:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.PASS,
            detail=f"Model is deterministic at temp=0. {n_prompts} prompts x {n_runs} runs, 100% token agreement.",
            metadata={"avg_agreement": avg_agreement, "n_prompts": n_prompts, "n_runs": n_runs},
        )

    if divergence_rate <= MINOR_DIVERGENCE_THRESHOLD:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.INFO,
            detail=f"Minor variance detected ({divergence_rate:.1%} divergence). Expected for float16 models.",
            metadata={"avg_agreement": avg_agreement, "divergence_rate": divergence_rate},
        )

    if divergence_rate <= MAJOR_DIVERGENCE_THRESHOLD:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.WARNING,
            detail=f"Moderate non-determinism detected ({divergence_rate:.1%} divergence).",
            remediation="Check if model is quantized. Quantized models should be deterministic at temp=0.",
            metadata={"avg_agreement": avg_agreement, "divergence_rate": divergence_rate},
        )

    return DiagnosticResult(
        check_id="1.1",
        name="Determinism",
        status=CheckStatus.FAIL,
        detail=f"Significant non-determinism detected ({divergence_rate:.1%} divergence). Infrastructure issue likely.",
        remediation="This suggests a hardware or driver issue. Check MLX version, Metal driver, and system stability.",
        metadata={"avg_agreement": avg_agreement, "divergence_rate": divergence_rate, "consistencies": all_consistencies},
    )
