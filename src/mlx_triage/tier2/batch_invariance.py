"""Test 2.1: Batch invariance."""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import DIAGNOSTIC_PROMPTS
from mlx_triage.utils.backends import ModelBackend
from mlx_triage.utils.comparison import divergence_point, token_agreement_rate

MINOR_DIVERGENCE_THRESHOLD = 0.02
MAJOR_DIVERGENCE_THRESHOLD = 0.05
DEFAULT_N_PROMPTS = 3


def check_batch_invariance(
    model_path: str,
    n_prompts: int = DEFAULT_N_PROMPTS,
    seed: int = 42,
    model: object | None = None,
    tokenizer: object | None = None,
    backend: ModelBackend | None = None,
) -> DiagnosticResult:
    """Verify output quality is invariant under batching."""
    if backend is None:
        from mlx_triage.utils.mlx_utils import MLXLMBackend

        backend = MLXLMBackend()

    if not backend.is_available():
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run batch invariance check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    if not hasattr(backend, "generate_batch"):
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.SKIP,
            detail=(
                "Current backend does not expose batched generation. "
                "A real single-vs-batch comparison cannot run yet."
            ),
            remediation="Extend ModelBackend with a real batch generation path.",
            metadata={"blocked_on": "backend_batch_generation"},
        )

    if model is None or tokenizer is None:
        model, tokenizer = backend.load(model_path)

    prompt_specs = DIAGNOSTIC_PROMPTS[:n_prompts]
    prompts = [prompt_spec["prompt"] for prompt_spec in prompt_specs]
    max_tokens = [prompt_spec["max_tokens"] for prompt_spec in prompt_specs]

    try:
        single_results = [
            backend.generate_text(
                model,
                tokenizer,
                prompt_spec["prompt"],
                max_tokens=prompt_spec["max_tokens"],
                temp=0.0,
                seed=seed,
            )
            for prompt_spec in prompt_specs
        ]
        batch_results = backend.generate_batch(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            temp=0.0,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.CRITICAL,
            detail=f"Batch invariance generation failed: {exc}",
            remediation="Inspect the backend batch path and MLX batching support.",
        )

    if len(batch_results) != len(prompt_specs):
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.CRITICAL,
            detail=(
                "Batch generation returned an unexpected number of results "
                f"({len(batch_results)} for {len(prompt_specs)} prompts)."
            ),
            remediation="Inspect backend.generate_batch result ordering and cardinality.",
        )

    comparisons: list[dict[str, object]] = []
    empty_output_detected = False

    for prompt_spec, single_result, batch_result in zip(
        prompt_specs, single_results, batch_results
    ):
        if not single_result.tokens or not batch_result.tokens:
            empty_output_detected = True

        agreement = token_agreement_rate(single_result.tokens, batch_result.tokens)
        comparisons.append(
            {
                "prompt_id": prompt_spec["id"],
                "agreement_rate": agreement,
                "single_text": single_result.text,
                "batch_text": batch_result.text,
                "divergence_point": divergence_point(
                    single_result.tokens, batch_result.tokens
                ),
            }
        )

    if empty_output_detected:
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.CRITICAL,
            detail="Empty output detected in single or batched generation.",
            remediation="Inspect generation stability before relying on batch comparisons.",
            metadata={"comparisons": comparisons, "n_prompts": n_prompts},
        )

    avg_agreement = sum(
        float(comparison["agreement_rate"]) for comparison in comparisons
    ) / len(comparisons)
    min_agreement = min(float(comparison["agreement_rate"]) for comparison in comparisons)
    divergence_rate = 1.0 - avg_agreement
    worst_prompt = min(comparisons, key=lambda comparison: comparison["agreement_rate"])
    worst_prompt_id = worst_prompt["prompt_id"]

    metadata = {
        "comparisons": comparisons,
        "avg_agreement": avg_agreement,
        "min_agreement": min_agreement,
        "divergence_rate": divergence_rate,
        "n_prompts": n_prompts,
        "worst_prompt_id": worst_prompt_id,
    }

    if all(float(comparison["agreement_rate"]) == 1.0 for comparison in comparisons):
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.PASS,
            detail=(
                "Single and batched generation matched across all tested prompts "
                "(100% token agreement)."
            ),
            metadata=metadata,
        )

    if divergence_rate <= MINOR_DIVERGENCE_THRESHOLD:
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.INFO,
            detail=(
                "Minor batch-sensitive variance detected "
                f"({divergence_rate:.1%} divergence, worst prompt {worst_prompt_id})."
            ),
            metadata=metadata,
        )

    if divergence_rate <= MAJOR_DIVERGENCE_THRESHOLD:
        return DiagnosticResult(
            check_id="2.1",
            name="Batch Invariance",
            status=CheckStatus.WARNING,
            detail=(
                "Moderate single-vs-batch divergence detected "
                f"({divergence_rate:.1%} divergence, worst prompt {worst_prompt_id})."
            ),
            remediation="Investigate batch-size sensitivity and compare with smaller completion batches.",
            metadata=metadata,
        )

    return DiagnosticResult(
        check_id="2.1",
        name="Batch Invariance",
        status=CheckStatus.FAIL,
        detail=(
            "Significant batch invariance failure detected "
            f"({divergence_rate:.1%} divergence, worst prompt {worst_prompt_id})."
        ),
        remediation="This suggests runtime or kernel-level batch sensitivity. Compare batch size 1 against multi-request execution.",
        metadata=metadata,
    )
