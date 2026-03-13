"""Test 2.3: Context length stress."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.utils.backends import ModelBackend

DEFAULT_CONTEXT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_POSITIONS = (0.25, 0.5, 0.75)
FILLER_SENTENCE = (
    "This is filler context for retrieval testing and should be ignored by the model."
)
QUESTION = "Question: What is the secret code? Answer with just the code."


def _token_count(tokenizer: object, text: str) -> int:
    """Count tokens using the tokenizer when possible."""
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text)
            return len(encoded)
        except TypeError:
            pass
    return len(text.split())


def _repeat_to_token_budget(tokenizer: object, budget: int) -> str:
    """Expand filler text until it roughly fills the requested token budget."""
    if budget <= 0:
        return ""

    parts: list[str] = []
    while _token_count(tokenizer, " ".join(parts)) < budget:
        parts.append(FILLER_SENTENCE)

    return " ".join(parts)


def _build_needle_prompt(
    tokenizer: object,
    target_tokens: int,
    position: float,
    needle: str,
) -> str:
    """Construct a retrieval prompt with the needle at a chosen relative position."""
    needle_sentence = f"Important fact: the secret code is {needle}."
    overhead = _token_count(tokenizer, f"{needle_sentence}\n\n{QUESTION}")
    filler_budget = max(target_tokens - overhead, 0)
    prefix_budget = int(filler_budget * position)
    suffix_budget = filler_budget - prefix_budget

    prefix = _repeat_to_token_budget(tokenizer, prefix_budget)
    suffix = _repeat_to_token_budget(tokenizer, suffix_budget)
    sections = [section for section in (prefix, needle_sentence, suffix, QUESTION) if section]
    return "\n\n".join(sections)


def _contains_needle(text: str, needle: str) -> bool:
    """Check whether the response clearly contains the expected code."""
    normalized_text = text.strip().lower()
    normalized_needle = needle.lower()
    return normalized_needle in normalized_text


def _read_max_context(model_path: str) -> int | None:
    """Read max context length from config.json when exposed."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def check_context_length(
    model_path: str,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    positions: Sequence[float] = DEFAULT_POSITIONS,
    seed: int = 42,
    model: object | None = None,
    tokenizer: object | None = None,
    backend: ModelBackend | None = None,
) -> DiagnosticResult:
    """Detect retrieval failures that emerge at longer context lengths."""
    if backend is None:
        from mlx_triage.utils.mlx_utils import MLXLMBackend

        backend = MLXLMBackend()

    if not backend.is_available():
        return DiagnosticResult(
            check_id="2.3",
            name="Context Length Stress",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run context length stress test.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    if model is None or tokenizer is None:
        model, tokenizer = backend.load(model_path)

    max_context = _read_max_context(model_path)
    lengths = [length for length in context_lengths if max_context is None or length <= max_context]
    if not lengths:
        return DiagnosticResult(
            check_id="2.3",
            name="Context Length Stress",
            status=CheckStatus.SKIP,
            detail="Could not determine any supported context lengths for this model.",
            remediation="Verify the model exposes max_position_embeddings or equivalent context metadata.",
        )

    results: list[dict[str, int | float | bool | str]] = []

    for length in lengths:
        for position in positions:
            needle = f"code-{length}-{int(position * 100)}"
            prompt = _build_needle_prompt(tokenizer, length, position, needle)
            try:
                response = backend.generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=16,
                    temp=0.0,
                    seed=seed,
                )
            except Exception as exc:
                return DiagnosticResult(
                    check_id="2.3",
                    name="Context Length Stress",
                    status=CheckStatus.CRITICAL,
                    detail=f"Generation failed at context length {length}: {exc}",
                    remediation="Investigate long-context stability, KV cache behavior, and model context support.",
                    metadata={"results": results, "failed_at_length": length},
                )

            results.append(
                {
                    "context_length": length,
                    "position": position,
                    "needle": needle,
                    "retrieved": _contains_needle(response.text, needle),
                }
            )

    per_length: dict[int, list[bool]] = defaultdict(list)
    for result in results:
        per_length[int(result["context_length"])].append(bool(result["retrieved"]))

    accuracies = {
        length: sum(outcomes) / len(outcomes)
        for length, outcomes in per_length.items()
    }
    ordered_lengths = sorted(accuracies)
    accuracy_series = [accuracies[length] for length in ordered_lengths]

    if all(accuracy == 1.0 for accuracy in accuracy_series):
        return DiagnosticResult(
            check_id="2.3",
            name="Context Length Stress",
            status=CheckStatus.PASS,
            detail="Needle retrieval remained stable across all tested context lengths.",
            metadata={"results": results, "accuracies": accuracies},
        )

    baseline_accuracy = accuracy_series[0]
    min_accuracy = min(accuracy_series)
    cliff_drop = baseline_accuracy - min_accuracy

    if baseline_accuracy >= 2 / 3 and cliff_drop >= 2 / 3 and min_accuracy <= 1 / 3:
        failing_length = next(
            length for length in ordered_lengths if accuracies[length] == min_accuracy
        )
        return DiagnosticResult(
            check_id="2.3",
            name="Context Length Stress",
            status=CheckStatus.FAIL,
            detail=(
                "Retrieval accuracy dropped sharply at longer contexts "
                f"(lowest accuracy {min_accuracy:.0%} at {failing_length} tokens)."
            ),
            remediation="Investigate RoPE precision, KV cache stability, and long-context runtime behavior.",
            metadata={"results": results, "accuracies": accuracies},
        )

    degraded_length = next(
        length for length in ordered_lengths if accuracies[length] < 1.0
    )
    return DiagnosticResult(
        check_id="2.3",
        name="Context Length Stress",
        status=CheckStatus.WARNING,
        detail=(
            "Retrieval accuracy degraded at longer context lengths "
            f"(first drop at {degraded_length} tokens)."
        ),
        remediation="Check long-context support and test with shorter prompts or lower context pressure.",
        metadata={"results": results, "accuracies": accuracies},
    )
