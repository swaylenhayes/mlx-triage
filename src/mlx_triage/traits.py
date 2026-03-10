"""Model traits assembly from diagnostic check results.

Collects metadata scattered across individual check results into a
consolidated traits object for downstream consumers (JSON reports,
capability endpoints).
"""

from __future__ import annotations

from mlx_triage.models import DiagnosticResult

# Model families known to never support thinking/reasoning tokens.
_NON_REASONING_FAMILIES = frozenset(
    {
        "llama",
        "mistral",
        "mistral3",
        "gemma",
        "gemma3",
        "gemma3_text",
        "phi",
        "phi3",
        "phi4",
        "starcoder",
        "gpt_neox",
        "mpt",
        "falcon",
        "nanbeige",
        "glm4_moe_lite",
        "nemotron_h",
        "lfm2",
    }
)


def collect_traits(checks: list[DiagnosticResult]) -> dict:
    """Assemble model traits from completed check results.

    Reads metadata from checks 0.2 (tokenizer), 0.4 (version/bugs),
    and 0.5 (architecture) to produce a flat traits dict.
    """
    by_id: dict[str, DiagnosticResult] = {c.check_id: c for c in checks}

    tok = by_id.get("0.2")
    ver = by_id.get("0.4")
    arch = by_id.get("0.5")

    has_chat_template = tok.metadata.get("has_chat_template") if tok else None
    has_thinking_tokens = tok.metadata.get("has_thinking_tokens") if tok else None
    is_vlm = arch.metadata.get("is_vlm") if arch else None
    architecture_type = arch.metadata.get("architecture") if arch else None
    known_issues = ver.metadata.get("matched_bug_ids", []) if ver else []

    model_type = ver.metadata.get("architecture") if ver else None
    reasoning_mechanism = _classify_reasoning(has_thinking_tokens, model_type)

    return {
        "has_chat_template": has_chat_template,
        "has_thinking_tokens": has_thinking_tokens,
        "reasoning_mechanism": reasoning_mechanism,
        "is_vlm": is_vlm,
        "architecture_type": architecture_type,
        "known_issues": known_issues,
    }


def _classify_reasoning(
    has_thinking_tokens: bool | None, model_type: str | None
) -> str:
    """Classify reasoning mechanism based on tokenizer + model family.

    Returns:
        "think_tag" -- model has <think>/</think> tokens
        "none" -- model is in a known non-reasoning family
        "unknown" -- insufficient information to classify
    """
    if has_thinking_tokens is True:
        return "think_tag"
    if has_thinking_tokens is False and model_type in _NON_REASONING_FAMILIES:
        return "none"
    return "unknown"
