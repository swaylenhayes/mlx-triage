"""Tests for model traits assembly."""

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.traits import collect_traits


def _make_checks(
    has_chat_template: bool = True,
    has_thinking_tokens: bool = False,
    model_type: str = "llama",
    architecture: str = "LlamaForCausalLM",
    is_vlm: bool = False,
    bug_ids: list[str] | None = None,
) -> list[DiagnosticResult]:
    """Helper to build a typical set of check results."""
    checks = [
        DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.PASS,
            detail="OK",
            metadata={
                "has_chat_template": has_chat_template,
                "has_thinking_tokens": has_thinking_tokens,
            },
        ),
        DiagnosticResult(
            check_id="0.4",
            name="MLX Version",
            status=CheckStatus.PASS,
            detail="OK",
            metadata={"mlx_version": "0.30.6", "architecture": model_type},
        ),
        DiagnosticResult(
            check_id="0.5",
            name="Architecture",
            status=CheckStatus.PASS if not is_vlm else CheckStatus.FAIL,
            detail="OK",
            metadata={"is_vlm": is_vlm, "architecture": architecture},
        ),
    ]
    if bug_ids is not None:
        checks[1].metadata["matched_bug_ids"] = bug_ids
    return checks


class TestCollectTraitsBasic:
    """Basic trait assembly from check metadata."""

    def test_all_trait_keys_present(self):
        traits = collect_traits(_make_checks())
        expected_keys = {
            "has_chat_template",
            "has_thinking_tokens",
            "reasoning_mechanism",
            "is_vlm",
            "architecture_type",
            "known_issues",
        }
        assert set(traits.keys()) == expected_keys

    def test_standard_model_traits(self):
        traits = collect_traits(_make_checks())
        assert traits["has_chat_template"] is True
        assert traits["has_thinking_tokens"] is False
        assert traits["is_vlm"] is False
        assert traits["architecture_type"] == "LlamaForCausalLM"
        assert traits["known_issues"] == []

    def test_empty_checks_returns_defaults(self):
        traits = collect_traits([])
        assert traits["has_chat_template"] is None
        assert traits["has_thinking_tokens"] is None
        assert traits["is_vlm"] is None
        assert traits["architecture_type"] is None
        assert traits["known_issues"] == []
        assert traits["reasoning_mechanism"] == "unknown"

    def test_known_issues_from_matched_bugs(self):
        traits = collect_traits(_make_checks(bug_ids=["MLX-001", "MLX-003"]))
        assert traits["known_issues"] == ["MLX-001", "MLX-003"]

    def test_vlm_model_traits(self):
        traits = collect_traits(
            _make_checks(
                is_vlm=True,
                architecture="Qwen3_5ForConditionalGeneration",
                model_type="qwen3_5",
            )
        )
        assert traits["is_vlm"] is True
        assert traits["architecture_type"] == "Qwen3_5ForConditionalGeneration"

    def test_check_with_empty_metadata(self):
        """Check 0.2 present but with empty metadata (e.g. CRITICAL return path)."""
        checks = [
            DiagnosticResult(
                check_id="0.2",
                name="Tokenizer Config",
                status=CheckStatus.CRITICAL,
                detail="No eos_token",
            ),
        ]
        traits = collect_traits(checks)
        assert traits["has_chat_template"] is None
        assert traits["has_thinking_tokens"] is None


class TestReasoningMechanism:
    """Reasoning mechanism classification."""

    def test_thinking_tokens_yields_think_tag(self):
        traits = collect_traits(
            _make_checks(
                has_thinking_tokens=True,
                model_type="qwen3",
                architecture="Qwen3ForCausalLM",
            )
        )
        assert traits["reasoning_mechanism"] == "think_tag"

    def test_known_non_reasoning_family_yields_none(self):
        traits = collect_traits(_make_checks(model_type="llama"))
        assert traits["reasoning_mechanism"] == "none"

    def test_gemma_non_reasoning(self):
        traits = collect_traits(
            _make_checks(model_type="gemma3", architecture="Gemma3ForCausalLM")
        )
        assert traits["reasoning_mechanism"] == "none"

    def test_mistral3_non_reasoning(self):
        traits = collect_traits(
            _make_checks(model_type="mistral3", architecture="Mistral3ForCausalLM")
        )
        assert traits["reasoning_mechanism"] == "none"

    def test_unknown_family_no_thinking_yields_unknown(self):
        traits = collect_traits(
            _make_checks(model_type="some_new_arch", architecture="SomeNewForCausalLM")
        )
        assert traits["reasoning_mechanism"] == "unknown"

    def test_missing_tokenizer_check_yields_unknown(self):
        checks = [
            DiagnosticResult(
                check_id="0.5",
                name="Arch",
                status=CheckStatus.PASS,
                detail="OK",
                metadata={"is_vlm": False, "architecture": "LlamaForCausalLM"},
            ),
        ]
        traits = collect_traits(checks)
        assert traits["reasoning_mechanism"] == "unknown"
