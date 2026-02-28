# tests/test_prompts.py
from mlx_triage.prompts.standard_suite import (
    DIAGNOSTIC_PROMPTS,
    get_prompt,
    get_all_prompts,
    EVAL_CORPUS,
)


def test_diagnostic_prompts_exist():
    assert len(DIAGNOSTIC_PROMPTS) >= 10


def test_prompt_categories():
    categories = {p["category"] for p in DIAGNOSTIC_PROMPTS}
    assert "math" in categories
    assert "code" in categories
    assert "structured" in categories


def test_each_prompt_has_required_fields():
    for prompt in DIAGNOSTIC_PROMPTS:
        assert "id" in prompt
        assert "category" in prompt
        assert "prompt" in prompt
        assert "max_tokens" in prompt


def test_get_prompt_by_id():
    prompt = get_prompt("math_basic")
    assert prompt is not None
    assert "2+2" in prompt["prompt"] or "math" in prompt["category"]


def test_get_all_prompts_returns_list():
    prompts = get_all_prompts()
    assert isinstance(prompts, list)
    assert len(prompts) == len(DIAGNOSTIC_PROMPTS)


def test_multi_turn_prompt():
    prompt = get_prompt("multi_turn")
    assert prompt is not None
    assert isinstance(prompt["prompt"], list)  # Chat messages format


def test_eval_corpus_exists():
    assert isinstance(EVAL_CORPUS, str)
    assert len(EVAL_CORPUS) >= 500  # At least 500 chars of evaluation text
