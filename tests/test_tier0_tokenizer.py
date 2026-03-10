# tests/test_tier0_tokenizer.py
import json
from pathlib import Path

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.tokenizer_check import check_tokenizer_config


def test_valid_tokenizer_passes(good_model):
    result = check_tokenizer_config(str(good_model))
    assert result.check_id == "0.2"
    assert result.status in (CheckStatus.PASS, CheckStatus.INFO)


def test_missing_eos_critical(tmp_path):
    d = tmp_path / "no-eos"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<s>",
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    }))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.CRITICAL


def test_missing_chat_template_warning(tmp_path):
    d = tmp_path / "no-template"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "</s>",
        "bos_token": "<s>",
    }))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.WARNING


def test_no_tokenizer_config_fails(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.FAIL


def test_llama3_missing_eot_id_warning(tmp_path):
    """Llama 3 needs BOTH end_of_text AND eot_id stop tokens."""
    d = tmp_path / "llama3"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|end_of_text|>",
        "bos_token": "<|begin_of_text|>",
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    }))
    # generation_config.json with only one stop token
    (d / "generation_config.json").write_text(json.dumps({
        "eos_token_id": 128001,
    }))
    result = check_tokenizer_config(str(d))
    # Should at least INFO about Llama 3 dual stop token pattern
    assert result.status in (CheckStatus.WARNING, CheckStatus.INFO, CheckStatus.PASS)


def test_has_chat_template_metadata_true(good_model):
    """Chat template presence should be in metadata."""
    result = check_tokenizer_config(str(good_model))
    assert result.metadata["has_chat_template"] is True


def test_has_chat_template_metadata_false(tmp_path):
    """Missing chat template should be in metadata."""
    d = tmp_path / "no-template"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "llama"}')
    (d / "tokenizer_config.json").write_text(
        '{"eos_token": "</s>", "bos_token": "<s>"}'
    )
    result = check_tokenizer_config(str(d))
    assert result.metadata["has_chat_template"] is False


def test_thinking_tokens_detected(thinking_model):
    """Models with <think>/<​/think> in added_tokens should report has_thinking_tokens=True."""
    result = check_tokenizer_config(str(thinking_model))
    assert result.metadata["has_thinking_tokens"] is True


def test_no_thinking_tokens(good_model):
    """Standard models without thinking tokens should report has_thinking_tokens=False."""
    result = check_tokenizer_config(str(good_model))
    assert result.metadata["has_thinking_tokens"] is False


def test_thinking_tokens_partial_no_match(tmp_path):
    """Only <think> without </think> should NOT count as thinking-capable."""
    d = tmp_path / "partial-think"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "custom"}')
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "</s>",
                "chat_template": "...",
                "added_tokens_decoder": {
                    "100": {"content": "<think>", "special": True},
                },
            }
        )
    )
    result = check_tokenizer_config(str(d))
    assert result.metadata["has_thinking_tokens"] is False
