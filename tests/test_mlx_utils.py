# tests/test_mlx_utils.py
"""Tests for MLX inference utilities. Mocks MLX to avoid hardware dependency."""
import sys
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from mlx_triage.utils.mlx_utils import (
    GenerationResult,
    check_mlx_available,
    generate_batch,
    generate_text,
    load_model,
)


def test_check_mlx_available_when_missing():
    with patch.dict("sys.modules", {"mlx": None, "mlx_lm": None}):
        assert check_mlx_available() is False


def test_generation_result_dataclass():
    result = GenerationResult(
        text="Hello world",
        tokens=[1, 2, 3],
        logprobs=[-0.5, -0.3, -0.1],
        generation_tps=50.0,
    )
    assert result.text == "Hello world"
    assert len(result.tokens) == 3
    assert result.logprobs is not None


def _mock_mlx_lm_modules():
    """Create mock mlx_lm modules so `from mlx_lm.sample_utils import make_sampler` succeeds."""
    mock_sample_utils = MagicMock()
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.sample_utils = mock_sample_utils
    return {
        "mlx_lm": mock_mlx_lm,
        "mlx_lm.sample_utils": mock_sample_utils,
    }


def test_generate_text_collects_tokens():
    """Verify generate_text collects tokens and logprobs from stream."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    @dataclass
    class FakeResponse:
        text: str
        token: int
        logprobs: list
        generation_tps: float = 50.0
        finish_reason: str | None = None

    responses = [
        FakeResponse(text="Hello", token=100, logprobs=[-0.5]),
        FakeResponse(text=" world", token=200, logprobs=[-0.3]),
    ]

    with patch("mlx_triage.utils.mlx_utils._stream_generate", return_value=iter(responses)), \
         patch.dict("sys.modules", _mock_mlx_lm_modules()):
        result = generate_text(mock_model, mock_tokenizer, "test prompt")

    assert result.text == "Hello world"
    assert result.tokens == [100, 200]
    assert len(result.logprobs) == 2


def test_generate_text_handles_chat_messages():
    """Verify chat-format prompts get template applied."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    @dataclass
    class FakeResponse:
        text: str
        token: int
        logprobs: list
        generation_tps: float = 50.0
        finish_reason: str | None = None

    responses = [FakeResponse(text="Hi", token=1, logprobs=[-0.1])]

    messages = [{"role": "user", "content": "Hello"}]

    with patch("mlx_triage.utils.mlx_utils._stream_generate", return_value=iter(responses)), \
         patch.dict("sys.modules", _mock_mlx_lm_modules()):
        result = generate_text(mock_model, mock_tokenizer, messages)

    mock_tokenizer.apply_chat_template.assert_called_once()
    assert result.text == "Hi"


def test_generate_batch_collects_texts_and_tokens():
    """Verify generate_batch returns GenerationResult objects per prompt."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [
        len(text),
        len(text) + 1,
    ]

    fake_response = type(
        "BatchResponse",
        (),
        {
            "texts": ["one", "two"],
            "stats": type("Stats", (), {"generation_tps": 42.0})(),
        },
    )()

    with patch(
        "mlx_triage.utils.mlx_utils._batch_generate", return_value=fake_response
    ), patch.dict("sys.modules", _mock_mlx_lm_modules()):
        results = generate_batch(mock_model, mock_tokenizer, ["prompt a", "prompt b"])

    assert len(results) == 2
    assert [result.text for result in results] == ["one", "two"]
    assert all(isinstance(result, GenerationResult) for result in results)
    assert results[0].tokens == [3, 4]
    assert results[0].generation_tps == 42.0


def test_generate_batch_handles_chat_messages():
    """Verify chat-format prompts are normalized before batched generation."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = [
        "formatted prompt 1",
        "formatted prompt 2",
    ]
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [
        len(text)
    ]

    fake_response = type(
        "BatchResponse",
        (),
        {
            "texts": ["reply 1", "reply 2"],
            "stats": type("Stats", (), {"generation_tps": 21.0})(),
        },
    )()

    with patch(
        "mlx_triage.utils.mlx_utils._batch_generate", return_value=fake_response
    ), patch.dict("sys.modules", _mock_mlx_lm_modules()):
        results = generate_batch(
            mock_model,
            mock_tokenizer,
            [
                [{"role": "user", "content": "hello"}],
                [{"role": "user", "content": "world"}],
            ],
        )

    assert [result.text for result in results] == ["reply 1", "reply 2"]
    assert mock_tokenizer.apply_chat_template.call_count == 2
