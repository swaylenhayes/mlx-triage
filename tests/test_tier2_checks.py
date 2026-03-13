"""Tests for Tier 2 checks."""

from unittest.mock import MagicMock

from mlx_triage.models import CheckStatus
from mlx_triage.tier2.batch_invariance import check_batch_invariance
from mlx_triage.tier2.context_length import check_context_length
from mlx_triage.tier2.memory_pressure import check_memory_pressure
from mlx_triage.utils.mlx_utils import GenerationResult


class _Tokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _BackendWithoutBatch:
    def is_available(self) -> bool:
        return True


class _BackendWithBatch:
    def __init__(
        self,
        single_results: list[GenerationResult],
        batch_results: list[GenerationResult],
    ) -> None:
        self._single_results = single_results
        self._batch_results = batch_results

    def is_available(self) -> bool:
        return True

    def generate_text(self, *_args, **_kwargs) -> GenerationResult:
        return self._single_results.pop(0)

    def generate_batch(self, *_args, **_kwargs) -> list[GenerationResult]:
        return self._batch_results


def _result(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens)


def test_batch_invariance_skips_without_batch_api():
    result = check_batch_invariance("/fake/model", backend=_BackendWithoutBatch())
    assert result.status == CheckStatus.SKIP
    assert "batched generation" in result.detail


def test_batch_invariance_passes_when_single_and_batch_match():
    backend = _BackendWithBatch(
        single_results=[
            _result("alpha", [1, 2, 3]),
            _result("beta", [4, 5, 6]),
        ],
        batch_results=[
            _result("alpha", [1, 2, 3]),
            _result("beta", [4, 5, 6]),
        ],
    )

    result = check_batch_invariance(
        "/fake/model",
        n_prompts=2,
        model=object(),
        tokenizer=object(),
        backend=backend,
    )

    assert result.status == CheckStatus.PASS
    assert result.metadata["avg_agreement"] == 1.0


def test_batch_invariance_warns_on_moderate_divergence():
    mostly_stable = list(range(20))
    slightly_shifted = mostly_stable[:-1] + [99]

    backend = _BackendWithBatch(
        single_results=[
            _result("alpha", [1, 2, 3]),
            _result("beta", mostly_stable),
        ],
        batch_results=[
            _result("alpha", [1, 2, 3]),
            _result("beta-ish", slightly_shifted),
        ],
    )

    result = check_batch_invariance(
        "/fake/model",
        n_prompts=2,
        model=object(),
        tokenizer=object(),
        backend=backend,
    )

    assert result.status == CheckStatus.WARNING
    assert result.metadata["divergence_rate"] > 0.02


def test_batch_invariance_fails_on_major_divergence():
    backend = _BackendWithBatch(
        single_results=[
            _result("alpha", [1, 2, 3]),
            _result("beta", [4, 5, 6]),
        ],
        batch_results=[
            _result("alpha", [9, 9, 9]),
            _result("beta-ish", [8, 8, 8]),
        ],
    )

    result = check_batch_invariance(
        "/fake/model",
        n_prompts=2,
        model=object(),
        tokenizer=object(),
        backend=backend,
    )

    assert result.status == CheckStatus.FAIL
    assert result.metadata["divergence_rate"] > 0.05


def test_memory_pressure_passes_when_reference_stable():
    backend = MagicMock()
    backend.is_available.return_value = True
    backend.generate_text.side_effect = [
        _result("apple\nbanana\npear", [1, 2, 3]),
        _result("pressure", [9]),
        _result("apple\nbanana\npear", [1, 2, 3]),
        _result("pressure", [9]),
        _result("apple\nbanana\npear", [1, 2, 3]),
    ]

    result = check_memory_pressure(
        "/fake/model",
        pressure_lengths=(16, 32),
        model=object(),
        tokenizer=object(),
        backend=backend,
        read_memory_stats=lambda: {"active_memory": 1, "peak_memory": 2},
    )

    assert result.status == CheckStatus.PASS
    assert len(result.metadata["points"]) == 2


def test_memory_pressure_fails_on_progressive_degradation():
    backend = MagicMock()
    backend.is_available.return_value = True
    backend.generate_text.side_effect = [
        _result("apple\nbanana\npear", [1, 2, 3]),
        _result("pressure", [9]),
        _result("apple\nbanana\nx", [1, 2, 9]),
        _result("pressure", [9]),
        _result("apple\nx\nx", [1, 9, 9]),
    ]

    result = check_memory_pressure(
        "/fake/model",
        pressure_lengths=(16, 32),
        model=object(),
        tokenizer=object(),
        backend=backend,
        read_memory_stats=lambda: {"active_memory": 1, "peak_memory": 2},
    )

    assert result.status == CheckStatus.FAIL
    assert result.metadata["min_agreement"] < 0.7


def test_context_length_passes_when_all_needles_retrieved():
    backend = MagicMock()
    backend.is_available.return_value = True
    backend.generate_text.side_effect = [
        _result("code-32-25", [1]),
        _result("code-32-75", [1]),
        _result("code-64-25", [1]),
        _result("code-64-75", [1]),
    ]

    result = check_context_length(
        "/fake/model",
        context_lengths=(32, 64),
        positions=(0.25, 0.75),
        model=object(),
        tokenizer=_Tokenizer(),
        backend=backend,
    )

    assert result.status == CheckStatus.PASS
    assert result.metadata["accuracies"][32] == 1.0
    assert result.metadata["accuracies"][64] == 1.0


def test_context_length_fails_on_cliff_drop():
    backend = MagicMock()
    backend.is_available.return_value = True
    backend.generate_text.side_effect = [
        _result("code-32-25", [1]),
        _result("code-32-75", [1]),
        _result("wrong", [1]),
        _result("still wrong", [1]),
    ]

    result = check_context_length(
        "/fake/model",
        context_lengths=(32, 64),
        positions=(0.25, 0.75),
        model=object(),
        tokenizer=_Tokenizer(),
        backend=backend,
    )

    assert result.status == CheckStatus.FAIL
    assert result.metadata["accuracies"][64] == 0.0
