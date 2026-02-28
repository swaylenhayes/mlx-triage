# tests/test_comparison.py
from mlx_triage.utils.comparison import (
    token_agreement_rate,
    divergence_point,
    multi_run_consistency,
)


def test_perfect_agreement():
    assert token_agreement_rate([1, 2, 3], [1, 2, 3]) == 1.0


def test_zero_agreement():
    assert token_agreement_rate([1, 2, 3], [4, 5, 6]) == 0.0


def test_partial_agreement():
    rate = token_agreement_rate([1, 2, 3, 4], [1, 2, 9, 9])
    assert rate == 0.5


def test_different_lengths():
    # Agreement rate computed over the shorter sequence
    rate = token_agreement_rate([1, 2, 3], [1, 2, 3, 4, 5])
    assert rate == 1.0


def test_empty_sequences():
    assert token_agreement_rate([], []) == 0.0
    assert token_agreement_rate([], [1, 2]) == 0.0


def test_divergence_point_identical():
    assert divergence_point([1, 2, 3], [1, 2, 3]) is None


def test_divergence_point_middle():
    assert divergence_point([1, 2, 3], [1, 9, 3]) == 1


def test_divergence_point_first():
    assert divergence_point([1, 2, 3], [9, 2, 3]) == 0


def test_divergence_point_length_diff():
    assert divergence_point([1, 2], [1, 2, 3]) == 2


def test_multi_run_consistency_perfect():
    runs = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    result = multi_run_consistency(runs)
    assert result["consistent"] is True
    assert result["agreement_rate"] == 1.0


def test_multi_run_consistency_mixed():
    runs = [[1, 2, 3], [1, 2, 3], [1, 9, 3]]
    result = multi_run_consistency(runs)
    assert result["consistent"] is False
    assert result["min_agreement"] < 1.0


def test_multi_run_consistency_single():
    result = multi_run_consistency([[1, 2, 3]])
    assert result["consistent"] is True
