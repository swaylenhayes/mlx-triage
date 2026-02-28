# src/mlx_triage/utils/comparison.py
"""Token and text comparison utilities for diagnostic checks."""

from __future__ import annotations


def token_agreement_rate(tokens_a: list[int], tokens_b: list[int]) -> float:
    """Compute token-by-token agreement rate between two sequences.

    Compares over the shorter sequence length.
    Returns 0.0 for empty sequences.
    """
    if not tokens_a or not tokens_b:
        return 0.0
    min_len = min(len(tokens_a), len(tokens_b))
    matches = sum(1 for a, b in zip(tokens_a[:min_len], tokens_b[:min_len]) if a == b)
    return matches / min_len


def divergence_point(tokens_a: list[int], tokens_b: list[int]) -> int | None:
    """Find the first position where two token sequences diverge.

    Returns None if sequences are identical (including length).
    Returns the shorter length if one is a prefix of the other.
    """
    for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
        if a != b:
            return i
    if len(tokens_a) != len(tokens_b):
        return min(len(tokens_a), len(tokens_b))
    return None


def multi_run_consistency(runs: list[list[int]]) -> dict:
    """Analyze token consistency across multiple generation runs.

    Compares all runs against the first run (reference).

    Returns:
        Dict with: consistent (bool), agreement_rate (float),
        min_agreement, max_agreement, num_runs.
    """
    if len(runs) < 2:
        return {"consistent": True, "agreement_rate": 1.0, "num_runs": len(runs)}

    reference = runs[0]
    agreements = [token_agreement_rate(reference, run) for run in runs[1:]]

    return {
        "consistent": all(a == 1.0 for a in agreements),
        "agreement_rate": sum(agreements) / len(agreements),
        "min_agreement": min(agreements),
        "max_agreement": max(agreements),
        "num_runs": len(runs),
    }
