"""Tier 2: Isolation Tests — requires MLX, isolates runtime-driven failures."""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier2.batch_invariance import check_batch_invariance
from mlx_triage.tier2.context_length import check_context_length
from mlx_triage.tier2.memory_pressure import check_memory_pressure
from mlx_triage.utils.backends import ModelBackend, get_backend


def run_tier2(model_path: str, backend: ModelBackend | None = None) -> TierReport:
    """Run all Tier 2 isolation checks.

    Tier 2 shares a single model load across its checks, like Tier 1.
    When MLX is unavailable, all Tier 2 checks are skipped.
    """
    if backend is None:
        backend = get_backend(model_path)

    if not backend.is_available():
        return TierReport.create(
            tier=2,
            model=model_path,
            checks=[
                DiagnosticResult(
                    check_id="2.1",
                    name="Batch Invariance",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
                DiagnosticResult(
                    check_id="2.2",
                    name="Memory Pressure Sweep",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
                DiagnosticResult(
                    check_id="2.3",
                    name="Context Length Stress",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
            ],
        )

    model, tokenizer = backend.load(model_path)

    checks = [
        check_batch_invariance(
            model_path,
            model=model,
            tokenizer=tokenizer,
            backend=backend,
        ),
        check_memory_pressure(
            model_path,
            model=model,
            tokenizer=tokenizer,
            backend=backend,
        ),
        check_context_length(
            model_path,
            model=model,
            tokenizer=tokenizer,
            backend=backend,
        ),
    ]

    return TierReport.create(tier=2, model=model_path, checks=checks)
