"""Tier 1: Statistical Smoke Tests — requires MLX, loads model once."""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier1.determinism import check_determinism
from mlx_triage.tier1.quantization_quality import check_quantization_quality
from mlx_triage.tier1.reference_divergence import check_reference_divergence
from mlx_triage.utils.mlx_utils import check_mlx_available, load_model


def run_tier1(model_path: str) -> TierReport:
    """Run all Tier 1 checks.

    If MLX is not available, all checks are skipped.
    Currently each check loads the model independently; shared model
    loading is a planned optimisation.
    """
    if not check_mlx_available():
        return TierReport.create(
            tier=1,
            model=model_path,
            checks=[
                DiagnosticResult(
                    check_id="1.1",
                    name="Determinism",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
                DiagnosticResult(
                    check_id="1.2",
                    name="Reference Divergence",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
                DiagnosticResult(
                    check_id="1.3",
                    name="Quantization Quality",
                    status=CheckStatus.SKIP,
                    detail="MLX not available.",
                    remediation="uv sync --extra mlx",
                ),
            ],
        )

    checks = [
        check_determinism(model_path),
        check_reference_divergence(model_path),
        check_quantization_quality(model_path),
    ]

    return TierReport.create(tier=1, model=model_path, checks=checks)
