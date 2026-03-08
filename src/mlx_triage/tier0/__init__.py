"""Tier 0: Sanity Checks — no inference required, < 30 seconds."""

from __future__ import annotations

from mlx_triage.models import TierReport
from mlx_triage.tier0.architecture_check import check_architecture
from mlx_triage.tier0.dtype_check import check_dtype_compatibility
from mlx_triage.tier0.tokenizer_check import check_tokenizer_config
from mlx_triage.tier0.version_check import check_mlx_version
from mlx_triage.tier0.weight_integrity import check_weight_integrity


def run_tier0(model_path: str) -> TierReport:
    """Run all Tier 0 checks and return a structured report."""
    checks = [
        check_dtype_compatibility(model_path),
        check_tokenizer_config(model_path),
        check_weight_integrity(model_path),
        check_mlx_version(model_path),
        check_architecture(model_path),
    ]
    return TierReport.create(tier=0, model=model_path, checks=checks)
