"""Check 0.5: Model architecture detection (VLM vs text-only).

Detects Vision-Language Model (VLM) architectures that cannot be loaded by
mlx-lm's text-only model loader. VLMs contain vision_tower weights that
mlx-lm rejects, causing a hard crash instead of a clean diagnostic.

Evidence basis: Qwen3.5-4B-4bit (Qwen3_5ForConditionalGeneration) crashes
mlx-lm with "Received 297 parameters not in model" — all vision_tower.* weights.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_triage.models import CheckStatus, DiagnosticResult

CHECK_ID = "0.5"
CHECK_NAME = "Architecture Compatibility"

# Keys in config.json that indicate a VLM architecture
_VLM_CONFIG_KEYS = {"vision_config", "vision_tower_config", "visual"}


def check_architecture(model_path: str) -> DiagnosticResult:
    """Check whether model architecture is compatible with text-only inference."""
    path = Path(model_path)
    config_path = path / "config.json"

    if not config_path.exists():
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail="config.json not found in model directory.",
        )

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail=f"Failed to parse config.json: {exc}",
        )

    architectures = config.get("architectures", [])
    architecture = architectures[0] if architectures else "unknown"

    # Detect VLM indicators
    vlm_indicators: list[str] = []
    for key in _VLM_CONFIG_KEYS:
        if key in config:
            vlm_indicators.append(key)

    is_vlm = len(vlm_indicators) > 0

    if is_vlm:
        indicator_str = ", ".join(sorted(vlm_indicators))
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail=(
                f"Vision-Language Model detected ({architecture}). "
                f"VLM config keys found: {indicator_str}. "
                f"mlx-lm text-only loader cannot load vision tower weights."
            ),
            remediation=(
                "Use mlx-vlm for VLM inference, or use a text-only variant of this model."
            ),
            metadata={
                "is_vlm": True,
                "architecture": architecture,
                "vlm_indicators": vlm_indicators,
            },
        )

    return DiagnosticResult(
        check_id=CHECK_ID,
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        detail=f"Text-only architecture ({architecture}). Compatible with mlx-lm.",
        metadata={
            "is_vlm": False,
            "architecture": architecture,
        },
    )
