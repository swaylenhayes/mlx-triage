"""Model backend abstraction for mlx-triage.

Defines the ModelBackend protocol and a factory function that auto-selects
the appropriate backend based on model architecture (text-only vs VLM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mlx_triage.utils.mlx_utils import GenerationResult


class BackendUnavailable(Exception):
    """Raised when the required model backend is not installed."""


@runtime_checkable
class ModelBackend(Protocol):
    """Interface that both mlx-lm and mlx-vlm backends satisfy."""

    def is_available(self) -> bool: ...

    def load(self, model_path: str) -> tuple[Any, Any]: ...

    def generate_text(
        self, model: Any, tokenizer: Any, prompt: str | list[dict], **kwargs: Any
    ) -> GenerationResult: ...

    def generate_batch(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str | list[dict]],
        **kwargs: Any,
    ) -> list[GenerationResult]: ...

    def compute_perplexity(self, model: Any, tokenizer: Any, text: str) -> float: ...


# Keys in config.json that indicate a VLM architecture.
_VLM_CONFIG_KEYS = {"vision_config", "vision_tower_config", "visual"}


def _read_config(model_path: str) -> dict:
    """Read config.json from model directory."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _is_vlm(config: dict) -> bool:
    """Check if config indicates a VLM architecture."""
    return bool(_VLM_CONFIG_KEYS & config.keys())


def get_backend(model_path: str) -> ModelBackend:
    """Select the appropriate backend based on model architecture."""
    config = _read_config(model_path)
    if _is_vlm(config):
        raise BackendUnavailable(
            "VLM model detected but mlx-vlm is not installed. "
            "Install VLM support: uv sync --extra vlm"
        )
    from mlx_triage.utils.mlx_utils import MLXLMBackend

    return MLXLMBackend()
