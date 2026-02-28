# src/mlx_triage/utils/mlx_utils.py
"""MLX model loading and generation utilities.

Wraps mlx_lm to provide a consistent interface for diagnostic checks.
All MLX imports are deferred so the module can be imported without MLX installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator


@dataclass
class GenerationResult:
    """Result of a single text generation including token-level data."""

    text: str
    tokens: list[int]
    logprobs: list[float] = field(default_factory=list)
    generation_tps: float = 0.0


def check_mlx_available() -> bool:
    """Check if MLX and mlx-lm are installed and importable."""
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def load_model(model_path: str) -> tuple:
    """Load an MLX model and tokenizer.

    Returns:
        (model, tokenizer) tuple.

    Raises:
        ImportError: If mlx-lm is not installed.
    """
    from mlx_lm import load

    return load(model_path)


def _stream_generate(model, tokenizer, prompt: str, **kwargs) -> Generator:
    """Thin wrapper around mlx_lm.stream_generate for mockability."""
    from mlx_lm import stream_generate

    return stream_generate(model, tokenizer, prompt, **kwargs)


def generate_text(
    model,
    tokenizer,
    prompt: str | list[dict],
    max_tokens: int = 256,
    temp: float = 0.0,
    seed: int | None = None,
) -> GenerationResult:
    """Generate text and collect token-level information.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        prompt: Text string or list of chat messages.
        max_tokens: Maximum tokens to generate.
        temp: Sampling temperature (0.0 = deterministic argmax).
        seed: Random seed for reproducibility.

    Returns:
        GenerationResult with text, tokens, logprobs, and TPS.
    """
    # Apply chat template if messages format
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    kwargs: dict = {"max_tokens": max_tokens, "temp": temp}
    if seed is not None:
        kwargs["seed"] = seed

    tokens: list[int] = []
    logprobs: list[float] = []
    text_parts: list[str] = []
    tps = 0.0

    for response in _stream_generate(model, tokenizer, prompt, **kwargs):
        tokens.append(response.token)
        if response.logprobs is not None:
            # Extract the log-probability for the sampled token.
            # In real mlx_lm, logprobs is an mx.array of shape (vocab_size,).
            # We handle scalars (.item()), indexable arrays, and plain floats.
            lp = response.logprobs
            if hasattr(lp, "item"):
                logprobs.append(float(lp.item()))
            elif hasattr(lp, "__getitem__"):
                try:
                    logprobs.append(float(lp[response.token]))
                except (IndexError, KeyError):
                    # Fallback: single-element sequence or pre-extracted value
                    logprobs.append(float(lp[0]))
            else:
                logprobs.append(float(lp))
        text_parts.append(response.text)
        tps = response.generation_tps

    return GenerationResult(
        text="".join(text_parts),
        tokens=tokens,
        logprobs=logprobs if logprobs else [],
        generation_tps=tps,
    )
