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

    # Build kwargs for stream_generate. mlx_lm >=0.30 uses a `sampler`
    # callable instead of a bare `temp` kwarg passed through to generate_step.
    from mlx_lm.sample_utils import make_sampler

    kwargs: dict = {"max_tokens": max_tokens, "sampler": make_sampler(temp=temp)}
    if seed is not None:
        import mlx.core as mx

        mx.random.seed(seed)

    tokens: list[int] = []
    logprobs: list[float] = []
    text_parts: list[str] = []
    tps = 0.0

    for response in _stream_generate(model, tokenizer, prompt, **kwargs):
        tokens.append(response.token)
        if response.logprobs is not None:
            # Extract the log-probability for the sampled token.
            # In mlx_lm >=0.30, logprobs is an mx.array of shape (vocab_size,).
            # We also handle test mocks that pass plain lists or floats.
            lp = response.logprobs
            try:
                # Check if this is an mx.array (has .size attribute as int/property)
                is_mx = hasattr(lp, "size") and hasattr(lp, "item")
                if is_mx:
                    size = lp.size if not callable(lp.size) else lp.size()
                    if size == 1:
                        # Scalar mx.array
                        logprobs.append(float(lp.item()))
                    else:
                        # Full vocab logprobs — index by token ID
                        logprobs.append(float(lp[response.token].item()))
                elif isinstance(lp, (int, float)):
                    logprobs.append(float(lp))
                elif isinstance(lp, (list, tuple)):
                    # Test mock: list of pre-extracted values
                    if len(lp) == 1:
                        logprobs.append(float(lp[0]))
                    elif response.token < len(lp):
                        logprobs.append(float(lp[response.token]))
                    else:
                        logprobs.append(float(lp[0]))
                elif hasattr(lp, "item"):
                    logprobs.append(float(lp.item()))
                else:
                    logprobs.append(float(lp))
            except (IndexError, KeyError, ValueError, TypeError):
                pass  # Skip logprob if extraction fails
        text_parts.append(response.text)
        tps = response.generation_tps

    return GenerationResult(
        text="".join(text_parts),
        tokens=tokens,
        logprobs=logprobs if logprobs else [],
        generation_tps=tps,
    )
