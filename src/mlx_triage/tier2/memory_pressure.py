"""Test 2.2: Memory pressure sweep."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.utils.backends import ModelBackend
from mlx_triage.utils.comparison import token_agreement_rate

MINOR_DIVERGENCE_THRESHOLD = 0.02
MAJOR_DIVERGENCE_THRESHOLD = 0.05
REFERENCE_PROMPT = "List exactly 3 fruits, one per line, nothing else."
PRESSURE_PROMPT = (
    "Write a long numbered list of concise facts about the solar system, "
    "one fact per line."
)


def _read_memory_stats() -> dict[str, int | None]:
    """Read MLX Metal memory stats when available."""
    try:
        import mlx.core as mx

        return {
            "active_memory": int(mx.metal.get_active_memory()),
            "peak_memory": int(mx.metal.get_peak_memory()),
        }
    except (AttributeError, ImportError, TypeError, ValueError):
        return {"active_memory": None, "peak_memory": None}


def _is_progressive_degradation(agreements: Sequence[float]) -> bool:
    """True when agreement degrades monotonically and meaningfully."""
    if len(agreements) < 2:
        return False
    if agreements[0] - agreements[-1] <= MAJOR_DIVERGENCE_THRESHOLD:
        return False
    return all(curr <= prev for prev, curr in zip(agreements, agreements[1:]))


def check_memory_pressure(
    model_path: str,
    pressure_lengths: Sequence[int] = (256, 512, 1024, 2048),
    reference_max_tokens: int = 32,
    seed: int = 42,
    model: object | None = None,
    tokenizer: object | None = None,
    backend: ModelBackend | None = None,
    read_memory_stats: Callable[[], dict[str, int | None]] = _read_memory_stats,
) -> DiagnosticResult:
    """Detect quality degradation that correlates with memory pressure."""
    if backend is None:
        from mlx_triage.utils.mlx_utils import MLXLMBackend

        backend = MLXLMBackend()

    if not backend.is_available():
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run memory pressure sweep.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    if model is None or tokenizer is None:
        model, tokenizer = backend.load(model_path)

    try:
        baseline = backend.generate_text(
            model,
            tokenizer,
            REFERENCE_PROMPT,
            max_tokens=reference_max_tokens,
            temp=0.0,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.CRITICAL,
            detail=f"Baseline generation failed before pressure sweep: {exc}",
            remediation="Verify Tier 1 generation works before running Tier 2.",
        )

    if not baseline.tokens:
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.CRITICAL,
            detail="Baseline reference prompt produced empty output.",
            remediation="Fix generation stability issues before running memory pressure diagnostics.",
        )

    points: list[dict[str, int | float | None]] = []

    for pressure_length in pressure_lengths:
        try:
            backend.generate_text(
                model,
                tokenizer,
                PRESSURE_PROMPT,
                max_tokens=pressure_length,
                temp=0.0,
                seed=seed,
            )
            reference = backend.generate_text(
                model,
                tokenizer,
                REFERENCE_PROMPT,
                max_tokens=reference_max_tokens,
                temp=0.0,
                seed=seed,
            )
        except Exception as exc:
            return DiagnosticResult(
                check_id="2.2",
                name="Memory Pressure Sweep",
                status=CheckStatus.CRITICAL,
                detail=(
                    "Generation failed under memory pressure at "
                    f"{pressure_length} tokens: {exc}"
                ),
                remediation="Investigate MLX memory pressure, KV cache behavior, and cache limits.",
                metadata={"failed_at_tokens": pressure_length, "points": points},
            )

        agreement = token_agreement_rate(baseline.tokens, reference.tokens)
        point: dict[str, int | float | None] = {
            "pressure_tokens": pressure_length,
            "agreement_rate": agreement,
        }
        point.update(read_memory_stats())
        points.append(point)

    agreements = [float(point["agreement_rate"]) for point in points]
    min_agreement = min(agreements)

    if all(agreement == 1.0 for agreement in agreements):
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.PASS,
            detail=(
                "Reference output remained identical across all memory pressure levels."
            ),
            metadata={"points": points},
        )

    if _is_progressive_degradation(agreements):
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.FAIL,
            detail=(
                "Reference output degraded progressively as memory pressure increased "
                f"(min agreement {min_agreement:.1%})."
            ),
            remediation="Investigate KV cache behavior, cache limits, and MLX memory pressure handling.",
            metadata={"points": points, "min_agreement": min_agreement},
        )

    if min_agreement <= 1.0 - MAJOR_DIVERGENCE_THRESHOLD:
        first_drop = next(
            point["pressure_tokens"]
            for point in points
            if float(point["agreement_rate"]) <= 1.0 - MAJOR_DIVERGENCE_THRESHOLD
        )
        return DiagnosticResult(
            check_id="2.2",
            name="Memory Pressure Sweep",
            status=CheckStatus.WARNING,
            detail=(
                "Reference output diverged under memory pressure "
                f"(first notable drop at {first_drop} tokens, min agreement {min_agreement:.1%})."
            ),
            remediation="Check for cache pressure, long-generation instability, and memory headroom.",
            metadata={"points": points, "min_agreement": min_agreement},
        )

    return DiagnosticResult(
        check_id="2.2",
        name="Memory Pressure Sweep",
        status=CheckStatus.INFO,
        detail=(
            "Minor variance observed under memory pressure "
            f"(min agreement {min_agreement:.1%})."
        ),
        metadata={"points": points, "min_agreement": min_agreement},
    )
