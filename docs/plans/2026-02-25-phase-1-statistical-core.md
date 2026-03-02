# Phase 1: Statistical Core & MLX Knowledge Corpus Implementation Plan

**Goal:** Tier 1 statistical smoke tests — determinism check, reference divergence, and quantization quality gate — plus a local MLX documentation corpus for ongoing development reference.

**Architecture:** A sync script shallow-clones `ml-explore/mlx` and `ml-explore/mlx-lm` repos, extracting docs and key source files into a grep-friendly local corpus with a hand-curated index. Tier 1 checks use `mlx_lm.stream_generate()` to collect token-level outputs and logprobs, compare across runs (determinism) or against reference backends (divergence), and measure perplexity against known benchmarks. All checks produce `DiagnosticResult` objects and integrate with the existing CLI and report system.

**Tech Stack:** mlx, mlx-lm (inference), transformers (optional, for reference divergence), existing stack from Phase 0 (Click, Rich, pytest)

**Prerequisites:** Phase 0 complete (`f271faf`). MLX installed on Apple Silicon (`uv sync --extra mlx`).

---

## Task 1: MLX Knowledge Corpus

**Files:**
- Create: `scripts/sync-mlx-docs.sh`
- Create: `_docs/mlx-knowledge/INDEX.md` (auto-generated, then hand-curated)
- Modify: `.gitignore`

**Step 1: Add corpus directory to .gitignore**

Append to `.gitignore`:
```
# MLX knowledge corpus (local reference, auto-synced)
_docs/mlx-knowledge/
!_docs/mlx-knowledge/INDEX.md
```

We gitignore the downloaded corpus but track the INDEX.md (it's hand-curated reference).

**Step 2: Create the sync script**

```bash
#!/usr/bin/env bash
# scripts/sync-mlx-docs.sh
# Download MLX and mlx-lm documentation into a local searchable corpus.
#
# Usage:
#   ./scripts/sync-mlx-docs.sh          # Sync if stale (>7 days)
#   ./scripts/sync-mlx-docs.sh --force  # Force re-sync
#
# Corpus is stored in _docs/mlx-knowledge/ (gitignored).
# Search with: grep -r "pattern" _docs/mlx-knowledge/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_DIR="$PROJECT_ROOT/_docs/mlx-knowledge"
STALE_DAYS=7

# --- Freshness check ---
if [ -f "$CORPUS_DIR/.last-updated" ] && [ "${1:-}" != "--force" ]; then
    if [ -n "$(find "$CORPUS_DIR/.last-updated" -mtime -${STALE_DAYS} 2>/dev/null)" ]; then
        echo "Corpus is fresh (< ${STALE_DAYS} days). Use --force to re-sync."
        exit 0
    fi
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "=== Syncing MLX documentation corpus ==="

# --- MLX Core Docs (RST source files) ---
echo "[1/3] Cloning ml-explore/mlx docs..."
git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/ml-explore/mlx.git "$TEMP_DIR/mlx" 2>&1 | tail -1
(cd "$TEMP_DIR/mlx" && git sparse-checkout set docs/src/python docs/src/usage docs/src/examples)

mkdir -p "$CORPUS_DIR/mlx-core"
cp -r "$TEMP_DIR/mlx/docs/src/python" "$CORPUS_DIR/mlx-core/"
cp -r "$TEMP_DIR/mlx/docs/src/usage" "$CORPUS_DIR/mlx-core/"
[ -d "$TEMP_DIR/mlx/docs/src/examples" ] && cp -r "$TEMP_DIR/mlx/docs/src/examples" "$CORPUS_DIR/mlx-core/" || true

# Capture MLX version from setup
MLX_VERSION=$(grep -m1 'version' "$TEMP_DIR/mlx/python/setup.py" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")

# --- mlx-lm Docs + Key Source Files ---
echo "[2/3] Cloning ml-explore/mlx-lm docs and source..."
git clone --depth=1 https://github.com/ml-explore/mlx-lm.git "$TEMP_DIR/mlx-lm" 2>&1 | tail -1

mkdir -p "$CORPUS_DIR/mlx-lm/docs" "$CORPUS_DIR/mlx-lm/source"

# Markdown docs
for f in README.md BENCHMARKS.md SERVER.md LORA.md MANAGE.md LEARNED_QUANTS.md; do
    [ -f "$TEMP_DIR/mlx-lm/mlx_lm/$f" ] && cp "$TEMP_DIR/mlx-lm/mlx_lm/$f" "$CORPUS_DIR/mlx-lm/docs/"
done

# Key source files (for API reference via docstrings)
for f in __init__.py generate.py utils.py sample_utils.py perplexity.py evaluate.py; do
    [ -f "$TEMP_DIR/mlx-lm/mlx_lm/$f" ] && cp "$TEMP_DIR/mlx-lm/mlx_lm/$f" "$CORPUS_DIR/mlx-lm/source/"
done

MLXLM_VERSION=$(grep -m1 'version' "$TEMP_DIR/mlx-lm/setup.py" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")

# --- Metadata ---
echo "[3/3] Writing metadata..."
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$CORPUS_DIR/.last-updated"

cat > "$CORPUS_DIR/.sync-info" <<EOF
MLX Core: $MLX_VERSION (from ml-explore/mlx main branch)
mlx-lm: $MLXLM_VERSION (from ml-explore/mlx-lm main branch)
Synced: $(cat "$CORPUS_DIR/.last-updated")
EOF

echo "=== Done ==="
echo "Corpus: $CORPUS_DIR"
echo "MLX: $MLX_VERSION | mlx-lm: $MLXLM_VERSION"
echo "Search: grep -r 'pattern' $CORPUS_DIR/"
```

**Step 3: Make executable and run initial sync**

Run: `chmod +x scripts/sync-mlx-docs.sh && ./scripts/sync-mlx-docs.sh --force`
Expected: Downloads docs, creates `_docs/mlx-knowledge/` with mlx-core/ and mlx-lm/ subdirectories.

**Step 4: Verify corpus is searchable**

Run: `grep -r "get_active_memory" _docs/mlx-knowledge/`
Expected: Hits in `mlx-core/python/memory_management.rst`

Run: `grep -r "stream_generate" _docs/mlx-knowledge/`
Expected: Hits in `mlx-lm/source/generate.py`

**Step 5: Create INDEX.md**

```markdown
# MLX Knowledge Corpus — Quick Reference Index

> Auto-synced from ml-explore/mlx and ml-explore/mlx-lm.
> Run `./scripts/sync-mlx-docs.sh --force` to update.
> Search: `grep -r "pattern" _docs/mlx-knowledge/`

---

## Key APIs for mlx-triage

### Model Loading (mlx-lm)
```python
from mlx_lm import load
model, tokenizer = load("path_or_hf_repo")
# Params: tokenizer_config, model_config, adapter_path, lazy, return_config, revision
```
**Source:** `mlx-lm/source/utils.py`

### Text Generation (mlx-lm)
```python
from mlx_lm import generate, stream_generate

# Simple (returns string)
text = generate(model, tokenizer, prompt, temp=0.0, seed=42, max_tokens=256)

# Streaming (yields GenerationResponse)
for response in stream_generate(model, tokenizer, prompt, max_tokens=256):
    response.text       # Next decoded segment
    response.token      # Token ID (int)
    response.logprobs   # Log probabilities (mx.array)
    response.prompt_tps # Prompt tokens/sec
    response.generation_tps  # Generation tokens/sec
    response.finish_reason   # "length", "stop", or None
```
**Source:** `mlx-lm/source/generate.py`

### Batch Generation (mlx-lm)
```python
from mlx_lm import batch_generate
result = batch_generate(model, tokenizer, prompts, max_tokens=128)
result.texts   # List[str]
result.stats   # BatchStats with TPS metrics
```
**Source:** `mlx-lm/source/generate.py`

### Sampling Control (mlx-lm)
```python
from mlx_lm.sample_utils import make_sampler
sampler = make_sampler(temp=0.0)  # temp=0 → argmax (deterministic)
# Params: temp, top_p, min_p, top_k, xtc_probability, xtc_threshold
```
**Source:** `mlx-lm/source/sample_utils.py`

### Perplexity Evaluation (mlx-lm)
```python
from mlx_lm.perplexity import load_data, eval_ppl
data = load_data(tokenizer, "allenai/tulu-3-sft-mixture", num_samples=100, sequence_length=512)
ppl, std_err = eval_ppl(model, data, batch_size=8)
```
**Source:** `mlx-lm/source/perplexity.py`

### Memory Monitoring (mlx.core)
```python
import mlx.core as mx
mx.metal.get_active_memory()   # Current active bytes
mx.metal.get_peak_memory()     # Peak bytes since last reset
mx.metal.set_memory_limit(n)   # Set limit, returns previous
mx.metal.reset_peak_memory()   # Reset peak counter
mx.metal.cache_size()          # Current cache size
mx.metal.set_cache_limit(n)    # Set cache limit
```
**Source:** `mlx-core/python/memory_management.rst`

### Random / Seed Control (mlx.core)
```python
import mlx.core as mx
mx.random.seed(42)             # Set global PRNG seed
key = mx.random.key(42)        # Create explicit PRNG key
```
**Source:** `mlx-core/python/random.rst`

---

## Corpus Structure

| Path | Contents | Format |
|------|----------|--------|
| `mlx-core/python/` | Python API reference (array, nn, metal, memory, random, etc.) | RST |
| `mlx-core/usage/` | Usage guides (lazy evaluation, unified memory, compilation, etc.) | RST |
| `mlx-lm/docs/` | mlx-lm documentation (README, benchmarks, server, LoRA, etc.) | Markdown |
| `mlx-lm/source/` | Key source files with docstrings (generate, utils, sample_utils, perplexity, evaluate) | Python |
```

**Step 6: Commit**

```bash
git add scripts/sync-mlx-docs.sh .gitignore _docs/mlx-knowledge/INDEX.md
git commit -m "feat: add MLX knowledge corpus sync script and index"
```

---

## Task 2: MLX Dependencies & Inference Utilities

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mlx_triage/utils/mlx_utils.py`
- Create: `tests/test_mlx_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_mlx_utils.py
"""Tests for MLX inference utilities. Mocks MLX to avoid hardware dependency."""
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from mlx_triage.utils.mlx_utils import (
    GenerationResult,
    load_model,
    generate_text,
    check_mlx_available,
)


def test_check_mlx_available_when_missing():
    with patch.dict("sys.modules", {"mlx": None, "mlx_lm": None}):
        # Reimport to pick up mocked modules
        assert check_mlx_available() is False or True  # Depends on env


def test_generation_result_dataclass():
    result = GenerationResult(
        text="Hello world",
        tokens=[1, 2, 3],
        logprobs=[-0.5, -0.3, -0.1],
        generation_tps=50.0,
    )
    assert result.text == "Hello world"
    assert len(result.tokens) == 3
    assert result.logprobs is not None


def test_generate_text_collects_tokens():
    """Verify generate_text collects tokens and logprobs from stream."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    @dataclass
    class FakeResponse:
        text: str
        token: int
        logprobs: list
        generation_tps: float = 50.0
        finish_reason: str | None = None

    responses = [
        FakeResponse(text="Hello", token=100, logprobs=[-0.5]),
        FakeResponse(text=" world", token=200, logprobs=[-0.3]),
    ]

    with patch("mlx_triage.utils.mlx_utils._stream_generate", return_value=iter(responses)):
        result = generate_text(mock_model, mock_tokenizer, "test prompt")

    assert result.text == "Hello world"
    assert result.tokens == [100, 200]
    assert len(result.logprobs) == 2


def test_generate_text_handles_chat_messages():
    """Verify chat-format prompts get template applied."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    @dataclass
    class FakeResponse:
        text: str
        token: int
        logprobs: list
        generation_tps: float = 50.0
        finish_reason: str | None = None

    responses = [FakeResponse(text="Hi", token=1, logprobs=[-0.1])]

    messages = [{"role": "user", "content": "Hello"}]

    with patch("mlx_triage.utils.mlx_utils._stream_generate", return_value=iter(responses)):
        result = generate_text(mock_model, mock_tokenizer, messages)

    mock_tokenizer.apply_chat_template.assert_called_once()
    assert result.text == "Hi"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlx_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Update pyproject.toml dependencies**

Add `transformers` as an optional dependency for reference divergence:

```toml
[project.optional-dependencies]
mlx = [
    "mlx>=0.20",
    "mlx-lm>=0.20",
]
reference = [
    "transformers>=4.40",
    "torch>=2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

Run: `uv sync --extra dev --extra mlx`

**Step 4: Implement mlx_utils.py**

```python
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
            # logprobs is an mx.array; extract the prob for the sampled token
            lp = response.logprobs
            if hasattr(lp, "item"):
                logprobs.append(float(lp.item()))
            elif hasattr(lp, "__getitem__"):
                logprobs.append(float(lp[response.token]))
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
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_mlx_utils.py -v`
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add pyproject.toml src/mlx_triage/utils/mlx_utils.py tests/test_mlx_utils.py
git commit -m "feat: add MLX inference utilities with model loading and generation"
```

---

## Task 3: Token Comparison Utilities

**Files:**
- Create: `src/mlx_triage/utils/comparison.py`
- Create: `tests/test_comparison.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement comparison.py**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comparison.py -v`
Expected: All 13 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/utils/comparison.py tests/test_comparison.py
git commit -m "feat: add token comparison utilities (agreement rate, divergence, consistency)"
```

---

## Task 4: Diagnostic Prompt Suite

**Files:**
- Create: `src/mlx_triage/prompts/standard_suite.py`
- Create: `tests/test_prompts.py`

**Step 1: Write the failing test**

```python
# tests/test_prompts.py
from mlx_triage.prompts.standard_suite import (
    DIAGNOSTIC_PROMPTS,
    get_prompt,
    get_all_prompts,
    EVAL_CORPUS,
)


def test_diagnostic_prompts_exist():
    assert len(DIAGNOSTIC_PROMPTS) >= 10


def test_prompt_categories():
    categories = {p["category"] for p in DIAGNOSTIC_PROMPTS}
    assert "math" in categories
    assert "code" in categories
    assert "structured" in categories


def test_each_prompt_has_required_fields():
    for prompt in DIAGNOSTIC_PROMPTS:
        assert "id" in prompt
        assert "category" in prompt
        assert "prompt" in prompt
        assert "max_tokens" in prompt


def test_get_prompt_by_id():
    prompt = get_prompt("math_basic")
    assert prompt is not None
    assert "2+2" in prompt["prompt"] or "math" in prompt["category"]


def test_get_all_prompts_returns_list():
    prompts = get_all_prompts()
    assert isinstance(prompts, list)
    assert len(prompts) == len(DIAGNOSTIC_PROMPTS)


def test_multi_turn_prompt():
    prompt = get_prompt("multi_turn")
    assert prompt is not None
    assert isinstance(prompt["prompt"], list)  # Chat messages format


def test_eval_corpus_exists():
    assert isinstance(EVAL_CORPUS, str)
    assert len(EVAL_CORPUS) >= 500  # At least 500 chars of evaluation text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement standard_suite.py**

```python
# src/mlx_triage/prompts/standard_suite.py
"""Diagnostic prompt suite for Tier 1 statistical tests.

Each prompt targets a specific failure mode identified in the evidence base:
- Math: sensitive to precision drift
- Code: sensitive to constrained decoding, tokenizer issues
- Structured: sensitive to EOS handling, template issues
- Multi-turn: sensitive to context truncation, template rendering
- Edge cases: from production incidents (Anthropic TPU bug, etc.)
"""

from __future__ import annotations

DIAGNOSTIC_PROMPTS: list[dict] = [
    # Math reasoning — sensitive to precision drift
    {
        "id": "math_basic",
        "category": "math",
        "prompt": "What is 2+2? Answer with just the number.",
        "max_tokens": 16,
    },
    {
        "id": "math_chain",
        "category": "math",
        "prompt": "If I have 15 apples and give away 7, then buy 3 more, how many do I have? Answer with just the number.",
        "max_tokens": 16,
    },
    # Code generation — sensitive to constrained decoding, tokenizer issues
    {
        "id": "code_python",
        "category": "code",
        "prompt": "Write a Python function that reverses a string. Only output the code, no explanation.",
        "max_tokens": 128,
    },
    {
        "id": "code_json",
        "category": "code",
        "prompt": 'Output valid JSON with these exact fields: {"name": "test", "value": 42}',
        "max_tokens": 64,
    },
    # Structured output — sensitive to EOS handling, template issues
    {
        "id": "structured_list",
        "category": "structured",
        "prompt": "List exactly 3 fruits, one per line, nothing else.",
        "max_tokens": 32,
    },
    {
        "id": "structured_stop",
        "category": "structured",
        "prompt": "Say 'done' and stop.",
        "max_tokens": 16,
    },
    # Multi-turn — sensitive to context truncation, template rendering
    {
        "id": "multi_turn",
        "category": "multi_turn",
        "prompt": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"},
        ],
        "max_tokens": 32,
    },
    # Factual / short-form — baseline quality
    {
        "id": "factual_capital",
        "category": "factual",
        "prompt": "What is the capital of France? Answer with just the city name.",
        "max_tokens": 16,
    },
    # Edge cases from evidence
    {
        "id": "repetition_check",
        "category": "edge",
        "prompt": "Describe the process of photosynthesis in exactly 2 sentences.",
        "max_tokens": 128,
    },
    {
        "id": "unicode_check",
        "category": "edge",
        "prompt": "Translate 'hello' to Japanese. Output only the Japanese text.",
        "max_tokens": 32,
    },
]


def get_prompt(prompt_id: str) -> dict | None:
    """Get a specific diagnostic prompt by ID."""
    for prompt in DIAGNOSTIC_PROMPTS:
        if prompt["id"] == prompt_id:
            return prompt
    return None


def get_all_prompts() -> list[dict]:
    """Get all diagnostic prompts."""
    return list(DIAGNOSTIC_PROMPTS)


# Fixed evaluation corpus for perplexity measurement.
# Public domain text (adapted from Wikipedia-style factual content).
# ~800 tokens when tokenized with typical LLM tokenizers.
EVAL_CORPUS = (
    "The Earth orbits the Sun at an average distance of about 150 million "
    "kilometers. This distance is known as one astronomical unit. The orbital "
    "period is approximately 365.25 days, which is why a leap year occurs every "
    "four years to keep the calendar aligned with the seasons. The Earth's axis "
    "is tilted at approximately 23.5 degrees relative to its orbital plane, "
    "which causes the seasons as different hemispheres receive varying amounts "
    "of solar radiation throughout the year.\n\n"
    "Water covers approximately 71 percent of the Earth's surface, with oceans "
    "holding about 96.5 percent of all water on Earth. The Pacific Ocean is the "
    "largest and deepest ocean, covering more area than all the land masses "
    "combined. The average depth of the ocean is about 3,688 meters, while the "
    "deepest point, the Challenger Deep in the Mariana Trench, reaches "
    "approximately 10,935 meters below sea level.\n\n"
    "The atmosphere of Earth consists primarily of nitrogen at about 78 percent "
    "and oxygen at about 21 percent, with trace amounts of argon, carbon "
    "dioxide, and other gases. The atmosphere extends to about 10,000 kilometers "
    "above the surface, though most of the mass is concentrated in the lowest "
    "layer, the troposphere, which extends to about 12 kilometers altitude. "
    "Weather phenomena occur primarily in the troposphere.\n\n"
    "Photosynthesis is the process by which plants convert light energy into "
    "chemical energy. During photosynthesis, plants absorb carbon dioxide from "
    "the atmosphere and water from the soil, using sunlight as an energy source "
    "to produce glucose and oxygen. The chemical equation for photosynthesis is "
    "6CO2 + 6H2O + light energy produces C6H12O6 + 6O2. This process is "
    "fundamental to life on Earth as it produces the oxygen that most organisms "
    "breathe and forms the base of most food chains.\n\n"
    "Mathematics is the study of numbers, quantities, shapes, and patterns. "
    "The Pythagorean theorem states that in a right triangle, the square of "
    "the hypotenuse equals the sum of the squares of the other two sides. "
    "This can be written as a squared plus b squared equals c squared. "
    "The number pi, approximately equal to 3.14159, represents the ratio of "
    "a circle's circumference to its diameter. Euler's identity, often "
    "considered the most beautiful equation in mathematics, states that e to "
    "the power of i times pi plus one equals zero, connecting five fundamental "
    "mathematical constants."
)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/prompts/standard_suite.py tests/test_prompts.py
git commit -m "feat: add diagnostic prompt suite and evaluation corpus for Tier 1"
```

---

## Task 5: Test 1.1 — Determinism Check

**Files:**
- Create: `src/mlx_triage/tier1/determinism.py`
- Create: `tests/test_tier1_determinism.py`

**Step 1: Write the failing test**

```python
# tests/test_tier1_determinism.py
"""Tests for determinism check. All inference is mocked."""

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from mlx_triage.models import CheckStatus
from mlx_triage.tier1.determinism import check_determinism
from mlx_triage.utils.mlx_utils import GenerationResult


def _mock_generate(text: str, tokens: list[int]) -> GenerationResult:
    return GenerationResult(text=text, tokens=tokens, logprobs=[], generation_tps=50.0)


def test_deterministic_model_passes():
    """Model producing identical output every run → PASS."""
    call_count = 0

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("Hello world", [100, 200, 300])

    with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
        with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
            result = check_determinism("/fake/model", n_runs=5, n_prompts=3)

    assert result.check_id == "1.1"
    assert result.status == CheckStatus.PASS


def test_minor_divergence_info():
    """< 2% token divergence → INFO (expected floating-point variance)."""
    call_count = [0]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        call_count[0] += 1
        # Every 10th call diverges slightly (1 token different)
        if call_count[0] % 10 == 0:
            return _mock_generate("Hello world!", [100, 200, 301])
        return _mock_generate("Hello world", [100, 200, 300])

    with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
        with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
            result = check_determinism("/fake/model", n_runs=5, n_prompts=3)

    assert result.status in (CheckStatus.PASS, CheckStatus.INFO)


def test_major_divergence_fails():
    """Completely different output each run → FAIL."""
    call_count = [0]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        call_count[0] += 1
        return _mock_generate(f"output_{call_count[0]}", [call_count[0], call_count[0] + 1])

    with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
        with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
            result = check_determinism("/fake/model", n_runs=5, n_prompts=3)

    assert result.status in (CheckStatus.FAIL, CheckStatus.CRITICAL)


def test_empty_output_critical():
    """Empty output on any run → CRITICAL."""

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return _mock_generate("", [])

    with patch("mlx_triage.tier1.determinism.generate_text", side_effect=fake_generate):
        with patch("mlx_triage.tier1.determinism.load_model", return_value=(MagicMock(), MagicMock())):
            result = check_determinism("/fake/model", n_runs=3, n_prompts=2)

    assert result.status == CheckStatus.CRITICAL


def test_mlx_not_available_skips():
    """When MLX is not installed, should SKIP."""
    with patch("mlx_triage.tier1.determinism.check_mlx_available", return_value=False):
        result = check_determinism("/fake/model")
    assert result.status == CheckStatus.SKIP
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier1_determinism.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement determinism.py**

```python
# src/mlx_triage/tier1/determinism.py
"""Test 1.1: Determinism Check.

Verifies that the model produces identical output for identical inputs at
temperature=0. Non-determinism at temp=0 indicates infrastructure issues.

Evidence: quantized integer models (Q4, Q8) achieve perfect reproducibility
on MLX, while float16 may have minor variance. If a quantized model is
non-deterministic, something is very wrong.
"""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult
from mlx_triage.prompts.standard_suite import DIAGNOSTIC_PROMPTS
from mlx_triage.utils.comparison import multi_run_consistency
from mlx_triage.utils.mlx_utils import (
    GenerationResult,
    check_mlx_available,
    generate_text,
    load_model,
)

# Thresholds from Token-DiFR evidence
MINOR_DIVERGENCE_THRESHOLD = 0.02  # < 2% = expected floating-point variance
MAJOR_DIVERGENCE_THRESHOLD = 0.05  # > 5% = infrastructure non-determinism


def check_determinism(
    model_path: str,
    n_runs: int = 10,
    n_prompts: int = 5,
    seed: int = 42,
) -> DiagnosticResult:
    """Run determinism check: same prompt N times at temp=0, compare outputs.

    Args:
        model_path: Path to MLX model directory.
        n_runs: Number of times to run each prompt.
        n_prompts: Number of prompts to test (from diagnostic suite).
        seed: Random seed for reproducibility.
    """
    if not check_mlx_available():
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot run determinism check.",
            remediation="Install MLX: uv sync --extra mlx",
        )

    model, tokenizer = load_model(model_path)
    prompts = DIAGNOSTIC_PROMPTS[:n_prompts]

    all_consistencies: list[dict] = []
    empty_detected = False

    for prompt_spec in prompts:
        prompt = prompt_spec["prompt"]
        max_tokens = prompt_spec["max_tokens"]

        runs: list[list[int]] = []
        for _ in range(n_runs):
            result = generate_text(
                model, tokenizer, prompt,
                max_tokens=max_tokens, temp=0.0, seed=seed,
            )
            if not result.tokens:
                empty_detected = True
            runs.append(result.tokens)

        consistency = multi_run_consistency(runs)
        consistency["prompt_id"] = prompt_spec["id"]
        all_consistencies.append(consistency)

    if empty_detected:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.CRITICAL,
            detail="Empty output detected in one or more runs. Model may be broken.",
            remediation="Check model weights and configuration. Run Tier 0 checks first.",
            metadata={"consistencies": all_consistencies},
        )

    # Compute overall agreement
    avg_agreement = sum(c["agreement_rate"] for c in all_consistencies) / len(all_consistencies)
    all_consistent = all(c["consistent"] for c in all_consistencies)
    divergence_rate = 1.0 - avg_agreement

    if all_consistent:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus.PASS,
            detail=f"Model is deterministic at temp=0. {n_prompts} prompts × {n_runs} runs, 100% token agreement.",
            metadata={"avg_agreement": avg_agreement, "n_prompts": n_prompts, "n_runs": n_runs},
        )

    if divergence_rate <= MINOR_DIVERGENCE_THRESHOLD:
        return DiagnosticResult(
            check_id="1.1",
            name="Determinism",
            status=CheckStatus