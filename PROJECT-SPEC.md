# mlx-triage: MLX Inference Quality Diagnostic Toolkit

## Project Spec v1.0 — February 2026

---

## Executive Summary

**The problem:** When an MLX-served model produces bad output on Apple Silicon, practitioners can't tell whether the issue is the model's weights or the inference infrastructure. They waste hours (sometimes weeks) debugging the wrong layer.

**The gap:** The research community has strong evidence that infrastructure doesn't matter *in general* and strong evidence it matters *in specific cases*, but nobody has built a systematic triage framework for practitioners to know **which regime they're in** — especially for the MLX/Apple Silicon stack.

**The tool:** `mlx-triage` is a Python CLI that runs a tiered diagnostic protocol against any MLX-served model, producing a structured report that tells you whether your quality issue is infrastructure or model, and what to do about it.

**The contribution:** This fills a documented gap confirmed by 11 independent frontier model analyses, two systematic reviews (Consensus PRISMA, 1,099→50 papers), and production postmortems from Anthropic, Meta, and baa.ai. The specific intersection of "batch/scheduler diagnostics × unified memory/Metal shader effects" and "XAI diagnostics × unified memory/Metal shader effects" are identified as **complete gaps** in the research landscape.

**Target platform:** Apple Silicon (M1–M4), MLX framework, local inference. Your M2 Max 96GB is the primary development and test target.

---

## Architecture Overview

### Design Philosophy

The tool follows the **tiered diagnostic protocol** that emerged from the Variant C evidence synthesis. Each tier is more expensive but more precise. Most issues resolve at Tier 0 or Tier 1 — the tool should make that fast and obvious.

```
┌─────────────────────────────────────────────────────┐
│                   mlx-triage CLI                     │
│                                                      │
│  mlx-triage check <model_path> [--tier 0|1|2|3]     │
│  mlx-triage report <model_path> --output report.json │
│  mlx-triage compare <model_a> <model_b>              │
│  mlx-triage monitor <model_path> --continuous         │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Tier 0: Sanity    │  < 30 seconds
    │   (Config Checks)   │  No inference required
    ├─────────────────────┤
    │   Tier 1: Smoke     │  1–5 minutes
    │   (Statistical)     │  ~100 inference calls
    ├─────────────────────┤
    │   Tier 2: Isolation │  5–30 minutes
    │   (Batch & Memory)  │  ~500 inference calls
    ├─────────────────────┤
    │   Tier 3: Deep      │  30–120 minutes
    │   (Activation Diff) │  Layer-wise analysis
    └─────────────────────┘
```

### Core Dependencies

```
mlx                  # Apple's ML framework
mlx-lm               # MLX language model utilities
transformers         # HuggingFace (reference implementation)
numpy                # Numerical operations
rich                 # Terminal UI / progress / tables
click                # CLI framework
```

Optional (for Tier 3 and cross-runtime comparison):
```
llama-cpp-python     # llama.cpp bindings for cross-runtime checks
lm-eval              # EleutherAI evaluation harness
```

---

## Tier 0: Sanity Checks (The Free Wins)

**Cost:** < 30 seconds, no inference required
**What it catches:** ~40% of all infrastructure-vs-model confusion (estimated from Ollama template disaster scale + dtype mismatch incidents)

### Check 0.1: Dtype Compatibility Audit

**Evidence basis:** Gemma 3 BF16-to-FP16 incident (all sizes produce empty/nonsense output in FP16), baa.ai bfloat16 silent corruption bug, HuggingFace PR #29285 (BF16 RoPE precision loss at 8K+ tokens).

**Implementation:**

```python
def check_dtype_compatibility(model_path: str) -> DiagnosticResult:
    """
    1. Read model config.json for training dtype
    2. Read MLX model weights to detect stored dtype
    3. Flag mismatches:
       - Model trained in BF16 but weights stored as FP16 → CRITICAL
       - Model trained in FP32 but loaded as FP16 → WARNING
       - Any FP16 usage with non-standard architectures → WARNING
    4. Check for known problematic models (Gemma 3, GatedDeltaNet)
    """
```

**Known failure signatures (from evidence):**
- Gemma 3 in FP16: activations reach 800,000 after layer norm, FP16 max is 65,504 → overflow → NaN
- baa.ai Qwen3.5-397B: `mx.save_safetensors` with bfloat16 produces structurally valid but numerically garbage files
- GatedDeltaNet/recurrent architectures: float16 accumulation → NaN → collapse

### Check 0.2: Tokenizer & EOS Configuration Audit

**Evidence basis:** Ollama 400+ issues from silent context truncation, llama.cpp ChatML fallback, vLLM double-BOS injection, Llama 3 dual stop token issue.

**Implementation:**

```python
def check_tokenizer_config(model_path: str) -> DiagnosticResult:
    """
    1. Load tokenizer and verify:
       - EOS token(s) are defined and match model's training config
       - Chat template exists and renders correctly (not falling back to ChatML)
       - BOS token count in rendered template matches expected (detect double-BOS)
    2. Render a sample conversation through the chat template
       and verify output structure
    3. Check for known problematic patterns:
       - Llama 3: must have BOTH <|end_of_text|> AND <|eot_id|>
       - Models with custom stop tokens not in default config
    """
```

### Check 0.3: Weight File Integrity

**Evidence basis:** baa.ai postmortem — `mx.save_safetensors` with bfloat16 produces files that pass structural validation but contain garbage weights. "What is 2+2?" → "!!!!!!!!!!!!!!!!!!!"

**Implementation:**

```python
def check_weight_integrity(model_path: str) -> DiagnosticResult:
    """
    1. For each safetensors shard:
       - Compute SHA-256 checksum
       - Compare against expected (if available from HuggingFace hub)
    2. Statistical sanity check on weight tensors:
       - Check for NaN/Inf values
       - Check weight distribution (mean, std, min, max)
       - Flag if any layer has suspiciously uniform or zero values
       - Flag if weight magnitudes are outside expected range for architecture
    3. Quick smoke test: load a single layer, multiply by known input,
       verify output is numerically reasonable
    """
```

### Check 0.4: MLX Version & Known Bug Check

**Evidence basis:** MLX issue #488 (naive reductions in float16, open since Jan 2024), issue #2695 (addmm wrong results), qmv kernel infinite loop bug.

**Implementation:**

```python
def check_mlx_version(model_path: str) -> DiagnosticResult:
    """
    1. Get installed MLX version
    2. Check against known-bug database:
       - mlx < X.Y.Z: qmv infinite loop at specific tensor dimensions
       - mlx < X.Y.Z: addmm float16 CPU wrong results
       - mlx < X.Y.Z: Conv1d composition drift
    3. Check model architecture against known MLX compatibility issues:
       - MoE models: check for MLA bottleneck migration bugs
       - Vision models: check for GELU approximation mismatch
       - Recurrent models: check for float16 accumulation risk
    """
```

### Tier 0 Output

```json
{
  "tier": 0,
  "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
  "timestamp": "2026-02-25T14:30:00Z",
  "checks": {
    "dtype_compatibility": {
      "status": "PASS",
      "detail": "Training dtype BF16, stored as 4-bit quantized. Compatible."
    },
    "tokenizer_config": {
      "status": "WARNING",
      "detail": "Chat template renders correctly but model defines 3 stop tokens; only 2 are in generation_config.json.",
      "remediation": "Add missing stop token to generation_config.json or pass explicitly."
    },
    "weight_integrity": {
      "status": "PASS",
      "detail": "All shards match hub checksums. No NaN/Inf detected."
    },
    "mlx_version": {
      "status": "PASS",
      "detail": "mlx 0.25.1 — no known bugs affecting this architecture."
    }
  },
  "verdict": "WARNING — 1 issue found. Fix tokenizer config before proceeding to Tier 1.",
  "should_continue": true
}
```

---

## Tier 1: Statistical Smoke Test (Token-DiFR Lite)

**Cost:** 1–5 minutes, ~100 inference calls
**What it catches:** Quantization-induced quality degradation, systematic output divergence from reference, non-determinism at temperature=0

**Evidence basis:** Token-DiFR (Nov 2025) — detects KV cache quantization with ~10K tokens, 4-bit model quantization with ~1K tokens, incorrect sampling seeds with ~100 tokens. The key insight: LLM inference is ~98% deterministic when sampling seed is fixed.

### Test 1.1: Determinism Check

**Purpose:** Verify the model produces identical output for identical inputs at temperature=0. If it doesn't, infrastructure non-determinism is confirmed.

```python
def test_determinism(model_path: str, n_runs: int = 10) -> DiagnosticResult:
    """
    1. Select 5 diverse test prompts (math, code, factual, creative, multi-turn)
    2. Run each prompt n_runs times at temperature=0, same seed
    3. Compare outputs token-by-token
    4. Report:
       - Perfect match across all runs → PASS (deterministic)
       - Minor divergence (< 2% token positions) → INFO (expected floating-point)
       - Major divergence (> 5% token positions) → FAIL (infrastructure non-determinism)
       - NaN or empty outputs in any run → CRITICAL

    Key insight from evidence: quantized integer models (Q4_K_M, Q8_0)
    achieve perfect reproducibility on MLX, while float16 does not.
    If quantized model is non-deterministic, something is very wrong.
    """
```

### Test 1.2: Reference Divergence (Token-DiFR Core)

**Purpose:** Compare MLX model output against a trusted reference (HuggingFace Transformers with same weights) to measure infrastructure-induced divergence.

```python
def test_reference_divergence(
    model_path: str,
    reference_backend: str = "transformers"
) -> DiagnosticResult:
    """
    1. Load same base weights in both MLX and reference backend
    2. Run standardized prompt suite (20 prompts × 256 max tokens = ~5K tokens)
    3. For each prompt, compare:
       - First-token agreement (prefill path comparison)
       - Token-by-token agreement rate over full generation
       - Log-probability divergence at each position (if accessible)
    4. Compute divergence score:
       - Agreement > 95% → PASS (normal floating-point variance)
       - Agreement 80-95% → WARNING (possible quantization artifact)
       - Agreement < 80% → FAIL (infrastructure defect likely)

    Note: This is a simplified Token-DiFR. The full version uses statistical
    tests on divergence distributions; we use agreement rate as a proxy.
    """
```

### Test 1.3: Quantization Quality Gate

**Purpose:** Quick perplexity/benchmark comparison to verify the quantized model is within expected quality band of the base model.

```python
def test_quantization_quality(model_path: str) -> DiagnosticResult:
    """
    1. Run a small perplexity evaluation (WikiText-2 subset, ~1000 tokens)
    2. Compare against known perplexity for this model at this quantization level
       (from community benchmarks or our own reference table)
    3. Thresholds (from oobabooga community benchmark evidence):
       - Perplexity within 0.1 of reference → PASS
       - Perplexity within 0.5 of reference → WARNING (check quantization method)
       - Perplexity > 0.5 above reference → FAIL (quantization or infrastructure issue)

    Evidence context: oobabooga found 0.069 perplexity spread across ALL 4-bit
    methods on Llama-2-13B. Spread > 0.1 is abnormal.
    """
```

### Prompt Suite Design

The test prompts should cover failure modes identified in the evidence:

```python
DIAGNOSTIC_PROMPTS = {
    # Math reasoning (sensitive to precision drift)
    "math_basic": "What is 2+2? Answer with just the number.",
    "math_chain": "If I have 15 apples and give away 7, then buy 3 more, how many do I have?",

    # Code generation (sensitive to constrained decoding, tokenizer issues)
    "code_python": "Write a Python function that reverses a string.",
    "code_json": 'Output valid JSON: {"name": "test", "value": 42}',

    # Long-context (sensitive to RoPE precision, KV cache management)
    "context_needle": "<long context with embedded fact to retrieve>",

    # Structured output (sensitive to EOS handling, template issues)
    "structured": "List exactly 3 fruits, one per line, nothing else.",

    # Multi-turn (sensitive to context truncation, template rendering)
    "multi_turn": [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice!"},
        {"role": "user", "content": "What's my name?"}
    ],

    # Edge cases from evidence
    "repetition_check": "Describe the process of photosynthesis in 2 sentences.",
    "unicode_check": "Translate 'hello' to Japanese.",  # Anthropic TPU bug produced stray CJK
    "eos_boundary": "Say 'done' and stop."  # Tests EOS handling
}
```

---

## Tier 2: Isolation Tests (Batch & Memory)

**Cost:** 5–30 minutes, ~500 inference calls
**What it catches:** Batch-size-dependent quality degradation, memory pressure effects, KV cache corruption under load

**Evidence basis:** vLLM issue #5898 (batch-size accuracy drop), Thinking Machines Lab batch invariance analysis (3 categories of non-invariant kernels), mlx-lm cache_limit progressive degradation, vLLM-Ascend KV cache poisoning.

### Test 2.1: Batch Invariance

**Purpose:** Verify output quality is independent of batch size. This is the Thinking Machines Lab diagnostic directly applied to MLX.

```python
def test_batch_invariance(model_path: str) -> DiagnosticResult:
    """
    1. Run 10 prompts individually (batch_size=1) → collect reference outputs
    2. Run same 10 prompts as a single batch → collect batched outputs
    3. Compare token-by-token:
       - Perfect match → PASS (batch invariant)
       - Minor divergence (< 2%) → INFO (floating-point non-associativity, expected)
       - Major divergence (> 5%) → FAIL (batch invariance violation)
       - Quality degradation in batch but not individual → CRITICAL

    Note: MLX historically uses static batching, but vllm-mlx introduces
    continuous batching. Test both paths if vllm-mlx is installed.
    """
```

### Test 2.2: Memory Pressure Sweep

**Purpose:** Detect quality degradation correlated with memory pressure. The mlx-lm `cache_limit` issue caused progressive degradation during inference as memory grew.

```python
def test_memory_pressure(model_path: str) -> DiagnosticResult:
    """
    1. Establish baseline quality at low memory usage (short prompt, fresh model)
    2. Progressively increase memory pressure:
       a. Generate increasingly long outputs (256, 512, 1024, 2048 tokens)
       b. At each step, re-run a reference prompt and check output consistency
    3. Monitor system memory usage at each step
    4. Report:
       - Consistent output quality across all pressure levels → PASS
       - Quality drops at specific memory threshold → WARNING (report threshold)
       - Progressive degradation correlated with memory → FAIL (likely cache issue)
       - OOM or crash → CRITICAL (report memory level at failure)

    Implementation note: Use mlx.core.metal.get_active_memory() and
    get_peak_memory() to track MLX-specific memory state.
    """
```

### Test 2.3: Context Length Stress

**Purpose:** Detect RoPE precision loss and KV cache management failures at longer context lengths.

```python
def test_context_length(model_path: str) -> DiagnosticResult:
    """
    Evidence: BF16 RoPE loses position precision at 8K+ tokens
    (positions 8188-8191 all appear as 8192).

    1. Create "needle in a haystack" test at multiple context lengths:
       512, 1K, 2K, 4K, 8K, 16K (up to model's max)
    2. Embed a unique fact at position 25%, 50%, 75% of context
    3. Ask model to retrieve the fact
    4. Report retrieval accuracy by context length and position:
       - Consistent accuracy → PASS
       - Accuracy drops at specific length → WARNING (report threshold)
       - Cliff-edge drop (e.g., 95% → 20%) → FAIL (likely RoPE or KV issue)
    """
```

---

## Tier 3: Deep Diagnostics (Activation Diffing)

**Cost:** 30–120 minutes, layer-wise analysis
**What it catches:** Specific layer-level infrastructure failures, quantization hotspots, numerical drift accumulation
**When to use:** Only when Tiers 0–2 have confirmed an issue exists but haven't pinpointed the cause

**Evidence basis:** MLX Metal float16 investigation (errors compound through layers: 1e5 at layer 20, 1e25 at layer 60, NaN at layer 80), PaliGemma vision encoder divergence (>100K sum-abs-diff from GELU approximation mismatch).

### Test 3.1: Layer-wise Activation Comparison

```python
def test_activation_diff(
    model_path: str,
    reference_backend: str = "transformers"
) -> DiagnosticResult:
    """
    1. Hook into each layer's output in both MLX and reference backend
    2. Run identical input through both
    3. At each layer, compute:
       - Mean Absolute Error (MAE) between MLX and reference activations
       - Max Absolute Error
       - Cosine similarity of activation vectors
    4. Plot error accumulation across layers
    5. Identify:
       - Sharp error jump at specific layer → architecture-specific bug
       - Gradual drift → precision accumulation (expected in float16)
       - Error spike at attention layers → RoPE or KV cache issue
       - Error spike at norm layers → RMSNorm precision issue
    """
```

### Test 3.2: Cross-Runtime Comparison

**Purpose:** Run the same weights through MLX and llama.cpp (GGUF) to isolate MLX-specific issues.

```python
def test_cross_runtime(
    model_path: str,
    gguf_path: str = None
) -> DiagnosticResult:
    """
    Evidence: MLX 4-bit vs llama.cpp Q4_K_M shows 4.0 vs 4.7 bits-per-weight
    discrepancy, yet MMLU Pro scores are nearly identical (36.15% vs 36.10%).

    1. If GGUF path not provided, check if equivalent GGUF exists on HuggingFace
    2. Run identical prompts through both runtimes
    3. Compare:
       - Token agreement rate
       - Perplexity on shared evaluation set
       - Structured output compliance (JSON validity, list formatting)
    4. Report:
       - Similar quality → issue is model-level (both runtimes affected)
       - MLX worse → infrastructure issue specific to MLX
       - llama.cpp worse → likely quantization format difference, not bug
    """
```

---

## Project Structure

```
mlx-triage/
├── README.md
├── pyproject.toml
├── src/
│   └── mlx_triage/
│       ├── __init__.py
│       ├── cli.py                 # Click CLI entry point
│       ├── config.py              # Known bugs database, thresholds
│       ├── report.py              # Report generation (JSON, terminal, markdown)
│       │
│       ├── tier0/                 # Sanity checks (no inference)
│       │   ├── __init__.py
│       │   ├── dtype_check.py
│       │   ├── tokenizer_check.py
│       │   ├── weight_integrity.py
│       │   └── version_check.py
│       │
│       ├── tier1/                 # Statistical smoke tests
│       │   ├── __init__.py
│       │   ├── determinism.py
│       │   ├── reference_divergence.py
│       │   └── quantization_quality.py
│       │
│       ├── tier2/                 # Isolation tests
│       │   ├── __init__.py
│       │   ├── batch_invariance.py
│       │   ├── memory_pressure.py
│       │   └── context_length.py
│       │
│       ├── tier3/                 # Deep diagnostics
│       │   ├── __init__.py
│       │   ├── activation_diff.py
│       │   └── cross_runtime.py
│       │
│       ├── prompts/               # Diagnostic prompt suites
│       │   ├── __init__.py
│       │   └── standard_suite.py
│       │
│       └── utils/                 # Shared utilities
│           ├── __init__.py
│           ├── mlx_utils.py       # MLX-specific helpers
│           ├── memory_monitor.py  # Memory tracking
│           └── comparison.py      # Token/activation comparison functions
│
├── tests/                         # Test suite
│   ├── test_tier0.py
│   ├── test_tier1.py
│   ├── test_tier2.py
│   └── test_tier3.py
│
├── known_bugs.yaml                # Database of known MLX/framework bugs
├── reference_benchmarks.yaml      # Expected perplexity/quality for common models
└── docs/
    ├── METHODOLOGY.md             # Research basis and evidence citations
    ├── CONTRIBUTING.md
    └── examples/
        ├── basic_triage.md
        └── deep_diagnostic.md
```

---

## Implementation Plan

### Phase 0: Scaffolding (1–2 sessions)

**Goal:** CLI skeleton that runs, produces output, and feels like a real tool.

**Tasks:**
1. Initialize project with `uv init mlx-triage`
2. Set up Click CLI with `check`, `report`, `compare` commands
3. Implement Tier 0 checks (these require no inference, just config parsing)
4. Build the Rich terminal output (tables, progress bars, colored status)
5. JSON report generation

**Ship criteria:** `mlx-triage check <any-mlx-model>` produces a Tier 0 report in < 30 seconds.

**Why start here:** Tier 0 is the highest-value-per-effort tier. The dtype check alone would have caught the Gemma 3 FP16 disaster and the baa.ai bfloat16 corruption. You'll learn the MLX model loading internals without needing to run inference yet.

### Phase 1: Statistical Core (2–3 sessions)

**Goal:** Tier 1 determinism and reference divergence tests working.

**Tasks:**
1. Build the diagnostic prompt suite
2. Implement determinism check (run same prompt N times, compare)
3. Implement reference divergence (MLX vs Transformers on same weights)
4. Implement perplexity gate (WikiText-2 subset)
5. Calibrate thresholds against 3–5 known-good models on your M2 Max

**Ship criteria:** `mlx-triage check <model> --tier 1` produces a Tier 0+1 report that correctly identifies a known-good model as PASS and a deliberately misconfigured model as FAIL.

**Key learning:** This phase teaches you how MLX inference actually works — model loading, tokenization, generation loops, logit extraction. This is the core competency for the rest of the tool.

### Phase 2: Isolation Tests (2–3 sessions)

**Goal:** Tier 2 batch and memory diagnostics.

**Tasks:**
1. Implement batch invariance test
2. Implement memory pressure sweep with MLX Metal memory monitoring
3. Implement context length stress test (needle-in-a-haystack)
4. Build the "diagnostic divergence" heuristic from Gemini's evidence: if the failure is an immediate crash/OOM/NaN, it's infrastructure; if it's gradual quality degradation, it's probably model

**Ship criteria:** `mlx-triage check <model> --tier 2` produces a full diagnostic report. Bonus: reproduce a known MLX issue (e.g., the qmv infinite loop on specific tensor dimensions) and verify the tool catches it.

### Phase 3: Deep Diagnostics & Contribution (3–5 sessions)

**Goal:** Tier 3 activation diffing and the write-up that makes this a contribution.

**Tasks:**
1. Implement layer-wise activation hooking in MLX
2. Implement cross-runtime comparison (MLX vs llama.cpp)
3. Build visualization: error accumulation plots across layers
4. Write METHODOLOGY.md with full evidence citations from our research
5. Package as pip-installable tool
6. Write a blog post / technical report suitable for the MLX community

**Ship criteria:** A public GitHub repo with documentation, a blog post explaining the methodology and evidence basis, and at least one real-world diagnostic case study run on your M2 Max.

### Phase 4: Community & Iteration (ongoing)

**Goal:** Get feedback, iterate, build the reference benchmark database.

**Tasks:**
1. Share with MLX community (GitHub discussions, Discord)
2. Collect benchmark data from community runs on different hardware (M1, M2, M3, M4)
3. Expand known_bugs.yaml as new MLX issues are discovered
4. Consider integration with mlx-lm as a diagnostic subcommand
5. Write up for potential workshop paper (the Consensus gap analysis gives you the framing)

---

## Known Bug Database Schema

This is the heart of the tool's institutional knowledge — a curated database of documented infrastructure failures that the tool can check against.

```yaml
# known_bugs.yaml
bugs:
  - id: MLX-001
    title: "Float16 addmm CPU wrong results"
    mlx_issue: 2695
    affected_versions: ["< 0.22.0"]  # Update when fixed
    severity: critical
    detection: "tier0.dtype_check"
    symptom: "Completely wrong matrix multiplication results on CPU with float16"
    architecture: ["all"]
    remediation: "Use bfloat16 or float32, or upgrade MLX"

  - id: MLX-002
    title: "qmv kernel infinite loop at specific tensor dimensions"
    mlx_issue: null  # Community reported
    affected_versions: ["< 0.24.0"]
    severity: critical
    detection: "tier0.version_check"
    symptom: "Model enters infinite generation loop at end of long prompts (4-bit only)"
    architecture: ["llama", "mistral"]
    remediation: "Use 8-bit quantization or upgrade MLX"

  - id: MLX-003
    title: "Metal float16 non-determinism with error accumulation"
    mlx_issue: 488
    affected_versions: ["all"]  # Still open as of Feb 2026
    severity: warning
    detection: "tier1.determinism"
    symptom: "Non-reproducible outputs at temperature=0 with float16 weights"
    architecture: ["all"]
    remediation: "Use quantized integer models (Q4, Q8) for reproducibility"

  - id: MLX-004
    title: "bfloat16 save_safetensors silent corruption"
    source: "baa.ai postmortem"
    affected_versions: ["unknown"]
    severity: critical
    detection: "tier0.weight_integrity"
    symptom: "Structurally valid safetensors file with numerically garbage weights"
    architecture: ["all"]
    remediation: "Verify weight checksums against hub; re-download if mismatch"

  - id: MLX-005
    title: "Conv1d composition drift"
    mlx_issue: 2122
    affected_versions: ["all"]
    severity: warning
    detection: "tier3.activation_diff"
    symptom: "Sequential Conv1d operations accumulate MAE ~0.04 vs PyTorch"
    architecture: ["whisper", "wav2vec", "audio_models"]
    remediation: "Use float32 for audio model inference or validate output quality"

  - id: MLX-006
    title: "GELU approximation mismatch in vision encoders"
    source: "PaliGemma MLX issue"
    affected_versions: ["varies"]
    severity: high
    detection: "tier3.activation_diff"
    symptom: "Vision encoder activations diverge >100K sum-abs-diff from reference"
    architecture: ["paligemma", "clip", "siglip"]
    remediation: "Verify GELU implementation matches model's training config"
```

---

## Reference Benchmark Schema

Community-sourced expected quality levels for common models, used by Tier 1 to set pass/fail thresholds.

```yaml
# reference_benchmarks.yaml
models:
  - name: "Llama-3.1-8B-Instruct"
    quantizations:
      4bit:
        perplexity_wikitext2: 6.24
        perplexity_tolerance: 0.15
        mmlu_score: 0.68
        humaneval_pass1: 0.51
      8bit:
        perplexity_wikitext2: 6.12
        perplexity_tolerance: 0.10
      fp16:
        perplexity_wikitext2: 6.10
        perplexity_tolerance: 0.05
    notes: "Known qmv loop issue at 4-bit on older MLX versions"

  - name: "Qwen2.5-32B-Instruct"
    quantizations:
      4bit:
        perplexity_wikitext2: 4.89
        perplexity_tolerance: 0.15
        humaneval_pass1: 0.58
    notes: "Large model; memory pressure tests important on < 64GB systems"
```

---

## What You Already Know That Helps

This project builds directly on skills you have:

**Product management (PG&E Construct):** You know how to build tools for field practitioners who need clear, actionable diagnostic output — not academic analysis. The tiered protocol design is fundamentally a UX decision: give the fast answer first, offer depth only when needed.

**MLX model research:** Your MLX model evaluation work means you already understand quantization levels, model architectures, and the practical tradeoffs on M2 Max. You've run dozens of models through this stack.

**Python CLI tooling:** Click, Rich, UV — all in your existing toolchain. The scaffolding phase is stuff you've done before.

**Multi-agent development:** This project is perfectly structured for Claude Code / Codex / Gemini CLI. Each tier is an independent module with clear inputs, outputs, and test criteria. Hand each phase to your agent bench and compare their implementations.

---

## What You'll Need to Learn

Being transparent about the growth edges:

**MLX internals:** How `mx.eval()` works, Metal kernel dispatch, memory allocation. Start with the MLX docs on lazy evaluation — it's the single most important concept for understanding why MLX behaves differently from PyTorch.

**Activation hooking:** Intercepting intermediate layer outputs in MLX. This is well-documented in PyTorch (`register_forward_hook`) but less so in MLX. You may need to modify the model's forward pass directly. This is a Phase 3 concern — don't worry about it until you get there.

**Perplexity calculation:** The math is simple (cross-entropy loss over a corpus), but the implementation details matter (stride, context window, tokenization). The `lm-eval` harness handles this, but understanding what it's doing helps you interpret results.

**Statistical testing:** Token-DiFR uses statistical tests to determine if divergence is significant. For v1, simple agreement rates are fine. For a contribution-quality tool, you'd want proper hypothesis testing (chi-squared on token agreement, Kolmogorov-Smirnov on logit distributions).

---

## Success Criteria

### Minimum Viable Tool (end of Phase 1)
- CLI that produces Tier 0 + Tier 1 reports for any MLX model
- Correctly identifies at least 3 known issue categories
- JSON report output suitable for sharing/comparison
- README with clear installation and usage instructions

### Contribution-Ready (end of Phase 3)
- Full Tier 0–3 diagnostic coverage
- At least 5 real-world diagnostic case studies on M2 Max
- METHODOLOGY.md citing the evidence base from our Variant C research
- Blog post or technical report explaining the gap and how this tool fills it
- Published to PyPI as `mlx-triage`

### Research Contribution (Phase 4+)
- Community benchmark database with data from 3+ hardware configurations
- Workshop paper or technical report with the tiered protocol as the core contribution
- Integration with or endorsement from MLX community maintainers

---

## Evidence Trail

This project's methodology is grounded in the following evidence, collected and validated through multi-agent peer review (11 frontier models, 2 systematic reviews, February 2026):

**Production postmortems:** Anthropic September 2025 (3 simultaneous infra bugs, weeks of degradation, standard evals failed to detect), baa.ai Qwen3.5-397B quantization (4 MLX bugs, 2 causing total collapse), Meta HawkEye (50% reduction in time-to-root-cause).

**Empirical studies:** 929-bug taxonomy across 5 inference engines, 500K+ evaluation study (ACL 2025), oobabooga community benchmark (0.069 perplexity spread across all 4-bit methods), JarvisLabs HumanEval cross-framework comparison (identical 51.8% Pass@1).

**MLX-specific evidence:** Metal float16 non-determinism investigation (NaN by layer 80), MLX-VLM testing report (38 VLMs, 12 with quality issues, all traced to model capability), MMLU Pro MLX-vs-llama.cpp (36.15% vs 36.10%), PaliGemma GELU mismatch, bfloat16 save corruption.

**Diagnostic tools referenced:** Token-DiFR (statistical inference verification), Thinking Machines Lab batch_invariant_ops (deterministic serving at ~34% overhead), Meta HawkEye (decision-tree debugging), EleutherAI lm-evaluation-harness.

**Research gap confirmation:** Consensus PRISMA review (1,099→50 papers) identifies "Batch/Scheduler Bugs × Unified Memory/Metal Shader Effects" and "XAI Diagnostics × Unified Memory/Metal Shader Effects" as complete gaps.
