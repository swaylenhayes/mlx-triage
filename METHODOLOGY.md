# Methodology

mlx-triage's diagnostic protocol is grounded in systematic analysis of MLX inference quality issues across multiple model architectures and quantization levels.

## The Problem

When an MLX-served model produces poor output, practitioners face a diagnostic ambiguity: is the problem in the model weights, the quantization, the inference infrastructure, or the configuration? Without systematic tooling, debugging this requires significant expertise and time.

## Evidence Basis

The diagnostic protocol was designed from analysis of:

- **Infrastructure defect taxonomy** — categorization of MLX bugs by type, severity, and affected components
- **First-party experiments** — controlled experiments isolating infrastructure vs. model quality effects
- **Cross-model synthesis** — patterns observed across multiple model families and quantization levels
- **Systematic review** — structured review of MLX infrastructure constraints and model quality factors

## Tiered Protocol

The protocol uses escalating tiers, each more expensive than the last:

### Tier 0 — Sanity Checks

**Cost:** < 30 seconds, no MLX required.
**Method:** Direct file inspection (safetensors headers, JSON configs).

Catches the most common issues without loading the model:
- Dtype mismatches between training and storage formats (evidence: Gemma 3 BF16→FP16 failure pattern)
- Missing or misconfigured tokenizer settings (evidence: Llama 3 dual-stop-token edge case)
- Corrupt or numerically invalid weight files (evidence: safetensors BF16 silent corruption)
- Known MLX bugs affecting the installed version and model architecture

### Tier 1 — Statistical Smoke Tests

**Cost:** 5–15 minutes, MLX required.
**Method:** Model inference with statistical analysis.

Runs the model and measures output quality:
- **Determinism** — 10 identical runs at temp=0 across 5 diagnostic prompts. 100% agreement expected; divergence indicates infrastructure non-determinism (evidence: MLX issue #488, float16 error accumulation)
- **Reference divergence** — compare MLX output against PyTorch/Transformers on the same prompts. Token agreement thresholds derived from empirical validation across 15 models
- **Quantization quality** — perplexity measured on a fixed 800-token eval corpus. Thresholds calibrated from validation data: <15 (PASS), <25 (INFO), <50 (WARNING), ≥50 (FAIL)

### Tier 2–3 — Planned

Isolation tests (batch invariance, memory pressure, context length stress) and deep diagnostics (layer-wise activation comparison, cross-runtime analysis). See [roadmap](docs/roadmap.md).

## Known Bugs Database

The tool ships with a curated database of documented MLX issues (`src/mlx_triage/data/known_bugs.yaml`). Each entry includes:
- Source (MLX GitHub issue or community postmortem)
- Affected versions and architectures
- Severity classification
- Detection tier and remediation steps

This database is the tool's institutional knowledge. Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- **Apple Silicon only** — MLX runs exclusively on Apple Silicon. Tier 0 file checks work anywhere, but Tier 1+ requires an Apple Silicon Mac.
- **Local models only** — v0.1 requires a local model directory path, not a HuggingFace model ID.
- **Reference backend precision** — the PyTorch reference runs at FP32 on CPU, which may show small numerical differences from the model's native precision.
- **Perplexity thresholds** — calibrated on 15 models. Edge cases (very large or very small models) may need threshold adjustment in future versions.

## Validation

The diagnostic protocol has been validated against 15 models across 5 families, 4 quantization levels, and sizes from 0.6B to 30B parameters. See [validation results](docs/validation-results.md) for the full data.
