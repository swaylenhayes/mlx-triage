# mlx-triage

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Tests: 102 passing](https://img.shields.io/badge/tests-102%20passing-brightgreen.svg)
![Models Validated: 13](https://img.shields.io/badge/models%20validated-13-orange.svg)

**Diagnose MLX model quality issues on Apple Silicon.**

You downloaded an MLX model, ran it, got garbage output. Is it the model? The quantization? A known MLX bug? Your config? Without systematic tooling, answering that question takes hours of manual debugging.

mlx-triage tells you in 30 seconds — without loading the model into memory.

```bash
pip install mlx-triage
mlx-triage check ./my-model
```

![mlx-triage demo](docs/assets/demo.gif)

## What It Checks

Validated against 13 models across 5 families, 4 quantization levels (bf16 through 4-bit), and sizes from 0.6B to 30B parameters. Zero false negatives. [Full results](docs/validation-results.md).

### Tier 0 — Sanity Checks (no MLX needed, < 30 seconds)

| Check | What it catches |
|-------|----------------|
| **Dtype Compatibility** | BF16→FP16 precision loss, training/storage dtype mismatches |
| **Tokenizer & EOS Config** | Missing EOS tokens, chat template issues, Llama 3 dual-stop-token edge cases |
| **Weight File Integrity** | NaN/Inf values, all-zero layers, corrupt safetensors headers |
| **MLX Version & Known Bugs** | Outdated MLX with documented bugs affecting your model architecture |

### Tier 1 — Statistical Smoke Tests (MLX required)

| Check | What it catches |
|-------|----------------|
| **Determinism** | Non-reproducible outputs at temp=0 (infrastructure issue, not model) |
| **Reference Divergence** | MLX output diverging from PyTorch/Transformers reference |
| **Quantization Quality** | Excessive perplexity indicating broken quantization |

## Install

Requires Python 3.11+ and macOS on Apple Silicon (M1–M4).

```bash
# From PyPI
pip install mlx-triage

# With MLX for Tier 1 checks
pip install "mlx-triage[mlx]"

# With reference comparison (Tier 1, Test 1.2)
pip install "mlx-triage[reference]"

# Development
git clone https://github.com/swaylenhayes/mlx-triage.git
cd mlx-triage
uv sync --extra dev
```

## Usage

```bash
# Tier 0 only (default — no MLX needed)
mlx-triage check /path/to/model

# Tier 0 + Tier 1
mlx-triage check /path/to/model --tier 1

# JSON output
mlx-triage check /path/to/model --format json

# Save report to file
mlx-triage check /path/to/model --tier 1 --output report.json
```

Tier 0 runs in under 30 seconds on any model. Tier 1 requires MLX and takes 5–15 minutes depending on model size.

## How It Works

mlx-triage uses a tiered diagnostic protocol — each tier increases in depth and cost:

1. **Tier 0** reads model files directly (safetensors headers, config JSON, tokenizer config) without loading the model into memory. This catches the most common issues instantly.

2. **Tier 1** loads the model via MLX and runs statistical tests — determinism checks (10 runs at temp=0), perplexity measurement against a fixed eval corpus, and optional comparison against a PyTorch reference backend.

3. **Tiers 2–3** (planned) will add isolation tests (batch invariance, memory pressure, context length stress) and deep diagnostics (layer-wise activation comparison, cross-runtime analysis).

If Tier 0 finds critical issues, Tier 1 is skipped — fix the fundamentals first.

## Known Bugs Database

mlx-triage ships with a curated database of documented MLX bugs ([`known_bugs.yaml`](src/mlx_triage/data/known_bugs.yaml)), sourced from MLX GitHub issues and community postmortems. The version check cross-references your installed MLX version and model architecture against known issues.

Contributing a bug report to the database is the easiest way to help — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Research Basis

The diagnostic protocol is grounded in systematic analysis of MLX infrastructure defects across multiple model architectures and quantization levels. See [METHODOLOGY.md](METHODOLOGY.md) for the evidence basis, including infrastructure defect taxonomy, first-party experiments, and cross-model synthesis.

## Contributing

Contributions welcome — especially to the known bugs database. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)

---

If mlx-triage saved you debugging time, a star helps others find it.
