---
title: README
type: note
permalink: mlxtriage/readme
---

# mlx-triage

[![PyPI version](https://img.shields.io/pypi/v/mlx-triage.svg)](https://pypi.org/project/mlx-triage/)
[![CI](https://github.com/swaylenhayes/mlx-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/swaylenhayes/mlx-triage/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/mlx-triage.svg)](https://pypi.org/project/mlx-triage/)
[![License: MIT](https://img.shields.io/github/license/swaylenhayes/mlx-triage.svg)](https://github.com/swaylenhayes/mlx-triage/blob/main/LICENSE)
![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)
![Validated: 32 models across 10 families](https://img.shields.io/badge/validated-32%20models%20across%2010%20families-orange.svg)

**Your MLX model is producing garbage. Is it the weights? A known MLX bug? Your quantization settings?**

mlx-triage answers that in 30 seconds — without loading the model into memory.

```bash
pip install mlx-triage
mlx-triage check ./my-model
```

![mlx-triage demo](docs/assets/demo.gif)

## What It Checks

Tested against **32 models** across **10 families** (Qwen, Gemma, GLM, Mistral/Devstral, LiquidAI, GPT-OSS, Nemotron, Llama, Phi, Nanbeige), **7 quantization formats** (bf16 through QAT 4-bit and MXFP4), from 0.6B to 35B parameters. Zero false negatives. [Full validation results ->](docs/validation-results.md)

### Tier 0 — Sanity Checks (no MLX needed, < 30 seconds)

| Check | What it catches |
|-------|----------------|
| **Dtype Compatibility** | BF16->FP16 precision loss, training/storage dtype mismatches |
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

Requires Python 3.11+ and macOS on Apple Silicon (M1-M4).

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

# Require full execution (fail if any check is skipped)
mlx-triage check /path/to/model --tier 1 --format json --strict

# Save report to file
mlx-triage check /path/to/model --tier 1 --output report.json
```

Tier 0 runs in under 30 seconds on any model. Tier 1 requires MLX and takes 5-15 minutes depending on model size.

## Reliability Claims in JSON Output

Each JSON report now includes:

- `claim_level`: `runtime-qualified` when all checks executed, `preflight-only` when any check was skipped
- `checks_executed`: Number of checks that ran
- `checks_skipped`: Number of checks skipped
- `skipped_check_ids`: IDs of skipped checks

Use `--strict` in CI or external reporting workflows to enforce full execution. In strict mode, mlx-triage exits with a non-zero status if any check is skipped.

## How It Works

mlx-triage uses a tiered diagnostic protocol — each tier increases in depth and cost:

1. **Tier 0** reads model files directly (safetensors headers, config JSON, tokenizer config) without loading the model into memory. This catches the most common issues instantly.

2. **Tier 1** loads the model via MLX and runs statistical tests — determinism checks (10 runs at temp=0), perplexity measurement against a fixed eval corpus, and optional comparison against a PyTorch reference backend.

3. **Tiers 2-3** (planned) will add isolation tests (batch invariance, memory pressure, context length stress) and deep diagnostics (layer-wise activation comparison, cross-runtime analysis).

If Tier 0 finds critical issues, Tier 1 is skipped — fix the fundamentals first.

## Known Bugs Database

mlx-triage ships with a curated database of documented MLX bugs ([`known_bugs.yaml`](src/mlx_triage/data/known_bugs.yaml)), cross-referenced against your installed MLX version and model architecture. Running MLX < 0.22.0 with float16 weights? It flags the known qmv kernel overflow. Got a 4-bit Llama model looping on long prompts? There's a documented bug for that. Safetensors file looks valid but weights are numerically garbage? That's a known silent bfloat16 corruption path.

Contributing a bug report to the database is the easiest way to help — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Research Basis

The diagnostic protocol is grounded in systematic analysis of MLX infrastructure defects across multiple model architectures and quantization levels. See [METHODOLOGY.md](METHODOLOGY.md) for the evidence basis, including infrastructure defect taxonomy, first-party experiments, and cross-model synthesis.

## Contributing

Contributions welcome — especially to the known bugs database. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)

---

If mlx-triage saved you a debugging session, **star it** — it helps other MLX developers find the tool.