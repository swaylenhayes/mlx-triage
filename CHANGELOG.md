# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 — Initial Release

### Tier 0 — Sanity Checks (no MLX required)

- Dtype compatibility audit (BF16/FP16 mismatch detection)
- Tokenizer and EOS configuration validation
- Weight file integrity checks (NaN/Inf, all-zero layers)
- MLX version and known bug cross-reference

### Tier 1 — Statistical Smoke Tests (MLX required)

- Determinism check (10 runs at temp=0)
- Reference divergence (MLX vs PyTorch/Transformers)
- Quantization quality gate (perplexity measurement)

### Infrastructure

- Rich terminal output with colored status indicators
- JSON output format for programmatic use
- Known bugs database (6 curated entries with MLX issue references)
- Validated against 15 models across 5 families, 4 quantization levels
