---
title: CHANGELOG
type: note
permalink: mlxtriage/changelog
---

# Changelog

All notable changes to this project will be documented in this file.

## v0.2.0 — Reliability Hardening and Traits (2026-03-11)

### New Features

- Added `claim_level` to JSON output: `runtime-qualified` when all checks execute and `preflight-only` when any check is skipped
- Added reliability metadata fields in JSON output: `checks_executed`, `checks_skipped`, and `skipped_check_ids`
- Added `--strict` CLI mode to exit non-zero when any check is skipped
- Added BF16-aware weight integrity inspection with MLX when available (NaN/Inf/all-zero detection)
- Added model-intrinsic trait assembly and Tier 0 trait output in JSON
- Added VLM architecture detection (Check T0.5)
- Expanded tokenizer diagnostics with thinking-token detection and explicit `has_chat_template` metadata

### Improvements

- Introduced backend-agnostic check interfaces via a `ModelBackend` protocol
- Expanded validation and public documentation for model-family and quantization coverage

## v0.1.1 — Packaging and Publication Update (2026-03-03)

### Improvements

- Added trusted publishing workflow for PyPI releases on version tags
- Updated package release plumbing for PyPI activation
- Refined public documentation and validation tables for first publication

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