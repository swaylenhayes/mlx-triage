---
title: roadmap
type: note
permalink: mlxtriage/docs/roadmap
---

# Roadmap — mlx-triage

_Last updated: 2026-03-03 (session 2)_

> Items organized by priority tier (P0-P3). P0 = must have / blocking. P3 = backlog / future.

---

## Vision

A practitioner-facing diagnostic CLI that tells you whether your MLX model quality issue is infrastructure or model — in under 30 seconds for the common case, with escalating depth for harder problems.

---

## P0 — Phase 0: Scaffolding & Tier 0 (COMPLETE)

- [x] Project scaffolding (pyproject.toml, src layout, uv sync)
- [x] Core data models (CheckStatus, DiagnosticResult, TierReport)
- [x] Known bugs database (YAML + config loader)
- [x] CLI skeleton (Click: check command)
- [x] Check 0.1: Dtype compatibility audit
- [x] Check 0.2: Tokenizer & EOS configuration audit
- [x] Check 0.3: Weight file integrity
- [x] Check 0.4: MLX version & known bug check
- [x] Tier 0 runner (orchestrate all checks)
- [x] Report generation (JSON + Rich terminal)
- [x] CLI integration (end-to-end wiring)

**Ship criteria:** `mlx-triage check <any-mlx-model>` produces a Tier 0 report in < 30 seconds. **MET.**

---

## P1 — Phase 1: Statistical Core (Tier 1) — COMPLETE

- [x] MLX inference utilities (model loading, text generation)
- [x] Token comparison utilities (agreement rate, divergence point)
- [x] Diagnostic prompt suite (10 prompts + eval corpus)
- [x] Test 1.1: Determinism check (N runs at temp=0)
- [x] Test 1.2: Reference divergence (MLX vs Transformers)
- [x] Test 1.3: Quantization quality gate (perplexity)
- [x] Tier 1 runner with shared model loading
- [x] E2E validation on 13 real models (M2 Max)

**Ship criteria:** Tier 0+1 correctly identifies known-good as PASS and broken configs as FAIL. **MET.**

---

## P1.5 — Phase 1.5: Launch Packaging — COMPLETE

- [x] Pre-launch bug audit (15 bugs fixed)
- [x] Contribution workflow audit and fixes
- [x] README.md overhaul (badges, demo GIF, example output)
- [x] METHODOLOGY.md (evidence basis, tiered protocol, limitations)
- [x] CONTRIBUTING.md (known bugs YAML guide, TDD workflow)
- [x] LICENSE (MIT)
- [x] CHANGELOG.md
- [x] pyproject.toml polish (classifiers, URLs, author)
- [x] GitHub Actions CI (Python 3.11-3.13, macOS)
- [x] GitHub issue templates (bug report, known MLX bug report)
- [x] Terminal demo GIF (VHS)
- [x] Blog post outline
- [x] E2E install validation (clean venv, all paths tested)

**Ship criteria:** Repository is polished and ready for contributors. **MET.**

---

## P1.5b — Publish v0.1 (IN PROGRESS)

- [x] Create GitHub repository (public)
- [x] Push all code
- [x] Verify CI passes (3 Python versions on macOS)
- [x] Git tag `v0.1.0`
- [x] GitHub release with release notes
- [x] Pre-publish security/attribution audit
- [x] README and presentation polish
- [x] Publish to PyPI (v0.1.0 + v0.1.1 live)
- [x] Replace static badges with live CI/PyPI badges
- [x] Expanded model validation (13 → 32 models, 10 families, 7 quant formats)
- [x] Check 0.5: VLM architecture detection — detect `vision_config` / multimodal architectures in config.json and return a clean FAIL with guidance instead of crashing at model load
- [ ] Blog post (outline at `_private/plans/blog-post-outline.md`)
- [ ] Write-up: "Silent chat_template stripping in MLX Community conversions" — several popular model families (Ministral, GLM, Nemotron, Qwen3.5-MoE) ship without chat_template after conversion, causing silent prompt format fallback
- [ ] Community outreach (HN, r/LocalLLaMA, MLX Discussions)

**Ship criteria:** `pip install mlx-triage` works from clean venv. Blog post live.

---

## P2 — Phase 2: Isolation Tests (Tier 2) — Post v0.1

- [ ] Test 2.1: Batch invariance (single vs batched output comparison)
- [ ] Test 2.2: Memory pressure sweep (MLX Metal memory monitoring)
- [ ] Test 2.3: Context length stress (needle-in-a-haystack at multiple lengths)
- [ ] Diagnostic divergence heuristic (crash = infra, gradual degradation = model)
- [ ] Tier 2 runner + report integration

**Ship criteria:** `mlx-triage check <model> --tier 2` produces full T0+T1+T2 diagnostic report.

---

## P3 — Phase 3: Deep Diagnostics (Tier 3) — Post v0.2

- [ ] Test 3.1: Layer-wise activation comparison (MLX vs reference)
- [ ] Test 3.2: Cross-runtime comparison (MLX vs llama.cpp)
- [ ] Error accumulation visualization

**Ship criteria:** Full Tier 0-3 coverage, 5+ case studies.

---

## Deferred

| Item | Reason deferred |
|------|----------------|
| HuggingFace model ID resolution | Local paths sufficient for v0.1 |
| `monitor` command (continuous) | Needs Tier 1+ first |
| Community benchmark database | Post v0.1 launch |
| Multi-hardware data collection | Post v0.1 launch |
| mlx-lm integration as subcommand | Post v0.1 launch |
| Homebrew tap | Nice-to-have, not needed for v0.1 |
| Research artifact review & cleanup | Raw deep-research reports need digesting into structured, actionable docs; part of broader content processing pass |
| "What now?" — guided remediation after diagnosis | When mlx-triage surfaces a problem (bad quant, missing template, known bug), what should the user actually *do*? Explore: actionable next-step guidance, fix-it commands, alternative model suggestions, links to upstream issues. Could be a `--explain` flag, a remediation section in reports, or a `mlx-triage fix` subcommand. Speculative — needs design session. |
| VLM diagnostic support | Investigate whether mlx-triage can extend to Vision-Language Models (VLMs). Qwen3.5-4B is a VLM with `vision_tower` weights that `mlx-lm` can't load — Tier 1 checks fail entirely. Questions to scope: Can `mlx-vlm` be used as an alternative loader? What VLM-specific failure modes exist (vision encoder corruption, cross-attention misalignment, image preprocessing config)? Would VLM diagnostics need new check types (image input tests, multimodal determinism) or just a different model loader? Depends on Check 0.5 (VLM detection) shipping first. |

---

## Completed

| Item | Phase | Date | Notes |
|------|-------|------|-------|
| Research evidence collection | Pre-project | 2026-02-25 | 11 frontier model analyses, 2 systematic reviews |
| Phase 0 implementation | Phase 0 | 2026-02-25 | 54 tests, 4 checks, CLI + reports |
| v0.1 launch design | Phase 1 | 2026-02-28 | Approved design doc |
| Phase 1 implementation | Phase 1 | 2026-02-28 | 9 tasks, E2E validated on 13 models |
| Docs audit | Phase 1.5 | 2026-03-02 | Scrubbed 11 files, no AI attribution |
| Pre-launch bug audit | Phase 1.5 | 2026-03-02 | 15 bugs fixed, 102 tests |
| Launch packaging | Phase 1.5 | 2026-03-03 | README, CONTRIBUTING, METHODOLOGY, CI, issue templates, demo GIF |
| Documentation polish | Phase 1.5 | 2026-03-03 | Badges, example output, README improvements |
| GitHub publish | Phase 1.5b | 2026-03-03 | Repo live, CI green, v0.1.0 tagged and released |
| Presentation polish | Phase 1.5b | 2026-03-03 | Description, tags, README rewrite |


---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-25 | Tiered diagnostic protocol (0-3) | Most issues resolve at Tier 0/1; expensive diagnostics only when needed |
| 2026-02-25 | Python CLI (not web app) | Target audience is ML practitioners in terminals |
| 2026-02-25 | safetensors header reading for dtype check | Works without MLX installed; fastest path for Tier 0 |
| 2026-02-25 | Known bugs as YAML database | Community-contributable; tool gets smarter without code changes |
| 2026-02-28 | v0.1 = Tier 0 + Tier 1 + full packaging | Contribution-ready first impression |
| 2026-02-28 | GitHub + PyPI first, blog post fast follow | Get installable first, then announce |
| 2026-02-28 | Reference divergence as optional dep | Core differentiator; graceful skip if not installed |
| 2026-02-28 | Shared model loading in run_tier1() | Avoids 3x multi-GB load |
| 2026-03-03 | Publish to PyPI — YES | Both audit agents agree; README promises it; wheel builds clean |
| 2026-03-03 | Alpha classifier (not Beta) | More honest for v0.1 |
| 2026-03-03 | Plans moved to _private | Internal workflow language not suitable for public repo |
| 2026-03-03 | Validation trimmed to 13 models | Removed incomplete download and wrong model type — not real results |