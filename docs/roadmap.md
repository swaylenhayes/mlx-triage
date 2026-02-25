# Roadmap — mlx-triage

_Last updated: 2026-02-25_

> Items organized by priority tier (P0–P3). P0 = must have / blocking. P3 = backlog / future.

---

## Vision

A practitioner-facing diagnostic CLI that tells you whether your MLX model quality issue is infrastructure or model — in under 30 seconds for the common case, with escalating depth for harder problems.

---

## P0 — Phase 0: Scaffolding & Tier 0 (Current)

- [ ] Project scaffolding (pyproject.toml, src layout, uv sync)
- [ ] Core data models (CheckStatus, DiagnosticResult, TierReport)
- [ ] Known bugs database (YAML + config loader)
- [ ] CLI skeleton (Click: check, report, compare)
- [ ] Check 0.1: Dtype compatibility audit
- [ ] Check 0.2: Tokenizer & EOS configuration audit
- [ ] Check 0.3: Weight file integrity
- [ ] Check 0.4: MLX version & known bug check
- [ ] Tier 0 runner (orchestrate all checks)
- [ ] Report generation (JSON + Rich terminal)
- [ ] CLI integration (end-to-end wiring)
- [ ] Project docs update

**Ship criteria:** `mlx-triage check <any-mlx-model>` produces a Tier 0 report in < 30 seconds.

**Plan:** `docs/plans/2026-02-25-phase-0-scaffolding.md`

---

## P1 — Phase 1: Statistical Core (Tier 1)

- [ ] Diagnostic prompt suite (10 diverse prompts covering math, code, long-context, structured output, multi-turn)
- [ ] Test 1.1: Determinism check (N runs at temp=0, token-by-token comparison)
- [ ] Test 1.2: Reference divergence (MLX vs Transformers, Token-DiFR lite)
- [ ] Test 1.3: Quantization quality gate (perplexity on WikiText-2 subset)
- [ ] Threshold calibration against 3-5 known-good models on M2 Max
- [ ] Tier 1 runner + report integration

**Ship criteria:** `mlx-triage check <model> --tier 1` correctly identifies known-good as PASS and deliberately misconfigured as FAIL.

---

## P2 — Phase 2: Isolation Tests (Tier 2)

- [ ] Test 2.1: Batch invariance (single vs batched output comparison)
- [ ] Test 2.2: Memory pressure sweep (MLX Metal memory monitoring)
- [ ] Test 2.3: Context length stress (needle-in-a-haystack at multiple lengths)
- [ ] Diagnostic divergence heuristic (crash = infra, gradual degradation = model)
- [ ] Tier 2 runner + report integration

**Ship criteria:** `mlx-triage check <model> --tier 2` produces full T0+T1+T2 diagnostic report.

---

## P3 — Phase 3: Deep Diagnostics & Contribution (Tier 3)

- [ ] Test 3.1: Layer-wise activation comparison (MLX vs reference)
- [ ] Test 3.2: Cross-runtime comparison (MLX vs llama.cpp)
- [ ] Error accumulation visualization
- [ ] METHODOLOGY.md with full evidence citations
- [ ] PyPI packaging (`pip install mlx-triage`)
- [ ] Blog post / technical report

**Ship criteria:** Full Tier 0-3 coverage, 5+ case studies, published to PyPI.

---

## Deferred

| Item | Reason deferred |
|------|----------------|
| HuggingFace model ID resolution | Local paths sufficient for v0.1 |
| `monitor` command (continuous) | Needs Tier 1+ first |
| Community benchmark database | Phase 4 (post-launch) |
| Multi-hardware data collection | Phase 4 |
| mlx-lm integration as subcommand | Phase 4 |
| Workshop paper | Phase 4 |

---

## Completed

| Item | Phase | Date | Notes |
|------|-------|------|-------|
| Research evidence collection | Pre-project | 2026-02-25 | 11 frontier model analyses, 2 systematic reviews |
| PROJECT-SPEC.md | Pre-project | 2026-02-25 | Full spec with tiered diagnostic protocol |
| Phase 0 implementation plan | Phase 0 | 2026-02-25 | 12 tasks, 49 tests, TDD approach |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-25 | Tiered diagnostic protocol (0-3) | Matches evidence: most issues resolve at Tier 0/1; expensive deep diagnostics only when needed |
| 2026-02-25 | Python CLI (not web app) | Target audience is ML practitioners who work in terminals |
| 2026-02-25 | safetensors header reading for dtype check | Works without MLX installed; fastest path for Tier 0 |
| 2026-02-25 | Known bugs as YAML database | Community-contributable; tool gets smarter without code changes |
| 2026-02-25 | TDD approach with pytest | Every check function has tests; enables confident refactoring |

---

## Specialist Consultation Guide

| Feature area | Primary approach | Notes |
|-------------|-----------------|-------|
| Tier 0 checks | Subagent-driven TDD | Independent modules, parallel-safe |
| Tier 1 (inference) | Sequential, needs MLX | Requires actual model loading |
| Tier 3 (activation) | Research-heavy | May need MLX internals expertise |

See `docs/agents.md` for full agent orchestration guide.
