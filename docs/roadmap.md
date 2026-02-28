# Roadmap — mlx-triage

_Last updated: 2026-02-28_

> Items organized by priority tier (P0–P3). P0 = must have / blocking. P3 = backlog / future.

---

## Vision

A practitioner-facing diagnostic CLI that tells you whether your MLX model quality issue is infrastructure or model — in under 30 seconds for the common case, with escalating depth for harder problems.

---

## P0 — Phase 0: Scaffolding & Tier 0 (COMPLETE)

- [x] Project scaffolding (pyproject.toml, src layout, uv sync)
- [x] Core data models (CheckStatus, DiagnosticResult, TierReport)
- [x] Known bugs database (YAML + config loader)
- [x] CLI skeleton (Click: check, report, compare)
- [x] Check 0.1: Dtype compatibility audit
- [x] Check 0.2: Tokenizer & EOS configuration audit
- [x] Check 0.3: Weight file integrity
- [x] Check 0.4: MLX version & known bug check
- [x] Tier 0 runner (orchestrate all checks)
- [x] Report generation (JSON + Rich terminal)
- [x] CLI integration (end-to-end wiring)
- [x] Project docs update

**Ship criteria:** `mlx-triage check <any-mlx-model>` produces a Tier 0 report in < 30 seconds. **MET.**

**Plan:** `docs/plans/2026-02-25-phase-0-scaffolding.md`

---

## P1 — Phase 1: Statistical Core (Tier 1) — IN PROGRESS

**Design:** `docs/plans/2026-02-28-v01-launch-design.md` (approved 2026-02-28)
**Implementation plan:** `docs/plans/2026-02-25-phase-1-statistical-core.md`

- [ ] Task 1: MLX knowledge corpus (sync script)
- [ ] Task 2: MLX inference utilities (`mlx_utils.py`)
- [ ] Task 3: Token comparison utilities (`comparison.py`)
- [ ] Task 4: Diagnostic prompt suite (10 prompts + eval corpus)
- [ ] Task 5: Test 1.1 — Determinism check (N runs at temp=0)
- [ ] Task 6: Test 1.2 — Reference divergence (MLX vs Transformers)
- [ ] Task 7: Test 1.3 — Quantization quality gate (perplexity)
- [ ] Tier 1 runner + CLI `--tier 1` wiring
- [ ] End-to-end validation on real models (M2 Max)

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

## P1.5 — Phase 1.5: Launch Packaging (parallel with Phase 1)

**Agent:** Codex (when available)
**Design:** `docs/plans/2026-02-28-v01-launch-design.md`

- [ ] README.md overhaul (install, usage, architecture, evidence, badges)
- [ ] METHODOLOGY.md (evidence citations, tiered protocol, research gaps)
- [ ] CONTRIBUTING.md (known bugs YAML, check contribution guide, TDD)
- [ ] LICENSE file (MIT)
- [ ] pyproject.toml polish (classifiers, URLs, long description)
- [ ] GitHub Actions CI (Python 3.11-3.13, macOS)
- [ ] Blog post outline (`docs/plans/blog-post-outline.md`)

**Ship criteria:** All docs in repo, CI green, `pip install mlx-triage` works.

---

## P2 — Phase 1.5b: Publish v0.1

- [ ] Merge packaging + implementation
- [ ] Full test suite green
- [ ] `uv build` + PyPI publish
- [ ] Git tag `v0.1.0`
- [ ] Push to GitHub (public)
- [ ] Blog post (fast follow)

**Ship criteria:** `pip install mlx-triage` works from clean venv. Blog post live.

---

## P3 — Phase 2: Isolation Tests (Tier 2) — Post v0.1

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
| Workshop paper | Post v0.1 launch |
| Homebrew tap | Nice-to-have, not needed for v0.1 |

---

## Completed

| Item | Phase | Date | Notes |
|------|-------|------|-------|
| Research evidence collection | Pre-project | 2026-02-25 | 11 frontier model analyses, 2 systematic reviews |
| PROJECT-SPEC.md | Pre-project | 2026-02-25 | Full spec with tiered diagnostic protocol |
| Phase 0 implementation plan | Phase 0 | 2026-02-25 | 12 tasks, TDD approach |
| Phase 0 implementation | Phase 0 | 2026-02-25 | 54 tests, 4 checks, CLI + reports. Commit f271faf |
| Phase 1 implementation plan | Phase 1 | 2026-02-25 | 7 tasks, TDD. `docs/plans/2026-02-25-phase-1-statistical-core.md` |
| v0.1 launch design | Phase 1 | 2026-02-28 | Approved. `docs/plans/2026-02-28-v01-launch-design.md`. Commit 18e9e52 |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-25 | Tiered diagnostic protocol (0-3) | Matches evidence: most issues resolve at Tier 0/1; expensive deep diagnostics only when needed |
| 2026-02-25 | Python CLI (not web app) | Target audience is ML practitioners who work in terminals |
| 2026-02-25 | safetensors header reading for dtype check | Works without MLX installed; fastest path for Tier 0 |
| 2026-02-25 | Known bugs as YAML database | Community-contributable; tool gets smarter without code changes |
| 2026-02-25 | TDD approach with pytest | Every check function has tests; enables confident refactoring |
| 2026-02-28 | v0.1 = Tier 0 + Tier 1 + full packaging (Option C) | Contribution-ready first impression for flywheel |
| 2026-02-28 | GitHub + PyPI first, blog post fast follow | Get installable, then drive traffic |
| 2026-02-28 | Claude Code primary, Codex for packaging | Matches multi-agent ecosystem; memo checkpoints for big decisions |
| 2026-02-28 | Reference divergence as optional dep | Core differentiator; graceful skip if not installed |
| 2026-02-28 | Shared model loading in run_tier1() | Checks take (model, tokenizer, model_path); avoids 3x multi-GB load |
| 2026-02-28 | Context resilience: files + Sediment + server-memory + git | Belt and suspenders until Falcor DB memory migration |

---

## Specialist Consultation Guide

| Feature area | Primary approach | Notes |
|-------------|-----------------|-------|
| Tier 0 checks | Subagent-driven TDD | Independent modules, parallel-safe |
| Tier 1 (inference) | Sequential, needs MLX | Requires actual model loading |
| Tier 3 (activation) | Research-heavy | May need MLX internals expertise |

See `docs/agents.md` for full agent orchestration guide.
