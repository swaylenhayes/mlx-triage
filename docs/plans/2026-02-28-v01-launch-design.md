# mlx-triage v0.1 Launch Design

> **Approved:** 2026-02-28
> **Scope:** Phase 1 (Statistical Core) + Phase 1.5 (Launch Packaging) → PyPI publish + blog post

---

## Summary

Ship mlx-triage v0.1 as a contribution-ready open-source tool on PyPI with full documentation, CI, and a blog post. The release includes Tier 0 (sanity checks, already complete) and Tier 1 (statistical smoke tests: determinism, reference divergence, quantization quality gate). The goal is a strong first impression that establishes credibility and starts a flywheel of interest, usage, and contributions in the Apple Silicon ML research community.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| First release scope | Tier 0 + Tier 1 + full packaging (Option C) | Contribution-ready first impression for flywheel |
| Distribution | GitHub + PyPI first, blog post fast follow | Get installable, then drive traffic |
| Workstream model | Primary and parallel workstreams with checkpoints | Matches existing multi-workstream workflow |
| Phase 1 plan | Execute existing plan as-is + add packaging phase | Plan is solid, no rework needed |
| Reference divergence | Include as optional dependency (`mlx-triage[reference]`) | Core differentiator, graceful skip if not installed |
| Model loading | Shared load in `run_tier1()`, pass tuple to checks | Avoids loading multi-GB model 3x |
| Context resilience | Files + git at every task boundary | Multiple redundant state preservation mechanisms |

---

## Architecture

### What stays the same from Phase 0

- DiagnosticResult → TierReport → Report renderer pipeline
- Check functions return DiagnosticResult, never raise for expected conditions
- Known bugs YAML database as institutional knowledge
- Click CLI with lazy imports
- TDD discipline: test first, implement, verify, commit

### What gets added

```
src/mlx_triage/
├── utils/
│   ├── mlx_utils.py          ← NEW: model loading, generate_text(), check_mlx_available()
│   └── comparison.py          ← NEW: token_agreement_rate(), divergence_point(), multi_run_consistency()
├── prompts/
│   └── standard_suite.py      ← NEW: 10 diagnostic prompts + eval corpus
├── tier1/
│   ├── __init__.py            ← MODIFY: add run_tier1() orchestrator
│   ├── determinism.py         ← NEW: Test 1.1
│   ├── reference_divergence.py ← NEW: Test 1.2
│   └── quantization_quality.py ← NEW: Test 1.3
├── cli.py                     ← MODIFY: wire --tier parameter
└── report.py                  ← MODIFY: handle Tier 1 results
```

### Model loading lifecycle

Tier 1 loads the model once in `run_tier1()` and passes `(model, tokenizer)` to each check. Checks do not load models themselves.

```python
def run_tier1(model_path: str) -> TierReport:
    model, tokenizer = load_model(model_path)
    checks = [
        check_determinism(model, tokenizer, model_path),
        check_reference_divergence(model, tokenizer, model_path),
        check_quantization_quality(model, tokenizer, model_path),
    ]
    return TierReport.create(tier=1, model=model_path, checks=checks)
```

**Deviation from existing Phase 1 plan:** Check function signatures change from `(model_path)` to `(model, tokenizer, model_path)`. This avoids loading a multi-GB model 3 times.

### CLI --tier flag behavior

```
mlx-triage check <path> --tier 0  →  run_tier0(path) only
mlx-triage check <path> --tier 1  →  run_tier0(path) + run_tier1(path)
mlx-triage check <path>           →  run_tier0(path) (default, unchanged)
```

Tier 1 always runs Tier 0 first. If Tier 0 returns CRITICAL, Tier 1 is skipped: "Tier 0 found critical issues. Fix those before running Tier 1."

### Optional dependency gating

```
mlx-triage[mlx]        →  unlocks Tier 1 (determinism, quantization quality)
mlx-triage[reference]  →  unlocks Test 1.2 (reference divergence via transformers)
mlx-triage[dev]        →  pytest, coverage
```

Missing dependencies → `CheckStatus.SKIP` with remediation instructions.

---

## Work Streams

### Stream A: Implementation

Executes the existing Phase 1 plan (`docs/plans/2026-02-25-phase-1-statistical-core.md`) task-by-task with TDD, plus integration work.

| Session | Tasks | Ship criteria |
|---------|-------|--------------|
| S1 | Task 1: MLX knowledge corpus | `grep -r "stream_generate" _docs/mlx-knowledge/` returns hits |
| | Task 2: MLX inference utilities | `uv run pytest tests/test_mlx_utils.py -v` passes |
| | Task 3: Token comparison utilities | `uv run pytest tests/test_comparison.py -v` passes |
| S2 | Task 4: Diagnostic prompt suite | `uv run pytest tests/test_prompts.py -v` passes |
| | Task 5: Test 1.1 Determinism check | `uv run pytest tests/test_tier1_determinism.py -v` passes |
| S3 | Task 6: Test 1.2 Reference divergence | `uv run pytest tests/test_tier1_divergence.py -v` passes |
| | Task 7: Test 1.3 Quantization quality gate | `uv run pytest tests/test_tier1_quantization.py -v` passes |
| S4 | Tier 1 runner + CLI wiring | `uv run mlx-triage check <model> --tier 1` produces Tier 0+1 report |
| | End-to-end on real models (M2 Max) | Quantized model: determinism PASS. Meaningful divergence report. |

### Stream B: Packaging (parallel)

| Task | Ship criteria |
|------|--------------|
| README.md overhaul | Install, usage, architecture diagram, evidence summary, badges |
| METHODOLOGY.md | Evidence citations, tiered protocol explanation, research gaps |
| CONTRIBUTING.md | Known bugs YAML format, check contribution guide, TDD convention |
| LICENSE file | MIT text |
| pyproject.toml polish | Classifiers, URLs, long description |
| GitHub Actions CI | Tests pass on push/PR, Python 3.11-3.13, macOS |
| Blog post outline | Saved as `docs/plans/blog-post-outline.md` |

### Stream D: Publish

| Task | Ship criteria |
|------|--------------|
| Merge Stream A + B | All docs in repo, CI green |
| Full test suite | `uv run pytest -v` — all pass |
| Build + publish | `pip install mlx-triage` works from clean venv |
| Git tag | `v0.1.0` |
| Push to GitHub (public) | Repo visible, README renders, CI green |
| State updates | `current-state.md`, `roadmap.md` |

### Fast Follow: Blog Post

| Task | Venue |
|------|-------|
| Write from outline | dev.to or personal site |
| Cross-post | MLX GitHub discussions, Hacker News, relevant Discord |

---

## Context Resilience Protocol

Every task boundary includes a save checkpoint:

**After each completed task:**
1. Git commit (conventional commits)
2. Update `docs/current-state.md`
3. Update `docs/roadmap.md`

**At session boundaries (all of above plus):**
4. Close-out summary in conversation

**Recovery contract** (for any new session or crash recovery):
```
1. Read docs/current-state.md     ← where are we, what's next
2. Read docs/roadmap.md           ← what's done, what's left
3. git log --oneline -10          ← what was actually committed
```

---

## Future Phases (Post v0.1)

| Phase | What | When |
|-------|------|------|
| v0.2 — Tier 2 | Batch invariance, memory pressure sweep, context length stress | After community feedback |
| v0.3 — Tier 3 | Layer-wise activation comparison, cross-runtime (llama.cpp) | When there's demand |
| v1.0 — Community | HuggingFace model ID resolution, community benchmark DB, multi-hardware data | Flywheel spinning |

---

_Design approved 2026-02-28. Next step: implementation plan via writing-plans skill._
