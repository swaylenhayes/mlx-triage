# Current State — mlx-triage

_Last updated: 2026-02-25_

---

## Where We Are

**Phase: Phase 0 — Scaffolding & Tier 0 — COMPLETE**

Full CLI tool with 4 Tier 0 diagnostic checks, Rich terminal + JSON output, and a curated known-bugs YAML database. Verified against a real MLX model (chatterbox-turbo-fp16). 54 tests, all passing.

---

## Completed This Session

- Created Phase 0 implementation plan (12 tasks, TDD)
- Built project scaffolding (pyproject.toml, src layout, uv sync)
- Implemented core data models (CheckStatus, DiagnosticResult, TierReport)
- Created known-bugs YAML database (6 curated MLX bugs from evidence base)
- Implemented 4 Tier 0 checks in parallel via subagents:
  - 0.1: Dtype compatibility (Gemma 3 FP16 overflow pattern)
  - 0.2: Tokenizer & EOS config (Ollama template disaster pattern)
  - 0.3: Weight integrity (baa.ai BF16 corruption pattern)
  - 0.4: MLX version & known bug check
- Built Tier 0 runner, report generation (JSON + Rich), CLI integration
- Manual verification against real cached MLX model
- Updated all project docs (CLAUDE.md, roadmap, agents)
- Committed as `f271faf` on main

---

## Immediate Next Step

> Begin Phase 1: Statistical Core (Tier 1).

Supporting context:
- Phase 1 needs: diagnostic prompt suite, determinism check, reference divergence, quantization quality gate
- Requires MLX installed (Tier 1 runs inference)
- Plan Phase 1 before implementing — create `docs/plans/YYYY-MM-DD-phase-1-statistical-core.md`
- See PROJECT-SPEC.md "Phase 1: Statistical Core" section for full details

---

## Open Questions / Blockers

- **MLX dependency**: Tier 1 requires MLX for inference. Install in dev environment before starting Phase 1.
- **Reference models**: Need 3-5 known-good models for threshold calibration (see PROJECT-SPEC.md Phase 1 tasks)

---

## How to Verify

```bash
# Tier 0 works:
uv run mlx-triage check <path-to-any-mlx-model-directory>
# Expected: Rich terminal output with 4 diagnostic checks

uv run mlx-triage check <path> --format json
# Expected: JSON report with checks, verdict, should_continue

uv run pytest -v
# Expected: 54 tests, all passing in <1s
```

---

## Key Spec Notes / Corrections

- safetensors BF16: numpy doesn't support bfloat16 natively — test fixtures create binary headers directly
- MLX is an optional dependency — Tier 0 works without it (version check gracefully skips)
- `model_path` = local directory path, not a HuggingFace model ID (for v0.1)
- Version check agent added smart severity handling: "all"-version bugs capped at WARNING to prevent always-CRITICAL

---

## Pickup Context

If resuming in a new session, start here:

- **First action**: Plan Phase 1 (write `docs/plans/` document)
- **Reference files**: `PROJECT-SPEC.md` Phase 1 section, `docs/plans/2026-02-25-phase-0-scaffolding.md`
- **Current branch**: `main` at `f271faf`
- **Phase status**: Phase 0 COMPLETE. Phase 1 not started.
