# Current State — mlx-triage

_Last updated: 2026-02-28_

---

## Where We Are

**Phase: Phase 1 — Statistical Core — PLANNING COMPLETE, IMPLEMENTATION NEXT**

Phase 0 (Tier 0) is complete. v0.1 launch design approved. Next: execute Phase 1 implementation plan, then Phase 1.5 (launch packaging), then publish to PyPI.

---

## Completed This Session (2026-02-28)

- Deep exploration of full repository: source code, tests, evidence base, research artifacts, context docs
- Brainstorming and design process for v0.1 launch strategy
- Decided: Option C (Tier 0 + Tier 1 + full packaging) for first public release
- Decided: Approach 2 (interleaved build + package) with Claude Code primary, Codex for packaging
- Decided: GitHub + PyPI first, blog post fast follow
- Decided: Reference divergence as optional dependency (`mlx-triage[reference]`)
- Decided: Shared model loading in `run_tier1()` (deviation from original Phase 1 plan)
- Decided: Context resilience via files + Sediment + server-memory + git until memory migration
- Wrote and committed v0.1 launch design: `docs/plans/2026-02-28-v01-launch-design.md` (commit `18e9e52`)

### Previous session (2026-02-25)

- Phase 0 fully implemented: 4 Tier 0 checks, 54 tests, CLI + reports (commit `f271faf`)
- Phase 1 implementation plan written: `docs/plans/2026-02-25-phase-1-statistical-core.md`

---

## Immediate Next Step

> Execute Phase 1 implementation plan, starting with Task 1 (MLX knowledge corpus).

Supporting context:
- Implementation plan: `docs/plans/2026-02-25-phase-1-statistical-core.md` (7 tasks, TDD)
- Launch design: `docs/plans/2026-02-28-v01-launch-design.md` (approved)
- Key deviation: Tier 1 check signatures use `(model, tokenizer, model_path)` not just `(model_path)`
- Stream B (Codex packaging) can start in parallel when Codex is freed up

---

## Open Questions / Blockers

- **MLX dependency**: Run `uv sync --extra mlx` before starting Phase 1 tasks that need inference
- **Reference models**: Need 2-3 known-good cached MLX models for end-to-end validation in S4
- **Codex availability**: Stream B (packaging) starts when Codex is freed from backend server work

---

## How to Verify

```bash
# Tier 0 still works:
uv run mlx-triage check <path-to-any-mlx-model-directory>
uv run pytest -v
# Expected: 54 tests, all passing in <1s
```

---

## Key Spec Notes / Corrections

- safetensors BF16: numpy doesn't support bfloat16 natively — test fixtures create binary headers directly
- MLX is an optional dependency — Tier 0 works without it (version check gracefully skips)
- `model_path` = local directory path, not a HuggingFace model ID (for v0.1)
- Version check: "all"-version bugs capped at WARNING to prevent always-CRITICAL
- **NEW**: Tier 1 checks take `(model, tokenizer, model_path)` — shared model loading in runner

---

## Pickup Context

If resuming in a new session, start here:

- **First action**: Execute Phase 1 Task 1 (MLX knowledge corpus) from `docs/plans/2026-02-25-phase-1-statistical-core.md`
- **Reference files**: Launch design at `docs/plans/2026-02-28-v01-launch-design.md`, Phase 1 plan at `docs/plans/2026-02-25-phase-1-statistical-core.md`
- **Current branch**: `main` at `18e9e52`
- **Phase status**: Phase 0 COMPLETE. Phase 1 design COMPLETE. Phase 1 implementation NOT STARTED.
