# Current State — mlx-triage

_Last updated: 2026-02-28_

---

## Where We Are

**Phase: Phase 1 — Statistical Core — COMPLETE. Phase 1.5 (packaging) NEXT.**

Phase 0 (Tier 0) and Phase 1 (Tier 1) are fully implemented and E2E validated. 101 tests, all passing. Next: Phase 1.5 (launch packaging) + publish to PyPI.

---

## Completed This Session (2026-02-28, session 2)

Phase 1 implementation — 9 tasks completed via subagent-driven execution:

| Commit | Task | Tests Added |
|--------|------|-------------|
| `b929547` | Task 1: MLX knowledge corpus sync script + INDEX.md | — |
| `20b276a` | Task 3: Token comparison utilities | +12 |
| `5039e39` | Task 2: MLX inference utilities + pyproject.toml deps | +4 |
| `c4646c9` | Task 4: Diagnostic prompt suite (10 prompts + eval corpus) | +7 |
| `1e6d30c` | Task 5: Test 1.1 — Determinism check | +4 |
| `d54d38f` | Task 7: Test 1.3 — Quantization quality gate | +6 |
| `0d7ceed` | Task 6: Test 1.2 — Reference divergence | +5 |
| `9c01dad` | Tier 1 runner + CLI `--tier 1` wiring | +9 |
| `74288f7` | Fix: mlx_lm >=0.30 API compatibility (sampler, logprobs) | — |

E2E validation on real models:
- **Llama 3.2 1B Instruct 4-bit**: Tier 0 all PASS, determinism PASS, perplexity 3.6 PASS
- **Qwen3 0.6B 8-bit**: Tier 0 all PASS, determinism PASS, perplexity 3.6 PASS
- Reference divergence (1.2) SKIPs gracefully when torch not installed

### Previous sessions

- **2026-02-28 session 1**: v0.1 launch design approved (commit `18e9e52`)
- **2026-02-25**: Phase 0 implemented — 4 Tier 0 checks, 54 tests (commit `f271faf`)

---

## Immediate Next Step

> Start Phase 1.5 (launch packaging) — or dispatch to Codex if available.

Tasks (from roadmap):
1. README.md overhaul
2. METHODOLOGY.md
3. CONTRIBUTING.md
4. LICENSE (MIT)
5. pyproject.toml polish
6. GitHub Actions CI
7. Blog post outline

---

## Open Questions / Blockers

- **Check 0.4 false negative**: `check_mlx_version` reports "MLX is not installed" even when MLX is installed via `uv sync --extra mlx`. Likely an import path issue in the version check. Non-blocking but should be fixed before v0.1.
- **Shared model loading**: Checks currently load models independently. The `run_tier1()` shared loading optimization (load once, pass to checks) is deferred — not blocking v0.1 but should be done before adding Tier 2.
- **Codex availability**: Stream B (packaging) can start anytime.

---

## How to Verify

```bash
# Full test suite:
uv run pytest -v
# Expected: 101 tests, all passing in <1s

# Tier 0 only:
uv run mlx-triage check <model-path>

# Tier 0 + Tier 1 (requires uv sync --extra mlx):
uv run mlx-triage check <model-path> --tier 1

# JSON output:
uv run mlx-triage check <model-path> --tier 1 --format json
```

---

## Key Spec Notes / Corrections

- safetensors BF16: numpy doesn't support bfloat16 natively — test fixtures create binary headers directly
- MLX is an optional dependency — Tier 0 works without it (version check gracefully skips)
- `model_path` = local directory path, not a HuggingFace model ID (for v0.1)
- Version check: "all"-version bugs capped at WARNING to prevent always-CRITICAL
- Tier 1 checks currently take `(model_path)` and load internally — shared loading deferred
- **mlx_lm API**: v0.30+ uses `make_sampler(temp=...)` instead of `temp` kwarg; `response.logprobs` is `mx.array` of shape `(vocab_size,)`, not scalar

---

## Pickup Context

If resuming in a new session, start here:

- **First action**: Start Phase 1.5 (packaging) — see `docs/plans/2026-02-28-v01-launch-design.md` Stream B
- **Reference files**: Launch design at `docs/plans/2026-02-28-v01-launch-design.md`, roadmap at `docs/roadmap.md`
- **Current branch**: `main` at `74288f7`
- **Phase status**: Phase 0 COMPLETE. Phase 1 COMPLETE. Phase 1.5 NOT STARTED.
- **Test count**: 101 tests, all passing
