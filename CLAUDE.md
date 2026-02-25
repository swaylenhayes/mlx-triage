# CLAUDE.md — mlx-triage

> Project-specific context only. Global settings live in ~/.claude/CLAUDE.md.

---

## Project Overview

**Name**: mlx-triage
**Type**: Python CLI tool
**Status**: Active development — Phase 0 (scaffolding)
**Primary language(s)**: Python 3.11+

A tiered diagnostic CLI that runs checks against MLX-served models on Apple Silicon to determine whether quality issues stem from the model weights or the inference infrastructure.

---

## Stack & Key Dependencies

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | Python 3.11+ | Type hints required |
| CLI | Click >= 8.1 | Command routing |
| Terminal UI | Rich >= 13.0 | Tables, progress bars, colored status |
| Config | PyYAML >= 6.0 | Known bugs database |
| Weights | safetensors >= 0.4 | Header reading, weight inspection |
| Numerics | numpy >= 1.26 | Weight statistics |
| ML (optional) | mlx, mlx-lm | Required for Tier 1+ (inference) |
| Build / Package | uv + hatchling | src layout |
| Deploy target | macOS / Apple Silicon (M1–M4) | M2 Max 96GB is dev target |

---

## Project Layout

```
mlx-triage/
├── CLAUDE.md
├── PROJECT-SPEC.md              # Full spec with evidence basis
├── pyproject.toml
├── src/mlx_triage/
│   ├── __init__.py
│   ├── cli.py                   # Click CLI entry point
│   ├── config.py                # Known bugs database loader
│   ├── models.py                # Core data models
│   ├── report.py                # Report generation (JSON, terminal)
│   ├── tier0/                   # Sanity checks (no inference)
│   ├── tier1/                   # Statistical smoke tests (Phase 1)
│   ├── tier2/                   # Isolation tests (Phase 2)
│   ├── tier3/                   # Deep diagnostics (Phase 3)
│   ├── prompts/                 # Diagnostic prompt suites
│   ├── utils/                   # Shared utilities
│   └── data/
│       └── known_bugs.yaml      # Curated MLX bug database
├── tests/
├── docs/
│   ├── plans/                   # Implementation plans
│   ├── research/                # C2 research artifacts
│   ├── evidence/                # Primary source evidence
│   └── context/                 # Context notes from spec creation
```

Key files to know:
- `PROJECT-SPEC.md` — the full spec with architecture, evidence trail, and phased delivery
- `src/mlx_triage/data/known_bugs.yaml` — curated database of MLX bugs (the tool's institutional knowledge)
- `docs/plans/` — implementation plans per phase

---

## How to Run

```bash
# Install / setup
uv sync --extra dev

# Run CLI
uv run mlx-triage check <model_path>
uv run mlx-triage check <model_path> --format json
uv run mlx-triage check <model_path> --output report.json --format json

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=mlx_triage -v
```

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| safetensors header reading (not MLX load) for Tier 0 | Tier 0 should work without MLX installed; direct binary header read is fastest |
| hatchling build backend with src layout | Standard Python packaging, clean import paths, uv-compatible |
| MLX as optional dependency | Tier 0 runs without MLX; Tier 1+ needs it for inference |
| Known bugs as YAML (not Python dict) | Non-developers can contribute; tool gets smarter without code changes |
| TierReport uses computed properties (not stored verdict) | Verdict always reflects current checks; no stale state |

---

## Project Conventions

- TDD: Write failing test first, then implement, then verify. Every check has tests.
- Each tier is an independent module with clear inputs (model_path) and outputs (DiagnosticResult)
- Check functions return `DiagnosticResult`, never raise exceptions for expected conditions
- `CheckStatus` severity ordering: PASS < INFO < SKIP < WARNING < FAIL < CRITICAL
- Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, `test:`)

---

## Known Gotchas

- safetensors BF16 dtype: numpy doesn't support bfloat16 natively. For test fixtures, create safetensors manually via binary header writing (see `tests/conftest.py`)
- MLX version check needs `import mlx` — must be mocked in tests when MLX isn't installed
- `model_path` currently means a local directory path (not a HuggingFace model ID)

---

## Agents for This Project

See `docs/agents.md` for full orchestration guide.

| Task type | Agent |
|-----------|-------|
| Check implementation (TDD) | feature-dev (subagent-driven) |
| Code review after each task | code-reviewer |
| Codebase exploration | code-explorer |

---

## Out of Scope

- HuggingFace model ID resolution (local paths only for v0.1)
- GPU inference benchmarking (this is a diagnostic tool, not a benchmark)
- Support for non-MLX frameworks (PyTorch, TensorFlow)
- Web UI or API server (CLI only)

---

*Last updated: 2026-02-25*
