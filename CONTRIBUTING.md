# Contributing to mlx-triage

Thanks for your interest in contributing. Here's how to get involved.

## Report a Known MLX Bug

The easiest and highest-impact contribution is adding to the known bugs database.

**File:** [`src/mlx_triage/data/known_bugs.yaml`](src/mlx_triage/data/known_bugs.yaml)

Each entry follows this format:

```yaml
- id: MLX-NNN
  title: "Short description of the bug"
  mlx_issue: 1234          # GitHub issue number, or null
  source: "optional source" # If no GitHub issue
  affected_versions: ["< 0.24.0"]  # or ["all"]
  severity: critical        # critical | high | warning
  detection: "tier0.version_check"  # Which check catches this
  symptom: "What the user sees when this bug bites them"
  architecture: ["llama", "mistral"]  # or ["all"]
  remediation: "What the user should do"
```

Open a PR with your addition. Include a link to the source (GitHub issue, postmortem, etc.).

## Add a Diagnostic Check

All checks follow the same pattern:

1. A check function in `src/mlx_triage/tierN/` that returns `DiagnosticResult`
2. Tests in `tests/test_tierN_*.py`
3. Wire into the tier runner (`src/mlx_triage/tierN/__init__.py`)

### Development Setup

```bash
git clone https://github.com/swaylenhayes/mlx-triage.git
cd mlx-triage
uv sync --extra dev
uv run pytest -v  # Should see 102 tests pass
```

### TDD Workflow

This project follows test-driven development:

1. Write a failing test in `tests/`
2. Run it to confirm it fails: `uv run pytest tests/test_file.py::test_name -v`
3. Write the minimal implementation to make it pass
4. Run the full suite: `uv run pytest -v`
5. Commit with a conventional commit message (`feat:`, `fix:`, `test:`, `docs:`)

### Check Function Contract

Every check function must:

- Return `DiagnosticResult` (never raise exceptions for expected conditions)
- Use `CheckStatus` severity levels: PASS < INFO < SKIP < WARNING < FAIL < CRITICAL
- Include `remediation` text for any status above PASS
- Include a `check_id` (e.g., `"0.1"`) and human-readable `name`

```python
from mlx_triage.models import CheckStatus, DiagnosticResult

def check_something(model_path: str) -> DiagnosticResult:
    # ... inspection logic ...
    return DiagnosticResult(
        check_id="0.5",
        name="Something Check",
        status=CheckStatus.PASS,
        detail="Everything looks good.",
    )
```

## Code Style

- Python 3.11+ with type hints
- No external linter enforced yet -- keep it clean and consistent with existing code
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`

## Questions?

Open an issue. We're friendly.
