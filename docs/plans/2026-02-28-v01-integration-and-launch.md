# v0.1 Integration, Packaging & Launch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire Tier 1 checks into the CLI, add launch packaging (README, METHODOLOGY, CI, PyPI), and publish mlx-triage v0.1.0 as a contribution-ready open-source tool.

**Architecture:** This plan picks up after Phase 1 Tasks 1-7 (from `docs/plans/2026-02-25-phase-1-statistical-core.md`). It adds the Tier 1 runner that orchestrates checks with shared model loading, wires the `--tier` CLI flag, creates all packaging artifacts, and publishes to PyPI. Stream A (Tasks 8-11) runs in Claude Code. Stream B (Tasks P1-P6) runs in Codex and can execute in parallel with Stream A.

**Tech Stack:** Existing Phase 0+1 stack. New: GitHub Actions, `twine`/`uv publish` for PyPI.

**Prerequisites:** Phase 1 Tasks 1-7 complete (all Tier 1 checks implemented and tested).

**Design doc:** `docs/plans/2026-02-28-v01-launch-design.md`

**IMPORTANT — Signature deviation from Phase 1 plan:** The existing Phase 1 plan has Tier 1 checks taking `(model_path)` and loading models internally. Per the approved design, Tier 1 checks MUST take `(model, tokenizer, model_path)` instead. The runner loads the model once and passes it to each check. If Phase 1 Tasks 5-7 were implemented with the old signature, update them in Task 8.

---

## Stream A: Integration (Claude Code)

### Task 8: Tier 1 Runner

**Files:**
- Create: `src/mlx_triage/tier1/__init__.py` (replace empty init)
- Create: `tests/test_tier1_runner.py`

**Step 1: Write the failing test**

```python
# tests/test_tier1_runner.py
"""Tests for Tier 1 runner — orchestrates all Tier 1 checks."""

from unittest.mock import patch, MagicMock

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.tier1 import run_tier1


def _mock_result(check_id: str, name: str, status: CheckStatus) -> DiagnosticResult:
    return DiagnosticResult(
        check_id=check_id, name=name, status=status, detail="Mock detail."
    )


def test_run_tier1_returns_tier_report():
    """run_tier1 should return a TierReport with tier=1."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("mlx_triage.tier1.check_mlx_available", return_value=True), \
         patch("mlx_triage.tier1.load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("mlx_triage.tier1.check_determinism", return_value=_mock_result("1.1", "Determinism", CheckStatus.PASS)), \
         patch("mlx_triage.tier1.check_reference_divergence", return_value=_mock_result("1.2", "Reference Divergence", CheckStatus.PASS)), \
         patch("mlx_triage.tier1.check_quantization_quality", return_value=_mock_result("1.3", "Quantization Quality", CheckStatus.PASS)):
        report = run_tier1("/fake/model")

    assert isinstance(report, TierReport)
    assert report.tier == 1
    assert len(report.checks) == 3
    assert report.worst_status == CheckStatus.PASS


def test_run_tier1_passes_model_to_checks():
    """Checks should receive the loaded model and tokenizer, not load their own."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("mlx_triage.tier1.check_mlx_available", return_value=True), \
         patch("mlx_triage.tier1.load_model", return_value=(mock_model, mock_tokenizer)) as mock_load, \
         patch("mlx_triage.tier1.check_determinism", return_value=_mock_result("1.1", "Determinism", CheckStatus.PASS)) as mock_det, \
         patch("mlx_triage.tier1.check_reference_divergence", return_value=_mock_result("1.2", "Reference Divergence", CheckStatus.PASS)) as mock_ref, \
         patch("mlx_triage.tier1.check_quantization_quality", return_value=_mock_result("1.3", "Quantization Quality", CheckStatus.PASS)) as mock_quant:
        run_tier1("/fake/model")

    # Model loaded exactly once
    mock_load.assert_called_once_with("/fake/model")
    # Each check receives model + tokenizer
    mock_det.assert_called_once_with(mock_model, mock_tokenizer, "/fake/model")
    mock_ref.assert_called_once_with(mock_model, mock_tokenizer, "/fake/model")
    mock_quant.assert_called_once_with(mock_model, mock_tokenizer, "/fake/model")


def test_run_tier1_skips_when_mlx_unavailable():
    """When MLX is not installed, return a report with all SKIP checks."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=False):
        report = run_tier1("/fake/model")

    assert report.tier == 1
    assert len(report.checks) == 1
    assert report.checks[0].status == CheckStatus.SKIP
    assert "MLX" in report.checks[0].detail


def test_run_tier1_handles_load_failure():
    """If model loading fails, return CRITICAL."""
    with patch("mlx_triage.tier1.check_mlx_available", return_value=True), \
         patch("mlx_triage.tier1.load_model", side_effect=Exception("Model not found")):
        report = run_tier1("/fake/model")

    assert report.tier == 1
    assert len(report.checks) == 1
    assert report.checks[0].status == CheckStatus.CRITICAL
    assert "load" in report.checks[0].detail.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier1_runner.py -v`
Expected: FAIL — `ImportError` (run_tier1 not defined)

**Step 3: Implement the Tier 1 runner**

```python
# src/mlx_triage/tier1/__init__.py
"""Tier 1: Statistical Smoke Tests — requires MLX, ~100 inference calls."""

from __future__ import annotations

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.utils.mlx_utils import check_mlx_available, load_model
from mlx_triage.tier1.determinism import check_determinism
from mlx_triage.tier1.reference_divergence import check_reference_divergence
from mlx_triage.tier1.quantization_quality import check_quantization_quality


def run_tier1(model_path: str) -> TierReport:
    """Run all Tier 1 checks with shared model loading.

    Loads the model once and passes (model, tokenizer) to each check.
    Returns a TierReport with tier=1.
    """
    if not check_mlx_available():
        return TierReport.create(
            tier=1,
            model=model_path,
            checks=[
                DiagnosticResult(
                    check_id="1.0",
                    name="Tier 1 Prerequisites",
                    status=CheckStatus.SKIP,
                    detail="MLX is not installed. Cannot run Tier 1 checks.",
                    remediation="Install MLX: uv sync --extra mlx",
                )
            ],
        )

    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        return TierReport.create(
            tier=1,
            model=model_path,
            checks=[
                DiagnosticResult(
                    check_id="1.0",
                    name="Model Loading",
                    status=CheckStatus.CRITICAL,
                    detail=f"Failed to load model: {e}",
                    remediation="Verify model path and file integrity. Run Tier 0 checks first.",
                )
            ],
        )

    checks = [
        check_determinism(model, tokenizer, model_path),
        check_reference_divergence(model, tokenizer, model_path),
        check_quantization_quality(model, tokenizer, model_path),
    ]
    return TierReport.create(tier=1, model=model_path, checks=checks)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier1_runner.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier1/__init__.py tests/test_tier1_runner.py
git commit -m "feat: add Tier 1 runner with shared model loading"
```

---

### Task 9: Wire --tier Flag in CLI

**Files:**
- Modify: `src/mlx_triage/cli.py`
- Create: `tests/test_cli_tier1.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_tier1.py
"""Tests for --tier flag wiring in CLI."""

from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from mlx_triage.cli import cli
from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport


def _mock_tier_report(tier: int) -> TierReport:
    return TierReport.create(
        tier=tier,
        model="/fake/model",
        checks=[
            DiagnosticResult(
                check_id=f"{tier}.1",
                name="Mock Check",
                status=CheckStatus.PASS,
                detail="All good.",
            )
        ],
    )


def test_tier0_is_default(tmp_path):
    """Default --tier is 0, runs only Tier 0."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"torch_dtype": "float32"}')

    with patch("mlx_triage.cli.run_tier0", return_value=_mock_tier_report(0)) as mock_t0:
        runner = CliRunner()
        result = runner.invoke(cli, ["check", str(model_dir)])

    mock_t0.assert_called_once()
    assert result.exit_code == 0


def test_tier1_runs_both_tiers(tmp_path):
    """--tier 1 runs Tier 0 first, then Tier 1."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"torch_dtype": "float32"}')

    t0_report = _mock_tier_report(0)
    t1_report = _mock_tier_report(1)

    with patch("mlx_triage.cli.run_tier0", return_value=t0_report) as mock_t0, \
         patch("mlx_triage.cli.run_tier1", return_value=t1_report) as mock_t1:
        runner = CliRunner()
        result = runner.invoke(cli, ["check", str(model_dir), "--tier", "1"])

    mock_t0.assert_called_once()
    mock_t1.assert_called_once()
    assert result.exit_code == 0


def test_tier1_skipped_when_tier0_critical(tmp_path):
    """If Tier 0 returns CRITICAL, Tier 1 is skipped."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"torch_dtype": "float32"}')

    t0_report = TierReport.create(
        tier=0,
        model=str(model_dir),
        checks=[
            DiagnosticResult(
                check_id="0.1",
                name="Dtype",
                status=CheckStatus.CRITICAL,
                detail="BF16 to FP16 mismatch.",
            )
        ],
    )

    with patch("mlx_triage.cli.run_tier0", return_value=t0_report) as mock_t0, \
         patch("mlx_triage.cli.run_tier1") as mock_t1:
        runner = CliRunner()
        result = runner.invoke(cli, ["check", str(model_dir), "--tier", "1"])

    mock_t0.assert_called_once()
    mock_t1.assert_not_called()
    assert result.exit_code == 0
    assert "critical" in result.output.lower() or "CRITICAL" in result.output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_tier1.py -v`
Expected: FAIL — `run_tier1` not imported in cli.py

**Step 3: Update cli.py to wire --tier**

```python
# src/mlx_triage/cli.py
"""CLI entry point for mlx-triage."""

from __future__ import annotations

import click

from mlx_triage import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """MLX Inference Quality Diagnostic Toolkit.

    Run tiered diagnostic checks against MLX-served models to determine
    whether quality issues stem from the model or the inference infrastructure.
    """


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--tier",
    type=click.IntRange(0, 3),
    default=0,
    help="Maximum diagnostic tier to run (0-3).",
)
@click.option("--output", type=click.Path(), default=None, help="Save report to file.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "terminal", "markdown"]),
    default="terminal",
    help="Output format.",
)
def check(model_path: str, tier: int, output: str | None, fmt: str) -> None:
    """Run diagnostic checks against an MLX model."""
    from mlx_triage.report import render_json, render_terminal, write_report
    from mlx_triage.tier0 import run_tier0

    reports: list = []

    # Tier 0 always runs first
    t0_report = run_tier0(model_path)
    reports.append(t0_report)

    # Tier 1 runs if requested and Tier 0 didn't find critical issues
    if tier >= 1:
        if t0_report.worst_status.value == "CRITICAL":
            click.echo(
                click.style(
                    "Tier 0 found CRITICAL issues. Fix those before running Tier 1.",
                    fg="red",
                    bold=True,
                )
            )
        else:
            from mlx_triage.tier1 import run_tier1

            t1_report = run_tier1(model_path)
            reports.append(t1_report)

    # Render all reports
    for report in reports:
        if output:
            write_report(report, output, fmt=fmt)
            click.echo(f"Report saved to {output}")
        elif fmt == "json":
            click.echo(render_json(report))
        else:
            click.echo(render_terminal(report))


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), required=True, help="Output file path.")
def report(model_path: str, output: str) -> None:
    """Generate a full diagnostic report."""
    click.echo(f"Generating report for {model_path}...")
    # TODO: Wire to full multi-tier report generator


@cli.command()
@click.argument("model_a", type=click.Path(exists=True))
@click.argument("model_b", type=click.Path(exists=True))
def compare(model_a: str, model_b: str) -> None:
    """Compare diagnostics between two models."""
    click.echo(f"Comparing {model_a} vs {model_b}...")
    # TODO: Wire to comparison logic
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_tier1.py -v`
Expected: All 3 tests PASS.

Run: `uv run pytest -v`
Expected: Full suite passes (existing tests unbroken).

**Step 5: Commit**

```bash
git add src/mlx_triage/cli.py tests/test_cli_tier1.py
git commit -m "feat: wire --tier flag to run Tier 0 + Tier 1 with CRITICAL gate"
```

---

### Task 10: Multi-Tier JSON Report

**Files:**
- Modify: `src/mlx_triage/report.py`
- Modify: `tests/test_report.py`

**Step 1: Write the failing test**

Add to `tests/test_report.py`:

```python
def test_render_json_tier1():
    """Tier 1 report renders with check_id prefix 1.x."""
    report = TierReport.create(
        tier=1,
        model="/test/model",
        checks=[
            DiagnosticResult(
                check_id="1.1",
                name="Determinism",
                status=CheckStatus.PASS,
                detail="Deterministic at temp=0.",
                metadata={"avg_agreement": 1.0, "n_runs": 10},
            ),
        ],
    )
    output = render_json(report)
    data = json.loads(output)
    assert data["tier"] == 1
    assert "1.1" in data["checks"]
    assert data["checks"]["1.1"]["metadata"]["avg_agreement"] == 1.0
```

**Step 2: Run test to verify it passes (it should — JSON rendering is tier-agnostic)**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS — existing render_json already handles any tier. If it passes, no code change needed.

**Step 3: If test already passes, commit the test only**

```bash
git add tests/test_report.py
git commit -m "test: add Tier 1 report rendering test"
```

---

### Task 11: End-to-End Validation on Real Models

**This task is manual, not TDD. Run on your M2 Max with real cached MLX models.**

**Step 1: Ensure MLX is installed**

Run: `uv sync --extra dev --extra mlx`
Expected: mlx and mlx-lm install successfully.

**Step 2: Run Tier 0 against a known-good model**

Run: `uv run mlx-triage check ~/.cache/huggingface/hub/models--mlx-community--<your-model>/snapshots/<hash>/`
Expected: Tier 0 report with all PASS or INFO.

**Step 3: Run Tier 0+1 against the same model**

Run: `uv run mlx-triage check ~/.cache/huggingface/hub/models--mlx-community--<your-model>/snapshots/<hash>/ --tier 1`
Expected: Tier 0 report, then Tier 1 report. Determinism check should PASS for quantized models.

**Step 4: Run Tier 0+1 with JSON output**

Run: `uv run mlx-triage check <path> --tier 1 --format json`
Expected: Valid JSON with tier 0 and tier 1 data.

**Step 5: Document results in current-state.md**

Record: which model, what results, any issues found.

**Step 6: Commit state update**

```bash
git add docs/current-state.md
git commit -m "docs: record end-to-end validation results for v0.1"
```

---

## Stream B: Packaging (Codex — can run in parallel with Stream A)

### Task P1: README.md Overhaul

**Files:**
- Modify: `README.md`

**Step 1: Replace README.md with full content**

```markdown
# mlx-triage

**MLX Inference Quality Diagnostic Toolkit**

[![CI](https://github.com/<owner>/mlx-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/mlx-triage/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlx-triage.svg)](https://pypi.org/project/mlx-triage/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-triage.svg)](https://pypi.org/project/mlx-triage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

When an MLX-served model produces bad output on Apple Silicon, practitioners can't tell whether the issue is the **model's weights** or the **inference infrastructure**. They waste hours — sometimes weeks — debugging the wrong layer.

**mlx-triage** is a tiered diagnostic CLI that runs checks against any MLX-served model and produces a structured report telling you whether your quality issue is infrastructure or model, and what to do about it.

```
┌────────────────────────────────────┐
│        mlx-triage check            │
├────────────────────────────────────┤
│  Tier 0: Sanity      < 30 sec     │  Config checks, no inference
│  Tier 1: Statistical  1-5 min     │  Determinism, divergence, quality
│  Tier 2: Isolation    5-30 min    │  Batch, memory, context (planned)
│  Tier 3: Deep         30-120 min  │  Activation diffing (planned)
└────────────────────────────────────┘
```

## Install

```bash
# Core (Tier 0 — no MLX required)
pip install mlx-triage

# With MLX support (Tier 1)
pip install mlx-triage[mlx]

# With reference backend for cross-framework comparison
pip install mlx-triage[reference]

# Development
pip install mlx-triage[dev]
```

Or with `uv`:

```bash
uv pip install mlx-triage[mlx]
```

## Quick Start

```bash
# Run Tier 0 sanity checks (fast, no inference)
mlx-triage check /path/to/mlx/model

# Run Tier 0 + Tier 1 statistical tests
mlx-triage check /path/to/mlx/model --tier 1

# JSON output
mlx-triage check /path/to/mlx/model --format json

# Save report to file
mlx-triage check /path/to/mlx/model --tier 1 --output report.json --format json
```

## What It Checks

### Tier 0: Sanity Checks (< 30 seconds, no inference)

| Check | What it catches | Evidence basis |
|-------|----------------|----------------|
| **Dtype compatibility** | BF16→FP16 overflow (Gemma 3 disaster), precision mismatches | Gemma 3 incident, baa.ai postmortem |
| **Tokenizer & EOS** | Missing stop tokens, wrong chat templates, double-BOS | 400+ Ollama issues, llama.cpp fallback |
| **Weight integrity** | NaN/Inf in weights, all-zero layers, corrupted files | baa.ai bfloat16 silent corruption |
| **MLX version** | Known bugs for your MLX version + model architecture | MLX issues #488, #2695, qmv loop |

### Tier 1: Statistical Smoke Tests (1-5 minutes)

| Check | What it catches | Evidence basis |
|-------|----------------|----------------|
| **Determinism** | Non-reproducible output at temp=0 | Metal float16 non-determinism, Token-DiFR |
| **Reference divergence** | MLX output differs from Transformers on same weights | Token-DiFR methodology |
| **Quantization quality** | Perplexity outside expected band for quantization level | oobabooga community benchmarks |

## Evidence Basis

This tool's methodology is grounded in:

- **11 frontier model analyses** across Claude, GPT, Gemini, Grok, Kimi, and Sonar
- **2 systematic reviews** (Consensus PRISMA: 1,099→50 papers)
- **Production postmortems** from Anthropic (3 simultaneous infra bugs), baa.ai (4 MLX bugs), Meta (HawkEye)
- **929-bug taxonomy** across 5 inference engines
- **Research gap confirmation**: "Batch/Scheduler × Unified Memory" and "XAI × Metal Shaders" are complete gaps

See [METHODOLOGY.md](METHODOLOGY.md) for the full evidence trail.

## Target Platform

Apple Silicon (M1–M4), MLX framework, local inference. Developed and tested on M2 Max 96GB.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The easiest way to contribute is adding entries to the [known bugs database](src/mlx_triage/data/known_bugs.yaml) — the tool gets smarter without any code changes.

## License

MIT — see [LICENSE](LICENSE).
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: overhaul README with full install, usage, and evidence summary"
```

---

### Task P2: METHODOLOGY.md

**Files:**
- Create: `METHODOLOGY.md`

**Step 1: Create METHODOLOGY.md**

Draw from these source files (read them for citation details):
- `docs/evidence/claude-c1-infrastructure-defects.md` — Token-DiFR, batch invariance, Anthropic postmortem
- `docs/evidence/gemini-c1-infrastructure-defects.md` — 929-bug taxonomy, defect categories
- `docs/evidence/consensus-c1-prisma-review.pdf` — PRISMA gap analysis
- `docs/evidence/experiment-consolidation.md` — 7-experiment origin story
- `docs/context/After Submission of C1 Research Artifacts.md` — cross-model synthesis
- `docs/context/After Submission of C2 Research Artifacts.md` — C1+C2 synthesis

**Structure:**

```markdown
# Methodology

## The Problem

[1-2 paragraphs: when MLX models produce bad output, practitioners can't tell if it's
the model or the infrastructure. Evidence: Anthropic postmortem, baa.ai postmortem]

## The Research Gap

[1-2 paragraphs: Consensus PRISMA review confirms two complete gaps at the intersection
of batch/scheduler diagnostics × unified memory/Metal effects, and XAI diagnostics ×
unified memory effects. No systematic triage framework exists for MLX/Apple Silicon.]

## The Evidence Base

[Structured summary of evidence sources organized by category:
- Production postmortems (Anthropic, baa.ai, Meta HawkEye)
- Empirical studies (929-bug taxonomy, 500K evaluations, oobabooga benchmarks)
- MLX-specific evidence (Metal float16, qmv loop, Conv1d drift, GELU mismatch)
- Diagnostic tools referenced (Token-DiFR, batch_invariant_ops, HawkEye, lm-eval)]

## The Tiered Protocol

[Explanation of why tiered: most issues resolve at Tier 0/1. Cite the C1+C2 synthesis:
"in steady-state, infrastructure is faithful; during setup/config, infrastructure
failures routinely masquerade as model failures."]

## Three-Layered Conclusion

[From the approved design:
1. Steady-state: model quality is binding constraint
2. Setup/config: infrastructure failures are common and misattributed
3. The boundary between regimes is not well-characterized — this tool helps]

## Citations

[Full citation list with links to GitHub issues, papers, blog posts]
```

**Step 2: Commit**

```bash
git add METHODOLOGY.md
git commit -m "docs: add METHODOLOGY.md with full evidence trail and research gap analysis"
```

---

### Task P3: CONTRIBUTING.md + LICENSE

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `LICENSE`

**Step 1: Create CONTRIBUTING.md**

```markdown
# Contributing to mlx-triage

Thank you for your interest in contributing! This project fills a documented gap in the
MLX/Apple Silicon diagnostic tooling landscape, and community contributions make it
stronger.

## Easiest Way to Contribute: Known Bugs Database

The `src/mlx_triage/data/known_bugs.yaml` file is the tool's institutional knowledge.
Adding a new entry requires no code changes — just YAML:

```yaml
- id: MLX-007
  title: "Description of the bug"
  mlx_issue: 1234  # GitHub issue number, or null
  source: "where you found it"  # if not a GitHub issue
  affected_versions: ["< 0.26.0"]  # or ["all"]
  severity: critical  # critical, high, warning, info
  detection: "tier0.version_check"  # which check detects it
  symptom: "What the user sees"
  architecture: ["llama", "mistral"]  # or ["all"]
  remediation: "How to fix it"
```

## Adding a New Check

1. Create the check function in the appropriate tier module (e.g., `src/mlx_triage/tier0/`)
2. It MUST return a `DiagnosticResult` (never raise for expected conditions)
3. Write tests first (TDD) — see existing tests in `tests/` for patterns
4. Add it to the tier's runner (`__init__.py`)

## Development Setup

```bash
git clone https://github.com/<owner>/mlx-triage.git
cd mlx-triage
uv sync --extra dev --extra mlx
uv run pytest -v
```

## Code Style

- Type hints required on all functions
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`
- `CheckStatus` severity ordering: PASS < INFO < SKIP < WARNING < FAIL < CRITICAL
- Checks return `DiagnosticResult`, never raise for expected failures

## Testing

Every check function has tests. Run:

```bash
uv run pytest -v                    # All tests
uv run pytest --cov=mlx_triage -v   # With coverage
```

Tier 1+ tests mock MLX to avoid hardware dependency. Real model testing is done
manually on Apple Silicon.
```

**Step 2: Create LICENSE**

Standard MIT license text with `2026 Swaylen Hayes` as copyright holder.

**Step 3: Commit**

```bash
git add CONTRIBUTING.md LICENSE
git commit -m "docs: add CONTRIBUTING.md and MIT LICENSE"
```

---

### Task P4: pyproject.toml Polish

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add classifiers and metadata**

Add to the `[project]` section:

```toml
authors = [
    {name = "Swaylen Hayes"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["mlx", "apple-silicon", "diagnostics", "llm", "inference", "triage"]

[project.urls]
Homepage = "https://github.com/<owner>/mlx-triage"
Repository = "https://github.com/<owner>/mlx-triage"
Issues = "https://github.com/<owner>/mlx-triage/issues"
```

Also add `transformers` optional dependency group:

```toml
[project.optional-dependencies]
mlx = [
    "mlx>=0.20",
    "mlx-lm>=0.20",
]
reference = [
    "transformers>=4.40",
    "torch>=2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add PyPI classifiers, URLs, and reference dependency group"
```

---

### Task P5: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Run tests
        run: uv run pytest -v --tb=short

      - name: Run tests with coverage
        if: matrix.python-version == '3.12'
        run: uv run pytest --cov=mlx_triage --cov-report=term-missing -v
```

Note: This runs Tier 0 tests only (no MLX on CI runners). Tier 1 tests use mocks and will pass without MLX installed.

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions test matrix for Python 3.11-3.13 on macOS"
```

---

### Task P6: Blog Post Outline

**Files:**
- Create: `docs/plans/blog-post-outline.md`

**Step 1: Write outline**

```markdown
# Blog Post Outline: "When Your MLX Model Produces Garbage, Is It the Model or the Infrastructure?"

## Target: dev.to or personal blog. Cross-post to HN, MLX Discord.

## Structure

### 1. The Hook (200 words)
- Open with the Anthropic postmortem: 3 simultaneous infra bugs, weeks of degraded
  output, users blamed "intentional throttling"
- "This happens to every team serving LLMs. On Apple Silicon, it's worse because
  nobody's built the diagnostic tooling."

### 2. The Problem (300 words)
- When MLX model produces bad output, is it weights or infrastructure?
- The two failure regimes: steady-state (usually weights) vs setup/config (often infra)
- Why it's especially hard on Apple Silicon: Metal shader non-determinism, unified
  memory, no paged KV cache, quantization differences from CUDA

### 3. The Evidence (400 words)
- 929 bugs across 5 inference engines
- baa.ai: "What is 2+2?" → "!!!!!!!!!!!!!!!!!!!" (bfloat16 corruption)
- Gemma 3: activations hit 800K, FP16 max is 65K → overflow → NaN
- The Consensus PRISMA gap: nobody's researched this intersection
- Your origin story: 7 experiments, 20+ models, infrastructure failures
  masquerading as model failures (from experiment-consolidation.md)

### 4. The Tool (500 words)
- What mlx-triage does: tiered diagnostic protocol
- Tier 0: catches ~40% of issues in 30 seconds, no inference needed
- Tier 1: statistical smoke tests with determinism, reference divergence
- Show real output examples
- `pip install mlx-triage[mlx]` — try it yourself

### 5. What's Next (200 words)
- Tier 2 (batch invariance, memory pressure) and Tier 3 (activation diffing) planned
- Known bugs database is community-contributable
- Call for contributors: run it on your hardware, report results, add bugs
- Link to GitHub repo, METHODOLOGY.md, CONTRIBUTING.md

## Assets Needed
- Terminal screenshot of Tier 0 output (Rich tables)
- Architecture diagram (the tier pyramid)
- Before/after table showing evidence basis
```

**Step 2: Commit**

```bash
git add docs/plans/blog-post-outline.md
git commit -m "docs: add blog post outline for v0.1 launch announcement"
```

---

## Stream D: Publish

### Task F1: Merge + Final Validation

**Step 1: Ensure all Stream A and B changes are on main**

Run: `git log --oneline -20`
Verify: All tasks committed.

**Step 2: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests pass.

Run: `uv run pytest --cov=mlx_triage -v`
Expected: Reasonable coverage (>80%).

**Step 3: Verify CLI end-to-end**

Run: `uv run mlx-triage --version`
Expected: `mlx-triage, version 0.1.0`

Run: `uv run mlx-triage check <real-model-path> --tier 1 --format json | python -m json.tool`
Expected: Valid, pretty-printed JSON.

---

### Task F2: Build + Publish to PyPI

**Step 1: Build the package**

Run: `uv build`
Expected: `dist/mlx_triage-0.1.0-py3-none-any.whl` and `dist/mlx_triage-0.1.0.tar.gz`

**Step 2: Test install from wheel**

```bash
# Create a clean test venv
python -m venv /tmp/test-mlx-triage
source /tmp/test-mlx-triage/bin/activate
pip install dist/mlx_triage-0.1.0-py3-none-any.whl
mlx-triage --version
deactivate
rm -rf /tmp/test-mlx-triage
```

Expected: `mlx-triage, version 0.1.0`

**Step 3: Publish to PyPI**

Run: `uv publish` (or `twine upload dist/*`)
Note: Requires PyPI API token configured. Set up at https://pypi.org/manage/account/token/

**Step 4: Verify PyPI install**

```bash
python -m venv /tmp/test-pypi
source /tmp/test-pypi/bin/activate
pip install mlx-triage
mlx-triage --version
deactivate
rm -rf /tmp/test-pypi
```

Expected: `mlx-triage, version 0.1.0`

---

### Task F3: Tag + Push to GitHub

**Step 1: Tag the release**

```bash
git tag -a v0.1.0 -m "v0.1.0: Tier 0 + Tier 1 diagnostic toolkit for MLX on Apple Silicon"
```

**Step 2: Push to GitHub**

```bash
git push origin main --tags
```

**Step 3: Create GitHub release**

```bash
gh release create v0.1.0 --title "v0.1.0" --notes "$(cat <<'EOF'
## mlx-triage v0.1.0

First public release of the MLX Inference Quality Diagnostic Toolkit.

### What's included

- **Tier 0: Sanity Checks** (< 30 seconds, no inference)
  - Dtype compatibility audit
  - Tokenizer & EOS configuration audit
  - Weight file integrity check
  - MLX version & known bug check

- **Tier 1: Statistical Smoke Tests** (1-5 minutes)
  - Determinism check (N runs at temp=0)
  - Reference divergence (MLX vs Transformers)
  - Quantization quality gate (perplexity)

### Install

```bash
pip install mlx-triage[mlx]
```

### Usage

```bash
mlx-triage check /path/to/mlx/model --tier 1
```

See [README](README.md) for full documentation and [METHODOLOGY.md](METHODOLOGY.md) for the research basis.
EOF
)"
```

**Step 4: Update state docs**

Update `docs/current-state.md` and `docs/roadmap.md` to reflect v0.1.0 published.

```bash
git add docs/current-state.md docs/roadmap.md
git commit -m "docs: update state for v0.1.0 release"
```

---

## Execution Order Summary

```
Phase 1 Tasks 1-7 (existing plan)     ← Claude Code, Sessions S1-S3
    ↓
Task 8: Tier 1 Runner                 ← Claude Code, Session S4
Task 9: CLI --tier wiring             ← Claude Code, Session S4
Task 10: Multi-tier report test       ← Claude Code, Session S4
Task 11: E2E validation (manual)      ← Claude Code, Session S4
    ↓
Tasks P1-P6: Packaging                ← Codex (parallel with S2-S4)
    ↓
Task F1: Merge + validate             ← Claude Code, Session S5
Task F2: Build + PyPI publish          ← Claude Code, Session S5
Task F3: Tag + GitHub release          ← Claude Code, Session S5
```

**Inter-agent memo checkpoint:** After Task 11 (E2E validation passes), before Task F1.
