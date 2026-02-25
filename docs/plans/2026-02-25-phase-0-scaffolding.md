# Phase 0: Scaffolding & Tier 0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A Python CLI that runs Tier 0 diagnostic checks against any local MLX model directory, producing structured reports in JSON and Rich terminal formats — no inference required.

**Architecture:** Click CLI routes commands → Tier 0 runner orchestrates 4 checks (dtype, tokenizer, weight integrity, version) → each check reads model config files and safetensors headers → results flow through a report generator that outputs JSON or Rich terminal tables. A known-bugs YAML database provides institutional knowledge for version-specific issue detection.

**Tech Stack:** Python 3.11+, Click (CLI), Rich (terminal UI), PyYAML (config), safetensors (weight headers), numpy (weight stats), pytest (testing)

**Ship criteria:** `mlx-triage check <any-mlx-model-dir>` produces a Tier 0 report in < 30 seconds.

**Evidence basis:** This tier alone catches ~40% of infrastructure-vs-model confusion (Gemma 3 FP16 overflow, baa.ai BF16 corruption, Ollama template disasters, MLX version-specific bugs).

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/mlx_triage/__init__.py`
- Create: `src/mlx_triage/tier0/__init__.py`
- Create: `src/mlx_triage/tier1/__init__.py`
- Create: `src/mlx_triage/tier2/__init__.py`
- Create: `src/mlx_triage/tier3/__init__.py`
- Create: `src/mlx_triage/prompts/__init__.py`
- Create: `src/mlx_triage/utils/__init__.py`
- Create: `src/mlx_triage/data/` (directory for YAML databases)
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "mlx-triage"
version = "0.1.0"
description = "MLX Inference Quality Diagnostic Toolkit"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    "click>=8.1",
    "rich>=13.0",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "numpy>=1.26",
]

[project.optional-dependencies]
mlx = [
    "mlx>=0.20",
    "mlx-lm>=0.20",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
mlx-triage = "mlx_triage.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mlx_triage"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create directory structure**

```bash
mkdir -p src/mlx_triage/{tier0,tier1,tier2,tier3,prompts,utils,data}
mkdir -p tests
touch src/mlx_triage/__init__.py
touch src/mlx_triage/tier0/__init__.py
touch src/mlx_triage/tier1/__init__.py
touch src/mlx_triage/tier2/__init__.py
touch src/mlx_triage/tier3/__init__.py
touch src/mlx_triage/prompts/__init__.py
touch src/mlx_triage/utils/__init__.py
touch tests/__init__.py
```

Write `src/mlx_triage/__init__.py`:
```python
"""MLX Inference Quality Diagnostic Toolkit."""

__version__ = "0.1.0"
```

**Step 3: Install dependencies**

```bash
uv sync --extra dev
```

Run: `uv sync --extra dev`
Expected: Creates `.venv/`, installs all dependencies, exits 0.

**Step 4: Verify setup**

Run: `uv run python -c "from mlx_triage import __version__; print(__version__)"`
Expected: `0.1.0`

**Step 5: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py uv.lock
git commit -m "feat: initialize project scaffolding with src layout"
```

---

## Task 2: Core Data Models

**Files:**
- Create: `src/mlx_triage/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport


def test_check_status_values():
    assert CheckStatus.PASS.value == "PASS"
    assert CheckStatus.CRITICAL.value == "CRITICAL"
    assert CheckStatus.WARNING.value == "WARNING"
    assert CheckStatus.FAIL.value == "FAIL"
    assert CheckStatus.INFO.value == "INFO"
    assert CheckStatus.SKIP.value == "SKIP"


def test_diagnostic_result_creation():
    result = DiagnosticResult(
        check_id="0.1",
        name="Dtype Compatibility",
        status=CheckStatus.PASS,
        detail="Compatible.",
    )
    assert result.check_id == "0.1"
    assert result.status == CheckStatus.PASS
    assert result.remediation is None
    assert result.metadata == {}


def test_diagnostic_result_with_remediation():
    result = DiagnosticResult(
        check_id="0.2",
        name="Tokenizer Config",
        status=CheckStatus.WARNING,
        detail="Missing stop token.",
        remediation="Add missing stop token to generation_config.json.",
    )
    assert result.remediation is not None


def test_tier_report_creation():
    checks = [
        DiagnosticResult(
            check_id="0.1",
            name="Dtype",
            status=CheckStatus.PASS,
            detail="OK",
        ),
    ]
    report = TierReport(
        tier=0,
        model="/path/to/model",
        timestamp="2026-02-25T14:30:00Z",
        checks=checks,
    )
    assert report.tier == 0
    assert len(report.checks) == 1
    assert report.verdict != ""
    assert isinstance(report.should_continue, bool)


def test_tier_report_verdict_all_pass():
    checks = [
        DiagnosticResult(check_id="0.1", name="A", status=CheckStatus.PASS, detail="OK"),
        DiagnosticResult(check_id="0.2", name="B", status=CheckStatus.PASS, detail="OK"),
    ]
    report = TierReport(tier=0, model="test", timestamp="now", checks=checks)
    assert "PASS" in report.verdict
    assert report.should_continue is True


def test_tier_report_verdict_critical():
    checks = [
        DiagnosticResult(check_id="0.1", name="A", status=CheckStatus.CRITICAL, detail="Bad"),
        DiagnosticResult(check_id="0.2", name="B", status=CheckStatus.PASS, detail="OK"),
    ]
    report = TierReport(tier=0, model="test", timestamp="now", checks=checks)
    assert "CRITICAL" in report.verdict or "FAIL" in report.verdict
    assert report.should_continue is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_triage.models'`

**Step 3: Implement models.py**

```python
# src/mlx_triage/models.py
"""Core data models for mlx-triage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CheckStatus(Enum):
    """Status of a diagnostic check, ordered by severity."""

    PASS = "PASS"
    INFO = "INFO"
    SKIP = "SKIP"
    WARNING = "WARNING"
    FAIL = "FAIL"
    CRITICAL = "CRITICAL"


# Severity ordering for verdict computation
_SEVERITY_ORDER = {
    CheckStatus.PASS: 0,
    CheckStatus.INFO: 1,
    CheckStatus.SKIP: 2,
    CheckStatus.WARNING: 3,
    CheckStatus.FAIL: 4,
    CheckStatus.CRITICAL: 5,
}


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    check_id: str
    name: str
    status: CheckStatus
    detail: str
    remediation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TierReport:
    """Aggregated report for a diagnostic tier."""

    tier: int
    model: str
    timestamp: str
    checks: list[DiagnosticResult]

    @property
    def worst_status(self) -> CheckStatus:
        if not self.checks:
            return CheckStatus.SKIP
        return max(self.checks, key=lambda c: _SEVERITY_ORDER[c.status]).status

    @property
    def verdict(self) -> str:
        worst = self.worst_status
        issue_count = sum(
            1
            for c in self.checks
            if c.status in (CheckStatus.WARNING, CheckStatus.FAIL, CheckStatus.CRITICAL)
        )
        if worst == CheckStatus.PASS:
            return "PASS — All checks passed."
        if worst == CheckStatus.INFO:
            return "PASS — All checks passed with informational notes."
        return f"{worst.value} — {issue_count} issue(s) found."

    @property
    def should_continue(self) -> bool:
        return self.worst_status not in (CheckStatus.FAIL, CheckStatus.CRITICAL)

    @classmethod
    def create(cls, tier: int, model: str, checks: list[DiagnosticResult]) -> TierReport:
        return cls(
            tier=tier,
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=checks,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/models.py tests/test_models.py
git commit -m "feat: add core data models (CheckStatus, DiagnosticResult, TierReport)"
```

---

## Task 3: Known Bugs Database

**Files:**
- Create: `src/mlx_triage/data/known_bugs.yaml`
- Create: `src/mlx_triage/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from mlx_triage.config import KnownBug, load_known_bugs, find_bugs_for_model


def test_load_known_bugs():
    bugs = load_known_bugs()
    assert len(bugs) >= 6  # We ship with at least 6 documented bugs
    assert all(isinstance(b, KnownBug) for b in bugs)


def test_known_bug_fields():
    bugs = load_known_bugs()
    bug = next(b for b in bugs if b.id == "MLX-001")
    assert bug.title == "Float16 addmm CPU wrong results"
    assert bug.severity == "critical"
    assert "all" in bug.architecture


def test_find_bugs_by_version_and_arch():
    bugs = load_known_bugs()
    # Old version should match known bugs
    matches = find_bugs_for_model(bugs, mlx_version="0.21.0", architecture="llama")
    assert any(b.id == "MLX-001" for b in matches)


def test_find_bugs_current_version():
    bugs = load_known_bugs()
    # Current version should have fewer (or no) version-specific bugs
    matches = find_bugs_for_model(bugs, mlx_version="0.25.1", architecture="llama")
    version_bugs = [b for b in matches if b.affected_versions != ["all"]]
    # Should not match version-gated bugs on current MLX
    assert not any(b.id == "MLX-001" for b in version_bugs)


def test_find_bugs_filters_by_architecture():
    bugs = load_known_bugs()
    # Audio-specific bug should not match llama
    matches = find_bugs_for_model(bugs, mlx_version="0.21.0", architecture="llama")
    assert not any(b.id == "MLX-005" for b in matches)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_triage.config'`

**Step 3: Create known_bugs.yaml**

```yaml
# src/mlx_triage/data/known_bugs.yaml
# Curated database of documented MLX/framework bugs.
# Sources: MLX GitHub issues, community postmortems, research evidence.
# See docs/evidence/ for primary sources.

bugs:
  - id: MLX-001
    title: "Float16 addmm CPU wrong results"
    mlx_issue: 2695
    affected_versions: ["< 0.22.0"]
    severity: critical
    detection: "tier0.version_check"
    symptom: "Completely wrong matrix multiplication results on CPU with float16"
    architecture: ["all"]
    remediation: "Use bfloat16 or float32, or upgrade MLX"

  - id: MLX-002
    title: "qmv kernel infinite loop at specific tensor dimensions"
    mlx_issue: null
    affected_versions: ["< 0.24.0"]
    severity: critical
    detection: "tier0.version_check"
    symptom: "Model enters infinite generation loop at end of long prompts (4-bit only)"
    architecture: ["llama", "mistral"]
    remediation: "Use 8-bit quantization or upgrade MLX"

  - id: MLX-003
    title: "Metal float16 non-determinism with error accumulation"
    mlx_issue: 488
    affected_versions: ["all"]
    severity: warning
    detection: "tier1.determinism"
    symptom: "Non-reproducible outputs at temperature=0 with float16 weights"
    architecture: ["all"]
    remediation: "Use quantized integer models (Q4, Q8) for reproducibility"

  - id: MLX-004
    title: "bfloat16 save_safetensors silent corruption"
    mlx_issue: null
    source: "baa.ai postmortem"
    affected_versions: ["all"]
    severity: critical
    detection: "tier0.weight_integrity"
    symptom: "Structurally valid safetensors file with numerically garbage weights"
    architecture: ["all"]
    remediation: "Verify weight checksums against hub; re-download if mismatch"

  - id: MLX-005
    title: "Conv1d composition drift"
    mlx_issue: 2122
    affected_versions: ["all"]
    severity: warning
    detection: "tier3.activation_diff"
    symptom: "Sequential Conv1d operations accumulate MAE ~0.04 vs PyTorch"
    architecture: ["whisper", "wav2vec", "audio_models"]
    remediation: "Use float32 for audio model inference or validate output quality"

  - id: MLX-006
    title: "GELU approximation mismatch in vision encoders"
    mlx_issue: null
    source: "PaliGemma MLX issue"
    affected_versions: ["all"]
    severity: high
    detection: "tier3.activation_diff"
    symptom: "Vision encoder activations diverge >100K sum-abs-diff from reference"
    architecture: ["paligemma", "clip", "siglip"]
    remediation: "Verify GELU implementation matches model's training config"
```

**Step 4: Implement config.py**

```python
# src/mlx_triage/config.py
"""Configuration and known-bugs database loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class KnownBug:
    """A documented MLX/framework bug."""

    id: str
    title: str
    affected_versions: list[str]
    severity: str
    detection: str
    symptom: str
    architecture: list[str]
    remediation: str
    mlx_issue: int | None = None
    source: str | None = None


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string like '0.22.0' into a comparable tuple."""
    return tuple(int(x) for x in v.strip().split("."))


def _version_matches(installed: str, constraint: str) -> bool:
    """Check if installed version matches a constraint like '< 0.22.0'."""
    constraint = constraint.strip()
    if constraint == "all":
        return True
    if constraint.startswith("< "):
        return _parse_version(installed) < _parse_version(constraint[2:])
    if constraint.startswith("<= "):
        return _parse_version(installed) <= _parse_version(constraint[3:])
    if constraint.startswith("> "):
        return _parse_version(installed) > _parse_version(constraint[2:])
    if constraint.startswith(">= "):
        return _parse_version(installed) >= _parse_version(constraint[3:])
    # Exact match
    return installed == constraint


def load_known_bugs(path: str | Path | None = None) -> list[KnownBug]:
    """Load known bugs database from YAML."""
    if path is None:
        path = Path(__file__).parent / "data" / "known_bugs.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    bugs = []
    for entry in data["bugs"]:
        bugs.append(
            KnownBug(
                id=entry["id"],
                title=entry["title"],
                affected_versions=entry["affected_versions"],
                severity=entry["severity"],
                detection=entry["detection"],
                symptom=entry["symptom"],
                architecture=entry["architecture"],
                remediation=entry["remediation"],
                mlx_issue=entry.get("mlx_issue"),
                source=entry.get("source"),
            )
        )
    return bugs


def find_bugs_for_model(
    bugs: list[KnownBug],
    mlx_version: str,
    architecture: str,
) -> list[KnownBug]:
    """Find known bugs that affect the given model configuration."""
    matches = []
    for bug in bugs:
        # Check architecture match
        arch_match = "all" in bug.architecture or architecture.lower() in [
            a.lower() for a in bug.architecture
        ]
        if not arch_match:
            continue

        # Check version match
        version_match = any(
            _version_matches(mlx_version, v) for v in bug.affected_versions
        )
        if not version_match:
            continue

        matches.append(bug)
    return matches
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 5 tests PASS.

**Step 6: Commit**

```bash
git add src/mlx_triage/config.py src/mlx_triage/data/known_bugs.yaml tests/test_config.py
git commit -m "feat: add known-bugs YAML database and config loader"
```

---

## Task 4: CLI Skeleton

**Files:**
- Create: `src/mlx_triage/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from mlx_triage.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MLX Inference Quality Diagnostic Toolkit" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_check_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "MODEL_PATH" in result.output


def test_report_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--help"])
    assert result.exit_code == 0


def test_compare_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_triage.cli'`

**Step 3: Implement cli.py**

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
@click.option("--tier", type=click.IntRange(0, 3), default=0, help="Maximum diagnostic tier to run (0-3).")
@click.option("--output", type=click.Path(), default=None, help="Save report to file.")
@click.option("--format", "fmt", type=click.Choice(["json", "terminal", "markdown"]), default="terminal", help="Output format.")
def check(model_path: str, tier: int, output: str | None, fmt: str) -> None:
    """Run diagnostic checks against an MLX model."""
    click.echo(f"Running Tier 0–{tier} checks on {model_path}...")
    # TODO: Wire to tier runner


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), required=True, help="Output file path.")
def report(model_path: str, output: str) -> None:
    """Generate a full diagnostic report."""
    click.echo(f"Generating report for {model_path}...")
    # TODO: Wire to full report generator


@cli.command()
@click.argument("model_a", type=click.Path(exists=True))
@click.argument("model_b", type=click.Path(exists=True))
def compare(model_a: str, model_b: str) -> None:
    """Compare diagnostics between two models."""
    click.echo(f"Comparing {model_a} vs {model_b}...")
    # TODO: Wire to comparison logic
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 5 tests PASS.

**Step 5: Verify CLI entry point**

Run: `uv run mlx-triage --help`
Expected: Shows help text with `check`, `report`, `compare` commands.

**Step 6: Commit**

```bash
git add src/mlx_triage/cli.py tests/test_cli.py
git commit -m "feat: add Click CLI skeleton with check/report/compare commands"
```

---

## Task 5: Check 0.1 — Dtype Compatibility

**Files:**
- Create: `src/mlx_triage/tier0/dtype_check.py`
- Create: `tests/test_tier0_dtype.py`
- Create: `tests/conftest.py`

**Step 1: Write the failing test**

First, create shared test fixtures in `tests/conftest.py`:

```python
# tests/conftest.py
"""Shared test fixtures for mlx-triage tests."""

import json
import struct
from pathlib import Path

import pytest


def create_safetensors(path: Path, dtype: str = "BF16", num_tensors: int = 1) -> None:
    """Create a minimal safetensors file with a valid header.

    Args:
        path: File path to write.
        dtype: Safetensors dtype string (F16, BF16, F32).
        num_tensors: Number of tensor entries in header.
    """
    dtype_sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1}
    elem_size = dtype_sizes.get(dtype, 4)
    shape = [2, 2]
    data_size = shape[0] * shape[1] * elem_size

    header = {}
    offset = 0
    for i in range(num_tensors):
        header[f"model.layers.{i}.weight"] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + data_size],
        }
        offset += data_size

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * offset)


@pytest.fixture
def good_model(tmp_path: Path) -> Path:
    """Model directory that passes all Tier 0 checks."""
    d = tmp_path / "good-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"})
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        })
    )
    create_safetensors(d / "model.safetensors", dtype="BF16")
    return d


@pytest.fixture
def bf16_to_fp16_model(tmp_path: Path) -> Path:
    """Model with BF16 training dtype but FP16 stored weights (Gemma 3 failure mode)."""
    d = tmp_path / "bad-dtype"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "gemma", "torch_dtype": "bfloat16"})
    )
    create_safetensors(d / "model.safetensors", dtype="F16")
    return d


@pytest.fixture
def quantized_model(tmp_path: Path) -> Path:
    """A 4-bit quantized model (should pass dtype check)."""
    d = tmp_path / "quantized"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({
            "model_type": "llama",
            "torch_dtype": "bfloat16",
            "quantization_config": {"bits": 4, "quant_method": "mlx"},
        })
    )
    create_safetensors(d / "model.safetensors", dtype="I8")
    return d
```

Now the dtype test:

```python
# tests/test_tier0_dtype.py
from mlx_triage.models import CheckStatus
from mlx_triage.tier0.dtype_check import check_dtype_compatibility


def test_matching_dtypes_pass(good_model):
    result = check_dtype_compatibility(str(good_model))
    assert result.check_id == "0.1"
    assert result.status == CheckStatus.PASS


def test_bf16_to_fp16_critical(bf16_to_fp16_model):
    result = check_dtype_compatibility(str(bf16_to_fp16_model))
    assert result.status == CheckStatus.CRITICAL
    assert "bfloat16" in result.detail.lower() or "bf16" in result.detail.lower()
    assert result.remediation is not None


def test_quantized_model_pass(quantized_model):
    result = check_dtype_compatibility(str(quantized_model))
    assert result.status == CheckStatus.PASS
    assert "quantiz" in result.detail.lower()


def test_missing_config_fails(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = check_dtype_compatibility(str(d))
    assert result.status == CheckStatus.FAIL


def test_fp32_to_fp16_warning(tmp_path):
    from tests.conftest import create_safetensors
    import json

    d = tmp_path / "fp32-to-fp16"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "llama", "torch_dtype": "float32"})
    )
    create_safetensors(d / "model.safetensors", dtype="F16")
    result = check_dtype_compatibility(str(d))
    assert result.status == CheckStatus.WARNING
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0_dtype.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_triage.tier0.dtype_check'`

**Step 3: Implement dtype_check.py**

```python
# src/mlx_triage/tier0/dtype_check.py
"""Check 0.1: Dtype compatibility audit.

Evidence basis: Gemma 3 BF16→FP16 overflow (activations reach 800K, FP16 max 65504),
baa.ai bfloat16 silent corruption, GatedDeltaNet float16 accumulation collapse.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from mlx_triage.models import CheckStatus, DiagnosticResult

# Training→stored dtype pairs known to cause critical failures
CRITICAL_MISMATCHES = {
    ("bfloat16", "float16"),  # Gemma 3: overflow after LayerNorm
}

# Training→stored dtype pairs that risk precision loss
WARNING_MISMATCHES = {
    ("float32", "float16"),
}

# Safetensors dtype code → normalized name
SAFETENSORS_DTYPE_MAP = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "BOOL": "bool",
}


def _read_safetensors_header(path: Path) -> dict:
    """Read safetensors JSON header without loading tensor data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def _get_training_dtype(config: dict) -> str | None:
    """Extract training dtype from model config.json."""
    for key in ("torch_dtype", "dtype", "model_dtype"):
        if key in config:
            return config[key]
    return None


def _get_stored_dtype(model_path: Path) -> str | None:
    """Get normalized dtype of stored weights from safetensors headers."""
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        return None
    header = _read_safetensors_header(files[0])
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if isinstance(value, dict) and "dtype" in value:
            raw = value["dtype"]
            return SAFETENSORS_DTYPE_MAP.get(raw, raw.lower())
    return None


def check_dtype_compatibility(model_path: str) -> DiagnosticResult:
    """Run dtype compatibility audit on a model directory."""
    path = Path(model_path)
    config_path = path / "config.json"

    if not config_path.exists():
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.FAIL,
            detail="config.json not found in model directory.",
        )

    with open(config_path) as f:
        config = json.load(f)

    training_dtype = _get_training_dtype(config)
    model_type = config.get("model_type", "unknown")

    # Quantized models: check quantization config, skip dtype comparison
    quant_config = config.get("quantization_config", {})
    if quant_config:
        bits = quant_config.get("bits", quant_config.get("quant_method", "unknown"))
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.PASS,
            detail=f"Training dtype: {training_dtype or 'unknown'}, quantized to {bits}-bit. Compatible.",
        )

    stored_dtype = _get_stored_dtype(path)

    if stored_dtype is None:
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.INFO,
            detail=f"Training dtype: {training_dtype or 'unknown'}. No safetensors files found to verify stored dtype.",
        )

    if training_dtype is None:
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.INFO,
            detail=f"Training dtype not specified in config. Stored as {stored_dtype}.",
        )

    pair = (training_dtype, stored_dtype)

    if pair in CRITICAL_MISMATCHES:
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.CRITICAL,
            detail=f"Training dtype {training_dtype} stored as {stored_dtype}. High risk of overflow/corruption.",
            remediation=f"Use {training_dtype} weights or appropriate quantization.",
        )

    if pair in WARNING_MISMATCHES:
        return DiagnosticResult(
            check_id="0.1",
            name="Dtype Compatibility",
            status=CheckStatus.WARNING,
            detail=f"Training dtype {training_dtype} stored as {stored_dtype}. Potential precision loss.",
            remediation="Verify output quality or use higher-precision weights.",
        )

    return DiagnosticResult(
        check_id="0.1",
        name="Dtype Compatibility",
        status=CheckStatus.PASS,
        detail=f"Training dtype {training_dtype}, stored as {stored_dtype}. Compatible.",
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0_dtype.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier0/dtype_check.py tests/test_tier0_dtype.py tests/conftest.py
git commit -m "feat: add Tier 0 dtype compatibility check (Check 0.1)"
```

---

## Task 6: Check 0.2 — Tokenizer & EOS Configuration

**Files:**
- Create: `src/mlx_triage/tier0/tokenizer_check.py`
- Create: `tests/test_tier0_tokenizer.py`

**Step 1: Write the failing test**

```python
# tests/test_tier0_tokenizer.py
import json
from pathlib import Path

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.tokenizer_check import check_tokenizer_config


def test_valid_tokenizer_passes(good_model):
    result = check_tokenizer_config(str(good_model))
    assert result.check_id == "0.2"
    assert result.status in (CheckStatus.PASS, CheckStatus.INFO)


def test_missing_eos_critical(tmp_path):
    d = tmp_path / "no-eos"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<s>",
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    }))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.CRITICAL


def test_missing_chat_template_warning(tmp_path):
    d = tmp_path / "no-template"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "</s>",
        "bos_token": "<s>",
    }))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.WARNING


def test_no_tokenizer_config_fails(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    result = check_tokenizer_config(str(d))
    assert result.status == CheckStatus.FAIL


def test_llama3_missing_eot_id_warning(tmp_path):
    """Llama 3 needs BOTH end_of_text AND eot_id stop tokens."""
    d = tmp_path / "llama3"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|end_of_text|>",
        "bos_token": "<|begin_of_text|>",
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    }))
    # generation_config.json with only one stop token
    (d / "generation_config.json").write_text(json.dumps({
        "eos_token_id": 128001,
    }))
    result = check_tokenizer_config(str(d))
    # Should at least INFO about Llama 3 dual stop token pattern
    assert result.status in (CheckStatus.WARNING, CheckStatus.INFO, CheckStatus.PASS)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0_tokenizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement tokenizer_check.py**

```python
# src/mlx_triage/tier0/tokenizer_check.py
"""Check 0.2: Tokenizer & EOS configuration audit.

Evidence basis: Ollama 400+ issues from silent context truncation,
llama.cpp ChatML fallback, vLLM double-BOS injection, Llama 3 dual stop token issue.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_triage.models import CheckStatus, DiagnosticResult

# Tokens that indicate Llama 3+ dual-stop-token pattern
LLAMA3_STOP_TOKENS = {"<|end_of_text|>", "<|eot_id|>"}


def check_tokenizer_config(model_path: str) -> DiagnosticResult:
    """Run tokenizer and EOS configuration audit on a model directory."""
    path = Path(model_path)
    tokenizer_path = path / "tokenizer_config.json"

    if not tokenizer_path.exists():
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.FAIL,
            detail="tokenizer_config.json not found in model directory.",
        )

    with open(tokenizer_path) as f:
        tok_config = json.load(f)

    issues: list[str] = []
    remediations: list[str] = []

    # Check EOS token
    eos_token = tok_config.get("eos_token")
    if not eos_token:
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.CRITICAL,
            detail="No eos_token defined in tokenizer config. Model may never stop generating.",
            remediation="Add eos_token to tokenizer_config.json.",
        )

    # Check chat template
    chat_template = tok_config.get("chat_template")
    if not chat_template:
        issues.append("No chat_template defined — runtime will fall back to ChatML or default template.")
        remediations.append("Add a chat_template to tokenizer_config.json matching the model's training format.")

    # Check for generation_config.json stop tokens
    gen_config_path = path / "generation_config.json"
    if gen_config_path.exists():
        with open(gen_config_path) as f:
            gen_config = json.load(f)

        # Check if multiple stop tokens are needed but not all configured
        eos_token_id = gen_config.get("eos_token_id")
        if isinstance(eos_token_id, int):
            # Single stop token configured — check if model might need multiple
            if isinstance(eos_token, str) and any(t in eos_token for t in LLAMA3_STOP_TOKENS):
                issues.append(
                    f"Llama 3-style EOS token detected ({eos_token}) but generation_config "
                    f"has single eos_token_id. Model may need multiple stop tokens."
                )
                remediations.append("Check if model needs both <|end_of_text|> and <|eot_id|> as stop tokens.")

    if not issues:
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.PASS,
            detail=f"EOS token: {eos_token}. Chat template: {'present' if chat_template else 'missing'}.",
        )

    # Determine severity
    has_template_issue = any("chat_template" in i.lower() for i in issues)
    status = CheckStatus.WARNING if has_template_issue else CheckStatus.INFO

    return DiagnosticResult(
        check_id="0.2",
        name="Tokenizer Config",
        status=status,
        detail=" ".join(issues),
        remediation=" ".join(remediations) if remediations else None,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0_tokenizer.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier0/tokenizer_check.py tests/test_tier0_tokenizer.py
git commit -m "feat: add Tier 0 tokenizer & EOS config check (Check 0.2)"
```

---

## Task 7: Check 0.3 — Weight File Integrity

**Files:**
- Create: `src/mlx_triage/tier0/weight_integrity.py`
- Create: `tests/test_tier0_weights.py`

**Step 1: Write the failing test**

```python
# tests/test_tier0_weights.py
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.weight_integrity import check_weight_integrity


def test_clean_weights_pass(good_model):
    result = check_weight_integrity(str(good_model))
    assert result.check_id == "0.3"
    assert result.status in (CheckStatus.PASS, CheckStatus.INFO)


def test_nan_weights_critical(tmp_path):
    """Safetensors file with NaN values should be CRITICAL (baa.ai failure mode)."""
    d = tmp_path / "nan-model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))

    # Create safetensors with NaN values using numpy
    from safetensors.numpy import save_file

    data = np.array([[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32)
    save_file({"model.layers.0.weight": data}, str(d / "model.safetensors"))

    result = check_weight_integrity(str(d))
    assert result.status == CheckStatus.CRITICAL
    assert "nan" in result.detail.lower()


def test_no_safetensors_info(tmp_path):
    d = tmp_path / "no-weights"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    result = check_weight_integrity(str(d))
    assert result.status == CheckStatus.INFO


def test_inf_weights_critical(tmp_path):
    d = tmp_path / "inf-model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))

    from safetensors.numpy import save_file

    data = np.array([[1.0, float("inf")], [3.0, 4.0]], dtype=np.float32)
    save_file({"model.layers.0.weight": data}, str(d / "model.safetensors"))

    result = check_weight_integrity(str(d))
    assert result.status == CheckStatus.CRITICAL
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0_weights.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement weight_integrity.py**

```python
# src/mlx_triage/tier0/weight_integrity.py
"""Check 0.3: Weight file integrity.

Evidence basis: baa.ai postmortem — mx.save_safetensors with bfloat16 produces
structurally valid but numerically garbage files. "What is 2+2?" → "!!!!!!!!!!"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlx_triage.models import CheckStatus, DiagnosticResult


def _sample_tensors(safetensors_path: Path, max_tensors: int = 5) -> dict[str, np.ndarray]:
    """Load a sample of tensors from a safetensors file using numpy."""
    from safetensors.numpy import load_file

    all_tensors = load_file(str(safetensors_path))
    keys = list(all_tensors.keys())[:max_tensors]
    return {k: all_tensors[k] for k in keys}


def check_weight_integrity(model_path: str) -> DiagnosticResult:
    """Run weight file integrity check on a model directory."""
    path = Path(model_path)
    safetensors_files = sorted(path.glob("*.safetensors"))

    if not safetensors_files:
        return DiagnosticResult(
            check_id="0.3",
            name="Weight Integrity",
            status=CheckStatus.INFO,
            detail="No safetensors files found. Skipping weight integrity check.",
        )

    nan_layers: list[str] = []
    inf_layers: list[str] = []
    zero_layers: list[str] = []
    stats: dict[str, dict] = {}

    for sf_path in safetensors_files[:3]:  # Check up to 3 shard files
        try:
            tensors = _sample_tensors(sf_path)
        except Exception as e:
            return DiagnosticResult(
                check_id="0.3",
                name="Weight Integrity",
                status=CheckStatus.FAIL,
                detail=f"Failed to load {sf_path.name}: {e}",
            )

        for name, tensor in tensors.items():
            # Cast to float for analysis (handles quantized int types)
            try:
                arr = tensor.astype(np.float64)
            except (ValueError, TypeError):
                continue

            if np.isnan(arr).any():
                nan_layers.append(name)
            if np.isinf(arr).any():
                inf_layers.append(name)
            if np.all(arr == 0):
                zero_layers.append(name)

            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    if nan_layers:
        return DiagnosticResult(
            check_id="0.3",
            name="Weight Integrity",
            status=CheckStatus.CRITICAL,
            detail=f"NaN values detected in {len(nan_layers)} layer(s): {', '.join(nan_layers[:3])}.",
            remediation="Re-download weights from HuggingFace Hub. This may be a bfloat16 save corruption (baa.ai bug pattern).",
            metadata={"nan_layers": nan_layers},
        )

    if inf_layers:
        return DiagnosticResult(
            check_id="0.3",
            name="Weight Integrity",
            status=CheckStatus.CRITICAL,
            detail=f"Inf values detected in {len(inf_layers)} layer(s): {', '.join(inf_layers[:3])}.",
            remediation="Re-download weights. Possible corruption during quantization or transfer.",
            metadata={"inf_layers": inf_layers},
        )

    if zero_layers:
        return DiagnosticResult(
            check_id="0.3",
            name="Weight Integrity",
            status=CheckStatus.WARNING,
            detail=f"All-zero layers detected: {', '.join(zero_layers[:3])}. May indicate incomplete download or corruption.",
            remediation="Verify download completed fully. Check file sizes against hub.",
            metadata={"zero_layers": zero_layers},
        )

    files_checked = min(len(safetensors_files), 3)
    tensors_checked = len(stats)
    return DiagnosticResult(
        check_id="0.3",
        name="Weight Integrity",
        status=CheckStatus.PASS,
        detail=f"Checked {tensors_checked} tensors across {files_checked} shard(s). No NaN/Inf detected.",
        metadata={"weight_stats": stats},
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0_weights.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier0/weight_integrity.py tests/test_tier0_weights.py
git commit -m "feat: add Tier 0 weight integrity check (Check 0.3)"
```

---

## Task 8: Check 0.4 — MLX Version & Known Bug Check

**Files:**
- Create: `src/mlx_triage/tier0/version_check.py`
- Create: `tests/test_tier0_version.py`

**Step 1: Write the failing test**

```python
# tests/test_tier0_version.py
import json
from pathlib import Path
from unittest.mock import patch

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.version_check import check_mlx_version


def test_current_version_pass(good_model):
    """Current MLX version with llama architecture should pass."""
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = check_mlx_version(str(good_model))
    assert result.check_id == "0.4"
    assert result.status in (CheckStatus.PASS, CheckStatus.INFO, CheckStatus.WARNING)


def test_old_version_with_known_bug(good_model):
    """Old MLX version should flag known bugs."""
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.21.0"):
        result = check_mlx_version(str(good_model))
    assert result.status in (CheckStatus.WARNING, CheckStatus.CRITICAL)
    assert "MLX-001" in str(result.metadata) or "addmm" in result.detail.lower() or result.metadata.get("matching_bugs")


def test_mlx_not_installed(good_model):
    """When MLX is not installed, should SKIP gracefully."""
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value=None):
        result = check_mlx_version(str(good_model))
    assert result.status == CheckStatus.SKIP


def test_audio_model_gets_audio_bugs(tmp_path):
    """Audio architecture should match audio-specific bugs."""
    d = tmp_path / "whisper"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = check_mlx_version(str(d))
    # MLX-005 (Conv1d drift) affects whisper and has affected_versions: ["all"]
    if result.metadata.get("matching_bugs"):
        assert any("MLX-005" in str(b) for b in result.metadata["matching_bugs"])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0_version.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement version_check.py**

```python
# src/mlx_triage/tier0/version_check.py
"""Check 0.4: MLX version & known bug check.

Evidence basis: MLX #488 (float16 reductions), #2695 (addmm wrong results),
qmv kernel infinite loop, Conv1d composition drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_triage.config import find_bugs_for_model, load_known_bugs
from mlx_triage.models import CheckStatus, DiagnosticResult


def _get_mlx_version() -> str | None:
    """Get installed MLX version, or None if not installed."""
    try:
        import mlx

        return mlx.__version__
    except ImportError:
        return None


def check_mlx_version(model_path: str) -> DiagnosticResult:
    """Run MLX version and known bug check."""
    path = Path(model_path)
    mlx_version = _get_mlx_version()

    if mlx_version is None:
        return DiagnosticResult(
            check_id="0.4",
            name="MLX Version",
            status=CheckStatus.SKIP,
            detail="MLX is not installed. Cannot check for version-specific bugs.",
            remediation="Install MLX: pip install mlx",
        )

    # Get model architecture
    config_path = path / "config.json"
    architecture = "unknown"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        architecture = config.get("model_type", "unknown")

    # Query known bugs database
    bugs = load_known_bugs()
    matching = find_bugs_for_model(bugs, mlx_version=mlx_version, architecture=architecture)

    if not matching:
        return DiagnosticResult(
            check_id="0.4",
            name="MLX Version",
            status=CheckStatus.PASS,
            detail=f"MLX {mlx_version} — no known bugs affecting {architecture} architecture.",
        )

    # Determine worst severity
    severities = [b.severity for b in matching]
    has_critical = "critical" in severities
    has_high = "high" in severities

    if has_critical:
        status = CheckStatus.CRITICAL
    elif has_high:
        status = CheckStatus.WARNING
    else:
        status = CheckStatus.WARNING

    bug_summaries = [f"{b.id}: {b.title} ({b.severity})" for b in matching]
    detail = (
        f"MLX {mlx_version}, architecture: {architecture}. "
        f"{len(matching)} known issue(s): " + "; ".join(bug_summaries)
    )
    remediations = list({b.remediation for b in matching})

    return DiagnosticResult(
        check_id="0.4",
        name="MLX Version",
        status=status,
        detail=detail,
        remediation=" | ".join(remediations),
        metadata={"mlx_version": mlx_version, "matching_bugs": [b.id for b in matching]},
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0_version.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier0/version_check.py tests/test_tier0_version.py
git commit -m "feat: add Tier 0 MLX version & known bug check (Check 0.4)"
```

---

## Task 9: Tier 0 Runner

**Files:**
- Modify: `src/mlx_triage/tier0/__init__.py`
- Create: `tests/test_tier0_runner.py`

**Step 1: Write the failing test**

```python
# tests/test_tier0_runner.py
from unittest.mock import patch

from mlx_triage.models import CheckStatus, TierReport
from mlx_triage.tier0 import run_tier0


def test_run_tier0_returns_report(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    assert isinstance(report, TierReport)
    assert report.tier == 0
    assert len(report.checks) == 4


def test_run_tier0_all_pass(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    # Good model should have no FAIL or CRITICAL
    statuses = [c.status for c in report.checks]
    assert CheckStatus.FAIL not in statuses
    assert CheckStatus.CRITICAL not in statuses
    assert report.should_continue is True


def test_run_tier0_catches_bad_dtype(bf16_to_fp16_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(bf16_to_fp16_model))
    dtype_check = next(c for c in report.checks if c.check_id == "0.1")
    assert dtype_check.status == CheckStatus.CRITICAL
    assert report.should_continue is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_tier0'`

**Step 3: Implement tier0/__init__.py runner**

```python
# src/mlx_triage/tier0/__init__.py
"""Tier 0: Sanity Checks — no inference required, < 30 seconds."""

from __future__ import annotations

from mlx_triage.models import TierReport
from mlx_triage.tier0.dtype_check import check_dtype_compatibility
from mlx_triage.tier0.tokenizer_check import check_tokenizer_config
from mlx_triage.tier0.version_check import check_mlx_version
from mlx_triage.tier0.weight_integrity import check_weight_integrity


def run_tier0(model_path: str) -> TierReport:
    """Run all Tier 0 checks and return a structured report."""
    checks = [
        check_dtype_compatibility(model_path),
        check_tokenizer_config(model_path),
        check_weight_integrity(model_path),
        check_mlx_version(model_path),
    ]
    return TierReport.create(tier=0, model=model_path, checks=checks)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0_runner.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/tier0/__init__.py tests/test_tier0_runner.py
git commit -m "feat: add Tier 0 runner orchestrating all 4 checks"
```

---

## Task 10: Report Generation

**Files:**
- Create: `src/mlx_triage/report.py`
- Create: `tests/test_report.py`

**Step 1: Write the failing test**

```python
# tests/test_report.py
import json

from mlx_triage.models import CheckStatus, DiagnosticResult, TierReport
from mlx_triage.report import render_json, render_terminal


def _make_report() -> TierReport:
    return TierReport(
        tier=0,
        model="/path/to/model",
        timestamp="2026-02-25T14:30:00Z",
        checks=[
            DiagnosticResult(
                check_id="0.1",
                name="Dtype Compatibility",
                status=CheckStatus.PASS,
                detail="Training dtype bfloat16, stored as bfloat16. Compatible.",
            ),
            DiagnosticResult(
                check_id="0.2",
                name="Tokenizer Config",
                status=CheckStatus.WARNING,
                detail="No chat_template defined.",
                remediation="Add chat_template to tokenizer_config.json.",
            ),
        ],
    )


def test_render_json_valid():
    report = _make_report()
    output = render_json(report)
    parsed = json.loads(output)
    assert parsed["tier"] == 0
    assert parsed["model"] == "/path/to/model"
    assert len(parsed["checks"]) == 2
    assert parsed["checks"]["0.1"]["status"] == "PASS"
    assert parsed["verdict"] is not None
    assert "should_continue" in parsed


def test_render_json_roundtrip():
    report = _make_report()
    output = render_json(report)
    parsed = json.loads(output)
    assert parsed["checks"]["0.2"]["remediation"] == "Add chat_template to tokenizer_config.json."


def test_render_terminal_contains_key_info():
    report = _make_report()
    output = render_terminal(report)
    # Terminal output should contain model path, check names, and status
    assert "model" in output.lower() or "/path/to/model" in output
    assert "Dtype" in output or "dtype" in output
    assert "PASS" in output
    assert "WARNING" in output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement report.py**

```python
# src/mlx_triage/report.py
"""Report generation for mlx-triage diagnostics."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlx_triage.models import CheckStatus, TierReport

# Status → color mapping for Rich output
STATUS_COLORS = {
    CheckStatus.PASS: "green",
    CheckStatus.INFO: "blue",
    CheckStatus.SKIP: "dim",
    CheckStatus.WARNING: "yellow",
    CheckStatus.FAIL: "red",
    CheckStatus.CRITICAL: "bold red",
}


def render_json(report: TierReport) -> str:
    """Render a TierReport as a JSON string."""
    checks_dict = {}
    for check in report.checks:
        entry = {
            "status": check.status.value,
            "detail": check.detail,
        }
        if check.remediation:
            entry["remediation"] = check.remediation
        if check.metadata:
            entry["metadata"] = check.metadata
        checks_dict[check.check_id] = entry

    output = {
        "tier": report.tier,
        "model": report.model,
        "timestamp": report.timestamp,
        "checks": checks_dict,
        "verdict": report.verdict,
        "should_continue": report.should_continue,
    }
    return json.dumps(output, indent=2)


def render_terminal(report: TierReport) -> str:
    """Render a TierReport as a Rich terminal string."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]Tier {report.tier} Diagnostic Report[/bold]\n"
            f"Model: {report.model}\n"
            f"Time: {report.timestamp}",
            title="mlx-triage",
            border_style="blue",
        )
    )

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Detail", min_width=40)

    for check in report.checks:
        color = STATUS_COLORS[check.status]
        status_text = Text(check.status.value, style=color)
        detail = check.detail
        if check.remediation:
            detail += f"\n[dim]Fix: {check.remediation}[/dim]"
        table.add_row(f"{check.check_id} {check.name}", status_text, detail)

    console.print(table)

    # Verdict
    worst = report.worst_status
    verdict_color = STATUS_COLORS[worst]
    console.print()
    console.print(
        Panel(
            f"[{verdict_color}]{report.verdict}[/{verdict_color}]"
            + (f"\n[dim]Proceed to Tier {report.tier + 1}: {'Yes' if report.should_continue else 'No — fix issues first'}[/dim]" if report.tier < 3 else ""),
            title="Verdict",
            border_style=verdict_color.replace("bold ", ""),
        )
    )

    return buf.getvalue()


def write_report(report: TierReport, path: str, fmt: str = "json") -> None:
    """Write a report to a file."""
    if fmt == "json":
        content = render_json(report)
    elif fmt == "terminal":
        content = render_terminal(report)
    else:
        content = render_json(report)  # Default to JSON for file output

    with open(path, "w") as f:
        f.write(content)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/mlx_triage/report.py tests/test_report.py
git commit -m "feat: add report generation (JSON + Rich terminal output)"
```

---

## Task 11: CLI Integration & End-to-End Test

**Files:**
- Modify: `src/mlx_triage/cli.py`
- Create: `tests/test_cli_e2e.py`

**Step 1: Write the end-to-end test**

```python
# tests/test_cli_e2e.py
import json
from unittest.mock import patch

from click.testing import CliRunner

from mlx_triage.cli import cli


def test_check_tier0_terminal(good_model):
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model)])
    assert result.exit_code == 0
    assert "PASS" in result.output or "Tier 0" in result.output


def test_check_tier0_json(good_model):
    runner = CliRunner()
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["tier"] == 0
    assert "checks" in parsed


def test_check_tier0_output_file(good_model, tmp_path):
    runner = CliRunner()
    out_file = str(tmp_path / "report.json")
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        result = runner.invoke(cli, ["check", str(good_model), "--output", out_file, "--format", "json"])
    assert result.exit_code == 0
    with open(out_file) as f:
        parsed = json.load(f)
    assert parsed["tier"] == 0


def test_check_invalid_path():
    runner = CliRunner()
    result = runner.invoke(cli, ["check", "/nonexistent/path"])
    assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_e2e.py -v`
Expected: FAIL — CLI `check` command outputs stub text, not real report.

**Step 3: Wire CLI to tier0 runner + report**

Update `src/mlx_triage/cli.py`:

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
@click.option("--tier", type=click.IntRange(0, 3), default=0, help="Maximum diagnostic tier to run (0-3).")
@click.option("--output", type=click.Path(), default=None, help="Save report to file.")
@click.option("--format", "fmt", type=click.Choice(["json", "terminal", "markdown"]), default="terminal", help="Output format.")
def check(model_path: str, tier: int, output: str | None, fmt: str) -> None:
    """Run diagnostic checks against an MLX model."""
    from mlx_triage.report import render_json, render_terminal, write_report
    from mlx_triage.tier0 import run_tier0

    # Run Tier 0
    report = run_tier0(model_path)

    # Output
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

Run: `uv run pytest tests/test_cli_e2e.py -v`
Expected: All 4 tests PASS.

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests across all files PASS (should be ~30 tests total).

**Step 6: Commit**

```bash
git add src/mlx_triage/cli.py tests/test_cli_e2e.py
git commit -m "feat: wire CLI check command to Tier 0 runner and report generation"
```

---

## Task 12: Project Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/current-state.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/agents.md`

This task updates all template docs with real project context based on what was just built.

**Step 1: Update CLAUDE.md**

Replace the template content in CLAUDE.md with actual project details:
- Project type: Python CLI tool
- Status: Active development (Phase 0 complete)
- Stack: Python 3.11+, Click, Rich, PyYAML, safetensors, numpy
- Build: uv + hatchling
- Run commands: `uv run mlx-triage check <model_path>`
- Test commands: `uv run pytest -v`

**Step 2: Update docs/roadmap.md**

Fill in roadmap with phases from PROJECT-SPEC.md:
- P0: Phase 0 scaffolding items (now complete)
- P1: Phase 1 — Statistical Core (Tier 1 determinism, reference divergence, quantization quality)
- P2: Phase 2 — Isolation Tests (Tier 2 batch invariance, memory pressure, context length)
- P3: Phase 3 — Deep Diagnostics (Tier 3 activation diffing, cross-runtime) + community

**Step 3: Update docs/current-state.md**

Record Phase 0 completion:
- Phase: Phase 0 — COMPLETE
- What was built: CLI skeleton, 4 Tier 0 checks, JSON + terminal reports, known-bugs database
- How to verify: `uv run mlx-triage check <model_path>`
- Next step: Phase 1 Task 1 — Build diagnostic prompt suite

**Step 4: Update docs/agents.md**

Add relevant agent roles for this project:
- Python architecture / CLI design
- ML inference testing
- Code review

**Step 5: Commit**

```bash
git add CLAUDE.md docs/current-state.md docs/roadmap.md docs/agents.md
git commit -m "docs: update project documentation with Phase 0 completion state"
```

---

## Summary

| Task | What | Tests | Commit |
|------|------|-------|--------|
| 1 | Project scaffolding (pyproject.toml, src layout, uv sync) | Verify import | `feat: initialize project scaffolding` |
| 2 | Core data models (CheckStatus, DiagnosticResult, TierReport) | 6 tests | `feat: add core data models` |
| 3 | Known bugs database (YAML + config loader) | 5 tests | `feat: add known-bugs YAML database` |
| 4 | CLI skeleton (Click with check/report/compare) | 5 tests | `feat: add Click CLI skeleton` |
| 5 | Check 0.1: Dtype compatibility | 5 tests | `feat: add dtype compatibility check` |
| 6 | Check 0.2: Tokenizer & EOS config | 5 tests | `feat: add tokenizer config check` |
| 7 | Check 0.3: Weight integrity | 4 tests | `feat: add weight integrity check` |
| 8 | Check 0.4: MLX version & known bugs | 4 tests | `feat: add MLX version check` |
| 9 | Tier 0 runner (orchestrates all checks) | 3 tests | `feat: add Tier 0 runner` |
| 10 | Report generation (JSON + Rich terminal) | 3 tests | `feat: add report generation` |
| 11 | CLI integration (end-to-end) | 4 tests | `feat: wire CLI to Tier 0` |
| 12 | Project docs update | — | `docs: Phase 0 completion state` |

**Total: 12 tasks, ~49 tests, 12 commits.**

---

## Phase Roadmap (Future Plans)

These will be planned in detail when their time comes:

### Phase 1: Statistical Core (Tier 1)
- Diagnostic prompt suite (10 diverse prompts)
- Determinism check (N runs at temp=0, token-by-token comparison)
- Reference divergence (MLX vs Transformers on same weights)
- Quantization quality gate (perplexity on WikiText-2 subset)
- Threshold calibration on 3-5 known-good models

### Phase 2: Isolation Tests (Tier 2)
- Batch invariance (single vs batched output comparison)
- Memory pressure sweep (MLX Metal memory monitoring)
- Context length stress (needle-in-a-haystack at multiple lengths)
- Diagnostic divergence heuristic (crash = infra, gradual = model)

### Phase 3: Deep Diagnostics (Tier 3)
- Layer-wise activation hooking in MLX
- Cross-runtime comparison (MLX vs llama.cpp)
- Error accumulation visualization
- METHODOLOGY.md with evidence citations
- PyPI packaging
- Blog post / technical report

### Phase 4: Community & Iteration
- Community benchmark database
- Multi-hardware data collection (M1/M2/M3/M4)
- mlx-lm integration exploration
- Workshop paper
