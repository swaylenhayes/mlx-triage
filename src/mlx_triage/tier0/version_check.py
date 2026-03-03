"""Check 0.4: MLX Version & Known Bug Check.

Detects the installed MLX version and cross-references the known bugs
database to surface any bugs that affect the current model configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_triage.config import find_bugs_for_model, load_known_bugs
from mlx_triage.models import CheckStatus, DiagnosticResult

CHECK_ID = "0.4"
CHECK_NAME = "MLX Version & Known Bug Check"


def _get_mlx_version() -> str | None:
    """Return the installed MLX version string, or None if MLX is not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("mlx")
    except PackageNotFoundError:
        return None


def check_mlx_version(model_path: str) -> DiagnosticResult:
    """Run MLX version and known-bug check against *model_path*.

    Steps:
        1. Get installed MLX version (SKIP if not installed).
        2. Read ``config.json`` from *model_path* to determine ``model_type``.
        3. Query the known-bugs database for matching bugs.
        4. Return a result whose severity reflects the worst matching bug.
    """
    mlx_version = _get_mlx_version()

    if mlx_version is None:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.SKIP,
            detail="MLX is not installed.",
            remediation="Install MLX: pip install mlx",
        )

    # Read model architecture from config.json
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.SKIP,
            detail=f"config.json not found in {model_path}.",
            remediation="Ensure the model directory contains a valid config.json.",
            metadata={"mlx_version": mlx_version},
        )

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            detail=f"Failed to parse config.json: {exc}",
            metadata={"mlx_version": mlx_version},
        )

    architecture = config.get("model_type", "unknown")

    # Query known bugs
    bugs = load_known_bugs()
    matches = find_bugs_for_model(bugs, mlx_version, architecture)

    if not matches:
        return DiagnosticResult(
            check_id=CHECK_ID,
            name=CHECK_NAME,
            status=CheckStatus.PASS,
            detail=f"MLX {mlx_version}, architecture '{architecture}': no known bugs match.",
            metadata={"mlx_version": mlx_version, "architecture": architecture},
        )

    # Determine worst severity among matched bugs.
    # Bugs that apply to ALL versions are perpetual advisories — cap their
    # contribution at WARNING.  Only version-specific bugs (those with at
    # least one concrete version constraint) can escalate to CRITICAL.
    matched_ids = [b.id for b in matches]

    version_specific = [b for b in matches if b.affected_versions != ["all"]]
    advisory = [b for b in matches if b.affected_versions == ["all"]]

    effective_severities: set[str] = set()
    for b in version_specific:
        effective_severities.add(b.severity)
    for b in advisory:
        # Cap advisory bugs at "warning" for status determination
        effective_severities.add(
            b.severity if b.severity in ("warning", "info") else "warning"
        )

    if "critical" in effective_severities:
        status = CheckStatus.CRITICAL
    elif "high" in effective_severities:
        status = CheckStatus.FAIL
    elif "warning" in effective_severities:
        status = CheckStatus.WARNING
    else:
        status = CheckStatus.INFO

    remediation_lines = [f"  - {b.id}: {b.remediation}" for b in matches]
    remediation = "Matched known bugs:\n" + "\n".join(remediation_lines)

    return DiagnosticResult(
        check_id=CHECK_ID,
        name=CHECK_NAME,
        status=status,
        detail=(
            f"MLX {mlx_version}, architecture '{architecture}': "
            f"{len(matches)} known bug(s) match."
        ),
        remediation=remediation,
        metadata={
            "mlx_version": mlx_version,
            "architecture": architecture,
            "matched_bug_ids": matched_ids,
        },
    )
