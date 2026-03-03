"""Configuration and known-bugs database loader."""

from __future__ import annotations

from dataclasses import dataclass
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
    """Parse a version string like '0.22.0' into a comparable tuple.

    Stops at the first non-numeric component, so '0.22.0.dev0' -> (0, 22, 0).
    """
    parts: list[int] = []
    for x in v.strip().split("."):
        try:
            parts.append(int(x))
        except ValueError:
            break
    return tuple(parts)


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
