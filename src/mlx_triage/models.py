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
