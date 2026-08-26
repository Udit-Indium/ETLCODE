"""Shared validation report/error model used by the SEMS agents to report
findings in a common shape, and to persist them to outputs/ uniformly."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(str, Enum):
    SYNTAX = "syntax"
    SECURITY = "security"
    OTHER = "other"


@dataclass
class ValidationError:
    category: str
    title: str
    reason: str
    description: str
    source_agent: str
    severity: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    agent: str
    passed: bool
    summary: str
    errors: List[ValidationError] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "passed": self.passed,
            "summary": self.summary,
            "errors": [e.to_dict() for e in self.errors],
            "details": self.details,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.agent} Validation Report",
            "",
            f"**Status**: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"**Summary**: {self.summary}",
            "",
        ]
        if self.errors:
            lines.append("## Findings")
            lines.append("")
            for e in self.errors:
                lines.append(f"- **[{e.severity.upper()}] {e.title}** ({e.category}) — {e.reason}")
                if e.location:
                    lines.append(f"  - Location: {e.location}")
                if e.suggestion:
                    lines.append(f"  - Suggestion: {e.suggestion}")
        return "\n".join(lines)


def write_report(
    outputs_dir: Path,
    stem: str,
    agent_name: str,
    report: ValidationReport,
    *,
    markdown_override: Optional[str] = None,
) -> Dict[str, Path]:
    """Write ``<stem>_<agent_name>_report.{md,json}`` into ``outputs_dir``."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    md_path = outputs_dir / f"{stem}_{agent_name}_report.md"
    json_path = outputs_dir / f"{stem}_{agent_name}_report.json"

    md_path.write_text(markdown_override or report.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {"markdown": md_path, "json": json_path}
