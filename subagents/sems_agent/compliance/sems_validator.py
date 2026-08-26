"""
SEMS Compliance Validator — unified pre-SonarQube compliance report.

Wraps the existing two compliance modules (compliance_checker.py for
AST + regex SEMS rules, pre_sonar_check.py for the 10 external static-analysis
tools) behind a single entry point that:

  1. Runs Layer 1 — syntax check (ast.parse)
  2. Runs Layer 2 — AST + regex SEMS rules
  3. Runs Layer 3 — 10 external tools (black, isort, flake8, pylint,
                    pylint-simplify, mypy, bandit, radon, pydocstyle,
                    pytest-cov)
  4. Aggregates findings into the 5 BRD SEMS areas
        - Code readability and structure
        - Modular design
        - Logging practices
        - Error handling
        - Security
  5. Flags severity-aware blocking findings (hard SEMS violations and
     MAJOR+ external-tool findings) per BRD area, for information only.

This module reports SEMS compliance — it does not gate or block anything.
It is also read-only and never modifies the target file. When Layer 1
detects a syntax error, Layers 2 and 3 are skipped and the syntax-fail
report is returned as-is.

    # Pipeline integration
    from subagents.sems_agent.compliance.sems_validator import run_sems_gate
    report = run_sems_gate(output_path)
    print(render_markdown(report))
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .compliance_checker import (
    ComplianceResult,
    SEMSViolation,
    check_compliance,
)
from .pre_sonar_check import (
    BLOCKER,
    CRITICAL,
    HOTSPOT,
    MAJOR,
    Issue,
    ValidationReport,
    validate_file,
)

logger = logging.getLogger(__name__)

# ── BRD SEMS area definitions ─────────────────────────────────────────────────

BRD_AREAS = (
    "code_readability_and_structure",
    "modular_design",
    "logging_practices",
    "error_handling",
    "security",
)

# Rule-ID → BRD area. Used for SEMSViolation objects from compliance_checker.
_SEMS_RULE_TO_BRD_AREA: Dict[str, str] = {
    # syntax / structure
    "SYN001": "code_readability_and_structure",
    # security
    "SEC001": "security",
    "SEC002": "security",
    "SEC003": "security",
    # error handling
    "SEMS_ERR001": "error_handling",
    "ERR001": "error_handling",
    "LOG001": "logging_practices",
    "LOG002": "logging_practices",
    "LOG003": "logging_practices",
    "ERR002": "error_handling",
    # readability / structure
    "DOC001": "code_readability_and_structure",
    "DOC002": "code_readability_and_structure",
    "DOC003": "code_readability_and_structure",
    "SPARK003": "code_readability_and_structure",
    # modular design
    "STYLE001": "modular_design",
    # Spark best-practice / Databricks compat
    "SPARK001": "modular_design",
    "SPARK002": "modular_design",
    "SPARK004": "modular_design",
    "SPARK005": "modular_design",
    "SPARK006": "modular_design",
    "SPARK007": "modular_design",
    "SPARK008": "modular_design",
    "SPARK009": "code_readability_and_structure",
    "SPARK010": "modular_design",
    # Shell company rules — security / PII
    "SHELL_PII001": "security",
    "SHELL_PII002": "security",
    # Shell company rules — naming conventions
    "SHELL_NAME001": "code_readability_and_structure",
    # Shell company rules — Spark / Databricks
    "SHELL_SPARK001": "modular_design",
    "SHELL_SPARK002": "modular_design",
    "SHELL_STAM001": "modular_design",
    "SHELL_STAM002": "modular_design",
    # Construction Policy alignment — literals, comments, quality metrics
    "LIT001": "code_readability_and_structure",
    "LIT002": "code_readability_and_structure",
    "COM001": "code_readability_and_structure",
    "COM002": "code_readability_and_structure",
    "FUNC001": "code_readability_and_structure",
    "EXPR001": "code_readability_and_structure",
    "FANOUT001": "modular_design",
    "NEST001": "modular_design",
    "CONST001": "code_readability_and_structure",
    "FILE001": "modular_design",
    "DUP001": "modular_design",
    "STMT001": "code_readability_and_structure",
    "PARAM001": "code_readability_and_structure",
    "ATTR001": "code_readability_and_structure",
}

# External-tool name → BRD area. Used for pre_sonar_check.Issue objects.
_TOOL_TO_BRD_AREA: Dict[str, str] = {
    "black": "code_readability_and_structure",
    "isort": "code_readability_and_structure",
    "flake8": "code_readability_and_structure",
    "pylint": "code_readability_and_structure",
    "pylint-simplify": "modular_design",
    "mypy": "error_handling",
    "bandit": "security",
    "radon": "modular_design",
    "pydocstyle": "code_readability_and_structure",
    "pytest-cov": "modular_design",
}


def _brd_area_for_violation(violation: SEMSViolation) -> str:
    return _SEMS_RULE_TO_BRD_AREA.get(
        violation.rule_id, "code_readability_and_structure"
    )


def _brd_area_for_issue(issue: Issue) -> str:
    # Pylint W0702 (bare-except) and B902 → error handling
    if issue.tool == "pylint" and issue.rule_id in {"W0702", "W0703", "W0706"}:
        return "error_handling"
    return _TOOL_TO_BRD_AREA.get(issue.tool, "code_readability_and_structure")


# ── Aggregated report ────────────────────────────────────────────────────────


@dataclass
class AreaSummary:
    name: str
    sems_findings: int = 0  # total SEMS violations (display only)
    tool_findings: int = 0  # total tool findings (display only)
    sems_blocking: int = 0  # hard-severity SEMS violations that block the gate
    tool_blocking: int = 0  # MAJOR/CRITICAL/BLOCKER or security-hotspot tool findings that block the gate

    @property
    def total(self) -> int:
        return self.sems_findings + self.tool_findings

    @property
    def passed(self) -> bool:
        # Severity classification: SOFT SEMS warnings and MINOR tool findings
        # are informational only. Hard SEMS violations and MAJOR+ tool
        # findings are flagged as blocking-severity (informational — this
        # does not gate or block anything).
        return self.sems_blocking == 0 and self.tool_blocking == 0


@dataclass
class SemsReport:
    target: str
    syntax_ok: bool
    sems_result: Optional[ComplianceResult] = None
    tool_report: Optional[ValidationReport] = None
    syntax_error: Optional[str] = None
    area_summaries: Dict[str, AreaSummary] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when Layer 1 parsed cleanly and no BRD area has a
        blocking-severity finding (hard SEMS violation, MAJOR+ tool finding,
        missing/crashed analyzer, or security hotspot)."""
        return self.syntax_ok and all(a.passed for a in self.area_summaries.values())


# ── Validator core ───────────────────────────────────────────────────────────


def _layer1_syntax(source: str) -> Optional[str]:
    """Layer 1 — ast.parse. Returns the error message if invalid, else None."""
    try:
        ast.parse(source)
        return None
    except SyntaxError as exc:
        return f"{exc.msg} at line {exc.lineno}"


def _aggregate_areas(
    sems_result: Optional[ComplianceResult],
    tool_report: Optional[ValidationReport],
) -> Dict[str, AreaSummary]:
    """Build a per-BRD-area summary from SEMS violations + external tool findings."""
    summaries: Dict[str, AreaSummary] = {
        area: AreaSummary(name=area) for area in BRD_AREAS
    }
    if sems_result:
        for sv in sems_result.structured_violations:
            area = _brd_area_for_violation(sv)
            summaries[area].sems_findings += 1
            if sv.severity != "soft":
                summaries[area].sems_blocking += 1
    if tool_report:
        for issue in tool_report.all_issues:
            area = _brd_area_for_issue(issue)
            summaries[area].tool_findings += 1
            if (
                issue.sonar_severity in (BLOCKER, CRITICAL, MAJOR)
                or issue.sonar_type == HOTSPOT
            ):
                summaries[area].tool_blocking += 1
    return summaries


def validate_source(
    source: str,
    target_path: Path,
) -> SemsReport:
    """
    Validate a PySpark source string through all 3 layers (single attempt).
    Reports findings and their severity (hard SEMS violations and MAJOR+
    external-tool findings are flagged as blocking-severity); does not gate
    or block anything. Never modifies the source file.
    """
    syntax_error = _layer1_syntax(source)
    if syntax_error:
        return SemsReport(
            target=str(target_path),
            syntax_ok=False,
            syntax_error=syntax_error,
            area_summaries=_aggregate_areas(None, None),
        )

    sems_result = check_compliance(source)
    tool_report = validate_file(target_path, fix=False)
    areas = _aggregate_areas(sems_result, tool_report)

    return SemsReport(
        target=str(target_path),
        syntax_ok=True,
        sems_result=sems_result,
        tool_report=tool_report,
        area_summaries=areas,
    )


# ── Report rendering ─────────────────────────────────────────────────────────

_AREA_LABEL: Dict[str, str] = {
    "code_readability_and_structure": "Code readability and structure",
    "modular_design": "Modular design",
    "logging_practices": "Logging practices",
    "error_handling": "Error handling",
    "security": "Security",
}


def render_markdown(report: SemsReport) -> str:
    lines: List[str] = []
    lines.append("# SEMS Compliance Validation Report")
    lines.append("")
    lines.append(f"**Target**: `{report.target}`")
    lines.append("")
    lines.append(f"**Overall status**: {'PASS' if report.passed else 'FAIL'}")
    lines.append("")

    if not report.syntax_ok:
        lines.append("## Layer 1 — Syntax")
        lines.append("")
        lines.append(f"`SyntaxError`: {report.syntax_error}")
        lines.append("")
        lines.append("_Layers 2 and 3 were skipped because the source did not parse._")
        return "\n".join(lines)

    lines.append("## BRD Coverage (5 SEMS areas)")
    lines.append("")
    lines.append(
        "| Area | Status | SEMS findings | Tool findings | Blocking SEMS | Blocking tools |"
    )
    lines.append(
        "|------|--------|---------------|---------------|---------------|----------------|"
    )
    for area in BRD_AREAS:
        summary = report.area_summaries[area]
        status = "PASS" if summary.passed else "FAIL"
        lines.append(
            f"| {_AREA_LABEL[area]} | {status} | "
            f"{summary.sems_findings} | {summary.tool_findings} | "
            f"{summary.sems_blocking} | {summary.tool_blocking} |"
        )
    lines.append("")

    if report.tool_report:
        lines.append("## 10-Policy Tool Summary")
        lines.append("")
        lines.append("| Tool | Status | Issues | Notes |")
        lines.append("|------|--------|--------|-------|")
        for run in report.tool_report.tool_runs:
            status = "PASS" if run.passed else "ISSUES"
            issue_count = len(run.issues) if run.issues else 0
            note = run.notes or ""
            lines.append(f"| {run.name} | {status} | {issue_count} | {note} |")
        lines.append("")

    lines.append("## Findings by BRD Area")
    for area in BRD_AREAS:
        summary = report.area_summaries[area]
        if summary.total == 0:
            continue
        lines.append("")
        status = "PASS" if summary.passed else "FAIL"
        lines.append(f"### {_AREA_LABEL[area]} ({status}, {summary.total})")
        lines.append("")
        if report.sems_result:
            relevant_sems = [
                sv
                for sv in report.sems_result.structured_violations
                if _brd_area_for_violation(sv) == area
            ]
            for sv in relevant_sems:
                lines.append(
                    f"- **[{sv.severity.upper()}] {sv.rule_id}** — {sv.message}"
                )
                lines.append(f"  _Remediation_: {sv.remediation.strip()}")
        if report.tool_report:
            relevant_tool = [
                issue
                for issue in report.tool_report.all_issues
                if _brd_area_for_issue(issue) == area
            ]
            for issue in relevant_tool:
                loc = f"line {issue.line}" if issue.line else "module"
                lines.append(
                    f"- **[{issue.sonar_severity}] {issue.tool}/{issue.rule_id}** "
                    f"({loc}) — {issue.message}"
                )

    lines.append("")
    lines.append("## Remediation Checklist")
    lines.append("")
    if all(summary.passed for summary in report.area_summaries.values()):
        lines.append("- [x] All 5 SEMS areas have no blocking findings")
    else:
        for area in BRD_AREAS:
            summary = report.area_summaries[area]
            if not summary.passed:
                lines.append(
                    f"- [ ] Fix {summary.total} finding(s) in **{_AREA_LABEL[area]}**"
                )
    lines.append("")
    return "\n".join(lines)


def render_json(report: SemsReport) -> dict:
    return {
        "target": report.target,
        "syntax_ok": report.syntax_ok,
        "syntax_error": report.syntax_error,
        "passed": report.passed,
        "areas": {
            area: {
                "label": _AREA_LABEL[area],
                "passed": summary.passed,
                "sems_findings": summary.sems_findings,
                "tool_findings": summary.tool_findings,
                "sems_blocking": summary.sems_blocking,
                "tool_blocking": summary.tool_blocking,
            }
            for area, summary in report.area_summaries.items()
        },
        "sems_violations": [
            asdict(sv)
            for sv in (
                report.sems_result.structured_violations if report.sems_result else []
            )
        ],
        "tool_report": report.tool_report.to_dict() if report.tool_report else None,
    }


def _write_reports(report: SemsReport, target_path: Path) -> Dict[str, Path]:
    md_path = target_path.with_name(target_path.stem + "_sems_validation.md")
    json_path = target_path.with_name(target_path.stem + "_sems_validation.json")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(render_json(report), indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


# ── Pipeline-callable entry point ────────────────────────────────────────────


def run_sems_gate(target_path: Path | str, write_reports: bool = True) -> SemsReport:
    """
    Validate a single PySpark file end-to-end.

    Runs all 3 layers (syntax → SEMS rules → external tools) and aggregates
    findings into the 5 BRD areas. Read-only: never modifies the target
    file. When Layer 1 detects a syntax error, Layers 2 and 3 are skipped
    and the syntax-fail report is returned as-is. This is a compliance
    report, not a gate — it never blocks anything.

    Args:
        write_reports: When True (the default, used by the CLI), writes
            ``<stem>_sems_validation.{md,json}`` next to the target file.
            Callers that persist their own report (e.g. sems_agent, which
            writes ``outputs/<stem>_sems_report.{md,json}``) should pass
            False to avoid two divergent report files for the same run.
    """
    path = Path(target_path)
    source = path.read_text(encoding="utf-8")

    report = validate_source(source, path)

    if not report.syntax_ok:
        logger.warning("SEMS gate: Layer 1 SyntaxError — %s", report.syntax_error)
    else:
        failing_areas = [k for k, v in report.area_summaries.items() if not v.passed]
        if failing_areas:
            logger.warning(
                "SEMS gate: blocking-severity findings in areas: %s",
                ", ".join(failing_areas),
            )
        else:
            logger.info("SEMS gate: no blocking-severity findings")

    if write_reports:
        written = _write_reports(report, path)
        logger.info("SEMS validation report written to: %s", written["markdown"])
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def _format_console_summary(report: SemsReport) -> str:
    lines = [
        "",
        "=" * 72,
        f"  SEMS COMPLIANCE VALIDATION — {Path(report.target).name}",
        "=" * 72,
        "",
        f"  Overall status: {'PASS' if report.passed else 'FAIL'}",
        f"  Syntax       : {'OK' if report.syntax_ok else 'FAIL — ' + (report.syntax_error or '')}",
        "",
        "  BRD Area Coverage:",
    ]
    for area in BRD_AREAS:
        summary = report.area_summaries[area]
        status = "PASS" if summary.passed else "FAIL"
        lines.append(
            f"    {status:<5}  {_AREA_LABEL[area]:<35}  "
            f"SEMS={summary.sems_findings}  TOOLS={summary.tool_findings}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "SEMS Compliance Validator — runs 3 layers (syntax, SEMS rules, "
            "10 external tools) and reports findings by severity. Report only; "
            "does not gate or block."
        ),
    )
    parser.add_argument("target", help="Python file to validate")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Print the JSON report to stdout (in addition to the file)",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 2
    if not target.is_file():
        print(
            f"Error: {target} is not a file (directory-mode not supported)",
            file=sys.stderr,
        )
        return 2

    report = run_sems_gate(target)
    print(_format_console_summary(report))
    if args.emit_json:
        print(json.dumps(render_json(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
