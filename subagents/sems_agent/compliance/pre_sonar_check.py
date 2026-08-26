"""
Pre-SonarQube SEMS compliance validator for generated PySpark code.

Maps findings from 10 static analysis tools to SonarQube issue categories
(Bugs, Vulnerabilities, Security Hotspots, Code Smells) and applies the
Sonar Way Quality Gate thresholds before handover to the Shell cell team.

SonarQube category mapping:
  black + isort       → CODE_SMELL / MINOR    (formatting/import order)
  flake8 E/W          → CODE_SMELL / MINOR-MAJOR
  flake8 E711/E712    → BUG / MAJOR           (comparison logic errors)
  pylint error/fatal  → BUG / CRITICAL-BLOCKER
  pylint warning      → CODE_SMELL / MAJOR
  pylint convention   → CODE_SMELL / MINOR
  pylint-simplify     → CODE_SMELL / MINOR-MAJOR (simplification recommendations)
  mypy error          → BUG / MAJOR
  bandit HIGH+HIGH    → VULNERABILITY / CRITICAL
  bandit HIGH+MED     → VULNERABILITY / MAJOR
  bandit MED          → SECURITY_HOTSPOT / MINOR
  radon CC 10-14      → CODE_SMELL / MAJOR     (Construction Policy: needs justification)
  radon CC 15+        → CODE_SMELL / CRITICAL  (Construction Policy: unacceptable)
  pydocstyle          → CODE_SMELL / MINOR
  pytest-cov <80%     → CODE_SMELL / MAJOR     (Sonar Way Quality Gate: coverage on new code)
  pytest-cov <50%     → CODE_SMELL / CRITICAL
  pytest-cov no tests → CODE_SMELL / CRITICAL  (coverage cannot be measured at all)

Usage:
    python -m src.compliance.pre_sonar_check <file_or_dir> [--fix] [--json]
    python src/compliance/pre_sonar_check.py output/customer_analytics_maverick_pyspark.py
    python src/compliance/pre_sonar_check.py output/ --fix
"""

import argparse
import concurrent.futures
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PYLINTRC = Path(__file__).parent.parent / ".pylintrc"

# ── SonarQube taxonomy ────────────────────────────────────────────────────────

BUG = "BUG"
VULNERABILITY = "VULNERABILITY"
HOTSPOT = "SECURITY_HOTSPOT"
SMELL = "CODE_SMELL"

BLOCKER = "BLOCKER"
CRITICAL = "CRITICAL"
MAJOR = "MAJOR"
MINOR = "MINOR"
INFO = "INFO"

# Sonar Way Quality Gate — hard limits. Tool availability is also mandatory:
_GATE_MAX_BUGS = 0
_GATE_MAX_VULNS = 0
_GATE_MAX_HOTSPOTS = 0
# Construction Policy quality-metric bands: 1-9 acceptable, 10-14 needs
# justification, 15+ unacceptable.
# Mirrored in the converter's generation guidance
# (conversion_loop/.../py2snow-skill/references/SEMS_Compliance.md §9) — keep in
# sync if retuned. See also _FANOUT_*/_EXPR_*/_NEST_* in compliance_checker.py.
_MAX_CYCLOMATIC_COMPLEXITY = 9
_CC_UNACCEPTABLE_MIN = 15

# Sonar Way Quality Gate — coverage on new code must be >= 80%.
_MIN_COVERAGE_PCT = 80.0
_CRITICAL_COVERAGE_PCT = 50.0  # below this, treat as CRITICAL rather than MAJOR

_SEVERITY_RANK = {BLOCKER: 5, CRITICAL: 4, MAJOR: 3, MINOR: 2, INFO: 1}
_RATING = {5: "E", 4: "D", 3: "C", 2: "B", 1: "B", 0: "A"}

# flake8 codes that are logic bugs, not just style — SonarQube treats these as BUG
_FLAKE8_BUG_CODES = {"E711", "E712", "E714", "E715", "E721", "W605", "F821", "F901"}

# flake8 codes with MAJOR severity in SonarQube (not just style noise)
_FLAKE8_MAJOR_PREFIXES = {"E7", "W6", "C9"}

# Databricks notebook globals injected by the cluster at runtime — not real undefined errors
_DATABRICKS_GLOBALS = frozenset(
    {
        "dbutils",
        "spark",
        "sc",
        "sqlContext",
        "display",
        "displayHTML",
        "getArgument",
        "table",
        "udf",
    }
)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Issue:
    file: str
    line: int
    col: int
    tool: str
    rule_id: str
    sonar_type: str
    sonar_severity: str
    message: str

    def format_line(self) -> str:
        loc = f"{Path(self.file).name}:{self.line}"
        return (
            f"    [{self.sonar_severity:<8}] ({self.tool:<12}) "
            f"{loc:<45} {self.rule_id:<12}  {self.message}"
        )


@dataclass
class ToolRun:
    name: str
    passed: bool
    issues: List[Issue] = field(default_factory=list)
    notes: str = ""


@dataclass
class ValidationReport:
    target: str
    tool_runs: List[ToolRun] = field(default_factory=list)

    @property
    def all_issues(self) -> List[Issue]:
        return [issue for run in self.tool_runs for issue in run.issues]

    def bugs(self) -> List[Issue]:
        return [i for i in self.all_issues if i.sonar_type == BUG]

    def vulnerabilities(self) -> List[Issue]:
        return [i for i in self.all_issues if i.sonar_type == VULNERABILITY]

    def hotspots(self) -> List[Issue]:
        return [i for i in self.all_issues if i.sonar_type == HOTSPOT]

    def smells(self) -> List[Issue]:
        return [i for i in self.all_issues if i.sonar_type == SMELL]

    def _worst_rank(self, issues: List[Issue]) -> int:
        return max((_SEVERITY_RANK.get(i.sonar_severity, 0) for i in issues), default=0)

    def reliability_rating(self) -> str:
        return _RATING[self._worst_rank(self.bugs())]

    def security_rating(self) -> str:
        return _RATING[self._worst_rank(self.vulnerabilities())]

    def maintainability_rating(self) -> str:
        return _RATING[self._worst_rank(self.smells())]

    def quality_gate_passed(self) -> bool:
        critical_smells = any(
            _SEVERITY_RANK.get(i.sonar_severity, 0) >= _SEVERITY_RANK[CRITICAL]
            for i in self.smells()
        )
        missing_policy = any(
            i.rule_id in ("TOOL_MISSING", "TOOL_ERROR") for i in self.all_issues
        )
        return (
            len(self.bugs()) <= _GATE_MAX_BUGS
            and len(self.vulnerabilities()) <= _GATE_MAX_VULNS
            and len(self.hotspots()) <= _GATE_MAX_HOTSPOTS
            and not critical_smells
            and not missing_policy
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "quality_gate_passed": self.quality_gate_passed(),
            "ratings": {
                "reliability": self.reliability_rating(),
                "security": self.security_rating(),
                "maintainability": self.maintainability_rating(),
            },
            "counts": {
                "bugs": len(self.bugs()),
                "vulnerabilities": len(self.vulnerabilities()),
                "security_hotspots": len(self.hotspots()),
                "code_smells": len(self.smells()),
                "total": len(self.all_issues),
            },
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "tool": i.tool,
                    "rule_id": i.rule_id,
                    "sonar_type": i.sonar_type,
                    "sonar_severity": i.sonar_severity,
                    "message": i.message,
                }
                for i in self.all_issues
            ],
        }


# ── Subprocess helper ─────────────────────────────────────────────────────────


_TOOL_TIMEOUT_SECS = 300  # matches code_convertor_agent.run_pyspark_file_tool


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=_TOOL_TIMEOUT_SECS
    )
    return result.returncode, result.stdout, result.stderr


def _missing_tool_run(tool_name: str, target: Path, executable: str) -> ToolRun:
    """Return a blocking issue when a mandatory external analyzer is unavailable."""
    return ToolRun(
        name=tool_name,
        passed=False,
        notes=f"{executable} executable not found",
        issues=[
            Issue(
                file=str(target),
                line=0,
                col=0,
                tool=tool_name,
                rule_id="TOOL_MISSING",
                sonar_type=SMELL,
                sonar_severity=MAJOR,
                message=(
                    f"Mandatory SEMS analyzer '{executable}' is not installed "
                    "or not available on PATH."
                ),
            )
        ],
    )


def _tool_error_run(tool_name: str, target: Path, detail: str) -> ToolRun:
    """Return a blocking issue when a mandatory analyzer crashes or emits garbage.

    A crashed analyzer must fail the gate the same way a missing one does —
    otherwise a corrupt rcfile or incompatible tool version silently skips
    one of the 10 mandatory policies.
    """
    return ToolRun(
        name=tool_name,
        passed=False,
        notes=f"{tool_name} failed: {detail}",
        issues=[
            Issue(
                file=str(target),
                line=0,
                col=0,
                tool=tool_name,
                rule_id="TOOL_ERROR",
                sonar_type=SMELL,
                sonar_severity=MAJOR,
                message=(
                    f"Mandatory SEMS analyzer '{tool_name}' did not produce a "
                    f"usable result: {detail}"
                ),
            )
        ],
    )


# ── Tool runners ──────────────────────────────────────────────────────────────


def run_black(target: Path, fix: bool = False) -> ToolRun:
    if fix:
        try:
            code, _, stderr = _run(["black", "--quiet", str(target)])
        except FileNotFoundError:
            return _missing_tool_run("black", target, "black")
        if code != 0:
            # black exits non-zero when it cannot format (e.g. internal error,
            # invalid syntax) — never report "Auto-formatted" for a file it
            # left untouched.
            return _tool_error_run(
                "black", target, (stderr.strip() or f"exit code {code}")[:300]
            )
        return ToolRun(name="black", passed=True, notes="Auto-formatted")
    try:
        code, _, _ = _run(["black", "--check", "--quiet", str(target)])
    except FileNotFoundError:
        return _missing_tool_run("black", target, "black")
    if code == 0:
        return ToolRun(name="black", passed=True, notes="Formatting clean")
    return ToolRun(
        name="black",
        passed=False,
        notes="Run with --fix to auto-format",
        issues=[
            Issue(
                file=str(target),
                line=0,
                col=0,
                tool="black",
                rule_id="PEP8-format",
                sonar_type=SMELL,
                sonar_severity=MINOR,
                message="File is not black-formatted — run: black <file>",
            )
        ],
    )


def run_isort(target: Path, fix: bool = False) -> ToolRun:
    if fix:
        try:
            code, _, stderr = _run(
                ["isort", "--quiet", "--profile", "black", str(target)]
            )
        except FileNotFoundError:
            return _missing_tool_run("isort", target, "isort")
        if code != 0:
            # isort exits non-zero when it cannot sort (e.g. unparsable file) —
            return _tool_error_run(
                "isort", target, (stderr.strip() or f"exit code {code}")[:300]
            )
        return ToolRun(name="isort", passed=True, notes="Imports sorted")
    try:
        code, _, _ = _run(
            ["isort", "--check-only", "--quiet", "--profile", "black", str(target)]
        )
    except FileNotFoundError:
        return _missing_tool_run("isort", target, "isort")
    if code == 0:
        return ToolRun(name="isort", passed=True, notes="Import order clean")
    return ToolRun(
        name="isort",
        passed=False,
        notes="Run with --fix to auto-sort",
        issues=[
            Issue(
                file=str(target),
                line=0,
                col=0,
                tool="isort",
                rule_id="import-order",
                sonar_type=SMELL,
                sonar_severity=MINOR,
                message="Imports are not sorted — run: isort --profile black <file>",
            )
        ],
    )


def run_flake8(target: Path) -> ToolRun:
    fmt = "%(path)s||%(row)d||%(col)d||%(code)s||%(text)s"
    try:
        code, stdout, stderr = _run(
            [
                "flake8",
                f"--format={fmt}",
                "--max-line-length=120",
                "--max-complexity=15",
                # E402: imports not at top — false positive for Databricks # COMMAND ---------- cell format
                # F811: redefinition of unused — each notebook cell re-imports its deps by design
                "--extend-ignore=E203,W503,W504,E402,F811",
                str(target),
            ]
        )
    except FileNotFoundError:
        return _missing_tool_run("flake8", target, "flake8")
    # flake8 exits non-zero whenever it reports findings — but if it crashes or emits nothing, fail the gate as TOOL_ERROR.
    if code != 0 and not stdout.strip():
        return _tool_error_run("flake8", target, (stderr.strip() or "no output")[:300])
    issues = []
    for raw_line in stdout.splitlines():
        parts = raw_line.split("||")
        if len(parts) < 5:
            continue
        filepath, row, col, code, text = (
            parts[0],
            int(parts[1]),
            int(parts[2]),
            parts[3],
            parts[4],
        )
        # F821 for Databricks cluster globals is a runtime injection, not a real undefined error — ignore it.
        if code == "F821" and any(f"'{g}'" in text for g in _DATABRICKS_GLOBALS):
            continue
        if code in _FLAKE8_BUG_CODES:
            sonar_type, severity = BUG, MAJOR
        elif code == "F401":
            sonar_type, severity = SMELL, MAJOR  # unused import = MAJOR in SonarQube
        elif any(code.startswith(p) for p in _FLAKE8_MAJOR_PREFIXES):
            sonar_type, severity = SMELL, MAJOR
        else:
            sonar_type, severity = SMELL, MINOR
        issues.append(
            Issue(
                file=filepath,
                line=row,
                col=col,
                tool="flake8",
                rule_id=code,
                sonar_type=sonar_type,
                sonar_severity=severity,
                message=text,
            )
        )
    return ToolRun(name="flake8", passed=not issues, issues=issues)


def run_pylint(target: Path) -> ToolRun:
    cmd = ["pylint", "--output-format=json"]
    if _PYLINTRC.exists():
        cmd.append(f"--rcfile={_PYLINTRC}")
    cmd.append(str(target))

    code, stdout, stderr = _run(cmd)

    if code != 0 and not stdout.strip():
        return _tool_error_run("pylint", target, (stderr.strip() or "no output")[:300])
    try:
        raw_items = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return _tool_error_run(
            "pylint", target, "non-JSON stdout: " + (stderr.strip() or stdout.strip())[:300]
        )

    score_match = re.search(r"rated at ([\d.]+/10)", stderr + stdout)
    score_note = f"Score: {score_match.group(1)}" if score_match else ""

    issues = []
    for item in raw_items:
        msg_type = item.get("type", "")
        if msg_type == "fatal":
            sonar_type, severity = BUG, BLOCKER
        elif msg_type == "error":
            sonar_type, severity = BUG, CRITICAL
        elif msg_type == "warning":
            sonar_type, severity = SMELL, MAJOR
        else:
            sonar_type, severity = SMELL, MINOR

        issues.append(
            Issue(
                file=item.get("path", str(target)),
                line=item.get("line", 0),
                col=item.get("column", 0),
                tool="pylint",
                rule_id=item.get("message-id", ""),
                sonar_type=sonar_type,
                sonar_severity=severity,
                message=f"{item.get('symbol', '')}: {item.get('message', '')}",
            )
        )

    return ToolRun(name="pylint", passed=not issues, issues=issues, notes=score_note)


def run_pylint_simplify(target: Path) -> ToolRun:
    """
    Run pylint's built-in R17xx refactor-suggestion rules to surface code
    simplification recommendations; everything else is disabled so this
    runner does not duplicate findings from run_pylint().

    Historically this loaded a 'pylint_simplify' plugin for extra 'simplify-*'
    rule IDs on top of the R17xx set, but that package is not published to
    PyPI under any name — there is nothing to install. The R17xx codes below
    are native to pylint and need no plugin, so this now runs unconditionally.
    """
    cmd = [
        "pylint",
        "--disable=all",
        "--enable="
        "R1701,R1702,R1703,R1704,R1705,R1706,R1707,R1708,R1709,R1710,"
        "R1711,R1712,R1713,R1714,R1715,R1716,R1717,R1718,R1719,R1720,R1721,"
        "R1722,R1723,R1724,R1725,R1726,R1727,R1728,R1729,R1730,R1731,R1732,"
        "R1733,R1734,R1735,R1736",
        "--output-format=json",
        str(target),
    ]
    code, stdout, stderr = _run(cmd)

    if code != 0 and not stdout.strip():
        return _tool_error_run(
            "pylint-simplify", target, (stderr.strip() or "no output")[:300]
        )
    try:
        raw_items = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return _tool_error_run(
            "pylint-simplify",
            target,
            "non-JSON stdout: " + (stderr.strip() or stdout.strip())[:300],
        )

    issues = []
    for item in raw_items:
        # Plugin reports refactor opportunities — usually safe-to-apply auto-fixes.
        # High-confidence (collapsible if/else, redundant returns) → MAJOR.
        # Low-confidence stylistic suggestions → MINOR.
        rule_id = item.get("message-id", "")
        high_confidence_ids = {"R1705", "R1710", "R1714", "R1720", "R1723"}
        severity = MAJOR if rule_id in high_confidence_ids else MINOR
        issues.append(
            Issue(
                file=item.get("path", str(target)),
                line=item.get("line", 0),
                col=item.get("column", 0),
                tool="pylint-simplify",
                rule_id=rule_id,
                sonar_type=SMELL,
                sonar_severity=severity,
                message=f"{item.get('symbol', '')}: {item.get('message', '')}",
            )
        )
    return ToolRun(name="pylint-simplify", passed=not issues, issues=issues)


def run_mypy(target: Path) -> ToolRun:
    try:
        exit_code, stdout, stderr = _run(
            [
                "mypy",
                "--ignore-missing-imports",
                "--no-error-summary",
                "--show-error-codes",
                # Suppress error codes that are false positives in generated PySpark notebooks:
                # index/operator  — Dict[str, Any] CONFIG dict subscript access
                # call-overload   — pandas_udf overload stubs are incomplete
                # call-arg        — PySpark max/min/sum shadow builtins inside pandas UDF lambdas
                # arg-type        — same; also importlib module_from_spec None-narrowing pattern
                # union-attr      — module spec/loader may be None (dynamic import pattern)
                "--disable-error-code=index",
                "--disable-error-code=operator",
                "--disable-error-code=call-overload",
                "--disable-error-code=call-arg",
                "--disable-error-code=arg-type",
                "--disable-error-code=union-attr",
                str(target),
            ]
        )
    except FileNotFoundError:
        return _missing_tool_run("mypy", target, "mypy")
    # mypy exits 1 with findings on stdout; but if it crashes or emits nothing, fail the gate as TOOL_ERROR.
    if exit_code != 0 and not stdout.strip():
        return _tool_error_run("mypy", target, (stderr.strip() or "no output")[:300])
    issues = []
    for raw_line in stdout.splitlines():
        match = re.match(
            r"^(.+?):(\d+)(?::\d+)?:\s+(error|warning|note):\s+(.+?)(?:\s+\[(.+)\])?$",
            raw_line,
        )
        if not match:
            continue
        filepath, lineno, level, message, code = match.groups()
        if level == "note":
            continue
        # Only report issues from the target file — mypy may scan imported modules too.
        if Path(filepath).resolve() != target.resolve():
            continue
        # dbutils, spark, etc. are Databricks runtime globals — not undefined in cluster context
        if any(f'Name "{g}" is not defined' in message for g in _DATABRICKS_GLOBALS):
            continue
        issues.append(
            Issue(
                file=filepath,
                line=int(lineno),
                col=0,
                tool="mypy",
                rule_id=code or "mypy",
                sonar_type=BUG if level == "error" else SMELL,
                sonar_severity=MAJOR if level == "error" else MINOR,
                message=message,
            )
        )
    return ToolRun(name="mypy", passed=not issues, issues=issues)


def run_bandit(target: Path) -> ToolRun:
    code, stdout, stderr = _run(["bandit", "-f", "json", "-ll", "-r", str(target)])
    if code != 0 and not stdout.strip():
        return _tool_error_run("bandit", target, (stderr.strip() or "no output")[:300])
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return _tool_error_run(
            "bandit", target, "non-JSON stdout: " + (stderr.strip() or stdout.strip())[:300]
        )

    issues = []
    for result in data.get("results", []):
        sev = result.get("issue_severity", "LOW")
        conf = result.get("issue_confidence", "LOW")

        if sev == "HIGH" and conf in ("HIGH", "MEDIUM"):
            sonar_type, severity = VULNERABILITY, CRITICAL
        elif sev == "HIGH":
            sonar_type, severity = VULNERABILITY, MAJOR
        elif sev == "MEDIUM":
            sonar_type, severity = HOTSPOT, MINOR
        else:
            sonar_type, severity = SMELL, INFO

        issues.append(
            Issue(
                file=result.get("filename", str(target)),
                line=result.get("line_number", 0),
                col=0,
                tool="bandit",
                rule_id=result.get("test_id", ""),
                sonar_type=sonar_type,
                sonar_severity=severity,
                message=(
                    f"{result.get('test_name', '')}: {result.get('issue_text', '')} "
                    f"[sev:{sev} conf:{conf}]"
                ),
            )
        )
    return ToolRun(name="bandit", passed=not issues, issues=issues)


def run_radon(target: Path) -> ToolRun:
    code, stdout, stderr = _run(["radon", "cc", "-j", str(target)])
    if code != 0 and not stdout.strip():
        return _tool_error_run("radon", target, (stderr.strip() or "no output")[:300])
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return _tool_error_run(
            "radon", target, "non-JSON stdout: " + (stderr.strip() or stdout.strip())[:300]
        )

    issues = []
    max_cc = 0
    for filepath, blocks in data.items():
        # radon reports {"error": "..."} instead of a block list when it
        # cannot analyze the file (e.g. syntax error).
        if isinstance(blocks, dict):
            return _tool_error_run("radon", target, str(blocks.get("error", blocks))[:300])
        for block in blocks:
            cc = block.get("complexity", 0)
            max_cc = max(max_cc, cc)
            if cc > _MAX_CYCLOMATIC_COMPLEXITY:
                severity = CRITICAL if cc >= _CC_UNACCEPTABLE_MIN else MAJOR
                issues.append(
                    Issue(
                        file=filepath,
                        line=block.get("lineno", 0),
                        col=0,
                        tool="radon",
                        rule_id=f"cc-{cc}",
                        sonar_type=SMELL,
                        sonar_severity=severity,
                        message=(
                            f"{block.get('type', 'block')} '{block.get('name', '')}' "
                            f"cyclomatic complexity {cc} rank {block.get('rank', '?')} "
                            f"(acceptable: <={_MAX_CYCLOMATIC_COMPLEXITY}, "
                            f"unacceptable: >={_CC_UNACCEPTABLE_MIN})"
                        ),
                    )
                )

    rank_map = [(5, "A"), (10, "B"), (15, "C"), (20, "D"), (25, "E"), (9999, "F")]
    rank = next(r for threshold, r in rank_map if max_cc <= threshold)
    return ToolRun(
        name="radon",
        passed=not issues,
        issues=issues,
        notes=f"Max CC: {max_cc} (rank {rank})",
    )


def run_pydocstyle(target: Path) -> ToolRun:
    try:
        code, stdout, stderr = _run(
            [
                "pydocstyle",
                "--convention=google",
                "--add-ignore=D100,D104,D205,D400,D401",
                str(target),
            ]
        )
    except FileNotFoundError:
        return _missing_tool_run("pydocstyle", target, "pydocstyle")
    # pydocstyle exits 1 with findings on stdout; a non-zero exit with nothing
    # parseable (bad config, parse crash) must fail the gate as TOOL_ERROR.
    if code != 0 and not stdout.strip():
        return _tool_error_run(
            "pydocstyle", target, (stderr.strip() or "no output")[:300]
        )
    issues = []
    lines = stdout.splitlines()
    i = 0
    while i < len(lines) - 1:
        loc_line = lines[i].strip()
        msg_line = lines[i + 1].strip()
        loc_match = re.match(r"^(.+?):(\d+)\s+(?:at|in)\s+.+:?$", loc_line)
        msg_match = re.match(r"^(D\d+):\s*(.+)$", msg_line)
        if loc_match and msg_match:
            issues.append(
                Issue(
                    file=loc_match.group(1),
                    line=int(loc_match.group(2)),
                    col=0,
                    tool="pydocstyle",
                    rule_id=msg_match.group(1),
                    sonar_type=SMELL,
                    sonar_severity=MINOR,
                    message=msg_match.group(2),
                )
            )
            i += 2
        else:
            i += 1
    return ToolRun(name="pydocstyle", passed=not issues, issues=issues)


def _find_companion_test_file(target: Path) -> Optional[Path]:
    """Locate a pytest file covering *target* using common naming conventions."""
    candidates = [
        target.with_name(f"test_{target.stem}.py"),
        target.with_name(f"{target.stem}_test.py"),
        target.with_name("pyspark_pytest.py"),
    ]
    for candidate in candidates:
        if candidate != target and candidate.exists():
            return candidate
    return None


def run_pytest_cov(target: Path) -> ToolRun:
    """
    Run pytest-cov against a companion test file for *target* and gate on the
    Sonar Way Quality Gate coverage-on-new-code threshold (>=80%).

    Looks for test_<stem>.py, <stem>_test.py, or pyspark_pytest.py next to the
    target. When no test file exists, coverage cannot be measured — that is
    reported as its own CRITICAL finding rather than silently passing.
    """
    test_file = _find_companion_test_file(target)
    if test_file is None:
        return ToolRun(
            name="pytest-cov",
            passed=False,
            notes="No companion test file found (test_<name>.py / <name>_test.py)",
            issues=[
                Issue(
                    file=str(target),
                    line=0,
                    col=0,
                    tool="pytest-cov",
                    rule_id="NO_TESTS",
                    sonar_type=SMELL,
                    sonar_severity=CRITICAL,
                    message=(
                        f"No test file found for {target.name} — unit test "
                        f"coverage cannot be measured (gate: {_MIN_COVERAGE_PCT:.0f}%)."
                    ),
                )
            ],
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_report = Path(tmp_dir) / "coverage.json"
        try:
            _, stdout, stderr = _run(
                [
                    "pytest",
                    str(test_file),
                    f"--cov={target.stem}",
                    f"--cov-report=json:{json_report}",
                    "--cov-report=",  # suppress the terminal report; JSON is parsed instead
                    "-q",
                ]
            )
        except FileNotFoundError:
            return _missing_tool_run("pytest-cov", target, "pytest")

        if not json_report.exists():
            # Covers both a crashed run and pytest-cov not being installed
            # (pytest then rejects the unrecognized --cov argument).
            return _tool_error_run(
                "pytest-cov",
                target,
                (stderr.strip() or stdout.strip() or "no coverage report produced")[:300],
            )
        try:
            data = json.loads(json_report.read_text())
        except json.JSONDecodeError:
            return _tool_error_run("pytest-cov", target, "non-JSON coverage report")

    percent = data.get("totals", {}).get("percent_covered", 0.0)

    if percent >= _MIN_COVERAGE_PCT:
        return ToolRun(name="pytest-cov", passed=True, notes=f"Coverage: {percent:.1f}%")

    severity = CRITICAL if percent < _CRITICAL_COVERAGE_PCT else MAJOR
    return ToolRun(
        name="pytest-cov",
        passed=False,
        notes=f"Coverage: {percent:.1f}% (below {_MIN_COVERAGE_PCT:.0f}% gate)",
        issues=[
            Issue(
                file=str(target),
                line=0,
                col=0,
                tool="pytest-cov",
                rule_id=f"coverage-{percent:.0f}",
                sonar_type=SMELL,
                sonar_severity=severity,
                message=(
                    f"Test coverage {percent:.1f}% is below the "
                    f"{_MIN_COVERAGE_PCT:.0f}% Sonar Way Quality Gate threshold."
                ),
            )
        ],
    )


# ── Report renderer ───────────────────────────────────────────────────────────


def _gate_status(count: int, maximum: int) -> str:
    return "PASS" if count <= maximum else "FAIL"


def print_report(report: ValidationReport) -> None:
    gate_passed = report.quality_gate_passed()
    gate_label = "PASSED" if gate_passed else "FAILED"

    bugs = report.bugs()
    vulns = report.vulnerabilities()
    hotspots = report.hotspots()
    smells = report.smells()

    print()
    print("=" * 72)
    print("  PRE-SONARQUBE SEMS VALIDATION REPORT")
    print(f"  Target : {report.target}")
    print("=" * 72)
    print()
    print(f"  QUALITY GATE: {'✓' if gate_passed else '✗'} {gate_label}")
    print()

    hdr = f"  {'Category':<35} {'Issues':>6}  {'Rating':<7} {'Gate':<6}  Threshold"
    print(hdr)
    print("  " + "-" * 65)
    print(
        f"  {'Reliability  (Bugs)':<35} {len(bugs):>6}  "
        f"{report.reliability_rating():<7} {_gate_status(len(bugs), _GATE_MAX_BUGS):<6}  "
        f"max {_GATE_MAX_BUGS}"
    )
    print(
        f"  {'Security     (Vulnerabilities)':<35} {len(vulns):>6}  "
        f"{report.security_rating():<7} {_gate_status(len(vulns), _GATE_MAX_VULNS):<6}  "
        f"max {_GATE_MAX_VULNS}"
    )
    print(
        f"  {'Security     (Hotspots)':<35} {len(hotspots):>6}  "
        f"{'N/A':<7} {_gate_status(len(hotspots), _GATE_MAX_HOTSPOTS):<6}  "
        f"max {_GATE_MAX_HOTSPOTS}"
    )
    print(
        f"  {'Maintainability (Code Smells)':<35} {len(smells):>6}  "
        f"{report.maintainability_rating():<7} {'WARN':<6}  "
        f"no CRITICAL/BLOCKER"
    )

    sections = [
        (BUG, bugs, "BUGS"),
        (VULNERABILITY, vulns, "VULNERABILITIES"),
        (HOTSPOT, hotspots, "SECURITY HOTSPOTS"),
        (SMELL, smells, "CODE SMELLS"),
    ]
    for _, issues, label in sections:
        if not issues:
            continue
        print()
        print(f"  ── {label} ({len(issues)}) " + "─" * max(1, 64 - len(label) - 7))
        sorted_issues = sorted(
            issues,
            key=lambda x: (_SEVERITY_RANK.get(x.sonar_severity, 0), x.line),
            reverse=True,
        )
        for issue in sorted_issues:
            print(issue.format_line())

    print()
    print("  ── TOOL SUMMARY " + "─" * 55)
    print(f"  {'Tool':<14}  {'Status':<8}  {'Issues':>6}  Notes")
    print("  " + "-" * 55)
    for run in report.tool_runs:
        status = "PASS" if run.passed else "ISSUES"
        count = str(len(run.issues)) if run.issues else "-"
        print(f"  {run.name:<14}  {status:<8}  {count:>6}  {run.notes}")
    print()


# ── Validation entry point ────────────────────────────────────────────────────


def _run_tool(entry: Tuple[str, str, Callable[[], ToolRun]], target: Path) -> ToolRun:
    tool_name, executable, runner = entry
    try:
        return runner()
    except FileNotFoundError:
        logger.warning("Mandatory tool not installed: %s", executable)
        return _missing_tool_run(tool_name, target, executable)
    except Exception as exc:  # noqa: BLE001 — any analyzer crash must still yield a report, not a tool error
        logger.warning("Tool %s crashed: %s", tool_name, exc, exc_info=True)
        return _tool_error_run(tool_name, target, str(exc)[:300])


def validate_file(target: Path, fix: bool = False) -> ValidationReport:
    report = ValidationReport(target=str(target))
    # black/isort mutate the file in place when fix=True; every other tool
    # reads the file, so formatting must finish first and in order.
    mutating_runners = [
        ("black", "black", lambda: run_black(target, fix)),
        ("isort", "isort", lambda: run_isort(target, fix)),
    ]
    read_only_runners = [
        ("flake8", "flake8", lambda: run_flake8(target)),
        ("pylint", "pylint", lambda: run_pylint(target)),
        ("pylint-simplify", "pylint", lambda: run_pylint_simplify(target)),
        ("mypy", "mypy", lambda: run_mypy(target)),
        ("bandit", "bandit", lambda: run_bandit(target)),
        ("radon", "radon", lambda: run_radon(target)),
        ("pydocstyle", "pydocstyle", lambda: run_pydocstyle(target)),
        ("pytest-cov", "pytest", lambda: run_pytest_cov(target)),
    ]

    if fix:
        for entry in mutating_runners:
            report.tool_runs.append(_run_tool(entry, target))
    else:
        # fix=False: black/isort only check, so they're read-only too.
        read_only_runners = mutating_runners + read_only_runners

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(read_only_runners)
    ) as pool:
        # map() preserves input order regardless of completion order, so
        # tool_runs stays deterministically ordered like the sequential loop.
        report.tool_runs.extend(
            pool.map(lambda entry: _run_tool(entry, target), read_only_runners)
        )
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-SonarQube SEMS validator — mirrors Sonar Way Quality Gate for Python/PySpark"
    )
    parser.add_argument("target", help="Python file or directory to validate")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix formatting with black + isort before linting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Emit machine-readable JSON report (for CI pipelines)",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 2

    py_files = list(target.rglob("*.py")) if target.is_dir() else [target]
    if not py_files:
        print(f"No Python files found in {target}", file=sys.stderr)
        return 2

    all_passed = True
    for py_file in py_files:
        report = validate_file(py_file, fix=args.fix)
        if args.output_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_report(report)
        if not report.quality_gate_passed():
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
