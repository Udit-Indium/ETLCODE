"""
SEMS Gap Analyzer Agent.

Takes a PySpark script (file path or source string), runs the full 3-layer
SEMS compliance check, and uses an LLM agent to produce specific before/after
code changes for every gap so the script can achieve SEMS compliance.

Layers run:
  1. Syntax check (ast.parse)
  2. AST + regex SEMS rules (compliance_checker)
  3. 10 external static-analysis tools (pre_sonar_check)

Before layer 3 runs, black and isort are applied in --fix mode directly to
the target file (see _auto_format): those two findings are purely mechanical,
so they are auto-corrected on disk here instead of being routed through the
LLM fix loop below. Every other tool remains check-only.

The LLM agent adds code-level context that pure rule descriptions lack:
  - Extracts the exact problematic code lines from the source
  - Generates a SEMS-compliant replacement snippet
  - Explains the Shell STAM rationale per gap

Usage:
    # Programmatic
    from subagents.sems_agent import run_sems_gap_analyzer
    result = run_sems_gap_analyzer("outputs/my_pipeline.py")
    print(result.gap_report_markdown)

    # CLI
    python -m subagents.sems_agent.sems_gap_analyzer_agent outputs/my_pipeline.py
    python -m subagents.sems_agent.sems_gap_analyzer_agent outputs/my_pipeline.py --json
    python -m subagents.sems_agent.sems_gap_analyzer_agent outputs/my_pipeline.py --out report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from google.adk.models.lite_llm import LiteLlm
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext

from .model_utils import ForceToolLiteLlm

from .compliance.pre_sonar_check import HOTSPOT, run_black, run_isort
from .compliance.sems_validator import (
    BRD_AREAS,
    _AREA_LABEL,
    _brd_area_for_issue,
    _brd_area_for_violation,
    validate_source,
)
from .compliance.utils import source_window as _source_window
from ...utils import canonical_stem, resolve_converted_path

load_dotenv()

logger = logging.getLogger(__name__)

OUTPUTS_DIR = pathlib.Path(__file__).parents[2] / "outputs"

# How many source lines to show around a violation when building the prompt.
_CONTEXT_WINDOW = 5

# Truncate source in prompt to avoid exceeding model context.
_MAX_SOURCE_CHARS = 12_000

# Severities that count as "blocking" — the LLM writes full prose only for
# these; everything else is compiled into a deterministic table (see
# _render_gap_table / write_gap_report) so large gap sets can never be
# silently truncated by the model.
_BLOCKING_SEVERITIES = frozenset({"hard", "BLOCKER", "CRITICAL", "MAJOR"})


def _gap_is_blocking(severity: str, sonar_type: Optional[str] = None) -> bool:
    """Mirror sems_validator's per-BRD-area blocking rule (see
    AreaSummary.passed / _aggregate_areas in compliance/sems_validator.py):
    MAJOR+ severity, OR a SECURITY_HOTSPOT tool finding regardless of its
    severity label (bandit reports MEDIUM-severity hotspots as sonar
    severity MINOR, which would otherwise be missed here)."""
    return severity in _BLOCKING_SEVERITIES or sonar_type == HOTSPOT


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class GapItem:
    rule_id: str
    severity: str
    brd_area: str
    location: str
    description: str
    remediation: str
    auto_fixable: bool
    is_blocking: bool


@dataclass
class GapAnalysisResult:
    target: str
    total_gaps: int
    blocking_gaps: int
    gaps: List[GapItem] = field(default_factory=list)
    area_summaries: Dict[str, Any] = field(default_factory=dict)
    gap_report_markdown: str = ""
    syntax_error: Optional[str] = None


# ── Prompt builders ───────────────────────────────────────────────────────────

# Rule IDs that flag the analysis environment itself (missing/crashing tool),
# not a location in the generated code — no code_context should be attached.
_ENVIRONMENT_RULE_IDS = frozenset({"TOOL_MISSING", "TOOL_ERROR"})


def _line_no_from_location(location: str) -> Optional[int]:
    """Parse the ``line N`` location string used throughout GapItem/gap dicts."""
    if location.startswith("line "):
        try:
            return int(location.split()[1])
        except (IndexError, ValueError):
            return None
    return None


def _code_context_for_gap(source_lines: List[str], rule_id: str, location: str) -> Optional[str]:
    """Return the real source window for a gap, or None when there is no
    corresponding code location (e.g. a missing/crashing external tool)."""
    if rule_id in _ENVIRONMENT_RULE_IDS or not source_lines:
        return None
    line_no = _line_no_from_location(location)
    # Module-level gaps (e.g. missing docstring) have no line number — fall
    # back to the file header, which is where those rules actually apply.
    return _source_window(source_lines, line_no or 1, _CONTEXT_WINDOW)


def _build_violations_block(source_lines: List[str], gaps: List[GapItem]) -> str:
    """Format each gap with its code context for the LLM prompt."""
    if not gaps:
        return "No violations found."
    parts = []
    for idx, gap in enumerate(gaps, 1):
        parts.append(f"### Gap {idx}: [{gap.severity.upper()}] {gap.rule_id} — {gap.brd_area}")
        parts.append(f"**Location**: {gap.location}")
        parts.append(f"**Description**: {gap.description}")
        parts.append(f"**Rule remediation hint**: {gap.remediation.strip()}")

        code_context = _code_context_for_gap(source_lines, gap.rule_id, gap.location)
        if code_context:
            parts.append("**Code context**:")
            parts.append("```python")
            parts.append(code_context)
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _build_agent_prompt(source: str, gaps: List[GapItem], target_name: str) -> str:
    """Construct the full LLM prompt for gap-by-gap analysis."""
    source_lines = source.splitlines()

    # Truncate very large files — show head + tail
    display_source = source
    if len(source) > _MAX_SOURCE_CHARS:
        half = _MAX_SOURCE_CHARS // 2
        display_source = (
            source[:half]
            + f"\n\n... [truncated {len(source) - _MAX_SOURCE_CHARS} chars] ...\n\n"
            + source[-half:]
        )

    violations_block = _build_violations_block(source_lines, gaps)

    return f"""You are a SEMS (Shell Engineering Management System) compliance expert for Shell's Databricks STAM (Standard Analytical Model) layer.

You have been given a PySpark script named `{target_name}` and a list of SEMS compliance gaps detected by the static analysis engine.

Your task: for each gap, produce a **concrete code fix** showing exactly what must change.

---

## PySpark Source (`{target_name}`)

```python
{display_source}
```

---

## Detected SEMS Compliance Gaps ({len(gaps)} total)

{violations_block}

---

## Instructions

For each gap listed above, write a section with this exact structure:

```
### Gap N: [SEVERITY] RULE_ID — BRD Area
**What is wrong**: One sentence describing the specific problem in THIS script.
**Before (non-compliant)**:
```python
<exact problematic code from the script>
```
**After (SEMS-compliant fix)**:
```python
<corrected code that satisfies the SEMS rule>
```
**Why this matters**: One or two sentences on the SEMS/STAM rationale.
```

After all gaps, add:

```
## Summary

| # | Rule | Severity | BRD Area | Auto-fixable | Effort |
|---|------|----------|----------|--------------|--------|
| 1 | ...  | ...      | ...      | Yes/No       | Low/Medium/High |
...

**Total gaps**: N
**Blocking (hard)**: N
**Recommended fix order**: list rule IDs from highest to lowest priority.
```

Be precise — show real code from the script, not generic placeholders.
If a gap is module-level (no specific line), reference the relevant section of the script.
If a gap is marked auto_fixable=True, note that it is a mechanical fix that could be automated separately.
"""


# ── Agent runner ─────────────────────────────────────────────────────────────

def _run_llm_analysis(prompt: str) -> str:
    """Run a single-turn LLM completion to generate the gap report.

    A direct litellm call: this path has no tools and a single turn, so it
    needs no agent framework (the pipeline path uses the Google ADK agent
    below instead).
    """
    import litellm

    response = litellm.completion(
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a SEMS compliance expert for Shell's Databricks STAM layer. "
                    "Produce precise, actionable gap analysis reports with before/after code fixes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


# ── Gap extraction helpers ────────────────────────────────────────────────────

_AUTO_FIXABLE_RULES = frozenset({
    "SPARK003",   # pyspark typo
    "DOC001",     # module docstring
    "DOC002",     # missing Args: entries — the missing param names are in the message
    "DOC003",     # missing Returns:/Yields: section
    "SEMS_ERR001",  # spark action try/except
    "SPARK001",   # unbounded collect
    "ERR002",     # exception reraise
    "LOG001",     # print → logging
    "F821",       # flake8: undefined name
    "E0602",      # pylint: undefined-variable
    "name-defined",  # mypy: name not defined
    "F401",       # flake8: imported but unused
    "W0611",      # pylint: unused-import
})

#: Rule IDs whose remediation requires creating a NEW file — e.g. NO_TESTS
#: (pytest-cov, see run_pytest_cov in compliance/pre_sonar_check.py) fires
#: when no companion tests/test_<name>.py exists, and the only way to
#: "resolve" it is to create that file. No fixer tool in this pipeline can do
#: that: replace_functions_tool/_apply_function_patch and the SEMS parallel
#: fixers' propose_function_patch_tool all merge into the ONE canonical
#: converted file (_canonical_output_path) — there is no "create a sibling
#: file" tool anywhere. Handing a gap like this to a fixer doesn't fail
#: safely: the model is instructed to resolve everything in its bucket, so it
#: eventually improvises — observed in practice as an agent pasting an
#: entire unittest suite plus a self-import into the PRODUCTION file, which
#: cascaded into hundreds of new gaps and crashed the final Databricks run
#: (ModuleNotFoundError from the self-import). Excluded from fixer
#: eligibility (_make_seed_sems_fix_bucket in code_convertor_agent.py) and
#: from the escalation gate (_actionable_blocking_count below) — still
#: reported in full via the normal gap-report prose, just never handed to a
#: model that can't actually fix it.
_MANUAL_ACTION_ONLY_RULES = frozenset({"NO_TESTS"})


def _actionable_blocking_count(gap_state: Dict[str, Any]) -> int:
    """Blocking-gap count used for loop escalation (check_sems_compliance /
    check_sems_compliance_final) — excludes _MANUAL_ACTION_ONLY_RULES gaps,
    which no fixer can ever resolve, so the loop doesn't burn every
    iteration waiting on something nothing will act on. The raw
    gap_state["blocking_gaps"] count is left untouched for the report
    header — this only affects the pass/fail gate.
    """
    return sum(
        1
        for g in gap_state.get("gaps", [])
        if g.get("is_blocking") and g.get("rule_id") not in _MANUAL_ACTION_ONLY_RULES
    )


def _collect_gaps(sems_result, tool_report) -> List[GapItem]:
    """Flatten all SEMS violations and tool findings into GapItem list."""
    gaps: List[GapItem] = []

    if sems_result:
        for sv in sems_result.structured_violations:
            gaps.append(GapItem(
                rule_id=sv.rule_id,
                severity=sv.severity,
                brd_area=_brd_area_for_violation(sv),
                location=f"line {sv.line_number}" if sv.line_number else "module",
                description=sv.message,
                remediation=sv.remediation or "",
                auto_fixable=sv.rule_id in _AUTO_FIXABLE_RULES,
                is_blocking=_gap_is_blocking(sv.severity),
            ))

    if tool_report:
        for issue in tool_report.all_issues:
            gaps.append(GapItem(
                rule_id=issue.rule_id or issue.tool,
                severity=issue.sonar_severity or "INFO",
                brd_area=_brd_area_for_issue(issue),
                location=f"line {issue.line}" if issue.line else "module",
                description=issue.message or "",
                remediation=f"Fix flagged by {issue.tool}: {issue.message or ''}",
                auto_fixable=issue.tool in {"black", "isort"},
                is_blocking=_gap_is_blocking(issue.sonar_severity, issue.sonar_type),
            ))

    return gaps


# ── Public entry point ───────────────────────────────────────────────────────

def run_sems_gap_analyzer(
    script_path: Optional[str | Path] = None,
    source_code: Optional[str] = None,
    *,
    include_llm_analysis: bool = True,
) -> GapAnalysisResult:
    """
    Analyze a PySpark script for SEMS compliance gaps and return a detailed
    `GapAnalysisResult` including an LLM-generated before/after remediation report.

    Provide either `script_path` (path to a .py file) or `source_code` (raw string).
    When `include_llm_analysis=False`, only the static analysis report is returned
    (no LLM call) — useful for testing or when no API key is available.

    Note: this reformats the target file in place with black/isort (--fix)
    before analyzing it — see `_auto_format`. For `script_path`, that means
    the real file on disk gets reformatted as a side effect of calling this
    function. For `source_code`, only the discarded temp file is touched.
    """
    if script_path is None and source_code is None:
        raise ValueError("Provide either script_path or source_code.")

    if script_path is not None:
        path = Path(script_path).resolve()
        source = path.read_text(encoding="utf-8")
        return _analyze(path, source, path.name, include_llm_analysis)

    # Write source to a temp file so the validator can run external tools —
    # and always remove it afterwards, on every return and exception path, so
    # repeated inline-source calls don't accumulate temp files.
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    tmp.write(source_code)
    tmp.close()
    path = Path(tmp.name)
    try:
        return _analyze(path, source_code, "<inline_script>", include_llm_analysis)
    finally:
        path.unlink(missing_ok=True)


def _auto_format(path: Path, source: str) -> str:
    """Deterministically fix black/isort findings in place before analysis.

    These two tools are 100% mechanical — there is no ambiguity to hand off
    to the LLM-based sems_fix_agent, which would otherwise have to
    hand-reproduce formatter/import-order rules unreliably (and slower, via
    extra read/replace tool calls). Running them here means black/isort
    should never surface as a gap in a normal run; the file is reformatted
    on disk and the returned source reflects that (falls back to the
    original source if the file couldn't be read back, e.g. deleted mid-run).
    A genuine syntax error simply makes black/isort no-ops (non-zero exit,
    file left untouched) — validate_source's own Layer 1 check still catches
    it afterwards.
    """
    run_black(path, fix=True)
    run_isort(path, fix=True)
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return source


def _analyze(
    path: Path,
    source: str,
    target_name: str,
    include_llm_analysis: bool,
) -> GapAnalysisResult:
    """Run the 3-layer gap analysis on an on-disk file (see run_sems_gap_analyzer)."""
    logger.info("SEMS Gap Analyzer: validating %s", target_name)

    source = _auto_format(path, source)
    sems_report = validate_source(source, path)

    if sems_report.syntax_error:
        return GapAnalysisResult(
            target=target_name,
            total_gaps=1,
            blocking_gaps=1,
            gaps=[GapItem(
                rule_id="SYNTAX",
                severity="hard",
                brd_area="code_readability_and_structure",
                location="module",
                description=f"Syntax error: {sems_report.syntax_error}",
                remediation="Fix the Python syntax error before SEMS analysis can proceed.",
                auto_fixable=False,
                is_blocking=True,
            )],
            syntax_error=sems_report.syntax_error,
            gap_report_markdown=(
                f"# SEMS Gap Analysis — {target_name}\n\n"
                f"## Syntax Error\n\n"
                f"```\n{sems_report.syntax_error}\n```\n\n"
                f"_SEMS analysis cannot proceed until the syntax error is resolved._"
            ),
        )

    gaps = _collect_gaps(sems_report.sems_result, sems_report.tool_report)
    blocking = sum(1 for g in gaps if g.is_blocking)

    area_summaries = {
        area: {
            "label": _AREA_LABEL[area],
            "passed": sems_report.area_summaries[area].passed,
            "total": sems_report.area_summaries[area].total,
            "blocking": (
                sems_report.area_summaries[area].sems_blocking
                + sems_report.area_summaries[area].tool_blocking
            ),
        }
        for area in BRD_AREAS
    }

    if not gaps:
        report_md = (
            f"# SEMS Gap Analysis — {target_name}\n\n"
            f"No SEMS compliance gaps detected."
        )
        return GapAnalysisResult(
            target=target_name,
            total_gaps=0,
            blocking_gaps=0,
            gaps=[],
            area_summaries=area_summaries,
            gap_report_markdown=report_md,
        )

    if include_llm_analysis:
        logger.info(
            "SEMS Gap Analyzer: %d gap(s) found — requesting LLM remediation analysis",
            len(gaps),
        )
        prompt = _build_agent_prompt(source, gaps, target_name)
        try:
            llm_report = _run_llm_analysis(prompt)
        except Exception as exc:
            logger.warning("LLM gap analysis failed: %s — falling back to static report", exc)
            llm_report = _static_gap_report(target_name, gaps)
    else:
        llm_report = _static_gap_report(target_name, gaps)

    # Prefix the report with the gap-count header if not already present
    if not llm_report.startswith("# SEMS Gap Analysis"):
        header = (
            f"# SEMS Gap Analysis — {target_name}\n\n"
            f"**Total Gaps**: {len(gaps)}  |  **Blocking**: {blocking}\n\n"
            "---\n\n"
        )
        llm_report = header + llm_report

    return GapAnalysisResult(
        target=target_name,
        total_gaps=len(gaps),
        blocking_gaps=blocking,
        gaps=gaps,
        area_summaries=area_summaries,
        gap_report_markdown=llm_report,
    )


def _static_gap_report(
    target_name: str,
    gaps: List[GapItem],
) -> str:
    """Fallback Markdown report when the LLM call is skipped or fails."""
    lines = [
        f"# SEMS Gap Analysis — {target_name}",
        "",
        f"**Total gaps**: {len(gaps)}",
        "",
        "## Gaps",
        "",
    ]
    for idx, gap in enumerate(gaps, 1):
        lines.append(f"### Gap {idx}: [{gap.severity.upper()}] {gap.rule_id}")
        lines.append(f"- **BRD area**: {_AREA_LABEL.get(gap.brd_area, gap.brd_area)}")
        lines.append(f"- **Location**: {gap.location}")
        lines.append(f"- **Description**: {gap.description}")
        lines.append(f"- **Remediation**: {gap.remediation.strip()}")
        lines.append(f"- **Auto-fixable**: {'Yes (use --fix)' if gap.auto_fixable else 'No — manual change required'}")
        lines.append("")
    return "\n".join(lines)


# ── Deterministic report assembly (ADK path) ─────────────────────────────────
#
# The LLM only authors free-form prose for blocking-severity gaps (usually a
# handful). Everything else — the non-blocking bulk, the summary table, the
# fix-order/auto-fixable/manual lists — is built here from the same gap data
# the tool already returned, so a report can never silently drop gaps just
# because the model chose to sample or summarize instead of enumerating.
#
# The LLM can also under-deliver on the blocking set itself (observed in
# practice: it wrote prose for 3 of 21 blocking gaps and just stopped). See
# _find_missed_blocking_gaps — any blocking gap not actually covered in the
# prose gets backstopped as an auto-listed table row, the same way the
# non-blocking bulk already is, so a CRITICAL/MAJOR finding can never
# silently vanish just because the model truncated.

def _md_escape_cell(text: str, *, max_len: int = 140) -> str:
    """Flatten a string for safe embedding in a Markdown table cell."""
    flat = " ".join((text or "").split())
    flat = flat.replace("|", "\\|")
    if len(flat) > max_len:
        flat = flat[: max_len - 1].rstrip() + "…"
    return flat


def _gap_effort(gap: Dict[str, Any]) -> str:
    if gap.get("rule_id") in _ENVIRONMENT_RULE_IDS:
        return "Installation"
    return "Auto" if gap.get("auto_fixable") else "Manual"


_SEVERITY_RANK = {
    "hard": 3, "BLOCKER": 3, "CRITICAL": 3, "MAJOR": 2,
    "soft": 1, "MINOR": 1, "INFO": 0,
}


def _order_gaps_blocking_first(gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort gaps worst-first: blocking-severity items (MAJOR+, or any
    SECURITY_HOTSPOT finding regardless of its severity label — see
    _gap_is_blocking) come before non-blocking ones, highest severity first
    within each group. Used both for the Markdown summary table and for the
    persisted GAP_STATE_FILE, so a consumer of either the .md or the .json
    sees the (usually few) blocking gaps immediately rather than having to
    scan past hundreds of soft/minor entries to find them."""
    return sorted(
        gaps,
        key=lambda g: (bool(g.get("is_blocking")), _SEVERITY_RANK.get(g.get("severity", ""), 0)),
        reverse=True,
    )


def _render_gap_table(gaps: List[Dict[str, Any]], *, heading: str) -> str:
    """One compact row per gap — used for the non-blocking bulk so every gap
    is guaranteed to appear no matter how many there are."""
    if not gaps:
        return ""
    lines = [
        heading,
        "",
        "| # | Rule | Severity | BRD Area | Location | Description | Auto-fixable |",
        "|---|------|----------|----------|----------|--------------|---------------|",
    ]
    for idx, gap in enumerate(gaps, 1):
        lines.append(
            f"| {idx} | {gap.get('rule_id', '')} | {gap.get('severity', '')} | "
            f"{_AREA_LABEL.get(gap.get('brd_area', ''), gap.get('brd_area', ''))} | "
            f"{_md_escape_cell(gap.get('location', ''), max_len=40)} | "
            f"{_md_escape_cell(gap.get('description', ''))} | "
            f"{'Yes' if gap.get('auto_fixable') else 'No'} |"
        )
    return "\n".join(lines)


def _render_summary_table(gaps: List[Dict[str, Any]]) -> str:
    """Full summary table + fix-order/auto-fixable/manual lists — mechanical
    formatting of data already returned by run_gap_analysis_tool."""
    if not gaps:
        return ""
    ordered = _order_gaps_blocking_first(gaps)

    lines = [
        "### Summary",
        "",
        "| # | Rule | Severity | BRD Area | Auto-fixable | Effort |",
        "|---|------|----------|----------|--------------|--------|",
    ]
    for idx, gap in enumerate(ordered, 1):
        lines.append(
            f"| {idx} | {gap.get('rule_id', '')} | {gap.get('severity', '')} | "
            f"{_AREA_LABEL.get(gap.get('brd_area', ''), gap.get('brd_area', ''))} | "
            f"{'Yes' if gap.get('auto_fixable') else 'No'} | {_gap_effort(gap)} |"
        )

    seen: set = set()
    fix_order = []
    for gap in ordered:
        rid = gap.get("rule_id", "")
        if rid not in seen:
            seen.add(rid)
            fix_order.append(rid)

    auto_fixable_rules = sorted({g["rule_id"] for g in gaps if g.get("auto_fixable")})
    manual_rules = sorted({g["rule_id"] for g in gaps if not g.get("auto_fixable")})

    lines.append("")
    lines.append(f"**Total gaps**: {len(gaps)}")
    lines.append(f"**Recommended fix order**: {', '.join(fix_order) if fix_order else 'None'}")
    lines.append(f"**Auto-fixable gaps**: {', '.join(auto_fixable_rules) if auto_fixable_rules else 'None'}")
    lines.append(f"**Manual fixes required**: {', '.join(manual_rules) if manual_rules else 'None'}")
    return "\n".join(lines)


_FLAGGED_LINE_RE = re.compile(r"^>>>\s*\d+\s*\|\s?(.*)$", re.MULTILINE)


def _flagged_line_text(code_context: Optional[str]) -> Optional[str]:
    """Extract the exact flagged source line (gutter stripped) from a gap's
    code_context, e.g. ">>>   11 | from pyspark.sql.types import ..." ->
    "from pyspark.sql.types import ...". None when there's no code_context
    (TOOL_MISSING/TOOL_ERROR gaps) or nothing to extract."""
    if not code_context:
        return None
    match = _FLAGGED_LINE_RE.search(code_context)
    if not match:
        return None
    return match.group(1).strip() or None


def _find_missed_blocking_gaps(
    llm_content: str, blocking_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Blocking-severity gaps whose prose section is missing from ``llm_content``.

    The prompt requires every Before/After block to quote the gap's
    code_context flagged line verbatim, so a gap counts as covered when that
    exact source line appears in the model's output. This also correctly
    treats two raw findings that flag the identical source line (e.g. flake8
    and pylint both flagging the same unused import) as covered by one prose
    section, rather than demanding a redundant duplicate. Gaps with no
    code_context (TOOL_MISSING/TOOL_ERROR) fall back to a rule_id-in-content
    check. Anything not covered is returned so the caller can render it as a
    fallback table row instead of silently dropping it.
    """
    missed = []
    for gap in blocking_items:
        flagged_line = _flagged_line_text(gap.get("code_context"))
        if flagged_line:
            covered = flagged_line in llm_content
        else:
            rule_id = gap.get("rule_id", "")
            covered = bool(rule_id) and rule_id in llm_content
        if not covered:
            missed.append(gap)
    return missed


_ENV_SHOW_NON_BLOCKING = "SEMS_GAP_REPORT_SHOW_NON_BLOCKING"


def _include_non_blocking_gaps(context: ToolContext) -> bool:
    """Whether to append the full itemized non-blocking gap table + summary.

    ON by default: the report shows full detail for blocking/MAJOR+ gaps plus
    an itemized table of non-blocking findings, so nothing is silently
    dropped — non-blocking gaps are never auto-fixed (see sems_fix_agent,
    which only ever reads blocking gaps), so this table is the only place a
    human ever sees them. Turn off deliberately, per run, by either:
      - setting session state ``gap_report_include_non_blocking`` to False
        (e.g. from the caller that seeds ``converted_pyspark_file_path``), or
      - setting the environment variable SEMS_GAP_REPORT_SHOW_NON_BLOCKING=0.
    """
    state_flag = context.state.get("gap_report_include_non_blocking")
    if state_flag is not None:
        return bool(state_flag)
    env_flag = os.getenv(_ENV_SHOW_NON_BLOCKING, "").strip().lower()
    if env_flag in {"1", "true", "yes"}:
        return True
    if env_flag in {"0", "false", "no"}:
        return False
    return True


def _compose_gap_report(context: ToolContext, llm_content: str) -> str:
    """Assemble the full report: header + LLM-authored blocking-gap prose,
    plus — only when _include_non_blocking_gaps(context) is True — an
    auto-generated non-blocking table + summary. Guarantees every gap
    run_gap_analysis_tool returned is at least counted, and (when the flag is
    on) fully itemized, regardless of what the LLM chose to write out."""
    gap_state = read_gap_state()
    gap_items: List[Dict[str, Any]] = gap_state.get("gaps", [])
    total = gap_state.get("total_gaps", len(gap_items))
    blocking_n = gap_state.get("blocking_gaps", 0)
    target = gap_state.get("target", "converted file")

    header = (
        f"# SEMS Gap Analysis — {target}\n"
        f"**Total Gaps**: {total}  |  **Blocking**: {blocking_n}\n"
    )

    if not gap_items:
        return header + "\nNo SEMS compliance gaps detected."

    parts = [header, "---", llm_content.strip()]

    blocking_items = [g for g in gap_items if g.get("is_blocking")]
    missed_blocking = _find_missed_blocking_gaps(llm_content, blocking_items)
    if missed_blocking:
        parts.append(_render_gap_table(
            missed_blocking,
            heading=(
                f"### Additional blocking gaps not detailed above "
                f"({len(missed_blocking)}) — auto-listed"
            ),
        ))

    if _include_non_blocking_gaps(context):
        non_blocking = [g for g in gap_items if not g.get("is_blocking")]
        if non_blocking:
            parts.append(_render_gap_table(
                non_blocking,
                heading=f"### Non-blocking gaps ({len(non_blocking)}) — auto-listed",
            ))
        parts.append(_render_summary_table(gap_items))
    else:
        non_blocking_n = total - blocking_n
        if non_blocking_n > 0:
            parts.append(
                f"_{non_blocking_n} non-blocking gap(s) not itemized in this report. "
                f"Set session state `gap_report_include_non_blocking=True` (or env "
                f"`{_ENV_SHOW_NON_BLOCKING}=1`) and re-run to list them._"
            )

    return "\n\n".join(p for p in parts if p)


# ── ADK Agent (Maverick / Databricks) ────────────────────────────────────────

def _load_skill_guidance() -> str:
    """Read sems-check-skill SKILL.md as plain text."""
    skill_dir = pathlib.Path(__file__).parent / "skills" / "sems-check-skill"
    parts = []
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        parts.append(skill_md.read_text())
    refs = skill_dir / "references"
    if refs.exists():
        for ref in sorted(refs.glob("*.md")):
            parts.append(f"\n\n===== references/{ref.name} =====\n{ref.read_text()}")
    return "".join(parts)


_SKILL_GUIDANCE = _load_skill_guidance()

# Real ADK Skill registration (mirrors code_convertor_agent.py's py2snow-skill
# wiring) — gives the model on-demand list_skills/load_skill/load_skill_resource
# tools over sems-check-skill. Kept ALONGSIDE the _SKILL_GUIDANCE text injection
# below (not instead of it): these agents run single_turn with
# include_contents="none", so a fresh context every turn can't reliably reach
# for an on-demand tool before the forced run_gap_analysis_tool call — the
# inlined text is what actually guarantees the guidance is seen.
sems_gap_skill = load_skill_from_dir(pathlib.Path(__file__).parent / "skills" / "sems-check-skill")
sems_gap_skill_toolset = SkillToolset(skills=[sems_gap_skill])

#: Full gap list (every severity), persisted to disk instead of ADK session
#: state. A gap set can be dozens of items, each carrying a source
#: code_context snippet — keeping that in session state would echo the whole
#: thing into every prompt of every agent sharing this session, the same
#: reasoning code_convertor_agent.py's AST-inventory comments give for
#: keeping the parsed inventory out of state.
GAP_STATE_FILE = OUTPUTS_DIR / "sems_gap_state.json"

#: Which blocking gaps sems_fix_agent has already been handed this analysis
#: pass, so _seed_sems_fix_bucket (code_convertor_agent.py) can select the
#: next bucket (see SEMS_GAP_BATCH_SIZE there) instead of one giant response.
#: Reset every time run_gap_analysis_tool writes a fresh gap list.
GAP_BATCH_CURSOR_FILE = OUTPUTS_DIR / "sems_gap_batch_cursor.json"


def _write_gap_state(target: str, total: int, blocking: int, gaps: List[Dict[str, Any]]) -> None:
    """Persist the gap analysis to disk and reset the fix-loop batch cursor.

    write_gap_report / write_final_gap_report / _seed_sems_fix_bucket all
    read this back with read_gap_state() rather than through session state.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GAP_STATE_FILE.write_text(
        json.dumps(
            {"target": target, "total_gaps": total, "blocking_gaps": blocking, "gaps": gaps},
            indent=2,
        ),
        encoding="utf-8",
    )
    # A fresh analysis pass invalidates whatever bucket cursor was in
    # progress against the previous gap list.
    GAP_BATCH_CURSOR_FILE.write_text(json.dumps({"served": []}), encoding="utf-8")


def read_gap_state() -> Dict[str, Any]:
    """Read the gap analysis run_gap_analysis_tool last persisted to disk.

    Returns an empty shell when analysis has not run yet.
    """
    try:
        data = json.loads(GAP_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        return {"target": "converted file", "total_gaps": 0, "blocking_gaps": 0, "gaps": []}
    return data


def run_gap_analysis_tool(context: ToolContext, converted_file_path: str = "") -> dict:
    """Run the SEMS gap analysis on the converted PySpark file.

    Always runs the full 3-layer static gap analysis (syntax + AST/regex rules
    + 10 external tools). As a side effect, reformats the file in place with
    black/isort (--fix) before checking it — see ``_auto_format`` — so those
    two findings are resolved deterministically and should not appear in the
    gap list. The full gap list (every severity) is written to
    ``GAP_STATE_FILE`` on disk (not session state — see that constant's
    docstring) for write_gap_report to compose the report from; only
    blocking-severity gaps are returned here into the model's own context,
    since that's all the agent is asked to write prose for.

    Args:
        context: The agent state of type ToolContext.
        converted_file_path: Optional path to the converted PySpark file. When omitted,
            the newest ``*_spark.py`` in outputs/ is used.

    Returns:
        Dict with ``total_gaps``/``blocking_gaps`` (counts over ALL gaps),
        ``area_summaries``, and ``gaps`` — already filtered to blocking gaps
        only (see ``_gap_is_blocking``: MAJOR+ severity, or a SECURITY_HOTSPOT
        tool finding regardless of severity label), each with rule_id,
        severity, BRD area, location, description, remediation hint,
        auto_fixable, and a ``code_context`` field with the real source lines
        around the violation — ``null`` for gaps that flag the analysis
        environment itself, e.g. a missing/crashing external tool).
    """
    # Resolve file path from argument → session state → newest *_spark.py,
    # via the same shared resolver sems_agent uses (existence-checked, with
    # glob fallback), so both tools always pick the same file. No *.py
    # fallback: a stray non-converted file in outputs/ must error, not be
    # gap-analyzed.
    path = resolve_converted_path(
        converted_file_path
        or context.state.get("converted_pyspark_file_path")
        or context.state.get("converted_file_path", ""),
        fallback_any_py=False,
    )
    if path is None:
        return {"error": "Could not locate a converted PySpark file.", "total_gaps": 0, "gaps": []}

    #todo remove it temp for sems_only_agent
    context.state["converted_pyspark_file_path"] = str(path)
    # Static-only analysis — the ADK agent (Maverick) provides the LLM remediation.
    result = run_sems_gap_analyzer(script_path=path, include_llm_analysis=False)

    # Real lines from the converted file, keyed to each gap by rule_id/location,
    # so the LLM writes remediation grounded in the actual generated code
    # instead of a hallucinated "typical" pattern.
    source_lines = path.read_text(encoding="utf-8").splitlines()

    gaps_payload = [
        {
            "rule_id": g.rule_id,
            "severity": g.severity,
            "brd_area": g.brd_area,
            "location": g.location,
            "description": g.description,
            "remediation": g.remediation,
            "auto_fixable": g.auto_fixable,
            "is_blocking": g.is_blocking,
            "code_context": _code_context_for_gap(source_lines, g.rule_id, g.location),
        }
        for g in result.gaps
    ]

    # Blocking gaps first (worst severity first within each group) so they
    # surface immediately in GAP_STATE_FILE's "gaps" array instead of being
    # buried among however many hundred non-blocking entries — the .md report
    # already gives blocking gaps this same priority placement.
    gaps_payload = _order_gaps_blocking_first(gaps_payload)

    # Persist to disk for write_gap_report/write_final_gap_report/
    # _seed_sems_fix_bucket — those compose the non-blocking table + summary
    # (and the fix agent's buckets) deterministically from this data, so the
    # final report can never drop a gap the model didn't get around to
    # writing prose for.
    _write_gap_state(result.target, result.total_gaps, result.blocking_gaps, gaps_payload)

    # Only blocking-severity gaps go back into the model's context — STEP 2 of
    # the agent instruction only writes prose for these, and the non-blocking
    # bulk is rendered straight from GAP_STATE_FILE above by write_gap_report,
    # never seen by the model. Sending the full list here (hundreds of gaps
    # with code_context on a large file) is what blew past the Databricks
    # workspace input-tokens-per-minute limit.
    blocking_payload = [g for g in gaps_payload if g["is_blocking"]]

    return {
        "target": result.target,
        "total_gaps": result.total_gaps,
        "blocking_gaps": result.blocking_gaps,
        "area_summaries": result.area_summaries,
        "syntax_error": result.syntax_error,
        "gaps": blocking_payload,
    }


def write_gap_report(context: ToolContext, content: str) -> dict:
    """Compose and write the gap analysis report to the outputs directory.

    Args:
        context: Agent state.
        content: Markdown prose for the blocking-severity gaps only (or a
            one-line "No blocking gaps." / "No SEMS compliance gaps found."
            note) — everything else (non-blocking table, summary, fix order)
            is composed deterministically from ``run_gap_analysis_tool``'s
            data via ``_compose_gap_report``.

    Returns:
        Dict with the path written.
    """
    converted_path_str = (
        context.state.get("converted_pyspark_file_path")
        or context.state.get("converted_file_path", "")
    )
    print(f"=========== 1 {converted_path_str}")
    stem = canonical_stem(converted_path_str) if converted_path_str else "pipeline"
    print(f"=========== 2 {stem}")
    full_content = _compose_gap_report(context, content)
    report_path = OUTPUTS_DIR / f"{stem}_gap_analysis_report.md"
    print(f"=========== 3 {report_path}")
    report_path.write_text(full_content, encoding="utf-8")
    context.state["gap_report_path"] = str(report_path)
    context.state["gap_report_content"] = full_content
    return {"message": f"Gap analysis report written to {report_path}", "path": str(report_path)}


def write_final_gap_report(context: ToolContext, content: str) -> dict:
    """Compose and write the FINAL gap analysis report — the re-check run
    after sems_fix_agent has applied fixes. Overwrites the same
    ``{stem}_gap_analysis_report.md`` that write_gap_report wrote pre-fix, so
    the report always reflects the converted file's current compliance state
    rather than accumulating a separate stale pre-fix copy.

    Args:
        context: Agent state.
        content: Markdown prose for the blocking-severity gaps only (see
            write_gap_report) — the rest is composed deterministically.

    Returns:
        Dict with the path written.
    """
    converted_path_str = (
        context.state.get("converted_pyspark_file_path")
        or context.state.get("converted_file_path", "")
    )
    stem = canonical_stem(converted_path_str) if converted_path_str else "pipeline"

    full_content = _compose_gap_report(context, content)
    report_path = OUTPUTS_DIR / f"{stem}_gap_analysis_report.md"
    report_path.write_text(full_content, encoding="utf-8")
    context.state["gap_report_final_path"] = str(report_path)
    context.state["gap_report_final_content"] = full_content
    return {"message": f"Final gap analysis report written to {report_path}", "path": str(report_path)}


def _manual_action_rule_ids(gap_state: Dict[str, Any]) -> List[str]:
    """Blocking gaps' rule_ids that are in _MANUAL_ACTION_ONLY_RULES and still
    present — surfaced in state so a caller can tell "compliant, but still
    needs a human to do X" apart from "genuinely zero gaps."""
    return sorted({
        g.get("rule_id", "")
        for g in gap_state.get("gaps", [])
        if g.get("is_blocking") and g.get("rule_id") in _MANUAL_ACTION_ONLY_RULES
    })


def check_sems_compliance(callback_context: CallbackContext) -> None:
    """Escalation criteria for the sems_correction_loop_agent LoopAgent.

    Reads the blocking-gap count run_gap_analysis_tool just persisted to
    GAP_STATE_FILE on disk (written during this same agent turn) and
    escalates once the ACTIONABLE count (see _actionable_blocking_count) is
    zero, stopping the loop before max_iterations rather than continuing to
    fix a file that is already as SEMS-compliant as this pipeline can make
    it. Gaps in _MANUAL_ACTION_ONLY_RULES (e.g. NO_TESTS) don't count toward
    this — no fixer can ever resolve them, so waiting on them would just burn
    every iteration for nothing; they're still reported via
    sems_manual_action_required below.
    """
    state = callback_context.state
    gap_state = read_gap_state()
    if _actionable_blocking_count(gap_state) == 0:
        state["sems_compliance_passed"] = True
        state["sems_manual_action_required"] = _manual_action_rule_ids(gap_state)
        callback_context.actions.escalate = True
    else:
        state["sems_compliance_passed"] = False
    return None


def check_sems_compliance_final(callback_context: CallbackContext) -> None:
    """Records the FINAL SEMS compliance verdict, using the blocking-gap
    count this agent's own run_gap_analysis_tool call just persisted to disk.

    Unlike check_sems_compliance (used inside the fix LoopAgent), this never
    escalates — sems_gap_analyzer_final_agent is a one-shot terminal re-check
    that runs once after the fix loop ends (whether it escalated early or
    exhausted max_iterations), so there is no loop left to break out of. It
    exists so `sems_compliance_passed` always reflects the code's truly-final
    state rather than a stale pre-last-fix snapshot from inside the loop.
    Uses the same actionable-count exclusion as check_sems_compliance — see
    its docstring for why _MANUAL_ACTION_ONLY_RULES gaps don't count.
    """
    state = callback_context.state
    gap_state = read_gap_state()
    state["sems_compliance_passed"] = _actionable_blocking_count(gap_state) == 0
    state["sems_manual_action_required"] = _manual_action_rule_ids(gap_state)
    return None


# ForceToolLiteLlm forces the run_gap_analysis_tool call per request, then
# relaxes so the model can produce the remediation report.
_gap_model = ForceToolLiteLlm(
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    num_retries=1,
    forced_tool="run_gap_analysis_tool",
)


def _gap_force(callback_context, llm_request):
    """Inject the gap-skill guidance into every model request."""
    llm_request.append_instructions([_SKILL_GUIDANCE])
    return None


def _gap_analyzer_instruction(write_tool_name: str, *, extra: str = "") -> str:
    """Build the gap-analyzer instruction, parameterized by which write tool to
    call — lets sems_gap_analyzer_agent (pre-fix) and sems_gap_analyzer_final_agent
    (post-fix) share one instruction body without duplicating it."""
    return f"""
    You are the SEMS gap analysis agent. Your job is to write detailed, developer-ready
    before/after remediation prose for the BLOCKING-severity gaps only. Everything else
    (the header, the non-blocking gaps, the summary table, the fix order) is compiled
    automatically from the same data by the write tool — you never touch those, and you
    never need to enumerate large numbers of gaps yourself.
    {extra}
    You also have access to skill tools (list_skills / load_skill / load_skill_resource /
    run_skill_script) for the sems-check-skill. Do NOT call any of them — the skill's full
    guidance (including the SEMS rules reference) is already provided to you below/in your
    instructions, so there is nothing left to look up. Your very first tool call, every
    time, must be run_gap_analysis_tool.
    ─────────────────────────────────────────────────────────────────────────────
    STEP 1 — Gather gap data
    ─────────────────────────────────────────────────────────────────────────────
    Call **run_gap_analysis_tool** exactly once. It returns:
      - `gaps`: list of violations, each with rule_id, severity, brd_area, location,
        description, remediation hint, auto_fixable flag, and **code_context** —
        the ACTUAL source lines from the converted PySpark file surrounding the
        violation (the flagged line is marked with `>>>`). `code_context` is
        `null` only for gaps about the analysis environment itself (rule_id
        TOOL_MISSING / TOOL_ERROR — a missing or crashing external tool), which
        have no corresponding code location.
      - `total_gaps`, `blocking_gaps`
      - `area_summaries`: per-BRD-area pass/fail breakdown

    If `total_gaps=0`, call **{write_tool_name}** with a one-line
    "No SEMS compliance gaps found." message and stop.

    ─────────────────────────────────────────────────────────────────────────────
    STEP 2 — Write prose for every gap you were given
    ─────────────────────────────────────────────────────────────────────────────
    `gaps` is ALREADY filtered to blocking-severity items (MAJOR+, or a security
    hotspot regardless of its severity label) — write about every gap in that list.
    Do NOT re-filter it by severity text yourself (a hotspot may show severity
    "MINOR" and still belong here), and do NOT write anything about gaps NOT in
    this list — those are compiled and appended automatically after your content,
    from the exact same data, so omitting them here is correct, not an oversight.

    If `gaps` is empty, your entire output is the one line
    "No blocking-severity gaps." — do not write anything else, do not touch the
    non-blocking gaps.

    For each blocking gap, write a section using this exact structure:

    ### Gap N: [SEVERITY] RULE_ID — BRD Area
    **What is wrong**: one sentence describing the specific problem, naming the
    real variable/function/line from this gap's `code_context`.
    **Before (non-compliant)**:
    ```python
    <the exact lines from this gap's code_context, verbatim — strip only the
    leading line-number/">>>" gutter. Never substitute a generic example.>
    ```
    **After (SEMS-compliant fix)**:
    ```python
    <the SAME lines with the minimal real edit that fixes THIS violation (e.g.
    add the missing import the code actually needs, delete the actual dead
    variable, rename the actual DataFrame variable) — must be valid, runnable
    Python, never a placeholder like `some_value` or an invented keyword.>
    ```
    **Why this matters**: one sentence on the SEMS/STAM rationale.
    **Auto-fixable**: Yes / No — manual change required.

    If `code_context` is `null` (TOOL_MISSING / TOOL_ERROR — an environment
    issue, not a code issue), skip the Before/After code blocks entirely and
    instead show the shell command that installs or repairs the tool, e.g.:
    **Fix**:
    ```sh
    pip install bandit
    ```
    Only name a package you can confirm exists on PyPI under that exact name
    (the tool name reported by run_gap_analysis_tool) — never guess a plugin
    or package name.

    Never fabricate a gap, a line, or a code snippet that isn't backed by this
    gap's `code_context`. If you are not fully certain what the surrounding
    code looks like, use `code_context` exactly as given rather than guessing.

    N here numbers only the blocking gaps you are writing about (1, 2, 3, ...) —
    it does not need to match the gap's position in the full `gaps` list.

    ─────────────────────────────────────────────────────────────────────────────
    STEP 3 — Write the report
    ─────────────────────────────────────────────────────────────────────────────
    Call **{write_tool_name}** with ONLY the blocking-gap sections you wrote in
    STEP 2 (or the one-line note if there were none / zero total gaps). Do NOT
    include a header, a non-blocking gap list, a summary table, or a fix-order
    list — the tool builds and appends all of that itself from the full gap data,
    so anything you add there yourself would just duplicate it.

    Strict rules:
    - Use only the gaps returned by run_gap_analysis_tool — do not fabricate violations.
    - Every Before/After snippet must come from that gap's real `code_context` —
      never a generic or hypothetical pattern, even if it "typifies" the rule.
    - The write tool must be called even if there are zero blocking gaps.
    """


sems_gap_analyzer_agent = Agent(
    name="sems_gap_analyzer_agent",
    model=_gap_model,
    before_model_callback=_gap_force,
    description=(
        "Analyzes every SEMS compliance gap in the converted PySpark file and produces "
        "a detailed before/after remediation report, ordered by priority."
    ),
    instruction=_gap_analyzer_instruction("write_gap_report"),
    tools=[sems_gap_skill_toolset, run_gap_analysis_tool, write_gap_report],
    mode="single_turn",
    # This agent re-runs the static analysis fresh off disk every call and
    # needs nothing from earlier loop iterations — without this, each
    # sems_correction_loop_agent iteration resends the ENTIRE prior
    # conversation (including sems_fix_agent's tool dumps), which is what
    # blew past the Databricks workspace input-tokens-per-minute limit.
    include_contents="none",
    output_key="gap_analysis_output",
    after_agent_callback=check_sems_compliance,
)


# ---------------------------------------------------------------------------
# Final gap analyzer — a SECOND instance run after sems_fix_agent applies
# fixes, to report what remains. Distinct object (ADK forbids one instance
# under two parents), writing to a separate _final report so the pre-fix
# report is preserved rather than overwritten.
# ---------------------------------------------------------------------------

sems_gap_analyzer_final_agent = Agent(
    name="sems_gap_analyzer_final_agent",
    model=_gap_model,
    before_model_callback=_gap_force,
    description=(
        "Re-runs the SEMS gap analysis after sems_fix_agent has applied fixes, "
        "reporting what compliance gaps remain."
    ),
    instruction=_gap_analyzer_instruction(
        "write_final_gap_report",
        extra=(
            "\n    This is the FINAL pass, run after sems_fix_agent has already applied "
            "fixes for the earlier report — report what compliance gaps remain now.\n"
        ),
    ),
    tools=[sems_gap_skill_toolset, run_gap_analysis_tool, write_final_gap_report],
    mode="single_turn",
    include_contents="none",
    output_key="gap_analysis_final_output",
    after_agent_callback=check_sems_compliance_final,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "SEMS Gap Analyzer Agent — runs the full 3-layer SEMS compliance check "
            "on a PySpark script and uses an LLM to generate specific before/after "
            "code fixes for every gap."
        )
    )
    parser.add_argument("target", help="PySpark .py file to analyze")
    parser.add_argument(
        "--json", action="store_true", dest="emit_json",
        help="Print JSON gap list to stdout (in addition to the Markdown report)",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Write the Markdown gap report to FILE",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the LLM analysis and return the static rule-based report only",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 2

    result = run_sems_gap_analyzer(
        script_path=target,
        include_llm_analysis=not args.no_llm,
    )

    print()
    print("=" * 72)
    print(f"  SEMS GAP ANALYZER — {target.name}")
    print("=" * 72)
    print(f"  Total Gaps   : {result.total_gaps}  |  Blocking: {result.blocking_gaps}")
    if result.syntax_error:
        print(f"  Syntax Error : {result.syntax_error}")
    print()
    print("  BRD Area Coverage:")
    for area, summary in result.area_summaries.items():
        status = "PASS" if summary["passed"] else "FAIL"
        print(
            f"    {status:<5}  {summary['label']:<35}  "
            f"Total={summary['total']}  Blocking={summary['blocking']}"
        )
    print("=" * 72)
    print()
    print(result.gap_report_markdown)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(result.gap_report_markdown, encoding="utf-8")
        print(f"\nGap report written to: {out_path}")

    if args.emit_json:
        json_output = {
            "target": result.target,
            "total_gaps": result.total_gaps,
            "blocking_gaps": result.blocking_gaps,
            "area_summaries": result.area_summaries,
            "gaps": _order_gaps_blocking_first([asdict(g) for g in result.gaps]),
        }
        print(json.dumps(json_output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
