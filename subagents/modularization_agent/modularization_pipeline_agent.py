"""Modularization pipeline agent.

Repackages the single, SEMS-compliant converted file the rest of the
pipeline produces into the Shell modularization layout: main.py (business
logic), utilities.py (reusable helpers), config.py (configuration), usage.py
(a runnable example), README.md, LICENSE.txt, sonar-project.properties, and
tests/test_main.py.

This is what closes the NO_TESTS gap that pytest-cov (see
compliance/pre_sonar_check.py) already raises but that sems_gap_analyzer_agent
explicitly cannot fix — see ``_MANUAL_ACTION_ONLY_RULES`` in
sems_gap_analyzer_agent.py: no fixer in that loop can create a new file, only
patch the one canonical converted file. This stage is that missing fixer,
run once after the file is already fully SEMS-compliant.

Two stages, mirroring sems_correction_loop_agent's analyze-then-fix shape:
  1. modularization_split_agent      - deterministic AST split + template
     files, forced through run_modularization_tool. No model judgment.
  2. modularization_test_loop_agent  - an LLM writes real pytest tests for
     main.py and checks coverage in the same turn, looping until pytest-cov
     clears the Sonar Way 80% gate or max_iterations is reached.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import tempfile
from typing import Any, Dict

from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

from ...utils import canonical_stem, resolve_converted_path
from ..sems_agent.model_utils import ForceToolLiteLlm
from . import splitter, templates

logger = logging.getLogger(__name__)

#: Same layout convention as sems_gap_analyzer_agent.OUTPUTS_DIR — this file
#: lives at subagents/modularization_agent/, so parents[2] is the project root.
OUTPUTS_DIR = pathlib.Path(__file__).parents[2] / "outputs"

#: Sonar Way Quality Gate — coverage on new code must be >= 80% (mirrors
#: _MIN_COVERAGE_PCT in compliance/pre_sonar_check.py).
_MIN_COVERAGE_PCT = 80.0
_COVERAGE_TIMEOUT_SECS = 300

#: Substrings in a coverage-check error that no amount of extra test-writing
#: can fix — an environment problem, not a test-content problem. The loop
#: escalates immediately on these instead of burning every iteration.
_ENV_ERROR_MARKERS = ("not installed", "timed out")


def _modularized_dir(stem: str) -> pathlib.Path:
    return OUTPUTS_DIR / f"{stem}_modularized"


# ── Stage 1: deterministic split ──────────────────────────────────────────


def run_modularization_tool(context: ToolContext, converted_file_path: str = "") -> Dict[str, Any]:
    """Split the final converted file into the Shell modularization layout.

    Deterministic — see ``splitter.split_module``. Writes main.py,
    utilities.py, config.py, usage.py, README.md, LICENSE.txt, and
    sonar-project.properties under ``outputs/<stem>_modularized/``, plus an
    empty ``tests/__init__.py`` so pytest treats tests/ as a package.

    Args:
        context: The agent state of type ToolContext.
        converted_file_path: Optional path to the converted file. When
            omitted, the same resolution order as the SEMS stage is used
            (session state, then the newest ``*_spark.py`` in outputs/).

    Returns:
        Dict with ``modularized_dir``, the files written, the entrypoint the
        splitter detected, the main/utility function lists, and any
        best-effort warnings from the split.
    """
    path = resolve_converted_path(
        converted_file_path
        or context.state.get("converted_pyspark_file_path")
        or context.state.get("converted_file_path", ""),
        fallback_any_py=False,
    )
    if path is None:
        return {"error": "Could not locate a converted file to modularize."}

    source = path.read_text(encoding="utf-8")
    stem = canonical_stem(path)
    result = splitter.split_module(source, module_basename=stem)
    if not result.ok:
        return {"error": result.error, "warnings": result.warnings}

    out_dir = _modularized_dir(stem)
    (out_dir / "tests").mkdir(parents=True, exist_ok=True)
    (out_dir / "main.py").write_text(result.main_code, encoding="utf-8")
    (out_dir / "utilities.py").write_text(result.utilities_code, encoding="utf-8")
    (out_dir / "config.py").write_text(result.config_code, encoding="utf-8")
    (out_dir / "usage.py").write_text(result.usage_code, encoding="utf-8")
    (out_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (out_dir / "README.md").write_text(
        templates.render_readme(
            stem,
            main_functions=result.main_functions,
            utility_functions=result.utility_functions,
            entrypoint=result.entrypoint,
        ),
        encoding="utf-8",
    )
    (out_dir / "LICENSE.txt").write_text(templates.render_license(), encoding="utf-8")
    (out_dir / "sonar-project.properties").write_text(
        templates.render_sonar_properties(stem, stem), encoding="utf-8"
    )

    context.state["modularized_dir"] = str(out_dir)
    context.state["modularized_stem"] = stem

    return {
        "modularized_dir": str(out_dir),
        "files_written": [
            "main.py", "utilities.py", "config.py", "usage.py",
            "README.md", "LICENSE.txt", "sonar-project.properties", "tests/__init__.py",
        ],
        "entrypoint": result.entrypoint,
        "main_functions": result.main_functions,
        "utility_functions": result.utility_functions,
        "warnings": result.warnings,
    }


_split_model = ForceToolLiteLlm(
    model=LiteLlm(model="databricks/databricks-claude-sonnet-4-6"),
    num_retries=1,
    forced_tool="run_modularization_tool",
)

modularization_split_agent = Agent(
    name="modularization_split_agent",
    model=_split_model,
    description=(
        "Deterministically splits the final converted file into the Shell "
        "modularization layout (main.py/utilities.py/config.py/usage.py plus "
        "README/LICENSE/sonar-project.properties) by calling run_modularization_tool."
    ),
    instruction="""
    Call **run_modularization_tool** exactly once, with no arguments — it resolves the
    converted file itself. This is a deterministic, AST-based split; there is nothing
    for you to decide or reason about. After the call, report only the list of files
    written and the entrypoint/function counts it returned. Do not open, read, or
    reproduce any of the generated source.
    """,
    tools=[run_modularization_tool],
    mode="single_turn",
    include_contents="none",
    output_key="modularization_split_output",
)


# ── Stage 2: iterative test generation ────────────────────────────────────


def get_modularization_context_tool(context: ToolContext) -> Dict[str, Any]:
    """Return main.py/utilities.py source and the latest coverage result.

    Grounds the test-writer agent in the real generated code and, on repeat
    loop iterations, the exact lines pytest-cov still reports as uncovered —
    the same "ground remediation in real code, not a guess" principle
    sems_gap_analyzer_agent applies via code_context.
    """
    out_dir_str = context.state.get("modularized_dir")
    if not out_dir_str:
        return {"error": "run_modularization_tool has not run yet."}
    out_dir = pathlib.Path(out_dir_str)
    test_path = out_dir / "tests" / "test_main.py"
    return {
        "modularized_dir": str(out_dir),
        "main_source": (out_dir / "main.py").read_text(encoding="utf-8"),
        "utilities_source": (out_dir / "utilities.py").read_text(encoding="utf-8"),
        "existing_test_file": test_path.read_text(encoding="utf-8") if test_path.exists() else "",
        "last_coverage_result": context.state.get("modularization_coverage_result"),
    }


def write_test_file_tool(context: ToolContext, content: str) -> Dict[str, Any]:
    """Overwrite tests/test_main.py with the given content.

    Args:
        context: Agent state.
        content: The COMPLETE test file source — this replaces
            tests/test_main.py, it does not patch it.
    """
    out_dir_str = context.state.get("modularized_dir")
    if not out_dir_str:
        return {"error": "run_modularization_tool has not run yet."}
    test_path = pathlib.Path(out_dir_str) / "tests" / "test_main.py"
    test_path.write_text(content, encoding="utf-8")
    return {"message": f"Test file written to {test_path}", "path": str(test_path)}


def _run_pytest_cov(out_dir: pathlib.Path) -> Dict[str, Any]:
    """Run pytest-cov against tests/test_main.py, targeting main.py.

    Run with cwd=out_dir so `--cov=main` resolves against this project's own
    main.py and `from main import ...` / `from utilities import ...` /
    `from config import ...` inside the generated tests resolve as plain
    imports, exactly as documented in README.md's Usage section.
    """
    test_path = out_dir / "tests" / "test_main.py"
    if not test_path.exists():
        return {"passed": False, "percent": 0.0, "error": "tests/test_main.py has not been written yet."}

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_report = pathlib.Path(tmp_dir) / "coverage.json"
        try:
            proc = subprocess.run(
                [
                    "pytest",
                    str(test_path),
                    "--cov=main",
                    f"--cov-report=json:{json_report}",
                    "--cov-report=",
                    "-q",
                ],
                cwd=str(out_dir),
                capture_output=True,
                text=True,
                timeout=_COVERAGE_TIMEOUT_SECS,
            )
        except FileNotFoundError:
            return {"passed": False, "percent": 0.0, "error": "pytest is not installed in this environment."}
        except subprocess.TimeoutExpired:
            return {"passed": False, "percent": 0.0, "error": "pytest run timed out."}

        if not json_report.exists():
            detail = (proc.stderr.strip() or proc.stdout.strip() or "no coverage report produced")[:2000]
            return {"passed": False, "percent": 0.0, "error": detail, "pytest_stdout_tail": proc.stdout[-2000:]}

        try:
            data = json.loads(json_report.read_text())
        except json.JSONDecodeError:
            return {"passed": False, "percent": 0.0, "error": "non-JSON coverage report"}

    percent = data.get("totals", {}).get("percent_covered", 0.0)
    missing_lines = {
        filename: info.get("missing_lines", [])
        for filename, info in data.get("files", {}).items()
        if info.get("missing_lines")
    }
    return {
        "passed": percent >= _MIN_COVERAGE_PCT,
        "percent": round(percent, 1),
        "missing_lines": missing_lines,
        "pytest_stdout_tail": proc.stdout[-2000:],
    }


def run_coverage_check_tool(context: ToolContext) -> Dict[str, Any]:
    """Run pytest-cov against the modularized project and persist the result.

    Args:
        context: Agent state.

    Returns:
        Dict with ``passed`` (bool, >= 80%), ``percent``, ``missing_lines``
        (per file, when below the gate), and ``error`` when the run itself
        failed rather than merely falling short of the coverage gate.
    """
    out_dir_str = context.state.get("modularized_dir")
    if not out_dir_str:
        return {"error": "run_modularization_tool has not run yet."}
    result = _run_pytest_cov(pathlib.Path(out_dir_str))
    context.state["modularization_coverage_result"] = result
    return result


def check_modularization_coverage(callback_context: CallbackContext) -> None:
    """Escalation criteria for modularization_test_loop_agent.

    Escalates once pytest-cov reports >= 80% coverage, OR once the failure is
    an environment problem (pytest missing, timeout) that writing more tests
    can never fix — looping on those would just burn every iteration for
    nothing, the same reasoning check_sems_compliance applies to
    _MANUAL_ACTION_ONLY_RULES gaps.
    """
    state = callback_context.state
    result = state.get("modularization_coverage_result") or {}
    state["modularization_tests_passed"] = bool(result.get("passed"))
    error = str(result.get("error") or "")
    if result.get("passed") or any(marker in error for marker in _ENV_ERROR_MARKERS):
        if error:
            state["modularization_tests_blocked_reason"] = error
        callback_context.actions.escalate = True


_test_writer_model = ForceToolLiteLlm(
    model=LiteLlm(model="databricks/databricks-claude-sonnet-4-6"),
    num_retries=1,
    forced_tool="get_modularization_context_tool",
)

modularization_test_writer_agent = Agent(
    name="modularization_test_writer_agent",
    model=_test_writer_model,
    description=(
        "Writes real pytest unit tests for main.py and checks coverage, "
        "looping until the Sonar Way 80% gate is cleared."
    ),
    instruction="""
    You are writing tests/test_main.py for the modularized project run_modularization_tool
    already produced. The goal is real, passing pytest coverage of main.py of at least
    80% (the Sonar Way Quality Gate) — not a placeholder test file.

    STEP 1 — Call **get_modularization_context_tool** exactly once. It returns:
      - main_source: the full source of main.py you are testing.
      - utilities_source: the helper functions main.py depends on.
      - existing_test_file: the current tests/test_main.py, if a prior loop iteration
        already wrote one — build on it and extend it rather than starting over.
      - last_coverage_result: the previous run's percent/missing_lines/error, if any.
        Use missing_lines to target exactly the lines still uncovered (error paths,
        empty-input edge cases, conditional branches) instead of re-testing what
        already passes.

    STEP 2 — Write pytest test functions covering every function in main.py:
      - Real assertions grounded in the actual function logic: construct realistic
        input values (or small DataFrames) and assert on the actual expected output.
        Never write `assert True` or a bare smoke test that calls a function and
        checks nothing.
      - Import directly — `from main import <name>`, `from utilities import <name>`,
        `from config import <name>` — these files sit alongside tests/ in the same
        directory (see the resolved modularized_dir).
      - If a function needs a SparkSession, add ONE local, session-scoped fixture:
            @pytest.fixture(scope="session")
            def spark():
                from pyspark.sql import SparkSession
                return SparkSession.builder.master("local[1]").appName("tests").getOrCreate()
        `.master("local[1]")` is correct here — the SEMS rule against `.master("local")`
        (SHELL_STAM001) governs production pipeline code, not local test fixtures.

    STEP 3 — Call **write_test_file_tool** with the COMPLETE test file content (it
    overwrites tests/test_main.py, so include everything, not a diff against the
    existing file).

    STEP 4 — Call **run_coverage_check_tool** to measure the result.

    Stop after STEP 4. Do not call any tool a second time in this turn — if more
    work is needed, the loop will re-invoke you with the fresh coverage numbers.
    """,
    tools=[
        get_modularization_context_tool,
        write_test_file_tool,
        run_coverage_check_tool,
    ],
    mode="single_turn",
    include_contents="none",
    output_key="modularization_test_output",
    after_agent_callback=check_modularization_coverage,
)

modularization_test_loop_agent = LoopAgent(
    name="modularization_test_loop_agent",
    description=(
        "Writes and iterates tests/test_main.py until pytest-cov reports >= 80% "
        "coverage on main.py or max_iterations is reached."
    ),
    sub_agents=[modularization_test_writer_agent],
    # Each iteration both writes tests AND checks coverage in the same turn (see
    # STEP 3/STEP 4 above), so — unlike sems_gap_fix_loop_agent — there's no
    # analyze/fix staleness gap needing a separate final re-check agent after
    # the loop. 4 rounds is enough to converge on a single generated file
    # without the multi-area fan-out sems_gap_fix_loop_agent has to budget for.
    max_iterations=4,
)

modularization_agent = SequentialAgent(
    name="modularization_agent",
    description=(
        "Repackages the final SEMS-compliant converted file into the Shell "
        "modularization layout (main.py/utilities.py/config.py/usage.py, "
        "README.md, LICENSE.txt, sonar-project.properties) and writes/iterates "
        "real pytest unit tests until the 80% coverage gate is met."
    ),
    sub_agents=[modularization_split_agent, modularization_test_loop_agent],
)
