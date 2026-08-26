from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext

from google.adk.models.lite_llm import LiteLlm
from ...utils import canonical_stem
from ..conversion_loop.code_converter import execute_pyspark_script_tool, sems_gap_parallel_fix_agent
from .model_utils import ForceToolLiteLlm
from .sems_gap_analyzer_agent import OUTPUTS_DIR, sems_gap_analyzer_agent, sems_gap_analyzer_final_agent

sems_gap_fix_loop_agent = LoopAgent(
    name="sems_gap_fix_loop_agent",
    description=(
        "Runs the SEMS gap analyzer on the converted PySpark file and fixes "
        "blocking-severity compliance gaps (security, error handling, logging, "
        "readability, modular design), looping until zero blocking gaps remain "
        "or max_iterations is reached."
    ),
    sub_agents=[sems_gap_analyzer_agent, sems_gap_parallel_fix_agent],
    # sems_gap_parallel_fix_agent runs one fixer per BRD area IN PARALLEL, each
    # fixing exactly ONE bucket of gaps it owns this round 
    max_iterations=7,
)


_EXECUTION_HEADING = "## Final Databricks Execution Check"


def _record_final_execution_result(callback_context: CallbackContext) -> None:
    """Persist the final Databricks execution verdict and append it to the same
    gap-analysis report sems_gap_analyzer_final_agent just wrote.

    sems_gap_analyzer_agent's run_gap_analysis_tool reformats the file with
    black/isort on EVERY call (see _auto_format in sems_gap_analyzer_agent.py) —
    including the pass that finds zero blocking gaps and escalates the loop,
    and sems_gap_analyzer_final_agent's own guaranteed pass after the loop
    ends. final_execution_check_agent then runs the file on databricks.
    """
    state = callback_context.state
    status = state.get("pyspark_execution_status")
    passed = status == "SUCCESS"
    state["final_databricks_execution_passed"] = passed

    converted_path_str = (
        state.get("converted_pyspark_file_path")
        or state.get("converted_file_path", "")
    )
    stem = canonical_stem(converted_path_str) if converted_path_str else "pipeline"
    report_path = OUTPUTS_DIR / f"{stem}_gap_analysis_report.md"
    if not report_path.exists():
        return

    run_id = state.get("pyspark_execution_run_id", "unknown")
    if passed:
        section = (
            f"\n\n---\n\n{_EXECUTION_HEADING}\n\n"
            f"**Status**: PASS — the final converted file ran successfully on "
            f"Databricks (run_id: {run_id}).\n"
        )
    else:
        error = state.get("pyspark_execution_error", "") or "no error detail captured"
        section = (
            f"\n\n---\n\n{_EXECUTION_HEADING}\n\n"
            f"**Status**: FAIL — the final converted file did NOT run successfully "
            f"on Databricks (status: {status or 'unknown'}, run_id: {run_id}).\n\n"
            f"```\n{error}\n```\n"
        )
    with report_path.open("a", encoding="utf-8") as f:
        f.write(section)


# ForceToolLiteLlm guarantees the model actually calls execute_pyspark_script_tool
# instead of answering in plain text (see model_utils.ForceToolLiteLlm docstring).
_final_execution_model = ForceToolLiteLlm(
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    num_retries=1,
    forced_tool="execute_pyspark_script_tool",
)

# Runs once, after nothing else in the SEMS stage will touch the file again,
# to confirm the literal final on-disk artifact — not just the pre-last-
# reformat version — actually executes on Databricks.
final_execution_check_agent = Agent(
    name="final_execution_check_agent",
    model=_final_execution_model,
    description=(
        "Runs the truly-final converted file on Databricks one more time, after "
        "the SEMS gap-fix loop and the final gap analyzer have both finished "
        "mutating it, so the literal on-disk artifact is confirmed to actually "
        "execute rather than only the pre-last-reformat version."
    ),
    instruction="""
    The SEMS gap-fix loop and the final gap analyzer have both finished — the file on
    disk will not change again after this. Its very last mutation was a black/isort
    reformat done by the gap analyzer itself, which has never been executed.

    Call **execute_pyspark_script_tool** exactly once. It runs the current converted
    file on real Databricks compute and reports success/failure. Do NOT call any other
    tool, do NOT attempt to fix anything based on the result — this is a read-only
    final sanity check, not another fix iteration. Stop after the one call.
    """,
    tools=[execute_pyspark_script_tool],
    mode="single_turn",
    include_contents="none",
    output_key="final_execution_output",
    after_agent_callback=_record_final_execution_result,
)

sems_correction_loop_agent = SequentialAgent(
    name="sems_correction_loop_agent",
    description=(
        "Runs the SEMS fix loop (gap analysis + fixes, looping until zero "
        "blocking gaps remain or max_iterations is reached), re-runs the gap "
        "analyzer once more to produce the authoritative post-fix compliance "
        "report, then executes the truly-final file on Databricks once more to "
        "confirm the on-disk artifact actually runs."
    ),
    sub_agents=[sems_gap_fix_loop_agent, sems_gap_analyzer_final_agent, final_execution_check_agent],
)
