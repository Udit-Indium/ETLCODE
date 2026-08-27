"""Standalone parity-test app: write a pytest suite, run it, repair what fails.

Deliberately NOT a stage in the conversion pipeline. Every sub_agent of the
orchestrator shares one session, so whatever runs last inherits all the events
before it — running fourth put this agent's prompt at 176,000-202,000 tokens
against a 200,000 ITPM quota, almost entirely from conversion history it never
read. A separate app gets a fresh session and starts near zero.

Run it after the pipeline finishes:

    cd <repo>/subagents && adk web        # then pick "parity_app"

It finds its own target — the newest `*_spark.py` in outputs/, or whatever
PARITY_TARGET_FILE names — so it needs nothing handed to it. (It will still use
`converted_pyspark_file_path` if a session happens to carry one, which is what
lets `build_parity_loop()` be dropped back into a sub_agents list unchanged if
you ever want the stage back.)

A failing suite is repaired, not just reported: write tests -> run -> fix the
converted module -> re-test, stopping when every function has a test and the
suite is green. The fixer edits the converted PySpark only, never the tests —
a correct test that fails is the signal the conversion is wrong.

Three steps, each with exactly one job, because two of them are LLM agents that
should not be paid for a turn they have nothing to do in:

    writer  -> writes outputs/pyspark_pytest.py, and NOTHING else. It does not
               run the suite, judge it, or touch the converted module.
    verdict -> runs the suite on Databricks and records the verdict. No model,
               no tools, no cost. Stops the loop when coverage is complete and
               the suite passes.
    fixer   -> edits the converted module against the failures. Never the tests.

The verdict is a step rather than a callback on the writer because ADK ends an
agent's run as soon as a before-agent callback returns content, and returns
BEFORE the after-agent callback. While the two were welded together the writer
could not be skipped without also skipping the suite run and the escalate that
ends the loop — so it was handed its whole prompt every iteration just to reply
that it had nothing left to write.

Two things come out of a run:

  * `outputs/pyspark_pytest.py` — the generated suite itself, and the ONLY
    artefact that outlives the session. It is also the input to the next run:
    `add_pytest_tests_tool` merges each batch into whatever is already there.
  * `state["parity_test_status"]` — the verdict (status, the functions still
    missing a test, and the pytest errors on a failing run), written on every
    pass of `check_test_case_status`. It lives in session state and disappears
    with the session, so read it while the run is live.

There is deliberately no durable result JSON and no pass receipt. Earlier
versions wrote `parity_result.json` and `parity_last_pass.json`, the latter to
skip an unchanged module on the next run; both went when the agent was rewritten
around the plain write/run/fix loop. If you want the verdict on disk, add it —
do not go looking for a file that used to be written.
"""

from __future__ import annotations
import os

from google.adk.agents import LoopAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from .agent import build_parity_agent, build_parity_verdict_agent
# The fixer lives with the converter because it uses the converter's tools
# (replace_functions_tool, execute_pyspark_script_tool, the conventions skill)
# to edit the module in place.
from ..conversion_loop.code_converter import build_code_fixer_agent

MAX_ITERATIONS = 12


def build_parity_loop(name: str = "parity_test_agent") -> LoopAgent:
    """Construct a FRESH parity stage, agent and loop both.

    Built per call rather than shared, for the same reason `build_parity_agent`
    is a factory: ADK sets `parent_agent` on every entry of a `sub_agents` list,
    so one instance used by both the standalone App and the orchestrator would
    end up owned by whichever imported last.
    """
    return LoopAgent(
        name=name,
        description=(
            "Writes and runs a pytest parity suite against the converted "
            "PySpark module and repairs that module against any failures, "
            "looping until every function has a test and the suite passes."
        ),
        sub_agents=[
            build_parity_agent(f"{name}_writer"),
            build_parity_verdict_agent(f"{name}_verdict"),
            build_code_fixer_agent(f"{name}_fixer"),
        ],
        max_iterations=MAX_ITERATIONS,
    )
root_agent = build_parity_loop("parity_app")

#: How often session history is summarised. Tunable without a code edit,
#: because it is the one lever that shrinks the prompt without changing what an
#: agent can see inside a turn — it trades summarisation calls for prompt size,
#: so a wrong value costs money or latency, never correctness.
#:
#: Lowered from 5 to 3 after a run hit the Databricks ITPM ceiling: a single
#: request carried a 202,272-token prompt against a 200,000 ITPM quota, so no
#: pacing could make it fit. Later stages inherit every earlier stage's events
#: in one shared session, so the prompt grows across the whole pipeline rather
#: than within one agent.
COMPACTION_INTERVAL = int(os.environ.get("ADK_COMPACTION_INTERVAL", "3"))
COMPACTION_OVERLAP = int(os.environ.get("ADK_COMPACTION_OVERLAP", "2"))

events_compaction_config = EventsCompactionConfig(
    compaction_interval=COMPACTION_INTERVAL,
    overlap_size=COMPACTION_OVERLAP,
)
app = App(
    name="parity_app",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
__all__ = [
    "app",
    "root_agent",
    "build_parity_loop",
    "build_parity_agent",
    "build_parity_verdict_agent",
]
