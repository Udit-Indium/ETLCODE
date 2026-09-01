import os
from google.adk.agents import SequentialAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

from .subagents.code_parser import code_parser_agent
from .subagents.conversion_loop import (
    conversion_loop_agent,
    documentation_generator_agent,
    semantic_validation_loop_agent,
)
from .subagents.llm_call_logger import plugins as llm_log_plugins

root_agent = SequentialAgent(
    name="orchestrator_agent",
    description=(
        "Orchestrates the Python-to-PySpark conversion: parses the source script "
        "(restructuring a flat script into functions first, then building the AST "
        "inventory), runs the conversion loop that converts and fact-checks until "
        "the case facts match, runs the semantic-validation loop that compares "
        "Python vs PySpark outputs on a dummy dataset and fixes the converted code "
        "until they match, then documents the finished module — documentation runs "
        "LAST so it describes the code as the semantic fixer left it, and can "
        "report the validation verdict. The pytest parity suite is a separate app, "
        "run on demand against the finished module rather than as a stage here."
    ),
    sub_agents=[
        code_parser_agent,
        conversion_loop_agent,
        semantic_validation_loop_agent,
        documentation_generator_agent,
    ],
)

COMPACTION_INTERVAL = int(os.environ.get("ADK_COMPACTION_INTERVAL", "3"))
COMPACTION_OVERLAP = int(os.environ.get("ADK_COMPACTION_OVERLAP", "2"))

events_compaction_config = EventsCompactionConfig(
    compaction_interval=COMPACTION_INTERVAL,
    overlap_size=COMPACTION_OVERLAP,
)

app = App(
    name="code_converter",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
    plugins=[*llm_log_plugins()],
)
