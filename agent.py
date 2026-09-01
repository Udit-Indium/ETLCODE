"""ADK discovery entrypoint.

`adk web` (NestedAgentLoader) only treats a directory as an app when it
contains `agent.py` or `root_agent.yaml` — a package that merely exports
`root_agent` from `__init__.py` is skipped during discovery, which is why the
dev UI reported "No agents found in current folder". Without this file the only
directory here that qualified was `subagents/parity_app` (it has an `agent.py`),
so the dev UI listed that one broken entry and never offered this pipeline at
all. The real wiring stays in `orchestrator_agent.py`; this module just
re-exports it.

`app` must be re-exported, not just `root_agent`. The loader checks this module
for an `app` attribute FIRST and only falls back to `root_agent` -- so exporting
the bare agent loads it without its App, silently discarding both the
events-compaction config and the plugins, which is where the call/state log
lives.
"""
from .subagents import litellm_patch
from .orchestrator_agent import app, root_agent

__all__ = ["app", "root_agent", litellm_patch]
