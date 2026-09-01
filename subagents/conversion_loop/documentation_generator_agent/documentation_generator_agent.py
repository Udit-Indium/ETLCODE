"""Documentation writer for the converted PySpark module.

Reads the original script and the converted module — both paths come from state,
nothing is hardcoded — and writes MIGRATION_DOCUMENTATION.md next to the
converted file.
"""

from __future__ import annotations

import pathlib

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

DOC_FILENAME = "MIGRATION_DOCUMENTATION.md"

#: What replaces the document once it is safely on disk. Keeps the function call
#: in the transcript readable without carrying the payload — see
#: `discard_document_content`.
DOC_PLACEHOLDER = "<document omitted from history — it was written to {path}>"

#: Heading of the section `add_validation_notes_to_document` maintains. Anchored
#: at a line start when the tool looks for a previous run's section, so a rerun
#: replaces it instead of stacking a second copy.
VALIDATION_HEADING = "## Semantic validation"

#: `_semantic_compare` records up to 50 differences. The document wants the shape
#: of the mismatch, not the full log — the two JSON outputs on disk hold that.
MAX_LISTED_DIFFERENCES = 10

#: How the check works, in the reader's terms. Static prose: the method is a
#: property of the validation agent, not of any one run.
HOW_VALIDATION_WORKS = (
    "**How it works.** The semantic-validation agent runs the same data through "
    "both pipelines and compares what comes back. It builds a small dummy dataset "
    "(only when the pipeline reads external data — a self-contained pipeline is "
    "left alone), drives the original Python module and the converted PySpark "
    "module through two thin runner scripts, and has each write its final rows as "
    "canonical JSON: `orient=records`, rounded floats, ISO timestamps, so the two "
    "files are directly comparable. The comparison is order-insensitive and "
    "float-tolerant to 1e-6 — column sets first, then row count, then every cell. "
    "A difference always means the CONVERTED code is wrong: the source, the dummy "
    "data and the Python baseline are never edited to force a match. The code "
    "fixer repairs the PySpark module and the loop re-runs until the outputs match "
    "or the iteration budget is spent."
)


def read_file_tool(context: ToolContext, path: str) -> dict:
    """Read a file from disk and return its text.

    Use it for the original Python script and the converted PySpark module —
    their paths are in the prompt.

    Args:
        path: absolute path of the file to read.
    """
    try:
        return {
            "status": "success",
            "path": path,
            "content": pathlib.Path(path).read_text(encoding="utf-8"),
        }
    except OSError as exc:
        return {"status": "error", "path": path, "error": str(exc)}


def write_documentation_tool(context: ToolContext, markdown: str) -> dict:
    """Save the finished migration document.

    It is written as MIGRATION_DOCUMENTATION.md in the same folder as the
    converted PySpark module, so the document travels with the code it describes.

    Args:
        markdown: the complete document, in Markdown.
    """
    converted = context.state.get("converted_pyspark_file_path")
    if not converted:
        return {
            "status": "error",
            "error": "No converted file path in state — there is nothing to document.",
        }

    path = pathlib.Path(str(converted)).with_name(DOC_FILENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((markdown or "").strip() + "\n", encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "error": f"could not write {path}: {exc}"}

    context.state["documentation_file_path"] = str(path)
    return {"status": "success", "saved_file_path": str(path)}


def _validation_section(state) -> str:
    """Render the `## Semantic validation` section from state.

    Deterministic on purpose. Every fact here — the verdict, the differences, the
    artefact paths — was put in state by the validation agent's
    `check_semantic_match`. Handing the model the raw verdict and asking for prose
    would add one more place for a "passed" to appear over a run that did not.
    """
    verdict = state.get("semantic_match") or {}
    matched = bool(verdict.get("match"))
    differences = list(verdict.get("differences") or ())
    message = str(verdict.get("message") or "").strip()

    lines = [VALIDATION_HEADING, "", HOW_VALIDATION_WORKS, ""]

    artefacts = [
        ("dummy dataset", state.get("semantic_dummy_dir")),
        ("Python baseline output", state.get("semantic_python_output_path")),
        ("PySpark output", state.get("semantic_pyspark_output_path")),
    ]
    artefacts = [(label, value) for label, value in artefacts if value]
    if artefacts:
        lines += ["**What it produced**", "", "| artefact | path |", "| --- | --- |"]
        lines += [f"| {label} | `{value}` |" for label, value in artefacts]
        lines.append("")

    status = "MATCH" if matched else "NO MATCH"
    lines.append(f"**Verdict: {status}**" + (f" — {message}" if message else ""))

    if not matched and differences:
        shown = differences[:MAX_LISTED_DIFFERENCES]
        tail = "" if len(differences) <= MAX_LISTED_DIFFERENCES else (
            f" (first {MAX_LISTED_DIFFERENCES} shown)"
        )
        lines += ["", f"{len(differences)} difference(s) recorded{tail}:", ""]
        lines += [f"- {d}" for d in shown]

    setup_error = state.get("semantic_setup_error")
    if setup_error:
        lines += ["", f"**Setup problem:** {setup_error}"]

    if not matched:
        lines += [
            "",
            "The converted module still differs from the original on this dataset. "
            "Treat the differences above as open items on top of the gaps listed "
            "earlier in this document.",
        ]

    return "\n".join(lines)


def add_validation_notes_to_document(context: ToolContext) -> dict:
    """Append the semantic-validation outcome to the migration document.

    Adds a short `## Semantic validation` section covering how the check works,
    what it produced, and whether the two pipelines agreed.

    Takes no arguments: everything it writes is already in state, recorded by the
    semantic-validation agent. Call it after that agent has run — if no verdict
    has been recorded yet it changes nothing and reports `skipped`. Calling it
    again replaces the section rather than appending a second copy.
    """
    doc_path = context.state.get("documentation_file_path")
    if not doc_path:
        # The document is always written next to the converted module, so its
        # path is recoverable even if this tool runs in a session where
        # `write_documentation_tool` never published the key.
        converted = context.state.get("converted_pyspark_file_path")
        if not converted:
            return {
                "status": "error",
                "error": "No document path in state and no converted file to derive it from.",
            }
        doc_path = str(pathlib.Path(str(converted)).with_name(DOC_FILENAME))

    path = pathlib.Path(doc_path)
    if not path.exists():
        return {
            "status": "error",
            "error": f"{path} does not exist — write the document before appending to it.",
        }

    if not context.state.get("semantic_match"):
        return {
            "status": "skipped",
            "reason": (
                "Semantic validation has not recorded a verdict yet, so there is "
                "nothing to report. The document is unchanged."
            ),
        }

    try:
        existing = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "error": f"could not read {path}: {exc}"}

    # Drop a previous run's section, anchored at a line start so the heading has
    # to be a heading and not a mention of one inside the prose.
    head = existing.split("\n" + VALIDATION_HEADING)[0].rstrip()
    section = _validation_section(context.state)

    try:
        path.write_text(head + "\n\n" + section + "\n", encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "error": f"could not write {path}: {exc}"}

    return {
        "status": "success",
        "saved_file_path": str(path),
        "match": bool((context.state.get("semantic_match") or {}).get("match")),
    }


def seed_paths(callback_context: CallbackContext) -> None:
    """Make sure both path keys exist before the prompt interpolates them.

    The instruction references {source_script_path} and
    {converted_pyspark_file_path}; ADK raises on a missing key, so an absent
    path has to arrive as an empty string the agent can report, not as a crash.
    """
    state = callback_context.state
    for key in ("source_script_path", "converted_pyspark_file_path"):
        if not state.get(key):
            state[key] = ""
    return None


def discard_document_content(callback_context: CallbackContext) -> None:
    """Drop the finished document out of the session once it is on disk.

    The document is this agent's whole output, and it reached
    `write_documentation_tool` as a function-call ARGUMENT — so it lives in this
    agent's events. ADK relays one agent's events into the next agent's request
    verbatim: `_present_other_agent_message` renders them as "[agent] called tool
    `write_documentation_tool` with parameters: <the entire document>". This agent
    now runs LAST, so there is no following stage to hand it to — but the events
    outlive the turn: the compactor still summarises them, and anything that
    resumes this session carries a full migration document in every prompt from
    here on.

    So: rewrite the argument to a one-line pointer at the saved file. The file is
    the artefact; the only thing published to state is its path,
    `documentation_file_path`.

    Two limits worth knowing. This edits the in-memory event list, which is what
    the request builder reads — a session reloaded from a persistent session
    service would bring the original argument back. And the agent's own text
    turns are left untouched: they are short summaries, and the instruction has
    the document going through the tool, not through the reply.
    """
    invocation = getattr(callback_context, "_invocation_context", None)
    session = getattr(invocation, "session", None)
    saved = callback_context.state.get("documentation_file_path") or "disk"
    placeholder = DOC_PLACEHOLDER.format(path=saved)

    for event in getattr(session, "events", None) or ():
        if getattr(event, "author", None) != callback_context.agent_name:
            continue
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or ():
            call = getattr(part, "function_call", None)
            if not call or call.name != "write_documentation_tool":
                continue
            if call.args and call.args.get("markdown"):
                call.args["markdown"] = placeholder
    return None


documentation_generator_agent = Agent(
    name="documentation_generator_agent",
    model=LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    description=(
        "Writes MIGRATION_DOCUMENTATION.md for the converted PySpark module: what "
        "was converted, coverage, function-by-function mapping against the original "
        "script, gaps, hardcoded paths and credentials the reader must change, and "
        "the manual effort those gaps represent. Once semantic validation has run, "
        "appends how that check works and whether the two pipelines agreed."
    ),
    instruction="""You write the hand-over document for a Python → PySpark migration.
    Your reader is the engineer who has to run the converted code on THEIR system
    tomorrow, and the manager who has to budget whatever is left to do.

    The two files you document:
      * original Python script : {source_script_path}
      * converted PySpark file : {converted_pyspark_file_path}

    HOW TO WORK:
    1. Read BOTH files with **read_file_tool**. If a path is empty or the read fails,
       say so in the document instead of guessing.
    2. Compare them function by function, Class by Class, constant by constant, or file may be refactored as well if input file it ipynb file, then write the document with these sections:

       `# PySpark Migration Documentation`
       A short executive summary: the two file paths, how many source functions were
       converted, and the headline risks.

       `## What was converted`
       What the original script did, what the converted module does, and how it is
       meant to be run.

       `## Coverage`
       Counts and a percentage: source functions vs converted functions, classes,
       module-level constants. Name anything present in the source and absent from the
       converted file.

       `## Function-by-function, Class by Class, Constant by Constant, or file may be refactored as well if input file is ipynb file`
       A table: source function | what it does | converted function | status
       (converted / missing / changed) | notes. One row per source function. Add rows
       for converted functions that have NO source counterpart (helpers introduced by
       the conversion) and mark them as such.
       If the script is a jupyter notebbok then check which lines are combined to form a function and what it is covering.

       `## Gaps and differences`
       Everything a human still has to resolve: functions not converted, stubs or
       TODOs, changed constant values, pandas idioms left in Spark code, logic that
       may not be equivalent. Give the function name and line for each.

       `## Hardcoded paths to change`
       A table of every file path, URL, mount point, table or dataset name baked into
       the converted code: value | where (function + line) | what the reader must
       change it to.

       `## Credentials and configuration to change`
       A table of anything credential- or environment-shaped: hardcoded passwords,
       tokens, API keys, hosts, accounts, catalogs, and the environment variables or
       secret scopes the code expects to be set. Say what the reader must provide for
       each. NEVER copy a secret value into the document — write `<redacted>`.

       `## Manual effort required`
       For each gap above, an effort estimate in hours, then a total. Show your
       reasoning briefly (what makes each item cheap or expensive). Be honest about
       the uncertainty and give a range.

       `## Before you run this`
       A numbered checklist: credentials to set, paths to repoint, gaps to close,
       how to verify the output against the original.

    3. Call **write_documentation_tool(markdown=...)** ONCE with the complete
       document.

    4. Call **add_validation_notes_to_document()** ONCE. It takes no arguments and
       composes the semantic-validation section itself from the recorded verdict.
       Do NOT write that section yourself in step 2, and do not describe the
       validation outcome in your own words anywhere in the document — this tool
       is the only thing that reports it. If it returns `skipped`, semantic
       validation had not run yet; leave it at that and say nothing about
       validation.

    Then stop.

    STRICT RULES
    - Document only what you actually read. Never invent a function name, path,
      credential, or line number.
    - Do not soften the numbers: if 12 of 19 functions are converted, say 12 of 19.
    - Redact secret values.
    - Do not paste function bodies into the document — describe behaviour instead.
    - Markdown only; do not wrap the whole document in a code fence.
    """,
    tools=[
        read_file_tool,
        write_documentation_tool,
        add_validation_notes_to_document,
    ],
    mode="task",
    include_contents="none",
    after_agent_callback=discard_document_content,
)
