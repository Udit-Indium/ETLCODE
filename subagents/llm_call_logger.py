"""A readable audit log of every prompt this pipeline sends, and the state behind it.

The problem this exists for: the only evidence of prompt size was one line from
the rate limiter -- `RateLimiter | ... | prompt=95835` -- a `len(text) // 4`
guess with no breakdown and no record of the content. When a prompt turns out to
be 95,000 tokens there was no way to see WHAT those tokens were.

This writes one file per run, in banner-separated sections:

    RUN / INVOCATION   the knobs that shape prompt size, and every agent's model
    AGENT HANDOVER     session state at the boundary: every key sized, and the
                       content of whatever changed since the last handover
    LLM CALL           the exact input, split into system / tools / contents,
                       with an estimate for each part
    LLM RESPONSE       what the provider actually charged, next to the estimate
    TOOL CALL/RESULT   where the next prompt's bulk comes from
    AGENT DONE         state after the turn, and what the turn changed

Estimates reuse `estimate_tokens` from `rate_limit`, deliberately: a second
estimator would produce numbers that look comparable to the limiter's budget and
are not.

Two things this log makes visible that nothing else did:

* **Tool declarations are unpaced.** `rate_limit.estimate_message_tokens` walks
  `messages` only -- the word `tools` does not appear in that module -- so the
  function-declaration payload is charged to nobody. Here it is a line item.
* **State is prompt.** Agent instructions interpolate state keys directly
  (`{status}`, `{next_batch_source}`, `{case_facts}`, `{converted_inventory}`,
  `{semantic_match}`, `{source_index}`), so a key that grows costs that many
  tokens on every turn of that agent. The handover table prices each key.

Known gaps, stated where they will be read:

* ADK's compaction summarizer calls `generate_content_async` directly rather
  than through `base_llm_flow`, so its model calls never reach a plugin hook and
  do not appear here. Everything an agent does is covered.
* The `model` on an `LLM CALL` record is the model ADK asked for, not
  necessarily the one that answered. `rate_limit.acall_with_rate_limit` absorbs
  a 429 and retries against `MODEL_FALLBACKS` beneath these hooks, so a call
  logged as opus-4-7 can be served by sonnet-4-6 -- and the `usage_metadata` in
  the matching `LLM RESPONSE` is then the fallback's. A long gap between the two
  records' timestamps is the only visible sign. For the same reason a throttled
  call produces no `LLM ERROR` record: the 429 never surfaces as an exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool

from .rate_limit import estimate_tokens

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("LLM_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}
LOG_FILE_OVERRIDE = os.environ.get("LLM_LOG_FILE", "").strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning("%s is not an integer; using %d", name, default)
        return default


#: Two bounds on log SIZE, both off by default so the log is what it has always
#: been. They exist because this pipeline is bigger than the one this logger was
#: written against -- 16 conversion iterations plus 3 semantic ones, sharing one
#: session, with nothing compacting inside a single invocation -- which projects
#: to tens of megabytes a run.
#:
#: Neither one touches accounting. `_sized` and `_fingerprint` run on the FULL
#: text before anything is elided, so every char count, token estimate, `re-sent`
#: percentage and `fp=` back-reference stays exactly as it would have been. Only
#: the bodies shrink.
#:
#: `LLM_LOG_MAX_BLOCK_CHARS`  0 = unbounded. Otherwise keep this many characters
#:                            of head and of tail in every verbatim block.
#: `LLM_LOG_REPEAT_BODIES`    0 = print the header of a payload already sent
#:                            byte-identically in an earlier call, but not its
#:                            body again. The body is in the file at the call the
#:                            header points to, findable by its fingerprint --
#:                            which is what fingerprints are for. Measured
#:                            re-send rate on the converter is 61-86%, so this is
#:                            the setting that makes a full run readable.
MAX_BLOCK_CHARS = _int_env("LLM_LOG_MAX_BLOCK_CHARS", 0)
REPEAT_BODIES = os.environ.get("LLM_LOG_REPEAT_BODIES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

WIDTH = 88
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

#: Environment knobs that change how big a prompt gets. Recorded in the header so
#: a log can be read months later without guessing what it was configured with.
SIZE_KNOBS = (
    "ADK_COMPACTION_INTERVAL",
    "ADK_COMPACTION_OVERLAP",
    "RATE_LIMIT_DEFAULT_MAX_TOKENS",
    "RATE_LIMIT_BURST_FRACTION",
    "RATE_LIMIT_FALLBACKS",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _render(value: Any) -> str:
    """Turn a state value or tool payload into the text that will be logged.

    Strings pass through unquoted -- most of the big ones are source code, and
    `json.dumps` would render a 90KB function body as one unreadable line of
    `\\n` escapes.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


def _json(obj: Any) -> str:
    """Serialise a pydantic model (or anything else) for logging."""
    dump = getattr(obj, "model_dump_json", None)
    if dump is not None:
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    return _render(obj)


def _sized(text: str) -> tuple[int, int]:
    """`(chars, estimated tokens)` for `text`."""
    return len(text), estimate_tokens(text)


#: Below this, a repeated payload is noise rather than a token problem -- role
#: preambles and one-line tool args recur constantly and cost nothing.
REPEAT_MIN_CHARS = 200

#: Ceiling on the fingerprint table. Reached only by a very long run; clearing it
#: costs some "REPEAT of #n" back-references, never correctness.
MAX_FINGERPRINTS = 20_000


def _fingerprint(text: str) -> str:
    """A short content hash, so identical payloads are greppable across calls.

    The whole point of the log is finding what gets re-sent unchanged. Comparing
    megabytes of prompt by eye does not scale; comparing eight hex characters
    does.
    """
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]


def _model_name(agent: Any) -> str:
    """The endpoint an agent talks to, whether `model` is a string or a LiteLlm."""
    model = getattr(agent, "model", None)
    if model is None:
        return ""
    return str(getattr(model, "model", model))


class _Writer:
    """The log file, created on first use.

    Lazy on purpose: importing this module must not litter `outputs/` for a
    process that never runs an agent. One lock, and a flush per record, so
    `tail -f` shows a run as it happens and the two callback paths cannot
    interleave a half-written block.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._failed = False
        self.path: Path | None = None

    def _open(self):
        if self._handle is not None:
            return self._handle
        if LOG_FILE_OVERRIDE:
            path = Path(LOG_FILE_OVERRIDE)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / "llm_logs" / f"run_{stamp}_{os.getpid()}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self.path = path
        logger.info("LLM call log: %s", path)
        return self._handle

    def write(self, text: str) -> None:
        """Append `text`. Never raises -- and gives up permanently on failure.

        Instrumentation that can end a run is worse than no instrumentation, so
        the first failure disables the writer and says so once.
        """
        if self._failed:
            return
        try:
            with self._lock:
                handle = self._open()
                handle.write(text)
                handle.write("\n")
                handle.flush()
        except Exception as exc:
            self._failed = True
            logger.warning("LLM call logging disabled after a write failure: %s", exc)


WRITER = _Writer()


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _banner(title: str, char: str = "=") -> list[str]:
    return ["", _rule(char), f"[{_now()}]  {title}", _rule(char)]


def _section(title: str, meta: str = "") -> str:
    """A `---- TITLE  (meta) --------` header, ruled out to the full width."""
    head = f"  ---- {title}" + (f"  ({meta}) " if meta else " ")
    return head + "-" * max(2, WIDTH - len(head))


def _block(text: str) -> list[str]:
    """Verbatim content, every line prefixed so it cannot be mistaken for structure.

    The one choke point every dump passes through -- state values, system
    instruction, tool declarations, content parts, tool args, tool results -- so
    `LLM_LOG_MAX_BLOCK_CHARS` is enforced here and nowhere else.
    """
    if not text:
        return ["  | <empty>"]
    if MAX_BLOCK_CHARS and len(text) > 2 * MAX_BLOCK_CHARS:
        elided = len(text) - 2 * MAX_BLOCK_CHARS
        text = (
            f"{text[:MAX_BLOCK_CHARS]}\n"
            f"... <elided {_count(elided)} chars; "
            f"raise LLM_LOG_MAX_BLOCK_CHARS or set it to 0 for the whole thing> ...\n"
            f"{text[-MAX_BLOCK_CHARS:]}"
        )
    return [f"  | {line}" for line in text.splitlines() or [""]]


def _body(text: str, is_repeat: bool) -> list[str]:
    """`_block`, unless this payload is a byte-identical re-send being suppressed.

    The header the caller has already emitted carries the fingerprint and the
    call number to look it up under, so nothing becomes unfindable -- it just
    stops being duplicated.
    """
    if is_repeat and not REPEAT_BODIES:
        return ["  | <body suppressed: identical to the call named above."
                " LLM_LOG_REPEAT_BODIES=1 to print it>"]
    return _block(text)


def _field(label: str, value: Any, width: int = 13) -> str:
    return f"  {label:<{width}}: {value}"


def _count(n: int) -> str:
    return f"{n:,}"


def _size_table(entries: list[tuple[str, str, int, int]]) -> list[str]:
    """Rows of `[ n] name  type  chars  tokens`, biggest first."""
    lines = []
    for i, (name, kind, chars, tokens) in enumerate(entries, start=1):
        lines.append(
            f"    [{i:>2}] {name:<34.34} {kind:<8.8} "
            f"{_count(chars):>11} chars  ~{_count(tokens):>9} tok"
        )
    return lines


def _state_of(context: Any) -> dict[str, Any]:
    """Read a context's session state as a plain dict.

    ADK's `State` is dict-LIKE but not a full mapping -- the converter package
    already documents `pop` raising AttributeError on it -- so this goes through
    `to_dict()` and only falls back to `dict()`.
    """
    state = getattr(context, "state", None)
    if state is None:
        return {}
    to_dict = getattr(state, "to_dict", None)
    if to_dict is not None:
        try:
            return dict(to_dict())
        except Exception:
            pass
    try:
        return dict(state)
    except Exception:
        return {}


def _agent_tree(agent: Any, depth: int = 0) -> list[str]:
    """Every agent under `agent`, with the settings that drive its prompt size."""
    indent = "    " + "  " * depth
    model = _model_name(agent) or "-"
    mode = getattr(agent, "mode", None) or "-"
    contents = getattr(agent, "include_contents", None) or "default"
    tools = getattr(agent, "tools", None) or ()
    line = (
        f"{indent}{getattr(agent, 'name', '?')}  [{type(agent).__name__}]  "
        f"model={model}  mode={mode}  include_contents={contents}  tools={len(tools)}"
    )
    lines = [line]
    for child in getattr(agent, "sub_agents", None) or ():
        lines.extend(_agent_tree(child, depth + 1))
    return lines


class LlmCallLogPlugin(BasePlugin):
    """Records prompts, state and token accounting to `WRITER`.

    Registered on `App.plugins` rather than on the agents themselves: four agents
    in this tree already own `before_agent_callback`/`after_agent_callback` for
    real pipeline logic, and plugin callbacks run alongside those instead of
    replacing them.

    Every hook returns None unconditionally. A non-None return from a plugin hook
    short-circuits the rest of the callback chain -- including the agents' own
    callbacks -- so a logger that returned a value would silently change what the
    pipeline does. Concretely, a non-None `after_agent_callback` would suppress
    `check_fact_status` and `check_semantic_match`, which is where
    `actions.escalate = True` is set: both loops would then run to
    `max_iterations` every time instead of stopping when they match.

    One ordering consequence, so the log is not misread: plugin hooks run BEFORE
    an agent's own callbacks, so every state snapshot here is the state as the
    agent was entered or left, without what that agent's own before/after
    callback is about to write. Seeded values (`status`, `next_batch_source`) and
    verdicts (`fact_check_passed`, `semantic_match`) therefore show up as
    `[CHANGED]` at the NEXT snapshot. The fully interpolated instruction in the
    following `LLM CALL` block is the authoritative view of what was sent.
    """

    def __init__(self, name: str = "llm_call_logger") -> None:
        super().__init__(name=name)
        self._calls = 0
        self._header_written = False
        #: (invocation, agent) -> (call number, estimated prompt tokens), so a
        #: response can be printed against the estimate it is answering.
        self._pending: dict[tuple[str, str], tuple[int, int]] = {}
        #: invocation -> {state key: rendered text} at the previous handover.
        #: What makes "changed since the last handover" answerable.
        self._last_state: dict[str, dict[str, str]] = {}
        self._last_agent: dict[str, str] = {}
        #: content fingerprint -> where it was first sent. Turns "is the same
        #: data being appended over and over?" into a lookup instead of a diff.
        self._seen: dict[str, tuple[int, str]] = {}

    def _repeat_tag(self, text: str, number: int, label: str) -> tuple[str, bool]:
        """`(marker, is_repeat)` for one payload.

        Records `text` against `label` the first time it is seen, and thereafter
        points back at that first call. `is_repeat` is what the per-call re-sent
        tally is built from.
        """
        if len(text) < REPEAT_MIN_CHARS:
            return "", False
        fingerprint = _fingerprint(text)
        prior = self._seen.get(fingerprint)
        if prior is not None:
            return f" fp={fingerprint} REPEAT of call #{prior[0]} {prior[1]}", True
        if len(self._seen) >= MAX_FINGERPRINTS:
            self._seen.clear()
        self._seen[fingerprint] = (number, label)
        return f" fp={fingerprint} FIRST-SENT", False

    # ---------------------------------------------------------------- helpers

    def _emit(self, lines: list[str]) -> None:
        WRITER.write("\n".join(lines))

    def _state_section(
        self,
        invocation: str,
        agent_name: str,
        state: dict[str, Any],
        *,
        during_turn: bool = False,
    ) -> list[str]:
        """The handover record: every key priced, and every key's content.

        Every key is dumped in full, tagged `[NEW]` / `[CHANGED]` / `[unchanged]`
        against the previous handover. Printing only what moved was the obvious
        saving and not worth taking: the pipeline's before-agent callbacks
        rewrite `status` and `next_batch_source` on every turn, so almost nothing
        qualified as unchanged, and a reader looking at one handover would have
        had to scroll back through earlier blocks to reconstruct the rest.
        """
        rendered = {key: _render(value) for key, value in state.items()}
        previous = self._last_state.get(invocation, {})
        first = invocation not in self._last_state

        entries = sorted(
            (
                (key, type(state[key]).__name__, *_sized(text))
                for key, text in rendered.items()
            ),
            key=lambda row: row[2],
            reverse=True,
        )
        total_chars = sum(row[2] for row in entries)
        total_tokens = sum(row[3] for row in entries)

        lines = [
            _section(
                "STATE",
                f"{len(entries)} keys | {_count(total_chars)} chars | "
                f"~{_count(total_tokens)} est tokens",
            )
        ]
        if not entries:
            lines.append("    <state is empty>")
        lines.extend(_size_table(entries))

        added = [k for k in rendered if k not in previous]
        changed = [k for k in rendered if k in previous and previous[k] != rendered[k]]
        removed = [k for k in previous if k not in rendered]
        if not first:
            delta = total_chars - sum(len(v) for v in previous.values())
            since = (
                "changed during this turn"
                if during_turn
                else f"changed since {self._last_agent.get(invocation, '?')}"
            )
            lines.append(
                f"    {since}: "
                f"added={added or '-'} changed={changed or '-'} removed={removed or '-'} "
                f"({delta:+,} chars)"
            )

        if entries:
            lines.append(_section("STATE CONTENT"))
            for key, kind, chars, tokens in entries:
                mark = (
                    "  [NEW]" if key in added
                    else "  [CHANGED]" if key in changed
                    else "  [unchanged]"
                )
                head = (
                    f"  +-- {key}{mark}  ({kind}, {_count(chars)} chars, "
                    f"~{_count(tokens)} tok) "
                )
                lines.append(head + "-" * max(2, WIDTH - len(head)))
                lines.extend(_block(rendered[key]))
                lines.append("  +" + "-" * (WIDTH - 3))

        self._last_state[invocation] = rendered
        self._last_agent[invocation] = agent_name
        return lines

    # ------------------------------------------------------------------ hooks

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        try:
            root = getattr(invocation_context, "agent", None)
            session = getattr(invocation_context, "session", None)
            lines: list[str] = []
            if not self._header_written:
                self._header_written = True
                lines.extend(_banner("LLM CALL & STATE AUDIT LOG"))
                lines.append(_field("started", datetime.now(timezone.utc).isoformat(), 20))
                lines.append(_field("pid", os.getpid(), 20))
                lines.append(_section("PROMPT-SIZE KNOBS"))
                for knob in SIZE_KNOBS:
                    lines.append(_field(knob, os.environ.get(knob, "<unset>"), 44))
                # This log's own settings, recorded because a reader must never
                # have to guess whether a body is short or merely truncated.
                lines.append(
                    _field(
                        "LLM_LOG_MAX_BLOCK_CHARS",
                        f"{_count(MAX_BLOCK_CHARS)} head+tail chars per block"
                        if MAX_BLOCK_CHARS
                        else "0  (blocks printed in full)",
                        44,
                    )
                )
                lines.append(
                    _field(
                        "LLM_LOG_REPEAT_BODIES",
                        "1  (re-sent payloads printed again)"
                        if REPEAT_BODIES
                        else "0  (re-sent payloads: header only, body at the call it names)",
                        44,
                    )
                )
                for key, value in sorted(os.environ.items()):
                    if key.startswith(("DATABRICKS_ITPM_", "DATABRICKS_OTPM_", "DATABRICKS_RPM_")):
                        lines.append(_field(key, value, 44))

            lines.extend(_banner(f"INVOCATION START  |  {getattr(session, 'app_name', '?')}"))
            lines.append(_field("invocation", getattr(invocation_context, "invocation_id", "?")))
            lines.append(_field("session", getattr(session, "id", "?")))
            if root is not None:
                lines.append(_section("AGENT TREE"))
                lines.extend(_agent_tree(root))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: before_run failed", exc_info=True)
        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ):
        try:
            invocation = callback_context.invocation_id
            name = getattr(agent, "name", "?")
            model = _model_name(agent)
            lines = _banner(f"AGENT HANDOVER  ->  {name}")
            lines.append(_field("invocation", invocation))
            lines.append(_field("agent type", type(agent).__name__))
            if model:
                lines.append(
                    _field(
                        "model",
                        f"{model}   (include_contents="
                        f"{getattr(agent, 'include_contents', None) or 'default'})",
                    )
                )
            branch = getattr(
                getattr(callback_context, "_invocation_context", None), "branch", None
            )
            if branch:
                lines.append(_field("branch", branch))
            lines.extend(
                self._state_section(invocation, name, _state_of(callback_context))
            )
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: before_agent failed", exc_info=True)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ):
        try:
            invocation = callback_context.invocation_id
            name = getattr(agent, "name", "?")
            lines = _banner(f"AGENT DONE  <-  {name}", "-")
            output_key = getattr(agent, "output_key", None)
            state = _state_of(callback_context)
            if output_key:
                chars, tokens = _sized(_render(state.get(output_key, "")))
                lines.append(
                    _field(
                        "output_key",
                        f"{output_key}  ({_count(chars)} chars, ~{_count(tokens)} tok)",
                    )
                )
            lines.extend(self._state_section(invocation, name, state, during_turn=True))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: after_agent failed", exc_info=True)
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ):
        try:
            self._calls += 1
            number = self._calls
            agent = callback_context.agent_name
            config = getattr(llm_request, "config", None)

            system_text = ""
            instruction = getattr(config, "system_instruction", None)
            if isinstance(instruction, str):
                system_text = instruction
            elif instruction is not None:
                system_text = "\n".join(
                    part.text for part in (getattr(instruction, "parts", None) or ())
                    if getattr(part, "text", None)
                )
            system_chars, system_tokens = _sized(system_text)

            declarations: list[tuple[str, str, int, int]] = []
            declaration_text: list[str] = []
            for tool in getattr(config, "tools", None) or ():
                for decl in getattr(tool, "function_declarations", None) or ():
                    text = _json(decl)
                    declaration_text.append(text)
                    chars, tokens = _sized(text)
                    declarations.append((getattr(decl, "name", "?"), "decl", chars, tokens))
            tool_tokens = sum(row[3] for row in declarations)
            tool_chars = sum(row[2] for row in declarations)

            system_tag, system_repeat = self._repeat_tag(system_text, number, "system")
            tools_joined = "\n".join(declaration_text)
            tools_tag, tools_repeat = self._repeat_tag(tools_joined, number, "tools")

            contents = list(getattr(llm_request, "contents", None) or [])
            content_lines: list[str] = []
            content_tokens = 0
            repeated_content_tokens = 0
            for index, content in enumerate(contents):
                role = getattr(content, "role", "?") or "?"
                for part in getattr(content, "parts", None) or ():
                    kind, label, text = _describe_part(part)
                    chars, tokens = _sized(text)
                    content_tokens += tokens
                    tag, repeat = self._repeat_tag(text, number, f"msg {index}")
                    if repeat:
                        repeated_content_tokens += tokens
                    head = (
                        f"  +-- [msg {index}] {role} | {kind}{label}  "
                        f"({_count(chars)} chars, ~{_count(tokens)} tok){tag} "
                    )
                    content_lines.append(head + "-" * max(2, WIDTH - len(head)))
                    content_lines.extend(_body(text, repeat))

            total = system_tokens + tool_tokens + content_tokens
            resent = (
                repeated_content_tokens
                + (system_tokens if system_repeat else 0)
                + (tool_tokens if tools_repeat else 0)
            )
            max_output = getattr(config, "max_output_tokens", None)
            self._pending[(callback_context.invocation_id, agent)] = (number, total)

            lines = _banner(f"LLM CALL #{number}  |  {agent}")
            lines.append(_field("model", getattr(llm_request, "model", None) or "?"))
            lines.append(
                _field(
                    "est. input",
                    f"~{_count(total)} tok   = system {_count(system_tokens)} "
                    f"+ tools {_count(tool_tokens)} + contents {_count(content_tokens)}",
                )
            )
            lines.append(_field("max output", f"{_count(max_output)} tok" if max_output else "-"))
            # The line to read when the question is "why is this so big?". A high
            # percentage means the prompt is mostly payload an earlier call
            # already carried -- history and re-sent tool results -- not new work.
            lines.append(
                _field(
                    "re-sent",
                    f"~{_count(resent)} tok ({resent / total * 100:.0f}%) byte-identical "
                    f"to earlier calls   [system={'repeat' if system_repeat else 'new'} "
                    f"tools={'repeat' if tools_repeat else 'new'} "
                    f"contents={_count(repeated_content_tokens)}/{_count(content_tokens)}]"
                    if total
                    else "n/a",
                )
            )

            lines.append(
                _section(
                    "SYSTEM INSTRUCTION",
                    f"{_count(system_chars)} chars | ~{_count(system_tokens)} est tok"
                    f"{system_tag}",
                )
            )
            lines.extend(_body(system_text, system_repeat))

            lines.append(
                _section(
                    "TOOL DECLARATIONS",
                    f"{len(declarations)} tools | {_count(tool_chars)} chars | "
                    f"~{_count(tool_tokens)} est tok{tools_tag}",
                )
            )
            if declarations:
                lines.extend(_size_table(sorted(declarations, key=lambda r: r[2], reverse=True)))
                lines.extend(_body(tools_joined, tools_repeat))
            else:
                lines.append("    <no tools declared>")

            lines.append(
                _section(
                    "CONTENTS",
                    f"{len(contents)} messages | ~{_count(content_tokens)} est tok | "
                    f"~{_count(repeated_content_tokens)} tok re-sent",
                )
            )
            lines.extend(content_lines or ["    <no conversation history sent>"])
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: before_model failed", exc_info=True)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ):
        try:
            # A streamed call fires this hook per chunk, so the partials are
            # dropped -- logging them would bury the run in hundreds of
            # near-identical blocks.
            if getattr(llm_response, "partial", None) is True:
                return None
            agent = callback_context.agent_name
            key = (callback_context.invocation_id, agent)
            # ...and at the END of a stream LiteLlm can yield TWO non-partial
            # aggregates: the text one, then the tool-call one. Only the first
            # carries usage_metadata (it is consumed and cleared), so reporting
            # both would print a second block claiming the call was estimated at
            # zero and its usage unreported. One call gets one response block:
            # a missing pending entry means this call was already reported.
            if key not in self._pending:
                return None
            number, estimated = self._pending.pop(key)
            usage = getattr(llm_response, "usage_metadata", None)
            lines = _banner(f"LLM RESPONSE #{number}  |  {agent}", "-")
            if usage is None:
                lines.append(_field("usage", "<not reported>"))
            else:
                actual = getattr(usage, "prompt_token_count", None) or 0
                delta = actual - estimated
                drift = f"{delta:+,} / {delta / actual * 100:+.1f}%" if actual else "n/a"
                lines.append(
                    _field(
                        "actual input",
                        f"{_count(actual)} tok   (estimated {_count(estimated)} "
                        f"-> off by {drift})",
                    )
                )
                lines.append(
                    _field(
                        "cached input",
                        f"{_count(getattr(usage, 'cached_content_token_count', None) or 0)} tok",
                    )
                )
                lines.append(
                    _field(
                        "output",
                        f"{_count(getattr(usage, 'candidates_token_count', None) or 0)} tok"
                        f"   (thoughts "
                        f"{_count(getattr(usage, 'thoughts_token_count', None) or 0)})",
                    )
                )
                lines.append(
                    _field(
                        "total",
                        f"{_count(getattr(usage, 'total_token_count', None) or 0)} tok",
                    )
                )
            finish = getattr(llm_response, "finish_reason", None)
            lines.append(_field("finish", getattr(finish, "name", None) or finish or "-"))
            error = getattr(llm_response, "error_message", None)
            if error:
                lines.append(_field("error", error))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: after_model failed", exc_info=True)
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ):
        try:
            agent = callback_context.agent_name
            number, estimated = self._pending.pop(
                (callback_context.invocation_id, agent), (self._calls, None)
            )
            lines = _banner(f"LLM ERROR #{number}  |  {agent}", "-")
            lines.append(
                _field("estimated", f"{_count(estimated)} tok" if estimated else "unknown")
            )
            lines.append(_field("error", f"{type(error).__name__}: {error}"))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: on_model_error failed", exc_info=True)
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: Any
    ):
        try:
            text = _render(tool_args)
            chars, tokens = _sized(text)
            lines = _banner(
                f"TOOL CALL  {getattr(tool, 'name', '?')}  |  "
                f"{getattr(tool_context, 'agent_name', '?')}",
                "-",
            )
            lines.append(_field("args", f"{_count(chars)} chars, ~{_count(tokens)} tok"))
            lines.extend(_block(text))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: before_tool failed", exc_info=True)
        return None

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: Any, result: Any
    ):
        try:
            text = _render(result)
            chars, tokens = _sized(text)
            lines = _banner(
                f"TOOL RESULT  {getattr(tool, 'name', '?')}  |  "
                f"{getattr(tool_context, 'agent_name', '?')}",
                "-",
            )
            # Worth spelling out: this result becomes a `function_response` part
            # in EVERY later prompt of this agent's turn, so its size is paid
            # once per remaining model call, not once.
            lines.append(
                _field(
                    "result",
                    f"{_count(chars)} chars, ~{_count(tokens)} tok "
                    "(re-sent in every later prompt this turn)",
                )
            )
            lines.extend(_block(text))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: after_tool failed", exc_info=True)
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        try:
            invocation = getattr(invocation_context, "invocation_id", "")
            self._last_state.pop(invocation, None)
            self._last_agent.pop(invocation, None)
            lines = _banner(f"INVOCATION END  |  {invocation}", "-")
            lines.append(_field("model calls", self._calls))
            if WRITER.path is not None:
                lines.append(_field("log file", WRITER.path))
            self._emit(lines)
        except Exception:
            logger.debug("llm_call_logger: after_run failed", exc_info=True)
        return None


def _describe_part(part: Any) -> tuple[str, str, str]:
    """`(kind, label, text)` for one content part.

    Duck-typed rather than isinstance-checked: `types.Part` carries every field
    as an Optional, so which one is set IS the discriminator.
    """
    call = getattr(part, "function_call", None)
    if call is not None:
        return "function_call", f" -> {getattr(call, 'name', '?')}", _render(
            getattr(call, "args", None)
        )
    response = getattr(part, "function_response", None)
    if response is not None:
        return "function_response", f" <- {getattr(response, 'name', '?')}", _render(
            getattr(response, "response", None)
        )
    inline = getattr(part, "inline_data", None)
    if inline is not None:
        data = getattr(inline, "data", b"") or b""
        return "inline_data", f" [{getattr(inline, 'mime_type', '?')}]", f"<{len(data)} bytes>"
    text = getattr(part, "text", None)
    if text is not None:
        kind = "thought" if getattr(part, "thought", None) else "text"
        return kind, "", text
    return "other", "", _json(part)


def plugins() -> list[BasePlugin]:
    """The plugin list for an `App`, empty when `LLM_LOG=0`.

    A factory rather than a shared instance: two apps are separate runs with
    separate call counters, and one instance shared between them would number
    their calls in a single confusing sequence.

    What a factory does NOT separate is the file. `WRITER` is a module-level
    singleton, so if a second `App` in this process ever registers these plugins
    too -- `subagents/parity_app` would, once its `__init__ .py` filename is
    fixed -- both apps append to the same log, and a second header banner appears
    partway down it. Per-app files would mean moving `_Writer` onto the instance.
    """
    return [LlmCallLogPlugin()] if ENABLED else []
