import ast
import base64
import json
import os
import pathlib
import re
import time
import uuid
import requests
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents import Agent, ParallelAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from dotenv import load_dotenv
from typing import Any, Callable, Dict, List, Optional
load_dotenv()


OUTPUTS_DIR = pathlib.Path(__file__).parents[3] / "outputs"
_FILE_HEADER = '"""Auto-assembled PySpark conversion (built incrementally, one batch\nof functions per loop iteration)."""\n'
_MAX_LOG_CHARS = 3000

_SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "py2snow-skill"


def _tail(text: "str | None", limit: int = _MAX_LOG_CHARS) -> str:
    """Keep only the last `limit` chars of tool output so verbose PySpark logs
    don't accumulate in the LLM context window."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return "…[truncated earlier output]…\n" + text[-limit:]


def _error_summary(stderr: "str | None", file_path: str, limit: int = 400) -> str:
    """Extract an actionable one-liner from a Python traceback: which function in
    the converted file failed (line + name) and the exception message. Lets the
    converter fix the exact function instead of scanning raw Spark log spam."""
    if not stderr:
        return ""
    fname = os.path.basename(file_path or "")
    lines = stderr.splitlines()
    exc = ""
    for l in reversed(lines):
        s = l.strip()
        if s and ("Error" in s or "Exception" in s) and ":" in s \
                and not s.startswith(("WARN", "File ")):
            exc = s
            break
    where = ""
    for l in lines:
        s = l.strip()
        if fname and fname in s and ", in " in s:  # File ".../x.py", line N, in func
            where = " " + s.split(fname, 1)[1].strip().lstrip('",').strip()
    summary = (f"Crashed at{where}. {exc}").strip()
    return summary[:limit]

UNAVAILABLE_ON_DATABRICKS = {
    "xlwings": "pandas + openpyxl",
    "win32com": "pandas + openpyxl",
    "pywin32": "pandas + openpyxl",
    "xlsxwriter": "pandas + openpyxl",
    "pyautogui": "a headless equivalent",
}

#: Modules a Databricks cluster does not ship but CAN be installed, mapped to the
#: pip requirement that provides each. The opposite of UNAVAILABLE_ON_DATABRICKS:
#: those can never work on a cluster and the fix is to rewrite the code; these
#: work fine once installed and the fix is one line.
#:
#: Keys are matched longest-first against dotted module names, so
#: `databricks.sql` resolves to the connector while `databricks.sdk` — which
#: clusters DO ship — is left alone.
PIP_INSTALLABLE_ON_DATABRICKS = {
    "databricks.sql": "databricks-sql-connector",
    # NOT keyed on bare "databricks". `_requirement_for` walks up the dotted
    # name, so that key would also claim `databricks.sdk` — which clusters DO
    # ship — and every SDK import would trigger a pointless install.
    # Mandated by the conversion conventions for Excel I/O, and absent from
    # serverless environments — which is exactly the crash this handles.
    "openpyxl": "openpyxl",
}

_MODULE_NOT_FOUND = re.compile(
    r"(?:ModuleNotFoundError|ImportError).*?['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]"
)


def _unavailable_module(text: str) -> str:
    """Return the unavailable module named in `text`, or "".

    Matches on the import error the cluster raises, not on the source, so it
    fires for a transitive import too.
    """
    if not text:
        return ""
    for match in _MODULE_NOT_FOUND.finditer(text):
        root = match.group(1).split(".")[0]
        if root in UNAVAILABLE_ON_DATABRICKS:
            return root
    # Fall back to a bare mention: some Databricks errors omit the quotes.
    lowered = text.lower()
    for name in UNAVAILABLE_ON_DATABRICKS:
        if name in lowered and ("modulenotfound" in lowered or "no module named" in lowered):
            return name
    return ""


def _imported_modules(code: str) -> set[str]:
    """Every dotted module name `code` imports, plus each of its parents.

    `from databricks import sql` records "databricks.sql" AND "databricks", so a
    requirement keyed on either spelling is found. Parsed rather than grepped:
    the word "openpyxl" in a docstring must not trigger an install.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()          # the run itself will report the syntax error

    names: set[str] = set()

    def record(dotted: str) -> None:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            names.add(".".join(parts[:i]))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:        # relative import; never a pip package
                continue
            module = node.module or ""
            if module:
                record(module)
                for alias in node.names:
                    record(f"{module}.{alias.name}")
    return names


def _requirement_for(module: str) -> str:
    """The pip requirement providing `module`, or "" — longest match wins."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        found = PIP_INSTALLABLE_ON_DATABRICKS.get(".".join(parts[:i]))
        if found:
            return found
    return ""


def _pip_requirements(code: str, remembered: object = ()) -> list[str]:
    """Packages to install before running `code`.

    `remembered` carries requirements a PREVIOUS run proved necessary — a module
    imported transitively by a dependency is invisible to the scan below, so it
    is learned from the failure once and applied thereafter.
    """
    needed = {
        req
        for module in _imported_modules(code)
        if (req := _requirement_for(module))
    }
    if isinstance(remembered, (list, tuple, set)):
        needed.update(str(r) for r in remembered if r)
    return sorted(needed)


def _missing_requirement(text: str) -> str:
    """The pip requirement for an installable module missing in `text`, or "".

    Distinct from `_unavailable_module`, which names modules no cluster can ever
    provide. This one names the modules a single `%pip install` fixes.
    """
    if not text:
        return ""
    for match in _MODULE_NOT_FOUND.finditer(text):
        found = _requirement_for(match.group(1))
        if found:
            return found
    return ""


#: Databricks splits a notebook into cells on this marker, and a `%pip` magic
#: only takes effect for LATER cells — so the install, the interpreter restart
#: and the converted code have to be three separate ones.
_CELL = "\n\n# COMMAND ----------\n\n"


def _notebook_with_installs(code: str, requirements: list[str]) -> str:
    """Prefix `code` with a `%pip install` cell, or return it unchanged.

    `dbutils.library.restartPython()` is not optional: without it the packages
    land on disk but the already-running interpreter keeps the import table it
    started with, so the very next import still raises ModuleNotFoundError.

    Line numbers in a traceback stay honest because the converted code keeps its
    own cell and is reported relative to it.
    """
    if not requirements:
        return code
    return _CELL.join((
        f"# Databricks notebook source\n%pip install {' '.join(requirements)} --quiet",
        "dbutils.library.restartPython()",
        code,
    ))


def _as_dict(value):
    """Tolerate an inventory that is already a dict, or is raw JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}

AST_INVENTORY = OUTPUTS_DIR / "ast_inventory.json"
MAX_CONSTANT_VALUE_CHARS = 500

EXCEL_IO_NAMES = {"read_excel", "to_excel", "ExcelWriter", "ExcelFile"}


def _inventory() -> dict:
    """Load the parsed AST inventory from disk.

    The parser deliberately does NOT put this in state: for a script of a few
    hundred functions it is tens of kilobytes of JSON, and state is echoed into
    every prompt of every agent in the session.

    Returns an empty dict when the parser has not run or the file is
    unreadable — callers already treat an empty inventory as "nothing to do",
    so raising would turn an ordering problem into a crash.
    """
    try:
        return _as_dict(AST_INVENTORY.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _source_function_names() -> list[str]:
    """Every function name in the parsed SOURCE script (order preserved)."""
    apc = _inventory()
    return [
        f.get("name")
        for f in (apc.get("functions") or [])
        if isinstance(f, dict) and f.get("name")
    ]


def _canonical_output_path(state) -> pathlib.Path:
    """The ONE stable file every batch is appended to.

    Derived once from the source script name and cached in state so retries and
    the downstream fixer agents all target the same file (prevents the
    `_spark.py` vs `_spark_complete.py` divergence).
    """
    existing = state.get("converted_pyspark_file_path")
    if existing:
        return pathlib.Path(existing)
    stem = "converted"
    apc = _inventory()
    script_path = (apc.get("metadata") or {}).get("script_path")
    if script_path:
        stem = pathlib.Path(script_path).stem
    # Must be importable: the semantic runner does
    # `from <stem>_spark import ...`.
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower() or "converted"
    if stem[0].isdigit():
        stem = f"s_{stem}"
    p = OUTPUTS_DIR / f"{stem}_spark.py"
    state["converted_pyspark_file_path"] = str(p)
    return p


def _module_const_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _pandas_violations(src: str) -> list[str]:
    """Find pandas idioms in submitted code that must be native Spark instead.

    This is the rule the conventions repeated hardest, and repetition is an
    expensive way to enforce anything — it cost ~10k tokens of prompt on every
    turn and the model could still ignore it. Checking here is deterministic and
    unskippable, so the prompt only has to STATE the rule once.

    Deliberately high-confidence only. numpy is NOT flagged: data-generation
    functions are required to build rows in plain Python (with seeded numpy /
    random) and hand them to spark.createDataFrame — banning it would reject
    correct conversions.

    Also flags imports that cannot exist on a Databricks cluster
    (UNAVAILABLE_ON_DATABRICKS). Catching those here rather than at run time
    saves a full Databricks round-trip on a failure that is guaranteed and
    unfixable by retrying.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []          # the caller reports the syntax error itself

    found: list[str] = []
    pandas_imports: list[str] = []
    non_excel_pandas = False
    for node in ast.walk(tree):
        # import pandas / from pandas import ...
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root == "pandas":
                    pandas_imports.append(
                        f"line {node.lineno}: `import pandas` — use the Spark DataFrame API"
                    )
                elif root in UNAVAILABLE_ON_DATABRICKS:
                    found.append(
                        f"line {node.lineno}: `import {root}` — not available on "
                        f"Databricks; use {UNAVAILABLE_ON_DATABRICKS[root]}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root == "pandas":
                pandas_imports.append(
                    f"line {node.lineno}: `from pandas import …` — use the Spark DataFrame API"
                )
            elif root in UNAVAILABLE_ON_DATABRICKS:
                found.append(
                    f"line {node.lineno}: `from {root} import …` — not available on "
                    f"Databricks; use {UNAVAILABLE_ON_DATABRICKS[root]}"
                )

        elif isinstance(node, ast.Attribute):
            # pd.<anything>, except the Excel I/O that hard rule 5 REQUIRES
            if isinstance(node.value, ast.Name) and node.value.id in ("pd", "pandas"):
                if node.attr not in EXCEL_IO_NAMES:
                    non_excel_pandas = True
                    found.append(f"line {node.lineno}: `{node.value.id}.{node.attr}` — pandas call, convert to Spark")
            # .iloc / .loc positional indexing
            elif node.attr in ("iloc", "loc"):
                non_excel_pandas = True
                found.append(f"line {node.lineno}: `.{node.attr}` — no positional indexing in Spark; use filter/select")

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in EXCEL_IO_NAMES:
                continue
            non_excel_pandas = True
            if name == "merge":
                found.append(f"line {node.lineno}: `.merge(...)` — use `.join(other, on=…, how=…)`")
            elif name == "toPandas":
                found.append(f"line {node.lineno}: `.toPandas()` — collapses the frame to the driver")
            elif name == "rename" and any(k.arg == "columns" for k in node.keywords):
                found.append(f"line {node.lineno}: `.rename(columns=…)` — use `.withColumnRenamed(old, new)`")

    # A pandas import is only a violation when pandas is used for something
    # other than Excel I/O. Hard rule 5 requires pandas + openpyxl for reading
    # and writing workbooks (xlwings cannot run on a cluster), so flagging the
    # import there would tell the converter to undo the very fix the rule asks
    # for — the two checks would fight and the loop would never settle.
    if pandas_imports and non_excel_pandas:
        found = pandas_imports + found

    # de-dupe, keep order
    return list(dict.fromkeys(found))


def _merge_snippet(existing_src: str, snippet: str) -> tuple[str, list[str]]:
    """Deterministically merge a batch of new code into the existing file.

    Keeps a single import block + constants + function/class defs, de-duping by
    name so a batch never clobbers or duplicates already-converted functions.
    The file is rebuilt in Python (not by the LLM), so it never truncates no
    matter how many functions accumulate. Returns (new_source, added_names).
    """
    return _assemble(existing_src, snippet, replace=False)


def _raw_segments(src: str):
    """Split module source into top-level segments, keeping the ORIGINAL text of
    each (comments and formatting intact). ast is used only to locate line spans.

    Yields tuples (kind, key, raw_text) where kind is
    'import' | 'const' | 'def' | 'other'. Any `#` comment lines sitting directly
    above a segment are attached to it.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    prev_end = 0 
    for node in tree.body:
        node_start = node.lineno
        if getattr(node, "decorator_list", None):
            node_start = min(d.lineno for d in node.decorator_list)
        top = node_start - 1
        while top - 1 >= prev_end and lines[top - 1].strip().startswith("#"):
            top -= 1
        end = node.end_lineno
        raw = "".join(lines[top:end]).rstrip("\n")
        prev_end = end
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield "import", ast.unparse(node), raw, frozenset()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "def", node.name, raw, frozenset()
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            key = tuple(sorted(_module_const_names(
                ast.Module(body=[node], type_ignores=[]))))
            rhs = node.value
            refs = frozenset(
                n.id for n in ast.walk(rhs) if isinstance(n, ast.Name)
            ) if rhs is not None else frozenset()
            yield "const", (key or ("<expr>",)), raw, refs
        else:
            yield "other", None, raw, frozenset()


def _assemble(existing_src: str, snippet: str, replace: bool) -> tuple[str, list[str]]:
    """Rebuild the file from existing content + a snippet, deterministically,
    PRESERVING the original source text of each piece (comments included).

    Guarantees a well-formed library module:
      * imports are hoisted (deduped) to the top,
      * constants and defs keep their ENCOUNTER ORDER (constants are NEVER
        reordered above the functions — that was what produced
        `NameError: <fn> not defined` when a stray top-level call got hoisted),
      * stray top-level executable code and `__main__` guards are DROPPED (a
        converted function library must not run anything on import), and
      * an assignment whose right-hand side CALLS a locally-defined function
        (e.g. `order_df = make_orders_df()`) is DROPPED as demo/scratch code.

    For functions/classes:
      * replace=False (append/convert): a name already present is KEPT as-is.
      * replace=True (fix): a name already present is OVERWRITTEN in place.
    Returns (new_source, changed_names).
    """
    import_order: list[str] = []
    imports: dict[str, str] = {}
    body_order: list[tuple] = []         
    body: dict[tuple, str] = {}
    refs: dict[tuple, frozenset] = {}   
    def_names: set = set()
    changed: list[str] = []

    def _ingest(src: str, is_snippet: bool):
        if not src.strip():
            return
        for kind, key, raw, node_refs in _raw_segments(src):
            if kind == "import":
                if key not in imports:
                    import_order.append(key)
                    imports[key] = raw
            elif kind == "def":
                def_names.add(key)
                bk = ("def", key)
                if bk not in body:
                    body_order.append(bk)
                    body[bk] = raw
                    if is_snippet:
                        changed.append(key)
                elif is_snippet and replace:
                    body[bk] = raw           # overwrite in place, keep position
                    changed.append(key)
            elif kind == "const":
                bk = ("const", key)
                if bk not in body:
                    body_order.append(bk)
                    body[bk] = raw
                    refs[bk] = node_refs
                elif is_snippet and replace:
                    body[bk] = raw
                    refs[bk] = node_refs
            # 'other' is dropped

    _ingest(existing_src, False)   # existing content first (defines order)
    _ingest(snippet, True)         # then the snippet

    # Drop assignments that call a locally-defined function (stray demo code).
    kept = [
        bk for bk in body_order
        if not (bk[0] == "const" and (refs.get(bk, frozenset()) & def_names))
    ]

    parts = [_FILE_HEADER.rstrip("\n")]
    if import_order:
        parts.append("\n".join(imports[k] for k in import_order))
    if kept:
        parts.append("\n\n\n".join(body[bk] for bk in kept))
    return "\n\n".join(parts).rstrip() + "\n", changed


PROGRESS_FILE = OUTPUTS_DIR / "migration_progress.json"


def _write_progress(state, converted: set[str]) -> dict:
    """Record migration progress to outputs/migration_progress.json.

    Written after every batch so the run's state survives outside the LLM
    context — the agent reads it back with read_migration_progress_tool instead
    of us re-injecting the whole picture into its prompt each turn.
    """
    source = _source_function_names()
    remaining = [n for n in source if n not in converted]
    progress = {
        "source_function_count": len(source),
        "converted_count": len(source) - len(remaining),
        "remaining_count": len(remaining),
        "percent_complete": round(
            100.0 * (len(source) - len(remaining)) / len(source), 1
        ) if source else 0.0,
        "converted": [n for n in source if n in converted],
        "remaining": remaining,
        "extra_in_output": sorted(converted - set(source)),
        "output_file": state.get("converted_pyspark_file_path"),
    }
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    except OSError:
        pass
    return progress

SOURCE_SCRIPT = OUTPUTS_DIR / "source_script.py"


def _source_text() -> str:
    """The source script, read from the parser's canonical output path.

    Read from a fixed path rather than from a path published in state. The
    parser deliberately puts neither the script nor its location in state, so
    the previous `state.get("source_script_path")` returned None and every tool
    built on this silently saw an empty source: no function index, no bodies to
    convert, and no error explaining it.
    """
    try:
        return SOURCE_SCRIPT.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_source_index_tool(context: ToolContext, want: str = "constants") -> dict:
    """Source metadata WITHOUT function bodies. Ask for the part you need.

    Args:
        want: `"constants"` (default) returns module constants with their exact
            values — the reason this tool exists, since the case-fact checker
            reports a mismatch as a bare NAME and the value lives nowhere else
            the converter can reach. `"functions"` returns names and parameter
            lists. `"all"` returns both, and costs roughly four times a
            constants-only call.

    Returning everything by default was waste on every call: a whole-file
    function index is re-sent on each later turn that carries history, and the
    converter almost never needs it — the next batch's full source is injected
    into the prompt, and progress comes back from
    add_converted_functions_tool. The constants are the part it genuinely
    cannot get any other way.
    """
    want = (want or "constants").strip().lower()
    if want not in ("constants", "functions", "all"):
        return {"available": False,
                "error": f"want must be 'constants', 'functions' or 'all', not {want!r}"}

    out: dict = {"available": True}

    if want in ("constants", "all"):
        out["constants"] = _source_constants()

    if want in ("functions", "all"):
        src = _source_text()
        if not src.strip():
            return {"available": False, "count": 0, "functions": [],
                    "error": "source script not found"}
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            return {"available": False, "count": 0, "functions": [],
                    "error": f"source has a syntax error: {exc}"}

        functions = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # `kind` is only worth the bytes when it is NOT the ordinary case.
            entry = {"name": node.name}
            if isinstance(node, ast.ClassDef):
                entry["kind"] = "ClassDef"
            else:
                entry["parameters"] = [arg.arg for arg in node.args.args]
            functions.append(entry)

        out["count"] = len(functions)
        out["functions"] = functions

    return out


def _source_constants() -> dict:
    """Module-level constant name -> literal value, from the AST inventory.

    Long values are truncated with a marker rather than dropped: a constant the
    converter can see is wrong is far more useful than one it cannot see at all.
    """
    consts = _inventory().get("constants") or {}
    if not isinstance(consts, dict):
        return {}
    out = {}
    for name in sorted(consts):
        value = consts[name]
        if isinstance(value, str) and len(value) > MAX_CONSTANT_VALUE_CHARS:
            value = value[:MAX_CONSTANT_VALUE_CHARS] + "...<truncated>"
        out[name] = value
    return out


def read_migration_progress_tool(context: ToolContext) -> dict:
    """How much of the migration is done: converted vs remaining, by name.

    Read from outputs/migration_progress.json, refreshed after every batch. Use
    it to confirm what is left rather than assuming.
    """
    try:
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        # Keep the tool response compact; the full lists remain on disk.
        return {
            "source_function_count": progress.get("source_function_count", 0),
            "converted_count": progress.get("converted_count", 0),
            "remaining_count": progress.get("remaining_count", 0),
            "percent_complete": progress.get("percent_complete", 0.0),
            "remaining_sample": (progress.get("remaining") or [])[:5],
            "extra_in_output_count": len(progress.get("extra_in_output") or []),
            "output_file": progress.get("output_file"),
        }
    except (OSError, ValueError):
        source = _source_function_names()
        return {"source_function_count": len(source), "converted_count": 0,
                "remaining_count": len(source), "percent_complete": 0.0,
                "remaining_sample": source[:5], "extra_in_output_count": 0,
                "output_file": context.state.get("converted_pyspark_file_path")}


def _parsable_prefix(snippet: str) -> tuple[str, str]:
    """Split `snippet` into the longest parsable prefix and the broken tail.

    Exists because a MAX_TOKENS cutoff truncates the model mid-function, and the
    old behaviour threw the WHOLE batch away for it: three complete conversions
    plus one half-written one parsed as a SyntaxError, the tool rejected all
    four, and the loop re-converted the lot on the next turn — burning the same
    output budget and usually truncating in the same place. Since the cut is
    always at the END, everything before the last complete top-level construct
    is good code that has already been paid for.

    Cut points are the starts of top-level constructs (column 0, non-blank), so
    a prefix never ends inside a function body.

    Returns:
        `(parsable, dropped)`. `parsable` is `""` when nothing survives — a
        snippet broken from its first line, which is a real syntax error rather
        than a truncation.
    """
    try:
        ast.parse(snippet)
        return snippet, ""
    except SyntaxError:
        pass

    lines = snippet.splitlines(keepends=True)
    starts = [
        i for i, line in enumerate(lines)
        if line.strip() and not line[:1].isspace()
    ]
    for cut in reversed(starts):
        prefix = "".join(lines[:cut])
        if not prefix.strip():
            continue
        try:
            ast.parse(prefix)
            return prefix, "".join(lines[cut:])
        except SyntaxError:
            continue
    return "", snippet


def add_converted_functions_tool(context: ToolContext, functions_code: str) -> dict:
    """Append a BATCH of newly-converted PySpark code to the single output file.

    Pass ONLY the functions/classes you converted this turn (plus any imports or
    module-level constants they need) — never re-send functions already in the
    file. The file is reassembled deterministically in Python, so functions
    accumulate across loop iterations without you ever having to reproduce the
    whole file (which is what causes truncation / `# continue similarly` stubs).
    """
    p = _canonical_output_path(context.state)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    dropped = ""
    try:
        merged, added = _merge_snippet(existing, functions_code)
    except SyntaxError as exc:
        # Probably a MAX_TOKENS truncation rather than a bad conversion. Keep
        # the complete functions instead of discarding the batch; the loop's
        # work-list is derived from what is on disk, so whatever was cut simply
        # comes back in the next batch.
        salvaged, dropped = _parsable_prefix(functions_code)
        if not salvaged.strip():
            return {
                "status": "error",
                "error": f"The submitted functions_code has a syntax error: {exc}. "
                         "Fix it and resubmit only that batch.",
            }
        try:
            merged, added = _merge_snippet(existing, salvaged)
        except SyntaxError as exc2:
            return {
                "status": "error",
                "error": f"The submitted functions_code has a syntax error: {exc2}. "
                         "Fix it and resubmit only that batch.",
            }
    p.write_text(merged, encoding="utf-8")

    # Refresh the progress file, and report the SHORT form back — counts and
    # what is left, not the full list of everything already done.
    total = {
        n.name
        for n in ast.parse(merged).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    progress = _write_progress(context.state, total)
    # Every field here is re-sent on each later turn that carries history, so
    # this dict pays its size many times over. It keeps only what changes the
    # model's next action: what landed, and how much is left. Dropped as dead
    # weight: `saved_file_path` (identical on every call, and already in
    # state), `percent_complete` (derivable from the two counts), and the
    # 20-name `remaining_sample` — the next batch's actual source is injected
    # into the prompt each iteration, so a long name list steers nothing. A
    # short sample stays, because "0 remaining" and "3 remaining" read very
    # differently when the model decides whether to stop.
    result = {
        "status": "success",
        "functions_added_this_batch": added,
        "converted_count": progress["converted_count"],
        "remaining_count": progress["remaining_count"],
        "remaining_sample": progress["remaining"][:5],
    }

    # Deterministic check of the rule the conventions repeat hardest. Reported
    # rather than rejected: `.toPandas()` on a SMALL aggregate is required by
    # the visualisation conventions, so a hard block would refuse correct
    # plotting conversions. The batch is already saved; this tells the model
    # exactly what to fix with replace_functions_tool.
    if dropped:
        # Reported, not raised: the complete functions are already saved, and
        # the next work-list is derived from the file on disk, so the model
        # must NOT resend them.
        result["status"] = "partial"
        result["truncated"] = True
        result["truncated_tail_chars"] = len(dropped)
        result["action_required"] = (
            "Your submission was cut off mid-function — you hit the output "
            "token limit. The COMPLETE functions in it were saved; the "
            "incomplete tail was discarded. Do NOT resend what was saved. "
            "Convert fewer functions per call: submit one or two at a time "
            "with add_converted_functions_tool, using several calls, rather "
            "than building one large submission. `remaining_sample` below is "
            "authoritative for what is still outstanding."
        )

    violations = _pandas_violations(functions_code)
    if violations:
        result["pandas_violations"] = violations
        pandas_note = (
            "This batch uses pandas idioms that must be native Spark. Fix them "
            "with replace_functions_tool before moving on. If a `.toPandas()` is "
            "a deliberate small-aggregate collect for plotting, say so and keep it."
        )
        # Append rather than assign: a truncated batch can ALSO contain pandas
        # idioms, and overwriting would hide whichever notice was set first.
        result["action_required"] = (
            f"{result['action_required']} {pandas_note}"
            if result.get("action_required") else pandas_note
        )
    return result

def _apply_function_patch(p: pathlib.Path, functions_code: str) -> List[str]:
    """Merge `functions_code` into the file at `p`, replacing same-named
    functions/classes/constants in place and appending brand-new ones.

    The shared core behind replace_functions_tool AND the SEMS parallel-fixer
    patch-integration callback (_apply_sems_fix_patches) — both need the exact
    same read-merge-write behavior, just triggered from different callers.
    Raises SyntaxError if `functions_code` (or the resulting merge) is invalid
    Python, same as _assemble.
    """
    existing = p.read_text(encoding="utf-8")
    merged, changed = _assemble(existing, functions_code, replace=True)
    p.write_text(merged, encoding="utf-8")
    return changed


def replace_functions_tool(context: ToolContext, functions_code: str) -> dict:
    """Surgically REPLACE specific functions in the converted file, in place.

    Used by the fixer agents. Submit ONLY the corrected version(s) of the
    function(s) you are fixing (plus any new imports/constants they need). Each
    function in `functions_code` overwrites the same-named function already in
    the file; every OTHER function is left byte-for-byte untouched. Brand-new
    names are appended. This means you NEVER reproduce the whole file — so a big
    file can't get truncated into half-converted code. Fix in small batches.
    """
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {
            "status": "error",
            "error": "No converted file exists yet — nothing to fix.",
        }
    try:
        changed = _apply_function_patch(p, functions_code)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"The submitted functions_code has a syntax error: {exc}. "
                     "Fix it and resubmit only that batch.",
        }
    total = {
        n.name
        for n in ast.parse(p.read_text(encoding="utf-8")).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        "status": "success",
        "saved_file_path": str(p),
        "functions_replaced_or_added": changed,
        "total_function_count": len(total),
    }


def read_functions_tool(context: ToolContext, function_names: list[str]) -> dict:
    """Return the current source of ONLY the named functions from the converted
    file (so a fixer can inspect just the ones it needs without pulling the whole
    file into context). Unknown names are reported under `not_found`."""
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {"exists": False, "functions": {}, "not_found": list(function_names or [])}
    tree = ast.parse(p.read_text(encoding="utf-8"))
    by_name = {
        n.name: ast.unparse(n)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    requested = list(function_names or [])
    found = {name: by_name[name] for name in requested if name in by_name}
    not_found = [name for name in requested if name not in by_name]
    return {"exists": True, "functions": found, "not_found": not_found}


def read_source_functions_tool(context: ToolContext, function_names: list[str]) -> dict:
    """Return the ORIGINAL Python source of the named functions (ground truth).
 
    Use this before fixing/converting a function so you match the source's real
    behaviour instead of guessing. Read from outputs/source_script.py — pull only
    the handful you are working on this turn, never the whole file.
    Unknown names are listed under `not_found`."""
    src = _source_text()
    if not src.strip():
        return {
            "available": False, "functions": {},
            "not_found": list(function_names or []),
            "note": (
                f"The original source is not readable at {SOURCE_SCRIPT}, so there "
                f"is NO ground truth available for any function this run. Say so "
                f"rather than guessing at behaviour."
            ),
        }
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {
            "available": False, "functions": {},
            "not_found": list(function_names or []),
            "note": (
                f"The original source at {SOURCE_SCRIPT} does not parse, so no "
                f"ground truth is available. Say so rather than guessing."
            ),
        }
    by_name = {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    requested = list(function_names or [])
    found = {name: by_name[name] for name in requested if name in by_name}
    not_found = [name for name in requested if name not in by_name]
    result = {"available": True, "functions": found, "not_found": not_found}
    if not_found:
        result["note"] = (
            "These names have no counterpart in the original script. That is "
            "normal for helpers the conversion introduced (session builders, "
            "save/write helpers, port-only utilities) -- it does NOT mean they "
            "are hallucinated and must NOT be deleted. Define their expected "
            "behaviour from the conversion conventions and from their call "
            "sites in the converted module instead."
        )
    return result


def read_converted_file_tool(context: ToolContext) -> dict:
    """List what is in the converted file: function names, and how big it is.

    Deliberately does NOT return the code. This answers "which functions exist?",
    which is the question it is almost always asked; pulling the whole file to
    answer it put thousands of tokens into context that then rode along in every
    later request. When you need a body, call **read_functions_tool** with the
    names you got from here."""
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {"exists": False, "function_names": [], "line_count": 0}
    src = p.read_text(encoding="utf-8")
    names = sorted(
        n.name
        for n in ast.parse(src).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return {"exists": True, "function_names": names,
            "line_count": len(src.splitlines())}



#: Hard ceiling on functions per batch.
BATCH_SIZE = 4

#: Ceiling on the SOURCE characters in one batch, which is the real constraint.
#: A batch is capped by whichever limit binds first.
#:
#: A fixed count alone is the wrong unit and it broke: BATCH_SIZE was tuned when
#: the refactor stage emitted ~4-statement functions, then the refactor moved to
#: coarser blocks with a median of ~13, so "4 functions" silently became roughly
#: three times the output and the converter started hitting MAX_TOKENS
#: mid-submission. Converted PySpark runs longer than the Python it came from
#: (typed signature, docstring, explicit column expressions), so the budget is
#: well under the output cap rather than close to it.
BATCH_CHAR_BUDGET = 2500


def _missing_constants(state) -> dict:
    """Source constants that are NOT yet in the converted file, with values.

    The converter's instructions say to emit every module-level constant with
    its exact value, but nothing was putting those values in front of it:
    `_next_batch_source` collects function and class bodies only, and the
    constants live at module level, so the model saw them only if it chose to
    call read_source_index_tool on its own initiative. It usually did not — the
    converted file came back with 0 of the source's 2 constants, and the
    case-fact checker then failed the run for a value the model was never shown.

    Values come from the inventory UNTRUNCATED: a shortened literal would be
    wrong code, and a wrong constant is worse than a long prompt.

    Recomputed against the file on disk each turn, so it is self-clearing —
    once a constant is written it stops being injected, and if a batch is lost
    it reappears without any state to go stale.
    """
    consts = _inventory().get("constants") or {}
    if not isinstance(consts, dict) or not consts:
        return {}
    have: set[str] = set()
    path = _canonical_output_path(state)
    if path.exists():
        try:
            have = _module_const_names(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError):
            have = set()
    return {name: consts[name] for name in sorted(consts) if name not in have}

def _apply_import_removal(p: pathlib.Path, names: list[str]) -> tuple[set[str], set[str]]:
    """Remove specific UNUSED names from the module-level import block of the
    file at `p`. Returns `(removed, not_found)`.

    The shared core behind remove_unused_imports_tool AND the SEMS
    parallel-fixer patch-integration callback (_apply_sems_fix_patches).
    """
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    targets = set(names or [])

    removed: set[str] = set()
    # (start, end, replacement) 0-indexed half-open line ranges; replacement of
    # None means delete the span outright. Applied back-to-front so earlier
    # edits don't shift the line numbers later edits were computed against.
    edits: list[tuple[int, int, Optional[str]]] = []

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        bound_names = [
            (a.asname or a.name.split(".")[0]) if isinstance(node, ast.Import) else (a.asname or a.name)
            for a in node.names
        ]
        hit = [n for n in bound_names if n in targets]
        if not hit:
            continue
        removed.update(hit)
        start, end = node.lineno - 1, node.end_lineno  # 0-indexed [start, end)
        keep_aliases = [a for a, n in zip(node.names, bound_names) if n not in targets]
        if not keep_aliases:
            edits.append((start, end, None))
        else:
            new_node = (
                ast.ImportFrom(module=node.module, names=keep_aliases, level=node.level)
                if isinstance(node, ast.ImportFrom)
                else ast.Import(names=keep_aliases)
            )
            edits.append((start, end, ast.unparse(new_node) + "\n"))

    if removed:
        for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
            lines[start:end] = [replacement] if replacement is not None else []
        new_src = "".join(lines)
        ast.parse(new_src)  # sanity-check the edit kept the file syntactically valid
        p.write_text(new_src, encoding="utf-8")

    return removed, targets - removed


def remove_unused_imports_tool(context: ToolContext, names: list[str]) -> dict:
    """Remove specific UNUSED names from the module-level import block.

    Dedicated counterpart to replace_functions_tool/add_converted_functions_tool
    for import cleanup: those tools' merge logic only ever UNIONS imports
    (ingests existing imports, then adds any new ones from the snippet) so a
    batch can never silently drop an import some other function still needs —
    which also means neither tool has any way to delete one. This is the one
    tool that can, for the F401 ("imported but unused") / W0611 class of gap.

    Pass the exact NAME the linter flagged as unused — the identifier bound in
    this module's namespace (e.g. "date" for `from datetime import date`, or
    "np" for `import numpy as np`), NOT the module path. A
    `from X import A, B` statement keeps whichever of A/B you don't list; a
    bare `import X` is dropped entirely when X is listed. Only the import
    line(s) are touched — every function/class/constant is left
    byte-for-byte untouched.

    Args:
        context: The agent state of type ToolContext.
        names: Bound names to remove from the import block.

    Returns:
        Dict with `removed` (names actually deleted) and `not_found`.
    """
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {"status": "error", "error": "No converted file exists yet — nothing to fix."}

    removed, not_found = _apply_import_removal(p, names)
    if not removed:
        return {"status": "no_change", "removed": [], "not_found": sorted(not_found)}
    return {
        "status": "success",
        "removed": sorted(removed),
        "not_found": sorted(not_found),
    }

def _next_batch_source(state) -> str:
    """Source of the next BATCH_SIZE functions still to convert.

    Injected directly rather than fetched with a tool. A tool call costs a whole
    extra model round-trip, and every round-trip re-sends the ~10k-token
    conventions block — so paying ~2k of prompt here saves ~12k of round-trip.
    The model gets exactly the same bodies either way.
    """
    # Derive the work-list from disk instead of storing the full list in ADK state.
    # ADK may carry state into every model turn, so keeping hundreds of function
    # names in state can become a large repeated prompt payload.
    source_names = _source_function_names()
    output_path = _canonical_output_path(state)
    converted_names: set[str] = set()
    if output_path.exists():
        try:
            tree = ast.parse(output_path.read_text(encoding="utf-8"))
            converted_names = {
                n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        except (OSError, SyntaxError):
            converted_names = set()
    missing = [name for name in source_names if name not in converted_names]
    if not missing:
        return "(nothing left to convert)"
    src = _source_text()
    if not src.strip():
        return "(source unavailable — use read_source_functions_tool)"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(source does not parse — use read_source_functions_tool)"
    by_name = {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    # Fill the batch by size, not just by count: one long function is a whole
    # batch, several short ones can share. The first function is always taken
    # even if it alone busts the budget, otherwise an oversized function would
    # never be offered and the loop would spin on it forever.
    batch: list[str] = []
    used = 0
    for name in missing[:BATCH_SIZE]:
        body = by_name.get(name) or ""
        if batch and used + len(body) > BATCH_CHAR_BUDGET:
            break
        batch.append(name)
        used += len(body)

    parts = [by_name[n] for n in batch if by_name.get(n)]
    missing_bodies = [n for n in batch if not by_name.get(n)]
    out = "\n\n".join(parts)
    if missing_bodies:
        out += ("\n\n# not found in the source: " + ", ".join(missing_bodies)
                + " — fetch with read_source_functions_tool")
    out = out or "(no bodies found — use read_source_functions_tool)"

    # Prepend any constants still absent from the converted file. Injected
    # rather than left to a tool call, because the model has to know the value
    # to write it and cannot infer 0.908 from the name USD_2_EUR.
    pending = _missing_constants(state)
    if pending:
        block = "\n".join(f"{name} = {value!r}" for name, value in pending.items())
        out = (
            "# MODULE-LEVEL CONSTANTS still missing from the converted file.\n"
            "# Emit these VERBATIM at module level in this batch, exactly once,\n"
            "# with these exact values. They are not optional and cannot be\n"
            "# derived from anything else you can see.\n"
            f"{block}\n\n"
            + out
        )
    return out


def _compact_case_fact_status(state) -> dict:
    """Build a small status object for the model without carrying the full work-list.

    The complete migration lists live in migration_progress.json / the converted file.
    Keeping them out of ADK state prevents them from being echoed into every LLM turn.
    """
    existing = state.get("status") or {}
    converted_names = _converted_function_names(state)
    missing = [
        name for name in _source_function_names()
        if name not in converted_names
    ]
    # `constant_mismatch_sample` is the key the case-fact checker actually
    # writes. Reading the pre-trim name (`constant_value_mismatch`) yielded an
    # empty list on every turn, so the converter was told the constants were
    # fine no matter how many the checker had flagged — and constants are the
    # one thing the loop cannot discover for itself, because the checker
    # escalates only when they all match.
    mismatch = existing.get("constant_mismatch_sample") or []
    mismatch_count = existing.get("constant_value_mismatch_count", len(mismatch))
    return {
        "status": existing.get("status", "error"),
        "function_missing_count": len(missing),
        "function_missing_sample": missing[:20],
        "constant_value_mismatch": mismatch[:20],
        "constant_value_mismatch_count": mismatch_count,
        "constant_value_mismatch_truncated": mismatch_count > len(mismatch[:20]),
        "message": (
            "Convert the next batch of functions. The authoritative work-list "
            "is derived from the source and converted files on disk."
        ),
    }


def _converted_function_names(state) -> set[str]:
    """Return names already present in the assembled converted file."""
    p = _canonical_output_path(state)
    if not p.exists():
        return set()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    return {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def seed_fact_status(callback_context: CallbackContext) -> None:
    """Prepare only compact, deterministic state for the converter turn."""
    state = callback_context.state
    state["status"] = _compact_case_fact_status(state)
    state["next_batch_source"] = _next_batch_source(state)
    return None


def _skip_agent_response(message: str):
    """Content that makes ADK skip the agent, or None if it cannot.

    Returning `types.Content` from a before-agent callback is ADK's documented
    way to bypass the model call. Wrapped because the import path has moved
    between versions and a skip optimisation must never be what breaks a run:
    on any failure this returns None, the callback falls through, and the agent
    runs exactly as it did before.
    """
    try:
        from google.genai import types

        return types.Content(role="model", parts=[types.Part(text=message)])
    except Exception:
        return None


def seed_parity_fixer(callback_context: CallbackContext):
    """Prepare the parity fixer, and skip it when there is nothing to fix.

    The parity loop is [writer, fixer], but the writer only runs the suite on
    its LAST batch — every earlier batch leaves no pytest result at all. The
    fixer was still invoked each time, read an empty result, announced that
    nothing was failing and stopped: a full model turn, and a full prompt, to
    say "no work". On a three-batch module that is two wasted turns before the
    suite has run once.

    Skipping is safe precisely because there is nothing to do: no failing test
    means no function to repair, which is the same conclusion the agent reached
    on its own, only without paying for it.
    """
    state = callback_context.state
    if state.get("pyspark_conventions"):
        state["pyspark_conventions"] = ""

    result = state.get("pytest_last_result")
    if result is None:
        result = {
            "passed": None,
            "failed_tests": [],
            "failed_count": 0,
            "note": "The suite has not run yet — there is nothing to fix.",
        }
        state["pytest_last_result"] = result

    failing = (result or {}).get("failed_tests") or []
    broke = (result or {}).get("run_error")
    if not failing and not broke:
        return _skip_agent_response(
            "No failing tests to repair — the suite has not run yet, or it passed."
        )
    return None


def seed_conventions(callback_context: CallbackContext) -> None:
    """Keep fixer state compact; conventions are supplied by the SkillToolset."""
    state = callback_context.state
    # Do not inject the complete SKILL.md/reference corpus into state.
    # The SkillToolset remains available to the fixer when it needs conventions.
    #
    # Cleared by assignment, not `state.pop(...)`: ADK's State is dict-LIKE but
    # does not implement the full mapping API, and `pop` raised
    # `AttributeError: 'State' object has no attribute 'pop'` — which killed the
    # semantic fixer before it ran, every time. Only `.get()` and `[]=` are
    # relied on here; those are what the rest of this package uses.
    if state.get("pyspark_conventions"):
        state["pyspark_conventions"] = ""

    # Seed the key this agent's prompt interpolates. It is written by the parity
    # agent's pytest run, which has not happened on the first iteration of the
    # loop — the writer adds a batch and stops, then this agent runs. Without a
    # default the prompt raises
    # `KeyError: Context variable not found: pytest_last_result` before the
    # agent starts, so the fixer never ran at all on iteration one.
    #
    # Seeded in the CONSUMER's own callback rather than the producer's, so it
    # holds no matter which agent runs first or whether the suite ever ran.
    if state.get("pytest_last_result") is None:
        state["pytest_last_result"] = {
            "passed": None,
            "failed_tests": [],
            "failed_count": 0,
            "note": "The suite has not run yet — there is nothing to fix. "
                    "Make NO changes and stop.",
        }

    # Same treatment for the semantic fixer, which shares this callback. Today
    # `semantic_match` is seeded by the semantic validator's own callback and
    # that agent always runs first in its loop — so this is currently
    # redundant. It is here because relying on run order is exactly what failed
    # above: a key that exists only because some other agent went first is one
    # reordering away from a KeyError before the agent starts.
    if state.get("semantic_match") is None:
        state["semantic_match"] = {
            "match": False,
            "differences": [],
            "message": "Semantic validation has not run yet.",
        }
    return None


HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER=os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json"
}

def _persist_execution_state(context: ToolContext, result: dict) -> dict:
    """Mirror this call's outcome into state so callbacks outside the tool's
    own return value (e.g. sems_correction_loop_agent's final execution
    check) can read the real result instead of falling back to defaults.
    """
    context.state["pyspark_execution_status"] = result.get("status")
    context.state["pyspark_execution_run_id"] = result.get("run_id", "unknown")
    context.state["pyspark_execution_error"] = result.get("error", "")
    return result

def execute_pyspark_script_tool(context: ToolContext) -> dict:
    """Run the converted PySpark file on Databricks to catch syntax/runtime errors.

    Uploads the current converted file (path read from state) to the Databricks
    workspace as a notebook, submits it to serverless compute, waits for the run
    to finish, and deletes the uploaded notebook. Returns a compact dict so the
    agent can decide whether to fix and retry — the raw Databricks payload is
    NOT returned in full (it is large and would flood the context window).
    """
    python_script_path = context.state.get("converted_pyspark_file_path")
    if not python_script_path:
        return _persist_execution_state(context, {
            "success": False,
            "error": "No converted PySpark file found in state. Call add_converted_functions_tool first.",
        })

    python_script_path = str(python_script_path)

    file_name = f"generated_{uuid.uuid4().hex}.py"
    workspace_path = f"/Workspace/Users/{USER}@shell.com/Drafts/{file_name}"
    # Read before the try block, so guard it separately: an unreadable file here
    # would otherwise escape as an exception instead of a tool result.
    try:
        # Explicit encoding: the platform default is cp1252 on Windows, which
        # cannot read the non-ASCII characters a converted file may carry.
        with open(python_script_path, "r", encoding="utf-8") as file:
            code = file.read()
    except OSError as exc:
        return _persist_execution_state(context, {
            "success": False,
            "file_path": python_script_path,
            "error": f"Could not read the converted file: {exc}",
        })
    timeout = 600

    try:
        upload_payload = {
            "path":workspace_path,
            "format":"SOURCE",
            "language":"PYTHON",
            "overwrite":True,
            "content":base64.b64encode(code.encode()).decode()
        }

        r = requests.post(
            f"{HOST}/api/2.0/workspace/import",
            headers=HEADERS,
            json=upload_payload
        )

        if not r.ok:
            return _persist_execution_state(context, {
                "success": False,
                "file_path": python_script_path,
                "status_code": r.status_code,
                "error": _tail(r.text, 1000),
            })
        r.raise_for_status()

        submit_payload = {
            "run_name":"varification",
            "tasks":[
                {
                    "task_key":"execute",
                    "notebook_task":{
                        "notebook_path":workspace_path,
                    },
                    "environment_key":"default_python"
                }

            ],
            "environments":[
                {
                    "environment_key":"default_python",
                    "spec":{
                        "environment_version":"4"
                    }
                }
            ]
        }

        r = requests.post(
            f"{HOST}/api/2.2/jobs/runs/submit",
            headers=HEADERS,
            json=submit_payload
        )

        r.raise_for_status()

        run_id = r.json()["run_id"]

        start = time.time()

        while True:
            r = requests.get(
                f"{HOST}/api/2.2/jobs/runs/get",
                headers=HEADERS,
                params={"run_id": run_id},
            )

            r.raise_for_status()
            info = r.json()
            task = info["tasks"][0]
            task_run_id = task["run_id"]
            state = task["state"]["life_cycle_state"]

            if state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break

            if time.time()-start>timeout:
                return _persist_execution_state(context, {
                    "success": False,
                    "file_path": python_script_path,
                    "status": "TIMEOUT",
                    "run_id": run_id,
                    "life_cycle_state": state,
                })

            time.sleep(5)

        r = requests.get(
            f"{HOST}/api/2.2/jobs/runs/get-output",
            headers=HEADERS,
            params={"run_id":task_run_id}
        )
        output = r.json()

        status = info["state"].get("result_state")
        success = status == "SUCCESS"
        out = {
            "success": success,
            "file_path": python_script_path,
            "status": status,
            "run_id": run_id,
            "life_cycle_state": state,
        }
        if not success:
            error_text = output.get("error") or ""
            trace = output.get("error_trace") or ""
            out["error"] = _tail(error_text)
            out["error_summary"] = _error_summary(trace or error_text, python_script_path)
            missing = _unavailable_module(f"{trace}\n{error_text}")
            if missing:
                # Terminal, not flaky: re-running cannot install a desktop
                # application onto a cluster. Say so explicitly, because the
                # generic "read error_summary and fix the broken piece"
                # instruction otherwise reads as "retry the same code".
                out["fatal"] = True
                out["unavailable_module"] = missing
                out["action_required"] = (
                    f"`{missing}` does not exist on Databricks and never will — "
                    f"it drives a local desktop application, so no cluster can "
                    f"import it. DO NOT re-run this script and DO NOT retry the "
                    f"import. Rewrite the offending function(s) now with "
                    f"{UNAVAILABLE_ON_DATABRICKS[missing]} using "
                    f"replace_functions_tool, then run once more. Excel reads "
                    f"become `pandas.read_excel(path, sheet_name=...)`; writes "
                    f"become `pandas.DataFrame.to_excel(path, sheet_name=..., "
                    f"index=False)` or an `openpyxl` workbook. Keep the same "
                    f"file paths, sheet names, and cell ranges as the source."
                )
        return _persist_execution_state(context, out)

    except Exception as exc:
        return _persist_execution_state(context, {
            "success": False,
            "file_path": python_script_path,
            "error": f"Databricks execution failed: {type(exc).__name__}: {exc}",
        })

    finally:
        try:
            requests.post(
                f"{HOST}/api/2.0/workspace/delete",
                headers=HEADERS,
                json={
                    "path":workspace_path,
                    "recursive":False,
                },
            )
            print("workspace_deleted")
        except Exception as ex:
            print("cleanup failed")


py_to_spark_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "py2snow-skill"
)

my_skill_toolset = SkillToolset(
    skills=[py_to_spark_skill]
)

code_convertor_agent = Agent(
    name="agent_code_converter",
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction= """You are an expert coder who converts a Python ELT script into equivalent,
    distributed **PySpark** code. The output file is built up INCREMENTALLY across several
    turns — you convert a BATCH of functions each turn and APPEND them to the single output
    file. You must NEVER try to output the whole file at once (that truncates and produces
    "# continue similarly" stubs — which is a failure).

    MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
    for the native-Spark conversion rules. The skill MUST remain available and is the
    authoritative source for detailed conversion conventions. Use its resources when
    needed, but do not repeatedly reload or reread the same skill resource in one turn,
    and never reproduce the entire skill/reference corpus in the response. In particular,
    use native Spark APIs and avoid pandas idioms
    (`pd.`, `.merge`, `.rename(columns=...)`, `.iloc`, `df.apply`) and numpy column-building
    patterns unless the source is a deterministic data-generation function.

    THIS TURN'S BATCH — the original source of the next functions to convert is
    already here. Convert exactly these; do NOT call a tool to fetch them unless a
    body is actually missing or you need to verify a specific function:
    <batch_source>
    {next_batch_source}
    </batch_source>

    Other tools, only if you actually need them (each call costs a full round-trip):
      * **read_source_functions_tool(function_names=[...])** — ONLY if a body is
        missing from the supplied batch or you genuinely need to re-check it.
      * **read_source_index_tool(want=...)** — `want="constants"` (the default)
        for module constants and their exact values; `want="functions"` for
        names and parameters; `want="all"` for both, which is ~4x the size.
        Ask for the smallest part that answers your question.
      * **read_migration_progress_tool()** — ONLY if progress is unclear.
      * **read_converted_file_tool()** — ONLY if you need output function names.
      * Do NOT repeatedly call a tool for information already present in this turn.
      * Do NOT repeatedly load/re-read the same skill resource in one turn. The
        py2snow-skill remains authoritative and available through SkillToolset.

    WORK-STATE (compact):
    <case_fact_status>
    {status}
    </case_fact_status>

    The complete work-list is derived from the source and converted files on disk.
    Do not ask for, reproduce, or store the whole work-list in state; convert only
    <batch_source>.

    You can use the **py2snow-skill** to guide each conversion.

    HOW TO WORK (batched append — follow exactly):
    1. Convert every function in <batch_source> above. Real, complete
       implementations: ABSOLUTELY NO placeholder comments (never write
       "# continue similarly", "# add remaining functions", "# TODO", "...", or an
       empty/`pass` body). A stubbed function does not count as converted and will
       just come back in the next work-list. Your output limit is ~20k tokens for
       the WHOLE turn, including the tool call itself, so a batch of two or
       three real functions can exceed it. Submit as you go — call
       add_converted_functions_tool with one or two functions, then call it
       again for the next one or two — rather than converting everything and
       submitting once. Several small calls always beat one that gets cut off.
       If you are cut off anyway, the complete functions you sent are saved and
       reported back; do NOT resend them.
    3. Call **add_converted_functions_tool(functions_code=...)** passing ONLY the
       functions you just converted — one or two per call, calling it as many
       times as the batch needs. On the FIRST batch also include: the needed `import` lines (pyspark
       imports, etc.) AND every module-level constant from the source with its EXACT value
       Any constant still missing is listed AT THE TOP of <batch_source> with its
       exact value — copy those lines verbatim to module level; they are not
       optional. The list shrinks as you write them, so an empty one means the
       constants are done. `constant_value_mismatch` names constants whose value
       is wrong; read_source_index_tool() can re-read all of them if you need to
       check one.
       Do NOT resend functions already in the file — the tool merges and de-dupes by name;
       it returns `converted_count` and `remaining_count` — use `remaining_count` as
       the authoritative progress signal, not the short `remaining_sample`.
    4. STOP your turn as soon as the batch is appended. The loop re-invokes you with a freshly computed next batch; keep going batch by batch until `remaining_count` is 0.
    5. ONLY when `remaining_count` comes back 0 — the final batch — call
       **execute_pyspark_script_tool** once as a whole-file check. Do NOT run it after
       every batch: the converted file is a function library, so running it only
       proves the file imports, and each run costs a full round-trip. If it reports
       success=false, read `error_summary` and fix just the broken piece.
       If the result has a `pip_required` field, the cluster was simply missing an
       installable package. That is NOT a code problem: do NOT edit the code and do
       NOT drop the import — call execute_pyspark_script_tool again and the next run
       installs it. (`openpyxl` and `databricks-sql-connector` are installed
       automatically whenever the converted code imports them, so keep using
       `pandas` + `openpyxl` for Excel exactly as the conventions require.)
       If the result has `fatal: true` and an `unavailable_module`, STOP re-running:
       that module cannot exist on a Databricks cluster, so the same run will fail
       forever. Do exactly what `action_required` says — rewrite those functions with
       the named replacement, then run once more.

    **STRICT RULES**
    - Do not change the underlying logic of the functions.
    - Do not infer/rename columns — use the source script column names.
    - Do not change constant values; include every source constant with its correct value,
      and do NOT invent constants that are not in the source.
    - NEVER emit `xlwings` (or `win32com`/`pywin32`/`xlsxwriter`). They drive a local
      Excel/desktop application over COM and cannot run on a Databricks cluster. Convert
      Excel work to `pandas` + `openpyxl`: `pandas.read_excel(path, sheet_name=...)` to
      read, `DataFrame.to_excel(path, sheet_name=..., index=False)` or an `openpyxl`
      workbook to write. Keep the source's file paths, sheet names, and cell ranges
      exactly. This is the one place a pandas call is CORRECT rather than a violation —
      Excel I/O is driver-side file work, not a distributed transform.
    - Convert same-named functions (the PySpark function name must equal the source name).
    - Convert EVERY source function, including the orchestrator (e.g. `run_all`) — it is a
      normal function. Do NOT invent helpers that are not in the source (the conventions
      skeleton already provides `get_spark`; do not add a `get_spark_session`).
    - IMPORTS: every non-local name you use MUST be imported — follow section "3a. Imports
      discipline" of the conventions above (name → import lookup). Re-check imports for each
      batch before submitting; a missing import becomes a NameError at test time.
    - SEMS / clean-code compliance is MANDATORY (see the "SEMS Compliance Conventions"
      section above): typed signatures, a docstring on every function naming its source
      function, comments on non-trivial transforms, `logging` (never `print`), specific
      `try/except` (never bare/broad, never `except: pass`) around risky ops, values from
      `CONFIG` (no magic literals), and no dead/commented-out code, TODOs, or stubs.
    - Never re-emit or overwrite the whole file; only append your new batch.
    - Do NOT emit module-level executable code, demo calls, or `if __name__ == "__main__"`
      blocks — the file is a function library (such lines are stripped anyway).
    - Data-generation functions (those using numpy / `random` / seeded generators to build
      synthetic rows) must build the data in PLAIN Python and pass it to
      `spark.createDataFrame(...)`. Do NOT translate seeded random generation into
      `F.rand()` column expressions — it changes behaviour and breaks determinism.
    """,

    tools=[
        my_skill_toolset,
        read_source_index_tool,
        read_source_functions_tool,
        read_migration_progress_tool,
        add_converted_functions_tool,
        read_converted_file_tool,
        execute_pyspark_script_tool,
    ],

    mode="task",
    output_key="code_converter_output",
    before_agent_callback=seed_fact_status,
)

code_fixer_agent = Agent(
    name="code_fixer_agent",
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction= """You are an expert PySpark engineer. A converted PySpark pipeline
    already exists on disk (it may contain dozens of functions), but its pytest parity
    suite is FAILING. Fix ONLY the converted code — you do NOT edit the tests, and you
    fix ONLY the functions that are actually failing, one small batch at a time.
 
    MANDATORY conversion conventions — the corrected code MUST follow these; never use
    pandas (`pd.`, `.merge`, `.rename(columns=)`, `.iloc`, `df.apply`) or numpy (`np.`),
    use native Spark DataFrame / `pyspark.sql.functions`:
    <pyspark_conversion_conventions>
    {pyspark_conventions}
    </pyspark_conversion_conventions>
 
    Condensed pytest result — this lists only the FAILING tests and a short error for
    each (test `test_<function_name>` maps to the function `<function_name>`):
    <pytest_result>
    {pytest_last_stdout}
    </pytest_result>
 
    Latest parity verdict:
    <parity_test_status>
    {parity_test_status}
    </parity_test_status>
 
    HOW TO WORK (surgical, batched — follow exactly):
    1. From <pytest_result>, list the FAILING functions (strip the `test_` prefix from
       each failing test name). If nothing is failing, make NO changes and stop.
    2. Take the first few failing functions (up to ~8). Call **read_functions_tool** with
       exactly those names to get their CURRENT (converted) source, AND
       **read_source_functions_tool** with the same names to get the ORIGINAL Python
       source. Do NOT pull the whole file.
    3. Compare the two. The original Python is the GROUND TRUTH for what the function must
       do — fix the converted function so its behaviour matches the original. If the short
       error references something that does not exist in the original (a column, a call, a
       line), that is a hallucination in the converted code — remove it.
    4. Call **replace_functions_tool(functions_code=...)** passing ONLY the corrected
       function(s) (plus any missing import/constant they need). It replaces those
       functions in place and leaves every other function untouched — NEVER paste the
       whole file, NEVER re-send functions you are not changing, NEVER add module-level
       calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
    5. Call **execute_pyspark_script_tool** ONCE as a syntax/runtime sanity check.
    6. If more failing functions remain, repeat from step 2 for the next batch. Then stop
       — the parity agent re-runs the full suite and returns any still-failing functions.
 
    **STRICT RULES**
    - Do NOT change the underlying business logic; match the ORIGINAL Python behaviour.
    - Do NOT change or infer column names; do NOT change any constant values.
    - Data-generation functions (e.g. those using numpy / `random` / seeded generators to
      build synthetic rows) must build the data in PLAIN Python and pass it to
      `spark.createDataFrame(...)`. Do NOT translate seeded random generation into
      `F.rand()` column expressions — that changes behaviour and is a common wrong fix.
    - Keep the corrected function SEMS-compliant per the conventions above (typed signature,
      docstring, comments on non-trivial logic, `logging` not `print`, specific `try/except`
      never bare, values from `CONFIG`, no dead/commented-out code or TODOs).
    - The success signal for execute_pyspark_script_tool is the `success` field
      (`status == "SUCCESS"` on the Databricks run). Do NOT re-run when success is true.
      A `status` of "TIMEOUT" means the Databricks run did not finish in time — that is
      an infrastructure result, not evidence that a function is wrong.
    - You do NOT run the pytest suite yourself.
    """,
 
    tools=[
        my_skill_toolset,
        read_functions_tool,
        read_source_functions_tool,
        replace_functions_tool,
        execute_pyspark_script_tool,
    ],
 
    mode="task",
    output_key="pyspark_conversion_output",
    before_agent_callback=seed_conventions,
)

semantic_code_fixer_agent = Agent(
    name="semantic_code_fixer_agent",
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction= """You are an expert PySpark engineer. A converted PySpark pipeline
    exists on disk (possibly dozens of functions), but when the SAME data is run through
    both the source Python pipeline and the converted PySpark pipeline, their OUTPUTS DO
    NOT MATCH. Fix ONLY the converted PySpark code so its output equals the Python output,
    editing ONLY the functions responsible for the differences — surgically, in batches.

    MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
    for native Spark rules. Do not reproduce the full skill/reference corpus in context.
    Never introduce pandas idioms (`pd.`, `.merge`, `.rename(columns=)`, `.iloc`, `df.apply`)
    or numpy column-building patterns into the corrected code.

    Semantic comparison verdict (differences between the two outputs — these describe
    columns/values, so reason about WHICH function produces each differing column):
    <semantic_match>
    {semantic_match}
    </semantic_match>

    HOW TO WORK (surgical, batched — follow exactly):
    1. If `semantic_match.match` is already true, make NO changes and stop.
    2. From `semantic_match.differences`, work out the small set of functions responsible
       (the ones that compute/transform the differing columns). If you are unsure which
       functions exist, call **read_converted_file_tool** ONCE — it lists names only.
    3. For those functions, call **read_functions_tool** (current converted source) AND
       **read_source_functions_tool** (ORIGINAL Python source = ground truth). Do NOT pull
       the whole file if you only need a few functions.
    4. Diagnose each difference against the ORIGINAL Python behaviour (wrong join type,
       aggregation, window/ordering semantics, type/precision, null-handling, column
       names) and correct only those functions so they reproduce the Python result.
    5. Call **replace_functions_tool(functions_code=...)** with ONLY the corrected
       function(s). It replaces them in place and leaves every other function untouched —
       NEVER paste the whole file, NEVER re-send functions you are not changing, NEVER add
       module-level calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
    6. Call **execute_pyspark_script_tool** ONCE as a syntax/runtime sanity check, then stop.
       The semantic agent re-runs both pipelines and returns a fresh diff if needed.

    **STRICT RULES**
    - Fix ONLY the converted PySpark code. Do NOT change the source Python, the dummy
      dataset, the runner scripts, or the recorded Python output.
    - Do NOT change constant values or the intended business logic; match the ORIGINAL
      Python behaviour so the outputs are equal.
    - Data-generation functions (numpy / `random` / seeded generators) must build data in
      PLAIN Python and use `spark.createDataFrame(...)`; do NOT translate seeded random
      generation into `F.rand()` column expressions.
    - Keep the corrected function SEMS-compliant per the conventions above (typed signature,
      docstring, comments on non-trivial logic, `logging` not `print`, specific `try/except`
      never bare, values from `CONFIG`, no dead/commented-out code or TODOs).
    - The success signal for execute_pyspark_script_tool is the `success` field
      (`status == "SUCCESS"` on the Databricks run). Do NOT re-run when success is true.
      A `status` of "TIMEOUT" means the Databricks run did not finish in time — that is
      an infrastructure result, not evidence that a function is wrong.
    """,

    tools=[
        my_skill_toolset,
        read_functions_tool,
        read_source_functions_tool,
        read_converted_file_tool,
        replace_functions_tool,
        execute_pyspark_script_tool,
    ],

    mode="task",
    output_key="semantic_code_fixer_output",
    before_agent_callback=seed_conventions,
)

def _function_spans(tree: ast.Module) -> List[tuple]:
    """(start_line, end_line, name) for every function/class def in the module."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            spans.append((start, end, node.name))
    return spans
 
 
def _enclosing_function(spans: List[tuple], line_no: Optional[int]) -> Optional[str]:
    """The INNERMOST def/class containing line_no (smallest span wins)."""
    if line_no is None:
        return None
    candidates = [(end - start, name) for start, end, name in spans if start <= line_no <= end]
    if not candidates:
        return None
    return min(candidates)[0] and min(candidates)[1] or min(candidates)[1]


#: Bucket size for sems_fix_agent — mirrors BATCH_SIZE above. Handing the
#: model every blocking gap in one tool response risks the same MAX_TOKENS /
#: per-minute rate-limit blowout that batching solves for the conversion loop
#: (see code_convertor_agent's docstring history), except here the "size" of
#: one item is a whole code_context snippet rather than a function body.
SEMS_GAP_BATCH_SIZE = 5

#: Ceiling on code_context + description characters in one bucket. A bucket
#: is capped by whichever limit binds first, same reasoning as
#: BATCH_CHAR_BUDGET: one gap's context can dwarf several others'.
SEMS_GAP_BATCH_CHAR_BUDGET = 3000


def _gap_key(gap: Dict[str, Any]) -> str:
    """Stable identity for a gap within one analysis pass: rule + location."""
    return f"{gap.get('rule_id', '')}|{gap.get('location', '')}"


def _render_sems_bucket(bucket: List[Dict[str, Any]]) -> str:
    """Format one bucket of blocking gaps as plain text for direct prompt
    injection — mirrors _next_batch_source's approach in the conversion loop:
    a bounded batch is computed once and injected straight into the prompt,
    rather than the model spending a tool round-trip to fetch it."""
    if not bucket:
        return "(no blocking gaps in this bucket — nothing to fix)"
    parts = []
    for i, g in enumerate(bucket, 1):
        parts.append(
            f"### Gap {i}: [{g.get('severity', '')}] {g.get('rule_id', '')}"
            f" — {g.get('enclosing_function') or 'module level'}\n"
            f"Location: {g.get('location', '')}\n"
            f"Description: {g.get('description', '')}\n"
            f"Remediation hint: {g.get('remediation', '')}\n"
            "Code context:\n"
            f"{g.get('code_context') or '(no code context — environment issue; see remediation hint)'}"
        )
    return "\n\n".join(parts)


def _scoped_rule_guidance(bucket: List[Dict[str, Any]]) -> str:
    """Full-catalog guidance (severity/category/description/remediation) for
    ONLY the rule_ids present in `bucket`, instead of the whole SEMS rule
    catalog (SKILL.md + references — ~13k chars covering ~50 rule IDs). A
    5-gap bucket typically cites 2-3 distinct rule_ids, so injecting the full
    catalog on every model call in the turn was mostly dead weight — this
    looks up just the rules this bucket actually needs from the same
    machine-readable rules.yaml the compliance checker itself enforces
    against, so it can never drift out of sync with the real rule set.

    Rule IDs with no rules.yaml entry (third-party tool codes like F401/E0602
    from flake8/pylint/mypy/bandit) are silently skipped — those gaps' own
    `description`/`remediation` fields (already in the rendered bucket) are
    the only guidance that exists for them.
    """
    # Deferred import: subagents.sems_agent's package __init__ reaches this
    # module too (sems_correction_loop_agent -> code_converter), so importing
    # at module load time would be circular.
    from ...sems_agent.compliance.compliance_checker import RULE_CATALOG

    rule_ids = sorted({g.get("rule_id") for g in bucket if g.get("rule_id")})
    lines = []
    for rid in rule_ids:
        entry = RULE_CATALOG.get(rid)
        if not entry:
            continue
        remediation = entry.get("remediation") or entry.get("fix_hint") or ""
        lines.append(
            f"- {rid} ({entry.get('severity', '')}, {entry.get('category', '')}): "
            f"{entry.get('description', '')} — {remediation}".strip()
        )
    if not lines:
        return "(none of this bucket's rule IDs are in the SEMS catalog — rely on each gap's own remediation hint above.)"
    return "\n".join(lines)


#: Fixed priority order for resolving which BRD area owns a target key when
#: gaps from more than one area land on the same function/constant/import
#: block in one round (see _assign_gap_owners). Mirrors BRD_AREAS in
#: compliance/sems_validator.py — duplicated as string literals rather than
#: imported, for the same circular-import reason _scoped_rule_guidance
#: defers its RULE_CATALOG import.
_AREA_PRIORITY = (
    "security",
    "error_handling",
    "modular_design",
    "logging_practices",
    "code_readability_and_structure",
)

_AREA_LABEL_FOR_PROMPT = {
    "security": "Security",
    "error_handling": "Error Handling",
    "modular_design": "Modular Design / Code Structure",
    "logging_practices": "Logging Practices",
    "code_readability_and_structure": "Code Readability & Structure",
}


def _target_key_for_gap(gap: Dict[str, Any]) -> str:
    """Stable identity for the CODE a gap's fix would touch — the unit
    ownership is assigned over (see _assign_gap_owners). Two gaps sharing a
    target key must be owned by the same area, or one area's full-function
    rewrite would silently discard the other's fix to the same function."""
    enclosing = gap.get("enclosing_function")
    if enclosing:
        return f"func:{enclosing}"
    if gap.get("rule_id") in {"F401", "W0611"}:
        return "__imports__"
    return "__module__"


def _assign_gap_owners(gaps: List[Dict[str, Any]]) -> Dict[str, str]:
    """One owning BRD area per target key: the highest-_AREA_PRIORITY area
    among the areas of every gap that shares that key.

    This is what makes the 5 parallel per-area fixers conflict-free BY
    CONSTRUCTION rather than by detecting conflicts after the fact: a
    function with both a security gap and a logging gap is entirely owned by
    security this round (fixing both), so the logging fixer never proposes a
    stale-based rewrite of the same function. Pure function of `gaps` — every
    one of the 5 per-area seed callbacks (_make_seed_sems_fix_bucket) computes
    the identical mapping independently, from the same on-disk gap snapshot,
    so there is nothing to coordinate.
    """
    owners: Dict[str, str] = {}
    for g in gaps:
        area = g.get("brd_area")
        if area not in _AREA_PRIORITY:
            continue
        key = _target_key_for_gap(g)
        current = owners.get(key)
        if current is None or _AREA_PRIORITY.index(area) < _AREA_PRIORITY.index(current):
            owners[key] = area
    return owners


def _make_seed_sems_fix_bucket(
    area: str,
) -> Callable[[CallbackContext], Optional[genai_types.Content]]:
    """Factory for one BRD area's before_agent_callback — see
    _seed_sems_fix_bucket's original single-fixer version (now replaced) for
    the batching rationale, unchanged here beyond partitioning by owning area.

    Returns a genai_types.Content when this area's bucket is empty, which
    short-circuits the agent's turn before any LLM call is made (see the
    empty-bucket branch below) — otherwise returns None, letting the agent's
    own turn proceed normally.
    """

    def _seed(callback_context: CallbackContext) -> Optional[genai_types.Content]:
        # Deferred import: subagents.sems_agent's package __init__ reaches this
        # module too (sems_correction_loop_agent -> code_converter), so importing
        # at module load time would be circular.
        from ...sems_agent.sems_gap_analyzer_agent import (
            GAP_BATCH_CURSOR_FILE,
            _MANUAL_ACTION_ONLY_RULES,
            _line_no_from_location,
            read_gap_state,
        )

        state = callback_context.state

        gap_items: List[Dict[str, Any]] = read_gap_state().get("gaps", [])
        # gap_items already carries "is_blocking" (see _gap_is_blocking in
        # sems_gap_analyzer_agent.py) — reuse it instead of re-deriving blocking
        # status from severity text, which would miss SECURITY_HOTSPOT findings
        # that bandit reports at sonar severity MINOR.
        #
        # _MANUAL_ACTION_ONLY_RULES (e.g. NO_TESTS) gaps require creating a NEW
        # file no fixer tool can create — excluded here so no area is ever
        # handed one to "solve" (see that constant's docstring for the
        # incident where a fixer improvised by pasting a test suite + a
        # self-import into the production file instead).
        blocking = [
            g for g in gap_items
            if g.get("is_blocking") and g.get("rule_id") not in _MANUAL_ACTION_ONLY_RULES
        ]

        converted_path_str = state.get("converted_pyspark_file_path") or state.get("converted_file_path")
        spans: List[tuple] = []
        if converted_path_str:
            p = pathlib.Path(converted_path_str)
            if p.exists():
                try:
                    spans = _function_spans(ast.parse(p.read_text(encoding="utf-8")))
                except SyntaxError:
                    spans = []

        enriched = [
            {**g, "enclosing_function": _enclosing_function(spans, _line_no_from_location(g.get("location", "")))}
            for g in blocking
        ]

        # Recomputed fresh from the same on-disk snapshot every one of the 5
        # areas' callbacks reads — see _assign_gap_owners's docstring for why
        # this needs no cross-area coordination.
        owners = _assign_gap_owners(enriched)
        owned = [g for g in enriched if owners.get(_target_key_for_gap(g)) == area]

        try:
            served = set(json.loads(GAP_BATCH_CURSOR_FILE.read_text(encoding="utf-8")).get("served", []))
        except (OSError, ValueError):
            served = set()

        unserved = [g for g in owned if _gap_key(g) not in served]

        bucket: List[Dict[str, Any]] = []
        used_chars = 0
        for g in unserved[:SEMS_GAP_BATCH_SIZE]:
            size = len(g.get("code_context") or "") + len(g.get("description") or "")
            # The first gap is always taken even if it alone busts the budget,
            # otherwise an oversized code_context would never be offered and the
            # fixer would spin on it forever.
            if bucket and used_chars + size > SEMS_GAP_BATCH_CHAR_BUDGET:
                break
            bucket.append(g)
            used_chars += size

        # A flat cursor set shared across all 5 areas is safe: `owned` is
        # already area-exclusive (see _assign_gap_owners), so a gap key
        # marked served here can never collide with another area's bucket —
        # it only ever prevents THIS area's own leftovers from being re-served
        # within the same round.
        served |= {_gap_key(g) for g in bucket}
        GAP_BATCH_CURSOR_FILE.write_text(json.dumps({"served": sorted(served)}), encoding="utf-8")

        state[f"sems_fix_bucket_count__{area}"] = len(bucket)
        state[f"sems_fix_remaining_count__{area}"] = len(unserved) - len(bucket)
        state[f"sems_fix_total_blocking__{area}"] = len(owned)
        state[f"sems_fix_bucket__{area}"] = _render_sems_bucket(bucket)
        state[f"sems_fix_rule_guidance__{area}"] = _scoped_rule_guidance(bucket)
        # Reset fresh every round so a no-op round (nothing owned, or nothing
        # proposed) never leaves _apply_sems_fix_patches replaying a stale
        # patch from an earlier loop iteration.
        state[f"sems_fix_patches__{area}"] = {"functions_code": [], "removed_import_names": []}

        if not bucket:
            # No owned gaps this round — skip the model call entirely instead
            # of spending a full turn just to say "nothing to fix". Returning
            # content from before_agent_callback sets ctx.end_invocation=True
            # (see google/adk/agents/base_agent.py's _handle_before_agent_callback),
            # so _run_async_impl (the LLM call) never runs. With 5 areas
            # dispatched concurrently under sems_gap_parallel_fix_agent, and
            # most rounds having blocking gaps in only 1-3 of the 5
            # categories, this cuts real concurrent load against the
            # Databricks endpoint down to just the areas with actual work —
            # unnecessary concurrent calls were a plausible contributor to
            # observed MODEL_RETURNED_NO_CONTENT failures under load.
            note = f"No blocking gaps owned by {_AREA_LABEL_FOR_PROMPT[area]} this round — nothing to fix."
            state[f"sems_fix_output_{area}"] = note
            return genai_types.Content(role="model", parts=[genai_types.Part(text=note)])

        return None

    return _seed


def _make_propose_function_patch_tool(area: str):
    """Factory for one area's propose_function_patch_tool — a distinct
    closure per area (capturing `area`) rather than one tool reading a shared
    "current area" from state, so nothing depends on call ordering between
    the 5 parallel fixers."""

    def propose_function_patch_tool(context: ToolContext, functions_code: str) -> dict:
        """Submit corrected function(s)/constant(s) for THIS category's bucket.

        Unlike replace_functions_tool, this does NOT write to disk — it queues
        the patch so a single integration step (_apply_sems_fix_patches) can
        apply every category's patches after all 5 areas finish this round,
        which is what prevents two areas from overwriting the same function
        based on a stale read of each other's in-flight edits. Submit ONLY the
        corrected function(s)/constant(s) you are fixing (plus any
        newly-needed import) — exactly the same shape replace_functions_tool
        expects.
        """
        key = f"sems_fix_patches__{area}"
        patches = dict(context.state.get(key) or {"functions_code": [], "removed_import_names": []})
        patches["functions_code"] = [*patches.get("functions_code", []), functions_code]
        context.state[key] = patches
        return {"status": "queued", "area": area}

    return propose_function_patch_tool


def _make_propose_import_removal_tool(area: str):
    """Factory for one area's propose_import_removal_tool — see
    _make_propose_function_patch_tool for why this queues instead of writing
    straight to disk."""

    def propose_import_removal_tool(context: ToolContext, names: list[str]) -> dict:
        """Queue unused import name(s) for removal — see
        remove_unused_imports_tool for exactly what `names` means (bound
        names, not module paths). Applied by the integration step after all
        5 categories finish this round, not written to disk immediately.
        """
        key = f"sems_fix_patches__{area}"
        patches = dict(context.state.get(key) or {"functions_code": [], "removed_import_names": []})
        patches["removed_import_names"] = [*patches.get("removed_import_names", []), *(names or [])]
        context.state[key] = patches
        return {"status": "queued", "area": area, "names": list(names or [])}

    return propose_import_removal_tool


def _apply_sems_fix_patches(callback_context: CallbackContext) -> None:
    """Integration step: after all 5 per-area fixers finish (this is
    sems_gap_parallel_fix_agent's after_agent_callback), apply every area's
    queued patches to the converted file, one area at a time, in
    _AREA_PRIORITY order.

    Deliberately plain Python, not another LLM call: merging is already
    100% mechanical via _apply_function_patch/_apply_import_removal — the
    exact same code replace_functions_tool/remove_unused_imports_tool use —
    so there is no judgment call left for a model to make. Sequential
    application is safe because _assign_gap_owners already guarantees no two
    areas' function-code patches target the same function/constant this
    round; import-name removals are naturally safe to apply in any order
    (removing an already-absent name is just reported as not_found).
    """
    state = callback_context.state
    converted_path_str = state.get("converted_pyspark_file_path") or state.get("converted_file_path")
    if not converted_path_str:
        return None
    p = pathlib.Path(converted_path_str)
    if not p.exists():
        return None

    errors: List[str] = []
    for area in _AREA_PRIORITY:
        patches = state.get(f"sems_fix_patches__{area}") or {}
        for functions_code in patches.get("functions_code", []):
            try:
                _apply_function_patch(p, functions_code)
            except SyntaxError as exc:
                errors.append(f"{area}: function patch syntax error: {exc}")
        import_names = patches.get("removed_import_names") or []
        if import_names:
            try:
                _apply_import_removal(p, import_names)
            except SyntaxError as exc:
                errors.append(f"{area}: import removal syntax error: {exc}")

    if errors:
        state["sems_fix_patch_errors"] = errors
    return None


_SEMS_FIX_INSTRUCTION_TEMPLATE = """You are an expert PySpark engineer. A converted PySpark pipeline exists on
    disk, but the SEMS gap analyzer just flagged BLOCKING-severity compliance gaps in it. You
    own ONLY the **%LABEL%** category this round — 4 other fixers are running at the same time
    for the other categories, each owning a disjoint set of gaps (see _assign_gap_owners), so
    you never need to worry about another fixer touching the same function/constant/import
    block this round.

    THIS TURN'S BUCKET — already selected for you; do NOT fetch it via a tool, it is already
    below ({sems_fix_bucket_count__%AREA%} gap(s) in this bucket, {sems_fix_remaining_count__%AREA%}
    still queued for later loop iterations, {sems_fix_total_blocking__%AREA%} blocking gaps
    owned by %LABEL% this pass):
    <bucket>
    {sems_fix_bucket__%AREA%}
    </bucket>

    Rule guidance for ONLY this bucket's rule IDs (severity, category, description, full
    remediation — not just the gap's own one-line hint):
    <sems_rule_guidance>
    {sems_fix_rule_guidance__%AREA%}
    </sems_rule_guidance>

    HOW TO WORK (surgical, this ONE bucket, then stop — follow exactly):
    1. If <bucket> says "no blocking gaps in this bucket", make NO changes and stop — there
       is nothing to fix.
    2. For each gap in <bucket>, read its rule, description, remediation hint, code context
       (the real flagged lines), and enclosing function — plus the matching entry (if any)
       in <sems_rule_guidance> for the fuller rationale.
    3. Group the bucket's gaps by enclosing function so you fix each function once even if
       it has multiple gaps. For gaps WITH an enclosing function: call
       **read_functions_tool** with that name to get its current source. For gaps with no
       enclosing function that are an UNUSED-IMPORT finding (rule F401 or W0611 —
       "imported but unused"): do NOT try to read or edit the file for these, go straight to
       **propose_import_removal_tool** in step 5. For any OTHER gap with no enclosing function
       (e.g. a hardcoded credential in a top-level constant): call
       **read_converted_file_tool** to see the current module-level code around it. Skip
       gaps whose rule is TOOL_MISSING or TOOL_ERROR — those flag a missing or crashing
       external analysis tool, not a defect in the code; there is nothing to edit.
    4. Fix each flagged issue per its remediation hint, addressing the SEMS/STAM rationale
       (e.g. move a hardcoded secret to `CONFIG`/an env var, replace `eval`/`exec`, mask a PII
       column before it's written or logged, replace a bare `except:` with a specific one,
       replace `print` with `logging`, add the missing docstring/type hints) — consult
       <sems_rule_guidance> above for the rule rationale, not just the gap's short
       remediation hint. Do NOT change the function's underlying business logic, column
       names, or constant values — fix ONLY what the gap flags.
    5. Apply the fixes — as PROPOSALS, not direct edits (a separate integration step applies
       every category's proposals to disk once all 5 categories finish this round):
       - For UNUSED-IMPORT gaps (F401/W0611): call
         **propose_import_removal_tool(names=[...])** with the exact bound name(s) each
         gap's description/code context flags as unused (e.g. "date", "math", "np") —
         never the module path.
       - For everything else: call **propose_function_patch_tool(functions_code=...)** with
         ONLY the corrected function(s)/constant(s) (plus any newly-needed import) — NEVER
         paste the whole file, NEVER re-send code you did not change, NEVER add module-level
         calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
    6. STOP once you have proposed fixes for every gap in this ONE bucket, even if
       {sems_fix_remaining_count__%AREA%} is greater than 0 — there is no tool to fetch
       another bucket this turn. The gap analyzer re-runs the full check next loop
       iteration and this callback selects the next bucket fresh, across as many loop
       iterations as it takes.

    **STRICT RULES**
    - Fix ONLY what each gap flags. Do NOT refactor, rename, or "improve" unrelated code.
    - Do NOT change business logic, column names, or constant VALUES — only what makes the
      flagged line SEMS-compliant (e.g. reading a secret from `CONFIG`/env instead of a
      literal is fine; changing what the pipeline computes is not).
    - Never invent a fix for a gap that is not in <bucket> above.
    - Never propose a fix for a function/constant that isn't in your own bucket, even if you
      notice an unrelated gap elsewhere — another category owns it this round.
    - If a tool call reports it made no change (e.g. `status: "no_change"` or 0
      functions/imports affected), do NOT retry the same gap with a different
      creative workaround — move on to the next gap in the bucket, or stop if that was
      the last one.
    - You do NOT re-run the SEMS gap analysis yourself.
    - You do not execute the script on Databricks - that only happens once, at the very
      end of the SEMS stage (final_execution_check_agent), to save cluster cost.
    """


def _make_sems_fix_agent(area: str) -> Agent:
    """Factory for one BRD area's SEMS fixer. See sems_gap_parallel_fix_agent
    for why 5 of these run concurrently under a ParallelAgent instead of one
    sequential fixer, and _assign_gap_owners for why that is conflict-free.
    """
    instruction = (
        _SEMS_FIX_INSTRUCTION_TEMPLATE
        .replace("%AREA%", area)
        .replace("%LABEL%", _AREA_LABEL_FOR_PROMPT[area])
    )
    return Agent(
        name=f"sems_fix_agent__{area}",
        model=LiteLlm(
            model="databricks/databricks-claude-sonnet-4-6",
            max_tokens=4096,
        ),
        instruction=instruction,
        tools=[
            read_functions_tool,
            read_converted_file_tool,
            _make_propose_function_patch_tool(area),
            _make_propose_import_removal_tool(area),
        ],
        mode="task",
        # Same rationale as the original single-fixer agent: every tool here
        # reads disk (read-only) or writes only to this area's own state
        # slice — nothing depends on earlier loop iterations or on the other
        # 4 areas' turns this round.
        include_contents="none",
        before_agent_callback=_make_seed_sems_fix_bucket(area),
        output_key=f"sems_fix_output_{area}",
    )


sems_gap_parallel_fix_agent = ParallelAgent(
    name="sems_gap_parallel_fix_agent",
    description=(
        "Runs one SEMS fixer per BRD area (security, error handling, modular "
        "design, logging, readability) in parallel, each restricted to its "
        "own conflict-free slice of blocking gaps (see _assign_gap_owners), "
        "then applies every area's proposed patches to disk in one "
        "deterministic integration pass (_apply_sems_fix_patches) once all 5 "
        "finish."
    ),
    sub_agents=[_make_sems_fix_agent(area) for area in _AREA_PRIORITY],
    after_agent_callback=_apply_sems_fix_patches,
)

def build_code_fixer_agent(name: str = "code_fixer_agent"):
    """Repairs the converted module against a FAILING pytest parity suite.

    Reads `pytest_last_result` — a dict written by the parity agent listing which
    tests failed and why — and edits only the functions it names, via
    replace_functions_tool. It never touches the tests: a test that correctly
    encodes the source behaviour and fails is evidence the conversion is wrong,
    so letting the fixer "solve" it by weakening the test would hide the bug it
    exists to catch.

    A factory for the same reason the parity agents are: ADK stamps
    `parent_agent` onto every `sub_agents` entry, and the parity loop is built
    twice (pipeline stage and standalone app), so a shared instance would be
    re-parented by whichever was constructed last.
    """
    return Agent(
        name=name,
        model = LiteLlm(
            model="databricks/databricks-claude-opus-4-7",
        ),
        instruction= """You are an expert PySpark engineer. A converted PySpark pipeline
        already exists on disk (it may contain dozens of functions), but its pytest parity
        suite is FAILING. Fix ONLY the converted code — you do NOT edit the tests, and you
        fix ONLY the functions that are actually failing, one small batch at a time.

        MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
        for native Spark rules. Do not reproduce the full skill/reference corpus in context.
        Never introduce pandas idioms (`pd.`, `.merge`, `.rename(columns=)`, `.iloc`, `df.apply`)
        or numpy column-building patterns into the corrected code.

        Failing tests, as structured data. `failed_tests[].test` is
        `test_<function_name>`, so strip the `test_` prefix to get the function to
        fix. `run_error` appears instead when the run itself broke (import error,
        cluster problem) rather than any individual test failing:
        <pytest_result>
        {pytest_last_result}
        </pytest_result>

        HOW TO WORK (surgical, batched — follow exactly):
        1. From <pytest_result>, list the FAILING functions (strip the `test_` prefix from
           each failing test name). If nothing is failing, make NO changes and stop.
        2. Take the first few failing functions (up to 4). Call **read_functions_tool** with
           exactly those names to get their CURRENT (converted) source, AND
           **read_source_functions_tool** with the same names to get the ORIGINAL Python
           source. Do NOT pull the whole file.
        3. Compare the two. The original Python is the GROUND TRUTH for what the function must
           do — fix the converted function so its behaviour matches the original. If the short
           error references something that does not exist in the original (a column, a call, a
           line), that is a hallucination in the converted code — remove it.
        4. Call **replace_functions_tool(functions_code=...)** passing ONLY the corrected
           function(s) (plus any missing import/constant they need). It replaces those
           functions in place and leaves every other function untouched — NEVER paste the
           whole file, NEVER re-send functions you are not changing, NEVER add module-level
           calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
        5. Call **execute_pyspark_script_tool** ONCE as a syntax/runtime sanity check.
        6. If more failing functions remain, repeat from step 2 for the next batch. Then stop
           — the parity agent re-runs the full suite and returns any still-failing functions.

        **STRICT RULES**
        - Do NOT change the underlying business logic; match the ORIGINAL Python behaviour.
        - Do NOT change or infer column names; do NOT change any constant values.
        - Data-generation functions (e.g. those using numpy / `random` / seeded generators to
          build synthetic rows) must build the data in PLAIN Python and pass it to
          `spark.createDataFrame(...)`. Do NOT translate seeded random generation into
          `F.rand()` column expressions — that changes behaviour and is a common wrong fix.
        - Keep the corrected function SEMS-compliant per the conventions above (typed signature,
          docstring, comments on non-trivial logic, `logging` not `print`, specific `try/except`
          never bare, values from `CONFIG`, no dead/commented-out code or TODOs).
        - The success signal for execute_pyspark_script_tool is the `success` field
          (`status == "SUCCESS"` on the Databricks run). Do NOT re-run when success is true.
          A `status` of "TIMEOUT" means the Databricks run did not finish in time — that is
          an infrastructure result, not evidence that a function is wrong.
        - You do NOT run the pytest suite yourself.
        """,

        tools=[
            my_skill_toolset,
            read_functions_tool,
            read_source_functions_tool,
            replace_functions_tool,
            execute_pyspark_script_tool,
        ],

        mode="task",
        output_key="code_fixer_output",
        before_agent_callback=seed_parity_fixer,
    )
