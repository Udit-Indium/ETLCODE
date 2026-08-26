"""
SEMS (Shell Engineering Management System) compliance checker.

Performs static analysis on generated PySpark code using four layers:

  Layer 0 — Syntax validity: explicit SyntaxError → hard violation (SYN001)
  Layer 1 — python -m py_compile equivalent (handled upstream)
  Layer 2 — AST parsing: structural checks (try-body coverage, typo detection)
  Layer 3 — PySpark API allowlist: Levenshtein typo detection with suggestions

Rules are declared in rules.yaml for documentation and remediation text.
Adding a YAML entry alone does NOT activate enforcement — a Python check
function must also be added here.  Company rules (SHELL_*) are enforced
via dedicated check_* functions in the "Company rule checks" section below.
"""

import ast
import difflib
import io
import logging
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Rule catalog ───────────────────────────────────────────────────────────────

_RULES_PATH = Path(__file__).resolve().parents[1] / "rules.yaml"


def _load_rule_catalog() -> Dict[str, dict]:
    """Load the SEMS rule catalog from rules.yaml (both rules and company_rules sections)."""
    if not _YAML_AVAILABLE or not _RULES_PATH.exists():
        logger.warning(
            "SEMS rule catalog unavailable (%s) — remediation strings will be "
            "empty for all rules. Install PyYAML and verify %s exists.",
            "PyYAML not installed" if not _YAML_AVAILABLE else "file not found",
            _RULES_PATH,
        )
        return {}
    try:
        with _RULES_PATH.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        catalog = {r["id"]: r for r in data.get("rules", [])}
        catalog.update({r["id"]: r for r in data.get("company_rules", [])})
        return catalog
    except Exception as exc:
        logger.warning(
            "Could not load SEMS rule catalog from %s: %s — "
            "remediation strings will be empty for all rules.",
            _RULES_PATH,
            exc,
        )
        return {}


RULE_CATALOG: Dict[str, dict] = _load_rule_catalog()


def _remediation(rule_id: str, fallback: str = "") -> str:
    entry = RULE_CATALOG.get(rule_id, {})
    return str(entry.get("remediation") or entry.get("fix_hint") or fallback)


# ── Category weights (must sum to 1.0) ────────────────────────────────────────

CATEGORY_WEIGHTS: Dict[str, float] = {
    "syntax": 0.20,
    "security": 0.25,
    "spark_best_practices": 0.20,
    "error_handling": 0.15,
    "documentation": 0.10,
    "databricks_compatibility": 0.10,
}

_HARD_DEDUCTION = 30  # points removed per hard violation from its category
_SOFT_DEDUCTION = 15  # points removed per soft violation from its category
_REJECT_SCORE_THRESHOLD = 60  # overall_score below this also sets passed=False

# ── PySpark API allowlist (pyspark.sql.functions) ─────────────────────────────

_STATIC_PYSPARK_FUNCTIONS_ALLOWLIST: frozenset = frozenset(
    [
        "abs",
        "acos",
        "add_months",
        "aggregate",
        "any_value",
        "approx_count_distinct",
        "approxCountDistinct",
        "array",
        "array_contains",
        "array_distinct",
        "array_except",
        "array_intersect",
        "array_join",
        "array_max",
        "array_min",
        "array_position",
        "array_remove",
        "array_repeat",
        "array_sort",
        "array_union",
        "arrays_overlap",
        "arrays_zip",
        "asc",
        "asc_nulls_first",
        "asc_nulls_last",
        "ascii",
        "asin",
        "atan",
        "atan2",
        "avg",
        "base64",
        "bin",
        "bitwiseNOT",
        "broadcast",
        "bround",
        "cbrt",
        "ceil",
        "coalesce",
        "col",
        "collect_list",
        "collect_set",
        "column",
        "concat",
        "concat_ws",
        "conv",
        "corr",
        "cos",
        "cosh",
        "count",
        "countDistinct",
        "covar_pop",
        "covar_samp",
        "create_map",
        "crc32",
        "cume_dist",
        "current_date",
        "current_timestamp",
        "current_user",
        "date_add",
        "date_diff",
        "date_format",
        "date_sub",
        "date_trunc",
        "datediff",
        "dayofmonth",
        "dayofweek",
        "dayofyear",
        "decode",
        "degrees",
        "dense_rank",
        "desc",
        "desc_nulls_first",
        "desc_nulls_last",
        "element_at",
        "encode",
        "exists",
        "exp",
        "explode",
        "explode_outer",
        "expm1",
        "expr",
        "factorial",
        "filter",
        "first",
        "flatten",
        "floor",
        "forall",
        "format_number",
        "format_string",
        "from_csv",
        "from_json",
        "from_unixtime",
        "from_utc_timestamp",
        "get_json_object",
        "greatest",
        "grouping",
        "grouping_id",
        "hash",
        "hex",
        "hour",
        "hypot",
        "initcap",
        "input_file_name",
        "instr",
        "isnan",
        "isnull",
        "json_tuple",
        "kurtosis",
        "lag",
        "last",
        "last_day",
        "lead",
        "least",
        "length",
        "levenshtein",
        "lit",
        "locate",
        "log",
        "log10",
        "log1p",
        "log2",
        "lower",
        "lpad",
        "ltrim",
        "map_concat",
        "map_entries",
        "map_filter",
        "map_from_arrays",
        "map_from_entries",
        "map_keys",
        "map_values",
        "map_zip_with",
        "max",
        "mean",
        "median",
        "min",
        "minute",
        "mode",
        "monotonically_increasing_id",
        "month",
        "months_between",
        "named_struct",
        "nanvl",
        "next_day",
        "nth_value",
        "ntile",
        "overlay",
        "pandas_udf",
        "percent_rank",
        "percentile_approx",
        "posexplode",
        "posexplode_outer",
        "pow",
        "product",
        "quarter",
        "radians",
        "raise_error",
        "rand",
        "randn",
        "rank",
        "regexp_extract",
        "regexp_replace",
        "repeat",
        "reverse",
        "rint",
        "round",
        "row_number",
        "rpad",
        "rtrim",
        "schema_of_csv",
        "schema_of_json",
        "second",
        "sequence",
        "sha1",
        "sha2",
        "shiftLeft",
        "shiftRight",
        "shiftRightUnsigned",
        "shuffle",
        "signum",
        "sin",
        "sinh",
        "size",
        "skewness",
        "slice",
        "sort_array",
        "soundex",
        "spark_partition_id",
        "split",
        "sqrt",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "struct",
        "substring",
        "substring_index",
        "sum",
        "sum_distinct",
        "sumDistinct",
        "tan",
        "tanh",
        "timestamp_seconds",
        "to_date",
        "to_json",
        "to_timestamp",
        "to_utc_timestamp",
        "transform",
        "transform_keys",
        "transform_values",
        "translate",
        "trim",
        "trunc",
        "udf",
        "unbase64",
        "unhex",
        "unix_timestamp",
        "upper",
        "var_pop",
        "var_samp",
        "variance",
        "weekofyear",
        "when",
        "window",
        "xxhash64",
        "year",
        "zip_with",
    ]
)

try:
    import pyspark.sql.functions as _pyspark_functions_module

    _LIVE_PYSPARK_NAMES: frozenset = frozenset(
        n for n in dir(_pyspark_functions_module) if not n.startswith("_")
    )
except ImportError:
    _LIVE_PYSPARK_NAMES = frozenset()

PYSPARK_FUNCTIONS_ALLOWLIST: frozenset = (
    _STATIC_PYSPARK_FUNCTIONS_ALLOWLIST | _LIVE_PYSPARK_NAMES
)

# ── Regex patterns ────────────────────────────────────────────────────────────

# subprocess: from-import, __import__(), importlib.import_module()
# Multi-import ("import os, subprocess") and direct usage are caught via AST below.
SUBPROCESS_IMPORT_PATTERN = re.compile(
    r"\bfrom\s+subprocess\s+import\b"
    r"|__import__\s*\(\s*['\"]subprocess['\"]\s*\)"
    r"|importlib\.import_module\s*\(\s*['\"]subprocess['\"]\s*\)"
)
EVAL_EXEC_PATTERN = re.compile(r"\b(eval|exec)\s*\(")
_LOOP_HEADER_PATTERN = re.compile(r"^(\s*)(?:for|while)\b.*:\s*$")
_DBUTILS_FS_PATTERN = re.compile(r"dbutils\.fs\.")

PII_KEYWORDS = (
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "email",
    "ssn",
    "credit_card",
    "creditcard",
    "phone",
    "date_of_birth",
    "full_name",
)
# Matches logger.xxx(...), logging.xxx(...), and print(...) containing PII keywords.
_PII_LOG_PATTERN = re.compile(
    r"(?:logger\.\w+|logging\.(?:info|warning|error|debug|critical)|print)\s*\([^)]*\b("
    + "|".join(PII_KEYWORDS)
    + r")\b",
    re.IGNORECASE,
)

# Local path prefixes that should not appear in Databricks workloads
_LOCAL_PATH_PREFIXES: Tuple[str, ...] = (
    "/home/",
    "/tmp/",
    "/var/",
    "/Workspace/",
    "C:\\",
    "C:/",
)

# PII column names that must be masked before Delta writes
_PII_COLUMN_NAMES: Tuple[str, ...] = (
    "email",
    "phone",
    "phone_number",
    "national_id",
    "ssn",
    "date_of_birth",
)

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class SEMSViolation:
    """A single structured SEMS rule violation with remediation guidance."""

    rule_id: str
    severity: str  # "hard" | "soft"
    category: str
    message: str
    remediation: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    section_index: Optional[int] = None  # populated by check_script_compliance


@dataclass
class ComplianceResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structured_violations: List[SEMSViolation] = field(default_factory=list)

    def summary(self) -> str:
        status = "COMPLIANT" if self.passed else "NON-COMPLIANT"
        lines = [f"SEMS Compliance: {status}"]
        if self.violations:
            lines.append("  Violations (must fix):")
            lines.extend(f"    [FAIL] {v}" for v in self.violations)
        if self.warnings:
            lines.append("  Warnings (should fix):")
            lines.extend(f"    [WARN] {w}" for w in self.warnings)
        return "\n".join(lines)

    def category_scores(self) -> Dict[str, int]:
        """Compute per-category scores (0-100) from structured violations."""
        scores: Dict[str, int] = {cat: 100 for cat in CATEGORY_WEIGHTS}
        for sv in self.structured_violations:
            cat = sv.category
            if cat not in scores:
                continue
            deduction = _HARD_DEDUCTION if sv.severity == "hard" else _SOFT_DEDUCTION
            scores[cat] = max(0, scores[cat] - deduction)
        return scores

    def overall_score(self) -> int:
        """Weighted average of category scores, rounded to nearest integer."""
        scores = self.category_scores()
        return round(sum(scores[cat] * w for cat, w in CATEGORY_WEIGHTS.items()))


# ── AST helpers ───────────────────────────────────────────────────────────────


def _build_parent_map(tree: ast.AST) -> Dict[int, ast.AST]:
    """Return a map of id(child) → parent for every node in the tree."""
    parent_map: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    return parent_map


def _levenshtein(a: str, b: str) -> int:
    """Compute true Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _chain_contains(attr_node: ast.Attribute, names: frozenset) -> bool:
    """Return True if any identifier in the attribute chain is in *names*."""
    val: ast.expr = attr_node.value
    while isinstance(val, ast.Attribute):
        if val.attr in names:
            return True
        val = val.value
    return isinstance(val, ast.Name) and val.id in names


def _code_only(source: str) -> str:
    """Return source with # comment text replaced by spaces, preserving positions."""
    result = list(source)
    try:
        lines = source.splitlines(keepends=True)
        for tok_type, _, (srow, scol), (erow, ecol), _ in tokenize.generate_tokens(
            io.StringIO(source).readline
        ):
            if tok_type != tokenize.COMMENT:
                continue
            # Comments are always single-line; replace with spaces.
            line_start = sum(len(lines[i]) for i in range(srow - 1))
            # Erase from comment start to end of non-newline content on that line.
            line_content_end = line_start + len(lines[srow - 1].rstrip("\r\n"))
            for i in range(line_start + scol, min(line_content_end, len(result))):
                result[i] = " "
    except tokenize.TokenError:
        pass
    return "".join(result)


def _is_in_try_body(node: ast.AST, parent_map: Dict[int, ast.AST]) -> bool:
    """Return True only when node is directly inside a try *body* (not except/else/finally).

    Stops at function/class scope boundaries: a Spark action inside a nested
    function defined in a try block is NOT considered protected — the function
    may be called later from outside any try.
    """
    current: ast.AST = node
    while id(current) in parent_map:
        parent = parent_map[id(current)]
        # Crossed a scope boundary — any enclosing try does not protect calls
        # made when the function/class is actually executed outside that try.
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(parent, ast.ExceptHandler):
            return False
        if isinstance(parent, ast.Try):
            return current in parent.body
        current = parent
    return False


def _is_positive_int_literal(arg: ast.expr) -> Optional[bool]:
    """
    Return True  if arg is definitely a positive integer literal (1, 2, …),
           False if definitely non-positive (0, -N, …),
           None  if uncertain (variable, expression).
    """
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
        return arg.value > 0
    # ast represents -1 as UnaryOp(USub, Constant(1)); always non-positive.
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
        return False
    return None  # variable or expression: assume caller's choice is intentional


def _has_limit_in_chain(call_node: ast.Call) -> bool:
    """
    Return True when the DataFrame chain preceding .collect() already has a
    .limit(N>0) or .take(N) call, making the collection safe.

    Note: .first() and .head() are intentionally excluded — they return a
    Row/list rather than a DataFrame, so chaining .collect() after them
    would fail at runtime and does not constitute a valid guard.
    """
    func = call_node.func
    if not isinstance(func, ast.Attribute):
        return False
    value: Optional[ast.expr] = func.value
    while isinstance(value, ast.Call):
        inner_func = value.func
        if not isinstance(inner_func, ast.Attribute):
            break
        attr = inner_func.attr
        if attr in ("limit", "take"):
            if attr == "limit" and value.args:
                positivity = _is_positive_int_literal(value.args[0])
                if positivity is False:
                    # Definitely non-positive: not a valid guard; keep walking chain.
                    value = inner_func.value
                    continue
            return True  # positive, .take(), or non-literal argument
        value = inner_func.value if isinstance(inner_func, ast.Attribute) else None
        if value is None:
            break
    return False


def _get_functions_alias_info(
    tree: ast.AST,
) -> Tuple[Optional[str], Set[str], bool]:
    """
    Resolve the local alias for pyspark.sql.functions.

    Returns (alias, directly_imported_names, alias_reassigned) where:
      alias                : module alias (e.g. 'F'), or None
      directly_imported_names : names from direct imports ('from ... import col')
      alias_reassigned     : True if the alias name is later re-bound to something else

    Returns (None, set(), False) for star imports — all typo checks are skipped.
    """
    alias: Optional[str] = None
    directly_imported: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "pyspark.sql.functions":
                for a in node.names:
                    if a.name == "*":
                        return None, set(), False
                    directly_imported.add(a.asname or a.name)
            elif module in ("pyspark.sql", "pyspark"):
                for a in node.names:
                    if a.name == "functions":
                        alias = a.asname or a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if (a.name or "") == "pyspark.sql.functions" and a.asname:
                    # Bare `import pyspark.sql.functions` binds only `pyspark`,
                    # not `functions` — usage is fully dotted and not tracked
                    # by this check, so only an explicit asname sets the alias.
                    alias = a.asname

    # Detect reassignment of the module alias (conservative: skip check if any exists).
    alias_reassigned = False
    if alias is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == alias:
                        alias_reassigned = True
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                target = node.target  # type: ignore[union-attr]
                if isinstance(target, ast.Name) and target.id == alias:
                    alias_reassigned = True

    return alias, directly_imported, alias_reassigned


def _format_except_type(node_type: Optional[ast.expr]) -> str:
    """Render an ExceptHandler type node as a short human-readable string."""
    if node_type is None:
        return "bare 'except:'"
    try:
        return f"'except {ast.unparse(node_type)}:'"  # Python 3.9+
    except AttributeError:
        if isinstance(node_type, ast.Name):
            return f"'except {node_type.id}:'"
        return "'except <broad handler>:'"


# ── AST-based check functions ─────────────────────────────────────────────────


def check_syntax_validity(source_code: str) -> List[SEMSViolation]:
    """SYN001: Hard violation if source_code cannot be parsed by ast.parse()."""
    try:
        ast.parse(source_code)
        return []
    except SyntaxError as exc:
        return [
            SEMSViolation(
                rule_id="SYN001",
                severity="hard",
                category="syntax",
                message=f"[SYN001] SyntaxError: {exc.msg} (line {exc.lineno})",
                remediation="Fix the Python syntax error before applying SEMS checks.",
                line_number=exc.lineno,
            )
        ]


def check_unbounded_collect(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK001: Flag .collect() calls that have no .limit(N>0) or .take(N) guard.

    Limitation: this check is type-blind — it flags any zero-arg .collect() call
    regardless of whether the object is a Spark DataFrame.  Dataflow patterns such
    as `small_df = df.limit(100); small_df.collect()` are NOT detected as safe
    because the limit appears on a separate assignment line.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "SPARK001", "Add .limit(N) before .collect(), or replace with .take(N) / .first()."
    )
    violations: List[SEMSViolation] = []
    seen_lines: set = set()

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "collect"
            and not node.args
            and not node.keywords
        ):
            continue
        line = getattr(node, "lineno", None)
        if line in seen_lines:
            continue
        if not _has_limit_in_chain(node):
            seen_lines.add(line)
            violations.append(
                SEMSViolation(
                    rule_id="SPARK001",
                    severity="soft",
                    category="spark_best_practices",
                    message=(
                        f"[SPARK001] Unbounded .collect() at line {line} — "
                        "no .limit(N>0) or .take(N) guard detected. "
                        "This can OOM the Spark driver on large DataFrames."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    return violations


def check_pyspark_api_typos(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SPARK003: Detect likely pyspark.sql.functions typos.

    Checks two import forms:
      1. Module alias (e.g. F.colz) — checks every F.<name> call.
      2. Direct-name imports (e.g. from pyspark.sql.functions import colz) —
         flags the import line itself when the imported name is not in the allowlist.

    Confidence label is based on true Levenshtein edit distance (not length proxy).
    The check is skipped entirely when the alias has been reassigned.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    alias, directly_imported, alias_reassigned = _get_functions_alias_info(tree)
    violations: List[SEMSViolation] = []

    def _flag(func_name: str, line: Optional[int], display_prefix: str) -> None:
        if func_name in PYSPARK_FUNCTIONS_ALLOWLIST:
            return
        import difflib as _dl

        candidates = list(PYSPARK_FUNCTIONS_ALLOWLIST)
        close = _dl.get_close_matches(func_name, candidates, n=5, cutoff=0.6)
        ranked = sorted(close, key=lambda m: _levenshtein(func_name, m))
        suggestion = ranked[0] if ranked else None
        if suggestion:
            dist = _levenshtein(func_name, suggestion)
            confidence = "95%+" if dist == 1 else ("80%" if dist <= 2 else "low")
            hint = (
                f"Did you mean '{display_prefix}{suggestion}()'? "
                f"(confidence {confidence}, edit distance {dist})"
            )
        else:
            hint = "No close match found — check pyspark.sql.functions documentation."

        rem = _remediation(
            "SPARK003",
            f"Correct '{display_prefix}{func_name}' to a valid pyspark.sql.functions name.",
        )
        violations.append(
            SEMSViolation(
                rule_id="SPARK003",
                severity="hard",
                category="syntax",
                message=(
                    f"[SPARK003] Unknown PySpark function '{display_prefix}{func_name}'"
                    + (f" at line {line}" if line else "")
                    + f". {hint}"
                ),
                remediation=rem,
                line_number=line,
                suggestion=suggestion,
            )
        )

    # Check alias-prefixed calls (F.xxx) only when alias is not reassigned.
    if alias is not None and not alias_reassigned:
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == alias
            ):
                continue
            _flag(node.func.attr, getattr(node, "lineno", None), f"{alias}.")

    # Check directly imported names (flags at the import line).
    if directly_imported:
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "pyspark.sql.functions"
            ):
                continue
            for a in node.names:
                imported_name = a.asname or a.name
                if a.name not in PYSPARK_FUNCTIONS_ALLOWLIST:
                    _flag(a.name, getattr(node, "lineno", None), "")

    return violations


def check_spark_actions_try_except(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SEMS_ERR001: Detect Spark action calls that are NOT inside a try *body*.

    Checked actions: .count(), .collect(), .show(), .take(), .first(),
    .toPandas(), .saveAsTable().

    A call inside an except/else/finally handler is intentionally NOT treated
    as protected — those contexts are not the normal execution path.
    """
    _SPARK_ACTIONS = frozenset(
        {"count", "collect", "show", "take", "first", "toPandas",
         "saveAsTable", "insertInto", "save", "parquet", "json", "csv", "orc"}
    )

    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    parent_map = _build_parent_map(tree)
    violations: List[SEMSViolation] = []
    seen_lines: set = set()
    remediation = _remediation(
        "SEMS_ERR001", "Wrap Spark actions in try/except to handle AnalysisException."
    )

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SPARK_ACTIONS
        ):
            continue
        # Skip aggregation-function forms with arguments: F.count("col"),
        # F.first("col"), F.last("col") are not DataFrame actions.
        if node.func.attr in ("count", "first", "last") and (node.args or node.keywords):
            continue

        line = getattr(node, "lineno", None)
        if line in seen_lines:
            continue

        if not _is_in_try_body(node, parent_map):
            seen_lines.add(line)
            violations.append(
                SEMSViolation(
                    rule_id="SEMS_ERR001",
                    severity="soft",
                    category="error_handling",
                    message=(
                        f"[SEMS_ERR001] .{node.func.attr}() at line {line} is not "
                        "inside a try body — Spark actions can raise AnalysisException."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    return violations


def check_module_io_docstring(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    DOC001: Verify the module-level docstring contains Input: and Output: sections
    so the STAM lineage auditor can parse data provenance automatically.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    docstring = ast.get_docstring(tree)
    remediation = _remediation(
        "DOC001", "Add Input: and Output: sections to the module docstring."
    )

    if not docstring:
        return [
            SEMSViolation(
                rule_id="DOC001",
                severity="soft",
                category="documentation",
                message=(
                    "[DOC001] Module has no docstring — Input:/Output: sections "
                    "are required for STAM lineage auditing."
                ),
                remediation=remediation,
            )
        ]

    has_input = bool(re.search(r"(?i)\bInputs?\s*:", docstring))
    has_output = bool(re.search(r"(?i)\bOutputs?\s*:", docstring))

    if has_input and has_output:
        return []

    missing = []
    if not has_input:
        missing.append("Input:")
    if not has_output:
        missing.append("Output:")
    return [
        SEMSViolation(
            rule_id="DOC001",
            severity="soft",
            category="documentation",
            message=(
                f"[DOC001] Module docstring is missing section(s): {', '.join(missing)}. "
                "STAM lineage auditing requires explicit Input/Output declarations."
            ),
            remediation=remediation,
        )
    ]


_DOC_ARGS_HEADER = re.compile(r"(?im)^[ \t]*(?:args|arguments|parameters)\s*:\s*$")
_DOC_RETURNS_HEADER = re.compile(r"(?im)^[ \t]*(?:returns?|yields?)\s*:\s*$")
_DOC_PARAM_LINE = re.compile(r"^[ \t]+(\*{0,2}[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:")


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _documented_args(docstring: str) -> Set[str]:
    """Return parameter names documented under an Args:/Arguments:/Parameters: section.

    Google-style docstrings are dedented by ast.get_docstring(clean=True) before
    reaching this function, so the section header sits at whatever indentation
    is common to the whole docstring, and its param lines are indented deeper
    than that. The section ends at the first line dedented back to (or past)
    the header's own indentation — which also naturally closes it on the next
    section header (Returns:, Raises:, ...) without needing to name every one.
    """
    documented: Set[str] = set()
    in_section = False
    header_indent = 0
    for line in docstring.splitlines():
        if _DOC_ARGS_HEADER.match(line):
            in_section = True
            header_indent = _line_indent(line)
            continue
        if not in_section:
            continue
        if not line.strip():
            continue  # blank separator line inside the section
        if _line_indent(line) <= header_indent:
            in_section = False
            continue
        match = _DOC_PARAM_LINE.match(line)
        if match:
            documented.add(match.group(1).lstrip("*"))
    return documented


def _has_returns_or_yields_section(docstring: str) -> bool:
    return any(_DOC_RETURNS_HEADER.match(line) for line in docstring.splitlines())


def _signature_param_names(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> List[str]:
    """Parameter names a docstring is expected to document, excluding self/cls."""
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def _is_property_getter(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> bool:
    """True for @property/@cached_property — idiomatic to skip a Returns: section
    since the docstring summary itself typically describes the returned value."""
    for dec in node.decorator_list:
        name = dec.id if isinstance(dec, ast.Name) else dec.attr if isinstance(dec, ast.Attribute) else None
        if name in ("property", "cached_property"):
            return True
    return False


def _scan_return_shape(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> Tuple[bool, bool]:
    """Return (has_return_value, is_generator) from the function's OWN body only —
    does not descend into nested function/class/lambda scopes."""
    has_return_value = False
    is_generator = False

    def _walk(n: ast.AST) -> None:
        nonlocal has_return_value, is_generator
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                if not (isinstance(child.value, ast.Constant) and child.value.value is None):
                    has_return_value = True
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                is_generator = True
            _walk(child)

    _walk(node)
    return has_return_value, is_generator


def check_function_docstring_content(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    DOC002 / DOC003: For a function/method that already HAS a docstring, verify
    it actually documents the interface — every parameter under an Args: section
    (DOC002), and a Returns:/Yields: section when the function returns or yields
    a value (DOC003).

    A docstring's mere PRESENCE on public names is pydocstyle's job (D101/D102/
    D103); this rule only evaluates docstrings that already exist, so it never
    double-reports a missing one — a placeholder like '''stuff.''' is exactly
    the case this rule exists to catch, and it also runs on private functions,
    which pydocstyle's presence check ignores entirely.

    Limitation: checks are structural (is the name/section present), not
    semantic — a docstring can still describe a parameter incorrectly or
    write a vacuous Returns: line and pass.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    args_remediation = _remediation(
        "DOC002", "Document every parameter under an Args: section in the docstring."
    )
    returns_remediation = _remediation(
        "DOC003", "Add a Returns:/Yields: section documenting the returned value."
    )
    violations: List[SEMSViolation] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            continue
        line = getattr(node, "lineno", None)

        required_params = _signature_param_names(node)
        if required_params:
            documented = _documented_args(docstring)
            missing_params = [p for p in required_params if p not in documented]
            if missing_params:
                violations.append(
                    SEMSViolation(
                        rule_id="DOC002",
                        severity="soft",
                        category="documentation",
                        message=(
                            f"[DOC002] '{node.name}' at line {line} has a docstring but "
                            f"does not document parameter(s): {', '.join(missing_params)}."
                        ),
                        remediation=args_remediation,
                        line_number=line,
                    )
                )

        if not _is_property_getter(node):
            has_return_value, is_generator = _scan_return_shape(node)
            if (has_return_value or is_generator) and not _has_returns_or_yields_section(docstring):
                kind = "yields" if is_generator and not has_return_value else "returns"
                violations.append(
                    SEMSViolation(
                        rule_id="DOC003",
                        severity="soft",
                        category="documentation",
                        message=(
                            f"[DOC003] '{node.name}' at line {line} {kind} a value but its "
                            "docstring has no Returns:/Yields: section."
                        ),
                        remediation=returns_remediation,
                        line_number=line,
                    )
                )

    return violations


# GDPR-relevant PII field names → hard violation (SHELL_PII002).
# Generic credential keywords → soft violation (LOG003).
_HARD_PII_FIELDS: frozenset = frozenset(
    {"email", "ssn", "credit_card", "creditcard", "phone", "phone_number",
     "national_id", "date_of_birth", "full_name"}
)


def check_pii_in_logs(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SHELL_PII002 (hard) for GDPR-relevant PII field names in log/print calls.
    LOG003 (soft) for other sensitive keywords (password, token, secret, etc.).

    AST-based: handles multiline calls and f-strings correctly; does not
    false-positive on comments or docstrings.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    _LOG_ATTRS = frozenset(
        {"info", "warning", "error", "debug", "critical", "warn", "exception"}
    )
    _PII_KEYWORDS_SET = frozenset(k.lower() for k in PII_KEYWORDS)

    def _keywords_in_arg(arg: ast.expr) -> Tuple[Set[str], Set[str], bool]:
        """Extract PII-keyword matches from an argument expression subtree.

        Returns (matches from variable/attribute names, matches from string
        literals, whether the subtree contains dynamic content — a variable,
        attribute, call, subscript, or f-string placeholder).
        """
        name_hits: Set[str] = set()
        const_hits: Set[str] = set()
        has_dynamic = False
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name):
                has_dynamic = True
                name_lower = sub.id.lower()
                if name_lower in _PII_KEYWORDS_SET:
                    name_hits.add(name_lower)
            elif isinstance(sub, ast.Attribute):
                has_dynamic = True
                attr_lower = sub.attr.lower()
                if attr_lower in _PII_KEYWORDS_SET:
                    name_hits.add(attr_lower)
            elif isinstance(sub, (ast.Call, ast.Subscript, ast.FormattedValue)):
                has_dynamic = True
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                text_lower = sub.value.lower()
                for kw in _PII_KEYWORDS_SET:
                    # Use word-boundary match to avoid "email_service" → "email" false pos.
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                        const_hits.add(kw)
        return name_hits, const_hits, has_dynamic

    violations: List[SEMSViolation] = []
    seen: Set[Tuple[Optional[int], str]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_log_call = (
            (isinstance(func, ast.Attribute) and func.attr in _LOG_ATTRS)
            or (isinstance(func, ast.Name) and func.id == "print")
        )
        if not is_log_call:
            continue

        line = getattr(node, "lineno", None)
        all_args = list(node.args) + [kw.value for kw in node.keywords]
        name_hits: Set[str] = set()
        const_hits: Set[str] = set()
        call_has_dynamic = False
        for arg in all_args:
            arg_names, arg_consts, arg_dynamic = _keywords_in_arg(arg)
            name_hits.update(arg_names)
            const_hits.update(arg_consts)
            call_has_dynamic = call_has_dynamic or arg_dynamic

        # Keywords inside plain string literals only count when the call also
        # logs dynamic data: print("Email sent successfully") exposes nothing,
        # while logger.info("email: %s", addr) does.
        keywords_found = name_hits | (const_hits if call_has_dynamic else set())

        for keyword in sorted(keywords_found):
            key = (line, keyword)
            if key in seen:
                continue
            seen.add(key)
            if keyword in _HARD_PII_FIELDS:
                violations.append(
                    SEMSViolation(
                        rule_id="SHELL_PII002",
                        severity="hard",
                        category="security",
                        message=(
                            f"[SHELL_PII002] Log/print call at line {line} exposes "
                            f"PII field '{keyword}' — GDPR violation."
                        ),
                        remediation=_remediation(
                            "SHELL_PII002",
                            "Remove PII from log/print calls or replace with a truncated hash.",
                        ),
                        line_number=line,
                    )
                )
            else:
                violations.append(
                    SEMSViolation(
                        rule_id="LOG003",
                        severity="soft",
                        category="error_handling",
                        message=(
                            f"[LOG003] Log/print call at line {line} references "
                            f"sensitive keyword '{keyword}'. Redact before logging."
                        ),
                        remediation=_remediation(
                            "LOG003",
                            "Redact sensitive data before logging or move to logger.debug().",
                        ),
                        line_number=line,
                    )
                )

    return violations


def check_bare_except(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    ERR002: Detect overly broad except handlers without re-raise.

    Broad handlers: bare 'except:', 'except Exception:', 'except BaseException:',
    and tuple handlers that include Exception or BaseException.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "ERR002", "Catch specific exception types and re-raise after logging."
    )
    violations: List[SEMSViolation] = []

    def _is_broad(handler: ast.ExceptHandler) -> bool:
        if handler.type is None:
            return True
        if isinstance(handler.type, ast.Name) and handler.type.id in (
            "Exception",
            "BaseException",
        ):
            return True
        if isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name) and elt.id in (
                    "Exception",
                    "BaseException",
                ):
                    return True
        return False

    def _body_has_reraise(handler: ast.ExceptHandler) -> bool:
        # Walk only the handler body, excluding nested functions/classes so that a
        # raise inside a nested def does not count as re-raising the outer exception.
        def _walk_no_scope(node: ast.AST):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return  # stop descending at nested scope boundaries
            yield node
            for child in ast.iter_child_nodes(node):
                yield from _walk_no_scope(child)

        for stmt in handler.body:
            for child in _walk_no_scope(stmt):
                if isinstance(child, ast.Raise):
                    return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_broad(node):
            continue
        if _body_has_reraise(node):
            continue
        line = getattr(node, "lineno", None)
        kind = _format_except_type(node.type)
        violations.append(
            SEMSViolation(
                rule_id="ERR002",
                severity="soft",
                category="error_handling",
                message=(
                    f"[ERR002] {kind} at line {line} does not re-raise — "
                    "this swallows Spark failures silently."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


# ── Company rule checks ───────────────────────────────────────────────────────

_VALID_TABLE_NAME = re.compile(
    r"^[a-z][a-z0-9_]*_[a-z_]+_(raw|staging|curated|published)$"
)

def check_table_naming_convention(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SHELL_NAME001: Table names must follow <domain>_<entity>_<layer> convention."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    violations: List[SEMSViolation] = []
    remediation = _remediation(
        "SHELL_NAME001",
        "Rename to <domain>_<entity>_(raw|staging|curated|published).",
    )
    _SPARK_TABLE_METHODS = frozenset(
        {"table", "forName", "writeTo", "saveAsTable", "insertInto"}
    )
    _SPARK_TABLE_RECEIVERS = frozenset(
        {"spark", "read", "DeltaTable", "delta", "catalog"}
    )

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SPARK_TABLE_METHODS
        ):
            continue
        # writeTo()/saveAsTable()/insertInto() are Spark write APIs — no further
        # receiver check needed since no standard Python library uses them with
        # a table-name string. For table()/forName(), require a known Spark
        # identifier in the receiver chain to avoid flagging unrelated
        # user-defined methods.
        if node.func.attr in ("table", "forName") and not _chain_contains(
            node.func, _SPARK_TABLE_RECEIVERS
        ):
            continue
        for arg in node.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            raw_name = arg.value
            table_segment = raw_name.split(".")[-1]
            if not _VALID_TABLE_NAME.match(table_segment):
                line = getattr(node, "lineno", None)
                violations.append(
                    SEMSViolation(
                        rule_id="SHELL_NAME001",
                        severity="soft",
                        category="databricks_compatibility",
                        message=(
                            f"[SHELL_NAME001] Table name '{raw_name}' at line {line} "
                            "does not follow <domain>_<entity>_<layer> convention."
                        ),
                        remediation=remediation,
                        line_number=line,
                    )
                )
    return violations


def check_hard_coded_repartition(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SHELL_SPARK001: .repartition(N) with a literal integer is a bottleneck."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    violations: List[SEMSViolation] = []
    remediation = _remediation(
        "SHELL_SPARK001",
        "Remove .repartition() and let AQE manage partitioning, or derive N from config.",
    )
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "repartition"
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                line = getattr(node, "lineno", None)
                violations.append(
                    SEMSViolation(
                        rule_id="SHELL_SPARK001",
                        severity="soft",
                        category="spark_best_practices",
                        message=(
                            f"[SHELL_SPARK001] Hard-coded .repartition({arg.value}) "
                            f"at line {line} — use AQE or derive from config."
                        ),
                        remediation=remediation,
                        line_number=line,
                    )
                )
    return violations


def check_widget_without_default(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SHELL_STAM002: dbutils.widgets.get() without a prior default registration.

    Only flags genuine dbutils.widgets.get() calls — not arbitrary .get() calls
    on other objects.  Enforces source ordering: the registration must appear
    before the get() call in the file.
    """
    _REGISTER_METHODS = frozenset({"text", "dropdown", "combobox", "multiselect"})
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    def _is_dbutils_widgets_call(node: ast.Call, method: str) -> bool:
        """Return True iff node is exactly dbutils.widgets.<method>(...)."""
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == method):
            return False
        val = node.func.value
        return (
            isinstance(val, ast.Attribute)
            and val.attr == "widgets"
            and isinstance(val.value, ast.Name)
            and val.value.id == "dbutils"
        )

    # Collect (lineno, kind, widget_name) for all dbutils.widgets calls.
    events: List[Tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name_val = node.args[0].value
        if not isinstance(name_val, str):
            continue
        lineno = getattr(node, "lineno", 0) or 0
        attr = node.func.attr
        if attr in _REGISTER_METHODS and _is_dbutils_widgets_call(node, attr):
            events.append((lineno, "register", name_val))
        elif attr == "get" and _is_dbutils_widgets_call(node, "get"):
            events.append((lineno, "get", name_val))

    # get() calls inside function/lambda bodies execute only when called, so a
    # module-level registration textually below the def still runs first at
    # runtime. Source order is therefore only enforced for module-level get()
    # calls; deferred get() calls accept a registration anywhere in the file.
    deferred_spans = [
        (n.lineno, getattr(n, "end_lineno", None) or n.lineno)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]

    def _is_deferred(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in deferred_spans)

    all_registered = {name for _, kind, name in events if kind == "register"}

    # Process in source order: registration only counts when it precedes the get().
    events.sort(key=lambda e: e[0])
    registered: Set[str] = set()
    violations: List[SEMSViolation] = []
    for lineno, kind, name in events:
        if kind == "register":
            registered.add(name)
        elif kind == "get" and name not in registered and not (
            _is_deferred(lineno) and name in all_registered
        ):
            violations.append(
                SEMSViolation(
                    rule_id="SHELL_STAM002",
                    severity="soft",
                    category="databricks_compatibility",
                    message=(
                        f"[SHELL_STAM002] dbutils.widgets.get('{name}') at line {lineno} "
                        "has no prior .text()/.dropdown() default — fails on non-interactive runs."
                    ),
                    remediation=_remediation(
                        "SHELL_STAM002",
                        f"Add dbutils.widgets.text('{name}', '<default>') before the get() call.",
                    ),
                    line_number=lineno,
                )
            )
    return violations


def check_local_spark_master(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SHELL_STAM001: Detect SparkSession.builder.master('local...') calls."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    # "builder" is intentionally excluded: it matches too broadly (any
    # unrelated builder.master("local") call). SparkSession and spark cover
    # all realistic Spark chains (SparkSession.builder.master / spark.master).
    _SPARK_BUILDER_NAMES = frozenset({"SparkSession", "spark"})

    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "master"
        ):
            continue
        # Only flag .master() calls on a SparkSession/builder chain, not unrelated objects.
        if not _chain_contains(node.func, _SPARK_BUILDER_NAMES):
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("local")
            ):
                line = getattr(node, "lineno", None)
                violations.append(
                    SEMSViolation(
                        rule_id="SHELL_STAM001",
                        severity="hard",
                        category="databricks_compatibility",
                        message=(
                            f"[SHELL_STAM001] .master({arg.value!r}) at line {line} — "
                            "local mode is not permitted in STAM workloads."
                        ),
                        remediation=_remediation(
                            "SHELL_STAM001",
                            "Remove .master(...). The Databricks cluster provides the SparkSession.",
                        ),
                        line_number=line,
                    )
                )
    return violations


def _find_stmt_root(
    node: ast.AST, parent_map: Dict[int, ast.AST]
) -> Optional[ast.AST]:
    """Walk up to the enclosing statement node (direct child of a block)."""
    current = node
    while id(current) in parent_map:
        parent = parent_map[id(current)]
        if isinstance(
            parent,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
             ast.For, ast.While, ast.With, ast.If, ast.Try, ast.ExceptHandler),
        ):
            return current
        current = parent
    return None


def check_jdbc_no_partition(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SHELL_SPARK002: Detect JDBC writes that omit the partitionColumn option."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    parent_map = _build_parent_map(tree)
    violations: List[SEMSViolation] = []
    seen_lines: set = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        attr = node.func.attr
        is_format_jdbc = attr == "format" and bool(
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "jdbc"
        )
        is_direct_jdbc = attr == "jdbc"

        if not (is_format_jdbc or is_direct_jdbc):
            continue

        # Only flag write paths — skip spark.read.format("jdbc").load() patterns.
        if not _chain_contains(node.func, frozenset({"write"})):
            continue

        line = getattr(node, "lineno", None)
        if line in seen_lines:
            continue

        # All four options are required together for parallel JDBC writes.
        _REQUIRED_OPTS = frozenset(
            {"partitionColumn", "lowerBound", "upperBound", "numPartitions"}
        )
        root = _find_stmt_root(node, parent_map)
        subtree = root if root is not None else node
        # Only count option keys that appear as the first argument of an .option()
        # call — prevents a loose string constant "partitionColumn" elsewhere in
        # the statement from incorrectly satisfying the check.
        found_opts: Set[str] = {
            n.args[0].value
            for n in ast.walk(subtree)
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "option"
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
                and n.args[0].value in _REQUIRED_OPTS
            )
        }
        # For direct .jdbc() calls, partition options may appear as keyword args.
        if is_direct_jdbc:
            found_opts.update(
                kw.arg for kw in node.keywords if kw.arg in _REQUIRED_OPTS
            )
        missing_opts = _REQUIRED_OPTS - found_opts

        if missing_opts:
            seen_lines.add(line)
            violations.append(
                SEMSViolation(
                    rule_id="SHELL_SPARK002",
                    severity="hard",
                    category="spark_best_practices",
                    message=(
                        f"[SHELL_SPARK002] JDBC write at line {line} is missing "
                        f"required option(s): {sorted(missing_opts)}. "
                        "All of partitionColumn, lowerBound, upperBound, numPartitions are required."
                    ),
                    remediation=_remediation(
                        "SHELL_SPARK002",
                        "Add .option('partitionColumn', ...), .option('lowerBound', ...), "
                        ".option('upperBound', ...), .option('numPartitions', ...) to the JDBC write.",
                    ),
                    line_number=line,
                )
            )
    return violations


def _is_delta_write_node(
    write_attr: ast.Attribute, parent_map: Dict[int, ast.AST]
) -> bool:
    """Return True if write_attr's enclosing statement targets a Delta/table sink.

    writeTo() is the V2 table API — always Delta-like.
    For .write, require .format("delta"), .saveAsTable(), or .insertInto() in the
    same statement to avoid flagging JSON/CSV/Parquet writes as SHELL_PII001.
    """
    if write_attr.attr == "writeTo":
        return True
    stmt = _find_stmt_root(write_attr, parent_map)
    if stmt is None:
        return False
    for sub in ast.walk(stmt):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
            continue
        if sub.func.attr in ("saveAsTable", "insertInto"):
            return True
        if (
            sub.func.attr == "format"
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and isinstance(sub.args[0].value, str)
            and sub.args[0].value.lower() == "delta"
        ):
            return True
    return False


def check_pii_column_write(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SHELL_PII001: Best-effort detection of PII columns written to Delta without masking.

    - Uses AST string constants *and* attribute names to detect column references,
      catching both df.select("email") and df.select(df.email) patterns.
    - Only fires on Delta/table sinks (.format("delta"), .saveAsTable(), .writeTo()).
    - Position-aware: masking must appear strictly before the write in source order.
      For same-line cases, column offset is used so that
      df.sha2(...).write... is OK but df.write...; sha2(...) is still flagged.
    - Reports the line of the write statement.

    Limitation: cross-assignment or cross-file dataflow is not tracked.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    # Collect write-site (lineno, prefix_col) pairs, restricted to Delta/table sinks.
    # lineno is 1-based.  prefix_col is the end column of the receiver expression
    # (i.e. the position just before the dot in ".write"), so that
    # source_lines[lineno-1][:prefix_col] covers everything on that line that
    # appears *before* the write call — without including post-write same-line code.
    parent_map = _build_parent_map(tree)
    write_sites: List[tuple] = []  # List[Tuple[int, int]]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("write", "writeTo"):
            if _is_delta_write_node(node, parent_map):
                # Use the receiver's END position (not node.lineno, which is the
                # first line of the whole chain): in a multi-line chained call,
                # masking applied earlier in the same chain sits between the
                # chain's first line and the .write token, and must be included
                # in the "before the write" window.
                ln = getattr(node.value, "end_lineno", None)
                # end_col_offset of .value is the column where the receiver ends;
                # .write/.writeTo begins right after (the dot is at that position).
                prefix_col = getattr(node.value, "end_col_offset", None)
                if ln is not None and prefix_col is not None:
                    write_sites.append((ln, prefix_col))

    if not write_sites:
        return []

    # Collect PII column names from string constants ("email") and attribute
    # accesses (df.email) — both represent column references in Spark code.
    referenced_pii: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            segment = node.value.lower().split(".")[-1]
            if segment in _PII_COLUMN_NAMES:
                referenced_pii.add(segment)
        elif isinstance(node, ast.Attribute):
            attr = node.attr.lower()
            if attr in _PII_COLUMN_NAMES:
                referenced_pii.add(attr)

    violations: List[SEMSViolation] = []
    source_lines = source_code.splitlines()

    for pii_col in sorted(referenced_pii):
        # Accept both "email" (quoted) and df.email (attribute) column references
        # inside the masking call — mirroring how references are collected.
        # The column must be preceded by a quote or dot so bare-identifier
        # mentions in comments/strings don't count as masking, and the leading
        # \b keeps verify_hash()/geohash() from matching as 'hash'.
        col_mask = re.compile(
            r"\b(?:sha2|sha256|hash|md5)\s*\([^)]*(?:['\"]|\.)"
            + re.escape(pii_col)
            + r"\b",
            re.IGNORECASE,
        )
        redacted_pat = re.compile(
            r"""withColumn\s*\(\s*['"]"""
            + re.escape(pii_col)
            + r"""['"]\s*,\s*(?:F\.)?lit\s*\(\s*['"]REDACTED['"]""",
            re.IGNORECASE,
        )
        for write_line, prefix_col in write_sites:
            # Build the source text strictly before the write site:
            # - All complete lines before write_line.
            # - The write line up to prefix_col (where the receiver expression ends
            #   and the dot in ".write" begins), so masking that precedes the write
            #   in the same chained call is included, while post-write same-line
            #   code (e.g. after a semicolon) is excluded.
            lines_before = source_lines[: write_line - 1]
            partial_write_line = source_lines[write_line - 1][:prefix_col]
            source_before = "\n".join(lines_before + [partial_write_line])
            if col_mask.search(source_before) or redacted_pat.search(source_before):
                continue
            violations.append(
                SEMSViolation(
                    rule_id="SHELL_PII001",
                    severity="hard",
                    category="security",
                    message=(
                        f"[SHELL_PII001] PII column '{pii_col}' may be written to Delta "
                        f"at line {write_line} without prior masking — "
                        f"no sha2/hash/lit('REDACTED') for '{pii_col}' found before the write."
                    ),
                    remediation=_remediation(
                        "SHELL_PII001",
                        f"Apply .withColumn('{pii_col}', F.sha2(F.col('{pii_col}'), 256)) before the write.",
                    ),
                    line_number=write_line,
                )
            )
    return violations


# ── Regex-based check functions (all now return List[SEMSViolation]) ──────────


_CRED_NAMES: frozenset = frozenset(
    {"password", "passwd", "pwd", "token", "secret", "api_key", "apikey"}
)
_MIN_CRED_LEN = 4  # ignore placeholder values shorter than this


def check_for_hardcoded_credentials(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SEC001: Detect hardcoded credentials.

    Uses AST for assignment, dict-literal, and subscript patterns — this avoids
    false positives from # comments and docstrings that merely mention credential
    keywords in examples.

    Regex is used only for Bearer tokens and Databricks PATs, which have a
    distinctive shape unlikely to appear in documentation strings.
    """
    remediation = _remediation(
        "SEC001",
        "Move secrets to a Databricks secret scope: dbutils.secrets.get(scope=..., key=...).",
    )
    violations: List[SEMSViolation] = []
    seen_lines: set = set()

    def _add(line_no: Optional[int], detail: str) -> None:
        if line_no not in seen_lines:
            seen_lines.add(line_no)
            violations.append(
                SEMSViolation(
                    rule_id="SEC001",
                    severity="hard",
                    category="security",
                    message=f"[SEC001] {detail}",
                    remediation=remediation,
                    line_number=line_no,
                )
            )

    # ── AST-based patterns (immune to comment/docstring false positives) ───────
    try:
        if tree is None:
            tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # var = "..."  where var is a credential name
                    if (
                        isinstance(target, ast.Name)
                        and target.id.lower() in _CRED_NAMES
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        and len(node.value.value) >= _MIN_CRED_LEN
                    ):
                        line = getattr(node, "lineno", None)
                        _add(line, f"Hardcoded credential '{target.id}' at line {line}.")

                    # config["token"] = "..."
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                        and target.slice.value.lower() in _CRED_NAMES
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        and len(node.value.value) >= _MIN_CRED_LEN
                    ):
                        line = getattr(node, "lineno", None)
                        _add(line, f"Hardcoded credential key '{target.slice.value}' (subscript) at line {line}.")

            # password: str = "..."  annotated assignment
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id.lower() in _CRED_NAMES
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and len(node.value.value) >= _MIN_CRED_LEN
            ):
                line = getattr(node, "lineno", None)
                _add(line, f"Hardcoded credential '{node.target.id}' at line {line}.")

            # connect(password="...")  keyword argument
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg
                        and kw.arg.lower() in _CRED_NAMES
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                        and len(kw.value.value) >= _MIN_CRED_LEN
                    ):
                        line = getattr(kw.value, "lineno", None)
                        _add(line, f"Hardcoded credential '{kw.arg}' (keyword argument) at line {line}.")

            # {"password": "..."} dict literal
            if isinstance(node, ast.Dict):
                for key, val in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value.lower() in _CRED_NAMES
                        and isinstance(val, ast.Constant)
                        and isinstance(val.value, str)
                        and len(val.value) >= _MIN_CRED_LEN
                    ):
                        line = getattr(node, "lineno", None)
                        _add(line, f"Hardcoded credential key '{key.value}' in dict literal at line {line}.")
    except SyntaxError:
        pass

    # ── AST-based: Bearer tokens and Databricks PATs (skips docstrings) ─────────
    # Scanning non-docstring string literals avoids false positives from examples
    # in module/function docstrings that mention token patterns.
    _BEARER_RE = re.compile(r"(?i)Bearer\s+[A-Za-z0-9\-_\.]{20,}")
    _DAPI_RE = re.compile(r"dapi[a-f0-9]{32}")
    try:
        if tree is None:
            tree = ast.parse(source_code)
        _tok_tree = tree
        _docstring_ids: set = set()
        for _dn in ast.walk(_tok_tree):
            if isinstance(_dn, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (
                    _dn.body
                    and isinstance(_dn.body[0], ast.Expr)
                    and isinstance(_dn.body[0].value, ast.Constant)
                ):
                    _docstring_ids.add(id(_dn.body[0].value))
        for _sn in ast.walk(_tok_tree):
            if not (isinstance(_sn, ast.Constant) and isinstance(_sn.value, str)):
                continue
            if id(_sn) in _docstring_ids:
                continue
            _sval = _sn.value
            for _re in (_BEARER_RE, _DAPI_RE):
                _sm = _re.search(_sval)
                if _sm:
                    _sl = getattr(_sn, "lineno", None)
                    _add(_sl, f"Possible hardcoded credential at line {_sl}: {_sm.group()[:60]!r}")
    except SyntaxError:
        pass

    return violations


def check_for_logging_usage(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """LOG001: Verify logging is imported and used (AST-based; immune to docstring suppression)."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "LOG001",
        "Add 'import logging' and replace print() calls with logger.info/error().",
    )

    has_logging_import = any(
        (isinstance(n, ast.Import) and any(a.name == "logging" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("logging"))
        for n in ast.walk(tree)
    )
    if not has_logging_import:
        return [
            SEMSViolation(
                rule_id="LOG001",
                severity="soft",
                category="error_handling",
                message=(
                    "[LOG001] 'import logging' not found — use the logging module instead of print()."
                ),
                remediation=remediation,
            )
        ]

    _LOG_METHOD_NAMES = frozenset(
        {"info", "warning", "error", "debug", "critical", "warn", "exception", "getLogger"}
    )
    has_logger_call = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in _LOG_METHOD_NAMES
        for n in ast.walk(tree)
    )
    if not has_logger_call:
        return [
            SEMSViolation(
                rule_id="LOG001",
                severity="soft",
                category="error_handling",
                message=(
                    "[LOG001] logging module imported but no logger calls found — "
                    "add logger.info/error() calls."
                ),
                remediation=remediation,
            )
        ]
    return []


def check_for_error_handling(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """ERR001 (structural): Warn when no try/except block is present (AST-based).

    Distinct from LOG001 (logging usage) and SEMS_ERR001 (a specific Spark
    action outside a try body) — this is the module-level "no error handling
    at all" signal, so it carries its own rule ID with its own remediation.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    if not any(isinstance(n, ast.Try) for n in ast.walk(tree)):
        return [
            SEMSViolation(
                rule_id="ERR001",
                severity="soft",
                category="error_handling",
                message="[ERR001] No try/except block found — add error handling for external calls.",
                remediation=_remediation(
                    "ERR001", "Wrap external/Spark calls in try/except."
                ),
            )
        ]
    return []


def check_for_bare_print_statements(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """LOG002: Flag bare print() calls (AST-based; one violation per call site with line number)."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "LOG002", "Replace print() with logger.info() or logger.debug()."
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            line = getattr(node, "lineno", None)
            violations.append(
                SEMSViolation(
                    rule_id="LOG002",
                    severity="soft",
                    category="error_handling",
                    message=(
                        f"[LOG002] Bare print() call at line {line} — "
                        "replace with logger.info/debug()."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    return violations


def check_for_spark_session_misuse(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK008: Warn about multiple SparkSession.builder calls (AST-based; reports line number)."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    builder_lines: List[int] = [
        getattr(node, "lineno", 0) or 0
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "builder"
            and isinstance(node.value, ast.Name)
            and node.value.id == "SparkSession"
        )
    ]
    if len(builder_lines) > 1:
        line = builder_lines[1]
        return [
            SEMSViolation(
                rule_id="SPARK008",
                severity="soft",
                category="spark_best_practices",
                message=(
                    f"[SPARK008] Multiple SparkSession.builder calls "
                    f"(second at line {line}) — use the shared 'spark' instance."
                ),
                remediation=_remediation(
                    "SPARK008",
                    "Use the shared 'spark' variable provided by the Databricks cluster.",
                ),
                line_number=line,
            )
        ]
    return []


def check_for_unsafe_operations(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SEC002/SEC003: Detect subprocess imports/usage and bare eval()/exec() calls.

    AST-based when the source parses: catches import/from-import/multi-import,
    direct subprocess.xxx() calls, __import__("subprocess"), and
    importlib.import_module("subprocess") — with no false positives from
    docstrings or string literals that merely mention subprocess or eval.
    Falls back to regex on comment-stripped text only when the file does not
    parse (strings cannot be stripped without a parse, so the fallback may
    match inside literals; it only runs for broken files).
    """
    violations: List[SEMSViolation] = []
    _sec002_seen_lines: set = set()

    def _add_sec002(line_no: Optional[int]) -> None:
        if line_no in _sec002_seen_lines:
            return
        _sec002_seen_lines.add(line_no)
        violations.append(
            SEMSViolation(
                rule_id="SEC002",
                severity="hard",
                category="security",
                message=(
                    f"[SEC002] subprocess usage at line {line_no} — "
                    "shell execution is not permitted in Databricks STAM workloads."
                ),
                remediation=_remediation(
                    "SEC002",
                    "Remove subprocess usage. Use dbutils for file ops or Spark for data.",
                ),
                line_number=line_no,
            )
        )

    def _add_sec003(line_no: Optional[int], func_desc: str) -> None:
        violations.append(
            SEMSViolation(
                rule_id="SEC003",
                severity="hard",
                category="security",
                message=(
                    f"[SEC003] {func_desc} at line {line_no} — "
                    "dynamic execution is a security risk; refactor to explicit logic."
                ),
                remediation=_remediation(
                    "SEC003",
                    "Replace eval()/exec() with explicit, static logic.",
                ),
                line_number=line_no,
            )
        )

    if tree is not None:
        _tree = tree
    else:
        try:
            _tree = ast.parse(source_code)
        except SyntaxError:
            _tree = None

    if _tree is not None:
        for _node in ast.walk(_tree):
            # import X, subprocess  or  import subprocess
            if isinstance(_node, ast.Import):
                for alias in _node.names:
                    if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                        _add_sec002(_node.lineno)
            # from subprocess import run  /  from subprocess.xyz import ...
            elif isinstance(_node, ast.ImportFrom):
                module = _node.module or ""
                if module == "subprocess" or module.startswith("subprocess."):
                    _add_sec002(_node.lineno)
            elif isinstance(_node, ast.Call):
                func = _node.func
                line_no = getattr(_node, "lineno", None)
                # subprocess.xxx(...) — direct attribute call on the subprocess Name
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"
                ):
                    _add_sec002(line_no)
                # __import__("subprocess")  /  importlib.import_module("subprocess")
                elif (
                    (isinstance(func, ast.Name) and func.id == "__import__")
                    or (isinstance(func, ast.Attribute) and func.attr == "import_module")
                ) and (
                    _node.args
                    and isinstance(_node.args[0], ast.Constant)
                    and _node.args[0].value == "subprocess"
                ):
                    _add_sec002(line_no)
                # bare eval()/exec() Name calls only — df.eval(...) attribute
                # calls and "eval" inside string literals are not flagged.
                elif isinstance(func, ast.Name) and func.id in ("eval", "exec"):
                    _add_sec003(line_no, f"{func.id}()")
    else:
        # Source is invalid Python — regex fallback on comment-stripped text.
        clean = _code_only(source_code)
        for match in SUBPROCESS_IMPORT_PATTERN.finditer(clean):
            _add_sec002(clean[: match.start()].count("\n") + 1)
        for match in EVAL_EXEC_PATTERN.finditer(clean):
            _add_sec003(clean[: match.start()].count("\n") + 1, "eval()/exec()")

    return violations


def check_for_pandas_udf(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK007: Warn about pandas_udf usage (AST-based; reports line number)."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == "pandas_udf") or (
            isinstance(node, ast.Attribute) and node.attr == "pandas_udf"
        ):
            line = getattr(node, "lineno", None)
            return [
                SEMSViolation(
                    rule_id="SPARK007",
                    severity="soft",
                    category="spark_best_practices",
                    message=(
                        f"[SPARK007] pandas_udf detected at line {line} — "
                        "verify this is intentional; prefer native PySpark functions."
                    ),
                    remediation=_remediation(
                        "SPARK007",
                        "Replace pandas_udf with native pyspark.sql.functions equivalents.",
                    ),
                    line_number=line,
                )
            ]
    return []


def check_for_dbutils_fs_in_loop(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK010: Detect dbutils.fs calls inside loops (reports the offending line number).

    AST-based: walks every for/while loop body, so nesting, comments,
    multi-line strings, and continuation lines cannot break detection.
    A loop's own iterable (``for f in dbutils.fs.ls(...)``) runs once and
    is not flagged unless the loop is itself nested inside another loop.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr == "fs"
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "dbutils"
                ):
                    line_no = getattr(sub, "lineno", None)
                    return [
                        SEMSViolation(
                            rule_id="SPARK010",
                            severity="soft",
                            category="databricks_compatibility",
                            message=(
                                f"[SPARK010] dbutils.fs call inside a loop at line {line_no} — "
                                "use Spark glob patterns or spark.read for directory-level operations."
                            ),
                            remediation=_remediation(
                                "SPARK010",
                                "Replace dbutils.fs inside loops with spark.read.",
                            ),
                            line_number=line_no,
                        )
                    ]
    return []


def check_driver_side_loops(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SPARK006: Detect loops and comprehensions that iterate over driver-collected Spark data.

    Handles:
      - Direct:    for row in df.collect(): ...
      - Indirect:  rows = df.collect(); for row in rows: ...
      - List comp: [x for x in df.collect()]
      - toPandas iteration patterns
      - Tuple targets: for (a, b) in df.collect(): ...
    """
    _SPARK_COLLECTION_METHODS = frozenset(
        {"collect", "toPandas", "toLocalIterator"}
    )

    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "SPARK006",
        "Use Spark transformations (map, filter, groupBy) instead of driver-side loops.",
    )
    violations: List[SEMSViolation] = []

    # Pass 1: record variable names assigned from Spark collection methods.
    spark_vars: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
            if val.func.attr in _SPARK_COLLECTION_METHODS:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        spark_vars.add(target.id)

    def _check_iter(iter_node: ast.expr, line: Optional[int]) -> None:
        # Direct call: df.collect() / df.toPandas() / df.toLocalIterator()
        if (
            isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Attribute)
            and iter_node.func.attr in _SPARK_COLLECTION_METHODS
        ):
            violations.append(
                SEMSViolation(
                    rule_id="SPARK006",
                    severity="soft",
                    category="spark_best_practices",
                    message=(
                        f"[SPARK006] Driver-side iteration over "
                        f".{iter_node.func.attr}() at line {line}. "
                        "Use Spark transformations instead."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
        # Indirect: iterating over a variable known to hold collected data.
        elif isinstance(iter_node, ast.Name) and iter_node.id in spark_vars:
            violations.append(
                SEMSViolation(
                    rule_id="SPARK006",
                    severity="soft",
                    category="spark_best_practices",
                    message=(
                        f"[SPARK006] Driver-side loop over '{iter_node.id}' at line {line} "
                        "(variable assigned from a Spark collection call). "
                        "Use Spark transformations instead."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            _check_iter(node.iter, getattr(node, "lineno", None))
        elif isinstance(node, ast.comprehension):
            # ast.comprehension carries no lineno of its own — use the iterable's.
            _check_iter(node.iter, getattr(node.iter, "lineno", None))

    return violations


def check_local_file_paths(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    SPARK009: Detect local filesystem path strings in the code.

    Operates on AST string literals only, avoiding false positives from comments
    and docstrings that merely *mention* local paths.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "SPARK009",
        "Replace local paths with DBFS (/dbfs/mnt/...) or cloud URIs (s3://, abfss://).",
    )
    violations: List[SEMSViolation] = []
    seen_lines: set = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        val = node.value
        if len(val) <= 3:
            continue
        if not any(val.startswith(prefix) for prefix in _LOCAL_PATH_PREFIXES):
            continue
        line = getattr(node, "lineno", None)
        if line in seen_lines:
            continue
        seen_lines.add(line)
        violations.append(
            SEMSViolation(
                rule_id="SPARK009",
                severity="soft",
                category="databricks_compatibility",
                message=(
                    f"[SPARK009] Local path {val[:60]!r} at line {line} — "
                    "use DBFS or cloud storage instead."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


# ── Construction Policy alignment (literals, comments, quality metrics) ──────


def check_repeated_magic_literals(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """LIT001: A numeric literal reused across conditions/expressions should be a named constant."""
    _EXCLUDED_VALUES = frozenset({-1, 0, 1, 2})
    _MIN_OCCURRENCES = 3

    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    # Literals already assigned to an UPPER_CASE name are treated as declared
    # constants and are not flagged.
    declared_constant_values: Set[object] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and not isinstance(node.value.value, bool)
        ):
            declared_constant_values.add(node.value.value)

    occurrences: Dict[object, List[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
        elif isinstance(node, ast.BinOp):
            operands = [node.left, node.right]
        else:
            continue
        line = getattr(node, "lineno", None)
        for operand in operands:
            if not isinstance(operand, ast.Constant):
                continue
            value = operand.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value in _EXCLUDED_VALUES or value in declared_constant_values:
                continue
            occurrences.setdefault(value, []).append(line)

    remediation = _remediation(
        "LIT001",
        "Extract the repeated literal into a module-level UPPER_CASE constant.",
    )
    violations: List[SEMSViolation] = []
    for value, lines in occurrences.items():
        if len(lines) < _MIN_OCCURRENCES:
            continue
        first_line = lines[0]
        violations.append(
            SEMSViolation(
                rule_id="LIT001",
                severity="soft",
                category="syntax",
                message=(
                    f"[LIT001] Magic literal {value!r} used {len(lines)} times "
                    f"in conditions/expressions (first at line {first_line}) — "
                    "extract it into a named constant."
                ),
                remediation=remediation,
                line_number=first_line,
            )
        )
    return violations


_TODO_PATTERN = re.compile(r"#.*\b(TODO|FIXME|XXX)\b", re.IGNORECASE)


def check_todo_comments(source_code: str, _tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """COM001: TODO/FIXME/XXX comments must be resolved before release."""
    remediation = _remediation(
        "COM001",
        "Resolve or remove the TODO/FIXME comment before release.",
    )
    violations: List[SEMSViolation] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source_code).readline):
            if tok.type != tokenize.COMMENT:
                continue
            match = _TODO_PATTERN.search(tok.string)
            if not match:
                continue
            line = tok.start[0]
            violations.append(
                SEMSViolation(
                    rule_id="COM001",
                    severity="soft",
                    category="documentation",
                    message=(
                        f"[COM001] {match.group(1).upper()} comment at line {line} "
                        "— resolve or remove before release."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return violations


_TRIVIAL_COMMENT_MARKERS = (
    "noqa", "type:", "coding:", "-*-", "!/usr/bin", "pragma", "pylint:",
)


_CODE_LIKE_STATEMENT_TYPES = (
    ast.Assign, ast.AugAssign, ast.Return, ast.Import, ast.ImportFrom,
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith,
    ast.Raise, ast.Global, ast.Nonlocal, ast.Delete,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)


def _looks_like_code(text: str) -> bool:
    """Return True when *text* parses as a substantive Python statement.

    Uses a strict allow-list rather than excluding a few trivial cases: ETL
    pipelines commonly leave short "field: transformation" (AnnAssign) or
    narrative comments (e.g. "union + dropDuplicates", a bare "dropna(...)"
    with no receiver) that happen to parse but describe the following line
    rather than being disabled code — those are deliberately excluded.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _TRIVIAL_COMMENT_MARKERS):
        return False
    try:
        parsed = ast.parse(stripped)
    except (SyntaxError, ValueError):
        return False
    for stmt in parsed.body:
        if isinstance(stmt, _CODE_LIKE_STATEMENT_TYPES):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
        ):
            return True
    return False


def check_commented_out_code(source_code: str, _tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """COM002: Flag comments whose content parses as commented-out code."""
    remediation = _remediation(
        "COM002",
        "Delete the commented-out code; rely on Git history to recover it if needed.",
    )
    violations: List[SEMSViolation] = []
    lines = source_code.splitlines()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source_code).readline):
            if tok.type != tokenize.COMMENT:
                continue
            line_no, col = tok.start
            line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
            if line_text[:col].strip():
                continue  # inline comment after real code, not a commented-out line
            candidate = tok.string.lstrip("#").strip()
            if not _looks_like_code(candidate):
                continue
            violations.append(
                SEMSViolation(
                    rule_id="COM002",
                    severity="soft",
                    category="documentation",
                    message=f"[COM002] Commented-out code at line {line_no}: {candidate[:60]!r}",
                    remediation=remediation,
                    line_number=line_no,
                )
            )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return violations


_BOOLEAN_NAME_PREFIXES = ("is_", "has_", "can_", "should_", "was_", "does_")


def check_boolean_function_naming(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """FUNC001: Functions with an explicit -> bool return type should read as predicates."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "FUNC001",
        "Rename to a predicate-style name (is_/has_/can_/should_).",
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (isinstance(node.returns, ast.Name) and node.returns.id == "bool"):
            continue
        name = node.name
        if name.startswith("__") and name.endswith("__"):
            continue
        if name.lower().startswith(_BOOLEAN_NAME_PREFIXES):
            continue
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="FUNC001",
                severity="soft",
                category="syntax",
                message=(
                    f"[FUNC001] Function '{name}' at line {line} returns bool but "
                    "isn't named like a predicate (is_/has_/can_/should_)."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


# Construction-Policy metric bands. These are mirrored in the converter's
# generation guidance (conversion_loop/.../py2snow-skill/references/
# SEMS_Compliance.md §9) so generated code targets the same thresholds — keep
# the two in sync if you retune these. Same applies to _EXPR_* and _NEST_*
# below and _MAX_CYCLOMATIC_COMPLEXITY in pre_sonar_check.py.
_FANOUT_SOFT_MIN = 8
_FANOUT_HARD_MIN = 11

# Idiomatic PySpark DataFrame/SparkSession chain methods — excluded from the
# fan-out count. Chaining .withColumn()/.select()/.filter()/... on a single
# DataFrame is standard PySpark style, not the kind of multi-dependency
# coupling the fan-out metric is meant to catch (unlike calling many distinct
# user-defined helper functions/modules).
_SPARK_CHAIN_METHODS = frozenset({
    "withColumn", "withColumnRenamed", "select", "selectExpr", "filter", "where",
    "groupBy", "groupby", "agg", "join", "crossJoin", "drop", "dropDuplicates",
    "distinct", "orderBy", "sort", "alias", "cast", "over", "partitionBy",
    "otherwise", "isNull", "isNotNull", "asc", "desc", "fillna", "na", "dropna",
    "union", "unionByName", "collect", "count", "show", "printSchema", "toDF",
    "createOrReplaceTempView", "createGlobalTempView", "persist", "cache",
    "unpersist", "coalesce", "repartition", "limit", "take", "first", "toPandas",
    "write", "read", "format", "save", "saveAsTable", "option", "options", "mode",
    "table", "load", "schema", "withWatermark", "pivot", "explain",
})


def check_function_fan_out(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """FANOUT001: Count distinct callees per function against the quality-metric bands."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "FANOUT001",
        "Split the function or extract cohesive groups of calls into helpers to reduce coupling.",
    )
    _excluded = _SPARK_CHAIN_METHODS | _STATIC_PYSPARK_FUNCTIONS_ALLOWLIST
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        callees: Set[str] = set()
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name not in _excluded:
                callees.add(name)
        fan_out = len(callees)
        if fan_out < _FANOUT_SOFT_MIN:
            continue
        severity = "hard" if fan_out >= _FANOUT_HARD_MIN else "soft"
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="FANOUT001",
                severity=severity,
                category="syntax",
                message=(
                    f"[FANOUT001] Function '{node.name}' at line {line} calls "
                    f"{fan_out} distinct dependencies (threshold: 7)."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


_EXPR_SOFT_MIN = 6
_EXPR_HARD_MIN = 9


def check_expression_term_count(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """EXPR001: Count terms in boolean/comparison chains against the quality-metric bands."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "EXPR001",
        "Extract named sub-expressions or helper predicates to shorten the expression.",
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            term_count = len(node.values)
        elif isinstance(node, ast.Compare):
            term_count = 1 + len(node.comparators)
        else:
            continue
        if term_count < _EXPR_SOFT_MIN:
            continue
        severity = "hard" if term_count >= _EXPR_HARD_MIN else "soft"
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="EXPR001",
                severity=severity,
                category="syntax",
                message=f"[EXPR001] Expression at line {line} has {term_count} terms (threshold: 5).",
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


_NEST_DECISION_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
_NEST_SOFT_MIN = 5
_NEST_HARD_MIN = 7


def _max_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Return the deepest chain of nested decision/loop/try/with blocks under *node*."""
    deepest = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NEST_DECISION_NODES):
            deepest = max(deepest, _max_nesting_depth(child, current_depth + 1))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # nested scopes are measured independently, on their own pass
        else:
            deepest = max(deepest, _max_nesting_depth(child, current_depth))
    return deepest


def check_decision_nesting_depth(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """NEST001: Tiered decision-nesting-depth check aligned to the policy's bands."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "NEST001",
        "Flatten nested blocks with guard clauses or extract helper functions.",
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        depth = _max_nesting_depth(node)
        if depth < _NEST_SOFT_MIN:
            continue
        severity = "hard" if depth >= _NEST_HARD_MIN else "soft"
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="NEST001",
                severity=severity,
                category="syntax",
                message=(
                    f"[NEST001] Function '{node.name}' at line {line} has decision "
                    f"nesting depth {depth} (threshold: 4)."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


def check_constant_reassignment(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """CONST001: A module-level UPPER_CASE constant should be assigned exactly once."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "CONST001",
        "Assign the constant once at module level, or rename it out of UPPER_CASE if it must vary.",
    )
    # Only top-level module statements — assignments nested in a function/if/
    # loop are a different scope and not a "public static" in the policy sense.
    assignment_lines: Dict[str, List[int]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if target.id.isupper():
                assignment_lines.setdefault(target.id, []).append(getattr(node, "lineno", None))

    violations: List[SEMSViolation] = []
    for name, lines in assignment_lines.items():
        if len(lines) < 2:
            continue
        violations.append(
            SEMSViolation(
                rule_id="CONST001",
                severity="soft",
                category="syntax",
                message=(
                    f"[CONST001] Constant '{name}' is reassigned {len(lines)} times "
                    f"at module level (lines {lines}) — public constants should be read-only."
                ),
                remediation=remediation,
                line_number=lines[-1],
            )
        )
    return violations


def check_single_abstract_concept(
    source_code: str, tree: Optional[ast.AST] = None
) -> List[SEMSViolation]:
    """FILE001: A file should present one abstract concept — flag 2+ top-level classes.

    Construction Policy 2a ("one abstract concept, e.g. one class, per file").
    Only module-body ClassDef nodes are counted; nested/helper classes and
    files with zero classes (the common PySpark ETL script) are never flagged.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    top_level_classes = [
        node for node in getattr(tree, "body", []) if isinstance(node, ast.ClassDef)
    ]
    if len(top_level_classes) < 2:
        return []

    remediation = _remediation(
        "FILE001",
        "Split the file so each module defines a single class / abstract concept.",
    )
    names = ", ".join(cls.name for cls in top_level_classes)
    return [
        SEMSViolation(
            rule_id="FILE001",
            severity="soft",
            category="syntax",
            message=(
                f"[FILE001] File defines {len(top_level_classes)} top-level classes "
                f"({names}) — one abstract concept per file is expected."
            ),
            remediation=remediation,
            line_number=getattr(top_level_classes[0], "lineno", None),
        )
    ]


def check_uncommented_empty_statement(
    source_code: str, tree: Optional[ast.AST] = None
) -> List[SEMSViolation]:
    """STMT001: An intentionally empty statement (``pass`` / ``...``) must be documented.

    Construction Policy 11b ("intentionally empty statements are explicitly
    commented"). A statement is treated as documented when a comment sits on
    the same line or on the line directly above it.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    comment_lines: Set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source_code).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    remediation = _remediation(
        "STMT001",
        "Add a comment on or above the empty statement explaining why it is intentionally empty.",
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        is_pass = isinstance(node, ast.Pass)
        is_ellipsis = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
        )
        if not (is_pass or is_ellipsis):
            continue
        line = getattr(node, "lineno", None)
        if line is None:
            continue
        if line in comment_lines or (line - 1) in comment_lines:
            continue
        kind = "pass" if is_pass else "..."
        violations.append(
            SEMSViolation(
                rule_id="STMT001",
                severity="soft",
                category="documentation",
                message=(
                    f"[STMT001] Empty statement '{kind}' at line {line} has no "
                    "explanatory comment — intentionally empty statements must be documented."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


_STR_LITERAL_MIN_LEN = 2  # ignore single-character strings (often format specifiers)
_STR_LITERAL_EXCLUDED: frozenset = frozenset({"", " ", "\n", "\t"})


def check_repeated_string_literals(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """LIT002: A string literal reused 3+ times in comparisons/binops should be a named constant.

    Mirrors LIT001 (numeric literals) for strings, with the same conservative
    Compare/BinOp scope to avoid flagging every column name or log message —
    only strings repeatedly compared against (e.g. 'if status == "ACTIVE":')
    are the "should be a constant" case this targets.
    """
    _MIN_OCCURRENCES = 3

    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    declared_constant_values: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            declared_constant_values.add(node.value.value)

    occurrences: Dict[str, List[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
        elif isinstance(node, ast.BinOp):
            operands = [node.left, node.right]
        else:
            continue
        line = getattr(node, "lineno", None)
        for operand in operands:
            if not isinstance(operand, ast.Constant) or not isinstance(operand.value, str):
                continue
            value = operand.value
            if (
                value in _STR_LITERAL_EXCLUDED
                or len(value) < _STR_LITERAL_MIN_LEN
                or value in declared_constant_values
            ):
                continue
            occurrences.setdefault(value, []).append(line)

    remediation = _remediation(
        "LIT002",
        "Extract the repeated string literal into a module-level UPPER_CASE constant.",
    )
    violations: List[SEMSViolation] = []
    for value, lines in occurrences.items():
        if len(lines) < _MIN_OCCURRENCES:
            continue
        first_line = lines[0]
        violations.append(
            SEMSViolation(
                rule_id="LIT002",
                severity="soft",
                category="syntax",
                message=(
                    f"[LIT002] Magic string literal {value!r} used {len(lines)} times "
                    f"in conditions/expressions (first at line {first_line}) — "
                    "extract it into a named constant."
                ),
                remediation=remediation,
                line_number=first_line,
            )
        )
    return violations


def _walk_own_scope(node: ast.AST):
    """Yield descendants of *node*, stopping at nested function/class/lambda scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _walk_own_scope(child)


def check_parameter_reassignment(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """PARAM001: A parameter is rebound to a value that does not derive from itself.

    Construction Policy 9f ("in-parameter values are not modified"). PySpark
    idiomatically re-derives a parameter from itself across a transformation
    chain (df = df.filter(...)) — that is NOT flagged. Only a rebind whose
    right-hand side has no reference back to the same parameter name is
    treated as discarding the caller's argument for an unrelated value.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "PARAM001",
        "Assign the new value to a separate local variable instead of overwriting the parameter.",
    )
    violations: List[SEMSViolation] = []

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        param_names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        param_names -= {"self", "cls"}
        if not param_names:
            continue

        seen_lines: Set[int] = set()
        for stmt in _walk_own_scope(fn):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not (isinstance(target, ast.Name) and target.id in param_names):
                continue
            rhs_names = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            if target.id in rhs_names:
                continue  # self-derived chain (df = df.filter(...)) — idiomatic, not flagged
            line = getattr(stmt, "lineno", None)
            if line in seen_lines:
                continue
            seen_lines.add(line)
            violations.append(
                SEMSViolation(
                    rule_id="PARAM001",
                    severity="soft",
                    category="syntax",
                    message=(
                        f"[PARAM001] Parameter '{target.id}' at line {line} is reassigned to "
                        "a value unrelated to its input — the caller's argument is discarded."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    return violations


def check_unused_instance_attributes(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """ATTR001: Flag an instance attribute assigned via self.x = ... but never read.

    Construction Policy 5b ("all instance and class-static variables are
    used"). Scoped to a single class body: an attribute read from outside the
    class (a subclass in another file, external caller, getattr()/dynamic
    access) is invisible here, so a finding is a lead to verify, not a
    guaranteed dead attribute. Dunder attributes, and methods decorated
    @staticmethod/@classmethod (no instance binding), are excluded.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "ATTR001",
        "Remove the attribute if it is genuinely unused, or reference it where it is needed.",
    )
    violations: List[SEMSViolation] = []

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        assigned: Dict[str, int] = {}
        read_names: Set[str] = set()

        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_names = {
                d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                for d in method.decorator_list
            }
            if "staticmethod" in decorator_names or "classmethod" in decorator_names:
                continue
            all_args = (*method.args.posonlyargs, *method.args.args)
            if not all_args or all_args[0].arg != "self":
                continue

            for node in ast.walk(method):
                if (
                    isinstance(node, ast.AugAssign)
                    and isinstance(node.target, ast.Attribute)
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                ):
                    # A += read-modify-writes the attribute — counts as both.
                    read_names.add(node.target.attr)
                    assigned.setdefault(node.target.attr, getattr(node, "lineno", None))
                    continue
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    continue
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    continue
                if isinstance(node.ctx, ast.Store):
                    assigned.setdefault(node.attr, getattr(node, "lineno", None))
                else:
                    read_names.add(node.attr)

        for attr, line in assigned.items():
            if attr in read_names:
                continue
            violations.append(
                SEMSViolation(
                    rule_id="ATTR001",
                    severity="soft",
                    category="syntax",
                    message=(
                        f"[ATTR001] Instance attribute 'self.{attr}' at line {line} in class "
                        f"'{cls.name}' is assigned but never read within the class body."
                    ),
                    remediation=remediation,
                    line_number=line,
                )
            )
    return violations


_MAX_FUNCTION_LINES = 60


def check_function_length(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """STYLE001: A function/method body should not exceed the line-count limit.

    Measured from the def line through the last line of the body (inclusive).
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "STYLE001", "Extract cohesive blocks into helper functions to shorten the body."
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        length = end - start + 1
        if length <= _MAX_FUNCTION_LINES:
            continue
        violations.append(
            SEMSViolation(
                rule_id="STYLE001",
                severity="soft",
                category="syntax",
                message=(
                    f"[STYLE001] Function '{node.name}' at line {start} is {length} lines "
                    f"long (limit: {_MAX_FUNCTION_LINES})."
                ),
                remediation=remediation,
                line_number=start,
            )
        )
    return violations


_MIN_DUPLICATE_STATEMENTS = 4  # mirrors pylint/SonarQube's default min-similarity-lines


def _duplicate_check_statements(body: List[ast.stmt]) -> List[ast.stmt]:
    """Return a function body with a leading docstring stripped, if present."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def check_duplicate_code_blocks(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """
    DUP001: Detect a run of >= _MIN_DUPLICATE_STATEMENTS consecutive statements
    that is structurally identical across two different functions — the
    copy-pasted-transformation pattern ("write clean_columns() once, reuse it")
    that pylint's own duplicate-code checker (R0801) cannot catch here because
    it only compares across multiple files, never within a single one.

    Matching is STRUCTURAL (ast.dump with names retained), so it favors
    precision over recall: identical copy-pasted code is caught; the same
    logic re-implemented with renamed variables/columns is not (that needs
    semantic diffing, out of scope for a static AST rule). Nested functions
    are compared like any other function; a match entirely inside a single
    function (e.g. two branches of an if/else) is not considered, since this
    rule targets cross-function duplication specifically.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    entries: List[Tuple[ast.AST, List[ast.stmt], List[str]]] = []
    for func in functions:
        stmts = _duplicate_check_statements(func.body)
        if len(stmts) < _MIN_DUPLICATE_STATEMENTS:
            continue
        normalized = [ast.dump(s, annotate_fields=False, include_attributes=False) for s in stmts]
        entries.append((func, stmts, normalized))

    remediation = _remediation(
        "DUP001",
        "Extract the shared statements into a helper function and call it from both sites.",
    )
    violations: List[SEMSViolation] = []
    seen_matches: Set[Tuple[str, str, Optional[int], Optional[int]]] = set()

    for i in range(len(entries)):
        func_a, stmts_a, norm_a = entries[i]
        for j in range(i + 1, len(entries)):
            func_b, stmts_b, norm_b = entries[j]
            matcher = difflib.SequenceMatcher(a=norm_a, b=norm_b, autojunk=False)
            for block in matcher.get_matching_blocks():
                if block.size < _MIN_DUPLICATE_STATEMENTS:
                    continue
                line_a = getattr(stmts_a[block.a], "lineno", None)
                line_b = getattr(stmts_b[block.b], "lineno", None)
                key = (func_a.name, func_b.name, line_a, line_b)
                if key in seen_matches:
                    continue
                seen_matches.add(key)
                violations.append(
                    SEMSViolation(
                        rule_id="DUP001",
                        severity="soft",
                        category="syntax",
                        message=(
                            f"[DUP001] {block.size} duplicated statement(s) found in "
                            f"'{func_a.name}' (line {line_a}) and '{func_b.name}' "
                            f"(line {line_b}) — extract a shared helper function."
                        ),
                        remediation=remediation,
                        line_number=line_b,
                    )
                )
    return violations


_COUNT_EMPTINESS_OPS = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)


def check_full_count_for_emptiness(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK002: A bare .count() compared against 0 triggers a full-DataFrame scan.

    df.isEmpty() (Spark 3.3+) or df.limit(1).count() short-circuit after the
    first row; a plain count()==0 / count()>0 check scans the whole
    DataFrame just to answer a yes/no question.
    """
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation("SPARK002", "Replace with df.isEmpty() or df.limit(1).count().")
    violations: List[SEMSViolation] = []
    seen_lines: Set[int] = set()

    def _is_bare_count(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "count"
            and not node.args
            and not node.keywords
        )

    def _is_zero(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, bool)
            and node.value == 0
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], _COUNT_EMPTINESS_OPS):
            continue
        left, right = node.left, node.comparators[0]
        if _is_bare_count(left) and _is_zero(right):
            pass
        elif _is_bare_count(right) and _is_zero(left):
            pass
        else:
            continue
        line = getattr(node, "lineno", None)
        if line in seen_lines:
            continue
        seen_lines.add(line)
        violations.append(
            SEMSViolation(
                rule_id="SPARK002",
                severity="soft",
                category="spark_best_practices",
                message=(
                    f"[SPARK002] .count() compared against 0 at line {line} — "
                    "triggers a full cluster scan just to check emptiness."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


def check_for_to_pandas(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK004: toPandas() collects the entire DataFrame onto the driver."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "SPARK004",
        "Filter to a small result set first, or use Spark write APIs instead of toPandas().",
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "toPandas"
        ):
            continue
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="SPARK004",
                severity="soft",
                category="spark_best_practices",
                message=(
                    f"[SPARK004] .toPandas() at line {line} collects the entire "
                    "DataFrame to the driver."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


def check_for_to_local_iterator(source_code: str, tree: Optional[ast.AST] = None) -> List[SEMSViolation]:
    """SPARK005: toLocalIterator() streams every row to the driver, one partition at a time."""
    if tree is None:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

    remediation = _remediation(
        "SPARK005", "Only use toLocalIterator() on a small, already-filtered DataFrame."
    )
    violations: List[SEMSViolation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "toLocalIterator"
        ):
            continue
        line = getattr(node, "lineno", None)
        violations.append(
            SEMSViolation(
                rule_id="SPARK005",
                severity="soft",
                category="spark_best_practices",
                message=(
                    f"[SPARK005] .toLocalIterator() at line {line} streams every "
                    "row to the driver."
                ),
                remediation=remediation,
                line_number=line,
            )
        )
    return violations


# ── Main compliance entry point ───────────────────────────────────────────────


def check_compliance(source_code: str, cell_index: int = -1) -> ComplianceResult:
    """
    Run all SEMS checks on a generated PySpark source string and return a
    ComplianceResult.

    passed is False when:
      - Any hard violation is present (credentials, syntax errors, unsafe ops, …), OR
      - The overall_score falls below the Reject threshold (60), which catches
        files with many soft violations but no single hard failure.
    """
    # check_script_compliance passes dict keys straight through, which may be
    # section names rather than ints — never compare non-ints against 0.
    if isinstance(cell_index, int):
        context_description = f"cell {cell_index}" if cell_index >= 0 else "notebook"
    else:
        context_description = f"section {cell_index}"
    structured: List[SEMSViolation] = []

    # ── Layer 0: syntax check — abort early on parse failure ──────────────────
    syntax_violations = check_syntax_validity(source_code)
    if syntax_violations:
        structured.extend(syntax_violations)
        violations = [sv.message for sv in structured if sv.severity == "hard"]
        return ComplianceResult(
            passed=False,
            violations=violations,
            warnings=[],
            structured_violations=structured,
        )

    # check_syntax_validity already proved source_code parses; parse once
    # here and hand the shared tree to every check below instead of each of
    # them re-running ast.parse on the same source.
    try:
        tree: Optional[ast.AST] = ast.parse(source_code)
    except SyntaxError:
        tree = None

    # ── Hard violation checks ─────────────────────────────────────────────────
    structured.extend(check_for_hardcoded_credentials(source_code, tree))
    structured.extend(check_for_unsafe_operations(source_code, tree))
    structured.extend(check_pyspark_api_typos(source_code, tree))

    # ── Company rule checks ───────────────────────────────────────────────────
    structured.extend(check_local_spark_master(source_code, tree))
    structured.extend(check_jdbc_no_partition(source_code, tree))
    structured.extend(check_pii_column_write(source_code, tree))
    structured.extend(check_table_naming_convention(source_code, tree))
    structured.extend(check_hard_coded_repartition(source_code, tree))
    structured.extend(check_widget_without_default(source_code, tree))

    # ── Soft violation checks ─────────────────────────────────────────────────
    structured.extend(check_for_error_handling(source_code, tree))
    structured.extend(check_for_logging_usage(source_code, tree))
    structured.extend(check_for_bare_print_statements(source_code, tree))
    structured.extend(check_for_spark_session_misuse(source_code, tree))
    structured.extend(check_driver_side_loops(source_code, tree))
    structured.extend(check_local_file_paths(source_code, tree))
    structured.extend(check_for_pandas_udf(source_code, tree))
    structured.extend(check_for_dbutils_fs_in_loop(source_code, tree))
    structured.extend(check_unbounded_collect(source_code, tree))
    structured.extend(check_spark_actions_try_except(source_code, tree))
    structured.extend(check_module_io_docstring(source_code, tree))
    structured.extend(check_function_docstring_content(source_code, tree))
    structured.extend(check_pii_in_logs(source_code, tree))
    structured.extend(check_bare_except(source_code, tree))

    # ── Construction Policy alignment checks ──────────────────────────────────
    structured.extend(check_repeated_magic_literals(source_code, tree))
    structured.extend(check_repeated_string_literals(source_code, tree))
    structured.extend(check_todo_comments(source_code, tree))
    structured.extend(check_commented_out_code(source_code, tree))
    structured.extend(check_boolean_function_naming(source_code, tree))
    structured.extend(check_function_fan_out(source_code, tree))
    structured.extend(check_expression_term_count(source_code, tree))
    structured.extend(check_decision_nesting_depth(source_code, tree))
    structured.extend(check_constant_reassignment(source_code, tree))
    structured.extend(check_single_abstract_concept(source_code, tree))
    structured.extend(check_uncommented_empty_statement(source_code, tree))
    structured.extend(check_parameter_reassignment(source_code, tree))
    structured.extend(check_unused_instance_attributes(source_code, tree))
    structured.extend(check_function_length(source_code, tree))
    structured.extend(check_duplicate_code_blocks(source_code, tree))

    # ── Spark best-practice checks  ──────
    structured.extend(check_full_count_for_emptiness(source_code, tree))
    structured.extend(check_for_to_pandas(source_code, tree))
    structured.extend(check_for_to_local_iterator(source_code, tree))

    # Derive backward-compatible string lists from structured violations.
    violations = [sv.message for sv in structured if sv.severity == "hard"]
    warnings = [sv.message for sv in structured if sv.severity == "soft"]

    # Hard violations alone fail; also fail when score falls into Reject tier.
    result = ComplianceResult(
        passed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        structured_violations=structured,
    )
    if result.passed and result.overall_score() < _REJECT_SCORE_THRESHOLD:
        result.passed = False

    if not result.passed:
        logger.warning(
            "SEMS violations in %s:\n%s", context_description, result.summary()
        )
    elif warnings:
        logger.info("SEMS warnings in %s:\n%s", context_description, result.summary())
    else:
        logger.info("SEMS compliance: PASSED for %s", context_description)

    return result


def check_script_compliance(translated_sections: dict) -> ComplianceResult:
    """Run compliance checks across all translated sections and merge results."""
    all_violations: List[str] = []
    all_warnings: List[str] = []
    all_structured: List[SEMSViolation] = []
    any_section_failed = False

    for section_index, source_code in translated_sections.items():
        result = check_compliance(source_code, cell_index=section_index)
        if not result.passed:
            # Capture score-only failures that leave result.violations empty.
            any_section_failed = True
        prefix = f"[section {section_index}] "
        all_violations.extend(prefix + v for v in result.violations)
        all_warnings.extend(prefix + w for w in result.warnings)
        for sv in result.structured_violations:
            sv.section_index = section_index
            all_structured.append(sv)

    passed = len(all_violations) == 0 and not any_section_failed
    merged = ComplianceResult(
        passed=passed,
        violations=all_violations,
        warnings=all_warnings,
        structured_violations=all_structured,
    )
    # Soft violations spread across sections can individually pass the per-section
    # threshold but push the combined score below the reject threshold.
    if merged.passed and merged.overall_score() < _REJECT_SCORE_THRESHOLD:
        merged.passed = False
    return merged
