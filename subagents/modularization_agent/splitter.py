"""Deterministic AST-based module splitter.

Repackages a single finished script (the output of the conversion + SEMS
compliance stages) into the Shell modularization layout:

    main.py        business-logic functions (the pipeline's own logic)
    utilities.py   reusable helper functions + any classes, generic enough
                   to not depend on main.py
    config.py      module-level configuration constants
    usage.py       a runnable example (from the source's own
                   ``if __name__ == "__main__":`` block, or a synthesized one)

No model call is involved: naming/classification comes entirely from AST
analysis (call-graph fan-in/fan-out plus a naming-prefix convention), the same
philosophy as ``subagents/script_refactor`` — deterministic, no credentials,
same output every run.

The one invariant every render step preserves: **utilities.py never imports
from main.py**. That's what keeps main/utilities/config a DAG instead of a
cycle. See ``_break_utility_to_main_cycles``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

#: A function whose name starts with one of these is treated as a generic,
#: reusable helper regardless of how many callers it has.
_UTILITY_PREFIXES = (
    "validate_", "format_", "log_", "parse_", "clean_", "sanitize_",
    "normalize_", "is_", "has_", "can_", "hash_", "load_config", "get_config",
)

#: Common orchestrator/entrypoint names, checked before falling back to
#: call-graph heuristics.
_ENTRYPOINT_NAMES = ("main", "run", "run_pipeline", "run_job", "execute")


@dataclass
class FunctionInfo:
    """One top-level function/async-function and its place in the call graph."""

    name: str
    node: ast.AST
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    category: str = "main"  # "main" | "utility" | "entrypoint"


@dataclass
class ModularizationResult:
    """Output of :func:`split_module`. Check ``ok`` before using the code fields."""

    ok: bool
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    main_code: str = ""
    utilities_code: str = ""
    config_code: str = ""
    usage_code: str = ""
    main_functions: List[str] = field(default_factory=list)
    utility_functions: List[str] = field(default_factory=list)
    entrypoint: Optional[str] = None


# ── AST helpers ────────────────────────────────────────────────────────────


def _segment(source: str, node: ast.AST) -> str:
    """Return the exact source text for ``node``, decorators included.

    ``ast.get_source_segment`` alone can omit decorator lines (a decorated
    ``FunctionDef``'s own ``lineno`` points at the ``def`` keyword, not the
    first decorator), so a decorated node is sliced by line range instead.
    """
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        lines = source.splitlines(keepends=True)
        start = decorators[0].lineno
        end = node.end_lineno or start
        return "".join(lines[start - 1 : end]).rstrip("\n")
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive, ast.unparse rarely fails
        return ""


def _is_main_guard(node: ast.stmt) -> bool:
    """True for a top-level ``if __name__ == "__main__":`` block."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]

    def _is_dunder_name(n: ast.AST) -> bool:
        return isinstance(n, ast.Name) and n.id == "__name__"

    def _is_main_str(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and n.value == "__main__"

    return (_is_dunder_name(left) and _is_main_str(right)) or (
        _is_main_str(left) and _is_dunder_name(right)
    )


def _is_config_constant(node: ast.stmt) -> bool:
    """True for a module-level ``UPPER_CASE = <expr>`` (or annotated) assignment.

    Unlike ``script_refactor``'s hoisting check, the value is not required to
    be ``ast.literal_eval``-able — real configs nest things like ``np.inf``
    inside a list, which isn't a literal but is still exactly the kind of
    value that belongs in config.py.
    """
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return False
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        target = node.target
    else:
        return False
    return target.id.isupper() and target.id.isidentifier()


def _assign_target_name(node: ast.stmt) -> Optional[str]:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _bound_names(node: ast.stmt) -> Set[str]:
    """Names an import statement binds into the module namespace."""
    names: Set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            names.add("*" if alias.name == "*" else (alias.asname or alias.name))
    return names


def _free_names(nodes: Sequence[ast.AST]) -> Set[str]:
    """Every identifier referenced anywhere within ``nodes``."""
    names: Set[str] = set()
    for node in nodes:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                names.add(inner.id)
    return names


def _imports_for(imports: Sequence[ast.stmt], needed_names: Set[str]) -> List[ast.stmt]:
    """Imports whose bound name(s) are referenced by ``needed_names``.

    A partially-used ``from x import a, b, c`` is kept whole rather than
    pruned to only the used names — simpler and safer than rewriting the
    import statement, at the cost of an occasional unused-import lint note
    (a separate, already-handled SEMS concern, not this stage's job).
    """
    selected = []
    for imp in imports:
        bound = _bound_names(imp)
        if "*" in bound or bound & needed_names:
            selected.append(imp)
    return selected


def _top_level_bound_names(nodes: Sequence[ast.stmt]) -> Set[str]:
    """Names these statements bind at module level — assignment targets plus
    def/class names. Used to compute what a sibling file can import from
    utilities.py, e.g. a plain ``logger = logging.getLogger(...)`` setup
    statement makes ``logger`` importable just like a function would.
    """
    names: Set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _is_utility_name(name: str) -> bool:
    stripped = name.lstrip("_")
    return stripped.startswith(_UTILITY_PREFIXES)


def _detect_entrypoint(
    functions: Dict[str, FunctionInfo],
    function_order: List[str],
    main_guard: Optional[ast.If],
) -> Optional[str]:
    """Pick the function that represents "run the whole thing"."""
    if main_guard is not None:
        called = [
            n.func.id
            for n in ast.walk(main_guard)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions
        ]
        if called:
            orchestrators = [c for c in called if functions[c].calls]
            if orchestrators:
                return max(orchestrators, key=lambda c: len(functions[c].calls))
            return called[0]

    for name in _ENTRYPOINT_NAMES:
        if name in functions:
            return name

    roots = [name for name in function_order if not functions[name].called_by]
    if roots:
        return max(roots, key=lambda name: len(functions[name].calls))

    return max(function_order, key=lambda name: len(functions[name].calls)) if function_order else None


def _break_utility_to_main_cycles(functions: Dict[str, FunctionInfo], main_set: Set[str]) -> Set[str]:
    """Reclassify any "utility" function that calls a "main" function.

    utilities.py must never import from main.py (that would make main.py and
    utilities.py import each other). A function initially tagged "utility"
    purely by naming or reuse count, but that itself calls pipeline-specific
    "main" logic, is not actually a generic helper — moving it to main.py is
    both the safe choice and the more honest classification.

    Returns the set of names reclassified this pass (the caller loops until
    this is empty, since one reclassification can cascade into another).
    """
    newly_reclassified: Set[str] = set()
    for name, info in functions.items():
        if info.category == "utility" and info.calls & main_set:
            info.category = "main"
            main_set.add(name)
            newly_reclassified.add(name)
    return newly_reclassified


# ── Rendering ────────────────────────────────────────────────────────────


def _render_file(
    source: str,
    docstring: Optional[str],
    imports: Sequence[ast.stmt],
    extra_import_lines: Sequence[str],
    body_nodes: Sequence[ast.AST],
) -> str:
    parts: List[str] = []
    if docstring:
        parts.append(docstring.strip())
    import_lines = [_segment(source, imp) for imp in imports]
    import_lines.extend(extra_import_lines)
    if import_lines:
        parts.append("\n".join(import_lines))
    for node in body_nodes:
        segment = _segment(source, node)
        if segment.strip():
            parts.append(segment)
    return "\n\n\n".join(p for p in parts if p and p.strip()) + "\n"


# ── Entry point ──────────────────────────────────────────────────────────


def split_module(source: str, module_basename: str = "main") -> ModularizationResult:
    """Split a single script's source into the modularized file contents.

    Args:
        source: the complete, already SEMS-compliant script text.
        module_basename: used only for docstrings/messages (e.g. "Configuration
            values for <module_basename>.") — never affects import names, which
            are always the fixed ``main``/``utilities``/``config`` module names.

    Returns:
        A :class:`ModularizationResult`. Check ``ok`` before using the code
        fields — a false ``ok`` means the input could not be split.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ModularizationResult(ok=False, error=f"input does not parse at line {exc.lineno}: {exc.msg}")

    warnings: List[str] = []
    body = list(tree.body)

    docstring_node: Optional[ast.stmt] = None
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            docstring_node = body[0]
            body = body[1:]

    imports: List[ast.stmt] = []
    constants: List[ast.stmt] = []
    classes: List[ast.ClassDef] = []
    functions: Dict[str, FunctionInfo] = {}
    function_order: List[str] = []
    setup_stmts: List[ast.stmt] = []
    main_guard: Optional[ast.If] = None

    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                warnings.append(
                    f"line {node.lineno}: `from ... import *` — every generated file "
                    "keeps this import, since which names it binds can't be known statically."
                )
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = FunctionInfo(name=node.name, node=node)
            function_order.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        elif _is_main_guard(node):
            main_guard = node
        elif _is_config_constant(node):
            constants.append(node)
        else:
            setup_stmts.append(node)

    if not functions:
        return ModularizationResult(
            ok=False,
            error="no top-level functions found in the source — nothing to modularize into main.py/utilities.py.",
            warnings=warnings,
        )

    # Call graph among top-level functions (nested defs, e.g. a closure inside
    # one function, are invisible here by design — they travel with their
    # parent function's source segment automatically).
    for name, info in functions.items():
        for inner in ast.walk(info.node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in functions
                and inner.func.id != name
            ):
                info.calls.add(inner.func.id)
    for name, info in functions.items():
        for callee in info.calls:
            functions[callee].called_by.add(name)

    entrypoint = _detect_entrypoint(functions, function_order, main_guard)
    if entrypoint is None:
        warnings.append(
            "no entrypoint/orchestrator function could be detected — usage.py lists "
            "every main.py function instead of calling one end-to-end pipeline function."
        )

    main_set: Set[str] = {entrypoint} if entrypoint else set()
    for name, info in functions.items():
        if name == entrypoint:
            info.category = "entrypoint"
            continue
        is_utility = _is_utility_name(name) or len(info.called_by) >= 2
        info.category = "utility" if is_utility else "main"
        if info.category == "main":
            main_set.add(name)

    reclassified: Set[str] = set()
    for _ in range(len(functions)):
        newly_reclassified = _break_utility_to_main_cycles(functions, main_set)
        if not newly_reclassified:
            break
        reclassified |= newly_reclassified
    if reclassified:
        warnings.append(
            f"{len(reclassified)} function(s) looked like reusable helpers by name/reuse "
            f"but call main.py logic, so they were kept in main.py to avoid a "
            f"utilities.py -> main.py import cycle: {', '.join(sorted(reclassified))}."
        )

    main_functions = [n for n in function_order if functions[n].category in ("main", "entrypoint")]
    utility_functions = [n for n in function_order if functions[n].category == "utility"]

    config_names = {name for node in constants if (name := _assign_target_name(node))}
    utility_names = (
        set(utility_functions) | {c.name for c in classes} | _top_level_bound_names(setup_stmts)
    )

    # ── config.py ──
    if constants:
        config_imports = _imports_for(imports, _free_names(constants))
        config_doc = f'"""Configuration values for {module_basename}."""'
        config_code = _render_file(source, config_doc, config_imports, [], constants)
    else:
        warnings.append("no module-level UPPER_CASE constants were found to hoist into config.py.")
        config_code = (
            f'"""Configuration values for {module_basename}.\n\n'
            "No module-level constants were found in the source to hoist here — "
            'add configuration values as needed.\n"""\n'
        )

    # ── utilities.py ──
    utility_nodes: List[ast.AST] = [*classes, *setup_stmts, *(functions[n].node for n in utility_functions)]
    if utility_nodes:
        utility_free = _free_names(utility_nodes)
        utility_imports = _imports_for(imports, utility_free)
        config_refs = sorted(utility_free & config_names)
        extra = [f"from config import {', '.join(config_refs)}"] if config_refs else []
        utilities_doc = f'"""Reusable helper functions for {module_basename}."""'
        utilities_code = _render_file(source, utilities_doc, utility_imports, extra, utility_nodes)
    else:
        utilities_code = (
            f'"""Reusable helper functions for {module_basename}.\n\n'
            "No generic, reusable helpers were extracted — every function in the "
            'source was pipeline-specific business logic and lives in main.py.\n"""\n'
        )

    # ── main.py ──
    main_nodes: List[ast.AST] = [functions[n].node for n in main_functions]
    main_free = _free_names(main_nodes)
    main_imports = _imports_for(imports, main_free)
    config_refs = sorted(main_free & config_names)
    utility_refs = sorted(main_free & utility_names)
    extra = []
    if config_refs:
        extra.append(f"from config import {', '.join(config_refs)}")
    if utility_refs:
        extra.append(f"from utilities import {', '.join(utility_refs)}")
    main_doc = _segment(source, docstring_node) if docstring_node else f'"""Business logic for {module_basename}."""'
    main_code = _render_file(source, main_doc, main_imports, extra, main_nodes)

    # ── usage.py ──
    if main_guard is not None:
        guard_free = _free_names([main_guard])
        guard_imports = _imports_for(imports, guard_free)
        extra = []
        main_refs = sorted(guard_free & set(main_functions))
        utility_refs = sorted(guard_free & utility_names)
        config_refs = sorted(guard_free & config_names)
        if main_refs:
            extra.append(f"from main import {', '.join(main_refs)}")
        if utility_refs:
            extra.append(f"from utilities import {', '.join(utility_refs)}")
        if config_refs:
            extra.append(f"from config import {', '.join(config_refs)}")
        usage_doc = f'"""Example usage of {module_basename}."""'
        usage_code = _render_file(source, usage_doc, guard_imports, extra, [main_guard])
    else:
        entry = entrypoint or (main_functions[0] if main_functions else None)
        if entry:
            params = [a.arg for a in functions[entry].node.args.args]
            call_args = ", ".join(f"{p}=None" for p in params)
            comment = "  # TODO: supply real argument values" if params else ""
            usage_code = (
                f'"""Example usage of {module_basename}."""\n\n'
                f"from main import {entry}\n\n\n"
                'if __name__ == "__main__":\n'
                f"    result = {entry}({call_args}){comment}\n"
                "    print(result)\n"
            )
        else:  # pragma: no cover - unreachable, `functions` was checked non-empty above
            usage_code = (
                f'"""Example usage of {module_basename}.\n\n'
                "No entrypoint could be detected automatically — import the function(s) "
                'you need from main and call them here.\n"""\n'
            )

    return ModularizationResult(
        ok=True,
        warnings=warnings,
        main_code=main_code,
        utilities_code=utilities_code,
        config_code=config_code,
        usage_code=usage_code,
        main_functions=main_functions,
        utility_functions=utility_functions,
        entrypoint=entrypoint,
    )


# ── CLI ──────────────────────────────────────────────────────────────────


def _cli(argv: Optional[List[str]] = None) -> int:
    """
        python -m subagents.modularization_agent.splitter path/to/converted_script.py -o outdir/
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="modularization_agent.splitter",
        description="Deterministically split a flat script into the Shell modularization layout.",
    )
    parser.add_argument("input", help="the converted .py script to split")
    parser.add_argument("-o", "--output-dir", help="directory to write main.py/utilities.py/config.py/usage.py into")
    args = parser.parse_args(argv)

    path = Path(args.input)
    result = split_module(path.read_text(encoding="utf-8"), module_basename=path.stem)

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print(f"entrypoint: {result.entrypoint}", file=sys.stderr)
    print(f"main.py functions: {', '.join(result.main_functions)}", file=sys.stderr)
    print(f"utilities.py functions: {', '.join(result.utility_functions) or '(none)'}", file=sys.stderr)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "main.py").write_text(result.main_code, encoding="utf-8")
        (out_dir / "utilities.py").write_text(result.utilities_code, encoding="utf-8")
        (out_dir / "config.py").write_text(result.config_code, encoding="utf-8")
        (out_dir / "usage.py").write_text(result.usage_code, encoding="utf-8")
        print(f"written to {out_dir}", file=sys.stderr)
    else:
        for label, code in (
            ("config.py", result.config_code),
            ("utilities.py", result.utilities_code),
            ("main.py", result.main_code),
            ("usage.py", result.usage_code),
        ):
            print(f"\n# ===== {label} =====\n")
            print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
