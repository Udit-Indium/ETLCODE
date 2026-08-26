---
name: sems-check-skill
description: >
  Run the full 3-layer SEMS compliance check on a converted PySpark file and
  report the per-BRD-area breakdown and every blocking-severity finding. This
  is a compliance report, not a pass/fail gate.  Triggers: "run sems check",
  "check sems compliance", "validate sems", "sems check", "compliance check",
  "sems gate", "check the converted file".
compatibility: PySpark 3.5+; Shell STAM; Databricks
---

# SEMS Compliance Check Skill

Run the full 3-layer SEMS compliance check on the converted PySpark file.
All rule definitions and remediation hints are in
`references/sems_rules_reference.md` — read that file to interpret results.

## The 3 Layers

| Layer | What it checks | Blocking-severity? |
|-------|---------------|--------------|
| 1 — Syntax | `ast.parse` — valid Python | Always reported |
| 2 — SEMS rules | 30+ AST + regex rules across 6 categories (incl. Construction Policy) | Hard violations only |
| 3 — External tools | black, isort, flake8, pylint, pylint-simplify, mypy, bandit, radon, pydocstyle, pytest-cov | MAJOR+ findings only |

## BRD Areas checked

| BRD Area | SEMS rules mapped | External tools mapped |
|----------|-------------|-------------|
| Code readability and structure | SYN001, DOC001, DOC002, DOC003, SPARK003, SPARK009, SHELL_NAME001, SHELL_NAME002, LIT001, COM001, COM002, FUNC001, EXPR001, CONST001, STMT001 | black, isort, flake8, pylint, pydocstyle |
| Modular design | STYLE001, SPARK001, SPARK002, SPARK004–008, SPARK010, SHELL_SPARK001, SHELL_SPARK002, SHELL_STAM001, SHELL_STAM002, FANOUT001, NEST001, FILE001, DUP001 | pylint-simplify, radon, pytest-cov |
| Logging practices | LOG001, LOG002, LOG003 | — |
| Error handling | SEMS_ERR001, ERR001, ERR002 | mypy |
| Security | SEC001, SEC002, SEC003, SHELL_PII001, SHELL_PII002 | bandit |

---

## Workflow (follow every time)

**STEP 1 — Call `sems_check_tool` exactly once.**
Do NOT skip this call or fabricate results. Pass `converted_file_path` only when
the caller specifies a path; otherwise leave it empty and the tool finds the
newest `*_spark.py` in `outputs/` automatically.

The tool returns:
- `report` — full markdown (use this verbatim in your reply if the caller asks)
- `errors` — structured list: category, title, reason, severity, location, suggestion
- `area_summaries` — per-BRD-area: passed, sems_findings, tool_findings, sems_blocking, tool_blocking
  (`passed` here means "no blocking-severity findings in this area" — it is not an overall verdict)

**STEP 2 — Report the compliance summary.**
Output in this exact order:

1. BRD Area Coverage table (one row per area):

   | BRD Area | Status | SEMS findings | Tool findings | Blocking |
   |----------|--------|---------------|---------------|----------|
   | ...      | PASS / FAIL | N | N | N |

2. **Top blocking-severity findings** (all hard + MAJOR+ items):
   For each: rule_id, severity, location, one-line description.

**STEP 3 — State next steps.**
- **If no blocking-severity findings**: "No blocking-severity SEMS findings."
- **If there are blocking-severity findings**:
  - List auto-fixable rules → "These can be resolved by a separate auto-fix pass."
  - List manual-fix rules → "These require developer intervention."
  - See `references/sems_rules_reference.md` for per-rule fix guidance.

---

## Non-negotiable rules

- **Call `sems_check_tool` exactly once per turn** — never call it twice.
- **Never fabricate or guess results** — report only what the tool returned.
- **Always show the BRD area table** — it gives the developer the context to prioritise.
- **This is a compliance report, not a pass/fail gate** — never phrase the result as PASS/FAIL overall; hard SEMS violations and MAJOR+ tool findings are flagged as blocking-severity, surface them prominently.
- **Do not apply fixes yourself** — this agent only reports; both auto-fixable and manual fixes stay outside its scope.

## ACT, DON'T NARRATE

Call `sems_check_tool`, then immediately report results. Never reply with
"I will now run the check…" and stop. Only pause if the tool genuinely errors.
