# SEMS Rule Reference

All rules the SEMS compliance gate checks.
`hard` severity → blocks gate.  `soft` → counted as warning.

---

## Category weights (used for overall score)

| Category | Weight |
|---|---|
| syntax | 0.20 |
| security | 0.25 |
| spark_best_practices | 0.20 |
| error_handling | 0.15 |
| documentation | 0.10 |
| databricks_compatibility | 0.10 |

---

## Syntax rules (hard)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SYN001 | hard | Source cannot be parsed by `ast.parse()` | Fix the Python syntax error before any other SEMS check can run |

---

## Security rules (hard)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SEC001 | hard | Hardcoded credential (password, token, secret, API key) | Move to Databricks secret scope: `dbutils.secrets.get(scope='<scope>', key='<key>')` |
| SEC002 | hard | subprocess import — shell execution not permitted in STAM workloads | Remove subprocess; use dbutils or Spark |
| SEC003 | hard | eval()/exec() — dynamic code execution | Refactor to explicit, static logic |
| SHELL_PII001 | hard | PII column written to Delta without masking | SHA-256 hash: `F.sha2(F.col("email"), 256)` before write |
| SHELL_PII002 | hard | PII field logged or printed in plain text | Replace PII value with hash/placeholder in log call |

---

## Syntax / API correctness rules

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SPARK003 | hard | PySpark API typo — function not in allowed list | Correct function name (error message shows closest match) |

---

## Spark best-practices rules (soft)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SPARK001 | soft | Unbounded collect() — no .limit(N) guard | Add `.limit(N)` before `.collect()`, or use `.take(N)` / `.first()` |
| SPARK002 | soft | Full `.count()` — triggers a full cluster scan | Replace with `df.isEmpty()` or `df.limit(1).count()` |
| SPARK004 | soft | toPandas() — collects entire DataFrame to driver | Filter to small result set first, or use Spark write APIs |
| SPARK005 | soft | toLocalIterator() — streams all rows to driver | Only use on small, already-filtered DataFrames |
| SPARK006 | soft | Driver-side loop iterating over Spark data | Replace with `withColumn()`, `groupBy()`, `filter()` |
| SPARK007 | soft | pandas_udf — prefer native PySpark functions | Replace with native `pyspark.sql.functions` equivalents |
| SPARK008 | soft | Multiple SparkSession.builder calls | Use cluster-provided `spark`; call builder at most once |
| STYLE001 | soft | Function exceeds 60-line limit | Extract helper functions |
| SHELL_SPARK001 | soft | Hard-coded partition count in repartition(N) | Use `spark.conf.get("spark.sql.shuffle.partitions", "200")` or let AQE manage |
| SHELL_SPARK002 | hard | JDBC write without partitionColumn — single-partition bottleneck | Add `.option("partitionColumn", ...)` + lowerBound/upperBound/numPartitions |

---

## Error handling rules (soft)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SEMS_ERR001 | soft | Spark action (.count()/.collect()) not wrapped in try/except | Wrap in `try/except`, log, re-raise |
| ERR001 | soft | No try/except block present anywhere in the module | Wrap external calls (Spark actions, file I/O, JDBC) in try/except so failures are logged and re-raised instead of aborting the job |
| ERR002 | soft | Bare except/except Exception without re-raise | Use specific exception type; always log and re-raise |
| LOG001 | soft | logging module not imported or logger not used | Add `import logging`; replace `print()` with `logger.*()` |
| LOG002 | soft | Bare print() statements | Replace with `logger.info()` or `logger.debug()` |
| LOG003 | soft | Possible PII or secret in log message | Redact the value before logging |

---

## Documentation rules (soft)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| DOC001 | soft | Module docstring missing Input:/Output: sections | Add I/O sections naming Delta tables and execution context |
| DOC002 | soft | Function/method has a docstring but doesn't document every parameter | Add each parameter under an Args: section |
| DOC003 | soft | Function/method returns/yields a value but its docstring has no Returns:/Yields: section | Add a Returns:/Yields: section (exempt: `@property`/`@cached_property`) |

---

## Databricks compatibility rules (soft)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SPARK009 | soft | Local filesystem path — use DBFS or cloud storage URIs | Replace `/home/`, `/tmp/`, `C:\` with `/dbfs/mnt/...` or `s3://`, `abfss://` |
| SPARK010 | soft | dbutils.fs call inside a loop | Use `spark.read` for directory reads; collect paths first |
| SHELL_STAM001 | hard | SparkSession with `.master('local')` | Remove `.master(...)` — cluster injects SparkSession via `spark` variable |
| SHELL_STAM002 | soft | Widget value read without a default | Add `dbutils.widgets.text("<name>", "<default>")` before the `get` call |

---

## Shell naming-convention rules (soft)

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| SHELL_NAME001 | soft | Delta table name not in `<domain>_<entity>_<layer>` format | Rename to e.g. `trading_position_raw`, `hr_employee_curated` |

---

## Construction Policy alignment rules

These rules implement Shell's software Construction Policy expectations that are
statically checkable. Severity follows the Quality Metrics bands where relevant:
a finding in the "needs justification" band is `soft`; a finding in the
"unacceptable" band is `hard`.

| ID | Severity | Policy ref | Description | Fix |
|----|----------|-----------|-------------|-----|
| LIT001 | soft | 6a | Numeric literal reused 3+ times in conditions/expressions | Extract into a module-level UPPER_CASE constant |
| LIT002 | soft | 6a | String literal reused 3+ times in conditions/expressions | Extract into a module-level UPPER_CASE constant |
| COM001 | soft | 3e | TODO/FIXME/XXX marker present | Resolve or remove before release; track follow-up in the SCM/issue tracker |
| COM002 | soft | 3d | Commented-out code detected | Delete it — Git history preserves it |
| FUNC001 | soft | 8b | `-> bool` function not named as a predicate | Rename to `is_`/`has_`/`can_`/`should_` style |
| EXPR001 | soft / hard | 10a | Boolean/comparison chain with too many terms | Extract named sub-expressions/helper predicates |
| FANOUT001 | soft / hard | 9d | Function calls too many distinct dependencies | Split the function or extract cohesive call groups |
| NEST001 | soft / hard | 9d | Decision nesting depth too deep | Flatten with guard clauses/early returns or extract helpers |
| CONST001 | soft | 7e | Module-level UPPER_CASE constant reassigned | Assign once; rename out of UPPER_CASE if it must vary |
| FILE001 | soft | 2a | File defines more than one top-level class | Split so each module holds one class/abstract concept |
| STMT001 | soft | 11b | Intentionally empty statement (`pass`/`...`) not commented | Add a comment on or directly above the empty statement |
| PARAM001 | soft | 9f | Parameter reassigned to a value unrelated to its input | Assign to a new local variable instead; self-derived chains (`df = df.filter(...)`) are not flagged |
| ATTR001 | soft | 5b | `self.x` assigned but never read within the class body | Remove the attribute, or reference it where needed |
| DUP001 | soft | — * | 4+ structurally-identical statements repeated across two functions | Extract the shared block into a helper function and call it from both sites |

\* No numbered clause in the current Construction Policy document — DUP001 operationalizes
the general "never duplicate transformations" principle, and fills the gap left by
pylint's `duplicate-code` (R0801), which only compares across multiple files and so
cannot see duplication within a single generated script.

## Quality Metrics bands (Construction Policy)

The metric thresholds below are enforced exactly as the policy specifies.
"Needs justification" maps to `soft`; "unacceptable" maps to `hard`.

| Metric | Scope | Rule / tool | Acceptable | Needs justification (soft) | Unacceptable (hard) |
|--------|-------|-------------|------------|----------------------------|---------------------|
| Cyclomatic complexity | Method | radon (Layer 3) | 1–9 | 10–14 (MAJOR) | 15+ (CRITICAL) |
| Decision nesting depth | Method | NEST001 | 1–4 | 5–6 | 7+ |
| Fan out | Method | FANOUT001 | 0–7 | 8–10 | 11+ |
| Number of terms | Expression | EXPR001 | 1–5 | 6–8 | 9+ |

---

## Auto-fixable rules

SPARK003, DOC001, DOC002, DOC003, SEMS_ERR001, SPARK001, ERR002, LOG001, F821 (flake8), E0602 (pylint), name-defined (mypy)
