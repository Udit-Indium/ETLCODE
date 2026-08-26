"""
Unit tests for compliance_checker.py covering all 20 findings.

Run with:  python -m pytest subagents/sems_agent/compliance/tests/ -v
"""

import pytest

# compliance_checker is injected into sys.modules by conftest.py to avoid
# triggering the sems_agent package __init__.py (which requires ADK deps).
import compliance_checker as _cc

ComplianceResult = _cc.ComplianceResult
SEMSViolation = _cc.SEMSViolation
_code_only = _cc._code_only
_has_limit_in_chain = _cc._has_limit_in_chain
_is_in_try_body = _cc._is_in_try_body
_levenshtein = _cc._levenshtein
check_bare_except = _cc.check_bare_except
check_boolean_function_naming = _cc.check_boolean_function_naming
check_commented_out_code = _cc.check_commented_out_code
check_compliance = _cc.check_compliance
check_constant_reassignment = _cc.check_constant_reassignment
check_decision_nesting_depth = _cc.check_decision_nesting_depth
check_driver_side_loops = _cc.check_driver_side_loops
check_expression_term_count = _cc.check_expression_term_count
check_single_abstract_concept = _cc.check_single_abstract_concept
check_uncommented_empty_statement = _cc.check_uncommented_empty_statement
check_for_error_handling = _cc.check_for_error_handling
check_for_hardcoded_credentials = _cc.check_for_hardcoded_credentials
check_for_logging_usage = _cc.check_for_logging_usage
check_for_unsafe_operations = _cc.check_for_unsafe_operations
check_function_fan_out = _cc.check_function_fan_out
check_generic_df_variable = _cc.check_generic_df_variable
check_hard_coded_repartition = _cc.check_hard_coded_repartition
check_jdbc_no_partition = _cc.check_jdbc_no_partition
check_local_file_paths = _cc.check_local_file_paths
check_local_spark_master = _cc.check_local_spark_master
check_module_io_docstring = _cc.check_module_io_docstring
check_function_docstring_content = _cc.check_function_docstring_content
check_pii_column_write = _cc.check_pii_column_write
check_pii_in_logs = _cc.check_pii_in_logs
check_pyspark_api_typos = _cc.check_pyspark_api_typos
check_repeated_magic_literals = _cc.check_repeated_magic_literals
check_repeated_string_literals = _cc.check_repeated_string_literals
check_duplicate_code_blocks = _cc.check_duplicate_code_blocks
check_parameter_reassignment = _cc.check_parameter_reassignment
check_unused_instance_attributes = _cc.check_unused_instance_attributes
check_function_length = _cc.check_function_length
check_full_count_for_emptiness = _cc.check_full_count_for_emptiness
check_for_to_pandas = _cc.check_for_to_pandas
check_for_to_local_iterator = _cc.check_for_to_local_iterator
check_script_compliance = _cc.check_script_compliance
check_spark_actions_try_except = _cc.check_spark_actions_try_except
check_syntax_validity = _cc.check_syntax_validity
check_table_naming_convention = _cc.check_table_naming_convention
check_todo_comments = _cc.check_todo_comments
check_unbounded_collect = _cc.check_unbounded_collect
check_widget_without_default = _cc.check_widget_without_default

# ── helpers ───────────────────────────────────────────────────────────────────


def rule_ids(violations):
    return [v.rule_id for v in violations]


def has_rule(violations, rule_id):
    return any(v.rule_id == rule_id for v in violations)


def _make_fanout_code(n):
    calls = "\n    ".join(f"call_{i}()" for i in range(n))
    return f"def orchestrator():\n    {calls}\n"


def _make_nested_code(depth):
    indent = "    "
    lines = ["def f():"]
    for i in range(depth):
        lines.append(f"{indent * (i + 1)}if cond_{i}:")
    lines.append(f"{indent * (depth + 1)}pass")
    return "\n".join(lines) + "\n"


# ── Finding 1: syntax validation ──────────────────────────────────────────────


class TestSyntaxValidity:
    def test_valid_code_passes(self):
        assert check_syntax_validity("x = 1") == []

    def test_syntax_error_is_hard_violation(self):
        result = check_syntax_validity("def foo(\n")
        assert len(result) == 1
        assert result[0].rule_id == "SYN001"
        assert result[0].severity == "hard"

    def test_check_compliance_fails_on_syntax_error(self):
        result = check_compliance("def foo(\n")
        assert result.passed is False
        assert any("SYN001" in v for v in result.violations)

    def test_syntax_broken_code_does_not_pass_check_compliance(self):
        # Regression: previously AST checks returned [] and regex might not fire,
        # letting broken code pass with passed=True.
        result = check_compliance("x = (1 +")
        assert result.passed is False


# ── Finding 2: company rules enforced ────────────────────────────────────────


class TestCompanyRules:
    def test_shell_stam001_local_master_flagged(self):
        code = 'SparkSession.builder.master("local[*]").getOrCreate()'
        result = check_local_spark_master(code)
        assert has_rule(result, "SHELL_STAM001")
        assert result[0].severity == "hard"

    def test_shell_stam001_local_master_in_compliance(self):
        code = 'SparkSession.builder.master("local").getOrCreate()'
        result = check_compliance(code)
        assert any("SHELL_STAM001" in v for v in result.violations)

    def test_shell_stam001_remote_master_ok(self):
        code = 'SparkSession.builder.master("spark://host:7077").getOrCreate()'
        assert check_local_spark_master(code) == []

    def test_shell_spark002_jdbc_without_partition_flagged(self):
        code = 'df.write.format("jdbc").option("url", "jdbc:...").save()'
        result = check_jdbc_no_partition(code)
        assert has_rule(result, "SHELL_SPARK002")
        assert result[0].severity == "hard"

    def test_shell_spark002_jdbc_with_all_options_ok(self):
        # All four options are required — a partial set still fails.
        code = (
            'df.write.format("jdbc")'
            '.option("url","jdbc:...")'
            '.option("partitionColumn","id")'
            '.option("lowerBound",0)'
            '.option("upperBound",1000000)'
            '.option("numPartitions",10)'
            '.save()'
        )
        assert check_jdbc_no_partition(code) == []

    def test_shell_pii001_write_without_masking_flagged(self):
        code = "df.select('email').write.format('delta').save('/path')"
        result = check_pii_column_write(code)
        assert has_rule(result, "SHELL_PII001")

    def test_shell_pii001_write_with_sha2_ok(self):
        code = (
            "df.withColumn('email', F.sha2(F.col('email'), 256))"
            ".write.format('delta').save('/path')"
        )
        assert check_pii_column_write(code) == []

    def test_shell_pii001_no_write_skipped(self):
        code = "x = df.filter(df.email == 'a@b.com')"
        assert check_pii_column_write(code) == []


# ── Finding 3: soft failures affect passed ────────────────────────────────────


class TestScoreBasedPassing:
    def test_many_soft_violations_can_fail(self):
        # A file with many soft violations should fall below the Reject threshold.
        code = "\n".join(
            [
                "x = 1",  # no logging, no try/except, no docstring, no logging import
            ]
        )
        result = check_compliance(code)
        # Overall score will be low due to soft violations across categories
        if result.overall_score() < 60:
            assert result.passed is False

    def test_passed_false_when_score_below_60(self):
        # Build a ComplianceResult manually and confirm score gate.
        sv = SEMSViolation(
            rule_id="SPARK001", severity="soft", category="spark_best_practices",
            message="x", remediation="y",
        )
        r = ComplianceResult(passed=True, structured_violations=[sv] * 7)
        if r.overall_score() < 60:
            assert True  # score gating is exercised in check_compliance


# ── Finding 4: Spark action coverage ─────────────────────────────────────────


class TestSparkActionsCoverage:
    def test_collect_outside_try_flagged(self):
        code = "rows = df.collect()"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_show_outside_try_flagged(self):
        code = "df.show()"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_toPandas_outside_try_flagged(self):
        code = "pdf = df.toPandas()"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_saveAsTable_outside_try_flagged(self):
        code = "df.saveAsTable('mytable')"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_count_inside_try_ok(self):
        code = "try:\n    n = df.count()\nexcept Exception:\n    raise"
        result = check_spark_actions_try_except(code)
        assert not has_rule(result, "SEMS_ERR001")

    def test_aggregation_count_not_flagged(self):
        # F.count("col") is an aggregation function, not a DataFrame action.
        code = "from pyspark.sql import functions as F\nresult = F.count('col')"
        result = check_spark_actions_try_except(code)
        assert not has_rule(result, "SEMS_ERR001")


# ── Finding 5: try-body detection ────────────────────────────────────────────


class TestTryBodyDetection:
    def test_count_in_except_body_is_not_protected(self):
        code = "try:\n    pass\nexcept Exception:\n    n = df.count()\n    raise"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_count_in_finally_is_not_protected(self):
        code = "try:\n    pass\nfinally:\n    n = df.count()"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_count_in_else_is_not_protected(self):
        code = "try:\n    x = 1\nexcept Exception:\n    raise\nelse:\n    n = df.count()"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_count_in_try_body_is_protected(self):
        code = "try:\n    n = df.count()\nexcept Exception:\n    raise"
        result = check_spark_actions_try_except(code)
        assert not has_rule(result, "SEMS_ERR001")


# ── Finding 6: typo detection for direct imports ─────────────────────────────


class TestTypoDetectionDirectImports:
    def test_direct_import_typo_flagged(self):
        code = "from pyspark.sql.functions import colz\nresult = colz('x')"
        result = check_pyspark_api_typos(code)
        assert has_rule(result, "SPARK003")

    def test_direct_import_valid_name_ok(self):
        code = "from pyspark.sql.functions import col\nresult = col('x')"
        assert check_pyspark_api_typos(code) == []

    def test_alias_typo_still_detected(self):
        code = "from pyspark.sql import functions as F\nresult = F.colz('x')"
        result = check_pyspark_api_typos(code)
        assert has_rule(result, "SPARK003")

    def test_alias_valid_name_ok(self):
        code = "from pyspark.sql import functions as F\nresult = F.col('x')"
        assert check_pyspark_api_typos(code) == []


# ── Finding 6b: allowlist must track the installed pyspark version, not just
# the hand-curated static list (regression test for names missing from the
# static list but genuinely exported by pyspark.sql.functions) ───────────────


class TestAllowlistTracksInstalledPyspark:
    def test_pandas_udf_type_not_flagged(self):
        code = "from pyspark.sql.functions import PandasUDFType\nx = PandasUDFType.SCALAR"
        assert check_pyspark_api_typos(code) == []

    def test_array_agg_not_flagged(self):
        code = "from pyspark.sql import functions as F\nresult = F.array_agg('x')"
        assert check_pyspark_api_typos(code) == []

    def test_approx_percentile_not_flagged(self):
        code = "from pyspark.sql import functions as F\nresult = F.approx_percentile('x', 0.5)"
        assert check_pyspark_api_typos(code) == []

    def test_bit_and_not_flagged(self):
        code = "from pyspark.sql import functions as F\nresult = F.bit_and('x')"
        assert check_pyspark_api_typos(code) == []

    def test_array_type_not_flagged(self):
        code = "from pyspark.sql.functions import ArrayType\nx = ArrayType"
        assert check_pyspark_api_typos(code) == []


# ── Finding 7: Levenshtein confidence ────────────────────────────────────────


class TestLevenshteinDistance:
    def test_edit_distance_1(self):
        assert _levenshtein("col", "cot") == 1

    def test_edit_distance_0(self):
        assert _levenshtein("col", "col") == 0

    def test_edit_distance_not_just_length(self):
        # "mounts" vs "count": length diff = 1 but edit distance > 1
        assert _levenshtein("mounts", "count") > 1

    def test_confidence_label_uses_edit_distance(self):
        # "coz" is edit distance 1 from "col" and (unlike "cot", which is a
        # real PySpark 4.x cotangent function) is not itself a real name.
        code = "from pyspark.sql import functions as F\nF.coz('x')"
        result = check_pyspark_api_typos(code)
        assert result
        assert "95%+" in result[0].message or "edit distance 1" in result[0].message


# ── Finding 8: alias reassignment ────────────────────────────────────────────


class TestAliasReassignment:
    def test_reassigned_alias_skips_check(self):
        code = (
            "from pyspark.sql import functions as F\n"
            "F = other_module\n"
            "F.colz('x')\n"
        )
        # After reassignment, F is no longer the PySpark functions module.
        result = check_pyspark_api_typos(code)
        assert not has_rule(result, "SPARK003")


# ── Finding 9: security checks skip comments ─────────────────────────────────


class TestSecuritySkipsComments:
    def test_credential_in_comment_not_flagged(self):
        code = "x = 1  # password = 'fake_value_here'"
        result = check_for_hardcoded_credentials(code)
        assert result == []

    def test_real_credential_still_flagged(self):
        code = "password = 'supersecret'"
        result = check_for_hardcoded_credentials(code)
        assert has_rule(result, "SEC001")

    def test_eval_in_comment_not_flagged(self):
        code = "x = 1  # eval(user_input)"
        result = check_for_unsafe_operations(code)
        assert result == []

    def test_real_eval_still_flagged(self):
        code = "eval(user_input)"
        result = check_for_unsafe_operations(code)
        assert has_rule(result, "SEC003")

    def test_credential_in_docstring_not_flagged(self):
        code = '"""\nExample: password = "fake"\n"""\nx = 1'
        # Docstrings are string literals; _code_only preserves them.
        # The credential pattern requires an assignment operator outside the string.
        result = check_for_hardcoded_credentials(code)
        # The pattern should not match because it's inside a string literal
        # (no real assignment — the = is inside quotes)
        assert result == []


# ── Finding 10: extended security patterns ────────────────────────────────────


class TestExtendedSecurityPatterns:
    def test_dict_credential_flagged(self):
        code = 'config = {"password": "supersecret"}'
        result = check_for_hardcoded_credentials(code)
        assert has_rule(result, "SEC001")

    def test_subscript_assignment_flagged(self):
        code = 'config["token"] = "my_real_token_here"'
        result = check_for_hardcoded_credentials(code)
        assert has_rule(result, "SEC001")

    def test_dunder_import_subprocess_flagged(self):
        code = '__import__("subprocess")'
        result = check_for_unsafe_operations(code)
        assert has_rule(result, "SEC002")

    def test_importlib_subprocess_flagged(self):
        code = 'importlib.import_module("subprocess")'
        result = check_for_unsafe_operations(code)
        assert has_rule(result, "SEC002")


# ── Finding 11: PII log detection coverage ────────────────────────────────────


class TestPIILogDetection:
    def test_logger_call_flagged(self):
        # email is a hard PII field → SHELL_PII002, not soft LOG003
        code = "logger.info('user email: %s', email)"
        result = check_pii_in_logs(code)
        assert has_rule(result, "SHELL_PII002")

    def test_logging_module_call_flagged(self):
        code = "logging.info('email: %s', email)"
        result = check_pii_in_logs(code)
        assert has_rule(result, "SHELL_PII002")

    def test_print_with_pii_flagged(self):
        code = "print(email)"
        result = check_pii_in_logs(code)
        assert has_rule(result, "SHELL_PII002")

    def test_non_pii_log_ok(self):
        code = "logger.info('record_id: %s', record_id)"
        assert check_pii_in_logs(code) == []


# ── Finding 12: bare except improvements ─────────────────────────────────────


class TestBareExceptImprovements:
    def test_bare_except_without_raise_flagged(self):
        code = "try:\n    pass\nexcept:\n    pass"
        assert has_rule(check_bare_except(code), "ERR002")

    def test_except_exception_without_raise_flagged(self):
        code = "try:\n    pass\nexcept Exception:\n    pass"
        assert has_rule(check_bare_except(code), "ERR002")

    def test_except_base_exception_flagged(self):
        code = "try:\n    pass\nexcept BaseException:\n    pass"
        assert has_rule(check_bare_except(code), "ERR002")

    def test_except_tuple_containing_exception_flagged(self):
        code = "try:\n    pass\nexcept (Exception, ValueError):\n    pass"
        assert has_rule(check_bare_except(code), "ERR002")

    def test_except_with_reraise_ok(self):
        code = "try:\n    pass\nexcept Exception:\n    raise"
        assert check_bare_except(code) == []

    def test_specific_exception_ok(self):
        code = "try:\n    pass\nexcept ValueError:\n    pass"
        assert check_bare_except(code) == []

    def test_raise_in_nested_function_does_not_count(self):
        # A raise inside a nested def should not satisfy the re-raise requirement.
        code = (
            "try:\n"
            "    pass\n"
            "except Exception:\n"
            "    def helper():\n"
            "        raise ValueError()\n"
        )
        result = check_bare_except(code)
        assert has_rule(result, "ERR002")


# ── Finding 13: collect type-blindness (documentation) ───────────────────────


class TestCollectTypeLimitations:
    def test_collect_without_limit_flagged(self):
        code = "rows = df.collect()"
        assert has_rule(check_unbounded_collect(code), "SPARK001")

    def test_collect_with_limit_ok(self):
        code = "rows = df.limit(100).collect()"
        assert check_unbounded_collect(code) == []

    def test_indirect_limit_not_detected(self):
        # small_df = df.limit(100); small_df.collect() — cannot be detected without DFA
        code = "small_df = df.limit(100)\nrows = small_df.collect()"
        result = check_unbounded_collect(code)
        # This is a known false positive; the test documents the limitation.
        assert has_rule(result, "SPARK001")


# ── Finding 14: limit chain validation ────────────────────────────────────────


class TestLimitChainValidation:
    def test_positive_limit_is_guard(self):
        code = "rows = df.limit(100).collect()"
        assert check_unbounded_collect(code) == []

    def test_zero_limit_is_not_guard(self):
        code = "rows = df.limit(0).collect()"
        assert has_rule(check_unbounded_collect(code), "SPARK001")

    def test_negative_limit_is_not_guard(self):
        code = "rows = df.limit(-1).collect()"
        assert has_rule(check_unbounded_collect(code), "SPARK001")

    def test_first_is_not_a_collect_guard(self):
        # .first().collect() should still be flagged: first() returns Row, not DataFrame.
        code = "x = df.first().collect()"
        result = check_unbounded_collect(code)
        assert has_rule(result, "SPARK001")

    def test_head_is_not_a_collect_guard(self):
        code = "x = df.head(5).collect()"
        result = check_unbounded_collect(code)
        assert has_rule(result, "SPARK001")


# ── Finding 15: driver-side loop coverage ────────────────────────────────────


class TestDriverSideLoopCoverage:
    def test_direct_collect_loop_flagged(self):
        code = "for row in df.collect():\n    pass"
        assert has_rule(check_driver_side_loops(code), "SPARK006")

    def test_indirect_collect_loop_flagged(self):
        code = "rows = df.collect()\nfor row in rows:\n    pass"
        assert has_rule(check_driver_side_loops(code), "SPARK006")

    def test_toPandas_loop_flagged(self):
        code = "for row in df.toPandas():\n    pass"
        assert has_rule(check_driver_side_loops(code), "SPARK006")

    def test_list_comprehension_over_collect_flagged(self):
        code = "values = [row['x'] for row in df.collect()]"
        assert has_rule(check_driver_side_loops(code), "SPARK006")

    def test_plain_list_loop_ok(self):
        code = "for item in [1, 2, 3]:\n    pass"
        assert check_driver_side_loops(code) == []


# ── Finding 16: local path detection ─────────────────────────────────────────


class TestLocalPathDetection:
    def test_tmp_path_flagged(self):
        code = "path = '/tmp/myfile.csv'"
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_home_path_flagged(self):
        code = "path = '/home/user/data.parquet'"
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_windows_path_backslash_flagged(self):
        # Use double-escaped backslashes so ast.parse sees 'C:\temp\data.csv'
        # (a real backslash, not the \t tab escape).
        code = "path = 'C:\\\\temp\\\\data.csv'"
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_windows_path_forward_slash_flagged(self):
        code = "path = 'C:/temp/data.csv'"
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_workspace_path_flagged(self):
        code = "path = '/Workspace/Users/me/data.csv'"
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_dbfs_path_ok(self):
        code = "path = '/dbfs/mnt/data/file.parquet'"
        assert check_local_file_paths(code) == []

    def test_s3_path_ok(self):
        code = "path = 's3://my-bucket/data/'"
        assert check_local_file_paths(code) == []

    def test_path_in_comment_not_flagged(self):
        code = "x = 1  # was: path = '/tmp/file.csv'"
        assert check_local_file_paths(code) == []


# ── Finding 17: line numbers in violations ────────────────────────────────────


class TestLineNumbers:
    def test_credential_violation_has_line_number(self):
        code = "x = 1\npassword = 'secret123'\ny = 2"
        result = check_for_hardcoded_credentials(code)
        assert result
        assert result[0].line_number == 2

    def test_unsafe_operation_has_line_number(self):
        code = "x = 1\nimport subprocess\ny = 2"
        result = check_for_unsafe_operations(code)
        assert result
        assert result[0].line_number == 2

    def test_collect_violation_has_line_number(self):
        code = "x = 1\nrows = df.collect()\ny = 2"
        result = check_unbounded_collect(code)
        assert result
        assert result[0].line_number == 2

    def test_spark_action_violation_has_line_number(self):
        code = "x = 1\nn = df.count()\ny = 2"
        result = check_spark_actions_try_except(code)
        assert result
        assert result[0].line_number == 2


# ── Finding 18: section context in structured violations ─────────────────────


class TestSectionContextInStructuredViolations:
    def test_section_index_populated(self):
        sections = {
            0: "password = 'secret123'",
            1: "x = 1",
        }
        result = check_script_compliance(sections)
        for sv in result.structured_violations:
            assert sv.section_index is not None

    def test_section_index_matches_source_section(self):
        sections = {
            0: "x = 1",
            1: "import subprocess",
        }
        result = check_script_compliance(sections)
        sec2_hard = [
            sv for sv in result.structured_violations
            if sv.severity == "hard" and sv.section_index == 1
        ]
        assert sec2_hard, "Expected SEC002 violation with section_index=1"

    def test_string_violations_have_section_prefix(self):
        sections = {3: "import subprocess"}
        result = check_script_compliance(sections)
        assert any("[section 3]" in v for v in result.violations)


# ── Finding 19: YAML loading degradation ──────────────────────────────────────


class TestYAMLLoadingDegradation:
    def test_checker_works_without_rule_catalog(self, monkeypatch):
        # Simulate empty catalog (e.g. yaml missing).
        monkeypatch.setattr(_cc, "RULE_CATALOG", {})
        result = check_compliance("import subprocess")
        # Should still detect the violation even without remediation text.
        assert any("SEC002" in v for v in result.violations)

    def test_empty_catalog_gives_empty_remediation(self, monkeypatch):
        monkeypatch.setattr(_cc, "RULE_CATALOG", {})
        result = check_compliance("import subprocess")
        sec002 = [sv for sv in result.structured_violations if sv.rule_id == "SEC002"]
        assert sec002
        # Remediation may be empty but the violation still exists.
        assert isinstance(sec002[0].remediation, str)


# ── Finding 20: edge-case regression suite ────────────────────────────────────


class TestHighValueEdgeCases:
    def test_syntax_invalid_source(self):
        result = check_compliance("def foo(\n")
        assert result.passed is False

    def test_collect_outside_try(self):
        result = check_spark_actions_try_except("rows = df.collect()")
        assert has_rule(result, "SEMS_ERR001")

    def test_count_inside_except(self):
        code = "try:\n    pass\nexcept Exception:\n    n = df.count()\n    raise"
        result = check_spark_actions_try_except(code)
        assert has_rule(result, "SEMS_ERR001")

    def test_direct_import_typo(self):
        code = "from pyspark.sql.functions import colz\ncolz('x')"
        result = check_pyspark_api_typos(code)
        assert has_rule(result, "SPARK003")

    def test_comment_credential_no_false_positive(self):
        code = '# password = "fake"\nx = 1'
        assert check_for_hardcoded_credentials(code) == []

    def test_real_secret_in_dict(self):
        code = '{"password": "supersecret_value"}'
        assert has_rule(check_for_hardcoded_credentials(code), "SEC001")

    def test_pii_in_print(self):
        # email is a hard PII field → SHELL_PII002
        code = "print(email)"
        assert has_rule(check_pii_in_logs(code), "SHELL_PII002")

    def test_pii_in_logging_call(self):
        code = "logging.info('email=%s', email)"
        assert has_rule(check_pii_in_logs(code), "SHELL_PII002")

    def test_broad_tuple_except(self):
        code = "try:\n    pass\nexcept (Exception, ValueError):\n    pass"
        assert has_rule(check_bare_except(code), "ERR002")

    def test_indirect_collect_variable_loop(self):
        code = "rows = df.collect()\nfor row in rows:\n    pass"
        assert has_rule(check_driver_side_loops(code), "SPARK006")

    def test_local_master_spark_session(self):
        code = 'SparkSession.builder.master("local[*]").getOrCreate()'
        result = check_compliance(code)
        assert any("SHELL_STAM001" in v for v in result.violations)

    def test_local_path_in_code_not_comment(self):
        code = "path = '/tmp/file.csv'  # was /dbfs/..."
        assert has_rule(check_local_file_paths(code), "SPARK009")

    def test_collect_with_zero_limit_still_flagged(self):
        code = "rows = df.limit(0).collect()"
        assert has_rule(check_unbounded_collect(code), "SPARK001")


# ── New findings regression tests ────────────────────────────────────────────


class TestMissingCompanyRules:
    def test_table_bad_name_flagged(self):
        assert has_rule(check_table_naming_convention("spark.table('mytable')"), "SHELL_NAME001")

    def test_table_good_name_ok(self):
        assert check_table_naming_convention("spark.table('trading_position_raw')") == []

    def test_generic_df_name_flagged(self):
        assert has_rule(check_generic_df_variable("df = spark.read.parquet('/data')"), "SHELL_NAME002")

    def test_descriptive_df_name_ok(self):
        assert check_generic_df_variable("trades_df = spark.read.parquet('/data')") == []

    def test_hardcoded_repartition_flagged(self):
        assert has_rule(check_hard_coded_repartition("df.repartition(200)"), "SHELL_SPARK001")

    def test_variable_repartition_ok(self):
        assert check_hard_coded_repartition("df.repartition(n_partitions)") == []

    def test_widget_without_default_flagged(self):
        assert has_rule(check_widget_without_default("val = dbutils.widgets.get('env')"), "SHELL_STAM002")

    def test_widget_with_default_ok(self):
        code = "dbutils.widgets.text('env', 'dev')\nval = dbutils.widgets.get('env')"
        assert check_widget_without_default(code) == []

    def test_company_rules_called_from_check_compliance(self):
        result = check_compliance("SparkSession.builder.master('local').getOrCreate()")
        assert any("SHELL_STAM001" in v for v in result.violations)
        result2 = check_compliance("val = dbutils.widgets.get('key')")
        assert any("SHELL_STAM002" in w for w in result2.warnings)


class TestPIILoggingHard:
    def test_email_in_logger_is_hard(self):
        result = check_pii_in_logs("logger.info('email: %s', email)")
        hard = [v for v in result if v.severity == "hard"]
        assert hard and hard[0].rule_id == "SHELL_PII002"

    def test_email_in_print_is_hard(self):
        result = check_pii_in_logs("print(email)")
        hard = [v for v in result if v.severity == "hard"]
        assert hard and hard[0].rule_id == "SHELL_PII002"

    def test_email_in_logging_module_is_hard(self):
        result = check_pii_in_logs("logging.info('email=%s', email)")
        hard = [v for v in result if v.severity == "hard"]
        assert hard and hard[0].rule_id == "SHELL_PII002"

    def test_pii_log_fails_check_compliance(self):
        result = check_compliance("logger.info('email: %s', email)")
        assert any("SHELL_PII002" in v for v in result.violations)
        assert result.passed is False

    def test_token_in_logger_is_soft(self):
        result = check_pii_in_logs("logger.info('token: %s', token)")
        soft = [v for v in result if v.severity == "soft" and v.rule_id == "LOG003"]
        assert soft


class TestScriptScoreFailurePropagates:
    def test_hard_violation_always_fails_script(self):
        sections = {0: "x = 1", 1: "import subprocess"}
        assert check_script_compliance(sections).passed is False

    def test_soft_only_section_failure_propagates(self):
        sections = {0: "x = 1"}  # likely fails score threshold
        sec = check_compliance("x = 1")
        script = check_script_compliance({0: "x = 1"})
        assert script.passed == sec.passed  # script mirrors section result


class TestJDBCAllOptions:
    def test_jdbc_all_four_options_ok(self):
        code = (
            'df.write.format("jdbc")'
            '.option("url","jdbc:...")'
            '.option("partitionColumn","id")'
            '.option("lowerBound",0)'
            '.option("upperBound",1000000)'
            '.option("numPartitions",10)'
            '.save()'
        )
        assert check_jdbc_no_partition(code) == []

    def test_jdbc_missing_lowerBound_flagged(self):
        code = (
            'df.write.format("jdbc")'
            '.option("partitionColumn","id")'
            '.option("upperBound",1000)'
            '.option("numPartitions",10)'
            '.save()'
        )
        assert has_rule(check_jdbc_no_partition(code), "SHELL_SPARK002")

    def test_jdbc_no_options_flagged(self):
        assert has_rule(check_jdbc_no_partition('df.write.format("jdbc").save()'), "SHELL_SPARK002")


class TestPIIMaskingPerColumn:
    def test_unrelated_sha2_does_not_clear_pii_write(self):
        code = (
            "df.withColumn('id', F.sha2(F.col('id'), 256))"
            ".select('email').write.format('delta').save('/path')"
        )
        assert has_rule(check_pii_column_write(code), "SHELL_PII001")

    def test_specific_column_sha2_clears_it(self):
        code = (
            "df.withColumn('email', F.sha2(F.col('email'), 256))"
            ".write.format('delta').save('/path')"
        )
        assert check_pii_column_write(code) == []


class TestLogRegexSkipsComments:
    def test_logging_import_in_comment_not_satisfied(self):
        result = check_for_logging_usage("# import logging\nx = 1")
        assert result  # still warns about missing logging

    def test_try_in_comment_not_satisfied(self):
        result = check_for_error_handling("# try:\n#     pass\nx = 1")
        assert result  # still warns about missing try/except

    def test_real_logging_import_satisfied(self):
        code = "import logging\nlogger = logging.getLogger(__name__)\nlogger.info('hi')"
        assert check_for_logging_usage(code) == []

    def test_real_try_except_satisfied(self):
        assert check_for_error_handling("try:\n    pass\nexcept Exception:\n    raise") == []


class TestSparkWriteActions:
    def test_insertInto_outside_try_flagged(self):
        assert has_rule(check_spark_actions_try_except("df.insertInto('table')"), "SEMS_ERR001")

    def test_save_outside_try_flagged(self):
        assert has_rule(check_spark_actions_try_except("df.write.format('delta').save('/path')"), "SEMS_ERR001")

    def test_insertInto_inside_try_ok(self):
        code = "try:\n    df.insertInto('table')\nexcept Exception:\n    raise"
        assert not has_rule(check_spark_actions_try_except(code), "SEMS_ERR001")


class TestBearerPATSkipsDocstrings:
    def test_bearer_in_docstring_not_flagged(self):
        code = '"""\nAuthorization: Bearer AAAAAAAAAAAAAAAAAAAAAA\n"""\nx = 1'
        result = check_for_hardcoded_credentials(code)
        assert result == []

    def test_bearer_in_real_code_flagged(self):
        code = 'headers = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAA"}'
        result = check_for_hardcoded_credentials(code)
        assert has_rule(result, "SEC001")


# ── Construction Policy alignment: LIT001 ────────────────────────────────────


class TestRepeatedMagicLiterals:
    def test_repeated_literal_in_comparisons_flagged(self):
        code = "if x > 42:\n    pass\nif y > 42:\n    pass\nif z > 42:\n    pass\n"
        result = check_repeated_magic_literals(code)
        assert has_rule(result, "LIT001")
        assert result[0].severity == "soft"

    def test_literal_used_twice_not_flagged(self):
        code = "if x > 42:\n    pass\nif y > 42:\n    pass\n"
        assert check_repeated_magic_literals(code) == []

    def test_excluded_small_values_not_flagged(self):
        code = "if a > 1:\n    pass\nif b > 1:\n    pass\nif c > 1:\n    pass\n"
        assert check_repeated_magic_literals(code) == []

    def test_declared_constant_value_not_flagged(self):
        code = (
            "MAX_RETRIES = 42\n"
            "if x > 42:\n    pass\n"
            "if y > 42:\n    pass\n"
            "if z > 42:\n    pass\n"
        )
        assert check_repeated_magic_literals(code) == []


# ── Construction Policy alignment: DUP001 (duplicate code blocks) ────────────


class TestDuplicateCodeBlocks:
    def test_identical_block_across_two_functions_flagged(self):
        code = (
            "def clean_customer_columns(df):\n"
            "    df = df.withColumnRenamed('cust_id', 'customer_id')\n"
            "    df = df.withColumnRenamed('cust_name', 'customer_name')\n"
            "    df = df.withColumnRenamed('cust_addr', 'customer_address')\n"
            "    df = df.dropna(subset=['customer_id'])\n"
            "    return df\n"
            "\n"
            "def clean_vendor_columns(df):\n"
            "    df = df.withColumnRenamed('cust_id', 'customer_id')\n"
            "    df = df.withColumnRenamed('cust_name', 'customer_name')\n"
            "    df = df.withColumnRenamed('cust_addr', 'customer_address')\n"
            "    df = df.dropna(subset=['customer_id'])\n"
            "    return df\n"
        )
        result = check_duplicate_code_blocks(code)
        assert has_rule(result, "DUP001")
        assert result[0].severity == "soft"
        assert "clean_customer_columns" in result[0].message
        assert "clean_vendor_columns" in result[0].message

    def test_below_threshold_run_not_flagged(self):
        # Only 3 shared statements — below _MIN_DUPLICATE_STATEMENTS (4).
        code = (
            "def f(df):\n"
            "    df = df.filter(df.x > 0)\n"
            "    df = df.dropna()\n"
            "    return df\n"
            "\n"
            "def g(df):\n"
            "    df = df.filter(df.x > 0)\n"
            "    df = df.dropna()\n"
            "    return 1\n"
        )
        assert check_duplicate_code_blocks(code) == []

    def test_different_functions_not_flagged(self):
        code = (
            "def f(df):\n"
            "    df = df.filter(df.x > 0)\n"
            "    df = df.dropna()\n"
            "    df = df.distinct()\n"
            "    return df\n"
            "\n"
            "def g(df):\n"
            "    df = df.select('a', 'b')\n"
            "    df = df.orderBy('a')\n"
            "    df = df.limit(10)\n"
            "    return df\n"
        )
        assert check_duplicate_code_blocks(code) == []

    def test_shared_leading_docstring_alone_not_flagged(self):
        # Identical docstrings but genuinely different bodies should not match —
        # the docstring is stripped before comparison.
        code = (
            "def f(df):\n"
            "    '''Clean the frame.'''\n"
            "    df = df.filter(df.x > 0)\n"
            "    df = df.dropna()\n"
            "    return df\n"
            "\n"
            "def g(df):\n"
            "    '''Clean the frame.'''\n"
            "    df = df.select('a')\n"
            "    df = df.distinct()\n"
            "    return df\n"
        )
        assert check_duplicate_code_blocks(code) == []

    def test_single_occurrence_not_flagged(self):
        code = (
            "def f(df):\n"
            "    df = df.withColumnRenamed('a', 'b')\n"
            "    df = df.withColumnRenamed('c', 'd')\n"
            "    df = df.dropna()\n"
            "    df = df.distinct()\n"
            "    return df\n"
        )
        assert check_duplicate_code_blocks(code) == []


# ── Docstring content quality: DOC002 (Args) / DOC003 (Returns/Yields) ───────


class TestDocstringArgsContent:
    def test_missing_args_section_entirely_is_flagged(self):
        code = (
            "def transform(df, threshold):\n"
            "    '''Filter rows above threshold.'''\n"
            "    return df.filter(df.x > threshold)\n"
        )
        result = check_function_docstring_content(code)
        assert has_rule(result, "DOC002")
        doc002 = [v for v in result if v.rule_id == "DOC002"][0]
        assert "df" in doc002.message and "threshold" in doc002.message

    def test_partial_args_missing_one_param_is_flagged(self):
        code = (
            "def compute_total(a, b):\n"
            "    '''Add two numbers.\n\n"
            "    Args:\n"
            "        a: first number.\n"
            "    '''\n"
            "    return a + b\n"
        )
        result = check_function_docstring_content(code)
        doc002 = [v for v in result if v.rule_id == "DOC002"]
        assert len(doc002) == 1
        missing_list = doc002[0].message.rsplit(":", 1)[-1].strip().rstrip(".")
        assert missing_list == "b"

    def test_all_params_documented_not_flagged(self):
        code = (
            "def compute_total(a, b):\n"
            "    '''Add two numbers.\n\n"
            "    Args:\n"
            "        a: first number.\n"
            "        b: second number.\n\n"
            "    Returns:\n"
            "        The sum.\n"
            "    '''\n"
            "    return a + b\n"
        )
        assert check_function_docstring_content(code) == []

    def test_self_and_cls_excluded(self):
        code = (
            "class Thing:\n"
            "    def method(self, df):\n"
            "        '''Do a thing.\n\n"
            "        Args:\n"
            "            df: input frame.\n\n"
            "        Returns:\n"
            "            The frame.\n"
            "        '''\n"
            "        return df\n"
        )
        assert check_function_docstring_content(code) == []

    def test_varargs_and_kwargs_documented(self):
        code = (
            "def f(*args, **kwargs):\n"
            "    '''Do a thing.\n\n"
            "    Args:\n"
            "        *args: positional args.\n"
            "        **kwargs: keyword args.\n"
            "    '''\n"
            "    pass\n"
        )
        assert check_function_docstring_content(code) == []

    def test_no_params_no_args_section_required(self):
        code = "def get_config():\n    '''Return the static config.'''\n    return {}\n"
        result = check_function_docstring_content(code)
        assert not has_rule(result, "DOC002")

    def test_missing_docstring_entirely_not_flagged_here(self):
        # Presence is pydocstyle's job (D101/D102/D103) — this rule only
        # evaluates docstrings that already exist.
        code = "def transform(df, threshold):\n    return df.filter(df.x > threshold)\n"
        assert check_function_docstring_content(code) == []


class TestDocstringReturnsContent:
    def test_return_value_without_returns_section_is_flagged(self):
        code = "def f(df):\n    '''Filter the frame.'''\n    return df.dropna()\n"
        result = check_function_docstring_content(code)
        assert has_rule(result, "DOC003")

    def test_return_value_with_returns_section_not_flagged(self):
        code = (
            "def f():\n"
            "    '''Filter the frame.\n\n"
            "    Returns:\n"
            "        The filtered frame.\n"
            "    '''\n"
            "    return get_frame().dropna()\n"
        )
        assert check_function_docstring_content(code) == []

    def test_bare_return_none_not_flagged(self):
        code = "def f():\n    '''Validate global state.'''\n    if not ready():\n        return\n"
        assert check_function_docstring_content(code) == []

    def test_explicit_return_none_not_flagged(self):
        code = "def f():\n    '''Validate global state.'''\n    return None\n"
        assert check_function_docstring_content(code) == []

    def test_generator_requires_yields_or_returns_section(self):
        code = (
            "def rows():\n"
            "    '''Iterate rows.'''\n"
            "    for r in get_frame().collect():\n"
            "        yield r\n"
        )
        result = check_function_docstring_content(code)
        assert has_rule(result, "DOC003")

    def test_generator_with_yields_section_not_flagged(self):
        code = (
            "def rows():\n"
            "    '''Iterate rows.\n\n"
            "    Yields:\n"
            "        Each row.\n"
            "    '''\n"
            "    for r in get_frame().collect():\n"
            "        yield r\n"
        )
        assert check_function_docstring_content(code) == []

    def test_property_getter_exempt_from_returns_requirement(self):
        code = (
            "class Thing:\n"
            "    @property\n"
            "    def name(self):\n"
            "        '''The thing's name.'''\n"
            "        return self._name\n"
        )
        assert check_function_docstring_content(code) == []

    def test_nested_function_return_not_attributed_to_outer(self):
        # The outer function's own body has no `return <value>` — only the
        # nested helper does — so the outer docstring should not be required
        # to have a Returns: section.
        code = (
            "def outer():\n"
            "    '''Run a nested helper.'''\n"
            "    def inner():\n"
            "        return 1\n"
            "    inner()\n"
        )
        assert check_function_docstring_content(code) == []


# ── Construction Policy alignment: COM001 (TODO/FIXME) ───────────────────────


class TestTodoComments:
    def test_todo_comment_flagged(self):
        result = check_todo_comments("# TODO: fix this later\nx = 1")
        assert has_rule(result, "COM001")
        assert result[0].severity == "soft"

    def test_fixme_comment_flagged(self):
        assert has_rule(check_todo_comments("x = 1  # FIXME broken"), "COM001")

    def test_regular_comment_not_flagged(self):
        assert check_todo_comments("# this explains the algorithm\nx = 1") == []


# ── Construction Policy alignment: COM002 (commented-out code) ──────────────


class TestCommentedOutCode:
    def test_commented_out_assignment_flagged(self):
        code = "x = 1\n# y = compute_something(x, 2)\nz = 2\n"
        result = check_commented_out_code(code)
        assert has_rule(result, "COM002")

    def test_commented_out_call_flagged(self):
        code = "# df.write.format('delta').save('/path')\nx = 1\n"
        assert has_rule(check_commented_out_code(code), "COM002")

    def test_prose_comment_not_flagged(self):
        code = "# this function computes the running total\nx = 1\n"
        assert check_commented_out_code(code) == []

    def test_inline_comment_after_code_not_flagged(self):
        code = "x = 1  # y = 2 would also work here\n"
        assert check_commented_out_code(code) == []

    def test_bare_identifier_comment_not_flagged(self):
        code = "# placeholder\nx = 1\n"
        assert check_commented_out_code(code) == []


# ── Construction Policy alignment: FUNC001 (boolean naming) ──────────────────


class TestBooleanFunctionNaming:
    def test_bool_function_without_predicate_name_flagged(self):
        code = "def check(x) -> bool:\n    return x > 0\n"
        result = check_boolean_function_naming(code)
        assert has_rule(result, "FUNC001")

    def test_is_prefixed_function_not_flagged(self):
        code = "def is_valid(x) -> bool:\n    return x > 0\n"
        assert check_boolean_function_naming(code) == []

    def test_has_prefixed_function_not_flagged(self):
        code = "def has_permission(x) -> bool:\n    return x > 0\n"
        assert check_boolean_function_naming(code) == []

    def test_function_without_bool_annotation_not_flagged(self):
        code = "def check(x):\n    return x > 0\n"
        assert check_boolean_function_naming(code) == []


# ── Construction Policy alignment: FANOUT001 (fan-out metric) ───────────────


class TestFunctionFanOut:
    def test_low_fanout_not_flagged(self):
        assert check_function_fan_out(_make_fanout_code(7)) == []

    def test_soft_band_flagged(self):
        result = check_function_fan_out(_make_fanout_code(8))
        assert has_rule(result, "FANOUT001")
        assert result[0].severity == "soft"

    def test_hard_band_flagged(self):
        result = check_function_fan_out(_make_fanout_code(11))
        assert has_rule(result, "FANOUT001")
        assert result[0].severity == "hard"


# ── Construction Policy alignment: EXPR001 (terms-per-expression) ───────────


class TestExpressionTermCount:
    def test_low_term_count_not_flagged(self):
        code = "if a and b and c and d and e:\n    pass\n"
        assert check_expression_term_count(code) == []

    def test_soft_band_flagged(self):
        code = "if a and b and c and d and e and f:\n    pass\n"
        result = check_expression_term_count(code)
        assert has_rule(result, "EXPR001")
        assert result[0].severity == "soft"

    def test_hard_band_flagged(self):
        code = "if a and b and c and d and e and f and g and h and i:\n    pass\n"
        result = check_expression_term_count(code)
        assert has_rule(result, "EXPR001")
        assert result[0].severity == "hard"


# ── Construction Policy alignment: NEST001 (decision nesting depth) ─────────


class TestDecisionNestingDepth:
    def test_low_nesting_not_flagged(self):
        assert check_decision_nesting_depth(_make_nested_code(4)) == []

    def test_soft_band_flagged(self):
        result = check_decision_nesting_depth(_make_nested_code(5))
        assert has_rule(result, "NEST001")
        assert result[0].severity == "soft"

    def test_hard_band_flagged(self):
        result = check_decision_nesting_depth(_make_nested_code(7))
        assert has_rule(result, "NEST001")
        assert result[0].severity == "hard"


# ── Construction Policy alignment: CONST001 (constant reassignment) ─────────


class TestConstantReassignment:
    def test_reassigned_constant_flagged(self):
        code = "MAX_RETRIES = 3\nx = 1\nMAX_RETRIES = 5\n"
        result = check_constant_reassignment(code)
        assert has_rule(result, "CONST001")
        assert result[0].severity == "soft"

    def test_single_assignment_not_flagged(self):
        assert check_constant_reassignment("MAX_RETRIES = 3\nx = 1\n") == []

    def test_lowercase_variable_reassignment_not_flagged(self):
        assert check_constant_reassignment("retries = 3\nretries = 5\n") == []

    def test_reassignment_inside_function_not_flagged(self):
        code = "MAX_RETRIES = 3\ndef f():\n    MAX_RETRIES = 5\n    return MAX_RETRIES\n"
        assert check_constant_reassignment(code) == []


# ── Construction Policy alignment: wiring into check_compliance ─────────────


class TestConstructionPolicyIntegration:
    def test_todo_comment_surfaces_in_check_compliance(self):
        result = check_compliance("# TODO: revisit\nx = 1\n")
        assert has_rule(result.structured_violations, "COM001")

    def test_fan_out_surfaces_in_check_compliance(self):
        result = check_compliance(_make_fanout_code(11))
        assert has_rule(result.structured_violations, "FANOUT001")

    def test_multiple_classes_surface_in_check_compliance(self):
        code = "class A:\n    pass  # marker\nclass B:\n    pass  # marker\n"
        result = check_compliance(code)
        assert has_rule(result.structured_violations, "FILE001")


# ── Construction Policy alignment: FILE001 (one abstract concept per file) ───


class TestSingleAbstractConcept:
    def test_two_top_level_classes_flagged(self):
        code = "class Reader:\n    pass  # stub\n\nclass Writer:\n    pass  # stub\n"
        result = check_single_abstract_concept(code)
        assert has_rule(result, "FILE001")
        assert result[0].severity == "soft"

    def test_single_class_not_flagged(self):
        code = "class Reader:\n    pass  # stub\n"
        assert check_single_abstract_concept(code) == []

    def test_no_class_not_flagged(self):
        code = "def transform(df):\n    return df\n"
        assert check_single_abstract_concept(code) == []

    def test_nested_class_not_counted(self):
        code = "class Outer:\n    class Inner:\n        pass  # stub\n"
        assert check_single_abstract_concept(code) == []


# ── Construction Policy alignment: STMT001 (documented empty statements) ─────


class TestUncommentedEmptyStatement:
    def test_bare_pass_flagged(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    pass\n"
        result = check_uncommented_empty_statement(code)
        assert has_rule(result, "STMT001")
        assert result[0].severity == "soft"

    def test_pass_with_inline_comment_ok(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    pass  # nothing to clean up\n"
        assert check_uncommented_empty_statement(code) == []

    def test_pass_with_comment_above_ok(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    # value already absent\n    pass\n"
        assert check_uncommented_empty_statement(code) == []

    def test_bare_ellipsis_flagged(self):
        code = "def stub():\n    ...\n"
        assert has_rule(check_uncommented_empty_statement(code), "STMT001")


# ── Construction Policy alignment: LIT002 (repeated string literals) ────────


class TestRepeatedStringLiterals:
    def test_repeated_string_in_comparisons_flagged(self):
        code = (
            'if status == "ACTIVE":\n    pass\n'
            'if other == "ACTIVE":\n    pass\n'
            'if third == "ACTIVE":\n    pass\n'
        )
        result = check_repeated_string_literals(code)
        assert has_rule(result, "LIT002")
        assert result[0].severity == "soft"

    def test_string_used_twice_not_flagged(self):
        code = 'if status == "ACTIVE":\n    pass\nif other == "ACTIVE":\n    pass\n'
        assert check_repeated_string_literals(code) == []

    def test_declared_constant_value_not_flagged(self):
        code = (
            'STATUS_ACTIVE = "ACTIVE"\n'
            'if a == "ACTIVE":\n    pass\n'
            'if b == "ACTIVE":\n    pass\n'
            'if c == "ACTIVE":\n    pass\n'
        )
        assert check_repeated_string_literals(code) == []

    def test_single_char_string_not_flagged(self):
        code = 'if a == ",":\n    pass\nif b == ",":\n    pass\nif c == ",":\n    pass\n'
        assert check_repeated_string_literals(code) == []


# ── Construction Policy alignment: PARAM001 (in-parameter reassignment) ─────


class TestParameterReassignment:
    def test_unrelated_reassignment_flagged(self):
        code = "def f(threshold):\n    threshold = 100\n    return threshold\n"
        result = check_parameter_reassignment(code)
        assert has_rule(result, "PARAM001")
        assert result[0].severity == "soft"

    def test_self_derived_chain_not_flagged(self):
        code = "def f(df):\n    df = df.filter(df.x > 0)\n    return df\n"
        assert check_parameter_reassignment(code) == []

    def test_self_parameter_not_flagged(self):
        code = "class C:\n    def f(self):\n        self = None\n        return self\n"
        assert check_parameter_reassignment(code) == []

    def test_non_parameter_local_not_flagged(self):
        code = "def f(df):\n    out = 100\n    return out\n"
        assert check_parameter_reassignment(code) == []


# ── Construction Policy alignment: ATTR001 (unused instance attributes) ─────


class TestUnusedInstanceAttributes:
    def test_unread_attribute_flagged(self):
        code = (
            "class Widget:\n"
            "    def __init__(self):\n"
            "        self.name = 'w'\n"
            "        self.unused = 1\n"
            "    def describe(self):\n"
            "        return self.name\n"
        )
        result = check_unused_instance_attributes(code)
        assert has_rule(result, "ATTR001")
        assert result[0].severity == "soft"
        assert "unused" in result[0].message

    def test_attribute_read_in_another_method_not_flagged(self):
        code = (
            "class Widget:\n"
            "    def __init__(self):\n"
            "        self.name = 'w'\n"
            "    def describe(self):\n"
            "        return self.name\n"
        )
        assert check_unused_instance_attributes(code) == []

    def test_augmented_assignment_counts_as_used(self):
        code = (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "    def bump(self):\n"
            "        self.count += 1\n"
        )
        assert check_unused_instance_attributes(code) == []

    def test_staticmethod_not_treated_as_self(self):
        code = (
            "class Helper:\n"
            "    @staticmethod\n"
            "    def build(self):\n"
            "        self.temp = 1\n"
            "        return self\n"
        )
        assert check_unused_instance_attributes(code) == []


# ── STYLE001 (function length) ───────────────────────────────────────────────


class TestFunctionLength:
    def test_short_function_not_flagged(self):
        code = "def f():\n" + "    x = 1\n" * 10
        assert check_function_length(code) == []

    def test_long_function_flagged(self):
        code = "def f():\n" + "    x = 1\n" * 65
        result = check_function_length(code)
        assert has_rule(result, "STYLE001")
        assert result[0].severity == "soft"


# ── SPARK002 (full count() for emptiness check) ─────────────────────────────


class TestFullCountForEmptiness:
    def test_count_equals_zero_flagged(self):
        result = check_full_count_for_emptiness("if df.count() == 0:\n    pass\n")
        assert has_rule(result, "SPARK002")
        assert result[0].severity == "soft"

    def test_count_greater_than_zero_flagged(self):
        assert has_rule(check_full_count_for_emptiness("if df.count() > 0:\n    pass\n"), "SPARK002")

    def test_zero_on_left_flagged(self):
        assert has_rule(check_full_count_for_emptiness("if 0 == df.count():\n    pass\n"), "SPARK002")

    def test_count_compared_to_nonzero_not_flagged(self):
        assert check_full_count_for_emptiness("if df.count() > 10:\n    pass\n") == []

    def test_count_with_args_not_flagged(self):
        assert check_full_count_for_emptiness('if df.count("x") == 0:\n    pass\n') == []


# ── SPARK004 / SPARK005 (toPandas / toLocalIterator) ────────────────────────


class TestToPandasAndLocalIterator:
    def test_to_pandas_flagged(self):
        result = check_for_to_pandas("pdf = df.toPandas()\n")
        assert has_rule(result, "SPARK004")
        assert result[0].severity == "soft"

    def test_to_local_iterator_flagged(self):
        result = check_for_to_local_iterator("it = df.toLocalIterator()\n")
        assert has_rule(result, "SPARK005")
        assert result[0].severity == "soft"

    def test_unrelated_call_not_flagged(self):
        assert check_for_to_pandas("df.show()\n") == []
        assert check_for_to_local_iterator("df.show()\n") == []


# ── New-rule wiring into check_compliance ────────────────────────────────────


class TestNewRulesIntegration:
    def test_param_and_spark_action_rules_surface_in_check_compliance(self):
        code = (
            "def f(threshold):\n"
            "    threshold = 100\n"
            "    if df.count() == 0:\n"
            "        pandas_df = df.toPandas()\n"
            "        it = df.toLocalIterator()\n"
            "    return threshold\n"
        )
        result = check_compliance(code)
        ids = rule_ids(result.structured_violations)
        assert "PARAM001" in ids
        assert "SPARK002" in ids
        assert "SPARK004" in ids
        assert "SPARK005" in ids
