"""
Unit tests for pre_sonar_check.py — tool runners and the quality gate.

External tools are never executed: every runner test monkeypatches
pre_sonar_check._run to return a canned (returncode, stdout, stderr) triple.

Run with:  python -m pytest subagents/sems_agent/compliance/tests/ -v
"""

import json
from pathlib import Path

import pytest

# pre_sonar_check is injected into sys.modules by conftest.py to avoid
# triggering the sems_agent package __init__.py (which requires ADK deps).
import pre_sonar_check as psc

TARGET = Path("/fake/converted_spark.py")


def fake_run(monkeypatch, returncode, stdout, stderr=""):
    monkeypatch.setattr(psc, "_run", lambda cmd: (returncode, stdout, stderr))


def make_issue(**overrides):
    defaults = dict(
        file=str(TARGET),
        line=1,
        col=0,
        tool="pylint",
        rule_id="X0001",
        sonar_type=psc.SMELL,
        sonar_severity=psc.MINOR,
        message="msg",
    )
    defaults.update(overrides)
    return psc.Issue(**defaults)


def report_with(issues):
    run = psc.ToolRun(name="fake", passed=not issues, issues=issues)
    return psc.ValidationReport(target=str(TARGET), tool_runs=[run])


# ── Quality gate ──────────────────────────────────────────────────────────────


class TestQualityGate:
    def test_clean_report_passes(self):
        assert report_with([]).quality_gate_passed() is True

    def test_minor_smell_does_not_block(self):
        report = report_with([make_issue()])
        assert report.quality_gate_passed() is True

    def test_bug_blocks(self):
        report = report_with([make_issue(sonar_type=psc.BUG, sonar_severity=psc.MAJOR)])
        assert report.quality_gate_passed() is False

    def test_vulnerability_blocks(self):
        report = report_with(
            [make_issue(sonar_type=psc.VULNERABILITY, sonar_severity=psc.CRITICAL)]
        )
        assert report.quality_gate_passed() is False

    def test_critical_smell_blocks(self):
        report = report_with([make_issue(sonar_severity=psc.CRITICAL)])
        assert report.quality_gate_passed() is False

    def test_tool_missing_blocks(self):
        # A mandatory analyzer that is not installed must fail the gate.
        run = psc._missing_tool_run("pylint", TARGET, "pylint")
        report = psc.ValidationReport(target=str(TARGET), tool_runs=[run])
        assert any(i.rule_id == "TOOL_MISSING" for i in report.all_issues)
        assert report.quality_gate_passed() is False

    def test_tool_error_blocks(self):
        # Regression: a crashed analyzer used to be reported as a clean pass.
        run = psc._tool_error_run("pylint", TARGET, "boom")
        report = psc.ValidationReport(target=str(TARGET), tool_runs=[run])
        assert run.passed is False
        assert any(i.rule_id == "TOOL_ERROR" for i in report.all_issues)
        assert report.quality_gate_passed() is False


# ── pylint runner ─────────────────────────────────────────────────────────────


def pylint_item(msg_type="warning", message_id="W0702", **overrides):
    item = {
        "type": msg_type,
        "path": str(TARGET),
        "line": 3,
        "column": 0,
        "message-id": message_id,
        "symbol": "sym",
        "message": "msg",
    }
    item.update(overrides)
    return item


class TestRunPylint:
    def test_clean_run_passes(self, monkeypatch):
        fake_run(monkeypatch, 0, "[]")
        run = psc.run_pylint(TARGET)
        assert run.passed is True
        assert run.issues == []

    def test_error_maps_to_critical_bug(self, monkeypatch):
        fake_run(monkeypatch, 2, json.dumps([pylint_item("error", "E0602")]))
        run = psc.run_pylint(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_type == psc.BUG
        assert run.issues[0].sonar_severity == psc.CRITICAL

    def test_warning_maps_to_major_smell(self, monkeypatch):
        fake_run(monkeypatch, 4, json.dumps([pylint_item("warning")]))
        run = psc.run_pylint(TARGET)
        assert run.issues[0].sonar_type == psc.SMELL
        assert run.issues[0].sonar_severity == psc.MAJOR

    def test_nonzero_exit_with_findings_is_not_tool_error(self, monkeypatch):
        # pylint exits non-zero whenever it reports anything — only a
        # non-zero exit WITHOUT output signals a crash.
        fake_run(monkeypatch, 20, json.dumps([pylint_item()]))
        run = psc.run_pylint(TARGET)
        assert all(i.rule_id != "TOOL_ERROR" for i in run.issues)

    def test_crash_without_output_is_tool_error(self, monkeypatch):
        # Regression: a crashed pylint used to be reported as a clean pass.
        fake_run(monkeypatch, 32, "", "usage error")
        run = psc.run_pylint(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_non_json_stdout_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 0, "Traceback (most recent call last): ...")
        run = psc.run_pylint(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"


# ── pylint-simplify runner ────────────────────────────────────────────────────


class TestRunPylintSimplify:
    def test_crash_without_output_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 32, "", "usage error")
        run = psc.run_pylint_simplify(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_non_json_stdout_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 0, "garbage")
        run = psc.run_pylint_simplify(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_high_confidence_refactor_is_major(self, monkeypatch):
        fake_run(monkeypatch, 8, json.dumps([pylint_item("refactor", "R1705")]))
        run = psc.run_pylint_simplify(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_severity == psc.MAJOR

    def test_clean_run_passes(self, monkeypatch):
        fake_run(monkeypatch, 0, "[]")
        run = psc.run_pylint_simplify(TARGET)
        assert run.passed is True


# ── bandit runner ─────────────────────────────────────────────────────────────


class TestRunBandit:
    def test_clean_run_passes(self, monkeypatch):
        fake_run(monkeypatch, 0, json.dumps({"results": []}))
        assert psc.run_bandit(TARGET).passed is True

    def test_high_severity_high_confidence_is_critical_vulnerability(self, monkeypatch):
        result = {
            "results": [
                {
                    "filename": str(TARGET),
                    "line_number": 5,
                    "test_id": "B602",
                    "test_name": "subprocess_popen_with_shell_equals_true",
                    "issue_text": "shell=True",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                }
            ]
        }
        fake_run(monkeypatch, 1, json.dumps(result))
        run = psc.run_bandit(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_type == psc.VULNERABILITY
        assert run.issues[0].sonar_severity == psc.CRITICAL

    def test_crash_without_output_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 2, "", "bandit internal error")
        run = psc.run_bandit(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_non_json_stdout_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 0, "not json at all")
        run = psc.run_bandit(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"


# ── radon runner ──────────────────────────────────────────────────────────────


class TestRunRadon:
    def test_low_complexity_passes(self, monkeypatch):
        blocks = [{"complexity": 3, "lineno": 1, "type": "function", "name": "f", "rank": "A"}]
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): blocks}))
        assert psc.run_radon(TARGET).passed is True

    def test_high_complexity_is_flagged(self, monkeypatch):
        cc = psc._MAX_CYCLOMATIC_COMPLEXITY + 10
        blocks = [{"complexity": cc, "lineno": 9, "type": "function", "name": "g", "rank": "F"}]
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): blocks}))
        run = psc.run_radon(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == f"cc-{cc}"

    def test_needs_justification_band_is_major(self, monkeypatch):
        # Construction Policy: CC 10-14 is "needs justification" -> MAJOR.
        cc = psc._CC_UNACCEPTABLE_MIN - 1
        blocks = [{"complexity": cc, "lineno": 9, "type": "function", "name": "g", "rank": "C"}]
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): blocks}))
        run = psc.run_radon(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_severity == psc.MAJOR

    def test_unacceptable_band_is_critical(self, monkeypatch):
        # Construction Policy: CC 15+ is "unacceptable" -> CRITICAL.
        cc = psc._CC_UNACCEPTABLE_MIN
        blocks = [{"complexity": cc, "lineno": 9, "type": "function", "name": "g", "rank": "D"}]
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): blocks}))
        run = psc.run_radon(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_severity == psc.CRITICAL

    def test_acceptable_band_not_flagged(self, monkeypatch):
        # Construction Policy: CC 1-9 is "acceptable" -> no issue at all.
        cc = psc._MAX_CYCLOMATIC_COMPLEXITY
        blocks = [{"complexity": cc, "lineno": 9, "type": "function", "name": "g", "rank": "A"}]
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): blocks}))
        run = psc.run_radon(TARGET)
        assert run.passed is True

    def test_error_dict_is_tool_error_not_crash(self, monkeypatch):
        # Regression: radon emits {"file.py": {"error": "..."}} for unparseable
        # files, which used to raise AttributeError inside the runner.
        fake_run(monkeypatch, 0, json.dumps({str(TARGET): {"error": "invalid syntax"}}))
        run = psc.run_radon(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_crash_without_output_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 1, "", "boom")
        run = psc.run_radon(TARGET)
        assert run.issues[0].rule_id == "TOOL_ERROR"

    def test_non_json_stdout_is_tool_error(self, monkeypatch):
        fake_run(monkeypatch, 0, "garbage")
        run = psc.run_radon(TARGET)
        assert run.issues[0].rule_id == "TOOL_ERROR"


# ── flake8 runner — F821 Databricks-globals filter ────────────────────────────


def flake8_line(code, text, row=3):
    return f"{TARGET}||{row}||1||{code}||{text}"


class TestRunFlake8F821Filter:
    def test_databricks_global_is_suppressed(self, monkeypatch):
        fake_run(monkeypatch, 1, flake8_line("F821", "undefined name 'spark'"))
        assert psc.run_flake8(TARGET).issues == []

    def test_real_undefined_name_is_kept(self, monkeypatch):
        # Regression: substring matching let 'sc' inside "'schema'" suppress
        # real undefined-name bugs.
        fake_run(monkeypatch, 1, flake8_line("F821", "undefined name 'schema'"))
        run = psc.run_flake8(TARGET)
        assert len(run.issues) == 1
        assert run.issues[0].sonar_type == psc.BUG

    def test_substring_of_global_is_kept(self, monkeypatch):
        fake_run(monkeypatch, 1, flake8_line("F821", "undefined name 'my_table_df'"))
        assert len(psc.run_flake8(TARGET).issues) == 1

    def test_non_f821_codes_unaffected(self, monkeypatch):
        fake_run(monkeypatch, 1, flake8_line("E711", "comparison to None"))
        run = psc.run_flake8(TARGET)
        assert run.issues[0].sonar_type == psc.BUG


# ── pytest-cov runner ─────────────────────────────────────────────────────────


class TestFindCompanionTestFile:
    def test_finds_test_prefix_convention(self, tmp_path):
        target = tmp_path / "converted_spark.py"
        test_file = tmp_path / "test_converted_spark.py"
        test_file.write_text("def test_x(): pass")
        assert psc._find_companion_test_file(target) == test_file

    def test_finds_test_suffix_convention(self, tmp_path):
        target = tmp_path / "converted_spark.py"
        test_file = tmp_path / "converted_spark_test.py"
        test_file.write_text("def test_x(): pass")
        assert psc._find_companion_test_file(target) == test_file

    def test_falls_back_to_pyspark_pytest_convention(self, tmp_path):
        target = tmp_path / "converted_spark.py"
        test_file = tmp_path / "pyspark_pytest.py"
        test_file.write_text("def test_x(): pass")
        assert psc._find_companion_test_file(target) == test_file

    def test_returns_none_when_no_test_file(self, tmp_path):
        target = tmp_path / "converted_spark.py"
        assert psc._find_companion_test_file(target) is None


def fake_coverage_run(monkeypatch, percent, write_report=True, stderr=""):
    def fake(cmd):
        json_arg = next(a for a in cmd if a.startswith("--cov-report=json:"))
        json_path = Path(json_arg.split("--cov-report=json:", 1)[1])
        if write_report:
            json_path.write_text(json.dumps({"totals": {"percent_covered": percent}}))
        return (0, "", stderr)

    monkeypatch.setattr(psc, "_run", fake)


class TestRunPytestCov:
    def test_no_companion_test_file_is_critical(self, monkeypatch):
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: None)
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "NO_TESTS"
        assert run.issues[0].sonar_severity == psc.CRITICAL

    def test_coverage_at_or_above_gate_passes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: tmp_path / "test_x.py")
        fake_coverage_run(monkeypatch, 92.5)
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is True
        assert run.issues == []

    def test_coverage_below_gate_is_major(self, monkeypatch, tmp_path):
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: tmp_path / "test_x.py")
        fake_coverage_run(monkeypatch, 65.0)
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_severity == psc.MAJOR

    def test_coverage_below_critical_threshold_is_critical(self, monkeypatch, tmp_path):
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: tmp_path / "test_x.py")
        fake_coverage_run(monkeypatch, 30.0)
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is False
        assert run.issues[0].sonar_severity == psc.CRITICAL

    def test_missing_pytest_executable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: tmp_path / "test_x.py")

        def raise_missing(cmd):
            raise FileNotFoundError("pytest not found")

        monkeypatch.setattr(psc, "_run", raise_missing)
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_MISSING"

    def test_missing_json_report_is_tool_error(self, monkeypatch, tmp_path):
        # Covers both a crashed pytest run and pytest-cov not being installed.
        monkeypatch.setattr(psc, "_find_companion_test_file", lambda t: tmp_path / "test_x.py")
        fake_coverage_run(monkeypatch, 0, write_report=False, stderr="unrecognized arguments: --cov")
        run = psc.run_pytest_cov(TARGET)
        assert run.passed is False
        assert run.issues[0].rule_id == "TOOL_ERROR"


# ── validate_file runner loop ─────────────────────────────────────────────────


class TestValidateFile:
    def test_missing_executable_becomes_tool_missing(self, monkeypatch):
        def ok(name):
            return lambda *a, **kw: psc.ToolRun(name=name, passed=True)

        def missing(*a, **kw):
            raise FileNotFoundError("pylint not on PATH")

        for fn in (
            "run_black",
            "run_isort",
            "run_flake8",
            "run_pylint_simplify",
            "run_mypy",
            "run_bandit",
            "run_radon",
            "run_pydocstyle",
            "run_pytest_cov",
        ):
            monkeypatch.setattr(psc, fn, ok(fn.replace("run_", "").replace("_", "-")))
        monkeypatch.setattr(psc, "run_pylint", missing)

        report = psc.validate_file(TARGET)
        assert len(report.tool_runs) == 10
        assert any(i.rule_id == "TOOL_MISSING" for i in report.all_issues)
        assert report.quality_gate_passed() is False
