"""
Unit tests for sems_validator.py — layer gating, BRD-area aggregation, and the
run_sems_gate entry point.

Hermetic: external tools are never invoked. check_compliance / validate_file /
validate_source are monkeypatched at the sems_validator module level.

Run with:  python -m pytest subagents/sems_agent/compliance/tests/ -v
"""

from pathlib import Path

import pytest

# Injected into sys.modules by conftest.py to avoid the ADK-heavy
# sems_agent package __init__.py.
import pre_sonar_check as psc
import sems_validator as sv

BROKEN_SOURCE = "def f(:\n    pass\n"
FIXED_SOURCE = "def f():\n    pass\n"


def make_issue(tool="pylint", severity=psc.MINOR, rule_id="X0001"):
    return psc.Issue(
        file="t.py",
        line=1,
        col=0,
        tool=tool,
        rule_id=rule_id,
        sonar_type=psc.SMELL,
        sonar_severity=severity,
        message="msg",
    )


def make_violation(rule_id="SEC001", severity="hard"):
    return sv.SEMSViolation(
        rule_id=rule_id,
        severity=severity,
        category="security",
        message=f"[{rule_id}] msg",
        remediation="fix it",
        line_number=1,
    )


def make_compliance_result(violations=()):
    structured = list(violations)
    return sv.ComplianceResult(
        passed=not any(v.severity == "hard" for v in structured),
        violations=[v.message for v in structured if v.severity == "hard"],
        warnings=[v.message for v in structured if v.severity == "soft"],
        structured_violations=structured,
    )


# ── Layer 1 ───────────────────────────────────────────────────────────────────


class TestLayer1Syntax:
    def test_valid_source_returns_none(self):
        assert sv._layer1_syntax("x = 1\n") is None

    def test_syntax_error_returns_message_with_line(self):
        msg = sv._layer1_syntax(BROKEN_SOURCE)
        assert msg is not None
        assert "line 1" in msg


# ── BRD-area aggregation ──────────────────────────────────────────────────────


class TestAggregateAreas:
    def test_hard_sems_violation_blocks_its_area(self):
        result = make_compliance_result([make_violation("SEC001", "hard")])
        areas = sv._aggregate_areas(result, None)
        assert areas["security"].sems_blocking == 1
        assert areas["security"].passed is False

    def test_soft_sems_violation_reports_but_does_not_block(self):
        result = make_compliance_result([make_violation("SPARK001", "soft")])
        areas = sv._aggregate_areas(result, None)
        assert areas["modular_design"].sems_findings == 1
        assert areas["modular_design"].sems_blocking == 0
        assert all(a.passed for a in areas.values())

    def test_unknown_rule_defaults_to_readability(self):
        result = make_compliance_result([make_violation("NEW_RULE_999", "soft")])
        areas = sv._aggregate_areas(result, None)
        assert areas["code_readability_and_structure"].sems_findings == 1

    def test_major_tool_issue_blocks(self):
        run = psc.ToolRun(
            name="bandit", passed=False, issues=[make_issue("bandit", psc.MAJOR)]
        )
        report = psc.ValidationReport(target="t.py", tool_runs=[run])
        areas = sv._aggregate_areas(None, report)
        assert areas["security"].tool_blocking == 1
        assert areas["security"].passed is False

    def test_minor_tool_issue_does_not_block(self):
        run = psc.ToolRun(
            name="bandit", passed=False, issues=[make_issue("bandit", psc.MINOR)]
        )
        report = psc.ValidationReport(target="t.py", tool_runs=[run])
        areas = sv._aggregate_areas(None, report)
        assert areas["security"].tool_findings == 1
        assert areas["security"].tool_blocking == 0


# ── validate_source ───────────────────────────────────────────────────────────


class TestValidateSource:
    def test_syntax_error_short_circuits_layers_2_and_3(self, monkeypatch):
        called = []
        monkeypatch.setattr(sv, "check_compliance", lambda s: called.append("sems"))
        monkeypatch.setattr(sv, "validate_file", lambda p, fix: called.append("tools"))
        report = sv.validate_source(BROKEN_SOURCE, Path("t.py"))
        assert report.syntax_ok is False
        assert report.syntax_error is not None
        assert called == []

    def test_clean_source_has_no_blocking_findings(self, monkeypatch):
        monkeypatch.setattr(sv, "check_compliance", lambda s: make_compliance_result())
        monkeypatch.setattr(
            sv,
            "validate_file",
            lambda p, fix: psc.ValidationReport(target=str(p)),
        )
        report = sv.validate_source("x = 1\n", Path("t.py"))
        assert report.syntax_ok is True
        assert all(a.passed for a in report.area_summaries.values())

    def test_hard_violation_blocks_its_area(self, monkeypatch):
        monkeypatch.setattr(
            sv,
            "check_compliance",
            lambda s: make_compliance_result([make_violation("SEC001", "hard")]),
        )
        monkeypatch.setattr(
            sv,
            "validate_file",
            lambda p, fix: psc.ValidationReport(target=str(p)),
        )
        report = sv.validate_source("x = 1\n", Path("t.py"))
        assert report.area_summaries["security"].passed is False

    def test_tool_error_blocks_its_area(self, monkeypatch):
        # A crashed mandatory analyzer must be flagged as blocking (TOOL_ERROR is MAJOR).
        monkeypatch.setattr(sv, "check_compliance", lambda s: make_compliance_result())
        monkeypatch.setattr(
            sv,
            "validate_file",
            lambda p, fix: psc.ValidationReport(
                target=str(p),
                tool_runs=[psc._tool_error_run("pylint", Path("t.py"), "boom")],
            ),
        )
        report = sv.validate_source("x = 1\n", Path("t.py"))
        assert not all(a.passed for a in report.area_summaries.values())


# ── run_sems_gate ─────────────────────────────────────────────────────────────


@pytest.fixture()
def gate_env(monkeypatch, tmp_path):
    """Hermetic run_sems_gate: fake validate_source records the on-disk file
    content at validation time; report writing is disabled."""
    target = tmp_path / "job_spark.py"
    disk_at_validation = []

    def fake_validate_source(source, path):
        disk_at_validation.append(path.read_text(encoding="utf-8"))
        error = sv._layer1_syntax(source)
        if error:
            return sv.SemsReport(
                target=str(path), syntax_ok=False, syntax_error=error
            )
        return sv.SemsReport(target=str(path), syntax_ok=True)

    monkeypatch.setattr(sv, "validate_source", fake_validate_source)
    monkeypatch.setattr(
        sv, "_write_reports", lambda report, path: {"markdown": path, "json": path}
    )
    return target, disk_at_validation


class TestRunSemsGate:
    def test_syntax_error_leaves_file_untouched(self, gate_env):
        target, _ = gate_env
        target.write_text(BROKEN_SOURCE, encoding="utf-8")

        report = sv.run_sems_gate(target)

        assert target.read_text(encoding="utf-8") == BROKEN_SOURCE
        assert report.syntax_ok is False

    def test_valid_file_reported_without_modification(self, gate_env):
        target, _ = gate_env
        target.write_text(FIXED_SOURCE, encoding="utf-8")

        report = sv.run_sems_gate(target)

        assert target.read_text(encoding="utf-8") == FIXED_SOURCE
        assert report.syntax_ok is True
