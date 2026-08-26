"""
Ensure the compliance modules are importable without triggering the sems_agent
package __init__.py (which requires ADK and other heavy dependencies).

compliance_checker.py has no package-relative imports, so it is loaded as a
plain top-level module. pre_sonar_check.py and sems_validator.py use relative
imports (``from .compliance_checker import ...``), so they are loaded through
a synthetic package whose __path__ points at the compliance directory, then
aliased to top-level names for the tests.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

_COMPLIANCE_DIR = Path(__file__).resolve().parents[1]

_CHECKER_PATH = _COMPLIANCE_DIR / "compliance_checker.py"
_spec = importlib.util.spec_from_file_location("compliance_checker", _CHECKER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["compliance_checker"] = _mod
_spec.loader.exec_module(_mod)

_PKG_NAME = "sems_compliance_pkg"
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [str(_COMPLIANCE_DIR)]
    sys.modules[_PKG_NAME] = _pkg
sys.modules["pre_sonar_check"] = importlib.import_module(f"{_PKG_NAME}.pre_sonar_check")
sys.modules["sems_validator"] = importlib.import_module(f"{_PKG_NAME}.sems_validator")
