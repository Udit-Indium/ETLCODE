"""Static template renderers for the Shell modularization deliverables:
README.md, LICENSE.txt, and sonar-project.properties.

These are documentation/config scaffolding, not generated code — nothing
here needs a model call or AST analysis, so it's plain string templating
parameterized by what ``splitter.split_module`` already found.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence


def _bullet_list(names: Sequence[str], empty_note: str) -> str:
    if not names:
        return f"- _{empty_note}_"
    return "\n".join(f"- `{name}`" for name in names)


def render_readme(
    module_name: str,
    *,
    main_functions: Sequence[str],
    utility_functions: Sequence[str],
    entrypoint: Optional[str],
) -> str:
    """Render README.md documenting the modularized project layout."""
    entry_reference = f"`{entrypoint}`" if entrypoint else "the function(s) documented in `main.py`"
    example_call = entrypoint or "<function_name>"

    return f'''# {module_name}

Modularized output of the `{module_name}` conversion pipeline: restructured from a
single generated script into a standard project layout so it can be reused,
tested, and scanned independently of the notebook it came from.

## What this project does

Runs the `{module_name}` pipeline logic. The primary entrypoint is {entry_reference}.

## Project structure

```
{module_name}/
├── README.md                  This file
├── LICENSE.txt                 Licensing information
├── main.py                     Core business logic
├── utilities.py                Reusable helper functions
├── config.py                   Configuration values
├── usage.py                    Example entry point showing how to call main.py
├── sonar-project.properties    SonarQube scan configuration
└── tests/
    └── test_main.py             Unit tests for main.py
```

## Prerequisites

- Python 3.9+
- Whatever packages this module's `import` statements require. Dependency
  pinning is intentionally not handled through `config.py` — add a
  `requirements.txt` or `pyproject.toml` alongside this README for your
  environment's package manager.

## Setup

Install the dependencies for your environment, then run from this directory
so `main`, `utilities`, and `config` resolve as plain imports.

## Usage

```
python usage.py
```

or, programmatically:

```python
from main import {example_call}

result = {example_call}(...)
```

## Functions

### main.py — business logic
{_bullet_list(main_functions, "no business-logic functions were extracted")}

### utilities.py — reusable helpers
{_bullet_list(utility_functions, "every function was pipeline-specific and lives in main.py instead")}

## Running the unit tests

```
pytest tests/ --cov=main --cov-report=term-missing
```

Target: at least 80% coverage on `main.py`, per the Sonar Way Quality Gate.

## Running SonarQube

```
sonar-scanner -Dproject.settings=sonar-project.properties
```

## Expected inputs / outputs

See the docstrings and type hints on the functions listed above — each
documents its own Args:/Returns: contract.
'''


def render_license(company_name: str = "Shell") -> str:
    """Render a placeholder LICENSE.txt.

    This deliberately does not invent real legal terms — see the TODO in the
    body. Compare SHELL_NAME001 in rules.yaml, which takes the same stance
    ("leave a TODO and flag for the data steward rather than guessing") for
    an unresolvable naming convention.
    """
    year = date.today().year
    return f'''Copyright (c) {year} {company_name}. All rights reserved.

This file is a placeholder. It does not constitute an actual license grant.

TODO(legal): confirm the licensing / proprietary-use terms that apply to this
codebase with the {company_name} legal and compliance team, and replace this
notice with the approved text before this project is distributed or shared
outside the team that generated it.
'''


def render_sonar_properties(project_key: str, project_name: str) -> str:
    """Render sonar-project.properties for the modularized layout."""
    return f'''sonar.projectKey={project_key}
sonar.projectName={project_name}
sonar.projectVersion=1.0
sonar.sources=.
sonar.tests=tests
sonar.exclusions=tests/**,usage.py
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.version=3.9,3.10,3.11,3.12
sonar.sourceEncoding=UTF-8
'''
