"""Structural guard on the ACL proof in test_acl_gating.py.

That file proves every store choke point consults `can_access()` by
monkeypatching one name: `qmd_py.store._common.can_access`. The proof
only holds while `_common` is the *only* module in the store package
that imports `can_access` - any sibling importing it directly would bind
its own reference at import time, escape the patch, and be silently
excluded from the proof while still looking gated.

Pure source inspection: no Postgres, no router.
"""

import ast
import pathlib

import qmd_py.store

_ALLOWED = {"_common.py"}


def _modules_importing_can_access() -> set[str]:
    package_dir = pathlib.Path(qmd_py.store.__file__).parent
    importers = set()
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "can_access" for alias in node.names
            ):
                importers.add(path.name)
    return importers


def test_only_common_imports_can_access() -> None:
    assert _modules_importing_can_access() == _ALLOWED


def test_common_actually_still_imports_it() -> None:
    """Guards the guard: if `can_access` were renamed or the import
    dropped, the check above would pass vacuously on an empty set."""
    assert "_common.py" in _modules_importing_can_access()
