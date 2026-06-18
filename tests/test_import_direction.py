"""Enforce the dependency direction: the library must not import the drivers.

The core (the pipeline/ package and sam2_utils/) may import each other and
third-party packages, but never the drivers (batch, gui, run_aval, pull_worm) or eval.
Drivers and eval import the library, not the reverse. See ADR 0001 and ADR 0011.

This is a static AST check, so it needs no GPU and no optional deps. It replaces
an external import linter, which does not handle this flat single-file-module
layout cleanly.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = sorted((ROOT / "pipeline").glob("*.py")) + sorted((ROOT / "sam2_utils").glob("*.py"))
FORBIDDEN = {"batch", "gui", "run_aval", "pull_worm", "eval"}


def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:        # absolute imports only
                roots.add(node.module.split(".")[0])
    return roots


def test_library_does_not_import_drivers():
    offenders = {}
    for path in LIBRARY:
        bad = _imported_roots(path) & FORBIDDEN
        if bad:
            offenders[path.name] = sorted(bad)
    assert not offenders, f"library modules import drivers/eval: {offenders}"


if __name__ == "__main__":
    test_library_does_not_import_drivers()
    print(f"OK: dependency direction holds across {len(LIBRARY)} library modules")
