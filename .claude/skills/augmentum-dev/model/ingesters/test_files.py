"""Test-files ingester — populates ``test_files`` with the test suite
catalog and a best-effort target-module mapping.

Heuristics for target-module attribution (in order):
  1. Filename: ``tests/test_<X>.py`` → look for ``augmentum/**/<X>.py``
     OR ``augmentum/**/<X>/__init__.py`` matching anywhere in the tree.
  2. Imports: any ``from augmentum.X.Y import …`` line collected from
     the test file's source contributes ``augmentum/X/Y.py`` to targets.
     This catches tests that span modules (e.g.
     test_chain_integration.py → tools/chain.py + tools/registry.py).
  3. Test count: number of ``def test_<…>`` definitions parsed from
     the source. Cheap proxy for "how much coverage does this file
     contribute" without parametrize expansion.

The targets list is JSON-encoded for forward compatibility — future
queries may want richer metadata per target (line ranges, pytest
markers) without a schema migration.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

TESTS_DIR = Path("tests")
TEST_FILE_RE = re.compile(r"^test_(.+)\.py$")
DEF_TEST_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+", re.MULTILINE)
IMPORT_RE = re.compile(
    r"^\s*from\s+(augmentum(?:\.\w+)+)\s+import\s+",
    re.MULTILINE,
)


def _candidate_target_modules(
    test_path: Path,
    project_root: Path,
    augmentum_files: set[str],
) -> list[str]:
    """Best-effort: which production modules does ``test_path`` exercise?"""
    targets: set[str] = set()

    # 1. Filename heuristic
    m = TEST_FILE_RE.match(test_path.name)
    if m:
        stem = m.group(1)
        # Try direct file match (test_foo.py → augmentum/**/foo.py)
        for f in augmentum_files:
            base = Path(f).stem
            if base == stem:
                targets.add(f)
        # Also match underscore prefix (test_files_routes.py → files_routes.py)
        # Already covered by the stem comparison above.

    # 2. Import-based attribution
    try:
        text = test_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return sorted(targets)
    for imp in IMPORT_RE.finditer(text):
        module_dotted = imp.group(1)
        # augmentum.foo.bar -> augmentum/foo/bar.py
        module_path = module_dotted.replace(".", "/") + ".py"
        if module_path in augmentum_files:
            targets.add(module_path)
        # Also try as a package __init__.py
        pkg_path = module_dotted.replace(".", "/") + "/__init__.py"
        if pkg_path in augmentum_files:
            targets.add(pkg_path)

    return sorted(targets)


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    tests_dir = project_root / TESTS_DIR
    if not tests_dir.is_dir():
        return

    # Build the set of in-repo augmentum/*.py paths once for fast
    # membership tests during attribution.
    augmentum_files = {
        row["path"]
        for row in db.execute(
            "SELECT path FROM files WHERE path LIKE 'augmentum/%' AND lang = 'python'"
        )
    }

    file_id_by_path: dict[str, int] = {
        row["path"]: int(row["id"])
        for row in db.execute("SELECT id, path FROM files WHERE path LIKE 'tests/%'")
    }

    db.execute("BEGIN")
    try:
        db.execute("DELETE FROM test_files")
        for test_path in sorted(tests_dir.rglob("test_*.py")):
            try:
                rel = test_path.relative_to(project_root)
            except ValueError:
                continue
            rel_str = rel.as_posix()
            file_id = file_id_by_path.get(rel_str)
            if file_id is None:
                continue
            try:
                source = test_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            test_count = len(DEF_TEST_RE.findall(source))
            targets = _candidate_target_modules(test_path, project_root, augmentum_files)
            db.execute(
                """INSERT INTO test_files (file_id, test_count, target_modules)
                   VALUES (?, ?, ?)""",
                (file_id, test_count, json.dumps(targets)),
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
