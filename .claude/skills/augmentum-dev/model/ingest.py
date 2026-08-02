"""Ingester refresh loop.

Each ingester is a pure ``(project_root, sqlite3.Connection) -> None``
function. The orchestrator (``refresh``) runs them in dependency order;
ingesters internally honour mtime/sha gating so re-runs on a clean
working tree are near-instant.

Ordering matters: ``files`` populates the file_id catalog every other
ingester references; ``migrations`` populates rows that ``tables``
joins against. The order encoded below mirrors that dependency.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from .ingesters import (
    endpoints,
    files,
    handler_signatures,
    js_calls,
    migrations,
    registrations,
    settings,
    tables,
    test_files,
)

# Ordered list of (name, callable). Stable order = stable refresh
# semantics; new ingesters append to the end unless they have a
# specific dependency. ``files`` MUST run first — every other
# ingester resolves source paths to file_id via the files table.
# ``handler_signatures`` MUST run after ``endpoints`` (FK dependency).
INGESTERS: list[tuple[str, Callable[[Path, sqlite3.Connection], None]]] = [
    ("files",              files.ingest),
    ("migrations",         migrations.ingest),
    ("tables",             tables.ingest),
    ("registrations",      registrations.ingest),
    ("endpoints",          endpoints.ingest),
    ("js_calls",           js_calls.ingest),
    ("settings",           settings.ingest),
    ("test_files",         test_files.ingest),
    ("handler_signatures", handler_signatures.ingest),
]


def refresh(db: sqlite3.Connection, project_root: Path) -> dict[str, int]:
    """Run every ingester in order. Returns per-ingester row counts."""
    counts: dict[str, int] = {}
    for name, fn in INGESTERS:
        fn(project_root, db)
        counts[name] = _row_count_for(db, name)
    return counts


def _row_count_for(db: sqlite3.Connection, ingester_name: str) -> int:
    table = ingester_name  # convention — ingester name == primary table
    try:
        cur = db.execute(f"SELECT COUNT(*) FROM {table}")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
