"""SQLite connection management for the codebase model.

The model lives at ``.augmentum-dev-cache/codebase.db`` under the project
root — gitignored, rebuilt opportunistically. WAL mode + 30s busy timeout
so concurrent audit invocations don't lock each other out.

Schema is applied on first open (idempotent ``CREATE IF NOT EXISTS``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_FILE = Path(__file__).parent / "schema.sql"
CACHE_DIR_NAME = ".augmentum-dev-cache"
DB_FILENAME = "codebase.db"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` looking for the augmentum project root.

    Project root is identified by the simultaneous presence of
    ``augmentum/proxy/`` and ``ui/`` (matching the heuristic in the
    other skill scripts). Raises FileNotFoundError if not found.
    """
    p = (start or Path(__file__).resolve()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    raise FileNotFoundError("augmentum project root not found from " + str(p))


def _cache_db_path(project_root: Path) -> Path:
    cache_dir = project_root / CACHE_DIR_NAME
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / DB_FILENAME


def open_model(project_root: Path | None = None) -> sqlite3.Connection:
    """Open (and lazily initialise) the codebase model db."""
    root = project_root or find_project_root()
    db_path = _cache_db_path(root)
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    return conn
