"""Registrations ingester — captures app.include_router() calls.

Source: ``augmentum/proxy/server.py`` (the only place that registers
routers in this codebase). Captures (line, router_var) pairs so the
facts registry can compute count + first-line claims.

Mtime-incremental at the ingester level: full re-parse if server.py
changed since last ingest, otherwise no-op.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

SERVER_PY = Path("augmentum") / "proxy" / "server.py"
INCLUDE_ROUTER_RE = re.compile(
    r"^\s*app\.include_router\(\s*([A-Za-z_][\w]*)",
)


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    server_path = project_root / SERVER_PY
    if not server_path.is_file():
        return

    file_row = db.execute(
        "SELECT id FROM files WHERE path = ?",
        (SERVER_PY.as_posix(),),
    ).fetchone()
    if not file_row:
        # files ingester didn't pick up server.py — nothing to FK to.
        return
    file_id = int(file_row["id"])

    rows = []
    for lineno, line in enumerate(
        server_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        m = INCLUDE_ROUTER_RE.match(line)
        if not m:
            continue
        rows.append((file_id, lineno, m.group(1)))

    db.execute("BEGIN")
    try:
        # Server.py is the only registrar — wipe + reinsert is fine.
        db.execute("DELETE FROM registrations WHERE file_id = ?", (file_id,))
        if rows:
            db.executemany(
                "INSERT INTO registrations (file_id, line, router_var) "
                "VALUES (?, ?, ?)",
                rows,
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
