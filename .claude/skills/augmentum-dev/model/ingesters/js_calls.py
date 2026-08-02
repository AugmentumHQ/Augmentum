"""JS-calls ingester — populates the ``js_calls`` table.

Source: ``references/frontend_api_calls.json`` (already auto-generated
by ``refresh_refs.py`` from fetch / WebSocket / sendBeacon call sites
in ``ui/scripts/``).

Path normalisation must match the endpoints ingester so the orphan
JOIN (endpoints LEFT JOIN js_calls ON method, path_template) works
without a fuzzy match.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

CALLS_JSON = Path(".claude") / "skills" / "augmentum-dev" / "references" / "frontend_api_calls.json"
PARAM_NAME_RE = re.compile(r"\{[^/{}]+\}")


def _normalise_path(p: str) -> str:
    return PARAM_NAME_RE.sub("{param}", p)


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    src = project_root / CALLS_JSON
    if not src.is_file():
        return
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    calls = payload.get("calls") or []
    if not calls:
        return

    file_id_by_path: dict[str, int] = {
        row["path"]: int(row["id"])
        for row in db.execute("SELECT id, path FROM files")
    }

    db.execute("BEGIN")
    try:
        db.execute("DELETE FROM js_calls")
        for entry in calls:
            raw_url = entry.get("url") or ""
            method = (entry.get("method") or "").upper() or None
            file = entry.get("file") or ""
            line = int(entry.get("line") or 0)
            if not raw_url or not file:
                continue
            file_id = file_id_by_path.get(file)
            if file_id is None:
                continue
            db.execute(
                """INSERT INTO js_calls
                   (file_id, line, method, path_template, has_error_handler)
                   VALUES (?, ?, ?, ?, 0)""",
                (file_id, line, method, _normalise_path(raw_url)),
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
