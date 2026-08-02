"""Migrations ingester — populates the ``migrations`` table.

Parses ``augmentum/state/migrations/NNN_*.sql`` filenames for the
number + slug. Stores the raw SQL text so the ``tables`` ingester
can re-derive scoping facts without a second filesystem pass.

Mtime-incremental: a migration row is rewritten only if the source
file's stored sha doesn't match (cheap path: skip on mtime equality).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

MIG_NAME_RE = re.compile(r"^(\d+)_(.+)\.sql$")
MIG_DIR = Path("augmentum") / "state" / "migrations"


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    """Refresh ``migrations`` rows from the migrations dir."""
    mig_dir = project_root / MIG_DIR
    if not mig_dir.is_dir():
        return

    # Map known sql files to their file_id so we can write the FK.
    rel_prefix = MIG_DIR.as_posix()
    file_id_by_path: dict[str, int] = {
        row["path"]: int(row["id"])
        for row in db.execute(
            "SELECT id, path FROM files WHERE path LIKE ?",
            (rel_prefix + "/%",),
        )
    }

    seen_numbers: set[int] = set()
    db.execute("BEGIN")
    try:
        # Cache existing (number, sha) so we skip identical re-ingests.
        existing = {
            int(row["number"]): row["raw_sql"]
            for row in db.execute("SELECT number, raw_sql FROM migrations")
        }
        for sql_path in sorted(mig_dir.glob("*.sql")):
            m = MIG_NAME_RE.match(sql_path.name)
            if not m:
                continue
            number = int(m.group(1))
            slug = m.group(2)
            rel = (MIG_DIR / sql_path.name).as_posix()
            file_id = file_id_by_path.get(rel)
            if file_id is None:
                # files ingester missed it (shouldn't happen — it
                # walks augmentum/). Skip rather than insert NULL FK.
                continue
            raw_sql = sql_path.read_text(encoding="utf-8", errors="replace")
            seen_numbers.add(number)
            if existing.get(number) == raw_sql:
                continue
            db.execute(
                """INSERT INTO migrations (number, slug, file_id, raw_sql)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(number) DO UPDATE SET
                       slug = excluded.slug,
                       file_id = excluded.file_id,
                       raw_sql = excluded.raw_sql""",
                (number, slug, file_id, raw_sql),
            )
        # Drop migrations that vanished from disk.
        if seen_numbers:
            placeholders = ",".join("?" for _ in seen_numbers)
            db.execute(
                f"DELETE FROM migrations WHERE number NOT IN ({placeholders})",
                tuple(seen_numbers),
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
