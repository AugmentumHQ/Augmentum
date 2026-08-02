"""Tables ingester — derives the canonical user-scoped tables list.

Reads each migration's raw SQL (already stored in the ``migrations``
table) and identifies:
  * tables defined with a ``user_id`` column in their CREATE TABLE
    statement (scoping_kind='create')
  * tables retroactively scoped via ``ALTER TABLE … ADD COLUMN
    user_id`` (scoping_kind='alter')

For each table found, the lowest-numbered migration that scoped it
becomes ``scoping_migration``. The lowest-numbered migration that
defined the table at all (CREATE) becomes ``defining_migration``.

This is the single source of truth for the user-scoped table list
that CLAUDE.md / SKILL.md document.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Captures the table name of any CREATE TABLE [IF NOT EXISTS] statement
# whose body (up to the next semicolon) contains a column literally
# named user_id. The DOTALL flag lets the body span lines.
CREATE_USER_SCOPED_RE = re.compile(
    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\([^;]*\buser_id\b",
    re.IGNORECASE | re.DOTALL,
)
# Captures the table name when user_id is added retroactively.
ALTER_ADD_USER_ID_RE = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+user_id\b",
    re.IGNORECASE,
)
# Captures the table name of any CREATE TABLE (whether or not it has
# user_id). Used so ``defining_migration`` is correct even for tables
# that gain user_id later via ALTER.
CREATE_ANY_RE = re.compile(
    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\(",
    re.IGNORECASE,
)
# Captures ``ALTER TABLE old RENAME TO new``. Needed so renames carry
# the user_scoped flag forward to the new name. Without this, a rename
# leaves the audit listing both names (the old as user-scoped, the new
# as missing entirely).
ALTER_RENAME_RE = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+RENAME\s+TO\s+(\w+)\b",
    re.IGNORECASE,
)


def ingest(project_root: Path, db: sqlite3.Connection) -> None:  # noqa: ARG001
    """Re-derive the ``tables`` table from migrations. Always full rebuild;
    cheap because ``migrations`` row count is < a few hundred and SQL
    text fits comfortably in memory.
    """
    rows = list(db.execute(
        "SELECT number, raw_sql FROM migrations ORDER BY number"
    ))
    # First pass: defining_migration for every CREATE TABLE we see.
    defining: dict[str, int] = {}
    for r in rows:
        for m in CREATE_ANY_RE.finditer(r["raw_sql"]):
            name = m.group(1).lower()
            defining.setdefault(name, int(r["number"]))

    # Second pass: scoping_migration / scoping_kind for tables that
    # ever gained a user_id column.
    scoping: dict[str, tuple[int, str]] = {}
    for r in rows:
        sql = r["raw_sql"]
        for m in CREATE_USER_SCOPED_RE.finditer(sql):
            name = m.group(1).lower()
            scoping.setdefault(name, (int(r["number"]), "create"))
        for m in ALTER_ADD_USER_ID_RE.finditer(sql):
            name = m.group(1).lower()
            scoping.setdefault(name, (int(r["number"]), "alter"))

    # Third pass: follow ALTER TABLE … RENAME TO … in migration order.
    # The new name inherits the defining + scoping metadata; the old
    # name is dropped from the canonical lists. Without this the audit
    # would double-count (showing both names) and the user_scoped list
    # would skip the new name entirely.
    for r in rows:
        for m in ALTER_RENAME_RE.finditer(r["raw_sql"]):
            old = m.group(1).lower()
            new = m.group(2).lower()
            if old in defining and new not in defining:
                defining[new] = defining.pop(old)
            elif old in defining:
                # Both names defined — keep the new one's definition.
                defining.pop(old)
            if old in scoping and new not in scoping:
                scoping[new] = scoping.pop(old)
            elif old in scoping:
                scoping.pop(old)

    # Build the final row set: every table we ever saw, scoped or not.
    db.execute("BEGIN")
    try:
        db.execute("DELETE FROM tables")
        for name, def_mig in defining.items():
            scope = scoping.get(name)
            user_scoped = 1 if scope else 0
            scope_mig = scope[0] if scope else None
            scope_kind = scope[1] if scope else None
            db.execute(
                """INSERT INTO tables
                   (name, defining_migration, user_scoped, scoping_migration, scoping_kind)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, def_mig, user_scoped, scope_mig, scope_kind),
            )
        # Tables scoped by ALTER without an observed CREATE in this
        # corpus (rare — historical artifact) still deserve a row.
        for name, (scope_mig, scope_kind) in scoping.items():
            if name in defining:
                continue
            db.execute(
                """INSERT INTO tables
                   (name, defining_migration, user_scoped, scoping_migration, scoping_kind)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, None, 1, scope_mig, scope_kind),
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
