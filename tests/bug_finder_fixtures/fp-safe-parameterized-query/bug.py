"""FP-bait: looks like SQL injection at a glance, but isn't.

The query string is built with f-string formatting, but only static column
names — never user input. The user value is passed via a placeholder.
A heuristic detector that flags any f-string near `execute` will trip.
"""

from __future__ import annotations

import sqlite3

ALLOWED_COLUMNS = {"id", "email", "name"}


def find_user_field(conn: sqlite3.Connection, column: str, user_id: int) -> str | None:
    # f-string IS used — but on `column`, which we validate against a
    # constant whitelist. user_id flows through a placeholder. No injection.
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"unknown column {column!r}")
    cur = conn.cursor()
    cur.execute(f"SELECT {column} FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    return row[0] if row else None
