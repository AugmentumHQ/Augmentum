"""Toy user-lookup endpoint."""

from __future__ import annotations

import sqlite3


def find_user_by_name(conn: sqlite3.Connection, name: str) -> list[tuple]:
    cur = conn.cursor()
    # BUG: name is interpolated directly into the SQL string. A name of
    # `' OR '1'='1` returns every row; `'; DROP TABLE users; --` is worse.
    cur.execute(f"SELECT id, email FROM users WHERE name = '{name}'")
    return cur.fetchall()


def setup_demo(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.executemany(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        [("alice", "a@x"), ("bob", "b@x")],
    )
    conn.commit()
