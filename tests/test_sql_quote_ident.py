"""Tests for the centralized SQL identifier-quoting helper + its use at
the untrusted-input surface (knowledge importer). Audit 2026-06-17."""
from __future__ import annotations

import os
import sqlite3
import tempfile


def test_quote_ident_basic():
    from augmentum.utils.sql import quote_ident
    assert quote_ident("users") == '"users"'
    assert quote_ident("col one") == '"col one"'  # space would break bare f-string


def test_quote_ident_escapes_embedded_quotes():
    from augmentum.utils.sql import quote_ident
    # The injection-shaped name is neutralized: embedded " is doubled and
    # the whole thing stays a single quoted identifier.
    assert quote_ident('weird"; DROP TABLE x --') == '"weird""; DROP TABLE x --"'
    # Non-str input is coerced rather than crashing.
    assert quote_ident(123) == '"123"'


def test_analyzer_reuses_canonical_helper():
    # The coder analyzer now delegates to the shared implementation.
    from augmentum.coder.analyzers.builtin.sqlite_analyzer import _quote_ident
    from augmentum.utils.sql import quote_ident
    assert _quote_ident is quote_ident


def _make_db_bytes() -> bytes:
    """A SQLite file whose TABLE NAME contains a space — a bare f-string
    ``FROM my docs`` is a syntax error; only a quoted identifier works."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        con = sqlite3.connect(path)
        con.execute('CREATE TABLE "my docs" (title TEXT, body TEXT)')
        con.execute(
            'INSERT INTO "my docs" (title, body) VALUES (?, ?)',
            ("Hello", "This is a body long enough to pass the importer filter."),
        )
        con.commit()
        con.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def test_importer_handles_hostile_table_name():
    """The importer quotes identifiers from the uploaded file's
    sqlite_master, so a table name that would break a bare f-string is
    handled instead of silently yielding nothing."""
    from augmentum.knowledge.importer import _extract_sqlite

    chunks = _extract_sqlite(_make_db_bytes(), "x.db", "test")
    assert len(chunks) >= 1
    assert any("This is a body" in c.content for c in chunks)
