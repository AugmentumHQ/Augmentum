"""Small SQL helpers.

SQLite (like all SQL) doesn't accept *identifiers* — table/column names —
as bound parameters; only values bind. Code that must interpolate an
identifier into a statement has to quote it itself. :func:`quote_ident`
is the one sanctioned way to do that: wrap in double quotes and double
any embedded quote, per the SQL identifier-quoting rule.

Use it for EVERY interpolated identifier, even when the source looks
trusted (``sqlite_master`` / ``PRAGMA`` output, hardcoded frozensets) —
it's defense-in-depth, it removes the "one copy-paste from an injection"
fragility of bare f-string SQL, and it keeps the red_team / security
scanners honest without per-site "trust me, this is safe" assertions.

This is the canonical implementation; the knowledge importer and the
coder SQLite analyzer both consume it (audit 2026-06-17).
"""
from __future__ import annotations


def quote_ident(name: str) -> str:
    """Return *name* as a safely-quoted SQL identifier.

    >>> quote_ident("users")
    '"users"'
    >>> quote_ident('weird"; DROP TABLE x --')
    '"weird""; DROP TABLE x --"'
    """
    return '"' + str(name).replace('"', '""') + '"'


__all__ = ["quote_ident"]
