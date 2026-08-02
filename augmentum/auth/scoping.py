"""Data isolation helpers for multi-tenant queries."""

from __future__ import annotations


def user_where(user_id: str) -> tuple[str, tuple]:
    """Return SQL clause and params for user scoping.

    Usage:
        clause, params = user_where(user_id)
        await db.execute(f"SELECT * FROM t WHERE id = ?{clause}", (id,) + params)
    """
    return " AND user_id = ?", (user_id,)


def user_insert_fields() -> str:
    """Return ', user_id' for appending to INSERT column lists."""
    return ", user_id"


def user_insert_placeholder() -> str:
    """Return ', ?' for appending to INSERT value placeholders."""
    return ", ?"
