"""Per-owner SQL scoping helper (audit 2026-06-17).

Several companion-autonomy queries (initiative scoring, revisit caps,
dream lookup) were keyed by ``companion_id`` only, so on a multi-tenant
box one user's data drove another user's scoring/queue. This helper
produces a WHERE fragment that scopes to one owner while leaving the
unowned single-user box (``owner_user_id == ''``) as a pass-through —
a backward-compatible no-op there.
"""
from __future__ import annotations


def owner_clause(owner_user_id: str) -> tuple[str, tuple[str, str]]:
    """Return ``(sql_fragment, params)`` scoping a query to ``owner``.

    Splice the fragment into a WHERE (it begins with ``AND``) and unpack
    the params right after the query's existing binds::

        frag, p = owner_clause(owner)
        cur = await conn.execute(
            f"SELECT ... WHERE companion_id = ? {frag}",
            (companion_id, *p),
        )

    When ``owner`` is ``''`` the ``? = ''`` branch short-circuits true so
    no filtering happens — the legacy unowned single-user behavior.
    """
    return "AND (? = '' OR user_id = ?)", (owner_user_id, owner_user_id)


def owner_clause_nullable(owner_user_id: str) -> tuple[str, tuple[str, str]]:
    """Like :func:`owner_clause` but also matches rows whose ``user_id``
    is NULL — needed for tables backfilled with nullable user_id (e.g.
    ``dream_entries``, mig 089) where pre-pivot rows must stay visible on
    the unowned box."""
    return (
        "AND (? = '' OR user_id = ? OR user_id IS NULL)",
        (owner_user_id, owner_user_id),
    )
