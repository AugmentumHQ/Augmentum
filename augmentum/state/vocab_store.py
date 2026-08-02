"""Vocabulary-learning persistence — the ``vocab_state`` table.

One row per (user, language, word) tracking that user's spaced-
repetition state for that vocabulary item. All CRUD is user-scoped. The
FSRS algorithm itself lives in ``augmentum/learning/fsrs.py``; this layer
is pure persistence — callers compute the new FSRS state and pass it in,
matching the thin-wrapper style of the other ``augmentum/state/*_store``
modules.

Timestamps are stored as SQLite ``datetime('now')``-format strings
(``"YYYY-MM-DD HH:MM:SS"`` in UTC) so they string-compare correctly with
the column defaults. Use the module-level :func:`now_ts` / :func:`future_ts`
helpers when constructing timestamps that go into this table.

See ``augmentum/state/migrations/145_language_learning.sql`` and
``docs/superpowers/specs/2026-05-11-language-learning-system.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Matches SQLite's ``datetime('now')`` output so our values and the
# column defaults sort identically as plain strings.
_TS_FMT = "%Y-%m-%d %H:%M:%S"

# A freshly-clicked, never-reviewed word becomes due this many days out.
FIRST_DUE_DAYS: int = 1


def now_ts() -> str:
    return datetime.now(UTC).strftime(_TS_FMT)


def future_ts(days: float) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).strftime(_TS_FMT)


def parse_ts(value: str | None) -> datetime | None:
    """Parse a stored timestamp back to an aware UTC datetime, or None."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _TS_FMT).replace(tzinfo=UTC)
    except ValueError:
        # Tolerate ISO-ish values that may have leaked in.
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None


def elapsed_days_since(value: str | None) -> float:
    """Whole-and-fractional days between ``value`` and now (>= 0)."""
    dt = parse_ts(value)
    if dt is None:
        return 0.0
    delta = datetime.now(UTC) - dt
    return max(0.0, delta.total_seconds() / 86400.0)


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row, strict=True))


class VocabStore:
    """CRUD for the ``vocab_state`` table."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Writes ────────────────────────────────────────────────────────

    async def add_word(
        self,
        *,
        user_id: str,
        lang_code: str,
        word_id: str,
        source_surface: str,
        source_ref: str = "",
    ) -> bool:
        """Add a word to the user's learning queue. Idempotent.

        The act of adding is itself one comprehension exposure, so a new
        row starts with ``exposure_input = 1``. Returns True if a new row
        was inserted, False if the word was already queued (re-clicking
        is a no-op — the caller can surface "already in your queue").
        """
        if not user_id:
            raise ValueError("vocab_store.add_word requires user_id")
        if not lang_code or not word_id:
            raise ValueError("vocab_store.add_word requires lang_code and word_id")
        due = future_ts(FIRST_DUE_DAYS)
        cursor = await self._conn.execute(
            """INSERT INTO vocab_state
                   (user_id, lang_code, word_id, fsrs_due_at, mastery_state,
                    source_surface, source_ref, exposure_input)
               VALUES (?, ?, ?, ?, 'new', ?, ?, 1)
               ON CONFLICT(user_id, lang_code, word_id) DO NOTHING""",
            (user_id, lang_code, word_id, due, source_surface, source_ref or ""),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    async def clear_lang(
        self,
        *,
        user_id: str,
        lang_code: str,
    ) -> int:
        """Delete every row in ``vocab_state`` for this user / language —
        used by the "reseed from curriculum" path to wipe queues that
        accumulated low-value entries from the legacy frequency seeder
        (e.g., ja queues full of homophonic single-kanji clutter).
        Returns the number of rows removed.
        """
        if not user_id:
            raise ValueError("vocab_store.clear_lang requires user_id")
        cur = await self._conn.execute(
            "DELETE FROM vocab_state WHERE user_id = ? AND lang_code = ?",
            (user_id, lang_code),
        )
        await self._conn.commit()
        return int(cur.rowcount or 0)

    async def seed_words(
        self,
        *,
        user_id: str,
        lang_code: str,
        word_ids: list[str],
        source_surface: str = "seeded",
    ) -> int:
        """Bulk-add a starter set. Idempotent per word; seeded words get
        no initial exposure (the user hasn't actually met them in
        context yet) and are due **immediately** — seeding is an explicit
        "give me words to drill now" action, unlike a click-while-reading
        which lands tomorrow. Returns the count of newly-inserted rows."""
        if not user_id:
            raise ValueError("vocab_store.seed_words requires user_id")
        if not word_ids:
            return 0
        due = now_ts()  # immediately reviewable
        added = 0
        for wid in word_ids:
            cur = await self._conn.execute(
                """INSERT INTO vocab_state
                       (user_id, lang_code, word_id, fsrs_due_at,
                        mastery_state, source_surface)
                   VALUES (?, ?, ?, ?, 'new', ?)
                   ON CONFLICT(user_id, lang_code, word_id) DO NOTHING""",
                (user_id, lang_code, wid, due, source_surface),
            )
            added += bool(cur.rowcount)
        await self._conn.commit()
        return added

    async def bump_exposure(
        self,
        *,
        user_id: str,
        lang_code: str,
        word_id: str,
        output: bool = False,
    ) -> bool:
        """Increment an exposure counter for an already-tracked word.

        No-op (returns False) if the word isn't in the user's queue — we
        only count exposures for words the user has chosen to learn.
        """
        col = "exposure_output" if output else "exposure_input"
        cursor = await self._conn.execute(
            f"""UPDATE vocab_state SET {col} = {col} + 1
                WHERE user_id = ? AND lang_code = ? AND word_id = ?""",
            (user_id, lang_code, word_id),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    async def update_after_grade(
        self,
        *,
        user_id: str,
        lang_code: str,
        word_id: str,
        difficulty: float,
        stability: float,
        due_at: str,
        reps: int,
        lapses: int,
        grade: int,
        mastery_state: str,
    ) -> bool:
        """Persist the new FSRS state after a review.

        The caller computes the new state via ``learning.fsrs.schedule``
        and the new ``due_at`` via :func:`future_ts`. Returns False if
        the word isn't in the user's queue.
        """
        cursor = await self._conn.execute(
            """UPDATE vocab_state
                  SET fsrs_difficulty = ?, fsrs_stability = ?, fsrs_due_at = ?,
                      fsrs_reps = ?, fsrs_lapses = ?, fsrs_last_grade = ?,
                      mastery_state = ?, last_reviewed_at = ?
                WHERE user_id = ? AND lang_code = ? AND word_id = ?""",
            (
                difficulty, stability, due_at, reps, lapses, grade,
                mastery_state, now_ts(), user_id, lang_code, word_id,
            ),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    async def remove_word(
        self, *, user_id: str, lang_code: str, word_id: str
    ) -> bool:
        cursor = await self._conn.execute(
            """DELETE FROM vocab_state
               WHERE user_id = ? AND lang_code = ? AND word_id = ?""",
            (user_id, lang_code, word_id),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    # ── Reads ─────────────────────────────────────────────────────────

    async def get_word(
        self, *, user_id: str, lang_code: str, word_id: str
    ) -> dict | None:
        cursor = await self._conn.execute(
            """SELECT * FROM vocab_state
               WHERE user_id = ? AND lang_code = ? AND word_id = ?""",
            (user_id, lang_code, word_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def get_due(
        self,
        *,
        user_id: str,
        lang_code: str,
        limit: int = 20,
        now: str | None = None,
    ) -> list[dict]:
        """Cards whose ``fsrs_due_at`` has passed, soonest-due first."""
        cutoff = now or now_ts()
        limit = max(1, min(int(limit), 200))
        cursor = await self._conn.execute(
            """SELECT * FROM vocab_state
               WHERE user_id = ? AND lang_code = ? AND fsrs_due_at <= ?
               ORDER BY fsrs_due_at ASC
               LIMIT ?""",
            (user_id, lang_code, cutoff, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def count_due(
        self, *, user_id: str, lang_code: str, now: str | None = None
    ) -> int:
        cutoff = now or now_ts()
        cursor = await self._conn.execute(
            """SELECT COUNT(*) FROM vocab_state
               WHERE user_id = ? AND lang_code = ? AND fsrs_due_at <= ?""",
            (user_id, lang_code, cutoff),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def counts_by_mastery(
        self, *, user_id: str, lang_code: str
    ) -> dict[str, int]:
        cursor = await self._conn.execute(
            """SELECT mastery_state, COUNT(*) FROM vocab_state
               WHERE user_id = ? AND lang_code = ?
               GROUP BY mastery_state""",
            (user_id, lang_code),
        )
        rows = await cursor.fetchall()
        return {state: int(n) for state, n in rows}

    async def list_all(
        self, *, user_id: str, lang_code: str, limit: int = 1000
    ) -> list[dict]:
        limit = max(1, min(int(limit), 10000))
        cursor = await self._conn.execute(
            """SELECT * FROM vocab_state
               WHERE user_id = ? AND lang_code = ?
               ORDER BY first_seen_at DESC, word_id ASC
               LIMIT ?""",
            (user_id, lang_code, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]
