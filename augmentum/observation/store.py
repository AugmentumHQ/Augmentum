"""SQLite-backed L0 observation store.

Per the substrate spec, the store is tokenizer-agnostic: text in,
text out. Re-tokenization into a model's vocab happens lazily at
cache-export time (``exporter.py``), not here. That decoupling is
what lets one user's observation history outlive any individual
model swap.

User-scoping is mandatory per ``CLAUDE.md``'s multi-tenant data-
isolation pattern — every method accepts ``*, user_id: str`` and
the underlying table carries a ``user_id`` column.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from augmentum.observation.fingerprint import fingerprint_prefix
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Observation:
    """One L0 row.

    Public shape that consumers read. The store accepts (prefix_text,
    continuation, surface, mode) at ingest time and computes
    ``fingerprint`` internally — callers never construct
    ``Observation`` directly when writing.
    """

    user_id: str
    surface: str
    mode: str
    fingerprint: str
    prefix_text: str
    continuation: str
    observation_count: int
    last_seen_ts: int
    decay_weight: float


class ObservationStore:
    """L0 CRUD + top-K query.

    The store is intentionally thin — every consumer (cache exporter,
    future autocomplete, future companion gate) reads via ``top_k`` or
    ``slice`` and applies its own ranking on top. Centralizing the
    ranking here would couple the store to a single consumer's needs;
    keeping it dumb keeps the substrate-paying-back property real.
    """

    def __init__(self, conn: Any) -> None:
        """``conn`` is an ``aiosqlite.Connection`` open against the
        main Augmentum DB. The store doesn't manage the connection
        lifecycle — the caller is responsible (matches every other
        store in this codebase).
        """
        self._conn = conn

    # ── Ingest ────────────────────────────────────────────────────────

    async def observe(
        self,
        *,
        user_id: str,
        prefix_text: str,
        continuation: str,
        surface: str = "chat",
        mode: str = "",
        weight: float = 1.0,
    ) -> None:
        """Record (or bump) one observation row.

        Empty user_id is silently dropped — every L0 row must be
        user-scoped; an anonymous observation would pollute the
        cross-tenant index. Empty prefix or continuation are also
        dropped (no signal). Callers don't have to guard upstream.
        """
        if not user_id or not prefix_text or not continuation:
            return

        fp = fingerprint_prefix(prefix_text, surface=surface, mode=mode)
        now = int(time.time())

        # Upsert — bump observation_count on conflict, refresh last_seen,
        # ride the decay_weight at 1.0 on fresh writes (decay is applied
        # at query time, not write time).
        await self._conn.execute(
            """
            INSERT INTO bom_observations_exact (
                user_id, surface, mode, fingerprint, prefix_text,
                continuation, observation_count, last_seen_ts, decay_weight
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(user_id, fingerprint, continuation) DO UPDATE SET
                observation_count = observation_count + 1,
                last_seen_ts = excluded.last_seen_ts,
                decay_weight = MIN(1.0, decay_weight + 0.05)
            """,
            (user_id, surface, mode, fp, prefix_text, continuation,
             now, max(0.0, min(weight, 1.0))),
        )

    async def observe_many(
        self,
        rows: Iterable[tuple[str, str]],
        *,
        user_id: str,
        surface: str = "chat",
        mode: str = "",
    ) -> int:
        """Batch ingest. ``rows`` is an iterable of (prefix_text,
        continuation) pairs. Returns the count actually written.

        Used by the seeder which produces thousands of pairs at a time;
        the per-call ``observe`` path would be too chatty against
        aiosqlite's per-statement overhead.
        """
        if not user_id:
            return 0
        written = 0
        for prefix_text, continuation in rows:
            if not prefix_text or not continuation:
                continue
            await self.observe(
                user_id=user_id,
                prefix_text=prefix_text,
                continuation=continuation,
                surface=surface,
                mode=mode,
            )
            written += 1
        return written

    # ── Query ─────────────────────────────────────────────────────────

    async def top_k(
        self,
        *,
        user_id: str,
        k: int = 50_000,
        surface: str | None = None,
        mode: str | None = None,
    ) -> list[Observation]:
        """Highest-ranked observations for this user.

        Ranking: ``observation_count * decay_weight`` desc, tie-broken
        by ``last_seen_ts`` desc. Both factors live on the row so the
        sort happens server-side.

        Optional ``surface`` / ``mode`` filters narrow the result set
        for future per-surface consumers; the cache exporter passes
        neither (it wants the whole user-scoped picture).
        """
        if not user_id:
            return []

        where = ["user_id = ?"]
        params: list[Any] = [user_id]
        if surface is not None:
            where.append("surface = ?")
            params.append(surface)
        if mode is not None:
            where.append("mode = ?")
            params.append(mode)
        where_sql = " AND ".join(where)
        params.append(int(max(1, k)))

        cursor = await self._conn.execute(
            f"""
            SELECT user_id, surface, mode, fingerprint, prefix_text,
                   continuation, observation_count, last_seen_ts, decay_weight
            FROM bom_observations_exact
            WHERE {where_sql}
            ORDER BY (observation_count * decay_weight) DESC, last_seen_ts DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        return [Observation(*row) for row in rows]

    async def count(self, *, user_id: str) -> int:
        """Total L0 rows for one user — used by the rebuild endpoint to
        report state to the operator."""
        if not user_id:
            return 0
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM bom_observations_exact WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Autocomplete ──────────────────────────────────────────────────

    async def complete(
        self,
        *,
        user_id: str,
        current_text: str,
        surface: str = "chat",
        mode: str = "",
        k: int = 5,
        tail_lengths: tuple[int, ...] = (8, 5, 3),
    ) -> tuple[str, list[tuple[str, int]]]:
        """Return ranked continuations for the tail of ``current_text``.

        Strategy: take the last N words of ``current_text`` for each N
        in ``tail_lengths`` (longest first), fingerprint that tail
        against (surface, mode), and look it up in the store. The first
        tail length that returns hits wins — longer tails are more
        context-distinctive and produce higher-quality continuations.

        Returns ``(matched_prefix, [(continuation, count), ...])``. When
        no tail length yields a hit, returns ``("", [])`` so the caller
        can silent-skip without disambiguating "no matches" from "user
        not authenticated."
        """
        if not user_id or not current_text:
            return ("", [])

        # Split into words; the autocomplete consumer cares about the
        # tail of what the user has actually typed.
        words = current_text.split()
        if not words:
            return ("", [])

        for tail_n in tail_lengths:
            if tail_n <= 0 or tail_n > len(words):
                continue
            tail = " ".join(words[-tail_n:])
            fp = fingerprint_prefix(tail, surface=surface, mode=mode)
            cursor = await self._conn.execute(
                """
                SELECT continuation, observation_count, decay_weight
                FROM bom_observations_exact
                WHERE user_id = ? AND fingerprint = ?
                ORDER BY (observation_count * decay_weight) DESC, last_seen_ts DESC
                LIMIT ?
                """,
                (user_id, fp, int(max(1, k))),
            )
            rows = await cursor.fetchall()
            if rows:
                # Filter the immediate-dup case ("the the"): if the
                # continuation's first word equals the user's last
                # typed word, drop. We don't filter "word appears
                # anywhere in current text" because that's too
                # aggressive — a continuation can legitimately repeat
                # a word the user used in a different context earlier.
                last_word = words[-1].lower().rstrip(".,!?;:")
                hits: list[tuple[str, int]] = []
                for cont, count, _weight in rows:
                    if not cont:
                        continue
                    first_cont_word = cont.strip().split(" ", 1)[0].lower()
                    if first_cont_word == last_word:
                        continue
                    hits.append((cont, int(count)))
                if hits:
                    return (tail, hits)

        return ("", [])

    # ── Admin ─────────────────────────────────────────────────────────

    async def purge(self, *, user_id: str) -> int:
        """Drop every observation for this user. Returns rows deleted.

        Hooked by the inspect/edit surface (future) and by
        delete_user's cascade — though the FK ON DELETE CASCADE means
        the user-delete path doesn't need this explicitly.
        """
        if not user_id:
            return 0
        cursor = await self._conn.execute(
            "DELETE FROM bom_observations_exact WHERE user_id = ?",
            (user_id,),
        )
        return cursor.rowcount or 0
