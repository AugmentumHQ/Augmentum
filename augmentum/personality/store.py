"""Personality facet store — Hebbian cooccurrence ops over the facet graph.

Mirrors `augmentum.memory.store.MemoryStore`'s cooccurrence pattern (per
migration 050 / `_record_cooccurrence` / `_apply_hebbian_boost` /
`decay_cooccurrence`). All operations user_id-scoped per CLAUDE.md
multi-tenancy invariant. Additionally companion_id-scoped because the
household model (commitment #7) allows multiple companions per user,
each with their own facet graph.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from augmentum.personality.models import (
    Facet,
    FacetActivation,
    FacetActivationSource,
    FacetCategory,
)
from augmentum.personality.vocabulary import SEED_FACETS
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)

# Mirror the threshold MemoryStore uses for hebbian boost (count >= 3).
# Below 3 the association is noise; at 3+ it has signal weight.
COOCCURRENCE_FLOOR = 3

# Weekly decay factor (mirrors memory_cooccurrence — multiply count by 0.99).
# Combined with the MAX(1, ...) floor in `decay_cooccurrence`, this means
# pairs decay toward count=1 but never below — `COOCCURRENCE_FLOOR` still
# gates retrieval, so count=1 pairs are present but inert.
DECAY_FACTOR = 0.99


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return facet pair in canonical alphabetical order so we never
    double-insert (warm-tender vs tender-warm). Matches the canonical_sort
    behavior of `MemoryStore._record_cooccurrence`."""
    return (a, b) if a <= b else (b, a)


class PersonalityStore:
    """Persistence for personality facets, activations, and Hebbian graphs.

    Scoped by `(user_id, companion_id)`. Multiple companions in the same
    household have independent facet graphs because they ARE different
    beings (commitments 3 + 7). All queries include both predicates.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._write_lock = asyncio.Lock()
        self._known_facets_cache: set[str] | None = None

    @property
    def _conn(self):
        return self._backend.conn

    async def _known_facets(self) -> set[str]:
        """Cached set of facet names from the vocabulary table.

        Invalidated on `seed_vocabulary()` writes. Reads against this are
        in the hot path (every `record_activations` call) so caching matters.

        Auto-seeds on first access if the table is empty — addresses the
        integration foot-gun where a caller instantiates the store and
        forgets to call `seed_vocabulary()`. The alternative (silently
        dropping every labeled facet) is a much worse failure mode.
        """
        if self._known_facets_cache is None:
            cursor = await self._conn.execute(
                "SELECT name FROM personality_facets"
            )
            rows = await cursor.fetchall()
            names = {row[0] for row in rows}
            if not names:
                log.info("personality.vocabulary_auto_seeded")
                await self.seed_vocabulary()
                cursor = await self._conn.execute(
                    "SELECT name FROM personality_facets"
                )
                rows = await cursor.fetchall()
                names = {row[0] for row in rows}
            self._known_facets_cache = names
        return self._known_facets_cache

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    async def seed_vocabulary(self) -> int:
        """Insert SEED_FACETS into personality_facets (idempotent).

        Returns the number of newly-inserted rows. Safe to call on every
        startup — existing facets are left alone via INSERT OR IGNORE.
        """
        async with self._write_lock:
            cursor = await self._conn.executemany(
                "INSERT OR IGNORE INTO personality_facets "
                "(name, description, category) VALUES (?, ?, ?)",
                [(f.name, f.description, f.category.value) for f in SEED_FACETS],
            )
            await self._conn.commit()
            inserted = cursor.rowcount or 0
        # Invalidate cache so next read picks up new rows.
        self._known_facets_cache = None
        log.info("personality.vocabulary_seeded", inserted=inserted, total=len(SEED_FACETS))
        return inserted

    async def get_facet(self, name: str) -> Facet | None:
        cursor = await self._conn.execute(
            "SELECT name, description, category, created_at "
            "FROM personality_facets WHERE name = ?",
            (name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Facet(
            name=row[0],
            description=row[1],
            category=FacetCategory(row[2]),
            created_at=row[3] or "",
        )

    async def list_facets(
        self,
        *,
        category: FacetCategory | None = None,
    ) -> list[Facet]:
        if category is not None:
            cursor = await self._conn.execute(
                "SELECT name, description, category, created_at "
                "FROM personality_facets WHERE category = ? ORDER BY name",
                (category.value,),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT name, description, category, created_at "
                "FROM personality_facets ORDER BY category, name"
            )
        rows = await cursor.fetchall()
        return [
            Facet(
                name=row[0],
                description=row[1],
                category=FacetCategory(row[2]),
                created_at=row[3] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Activations (per-turn audit log) + cooccurrence updates
    # ------------------------------------------------------------------

    async def record_activations(
        self,
        facets: list[tuple[str, float]],
        *,
        user_id: str,
        companion_id: str,
        session_id: str | None = None,
        turn_id: str | None = None,
        source: FacetActivationSource = FacetActivationSource.SELF_LABEL,
    ) -> list[int]:
        """Insert one activation row per active facet, then batch-update
        pairwise cooccurrence.

        Returns the list of inserted activation ids. Empty input returns
        empty list (no write). Empty user_id or companion_id raises
        ValueError — matches dream_journal convention (a missing tenant id
        is a bug, not a silent multi-tenant data merge).

        Unknown facet names (typos, hallucinations from the labeler) are
        dropped with a warning; the row IS still inserted for known facets
        in the same batch.
        """
        if not user_id:
            raise ValueError("personality_facet_activations write requires user_id")
        if not companion_id:
            raise ValueError("personality_facet_activations write requires companion_id")
        if not facets:
            return []

        known = await self._known_facets()
        valid = [(name, intensity) for name, intensity in facets if name in known]
        dropped = [name for name, _ in facets if name not in known]
        if dropped:
            log.warning(
                "personality.unknown_facets_dropped",
                user_id=user_id,
                companion_id=companion_id,
                dropped=dropped,
            )
        if not valid:
            return []

        async with self._write_lock:
            ids: list[int] = []
            for name, intensity in valid:
                cursor = await self._conn.execute(
                    "INSERT INTO personality_facet_activations "
                    "(user_id, companion_id, session_id, turn_id, "
                    "facet, intensity, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        companion_id,
                        session_id,
                        turn_id,
                        name,
                        intensity,
                        source.value,
                    ),
                )
                ids.append(cursor.lastrowid)

            # Batch cooccurrence updates for all pairs in this activation set.
            facet_names = [name for name, _ in valid]
            pairs = []
            for i in range(len(facet_names)):
                for j in range(i + 1, len(facet_names)):
                    a, b = _canonical_pair(facet_names[i], facet_names[j])
                    pairs.append((user_id, companion_id, a, b))

            if pairs:
                await self._conn.executemany(
                    "INSERT INTO personality_facet_cooccurrence "
                    "(user_id, companion_id, facet_a, facet_b) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(user_id, companion_id, facet_a, facet_b) "
                    "DO UPDATE SET count = count + 1, "
                    "last_updated = datetime('now')",
                    pairs,
                )

            await self._conn.commit()
            return ids

    async def query_recent_activations(
        self,
        *,
        user_id: str,
        companion_id: str,
        recent_hours: int = 24,
        limit: int = 100,
    ) -> list[FacetActivation]:
        """Return recent facet activations for the persona-kernel digester.

        Returns one row per activation event (NOT aggregated). Caller
        aggregates as needed for composition.
        """
        if not user_id or not companion_id:
            return []

        cursor = await self._conn.execute(
            "SELECT id, user_id, companion_id, session_id, turn_id, "
            "facet, intensity, source, activated_at "
            "FROM personality_facet_activations "
            "WHERE user_id = ? AND companion_id = ? "
            "AND activated_at >= datetime('now', ?) "
            "ORDER BY activated_at DESC LIMIT ?",
            (user_id, companion_id, f"-{recent_hours} hours", limit),
        )
        rows = await cursor.fetchall()
        return [
            FacetActivation(
                id=row[0],
                user_id=row[1],
                companion_id=row[2],
                session_id=row[3],
                turn_id=row[4],
                facet=row[5],
                intensity=row[6],
                source=FacetActivationSource(row[7]),
                activated_at=row[8],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Cooccurrence (facet × facet Hebbian graph)
    # ------------------------------------------------------------------

    async def query_cooccurrent_facets(
        self,
        facets: list[str],
        *,
        user_id: str,
        companion_id: str,
        limit: int = 10,
    ) -> list[tuple[str, int]]:
        """Given currently-active facets, return facets that historically
        co-fire with them, ranked by total cooccurrence count.

        Floor of `COOCCURRENCE_FLOOR` — below that the association is noise.
        Excludes the input facets from the returned list.
        """
        if not user_id or not companion_id or not facets:
            return []

        placeholders = ",".join("?" for _ in facets)
        cursor = await self._conn.execute(
            f"""
            SELECT other_facet, SUM(count) AS total_count FROM (
                SELECT facet_b AS other_facet, count
                FROM personality_facet_cooccurrence
                WHERE user_id = ? AND companion_id = ?
                  AND facet_a IN ({placeholders})
                  AND facet_b NOT IN ({placeholders})
                  AND count >= ?
                UNION ALL
                SELECT facet_a AS other_facet, count
                FROM personality_facet_cooccurrence
                WHERE user_id = ? AND companion_id = ?
                  AND facet_b IN ({placeholders})
                  AND facet_a NOT IN ({placeholders})
                  AND count >= ?
            )
            GROUP BY other_facet
            ORDER BY total_count DESC
            LIMIT ?
            """,
            (
                user_id, companion_id, *facets, *facets, COOCCURRENCE_FLOOR,
                user_id, companion_id, *facets, *facets, COOCCURRENCE_FLOOR,
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def decay_cooccurrence(
        self,
        *,
        user_id: str = "",
        companion_id: str = "",
        decay_factor: float = DECAY_FACTOR,
    ) -> int:
        """Apply weekly decay to cooccurrence counts. Mirrors
        `MemoryStore.decay_cooccurrence` (multiply, then prune rows with
        count <= 0). Decay is applied to BOTH the facet-cooccurrence and
        the memory-associations graphs so they stay aligned in time.

        Empty user_id decays globally (periodic maintenance). Both ids
        present scopes to that specific relationship.

        Returns total rows pruned (facet cooccurrence + memory associations).
        """
        async with self._write_lock:
            cooccur_where = ""
            cooccur_params: list = [decay_factor]
            assoc_where = ""
            assoc_params: list = [decay_factor]
            prune_cooccur_where = " WHERE count <= 0"
            prune_assoc_where = " WHERE count <= 0"
            prune_params: list = []

            if user_id:
                cooccur_where = " WHERE user_id = ?"
                cooccur_params.append(user_id)
                assoc_where = " WHERE user_id = ?"
                assoc_params.append(user_id)
                prune_cooccur_where += " AND user_id = ?"
                prune_assoc_where += " AND user_id = ?"
                prune_params.append(user_id)
                if companion_id:
                    cooccur_where += " AND companion_id = ?"
                    cooccur_params.append(companion_id)
                    assoc_where += " AND companion_id = ?"
                    assoc_params.append(companion_id)
                    prune_cooccur_where += " AND companion_id = ?"
                    prune_assoc_where += " AND companion_id = ?"
                    prune_params.append(companion_id)

            # Mirrors MemoryStore.decay_cooccurrence:961 — floor at 1 so the
            # association never decays out of existence. Without this floor,
            # any pair with count=1 immediately decays to 0 (CAST(0.99) = 0)
            # and is pruned, meaning pairs that fire infrequently can never
            # accumulate past the COOCCURRENCE_FLOOR signal threshold.
            await self._conn.execute(
                f"UPDATE personality_facet_cooccurrence "
                f"SET count = MAX(1, CAST(count * ? AS INTEGER)){cooccur_where}",
                cooccur_params,
            )
            cursor = await self._conn.execute(
                f"DELETE FROM personality_facet_cooccurrence{prune_cooccur_where}",
                prune_params,
            )
            cooccur_pruned = cursor.rowcount or 0

            await self._conn.execute(
                f"UPDATE personality_memory_associations "
                f"SET count = MAX(1, CAST(count * ? AS INTEGER)){assoc_where}",
                assoc_params,
            )
            cursor = await self._conn.execute(
                f"DELETE FROM personality_memory_associations{prune_assoc_where}",
                prune_params,
            )
            assoc_pruned = cursor.rowcount or 0

            await self._conn.commit()
            total_pruned = cooccur_pruned + assoc_pruned
            log.info(
                "personality.decay_applied",
                user_id=user_id,
                companion_id=companion_id,
                decay_factor=decay_factor,
                cooccur_pruned=cooccur_pruned,
                assoc_pruned=assoc_pruned,
            )
            return total_pruned

    # ------------------------------------------------------------------
    # Memory associations (cross-table — the elegant bit)
    # ------------------------------------------------------------------

    async def record_memory_associations(
        self,
        memory_ids: list[str],
        facets: list[str],
        *,
        user_id: str,
        companion_id: str,
    ) -> int:
        """Strengthen memory↔facet associations for every (memory, facet) pair.

        Called when a turn surfaces both certain memories AND certain
        facets — this is the cross-table Hebbian update that captures
        "you bring out a certain side of me when we talk about X" patterns.

        Returns the number of (memory, facet) pairs touched.
        """
        if not user_id:
            raise ValueError(
                "personality_memory_associations write requires user_id"
            )
        if not companion_id:
            raise ValueError(
                "personality_memory_associations write requires companion_id"
            )
        if not memory_ids or not facets:
            return 0

        known = await self._known_facets()
        valid_facets = [f for f in facets if f in known]
        if not valid_facets:
            return 0

        pairs = [
            (user_id, companion_id, memory_id, facet)
            for memory_id in memory_ids
            for facet in valid_facets
        ]

        async with self._write_lock:
            await self._conn.executemany(
                "INSERT INTO personality_memory_associations "
                "(user_id, companion_id, memory_id, facet) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, companion_id, memory_id, facet) "
                "DO UPDATE SET count = count + 1, "
                "last_updated = datetime('now')",
                pairs,
            )
            await self._conn.commit()
            return len(pairs)

    async def query_facets_for_memories(
        self,
        memory_ids: list[str],
        *,
        user_id: str,
        companion_id: str,
        limit: int = 8,
    ) -> list[tuple[str, int]]:
        """Given memory IDs retrieved this turn, return associated facets
        ranked by historical cooccurrence count. This is the cross-table
        query that pulls memory retrieval into personality composition.

        Floor of `COOCCURRENCE_FLOOR` applies.
        """
        if not user_id or not companion_id or not memory_ids:
            return []

        placeholders = ",".join("?" for _ in memory_ids)
        cursor = await self._conn.execute(
            f"""
            SELECT facet, SUM(count) AS total_count
            FROM personality_memory_associations
            WHERE user_id = ? AND companion_id = ?
              AND memory_id IN ({placeholders})
              AND count >= ?
            GROUP BY facet
            ORDER BY total_count DESC
            LIMIT ?
            """,
            (user_id, companion_id, *memory_ids, COOCCURRENCE_FLOOR, limit),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]
