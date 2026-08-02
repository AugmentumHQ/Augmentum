"""Lesson registry — the learn-from-correction substrate.

The capability-side accumulation thesis (``2026-05-23-accumulation-
thesis.md``) names a "failure-mode registry — known traps" as a sibling
to the skill graph, and the companion directory README promises an
``anti_patterns.db``. Neither was ever wired. This is it, in the small.

Where :mod:`augmentum.companion.skills` accumulates what *worked* (an
approach indexed by problem shape, confidence that rises with success),
this module accumulates what she was *corrected on*: a lesson indexed by
the **situation** it applies to, with a **strength** that rises each time
the same correction recurs or she successfully avoids the trap.

A lesson is the inverse of a skill:

    skill   : "when situation ~ X, approach Y is good"   (recall)
    lesson  : "when situation ~ X, the trap is Y — do Z" (guardrail)

**The felt benefit.** A correction stops being a one-turn event. Once
captured, the lesson is retrieved at compose time (by situation
similarity) and injected as a guardrail, so the same mistake doesn't
recur — across sessions and across modalities, because chat and voice
compose through the same prompt.

**Discipline.** Lessons shape *how* she responds within a turn. They do
NOT touch the personality doc, the persona kernel, or the genesis anchor
— so the thesis's "no identity mutation without consent" commitment is
untouched. Lessons are per-user by default (a correction from one user
isn't a truth about another); ``user_id IS NULL`` is reserved for
genuinely cross-user lessons (future household work).

**Containment.** This module is pure substrate: capture, query,
reinforce, list. The capture *judgment* (what counts as a correction,
what the durable lesson is) lives one layer up in
:mod:`augmentum.companion_runtime.lessons_capture`.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Value objects ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Lesson:
    """A held lesson — a correction indexed by the situation it applies to.

    ``strength`` ∈ [0, 1]. New lessons start at 0.5. Rises when the same
    correction recurs (``reinforce(seen=True)``) or when she successfully
    avoids the trap (``reinforce(applied=True)``); a lesson never has its
    strength lowered automatically — a correction the user made is real
    until they retract it (``retire``).
    """
    id: int
    companion_id: str
    user_id: str | None
    situation: str
    trap: str
    better: str
    strength: float
    times_seen: int
    times_applied: int
    source: str
    evidence: str
    status: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "companion_id": self.companion_id,
            "user_id": self.user_id,
            "situation": self.situation,
            "trap": self.trap,
            "better": self.better,
            "strength": self.strength,
            "times_seen": self.times_seen,
            "times_applied": self.times_applied,
            "source": self.source,
            "evidence": self.evidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class RelevantLesson:
    """A lesson plus its relevance score for the current intent.

    ``relevance`` is cosine similarity (in [0, 1]) of the intent
    embedding against the lesson's situation embedding.
    ``effective_score`` weights relevance by strength so weakly-held
    lessons don't outrank firmly-held ones.
    """
    lesson: Lesson
    relevance: float
    effective_score: float


# ── Embedding helpers (same binary form as the skill graph) ───────────


def _encode_embedding(emb: list[float] | None) -> bytes | None:
    if not emb:
        return None
    return struct.pack(f"<{len(emb)}f", *emb)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    if len(blob) % 4 != 0:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine *similarity* in [0, 1] for non-zero vectors; 0.0 on
    degenerate inputs."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


async def _embed(text: str) -> list[float] | None:
    """Embed ``text`` off the event loop (the embed call is synchronous
    and may cold-load a model). Returns None on failure — retrieval
    degrades to "no relevant lessons" rather than blocking."""
    if not text.strip():
        return None
    try:
        from augmentum.memory.embeddings import EmbeddingService
        return await asyncio.to_thread(EmbeddingService.embed_one, text)
    except Exception:
        log.debug("lesson_embed_failed", exc_info=True)
        return None


# ── LessonGraph — the storage/query/curation interface ────────────────


class LessonGraph:
    """The lesson registry.

    Use when:
    - A correction is observed and should be held as a lesson
      (``capture``).
    - The composer wants lessons relevant to the current intent
      (``query_relevant``).
    - A held lesson recurred or was honored (``reinforce``).
    - The user retracts a correction (``retire``).

    Lifecycle mirrors :class:`~augmentum.companion.skills.SkillGraph`:
    instantiate with the shared backend + (optional) bus, then call
    async methods. Writes use the shared state-layer connection.
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        *,
        bus: PresenceBus | None = None,
        companion_id: str = "becca",
    ) -> None:
        self._backend = backend
        self._bus = bus
        self._companion_id = companion_id

    # ── Capture ───────────────────────────────────────────────────────

    async def capture(
        self,
        *,
        situation: str,
        trap: str,
        better: str,
        user_id: str | None = None,
        source: str = "reflection",
        evidence: str = "",
        strength: float = 0.5,
    ) -> Lesson:
        """Capture a correction as a held lesson.

        Deduplicated on a near-identical ``(situation, trap)`` for the
        same user: a re-capture of an existing lesson **reinforces** it
        (strength up, ``times_seen``++) rather than inserting a
        duplicate — recurrence is the strongest signal that a lesson is
        real, so it should make the lesson firmer, not noisier.

        The embedding is computed from ``situation`` (that's what
        retrieval matches the intent against). Embedding failure is
        non-fatal — the row stores NULL and simply won't surface via
        similarity until re-captured.
        """
        situation = (situation or "").strip()
        better = (better or "").strip()
        if not situation or not better:
            raise ValueError("LessonGraph.capture requires situation and better")
        trap = (trap or "").strip()

        existing = await self._fetch_by_match(situation, trap, user_id=user_id)
        if existing is not None:
            await self.reinforce(existing.id, seen=True)
            updated = await self._fetch_by_id(existing.id)
            assert updated is not None
            return updated

        emb_blob = _encode_embedding(await _embed(situation))
        cursor = await self._backend.conn.execute(
            "INSERT INTO companion_lessons "
            "(companion_id, user_id, situation, trap, better, embedding, "
            " strength, source, evidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._companion_id, user_id, situation, trap, better,
                emb_blob, _clamp01(strength), source, evidence[:300],
            ),
        )
        await self._backend.conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        if not row_id:
            raise RuntimeError("lesson_capture_failed: no row_id")
        lesson = await self._fetch_by_id(int(row_id))
        assert lesson is not None

        if self._bus is not None:
            try:
                await self._bus.publish_topic(
                    "lesson.captured",
                    {
                        "lesson_id": lesson.id,
                        "situation": lesson.situation,
                        "user_id": lesson.user_id,
                    },
                    source_companion_id=self._companion_id,
                )
            except Exception:
                log.debug("lesson_capture_bus_emit_failed", exc_info=True)

        log.info(
            "lesson_captured",
            companion_id=self._companion_id, user_id=user_id,
            lesson_id=lesson.id, source=source,
        )
        return lesson

    # ── Reinforcement ─────────────────────────────────────────────────

    async def reinforce(
        self, lesson_id: int, *, seen: bool = False, applied: bool = False,
    ) -> Lesson | None:
        """Strengthen a held lesson.

        ``seen`` — the same correction recurred (the user had to say it
        again). ``applied`` — she successfully avoided the trap. Both
        raise strength via a bounded EWMA toward 1.0 (rate 0.15: ~10
        events to fully firm up) and bump the matching counter.
        """
        current = await self._fetch_by_id(lesson_id)
        if current is None:
            return None
        learning_rate = 0.15
        new_strength = _clamp01(
            current.strength + learning_rate * (1.0 - current.strength),
        )
        seen_delta = 1 if seen else 0
        applied_delta = 1 if applied else 0
        await self._backend.conn.execute(
            "UPDATE companion_lessons "
            "SET strength = ?, "
            "    times_seen = times_seen + ?, "
            "    times_applied = times_applied + ?, "
            "    updated_at = datetime('now') "
            "WHERE id = ?",
            (new_strength, seen_delta, applied_delta, lesson_id),
        )
        await self._backend.conn.commit()
        return await self._fetch_by_id(lesson_id)

    async def retire(self, lesson_id: int) -> bool:
        """Retire a lesson (the user retracted the correction). Keeps the
        row for the audit trail; it just stops being retrieved."""
        await self._backend.conn.execute(
            "UPDATE companion_lessons SET status = 'retired', "
            "updated_at = datetime('now') WHERE id = ?",
            (lesson_id,),
        )
        await self._backend.conn.commit()
        return True

    # ── Retrieval ─────────────────────────────────────────────────────

    async def query_relevant(
        self,
        intent_text: str,
        *,
        user_id: str | None = None,
        top_k: int = 3,
        min_relevance: float = 0.6,
        min_strength: float = 0.5,
    ) -> list[RelevantLesson]:
        """Return top-K lessons relevant to ``intent_text``.

        Cosine similarity of the intent against each lesson's situation
        embedding, filtered to active lessons at or above
        ``min_strength``. Returns an empty list when nothing fits — she
        doesn't carry a guardrail into a situation it wasn't about.
        """
        if not intent_text.strip():
            return []
        intent_vec = await _embed(intent_text)
        if intent_vec is None:
            return []

        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND status = 'active' "
                "  AND strength >= ? "
                "ORDER BY updated_at DESC LIMIT 200",
                (self._companion_id, min_strength),
            )
        else:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND status = 'active' "
                "  AND (user_id = ? OR user_id IS NULL) "
                "  AND strength >= ? "
                "ORDER BY updated_at DESC LIMIT 200",
                (self._companion_id, user_id, min_strength),
            )
        rows = await cur.fetchall()
        await cur.close()

        scored: list[RelevantLesson] = []
        for r in rows:
            lesson = await self._fetch_by_id(int(r[0]))
            if lesson is None:
                continue
            cur = await self._backend.conn.execute(
                "SELECT embedding FROM companion_lessons WHERE id = ?",
                (lesson.id,),
            )
            embrow = await cur.fetchone()
            await cur.close()
            emb = _decode_embedding(embrow[0] if embrow else None)
            if emb is None:
                continue
            relevance = _cosine(intent_vec, emb)
            if relevance < min_relevance:
                continue
            scored.append(RelevantLesson(
                lesson=lesson,
                relevance=relevance,
                effective_score=relevance * lesson.strength,
            ))

        scored.sort(key=lambda r: -r.effective_score)
        return scored[:top_k]

    # ── Listing (Observatory + tests) ─────────────────────────────────

    async def list_lessons(
        self,
        *,
        user_id: str | None = None,
        status: str = "active",
        limit: int = 100,
    ) -> list[Lesson]:
        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND status = ? "
                "ORDER BY strength DESC, updated_at DESC LIMIT ?",
                (self._companion_id, status, limit),
            )
        else:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND status = ? "
                "  AND (user_id = ? OR user_id IS NULL) "
                "ORDER BY strength DESC, updated_at DESC LIMIT ?",
                (self._companion_id, status, user_id, limit),
            )
        rows = await cur.fetchall()
        await cur.close()
        out: list[Lesson] = []
        for r in rows:
            lesson = await self._fetch_by_id(int(r[0]))
            if lesson is not None:
                out.append(lesson)
        return out

    # ── Internal fetch helpers ────────────────────────────────────────

    async def _fetch_by_id(self, lesson_id: int) -> Lesson | None:
        cur = await self._backend.conn.execute(
            "SELECT id, companion_id, user_id, situation, trap, better, "
            "       strength, times_seen, times_applied, source, evidence, "
            "       status, created_at, updated_at "
            "FROM companion_lessons WHERE id = ?",
            (lesson_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        return Lesson(
            id=int(row[0]),
            companion_id=str(row[1]),
            user_id=row[2],
            situation=str(row[3] or ""),
            trap=str(row[4] or ""),
            better=str(row[5] or ""),
            strength=float(row[6]),
            times_seen=int(row[7]),
            times_applied=int(row[8]),
            source=str(row[9] or ""),
            evidence=str(row[10] or ""),
            status=str(row[11]),
            created_at=str(row[12] or ""),
            updated_at=str(row[13] or ""),
        )

    async def _fetch_by_match(
        self, situation: str, trap: str, *, user_id: str | None = None,
    ) -> Lesson | None:
        """Find an existing lesson with the same situation + trap for this
        user. Case-insensitive exact match — the cheap dedup that catches
        the common "same correction, same words" recurrence. Embedding-
        nearest dedup is a future refinement."""
        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND user_id IS NULL "
                "  AND status = 'active' "
                "  AND LOWER(situation) = LOWER(?) AND LOWER(trap) = LOWER(?) "
                "LIMIT 1",
                (self._companion_id, situation, trap),
            )
        else:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_lessons "
                "WHERE companion_id = ? AND user_id = ? "
                "  AND status = 'active' "
                "  AND LOWER(situation) = LOWER(?) AND LOWER(trap) = LOWER(?) "
                "LIMIT 1",
                (self._companion_id, user_id, situation, trap),
            )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        return await self._fetch_by_id(int(row[0]))


__all__ = [
    "Lesson",
    "RelevantLesson",
    "LessonGraph",
]
