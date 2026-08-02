"""The Evidence Bus — convergent corroboration for Earned Understanding (P2).

The unifying intake for "everything that can become a memory." A chat
statement, a "Jazz" playlist, a bookmark-with-note, a re-visited article all
arrive here as *evidence*, never as fact — cheap, abundant, and invisible
(never injected, never recited).

A belief earns durability by **convergence, not repetition**: independent
sources agreeing is what counts. Saying "I like jazz" twice in one breath is
one channel agreeing with itself (weak); saying it + owning a jazz playlist +
twelve jazz visits is three independent channels converging (strong). This is
how a person comes to trust something — they believe it more when it shows up
from unrelated directions.

Two surfaces:
  * ``convergence_score`` — a continuous, independence-weighted strength for
    the Mirror/UI and future tuning. Per-source trust weight, with geometric
    novelty decay WITHIN a source so a 4th same-channel hit adds little.
  * ``corroborate_belief`` — the ladder bridge: records evidence and, when a
    NEW independent source appears, advances the existing promotion ladder by
    one access step (reusing ``MemoryStore._maybe_promote`` rather than a
    parallel path). Same-source repeats are recorded but never advance it.

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore

log = get_logger(__name__)


# Per-source trust weight: deliberate channels (the user chose to act) outrank
# behavioral ones, which outrank ambient ones. A source not listed gets the
# neutral default — unknown channels are trusted moderately, never maximally.
SOURCE_TRUST: dict[str, float] = {
    # Deliberate — the user said it on purpose
    "chat_explicit": 1.0,   # "remember that …"
    "memory_save": 1.0,     # the memory.save verb
    "playlist": 0.85,       # named a playlist
    "bookmark": 0.8,        # marked a spot, maybe with a note
    "browse_note": 0.8,     # wrote a note about it
    # Behavioral — the user did it, didn't declare it
    "chat": 0.6,            # passively extracted from conversation
    "media_play": 0.55,     # played it
    "reading": 0.55,        # read it
    "game": 0.5,            # played a learning game in this area
    # Ambient — a faint signal
    "browse": 0.4,          # visited a page
    "view": 0.35,           # opened/looked at
}
_DEFAULT_TRUST = 0.5

# Geometric decay for repeated evidence from the SAME source. The n-th hit from
# one channel contributes trust * NOVELTY_DECAY**(n-1): the first hit is full
# weight, the rest taper. This is what makes independence dominate repetition.
NOVELTY_DECAY = 0.5


def source_trust(source: str) -> float:
    return SOURCE_TRUST.get((source or "").strip().lower(), _DEFAULT_TRUST)


def convergence_score(source_counts: dict[str, int]) -> float:
    """Independence-weighted corroboration strength from a per-source tally.

    ``score = Σ_sources  trust(source) · (1 - decay**count)/(1 - decay)``

    A single source saturates toward ``trust/(1-decay)`` no matter how many
    times it repeats; new *independent* sources each add their full trust on
    first hit. Triangulation across channels dominates repetition within one.
    """
    total = 0.0
    for source, count in (source_counts or {}).items():
        n = max(0, int(count))
        if n == 0:
            continue
        geometric = (1.0 - NOVELTY_DECAY ** n) / (1.0 - NOVELTY_DECAY)
        total += source_trust(source) * geometric
    return round(total, 4)


@dataclass(slots=True)
class CorroborationResult:
    """Outcome of a single corroborate_belief call."""
    recorded: bool
    new_source: bool                 # did this add a channel not seen before for this belief?
    distinct_sources: int            # how many independent channels now corroborate it
    score: float                     # independence-weighted convergence strength
    promoted_checked: bool = False   # did we run the ladder (only on a new source)?
    evidence_id: str = ""
    sources: dict[str, int] = field(default_factory=dict)


class EvidenceStore:
    """Records evidence and computes convergence over a shared SQLite backend.

    Decoupled from MemoryStore: it owns only the ``evidence`` table. The one
    place it touches a belief — ``corroborate_belief`` — does so through the
    passed MemoryStore's public surface (``bump_access`` + ``_maybe_promote``),
    so the promotion logic stays single-sourced in MemoryStore.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @property
    def _conn(self):
        return self._backend.conn

    async def record(
        self,
        *,
        user_id: str,
        source: str,
        claim: str = "",
        subject: str = "",
        memory_id: str | None = None,
        weight: float = 1.0,
        companion_id: str | None = None,
    ) -> str:
        """Insert one evidence row. Returns its id."""
        if not user_id or not source:
            raise ValueError("evidence requires user_id and source")
        eid = uuid.uuid4().hex
        await self._conn.execute(
            "INSERT INTO evidence "
            "(id, user_id, companion_id, memory_id, subject, claim, source, weight) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, user_id, companion_id, memory_id, subject or "",
             claim or "", source.strip().lower(), float(weight)),
        )
        await self._backend.conn.commit()
        return eid

    async def source_counts(self, *, user_id: str, memory_id: str) -> dict[str, int]:
        """Per-source evidence tally for one belief."""
        cur = await self._conn.execute(
            "SELECT source, COUNT(*) FROM evidence "
            "WHERE user_id = ? AND memory_id = ? GROUP BY source",
            (user_id, memory_id),
        )
        return {row[0]: row[1] for row in await cur.fetchall()}

    async def distinct_sources(self, *, user_id: str, memory_id: str) -> int:
        cur = await self._conn.execute(
            "SELECT COUNT(DISTINCT source) FROM evidence "
            "WHERE user_id = ? AND memory_id = ?",
            (user_id, memory_id),
        )
        return int((await cur.fetchone())[0] or 0)

    async def score_for(self, *, user_id: str, memory_id: str) -> float:
        return convergence_score(await self.source_counts(user_id=user_id, memory_id=memory_id))

    async def evidence_for(
        self, *, user_id: str, memory_id: str, limit: int = 50,
    ) -> list[dict]:
        """The Mirror's trail: why she believes this, newest first."""
        cur = await self._conn.execute(
            "SELECT id, source, claim, subject, weight, created_at FROM evidence "
            "WHERE user_id = ? AND memory_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, memory_id, limit),
        )
        return [
            {"id": r[0], "source": r[1], "claim": r[2], "subject": r[3],
             "weight": r[4], "created_at": r[5]}
            for r in await cur.fetchall()
        ]

    async def corroborate_belief(
        self,
        memory_store: MemoryStore,
        *,
        user_id: str,
        memory_id: str,
        source: str,
        claim: str = "",
        subject: str = "",
        weight: float = 1.0,
        companion_id: str | None = None,
    ) -> CorroborationResult:
        """Record evidence for a belief and, if it's a NEW independent source,
        advance the promotion ladder by one step.

        The independence rule lives here: a channel that already corroborates
        this belief records the evidence (it still feeds the score + trail) but
        does NOT bump the ladder — only triangulation across distinct sources
        earns durability. Promotion itself stays in MemoryStore._maybe_promote.
        """
        src = (source or "").strip().lower()
        was_new = src not in await self.source_counts(user_id=user_id, memory_id=memory_id)

        eid = await self.record(
            user_id=user_id, source=src, claim=claim, subject=subject,
            memory_id=memory_id, weight=weight, companion_id=companion_id,
        )

        promoted_checked = False
        if was_new:
            # One new independent channel → one step up the existing ladder.
            bumped = await memory_store.bump_access(memory_id, user_id=user_id)
            if bumped:
                await memory_store._maybe_promote(memory_id, user_id=user_id)  # noqa: SLF001
                promoted_checked = True

        counts = await self.source_counts(user_id=user_id, memory_id=memory_id)
        result = CorroborationResult(
            recorded=True,
            new_source=was_new,
            distinct_sources=len(counts),
            score=convergence_score(counts),
            promoted_checked=promoted_checked,
            evidence_id=eid,
            sources=counts,
        )
        log.debug(
            "evidence_corroborated",
            memory_id=memory_id, source=src, new_source=was_new,
            distinct=result.distinct_sources, score=result.score,
        )
        return result
