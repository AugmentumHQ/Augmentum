"""Skill graph + outcome ledger — capability accumulation substrate.

Accumulation thesis Step 3:
``docs/superpowers/specs/2026-05-23-accumulation-thesis.md``.

Identity-side accumulation (kernel digest, exemplar library, behavior
contract) makes her recognizably herself across years. This module is
the capability-side: the substrate that makes her *measurably better*
at the things she does with this user, across years. Same shape as
identity accumulation — accumulation, curation, abstraction — applied
to a different axis.

**The three primitives:**

- **Skill** — a named approach to a recurring problem shape. Each
  carries her own description (in her voice), the problem_shape it
  addresses, a confidence score, and accumulated success/failure
  counts. Skills are what get retrieved at compose time so prior
  approaches inform current responses.

- **Instance** — every time a skill was applied to a specific
  situation. Carries the context + the approach she took + refs back
  to the originating turn. The instance is the evidence trail.

- **Outcome** — what happened after. Signal in [-1, +1]; many will
  start as 'unknown' until inferred from user response or later
  observation. Outcomes are what move confidence over time.

**Discipline.** Skills accumulate honest evidence. The default
outcome is 'unknown' — not 'success'. Confidence only rises when
something actually worked, and falls when corrected. This is the
structural commitment against confabulated capability.

**Containment.** Skills are companion-scoped + (usually) user-scoped.
A skill that worked for one user doesn't automatically apply to
another — the relationship-specificity that makes accumulation
meaningful would be undermined by cross-user generalization. The
``user_id IS NULL`` form exists for genuinely cross-user skills
(future household work) but the default is per-user.

**This module is the substrate, not the consolidator.** It exposes
CRUD + relevance queries. The abstraction loop (the "across these 12
instances the actual pattern is Y" step) lives in a future
:mod:`companion.consolidation_skills` module that reads from here
and proposes new skill nodes the user approves.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.state.backends.sqlite import transactional_write
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Value objects ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Skill:
    """A skill node — a named approach indexed by problem shape.

    ``confidence`` ∈ [0, 1]. New skills start at 0.5 (the "we'll see"
    default). Rises with successes; falls with failures + corrections.

    ``status`` lifecycle: ``active`` (eligible for retrieval and
    application) → ``suppressed`` (user said don't use this) →
    ``retired`` (consolidated into a more general skill, or the
    relationship moved past it).
    """
    id: int
    companion_id: str
    user_id: str | None
    name: str
    description: str
    problem_shape: str
    confidence: float
    instances_count: int
    successes_count: int
    failures_count: int
    status: str
    abstracted_from_ids: list[int]
    created_at: str
    updated_at: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "companion_id": self.companion_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "problem_shape": self.problem_shape,
            "confidence": self.confidence,
            "instances_count": self.instances_count,
            "successes_count": self.successes_count,
            "failures_count": self.failures_count,
            "status": self.status,
            "abstracted_from_ids": list(self.abstracted_from_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SkillInstance:
    """One application of a skill to a specific situation."""
    id: int
    skill_id: int
    companion_id: str
    user_id: str | None
    context: str
    approach: str
    session_id: str
    invocation_id: str
    turn_ref: dict | None
    created_at: str


@dataclass(frozen=True, slots=True)
class SkillOutcome:
    """The signal an instance produced."""
    id: int
    instance_id: int
    outcome: str            # accepted|rejected|corrected|shipped|problematic|unknown
    signal: float           # in [-1, +1]
    evidence: str
    detected_at: str
    detected_by: str        # user_explicit|inferred|autonomous_check


@dataclass(frozen=True, slots=True)
class RelevantSkill:
    """A skill plus its relevance score for the current intent.

    Returned from :meth:`SkillGraph.query_relevant`. The
    ``relevance`` is cosine similarity (in [0, 1]) of the intent
    embedding against the skill's problem_shape embedding.
    ``effective_score`` weights relevance by confidence so untested
    skills don't outrank well-grounded ones.
    """
    skill: Skill
    relevance: float
    effective_score: float


# Valid outcome enumeration. Strings (not Enums) for SQLite storage
# friendliness; module-level constants keep the values searchable.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_CORRECTED = "corrected"
OUTCOME_SHIPPED = "shipped"
OUTCOME_PROBLEMATIC = "problematic"
OUTCOME_UNKNOWN = "unknown"

# Default outcome → signal mapping. The signal is what rolls into the
# skill's confidence over time; the mapping picks reasonable defaults.
# Callers can pass an explicit signal to override (useful for
# inferred outcomes with degraded confidence).
_OUTCOME_SIGNAL_DEFAULT: dict[str, float] = {
    OUTCOME_ACCEPTED: +0.6,
    OUTCOME_REJECTED: -0.8,
    OUTCOME_CORRECTED: -0.5,
    OUTCOME_SHIPPED: +1.0,
    OUTCOME_PROBLEMATIC: -1.0,
    OUTCOME_UNKNOWN: 0.0,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _encode_embedding(emb: list[float] | None) -> bytes | None:
    """Pack a float vector into the same binary form
    :mod:`companion_runtime.identity` uses for the kernel embedding."""
    if not emb:
        return None
    import struct
    return struct.pack(f"<{len(emb)}f", *emb)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    import struct
    if len(blob) % 4 != 0:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1] when both vectors are non-zero.
    Cosine *similarity*, not distance — callers want higher = more
    relevant. Returns 0.0 on degenerate inputs."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _clamp_confidence(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ── SkillGraph — the storage/query/curation interface ────────────────


class SkillGraph:
    """The skill graph + outcome ledger.

    Use when:
    - The companion needs to register a new approach as a skill.
    - A turn applied a skill and we want to record the instance.
    - An outcome lands and confidence needs to update.
    - The composer wants relevant skills for the current intent.

    Lifecycle: instantiate with a backend (the SQLiteBackend the rest
    of the companion uses) and a bus (for emit). Methods are
    async-safe — underlying writes use the same connection lock the
    state layer uses.
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

    # ── Skill CRUD ────────────────────────────────────────────────────

    async def register_skill(
        self,
        name: str,
        *,
        description: str,
        problem_shape: str,
        user_id: str | None = None,
        confidence: float = 0.5,
    ) -> Skill:
        """Register a new skill node. Idempotent on
        ``(companion_id, user_id, name)`` — a second registration with
        the same name updates the existing row's description +
        problem_shape rather than creating a duplicate.

        Embedding is computed from ``problem_shape``. Embedding failures
        are silent (the row stores ``NULL``); retrieval still works via
        name + description fall-backs, just without similarity ranking.
        """
        from augmentum.memory.embeddings import EmbeddingService

        emb_blob: bytes | None = None
        if problem_shape.strip():
            try:
                vec = EmbeddingService.embed_one(problem_shape)
                emb_blob = _encode_embedding(vec)
            except Exception as exc:
                log.warning(
                    "skill_embedding_failed",
                    name=name,
                    error=str(exc)[:200],
                )

        # Upsert: try INSERT; on conflict update by (companion_id, user_id, name).
        # SQLite's ON CONFLICT requires a unique index; we don't have one, so
        # do the lookup explicitly. Tradeoff: one extra SELECT per register
        # but predictable + no schema change required.
        existing = await self._fetch_by_name(name, user_id=user_id)
        confidence = _clamp_confidence(confidence)

        if existing is not None:
            await self._backend.conn.execute(
                "UPDATE companion_skills "
                "SET description = ?, problem_shape = ?, embedding = ?, "
                "    updated_at = datetime('now') "
                "WHERE id = ?",
                (description, problem_shape, emb_blob, existing.id),
            )
            await self._backend.conn.commit()
            updated = await self._fetch_by_id(existing.id)
            assert updated is not None
            return updated

        cursor = await self._backend.conn.execute(
            "INSERT INTO companion_skills "
            "(companion_id, user_id, name, description, problem_shape, "
            " embedding, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._companion_id,
                user_id,
                name,
                description,
                problem_shape,
                emb_blob,
                confidence,
            ),
        )
        await self._backend.conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        if not row_id:
            raise RuntimeError("skill_register_failed: no row_id")
        skill = await self._fetch_by_id(int(row_id))
        assert skill is not None

        if self._bus is not None:
            try:
                await self._bus.publish_topic(
                    "skill.registered",
                    {
                        "skill_id": skill.id,
                        "name": skill.name,
                        "user_id": skill.user_id,
                    },
                    source_companion_id=self._companion_id,
                )
            except Exception:
                log.debug("skill_register_bus_emit_failed", exc_info=True)

        return skill

    async def get(self, skill_id: int) -> Skill | None:
        return await self._fetch_by_id(skill_id)

    async def get_by_name(
        self, name: str, *, user_id: str | None = None,
    ) -> Skill | None:
        return await self._fetch_by_name(name, user_id=user_id)

    # ── Instance recording ────────────────────────────────────────────

    async def record_instance(
        self,
        skill_id: int,
        *,
        context: str,
        approach: str,
        user_id: str | None = None,
        session_id: str = "",
        invocation_id: str = "",
        turn_ref: dict | None = None,
    ) -> SkillInstance:
        """Record an application of a skill to a specific situation.

        Bumps the skill's ``instances_count``. Does NOT update
        confidence — confidence only moves when an outcome lands.
        """
        turn_ref_json = json.dumps(turn_ref or {})
        # INSERT + count-bump in one transaction so a crash between them
        # can't drift instances_count from the actual instance rows
        # (audit 2026-06-17).
        async with transactional_write(self._backend.conn) as conn:
            cursor = await conn.execute(
                "INSERT INTO companion_skill_instances "
                "(companion_id, user_id, skill_id, context, approach, "
                " session_id, invocation_id, turn_ref) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._companion_id,
                    user_id,
                    skill_id,
                    context,
                    approach,
                    session_id,
                    invocation_id,
                    turn_ref_json,
                ),
            )
            row_id = cursor.lastrowid
            await cursor.close()
            if not row_id:
                raise RuntimeError("skill_instance_insert_failed: no row_id")

            # Bump the parent skill's instances_count (same txn).
            await conn.execute(
                "UPDATE companion_skills "
                "SET instances_count = instances_count + 1, "
                "    updated_at = datetime('now') "
                "WHERE id = ?",
                (skill_id,),
            )

        instance = await self._fetch_instance(int(row_id))
        assert instance is not None

        if self._bus is not None:
            try:
                await self._bus.publish_topic(
                    "skill.instance_recorded",
                    {
                        "instance_id": instance.id,
                        "skill_id": instance.skill_id,
                        "user_id": instance.user_id,
                        "session_id": instance.session_id,
                    },
                    source_companion_id=self._companion_id,
                )
            except Exception:
                log.debug("skill_instance_bus_emit_failed", exc_info=True)

        return instance

    # ── Outcome recording ─────────────────────────────────────────────

    async def record_outcome(
        self,
        instance_id: int,
        *,
        outcome: str,
        evidence: str = "",
        signal: float | None = None,
        detected_by: str = "inferred",
        user_id: str = "",
    ) -> SkillOutcome:
        """Record an outcome for an instance. Updates parent skill's
        confidence + success/failure counts based on the signal.

        ``signal`` defaults from :data:`_OUTCOME_SIGNAL_DEFAULT` when
        not specified. Pass an explicit value when the inference is
        degraded (e.g. low-confidence inferred outcome should pass a
        smaller signal magnitude).
        """
        if outcome not in _OUTCOME_SIGNAL_DEFAULT:
            outcome = OUTCOME_UNKNOWN
        if signal is None:
            signal = _OUTCOME_SIGNAL_DEFAULT[outcome]
        # Clamp signal to [-1, +1] defensively
        signal = max(-1.0, min(1.0, float(signal)))

        # Outcome INSERT + confidence/count UPDATE in one transaction so a
        # crash between them can't drift successes/failures counts from the
        # outcome rows (audit 2026-06-17). skill_id is None when the
        # instance has no parent skill — then we record the outcome only.
        skill_id: int | None = None
        new_confidence: float | None = None
        async with transactional_write(self._backend.conn) as conn:
            cursor = await conn.execute(
                "INSERT INTO companion_skill_outcomes "
                "(instance_id, outcome, signal, evidence, detected_by, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (instance_id, outcome, signal, evidence, detected_by, user_id),
            )
            row_id = cursor.lastrowid
            await cursor.close()
            if not row_id:
                raise RuntimeError("skill_outcome_insert_failed: no row_id")

            # Look up the parent skill via the instance
            cur = await conn.execute(
                "SELECT skill_id FROM companion_skill_instances WHERE id = ?",
                (instance_id,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row:
                skill_id = int(row[0])
                # Bounded EWMA-style confidence update so one bad outcome
                # doesn't tank an established skill and one good outcome
                # doesn't promote a brand-new one. Weight 0.15 ≈ "10
                # outcomes to fully shift".
                learning_rate = 0.15
                success_delta = 1 if signal > 0.5 else 0
                failure_delta = 1 if signal < -0.5 else 0

                cur = await conn.execute(
                    "SELECT confidence FROM companion_skills WHERE id = ?",
                    (skill_id,),
                )
                crow = await cur.fetchone()
                await cur.close()
                current_confidence = float(crow[0]) if crow else 0.5

                # Map signal in [-1, +1] to target_confidence in [0, 1]
                target = (signal + 1.0) / 2.0
                new_confidence = _clamp_confidence(
                    current_confidence + learning_rate * (target - current_confidence),
                )

                await conn.execute(
                    "UPDATE companion_skills "
                    "SET confidence = ?, "
                    "    successes_count = successes_count + ?, "
                    "    failures_count = failures_count + ?, "
                    "    updated_at = datetime('now') "
                    "WHERE id = ?",
                    (new_confidence, success_delta, failure_delta, skill_id),
                )

        outcome_row = await self._fetch_outcome(int(row_id))
        assert outcome_row is not None

        if self._bus is not None and skill_id is not None:
            try:
                await self._bus.publish_topic(
                    "skill.outcome_observed",
                    {
                        "outcome_id": outcome_row.id,
                        "instance_id": outcome_row.instance_id,
                        "skill_id": skill_id,
                        "outcome": outcome_row.outcome,
                        "signal": outcome_row.signal,
                        "new_confidence": new_confidence,
                    },
                    source_companion_id=self._companion_id,
                )
            except Exception:
                log.debug("skill_outcome_bus_emit_failed", exc_info=True)

        return outcome_row

    # ── Retrieval ─────────────────────────────────────────────────────

    async def query_relevant(
        self,
        intent_text: str,
        *,
        user_id: str | None = None,
        top_k: int = 4,
        min_relevance: float = 0.6,
        min_confidence: float = 0.5,
    ) -> list[RelevantSkill]:
        """Return top-K skills relevant to ``intent_text``.

        Cosine similarity against problem_shape embeddings. Filtered
        to active skills above ``min_confidence``. Untested skills
        stay out of compose-time injection (their default confidence
        is 0.5; if min_confidence > 0.5 they don't make it).

        Returns empty list when there's nothing to inject — *she
        doesn't pretend to know what she doesn't know.*
        """
        if not intent_text.strip():
            return []

        # Embed the intent. Failures gracefully return empty.
        try:
            from augmentum.memory.embeddings import EmbeddingService
            intent_vec = EmbeddingService.embed_one(intent_text)
        except Exception:
            log.debug("skill_query_embed_failed", exc_info=True)
            return []

        # Pull candidate active skills. We scan candidates and rank
        # in Python — SQLite doesn't have native vector ops, and the
        # eligible-skill count is small (tens, not thousands) since
        # skills are curated artifacts not raw data.
        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND status = 'active' "
                "  AND confidence >= ? "
                "ORDER BY updated_at DESC LIMIT 200",
                (self._companion_id, min_confidence),
            )
        else:
            # Per-user skills + cross-user (user_id IS NULL) shared skills
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND status = 'active' "
                "  AND (user_id = ? OR user_id IS NULL) "
                "  AND confidence >= ? "
                "ORDER BY updated_at DESC LIMIT 200",
                (self._companion_id, user_id, min_confidence),
            )
        rows = await cur.fetchall()
        await cur.close()

        scored: list[RelevantSkill] = []
        for r in rows:
            skill = await self._fetch_by_id(int(r[0]))
            if skill is None:
                continue
            # Fetch embedding separately to keep _fetch_by_id light
            cur = await self._backend.conn.execute(
                "SELECT embedding FROM companion_skills WHERE id = ?",
                (skill.id,),
            )
            embrow = await cur.fetchone()
            await cur.close()
            skill_emb = _decode_embedding(embrow[0] if embrow else None)
            if skill_emb is None:
                continue
            relevance = _cosine(intent_vec, skill_emb)
            if relevance < min_relevance:
                continue
            effective = relevance * skill.confidence
            scored.append(RelevantSkill(
                skill=skill,
                relevance=relevance,
                effective_score=effective,
            ))

        scored.sort(key=lambda r: -r.effective_score)
        return scored[:top_k]

    # ── Listing / pagination (for Observatory + tests) ────────────────

    async def list_skills(
        self,
        *,
        user_id: str | None = None,
        status: str = "active",
        limit: int = 100,
    ) -> list[Skill]:
        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND status = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (self._companion_id, status, limit),
            )
        else:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND status = ? "
                "  AND (user_id = ? OR user_id IS NULL) "
                "ORDER BY updated_at DESC LIMIT ?",
                (self._companion_id, status, user_id, limit),
            )
        rows = await cur.fetchall()
        await cur.close()
        out: list[Skill] = []
        for r in rows:
            s = await self._fetch_by_id(int(r[0]))
            if s is not None:
                out.append(s)
        return out

    async def list_instances(
        self, skill_id: int, *, limit: int = 50,
    ) -> list[SkillInstance]:
        cur = await self._backend.conn.execute(
            "SELECT id FROM companion_skill_instances "
            "WHERE skill_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (skill_id, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
        out: list[SkillInstance] = []
        for r in rows:
            i = await self._fetch_instance(int(r[0]))
            if i is not None:
                out.append(i)
        return out

    # ── Internal fetch helpers ────────────────────────────────────────

    async def _fetch_by_id(self, skill_id: int) -> Skill | None:
        cur = await self._backend.conn.execute(
            "SELECT id, companion_id, user_id, name, description, "
            "       problem_shape, confidence, instances_count, "
            "       successes_count, failures_count, abstracted_from_ids, "
            "       status, created_at, updated_at "
            "FROM companion_skills WHERE id = ?",
            (skill_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        try:
            abstracted = json.loads(row[10] or "[]")
        except Exception:
            abstracted = []
        return Skill(
            id=int(row[0]),
            companion_id=str(row[1]),
            user_id=row[2],
            name=str(row[3]),
            description=str(row[4] or ""),
            problem_shape=str(row[5] or ""),
            confidence=float(row[6]),
            instances_count=int(row[7]),
            successes_count=int(row[8]),
            failures_count=int(row[9]),
            abstracted_from_ids=abstracted,
            status=str(row[11]),
            created_at=str(row[12] or ""),
            updated_at=str(row[13] or ""),
        )

    async def _fetch_by_name(
        self, name: str, *, user_id: str | None = None,
    ) -> Skill | None:
        # ``user_id is None`` here means "look up cross-user shared
        # skill"; SQL needs IS NULL specifically since = NULL never
        # matches. Two-branch query.
        if user_id is None:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND name = ? AND user_id IS NULL "
                "LIMIT 1",
                (self._companion_id, name),
            )
        else:
            cur = await self._backend.conn.execute(
                "SELECT id FROM companion_skills "
                "WHERE companion_id = ? AND name = ? AND user_id = ? "
                "LIMIT 1",
                (self._companion_id, name, user_id),
            )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        return await self._fetch_by_id(int(row[0]))

    async def _fetch_instance(self, instance_id: int) -> SkillInstance | None:
        cur = await self._backend.conn.execute(
            "SELECT id, skill_id, companion_id, user_id, context, "
            "       approach, session_id, invocation_id, turn_ref, "
            "       created_at "
            "FROM companion_skill_instances WHERE id = ?",
            (instance_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        try:
            turn_ref = json.loads(row[8] or "{}")
        except Exception:
            turn_ref = {}
        return SkillInstance(
            id=int(row[0]),
            skill_id=int(row[1]),
            companion_id=str(row[2]),
            user_id=row[3],
            context=str(row[4] or ""),
            approach=str(row[5] or ""),
            session_id=str(row[6] or ""),
            invocation_id=str(row[7] or ""),
            turn_ref=turn_ref or None,
            created_at=str(row[9] or ""),
        )

    async def _fetch_outcome(self, outcome_id: int) -> SkillOutcome | None:
        cur = await self._backend.conn.execute(
            "SELECT id, instance_id, outcome, signal, evidence, "
            "       detected_at, detected_by "
            "FROM companion_skill_outcomes WHERE id = ?",
            (outcome_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        return SkillOutcome(
            id=int(row[0]),
            instance_id=int(row[1]),
            outcome=str(row[2]),
            signal=float(row[3]),
            evidence=str(row[4] or ""),
            detected_at=str(row[5] or ""),
            detected_by=str(row[6]),
        )


__all__ = [
    "Skill",
    "SkillInstance",
    "SkillOutcome",
    "RelevantSkill",
    "SkillGraph",
    "OUTCOME_ACCEPTED",
    "OUTCOME_REJECTED",
    "OUTCOME_CORRECTED",
    "OUTCOME_SHIPPED",
    "OUTCOME_PROBLEMATIC",
    "OUTCOME_UNKNOWN",
]
