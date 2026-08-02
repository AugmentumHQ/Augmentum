"""The Companion class — the single object that *is* a companion.

See the accumulation thesis (Step 2):
``docs/superpowers/specs/2026-05-23-accumulation-thesis.md``.

This is Phase 1: a thin façade over the existing systems. Every
method delegates to ``CompanionRuntime`` and its constituents. The
point is to have *one place to look* — and *one place every other
module should talk to* — so future PRs can migrate read/write sites
into routing through this class without behavior change at each step.

**Construction.** ``Companion(runtime)`` wraps an existing live
``CompanionRuntime``. The Companion does not own the runtime; it
exposes the runtime's parts in a unified, well-named interface.
The runtime continues to be the lifecycle owner — start(), stop(),
tick loop, observer, drift audit all stay there.

**Per-user views.** ``companion.for_user(user_id)`` returns a
:class:`CompanionUserView` — the per-user slice of her. Identity,
state, and the user_affect read all scope to that user; the
constitutional layer (kernel digest, behavior contract, exemplar
library) is shared. This is the architectural answer to the
multi-tenancy ambiguity: shared substrate, divergent relationships.

**Why a façade first.** A big-bang refactor (every read/write site
through the class in one PR) would be risky and hard to review. The
façade lets every PR after this one migrate one site at a time
without breaking anything. By the end of the migration the façade
methods are the only paths; before then, they coexist with direct
access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.bus import PresenceBus, PresenceEvent
    from augmentum.companion_runtime.identity import CompanionIdentity
    from augmentum.companion_runtime.memory import CompanionMemory
    from augmentum.companion_runtime.perception.user_affect import (
        UserAffectObservation,
        UserAffectTracker,
    )
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent, Response
    from augmentum.companion_runtime.state import CompanionState

log = get_logger(__name__)


class Companion:
    """The single addressable companion object.

    Composes the constituent systems and exposes them through one
    interface. Lifecycle stays with the underlying
    :class:`CompanionRuntime`; this is a *view* + *router*, not a
    new owner.

    Usage from the rest of the codebase:

    .. code-block:: python

        companion = app.state.companions["becca"]
        view = companion.for_user(user_id)
        digest = view.identity.persona_kernel_digest
        await view.observe(event)
    """

    def __init__(self, runtime: CompanionRuntime) -> None:
        self._runtime = runtime
        self.name: str = runtime.companion_id
        # Lazy — skills graph is constructed on first access so older
        # runtimes / test fixtures without a backend don't crash on
        # Companion construction.
        self._skills: Any = None
        # Lazy — lesson registry (mig 270), same construction discipline.
        self._lessons: Any = None

    # ── Shared constitutional layer ───────────────────────────────────

    @property
    def runtime(self) -> CompanionRuntime:
        """The underlying runtime. Exposed for code that still needs
        direct access during the migration. New code should prefer the
        named delegators below or :meth:`for_user`."""
        return self._runtime

    @property
    def bus(self) -> PresenceBus:
        """The presence bus. Every event with this companion's
        ``source_companion_id`` flows through here."""
        return self._runtime.bus

    @property
    def memory(self) -> CompanionMemory:
        """The memory facade (journal + observations + creations +
        relationship slice). Note: per-user scoping is handled inside
        most memory methods via the ``user_id`` kwarg."""
        return self._runtime.memory

    @property
    def user_affect(self) -> UserAffectTracker | None:
        """The user-observed-affect tracker (Synapse §2). May be
        ``None`` on older runtimes that haven't initialized it."""
        return getattr(self._runtime, "user_affect", None)

    @property
    def skills(self) -> Any:
        """The skill graph + outcome ledger (thesis Step 3).

        Lazily constructs a :class:`SkillGraph` over the runtime's
        backend on first access. Returns ``None`` when the backend
        isn't available (test fixtures, degraded runtime).
        """
        if self._skills is not None:
            return self._skills
        backend = getattr(self._runtime, "backend", None)
        if backend is None:
            return None
        try:
            from augmentum.companion.skills import SkillGraph
            self._skills = SkillGraph(
                backend,
                bus=self._runtime.bus,
                companion_id=self._runtime.companion_id,
            )
            return self._skills
        except Exception:
            log.debug("companion_skills_init_failed", exc_info=True)
            return None

    @property
    def lessons(self) -> Any:
        """The lesson registry (mig 270) — the learn-from-correction
        sibling of the skill graph.

        Lazily constructs a :class:`~augmentum.companion.lessons.LessonGraph`
        over the runtime's backend on first access. Returns ``None`` when
        the backend isn't available (test fixtures, degraded runtime).
        """
        if self._lessons is not None:
            return self._lessons
        backend = getattr(self._runtime, "backend", None)
        if backend is None:
            return None
        try:
            from augmentum.companion.lessons import LessonGraph
            self._lessons = LessonGraph(
                backend,
                bus=self._runtime.bus,
                companion_id=self._runtime.companion_id,
            )
            return self._lessons
        except Exception:
            log.debug("companion_lessons_init_failed", exc_info=True)
            return None

    @property
    def companion_id(self) -> str:
        """Stable id (same as ``self.name``). Kept for parity with the
        runtime's existing attribute name."""
        return self._runtime.companion_id

    @property
    def owner_user_id(self) -> str:
        """The user_id bound as owner in single-companion phase."""
        return self._runtime.owner_user_id

    @property
    def started(self) -> bool:
        """Whether the runtime has completed start()."""
        return bool(getattr(self._runtime, "_started", False))

    # ── Per-user view ─────────────────────────────────────────────────

    def for_user(self, user_id: str) -> CompanionUserView:
        """Return the per-user slice of this companion.

        Identity, state, and affect read are user-scoped. The shared
        constitutional layer (kernel digest, behavior contract,
        exemplar library) is shared across all users. Two views for
        different users see the same constitution but different lived
        history.

        Empty ``user_id`` returns a view over the legacy seed identity
        — preserves the pre-Piece-1 API for callers that haven't
        adopted per-user scoping. New code should pass a real user_id
        from request scope.
        """
        return CompanionUserView(self, user_id)

    # ── Convenience delegators ────────────────────────────────────────

    async def observe(self, event: PresenceEvent) -> None:
        """Publish an event on her bus. This is the canonical write
        path for anything the world does that she should be aware of."""
        await self._runtime.bus.publish(event)

    async def publish_topic(
        self,
        topic: str,
        payload: dict | None = None,
        *,
        propagation: str = "full",
    ) -> None:
        """Convenience: publish a topic without constructing an Event."""
        await self._runtime.bus.publish_topic(
            topic,
            payload or {},
            source_companion_id=self.name,
            propagation=propagation,
        )

    async def submit_intent(self, intent: Intent) -> Response:
        """Submit an intent to dispatch. Same semantics as
        ``runtime.submit_intent``."""
        return await self._runtime.submit_intent(intent)

    async def snapshot(self) -> dict:
        """Health snapshot of the companion. Used by the Observatory
        + telemetry surfaces."""
        return await self._runtime.snapshot()

    # ── Lifecycle (Phase 2+) ──────────────────────────────────────────
    #
    # These are placeholders for the larger interface the thesis
    # describes (respond, express, receive_feedback, consolidate).
    # They live here so future PRs have a stable target to migrate
    # into, even though the implementation today routes through the
    # existing paths.

    async def consolidate(self) -> dict:
        """Run a consolidation pass (Synapse §4). Currently delegates
        to the explicit consolidation module; future Phase 2 makes
        this method the single trigger across all consolidation
        subsystems."""
        # Honest contract: this method does NOT run consolidation yet
        # (Phase 2). Returning ok:True made callers/telemetry read a no-op
        # as a successful pass (audit 2026-06-17). Report ok:False +
        # available so a caller can route to the real entry point.
        try:
            from augmentum.companion_runtime import consolidation  # noqa: F401
            return {
                "ok": False,
                "note": "not wired — call consolidation.propose_candidate directly (Phase 2)",
                "available": True,
            }
        except Exception:
            return {"ok": False, "note": "consolidation module unavailable", "available": False}


class CompanionUserView:
    """The per-user slice of a companion.

    Wraps a :class:`Companion` + a ``user_id`` so callers don't have
    to thread ``user_id`` through every call site. Reads scope to the
    user automatically.

    Usage:

    .. code-block:: python

        view = companion.for_user(request.scope["user"].id)
        digest = (await view.identity()).persona_kernel_digest
        affect = view.read_affect()
    """

    __slots__ = ("_companion", "user_id")

    def __init__(self, companion: Companion, user_id: str) -> None:
        self._companion = companion
        self.user_id = user_id

    @property
    def companion(self) -> Companion:
        return self._companion

    @property
    def name(self) -> str:
        return self._companion.name

    # ── Identity (per-user) ───────────────────────────────────────────

    async def identity(self) -> CompanionIdentity:
        """Per-user :class:`CompanionIdentity`. Lazily provisioned via
        the runtime's ``get_identity``."""
        return await self._companion.runtime.get_identity(self.user_id)

    # ── State (per-user) ──────────────────────────────────────────────

    async def state(self) -> CompanionState:
        """Per-user :class:`CompanionState`. Lazily provisioned via
        the runtime's ``get_state``."""
        return await self._companion.runtime.get_state(self.user_id)

    # ── Memory writes (user-scoped) ───────────────────────────────────

    async def journal(
        self,
        content: str,
        *,
        entry_type: str = "observation",
        affect_tag: str | None = None,
        related_memory_ids: list[str] | None = None,
        content_refs: list[dict] | None = None,
        place_ref: str = "",
        embed: bool = True,
        source: str = "autonomous",
        confidence_numeric: float = 0.6,
    ) -> int:
        """Write a journal entry scoped to this user."""
        memory = self._companion.memory
        return await memory.journal(
            content=content,
            entry_type=entry_type,
            user_id=self.user_id or None,
            affect_tag=affect_tag,
            related_memory_ids=related_memory_ids,
            content_refs=content_refs,
            place_ref=place_ref,
            embed=embed,
            source=source,
            confidence_numeric=confidence_numeric,
        )

    # ── Affect (user-scoped read) ─────────────────────────────────────

    def read_affect(self) -> UserAffectObservation | None:
        """Current decayed affect read for this user. Returns ``None``
        when no tracker is available; otherwise a
        :class:`UserAffectObservation` (confidence 0.0 when no recent
        observation exists)."""
        tracker = self._companion.user_affect
        if tracker is None:
            return None
        return tracker.read(self.user_id)

    # ── Convenience pass-throughs ─────────────────────────────────────

    @property
    def bus(self) -> PresenceBus:
        return self._companion.bus

    async def observe(self, event: PresenceEvent) -> None:
        await self._companion.observe(event)

    async def publish_topic(
        self,
        topic: str,
        payload: dict | None = None,
        *,
        propagation: str = "full",
    ) -> None:
        await self._companion.publish_topic(
            topic, payload, propagation=propagation,
        )

    # ── Skill graph (per-user retrieval at compose time) ──────────────

    async def relevant_skills(
        self,
        intent_text: str,
        *,
        top_k: int | None = None,
        min_relevance: float | None = None,
        min_confidence: float | None = None,
    ) -> list:
        """Top-K skills relevant to ``intent_text`` for this user.

        Reads gating thresholds from settings unless explicit values
        are passed. Returns ``[]`` when the skill graph isn't
        available or the feature flag is off — callers proceed as if
        there were no relevant skills (which is the honest default).
        """
        from augmentum.config import settings as _settings
        if not getattr(_settings, "companion_skills_enabled", False):
            return []
        graph = self._companion.skills
        if graph is None:
            return []
        return await graph.query_relevant(
            intent_text,
            user_id=self.user_id,
            top_k=top_k or int(getattr(_settings, "companion_skill_inject_top_k", 4)),
            min_relevance=(
                min_relevance
                if min_relevance is not None
                else float(getattr(_settings, "companion_skill_relevance_threshold", 0.6))
            ),
            min_confidence=(
                min_confidence
                if min_confidence is not None
                else float(getattr(_settings, "companion_skill_min_confidence_for_inject", 0.5))
            ),
        )

    # ── Lesson registry (per-user retrieval at compose time) ──────────

    async def relevant_lessons(
        self,
        intent_text: str,
        *,
        top_k: int | None = None,
        min_relevance: float | None = None,
        min_strength: float | None = None,
    ) -> list:
        """Top-K lessons relevant to ``intent_text`` for this user.

        The guardrail half of compose-time accumulation: corrections the
        user made, retrieved by situation similarity. Returns ``[]`` when
        the registry isn't available or ``companion_lessons_enabled`` is
        off — callers proceed as if there were no relevant lessons (the
        honest default: she doesn't carry a guardrail into a situation it
        wasn't about).
        """
        from augmentum.config import settings as _settings
        if not getattr(_settings, "companion_lessons_enabled", False):
            return []
        graph = self._companion.lessons
        if graph is None:
            return []
        return await graph.query_relevant(
            intent_text,
            user_id=self.user_id,
            top_k=top_k or int(getattr(_settings, "companion_lessons_inject_top_k", 3)),
            min_relevance=(
                min_relevance
                if min_relevance is not None
                else float(getattr(_settings, "companion_lessons_relevance_threshold", 0.6))
            ),
            min_strength=(
                min_strength
                if min_strength is not None
                else float(getattr(_settings, "companion_lessons_min_strength_for_inject", 0.5))
            ),
        )


__all__ = ["Companion", "CompanionUserView"]
