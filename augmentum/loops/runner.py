"""LoopRunner — shared agent-loop substrate.

This file ships the runner's *shape* for PR-2.1. ``run()`` is a stub
that raises ``NotImplementedError`` — the real implementation lands in
PR-2.4 once PR-2.2 (ledger) and PR-2.3 (breakers) have been extracted.

The constructor signature is the contract every mode will adapt to:

* an :class:`Intensity` chosen at call time
* one of each capability protocol (:class:`FileIO`,
  :class:`ToolExecutor`, optional :class:`PermissionGate`,
  optional :class:`QuestionAsker`)
* a :class:`ChunkEmitter` to stream results back

By defining the surface in PR-2.1 we let downstream PRs land
incrementally without re-cutting callers. Anyone touching coder or
agentic between now and PR-2.6 should construct a LoopRunner the same
way regardless of which PRs have shipped — the only thing changing is
how much real work ``run()`` does.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from augmentum.loops.protocols import (
    ChunkEmitter,
    FileIO,
    PermissionGate,
    QuestionAsker,
    ToolExecutor,
)
from augmentum.loops.tier import HEAVY, Intensity


class RunStatus(str, enum.Enum):
    """Final outcome of a :meth:`LoopRunner.run` call."""

    COMPLETED = "completed"
    """All promises verified, model emitted a clean stop signal."""

    STOPPED_BY_BREAKER = "stopped_by_breaker"
    """A soft circuit-breaker fired (PR-2.3). Detail in ``stop_reason``."""

    BUDGET_EXCEEDED = "budget_exceeded"
    """Hard iteration ceiling was hit before completion."""

    VERIFY_FAILED = "verify_failed"
    """Verify-gate (PR-2.5) refused to release a terminating chunk —
    one or more Promises remain in PENDING/IN_PROGRESS state."""

    CANCELLED = "cancelled"
    """Caller cancelled mid-loop (e.g. user clicked Stop)."""

    ERRORED = "errored"
    """Unhandled exception bubbled out of the act loop. ``error_text``
    carries the message; the broker logs the traceback."""


@dataclass
class RunResult:
    """Summary returned at the end of a :meth:`LoopRunner.run`.

    Streaming output reaches the caller through the
    :class:`ChunkEmitter`; the result captures the *summary* the next
    turn / verifier / UI needs.
    """

    status: RunStatus
    iterations_used: int = 0
    stop_reason: str = ""
    """Free-text reason. For STOPPED_BY_BREAKER, this is the
    breaker's id (e.g. ``test_failure_streak``)."""

    error_text: str = ""
    """Populated when ``status == ERRORED``. Empty otherwise."""

    promises_pending: int = 0
    """Count of Promises still unverified at exit. Always 0 for
    COMPLETED; may be non-zero for VERIFY_FAILED / BUDGET_EXCEEDED."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Mode-specific extras the caller may want — e.g. the coder's
    turn_id, the agentic mode's flow_id."""


class LoopRunner:
    """Drive a single act-loop session.

    The runner is intentionally short-lived: one instance per call to
    :meth:`run`. The constructor accepts everything the loop needs as
    explicit dependencies — no globals, no module-level state — so
    tests can drive it with stubs and so the same class works for
    coder, agentic, and any future surface.

    Stateful machinery the runner will own in later PRs:

    * **ObservationLedger** (PR-2.2) — validation errors, tool failures,
      tool-call fingerprints, prior-turn summaries.
    * **BreakerRegistry** (PR-2.3) — soft circuit-breaker dispatch.
    * **VerifyGate** (PR-2.5) — mandatory pre-terminate Promise check.

    These are accessible after construction via private attributes so
    PR-2.2/2.3/2.5 can populate them without re-cutting the public
    surface.
    """

    def __init__(
        self,
        *,
        intensity: Intensity = HEAVY,
        tools: ToolExecutor,
        files: FileIO,
        emitter: ChunkEmitter,
        permission: PermissionGate | None = None,
        questions: QuestionAsker | None = None,
    ) -> None:
        self.intensity = intensity
        self._tools = tools
        self._files = files
        self._emitter = emitter
        self._permission = permission
        self._questions = questions

        # Reserved slots — PR-2.2 / PR-2.3 / PR-2.5 populate these. The
        # attributes exist now so callers can hold a stable reference
        # and so type checkers see the shape from day one.
        self._ledger: Any | None = None
        self._breakers: Any | None = None
        self._verify_gate: Any | None = None

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        model: str,
        extra: dict[str, Any] | None = None,
    ) -> RunResult:
        """Execute one act-loop session and return its summary.

        Stub for PR-2.1 — raises :class:`NotImplementedError`. PR-2.4
        wires the actual loop after the ledger (PR-2.2) and breakers
        (PR-2.3) are in place.

        Parameters
        ----------
        prompt:
            The user's request for this turn.
        system_prompt:
            Mode-specific system text. The runner appends prior-turn
            memory + observation ledger surface on top of this.
        model:
            Backend model id (e.g. ``"glm-4.7"``,
            ``"claude-opus-4-7"``). Passed through to ``ToolExecutor``;
            the runner itself is model-agnostic.
        extra:
            Mode-specific extras (e.g. ``{"workspace_id": ...,
            "session_id": ...}``). The runner stores these on
            :class:`RunResult`.metadata.
        """
        raise NotImplementedError(
            "LoopRunner.run() is a PR-2.1 stub; the real loop lands in "
            "PR-2.4 once PR-2.2 (ledger) and PR-2.3 (breakers) are in "
            "place. See docs/superpowers/specs/"
            "2026-05-29-integrated-coding-nervous-system.md."
        )


__all__ = [
    "LoopRunner",
    "RunResult",
    "RunStatus",
]
