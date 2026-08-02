"""PresenceState + PresenceContext + transition table.

Pure data + predicates. Zero I/O, zero asyncio, zero side effects.
The orchestrator in pipeline.py owns the I/O and coordination; this
module is just the state model.

Why a data-driven transition table instead of if/elif in the pipeline:
when phases 4-6 land, the matrix of (state, event) pairs grows. Keeping
it as a literal table makes the validity rules auditable in one place,
test-able exhaustively, and refactorable without touching the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PresenceState(str, Enum):
    """The 8 states of the presence pipeline state machine.

    Subclassing str so the enum value serializes naturally to JSON for
    the WS protocol without a custom encoder.
    """

    IDLE = "idle"
    LISTENING = "listening"
    GENERATING_SPECULATIVE = "generating_speculative"
    GENERATING = "generating"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    CANCEL_SPECULATIVE = "cancel_speculative"
    ERROR = "error"


class PresenceEvent(str, Enum):
    """The 12 events the pipeline orchestrator dispatches transitions on.

    Some events carry payloads (llm_token has the token text, on_error has
    the exception). The transition table is keyed by (state, event) and is
    agnostic to payloads — payloads are consumed by the orchestrator's
    handler when it applies side effects + updates context.
    """

    SPEECH_DETECTED = "speech_detected"
    TURN_LIKELY = "turn_likely"
    TURN_COMMITTED = "turn_committed"
    SPEECH_CONTINUED = "speech_continued"
    LLM_TOKEN = "llm_token"
    FIRST_CHUNK_READY = "first_chunk_ready"
    CHUNK_QUEUE_EMPTY = "chunk_queue_empty"
    INTERRUPT_VAD = "interrupt_vad"
    USER_BACKCHANNEL_DETECTED = "user_backchannel_detected"
    BEAT_COMPLETE = "beat_complete"
    CLEANUP_COMPLETE = "cleanup_complete"
    ERROR_OCCURRED = "error_occurred"
    RECOVERED = "recovered"


# Valid (from_state, event) -> to_state transitions. Any (state, event)
# pair not in this table is an invalid transition — the orchestrator logs
# a warning and remains in the current state. This is the single source
# of truth for the state machine; tests assert this table is exhaustive
# for every state + every event combination we ship.
#
# Self-loops are explicit (e.g. LISTENING + SPEECH_CONTINUED -> LISTENING)
# rather than implicit — keeps the table readable and avoids "did I
# forget that edge?" ambiguity.
VALID_TRANSITIONS: dict[
    tuple[PresenceState, PresenceEvent], PresenceState
] = {
    # ── IDLE ──────────────────────────────────────────────────────
    (PresenceState.IDLE, PresenceEvent.SPEECH_DETECTED): PresenceState.LISTENING,

    # ── LISTENING ─────────────────────────────────────────────────
    (PresenceState.LISTENING, PresenceEvent.TURN_LIKELY):
        PresenceState.GENERATING_SPECULATIVE,
    (PresenceState.LISTENING, PresenceEvent.TURN_COMMITTED):
        PresenceState.GENERATING,
    (PresenceState.LISTENING, PresenceEvent.SPEECH_CONTINUED):
        PresenceState.LISTENING,  # explicit self-loop

    # ── GENERATING_SPECULATIVE ────────────────────────────────────
    (PresenceState.GENERATING_SPECULATIVE, PresenceEvent.TURN_COMMITTED):
        PresenceState.GENERATING,
    (PresenceState.GENERATING_SPECULATIVE, PresenceEvent.SPEECH_CONTINUED):
        PresenceState.CANCEL_SPECULATIVE,
    (PresenceState.GENERATING_SPECULATIVE, PresenceEvent.LLM_TOKEN):
        PresenceState.GENERATING_SPECULATIVE,  # buffer, stay in state

    # ── GENERATING ────────────────────────────────────────────────
    (PresenceState.GENERATING, PresenceEvent.LLM_TOKEN):
        PresenceState.GENERATING,  # buffer until first_chunk_ready
    (PresenceState.GENERATING, PresenceEvent.FIRST_CHUNK_READY):
        PresenceState.SPEAKING,

    # ── SPEAKING ──────────────────────────────────────────────────
    (PresenceState.SPEAKING, PresenceEvent.INTERRUPT_VAD):
        PresenceState.INTERRUPTED,
    (PresenceState.SPEAKING, PresenceEvent.CHUNK_QUEUE_EMPTY):
        PresenceState.IDLE,
    (PresenceState.SPEAKING, PresenceEvent.USER_BACKCHANNEL_DETECTED):
        PresenceState.SPEAKING,  # explicit self-loop; backchannel doesn't interrupt

    # ── INTERRUPTED ───────────────────────────────────────────────
    (PresenceState.INTERRUPTED, PresenceEvent.BEAT_COMPLETE):
        PresenceState.LISTENING,

    # ── CANCEL_SPECULATIVE ────────────────────────────────────────
    (PresenceState.CANCEL_SPECULATIVE, PresenceEvent.CLEANUP_COMPLETE):
        PresenceState.LISTENING,

    # ── ERROR ─────────────────────────────────────────────────────
    (PresenceState.ERROR, PresenceEvent.RECOVERED): PresenceState.IDLE,
}


# Any state can be forced into ERROR by ERROR_OCCURRED — we generate
# these entries programmatically rather than hand-listing 8 rows.
for _from in PresenceState:
    if _from is PresenceState.ERROR:
        continue
    VALID_TRANSITIONS[(_from, PresenceEvent.ERROR_OCCURRED)] = PresenceState.ERROR


def is_valid_transition(
    from_state: PresenceState, event: PresenceEvent,
) -> bool:
    """True iff the (state, event) pair has a defined transition."""
    return (from_state, event) in VALID_TRANSITIONS


def next_state(
    from_state: PresenceState, event: PresenceEvent,
) -> PresenceState | None:
    """Resolve the next state for (from_state, event), or None if invalid.

    The orchestrator uses this for state transition; tests use it to
    exhaustively verify the table. Returning None on invalid transition
    (rather than raising) lets the orchestrator make the policy decision
    about what to do — currently: log warning + stay.
    """
    return VALID_TRANSITIONS.get((from_state, event))


@dataclass
class PendingAction:
    """A side-effect verb queued during speculative generation.

    Per decision D3: when the pipeline speculatively kicks off LLM
    generation at Smart Turn p>0.7, any verbs the LLM invokes are
    deferred to this buffer rather than firing immediately. On
    turn_committed they all fire (in order). On cancel_speculative
    they're all dropped.

    The verb dispatcher integration arrives in Phase 4 when speculative
    generation lands; Phase 1 just sets up the buffer + flush + drop
    primitives on the context.
    """

    verb_id: str
    args: dict[str, Any] = field(default_factory=dict)
    queued_at: float = 0.0  # monotonic timestamp


@dataclass
class PresenceContext:
    """Per-conversation state held by the orchestrator.

    Multi-tenant invariant: every PresenceContext is constructed with a
    user_id (extracted from request scope at WS open). The orchestrator
    keys pipeline instances by (user_id, session_id) so two users with
    the same session id never share state.

    Mutable fields are updated by the orchestrator's transition handlers;
    nothing outside the orchestrator should mutate the context directly.
    """

    session_id: str
    user_id: str

    # Lifecycle
    created_at: float = 0.0      # monotonic seconds since module load
    state_entered_at: float = 0.0  # transition-into-current-state timestamp

    # Accumulators
    partial_transcript: str = ""    # ASR partial during LISTENING
    llm_token_buffer: str = ""      # streamed tokens awaiting chunker handoff
    pending_actions: list[PendingAction] = field(default_factory=list)

    # Recovery flags consumed by next turn's prompt frame
    was_interrupted: bool = False
    mid_phrase: str = ""           # what Becca was saying when interrupted

    # Diagnostic / observability — appended every transition for tests +
    # future telemetry. Bounded to avoid unbounded memory growth in long
    # sessions; oldest entries fall off when len > MAX_HISTORY.
    transition_count: int = 0

    def reset_turn_buffers(self) -> None:
        """Clear per-turn accumulators when a turn closes (idle / listening).

        Called by transition handlers that complete a turn cycle. Does NOT
        clear was_interrupted / mid_phrase — those are intentionally
        preserved across the IDLE/LISTENING boundary so the next turn's
        prompt frame can use them.
        """
        self.partial_transcript = ""
        self.llm_token_buffer = ""
        self.pending_actions.clear()

    def commit_pending_actions(self) -> list[PendingAction]:
        """Flush + return the deferred actions queue (turn_committed path).

        Caller is responsible for invoking them against the verb dispatcher.
        Returning the list rather than firing them here keeps state.py
        side-effect-free.
        """
        actions = list(self.pending_actions)
        self.pending_actions.clear()
        return actions

    def drop_pending_actions(self) -> int:
        """Drop the deferred actions queue (cancel_speculative path).

        Returns the dropped count for diagnostic logging.
        """
        n = len(self.pending_actions)
        self.pending_actions.clear()
        return n

    def clear_recovery_flags(self) -> None:
        """Clear was_interrupted + mid_phrase once the next turn consumes them.

        Called by the orchestrator's TURN_COMMITTED handler after the
        prompt frame reads these fields, so the flags don't persist
        beyond the turn they apply to.
        """
        self.was_interrupted = False
        self.mid_phrase = ""


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a single state transition, emitted to listeners.

    Frozen because listeners may store these for telemetry / replay; we
    don't want a listener mutating one and corrupting another's view.
    """

    from_state: PresenceState
    to_state: PresenceState
    event: PresenceEvent
    timestamp: float  # monotonic seconds
    session_id: str
    user_id: str
