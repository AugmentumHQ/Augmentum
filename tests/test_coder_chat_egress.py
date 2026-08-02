"""Tests for chat_egress metadata validation (Phase 7).

The Phase/Status Literals in chat_egress.py are meant to be the
single source of truth for what the coder loop emits as meta events.
Runtime validation in emit() / emit_relay() / _meta_chunk catches
drift — a new status emitted without a corresponding Literal entry
raises in strict mode (default).
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.chat_egress import (
    _VALID_PHASES,
    _VALID_STATUSES,
    _validate_metadata,
    emit,
    emit_relay,
)
from augmentum.models.base import InternalStreamChunk


# ---------------------------------------------------------------------------
# The known-good set stays in sync with what the loop actually emits.
# ---------------------------------------------------------------------------


def test_valid_phases_covers_known_five():
    # Regression guard: the five phases the loop uses. If any of these
    # disappear, a bunch of emit sites will start raising.
    assert _VALID_PHASES >= {
        "planning", "executing", "passthrough",
        "conversational", "waiting",
    }


def test_valid_statuses_covers_termination_reasons():
    # Every hybrid-loop break path has a corresponding status.
    assert {
        "tasks_completed",
        "validation_error_break",
        "test_failure_streak_break",
        "same_file_edit_break",
        "action_stagnation_break",
        "inspection_loop_break",
        "max_iterations_reached",
        "fallback_summary",
    } <= _VALID_STATUSES


def test_valid_statuses_covers_nudges():
    assert {
        "continuation_nudge",
        "operate_evidence_nudge",
        "stagnation_nudge",
        "unclaimed_code_block_nudge",
        "content_loop_nudge",
        "inspection_loop_nudge",
    } <= _VALID_STATUSES


# ---------------------------------------------------------------------------
# _validate_metadata behaviour
# ---------------------------------------------------------------------------


def test_valid_metadata_does_not_raise():
    _validate_metadata("executing", "streaming")


def test_unknown_phase_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "1")
    with pytest.raises(ValueError, match="phase="):
        _validate_metadata("bogus_phase", "streaming")


def test_unknown_status_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "1")
    with pytest.raises(ValueError, match="status="):
        _validate_metadata("executing", "bogus_status")


def test_both_unknown_raises_once_with_both(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "1")
    with pytest.raises(ValueError) as ei:
        _validate_metadata("bogus_phase", "bogus_status")
    assert "phase=" in str(ei.value)
    assert "status=" in str(ei.value)


def test_env_override_downgrades_to_warning(monkeypatch, caplog):
    # Emergency bypass — some in-flight deploy emits a new status the
    # Literal hasn't caught up on yet. Setting =0 / false / no / off
    # logs instead of raises.
    for val in ("0", "false", "no", "off"):
        monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", val)
        # Should NOT raise
        _validate_metadata("executing", "completely_new_status")


# ---------------------------------------------------------------------------
# emit() / emit_relay() integration
# ---------------------------------------------------------------------------


def test_emit_rejects_unknown_status_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "1")
    with pytest.raises(ValueError):
        emit(
            "hello",
            phase="executing", status="totally_new_status_never_seen",
            model="test",
        )


def test_emit_accepts_known_status():
    chunk = emit(
        "hello",
        phase="executing", status="streaming",
        model="test",
    )
    assert chunk.content_delta == "hello"
    assert chunk.augmentum == {
        "mode": "coder", "phase": "executing", "status": "streaming",
    }


def test_emit_relay_rejects_unknown_status_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "1")
    source = InternalStreamChunk(content_delta="ok", model="m")
    with pytest.raises(ValueError):
        emit_relay(
            source,
            phase="executing", status="never_heard_of_it",
            model_fallback="test",
        )


def test_emit_relay_accepts_known_status():
    source = InternalStreamChunk(content_delta="ok", model="m")
    chunk = emit_relay(
        source,
        phase="passthrough", status="streaming",
        model_fallback="test",
    )
    assert chunk.content_delta == "ok"
    assert chunk.augmentum["mode"] == "coder"
    assert chunk.augmentum["phase"] == "passthrough"


def test_extra_keys_not_validated():
    # The validator only touches phase/status. Callers can freely add
    # custom keys via ``extra={...}`` — those are not schema-checked.
    chunk = emit(
        "hello",
        phase="executing", status="streaming",
        model="test",
        extra={"custom_key": "arbitrary_value", "count": 42},
    )
    assert chunk.augmentum["custom_key"] == "arbitrary_value"
    assert chunk.augmentum["count"] == 42


# ---------------------------------------------------------------------------
# StreamProgressTracker — streaming sub-state transitions
# ---------------------------------------------------------------------------


def test_stream_progress_tracker_emits_awaiting_first_token_on_begin():
    """``begin`` returns the initial TTFT / prefix-eval marker chunk."""
    from augmentum.modes.coder.chat_egress import StreamProgressTracker

    tracker = StreamProgressTracker()
    chunk = tracker.begin(phase="planning", model="test-model")
    assert chunk.augmentum["status"] == "awaiting_first_token"
    assert chunk.augmentum["phase"] == "planning"
    assert tracker.substate == "awaiting_first_token"


def test_stream_progress_tracker_thinking_then_responding():
    """A reasoning-model stream: thinking tokens first, then visible."""
    from augmentum.modes.coder.chat_egress import StreamProgressTracker

    tracker = StreamProgressTracker()
    tracker.begin(phase="executing", model="r1")

    think_chunk = InternalStreamChunk(thinking_delta="planning...")
    out = tracker.update(think_chunk, phase="executing", model="r1")
    assert out is not None
    assert out.augmentum["status"] == "thinking"
    assert tracker.substate == "thinking"

    # More thinking → no re-emit (transitions only)
    again = tracker.update(
        InternalStreamChunk(thinking_delta="still planning"),
        phase="executing", model="r1",
    )
    assert again is None

    # Visible content → transition to responding
    visible = InternalStreamChunk(content_delta="Here is the answer")
    out = tracker.update(visible, phase="executing", model="r1")
    assert out is not None
    assert out.augmentum["status"] == "responding"
    assert tracker.substate == "responding"


def test_stream_progress_tracker_non_reasoning_skips_thinking():
    """Non-reasoning model: first chunk is visible → straight to responding."""
    from augmentum.modes.coder.chat_egress import StreamProgressTracker

    tracker = StreamProgressTracker()
    tracker.begin(phase="executing", model="gpt-4")

    out = tracker.update(
        InternalStreamChunk(content_delta="Hello"),
        phase="executing", model="gpt-4",
    )
    assert out is not None
    assert out.augmentum["status"] == "responding"


def test_stream_progress_tracker_ignores_empty_chunks():
    """Finish-reason-only / tool-call-only chunks carry neither
    content_delta nor thinking_delta; the tracker must not transition."""
    from augmentum.modes.coder.chat_egress import StreamProgressTracker

    tracker = StreamProgressTracker()
    tracker.begin(phase="executing", model="test")

    empty = InternalStreamChunk(done=True, finish_reason="stop")
    out = tracker.update(empty, phase="executing", model="test")
    assert out is None
    assert tracker.substate == "awaiting_first_token"


def test_stream_progress_tracker_transitions_are_one_shot_per_state():
    """Every chunk of the same class re-checks but only emits on the
    boundary — 1000 consecutive content deltas produce 1 transition."""
    from augmentum.modes.coder.chat_egress import StreamProgressTracker

    tracker = StreamProgressTracker()
    tracker.begin(phase="planning", model="m")

    transitions = 0
    for i in range(50):
        out = tracker.update(
            InternalStreamChunk(content_delta=f"t{i}"),
            phase="planning", model="m",
        )
        if out is not None:
            transitions += 1
    assert transitions == 1


def test_stream_progress_tracker_thinking_and_responding_both_valid_statuses():
    """The three new sub-status strings must live in _VALID_STATUSES
    so emit() doesn't raise when the tracker fires."""
    assert "awaiting_first_token" in _VALID_STATUSES
    assert "thinking" in _VALID_STATUSES
    assert "responding" in _VALID_STATUSES
