"""Endpointing recovery — the cut-off / hang-too-long tension (2026-06-13).

Two opposite real-world failures of the same machinery:
  - "cuts me off, can't finish my thought" — the continuation merge
    window was gated on is_speaking (flips at llm_start, before audio),
    so a continuation during the ~800ms 'thinking' gap was dispatched
    alone and lost its first half. Fix: gate on tts_started.
  - "feels super long on background noise" — the veto deadline deferred
    forever whenever VAD read speech, which noise satisfies. Fix: cap
    the deferrals.

The deferral-cap decision is extracted as a pure predicate so the
off-by-one is pinned; the flag-timing invariant the merge fix depends
on is pinned on VoiceSession directly.
"""

from __future__ import annotations

from augmentum.proxy.voice_routes import _should_defer_veto
from augmentum.voice.pipeline import VoiceSession

# ── Deferral cap (the "feels super long" fix) ─────────────────────────


def test_defers_while_under_cap_and_speaking():
    assert _should_defer_veto(0, 3, True) is True
    assert _should_defer_veto(2, 3, True) is True


def test_stops_deferring_at_cap():
    # 3 deferrals already taken, cap is 3 → finalize, don't hang further.
    assert _should_defer_veto(3, 3, True) is False
    assert _should_defer_veto(5, 3, True) is False


def test_never_defers_when_vad_silent():
    # No speech at the deadline → finalize regardless of count.
    assert _should_defer_veto(0, 3, False) is False


def test_cap_zero_disables_deferral():
    # max_deferrals=0 → finalize at the first deadline, never extend.
    assert _should_defer_veto(0, 0, True) is False


# ── Flag-timing invariant (the "cuts me off" fix depends on this) ─────


def test_is_speaking_and_tts_started_are_independent():
    s = VoiceSession(session_id="t")
    # Fresh turn: neither set.
    assert s.is_speaking is False
    assert s.tts_started is False
    # llm_start flips is_speaking but NOT tts_started — this is the
    # window the merge gate must stay OPEN through (gate on tts_started,
    # not is_speaking).
    s.is_speaking = True
    assert s.tts_started is False
    # First reply audio flips tts_started — the real "she's audibly
    # replying" boundary where a continuation becomes a barge-in.
    s.tts_started = True
    assert s.is_speaking and s.tts_started


def test_settings_registered():
    from augmentum.config import settings
    from augmentum.proxy.config_routes import _TOOL_SETTINGS
    assert hasattr(settings, "voice_smart_turn_max_deferrals")
    assert "voice_smart_turn_max_deferrals" in _TOOL_SETTINGS
    assert hasattr(settings, "voice_fast_endpoint_ms")
