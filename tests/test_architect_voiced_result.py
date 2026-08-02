"""The companion-lying fix: the architect router must NOT voice its optimistic
pre-narration ("Playing X — say cancel if you'd rather not") over a handler that
never actuated the intent (a library miss / empty / parked clarification).

Root cause was ``architect/router.py`` overriding the handler's speak with
``decision.response_text`` UNCONDITIONALLY. The fix gates the override on
``ActionResult.fulfilled`` via ``_voiced_result``; handlers set ``fulfilled=False``
on miss/park/refuse. These tests lock both halves.
"""
from __future__ import annotations

import asyncio

from augmentum.architect.router import _voiced_result
from augmentum.intent.action import ActionResult, SessionContext
from augmentum.intent.builtin.media import _media_play, _media_recommend


# ── _voiced_result: the gate ────────────────────────────────────────────────

def test_override_wins_when_fulfilled():
    handler = ActionResult(short_circuit=True, fulfilled=True, speak="Starting Dune.")
    out = _voiced_result(handler, "Playing Dune — say cancel if you'd rather not.")
    assert out.speak == "Playing Dune — say cancel if you'd rather not."


def test_honest_line_kept_when_not_fulfilled():
    # THE bug: a miss must keep its honest line, not the optimistic confirmation.
    handler = ActionResult(
        short_circuit=True, fulfilled=False,
        speak="I don't see favorites in your library. I can search YouTube or the web.",
    )
    out = _voiced_result(handler, "Playing your favorites — say cancel if you'd rather not.")
    assert out.speak == handler.speak
    assert "Playing" not in out.speak


def test_no_response_text_leaves_result_untouched():
    handler = ActionResult(short_circuit=True, fulfilled=True, speak="ok")
    assert _voiced_result(handler, "") is handler


def test_override_preserves_surface_emit_clarify_digest():
    # The old manual rebuild silently dropped clarify/digest; replace() keeps them.
    handler = ActionResult(
        short_circuit=True, fulfilled=True, speak="x",
        surface_emit={"channel": "media.resume", "payload": {"file_id": "fi_1"}},
        clarify={"missing": ["q"]}, digest="played dune", toast="Playing Dune",
    )
    out = _voiced_result(handler, "Playing Dune.")
    assert out.speak == "Playing Dune."
    assert out.surface_emit == handler.surface_emit
    assert out.clarify == handler.clarify
    assert out.digest == handler.digest
    assert out.toast == handler.toast


# ── handlers actually set fulfilled=False on the non-actuating paths ─────────

def _anon():
    return SessionContext(user_id="", session_id="s1")


def test_media_play_signed_out_is_not_fulfilled():
    r = asyncio.run(_media_play("", _anon(), {"query": "anything"}))
    assert r.fulfilled is False
    # And the router would keep that honest refusal, not voice "Playing…".
    assert _voiced_result(r, "Playing anything.").speak == r.speak


def test_media_play_no_query_parks_not_fulfilled():
    # A real user but no query → parks "What should I play?" — must not be
    # overridden by an optimistic confirmation.
    r = asyncio.run(_media_play("", SessionContext(user_id="usr_1", session_id="s1"), {}))
    assert r.fulfilled is False
    assert r.speak == "What should I play?"
    assert _voiced_result(r, "Playing your music.").speak == "What should I play?"


def test_media_recommend_signed_out_is_not_fulfilled():
    r = asyncio.run(_media_recommend("", _anon(), {}))
    assert r.fulfilled is False


def test_default_action_result_is_fulfilled():
    # Back-compat: every existing handler that doesn't opt out keeps the
    # router's tier-appropriate language (the common, correct case).
    assert ActionResult().fulfilled is True
