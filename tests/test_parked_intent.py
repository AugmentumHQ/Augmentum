"""Parked-intent slot-filling — Companion Agency MVP #2.

A clarifying question PARKS its verb; the user's answer FILLS the slot
via the router's confidence stack instead of re-deriving the intent.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from augmentum.intent.action import ReferentCache, SessionContext
from augmentum.intent.dispatch import (
    PENDING_INTENT_TTL_S,
    get_fresh_pending_intent,
)


def _park(refs, asked_at=None):
    refs.pending_intent = {
        "action_id": "grove.play_matching",
        "args": {},
        "missing": ["query"],
        "question": "What would you like to play?",
        "asked_at": asked_at if asked_at is not None else time.time(),
    }
    return refs.pending_intent


def test_fresh_park_returned():
    refs = ReferentCache()
    pi = _park(refs)
    assert get_fresh_pending_intent(refs) is pi


def test_stale_park_cleared():
    refs = ReferentCache()
    _park(refs, asked_at=time.time() - PENDING_INTENT_TTL_S - 5)
    assert get_fresh_pending_intent(refs) is None
    assert refs.pending_intent is None


def test_empty_cache_none():
    assert get_fresh_pending_intent(ReferentCache()) is None


@pytest.mark.asyncio
async def test_grove_empty_query_parks():
    from augmentum.architect.primitives.grove_match import (
        _grove_play_matching_handler,
    )
    refs = ReferentCache()
    session = SessionContext(user_id="u1", session_id="s1", referents=refs)
    res = await _grove_play_matching_handler("", session, {"query": ""})
    assert "What would you like to play" in res.speak
    pi = refs.pending_intent
    assert pi is not None
    assert pi["action_id"] == "grove.play_matching"
    assert pi["missing"] == ["query"]


@pytest.mark.asyncio
async def test_media_play_empty_query_parks():
    from augmentum.intent.builtin.media import _media_play
    refs = ReferentCache()
    session = SessionContext(
        user_id="u1", session_id="s1", referents=refs,
        app_state=SimpleNamespace(file_index=None),
    )
    res = await _media_play("", session, {})
    assert "What should I play" in res.speak
    assert refs.pending_intent["action_id"] == "media.play"


@pytest.mark.asyncio
async def test_dispatch_clears_park_but_not_fresh_repark():
    # The registry bridge clears a pre-existing park on success, but a
    # handler that JUST parked (clarify result) must keep its park.
    from augmentum.companion_runtime.tool_protocol import ToolCall
    from augmentum.companion_runtime.tools import execute_tool

    class FakeBus:
        async def publish_topic(self, *a, **kw):
            pass

    app_state = SimpleNamespace()
    rt = SimpleNamespace(
        _app_state=app_state, companion_id="becca",
        bus=FakeBus(), state_manager=None,
    )
    from augmentum.intent.dispatch import get_referent_cache
    refs = get_referent_cache(app_state, "u1", "s1")
    _park(refs)

    # Successful real dispatch (navigate has no clarify path) clears it.
    call = ToolCall(
        kind="tool", name="navigate.open_surface",
        args={"surface": "notes"}, raw="", span=(0, 0),
    )
    r = await execute_tool(call, rt, user_id="u1", session_id="s1")
    assert r.ok
    assert refs.pending_intent is None

    # A clarify dispatch (media.play, no query) writes a FRESH park —
    # the clear must not erase it.
    call2 = ToolCall(
        kind="tool", name="media.play", args={}, raw="", span=(0, 0),
    )
    r2 = await execute_tool(call2, rt, user_id="u1", session_id="s1")
    assert r2.ok
    assert refs.pending_intent is not None
    assert refs.pending_intent["action_id"] == "media.play"


def test_router_stack_renders_pending():
    from augmentum.architect.router import ConfidenceStack, _format_signals
    stack = ConfidenceStack(
        pending_intent_id="grove.play_matching",
        pending_intent_missing=["query"],
        pending_intent_question="What would you like to play?",
    )
    text = _format_signals(stack)
    assert "PENDING CLARIFICATION" in text
    assert "grove.play_matching" in text


# ---------------------------------------------------------------------------
# Continuity v2 — handler-level clarify parking (ActionResult.clarify)
# ---------------------------------------------------------------------------


class TestParkClarify:
    def test_basic_shape(self):
        from augmentum.intent.dispatch import park_clarify
        refs = ReferentCache()
        park_clarify(
            refs,
            action_id="weather.today",
            args={"remember_home": "false"},
            clarify={"missing": ["location"]},
            question="What city should I use?",
        )
        pi = refs.pending_intent
        assert pi["action_id"] == "weather.today"
        assert pi["missing"] == ["location"]
        assert pi["args"] == {"remember_home": "false"}
        assert pi["question"] == "What city should I use?"
        assert get_fresh_pending_intent(refs) is pi

    def test_args_override_merge(self):
        # A clarify can scrub a bad value (geocode-miss clears the
        # unresolvable location so it doesn't stick to the refill).
        from augmentum.intent.dispatch import park_clarify
        refs = ReferentCache()
        park_clarify(
            refs,
            action_id="weather.today",
            args={"location": "zzzville", "remember_home": "true"},
            clarify={"missing": ["location"], "args": {"location": ""}},
            question="I couldn't find a place called zzzville.",
        )
        assert refs.pending_intent["args"] == {
            "location": "", "remember_home": "true",
        }

    def test_noop_on_none_refs_and_bad_clarify(self):
        from augmentum.intent.dispatch import park_clarify
        park_clarify(
            None, action_id="x", args={}, clarify={"missing": []}, question="",
        )  # must not raise
        refs = ReferentCache()
        park_clarify(
            refs, action_id="x", args={}, clarify="not-a-dict",  # type: ignore[arg-type]
            question="",
        )
        assert refs.pending_intent is None


class TestWeatherClarify:
    @pytest.mark.asyncio
    async def test_no_home_ask_carries_clarify(self, monkeypatch):
        from augmentum.config import settings as app_settings
        from augmentum.intent.builtin.weather import _weather_today_handler
        monkeypatch.setattr(app_settings, "location", "", raising=False)
        session = SessionContext(
            user_id="u1", session_id="s1", referents=ReferentCache(),
            app_state=SimpleNamespace(settings_store=None),
        )
        res = await _weather_today_handler("", session, {})
        assert "what city" in res.speak.lower()
        assert res.clarify == {"missing": ["location"]}

    @pytest.mark.asyncio
    async def test_geocode_miss_carries_clarify(self, monkeypatch):
        from augmentum.intent.builtin.weather import _weather_today_handler
        from augmentum.sources import open_meteo

        async def _no_place(_q):
            return None

        monkeypatch.setattr(open_meteo, "geocode", _no_place)
        session = SessionContext(
            user_id="u1", session_id="s1", referents=ReferentCache(),
            app_state=SimpleNamespace(settings_store=None),
        )
        res = await _weather_today_handler(
            "", session, {"location": "zzzville"},
        )
        assert "couldn't find" in res.speak.lower()
        assert res.clarify["missing"] == ["location"]
        # The unresolvable value is scrubbed so it can't stick.
        assert res.clarify["args"] == {"location": ""}

    @pytest.mark.asyncio
    async def test_execute_tool_parks_weather_clarify(self, monkeypatch):
        # End-to-end through the native-loop registry path: the
        # handler's clarify result must land in pending_intent and
        # survive the dispatch-resolves-park identity clear.
        from augmentum.companion_runtime.tool_protocol import ToolCall
        from augmentum.companion_runtime.tools import execute_tool
        from augmentum.config import settings as app_settings
        from augmentum.intent.dispatch import get_referent_cache

        monkeypatch.setattr(app_settings, "location", "", raising=False)

        class FakeBus:
            async def publish_topic(self, *a, **kw):
                pass

        app_state = SimpleNamespace(settings_store=None)
        rt = SimpleNamespace(
            _app_state=app_state, companion_id="becca",
            bus=FakeBus(), state_manager=None,
        )
        refs = get_referent_cache(app_state, "u1", "s1")
        call = ToolCall(
            kind="tool", name="weather.today", args={}, raw="", span=(0, 0),
        )
        r = await execute_tool(call, rt, user_id="u1", session_id="s1")
        assert r.ok
        pi = refs.pending_intent
        assert pi is not None
        assert pi["action_id"] == "weather.today"
        assert pi["missing"] == ["location"]


class TestRosterPin:
    def test_pin_survives_relevance_clipping(self):
        # "Springfield, Illinois" shares no vocabulary with
        # weather.today — relevance ranking deferred it out of the
        # roster on the clarify-answer turn (companion_eval,
        # 2026-06-11). The pin must force it in.
        from augmentum.companion_runtime.tools import enumerate_tools
        names = [
            t["name"]
            for t in enumerate_tools(
                "Springfield, Illinois", pin=("weather.today",),
            )
        ]
        assert "weather.today" in names

    def test_pending_pin_reads_fresh_park(self):
        from augmentum.companion_runtime.tools import pending_pin
        from augmentum.intent.dispatch import get_referent_cache
        app_state = SimpleNamespace()
        refs = get_referent_cache(app_state, "u1", "s1")
        _park(refs)
        assert pending_pin(app_state, "u1", "s1") == ("grove.play_matching",)
        refs.pending_intent = None
        assert pending_pin(app_state, "u1", "s1") == ()


class TestOpenThreadGate:
    def test_pending_park_opens_thread(self):
        from augmentum.proxy.voice_routes import _is_open_thread
        refs = ReferentCache()
        _park(refs)
        assert _is_open_thread(refs, "Sure thing.") is True

    def test_question_mark_opens_thread(self):
        from augmentum.proxy.voice_routes import _is_open_thread
        assert _is_open_thread(
            ReferentCache(), "What city should I use? ",
        ) is True

    def test_statement_no_park_closed(self):
        from augmentum.proxy.voice_routes import _is_open_thread
        assert _is_open_thread(ReferentCache(), "Here you go.") is False
        assert _is_open_thread(ReferentCache(), "") is False

    def test_stale_park_closed(self):
        from augmentum.proxy.voice_routes import _is_open_thread
        refs = ReferentCache()
        _park(refs, asked_at=time.time() - PENDING_INTENT_TTL_S - 5)
        assert _is_open_thread(refs, "Here you go.") is False
