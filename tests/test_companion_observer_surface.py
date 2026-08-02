"""Tests for Piece 8' — observer captures surface.* topics.

The observer's hardcoded ``_OBSERVED_TOPICS`` set previously filtered
out everything except chat/mode/tool/voice events. Pieces 8' extends it
to also match ``surface.*`` prefixes so cross-surface activity (browse,
media, image, file, etc.) lands in the recent deque without requiring
custom _handle branches per topic.

These tests verify:
* The filter accepts surface.* topics (and still accepts the legacy set)
* The recent deque captures them with topic + payload + timestamp
* Unrelated topics are still rejected (the prefix is precise)
"""

from __future__ import annotations

import asyncio

import pytest


async def _make_observer():
    """Spin up a real bus + observer attached to a minimal runtime."""
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.observer import BeccaObserver

    class _FakeMemory:
        async def journal(self, *args, **kwargs):
            return 0

    class _FakeRuntime:
        def __init__(self):
            self.bus = PresenceBus()
            self.companion_id = "becca"
            self.memory = _FakeMemory()

    runtime = _FakeRuntime()
    observer = BeccaObserver(runtime)
    await observer.start()
    return runtime, observer


@pytest.mark.asyncio
async def test_observer_captures_surface_browse_opened():
    """surface.browse.opened lands in recent deque."""
    runtime, observer = await _make_observer()
    try:
        await runtime.bus.publish_topic(
            "surface.browse.opened",
            {"url": "https://example.com/article", "user_id": "u1"},
            source_companion_id="becca",
        )
        # Give the observer's task a moment to drain
        await asyncio.sleep(0.05)

        recent = list(runtime.observed_state["recent"])
        topics = [r["topic"] for r in recent]
        assert "surface.browse.opened" in topics
        # Payload preserved
        opened = next(r for r in recent if r["topic"] == "surface.browse.opened")
        assert opened["payload"]["url"] == "https://example.com/article"
        assert opened["payload"]["user_id"] == "u1"
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_captures_other_surface_topics():
    """Any topic with the surface.* prefix lands — not just browse."""
    runtime, observer = await _make_observer()
    try:
        for topic, payload in [
            ("surface.media.played", {"file_id": "fi_1", "user_id": "u1"}),
            ("surface.image.generated", {"id": "img_1", "user_id": "u1"}),
            ("surface.file.imported", {"file_id": "fi_2", "user_id": "u1"}),
            ("surface.coder.turn_failed", {"turn_id": "t1", "user_id": "u1"}),
        ]:
            await runtime.bus.publish_topic(topic, payload, source_companion_id="becca")
        await asyncio.sleep(0.05)

        topics = {r["topic"] for r in runtime.observed_state["recent"]}
        assert "surface.media.played" in topics
        assert "surface.image.generated" in topics
        assert "surface.file.imported" in topics
        assert "surface.coder.turn_failed" in topics
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_still_captures_legacy_topics():
    """Adding the surface.* prefix must not regress the original filter."""
    runtime, observer = await _make_observer()
    try:
        await runtime.bus.publish_topic(
            "chat.turn_started",
            {"mode": "passthrough"},
            source_companion_id="becca",
        )
        await asyncio.sleep(0.05)
        topics = {r["topic"] for r in runtime.observed_state["recent"]}
        assert "chat.turn_started" in topics
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_rejects_unrelated_topics():
    """Topics outside both the legacy set and surface.* prefix drop."""
    runtime, observer = await _make_observer()
    try:
        # Pick a topic the bus might carry but the observer shouldn't aggregate:
        # tick.fired is published by the tick loop every tick, dropping it
        # is intentional so the recent deque isn't all ticks.
        await runtime.bus.publish_topic(
            "tick.fired",
            {"count": 1},
            source_companion_id="becca",
        )
        await asyncio.sleep(0.05)
        topics = {r["topic"] for r in runtime.observed_state["recent"]}
        assert "tick.fired" not in topics
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_safe_no_op_when_no_runtime():
    """emit_safe must be silent when no runtime is attached to app state.
    This is the contract the browse route + future surface emits rely on
    so they never crash the user's read."""
    from augmentum.companion_runtime.bus import emit_safe

    class _FakeAppState:
        pass

    # No companion_runtime attribute → emit returns silently
    await emit_safe(_FakeAppState(), "surface.browse.opened", {"url": "x"})

    # Explicit None → also silent
    state = _FakeAppState()
    state.companion_runtime = None
    await emit_safe(state, "surface.browse.opened", {"url": "x"})
