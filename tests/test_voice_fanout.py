"""Tests for the voice session fanout.

Pins:
  - open_session / close_session lifecycle
  - publish_nowait reaches subscribers; drops oldest on backpressure
  - subscribers see events in sequence order
  - cross-user subscribe is denied (multi-tenant invariant)
  - subscribing to unknown session returns no events (no hang)
  - close_session wakes subscribers cleanly via sentinel
  - VoiceFanoutSocket mirrors send_json/send_text/send_bytes
    while still forwarding to the wrapped websocket
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.voice.fanout import (
    FANOUT_KIND_BYTES,
    FANOUT_KIND_JSON,
    FANOUT_KIND_TEXT,
    VoiceFanout,
    VoiceFanoutSocket,
    wrap_websocket,
)


# ── Session lifecycle ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_and_close_session():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")
    assert fan.session_count() == 1
    await fan.close_session("vs_1")
    assert fan.session_count() == 0


@pytest.mark.asyncio
async def test_close_unknown_session_is_noop():
    fan = VoiceFanout()
    await fan.close_session("vs_nope")  # must not raise


@pytest.mark.asyncio
async def test_double_open_is_logged_but_safe():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")
    await fan.open_session("vs_1", user_id="alice")  # logged warning, no crash
    assert fan.session_count() == 1


# ── Publish / subscribe ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_receives_published_events():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")

    received: list[tuple[str, object]] = []

    async def consumer():
        async for event in fan.subscribe("vs_1", user_id="alice"):
            received.append((event.kind, event.payload))
            if len(received) >= 3:
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer subscribe

    fan.publish_nowait("vs_1", FANOUT_KIND_JSON, {"type": "transcript", "text": "hi"})
    fan.publish_nowait("vs_1", FANOUT_KIND_BYTES, b"\x00\x01")
    fan.publish_nowait("vs_1", FANOUT_KIND_TEXT, "raw")

    await asyncio.wait_for(task, timeout=2.0)
    assert received == [
        (FANOUT_KIND_JSON, {"type": "transcript", "text": "hi"}),
        (FANOUT_KIND_BYTES, b"\x00\x01"),
        (FANOUT_KIND_TEXT, "raw"),
    ]
    await fan.close_session("vs_1")


@pytest.mark.asyncio
async def test_publish_to_closed_session_is_silent():
    fan = VoiceFanout()
    # No open — publish must not raise.
    fan.publish_nowait("vs_nope", FANOUT_KIND_JSON, {"x": 1})


@pytest.mark.asyncio
async def test_multiple_subscribers_each_get_a_copy():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")

    queues: list[list] = [[], []]

    async def consumer(idx: int):
        async for event in fan.subscribe("vs_1", user_id="alice"):
            queues[idx].append(event.payload)
            if len(queues[idx]) >= 2:
                return

    tasks = [asyncio.create_task(consumer(0)), asyncio.create_task(consumer(1))]
    await asyncio.sleep(0)

    fan.publish_nowait("vs_1", FANOUT_KIND_JSON, {"n": 1})
    fan.publish_nowait("vs_1", FANOUT_KIND_JSON, {"n": 2})

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    assert queues[0] == [{"n": 1}, {"n": 2}]
    assert queues[1] == [{"n": 1}, {"n": 2}]
    await fan.close_session("vs_1")


@pytest.mark.asyncio
async def test_close_session_wakes_subscribers():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")

    events: list = []
    finished = asyncio.Event()

    async def consumer():
        async for event in fan.subscribe("vs_1", user_id="alice"):
            events.append(event)
        finished.set()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await fan.close_session("vs_1")
    await asyncio.wait_for(finished.wait(), timeout=2.0)
    assert events == []
    task.cancel()


@pytest.mark.asyncio
async def test_subscribe_unknown_session_returns_immediately():
    """No hang — if the session doesn't exist, the iterator
    completes empty so callers can branch cleanly."""
    fan = VoiceFanout()
    received: list = []

    async def consumer():
        async for event in fan.subscribe("vs_nope", user_id="alice"):
            received.append(event)

    await asyncio.wait_for(consumer(), timeout=2.0)
    assert received == []


@pytest.mark.asyncio
async def test_cross_user_subscribe_denied():
    """Multi-tenant: a different user attempting to subscribe to
    Alice's session sees an empty iterator + a warning is logged."""
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")
    received: list = []

    async def consumer():
        async for event in fan.subscribe("vs_1", user_id="bob"):
            received.append(event)

    await asyncio.wait_for(consumer(), timeout=2.0)
    assert received == []
    # Alice's stream is unaffected — verify by checking a legitimate
    # subscriber still works after the rejected attempt.
    legit_events: list = []

    async def alice_consumer():
        async for event in fan.subscribe("vs_1", user_id="alice"):
            legit_events.append(event)
            if legit_events:
                return

    task = asyncio.create_task(alice_consumer())
    await asyncio.sleep(0)
    fan.publish_nowait("vs_1", FANOUT_KIND_JSON, {"x": 1})
    await asyncio.wait_for(task, timeout=2.0)
    assert len(legit_events) == 1
    await fan.close_session("vs_1")


@pytest.mark.asyncio
async def test_backpressure_does_not_block_publisher():
    """The voice loop calls publish_nowait — it must never block,
    even when a subscriber's queue is full. Test by flooding the
    queue with no consumer draining; publish_nowait still returns
    immediately."""
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")

    # Attach a subscriber but never drain — its queue fills up.
    sub_task = asyncio.create_task(_attach_silent_subscriber(fan, "vs_1"))
    await asyncio.sleep(0)  # let subscribe() register the queue

    # Flood far beyond the default queue maxsize (256). Each call
    # must complete synchronously; if any blocks, the loop hangs.
    import time
    start = time.monotonic()
    for i in range(1000):
        fan.publish_nowait("vs_1", FANOUT_KIND_JSON, {"n": i})
    elapsed = time.monotonic() - start
    # 1000 publishes should take milliseconds, not seconds.
    assert elapsed < 1.0, f"publish_nowait blocked? took {elapsed}s"

    sub_task.cancel()
    with contextlib_suppress(asyncio.CancelledError):
        await sub_task
    await fan.close_session("vs_1")


# Helper: subscribe and idle, never drain the queue.
async def _attach_silent_subscriber(fan: VoiceFanout, session_id: str) -> None:
    async for _event in fan.subscribe(session_id, user_id="alice"):
        # Don't process — let the queue fill.
        await asyncio.sleep(60)  # effectively block

# Tiny inline contextlib.suppress replacement to avoid the import dance.
class contextlib_suppress:
    def __init__(self, *excs):
        self._excs = excs
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return et is not None and issubclass(et, self._excs)


# ── VoiceFanoutSocket proxy ───────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_fanout_socket_mirrors_send_json():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")
    ws = MagicMock()
    ws.send_json = AsyncMock()
    proxy = wrap_websocket(ws, fan, "vs_1")

    received: list = []

    async def consumer():
        async for event in fan.subscribe("vs_1", user_id="alice"):
            received.append((event.kind, event.payload))
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await proxy.send_json({"type": "tts_start"})
    await asyncio.wait_for(task, timeout=1.0)

    # Real WS got the call.
    ws.send_json.assert_awaited_once_with({"type": "tts_start"})
    # Fanout subscriber received a mirror.
    assert received == [(FANOUT_KIND_JSON, {"type": "tts_start"})]
    await fan.close_session("vs_1")


@pytest.mark.asyncio
async def test_voice_fanout_socket_mirrors_send_bytes():
    fan = VoiceFanout()
    await fan.open_session("vs_1", user_id="alice")
    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    proxy = wrap_websocket(ws, fan, "vs_1")

    received: list = []

    async def consumer():
        async for event in fan.subscribe("vs_1", user_id="alice"):
            received.append(event.payload)
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    audio = b"\x89PNG fake audio"
    await proxy.send_bytes(audio)
    await asyncio.wait_for(task, timeout=1.0)

    ws.send_bytes.assert_awaited_once_with(audio)
    assert received == [audio]
    await fan.close_session("vs_1")


@pytest.mark.asyncio
async def test_voice_fanout_socket_passes_through_unknown_attrs():
    """The pipeline calls things like ws.receive_text() and
    ws.client_state — those must pass through to the underlying
    socket unchanged."""
    fan = VoiceFanout()
    ws = MagicMock()
    ws.receive_text = AsyncMock(return_value="hello")
    ws.client_state = "CONNECTED"
    proxy = VoiceFanoutSocket(ws, fan, "vs_1")

    # Method passes through.
    got = await proxy.receive_text()
    assert got == "hello"
    # Attribute passes through.
    assert proxy.client_state == "CONNECTED"
