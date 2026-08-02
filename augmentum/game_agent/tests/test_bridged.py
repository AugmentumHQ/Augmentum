"""BridgedAdapter unit tests — focus on the frame ring buffer.

These tests stub the WebSocket entirely; we exercise the ``_handle_frame``
path directly so the buffer behavior is testable without a live emulator
or a Starlette runtime.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from augmentum.game_agent.surfaces.bridged import BridgedAdapter


class _StubWS:
    """Minimal Starlette WebSocket lookalike for adapter construction."""

    async def send_text(self, _data: str) -> None:
        return None

    async def close(self) -> None:
        return None


def _new_adapter() -> BridgedAdapter:
    return BridgedAdapter(
        websocket=_StubWS(),  # type: ignore[arg-type]
        surface_kind="emulatorjs",
        semantic_inputs=["a", "b", "start"],
        log_schema="test.v1",
    )


def _frame_msg(payload: bytes) -> dict[str, str]:
    return {
        "kind": "frame",
        "png_b64": base64.b64encode(payload).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_snapshot_frames_returns_empty_before_any_arrive() -> None:
    """@example: a fresh adapter has no frames; snapshot_frames returns []."""

    adapter = _new_adapter()
    out = await adapter.snapshot_frames(n=3)
    assert out == []
    assert await adapter.snapshot_frame() is None


@pytest.mark.asyncio
async def test_ring_buffer_keeps_last_n_frames_in_order() -> None:
    """@example: frames arrive in time order, oldest evicted past maxlen.

    The ring buffer's maxlen is 8 by construction; we feed 10 frames and
    confirm that snapshot_frames(n) returns the freshest n, preserving
    their arrival order. The "oldest -> newest" contract the LLM relies
    on is rooted here.
    """

    adapter = _new_adapter()
    # Drive _handle_frame directly. The emit callback is unused for
    # frame messages (only events go through it).
    async def _noop_emit(_payload):  # noqa: ANN001
        return None
    for i in range(10):
        await adapter._handle_frame(_frame_msg(bytes([i])), _noop_emit)

    # Buffer caps at 8: bytes 2..9 survive.
    last_three = await adapter.snapshot_frames(n=3)
    assert last_three == [b"\x07", b"\x08", b"\x09"]

    full = await adapter.snapshot_frames(n=20)
    assert full == [bytes([i]) for i in range(2, 10)]
    assert len(full) == 8


@pytest.mark.asyncio
async def test_snapshot_frame_returns_newest_for_legacy_callers() -> None:
    """@example: the single-frame API returns the latest entry."""

    adapter = _new_adapter()
    async def _noop_emit(_payload):  # noqa: ANN001
        return None
    await adapter._handle_frame(_frame_msg(b"a"), _noop_emit)
    await adapter._handle_frame(_frame_msg(b"b"), _noop_emit)
    await adapter._handle_frame(_frame_msg(b"c"), _noop_emit)
    assert await adapter.snapshot_frame() == b"c"


@pytest.mark.asyncio
async def test_bad_base64_frame_is_dropped_buffer_unchanged() -> None:
    """@example: a malformed frame is logged-and-dropped, not stored.

    ROOT CAUSE:
      An earlier version stored b'' for bad-b64 messages, which the
      slow path then forwarded to the LLM as an empty PNG -- not a
      friendly error mode. Skipping the buffer entirely keeps the
      sequence honest.
    """

    adapter = _new_adapter()
    async def _noop_emit(_payload):  # noqa: ANN001
        return None
    await adapter._handle_frame(_frame_msg(b"ok"), _noop_emit)
    # Inject an obviously bad payload via the same handler path:
    await adapter._handle_frame(
        {"kind": "frame", "png_b64": "!!!not_base64!!!"},
        _noop_emit,
    )
    assert await adapter.snapshot_frames(n=4) == [b"ok"]


def test_snapshot_frames_clamps_zero_and_negative() -> None:
    """@example: n <= 0 returns an empty list (defensive)."""

    adapter = _new_adapter()
    assert asyncio.run(adapter.snapshot_frames(n=0)) == []
    assert asyncio.run(adapter.snapshot_frames(n=-3)) == []


# ── Input delivery: request_id ack tracking ───────────────────────────


class _SendCapturingWS:
    """WebSocket double that records every outbound JSON payload.

    The resolver calls ``ws.send_text(json)``; tests then inspect what
    was sent (request_id, semantic, duration) without involving a real
    network. ``receive_text`` blocks forever -- production tests drive
    the response side via _handle_frame directly rather than feeding
    bytes through the read loop.
    """

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._never_returns: asyncio.Future = asyncio.get_event_loop().create_future()

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        # Never resolves; the read loop's task is cancelled by stop().
        return await self._never_returns

    async def close(self) -> None:
        if not self._never_returns.done():
            self._never_returns.cancel()


def _new_adapter_with_ws(ws) -> BridgedAdapter:
    return BridgedAdapter(
        websocket=ws,
        surface_kind="emulatorjs",
        semantic_inputs=["a", "b", "start"],
        log_schema="pokemon_rs.v1",
    )


@pytest.mark.asyncio
async def test_outbound_resolver_includes_request_id_in_payload() -> None:
    """@example: every dispatched action carries a unique request_id.

    Closes the loop with the iframe / parent stack: the iframe echoes
    request_id on AGENT_INPUT_ACK so the server can resolve its
    pending event. Without an id the delivery guarantee can't function.
    """

    ws = _SendCapturingWS()
    adapter = _new_adapter_with_ws(ws)
    # No emit callback wired: the resolver must still complete (we'll
    # cancel out via the ack event so the timeout never fires).
    resolver = adapter.resolver
    # Resolve the press in a background task; complete it via ack.
    task = asyncio.create_task(resolver.apply("a", 100))
    # Wait until the WS captured the send (loop yields)
    for _ in range(50):
        if ws.sent:
            break
        await asyncio.sleep(0)
    assert ws.sent, "resolver did not send anything to the WS"
    payload = ws.sent[0]
    assert payload["action"] == "a"
    assert payload["duration_ms"] == 100
    assert "request_id" in payload
    assert payload["request_id"].startswith("act-")
    # Resolve the pending event so the task can return.
    rid = payload["request_id"]
    pending = adapter._pending_acks.get(rid)
    assert pending is not None
    pending.set()
    await task
    # And the pending entry is reaped.
    assert rid not in adapter._pending_acks


@pytest.mark.asyncio
async def test_outbound_resolver_returns_quickly_when_ack_arrives() -> None:
    """@example: an ACK that arrives via _handle_frame resolves the wait.

    Mirrors the production path: iframe posts AGENT_INPUT_ACK → parent
    forwards as ``{kind:"event", data:{event:"input_ack", request_id}}``
    → adapter's _handle_frame sees event.request_id and sets the
    pending Event → resolver returns immediately.
    """

    ws = _SendCapturingWS()
    adapter = _new_adapter_with_ws(ws)
    # Start the read loop so _handle_frame is exercised, but we'll
    # bypass it by calling _handle_frame directly.
    received: list = []

    async def _emit(payload):  # noqa: ANN001
        received.append(payload)
    await adapter.start(_emit)
    try:
        task = asyncio.create_task(adapter.resolver.apply("start", 200))
        # Wait until WS captured the dispatch
        for _ in range(50):
            if ws.sent:
                break
            await asyncio.sleep(0)
        rid = ws.sent[0]["request_id"]
        # Simulate the iframe's ACK arriving back through _handle_frame.
        await adapter._handle_frame(
            {
                "kind": "event",
                "data": {
                    "event": "input_ack",
                    "request_id": rid,
                    "button": "start",
                    "held_ms": 200,
                    "tick_count": 12,
                    "effect_score": 300,
                },
            },
            _emit,
        )
        # Resolver should return without hitting the timeout.
        await asyncio.wait_for(task, timeout=2.0)
        # The event was also emitted normally so the agent sees it.
        assert any(
            isinstance(p.data, dict) and p.data.get("event") == "input_ack"
            for p in received
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_outbound_resolver_emits_timeout_event_on_lost_ack() -> None:
    """@example: a missing ACK surfaces ``input_ack_timeout`` to the log.

    ROOT CAUSE for choosing observe-not-retry:
      The iframe MIGHT have processed the press successfully but the
      ACK got lost on the way back. Silently retrying would cause
      double-fire on non-idempotent presses (move-twice, fire-twice).
      Surfacing the timeout to the agent lets it decide on a per-action
      basis whether to re-emit.
    """

    ws = _SendCapturingWS()
    adapter = _new_adapter_with_ws(ws)
    # Shrink the overhead budget so this test runs fast — total
    # timeout for a 10-ms press is then max(0.6, (10 + 30) / 1000) = 0.6 s.
    adapter._ACK_OVERHEAD_MS = 30
    received: list = []

    async def _emit(payload):  # noqa: ANN001
        received.append(payload)
    await adapter.start(_emit)
    try:
        # No ACK will arrive. The resolver should time out at ~0.6 s
        # and emit the input_ack_timeout event.
        await adapter.resolver.apply("a", 10)
        # Find the timeout event in what was emitted.
        timeouts = [
            p for p in received
            if isinstance(p.data, dict) and p.data.get("event") == "input_ack_timeout"
        ]
        assert len(timeouts) == 1
        assert timeouts[0].data["semantic"] == "a"
        assert timeouts[0].data["duration_ms"] == 10
        assert timeouts[0].data["request_id"].startswith("act-")
        # Pending entry is reaped.
        assert not adapter._pending_acks
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_outbound_resolver_uses_unique_request_ids_per_press() -> None:
    """@example: two consecutive presses get distinct request_ids."""

    ws = _SendCapturingWS()
    adapter = _new_adapter_with_ws(ws)

    async def _emit(_p):  # noqa: ANN001
        return None
    await adapter.start(_emit)
    try:
        # Drive press 1, ack, press 2, ack — serialized through the
        # same path the orchestrator's action_worker uses.
        for _i in range(2):
            task = asyncio.create_task(adapter.resolver.apply("a", 50))
            for _ in range(50):
                if ws.sent and len(ws.sent) > _i:
                    break
                await asyncio.sleep(0)
            rid = ws.sent[-1]["request_id"]
            pending = adapter._pending_acks.get(rid)
            assert pending is not None
            pending.set()
            await task
        rids = [s["request_id"] for s in ws.sent]
        assert len(set(rids)) == 2, f"request_ids must be unique: {rids}"
    finally:
        await adapter.stop()
