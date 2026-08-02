"""Tests for ReceiverRegistry.

Pins:
  - attach returns a ConnectedReceiver with unique registration_id
  - detach removes the entry and is idempotent
  - get / list_for_user filter correctly by user_id
  - send writes the serialised cmd JSON to the WS
  - send to a missing/dead receiver returns False (and dead receivers
    auto-detach on send failure)
  - broadcast hits every receiver owned by the user
  - record_event(ready) populates the receiver's info + label
  - subscribe yields matching events
  - close_all closes every WS
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from augmentum.cast.receiver_protocol import (
    CMD_PLAY,
    EVENT_PLAYBACK_PROGRESS,
    EVENT_READY,
    ReceiverCmd,
    ReceiverEvent,
)
from augmentum.cast.receiver_registry import ReceiverRegistry

# ── Fake WebSocket ────────────────────────────────────────────────


class _FakeWS:
    """Minimal WS double — captures send_json calls + tracks close."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed: bool = False
        self.close_code: int | None = None
        self._fail = fail_on_send

    async def send_json(self, obj: dict[str, Any]) -> None:
        if self._fail:
            raise ConnectionError("simulated wire failure")
        self.sent.append(obj)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code


# ── attach / detach ───────────────────────────────────────────────


def test_attach_returns_unique_registration_id():
    reg = ReceiverRegistry()
    a = reg.attach(ws=_FakeWS(), user_id="u1")
    b = reg.attach(ws=_FakeWS(), user_id="u1")
    assert a.registration_id != b.registration_id
    assert a.registration_id.startswith("rcv_")
    assert reg.count() == 2


def test_detach_removes_entry_and_is_idempotent():
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="u1")
    assert reg.detach(r.registration_id) is True
    assert reg.detach(r.registration_id) is False
    assert reg.count() == 0


def test_list_for_user_filters_by_owner():
    reg = ReceiverRegistry()
    a = reg.attach(ws=_FakeWS(), user_id="alice")
    b = reg.attach(ws=_FakeWS(), user_id="bob")
    c = reg.attach(ws=_FakeWS(), user_id="alice")
    alice = reg.list_for_user("alice")
    bob = reg.list_for_user("bob")
    assert sorted(r.registration_id for r in alice) == sorted([a.registration_id, c.registration_id])
    assert [r.registration_id for r in bob] == [b.registration_id]


# ── send ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_writes_serialised_cmd():
    reg = ReceiverRegistry()
    ws = _FakeWS()
    r = reg.attach(ws=ws, user_id="u1")

    ok = await reg.send(r.registration_id, ReceiverCmd(cmd=CMD_PLAY, args={"url": "x"}))
    assert ok is True
    assert len(ws.sent) == 1
    msg = ws.sent[0]
    assert msg["type"] == "cmd"
    assert msg["cmd"] == "play"
    assert msg["args"] == {"url": "x"}


@pytest.mark.asyncio
async def test_send_to_unknown_returns_false():
    reg = ReceiverRegistry()
    ok = await reg.send("rcv_missing", ReceiverCmd(cmd="stop"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_to_broken_ws_detaches_receiver():
    """A WS that raises on send is dead — drop it from the registry
    so subsequent broadcasts don't keep retrying against it."""
    reg = ReceiverRegistry()
    ws = _FakeWS(fail_on_send=True)
    r = reg.attach(ws=ws, user_id="u1")

    ok = await reg.send(r.registration_id, ReceiverCmd(cmd="stop"))
    assert ok is False
    assert reg.get(r.registration_id) is None


# ── broadcast ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_hits_every_receiver_for_user():
    reg = ReceiverRegistry()
    ws_a = _FakeWS()
    ws_b = _FakeWS()
    ws_c = _FakeWS()
    reg.attach(ws=ws_a, user_id="u1")
    reg.attach(ws=ws_b, user_id="u2")  # different user
    reg.attach(ws=ws_c, user_id="u1")

    n = await reg.broadcast("u1", ReceiverCmd(cmd="stop"))
    assert n == 2
    assert len(ws_a.sent) == 1
    assert len(ws_b.sent) == 0  # other user not touched
    assert len(ws_c.sent) == 1


# ── record_event ──────────────────────────────────────────────────


def test_record_event_ready_populates_info_and_label():
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="u1")
    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={"platform": "browser", "label": "Onn 4K Box", "screen_w": 1920},
    ))
    updated = reg.get(r.registration_id)
    assert updated is not None
    assert updated.info["platform"] == "browser"
    assert updated.info["screen_w"] == 1920
    assert updated.label == "Onn 4K Box"


def test_record_event_unknown_receiver_is_noop():
    """An event for a receiver that's already detached must not raise."""
    reg = ReceiverRegistry()
    reg.record_event("rcv_ghost", ReceiverEvent(event=EVENT_READY))


# ── subscribe ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_yields_matching_events():
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="alice")

    received: list[tuple[str, ReceiverEvent]] = []

    async def consumer():
        async for tup in reg.subscribe(user_id="alice"):
            received.append(tup)
            if len(received) >= 2:
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer subscribe

    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_PLAYBACK_PROGRESS, data={"position_s": 10},
    ))
    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_PLAYBACK_PROGRESS, data={"position_s": 15},
    ))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 2
    assert received[0][0] == r.registration_id
    assert received[0][1].data["position_s"] == 10


@pytest.mark.asyncio
async def test_subscribe_user_filter_excludes_others():
    reg = ReceiverRegistry()
    alice_r = reg.attach(ws=_FakeWS(), user_id="alice")
    bob_r = reg.attach(ws=_FakeWS(), user_id="bob")

    received: list[tuple[str, ReceiverEvent]] = []

    async def consumer():
        async for tup in reg.subscribe(user_id="alice"):
            received.append(tup)
            if received:
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    # Bob's event should NOT reach Alice's subscriber.
    reg.record_event(bob_r.registration_id, ReceiverEvent(event=EVENT_PLAYBACK_PROGRESS))
    # Alice's event triggers consumer exit.
    reg.record_event(alice_r.registration_id, ReceiverEvent(event=EVENT_PLAYBACK_PROGRESS))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 1
    assert received[0][0] == alice_r.registration_id


# ── Capability negotiation (Phase C) ──────────────────────────────


def test_receiver_capabilities_empty_until_ready():
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="u1")
    assert reg.receiver_capabilities(r.registration_id) == {}
    assert reg.receiver_supports(r.registration_id, "media.video") is False


def test_receiver_capabilities_populated_by_ready_event():
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="u1")
    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={
            "platform": "android-tv",
            "surface_capabilities": {
                "media.video": {"schema_version": 1, "codecs": ["h264"]},
                "html.generic": {"schema_version": 1},
            },
        },
    ))
    caps = reg.receiver_capabilities(r.registration_id)
    assert "media.video" in caps
    assert "html.generic" in caps
    assert reg.receiver_supports(r.registration_id, "media.video") is True
    assert reg.receiver_supports(r.registration_id, "media.image") is False  # unsupported
    assert reg.receiver_supports(r.registration_id, "") is False  # empty kind


def test_receiver_capabilities_for_unknown_id():
    reg = ReceiverRegistry()
    assert reg.receiver_capabilities("rcv_nope") == {}
    assert reg.receiver_supports("rcv_nope", "media.video") is False


def test_find_receivers_with_capability():
    """Filter user-owned receivers by what they natively render —
    used by future routing to prefer native-fast-path receivers."""
    reg = ReceiverRegistry()
    a = reg.attach(ws=_FakeWS(), user_id="u1")
    b = reg.attach(ws=_FakeWS(), user_id="u1")
    c = reg.attach(ws=_FakeWS(), user_id="other-user")

    reg.record_event(a.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={"surface_capabilities": {"vrm.avatar": {"schema_version": 1}}},
    ))
    reg.record_event(b.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={"surface_capabilities": {"media.video": {"schema_version": 1}}},
    ))
    # c (other user) also advertises vrm — must NOT match u1's filter.
    reg.record_event(c.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={"surface_capabilities": {"vrm.avatar": {"schema_version": 1}}},
    ))

    vrm_receivers = reg.find_receivers_with_capability("u1", "vrm.avatar")
    assert [r.registration_id for r in vrm_receivers] == [a.registration_id]

    video_receivers = reg.find_receivers_with_capability("u1", "media.video")
    assert [r.registration_id for r in video_receivers] == [b.registration_id]

    none = reg.find_receivers_with_capability("u1", "obscure.kind")
    assert none == []


def test_receiver_capabilities_survives_non_ready_events():
    """Subsequent non-ready events must not wipe the capabilities
    set on the original ready — they're stable for the connection."""
    reg = ReceiverRegistry()
    r = reg.attach(ws=_FakeWS(), user_id="u1")
    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_READY,
        data={"surface_capabilities": {"media.video": {"schema_version": 1}}},
    ))
    reg.record_event(r.registration_id, ReceiverEvent(
        event=EVENT_PLAYBACK_PROGRESS, data={"position_s": 10},
    ))
    # Capabilities still there.
    assert reg.receiver_supports(r.registration_id, "media.video") is True


# ── close_all ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_all_closes_every_ws():
    reg = ReceiverRegistry()
    ws_a = _FakeWS()
    ws_b = _FakeWS()
    reg.attach(ws=ws_a, user_id="u1")
    reg.attach(ws=ws_b, user_id="u2")

    await reg.close_all()

    assert ws_a.closed is True
    assert ws_b.closed is True
    assert reg.count() == 0
