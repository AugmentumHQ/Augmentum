"""Tests for CastInputRegistry — phone↔container routing for cast input.

Pins:
  - attach_container returns a ConnectedContainer and is replace-on-resume
  - attach_phone assigns slot from pad_index under ``index`` routing
  - attach_phone leaves slot=-1 under ``firstpress`` until first press
  - route_input adds slot to outbound frame and writes to container WS
  - firstpress claims next free slot only when a button is pressed
  - route_rumble routes by slot to the owning phone
  - detach is idempotent and releases the slot
  - send failure auto-drops the broken WS
  - close_all closes everything cleanly
"""
from __future__ import annotations

from typing import Any

import pytest

from augmentum.cast.input_bridge import (
    MAX_PADS_PER_SESSION,
    ROUTING_FIRSTPRESS,
    ROUTING_INDEX,
    SLOT_UNCLAIMED,
    CastInputRegistry,
)


# ── Fake WebSocket ────────────────────────────────────────────────


class _FakeWS:
    """Captures send_json + close. Mirrors test_cast_receiver_registry shape."""

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


def _input_frame(buttons: list[int] | None = None) -> dict[str, Any]:
    """Build a minimally-shaped phone→server input frame."""
    return {
        "seq": 1,
        "t_send": 0.0,
        "event": {
            "kind": "gamepad_state",
            "pad_index": 0,
            "buttons": buttons if buttons is not None else [0] * 17,
            "axes": [0.0, 0.0, 0.0, 0.0],
        },
    }


# ── attach_container ──────────────────────────────────────────────


def test_attach_container_stores_session():
    reg = CastInputRegistry()
    ws = _FakeWS()
    c = reg.attach_container(
        ws=ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX, system_id="snes",
    )
    assert c.session_id == "s1"
    assert c.pad_routing == ROUTING_INDEX
    assert reg.get_container("s1") is c


def test_attach_container_replaces_on_resume():
    reg = CastInputRegistry()
    old_ws = _FakeWS()
    reg.attach_container(ws=old_ws, session_id="s1", user_id="u1")
    new_ws = _FakeWS()
    new_c = reg.attach_container(ws=new_ws, session_id="s1", user_id="u1")
    assert reg.get_container("s1") is new_c
    # Old WS is left to its own close path (the prior route handler's
    # finally block); registry just drops the record.


def test_attach_container_normalises_unknown_routing():
    reg = CastInputRegistry()
    c = reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_routing="bogus",
    )
    assert c.pad_routing == ROUTING_INDEX


# ── attach_phone slot claim ───────────────────────────────────────


@pytest.mark.asyncio
async def test_phone_index_routing_claims_slot_on_first_frame():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_routing=ROUTING_INDEX,
    )
    p0 = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    p1 = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=1,
    )
    # Both attach with SLOT_UNCLAIMED — first frame claims.
    await reg.route_input(attachment_id=p0.attachment_id, frame=_input_frame())
    await reg.route_input(attachment_id=p1.attachment_id, frame=_input_frame())
    assert p0.slot == 0
    assert p1.slot == 1


@pytest.mark.asyncio
async def test_phone_index_routing_falls_through_when_taken():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_routing=ROUTING_INDEX,
    )
    a = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    b = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=a.attachment_id, frame=_input_frame())
    await reg.route_input(attachment_id=b.attachment_id, frame=_input_frame())
    assert a.slot == 0
    assert b.slot == 1


def test_phone_firstpress_routing_waits_for_button():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_FIRSTPRESS,
    )
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    assert p.slot == SLOT_UNCLAIMED


def test_phone_attach_no_container_yet():
    """A phone attaching before its container's bridge has dialled in
    keeps SLOT_UNCLAIMED — the index strategy needs the container's
    pad_routing to know it should pre-claim."""
    reg = CastInputRegistry()
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=2,
    )
    assert p.slot == SLOT_UNCLAIMED


# ── route_input ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_input_forwards_with_slot():
    reg = CastInputRegistry()
    container_ws = _FakeWS()
    reg.attach_container(
        ws=container_ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=2,
    )
    # Under index routing, the first frame claims pad_index 2 → slot 2.
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame([1] + [0] * 16),
    )
    assert ok is True
    assert len(container_ws.sent) == 1
    assert container_ws.sent[0]["slot"] == 2


@pytest.mark.asyncio
async def test_route_input_drops_when_no_container():
    reg = CastInputRegistry()
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
    )
    assert ok is False


@pytest.mark.asyncio
async def test_route_input_drops_when_bad_attachment():
    reg = CastInputRegistry()
    ok = await reg.route_input(
        attachment_id="cip_doesnotexist", frame=_input_frame(),
    )
    assert ok is False


@pytest.mark.asyncio
async def test_route_input_firstpress_claims_on_button():
    reg = CastInputRegistry()
    container_ws = _FakeWS()
    reg.attach_container(
        ws=container_ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_FIRSTPRESS,
    )
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )

    # Neutral frame: forwarded with slot=-1, no claim
    await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
    )
    assert phone.slot == SLOT_UNCLAIMED
    assert container_ws.sent[0]["slot"] == SLOT_UNCLAIMED

    # Button press: claims slot 0
    await reg.route_input(
        attachment_id=phone.attachment_id,
        frame=_input_frame([1] + [0] * 16),
    )
    assert phone.slot == 0
    assert container_ws.sent[1]["slot"] == 0


@pytest.mark.asyncio
async def test_route_input_drops_broken_container():
    reg = CastInputRegistry()
    bad_ws = _FakeWS(fail_on_send=True)
    reg.attach_container(ws=bad_ws, session_id="s1", user_id="u1")
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
    )
    assert ok is False
    # Broken container auto-detached
    assert reg.get_container("s1") is None


# ── route_rumble ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_rumble_lands_on_owning_phone():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    phone_ws = _FakeWS()
    phone = reg.attach_phone(
        ws=phone_ws, session_id="s1", user_id="u1", pad_index=1,
    )
    # Claim slot 1 by routing one frame through the registry first.
    await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
    )
    frame = {"kind": "rumble", "slot": 1, "duration_ms": 200,
             "strong": 0.7, "weak": 0.3}
    sent = await reg.route_rumble(session_id="s1", frame=frame)
    assert sent == 1
    # The phone received: forwarded input frame ack (none in this
    # direction), then the rumble. We only assert the rumble is in
    # phone_ws.sent because the input route doesn't write to phone_ws.
    assert frame in phone_ws.sent


@pytest.mark.asyncio
async def test_route_rumble_drops_when_slot_unowned():
    reg = CastInputRegistry()
    reg.attach_container(ws=_FakeWS(), session_id="s1", user_id="u1")
    sent = await reg.route_rumble(
        session_id="s1",
        frame={"kind": "rumble", "slot": 3, "duration_ms": 50,
               "strong": 0.5, "weak": 0.0},
    )
    assert sent == 0


@pytest.mark.asyncio
async def test_route_rumble_drops_when_slot_missing():
    reg = CastInputRegistry()
    reg.attach_container(ws=_FakeWS(), session_id="s1", user_id="u1")
    sent = await reg.route_rumble(
        session_id="s1",
        frame={"kind": "rumble", "duration_ms": 50},
    )
    assert sent == 0


# ── detach ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detach_phone_releases_slot():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=p.attachment_id, frame=_input_frame())
    assert p.slot == 0
    assert await reg.detach_phone(p.attachment_id) is True
    # Re-attaching + routing must claim slot 0 again
    p2 = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=p2.attachment_id, frame=_input_frame())
    assert p2.slot == 0


@pytest.mark.asyncio
async def test_detach_is_idempotent():
    reg = CastInputRegistry()
    reg.attach_container(ws=_FakeWS(), session_id="s1", user_id="u1")
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    assert await reg.detach_phone(p.attachment_id) is True
    assert await reg.detach_phone(p.attachment_id) is False
    assert reg.detach_container("s1") is True
    assert reg.detach_container("s1") is False


@pytest.mark.asyncio
async def test_detach_container_clears_slot_table():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=p.attachment_id, frame=_input_frame())
    assert reg.detach_container("s1") is True
    # New container on same session_id starts with fresh slot table
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    p2 = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=p2.attachment_id, frame=_input_frame())
    assert p2.slot == 0


@pytest.mark.asyncio
async def test_detach_phone_sends_neutral_release_to_container():
    """A phone yanked mid-press must release buttons in the container.

    Without the synthetic neutral frame, the container daemon's
    last-applied state diff would never see a transition back to
    zero, leaving buttons stuck inside the emulator.
    """
    reg = CastInputRegistry()
    c_ws = _FakeWS()
    reg.attach_container(
        ws=c_ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    # Send a "button pressed" frame so the slot gets claimed.
    pressed = _input_frame(buttons=[1] + [0] * 16)
    await reg.route_input(attachment_id=p.attachment_id, frame=pressed)
    assert p.slot == 0
    c_ws.sent.clear()

    # Detach should send a release frame with all buttons zeroed.
    assert await reg.detach_phone(p.attachment_id) is True
    assert len(c_ws.sent) == 1
    release = c_ws.sent[0]
    assert release["slot"] == 0
    assert release["event"]["kind"] == "gamepad_state"
    assert release["event"]["buttons"] == [0] * 17
    assert release["event"]["axes"] == [0.0] * 4


@pytest.mark.asyncio
async def test_detach_phone_no_release_when_slot_unclaimed():
    """Don't send a release frame for a phone that never claimed a slot."""
    reg = CastInputRegistry()
    c_ws = _FakeWS()
    reg.attach_container(
        ws=c_ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_FIRSTPRESS,
    )
    p = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    # Phone never pressed anything — slot stays unclaimed.
    assert p.slot == SLOT_UNCLAIMED
    assert await reg.detach_phone(p.attachment_id) is True
    assert c_ws.sent == []


# ── multi-session isolation ───────────────────────────────────────


@pytest.mark.asyncio
async def test_sessions_dont_share_slots():
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    reg.attach_container(
        ws=_FakeWS(), session_id="s2", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    a = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=0,
    )
    b = reg.attach_phone(
        ws=_FakeWS(), session_id="s2", user_id="u1", pad_index=0,
    )
    await reg.route_input(attachment_id=a.attachment_id, frame=_input_frame())
    await reg.route_input(attachment_id=b.attachment_id, frame=_input_frame())
    assert a.slot == 0
    assert b.slot == 0  # different session, no collision


@pytest.mark.asyncio
async def test_max_pads_per_session():
    """Beyond MAX_PADS_PER_SESSION, slot falls back to SLOT_UNCLAIMED."""
    reg = CastInputRegistry()
    reg.attach_container(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    phones = [
        reg.attach_phone(
            ws=_FakeWS(), session_id="s1", user_id="u1", pad_index=i,
        )
        for i in range(MAX_PADS_PER_SESSION + 1)
    ]
    for p in phones:
        await reg.route_input(
            attachment_id=p.attachment_id, frame=_input_frame(),
        )
    assert {p.slot for p in phones[:MAX_PADS_PER_SESSION]} == set(
        range(MAX_PADS_PER_SESSION),
    )
    assert phones[MAX_PADS_PER_SESSION].slot == SLOT_UNCLAIMED


# ── browser-cast (target_receiver_id) routing ─────────────────────
#
# Phones attached with target_receiver_id=<id> route input to a
# kiosk play surface on a TV receiver instead of an AGSP container.
# Frames fan out as CMD_INPUT_GAMEPAD via the receiver_registry.


class _FakeReceiverRegistry:
    """Minimal stand-in for the real registry — just records `send` calls.

    The real registry has many more methods but route_input only needs
    `send(registration_id, cmd) -> bool`. We can simulate "receiver
    gone" by setting the receiver_id in `_offline`.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []
        self._offline: set[str] = set()

    def take_offline(self, registration_id: str) -> None:
        self._offline.add(registration_id)

    async def send(self, registration_id: str, cmd: Any) -> bool:
        if registration_id in self._offline:
            return False
        self.sent.append((registration_id, cmd))
        return True


@pytest.mark.asyncio
async def test_route_input_dispatches_to_receiver_when_target_set():
    """A phone attached with target_receiver_id routes via the receiver
    registry as a CMD_INPUT_GAMEPAD command instead of via the
    container WS. No container is required for browser-cast."""
    from augmentum.cast.receiver_protocol import CMD_INPUT_GAMEPAD
    reg = CastInputRegistry()
    rr = _FakeReceiverRegistry()
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="", user_id="u1", pad_index=2,
        target_receiver_id="rcv_abc",
    )
    ok = await reg.route_input(
        attachment_id=phone.attachment_id,
        frame=_input_frame(buttons=[1] + [0] * 16),
        receiver_registry=rr,
    )
    assert ok is True
    assert len(rr.sent) == 1
    rcv_id, cmd = rr.sent[0]
    assert rcv_id == "rcv_abc"
    assert cmd.cmd == CMD_INPUT_GAMEPAD
    assert cmd.args["pad_index"] == 2
    assert cmd.args["slot"] == 2
    assert cmd.args["buttons"][0] == 1


@pytest.mark.asyncio
async def test_route_input_to_receiver_requires_registry():
    """Missing receiver_registry is a programming error — return False
    without dispatching so the broken path is visible in logs."""
    reg = CastInputRegistry()
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="", user_id="u1",
        target_receiver_id="rcv_abc",
    )
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
        receiver_registry=None,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_route_input_to_receiver_drops_phone_when_receiver_offline():
    """If receiver_registry.send returns False, the receiver is gone —
    drop the phone so the next attach starts clean rather than queuing
    against a dead target."""
    reg = CastInputRegistry()
    rr = _FakeReceiverRegistry()
    rr.take_offline("rcv_dead")
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="", user_id="u1",
        target_receiver_id="rcv_dead",
    )
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
        receiver_registry=rr,
    )
    assert ok is False
    assert reg.get_phone(phone.attachment_id) is None


@pytest.mark.asyncio
async def test_route_input_to_receiver_ignores_non_gamepad_frames():
    """Echo pings, rumble acks, and other non-state frames don't fan
    out to the iframe — only gamepad_state events carry input the
    receiver shim cares about."""
    reg = CastInputRegistry()
    rr = _FakeReceiverRegistry()
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="", user_id="u1",
        target_receiver_id="rcv_abc",
    )
    # Non-gamepad-state event — should be ignored.
    frame = {"seq": 1, "event": {"kind": "rumble", "value": 0.5}}
    ok = await reg.route_input(
        attachment_id=phone.attachment_id, frame=frame,
        receiver_registry=rr,
    )
    assert ok is False
    assert rr.sent == []


@pytest.mark.asyncio
async def test_route_input_browser_cast_doesnt_touch_container():
    """A browser-cast phone with the same session_id as a container
    should bypass the container entirely. Defensive — today the WS
    handler refuses to set both, but the registry shouldn't rely on
    that invariant."""
    reg = CastInputRegistry()
    container_ws = _FakeWS()
    reg.attach_container(
        ws=container_ws, session_id="s1", user_id="u1",
        pad_routing=ROUTING_INDEX,
    )
    rr = _FakeReceiverRegistry()
    phone = reg.attach_phone(
        ws=_FakeWS(), session_id="s1", user_id="u1",
        target_receiver_id="rcv_abc",
    )
    await reg.route_input(
        attachment_id=phone.attachment_id, frame=_input_frame(),
        receiver_registry=rr,
    )
    # Container saw nothing — receiver saw the frame.
    assert container_ws.sent == []
    assert len(rr.sent) == 1


# ── shutdown ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_all_closes_all_ws():
    reg = CastInputRegistry()
    c_ws = _FakeWS()
    p_ws = _FakeWS()
    reg.attach_container(ws=c_ws, session_id="s1", user_id="u1")
    reg.attach_phone(
        ws=p_ws, session_id="s1", user_id="u1", pad_index=0,
    )
    await reg.close_all()
    assert c_ws.closed is True
    assert p_ws.closed is True
    assert reg.get_container("s1") is None
