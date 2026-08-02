"""ConnectHub — presence + routing behaviour.

The hub holds the in-memory side of Connect: who's connected, who
to send a routed envelope to. We exercise it without spinning a
real FastAPI stack — the WS is a thin async stub with a captured
outbound queue.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_INVITE,
    EVENT_PRESENCE_UPDATE,
    ConnectEnvelope,
)


class FakeWS:
    """Minimal stand-in: captures sent payloads, optionally raises."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.sent: list[str] = []
        self.fail_on_send = fail_on_send

    async def send_text(self, payload: str) -> None:
        if self.fail_on_send:
            raise RuntimeError("simulated send failure")
        self.sent.append(payload)


@pytest.mark.asyncio
class TestAttachDetach:
    async def test_attach_records_user(self) -> None:
        hub = ConnectHub()
        ws = FakeWS()

        att = await hub.attach(ws=ws, user_id="u1", user_did="u1@here")

        assert att.user_id == "u1"
        assert hub.is_online("u1")
        assert hub.online_user_ids() == ["u1"]

    async def test_multiple_attachments_per_user_allowed(self) -> None:
        # Desktop + phone for the same user; both should be tracked
        # and routed to.
        hub = ConnectHub()
        ws_a, ws_b = FakeWS(), FakeWS()
        await hub.attach(ws=ws_a, user_id="u1", user_did="u1@here")
        await hub.attach(ws=ws_b, user_id="u1", user_did="u1@here")

        env = ConnectEnvelope(kind="event", verb=EVENT_INVITE, peer="alice")
        delivered = await hub.route_to_user(
            target_user_id="u1", envelope=env,
        )

        assert delivered == 2
        assert len(ws_a.sent) == 1
        assert len(ws_b.sent) == 1

    async def test_detach_removes_user_when_last_connection_gone(self) -> None:
        hub = ConnectHub()
        ws = FakeWS()
        att = await hub.attach(ws=ws, user_id="u1", user_did="u1@here")

        await hub.detach(att.connection_id)

        assert not hub.is_online("u1")
        assert hub.online_user_ids() == []

    async def test_detach_keeps_user_when_other_connections_remain(self) -> None:
        hub = ConnectHub()
        ws_a, ws_b = FakeWS(), FakeWS()
        att_a = await hub.attach(ws=ws_a, user_id="u1", user_did="u1@here")
        await hub.attach(ws=ws_b, user_id="u1", user_did="u1@here")

        await hub.detach(att_a.connection_id)

        assert hub.is_online("u1")

    async def test_detach_unknown_connection_is_noop(self) -> None:
        # Defensive — finally-blocks fire detach unconditionally, so
        # double-detach must not throw.
        hub = ConnectHub()
        await hub.detach("conn-does-not-exist")
        assert hub.online_user_ids() == []


@pytest.mark.asyncio
class TestRouting:
    async def test_route_to_offline_user_returns_zero(self) -> None:
        # The route layer interprets zero as "missed call / not online"
        # and decides whether to persist or error — the hub doesn't
        # make that policy call.
        hub = ConnectHub()
        env = ConnectEnvelope(kind="event", verb=EVENT_INVITE)
        assert await hub.route_to_user(target_user_id="ghost", envelope=env) == 0

    async def test_failed_send_does_not_break_remaining_targets(self) -> None:
        # If one of a user's devices has a wedged WS, the others still
        # get the message. Cleaner than dropping the whole fan-out on
        # the first failure.
        hub = ConnectHub()
        bad = FakeWS(fail_on_send=True)
        good = FakeWS()
        await hub.attach(ws=bad, user_id="u1", user_did="u1@here")
        await hub.attach(ws=good, user_id="u1", user_did="u1@here")

        env = ConnectEnvelope(kind="event", verb=EVENT_INVITE)
        delivered = await hub.route_to_user(target_user_id="u1", envelope=env)

        assert delivered == 1
        assert len(good.sent) == 1


@pytest.mark.asyncio
class TestPresenceBroadcast:
    async def test_new_attachment_notifies_others(self) -> None:
        # Bob is online. Alice connects. Bob should see a
        # presence_update event for Alice.
        hub = ConnectHub()
        bob_ws = FakeWS()
        await hub.attach(ws=bob_ws, user_id="bob", user_did="bob@here")
        bob_ws.sent.clear()  # ignore bob's own presence broadcast

        alice_ws = FakeWS()
        await hub.attach(ws=alice_ws, user_id="alice", user_did="alice@here")

        # Bob should have received exactly one event: alice came online.
        assert len(bob_ws.sent) == 1
        import json as _json

        parsed = _json.loads(bob_ws.sent[0])
        assert parsed["type"] == "event"
        assert parsed["event"] == EVENT_PRESENCE_UPDATE
        assert parsed["data"]["status"] == "online"
        assert parsed["data"]["peer_did"] == "alice@here"

    async def test_second_attachment_for_same_user_does_not_rebroadcast(
        self,
    ) -> None:
        # Bob connects on his phone too — alice shouldn't see a
        # second "bob came online" event.
        hub = ConnectHub()
        alice_ws = FakeWS()
        await hub.attach(ws=alice_ws, user_id="alice", user_did="alice@here")
        bob_ws1 = FakeWS()
        await hub.attach(ws=bob_ws1, user_id="bob", user_did="bob@here")
        alice_ws.sent.clear()  # bob's first connection notice

        bob_ws2 = FakeWS()
        await hub.attach(ws=bob_ws2, user_id="bob", user_did="bob@here")

        assert alice_ws.sent == []

    async def test_last_detachment_broadcasts_offline(self) -> None:
        hub = ConnectHub()
        alice_ws = FakeWS()
        await hub.attach(ws=alice_ws, user_id="alice", user_did="alice@here")
        bob_ws = FakeWS()
        att_b = await hub.attach(ws=bob_ws, user_id="bob", user_did="bob@here")
        alice_ws.sent.clear()

        await hub.detach(att_b.connection_id)

        assert len(alice_ws.sent) == 1
        import json as _json

        parsed = _json.loads(alice_ws.sent[0])
        assert parsed["data"]["status"] == "offline"
        assert parsed["data"]["peer_did"] == "bob@here"
