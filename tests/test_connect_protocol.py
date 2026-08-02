"""Connect signaling envelope — wire-format round-trips.

Pins the JSON shape every client and server has to agree on. The
envelope is the integration contract between the UI and the
signaling hub; if the round-trip ever drifts, every Connect
client breaks silently.
"""

from __future__ import annotations

import json

from augmentum.connect.protocol import (
    CALL_PROTOCOL_VERSION,
    DEFAULT_INVITE_LIFETIME_MS,
    EVENT_CANDIDATES,
    EVENT_NEGOTIATE,
    EVENT_PONG,
    EVENT_SELECT_ANSWER,
    EVENT_WELCOME,
    MAX_ENVELOPE_BYTES,
    MSG_CANDIDATES,
    MSG_INVITE,
    MSG_NEGOTIATE,
    MSG_PING,
    MSG_SELECT_ANSWER,
    ConnectEnvelope,
    deserialise_envelope,
    serialise_envelope,
)


class TestRoundTrip:
    def test_msg_with_data_round_trips(self) -> None:
        env = ConnectEnvelope(
            kind="msg",
            verb=MSG_INVITE,
            corr_id="req-7",
            peer="alice@instance",
            data={"modalities": "audio,video", "call_id": "c-1"},
        )
        payload = serialise_envelope(env)
        parsed = json.loads(payload)

        # Wire-format spec: type=kind, key=verb, "to" for outbound
        # msgs / "from" for inbound events, "id" for correlation,
        # "data" for payload.
        assert parsed == {
            "type": "msg",
            "msg": MSG_INVITE,
            "id": "req-7",
            "to": "alice@instance",
            "data": {"modalities": "audio,video", "call_id": "c-1"},
        }

        decoded = deserialise_envelope(payload)
        assert decoded == env

    def test_event_uses_from_not_to(self) -> None:
        env = ConnectEnvelope(
            kind="event", verb=EVENT_WELCOME, peer="bob@instance",
            data={"user_did": "bob@instance"},
        )
        parsed = json.loads(serialise_envelope(env))
        assert "from" in parsed
        assert "to" not in parsed
        assert parsed["from"] == "bob@instance"

    def test_omits_empty_optionals(self) -> None:
        # A ping with no correlation token shouldn't emit "id" or
        # "to"/"from" / "data". Keeps the wire payload compact and
        # makes intent obvious in devtools.
        env = ConnectEnvelope(kind="msg", verb=MSG_PING)
        parsed = json.loads(serialise_envelope(env))
        assert parsed == {"type": "msg", "msg": MSG_PING}


class TestDeserialiseTolerance:
    def test_garbage_returns_none(self) -> None:
        assert deserialise_envelope("not json") is None
        assert deserialise_envelope("123") is None  # not an object
        assert deserialise_envelope("null") is None

    def test_missing_type_returns_none(self) -> None:
        assert deserialise_envelope('{"msg": "ping"}') is None

    def test_unknown_type_returns_none(self) -> None:
        # Forward compat for kind happens via "msg"/"event" — anything
        # else is a malformed envelope, not a future verb.
        assert deserialise_envelope('{"type": "cmd", "cmd": "play"}') is None

    def test_unknown_verb_round_trips_unchanged(self) -> None:
        # Forward compat for verbs: server should accept newer
        # constants the deployment doesn't recognise yet. The handler
        # then ignores by string comparison.
        env = deserialise_envelope(
            '{"type": "msg", "msg": "future_verb", "id": "x"}',
        )
        assert env is not None
        assert env.verb == "future_verb"
        assert env.corr_id == "x"

    def test_non_dict_data_falls_back_to_empty(self) -> None:
        # A buggy client sending data as a list shouldn't take the
        # connection down — fall back to empty dict and continue.
        env = deserialise_envelope(
            '{"type": "msg", "msg": "ping", "data": [1, 2, 3]}',
        )
        assert env is not None
        assert env.data == {}


class TestPongShape:
    def test_pong_event_with_corr_id(self) -> None:
        env = ConnectEnvelope(
            kind="event", verb=EVENT_PONG, corr_id="ping-1",
            data={"server_time": 1700000000},
        )
        parsed = json.loads(serialise_envelope(env))
        assert parsed["type"] == "event"
        assert parsed["event"] == EVENT_PONG
        assert parsed["id"] == "ping-1"
        assert parsed["data"]["server_time"] == 1700000000


class TestProtocolConstants:
    def test_version_pinned_at_1(self) -> None:
        # Bumping CALL_PROTOCOL_VERSION requires every signaling
        # implementation to adjust how it routes by version — the
        # constant should not move quietly. If a future change needs
        # to bump it, update this test too.
        assert CALL_PROTOCOL_VERSION == 1

    def test_default_invite_lifetime_is_60s(self) -> None:
        # Matrix MSC2746 ships 60_000ms as the default invite
        # lifetime. We mirror that so cross-implementation behaviour
        # converges on the same UX (caller-side timeout = receiver-
        # side missed-call gap).
        assert DEFAULT_INVITE_LIFETIME_MS == 60_000

    def test_max_envelope_64kb(self) -> None:
        # Cap matches the existing sendBeacon ceiling elsewhere in
        # Augmentum and is large enough for SDPs + batched ICE.
        assert MAX_ENVELOPE_BYTES == 64 * 1024


class TestBatchedCandidates:
    def test_candidates_verb_renamed_from_singular(self) -> None:
        # Per Matrix/LiveKit/Janus convention, ICE candidates are
        # transmitted as a batched array, not one-at-a-time. The
        # verb constant carries the plural to make the wire shape
        # obvious at the call site.
        assert MSG_CANDIDATES == "candidates"
        assert EVENT_CANDIDATES == "candidates"

    def test_candidates_envelope_carries_array(self) -> None:
        # The data payload should be array-shaped — single-candidate
        # senders just wrap in a 1-element list. Empty array is the
        # end-of-gathering sentinel.
        env = ConnectEnvelope(
            kind="msg", verb=MSG_CANDIDATES, peer="bob@h",
            data={
                "call_id": "c-1",
                "party_id": "abcdef12",
                "candidates": [
                    {"candidate": "candidate:1 1 udp ...", "sdpMid": "0",
                     "sdpMLineIndex": 0},
                    {"candidate": "candidate:2 1 udp ...", "sdpMid": "0",
                     "sdpMLineIndex": 0},
                ],
            },
        )
        parsed = json.loads(serialise_envelope(env))
        assert isinstance(parsed["data"]["candidates"], list)
        assert len(parsed["data"]["candidates"]) == 2


class TestMultiDeviceVerbs:
    def test_select_answer_constants_pair(self) -> None:
        # MSG↔EVENT pairs share the same string verb so a router
        # can transcribe kind without re-mapping the verb.
        assert MSG_SELECT_ANSWER == EVENT_SELECT_ANSWER == "select_answer"

    def test_negotiate_constants_pair(self) -> None:
        assert MSG_NEGOTIATE == EVENT_NEGOTIATE == "negotiate"
