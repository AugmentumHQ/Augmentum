"""Tests for the receiver WebSocket message protocol.

Pins:
  - ReceiverCmd / ReceiverEvent roundtrip via serialise + deserialise
  - deserialise tolerates raw JSON strings and bytes
  - deserialise returns None for missing or wrong type
  - forward compat: extra fields are silently dropped
"""
from __future__ import annotations

import json

from augmentum.cast.receiver_protocol import (
    CMD_PLAY,
    CMD_SURFACE_CLOSE,
    CMD_SURFACE_FOCUS,
    CMD_SURFACE_OPEN,
    CMD_SURFACE_STATE,
    EVENT_READY,
    EVENT_SURFACE_CLOSED,
    EVENT_SURFACE_OPENED,
    EVENT_SURFACE_STATE,
    SLOT_COMPANION,
    SLOT_MAIN,
    SLOT_OVERLAY,
    SLOT_PIP,
    SLOT_TICKER,
    SLOTS,
    SURFACE_IMAGE,
    SURFACE_VIDEO,
    ReceiverCmd,
    ReceiverEvent,
    deserialise_cmd,
    deserialise_event,
    is_valid_slot,
    serialise_cmd,
    serialise_event,
)


def test_cmd_roundtrip_via_dict():
    cmd = ReceiverCmd(
        cmd=CMD_PLAY, id="x1",
        args={"url": "/api/cast/render-output/ro_xyz", "title": "Hi"},
    )
    raw = serialise_cmd(cmd)
    assert raw["type"] == "cmd"
    assert raw["cmd"] == "play"
    assert raw["id"] == "x1"
    parsed = deserialise_cmd(raw)
    assert parsed == cmd


def test_cmd_roundtrip_via_json_string():
    cmd = ReceiverCmd(cmd=CMD_PLAY, args={"url": "x"})
    raw = json.dumps(serialise_cmd(cmd))
    parsed = deserialise_cmd(raw)
    assert isinstance(parsed, ReceiverCmd)
    assert parsed.cmd == CMD_PLAY


def test_cmd_roundtrip_via_json_bytes():
    cmd = ReceiverCmd(cmd="stop")
    raw = json.dumps(serialise_cmd(cmd)).encode("utf-8")
    parsed = deserialise_cmd(raw)
    assert isinstance(parsed, ReceiverCmd)
    assert parsed.cmd == "stop"


def test_event_roundtrip():
    event = ReceiverEvent(
        event=EVENT_READY, id="",
        data={"platform": "browser", "screen_w": 1920, "screen_h": 1080},
    )
    raw = serialise_event(event)
    assert raw["type"] == "event"
    assert raw["event"] == "ready"
    parsed = deserialise_event(raw)
    assert parsed == event


def test_deserialise_cmd_rejects_wrong_type():
    """An event payload passed to deserialise_cmd should return None,
    not silently coerce — the type discriminator is part of the
    contract."""
    raw = serialise_event(ReceiverEvent(event=EVENT_READY))
    assert deserialise_cmd(raw) is None


def test_deserialise_event_rejects_wrong_type():
    raw = serialise_cmd(ReceiverCmd(cmd=CMD_PLAY))
    assert deserialise_event(raw) is None


def test_deserialise_cmd_returns_none_for_missing_cmd_field():
    assert deserialise_cmd({"type": "cmd"}) is None


def test_deserialise_event_returns_none_for_missing_event_field():
    assert deserialise_event({"type": "event"}) is None


def test_deserialise_returns_none_for_invalid_input():
    assert deserialise_cmd("not json") is None
    assert deserialise_cmd(b"\x00\x01garbage") is None
    assert deserialise_cmd(None) is None
    assert deserialise_cmd(42) is None
    assert deserialise_event([1, 2, 3]) is None


# ── Surface verbs (Phase A) ───────────────────────────────────────


def test_surface_open_cmd_roundtrip():
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_OPEN, id="op-1",
        args={
            "surface_id": "srf_abc",
            "surface_kind": SURFACE_IMAGE,
            "surface_url": "/api/cast/render-output/ro_xyz",
            "slot": SLOT_MAIN,
            "state": {"url": "/api/cast/render-output/ro_xyz"},
        },
    )
    parsed = deserialise_cmd(serialise_cmd(cmd))
    assert parsed == cmd
    assert parsed.args["surface_kind"] == "media.image"


def test_surface_close_focus_state_cmds():
    for cmd_kind, args in [
        (CMD_SURFACE_CLOSE, {"surface_id": "srf_abc"}),
        (CMD_SURFACE_FOCUS, {"slot": SLOT_PIP}),
        (CMD_SURFACE_STATE, {"surface_id": "srf_abc", "patch": {"paused": True}}),
    ]:
        cmd = ReceiverCmd(cmd=cmd_kind, args=args)
        parsed = deserialise_cmd(serialise_cmd(cmd))
        assert parsed is not None
        assert parsed.cmd == cmd_kind
        assert parsed.args == args


def test_surface_events_roundtrip():
    for event_kind, data in [
        (EVENT_SURFACE_OPENED, {"surface_id": "srf_abc", "kind": SURFACE_VIDEO}),
        (EVENT_SURFACE_CLOSED, {"surface_id": "srf_abc", "reason": "ended"}),
        (EVENT_SURFACE_STATE, {"surface_id": "srf_abc",
                               "state": {"position_s": 42.5, "duration_s": 1800}}),
    ]:
        ev = ReceiverEvent(event=event_kind, data=data)
        parsed = deserialise_event(serialise_event(ev))
        assert parsed == ev


def test_slot_constants_complete():
    """SLOTS tuple must include every named slot constant — protects
    against typo drift where adding a SLOT_X but forgetting SLOTS
    would silently let invalid slot names through is_valid_slot."""
    assert set(SLOTS) == {
        SLOT_MAIN, SLOT_PIP, SLOT_OVERLAY, SLOT_TICKER, SLOT_COMPANION,
    }


def test_is_valid_slot():
    for s in SLOTS:
        assert is_valid_slot(s) is True
    assert is_valid_slot("not-a-slot") is False
    assert is_valid_slot("") is False


def test_unknown_surface_kind_is_just_a_string():
    """The forward-compat property: kind is opaque to the protocol.
    A receiver running an older codebase sees an unknown kind and
    falls through to iframe-loading; the protocol doesn't care."""
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_OPEN,
        args={
            "surface_id": "srf_x",
            "surface_kind": "future.modality.we-havent-thought-of",
            "surface_url": "/ui/future-surface/",
            "slot": SLOT_MAIN,
            "state": {},
        },
    )
    parsed = deserialise_cmd(serialise_cmd(cmd))
    assert parsed is not None
    assert parsed.args["surface_kind"] == "future.modality.we-havent-thought-of"


def test_forward_compat_extra_fields_are_dropped():
    """A peer running a future protocol version may add fields we
    don't recognise. The deserialiser should ignore them, not raise."""
    raw = {
        "type": "cmd",
        "cmd": "play",
        "id": "abc",
        "args": {"url": "x"},
        "schema_version": 99,
        "future_field": "ignored",
    }
    parsed = deserialise_cmd(raw)
    assert isinstance(parsed, ReceiverCmd)
    assert parsed.cmd == "play"
    assert parsed.args == {"url": "x"}
