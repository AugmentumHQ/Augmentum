"""Tests for the surface dispatcher (augmentum/cast/surface.py).

Pins:
  - cast_surface returns surface_id on success, None on send failure
  - invalid slot rejected with None (forward-compat soft fail)
  - registry.send called with the right ReceiverCmd shape
  - close_surface / focus_slot / patch_surface_state route cleanly
  - legacy wrappers (cast_image / cast_html / cast_video / cast_audio)
    produce the correct surface_kind + state shape
  - make_surface_id generates unique values
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.cast.receiver_protocol import (
    CMD_SURFACE_CLOSE,
    CMD_SURFACE_FOCUS,
    CMD_SURFACE_OPEN,
    CMD_SURFACE_STATE,
    SLOT_MAIN,
    SLOT_PIP,
    SURFACE_AUDIO,
    SURFACE_HTML,
    SURFACE_IMAGE,
    SURFACE_VIDEO,
)
from augmentum.cast.surface import (
    cast_audio,
    cast_html,
    cast_image,
    cast_surface,
    cast_video,
    close_surface,
    focus_slot,
    make_surface_id,
    patch_surface_state,
)


def _fake_registry(send_result: bool = True) -> MagicMock:
    reg = MagicMock()
    reg.send = AsyncMock(return_value=send_result)
    return reg


# ── make_surface_id ───────────────────────────────────────────────


def test_make_surface_id_is_unique_and_prefixed():
    ids = {make_surface_id() for _ in range(50)}
    assert len(ids) == 50
    for sid in ids:
        assert sid.startswith("srf_")
        # 8 hex bytes = 16 hex chars after the prefix
        assert len(sid) == len("srf_") + 16


# ── cast_surface ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cast_surface_happy_path():
    reg = _fake_registry()
    sid = await cast_surface(
        reg, "rcv_1",
        surface_kind=SURFACE_HTML,
        surface_url="/ui/comics/?id=42",
        slot=SLOT_MAIN,
        state={"page": 42},
    )
    assert sid is not None
    assert sid.startswith("srf_")
    reg.send.assert_awaited_once()
    arg_rcv, arg_cmd = reg.send.call_args.args
    assert arg_rcv == "rcv_1"
    assert arg_cmd.cmd == CMD_SURFACE_OPEN
    assert arg_cmd.args["surface_kind"] == SURFACE_HTML
    assert arg_cmd.args["surface_url"] == "/ui/comics/?id=42"
    assert arg_cmd.args["slot"] == SLOT_MAIN
    assert arg_cmd.args["state"] == {"page": 42}
    assert arg_cmd.args["surface_id"] == sid


@pytest.mark.asyncio
async def test_cast_surface_returns_none_on_send_failure():
    """Caller distinguishes 'I sent it' from 'the receiver was gone';
    a None return keeps the contract honest."""
    reg = _fake_registry(send_result=False)
    sid = await cast_surface(
        reg, "rcv_dead",
        surface_kind=SURFACE_HTML, surface_url="/ui/x/",
    )
    assert sid is None


@pytest.mark.asyncio
async def test_cast_surface_rejects_invalid_slot():
    """Forward-compat soft fail: unknown slots → None, no send.
    Future receivers may have more slots, but we won't ship one
    we don't recognise."""
    reg = _fake_registry()
    sid = await cast_surface(
        reg, "rcv_1",
        surface_kind=SURFACE_HTML, surface_url="/ui/x/",
        slot="nonexistent-slot",
    )
    assert sid is None
    reg.send.assert_not_called()


@pytest.mark.asyncio
async def test_cast_surface_caller_can_supply_surface_id():
    """Tests / future controller flows pass an explicit surface_id so
    they can refer to it in subsequent patches without a roundtrip."""
    reg = _fake_registry()
    sid = await cast_surface(
        reg, "rcv_1",
        surface_kind=SURFACE_HTML, surface_url="/ui/x/",
        surface_id="srf_caller-controlled",
    )
    assert sid == "srf_caller-controlled"
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_id"] == "srf_caller-controlled"


@pytest.mark.asyncio
async def test_cast_surface_forwards_unknown_kind():
    """Forward-compat: cast_surface doesn't validate kind. A future
    surface_kind passes through verbatim — receivers handle unknowns."""
    reg = _fake_registry()
    sid = await cast_surface(
        reg, "rcv_1",
        surface_kind="future.thing",
        surface_url="/ui/future/",
    )
    assert sid is not None
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == "future.thing"


# ── close_surface ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_surface_sends_correct_cmd():
    reg = _fake_registry()
    ok = await close_surface(reg, "rcv_1", surface_id="srf_x")
    assert ok is True
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.cmd == CMD_SURFACE_CLOSE
    assert arg_cmd.args == {"surface_id": "srf_x"}


# ── focus_slot ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_focus_slot_validates_slot_and_sends_cmd():
    reg = _fake_registry()
    ok = await focus_slot(reg, "rcv_1", slot=SLOT_PIP)
    assert ok is True
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.cmd == CMD_SURFACE_FOCUS
    assert arg_cmd.args == {"slot": SLOT_PIP}


@pytest.mark.asyncio
async def test_focus_slot_rejects_invalid_slot():
    reg = _fake_registry()
    ok = await focus_slot(reg, "rcv_1", slot="garbage")
    assert ok is False
    reg.send.assert_not_called()


# ── patch_surface_state ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_surface_state_sends_correct_cmd():
    reg = _fake_registry()
    ok = await patch_surface_state(
        reg, "rcv_1",
        surface_id="srf_x", patch={"paused": True, "position_s": 42},
    )
    assert ok is True
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.cmd == CMD_SURFACE_STATE
    assert arg_cmd.args["surface_id"] == "srf_x"
    assert arg_cmd.args["patch"] == {"paused": True, "position_s": 42}


# ── Legacy translators ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cast_image_emits_surface_image_kind():
    reg = _fake_registry()
    sid = await cast_image(reg, "rcv_1", url="https://example.com/a.png")
    assert sid is not None
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.cmd == CMD_SURFACE_OPEN
    assert arg_cmd.args["surface_kind"] == SURFACE_IMAGE
    assert arg_cmd.args["surface_url"] == "https://example.com/a.png"
    assert arg_cmd.args["state"]["url"] == "https://example.com/a.png"


@pytest.mark.asyncio
async def test_cast_html_emits_html_generic_kind():
    reg = _fake_registry()
    await cast_html(reg, "rcv_1", url="/ui/comics/?id=42")
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == SURFACE_HTML
    assert arg_cmd.args["state"]["url"] == "/ui/comics/?id=42"


@pytest.mark.asyncio
async def test_cast_video_includes_title_and_poster_in_state():
    reg = _fake_registry()
    await cast_video(
        reg, "rcv_1",
        url="https://example.com/movie.mp4",
        title="The Hobbit",
        poster_url="https://example.com/poster.jpg",
    )
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == SURFACE_VIDEO
    assert arg_cmd.args["state"]["title"] == "The Hobbit"
    assert arg_cmd.args["state"]["poster_url"] == "https://example.com/poster.jpg"


@pytest.mark.asyncio
async def test_cast_audio_emits_audio_kind():
    reg = _fake_registry()
    await cast_audio(reg, "rcv_1", url="https://example.com/song.mp3", title="x")
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == SURFACE_AUDIO
    assert arg_cmd.args["state"]["title"] == "x"


# ── cast_vrm (Phase G) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cast_vrm_emits_vrm_kind_with_default_url():
    """No avatar_id → URL has no query, state carries empty avatar_id
    so the surface knows to fetch /api/avatar/bundled[0] itself."""
    from augmentum.cast.receiver_protocol import SURFACE_VRM
    from augmentum.cast.surface import cast_vrm
    reg = _fake_registry()
    sid = await cast_vrm(reg, "rcv_1")
    assert sid is not None
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == SURFACE_VRM
    assert arg_cmd.args["surface_url"] == "/ui/cast-vrm/"
    assert arg_cmd.args["state"]["avatar_id"] == ""


@pytest.mark.asyncio
async def test_cast_vrm_with_avatar_id_threads_through_url_and_state():
    from augmentum.cast.receiver_protocol import SURFACE_VRM
    from augmentum.cast.surface import cast_vrm
    reg = _fake_registry()
    await cast_vrm(reg, "rcv_1", avatar_id="becca-001")
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["surface_kind"] == SURFACE_VRM
    assert arg_cmd.args["surface_url"] == "/ui/cast-vrm/?avatar_id=becca-001"
    assert arg_cmd.args["state"]["avatar_id"] == "becca-001"


@pytest.mark.asyncio
async def test_cast_vrm_state_merges_with_extra():
    from augmentum.cast.surface import cast_vrm
    reg = _fake_registry()
    await cast_vrm(reg, "rcv_1", avatar_id="x", state={"emotion": "happy"})
    arg_cmd = reg.send.call_args.args[1]
    assert arg_cmd.args["state"]["avatar_id"] == "x"
    assert arg_cmd.args["state"]["emotion"] == "happy"
