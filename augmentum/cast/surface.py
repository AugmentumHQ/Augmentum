"""Surface dispatch helpers — the agnostic cast surface.

This module turns the receiver_registry's low-level ``send(cmd)`` API
into a high-level "put this surface on that receiver" API. Callers
(routes, companion, UI helpers) hand in a kind + url + slot + state;
this module builds the ReceiverCmd and ships it.

Why a separate module:

  receiver_registry owns connection lifecycle + raw send. surface.py
  owns *what to send*. Splitting keeps the registry generic — any
  future cmd shape (e.g. controller leases, mirror handshakes) lands
  here without bloating the connection manager.

Legacy translations (show_image / show_html / play) become thin
wrappers around ``cast_surface`` so a single dispatch path serves
both old and new receivers. Old receivers handle the legacy cmds
directly; new receivers handle ``surface_open``. Either way the
server speaks one language.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from augmentum.cast.receiver_protocol import (
    CMD_SURFACE_CLOSE,
    CMD_SURFACE_FOCUS,
    CMD_SURFACE_OPEN,
    CMD_SURFACE_STATE,
    SLOT_MAIN,
    SURFACE_AUDIO,
    SURFACE_HTML,
    SURFACE_IMAGE,
    SURFACE_VIDEO,
    ReceiverCmd,
    is_valid_slot,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.cast.receiver_registry import ReceiverRegistry

log = get_logger(__name__)


def make_surface_id() -> str:
    """Generate a unique surface_id for a fresh ``surface_open``.

    Caller-controlled (not auto-assigned by the receiver) so the
    sender can refer to the surface in subsequent ``surface_state``
    or ``surface_close`` cmds before any round-trip.
    """
    return f"srf_{secrets.token_hex(8)}"


async def cast_surface(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    surface_kind: str,
    surface_url: str,
    slot: str = SLOT_MAIN,
    state: dict[str, Any] | None = None,
    surface_id: str | None = None,
    cmd_id: str = "",
) -> str | None:
    """Open a surface on a receiver. Returns the ``surface_id`` on
    successful send, ``None`` when the receiver is gone / unreachable.

    Forward-compat: ``surface_kind`` is treated as opaque — unknown
    kinds reach the receiver, which falls back to generic iframe load.
    """
    if not is_valid_slot(slot):
        log.warning("cast_surface_invalid_slot", slot=slot, receiver_id=receiver_id)
        return None
    sid = surface_id or make_surface_id()
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_OPEN,
        id=cmd_id,
        args={
            "surface_id": sid,
            "surface_kind": surface_kind,
            "surface_url": surface_url,
            "slot": slot,
            "state": dict(state or {}),
        },
    )
    ok = await registry.send(receiver_id, cmd)
    if not ok:
        return None
    return sid


async def close_surface(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    surface_id: str,
    cmd_id: str = "",
) -> bool:
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_CLOSE,
        id=cmd_id,
        args={"surface_id": surface_id},
    )
    return await registry.send(receiver_id, cmd)


async def focus_slot(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    slot: str,
    cmd_id: str = "",
) -> bool:
    if not is_valid_slot(slot):
        return False
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_FOCUS,
        id=cmd_id,
        args={"slot": slot},
    )
    return await registry.send(receiver_id, cmd)


async def patch_surface_state(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    surface_id: str,
    patch: dict[str, Any],
    cmd_id: str = "",
) -> bool:
    cmd = ReceiverCmd(
        cmd=CMD_SURFACE_STATE,
        id=cmd_id,
        args={"surface_id": surface_id, "patch": dict(patch or {})},
    )
    return await registry.send(receiver_id, cmd)


# ── Legacy translations ───────────────────────────────────────────


async def cast_image(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    url: str,
    slot: str = SLOT_MAIN,
    state: dict[str, Any] | None = None,
) -> str | None:
    """Legacy ``show_image``-equivalent. Routes through cast_surface
    so old + new receivers share one dispatch path."""
    return await cast_surface(
        registry, receiver_id,
        surface_kind=SURFACE_IMAGE, surface_url=url,
        slot=slot, state={"url": url, **(state or {})},
    )


async def cast_html(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    url: str,
    slot: str = SLOT_MAIN,
    state: dict[str, Any] | None = None,
) -> str | None:
    return await cast_surface(
        registry, receiver_id,
        surface_kind=SURFACE_HTML, surface_url=url,
        slot=slot, state={"url": url, **(state or {})},
    )


async def cast_video(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    url: str,
    slot: str = SLOT_MAIN,
    title: str = "",
    poster_url: str = "",
    state: dict[str, Any] | None = None,
) -> str | None:
    merged_state = {"url": url, "title": title, "poster_url": poster_url}
    if state:
        merged_state.update(state)
    return await cast_surface(
        registry, receiver_id,
        surface_kind=SURFACE_VIDEO, surface_url=url,
        slot=slot, state=merged_state,
    )


async def cast_audio(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    url: str,
    slot: str = SLOT_MAIN,
    title: str = "",
    state: dict[str, Any] | None = None,
) -> str | None:
    merged_state = {"url": url, "title": title}
    if state:
        merged_state.update(state)
    return await cast_surface(
        registry, receiver_id,
        surface_kind=SURFACE_AUDIO, surface_url=url,
        slot=slot, state=merged_state,
    )


async def cast_vrm(
    registry: "ReceiverRegistry",
    receiver_id: str,
    *,
    avatar_id: str = "",
    slot: str = SLOT_MAIN,
    state: dict[str, Any] | None = None,
) -> str | None:
    """Cast the VRM companion surface to a receiver.

    ``avatar_id`` is optional — when empty the surface fetches
    /api/avatar/bundled and picks the first available. State patches
    (emotion, look_at, animation) flow through patch_surface_state
    after the surface is open.
    """
    from augmentum.cast.receiver_protocol import SURFACE_VRM
    qs = f"?avatar_id={avatar_id}" if avatar_id else ""
    merged_state = {"avatar_id": avatar_id}
    if state:
        merged_state.update(state)
    return await cast_surface(
        registry, receiver_id,
        surface_kind=SURFACE_VRM,
        surface_url=f"/ui/cast-vrm/{qs}",
        slot=slot, state=merged_state,
    )
