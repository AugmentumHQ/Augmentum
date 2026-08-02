"""Built-in surface recipes.

Recipes are not executable logic. They are protocol contracts the UI and
future device drivers can use to decide which roles, transports, and
state fields make sense for a surface kind.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_RECIPES: tuple[dict[str, Any], ...] = (
    {
        "kind": "comic.reader.webtoon",
        "label": "Comic / Webtoon Reader",
        "summary": "Phone controls page and scroll state; TV renders token-gated comic pages.",
        "roles": ["controller", "display", "observer"],
        "transports": ["ble_handoff", "authenticated_patch", "public_long_poll", "public_asset_token"],
        "content_ref": {
            "required": ["file_id"],
            "optional": ["provider", "series_id", "chapter_id", "title"],
        },
        "state_schema": {
            "reader": {
                "page": 1,
                "page_count": 0,
                "scroll_ratio": 0.0,
                "zoom": 1.0,
                "mode": "webtoon",
            }
        },
        "device_capabilities": [
            "surface.follow_state@1",
            "display.comic_read@1",
            "input.touch_scroll@1",
        ],
    },
    {
        "kind": "media.watch",
        "label": "Watch Together",
        "summary": "One timeline controls video/audio receivers, local browsers, and media-server sessions.",
        "roles": ["controller", "display", "speaker", "observer"],
        "transports": ["ble_handoff", "authenticated_patch", "public_long_poll", "cast_blob", "provider_remote"],
        "content_ref": {
            "required": ["file_id"],
            "optional": ["media_kind", "subtitle_stream_index", "audio_stream_index"],
        },
        "state_schema": {
            "playback": {
                "position_s": 0.0,
                "duration_s": 0.0,
                "paused": True,
                "rate": 1.0,
                "volume": 1.0,
            }
        },
        "device_capabilities": ["media.video_play@1", "media.audio_play@1", "surface.follow_state@1"],
    },
    {
        "kind": "game.stream.controller",
        "label": "Game Stream",
        "summary": "Augmentum game stream as the display plane; phone/browser controllers patch input state.",
        "roles": ["controller", "display", "spectator"],
        "transports": ["ble_handoff", "webrtc_stream", "authenticated_patch", "public_long_poll"],
        "content_ref": {
            "required": ["game_id"],
            "optional": ["stream_session_id", "profile_id", "save_id"],
        },
        "state_schema": {
            "input": {
                "controllers": {},
                "focus": "player1",
            },
            "stream": {
                "ready": False,
                "latency_ms": 0,
            },
        },
        "device_capabilities": ["surface.follow_state@1", "input.gamepad_bridge@1"],
    },
    {
        "kind": "browser.surface",
        "label": "Browser Surface",
        "summary": "A paired browser tab becomes a controllable viewport without pixel mirroring.",
        "roles": ["controller", "display", "agent"],
        "transports": ["ble_handoff", "authenticated_patch", "public_long_poll", "dom_action_bridge_future"],
        "content_ref": {
            "required": ["url"],
            "optional": ["app_id", "artifact_id", "trusted_origin"],
        },
        "state_schema": {
            "viewport": {
                "scroll_x": 0,
                "scroll_y": 0,
                "scale": 1.0,
            },
            "intent": {},
        },
        "device_capabilities": ["surface.follow_state@1", "display.web_show@1"],
    },
    {
        "kind": "avatar.stage",
        "label": "Avatar Stage",
        "summary": "One surface state drives VRM pose, expression, speech, and stage placement.",
        "roles": ["director", "display", "voice"],
        "transports": ["authenticated_patch", "public_long_poll"],
        "content_ref": {
            "required": ["avatar_id"],
            "optional": ["voice_session_id", "room_id"],
        },
        "state_schema": {
            "avatar": {
                "pose": "idle",
                "expression": "neutral",
                "speaking": False,
            }
        },
        "device_capabilities": ["avatar.set_pose@1", "surface.follow_state@1"],
    },
    {
        "kind": "xr.room",
        "label": "XR Room",
        "summary": "Quest/WebXR spaces subscribe to the same device and media surface graph.",
        "roles": ["headset", "controller", "display", "agent"],
        "transports": ["webxr_session", "authenticated_patch", "public_long_poll"],
        "content_ref": {
            "required": ["room_id"],
            "optional": ["voice_session_id", "artifact_id", "media_file_id"],
        },
        "state_schema": {
            "room": {
                "seat_id": "default",
                "panels": [],
                "anchors": {},
            }
        },
        "device_capabilities": ["surface.follow_state@1", "display.web_show@1"],
    },
)


def list_recipes() -> list[dict[str, Any]]:
    return [deepcopy(r) for r in _RECIPES]


def get_recipe(kind: str) -> dict[str, Any] | None:
    wanted = str(kind or "").strip()
    for recipe in _RECIPES:
        if recipe["kind"] == wanted:
            return deepcopy(recipe)
    return None
