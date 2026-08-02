"""Persistence and contracts for Augmentum's WebXR app surface.

The browser runtime owns WebXR APIs; the server owns durable context:
session identity, room/seat state, resume snapshots, and telemetry.  Keeping
that state user-scoped lets a Quest Browser/PWA session feel app-like without
letting shared-headset users see each other's spatial room.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import aiosqlite

DEFAULT_ROOM_ID = "modern-room"
DEFAULT_SEAT_ID = "default"

XR_SURFACES: list[dict[str, Any]] = [
    {
        "id": "chat",
        "label": "Chat",
        "action": "chat",
        "panelKind": "transcript",
        "placement": "left-near",
        "hubHint": "conversation + pins",
        "embedUrl": "/ui/?xrEmbed=1&mode=passthrough&xrSurface=chat",
        "voiceCue": "Ask questions, continue the thread, or pin an answer.",
        "primaryActions": ["reply", "summarize", "pin"],
        "contextSources": ["active_chat_session", "voice_transcript"],
    },
    {
        "id": "analytical",
        "label": "Analyze",
        "action": "analytical",
        "panelKind": "reasoning-board",
        "placement": "left-stage",
        "hubHint": "research + reasoning",
        "embedUrl": "/ui/?xrEmbed=1&mode=analytical&xrSurface=analytical",
        "voiceCue": "Analyze sources, compare evidence, and explain the reasoning.",
        "primaryActions": ["search", "compare", "explain"],
        "contextSources": ["active_chat_session", "reasoning_trace", "document_context"],
    },
    {
        "id": "agentic",
        "label": "Build",
        "action": "agentic",
        "panelKind": "task-board",
        "placement": "right-stage",
        "hubHint": "tasks + execution",
        "embedUrl": "/ui/?xrEmbed=1&mode=agentic&xrSurface=agentic",
        "voiceCue": "Plan work, track steps, and execute agentic tasks.",
        "primaryActions": ["plan", "execute", "check_status"],
        "contextSources": ["active_task", "tool_runs", "progress_state"],
    },
    {
        "id": "narrative",
        "label": "Story",
        "action": "narrative",
        "panelKind": "character-stage",
        "placement": "center-stage",
        "hubHint": "characters + scene",
        "embedUrl": "/ui/?xrEmbed=1&mode=narrative&xrSurface=narrative",
        "voiceCue": "Bring characters into the call and continue the scene.",
        "primaryActions": ["continue_scene", "switch_speaker", "summarize_scene"],
        "contextSources": ["active_narrative_session", "characters", "scene_state"],
    },
    {
        "id": "files",
        "label": "Files",
        "action": "files",
        "panelKind": "document-shelf",
        "placement": "left-shelf",
        "hubHint": "docs + context",
        "embedUrl": "/ui/?xrEmbed=1&surface=files&xrSurface=files",
        "voiceCue": "Open documents, attach context, or compare sources.",
        "primaryActions": ["open", "attach", "compare"],
        "contextSources": ["selected_files", "recent_files", "document_context"],
    },
    {
        "id": "browse",
        "label": "Browse",
        "action": "browse",
        "panelKind": "research-wall",
        "placement": "right-wall",
        "hubHint": "search + sources",
        "embedUrl": "/ui/?xrEmbed=1&surface=browse&xrSurface=browse",
        "voiceCue": "Search, summarize a page, or save a source.",
        "primaryActions": ["search", "summarize_page", "save_source", "play_media"],
        "contextSources": ["current_page", "search_results", "saved_sources", "media_handoff"],
    },
    {
        "id": "coder",
        "label": "Coder",
        "action": "coder",
        "panelKind": "workbench",
        "placement": "right-near",
        "hubHint": "plans + diffs + tests",
        "embedUrl": "/ui/?xrEmbed=1&mode=coder&xrSurface=coder",
        "voiceCue": "Inspect a plan, review diffs, approve commands, or run checks.",
        "primaryActions": ["show_plan", "review_diff", "run_checks", "approve"],
        "contextSources": ["active_workspace", "coder_run", "diff", "terminal"],
    },
    {
        "id": "notes",
        "label": "Notes",
        "action": "notes",
        "panelKind": "notebook",
        "placement": "left-desk",
        "hubHint": "dictation + clips",
        "embedUrl": "/ui/?xrEmbed=1&surface=notes&xrSurface=notes",
        "voiceCue": "Dictate notes, clip findings, and organize takeaways.",
        "primaryActions": ["dictate", "clip", "organize"],
        "contextSources": ["active_note", "recent_notes", "clips"],
    },
    {
        "id": "studio",
        "label": "Studio",
        "action": "studio",
        "panelKind": "canvas-wall",
        "placement": "far-wall",
        "hubHint": "images + artifacts",
        "embedUrl": "/ui/?xrEmbed=1&surface=studio&xrSurface=studio",
        "voiceCue": "Generate, inspect, edit, and save visual artifacts.",
        "primaryActions": ["generate", "variant", "edit", "save"],
        "contextSources": ["active_artifact", "image_job", "studio_selection"],
    },
    {
        "id": "media",
        "label": "Media",
        "action": "media",
        "panelKind": "theater",
        "placement": "far-center",
        "hubHint": "watch + read + listen",
        "embedUrl": "/ui/?xrEmbed=1&surface=media&xrSurface=media",
        "voiceCue": "Navigate shows, movies, comics, audiobooks, images, local files, and games.",
        "primaryActions": [
            "continue",
            "shows_movies",
            "comics",
            "audiobooks",
            "images",
            "local_files",
            "games",
        ],
        "contextSources": [
            "now_playing",
            "media_queue",
            "captions",
            "comics",
            "audiobooks",
            "local_images",
            "game_library",
        ],
    },
    {
        "id": "devices",
        "label": "Devices",
        "action": "devices",
        "panelKind": "control-console",
        "placement": "right-console",
        "hubHint": "cast + pair",
        "embedUrl": "/ui/?xrEmbed=1&surface=devices&xrSurface=devices",
        "voiceCue": "Cast, control playback, pair devices, and monitor sessions.",
        "primaryActions": ["cast", "volume", "pair", "stop"],
        "contextSources": ["connected_devices", "cast_sessions"],
    },
    {
        "id": "games",
        "label": "Games",
        "action": "games",
        "panelKind": "arcade",
        "placement": "far-right",
        "hubHint": "launch + stream",
        "embedUrl": "/ui/?xrEmbed=1&surface=games&xrSurface=games",
        "voiceCue": "Launch games, manage streaming sessions, and keep voice nearby.",
        "primaryActions": ["launch", "resume", "controller_mode", "stop_stream"],
        "contextSources": ["game_library", "active_stream", "save_state"],
    },
]

DEFAULT_SEAT: dict[str, Any] = {
    "id": DEFAULT_SEAT_ID,
    "label": "Default seat",
    "x": -0.30,
    "y": 0.0,
    "z": 2.30,
    "rotY": -1.5707963267948966,
    "envId": DEFAULT_ROOM_ID,
    "avatar": {
        "x": 0.75,
        "y": -0.24,
        "z": 1.95,
        "rotY": 3.11,
    },
    "metadata": {},
}

ROOM_MANIFESTS: dict[str, dict[str, Any]] = {
    "modern-room": {
        "id": "modern-room",
        "label": "Modern Room",
        "assetUrl": "/ui/lib/scenes/modern-room.glb",
        "provides": ["standing", "sit-couch", "floor-clearance"],
        "anchors": {
            "avatarSeat": {"x": 0.75, "y": -0.24, "z": 1.95, "rotY": 3.11},
            "transcriptPanel": {"x": -0.55, "y": 1.25, "z": 0.95, "rotY": 0.18},
            "toolPanel": {"x": 0.90, "y": 1.20, "z": 1.05, "rotY": -0.25},
            "controlsDock": {"x": 0.0, "y": 0.95, "z": 0.70, "rotY": 0.0},
            "hubPanel": {"x": -0.42, "y": 1.02, "z": -1.06, "rotY": 0.16},
        },
    },
    "none": {
        "id": "none",
        "label": "Void",
        "assetUrl": "",
        "provides": ["standing"],
        "anchors": {},
    },
}

DEFAULT_INPUT_PREFERENCES: dict[str, Any] = {
    "hands": True,
    "controllers": True,
    "oneHanded": True,
    "ray": True,
    "poke": True,
    "pinchSelect": True,
    "directTouch": True,
    "twoHandResize": True,
    "palmSummon": True,
    "gripRecenter": True,
}

DEFAULT_ROOM_STATE: dict[str, Any] = {
    "activeSurface": "voice",
    "selectedPanelAction": "",
    "openSurfaces": ["voice"],
    "surfacePanels": {},
    "hub": {
        "enabled": True,
        "selectedSurface": "voice",
        "layout": "seat-relative-dock",
    },
}

DEFAULT_PERFORMANCE_PROFILE: dict[str, Any] = {
    "targetRefreshHz": [72, 90],
    "framebufferScale": 0.85,
    "fixedFoveation": "medium",
    "preferWorldLockedText": True,
    "metrics": ["fpsBucket", "droppedFrameStreak", "assetLoadMs", "voiceWsReconnects"],
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row)


def _normalize_room_id(room_id: str | None) -> str:
    rid = (room_id or DEFAULT_ROOM_ID).strip() or DEFAULT_ROOM_ID
    return rid if rid in ROOM_MANIFESTS else DEFAULT_ROOM_ID


def _normalize_seat_id(seat_id: str | None) -> str:
    return (seat_id or DEFAULT_SEAT_ID).strip()[:80] or DEFAULT_SEAT_ID


def _seat_from_row(row: dict[str, Any] | None, seat_id: str = DEFAULT_SEAT_ID) -> dict[str, Any]:
    if not row:
        seat = dict(DEFAULT_SEAT)
        seat["id"] = seat_id or DEFAULT_SEAT_ID
        return seat
    avatar: dict[str, Any] = {}
    if row.get("avatar_x") is not None:
        avatar["x"] = row.get("avatar_x")
    if row.get("avatar_y") is not None:
        avatar["y"] = row.get("avatar_y")
    if row.get("avatar_z") is not None:
        avatar["z"] = row.get("avatar_z")
    if row.get("avatar_rot_y") is not None:
        avatar["rotY"] = row.get("avatar_rot_y")
    return {
        "id": row.get("id") or DEFAULT_SEAT_ID,
        "label": row.get("label") or "Default seat",
        "x": float(row.get("x") or 0.0),
        "y": float(row.get("y") or 0.0),
        "z": float(row.get("z") or 0.0),
        "rotY": float(row.get("rot_y") or 0.0),
        "envId": row.get("env_id") or DEFAULT_ROOM_ID,
        "avatar": avatar or dict(DEFAULT_SEAT["avatar"]),
        "metadata": _json_loads(row.get("metadata_json"), {}),
    }


def _inflate_session(row: dict[str, Any], seat: dict[str, Any] | None = None) -> dict[str, Any]:
    room_id = _normalize_room_id(row.get("room_id"))
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id"),
        "voice_session_id": row.get("voice_session_id") or "",
        "surface": row.get("surface") or "voice",
        "room_id": room_id,
        "seat_id": row.get("seat_id") or DEFAULT_SEAT_ID,
        "status": row.get("status") or "preflight",
        "device_hint": _json_loads(row.get("device_hint_json"), {}),
        "room_state": _json_loads(row.get("room_state_json"), dict(DEFAULT_ROOM_STATE)),
        "input_preferences": _json_loads(
            row.get("input_preferences_json"),
            dict(DEFAULT_INPUT_PREFERENCES),
        ),
        "performance_profile": _json_loads(
            row.get("performance_profile_json"),
            dict(DEFAULT_PERFORMANCE_PROFILE),
        ),
        "last_snapshot": _json_loads(row.get("last_snapshot_json"), {}),
        "room_manifest": ROOM_MANIFESTS[room_id],
        "surface_catalog": list(XR_SURFACES),
        "seat_layout": seat,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


class XRSessionStore:
    """User-scoped persistence for immersive browser/PWA sessions."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = aiosqlite.Row

    async def capabilities(self) -> dict[str, Any]:
        return {
            "ok": True,
            "server": {
                "sessionApi": 1,
                "resumeSnapshots": True,
                "seatCalibration": True,
                "telemetryEvents": True,
                "surfaceHub": True,
                "spatialPanels": True,
                "handPanelGestures": True,
                "webEmbeds": True,
            },
            "webxr": {
                "sessionMode": "immersive-vr",
                "sessionModes": ["immersive-vr", "immersive-ar"],
                "requiredFeatures": ["local-floor"],
                "optionalFeatures": [
                    "bounded-floor",
                    "hand-tracking",
                    "dom-overlay",
                    "layers",
                    "medium-fixed-foveation-level",
                ],
                "mixedReality": {
                    "sessionMode": "immersive-ar",
                    "requiredFeatures": ["local-floor"],
                    "optionalFeatures": [
                        "bounded-floor",
                        "hand-tracking",
                        "dom-overlay",
                        "hit-test",
                        "anchors",
                        "plane-detection",
                        "mesh-detection",
                        "depth-sensing",
                        "layers",
                        "secondary-views",
                    ],
                    "fallbackRequiredFeatures": ["local"],
                },
            },
            "rooms": list(ROOM_MANIFESTS.values()),
            "surfaces": list(XR_SURFACES),
            "defaultRoomId": DEFAULT_ROOM_ID,
            "defaultSeatId": DEFAULT_SEAT_ID,
            "roomStateDefaults": dict(DEFAULT_ROOM_STATE),
            "inputDefaults": dict(DEFAULT_INPUT_PREFERENCES),
            "performanceDefaults": dict(DEFAULT_PERFORMANCE_PROFILE),
        }

    async def get_seat(
        self, *, user_id: str, seat_id: str = DEFAULT_SEAT_ID
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("get_seat requires user_id")
        sid = _normalize_seat_id(seat_id)
        cur = await self._conn.execute(
            "SELECT * FROM xr_seats WHERE user_id = ? AND id = ?",
            (user_id, sid),
        )
        row = await cur.fetchone()
        return _seat_from_row(_row_dict(row) if row else None, sid)

    async def upsert_seat(
        self,
        *,
        user_id: str,
        seat_id: str = DEFAULT_SEAT_ID,
        label: str = "Default seat",
        x: float = -0.30,
        y: float = 0.0,
        z: float = 2.30,
        rot_y: float = 3.141592653589793,
        env_id: str = DEFAULT_ROOM_ID,
        avatar: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("upsert_seat requires user_id")
        sid = _normalize_seat_id(seat_id)
        avatar = avatar or {}
        await self._conn.execute(
            """INSERT INTO xr_seats
               (id, user_id, label, x, y, z, rot_y, env_id,
                avatar_x, avatar_y, avatar_z, avatar_rot_y, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, id) DO UPDATE SET
                 label = excluded.label,
                 x = excluded.x,
                 y = excluded.y,
                 z = excluded.z,
                 rot_y = excluded.rot_y,
                 env_id = excluded.env_id,
                 avatar_x = excluded.avatar_x,
                 avatar_y = excluded.avatar_y,
                 avatar_z = excluded.avatar_z,
                 avatar_rot_y = excluded.avatar_rot_y,
                 metadata_json = excluded.metadata_json,
                 updated_at = datetime('now')""",
            (
                sid,
                user_id,
                (label or "Default seat")[:120],
                float(x),
                float(y),
                float(z),
                float(rot_y),
                _normalize_room_id(env_id),
                avatar.get("x"),
                avatar.get("y"),
                avatar.get("z"),
                avatar.get("rotY", avatar.get("rot_y")),
                _json_dumps(metadata or {}),
            ),
        )
        await self._conn.commit()
        return await self.get_seat(user_id=user_id, seat_id=sid)

    async def create_or_resume_session(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        surface: str = "voice",
        voice_session_id: str = "",
        room_id: str = DEFAULT_ROOM_ID,
        seat_id: str = DEFAULT_SEAT_ID,
        device_hint: dict[str, Any] | None = None,
        pwa: bool = False,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("create_or_resume_session requires user_id")
        sid = (session_id or "").strip()
        seat_id = _normalize_seat_id(seat_id)
        room_id = _normalize_room_id(room_id)
        device = dict(device_hint or {})
        device["pwa"] = bool(pwa)
        perf = dict(DEFAULT_PERFORMANCE_PROFILE)
        inputs = dict(DEFAULT_INPUT_PREFERENCES)
        room_state = dict(DEFAULT_ROOM_STATE)

        if sid:
            existing = await self.get_session(sid, user_id=user_id)
            if existing is not None:
                await self.patch_session(
                    sid,
                    user_id=user_id,
                    status="preflight",
                    surface=surface,
                    voice_session_id=voice_session_id,
                    room_id=room_id,
                    seat_id=seat_id,
                    device_hint=device,
                )
                refreshed = await self.get_session(sid, user_id=user_id)
                if refreshed is None:
                    raise RuntimeError("XR session disappeared during resume")
                return refreshed

        sid = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO xr_sessions
               (id, user_id, voice_session_id, surface, room_id, seat_id,
                status, device_hint_json, input_preferences_json,
                room_state_json, performance_profile_json)
               VALUES (?, ?, ?, ?, ?, ?, 'preflight', ?, ?, ?, ?)""",
            (
                sid,
                user_id,
                voice_session_id or "",
                (surface or "voice")[:80],
                room_id,
                seat_id,
                _json_dumps(device),
                _json_dumps(inputs),
                _json_dumps(room_state),
                _json_dumps(perf),
            ),
        )
        await self._conn.commit()
        session = await self.get_session(sid, user_id=user_id)
        if session is None:
            raise RuntimeError("XR session insert failed")
        return session

    async def get_session(
        self, session_id: str, *, user_id: str = ""
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM xr_sessions WHERE id = ?"
        params: list[Any] = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cur = await self._conn.execute(query, params)
        row = await cur.fetchone()
        if not row:
            return None
        raw = _row_dict(row)
        seat = await self.get_seat(
            user_id=raw["user_id"],
            seat_id=raw.get("seat_id") or DEFAULT_SEAT_ID,
        )
        return _inflate_session(raw, seat=seat)

    async def patch_session(
        self,
        session_id: str,
        *,
        user_id: str,
        status: str | None = None,
        surface: str | None = None,
        voice_session_id: str | None = None,
        room_id: str | None = None,
        seat_id: str | None = None,
        device_hint: dict[str, Any] | None = None,
        room_state: dict[str, Any] | None = None,
        input_preferences: dict[str, Any] | None = None,
        performance_profile: dict[str, Any] | None = None,
        last_snapshot: dict[str, Any] | None = None,
    ) -> bool:
        if not user_id:
            raise ValueError("patch_session requires user_id")
        sets = ["updated_at = datetime('now')"]
        params: list[Any] = []
        fields = {
            "status": status,
            "surface": surface,
            "voice_session_id": voice_session_id,
        }
        for column, value in fields.items():
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(str(value)[:120])
        if room_id is not None:
            sets.append("room_id = ?")
            params.append(_normalize_room_id(room_id))
        if seat_id is not None:
            sets.append("seat_id = ?")
            params.append(_normalize_seat_id(seat_id))
        json_fields = {
            "device_hint_json": device_hint,
            "room_state_json": room_state,
            "input_preferences_json": input_preferences,
            "performance_profile_json": performance_profile,
            "last_snapshot_json": last_snapshot,
        }
        for column, value in json_fields.items():
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(_json_dumps(value))
        params.extend([session_id, user_id])
        cur = await self._conn.execute(
            f"UPDATE xr_sessions SET {', '.join(sets)} WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def resume_snapshot(
        self, session_id: str, *, user_id: str
    ) -> dict[str, Any] | None:
        session = await self.get_session(session_id, user_id=user_id)
        if session is None:
            return None
        return {
            "session": session,
            "snapshot": session.get("last_snapshot") or {},
            "room": session.get("room_manifest"),
            "seat": session.get("seat_layout"),
        }

    async def record_event(
        self,
        session_id: str,
        *,
        user_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int | None:
        if not user_id:
            raise ValueError("record_event requires user_id")
        exists = await self.get_session(session_id, user_id=user_id)
        if exists is None:
            return None
        cur = await self._conn.execute(
            """INSERT INTO xr_session_events
               (session_id, user_id, event_type, payload_json)
               VALUES (?, ?, ?, ?)""",
            (session_id, user_id, (event_type or "event")[:120], _json_dumps(payload or {})),
        )
        await self._conn.commit()
        return int(cur.lastrowid or 0)

    async def list_events(
        self, session_id: str, *, user_id: str, limit: int = 200
    ) -> list[dict[str, Any]] | None:
        exists = await self.get_session(session_id, user_id=user_id)
        if exists is None:
            return None
        cur = await self._conn.execute(
            """SELECT id, session_id, event_type, payload_json, created_at
               FROM xr_session_events
               WHERE session_id = ? AND user_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_id, user_id, max(1, min(int(limit), 500))),
        )
        rows = await cur.fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            d = _row_dict(row)
            events.append({
                "id": d["id"],
                "session_id": d["session_id"],
                "type": d["event_type"],
                "payload": _json_loads(d["payload_json"], {}),
                "created_at": d["created_at"],
            })
        events.reverse()
        return events
