"""SQLite store for durable surface sessions."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class SurfaceConflictError(RuntimeError):
    """Raised when a caller patches against a stale revision."""

    def __init__(self, *, expected: int, actual: int) -> None:
        super().__init__(f"surface revision conflict: expected {expected}, actual {actual}")
        self.expected = expected
        self.actual = actual


def _dump(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        return json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return default


def _load(text: str, default: Any) -> Any:
    if not text:
        return default
    try:
        loaded = json.loads(text)
        return default if loaded is None else loaded
    except (json.JSONDecodeError, TypeError):
        return default


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_participant(raw: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    now = _utc_now()
    participant_id = (
        str(raw.get("participant_id") or raw.get("id") or "").strip()
        or str(raw.get("device_id") or "").strip()
        or f"part_{secrets.token_hex(6)}"
    )
    role = str(raw.get("role") or (existing or {}).get("role") or "observer").strip().lower()
    participant = {
        "id": participant_id,
        "role": role,
        "device_id": str(raw.get("device_id") or (existing or {}).get("device_id") or "").strip(),
        "label": str(raw.get("label") or (existing or {}).get("label") or "").strip(),
        "capabilities": list(raw.get("capabilities") or (existing or {}).get("capabilities") or []),
        "transport": str(raw.get("transport") or (existing or {}).get("transport") or "browser").strip(),
        "metadata": dict(raw.get("metadata") or (existing or {}).get("metadata") or {}),
        "joined_at": str((existing or {}).get("joined_at") or now),
        "last_seen_at": now,
    }
    return participant


class SurfaceStore:
    """User-scoped persistence for surface sessions and events."""

    _SESSION_COLS = (
        "id, user_id, kind, title, content_ref_json, state_json, "
        "participants_json, status, revision, pairing_code, created_at, "
        "updated_at, ended_at"
    )
    _EVENT_COLS = (
        "id, session_id, user_id, revision, event_type, "
        "source_participant_id, payload_json, created_at"
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def _row_to_session(self, row) -> dict[str, Any]:
        (
            id_, user_id, kind, title, content_ref_json, state_json,
            participants_json, status, revision, pairing_code,
            created_at, updated_at, ended_at,
        ) = row
        return {
            "id": id_,
            "user_id": user_id,
            "kind": kind,
            "title": title or "",
            "content_ref": _load(content_ref_json, {}),
            "state": _load(state_json, {}),
            "participants": _load(participants_json, []),
            "status": status,
            "revision": int(revision or 0),
            "pairing_code": pairing_code or "",
            "created_at": created_at or "",
            "updated_at": updated_at or "",
            "ended_at": ended_at or "",
        }

    def _row_to_event(self, row) -> dict[str, Any]:
        (
            id_, session_id, user_id, revision, event_type,
            source_participant_id, payload_json, created_at,
        ) = row
        return {
            "id": int(id_),
            "session_id": session_id,
            "user_id": user_id,
            "revision": int(revision or 0),
            "type": event_type,
            "source_participant_id": source_participant_id or "",
            "payload": _load(payload_json, {}),
            "created_at": created_at or "",
        }

    async def create(
        self,
        *,
        user_id: str,
        kind: str,
        title: str = "",
        content_ref: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        participants: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("surface session requires user_id")
        session_id = f"surf_{secrets.token_hex(8)}"
        pairing_code = f"{secrets.randbelow(1_000_000):06d}"
        normalized = [
            _normalize_participant(p)
            for p in (participants or [])
            if isinstance(p, dict)
        ]
        await self._conn.execute(
            "INSERT INTO surface_sessions "
            "(id, user_id, kind, title, content_ref_json, state_json, "
            " participants_json, status, revision, pairing_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 0, ?)",
            (
                session_id,
                user_id,
                str(kind or "surface.generic").strip() or "surface.generic",
                str(title or "").strip(),
                _dump(content_ref, "{}"),
                _dump(state, "{}"),
                _dump(normalized, "[]"),
                pairing_code,
            ),
        )
        await self._conn.commit()
        session = await self.get(session_id, user_id=user_id)
        assert session is not None
        await self.record_event(
            session_id=session_id,
            user_id=user_id,
            revision=0,
            event_type="surface.session.created",
            payload={
                "kind": session["kind"],
                "title": session["title"],
                "content_ref": session["content_ref"],
            },
        )
        log.info("surface_session_created", user_id=user_id, session_id=session_id, kind=session["kind"])
        return session

    async def get(self, session_id: str, *, user_id: str) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SESSION_COLS} FROM surface_sessions "
            "WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_session(row) if row else None

    async def list_for_user(
        self,
        *,
        user_id: str,
        include_ended: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        limit = min(200, max(1, int(limit or 50)))
        if include_ended:
            cursor = await self._conn.execute(
                f"SELECT {self._SESSION_COLS} FROM surface_sessions "
                "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cursor = await self._conn.execute(
                f"SELECT {self._SESSION_COLS} FROM surface_sessions "
                "WHERE user_id = ? AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
        return [self._row_to_session(r) for r in await cursor.fetchall()]

    async def join(
        self,
        session_id: str,
        *,
        user_id: str,
        participant: dict[str, Any],
        event_type: str = "surface.participant.joined",
    ) -> dict[str, Any] | None:
        session = await self.get(session_id, user_id=user_id)
        if session is None or session["status"] != "active":
            return None
        participants = list(session.get("participants") or [])
        participant_id = (
            str(participant.get("participant_id") or participant.get("id") or "").strip()
            or str(participant.get("device_id") or "").strip()
        )
        existing_idx = -1
        existing = None
        if participant_id:
            for idx, item in enumerate(participants):
                if str(item.get("id") or "") == participant_id:
                    existing_idx = idx
                    existing = item
                    break
        normalized = _normalize_participant(participant, existing=existing)
        if existing_idx >= 0:
            participants[existing_idx] = normalized
        else:
            participants.append(normalized)

        revision = int(session["revision"]) + 1
        await self._conn.execute(
            "UPDATE surface_sessions SET participants_json = ?, revision = ?, "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (_dump(participants, "[]"), revision, session_id, user_id),
        )
        await self._conn.commit()
        await self.record_event(
            session_id=session_id,
            user_id=user_id,
            revision=revision,
            event_type=event_type,
            source_participant_id=normalized["id"],
            payload={"participant": normalized},
        )
        return await self.get(session_id, user_id=user_id)

    async def heartbeat(
        self,
        session_id: str,
        *,
        user_id: str,
        participant_id: str,
    ) -> dict[str, Any] | None:
        session = await self.get(session_id, user_id=user_id)
        if session is None or session["status"] != "active":
            return None
        participants = list(session.get("participants") or [])
        changed = False
        now = _utc_now()
        for item in participants:
            if str(item.get("id") or "") == str(participant_id or "").strip():
                item["last_seen_at"] = now
                changed = True
                break
        if not changed:
            return session
        await self._conn.execute(
            "UPDATE surface_sessions SET participants_json = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (_dump(participants, "[]"), session_id, user_id),
        )
        await self._conn.commit()
        return await self.get(session_id, user_id=user_id)

    async def patch_state(
        self,
        session_id: str,
        *,
        user_id: str,
        patch: dict[str, Any],
        source_participant_id: str = "",
        base_revision: int | None = None,
        replace: bool = False,
    ) -> dict[str, Any] | None:
        session = await self.get(session_id, user_id=user_id)
        if session is None or session["status"] != "active":
            return None
        current_revision = int(session["revision"] or 0)
        if base_revision is not None and int(base_revision) != current_revision:
            raise SurfaceConflictError(expected=int(base_revision), actual=current_revision)

        next_state = dict(patch or {}) if replace else _deep_merge(dict(session.get("state") or {}), patch or {})
        revision = current_revision + 1
        await self._conn.execute(
            "UPDATE surface_sessions SET state_json = ?, revision = ?, "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (_dump(next_state, "{}"), revision, session_id, user_id),
        )
        await self._conn.commit()
        await self.record_event(
            session_id=session_id,
            user_id=user_id,
            revision=revision,
            event_type="surface.state.patched",
            source_participant_id=source_participant_id,
            payload={
                "patch": patch or {},
                "replace": bool(replace),
                "state": next_state,
            },
        )
        return await self.get(session_id, user_id=user_id)

    async def end(
        self,
        session_id: str,
        *,
        user_id: str,
        source_participant_id: str = "",
    ) -> dict[str, Any] | None:
        session = await self.get(session_id, user_id=user_id)
        if session is None:
            return None
        if session["status"] == "ended":
            return session
        revision = int(session["revision"] or 0) + 1
        await self._conn.execute(
            "UPDATE surface_sessions SET status = 'ended', revision = ?, "
            "ended_at = datetime('now'), updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (revision, session_id, user_id),
        )
        await self._conn.commit()
        await self.record_event(
            session_id=session_id,
            user_id=user_id,
            revision=revision,
            event_type="surface.session.ended",
            source_participant_id=source_participant_id,
            payload={},
        )
        return await self.get(session_id, user_id=user_id)

    async def record_event(
        self,
        *,
        session_id: str,
        user_id: str,
        revision: int,
        event_type: str,
        source_participant_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cursor = await self._conn.execute(
            "INSERT INTO surface_events "
            "(session_id, user_id, revision, event_type, source_participant_id, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                user_id,
                int(revision or 0),
                str(event_type or "surface.event"),
                str(source_participant_id or ""),
                _dump(payload, "{}"),
            ),
        )
        await self._conn.commit()
        event_id = int(cursor.lastrowid or 0)
        event = await self.get_event(event_id, user_id=user_id)
        assert event is not None
        return event

    async def get_event(self, event_id: int, *, user_id: str) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            f"SELECT {self._EVENT_COLS} FROM surface_events "
            "WHERE id = ? AND user_id = ?",
            (int(event_id), user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_event(row) if row else None

    async def events_after(
        self,
        session_id: str,
        *,
        user_id: str,
        after_revision: int = -1,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(500, max(1, int(limit or 100)))
        cursor = await self._conn.execute(
            f"SELECT {self._EVENT_COLS} FROM surface_events "
            "WHERE user_id = ? AND session_id = ? AND revision > ? "
            "ORDER BY id ASC LIMIT ?",
            (user_id, session_id, int(after_revision), limit),
        )
        return [self._row_to_event(r) for r in await cursor.fetchall()]
