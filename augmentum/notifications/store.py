"""Notification store — aiosqlite-backed publish/read primitives.

The wire-shape contract:

* ``publish()`` is idempotent on ``(user_id, source, dedupe_key)``
  when ``dedupe_key`` is non-empty. Reposting with the same key
  updates body / actions / payload / importance / channel and
  refreshes ``updated_at``, while preserving ``created_at`` and
  resetting ``read_at`` + ``dismissed_at`` so the event re-surfaces.

* ``list_for_user()`` returns rows ordered by ``created_at DESC``,
  with optional filters for unread-only, undismissed-only, and
  thread-scoped views.

* ``mark_read`` / ``dismiss`` set lifecycle timestamps idempotently
  — calling twice is a no-op, never an error.

* ``mute_channel`` lazily materializes the row in
  ``notification_channels`` so a user can mute a default-catalog
  channel without the row pre-existing.

Design references (see also the design doc):

* Dedupe-by-key with in-place update mirrors freedesktop
  ``replaces_id`` semantics and Android's ``(package, tag, id)``
  identity. The partial unique index in migration 221 enforces
  uniqueness only for non-empty keys, so "no dedup, every publish
  distinct" works for one-shot toasts.

* Per-user isolation: every method accepts ``user_id`` keyword and
  filters by it. There's no cross-user read path.

* All times are stored as ISO-8601 UTC strings (matches existing
  Augmentum stores like ``memory/notifications.py``). The integer
  ``expires_at`` parameter on ``publish`` is rendered to ISO for
  storage so it round-trips cleanly.

Phase 1 scope: synchronous store primitives. WS fan-out at publish
time is the next task — when wired, it dispatches to subscribers
without changing this module's surface.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.notifications.catalog import (
    ChannelTemplate,
    DEFAULT_CHANNELS,
    catalog_channel,
    normalise_importance,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite


log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    """16-char hex slug. Matches the coder turn-archive convention."""

    return uuid.uuid4().hex[:16]


# ── Typed return shapes ──────────────────────────────────────────


@dataclass(frozen=True)
class NotificationAction:
    """One actionable button on a notification.

    The UI renders these as buttons; clicks POST to the action
    callback (deferred — not in this turn). Use ``style`` to hint
    visual treatment (``"primary"`` is the default-emphasized one,
    ``"danger"`` for destructive actions, ``"default"`` otherwise).
    """

    id: str
    label: str
    style: str = "default"
    href: str = ""

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"id": self.id, "label": self.label}
        if self.style and self.style != "default":
            out["style"] = self.style
        if self.href:
            out["href"] = self.href
        return out


@dataclass(frozen=True)
class NotificationChannel:
    """Resolved channel — template + any per-user override merged."""

    channel_id: str
    name: str
    description: str
    importance: int
    default_sound: str = ""
    muted_until: str = ""  # ISO timestamp; '' = not muted
    user_customized: bool = False


@dataclass(frozen=True)
class Notification:
    """One persisted notification row."""

    notification_id: str
    user_id: str
    channel_id: str
    source: str
    title: str
    body: str = ""
    icon: str = ""
    importance: int = 2
    dedupe_key: str = ""
    thread_id: str = ""
    actions: tuple[NotificationAction, ...] = field(default_factory=tuple)
    payload: dict[str, Any] = field(default_factory=dict)
    transient: bool = False
    expires_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    delivered_at: str = ""
    read_at: str = ""
    dismissed_at: str = ""


# ── Channel resolution ───────────────────────────────────────────


def _resolved_from_template(template: ChannelTemplate) -> NotificationChannel:
    return NotificationChannel(
        channel_id=template.channel_id,
        name=template.name,
        description=template.description,
        importance=template.importance,
        default_sound=template.default_sound,
        muted_until="",
        user_customized=False,
    )


def _resolved_from_row(
    template: ChannelTemplate | None, row: tuple,
) -> NotificationChannel:
    """Merge a DB row (the user's override) with the template defaults.

    The DB row is authoritative for ``importance`` + ``muted_until``;
    name / description / default_sound fall back to the template
    when present, otherwise to whatever the row carries.
    """

    (channel_id, _user_id, name, description, importance,
     default_sound, muted_until, _created_at) = row
    return NotificationChannel(
        channel_id=channel_id,
        name=name or (template.name if template else channel_id),
        description=description or (template.description if template else ""),
        importance=normalise_importance(int(importance)),
        default_sound=default_sound or (template.default_sound if template else ""),
        muted_until=muted_until or "",
        user_customized=True,
    )


async def resolved_channels(
    conn: "aiosqlite.Connection", *, user_id: str,
) -> list[NotificationChannel]:
    """Catalog defaults merged with this user's customizations.

    Order matches the catalog (templates first, in canonical order).
    User-customized channels NOT in the catalog (legacy templates,
    user-defined ad-hoc channels) trail the list in
    ``channel_id`` order.
    """

    cur = await conn.execute(
        """SELECT channel_id, user_id, name, description, importance,
                  default_sound, muted_until, created_at
             FROM notification_channels
            WHERE user_id = ?""",
        (user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    by_id: dict[str, tuple] = {r[0]: r for r in rows}

    out: list[NotificationChannel] = []
    catalog_ids: set[str] = set()
    for tmpl in DEFAULT_CHANNELS:
        catalog_ids.add(tmpl.channel_id)
        row = by_id.get(tmpl.channel_id)
        if row is None:
            out.append(_resolved_from_template(tmpl))
        else:
            out.append(_resolved_from_row(tmpl, row))
    # Trailing: rows that don't correspond to any catalog template.
    for cid, row in sorted(by_id.items()):
        if cid in catalog_ids:
            continue
        out.append(_resolved_from_row(None, row))
    return out


async def mute_channel(
    conn: "aiosqlite.Connection", *,
    user_id: str, channel_id: str, until_iso: str | None,
) -> None:
    """Set or clear a per-user mute on a channel.

    Lazy-materializes the row from the catalog template if no
    override exists yet. ``until_iso=None`` clears the mute.
    """

    template = catalog_channel(channel_id)
    # Existing row?
    cur = await conn.execute(
        """SELECT 1 FROM notification_channels
            WHERE user_id = ? AND channel_id = ?""",
        (user_id, channel_id),
    )
    row = await cur.fetchone()
    await cur.close()

    if row is None:
        # Materialize from the template (or with safe defaults if the
        # channel_id is unknown — we don't want to refuse a mute on a
        # forward-compat channel id that doesn't exist in our catalog).
        name = template.name if template else channel_id
        description = template.description if template else ""
        importance = template.importance if template else 2
        default_sound = template.default_sound if template else ""
        await conn.execute(
            """INSERT INTO notification_channels
                   (channel_id, user_id, name, description, importance,
                    default_sound, muted_until, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (channel_id, user_id, name, description, importance,
             default_sound, until_iso or None, _now_iso()),
        )
    else:
        await conn.execute(
            """UPDATE notification_channels
                  SET muted_until = ?
                WHERE user_id = ? AND channel_id = ?""",
            (until_iso or None, user_id, channel_id),
        )
    await conn.commit()


# ── Publish + read ───────────────────────────────────────────────


def _serialise_actions(actions: list[NotificationAction] | None) -> str:
    if not actions:
        return "[]"
    return json.dumps([a.to_dict() for a in actions], separators=(",", ":"))


def _serialise_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "{}"
    return json.dumps(payload, separators=(",", ":"))


def _expires_to_iso(expires_at: int | str | None) -> str | None:
    if expires_at is None:
        return None
    if isinstance(expires_at, str):
        return expires_at
    # Treat int/float as a unix timestamp.
    return datetime.fromtimestamp(int(expires_at), UTC).isoformat()


async def publish(
    conn: "aiosqlite.Connection", *,
    user_id: str,
    channel_id: str,
    source: str,
    title: str,
    body: str = "",
    importance: int | None = None,
    dedupe_key: str = "",
    thread_id: str = "",
    actions: list[NotificationAction] | None = None,
    payload: dict[str, Any] | None = None,
    transient: bool = False,
    expires_at: int | str | None = None,
    icon: str = "",
) -> str:
    """Insert (or update-in-place by dedupe_key) one notification.

    Returns the row's notification_id. With a non-empty
    ``dedupe_key``, an existing row's body / actions / payload /
    importance / channel / title / icon refresh and ``read_at`` +
    ``dismissed_at`` clear (so the event re-surfaces); ``created_at``
    is preserved so feed ordering doesn't churn.

    Importance resolution: if ``importance`` is None, the catalog
    template's importance is used (DEFAULT for unknown channels).
    Pass an int to override on a per-event basis (e.g. one critical
    invite on a normally-DEFAULT channel).
    """

    if not user_id:
        raise ValueError("user_id is required (per-user isolation)")
    if not channel_id:
        raise ValueError("channel_id is required")
    if not source:
        raise ValueError("source is required")
    if not title:
        raise ValueError("title is required")

    if importance is None:
        template = catalog_channel(channel_id)
        importance = template.importance if template else 2
    importance = normalise_importance(importance)

    now = _now_iso()
    actions_json = _serialise_actions(actions)
    payload_json = _serialise_payload(payload)
    expires_iso = _expires_to_iso(expires_at)
    transient_i = 1 if transient else 0

    existing_id: str | None = None
    existing_created_at: str | None = None
    if dedupe_key:
        cur = await conn.execute(
            """SELECT notification_id, created_at
                 FROM notifications
                WHERE user_id = ? AND source = ? AND dedupe_key = ?""",
            (user_id, source, dedupe_key),
        )
        row = await cur.fetchone()
        await cur.close()
        if row is not None:
            existing_id, existing_created_at = row[0], row[1]

    if existing_id is not None:
        # In-place update. Refresh content; preserve created_at;
        # clear read/dismissed so the event re-surfaces.
        await conn.execute(
            """UPDATE notifications
                  SET channel_id    = ?,
                      thread_id     = ?,
                      importance    = ?,
                      title         = ?,
                      body          = ?,
                      icon          = ?,
                      actions_json  = ?,
                      payload_json  = ?,
                      transient     = ?,
                      expires_at    = ?,
                      updated_at    = ?,
                      delivered_at  = NULL,
                      read_at       = NULL,
                      dismissed_at  = NULL
                WHERE notification_id = ?""",
            (channel_id, thread_id, importance, title, body, icon,
             actions_json, payload_json, transient_i, expires_iso, now,
             existing_id),
        )
        await conn.commit()
        return existing_id

    notification_id = _new_id()
    await conn.execute(
        """INSERT INTO notifications
               (notification_id, user_id, channel_id, source,
                dedupe_key, thread_id, importance, title, body, icon,
                actions_json, payload_json, transient, expires_at,
                created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (notification_id, user_id, channel_id, source,
         dedupe_key, thread_id, importance, title, body, icon,
         actions_json, payload_json, transient_i, expires_iso,
         now, now),
    )
    await conn.commit()
    return notification_id


def _row_to_notification(row: tuple) -> Notification:
    (notification_id, user_id, channel_id, source, dedupe_key,
     thread_id, importance, title, body, icon, actions_json,
     payload_json, transient_i, expires_at, created_at, updated_at,
     delivered_at, read_at, dismissed_at) = row

    actions: tuple[NotificationAction, ...] = ()
    try:
        decoded = json.loads(actions_json or "[]")
        if isinstance(decoded, list):
            actions = tuple(
                NotificationAction(
                    id=str(a.get("id", "")),
                    label=str(a.get("label", "")),
                    style=str(a.get("style", "default")),
                    href=str(a.get("href", "")),
                )
                for a in decoded
                if isinstance(a, dict) and a.get("id") and a.get("label")
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        log.warning(
            "notification_actions_json_invalid",
            notification_id=notification_id, raw=actions_json,
        )

    payload: dict[str, Any] = {}
    try:
        decoded_p = json.loads(payload_json or "{}")
        if isinstance(decoded_p, dict):
            payload = decoded_p
    except (json.JSONDecodeError, ValueError, TypeError):
        log.warning(
            "notification_payload_json_invalid",
            notification_id=notification_id, raw=payload_json,
        )

    return Notification(
        notification_id=notification_id,
        user_id=user_id,
        channel_id=channel_id,
        source=source,
        dedupe_key=dedupe_key or "",
        thread_id=thread_id or "",
        importance=normalise_importance(int(importance)),
        title=title,
        body=body or "",
        icon=icon or "",
        actions=actions,
        payload=payload,
        transient=bool(transient_i),
        expires_at=expires_at or "",
        created_at=created_at or "",
        updated_at=updated_at or "",
        delivered_at=delivered_at or "",
        read_at=read_at or "",
        dismissed_at=dismissed_at or "",
    )


_SELECT_COLS = (
    "notification_id, user_id, channel_id, source, dedupe_key, "
    "thread_id, importance, title, body, icon, actions_json, "
    "payload_json, transient, expires_at, created_at, updated_at, "
    "delivered_at, read_at, dismissed_at"
)


async def list_for_user(
    conn: "aiosqlite.Connection", *,
    user_id: str,
    include_read: bool = True,
    include_dismissed: bool = False,
    thread_id: str | None = None,
    limit: int = 100,
) -> list[Notification]:
    """Read this user's feed, newest-first.

    Defaults to "everything not dismissed, including already-read."
    Set ``include_read=False`` for an unread-only inbox; set
    ``include_dismissed=True`` to fetch the historical archive.
    """

    if not user_id:
        return []

    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]
    if not include_dismissed:
        clauses.append("dismissed_at IS NULL")
    if not include_read:
        clauses.append("read_at IS NULL")
    if thread_id is not None:
        clauses.append("thread_id = ?")
        params.append(thread_id)

    # Tiebreaker on notification_id: two events stamped in the
    # same low-resolution clock tick (Windows datetime.now() is
    # 15ms-resolution) would otherwise have unstable feed ordering.
    sql = (
        f"SELECT {_SELECT_COLS} FROM notifications "
        f"WHERE {' AND '.join(clauses)} "
        f"ORDER BY created_at DESC, notification_id DESC LIMIT ?"
    )
    params.append(int(limit))
    cur = await conn.execute(sql, params)
    rows = await cur.fetchall()
    await cur.close()
    return [_row_to_notification(r) for r in rows]


async def get_notification(
    conn: "aiosqlite.Connection", *, user_id: str, notification_id: str,
) -> Notification | None:
    """Fetch one notification. Returns None if missing or wrong user."""

    cur = await conn.execute(
        f"SELECT {_SELECT_COLS} FROM notifications "
        "WHERE notification_id = ? AND user_id = ?",
        (notification_id, user_id),
    )
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        return None
    return _row_to_notification(row)


async def mark_delivered(
    conn: "aiosqlite.Connection", *, user_id: str, notification_id: str,
) -> bool:
    """Stamp ``delivered_at``. Idempotent. Returns whether it changed."""

    cur = await conn.execute(
        """UPDATE notifications
              SET delivered_at = ?
            WHERE notification_id = ? AND user_id = ? AND delivered_at IS NULL""",
        (_now_iso(), notification_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def mark_read(
    conn: "aiosqlite.Connection", *, user_id: str, notification_id: str,
) -> bool:
    """Stamp ``read_at``. Idempotent. Returns whether it changed."""

    cur = await conn.execute(
        """UPDATE notifications
              SET read_at = ?
            WHERE notification_id = ? AND user_id = ? AND read_at IS NULL""",
        (_now_iso(), notification_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def dismiss(
    conn: "aiosqlite.Connection", *, user_id: str, notification_id: str,
) -> bool:
    """Stamp ``dismissed_at``. Idempotent. Returns whether it changed."""

    cur = await conn.execute(
        """UPDATE notifications
              SET dismissed_at = ?
            WHERE notification_id = ? AND user_id = ? AND dismissed_at IS NULL""",
        (_now_iso(), notification_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def dismiss_by_dedupe_key(
    conn: "aiosqlite.Connection", *,
    user_id: str, source: str, dedupe_key: str,
) -> int:
    """Stamp ``dismissed_at`` on every notification matching the dedupe
    triplet that's still un-dismissed.

    Used by missed-call detection (and other "the row is no longer
    actionable" cases) to clear the ringing banner from the feed when
    the timer fires. Idempotent — already-dismissed rows are left alone.

    Returns the number of rows newly stamped.
    """

    if not dedupe_key:
        return 0
    cur = await conn.execute(
        """UPDATE notifications
              SET dismissed_at = ?
            WHERE user_id = ? AND source = ? AND dedupe_key = ?
              AND dismissed_at IS NULL""",
        (_now_iso(), user_id, source, dedupe_key),
    )
    await conn.commit()
    return cur.rowcount


async def expire_transient(
    conn: "aiosqlite.Connection", *, now_iso: str | None = None,
) -> int:
    """Sweep transient notifications past expires_at.

    Returns the number of rows deleted. The "transient" flag means
    "auto-dismiss after display" — once expired, the row is removed
    entirely rather than left in the feed with ``dismissed_at`` set.
    Persistent (non-transient) expired notifications stay in the
    feed but UIs can filter them out by ``expires_at``.
    """

    effective_now = now_iso or _now_iso()
    cur = await conn.execute(
        """DELETE FROM notifications
            WHERE transient = 1
              AND expires_at IS NOT NULL
              AND expires_at <= ?""",
        (effective_now,),
    )
    await conn.commit()
    return cur.rowcount or 0


# ── Class wrapper (matches the design doc surface) ───────────────


class NotificationStore:
    """Thin wrapper around the module functions for callers that want
    a single injectable object (e.g. the future route layer).

    All methods forward to the free functions above. The class is
    pure sugar; tests can keep using free functions directly.
    """

    def __init__(self, conn: "aiosqlite.Connection") -> None:
        self._conn = conn

    async def publish(self, **kwargs: Any) -> str:
        return await publish(self._conn, **kwargs)

    async def list_for_user(self, **kwargs: Any) -> list[Notification]:
        return await list_for_user(self._conn, **kwargs)

    async def get(self, *, user_id: str, notification_id: str) -> Notification | None:
        return await get_notification(
            self._conn, user_id=user_id, notification_id=notification_id,
        )

    async def mark_delivered(self, *, user_id: str, notification_id: str) -> bool:
        return await mark_delivered(
            self._conn, user_id=user_id, notification_id=notification_id,
        )

    async def mark_read(self, *, user_id: str, notification_id: str) -> bool:
        return await mark_read(
            self._conn, user_id=user_id, notification_id=notification_id,
        )

    async def dismiss(self, *, user_id: str, notification_id: str) -> bool:
        return await dismiss(
            self._conn, user_id=user_id, notification_id=notification_id,
        )

    async def mute_channel(
        self, *, user_id: str, channel_id: str, until_iso: str | None,
    ) -> None:
        await mute_channel(
            self._conn,
            user_id=user_id, channel_id=channel_id, until_iso=until_iso,
        )

    async def resolved_channels(self, *, user_id: str) -> list[NotificationChannel]:
        return await resolved_channels(self._conn, user_id=user_id)

    async def expire_transient(self, *, now_iso: str | None = None) -> int:
        return await expire_transient(self._conn, now_iso=now_iso)
