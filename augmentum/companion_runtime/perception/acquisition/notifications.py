"""Notification acquisition store — L0 of the perception pipeline.

The on-device ``NotificationListenerService`` reads posted notifications across
all apps, normalizes each to the small typed shape below, and uploads a batch to
the user's own server (``POST /api/perception/notifications``). This module is
the persistence layer that batch lands in: dedup-on-insert, recent-read for the
fuser, and a retention prune.

Discipline:
  * **User-scoped** — every fn takes ``user_id`` and writes/reads only that
    user's rows (multi-tenant rule); never the anon sentinel.
  * **Dedup** — Android re-posts a notification on every update (typing dots,
    delivery ticks). We collapse on ``(user_id, dedup_key)`` so an updated
    notification doesn't read as a fresh signal and inflate "pressure".
  * **Retention** — fusion only needs the recent past; ``prune_notifications``
    drops rows older than the window so the store doesn't accumulate a forever
    log of everything that ever buzzed the phone.

The rows are raw material, never surfaced as-is — the fuser
(``perception/fusers/notifications.py``) correlates them into insights.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Retention: fusion looks back hours, not days. A week of headroom covers
# "last contact"-style lookbacks without keeping a permanent notification log.
DEFAULT_RETENTION_DAYS = 7
# Defensive cap on a single upload batch so a misbehaving client can't wedge
# the write path; the listener batches modestly anyway.
MAX_BATCH = 200
# Bodies can be long (a pasted paragraph); store a useful prefix, not an essay.
_MAX_BODY = 2000
_MAX_FIELD = 400


@dataclass(frozen=True, slots=True)
class NotificationObservation:
    """One normalized notification entity (the typed L0 record)."""

    source_pkg: str = ""
    source_app: str = ""
    category: str = ""
    title: str = ""
    body: str = ""
    person: str = ""
    is_message: bool = False
    posted_at: float = 0.0
    dedup_key: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> NotificationObservation | None:
        """Build from one client-uploaded JSON object. Returns None if it's too
        empty to be worth storing (no app + no text). Best-effort, never raises —
        a bad item in a batch is skipped, not fatal."""
        if not isinstance(raw, dict):
            return None
        pkg = _clip(raw.get("source_pkg") or raw.get("package") or "", _MAX_FIELD)
        title = _clip(raw.get("title") or "", _MAX_FIELD)
        body = _clip(raw.get("body") or raw.get("text") or "", _MAX_BODY)
        if not pkg and not title and not body:
            return None
        app = _clip(raw.get("source_app") or raw.get("app") or "", _MAX_FIELD)
        category = _clip(raw.get("category") or "", 64)
        person = _clip(raw.get("person") or "", _MAX_FIELD)
        is_message = bool(raw.get("is_message")) or category == "msg"
        try:
            posted_at = float(raw.get("posted_at") or 0.0)
        except (TypeError, ValueError):
            posted_at = 0.0
        # A stable dedup key so re-posts of the same notification collapse. The
        # client should send notif_key (Android's StatusBarNotification.key);
        # fall back to a content fingerprint so we still dedup if it doesn't.
        notif_key = _clip(raw.get("notif_key") or raw.get("key") or "", _MAX_FIELD)
        dedup = notif_key or f"{pkg}|{title}|{body[:80]}"
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            source_pkg=pkg, source_app=app, category=category, title=title,
            body=body, person=person or (title if is_message else ""),
            is_message=is_message, posted_at=posted_at, dedup_key=dedup,
            payload=payload,
        )


def _clip(value: Any, n: int) -> str:
    s = str(value or "").strip()
    return s[:n]


async def record_notifications(
    backend: Any,
    *,
    user_id: str,
    observations: list[NotificationObservation],
    now: float | None = None,
) -> int:
    """Persist a batch of normalized notifications for ``user_id``. Returns the
    number of NEW rows stored (dedup-skipped re-posts don't count). Best-effort:
    a single bad row is logged and skipped, never fails the batch."""
    if not user_id or not observations:
        return 0
    now = now if now is not None else time.time()
    stored = 0
    for obs in observations[:MAX_BATCH]:
        if not isinstance(obs, NotificationObservation):
            continue
        try:
            cur = await backend.conn.execute(
                "INSERT OR IGNORE INTO notification_observations "
                "(user_id, source_pkg, source_app, category, title, body, person, "
                " is_message, posted_at, ingested_at, dedup_key, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, obs.source_pkg, obs.source_app, obs.category,
                    obs.title, obs.body, obs.person, 1 if obs.is_message else 0,
                    obs.posted_at or now, now, obs.dedup_key,
                    json.dumps(obs.payload, ensure_ascii=False),
                ),
            )
            await cur.close()
            stored += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        except Exception:  # noqa: BLE001 — one bad row can't fail the batch
            log.warning("notification_record_failed", user_id=user_id, exc_info=True)
            continue
    try:
        await backend.conn.commit()
    except Exception:  # noqa: BLE001
        log.warning("notification_record_commit_failed", exc_info=True)
    return stored


async def recent_notifications(
    backend: Any,
    *,
    user_id: str,
    since_s: float = 21600.0,
    limit: int = 100,
    now: float | None = None,
) -> list[NotificationObservation]:
    """Most-recent notifications for ``user_id`` within ``since_s`` seconds
    (default 6h — the fusion lookback). Newest first. The fuser reads this list
    out of the signal bag; it does NOT touch the DB itself (stays pure)."""
    if not user_id:
        return []
    now = now if now is not None else time.time()
    cutoff = now - max(0.0, since_s)
    try:
        cur = await backend.conn.execute(
            "SELECT source_pkg, source_app, category, title, body, person, "
            "       is_message, posted_at, dedup_key, payload "
            "FROM notification_observations "
            "WHERE user_id = ? AND posted_at >= ? "
            "ORDER BY posted_at DESC LIMIT ?",
            (user_id, cutoff, max(1, min(limit, MAX_BATCH))),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:  # noqa: BLE001 — perception degrades to no signal
        log.warning("notification_recent_failed", user_id=user_id, exc_info=True)
        return []
    out: list[NotificationObservation] = []
    for r in rows:
        try:
            payload = json.loads(r[9]) if r[9] else {}
        except (ValueError, TypeError):
            payload = {}
        out.append(NotificationObservation(
            source_pkg=r[0], source_app=r[1], category=r[2], title=r[3],
            body=r[4], person=r[5], is_message=bool(r[6]),
            posted_at=float(r[7] or 0.0), dedup_key=r[8],
            payload=payload if isinstance(payload, dict) else {},
        ))
    return out


async def prune_notifications(
    backend: Any,
    *,
    user_id: str = "",
    retention_days: int = DEFAULT_RETENTION_DAYS,
    now: float | None = None,
) -> int:
    """Drop rows older than the retention window. Scoped to ``user_id`` when
    given, else prunes globally (a maintenance sweep). Returns rows deleted."""
    now = now if now is not None else time.time()
    cutoff = now - max(1, retention_days) * 86400.0
    try:
        if user_id:
            cur = await backend.conn.execute(
                "DELETE FROM notification_observations "
                "WHERE user_id = ? AND posted_at < ?",
                (user_id, cutoff),
            )
        else:
            cur = await backend.conn.execute(
                "DELETE FROM notification_observations WHERE posted_at < ?",
                (cutoff,),
            )
        deleted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        await cur.close()
        await backend.conn.commit()
        return deleted
    except Exception:  # noqa: BLE001
        log.warning("notification_prune_failed", exc_info=True)
        return 0
