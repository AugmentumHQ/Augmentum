"""Persistent key-value settings store backed by SQLite.

Two tables live side by side:

* ``app_settings`` — install-wide values (rate limits, auth config, model
  routing roles, server secrets, system model choices). One row per key.
* ``user_settings`` — per-tenant overrides (UI preferences, personal
  prompts, typography, voice prefs, etc.). Composite PK on
  ``(user_id, key)`` so two users can hold different values.

Stage D migration 094 introduced ``user_settings``. The read path for
tenant-facing keys falls back from user → global, so nothing breaks for
pre-existing installs: the first time a tenant saves their own value,
they diverge from the install default.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


T = TypeVar("T")


# Settings writes are interactive (user clicks "Save" in Settings UI)
# and should win short races against background telemetry. SQLite's
# busy_handler covers most contention via PRAGMA busy_timeout, but
# the brief PENDING → EXCLUSIVE transition during another connection's
# COMMIT can return SQLITE_BUSY without invoking the handler — observed
# 2026-06-10 when ResourceLedger's 3.7s WSL2-fsync COMMIT raced with a
# /api/config/tools PUT, surfacing as a 500. Retry briefly then surface
# clearly if the lock truly didn't clear (rare; almost always one retry
# is enough — the lock-transition window is tens of ms).
_LOCK_RETRY_ATTEMPTS = 4
_LOCK_RETRY_INITIAL_DELAY_S = 0.05  # 50ms, doubling: 50 → 100 → 200 → 400 ≈ 750ms total


async def _retry_on_lock(
    op: Callable[[], Awaitable[T]], *, op_name: str,
) -> T:
    """Run *op*, retrying on ``database is locked`` with exponential backoff.

    Only retries on the specific OperationalError("database is locked")
    — any other failure propagates immediately. Total budget is small
    (~750ms across 4 attempts) so a genuinely wedged DB still surfaces
    quickly instead of hanging the UI.
    """
    delay = _LOCK_RETRY_INITIAL_DELAY_S
    last_exc: BaseException | None = None
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        try:
            return await op()
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc):
                raise
            last_exc = exc
            if attempt == _LOCK_RETRY_ATTEMPTS - 1:
                log.warning(
                    "settings_write_lock_retries_exhausted",
                    op=op_name, attempts=_LOCK_RETRY_ATTEMPTS,
                )
                break
            await asyncio.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc


class SettingsStore:
    """Read/write arbitrary key-value settings that survive restarts.

    **In-process write-through cache.** Settings are read on a lot of hot
    paths (every ``/api/config/ui`` poll, per-mode injection toggles, dream
    scheduler thresholds, powers state, …) but written rarely — and only
    ever through this class's ``set``/``set_user`` in this single process.
    So each read is served from a dict after the first miss, and writes
    update the dict in lock-step with the DB. This removes the bulk of the
    ``SELECT value FROM (app|user)_settings WHERE …`` traffic from the
    shared aiosqlite connection (it was a steady stream of ~100-250ms
    round-trips under load on slow Docker storage). Behaviour is
    unchanged: the cache only ever holds what the DB holds.

    A cached entry of ``None`` means "known-absent" (so repeated reads of
    an unset key don't re-hit the DB either). Presence of the dict key is
    the cache-hit signal; the value may legitimately be ``None``.

    Caveat: a *second* ``SettingsStore`` over the same connection (e.g.
    ``coder.git_credentials``) keeps its own cache. They serve disjoint
    key namespaces in practice; if that ever changes, route both through
    one shared instance instead.
    """

    # Sentinel kept private — callers never see it; cache lookups use the
    # ``in`` test, not equality against this.
    _MISS = object()

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        self._global_cache: dict[str, str | None] = {}
        self._user_cache: dict[tuple[str, str], str | None] = {}
        # ``get_all`` was called and the result is authoritative for the
        # whole table — lets a subsequent miss on an unknown key answer
        # "absent" without a DB hit.
        self._global_fully_loaded = False
        self._user_fully_loaded: set[str] = set()

    # ------------------------------------------------------------------
    # Global (app_settings) — install-wide values
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Return the install-wide value for *key*, or ``None`` if unset."""
        if key in self._global_cache:
            return self._global_cache[key]
        if self._global_fully_loaded:
            return None
        cursor = await self._conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,),
        )
        row = await cursor.fetchone()
        val = row[0] if row else None
        self._global_cache[key] = val
        return val

    async def set(self, key: str, value: str | None) -> None:
        """Persist the install-wide *key*/*value*. Pass ``None`` to delete."""
        async def _write():
            if value is None:
                await self._conn.execute(
                    "DELETE FROM app_settings WHERE key = ?", (key,),
                )
            else:
                await self._conn.execute(
                    "INSERT INTO app_settings (key, value, updated_at) "
                    "VALUES (?, ?, datetime('now')) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (key, value),
                )
            await self._conn.commit()
        await _retry_on_lock(_write, op_name="settings.set")
        self._global_cache[key] = value

    async def get_all(self) -> dict[str, str]:
        """Return all install-wide settings as a dict."""
        cursor = await self._conn.execute("SELECT key, value FROM app_settings")
        rows = await cursor.fetchall()
        result = {row[0]: row[1] for row in rows}
        # Refresh the cache wholesale and remember it's authoritative.
        self._global_cache = dict(result)
        self._global_fully_loaded = True
        return result

    # ------------------------------------------------------------------
    # Per-user (user_settings) — tenant-scoped values
    # ------------------------------------------------------------------

    async def get_user(self, user_id: str, key: str) -> str | None:
        """Return ``user_id``'s value for *key*, or ``None`` if unset.

        Does NOT fall back to the global value — callers that want
        fallback use :meth:`get_user_or_global`.
        """
        if not user_id:
            return None
        ck = (user_id, key)
        if ck in self._user_cache:
            return self._user_cache[ck]
        if user_id in self._user_fully_loaded:
            return None
        cursor = await self._conn.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        row = await cursor.fetchone()
        val = row[0] if row else None
        self._user_cache[ck] = val
        return val

    async def set_user(self, user_id: str, key: str, value: str | None) -> None:
        """Persist a per-user override for *key*. Pass ``None`` to delete."""
        if not user_id:
            raise ValueError("set_user requires a non-empty user_id")

        async def _write():
            if value is None:
                await self._conn.execute(
                    "DELETE FROM user_settings WHERE user_id = ? AND key = ?",
                    (user_id, key),
                )
            else:
                await self._conn.execute(
                    "INSERT INTO user_settings (user_id, key, value, updated_at) "
                    "VALUES (?, ?, ?, datetime('now')) "
                    "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (user_id, key, value),
                )
            await self._conn.commit()
        await _retry_on_lock(_write, op_name="settings.set_user")
        self._user_cache[(user_id, key)] = value

    async def get_user_or_global(self, user_id: str, key: str) -> str | None:
        """Return ``user_id``'s value for *key*, falling back to the
        install-wide value if the user has no override."""
        user_val = await self.get_user(user_id, key)
        if user_val is not None:
            return user_val
        return await self.get(key)

    async def get_all_user(self, user_id: str) -> dict[str, str]:
        """Return all per-user overrides for ``user_id`` as a dict."""
        if not user_id:
            return {}
        cursor = await self._conn.execute(
            "SELECT key, value FROM user_settings WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        result = {row[0]: row[1] for row in rows}
        # Refresh this user's slice of the cache and mark it authoritative
        # (drop any stale per-key entries for this user first).
        self._user_cache = {
            ck: v for ck, v in self._user_cache.items() if ck[0] != user_id
        }
        for k, v in result.items():
            self._user_cache[(user_id, k)] = v
        self._user_fully_loaded.add(user_id)
        return result

    async def has_any_user_value(self, key: str, value: str) -> bool:
        """Return True if at least one row in ``user_settings`` has ``key = value``.

        Used by process-singleton subsystems (dreams, compactor, …) that need
        to decide "does anyone want me running?" without loading every user's
        settings. The composite PK on ``(user_id, key)`` makes this a single
        covering lookup in SQLite.
        """
        cursor = await self._conn.execute(
            "SELECT 1 FROM user_settings WHERE key = ? AND value = ? LIMIT 1",
            (key, value),
        )
        row = await cursor.fetchone()
        return row is not None
