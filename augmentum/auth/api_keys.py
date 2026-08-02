"""Inbound API keys.

Lets external OpenAI-compatible clients (OpenWebUI, SillyTavern,
Cursor, etc.) authenticate to ``/v1/*`` and ``/api/*`` without a
browser session. Distinct from ``user_api_keys`` (071_users_auth.sql)
which stores OUTBOUND keys the user gives us for upstream providers.

Key format:
    sk-aug-<32 url-safe chars>
Prefix lets the middleware short-circuit on shape before doing a DB
lookup. Storage is the SHA-256 of the raw key; the raw value is
returned to the user exactly once at creation.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import TYPE_CHECKING

from augmentum.auth.models import User
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Public prefix for inbound API keys. ``sk-`` makes most OpenAI clients
# treat it as a valid Bearer token; ``-aug-`` namespaces it so the
# middleware can route on the prefix without parsing.
KEY_PREFIX = "sk-aug-"

# 24 bytes random → ~32 URL-safe chars → 192 bits entropy.
_KEY_BODY_BYTES = 24


def is_api_key(token: str) -> bool:
    """Cheap prefix check used by the middleware to dispatch."""
    return bool(token) and token.startswith(KEY_PREFIX)


def _hash(raw: str) -> str:
    """Storage hash for an API key. SHA-256 is fine — the raw key has
    192 bits of entropy, so we don't need a slow KDF."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_raw() -> str:
    """Create a fresh API key (returned to the user once, never stored)."""
    return KEY_PREFIX + secrets.token_urlsafe(_KEY_BODY_BYTES)


class ApiKeyManager:
    """Manage inbound API keys backed by ``augmentum_api_keys``.

    Holds the same aiosqlite connection as ``SessionManager`` and a
    small in-memory cache so the per-request validation cost stays
    sub-millisecond on the hot path.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        # Hash → (user, cached_at_monotonic). Same TTL pattern as
        # SessionManager._token_cache so behavior is consistent.
        self._cache: dict[str, tuple[User, float]] = {}
        self._cache_ttl = 60.0

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(
        self, user_id: str, name: str = "", scope: str = "chat",
    ) -> tuple[str, dict]:
        """Create a new API key for ``user_id``.

        Returns ``(raw_key, metadata)``. The raw key is the only time
        the caller can see it; we only persist its hash. ``metadata``
        is the row representation suitable for an API response.
        """
        raw = _generate_raw()
        key_hash = _hash(raw)
        key_id = f"akey_{secrets.token_hex(8)}"
        prefix = raw[: len(KEY_PREFIX) + 5]  # e.g. "sk-aug-AbCdE"
        scope = (scope or "chat").strip().lower()
        if scope not in ("chat", "admin"):
            scope = "chat"
        await self._db.execute(
            """INSERT INTO augmentum_api_keys
               (id, user_id, name, key_hash, key_prefix, scope)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key_id, user_id, name or "", key_hash, prefix, scope),
        )
        await self._db.commit()
        log.info("api_key_created", user_id=user_id, key_id=key_id, scope=scope)
        return raw, {
            "id": key_id,
            "name": name or "",
            "prefix": prefix,
            "scope": scope,
            "created_at": "",  # populated on subsequent list calls
            "last_used_at": None,
        }

    async def ensure(
        self, user_id: str, name: str, raw_key: str, scope: str = "chat",
    ) -> tuple[str, dict]:
        """Idempotently ensure a SPECIFIC (caller-supplied) key exists.

        Unlike :meth:`create` (random, unrecoverable), this takes an
        externally-derived raw key — used for STABLE per-service keys that must
        survive an app's reinstall. The app persists the raw key in its own
        config volume; deriving the same value (see ``derive_secret``) and
        re-inserting its hash keeps the two matched across an uninstall +
        reinstall, so the app doesn't silently 401 on a now-revoked old key.
        Reuses the existing row if the hash is already present.
        """
        key_hash = _hash(raw_key)
        cur = await self._db.execute(
            "SELECT id FROM augmentum_api_keys WHERE key_hash = ?", (key_hash,),
        )
        row = await cur.fetchone()
        scope = (scope or "chat").strip().lower()
        if scope not in ("chat", "admin"):
            scope = "chat"
        if row:
            return raw_key, {"id": row[0], "name": name or "", "scope": scope}
        key_id = f"akey_{secrets.token_hex(8)}"
        await self._db.execute(
            """INSERT INTO augmentum_api_keys
               (id, user_id, name, key_hash, key_prefix, scope)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key_id, user_id, name or "", key_hash,
             raw_key[: len(KEY_PREFIX) + 5], scope),
        )
        await self._db.commit()
        log.info("api_key_ensured", user_id=user_id, key_id=key_id, scope=scope)
        return raw_key, {"id": key_id, "name": name or "", "scope": scope}

    async def list_for_user(self, user_id: str) -> list[dict]:
        """List a user's API keys (metadata only — never returns hashes
        or any value from which the raw key could be recovered)."""
        cursor = await self._db.execute(
            """SELECT id, name, key_prefix, scope, created_at, last_used_at
               FROM augmentum_api_keys WHERE user_id = ?
               ORDER BY created_at DESC""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1] or "",
                "prefix": row[2],
                "scope": row[3],
                "created_at": row[4],
                "last_used_at": row[5],
            }
            for row in rows
        ]

    async def revoke(self, key_id: str, user_id: str) -> bool:
        """Delete a key. Returns True when a row was actually removed
        (so callers can distinguish 404 from 200)."""
        # Look up hash first so we can invalidate the cache after delete.
        cursor = await self._db.execute(
            "SELECT key_hash FROM augmentum_api_keys WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        key_hash = row[0]

        await self._db.execute(
            "DELETE FROM augmentum_api_keys WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await self._db.commit()
        self._cache.pop(key_hash, None)
        log.info("api_key_revoked", user_id=user_id, key_id=key_id)
        return True

    # ------------------------------------------------------------------
    # Validation (hot path)
    # ------------------------------------------------------------------

    async def validate(self, raw_key: str) -> User | None:
        """Resolve a raw API key to its owning user, or None.

        Cached for ``_cache_ttl`` seconds so a chatty client doesn't
        round-trip SQLite on every request.
        """
        if not is_api_key(raw_key):
            return None

        key_hash = _hash(raw_key)
        now = time.monotonic()
        cached = self._cache.get(key_hash)
        if cached and now - cached[1] < self._cache_ttl:
            return cached[0]

        cursor = await self._db.execute(
            """SELECT u.id, u.username, u.display_name, u.role, u.is_active,
                      u.quota_bytes, u.created_at, u.updated_at
               FROM augmentum_api_keys k
               JOIN users u ON k.user_id = u.id
               WHERE k.key_hash = ?""",
            (key_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            self._cache.pop(key_hash, None)
            return None

        user = User(
            id=row[0], username=row[1], display_name=row[2], role=row[3],
            is_active=bool(row[4]), quota_bytes=row[5], created_at=row[6],
            updated_at=row[7],
        )
        if not user.is_active:
            return None

        self._cache[key_hash] = (user, now)

        # Debounced last_used update. Skipped on cache hits so a
        # busy client doesn't write per request.
        if cached is None:
            await self._db.execute(
                "UPDATE augmentum_api_keys SET last_used_at = datetime('now') "
                "WHERE key_hash = ?",
                (key_hash,),
            )
            # Piggyback commit on the next write.

        return user

    def invalidate_cache(self) -> None:
        """Drop the entire validation cache. Used by admin operations
        that may have invalidated keys without going through ``revoke``."""
        self._cache.clear()

    def invalidate_user_cache(self, user_id: str) -> None:
        """Drop cached entries for all keys belonging to ``user_id``.

        Called when a user's ``is_active`` or role changes so a
        deactivated user can't continue using a cached key for up to
        ``_cache_ttl`` seconds. ``SessionManager`` chains this from
        its own ``_invalidate_user_cache`` so a single ``update_user``
        flushes both caches.
        """
        if not user_id:
            return
        to_remove = [
            h for h, (user, _) in self._cache.items()
            if user.id == user_id
        ]
        for h in to_remove:
            self._cache.pop(h, None)
