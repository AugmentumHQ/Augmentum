"""Auth session token management, lockout enforcement, and WS tickets."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.auth.models import User, WsTicket, is_reserved_username
from augmentum.auth.passwords import hash_password
from augmentum.config import settings
from augmentum.utils.logging import get_logger


def _hash_token(raw: str) -> str:
    """Storage hash for an auth session token. SHA-256 — the raw token has
    256 bits of entropy from secrets.token_hex(32), so a fast hash is fine
    (no offline brute-force concern). Validate path tolerates short or
    malformed client input — the hash just won't match anything in the DB.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Minimum raw-token length we will MINT. The single-pass SHA-256 in
# `_hash_token` is only safe while the minter keeps ~128+ bits of entropy
# in the raw token. If anyone ever shortens the mint to e.g. token_hex(8),
# the stored hashes become offline-brute-forceable. This floor catches
# that regression at session-create time. Validate-time inputs are NOT
# checked (a short cookie from a client just doesn't match — that's fine).
_MIN_MINTED_TOKEN_CHARS = 32


def _canonical_username(raw: str) -> str:
    """Canonical form for username comparison + storage.

    Login lockout (per-username threshold) and the UNIQUE constraint on
    users.username both lean on case-insensitive equality — without
    canonicalisation, an attacker can rotate `Bob` / `bob` / `BOB` to
    multiply their per-username attempt budget, and two users could
    register `Alice` and `alice` independently. casefold() is the
    Unicode-aware lowercase ('ß' → 'ss', 'Σ' → 'σ') and matches what
    Python's `str.lower()` does for ASCII.
    """
    return raw.casefold().strip()


def _normalise_session_source(raw: str) -> str:
    """Compact source tag stored on auth_sessions."""
    source = "".join(
        ch for ch in (raw or "web").strip().lower()
        if ch.isalnum() or ch in ("_", "-")
    )
    return (source or "web")[:40]


# Session sources that are long-lived paired-device credentials (an
# always-on TV receiver, a trusted phone) rather than interactive
# logins. These are pruned per-source in ``create_session`` instead of
# sharing the browser LRU pool — see the comment at the prune site.
# Values must match what the mint sites pass AFTER normalisation
# (cast_routes.py establish-session, mobile_pair_routes.py).
_DEVICE_SESSION_SOURCES = frozenset({"cast_receiver", "android"})


if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


async def _retry_on_locked(
    fn,
    *,
    op: str,
    attempts: int = 3,
    base_delay_s: float = 0.2,
) -> None:
    """Run ``fn`` with a small backoff on SQLITE_BUSY.

    The two best-effort auth hygiene writes (record/clear of
    ``failed_login_attempts``) sit on the login hot path and have no
    business 500'ing during a transient lock storm. We try a few
    times with linear backoff, then log + swallow. The caller never
    learns the difference; the stored attempt count drifts by at most
    a few rows over the lock window which is fine — lockout still
    works because the threshold is in the high single digits and
    losing one row out of N doesn't flip the verdict.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            await fn()
            return
        except Exception as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                # Not a transient — bubble up so we don't mask a real
                # bug behind blanket retry.
                raise
            last_exc = exc
            if i < attempts - 1:
                await asyncio.sleep(base_delay_s * (i + 1))
    log.warning(
        "auth_hygiene_write_dropped",
        op=op,
        attempts=attempts,
        error=str(last_exc) if last_exc else "",
    )


class SessionManager:
    """Manages auth sessions, login attempts, and WebSocket tickets."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._ws_tickets: dict[str, WsTicket] = {}
        # In-memory LRU cache for token → User lookups.
        # 5min TTL: short enough that a revoked token loses its cache slot
        # quickly, long enough that the auth path doesn't churn the DB.
        # Prior value (60s) caused 51 of 86 slow_db_ops in a typical
        # 30-min window as every active session retried the DB lookup
        # every minute.
        self._token_cache: dict[str, tuple[User, float]] = {}
        self._cache_ttl = 300.0  # seconds (5 minutes)
        # Per-token timestamp of the last in-DB last_activity bump.
        # Lets us throttle the UPDATE without re-reading the column —
        # 5min gates with no fsync cost on the hot path.
        self._last_activity_writes: dict[str, float] = {}
        self._last_activity_min_interval = 300.0  # seconds
        # Sibling caches we chain invalidation through when a user's
        # ``is_active`` / role changes. Today: ApiKeyManager. The
        # callback-list form keeps the dependency one-directional —
        # SessionManager doesn't import the api_keys module.
        self._user_cache_invalidators: list = []

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    async def create_user(
        self, username: str, password: str, *, role: str = "user",
        display_name: str = "", email: str = "",
    ) -> User:
        """Create a new user account.

        Stores ``username`` in canonical (casefolded) form so duplicate
        registrations differing only in case fail the UNIQUE constraint
        instead of creating two distinct accounts. ``display_name`` keeps
        the original casing for UI presentation.

        Also binds the companion to this user when no owner is set yet —
        mirrors the auto-bind path in ``CompanionRuntime._resolve_owner_
        user_id``, but at signup time so dreams + drift audits resolve
        without waiting for the next container restart. Multi-user
        installs that have already explicitly bound an owner are left
        alone (we don't clobber the existing value).

        Reserved-name defense: rejects any name matching the centralised
        list in ``augmentum/auth/models.py::RESERVED_USERNAMES`` or
        ``RESERVED_USERNAME_PREFIXES``. Route handlers should validate
        BEFORE reaching the CRUD layer (they return a clean 400) — this
        check is a belt-and-suspenders backstop so a future call site
        that forgot the route-level check still can't squat a reserved
        name. ``ValueError`` is the right signal for programmer-error
        paths; user-facing surfaces should validate up-front.
        """
        canon = _canonical_username(username)
        if is_reserved_username(canon):
            raise ValueError(f"Username '{canon}' is reserved")
        user_id = f"usr_{secrets.token_hex(8)}"
        pw_hash = hash_password(password)
        await self._db.execute(
            """INSERT INTO users (id, username, display_name, password_hash, role, email)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, canon, display_name or username, pw_hash, role, (email or "").strip()),
        )
        await self._bind_companion_if_unset(user_id)
        await self._db.commit()
        log.info("user_created", user_id=user_id, username=canon, role=role)
        return User(
            id=user_id, username=canon, display_name=display_name or username,
            role=role, email=(email or "").strip(),
        )

    async def create_first_admin(
        self, username: str, password: str, *, display_name: str = "",
    ) -> User | None:
        """Atomic create-first-admin for the setup endpoint.

        Returns the new admin User on success, or ``None`` if another
        request already populated the ``users`` table while this one was
        in flight. The check-and-insert collapses to a single statement
        (``INSERT ... WHERE NOT EXISTS (SELECT 1 FROM users LIMIT 1)``)
        which SQLite serialises via the database write lock, closing the
        TOCTOU race in the old ``user_count()`` then ``create_user()``
        pattern. The race window mattered on fresh deploys where two
        concurrent POSTs to ``/api/auth/setup`` could both observe
        count=0 and both produce admin accounts.

        Callers must still check the return value: ``None`` means setup
        was completed by another caller — treat the same as a 403.

        Reserved-name defense (same as ``create_user``): rejects reserved
        names via the centralised list. The setup route should validate
        first; this is the backstop.
        """
        canon = _canonical_username(username)
        if is_reserved_username(canon):
            raise ValueError(f"Username '{canon}' is reserved")
        user_id = f"usr_{secrets.token_hex(8)}"
        pw_hash = hash_password(password)
        # Atomic check-and-insert: refuse if ANY user exists OR the durable
        # setup latch is set. Both subqueries live inside the one INSERT
        # statement SQLite serialises via the write lock, so N racing
        # requests still produce at most one admin — and the latch means a
        # later-emptied users table can't silently re-arm this endpoint.
        cur = await self._db.execute(
            """INSERT INTO users (id, username, display_name, password_hash, role)
               SELECT ?, ?, ?, ?, 'admin'
               WHERE NOT EXISTS (SELECT 1 FROM users LIMIT 1)
                 AND NOT EXISTS (
                     SELECT 1 FROM app_settings
                      WHERE key = 'auth_setup_completed' AND value = '1'
                 )""",
            (user_id, canon, display_name or username, pw_hash),
        )
        rows_affected = cur.rowcount
        await cur.close()
        if not rows_affected:
            # Lost the race, or the durable setup latch is already set —
            # either way setup is closed. Roll back and treat as a 403.
            await self._db.rollback()
            log.info("create_first_admin_lost_race", attempted_username=canon)
            return None
        # Latch setup closed in the SAME transaction as the admin insert, so
        # "first admin exists" is durable even if the users row is later
        # removed (migration rebuild, last-user delete, partial restore).
        # See setup_completed() / ensure_setup_latch().
        await self._db.execute(
            """INSERT INTO app_settings (key, value, updated_at)
               VALUES ('auth_setup_completed', '1', datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                   value = '1', updated_at = datetime('now')""",
        )
        await self._bind_companion_if_unset(user_id)
        await self._db.commit()
        log.info("first_admin_created", user_id=user_id, username=canon)
        return User(
            id=user_id, username=canon, display_name=display_name or username,
            role="admin",
        )

    async def _bind_companion_if_unset(self, user_id: str) -> None:
        """If ``companion_default_owner_user_id`` is empty, bind it to
        ``user_id`` so the companion runtime can resolve who to dream
        about. Idempotent + safe to skip silently — auth flow shouldn't
        fail because the companion subsystem isn't initialized yet
        (fresh install, missing tables, etc).

        Touches two pieces of state:

          * ``app_settings.companion_default_owner_user_id`` — the
            explicit override the CompanionRuntime reads via
            ``settings.companion_default_owner_user_id`` at start.
            Persisted so a container restart still has the binding.

          * ``companion_identities.owner_user_id`` for the global
            ``companion_id='becca'`` row — applied immediately so the
            currently-running runtime sees an owner on next identity
            read, not just after restart.
        """
        try:
            cur = await self._db.execute(
                "SELECT value FROM app_settings WHERE key = ?",
                ("companion_default_owner_user_id",),
            )
            row = await cur.fetchone()
            await cur.close()
        except Exception:
            log.debug("companion_owner_autobind_settings_read_failed", exc_info=True)
            return

        existing = (row[0] if row else "") or ""
        if existing.strip():
            # Already pinned to a user; respect that.
            return

        try:
            await self._db.execute(
                """INSERT INTO app_settings (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET
                       value=excluded.value, updated_at=datetime('now')""",
                ("companion_default_owner_user_id", user_id),
            )
            await self._db.execute(
                """UPDATE companion_identities
                      SET owner_user_id = ?
                    WHERE companion_id = 'becca'
                      AND (owner_user_id IS NULL OR owner_user_id = '')""",
                (user_id,),
            )
            log.info(
                "companion_owner_autobound_on_signup",
                user_id=user_id,
                note=(
                    "first signup with no explicit owner — companion "
                    "now bound to this user; restart picks up explicit "
                    "setting"
                ),
            )
        except Exception:
            # If companion_identities doesn't exist yet (migrations not
            # run, schema drift, etc.) the auto-bind silently degrades —
            # the explicit setting still persists for next restart.
            log.warning(
                "companion_owner_autobind_failed", user_id=user_id,
                exc_info=True,
            )

    async def get_user_by_username(self, username: str) -> User | None:
        """Lookup user by username (case-insensitive)."""
        cursor = await self._db.execute(
            "SELECT id, username, display_name, password_hash, role, is_active, "
            "quota_bytes, created_at, updated_at, content_level FROM users WHERE username = ?",
            (_canonical_username(username),),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return User(
            id=row[0], username=row[1], display_name=row[2], role=row[4],
            is_active=bool(row[5]), quota_bytes=row[6],
            created_at=row[7], updated_at=row[8],
            content_level=row[9] or "unrestricted",
        )

    async def get_or_create_fabric_peer_user(
        self, sender_node_id: str, *, hostname: str = "",
    ) -> User | None:
        """Get-or-create the local user that represents a remote fabric peer.

        Cross-peer dispatch arrives at this node with a signed envelope from
        peer P. The envelope proves the sender's identity; the dispatch
        still needs to run AS some local user (for data isolation, store
        scoping, handler caches, etc.). Local user accounts don't span
        peers (a fresh install on host B has no knowledge of host A's
        ``usr_<hex>`` ids), so we mint a per-peer service user the first
        time we receive a request from that peer.

        Properties of the peer-user:

          - id is ``fabric:<short-node-id>`` (deterministic per peer, so
            repeated dispatches consistently resolve to the same row)
          - role is ``"peer"`` so admin/UI filters can exclude these
          - password_hash is a sentinel that never matches any password
            (peers never log in via the normal flow)
          - data created by this user (chats, images, etc.) is owned by
            it, providing isolation at the peer boundary

        ``hostname`` is informational only; used as display_name on
        creation. Returns ``None`` only on DB failure.
        """
        if not sender_node_id:
            return None

        short = sender_node_id[:16]
        peer_user_id = f"fabric:{short}"
        # Try the cheap read path first.
        existing = await self.get_user_by_id(peer_user_id)
        if existing is not None and existing.is_active:
            return existing

        # Need to create. Use a stable username (no random suffix) so
        # collisions on the UNIQUE constraint resolve to the existing
        # row deterministically. Username goes through ``_canonical_
        # username`` so case-insensitive lookups work.
        username = f"fabric_peer_{short}"
        canon = _canonical_username(username)
        display = hostname or f"Fabric peer {short[:8]}"
        # Sentinel hash that won't match any password the verify path
        # could generate. ``hash_password`` produces an Argon2id string
        # starting with ``$argon2id$``; a literal that doesn't is safe
        # against accidental match.
        sentinel_hash = "DISABLED:fabric-peer:no-login"
        try:
            await self._db.execute(
                """INSERT INTO users (id, username, display_name, password_hash, role)
                   VALUES (?, ?, ?, ?, ?)""",
                (peer_user_id, canon, display, sentinel_hash, "peer"),
            )
            await self._db.commit()
            log.info(
                "fabric_peer_user_created",
                peer_user_id=peer_user_id, sender_node_id=sender_node_id,
            )
        except Exception:
            # UNIQUE collision in a concurrent dispatch is the most
            # likely cause — re-read and return whichever row won.
            log.debug("fabric_peer_user_create_collision", exc_info=True)
            await self._db.rollback()
            return await self.get_user_by_id(peer_user_id)

        return await self.get_user_by_id(peer_user_id)

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Lookup user by ID."""
        cursor = await self._db.execute(
            "SELECT id, username, display_name, password_hash, role, is_active, "
            "quota_bytes, created_at, updated_at, content_level FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return User(
            id=row[0], username=row[1], display_name=row[2], role=row[4],
            is_active=bool(row[5]), quota_bytes=row[6],
            created_at=row[7], updated_at=row[8],
            content_level=row[9] or "unrestricted",
        )

    async def get_password_hash(self, username: str) -> str | None:
        """Get just the password hash for a username (for verify)."""
        cursor = await self._db.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (_canonical_username(username),),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def list_users(self) -> list[User]:
        """List all users."""
        cursor = await self._db.execute(
            "SELECT id, username, display_name, password_hash, role, is_active, "
            "quota_bytes, created_at, updated_at, content_level FROM users ORDER BY created_at",
        )
        rows = await cursor.fetchall()
        return [
            User(id=r[0], username=r[1], display_name=r[2], role=r[4],
                 is_active=bool(r[5]), quota_bytes=r[6], created_at=r[7], updated_at=r[8],
                 content_level=r[9] or "unrestricted")
            for r in rows
        ]

    async def user_count(self) -> int:
        """Return total number of users."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM users")
        row = await cursor.fetchone()
        return row[0]

    async def setup_completed(self) -> bool:
        """Whether first-run admin setup has already happened — DURABLY.

        The primary signal is the persisted ``auth_setup_completed`` latch
        (set atomically with the first admin in ``create_first_admin``); the
        live ``user_count`` is the fallback. Reading the latch first means
        setup stays CLOSED even if the users table is later emptied — a
        migration that rebuilds it, the last user deleted, a partial DB
        restore — closing the "users looks empty → admin creation re-arms"
        privilege-escalation-on-reset window the bare count check left open.
        """
        try:
            cur = await self._db.execute(
                "SELECT 1 FROM app_settings "
                "WHERE key = 'auth_setup_completed' AND value = '1' LIMIT 1"
            )
            row = await cur.fetchone()
            await cur.close()
            if row:
                return True
        except Exception:
            # app_settings unreadable — defer to the user_count signal
            # rather than fail open.
            log.debug("setup_latch_read_failed", exc_info=True)
        return (await self.user_count()) > 0

    async def ensure_setup_latch(self) -> None:
        """Backfill the durable setup latch for installs created before it
        existed. Idempotent: if any user exists and the latch isn't set, set
        it — so a pre-existing install can't have setup re-open if its users
        table is later emptied. No-op on a genuinely fresh install (no users
        → no latch → first-run setup stays available). Called once at startup.
        """
        try:
            cur = await self._db.execute(
                "SELECT 1 FROM app_settings "
                "WHERE key = 'auth_setup_completed' AND value = '1' LIMIT 1"
            )
            already = await cur.fetchone()
            await cur.close()
            if already:
                return
            if (await self.user_count()) > 0:
                await self._db.execute(
                    """INSERT INTO app_settings (key, value, updated_at)
                       VALUES ('auth_setup_completed', '1', datetime('now'))
                       ON CONFLICT(key) DO UPDATE SET
                           value = '1', updated_at = datetime('now')"""
                )
                await self._db.commit()
                log.info("auth_setup_latch_backfilled")
        except Exception:
            log.debug("setup_latch_backfill_failed", exc_info=True)

    async def active_admin_count(self) -> int:
        """Return count of active admin users — used for last-admin protection."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def ensure_admin_exists(self) -> str | None:
        """Self-heal: every install needs an owner/admin.

        If no active admin exists, promote the longest-standing active
        user to admin and return their id. Complements
        ``create_first_admin`` (the setup-flow path): it covers installs
        whose users were created outside that flow (legacy rows, direct
        provisioning), which otherwise leave admin-only settings — e.g.
        coder subagent dispatch, ``PUT /api/config/tools`` — permanently
        un-toggleable for the sole operator (the controls disable + any
        save 403s and reverts).

        Idempotent and safe: a no-op the moment any active admin exists,
        so it never touches a configured multi-tenant install (which the
        ``_would_remove_last_admin`` guard already keeps non-empty). On a
        genuinely fresh install (zero users) it also no-ops — first-run
        setup creates the admin.
        """
        if await self.active_admin_count() > 0:
            return None
        cursor = await self._db.execute(
            "SELECT id, username FROM users WHERE is_active = 1 "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        user_id, username = row[0], row[1]
        await self._db.execute(
            "UPDATE users SET role = 'admin', updated_at = datetime('now') "
            "WHERE id = ?",
            (user_id,),
        )
        await self._db.commit()
        self._invalidate_user_cache(user_id)
        log.warning(
            "auth_admin_self_heal_promoted",
            user_id=user_id,
            username=username,
            reason="no active admin existed; promoted longest-standing user",
        )
        return user_id

    async def _would_remove_last_admin(self, user_id: str, *, new_role: str | None = None,
                                       new_active: bool | None = None,
                                       deleting: bool = False) -> bool:
        """Return True if applying the change to user_id would leave 0 active admins."""
        target = await self.get_user_by_id(user_id)
        if not target or target.role != "admin" or not target.is_active:
            return False  # target isn't an active admin → can't remove the last one
        if not deleting and new_role in (None, "admin") and new_active in (None, True):
            return False  # change doesn't affect admin status
        return await self.active_admin_count() <= 1

    async def update_user(self, user_id: str, **fields) -> bool:
        """Update user fields. Allowed: display_name, role, is_active, quota_bytes, content_level."""
        allowed = {"display_name", "role", "is_active", "quota_bytes", "content_level"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        # Reject unknown content_level values so a typo can't silently
        # bypass filtering. Treat anything not in the canonical set as
        # 'unrestricted' would be a security footgun.
        if "content_level" in updates and updates["content_level"] not in ("unrestricted", "family"):
            return False
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]
        await self._db.execute(
            f"UPDATE users SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
            values,
        )
        await self._db.commit()
        # Invalidate cache entries for this user
        self._invalidate_user_cache(user_id)
        return True

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user and every user_id-scoped row they own.

        Only ~10 user-scoped tables declare ``ON DELETE CASCADE`` (auth_sessions,
        user_api_keys, user_settings, user_media_servers, augmentum_api_keys,
        media_library_views, saved_devices/device_pairings/device_play_history,
        vocab_state, game_results). The other ~80 use plain ``REFERENCES
        users(id)`` with default NO ACTION — historically because adding
        cascade in SQLite requires table recreation per migration.

        Rather than recreate 80 tables, we discover every user-scoped table
        at runtime (any table with a ``user_id`` column) and explicitly
        DELETE matching rows inside a transaction with deferred FK
        enforcement. Deferring means FK violations are tolerated mid-txn
        as long as the committed state is consistent — which it is here,
        since we're removing the user row plus every reference to it. FK
        enforcement at the connection level remains ON throughout;
        ``PRAGMA defer_foreign_keys`` is per-transaction and auto-resets
        at COMMIT.

        Audit log rows are preserved by design — ``auth_audit_log`` uses
        ``actor_user_id`` / ``target_user_id`` column names, so it isn't
        picked up by the ``user_id``-column discovery query.
        """
        # Discover all user-scoped tables. Exclude the users table itself
        # (its column is 'id', not 'user_id') and sqlite internals.
        async with self._db.execute(
            "SELECT m.name FROM sqlite_master m "
            "WHERE m.type = 'table' "
            "  AND m.name NOT LIKE 'sqlite_%' "
            "  AND m.name != 'users' "
            "  AND EXISTS ("
            "      SELECT 1 FROM pragma_table_info(m.name) p "
            "      WHERE p.name = 'user_id'"
            "  )"
        ) as cur:
            tables = [row[0] for row in await cur.fetchall()]

        # Flush any pending implicit transaction so our explicit BEGIN
        # starts cleanly and PRAGMA defer_foreign_keys applies to it.
        await self._db.commit()

        deleted_counts: dict[str, int] = {}
        deleted_user = False
        await self._db.execute("BEGIN")
        try:
            await self._db.execute("PRAGMA defer_foreign_keys = ON")
            for table in tables:
                cursor = await self._db.execute(
                    f'DELETE FROM "{table}" WHERE user_id = ?',
                    (user_id,),
                )
                if cursor.rowcount > 0:
                    deleted_counts[table] = cursor.rowcount
            cursor = await self._db.execute(
                "DELETE FROM users WHERE id = ?", (user_id,),
            )
            deleted_user = cursor.rowcount > 0
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

        self._invalidate_user_cache(user_id)

        if deleted_user:
            # On-disk project bare repos live outside any table (see
            # [[project_user_deletion_strands_data]] +
            # docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md
            # risk register: "bare repo dir on disk not in any table →
            # easy to miss in delete_user"). Sweep here so the project
            # entity cascade is complete. Best-effort: log + swallow on
            # failure, since the DB cascade is the source of truth.
            try:
                projects_dir = Path(settings.data_dir) / "projects" / user_id
                if projects_dir.exists():
                    await asyncio.to_thread(
                        shutil.rmtree, projects_dir, ignore_errors=True,
                    )
            except Exception:
                log.warning(
                    "auth.user_delete_projects_dir_cleanup_failed",
                    user_id=user_id,
                    exc_info=True,
                )

            log.info(
                "auth.user_deleted",
                user_id=user_id,
                tables_cleared=len(deleted_counts),
                row_counts=deleted_counts,
            )
        return deleted_user

    async def write_audit(self, *, actor: User | None, target: User | None,
                          action: str, detail: str = "", ip_address: str = "") -> None:
        """Append an entry to the auth audit log. Best-effort — failures are logged but not raised."""
        try:
            await self._db.execute(
                """INSERT INTO auth_audit_log
                   (actor_user_id, actor_username, target_user_id, target_username,
                    action, detail, ip_address)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    actor.id if actor else None,
                    actor.username if actor else None,
                    target.id if target else None,
                    target.username if target else None,
                    action,
                    detail or "",
                    ip_address or "",
                ),
            )
            await self._db.commit()
        except Exception:
            log.warning("auth_audit_write_failed", action=action, exc_info=True)

    async def list_audit(self, limit: int = 100) -> list[dict]:
        """Return recent audit entries (newest first)."""
        limit = max(1, min(int(limit or 100), 500))
        cursor = await self._db.execute(
            """SELECT id, actor_username, target_username, action, detail,
                      ip_address, created_at
               FROM auth_audit_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "actor": r[1] or "",
                "target": r[2] or "",
                "action": r[3],
                "detail": r[4] or "",
                "ip_address": r[5] or "",
                "created_at": r[6],
            }
            for r in rows
        ]

    async def update_password(self, user_id: str, new_password: str) -> None:
        """Update a user's password hash and revoke all OTHER sessions."""
        pw_hash = hash_password(new_password)
        await self._db.execute(
            "UPDATE users SET password_hash = ?, updated_at = datetime('now') WHERE id = ?",
            (pw_hash, user_id),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Session tokens
    # ------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        ip_address: str = "",
        user_agent: str = "",
        *,
        source: str = "web",
        source_device_id: str = "",
        ttl_hours: float | None = None,
    ) -> str:
        """Create a new auth session. Returns the opaque raw token to the
        caller (set as cookie); only the SHA-256 hash is persisted, so a
        leaked DB backup can't be replayed against the live server.

        ``ttl_hours`` overrides the global ``auth_session_ttl_hours`` for
        this session only — used by the cast pairing flow so a home TV
        gets a long-lived credential (silent reconnect across restarts)
        while an away/public TV gets a short one (re-pair sooner)."""
        raw = secrets.token_hex(32)
        # Regression guard against accidentally shortening the mint —
        # see _MIN_MINTED_TOKEN_CHARS for rationale.
        if len(raw) < _MIN_MINTED_TOKEN_CHARS:
            raise RuntimeError(
                f"refusing to mint a session token shorter than "
                f"{_MIN_MINTED_TOKEN_CHARS} chars",
            )
        token_hash = _hash_token(raw)
        ttl = ttl_hours if (ttl_hours and ttl_hours > 0) else settings.auth_session_ttl_hours
        expires = (datetime.utcnow() + timedelta(hours=ttl)).isoformat()
        source = _normalise_session_source(source)
        source_device_id = (source_device_id or "").strip()[:160]
        await self._db.execute(
            """INSERT INTO auth_sessions
                  (token, user_id, expires_at, ip_address, user_agent,
                   source, source_device_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                token_hash,
                user_id,
                expires,
                ip_address,
                user_agent,
                source,
                source_device_id,
            ),
        )
        await self._db.commit()

        # Enforce max sessions per user. Order by ``last_activity`` so
        # active sessions are preserved over stale ones — a fresh login
        # from a second tab shouldn't boot the browser cookie you're
        # actively using just because it was created hours ago.
        # ``created_at`` is the tiebreaker for sessions that share a
        # last_activity timestamp (e.g. a brand-new session whose
        # validate_token hasn't bumped its activity yet). The newly
        # inserted row above always wins because its created_at is the
        # most recent.
        #
        # Device-class sessions (paired always-on devices: cast receivers,
        # trusted mobile) are pruned WITHIN their own source only. An idle
        # TV is by definition the stalest session a user owns, so sharing
        # one LRU pool with browser logins meant every burst of web/agent
        # logins silently evicted the TV's 1-year "home" credential and
        # forced a QR re-pair (2026-07-01 incident). Expired rows are
        # reaped separately by cleanup_expired(), so the exemption can't
        # accumulate dead sessions.
        max_sess = settings.auth_max_sessions_per_user
        if source in _DEVICE_SESSION_SOURCES:
            if source_device_id:
                # A re-paired device replaces its own previous sessions —
                # the device holds exactly one cookie, so older rows for
                # the same physical device are dead weight.
                await self._db.execute(
                    """DELETE FROM auth_sessions
                       WHERE user_id = ? AND source = ?
                         AND source_device_id = ? AND token != ?""",
                    (user_id, source, source_device_id, token_hash),
                )
            await self._db.execute(
                """DELETE FROM auth_sessions
                   WHERE user_id = ? AND source = ?
                     AND token NOT IN (
                       SELECT token FROM auth_sessions
                       WHERE user_id = ? AND source = ?
                       ORDER BY last_activity DESC, created_at DESC
                       LIMIT ?
                     )""",
                (user_id, source, user_id, source, max_sess),
            )
        else:
            _dev = tuple(sorted(_DEVICE_SESSION_SOURCES))
            _ph = ",".join("?" * len(_dev))
            await self._db.execute(
                f"""DELETE FROM auth_sessions
                   WHERE user_id = ?
                     AND source NOT IN ({_ph})
                     AND token NOT IN (
                       SELECT token FROM auth_sessions
                       WHERE user_id = ?
                         AND source NOT IN ({_ph})
                       ORDER BY last_activity DESC, created_at DESC
                       LIMIT ?
                     )""",
                (user_id, *_dev, user_id, *_dev, max_sess),
            )
        await self._db.commit()

        log.info(
            "session_created",
            user_id=user_id,
            source=source,
            source_device_id=source_device_id,
        )
        return raw

    async def validate_token(self, token: str) -> User | None:
        """Validate a session token. Returns User or None.

        ``token`` is the raw value the client presents (cookie / Bearer);
        we hash it before the DB lookup since only the hash is stored.
        In-process caches are also keyed by the hash, so a memory dump
        wouldn't surface unhashed tokens.
        """
        token_hash = _hash_token(token)

        # Check in-memory cache first (keyed by hash to keep raw tokens
        # out of process memory after the initial hashing).
        now = time.monotonic()
        cached = self._token_cache.get(token_hash)
        if cached:
            user, cached_at = cached
            if now - cached_at < self._cache_ttl:
                return user

        # DB lookup
        cursor = await self._db.execute(
            """SELECT u.id, u.username, u.display_name, u.role, u.is_active,
                      u.quota_bytes, u.created_at, u.updated_at, u.content_level,
                      COALESCE(s.source, 'web'), COALESCE(s.source_device_id, '')
               FROM auth_sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.token = ? AND s.expires_at > datetime('now')""",
            (token_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            self._token_cache.pop(token_hash, None)
            return None

        user = User(
            id=row[0], username=row[1], display_name=row[2], role=row[3],
            is_active=bool(row[4]), quota_bytes=row[5], created_at=row[6],
            updated_at=row[7],
            content_level=row[8] or "unrestricted",
            session_source=row[9] or "web",
        )

        if not user.is_active:
            return None

        session_source = row[9] or "web"
        source_device_id = row[10] or ""
        if session_source in {"android", "mobile"}:
            if not source_device_id:
                self._token_cache.pop(token_hash, None)
                log.info(
                    "session_rejected_for_unbound_mobile_device",
                    user_id=user.id,
                    source=session_source,
                )
                return None
            cursor = await self._db.execute(
                """SELECT revoked_at
                   FROM trusted_mobile_devices
                   WHERE user_id = ? AND device_id = ?
                   LIMIT 1""",
                (user.id, source_device_id),
            )
            device_row = await cursor.fetchone()
            if device_row is None or (device_row[0] or ""):
                self._token_cache.pop(token_hash, None)
                log.info(
                    "session_rejected_for_revoked_device",
                    user_id=user.id,
                    source=session_source,
                    source_device_id=source_device_id,
                )
                return None

        # Cache it (by hash, not raw)
        self._token_cache[token_hash] = (user, now)

        # last_activity is best-effort telemetry — nothing else in the
        # codebase reads it (verified by grep 2026-05-22). Throttle hard
        # AND commit explicitly. The previous "piggyback on next write"
        # pattern left the main backend conn holding the SQLite writer
        # RESERVED lock until some unrelated commit ran, blocking every
        # other connection's BEGIN IMMEDIATE for up to busy_timeout.
        # That was the root cause of resource_ledger_persist_failed
        # storms on 2026-05-22.
        last_write = self._last_activity_writes.get(token_hash, 0.0)
        if now - last_write >= self._last_activity_min_interval:
            try:
                await self._db.execute(
                    "UPDATE auth_sessions SET last_activity = datetime('now') WHERE token = ?",
                    (token_hash,),
                )
                await self._db.commit()
                self._last_activity_writes[token_hash] = now
            except Exception:
                # last_activity is non-critical; never fail validation
                # just because the telemetry write hit a transient lock.
                log.debug("validate_token_activity_bump_failed", exc_info=True)

        return user

    async def revoke_session(self, token: str) -> None:
        """Revoke a single session. ``token`` is the raw client-side value."""
        token_hash = _hash_token(token)
        await self._db.execute(
            "DELETE FROM auth_sessions WHERE token = ?",
            (token_hash,),
        )
        await self._db.commit()
        self._token_cache.pop(token_hash, None)
        self._last_activity_writes.pop(token_hash, None)

    async def revoke_all_sessions(self, user_id: str, *, except_token: str = "") -> None:
        """Revoke all sessions for a user, optionally keeping one.

        ``except_token`` is the raw client-side value; we hash it before
        comparing against the stored column.
        """
        if except_token:
            await self._db.execute(
                "DELETE FROM auth_sessions WHERE user_id = ? AND token != ?",
                (user_id, _hash_token(except_token)),
            )
        else:
            await self._db.execute(
                "DELETE FROM auth_sessions WHERE user_id = ?", (user_id,),
            )
        await self._db.commit()
        self._invalidate_user_cache(user_id)

    async def revoke_sessions_for_source_device(
        self,
        user_id: str,
        *,
        source: str,
        source_device_id: str,
    ) -> int:
        """Revoke all sessions for a paired device.

        Used when a trusted Android/mobile device is revoked. Returns the
        SQLite rowcount so route handlers can report how many sessions were
        invalidated.
        """
        source = _normalise_session_source(source)
        source_device_id = (source_device_id or "").strip()[:160]
        if not user_id or not source_device_id:
            return 0
        cursor = await self._db.execute(
            """DELETE FROM auth_sessions
               WHERE user_id = ? AND source = ? AND source_device_id = ?""",
            (user_id, source, source_device_id),
        )
        await self._db.commit()
        self._invalidate_user_cache(user_id)
        count = int(cursor.rowcount or 0)
        log.info(
            "sessions_revoked_for_source_device",
            user_id=user_id,
            source=source,
            source_device_id=source_device_id,
            count=count,
        )
        return count

    def _invalidate_user_cache(self, user_id: str) -> None:
        """Remove all cached tokens for a user. Also fires registered
        sibling-cache invalidators (e.g. ApiKeyManager) so a single
        ``update_user`` / ``delete_user`` clears every auth surface
        that holds a stale ``User`` reference."""
        to_remove = [t for t, (u, _) in self._token_cache.items() if u.id == user_id]
        for t in to_remove:
            self._token_cache.pop(t, None)
        for invalidator in self._user_cache_invalidators:
            try:
                invalidator(user_id)
            except Exception as exc:  # noqa: BLE001 — sibling failures must not break primary path
                log.warning(
                    "auth_cache_invalidator_failed",
                    user_id=user_id, error=str(exc),
                )

    def register_user_cache_invalidator(self, fn) -> None:
        """Register a ``callable(user_id) -> None`` that will be called
        whenever this manager invalidates its own per-user cache. Used
        by sibling auth caches (ApiKeyManager) to stay coherent with
        SessionManager's view of user state."""
        self._user_cache_invalidators.append(fn)

    # ------------------------------------------------------------------
    # Login rate limiting
    # ------------------------------------------------------------------

    async def check_lockout(self, username: str, ip_address: str) -> int | None:
        """Check if login is locked out. Returns retry-after seconds or None.

        Username is canonicalised (case-insensitive) so attackers can't
        rotate `Bob` / `bob` / `BOB` to multiply their attempt budget.
        """
        threshold = settings.auth_lockout_threshold
        minutes = settings.auth_lockout_minutes
        canon = _canonical_username(username)

        # SQL-side cutoff: SQLite's datetime('now') stores rows with a space
        # separator ("2026-04-23 18:31:36") but Python's isoformat() uses 'T'.
        # String-compare was silently dropping every same-day row, so the
        # previous code effectively disabled lockout entirely.
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM failed_login_attempts "
            "WHERE username = ? AND attempted_at > datetime('now', ?)",
            (canon, f"-{minutes} minutes"),
        )
        count = (await cursor.fetchone())[0]
        if count >= threshold:
            return minutes * 60

        ip_threshold = settings.auth_ip_lockout_threshold
        ip_minutes = settings.auth_ip_lockout_minutes
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM failed_login_attempts "
            "WHERE ip_address = ? AND attempted_at > datetime('now', ?)",
            (ip_address, f"-{ip_minutes} minutes"),
        )
        ip_count = (await cursor.fetchone())[0]
        if ip_count >= ip_threshold:
            return ip_minutes * 60

        return None

    async def record_failed_attempt(self, username: str, ip_address: str) -> None:
        """Record a failed login attempt under the canonical username.

        Wrapped in a short retry: this fires on the auth hot path and
        is best-effort hygiene rather than security-critical. A
        transient ``database is locked`` (lock storm during startup
        backup, busy-timeout overshoot) must not surface as a 500 to
        the user. If retries exhaust, we log and continue — missing
        one attempt row is OK; failing the login isn't.
        """
        async def _do() -> None:
            await self._db.execute(
                "INSERT INTO failed_login_attempts (username, ip_address) VALUES (?, ?)",
                (_canonical_username(username), ip_address),
            )
            await self._db.commit()
        await _retry_on_locked(_do, op="record_failed_attempt")

    async def clear_failed_attempts(self, username: str) -> None:
        """Clear failed attempts for a username (on successful login).

        Same retry treatment as ``record_failed_attempt`` — losing a
        row to a transient lock here just means the cleared username
        still carries stale attempt counts for a while, which is
        annoying but not security-critical. Failing the login is
        worse.
        """
        async def _do() -> None:
            await self._db.execute(
                "DELETE FROM failed_login_attempts WHERE username = ?",
                (_canonical_username(username),),
            )
            await self._db.commit()
        await _retry_on_locked(_do, op="clear_failed_attempts")

    async def cleanup_expired(self) -> int:
        """Delete expired sessions and old login attempts. Returns count deleted."""
        cursor = await self._db.execute(
            "DELETE FROM auth_sessions WHERE expires_at < datetime('now')",
        )
        sess_count = cursor.rowcount
        # Use SQLite's native datetime() arithmetic — same row format as the
        # default `datetime('now')` writes (space separator). The previous
        # Python isoformat() cutoff used a `T` separator, so within-the-day
        # rows were sometimes deleted incorrectly (T > space lexicographically).
        await self._db.execute(
            "DELETE FROM failed_login_attempts "
            "WHERE attempted_at < datetime('now', '-24 hours')",
        )
        await self._db.commit()

        # Prune cache
        now = time.monotonic()
        stale = [t for t, (_, ts) in self._token_cache.items() if now - ts > self._cache_ttl]
        for t in stale:
            self._token_cache.pop(t, None)

        if sess_count:
            log.info("auth_cleanup", expired_sessions=sess_count)
        return sess_count

    # ------------------------------------------------------------------
    # WebSocket tickets
    # ------------------------------------------------------------------

    def create_ws_ticket(self, user_id: str) -> str:
        """Create a short-lived one-time WebSocket ticket."""
        ticket = secrets.token_hex(16)
        ttl = settings.auth_ws_ticket_ttl_seconds
        self._ws_tickets[ticket] = WsTicket(
            ticket=ticket,
            user_id=user_id,
            expires_at=time.monotonic() + ttl,
        )
        # Prune expired tickets
        now = time.monotonic()
        expired = [k for k, v in self._ws_tickets.items() if v.expires_at < now]
        for k in expired:
            del self._ws_tickets[k]
        return ticket

    def validate_ws_ticket(self, ticket: str) -> str | None:
        """Validate and consume a WS ticket. Returns user_id or None."""
        ws = self._ws_tickets.pop(ticket, None)
        if not ws:
            return None
        if time.monotonic() > ws.expires_at:
            return None
        return ws.user_id

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def backfill_user_id(self, user_id: str) -> int:
        """Set user_id on all rows where user_id IS NULL. Returns rows updated."""
        tables = [
            "sessions", "ui_sessions", "ui_characters",
            "facts", "entities", "plot_threads", "contradictions",
            "lorebook_entries", "assumptions", "character_cards", "narrative_memory",
            "memories", "kg_nodes", "kg_edges", "memory_cooccurrence", "memory_events",
            "image_generations", "chat_images", "image_cache",
            "documents", "document_chunks", "session_documents",
            "artifacts", "custom_flows", "reasoning_flows", "coder_sessions",
        ]
        total = 0
        for table in tables:
            try:
                cursor = await self._db.execute(
                    f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL",
                    (user_id,),
                )
                total += cursor.rowcount
            except Exception as exc:
                # Table may not exist yet on a fresh DB; backfill is
                # idempotent and runs on every startup.
                log.debug("backfill_table_skipped", table=table, error=str(exc))
        await self._db.commit()
        if total:
            log.info("backfill_complete", user_id=user_id, rows_updated=total)
        return total
