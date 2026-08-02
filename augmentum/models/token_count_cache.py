"""Token count cache - SQLite-backed cache for tokenizer results.

Maps (model_id, text) to token_count and the full token array. The
production consumer is the full-prompt cache used by
``LlamaCppBackend._build_token_prompt`` to skip re-tokenization on
repeat sends of the same rendered prompt.
"""
from __future__ import annotations

import asyncio
import hashlib
import struct
import time
from pathlib import Path
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _pack_tokens(tokens: list[int]) -> bytes:
    """Pack a list of token IDs into a compact binary blob (int32 LE)."""
    return struct.pack(f"<{len(tokens)}i", *tokens)


def _unpack_tokens(blob: bytes) -> list[int]:
    """Unpack a binary blob back into a list of token IDs."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}i", blob))


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS token_counts (
    id          TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    tokens      BLOB,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL,
    use_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tc_model ON token_counts(model_id);
CREATE INDEX IF NOT EXISTS idx_tc_last_used ON token_counts(last_used);
"""

# Upgrade existing tables that lack the tokens column
_UPGRADE_SQL = """
ALTER TABLE token_counts ADD COLUMN tokens BLOB;
"""

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
)

_RECOVERABLE_ERRORS = (
    "disk i/o error",
    "database disk image is malformed",
    "file is not a database",
    "unable to open database file",
    "database or disk is full",
)


class TokenCountCache:
    """Async SQLite cache for token counts."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._healthy = True  # set to False on persistent errors to avoid spam
        self._recover_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(model_id: str, text: str) -> str:
        """Cache key: sha256(model_id:text)[:16]."""
        return hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest()[:16]

    @staticmethod
    def _text_hash(text: str) -> str:
        """Content hash: sha256(text)[:16]."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def _open_db(self) -> aiosqlite.Connection:
        """Open a cache connection with basic safety pragmas applied."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self._db_path)
        for pragma in _PRAGMAS:
            await db.execute(pragma)
        return db

    async def _close_quietly(self, db: aiosqlite.Connection | None) -> None:
        """Best-effort connection close for recovery paths."""
        if db is None:
            return
        try:
            await db.close()
        except aiosqlite.Error as exc:
            log.debug("token_cache_close_failed", error=str(exc))

    @staticmethod
    def _is_recoverable_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return any(marker in err for marker in _RECOVERABLE_ERRORS)

    async def _init_db_once(self) -> None:
        """Create/upgrade schema without retry or recovery wrapper."""
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            await db.executescript(_CREATE_SQL)
            try:
                await db.execute(_UPGRADE_SQL)
            except aiosqlite.OperationalError:
                # Column already exists — UPGRADE is idempotent by intent.
                pass
            await db.commit()
        finally:
            await self._close_quietly(db)

    async def _reset_cache_db(self, trigger: str, exc: Exception) -> bool:
        """Rebuild the cache DB after a disposable-cache failure."""
        async with self._recover_lock:
            base = Path(self._db_path)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = ""

            try:
                if base.exists():
                    backup = base.with_name(f"{base.stem}_corrupt_{stamp}{base.suffix}")
                    base.replace(backup)
                    backup_path = str(backup)
            except OSError as rename_exc:
                log.debug("token_cache_backup_rename_failed", error=str(rename_exc))
                try:
                    base.unlink(missing_ok=True)
                except OSError:
                    pass

            for suffix in ("-wal", "-shm"):
                try:
                    Path(f"{self._db_path}{suffix}").unlink(missing_ok=True)
                except OSError:
                    pass

            try:
                await self._init_db_once()
                self._healthy = True
                log.warning(
                    "token_cache_reset",
                    trigger=trigger,
                    backup=backup_path or None,
                    error=str(exc)[:200],
                )
                return True
            except Exception as reset_exc:
                self._healthy = False
                log.warning(
                    "token_cache_reset_failed",
                    trigger=trigger,
                    original_error=str(exc)[:200],
                    error=str(reset_exc)[:200],
                )
                return False

    async def _handle_error(self, event: str, exc: Exception) -> None:
        """Recover disposable cache failures when possible, else disable cache."""
        if self._is_recoverable_error(exc):
            recovered = await self._reset_cache_db(event, exc)
            if recovered:
                return
        log.warning(event, error=str(exc)[:200])
        self._healthy = False

    # ------------------------------------------------------------------
    # DB lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create the token_counts table (standalone / test use)."""
        try:
            await self._init_db_once()
            self._healthy = True
        except Exception as exc:
            if self._is_recoverable_error(exc):
                recovered = await self._reset_cache_db("token_cache_init_failed", exc)
                if recovered:
                    return
            log.warning("token_cache_init_failed", error=str(exc)[:200])
            self._healthy = False

    async def _execute(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        """Low-level execute, returns all rows."""
        if not self._healthy:
            return []
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            await db.commit()
            return rows
        except Exception as exc:
            await self._handle_error("token_cache_execute_error", exc)
            return []
        finally:
            await self._close_quietly(db)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get_count(self, model_id: str, text: str) -> int | None:
        """Look up a cached count. Updates last_used and use_count on hit."""
        if not self._healthy:
            return None
        key = self._key(model_id, text)
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute(
                "SELECT token_count FROM token_counts WHERE id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            await db.execute(
                "UPDATE token_counts SET last_used = ?, use_count = use_count + 1 "
                "WHERE id = ?",
                (time.time(), key),
            )
            await db.commit()
            return row[0]
        except Exception as exc:
            await self._handle_error("token_cache_get_count_error", exc)
            return None
        finally:
            await self._close_quietly(db)

    async def store_count(self, model_id: str, text: str, count: int) -> None:
        """Upsert a token count."""
        if not self._healthy:
            return
        key = self._key(model_id, text)
        now = time.time()
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            await db.execute(
                "INSERT INTO token_counts (id, model_id, source_hash, token_count, "
                "created_at, last_used, use_count) VALUES (?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET "
                "token_count = excluded.token_count, last_used = excluded.last_used",
                (key, model_id, self._text_hash(text), count, now, now),
            )
            await db.commit()
        except Exception as exc:
            await self._handle_error("token_cache_store_count_error", exc)
        finally:
            await self._close_quietly(db)

    async def get_or_fetch(
        self, model_id: str, text: str, backend_url: str
    ) -> int:
        """Return cached count or call llama-server /tokenize endpoint."""
        cached = await self.get_count(model_id, text)
        if cached is not None:
            return cached

        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{backend_url.rstrip('/')}/tokenize"
            resp = await client.post(url, json={"content": text})
            resp.raise_for_status()
            tokens = resp.json().get("tokens", [])
            count = len(tokens)

        await self.store_count(model_id, text, count)
        return count

    # ------------------------------------------------------------------
    # Token array storage (for pre-tokenized prompt assembly)
    # ------------------------------------------------------------------

    async def get_tokens(self, model_id: str, text: str) -> list[int] | None:
        """Return cached token array, or None on miss."""
        if not self._healthy:
            return None
        key = self._key(model_id, text)
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute(
                "SELECT tokens FROM token_counts WHERE id = ? AND tokens IS NOT NULL",
                (key,),
            )
            row = await cursor.fetchone()
            if row is None or row[0] is None:
                return None
            await db.execute(
                "UPDATE token_counts SET last_used = ?, use_count = use_count + 1 "
                "WHERE id = ?",
                (time.time(), key),
            )
            await db.commit()
            return _unpack_tokens(row[0])
        except Exception as exc:
            await self._handle_error("token_cache_get_error", exc)
            return None
        finally:
            await self._close_quietly(db)

    async def store_tokens(
        self, model_id: str, text: str, tokens: list[int]
    ) -> None:
        """Store a token array (and its count)."""
        if not self._healthy:
            return
        key = self._key(model_id, text)
        now = time.time()
        blob = _pack_tokens(tokens)
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            await db.execute(
                "INSERT INTO token_counts "
                "(id, model_id, source_hash, token_count, tokens, created_at, last_used, use_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET "
                "token_count = excluded.token_count, tokens = excluded.tokens, "
                "last_used = excluded.last_used",
                (key, model_id, self._text_hash(text), len(tokens), blob, now, now),
            )
            await db.commit()
        except Exception as exc:
            await self._handle_error("token_cache_store_error", exc)
        finally:
            await self._close_quietly(db)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def evict_stale(
        self, max_age_days: int = 30, min_uses: int = 3
    ) -> int:
        """Delete old, rarely-used entries. Returns count deleted."""
        if not self._healthy:
            return 0
        cutoff = time.time() - max_age_days * 86400
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute(
                "DELETE FROM token_counts WHERE last_used < ? AND use_count < ?",
                (cutoff, min_uses),
            )
            await db.commit()
            return cursor.rowcount
        except Exception as exc:
            await self._handle_error("token_cache_evict_error", exc)
            return 0
        finally:
            await self._close_quietly(db)

    async def purge_model(self, model_id: str) -> int:
        """Delete all counts for a model. Returns count deleted."""
        if not self._healthy:
            return 0
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute(
                "DELETE FROM token_counts WHERE model_id = ?", (model_id,)
            )
            await db.commit()
            return cursor.rowcount
        except Exception as exc:
            await self._handle_error("token_cache_purge_error", exc)
            return 0
        finally:
            await self._close_quietly(db)

    async def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        if not self._healthy:
            return {"total_entries": 0, "distinct_models": 0}
        db: aiosqlite.Connection | None = None
        try:
            db = await self._open_db()
            cursor = await db.execute("SELECT COUNT(*) FROM token_counts")
            total = (await cursor.fetchone())[0]
            cursor = await db.execute(
                "SELECT COUNT(DISTINCT model_id) FROM token_counts"
            )
            models = (await cursor.fetchone())[0]
            return {"total_entries": total, "distinct_models": models}
        except Exception as exc:
            await self._handle_error("token_cache_stats_error", exc)
            return {"total_entries": 0, "distinct_models": 0}
        finally:
            await self._close_quietly(db)
