"""Persistent metadata for llama.cpp KV slot saves.

The slot files themselves live under llama-server's ``--slot-save-path``.
This manifest stores the Augmentum-side metadata we need to decide whether a
restore is safe, when a session should expire, and which saves are pinned.

The implementation uses ``sqlite3`` (sync) so the slot-eviction code in
``llama_server_manager`` — which runs in sync ``build_args`` before the
event loop is involved — can use the manifest directly. For async hot
paths (every chat turn calls record_save / get_session etc.), use the
``*_async`` wrappers which off-load to a thread so they never block the
loop.

Self-recovery: a corrupted manifest file (disk error, ungraceful shutdown
mid-write, antivirus tampering) used to disable warm-resume across all
conversations until a manual delete. Now follows the same backup-and-
rebuild pattern as ``TokenCountCache``: a recoverable sqlite error
triggers a one-shot reset that renames the corrupted DB with a
timestamped suffix and re-initializes the schema. Worst case we lose
the warm-resume manifest for whatever sessions were saved up to that
point — slot files on disk are untouched, so the next save repopulates.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# sqlite errors we treat as "the file is broken; rebuild from scratch."
# Mirrors TokenCountCache's _RECOVERABLE_ERRORS so a single corruption
# event recovers identically across both stores.
_RECOVERABLE_ERRORS: tuple[str, ...] = (
    "disk i/o error",
    "database disk image is malformed",
    "file is not a database",
    "unable to open database file",
    "database or disk is full",
)


class KVSessionManifest:
    """Small SQLite registry for persisted KV sessions."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path))
        self._lock = threading.Lock()
        # Set False after a recovery attempt fails; suppresses repeated
        # logging on subsequent calls. Reset to True on successful
        # recovery.
        self._healthy = True
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # busy_timeout=30000 (30s) covers heavy contention windows
        # without making non-OOM bugs hang the UI for minutes.
        # Matches the canonical augmentum.db setting in
        # state/backends/sqlite.py::AUGMENTUM_DB_PRAGMAS so behavior
        # is consistent across stores.
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # Columns added after initial schema. Each is applied via best-effort
    # ALTER TABLE; sqlite raises "duplicate column" on re-run which we swallow.
    _MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
        ("flash_attn", "INTEGER NOT NULL DEFAULT 0"),
        ("gpu_layers", "INTEGER NOT NULL DEFAULT 0"),
        ("gpu_layers_mode", "TEXT NOT NULL DEFAULT ''"),
        ("batch_size", "INTEGER NOT NULL DEFAULT 0"),
        ("draft_model", "TEXT NOT NULL DEFAULT ''"),
        ("draft_max", "INTEGER NOT NULL DEFAULT 0"),
        # Architectural fingerprints (llama.cpp Discussion #15569 must-match
        # list). Catch the case where ``model_id`` + ``model_mtime`` coincide
        # across different models — rare but real (e.g. user overwrites a
        # GGUF in place with a same-mtime different model). All three values
        # come free from ``ModelProfile`` which is already parsed at load
        # time, so adding them costs only schema width + one comparison.
        ("n_embed", "INTEGER NOT NULL DEFAULT 0"),
        ("n_layers_total", "INTEGER NOT NULL DEFAULT 0"),
        ("n_heads_kv", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS kv_sessions (
                        model_key TEXT NOT NULL,
                        session_key TEXT NOT NULL,
                        mode TEXT NOT NULL DEFAULT '',
                        slot_dir TEXT NOT NULL DEFAULT '',
                        slot_filename TEXT NOT NULL DEFAULT '',
                        model_id TEXT NOT NULL DEFAULT '',
                        model_path TEXT NOT NULL DEFAULT '',
                        model_mtime REAL NOT NULL DEFAULT 0,
                        ctx_size INTEGER NOT NULL DEFAULT 0,
                        kv_cache_type TEXT NOT NULL DEFAULT '',
                        template_fingerprint TEXT NOT NULL DEFAULT '',
                        system_prompt_hash TEXT NOT NULL DEFAULT '',
                        prompt_fingerprint TEXT NOT NULL DEFAULT '',
                        prompt_message_count INTEGER NOT NULL DEFAULT 0,
                        last_accessed REAL NOT NULL DEFAULT 0,
                        last_saved REAL NOT NULL DEFAULT 0,
                        expires_at REAL NOT NULL DEFAULT 0,
                        pinned INTEGER NOT NULL DEFAULT 0,
                        last_restore_result TEXT NOT NULL DEFAULT '',
                        last_skip_reason TEXT NOT NULL DEFAULT '',
                        flash_attn INTEGER NOT NULL DEFAULT 0,
                        gpu_layers INTEGER NOT NULL DEFAULT 0,
                        gpu_layers_mode TEXT NOT NULL DEFAULT '',
                        batch_size INTEGER NOT NULL DEFAULT 0,
                        draft_model TEXT NOT NULL DEFAULT '',
                        draft_max INTEGER NOT NULL DEFAULT 0,
                        n_embed INTEGER NOT NULL DEFAULT 0,
                        n_layers_total INTEGER NOT NULL DEFAULT 0,
                        n_heads_kv INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (model_key, session_key)
                    )
                    """
            )
            # Migrate older DBs that predate the load-shape columns.
            for col_name, col_decl in self._MIGRATION_COLUMNS:
                try:
                    conn.execute(
                        f"ALTER TABLE kv_sessions ADD COLUMN {col_name} {col_decl}"
                    )
                except sqlite3.OperationalError:
                    # Column already exists (post-CREATE or already migrated).
                    pass
            conn.execute(
                """
                    CREATE INDEX IF NOT EXISTS idx_kv_sessions_slot_dir
                    ON kv_sessions(slot_dir)
                    """
            )
            conn.execute(
                """
                    CREATE INDEX IF NOT EXISTS idx_kv_sessions_expires_at
                    ON kv_sessions(expires_at)
                    """
            )
            # Replay sources — the durable half of the KV resume
            # ladder's rung 2. Stores the exact message prefix a
            # session's last engine request actually served, so a
            # restart (or a --kv-unified config where slot-file
            # restore is structurally unavailable) can recompute the
            # KV via a background prewarm instead of making the user
            # pay full prefill at the keyboard. Keyed by session_key
            # alone — deliberately NOT model-scoped: replay re-renders
            # through whatever model is live, so a model swap keeps
            # every warm candidate (tensor restore can't do that).
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS kv_replay_sources (
                        session_key TEXT PRIMARY KEY,
                        mode TEXT NOT NULL DEFAULT '',
                        messages_json TEXT NOT NULL,
                        fingerprint TEXT NOT NULL DEFAULT '',
                        message_count INTEGER NOT NULL DEFAULT 0,
                        approx_chars INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL DEFAULT 0,
                        expires_at REAL NOT NULL DEFAULT 0,
                        sampling_json TEXT NOT NULL DEFAULT ''
                    )
                    """
            )
            conn.execute(
                """
                    CREATE INDEX IF NOT EXISTS idx_kv_replay_updated_at
                    ON kv_replay_sources(updated_at)
                    """
            )
            # Additive upgrade for DBs created before sampling capture
            # (the speculation rung needs the previous turn's sampling
            # to fingerprint-match the next real request).
            try:
                conn.execute(
                    "ALTER TABLE kv_replay_sources "
                    "ADD COLUMN sampling_json TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # column already exists (fresh CREATE or prior ALTER)

    # ------------------------------------------------------------------
    # Self-recovery (mirrors TokenCountCache pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_recoverable_error(exc: BaseException) -> bool:
        """Heuristic: is this sqlite error one we can fix by rebuilding?

        Matches the sqlite messages that indicate corruption / I/O
        failure / disk-full — anything where re-initializing the file
        from scratch is a valid recovery. Programming errors (wrong
        SQL, schema mismatch from a real bug) don't match these
        strings and propagate normally.
        """
        if not isinstance(exc, sqlite3.Error):
            return False
        return any(marker in str(exc).lower() for marker in _RECOVERABLE_ERRORS)

    def _reset_db(self, trigger: str, exc: BaseException) -> bool:
        """Back up the corrupted DB and reinit a fresh one.

        Renames the corrupted file with a ``_corrupt_YYYYMMDD_HHMMSS``
        suffix so a forensic look is still possible, then recreates
        the schema. WAL/SHM sidecars are removed too — keeping them
        across a rebuild would re-corrupt the new file.

        Windows file-handle nuance: any sqlite Connection from a prior
        ``_connect()`` call may still hold the file handle past Python
        scope-exit (the GC hasn't run yet). On Linux the kernel-level
        unlink-while-open is fine, but Windows blocks the rename with
        a sharing violation. Force a ``gc.collect()`` before rename so
        outstanding Connection objects close their handles.

        Returns True on successful rebuild, False if even the rebuild
        fails (e.g. disk truly full / read-only) — caller marks
        ``_healthy = False`` so subsequent attempts no-op silently.
        """
        # Drop any lingering sqlite Connection objects so Windows
        # releases the file handle before we try to rename. Cheap; runs
        # only on the recovery path.
        gc.collect()

        base = Path(self._db_path)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = ""
        rename_error: Exception | None = None

        try:
            if base.exists():
                backup = base.with_name(
                    f"{base.stem}_corrupt_{stamp}{base.suffix}"
                )
                base.replace(backup)
                backup_path = str(backup)
        except Exception as exc_rename:
            rename_error = exc_rename
            # Couldn't rename — try to delete instead so init starts
            # from scratch. If THAT also fails the file is genuinely
            # locked; we'll fall through and let _init_db raise.
            try:
                base.unlink(missing_ok=True)
            except Exception as unlink_exc:
                log.debug("kv_manifest_corrupt_unlink_failed: path=%s error=%s",
                          str(base), str(unlink_exc)[:120])

        for suffix in ("-wal", "-shm"):
            try:
                Path(f"{self._db_path}{suffix}").unlink(missing_ok=True)
            except Exception as side_exc:
                log.debug("kv_manifest_sidecar_unlink_failed: suffix=%s error=%s",
                          suffix, str(side_exc)[:120])

        try:
            self._init_db()
            self._healthy = True
            log.warning(
                "kv_manifest_reset trigger=%s backup=%s error=%s rename_failure=%s",
                trigger,
                backup_path or "<none>",
                str(exc)[:200],
                str(rename_error)[:120] if rename_error else "<none>",
            )
            return True
        except Exception as reset_exc:
            self._healthy = False
            log.warning(
                "kv_manifest_reset_failed trigger=%s original=%s error=%s",
                trigger, str(exc)[:200], str(reset_exc)[:200],
            )
            return False

    def _handle_error(self, event: str, exc: BaseException) -> None:
        """Recover when possible, otherwise mark unhealthy and log.

        Called from every public method's exception handler. After this
        returns, the calling method returns its safe-default (None for
        get*, [] for list*, no-op for mutators) so callers don't see
        the underlying sqlite exception.
        """
        if self._is_recoverable_error(exc):
            if self._reset_db(event, exc):
                return
        if self._healthy:
            # Suppress duplicate logging on repeat failures.
            log.warning("%s error=%s", event, str(exc)[:200])
        self._healthy = False

    @staticmethod
    def _expiry_from_ttl(ttl_days: float, now: float) -> float:
        if ttl_days <= 0:
            return 0.0
        return now + float(ttl_days) * 86400.0

    def get_session(self, model_key: str, session_key: str) -> dict[str, Any] | None:
        if not self._healthy:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    """
                        SELECT * FROM kv_sessions
                        WHERE model_key = ? AND session_key = ?
                        """,
                    (model_key, session_key),
                ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_get_session_error", exc)
            return None

    def list_sessions(self, slot_dir: str) -> list[dict[str, Any]]:
        if not self._healthy:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    """
                        SELECT * FROM kv_sessions
                        WHERE slot_dir = ?
                        ORDER BY last_accessed DESC, last_saved DESC
                        """,
                    (slot_dir,),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_list_sessions_error", exc)
            return []

    def list_model_sessions(self, model_key: str) -> list[dict[str, Any]]:
        """Sessions for ``model_key`` ordered most-recently-used first.

        Used by the restart-warm path to find the best candidate to
        hydrate slot 0 with after the model becomes ready.
        """
        if not self._healthy:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    """
                        SELECT * FROM kv_sessions
                        WHERE model_key = ?
                        ORDER BY last_accessed DESC, last_saved DESC
                        """,
                    (model_key,),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_list_model_sessions_error", exc)
            return []

    def list_expired_sessions(
        self,
        slot_dir: str,
        *,
        now: float | None = None,
        pinned_sessions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._healthy:
            return []
        now_ts = time.time() if now is None else now
        pinned = pinned_sessions or set()
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    """
                        SELECT * FROM kv_sessions
                        WHERE slot_dir = ?
                          AND expires_at > 0
                          AND expires_at <= ?
                        """,
                    (slot_dir, now_ts),
                ).fetchall()
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_list_expired_error", exc)
            return []
        expired: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            if record["session_key"] in pinned or bool(record["pinned"]):
                continue
            expired.append(record)
        return expired

    def record_save(
        self,
        *,
        model_key: str,
        session_key: str,
        mode: str,
        slot_dir: str,
        slot_filename: str,
        model_id: str,
        model_path: str,
        model_mtime: float,
        ctx_size: int,
        kv_cache_type: str,
        template_fingerprint: str,
        system_prompt_hash: str,
        prompt_fingerprint: str,
        prompt_message_count: int,
        ttl_days: float,
        pinned: bool,
        flash_attn: bool = False,
        gpu_layers: int = 0,
        gpu_layers_mode: str = "",
        batch_size: int = 0,
        draft_model: str = "",
        draft_max: int = 0,
        n_embed: int = 0,
        n_layers_total: int = 0,
        n_heads_kv: int = 0,
    ) -> None:
        if not self._healthy:
            return
        now = time.time()
        expires_at = self._expiry_from_ttl(ttl_days, now)
        params = (
            model_key,
            session_key,
            mode,
            slot_dir,
            slot_filename,
            model_id,
            model_path,
            float(model_mtime or 0.0),
            int(ctx_size or 0),
            kv_cache_type or "",
            template_fingerprint or "",
            system_prompt_hash or "",
            prompt_fingerprint or "",
            int(prompt_message_count or 0),
            now,
            now,
            expires_at,
            1 if pinned else 0,
            "saved",
            1 if flash_attn else 0,
            int(gpu_layers or 0),
            gpu_layers_mode or "",
            int(batch_size or 0),
            draft_model or "",
            int(draft_max or 0),
            int(n_embed or 0),
            int(n_layers_total or 0),
            int(n_heads_kv or 0),
        )
        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO kv_sessions (
                            model_key, session_key, mode, slot_dir, slot_filename,
                            model_id, model_path, model_mtime, ctx_size, kv_cache_type,
                            template_fingerprint, system_prompt_hash, prompt_fingerprint,
                            prompt_message_count, last_accessed, last_saved, expires_at,
                            pinned, last_restore_result, last_skip_reason,
                            flash_attn, gpu_layers, gpu_layers_mode, batch_size,
                            draft_model, draft_max,
                            n_embed, n_layers_total, n_heads_kv
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(model_key, session_key) DO UPDATE SET
                            mode = excluded.mode,
                            slot_dir = excluded.slot_dir,
                            slot_filename = excluded.slot_filename,
                            model_id = excluded.model_id,
                            model_path = excluded.model_path,
                            model_mtime = excluded.model_mtime,
                            ctx_size = excluded.ctx_size,
                            kv_cache_type = excluded.kv_cache_type,
                            template_fingerprint = excluded.template_fingerprint,
                            system_prompt_hash = excluded.system_prompt_hash,
                            prompt_fingerprint = excluded.prompt_fingerprint,
                            prompt_message_count = excluded.prompt_message_count,
                            last_accessed = excluded.last_accessed,
                            last_saved = excluded.last_saved,
                            expires_at = excluded.expires_at,
                            pinned = excluded.pinned,
                            last_restore_result = 'saved',
                            last_skip_reason = '',
                            flash_attn = excluded.flash_attn,
                            gpu_layers = excluded.gpu_layers,
                            gpu_layers_mode = excluded.gpu_layers_mode,
                            batch_size = excluded.batch_size,
                            draft_model = excluded.draft_model,
                            draft_max = excluded.draft_max,
                            n_embed = excluded.n_embed,
                            n_layers_total = excluded.n_layers_total,
                            n_heads_kv = excluded.n_heads_kv
                        """,
                        params,
                    )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_record_save_error", exc)

    def touch_session(
        self,
        *,
        model_key: str,
        session_key: str,
        ttl_days: float,
        mode: str = "",
        pinned: bool | None = None,
        restored: bool | None = None,
    ) -> None:
        if not self._healthy:
            return
        now = time.time()
        expires_at = self._expiry_from_ttl(ttl_days, now)
        sets = ["last_accessed = ?", "expires_at = ?"]
        params: list[Any] = [now, expires_at]
        if mode:
            sets.append("mode = ?")
            params.append(mode)
        if pinned is not None:
            sets.append("pinned = ?")
            params.append(1 if pinned else 0)
        if restored is not None:
            sets.append("last_restore_result = ?")
            params.append("restored" if restored else "miss")
            sets.append("last_skip_reason = ''")
        params.extend([model_key, session_key])

        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    f"""
                        UPDATE kv_sessions
                        SET {", ".join(sets)}
                        WHERE model_key = ? AND session_key = ?
                        """,
                    params,
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_touch_session_error", exc)

    def mark_restore_skip(self, model_key: str, session_key: str, reason: str) -> None:
        if not self._healthy:
            return
        now = time.time()
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                        UPDATE kv_sessions
                        SET last_accessed = ?,
                            last_restore_result = 'skipped',
                            last_skip_reason = ?
                        WHERE model_key = ? AND session_key = ?
                        """,
                    (now, reason, model_key, session_key),
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_mark_skip_error", exc)

    def set_pinned(self, session_key: str, pinned: bool) -> None:
        if not self._healthy:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE kv_sessions SET pinned = ? WHERE session_key = ?",
                    (1 if pinned else 0, session_key),
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_set_pinned_error", exc)

    def delete_session(self, model_key: str, session_key: str) -> None:
        if not self._healthy:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                        DELETE FROM kv_sessions
                        WHERE model_key = ? AND session_key = ?
                        """,
                    (model_key, session_key),
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_delete_error", exc)

    # ------------------------------------------------------------------
    # Replay sources (KV resume ladder — rung 2)
    # ------------------------------------------------------------------

    def record_replay_source(
        self,
        *,
        session_key: str,
        mode: str,
        messages_json: str,
        fingerprint: str,
        message_count: int,
        ttl_days: float,
        sampling_json: str = "",
    ) -> None:
        """Upsert the replayable prefix for ``session_key``.

        ``messages_json`` is the serialized ``[{role, content}, ...]``
        list exactly as the engine served it (post-augmentation), so a
        later replay re-renders byte-identically through apply-template.
        ``sampling_json`` snapshots the request's completion-shaping
        fields for the speculation rung's fingerprint match.
        """
        if not self._healthy or not session_key:
            return
        now = time.time()
        expires_at = self._expiry_from_ttl(ttl_days, now)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                        INSERT INTO kv_replay_sources (
                            session_key, mode, messages_json, fingerprint,
                            message_count, approx_chars, updated_at, expires_at,
                            sampling_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_key) DO UPDATE SET
                            mode = excluded.mode,
                            messages_json = excluded.messages_json,
                            fingerprint = excluded.fingerprint,
                            message_count = excluded.message_count,
                            approx_chars = excluded.approx_chars,
                            updated_at = excluded.updated_at,
                            expires_at = excluded.expires_at,
                            sampling_json = excluded.sampling_json
                        """,
                    (
                        session_key,
                        mode or "",
                        messages_json,
                        fingerprint or "",
                        int(message_count or 0),
                        len(messages_json),
                        now,
                        expires_at,
                        sampling_json or "",
                    ),
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_record_replay_error", exc)

    def get_replay_source(self, session_key: str) -> dict[str, Any] | None:
        """Return the replay row for ``session_key``, or None.

        Expired rows are returned too — the caller decides whether
        expiry matters (an on-open resume for a session the user just
        deliberately reopened is still worth replaying; the boot-warm
        loop skips expired rows).
        """
        if not self._healthy or not session_key:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM kv_replay_sources WHERE session_key = ?",
                    (session_key,),
                ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_get_replay_error", exc)
            return None

    def list_replay_sources(self, *, limit: int = 32) -> list[dict[str, Any]]:
        """MRU-ordered replay rows (metadata only — no messages_json).

        The boot-warm loop walks this to find replay candidates; loading
        every row's full prefix up front would drag megabytes through
        sqlite for candidates that may be skipped, so the JSON is
        fetched per-session via :meth:`get_replay_source`.
        """
        if not self._healthy:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    """
                        SELECT session_key, mode, fingerprint, message_count,
                               approx_chars, updated_at, expires_at
                        FROM kv_replay_sources
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                    (max(1, int(limit)),),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_list_replay_error", exc)
            return []

    def delete_replay_source(self, session_key: str) -> None:
        if not self._healthy or not session_key:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "DELETE FROM kv_replay_sources WHERE session_key = ?",
                    (session_key,),
                )
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_delete_replay_error", exc)

    def prune_replay_sources(
        self, *, max_rows: int, now: float | None = None,
    ) -> tuple[int, int]:
        """Drop expired rows, then evict oldest beyond ``max_rows``.

        Returns ``(expired_count, evicted_count)`` so the caller can log
        what was dropped — never a silent cap.
        """
        if not self._healthy:
            return (0, 0)
        now_ts = time.time() if now is None else now
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    """
                        DELETE FROM kv_replay_sources
                        WHERE expires_at > 0 AND expires_at <= ?
                        """,
                    (now_ts,),
                )
                expired = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                evicted = 0
                if max_rows > 0:
                    cur = conn.execute(
                        """
                            DELETE FROM kv_replay_sources
                            WHERE session_key NOT IN (
                                SELECT session_key FROM kv_replay_sources
                                ORDER BY updated_at DESC
                                LIMIT ?
                            )
                            """,
                        (int(max_rows),),
                    )
                    evicted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            return (expired, evicted)
        except sqlite3.Error as exc:
            self._handle_error("kv_manifest_prune_replay_error", exc)
            return (0, 0)

    # ------------------------------------------------------------------
    # Async wrappers for hot-path callers
    # ------------------------------------------------------------------
    #
    # Sync sqlite under WAL is fast (sub-millisecond writes) but it
    # still blocks the event loop. The four methods below are called
    # from llama_cpp.py's per-request flow; off-loading to a thread
    # keeps the loop responsive under concurrent traffic. Sync callers
    # (slot eviction, pin/unpin) keep using the original methods.

    async def get_session_async(
        self, model_key: str, session_key: str,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.get_session, model_key, session_key)

    async def record_save_async(self, **kwargs: Any) -> None:
        # Bind kwargs into a thunk because to_thread doesn't forward kwargs.
        await asyncio.to_thread(lambda: self.record_save(**kwargs))

    async def touch_session_async(self, **kwargs: Any) -> None:
        await asyncio.to_thread(lambda: self.touch_session(**kwargs))

    async def mark_restore_skip_async(
        self, model_key: str, session_key: str, reason: str,
    ) -> None:
        await asyncio.to_thread(
            self.mark_restore_skip, model_key, session_key, reason,
        )

    async def record_replay_source_async(self, **kwargs: Any) -> None:
        await asyncio.to_thread(lambda: self.record_replay_source(**kwargs))

    async def get_replay_source_async(
        self, session_key: str,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.get_replay_source, session_key)

    async def list_replay_sources_async(
        self, *, limit: int = 32,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(lambda: self.list_replay_sources(limit=limit))

    async def prune_replay_sources_async(
        self, *, max_rows: int,
    ) -> tuple[int, int]:
        return await asyncio.to_thread(
            lambda: self.prune_replay_sources(max_rows=max_rows)
        )
