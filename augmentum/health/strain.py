"""StrainMonitor — periodic, durable server-strain sampler.

Complements the ``event_loop_stall`` watchdog (one log line per stall) with a
queryable time series. Each sample captures event-loop lag, in-flight request
count, shared single-slot resource state (engine, DB writer-lock latency,
GPU/RAM), per-mode session counts, and — the point of the whole thing — how
many distinct clients (browsers/tabs/devices) and users were concurrently
active. That last part lets multi-browser contention be hunted after the fact:
"at 17:42 lag jumped to 2.3s with 3 clients active, coder + voice + a third
browser polling."

Design constraints honoured:
  - Reads only cheap in-memory ``app.state`` (no Docker / nvidia-smi calls);
    GPU/RAM come from the ResourceLedger's *cached* last snapshot.
  - psutil reads run in a thread (``async_blocking`` clean).
  - The insert uses ``BEGIN IMMEDIATE`` and times itself, so the sampler's own
    writer-lock acquisition doubles as a DB-contention probe.
  - Server-level table (no user_id), mirroring resource_snapshots.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite
    from fastapi import FastAPI

log = get_logger(__name__)

# A client (tab) is "active" if its last request landed within this window.
_CLIENT_FRESH_S = 30.0
# Prune at most this often; keep this many days of samples.
_PRUNE_INTERVAL_S = 3600.0
_RETENTION_DAYS = 7

# Threshold trips → a grep-able ``strain_sample`` WARNING (like event_loop_stall).
_LAG_WARN_MS = 1000.0
_DB_WRITE_WARN_MS = 400.0
_INFLIGHT_WARN = 20

_INSERT_SQL = """
INSERT INTO strain_samples (
    timestamp, event_loop_lag_ms, inflight_requests, slow_requests,
    active_clients, active_users, ws_presence, ws_notify,
    sessions_narrative, sessions_agentic, sessions_coder,
    engine_model, engine_secondary, db_write_ms,
    gpu_used_mb, gpu_free_mb, ram_used_mb, ram_free_mb, proc_rss_mb,
    context_json
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def _safe_len(obj: Any) -> int:
    try:
        return len(obj)
    except Exception:
        return 0


def _proc_and_ram_mb() -> tuple[int, int, int]:
    """(proc_rss_mb, ram_used_mb, ram_free_mb) via psutil. Runs in a thread."""
    try:
        import psutil
    except Exception:
        return (0, 0, 0)
    rss = used = free = 0
    try:
        rss = int(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    try:
        # Container-aware: strain must reflect OUR ceiling, not the host's.
        # Reporting a 94 GiB WSL VM's headroom while the container is
        # seconds from its own limit is how strain stayed green through
        # the 2026-07-25 exhaustion.
        from augmentum.resource import hostmem

        info = hostmem.memory_info()
        used = info.used_mib
        free = info.available_mib
    except Exception:
        pass
    return (rss, used, free)


class StrainMonitor:
    """Owns a DEDICATED aiosqlite connection (NOT the shared state-layer one)
    + a reference to the app so it can read live ``app.state`` at sample time.

    Why a private connection: the sampler issues an explicit ``BEGIN
    IMMEDIATE`` to *measure* writer-lock contention. On the shared connection
    that crashes — another coroutine inside a ``transactional_write`` block
    already has an implicit transaction open, so ``BEGIN IMMEDIATE`` raises
    "cannot start a transaction within a transaction" (observed 2026-06-20),
    and ``last_insert_rowid()`` could resolve to a different writer's row.
    A private handle makes the BEGIN safe AND turns it into a real contention
    probe against the rest of the app's writers (matches the resource ledger's
    dedicated ``_ledger_conn`` pattern)."""

    def __init__(
        self,
        conn: "aiosqlite.Connection | None",
        app: "FastAPI",
        *,
        db_path: str | None = None,
    ) -> None:
        # ``conn`` is retained only as a fallback / liveness signal; the write
        # path uses the private connection opened lazily from ``db_path``.
        self._shared_conn = conn
        self._db_path = db_path
        self._own_conn: aiosqlite.Connection | None = None
        self._app = app
        self._last_prune = 0.0

    async def _conn(self) -> "aiosqlite.Connection | None":
        """Return the connection to write samples on.

        Prefers the sampler's PRIVATE connection (opened lazily from
        ``db_path``) so ``BEGIN IMMEDIATE`` is safe + measures real contention.
        Falls back to the shared connection only when no ``db_path`` was given
        (tests passing an in-memory handle, or a non-file backend) — there the
        single-threaded access makes the explicit BEGIN harmless."""
        if self._own_conn is not None:
            return self._own_conn
        if not self._db_path:
            return self._shared_conn
        try:
            import aiosqlite as _aiosqlite

            from augmentum.state.backends.sqlite import apply_augmentum_pragmas

            self._own_conn = await _aiosqlite.connect(self._db_path)
            await apply_augmentum_pragmas(self._own_conn)
        except Exception as exc:
            log.warning("strain_monitor_conn_open_failed", error=str(exc))
            self._own_conn = None
        # Fall back to the shared connection if the private open failed, so a
        # transient open error degrades to "still sampling" not "silent stop".
        return self._own_conn or self._shared_conn

    async def aclose(self) -> None:
        """Close the private connection on shutdown."""
        if self._own_conn is not None:
            try:
                await self._own_conn.close()
            except Exception as exc:
                log.debug("strain_monitor_close_failed", error=str(exc))
            self._own_conn = None

    # ------------------------------------------------------------------ sample
    def _count_active_clients(self) -> tuple[int, int]:
        """Distinct fresh clients + distinct users among them.

        ``app.state.active_clients`` is ``{client_id: (last_seen_monotonic,
        user_id)}`` maintained by the in-flight middleware.
        """
        clients = getattr(self._app.state, "active_clients", None)
        if not clients:
            return (0, 0)
        now = time.monotonic()
        users: set[str] = set()
        n = 0
        # Copy items so a concurrent middleware write can't break iteration.
        for _cid, val in list(clients.items()):
            try:
                last_seen, user_id = val
            except Exception:
                continue
            if now - last_seen <= _CLIENT_FRESH_S:
                n += 1
                if user_id:
                    users.add(user_id)
        return (n, len(users))

    def _engine_models(self) -> tuple[str, str]:
        primary = getattr(self._app.state, "llama_manager", None)
        prim_id = getattr(primary, "model_id", "") or ""
        secondary = getattr(self._app.state, "secondary_slot", None)
        sec_mgr = getattr(secondary, "manager", None) if secondary else None
        sec_id = getattr(sec_mgr, "model_id", "") or ""
        return (prim_id, sec_id)

    def _coder_workspaces(self) -> int:
        cm = getattr(self._app.state, "container_manager", None)
        if cm is None:
            return 0
        # Prefer an in-memory map over an async Docker list call.
        for attr in ("_containers", "_workspaces", "containers", "workspaces"):
            m = getattr(cm, attr, None)
            if isinstance(m, dict):
                return len(m)
        return 0

    def _notify_connections(self) -> int:
        hub = getattr(self._app.state, "notification_hub", None)
        if hub is None:
            return 0
        for attr in ("_connections", "connections", "_clients", "clients"):
            m = getattr(hub, attr, None)
            if isinstance(m, (dict, list, set)):
                return _safe_len(m)
        return 0

    async def sample_and_store(self) -> dict[str, Any] | None:
        """Collect one strain sample, persist it, and return it (or None if
        there's no SQLite backend to write to)."""
        if not self._db_path and self._shared_conn is None:
            return None
        st = self._app.state

        lag_ms = float(getattr(st, "last_event_loop_lag_s", 0.0) or 0.0) * 1000.0
        inflight = int(getattr(st, "inflight_requests", 0) or 0)
        # slow_request counter is read-and-reset so each sample reports the
        # delta since the previous sample.
        slow_reqs = int(getattr(st, "slow_request_count", 0) or 0)
        try:
            st.slow_request_count = 0
        except Exception:
            pass

        active_clients, active_users = self._count_active_clients()
        ws_presence = _safe_len(getattr(st, "presence_pipelines", None))
        ws_notify = self._notify_connections()
        sess_narr = _safe_len(getattr(st, "narrative_engines", None))
        sess_agentic = _safe_len(getattr(st, "agentic_handlers", None))
        sess_coder = self._coder_workspaces()
        engine_model, engine_secondary = self._engine_models()

        # GPU/RAM from the ResourceLedger's CACHED snapshot — never force a
        # fresh nvidia-smi here (that'd make a 10s sampler expensive).
        gpu_used = gpu_free = ram_used = ram_free = 0
        ledger = getattr(st, "resource_ledger", None)
        snap = getattr(ledger, "last_snapshot", None) if ledger else None
        if snap is not None:
            gpu_used = int(getattr(snap, "gpu_used_mb", 0) or 0)
            gpu_free = int(getattr(snap, "gpu_free_mb", 0) or 0)
            ram_used = int(getattr(snap, "ram_used_mb", 0) or 0)
            ram_free = int(getattr(snap, "ram_free_mb", 0) or 0)

        proc_rss, ps_ram_used, ps_ram_free = await asyncio.to_thread(_proc_and_ram_mb)
        # Prefer ledger RAM if present, else psutil.
        if not ram_used:
            ram_used, ram_free = ps_ram_used, ps_ram_free

        context = {
            "ws": {"presence": ws_presence, "notify": ws_notify},
            "sessions": {
                "narrative": sess_narr,
                "agentic": sess_agentic,
                "coder": sess_coder,
            },
        }

        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        row = (
            ts, round(lag_ms, 1), inflight, slow_reqs,
            active_clients, active_users, ws_presence, ws_notify,
            sess_narr, sess_agentic, sess_coder,
            engine_model, engine_secondary, 0.0,  # db_write_ms patched below
            gpu_used, gpu_free, ram_used, ram_free, proc_rss,
            json.dumps(context),
        )

        db = await self._conn()
        if db is None:
            return None

        # Time the writer-lock acquisition + insert: this IS the DB-contention
        # probe. On our PRIVATE connection ``BEGIN IMMEDIATE`` acquires the
        # SQLite write lock; if another connection (the shared state backend)
        # is mid-write, this latency spikes — which is exactly what we want to
        # measure. We capture the inserted rowid from the cursor (not the
        # connection-global ``last_insert_rowid()``) so the latency backfill
        # targets the right row even if anything else writes in between.
        write_start = time.monotonic()
        try:
            await db.execute("BEGIN IMMEDIATE")
            cur = await db.execute(_INSERT_SQL, row)
            rowid = cur.lastrowid
            await db.commit()
        except Exception as exc:
            try:
                await db.rollback()
            except Exception:
                pass
            log.warning("strain_sample_write_failed", error=str(exc))
            return None
        db_write_ms = (time.monotonic() - write_start) * 1000.0

        # Backfill the measured write latency into the just-inserted row.
        try:
            await db.execute(
                "UPDATE strain_samples SET db_write_ms=? WHERE id=?",
                (round(db_write_ms, 1), rowid),
            )
            await db.commit()
        except Exception:
            pass

        sample = {
            "timestamp": ts,
            "event_loop_lag_ms": round(lag_ms, 1),
            "inflight_requests": inflight,
            "slow_requests": slow_reqs,
            "active_clients": active_clients,
            "active_users": active_users,
            "sessions_coder": sess_coder,
            "engine_model": engine_model,
            "engine_secondary": engine_secondary,
            "db_write_ms": round(db_write_ms, 1),
            "proc_rss_mb": proc_rss,
            **context,
        }

        # Grep-able WARNING only when something is actually strained — keeps the
        # log quiet at idle while still surfacing the bad moments like
        # event_loop_stall does.
        if (
            lag_ms >= _LAG_WARN_MS
            or db_write_ms >= _DB_WRITE_WARN_MS
            or inflight >= _INFLIGHT_WARN
        ):
            log.warning(
                "strain_sample",
                lag_ms=round(lag_ms, 1),
                db_write_ms=round(db_write_ms, 1),
                inflight=inflight,
                active_clients=active_clients,
                active_users=active_users,
                coder=sess_coder,
                engine=engine_model,
                proc_rss_mb=proc_rss,
            )

        await self._maybe_prune()
        return sample

    async def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < _PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        db = await self._conn()
        if db is None:
            return
        try:
            await db.execute(
                "DELETE FROM strain_samples "
                "WHERE timestamp < datetime('now', ?)",
                (f"-{_RETENTION_DAYS} days",),
            )
            await db.commit()
        except Exception as exc:
            log.debug("strain_prune_failed", error=str(exc))
