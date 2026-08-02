"""The real Augmentum health probes — wiring the Health Signal to what the app
already tracks, so the signal is comprehensive by composition rather than by
re-measuring.

Two contexts:

* **Runtime probes** (this module's focus) read the LIVE app's existing
  telemetry in-process via ``app.state`` — the same sources the /api/health*
  endpoints expose: provider backends, the service-health registry, the
  strain-sample series, and SQLite integrity. These are what tell us a promoted
  change kept the running app healthy (boot-health + rollback trigger).
* **Static probes** live in ``gate``/``health.default_probes`` (compile, ruff,
  pytest) and assess a candidate's files before it ever runs.

Each probe is a zero-arg async callable returning a ``DimensionResult`` (closed
over the dependency it reads), so the registry stays uniform and probes are
trivially testable with a fake ``app_state``.
"""

from __future__ import annotations

from typing import Any

from augmentum.selfedit.health import DimensionResult, Probe, register_probe
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _conn(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


def backends_probe(app_state: Any) -> Probe:
    """Are the model/provider backends actually serving? (Same check as
    /api/health.) Weighted heavily — a boot that can't serve models is the
    classic 'alive but useless' regression."""
    async def p() -> DimensionResult:
        reg = getattr(app_state, "provider_registry", None)
        backends = getattr(reg, "backends", {}) if reg else {}
        if not backends:
            return DimensionResult("backends", ok=False, score=0.0,
                                   detail="no backends registered", weight=2.0)
        results: dict[str, bool] = {}
        for key, backend in backends.items():
            try:
                await backend.list_models()
                results[key] = True
            except Exception as exc:  # noqa: BLE001
                log.debug("health_backend_probe_error", backend=key, error=repr(exc))
                results[key] = False
        ok_count = sum(1 for v in results.values() if v)
        total = len(results) or 1
        detail = ", ".join(f"{k}:{'ok' if v else 'down'}" for k, v in results.items())
        return DimensionResult("backends", ok=(ok_count == total),
                               score=ok_count / total, detail=detail, weight=2.0)
    return p


def services_probe(app_state: Any) -> Probe:
    """Tracked dependencies (searxng/executor/tts/stt/…) via the service-health
    registry. Advisory: external deps being down isn't the app code's fault, but
    a self-edit that knocks a service down still shows as a score regression."""
    async def p() -> DimensionResult:
        health = getattr(app_state, "service_health", None)
        snap = health.snapshot() if health else {}
        if not snap:
            return DimensionResult("services", ok=True, score=1.0,
                                   detail="no services tracked", measured=False, required=False)
        statuses = {
            name: (info.get("status") if isinstance(info, dict) else str(info))
            for name, info in snap.items()
        }

        def val(s: str) -> float:
            return {"up": 1.0, "degraded": 0.5, "unknown": 0.8}.get(s, 0.0)

        total = len(statuses) or 1
        score = sum(val(s) for s in statuses.values()) / total
        down = [n for n, s in statuses.items() if s == "down"]
        degraded = [n for n, s in statuses.items() if s == "degraded"]
        detail = (f"down={down} degraded={degraded}" if (down or degraded)
                  else f"{total} up")
        return DimensionResult("services", ok=not down, score=score,
                               detail=detail, required=False)
    return p


def db_integrity_probe(app_state: Any) -> Probe:
    """SQLite integrity (PRAGMA quick_check) — data safety is non-negotiable, so
    weighted heavily and required."""
    async def p() -> DimensionResult:
        conn = _conn(app_state)
        if conn is None:
            return DimensionResult("db_integrity", ok=True, score=1.0,
                                   detail="no sqlite backend", measured=False)
        try:
            cur = await conn.execute("PRAGMA quick_check")
            rows = await cur.fetchall()
            await cur.close()
        except Exception as exc:  # noqa: BLE001
            return DimensionResult("db_integrity", ok=False, score=0.0, detail=repr(exc), weight=2.0)
        res = (str(rows[0][0]).strip().lower() if rows else "")
        ok = res == "ok"
        return DimensionResult("db_integrity", ok=ok, score=1.0 if ok else 0.0,
                               detail=res[:300], weight=2.0)
    return p


def strain_probe(app_state: Any, *, window_minutes: int = 5,
                 lag_warn_ms: float = 250.0, lag_bad_ms: float = 2000.0) -> Probe:
    """Runtime degradation from event-loop lag. Uses the WORST lag over a recent
    window — not the latest sample — because real lag is bursty (observed 0ms
    typical, 47s spikes), so a single good instant would mask intermittent
    stalls. Graded + advisory: feeds the fitness score + regression detection
    (a self-edit that introduces stalls raises the recent max → caught as a
    delta vs baseline); only sustained-bad flips it not-ok."""
    async def p() -> DimensionResult:
        conn = _conn(app_state)
        if conn is None:
            return DimensionResult("strain", ok=True, score=1.0, measured=False, required=False)
        try:
            cur = await conn.execute(
                "SELECT max(event_loop_lag_ms), count(*) FROM strain_samples "
                "WHERE timestamp >= datetime('now', ?)",
                (f"-{int(window_minutes)} minutes",),
            )
            row = await cur.fetchone()
            await cur.close()
        except Exception as exc:  # noqa: BLE001
            return DimensionResult("strain", ok=True, score=1.0,
                                   detail=f"no strain data: {exc!r}", measured=False, required=False)
        worst = row[0] if row else None
        count = int(row[1]) if row and row[1] is not None else 0
        if worst is None or count == 0:
            return DimensionResult("strain", ok=True, score=1.0,
                                   detail="no recent samples", measured=False, required=False)
        worst = float(worst)
        if worst <= lag_warn_ms:
            score = 1.0
        elif worst >= lag_bad_ms:
            score = 0.0
        else:
            score = 1.0 - (worst - lag_warn_ms) / (lag_bad_ms - lag_warn_ms)
        return DimensionResult("strain", ok=(worst < lag_bad_ms), score=score,
                               detail=f"max_lag_{window_minutes}m={worst:.0f}ms (n={count})",
                               required=False)
    return p


def runtime_probes(app_state: Any) -> dict[str, Probe]:
    """The live-app health set, all reading existing telemetry in-process."""
    return {
        "backends": backends_probe(app_state),
        "services": services_probe(app_state),
        "db_integrity": db_integrity_probe(app_state),
        "strain": strain_probe(app_state),
    }


def register_runtime_probes(app_state: Any) -> int:
    """Register the live-app probes into the health registry (called at startup),
    so ``health.assess()`` over the registry reports the running app's health."""
    probes = runtime_probes(app_state)
    for name, probe in probes.items():
        register_probe(name, probe)
    log.info("selfedit_runtime_probes_registered", count=len(probes))
    return len(probes)
