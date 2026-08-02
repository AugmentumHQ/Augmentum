"""Runtime health-probe tests — the live-app dimensions, via a fake app_state."""

from __future__ import annotations

from types import SimpleNamespace

import aiosqlite

from augmentum.selfedit import health as H
from augmentum.selfedit import probes as P


class _Backend:
    def __init__(self, ok=True):
        self._ok = ok

    async def list_models(self):
        if not self._ok:
            raise RuntimeError("backend down")
        return ["m1", "m2"]


def _state(*, backends=None, services=None, conn=None):
    return SimpleNamespace(
        provider_registry=SimpleNamespace(backends=backends or {}),
        service_health=(SimpleNamespace(snapshot=lambda: services)
                        if services is not None else None),
        state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)),
    )


async def _strain_conn(*lags):
    """In-memory strain_samples with the given recent lag readings (newest last)."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE strain_samples (timestamp TEXT, event_loop_lag_ms REAL)")
    for lag in lags:
        await conn.execute("INSERT INTO strain_samples VALUES (datetime('now'), ?)", (lag,))
    await conn.commit()
    return conn


async def test_backends_probe_all_ok_and_partial():
    ok = await P.backends_probe(_state(backends={"a": _Backend(), "b": _Backend()}))()
    assert ok.ok is True and ok.score == 1.0

    partial = await P.backends_probe(_state(backends={"a": _Backend(), "b": _Backend(ok=False)}))()
    assert partial.ok is False and partial.score == 0.5 and "down" in partial.detail

    none = await P.backends_probe(_state(backends={}))()
    assert none.ok is False  # nothing serving is a failure


async def test_services_probe_status_grading():
    up = await P.services_probe(_state(services={"x": {"status": "up"}, "y": {"status": "up"}}))()
    assert up.ok is True and up.score == 1.0

    down = await P.services_probe(_state(services={"x": {"status": "up"}, "y": {"status": "down"}}))()
    assert down.ok is False and down.score < 1.0  # a down dep sinks ok

    degraded = await P.services_probe(_state(services={"x": {"status": "degraded"}}))()
    assert degraded.ok is True and 0.0 < degraded.score < 1.0  # degraded ≠ failure

    untracked = await P.services_probe(_state(services={}))()
    assert untracked.measured is False


async def test_db_integrity_probe():
    conn = await _strain_conn()
    try:
        r = await P.db_integrity_probe(_state(conn=conn))()
        assert r.ok is True and r.score == 1.0 and r.detail == "ok"
    finally:
        await conn.close()
    # no sqlite → unmeasured, not a false failure
    nodb = await P.db_integrity_probe(_state(conn=None))()
    assert nodb.measured is False


async def test_strain_probe_grading():
    low = await _strain_conn(50)
    try:
        r = await P.strain_probe(_state(conn=low))()
        assert r.score == 1.0 and r.ok is True
    finally:
        await low.close()

    high = await _strain_conn(3000)
    try:
        r = await P.strain_probe(_state(conn=high))()
        assert r.score == 0.0 and r.ok is False  # sustained-bad flips not-ok
    finally:
        await high.close()

    # The fix the real data demanded: a recent spike must be caught even when the
    # LATEST sample is fine (window-max, not latest-only).
    bursty = await _strain_conn(3000, 0)  # spike then back to healthy
    try:
        r = await P.strain_probe(_state(conn=bursty))()
        assert r.score == 0.0 and "3000" in r.detail  # window-max catches it
    finally:
        await bursty.close()

    empty = await _strain_conn()
    try:
        r = await P.strain_probe(_state(conn=empty))()
        assert r.measured is False  # no recent samples → unmeasured
    finally:
        await empty.close()


async def test_register_runtime_probes_into_signal():
    H.clear_registry()
    try:
        conn = await _strain_conn(10)
        try:
            st = _state(backends={"a": _Backend()},
                        services={"s": {"status": "up"}}, conn=conn)
            n = P.register_runtime_probes(st)
            assert n == 4
            report = await H.assess()  # over the registry
            names = {d.name for d in report.dimensions}
            assert {"backends", "services", "db_integrity", "strain"} <= names
            assert report.ok is True
        finally:
            await conn.close()
    finally:
        H.clear_registry()
