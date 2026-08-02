"""FabricClient reconnect-loop logging: a persistently-dead peer must warn
ONCE on the up->down transition, then drop to debug while it stays down, and
emit a single info on recovery. Prevents the ~2880-warnings/day log spam a
powered-off peer produced (observed live 2026-06-20)."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.fabric import client as fc
from augmentum.fabric.client import FabricClient


def _make_client() -> FabricClient:
    # _maintain_connection only touches self._stopping and self._one_session;
    # the identity/coordinator/http args are never read on that path.
    return FabricClient(identity=object(), coordinator=object(), http_client=object())


class _Peer:
    node_id = "deadpeer"
    addr = "wss://10.0.0.9:6443"


def _drive(client: FabricClient, *, session_impl, stop_after: int):
    """Run _maintain_connection with a fake _one_session, stopping the loop
    after ``stop_after`` iterations. Backoff is shrunk so the test is fast."""
    calls = {"n": 0}

    async def _session(peer):
        calls["n"] += 1
        if calls["n"] >= stop_after:
            client._stopping.set()
        await session_impl(peer, calls["n"])

    client._one_session = _session  # type: ignore[assignment]
    asyncio.run(client._maintain_connection(_Peer()))
    return calls["n"]


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr(fc, "_RECONNECT_BACKOFF_INITIAL_S", 0.0)
    monkeypatch.setattr(fc, "_RECONNECT_BACKOFF_MAX_S", 0.0)


@pytest.fixture
def _log_counter(monkeypatch):
    counts = {"warning": [], "debug": [], "info": []}
    monkeypatch.setattr(fc.log, "warning", lambda ev, **kw: counts["warning"].append((ev, kw)))
    monkeypatch.setattr(fc.log, "debug", lambda ev, **kw: counts["debug"].append((ev, kw)))
    monkeypatch.setattr(fc.log, "info", lambda ev, **kw: counts["info"].append((ev, kw)))
    return counts


def test_persistent_failure_warns_once_then_debug(_log_counter):
    client = _make_client()

    async def _always_fail(peer, n):
        raise ConnectionRefusedError("Connection refused")

    _drive(client, session_impl=_always_fail, stop_after=5)

    warns = [e for e, _ in _log_counter["warning"] if e == "fabric_client_session_failed"]
    debugs = [e for e, _ in _log_counter["debug"] if e == "fabric_client_session_failed"]
    # Exactly one warning (the transition into down), the rest at debug.
    assert len(warns) == 1
    assert len(debugs) == 4
    # The first failure carries consecutive_failures=1.
    first = next(kw for e, kw in _log_counter["warning"] if e == "fabric_client_session_failed")
    assert first["consecutive_failures"] == 1


def test_recovery_emits_info_and_resets_breaker(_log_counter):
    client = _make_client()

    async def _fail_then_recover(peer, n):
        # Fail the first two attempts, then "connect" cleanly (no raise).
        if n <= 2:
            raise ConnectionRefusedError("Connection refused")
        # n==3: clean session return → recovery path.

    _drive(client, session_impl=_fail_then_recover, stop_after=3)

    recovered = [kw for e, kw in _log_counter["info"] if e == "fabric_client_peer_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["after_failures"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
