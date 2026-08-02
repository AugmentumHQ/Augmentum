"""Tests for augmentum/resource/host_probe.py — the host stats agent client."""

from __future__ import annotations

import httpx
import pytest

from augmentum.resource import host_probe


@pytest.fixture(autouse=True)
def _reset_host_probe_state(monkeypatch):
    """Clear the module-level cache + backoff timers before each test."""
    monkeypatch.setattr(host_probe, "_cache", None, raising=False)
    monkeypatch.setattr(host_probe, "_cache_at", 0.0, raising=False)
    monkeypatch.setattr(host_probe, "_last_attempt_at", 0.0, raising=False)
    monkeypatch.setattr(host_probe, "_logged_discovery", False, raising=False)
    monkeypatch.setattr(host_probe, "_lock", None, raising=False)
    # Default: no env config, not containerised.
    monkeypatch.delenv("AUGMENTUM_HOST_STATS_URL", raising=False)
    monkeypatch.delenv("AUGMENTUM_HOST_STATS_TOKEN", raising=False)
    monkeypatch.setattr(host_probe.os.path, "exists", lambda p: False)


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient.get used by probe_host_stats."""

    def __init__(self, *, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.calls: list[str] = []

    async def get(self, url, **_kw):
        self.calls.append(url)
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._payload)


_AGENT_PAYLOAD = {
    "ram": {"total_mb": 65536, "used_mb": 40000, "free_mb": 25000},
    "cpu_pct": 31.4,
    "cpu_count": 24,
    "os": "Windows",
    "hostname": "DESKTOP-ABC",
    "agent": "augmentum-host-stats/1",
}


def test_resolve_url_env_override(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_HOST_STATS_URL", "http://example:7000/stats")
    assert host_probe._resolve_url() == "http://example:7000/stats"


def test_resolve_url_default_in_container(monkeypatch):
    monkeypatch.setattr(host_probe.os.path, "exists", lambda p: p == "/.dockerenv")
    assert host_probe._resolve_url() == host_probe._DEFAULT_CONTAINER_URL


def test_resolve_url_disabled_on_bare_metal():
    assert host_probe._resolve_url() == ""


def test_with_token(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_HOST_STATS_TOKEN", "s3cret")
    assert host_probe._with_token("http://h/stats") == "http://h/stats?token=s3cret"
    assert host_probe._with_token("http://h/stats?x=1") == "http://h/stats?x=1&token=s3cret"


def test_with_token_absent():
    assert host_probe._with_token("http://h/stats") == "http://h/stats"


def test_parse_maps_payload():
    stats = host_probe._parse(_AGENT_PAYLOAD)
    assert stats.ram_total_mb == 65536
    assert stats.ram_used_mb == 40000
    assert stats.ram_free_mb == 25000
    assert stats.cpu_pct == 31.4
    assert stats.cpu_count == 24
    assert stats.os_name == "Windows"
    assert stats.hostname == "DESKTOP-ABC"


def test_parse_tolerates_missing_fields():
    stats = host_probe._parse({})
    assert stats.ram_total_mb == 0
    assert stats.cpu_pct == 0.0
    assert stats.os_name == ""



async def test_probe_returns_none_when_disabled():
    client = _FakeClient(payload=_AGENT_PAYLOAD)
    assert await host_probe.probe_host_stats(client) is None
    assert client.calls == []  # never even tried



async def test_probe_success_and_cache(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_HOST_STATS_URL", "http://h:6109/stats")
    client = _FakeClient(payload=_AGENT_PAYLOAD)

    first = await host_probe.probe_host_stats(client)
    assert first is not None
    assert first.ram_total_mb == 65536
    assert client.calls == ["http://h:6109/stats"]

    # Second call within the hit-TTL must be served from cache (no new GET).
    second = await host_probe.probe_host_stats(client)
    assert second is first
    assert client.calls == ["http://h:6109/stats"]



async def test_probe_miss_backs_off(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_HOST_STATS_URL", "http://h:6109/stats")
    client = _FakeClient(exc=httpx.ConnectError("refused"))

    assert await host_probe.probe_host_stats(client) is None
    assert len(client.calls) == 1

    # Immediately after a failure we should NOT retry (60s backoff).
    assert await host_probe.probe_host_stats(client) is None
    assert len(client.calls) == 1

    # ...but once the backoff window elapses, we try again.
    monkeypatch.setattr(host_probe, "_last_attempt_at",
                        host_probe._last_attempt_at - host_probe._MISS_RETRY_S - 1)
    assert await host_probe.probe_host_stats(client) is None
    assert len(client.calls) == 2
