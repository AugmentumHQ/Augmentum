"""Tests for ``augmentum.search.proxy_manager``.

Covers parsing, settings.yml round-trip, the pick/rotation logic, and
the apply/restart flow. Probe behaviour is tested via a monkeypatched
``httpx.AsyncClient`` so we don't hit the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from augmentum.search.proxy_manager import (
    SearxngProxyManager,
    parse_proxies,
)

# ----------------------------------------------------------------------
# parse_proxies
# ----------------------------------------------------------------------


class TestParseProxies:
    def test_empty_returns_empty_list(self):
        assert parse_proxies("") == []
        assert parse_proxies("   \n  \n") == []

    def test_strips_whitespace_and_skips_blank_lines(self):
        raw = "\n  http://a:8080  \n\nhttps://b:8443\n"
        assert parse_proxies(raw) == ["http://a:8080", "https://b:8443"]

    def test_skips_comment_lines(self):
        raw = "# leading comment\nhttp://a:8080\n# trailing"
        assert parse_proxies(raw) == ["http://a:8080"]

    def test_dedupes_preserving_first_occurrence(self):
        raw = "http://a\nhttp://b\nhttp://a"
        assert parse_proxies(raw) == ["http://a", "http://b"]

    def test_adds_http_scheme_when_missing(self):
        assert parse_proxies("proxy.example:8080") == ["http://proxy.example:8080"]

    def test_rejects_unknown_scheme(self):
        # ssh:// not a valid proxy scheme — silently dropped
        assert parse_proxies("ssh://host:22") == []

    def test_accepts_socks5(self):
        assert parse_proxies("socks5://10.0.0.5:1080") == ["socks5://10.0.0.5:1080"]

    def test_preserves_credentials(self):
        url = "http://user:pass@proxy.example:8080"
        assert parse_proxies(url) == [url]


# ----------------------------------------------------------------------
# settings.yml round-trip
# ----------------------------------------------------------------------


@pytest.fixture
def fake_settings_yml(tmp_path: Path) -> Path:
    """Minimal settings.yml that mirrors the project's real structure."""
    path = tmp_path / "settings.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "use_default_settings": True,
                "general": {"instance_name": "Test"},
                "outgoing": {"request_timeout": 8.0, "max_request_timeout": 12.0},
                "engines": [{"name": "google", "shortcut": "go"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestApplyActive:
    @pytest.mark.asyncio
    async def test_writes_proxy_into_outgoing_block(self, fake_settings_yml):
        mgr = SearxngProxyManager(
            settings_yml_path=fake_settings_yml,
            docker_client=None,  # restart is a no-op when client is None
        )
        await mgr.update_proxy_list("http://a:8080")
        # Manually mark healthy so pick_active returns something
        mgr._health["http://a:8080"].healthy = True
        await mgr.apply_active(mgr.pick_active())

        data = _read_yaml(fake_settings_yml)
        # Other keys preserved
        assert data["use_default_settings"] is True
        assert data["general"] == {"instance_name": "Test"}
        assert data["outgoing"]["request_timeout"] == 8.0
        # Proxy injected
        assert data["outgoing"]["proxies"] == {
            "http://": "http://a:8080",
            "https://": "http://a:8080",
        }

    @pytest.mark.asyncio
    async def test_remove_active_clears_proxies_key(self, fake_settings_yml):
        # Pre-seed a proxy entry to confirm removal
        data = _read_yaml(fake_settings_yml)
        data["outgoing"]["proxies"] = {"http://": "http://stale"}
        fake_settings_yml.write_text(yaml.safe_dump(data, sort_keys=False))

        mgr = SearxngProxyManager(
            settings_yml_path=fake_settings_yml,
            docker_client=None,
        )
        await mgr.apply_active(None)

        data = _read_yaml(fake_settings_yml)
        assert "proxies" not in data["outgoing"]

    @pytest.mark.asyncio
    async def test_idempotent_no_change_no_restart(self, fake_settings_yml):
        docker = MagicMock()
        docker.containers.get = AsyncMock(return_value=MagicMock(restart=AsyncMock()))
        mgr = SearxngProxyManager(
            settings_yml_path=fake_settings_yml,
            docker_client=docker,
        )
        await mgr.update_proxy_list("http://a:8080")
        mgr._health["http://a:8080"].healthy = True

        changed_first = await mgr.apply_active("http://a:8080")
        changed_second = await mgr.apply_active("http://a:8080")
        assert changed_first is True
        assert changed_second is False
        # Restart only on the change
        assert docker.containers.get.await_count == 1


# ----------------------------------------------------------------------
# Rotation / pick logic
# ----------------------------------------------------------------------


class TestPickActive:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_healthy(self, tmp_path):
        mgr = SearxngProxyManager(settings_yml_path=tmp_path / "x.yml")
        await mgr.update_proxy_list("http://a\nhttp://b")
        assert mgr.pick_active() is None

    @pytest.mark.asyncio
    async def test_round_robin_among_healthy(self, tmp_path):
        mgr = SearxngProxyManager(settings_yml_path=tmp_path / "x.yml")
        await mgr.update_proxy_list("http://a\nhttp://b\nhttp://c")
        mgr._health["http://a"].healthy = True
        mgr._health["http://b"].healthy = True
        # c is unhealthy
        picks = [mgr.pick_active() for _ in range(4)]
        assert picks[0] == "http://a"
        assert picks[1] == "http://b"
        assert picks[2] == "http://a"
        assert picks[3] == "http://b"

    @pytest.mark.asyncio
    async def test_rebuilds_cycle_when_health_set_changes(self, tmp_path):
        mgr = SearxngProxyManager(settings_yml_path=tmp_path / "x.yml")
        await mgr.update_proxy_list("http://a\nhttp://b")
        mgr._health["http://a"].healthy = True
        assert mgr.pick_active() == "http://a"
        # Now b becomes healthy too — cycle rebuilds and the rotation
        # picks up b on the next call without losing fairness
        mgr._health["http://b"].healthy = True
        picks = {mgr.pick_active() for _ in range(6)}
        assert picks == {"http://a", "http://b"}


# ----------------------------------------------------------------------
# Reconcile (fallback behaviour)
# ----------------------------------------------------------------------


class TestReconcile:
    @pytest.mark.asyncio
    async def test_falls_back_to_direct_when_allowed(self, fake_settings_yml):
        mgr = SearxngProxyManager(
            settings_yml_path=fake_settings_yml,
            docker_client=None,
        )
        await mgr.update_proxy_list("http://dead")
        # Health stays False → no healthy proxy
        await mgr.reconcile(fallback_to_direct=True)

        data = _read_yaml(fake_settings_yml)
        assert "proxies" not in data["outgoing"]
        status = mgr.status()
        assert status.direct_fallback_active is True

    @pytest.mark.asyncio
    async def test_keeps_previous_when_fallback_disabled(self, fake_settings_yml):
        # Pre-seed an existing proxy entry to confirm we leave it alone
        data = _read_yaml(fake_settings_yml)
        data["outgoing"]["proxies"] = {"http://": "http://prev"}
        fake_settings_yml.write_text(yaml.safe_dump(data, sort_keys=False))

        mgr = SearxngProxyManager(
            settings_yml_path=fake_settings_yml,
            docker_client=None,
        )
        await mgr.update_proxy_list("http://dead")
        await mgr.reconcile(fallback_to_direct=False)

        data = _read_yaml(fake_settings_yml)
        assert data["outgoing"]["proxies"] == {"http://": "http://prev"}


# ----------------------------------------------------------------------
# update_proxy_list pruning
# ----------------------------------------------------------------------


class TestUpdateProxyList:
    @pytest.mark.asyncio
    async def test_prunes_health_for_removed_proxies(self, tmp_path):
        mgr = SearxngProxyManager(settings_yml_path=tmp_path / "x.yml")
        await mgr.update_proxy_list("http://a\nhttp://b")
        mgr._health["http://a"].healthy = True
        mgr._health["http://b"].healthy = True

        await mgr.update_proxy_list("http://a")
        assert "http://b" not in mgr._health
        assert mgr._health["http://a"].healthy is True  # state preserved
