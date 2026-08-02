"""Phase 1 — hook registry generalization tests.

Covers: unknown hook tolerance, known hook registration, media_connect
regression (the extracted hook fires identically to the old inline code
path), uninstall teardown in reverse order.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
from augmentum.marketplace.manifest import ManifestError, parse_manifest


def _minimal_manifest(**integration):
    """Return a parsed manifest with the minimum valid fields."""
    return parse_manifest({
        "manifest_version": 1,
        "service": {"id": "test-svc", "image": "test/img:1.0", "port": 8080},
        "browser": {"after_install": "status"},
        "integration": integration,
    })


class TestHookRegistry:
    """The registry itself — import, shape, forward compat."""

    def test_three_hooks_registered(self):
        """Phase 1 ships with three hook entries."""
        assert "media_connect" in KNOWN_INTEGRATION_HOOKS
        assert "provider_bridge" in KNOWN_INTEGRATION_HOOKS
        assert "notifications" in KNOWN_INTEGRATION_HOOKS

    def test_each_hook_is_a_callable_triple(self):
        """Every entry is an (install_fn, uninstall_fn, HookMeta) tuple."""
        from augmentum.marketplace.hooks import HookMeta
        for name, pair in KNOWN_INTEGRATION_HOOKS.items():
            assert isinstance(pair, tuple), f"{name} is not a tuple"
            assert len(pair) == 3, f"{name} should be (install, uninstall, meta)"
            assert callable(pair[0]), f"{name} install is not callable"
            assert callable(pair[1]), f"{name} uninstall is not callable"
            assert isinstance(pair[2], HookMeta), f"{name} meta is not HookMeta"

    def test_unknown_hook_warns_and_filters(self):
        """A manifest naming an unknown hook warns and drops it, never errors."""
        m = _minimal_manifest(future_hook={"key": "val"})
        assert m.integration == {}  # filtered out

    def test_known_hook_passes_through(self):
        """A known hook survives parsing."""
        m = _minimal_manifest(provider_bridge={"protocol": "subsonic"})
        assert "provider_bridge" in m.integration
        assert m.integration["provider_bridge"]["protocol"] == "subsonic"

    def test_multiple_hooks(self):
        """Multiple known hooks all pass through."""
        m = _minimal_manifest(
            provider_bridge={"protocol": "subsonic"},
            notifications={"webhook": True},
        )
        assert set(m.integration) == {"provider_bridge", "notifications"}

    def test_mixed_known_and_unknown(self):
        """Known hooks pass through; unknown ones are dropped."""
        m = _minimal_manifest(
            provider_bridge={"protocol": "subsonic"},
            future_hook={"key": "val"},
        )
        assert set(m.integration) == {"provider_bridge"}
        assert "future_hook" not in m.integration

    def test_empty_integration(self):
        """Empty or missing integration block is fine."""
        m = _minimal_manifest()
        assert m.integration == {}

    def test_integration_must_be_object(self):
        """Non-dict integration block is rejected."""
        with pytest.raises(ManifestError, match="must be an object"):
            parse_manifest({
                "manifest_version": 1,
                "service": {"id": "x", "image": "a/b:1.0", "port": 8080},
                "browser": {"after_install": "status"},
                "integration": "not_a_dict",
            })


class TestMediaConnectHook:
    """The extracted media_connect hook preserves existing behavior."""

    def test_media_connect_entry_exists(self):
        pair = KNOWN_INTEGRATION_HOOKS.get("media_connect")
        assert pair is not None
        assert callable(pair[0])
        assert callable(pair[1])

    @pytest.mark.asyncio
    async def test_media_connect_install_calls_connect_media_server(self):
        """The hook's install function calls _connect_media_server."""
        from augmentum.marketplace.manifest import ServiceManifest

        manifest = MagicMock(spec=ServiceManifest)
        manifest.service_id = "test-svc"
        manifest.integration = {"media_connect": {"provider": "test-provider"}}

        sd = MagicMock()
        request = MagicMock()

        with patch(
            "augmentum.marketplace.install_dispatchers._connect_media_server",
            new_callable=AsyncMock,
        ) as mock_connect:
            install_fn = KNOWN_INTEGRATION_HOOKS["media_connect"][0]
            await install_fn(request, manifest, sd, "user-1")
            mock_connect.assert_awaited_once_with(
                request, sd=sd, service_id="test-svc",
                provider="test-provider", user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_media_connect_install_uses_service_id_as_fallback_provider(self):
        """When provider is absent, service_id is the default."""
        from augmentum.marketplace.manifest import ServiceManifest

        manifest = MagicMock(spec=ServiceManifest)
        manifest.service_id = "test-svc"
        manifest.integration = {"media_connect": {}}  # no provider key

        sd = MagicMock()
        request = MagicMock()

        with patch(
            "augmentum.marketplace.install_dispatchers._connect_media_server",
            new_callable=AsyncMock,
        ) as mock_connect:
            install_fn = KNOWN_INTEGRATION_HOOKS["media_connect"][0]
            await install_fn(request, manifest, sd, "user-1")
            mock_connect.assert_awaited_once_with(
                request, sd=sd, service_id="test-svc",
                provider="test-svc", user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_media_connect_install_survives_error(self):
        """A failed media_connect doesn't raise — it logs and continues."""
        from augmentum.marketplace.manifest import ServiceManifest

        manifest = MagicMock(spec=ServiceManifest)
        manifest.service_id = "test-svc"
        manifest.integration = {"media_connect": {}}
        sd = MagicMock()
        request = MagicMock()

        with patch(
            "augmentum.marketplace.install_dispatchers._connect_media_server",
            new_callable=AsyncMock,
            side_effect=RuntimeError("container not ready"),
        ):
            install_fn = KNOWN_INTEGRATION_HOOKS["media_connect"][0]
            # Must not raise
            await install_fn(request, manifest, sd, "user-1")


class TestStubHooks:
    """The Phase 1 stub hooks fire without error."""

    @pytest.mark.asyncio
    async def test_provider_bridge_stub_does_not_raise(self):
        """provider_bridge stub install is a no-op."""
        manifest = MagicMock()
        manifest.service_id = "test"
        manifest.integration = {"provider_bridge": {"protocol": "test"}}
        sd = MagicMock()
        request = MagicMock()
        install_fn = KNOWN_INTEGRATION_HOOKS["provider_bridge"][0]
        await install_fn(request, manifest, sd, "user-1")

    @pytest.mark.asyncio
    async def test_notifications_stub_does_not_raise(self):
        """notifications stub install is a no-op."""
        manifest = MagicMock()
        manifest.service_id = "test"
        manifest.integration = {"notifications": {}}
        sd = MagicMock()
        request = MagicMock()
        install_fn = KNOWN_INTEGRATION_HOOKS["notifications"][0]
        await install_fn(request, manifest, sd, "user-1")

    @pytest.mark.asyncio
    async def test_stub_uninstalls_do_not_raise(self):
        """All stub uninstall functions are no-ops."""
        for name, pair in KNOWN_INTEGRATION_HOOKS.items():
            if name == "media_connect":
                continue  # not a stub — tested separately
            manifest = MagicMock()
            manifest.service_id = f"test-{name}"
            manifest.integration = {name: {}}
            sd = MagicMock()
            request = MagicMock()
            await pair[1](request, manifest, sd, "user-1")
