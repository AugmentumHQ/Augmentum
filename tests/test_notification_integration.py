"""Phase 4 — notification hook, ingest endpoint, and channel tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNotificationChannel:
    """The service.alert channel is registered in the catalog."""

    def test_service_alert_channel_exists(self):
        from augmentum.notifications.catalog import catalog_channel, DEFAULT_CHANNELS
        ch = catalog_channel("service.alert")
        assert ch is not None
        assert ch.channel_id == "service.alert"
        assert ch.name == "Service alert"
        assert ch.importance == 3  # IMPORTANCE_HIGH

    def test_all_channel_ids_have_service_alert(self):
        from augmentum.notifications.catalog import catalog_channel_ids
        assert "service.alert" in catalog_channel_ids()


class TestNotificationsHook:
    """The notifications hook is no longer a stub."""

    def test_hook_is_registered(self):
        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        assert "notifications" in KNOWN_INTEGRATION_HOOKS
        install_fn = KNOWN_INTEGRATION_HOOKS["notifications"][0]
        uninstall_fn = KNOWN_INTEGRATION_HOOKS["notifications"][1]
        assert callable(install_fn)
        assert callable(uninstall_fn)

    @pytest.mark.asyncio
    async def test_install_stores_webhook_token_in_config_json(self):
        """The install function persists a webhook token."""
        manifest = MagicMock()
        manifest.service_id = "uptime-kuma"
        manifest.integration = {"notifications": {"events": ["monitor.*"]}}
        sd = MagicMock()
        sd.internal_port = 3001

        request = MagicMock()
        mgr_mock = MagicMock()
        mgr_mock.read_config_json = AsyncMock(return_value={})
        mgr_mock.update_config_json = AsyncMock()
        request.app.state.service_manager = mgr_mock

        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        install_fn = KNOWN_INTEGRATION_HOOKS["notifications"][0]
        await install_fn(request, manifest, sd, "user-1")

        # Should have called update_config_json with a webhook_token
        mgr_mock.update_config_json.assert_awaited_once()
        call_args = mgr_mock.update_config_json.call_args[0]
        cfg = call_args[1]
        assert "webhook_token" in cfg
        assert cfg["webhook_enabled"] is True
        assert len(cfg["webhook_token"]) == 48  # hex(24 bytes)

    @pytest.mark.asyncio
    async def test_uninstall_removes_webhook_token(self):
        """The uninstall function clears the webhook token."""
        manifest = MagicMock()
        manifest.service_id = "uptime-kuma"
        manifest.integration = {"notifications": {}}
        sd = MagicMock()
        request = MagicMock()
        mgr_mock = MagicMock()
        mgr_mock.read_config_json = AsyncMock(
            return_value={"webhook_token": "test-token", "webhook_enabled": True},
        )
        mgr_mock.update_config_json = AsyncMock()
        request.app.state.service_manager = mgr_mock

        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        uninstall_fn = KNOWN_INTEGRATION_HOOKS["notifications"][1]
        await uninstall_fn(request, manifest, sd, "user-1")

        call_args = mgr_mock.update_config_json.call_args[0]
        cfg = call_args[1]
        assert "webhook_token" not in cfg
        assert cfg["webhook_enabled"] is False

    @pytest.mark.asyncio
    async def test_install_survives_missing_manager(self):
        """No service_manager on app.state — logs and continues."""
        manifest = MagicMock()
        manifest.service_id = "test"
        manifest.integration = {"notifications": {}}
        sd = MagicMock()
        request = MagicMock()
        request.app.state.service_manager = None

        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        install_fn = KNOWN_INTEGRATION_HOOKS["notifications"][0]
        # Must not raise
        await install_fn(request, manifest, sd, "user-1")


class TestIngestEndpoint:
    """The POST /api/notifications/ingest endpoint validates tokens and publishes."""

    def _make_request(self, body: dict, *, conn=None, users=None):
        """Build a FastAPI Request mock with app.state wired."""
        request = MagicMock()
        request.json = AsyncMock(return_value=body)

        if conn is None:
            conn = AsyncMock()

        # Mock the DB queries
        async def mock_execute(sql, params=None):
            m = MagicMock()
            if "SELECT config_json FROM managed_services" in sql:
                sid = params[0] if params else ""
                if sid == "uptime-kuma":
                    m.fetchone = AsyncMock(return_value=(
                        json.dumps({"webhook_token": "valid-token", "webhook_enabled": True}),
                    ))
                else:
                    m.fetchone = AsyncMock(return_value=None)
            elif "SELECT id FROM users WHERE is_admin" in sql:
                rows = users if users is not None else [("admin-user-id",)]
                m.fetchone = AsyncMock(return_value=rows[0] if rows else None)
            else:
                m.fetchone = AsyncMock(return_value=None)
            m.close = AsyncMock()
            return m

        conn.execute = mock_execute
        request.app.state.state_manager = MagicMock()
        request.app.state.state_manager.backend = MagicMock()
        request.app.state.state_manager.backend.conn = conn

        return request

    @pytest.mark.asyncio
    async def test_valid_token_publishes(self):
        """A valid token returns 200 and publishes the notification."""
        with patch(
            "augmentum.notifications.store.publish",
            new_callable=AsyncMock,
            return_value="notif-123",
        ) as mock_publish:
            from augmentum.proxy.notification_routes import notification_ingest

            request = self._make_request({
                "service_id": "uptime-kuma",
                "token": "valid-token",
                "title": "blog.example.com is DOWN",
                "body": "HTTP 500 at 14:32",
                "source": "uptime-kuma",
                "dedupe_key": "mon-1",
            })
            response = await notification_ingest(request)
            assert response.status_code == 200
            data = json.loads(response.body)
            assert data["status"] == "published"
            assert data["notification_id"] == "notif-123"
            mock_publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self):
        """A bad token returns 401."""
        from augmentum.proxy.notification_routes import notification_ingest

        request = self._make_request({
            "service_id": "uptime-kuma",
            "token": "wrong-token",
            "title": "test",
        })
        response = await notification_ingest(request)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_service_returns_404(self):
        """An unknown service_id returns 404."""
        from augmentum.proxy.notification_routes import notification_ingest

        request = self._make_request({
            "service_id": "nonexistent",
            "token": "anything",
            "title": "test",
        })
        response = await notification_ingest(request)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_fields_returns_400(self):
        """Missing required fields return 400."""
        from augmentum.proxy.notification_routes import notification_ingest

        request = self._make_request({})
        response = await notification_ingest(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_no_admin_user_returns_404(self):
        """When no admin user exists, returns 404."""
        from augmentum.proxy.notification_routes import notification_ingest

        request = self._make_request(
            {
                "service_id": "uptime-kuma",
                "token": "valid-token",
                "title": "test",
            },
            users=[],  # no admin users
        )
        response = await notification_ingest(request)
        assert response.status_code == 404
