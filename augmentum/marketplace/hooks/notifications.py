"""notifications hook — wire an external service's events into Augmentum's
notification system.

Phase 4: webhook ingest endpoint that connected services POST to.
When a manifest declares ``integration.notifications``, the hook:
  1. Generates a per-service webhook token
  2. Stores it in ``managed_services.config_json``
  3. The service (Uptime Kuma, ntfy, etc.) POSTs events to
     ``POST /api/notifications/ingest`` with the token
  4. Augmentum validates and publishes to the ``service.alert`` channel

The webhook URL is ``{gate_domain}/api/notifications/ingest`` — the
service must be able to reach Augmentum's API. When no gate domain is
configured, the hook logs the token and the admin configures the
webhook manually in the service's own UI.
"""

from __future__ import annotations

import secrets
from typing import Any

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _install(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Generate a webhook token and store it so the ingest endpoint can
    validate incoming events from this service."""
    cfg = manifest.integration.get("notifications", {}) or {}

    # Generate a per-service webhook token.
    token = secrets.token_hex(24)

    # Persist the token in managed_services.config_json so it survives
    # restarts and the ingest endpoint can look it up.
    try:
        mgr = getattr(request.app.state, "service_manager", None)
        if mgr is not None and hasattr(mgr, "read_config_json"):
            existing = await mgr.read_config_json(manifest.service_id) or {}
        else:
            existing = {}
        existing["webhook_token"] = token
        existing["webhook_enabled"] = True
        if mgr is not None and hasattr(mgr, "update_config_json"):
            await mgr.update_config_json(manifest.service_id, existing)
    except Exception:
        log.warning(
            "notifications_hook_token_persist_failed",
            service_id=manifest.service_id, exc_info=True,
        )

    # Log the webhook URL so the admin can configure it in the service.
    # The ingest endpoint is always at /api/notifications/ingest on the
    # same Augmentum instance. Services on the Docker network reach it
    # at http://augmentum:6100/api/notifications/ingest.
    gate = _gate_domain()
    webhook_url = (
        f"https://{gate}/api/notifications/ingest"
        if gate
        else "http://augmentum:6100/api/notifications/ingest"
    )
    log.info(
        "notifications_hook_installed",
        service_id=manifest.service_id,
        webhook_url=webhook_url,
        event_type=str(cfg.get("events") or "service.*"),
    )


async def _uninstall(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Remove the webhook token so ingest requests from this service
    are rejected."""
    try:
        mgr = getattr(request.app.state, "service_manager", None)
        if mgr is not None and hasattr(mgr, "read_config_json"):
            existing = await mgr.read_config_json(manifest.service_id) or {}
        else:
            existing = {}
        existing.pop("webhook_token", None)
        existing["webhook_enabled"] = False
        if mgr is not None and hasattr(mgr, "update_config_json"):
            await mgr.update_config_json(manifest.service_id, existing)
    except Exception:
        log.warning(
            "notifications_hook_uninstall_persist_failed",
            service_id=manifest.service_id, exc_info=True,
        )

    log.info(
        "notifications_hook_uninstalled",
        service_id=manifest.service_id,
    )


def _gate_domain() -> str:
    """The configured front-gate domain, or ""."""
    try:
        from augmentum.config import settings
        return (settings.gate_domain or "").strip().lower()
    except Exception:
        return ""


KNOWN_INTEGRATION_HOOKS["notifications"] = (
    _install,
    _uninstall,
    HookMeta(
        label="Monitoring",
        icon="🔔",
        companion_hint="Companion alerts you when sites go down",
        status_provider="webhook",
    ),
)
