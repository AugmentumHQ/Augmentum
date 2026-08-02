"""augmentum_backend hook — hand Augmentum's own OpenAI-compatible API to a
provisioned service so it uses the user's Augmentum models / voices / (later)
embeddings out of the box, with ZERO manual configuration.

This is the "Augmentum provides what it does to these services" seam
(Open WebUI, Flowise, n8n, …): install the app and it's already talking to
your models.

Why the real work is NOT in ``_install`` here:
    Container env is baked at *create* time, so the key mint + env injection
    must run PRE-provision — it lives in
    ``install_dispatchers._install_service_manifest`` (right where the
    generate=True secret minting is). This hook module exists so that

      1. ``parse_manifest`` PRESERVES the ``augmentum_backend`` config block
         (it drops any integration hook not registered in
         ``KNOWN_INTEGRATION_HOOKS``), and
      2. uninstall REVOKES the minted key.

Manifest config::

    "integration": {
      "augmentum_backend": {
        "base_url_env": ["OPENAI_API_BASE_URL", ...],   # <- http://augmentum:6100/v1
        "api_key_env":  ["OPENAI_API_KEY", ...]          # <- the minted sk-aug- key
      }
    }

Trust boundary: the minted key is ``chat``-scoped (never admin), unique per
install, lives in the service's container env + the service's own DB, and is
revoked on uninstall. The service can call Augmentum's chat / embeddings /
audio / image endpoints *as this user*, nothing more.
"""

from __future__ import annotations

from typing import Any

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Internal base URL a provisioned service uses to reach Augmentum's
# OpenAI-compatible API. Services join the shared ``augmentum_default``
# network (providers/network.py::ensure_network), where the app container is
# reachable by its compose alias ``augmentum`` on port 6100.
AUGMENTUM_INTERNAL_BASE_URL = "http://augmentum:6100/v1"

# config_json key under which the install path stashes the minted key id so
# _uninstall can revoke it.
CONFIG_KEY_ID = "augmentum_backend_key_id"


async def _install(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """No-op: the key mint + env injection already ran PRE-provision (env is
    baked at container create; a post-install hook is too late). Present so the
    hook is registered — which is what makes parse_manifest keep the config
    block that the pre-provision step reads."""
    log.info("augmentum_backend_ready", service_id=getattr(manifest, "service_id", ""))


async def _uninstall(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Revoke the API key minted for this service at install time."""
    mgr = getattr(request.app.state, "service_manager", None)
    akm = getattr(request.app.state, "api_key_manager", None)
    if mgr is None or akm is None:
        return
    service_id = getattr(manifest, "service_id", "")
    try:
        cfg = await mgr.read_config_json(service_id)
    except Exception:  # noqa: BLE001 — best-effort cleanup
        cfg = {}
    key_id = (cfg or {}).get(CONFIG_KEY_ID)
    if not key_id:
        return
    try:
        await akm.revoke(str(key_id), user_id)
        log.info("augmentum_backend_key_revoked", service_id=service_id, key_id=key_id)
    except Exception:  # noqa: BLE001
        log.warning(
            "augmentum_backend_revoke_failed", service_id=service_id, exc_info=True,
        )


KNOWN_INTEGRATION_HOOKS["augmentum_backend"] = (
    _install,
    _uninstall,
    HookMeta(
        label="Augmentum AI",
        icon="🧠",
        companion_hint="Uses your Augmentum models and voices — no setup",
        status_provider="augmentum_backend",
        # Install-time env wiring — no runtime on/off. Show it as an
        # informational capability row, not a (broken) toggle.
        toggleable=False,
    ),
)
