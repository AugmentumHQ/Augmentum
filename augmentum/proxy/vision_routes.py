"""Vision provider status + control routes.

`/api/vision/status` returns the current availability of both
providers (primary and SmolVLM sibling) plus the sibling's lifecycle
state. Used by the settings UI to surface "vision is ready" / "sibling
starting up" / "model files missing" diagnostics.

`/api/vision/restart` is a manual sibling restart — useful when the
operator has changed model paths or wants to recover from a wedge
without restarting the whole container.

The provider abstraction itself lives in :mod:`augmentum.vision`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])


class VisionStatus(BaseModel):
    """Snapshot of the vision substrate."""

    enabled: bool                       # vision_provider_enabled setting
    has_router: bool                    # router instantiated successfully
    primary_available: bool             # primary model is VL-capable + ready
    smolvlm_available: bool             # sibling subprocess is ready
    sibling_port: int | None            # llama-server port for sibling
    sibling_state: str | None           # READY / STARTING / STOPPED / IDLE / etc.
    base_url: str | None                # sibling's OpenAI-compat base URL when up


@router.get("/status", response_model=VisionStatus)
async def vision_status(request: Request) -> VisionStatus:
    """Return current vision provider availability."""
    from augmentum.config import settings

    router_obj: Any = getattr(request.app.state, "vision_router", None)
    sibling: Any = getattr(request.app.state, "vision_sibling", None)

    if router_obj is None:
        return VisionStatus(
            enabled=bool(settings.vision_provider_enabled),
            has_router=False,
            primary_available=False,
            smolvlm_available=False,
            sibling_port=None,
            sibling_state=None,
            base_url=None,
        )

    primary_avail = False
    smolvlm_avail = False
    try:
        primary = router_obj.primary_provider
        if primary is not None:
            primary_avail = await primary.is_available()
    except Exception:
        primary_avail = False
    try:
        smolvlm = router_obj.smolvlm_provider
        if smolvlm is not None:
            smolvlm_avail = await smolvlm.is_available()
    except Exception:
        smolvlm_avail = False

    sibling_port: int | None = None
    sibling_state: str | None = None
    base_url: str | None = None
    if sibling is not None:
        sibling_port = sibling.config.backend_port
        mgr = sibling.manager
        if mgr is not None:
            sibling_state = mgr.state.name
            base_url = mgr.base_url
        else:
            sibling_state = "STOPPED"

    return VisionStatus(
        enabled=bool(settings.vision_provider_enabled),
        has_router=True,
        primary_available=primary_avail,
        smolvlm_available=smolvlm_avail,
        sibling_port=sibling_port,
        sibling_state=sibling_state,
        base_url=base_url,
    )


@router.post("/restart")
async def vision_restart(request: Request) -> dict[str, Any]:
    """Apply the current ``vision_provider_*`` settings to the running
    process. Reactive: this is the path that makes the master toggle
    actually do something without a server restart.

    Behavior matrix based on live ``settings.vision_provider_enabled`` +
    any existing sibling on ``app.state``:

        enabled  sibling   action
        ───────  ───────   ─────────────────────────────────────────
        True     None      Build sibling + provider; attach to router
        True     present   Stop old; rebuild from current settings;
                           attach (handles port / GPU / path changes)
        False    present   Stop sibling; detach provider from router
        False    None      No-op
    """
    from augmentum.config import settings
    from augmentum.vision import SmolVLMConfig, SmolVLMProvider, SmolVLMSibling

    app = request.app
    router_obj = getattr(app.state, "vision_router", None)
    if router_obj is None:
        raise HTTPException(
            status_code=503,
            detail="Vision router not initialized "
                   "(server still starting or init failed).",
        )

    old_sibling = getattr(app.state, "vision_sibling", None)
    if old_sibling is not None:
        try:
            await old_sibling.stop()
        except Exception as exc:
            log.warning("vision_restart_stop_failed", error=str(exc)[:200])

    if not settings.vision_provider_enabled:
        app.state.vision_sibling = None
        router_obj.set_smolvlm(None)
        log.info("vision_restart_disabled")
        return {"running": False, "enabled": False}

    cfg = SmolVLMConfig(
        base_model_path=settings.vision_provider_model_path,
        mmproj_path=settings.vision_provider_mmproj_path,
        backend_port=settings.vision_provider_backend_port,
        # CPU-only fallback by definition (retired the GPU vision sibling —
        # GPU vision is the classifier slot's job). gpu_layers stays 0.
    )
    new_sibling = SmolVLMSibling(cfg)
    ok = False
    try:
        ok = await new_sibling.start()
    except Exception as exc:
        log.warning("vision_restart_start_failed", error=str(exc)[:200])

    if not ok:
        app.state.vision_sibling = None
        router_obj.set_smolvlm(None)
        log.warning("vision_restart_start_returned_false")
        return {"running": False, "enabled": True}

    app.state.vision_sibling = new_sibling
    http_client = getattr(app.state, "http_client", None)
    if http_client is None:
        log.warning("vision_restart_no_http_client")
        router_obj.set_smolvlm(None)
        return {"running": True, "enabled": True, "warning": "no http_client"}

    router_obj.set_smolvlm(SmolVLMProvider(new_sibling, http_client))
    log.info("vision_restart_started", port=cfg.backend_port, gpu_layers=cfg.gpu_layers)
    return {"running": True, "enabled": True}
