"""Marketplace API — browse catalog, enable/disable/manage provider services."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.auth.guards import require_admin
from augmentum.providers.models import (
    GpuRequirements,
    HealthCheck,
    ServiceCategory,
    ServiceDefinition,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/marketplace", tags=["marketplace"])

# Models being pulled in the background (key: "service_id:model"), so repeated
# provider-status polls don't kick off duplicate downloads while one is running.
_MODEL_PULLS_INFLIGHT: set[str] = set()


def _start_model_pull(client, service_id: str, base_url: str,
                      model: str, manifest: dict) -> bool:
    """Fire a de-duped background model pull via the provider's manifest.

    The pull blocks the provider until the download finishes (minutes for a
    large model), so it runs as a detached task; the in-flight set keeps
    repeated polls / clicks from launching duplicates. Returns True iff a new
    pull was started (False = already running, or missing config)."""
    endpoint = (manifest or {}).get("endpoint", "")
    model = (model or "").strip()
    if not (client is not None and base_url and model and endpoint):
        return False
    key = f"{service_id}:{model}"
    if key in _MODEL_PULLS_INFLIGHT:
        return False
    _MODEL_PULLS_INFLIGHT.add(key)
    url = base_url.rstrip("/") + endpoint.replace("{model}", model)
    method = (manifest.get("method") or "POST").upper()

    async def _run() -> None:
        try:
            await client.request(method, url, timeout=900.0)
            log.info("provider_model_pulled", service=service_id, model=model)
        except Exception:
            log.warning("provider_model_pull_failed",
                        service=service_id, model=model, exc_info=True)
        finally:
            _MODEL_PULLS_INFLIGHT.discard(key)

    asyncio.create_task(_run())
    log.info("provider_model_pull_started", service=service_id, model=model)
    return True


async def _fetch_model_ids(client, url: str, cap: int = 0) -> list[str]:
    """GET an OpenAI-style model/registry list and return the model ids."""
    try:
        resp = await client.get(url, timeout=6.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        rows = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return []
        ids = [str(m.get("id") or m.get("model_id") or "")
               for m in rows if isinstance(m, dict)]
        ids = [i for i in ids if i]
        return ids[:cap] if cap else ids
    except Exception:
        return []


def _mgr(request: Request):
    mgr = getattr(request.app.state, "service_manager", None)
    if not mgr:
        raise RuntimeError("Service manager not available")
    return mgr


@router.get("/catalog")
async def list_catalog(request: Request, category: str | None = None):
    """List available provider services from the catalog."""
    mgr = _mgr(request)
    if category:
        try:
            cat = ServiceCategory(category)
        except ValueError:
            return JSONResponse({"error": f"Invalid category: {category}"}, 400)
        entries = mgr.catalog.list_by_category(cat)
    else:
        entries = mgr.catalog.list_all()

    managed = {s.id: s for s in await mgr.list_managed()}

    result = []
    for entry in entries:
        ms = managed.get(entry.id)
        result.append({
            "id": entry.id,
            "name": entry.name,
            "description": entry.description,
            "category": entry.category.value,
            "image": entry.image,
            "host_port": entry.host_port,
            "gpu": {"required": entry.gpu.required, "vram_mb": entry.gpu.vram_mb},
            "api_type": entry.api_type,
            "features": entry.features,
            "enabled": ms.enabled if ms else False,
            "status": ms.status if ms else "stopped",
        })
    return result


@router.get("/services")
async def list_services(request: Request):
    """List all managed (enabled) services with live status."""
    mgr = _mgr(request)
    services = await mgr.list_managed()
    return [
        {
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "image": s.image,
            "host_port": s.host_port,
            "enabled": s.enabled,
            "status": s.status,
            "error": s.error,
        }
        for s in services
    ]


@router.post("/services/{service_id}/enable")
async def enable_service(request: Request, service_id: str):
    """Enable a catalog service. Admin only — starts a shared container."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    try:
        ms = await mgr.enable_service(service_id)
        # Register the running service as a provider (+ hot-load) so it's
        # immediately usable, and hand the next-steps back for the UI card.
        provider = None
        try:
            from augmentum.providers.provider_bridge import (
                register_installed_service_provider,
            )
            reg = await register_installed_service_provider(
                request.app.state, service_id,
            )
            provider = reg.to_dict() if reg is not None else None
        except Exception:
            log.warning("provider_bridge_enable_failed",
                        service=service_id, exc_info=True)
        return {
            "id": ms.id,
            "name": ms.name,
            "status": ms.status,
            "host_port": ms.host_port,
            "container_id": ms.container_id[:12] if ms.container_id else None,
            "provider": provider,
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, 400)
    except Exception as exc:
        log.error("enable_service_failed", service=service_id, error=str(exc))
        return JSONResponse({"error": f"Failed to enable service: {exc}"}, 500)


@router.get("/services/{service_id}/provider-status")
async def provider_status(request: Request, service_id: str):
    """Post-install readiness + next-steps for the UI card.

    Reads the registered provider state, live-probes the service's health
    endpoint (short timeout), and returns the concrete next steps a user
    who just pressed Install wants. Pollable while a model downloads.

    Admin-only: it self-heals registration (writes the provider row /
    settings / hot-registers a backend), the same privileged side effect
    as ``enable``.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    if mgr is None or not hasattr(mgr, "get_definition"):
        return JSONResponse({"error": "Service manager unavailable"}, 503)
    sd = mgr.get_definition(service_id)
    if sd is None:
        return JSONResponse({"error": f"Unknown service: {service_id}"}, 404)

    from augmentum.providers.provider_bridge import (
        extract_default_model,
        register_provider_for_service,
        resolve_service_url,
    )

    # Re-run registration (idempotent upsert) so a service that was
    # enabled before the bridge existed, or whose row was lost, self-heals
    # on first status check.
    conn = None
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    conn = getattr(backend, "conn", None)
    reg = await register_provider_for_service(
        sd, conn=conn,
        settings_store=getattr(request.app.state, "settings_store", None),
        registry=getattr(request.app.state, "provider_registry", None),
        http_client=getattr(request.app.state, "http_client", None),
    )

    # Live reachability + model probe (best-effort, short timeout).
    _settings_key, base_url = resolve_service_url(sd)
    reachable = False
    models: list[str] = []
    probe_detail = ""
    if base_url:
        client = getattr(request.app.state, "http_client", None)
        try:
            if client is not None:
                health_url = base_url.rstrip("/") + (sd.health_endpoint or "/health")
                resp = await client.get(health_url, timeout=4.0)
                reachable = resp.status_code < 500
        except Exception as exc:
            probe_detail = f"not reachable yet: {type(exc).__name__}"
        # Try to list models (OpenAI-compatible services expose /v1/models).
        if reachable and client is not None:
            try:
                mresp = await client.get(base_url.rstrip("/") + "/v1/models", timeout=4.0)
                if mresp.status_code == 200:
                    data = mresp.json()
                    rows = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(rows, list):
                        models = [str(m.get("id") or m.get("model_id") or "")
                                  for m in rows if isinstance(m, dict)][:25]
            except Exception:
                pass

    # Durable model pull. Some providers (speaches) ignore PRELOAD_MODELS on
    # their published Docker image (Issue #77) → they launch healthy with ZERO
    # models, so nothing is usable. When the catalog declares a ``model_pull``
    # manifest and the service is up-but-empty, pull the default model in the
    # background via its real download endpoint. This is what makes the
    # provider actually work post-install without manual steps. Idempotent +
    # de-duped: a later poll sees the model appear and stops re-pulling.
    pulling = ""
    if reachable and client is not None and not models and sd.model_pull.get("endpoint"):
        pull_model = (sd.default_model or extract_default_model(sd)).strip()
        if pull_model:
            _start_model_pull(client, service_id, base_url, pull_model, sd.model_pull)
            if f"{service_id}:{pull_model}" in _MODEL_PULLS_INFLIGHT:
                pulling = pull_model

    out = reg.to_dict()
    out.update({
        "reachable": reachable,
        "models": models,
        "model_count": len(models),
        "expected_model": extract_default_model(sd),
        "pulling_model": pulling,
        "webui": f"http://localhost:{sd.host_port}" if sd.host_port else "",
        "probe_detail": probe_detail,
    })
    # Refresh next-steps in light of the live probe (downloading vs ready).
    if pulling:
        out["next_steps"] = [
            {"label": f"Downloading {pulling}…", "action": "wait",
             "detail": "Pulling the model so it's ready to use — this finishes "
                       "in the background; keep this open.", "url": ""},
            *out["next_steps"],
        ]
    elif reg.registered and not reachable:
        out["next_steps"] = [
            {"label": "Service starting…", "action": "wait",
             "detail": probe_detail or "container is booting / pulling the model", "url": ""},
            *out["next_steps"],
        ]
    return out


@router.get("/services/{service_id}/models")
async def list_service_models(request: Request, service_id: str):
    """List a provider's installed + pullable models for the manifest picker.

    Reads the catalog ``model_pull`` manifest and live-queries the running
    service for what's loaded (``list_installed``) and what can be pulled
    (``list_available``). Admin-only (same privilege as install)."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    if mgr is None or not hasattr(mgr, "get_definition"):
        return JSONResponse({"error": "Service manager unavailable"}, 503)
    sd = mgr.get_definition(service_id)
    if sd is None:
        return JSONResponse({"error": f"Unknown service: {service_id}"}, 404)

    manifest = sd.model_pull or {}
    from augmentum.providers.provider_bridge import (
        extract_default_model,
        resolve_service_url,
    )
    if not manifest.get("endpoint"):
        # Provider doesn't pull models (bundled/auto) — picker not applicable.
        return {"supported": False, "installed": [], "available": [],
                "default_model": (sd.default_model or extract_default_model(sd)),
                "pulling": []}

    _key, base_url = resolve_service_url(sd)
    client = getattr(request.app.state, "http_client", None)
    installed: list[str] = []
    available: list[str] = []
    if base_url and client is not None:
        if manifest.get("list_installed"):
            installed = await _fetch_model_ids(
                client, base_url.rstrip("/") + manifest["list_installed"])
        if manifest.get("list_available"):
            # Registries can be large (speaches has ~440) — cap for the UI.
            available = await _fetch_model_ids(
                client, base_url.rstrip("/") + manifest["list_available"], cap=120)
    pulling = [k.split(":", 1)[1] for k in _MODEL_PULLS_INFLIGHT
               if k.startswith(f"{service_id}:")]
    return {
        "supported": True,
        "installed": installed,
        "available": available,
        "default_model": (sd.default_model or extract_default_model(sd)),
        "pulling": pulling,
    }


@router.post("/services/{service_id}/pull-model")
async def pull_service_model(request: Request, service_id: str):
    """Pull a specific model into a running provider (manifest picker). Admin."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    if mgr is None or not hasattr(mgr, "get_definition"):
        return JSONResponse({"error": "Service manager unavailable"}, 503)
    sd = mgr.get_definition(service_id)
    if sd is None:
        return JSONResponse({"error": f"Unknown service: {service_id}"}, 404)
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "error": "Missing model"}, 400)
    if not sd.model_pull.get("endpoint"):
        return JSONResponse(
            {"ok": False, "error": "Provider does not support pulling models"}, 400)
    from augmentum.providers.provider_bridge import resolve_service_url
    _key, base_url = resolve_service_url(sd)
    client = getattr(request.app.state, "http_client", None)
    if client is None or not base_url:
        return JSONResponse({"ok": False, "error": "Service not reachable"}, 503)
    _start_model_pull(client, service_id, base_url, model, sd.model_pull)
    return {"ok": True, "pulling": model,
            "in_flight": f"{service_id}:{model}" in _MODEL_PULLS_INFLIGHT}


@router.post("/services/{service_id}/disable")
async def disable_service(request: Request, service_id: str):
    """Disable a service. Admin only — affects all tenants."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    try:
        await mgr.disable_service(service_id)
        return {"id": service_id, "status": "stopped"}
    except Exception as exc:
        log.error("disable_service_failed", service=service_id, error=str(exc))
        return JSONResponse({"error": f"Failed to disable service: {exc}"}, 500)


@router.get("/services/{service_id}/status")
async def service_status(request: Request, service_id: str):
    """Get live status of a managed service."""
    mgr = _mgr(request)
    status = await mgr.get_status(service_id)
    return {"id": service_id, "status": status.value}


@router.post("/services/{service_id}/logs")
async def service_logs(request: Request, service_id: str, tail: int = 100):
    """Get recent logs from a managed service container. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _mgr(request)
    container = await mgr._find_container(service_id)
    if not container:
        return JSONResponse({"error": "Container not found"}, 404)
    try:
        logs = await container.log(stdout=True, stderr=True, tail=tail)
        return {"logs": "".join(logs)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


@router.post("/services/custom")
async def create_custom_service(request: Request):
    """Create a custom provider service from user-supplied Docker config.

    Admin only — Docker config can mount volumes, expose ports, and set
    env vars install-wide.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    body = await request.json()
    mgr = _mgr(request)

    service_id = f"custom-{uuid.uuid4().hex[:8]}"
    try:
        gpu_data = body.get("gpu", {})
        hc_data = body.get("health_check")
        sd = ServiceDefinition(
            id=service_id,
            name=body["name"],
            description=body.get("description", "Custom provider"),
            category=ServiceCategory(body["category"]),
            image=body["image"],
            internal_port=body["internal_port"],
            host_port=body["host_port"],
            env=body.get("env", {}),
            volumes=body.get("volumes", {}),
            health_check=HealthCheck(**hc_data) if hc_data else None,
            gpu=GpuRequirements(**gpu_data) if gpu_data else GpuRequirements(),
            api_type=body.get("api_type", "openai_llm"),
            health_endpoint=body.get("health_endpoint", "/health"),
            command=body.get("command"),
            is_custom=True,
            mem_limit=body.get("mem_limit", ""),
            min_ram_mb=int(body.get("ram_mb") or 0),
            augmentum_env=body.get("augmentum_env", {}),
        )
    except (KeyError, ValueError) as exc:
        return JSONResponse({"error": f"Invalid config: {exc}"}, 400)

    # Inject into catalog and enable
    mgr.catalog._entries.append(sd)
    mgr.catalog._by_id[sd.id] = sd

    try:
        ms = await mgr.enable_service(service_id)
        return {
            "id": ms.id,
            "name": ms.name,
            "status": ms.status,
            "host_port": ms.host_port,
        }
    except Exception as exc:
        mgr.catalog._entries = [e for e in mgr.catalog._entries if e.id != service_id]
        mgr.catalog._by_id.pop(service_id, None)
        return JSONResponse({"error": str(exc)}, 500)


def _detect_hardware_sync(docker_available: bool) -> dict:
    """Hardware probe — sync because every method blocks (torch CUDA
    initialization, nvidia-smi subprocess, onnxruntime import). Caller
    wraps in ``asyncio.to_thread`` so the event loop keeps serving.
    """
    result = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_mb": 0,
        "docker_available": docker_available,
    }

    # Method 1: torch.cuda (most detailed)
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return result
    except (ImportError, RuntimeError) as exc:
        log.debug("marketplace_hw_torch_probe_failed", error=str(exc))

    # Method 2: nvidia-smi (works when torch CUDA has driver mismatch)
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result["gpu_available"] = True
                result["gpu_name"] = parts[0]
                try:
                    result["gpu_vram_mb"] = int(parts[1])
                except ValueError:
                    pass
                return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: onnxruntime CUDA provider (lightweight check)
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            result["gpu_available"] = True
            result["gpu_name"] = "NVIDIA GPU (via ONNX Runtime)"
    except ImportError as exc:
        log.debug("marketplace_hw_onnx_probe_failed", error=str(exc))

    return result


@router.get("/hardware")
async def detect_hardware(request: Request):
    """Detect GPU and system hardware for compatibility badges.

    All probe paths block (torch CUDA init, subprocess, heavy onnxruntime
    import). Hand off to a worker thread; the handler just orchestrates.
    """
    mgr = getattr(request.app.state, "service_manager", None)
    return await asyncio.to_thread(_detect_hardware_sync, mgr is not None)
