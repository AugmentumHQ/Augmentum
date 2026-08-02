"""REST API routes for resource monitoring."""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.resource.host_probe import probe_host_stats
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/resources", tags=["resources"])


def _get_ledger(request: Request):
    return getattr(request.app.state, "resource_ledger", None)


def _system_scope() -> str:
    """Return whether CPU/RAM readings reflect the host or the runtime container."""
    override = os.environ.get("AUGMENTUM_RESOURCE_SCOPE", "").strip().lower()
    if override in {"host", "runtime"}:
        return override
    return "runtime" if os.path.exists("/.dockerenv") else "host"


@router.get("/status")
async def resource_status(request: Request) -> JSONResponse:
    """Current resource state -- polls all backends live."""
    ledger = _get_ledger(request)
    if not ledger:
        return JSONResponse({"error": "Resource ledger not available"}, status_code=503)

    # Record panel access so the background sampler (resource/sampler.py) can
    # sample fast while someone's watching and slow when idle. Cheap wall-clock
    # stamp; the sampler reads it.
    try:
        import time as _t
        request.app.state.resource_panel_last_access = _t.monotonic()
    except Exception:  # pragma: no cover — never block /status on this
        pass

    # Optional bypass: ?fresh=1 forces a fresh ledger.collect() so the
    # popover's manual refresh button can override the cache TTL.
    fresh_q = (request.query_params.get("fresh") or "").strip().lower()
    force = fresh_q in ("1", "true", "yes")
    # Read path is cache-only by default: the background sampler
    # (resource/sampler.py) keeps every probe cache warm, so a normal poll
    # assembles from cache in dict-read time and NEVER awaits a live
    # nvidia-smi / docker-stats / host-agent probe (the ~2s slow_request
    # class, measured: container_probe cold = ~2s). ``?fresh=1`` opts the
    # popover's manual refresh into the live path.
    snap = await ledger.collect(force=force, cache_only=not force)

    # CPU utilization — interval=None returns the percentage since the
    # *last* call, which means a background sampler keeps this current
    # without blocking the event loop. The 100ms-block interval=0.1
    # path used previously cost ~100ms per /status hit; the rolling
    # since-last-call value tracks the same shape with zero latency.
    # First call after process start returns 0.0 (no prior sample),
    # which the UI renders as "—" — acceptable startup race.
    cpu_pct = 0.0
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
    except (ImportError, OSError) as exc:
        # psutil unavailable or /proc unreadable — cpu_pct stays 0 and
        # the UI shows "—" for that field.
        log.debug("resource_status_cpu_pct_failed", error=str(exc))
    system_scope = _system_scope()

    # Host-level RAM/CPU via the optional host stats agent. When Augmentum
    # runs in a container the readings above are the container's (on Docker
    # Desktop: the WSL2/Linux VM's) view, which doesn't match the host OS's
    # Task Manager. If scripts/host_stats_agent.py is running on the host we
    # surface its numbers alongside; otherwise host.available is False and
    # the UI shows the container view only.
    host_block: dict = {"available": False}
    host = None
    http_client = getattr(request.app.state, "http_client", None)
    if http_client is not None:
        try:
            host = await probe_host_stats(http_client, cache_only=not force)
        except Exception:
            host = None
    if host is not None:
        host_block = {
            "available": True,
            "source": "agent",
            "hostname": host.hostname,
            "os": host.os_name,
            "cpu_pct": round(host.cpu_pct, 1),
            "cpu_count": host.cpu_count,
            "ram": {
                "total_mb": host.ram_total_mb,
                "used_mb": host.ram_used_mb,
                "free_mb": host.ram_free_mb,
            },
        }

    # Ledger-tracked models (engine, image, in-process). Carry container=""
    # / controllable=False so the panel can treat every entry uniformly.
    # ``confidence`` (measured/declared/estimated) + ``as_of`` make the figures
    # honest in the UI (spec §4.6); the snapshot's collect time is the as_of for
    # every ledger model.
    try:
        snap_as_of = snap.timestamp.timestamp()
    except (AttributeError, OSError, ValueError):
        snap_as_of = 0.0
    models_out = [
        {
            "name": m.name,
            "subsystem": m.subsystem,
            "backend": m.backend,
            "device": m.device,
            "vram_mb": m.vram_mb,
            "ram_mb": m.ram_mb,
            "cpu_pct": None,
            "quantization": m.quantization,
            "parameter_size": m.parameter_size,
            "family": m.family,
            "pipeline_type": m.pipeline_type,
            "expires_at": m.expires_at,
            "status": m.status,
            "container": "",
            "controllable": False,
            "kind": "model",
            "confidence": getattr(m, "confidence", "measured"),
            "as_of": snap_as_of,
        }
        for m in snap.models
    ]
    # Sidecar containers (TTS / STT / classifier / vision) — GPU consumers the
    # ledger doesn't see. Appended so they surface with owner + pause/reload.
    sidecar_vram_mb = 0
    try:
        from augmentum.resource.container_probe import probe_sidecar_containers
        _known = {m["name"] for m in models_out}
        for s in await probe_sidecar_containers(request.app.state, cache_only=not force):
            if s["name"] not in _known:
                models_out.append(s)
                sidecar_vram_mb += int(s.get("vram_mb") or 0)
    except Exception:
        log.warning("resource_status_sidecar_probe_failed", exc_info=True)

    # Reconciliation residual (spec §4.6): the ledger computes unattributed VRAM
    # as device-used minus the nvidia-smi per-process sum. On WSL2 that per-
    # process list is empty (per-process VRAM is opaque there), so sidecar VRAM
    # we measured from the siblings' llama-server logs is sitting INSIDE that
    # residual. Subtract it back out so "shared/driver" shrinks to what's truly
    # unattributed. Skip when gpu_processes IS populated (native systems already
    # count the sidecar process — subtracting would double-count).
    unattributed_vram_mb = snap.unattributed_vram_mb
    if not snap.gpu_processes and sidecar_vram_mb:
        unattributed_vram_mb = max(0, unattributed_vram_mb - sidecar_vram_mb)

    return JSONResponse({
        "cpu_pct": round(cpu_pct, 1),
        "cpu_scope": system_scope,
        "host": host_block,
        "gpu": {
            "name": snap.gpu_name,
            "total_mb": snap.gpu_total_mb,
            "used_mb": snap.gpu_used_mb,
            "free_mb": snap.gpu_free_mb,
            "scope": "host" if snap.gpu_total_mb else "unknown",
        },
        "ram": {
            "total_mb": snap.ram_total_mb,
            "used_mb": snap.ram_used_mb,
            "free_mb": snap.ram_free_mb,
            "scope": system_scope,
        },
        "models": models_out,
        "gpu_processes": [
            {
                "pid": p.pid,
                "name": p.name,
                "vram_mb": p.vram_mb,
                "label": p.label,
            }
            for p in snap.gpu_processes
        ],
        "unattributed_vram_mb": unattributed_vram_mb,
        # Sprint A additions — transient inventory + disk + active jobs.
        # None of these triggered DB writes during the snapshot collect
        # (see § DB write budget in the design doc); they're surfaces
        # over RAM-cached state.
        "disk_destinations": [
            {
                "dir": d.dir,
                "modality": d.modality,
                "free_bytes": d.free_bytes,
                "total_bytes": d.total_bytes,
                "error": d.error,
            }
            for d in snap.disk_destinations
        ],
        "active_jobs": [
            {
                "job_id": j.job_id,
                "user_id": j.user_id,
                "kind": j.kind,
                "target_id": j.target_id,
                "progress_pct": j.progress_pct,
                "stage": j.stage,
                "started_at": j.started_at,
            }
            for j in snap.active_jobs
        ],
        "inventory": [
            {
                "name": e.name,
                "modality": e.modality,
                "backend": e.backend,
                "size_bytes": e.size_bytes,
                "location": e.location,
                "loaded": e.loaded,
                "capable": e.capable,
                "metadata": e.metadata,
            }
            for e in snap.inventory
        ],
        "inventory_etag": snap.inventory_etag,
    })


@router.get("/profiles")
async def model_profiles(request: Request) -> JSONResponse:
    """All known model profiles (learned from past observations)."""
    ledger = _get_ledger(request)
    if not ledger:
        return JSONResponse({"profiles": []})

    profiles = await ledger.list_profiles()
    return JSONResponse({"profiles": [asdict(p) for p in profiles]})


@router.get("/history")
async def resource_history(
    request: Request, hours: int = 24, limit: int = 100,
) -> JSONResponse:
    """GPU/RAM usage over time."""
    ledger = _get_ledger(request)
    if not ledger:
        return JSONResponse({"snapshots": []})

    history = await ledger.get_history(hours=hours, limit=limit)
    return JSONResponse({"snapshots": [
        {
            "timestamp": s.timestamp.isoformat(),
            "gpu_used_mb": s.gpu_used_mb,
            "gpu_free_mb": s.gpu_free_mb,
            "ram_used_mb": s.ram_used_mb,
            "ram_free_mb": s.ram_free_mb,
            "model_count": len(s.models),
        }
        for s in history
    ]})


@router.get("/check/{model_name:path}")
async def check_model_fit(request: Request, model_name: str) -> JSONResponse:
    """Check if a model can fit in available VRAM."""
    ledger = _get_ledger(request)
    if not ledger:
        return JSONResponse({"can_fit": True, "estimated_vram_mb": 0, "note": "No ledger available"})

    can_fit, vram_mb = await ledger.can_fit_model(model_name)
    snap = ledger.last_snapshot
    return JSONResponse({
        "can_fit": can_fit,
        "estimated_vram_mb": vram_mb,
        "gpu_free_mb": snap.gpu_free_mb if snap else 0,
    })


@router.post("/unload")
async def unload_model(request: Request) -> JSONResponse:
    """Unload a model from VRAM.

    Supports three backends:
      - ollama:   POST /api/chat with keep_alive=0
      - llamacpp: router-mode unload
      - lmstudio: POST /api/v1/models/unload with instance_id
    """
    body = await request.json()
    name = body.get("name", "").strip()
    backend = body.get("backend", "").strip().lower()

    if not name:
        return JSONResponse({"ok": False, "error": "Missing model name"}, status_code=400)

    # --- LM Studio: dedicated unload endpoint ---
    if backend in ("lm studio", "lmstudio"):
        return await _unload_lmstudio(request, name)

    # --- Augmentum Engine: forward to engine unload ---
    if backend == "engine":
        # Both the primary engine AND the secondary slot ("Slot B") surface
        # with backend="engine" in the panel, so disambiguate by model id:
        # if the requested model is the one resident in Slot B, unload THAT
        # slot — otherwise hitting X would stop the primary and leave Slot B
        # holding its VRAM (the "not getting reclaimed" symptom).
        secondary = getattr(request.app.state, "secondary_slot", None)
        sec_mgr = getattr(secondary, "manager", None) if secondary else None
        registry = getattr(request.app.state, "provider_registry", None)
        if sec_mgr is not None and getattr(sec_mgr, "model_id", "") == name:
            try:
                await secondary.unload()
                if registry:
                    registry.unpin_model(name)
                    registry.invalidate_model_map()
                object.__setattr__(settings, "engine_secondary_model", "")
                store = getattr(request.app.state, "settings_store", None)
                if store:
                    await store.set("engine_secondary_model", "")
                from augmentum.resource.ledger import invalidate as _invalidate_resource
                _invalidate_resource(request.app.state, "llm")
                log.info("engine_secondary_unloaded_via_resources", model=name)
                return JSONResponse({"ok": True})
            except Exception:
                log.warning("engine_secondary_unload_failed", model=name, exc_info=True)
                return JSONResponse({"ok": False, "error": "Slot B unload failed"}, status_code=500)

        # The managed classifier slot ("Slot C") ALSO surfaces with
        # backend="engine" now that it's tracked in the resource snapshot.
        # Same disambiguation as Slot B — match by resident model id and route
        # the unload to the slot, else X would stop the primary and leave the
        # resident classifier holding VRAM. Delegate to the canonical unload so
        # its extra cleanup (external-sidecar restore, backend-key removal,
        # settings clear) isn't duplicated/drift-prone.
        classifier = getattr(request.app.state, "classifier_slot", None)
        cls_mgr = getattr(classifier, "manager", None) if classifier else None
        if cls_mgr is not None and getattr(cls_mgr, "model_id", "") == name:
            try:
                from augmentum.proxy.model_routes import engine_classifier_unload

                await engine_classifier_unload(request)
                log.info("classifier_slot_unloaded_via_resources", model=name)
                return JSONResponse({"ok": True})
            except Exception:
                log.warning("classifier_slot_unload_failed", model=name, exc_info=True)
                return JSONResponse({"ok": False, "error": "Slot C unload failed"}, status_code=500)

        mgr = getattr(request.app.state, "llama_manager", None)
        if mgr:
            try:
                await mgr.stop()
                registry = getattr(request.app.state, "provider_registry", None)
                if registry:
                    registry.invalidate_model_map()
                log.info("engine_model_unloaded_via_resources", model=name, path="manager")
                return JSONResponse({"ok": True})
            except Exception:
                log.warning("engine_unload_failed", model=name, path="manager", exc_info=True)
                return JSONResponse({"ok": False, "error": "Engine unload failed"}, status_code=500)

        registry = getattr(request.app.state, "provider_registry", None)
        if registry:
            engine_backend = registry.get_backend("engine")
            if engine_backend and hasattr(engine_backend, "unload_model"):
                try:
                    success = await engine_backend.unload_model(name)
                    if success:
                        registry.invalidate_model_map()
                        log.info("engine_model_unloaded_via_resources", model=name, path="backend")
                    return JSONResponse({"ok": success})
                except Exception:
                    log.warning("engine_unload_failed", model=name, path="backend", exc_info=True)
                    return JSONResponse({"ok": False, "error": "Engine unload failed"}, status_code=500)
        return JSONResponse({"ok": False, "error": "Engine backend not available"}, status_code=503)

    # --- Ollama / llama.cpp: use ModelManager ---
    mm = getattr(request.app.state, "model_manager", None)
    if not mm:
        return JSONResponse({"ok": False, "error": "Model manager not available"}, status_code=503)

    backend_key = None
    if backend in ("llamacpp", "llama.cpp"):
        backend_key = "llamacpp"
    elif backend == "ollama":
        backend_key = "ollama"

    try:
        success = await mm.unload_model(name, backend_key=backend_key)
    except Exception:
        log.warning("unload_failed", model=name, backend=backend, exc_info=True)
        return JSONResponse({"ok": False, "error": "Unload request failed"}, status_code=500)

    if success:
        log.info("model_unloaded", model=name, backend=backend)
    return JSONResponse({"ok": success})


@router.get("/reclaim/preview")
async def reclaim_preview(request: Request) -> JSONResponse:
    """What a manual reclaim would free, without freeing anything.

    Separate from the execute call on purpose: the point of the manual phase
    is that the user sees the price (and what is refused, and why) before
    anything is unloaded. See resource/reclaim.py and spec §7.1.
    """
    from augmentum.resource import reclaim

    try:
        return JSONResponse(await reclaim.preview(request.app.state))
    except Exception:
        log.warning("reclaim_preview_failed", exc_info=True)
        return JSONResponse(
            {"error": "Could not build a reclaim plan"}, status_code=500
        )


@router.post("/reclaim")
async def reclaim_run(request: Request) -> JSONResponse:
    """Reclaim unused-but-held memory and report the MEASURED delta.

    Body: ``{"keys": ["slot_b", "allocator"]}`` — omit ``keys`` to take
    everything currently reclaimable. Candidates are re-checked at execute
    time, so a slot that went busy since the preview is skipped, not evicted.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    keys = body.get("keys")
    if keys is not None and not isinstance(keys, list):
        return JSONResponse({"ok": False, "error": "keys must be a list"}, status_code=400)

    from augmentum.resource import reclaim

    try:
        result = await reclaim.run(request.app.state, keys=keys)
    except Exception:
        log.warning("reclaim_run_failed", exc_info=True)
        return JSONResponse({"ok": False, "error": "Reclaim failed"}, status_code=500)

    # Any slot we stopped is no longer resident; drop the cached snapshot so
    # the panel's next poll shows the new reality rather than the old numbers.
    try:
        from augmentum.resource.ledger import invalidate as _invalidate_resource

        _invalidate_resource(request.app.state, "llm")
    except Exception:
        log.warning("reclaim_ledger_invalidate_failed", exc_info=True)
    return JSONResponse(result)


@router.post("/pause")
async def pause_container(request: Request) -> JSONResponse:
    """Stop a managed sidecar container (TTS/STT/classifier/vision) to free its
    VRAM. Reload it with /resume. Restricted to known sidecars server-side."""
    body = await request.json()
    container = (body.get("container") or "").strip()
    if not container:
        return JSONResponse({"ok": False, "error": "Missing container"}, status_code=400)
    from augmentum.resource.container_probe import set_container_paused
    ok, err = await set_container_paused(request.app.state, container, paused=True)
    if ok:
        log.info("sidecar_container_paused", container=container)
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False, "error": err or "Pause failed"}, status_code=500)


@router.post("/resume")
async def resume_container(request: Request) -> JSONResponse:
    """Start a previously-paused managed sidecar container."""
    body = await request.json()
    container = (body.get("container") or "").strip()
    if not container:
        return JSONResponse({"ok": False, "error": "Missing container"}, status_code=400)

    # Guard: if the managed classifier slot ("Slot C") currently holds the
    # "classifier" role, resuming the EXTERNAL classifier container would load a
    # duplicate model the registry never routes to (Slot C won the key) — the
    # "third model after unpause" bug. Refuse and point at the real action.
    if "classifier" in container.lower():
        from augmentum.models.classifier_slot import CLASSIFIER_BACKEND_KEY

        registry = getattr(request.app.state, "provider_registry", None)
        slot = getattr(request.app.state, "classifier_slot", None)
        slot_backend = getattr(slot, "_backend", None) if slot else None
        holder = registry._backends.get(CLASSIFIER_BACKEND_KEY) if registry else None
        if slot_backend is not None and holder is slot_backend:
            return JSONResponse(
                {
                    "ok": False,
                    "error": (
                        "Slot C is serving the classifier — resuming this "
                        "container loads an unused duplicate. Unload Slot C "
                        "first to hand the role back."
                    ),
                },
                status_code=409,
            )

    from augmentum.resource.container_probe import set_container_paused
    ok, err = await set_container_paused(request.app.state, container, paused=False)
    if ok:
        log.info("sidecar_container_resumed", container=container)
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False, "error": err or "Resume failed"}, status_code=500)


async def _unload_lmstudio(request: Request, model_name: str) -> JSONResponse:
    """Unload a model from LM Studio via POST /api/v1/models/unload."""
    from augmentum.models.openai_compat import OpenAIBackend

    registry = getattr(request.app.state, "provider_registry", None)
    if not registry:
        return JSONResponse({"ok": False, "error": "Provider registry not available"}, status_code=503)

    # Find the LM Studio backend
    for _key, backend in registry._backends.items():
        if not isinstance(backend, OpenAIBackend):
            continue
        url = backend._base_url.lower()
        if not ("lmstudio" in url or ":1234" in url):
            continue

        base = backend._base_url
        if base.endswith("/v1"):
            base = base[:-3]
        elif base.endswith("/v1/"):
            base = base[:-4]

        try:
            resp = await backend._client.post(
                f"{base}/api/v1/models/unload",
                json={"instance_id": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                log.info("lmstudio_model_unloaded", model=model_name)
                return JSONResponse({"ok": True})
            else:
                err = resp.text[:200]
                log.warning("lmstudio_unload_error", model=model_name, status=resp.status_code, body=err)
                return JSONResponse({"ok": False, "error": f"LM Studio returned {resp.status_code}"})
        except Exception:
            log.warning("lmstudio_unload_failed", model=model_name, exc_info=True)
            return JSONResponse({"ok": False, "error": "Could not reach LM Studio"}, status_code=502)

    return JSONResponse({"ok": False, "error": "LM Studio backend not found"}, status_code=404)
