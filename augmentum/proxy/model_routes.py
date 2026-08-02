"""Enhanced model management API routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace GGUF search — module-level cache + shared client
# ---------------------------------------------------------------------------

_hf_client: httpx.AsyncClient | None = None
_hf_cache: dict[str, tuple[float, list]] = {}  # key → (timestamp, results)
_HF_CACHE_TTL = 30  # seconds


def _get_hf_client() -> httpx.AsyncClient:
    """Lazy-init a shared httpx client for HuggingFace API calls."""
    global _hf_client
    if _hf_client is None or _hf_client.is_closed:
        _hf_client = httpx.AsyncClient(timeout=15.0)
    return _hf_client

router = APIRouter(prefix="/api/models", tags=["models"])
llamacpp_router = APIRouter(prefix="/api/llamacpp", tags=["llamacpp"])
engine_router = APIRouter(prefix="/api/engine", tags=["engine"])


@router.get("/status")
async def models_status(request: Request) -> dict:
    """Get all models across all backends."""
    manager = request.app.state.model_manager
    models = await manager.list_all_models()
    return {
        "models": [{"name": m.name, "size": m.size, "modified_at": m.modified_at} for m in models],
    }


@router.get("/running")
async def running_models(request: Request) -> dict:
    """Currently loaded models with resource usage."""
    manager = request.app.state.model_manager
    running = await manager.get_running_models()
    return {
        "models": [
            {
                "name": m.name,
                "backend": m.backend,
                "size_vram": m.size_vram,
                "size_ram": m.size_ram,
                "expires_at": m.expires_at,
            }
            for m in running
        ],
    }


def _gguf_download_destinations(request: Request) -> list[str]:
    """All directories the user is allowed to download GGUFs into.

    Union of: engine v2 model_dirs (which already include any host-mounted
    folders the user has registered) + the configured llamacpp_model_dir.
    The order puts the engine's dirs first so /models/host (the typical
    fast-storage mount) appears as the default in UI dropdowns.
    """
    dirs: list[str] = []
    seen: set[str] = set()
    llama_mgr = getattr(request.app.state, "llama_manager", None)
    if llama_mgr is not None:
        for d in llama_mgr.model_dirs:
            if d and d not in seen:
                dirs.append(d)
                seen.add(d)
    if settings.llamacpp_model_dir and settings.llamacpp_model_dir not in seen:
        dirs.append(settings.llamacpp_model_dir)
        seen.add(settings.llamacpp_model_dir)
    return dirs


@router.get("/download/destinations")
async def list_download_destinations(request: Request) -> JSONResponse:
    """Allowed GGUF download targets, used by the UI's destination dropdown."""
    dirs = _gguf_download_destinations(request)
    return JSONResponse({
        "destinations": dirs,
        "engine_default": settings.engine_model_dir,
        "llamacpp_default": settings.llamacpp_model_dir,
    })


# Per-destination disk usage cache. statfs is microseconds on a healthy
# fs but can stall seconds on a degraded one — we both cache (30s TTL,
# in-memory only, no DB hit) and run the syscall on the threadpool so a
# stuck mount can't freeze the event loop. Cache is keyed by the sorted
# tuple of dirs, so adding/removing a destination dir invalidates cleanly.
_STORAGE_CACHE: dict[str, dict] = {}
_STORAGE_TTL_S = 30.0


def _probe_disk_usage(dir_path: str) -> dict:
    """Single statfs probe. Sync — caller wraps in asyncio.to_thread."""
    try:
        usage = shutil.disk_usage(dir_path)
        return {
            "dir": dir_path,
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    except OSError as exc:
        return {"dir": dir_path, "error": str(exc)}


@router.get("/storage")
async def list_destination_storage(request: Request) -> JSONResponse:
    """Per-destination disk usage for the GGUF picker.

    Pure filesystem syscall — no SQLite, no aiosqlite. Cached 30s in
    process memory so the picker can re-fetch on every dest change
    without hammering statfs (and so the autosizing bar updates a few
    seconds after a download completes without a hand-rolled invalidate).
    """
    dirs = _gguf_download_destinations(request)
    cache_key = "|".join(sorted(dirs))
    cached = _STORAGE_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and (now - cached["ts"]) < _STORAGE_TTL_S:
        return JSONResponse({"destinations": cached["data"]})

    results = await asyncio.gather(
        *(asyncio.to_thread(_probe_disk_usage, d) for d in dirs)
    )
    _STORAGE_CACHE[cache_key] = {"ts": now, "data": list(results)}
    return JSONResponse({"destinations": list(results)})


def _request_user_id(request: Request) -> str:
    """Extract the authenticated user's id from the request scope."""
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


async def _enqueue_gguf_download(
    request: Request,
    *,
    repo_id: str,
    filename: str,
    model_dir: str,
    backend: str,
    total_size: int = 0,
) -> dict:
    """Enqueue a gguf_download job, deduping against any in-flight request for
    the same destination file. Returns ``{job_id, status, existing}``."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if not user_id or jobs_store is None or job_runner is None:
        raise HTTPException(status_code=503, detail="Background job queue unavailable")

    dest_path = os.path.join(model_dir, filename)
    existing = await jobs_store.list_for_user(
        user_id=user_id, job_type="gguf_download", limit=200,
    )
    for job in existing:
        if job.get("status") not in {"pending", "running"}:
            continue
        payload = job.get("payload") or {}
        existing_dest = os.path.join(
            str(payload.get("model_dir") or ""),
            str(payload.get("filename") or ""),
        )
        if existing_dest == dest_path:
            return {"job_id": job["id"], "status": job["status"], "existing": True}

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="gguf_download",
        payload={
            "repo_id": repo_id,
            "filename": filename,
            "model_dir": model_dir,
            "backend": backend,
            "total_size": int(total_size),
        },
        priority=10,
        max_attempts=3,
    )
    job_runner.wake()
    return {"job_id": job_id, "status": "queued", "existing": False}


async def _enqueue_gguf_bundle(
    request: Request,
    *,
    repo_id: str,
    files: list[dict],
    model_dir: str,
    backend: str,
) -> dict:
    """Enqueue ONE multi-file gguf_download job. The handler downloads all
    files in parallel — single-worker job runner means N separate jobs would
    queue serially, which is the slowness this endpoint exists to fix.

    Dedup: collapse against an existing in-flight bundle whose file set
    matches exactly (same model_dir + same sorted filenames). Partial
    overlap (e.g. one shard already enqueued separately) doesn't dedupe —
    the handler's per-file dest-exists check makes that safe.
    """
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if not user_id or jobs_store is None or job_runner is None:
        raise HTTPException(status_code=503, detail="Background job queue unavailable")
    if not files:
        raise HTTPException(status_code=400, detail="files must not be empty")

    requested_set = sorted(str(f.get("filename") or "").strip() for f in files)
    requested_dir = str(model_dir).rstrip(os.sep)

    existing = await jobs_store.list_for_user(
        user_id=user_id, job_type="gguf_download", limit=200,
    )
    for job in existing:
        if job.get("status") not in {"pending", "running"}:
            continue
        payload = job.get("payload") or {}
        if str(payload.get("model_dir") or "").rstrip(os.sep) != requested_dir:
            continue
        existing_files = payload.get("files") or []
        if not existing_files:
            # Single-file legacy job — only matches if our bundle is also
            # one file with that filename.
            single_fn = str(payload.get("filename") or "").strip()
            if single_fn and len(requested_set) == 1 and requested_set[0] == single_fn:
                return {"job_id": job["id"], "status": job["status"], "existing": True}
            continue
        existing_set = sorted(
            str(f.get("filename") or "").strip() for f in existing_files
        )
        if existing_set == requested_set:
            return {"job_id": job["id"], "status": job["status"], "existing": True}

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="gguf_download",
        payload={
            "repo_id": repo_id,
            "files": [
                {
                    "filename": str(f.get("filename") or "").strip(),
                    "total_size": int(f.get("total_size") or 0),
                }
                for f in files
            ],
            "model_dir": model_dir,
            "backend": backend,
        },
        priority=10,
        max_attempts=3,
    )
    job_runner.wake()
    return {"job_id": job_id, "status": "queued", "existing": False}


@router.post("/pull", response_model=None)
async def pull_model_unified(request: Request) -> JSONResponse | StreamingResponse:
    """Unified model pull.

    Admin-only — pulling weights consumes disk + bandwidth at deploy scale
    and writes to a shared model directory. Non-admin authed users can
    list and use models but can't trigger new pulls.

    For Ollama (default backend): proxies to Ollama's pull and streams progress
    inline — Ollama already handles its own resume.

    For llamacpp / engine: enqueues a background job (returns ``{job_id}``).
    The client then attaches to ``GET /downloads/{job_id}/stream`` for live
    progress, and the download survives client disconnect / page reload.

    Optional ``model_dir`` overrides the destination directory; must be one of
    the values returned by ``/download/destinations``.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    body = await request.json()
    backend = body.get("backend", "ollama")
    name = body.get("name", "")
    requested_dir = (body.get("model_dir") or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    manager = request.app.state.model_manager

    def _resolve_dir(default: str) -> tuple[str, str | None]:
        if not requested_dir:
            return default, None
        allowed = _gguf_download_destinations(request)
        if requested_dir not in allowed:
            return default, f"model_dir '{requested_dir}' not in allowed destinations"
        return requested_dir, None

    if backend == "llamacpp":
        filename = body.get("filename", "")
        if not filename:
            raise HTTPException(
                status_code=400,
                detail="filename is required for llama.cpp downloads",
            )
        chosen_dir, err = _resolve_dir(settings.llamacpp_model_dir)
        if err:
            raise HTTPException(status_code=400, detail=err)
        try:
            total_size = await manager.resolve_hf_file_size(name, filename)
        except Exception:
            total_size = 0
        result = await _enqueue_gguf_download(
            request,
            repo_id=name, filename=filename, model_dir=chosen_dir,
            backend="llamacpp", total_size=total_size,
        )
        return JSONResponse(result)

    if backend == "engine":
        filename = body.get("filename", "")
        repo = name
        if not filename:
            from augmentum.models.model_catalog import resolve_model_name
            resolved = resolve_model_name(name)
            if resolved:
                repo, filename = resolved
            else:
                return JSONResponse(
                    {"error": (
                        f"Unknown model '{name}'. Use 'name:quant' "
                        "(e.g. qwen3.5-7b:q4_k_m) or 'org/repo:file.gguf'"
                    )},
                    status_code=400,
                )
        chosen_dir, err = _resolve_dir(settings.engine_model_dir)
        if err:
            raise HTTPException(status_code=400, detail=err)
        try:
            total_size = await manager.resolve_hf_file_size(repo, filename)
        except Exception:
            total_size = 0
        result = await _enqueue_gguf_download(
            request,
            repo_id=repo, filename=filename, model_dir=chosen_dir,
            backend="engine", total_size=total_size,
        )
        return JSONResponse(result)

    # Default: Ollama pull
    async def _stream_ollama():
        async for chunk in manager.pull_model(name):
            yield json.dumps(chunk) + "\n"
        # Invalidate model map so new model appears immediately
        request.app.state.provider_registry.invalidate_model_map()
        from augmentum.proxy import system_events
        system_events.publish("models.installed", {"backend": "ollama", "name": name})

    return StreamingResponse(_stream_ollama(), media_type="application/x-ndjson")


@router.post("/pull/bundle", response_model=None)
async def pull_model_bundle(request: Request) -> JSONResponse:
    """Enqueue ONE multi-file download job (multi-shard model + mmproj).

    Request shape::

        {
            "name": "bartowski/Foo-GGUF",
            "filenames": [
                "Foo-Q4-00001-of-00004.gguf",
                "Foo-Q4-00002-of-00004.gguf",
                ...
                "mmproj-F16.gguf",
            ],
            "model_dir": "/models/host",       # optional
            "backend":   "engine" | "llamacpp"
        }

    The handler downloads all listed files in parallel; the front-end used
    to enqueue one job per file, which queued serially behind the single-
    worker job runner. For a 4-shard model that meant ~4× slower than it
    needed to be — this endpoint fixes that.
    """
    body = await request.json()
    backend = str(body.get("backend") or "").strip()
    if backend not in {"engine", "llamacpp"}:
        raise HTTPException(
            status_code=400,
            detail="backend must be 'engine' or 'llamacpp' for bundled pulls",
        )
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name (repo_id) is required")
    raw_filenames = body.get("filenames") or []
    if not isinstance(raw_filenames, list) or not raw_filenames:
        raise HTTPException(
            status_code=400,
            detail="filenames must be a non-empty list",
        )
    filenames = [str(f).strip() for f in raw_filenames if str(f).strip()]
    if not filenames:
        raise HTTPException(status_code=400, detail="filenames had no usable entries")

    requested_dir = (body.get("model_dir") or "").strip()
    default_dir = (
        settings.engine_model_dir if backend == "engine" else settings.llamacpp_model_dir
    )
    if requested_dir:
        allowed = _gguf_download_destinations(request)
        if requested_dir not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"model_dir '{requested_dir}' not in allowed destinations",
            )
        chosen_dir = requested_dir
    else:
        chosen_dir = default_dir

    manager = request.app.state.model_manager

    # Resolve missing sizes in parallel so the job's progress denominator
    # is accurate before the first byte flows. resolve_hf_file_size has its
    # own probe semaphore + cache so this is cheap on repeat calls.
    files: list[dict] = []
    sizes = await asyncio.gather(
        *(manager.resolve_hf_file_size(name, fn) for fn in filenames),
        return_exceptions=True,
    )
    for fn, size in zip(filenames, sizes, strict=True):
        files.append({
            "filename": fn,
            "total_size": int(size) if isinstance(size, int) and size > 0 else 0,
        })

    result = await _enqueue_gguf_bundle(
        request,
        repo_id=name, files=files, model_dir=chosen_dir, backend=backend,
    )
    return JSONResponse(result)


def _safe_repo_folder(repo_id: str) -> str:
    """Filesystem-safe folder name for a HF repo (``org/name`` → ``name``)."""
    import re as _re
    tail = repo_id.strip().rstrip("/").split("/")[-1]
    return _re.sub(r"[^A-Za-z0-9._-]", "_", tail) or "model"


def _build_vllm_cmd(model_name: str, serve_path: str, params: dict | None = None) -> str:
    """Compose the ``vllm serve`` command for a model, profile-aware.

    Defaults are derived from the model's config (via safetensors_profile) so we
    don't hardcode risky/wrong flags: ``--trust-remote-code`` is added ONLY when
    the model ships custom code (auto_map), and ``--max-model-len`` follows the
    model's real context (capped). ``params`` (from the per-model launch-params
    editor) overrides any derived default.
    """
    from augmentum.models.safetensors_profile import safetensors_profile

    prof = safetensors_profile(serve_path)
    p = params or {}
    dtype = str(p.get("dtype") or prof.get("dtype") or "bfloat16")
    if dtype not in ("bfloat16", "float16", "float32", "auto"):
        dtype = "bfloat16"
    # Context: honor an explicit override, else the model's context capped so a
    # 256K-context model doesn't reserve an impossible KV pool by default.
    ctx = int(p.get("max_model_len") or 0)
    if ctx <= 0:
        model_ctx = int(prof.get("context_length") or 0)
        ctx = min(model_ctx, 16384) if model_ctx else 8192
    trust = p.get("trust_remote_code")
    if trust is None:
        trust = bool(prof.get("needs_remote_code"))
    gpu_util = float(p.get("gpu_memory_utilization") or 0.90)
    tp = int(p.get("tensor_parallel_size") or 1)
    quant = str(p.get("quantization") or "").strip()

    parts = [
        f"vllm serve {serve_path}",
        f"--served-model-name {model_name}",
        "--host 0.0.0.0 --port ${PORT}",
        f"--dtype {dtype}",
        f"--max-model-len {ctx}",
        f"--gpu-memory-utilization {gpu_util}",
    ]
    # Only route through the Transformers modeling backend for custom-code models
    # (auto_map / trust_remote_code). Standard archs use vLLM's native, faster,
    # more up-to-date implementation — forcing --model-impl transformers on them
    # sends them through the (older, pinned) transformers and breaks brand-new
    # archs it doesn't recognize yet.
    if trust:
        parts.append("--model-impl transformers")
        parts.append("--trust-remote-code")
    if tp > 1:
        parts.append(f"--tensor-parallel-size {tp}")
    if quant:
        parts.append(f"--quantization {quant}")
    return " ".join(parts)


def _write_llama_swap_model_entry(model_name: str, serve_path: str, params: dict | None = None) -> bool:
    """Drop a per-model llama-swap config into the engine's --config-dir.

    The dir lives on the primary model dir, which is mirror-mounted into the vLLM
    engine container; llama-swap runs with --watch-config so it auto-reloads.
    Returns False (and logs) if the dir can't be written — the download still
    succeeds, the model just isn't registered for serving yet.
    """
    import os as _os

    primary = (settings.engine_model_dir or "").rstrip("/")
    if not primary:
        return False
    cfg_dir = f"{primary}/.augmentum-vllm"
    try:
        _os.makedirs(cfg_dir, exist_ok=True)
        # llama-swap ${PORT} macro is substituted per-upstream by llama-swap.
        cmd = _build_vllm_cmd(model_name, serve_path, params)
        # Minimal YAML — model names/paths are constrained charsets, so plain
        # formatting is safe (no arbitrary user strings that need escaping).
        doc = (
            "models:\n"
            f'  "{model_name}":\n'
            f"    cmd: >\n"
            f"      {cmd}\n"
            f"    ttl: 600\n"
        )
        with open(f"{cfg_dir}/{model_name}.yaml", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(doc)
        return True
    except OSError as exc:
        log.warning("llama_swap_entry_write_failed", model=model_name, error=str(exc))
        return False


@router.post("/pull/safetensors", response_model=None)
async def pull_safetensors_model(request: Request) -> JSONResponse:
    """Download a full safetensors repo into a per-repo subdir + register it with
    the vLLM engine (llama-swap).

    Request shape::

        {
            "name": "Nanbeige/Nanbeige4.1-3B",   # HF repo id
            "filenames": ["config.json", "model.safetensors", ...],  # full set
            "model_dir": "/models/host",          # parent (validated); optional
            "model_name": "Nanbeige4.1-3B"        # served name; optional
        }

    The repo is the serving unit, so all files land in ``<model_dir>/<repo>/`` and
    the engine serves that directory. Registration writes a per-model llama-swap
    config that the engine auto-reloads (--watch-config). Requires the vLLM engine
    to be installed (its --config-dir is where the entry lands).
    """
    body = await request.json()
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name (repo_id) is required")
    raw_filenames = body.get("filenames") or []
    filenames = [str(f).strip() for f in raw_filenames if str(f).strip()]
    if not filenames:
        raise HTTPException(status_code=400, detail="filenames must be a non-empty list")

    requested_dir = (body.get("model_dir") or "").strip()
    allowed = _gguf_download_destinations(request)
    parent = requested_dir or settings.engine_model_dir
    if parent not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"model_dir '{parent}' not in allowed destinations",
        )

    model_name = _safe_repo_folder(str(body.get("model_name") or name))
    subdir = f"{parent.rstrip('/')}/{_safe_repo_folder(name)}"
    # Augmentum has the parent model dir mounted, so it can create the subdir the
    # download job writes into (and the engine reads via the mirror mount).
    try:
        os.makedirs(subdir, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not create {subdir}: {exc}")

    manager = request.app.state.model_manager
    sizes = await asyncio.gather(
        *(manager.resolve_hf_file_size(name, fn) for fn in filenames),
        return_exceptions=True,
    )
    files: list[dict] = []
    for fn, size in zip(filenames, sizes, strict=True):
        files.append({
            "filename": fn,
            "total_size": int(size) if isinstance(size, int) and size > 0 else 0,
        })

    result = await _enqueue_gguf_bundle(
        request, repo_id=name, files=files, model_dir=subdir, backend="engine",
    )
    # Register now so the model appears in the engine's catalog; llama-swap won't
    # actually spawn vLLM until first request, by which time the files have landed.
    registered = _write_llama_swap_model_entry(model_name, subdir)
    result["model_name"] = model_name
    result["serve_path"] = subdir
    result["registered"] = registered
    return JSONResponse(result)


def _llama_swap_config_dir() -> str:
    primary = (settings.engine_model_dir or "").rstrip("/")
    return f"{primary}/.augmentum-vllm" if primary else ""


def _scan_safetensors_repos(dirs: list[str]) -> list[dict]:
    """Find downloaded safetensors model repos across the model dirs.

    A repo is an immediate subdir containing ``config.json`` + at least one
    ``*.safetensors``. Returns metadata + whether it's registered with the vLLM
    engine (a matching llama-swap config entry exists), so the library can show
    and manage them alongside GGUFs.
    """
    cfg_dir = _llama_swap_config_dir()
    out: list[dict] = []
    seen: set[str] = set()
    for base in dirs:
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            real = os.path.realpath(entry.path)
            if real in seen:
                continue
            try:
                names = os.listdir(entry.path)
            except OSError:
                continue
            has_config = "config.json" in names
            shards = [n for n in names if n.lower().endswith(".safetensors")]
            if not (has_config and shards):
                continue
            seen.add(real)
            total = 0
            for n in shards:
                try:
                    total += os.path.getsize(os.path.join(entry.path, n))
                except OSError:
                    pass
            name = _safe_repo_folder(entry.name)
            registered = bool(cfg_dir) and os.path.exists(f"{cfg_dir}/{name}.yaml")
            row = {
                "name": entry.name,
                "model_name": name,
                "path": entry.path,
                "model_dir": base,
                "size": total,
                "shards": len(shards),
                "registered": registered,
            }
            # Enrich with the capability profile (arch, context, params, tags) so
            # the library shows the same depth as GGUF models. Best-effort.
            try:
                from augmentum.models.safetensors_profile import safetensors_profile
                prof = safetensors_profile(entry.path)
                row.update({
                    "architecture": prof.get("architecture", ""),
                    "model_type": prof.get("model_type", ""),
                    "context_length": prof.get("context_length", 0),
                    "params_est": prof.get("params_est", 0),
                    "is_moe": prof.get("is_moe", False),
                    "expert_count": prof.get("expert_count", 0),
                    "dtype": prof.get("dtype", ""),
                    "reasoning_family": prof.get("reasoning_family", ""),
                    "vision": prof.get("vision", False),
                    "tools": prof.get("tools", False),
                    "needs_remote_code": prof.get("needs_remote_code", False),
                })
            except Exception as exc:  # noqa: BLE001
                log.warning("safetensors_profile_failed", repo=entry.path, error=str(exc))
            out.append(row)
    return out


@router.get("/safetensors/local")
async def list_local_safetensors(request: Request) -> JSONResponse:
    """List downloaded safetensors model repos (for the vLLM engine)."""
    dirs = _gguf_download_destinations(request)
    return JSONResponse({"repos": _scan_safetensors_repos(dirs)})


@router.delete("/safetensors/local")
async def delete_local_safetensors(request: Request) -> JSONResponse:
    """Delete a downloaded safetensors repo + its llama-swap registration.

    Body: ``{"path": "/models/host/Nanbeige4.2-3B"}``. The path must be a
    safetensors repo inside an allowed model dir — anything else is rejected.
    """
    import shutil as _shutil

    body = await request.json()
    raw_path = str(body.get("path") or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="path is required")
    target = os.path.realpath(raw_path)

    allowed = [os.path.realpath(d) for d in _gguf_download_destinations(request)]
    # Must live directly inside an allowed model dir (a repo subdir), never be
    # the model dir itself or outside it.
    parent = os.path.dirname(target)
    if parent not in allowed:
        raise HTTPException(status_code=400, detail="path is not inside an allowed model dir")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="repo not found")
    names = os.listdir(target)
    if "config.json" not in names or not any(n.lower().endswith(".safetensors") for n in names):
        raise HTTPException(status_code=400, detail="not a safetensors repo")

    model_name = _safe_repo_folder(os.path.basename(target))
    try:
        _shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"delete failed: {exc}")
    # Remove the llama-swap registration so the engine stops offering it.
    cfg_dir = _llama_swap_config_dir()
    unregistered = False
    if cfg_dir:
        entry = f"{cfg_dir}/{model_name}.yaml"
        try:
            if os.path.exists(entry):
                os.remove(entry)
                unregistered = True
        except OSError as exc:
            log.warning("llama_swap_entry_remove_failed", model=model_name, error=str(exc))
    return JSONResponse({"deleted": True, "model_name": model_name, "unregistered": unregistered})


def _params_sidecar_path(model_name: str) -> str:
    cfg_dir = _llama_swap_config_dir()
    return f"{cfg_dir}/{model_name}.params.json" if cfg_dir else ""


def _validate_safetensors_path(request: Request, raw_path: str) -> str:
    """Return the realpath of a safetensors repo inside an allowed model dir."""
    target = os.path.realpath(raw_path)
    allowed = [os.path.realpath(d) for d in _gguf_download_destinations(request)]
    if os.path.dirname(target) not in allowed:
        raise HTTPException(status_code=400, detail="path is not inside an allowed model dir")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="repo not found")
    names = os.listdir(target)
    if "config.json" not in names or not any(n.lower().endswith(".safetensors") for n in names):
        raise HTTPException(status_code=400, detail="not a safetensors repo")
    return target


@router.get("/safetensors/local/launch")
async def get_safetensors_launch(request: Request, path: str = "") -> JSONResponse:
    """Per-model vLLM launch params: derived defaults + saved overrides + profile.

    Powers the launch-params editor. ``derived`` are the profile-aware defaults
    the engine would use with no overrides; ``params`` are the user's saved
    overrides (empty until they customize)."""
    import json as _json

    from augmentum.models.safetensors_profile import safetensors_profile

    target = _validate_safetensors_path(request, path)
    model_name = _safe_repo_folder(os.path.basename(target))
    prof = safetensors_profile(target)
    model_ctx = int(prof.get("context_length") or 0)
    derived = {
        "dtype": prof.get("dtype") or "bfloat16",
        "max_model_len": (min(model_ctx, 16384) if model_ctx else 8192),
        "trust_remote_code": bool(prof.get("needs_remote_code")),
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "quantization": "",
    }
    saved: dict = {}
    sidecar = _params_sidecar_path(model_name)
    if sidecar and os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as fh:
                saved = _json.load(fh)
        except (OSError, ValueError):
            saved = {}
    return JSONResponse({
        "model_name": model_name,
        "profile": prof,
        "derived": derived,
        "params": saved if isinstance(saved, dict) else {},
        "model_max_context": model_ctx,
    })


@router.put("/safetensors/local/launch")
async def put_safetensors_launch(request: Request) -> JSONResponse:
    """Save per-model launch params + rewrite the llama-swap entry (auto-reload).

    Body: ``{"path": "...", "params": {dtype, max_model_len, trust_remote_code,
    gpu_memory_utilization, tensor_parallel_size, quantization}}``. Only known
    keys are honored; the engine picks up the change via --watch-config."""
    import json as _json

    body = await request.json()
    target = _validate_safetensors_path(request, str(body.get("path") or ""))
    model_name = _safe_repo_folder(os.path.basename(target))

    raw = body.get("params") or {}
    allowed_keys = {
        "dtype", "max_model_len", "trust_remote_code",
        "gpu_memory_utilization", "tensor_parallel_size", "quantization",
    }
    params = {k: v for k, v in raw.items() if k in allowed_keys} if isinstance(raw, dict) else {}

    # Persist overrides (sidecar) so they survive re-registration + reboot.
    sidecar = _params_sidecar_path(model_name)
    if sidecar:
        try:
            os.makedirs(os.path.dirname(sidecar), exist_ok=True)
            with open(sidecar, "w", encoding="utf-8", newline="\n") as fh:
                _json.dump(params, fh)
        except OSError as exc:
            log.warning("safetensors_params_persist_failed", model=model_name, error=str(exc))

    registered = _write_llama_swap_model_entry(model_name, target, params)
    return JSONResponse({
        "model_name": model_name,
        "registered": registered,
        "cmd": _build_vllm_cmd(model_name, target, params),
    })


@router.post("/vllm/load")
async def load_vllm_model(request: Request) -> JSONResponse:
    """Warm up a registered safetensors model on the running vLLM engine.

    llama-swap spins models up lazily on first request; this sends a tiny request
    so the model is loaded and READY (and surfaces a load error immediately rather
    than on the user's first chat). The model must already be registered (has a
    llama-swap entry) — Register-to-serve / download does that."""
    import os as _os

    import httpx

    body = await request.json()
    model_name = _safe_repo_folder(str(body.get("model_name") or ""))
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    base = (settings.vllm_base_url or _os.environ.get("AUGMENTUM_VLLM_BASE_URL", "")).rstrip("/")
    if not base:
        raise HTTPException(
            status_code=409,
            detail="The vLLM engine isn't registered — install it from Discover.",
        )
    try:
        # Generous timeout: first load spins up the vLLM upstream + loads weights.
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM engine unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"vLLM load failed: {resp.text[:400]}")
    return JSONResponse({"loaded": True, "model_name": model_name})


# ---------------------------------------------------------------------------
# Download management — list active downloads, cancel, re-attach progress.
# Backed by the gguf_download job handler so downloads survive client
# disconnect / page reload.
# ---------------------------------------------------------------------------

def _download_target_paths(job: dict, request: Request) -> tuple[str, str] | tuple[None, None]:
    """Resolve a job row to (final_path, partial_path) within allowed dirs."""
    payload = job.get("payload") or {}
    model_dir = str(payload.get("model_dir") or "").strip()
    filename = str(payload.get("filename") or "").strip()
    if not model_dir or not filename:
        return (None, None)

    target = os.path.realpath(os.path.join(model_dir, filename))
    allowed_dirs = [os.path.realpath(d) for d in _gguf_download_destinations(request)]
    if not any(
        target == d or target.startswith(d.rstrip(os.sep) + os.sep)
        for d in allowed_dirs
    ):
        return (None, None)
    return (target, target + ".part")


def _download_to_dict(job: dict, request: Request | None = None) -> dict:
    """Project a background_jobs row into the shape the UI expects."""
    payload = job.get("payload") or {}
    result = job.get("result") or {}
    total = int(payload.get("total_size") or result.get("size") or 0)
    progress = float(job.get("progress") or 0.0)
    completed = int(progress * total) if total > 0 else 0
    data = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage") or "",
        "progress": progress,
        "completed": completed,
        "total": total,
        "filename": str(payload.get("filename") or ""),
        "repo_id": str(payload.get("repo_id") or ""),
        "model_dir": str(payload.get("model_dir") or ""),
        "backend": str(payload.get("backend") or ""),
        "error": job.get("error") or "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
    }
    if request is not None:
        target, partial = _download_target_paths(job, request)
        has_partial = bool(partial and os.path.isfile(partial))
        data.update({
            "has_partial": has_partial,
            "partial_size": os.path.getsize(partial) if has_partial else 0,
            "has_file": bool(target and os.path.isfile(target)),
        })
    return data


@router.get("/downloads")
async def list_downloads(request: Request, limit: int = 50) -> JSONResponse:
    """Return the user's recent GGUF downloads (in-flight first)."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if not user_id or jobs_store is None:
        return JSONResponse({"downloads": []})
    jobs = await jobs_store.list_for_user(
        user_id=user_id, job_type="gguf_download", limit=limit,
    )
    # Active first, then recently completed.
    active = [j for j in jobs if j.get("status") in {"pending", "running"}]
    other = [j for j in jobs if j.get("status") not in {"pending", "running"}]
    return JSONResponse({
        "downloads": [_download_to_dict(j, request) for j in active + other],
    })


@router.post("/downloads/{job_id}/cancel")
async def cancel_download(job_id: str, request: Request) -> JSONResponse:
    """Request cancellation. The handler keeps the .part file so a future
    pull of the same file resumes from where it stopped."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if not user_id or jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")
    ok = await jobs_store.request_cancel(job_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Download not found")
    return JSONResponse({"job_id": job_id, "cancel_requested": True})


@router.post("/downloads/{job_id}/retry")
async def retry_download(job_id: str, request: Request) -> JSONResponse:
    """Retry a failed/cancelled GGUF download using the same job row."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if not user_id or jobs_store is None or job_runner is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    job = await jobs_store.get(job_id, user_id=user_id)
    if not job or job.get("job_type") != "gguf_download":
        raise HTTPException(status_code=404, detail="Download not found")
    if job.get("status") not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Only failed or cancelled downloads can be retried")

    ok = await jobs_store.reset_for_retry(job_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Download could not be retried")
    job_runner.wake()
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.delete("/downloads/{job_id}")
async def delete_download(
    job_id: str,
    request: Request,
    delete_partial: bool = Query(False, description="Also delete the resumable .part file"),
) -> JSONResponse:
    """Delete a terminal download row, optionally removing its .part file."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if not user_id or jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    job = await jobs_store.get(job_id, user_id=user_id)
    if not job:
        return JSONResponse({
            "job_id": job_id,
            "deleted": False,
            "already_gone": True,
            "partial_deleted": False,
        })
    if job.get("job_type") != "gguf_download":
        raise HTTPException(status_code=404, detail="Download not found")
    if job.get("status") in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="Active downloads must be cancelled before removal")

    partial_deleted = False
    if delete_partial:
        _target, partial = _download_target_paths(job, request)
        if not partial:
            raise HTTPException(status_code=400, detail="Download path is outside the configured model directories")
        if os.path.exists(partial):
            try:
                os.remove(partial)
                partial_deleted = True
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Partial delete failed: {exc}")

    ok = await jobs_store.delete_job(job_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Download not found")
    return JSONResponse({
        "job_id": job_id,
        "deleted": True,
        "partial_deleted": partial_deleted,
    })


@router.post("/downloads/cleanup")
async def cleanup_downloads(request: Request) -> JSONResponse:
    """Bulk-clear terminal download rows, optionally deleting .part files."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if not user_id or jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    requested_statuses = body.get("statuses") or ["completed", "failed", "cancelled"]
    statuses = tuple(
        s for s in requested_statuses
        if s in {"completed", "failed", "cancelled"}
    )
    if not statuses:
        raise HTTPException(status_code=400, detail="At least one terminal status is required")

    require_partial = body.get("require_partial")
    if require_partial not in {None, True, False}:
        raise HTTPException(status_code=400, detail="require_partial must be true, false, or omitted")
    delete_partial = bool(body.get("delete_partial"))

    jobs = await jobs_store.list_for_user(
        user_id=user_id, job_type="gguf_download", limit=500,
    )

    removed = 0
    partial_deleted = 0
    skipped = 0
    errors = 0
    for job in jobs:
        status = str(job.get("status") or "")
        if status not in statuses or status in {"pending", "running"}:
            continue

        _target, partial = _download_target_paths(job, request)
        has_partial = bool(partial and os.path.isfile(partial))
        if require_partial is True and not has_partial:
            continue
        if require_partial is False and has_partial:
            continue

        if delete_partial and has_partial:
            try:
                os.remove(partial)
                partial_deleted += 1
            except OSError:
                errors += 1
                log.warning("download_cleanup_partial_delete_failed", job_id=job.get("id"), exc_info=True)
                continue

        ok = await jobs_store.delete_job(str(job.get("id") or ""), user_id=user_id)
        if ok:
            removed += 1
        else:
            skipped += 1

    return JSONResponse({
        "removed": removed,
        "partial_deleted": partial_deleted,
        "skipped": skipped,
        "errors": errors,
    })


@router.get("/downloads/{job_id}/stream")
async def stream_download_progress(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream of progress for one download. Closes when the job reaches a
    terminal state (completed / failed / cancelled). Safe to (re)connect
    multiple times — each connection polls independently."""
    user_id = _request_user_id(request)
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if not user_id or jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    async def _events():
        last_emit = None
        terminal = {"completed", "failed", "cancelled"}
        while True:
            if await request.is_disconnected():
                break
            job = await jobs_store.get(job_id, user_id=user_id)
            if job is None:
                yield "event: error\ndata: {\"error\":\"not found\"}\n\n"
                return
            payload = _download_to_dict(job, request)
            # Emit only when something material changed, but always emit on
            # the first tick so the client gets initial state.
            sig = (payload["status"], payload["stage"], round(payload["progress"], 4))
            if sig != last_emit:
                last_emit = sig
                yield f"data: {json.dumps(payload)}\n\n"
            if payload["status"] in terminal:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/local")
async def delete_local_gguf(request: Request) -> JSONResponse:
    """Delete a downloaded GGUF file from disk.

    Body: ``{"path": "/models/host/foo.gguf"}``. The path must live inside
    one of the allowed download destinations (engine model_dirs +
    llamacpp_model_dir) — anything else is rejected to prevent stray rm.
    """
    user_id = _request_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    raw_path = (body.get("path") or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="path is required")

    # Resolve symlinks/.. to a canonical path before the prefix check, so a
    # crafted "/models/host/../etc/passwd" can't escape the allowlist.
    target = os.path.realpath(raw_path)
    if not target.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files can be deleted via this endpoint")

    allowed_dirs = [os.path.realpath(d) for d in _gguf_download_destinations(request)]
    if not any(
        target == d or target.startswith(d.rstrip(os.sep) + os.sep)
        for d in allowed_dirs
    ):
        raise HTTPException(
            status_code=400,
            detail="Path is outside the configured model directories",
        )

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found")

    size = 0
    try:
        size = os.path.getsize(target)
    except OSError:
        pass

    try:
        os.remove(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")

    # Also clean up any orphaned .part for the same target.
    with contextlib.suppress(OSError):
        os.remove(target + ".part")

    # Refresh registries so the deleted model disappears from pickers.
    try:
        registry = getattr(request.app.state, "provider_registry", None)
        if registry is not None:
            registry.invalidate_model_map()
    except Exception:
        log.warning("delete_local_gguf_invalidate_failed", exc_info=True)
    try:
        llama_mgr = getattr(request.app.state, "llama_manager", None)
        if llama_mgr is not None:
            await llama_mgr.scan_and_cache_profiles()
    except Exception:
        log.warning("delete_local_gguf_engine_scan_failed", exc_info=True)

    log.info("local_gguf_deleted", path=target, user_id=user_id, size=size)
    return JSONResponse({"path": target, "deleted": True, "size": size})


@router.get("/gguf/list")
async def list_gguf_files(
    request: Request,
    repo: str = Query(..., description="HuggingFace repo ID"),
) -> dict:
    """List available .gguf files in a HuggingFace repo."""
    manager = request.app.state.model_manager
    try:
        files = await manager.list_gguf_files(repo)
    except ImportError as exc:
        return {"error": str(exc), "files": []}
    except ValueError as exc:
        return {"error": str(exc), "files": []}
    return {"repo": repo, "files": files}


@router.get("/gguf/local")
async def list_local_gguf(request: Request) -> dict:
    """List locally downloaded GGUF files."""
    manager = request.app.state.model_manager
    files = manager.list_local_gguf(settings.llamacpp_model_dir)
    return {"model_dir": settings.llamacpp_model_dir, "files": files}


@router.get("/vision/captioner-options")
async def vision_captioner_options(request: Request) -> dict:
    """VL base+projector pairs on disk, for the vision-sibling picker.

    Powers the settings dropdowns that let a user choose the captioner
    (vision sibling) model from what they actually have installed, instead
    of typing two absolute GGUF paths. Only base models with at least one
    DIM-COMPATIBLE mmproj are offered (a projector that won't load is not a
    real option); incompatible candidates are still listed per-base with a
    reason so the UI can grey them out and explain why.

    On-demand only (opened settings panel) — it peeks GGUF headers, so it
    is deliberately kept off any hot path. Declared BEFORE the
    ``/{model_name:path}`` catch-alls so the literal path wins.
    """
    from pathlib import Path as _Path

    from augmentum.models.llama_server_manager import _MMPROJ_FILENAME_RE

    current = {
        "model_path": settings.vision_provider_model_path,
        "mmproj_path": settings.vision_provider_mmproj_path,
    }
    mgr = getattr(request.app.state, "llama_manager", None)
    if mgr is None:
        return {"available": False, "options": [], "current": current}

    options: list[dict] = []
    for f in mgr.discover_gguf_files():
        filename = f.get("filename", "")
        base_path = f.get("path", "")
        if not base_path or _MMPROJ_FILENAME_RE.search(filename):
            continue  # an mmproj is never itself a base model
        profile = mgr.profile_cache.get(base_path)
        candidates = mgr.suggest_mmproj_candidates(base_path, profile)
        if not any(c.get("compatible") for c in candidates):
            continue  # no projector would actually load → not a captioner option
        options.append({
            "base_path": base_path,
            "base_filename": filename,
            "base_name": _Path(filename).stem,
            "projectors": candidates,
        })
    options.sort(key=lambda o: o["base_name"].lower())
    return {"available": True, "options": options, "current": current}


@router.get("/{model_name:path}/sampling")
async def get_model_sampling(model_name: str, request: Request) -> JSONResponse:
    """Return the per-model sampling profile for *model_name*.

    ``effective`` = what would actually apply (user edit → install seed →
    family default). ``override`` = only the user's own saved edit (so the UI
    can show "inheriting default" vs "customized"). ``recommended`` = the
    family's known-good values, for a "reset to recommended" button.
    """
    from augmentum.models.sampling_profiles import (
        load_overrides,
        recommended_for,
        resolve_sampling,
    )
    store = getattr(request.app.state, "settings_store", None)
    uid = _request_user_id(request)
    override = await load_overrides(model_name, store, user_id=uid)
    recommended = recommended_for(model_name)
    effective = resolve_sampling(model_name, per_model=override)
    # Which sampler knobs the RESOLVED serving backend actually honors, so the
    # Tuning editor can hide the ones the provider drops (e.g. min_p/top_k on
    # OpenAI) instead of offering a dead control. ``None`` = couldn't resolve →
    # the editor falls back to showing every field (prior behavior).
    supported = None
    try:
        registry = getattr(request.app.state, "provider_registry", None)
        if registry is not None:
            backend, _clean = await registry.resolve_backend_for_model(model_name)
            supported = sorted(backend.supported_sampler_params(model_name))
    except Exception:
        log.warning("model_sampling_supported_resolve_failed",
                    model=model_name, exc_info=True)
    return JSONResponse({
        "model": model_name,
        "override": override.to_dict(),
        "recommended": recommended.to_dict(),
        "effective": effective.to_dict(),
        "supported": supported,
    })


@router.put("/{model_name:path}/sampling")
async def put_model_sampling(model_name: str, request: Request) -> JSONResponse:
    """Save (or clear) the user's per-model sampling override.

    Body: ``{temperature?, top_p?, top_k?, min_p?, repeat_penalty?}`` — any
    omitted/null field defers to the next layer. An empty body clears the
    override (revert to the install seed / family default).
    """
    from augmentum.models.sampling_profiles import (
        SamplingParams,
        recommended_for,
        resolve_sampling,
        save_overrides,
    )
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="settings store unavailable")
    uid = _request_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        body = await request.json()
    except Exception:
        body = {}
    params = SamplingParams.from_dict(body if isinstance(body, dict) else {})
    await save_overrides(model_name, params, store, user_id=uid)
    effective = resolve_sampling(model_name, per_model=params)
    return JSONResponse({
        "model": model_name,
        "override": params.to_dict(),
        "recommended": recommended_for(model_name).to_dict(),
        "effective": effective.to_dict(),
        "saved": True,
    })


@router.get("/{model_name:path}/info")
async def model_info(model_name: str, request: Request) -> dict:
    """Detailed model information."""
    manager = request.app.state.model_manager
    status = await manager.get_model_status(model_name)
    return {
        "name": status.name,
        "available": status.available,
        "backend": status.backend,
        "quantization": status.quantization,
        "parameter_count": status.parameter_count,
        "loaded": status.loaded,
    }


@router.post("/{model_name:path}/load")
async def load_model(model_name: str, request: Request) -> dict:
    """Pre-load a model into memory."""
    manager = request.app.state.model_manager
    success = await manager.load_model(model_name)
    return {"success": success, "model": model_name}


@router.post("/{model_name:path}/unload")
async def unload_model(model_name: str, request: Request) -> dict:
    """Unload a model from memory."""
    manager = request.app.state.model_manager
    success = await manager.unload_model(model_name)
    return {"success": success, "model": model_name}


# ── Projector pairing ──────────────────────────────────────────────────
#
# Vision capability for a local GGUF is gated on whether an mmproj
# projector is paired with it. The pairing decision lives in a sidecar
# JSON next to the base GGUF (see llama_server_manager.write_projector_
# sidecar) -- this mirrors how Jan / Ollama handle it: operator-declared,
# not heuristically guessed.


def _resolve_base_gguf_path(request: Request, model_name: str) -> tuple[str, Any]:
    """Look up the on-disk GGUF + cached profile for a model name.

    Returns ``(path, profile_or_none)``. Raises HTTPException(404) when
    the model is not a local GGUF (e.g., an Ollama-hosted model).
    """

    from pathlib import Path as _Path

    mgr = getattr(request.app.state, "llama_manager", None)
    if mgr is None:
        raise HTTPException(503, "llama-server manager not available")
    files = mgr.discover_gguf_files()
    match = next(
        (f for f in files if _Path(f["filename"]).stem == model_name),
        None,
    )
    if match is None:
        raise HTTPException(404, f"local GGUF not found for model {model_name!r}")
    profile = mgr.profile_cache.get(match["path"])
    return match["path"], profile


@router.get("/{model_name:path}/projector")
async def get_projector(model_name: str, request: Request) -> dict:
    """Return the current projector pairing plus all candidates.

    Drives the UI's "Pair projector" affordance. The response shape:

        {
          "current": "<mmproj path or empty>",
          "candidates": [
            {"path": ..., "filename": ..., "compatible": true,
             "reason": "", "projector_type": "qwen3vl_merger",
             "projection_dim": 4096, "is_current": false},
            ...
          ]
        }
    """

    from augmentum.models.llama_server_manager import read_projector_sidecar

    base_path, profile = _resolve_base_gguf_path(request, model_name)
    mgr = request.app.state.llama_manager
    return {
        "current": read_projector_sidecar(base_path),
        "candidates": mgr.suggest_mmproj_candidates(base_path, profile),
    }


@router.post("/{model_name:path}/projector")
async def set_projector(model_name: str, request: Request) -> dict:
    """Pair an mmproj with this base model, or unpair when path is empty.

    Body: ``{"mmproj_path": "<absolute path or empty>"}``. The pairing
    is dim-checked via :func:`validate_mmproj_pair` BEFORE the sidecar
    is written, so a confirmed pair will not crash llama-server at next
    load. Returns 400 on dim mismatch with the reason string.
    """

    from augmentum.models.llama_server_manager import (
        validate_mmproj_pair,
        write_projector_sidecar,
    )

    body = await request.json()
    mmproj = str(body.get("mmproj_path") or "").strip()
    base_path, profile = _resolve_base_gguf_path(request, model_name)

    if mmproj:
        ok, reason = validate_mmproj_pair(base_path, mmproj, profile)
        if not ok:
            raise HTTPException(400, f"incompatible projector: {reason}")

    write_projector_sidecar(base_path, mmproj)

    # Invalidate the model map so the dropdown sees the new vision flag
    # on its next fetch.
    registry = getattr(request.app.state, "provider_registry", None)
    if registry is not None:
        registry.invalidate_model_map()

    log.info(
        "projector_paired" if mmproj else "projector_unpaired",
        model=model_name,
        mmproj=mmproj or "(none)",
    )
    return {"model": model_name, "mmproj_path": mmproj}


# --- llama.cpp-specific routes ---


def _get_llamacpp_backend(request: Request):
    """Get the llama.cpp backend or raise 404."""
    registry = request.app.state.provider_registry
    backend = registry.get_backend("llamacpp")
    if not backend:
        engine_backend = registry.get_backend("engine")
        if engine_backend and all(
            hasattr(engine_backend, attr)
            for attr in ("health", "get_props", "get_slots", "tokenize", "detokenize")
        ):
            backend = engine_backend
    if not backend:
        raise HTTPException(status_code=404, detail="llama.cpp backend not configured")
    return backend


@llamacpp_router.get("/status")
async def llamacpp_status(request: Request) -> JSONResponse:
    """Get llama.cpp server status including health, props, slots, and LoRA info."""
    backend = _get_llamacpp_backend(request)

    health = await backend.health()
    props = await backend.get_props()
    slots = await backend.get_slots()

    # Try router mode — if list_router_models succeeds, it's in router mode
    router_models: list[dict] = []
    is_router = False
    try:
        router_models = await backend.list_router_models()
        is_router = len(router_models) > 1
    except Exception:
        log.debug("router_models_list_failed", exc_info=True)

    lora: list[dict] = []
    try:
        lora = await backend.list_lora_adapters()
    except Exception:
        log.warning("lora_adapter_list_failed")

    return JSONResponse({
        "health": health,
        "props": props,
        "slots": slots,
        "is_router_mode": is_router,
        "router_models": router_models,
        "lora_adapters": lora,
    })


@llamacpp_router.post("/models/load")
async def llamacpp_load_model(request: Request) -> JSONResponse:
    """Load a model (router mode)."""
    body = await request.json()
    model_name = body.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="model name required")

    backend = _get_llamacpp_backend(request)

    success = await backend.load_model(model_name)
    if not success:
        raise HTTPException(status_code=502, detail="Failed to load model")
    log.info("llamacpp_model_loaded", model=model_name)
    return JSONResponse({"status": "loaded", "model": model_name})


@llamacpp_router.post("/models/unload")
async def llamacpp_unload_model(request: Request) -> JSONResponse:
    """Unload a model (router mode)."""
    body = await request.json()
    model_name = body.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="model name required")

    backend = _get_llamacpp_backend(request)

    success = await backend.unload_model(model_name)
    if not success:
        raise HTTPException(status_code=502, detail="Failed to unload model")
    log.info("llamacpp_model_unloaded", model=model_name)
    return JSONResponse({"status": "unloaded", "model": model_name})


@llamacpp_router.get("/slots")
async def llamacpp_slots(request: Request) -> JSONResponse:
    """Get detailed llama.cpp slot information."""
    backend = _get_llamacpp_backend(request)
    slots = await backend.get_slots()
    return JSONResponse({"slots": slots})


@llamacpp_router.post("/slots/{slot_id}/erase")
async def llamacpp_erase_slot(slot_id: int, request: Request) -> JSONResponse:
    """Erase a slot's KV cache."""
    backend = _get_llamacpp_backend(request)
    try:
        resp = await backend._client.post(
            f"{backend._base_url}/slots/{slot_id}?action=erase",
            headers=backend._headers(),
        )
        resp.raise_for_status()
        return JSONResponse(resp.json())
    except Exception as exc:
        log.warning("llamacpp_erase_slot_failed", slot_id=slot_id, exc_info=True)
        raise HTTPException(status_code=502, detail="Slot operation failed") from exc


@llamacpp_router.get("/lora")
async def llamacpp_list_lora(request: Request) -> JSONResponse:
    """List LoRA adapters loaded in llama.cpp."""
    backend = _get_llamacpp_backend(request)
    adapters = await backend.list_lora_adapters()
    return JSONResponse({"adapters": adapters})


@llamacpp_router.post("/lora")
async def llamacpp_update_lora(request: Request) -> JSONResponse:
    """Update LoRA adapter scales."""
    body = await request.json()
    adapters = body.get("adapters", [])
    if not isinstance(adapters, list):
        raise HTTPException(status_code=400, detail="adapters must be a list")

    backend = _get_llamacpp_backend(request)

    success = await backend.set_lora_adapters(adapters)
    if not success:
        raise HTTPException(status_code=502, detail="Failed to update LoRA adapters")
    log.info("llamacpp_lora_updated", count=len(adapters))
    return JSONResponse({"status": "updated", "adapters": adapters})


@llamacpp_router.post("/tokenize")
async def llamacpp_tokenize(request: Request) -> JSONResponse:
    """Tokenize text using llama.cpp."""
    body = await request.json()
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    backend = _get_llamacpp_backend(request)
    tokens = await backend.tokenize(content)
    return JSONResponse({"tokens": tokens, "count": len(tokens)})


@llamacpp_router.post("/detokenize")
async def llamacpp_detokenize(request: Request) -> JSONResponse:
    """Detokenize tokens using llama.cpp."""
    body = await request.json()
    tokens = body.get("tokens", [])
    if not isinstance(tokens, list):
        raise HTTPException(status_code=400, detail="tokens must be a list")

    backend = _get_llamacpp_backend(request)
    content = await backend.detokenize(tokens)
    return JSONResponse({"content": content})


# ---------------------------------------------------------------------------
# Engine Management Routes
# ---------------------------------------------------------------------------

def _get_engine_backend(request: Request):
    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    if not backend:
        raise HTTPException(status_code=404, detail="Engine backend not configured")
    return backend


@engine_router.get("/models")
async def engine_list_models(request: Request) -> JSONResponse:
    """List models available on the Augmentum Engine."""
    backend = _get_engine_backend(request)
    models = await backend.list_models()
    return JSONResponse({
        "models": [{"name": m.name, "id": m.name, "size": m.size, "modified_at": m.modified_at} for m in models],
    })


@engine_router.get("/catalog")
async def engine_model_catalog() -> JSONResponse:
    """List friendly engine download aliases for the UI."""
    from augmentum.models.model_catalog import list_catalog

    return JSONResponse({"models": list_catalog()})


@engine_router.post("/models/load")
async def engine_load_model(request: Request) -> JSONResponse:
    """Load a model on the Augmentum Engine."""
    body = await request.json()
    model_name = body.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="model name required")

    backend = _get_engine_backend(request)
    success = await backend.load_model(model_name)
    if not success:
        raise HTTPException(status_code=502, detail="Failed to load model on engine")
    log.info("engine_model_loaded", model=model_name)
    # Invalidate model map so the new active model appears
    request.app.state.provider_registry.invalidate_model_map()
    return JSONResponse({"status": "loaded", "model": model_name})


@engine_router.post("/models/unload")
async def engine_unload_model(request: Request) -> JSONResponse:
    """Unload the current model — routes to v2 manager if available."""
    # v2 managed engine
    mgr = getattr(request.app.state, "llama_manager", None)
    if mgr:
        await mgr.stop()
        request.app.state.provider_registry.invalidate_model_map()
        return JSONResponse({"status": "unloaded"})
    # v1 fallback
    backend = _get_engine_backend(request)
    success = await backend.unload_model("")
    if not success:
        raise HTTPException(status_code=502, detail="Failed to unload model")
    log.info("engine_model_unloaded")
    request.app.state.provider_registry.invalidate_model_map()
    return JSONResponse({"status": "unloaded"})


@engine_router.get("/status")
async def engine_status(request: Request) -> JSONResponse:
    """Get engine status — routes to v2 manager if available, else v1."""
    # v2 managed engine
    mgr = getattr(request.app.state, "llama_manager", None)
    if mgr:
        # mgr.status() runs nvidia-smi (5s timeout) + check_alive() inline.
        # Off-load so the polling UI doesn't stall the event loop.
        return JSONResponse(await asyncio.to_thread(mgr.status))
    # v1 fallback
    backend = _get_engine_backend(request)
    if hasattr(backend, "engine_status"):
        status = await backend.engine_status()
        return JSONResponse(status)
    return JSONResponse({"status": "no engine configured"})


# ---------------------------------------------------------------------------
# Engine v2 endpoints — managed llama-server subprocess
# ---------------------------------------------------------------------------

_V2_NOT_ENABLED = "Managed engine not enabled (set AUGMENTUM_ENGINE_MANAGED=true)"


def _get_llama_manager(request: Request):
    """Return LlamaServerManager or raise 404."""
    mgr = getattr(request.app.state, "llama_manager", None)
    if mgr is None:
        raise HTTPException(status_code=404, detail=_V2_NOT_ENABLED)
    return mgr


def _mgr_hosts_model(mgr, model: str) -> bool:
    """True when ``mgr`` currently serves or is loading ``model`` — matched by
    its live ``model_id`` or an in-flight ``_load_progress`` entry."""
    if mgr is None:
        return False
    if getattr(mgr, "model_id", "") == model:
        return True
    load_prog = getattr(mgr, "_load_progress", None)
    return isinstance(load_prog, dict) and load_prog.get("model_id") == model


def _resolve_engine_manager_for_model(request: Request, model: str):
    """Return the engine manager that hosts ``model`` — the secondary slot
    ("Slot B") when pinned/loaded/loading there, the managed classifier slot
    ("Slot C") when loaded/loading there, else the primary engine.

    Used by the progress pollers so a cold load INTO a dedicated slot is
    watched on the RIGHT engine. Without this they'd read the primary's (idle)
    snapshot, never see the slot's load progress, and the chat stall banner
    would fire during a normal Slot B/C load. Returns ``None`` only when no
    engine exists.
    """
    primary = getattr(request.app.state, "llama_manager", None)
    model = (model or "").strip()
    if not model:
        return primary

    # Slot B — reachable by an explicit registry PIN as well as by live state.
    secondary = getattr(request.app.state, "secondary_slot", None)
    sec_mgr = getattr(secondary, "manager", None) if secondary else None
    if sec_mgr is not None:
        from augmentum.models.secondary_slot import SECONDARY_BACKEND_KEY

        registry = getattr(request.app.state, "provider_registry", None)
        getter = getattr(registry, "pinned_backend_for", None) if registry else None
        if callable(getter) and getter(model) == SECONDARY_BACKEND_KEY:
            return sec_mgr
        if _mgr_hosts_model(sec_mgr, model):
            return sec_mgr

    # Slot C — role-routed, never pinned/in the catalog map, so it's matched
    # purely on live state (loaded id or in-flight load progress).
    classifier = getattr(request.app.state, "classifier_slot", None)
    cls_mgr = getattr(classifier, "manager", None) if classifier else None
    if _mgr_hosts_model(cls_mgr, model):
        return cls_mgr

    return primary


def _extract_engine_load_options(body: dict, mgr) -> dict:
    """Normalize optional per-load engine tuning fields."""
    load_options = {}
    for key in (
        "ctx_size",
        "gpu_layers_mode",
        "gpu_layers",
        # MoE expert-offload layer count — used when gpu_layers_mode is
        # 'moe_first_n_cpu'. Ignored otherwise. See build_load_plan for
        # the full mode set + auto-promotion behaviour for MoE models.
        # The UI omits this key entirely when the form input is blank
        # (signal: "no opinion, let backend autofit pick"). An explicit
        # ``0`` from the form is preserved end-to-end and means "all
        # experts on GPU" — valid for tiny MoEs where the full expert
        # pool fits in VRAM.
        "moe_cpu_layers",
        # CPU thread pool. ``0`` defers to llama-server's default
        # (half the available hardware threads). Decode + prefill
        # pools are separate so prefill can use more threads on
        # hybrid CPUs without starving generation.
        "cpu_threads",
        "cpu_threads_batch",
        # Pin resident weights so the OS can't swap them out — useful
        # for partial-offload runs over long sessions.
        "mlock",
        # V-cache quantization override. When empty the V cache uses
        # the same type as K; set explicitly for the q4_0 K + q8_0 V
        # pattern. ``kv_cache_type`` (above) still controls K.
        "kv_cache_type_v",
        # LoRA hot-load: absolute path + optional scale weight.
        "lora_model",
        "lora_scale",
        # Sampler seed. Negative = random per request.
        "seed",
        # Multi-GPU placement (no-op on single-GPU hosts; the UI hides
        # the section based on plan.memory.gpu_count).
        "tensor_split",
        "main_gpu",
        "split_mode",
        "batch_size",
        # Physical batch (--ubatch-size). 0/omitted = server default (512).
        # Dominant prefill lever for CPU-MoE offload models (DeepSeek V4,
        # Qwen3.5-A10B class) — see build_load_plan's ubatch_size doc.
        "ubatch_size",
        "kv_cache_type",
        "flash_attn",
        "idle_timeout",
        "draft_max",
        "draft_ctx_size",
        "draft_gpu_layers",
        "draft_min",
        "draft_p_min",
        # MTP self-speculation per-load overrides. Omit to fall back to
        # the engine-wide ``engine_mtp_enabled`` / ``engine_mtp_n_max``
        # settings. Setting these per-load lets a user enable MTP only
        # on the GGUFs that actually have the heads (instead of toggling
        # the global flag on/off as they swap models).
        "mtp_enabled",
        "mtp_n_max",
        # Chat template overrides (see _build_cli_args). Mode is one of
        # embedded|builtin|custom; content is the raw Jinja template string
        # used only when mode=custom; reasoning_format overrides the global
        # engine_reasoning_format; chat_template_kwargs is a JSON string
        # forwarded to --chat-template-kwargs (e.g. {"enable_thinking": false}
        # for GLM-4.7-Flash whose distilled training drops the closing
        # </think> tag and leaks reasoning into the visible response).
        "chat_template_mode",
        "chat_template_content",
        "chat_template_kwargs",
        "reasoning_format",
        # Vision projector — when set, llama-server is launched with
        # --mmproj <path>, enabling multimodal chat completions. Absolute
        # path to the mmproj/clip GGUF that pairs with the loaded model.
        "mmproj_path",
        # Per-load vision toggle. True forces mmproj auto-pair even when
        # the global engine_auto_pair_mmproj is False; False suppresses
        # mmproj entirely for this load (text-only, KV restore works);
        # omitting the key falls back to the global setting. Explicit
        # ``mmproj_path`` above always wins.
        "vision_mode",
    ):
        if key in body:
            load_options[key] = body.get(key)

    if "draft_model" in body:
        draft = body.get("draft_model", "") or ""
        if draft and not os.path.isabs(draft):
            resolved_draft = mgr._resolve_model_path(draft)
            if resolved_draft is None:
                raise HTTPException(status_code=404, detail=f"Draft model not found: {draft}")
            draft = resolved_draft
        load_options["draft_model"] = draft

    return load_options


def _engine_load_config_changed(load_options: dict, mgr) -> bool:
    """True if ``load_options`` would change the VRAM-shaping config vs the
    currently-loaded model, so an already-resident model must still reload.

    Only the three fields the manager actually tracks + reports as the
    applied config (``ctx_size`` / ``gpu_layers`` / ``gpu_layers_mode``, see
    ``LlamaServerManager.status``) drive the reload decision. A key that is
    absent from the request means "no opinion" and never forces a reload.
    Conservative by construction: it errs toward reporting a change (=>
    reload, today's behaviour) rather than a false "unchanged" that would
    skip a reload the user actually wanted. The point is only to catch the
    common re-select-the-resident-model case, where the saved profile
    replays exactly the applied ctx/GPU layout.
    """
    if not load_options:
        return False
    current = {
        "ctx_size": getattr(mgr, "current_ctx_size", None),
        "gpu_layers": getattr(mgr, "current_gpu_layers", None),
        "gpu_layers_mode": getattr(mgr, "current_gpu_layers_mode", None),
    }
    for key, cur in current.items():
        if key not in load_options or load_options[key] in (None, ""):
            continue
        req = load_options[key]
        # numeric fields: coerce so 8192 == "8192" and 33 == 33.0
        if key in ("ctx_size", "gpu_layers"):
            try:
                if int(req) != int(cur if cur is not None else -1):
                    return True
            except (TypeError, ValueError):
                if str(req) != str(cur):
                    return True
        else:
            if str(req) != str(cur if cur is not None else ""):
                return True
    return False


@engine_router.get("/v2/status")
async def engine_v2_status(request: Request) -> JSONResponse:
    """Return managed llama-server status."""
    mgr = _get_llama_manager(request)
    try:
        # mgr.status() shells out to nvidia-smi (5s timeout) — keep it
        # off the event loop so the polling UI can't stall request handling.
        return JSONResponse(await asyncio.to_thread(mgr.status))
    except Exception as exc:
        log.warning("engine_v2_status_error", error=str(exc))
        return JSONResponse({"state": "error", "error": str(exc)})


@engine_router.get("/v2/prefill_progress")
async def engine_v2_prefill_progress(request: Request) -> JSONResponse:
    """Return the most-recent prefill progress snapshot.

    Updated by the manager's status parser whenever llama-server emits
    a ``slot print_timing: ... prompt processing, n_tokens = X,
    progress = Y, ...`` log line. The frontend polls this during the
    "Preparing context…" prefill stage to render a live progress bar
    that tells the user how close they are to TTFT — long-context
    narrative turns spend 30-180s here with no other visible signal.

    Returns ``{"active": false}`` when no progress has been observed
    or the last snapshot is older than ``stale_after_s`` (default 8s,
    well past the upstream emit interval).

    Pass ``?model=<id>`` so a model resident in the secondary slot is
    watched on its own engine rather than the primary.
    """
    from augmentum.models.load_progress import build_prefill_progress_payload

    model = request.query_params.get("model", "")
    mgr = _resolve_engine_manager_for_model(request, model)
    snapshot = getattr(mgr, "_prefill_progress", None) if mgr else None
    payload = build_prefill_progress_payload(snapshot)
    if not payload.get("active"):
        # Cross-peer fallback: a fabric-routed model has no local
        # snapshot. Surface the peer's prefill snapshot (pushed into the
        # coordinator cache by the FabricBackend dispatch) through this
        # same endpoint so the existing poller renders it unchanged.
        peer = _peer_prefill_progress(request, model)
        if peer is not None:
            return JSONResponse(peer)
    return JSONResponse(payload)


@engine_router.get("/v2/load_progress")
async def engine_v2_load_progress(request: Request) -> JSONResponse:
    """Return the most-recent model-load progress snapshot.

    Seeded by ``LlamaServerManager.start()`` with an ``expected_s``
    derived from the median of recent successful loads for the model
    (or a coarse file-size heuristic on first load). The frontend
    polls this during a ``stage_start: model_load`` stage to render a
    progress bar — turns the "Loading model · 47s of ~30s" wait into
    a comprehensible event instead of looking like the stream stalled.

    Returns ``{"active": false}`` when no load is in flight.
    Progress is capped at 95% until the manager actually transitions
    to READY (and clears the snapshot) so the bar doesn't claim
    100% while llama-server is still warming up.

    Pass ``?model=<id>`` so a cold load INTO the secondary slot is watched
    on its own engine rather than the primary (which would look idle and
    trip the stall banner).
    """
    from augmentum.models.load_progress import build_load_progress_payload

    model = request.query_params.get("model", "")
    mgr = _resolve_engine_manager_for_model(request, model)
    snapshot = getattr(mgr, "_load_progress", None) if mgr else None
    payload = build_load_progress_payload(snapshot)
    if not payload.get("active"):
        # Cross-peer fallback: a fabric-routed model loads on a peer and
        # has no local snapshot. The FabricBackend dispatch records the
        # peer's load progress into the coordinator cache; surface it
        # through this same endpoint so the existing load-progress.js
        # poller renders a peer load with byte-identical UX.
        peer = _peer_load_progress(request, model)
        if peer is not None:
            return JSONResponse(peer)
    return JSONResponse(payload)


def _fabric_coordinator(request: Request):
    """The fabric coordinator, or None when fabric is off / not started."""
    return getattr(request.app.state, "fabric_coordinator", None)


def _peer_load_progress(request: Request, model: str) -> dict | None:
    """Cross-peer model-load snapshot from the coordinator cache, already
    in wire shape, or None when there's nothing fresh for ``model``."""
    model = (model or "").strip()
    if not model:
        return None
    coord = _fabric_coordinator(request)
    getter = getattr(coord, "peer_load_progress", None) if coord else None
    return getter(model) if callable(getter) else None


def _peer_prefill_progress(request: Request, model: str) -> dict | None:
    """Cross-peer prefill snapshot from the coordinator cache, or None."""
    model = (model or "").strip()
    if not model:
        return None
    coord = _fabric_coordinator(request)
    getter = getattr(coord, "peer_prefill_progress", None) if coord else None
    return getter(model) if callable(getter) else None


@engine_router.get("/v2/models")
async def engine_v2_models(request: Request) -> JSONResponse:
    """Discover GGUF models + cached profiles."""
    mgr = _get_llama_manager(request)
    try:
        models = await mgr.discover_models()
    except Exception as exc:
        log.warning("engine_v2_discover_error", error=str(exc))
        models = []
    return JSONResponse({"models": models, "model_dirs": mgr.model_dirs})


@engine_router.post("/v2/models/plan")
async def engine_v2_plan(request: Request) -> JSONResponse:
    """Preview the built-in engine load plan for a model."""
    mgr = _get_llama_manager(request)
    body = await request.json()
    model_path = body.get("model_path") or body.get("model", "")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path or model required")

    if not os.path.isabs(model_path):
        resolved = mgr._resolve_model_path(model_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        model_path = resolved

    try:
        # build_load_plan() is sync + does subprocess.run(nvidia-smi, timeout=5)
        # inside its load-plan helpers when the GPU info cache is stale (TTL
        # 2s on the manager — every Load Setup open potentially refreshes
        # it). Wrapping in to_thread keeps that off the event loop so a
        # cache-miss can't stall every concurrent request for ~100-500ms
        # (or up to 5s in pathological cases). This was the source of the
        # ``event_loop_stall lag_s=3.81…6.74`` warnings observed alongside
        # heavy UI activity + concurrent GGUF downloads (2026-05-18).
        opts = _extract_engine_load_options(body, mgr)
        plan = await asyncio.to_thread(mgr.build_load_plan, model_path, load_options=opts)
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("engine_v2_plan_error", error=str(exc), model=model_path, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not build load plan: {exc}")

    return JSONResponse(plan)


@engine_router.post("/v2/models/load")
async def engine_v2_load(request: Request) -> JSONResponse:
    """Load or swap a model on the managed llama-server."""
    mgr = _get_llama_manager(request)
    body = await request.json()
    model_path = body.get("model_path") or body.get("model", "")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path or model required")

    # Resolve relative names to absolute paths
    if not os.path.isabs(model_path):
        resolved = mgr._resolve_model_path(model_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        model_path = resolved

    # Optional per-load spec-decode config (ephemeral — not persisted to settings).
    # Pass "" to explicitly disable; omit to keep whatever's configured.
    load_options = _extract_engine_load_options(body, mgr)

    from augmentum.models.llama_server_manager import ProcessState

    # Idempotent load: if the requested model is ALREADY resident with the
    # same VRAM-shaping config, don't stop()+start() it — that full unload/
    # reload can take up to minutes. This is the common case when the user
    # was temporarily on an API/cloud model (which leaves the local Slot-A
    # model loaded) and switches the primary chat model back through the
    # model-manager UI. The chat path (llama_cpp._ensure_server_locked) has
    # always skipped a same-model swap; this endpoint did not, so the Load
    # button forced a needless reload. A genuine change to ctx/gpu options
    # still falls through to swap() and reloads.
    already_loaded = (
        mgr.state == ProcessState.READY
        and bool(mgr.model_path)
        and os.path.realpath(model_path) == os.path.realpath(mgr.model_path)
        and not _engine_load_config_changed(load_options, mgr)
    )

    try:
        if already_loaded:
            log.info("engine_v2_load_noop_already_resident", model=mgr.model_id)
        elif mgr.state == ProcessState.READY:
            await mgr.swap(model_path, load_options=load_options)
        else:
            await mgr.start(model_path, load_options=load_options)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Model load timed out: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Model load failed: {exc}")
    except Exception as exc:
        log.error("engine_v2_load_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    # Persist the user's chosen options as the install-wide engine default
    # for this model, so a later lazy-load (after unload, idle timeout, or
    # crash) replays the same ctx/GPU layout instead of regressing to the
    # autofit-derived fallback.
    if load_options:
        await mgr.persist_load_options(mgr.model_id, load_options)

    request.app.state.provider_registry.invalidate_model_map()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info("engine_v2_model_loaded", model=model_path)
    return JSONResponse({"status": "loaded", "model_path": model_path, "model_id": mgr.model_id})


@engine_router.get("/v2/models/last-load")
async def engine_v2_last_load(request: Request, model: str = Query(...)) -> JSONResponse:
    """Return the load_options most recently used for a model.

    The engine load route auto-persists each successful load to
    ``app_settings["engine.last_load.<model_id>"]`` via
    ``LlamaServerManager.persist_load_options``. The Load Setup sheet
    polls this so it can pre-fill the form from "what I last ran"
    when the user hasn't explicitly clicked Save Default — without it,
    the per-load MTP toggle (and other knobs not in the frontend's
    ``engineModelLoadProfiles`` map) appear to reset on every open
    even though the backend remembers them and replays them on the
    next lazy load.
    """
    mgr = _get_llama_manager(request)
    model_id = (model or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")
    # _resolve_model_path expects a filename / stem; accept either an
    # absolute path or a bare name and normalize to the model_id the
    # manager uses as the storage key.
    if os.path.isabs(model_id):
        from pathlib import Path
        model_id = Path(model_id).stem
    saved = await mgr._load_saved_options(model_id)
    return JSONResponse({"model_id": model_id, "load_options": saved or {}})


@engine_router.post("/v2/models/unload")
async def engine_v2_unload(request: Request) -> JSONResponse:
    """Stop the managed llama-server."""
    mgr = _get_llama_manager(request)
    try:
        await mgr.stop()
    except Exception as exc:
        log.warning("engine_v2_unload_error", error=str(exc))
    request.app.state.provider_registry.invalidate_model_map()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info("engine_v2_model_unloaded")
    return JSONResponse({"status": "unloaded"})


# ---------------------------------------------------------------------------
# Secondary local engine ("Slot B") — a second resident user-chosen model
# ---------------------------------------------------------------------------


def _get_secondary_slot(request: Request):
    """Return the SecondarySlot or raise 404 when the feature is off."""
    slot = getattr(request.app.state, "secondary_slot", None)
    if slot is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Secondary model slot is not enabled. Turn on "
                "engine_secondary_enabled to load a second resident model."
            ),
        )
    return slot


@engine_router.post("/v2/secondary/load")
async def engine_secondary_load(request: Request) -> JSONResponse:
    """Load (or swap) a model into the secondary slot and pin it.

    The model becomes a second resident process alongside the primary
    engine. Chat requests for it route to the slot via the pin (no swap
    on the primary). Per-model load config (gpu-layer cap, ctx, idle
    timeout) travels with the model — when the body carries no explicit
    options, the slot replays the model's saved defaults.
    """
    slot = _get_secondary_slot(request)
    body = await request.json()
    model = body.get("model_path") or body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="model_path or model required")

    resolved = slot.resolve_model_path(model)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")

    # Admission control — a second resident model shares the GPU with the
    # primary. Consult the resource ledger (accumulated per-model VRAM
    # footprint vs live free VRAM) and reject up front with a clear message
    # instead of letting llama-server OOM-crash the subprocess. ``force``
    # bypasses for users who know they want partial offload / to evict.
    if not body.get("force"):
        ledger = getattr(request.app.state, "resource_ledger", None)
        if ledger is not None:
            try:
                size_bytes = os.path.getsize(resolved)
            except OSError:
                size_bytes = 0
            model_stem = os.path.splitext(os.path.basename(resolved))[0]
            ok, reason, needed_mb, free_mb = await ledger.check_engine_fit(
                model_stem, size_bytes=size_bytes,
            )
            if not ok:
                return JSONResponse(
                    {
                        "error": "insufficient_vram",
                        "detail": (
                            f"Won't fit in Slot B — {reason}. Stop another "
                            "model to free VRAM, or retry with force to try "
                            "anyway (it may run partly on CPU or fail to load)."
                        ),
                        "needed_mb": needed_mb,
                        "free_mb": free_mb,
                    },
                    status_code=507,
                )

    # Explicit per-load tuning (optional). Empty dict → fall through to the
    # model's saved defaults inside slot.load().
    load_options = _extract_engine_load_options(body, slot.manager) or None

    try:
        model_id = await slot.load(resolved, load_options=load_options)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Model load timed out: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Model load failed: {exc}") from exc
    except Exception as exc:
        log.error("engine_secondary_load_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    from augmentum.models.secondary_slot import SECONDARY_BACKEND_KEY

    registry = request.app.state.provider_registry
    registry.pin_model(model_id, SECONDARY_BACKEND_KEY)
    registry.invalidate_model_map()

    # Persist the loaded model so its picker entry re-pins to the slot
    # after a restart (the model itself lazy-loads on first use).
    object.__setattr__(settings, "engine_secondary_model", model_id)
    store = getattr(request.app.state, "settings_store", None)
    if store:
        await store.set("engine_secondary_model", model_id)

    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info("engine_secondary_model_loaded", model=model_id)
    return JSONResponse({"status": "loaded", "model_id": model_id, "slot": "secondary"})


@engine_router.post("/v2/secondary/unload")
async def engine_secondary_unload(request: Request) -> JSONResponse:
    """Stop the secondary slot subprocess, drop its pin, free its VRAM/RAM."""
    slot = _get_secondary_slot(request)
    mgr = slot.manager
    pinned_model = mgr.model_id if mgr else ""

    await slot.unload()

    registry = request.app.state.provider_registry
    if pinned_model:
        registry.unpin_model(pinned_model)
    else:
        registry.unpin_model()  # belt-and-suspenders: clear any slot pin
    registry.invalidate_model_map()

    object.__setattr__(settings, "engine_secondary_model", "")
    store = getattr(request.app.state, "settings_store", None)
    if store:
        await store.set("engine_secondary_model", "")

    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info("engine_secondary_model_unloaded", model=pinned_model or None)
    return JSONResponse({"status": "unloaded"})


@engine_router.get("/v2/secondary/status")
async def engine_secondary_status(request: Request) -> JSONResponse:
    """Combined status for the two-slot resource view.

    Returns the secondary slot's status plus the primary engine's, so the
    model manager can show both footprints and the shared free VRAM in one
    place when deciding whether a second model fits.
    """
    slot = getattr(request.app.state, "secondary_slot", None)
    primary = getattr(request.app.state, "llama_manager", None)
    out: dict = {"enabled": slot is not None}
    if slot is not None:
        out["secondary"] = slot.status()
    if primary is not None:
        # Primary status carries the authoritative gpu.vram_free_mib the
        # UI uses to show headroom for a second model.
        out["primary"] = primary.status()
    return JSONResponse(out)


def _get_classifier_slot(request: Request):
    """Return the managed ClassifierSlot ("Slot C") or raise 404 when off."""
    slot = getattr(request.app.state, "classifier_slot", None)
    if slot is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Managed classifier slot is not enabled. Turn on "
                "classifier_slot_enabled to load a swappable classifier model."
            ),
        )
    return slot


@engine_router.post("/v2/classifier/load")
async def engine_classifier_load(request: Request) -> JSONResponse:
    """Load (or swap) the model in the managed classifier slot ("Slot C").

    Unlike Slot B (which pins an arbitrary chat model), Slot C serves the
    classifier/utility (and, when VL+mmproj, vision) ROLES, so it registers
    under the ``"classifier"`` backend key and the role resolver returns its
    hosted id via ``classifier_sidecar_model`` — which we sync here. A swap
    briefly stops/starts the subprocess; during that window the classifier
    role falls through to primary (never blocks the 2.5s voice budget).
    """
    slot = _get_classifier_slot(request)
    body = await request.json()
    model = body.get("model_path") or body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="model_path or model required")

    resolved = slot.resolve_model_path(model)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")

    from augmentum.models.classifier_slot import CLASSIFIER_BACKEND_KEY

    # Takeover gate: on external-sidecar installs (compose.classifier.yaml /
    # AUGMENTUM_CLASSIFIER_BASE_URL) the "classifier" key is held by the
    # env-configured Docker container — model/ctx/mmproj frozen at container
    # create. Loading Slot C over it is a real routing change, so it needs an
    # explicit user opt-in (take_over=true; the UI confirms). The displaced
    # external backend is stashed and restored on unload.
    registry = request.app.state.provider_registry
    existing = registry._backends.get(CLASSIFIER_BACKEND_KEY)
    displacing_external = existing is not None and existing is not slot.backend
    if displacing_external and not body.get("take_over"):
        return JSONResponse(
            {
                "error": "external_classifier_active",
                "take_over_required": True,
                "detail": (
                    "The classifier role is currently served by the external "
                    "Docker sidecar (env-configured). Loading this model takes "
                    "over classification in-app — the sidecar container is "
                    "stopped to free its VRAM and resumed when you unload the "
                    "slot."
                ),
            },
            status_code=409,
        )

    # Admission control when offloading to GPU — mirror Slot B.
    if not body.get("force"):
        ledger = getattr(request.app.state, "resource_ledger", None)
        wants_gpu = int(body.get("gpu_layers", settings.classifier_slot_gpu_layers) or 0) > 0
        if ledger is not None and wants_gpu:
            try:
                size_bytes = os.path.getsize(resolved)
            except OSError:
                size_bytes = 0
            model_stem = os.path.splitext(os.path.basename(resolved))[0]
            ok, reason, needed_mb, free_mb = await ledger.check_engine_fit(
                model_stem, size_bytes=size_bytes,
            )
            if not ok:
                return JSONResponse(
                    {
                        "error": "insufficient_vram",
                        "detail": (
                            f"Won't fit in the classifier slot — {reason}. Free "
                            "VRAM or retry with force (may run partly on CPU)."
                        ),
                        "needed_mb": needed_mb,
                        "free_mb": free_mb,
                    },
                    status_code=507,
                )

    # Resident slot: keep idle_timeout=0 unless the caller overrides.
    load_options = _extract_engine_load_options(body, slot.manager) or {}
    load_options.setdefault("idle_timeout", 0)

    try:
        model_id = await slot.load(resolved, load_options=load_options)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Model load timed out: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Model load failed: {exc}") from exc
    except Exception as exc:
        log.error("classifier_slot_load_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    # Register (or take over) the key. When displacing the external sidecar,
    # stash it so unload can restore the pre-takeover registration.
    if displacing_external:
        request.app.state.classifier_external_backend = existing
        log.info("classifier_slot_took_over_external")
    if slot.backend is not None and registry._backends.get(CLASSIFIER_BACKEND_KEY) is not slot.backend:
        registry.register_backend(CLASSIFIER_BACKEND_KEY, slot.backend)
    # Keep the slot out of EVERY catalog listing (parity with Slot B / the
    # boot path). The slot's backend shares model_dirs with the primary engine,
    # so its list_models() returns the full GGUF catalog — un-excluded, every
    # name collides into name@engine / name@classifier and the bare names drop
    # out of the map, blanking the library/picker (0 models per drive) until a
    # restart. Idempotent.
    #
    # Exclusion alone is NOT enough, and the previous version of this comment
    # ("role resolution reaches the backend by key, not via the map, so
    # exclusion never breaks routing") was only half true. It holds for the
    # branch where ``classifier_model`` is blank — that one reads
    # ``_backends["classifier"]`` directly. But the moment the user NAMES a
    # classifier model, ``resolve_model_for_role`` resolves it BY NAME through
    # the catalog map, and the excluded slot isn't in the map, so the only
    # backend advertising that GGUF is the primary engine (Slot C shares its
    # model_dirs). The role then lands on Slot A and forces a swap — and with
    # the primary role on a different model, the two lanes ping-pong the engine
    # every few seconds (live-observed 2026-07-26: 12b ↔ E4B, ~5s reload each,
    # while Slot C sat idle on 8093 already serving the right model).
    #
    # The pin is what closes it, exactly as Slot B does: exclusion hides the
    # slot's ~400-name shared catalog, the pin re-adds the ONE name it actually
    # serves and routes that name to the slot. ``refresh_model_map`` re-injects
    # pins last, so this survives every subsequent probe.
    registry.exclude_backend_from_map(CLASSIFIER_BACKEND_KEY)
    if model_id:
        registry.pin_model(model_id, CLASSIFIER_BACKEND_KEY)
    registry.invalidate_model_map()
    await registry.refresh_model_map(force=True)

    # Coordinate the two classifier serving paths: now that Slot C holds the
    # role, STOP the displaced external container so it doesn't sit resident
    # holding a duplicate model's VRAM (the "third model after unpause" class).
    # Remember its name so unload can resume it (restore the fallback path).
    # Done AFTER registration so there's no window where the role is unserved.
    if displacing_external:
        try:
            from augmentum.resource.container_probe import (
                find_sidecar_container,
                set_container_paused,
            )

            cname = await find_sidecar_container(request.app.state, "classifier")
            if cname:
                ok, err = await set_container_paused(
                    request.app.state, cname, paused=True,
                )
                if ok:
                    request.app.state.classifier_external_container = cname
                    log.info("classifier_external_container_stopped_on_takeover", container=cname)
                else:
                    log.warning(
                        "classifier_external_container_stop_failed",
                        container=cname, error=err,
                    )
        except Exception:
            log.warning("classifier_external_container_stop_error", exc_info=True)

    # Persist + make the role resolver return the slot's real hosted id.
    store = getattr(request.app.state, "settings_store", None)
    object.__setattr__(settings, "classifier_slot_model", model_id)
    object.__setattr__(settings, "classifier_sidecar_model", model_id)
    if store:
        await store.set("classifier_slot_model", model_id)
        await store.set("classifier_sidecar_model", model_id)

    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info(
        "classifier_slot_model_loaded",
        model=model_id, vision=slot.is_vision_capable(),
    )
    return JSONResponse({
        "status": "loaded", "model_id": model_id, "slot": "classifier",
        "vision_capable": slot.is_vision_capable(),
    })


async def _probe_backend_reachable(backend, *, timeout: float = 2.0) -> tuple[str, bool]:
    """Best-effort ``(served_model_id, reachable)`` for a backend via
    ``list_models`` under a short timeout.

    Reachable AND serving a model → ``(name, True)``. Unreachable, timed out, or
    reachable-but-serving-nothing → ``("", False)`` — we treat "serves nothing"
    as not-ready so the caller never routes the classifier role to a backend
    that would fail the next request. Never raises.
    """
    try:
        models = await asyncio.wait_for(backend.list_models(), timeout=timeout)
    except Exception:  # noqa: BLE001 — any failure = not reachable
        return "", False
    if not models:
        return "", False
    return (getattr(models[0], "name", "") or ""), True


@engine_router.post("/v2/classifier/unload")
async def engine_classifier_unload(request: Request) -> JSONResponse:
    """Stop the classifier slot subprocess and free its VRAM/RAM.

    The classifier/utility roles fall back to primary afterward (existing
    tier-3). Only removes the ``"classifier"`` backend if it IS this slot's
    backend (never deletes an external Docker classifier registration).

    When handing the role back to the external sidecar, verifies the sidecar is
    actually reachable first and syncs ``classifier_sidecar_model`` to its served
    model — never leaves the key pointing at a dead body or paired with the
    unloaded slot's model name (2026-07-16 fix)."""
    slot = _get_classifier_slot(request)
    # Capture the pinned name BEFORE unload — ``slot.unload()`` clears the
    # manager's model_id, and without the name we can only clear ALL pins,
    # which would also drop Slot B's (parity with the secondary unload route).
    _mgr = slot.manager
    pinned_model = (getattr(_mgr, "model_id", "") or "") if _mgr else ""
    await slot.unload()

    from augmentum.models.classifier_slot import CLASSIFIER_BACKEND_KEY
    from augmentum.resource.container_probe import (
        find_sidecar_container,
        set_container_paused,
    )

    registry = request.app.state.provider_registry
    restored_external = False
    # ``None`` = leave classifier_sidecar_model untouched (we didn't change the
    # key). Any string (incl. "") = write it in lockstep with the key change so
    # the role never resolves a backend paired with a stale model name (Bug A,
    # 2026-07-16: unload left classifier_sidecar_model = the just-unloaded slot's
    # model while the key reverted to the external backend).
    new_sidecar_model: str | None = None
    displaced = getattr(request.app.state, "classifier_external_backend", None)
    if registry._backends.get(CLASSIFIER_BACKEND_KEY) is slot.backend:
        if displaced is not None:
            # Handing the role back to the external sidecar. Before pointing the
            # key at it, make SURE it's actually reachable — a dead container
            # here is the 2026-07-16 breakage (key -> corpse; every classify
            # fails with "Name or service not known"). Best-effort resume its
            # container first (the takeover-stop stash is unreliable, so also
            # look it up fresh), then health-probe.
            cname = (
                getattr(request.app.state, "classifier_external_container", "")
                or await find_sidecar_container(request.app.state, "classifier")
            )
            resumed = False
            if cname:
                try:
                    await set_container_paused(request.app.state, cname, paused=False)
                    resumed = True
                    log.info("classifier_external_container_resumed_on_unload", container=cname)
                except Exception:
                    log.warning("classifier_external_container_resume_error", exc_info=True)
            # Health-probe the external before routing to it. If we just resumed a
            # stopped container, give it a BOUNDED window to cold-load (a
            # llama-server needs several seconds to serve /v1/models) before
            # giving up — otherwise a transient stop would fail open permanently
            # (there's no reconcile loop yet). Already-running → first probe
            # returns instantly, no delay.
            attempts = 3 if resumed else 1
            ext_model, ext_ok = "", False
            for _i in range(attempts):
                ext_model, ext_ok = await _probe_backend_reachable(displaced, timeout=1.5)
                if ext_ok or _i == attempts - 1:
                    break
                await asyncio.sleep(1.5)
            if ext_ok:
                registry.register_backend(CLASSIFIER_BACKEND_KEY, displaced)
                request.app.state.classifier_external_backend = None
                request.app.state.classifier_external_container = None
                restored_external = True
                # Sync the sidecar model NAME to what the external actually
                # serves (Bug A) — never leave it as the unloaded slot's model.
                new_sidecar_model = ext_model or ""
                log.info("classifier_slot_restored_external", model=new_sidecar_model)
            else:
                # External unreachable even after a resume attempt — do NOT route
                # the classifier role to a corpse. Drop the key so it fails open
                # to the utility/primary tier (graceful degradation). Keep the
                # backend stash so a later manual container start + reload can
                # still restore it.
                registry._backends.pop(CLASSIFIER_BACKEND_KEY, None)
                new_sidecar_model = ""
                log.warning(
                    "classifier_slot_external_unreachable_on_restore",
                    container=cname or "",
                    hint="dropped classifier key; role fails open to primary until "
                         "the external sidecar is reachable again",
                )
        else:
            registry._backends.pop(CLASSIFIER_BACKEND_KEY, None)
            new_sidecar_model = ""
    # Drop the load-time routing pin. Unconditional (not inside the branches
    # above) because the pin was set whenever the slot loaded, regardless of
    # who ends up holding the key afterwards — leaving it would route the name
    # at a stopped subprocess. Targeted by name so Slot B's pin survives.
    if pinned_model:
        registry.unpin_model(pinned_model)
    registry.invalidate_model_map()

    store = getattr(request.app.state, "settings_store", None)
    object.__setattr__(settings, "classifier_slot_model", "")
    if store:
        await store.set("classifier_slot_model", "")
    if new_sidecar_model is not None:
        object.__setattr__(settings, "classifier_sidecar_model", new_sidecar_model)
        if store:
            await store.set("classifier_sidecar_model", new_sidecar_model)

    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "llm")
    log.info("classifier_slot_model_unloaded")
    return JSONResponse({"status": "unloaded", "restored_external": restored_external})


@engine_router.get("/v2/classifier/status")
async def engine_classifier_status(request: Request) -> JSONResponse:
    """Status for the managed classifier slot (incl. vision capability and
    which backend currently serves the classifier role: the managed slot,
    the external Docker sidecar, or none)."""
    slot = getattr(request.app.state, "classifier_slot", None)
    out: dict = {"enabled": slot is not None}
    if slot is not None:
        out["classifier"] = slot.status()
        from augmentum.models.classifier_slot import CLASSIFIER_BACKEND_KEY

        registry = request.app.state.provider_registry
        holder = registry._backends.get(CLASSIFIER_BACKEND_KEY)
        if holder is None:
            out["serving"] = "none"
        elif holder is slot.backend:
            out["serving"] = "managed"
        else:
            out["serving"] = "external"
    return JSONResponse(out)


@engine_router.post("/v2/models/dirs")
async def engine_v2_add_dir(request: Request) -> JSONResponse:
    """Add a model directory, persist, and scan for profiles."""
    mgr = _get_llama_manager(request)
    body = await request.json()
    dir_path = body.get("path", "").strip()
    if not dir_path:
        raise HTTPException(status_code=400, detail="path required")
    if not os.path.isdir(dir_path):
        in_docker = os.path.exists("/.dockerenv")
        hint = " Use 'Host Directories' to mount host paths into the container." if in_docker else ""
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {dir_path}.{hint}",
        )

    if dir_path not in mgr.model_dirs:
        mgr.model_dirs.append(dir_path)

    # Persist to settings store
    store = getattr(request.app.state, "settings_store", None)
    if store:
        await store.set("engine_v2_extra_model_dirs", ";".join(mgr.model_dirs))

    # Scan new dir for profiles
    new_count = await mgr.scan_and_cache_profiles()
    log.info("engine_v2_dir_added", path=dir_path, new_profiles=new_count)
    return JSONResponse({"model_dirs": mgr.model_dirs, "new_profiles": new_count})


@engine_router.delete("/v2/models/dirs")
async def engine_v2_remove_dir(request: Request) -> JSONResponse:
    """Remove a model directory and persist."""
    mgr = _get_llama_manager(request)
    body = await request.json()
    dir_path = body.get("path", "").strip()
    if not dir_path:
        raise HTTPException(status_code=400, detail="path required")

    if dir_path in mgr.model_dirs:
        mgr.model_dirs.remove(dir_path)

    store = getattr(request.app.state, "settings_store", None)
    if store:
        await store.set("engine_v2_extra_model_dirs", ";".join(mgr.model_dirs))
    log.info("engine_v2_dir_removed", path=dir_path)
    return JSONResponse({"model_dirs": mgr.model_dirs})


_PLATFORM_CACHE: dict[str, str] | None = None


def _detect_host_platform() -> dict[str, str]:
    """Best-effort detection of the host environment from inside the container.

    The interesting axis for users is whether *bind mounts go through a
    translation layer*. On native Linux (or a WSL2-internal Linux distro
    when Docker daemon runs inside WSL2 itself) bind mounts use ext4 and
    deliver full disk speed. On Docker Desktop for Windows/Mac the bind
    mounts go through 9p / osxfs, which is roughly 10× slower for large
    sequential reads — exactly the access pattern of GGUF model loads.

    Heuristics, in order:
      1. /proc/version → "microsoft" or "WSL2"  → WSL kernel
      2. /run/desktop/  exists                  → Docker Desktop sentinel
      3. /.dockerenv exists, no WSL marker      → native Linux Docker
      4. neither                                 → bare metal Linux
    """
    global _PLATFORM_CACHE
    if _PLATFORM_CACHE is not None:
        return _PLATFORM_CACHE

    in_docker = os.path.exists("/.dockerenv")
    proc_version = ""
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as fh:
            proc_version = fh.read().lower()
    except OSError:
        pass

    is_wsl = ("microsoft" in proc_version) or ("wsl2" in proc_version)
    has_dd_sentinel = os.path.isdir("/run/desktop") or os.path.isdir("/run/host-services")

    if not in_docker:
        platform_id = "linux_native"
        label = "Native Linux"
        hint = ""
    elif is_wsl and has_dd_sentinel:
        platform_id = "docker_desktop_windows"
        label = "Docker Desktop on Windows"
        hint = (
            "Files on the Windows filesystem (C:, D:, etc.) cross the "
            "WSL2 9p bridge and load roughly 10× slower than native. "
            "For full speed, store GGUFs inside WSL — for example "
            "/home/<you>/models on the Linux side, then mount that as "
            "/data/host-models/<name> in compose.yaml."
        )
    elif is_wsl:
        platform_id = "wsl2_native"
        label = "WSL2 (Linux distro)"
        hint = (
            "Bind mounts to paths inside this WSL distro are full speed. "
            "Mounts to Windows drives (/mnt/c, /mnt/d, …) cross the 9p "
            "bridge and load ~10× slower."
        )
    elif has_dd_sentinel:
        platform_id = "docker_desktop_mac"
        label = "Docker Desktop on macOS"
        hint = (
            "Bind mounts to host paths go through osxfs/VirtioFS and "
            "load slower than native. For fastest model loads, use a "
            "named Docker volume or copy models into /data/models."
        )
    else:
        platform_id = "linux_docker"
        label = "Linux + Docker"
        hint = ""

    _PLATFORM_CACHE = {
        "id": platform_id,
        "label": label,
        "perf_hint": hint,
    }
    return _PLATFORM_CACHE


def _count_ggufs(path: str, max_depth: int = 2) -> int:
    """Count .gguf files in a directory (up to max_depth levels deep).

    Mirrors LlamaServerManager._dir_has_gguf but returns a count rather
    than a bool. Cheap on small dirs; for huge model libraries (~100s
    of files) the os.scandir is still sub-millisecond.
    """
    if not path or not os.path.isdir(path):
        return 0
    count = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file() and entry.name.lower().endswith(".gguf"):
                        count += 1
                    elif entry.is_dir() and not entry.name.startswith(".") and max_depth > 0:
                        count += _count_ggufs(entry.path, max_depth - 1)
                except OSError:
                    continue
    except OSError:
        return 0
    return count


@engine_router.get("/v2/models/dirs")
async def engine_v2_list_dirs(request: Request) -> JSONResponse:
    """List current model directories with per-row health + platform context.

    Response shape:
      model_dirs: list of {path, gguf_count, slow, exists}
      host_mounts: list of {host_path, container_path, gguf_count, slow, active}
        - active = container_path is currently in mgr.model_dirs (compose
          mount completed and Augmentum was restarted)
      platform: {id, label, perf_hint}
        - id is one of linux_native | wsl2_native | docker_desktop_windows |
          docker_desktop_mac | linux_docker
      in_docker: bool

    The shape changed from list-of-strings to list-of-objects to give the
    UI per-row signals (slow mount, GGUF count). The provider list at
    /api/providers/builtin still returns plain string lists for its own
    summary card — that's a separate endpoint.
    """
    mgr = _get_llama_manager(request)

    host_mounts_raw: list[dict] = []
    try:
        store = getattr(request.app.state, "settings_store", None)
        if store:
            raw = (await store.get("engine_v2_host_mounts")) or "[]"
            host_mounts_raw = json.loads(raw or "[]")
    except Exception:
        host_mounts_raw = []

    active_paths = set(mgr.model_dirs)
    # Map container path back to the host path that registered it. After
    # the user adds the volume to compose.yaml and restarts, the
    # container path appears in mgr.model_dirs — at that point we want
    # to show it as an active row annotated with its host source, not
    # double-list it under both Active and Pending.
    hm_by_container: dict[str, str] = {
        m.get("container_path", ""): m.get("host_path", "")
        for m in host_mounts_raw
        if m.get("container_path")
    }

    # Expose the actual filesystem type per directory so the UI can
    # build precise tooltips ("Mount type: 9p — …") instead of a
    # generic "host bridge" string. Empty string if mountinfo isn't
    # available. Imported lazily to avoid a top-level dependency on
    # the engine module from a route file shared with non-engine paths.
    from augmentum.models.llama_server_manager import classify_mount_fs

    def _annotate_dir(path: str) -> dict:
        # Synchronous filesystem work — runs on a worker thread via the
        # gather() below. ``_count_ggufs`` recursively scandir's the
        # directory; on a 9p / virtiofs / cifs mount that can take
        # seconds for a large model library. ``_is_slow_mount`` and
        # ``classify_mount_fs`` are typically cheap (just /proc reads)
        # but go off-loop with the rest for consistency.
        return {
            "path": path,
            "gguf_count": _count_ggufs(path),
            "slow": mgr._is_slow_mount(path),
            "exists": os.path.isdir(path),
            "host_source": hm_by_container.get(path, ""),
            "mount_fs": classify_mount_fs(path),
        }

    # Run the annotation off the event loop so a slow-mount FS walk
    # can't freeze chat / health checks / Files-tab polls. Parallel via
    # gather() so the worst case is the slowest single directory rather
    # than the sum across all directories — important when one mount is
    # a network share and another is fast local SSD.
    #
    # This route was producing 7-8s event-loop stalls (lag_s=7.78 on
    # 2026-05-09T23:30:47) when mgr.model_dirs included a slow 9p
    # bridge — every coroutine on the loop blocked for the full FS
    # walk. After this change the loop yields between every FS syscall.
    model_dirs = (
        list(await asyncio.gather(*(
            asyncio.to_thread(_annotate_dir, p) for p in mgr.model_dirs
        )))
        if mgr.model_dirs
        else []
    )

    # Pending mounts: registered but the bind mount hasn't materialized
    # yet (compose.yaml not edited, or augmentum not restarted).
    host_mounts = [
        {
            "host_path": m.get("host_path", ""),
            "container_path": m.get("container_path", ""),
        }
        for m in host_mounts_raw
        if m.get("container_path", "") not in active_paths
    ]

    return JSONResponse({
        "model_dirs": model_dirs,
        "host_mounts": host_mounts,
        "platform": _detect_host_platform(),
        "in_docker": os.path.exists("/.dockerenv"),
    })


@engine_router.post("/v2/models/host-mount")
async def engine_v2_add_host_mount(request: Request) -> JSONResponse:
    """Register a host directory mount for GGUF models.

    Persists the mount config in the database so the UI can display it.
    Returns the Docker volume mount line the user needs to add to their
    compose config (the container can't modify host-side compose files).

    Body: {"host_path": "D:/Models"} or {"host_path": "/home/user/models"}
    """
    _get_llama_manager(request)
    body = await request.json()
    host_path = body.get("host_path", "").strip()
    if not host_path:
        raise HTTPException(status_code=400, detail="host_path required")

    safe_name = os.path.basename(host_path.rstrip("/\\")).replace(" ", "_").lower() or "models"
    container_path = f"/data/host-models/{safe_name}"

    # Persist in settings DB
    import json as _json
    store = getattr(request.app.state, "settings_store", None)
    mounts = []
    if store:
        raw = (await store.get("engine_v2_host_mounts")) or "[]"
        try:
            mounts = _json.loads(raw or "[]")
        except Exception:
            mounts = []

    # Check duplicate
    if any(m["host_path"] == host_path for m in mounts):
        return JSONResponse({
            "host_mounts": mounts,
            "message": "Already registered",
            "restart_required": False,
        })

    mounts.append({"host_path": host_path, "container_path": container_path})
    if store:
        await store.set("engine_v2_host_mounts", _json.dumps(mounts))

        # Also save the container path as a model dir for after restart
        saved = (await store.get("engine_v2_extra_model_dirs")) or ""
        dirs = [d.strip() for d in saved.split(";") if d.strip()] if saved else []
        if container_path not in dirs:
            dirs.append(container_path)
            await store.set("engine_v2_extra_model_dirs", ";".join(dirs))

    # Generate the compose volume line for the user
    volume_line = f"      - {host_path}:{container_path}:ro"

    return JSONResponse({
        "host_mounts": mounts,
        "container_path": container_path,
        "volume_line": volume_line,
        "restart_required": True,
        "message": f"Add this line to your compose.yaml under augmentum > volumes, then restart:\n{volume_line}",
    })


@engine_router.delete("/v2/models/host-mount")
async def engine_v2_remove_host_mount(request: Request) -> JSONResponse:
    """Remove a registered host directory mount."""
    _get_llama_manager(request)
    body = await request.json()
    host_path = body.get("host_path", "").strip()

    import json as _json
    store = getattr(request.app.state, "settings_store", None)
    mounts = []
    if store:
        raw = (await store.get("engine_v2_host_mounts")) or "[]"
        try:
            mounts = _json.loads(raw or "[]")
        except Exception:
            mounts = []
        mounts = [m for m in mounts if m["host_path"] != host_path]
        await store.set("engine_v2_host_mounts", _json.dumps(mounts))

    return JSONResponse({
        "host_mounts": mounts,
        "restart_required": True,
    })


@engine_router.get("/v2/browse")
async def engine_v2_browse(request: Request, path: str = "/") -> JSONResponse:
    """Browse the filesystem for GGUF model directories.

    Returns directories and .gguf files at the given path.
    Restricted to prevent sensitive file exposure — only shows
    directories and .gguf files, nothing else.
    """
    _get_llama_manager(request)  # ensure engine is enabled

    # Normalize and validate path
    real_path = os.path.realpath(path)
    if not os.path.isdir(real_path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    dirs: list[dict] = []
    gguf_files: list[dict] = []

    try:
        for entry in sorted(os.scandir(real_path), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=True):
                    # Count GGUFs in this subdir (1 level) for hint
                    gguf_count = 0
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_file() and sub.name.lower().endswith(".gguf"):
                                gguf_count += 1
                    except (PermissionError, OSError):
                        pass
                    dirs.append({
                        "name": entry.name,
                        "path": entry.path,
                        "gguf_count": gguf_count,
                    })
                elif entry.is_file() and entry.name.lower().endswith(".gguf"):
                    stat = entry.stat()
                    gguf_files.append({
                        "name": entry.name,
                        "path": entry.path,
                        "size": stat.st_size,
                    })
            except (PermissionError, OSError):
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    # Build breadcrumb parts
    parts = real_path.replace("\\", "/").split("/")
    breadcrumbs = []
    for i, part in enumerate(parts):
        if not part and i == 0:
            breadcrumbs.append({"name": "/", "path": "/"})
            continue
        if not part:
            continue
        crumb_path = "/".join(parts[:i + 1]) or "/"
        breadcrumbs.append({"name": part, "path": crumb_path})

    return JSONResponse({
        "path": real_path,
        "breadcrumbs": breadcrumbs,
        "dirs": dirs,
        "files": gguf_files,
        "parent": os.path.dirname(real_path) if real_path != "/" else None,
    })


@engine_router.get("/v2/cache/stats")
async def engine_v2_cache_stats(request: Request) -> JSONResponse:
    """Return token count cache statistics."""
    cache = getattr(request.app.state, "token_count_cache", None)
    if cache is None:
        raise HTTPException(status_code=404, detail=_V2_NOT_ENABLED)
    stats = await cache.stats()
    return JSONResponse(stats)


@engine_router.post("/v2/sessions/pin")
async def engine_v2_pin_session(request: Request) -> JSONResponse:
    """Pin a session's KV cache so it's protected from eviction.

    Body: {"session_id": "fingerprint"} — the session fingerprint
    (system prompt hash) or the X-Augmentum-Session header value.
    """
    mgr = _get_llama_manager(request)
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    mgr.pin_session(session_id)
    return JSONResponse({"status": "pinned", "session_id": session_id})


@engine_router.post("/v2/sessions/unpin")
async def engine_v2_unpin_session(request: Request) -> JSONResponse:
    """Unpin a session, allowing its KV cache to be evicted normally."""
    mgr = _get_llama_manager(request)
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    mgr.unpin_session(session_id)
    return JSONResponse({"status": "unpinned", "session_id": session_id})


@engine_router.get("/v2/sessions/pinned")
async def engine_v2_pinned_sessions(request: Request) -> JSONResponse:
    """List currently pinned sessions."""
    mgr = _get_llama_manager(request)
    return JSONResponse({"pinned": list(mgr._pinned_sessions)})


@engine_router.post("/v2/models/cache")
async def engine_v2_cache_model(request: Request) -> JSONResponse:
    """Copy a host-mounted model to local container storage for faster loading.

    Body: {"model": "model-name-or-path"}
    Only needed for models on bind-mounted directories (/models/host/).
    Models already in /data/models/ are already fast.
    """
    mgr = _get_llama_manager(request)
    body = await request.json()
    model = body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="model required")

    path = mgr._resolve_model_path(model)
    if not path:
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")

    if not mgr._is_slow_mount(path):
        return JSONResponse({"status": "already_fast", "path": path})

    try:
        local = await mgr.cache_host_model(path)
        return JSONResponse({
            "status": "cached",
            "original": path,
            "cached": local,
            "size_gb": round(os.path.getsize(local) / 1e9, 1),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@engine_router.post("/v2/prewarm")
async def engine_v2_prewarm(request: Request) -> JSONResponse:
    """Pre-warm the KV cache with conversation context.

    Called while the user is typing so the prefix cache is hot
    before they send. The next real request matches the cached
    prefix and skips prefill.

    Body: {"messages": [{role, content}, ...]}
    """
    mgr = _get_llama_manager(request)
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    if not backend or not hasattr(backend, "prewarm_context"):
        raise HTTPException(status_code=404, detail="Engine backend not available")

    # prewarm_context returns the serving slot id (0 is a valid
    # success) or None on failure — test None, not truthiness.
    warmed_slot = await backend.prewarm_context(messages)
    return JSONResponse({"prewarmed": warmed_slot is not None})


@engine_router.post("/v2/kv/resume")
async def engine_v2_kv_resume(request: Request) -> JSONResponse:
    """Fire the KV resume ladder for a session the user just opened.

    Body: {"session_id": "<UI chat session id>"}

    Derives the same opaque ``kv_session_key`` the chat path uses and
    runs restore→replay→cold in the background — replay of a long
    context can take tens of seconds, exactly the reading/typing window
    this overlaps. Always 200: a cold or not-ready outcome is normal,
    never an error.
    """
    body = await request.json()
    session_id = str(body.get("session_id", "") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    from augmentum.proxy.session import derive_kv_session_key
    session_key = derive_kv_session_key(_request_user_id(request), session_id)

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    ladder = getattr(backend, "resume_ladder", None)
    if ladder is None:
        return JSONResponse({"queued": False, "reason": "engine_backend_unavailable"})

    mgr = getattr(backend, "_manager", None)
    if mgr is None:
        return JSONResponse({"queued": False, "reason": "no_manager"})
    from augmentum.models.llama_server_manager import ProcessState
    if mgr.state != ProcessState.READY:
        return JSONResponse({"queued": False, "reason": "engine_not_ready"})

    task = asyncio.create_task(ladder.resume_session(session_key, source="open"))
    # resume_session never raises (it catches internally), but a
    # cancelled-at-shutdown task would still warn without this.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return JSONResponse({"queued": True, "session_key": session_key})


@engine_router.get("/v2/kv/resume-status")
async def engine_v2_kv_resume_status(request: Request) -> JSONResponse:
    """Recent resume-ladder outcomes (opaque session keys only).

    Diagnostic surface for acceptance runs: shows which rung each
    recent resume took (hot/restore/replay/cold) and why.
    """
    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    ladder = getattr(backend, "_resume_ladder", None)
    if ladder is None:
        return JSONResponse({"results": {}})
    return JSONResponse({"results": ladder.last_results})


@engine_router.post("/v2/kv/speculate")
async def engine_v2_kv_speculate(request: Request) -> JSONResponse:
    """Speculate the next turn from a typing-pause draft (ladder rung 3).

    Body: {"session_id": "...", "draft": "...", "prior_assistant": "..."}

    Local-engine only by construction — the speculator refuses to run
    without the managed llama-server, so drafts never reach a cloud
    backend. Fire-and-forget; always 200 (a skipped speculation is
    normal, never an error). Draft text is held in process memory only.
    """
    body = await request.json()
    session_id = str(body.get("session_id", "") or "").strip()
    draft = str(body.get("draft", "") or "")
    prior_assistant = str(body.get("prior_assistant", "") or "")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    from augmentum.config import settings as _cfg
    if not getattr(_cfg, "engine_speculation_enabled", False):
        return JSONResponse({"queued": False, "reason": "speculation_disabled"})
    if not draft.strip():
        return JSONResponse({"queued": False, "reason": "empty_draft"})

    from augmentum.proxy.session import derive_kv_session_key
    session_key = derive_kv_session_key(_request_user_id(request), session_id)

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    if not hasattr(backend, "turn_speculator"):
        return JSONResponse({"queued": False, "reason": "engine_backend_unavailable"})
    mgr = getattr(backend, "_manager", None)
    if mgr is None:
        return JSONResponse({"queued": False, "reason": "no_manager"})
    from augmentum.models.llama_server_manager import ProcessState
    if mgr.state != ProcessState.READY:
        return JSONResponse({"queued": False, "reason": "engine_not_ready"})

    speculator = backend.turn_speculator
    task = asyncio.create_task(speculator.speculate(
        session_key, draft=draft, prior_assistant=prior_assistant,
        source="typing",
    ))
    # speculate() catches internally; this only silences the
    # cancelled-at-shutdown warning.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return JSONResponse({"queued": True})


@engine_router.get("/v2/kv/speculate-status")
async def engine_v2_kv_speculate_status(request: Request) -> JSONResponse:
    """Speculator diagnostics: in-flight state, held entries, outcomes.

    Opaque session keys and sizes only — never draft or completion text.
    """
    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    speculator = getattr(backend, "_speculator", None)
    if speculator is None:
        return JSONResponse({"inflight": False, "entries": {}, "results": {}})
    return JSONResponse(speculator.status())


@engine_router.post("/v2/embeddings")
async def engine_v2_embeddings(request: Request) -> JSONResponse:
    """Generate embeddings using the loaded model.

    Body: {"input": "text" or ["text1", "text2"], "model": "optional"}
    Returns OpenAI-compatible embedding response.
    """
    mgr = _get_llama_manager(request)
    from augmentum.models.llama_server_manager import ProcessState
    if mgr.state != ProcessState.READY:
        raise HTTPException(status_code=503, detail="No model loaded. Load a model first.")

    body = await request.json()
    text_input = body.get("input", "")
    if not text_input:
        raise HTTPException(status_code=400, detail="input required")

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    result = await backend.embeddings(text_input, model=body.get("model", ""))
    return JSONResponse(result)


@engine_router.post("/v2/generate")
async def engine_v2_generate(request: Request) -> StreamingResponse:
    """Raw prompt completion (non-chat, no template).

    Body: {"prompt": "text", "stream": true, "n_predict": 128, ...}
    Sends directly to llama-server /completion without chat template.
    Useful for code completion, fill-in-middle, structured prompting.
    """
    mgr = _get_llama_manager(request)
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    from augmentum.models.llama_server_manager import ProcessState
    if mgr.state != ProcessState.READY:
        raise HTTPException(status_code=503, detail="No model loaded")

    mgr.touch()

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    base_url = backend._base_url
    client = backend._client  # reuse shared httpx client

    # Forward the request body to /completion with cache_prompt
    body.setdefault("cache_prompt", True)

    if body.get("stream", False):
        async def _stream():
            async with client.stream(
                "POST", f"{base_url}/completion",
                json=body, timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line + "\n"

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    else:
        try:
            resp = await client.post(
                f"{base_url}/completion",
                json=body, timeout=300.0,
            )
            resp.raise_for_status()
            return JSONResponse(resp.json())
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))


@engine_router.post("/v2/benchmark")
async def engine_v2_benchmark(request: Request) -> JSONResponse:
    """Run a quick benchmark on the currently loaded model.

    Body (optional): {"prompt": "...", "n_predict": 128}
    Returns: load_time, TTFT, prompt tok/s, generation tok/s, VRAM usage.
    """
    mgr = _get_llama_manager(request)
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    prompt = body.get("prompt", "Write a detailed paragraph about artificial intelligence and its impact on society.")
    n_predict = body.get("n_predict", 128)

    from augmentum.models.llama_server_manager import ProcessState

    registry = request.app.state.provider_registry
    backend = registry.get_backend("engine")
    base_url = backend._base_url
    client = backend._client

    result: dict = {
        "model_id": mgr.model_id,
        "state": mgr.state.value,
    }

    # If no model loaded, load the current model and measure load time
    if mgr.state != ProcessState.READY:
        raise HTTPException(status_code=503, detail="No model loaded. Load a model first.")

    # Get model profile info
    if mgr._last_profile:
        p = mgr._last_profile
        result["profile"] = {
            "architecture": p.architecture,
            "size_gb": p.size_gb,
            "n_layers": p.n_layers,
            "is_moe": p.is_moe,
            "context_length": p.context_length,
        }

    # GPU stats before
    gpu = mgr._query_gpu_info()
    if gpu:
        result["gpu"] = {
            "name": gpu.get("gpu_name", ""),
            "vram_total_mib": gpu.get("total_mib", 0),
            "vram_used_mib": gpu.get("used_mib", 0),
            "vram_free_mib": gpu.get("free_mib", 0),
        }

    # Apply chat template to make it realistic
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = await backend.apply_template(messages)
        if not rendered:
            rendered = prompt
    except Exception:
        rendered = prompt

    # Tokenize
    tokens = await backend.tokenize(rendered)
    prompt_token_count = len(tokens) if tokens else len(rendered) // 4

    # Run benchmark — non-streaming to get full timings
    t_start = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/completion",
            json={
                "prompt": tokens if tokens else rendered,
                "n_predict": n_predict,
                "cache_prompt": False,  # force full prefill for honest benchmark
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        t_end = time.monotonic()

        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"llama-server returned {resp.status_code}")

        data = resp.json()
        timings = data.get("timings", {})

        result["benchmark"] = {
            "prompt_tokens": timings.get("prompt_n", prompt_token_count),
            "generated_tokens": timings.get("predicted_n", 0),
            "prompt_ms": round(timings.get("prompt_ms", 0), 1),
            "generation_ms": round(timings.get("predicted_ms", 0), 1),
            "prompt_tps": round(timings.get("prompt_per_token_ms", 0), 2),
            "generation_tps": round(
                timings.get("predicted_n", 0) / (timings.get("predicted_ms", 1) / 1000), 1
            ) if timings.get("predicted_ms", 0) > 0 else 0,
            "prompt_tps_calc": round(
                timings.get("prompt_n", 0) / (timings.get("prompt_ms", 1) / 1000), 1
            ) if timings.get("prompt_ms", 0) > 0 else 0,
            "total_s": round(t_end - t_start, 3),
            "ttft_ms": round(timings.get("prompt_ms", 0), 1),
        }

        mgr.touch()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Benchmark timed out (120s)")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# HuggingFace Model Search (GGUF for llama-server; safetensors for the vLLM engine)
# ---------------------------------------------------------------------------

# Files that make up a servable safetensors repo: the weights + everything a
# runtime needs to load them (config, tokenizer, generation config) + any custom
# modeling code (trust_remote_code archs ship modeling_*.py / configuration_*.py).
# Exclude other weight formats (pytorch .bin/.pth, GGUF, flax/tf) to avoid
# double-downloading, and repo chrome (images, notebooks, docs).
_SAFETENSORS_EXCLUDE_EXT = (
    ".gguf", ".bin", ".pth", ".pt", ".h5", ".msgpack", ".onnx", ".tflite",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".md", ".ipynb", ".ckpt",
)


def _is_safetensors_serving_file(fname: str) -> bool:
    """True if ``fname`` belongs in a downloaded safetensors serving repo."""
    f = fname.lower()
    if f.endswith(".safetensors"):
        return True
    if any(f.endswith(ext) for ext in _SAFETENSORS_EXCLUDE_EXT):
        return False
    # Keep config/tokenizer/modeling code + small text metadata.
    return f.endswith((".json", ".py", ".model", ".txt", ".jinja", ".tiktoken"))


@engine_router.get("/v2/safetensors/capability")
async def engine_v2_safetensors_capability(request: Request) -> JSONResponse:
    """Whether a safetensors-capable engine (the vLLM sidecar) is installed and
    registered. The model manager gates its safetensors format toggle on this —
    no engine, no toggle (and thus no dead-end downloads nothing can serve)."""
    registry = getattr(request.app.state, "provider_registry", None)
    backends = getattr(registry, "_backends", {}) if registry is not None else {}
    return JSONResponse({"available": "vllm" in backends})


@engine_router.get("/v2/models/search")
async def engine_v2_search_models(
    request: Request,
    q: str = "",
    limit: int = 20,
    format: str = "gguf",
) -> JSONResponse:
    """Search HuggingFace Hub for GGUF (or safetensors) models by downloads.

    ``format`` selects the weight format: ``gguf`` (default — served by the
    bundled llama-server) or ``safetensors`` (served by the optional vLLM engine
    for architectures llama.cpp can't load). safetensors results carry the FULL
    repo file set (weights + config + tokenizer + custom modeling .py) since the
    whole repo is the serving unit, not a single quant file. The safetensors
    surface is only shown when a safetensors-capable engine is installed (the UI
    gates the toggle on capability).

    Works independently of the engine manager — only needs network access.
    Results are cached for 30 seconds to avoid hammering HF on every keystroke.
    """
    file_format = (format or "gguf").strip().lower()
    if file_format not in ("gguf", "safetensors"):
        file_format = "gguf"
    query = q.strip()
    if not query or len(query) < 2:
        return JSONResponse({"results": []})

    # Clamp limit
    limit = max(1, min(limit, 50))

    # Check cache
    cache_key = f"{file_format}:{query.lower()}:{limit}"
    now = time.monotonic()
    cached = _hf_cache.get(cache_key)
    if cached and (now - cached[0]) < _HF_CACHE_TTL:
        return JSONResponse({"results": cached[1]})

    client = _get_hf_client()

    # Auth header (optional)
    headers: dict[str, str] = {}
    hf_token = settings.huggingface_token
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        # Search for GGUF models sorted by downloads
        resp = await client.get(
            "https://huggingface.co/api/models",
            params={
                "search": query,
                "filter": file_format,
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
                "full": "true",  # includes siblings (file list)
            },
            headers=headers,
            timeout=15.0,
        )
        resp.raise_for_status()
        models_data = resp.json()
    except httpx.TimeoutException:
        log.warning("hf_search_timeout", query=query)
        return JSONResponse({"results": [], "error": "HuggingFace API timed out"}, status_code=504)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            log.warning("hf_search_rate_limited", query=query)
            return JSONResponse(
                {"results": [], "error": "Rate limited by HuggingFace"},
                status_code=429,
            )
        log.warning("hf_search_error", query=query, status=status)
        return JSONResponse(
            {"results": [], "error": f"HuggingFace API error: {status}"},
            status_code=502,
        )
    except Exception as exc:
        log.warning("hf_search_error", query=query, error=str(exc))
        return JSONResponse(
            {"results": [], "error": "Failed to reach HuggingFace"},
            status_code=502,
        )

    manager = getattr(request.app.state, "model_manager", None)

    async def _build_search_result(model: dict) -> dict | None:
        model_id = model.get("id", "") or model.get("modelId", "")
        if not model_id:
            return None

        author = model_id.split("/")[0] if "/" in model_id else ""
        siblings = model.get("siblings") or []

        # Collect downloadable files. GGUF: individual quant files the user picks
        # from. safetensors: the WHOLE serving set (weights + config + tokenizer +
        # custom modeling .py), since the repo is the serving unit, not one file.
        files = []
        for sib in siblings:
            fname = sib.get("rfilename", "")
            keep = (
                fname.lower().endswith(".gguf") if file_format == "gguf"
                else _is_safetensors_serving_file(fname)
            )
            if keep:
                files.append({"name": fname, "size": sib.get("size") or 0})

        # Only GGUF needs per-file size backfill (the picker shows per-quant size);
        # safetensors downloads as a bundle so an aggregate is enough.
        if manager is not None and files and file_format == "gguf":
            files = await manager.fill_missing_hf_file_sizes(model_id, files, limit=4)

        return {
            "id": model_id,
            "author": author,
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "tags": model.get("tags", []),
            "files": files,
            "format": file_format,
        }

    raw_results = await asyncio.gather(*(_build_search_result(model) for model in models_data))
    results = [result for result in raw_results if result is not None]

    # Cache results
    _hf_cache[cache_key] = (now, results)

    # Evict old cache entries (keep cache size bounded)
    if len(_hf_cache) > 100:
        stale = [k for k, (ts, _) in _hf_cache.items() if (now - ts) > _HF_CACHE_TTL]
        for k in stale:
            _hf_cache.pop(k, None)

    return JSONResponse({"results": results})
