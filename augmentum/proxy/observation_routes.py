"""Admin endpoint for the Observation Substrate.

Phase A surfaces one operation: rebuild-cache. The operator hits this
to drive the full pipeline (optional seed → top-K query → corpus →
llama-lookup-create) in isolation, gets a structured response back with
counts + file paths + timings, and can then flip the
``observation_lookup_cache_enabled`` flag to have LlamaServerManager
pick up the new cache on next model start.

Future endpoints (deferred per the substrate spec):
  GET    /api/observation/patterns   — human-rendered policy summary
  GET    /api/observation/raw        — last N observations
  DELETE /api/observation/cluster/{fp}
  POST   /api/observation/threshold
  POST   /api/observation/purge      — by surface / time / all

Auth: standard per-user via ``request.scope["user"]`` (multi-tenant
pattern from CLAUDE.md). The endpoint operates against the requester's
own observation history; the configured ``observation_primary_user_id``
setting is what LlamaServerManager actually reads at model-start time.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.observation.exporter import (
    cache_path_for,
    export_lookup_cache,
)
from augmentum.observation.seeder import seed_from_chat_history
from augmentum.observation.store import ObservationStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/observation", tags=["observation"])


def _require_user(request: Request) -> str:
    """Standard auth extraction. 401 when no user scope is attached.

    The Observation Substrate is per-user by construction — there is no
    'anonymous' path that would make sense to serve, so the refusal is
    structural rather than a separate authorization layer.
    """
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    return user_id


def _conn(request: Request) -> Any:
    """Pull the aiosqlite connection off app.state.

    Mirrors the pattern used by other routes that talk directly to the
    main store (chat_routes, character_routes). The state_manager.backend
    property is the canonical surface; the underlying ``_conn`` is what
    aiosqlite-using stores expect.
    """
    backend = getattr(request.app.state, "state_manager", None)
    raw = getattr(backend, "backend", None) if backend else None
    conn = getattr(raw, "_conn", None) or getattr(raw, "conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return conn


@router.post("/rebuild-cache")
async def rebuild_cache(request: Request) -> JSONResponse:
    """Run the full Observation Substrate pipeline for the calling user.

    Steps (in order, gated by settings):

      1. **Substrate enabled check.** Refuses with 409 if
         ``observation_substrate_enabled=False`` — the operator should
         flip the master switch first.
      2. **Seed.** When ``observation_seed_chat_history=True``, walks the
         user's ``ui_sessions`` and ingests (prefix, continuation) pairs.
         Idempotent (the store upserts; re-seeding bumps counts rather
         than duplicating).
      3. **Export.** Invokes ``llama-lookup-create`` against the currently-
         loaded model (queried off ``LlamaServerManager.current_model_path``)
         and atomically renames the result into
         ``/data/lookup_cache/{user}/{model_stem}.bin``.

    Returns: a structured report — counts, byte sizes, duration, cache
    path. Used by the operator to verify the pipeline ran end-to-end
    before flipping the ``observation_lookup_cache_enabled`` flag that
    causes LlamaServerManager to actually pass the cache to llama-server.

    409 on missing prereq (substrate disabled, no current model);
    503 if the database is down; 500 if the subprocess fails (the
    error message carries the lookup-create stderr tail).
    """
    user_id = _require_user(request)

    if not getattr(settings, "observation_substrate_enabled", False):
        raise HTTPException(
            status_code=409,
            detail=(
                "observation_substrate_enabled is False; "
                "set it via PUT /api/config/tools first"
            ),
        )

    conn = _conn(request)
    store = ObservationStore(conn)

    report: dict[str, Any] = {
        "user_id": user_id,
        "seeded": None,
        "observations_total": 0,
        "cache": None,
    }

    if getattr(settings, "observation_seed_chat_history", False):
        seed_counts = await seed_from_chat_history(
            store, user_id=user_id, conn=conn,
        )
        await conn.commit()
        report["seeded"] = seed_counts

    report["observations_total"] = await store.count(user_id=user_id)

    if report["observations_total"] == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "no observations for this user — enable "
                "observation_seed_chat_history or wait for ingestion to "
                "accumulate before rebuilding the cache"
            ),
        )

    # Cache export is gated by its own flag. When disabled, this endpoint
    # is effectively "run the seeder" — useful for the autocomplete-only
    # consumer that doesn't need the llama-server lookup cache (and
    # therefore doesn't need the bundled llama-lookup-create binary).
    if not getattr(settings, "observation_lookup_cache_enabled", False):
        report["cache"] = None
        report["cache_skipped_reason"] = (
            "observation_lookup_cache_enabled is False — seed-only run"
        )
        return JSONResponse({"ok": True, "report": report})

    # Find the currently-loaded model. LlamaServerManager is a process
    # singleton accessible via app.state.llama_server_manager (or the
    # generic provider registry). Probe the most direct path.
    manager = getattr(request.app.state, "llama_server_manager", None)
    model_path = getattr(manager, "current_model_path", "") if manager else ""
    if not model_path:
        # Fall back to the manager-registry style some installs use.
        registry = getattr(request.app.state, "provider_registry", None)
        if registry is not None:
            current = getattr(registry, "current_local_model_path", None)
            if callable(current):
                try:
                    model_path = await current()
                except Exception:
                    model_path = ""
            elif isinstance(current, str):
                model_path = current
    if not model_path:
        raise HTTPException(
            status_code=409,
            detail=(
                "no local model is currently loaded; load a model via "
                "llama-server first so the cache can be tokenized "
                "against its vocabulary"
            ),
        )

    max_entries = int(
        getattr(settings, "observation_lookup_cache_max_entries", 50_000) or 50_000
    )

    try:
        result = await export_lookup_cache(
            store,
            user_id=user_id,
            model_path=model_path,
            max_entries=max_entries,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Subprocess failure / timeout — surface the message verbatim
        # since it carries the lookup-create stderr tail for diagnosis.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    report["cache"] = {
        "path": str(result.cache_path),
        "bytes": result.cache_bytes,
        "observations_used": result.observations_used,
        "corpus_bytes": result.corpus_bytes,
        "duration_seconds": round(result.duration_seconds, 3),
    }

    # Mark whether LlamaServerManager would actually use this cache —
    # the operator wants to know if the next model start picks it up.
    primary_user = (
        getattr(settings, "observation_primary_user_id", "") or ""
    ).strip()
    report["next_model_start_will_use_cache"] = (
        bool(getattr(settings, "observation_lookup_cache_enabled", False))
        and primary_user == user_id
    )
    report["observation_primary_user_id"] = primary_user

    return JSONResponse({"ok": True, "report": report})


@router.get("/complete")
async def complete(
    request: Request,
    prefix: str = "",
    surface: str = "chat",
    mode: str = "",
    k: int = 5,
) -> JSONResponse:
    """Return ranked continuations for the tail of ``prefix``.

    The chat composer calls this on debounced typing to render ghost
    text. Returns:

      ``{matched_prefix, suggestions: [{continuation, count}]}``

    Returns an empty result (status 200, ``suggestions=[]``) when the
    substrate is disabled, the user has no observations yet, or no
    tail length matches — keeps the frontend's silent-skip path simple
    (one shape to handle).

    Bounds k to [1, 10] — anything past the top few is noise for the
    ghost-text UX, and the wider query is wasted DB work.
    """
    user_id = _require_user(request)

    # Substrate disabled → return empty so the frontend silently no-ops.
    # We don't 4xx here because the frontend doesn't know operator
    # config; making it figure that out is the wrong layer.
    if not getattr(settings, "observation_substrate_enabled", False):
        return JSONResponse({
            "matched_prefix": "",
            "suggestions": [],
            "substrate_enabled": False,
        })

    conn = _conn(request)
    store = ObservationStore(conn)
    matched, hits = await store.complete(
        user_id=user_id,
        current_text=prefix or "",
        surface=surface or "chat",
        mode=mode or "",
        k=max(1, min(int(k), 10)),
    )
    return JSONResponse({
        "matched_prefix": matched,
        "suggestions": [
            {"continuation": cont, "count": count}
            for cont, count in hits
        ],
        "substrate_enabled": True,
    })


@router.get("/status")
async def status(request: Request) -> JSONResponse:
    """Quick gauge of the substrate's state for the calling user.

    Used by the operator to check the pipeline state without firing a
    rebuild — observation count, whether a cache file exists for the
    currently-loaded model, and which flags are on.
    """
    user_id = _require_user(request)
    conn = _conn(request)
    store = ObservationStore(conn)

    total = await store.count(user_id=user_id)

    cache_present = False
    cache_path_str = ""
    manager = getattr(request.app.state, "llama_server_manager", None)
    model_path = getattr(manager, "current_model_path", "") if manager else ""
    if model_path:
        cache_path = cache_path_for(user_id, model_path)
        cache_path_str = str(cache_path)
        cache_present = cache_path.exists() and cache_path.stat().st_size > 0

    return JSONResponse({
        "ok": True,
        "user_id": user_id,
        "observations_total": total,
        "current_model_path": model_path,
        "cache_path": cache_path_str,
        "cache_present": cache_present,
        "flags": {
            "substrate_enabled": bool(
                getattr(settings, "observation_substrate_enabled", False)
            ),
            "seed_chat_history": bool(
                getattr(settings, "observation_seed_chat_history", False)
            ),
            "lookup_cache_enabled": bool(
                getattr(settings, "observation_lookup_cache_enabled", False)
            ),
            "primary_user_id": (
                getattr(settings, "observation_primary_user_id", "") or ""
            ),
        },
    })
