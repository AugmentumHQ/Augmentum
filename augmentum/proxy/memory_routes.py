"""Memory management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.config import settings
from augmentum.memory.models import MemoryTier, MemoryType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/memory", tags=["memory"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


class StoreRequest(BaseModel):
    content: str
    memory_type: str = "fact"
    importance: float = 0.5
    user_id: str = "default"
    scope: str | None = None


class EditRequest(BaseModel):
    content: str
    importance: float | None = None
    memory_type: str | None = None


def _get_store(request: Request):
    store = getattr(request.app.state, "memory_store", None)
    if store is None:
        return None
    return store


@router.get("/facts")
async def list_facts(
    request: Request,
    user_id: str = "default",
    type: str | None = None,
    include_expired: bool = False,
    limit: int = 50,
    offset: int = 0,
    scope: str | None = None,
) -> JSONResponse:
    """List memories with optional type filter and pagination."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    # Auth user_id overrides query parameter when auth is active
    uid = _user_id(request)
    if uid:
        user_id = uid

    memory_type = MemoryType(type) if type else None
    memories = await store.list_all(
        user_id=user_id,
        memory_type=memory_type,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
        scope=scope,
    )
    return JSONResponse({
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "importance": m.importance,
                "confidence": m.confidence,
                "source_type": m.source_type,
                "tier": m.tier if isinstance(m.tier, str) else m.tier.value,
                "access_count": m.access_count,
                "valid_from": m.valid_from,
                "valid_until": m.valid_until,
                "created_at": m.created_at,
                "updated_at": m.updated_at,
            }
            for m in memories
        ],
        "count": len(memories),
        "offset": offset,
        "limit": limit,
    })


@router.get("/search")
async def search_memories(
    request: Request,
    q: str,
    user_id: str = "default",
    limit: int = 10,
    scope: str | None = None,
) -> JSONResponse:
    """Semantic + keyword search for memories."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if uid:
        user_id = uid
    memories = await store.recall(q, user_id=user_id, limit=limit, scope=scope)
    return JSONResponse({
        "query": q,
        "results": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "importance": m.importance,
                "confidence": m.confidence,
                "tier": m.tier if isinstance(m.tier, str) else m.tier.value,
                "created_at": m.created_at,
            }
            for m in memories
        ],
        "count": len(memories),
    })


@router.post("/store")
async def store_memory(body: StoreRequest, request: Request) -> JSONResponse:
    """Manually store a memory."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    from augmentum.memory.models import SourceType

    uid = _user_id(request)
    effective_user_id = uid if uid else body.user_id
    memory_id = await store.store(
        content=body.content,
        memory_type=body.memory_type,
        user_id=effective_user_id,
        importance=body.importance,
        source_type=SourceType.USER_MANUAL,
        scope=body.scope,
    )
    return JSONResponse({"id": memory_id, "status": "stored"}, status_code=201)


@router.put("/facts/{memory_id}")
async def edit_memory(memory_id: str, body: EditRequest, request: Request) -> JSONResponse:
    """Edit a memory's content, importance, and/or type (re-embeds automatically)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    success = await store.edit(memory_id, body.content, user_id=uid)
    if not success:
        return JSONResponse({"error": "Memory not found"}, status_code=404)

    # Update importance and/or type if provided
    if body.importance is not None or body.memory_type is not None:
        conn = store._backend.conn
        updates = []
        params: list = []
        if body.importance is not None:
            updates.append("importance = ?")
            params.append(body.importance)
        if body.memory_type is not None:
            updates.append("memory_type = ?")
            params.append(body.memory_type)
        if updates:
            params.extend([memory_id, uid])
            await conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                params,
            )
            await conn.commit()

    return JSONResponse({"id": memory_id, "status": "updated"})


@router.put("/facts/{memory_id}/tier")
async def update_memory_tier(memory_id: str, request: Request) -> JSONResponse:
    """Manually promote or demote a memory's tier."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    new_tier = body.get("tier", "").strip().lower()
    valid_tiers = {t.value for t in MemoryTier}
    if new_tier not in valid_tiers:
        return JSONResponse(
            {"error": f"Invalid tier. Must be one of: {', '.join(sorted(valid_tiers))}"},
            status_code=400,
        )

    success = await store.update_tier(memory_id, new_tier, user_id=uid, source="manual")
    if not success:
        return JSONResponse({"error": "Memory not found"}, status_code=404)

    # Clear provisional TTL if promoting out of provisional
    if new_tier != MemoryTier.PROVISIONAL:
        conn = store._backend.conn
        await conn.execute(
            "UPDATE memories SET provisional_expires_at = NULL "
            "WHERE id = ? AND user_id = ?",
            (memory_id, uid),
        )
        await conn.commit()

    log.info("memory_tier_updated", memory_id=memory_id, tier=new_tier)
    return JSONResponse({"id": memory_id, "tier": new_tier, "status": "updated"})


@router.get("/facts/{memory_id}/tier-history")
async def get_tier_history(memory_id: str, request: Request) -> JSONResponse:
    """List tier transitions for a memory.

    Returns the audit trail produced by retroactive demotion + explicit
    promotions/reverts. Most recent first. Empty array if the memory has
    no transitions on record. Used by the inspector UI to show "demoted
    on date, reason X" and expose the revert button.
    """
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = store._backend.conn  # noqa: SLF001 — same pattern as update_memory_tier
    cursor = await conn.execute(
        "SELECT id, from_tier, to_tier, reason, transitioned_at "
        "FROM memory_tier_history "
        "WHERE memory_id = ? AND user_id = ? "
        "ORDER BY transitioned_at DESC",
        (memory_id, uid),
    )
    rows = await cursor.fetchall()
    return JSONResponse({
        "memory_id": memory_id,
        "transitions": [
            {
                "id": r[0],
                "from_tier": r[1],
                "to_tier": r[2],
                "reason": r[3] or "",
                "transitioned_at": r[4],
            }
            for r in rows
        ],
        "count": len(rows),
    })


# ---------------------------------------------------------------------------
# The Mirror (Earned Understanding P3) — why she believes a thing
# ---------------------------------------------------------------------------

def _origin_phrase(source_type: str, source_context) -> str:
    """Human phrasing for HOW a belief first entered — so the trail reads as
    'she believes this because…', not a row dump. ``source_context`` may be a
    dict or a JSON string (the Memory dataclass stores it as a string)."""
    ctx: dict = {}
    if isinstance(source_context, dict):
        ctx = source_context
    elif isinstance(source_context, str) and source_context.strip():
        try:
            import json as _json
            parsed = _json.loads(source_context)
            ctx = parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            ctx = {}
    ctx_source = str(ctx.get("source") or "").lower()
    if ctx_source == "playlist":
        return "Noticed from a playlist you made"
    if ctx_source == "offer":
        return "You confirmed this when I offered to remember it"
    st = (source_type or "").lower()
    if st in ("explicit", "user_manual"):
        return "You told me this directly"
    if st == "system":
        return "A pattern I noticed across what I already knew"
    if st == "extracted":
        return "Picked up from something you said in conversation"
    return "Learned over time"


def _days_since(iso: str) -> int | None:
    if not iso:
        return None
    try:
        from datetime import UTC, datetime
        raw = iso.replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return max(0, (datetime.now(UTC) - ts).days)
    except (TypeError, ValueError):
        return None


@router.get("/facts/{memory_id}/evidence")
async def get_belief_evidence(memory_id: str, request: Request) -> JSONResponse:
    """The Mirror: why she believes this — origin + corroborating evidence
    trail + convergence strength + staleness (the visible edge of decay).

    Every belief is explainable: the user can see WHICH independent signals
    converged on it ("you said it, you made a playlist, you keep coming back"),
    not just the conclusion. This is the OSS-native trust property — audit the
    reasoning, not just the output.
    """
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    mem = await store.get(memory_id, user_id=uid)
    if mem is None:
        return JSONResponse({"error": "Memory not found"}, status_code=404)

    tier = mem.tier if isinstance(mem.tier, str) else mem.tier.value
    source_type = mem.source_type if isinstance(mem.source_type, str) else getattr(
        mem.source_type, "value", "")
    source_context = getattr(mem, "source_context", None)

    trail: list[dict] = []
    score = 0.0
    distinct = 0
    ev = getattr(request.app.state, "evidence_store", None)
    if ev is not None:
        try:
            trail = await ev.evidence_for(user_id=uid, memory_id=memory_id)
            score = await ev.score_for(user_id=uid, memory_id=memory_id)
            distinct = await ev.distinct_sources(user_id=uid, memory_id=memory_id)
        except Exception:
            log.warning("belief_evidence_fetch_failed", memory_id=memory_id, exc_info=True)

    # Last reinforced = the freshest of (newest evidence, the belief's own
    # creation). Staleness here is the visible edge of decay — a belief nothing
    # has reinforced in a long while is a candidate to fade.
    reinforced_at = trail[0]["created_at"] if trail else (mem.created_at or "")
    return JSONResponse({
        "memory_id": memory_id,
        "content": mem.content,
        "tier": tier,
        "origin": _origin_phrase(source_type, source_context),
        "created_at": mem.created_at,
        "convergence": {"score": score, "distinct_sources": distinct},
        "last_reinforced_at": reinforced_at,
        "days_since_reinforced": _days_since(reinforced_at),
        "trail": trail,
    })


@router.post("/facts/{memory_id}/revert-tier")
async def revert_tier(memory_id: str, request: Request) -> JSONResponse:
    """Revert the most recent tier transition for this memory.

    Flips the memory back to its prior tier and writes a corresponding
    history row with reason='user_revert'. Returns 404 if there's no
    transition on record (nothing to revert).
    """
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    success = await store.revert_tier_transition(memory_id, uid)
    if not success:
        return JSONResponse(
            {"error": "No tier transition to revert (memory may have no history)"},
            status_code=404,
        )

    # Return the current tier after revert so the UI can update without
    # a separate fetch.
    conn = store._backend.conn  # noqa: SLF001
    cursor = await conn.execute(
        "SELECT tier FROM memories WHERE id = ? AND user_id = ?",
        (memory_id, uid),
    )
    row = await cursor.fetchone()
    current_tier = row[0] if row else None
    log.info("memory_tier_reverted_via_api", memory_id=memory_id, new_tier=current_tier)
    return JSONResponse({"id": memory_id, "tier": current_tier, "status": "reverted"})


@router.delete("/facts/{memory_id}")
async def delete_memory(memory_id: str, request: Request) -> JSONResponse:
    """Soft-delete a memory (sets valid_until)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    success = await store.forget(memory_id, user_id=uid)
    if not success:
        return JSONResponse({"error": "Memory not found or already deleted"}, status_code=404)
    return JSONResponse({"id": memory_id, "status": "deleted"})


@router.get("/stats")
async def memory_stats(request: Request, user_id: str = "default") -> JSONResponse:
    """Get memory statistics by type."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if uid:
        user_id = uid
    counts = await store.count(user_id)
    return JSONResponse({"user_id": user_id, "counts": counts})


@router.get("/diagnostics")
async def memory_diagnostics(request: Request) -> JSONResponse:
    """Diagnostic info about memory subsystem health."""
    store = _get_store(request)

    diag: dict[str, object] = {
        "memory_enabled": settings.memory_enabled,
        "memory_store_initialized": store is not None,
        "vec_enabled": store._vec_enabled if store else False,
        "llm_extraction_enabled": settings.memory_llm_extraction_enabled,
        "llm_extraction_model": getattr(settings, "memory_llm_extraction_model", ""),
        "core_profile_enabled": settings.memory_core_profile_enabled,
        "core_profile_initialized": getattr(request.app.state, "core_profile_manager", None) is not None,
        "scope_by_mode": settings.memory_scope_by_mode,
        "inject_analytical": settings.memory_inject_analytical,
        "inject_agentic": settings.memory_inject_agentic,
    }

    # Check embedding model
    try:
        from augmentum.memory.embeddings import _UNLOADED, EmbeddingService
        diag["embedding_model_loaded"] = EmbeddingService._model is not _UNLOADED
        diag["embedding_model"] = EmbeddingService.MODEL_NAME
    except Exception as e:
        diag["embedding_model_loaded"] = False
        diag["embedding_error"] = str(e)

    # Check backend availability for extraction
    registry = getattr(request.app.state, "provider_registry", None)
    if registry:
        try:
            backend = registry.default_backend
            diag["extraction_backend_available"] = backend is not None
            diag["extraction_backend_type"] = type(backend).__name__ if backend else None
        except (ValueError, KeyError):
            diag["extraction_backend_available"] = False
    else:
        diag["extraction_backend_available"] = False

    # Memory count — scope to authenticated user
    uid = _user_id(request)
    user_id = uid or "default"
    if store:
        try:
            counts = await store.count(user_id=user_id)
            diag["memory_count"] = counts
        except Exception:
            diag["memory_count"] = {"error": "failed to count"}

    return JSONResponse(diag)


@router.get("/facts/{memory_id}/history")
async def memory_history(memory_id: str, request: Request) -> JSONResponse:
    """Get the version history of a memory (superseded_by chain)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    history = await store.get_history(memory_id, user_id=uid)
    if not history:
        return JSONResponse({"error": "Memory not found"}, status_code=404)
    return JSONResponse({
        "memory_id": memory_id,
        "versions": [
            {
                "id": m.id,
                "content": m.content,
                "valid_from": m.valid_from,
                "valid_until": m.valid_until,
                "superseded_by": m.superseded_by,
                "created_at": m.created_at,
            }
            for m in history
        ],
    })


@router.post("/compact")
async def compact_memories(request: Request, user_id: str = "default") -> JSONResponse:
    """Trigger manual memory compaction cycle."""
    uid = _user_id(request)
    if uid:
        user_id = uid
    compactor = getattr(request.app.state, "memory_compactor", None)
    if compactor is None:
        # Even without background compactor, run a one-shot compaction
        store = _get_store(request)
        if store is None:
            return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

        from augmentum.memory.compactor import MemoryCompactor

        registry = getattr(request.app.state, "provider_registry", None)
        one_shot = MemoryCompactor(
            store=store,
            registry=registry,
        )
        stats = await one_shot.compact(user_id)
        return JSONResponse({"status": "compact_complete", "stats": stats})

    stats = await compactor.compact(user_id)
    return JSONResponse({"status": "compact_complete", "stats": stats})


@router.post("/rebuild-profile")
async def rebuild_profile(request: Request, user_id: str = "default") -> JSONResponse:
    """Force rebuild of the core memory profile."""
    uid = _user_id(request)
    if uid:
        user_id = uid
    profile_mgr = getattr(request.app.state, "core_profile_manager", None)
    if profile_mgr is None:
        # Try to initialize on-demand if store exists
        store = _get_store(request)
        if store and settings.memory_core_profile_enabled:
            from augmentum.memory.core_profile import CoreProfileManager
            profile_mgr = CoreProfileManager(
                store,
                max_tokens=settings.memory_core_profile_max_tokens,
                rebuild_interval=settings.memory_core_profile_rebuild_interval,
            )
            request.app.state.core_profile_manager = profile_mgr
        else:
            return JSONResponse({
                "error": "Core profile not available — enable it in Memory > Configuration.",
                "status": "unavailable",
            })

    profile_mgr.invalidate(user_id)
    profile = await profile_mgr.get_profile(user_id)
    return JSONResponse({
        "status": "rebuilt",
        "user_id": user_id,
        "profile_length": len(profile),
        "profile": profile,
    })


@router.get("/profile")
async def get_profile(request: Request, user_id: str = "default") -> JSONResponse:
    """Get the current core memory profile (without rebuilding)."""
    uid = _user_id(request)
    if uid:
        user_id = uid
    profile_mgr = getattr(request.app.state, "core_profile_manager", None)
    if profile_mgr is None:
        return JSONResponse({
            "user_id": user_id,
            "profile_length": 0,
            "profile": "",
            "enabled": settings.memory_core_profile_enabled,
            "note": "Core profile manager not initialized — enable it in Memory > Configuration.",
        })

    profile = await profile_mgr.get_profile(user_id)
    return JSONResponse({
        "user_id": user_id,
        "profile_length": len(profile),
        "profile": profile,
        "enabled": settings.memory_core_profile_enabled,
    })


@router.get("/context-preview")
async def context_preview(request: Request, user_id: str = "default") -> JSONResponse:
    """Preview what memory context would be injected for the current session.

    Returns core profile + total memory count + recent recalled memories.
    Lightweight endpoint for the chat UI memory indicator.
    """
    uid = _user_id(request)
    if uid:
        user_id = uid
    store = _get_store(request)
    profile_mgr = getattr(request.app.state, "core_profile_manager", None)

    profile = ""
    if profile_mgr:
        try:
            # Cache-only read: never block the request on a 30–130 s LLM
            # rebuild. If neither cache nor persisted profile exists, the
            # method returns "" and schedules a background rebuild so the
            # next preview call returns the fresh value. The previous
            # implementation called ``get_profile`` which awaits the LLM
            # synchronously, holding the engine slot and serialising
            # everything behind it (including the game-agent slow path,
            # which shares the same provider registry).
            profile = await profile_mgr.get_profile_cached_only(user_id)
        except Exception:
            log.warning("memory_profile_retrieval_failed", user_id=user_id, exc_info=True)

    total = 0
    tier_counts: dict[str, int] = {}
    if store:
        try:
            stats = await store.count(user_id)
            total = sum(stats.values())
            # Get tier breakdown
            conn = store._backend.conn
            cursor = await conn.execute(
                "SELECT tier, COUNT(*) FROM memories WHERE user_id = ? AND valid_until IS NULL GROUP BY tier",
                (user_id,),
            )
            tier_counts = {row[0]: row[1] for row in await cursor.fetchall()}
        except Exception:
            log.warning("memory_tier_count_failed", user_id=user_id, exc_info=True)

    return JSONResponse({
        "enabled": settings.memory_enabled,
        "profile": profile[:500] if profile else "",
        "profile_enabled": settings.memory_core_profile_enabled,
        "total_memories": total,
        "tiers": tier_counts,
    })


async def _prepare_stream_events(conn, events: list[dict], *, user_id: str) -> list[dict]:
    """Filter lifecycle events down to user-meaningful ones and enrich them.

    Lifecycle events are system telemetry; only the ones a user can act on
    or learn from belong in the timeline. Dropped here:
      - dream_cycle with 0 reflections and no portrait update (nothing happened)
      - non-manual tier_change (legacy auto-promotions double-logged a
        tier_change next to every promotion event; the store no longer
        writes them — this hides the historical rows)
      - promotion/tier_change whose memory was since deleted (card would
        have nothing to show)
    Surviving memory-linked events gain ``memory_content`` so the card can
    say WHICH memory moved ("'Name is Matt' → core").
    """
    kept: list[dict] = []
    for e in events:
        etype = e.get("event_type") or ""
        detail = e.get("detail") or {}
        if etype == "dream_cycle" and not detail.get("entries_count") and not detail.get("portrait_updated"):
            continue
        if etype == "tier_change" and detail.get("source") != "manual":
            continue
        kept.append(e)

    mem_ids = {e["memory_id"] for e in kept if e.get("memory_id")}
    contents: dict[str, str] = {}
    if mem_ids:
        placeholders = ",".join("?" * len(mem_ids))
        cur = await conn.execute(
            f"SELECT id, content FROM memories WHERE user_id = ? AND id IN ({placeholders})",
            [user_id, *mem_ids],
        )
        contents = {r[0]: r[1] for r in await cur.fetchall()}

    out: list[dict] = []
    for e in kept:
        mid = e.get("memory_id")
        if mid:
            if mid not in contents:
                continue  # memory deleted — orphaned event, skip
            e["memory_content"] = contents[mid]
        out.append(e)
    return out


@router.get("/stream")
async def memory_stream(
    request: Request,
    user_id: str = "default",
    limit: int = 50,
    offset: int = 0,
    tier: str | None = None,
    type: str | None = None,
) -> JSONResponse:
    """Unified chronological feed of memories + events for the Living Stream UI."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if uid:
        user_id = uid
    conn = store._backend.conn
    items = []
    pending_notifications = []
    pending_ids = set()

    # Fetch reviewable notifications first so provisional memories render as
    # review cards instead of duplicate normal memory rows.
    try:
        from augmentum.memory.notifications import get_pending
        pending_notifications = await get_pending(conn, user_id=user_id, limit=20)
        pending_ids = {n["id"] for n in pending_notifications}
    except Exception as exc:
        # Defaults at L501-502 keep the feed degrading to memories-
        # only; debug-log so a notifications regression is findable.
        log.debug("memory_stream_pending_fetch_failed", error=str(exc))

    # Fetch memories
    mem_query = """
        SELECT id, content, memory_type, importance, confidence, source_type, tier,
               access_count, created_at, evidence
        FROM memories
        WHERE user_id = ? AND valid_until IS NULL
    """
    mem_params: list = [user_id]
    if tier:
        mem_query += " AND tier = ?"
        mem_params.append(tier)
    if type:
        mem_query += " AND memory_type = ?"
        mem_params.append(type)
    mem_query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    mem_params.extend([limit, offset])

    cursor = await conn.execute(mem_query, mem_params)
    for row in await cursor.fetchall():
        if row[0] in pending_ids:
            continue
        items.append({
            "kind": "memory",
            "id": row[0],
            "content": row[1],
            "memory_type": row[2],
            "importance": row[3],
            "confidence": row[4],
            "source_type": row[5],
            "tier": row[6],
            "access_count": row[7],
            "created_at": row[8],
            "evidence": row[9],
        })

    # Add pending notifications after memories; final sort places them in time.
    for n in pending_notifications:
        items.append({"kind": "notification", **n})

    # Fetch events (non-filtered — always show timeline context).
    if not tier and not type:
        try:
            from augmentum.memory.events import get_events
            events = await get_events(conn, user_id=user_id, limit=limit, offset=offset)
            for e in await _prepare_stream_events(conn, events, user_id=user_id):
                items.append({"kind": "event", **e})
        except Exception as exc:
            log.debug("memory_stream_events_fetch_failed", error=str(exc))

    # Sort all items by created_at descending
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Count total memories for pagination
    count_cursor = await conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id = ? AND valid_until IS NULL",
        (user_id,),
    )
    total = (await count_cursor.fetchone())[0]

    return JSONResponse({
        "items": items[:limit],
        "total": total,
        "offset": offset,
        "limit": limit,
    })


# ---------------------------------------------------------------------------
# Orphan reassignment (one-shot fix for pre-auth memories stored as "default")
# ---------------------------------------------------------------------------


@router.post("/reassign-orphans")
async def reassign_orphan_memories(request: Request) -> JSONResponse:
    """Reassign memories stored under 'default' to the authenticated user.

    This is a one-shot fix for memories that were stored before the auth
    system was wired into the streaming extraction path.
    """
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid or uid == "default":
        return JSONResponse({"error": "Must be authenticated as a non-default user"}, status_code=400)

    conn = store._backend.conn
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id = 'default' AND valid_until IS NULL",
    )
    count = (await cursor.fetchone())[0]

    if count == 0:
        return JSONResponse({"reassigned": 0, "message": "No orphaned memories found"})

    await conn.execute(
        "UPDATE memories SET user_id = ? WHERE user_id = 'default'",
        (uid,),
    )
    await conn.commit()

    # Invalidate core profile cache so it rebuilds with the reassigned memories
    profile_mgr = getattr(request.app.state, "core_profile_manager", None)
    if profile_mgr:
        profile_mgr.invalidate(uid)
        profile_mgr.mark_stale(uid)

    log.info("orphan_memories_reassigned", from_user="default", to_user=uid, count=count)
    return JSONResponse({"reassigned": count, "message": f"Reassigned {count} memories to {uid}"})


# ---------------------------------------------------------------------------
# Memory Configuration
# ---------------------------------------------------------------------------

# Settings that can be changed at runtime via the UI
_MEMORY_SETTINGS: dict[str, tuple[type, object, object]] = {
    "memory_enabled": (bool, None, None),
    "memory_recall_limit": (int, 1, 20),
    "memory_recall_min_score": (float, 0.0, 1.0),
    "memory_summary_max_chars": (int, 50, 2000),
    "memory_llm_extraction_enabled": (bool, None, None),
    "memory_extraction_batch_size": (int, 1, 20),
    "memory_auto_approve": (bool, None, None),
    "memory_core_profile_enabled": (bool, None, None),
    "memory_core_profile_max_tokens": (int, 50, 2000),
    "memory_core_profile_rebuild_interval": (int, 1, 100),
    "memory_consolidation_enabled": (bool, None, None),
    "memory_compaction_enabled": (bool, None, None),
    "memory_compaction_interval_hours": (float, 1.0, 720.0),
    "memory_compaction_max_age_days": (float, 1.0, 365.0),
    "memory_scope_by_mode": (bool, None, None),
    "memory_inject_analytical": (bool, None, None),
    "memory_inject_agentic": (bool, None, None),
    # Reranking
    "reranker_enabled": (bool, None, None),
    "reranker_top_k": (int, 1, 50),
    # Document RAG
    "document_rag_enabled": (bool, None, None),
    "document_rag_recall_limit": (int, 1, 20),
    "document_rag_contextual_retrieval": (bool, None, None),
    "document_rag_query_analysis": (bool, None, None),
    "document_rag_query_analysis_timeout": (float, 0.1, 30.0),
    "document_rag_cliff_ratio": (float, 0.0, 1.0),
    "document_rag_max_context_tokens": (int, 100, 32000),
}


@router.get("/config")
async def get_memory_config() -> JSONResponse:
    """Get all memory-related configuration settings."""
    result = {}
    for key in _MEMORY_SETTINGS:
        result[key] = getattr(settings, key, None)
    # Add string settings (not in the typed dict — no min/max range)
    result["memory_llm_extraction_model"] = getattr(settings, "memory_llm_extraction_model", "")
    result["reranker_model"] = getattr(settings, "reranker_model", "")
    result["document_rag_query_analysis_model"] = getattr(
        settings, "document_rag_query_analysis_model", "",
    )
    return JSONResponse(result)


@router.put("/config")
async def update_memory_config(request: Request) -> JSONResponse:
    """Update memory configuration settings at runtime."""
    body = await request.json()
    store = getattr(request.app.state, "settings_store", None)

    updated: dict[str, object] = {}
    errors: list[str] = []

    for key, value in body.items():
        # Handle string settings (no min/max range)
        if key in (
            "memory_llm_extraction_model",
            "reranker_model",
            "document_rag_query_analysis_model",
        ):
            coerced = str(value)[:256]
            object.__setattr__(settings, key, coerced)
            updated[key] = coerced
            if store:
                await store.set(key, coerced)
            continue

        if key not in _MEMORY_SETTINGS:
            errors.append(f"Unknown setting: {key}")
            continue

        cast_fn, min_val, max_val = _MEMORY_SETTINGS[key]

        try:
            if cast_fn is bool:
                coerced = value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
            else:
                coerced = cast_fn(value)
                if min_val is not None and coerced < min_val:
                    errors.append(f"{key}: value {coerced} below minimum {min_val}")
                    continue
                if max_val is not None and coerced > max_val:
                    errors.append(f"{key}: value {coerced} above maximum {max_val}")
                    continue
        except (ValueError, TypeError):
            errors.append(f"{key}: invalid value {value!r}")
            continue

        object.__setattr__(settings, key, coerced)
        updated[key] = coerced

        if store:
            await store.set(key, str(coerced))

    log.info("memory_config_updated", updated=updated, errors=errors)

    # Return current state
    current = {k: getattr(settings, k, None) for k in _MEMORY_SETTINGS}
    current["memory_llm_extraction_model"] = getattr(settings, "memory_llm_extraction_model", "")
    current["reranker_model"] = getattr(settings, "reranker_model", "")
    result = {"updated": updated, "current": current}
    if errors:
        result["errors"] = errors
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Memory extraction notifications (frontend poll)
# ---------------------------------------------------------------------------


@router.get("/notifications")
async def get_memory_notifications(request: Request) -> JSONResponse:
    """Return pending memory extraction notifications for the frontend.

    Returns undelivered extracted memories so chat can announce what was
    learned once while leaving provisional items available for review.
    """
    from augmentum.memory.notifications import get_undelivered

    store = _get_store(request)
    if store is None:
        return JSONResponse({"notifications": []})

    uid = _user_id(request)
    user_id = uid if uid else request.query_params.get("user_id", "default")
    notifications = await get_undelivered(store._backend.conn, user_id=user_id)
    return JSONResponse({"notifications": notifications})


@router.post("/notifications/{memory_id}/approve")
async def approve_memory(memory_id: str, request: Request) -> JSONResponse:
    """Approve a PROVISIONAL memory — promote to ACTIVE tier."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from augmentum.memory.models import MemoryTier
    mem = await store.get(memory_id, user_id=uid)
    if not mem:
        return JSONResponse({"error": "Memory not found"}, status_code=404)

    tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
    if tier_val == MemoryTier.PROVISIONAL:
        # No tier_change event: the approved memory itself appears in the
        # stream — a "tier changed" card next to it would be duplicate noise.
        await store.update_tier(memory_id, MemoryTier.ACTIVE, user_id=uid, log_change=False)
        # Clear provisional TTL (leftover is harmless since cleanup filters
        # on tier='provisional', but clearing keeps data clean)
        try:
            await store._conn.execute(
                "UPDATE memories SET provisional_expires_at = NULL "
                "WHERE id = ? AND user_id = ?",
                (memory_id, uid),
            )
            await store._conn.commit()
        except Exception:
            log.warning("clear_provisional_ttl_failed", memory_id=memory_id, exc_info=True)

    # Mark as user-approved for dream system (set regardless of tier)
    try:
        await store._conn.execute(
            "UPDATE memories SET user_approved = 1 "
            "WHERE id = ? AND user_id = ?",
            (memory_id, uid),
        )
        await store._conn.commit()
    except Exception:
        log.warning("set_user_approved_failed", memory_id=memory_id, exc_info=True)

    try:
        from augmentum.memory.notifications import resolve_notification
        await resolve_notification(store._conn, memory_id, "approved", user_id=uid)
    except Exception:
        log.warning("resolve_memory_notification_failed", memory_id=memory_id, status="approved", exc_info=True)

    # Notify dream scheduler if enabled
    dream_scheduler = getattr(request.app.state, "dream_scheduler", None)
    if dream_scheduler is not None:
        dream_scheduler.notify_approval(memory_id, user_id=uid)

    return JSONResponse({"id": memory_id, "status": "approved", "tier": "active"})


@router.post("/notifications/{memory_id}/dismiss")
async def dismiss_memory(memory_id: str, request: Request) -> JSONResponse:
    """Dismiss an extracted memory — soft-delete it."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "Memory system not initialized"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    success = await store.forget(memory_id, user_id=uid)
    try:
        from augmentum.memory.notifications import resolve_notification
        await resolve_notification(store._conn, memory_id, "dismissed", user_id=uid)
    except Exception:
        log.warning("resolve_memory_notification_failed", memory_id=memory_id, status="dismissed", exc_info=True)
    return JSONResponse({"id": memory_id, "status": "dismissed", "deleted": success})
