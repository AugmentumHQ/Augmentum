"""UI-specific API endpoints served under /api/ui/."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.state.narrative_state import EntityType, PlotStatus
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/ui", tags=["ui"])

# Single-flight de-dup for ``GET /session/{id}/state``. The narrative panel UI
# fires this from four call sites (5s interval, session-change, inspector-open,
# initial mount); on a chat completion they pile up and we've measured 11
# concurrent calls in <200ms. Each rebuilds the same in-memory state into JSON.
# When a compute is in flight for (user_id, session_id), latecomers await the
# same future and reuse its dict — only one actually does the work.
_state_inflight: dict[tuple[str, str], asyncio.Future] = {}


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


async def _hydrate_narrative_engine(
    session_id: str,
    request: Request,
    *,
    user_id: str,
):
    """Ensure the UI sees the persisted narrative state for *session_id*.

    The inspector can create an in-memory engine before the first chat turn
    after a restart. In that case the engine exists but is still empty, while
    archive rows already live in SQLite. Hydrating here keeps the inspector's
    STATE and MEMORY layers in sync with the archive layer.
    """
    cache_key: str | tuple[str, str] = (user_id, session_id) if user_id else session_id
    engines = getattr(request.app.state, "narrative_engines", None)
    engine = engines.get(cache_key)

    should_hydrate = engine is None or engine.state.message_count <= 0
    if not should_hydrate:
        return engine

    sm = getattr(request.app.state, "state_manager", None)
    if not sm:
        return engine

    try:
        saved = await sm.load_narrative_state(session_id, user_id=user_id)
        if saved is None:
            return engine
        if engine is None:
            from augmentum.proxy.handler_factory import _get_or_create_engine

            engine = _get_or_create_engine(session_id, request.app.state, user_id=user_id)
        engine.load_state(saved)
    except Exception:
        log.warning("session_state_lazy_load_failed", session_id=session_id, exc_info=True)

    return engine


@router.get("/status")
async def get_status(request: Request) -> JSONResponse:
    """Comprehensive system status for the UI dashboard."""
    state = request.app.state

    # Backend availability
    backends = {}
    registry = state.provider_registry
    for name in registry.available_backends:
        backend = registry.get_backend(name)
        available = False
        try:
            models = await backend.list_models()
            available = len(models) > 0
        except Exception:
            log.warning("diagnostics_backend_probe_failed", backend=name)
        backends[name] = {"available": available}

    # Cache stats
    cache_stats = {}
    try:
        cache_stats = state.prompt_cache.stats.to_dict()
    except Exception:
        log.warning("diagnostics_cache_stats_failed")

    # Tool registry
    tool_names = []
    try:
        tool_names = [t.name for t in state.tool_registry.list_tools()]
    except Exception:
        log.warning("diagnostics_tool_list_failed")

    return JSONResponse({
        "version": "0.1.0",
        "active_sessions": len(state.narrative_engines),
        "default_backend": settings.default_backend,
        "backends": backends,
        "tools": tool_names,
        "cache": cache_stats,
        "config": {
            "uarf_proactive_search": settings.uarf_proactive_search,
            "uarf_proactive_math": settings.uarf_proactive_math,
            "uarf_proactive_code": settings.uarf_proactive_code,
            "narrative_auto_persist": settings.narrative_auto_persist,
            "prompt_cache_enabled": settings.prompt_cache_enabled,
        },
    })


@router.get("/about")
async def get_about() -> JSONResponse:
    """Lightweight project metadata for the Settings "About" footer.

    Deliberately cheap (no backend probes like ``/status``) so it can be
    fetched on every Settings open. Version is the canonical package
    version; the rest are static project facts surfaced so self-hosters
    can see what they're running and where it lives.
    """
    from augmentum import __version__ as augmentum_version

    return JSONResponse({
        "version": augmentum_version,
        "license": "AGPL-3.0-or-later",
        "repo": "https://github.com/AugmentumHQ/Augmentum",
        "sponsors": "https://github.com/sponsors/AugmentumHQ",
        "tip": "https://donate.stripe.com/dRm14pdwxcj5glcdQS0RG02",
    })


@router.get("/settings")
async def get_settings() -> JSONResponse:
    """Get user-facing settings (safe subset)."""
    return JSONResponse({
        "default_backend": settings.default_backend,
        "default_temperature": settings.default_temperature,
        "default_top_p": settings.default_top_p,
        "default_top_k": settings.default_top_k,
        "default_num_ctx": settings.default_num_ctx,
        "uarf_max_backtracks": settings.uarf_max_backtracks,
        "uarf_max_tool_calls_per_phase": settings.uarf_max_tool_calls_per_phase,
        "uarf_confidence_threshold": settings.uarf_confidence_threshold,
        "uarf_proactive_search": settings.uarf_proactive_search,
        "uarf_proactive_math": settings.uarf_proactive_math,
        "uarf_proactive_code": settings.uarf_proactive_code,
        "narrative_context_budget": settings.narrative_context_budget,
        "narrative_auto_persist": settings.narrative_auto_persist,
        "narrative_consistency_frequency": settings.narrative_consistency_frequency,
        "prompt_cache_enabled": settings.prompt_cache_enabled,
        "prompt_cache_max_entries": settings.prompt_cache_max_entries,
        "prompt_cache_ttl": settings.prompt_cache_ttl,
    })


class TitleRequest(BaseModel):
    message: str
    model: str = ""


@router.post("/generate-title")
async def generate_title(body: TitleRequest, request: Request) -> JSONResponse:
    """Generate a short chat title from the first user message."""
    text = (body.message or "").strip()
    if not text:
        return JSONResponse({"title": "New Chat"})

    registry = getattr(request.app.state, "provider_registry", None)
    backend = None
    resolved_model = ""
    if registry:
        try:
            backend, resolved_model = await registry.resolve_model_for_role(
                "utility",
                override=body.model or "",
                settings=settings,
            )
        except Exception as exc:
            log.debug("title_utility_role_resolve_failed", error=str(exc))

    if not backend:
        # No backend available — fall back to truncation
        return JSONResponse({"title": text[:50] + ("..." if len(text) > 50 else "")})

    try:
        resp = await backend.chat(InternalChatRequest(
            model=resolved_model,
            messages=[
                Message(role="system", content=(
                    "Generate a short title (3-6 words) for a chat that starts "
                    "with the user message below. Return ONLY the title text, "
                    "no quotes, no punctuation at the end, no explanation."
                )),
                Message(role="user", content=text[:500]),
            ],
            options={"temperature": 0.3, "num_predict": 20},
        ))
        title = (resp.message.content or "").strip().strip('"\'').strip()
        if not title or len(title) > 80:
            title = text[:50] + ("..." if len(text) > 50 else "")
        return JSONResponse({"title": title})
    except Exception:
        log.debug("generate_title_failed", exc_info=True)
        return JSONResponse({"title": text[:50] + ("..." if len(text) > 50 else "")})


@router.get("/sessions")
async def list_sessions(request: Request) -> JSONResponse:
    """List all active server-side sessions (narrative engines).

    Returns session IDs and summary info so the UI can browse sessions
    created by any client (SillyTavern, Tavo, Open WebUI, etc.).
    """
    uid = _user_id(request)
    engines = getattr(request.app.state, "narrative_engines", None)
    sessions = []
    for key, engine in engines.items():
        # Keys are (user_id, session_id) when auth is active, bare
        # session_id otherwise. Only return this user's engines.
        if isinstance(key, tuple):
            key_uid, session_id = key
            if uid and key_uid != uid:
                continue
        else:
            session_id = key
        character_count = sum(
            1
            for e in engine.state.entities.values()
            if e.entity_type == EntityType.CHARACTER
        )
        sessions.append({
            "session_id": session_id,
            "character_count": character_count,
            "plot_count": len(engine.state.plot_threads),
            "entity_count": len(engine.state.entities),
        })

    return JSONResponse({
        "sessions": sessions,
        "total": len(sessions),
    })


async def _build_session_state_payload(session_id: str, request: Request, uid: str) -> dict:
    """Build the narrative-panel state dict (formerly inline in the route).

    Pure function over engine state — no shared mutation. Safe to share the
    returned dict between concurrent callers (each wraps it in its own
    JSONResponse).
    """
    engine = await _hydrate_narrative_engine(session_id, request, user_id=uid)
    if not engine:
        return {"mode": "passthrough", "state": None}

    # Build character list
    characters = []
    for entity in engine.state.entities.values():
        if entity.entity_type == EntityType.CHARACTER:
            characters.append({
                "name": entity.name,
                "emotional_state": entity.state.emotional_state or "neutral",
                "physical_state": entity.state.physical_state or "",
                "location": entity.state.location or "",
                "inventory": entity.state.inventory or [],
                "relationships": entity.state.relationships or {},
            })

    # Build scene info
    scene = engine.world_state.to_dict()

    # Build plot threads
    plots = []
    for thread in engine.state.plot_threads:
        plots.append({
            "id": thread.id,
            "title": thread.title or "Untitled",
            "description": thread.description or "",
            "status": thread.status.value if isinstance(thread.status, PlotStatus) else str(thread.status),
        })

    # Build contradictions
    contradictions = []
    for c in engine.state.contradictions[-10:]:
        contradictions.append({
            "type": c.contradiction_type,
            "description": c.description,
            "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity),
            "message_index": c.message_index,
        })

    # Build facts
    facts = []
    for f in engine.state.get_recent_facts(10):
        facts.append({
            "content": f.content,
            "source": f.source,
            "confidence": f.confidence,
            "domain": f.domain,
        })

    # Build relationships
    relationships = engine.relationship_tracker.to_dict_list()

    # Build assumptions
    assumptions = []
    for a in getattr(engine.state, "assumptions", [])[-10:]:
        assumptions.append({
            "content": getattr(a, "content", str(a)),
            "confidence": getattr(a, "confidence", None),
            "validated": getattr(a, "validated", None),
        })

    # Three-layer memory: STATE snapshot + MEMORY ledger
    state_snapshot = {}
    if engine._state_snapshot:
        state_snapshot = {k: v for k, v in engine._state_snapshot.fields.items() if v}

    memory_ledger = []
    for entry in engine._memory_ledger:
        memory_ledger.append(entry.to_dict())

    return {
        "mode": "narrative",
        "state": {
            "characters": characters,
            "scene": scene,
            "plots": plots,
            "contradictions": contradictions,
            "facts": facts,
            "relationships": relationships,
            "assumptions": assumptions,
            "message_count": engine.state.message_count,
            "memory_summary": engine.state.memory_summary or "",
            "state_snapshot": state_snapshot,
            "memory_ledger": memory_ledger,
            # Live signal when a configured refresh model no longer resolves —
            # the panel poll turns this into an actionable toast (skip / use
            # chat model / use engine model). None when everything resolves.
            "model_alert": getattr(engine, "pending_model_alert", None),
        },
    }


@router.get("/session/{session_id}/state")
async def get_session_state(session_id: str, request: Request) -> JSONResponse:
    """Get narrative state for a session (used by narrative panel).

    Single-flight: concurrent calls for the same (user_id, session_id) share
    one compute via ``_state_inflight``. The four UI call sites can fire
    nearly simultaneously on a chat completion; without dedup we'd rebuild
    the same JSON 10+ times on the event loop.
    """
    uid = _user_id(request)
    key = (uid, session_id)

    existing = _state_inflight.get(key)
    if existing is not None and not existing.done():
        try:
            data = await existing
            return JSONResponse(data)
        except Exception as exc:
            # Leader failed; fall through and recompute ourselves.
            log.debug("state_inflight_leader_failed", session_id=session_id, error=str(exc))

    fut = asyncio.get_running_loop().create_future()
    _state_inflight[key] = fut
    try:
        data = await _build_session_state_payload(session_id, request, uid)
        if not fut.done():
            fut.set_result(data)
        return JSONResponse(data)
    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        # Only clear the slot if it's still ours — paranoia against a future
        # that has already been replaced.
        if _state_inflight.get(key) is fut:
            _state_inflight.pop(key, None)


@router.get("/session/{session_id}/request-log")
async def get_request_log(session_id: str, request: Request) -> JSONResponse:
    """Get request logs for a narrative session (context viewer).

    Returns the full log list for navigation.  ``log`` contains the most
    recent entry for backward compatibility.
    """
    uid = _user_id(request)
    engine = await _hydrate_narrative_engine(session_id, request, user_id=uid)
    if not engine:
        return JSONResponse({"logs": [], "log": None, "total": 0})

    logs = engine.request_logs
    return JSONResponse({
        "logs": logs,
        "log": logs[-1] if logs else None,
        "total": len(logs),
    })


@router.get("/session/{session_id}/ltm-prompt")
async def get_ltm_prompt(session_id: str, request: Request) -> JSONResponse:
    """Get the current LTM prompt template (custom or default based on card type and mode)."""
    from augmentum.modes.narrative.memory import (
        CardType,
        SummaryMode,
        build_state_memory_prompt,
    )

    uid = _user_id(request)
    engine = await _hydrate_narrative_engine(session_id, request, user_id=uid)

    # Determine card type from engine state
    card_type_str = engine.state.card_type if engine else "character"
    try:
        card_type = CardType(card_type_str)
    except ValueError:
        card_type = CardType.CHARACTER

    # Determine current mode
    try:
        mode = SummaryMode(settings.narrative_memory_mode)
    except ValueError:
        mode = SummaryMode.STANDARD

    # The default template is composed by build_state_memory_prompt — the static
    # per-card-type _STD/_LITE_TEMPLATES dicts were refactored into it. An empty
    # custom_prompt yields the default system prompt for this card_type + mode;
    # the placeholder state/batch args only shape the (discarded) user half.
    default_prompt, _user = build_state_memory_prompt(
        card_type, None, [], [], "", 1, 1, custom_prompt="", mode=mode,
    )
    custom_prompt = settings.narrative_memory_prompt or ""

    return JSONResponse({
        "default_prompt": default_prompt,
        "custom_prompt": custom_prompt,
        "card_type": card_type.value,
        "mode": mode.value,
    })


@router.get("/session/{session_id}/archive")
async def get_session_archive(session_id: str, request: Request) -> JSONResponse:
    """List archived exchanges for a narrative session (owner-scoped).

    Cache-validated: clients send ``If-None-Match`` from a prior response's
    ``ETag`` header; if the archive hasn't changed (same MAX(turn_number)
    + COUNT) we return ``304 Not Modified`` with no body. The narrative
    inspector polls this every 5s, and the archive only changes on a new
    turn commit, so the 304 path dominates steady-state cost.

    ``Cache-Control: private, max-age=0, must-revalidate`` instructs the
    browser to keep the response in its HTTP cache and revalidate on every
    fetch — which is exactly what makes ``If-None-Match`` round-trip happen.
    Without this header the browser may not store the ETag at all, defeating
    the whole mechanism.
    """
    sm = getattr(request.app.state, "state_manager", None)
    if not sm:
        return JSONResponse({"exchanges": []})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from augmentum.state.backends.sqlite import SQLiteBackend
        from augmentum.state.narrative_persistence import NarrativePersistence

        backend = getattr(sm, "backend", None)
        if not isinstance(backend, SQLiteBackend):
            return JSONResponse({"exchanges": []})
        p = NarrativePersistence(backend.conn)

        # Conditional GET: skip the heavy SELECT when the archive hasn't
        # moved since the client's last fetch. ETag is unquoted to match
        # what JSONResponse echoes back on round-trips; per RFC 9110 the
        # comparison is byte-for-byte after parsing.
        etag = await p.archive_etag(session_id, user_id=uid)
        if etag:
            inm = request.headers.get("if-none-match", "").strip().strip('"')
            if inm and inm == etag:
                # 304 must have no message body per RFC 9110 §15.4.5 — use
                # a plain Response, not JSONResponse(None) which would
                # serialise the literal ``null``.
                return Response(
                    status_code=304,
                    headers={
                        "ETag": f'"{etag}"',
                        "Cache-Control": "private, max-age=0, must-revalidate",
                    },
                )

        exchanges = await p.list_archive_exchanges(session_id, user_id=uid)
        headers: dict[str, str] = {
            "Cache-Control": "private, max-age=0, must-revalidate",
        }
        if etag:
            headers["ETag"] = f'"{etag}"'
        return JSONResponse({"exchanges": exchanges}, headers=headers)
    except Exception:
        log.warning("archive_list_endpoint_failed", session_id=session_id, exc_info=True)
        return JSONResponse({"exchanges": []})


@router.delete("/archive/{exchange_id}")
async def delete_archive_exchange(exchange_id: str, request: Request) -> JSONResponse:
    """Delete a single archived exchange by ID (owner-scoped)."""
    sm = getattr(request.app.state, "state_manager", None)
    if not sm:
        return JSONResponse({"ok": False})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from augmentum.state.backends.sqlite import SQLiteBackend
        from augmentum.state.narrative_persistence import NarrativePersistence

        backend = getattr(sm, "backend", None)
        if not isinstance(backend, SQLiteBackend):
            return JSONResponse({"ok": False})
        p = NarrativePersistence(backend.conn)
        deleted = await p.delete_archive_exchange(exchange_id, user_id=uid)
        return JSONResponse({"ok": deleted})
    except Exception:
        log.warning("archive_delete_endpoint_failed", id=exchange_id, exc_info=True)
        return JSONResponse({"ok": False})


class UpdateNarrativeStateRequest(BaseModel):
    """Accepts any combination of editable narrative state fields."""
    memory_summary: str | None = None
    state_snapshot: dict | None = None
    memory_ledger: list | None = None

    model_config = {"extra": "ignore"}


@router.patch("/session/{session_id}/state")
async def update_session_state(
    session_id: str, body: UpdateNarrativeStateRequest, request: Request,
) -> JSONResponse:
    """Update editable narrative state fields (summary, state snapshot, ledger)."""
    user = request.scope.get("user")
    _uid = user.id if user else ""
    _ck: str | tuple[str, str] = (_uid, session_id) if _uid else session_id
    engines = getattr(request.app.state, "narrative_engines", None)
    engine = engines.get(_ck)
    if not engine:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    updated: list[str] = []

    if body.memory_summary is not None:
        engine.update_summary(body.memory_summary)
        updated.append("memory_summary")

    if body.state_snapshot is not None:
        if hasattr(engine, "apply_edited_state"):
            engine.apply_edited_state(body.state_snapshot)
            updated.append("state_snapshot")

    if body.memory_ledger is not None:
        if hasattr(engine, "apply_edited_ledger"):
            engine.apply_edited_ledger(body.memory_ledger)
            updated.append("memory_ledger")

    # Persist to DB
    state_manager = getattr(request.app.state, "state_manager", None)
    if state_manager:
        try:
            engine.sync_to_state()
            await state_manager.save_narrative_state(
                session_id, engine.state, user_id=_uid,
            )
        except Exception:
            log.warning("narrative_state_edit_persist_failed", session_id=session_id)
            return JSONResponse({"error": "Failed to persist state"}, status_code=500)

    log.info("narrative_state_edited", session_id=session_id, fields=updated)
    return JSONResponse({"ok": True})


class FetchUrlRequest(BaseModel):
    url: str
    headers: dict[str, str] | None = None
    method: str | None = None  # GET (default), POST, etc.
    body: str | None = None    # raw body for POST/PUT (already serialized)


class CharacterPortraitRequest(BaseModel):
    name: str = ""
    description: str = ""
    personality: str = ""
    scenario: str = ""
    style: str = "portrait"  # portrait, full_body, scene, rpg_card, anime, photo, group_portrait
    group_members: list[dict] | None = None  # For group_portrait: [{name, description, personality}]
    model: str = ""


_PORTRAIT_SYSTEM = """\
You are an expert image prompt engineer. Given a character's details, create \
a concise image generation prompt for Stable Diffusion / FLUX models.

Rules:
- Extract ONLY visual details: appearance, clothing, hair, eyes, build, distinguishing features
- Ignore non-visual info: personality traits, backstory, dialogue, abilities (unless they have visual effects)
- Match the requested style:
  - "portrait": head and shoulders, focused on face, studio lighting
  - "full_body": full body shot, dynamic pose, environment hints
  - "scene": character in their environment, wider composition
  - "rpg_card": fantasy art card style, ornate border feel, dramatic pose
  - "anime": anime/manga art style, cel shading
  - "photo": photorealistic, DSLR quality, natural lighting
- Add appropriate quality tags (masterpiece, best quality, highly detailed)
- Add lighting/atmosphere appropriate to the style
- Use comma-separated tags, not sentences
- Return ONLY the prompt, nothing else
- Keep under 200 words"""


@router.post("/character-portrait-prompt")
async def character_portrait_prompt(
    body: CharacterPortraitRequest, request: Request,
) -> JSONResponse:
    """Generate an image prompt from character card data using the LLM."""
    provider_reg = getattr(request.app.state, "provider_registry", None)
    if not provider_reg or not provider_reg.backends:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    # Build character summary for the LLM
    parts = []
    if body.group_members and body.style == "group_portrait":
        names = ", ".join(m.get("name", "?") for m in body.group_members)
        parts.append(f"Group portrait of: {names}")
        for member in body.group_members:
            m_parts = []
            if member.get("name"):
                m_parts.append(member["name"])
            if member.get("description"):
                m_parts.append(member["description"][:500])
            if member.get("personality"):
                m_parts.append(f"Personality: {member['personality'][:200]}")
            parts.append(" — ".join(m_parts))
    else:
        if body.name:
            parts.append(f"Character name: {body.name}")
        if body.description:
            parts.append(f"Description: {body.description[:2000]}")
        if body.personality:
            parts.append(f"Personality: {body.personality[:500]}")
    if body.scenario:
        parts.append(f"Setting: {body.scenario[:500]}")
    parts.append(f"Requested style: {body.style}")

    user_msg = "\n".join(parts)

    try:
        backend, model_name = await provider_reg.resolve_model_for_role(
            "utility",
            override=body.model or "",
            settings=settings,
        )

        chat_request = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="system", content=_PORTRAIT_SYSTEM),
                Message(role="user", content=user_msg),
            ],
            temperature=0.7,
        )
        resp = await backend.chat(chat_request)

        prompt = (resp.message.content or "").strip()
        if not prompt:
            return JSONResponse(
                {"error": "LLM returned empty prompt"}, status_code=502,
            )

        return JSONResponse({"prompt": prompt})

    except Exception as exc:
        log.warning("character_portrait_prompt_failed", error=str(exc))
        return JSONResponse(
            {"error": f"Failed to generate prompt: {exc}"}, status_code=502,
        )


class EnhanceFieldRequest(BaseModel):
    field: str  # "appearance", "description", "personality", "scenario", "greeting"
    content: str  # current field text to enhance
    context_name: str = ""  # character/persona name for context
    context_fields: dict = {}  # other filled fields for cross-reference
    model: str = ""


_ENHANCE_FIELD_PROMPTS = {
    "appearance": """\
You are enhancing the APPEARANCE field of a roleplay character card.

This field describes ONLY what you can physically see: build, height, hair, \
eyes, skin, scars, clothing, accessories, posture, distinguishing marks.

Rules:
- Only expand on what the user wrote. Do NOT invent new facts, items, or \
details that aren't implied by their text.
- Short, dense descriptors. Not prose, not a story.
- No personality, no backstory, no habits, no motivations.
- Stay under 200 words.
- Return ONLY the enhanced text. No labels or commentary.""",

    "description": """\
You are enhancing the DESCRIPTION field of a roleplay character card.

This field is the main character reference. It typically includes physical \
appearance, backstory, role, what they do, what drives them, their \
reputation, and key relationships. It can contain anything that defines \
the character EXCEPT personality traits and speech patterns (those have \
their own field).

Rules:
- Only expand on what the user wrote. Do NOT invent new facts, locations, \
occupations, or details that aren't implied by their text.
- Short, dense sentences. Factual reference style, not a story.
- No personality traits, no speech patterns, no habits — those belong in \
the personality field.
- Stay under 200 words.
- Return ONLY the enhanced text. No labels or commentary.""",

    "personality": """\
You are enhancing the PERSONALITY field of a roleplay character card.

This field covers how the character behaves: temperament, speech patterns, \
habits, values, flaws, strengths, quirks.

Rules:
- Only expand on what the user wrote. Do NOT invent traits that aren't \
implied by their text.
- Dense, list-like descriptors. How they act and think.
- No backstory, no physical appearance.
- Stay under 200 words.
- Return ONLY the enhanced text. No labels or commentary.""",

    "scenario": """\
You are enhancing the SCENARIO field of a roleplay character card.

This field sets the scene: the current situation, setting, atmosphere, \
stakes, what's happening when the story begins.

Rules:
- Only expand on what the user wrote. Do NOT invent locations or plot \
points that aren't implied by their text.
- Brief and atmospheric. Scene-setting, not narration.
- Stay under 200 words.
- Return ONLY the enhanced text. No labels or commentary.""",

    "greeting": """\
You are enhancing the GREETING field of a roleplay character card.

This is the character's first message — written AS the character, in their \
voice, in roleplay prose. It should establish tone, setting, and personality \
through action and dialogue.

Rules:
- Only expand on what the user wrote. Keep their intent and tone.
- Write in first person or third person as the character, matching the \
style of the original text.
- This field IS allowed to be prose — it's the actual first message.
- Stay under 200 words.
- Return ONLY the enhanced text. No labels or commentary.""",
}

_ENHANCE_DEFAULT_SYSTEM = """\
You are enhancing a text field for a roleplay character card. Expand on \
what the user wrote with more specificity and detail, but do NOT invent \
facts that aren't implied by their text. Use short, dense descriptors. \
Stay under 200 words. Return ONLY the enhanced text."""


@router.post("/enhance-field")
async def enhance_field(
    body: EnhanceFieldRequest, request: Request,
) -> JSONResponse:
    """Enhance a character/persona text field using the LLM."""
    provider_reg = getattr(request.app.state, "provider_registry", None)
    if not provider_reg or not provider_reg.backends:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    content = body.content.strip()
    if not content:
        return JSONResponse({"error": "Nothing to enhance"}, status_code=400)

    # Pick the field-specific system prompt
    system_prompt = _ENHANCE_FIELD_PROMPTS.get(
        body.field, _ENHANCE_DEFAULT_SYSTEM,
    )

    # Build user message — just name context + the text to enhance
    parts = []
    if body.context_name:
        parts.append(f"Character name: {body.context_name}")
    parts.append(f"Enhance this:\n{content}")

    user_msg = "\n".join(parts)

    try:
        backend, model_name = await provider_reg.resolve_model_for_role(
            "utility",
            override=body.model or "",
            settings=settings,
        )

        chat_request = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=0.8,
        )
        resp = await backend.chat(chat_request)

        enhanced = (resp.message.content or "").strip()
        if not enhanced:
            return JSONResponse(
                {"error": "LLM returned empty result"}, status_code=502,
            )

        return JSONResponse({"enhanced": enhanced, "field": body.field})

    except Exception as exc:
        log.warning("enhance_field_failed", field=body.field, error=str(exc))
        return JSONResponse(
            {"error": f"Enhancement failed: {exc}"}, status_code=502,
        )


@router.post("/fetch-url")
async def fetch_url(body: FetchUrlRequest, request: Request) -> JSONResponse:
    """Proxy-fetch a URL and return its content.

    Used by the UI to import character cards from URLs (bypasses CORS).
    Uses SafeHttpClient to block SSRF against internal/private IPs while
    still allowing legitimate external hosts (chub.ai, characterhub, etc.).
    """
    from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

    url = body.url.strip()

    if not url.startswith(("http://", "https://")):
        return JSONResponse({"error": "URL must start with http:// or https://"}, status_code=400)

    # Validate URL against SSRF (private IPs, loopback, link-local, etc.)
    safe_client = SafeHttpClient(max_response_size=10_485_760)  # 10 MB
    try:
        hostname = safe_client._validate_url(url)
        await safe_client._check_resolved_ips(hostname)
    except SafeHttpError as exc:
        log.warning("ui_fetch_url_ssrf_blocked", url=url, error=str(exc))
        return JSONResponse({"error": f"URL blocked: {exc}"}, status_code=403)

    # Family-filter enforcement on chub.ai character search. The frontend
    # constructs the upstream URL with nsfw=true&nsfl=true based on the
    # user's "SFW only" toggle; if the account is content_level=family,
    # we override those flags here regardless of what the UI sent. This
    # is the actual gate — the UI toggle is best-effort UX.
    user = request.scope.get("user")
    if user is not None and getattr(user, "is_family_filtered", False):
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host == "api.chub.ai" and parsed.path.startswith("/search"):
            qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
            qs["nsfw"] = "false"
            qs["nsfl"] = "false"
            url = urlunparse(parsed._replace(query=urlencode(qs)))
            log.info(
                "ui_fetch_url_family_filter_applied",
                user_id=user.id, host=host,
            )

    # Detect a SFW-enforced chub.ai character search (either the user ticked
    # "SFW only" — reflected as nsfw=false in the URL — or the family override
    # above forced it). Used to keyword-filter the returned nodes below.
    _parsed_final = urlparse(url)
    chub_sfw_search = (
        (_parsed_final.hostname or "").lower() == "api.chub.ai"
        and _parsed_final.path.startswith("/search")
        and dict(parse_qsl(_parsed_final.query, keep_blank_values=True))
            .get("nsfw", "").lower() == "false"
    )

    http_client: httpx.AsyncClient = getattr(request.app.state, "http_client", None)
    try:
        # Default Accept is permissive — some character-card hosts (RisuRealm)
        # content-negotiate to HTML if we explicitly prefer it.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
        if body.headers:
            headers.update(body.headers)
        # Method + body forwarding — JannyAI's /api/v1/download is POST-only
        # and several other character APIs require POST with a JSON body.
        # Without forwarding these the proxy silently degrades every call to
        # GET, returning 405 / wrong content for half the supported sources.
        method = (body.method or "GET").upper()
        if method == "GET":
            resp = await http_client.get(url, follow_redirects=True,
                                          timeout=30.0, headers=headers)
        else:
            resp = await http_client.request(
                method, url, follow_redirects=True, timeout=30.0,
                headers=headers, content=body.body,
            )
        resp.raise_for_status()

        # Post-redirect SSRF check — verify final URL didn't redirect to private IP
        final_hostname = urlparse(str(resp.url)).hostname
        if final_hostname and final_hostname != hostname:
            await safe_client._check_resolved_ips(final_hostname)
    except Exception as exc:
        log.warning("ui_fetch_url_failed", url=url, error=str(exc))
        return JSONResponse({"error": f"Failed to fetch URL: {exc}"}, status_code=502)

    content_type = resp.headers.get("content-type", "")
    ct_lower = content_type.lower()

    # Binary content (PNG character cards, charx/zip bundles, octet streams) —
    # return base64-encoded so the frontend's PNG/ZIP sniffer can crack it.
    is_binary_ct = (
        "image/" in ct_lower
        or "application/zip" in ct_lower
        or "application/x-zip" in ct_lower
        or "application/octet-stream" in ct_lower
        or "application/vnd.character" in ct_lower  # charx custom types
    )
    if is_binary_ct or url.lower().endswith((".png", ".charx", ".zip")):
        return JSONResponse({
            "type": "binary",
            "content_type": content_type,
            "data": base64.b64encode(resp.content).decode("ascii"),
        })

    # HTML page — try to extract character card PNG link and follow it
    if "text/html" in ct_lower:
        import re

        html = resp.text
        # Try download_card_image link (aicharactercards.com)
        # Try og:image PNG link
        png_url = None
        dl_match = re.search(r'href="([^"]*download_card_image[^"]*)"', html)
        if dl_match:
            png_url = dl_match.group(1).replace("&#038;", "&").replace("&amp;", "&")
        else:
            og_match = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+\.png[^"]*)"', html)
            if og_match:
                png_url = og_match.group(1)

        if png_url:
            try:
                # SSRF-check the discovered PNG URL before fetching
                png_host = safe_client._validate_url(png_url)
                await safe_client._check_resolved_ips(png_host)
                png_resp = await http_client.get(
                    png_url, follow_redirects=True, timeout=30.0, headers=headers,
                )
                png_resp.raise_for_status()
                return JSONResponse({
                    "type": "binary",
                    "content_type": png_resp.headers.get("content-type", "image/png"),
                    "data": base64.b64encode(png_resp.content).decode("ascii"),
                })
            except Exception as exc:
                log.warning("ui_fetch_card_png_failed", url=png_url, error=str(exc))

        return JSONResponse({
            "type": "text",
            "data": html,
        })

    # Text/JSON content
    try:
        payload = resp.json()
        if chub_sfw_search:
            payload = _filter_chub_sfw_nodes(payload)
        return JSONResponse({
            "type": "json",
            "data": payload,
        })
    except Exception:
        # Fall back to text only when the server claimed text. Otherwise the
        # bytes are likely a charx/zip/PNG mis-labeled — return as binary so
        # the frontend's sniffer (PNG / ZIP signature, leading 0x7B) can
        # handle it instead of corrupting the bytes through .text decoding.
        if "text/" in ct_lower or "json" in ct_lower or "xml" in ct_lower:
            return JSONResponse({
                "type": "text",
                "data": resp.text,
            })
        return JSONResponse({
            "type": "binary",
            "content_type": content_type,
            "data": base64.b64encode(resp.content).decode("ascii"),
        })


def _filter_chub_sfw_nodes(payload: object) -> object:
    """Drop chub.ai search nodes that trip the SFW keyword backstop.

    The chub search response nests the result list at ``data.nodes`` (the
    frontend reads ``apiData.data || apiData``). We scan each node's
    name/tagline/description/topics — the upstream nsfw=false flag is
    trusted but not verified, so a mistagged-SFW card would otherwise pass
    through. Non-search payloads are returned untouched.
    """
    from augmentum.discovery.safety import is_unsafe_card_text

    if not isinstance(payload, dict):
        return payload
    container = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    nodes = container.get("nodes") if isinstance(container, dict) else None
    if not isinstance(nodes, list):
        return payload

    def _unsafe(node: object) -> bool:
        if not isinstance(node, dict):
            return False
        topics = node.get("topics") or []
        topic_text = " ".join(str(t) for t in topics) if isinstance(topics, list) else str(topics)
        blob = " ".join((
            str(node.get("name") or ""),
            str(node.get("fullPath") or ""),
            str(node.get("tagline") or ""),
            str(node.get("description") or ""),
            topic_text,
        ))
        return is_unsafe_card_text(blob)

    kept = [n for n in nodes if not _unsafe(n)]
    dropped = len(nodes) - len(kept)
    if dropped:
        container["nodes"] = kept
        log.info("chub_sfw_keyword_drop", dropped=dropped, kept=len(kept))
    return payload


# ---------------------------------------------------------------------------
# RisuRealm Character Search
# ---------------------------------------------------------------------------


def _find_js_array_end(text: str, arr_start: int) -> int:
    """Find the closing ``]`` of a JS array, respecting strings."""
    depth = 0
    in_string = False
    i = arr_start
    length = len(text)
    while i < length:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < length:
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return -1


def _quote_js_keys(raw: str) -> str:
    """Convert JS object notation to JSON by quoting unquoted keys."""
    out: list[str] = []
    in_string = False
    seg_start = 0
    i = 0
    length = len(raw)
    while i < length:
        ch = raw[i]
        if in_string:
            if ch == "\\" and i + 1 < length:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in ("{", ","):
            j = i + 1
            while j < length and raw[j] in (" ", "\t", "\n", "\r"):
                j += 1
            if j < length and raw[j].isalpha():
                k = j
                while k < length and (raw[k].isalnum() or raw[k] == "_"):
                    k += 1
                if k < length and raw[k] == ":":
                    out.append(raw[seg_start:j])
                    out.append('"')
                    out.append(raw[j:k])
                    out.append('"')
                    seg_start = k
                    i = k
                    continue
        i += 1
    out.append(raw[seg_start:])
    return "".join(out)


@router.get("/risurealm/search")
async def risurealm_search(
    request: Request,
    q: str = "",
    sort: str = "recommended",
    nsfw: bool = True,
    page: int = 1,
) -> JSONResponse:
    """Search RisuRealm for character cards."""
    import json as _json

    http_client: httpx.AsyncClient = getattr(request.app.state, "http_client", None)

    # Server-side SFW enforcement: family-filtered accounts can never
    # request NSFW results regardless of what the UI sends. The client
    # toggle is best-effort UX; this is the actual gate.
    user = request.scope.get("user")
    if user is not None and getattr(user, "is_family_filtered", False):
        nsfw = False

    params: dict[str, str] = {"sort": sort, "mode": "character"}
    if q:
        params["q"] = q
    # RisuRealm defaults to SFW upstream. The previous logic only forwarded
    # `nsfw=false`, so toggling SFW-OFF in the UI quietly stayed SFW. Always
    # forward both directions so the toggle actually means something.
    params["nsfw"] = "true" if nsfw else "false"
    if page > 1:
        params["page"] = str(page)

    try:
        resp = await http_client.get(
            "https://realm.risuai.net/",
            params=params,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*",
            },
            follow_redirects=True,
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.warning("risurealm_search_failed", error=str(exc))
        return JSONResponse(
            {"error": f"RisuRealm request failed: {exc}"},
            status_code=502,
        )

    html = resp.text
    cards: list[dict] = []

    try:
        idx = html.find("cards:[")
        if idx == -1:
            log.warning("risurealm_no_cards_marker", html_length=len(html),
                        has_sveltekit="__sveltekit" in html)
        else:
            arr_start = idx + 6
            arr_end = _find_js_array_end(html, arr_start)
            if arr_end > 0:
                raw = html[arr_start:arr_end]
                jsonified = _quote_js_keys(raw)
                parsed = _json.loads(jsonified)
                cards = [_normalise_risu_card(c) for c in parsed]
            else:
                log.warning("risurealm_array_end_not_found", arr_start=arr_start)
    except Exception:
        log.warning("risurealm_parse_failed", exc_info=True)

    # Keyword backstop when SFW is enforced: the upstream nsfw=false flag is
    # trusted but not verified, so drop any card whose name/description/tags
    # carry explicit English or Korean terms (RisuRealm is Korean-first).
    if not nsfw and cards:
        before = len(cards)
        cards = [c for c in cards if not _card_text_unsafe(c)]
        dropped = before - len(cards)
        if dropped:
            log.info("risurealm_sfw_keyword_drop", dropped=dropped, kept=len(cards))

    return JSONResponse({"cards": cards, "page": page})


def _card_text_unsafe(card: dict) -> bool:
    """True when a normalised import card trips the SFW keyword backstop.

    Scans the human-readable surface (name + description + tags) that the
    upstream SFW flag is supposed to have already excluded.
    """
    from augmentum.discovery.safety import is_unsafe_card_text

    tags = card.get("tags") or []
    tag_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)
    blob = " ".join((
        str(card.get("name") or ""),
        str(card.get("description") or ""),
        tag_text,
    ))
    return is_unsafe_card_text(blob)


def _normalise_risu_card(c: dict) -> dict:
    """Normalise a RisuRealm card object for the UI."""
    img_hash = c.get("img", "")
    return {
        "id": c.get("id", ""),
        "name": c.get("name", "Unknown"),
        "description": c.get("desc", ""),
        "creator": c.get("authorname", c.get("creator", "")),
        "downloads": c.get("download", "0"),
        "tags": c.get("tags", []),
        "image_url": f"https://sv.risuai.xyz/resource/{img_hash}" if img_hash else "",
        "has_lorebook": c.get("haslore", False),
        "type": c.get("type", "normal"),
        "license": c.get("license", ""),
        "date": c.get("date", 0),
    }


# ---------------------------------------------------------------------------
# Translate Card — translate character card fields to a target language
# ---------------------------------------------------------------------------

class TranslateCardRequest(BaseModel):
    fields: dict[str, str]
    target_language: str = "English"
    source_language: str = ""  # empty → auto-detect
    model: str = ""
    preview: bool = True  # frontend renders accept/reject diff before applying


_TRANSLATE_SYSTEM_AUTO = """\
You are a precise translator for character card data. \
Auto-detect the source language and translate each field value into {lang}. \
Preserve formatting, markdown, macros ({{{{char}}}}, {{{{user}}}}), \
HTML tags, and special tokens exactly as-is — only translate \
the natural-language text around them. \
If a field is already in {lang}, return it unchanged. \
Return valid JSON with the same keys."""

_TRANSLATE_SYSTEM_FIXED = """\
You are a precise translator for character card data. \
Translate each field value from {src} into {lang}. \
Preserve formatting, markdown, macros ({{{{char}}}}, {{{{user}}}}), \
HTML tags, and special tokens exactly as-is — only translate \
the natural-language text around them. \
Return valid JSON with the same keys."""


@router.post("/translate-card")
async def translate_card(
    body: TranslateCardRequest, request: Request,
) -> JSONResponse:
    """Translate character card fields using the LLM.

    Returns ``{"translated": {field: translated_text, ...}}``. When
    ``preview`` is true, also returns ``{"source": {field: original}}``
    so the frontend can render an accept/reject diff before applying.
    """
    provider_reg = getattr(request.app.state, "provider_registry", None)
    if not provider_reg or not provider_reg.backends:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    # Filter to non-empty fields only
    fields = {k: v for k, v in body.fields.items() if v and v.strip()}
    if not fields:
        return JSONResponse({"error": "No fields to translate"}, status_code=400)

    lang = body.target_language or "English"
    src = (body.source_language or "").strip()
    if src:
        system_prompt = _TRANSLATE_SYSTEM_FIXED.format(lang=lang, src=src)
    else:
        system_prompt = _TRANSLATE_SYSTEM_AUTO.format(lang=lang)

    import json
    src_label = f"from {src} " if src else ""
    user_msg = (
        f"Translate these character card fields {src_label}to {lang}. "
        f"Return JSON with exactly the same keys.\n\n"
        f"```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```"
    )

    raw = ""
    try:
        backend, model_name = await provider_reg.resolve_model_for_role(
            "utility",
            override=body.model or "",
            settings=settings,
        )

        chat_request = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        resp = await backend.chat(chat_request)
        raw = (resp.message.content or "").strip()

        # Parse JSON from response (may be wrapped in ```json fences)
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        translated = json.loads(text)
        if not isinstance(translated, dict):
            return JSONResponse(
                {"error": "LLM returned invalid format"}, status_code=502,
            )

        payload: dict = {"translated": translated}
        if body.preview:
            # Frontend renders accept/reject per field. Echo the source
            # fields the user-facing diff is computed against so the
            # client doesn't re-derive them from possibly-edited inputs.
            payload["source"] = fields
        return JSONResponse(payload)

    except json.JSONDecodeError:
        log.warning("translate_card_json_failed", raw=raw[:200])
        return JSONResponse(
            {"error": "LLM response was not valid JSON"}, status_code=502,
        )
    except Exception as exc:
        log.warning("translate_card_failed", error=str(exc))
        return JSONResponse(
            {"error": f"Translation failed: {exc}"}, status_code=502,
        )
