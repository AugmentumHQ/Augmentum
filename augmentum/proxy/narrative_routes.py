"""API routes for narrative features — prompt presets, regex scripts, groups, backgrounds."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.write_guard import (
    edit_stamp,
    incoming_stamp,
    is_stale,
    stale_payload,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/narrative", tags=["narrative"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _get_conn(request: Request):
    """Get the SQLite connection from app state."""
    state_mgr = getattr(request.app.state, "state_manager", None)
    if not state_mgr:
        return None
    backend = getattr(state_mgr, "backend", None)
    if not isinstance(backend, SQLiteBackend):
        return None
    return backend.conn


# ── Prompt Presets ──────────────────────────────────────────────────────

@router.get("/presets")
async def list_presets(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"presets": []})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from augmentum.modes.narrative.prompt_presets import PromptPresetStore
    store = PromptPresetStore(conn)
    presets = await store.list_presets(user_id=uid)
    return JSONResponse({
        "presets": [
            {
                "id": p.id, "name": p.name,
                "system_prompt": p.system_prompt,
                "jailbreak": p.jailbreak,
                "post_history": p.post_history,
                "author_note": p.author_note,
                "author_note_depth": p.author_note_depth,
                "is_default": p.is_default,
                "modular_config": p.modular_config,
                "anti_slop_phrases": p.anti_slop_phrases,
            }
            for p in presets
        ]
    })


@router.post("/presets")
async def save_preset(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()

    # Stale-write guard — a preset edited on two devices otherwise loses one
    # side silently (save_preset is INSERT OR REPLACE, so the losing edit is
    # not merely overwritten but the whole row is rebuilt). A body with no
    # id is a new preset and has nothing to be stale against.
    preset_id = body.get("id", "")
    if preset_id and await is_stale(
        conn, "prompt_presets", preset_id, incoming_stamp(body), user_id=uid,
    ):
        log.warning("preset_save_stale_rejected", preset_id=preset_id)
        return JSONResponse(stale_payload(preset_id), status_code=409)

    from augmentum.modes.narrative.prompt_presets import PromptPreset, PromptPresetStore
    store = PromptPresetStore(conn)
    preset = PromptPreset(
        id=body.get("id", ""),
        name=body.get("name", "Untitled"),
        system_prompt=body.get("system_prompt", ""),
        jailbreak=body.get("jailbreak", ""),
        post_history=body.get("post_history", ""),
        author_note=body.get("author_note", ""),
        author_note_depth=int(body.get("author_note_depth", 4)),
        is_default=bool(body.get("is_default", False)),
        modular_config=body.get("modular_config", "") or "",
        anti_slop_phrases=body.get("anti_slop_phrases", "") or "",
        client_updated_at=edit_stamp(body),
    )
    saved = await store.save_preset(preset, user_id=uid)
    return JSONResponse({"preset": {"id": saved.id, "name": saved.name}})


@router.delete("/presets/{preset_id}")
async def delete_preset(preset_id: str, request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    # Built-in presets are reseeded on every server boot, so deleting one
    # only causes a brief gap until restart — but it's still a footgun
    # when a user accidentally hits the X on "Default". Lock them.
    if preset_id.startswith("builtin_"):
        return JSONResponse(
            {"error": "Built-in presets cannot be deleted"},
            status_code=400,
        )
    from augmentum.modes.narrative.prompt_presets import PromptPresetStore
    store = PromptPresetStore(conn)
    deleted = await store.delete_preset(preset_id, user_id=uid)
    return JSONResponse({"deleted": deleted})


# ── Regex Scripts ──────────────────────────────────────────────────────

@router.get("/regex")
async def list_regex_scripts(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"scripts": []})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from augmentum.modes.narrative.regex_transformer import RegexScriptStore
    store = RegexScriptStore(conn)
    char = request.query_params.get("character")
    scripts = await store.list_scripts(character_name=char, user_id=uid)
    return JSONResponse({
        "scripts": [
            {
                "id": s.id, "name": s.name,
                "find_regex": s.find_regex,
                "replace_string": s.replace_string,
                "placement": s.placement,
                "enabled": s.enabled,
                "order_num": s.order_num,
                "character_name": s.character_name,
                # The client echoes this back as ``baseUpdatedAt`` so the
                # save guard can tell "nobody wrote since I loaded" from
                # "someone did". Without it the guard can never fire.
                "clientUpdatedAt": s.client_updated_at,
            }
            for s in scripts
        ]
    })


@router.post("/regex")
async def save_regex_script(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    from augmentum.modes.narrative.regex_transformer import RegexScript, RegexScriptStore
    store = RegexScriptStore(conn)

    # Validate regex pattern
    import re
    try:
        re.compile(body.get("find_regex", ""))
    except re.error as e:
        return JSONResponse({"error": f"Invalid regex: {e}"}, status_code=400)

    # Guard only an EDIT: a body with no id is a new script, which has
    # nothing stored to be stale against.
    script_id = body.get("id", "")
    if script_id and await is_stale(
        conn, "regex_scripts", script_id, incoming_stamp(body), user_id=uid,
    ):
        log.warning("regex_script_save_stale_rejected", script_id=script_id)
        return JSONResponse(stale_payload(script_id), status_code=409)

    script = RegexScript(
        id=script_id,
        name=body.get("name", "Untitled"),
        find_regex=body.get("find_regex", ""),
        replace_string=body.get("replace_string", ""),
        placement=body.get("placement", "output"),
        enabled=bool(body.get("enabled", True)),
        order_num=int(body.get("order_num", 100)),
        character_name=body.get("character_name"),
        client_updated_at=edit_stamp(body),
    )
    saved = await store.save_script(script, user_id=uid)
    return JSONResponse({"script": {"id": saved.id, "name": saved.name}})


@router.delete("/regex/{script_id}")
async def delete_regex_script(script_id: str, request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from augmentum.modes.narrative.regex_transformer import RegexScriptStore
    store = RegexScriptStore(conn)
    deleted = await store.delete_script(script_id, user_id=uid)
    return JSONResponse({"deleted": deleted})


@router.patch("/regex/{script_id}/toggle")
async def toggle_regex_script(script_id: str, request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    from augmentum.modes.narrative.regex_transformer import RegexScriptStore
    store = RegexScriptStore(conn)
    toggled = await store.toggle_script(script_id, bool(body.get("enabled", True)), user_id=uid)
    return JSONResponse({"toggled": toggled})


@router.post("/regex/test")
async def test_regex(request: Request) -> JSONResponse:
    """Test a regex pattern against sample text without saving."""
    body = await request.json()
    pattern = body.get("find_regex", "")
    replacement = body.get("replace_string", "")
    test_text = body.get("text", "")
    import asyncio
    import re
    try:
        compiled = re.compile(pattern)

        def _run_regex() -> dict:
            result = compiled.sub(replacement, test_text)
            matches = compiled.findall(test_text)
            return {
                "result": result,
                "match_count": len(matches),
                "matches": [str(m) for m in matches[:10]],
            }

        data = await asyncio.wait_for(
            asyncio.to_thread(_run_regex),
            timeout=2.0,
        )
        return JSONResponse(data)
    except TimeoutError:
        return JSONResponse(
            {"error": "Regex execution timed out (possible ReDoS pattern)"},
            status_code=400,
        )
    except re.error as e:
        return JSONResponse({"error": f"Invalid regex: {e}"}, status_code=400)


# ── Regex Presets ─────────────────────────────────────────────────────

@router.get("/regex/presets")
async def list_regex_presets() -> JSONResponse:
    """List available regex preset packs."""
    from augmentum.modes.narrative.regex_presets import PRESET_PACKS
    packs = []
    for key, pack in PRESET_PACKS.items():
        packs.append({
            "id": key,
            "name": pack["name"],
            "description": pack["description"],
            "tier": pack["tier"],
            "count": pack["count"],
        })
    packs.sort(key=lambda p: p["tier"])
    return JSONResponse({"packs": packs})


@router.post("/regex/presets/{pack_id}/install")
async def install_regex_preset(pack_id: str, request: Request) -> JSONResponse:
    """Install all scripts from a preset pack into the caller's library."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from augmentum.modes.narrative.regex_presets import PRESET_PACKS
    pack = PRESET_PACKS.get(pack_id)
    if not pack:
        return JSONResponse({"error": f"Unknown pack: {pack_id}"}, status_code=404)

    from augmentum.modes.narrative.regex_transformer import RegexScript, RegexScriptStore
    store = RegexScriptStore(conn)

    # Check existing scripts for THIS user only — name collisions with
    # another tenant shouldn't block installation.
    existing = await store.list_scripts(user_id=uid)
    existing_names = {s.name for s in existing}

    installed = 0
    skipped = 0
    for script_data in pack["scripts"]:
        if script_data["name"] in existing_names:
            skipped += 1
            continue
        script = RegexScript(
            name=script_data["name"],
            find_regex=script_data["find_regex"],
            replace_string=script_data["replace_string"],
            placement=script_data["placement"],
            order_num=script_data["order_num"],
        )
        await store.save_script(script, user_id=uid)
        installed += 1

    return JSONResponse({
        "installed": installed,
        "skipped": skipped,
        "pack": pack["name"],
    })


# ── Character Groups ──────────────────────────────────────────────────

@router.get("/groups")
async def list_groups(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"groups": []})
    from augmentum.modes.narrative.group_manager import GroupStore
    store = GroupStore(conn)
    groups = await store.list_groups(user_id=_user_id(request))
    return JSONResponse({
        "groups": [
            {
                "id": g.id, "name": g.name,
                "description": g.description,
                "member_names": g.member_names,
                "generation_mode": g.generation_mode,
                "member_summaries": g.member_summaries,
                "avatar": g.avatar,
                "muted_names": g.muted_names,
            }
            for g in groups
        ]
    })


@router.post("/groups")
async def save_group(request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    body = await request.json()
    from augmentum.modes.narrative.group_manager import CharacterGroup, GroupStore
    store = GroupStore(conn)
    members = body.get("member_names", [])
    if len(members) < 2:
        return JSONResponse({"error": "Groups require at least 2 members"}, status_code=400)
    raw_muted = body.get("muted_names", [])
    muted = [str(x) for x in raw_muted if x] if isinstance(raw_muted, list) else []
    # Only keep mute entries that correspond to real members — stale entries
    # after a member removal would be silently harmless but noisy in logs.
    member_set = {m for m in members}
    muted = [n for n in muted if n in member_set]
    group = CharacterGroup(
        id=body.get("id", ""),
        name=body.get("name", "Untitled Group"),
        description=body.get("description", ""),
        member_names=members,
        generation_mode=body.get("generation_mode", "round_robin"),
        member_summaries=body.get("member_summaries", {}),
        avatar=body.get("avatar", ""),
        muted_names=muted,
    )
    saved = await store.save_group(group, user_id=_user_id(request))
    return JSONResponse({"group": {
        "id": saved.id, "name": saved.name,
        "member_summaries": saved.member_summaries,
        "avatar": saved.avatar,
    }})


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, request: Request) -> JSONResponse:
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)
    from augmentum.modes.narrative.group_manager import GroupStore
    store = GroupStore(conn)
    deleted = await store.delete_group(group_id, user_id=_user_id(request))
    return JSONResponse({"deleted": deleted})


@router.post("/groups/generate-summary")
async def generate_group_summary(request: Request) -> JSONResponse:
    """Use the chat model to generate a compact character summary for group chat."""
    body = await request.json()
    card = body.get("card", {})
    char_name = body.get("character_name", "Character")

    # Build the character context from the card
    card_parts = []
    if card.get("description"):
        card_parts.append(f"Description: {card['description']}")
    if card.get("personality"):
        card_parts.append(f"Personality: {card['personality']}")
    if card.get("scenario"):
        card_parts.append(f"Scenario: {card['scenario']}")
    if card.get("systemPrompt"):
        card_parts.append(f"System prompt: {card['systemPrompt']}")
    if card.get("examples") or card.get("exampleDialogue"):
        card_parts.append(f"Example dialogue: {card.get('examples') or card.get('exampleDialogue')}")

    if not card_parts:
        return JSONResponse({"error": "No card data to summarize"}, status_code=400)

    card_text = "\n\n".join(card_parts)

    prompt = (
        f"You are summarizing the character '{char_name}' for a group chat system. "
        f"Other characters in the group will see this summary to understand who {char_name} is.\n\n"
        f"Requirements:\n"
        f"- Write 1-3 sentences maximum (under 150 characters ideal, 200 max)\n"
        f"- Focus on personality traits, speaking style, and key motivations\n"
        f"- Capture what makes this character DISTINCT from others\n"
        f"- Do NOT include physical appearance unless it's plot-relevant\n"
        f"- Do NOT use the character's name in the summary (it's already labeled)\n"
        f"- Write in present tense, third person\n\n"
        f"Full character card:\n{card_text}\n\n"
        f"Write ONLY the summary, nothing else:"
    )

    # Resolve the requested model or inherit the user's primary chat model.
    registry = getattr(request.app.state, "provider_registry", None)
    if not registry:
        return JSONResponse({"error": "No model backend available"}, status_code=503)

    try:
        model_requested = (body.get("model", "") or "").strip()
        backend, model_name = await registry.resolve_backend_with_fabric(model_requested)

        if not backend:
            return JSONResponse({"error": "No model backend available"}, status_code=503)

        from augmentum.models.base import InternalChatRequest, Message
        req = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="system", content="You write concise character summaries for a group chat system."),
                Message(role="user", content=prompt),
            ],
            temperature=0.7,
            max_tokens=200,
        )
        resp = await backend.chat(req)
        summary = resp.message.content.strip()
        # Clean up: remove quotes if the model wrapped it
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        return JSONResponse({"summary": summary})
    except Exception:
        log.warning("group_summary_generation_failed", exc_info=True)
        return JSONResponse({"error": "Summary generation failed"}, status_code=500)


@router.put("/groups/{group_id}/activate")
async def activate_group(group_id: str, request: Request) -> JSONResponse:
    """Bind a group to a session — enables group chat for that session."""
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)

    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)

    from augmentum.modes.narrative.group_manager import GroupStore, GroupTurnManager
    store = GroupStore(conn)
    group = await store.get_group(group_id, user_id=_user_id(request))
    if not group:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    # Set group on the narrative engine for this session.
    # The engine may not exist yet (created on first chat message), so
    # we also persist group_id to the narrative_memory table directly.
    from augmentum.proxy.handler_factory import _get_or_create_engine

    engines = getattr(request.app.state, "narrative_engines", None)
    if engines is None:
        from collections import OrderedDict
        engines = OrderedDict()
        request.app.state.narrative_engines = engines

    # Create engine now if it doesn't exist — ensures group_id is set
    from augmentum.modes.narrative.engine import NarrativeEngine
    from augmentum.config import settings
    uid = _user_id(request)
    cache_key: str | tuple[str, str] = (uid, session_id) if uid else session_id
    engine = engines.get(cache_key)
    if not engine:
        engine = NarrativeEngine(
            session_id=session_id,
            context_budget=settings.narrative_context_budget,
        )
        engines[cache_key] = engine

    engine.state.group_id = group_id
    engine.state.group_speaker_index = 0

    # Also update the handler's group context
    handlers = getattr(request.app.state, "narrative_handlers", {})
    handler = handlers.get(cache_key)
    if handler:
        tm = GroupTurnManager(group)
        handler._group_turn_manager = tm
        handler._active_group = group

    # Persist (uid required — empty user_id falls into the no-column INSERT
    # branch that writes NULL-uid rows readable by any tenant).
    sm = getattr(request.app.state, "state_manager", None)
    if sm:
        engine.sync_to_state()
        try:
            await sm.save_narrative_state(session_id, engine.state, user_id=uid)
        except Exception:
            log.warning("group_activate_persist_failed", session_id=session_id, exc_info=True)

    log.info("group_activated", group_id=group_id, session_id=session_id,
             group_name=group.name, engine_existed=engine is not None)

    return JSONResponse({
        "activated": True,
        "group_id": group_id,
        "group_name": group.name,
        "current_speaker": group.member_names[0] if group.member_names else "",
        "members": group.member_names,
        "mode": group.generation_mode,
    })


@router.delete("/groups/deactivate")
async def deactivate_group(request: Request) -> JSONResponse:
    """Unbind the group from a session — returns to single-character mode."""
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)

    uid = _user_id(request)
    cache_key: str | tuple[str, str] = (uid, session_id) if uid else session_id
    engines = getattr(request.app.state, "narrative_engines", {})
    engine = engines.get(cache_key)
    if engine:
        engine.state.group_id = ""
        engine.state.group_speaker_index = 0

        handlers = getattr(request.app.state, "narrative_handlers", {})
        handler = handlers.get(cache_key)
        if handler:
            handler._group_turn_manager = None
            handler._active_group = None

        sm = getattr(request.app.state, "state_manager", None)
        if sm:
            engine.sync_to_state()
            await sm.save_narrative_state(session_id, engine.state, user_id=uid)

    return JSONResponse({"deactivated": True})


@router.post("/groups/{group_id}/force-speaker")
async def force_speaker(group_id: str, request: Request) -> JSONResponse:
    """Force a specific character to speak next."""
    body = await request.json()
    session_id = body.get("session_id", "")
    speaker_name = body.get("speaker_name", "")
    if not session_id or not speaker_name:
        return JSONResponse({"error": "session_id and speaker_name required"}, status_code=400)

    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "No database"}, status_code=503)

    from augmentum.modes.narrative.group_manager import GroupStore, GroupTurnManager
    store = GroupStore(conn)
    group = await store.get_group(group_id, user_id=_user_id(request))
    if not group:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    # Find the speaker index
    tm = GroupTurnManager(group)
    if not tm.set_speaker(speaker_name):
        return JSONResponse({"error": f"Speaker '{speaker_name}' not in group"}, status_code=400)

    new_index = tm._current_index

    # Update engine state if available
    uid = _user_id(request)
    cache_key: str | tuple[str, str] = (uid, session_id) if uid else session_id
    engines = getattr(request.app.state, "narrative_engines", {})
    engine = engines.get(cache_key)
    if engine:
        engine.state.group_speaker_index = new_index

    # Update handler's cached turn manager if available
    handlers = getattr(request.app.state, "narrative_handlers", {})
    handler = handlers.get(cache_key)
    if handler and handler._group_turn_manager:
        handler._group_turn_manager.set_speaker(speaker_name)

    # Persist (uid required — see force-speaker route below for matching pattern).
    sm = getattr(request.app.state, "state_manager", None)
    if sm and engine:
        engine.sync_to_state()
        try:
            await sm.save_narrative_state(session_id, engine.state, user_id=uid)
        except Exception:
            log.warning("narrative_state_save_failed", session_id=session_id, exc_info=True)

    return JSONResponse({
        "speaker": tm.current_speaker,
        "turn_state": tm.to_dict(),
    })


@router.get("/groups/{group_id}/turn-state")
async def get_turn_state(group_id: str, request: Request) -> JSONResponse:
    """Get the current turn state for a group chat session."""
    session_id = request.query_params.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)

    uid = _user_id(request)
    cache_key: str | tuple[str, str] = (uid, session_id) if uid else session_id
    handlers = getattr(request.app.state, "narrative_handlers", {})
    handler = handlers.get(cache_key)
    if not handler or not handler._group_turn_manager:
        return JSONResponse({"turn_state": None})

    return JSONResponse({"turn_state": handler._group_turn_manager.to_dict()})


# ── Session Memory Settings ───────────────────────────────────────────

@router.get("/session/{session_id}/memory-settings")
async def get_session_memory_settings(session_id: str, request: Request) -> JSONResponse:
    """Return per-session LTM settings with effective (resolved) values."""
    from augmentum.config import settings as cfg
    from augmentum.modes.narrative.memory_settings import (
        FIELD_TO_GLOBAL,
        SessionMemorySettings,
        resolve_memory_setting,
    )

    # Try to load from engine state (in-memory) first, then DB
    uid = _user_id(request)
    cache_key: str | tuple[str, str] = (uid, session_id) if uid else session_id
    engines = getattr(request.app.state, "narrative_engines", {})
    engine = engines.get(cache_key)
    session_settings: SessionMemorySettings | None = None

    if engine and hasattr(engine.state, "memory_settings"):
        session_settings = engine.state.memory_settings

    if session_settings is None:
        # Load from DB
        sm = getattr(request.app.state, "state_manager", None)
        if sm:
            state = await sm.load_narrative_state(session_id, user_id=uid)
            if state:
                session_settings = getattr(state, "memory_settings", None)

    # Build response: overrides (what's set) + effective (resolved values)
    overrides = session_settings.to_dict() if session_settings else {}
    effective = {}
    for field_name, global_key in FIELD_TO_GLOBAL.items():
        effective[field_name] = resolve_memory_setting(
            session_settings, field_name, global_value=getattr(cfg, global_key),
        )

    return JSONResponse({
        "session_id": session_id,
        "overrides": overrides,
        "effective": effective,
    })


@router.put("/session/{session_id}/memory-settings")
async def update_session_memory_settings(
    session_id: str, request: Request,
) -> JSONResponse:
    """Update per-session LTM settings.  Only provided keys are changed."""
    from augmentum.config import settings as cfg
    from augmentum.modes.narrative.memory_settings import (
        FIELD_TO_GLOBAL,
        SessionMemorySettings,
        resolve_memory_setting,
    )

    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "Expected JSON object"}, status_code=400)

    # Validate known keys and types
    errors: list[str] = []
    _VALID_MODES = {"lite", "standard"}
    _BOOL_FIELDS = {"memory_enabled", "memory_state_enabled", "memory_ledger_enabled",
                    "memory_continuous_archive", "smart_retrieval", "memory_compaction_enabled"}
    for key, value in body.items():
        if key not in FIELD_TO_GLOBAL:
            errors.append(f"Unknown setting: {key}")
            continue
        if key == "memory_mode" and value not in _VALID_MODES:
            errors.append(f"memory_mode must be one of {_VALID_MODES}")
        if key == "memory_ledger_ceiling" and (not isinstance(value, int) or value < 0):
            errors.append("memory_ledger_ceiling must be a non-negative integer")
        if key == "smart_retrieval_count" and (not isinstance(value, int) or value < 1):
            errors.append("smart_retrieval_count must be a positive integer")
        if key == "memory_interval" and (not isinstance(value, int) or value < 1):
            errors.append("memory_interval must be a positive integer")
        if key in _BOOL_FIELDS and not isinstance(value, bool):
            errors.append(f"{key} must be a boolean")
    if errors:
        return JSONResponse({"errors": errors}, status_code=400)

    # Get or create engine + state
    from augmentum.proxy.handler_factory import _get_or_create_engine
    uid = _user_id(request)
    engines = getattr(request.app.state, "narrative_engines", None)
    if engines is None:
        from collections import OrderedDict
        engines = OrderedDict()
        request.app.state.narrative_engines = engines

    engine = _get_or_create_engine(session_id, request.app.state, user_id=uid)

    # If engine was just created (no messages yet), load persisted state from DB
    # so we don't overwrite existing memory/ledger data with empty defaults.
    if engine.state.message_count == 0:
        sm_check = getattr(request.app.state, "state_manager", None)
        if sm_check:
            persisted = await sm_check.load_narrative_state(session_id, user_id=uid)
            if persisted is not None:
                engine.load_state(persisted)

    # Merge into existing session settings
    existing = getattr(engine.state, "memory_settings", None)
    if existing is None:
        existing = SessionMemorySettings.init_from_globals()
    existing_dict = existing.to_dict()
    existing_dict.update(body)
    engine.state.memory_settings = SessionMemorySettings.from_dict(existing_dict)

    # Persist
    sm = getattr(request.app.state, "state_manager", None)
    if sm:
        engine.sync_to_state()
        try:
            await sm.save_narrative_state(session_id, engine.state, user_id=uid)
        except Exception:
            log.warning("session_memory_settings_save_failed",
                        session_id=session_id, exc_info=True)

    # Build effective response
    effective = {}
    for field_name, global_key in FIELD_TO_GLOBAL.items():
        effective[field_name] = resolve_memory_setting(
            engine.state.memory_settings, field_name,
            global_value=getattr(cfg, global_key),
        )

    return JSONResponse({
        "session_id": session_id,
        "overrides": engine.state.memory_settings.to_dict(),
        "effective": effective,
    })


@router.post("/session/{session_id}/resolve-model-alert")
async def resolve_model_alert(session_id: str, request: Request) -> JSONResponse:
    """Apply the user's choice for a stale narrative refresh-model alert.

    Raised by the handler when ``narrative_memory_model`` no longer resolves
    (a card/setting points at a model that's been removed or renamed). The
    panel surfaces it as an actionable toast; the three actions map to:

    * ``skip``           — dismiss for now; the setting is unchanged, so the
                           alert re-fires on the next refresh if still stale.
    * ``use_chat_model`` — clear ``narrative_memory_model`` so LTM rides the
                           active chat model ("Default (chat model)").
    * ``use_engine_model`` — pin ``narrative_memory_model`` to a concrete
                           engine model. ``model`` may name it explicitly;
                           otherwise the first engine-backed model is used.

    Every branch is an explicit user pick (never an auto-select) and clears
    the live alert so the toast doesn't re-nag until the model breaks again.
    """
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "Expected JSON object"}, status_code=400)
    action = str(body.get("action", "")).strip()
    if action not in {"skip", "use_chat_model", "use_engine_model"}:
        return JSONResponse({"error": "action must be skip|use_chat_model|use_engine_model"}, status_code=400)

    uid = _user_id(request)
    store = getattr(request.app.state, "settings_store", None)
    new_value: str | None = None

    if action == "use_chat_model":
        new_value = ""  # empty → resolves to the active chat model
    elif action == "use_engine_model":
        new_value = str(body.get("model", "")).strip()
        if not new_value:
            # First engine-backed model — the toast's "first available engine
            # model" option; the user explicitly opted into it.
            registry = getattr(request.app.state, "provider_registry", None)
            if registry is not None:
                try:
                    await registry.refresh_model_map()
                    default_key = getattr(registry, "_default", "")
                    for mname, mkey in getattr(registry, "_model_map", {}).items():
                        if mkey in ("engine", default_key):
                            new_value = mname.split("@", 1)[0]
                            break
                except Exception:
                    log.warning("resolve_model_alert_engine_pick_failed", exc_info=True)
            if not new_value:
                return JSONResponse({"error": "no engine model available"}, status_code=409)

    if new_value is not None and store is not None and uid:
        try:
            await store.set_user(uid, "narrative_memory_model", new_value)
        except Exception:
            log.warning("resolve_model_alert_setting_save_failed", exc_info=True)
            return JSONResponse({"error": "could not save setting"}, status_code=500)

    # Clear the live alert on the hydrated engine so the toast stops.
    # Engines are keyed by (user_id, session_id) when a user is present,
    # falling back to a bare session_id (see _hydrate_narrative_engine).
    engines = getattr(request.app.state, "narrative_engines", {}) or {}
    engine = engines.get((uid, session_id)) if uid else None
    if engine is None:
        engine = engines.get(session_id)
    if engine is not None:
        engine.pending_model_alert = None

    return JSONResponse({"ok": True, "action": action, "narrative_memory_model": new_value})


# ── Scene Backgrounds ─────────────────────────────────────────────────

@router.get("/background/{session_id}")
async def get_background(session_id: str, request: Request) -> JSONResponse:
    """Get the current auto-generated background URL for a narrative session."""
    backgrounds = getattr(request.app.state, "narrative_backgrounds", {})
    url = backgrounds.get(session_id)
    return JSONResponse({"url": url})


@router.delete("/background/{session_id}")
async def clear_background(session_id: str, request: Request) -> JSONResponse:
    """Clear the background for a narrative session."""
    backgrounds = getattr(request.app.state, "narrative_backgrounds", None)
    if backgrounds and session_id in backgrounds:
        del backgrounds[session_id]
    return JSONResponse({"cleared": True})


# ── Global Lorebook Collections ───────────────────────────────────────

@router.get("/lorebook/global")
async def list_global_collections(request: Request) -> JSONResponse:
    """List all global lorebook collections (without entries)."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"collections": []})
    try:
        cursor = await conn.execute(
            "SELECT id, name, description, entry_count, created_at "
            "FROM global_lorebook_collections ORDER BY name"
        )
        rows = await cursor.fetchall()
        return JSONResponse({"collections": [dict(r) for r in rows]})
    except Exception as exc:
        log.warning("global_lorebook_list_failed", error=str(exc))
        return JSONResponse({"collections": [], "error": str(exc)})


@router.get("/lorebook/global/{collection_id}")
async def get_global_collection(collection_id: str, request: Request) -> JSONResponse:
    """Get a collection with all its entries."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    import json

    cursor = await conn.execute(
        "SELECT id, name, description FROM global_lorebook_collections WHERE id=?",
        (collection_id,),
    )
    col = await cursor.fetchone()
    if not col:
        return JSONResponse({"error": "Not found"}, status_code=404)
    result = dict(col)

    cursor = await conn.execute(
        "SELECT id, name, keys, content, enabled, priority, position, "
        "sticky_turns, cooldown_turns, constant FROM global_lorebook_entries "
        "WHERE collection_id=? ORDER BY priority",
        (collection_id,),
    )
    rows = await cursor.fetchall()
    entries = []
    for r in rows:
        d = dict(r)
        d["keys"] = json.loads(d["keys"]) if d["keys"] else []
        d["enabled"] = bool(d["enabled"])
        d["constant"] = bool(d["constant"])
        entries.append(d)
    result["entries"] = entries
    return JSONResponse(result)


@router.post("/lorebook/global")
async def create_global_collection(request: Request) -> JSONResponse:
    """Save a character's entire lorebook as a named global collection."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    import json
    import uuid

    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    description = body.get("description", "").strip()
    entries = body.get("entries", [])
    collection_id = f"glc_{uuid.uuid4().hex[:12]}"

    await conn.execute(
        "INSERT INTO global_lorebook_collections (id, name, description, entry_count) "
        "VALUES (?, ?, ?, ?)",
        (collection_id, name, description, len(entries)),
    )

    # Batch entry inserts. Imports of large lorebooks (Janitor/SillyTavern
    # books with 100+ entries) would otherwise run 100+ sequential round-
    # trips through aiosqlite's worker thread, blocking other coroutines
    # for the duration of the import.
    rows = [
        (
            f"gle_{uuid.uuid4().hex[:12]}", collection_id,
            entry.get("name", ""),
            json.dumps(entry.get("keys", [])),
            entry.get("content", ""),
            1 if entry.get("enabled", True) else 0,
            entry.get("priority", 100),
            entry.get("position", "before_char"),
            entry.get("sticky_turns", 0),
            entry.get("cooldown_turns", 0),
            1 if entry.get("constant", False) else 0,
        )
        for entry in entries
    ]
    if rows:
        await conn.executemany(
            "INSERT INTO global_lorebook_entries "
            "(id, collection_id, name, keys, content, enabled, priority, "
            "position, sticky_turns, cooldown_turns, constant) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    await conn.commit()
    return JSONResponse({"id": collection_id, "name": name, "entries": len(entries)})


@router.put("/lorebook/global/{collection_id}")
async def update_global_collection(collection_id: str, request: Request) -> JSONResponse:
    """Update a collection's metadata (name, description)."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    body = await request.json()
    await conn.execute(
        "UPDATE global_lorebook_collections SET name=?, description=?, updated_at=datetime('now') WHERE id=?",
        (body.get("name", ""), body.get("description", ""), collection_id),
    )
    await conn.commit()
    return JSONResponse({"ok": True})


@router.delete("/lorebook/global/{collection_id}")
async def delete_global_collection(collection_id: str, request: Request) -> JSONResponse:
    """Delete a collection and all its entries (CASCADE)."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    cursor = await conn.execute(
        "DELETE FROM global_lorebook_collections WHERE id=?", (collection_id,),
    )
    await conn.commit()
    if cursor.rowcount == 0:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"deleted": True})


# ── Branch graph (Phase 4) ──────────────────────────────────────────────
#
# First-class endpoints over narrative_branches + narrative_state_snapshots +
# narrative_ledger_entries + narrative_archive (the branch-tagged tables from
# migrations 115-118). Lets the UI enumerate, preview, pin, and delete branches
# without needing a full client-side rebuild of branch state.

@router.get("/session/{session_id}/branches")
async def list_session_branches(session_id: str, request: Request) -> JSONResponse:
    """List all branches for a session, ordered by created_at.

    Query params:
      ?include_stale=false to omit branches with status=stale.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    include_stale = request.query_params.get("include_stale", "true").lower() != "false"

    from augmentum.state.narrative_persistence import NarrativePersistence
    persistence = NarrativePersistence(conn)
    branches = await persistence.list_branches(
        session_id, user_id=uid, include_stale=include_stale,
    )
    return JSONResponse({
        "branches": [
            {
                "branch_id": b.branch_id,
                "parent_branch_id": b.parent_branch_id,
                "branch_point": b.branch_point,
                "status": b.status,
                "created_at": b.created_at,
                "last_visited_at": b.last_visited_at,
            }
            for b in branches
        ],
    })


@router.get("/session/{session_id}/storage")
async def get_session_storage(session_id: str, request: Request) -> JSONResponse:
    """Per-branch + total storage observability for a session.

    Powers the LTM page storage widget so the user can make informed cleanup
    decisions instead of being surprised by SQLite file growth.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from augmentum.state.narrative_persistence import NarrativePersistence
    persistence = NarrativePersistence(conn)
    storage = await persistence.get_session_storage(session_id, user_id=uid)
    return JSONResponse({
        "session_id": storage.session_id,
        "total_branches": storage.total_branches,
        "total_archive_rows": storage.total_archive_rows,
        "total_ledger_entries": storage.total_ledger_entries,
        "total_snapshots": storage.total_snapshots,
        "total_approx_bytes": storage.total_approx_bytes,
        "branches": storage.branches,  # dict[branch_id, dict[counts]]
    })


@router.patch("/session/{session_id}/branches/{branch_id}/status")
async def set_branch_status_endpoint(
    session_id: str, branch_id: str, request: Request,
) -> JSONResponse:
    """Set a branch's lifecycle status. Body: {"status": "active"|"archived"}.

    The 'stale' state is system-set only (via the daily mark-stale sweep);
    users can opt INTO 'archived' (never auto-stale) or back to 'active'.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    new_status = body.get("status", "")
    if new_status not in ("active", "archived"):
        return JSONResponse(
            {"error": "status must be 'active' or 'archived' (stale is system-set only)"},
            status_code=400,
        )

    from augmentum.state.narrative_persistence import NarrativePersistence
    persistence = NarrativePersistence(conn)
    try:
        ok = await persistence.set_branch_status(
            session_id, branch_id, new_status, user_id=uid,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not ok:
        return JSONResponse({"error": "Branch not found"}, status_code=404)
    return JSONResponse({"branch_id": branch_id, "status": new_status})


@router.delete("/session/{session_id}/branches/{branch_id}")
async def delete_branch_endpoint(
    session_id: str, branch_id: str, request: Request,
) -> JSONResponse:
    """Permanently delete a branch + cascading content.

    Cascades archive rows, ledger entries, snapshots, branch row, vec rows
    in a single transaction. ``branch_id='main'`` is undeletable.

    Query params:
      ?cascade=true to also delete descendant branches recursively. Without
      it, returns 409 if descendants exist.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if branch_id == "main":
        return JSONResponse({"error": "main branch cannot be deleted"}, status_code=403)
    cascade = request.query_params.get("cascade", "false").lower() == "true"

    from augmentum.state.narrative_persistence import NarrativePersistence
    persistence = NarrativePersistence(conn)

    # Descendant check unless cascade=true
    if not cascade and await persistence.has_branch_descendants(
        session_id, branch_id, user_id=uid,
    ):
        return JSONResponse(
            {"error": "branch has descendants; use ?cascade=true to remove them"},
            status_code=409,
        )

    deleted_total: dict[str, int] = {
        "archive_rows": 0, "archive_vec_rows": 0,
        "ledger_entries": 0, "snapshots": 0, "branches": 0,
    }

    if cascade:
        # Walk descendants depth-first; delete from leaves upward so parent
        # rows are last to go.
        branches = await persistence.list_branches(
            session_id, user_id=uid, include_stale=True,
        )
        # Build child map
        children: dict[str, list[str]] = {}
        for b in branches:
            if b.parent_branch_id:
                children.setdefault(b.parent_branch_id, []).append(b.branch_id)

        def walk(start: str) -> list[str]:
            order: list[str] = []
            stack: list[str] = [start]
            while stack:
                cur = stack.pop()
                order.append(cur)
                stack.extend(children.get(cur, []))
            # Reverse so deepest descendants delete first
            return list(reversed(order))

        for bid in walk(branch_id):
            res = await persistence.delete_branch_cascade(session_id, bid, user_id=uid)
            for k, v in res.items():
                if k in deleted_total:
                    deleted_total[k] += v
    else:
        deleted_total = await persistence.delete_branch_cascade(
            session_id, branch_id, user_id=uid,
        )

    if deleted_total["branches"] == 0:
        return JSONResponse({"error": "Branch not found"}, status_code=404)
    return JSONResponse({"deleted": deleted_total, "branch_id": branch_id})


# ── Recall surface ──────────────────────────────────────────────────────
#
# Lookup endpoints over the narrative substrate (entities / facts /
# plot threads / archive). Designed as the data-layer half of the
# substrate-as-lookup-layer thesis (audit 2026-05-31): the engine
# stores STATE/LEDGER/ARCHIVE in the DB, but until now exposed them
# only via per-turn prompt injection. These routes let the UI and
# tests query the same data on demand. A follow-up will wire these
# functions as LLM-callable tools so the model itself can fetch
# context selectively instead of receiving a full snapshot each turn.


def _recall_persistence(request: Request):
    """Return a NarrativePersistence wired to the live SQLite conn."""
    conn = _get_conn(request)
    if not conn:
        return None
    from augmentum.state.narrative_persistence import NarrativePersistence
    return NarrativePersistence(conn)


def _recall_response(result) -> JSONResponse:
    return JSONResponse({
        "summary":          result.summary,
        "items":            result.items,
        "total_available":  result.total_available,
        "truncated":        result.truncated,
    })


@router.get("/recall/{session_id}/entity")
async def recall_entity_route(session_id: str, request: Request) -> JSONResponse:
    """Look up one entity by exact name or alias."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    persistence = _recall_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    name = request.query_params.get("name", "").strip()
    if not name:
        return JSONResponse(
            {"error": "Missing required query parameter 'name'"},
            status_code=400,
        )
    from augmentum.modes.narrative.recall import recall_entity
    result = await recall_entity(persistence, session_id, user_id=uid, name=name)
    return _recall_response(result)


@router.get("/recall/{session_id}/entities")
async def list_entities_route(session_id: str, request: Request) -> JSONResponse:
    """Enumerate entities; optional ``?type=character|location|item|faction``."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    persistence = _recall_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    from augmentum.modes.narrative.recall import list_entities
    from augmentum.state.narrative_state import EntityType
    type_param = request.query_params.get("type")
    etype = None
    if type_param:
        try:
            etype = EntityType(type_param)
        except ValueError:
            return JSONResponse(
                {"error": f"Unknown entity type '{type_param}'. "
                         f"Valid: {', '.join(t.value for t in EntityType)}"},
                status_code=400,
            )
    result = await list_entities(persistence, session_id, user_id=uid, entity_type=etype)
    return _recall_response(result)


@router.get("/recall/{session_id}/facts")
async def recall_facts_route(session_id: str, request: Request) -> JSONResponse:
    """Substring + tag search over the session's facts. ``?q=text&limit=N``."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    persistence = _recall_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    query = request.query_params.get("q", "")
    try:
        limit = int(request.query_params.get("limit", "5"))
    except ValueError:
        limit = 5
    from augmentum.modes.narrative.recall import recall_facts
    result = await recall_facts(
        persistence, session_id, user_id=uid, query=query, limit=limit,
    )
    return _recall_response(result)


@router.get("/recall/{session_id}/plot")
async def recall_plot_route(session_id: str, request: Request) -> JSONResponse:
    """Look up a plot thread by id or title substring. ``?q=text``."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    persistence = _recall_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse(
            {"error": "Missing required query parameter 'q'"},
            status_code=400,
        )
    from augmentum.modes.narrative.recall import recall_plot_thread
    result = await recall_plot_thread(persistence, session_id, user_id=uid, query=query)
    return _recall_response(result)


@router.get("/recall/{session_id}/archive")
async def recall_archive_route(session_id: str, request: Request) -> JSONResponse:
    """Semantic search over compacted ledger exchanges. ``?q=text&limit=N``."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    persistence = _recall_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse(
            {"error": "Missing required query parameter 'q'"},
            status_code=400,
        )
    try:
        limit = int(request.query_params.get("limit", "3"))
    except ValueError:
        limit = 3
    from augmentum.modes.narrative.recall import recall_archive
    result = await recall_archive(
        persistence, session_id, user_id=uid, query=query, limit=limit,
    )
    return _recall_response(result)


# ---------------------------------------------------------------------------
# World systems (card-declared manifest) — drawer data + user corrections.
# Spec: docs/superpowers/specs/2026-07-15-world-system-manifest-design.md


def _world_handler(request: Request, session_id: str):
    uid = _user_id(request)
    if not uid:
        return None, None
    cache_key: str | tuple[str, str] = (uid, session_id)
    handlers = getattr(request.app.state, "narrative_handlers", {})
    return uid, handlers.get(cache_key) or handlers.get(session_id)


@router.get("/world/{session_id}")
async def get_world_state(session_id: str, request: Request) -> JSONResponse:
    """Manifest summary + current tracker values for the World drawer.

    ``active: false`` when the session has no live handler or its card
    declares no manifest — the drawer renders nothing (absence, not an
    error state).
    """
    uid, handler = _world_handler(request, session_id)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if handler is None:
        return JSONResponse({"active": False})
    manifest = handler._world_manifest()
    if manifest is None:
        return JSONResponse({"active": False})
    store = handler._world_store(manifest)
    trackers = []
    for t in manifest.trackers:
        if not t.visible:
            continue
        owners = [o for o in store.owners() if o != "pc"] or []
        owners.insert(0, "")  # "" = the player character
        if t.scope == "world":
            owners = [""]
        for owner in owners:
            t_owner = "" if t.scope == "world" else owner
            if not store.revealed(t, t_owner):
                continue
            trackers.append({
                "id": t.id, "label": t.label, "kind": t.kind,
                "scope": t.scope, "owner": t_owner,
                "bands": t.bands or None,
                "value": store.get(t.id, t_owner),
            })
    return JSONResponse({
        "active": True,
        "world": manifest.name,
        "modules": manifest.modules,
        "player_roller": bool(manifest.dice and manifest.dice.player_roller),
        "sheet_command": manifest.sheet_command,
        "trackers": trackers,
        "tables": [
            {"id": tb.id, "label": tb.label} for tb in manifest.tables
        ],
    })


@router.post("/world/{session_id}/correct")
async def correct_world_tracker(session_id: str, request: Request) -> JSONResponse:
    """User correction from the drawer (spec D1): provenance=user, and the
    model is locked out of this tracker for USER_LOCK_TURNS turns."""
    uid, handler = _world_handler(request, session_id)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if handler is None:
        return JSONResponse({"error": "Session not active"}, status_code=404)
    manifest = handler._world_manifest()
    if manifest is None:
        return JSONResponse({"error": "No world manifest"}, status_code=404)
    body = await request.json()
    store = handler._world_store(manifest)
    ok, msg, value = store.shift(
        str(body.get("tracker") or ""),
        owner=str(body.get("owner") or ""),
        turn=handler._engine.state.message_count,
        to=body.get("to"),
        delta=body.get("delta"),
        by="user",
        reason="user correction",
    )
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    sm = getattr(request.app.state, "state_manager", None)
    if sm:
        try:
            handler._engine.sync_to_state()
            await sm.save_narrative_state(
                session_id, handler._engine.state, user_id=uid,
            )
        except Exception:
            log.warning(
                "world_correct_persist_failed",
                session_id=session_id, exc_info=True,
            )
    return JSONResponse({"ok": True, "value": value, "message": msg})


@router.post("/world/{session_id}/roll")
async def player_roll(session_id: str, request: Request) -> JSONResponse:
    """Player-initiated dice (composer Roll). Real server RNG — the result
    goes into the user's next message as an action beat; the model narrates
    around it. Requires the manifest to declare the dice module."""
    uid, handler = _world_handler(request, session_id)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if handler is None:
        return JSONResponse({"error": "Session not active"}, status_code=404)
    manifest = handler._world_manifest()
    if manifest is None or not manifest.has("dice"):
        return JSONResponse({"error": "Dice not declared"}, status_code=404)
    body = await request.json()
    from augmentum.modes.narrative.world_system import roll_dice
    result = roll_dice(str(body.get("expression") or ""))
    if result is None:
        return JSONResponse({"error": "Invalid expression"}, status_code=400)
    return JSONResponse(result)


@router.get("/world/{session_id}/suggestions")
async def get_world_suggestions(session_id: str, request: Request) -> JSONResponse:
    """Pending drift suggestions from the post-turn reconcile pass.
    Only fresh ones (current turn) — stale suggestions age out silently."""
    uid, handler = _world_handler(request, session_id)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if handler is None:
        return JSONResponse({"items": []})
    sug = getattr(handler, "_world_suggestions", None) or {}
    if sug.get("turn") != handler._engine.state.message_count:
        return JSONResponse({"items": []})
    return JSONResponse({"items": sug.get("items") or []})


@router.post("/world/{session_id}/suggestions/resolve")
async def resolve_world_suggestion(session_id: str, request: Request) -> JSONResponse:
    """Accept or dismiss one suggestion. Accept applies via the store with
    user provenance (spec D1 — the user's tap IS the authority)."""
    uid, handler = _world_handler(request, session_id)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if handler is None:
        return JSONResponse({"error": "Session not active"}, status_code=404)
    body = await request.json()
    tracker = str(body.get("tracker") or "")
    accept = bool(body.get("accept"))
    sug = getattr(handler, "_world_suggestions", None) or {}
    items = sug.get("items") or []
    match = next((i for i in items if i.get("tracker") == tracker), None)
    if match is None:
        return JSONResponse({"error": "Suggestion not found"}, status_code=404)
    result = {"ok": True, "applied": False}
    if accept:
        manifest = handler._world_manifest()
        if manifest is None:
            return JSONResponse({"error": "No world manifest"}, status_code=404)
        store = handler._world_store(manifest)
        ok, msg, value = store.shift(
            tracker, turn=handler._engine.state.message_count,
            to=match.get("to"), delta=match.get("delta"),
            by="user", reason=f"accepted suggestion: {match.get('reason', '')}",
        )
        if not ok:
            return JSONResponse({"error": msg}, status_code=400)
        result = {"ok": True, "applied": True, "value": value}
        sm = getattr(request.app.state, "state_manager", None)
        if sm:
            try:
                handler._engine.sync_to_state()
                await sm.save_narrative_state(
                    session_id, handler._engine.state, user_id=uid,
                )
            except Exception:
                log.warning("world_suggestion_persist_failed",
                            session_id=session_id, exc_info=True)
    sug["items"] = [i for i in items if i.get("tracker") != tracker]
    return JSONResponse(result)
