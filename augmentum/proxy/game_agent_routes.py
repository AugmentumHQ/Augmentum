"""Game agent endpoints.

Exposes the :mod:`augmentum.game_agent` engine over HTTP + WebSocket:

* ``POST   /api/game-agent/sessions``                                 -- start a session
* ``GET    /api/game-agent/sessions/{id}``                            -- session status
* ``GET    /api/game-agent/sessions/{id}/log``                        -- SSE stream of the NDJSON live log
* ``POST   /api/game-agent/sessions/{id}/stop``                       -- graceful stop
* ``WS     /api/game-agent/surfaces/{kind}/bridge/{id}``              -- adapter wire (js13k / luanti)

The router stores per-process session state in
``request.app.state.game_agent_sessions`` and reads the user-supplied
LLM from ``request.app.state.game_agent_llm`` (a
:class:`augmentum.game_agent.SlowPathLLM`-compatible callable). If no
LLM is configured, session creation returns 503.

Lifecycle
---------
1. Client posts ``{"surface": ..., "objective": ..., "semantic_inputs": [...]}``.
2. For *server-side* surfaces (``mock``), the orchestrator starts
   immediately and the response includes only ``session_id``.
3. For *bridged* surfaces (``js13k``, ``luanti``, ``emulatorjs``,
   ``emulator``), the response also includes ``bridge_ws_url``; the
   session is "pending" until a client opens that WS. When the WS
   accepts, we construct a :class:`BridgedAdapter` over the live
   connection and start the orchestrator. For ``emulator`` the WS
   client is the in-container ``agent-bridge.py`` daemon dialling
   out to the host rather than a browser dialling in -- the wire
   protocol is the same.
4. ``curated`` is scaffolded in the engine; the route currently
   returns 501 for it until its platform adapter is wired.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.websockets import WebSocketState

from augmentum.config import settings
from augmentum.game_agent import (
    BridgedAdapter,
    CompanionPersona,
    MockAdapter,
    Orchestrator,
    SlowPathLLM,
    SurfaceAdapter,
    SurfaceKind,
)
from augmentum.game_agent.control import (
    ComposedProfile,
    ProfileRegistry,
    default_registry,
)
from augmentum.game_agent.control.profile import ProfileLoadError
from augmentum.game_agent.game_journals import default_journal_sections
from augmentum.game_agent.journal import CompanionJournal
from augmentum.game_agent.log import tail_log
from augmentum.game_agent.persona_loader import load_persona
from augmentum.game_agent.probes import (
    pokemon_emerald_to_dict,
    pokemon_gsc_to_dict,
    pokemon_rby_to_dict,
    pokemon_rs_to_dict,
    zelda_links_awakening_dx_to_dict,
)
from augmentum.game_agent.rule_packs import rule_engine_for_log_schema
from augmentum.game_agent.voice_bridge import VoiceBridge

# Static probe preset table. Add new entries here as more presets ship.
_PROBE_PRESETS: dict[str, Any] = {
    "pokemon_rby": pokemon_rby_to_dict,                          # Gen-1 (GB/GBC, pret/pokered)
    "pokemon_gsc": pokemon_gsc_to_dict,                          # Gen-2 (GB/GBC, pret/pokecrystal)
    "pokemon_rs":  pokemon_rs_to_dict,                           # Gen-3 Ruby/Sapphire (US v1.0, pret/pokeruby)
    "pokemon_emerald": pokemon_emerald_to_dict,                  # Gen-3 Emerald (US BPEE, pointer-deref SaveBlock)
    "zelda_links_awakening_dx": zelda_links_awakening_dx_to_dict, # GBC (datacrystal)
}

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/game-agent", tags=["game-agent"])


# ── Per-process session registry ──────────────────────────────────────


SessionStatus = Literal["pending_bridge", "running", "stopped", "error"]


@dataclass
class SessionRecord:
    """One in-flight session.

    Held in ``app.state.game_agent_sessions`` keyed by session_id. The
    process owns this state -- restarts drop sessions, which is fine
    for the single-instance design Augmentum already assumes.
    """

    session_id: str
    surface: SurfaceKind
    objective: str
    log_path: Path
    status: SessionStatus
    # Authenticated user who created this session. Every read/stop/bridge
    # endpoint verifies the requester matches before granting access --
    # otherwise any authenticated user with a session_id could read or
    # terminate another user's session. Empty string = session was
    # created without auth (e.g. dev mode with auth middleware disabled).
    owner_user_id: str = ""
    semantic_inputs: list[str] = field(default_factory=list)
    log_schema: str = ""
    companion: bool = False
    persona: CompanionPersona | None = None
    rule_engine: Any | None = None  # augmentum.game_agent.rules.RuleEngine
    journal: CompanionJournal | None = None
    # Universal control profile pair. When non-None the BridgedAdapter
    # uses profile.semantic_inputs() as the authoritative vocabulary
    # and emits wire_kind/wire_code alongside the semantic action name.
    # Absent = legacy path: semantic_inputs is the vocabulary and the
    # iframe falls back to its built-in semantic -> button table.
    profile: ComposedProfile | None = None
    orchestrator: Orchestrator | None = None
    run_task: asyncio.Task[Any] | None = None
    error: str | None = None
    # Final scorecard (SessionEndPayload.progress → ProgressScore.to_dict()),
    # captured when the orchestrator finishes so an in-process consumer (the
    # game foundry loop) can read the score without re-parsing the NDJSON log.
    # None until the run ends (or if scoring failed).
    result_progress: dict[str, Any] | None = None
    # Per-session bearer token for the bridge WS. Set on creation; the
    # bridge route accepts a ?token=<x> query param as an alternative
    # to the cookie/owner-id auth path. This is the only way an
    # un-authenticated dialler -- specifically the in-container
    # agent-bridge.py daemon -- can connect: it has no user session,
    # only the token augmentum embedded in the URL it was handed.
    # ``secrets.token_urlsafe(32)`` gives 256 bits of entropy, plenty
    # to make brute force pointless. Constant-time compare on use.
    bridge_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))


# ── App-state helpers ─────────────────────────────────────────────────


def _sessions(request_or_ws: Request | WebSocket) -> dict[str, SessionRecord]:
    store = getattr(request_or_ws.app.state, "game_agent_sessions", None)
    if store is None:
        store = cast(dict[str, SessionRecord], {})
        request_or_ws.app.state.game_agent_sessions = store
    return cast(dict[str, SessionRecord], store)


def _llm(request_or_ws: Request | WebSocket) -> SlowPathLLM | None:
    return getattr(request_or_ws.app.state, "game_agent_llm", None)


def _chat_llm(request_or_ws: Request | WebSocket) -> Any:
    """Fast-turn chat bridge (call-mode window); None on older wiring."""

    return getattr(request_or_ws.app.state, "game_agent_chat_llm", None)


def _playbook(
    request_or_ws: Request | WebSocket, owner_user_id: str | None
) -> CompanionJournal | None:
    """Per-user cross-title playbook (lessons that transfer between
    games). Same storage/merge machinery as the title journal, under a
    reserved pseudo-title. None for anonymous sessions."""

    if not owner_user_id:
        return None
    return CompanionJournal.load_or_create(
        root_dir=_journal_dir(request_or_ws),
        user_id=owner_user_id,
        title_id="__playbook__",
    )


def _voice(request_or_ws: Request | WebSocket) -> VoiceBridge | None:
    return getattr(request_or_ws.app.state, "game_agent_voice", None)


def _state_conn(request_or_ws: Request | WebSocket) -> Any:
    """Pull the aiosqlite connection from the state manager, or None.

    Lazy: returns None when the state manager is not wired or the
    backend is not SQLite. The persona loader is fail-soft so a
    missing conn just means "no persona, anonymous companion".
    """

    sm = getattr(request_or_ws.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None) if backend else None


def _user_id(request_or_ws: Request | WebSocket) -> str:
    """Pull the authenticated user_id off the scope, or empty when absent."""

    scope = getattr(request_or_ws, "scope", {})
    user = scope.get("user") if isinstance(scope, dict) else None
    return getattr(user, "id", "") or ""


def _owned_session(
    request_or_ws: Request | WebSocket, session_id: str
) -> SessionRecord | None:
    """Look up a session and verify the requester owns it.

    Returns the record only when the authenticated user_id matches the
    session's ``owner_user_id``. Any mismatch (including
    authenticated-requester vs. anonymously-created session) yields
    ``None``, which callers convert to a 404 -- not 403 -- so other
    users' session ids stay unconfirmed in error responses.
    """

    record = _sessions(request_or_ws).get(session_id)
    if record is None:
        return None
    if record.owner_user_id != _user_id(request_or_ws):
        return None
    return record


def _log_dir(request_or_ws: Request | WebSocket) -> Path:
    raw = getattr(request_or_ws.app.state, "game_agent_log_dir", None)
    if raw is None:
        # Default to /tmp/augmentum-game-agent on POSIX, %TEMP% on Win.
        import tempfile

        raw = Path(tempfile.gettempdir()) / "augmentum-game-agent"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _journal_dir(request_or_ws: Request | WebSocket) -> Path:
    """Where CompanionJournal files live — the agent's long-horizon memory.

    Resolution order:
      1. ``app.state.game_agent_journal_dir`` (test/operator override)
      2. the ``game_agent_journal_dir`` setting, when non-empty
      3. ``/data/game-agent-journals`` when ``/data`` exists (Docker —
         survives container recreation; the log dir under /tmp does not)
      4. ``<game_agent_log_dir>/journals`` (ephemeral fallback)
    """

    raw = getattr(request_or_ws.app.state, "game_agent_journal_dir", None)
    if raw is None:
        configured = (getattr(settings, "game_agent_journal_dir", "") or "").strip()
        if configured:
            raw = configured
        elif Path("/data").is_dir():
            raw = "/data/game-agent-journals"
        else:
            raw = _log_dir(request_or_ws) / "journals"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gate(request_or_ws: Request | WebSocket) -> JSONResponse | None:
    if _llm(request_or_ws) is None:
        return JSONResponse(
            {"error": "game-agent LLM is not configured (app.state.game_agent_llm)"},
            status_code=503,
        )
    return None


def _profile_registry(request_or_ws: Request | WebSocket) -> ProfileRegistry:
    """Return the in-process control profile registry.

    Deployments override the bundled registry by setting
    ``app.state.profile_registry`` before the first session POST -- e.g.
    a custom registry that loads operator-supplied JSON profiles from
    a mounted directory. Falls back to the module-level ``default_registry``
    when nothing custom has been wired.
    """

    override = getattr(request_or_ws.app.state, "profile_registry", None)
    if isinstance(override, ProfileRegistry):
        return override
    return default_registry


# ── HTTP request / response models ────────────────────────────────────


class StartSessionBody(BaseModel):
    """Body of ``POST /api/game-agent/sessions``."""

    model_config = ConfigDict(extra="forbid")

    surface: SurfaceKind
    objective: str = Field(..., min_length=1, max_length=2048)
    semantic_inputs: list[str] | None = Field(
        default=None,
        description=(
            "Required for bridged surfaces (js13k, luanti). Ignored for mock "
            "(uses a built-in vocabulary)."
        ),
    )
    log_schema: str | None = Field(
        default=None,
        description=(
            "Surface-specific vocabulary descriptor, e.g. 'js13k.v1'. Required "
            "for bridged surfaces."
        ),
    )
    companion: bool = Field(
        default=False,
        description=(
            "Enable companion mode: the slow-path prompt grows a 'say' field, "
            "voice bridge speaks any non-empty utterance through the configured "
            "TTS provider, audio rides the same bridge WS as inputs/frames."
        ),
    )
    character_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Identity for companion mode: a row id in ``ui_characters``. When "
            "provided, the persona (name + personality) is injected into the "
            "slow-path prompt and the character's ``voice`` field (if set) "
            "overrides the default TTS voice for this session. Ignored when "
            "companion is False."
        ),
    )
    title_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Library artifact id of the game being played. When supplied the "
            "session attaches a CompanionJournal keyed by (user_id, title_id) "
            "so the agent's long-running memory survives session restarts. "
            "Omit to disable cross-session memory for this session."
        ),
    )
    controller_profile: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Universal control schema: id of the controller profile (Layer-2 "
            "abstract device, e.g. 'gba', 'gambatte'). When supplied alongside "
            "``game_profile`` the server composes them via the in-process "
            "ProfileRegistry; the resulting ComposedProfile drives the slow-"
            "path INPUT_HINTS block AND adds wire_kind/wire_code to outbound "
            "WS payloads so the iframe can dispatch without re-resolving. "
            "Omit (or omit only one of the pair) to use the legacy path "
            "where semantic_inputs is the agent's vocabulary directly."
        ),
    )
    game_profile: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Universal control schema: id of the game profile (Layer-1 action "
            "vocabulary, e.g. 'pokemon_rs', 'pokemon_rby'). Only meaningful "
            "when ``controller_profile`` is also supplied."
        ),
    )


class StartSessionResponse(BaseModel):
    """Response of ``POST /api/game-agent/sessions``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    bridge_ws_url: str | None = None
    """Set for bridged surfaces; ``None`` for server-side surfaces."""


class SessionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    surface: SurfaceKind
    objective: str
    status: SessionStatus
    error: str | None = None


# ── Internal session-creation helpers ─────────────────────────────────


async def _create_bridged_session(
    request: Request | WebSocket,
    *,
    session_id: str,
    log_path: Path,
    owner_user_id: str,
    surface: SurfaceKind,
    objective: str,
    semantic_inputs: list[str] | None,
    log_schema: str | None,
    companion: bool,
    persona: CompanionPersona | None,
    journal: CompanionJournal | None,
    controller_profile: str | None,
    game_profile: str | None,
) -> SessionRecord | JSONResponse:
    """Validate inputs, compose any control profile pair, build a
    pending-bridge :class:`SessionRecord`, and register it in
    ``app.state.game_agent_sessions``.

    Returns the registered record on success or a :class:`JSONResponse`
    with the appropriate 4xx so the caller can return it directly. The
    record's ``bridge_token`` is the bearer the caller embeds in the
    URL it hands out.
    """

    if not semantic_inputs or not log_schema:
        return JSONResponse(
            {
                "error": (
                    f"surface {surface!r} requires semantic_inputs and "
                    "log_schema in the request body"
                )
            },
            status_code=400,
        )
    # Universal control profile: opt-in. When both ids are present we
    # compose them here so any error surfaces as a 400 on the session
    # POST -- not 12 seconds later when the bridge opens. Either id
    # alone is a client mistake; reject loudly rather than silently
    # demoting to legacy mode.
    composed: ComposedProfile | None = None
    if controller_profile or game_profile:
        if not (controller_profile and game_profile):
            return JSONResponse(
                {
                    "error": (
                        "controller_profile and game_profile must be supplied "
                        "together; one alone is not enough to compose."
                    )
                },
                status_code=400,
            )
        try:
            composed = _profile_registry(request).compose(
                controller_profile, game_profile,
            )
        except ProfileLoadError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    # Resolve any reflex rule pack registered for this log schema
    # (e.g., pokemon_rs.v1 -> auto-advance-dialog rule). The
    # orchestrator gets a fresh engine per session; unknown schemas
    # fall back to an empty engine (slow path only).
    record_rules = rule_engine_for_log_schema(log_schema)
    record = SessionRecord(
        session_id=session_id,
        surface=surface,
        objective=objective,
        log_path=log_path,
        status="pending_bridge",
        owner_user_id=owner_user_id,
        semantic_inputs=list(semantic_inputs),
        log_schema=log_schema,
        companion=companion,
        persona=persona,
        rule_engine=record_rules,
        journal=journal,
        profile=composed,
    )
    _sessions(request)[session_id] = record
    return record


def _public_bridge_url(scheme: str, host: str, record: SessionRecord) -> str:
    """URL the *browser* dials -- scheme/host derived from the original
    HTTP request. The token is included so a no-cookie dialler (e.g.
    a test harness) can still authenticate."""

    return (
        f"{scheme}://{host}/api/game-agent/surfaces/{record.surface}/bridge/"
        f"{record.session_id}?token={record.bridge_token}"
    )


def _container_bridge_url(record: SessionRecord) -> str | None:
    """URL the *in-container agent-bridge.py daemon* dials.

    The daemon lives on the streaming container's network namespace,
    which can't reach augmentum at the same host as a browser. The
    operator-configured ``agent_bridge_base_url`` setting names a host
    the container can resolve (Docker Desktop:
    ``ws://host.docker.internal:8080``; same-compose-network:
    ``ws://augmentum:8080``). Returns ``None`` when the setting is
    blank -- the caller surfaces this as a 503 so an unconfigured
    deployment fails loudly instead of starting a container the daemon
    can never dial out from.
    """

    base = (settings.agent_bridge_base_url or "").rstrip("/")
    if not base:
        return None
    return (
        f"{base}/api/game-agent/surfaces/{record.surface}/bridge/"
        f"{record.session_id}?token={record.bridge_token}"
    )


async def create_emulator_companion_session(
    request: Request,
    *,
    objective: str,
    semantic_inputs: list[str],
    log_schema: str,
    character_id: str | None,
    title_id: str | None,
    controller_profile: str | None,
    game_profile: str | None,
) -> tuple[SessionRecord, str] | JSONResponse:
    """Create a paired ``surface:"emulator"`` companion session for a
    streamed game.

    Use when:
    - The game-stream routes layer is starting a streamed-emulator
      container with AI enabled. The caller passes through the
      companion knobs from the stream-start request body.

    Expects:
    - ``app.state.game_agent_llm`` is wired (returns 503 otherwise).
    - ``settings.agent_bridge_base_url`` is set (returns 503 otherwise)
      so the in-container daemon has a host it can dial.

    Returns:
    - ``(SessionRecord, container_bridge_url)`` on success. The caller
      passes ``container_bridge_url`` to ``runtime.start_session`` as
      ``agent_bridge_url``. The session is registered in
      ``app.state.game_agent_sessions`` and remains
      ``pending_bridge`` until the container's daemon dials in.
    - A :class:`JSONResponse` carrying the right 4xx/503 status code
      when any precondition fails; the caller returns it verbatim.
    """

    gate = _gate(request)
    if gate is not None:
        return gate

    import uuid

    session_id = f"s_{uuid.uuid4().hex[:10]}"
    log_path = _log_dir(request) / f"{session_id}.ndjson"
    owner_user_id = _user_id(request)

    persona: CompanionPersona | None = None
    if character_id:
        conn = _state_conn(request)
        if conn is not None:
            persona = await load_persona(conn, character_id, owner_user_id)

    journal: CompanionJournal | None = None
    if owner_user_id and title_id:
        journal = CompanionJournal.load_or_create(
            root_dir=_journal_dir(request),
            user_id=owner_user_id,
            title_id=title_id,
            seed=default_journal_sections(game_profile),
        )

    result = await _create_bridged_session(
        request,
        session_id=session_id,
        log_path=log_path,
        owner_user_id=owner_user_id,
        surface="emulator",
        objective=objective,
        semantic_inputs=semantic_inputs,
        log_schema=log_schema,
        companion=True,
        persona=persona,
        journal=journal,
        controller_profile=controller_profile,
        game_profile=game_profile,
    )
    if isinstance(result, JSONResponse):
        return result

    container_url = _container_bridge_url(result)
    if container_url is None:
        # We registered the SessionRecord above; tear it back down so a
        # 503 doesn't leak an orphan record into app.state.
        _sessions(request).pop(session_id, None)
        return JSONResponse(
            {
                "error": (
                    "AI sessions require AUGMENTUM_AGENT_BRIDGE_BASE_URL to be "
                    "set so the streaming container can dial back to augmentum. "
                    "Typical values: 'ws://host.docker.internal:8080' "
                    "(Docker Desktop) or 'ws://augmentum:8080' (same compose "
                    "network)."
                )
            },
            status_code=503,
        )
    return result, container_url


# ── Routes ────────────────────────────────────────────────────────────


@router.post("/sessions", response_model=StartSessionResponse)
async def start_session(body: StartSessionBody, request: Request) -> Any:
    gate = _gate(request)
    if gate is not None:
        return gate

    import uuid

    sessions = _sessions(request)
    session_id = f"s_{uuid.uuid4().hex[:10]}"
    log_path = _log_dir(request) / f"{session_id}.ndjson"
    owner_user_id = _user_id(request)

    # Resolve companion persona once up front so both surface branches
    # share the same identity. Solo sessions skip the lookup entirely.
    persona: CompanionPersona | None = None
    if body.companion and body.character_id:
        conn = _state_conn(request)
        if conn is not None:
            persona = await load_persona(conn, body.character_id, owner_user_id)

    # Cross-session journal: only attach when both the user is identified
    # AND the client supplied a title_id. Anonymous / unscoped sessions
    # run without persistent memory. CompanionJournal.load_or_create
    # fails soft on any disk / parsing error.
    journal: CompanionJournal | None = None
    if owner_user_id and body.title_id:
        journal = CompanionJournal.load_or_create(
            root_dir=_journal_dir(request),
            user_id=owner_user_id,
            title_id=body.title_id,
            seed=default_journal_sections(body.game_profile),
        )

    if body.surface == "mock":
        record = SessionRecord(
            session_id=session_id,
            surface=body.surface,
            objective=body.objective,
            log_path=log_path,
            status="running",
            owner_user_id=owner_user_id,
            companion=body.companion,
            persona=persona,
            journal=journal,
        )
        adapter: SurfaceAdapter = MockAdapter()
        record.orchestrator = Orchestrator(
            log_path=str(log_path),
            surface_kind=body.surface,
            adapter=adapter,
            llm=cast(SlowPathLLM, _llm(request)),
            objective=body.objective,
            session_id=session_id,
            companion=body.companion,
            voice_bridge=_voice(request),
            persona=persona,
            journal=journal,
            fast_llm=_chat_llm(request),
            playbook=_playbook(request, owner_user_id),
        )
        record.run_task = asyncio.create_task(
            _run_and_finalize(record), name=f"orch-{session_id}"
        )
        sessions[session_id] = record
        return StartSessionResponse(session_id=session_id, status="running")

    if body.surface in {"js13k", "luanti", "emulatorjs", "emulator"}:
        result = await _create_bridged_session(
            request,
            session_id=session_id,
            log_path=log_path,
            owner_user_id=owner_user_id,
            surface=body.surface,
            objective=body.objective,
            semantic_inputs=body.semantic_inputs,
            log_schema=body.log_schema,
            companion=body.companion,
            persona=persona,
            journal=journal,
            controller_profile=body.controller_profile,
            game_profile=body.game_profile,
        )
        if isinstance(result, JSONResponse):
            return result
        record = result
        scheme = request.url.scheme.replace("http", "ws")
        host = request.headers.get("host", "localhost")
        bridge_url = _public_bridge_url(scheme, host, record)
        return StartSessionResponse(
            session_id=session_id, status="pending_bridge", bridge_ws_url=bridge_url
        )

    # curated -- scaffold only on the engine side. Surface the
    # state plainly so callers know what's missing rather than guessing.
    return JSONResponse(
        {
            "error": (
                f"surface {body.surface!r} is scaffolded in the engine but the route "
                "layer does not yet construct its adapter. Wire it up before using."
            )
        },
        status_code=501,
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str, request: Request) -> Any:
    record = _owned_session(request, session_id)
    if record is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return SessionStatusResponse(
        session_id=record.session_id,
        surface=record.surface,
        objective=record.objective,
        status=record.status,
        error=record.error,
    )


@router.get("/sessions/{session_id}/log")
async def stream_session_log(session_id: str, request: Request) -> Any:
    record = _owned_session(request, session_id)
    if record is None:
        return JSONResponse({"error": "no such session"}, status_code=404)

    async def _stream() -> AsyncIterator[bytes]:
        # The log file may not exist yet for a freshly-created pending
        # session; tail_log waits for it to appear.
        async for entry in tail_log(record.log_path, from_start=True):
            line = json.dumps(entry, separators=(",", ":"))
            yield f"data: {line}\n\n".encode()
            # If the session ended, drain remaining and stop.
            if entry.get("kind") == "session_end":
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str, request: Request) -> Any:
    record = _owned_session(request, session_id)
    if record is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    if record.orchestrator is None:
        # Pending bridge with no client connected yet -- drop it.
        record.status = "stopped"
        return {"session_id": session_id, "status": "stopped"}
    record.orchestrator.stop("user_stopped")
    return {"session_id": session_id, "status": "stopping"}


def _foundry_payloads(request_or_ws: Request | WebSocket) -> dict[str, dict]:
    """Per-process store of foundry play-host payloads, keyed by session_id.

    The game foundry loop stashes the composed-game payload here before
    pointing the headless browser at the play-host URL; the route below
    renders it. Cleared by the loop when the play pass ends.
    """
    store = getattr(request_or_ws.app.state, "foundry_play_payloads", None)
    if store is None:
        store = cast(dict[str, dict], {})
        request_or_ws.app.state.foundry_play_payloads = store
    return cast(dict[str, dict], store)


@router.get("/foundry/play-host/{session_id}")
async def foundry_play_host(
    session_id: str, request: Request, token: str = "", view: int = 0,
) -> Any:
    """Serve the play-host page for a foundry-generated game.

    A thin bootstrap that reuses the SAME client ``composeBundle`` the human
    Play path uses (so the agent-bridge shim has ONE source), mounts the
    generated game in an iframe, and (in agent mode) lets the injected shim
    dial the same-origin bridge WS. Token-auth (constant-time) against the
    session's bridge token — this page is loaded by the headless browser which
    carries no cookie, exactly like the bridge WS itself.

    ``view=1`` renders the game WITHOUT the agent-bridge shim: a read-only
    preview for the theater to embed, so it never opens a second controlling
    connection to the bridge (which would double inputs against the agent).
    """
    record = _sessions(request).get(session_id)
    payload = _foundry_payloads(request).get(session_id)
    if record is None or payload is None:
        return JSONResponse({"error": "no such foundry session"}, status_code=404)
    if not (token and hmac.compare_digest(token, record.bridge_token)):
        return JSONResponse({"error": "bad or missing token"}, status_code=403)
    if view:
        # Strip the agent bridge — the theater just watches the artifact render.
        payload = {**payload, "agentBridge": None}
    return HTMLResponse(_render_play_host(payload))


def _render_play_host(payload: dict) -> str:
    """Render the bootstrap HTML embedding the composed-game payload.

    ``payload`` = {html, files: {relpath: content}, entry, agentBridge}.
    The bridge ``wsUrl`` is a same-origin PATH (the shim prefixes ws(s)://
    location.host), so the only reachability requirement is that the browser
    can load THIS page from augmentum.
    """
    data = json.dumps(payload)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>foundry play</title>"
        "<style>html,body{margin:0;height:100%;background:#000}"
        "iframe{border:0;width:100vw;height:100vh;display:block}</style></head>"
        "<body>"
        f"<script>window.__PLAY__ = {data};</script>"
        "<script type=\"module\">\n"
        "import { composeBundle } from '/ui/scripts/bundle-composer.js';\n"
        "const P = window.__PLAY__;\n"
        "const files = Object.entries(P.files || {}).map(([path, f]) => "
        "({ path, content: f.c, encoding: f.e || 'text' }));\n"
        "const doc = composeBundle(P.html, files, P.entry, {}, P.agentBridge);\n"
        "const f = document.createElement('iframe');\n"
        "f.setAttribute('sandbox', 'allow-scripts allow-same-origin');\n"
        "f.srcdoc = doc;\n"
        "document.body.appendChild(f);\n"
        "</script></body></html>"
    )


@router.get("/probes/{preset_name}")
async def get_probe_preset(preset_name: str) -> Any:
    """Return a RAM probe preset by name.

    The browser bridge fetches one of these on session start and uses
    it to drive its per-tick memory reads. Presets are static; adding
    a new one means adding a Python module under
    ``augmentum/game_agent/probes/`` and registering it in
    :data:`_PROBE_PRESETS`.
    """

    builder = _PROBE_PRESETS.get(preset_name)
    if builder is None:
        return JSONResponse(
            {"error": f"no probe preset named {preset_name!r}"},
            status_code=404,
        )
    return builder()


@router.websocket("/surfaces/{kind}/bridge/{session_id}")
async def bridge_ws(websocket: WebSocket, kind: SurfaceKind, session_id: str) -> None:
    await websocket.accept()

    if _llm(websocket) is None:
        await _ws_error(websocket, "game-agent LLM is not configured")
        return

    # Two-path auth. Token wins when present + correct -- this is the
    # only path the in-container agent-bridge.py daemon can take
    # because it has no user cookie. Browser clients fall through to
    # the user-owner check.
    token_param = websocket.query_params.get("token") or ""
    record = _sessions(websocket).get(session_id)
    if record is not None and token_param and hmac.compare_digest(
        record.bridge_token, token_param,
    ):
        # Token-authenticated dialler. The session_id alone is not
        # enough -- the token closes the oracle gap.
        pass
    else:
        record = _owned_session(websocket, session_id)
        # Anonymous sessions carry owner_user_id "" — which equals the
        # ""  every unauthenticated dialler reports, so the owner check
        # degenerates to "knows the session id" and silently re-opens the
        # exact oracle the bridge token exists to close. Demand the token
        # whenever there is no real owner to check against. Authenticated
        # sessions are unaffected. Real clients always dial the
        # server-issued bridge_ws_url, which already embeds the token.
        if record is not None and not record.owner_user_id:
            record = None
    if record is None:
        # 404-equivalent: hide whether the session exists from non-owners,
        # otherwise the WS endpoint becomes a session-id oracle.
        await _ws_error(websocket, "no such session")
        return
    if record.surface != kind:
        await _ws_error(websocket, f"session was created for surface {record.surface!r}, not {kind!r}")
        return
    if record.status != "pending_bridge":
        await _ws_error(websocket, f"session is in status {record.status!r}, cannot accept bridge")
        return
    if kind not in {"js13k", "luanti", "emulatorjs", "emulator"}:
        await _ws_error(websocket, f"surface {kind!r} is not a bridged surface")
        return

    adapter = BridgedAdapter(
        websocket=websocket,
        surface_kind=kind,
        semantic_inputs=record.semantic_inputs,
        log_schema=record.log_schema,
        # Bridge loss maps to "aborted" -- session ended externally, not by
        # the user or by completion. Keep this aligned with the
        # SessionEndPayload.reason Literal in schema.py.
        on_bridge_disconnect=lambda: _request_stop(record, "aborted"),
        profile=record.profile,
    )
    orchestrator = Orchestrator(
        log_path=str(record.log_path),
        surface_kind=kind,
        adapter=adapter,
        llm=cast(SlowPathLLM, _llm(websocket)),
        objective=record.objective,
        session_id=session_id,
        companion=record.companion,
        voice_bridge=_voice(websocket),
        persona=record.persona,
        rule_engine=record.rule_engine,
        journal=record.journal,
        fast_llm=_chat_llm(websocket),
        playbook=_playbook(websocket, record.owner_user_id),
    )
    record.orchestrator = orchestrator
    record.status = "running"
    record.run_task = asyncio.create_task(_run_and_finalize(record), name=f"orch-{session_id}")

    # Hold the WebSocket open until the orchestrator finishes; the
    # adapter's read loop is what actually does the receiving, but
    # we need the handler to stay alive so starlette doesn't close
    # the socket on us.
    import contextlib

    with contextlib.suppress(WebSocketDisconnect):
        await record.run_task


# ── Internals ─────────────────────────────────────────────────────────


async def _run_and_finalize(record: SessionRecord) -> None:
    """Drive an orchestrator to completion and update the SessionRecord."""

    assert record.orchestrator is not None
    try:
        end = await record.orchestrator.run()
        record.result_progress = getattr(end, "progress", None)
        record.status = "stopped"
    except Exception as exc:  # noqa: BLE001
        record.status = "error"
        record.error = str(exc)
        log.error("game_agent.session_error", session_id=record.session_id, error=str(exc))


def _request_stop(record: SessionRecord, reason: str) -> None:
    if record.orchestrator is not None:
        record.orchestrator.stop(reason)


async def _ws_error(ws: WebSocket, message: str) -> None:
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps({"error": message}))
        await ws.close()
