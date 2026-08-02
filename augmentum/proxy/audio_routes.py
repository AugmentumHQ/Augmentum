"""OpenAI-compatible audio routes — TTS and STT proxy endpoints.

Proxies requests to user-configured audio providers (any OpenAI-compatible
TTS/STT API). Providers are managed via the /api/audio/providers CRUD API
and stored in SQLite.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import numpy as np

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.proxy import system_events
from augmentum.proxy.session import SESSION_HEADER
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.http_client import SharedHTTPClient, normalize_base_url
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key, sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(tags=["audio"])

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AudioProviderCreate(BaseModel):
    id: str
    provider_type: str = Field(..., pattern="^(tts|stt)$")
    name: str
    base_url: str
    api_key: str | None = None
    default_model: str = ""
    default_voice: str = ""
    tts_chunking: str = "sentence"


class AudioProviderUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    default_model: str | None = None
    default_voice: str | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    tts_chunking: str | None = None


class TTSRequest(BaseModel):
    model: str = ""
    input: str
    voice: str = ""
    response_format: str = "mp3"
    speed: float = 1.0
    instruct: str = ""  # Emotion/style instruction (Qwen3-TTS)
    # Conversation id for context-aware engines (Sesame CSM). Forwarded to
    # the provider as the ``X-Augmentum-Session`` header so the sidecar can
    # condition prosody on the same conversation's recent turns. Normally
    # carried by that header on the wire; this body field is the channel
    # used across fabric, where only the JSON body is signed + forwarded.
    session_id: str = ""


class STTRequest(BaseModel):
    model: str = ""
    language: str = ""
    response_format: str = "json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn(request: Request):
    """Get the aiosqlite connection from app state."""
    sm = getattr(request.app.state, "state_manager", None)
    if sm and isinstance(sm.backend, SQLiteBackend):
        return sm.backend.conn
    return None


def _get_read_conn(request: Request):
    """Prefer the dedicated hot-read connection when configured.

    aiosqlite serialises every query on a connection through a single
    worker thread, so the TTS dispatch's ``audio_providers`` lookup —
    fired on every speech request, sometimes 1000s/hour for voice mode
    — was sitting behind whatever the main connection's writer queue
    was processing. Routing the SELECT through ``app.state.read_conn``
    keeps it off the writer thread.

    Falls back to the main connection when read_conn isn't initialised
    (early-lifespan, tests, or if the dedicated connection failed to
    open at startup), so callers don't have to branch.
    """
    read = getattr(request.app.state, "read_conn", None)
    if read is not None:
        return read
    return _get_conn(request)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


async def _get_default_provider(conn, provider_type: str) -> dict | None:
    """Fetch the default (or first enabled) provider for a given type.

    Local-first: any locally-configured provider wins. When no local
    default exists AND fabric is enabled AND a peer is advertising the
    requested kind (TTS or STT), synthesise a provider dict pointing
    at that peer. This is what makes "main box has no STT installed,
    voice-box runs Moonshine" work as a default — operators don't
    have to copy the voice-box URL into a local DB row by hand.
    """
    cursor = await conn.execute(
        "SELECT id, name, base_url, api_key, default_model, default_voice, tts_chunking "
        "FROM audio_providers WHERE provider_type = ? AND is_enabled = 1 "
        "ORDER BY is_default DESC LIMIT 1",
        (provider_type,),
    )
    row = await cursor.fetchone()
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "base_url": row[2],
            "api_key": decrypt_api_key(row[3]),
            "default_model": row[4],
            "default_voice": row[5],
            "tts_chunking": row[6] or "sentence",
        }

    return _fabric_default_provider(provider_type)


def _fabric_default_provider(provider_type: str) -> dict | None:
    """Return a synthesised provider dict for the first connected peer
    advertising the requested kind, or None if none are available.

    Picks the first match deterministically (capability registry order)
    — Phase 2 of distribution work; future enhancement could rank by
    advertised free capacity / observed latency like the LLM director
    does in :func:`director._score_llm_candidates`.
    """
    coord = _fabric_coordinator
    if coord is None:
        return None

    from augmentum.fabric.capabilities import (
        KIND_STT_TRANSCRIBE,
        KIND_TTS_SYNTHESIZE,
    )

    target_kind = KIND_TTS_SYNTHESIZE if provider_type == "tts" else (
        KIND_STT_TRANSCRIBE if provider_type == "stt" else ""
    )
    if not target_kind:
        return None

    try:
        matches = coord.find_peers_with_capability(target_kind)
    except Exception:
        log.warning("fabric_default_provider_lookup_failed", exc_info=True)
        return None

    for node_id, cap in matches:
        peer_pid = getattr(cap, "provider_id", "")
        if not peer_pid:
            continue
        fabric_id = f"{_FABRIC_PROVIDER_PREFIX}{node_id}:{peer_pid}"
        prov = _fabric_provider_dict(fabric_id)
        if prov is not None:
            return prov
    return None


def _fabric_audio_provider_entries(coordinator: Any = None) -> list[dict]:
    """Every connected fabric peer's TTS/STT providers, shaped for the
    ``/api/audio/providers`` list so the picker can SELECT a peer-hosted
    engine — not merely fall back to one as a silent default.

    This is the audio twin of the image ``/models`` peer-merge (Phase 8):
    a service installed on a peer (e.g. Speaches) is registered there by
    the provider bridge, advertised in that peer's heartbeat, and — with
    this merge — shows up selectable in every other node's picker, badged
    with the peer's hostname/icon. Selection just works: the request path
    already resolves a ``fabric:<node>:<pid>`` id via
    ``_get_provider_by_id`` → ``_fabric_provider_dict`` (which rebuilds
    the real peer URL at call time, so the list entry carries no URL).

    Self-contained for testability: takes the coordinator explicitly
    (defaults to the module global) and builds entries straight from the
    advertised capability + paired-peer metadata.
    """
    coord = coordinator if coordinator is not None else _fabric_coordinator
    if coord is None:
        return []

    from augmentum.fabric.capabilities import (
        KIND_STT_TRANSCRIBE,
        KIND_TTS_SYNTHESIZE,
    )

    out: list[dict] = []
    seen: set[str] = set()
    for kind, ptype in ((KIND_TTS_SYNTHESIZE, "tts"), (KIND_STT_TRANSCRIBE, "stt")):
        try:
            matches = coord.find_peers_with_capability(kind)
        except Exception:
            log.debug("fabric_audio_list_lookup_failed", exc_info=True)
            continue
        for node_id, cap in matches:
            peer_pid = getattr(cap, "provider_id", "")
            if not peer_pid:
                continue
            fabric_id = f"{_FABRIC_PROVIDER_PREFIX}{node_id}:{peer_pid}"
            if fabric_id in seen:
                continue
            seen.add(fabric_id)
            state = None
            try:
                state = coord.peer_state(node_id)
            except Exception:
                state = None
            paired = getattr(state, "paired", None) if state is not None else None
            host = (getattr(paired, "hostname", "") or "")[:40] or node_id[:12]
            icon = getattr(paired, "icon", "") or ""
            label = getattr(cap, "provider_name", "") or peer_pid
            out.append({
                "id": fabric_id,
                "provider_type": ptype,
                "name": f"{label} ({host})",
                "base_url": "",
                "default_model": getattr(cap, "default_model", "") or "",
                "default_voice": getattr(cap, "default_voice", "") or "",
                "is_enabled": True,
                "is_default": False,
                "tts_chunking": "sentence",
                "fabric": True,
                "fabric_node_id": node_id,
                "fabric_node_hostname": host,
                "fabric_node_icon": icon,
            })
    return out


async def _get_provider_by_id(conn, provider_id: str) -> dict | None:
    # Fabric-virtualised provider: synthesise a provider dict from the
    # peer capability registry. Lets the existing resolution path treat
    # peer-served engines as ordinary providers.
    if provider_id.startswith(_FABRIC_PROVIDER_PREFIX):
        return _fabric_provider_dict(provider_id)

    cursor = await conn.execute(
        "SELECT id, provider_type, name, base_url, api_key, default_model, "
        "default_voice, is_enabled, is_default, tts_chunking FROM audio_providers WHERE id = ?",
        (provider_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "provider_type": row[1],
        "name": row[2],
        "base_url": row[3],
        "api_key": decrypt_api_key(row[4]),
        "default_model": row[5],
        "default_voice": row[6],
        "is_enabled": bool(row[7]),
        "is_default": bool(row[8]),
        "tts_chunking": row[9] or "sentence",
    }


def _parse_fabric_provider_id(fabric_id: str) -> tuple[str, str] | None:
    """Split ``fabric:<node_id>:<peer_provider_id>`` into the parts.
    Returns ``(node_id, peer_provider_id)`` or None when malformed.
    Node IDs are URL-safe so no escaping concerns.
    """
    if not fabric_id.startswith(_FABRIC_PROVIDER_PREFIX):
        return None
    body = fabric_id[len(_FABRIC_PROVIDER_PREFIX):]
    # node_id is the first segment, peer_provider_id may itself contain
    # colons (e.g. "kokoro-builtin") — split exactly once.
    node_id, _, peer_pid = body.partition(":")
    if not node_id or not peer_pid:
        return None
    return node_id, peer_pid


def _fabric_provider_dict(fabric_id: str) -> dict | None:
    """Synthesise a provider dict matching ``_get_provider_by_id``'s
    shape from a fabric peer capability. Returns None when the peer
    is no longer connected or the capability has been withdrawn —
    callers treat that the same as a deleted/disabled DB row.

    The extra ``fabric_*`` fields aren't read by today's downstream
    TTS HTTP path (signed-header injection lands in a follow-up). They
    surface here now so the UI badge + admin status views have
    everything they need without a second lookup.
    """
    coordinator = _fabric_coordinator
    if coordinator is None:
        return None

    parsed = _parse_fabric_provider_id(fabric_id)
    if parsed is None:
        return None
    node_id, peer_provider_id = parsed

    from augmentum.fabric.capabilities import (
        KIND_STT_TRANSCRIBE,
        KIND_TTS_SYNTHESIZE,
    )

    peer_state = coordinator.peer_state(node_id)
    if peer_state is None or not peer_state.connected:
        return None

    for cap in peer_state.capabilities:
        cap_kind = getattr(cap, "kind", "")
        if cap_kind not in (KIND_TTS_SYNTHESIZE, KIND_STT_TRANSCRIBE):
            continue
        if getattr(cap, "provider_id", "") != peer_provider_id:
            continue
        provider_type = "tts" if cap_kind == KIND_TTS_SYNTHESIZE else "stt"
        peer = peer_state.paired
        addr = peer.addr if peer else ""
        if not addr:
            return None
        # Normalise to a wss/https URL the existing HTTP helpers accept.
        # Operators store addr as either bare host:port or with a
        # scheme; trust scheme when present, default to https.
        if "://" not in addr:
            addr = f"https://{addr}"
        # base_url is the API root — the TTS/STT code appends
        # "/v1/audio/speech" itself, so we deliberately leave the
        # capability's base_url_path off here. The path field is
        # reserved for future per-peer non-standard mounts.
        base_url = addr.rstrip("/")
        peer_name = getattr(cap, "provider_name", "") or peer_provider_id
        host = peer.hostname if peer else ""
        icon = peer.icon if peer else ""
        if host:
            host_label = f"{icon} {host}".strip() if icon else host
            display = f"{peer_name} ({host_label})"
        else:
            display = peer_name
        return {
            "id": fabric_id,
            "provider_type": provider_type,
            "name": display,
            "base_url": base_url,
            "api_key": "",  # peer auth rides on signed envelope headers, not API keys
            "default_model": getattr(cap, "default_model", ""),
            "default_voice": getattr(cap, "default_voice", ""),
            "is_enabled": True,
            "is_default": False,
            "tts_chunking": "sentence",
            "fabric_node_id": node_id,
            "fabric_node_addr": addr,
            "fabric_node_hostname": host,
            "fabric_node_icon": icon,
            "fabric_peer_provider_id": peer_provider_id,
        }
    return None


# Built-in in-process TTS engines (no sidecar, no HTTP). Two families:
#   * Kokoro — StyleTTS2, quality tier
#   * Pocket TTS — Kyutai flow-based LM + Mimi codec, multilingual CPU tier
_BUILTIN_TTS_IDS = {"kokoro-builtin", "pockettts-builtin"}

# Voice-walk basenames are interpolated into filesystem paths at delete
# time. The clone endpoint sanitises on the way in, but anything reaching
# the delete path that violates this pattern indicates either DB
# tampering or a future code path that skipped the sanitiser. Refusing
# the file unlink (DB row still gets removed) prevents an arbitrary-file
# delete primitive without changing the user-facing behaviour for any
# legitimate walk name.
_VOICE_WALK_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _is_safe_voice_walk_name(name: str) -> bool:
    """True iff ``name`` is a safe basename to interpolate into a path."""
    if not name:
        return False
    return bool(_VOICE_WALK_NAME_RE.match(name))


async def _builtin_tts_engine(provider_id: str):
    """Return a loaded built-in in-process TTS engine, or None if unavailable.

    Covers kokoro-builtin and pockettts-builtin (same `is_available` /
    `load_model` / `stream_speech` / `generate` / `_resolve_voice` shape).
    Never raises — the caller decides how to react to a model that won't load.
    """
    try:
        if provider_id == "kokoro-builtin":
            from augmentum.voice.kokoro_tts import KokoroTTS
            eng = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
        elif provider_id == "pockettts-builtin":
            from augmentum.voice.pocket_tts import PocketTTS
            eng = PocketTTS.instance(
                model_dir=settings.tts_pocket_model_dir,
                language=settings.tts_pocket_language,
            )
        else:
            return None
        if not eng.is_available:
            await load_model_off_loop(eng.load_model)
        return eng if eng.is_available else None
    except Exception as exc:  # noqa: BLE001
        log.warning("builtin_tts_engine_load_failed", provider=provider_id, error=str(exc))
        return None


def _build_headers(api_key: str | None, *, base_url: str = "") -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        if base_url and _is_deepgram(base_url):
            # Deepgram uses "Token {key}" auth format
            headers["Authorization"] = f"Token {api_key}"
        elif base_url and _is_elevenlabs(base_url):
            # ElevenLabs uses custom "xi-api-key" header
            headers["xi-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ---------------------------------------------------------------------------
# Shared httpx Clients (one verified, one unverified — reused across requests)
# ---------------------------------------------------------------------------

_audio_http = SharedHTTPClient()

# Max text length for TTS — prevents oversized payloads to providers
MAX_TTS_TEXT_CHARS = 4096


async def close_audio_clients() -> None:
    """Close module-level httpx clients. Called during server shutdown."""
    await _audio_http.close()
    if _fabric_audio_http is not None and not _fabric_audio_http.is_closed:
        await _fabric_audio_http.aclose()


# Keep the name ``_audio_client`` so that existing call-sites and the
# ``voice/pipeline.py`` import continue to work unchanged.
_audio_client = _audio_http.get


# Fabric peers present self-signed certs (Caddy local CA) and are trusted via
# ed25519-signed envelopes (peer_middleware), NOT the CA chain — so fabric
# audio dispatch always skips TLS verification, matching every other fabric
# client (see fabric/lifespan.py). The shared _audio_client can't be reused:
# a peer reachable at a LAN IP (e.g. 192.168.x) isn't classified local by
# is_local_url(), so it would verify and reject the self-signed cert — the
# fabric_tts CERTIFICATE_VERIFY_FAILED → 502 voice-preview failure.
_fabric_audio_http: httpx.AsyncClient | None = None
_fabric_audio_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def _fabric_audio_client(base_url: str = ""):
    """Pooled verify=False httpx client for fabric peer audio dispatch."""
    global _fabric_audio_http
    async with _fabric_audio_lock:
        if _fabric_audio_http is None or _fabric_audio_http.is_closed:
            _fabric_audio_http = httpx.AsyncClient(verify=False, follow_redirects=True)
    yield _fabric_audio_http


# ---------------------------------------------------------------------------
# TTS — POST /v1/audio/speech
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# ---------------------------------------------------------------------------
# Voice → Provider resolution cache
# Maps voice names to provider IDs so users can just pick a voice name
# and the system auto-routes to the correct TTS backend.
# Voice → full list of provider_ids advertising it. Index 0 is the
# preferred (local if available, else first peer); subsequent indices
# are peer alternatives in heartbeat-arrival order. Used by
# ``_apply_voice_routing_mode`` to honor the voice_routing_mode setting
# (auto = sources[0]; round_robin = cycle; pin = look for the pinned id).
_voice_sources_map: dict[str, list[str]] = {}

# Per-voice request counter for round-robin routing. Incremented on each
# resolved request that hits round-robin mode; modulo source count picks
# the dispatch target. Module-global is fine: this is a single-process
# counter and cross-process synchronization isn't necessary because the
# only consequence of skew between workers is slightly uneven distribution.
_voice_rr_counters: dict[str, int] = {}

# Populated by list_voices(), used by tts_speech() and voice_routes.py.
# ---------------------------------------------------------------------------
_voice_provider_map: dict[str, str] = {}  # voice_name → provider_id
_voice_map_ts: float = 0.0
_VOICE_MAP_TTL = 300.0  # 5 minutes
_voice_map_lock = asyncio.Lock()

# Fabric coordinator handle. Populated by lifespan after fabric init so
# voice-map refresh + provider-by-id can walk the peer capability
# registry without threading the coordinator through every call site.
# Stays None on solo installs / when fabric is disabled — every consumer
# guards on None and falls back to the local-only path.
_fabric_coordinator: Any = None

# Sentinel marking a virtualised provider_id: the receiver-side handle
# for a peer-served audio engine. Format:
#     fabric:<node_id>:<peer_provider_id>
# Single colons inside; the outer split on "::" (voice separator) stays
# unambiguous because peer_provider_ids never contain "::".
_FABRIC_PROVIDER_PREFIX = "fabric:"


def register_fabric_coordinator(coordinator: Any) -> None:
    """Lifespan hook: tell the audio routes module which fabric
    coordinator to consult for peer-served provider discovery. Idempotent
    — safe to call on every lifespan start; passing None deregisters
    (e.g. when fabric is being torn down for tests).
    """
    global _fabric_coordinator, _voice_provider_map, _voice_map_ts
    _fabric_coordinator = coordinator
    # Force a refresh on next lookup so newly-discovered peer voices
    # surface immediately rather than waiting out the 5-min TTL.
    _voice_provider_map = {}
    _voice_map_ts = 0.0


def invalidate_voice_caches() -> None:
    """Reset all voice-related caches.

    Must be called after any audio provider CRUD operation (create, update,
    delete, set-default) so that subsequent TTS requests resolve against
    current DB state instead of stale in-memory maps.
    """
    global _voice_provider_map, _voice_map_ts, _voice_api_cache
    _voice_provider_map = {}
    _voice_map_ts = 0.0
    _voice_api_cache = {}
    # Tell connected UIs to refetch /api/audio/voices. This is the post-CRUD
    # hook for audio providers (server-scoped, so broadcast), which is why
    # the publish lives here — every provider create/update/delete already
    # funnels through this function, so the UI signal can't drift from the
    # cache reset. Per-user voice mix/clone mutations emit voices.changed at
    # their own success paths with user_id scoping.
    system_events.publish("voices.changed", {"reason": "provider"})


async def resolve_voice_provider(
    conn, voice: str
) -> tuple[dict | None, str]:
    """Resolve a voice name to its owning provider.

    If voice is empty (i.e. "provider default"), skips the voice map
    entirely and returns the default TTS provider from the DB.
    If voice contains '::' (explicit prefix), uses that provider.
    Otherwise looks up the voice in the cached voice→provider map
    and routes to the correct backend automatically.

    Returns (provider_dict, clean_voice_name).
    """
    import time

    # Empty voice = "provider default" — go straight to DB default,
    # no map lookup needed.
    if not voice or not voice.strip():
        provider = await _get_default_provider(conn, "tts")
        return provider, voice

    # Explicit provider prefix: "qwen-tts::Vivian"
    if "::" in voice:
        provider_id, voice_name = voice.split("::", 1)
        provider = await _get_provider_by_id(conn, provider_id)
        if provider and provider.get("is_enabled", True):
            return provider, voice_name
        log.warning("voice_explicit_provider_not_found", provider_id=provider_id)
        # Fall through to auto-resolution

    global _voice_provider_map, _voice_map_ts

    # Refresh cache if stale (lock prevents concurrent rebuilds)
    now = time.time()
    async with _voice_map_lock:
        if now - _voice_map_ts > _VOICE_MAP_TTL or not _voice_provider_map:
            await _refresh_voice_provider_map(conn)

    # Look up voice in map — try exact match first. matched_name is the
    # canonical voice key in the source map (used to read full source
    # list for routing-mode application).
    matched_name = voice if voice in _voice_provider_map else ""

    if not matched_name:
        # Case-insensitive + base-name (strip language tag) fallback.
        # "Vivian" matches "Vivian (CN)"; "Ryan" matches "Ryan (EN)".
        voice_lower = voice.lower().strip()
        for vname in _voice_provider_map:
            vname_lower = vname.lower()
            if vname_lower == voice_lower:
                matched_name = vname
                break
            base_name = vname.split(" (")[0].lower() if " (" in vname else ""
            if base_name and base_name == voice_lower:
                matched_name = vname
                break

    if matched_name:
        # Apply routing mode against the FULL source list (not just the
        # preferred provider_id). In auto mode this returns the same
        # provider as the preferred map; in round_robin / pin modes it
        # may return a different peer source. The is_enabled check is
        # only meaningful for external provider rows; fabric peer
        # provider_ids are virtual handles routed via dispatcher.
        sources = _voice_sources_map.get(matched_name, [])
        if sources:
            provider_id = _apply_voice_routing_mode(matched_name, sources)
        else:
            provider_id = _voice_provider_map.get(matched_name, "")

        if provider_id:
            provider = await _get_provider_by_id(conn, provider_id)
            if provider and provider.get("is_enabled", True):
                # Return the original voice name (without tag) for the TTS
                # request — providers know their own voices by base name.
                return provider, voice

    # Fallback to default provider
    provider = await _get_default_provider(conn, "tts")
    return provider, voice


async def _refresh_voice_provider_map(conn) -> None:
    """Rebuild the voice→provider mapping from all enabled TTS providers.

    Two parallel maps are produced:

    * ``_voice_provider_map`` — name → preferred provider_id. Preserves
      local-first precedence for back-compat with callers that only read
      the single preferred target.
    * ``_voice_sources_map`` — name → ordered list of every provider_id
      that advertises the voice (preferred first). Consumed by
      ``_apply_voice_routing_mode`` to apply auto/round_robin/pin.
    """
    import time
    global _voice_provider_map, _voice_sources_map, _voice_map_ts

    new_map: dict[str, str] = {}
    new_sources: dict[str, list[str]] = {}

    def _add_source(vname: str, pid: str) -> None:
        """Add ``pid`` as a source for ``vname``. First add wins the
        ``_voice_provider_map`` slot (preferred-routing target); every
        add appends to ``_voice_sources_map`` (full source list).
        """
        if not vname or not pid:
            return
        if vname not in new_map:
            new_map[vname] = pid
        sources = new_sources.setdefault(vname, [])
        if pid not in sources:
            sources.append(pid)

    # Built-in Kokoro voices
    if settings.tts_kokoro_builtin and not settings.tts_kokoro_url:
        try:
            from augmentum.voice.kokoro_tts import KokoroTTS
            kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
            if kokoro.is_available:
                for name in kokoro.get_voices():
                    _add_source(name, "kokoro-builtin")
        except Exception as exc:
            log.debug("kokoro_voice_map_refresh_failed", error=str(exc))

    # Built-in PocketTTS voices — including cloned voices under
    # ``/data/voices/`` since pockettts-builtin resolves them
    # transparently via ``PocketTTS._resolve_clone_path``. Without this
    # block, custom voices like "eve" never make it into the map and
    # every request for them falls through to the default-provider
    # branch (which is usually kokoro-builtin), so the user gets the
    # wrong voice everywhere — char cards, narrative, global default.
    if settings.tts_pocket_builtin:
        try:
            from augmentum.voice.pocket_tts import PocketTTS
            pocket = PocketTTS.instance(model_dir=settings.tts_pocket_model_dir)
            if pocket.is_available:
                for name in pocket.get_voices():
                    _add_source(name, "pockettts-builtin")
                # Cloned voice files live alongside Chatterbox's clones
                # in /data/voices — PocketTTS picks them up at synthesis
                # time. Register each so the voice→provider lookup
                # routes them to pockettts-builtin instead of the
                # default-provider fallback.
                for v in _list_local_voices():
                    name = v.get("name") or v.get("id") or ""
                    if name:
                        _add_source(name, "pockettts-builtin")
        except Exception as exc:
            log.debug("pockettts_voice_map_refresh_failed", error=str(exc))

    # External providers
    try:
        cursor = await conn.execute(
            "SELECT id, name, base_url, api_key, default_model, default_voice "
            "FROM audio_providers WHERE provider_type = 'tts' AND is_enabled = 1"
        )
        rows = await cursor.fetchall()
        for r in rows:
            prov = {
                "id": r[0], "name": r[1], "base_url": r[2],
                "api_key": decrypt_api_key(r[3]), "default_model": r[4], "default_voice": r[5],
            }
            if prov["base_url"] == "builtin":
                continue
            try:
                voices = await _fetch_voices_from_provider(prov)
                for v in voices:
                    vname = v.get("name") or v.get("voice_id", "")
                    _add_source(vname, prov["id"])
            except Exception:
                log.warning("voice_fetch_failed", provider=prov.get("name"), exc_info=True)
    except Exception:
        log.warning("voice_provider_db_query_failed", exc_info=True)

    # Fabric peers: walk the capability registry and add peer voices to
    # the sources map under "fabric:<node>:<peer_provider_id>" handles.
    # Local provider_ids win the "preferred" slot via _add_source's
    # first-write-wins semantics — we never shadow a locally-wired voice
    # in the default routing path. But peer sources are now visible to
    # the routing layer, which is what enables round-robin/pin modes.
    peer_voice_count = 0
    if _fabric_coordinator is not None:
        try:
            from augmentum.fabric.capabilities import KIND_TTS_SYNTHESIZE
            for node_id, cap in _fabric_coordinator.find_peers_with_capability(
                KIND_TTS_SYNTHESIZE,
            ):
                peer_pid = getattr(cap, "provider_id", "")
                if not peer_pid:
                    continue
                fabric_id = f"{_FABRIC_PROVIDER_PREFIX}{node_id}:{peer_pid}"
                for vname in getattr(cap, "voices", []) or []:
                    if not vname:
                        continue
                    before = vname in new_sources
                    _add_source(vname, fabric_id)
                    if not before:
                        peer_voice_count += 1
        except Exception:
            log.warning("voice_provider_fabric_walk_failed", exc_info=True)

    _voice_provider_map = new_map
    _voice_sources_map = new_sources
    _voice_map_ts = time.time()
    log.info(
        "voice_provider_map_refreshed",
        voices=len(new_map),
        peer_voices=peer_voice_count,
        multi_source_voices=sum(1 for s in new_sources.values() if len(s) > 1),
    )


def _apply_voice_routing_mode(voice: str, sources: list[str]) -> str:
    """Pick the actual provider_id from a voice's source list, applying
    the operator-configured voice_routing_mode.

    ``sources`` is the full list of provider_ids that advertise this
    voice (preferred/local first, peers after). Returns the chosen
    provider_id; empty ``sources`` → empty string (caller falls back to
    its own defaults).

    Modes:

      * **auto**: return sources[0]. Local-first when local advertises.
      * **round_robin**: cycle through sources on each call so a voice
        with 3 sources gets requests distributed roughly evenly. Useful
        when one box is the chat box (busy decoding) and others are
        relatively idle for TTS.
      * **pin**: if the pinned provider_id appears in this voice's
        sources, route there. Otherwise fall back to sources[0] (don't
        fail the request just because the pinned box happens to not
        carry this specific voice — e.g. user pinned Box 3 for
        Chatterbox but is requesting a Kokoro voice Box 3 doesn't
        advertise).
    """
    if not sources:
        return ""
    mode = (settings.voice_routing_mode or "auto").strip().lower()
    if mode == "pin":
        pinned = (settings.voice_routing_pin_provider or "").strip()
        if pinned and pinned in sources:
            return pinned
        # Pinned target doesn't carry this voice; fall back to preferred.
        # Debug-level: a mass-pin will hit this for every voice the pinned
        # box doesn't have — info would be too chatty.
        log.debug(
            "voice_routing_pin_miss",
            voice=voice, pinned=pinned, sources=sources,
        )
        return sources[0]
    if mode == "round_robin":
        idx = _voice_rr_counters.get(voice, 0) % len(sources)
        _voice_rr_counters[voice] = idx + 1
        return sources[idx]
    # "auto" (default + fallback for unknown modes)
    return sources[0]


def _build_tts_stream(
    base_url: str,
    model: str,
    voice: str,
    text: str,
    response_format: str,
    speed: float,
    api_key: str | None,
    *,
    pre_cleaned: bool = False,
    instruct: str = "",
    stream_pcm: bool = False,
    provider_id: str = "",
    user_id: str = "",
    session_id: str = "",
):
    """Build an async generator that streams TTS audio from the provider.

    Handles provider-specific API differences:
    - Deepgram: POST /v1/speak?model={voice} with raw text body
    - ElevenLabs: POST /v1/text-to-speech/{voice_id}/stream with custom body
    - OpenAI / generic: POST /v1/audio/speech with JSON body
    - Fabric peer (provider_id starts with "fabric:") — OpenAI shape with
      signed envelope headers (X-Fabric-*); no API key path.

    ``user_id`` is required when ``provider_id`` is a fabric handle —
    the receiving peer's FabricPeerMiddleware verifies it against the
    signed envelope before honouring the request. Non-fabric calls
    ignore it.

    Text is automatically sanitized (markdown, special symbols, formatting
    artifacts stripped) before sending to the provider unless pre_cleaned=True.
    """
    if not pre_cleaned:
        if provider_id == "chatterbox-turbo":
            # Convert RP markers to Turbo tags before cleaning, then
            # clean the rest (markdown, symbols) while preserving [tags]
            from augmentum.voice.emotion import inject_turbo_tags
            text = inject_turbo_tags(text)
            from augmentum.voice.text_cleaning import clean_for_tts
            text = clean_for_tts(text, is_narrative=True, preserve_brackets=True)
        else:
            from augmentum.voice.text_cleaning import clean_for_tts
            text = clean_for_tts(text, is_narrative=True)

    # Guard: truncate oversized text to prevent provider errors
    if len(text) > MAX_TTS_TEXT_CHARS:
        log.warning("tts_text_truncated", original_len=len(text), max_len=MAX_TTS_TEXT_CHARS)
        truncated = text[:MAX_TTS_TEXT_CHARS]
        last_period = truncated.rfind(". ")
        if last_period > MAX_TTS_TEXT_CHARS // 2:
            text = truncated[:last_period + 1]
        else:
            text = truncated

    headers = _build_headers(api_key, base_url=base_url)
    clean_url = normalize_base_url(base_url)

    if _is_deepgram(base_url):
        # Deepgram Aura API: POST /v1/speak?model={voice}
        # The "model" param is the voice name (e.g. "aura-asteria-en").
        speak_model = voice or model
        # Deepgram encoding param maps from OpenAI format names
        dg_encoding = {"mp3": "mp3", "wav": "linear16", "opus": "opus",
                        "aac": "aac", "flac": "flac", "pcm": "linear16"}
        params = {"model": speak_model}
        enc = dg_encoding.get(response_format)
        if enc:
            params["encoding"] = enc
        headers["Content-Type"] = "application/json"

        async def _stream_dg():
            async with _audio_client(clean_url) as client, client.stream(
                "POST",
                f"{clean_url}/v1/speak",
                json={"text": text},
                params=params,
                headers=headers,
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    yield chunk

        return _stream_dg()

    if _is_elevenlabs(base_url):
        # ElevenLabs: POST /v1/text-to-speech/{voice_id}/stream
        # Uses xi-api-key header, voice in URL path, different body format.
        voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # default: Rachel

        # Map OpenAI response_format to ElevenLabs output_format
        el_formats = {
            "mp3": "mp3_44100_128",
            "opus": "opus_48000_128",
            "pcm": "pcm_24000",
            # Note: ElevenLabs has no WAV container — "wav" falls through to mp3_44100_128
            "aac": "mp3_44100_128",  # fallback
            "flac": "mp3_44100_128",  # fallback
        }
        output_format = el_formats.get(response_format, "mp3_44100_128")

        payload: dict[str, Any] = {
            "text": text,
            "model_id": model or "eleven_flash_v2_5",
        }
        # Add speed if not default
        if speed != 1.0:
            payload["voice_settings"] = {"speed": max(0.25, min(4.0, speed))}

        headers["Content-Type"] = "application/json"

        async def _stream_el():
            async with _audio_client(clean_url) as client, client.stream(
                "POST",
                f"{clean_url}/v1/text-to-speech/{voice_id}/stream",
                json=payload,
                headers=headers,
                params={"output_format": output_format},
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    yield chunk

        return _stream_el()

    # Fish Speech: POST /v1/tts with text + optional reference_id
    # Emotion tags are embedded in the text itself: (excited)Hello!
    if provider_id == "fish-tts" or "fish" in clean_url.lower():
        # Fish Speech API uses different field names than OpenAI
        fish_fmt = {"mp3": "mp3", "wav": "wav", "opus": "opus"}.get(response_format, "wav")
        payload_fish: dict[str, Any] = {
            "text": text,
            "format": fish_fmt,
            "streaming": True,
        }
        if speed != 1.0:
            payload_fish["speed"] = speed
        # If voice looks like a reference_id (pre-uploaded voice)
        if voice:
            payload_fish["reference_id"] = voice
        headers["Content-Type"] = "application/json"

        async def _stream_fish():
            async with _audio_client(clean_url) as client, client.stream(
                "POST",
                f"{clean_url}/v1/tts",
                json=payload_fish,
                headers=headers,
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    yield chunk

        return _stream_fish()

    # OpenAI / generic OpenAI-compatible
    headers["Content-Type"] = "application/json"
    # Context-aware engines (Sesame CSM) key their rolling self-context off
    # this header. Harmless to providers that ignore it. On the fabric path
    # the header can't ride along (only the signed body is forwarded), so it
    # travels as a payload field instead — re-attached on the receiver.
    if session_id:
        headers[SESSION_HEADER] = session_id
    payload_oai: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    if instruct:
        payload_oai["instructions"] = instruct
    if stream_pcm:
        payload_oai["stream"] = True
        payload_oai["response_format"] = "pcm"

    # All OpenAI-compatible providers use /v1/audio/speech
    speech_path = "/v1/audio/speech"

    # Fabric peer dispatch: delegate to the dedicated audio_client.
    # Pre-extraction this was 50+ lines of inline multipart/signing
    # logic; now it's a 5-line delegation matching the shape of the
    # other 5 fabric modality clients (image, knowledge, render, LLM,
    # plus the STT branch below).
    is_fabric = provider_id.startswith(_FABRIC_PROVIDER_PREFIX)
    if is_fabric:
        coord = _fabric_coordinator
        identity = getattr(coord, "_identity", None) if coord is not None else None
        if identity is None or not user_id:
            # Loud failure (vs silent empty generator) — matches the
            # 2026-05-23 fix that prevented the "TTS just stopped
            # working" symptom with only a debug warning.
            log.warning(
                "fabric_tts_call_missing_credentials",
                provider_id=provider_id,
                has_coord=coord is not None,
                has_user=bool(user_id),
            )
            raise RuntimeError(
                f"fabric TTS provider {provider_id!r} requested but credentials "
                f"are missing (has_coordinator={coord is not None}, "
                f"has_user={bool(user_id)}). Confirm fabric is started + the "
                f"request carries an authenticated user."
            )

        from augmentum.fabric.audio_client import tts_stream_via_peer
        return tts_stream_via_peer(
            http_client_factory=_fabric_audio_client,
            identity=identity,
            user_id=user_id,
            peer_base_url=clean_url,
            payload=payload_oai,
            session_id=session_id,
        )

    async def _stream_oai():
        async with _audio_client(clean_url) as client, client.stream(
            "POST",
            f"{clean_url}{speech_path}",
            json=payload_oai,
            headers=headers,
        ) as upstream:
            upstream.raise_for_status()
            async for chunk in upstream.aiter_bytes(chunk_size=4096):
                yield chunk

    return _stream_oai()


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono into a minimal WAV container. The user's STT clip
    arrives as headerless PCM; ffmpeg in the CSM sidecar can't sniff a format
    without the header, so we add one before sending it as cross-speaker
    context."""
    import struct
    n = len(pcm)
    byte_rate = sample_rate * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", n) + pcm
    )


def _is_csm_provider(provider_id: str) -> bool:
    """True if this provider is the CSM sidecar — local (``sesame-csm``) or a
    fabric handle to a peer's CSM (``fabric:<node>:sesame-csm``). Cross-speaker
    context is CSM-only; every other engine ignores conversation context."""
    pid = provider_id or ""
    return pid == "sesame-csm" or (
        pid.startswith(_FABRIC_PROVIDER_PREFIX) and pid.endswith(":sesame-csm")
    )


async def push_user_context(
    *, provider: dict | None, session_id: str, pcm_audio: bytes,
    sample_rate: int, transcript: str, user_id: str,
) -> bool:
    """Cross-speaker context: hand the user's just-spoken turn (the STT clip)
    to ``provider`` (the already-resolved companion TTS provider) so her next
    reply's prosody conditions on how they actually sounded — pace, energy,
    mood — not just the words.

    No-op unless ``provider`` IS CSM (local or fabric); this is a companion-
    voice feature, never general TTS. Best-effort by contract: returns False
    and logs on any failure, never raises into the voice turn. The clip is
    sent as a WAV; it lives only in the sidecar's RAM."""
    if not session_id or not pcm_audio or not provider:
        return False
    if not _is_csm_provider(provider.get("id", "")):
        return False

    wav = _pcm16_to_wav(pcm_audio, sample_rate)
    pid = provider.get("id", "")
    base_url = normalize_base_url(provider.get("base_url", ""))
    try:
        if pid.startswith(_FABRIC_PROVIDER_PREFIX):
            coord = _fabric_coordinator
            identity = getattr(coord, "_identity", None) if coord is not None else None
            if identity is None or not user_id:
                return False
            from augmentum.fabric.audio_client import push_user_context_via_peer
            await push_user_context_via_peer(
                http_client_factory=_fabric_audio_client,
                identity=identity, user_id=user_id,
                peer_base_url=base_url,
                audio_bytes=wav, filename="user_turn.wav",
                content_type="audio/wav",
                session_id=session_id, transcript=transcript,
            )
        else:
            async with _audio_client(base_url) as client:
                await client.post(
                    f"{base_url}/v1/context/user_turn",
                    files={"audio": ("user_turn.wav", wav, "audio/wav")},
                    data={"transcript": transcript},
                    headers={SESSION_HEADER: session_id},
                    timeout=15.0,
                )
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort; never break the turn
        log.warning("user_context_push_failed", provider=pid, error=str(exc)[:160])
        return False


async def csm_warm(*, provider: dict | None, user_id: str) -> bool:
    """Pre-load a CSM provider's model (conversation-scoped residency) so the
    first utterance isn't cold. No-op for non-CSM providers. Best-effort."""
    if not provider or not _is_csm_provider(provider.get("id", "")):
        return False
    pid = provider.get("id", "")
    base_url = normalize_base_url(provider.get("base_url", ""))
    try:
        if pid.startswith(_FABRIC_PROVIDER_PREFIX):
            coord = _fabric_coordinator
            identity = getattr(coord, "_identity", None) if coord is not None else None
            if identity is None or not user_id:
                return False
            from augmentum.fabric.audio_client import warmup_via_peer
            await warmup_via_peer(
                http_client_factory=_fabric_audio_client,
                identity=identity, user_id=user_id, peer_base_url=base_url,
            )
        else:
            async with _audio_client(base_url) as client:
                await client.post(f"{base_url}/warmup", timeout=10.0)
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.debug("csm_warm_failed", provider=pid, error=str(exc)[:160])
        return False


async def csm_unload(*, provider: dict | None, session_id: str, user_id: str) -> bool:
    """Release a CSM provider's VRAM + clear this session's cross-speaker
    context (the conversation ended). No-op for non-CSM. Best-effort."""
    if not provider or not _is_csm_provider(provider.get("id", "")):
        return False
    pid = provider.get("id", "")
    base_url = normalize_base_url(provider.get("base_url", ""))
    try:
        if pid.startswith(_FABRIC_PROVIDER_PREFIX):
            coord = _fabric_coordinator
            identity = getattr(coord, "_identity", None) if coord is not None else None
            if identity is None or not user_id:
                return False
            from augmentum.fabric.audio_client import unload_via_peer
            await unload_via_peer(
                http_client_factory=_fabric_audio_client,
                identity=identity, user_id=user_id, peer_base_url=base_url,
                session_id=session_id,
            )
        else:
            async with _audio_client(base_url) as client:
                await client.post(
                    f"{base_url}/unload",
                    params={"session": session_id} if session_id else {},
                    timeout=10.0,
                )
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.debug("csm_unload_failed", provider=pid, error=str(exc)[:160])
        return False


async def _safe_upstream_detail(resp: httpx.Response | None) -> str:
    """Best-effort body text from a streamed error response.

    The TTS dispatch uses ``client.stream(...)``; when the upstream
    returns an error status, ``raise_for_status()`` fires before the body
    is read AND ``prime_stream`` has already closed the stream context.
    Touching ``resp.text`` then raises ``ResponseNotRead`` — which masked
    a clean provider 503 (e.g. CSM still compiling) behind an opaque 500.
    Read defensively; return "" if the body is unavailable so the caller
    falls back to the status code."""
    if resp is None:
        return ""
    try:
        await resp.aread()
    except Exception:  # noqa: BLE001 — stream already closed/consumed
        pass
    try:
        return resp.text
    except Exception:  # noqa: BLE001 — body never buffered
        return ""


@router.post("/v1/audio/speech")
async def tts_speech(body: TTSRequest, request: Request):
    """Proxy text-to-speech to the configured TTS provider.

    Compatible with OpenAI's POST /v1/audio/speech endpoint.
    Handles Deepgram's native API format transparently.
    Uses built-in Kokoro when available (no external provider needed).
    """
    if not settings.audio_tts_enabled:
        raise HTTPException(503, "TTS is not enabled")

    # Bound input length so an exposed box can't be pinned by a giant
    # synthesis request. Default-on; 0 disables.
    _max_tts = int(getattr(settings, "api_tts_max_chars", 50_000))
    if _max_tts > 0 and len(body.input or "") > _max_tts:
        raise HTTPException(413, f"Input too long: {len(body.input)} chars (limit {_max_tts})")

    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    # Hot dispatch path: read provider/voice metadata off the dedicated
    # read connection so the SELECT doesn't queue behind writers on the
    # main aiosqlite worker thread. Falls back to the main conn if
    # read_conn isn't configured.
    read_conn = _get_read_conn(request)

    # Auto-resolve voice → provider (handles "provider::voice" prefix AND plain names)
    # Provider resolution must happen first so we can apply provider-aware cleaning.
    raw_voice = body.voice or ""
    provider, raw_voice = await resolve_voice_provider(read_conn, raw_voice)

    if not provider:
        raise HTTPException(503, "No TTS provider configured")

    # Per-voice pronunciation lexicon (migration 261) — applied BEFORE
    # cleaning so user entries beat every built-in normalization rule.
    # Fail-safe inside apply(); a lexicon problem never blocks speech.
    from augmentum.voice import lexicon_store
    body.input = await lexicon_store.apply(
        conn, body.input, user_id=_user_id(request), voice=raw_voice,
    )

    # Clean text for TTS — provider-aware so Chatterbox Turbo tags survive.
    # _build_tts_stream also cleans when pre_cleaned=False, but the HTTP endpoint
    # passes text directly (not through the voice pipeline), so we clean here and
    # mark it pre_cleaned to avoid double-stripping.
    from augmentum.voice.text_cleaning import clean_for_tts
    _is_turbo = provider.get("id") == "chatterbox-turbo"
    if _is_turbo:
        from augmentum.voice.emotion import inject_turbo_tags
        body.input = clean_for_tts(inject_turbo_tags(body.input), preserve_brackets=True) or body.input
    else:
        body.input = clean_for_tts(body.input) or body.input

    # Built-in Kokoro: in-process streaming (only when it's the resolved provider)
    if provider.get("id") == "kokoro-builtin":
        from augmentum.voice.kokoro_tts import KokoroTTS
        kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
        if not kokoro.is_available:
            await load_model_off_loop(kokoro.load_model)
        if not kokoro.is_available:
            raise HTTPException(503, "Built-in Kokoro TTS is not available — model failed to load")
        voice_name = raw_voice or provider.get("default_voice", "af_heart")
        media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")
        # Same opt-in signal as the Pocket TTS branch: ?stream=1 picks
        # the live-stream WAV path (one sentinel header + PCM as
        # produced), absence picks the buffered WAV path (one well-
        # formed file at the end). See KokoroTTS.stream_speech for the
        # rationale on each path; non-WAV formats ignore this flag.
        stream_chunks = request.query_params.get("stream") == "1"
        try:
            return StreamingResponse(
                content=kokoro.stream_speech(
                    body.input,
                    voice=voice_name,
                    speed=body.speed,
                    response_format=body.response_format,
                    stream_chunks=stream_chunks,
                ),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'inline; filename="speech.{body.response_format}"',
                },
            )
        except Exception as exc:
            log.warning("kokoro_builtin_http_tts_error", error=str(exc))
            raise HTTPException(503, f"Built-in Kokoro TTS failed: {exc}")

    # Built-in Pocket TTS: in-process streaming (Kyutai multilingual CPU engine)
    if provider.get("id") == "pockettts-builtin":
        eng = await _builtin_tts_engine("pockettts-builtin")
        if eng is None:
            raise HTTPException(503, "Built-in Pocket TTS is not available — model failed to load")
        voice_name = raw_voice or provider.get("default_voice", "alba")
        media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")
        # Client signals whether it can consume a true byte stream
        # (Android Chrome / desktop via ReadableStream.getReader) versus
        # forcing the response into a Blob (iOS Safari Audio element).
        # The WAV emission path differs: live stream gets one sentinel
        # header up front + PCM as the model produces it; buffered gets
        # one well-formed WAV with real sizes at the end. See
        # PocketTTS.stream_speech for the rationale on each path.
        stream_chunks = request.query_params.get("stream") == "1"

        async def _logged_stream():
            """Wrap the engine generator so mid-stream errors leave a
            trail in container logs. The bare ``StreamingResponse`` only
            surfaces construction-time exceptions — anything that raises
            inside the async-for would otherwise close the response with
            no diagnostic. ``yielded`` tells us whether any bytes
            actually shipped, which matters because an empty 200-OK
            body presents on the client as a silent no-op rather than a
            failure toast."""
            yielded = 0
            try:
                async for chunk in eng.stream_speech(
                    body.input, voice=voice_name, speed=body.speed,
                    response_format=body.response_format,
                    stream_chunks=stream_chunks,
                ):
                    yielded += len(chunk)
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "pocket_builtin_stream_raised",
                    error=str(exc),
                    yielded_bytes=yielded,
                    text_len=len(body.input or ""),
                    exc_info=True,
                )
                # Headers/status already sent at this point — re-raising
                # would 500 mid-stream which Starlette logs as a generic
                # "exception in response" with no useful detail. Just
                # close the stream cleanly; the client's empty-body
                # branch surfaces the failure.
                return
            if yielded == 0:
                log.warning(
                    "pocket_builtin_stream_empty",
                    text_len=len(body.input or ""),
                    voice=voice_name,
                )

        try:
            return StreamingResponse(
                content=_logged_stream(),
                media_type=media_type,
                headers={"Content-Disposition": f'inline; filename="speech.{body.response_format}"'},
            )
        except Exception as exc:
            log.warning("pocket_builtin_http_tts_error", error=str(exc))
            raise HTTPException(503, f"Built-in Pocket TTS failed: {exc}")

    base_url = normalize_base_url(provider["base_url"])
    model = body.model or provider["default_model"]
    voice = raw_voice or provider["default_voice"]

    if not model and not voice:
        raise HTTPException(400, "No TTS model specified and no default configured")

    media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")

    # Prime the upstream stream BEFORE handing it to StreamingResponse —
    # the actual httpx.stream("POST", ...) call lives inside the
    # generator returned by ``_build_tts_stream``, so a ConnectError
    # (provider container down) would otherwise raise AFTER Starlette
    # has already committed ``http.response.start`` and produce a
    # "Caught handled exception, but response already started." cascade
    # to a generic 500. Forcing the first chunk to flow here surfaces
    # connect failures as clean 502s.
    from augmentum.proxy.streaming import StreamPrimeError, prime_stream

    raw_stream = _build_tts_stream(
        base_url, model, voice, body.input,
        body.response_format, body.speed, provider["api_key"],
        pre_cleaned=True,
        instruct=body.instruct,
        provider_id=provider.get("id", ""),
        user_id=_user_id(request),
        session_id=request.headers.get(SESSION_HEADER, "") or body.session_id,
    )
    try:
        primed = await prime_stream(raw_stream)
    except StreamPrimeError as exc:
        cause = exc.cause
        if isinstance(cause, httpx.HTTPStatusError):
            status = cause.response.status_code if cause.response else 502
            body = sanitize_error_detail((await _safe_upstream_detail(cause.response))[:500])
            detail = body or f"provider returned {status}"
            log.warning("tts_upstream_error", status=status, detail=detail)
            raise HTTPException(status, f"TTS provider error: {detail}")
        log.warning("tts_connection_error", error=repr(cause))
        raise HTTPException(502, f"Could not reach TTS provider: {cause!r}")
    return StreamingResponse(
        content=primed,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="speech.{body.response_format}"',
        },
    )


async def tts_synthesize_bytes(
    conn, text: str, *, voice: str = "", speed: float = 1.0,
    response_format: str = "mp3", user_id: str = "",
) -> tuple[bytes, bool]:
    """Synthesize ``text`` to audio bytes using the configured TTS provider.

    Non-streaming counterpart of :func:`tts_speech` — collects the full
    audio into memory. Works with built-in engines (Kokoro / Pocket TTS) and
    external HTTP providers alike. Returns ``(audio_bytes, is_builtin)`` —
    ``is_builtin`` flags engines whose ``response_format="wav"`` output is a
    well-formed per-segment WAV (relevant when callers stitch long text).

    Raises ``HTTPException`` on provider/config errors, mirroring the route.
    """
    from augmentum.voice.text_cleaning import clean_for_tts

    if not getattr(settings, "audio_tts_enabled", True):
        raise HTTPException(503, "TTS is not enabled")
    provider, raw_voice = await resolve_voice_provider(conn, voice or "")
    if not provider:
        raise HTTPException(503, "No TTS provider configured")
    cleaned = clean_for_tts(text) or text
    builtin_id = provider.get("id", "") if provider.get("id") in _BUILTIN_TTS_IDS else ""

    if builtin_id:
        eng = await _builtin_tts_engine(builtin_id)
        if eng is None:
            raise HTTPException(503, f"Built-in TTS '{builtin_id}' is not available")
        voice_name = raw_voice or provider.get("default_voice", "")
        buf = bytearray()
        async for chunk in eng.stream_speech(
            cleaned, voice=voice_name, speed=speed, response_format=response_format,
        ):
            if chunk:
                buf += chunk
        if not buf:
            raise HTTPException(502, "TTS produced no audio")
        # True = a built-in engine that yields well-formed per-segment WAV
        # blobs (relevant when callers stitch long output).
        return bytes(buf), True

    base_url = normalize_base_url(provider["base_url"])
    model = provider["default_model"]
    voice_name = raw_voice or provider["default_voice"]
    if not model and not voice_name:
        raise HTTPException(400, "No TTS model specified and no default configured")
    try:
        buf = bytearray()
        async for chunk in _build_tts_stream(
            base_url, model, voice_name, cleaned, response_format, speed,
            provider["api_key"], pre_cleaned=True, provider_id=provider.get("id", ""),
            user_id=user_id,
        ):
            if chunk:
                buf += chunk
    except httpx.HTTPStatusError as exc:
        detail = sanitize_error_detail(exc.response.text[:500]) if exc.response else str(exc)
        raise HTTPException(exc.response.status_code, f"TTS provider error: {detail}")
    except httpx.RequestError:
        raise HTTPException(502, "Could not reach TTS provider")
    if not buf:
        raise HTTPException(502, "TTS produced no audio")
    return bytes(buf), False


# ---------------------------------------------------------------------------
# STT — POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------


def _extract_wav_pcm(wav_bytes: bytes) -> bytes:
    """Extract raw PCM data from a WAV file by parsing chunk headers.

    ffmpeg may insert LIST/INFO metadata between fmt and data chunks,
    making the data offset > 44 bytes.  A naive ``wav[44:]`` would
    include garbage bytes that corrupt STT input.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes  # Not a valid WAV — return as-is

    pos = 12  # Skip RIFF header + WAVE tag
    while pos < len(wav_bytes) - 8:
        chunk_id = wav_bytes[pos:pos + 4]
        chunk_size = int.from_bytes(wav_bytes[pos + 4:pos + 8], "little")
        if chunk_id == b"data":
            return wav_bytes[pos + 8:pos + 8 + chunk_size]
        pos += 8 + chunk_size
        # WAV chunks are word-aligned (pad to even)
        if chunk_size % 2:
            pos += 1

    # Fallback: skip standard 44-byte header
    return wav_bytes[44:]


async def _moonshine_batch_transcribe(audio_bytes: bytes, filename: str) -> str:
    """Transcribe audio using built-in Moonshine STT (in-process, no HTTP).

    Converts the audio to 16kHz mono WAV via ffmpeg, then feeds it through
    Moonshine's streaming transcriber in batch mode. This exposes the local
    STT engine via the standard OpenAI-compatible REST endpoint so external
    clients (phones, other apps) can use it without WebSocket.
    """
    import asyncio

    from augmentum.voice.moonshine_stt import MoonshineSTTSession

    # Ensure model is loaded
    MoonshineSTTSession.warmup()
    if not MoonshineSTTSession.is_available():
        raise HTTPException(503, "Moonshine STT model not available")

    # Determine audio format from both filename hint and byte signatures.
    # Client-side VAD sends WebM/Opus chunks from MediaRecorder — these may
    # not start with the EBML magic if the first chunk was discarded, so we
    # also trust the filename hint from the caller.
    is_wav = len(audio_bytes) > 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
    is_container = (
        filename.endswith((".webm", ".ogg", ".opus", ".mp3", ".m4a"))
        or (len(audio_bytes) > 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3")  # EBML
        or (len(audio_bytes) > 4 and audio_bytes[:4] == b"OggS")
    )

    if is_wav:
        pcm_bytes = _extract_wav_pcm(audio_bytes)
        log.info("moonshine_batch_format", fmt="wav", input_bytes=len(audio_bytes),
                 pcm_bytes=len(pcm_bytes))
    elif is_container:
        # Container format (WebM/Opus/Ogg) — transcode via ffmpeg
        from augmentum.voice.stt import _transcode_to_wav
        log.info("moonshine_batch_format", fmt="container", input_bytes=len(audio_bytes),
                 header=audio_bytes[:4].hex(), filename=filename)
        wav_bytes, _, _ = await asyncio.to_thread(_transcode_to_wav, audio_bytes)
        pcm_bytes = _extract_wav_pcm(wav_bytes)
        log.info("moonshine_batch_transcode", wav_bytes=len(wav_bytes),
                 pcm_bytes=len(pcm_bytes))
    else:
        # Raw PCM16 16kHz mono (from server VAD / AudioWorklet PCM path)
        pcm_bytes = audio_bytes
        log.info("moonshine_batch_format", fmt="raw_pcm", input_bytes=len(audio_bytes),
                 header=audio_bytes[:4].hex() if len(audio_bytes) >= 4 else "short")

    # Ensure PCM bytes are aligned to 2-byte int16 samples
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]

    # Batch transcription using Moonshine's non-streaming API.
    # transcribe_without_streaming() processes the entire audio in one shot
    # and returns a Transcript with lines and word-level confidence scores.
    # The streaming start/add_audio/stop cycle doesn't work for batch
    # (internal buffering expects continuous real-time frames).
    try:
        import numpy as np

        MoonshineSTTSession.warmup()
        if not MoonshineSTTSession.is_available():
            raise HTTPException(503, "Moonshine STT model not available")

        # Convert PCM16 → float32 [-1, 1]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        log.info("moonshine_batch_transcribe",
                 pcm_bytes=len(pcm_bytes),
                 duration_s=f"{len(pcm_bytes) / (16000 * 2):.1f}",
                 samples=len(samples))

        def _do_transcribe():
            """Run non-streaming transcription in a thread.

            Uses a transcriber DEDICATED to batch work — never the streaming
            _shared_transcriber, which the live STT path can wedge (an
            abandoned add_audio worker keeps mutating it past its 5s timeout),
            making transcribe_without_streaming return empty instantly. The
            batch fallback exists to rescue an empty stream, so it must run on
            a clean instance. The native ONNX session isn't reentrant, so we
            serialize concurrent batch calls on the shared batch lock.
            """
            transcriber = MoonshineSTTSession.get_batch_transcriber()
            if transcriber is None:
                return []

            with MoonshineSTTSession._batch_lock:
                transcript = transcriber.transcribe_without_streaming(
                    samples.tolist(), 16000,
                )
            results = []
            for line in transcript.lines:
                text = (line.text or "").strip()
                if text:
                    results.append(text)
            return results

        results = await asyncio.to_thread(_do_transcribe)

        transcript = " ".join(results).strip()
        log.info("moonshine_batch_result",
                 parts=len(results),
                 transcript=transcript[:100] if transcript else "(empty)")
        return transcript

    except HTTPException:
        raise
    except Exception as exc:
        log.warning("moonshine_batch_stt_error", error=str(exc))
        raise HTTPException(500, f"Moonshine STT failed: {exc}")


@router.post("/v1/audio/transcriptions")
async def transcribe_audio_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    response_format: str = Form("json"),
) -> dict[str, str]:
    """OpenAI-compatible STT endpoint.

    Accepts a multipart ``file`` upload (the OpenAI / whisper client
    contract) and returns ``{"text": "<transcript>"}``. Thin wrapper over
    :func:`transcribe_audio_bytes`, which is also called directly as a plain
    helper from non-HTTP surfaces (artifact transcribe, voice turn).

    Previously this route was bound straight to ``transcribe_audio_bytes``,
    whose ``audio_bytes: bytes`` / ``filename: str`` signature never matched
    the multipart ``file`` field — so every upload transcribed empty and the
    return was a bare JSON string, not ``{"text": ...}``. Whisper Race and
    Companion voice read ``j.text``; both now resolve.
    """
    data = await file.read()
    text = await transcribe_audio_bytes(
        request,
        data,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "audio/webm",
        language=language,
        response_format=response_format,
    )
    return {"text": text}


async def transcribe_audio_bytes(
    request: Request,
    audio_bytes: bytes,
    filename: str,
    content_type: str = "audio/webm",
    *,
    language: str = "",
    response_format: str = "json",
) -> str:
    """Run audio bytes through the configured STT provider and return the
    plain transcript text.

    Callers that need the full OpenAI-compat response (with timing, logprobs,
    etc.) should use the `/api/audio/transcriptions` endpoint directly — this
    helper returns only the transcript string so non-STT surfaces (e.g., an
    artifact transcribe button) don't have to unwrap the upstream envelope.
    """
    if not settings.audio_stt_enabled:
        raise HTTPException(503, "STT is not enabled")

    # Bound audio size at the app level (the transport cap is the upload
    # tier; this is the tighter STT-specific limit). Default-on; 0 disables.
    _max_stt = int(getattr(settings, "api_stt_max_bytes", 26_214_400))
    if _max_stt > 0 and len(audio_bytes) > _max_stt:
        raise HTTPException(413, f"Audio too large: {len(audio_bytes)} bytes (limit {_max_stt})")

    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    provider = await _get_default_provider(conn, "stt")
    if not provider:
        raise HTTPException(503, "No STT provider configured")

    raw_url = provider.get("base_url", "")
    if raw_url in ("builtin", "built-in") or provider.get("id") == "moonshine-stt":
        return await _moonshine_batch_transcribe(audio_bytes, filename)

    base_url = normalize_base_url(provider["base_url"])
    model = provider["default_model"]
    headers = _build_headers(provider["api_key"], base_url=base_url)

    # Fabric peer dispatch: delegate to the dedicated audio_client.
    # Same extraction as the TTS branch above — moves 80 lines of
    # inline multipart/signing logic into a 6-line delegation.
    is_fabric = provider.get("id", "").startswith(_FABRIC_PROVIDER_PREFIX)
    if is_fabric:
        coord = _fabric_coordinator
        identity = getattr(coord, "_identity", None) if coord is not None else None
        user_id = _user_id(request)
        if identity is None or not user_id:
            log.warning(
                "fabric_stt_call_missing_credentials",
                provider_id=provider.get("id"),
                has_coord=coord is not None, has_user=bool(user_id),
            )
            raise HTTPException(503, "Fabric STT dispatch unavailable")

        from augmentum.fabric.audio_client import (
            RemoteAudioError,
            stt_transcribe_via_peer,
        )
        try:
            return await stt_transcribe_via_peer(
                http_client_factory=_fabric_audio_client,
                identity=identity, user_id=user_id,
                peer_base_url=base_url,
                audio_bytes=audio_bytes, filename=filename,
                content_type=content_type,
                model=model, language=language,
                response_format=response_format,
            )
        except RemoteAudioError as exc:
            # Map typed transport / status errors to a 502 — the
            # peer is unreachable or misbehaving, distinct from a
            # local mis-config.
            raise HTTPException(502, str(exc)) from exc

    try:
        async with _audio_client(base_url) as client:
            if _is_deepgram(base_url):
                dg_params: dict[str, str] = {}
                if model:
                    dg_params["model"] = model
                if language:
                    dg_params["language"] = language
                dg_headers = {**headers, "Content-Type": content_type}
                upstream = await client.post(
                    f"{base_url}/v1/listen",
                    content=audio_bytes,
                    params=dg_params,
                    headers=dg_headers,
                )
                upstream.raise_for_status()
                dg_data = upstream.json()
                results = dg_data.get("results", {})
                channels = results.get("channels", [])
                if channels:
                    alts = channels[0].get("alternatives", [])
                    if alts:
                        return alts[0].get("transcript", "") or ""
                return ""

            files_data = {"file": (filename, audio_bytes, content_type)}
            form_data: dict[str, str] = {}
            if model:
                form_data["model"] = model
            if language:
                form_data["language"] = language
            if response_format:
                form_data["response_format"] = response_format

            upstream = await client.post(
                f"{base_url}/v1/audio/transcriptions",
                files=files_data,
                data=form_data,
                headers=headers,
            )
            upstream.raise_for_status()
            body = upstream.json()
            if isinstance(body, dict):
                return body.get("text", "") or ""
            return str(body)
    except httpx.HTTPStatusError as exc:
        detail = sanitize_error_detail(exc.response.text[:500]) if exc.response else str(exc)
        log.warning("stt_upstream_error", status=exc.response.status_code, detail=detail)
        raise HTTPException(exc.response.status_code, f"STT provider error: {detail}") from exc
    except httpx.RequestError as exc:
        log.warning("stt_connection_error", error=str(exc))
        raise HTTPException(502, "Could not reach STT provider") from exc


async def stt_transcribe(request: Request, file: UploadFile):
    """Proxy speech-to-text to the configured STT provider.

    Compatible with OpenAI's POST /v1/audio/transcriptions endpoint.
    Accepts multipart form with an audio file.
    Uses built-in Moonshine when available (no external provider needed).
    """
    if not settings.audio_stt_enabled:
        raise HTTPException(503, "STT is not enabled")

    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    provider = await _get_default_provider(conn, "stt")
    if not provider:
        raise HTTPException(503, "No STT provider configured")

    audio_bytes = await file.read()
    # Bound audio size (same cap as the /v1 bytes endpoint). 0 disables.
    _max_stt = int(getattr(settings, "api_stt_max_bytes", 26_214_400))
    if _max_stt > 0 and len(audio_bytes) > _max_stt:
        raise HTTPException(413, f"Audio too large: {len(audio_bytes)} bytes (limit {_max_stt})")

    # Built-in Moonshine: in-process batch transcription
    raw_url = provider.get("base_url", "")
    if raw_url in ("builtin", "built-in") or provider.get("id") == "moonshine-stt":
        text = await _moonshine_batch_transcribe(audio_bytes, file.filename or "audio.wav")
        return JSONResponse(content={"text": text})

    base_url = normalize_base_url(provider["base_url"])
    model = provider["default_model"]

    # Read form fields from query params or use provider defaults
    params = request.query_params
    language = params.get("language", "")
    response_format = params.get("response_format", "json")

    headers = _build_headers(provider["api_key"], base_url=base_url)
    content_type = file.content_type or "audio/webm"

    try:
        async with _audio_client(base_url) as client:
            if _is_deepgram(base_url):
                # Deepgram STT: POST /v1/listen?model={model} with raw audio body
                dg_params: dict[str, str] = {}
                if model:
                    dg_params["model"] = model
                if language:
                    dg_params["language"] = language
                dg_headers = {**headers, "Content-Type": content_type}

                upstream = await client.post(
                    f"{base_url}/v1/listen",
                    content=audio_bytes,
                    params=dg_params,
                    headers=dg_headers,
                )
                upstream.raise_for_status()
                dg_data = upstream.json()
                # Normalize Deepgram response to OpenAI format
                text = ""
                results = dg_data.get("results", {})
                channels = results.get("channels", [])
                if channels:
                    alts = channels[0].get("alternatives", [])
                    if alts:
                        text = alts[0].get("transcript", "")
                return JSONResponse(content={"text": text})

            # OpenAI-compatible: POST /v1/audio/transcriptions with multipart
            files_data = {
                "file": (file.filename or "audio.webm", audio_bytes, content_type),
            }
            form_data: dict[str, str] = {}
            if model:
                form_data["model"] = model
            if language:
                form_data["language"] = language
            if response_format:
                form_data["response_format"] = response_format

            upstream = await client.post(
                f"{base_url}/v1/audio/transcriptions",
                files=files_data,
                data=form_data,
                headers=headers,
            )
            upstream.raise_for_status()

            return JSONResponse(content=upstream.json())
    except httpx.HTTPStatusError as exc:
        detail = sanitize_error_detail(exc.response.text[:500]) if exc.response else str(exc)
        log.warning("stt_upstream_error", status=exc.response.status_code, detail=detail)
        raise HTTPException(exc.response.status_code, f"STT provider error: {detail}")
    except httpx.RequestError as exc:
        log.warning("stt_connection_error", error=str(exc))
        raise HTTPException(502, "Could not reach STT provider")


# ---------------------------------------------------------------------------
# Provider CRUD — /api/audio/providers
# ---------------------------------------------------------------------------


@router.get("/api/audio/providers")
async def list_audio_providers(request: Request):
    """List all configured audio providers."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content=[])

    cursor = await conn.execute(
        "SELECT id, provider_type, name, base_url, default_model, default_voice, "
        "is_enabled, is_default, tts_chunking FROM audio_providers ORDER BY provider_type, name"
    )
    rows = await cursor.fetchall()
    providers = [
        {
            "id": r[0],
            "provider_type": r[1],
            "name": r[2],
            "base_url": r[3],
            "default_model": r[4],
            "default_voice": r[5],
            "is_enabled": bool(r[6]),
            "is_default": bool(r[7]),
            "tts_chunking": r[8] or "sentence",
        }
        for r in rows
    ]
    # Merge connected fabric peers' TTS/STT providers so a peer-hosted
    # engine (e.g. Speaches installed on another node) is SELECTABLE here,
    # not just usable as a silent default fallback. Local-first: a peer
    # entry whose synthetic id already exists locally is skipped. Mirrors
    # the image /models peer-merge.
    try:
        local_ids = {p["id"] for p in providers}
        for entry in _fabric_audio_provider_entries():
            if entry["id"] not in local_ids:
                providers.append(entry)
    except Exception:
        log.debug("fabric_audio_provider_merge_failed", exc_info=True)
    return JSONResponse(content=providers)


@router.post("/api/audio/providers")
async def create_audio_provider(body: AudioProviderCreate, request: Request):
    """Add a new audio provider. Admin only — providers are shared infra."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    # Reject non-http/https schemes early. LAN-internal URLs (Ollama on
    # the home network, dockerised TTS sidecar at audio:8000) stay
    # allowed — only file://, gopher://, dict://, etc. are blocked.
    from augmentum.utils.safe_http import SafeHttpError, validate_provider_url
    try:
        body.base_url = validate_provider_url(body.base_url)
    except SafeHttpError as exc:
        raise HTTPException(400, str(exc))

    existing = await _get_provider_by_id(conn, body.id)
    if existing:
        raise HTTPException(409, f"Provider '{body.id}' already exists")

    # Check if this should be default (first of its type)
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM audio_providers WHERE provider_type = ?",
        (body.provider_type,),
    )
    count = (await cursor.fetchone())[0]
    is_default = 1 if count == 0 else 0

    await conn.execute(
        "INSERT INTO audio_providers (id, provider_type, name, base_url, api_key, "
        "default_model, default_voice, is_default, tts_chunking) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (body.id, body.provider_type, body.name, body.base_url, encrypt_api_key(body.api_key),
         body.default_model, body.default_voice, is_default, body.tts_chunking),
    )
    await conn.commit()

    invalidate_voice_caches()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "tts")
    _invalidate_resource(request.app.state, "stt")
    log.info("audio_provider_created", id=body.id, type=body.provider_type)
    return JSONResponse(content={"status": "created", "id": body.id, "is_default": bool(is_default)})


@router.put("/api/audio/providers/{provider_id}")
async def update_audio_provider(provider_id: str, body: AudioProviderUpdate, request: Request):
    """Update an existing audio provider. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    if body.base_url is not None:
        from augmentum.utils.safe_http import SafeHttpError, validate_provider_url
        try:
            body.base_url = validate_provider_url(body.base_url)
        except SafeHttpError as exc:
            raise HTTPException(400, str(exc))

    existing = await _get_provider_by_id(conn, provider_id)
    if not existing:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    updates = []
    params = []
    for field_name, col in [
        ("name", "name"), ("base_url", "base_url"), ("api_key", "api_key"),
        ("default_model", "default_model"), ("default_voice", "default_voice"),
        ("is_enabled", "is_enabled"), ("tts_chunking", "tts_chunking"),
    ]:
        val = getattr(body, field_name, None)
        if val is not None:
            updates.append(f"{col} = ?")
            if field_name == "api_key":
                val = encrypt_api_key(val)
            params.append(val if not isinstance(val, bool) else int(val))

    # Handle is_default: unset others of same type first
    if body.is_default is True:
        await conn.execute(
            "UPDATE audio_providers SET is_default = 0 WHERE provider_type = ?",
            (existing["provider_type"],),
        )
        updates.append("is_default = 1")

    if not updates:
        return JSONResponse(content={"status": "no_changes"})

    updates.append("updated_at = datetime('now')")
    params.append(provider_id)

    await conn.execute(
        f"UPDATE audio_providers SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    await conn.commit()
    invalidate_voice_caches()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "tts")
    _invalidate_resource(request.app.state, "stt")
    return JSONResponse(content={"status": "updated"})


@router.delete("/api/audio/providers/{provider_id}")
async def delete_audio_provider(provider_id: str, request: Request):
    """Delete an audio provider. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    existing = await _get_provider_by_id(conn, provider_id)
    if not existing:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    await conn.execute("DELETE FROM audio_providers WHERE id = ?", (provider_id,))
    await conn.commit()

    # If deleted provider was default, promote next one
    if existing["is_default"]:
        try:
            await conn.execute(
                "UPDATE audio_providers SET is_default = 1 "
                "WHERE provider_type = ? AND is_enabled = 1 "
                "ORDER BY created_at LIMIT 1",
                (existing["provider_type"],),
            )
            await conn.commit()
        except Exception:
            log.warning("audio_default_promotion_failed", provider_type=existing["provider_type"])

    invalidate_voice_caches()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "tts")
    _invalidate_resource(request.app.state, "stt")
    log.info("audio_provider_deleted", id=provider_id)
    return JSONResponse(content={"status": "deleted"})


@router.post("/api/audio/providers/{provider_id}/test")
async def test_audio_provider(provider_id: str, request: Request):
    """Test connectivity to an audio provider by fetching its model list."""
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    provider = await _get_provider_by_id(conn, provider_id)
    if not provider:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider["api_key"], base_url=base_url)

    try:
        async with _audio_client(base_url) as client:
            if _is_deepgram(base_url):
                # Deepgram doesn't have /v1/models — use /v1/projects to verify auth
                resp = await client.get(
                    f"{base_url}/v1/projects", headers=headers, timeout=10.0,
                )
                resp.raise_for_status()
                return JSONResponse(content={
                    "status": "ok",
                    "models": [provider.get("default_model", "aura-2-en")],
                })

            resp = await client.get(f"{base_url}/v1/models", headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            # ElevenLabs returns a flat list with model_id; OpenAI uses data[].id
            if isinstance(data, list):
                model_ids = [m.get("model_id", m.get("id", "?")) for m in data][:20]
            else:
                models = data.get("data", data.get("models", []))
                model_ids = [m.get("id", m.get("name", "?")) for m in models][:20]
            return JSONResponse(content={
                "status": "ok",
                "models": model_ids,
            })
    except Exception as exc:
        return JSONResponse(content={
            "status": "error",
            "error": sanitize_error_detail(str(exc)[:300]),
        }, status_code=200)


def _is_deepgram(base_url: str) -> bool:
    """Check if a base URL points to Deepgram's API."""
    return "deepgram.com" in base_url.lower()


def _is_elevenlabs(base_url: str) -> bool:
    """Check if a base URL points to ElevenLabs' API."""
    return "elevenlabs.io" in base_url.lower()


def _is_openai_tts(base_url: str) -> bool:
    """Check if a base URL points to OpenAI's API."""
    return "api.openai.com" in base_url.lower()


# Hardcoded voice lists for providers that don't expose a /voices endpoint.
_KNOWN_VOICES: dict[str, list[dict]] = {
    "openai": [
        {"id": "alloy", "name": "Alloy"},
        {"id": "ash", "name": "Ash"},
        {"id": "ballad", "name": "Ballad"},
        {"id": "cedar", "name": "Cedar"},
        {"id": "coral", "name": "Coral"},
        {"id": "echo", "name": "Echo"},
        {"id": "fable", "name": "Fable"},
        {"id": "marin", "name": "Marin"},
        {"id": "nova", "name": "Nova"},
        {"id": "onyx", "name": "Onyx"},
        {"id": "sage", "name": "Sage"},
        {"id": "shimmer", "name": "Shimmer"},
        {"id": "verse", "name": "Verse"},
    ],
    "deepgram": [
        # Aura 2 — current generation
        {"id": "aura-2-thalia-en", "name": "Thalia (Aura 2, Female)"},
        {"id": "aura-2-andromeda-en", "name": "Andromeda (Aura 2, Female)"},
        {"id": "aura-2-helena-en", "name": "Helena (Aura 2, Female)"},
        {"id": "aura-2-apollo-en", "name": "Apollo (Aura 2, Male)"},
        {"id": "aura-2-atlas-en", "name": "Atlas (Aura 2, Male)"},
        {"id": "aura-2-orion-en", "name": "Orion (Aura 2, Male)"},
        # Aura 1 — legacy, still functional
        {"id": "aura-asteria-en", "name": "Asteria (Female)"},
        {"id": "aura-luna-en", "name": "Luna (Female)"},
        {"id": "aura-stella-en", "name": "Stella (Female)"},
        {"id": "aura-athena-en", "name": "Athena (Female)"},
        {"id": "aura-hera-en", "name": "Hera (Female)"},
        {"id": "aura-orion-en", "name": "Orion (Male)"},
        {"id": "aura-arcas-en", "name": "Arcas (Male)"},
        {"id": "aura-perseus-en", "name": "Perseus (Male)"},
        {"id": "aura-angus-en", "name": "Angus (Male)"},
        {"id": "aura-orpheus-en", "name": "Orpheus (Male)"},
        {"id": "aura-helios-en", "name": "Helios (Male)"},
        {"id": "aura-zeus-en", "name": "Zeus (Male)"},
    ],
}


# Cache voice-listing endpoint availability so we don't spam 404s on every poll.
# Maps base_url → (has_api: bool, timestamp: float).
#
# Asymmetric TTLs:
#   positive (API works): 300s — saves real round-trips when API is healthy
#   negative (API 404'd):  30s  — recovers fast when a provider is still
#     loading. Chatterbox's healthcheck start_period is 300s, so a 300s
#     negative cache exactly matches the worst-case startup window: probe
#     during boot → cache "no API" → never re-probe until expiry. 30s is
#     long enough to suppress spammy polling on truly missing endpoints,
#     short enough that the voice tab populates within seconds of the
#     provider finishing its warmup.
import time as _time

_VOICE_CACHE_TTL_POSITIVE = 300.0  # seconds
_VOICE_CACHE_TTL_NEGATIVE = 30.0   # seconds
_voice_api_cache: dict[str, tuple[bool, float]] = {}


def _voice_cache_get(base_url: str) -> bool | None:
    """Get cached voice API availability, respecting TTL. Returns None if expired."""
    entry = _voice_api_cache.get(base_url)
    if entry is None:
        return None
    available, ts = entry
    ttl = _VOICE_CACHE_TTL_POSITIVE if available else _VOICE_CACHE_TTL_NEGATIVE
    if _time.monotonic() - ts > ttl:
        del _voice_api_cache[base_url]
        return None
    return available


def _voice_cache_set(base_url: str, available: bool) -> None:
    _voice_api_cache[base_url] = (available, _time.monotonic())


def _is_chatterbox_provider(provider: dict) -> bool:
    """Check if a provider is the bundled Chatterbox instance (standard or Turbo)."""
    pid = provider.get("id", "")
    base = (provider.get("base_url") or "").lower()
    return pid in ("chatterbox-tts", "chatterbox-turbo") or "chatterbox" in base


def _is_chatterbox_standard(provider: dict) -> bool:
    """Check if provider is travisvn/chatterbox-tts-api (no /v1 prefix on routes)."""
    pid = provider.get("id", "")
    return pid == "chatterbox-tts"


def _is_clone_capable_provider(provider: dict) -> bool:
    """Check if a provider supports voice cloning via local voice files.

    Returns True for providers whose synthesis path accepts a WAV path
    (or a name that resolves to one) as the voice argument:
      * ``chatterbox-tts`` — mounts ``/data/voices`` and resolves names.
      * ``pockettts-builtin`` — in-process; ``pocket_tts.PocketTTS``
        resolves a bare name against ``/data/voices`` before falling
        through to Pocket's built-in voice library.
    """
    pid = (provider or {}).get("id", "")
    return _is_chatterbox_provider(provider) or pid == "pockettts-builtin"


def _is_qwen_provider(provider: dict) -> bool:
    """Check if a provider is the bundled Qwen3-TTS instance."""
    pid = provider.get("id", "")
    base = (provider.get("base_url") or "").lower()
    return pid == "qwen-tts" or "qwen" in base


def _is_fish_provider(provider: dict) -> bool:
    """Check if a provider is a Fish Speech instance."""
    pid = provider.get("id", "")
    base = (provider.get("base_url") or "").lower()
    return pid == "fish-tts" or "fish" in base


def is_inline_emotion_provider(provider: dict) -> bool:
    """Check if provider uses inline emotion tags (Fish Speech style).

    These providers expect emotion tags embedded in the text itself
    (e.g. ``[excited]Hello![/excited]``) rather than a separate instruct parameter.
    """
    return _is_fish_provider(provider)


# Qwen3-TTS CustomVoice speaker metadata (native language + sort order)
_QWEN_SPEAKERS: dict[str, tuple[str, int]] = {
    # (language_tag, sort_order)  — EN first
    "ryan":     ("EN", 0),
    "aiden":    ("EN", 1),
    "vivian":   ("CN", 10),
    "serena":   ("CN", 11),
    "uncle_fu": ("CN", 12),
    "dylan":    ("CN", 13),
    "eric":     ("CN", 14),
    "ono_anna": ("JP", 20),
    "sohee":    ("KR", 21),
}


async def _fetch_voices_from_provider(provider: dict) -> list[dict]:
    """Fetch voice list from a single provider, returning normalized dicts."""
    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider.get("api_key"), base_url=base_url)

    # Return hardcoded voices for providers without listing endpoints
    if _is_openai_tts(base_url):
        return [v.copy() for v in _KNOWN_VOICES["openai"]]
    if _is_deepgram(base_url):
        return [v.copy() for v in _KNOWN_VOICES["deepgram"]]

    # Fish Speech: base voice + any uploaded reference voices
    if _is_fish_provider(provider):
        voices: list[dict] = [{"id": "", "name": "Default (Base Model)"}]
        try:
            req_headers = {**headers, "Accept": "application/json"}
            async with _audio_client(base_url) as client:
                resp = await client.get(
                    f"{base_url}/v1/references/list",
                    headers=req_headers, timeout=10.0,
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        # Fish may return msgpack — decode raw
                        raw = resp.content
                        # Simple msgpack str array extraction
                        data = {"reference_ids": []}
                    ref_ids = data.get("reference_ids", [])
                    for rid in ref_ids:
                        voices.append({"id": rid, "name": rid})
        except Exception as exc:
            log.debug("fish_reference_list_failed", error=str(exc))
        return voices

    has_local_voices = _is_clone_capable_provider(provider)

    # For clone-capable providers: return local cloned voices directly,
    # skip API if we already know it doesn't have a voice listing endpoint.
    if has_local_voices and _voice_cache_get(base_url) is False:
        voices = _list_local_voices()
        if voices:
            return voices
        if provider.get("default_voice"):
            return [{"id": provider["default_voice"], "name": provider["default_voice"]}]
        return []

    api_voices: list[dict] = []
    # Only attempt API listing if we haven't cached a negative result
    if _voice_cache_get(base_url) is not False:
        async with _audio_client(base_url) as client:
            found_api = False
            # Probe order: most-specific real route first. /voices is the
            # primary upstream path on chatterbox-tts-api (and most others
            # accept the OpenAI-shaped /v1/voices alias). /v1/audio/voices is
            # the OpenAI-omni shape (sglang-omni / Higgs Audio v3) and returns
            # {"voices": [...names...], "uploaded_voices": [...]} including
            # cloned voices. It's probed LAST so providers that answer /voices
            # break before reaching it, and the negative cache below prevents
            # repeat 404s for those that don't.
            for path in ("/v1/voices", "/voices", "/v1/audio/voices"):
                try:
                    resp = await client.get(f"{base_url}{path}", headers=headers, timeout=10.0)
                    if resp.status_code != 200:
                        continue
                    found_api = True
                    data = resp.json()
                    raw = data if isinstance(data, list) else data.get("voices", data.get("data", []))
                    for v in raw:
                        if isinstance(v, str):
                            api_voices.append({"id": v, "name": v})
                        elif isinstance(v, dict):
                            entry = {
                                "id": v.get("id", v.get("voice_id", v.get("name", ""))),
                                "name": v.get("name", v.get("id", "")),
                                **{k: v[k] for k in v if k not in ("id", "name")},
                            }
                            # Normalize: Kokoro/Pocket use `lang`, chatterbox
                            # (and other ISO-639 providers) use `language`.
                            # UI surfaces — especially the per-language
                            # learning picker — only check `lang`, so we
                            # mirror `language` over without dropping it.
                            if "language" in entry and "lang" not in entry:
                                entry["lang"] = entry["language"]
                            api_voices.append(entry)
                    break
                except Exception as exc:
                    log.debug("voice_api_probe_failed", base_url=base_url, error=str(exc))
                    continue
            # Cache the result so we don't keep hitting 404 endpoints
            _voice_cache_set(base_url, found_api)
            if not found_api:
                log.debug("voice_api_not_available", base_url=base_url)

    # Merge in locally-saved cloned voices for clone-capable providers
    if has_local_voices:
        local = _list_local_voices()
        api_ids = {v["id"] for v in api_voices}
        for lv in local:
            if lv["id"] not in api_ids:
                api_voices.append(lv)

    # Qwen: filter broken OpenAI aliases, annotate with language tags, sort EN first
    if api_voices and _is_qwen_provider(provider):
        # Only valid OpenAI aliases (map to real 0.6B speakers)
        _valid_aliases = {"alloy", "echo"}
        # Remove aliases that map to non-existent speakers (fable→Sophia, nova→Isabella, etc.)
        api_voices = [
            v for v in api_voices
            if v["id"].lower() in _QWEN_SPEAKERS
            or v["id"].lower() in _valid_aliases
            or v.get("cloned")
        ]
        for v in api_voices:
            vid = v["id"].lower()
            if vid in _QWEN_SPEAKERS:
                tag, _ = _QWEN_SPEAKERS[vid]
                if f"({tag})" not in v["name"]:
                    v["name"] = f"{v['name']} ({tag})"
        api_voices.sort(key=lambda v: _QWEN_SPEAKERS.get(v["id"].lower(), ("ZZ", 99))[1])

    if api_voices:
        return api_voices

    # Fallback: provider's default voice
    if provider.get("default_voice"):
        return [{"id": provider["default_voice"], "name": provider["default_voice"]}]
    return []


@router.post("/api/audio/csm/pin")
async def csm_pin_provider(request: Request):
    """Pin/unpin a CSM voice provider's model so it stays GPU-resident (no slow
    reload+compile) while testing. Body: {provider_id, pinned: bool}."""
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")
    body = await request.json()
    provider_id = str(body.get("provider_id", ""))
    pinned = bool(body.get("pinned", True))
    provider = await _get_provider_by_id(conn, provider_id)
    if not provider or not _is_csm_provider(provider.get("id", "")):
        raise HTTPException(400, "not a CSM voice provider")
    base_url = normalize_base_url(provider["base_url"])
    try:
        async with _audio_client(base_url) as client:
            resp = await client.post(f"{base_url}/{'pin' if pinned else 'unpin'}", timeout=12.0)
            resp.raise_for_status()
        return JSONResponse(content={"ok": True, "pinned": pinned})
    except Exception as exc:  # noqa: BLE001
        log.warning("csm_pin_failed", provider=provider_id, pinned=pinned, error=str(exc)[:160])
        raise HTTPException(502, f"pin request failed: {exc}") from exc


@router.get("/api/audio/csm/pin")
async def csm_pin_status(request: Request, provider_id: str):
    """Read a CSM provider's pin + load state (from its /health) so the UI
    toggle can reflect reality. Non-CSM providers report is_csm=False."""
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")
    provider = await _get_provider_by_id(conn, provider_id)
    if not provider or not _is_csm_provider(provider.get("id", "")):
        return JSONResponse(content={"is_csm": False})
    base_url = normalize_base_url(provider["base_url"])
    try:
        async with _audio_client(base_url) as client:
            h = (await client.get(f"{base_url}/health", timeout=8.0)).json()
        return JSONResponse(content={"is_csm": True, "reachable": True,
                                     "pinned": bool(h.get("pinned", False)),
                                     "loaded": bool(h.get("loaded", False))})
    except Exception:  # noqa: BLE001 — provider may be down
        return JSONResponse(content={"is_csm": True, "reachable": False,
                                     "pinned": False, "loaded": False})


@router.get("/v1/audio/voices")
async def openai_list_voices(request: Request):
    """OpenAI-compatible voice list — ``{"voices": [{"id","name"}, …]}``.

    Bridges Augmentum's TTS voice catalog onto the OpenAI-style path that
    OpenAI-compatible clients (Open WebUI, etc.) query to populate their voice
    picker. Without it, a client wired to Augmentum's ``/v1`` gets a 404 here
    and silently falls back to the hardcoded OpenAI voices (alloy/echo/…),
    which Augmentum can't synthesize — so the user's real voices never show.
    Reuses the same aggregation as ``/api/audio/voices``; the returned ``id``
    is what the client sends back as ``voice`` to ``/v1/audio/speech``, where
    ``resolve_voice_provider`` maps the plain name to its provider.
    """
    resp = await list_voices(request)
    import json as _json
    try:
        data = _json.loads(resp.body) if getattr(resp, "body", None) else []
    except Exception:  # noqa: BLE001
        data = []
    voices, seen = [], set()
    for v in data or []:
        vid = v.get("voice_id") or v.get("name") or ""
        if not vid or vid in seen:
            continue
        seen.add(vid)
        label = v.get("name") or vid
        prov = v.get("provider_name") or ""
        voices.append({"id": vid, "name": f"{label} ({prov})" if prov else label})
    return JSONResponse(content={"voices": voices})


@router.get("/api/audio/voices")
async def list_voices(request: Request, provider_id: str = ""):
    """Fetch available voices from TTS providers.

    If provider_id is given, fetches from that provider only.
    Otherwise aggregates voices from ALL enabled TTS providers,
    tagging each voice with its provider info.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content=[])

    if provider_id:
        provider = await _get_provider_by_id(conn, provider_id)
        if not provider:
            return JSONResponse(content=[])
        voices = await _fetch_voices_from_provider(provider)
        for v in voices:
            v["provider_id"] = provider["id"]
            v["provider_name"] = provider.get("name", provider["id"])
        return JSONResponse(content=voices)

    # Include built-in Kokoro voices (in-process, no provider needed)
    all_voices = []
    if settings.tts_kokoro_builtin and not settings.tts_kokoro_url:
        try:
            from augmentum.voice.kokoro_tts import _RECOMMENDED_GRADES, VOICE_META, KokoroTTS
            kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
            if not kokoro.is_available:
                await load_model_off_loop(kokoro.load_model)
            if kokoro.is_available:
                for name in kokoro.get_voices():
                    meta = VOICE_META.get(name, {})
                    grade = meta.get("grade", "")
                    all_voices.append({
                        "name": name,
                        "voice_id": name,
                        "provider_id": "kokoro-builtin",
                        "provider_name": "Kokoro (built-in)",
                        "grade": grade,
                        "gender": meta.get("gender", ""),
                        "lang": meta.get("lang", ""),
                        "description": meta.get("desc", ""),
                        "recommended": grade in _RECOMMENDED_GRADES,
                    })
        except Exception as exc:
            log.warning("kokoro_builtin_voice_list_error", error=str(exc), exc_info=True)

    # Include built-in PocketTTS voices (Kyutai pocket-tts, CPU)
    if settings.tts_pocket_builtin:
        try:
            from augmentum.voice.pocket_tts import VOICE_META as _POCKET_VOICE_META
            from augmentum.voice.pocket_tts import PocketTTS
            pkt = PocketTTS.instance(
                model_dir=settings.tts_pocket_model_dir,
                language=settings.tts_pocket_language,
            )
            if not pkt.is_available:
                await load_model_off_loop(pkt.load_model)
            if pkt.is_available:
                for name in pkt.get_voices():
                    meta = _POCKET_VOICE_META.get(name, {})
                    all_voices.append({
                        "name": name,
                        "voice_id": name,
                        "provider_id": "pockettts-builtin",
                        "provider_name": "Pocket TTS (built-in)",
                        "gender": meta.get("gender", ""),
                        "lang": meta.get("lang", "en"),
                        "description": meta.get("desc", ""),
                    })
                # Pocket is a clone-capable provider — its engine resolves
                # bare voice names against ``/data/voices/{name}.wav`` via
                # ``PocketTTS._resolve_clone_path``. Merge cloned voices in
                # so the picker surfaces them under the Pocket TTS group
                # alongside the built-ins. Mirrors the merge Chatterbox
                # does in the HTTP-provider listing path.
                try:
                    cloned = _list_local_voices()
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "pocket_local_voice_listing_failed",
                        error=str(exc)[:160],
                    )
                    cloned = []
                for cv in cloned:
                    cname = cv.get("name") or cv.get("id") or ""
                    if not cname:
                        continue
                    all_voices.append({
                        "name": cname,
                        "voice_id": cname,
                        "provider_id": "pockettts-builtin",
                        "provider_name": "Pocket TTS (built-in)",
                        "gender": "",
                        "lang": "en",
                        "description": "Cloned voice",
                        "cloned": True,
                        "file": cv.get("file", ""),
                        "size": cv.get("size", 0),
                    })
        except Exception as exc:
            log.warning("pocket_builtin_voice_list_error", error=str(exc), exc_info=True)

    # Include saved voice mixes from DB (with blend metadata if available)
    try:
        from augmentum.voice.kokoro_tts import RECOMMENDED_BLENDS
        _blend_meta = {b["name"]: b for b in RECOMMENDED_BLENDS}
    except ImportError:
        _blend_meta = {}
    try:
        # Include system/bundled blends (user_id IS NULL) plus the caller's
        # saved mixes. A tenant's mixes never appear for other tenants.
        uid = _user_id(request)
        cursor = await conn.execute(
            "SELECT name, blend_spec, provider_id FROM voice_mixes "
            "WHERE user_id IS NULL OR user_id = ? "
            "ORDER BY created_at",
            (uid,),
        )
        for row in await cursor.fetchall():
            mix_name, blend_spec, pid = row[0], row[1], row[2]
            meta = _blend_meta.get(mix_name, {})
            all_voices.append({
                "name": mix_name,
                "id": blend_spec,
                "voice_id": blend_spec,
                "provider_id": pid,
                "provider_name": (
                    "Kokoro (built-in)" if pid == "kokoro-builtin"
                    else "Pocket TTS (built-in)" if pid == "pockettts-builtin"
                    else pid
                ),
                "is_mix": True,
                "recommended": mix_name in _blend_meta,
                "description": meta.get("desc", ""),
                "gender": meta.get("gender", ""),
                "lang": meta.get("lang", ""),
            })
    except Exception as exc:
        # voice_mixes table may not exist yet on first run before the
        # migration runner has caught up (aiosqlite.OperationalError);
        # keep broad to avoid an import of aiosqlite just for this gate.
        log.debug("voice_mixes_list_skipped", error=str(exc))

    # Aggregate from all enabled TTS providers
    cursor = await conn.execute(
        "SELECT id, name, base_url, api_key, default_model, default_voice "
        "FROM audio_providers WHERE provider_type = 'tts' AND is_enabled = 1 "
        "ORDER BY is_default DESC, name"
    )
    rows = await cursor.fetchall()
    if not rows and not all_voices:
        return JSONResponse(content=[])
    for r in rows:
        prov = {
            "id": r[0], "name": r[1], "base_url": r[2],
            "api_key": decrypt_api_key(r[3]), "default_model": r[4], "default_voice": r[5],
        }
        # Skip built-in providers — their voices are already injected above
        if prov["base_url"] == "builtin":
            continue
        try:
            voices = await _fetch_voices_from_provider(prov)
            for v in voices:
                v["provider_id"] = prov["id"]
                v["provider_name"] = prov["name"]
            all_voices.extend(voices)
        except Exception:
            log.warning("voice_fetch_failed", provider=prov["id"], exc_info=True)

    # Attach a ``sources`` list to every locally-known voice. ``sources``
    # is the load-bearing field that scales the dropdown UX to N peers:
    # each voice has ONE row regardless of how many boxes can serve it,
    # and the row carries an array of {provider_id, node_id, hostname,
    # icon, is_local} entries the UI uses to render a "•N" badge and the
    # routing layer uses to pick the actual dispatch target.
    #
    # Pre-fix the voices endpoint silently dropped peer voices whose
    # names collided with local voices — for bundled engines (Kokoro,
    # Pocket TTS) every box has the same voice catalog, so 100% of peer
    # voices were filtered. The new design keeps ONE entry per voice
    # name but enumerates all sources.
    for v in all_voices:
        if "sources" not in v:
            v["sources"] = [{
                "provider_id": v.get("provider_id", ""),
                "provider_name": v.get("provider_name", ""),
                "is_local": True,
                "node_id": None,
                "hostname": None,
                "icon": None,
            }]

    # Index local entries by voice name so we can augment instead of
    # duplicate when peers advertise the same voice.
    voices_by_name: dict[str, dict] = {}
    for v in all_voices:
        vname = v.get("name") or v.get("voice_id", "")
        if vname and vname not in voices_by_name:
            voices_by_name[vname] = v

    # Fabric peers — voices come from the capability registry, no extra
    # HTTP fetch. For each peer voice: if local already has it, append
    # to that voice's ``sources`` (preserving local-first routing as the
    # default while making peer availability visible). If local doesn't
    # have it (custom-cloned voice on a peer, peer-only Chatterbox etc.),
    # create a fresh entry with the peer as the sole source. Voice
    # metadata (grade/gender/lang) is static per voice name for built-in
    # engines, so we look it up from local VOICE_META maps; a peer's
    # "af_heart" IS the same Kokoro voice with the same metadata.
    if _fabric_coordinator is not None:
        try:
            from augmentum.fabric.capabilities import KIND_TTS_SYNTHESIZE
            try:
                from augmentum.voice.kokoro_tts import (
                    _RECOMMENDED_GRADES as _KOKORO_REC,
                )
                from augmentum.voice.kokoro_tts import (
                    VOICE_META as _KOKORO_META,
                )
            except Exception:
                _KOKORO_META, _KOKORO_REC = {}, set()
            try:
                from augmentum.voice.pocket_tts import VOICE_META as _POCKET_META
            except Exception:
                _POCKET_META = {}

            for node_id, cap in _fabric_coordinator.find_peers_with_capability(
                KIND_TTS_SYNTHESIZE,
            ):
                peer_pid = getattr(cap, "provider_id", "")
                if not peer_pid:
                    continue
                state = _fabric_coordinator.peer_state(node_id)
                paired = state.paired if state else None
                hostname = paired.hostname if paired else ""
                icon = paired.icon if paired else ""
                fabric_id = f"{_FABRIC_PROVIDER_PREFIX}{node_id}:{peer_pid}"
                provider_label = getattr(cap, "provider_name", "") or peer_pid
                display_label = (
                    f"{provider_label} ({hostname})" if hostname else provider_label
                )
                engine = getattr(cap, "engine", "")

                peer_source = {
                    "provider_id": fabric_id,
                    "provider_name": display_label,
                    "is_local": False,
                    "node_id": node_id,
                    "hostname": hostname,
                    "icon": icon,
                }

                for vname in getattr(cap, "voices", []) or []:
                    if not vname:
                        continue
                    existing = voices_by_name.get(vname)
                    if existing is not None:
                        # Local (or earlier peer) already has this voice.
                        # Augment its sources so the UI can render an
                        # availability badge; routing decisions read this.
                        existing.setdefault("sources", []).append(peer_source)
                        continue
                    # Peer-only voice: create a fresh entry with the
                    # peer as the sole source. ``provider_id`` is the
                    # fabric id so existing surfaces that only read
                    # provider_id route correctly.
                    if engine == "kokoro":
                        meta = _KOKORO_META.get(vname, {})
                        grade = meta.get("grade", "")
                        v_entry = {
                            "name": vname,
                            "voice_id": vname,
                            "grade": grade,
                            "gender": meta.get("gender", ""),
                            "lang": meta.get("lang", ""),
                            "description": meta.get("desc", ""),
                            "recommended": grade in _KOKORO_REC,
                        }
                    elif engine == "pockettts":
                        meta = _POCKET_META.get(vname, {})
                        v_entry = {
                            "name": vname,
                            "voice_id": vname,
                            "gender": meta.get("gender", ""),
                            "lang": meta.get("lang", "en"),
                            "description": meta.get("desc", ""),
                        }
                    else:
                        v_entry = {"name": vname, "voice_id": vname}
                    v_entry["provider_id"] = fabric_id
                    v_entry["provider_name"] = display_label
                    v_entry["augmentum_peer"] = {
                        "node_id": node_id,
                        "hostname": hostname,
                        "icon": icon,
                    }
                    v_entry["sources"] = [peer_source]
                    all_voices.append(v_entry)
                    voices_by_name[vname] = v_entry
        except Exception:
            log.warning("voice_fabric_listing_failed", exc_info=True)

    # Update voice→provider map as a side effect (so TTS routing is fresh)
    global _voice_provider_map, _voice_map_ts
    import time as _time
    new_map: dict[str, str] = {}
    for v in all_voices:
        vname = v.get("name") or v.get("voice_id", "")
        pid = v.get("provider_id", "")
        if vname and pid and vname not in new_map:
            new_map[vname] = pid
    _voice_provider_map = new_map
    _voice_map_ts = _time.time()

    return JSONResponse(content=all_voices)


@router.get("/api/audio/fabric_diagnostic")
async def audio_fabric_diagnostic(request: Request):
    """Dump every connected fabric peer's advertised audio capabilities.

    Operator-facing diagnostic for verifying that cross-peer audio
    capability heartbeats are flowing end-to-end. Used to disambiguate
    "I don't see peer voices in the dropdown" — is it (a) the substrate
    not propagating caps, or (b) the dropdown UI filtering them out?
    This endpoint answers (a); the dropdown's own behavior is the
    answer to (b).

    Returns one entry per known peer (connected or not) with:
      - identity: node_id, hostname, icon, addr
      - connection: connected (bool), last_seen_monotonic
      - tts: list of TTSSynthesizeCapability dicts (engine, provider_id,
        voices, languages, ...)
      - stt: list of STTTranscribeCapability dicts
      - other_kinds: count of non-audio caps so operators can sanity-
        check that the peer is advertising other modalities too

    Also reports the LOCAL node's audio caps under ``local`` so the
    operator can compare what THIS box thinks it offers vs what each
    peer thinks IT offers — making cross-fleet drift visible.
    """
    out: dict[str, Any] = {
        "fabric_enabled": bool(settings.fabric_enabled),
        "local": {"tts": [], "stt": [], "other_kinds": 0},
        "peers": [],
    }

    coord = _fabric_coordinator
    if coord is None:
        out["error"] = (
            "fabric_coordinator not registered on this node — "
            "fabric is either disabled or not yet initialised"
        )
        return JSONResponse(content=out)

    from dataclasses import asdict

    from augmentum.fabric.capabilities import (
        KIND_STT_TRANSCRIBE,
        KIND_TTS_SYNTHESIZE,
    )

    # Local capabilities (whatever this node's extractors last produced).
    try:
        local_caps = coord.local_capabilities()
        for cap in local_caps:
            if cap.kind == KIND_TTS_SYNTHESIZE:
                out["local"]["tts"].append(asdict(cap))
            elif cap.kind == KIND_STT_TRANSCRIBE:
                out["local"]["stt"].append(asdict(cap))
            else:
                out["local"]["other_kinds"] += 1
    except Exception as exc:
        out["local"]["error"] = str(exc)[:200]

    # Per-peer view. ``_peers`` includes both connected and not-currently-
    # connected paired peers; the diagnostic surfaces both so an operator
    # can tell "peer X is paired but disconnected" apart from "peer X
    # never paired".
    try:
        for node_id, state in coord._peers.items():
            paired = state.paired
            tts_caps = []
            stt_caps = []
            other = 0
            for cap in state.capabilities:
                if cap.kind == KIND_TTS_SYNTHESIZE:
                    tts_caps.append(asdict(cap))
                elif cap.kind == KIND_STT_TRANSCRIBE:
                    stt_caps.append(asdict(cap))
                else:
                    other += 1
            out["peers"].append({
                "node_id": node_id,
                "hostname": paired.hostname if paired else "",
                "icon": paired.icon if paired else "",
                "addr": paired.addr if paired else "",
                "connected": bool(state.connected),
                "last_seen_monotonic": state.last_seen_monotonic,
                "tts": tts_caps,
                "stt": stt_caps,
                "other_kinds": other,
            })
    except Exception as exc:
        out["peers_error"] = str(exc)[:200]

    return JSONResponse(content=out)


# ---------------------------------------------------------------------------
# Provider-specific TTS — POST /v1/audio/speech with provider routing
# ---------------------------------------------------------------------------


@router.post("/api/audio/speech")
async def tts_speech_routed(body: TTSRequest, request: Request, provider_id: str = ""):
    """Generate speech via a specific provider.

    If provider_id is given, routes to that provider.
    Falls back to default TTS provider.
    """
    if not settings.audio_tts_enabled:
        raise HTTPException(503, "TTS is not enabled")

    # Per-voice pronunciation lexicon — same pass as the OpenAI-compat
    # endpoint above; this routed path also serves the voice previews,
    # so a saved row's test button speaks the corrected form.
    from augmentum.voice import lexicon_store
    body.input = await lexicon_store.apply(
        _get_conn(request), body.input,
        user_id=_user_id(request), voice=body.voice or "",
    )

    # Clean text for TTS — provider-aware so Chatterbox Turbo tags survive.
    # Done here rather than in _build_tts_stream to avoid double-cleaning.
    from augmentum.voice.text_cleaning import clean_for_tts
    if provider_id == "chatterbox-turbo":
        from augmentum.voice.emotion import inject_turbo_tags
        body.input = clean_for_tts(inject_turbo_tags(body.input), preserve_brackets=True) or body.input
    else:
        body.input = clean_for_tts(body.input) or body.input

    # Built-in Kokoro: generate speech in-process (no sidecar needed)
    if provider_id == "kokoro-builtin" or (
        not provider_id and settings.tts_kokoro_builtin and not settings.tts_kokoro_url
    ):
        from augmentum.voice.kokoro_tts import KokoroTTS
        kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
        if not kokoro.is_available:
            await load_model_off_loop(kokoro.load_model)
        if not kokoro.is_available:
            raise HTTPException(503, "Built-in Kokoro TTS is not available — model failed to load")
        voice_name = body.voice or "af_heart"
        media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")
        try:
            audio = await kokoro.generate(
                body.input,
                voice=voice_name,
                speed=body.speed,
                response_format=body.response_format,
            )
            return Response(
                content=audio,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'inline; filename="speech.{body.response_format}"',
                },
            )
        except Exception as exc:
            log.warning("kokoro_builtin_routed_tts_error", error=str(exc))
            raise HTTPException(503, f"Built-in Kokoro TTS failed: {exc}")

    # Built-in Kokoro voice combine (blend endpoint)
    if (provider_id == "kokoro-builtin" and hasattr(body, 'voices') and body.voices):
        from augmentum.voice.kokoro_tts import KokoroTTS
        kokoro = KokoroTTS.instance()
        if kokoro.is_available:
            voice_names = [v.get("name", "") for v in body.voices if v.get("name")]
            if len(voice_names) < 2:
                raise HTTPException(400, "At least 2 voices are required")

            parts = []
            for v in body.voices:
                name = v.get("name", "")
                weight = v.get("weight", 1)
                if name:
                    parts.append(f"{name}*{weight}" if weight != 1 else name)
            blend_spec = "+".join(parts)
            save_name = body.save_as or blend_spec

            kokoro._resolve_voice(blend_spec)

            return JSONResponse(content={
                "status": "ok",
                "combined_voice": blend_spec,
                "saved_as": save_name,
            })

    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    # Hot dispatch lookup → read connection. See ``tts_speech`` above.
    read_conn = _get_read_conn(request)

    if provider_id:
        provider = await _get_provider_by_id(read_conn, provider_id)
        if not provider:
            raise HTTPException(404, f"Provider '{provider_id}' not found")
    else:
        provider = await _get_default_provider(read_conn, "tts")
        if not provider:
            raise HTTPException(503, "No TTS provider configured")

    # Any built-in in-process engine reached via the fall-through (e.g. an
    # explicit ?provider_id=pockettts-builtin, or it being the DB default) —
    # generate in-process; never try to HTTP to "builtin".
    if provider.get("base_url") == "builtin" and provider.get("id") in _BUILTIN_TTS_IDS:
        eng = await _builtin_tts_engine(provider["id"])
        if eng is None:
            raise HTTPException(503, f"Built-in TTS '{provider['id']}' is not available — model failed to load")
        voice_name = body.voice or provider.get("default_voice", "")
        media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")
        try:
            audio = await eng.generate(
                body.input, voice=voice_name, speed=body.speed,
                response_format=body.response_format,
            )
            return Response(
                content=audio, media_type=media_type,
                headers={"Content-Disposition": f'inline; filename="speech.{body.response_format}"'},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("builtin_tts_routed_error", provider=provider["id"], error=str(exc))
            raise HTTPException(503, f"Built-in TTS '{provider['id']}' failed: {exc}")

    base_url = normalize_base_url(provider["base_url"])
    model = body.model or provider.get("default_model", "")
    voice = body.voice or provider.get("default_voice", "")

    media_type = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")

    # See ``tts_speech`` above for the prime-stream rationale.
    from augmentum.proxy.streaming import StreamPrimeError, prime_stream

    raw_stream = _build_tts_stream(
        base_url, model, voice, body.input,
        body.response_format, body.speed, provider.get("api_key"),
        pre_cleaned=True,
        instruct=body.instruct,
        provider_id=provider.get("id", ""),
        user_id=_user_id(request),
        session_id=request.headers.get(SESSION_HEADER, "") or body.session_id,
    )
    try:
        primed = await prime_stream(raw_stream)
    except StreamPrimeError as exc:
        cause = exc.cause
        if isinstance(cause, httpx.HTTPStatusError):
            status = cause.response.status_code if cause.response else 502
            body = sanitize_error_detail((await _safe_upstream_detail(cause.response))[:500])
            detail = body or f"provider returned {status}"
            log.warning("tts_routed_upstream_error", status=status, detail=detail)
            raise HTTPException(status, f"TTS provider error: {detail}")
        log.warning("tts_routed_connection_error", error=repr(cause))
        raise HTTPException(502, f"Could not reach TTS provider: {cause!r}")
    return StreamingResponse(
        content=primed,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="speech.{body.response_format}"'},
    )


# ---------------------------------------------------------------------------
# Kokoro Voice Mixer — combine/blend voices
# ---------------------------------------------------------------------------


class VoiceMixRequest(BaseModel):
    """Request to combine multiple Kokoro voices with weights."""
    voices: list[dict] = Field(..., min_length=2)  # [{"name": "af_bella", "weight": 2}, ...]
    save_as: str = ""  # Optional name to save the combined voice


@router.post("/api/audio/voices/combine")
async def combine_voices(body: VoiceMixRequest, request: Request, provider_id: str = "kokoro-builtin"):
    """Combine multiple voices into a named blend.

    For a built-in engine (Kokoro / Pocket TTS): averages the style vectors
    in-process via numpy and persists the blend spec to ``voice_mixes``.
    For sidecar Kokoro: proxies to Kokoro-FastAPI's combine endpoint.
    """
    # Built-in engine: blend in-process (same 256-dim averaging for both).
    if provider_id in _BUILTIN_TTS_IDS:
        eng = await _builtin_tts_engine(provider_id)
        if eng is not None:
            voice_names = [v.get("name", "") for v in body.voices if v.get("name")]
            if len(voice_names) < 2:
                raise HTTPException(400, "At least 2 voices are required")

            parts = []
            for v in body.voices:
                name = v.get("name", "")
                weight = v.get("weight", 1)
                if name:
                    parts.append(f"{name}*{weight}" if weight != 1 else name)
            blend_spec = "+".join(parts)
            save_name = body.save_as or blend_spec

            eng._resolve_voice(blend_spec)   # build + cache the blended embedding

            # Persist to DB so the mix survives restarts and appears in voice lists
            conn = _get_conn(request)
            if conn and body.save_as:
                uid = _user_id(request)
                try:
                    await conn.execute(
                        "INSERT OR REPLACE INTO voice_mixes "
                        "(name, blend_spec, provider_id, user_id) "
                        "VALUES (?, ?, ?, ?)",
                        (save_name, blend_spec, provider_id, uid or None),
                    )
                    await conn.commit()
                    system_events.publish("voices.changed", {"reason": "mix"}, user_id=uid)
                except Exception:
                    log.warning("voice_mix_save_failed", name=save_name, exc_info=True)

            return JSONResponse(content={
                "status": "ok",
                "combined_voice": blend_spec,
                "saved_as": save_name,
            })

    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    provider = await _get_provider_by_id(conn, provider_id)
    if not provider:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider.get("api_key"), base_url=base_url)
    headers["Content-Type"] = "application/json"

    # Build voice name list for Kokoro.
    # Kokoro's /v1/audio/voices/combine expects a plain JSON string
    # ("af_bella+af_sky") or a list of strings (["af_bella", "af_sky"]).
    voice_names = [v.get("name", "") for v in body.voices if v.get("name")]
    if len(voice_names) < 2:
        raise HTTPException(400, "At least 2 voices are required to combine")
    combined_voice = "+".join(voice_names)

    try:
        async with _audio_client(base_url) as client:
            # Send as a plain JSON list of voice name strings
            resp = await client.post(
                f"{base_url}/v1/audio/voices/combine",
                json=voice_names,
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code == 200:
                # Kokoro returns a .pt file (binary tensor) on success.
                # Save into the shared Kokoro voices volume so the
                # combined voice appears in Kokoro's /v1/audio/voices
                # listing automatically.
                save_name = body.save_as or combined_voice
                try:
                    import pathlib
                    # Shared volume: compose.kokoro.yaml mounts
                    # kokoro_voices → /data/kokoro_voices (Augmentum)
                    #               → /app/api/voices   (Kokoro)
                    voice_dir = pathlib.Path("/data/kokoro_voices")
                    if not voice_dir.exists():
                        # Fallback for non-Docker or missing volume
                        voice_dir = pathlib.Path("data/voice_cache")
                    voice_dir.mkdir(parents=True, exist_ok=True)
                    save_path = voice_dir / f"{save_name}.pt"
                    save_path.write_bytes(resp.content)
                    log.info("voice_combine_saved", path=str(save_path), size=len(resp.content))
                except Exception as save_exc:
                    log.warning("voice_combine_save_failed", error=str(save_exc))

                return JSONResponse(content={
                    "status": "ok",
                    "combined_voice": combined_voice,
                    "saved_as": save_name,
                })
            # Error — try to parse error detail
            try:
                error_detail = resp.json()
            except Exception:
                error_detail = sanitize_error_detail(resp.text[:300])
            return JSONResponse(content={
                "status": "error",
                "error": error_detail,
            }, status_code=resp.status_code)
    except Exception as exc:
        raise HTTPException(502, f"Voice combine failed: {exc}")


@router.delete("/api/audio/voices/mixes/{mix_name}")
async def delete_voice_mix(mix_name: str, request: Request):
    """Delete one of the caller's saved voice mixes.

    System/bundled mixes (``user_id IS NULL``) are not deletable via this
    endpoint — they belong to the server install, not any tenant.
    """
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "Unauthorized")

    cursor = await conn.execute(
        "SELECT name FROM voice_mixes WHERE name = ? AND user_id = ?",
        (mix_name, uid),
    )
    if not await cursor.fetchone():
        raise HTTPException(404, f"Mix '{mix_name}' not found")

    # Check if this is a voice walk — clean up .npy file from disk
    cursor2 = await conn.execute(
        "SELECT blend_spec FROM voice_mixes WHERE name = ? AND user_id = ?",
        (mix_name, uid),
    )
    row = await cursor2.fetchone()
    if row and row[0] and row[0].startswith("walk:"):
        walk_name = row[0].removeprefix("walk:")
        # Defense in depth: the create path (clone_voice_kokoro) already
        # sanitises voice_name to ``[A-Za-z0-9_-]{1,64}`` before writing
        # ``walk:<name>`` to blend_spec, so a name reaching here that
        # contains '..', '/', '\\', or anything else is either a direct
        # DB edit or a future code path that bypasses the sanitiser.
        # Refuse the file delete on a mismatched name (still drop the DB
        # row below — that's the user's actual intent) and contain the
        # resolved path to the walks directory.
        if _is_safe_voice_walk_name(walk_name):
            voice_dir = pathlib.Path("/data/kokoro_voices")
            if not voice_dir.exists():
                voice_dir = pathlib.Path("data/voice_cache")
            walks_base = (voice_dir.parent / "voice_walks").resolve()
            candidate = (walks_base / f"{walk_name}.npy").resolve()
            try:
                # On Python 3.9+ `is_relative_to` is available; fall back
                # to a string-prefix check that handles the case where
                # the path doesn't exist yet (resolve still works on
                # non-existent paths in strict=False mode, our default).
                contained = candidate.is_relative_to(walks_base)
            except AttributeError:
                contained = str(candidate).startswith(str(walks_base) + os.sep)
            if contained and candidate.is_file():
                candidate.unlink()
                log.info("voice_walk_npy_deleted", name=walk_name)
            elif not contained:
                log.warning(
                    "voice_walk_npy_path_escape_blocked",
                    name=walk_name, resolved=str(candidate), base=str(walks_base),
                )
        else:
            log.warning("voice_walk_npy_skip_unsafe_name", name=walk_name[:64])

    await conn.execute(
        "DELETE FROM voice_mixes WHERE name = ? AND user_id = ?",
        (mix_name, uid),
    )
    await conn.commit()
    system_events.publish("voices.changed", {"reason": "mix"}, user_id=uid)
    log.info("voice_mix_deleted", name=mix_name, user_id=uid)
    return JSONResponse(content={"status": "deleted"})


# ---------------------------------------------------------------------------
# Voice preview — quick TTS sample
# ---------------------------------------------------------------------------


@router.post("/api/audio/voices/preview")
async def preview_voice(request: Request, provider_id: str = "", voice: str = "", text: str = ""):
    """Generate a short TTS preview for a voice."""
    if not text:
        text = "Hello, this is a voice preview. How does this sound?"

    body = TTSRequest(input=text, voice=voice, response_format="mp3")
    return await tts_speech_routed(body, request, provider_id=provider_id)


# ---------------------------------------------------------------------------
# Provider WebUI quicklinks
# ---------------------------------------------------------------------------


_WEBUI_HINTS = {
    "speaches-stt": {"port": 6200, "path": "/", "label": "Speaches WebUI"},
    "kokoro-tts": {"port": 6300, "path": "/web", "label": "Kokoro Voice Playground"},
    # Chatterbox ships a separate React UI container (chatterbox-ui) on
    # 6401. If the user opted out of it, fall back to the API's /docs
    # explorer on 6400 — always present, no extra container.
    "chatterbox-tts": {
        "port": 6401,
        "path": "/",
        "label": "Chatterbox Voice Studio",
        "fallback_port": 6400,
        "fallback_path": "/docs",
        "fallback_label": "Chatterbox API Explorer",
    },
}


# Probe cache for fallback resolution. Hints with a `fallback_*` block are
# resolved at most once per process (UI lifetime — settings panel opens
# trigger this) to avoid repeated 1s timeouts when the studio container
# isn't running. Cleared on process restart, which is the right
# invalidation interval since compose changes require a restart anyway.
_webui_resolved: dict[str, dict] = {}


async def _resolve_webui_hint(http_client: httpx.AsyncClient, hint: dict) -> dict:
    """Probe the primary port; if unreachable + fallback configured, swap."""
    if not hint.get("fallback_port"):
        return hint
    port = hint["port"]
    if port in _webui_resolved:
        return _webui_resolved[port]
    # Probe the host-side port — the UI link is rendered for the browser,
    # so the browser-reachable port is what matters here. localhost from
    # inside the augmentum container also resolves to the host loopback
    # when running under Docker Desktop (which is what setup.bat targets).
    try:
        resp = await http_client.get(
            f"http://localhost:{port}{hint['path']}",
            timeout=httpx.Timeout(connect=0.5, read=0.5, write=0.5, pool=0.5),
        )
        if resp.status_code < 500:
            _webui_resolved[port] = hint
            return hint
    except Exception as exc:
        # Surface the cause so "broken WebUI link" debugging has a log
        # trail — connection refused vs. timeout vs. DNS error all
        # collapse into the same fallback below otherwise.
        log.debug("webui_probe_failed", port=port, error=str(exc)[:160])
    fallback = {
        "port": hint["fallback_port"],
        "path": hint["fallback_path"],
        "label": hint["fallback_label"],
    }
    _webui_resolved[port] = fallback
    return fallback


@router.get("/api/audio/providers/webui")
async def get_webui_links(request: Request):
    """Return WebUI quicklinks for providers that have them."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content=[])

    cursor = await conn.execute(
        "SELECT id, name FROM audio_providers WHERE is_enabled = 1"
    )
    rows = await cursor.fetchall()

    http_client = getattr(request.app.state, "http_client", None)

    links = []
    for r in rows:
        provider_id = r[0]
        hint = _WEBUI_HINTS.get(provider_id)
        if not hint:
            continue
        resolved = await _resolve_webui_hint(http_client, hint) if http_client else hint
        links.append({
            "provider_id": provider_id,
            "provider_name": r[1],
            "label": resolved["label"],
            "port": resolved["port"],
            "path": resolved["path"],
        })

    return JSONResponse(content=links)


# ---------------------------------------------------------------------------
# Bundled services status — tells the UI which packaged containers are active
# ---------------------------------------------------------------------------

# Well-known IDs for containers bundled via compose files
_BUNDLED_IDS = {"moonshine-stt", "speaches-stt", "kokoro-builtin", "pockettts-builtin", "kokoro-tts", "chatterbox-tts", "chatterbox-turbo", "qwen-tts", "fish-tts", "openai-tts"}


@router.get("/api/audio/providers/bundled")
async def get_bundled_services(request: Request):
    """Return which bundled audio containers are registered and enabled.

    The UI uses this to conditionally show Kokoro voice mixer,
    voice cloning (Chatterbox / Qwen3), etc. — only when the user
    installed the bundled container, not when using a remote API.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content={})

    placeholders = ",".join("?" for _ in _BUNDLED_IDS)
    cursor = await conn.execute(
        f"SELECT id, is_enabled FROM audio_providers WHERE id IN ({placeholders})",
        tuple(_BUNDLED_IDS),
    )
    rows = await cursor.fetchall()
    result = {r[0]: bool(r[1]) for r in rows}
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Voice cloning — save reference audio to shared voice directory
# ---------------------------------------------------------------------------

_VOICE_DIR: str = ""


def _get_voice_dir() -> str:
    """Resolve the voice directory path (lazy, cached)."""
    global _VOICE_DIR  # noqa: PLW0603
    if _VOICE_DIR:
        return _VOICE_DIR
    import pathlib

    from augmentum.config import settings as _cfg
    _VOICE_DIR = str(pathlib.Path(_cfg.data_dir) / "voices")
    return _VOICE_DIR


def _ensure_voice_dir() -> str:
    """Create the voice directory if it doesn't exist and return the path."""
    import pathlib
    voice_dir = _get_voice_dir()
    pathlib.Path(voice_dir).mkdir(parents=True, exist_ok=True)
    return voice_dir


def _sanitize_voice_name(voice_name: str, fallback: str = "clone") -> str:
    """Filesystem-safe voice name: alnum/-/_ only, capped at 64 chars.

    Single source of truth for the rule ``clone_voice`` applies, so the
    fabric voice-clone receiver derives identical on-disk names (and can't
    be tricked into path traversal via ``../`` in a peer-supplied name)."""
    safe = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in (voice_name or "")
    ).strip("_")[:64]
    return safe or fallback


def save_voice_clone_files(
    voice_name: str, audio_bytes: bytes, *, filename: str = "", transcript: str = "",
) -> tuple[str, str]:
    """Write a clone reference (clip + optional transcript) into the shared
    voice dir, the way ``clone_voice`` does locally.

    A co-located CSM sidecar (local or on this peer) reads the ``.wav`` +
    ``<name>.txt`` from the shared volume to build its ``(text, audio)``
    clone anchor. Returns ``(safe_name, saved_filename)``.
    """
    import pathlib
    safe = _sanitize_voice_name(voice_name)
    voice_dir = _ensure_voice_dir()
    ext = pathlib.Path(filename or "").suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}:
        ext = ".wav"
    save_path = pathlib.Path(voice_dir) / f"{safe}{ext}"
    save_path.write_bytes(audio_bytes)
    if transcript.strip():
        (pathlib.Path(voice_dir) / f"{safe}.txt").write_text(
            transcript.strip(), encoding="utf-8",
        )
    return safe, save_path.name


def _list_local_voices() -> list[dict]:
    """Scan the voice directory for audio files and return voice metadata."""
    import pathlib
    voice_dir = pathlib.Path(_get_voice_dir())
    if not voice_dir.exists():
        return []
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
    voices = []
    for f in sorted(voice_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in audio_exts:
            voices.append({
                "id": f.stem,
                "name": f.stem,
                "file": f.name,
                "size": f.stat().st_size,
                "cloned": True,
            })
    return voices


@router.get("/api/audio/voices/cloned")
async def list_cloned_voices():
    """List all locally-saved voice clone presets."""
    return JSONResponse(content=_list_local_voices())


@router.delete("/api/audio/voices/cloned/{voice_name}")
async def delete_cloned_voice(voice_name: str):
    """Delete a cloned voice file from the voice directory."""
    import pathlib
    voice_dir = pathlib.Path(_get_voice_dir())
    # Search for any audio file matching the voice name
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
    deleted = False
    for ext in audio_exts:
        target = voice_dir / f"{voice_name}{ext}"
        if target.is_file():
            target.unlink()
            deleted = True
            log.info("cloned_voice_deleted", voice_name=voice_name, file=target.name)
    if not deleted:
        raise HTTPException(404, f"Voice '{voice_name}' not found")
    # Cloned voices live on the shared server filesystem (not per-tenant),
    # so broadcast.
    system_events.publish("voices.changed", {"reason": "clone"})
    return JSONResponse(content={"status": "ok", "deleted": voice_name})


_CLONE_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def _normalize_clone_audio(audio_bytes: bytes, filename: str = "") -> tuple[bytes, str]:
    """Normalise an uploaded voice-clone reference to mono PCM16 WAV.

    Browsers and recorders hand us all sorts of encodings — notably
    **32-bit float WAV** (RIFF format tag 3), which PocketTTS's WAV loader
    rejects outright (``unknown format: 3`` → silent fallback to the default
    voice) and the Moonshine batch reader misreads as near-silence (empty
    transcript). Both failures were one root cause: the codec, not the
    content. ffmpeg decodes any input; we re-emit signed 16-bit PCM
    (RIFF tag 1), mono, at the SOURCE sample rate — no resample, so cloning
    fidelity is preserved; only the broken codec is replaced.

    Returns ``(wav_bytes, ".wav")`` on success. Falls back to the original
    bytes (with a safe extension) if ffmpeg is missing or the convert fails,
    so a clone is never lost to a transcode hiccup — it just stays in its
    original format and the caller's existing best-effort paths still run.
    """
    import os
    import pathlib
    import shutil
    import subprocess
    import tempfile

    src_ext = pathlib.Path(filename or "").suffix.lower()
    fallback_ext = src_ext if src_ext in _CLONE_AUDIO_EXTS else ".wav"

    if not shutil.which("ffmpeg"):
        log.warning("voice_clone_normalize_no_ffmpeg")
        return audio_bytes, fallback_ext

    inp_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=src_ext or ".bin", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name
        out_path = inp_path.rsplit(".", 1)[0] + ".clone.wav"
        # -c:a pcm_s16le → 16-bit PCM (RIFF tag 1, what every loader reads).
        # -ac 1 → mono (clone refs are single-speaker). No -ar: keep the
        # source rate so we don't throw away cloning detail.
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", inp_path, "-c:a", "pcm_s16le", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                wav_bytes = f.read()
            log.info(
                "voice_clone_normalized",
                from_size=len(audio_bytes), to_size=len(wav_bytes),
            )
            return wav_bytes, ".wav"
        log.warning(
            "voice_clone_normalize_failed",
            stderr=(result.stderr[:200].decode("utf-8", "replace")
                    if result.stderr else ""),
        )
        return audio_bytes, fallback_ext
    except Exception:
        log.warning("voice_clone_normalize_error", exc_info=True)
        return audio_bytes, fallback_ext
    finally:
        for p in (inp_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


@router.post("/api/audio/voices/clone")
async def clone_voice(
    request: Request,
    audio: UploadFile,
    voice_name: str = Form(""),
):
    """Save a voice reference audio file for Chatterbox zero-shot cloning.

    Accepts a short (5-10s) audio clip, saves it to the shared voice
    directory (bind-mounted into the Chatterbox container), and optionally
    transcribes it via the default STT provider for reference.

    Chatterbox uses the audio file directly as a voice reference when the
    voice name matches the filename (without extension) in its
    VOICE_LIBRARY_DIR.
    """
    if not voice_name:
        voice_name = (audio.filename or "clone").rsplit(".", 1)[0]
    # Sanitise — letters, numbers, underscores, hyphens only
    voice_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in voice_name
    ).strip("_")[:64]
    if not voice_name:
        voice_name = "clone"

    # 1) Read uploaded audio bytes
    audio_bytes = await audio.read()
    if len(audio_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(413, "Audio file too large (max 10 MB)")

    # 1b) Normalise to mono PCM16 WAV before anything touches the bytes. Raw
    #     uploads arrive in encodings the synth/STT can't read — notably
    #     32-bit float WAV, which PocketTTS rejects ("unknown format: 3" →
    #     silent fallback to the default voice) and the Moonshine batch reader
    #     hears as silence (empty transcript). One convert up front fixes the
    #     saved file, the transcription, AND the provider/fabric uploads — they
    #     all share these bytes + the normalised filename/content-type below.
    audio_bytes, clone_ext = await asyncio.to_thread(
        _normalize_clone_audio, audio_bytes, audio.filename or "",
    )
    clone_fname = f"{voice_name}{clone_ext}"
    clone_ctype = (
        "audio/wav" if clone_ext == ".wav" else (audio.content_type or "audio/wav")
    )

    # 2) Save to shared voice directory
    import pathlib
    voice_dir = _ensure_voice_dir()

    save_path = pathlib.Path(voice_dir) / clone_fname
    save_path.write_bytes(audio_bytes)
    log.info(
        "voice_clone_saved",
        voice_name=voice_name,
        file=save_path.name,
        size=len(audio_bytes),
    )

    # 3) Optional transcription via STT (informational only)
    transcript = ""
    conn = _get_conn(request)
    if conn:
        stt_provider = await _get_default_provider(conn, "stt")
        # Skip built-in providers (e.g. Moonshine) — they don't have an HTTP API
        if stt_provider and stt_provider["base_url"] not in ("builtin", ""):
            try:
                stt_url = normalize_base_url(stt_provider["base_url"])
                stt_headers = _build_headers(stt_provider.get("api_key"))
                stt_model = stt_provider.get("default_model", "")

                stt_files: dict[str, Any] = {
                    "file": (clone_fname, audio_bytes, clone_ctype),
                }
                stt_data: dict[str, str] = {}
                if stt_model:
                    stt_data["model"] = stt_model

                async with _audio_client(stt_url) as client:
                    stt_resp = await client.post(
                        f"{stt_url}/v1/audio/transcriptions",
                        files=stt_files,
                        data=stt_data,
                        headers=stt_headers,
                        timeout=60.0,
                    )
                    if stt_resp.status_code == 200:
                        stt_json = stt_resp.json()
                        transcript = stt_json.get("text", "")
                    else:
                        log.warning("voice_clone_stt_failed", status=stt_resp.status_code)
            except Exception:
                log.warning("voice_clone_stt_error", exc_info=True)

    # 3b) Fall back to built-in Moonshine when there's no HTTP STT provider
    # (the common case — Moonshine is the default). CSM cloning needs the
    # transcript, so we persist it beside the audio as <name>.txt. Other
    # engines ignore the sidecar file; the CSM sidecar reads it to build a
    # proper (text, audio) clone anchor. Shares every clone to CSM for free.
    if not transcript.strip():
        try:
            transcript = await _moonshine_batch_transcribe(
                audio_bytes, clone_fname,
            )
        except Exception:
            log.warning("voice_clone_moonshine_error", exc_info=True)
    if transcript.strip():
        try:
            (pathlib.Path(voice_dir) / f"{voice_name}.txt").write_text(
                transcript.strip(), encoding="utf-8",
            )
            log.info("voice_clone_transcript_saved", voice_name=voice_name)
        except Exception:
            log.warning("voice_clone_transcript_save_failed", exc_info=True)

    # 4) Try to upload to voice-cloning TTS providers (best-effort)
    uploaded_to_api = False
    if conn:
        _clone_providers = ["chatterbox-tts", "chatterbox-turbo"]
        fname = clone_fname
        ctype = clone_ctype
        for _cp_id in _clone_providers:
            cp = await _get_provider_by_id(conn, _cp_id)
            if not cp or cp["base_url"] in ("", "builtin"):
                continue
            cp_url = normalize_base_url(cp["base_url"])
            cp_headers = _build_headers(cp.get("api_key"))
            # travisvn/chatterbox-tts-api uses /voices, others use /v1/voices
            upload_path = "/voices" if _cp_id == "chatterbox-tts" else "/v1/voices"
            try:
                async with _audio_client(cp_url) as client:
                    resp = await client.post(
                        f"{cp_url}{upload_path}",
                        files={"voice_file": (fname, audio_bytes, ctype)},
                        data={"voice_name": voice_name},
                        headers=cp_headers,
                        timeout=30.0,
                    )
                    if resp.status_code in (200, 201):
                        uploaded_to_api = True
            except Exception:
                log.debug("voice_clone_api_upload_skipped", provider=_cp_id, exc_info=True)

        # OpenAI-omni style providers (Higgs Audio v3 via openai-tts) store
        # clones SERVER-SIDE through their own multipart API — name/audio_sample/
        # ref_text/consent — not chatterbox's /voices + voice_file shape. Register
        # the clone there too so it becomes a selectable voice. ref_text reuses
        # the transcript we already computed (Higgs uses it to improve clone
        # fidelity). Best-effort + model-agnostic: any /v1/audio/voices endpoint
        # with this shape benefits; others just 404 and we move on.
        ocp = await _get_provider_by_id(conn, "openai-tts")
        if ocp and ocp["base_url"] not in ("", "builtin"):
            ocp_url = normalize_base_url(ocp["base_url"])
            try:
                async with _audio_client(ocp_url) as client:
                    resp = await client.post(
                        f"{ocp_url}/v1/audio/voices",
                        files={"audio_sample": (clone_fname, audio_bytes, clone_ctype)},
                        data={
                            "name": voice_name,
                            "ref_text": transcript or "",
                            "consent": "User initiated this voice clone in Augmentum.",
                        },
                        headers=_build_headers(ocp.get("api_key")),
                        timeout=60.0,
                    )
                    if resp.status_code in (200, 201):
                        uploaded_to_api = True
                        log.info("voice_clone_openai_uploaded", voice_name=voice_name)
                    else:
                        log.debug(
                            "voice_clone_openai_upload_status",
                            status=resp.status_code, body=resp.text[:160],
                        )
            except Exception:
                log.debug("voice_clone_openai_upload_skipped", exc_info=True)

    # 5) Share the clone with fabric CSM peers. A *local* CSM reads this
    # clone from the shared /voices volume for free (the writes above) —
    # but a *remote* peer has its own volume, so the clip + transcript have
    # to cross the wire. Only CSM engines need this (they clone from a
    # (text, audio) anchor); other peer TTS engines manage their own refs.
    uploaded_to_fabric: list[str] = []
    coordinator = _fabric_coordinator
    if coordinator is not None and getattr(settings, "fabric_enabled", False):
        identity = getattr(coordinator, "_identity", None)
        eff_user_id = _user_id(request)
        if identity is not None and eff_user_id:
            from augmentum.fabric.audio_client import clone_upload_via_peer
            from augmentum.fabric.capabilities import KIND_TTS_SYNTHESIZE

            fname = clone_fname
            ctype = clone_ctype

            async def _push_clone(node_id: str, peer_pid: str) -> str | None:
                fabric_id = f"{_FABRIC_PROVIDER_PREFIX}{node_id}:{peer_pid}"
                fp = _fabric_provider_dict(fabric_id)
                if not fp or not fp.get("base_url"):
                    return None
                try:
                    await clone_upload_via_peer(
                        http_client_factory=_fabric_audio_client,
                        identity=identity,
                        user_id=eff_user_id,
                        peer_base_url=normalize_base_url(fp["base_url"]),
                        audio_bytes=audio_bytes,
                        filename=fname,
                        content_type=ctype,
                        voice_name=voice_name,
                        transcript=transcript,
                    )
                    return node_id
                except Exception as exc:  # noqa: BLE001 — best-effort per peer
                    log.warning(
                        "voice_clone_fabric_push_failed",
                        node_id=node_id, error=str(exc)[:160],
                    )
                    return None

            try:
                targets = [
                    (node_id, getattr(cap, "provider_id", ""))
                    for node_id, cap in coordinator.find_peers_with_capability(
                        KIND_TTS_SYNTHESIZE,
                    )
                    if getattr(cap, "engine", "") == "csm"
                    and getattr(cap, "provider_id", "")
                ]
                if targets:
                    results = await asyncio.gather(
                        *(_push_clone(nid, pid) for nid, pid in targets)
                    )
                    uploaded_to_fabric = [r for r in results if r]
            except Exception:
                log.warning("voice_clone_fabric_discovery_failed", exc_info=True)

    # Cloned voices live on the shared server filesystem (not per-tenant),
    # so broadcast the refresh signal.
    system_events.publish("voices.changed", {"reason": "clone"})
    return JSONResponse(content={
        "status": "ok",
        "voice_name": voice_name,
        "file": save_path.name,
        "transcript": transcript,
        "uploaded_to_api": uploaded_to_api,
        "uploaded_to_fabric": uploaded_to_fabric,
    })


# ---------------------------------------------------------------------------
# Kokoro Voice Walk — evolutionary voice cloning
# ---------------------------------------------------------------------------

@router.post("/api/audio/voices/clone-kokoro")
async def clone_voice_kokoro(
    request: Request,
    audio: UploadFile,
    voice_name: str = Form(""),
    steps: int = Form(1000),
    seed_voice: str = Form(""),
):
    """Clone a voice for Kokoro TTS via evolutionary embedding optimization.

    Accepts a short audio clip (5-30s) of the target speaker and optimizes
    a Kokoro voice embedding to match. Streams NDJSON progress updates.

    The result is saved to the voice_mixes table and immediately usable
    as a Kokoro voice in all TTS endpoints.

    Requires: pip install resemblyzer
    """
    if not settings.tts_kokoro_builtin:
        raise HTTPException(503, "Built-in Kokoro TTS is not enabled")

    from augmentum.voice.kokoro_tts import KokoroTTS
    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    if not kokoro.is_available:
        await load_model_off_loop(kokoro.load_model)
    if not kokoro.is_available:
        raise HTTPException(503, "Kokoro TTS model not loaded")

    # Read and validate audio
    audio_bytes = await audio.read()
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "Audio file too large (max 20 MB)")
    if len(audio_bytes) < 1000:
        raise HTTPException(400, "Audio file too small — need at least a few seconds of speech")

    # Sanitize voice name
    if not voice_name:
        voice_name = (audio.filename or "cloned").rsplit(".", 1)[0]
    voice_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in voice_name
    ).strip("_")[:64] or "cloned"

    steps = min(max(steps, 100), 5000)

    # Check resemblyzer is available before committing to the stream
    try:
        import resemblyzer  # noqa: F401
    except ImportError:
        raise HTTPException(
            503,
            "Voice cloning requires the 'resemblyzer' package. "
            "Install it: pip install resemblyzer",
        )

    # Decode audio to numpy float32
    try:
        target_audio, target_sr = await _decode_audio_to_numpy(audio_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not decode audio file: {exc}")

    from augmentum.voice.voice_walk import clone_voice_walk_stream

    async def _stream():
        async for update in clone_voice_walk_stream(
            kokoro,
            target_audio,
            target_sr,
            steps=steps,
            seed_voice=seed_voice,
        ):
            if update["status"] == "complete":
                # Extract and save the embedding before JSON serialization
                embedding = update.pop("_embedding", None)
                _save_walk_embedding(voice_name, embedding)

                # Cache in kokoro so it's immediately usable
                if embedding is not None:
                    with kokoro._voice_cache_lock:
                        kokoro._voice_cache[f"walk:{voice_name}"] = embedding

                # Persist to voice_mixes table
                conn = _get_conn(request)
                if conn:
                    uid = _user_id(request)
                    try:
                        await conn.execute(
                            "INSERT OR REPLACE INTO voice_mixes "
                            "(name, blend_spec, provider_id, user_id) "
                            "VALUES (?, ?, 'kokoro-builtin', ?)",
                            (voice_name, f"walk:{voice_name}", uid or None),
                        )
                        await conn.commit()
                        system_events.publish("voices.changed", {"reason": "walk"}, user_id=uid)
                    except Exception:
                        log.warning("voice_walk_save_failed", name=voice_name, exc_info=True)

                update["voice_name"] = voice_name
            else:
                # Strip numpy arrays from progress updates
                update.pop("_embedding", None)
            yield _json_line(update)

    return StreamingResponse(
        content=_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


async def _decode_audio_to_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes (any format) to mono float32 numpy array via ffmpeg."""
    import numpy as np
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)
    if proc.returncode != 0 or not stdout:
        err = stderr[:200].decode(errors="replace") if stderr else "unknown error"
        raise HTTPException(400, f"Failed to decode audio: {err}")
    samples = np.frombuffer(stdout, dtype=np.float32)
    return samples, 16000


def _save_walk_embedding(voice_name: str, embedding: np.ndarray | None) -> str:
    """Save a voice walk embedding to disk for persistence across restarts."""
    import pathlib

    import numpy as np
    walk_dir = pathlib.Path(settings.data_dir or "/data") / "voice_walks"
    walk_dir.mkdir(parents=True, exist_ok=True)
    path = walk_dir / f"{voice_name}.npy"
    if embedding is not None:
        np.save(str(path), embedding)
        log.info("voice_walk_embedding_saved", path=str(path))
    return str(path)


def _json_line(obj: dict) -> str:
    """Encode a dict as a JSON line for NDJSON streaming."""
    import json as _json
    return _json.dumps(obj, default=str) + "\n"


# ---------------------------------------------------------------------------
# Phoneme-driven lip sync test endpoint
# ---------------------------------------------------------------------------


class LipSyncTestRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "mp3"


@router.post("/api/voice/test/lipsync")
async def voice_test_lipsync(body: LipSyncTestRequest):
    """Generate Kokoro audio + matching viseme schedule for the avatar testbench.

    Synchronous (non-streaming) — computes the full audio in one shot so
    the duration is known exactly, then builds a phoneme schedule scaled
    to that duration. Returns audio inline as base64 alongside the schedule
    so the testbench can iterate without juggling separate URLs.

    This endpoint is for the avatar testbench / Phase 1 validation only —
    the live voice path uses streaming TTS in tts.py with schedule events
    sent over the same WebSocket as audio chunks.
    """
    import base64

    import numpy as np

    from augmentum.voice.kokoro_tts import (
        KokoroTTS,
        _apply_hbe,
        _apply_prosodic_steering,
        _encode_audio,
        _voice_lang,
    )
    from augmentum.voice.phoneme_lipsync import is_lang_supported, text_to_schedule

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    if not kokoro.is_available:
        await load_model_off_loop(kokoro.load_model)
    if not kokoro.is_available:
        raise HTTPException(503, "Built-in Kokoro TTS is not available")

    voice_name = body.voice or "af_heart"
    lang = _voice_lang(voice_name)

    resolved_voice = kokoro._resolve_voice(voice_name)
    if not voice_name.startswith("walk:"):
        resolved_voice = _apply_prosodic_steering(kokoro, resolved_voice, text)

    try:
        # Pre-HBE samples — these come straight from Kokoro at its native
        # sample rate. Duration measurement uses the post-HBE rate to match
        # what the client actually plays.
        samples, sr = await asyncio.to_thread(
            kokoro._kokoro.create,
            text,
            voice=resolved_voice,
            speed=body.speed,
            lang=lang,
        )
        samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)
    except Exception as exc:
        log.warning("voice_test_lipsync_synth_failed", error=str(exc))
        raise HTTPException(500, f"TTS synthesis failed: {exc}")

    duration_ms = int(round(len(samples) / sr * 1000)) if sr else 0

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    audio_bytes = await _encode_audio(pcm16, sr, body.response_format)

    schedule = None
    schedule_source = "skipped"
    if is_lang_supported(lang):
        schedule = text_to_schedule(text, duration_ms, lang=lang)
        schedule_source = "g2p_en" if schedule else "fallback"

    mime = _CONTENT_TYPES.get(body.response_format, "audio/mpeg")

    return JSONResponse(content={
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        "mime": mime,
        "duration_ms": duration_ms,
        "sample_rate": sr,
        "lang": lang,
        "voice": voice_name,
        "schedule": schedule,
        "schedule_source": schedule_source,
    })
