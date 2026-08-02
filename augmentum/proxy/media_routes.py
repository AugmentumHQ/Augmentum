"""Media-server management + streaming proxy.

Per-user CRUD for ``user_media_servers`` plus an authenticated streaming
endpoint that forwards Range requests and preserves status codes so
HTML5 ``<audio>`` / ``<video>`` seeking works through the proxy the same
way direct-streaming does.

All routes are user-scoped — the user's own browser is the only client
we expose these to. No admin gate (unlike ``provider_routes``, which
manages shared LLM infrastructure).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from augmentum.auth.guards import is_admin
from augmentum.media.detector import detect_servers
from augmentum.media.library_store import get_media_library_store
from augmentum.media.normalize import (
    author_for_match,
    normalize_name,
    tokens_match_as_related,
)
from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider
from augmentum.media.providers.base import (
    DEFAULT_PORTS,
    BrowseResult,
    provider_supports_browse,
    provider_supports_remote_control,
    provider_supports_remote_general_control,
)
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.providers.komga import KomgaProvider
from augmentum.media.providers.librivox import (
    ARCHIVE_COVER,
    LibrivoxProvider,
    normalise_details_to_catalog,
    normalise_librivox_sections,
)
from augmentum.media.providers.suwayomi import SuwayomiProvider
from augmentum.media.receivers import build_receiver_launch_plan, list_receiver_profiles
from augmentum.media.receivers.dlna import (
    discover_dlna_receivers,
    launch_dlna_receiver,
    send_dlna_general_command,
    send_dlna_playstate_command,
    snapshot_dlna_receiver,
)
from augmentum.media.receivers.runtime import ReceiverRuntime, TransportSession
from augmentum.media.store import MediaServerStore, purge_server_data
from augmentum.media.sync import sync_server
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger
from augmentum.vfs import register_file, unregister_file

log = get_logger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])


# --- Request shapes ------------------------------------------------------


class AddServerRequest(BaseModel):
    provider: str
    name: str
    base_url: str
    username: str | None = None
    password: str | None = None
    access_token: str | None = None
    # 'shared' = admin-published, visible read-only to every user.
    # Admin-only — non-admin posting scope='shared' gets a 403.
    scope: str | None = None


class UpdateServerRequest(BaseModel):
    name: str | None = None
    base_url: str | None = None
    username: str | None = None
    password: str | None = None
    access_token: str | None = None
    # Admin-only toggle. None = leave scope unchanged.
    scope: str | None = None


class ChangePasswordRequest(BaseModel):
    new_password: str = Field(min_length=8, max_length=256)
    # Step-up re-auth: the caller's OWN Augmentum password, re-verified
    # before this sensitive credential change is applied.
    augmentum_password: str = Field(default="", max_length=256)


class AccessRequest(BaseModel):
    # Step-up re-auth for the "Access & login" panel — revealing the managed
    # credential requires re-confirming the caller's Augmentum password.
    augmentum_password: str = Field(default="", max_length=256)


class ProgressUpdate(BaseModel):
    current_time_s: float
    duration_s: float
    is_finished: bool = False
    episode_id: str = ""


class PlaybackSelectionUpdate(BaseModel):
    media_source_id: str | None = None
    audio_stream_index: int | None = None
    subtitle_stream_index: int | None = None


class BookmarkCreate(BaseModel):
    position_s: float
    label: str = ""
    note: str = ""
    episode_id: str = ""


class RemoteSessionPlayRequest(BaseModel):
    session_id: str
    start_time_s: float = 0.0
    play_command: str = "PlayNow"
    media_source_id: str | None = None
    audio_stream_index: int | None = None
    subtitle_stream_index: int | None = None


class RemoteSessionCommandRequest(BaseModel):
    session_id: str
    command: str
    seek_position_s: float | None = None


class RemoteSessionGeneralCommandRequest(BaseModel):
    command: str
    arguments: dict[str, str | int | float | bool | None] | None = None


class TransportPlayRequest(BaseModel):
    transport: str
    receiver_id: str
    receiver_profile: str = "dlna_generic_video"


class UpdateLibraryRequest(BaseModel):
    display_name_override: str | None = None
    surface_group_override: str | None = None
    is_hidden: bool | None = None
    include_in_search: bool | None = None
    include_in_overview: bool | None = None
    sort_order: int | None = None


# --- Helpers -------------------------------------------------------------


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request) -> MediaServerStore | None:
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if isinstance(backend, SQLiteBackend):
        return MediaServerStore(backend.conn)
    return None


def _db_conn(request: Request):
    """The raw aiosqlite connection (for managed_services lookups), or None."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return backend.conn if isinstance(backend, SQLiteBackend) else None


async def _verify_step_up(request: Request, password: str) -> JSONResponse | None:
    """Re-verify the caller's Augmentum password for a sensitive media action.

    Augmentum's own login is the point of verification for managing a media
    server's credential (reveal + change). Mirrors PUT /api/auth/me/password:
    per-username+IP lockout, argon2 verify, clear-on-success. Returns an error
    JSONResponse to return directly, or None when verification passes.
    """
    sm = getattr(request.app.state, "session_manager", None)
    user = request.scope.get("user")
    if sm is None or user is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    from augmentum.auth.passwords import verify_password
    from augmentum.proxy.auth_routes import _get_ip
    ip = _get_ip(request)
    lockout = await sm.check_lockout(user.username, ip)
    if lockout is not None:
        return JSONResponse(
            {"error": "Too many failed attempts. Try again later.",
             "retry_after_seconds": lockout},
            status_code=429,
        )
    pw_hash = await sm.get_password_hash(user.username)
    if not pw_hash or not verify_password(pw_hash, password or ""):
        await sm.record_failed_attempt(user.username, ip)
        return JSONResponse(
            {"error": "Augmentum password is incorrect"}, status_code=401)
    await sm.clear_failed_attempts(user.username)
    return None


def _is_managed_instance(server) -> bool:
    """True when this ``user_media_servers`` row points at the container
    Augmentum itself provisioned for its provider.

    The install dispatcher always writes ``http://augmentum-<id>:<port>``
    (the ServiceManager container name; the bare ``<id>`` network alias
    also resolves to it), so the base_url hostname is the discriminator.
    A catalog definition existing for the provider says nothing about a
    given ROW — the user can manually connect an external Jellyfin/ABS
    running on another box, and that row must never inherit the managed
    instance's front door, host ports, or managed login.
    """
    try:
        host = (urlsplit(server.base_url).hostname or "").lower()
    except ValueError:
        return False
    provider = (server.provider or "").lower()
    return bool(provider) and host in {f"augmentum-{provider}", provider}


async def _heal_managed_token(request: Request, server, uid: str):
    """Refresh the stored token for an Augmentum-managed server.

    Augmentum owns the credential for provisioned servers, so a row should
    never get stuck on an empty/stale ``access_token`` (which surfaces as
    "Authentication rejected"). Before a test/sync we re-login with the
    *derived* credential and persist the resulting token across every
    user's row pointing at that instance (the credential is install-wide).
    No-op for manually-added servers, or on any failure (we just fall
    through to the existing token). Returns the (possibly refreshed)
    server row.
    """
    mgr = getattr(request.app.state, "service_manager", None)
    sd = mgr.get_definition(server.provider) if mgr else None
    from augmentum.providers.service_auth import (
        has_managed_credentials,
        resolve_managed_credentials,
    )
    if sd is None or not has_managed_credentials(sd) or not _is_managed_instance(server):
        return server
    http = _http(request)
    store = _get_store(request)
    if http is None or store is None:
        return server
    try:
        user, pw = await resolve_managed_credentials(server.provider, _db_conn(request))
        client = _provider_client(server.provider, http)
        token = await client.login(server.base_url, user, pw)
    except Exception as exc:  # noqa: BLE001 — heal is best-effort
        log.warning(
            "managed_token_heal_failed",
            provider=server.provider, error=str(exc),
        )
        return server
    if token and token != server.access_token:
        await store.update_token_for_provider(
            server.provider, token, base_url=server.base_url,
        )
        refreshed = await store.get_visible(server.id, user_id=uid)
        if refreshed is not None:
            return refreshed
    return server


def _http(request: Request) -> httpx.AsyncClient | None:
    """Shared async httpx client from app state — None during shutdown or
    when the client wasn't initialized (tests with minimal state)."""
    return getattr(request.app.state, "http_client", None)


def _get_index(request: Request):
    return getattr(request.app.state, "file_index", None)


# Short-TTL cache for the file_index entry on read-only media DISPLAY routes
# (cover / backdrop / cast-image / person-profile). A media detail view fires
# many parallel asset requests that all carry the SAME file_id, each re-running
# the identical file_index PK query against the shared aiosqlite connection —
# the ~18-cast-image burst that spiked slow_db_op + the event loop. These paths
# read only the item's IMMUTABLE server context (source_metadata.server_id), so
# a short TTL is safe: server_id never changes for a media item, and worst case
# is a ≤TTL delay on display for an item that was just re-synced/trashed (the
# image proxy simply 404s upstream). Keyed by (user_id, file_id) — no cross-user
# leak. NOT used on mutation paths (favorite / rename / trash / progress), which
# always read fresh.
_ENTRY_CACHE: dict[tuple[str, str], tuple[float, object]] = {}
_ENTRY_CACHE_TTL = 30.0
_ENTRY_CACHE_MAX = 1024


async def _cached_display_entry(idx, file_id: str, uid: str):
    """``idx.get`` with a short TTL, for read-only media display routes only."""
    key = (uid, file_id)
    now = time.monotonic()
    hit = _ENTRY_CACHE.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    entry = await idx.get(file_id, user_id=uid)
    if len(_ENTRY_CACHE) >= _ENTRY_CACHE_MAX:
        _ENTRY_CACHE.clear()  # simple memory bound
    _ENTRY_CACHE[key] = (now + _ENTRY_CACHE_TTL, entry)
    return entry


def _get_library_store():
    return get_media_library_store()


async def _enqueue_server_detach(
    request: Request, *, server_id: str, owner_user_id: str, action: str,
) -> str:
    """Queue teardown (or re-attach) of a server's rows for its borrowers.

    Never inline: a shared library is tens of thousands of file_index
    rows, each with an FTS update trigger, and the shared aiosqlite
    connection serializes every other surface behind it. See
    ``augmentum/media/detach.py``.

    Best-effort by design — both actions are idempotent, so a dropped
    enqueue is recoverable by repeating the un-share/re-share rather
    than something that can corrupt state. Failing the scope toggle
    because a job couldn't be queued would be the worse trade.
    """
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None:
        return ""
    try:
        job_id = await jobs_store.create(
            user_id=owner_user_id,
            job_type="media_server_detach",
            payload={
                "server_id": server_id,
                "owner_user_id": owner_user_id,
                "action": action,
            },
            priority=20,
            max_attempts=3,
        )
        if job_runner is not None:
            job_runner.wake()
        return str(job_id or "")
    except Exception:
        log.warning(
            "media_server_detach_enqueue_failed",
            server_id=server_id, action=action, exc_info=True,
        )
        return ""


def _receiver_runtime(request: Request) -> ReceiverRuntime:
    runtime = getattr(request.app.state, "receiver_runtime", None)
    if isinstance(runtime, ReceiverRuntime):
        return runtime
    runtime = ReceiverRuntime()
    request.app.state.receiver_runtime = runtime
    return runtime


# Sentinel server_id stamped on every LibriVox file_index row. Routes
# detect this and skip user_media_servers lookup — LibriVox is a built-in,
# credential-free provider that every user sees without connecting.
BUILTIN_LIBRIVOX = "builtin-librivox"
_REMOTE_PLAY_COMMANDS = frozenset({"PlayNow", "PlayNext", "PlayLast"})
_REMOTE_PLAYSTATE_COMMANDS = frozenset({
    "Stop",
    "Pause",
    "Unpause",
    "NextTrack",
    "PreviousTrack",
    "Seek",
    "Rewind",
    "FastForward",
    "PlayPause",
    "SeekRelative",
})
_REMOTE_GENERAL_COMMANDS = frozenset({
    "VolumeUp",
    "VolumeDown",
    "SetVolume",
    "Mute",
    "Unmute",
    "ToggleMute",
    "SetAudioStreamIndex",
    "SetSubtitleStreamIndex",
})


def _provider_client(provider: str, http_client: httpx.AsyncClient):
    if provider == "audiobookshelf":
        return AudiobookshelfProvider(http_client)
    if provider == "emby":
        return EmbyProvider(http_client)
    if provider == "jellyfin":
        return JellyfinProvider(http_client)
    if provider == "librivox":
        return LibrivoxProvider(http_client)
    if provider == "komga":
        return KomgaProvider(http_client)
    if provider == "suwayomi":
        return SuwayomiProvider(http_client)
    raise ValueError(f"Unknown provider: {provider}")


def _resolve_remote_cover_request(
    *,
    provider: str,
    client,
    base_url: str,
    access_token: str,
    external_id: str,
    cover_hint: str,
) -> tuple[str, str]:
    """Return ``(url, auth_header)`` for a remote provider cover request.

    Providers do not all authenticate cover bytes the same way:

    - Audiobookshelf expects either a Bearer header or the token embedded in
      the cover URL query string. We use the provider's canonical build helper
      here so the route doesn't accidentally downgrade that into Basic auth.
    - Komga and authenticated Suwayomi instances want HTTP Basic auth, so a
      stored relative ``cover_url`` hint is fine as long as we attach the
      header server-side.
    - No-auth Suwayomi returns public thumbnails, so the auth header stays
      empty naturally when ``access_token`` is blank.
    """
    if provider == "audiobookshelf":
        return client.build_cover_url(base_url, external_id, access_token), ""

    if provider in {"emby", "jellyfin"}:
        if cover_hint.startswith(("http://", "https://")):
            url = cover_hint
        elif cover_hint:
            path = cover_hint if cover_hint.startswith("/") else f"/{cover_hint}"
            api_base = client._api_base(base_url) if hasattr(client, "_api_base") else base_url.rstrip("/")
            url = f"{api_base}{path}"
        else:
            url = client.build_cover_url(base_url, external_id, access_token)
        if access_token and "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={access_token}"
        return url, ""

    if cover_hint.startswith(("http://", "https://")):
        url = cover_hint
    elif cover_hint:
        path = cover_hint if cover_hint.startswith("/") else f"/{cover_hint}"
        url = f"{base_url.rstrip('/')}{path}"
    else:
        url = client.build_cover_url(base_url, external_id, access_token)

    auth_header = ""
    if provider in {"komga", "suwayomi"} and access_token:
        auth_header = f"Basic {access_token}"
    return url, auth_header


def _is_builtin_server(server_id: str) -> bool:
    return server_id == BUILTIN_LIBRIVOX


def _builtin_provider_name(server_id: str) -> str:
    """Map sentinel server_id → provider.name for factory dispatch."""
    if server_id == BUILTIN_LIBRIVOX:
        return "librivox"
    return ""


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _remote_argument_map(
    arguments: dict[str, str | int | float | bool | None] | None,
) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, raw in (arguments or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        out[name] = None if raw is None else str(raw)
    return out


def _source_id_for_remote_item(server_id: str, external_id: str) -> str:
    return f"{server_id}:{external_id}"


async def _file_entry_for_remote_item(
    idx,
    *,
    user_id: str,
    provider: str,
    server_id: str,
    external_id: str,
):
    if not idx or not user_id or not provider or not server_id or not external_id:
        return None
    return await idx.get_by_source(
        provider,
        _source_id_for_remote_item(server_id, external_id),
        user_id=user_id,
    )


def _remote_session_payload(
    *,
    session,
    provider: str,
    server_id: str,
    current_file_id: str = "",
    current_file_name: str = "",
    current_cover_url: str = "",
) -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": provider,
        "server_id": server_id,
        "session": session.to_dict(),
        "current_file_id": current_file_id or "",
        "current_file_name": current_file_name or "",
        "current_cover_url": current_cover_url or "",
    }


def _stream_choice_from_meta(meta: dict, request: Request) -> dict[str, str]:
    query = request.query_params
    choice: dict[str, str] = {}

    media_source_id = (
        (query.get("media_source_id") or "").strip()
        or str(meta.get("preferred_media_source_id") or "").strip()
    )
    if media_source_id:
        choice["MediaSourceId"] = media_source_id

    audio_idx = _int_or_none(
        query.get("audio_stream_index")
        if "audio_stream_index" in query
        else meta.get("preferred_audio_stream_index")
    )
    if audio_idx is not None:
        choice["AudioStreamIndex"] = str(audio_idx)

    subtitle_idx = _int_or_none(query.get("subtitle_stream_index"))
    if subtitle_idx is not None and subtitle_idx >= 0:
        choice["SubtitleStreamIndex"] = str(subtitle_idx)

    return choice


def _preferred_stream_choice(meta: dict) -> dict[str, str]:
    choice: dict[str, str] = {}

    media_source_id = str(meta.get("preferred_media_source_id") or "").strip()
    if media_source_id:
        choice["MediaSourceId"] = media_source_id

    audio_idx = _int_or_none(meta.get("preferred_audio_stream_index"))
    if audio_idx is not None:
        choice["AudioStreamIndex"] = str(audio_idx)

    subtitle_idx = _int_or_none(meta.get("preferred_subtitle_stream_index"))
    if subtitle_idx is not None and subtitle_idx >= 0:
        choice["SubtitleStreamIndex"] = str(subtitle_idx)

    return choice


def _subtitle_choice_from_meta(meta: dict, request: Request) -> tuple[str, int | None]:
    query = request.query_params
    media_source_id = (
        (query.get("media_source_id") or "").strip()
        or str(meta.get("preferred_media_source_id") or "").strip()
    )
    subtitle_idx = _int_or_none(
        query.get("subtitle_stream_index")
        if "subtitle_stream_index" in query
        else meta.get("preferred_subtitle_stream_index")
    )
    return media_source_id, subtitle_idx


def _append_query_params(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query),
        parts.fragment,
    ))


_VTT_TIMING_RE = re.compile(
    r"^(?P<prefix>\s*)(?P<start>(?:\d+:)?\d{2}:\d{2}\.\d{3})"
    r"(?P<arrow>\s*-->\s*)"
    r"(?P<end>(?:\d+:)?\d{2}:\d{2}\.\d{3})(?P<tail>.*)$",
)


def _vtt_timestamp_to_seconds(raw: str) -> float | None:
    parts = [segment.strip() for segment in str(raw or "").split(":")]
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            return None
    except (TypeError, ValueError):
        return None
    return max(0.0, (hours * 3600.0) + (minutes * 60.0) + seconds)


def _seconds_to_vtt_timestamp(value: float) -> str:
    total_ms = max(0, int(round(float(value or 0.0) * 1000.0)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _shift_webvtt(text: str, offset_s: float) -> str:
    if not text or offset_s <= 0:
        return text
    blocks = re.split(r"\r?\n\r?\n", text)
    shifted: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        timing_idx = next(
            (idx for idx, line in enumerate(lines) if _VTT_TIMING_RE.match(line)),
            None,
        )
        if timing_idx is None:
            shifted.append(block)
            continue
        match = _VTT_TIMING_RE.match(lines[timing_idx])
        if not match:
            shifted.append(block)
            continue
        start_s = _vtt_timestamp_to_seconds(match.group("start"))
        end_s = _vtt_timestamp_to_seconds(match.group("end"))
        if start_s is None or end_s is None:
            shifted.append(block)
            continue
        next_end = end_s - offset_s
        if next_end <= 0:
            continue
        next_start = max(0.0, start_s - offset_s)
        lines[timing_idx] = (
            f"{match.group('prefix')}{_seconds_to_vtt_timestamp(next_start)}"
            f"{match.group('arrow')}{_seconds_to_vtt_timestamp(next_end)}"
            f"{match.group('tail')}"
        )
        shifted.append("\n".join(lines))
    return "\n\n".join(shifted)


_BROWSER_UNSUPPORTED_AUDIO_CODECS = frozenset({
    "ac3",
    "eac3",
    "dca",
    "dts",
    "dtshd_hra",
    "dtshd_ma",
    "truehd",
})


def _playback_source_by_id(playback: dict | None, source_id: str) -> dict | None:
    if not isinstance(playback, dict):
        return None
    sources = playback.get("media_sources") or []
    if not isinstance(sources, list) or not sources:
        return None
    selected_id = str(source_id or playback.get("selected_media_source_id") or "").strip()
    for source in sources:
        if not isinstance(source, dict):
            continue
        if str(source.get("id") or "").strip() == selected_id:
            return source
    return sources[0] if isinstance(sources[0], dict) else None


def _playback_audio_track(
    playback: dict | None,
    *,
    source_id: str,
    audio_stream_index: int | None,
) -> dict | None:
    source = _playback_source_by_id(playback, source_id)
    if not isinstance(source, dict):
        return None
    selected_idx = (
        audio_stream_index
        if audio_stream_index is not None
        else _int_or_none(playback.get("selected_audio_stream_index") if isinstance(playback, dict) else None)
    )
    for track in source.get("audio_tracks") or []:
        if not isinstance(track, dict):
            continue
        if _int_or_none(track.get("index")) == selected_idx:
            return track
    return None


async def _emby_compat_browser_stream_url(
    *,
    client,
    server,
    external_id: str,
    cached_meta: dict,
    stream_choice: dict[str, str],
    start_time_s: float | None = None,
) -> str:
    fetch_playback_info = getattr(client, "fetch_playback_info", None)
    build_browser_video_stream_url = getattr(client, "build_browser_video_stream_url", None)
    if not callable(fetch_playback_info) or not callable(build_browser_video_stream_url):
        return ""

    raw = await fetch_playback_info(
        server.base_url,
        server.access_token,
        external_id=external_id,
    )
    if not isinstance(raw, dict):
        return ""

    playback = _normalise_emby_compat_playback(
        {"_augmentum_playback_info": raw},
        cached_meta=cached_meta,
    )
    if not isinstance(playback, dict):
        return ""

    source_id = str(
        stream_choice.get("MediaSourceId")
        or playback.get("selected_media_source_id")
        or ""
    ).strip()
    audio_idx = _int_or_none(
        stream_choice.get("AudioStreamIndex")
        if "AudioStreamIndex" in stream_choice
        else playback.get("selected_audio_stream_index")
    )
    audio_track = _playback_audio_track(
        playback,
        source_id=source_id,
        audio_stream_index=audio_idx,
    )
    audio_codec = str(audio_track.get("codec") or "").strip().lower() if isinstance(audio_track, dict) else ""
    if audio_codec not in _BROWSER_UNSUPPORTED_AUDIO_CODECS:
        return ""

    play_session_id = str(raw.get("PlaySessionId") or "").strip()
    if not source_id or not play_session_id:
        return ""

    start_time_ticks = None
    if start_time_s is not None and start_time_s > 0:
        start_time_ticks = max(0, int(round(float(start_time_s) * 10_000_000.0)))

    return build_browser_video_stream_url(
        server.base_url,
        external_id=external_id,
        media_source_id=source_id,
        play_session_id=play_session_id,
        token=server.access_token,
        audio_stream_index=audio_idx,
        audio_codec="aac",
        max_audio_channels=2,
        start_time_ticks=start_time_ticks,
    )


# --- Server CRUD ---------------------------------------------------------


@router.get("/servers")
async def list_servers(request: Request) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"servers": [], "defaults": DEFAULT_PORTS})
    # Own rows ∪ admin-shared rows. ``is_owned_by_viewer`` on each entry
    # tells the UI whether to render edit/delete affordances.
    servers = await store.list_visible(user_id=uid)
    dicts = [s.to_dict(viewer_user_id=uid) for s in servers]
    # Managed-instance flag + front gate. Both are per-ROW facts, keyed on
    # whether the row's base_url actually points at the Augmentum-provisioned
    # container — a manually-connected external server of the same provider
    # must not advertise the managed instance's gate URL or access panel.
    from augmentum.config import settings as _settings
    gate = (_settings.gate_domain or "").strip().lower()
    from augmentum.providers.service_auth import needs_managed_auth
    mgr = getattr(request.app.state, "service_manager", None)
    for server, d in zip(servers, dicts, strict=True):
        sd = mgr.get_definition(server.provider) if mgr else None
        managed = sd is not None and _is_managed_instance(server)
        d["is_managed_instance"] = managed
        # Front gate: when configured, decorate gate-eligible (managed-auth)
        # instances with their dissolved-login URL so the UI shows an "Open
        # (signed in)" affordance. No-op when the gate is off or the
        # provider isn't gated.
        if gate and managed and needs_managed_auth(sd):
            d["gate_url"] = f"https://{d['provider']}.{gate}:6443"
    return JSONResponse({
        "servers": dicts,
        "defaults": DEFAULT_PORTS,
        "viewer_is_admin": is_admin(request),
    })


@router.get("/servers/{server_id}/console-credentials")
async def media_console_credentials(server_id: str, request: Request) -> JSONResponse:
    """Reveal the managed Basic-auth credential for a provisioned server.

    Admin-only — it's a secret. Lets the post-install setup card show the
    login for the server's own web console (the credential we baked into
    the container at provision time, derived from the install secret).
    Returns ``managed_auth: false`` for servers we don't manage auth for
    (manually-added or no-auth deployments) so the UI hides the panel.
    """
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)

    # Media service ids == provider names (the loader sets both), so the
    # provider name is the catalog key the credential is derived from.
    mgr = getattr(request.app.state, "service_manager", None)
    sd = mgr.get_definition(server.provider) if mgr else None
    from augmentum.providers.service_auth import (
        has_managed_credentials,
        resolve_managed_credentials,
    )
    if sd is None or not has_managed_credentials(sd) or not _is_managed_instance(server):
        return JSONResponse({"managed_auth": False})
    username, password = await resolve_managed_credentials(
        server.provider, _db_conn(request),
    )
    return JSONResponse({
        "managed_auth": True,
        "username": username,
        "password": password,
        "host_port": sd.host_port,
    })


@router.post("/servers/{server_id}/access")
async def media_server_access(
    server_id: str, body: AccessRequest, request: Request,
) -> JSONResponse:
    """Combined access info for a provisioned media server.

    Powers the persistent "Access & login" panel — reachable any time, not
    just on the install card. Returns the dedicated HTTPS front-door port
    (open in a browser / iframe), the raw LAN host port (for native TV/phone
    apps), and the managed login.

    Admin-only AND step-up: revealing the password re-verifies the caller's
    own Augmentum password (an unattended/hijacked admin session shouldn't
    silently surface a server credential). POST (not GET) so the password
    rides the body, never a URL/log. The post-install card uses the
    exempt ``/console-credentials`` reveal instead (the user just
    authenticated to install it).

    The managed credential is deterministic (re-derived from the install
    secret, or an explicit override), so "I lost the password" is solved by
    simply asking again. ``managed`` is False for manually-added servers we
    didn't provision (no front door, no managed login).
    """
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    step = await _verify_step_up(request, body.augmentum_password)
    if step is not None:
        return step
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)

    mgr = getattr(request.app.state, "service_manager", None)
    sd = mgr.get_definition(server.provider) if mgr else None
    from augmentum.providers.service_auth import (
        has_managed_credentials,
        resolve_managed_credentials,
    )
    # Row-level, not provider-level: a manually-connected external server
    # of a catalog provider is NOT managed — advertising the managed
    # instance's ports/login against it would be wrong (and, rendered by
    # the UI against location.hostname, would show a wrong address).
    managed = sd is not None and _is_managed_instance(server)
    out: dict = {
        "provider": server.provider,
        "managed": managed,
        "https_port": int(getattr(sd, "https_port", 0) or 0) if managed else 0,
        "raw_host_port": int(getattr(sd, "host_port", 0) or 0) if managed else 0,
        "managed_auth": False,
        "can_change_password": False,
    }
    if managed and has_managed_credentials(sd):
        username, password = await resolve_managed_credentials(
            server.provider, _db_conn(request),
        )
        out.update({
            "managed_auth": True,
            "username": username,
            "password": password,
            "can_change_password": True,
        })
    return JSONResponse(out)


@router.post("/servers/{server_id}/change-password")
async def media_change_password(
    server_id: str, body: ChangePasswordRequest, request: Request,
) -> JSONResponse:
    """Set a custom password for a provisioned media server's managed login.

    Admin-only. Persists an encrypted override, pushes the change to the
    running server (or recreates it, for Suwayomi's env-baked auth), then
    refreshes the stored token on EVERY user's row for that provider so a
    shared connection doesn't break for other users.
    """
    uid = _user_id(request)
    store = _get_store(request)
    http = _http(request)
    if not uid or not store or http is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    step = await _verify_step_up(request, body.augmentum_password)
    if step is not None:
        return step
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)

    mgr = getattr(request.app.state, "service_manager", None)
    sd = mgr.get_definition(server.provider) if mgr else None
    from augmentum.providers.service_auth import (
        has_managed_credentials,
        needs_managed_auth,
        resolve_managed_credentials,
        set_credential_override,
    )
    if sd is None or not has_managed_credentials(sd) or not _is_managed_instance(server):
        return JSONResponse(
            {"error": "This server's login isn't managed by Augmentum"},
            status_code=400,
        )

    db = _db_conn(request)
    username, current_pw = await resolve_managed_credentials(server.provider, db)
    new_pw = body.new_password
    restarting = False
    new_token = ""
    try:
        if needs_managed_auth(sd):
            # Env-baked auth (Suwayomi): persist override, then recreate the
            # container so the entrypoint rewrites the password. The token is
            # base64(user:pass), recomputed from the new credential.
            if not await set_credential_override(server.provider, new_pw, db):
                return JSONResponse(
                    {"error": "Could not persist the new password"}, status_code=500,
                )
            if mgr is not None:
                await mgr.recreate_with_new_credential(server.provider)
                restarting = True
            import base64 as _b64
            new_token = _b64.b64encode(f"{username}:{new_pw}".encode()).decode("ascii")
        else:
            # First-run-wizard servers (Jellyfin/ABS/Komga): change via the
            # server's own API, then persist the override.
            client = _provider_client(server.provider, http)
            if not hasattr(client, "change_password"):
                return JSONResponse(
                    {"error": "This server doesn't support changing the password"},
                    status_code=400,
                )
            new_token = await client.change_password(
                server.base_url, username, current_pw, new_pw,
            )
            await set_credential_override(server.provider, new_pw, db)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001 — surface a clean failure
        log.warning(
            "media_change_password_failed",
            provider=server.provider, error=str(exc),
        )
        return JSONResponse(
            {"error": f"Couldn’t change the password: {exc}"}, status_code=502,
        )

    # Refresh the token on every connected user's row for this INSTANCE
    # (same provider + same base_url). Manually-added external servers of
    # the same provider have their own credentials — never overwrite them.
    if new_token:
        try:
            await store.update_token_for_provider(
                server.provider, new_token, base_url=server.base_url,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "media_token_refresh_after_change_failed",
                provider=server.provider, error=str(exc),
            )

    return JSONResponse({"ok": True, "restarting": restarting})


@router.post("/servers")
async def add_server(body: AddServerRequest, request: Request) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    http = _http(request)
    if not uid or not store or http is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    # Sharing scope. Non-admins can only create 'private' rows.
    requested_scope = (body.scope or "private").strip().lower()
    if requested_scope not in ("private", "shared"):
        return JSONResponse(
            {"error": "scope must be 'private' or 'shared'"}, status_code=400,
        )
    if requested_scope == "shared" and not is_admin(request):
        return JSONResponse(
            {"error": "Only admins can share media servers with all users"},
            status_code=403,
        )

    # SSRF gate. Media servers legitimately live on LAN/loopback (Plex on
    # 192.168.x.x, Jellyfin on localhost in Docker, etc.) so we use the
    # `lan_ok` mode — it still blocks 169.254/16 (cloud metadata) and
    # multicast/broadcast/reserved ranges, both of which a "Plex server"
    # has no business pointing at.
    from augmentum.utils.safe_http import SafeHttpError, check_ssrf_user_url
    try:
        await check_ssrf_user_url(body.base_url, mode="lan_ok")
    except SafeHttpError as exc:
        return JSONResponse({"error": f"Invalid base_url: {exc}"}, status_code=400)

    # Providers that run with auth disabled by default (Suwayomi
    # local-only deployments are the canonical case) — the provider's
    # own login() validates the deployment instead of rejecting empty
    # credentials up front. Adding to this set is a deliberate act:
    # other providers always require a token or user+pass pair.
    _NO_AUTH_OK = {"suwayomi"}
    allow_no_auth = body.provider in _NO_AUTH_OK

    # Exchange user/pass for a token up front; refuse to save credentials
    # without a round-trip verifying they actually work. This catches
    # typos, wrong server URL, and mid-migration auth changes.
    token = (body.access_token or "").strip()
    if not token:
        if not allow_no_auth and (not body.username or not body.password):
            return JSONResponse(
                {"error": "Provide either an access token or username + password"},
                status_code=400,
            )
        try:
            client = _provider_client(body.provider, http)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            # For no-auth providers with empty creds, login() runs ping()
            # and returns "" — the stored token is empty and every
            # subsequent call sends no Authorization header.
            token = await client.login(
                body.base_url,
                body.username or "",
                body.password or "",
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except Exception as exc:
            log.warning("media_login_failed", provider=body.provider, error=str(exc))
            return JSONResponse(
                {"error": f"Login failed: {exc}"}, status_code=502,
            )

    try:
        server = await store.create(
            user_id=uid, provider=body.provider, name=body.name,
            base_url=body.base_url, access_token=token,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        # Likely a UNIQUE constraint collision (same URL already added).
        log.warning("media_server_create_failed", error=str(exc))
        return JSONResponse(
            {"error": "Server already exists for this URL"}, status_code=409,
        )

    # Apply scope after create. Admin-gated above; set_scope itself
    # enforces ownership so we can't accidentally toggle a row that
    # somehow leaked in from another caller.
    if requested_scope == "shared":
        flipped = await store.set_scope(
            server.id, scope="shared", owner_user_id=uid,
        )
        if flipped is not None:
            server = flipped

    # Auto-index the catalog so the freshly-added server is searchable + playable
    # right away — without this it sits at 0 items until a manual Sync, and
    # "play <title>" finds nothing.
    try:
        from augmentum.media.sync import enqueue_media_sync
        await enqueue_media_sync(request.app.state, user_id=uid, server_id=server.id)
    except Exception:
        log.warning("media_auto_sync_enqueue_failed", server_id=server.id, exc_info=True)

    return JSONResponse(
        {"server": server.to_dict(viewer_user_id=uid)}, status_code=201,
    )


@router.put("/servers/{server_id}")
async def update_server(
    server_id: str, body: UpdateServerRequest, request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    # Two ownership tiers:
    #   - Caller owns the row → can edit metadata, credentials, scope.
    #   - Caller doesn't own it but is admin AND the row is shared →
    #     can only flip scope back to 'private' (un-share). Metadata
    #     edits stay with the owner so we never overwrite their token
    #     without consent.
    # Anyone else gets 404 (not 403) so we don't leak existence.
    existing = await store.get(server_id, user_id=uid)
    owned_by_caller = existing is not None
    if not owned_by_caller:
        existing = await store.get_visible(server_id, user_id=uid)
        if not existing:
            return JSONResponse({"error": "Not found"}, status_code=404)

    caller_is_admin = is_admin(request)

    # Scope toggle. Admin-only AND must touch a row the admin owns —
    # set_scope re-checks ownership at the SQL layer to make this
    # belt-and-braces.
    if body.scope is not None:
        new_scope = body.scope.strip().lower()
        if new_scope not in ("private", "shared"):
            return JSONResponse(
                {"error": "scope must be 'private' or 'shared'"}, status_code=400,
            )
        if not caller_is_admin:
            return JSONResponse(
                {"error": "Only admins can change media-server sharing scope"},
                status_code=403,
            )
        if not owned_by_caller:
            # Even admins can't share another user's row — that would
            # leak someone else's token to everyone on the box. They
            # must re-add it under their own account.
            return JSONResponse(
                {
                    "error": (
                        "Only the user who connected this server can "
                        "change its sharing scope. Re-add it under your "
                        "account to share."
                    ),
                },
                status_code=403,
            )
        previous_scope = (existing.scope or "private") if existing else "private"
        try:
            await store.set_scope(
                server_id, scope=new_scope, owner_user_id=uid,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # set_scope flips one column and nothing else. Without this,
        # un-sharing leaves every borrower's file_index rows behind as
        # ghosts: still listed as playable cards, 502 on every stream
        # because get_visible now refuses them. Re-sharing is the exact
        # inverse — restore what we tombstoned, progress intact.
        if new_scope != previous_scope:
            await _enqueue_server_detach(
                request,
                server_id=server_id,
                owner_user_id=uid,
                action="detach" if new_scope == "private" else "reattach",
            )

    # Metadata / credential changes require ownership — non-owners
    # only get the scope toggle above (and only if admin + own).
    metadata_changing = (
        body.name is not None
        or body.base_url is not None
        or body.access_token is not None
        or (body.username and body.password)
    )
    if metadata_changing and not owned_by_caller:
        return JSONResponse(
            {
                "error": (
                    "Only the server's owner can edit its credentials "
                    "or URL. Sharing exposes the connection read-only."
                ),
            },
            status_code=403,
        )

    # SSRF gate — same lan_ok policy as add_server. Only run when the
    # caller is actually changing base_url (None means "leave as-is").
    if body.base_url:
        from augmentum.utils.safe_http import SafeHttpError, check_ssrf_user_url
        try:
            await check_ssrf_user_url(body.base_url, mode="lan_ok")
        except SafeHttpError as exc:
            return JSONResponse({"error": f"Invalid base_url: {exc}"}, status_code=400)

    new_token = body.access_token
    # If creds were swapped (new user/pass), re-exchange them for a token.
    if new_token is None and body.username and body.password:
        http = _http(request)
        if http is None:
            return JSONResponse({"error": "Service unavailable"}, status_code=503)
        try:
            client = _provider_client(existing.provider, http)
            new_token = await client.login(
                body.base_url or existing.base_url, body.username, body.password,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except Exception as exc:
            return JSONResponse({"error": f"Login failed: {exc}"}, status_code=502)

    # Only call update when there's actually something to write — when
    # the request was scope-only, store.update would no-op the metadata
    # fields anyway, but we still want the freshest scope reflected in
    # the response.
    if metadata_changing or new_token is not None:
        server = await store.update(
            server_id, user_id=uid, name=body.name, base_url=body.base_url,
            access_token=new_token,
        )
    else:
        server = await store.get_visible(server_id, user_id=uid)
    return JSONResponse(
        {"server": server.to_dict(viewer_user_id=uid) if server else None}
    )


@router.delete("/servers/{server_id}")
async def delete_server(server_id: str, request: Request) -> JSONResponse:
    """Unregister a media server and cascade-purge its cached library data.

    Without the cascade, file_index rows for the server's content stay
    behind as orphans (no FK constraint reaches them — `server_id` lives
    inside the JSON `source_metadata` blob). The orphans then 502 every
    time the user opens what looks like a normal chapter, because the
    streaming proxy can't resolve credentials for a server that no
    longer exists. `purge_server_data` removes those rows BEFORE the
    server registration is dropped, so the cleanup can find what to
    purge by server_id.

    Progress data isn't lost — Suwayomi, Komga, Audiobookshelf, Emby all
    track is_finished + last_position upstream. Re-adding the server and
    re-syncing rebuilds the cache with canonical state restored.
    """
    uid = _user_id(request)
    store = _get_store(request)
    idx = _get_index(request)
    if not uid or not store or not idx:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    # Refuse to delete an admin-shared row from a non-owner — the row's
    # owner (admin) still wants it published. Without this guard the
    # caller silently "succeeds" (store.delete filters by user_id and
    # returns 0 rows) while their own file_index catalog gets purged,
    # which is misleading.
    visible = await store.get_visible(server_id, user_id=uid)
    if visible and visible.user_id != uid:
        return JSONResponse(
            {
                "error": (
                    "This server was shared by an admin. Only the admin "
                    "who connected it can remove it."
                ),
            },
            status_code=403,
        )
    # Cascade FIRST — once the server row is gone, we can't find the
    # orphan file_index rows by server_id any longer (well, we can, but
    # the route would have to enumerate them and that's the script's job,
    # not the route's).
    # Deleting a SHARED server orphans every borrower exactly the way an
    # un-share does — purge_server_data is scoped to `user_id = caller`,
    # so it only cleans the owner's own rows. media_library_views does
    # FK-cascade here, which without this leaves borrowers in a
    # half-torn-down state: no library shelves, but the items still
    # listed and still 502ing. Tombstone theirs too.
    was_shared = bool(visible and (visible.scope or "") == "shared")
    purged = await purge_server_data(idx._db, server_id, user_id=uid)
    ok = await store.delete(server_id, user_id=uid)
    if was_shared:
        await _enqueue_server_detach(
            request, server_id=server_id, owner_user_id=uid, action="detach",
        )
    if not ok:
        # Server row was already gone (race or double-click). The cascade
        # already ran against any orphans, so the user's library is
        # consistent — don't surface this as an error.
        return JSONResponse({
            "status": "deleted",
            "purged": purged,
            "server_already_missing": True,
        })
    return JSONResponse({"status": "deleted", "purged": purged})


@router.post("/servers/{server_id}/test")
async def test_server(server_id: str, request: Request) -> JSONResponse:
    """Verify the stored token still authenticates against the server."""
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)
    http = _http(request)
    if http is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    try:
        client = _provider_client(server.provider, http)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Managed servers self-heal an empty/stale token from the derived cred.
    server = await _heal_managed_token(request, server, uid)
    ok = await client.verify_token(server.base_url, server.access_token)
    status = "ok" if ok else "error"
    detail = "" if ok else "Token no longer valid — re-enter credentials"
    await store.update(
        server_id, user_id=uid, status=status, status_detail=detail,
    )
    return JSONResponse({"status": status, "detail": detail})


@router.post("/servers/{server_id}/sync")
async def sync_server_route(server_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    if not uid or not store:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)
    http = _http(request)
    if http is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    # Managed servers self-heal an empty/stale token before sync (the bg job
    # reads the row from the DB, so persisting it here covers both paths).
    server = await _heal_managed_token(request, server, uid)

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is not None and job_runner is not None:
        try:
            existing_jobs = await jobs_store.list_for_user(
                user_id=uid, job_type="media_sync", limit=50,
            )
        except Exception:
            log.warning(
                "media_sync_existing_job_lookup_failed",
                server_id=server_id, user_id=uid, exc_info=True,
            )
            existing_jobs = []

        for job in existing_jobs:
            payload = job.get("payload") or {}
            if (
                job.get("status") in {"pending", "running"}
                and str(payload.get("server_id") or "") == server_id
            ):
                await store.update(
                    server_id,
                    user_id=uid,
                    status="syncing",
                    status_detail=str(job.get("stage") or "Syncing…"),
                )
                return JSONResponse(
                    {
                        "status": "queued",
                        "job_id": job["id"],
                        "server_id": server_id,
                    },
                    status_code=202,
                )

        try:
            await store.update(
                server_id,
                user_id=uid,
                status="syncing",
                status_detail="Queued sync",
            )
            job_id = await jobs_store.create(
                user_id=uid,
                job_type="media_sync",
                payload={"server_id": server_id},
                priority=20,
                max_attempts=2,
            )
            job_runner.wake()
        except Exception as exc:
            log.warning(
                "media_sync_enqueue_failed",
                server_id=server_id, user_id=uid, error=str(exc), exc_info=True,
            )
            await store.update(
                server_id,
                user_id=uid,
                status="error",
                status_detail=f"Sync enqueue failed: {exc}",
            )
            return JSONResponse(
                {"status": "error", "error": f"Sync enqueue failed: {exc}", "indexed": 0},
                status_code=502,
            )
        return JSONResponse(
            {"status": "queued", "job_id": job_id, "server_id": server_id},
            status_code=202,
        )

    await store.update(
        server_id, user_id=uid, status="syncing", status_detail="Syncing…",
    )
    indexed, err = await sync_server(server, store=store, http_client=http)
    if err:
        return JSONResponse({"status": "error", "error": err, "indexed": indexed},
                            status_code=502)
    return JSONResponse({"status": "ok", "indexed": indexed})


@router.get("/servers/{server_id}/libraries")
async def list_server_libraries(server_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    library_store = _get_library_store()
    if not uid or not store or library_store is None:
        return JSONResponse({"libraries": []})
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)
    rows = await library_store.list_for_server(user_id=uid, server_id=server_id)
    return JSONResponse({"libraries": [row.to_dict() for row in rows]})


@router.patch("/servers/{server_id}/libraries/{library_id}")
async def update_server_library(
    server_id: str,
    library_id: str,
    body: UpdateLibraryRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    library_store = _get_library_store()
    if not uid or not store or library_store is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Not found"}, status_code=404)
    updated = await library_store.update(
        library_id,
        user_id=uid,
        display_name_override=body.display_name_override,
        surface_group_override=body.surface_group_override,
        is_hidden=body.is_hidden,
        include_in_search=body.include_in_search,
        include_in_overview=body.include_in_overview,
        sort_order=body.sort_order,
    )
    if not updated or updated.server_id != server_id:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"library": updated.to_dict()})


# --- Detection -----------------------------------------------------------


@router.post("/detect")
async def detect(request: Request) -> JSONResponse:
    """Silent probe of default ports on host.docker.internal + 127.0.0.1.

    Returns every confirmed server along with whether the user has
    already added it (so the UI can skip re-offering known servers).
    """
    uid = _user_id(request)
    store = _get_store(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    http = _http(request)
    if http is None:
        return JSONResponse({"detected": []})

    infos = await detect_servers(http)
    result: list[dict[str, Any]] = []
    for info in infos:
        already_added = False
        if store:
            existing = await store.find_match(
                user_id=uid, provider=info.provider, base_url=info.base_url,
            )
            already_added = existing is not None
        result.append({
            "provider":       info.provider,
            "base_url":       info.base_url,
            "is_initialized": info.is_initialized,
            "server_name":    info.server_name,
            "already_added":  already_added,
        })
    return JSONResponse({"detected": result})


# --- Resume listening (discovery integration) ---------------------------


@router.get("/resume-listening")
async def resume_listening(request: Request) -> JSONResponse:
    """In-progress audiobooks, most-recently-played first. Up to 10 items.

    Source-agnostic — any ``kind='audio'`` row with ``progress_pct > 0``
    and ``is_finished == 0`` qualifies, so Audiobookshelf and LibriVox
    (and future Emby/Jellyfin) all surface on the same strip.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"items": []})

    in_progress = await idx.list_recent(
        user_id=uid,
        kind="audio",
        media_status="in_progress",
        sort="progress",
        limit=10,
    )

    items: list[dict] = []
    for e in in_progress:
        meta = e.source_metadata if isinstance(e.source_metadata, dict) else {}
        items.append({
            "file_id":         e.id,
            "title":           e.name,
            "author":          meta.get("author") or "",
            "cover_url":       f"/api/media/cover/{e.id}",
            "progress_pct":    float(meta.get("progress_pct") or 0.0),
            "duration_s":      float(meta.get("duration_s") or 0.0),
            "current_time_s":  float(meta.get("current_time_s") or 0.0),
            "last_update":     meta.get("last_update") or e.updated_at or "",
        })

    return JSONResponse({"items": items})


# --- Remote outputs ------------------------------------------------------


@router.get("/receiver-profiles")
async def media_receiver_profiles() -> JSONResponse:
    """Static receiver-profile catalog used by sender UIs.

    These are transport expectations, not live discovered devices. Device
    discovery remains client-side or sender-specific.
    """
    return JSONResponse({
        "profiles": [profile.to_dict() for profile in list_receiver_profiles()],
    })


@router.get("/outputs/{file_id}")
async def media_outputs(file_id: str, request: Request) -> JSONResponse:
    """Provider-native remote playback targets for a file.

    Browser-level output targets (AirPlay/Remote Playback/Cast) are detected
    client-side because they depend on the current browser and active media
    element. This endpoint only reports provider-managed clients, which lets
    the UI offer "Play on Emby/Jellyfin device" without embedding provider
    auth logic in the browser.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({
            "supports_provider_remote": False,
            "remote_sessions": [],
            "transport_receivers": [],
        })

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = str(meta.get("server_id") or "").strip()
    external_id = str(meta.get("external_id") or "").strip()
    entity_kind = str(meta.get("entity_kind") or "").strip()
    response = {
        "supports_provider_remote": False,
        "provider": "",
        "server_id": server_id,
        "entity_kind": entity_kind,
        "remote_sessions": [],
        "transport_receivers": [],
    }
    if not server_id or not external_id or _is_builtin_server(server_id):
        return JSONResponse(response)
    if entity_kind not in {"movie", "episode", "music_video"}:
        return JSONResponse(response)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    response["provider"] = server.provider

    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_control(client):
        return JSONResponse(response)

    sessions = await client.list_remote_sessions(
        server.base_url,
        server.access_token,
        media_type="Video",
    )
    response["supports_provider_remote"] = True
    response["remote_sessions"] = [session.to_dict() for session in sessions]
    try:
        receivers = await discover_dlna_receivers(http_client)
    except Exception as exc:
        log.debug("dlna_receiver_discovery_failed", error=str(exc))
        receivers = []
    if receivers:
        _receiver_runtime(request).remember_receivers(receivers)
        response["transport_receivers"] = [receiver.to_dict() for receiver in receivers]
    return JSONResponse(response)


@router.get("/outputs/{file_id}/launch-plan")
async def media_output_launch_plan(
    file_id: str,
    request: Request,
    receiver_profile: str = "cast_video",
) -> JSONResponse:
    """Return a provider-prepared receiver launch blueprint for one item.

    This is the transport-agnostic seam between provider playback metadata
    (Emby/Jellyfin) and future sender stacks (Cast, DLNA, etc.). It does not
    start playback; it only returns a receiver-specific plan.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({
            "supported": False,
            "receiver_profile": str(receiver_profile or "").strip(),
            "reason": "Authentication required",
        })

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = str(meta.get("server_id") or "").strip()
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({
            "supported": False,
            "receiver_profile": str(receiver_profile or "").strip(),
            "reason": "Receiver launch is not available for this source",
        })

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    plan = await build_receiver_launch_plan(
        server=server,
        client=client,
        file_id=file_id,
        entry_name=str(entry.name or "").strip() or "Video",
        cached_meta=meta,
        receiver_profile_id=str(receiver_profile or "").strip() or "cast_video",
    )
    return JSONResponse(plan.to_dict())


@router.post("/outputs/{file_id}/transport-play")
async def media_output_transport_play(
    file_id: str,
    body: TransportPlayRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = str(meta.get("server_id") or "").strip()
    external_id = str(meta.get("external_id") or "").strip()
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Transport handoff unavailable"}, status_code=400)

    transport = str(body.transport or "").strip().lower()
    if transport != "dlna":
        return JSONResponse({"error": "Unsupported transport"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    runtime = _receiver_runtime(request)
    receiver = runtime.get_receiver(body.receiver_id)
    if receiver is None:
        discovered = await discover_dlna_receivers(http_client)
        runtime.remember_receivers(discovered)
        receiver = runtime.get_receiver(body.receiver_id)
    if receiver is None:
        return JSONResponse({"error": "Receiver unavailable"}, status_code=404)

    plan = await build_receiver_launch_plan(
        server=server,
        client=client,
        file_id=file_id,
        entry_name=str(entry.name or "").strip() or "Video",
        cached_meta=meta,
        receiver_profile_id=str(
            body.receiver_profile or receiver.receiver_profile or "dlna_generic_video"
        ).strip() or "dlna_generic_video",
    )
    if not plan.supported:
        return JSONResponse(plan.to_dict(), status_code=400)

    ok = await launch_dlna_receiver(http_client, receiver, plan)
    if not ok:
        return JSONResponse({"error": "DLNA launch failed"}, status_code=502)

    session = TransportSession(
        user_id=uid,
        transport_kind="dlna",
        receiver_id=receiver.receiver_id,
        receiver_label=receiver.label,
        receiver_profile=receiver.receiver_profile,
        provider=server.provider,
        server_id=server.id,
        file_id=file_id,
        external_id=external_id,
        title=plan.title,
        thumbnail=plan.poster_url,
        supported_commands=list(receiver.supported_commands or []),
        receiver=receiver,
        extra={
            "launch_plan": plan.to_dict(),
        },
    )
    session.update_from_snapshot(await snapshot_dlna_receiver(http_client, receiver))
    runtime.put_session(session)
    return JSONResponse({
        "status": "ok",
        "transport": "dlna",
        "session": session.to_dict(),
    })


@router.post("/outputs/{file_id}/remote-play")
async def media_output_remote_play(
    file_id: str,
    body: RemoteSessionPlayRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = str(meta.get("server_id") or "").strip()
    external_id = str(meta.get("external_id") or "").strip()
    entity_kind = str(meta.get("entity_kind") or "").strip()
    if not server_id or not external_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Provider handoff unavailable"}, status_code=400)
    if entity_kind not in {"movie", "episode", "music_video"}:
        return JSONResponse({"error": "Item is not directly playable"}, status_code=400)

    play_command = str(body.play_command or "PlayNow").strip() or "PlayNow"
    if play_command not in _REMOTE_PLAY_COMMANDS:
        return JSONResponse({"error": "Invalid play command"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_control(client):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    media_source_id = (
        str(body.media_source_id or "").strip()
        or str(meta.get("preferred_media_source_id") or "").strip()
    )
    audio_idx = body.audio_stream_index
    if audio_idx is None:
        audio_idx = _int_or_none(meta.get("preferred_audio_stream_index"))
    subtitle_idx = body.subtitle_stream_index
    if subtitle_idx is None:
        subtitle_idx = _int_or_none(meta.get("preferred_subtitle_stream_index"))

    ok = await client.remote_play(
        server.base_url,
        server.access_token,
        session_id=body.session_id,
        external_id=external_id,
        start_time_s=float(body.start_time_s or 0.0),
        play_command=play_command,
        media_source_id=media_source_id,
        audio_stream_index=audio_idx,
        subtitle_stream_index=subtitle_idx,
    )
    if not ok:
        return JSONResponse({"error": "Remote play failed"}, status_code=502)
    return JSONResponse({
        "status": "ok",
        "provider": server.provider,
        "server_id": server.id,
        "session_id": body.session_id,
        "external_id": external_id,
    })


@router.post("/outputs/{file_id}/remote-command")
async def media_output_remote_command(
    file_id: str,
    body: RemoteSessionCommandRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = str(meta.get("server_id") or "").strip()
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Provider handoff unavailable"}, status_code=400)
    command = str(body.command or "").strip()
    if command not in _REMOTE_PLAYSTATE_COMMANDS:
        return JSONResponse({"error": "Invalid remote command"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_control(client):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    ok = await client.remote_command(
        server.base_url,
        server.access_token,
        session_id=body.session_id,
        command=command,
        seek_position_s=body.seek_position_s,
    )
    if not ok:
        return JSONResponse({"error": "Remote command failed"}, status_code=502)
    return JSONResponse({"status": "ok"})


@router.get("/transport-sessions/{session_id}")
async def media_transport_session(session_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    http_client = _http(request)
    if not uid or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    session = _receiver_runtime(request).get_session(session_id, user_id=uid)
    if session is None:
        return JSONResponse({"error": "Transport session not found"}, status_code=404)

    if session.transport_kind == "dlna" and session.receiver is not None:
        session.update_from_snapshot(await snapshot_dlna_receiver(http_client, session.receiver))

    return JSONResponse({
        "status": "ok",
        "transport": session.transport_kind,
        "session": session.to_dict(),
    })


@router.post("/transport-sessions/{session_id}/playstate")
async def media_transport_session_playstate(
    session_id: str,
    body: RemoteSessionCommandRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    http_client = _http(request)
    if not uid or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    runtime = _receiver_runtime(request)
    session = runtime.get_session(session_id, user_id=uid)
    if session is None:
        return JSONResponse({"error": "Transport session not found"}, status_code=404)

    ok = False
    if session.transport_kind == "dlna" and session.receiver is not None:
        ok = await send_dlna_playstate_command(
            http_client,
            session.receiver,
            command=body.command,
            seek_position_s=body.seek_position_s,
        )
        if ok and str(body.command or "").strip() == "Stop":
            runtime.remove_session(session_id, user_id=uid)
        elif ok:
            session.update_from_snapshot(await snapshot_dlna_receiver(http_client, session.receiver))
    if not ok:
        return JSONResponse({"error": "Transport command failed"}, status_code=502)
    return JSONResponse({"status": "ok"})


@router.post("/transport-sessions/{session_id}/general")
async def media_transport_session_general(
    session_id: str,
    body: RemoteSessionGeneralCommandRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    http_client = _http(request)
    if not uid or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    session = _receiver_runtime(request).get_session(session_id, user_id=uid)
    if session is None:
        return JSONResponse({"error": "Transport session not found"}, status_code=404)

    ok = False
    if session.transport_kind == "dlna" and session.receiver is not None:
        ok = await send_dlna_general_command(
            http_client,
            session.receiver,
            command=body.command,
            arguments=_remote_argument_map(body.arguments),
        )
        if ok:
            session.update_from_snapshot(await snapshot_dlna_receiver(http_client, session.receiver))
    if not ok:
        return JSONResponse({"error": "Transport command failed"}, status_code=502)
    return JSONResponse({"status": "ok"})


@router.get("/remote-sessions/{server_id}/{session_id}")
async def media_remote_session(
    server_id: str,
    session_id: str,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_control(client):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    sessions = await client.list_remote_sessions(
        server.base_url,
        server.access_token,
        media_type="Video",
    )
    match = next(
        (session for session in sessions if session.session_id == session_id),
        None,
    )
    if match is None:
        return JSONResponse({"error": "Remote session not found"}, status_code=404)

    current_file_id = ""
    current_file_name = ""
    current_cover_url = ""
    if idx is not None and match.now_playing_item_id:
        entry = await _file_entry_for_remote_item(
            idx,
            user_id=uid,
            provider=server.provider,
            server_id=server.id,
            external_id=match.now_playing_item_id,
        )
        if entry:
            current_file_id = entry.id
            current_file_name = entry.name or ""
            has_cover = False
            if isinstance(entry.source_metadata, dict):
                has_cover = bool(entry.source_metadata.get("has_cover") or False)
            if has_cover or entry.thumbnail:
                current_cover_url = f"/api/media/cover/{entry.id}"

    return JSONResponse(_remote_session_payload(
        session=match,
        provider=server.provider,
        server_id=server.id,
        current_file_id=current_file_id,
        current_file_name=current_file_name,
        current_cover_url=current_cover_url,
    ))


@router.post("/remote-sessions/{server_id}/{session_id}/playstate")
async def media_remote_session_playstate(
    server_id: str,
    session_id: str,
    body: RemoteSessionCommandRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)
    if str(body.session_id or "").strip() and str(body.session_id).strip() != session_id:
        return JSONResponse({"error": "Session mismatch"}, status_code=400)
    command = str(body.command or "").strip()
    if command not in _REMOTE_PLAYSTATE_COMMANDS:
        return JSONResponse({"error": "Invalid remote command"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_control(client):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    ok = await client.remote_command(
        server.base_url,
        server.access_token,
        session_id=session_id,
        command=command,
        seek_position_s=body.seek_position_s,
    )
    if not ok:
        return JSONResponse({"error": "Remote command failed"}, status_code=502)
    return JSONResponse({"status": "ok"})


@router.post("/remote-sessions/{server_id}/{session_id}/general")
async def media_remote_session_general(
    server_id: str,
    session_id: str,
    body: RemoteSessionGeneralCommandRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not store or http_client is None:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)
    command = str(body.command or "").strip()
    if command not in _REMOTE_GENERAL_COMMANDS:
        return JSONResponse({"error": "Invalid remote command"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)
    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not provider_supports_remote_general_control(client):
        return JSONResponse({"error": "Provider remote control unavailable"}, status_code=400)

    ok = await client.remote_general_command(
        server.base_url,
        server.access_token,
        session_id=session_id,
        command=command,
        arguments=_remote_argument_map(body.arguments),
    )
    if not ok:
        return JSONResponse({"error": "Remote command failed"}, status_code=502)
    return JSONResponse({"status": "ok"})


@router.get("/outputs/{file_id}/cast-load")
async def media_output_cast_load(file_id: str, request: Request) -> JSONResponse:
    """Compatibility wrapper around the generic receiver launch-plan route."""
    return await media_output_launch_plan(
        file_id,
        request,
        receiver_profile="cast_video",
    )


# --- Streaming proxy -----------------------------------------------------


@router.get("/stream/{file_id}")
async def stream_media(file_id: str, request: Request):
    """Authenticated, Range-forwarding proxy.

    Range / If-Range / If-Modified-Since headers are forwarded so
    ``<audio>`` / ``<video>`` seeking works. Response headers include
    Content-Range, Accept-Ranges, and Content-Length when upstream
    provides them. Upstream status codes pass through — a 206 Partial
    Content stays a 206; a 404 stays a 404.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    if not uid or not idx or not store:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    # FileEntry.source_metadata is already a dict (decoded by _row_to_entry).
    # Guard against the legacy call shape where it might arrive as a JSON
    # string so the endpoint stays robust to callsite drift.
    raw_meta = entry.source_metadata
    if isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        try:
            meta = json.loads(raw_meta or "{}")
        except Exception:
            meta = {}
    server_id = meta.get("server_id", "")
    if not server_id:
        return JSONResponse({"error": "Not a streamable entry"}, status_code=400)

    http_client = _http(request)
    if http_client is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    # Built-in LibriVox path: multi-file books mean the stream route
    # needs ?file=<index> to pick a chapter MP3. archive.org identifier
    # + filename in meta is all we need — no user_media_servers row exists.
    if _is_builtin_server(server_id):
        archive_id = meta.get("archive_identifier", "")
        audio_files = meta.get("audio_files") or []
        if not archive_id or not audio_files:
            return JSONResponse(
                {"error": "LibriVox entry missing archive metadata"},
                status_code=400,
            )
        try:
            file_idx = int(request.query_params.get("file", "0"))
        except ValueError:
            return JSONResponse({"error": "Invalid file index"}, status_code=400)
        if file_idx < 0 or file_idx >= len(audio_files):
            return JSONResponse({"error": "File index out of range"}, status_code=404)
        filename = audio_files[file_idx].get("name") or ""
        if not filename:
            return JSONResponse({"error": "Missing filename"}, status_code=500)
        provider_name = _builtin_provider_name(server_id)
        try:
            client = _provider_client(provider_name, http_client)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        upstream_url = client.build_stream_url(
            "", f"{archive_id}/{filename}", "",
        )
    else:
        server = await store.get_visible(server_id, user_id=uid)
        if not server:
            return JSONResponse({"error": "Server unavailable"}, status_code=502)
        try:
            client = _provider_client(server.provider, http_client)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        episode_id = str(request.query_params.get("episode_id") or "").strip()
        if server.provider == "audiobookshelf" and episode_id:
            if str(meta.get("selected_episode_id") or "").strip() == episode_id:
                stream_path = str(meta.get("selected_episode_stream_path") or "").strip()
            else:
                stream_path = ""
            if not stream_path:
                external_id = str(meta.get("external_id") or "").strip()
                if not external_id:
                    return JSONResponse(
                        {"error": "Not a streamable entry"}, status_code=400,
                    )
                raw = await client.fetch_item_details(
                    server.base_url,
                    server.access_token,
                    external_id=external_id,
                    episode_id=episode_id,
                )
                if raw is None:
                    return JSONResponse(
                        {"error": "Podcast episode unavailable"},
                        status_code=404,
                    )
                stream_path = client.episode_stream_path(raw, episode_id=episode_id)
            if not stream_path:
                return JSONResponse(
                    {"error": "Podcast episode has no playable audio"},
                    status_code=404,
                )
        else:
            stream_path = meta.get("stream_path", "")
            if not stream_path:
                return JSONResponse({"error": "Not a streamable entry"}, status_code=400)
        stream_choice = _stream_choice_from_meta(meta, request)
        start_time_s = _float_or_none(request.query_params.get("start_time_s"))
        upstream_url = client.build_stream_url(
            server.base_url, stream_path, server.access_token,
        )
        if server.provider in {"emby", "jellyfin"}:
            browser_safe_url = await _emby_compat_browser_stream_url(
                client=client,
                server=server,
                external_id=str(meta.get("external_id") or "").strip(),
                cached_meta=meta,
                stream_choice=stream_choice,
                start_time_s=start_time_s,
            )
            if browser_safe_url:
                upstream_url = browser_safe_url
            else:
                upstream_url = _append_query_params(
                    upstream_url,
                    stream_choice,
                )

    # Forward only the headers that affect byte-range negotiation. We
    # explicitly do NOT forward Cookie, Authorization, or custom headers —
    # the upstream server auth is the token we bake into the URL.
    forward = {}
    for h in ("range", "if-range", "if-modified-since", "if-none-match"):
        v = request.headers.get(h)
        if v is not None:
            forward[h.capitalize() if h != "range" else "Range"] = v

    # Open the upstream stream BEFORE constructing StreamingResponse so we
    # can pass through the real status code (206 for Range requests) and
    # the Content-Range / Accept-Ranges / Content-Length headers. Without
    # these, browsers treat a 206 body wrapped in our 200 envelope as an
    # invalid seek response and silently reload from byte 0 — "skip-30
    # restarts the book" class of bug. Audiobookshelf and LibriVox both
    # depend on this path, so the fix affects every audio source.
    return await _open_range_proxy(http_client, upstream_url, forward)


@router.get("/subtitle/{file_id}")
async def media_subtitle(file_id: str, request: Request):
    """Authenticated subtitle proxy for video text tracks.

    Emby/Jellyfin can expose a subtitle stream as WebVTT even when the
    main video is served as a plain direct stream. The browser only
    renders captions when it sees a real `<track>` source, so this route
    hides the upstream token and surfaces the subtitle bytes same-origin.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    if not uid or not idx or not store:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    raw_meta = entry.source_metadata
    if isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        try:
            meta = json.loads(raw_meta or "{}")
        except Exception:
            meta = {}

    server_id = str(meta.get("server_id") or "").strip()
    external_id = str(meta.get("external_id") or "").strip()
    if not server_id or not external_id:
        return JSONResponse({"error": "Not a streamable entry"}, status_code=400)
    if _is_builtin_server(server_id):
        return JSONResponse({"error": "Subtitles unsupported for this source"}, status_code=400)

    media_source_id, subtitle_idx = _subtitle_choice_from_meta(meta, request)
    if not media_source_id or subtitle_idx is None or subtitle_idx < 0:
        return JSONResponse({"error": "Subtitle selection not available"}, status_code=404)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)

    http_client = _http(request)
    if http_client is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    build_subtitle_url = getattr(client, "build_subtitle_url", None)
    if not callable(build_subtitle_url):
        return JSONResponse({"error": "Subtitles unsupported for this provider"}, status_code=400)

    upstream_url = build_subtitle_url(
        server.base_url,
        external_id=external_id,
        media_source_id=media_source_id,
        subtitle_stream_index=int(subtitle_idx),
        token=server.access_token,
        format="vtt",
    )
    if not upstream_url:
        return JSONResponse({"error": "Subtitle selection not available"}, status_code=404)

    try:
        resp = await http_client.get(
            upstream_url,
            headers={"Accept": "text/vtt,*/*"},
            timeout=20.0,
            follow_redirects=True,
        )
    except Exception as exc:
        log.warning("media_subtitle_proxy_failed", file_id=file_id, error=str(exc))
        return JSONResponse({"error": "Subtitle fetch failed"}, status_code=502)

    content_type = resp.headers.get("content-type") or "text/vtt"
    cache_control = resp.headers.get("cache-control") or "private, max-age=60"
    start_time_s = _float_or_none(request.query_params.get("start_time_s")) or 0.0
    if resp.status_code < 200 or resp.status_code >= 300:
        detail = (resp.text or "").strip()
        body = {"error": detail or "Subtitle unavailable"}
        return JSONResponse(body, status_code=resp.status_code)
    content = resp.content
    if start_time_s > 0 and "vtt" in content_type.lower():
        try:
            shifted = _shift_webvtt(resp.text, start_time_s)
            content = shifted.encode("utf-8")
        except Exception:
            content = resp.content
    return Response(
        content=content,
        status_code=resp.status_code,
        media_type=content_type,
        headers={
            "Cache-Control": cache_control,
        },
    )


@router.get("/comic/page/{file_id}")
async def comic_page(file_id: str, request: Request):
    """Authenticated per-page image proxy for comic items.

    Comic providers (Komga, Suwayomi, future Kavita) deliver pages as
    individual images at their own indexing conventions — Komga is
    1-indexed (``/books/{id}/pages/1``), Suwayomi is 0-indexed
    (``/manga/{m}/chapter/{c}/page/0``). This route normalizes to
    1-indexed externally (``?page=1`` = the first page) and converts
    per-provider internally. It also attaches the provider's auth header
    server-side, so the frontend never sees the Komga Basic token or the
    Suwayomi creds.

    Query flags:
      - ``thumb=1``: ask for a lighter preview image when the provider
        exposes one (Komga page thumbnails).
      - ``quality=raw``: ask for the original page bytes when the
        provider exposes a high-fidelity/raw variant (Komga).

    Body is streamed through as-is, preserving the upstream's
    Content-Type. No byte-range handling — page images are small (tens
    to hundreds of KB) and fit in a single request comfortably.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    if not uid or not idx or not store:
        log.warning(
            "comic_page_guard_failed",
            file_id=file_id,
            has_uid=bool(uid), has_idx=bool(idx), has_store=bool(store),
        )
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        page_1_indexed = int(request.query_params.get("page", "1"))
    except ValueError:
        return JSONResponse({"error": "Invalid page number"}, status_code=400)
    if page_1_indexed < 1:
        return JSONResponse({"error": "Page number is 1-indexed"}, status_code=400)
    want_thumb = request.query_params.get("thumb") in {"1", "true", "yes"}
    want_raw = (request.query_params.get("quality") or "").strip().lower() == "raw"

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        log.warning(
            "comic_page_entry_not_found",
            file_id=file_id, user_id=uid,
        )
        return JSONResponse({"error": "Not found"}, status_code=404)
    if entry.kind != "comic":
        log.warning(
            "comic_page_wrong_kind",
            file_id=file_id, kind=entry.kind, source=entry.source,
        )
        return JSONResponse(
            {"error": "Not a comic entry"}, status_code=400,
        )

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = meta.get("server_id", "")
    provider = meta.get("provider", "")
    external_id = meta.get("external_id", "")
    if not server_id or not provider or not external_id:
        return JSONResponse({"error": "Missing provider metadata"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)

    http_client = _http(request)
    if http_client is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    # Resolve the upstream per-page URL per provider. Each mapping is
    # explicit — no generic protocol method yet; when a third comic
    # provider lands we promote this to ``MediaProvider.build_page_url``.
    if provider == "komga":
        # Komga: /api/v1/books/{book_id}/pages/{N}, 1-indexed. external_id
        # IS the book_id; stream_path encodes the same thing via the
        # ``/file`` URL. We use external_id directly for clarity.
        page_suffix = (
            "/thumbnail" if want_thumb
            else ("/raw" if want_raw else "")
        )
        upstream_url = (
            f"{server.base_url.rstrip('/')}"
            f"/api/v1/books/{external_id}/pages/{page_1_indexed}{page_suffix}"
        )
    elif provider == "suwayomi":
        # Suwayomi external_id is either "{manga_id}.{source_order}" (legacy,
        # pre-GraphQL migration) or "{manga_id}.{source_order}.{chapter_db_id}"
        # (current, v2 GraphQL path). The chapter_db_id is only used by the
        # updateChapter mutation — image delivery just needs the first two.
        # Pages are 0-indexed on the upstream so we convert from our
        # 1-indexed API.
        parts = external_id.split(".")
        if len(parts) < 2:
            return JSONResponse(
                {"error": "Invalid Suwayomi external_id"}, status_code=500,
            )
        manga_id_s, source_order_s = parts[0], parts[1]
        upstream_url = (
            f"{server.base_url.rstrip('/')}"
            f"/api/v1/manga/{manga_id_s}/chapter/{source_order_s}"
            f"/page/{page_1_indexed - 1}"
        )
    else:
        return JSONResponse(
            {"error": f"Provider '{provider}' does not support per-page delivery"},
            status_code=400,
        )

    headers = {}
    if server.access_token:
        # Komga + authenticated Suwayomi → HTTP Basic header.
        headers["Authorization"] = f"Basic {server.access_token}"

    async def _fetch_once():
        async with http_client.stream(
            "GET", upstream_url,
            headers=headers,
            timeout=20.0,
            follow_redirects=True,
        ) as upstream:
            status = upstream.status_code
            content_type = upstream.headers.get("content-type", "application/octet-stream")
            if status < 200 or status >= 400:
                await upstream.aread()
                return (status, content_type, None)
            # Buffer small page images (tens to hundreds of KB) so we
            # can set accurate Content-Length without chunked framing.
            body = await upstream.aread()
            return (status, content_type, body)

    try:
        status, content_type, body = await _fetch_once()

        # Suwayomi-specific retry: its page endpoints 404 on chapters
        # that haven't been prepared yet (pageCount=0 in Suwayomi's DB,
        # source extension never ran to enumerate pages). Call the
        # fetchChapterPages mutation once and retry — idempotent for
        # chapters that were already cached. Only applies to Suwayomi;
        # Komga pages are always pre-fetched as stored .cbz entries.
        if status == 404 and provider == "suwayomi" and body is None:
            meta_extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
            chapter_db_id = meta_extra.get("chapter_db_id")
            if chapter_db_id:
                try:
                    client = _provider_client(provider, http_client)
                    await client.prepare_chapter(
                        server.base_url, server.access_token,
                        chapter_db_id=int(chapter_db_id),
                    )
                except Exception as exc:
                    log.warning(
                        "comic_page_prepare_failed",
                        file_id=file_id, error=str(exc),
                    )
                else:
                    status, content_type, body = await _fetch_once()

        if status < 200 or status >= 400 or body is None:
            return JSONResponse(
                {"error": f"Upstream returned {status}"},
                status_code=status if status in (404, 401, 403, 502, 503) else 502,
            )
    except httpx.RequestError as exc:
        log.warning(
            "comic_page_fetch_failed",
            provider=provider, file_id=file_id, page=page_1_indexed,
            error=str(exc),
        )
        return JSONResponse(
            {"error": "Could not reach provider"}, status_code=502,
        )

    return StreamingResponse(
        iter([body]),
        status_code=200,
        media_type=content_type,
        headers={
            # Comics are immutable once scanned; a generous cache saves
            # a round-trip on every re-read. The reader's own preload
            # layer adds in-memory caching on top.
            "Cache-Control": "private, max-age=3600",
            "Content-Length": str(len(body)),
        },
    )


@router.get("/comic/manifest/{file_id}")
async def comic_manifest(file_id: str, request: Request) -> JSONResponse:
    """Fresh per-chapter metadata for the reader mount path.

    The file_index row carries ``extra.page_count`` / ``current_page`` /
    ``is_finished`` from the last catalog sync. For chapters that were
    never opened (Suwayomi: ``pageCount=0`` until ``fetchChapterPages``
    runs) or that had progress updated out-of-band from another device,
    those values are stale. Calling this endpoint on reader open refreshes
    them so the UI doesn't fire edge-of-chapter arming against a zero
    count or a saved-end position that no longer exists.

    Also writes the refreshed values back into ``source_metadata`` so the
    Files grid's cached chapter row reflects the same state — no full
    re-sync needed to pick up the change.

    Shape is provider-agnostic:
      ``{ "page_count": int, "current_page": int, "is_finished": bool }``
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry or entry.kind != "comic":
        return JSONResponse({"error": "Not a comic entry"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    server_id = meta.get("server_id", "")
    provider = meta.get("provider", "")
    external_id = meta.get("external_id", "")
    if not server_id or not provider or not external_id:
        return JSONResponse({"error": "Missing provider metadata"}, status_code=400)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)

    page_count = int(extra.get("page_count") or 0)
    current_page = int(extra.get("current_page") or 0)
    is_finished = bool(extra.get("is_finished") or False)

    try:
        client = _provider_client(provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if provider == "suwayomi":
        # prepare_chapter is idempotent — it's cheap when the chapter is
        # already cached and populates pageCount when it isn't. Either way
        # the return value is the authoritative current page count. Needs
        # the chapter_db_id (3-part external_id); legacy 2-part rows can't
        # refresh this way, so they'd fall through with cached values.
        chapter_db_id = extra.get("chapter_db_id")
        if chapter_db_id:
            try:
                fresh_count = await client.prepare_chapter(
                    server.base_url, server.access_token,
                    chapter_db_id=int(chapter_db_id),
                )
                if fresh_count > 0:
                    page_count = fresh_count
            except Exception as exc:
                log.warning(
                    "comic_manifest_prepare_failed",
                    file_id=file_id, error=str(exc),
                )
    elif provider == "komga":
        # Komga returns readProgress + pagesCount on the book detail
        # endpoint in one call — no separate prepare step. Stale-safe:
        # if the book was read on another device we pick up the new
        # progress here.
        try:
            raw = await client.fetch_item_details(
                server.base_url, server.access_token, external_id=external_id,
            )
            if isinstance(raw, dict):
                media_block = raw.get("media") or {}
                pc = int(media_block.get("pagesCount") or 0)
                if pc > 0:
                    page_count = pc
                rp = raw.get("readProgress") or {}
                current_page = int(rp.get("page") or current_page)
                is_finished = bool(rp.get("completed") or False)
        except Exception as exc:
            log.warning(
                "comic_manifest_komga_failed",
                file_id=file_id, error=str(exc),
            )
    # Other providers (future Kavita etc.) fall through with the cached
    # values — the endpoint still returns a usable shape so the frontend
    # doesn't need provider-specific branching.

    # Clamp current_page to [0, page_count] — otherwise a stale value
    # past-the-end produces exactly the "open at end, press next = chapter
    # jump" trap the frontend also guards against.
    if page_count > 0 and current_page > page_count:
        current_page = page_count

    # Write the refreshed triple back into source_metadata so the Files
    # grid + chapter list pick up the change without a full re-sync.
    if page_count or current_page or is_finished:
        updated_meta = dict(meta)
        updated_extra = dict(extra)
        if page_count > 0: updated_extra["page_count"] = page_count
        if current_page > 0: updated_extra["current_page"] = current_page
        updated_extra["is_finished"] = is_finished
        updated_meta["extra"] = updated_extra
        try:
            await idx.update_source_metadata(entry.id, updated_meta, user_id=uid)
        except Exception as exc:
            log.warning(
                "comic_manifest_persist_failed",
                file_id=file_id, error=str(exc),
            )

    return JSONResponse({
        "page_count":  page_count,
        "current_page": current_page,
        "is_finished": is_finished,
    })


@router.get("/details/{file_id}")
async def media_details(
    file_id: str,
    request: Request,
    episode_id: str | None = None,
) -> JSONResponse:
    """Rich per-item detail for the detail panel.

    Library listings omit chapters, full author/narrator arrays, and
    descriptions — we fetch those on demand when the user opens a book.
    This endpoint also refreshes the row's progress + current_time in
    source_metadata so "phone listening while Augmentum is open" stays
    in sync without polling.

    Returns a normalized, provider-agnostic shape so future Emby /
    Jellyfin clients can reuse the same UI.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    server_id = meta.get("server_id", "")
    external_id = meta.get("external_id", "")
    if not server_id or not external_id:
        return JSONResponse({"error": "Not a streamable entry"}, status_code=400)

    # LibriVox: pin-time wrote every field we need into source_metadata,
    # so serve the detail panel straight from the cache. Archive.org's
    # metadata is effectively immutable per-identifier — no benefit to a
    # per-open refetch, and it saves a slow upstream call.
    if _is_builtin_server(server_id):
        details = _details_from_meta(entry, meta)
        details["cover_url"] = (
            f"/api/media/cover/{entry.id}" if meta.get("has_cover") else ""
        )
        details["license"] = meta.get("license", "public-domain")
        details["source_provider"] = "librivox"
        return JSONResponse(details)

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)

    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    active_episode_id = ""
    if server.provider == "audiobookshelf":
        active_episode_id = str(episode_id or meta.get("selected_episode_id") or "").strip()

    if server.provider == "audiobookshelf":
        raw = await client.fetch_item_details(
            server.base_url,
            server.access_token,
            external_id=external_id,
            episode_id=active_episode_id,
        )
    else:
        raw = await client.fetch_item_details(
            server.base_url, server.access_token, external_id=external_id,
        )
    if raw is None:
        # Fallback: return what we already have from source_metadata so
        # the UI can still render the detail view without chapters.
        return JSONResponse(_details_from_meta(entry, meta))

    # Normalise the provider shape before the UI sees it so any future
    # Emby/Jellyfin clients can return the same keys.
    if server.provider == "audiobookshelf":
        details = _normalise_abs_details(
            raw,
            cached_meta=meta,
            selected_episode_id=active_episode_id,
        )
    elif server.provider in {"emby", "jellyfin"}:
        details = _normalise_emby_compat_details(
            raw,
            cached_meta=meta,
            provider=server.provider,
        )
    else:
        details = _details_from_meta(entry, meta)
    details["id"] = entry.id
    details["cover_url"] = (
        f"/api/media/cover/{entry.id}" if meta.get("has_cover") else ""
    )
    details["source_provider"] = server.provider
    # series_id lookup — needed by surfaces that want sibling-chapter
    # awareness (cast-comic's auto-advance-to-next-chapter, web reader's
    # _view.chapterCache). Lives on file_index as a real column but
    # ``FileEntry.to_dict()`` doesn't surface it, so we fetch it
    # one-off here. Cheap (PK lookup) and only on detail open, not on
    # every list query.
    try:
        cur = await idx._db.execute(  # noqa: SLF001
            "SELECT series_id FROM file_index WHERE id = ? AND user_id = ?",
            (entry.id, uid),
        )
        row = await cur.fetchone()
        details["series_id"] = (row[0] if row and row[0] else "") or ""
    except Exception:  # noqa: BLE001
        # Defensive — if the schema or row is missing, leave the field
        # blank rather than 500ing the whole details fetch. Cast-comic
        # treats empty series_id as "no sibling awareness available".
        details["series_id"] = ""
    children = details.get("children") or []
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            child_external_id = str(child.get("external_id") or "").strip()
            if not child_external_id:
                continue
            child_entry = await idx.get_by_source(
                server.provider,
                f"{server.id}:{child_external_id}",
                user_id=uid,
            )
            if child_entry:
                child["file_id"] = child_entry.id
                child["playable"] = bool(
                    (child_entry.source_metadata or {}).get("stream_path")
                )

    next_up = details.get("next_up")
    if isinstance(next_up, dict):
        next_up_ext = str(next_up.get("external_id") or "").strip()
        if next_up_ext:
            next_up_entry = await idx.get_by_source(
                server.provider,
                f"{server.id}:{next_up_ext}",
                user_id=uid,
            )
            if next_up_entry:
                next_up["file_id"] = next_up_entry.id
                next_up["playable"] = bool(
                    (next_up_entry.source_metadata or {}).get("stream_path")
                )
        # Drop next_up entirely if we couldn't resolve a playable file —
        # rendering a CTA that does nothing is worse than hiding it.
        if not next_up.get("file_id") or not next_up.get("playable"):
            details["next_up"] = None

    # Refresh the stored metadata so the Files grid + library panel
    # reflect freshly-pulled progress without a full catalog sync.
    updated_meta = dict(meta)
    updated_meta["chapters"] = details["chapters"]

    # Progress-write race protection: when a client just posted to
    # /api/media/progress, the upstream provider takes a beat to
    # persist it. If we re-fetch in that window, ``details`` carries
    # the STALE upstream value (often 0 for a fresh push). Trusting
    # that would overwrite our local mirror with a regressed value
    # and break "click to resume" the moment the user re-casts.
    #
    # Compound heuristic:
    #
    #   (1) Hard invariant — NEVER let upstream's 0 (or near-0) replace
    #       a non-zero local value. Upstream returning 0 after we
    #       wrote 1234 is always a sync hiccup, never a legitimate
    #       reset. The narrow exception is a user explicitly hitting
    #       "Mark Unwatched" — that arrives separately as is_finished
    #       false + a zero push; in that case OUR push already cleared
    #       local first, so this branch is moot.
    #
    #   (2) Fresh-write window (90s) — if our local is younger than
    #       this, prefer local even when upstream returns a NON-ZERO
    #       value (handles upstream lag where it has an older snapshot).
    #
    #   (3) Otherwise upstream is authoritative — that's the
    #       legitimate "user resumed on another device" path.
    _local_pos = float(meta.get("current_time_s") or 0.0)
    _last_write_iso = str(meta.get("last_read_at") or "")
    _local_is_fresh = False
    if _local_pos > 0 and _last_write_iso:
        try:
            last_dt = datetime.fromisoformat(_last_write_iso.replace("Z", "+00:00"))
            now_dt = datetime.now(UTC)
            _local_is_fresh = (now_dt - last_dt).total_seconds() < 90.0
        except Exception:
            _local_is_fresh = False

    _upstream_pos = details.get("current_time_s")
    _upstream_is_zero = (
        _upstream_pos is None
        or (isinstance(_upstream_pos, (int, float)) and float(_upstream_pos) < 1.0)
    )
    # Hard invariant first — covers the "switched movies, returned later"
    # case where the 90s window has lapsed but upstream still returns 0.
    _keep_local = _local_pos > 0 and (_upstream_is_zero or _local_is_fresh)

    if _upstream_pos is not None and not _keep_local:
        updated_meta["current_time_s"] = _upstream_pos
    if details.get("progress_pct") is not None and not _keep_local:
        updated_meta["progress_pct"] = details["progress_pct"]
    if details.get("is_finished") is not None:
        updated_meta["is_finished"] = details["is_finished"]
    if details.get("duration_s"):
        updated_meta["duration_s"] = details["duration_s"]
    # Also overwrite the response payload's progress fields so the
    # CLIENT sees the protected local value, not the regressed
    # upstream one — matters because cast-video reads ``current_time_s``
    # off the response to seek the resume position on each remount.
    if _keep_local:
        details["current_time_s"] = _local_pos
        details["progress_pct"] = float(meta.get("progress_pct") or 0.0)
        log.info(
            "media_progress_local_kept",
            file_id=file_id,
            local_pos=_local_pos,
            upstream_pos=_upstream_pos,
            reason="upstream_zero" if _upstream_is_zero else "fresh_write_window",
        )
    if details.get("narrator"):
        updated_meta["narrator"] = details["narrator"]
    if details.get("description"):
        updated_meta["description"] = details["description"]
    if details.get("series"):
        updated_meta["series"] = details["series"]
    if details.get("entity_kind"):
        updated_meta["entity_kind"] = details["entity_kind"]
    if details.get("children") is not None:
        updated_meta["children"] = details["children"]
    updated_meta["next_up"] = details.get("next_up")
    # Persist the rich metadata so the Files grid + Series detail can
    # render hero/cast/chips without re-hitting the provider on every
    # navigation. These keys are the contract the frontend reads from
    # source_metadata when the entity is a Series/Season/Episode.
    for _k in (
        "tagline", "status", "end_year", "premiere_date",
        "official_rating", "community_rating", "network",
        "season_count", "episode_count", "genres", "has_backdrop",
    ):
        if details.get(_k) is not None:
            updated_meta[_k] = details[_k]
    people = details.get("people")
    if isinstance(people, dict):
        updated_meta["cast"] = people.get("cast") or []
        updated_meta["directors"] = people.get("directors") or []
        updated_meta["writers"] = people.get("writers") or []
    if details.get("selected_episode_id"):
        updated_meta["selected_episode_id"] = details["selected_episode_id"]
    if details.get("selected_episode_title"):
        updated_meta["selected_episode_title"] = details["selected_episode_title"]
    if details.get("selected_episode_stream_path"):
        updated_meta["selected_episode_stream_path"] = details["selected_episode_stream_path"]
    playback = details.get("playback")
    if isinstance(playback, dict):
        if playback.get("selected_media_source_id"):
            updated_meta["preferred_media_source_id"] = playback["selected_media_source_id"]
        if playback.get("selected_audio_stream_index") is not None:
            updated_meta["preferred_audio_stream_index"] = playback["selected_audio_stream_index"]
        if playback.get("selected_subtitle_stream_index") is not None:
            updated_meta["preferred_subtitle_stream_index"] = playback["selected_subtitle_stream_index"]
    await idx.update_source_metadata(entry.id, updated_meta, user_id=uid)
    details.pop("selected_episode_stream_path", None)

    return JSONResponse(details)


def _details_from_meta(entry, meta: dict) -> dict:
    """Render a details payload from only what we have cached locally.

    Shape is shared between ABS and LibriVox. Extra LibriVox-specific
    fields (authors_detailed, copyright_year, etc.) come back empty for
    ABS rows — frontend shows only what's populated.
    """
    # published_year for LibriVox rows comes from copyright_year; ABS rows
    # fill this via _normalise_abs_details, so this fallback path uses
    # whichever we cached (copyright_year is only set by the LibriVox pin).
    copyright_year = meta.get("copyright_year", "")
    published_year = None
    try:
        if copyright_year:
            published_year = int(copyright_year)
    except (ValueError, TypeError):
        published_year = None

    # audio_files only carries what the player needs: the per-file duration
    # powers book-time ↔ file-time translation when a book is served as N
    # separate MP3s (LibriVox). Single-file sources (ABS) get an empty list
    # and the player falls back to chapter-offset-within-one-file mode.
    audio_files_raw = meta.get("audio_files") or []
    audio_files: list[dict] = [
        {"duration_s": float(af.get("duration_s") or 0)}
        for af in audio_files_raw if isinstance(af, dict)
    ] if isinstance(audio_files_raw, list) else []
    entity_kind = str(meta.get("entity_kind") or meta.get("library_kind") or "").strip()

    return {
        "id":             entry.id,
        "title":          entry.name,
        "subtitle":       "",
        "author":         meta.get("author", ""),
        "narrator":       meta.get("narrator", ""),
        "series":         meta.get("series") or None,
        "published_year": published_year,
        "description":    meta.get("description", ""),
        "genres":         meta.get("genres") or [],
        "cover_url":      f"/api/media/cover/{entry.id}" if meta.get("has_cover") else "",
        "duration_s":     float(meta.get("duration_s") or 0),
        "current_time_s": float(meta.get("current_time_s") or 0),
        "progress_pct":   float(meta.get("progress_pct") or 0),
        "is_finished":    bool(meta.get("is_finished") or False),
        "chapters":       meta.get("chapters") or [],
        "audio_files":    audio_files,
        "entity_kind":    entity_kind,
        "children":       meta.get("children") or [],
        "selected_episode_id": str(meta.get("selected_episode_id") or ""),
        "selected_episode_title": str(meta.get("selected_episode_title") or ""),
        "playable":       bool(meta.get("stream_path") or meta.get("selected_episode_id")),
        # LibriVox-specific enrichment (empty for ABS, harmless to include):
        "authors_detailed": meta.get("authors_detailed") or [],
        "translators":      meta.get("translators") or [],
        "narrators":        meta.get("narrators") or [],
        "language":         meta.get("language", ""),
        "totaltime":        meta.get("totaltime", ""),
        "url_text_source":  meta.get("url_text_source", ""),
        "url_project":      meta.get("url_project", ""),
        "url_rss":          meta.get("url_rss", ""),
        "url_other":        meta.get("url_other", ""),
        "url_zip_file":     meta.get("url_zip_file", ""),
        "librivox_url":     meta.get("librivox_url", ""),
        # Read-along state (populated by the gutenberg_fetch background
        # job — see augmentum/jobs/handlers/gutenberg_fetch.py). The UI
        # uses gutenberg_status to choose between "Read along" button,
        # "Fetching…" chip, or silent hide.
        "gutenberg_status":     meta.get("gutenberg_status", ""),
        "gutenberg_word_count": int(meta.get("gutenberg_word_count") or 0),
        "gutenberg_byte_size":  int(meta.get("gutenberg_byte_size") or 0),
}


def _normalise_abs_podcast_children(
    raw: dict,
    *,
    active_episode_id: str,
    progress: dict,
) -> list[dict]:
    media = raw.get("media") or {}
    episodes_raw = media.get("episodes") or []
    if not isinstance(episodes_raw, list):
        return []

    current_time_s = float(progress.get("currentTime") or 0)
    progress_pct = float(progress.get("progress") or 0) * 100.0
    is_finished = bool(progress.get("isFinished") or False)

    children: list[dict] = []
    for episode in episodes_raw:
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("id") or "").strip()
        if not episode_id:
            continue
        audio_track = episode.get("audioTrack") or {}
        duration_s = float(
            episode.get("duration")
            or audio_track.get("duration")
            or 0
        )
        child = {
            "episode_id": episode_id,
            "name": str(
                episode.get("title")
                or episode.get("displayTitle")
                or episode.get("subtitle")
                or "Untitled Episode"
            ).strip() or "Untitled Episode",
            "description": str(episode.get("description") or "").strip(),
            "published_at": int(episode.get("publishedAt") or 0),
            "pub_date": str(episode.get("pubDate") or "").strip(),
            "duration_s": duration_s,
            "playable": bool(audio_track.get("contentUrl")),
            "stream_path": str(audio_track.get("contentUrl") or "").strip(),
            "is_selected": episode_id == active_episode_id,
            "current_time_s": 0.0,
            "progress_pct": 0.0,
            "is_finished": False,
        }
        if episode_id == active_episode_id:
            child["current_time_s"] = current_time_s
            child["progress_pct"] = max(0.0, min(100.0, progress_pct))
            child["is_finished"] = is_finished
        children.append(child)
    return children


def _normalise_abs_details(
    raw: dict,
    *,
    cached_meta: dict,
    selected_episode_id: str = "",
) -> dict:
    """Translate ABS's expanded LibraryItem into our normalized detail shape."""
    media = raw.get("media") or {}
    metadata = media.get("metadata") or {}
    entity_kind = str(
        cached_meta.get("entity_kind")
        or cached_meta.get("library_kind")
        or ("podcast" if media.get("episodes") else "book")
    ).strip().lower()

    # ABS's detail endpoint returns `authors`/`narrators` as objects with
    # {id, name} — flatten to a pre-joined string to match our listing
    # convention. Fall back to the pre-joined listing strings if the
    # objects aren't present.
    def _join_name_objects(objs, fallback: str) -> str:
        if not objs:
            return fallback
        names = [
            o.get("name", "") for o in objs
            if isinstance(o, dict) and o.get("name")
        ]
        return ", ".join(n for n in names if n) or fallback

    author = _join_name_objects(
        metadata.get("authors"), metadata.get("authorName", "") or "",
    )
    narrator = _join_name_objects(
        metadata.get("narrators"), metadata.get("narratorName", "") or "",
    )

    # Series can be either a single object or an array (multi-series
    # books exist, e.g. Wheel of Time + "Fantasy Masterworks"). Pick the
    # first since UI displays one.
    series_raw = metadata.get("series")
    if isinstance(series_raw, list):
        series_raw = series_raw[0] if series_raw else None
    series = None
    if isinstance(series_raw, dict):
        series = {
            "name":     series_raw.get("name", ""),
            "sequence": str(series_raw.get("sequence", "")),
        }
    elif metadata.get("seriesName"):
        series = {"name": metadata["seriesName"], "sequence": ""}

    chapters = [
        {
            "title": c.get("title") or f"Chapter {i + 1}",
            "start": float(c.get("start") or 0),
            "end":   float(c.get("end") or 0),
        }
        for i, c in enumerate(media.get("chapters") or [])
    ]

    # Progress: the expanded endpoint returns `userMediaProgress` with
    # the user's position — use it as the freshest source of truth.
    prog = raw.get("userMediaProgress") or {}
    current_time_s = float(prog.get("currentTime") or cached_meta.get("current_time_s") or 0)
    duration_s = float(prog.get("duration") or 0)
    if not duration_s:
        duration_s = float(media.get("duration") or 0) or float(
            cached_meta.get("duration_s") or 0
        )
    progress_pct = float(prog.get("progress") or 0) * 100.0
    if not progress_pct and duration_s > 0:
        progress_pct = (current_time_s / duration_s) * 100.0

    is_finished = bool(prog.get("isFinished") or False)
    children: list[dict] = []
    selected_episode_title = ""
    playback_title = ""
    selected_episode_stream_path = ""
    if entity_kind == "podcast":
        active_episode_id = str(
            selected_episode_id or cached_meta.get("selected_episode_id") or ""
        ).strip()
        children = _normalise_abs_podcast_children(
            raw,
            active_episode_id=active_episode_id,
            progress=prog if isinstance(prog, dict) else {},
        )
        active_child = next(
            (child for child in children if child.get("episode_id") == active_episode_id),
            None,
        )
        if active_child is not None:
            selected_episode_title = str(active_child.get("name") or "").strip()
            playback_title = selected_episode_title
            duration_s = float(active_child.get("duration_s") or duration_s or 0)
            progress_pct = float(active_child.get("progress_pct") or progress_pct or 0)
            current_time_s = float(active_child.get("current_time_s") or current_time_s or 0)
            is_finished = bool(active_child.get("is_finished") or False)
            selected_episode_stream_path = str(active_child.get("stream_path") or "").strip()
        else:
            current_time_s = float(cached_meta.get("current_time_s") or 0)
            duration_s = float(cached_meta.get("duration_s") or 0)
            progress_pct = float(cached_meta.get("progress_pct") or 0)
            is_finished = bool(cached_meta.get("is_finished") or False)
            selected_episode_stream_path = str(
                cached_meta.get("selected_episode_stream_path") or ""
            ).strip()

    return {
        "title":          metadata.get("title") or raw.get("name") or "Untitled",
        "subtitle":       metadata.get("subtitle") or "",
        "author":         author,
        "narrator":       narrator,
        "series":         series,
        "published_year": metadata.get("publishedYear") or None,
        "description":    metadata.get("description") or "",
        "genres":         metadata.get("genres") or [],
        "duration_s":     duration_s,
        "current_time_s": current_time_s,
        "progress_pct":   max(0.0, min(100.0, progress_pct)),
        "is_finished":    is_finished,
        "chapters":       chapters,
        "entity_kind":    entity_kind,
        "children":       children,
        "selected_episode_id": str(selected_episode_id or cached_meta.get("selected_episode_id") or ""),
        "selected_episode_title": selected_episode_title or str(cached_meta.get("selected_episode_title") or ""),
        "selected_episode_stream_path": selected_episode_stream_path,
        "playable":       bool(cached_meta.get("stream_path") or selected_episode_id or cached_meta.get("selected_episode_id")),
        "playback_title": playback_title,
    }


def _stream_label(stream: dict, *, kind: str, fallback: str) -> str:
    display = str(stream.get("DisplayTitle") or "").strip()
    if display:
        return display
    parts: list[str] = []
    language = (
        str(stream.get("DisplayLanguage") or "").strip()
        or str(stream.get("Language") or "").strip()
        or str(stream.get("Title") or "").strip()
    )
    if language:
        parts.append(language)
    title = str(stream.get("Title") or "").strip()
    if title and title.lower() not in {language.lower(), fallback.lower()}:
        parts.append(title)
    codec = str(stream.get("Codec") or "").strip().upper()
    if codec:
        parts.append(codec)
    if kind == "audio":
        channels = stream.get("Channels")
        if channels not in (None, ""):
            try:
                parts.append(f"{float(channels):g}ch")
            except (TypeError, ValueError):
                pass
    if kind == "subtitle" and bool(stream.get("IsForced")):
        parts.append("Forced")
    return " • ".join(p for p in parts if p) or fallback


def _media_source_label(source: dict, index: int, audio_tracks: list[dict]) -> str:
    name = str(source.get("Name") or "").strip()
    if name:
        return name
    parts: list[str] = []
    video_codec = str(source.get("VideoCodec") or "").strip().upper()
    container = str(source.get("Container") or "").strip().upper()
    height = _int_or_none(source.get("Height"))
    if height is None:
        media_streams = source.get("MediaStreams") or []
        if isinstance(media_streams, list):
            for stream in media_streams:
                if not isinstance(stream, dict):
                    continue
                if str(stream.get("Type") or "").lower() == "video":
                    height = _int_or_none(stream.get("Height"))
                    break
    if height:
        parts.append(f"{height}p")
    if video_codec:
        parts.append(video_codec)
    if container:
        parts.append(container)
    langs = []
    for track in audio_tracks:
        lang = str(track.get("language") or "").strip()
        if lang and lang not in langs:
            langs.append(lang)
    if langs:
        parts.append("/".join(langs[:2]))
    return " • ".join(parts) or f"Version {index + 1}"


def _choose_track_index(
    tracks: list[dict],
    *,
    preferred: int | None,
    default_idx: int | None = None,
    allow_none: bool = False,
) -> int | None:
    valid = {int(track.get("index")) for track in tracks if track.get("index") is not None}
    if preferred is not None and (preferred in valid or (allow_none and preferred == -1)):
        return preferred
    if default_idx is not None and default_idx in valid:
        return default_idx
    for track in tracks:
        if track.get("is_default") and track.get("index") is not None:
            return int(track["index"])
    if allow_none:
        return -1
    if tracks:
        first = tracks[0].get("index")
        return int(first) if first is not None else None
    return -1 if allow_none else None


def _normalise_emby_compat_playback(raw: dict, *, cached_meta: dict) -> dict | None:
    playback_raw = raw.get("_augmentum_playback_info")
    if not isinstance(playback_raw, dict):
        playback_raw = {}
    media_sources_raw = playback_raw.get("MediaSources")
    if not isinstance(media_sources_raw, list) or not media_sources_raw:
        media_sources_raw = raw.get("MediaSources") or []
    if not isinstance(media_sources_raw, list) or not media_sources_raw:
        return None

    preferred_source = str(cached_meta.get("preferred_media_source_id") or "").strip()
    preferred_audio = _int_or_none(cached_meta.get("preferred_audio_stream_index"))
    preferred_subtitle = _int_or_none(cached_meta.get("preferred_subtitle_stream_index"))

    sources: list[dict[str, Any]] = []
    for idx, source in enumerate(media_sources_raw):
        if not isinstance(source, dict):
            continue
        source_id = str(
            source.get("Id")
            or source.get("MediaSourceId")
            or f"source_{idx}"
        ).strip() or f"source_{idx}"
        streams_raw = source.get("MediaStreams") or []
        if not isinstance(streams_raw, list):
            streams_raw = []
        audio_tracks: list[dict[str, Any]] = []
        subtitle_tracks: list[dict[str, Any]] = []
        for stream in streams_raw:
            if not isinstance(stream, dict):
                continue
            stream_type = str(stream.get("Type") or "").lower()
            stream_index = _int_or_none(stream.get("Index"))
            if stream_index is None:
                continue
            if stream_type == "audio":
                display_language = str(
                    stream.get("DisplayLanguage")
                    or stream.get("Language")
                    or ""
                ).strip()
                language_code = str(stream.get("Language") or "").strip().lower()
                audio_tracks.append({
                    "index": stream_index,
                    "label": _stream_label(
                        stream, kind="audio", fallback=f"Audio {len(audio_tracks) + 1}",
                    ),
                    "language": display_language,
                    "language_code": language_code,
                    "codec": str(stream.get("Codec") or "").strip().lower(),
                    "channels": _int_or_none(stream.get("Channels")),
                    "title": str(stream.get("Title") or "").strip(),
                    "is_default": bool(stream.get("IsDefault") or False),
                })
            elif stream_type == "subtitle":
                display_language = str(
                    stream.get("DisplayLanguage")
                    or stream.get("Language")
                    or ""
                ).strip()
                language_code = str(stream.get("Language") or "").strip().lower()
                subtitle_tracks.append({
                    "index": stream_index,
                    "label": _stream_label(
                        stream,
                        kind="subtitle",
                        fallback=f"Subtitle {len(subtitle_tracks) + 1}",
                    ),
                    "language": display_language,
                    "language_code": language_code,
                    "title": str(stream.get("Title") or "").strip(),
                    "codec": str(stream.get("Codec") or "").strip().lower(),
                    "is_default": bool(stream.get("IsDefault") or False),
                    "is_forced": bool(stream.get("IsForced") or False),
                    "is_external": bool(stream.get("IsExternal") or False),
                    "is_text": bool(stream.get("IsTextSubtitleStream") or False),
                })
        subtitle_tracks.insert(0, {
            "index": -1,
            "label": "Off",
            "language": "",
            "language_code": "",
            "title": "",
            "codec": "",
            "is_default": not any(track.get("is_default") for track in subtitle_tracks),
            "is_forced": False,
            "is_external": False,
            "is_none": True,
            "is_text": True,
        })
        default_audio_idx = _int_or_none(source.get("DefaultAudioStreamIndex"))
        default_subtitle_idx = _int_or_none(source.get("DefaultSubtitleStreamIndex"))
        sources.append({
            "id": source_id,
            "label": _media_source_label(source, idx, audio_tracks),
            "container": str(source.get("Container") or "").strip().lower(),
            "video_codec": str(source.get("VideoCodec") or "").strip().lower(),
            "is_default": idx == 0,
            "audio_tracks": audio_tracks,
            "subtitle_tracks": subtitle_tracks,
            "default_audio_stream_index": default_audio_idx,
            "default_subtitle_stream_index": default_subtitle_idx,
        })

    if not sources:
        return None

    selected_source = preferred_source if preferred_source in {
        str(source.get("id") or "") for source in sources
    } else str(sources[0].get("id") or "")
    source_obj = next(
        (source for source in sources if str(source.get("id") or "") == selected_source),
        sources[0],
    )
    selected_audio = _choose_track_index(
        source_obj.get("audio_tracks") or [],
        preferred=preferred_audio,
        default_idx=_int_or_none(source_obj.get("default_audio_stream_index")),
    )
    selected_subtitle = _choose_track_index(
        source_obj.get("subtitle_tracks") or [],
        preferred=preferred_subtitle,
        default_idx=_int_or_none(source_obj.get("default_subtitle_stream_index")),
        allow_none=True,
    )

    for source in sources:
        is_selected_source = str(source.get("id") or "") == selected_source
        source["is_selected"] = is_selected_source
        for track in source.get("audio_tracks") or []:
            track["is_selected"] = bool(
                is_selected_source and track.get("index") == selected_audio
            )
        for track in source.get("subtitle_tracks") or []:
            track["is_selected"] = bool(
                is_selected_source and track.get("index") == selected_subtitle
            )

    return {
        "selected_media_source_id": selected_source,
        "selected_audio_stream_index": selected_audio,
        "selected_subtitle_stream_index": selected_subtitle,
        "media_sources": sources,
        "has_multiple_versions": len(sources) > 1,
    }


def _normalise_emby_compat_details(
    raw: dict,
    *,
    cached_meta: dict,
    provider: str,
) -> dict:
    """Translate Emby/Jellyfin item details into a provider-neutral shape."""
    item_type = str(raw.get("Type") or "").strip()
    run_time_ticks = int(raw.get("RunTimeTicks") or 0)
    duration_s = float(cached_meta.get("duration_s") or 0.0)
    if run_time_ticks > 0:
        duration_s = run_time_ticks / 10_000_000.0
    user_data = raw.get("UserData") or {}
    current_time_s = float(cached_meta.get("current_time_s") or 0.0)
    playback_ticks = int(user_data.get("PlaybackPositionTicks") or 0)
    if playback_ticks > 0:
        current_time_s = playback_ticks / 10_000_000.0
    progress_pct = float(cached_meta.get("progress_pct") or 0.0)
    if run_time_ticks > 0 and playback_ticks > 0:
        progress_pct = min(100.0, max(0.0, playback_ticks / run_time_ticks * 100.0))
    is_finished = bool(
        user_data.get("Played")
        or user_data.get("PlayedPercentage") == 100
        or progress_pct >= 99.9
    )
    genres = raw.get("Genres") or cached_meta.get("genres") or []
    if not isinstance(genres, list):
        genres = []

    people = raw.get("People") or []
    directors = [
        str(person.get("Name") or "").strip()
        for person in people
        if isinstance(person, dict)
        and str(person.get("Type") or "").lower() == "director"
        and str(person.get("Name") or "").strip()
    ]
    writers = [
        str(person.get("Name") or "").strip()
        for person in people
        if isinstance(person, dict)
        and str(person.get("Type") or "").lower() == "writer"
        and str(person.get("Name") or "").strip()
    ]
    # Cast: front the actor list (capped) with character names. Person IDs
    # + image tags are stored so a future /api/media/person endpoint can
    # serve thumbnails without a second Jellyfin round-trip.
    cast: list[dict] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        if str(person.get("Type") or "").lower() != "actor":
            continue
        name = str(person.get("Name") or "").strip()
        if not name:
            continue
        cast.append({
            "name": name,
            "role": str(person.get("Role") or "").strip(),
            "person_id": str(person.get("Id") or "").strip(),
            "image_tag": str(person.get("PrimaryImageTag") or "").strip(),
        })
        if len(cast) >= 18:
            break

    # Studios: take the first as "network" (HBO, Paramount, FX, etc.).
    studios_raw = raw.get("Studios") or []
    network = ""
    if isinstance(studios_raw, list) and studios_raw:
        first_studio = studios_raw[0]
        if isinstance(first_studio, dict):
            network = str(first_studio.get("Name") or "").strip()

    # Status / dates / ratings — present on Series, sometimes on Movies.
    status_raw = str(raw.get("Status") or "").strip()
    end_date_raw = str(raw.get("EndDate") or "").strip()
    end_year = 0
    if end_date_raw and len(end_date_raw) >= 4 and end_date_raw[:4].isdigit():
        end_year = int(end_date_raw[:4])
    premiere_date_raw = str(raw.get("PremiereDate") or "").strip()
    premiere_date = premiere_date_raw[:10] if len(premiere_date_raw) >= 10 else ""
    official_rating = str(raw.get("OfficialRating") or "").strip()
    try:
        community_rating = float(raw.get("CommunityRating") or 0) or 0.0
    except (TypeError, ValueError):
        community_rating = 0.0
    tagline = str(raw.get("Tagline") or "").strip()

    # Counts: ChildCount = direct children (seasons for Series, episodes
    # for Season). RecursiveItemCount = all descendants (total episodes
    # for Series). For movies these are zero/absent.
    child_count = int(raw.get("ChildCount") or 0)
    recursive_count = int(raw.get("RecursiveItemCount") or 0)
    season_count = child_count if item_type == "Series" else 0
    episode_count = (
        recursive_count if item_type == "Series"
        else child_count if item_type == "Season"
        else 0
    )

    # BackdropImageTags is a list; a non-empty list means the provider
    # has a landscape backdrop for this item. Frontend uses this flag
    # to decide between /api/media/backdrop/ and the poster /cover/.
    backdrop_tags = raw.get("BackdropImageTags") or []
    has_backdrop = bool(isinstance(backdrop_tags, list) and backdrop_tags)

    series = None
    if item_type in {"Episode", "Season"} and cached_meta.get("series_name"):
        series = {"name": str(cached_meta.get("series_name") or "").strip(), "sequence": ""}

    children_raw = raw.get("_augmentum_children") or []
    children: list[dict] = []
    if isinstance(children_raw, list):
        for child in children_raw:
            child_type = str(child.get("Type") or "").strip()
            children.append({
                "external_id": str(child.get("Id") or "").strip(),
                "name": str(child.get("Name") or "").strip(),
                "type": child_type,
                "entity_kind": {
                    "Series": "series",
                    "Season": "season",
                    "Episode": "episode",
                    "Movie": "movie",
                    "MusicVideo": "music_video",
                }.get(child_type, "other"),
                "season_number": int(child.get("ParentIndexNumber") or 0),
                "episode_number": int(child.get("IndexNumber") or 0),
                "is_finished": bool((child.get("UserData") or {}).get("Played") or False),
            })
    # Seasons/episodes arrive in whatever order the provider indexed them;
    # force ordered by (season, episode) so the UI list is never "random".
    children.sort(key=lambda c: (c.get("season_number") or 0, c.get("episode_number") or 0))

    # For a Series: walk the full recursive episode list to derive the
    # next-up target (resume-in-progress > first-unwatched > first-overall
    # for a full rewatch). file_id is filled in by the details route after
    # looking the episode up in file_index.
    next_up: dict | None = None
    if item_type == "Series":
        episodes_raw = raw.get("_augmentum_episodes") or []
        episodes: list[dict] = []
        if isinstance(episodes_raw, list):
            for ep in episodes_raw:
                if not isinstance(ep, dict):
                    continue
                user_data = ep.get("UserData") or {}
                episodes.append({
                    "external_id": str(ep.get("Id") or "").strip(),
                    "name": str(ep.get("Name") or "").strip(),
                    "season_number": int(ep.get("ParentIndexNumber") or 0),
                    "episode_number": int(ep.get("IndexNumber") or 0),
                    "is_finished": bool(user_data.get("Played") or False),
                    "position_ticks": int(user_data.get("PlaybackPositionTicks") or 0),
                    "run_time_ticks": int(ep.get("RunTimeTicks") or 0),
                })
        episodes.sort(key=lambda e: (e.get("season_number") or 0, e.get("episode_number") or 0))

        def _fmt_label(ep: dict, *, verb: str) -> str:
            s = ep.get("season_number") or 0
            n = ep.get("episode_number") or 0
            if s > 0 and n > 0:
                return f"{verb} S{s}E{n}"
            if n > 0:
                return f"{verb} Episode {n}"
            return verb

        # Specials live in Season 0 on Jellyfin/Emby. They shouldn't drive
        # "Start" / "Resume" — a user opening an unwatched show expects
        # S1E1, not a specials episode. Exclude season 0 from the picker
        # but leave them in children so they're still browsable.
        canonical = [ep for ep in episodes if (ep.get("season_number") or 0) >= 1]

        picked: dict | None = None
        mode = ""
        for ep in canonical:
            if ep["position_ticks"] > 0 and not ep["is_finished"]:
                picked, mode = ep, "resume"
                break
        if not picked:
            for ep in canonical:
                if not ep["is_finished"]:
                    picked, mode = ep, "start"
                    break
        if not picked and canonical:
            picked, mode = canonical[0], "rewatch"
        if picked:
            verb = {"resume": "Resume", "start": "Start", "rewatch": "Rewatch"}[mode]
            next_up = {
                "external_id": picked["external_id"],
                "file_id": "",
                "name": picked["name"],
                "season_number": picked["season_number"],
                "episode_number": picked["episode_number"],
                "mode": mode,
                "label": _fmt_label(picked, verb=verb),
                "current_time_s": picked["position_ticks"] / 10_000_000.0 if picked["position_ticks"] > 0 else 0.0,
            }

    description = str(raw.get("Overview") or cached_meta.get("overview") or "").strip()
    playback = _normalise_emby_compat_playback(raw, cached_meta=cached_meta)
    return {
        "title": raw.get("Name") or "Untitled",
        "subtitle": "",
        "author": ", ".join(directors or writers),
        "narrator": "",
        "series": series,
        "published_year": raw.get("ProductionYear") or cached_meta.get("year") or None,
        "description": description,
        "genres": [str(g).strip() for g in genres if str(g).strip()],
        "duration_s": duration_s,
        "current_time_s": current_time_s,
        "progress_pct": progress_pct,
        "is_finished": is_finished,
        "chapters": [],
        "entity_kind": cached_meta.get("entity_kind") or "",
        "season_number": int(cached_meta.get("season_number") or 0),
        "episode_number": int(cached_meta.get("episode_number") or 0),
        "children": children,
        "next_up": next_up,
        "provider_item_type": item_type,
        "library_name": cached_meta.get("library_name") or "",
        "provider_collection_type": cached_meta.get("provider_collection_type") or "",
        "playable": bool(cached_meta.get("stream_path")),
        "playback": playback,
        "people": {
            "directors": directors,
            "writers": writers,
            "cast": cast,
        },
        "tagline": tagline,
        "status": status_raw,
        "end_year": end_year,
        "premiere_date": premiere_date,
        "official_rating": official_rating,
        "community_rating": community_rating,
        "network": network,
        "season_count": season_count,
        "episode_count": episode_count,
        "has_backdrop": has_backdrop,
        "provider": provider,
    }


@router.post("/selection/{file_id}")
async def update_playback_selection(
    file_id: str,
    body: PlaybackSelectionUpdate,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    raw_meta = entry.source_metadata
    if isinstance(raw_meta, dict):
        meta = dict(raw_meta)
    else:
        try:
            meta = json.loads(raw_meta or "{}")
        except Exception:
            meta = {}

    meta["preferred_media_source_id"] = str(body.media_source_id or "").strip()
    if body.audio_stream_index is None:
        meta.pop("preferred_audio_stream_index", None)
    else:
        meta["preferred_audio_stream_index"] = int(body.audio_stream_index)
    if body.subtitle_stream_index is None:
        meta.pop("preferred_subtitle_stream_index", None)
    else:
        meta["preferred_subtitle_stream_index"] = int(body.subtitle_stream_index)

    await idx.update_source_metadata(file_id, meta, user_id=uid)
    return JSONResponse({
        "status": "ok",
        "selection": {
            "media_source_id": meta.get("preferred_media_source_id") or "",
            "audio_stream_index": meta.get("preferred_audio_stream_index"),
            "subtitle_stream_index": meta.get("preferred_subtitle_stream_index"),
        },
    })


@router.get("/related/{file_id}")
async def media_related(file_id: str, request: Request) -> JSONResponse:
    """Books sharing the same normalised author / narrator as this row.

    User-scoped at every step: the seed row is fetched via the
    user-scoped index, and the SQL for peers carries ``user_id = ?`` +
    ``is_trashed = 0`` + ``id != ?``. A malicious guess of another
    user's file_id resolves to None at the first lookup and returns an
    empty list rather than leaking peers.
    """
    by = (request.query_params.get("by") or "author").lower()
    if by not in ("author", "narrator", "genre", "series"):
        return JSONResponse(
            {"error": "invalid 'by' (expected 'author', 'narrator', 'genre' or 'series')"},
            status_code=400,
        )
    limit = max(1, min(int(request.query_params.get("limit", "20")), 50))

    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"items": [], "display_name": ""})

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"items": [], "display_name": ""})

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}

    if by == "genre":
        return await _related_by_genre(idx, entry, meta, uid=uid, limit=limit)
    if by == "series":
        return await _related_by_series(idx, entry, meta, uid=uid, limit=limit)
    key_norm = f"{by}_normalized"
    key_raw  = by
    target = (meta.get(key_norm) or "").strip()
    display_name = (meta.get(key_raw) or "").strip()
    # Author axis: recompute the match key from the RAW author with
    # non-author role credits stripped ("Okina Baba, Jenny McKeon -
    # translator" -> "Okina Baba"). This (a) fixes existing rows whose
    # stored author_normalized is polluted, without a re-sync, and (b)
    # lets series volumes that credit different translators/illustrators
    # match on the shared real author. Narrator/genre keep stored keys.
    if by == "author":
        target = normalize_name(author_for_match((meta.get("author") or "").strip()))
    if not target:
        return JSONResponse({"items": [], "display_name": display_name})

    # Pull every candidate row with a non-empty normalised credit on the
    # requested axis and filter in Python. Strict equality on the
    # canonical string is too brittle for real libraries: "JF Brink" and
    # "JF Brink TheFirstDefier" (an uploader concatenated a work title
    # into the author field) normalise to different tokens and would
    # never match. Likewise, a book with co-authors "Jane Austen,
    # Charles Dickens" would fail to find either author's solo books.
    # Token-subset matching handles both cases.
    db = idx._db
    json_path = f"$.{by}_normalized"
    json_raw_path = f"$.{by}"
    has_cover_path = "$.has_cover"
    progress_path = "$.progress_pct"
    narrator_path = "$.narrator"
    author_path = "$.author"
    # json_extract-per-column keeps us from decoding the whole
    # source_metadata blob per row (chapters/audio_files can be huge).
    # Cap at 5000 scanned rows — large enough for any realistic library,
    # small enough that Python-side filtering stays fast.
    cursor = await db.execute(
        "SELECT id, name, "
        "  COALESCE(json_extract(source_metadata, ?), '') AS norm, "
        "  COALESCE(json_extract(source_metadata, ?), '') AS raw_author, "
        "  COALESCE(json_extract(source_metadata, ?), '') AS raw_narrator, "
        "  COALESCE(json_extract(source_metadata, ?), 0)  AS has_cover, "
        "  COALESCE(json_extract(source_metadata, ?), 0)  AS progress_pct "
        "FROM file_index "
        "WHERE user_id = ? AND is_trashed = 0 AND id != ? "
        "  AND COALESCE(json_extract(source_metadata, ?), '') != '' "
        "ORDER BY created_at DESC LIMIT 5000",
        (
            json_path, author_path, narrator_path, has_cover_path, progress_path,
            uid, entry.id, json_path,
        ),
    )
    rows = await cursor.fetchall()

    items: list[dict] = []
    for row in rows:
        row_id, row_name, row_norm, raw_author, raw_narrator, has_cover, progress_pct = row
        cand_norm = row_norm or ""
        if by == "author":
            # Mirror the seed: strip non-author role credits before matching.
            cand_norm = normalize_name(author_for_match(raw_author or ""))
        if not tokens_match_as_related(target, cand_norm):
            continue
        items.append({
            "id":        row_id,
            "title":     row_name,
            "author":    raw_author or "",
            "narrator":  raw_narrator or "",
            "cover_url": f"/api/media/cover/{row_id}" if has_cover else "",
            "progress_pct": float(progress_pct or 0),
        })
        if len(items) >= limit:
            break

    return JSONResponse({
        "items":        items,
        "display_name": display_name,
        "match_key":    by,
    })


def _series_sequence_key(raw_seq: str) -> tuple[int, float, str]:
    """Sort key for a series sequence — numeric volumes first (1, 2, 10),
    then any non-numeric labels alphabetically. Returns a tuple so
    unparseable sequences sort *after* numbered ones rather than crashing
    the sort or landing at 0."""
    s = (raw_seq or "").strip()
    try:
        return (0, float(s), "")
    except (TypeError, ValueError):
        return (1, 0.0, s.lower())


async def _related_by_series(
    idx, entry, meta: dict, *, uid: str, limit: int,
) -> JSONResponse:
    """"More in this series" — sibling volumes/books sharing the series,
    ordered by volume sequence (1, 2, … 10) rather than scan recency.

    Mirrors the author/narrator axes' user-scoping + scan cap. Matches on
    the normalised series name (token-subset, same as author) so minor
    tagging variants still group. Same-kind only so an audiobook series
    doesn't pull in an unrelated comic of the same name.
    """
    seed_series = (meta.get("series_name") or "").strip()
    target = normalize_name(seed_series)
    if not target:
        return JSONResponse({"items": [], "display_name": "", "match_key": "series"})
    seed_kind = (entry.kind or "").lower()

    db = idx._db  # noqa: SLF001
    cursor = await db.execute(
        "SELECT id, name, "
        "  COALESCE(json_extract(source_metadata, '$.series_normalized'), '') AS norm, "
        "  COALESCE(json_extract(source_metadata, '$.series_sequence'), '')   AS seq, "
        "  COALESCE(json_extract(source_metadata, '$.has_cover'), 0)  AS has_cover, "
        "  COALESCE(json_extract(source_metadata, '$.progress_pct'), 0) AS progress_pct "
        "FROM file_index "
        "WHERE user_id = ? AND is_trashed = 0 AND id != ? AND kind = ? "
        "  AND COALESCE(json_extract(source_metadata, '$.series_normalized'), '') != '' "
        "LIMIT 5000",
        (uid, entry.id, seed_kind),
    )
    rows = await cursor.fetchall()

    matched: list[tuple[tuple[int, float, str], dict]] = []
    for row_id, row_name, row_norm, row_seq, has_cover, progress_pct in rows:
        if not tokens_match_as_related(target, row_norm or ""):
            continue
        matched.append((_series_sequence_key(row_seq), {
            "id":           row_id,
            "title":        row_name,
            "author":       "",
            "narrator":     "",
            "cover_url":    f"/api/media/cover/{row_id}" if has_cover else "",
            "progress_pct": float(progress_pct or 0),
        }))
    matched.sort(key=lambda t: t[0])
    items = [item for _, item in matched[:limit]]
    return JSONResponse({
        "items":        items,
        "display_name": seed_series,
        "match_key":    "series",
    })


async def _related_by_genre(
    idx, entry, meta: dict, *, uid: str, limit: int,
) -> JSONResponse:
    """"More like this" for video rows — peers sharing the most genres.

    Same user-scoping and scan-cap discipline as the author/narrator
    axes. Movies and series relate only to other movies/series (every
    episode of a sitcom sharing "Comedy" would drown the strip);
    non-video kinds relate within their own kind unrestricted.
    """
    seed_genres = [
        str(g).strip() for g in (meta.get("genres") or []) if str(g).strip()
    ]
    if not seed_genres:
        return JSONResponse({"items": [], "display_name": "", "match_key": "genre"})
    seed_set = {g.lower() for g in seed_genres}
    seed_kind = (entry.kind or "").lower()
    allowed_eks = {"movie", "series"} if seed_kind == "video" else None

    db = idx._db  # noqa: SLF001
    cursor = await db.execute(
        "SELECT id, name, "
        "  COALESCE(json_extract(source_metadata, '$.genres'), '[]') AS genres, "
        "  COALESCE(json_extract(source_metadata, '$.entity_kind'), '') AS ek, "
        "  COALESCE(json_extract(source_metadata, '$.has_cover'), 0)  AS has_cover, "
        "  COALESCE(json_extract(source_metadata, '$.progress_pct'), 0) AS progress_pct "
        "FROM file_index "
        "WHERE user_id = ? AND is_trashed = 0 AND id != ? AND kind = ? "
        "  AND COALESCE(json_extract(source_metadata, '$.genres'), '[]') != '[]' "
        "ORDER BY created_at DESC LIMIT 5000",
        (uid, entry.id, seed_kind),
    )
    rows = await cursor.fetchall()

    scored: list[tuple[int, dict]] = []
    for row_id, row_name, genres_raw, ek, has_cover, progress_pct in rows:
        if allowed_eks is not None and (ek or "").lower() not in allowed_eks:
            continue
        try:
            row_genres = {
                str(g).strip().lower() for g in json.loads(genres_raw or "[]")
            }
        except Exception:  # noqa: BLE001
            continue
        shared = len(seed_set & row_genres)
        if not shared:
            continue
        scored.append((shared, {
            "id":           row_id,
            "title":        row_name,
            "author":       "",
            "narrator":     "",
            "cover_url":    f"/api/media/cover/{row_id}" if has_cover else "",
            "progress_pct": float(progress_pct or 0),
        }))
    # Stable sort: most shared genres first; recency (scan order) breaks ties.
    scored.sort(key=lambda t: -t[0])
    items = [item for _, item in scored[:limit]]
    return JSONResponse({
        "items":        items,
        "display_name": seed_genres[0],
        "match_key":    "genre",
    })


@router.get("/backdrop/{file_id}")
async def media_backdrop(file_id: str, request: Request):
    """Authenticated proxy for a series/movie backdrop image.

    Jellyfin/Emby expose a separate Backdrop image type (landscape,
    higher-resolution) that makes for a much better detail-page hero
    than the square poster. Silently 404s if the item lacks one so the
    caller can fall back to the poster cover.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await _cached_display_entry(idx, file_id, uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    raw_meta = entry.source_metadata
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    server_id = meta.get("server_id", "")
    external_id = meta.get("external_id", "")
    if not server_id or not external_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "No backdrop"}, status_code=404)
    server = await store.get_visible(server_id, user_id=uid)
    if not server or server.provider not in {"emby", "jellyfin"}:
        return JSONResponse({"error": "Unsupported"}, status_code=404)
    # Ask for a generously-wide backdrop so it looks good on a 4K
    # desktop without being so huge that a phone eats data. maxWidth
    # hints Jellyfin; it down-scales server-side.
    url = (
        f"{server.base_url.rstrip('/')}/Items/{external_id}/Images/Backdrop/0"
        f"?maxWidth=1920&api_key={server.access_token or ''}"
    )
    return StreamingResponse(
        _proxy_cover(http_client, url),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/person/{file_id}/{person_id}/image")
async def media_person_image(file_id: str, person_id: str, request: Request):
    """Authenticated proxy for an actor/director headshot.

    ``file_id`` only supplies server context — the Jellyfin ``person_id``
    is itself a regular item id in the upstream catalog, so the URL
    shape mirrors /cover/ exactly but with the person id substituted.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await _cached_display_entry(idx, file_id, uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    raw_meta = entry.source_metadata
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    server_id = meta.get("server_id", "")
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "No image"}, status_code=404)
    server = await store.get_visible(server_id, user_id=uid)
    if not server or server.provider not in {"emby", "jellyfin"}:
        return JSONResponse({"error": "Unsupported"}, status_code=404)
    url = (
        f"{server.base_url.rstrip('/')}/Items/{person_id}/Images/Primary"
        f"?maxHeight=360&api_key={server.access_token or ''}"
    )
    return StreamingResponse(
        _proxy_cover(http_client, url),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/person/{file_id}/{person_id}")
async def media_person_profile(file_id: str, person_id: str, request: Request):
    """Profile + filmography for an Emby/Jellyfin person.

    Returns a shape the frontend can render without knowing the
    provider: bio, birth/death, birthplace, and a list of works they
    appeared in. Each work is enriched with our internal ``file_id``
    (via file_index lookup) when it's in the local catalog, so clicking
    a work routes into Files instead of an external URL.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    entry = await _cached_display_entry(idx, file_id, uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    raw_meta = entry.source_metadata
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    server_id = meta.get("server_id", "")
    if not server_id or _is_builtin_server(server_id):
        return JSONResponse({"error": "Unsupported"}, status_code=404)
    server = await store.get_visible(server_id, user_id=uid)
    if not server or server.provider not in {"emby", "jellyfin"}:
        return JSONResponse({"error": "Unsupported"}, status_code=404)

    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    base = server.base_url.rstrip("/")
    token = server.access_token or ""
    user_id = await client._resolve_user_id(base, token)
    api_base = client._api_base(base)
    headers = client._headers(token)

    # Person profile
    profile_url = f"{api_base}/Users/{user_id}/Items/{person_id}" if user_id else f"{api_base}/Items/{person_id}"
    try:
        p_resp = await http_client.get(profile_url, headers=headers, timeout=10.0, follow_redirects=True)
        p_body = p_resp.json() if p_resp.status_code == 200 else {}
    except Exception as exc:
        log.debug("person_profile_fetch_failed", error=str(exc))
        p_body = {}

    # Filmography — every Movie/Series this person appears in.
    # Jellyfin/Emby's `Person` param filters by NAME; `PersonIds` filters
    # by the person's item id. We have the id, so we must use PersonIds
    # — passing the uuid to `Person` would match nothing, and Jellyfin
    # silently falls back to unfiltered results (which is what produces
    # "every cast member returns the whole library").
    film_url = f"{api_base}/Users/{user_id}/Items" if user_id else f"{api_base}/Items"
    params = {
        "PersonIds": person_id,
        "IncludeItemTypes": "Movie,Series",
        "Recursive": "true",
        "Fields": "ProductionYear,Overview,SeriesId,ParentIndexNumber,IndexNumber",
        "SortBy": "PremiereDate,SortName",
        "SortOrder": "Descending",
        "Limit": 120,
    }
    if user_id and not client.item_list_path_uses_user:
        params["UserId"] = user_id
    try:
        f_resp = await http_client.get(film_url, headers=headers, params=params, timeout=15.0, follow_redirects=True)
        f_body = f_resp.json() if f_resp.status_code == 200 else {}
    except Exception as exc:
        log.debug("person_filmography_fetch_failed", error=str(exc))
        f_body = {}
    rows = f_body.get("Items") or f_body.get("items") or []

    # Resolve each work to our internal file_id so clicks route back
    # into Files instead of dead-ending at an external provider id.
    works: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        ext = str(row.get("Id") or "").strip()
        if not ext:
            continue
        work_entry = await idx.get_by_source(
            server.provider, f"{server.id}:{ext}", user_id=uid,
        )
        item_type = str(row.get("Type") or "").strip()
        works.append({
            "external_id": ext,
            "file_id": work_entry.id if work_entry else "",
            "name": str(row.get("Name") or "").strip(),
            "year": int(row.get("ProductionYear") or 0) or None,
            "entity_kind": "movie" if item_type == "Movie" else "series",
            "in_library": bool(work_entry),
        })

    premiere = str(p_body.get("PremiereDate") or "")
    end_date = str(p_body.get("EndDate") or "")
    production_locations = p_body.get("ProductionLocations") or []
    birth_place = ""
    if isinstance(production_locations, list) and production_locations:
        birth_place = str(production_locations[0] or "").strip()

    has_image = bool(p_body.get("ImageTags") and (p_body.get("ImageTags") or {}).get("Primary"))

    return JSONResponse({
        "person_id": person_id,
        "name": str(p_body.get("Name") or "").strip() or "Unknown",
        "overview": str(p_body.get("Overview") or "").strip(),
        "birth_date": premiere[:10] if len(premiere) >= 10 else "",
        "death_date": end_date[:10] if len(end_date) >= 10 else "",
        "birth_place": birth_place,
        "has_image": has_image,
        "works": works,
    })


def _cover_etag(file_id: str, updated_at: str, want_full: bool) -> str:
    """ETag for /api/media/cover/{file_id}.

    Strong tag — covers are content-addressable via (file_id + updated_at).
    The ``size`` variant matters because the full JPG and thumbnail are
    different bytes for the same row, and clients may interleave both.
    Length-capped to keep the header small.
    """
    import hashlib as _hashlib
    seed = f"{file_id}\x00{updated_at or ''}\x00{'f' if want_full else 't'}"
    return '"' + _hashlib.sha256(seed.encode()).hexdigest()[:16] + '"'


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    """RFC 7232 If-None-Match comparison. Accepts ``*``, single tag,
    comma list, and weak-prefixed tags (some proxies downgrade)."""
    if not if_none_match:
        return False
    s = if_none_match.strip()
    if s == "*":
        return True
    for token in s.split(","):
        t = token.strip()
        if t.startswith("W/"):
            t = t[2:].strip()
        if t == etag:
            return True
    return False


@router.get("/cover/{file_id}")
async def media_cover(file_id: str, request: Request):
    """Authenticated proxy for a media server's cover art.

    Cached aggressively (immutable by item-id + server) so the grid
    doesn't re-fetch when the user scrolls. Returns 404 silently when
    the row has no cover — the UI falls back to its kind icon.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await _cached_display_entry(idx, file_id, uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    # ETag short-circuit. Cover bytes are effectively immutable for a
    # given (file_id, updated_at, size variant) tuple — the upstream
    # regenerates under a fresh item_id when art is replaced, which bumps
    # updated_at. A revalidating client (any second-render across this
    # device or any other device that already cached) returns 304 with no
    # upstream proxy hit. Computed AFTER the FileEntry lookup so the tag
    # is honest about the row state, but BEFORE the metadata parse +
    # upstream resolution + proxy stream.
    want_full = request.query_params.get("size") == "full"
    etag = _cover_etag(file_id, entry.updated_at, want_full)
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(
            status_code=304,
            headers={
                "ETag": etag,
                "Cache-Control": "private, max-age=3600",
            },
        )

    # FileEntry.source_metadata is already a dict (decoded by _row_to_entry).
    # Guard against the legacy call shape where it might arrive as a JSON
    # string so the endpoint stays robust to callsite drift.
    raw_meta = entry.source_metadata
    if isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        try:
            meta = json.loads(raw_meta or "{}")
        except Exception:
            meta = {}
    server_id = meta.get("server_id", "")
    external_id = meta.get("external_id", "")
    has_cover = bool(meta.get("has_cover") or meta.get("cover_url"))
    if not server_id or not external_id or not has_cover:
        return JSONResponse({"error": "No cover"}, status_code=404)

    upstream_auth = ""
    if _is_builtin_server(server_id):
        # LibriVox-produced cover when available (higher quality than
        # archive.org's services/img guess). For grid rendering the
        # thumbnail suffices; for the detail view we'd prefer the full
        # JPG, but the cover proxy is used for both and the thumbnail is
        # the lighter default. The request can opt into ?size=full to
        # pull the higher-res variant for the detail panel.
        coverart_full = (meta.get("coverart_jpg") or "").strip()
        coverart_thumb = (meta.get("coverart_thumbnail") or "").strip()
        lv_cover = coverart_full if want_full else (coverart_thumb or coverart_full)
        if lv_cover:
            url = lv_cover
        else:
            provider_name = _builtin_provider_name(server_id)
            try:
                client = _provider_client(provider_name, http_client)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            url = client.build_cover_url("", external_id, "")
    else:
        server = await store.get_visible(server_id, user_id=uid)
        # Access token may be empty for providers running in no-auth mode
        # (Suwayomi's ``authMode = none`` is the canonical case). Don't
        # reject those — the upstream /thumbnail endpoint is public there
        # and works without any Authorization header.
        if not server:
            return JSONResponse({"error": "Server unavailable"}, status_code=502)
        try:
            client = _provider_client(server.provider, http_client)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        url, upstream_auth = _resolve_remote_cover_request(
            provider=server.provider,
            client=client,
            base_url=server.base_url,
            access_token=server.access_token or "",
            external_id=external_id,
            cover_hint=str(meta.get("cover_url") or "").strip(),
        )
    return StreamingResponse(
        _proxy_cover(http_client, url, auth_header=upstream_auth),
        media_type="image/jpeg",
        headers={
            # Item covers are effectively immutable (server regenerates
            # under a new item_id when edits happen), so browser caches
            # can hold them for the session. Short s-maxage guards
            # against stale covers when a user does replace art upstream.
            "Cache-Control": "private, max-age=3600",
            "ETag": etag,
        },
    )


# In-flight upstream-push tracker, keyed by (user_id, file_id). The
# UI posts progress every ~5s during playback, so back-to-back POSTs
# from a slow ABS/Emby/Jellyfin upstream would otherwise stack 10s
# timeouts and hold the request open until the upstream replied. By
# cancelling the prior in-flight task when a new push arrives, we
# guarantee:
#
#   - the client gets an immediate 200 (local write is the source of
#     truth for the UI's progress bar anyway)
#   - upstream sees AT MOST one in-flight write per (user, file) at a
#     time, so a wedged upstream can't accumulate dozens of tasks
#   - the most recent position wins — if push #1 was still mid-flight
#     when #2 arrives, #1 is cancelled because #2 carries a newer
#     timestamp, which is what we want to land upstream anyway
_inflight_progress_pushes: dict[tuple[str, str], asyncio.Task] = {}


def _schedule_progress_push(
    user_id: str, file_id: str, coro,
) -> None:
    """Fire-and-forget upstream progress push with per-key supersession.

    Cancels any in-flight push for the same (user_id, file_id), schedules
    a new one, and attaches a done-callback so exceptions don't get
    silently dropped (asyncio.create_task swallows them otherwise).
    Uses ``bg_tasks.track`` so the task is anchored in the project's
    standard background-task registry (GC-safe + in_flight_count
    telemetry).
    """
    from augmentum.utils.bg_tasks import track

    key = (user_id, file_id)
    prior = _inflight_progress_pushes.get(key)
    if prior is not None and not prior.done():
        prior.cancel()

    task = track(coro)
    _inflight_progress_pushes[key] = task

    def _done(t: asyncio.Task) -> None:
        # Only clear if we're still the registered task — a newer
        # push may have already replaced us in the dict.
        if _inflight_progress_pushes.get(key) is t:
            _inflight_progress_pushes.pop(key, None)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            log.warning(
                "media_progress_upstream_push_failed",
                user_id=user_id, file_id=file_id, error=str(exc)[:200],
            )

    task.add_done_callback(_done)


@router.post("/progress/{file_id}")
async def update_progress(
    file_id: str, body: ProgressUpdate, request: Request,
) -> JSONResponse:
    """Push playback position to the upstream server AND cache it locally.

    Both writes matter: upstream so your phone / desktop / whatever other
    ABS client resumes at the right spot; local so the Files panel
    progress bar updates without a catalog re-sync.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    store = _get_store(request)
    http_client = _http(request)
    if not uid or not idx or not store or http_client is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # FileEntry.source_metadata is already a dict (decoded by _row_to_entry).
    # Guard against the legacy call shape where it might arrive as a JSON
    # string so the endpoint stays robust to callsite drift.
    raw_meta = entry.source_metadata
    if isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        try:
            meta = json.loads(raw_meta or "{}")
        except Exception:
            meta = {}
    server_id = meta.get("server_id", "")
    external_id = meta.get("external_id", "")
    if not server_id or not external_id:
        return JSONResponse({"error": "Not a streamable entry"}, status_code=400)

    # Local cache update — always happens, even if upstream push fails.
    # Keeps the UI progress bar accurate during transient ABS outages.
    episode_id = str(body.episode_id or "").strip()
    duration_s = body.duration_s if body.duration_s > 0 else float(meta.get("duration_s") or 0)
    progress_pct = 0.0 if duration_s <= 0 else min(100.0, max(0.0, body.current_time_s / duration_s * 100.0))
    meta["current_time_s"] = body.current_time_s
    meta["duration_s"] = duration_s
    meta["progress_pct"] = progress_pct
    meta["is_finished"] = body.is_finished
    # Stamp the last-read time so the comics `continue` sort can order
    # series by how recently the user actually opened them. Distinct
    # from `fi.updated_at`, which also moves on catalog sync writes.
    meta["last_read_at"] = datetime.now(UTC).isoformat()
    if episode_id:
        meta["selected_episode_id"] = episode_id

    # Comic rows: the reader pushes progress through this same endpoint
    # with ``current_time_s`` carrying the page number (and ``duration_s``
    # the page count). But the reader's ``_extractMeta`` reads page
    # state from ``source_metadata.extra.current_page`` — different
    # field, different nesting. Without this mirror the reader's writes
    # never reach the field its reads consult, so reopening a chapter
    # always landed on page 1 even when the bottom mini-player or rail
    # said "Resume."
    #
    # Also keeps page_count in sync (the reader's duration_s carries
    # it too) so a chapter that grew or shrunk upstream gets reflected
    # without waiting for the next manifest refresh.
    if entry.kind == "comic":
        extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
        extra = dict(extra)  # don't mutate the original block
        page = int(round(body.current_time_s))
        page_count = int(round(duration_s)) if duration_s > 0 else int(extra.get("page_count") or 0)
        if page_count > 0:
            page = max(0, min(page, page_count))
            extra["page_count"] = page_count
        if page > 0:
            extra["current_page"] = page
        extra["is_finished"] = bool(body.is_finished)
        meta["extra"] = extra

    await idx.update_source_metadata(file_id, meta, user_id=uid)
    # Dedicated last_played_at column (migration 195). Set on EVERY
    # progress push regardless of media kind — drives the Continue
    # rail's "most recently played" ordering. Distinct from the
    # ``last_read_at`` JSON field above: the column survives catalog
    # sync, the JSON field doesn't. Belt-and-suspenders is intentional
    # because some surfaces still consult the JSON field, but the rail
    # itself orders by the column.
    try:
        await idx._db.execute(  # noqa: SLF001
            "UPDATE file_index SET last_played_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (file_id, uid),
        )
        await idx._db.commit()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        # Old DBs without the migration won't have the column yet;
        # log and move on rather than failing the whole progress push.
        # The source_metadata-side update above still lands.
        log.warning(
            "last_played_at_update_failed",
            file_id=file_id, error=str(exc),
        )

    # Nudge any of the user's connected receivers (cast-home idle screen,
    # cast-control phone remote that's currently mounted as a receiver
    # surface) to refresh their library so the Continue rail picks up the
    # row we just touched. Debounced per-user / per-cmd at 30s — /progress
    # fires every ~5s during playback and we don't want 12 fanouts/min.
    # Failures are swallowed: the rail still falls back to its own 5-min
    # polling interval so a missed broadcast just delays the update.
    cast_registry = getattr(request.app.state, "receiver_registry", None)
    if cast_registry is not None:
        try:
            from augmentum.cast.receiver_protocol import ReceiverCmd
            await cast_registry.broadcast_debounced(
                uid,
                ReceiverCmd(cmd="library_invalidate"),
                min_interval_s=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "library_invalidate_broadcast_failed",
                file_id=file_id, error=str(exc),
            )

    # Built-in LibriVox: no upstream account to push to. Local write is
    # the only store of record, and it already happened above.
    if _is_builtin_server(server_id):
        return JSONResponse({"status": "ok", "progress_pct": progress_pct})

    server = await store.get_visible(server_id, user_id=uid)
    if not server:
        return JSONResponse({"error": "Server unavailable"}, status_code=502)

    # Borrowed (admin-shared) server: the only credential we hold is the
    # OWNER's, so pushing upstream would write THIS user's playback into
    # the owner's Emby/ABS/Komga account — marking their titles watched
    # and moving their resume points. Same reasoning as the LibriVox
    # branch above: the local write is the store of record, there is just
    # no account of OURS to push to. See MediaServer.is_borrowed_by.
    if server.is_borrowed_by(uid):
        log.debug(
            "media_progress_push_skipped_borrowed_server",
            server_id=server_id, file_id=file_id,
            user_id=uid, owner_id=server.user_id,
        )
        return JSONResponse({
            "status": "ok",
            "progress_pct": progress_pct,
            # Lets the client say "tracked in Augmentum" instead of
            # implying the upstream service was updated.
            "upstream_synced": False,
        })

    try:
        client = _provider_client(server.provider, http_client)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Fire-and-forget upstream push. The local write above already
    # carries the UI's progress bar; the upstream push exists so that
    # other clients (phone, desktop, alternate ABS app) resume at the
    # right spot. A slow or wedged upstream used to block this endpoint
    # for the full HTTP timeout (~10s) on every 5s-poll from the player,
    # which both burned worker capacity and stalled the player's own
    # UI loop. The supersession logic in _schedule_progress_push
    # guarantees AT MOST one in-flight push per (user, file).
    if server.provider == "audiobookshelf":
        push_coro = client.push_progress(
            server.base_url,
            server.access_token,
            external_id=external_id,
            episode_id=episode_id,
            current_time_s=body.current_time_s,
            duration_s=duration_s,
            is_finished=body.is_finished,
        )
    else:
        push_coro = client.push_progress(
            server.base_url, server.access_token,
            external_id=external_id,
            current_time_s=body.current_time_s,
            duration_s=duration_s,
            is_finished=body.is_finished,
        )
    _schedule_progress_push(uid, file_id, push_coro)
    return JSONResponse({"status": "ok", "progress_pct": progress_pct})


# --- Audiobook bookmarks ------------------------------------------------


def _bookmark_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "file_id": row[1],
        "episode_id": row[2] or "",
        "position_s": float(row[3] or 0),
        "label": row[4] or "",
        "note": row[5] or "",
        "created_at": row[6] or "",
    }


_BOOKMARK_COLS = "id, file_id, episode_id, position_s, label, note, created_at"


@router.get("/bookmarks/{file_id}")
async def list_bookmarks(file_id: str, request: Request) -> JSONResponse:
    """List a title's bookmarks for the current user, ordered by position.

    Optional ``?episode_id=`` scopes to a single podcast episode; omit it
    (single-file audiobooks) to get every bookmark on the file.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or idx is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    episode_id = str(request.query_params.get("episode_id") or "").strip()
    sql = (
        f"SELECT {_BOOKMARK_COLS} FROM audiobook_bookmarks "
        "WHERE user_id = ? AND file_id = ?"
    )
    params: list[Any] = [uid, file_id]
    if episode_id:
        sql += " AND episode_id = ?"
        params.append(episode_id)
    sql += " ORDER BY position_s ASC"
    try:
        cursor = await idx._db.execute(sql, tuple(params))  # noqa: SLF001
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("bookmarks_list_failed", file_id=file_id, error=str(exc))
        return JSONResponse({"bookmarks": []})
    return JSONResponse({"bookmarks": [_bookmark_row_to_dict(r) for r in rows]})


@router.post("/bookmarks/{file_id}")
async def create_bookmark(
    file_id: str, body: BookmarkCreate, request: Request,
) -> JSONResponse:
    """Save a bookmark at ``position_s`` (book-level seconds) on a title."""
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or idx is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Confirm the title exists for this user before writing — keeps the
    # table from accumulating orphan rows pointing at nothing.
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    bookmark_id = f"bm_{uuid.uuid4().hex[:16]}"
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    row = (
        bookmark_id, uid, file_id, str(body.episode_id or "").strip(),
        max(0.0, float(body.position_s or 0)),
        (body.label or "").strip()[:200], (body.note or "").strip()[:2000],
        created_at,
    )
    try:
        await idx._db.execute(  # noqa: SLF001
            "INSERT INTO audiobook_bookmarks "
            "(id, user_id, file_id, episode_id, position_s, label, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        await idx._db.commit()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        log.warning("bookmark_create_failed", file_id=file_id, error=str(exc))
        return JSONResponse({"error": "Couldn't save bookmark"}, status_code=500)
    return JSONResponse({
        "bookmark": {
            "id": bookmark_id,
            "file_id": file_id,
            "episode_id": row[3],
            "position_s": row[4],
            "label": row[5],
            "note": row[6],
            "created_at": created_at,
        },
    })


@router.delete("/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: str, request: Request) -> JSONResponse:
    """Delete a bookmark. Scoped to the owning user — a foreign id no-ops."""
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or idx is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        await idx._db.execute(  # noqa: SLF001
            "DELETE FROM audiobook_bookmarks WHERE id = ? AND user_id = ?",
            (bookmark_id, uid),
        )
        await idx._db.commit()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        log.warning("bookmark_delete_failed", bookmark_id=bookmark_id, error=str(exc))
        return JSONResponse({"error": "Couldn't delete bookmark"}, status_code=500)
    return JSONResponse({"status": "ok"})


# --- Browse (LibriVox live catalog) -------------------------------------


# Tiny TTL cache so back-to-back browse calls (pagination clicks, tab
# switches) don't re-hit LibriVox every time. Keyed by (q, category,
# page, page_size) — the `pinned` flag is decorated *after* the cache
# lookup so cross-user cache hits don't leak another user's library
# state. 10 minutes covers typical browse sessions; LibriVox's catalog
# changes slowly enough that this is plenty fresh.
_BROWSE_CACHE_TTL_S = 600
_browse_cache: dict[tuple, tuple[float, list[dict]]] = {}


def _browse_cache_get(key: tuple) -> list[dict] | None:
    entry = _browse_cache.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if time.monotonic() > expires_at:
        _browse_cache.pop(key, None)
        return None
    return payload


def _browse_cache_set(key: tuple, payload: list[dict]) -> None:
    # Bound the cache to ~200 keys so a pathological query stream can't
    # grow it without bound. LRU-ish: when we hit the cap, drop the
    # oldest. Not a security boundary — just hygiene.
    if len(_browse_cache) > 200:
        oldest = min(_browse_cache.items(), key=lambda kv: kv[1][0])
        _browse_cache.pop(oldest[0], None)
    _browse_cache[key] = (time.monotonic() + _BROWSE_CACHE_TTL_S, payload)


@router.get("/browse/librivox")
async def browse_librivox(request: Request) -> JSONResponse:
    """Live search against LibriVox's public feed API.

    Results are ephemeral — they don't land in file_index until the user
    pins one via POST /api/media/pin. Each result carries a ``pinned``
    flag telling the UI whether this user has already added it, so the
    browse grid can render the right action button without a second
    round-trip.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    http_client = _http(request)
    if http_client is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    qp = request.query_params
    query = (qp.get("q") or "").strip()
    category = (qp.get("category") or "").strip()
    # ``recent=1`` → fetch LibriVox's "cataloged since N days ago" feed
    # instead of archive.org search. Used by the Catalog overlay as its
    # landing view (so the first thing users see is "what's new on
    # LibriVox" rather than an alphabetical firstpage). ``q``/``category``
    # take precedence — a search always supersedes the landing view.
    recent_mode = qp.get("recent") == "1" and not query and not category
    try:
        page = max(1, int(qp.get("page", "1")))
        page_size = max(1, min(48, int(qp.get("page_size", "24"))))
    except ValueError:
        return JSONResponse({"error": "Invalid pagination"}, status_code=400)

    # recent-mode cache key is distinct so a "popular first-page" result
    # for ("", "", 1, 24) can't accidentally serve as the recent feed.
    cache_key: tuple
    if recent_mode:
        cache_key = ("__recent__", page_size)
    else:
        cache_key = (query.lower(), category.lower(), page, page_size)
    cached = _browse_cache_get(cache_key)
    if cached is not None:
        results = cached
    else:
        provider = LibrivoxProvider(http_client)
        if not provider_supports_browse(provider):   # belt-and-braces
            return JSONResponse({"error": "browse unsupported"}, status_code=500)
        if recent_mode:
            # 30 days is wide enough to reliably produce a full grid
            # (LibriVox typically publishes 3-5 books/day) without
            # stretching into stale territory.
            hits = await provider.recently_added(days=30, limit=page_size)
        else:
            hits = await provider.browse(
                query=query, category=category, page=page, page_size=page_size,
            )
        results = [h.to_dict() for h in hits]
        _browse_cache_set(cache_key, results)

    # Decorate with per-user `pinned` flag AFTER cache lookup so the
    # cached payload stays user-agnostic. One SQL query resolves all
    # page items at once — O(1) round trips regardless of page size.
    idx = _get_index(request)
    pinned_map: dict[str, str] = {}
    if idx and results:
        external_ids = [r["external_id"] for r in results if r.get("external_id")]
        if external_ids:
            placeholders = ",".join("?" * len(external_ids))
            source_ids = [f"{BUILTIN_LIBRIVOX}:{eid}" for eid in external_ids]
            cursor = await idx._db.execute(
                f"SELECT id, source_id FROM file_index "
                f"WHERE user_id = ? AND source = ? "
                f"  AND source_id IN ({placeholders}) AND is_trashed = 0",
                [uid, "librivox", *source_ids],
            )
            for row in await cursor.fetchall():
                row_id, row_source_id = row
                # Strip sentinel prefix to map back to external_id.
                if row_source_id.startswith(f"{BUILTIN_LIBRIVOX}:"):
                    external_id = row_source_id[len(BUILTIN_LIBRIVOX) + 1:]
                    pinned_map[external_id] = row_id

    decorated = []
    for r in results:
        entry = dict(r)
        pinned_file_id = pinned_map.get(r.get("external_id", ""))
        entry["pinned"] = pinned_file_id is not None
        entry["pinned_file_id"] = pinned_file_id or None
        decorated.append(entry)

    # Recent-mode is a fixed-window snapshot (since=<30 days ago>), so
    # "load more" doesn't make sense — paging the since feed would just
    # re-request the same window. Force has_more=False and let the UI
    # drop the button.
    has_more = False if recent_mode else (len(results) >= page_size)

    return JSONResponse({
        "results":    decorated,
        "page":       page,
        "page_size":  page_size,
        "has_more":   has_more,
        "recent":     recent_mode,
    })


# --- Pin / unpin (built-in catalogs → file_index promotion) -------------


class PinRequest(BaseModel):
    provider:    str
    external_id: str
    name:        str
    author:      str = ""
    narrator:    str = ""
    description: str = ""
    cover_url:   str = ""
    duration_ms: int = 0
    license:     str = ""
    extra:       dict = Field(default_factory=dict)


@router.post("/pin")
async def pin_item(body: PinRequest, request: Request) -> JSONResponse:
    """Promote a browse result into the user's file_index.

    Currently only supports LibriVox — the shape is kept provider-neutral
    so a future Gutenberg/Project-whatever built-in can reuse it. Idempotent:
    re-pinning the same external_id returns the existing file_id instead
    of erroring.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    idx = _get_index(request)
    if not idx:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    http_client = _http(request)
    if http_client is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    if body.provider != "librivox":
        return JSONResponse(
            {"error": f"Pin unsupported for provider: {body.provider}"},
            status_code=400,
        )
    if not body.external_id or not body.name:
        return JSONResponse(
            {"error": "external_id and name are required"}, status_code=400,
        )

    source_id = f"{BUILTIN_LIBRIVOX}:{body.external_id}"

    # Idempotency: if the user already pinned this exact item, short-circuit.
    existing = await idx.get_by_source("librivox", source_id, user_id=uid)
    if existing is not None:
        return JSONResponse({"file_id": existing.id, "already_pinned": True})

    # Build a synthetic BrowseResult from the request body first — both
    # the primary (sections) and fallback (archive.org) normalisers need it.
    browse_result = BrowseResult(
        external_id=body.external_id,
        name=body.name,
        author=body.author,
        narrator=body.narrator,
        duration_ms=body.duration_ms,
        cover_url=body.cover_url or f"{ARCHIVE_COVER}/{body.external_id}",
        description=body.description,
        license=body.license or "public-domain",
        extra=body.extra or {},
    )

    provider = LibrivoxProvider(http_client)

    # Primary path: pull the canonical record from LibriVox (one call) and
    # build chapters directly from sections[]. Gets us per-chapter readers
    # and chapter-range-aware titles that archive.org's file listing
    # can't offer.
    detail: dict | None = None
    librivox_id = (body.extra or {}).get("librivox_id") or ""
    if librivox_id:
        lv_book = await provider.fetch_book_by_id(str(librivox_id))
        if lv_book is not None:
            candidate = normalise_librivox_sections(
                librivox_book=lv_book, browse_result=browse_result,
            )
            if candidate.get("audio_files"):
                detail = candidate

    # Fallback: some LibriVox rows don't have sections populated (old
    # catalog entries), or the client didn't forward a librivox_id.
    # archive.org /metadata gives us the raw file list to reconstruct
    # chapters — less rich (no per-chapter readers) but playable.
    if detail is None:
        archive_meta = await provider.fetch_item_details(
            "", "", external_id=body.external_id,
        )
        if archive_meta is None:
            return JSONResponse(
                {"error": "Could not resolve book metadata — neither "
                          "LibriVox sections nor archive.org was reachable"},
                status_code=502,
            )
        detail = normalise_details_to_catalog(
            archive_meta=archive_meta, browse_result=browse_result,
        )
    if not detail.get("audio_files"):
        return JSONResponse(
            {"error": "No playable MP3 chapters found for this item"},
            status_code=422,
        )

    duration_ms = int(detail["duration_ms"] or body.duration_ms or 0)
    duration_s = duration_ms / 1000.0 if duration_ms else 0.0

    # Compose the same source_metadata shape that ABS sync produces so the
    # rest of the media routes (details, cover, progress, stream) don't
    # need provider-specific branches beyond the sentinel check.
    source_metadata = {
        "server_id":           BUILTIN_LIBRIVOX,
        "provider":            "librivox",
        "external_id":         body.external_id,
        "stream_path":         "",   # sentinel: route uses ?file=<idx> instead
        "archive_identifier":  detail["archive_identifier"],
        "audio_files":         detail["audio_files"],
        "chapters":            detail["chapters"],
        "narrators":           detail.get("narrators") or [],
        "duration_ms":         duration_ms,
        "duration_s":          duration_s,
        "progress_pct":        0.0,
        "current_time_s":      0.0,
        "is_finished":         False,
        "cover_url":           browse_result.cover_url,
        "has_cover":           bool(browse_result.cover_url),
        "author":              body.author,
        "narrator":            detail["narrator"],
        "author_normalized":   normalize_name(body.author),
        "narrator_normalized": normalize_name(detail["narrator"]),
        "description":         body.description,
        "license":             browse_result.license,
        "genres":              detail["genres"],
        "language":            detail["language"],
        "librivox_url":        detail["librivox_url"],
        "librivox_id":         detail["librivox_id"],
        # Enrichment fields populated from the LibriVox feed at pin time —
        # let the detail panel render author life dates, year, text source
        # link, etc. without another round trip.
        "authors_detailed":    detail.get("authors_detailed") or [],
        "copyright_year":      detail.get("copyright_year", ""),
        "totaltime":           detail.get("totaltime", ""),
        "url_text_source":     detail.get("url_text_source", ""),
        "url_project":         detail.get("url_project", ""),
        "url_rss":             detail.get("url_rss", ""),
        "url_zip_file":        detail.get("url_zip_file", ""),
        # LibriVox-produced cover art (empty when the book has none —
        # cover proxy falls back to archive.org/services/img in that case).
        "coverart_jpg":        detail.get("coverart_jpg", ""),
        "coverart_thumbnail":  detail.get("coverart_thumbnail", ""),
        "extra":               {
            k: v for k, v in (body.extra or {}).items()
            if k not in ("chapters", "audio_files", "archive_identifier")
        },
    }

    file_id = await register_file(
        user_id=uid,
        source="librivox",
        source_id=source_id,
        name=body.name,
        mime_type="audio/mpeg",
        size_bytes=int(detail.get("size_bytes") or 0),
        real_path=None,
        description=body.description,
        thumbnail=None,
        source_metadata=source_metadata,
    )
    if not file_id:
        return JSONResponse(
            {"error": "Failed to register item — file index unavailable"},
            status_code=500,
        )

    log.info(
        "librivox_pinned", user_id=uid, external_id=body.external_id,
        file_id=file_id, chapters=len(detail["chapters"]),
    )

    # Fire-and-forget: enqueue a Gutenberg text fetch when the LibriVox
    # feed carries a usable source URL. The job runs asynchronously —
    # the pin response returns immediately so the UI isn't blocked on a
    # multi-second download. Progress lands on the file_index row's
    # source_metadata (gutenberg_status: fetching → fetched/unavailable).
    text_source = source_metadata.get("url_text_source") or ""
    if text_source and "gutenberg.org" in text_source.lower():
        jobs_store = getattr(request.app.state, "jobs_store", None)
        if jobs_store is not None:
            try:
                await jobs_store.create(
                    user_id=uid,
                    job_type="gutenberg_fetch",
                    payload={
                        "file_id": file_id,
                        "url_text_source": text_source,
                    },
                )
                runner = getattr(request.app.state, "job_runner", None)
                if runner is not None:
                    runner.wake()
            except Exception:
                # Don't fail the pin if enqueue hiccups — the user
                # already has the audiobook, the text is a bonus we'll
                # catch on a re-pin or a manual retrigger later.
                log.warning(
                    "gutenberg_fetch_enqueue_failed",
                    file_id=file_id, exc_info=True,
                )

    return JSONResponse({"file_id": file_id, "already_pinned": False})


@router.delete("/pin/{file_id}")
async def unpin_item(file_id: str, request: Request) -> JSONResponse:
    """Remove a pinned built-in library item from the user's index.

    Only targets rows with source='librivox' + sentinel server_id.
    User-owned ABS items go through the regular file-trash flow.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    idx = _get_index(request)
    if not idx:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    if meta.get("server_id") != BUILTIN_LIBRIVOX:
        return JSONResponse(
            {"error": "Unpin only applies to built-in library items"},
            status_code=400,
        )

    ok = await unregister_file(entry.source, entry.source_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "Delete failed"}, status_code=500)
    return JSONResponse({"status": "unpinned"})


@router.get("/gutenberg-text/{file_id}")
async def get_gutenberg_text(file_id: str, request: Request):
    """Return the fetched Project Gutenberg plaintext for a pinned book.

    Scoped to the requesting user — ``file_index.get`` already enforces
    ``user_id`` equality, so a cross-tenant ``file_id`` probe returns
    404. The blob itself lives at ``source_metadata.gutenberg_path``,
    written by the ``gutenberg_fetch`` background job.

    Returns:
      * 200 plaintext when the row is fetched and the blob is present
      * 404 when the file_id is missing, unpinned, or belongs to
        another user
      * 202 when the fetch is still pending (gives the UI a cue to
        poll instead of failing)
      * 410 when the source is permanently unavailable
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    idx = _get_index(request)
    if idx is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    entry = await idx.get(file_id, user_id=uid)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    status = meta.get("gutenberg_status", "")
    if status == "unavailable":
        return JSONResponse(
            {"error": "No Gutenberg source available for this recording",
             "reason": meta.get("gutenberg_error", "")},
            status_code=410,
        )
    if status != "fetched":
        # Pending or never-enqueued. Either way the UI should check
        # back — we don't return a body.
        return JSONResponse(
            {"status": status or "pending"}, status_code=202,
        )

    path_str = str(meta.get("gutenberg_path") or "")
    if not path_str:
        return JSONResponse({"error": "Fetched but no path stored"}, status_code=500)
    path = Path(path_str)
    if not path.is_file():
        # Blob went missing (manual cleanup, disk wipe, container
        # reset) — flag the row so a future re-fetch can recover it
        # instead of continuing to lie about having the text.
        fallback_meta = dict(meta)
        fallback_meta["gutenberg_status"] = "missing"
        try:
            await idx.update_source_metadata(file_id, fallback_meta, user_id=uid)
        except Exception:
            log.warning("gutenberg_text_missing_mark_failed", exc_info=True)
        return JSONResponse({"error": "Text blob missing on disk"}, status_code=410)

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("gutenberg_text_read_failed", file_id=file_id, err=str(exc))
        return JSONResponse({"error": "Could not read text"}, status_code=500)

    # UTF-8 plaintext. Caller is responsible for paginating / rendering;
    # we return the full body so the UI can do offline search, chapter
    # split, etc., without re-fetching.
    return PlainTextResponse(content)


# --- Proxy helpers -------------------------------------------------------


async def _proxy_cover(
    client: httpx.AsyncClient, url: str, *, auth_header: str = "",
):
    """Stream bytes from an upstream cover URL.

    ``auth_header`` is forwarded verbatim as the ``Authorization`` header
    when non-empty — required for Komga and Basic-auth Suwayomi
    deployments where the thumbnail endpoint is gated. Empty string ==
    no-auth (LibriVox, Suwayomi with ``authMode=none``) so we send a
    bare request.
    """
    headers = {"Authorization": auth_header} if auth_header else {}
    try:
        async with client.stream(
            "GET", url, headers=headers, timeout=15.0, follow_redirects=True,
        ) as upstream:
            if upstream.status_code != 200:
                # Log the failure so a broken cover doesn't look like a
                # frontend bug. 404 is legitimately common (provider has
                # no cover for this item); log those quietly at debug.
                (log.debug if upstream.status_code == 404 else log.warning)(
                    "media_cover_proxy_non_200",
                    url=url, status=upstream.status_code,
                )
                return
            async for chunk in upstream.aiter_bytes(chunk_size=32768):
                yield chunk
    except Exception as exc:
        log.debug("media_cover_proxy_failed", url=url, error=str(exc))


# Response headers we forward from upstream verbatim. Everything here is
# either byte-range-critical (without these the browser can't seek) or
# freshness metadata (last-modified / etag) that the browser uses for
# conditional revalidation. We intentionally do NOT forward cookies,
# server-identifying headers, or upstream's own auth/session tokens.
_STREAM_FORWARD_RESP_HEADERS = (
    "content-range",   # required for 206 seek responses
    "content-length",  # required so <audio> can show correct duration
    "accept-ranges",   # advertises seekability to the browser
    "content-type",    # e.g. audio/mpeg; default would be generic
    "last-modified",
    "etag",
)


async def _open_range_proxy(
    client: httpx.AsyncClient, url: str, headers: dict[str, str],
) -> StreamingResponse:
    """Forward a Range-aware stream request, status + headers intact.

    The handler must ``await`` this so upstream's headers are available
    before we construct the response. Ownership of the httpx stream
    transfers to the body generator; it's closed in the generator's
    finally block when the client disconnects or the stream ends.

    Returns a 502 JSON envelope on network errors so the ``<audio>``
    element surfaces a failure rather than hanging.
    """
    # follow_redirects: reverse proxies often 308 http→https or
    # slash-normalise; without this, media silently stalls with a
    # non-200 before ever hitting the real bytes.
    try:
        request = client.build_request(
            "GET", url, headers=headers, timeout=None,
        )
        upstream = await client.send(
            request, stream=True, follow_redirects=True,
        )
    except Exception as exc:
        log.warning("media_stream_proxy_open_failed", url=url, error=str(exc))
        return JSONResponse(
            {"error": "Upstream unavailable"}, status_code=502,
        )

    if upstream.status_code >= 400:
        # Upstream error — read the short body (if any), close the
        # connection, and surface the status so the <audio> element
        # stops trying instead of silently stalling.
        try:
            body = await upstream.aread()
        except Exception:
            body = b""
        await upstream.aclose()
        return StreamingResponse(
            iter([body or f"Upstream {upstream.status_code}".encode()]),
            status_code=upstream.status_code,
            media_type="text/plain",
        )

    fwd_headers: dict[str, str] = {}
    for h in _STREAM_FORWARD_RESP_HEADERS:
        v = upstream.headers.get(h)
        if v is not None:
            fwd_headers[h] = v

    async def _body():
        try:
            async for chunk in upstream.aiter_bytes(chunk_size=65536):
                yield chunk
        except Exception as exc:
            log.warning("media_stream_proxy_failed", url=url, error=str(exc))
        finally:
            # aclose is idempotent and must run even on client disconnect —
            # otherwise we'd leak an httpx stream per interrupted seek.
            try:
                await upstream.aclose()
            except Exception as exc:
                log.debug("media_proxy_upstream_aclose_failed", url=url, error=str(exc))

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,
        headers=fwd_headers,
    )
