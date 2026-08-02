"""Outbound TTS + STT client for cross-peer audio dispatch.

Sibling of ``image_client.py``, ``knowledge_client.py``, ``render_client.py``
— same shape, same trust model, same signing. Pre-extraction the TTS
and STT fabric dispatch lived inside route handlers in
``augmentum/proxy/audio_routes.py`` (and STT also in
``augmentum/voice/pipeline.py``), making audio the one modality where
the cross-peer dispatch wasn't isolated. The architecture review
flagged this as the only structural inconsistency among the six
modality clients; this module closes it.

The functions here POST to the receiver's dedicated fabric data-plane
endpoints (``/api/fabric/tts``, ``/api/fabric/stt``) with signed
envelopes. The receiver's :class:`FabricPeerMiddleware` verifies the
signature, sets ``scope["fabric_peer"]``, and the
``fabric_tts`` / ``fabric_stt`` route handlers do the rest (recursion
guard against fabric→fabric loops, delegation to the local synth path).

See :mod:`augmentum.fabric.pair_client` for the trust-model writeup
explaining why we accept self-signed Caddy certs (peer identity is
ed25519, not the TLS chain).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


# TTS streams run for the duration of synthesis (a few seconds for a
# sentence; tens of seconds for a paragraph). The ceiling has to cover the
# slowest realistic case: a long reply on a slow GPU (e.g. CSM running eager
# on a Turing card), where generation can crawl. Default 300s, overridable
# via AUGMENTUM_FABRIC_TTS_TIMEOUT_S; httpx manages per-chunk inner timeouts.
_TTS_STREAM_TIMEOUT_S = float(os.environ.get("AUGMENTUM_FABRIC_TTS_TIMEOUT_S", "300"))

# STT is a single request/response — audio file in, transcript out.
# Most files are <30s of audio = ~3-5s of inference. 60s leaves
# headroom for slow peers / large uploads.
_STT_REQUEST_TIMEOUT_S = 60.0


class RemoteAudioError(RuntimeError):
    """Raised when cross-peer audio dispatch fails. Subclasses
    RuntimeError so existing ``except Exception`` handlers in the
    audio pipeline catch it transparently. Distinct type lets callers
    that care (the route layer) tell ``peer dropped`` apart from
    other audio failures.
    """


# ── TTS (synthesis) ──────────────────────────────────────────────


async def tts_stream_via_peer(
    *,
    http_client_factory,
    identity: "FabricIdentity",
    user_id: str,
    peer_base_url: str,
    payload: dict[str, Any],
    session_id: str = "",
) -> AsyncIterator[bytes]:
    """Stream synthesized audio bytes from a fabric peer.

    Builds the signed envelope, POSTs to the peer's
    ``/api/fabric/tts``, and yields the response body in chunks
    suitable for piping back to the user's audio player. Raises
    :class:`RemoteAudioError` on any precondition or transport
    failure so the caller can surface a clean error rather than
    returning a silent empty generator (the 2026-05-23 incident's
    failure mode).

    ``http_client_factory`` is a callable that returns an
    ``httpx.AsyncClient`` context manager configured for the peer's
    base URL — usually ``audio_routes._audio_client(peer_base_url)``.
    Passed in (rather than constructed here) so this module stays
    independent of the audio-routes-specific client config (retry
    policy, connect timeout, etc.).

    ``payload`` is the OpenAI-compat TTS request body
    (``{"model": ..., "input": ..., "voice": ..., ...}``).

    ``session_id`` is folded into the body (not a header) so it survives
    the envelope signing + the receiver's body-only reconstruction. The
    receiver re-attaches it as ``X-Augmentum-Session`` to the local
    provider, which is how context-aware engines (Sesame CSM) keep their
    conversational prosody working across the fabric.
    """
    if identity is None or not user_id:
        raise RemoteAudioError(
            f"fabric TTS dispatch requires identity + user_id "
            f"(has_identity={identity is not None}, has_user={bool(user_id)})"
        )

    if session_id:
        payload = {**payload, "session_id": session_id}
    body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path = "/api/fabric/tts"

    from augmentum.fabric.peer_middleware import build_signed_peer_headers
    headers = build_signed_peer_headers(
        identity=identity, user_id=user_id,
        method="POST", path=path, body=body_bytes,
    )
    headers["Content-Type"] = "application/json"

    try:
        async with http_client_factory(peer_base_url) as client:
            async with client.stream(
                "POST",
                f"{peer_base_url}{path}",
                content=body_bytes,
                headers=headers,
                timeout=_TTS_STREAM_TIMEOUT_S,
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    yield chunk
    except httpx.HTTPStatusError as exc:
        log.warning(
            "fabric_tts_upstream_status",
            peer=peer_base_url,
            status=exc.response.status_code if exc.response else 0,
        )
        raise RemoteAudioError(
            f"fabric TTS peer returned {exc.response.status_code if exc.response else 0}"
        ) from exc
    except httpx.TransportError as exc:
        log.warning(
            "fabric_tts_transport_error",
            peer=peer_base_url, error=str(exc)[:160],
        )
        raise RemoteAudioError(
            f"fabric TTS peer unreachable: {str(exc)[:160]}"
        ) from exc


# ── STT (transcription) ──────────────────────────────────────────


async def stt_transcribe_via_peer(
    *,
    http_client_factory,
    identity: "FabricIdentity",
    user_id: str,
    peer_base_url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str = "",
    language: str = "",
    response_format: str = "json",
) -> str:
    """Transcribe an audio clip on a fabric peer + return the text.

    The standard ``files=files_data`` httpx call would build a
    multipart body internally with a per-call random boundary, which
    we'd have no way to hash for the signature. Instead we build the
    multipart body ourselves with a known boundary, hash it, sign
    it, and send via ``content=`` so the bytes on the wire exactly
    match what we signed.

    Raises :class:`RemoteAudioError` on any failure (caller catches
    + surfaces — STT failures should not be silent).
    """
    if identity is None or not user_id:
        raise RemoteAudioError(
            f"fabric STT dispatch requires identity + user_id "
            f"(has_identity={identity is not None}, has_user={bool(user_id)})"
        )

    boundary = f"----augmentumstt{uuid.uuid4().hex}"
    body_bytes = _build_multipart_body(
        boundary=boundary,
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        text_fields={
            "model": model,
            "language": language,
            "response_format": response_format,
        },
    )

    path = "/api/fabric/stt"
    from augmentum.fabric.peer_middleware import build_signed_peer_headers
    fabric_headers = build_signed_peer_headers(
        identity=identity, user_id=user_id,
        method="POST", path=path, body=body_bytes,
    )
    # Drop any Authorization that a parent header builder may have
    # added (no API key on peer hops) and overlay the multipart
    # content-type with our known boundary.
    fabric_headers = {
        k: v for k, v in fabric_headers.items()
        if k.lower() != "authorization"
    }
    fabric_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    try:
        async with http_client_factory(peer_base_url) as client:
            resp = await client.post(
                f"{peer_base_url}{path}",
                content=body_bytes,
                headers=fabric_headers,
                timeout=_STT_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning(
            "fabric_stt_upstream_status",
            peer=peer_base_url,
            status=exc.response.status_code if exc.response else 0,
        )
        raise RemoteAudioError(
            f"fabric STT peer returned {exc.response.status_code if exc.response else 0}"
        ) from exc
    except httpx.TransportError as exc:
        log.warning(
            "fabric_stt_transport_error",
            peer=peer_base_url, error=str(exc)[:160],
        )
        raise RemoteAudioError(
            f"fabric STT peer unreachable: {str(exc)[:160]}"
        ) from exc

    if isinstance(data, dict):
        return data.get("text", "") or ""
    return str(data)


async def clone_upload_via_peer(
    *,
    http_client_factory,
    identity: "FabricIdentity",
    user_id: str,
    peer_base_url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    voice_name: str,
    transcript: str = "",
) -> bool:
    """Push a voice-clone reference (clip + transcript) to a fabric peer.

    Context-aware engines like Sesame CSM clone from a ``(text, audio)``
    anchor, so the transcript travels with the clip. A *local* CSM picks
    up clones from the shared ``/voices`` volume the main app writes to —
    but a *remote* peer has its own volume, so the bytes have to cross the
    wire. The receiver (``fabric_routes.fabric_voice_clone``) writes them
    into ITS local voice dir, where its own CSM sidecar then finds them —
    same mechanism as local, just bridged.

    Mirrors :func:`stt_transcribe_via_peer`'s known-boundary multipart so
    the signed bytes match the wire exactly. Returns True on a 2xx;
    raises :class:`RemoteAudioError` on transport/status failure.
    """
    if identity is None or not user_id:
        raise RemoteAudioError(
            f"fabric voice-clone dispatch requires identity + user_id "
            f"(has_identity={identity is not None}, has_user={bool(user_id)})"
        )

    boundary = f"----augmentumclone{uuid.uuid4().hex}"
    body_bytes = _build_multipart_body(
        boundary=boundary,
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        text_fields={
            "voice_name": voice_name,
            "transcript": transcript,
        },
    )

    path = "/api/fabric/voice-clone"
    from augmentum.fabric.peer_middleware import build_signed_peer_headers
    fabric_headers = build_signed_peer_headers(
        identity=identity, user_id=user_id,
        method="POST", path=path, body=body_bytes,
    )
    fabric_headers = {
        k: v for k, v in fabric_headers.items()
        if k.lower() != "authorization"
    }
    fabric_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    try:
        async with http_client_factory(peer_base_url) as client:
            resp = await client.post(
                f"{peer_base_url}{path}",
                content=body_bytes,
                headers=fabric_headers,
                timeout=_STT_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.warning(
            "fabric_voice_clone_upstream_status",
            peer=peer_base_url,
            status=exc.response.status_code if exc.response else 0,
        )
        raise RemoteAudioError(
            f"fabric voice-clone peer returned "
            f"{exc.response.status_code if exc.response else 0}"
        ) from exc
    except httpx.TransportError as exc:
        log.warning(
            "fabric_voice_clone_transport_error",
            peer=peer_base_url, error=str(exc)[:160],
        )
        raise RemoteAudioError(
            f"fabric voice-clone peer unreachable: {str(exc)[:160]}"
        ) from exc
    return True


async def push_user_context_via_peer(
    *,
    http_client_factory,
    identity: "FabricIdentity",
    user_id: str,
    peer_base_url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    session_id: str,
    transcript: str = "",
) -> bool:
    """Push the USER's spoken turn (clip + transcript) to a fabric peer's
    CSM so its cross-speaker context conditions her next reply's prosody on
    how the user actually sounded.

    Unlike the clone bridge, this context lives in the remote *sidecar's*
    RAM, not on disk — so the receiver (``fabric_routes.fabric_user_context``)
    forwards it to its local sidecar rather than writing a file. ``session_id``
    travels as a form field and is re-attached as ``X-Augmentum-Session`` on
    that forward. Best-effort: raises :class:`RemoteAudioError` so the caller
    can swallow it without breaking the voice turn.
    """
    if identity is None or not user_id:
        raise RemoteAudioError(
            f"fabric user-context dispatch requires identity + user_id "
            f"(has_identity={identity is not None}, has_user={bool(user_id)})"
        )

    boundary = f"----augmentumusrctx{uuid.uuid4().hex}"
    body_bytes = _build_multipart_body(
        boundary=boundary,
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        text_fields={
            "session_id": session_id,
            "transcript": transcript,
        },
    )

    path = "/api/fabric/tts/user-context"
    from augmentum.fabric.peer_middleware import build_signed_peer_headers
    fabric_headers = build_signed_peer_headers(
        identity=identity, user_id=user_id,
        method="POST", path=path, body=body_bytes,
    )
    fabric_headers = {
        k: v for k, v in fabric_headers.items()
        if k.lower() != "authorization"
    }
    fabric_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    try:
        async with http_client_factory(peer_base_url) as client:
            resp = await client.post(
                f"{peer_base_url}{path}",
                content=body_bytes,
                headers=fabric_headers,
                timeout=_STT_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RemoteAudioError(
            f"fabric user-context peer returned "
            f"{exc.response.status_code if exc.response else 0}"
        ) from exc
    except httpx.TransportError as exc:
        raise RemoteAudioError(
            f"fabric user-context peer unreachable: {str(exc)[:160]}"
        ) from exc
    return True


async def _signed_peer_post(
    *, http_client_factory, identity, user_id: str, peer_base_url: str,
    path: str, body: dict | None = None,
) -> bool:
    """Tiny signed JSON POST to a peer data-plane endpoint (no streaming, no
    multipart). Used by the CSM residency pings (/warmup, /unload). Raises
    :class:`RemoteAudioError` on failure so callers can swallow it."""
    if identity is None or not user_id:
        raise RemoteAudioError(
            f"fabric peer post requires identity + user_id "
            f"(has_identity={identity is not None}, has_user={bool(user_id)})"
        )
    body_bytes = json.dumps(body or {}, separators=(",", ":")).encode("utf-8")
    from augmentum.fabric.peer_middleware import build_signed_peer_headers
    headers = build_signed_peer_headers(
        identity=identity, user_id=user_id,
        method="POST", path=path, body=body_bytes,
    )
    headers["Content-Type"] = "application/json"
    try:
        async with http_client_factory(peer_base_url) as client:
            resp = await client.post(
                f"{peer_base_url}{path}", content=body_bytes,
                headers=headers, timeout=15.0,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RemoteAudioError(
            f"fabric peer {path} returned "
            f"{exc.response.status_code if exc.response else 0}"
        ) from exc
    except httpx.TransportError as exc:
        raise RemoteAudioError(f"fabric peer {path} unreachable: {str(exc)[:160]}") from exc
    return True


async def warmup_via_peer(*, http_client_factory, identity, user_id: str,
                          peer_base_url: str) -> bool:
    """Ask a peer's CSM sidecar to pre-load (conversation-scoped residency)."""
    return await _signed_peer_post(
        http_client_factory=http_client_factory, identity=identity,
        user_id=user_id, peer_base_url=peer_base_url,
        path="/api/fabric/tts/warmup",
    )


async def unload_via_peer(*, http_client_factory, identity, user_id: str,
                          peer_base_url: str, session_id: str) -> bool:
    """Ask a peer's CSM sidecar to release VRAM + clear this session's
    cross-speaker context (the conversation ended)."""
    return await _signed_peer_post(
        http_client_factory=http_client_factory, identity=identity,
        user_id=user_id, peer_base_url=peer_base_url,
        path="/api/fabric/tts/unload", body={"session_id": session_id},
    )


def _build_multipart_body(
    *,
    boundary: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    text_fields: dict[str, str],
) -> bytes:
    """Build the multipart/form-data body with a known boundary.

    Extracted so the bytes we sign exactly match what we send. Mirror
    of the inline construction that previously lived in
    audio_routes.py; pulled here so signing + transport happen in
    one place.
    """
    parts: list[bytes] = []

    # file part
    parts.append(f"--{boundary}\r\n".encode("latin-1"))
    parts.append(
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\n'.encode("latin-1")
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("latin-1"))
    parts.append(audio_bytes)
    parts.append(b"\r\n")

    # text fields (skip empties — the receiver applies its own defaults)
    for field_name, field_value in text_fields.items():
        if not field_value:
            continue
        parts.append(f"--{boundary}\r\n".encode("latin-1"))
        parts.append(
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            .encode("latin-1")
        )
        parts.append(str(field_value).encode("utf-8"))
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode("latin-1"))
    return b"".join(parts)
