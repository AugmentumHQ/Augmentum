"""Tests for augmentum/fabric/audio_client.py.

Mirrors the shape of the existing knowledge_client / image_client /
render_client tests in test_fabric_modality_endpoints.py. Pins the
load-bearing contracts:

  * Refuses dispatch without identity / user_id (loud failure vs the
    pre-2026-05-23 silent empty-generator that left the user staring
    at dead audio).
  * Signs every request body (peer's middleware verifies).
  * Maps httpx transport / status errors to a typed RemoteAudioError
    so the route layer can surface a clean 5xx instead of bubbling
    a generic exception.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ── Helpers ───────────────────────────────────────────────────────


def _fake_identity():
    """Build a real FabricIdentity for signing tests."""
    import asyncio
    import aiosqlite

    async def _make():
        from augmentum.fabric.identity import FabricIdentity
        from augmentum.state.settings_store import SettingsStore
        conn = await aiosqlite.connect(":memory:")
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        return identity, conn

    return asyncio.get_event_loop().run_until_complete(_make())


# ── TTS ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tts_raises_when_identity_missing():
    """Pre-2026-05-23 the missing-credentials path returned a silent
    empty generator; the user heard nothing. Helper raises typed error
    so the caller can convert to a 503 instead."""
    from augmentum.fabric.audio_client import (
        RemoteAudioError, tts_stream_via_peer,
    )

    gen = tts_stream_via_peer(
        http_client_factory=MagicMock(),
        identity=None,
        user_id="u",
        peer_base_url="https://peer.local",
        payload={"input": "hello"},
    )
    with pytest.raises(RemoteAudioError) as excinfo:
        async for _ in gen:
            pass
    assert "identity" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_tts_raises_when_user_id_empty():
    from augmentum.fabric.audio_client import (
        RemoteAudioError, tts_stream_via_peer,
    )

    gen = tts_stream_via_peer(
        http_client_factory=MagicMock(),
        identity=MagicMock(),
        user_id="",
        peer_base_url="https://peer.local",
        payload={"input": "hello"},
    )
    with pytest.raises(RemoteAudioError):
        async for _ in gen:
            pass


@pytest.mark.asyncio
async def test_tts_streams_bytes_back_with_signed_headers():
    """Happy path: opens an httpx stream to /api/fabric/tts with a
    signed envelope and yields each upstream chunk."""
    import aiosqlite
    from augmentum.fabric.audio_client import tts_stream_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        # Fake upstream that yields three byte chunks.
        upstream = MagicMock()
        upstream.raise_for_status = MagicMock()
        async def _aiter(chunk_size=4096):
            for c in (b"chunk1", b"chunk2", b"chunk3"):
                yield c
        upstream.aiter_bytes = _aiter

        client = MagicMock()
        client.stream = MagicMock()
        client.stream.return_value.__aenter__ = AsyncMock(return_value=upstream)
        client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        chunks: list[bytes] = []
        async for c in tts_stream_via_peer(
            http_client_factory=_factory,
            identity=identity,
            user_id="usr_x",
            peer_base_url="https://peer.local",
            payload={"input": "hi", "voice": "af_heart"},
        ):
            chunks.append(c)

        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

        # Inspect the call to verify URL + signed headers.
        call = client.stream.call_args
        method, url = call.args[0], call.args[1]
        assert method == "POST"
        assert url.endswith("/api/fabric/tts")
        sent_headers = call.kwargs["headers"]
        for required in ("X-Fabric-Sender", "X-Fabric-Signature", "X-Fabric-Timestamp"):
            assert required in sent_headers
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_tts_transport_error_wrapped():
    """httpx.TransportError gets wrapped as RemoteAudioError so callers
    don't have to know about httpx internals."""
    import aiosqlite
    from augmentum.fabric.audio_client import (
        RemoteAudioError, tts_stream_via_peer,
    )
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        client = MagicMock()
        client.stream = MagicMock(side_effect=httpx.ConnectError("connection refused"))

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        gen = tts_stream_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="u",
            peer_base_url="https://peer.local",
            payload={"input": "hi"},
        )
        with pytest.raises(RemoteAudioError) as excinfo:
            async for _ in gen:
                pass
        assert "unreachable" in str(excinfo.value).lower()
    finally:
        await conn.close()


# ── STT ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stt_raises_when_identity_missing():
    from augmentum.fabric.audio_client import (
        RemoteAudioError, stt_transcribe_via_peer,
    )

    with pytest.raises(RemoteAudioError):
        await stt_transcribe_via_peer(
            http_client_factory=MagicMock(),
            identity=None, user_id="u",
            peer_base_url="https://peer.local",
            audio_bytes=b"audio", filename="a.wav", content_type="audio/wav",
        )


@pytest.mark.asyncio
async def test_stt_returns_text_with_signed_multipart():
    """Happy path: POSTs multipart/form-data to /api/fabric/stt with
    a signed envelope and returns the JSON ``text`` field."""
    import aiosqlite
    from augmentum.fabric.audio_client import stt_transcribe_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"text": "transcribed speech"})
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        out = await stt_transcribe_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="usr_x",
            peer_base_url="https://peer.local",
            audio_bytes=b"AUDIODATA", filename="rec.wav", content_type="audio/wav",
            model="whisper-1", language="en",
        )
        assert out == "transcribed speech"

        call = client.post.call_args
        # URL is positional arg 0 to client.post
        assert call.args[0].endswith("/api/fabric/stt")
        # Headers contain signed envelope + the known boundary content-type
        sent_headers = call.kwargs["headers"]
        assert "X-Fabric-Signature" in sent_headers
        assert sent_headers["Content-Type"].startswith("multipart/form-data; boundary=")
        # Body must contain the audio bytes verbatim (so signature
        # verification on the receiver can re-hash exactly what was sent)
        sent_body = call.kwargs["content"]
        assert b"AUDIODATA" in sent_body
        assert b'name="model"' in sent_body
        assert b"whisper-1" in sent_body
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_stt_http_status_error_wrapped():
    """HTTPStatusError → RemoteAudioError. Caller maps to its own
    appropriate HTTP response code."""
    import aiosqlite
    from augmentum.fabric.audio_client import (
        RemoteAudioError, stt_transcribe_via_peer,
    )
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        # Build an HTTPStatusError manually
        fake_resp = MagicMock()
        fake_resp.status_code = 500
        fake_resp.text = "internal"
        err = httpx.HTTPStatusError(
            "500 err", request=MagicMock(), response=fake_resp,
        )
        client = MagicMock()
        client.post = AsyncMock(side_effect=err)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        with pytest.raises(RemoteAudioError) as excinfo:
            await stt_transcribe_via_peer(
                http_client_factory=_factory,
                identity=identity, user_id="u",
                peer_base_url="https://peer.local",
                audio_bytes=b"a", filename="a.wav", content_type="audio/wav",
            )
        assert "500" in str(excinfo.value)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_tts_session_id_folded_into_signed_body():
    """Context-aware engines (CSM) key prosody off the conversation id.
    Over fabric only the signed body crosses, so session_id must ride
    INSIDE the body (a header would be dropped by the receiver's
    body-only reconstruction). Pin that it lands in the signed bytes."""
    import aiosqlite
    from augmentum.fabric.audio_client import tts_stream_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        upstream = MagicMock()
        upstream.raise_for_status = MagicMock()
        async def _aiter(chunk_size=4096):
            yield b"x"
        upstream.aiter_bytes = _aiter
        client = MagicMock()
        client.stream = MagicMock()
        client.stream.return_value.__aenter__ = AsyncMock(return_value=upstream)
        client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        async for _ in tts_stream_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="usr_x",
            peer_base_url="https://peer.local",
            payload={"input": "hi", "voice": "conversational_a-csm"},
            session_id="ses_abc123",
        ):
            pass

        sent_body = client.stream.call_args.kwargs["content"]
        # Compact JSON (separators=(",", ":")) — the field is in the signed bytes
        assert b'"session_id":"ses_abc123"' in sent_body
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_tts_no_session_id_means_no_field():
    """Default (no session) must NOT inject the key — keeps the body
    identical to pre-feature for non-CSM providers."""
    import aiosqlite
    from augmentum.fabric.audio_client import tts_stream_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        upstream = MagicMock()
        upstream.raise_for_status = MagicMock()
        async def _aiter(chunk_size=4096):
            yield b"x"
        upstream.aiter_bytes = _aiter
        client = MagicMock()
        client.stream = MagicMock()
        client.stream.return_value.__aenter__ = AsyncMock(return_value=upstream)
        client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        async for _ in tts_stream_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="usr_x",
            peer_base_url="https://peer.local",
            payload={"input": "hi"},
        ):
            pass

        assert b"session_id" not in client.stream.call_args.kwargs["content"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_clone_upload_via_peer_posts_signed_multipart():
    """Voice-clone push: POSTs the clip + transcript as signed
    multipart to /api/fabric/voice-clone so a remote CSM (which can't
    see the sender's /voices volume) gets the clone anchor."""
    import aiosqlite
    from augmentum.fabric.audio_client import clone_upload_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        ok = await clone_upload_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="usr_x",
            peer_base_url="https://peer.local",
            audio_bytes=b"REFCLIPBYTES", filename="matt.wav",
            content_type="audio/wav",
            voice_name="matt", transcript="this is my voice",
        )
        assert ok is True

        call = client.post.call_args
        assert call.args[0].endswith("/api/fabric/voice-clone")
        sent_headers = call.kwargs["headers"]
        assert "X-Fabric-Signature" in sent_headers
        assert "authorization" not in {k.lower() for k in sent_headers}
        sent_body = call.kwargs["content"]
        assert b"REFCLIPBYTES" in sent_body          # clip bytes verbatim
        assert b'name="voice_name"' in sent_body
        assert b"matt" in sent_body
        assert b'name="transcript"' in sent_body
        assert b"this is my voice" in sent_body
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_clone_upload_raises_without_identity():
    from augmentum.fabric.audio_client import (
        RemoteAudioError, clone_upload_via_peer,
    )

    with pytest.raises(RemoteAudioError):
        await clone_upload_via_peer(
            http_client_factory=MagicMock(),
            identity=None, user_id="u",
            peer_base_url="https://peer.local",
            audio_bytes=b"a", filename="a.wav", content_type="audio/wav",
            voice_name="x",
        )


@pytest.mark.asyncio
async def test_push_user_context_via_peer_posts_signed_multipart():
    """Cross-speaker context push: POSTs the user's clip + transcript as
    signed multipart to /api/fabric/tts/user-context, with the session id
    as a form field (the receiver re-attaches it as X-Augmentum-Session)."""
    import aiosqlite
    from augmentum.fabric.audio_client import push_user_context_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        ok = await push_user_context_via_peer(
            http_client_factory=_factory,
            identity=identity, user_id="usr_x",
            peer_base_url="https://peer.local",
            audio_bytes=b"USERWAVBYTES", filename="user_turn.wav",
            content_type="audio/wav",
            session_id="ses_xyz", transcript="hey what's up",
        )
        assert ok is True

        call = client.post.call_args
        assert call.args[0].endswith("/api/fabric/tts/user-context")
        sent_headers = call.kwargs["headers"]
        assert "X-Fabric-Signature" in sent_headers
        body = call.kwargs["content"]
        assert b"USERWAVBYTES" in body
        assert b'name="session_id"' in body
        assert b"ses_xyz" in body
        assert b'name="transcript"' in body
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_push_user_context_raises_without_identity():
    from augmentum.fabric.audio_client import (
        RemoteAudioError, push_user_context_via_peer,
    )

    with pytest.raises(RemoteAudioError):
        await push_user_context_via_peer(
            http_client_factory=MagicMock(),
            identity=None, user_id="u",
            peer_base_url="https://peer.local",
            audio_bytes=b"a", filename="a.wav", content_type="audio/wav",
            session_id="s",
        )


@pytest.mark.asyncio
async def test_push_user_context_noop_for_non_csm():
    """Cross-speaker context is CSM-only — a Kokoro/Pocket provider is a
    clean no-op (no network), so non-companion voices pay nothing."""
    from augmentum.proxy.audio_routes import push_user_context
    ok = await push_user_context(
        provider={"id": "kokoro-builtin", "base_url": "builtin"},
        session_id="s", pcm_audio=b"\x00\x01" * 100, sample_rate=16000,
        transcript="hi", user_id="u",
    )
    assert ok is False


@pytest.mark.asyncio
async def test_push_user_context_local_csm_posts_wav():
    """Local CSM: POSTs the user clip (WAV-wrapped) to /v1/context/user_turn
    with the session id as the X-Augmentum-Session header the sidecar keys on."""
    from contextlib import asynccontextmanager
    import augmentum.proxy.audio_routes as ar

    client = MagicMock()
    client.post = AsyncMock(return_value=MagicMock())

    @asynccontextmanager
    async def _fake_audio_client(_base):
        yield client

    with patch.object(ar, "_audio_client", _fake_audio_client):
        ok = await ar.push_user_context(
            provider={"id": "sesame-csm", "base_url": "http://sesame-csm:8920"},
            session_id="ses_1", pcm_audio=b"\x00\x01" * 200, sample_rate=16000,
            transcript="hey", user_id="u",
        )
    assert ok is True
    call = client.post.call_args
    assert call.args[0].endswith("/v1/context/user_turn")
    assert call.kwargs["headers"]["X-Augmentum-Session"] == "ses_1"
    # Audio is WAV-wrapped so the sidecar's ffmpeg can decode the raw PCM.
    assert call.kwargs["files"]["audio"][1][:4] == b"RIFF"


@pytest.mark.asyncio
async def test_warmup_and_unload_via_peer_signed_posts():
    """Residency pings: warmup is an empty signed POST; unload carries the
    session id in a signed JSON body. Both hit the fabric data-plane paths."""
    import aiosqlite
    from augmentum.fabric.audio_client import unload_via_peer, warmup_via_peer
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _factory(_base):
            yield client

        ok = await warmup_via_peer(
            http_client_factory=_factory, identity=identity,
            user_id="u", peer_base_url="https://peer.local",
        )
        assert ok is True
        assert client.post.call_args.args[0].endswith("/api/fabric/tts/warmup")
        assert "X-Fabric-Signature" in client.post.call_args.kwargs["headers"]

        ok = await unload_via_peer(
            http_client_factory=_factory, identity=identity,
            user_id="u", peer_base_url="https://peer.local", session_id="ses_9",
        )
        assert ok is True
        call = client.post.call_args
        assert call.args[0].endswith("/api/fabric/tts/unload")
        assert b"ses_9" in call.kwargs["content"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_csm_warm_unload_noop_for_non_csm():
    """Residency is CSM-only — a non-CSM provider is a clean no-op."""
    from augmentum.proxy.audio_routes import csm_unload, csm_warm
    prov = {"id": "kokoro-builtin", "base_url": "builtin"}
    assert await csm_warm(provider=prov, user_id="u") is False
    assert await csm_unload(provider=prov, session_id="s", user_id="u") is False


def test_multipart_body_includes_audio_and_text_fields():
    """The body bytes we sign MUST exactly match the bytes we send.
    Pin the multipart construction so future refactors don't break
    that invariant (signature verification fails the moment they
    diverge)."""
    from augmentum.fabric.audio_client import _build_multipart_body

    body = _build_multipart_body(
        boundary="testboundary",
        audio_bytes=b"AUDIO",
        filename="clip.wav",
        content_type="audio/wav",
        text_fields={"model": "whisper-1", "language": "en", "response_format": ""},
    )
    # Audio bytes embedded verbatim
    assert b"AUDIO" in body
    # File part header
    assert b'name="file"' in body
    assert b'filename="clip.wav"' in body
    assert b"Content-Type: audio/wav" in body
    # Text fields with non-empty values present
    assert b'name="model"' in body
    assert b"whisper-1" in body
    assert b'name="language"' in body
    assert b"en" in body
    # Empty text field is skipped (no name="response_format" anywhere)
    assert b'name="response_format"' not in body
    # Boundary brackets correctly
    assert body.startswith(b"--testboundary\r\n")
    assert body.endswith(b"--testboundary--\r\n")
