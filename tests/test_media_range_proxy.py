"""Tests for _open_range_proxy — the Range-aware media stream wrapper.

The bug these regress against: prior to 2026-04-20 the stream proxy
always returned HTTP 200 with no Content-Range header, regardless of
whether upstream returned 200 or 206. When the <audio> element asked
for ``Range: bytes=N-`` on a seek, upstream returned 206 Partial
Content but the browser saw our 200 envelope with raw bytes inside —
it couldn't tell what offset those bytes corresponded to, and silently
reloaded from byte 0. Every +30s skip appeared to restart the book.

These tests pin down the contract:
    - 206 upstream → 206 downstream, Content-Range forwarded
    - 200 upstream → 200 downstream, Accept-Ranges forwarded
    - 4xx/5xx upstream → same status downstream, body surfaces error
    - Connection errors → 502 envelope
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.proxy.media_routes import _open_range_proxy


def _fake_upstream(
    *, status: int = 206, headers: dict | None = None, body: bytes = b"chunk",
) -> MagicMock:
    """Build an httpx.Response-shaped mock with a streaming aiter_bytes."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.aclose = AsyncMock()
    resp.aread = AsyncMock(return_value=body)

    async def _aiter_bytes(chunk_size: int = 65536):
        # Yield a single chunk — enough to validate the body-generator
        # plumbing without fiddling with real chunk boundaries.
        yield body
    resp.aiter_bytes = _aiter_bytes
    return resp


def _fake_client(upstream: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.build_request = MagicMock(return_value=MagicMock())
    client.send = AsyncMock(return_value=upstream)
    return client


@pytest.mark.asyncio
async def test_206_passes_through_with_content_range():
    """The critical fix: a Range seek produces 206 Partial Content, and the
    Content-Range header must land on the downstream response so the
    browser can assemble the partial body into the correct position."""
    upstream = _fake_upstream(
        status=206,
        headers={
            "content-range":  "bytes 1000000-4000000/4194304",
            "content-length": "3000001",
            "accept-ranges":  "bytes",
            "content-type":   "audio/mpeg",
        },
    )
    client = _fake_client(upstream)
    resp = await _open_range_proxy(client, "https://u/x.mp3", {"Range": "bytes=1000000-"})

    assert resp.status_code == 206
    # Content-Range is the header that tells the browser "these bytes
    # are from offset 1M" — without it, 206 is meaningless.
    assert resp.headers["content-range"] == "bytes 1000000-4000000/4194304"
    assert resp.headers["content-length"] == "3000001"
    assert resp.headers["content-type"]  == "audio/mpeg"


@pytest.mark.asyncio
async def test_200_passes_through_with_accept_ranges():
    """Initial (unranged) GET returns 200. Accept-Ranges tells the browser
    that seeks will work — without it, Chrome won't even try to seek
    and falls back to full-file buffering."""
    upstream = _fake_upstream(
        status=200,
        headers={
            "accept-ranges":  "bytes",
            "content-length": "4194304",
            "content-type":   "audio/mpeg",
        },
    )
    client = _fake_client(upstream)
    resp = await _open_range_proxy(client, "https://u/x.mp3", {})

    assert resp.status_code == 200
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-length"] == "4194304"


@pytest.mark.asyncio
async def test_upstream_error_status_is_preserved():
    """A 404 from upstream must surface as 404 downstream — not a 200 with
    an HTML error in the body. The <audio> element's error event fires
    only on non-2xx; hiding errors behind 200 makes failures invisible."""
    upstream = _fake_upstream(status=404, body=b"not found")
    client = _fake_client(upstream)
    resp = await _open_range_proxy(client, "https://u/missing.mp3", {})

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_sensitive_upstream_headers_are_not_forwarded():
    """We must only forward byte-range-relevant and freshness headers.
    Set-Cookie, Server, etc. from upstream have no business leaking to
    the user's browser via our proxy — belt-and-braces against upstream
    quirks."""
    upstream = _fake_upstream(
        status=200,
        headers={
            "accept-ranges": "bytes",
            "content-type":  "audio/mpeg",
            "set-cookie":    "leak=bad; Path=/",
            "server":        "upstream/1.0",
            "x-custom":      "sensitive",
        },
    )
    client = _fake_client(upstream)
    resp = await _open_range_proxy(client, "https://u/x.mp3", {})

    assert "accept-ranges" in resp.headers
    assert "content-type" in resp.headers
    assert "set-cookie" not in resp.headers
    assert "server" not in resp.headers
    assert "x-custom" not in resp.headers


@pytest.mark.asyncio
async def test_connection_error_returns_502():
    """If we can't reach upstream at all (DNS failure, refused connection,
    timeout before response), surface a 502 so the <audio> element
    errors cleanly instead of hanging on a pending stream."""
    client = AsyncMock()
    client.build_request = MagicMock(return_value=MagicMock())
    client.send = AsyncMock(side_effect=RuntimeError("connection refused"))

    resp = await _open_range_proxy(client, "https://unreachable/x.mp3", {})
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_last_modified_and_etag_forwarded():
    """These aren't seek-critical but browsers use them for conditional
    revalidation — forwarding them lets If-Modified-Since / If-None-Match
    round-trips work through the proxy without re-downloading unchanged
    audio on repeat opens."""
    upstream = _fake_upstream(
        status=200,
        headers={
            "accept-ranges": "bytes",
            "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
            "etag":          '"abc123"',
        },
    )
    client = _fake_client(upstream)
    resp = await _open_range_proxy(client, "https://u/x.mp3", {})
    assert resp.headers["last-modified"] == "Wed, 21 Oct 2025 07:28:00 GMT"
    assert resp.headers["etag"] == '"abc123"'
