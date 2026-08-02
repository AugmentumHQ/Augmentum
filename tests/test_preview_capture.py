"""Live-preview screenshot capture — the GPU-frame path.

When the user's coder preview is open, browser_screenshot grabs the frame their
real GPU already rendered (via a proxy-injected agent + the preview-capture WS)
instead of re-rendering a heavy WebGL page in the GPU-less headless workspace.
These tests cover the server-side rendezvous, the HTML injection gate, and the
tool-side substitution + graceful fallbacks. See
augmentum/coder/preview_capture.py.
"""
from __future__ import annotations

import base64

import pytest

# A valid 1x1 PNG.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PNG_RAW = base64.b64decode(_PNG_B64)


def test_injection_gate():
    """The capture agent is injected into preview HTML only when the flag is on,
    and it carries the preserveDrawingBuffer hook (else WebGL toDataURL is blank)
    and the capture message handler."""
    import augmentum.proxy.coder_routes as CR

    html = b"<html><head></head><body></body></html>"
    on = CR._rewrite_html(html, "/api/coder/preview/w/8080/", inject_live_capture=True)
    off = CR._rewrite_html(html, "/b/", inject_live_capture=False)
    assert b"augmentum.preview.capture" in on
    assert b"preserveDrawingBuffer" in on
    assert b"toDataURL" in on
    assert b"augmentum.preview.capture" not in off


@pytest.mark.asyncio
async def test_broker_round_trip():
    from augmentum.coder.preview_capture import PreviewCaptureBroker

    b = PreviewCaptureBroker()
    assert not b.is_connected("w")
    assert await b.capture("w", timeout=0.2) is None  # no socket -> None

    async def sender(msg):
        # simulate the browser capturing and relaying a frame back
        b.resolve(msg["id"], {"data_url": "data:image/png;base64," + _PNG_B64, "width": 320, "height": 200})

    b.register("w", sender)
    assert b.is_connected("w")
    res = await b.capture("w", url="http://localhost:8080", timeout=1.0)
    assert res and res["data_url"].startswith("data:image/png") and res["width"] == 320
    b.unregister("w", sender)
    assert not b.is_connected("w")


@pytest.mark.asyncio
async def test_broker_times_out_when_browser_silent():
    from augmentum.coder.preview_capture import PreviewCaptureBroker

    b = PreviewCaptureBroker()

    async def silent(msg):
        return  # never resolves — e.g. the tab has no canvas / is frozen

    b.register("w", silent)
    assert await b.capture("w", timeout=0.2) is None  # times out -> None, never raises


class _StubCM:
    def __init__(self):
        self.written: dict[str, bytes] = {}

    async def file_write_bytes(self, ws, path, data):
        self.written[path] = data


@pytest.mark.asyncio
async def test_try_live_capture_writes_binary_png_and_gates_url():
    from augmentum.coder import browser as B
    from augmentum.coder.preview_capture import broker

    cm = _StubCM()

    async def sender(msg):
        broker.resolve(msg["id"], {"data_url": "data:image/png;base64," + _PNG_B64, "width": 64, "height": 48})

    broker.register("wsX", sender)
    try:
        # external URL -> never uses the live preview (would be the wrong page)
        assert await B._try_live_preview_capture(cm, "wsX", url="https://example.com", path="/p/s.png") is None
        # local preview -> live capture, written as binary PNG (byte-exact)
        p = "/workspace/.augmentum/browser-screenshots/shot.png"
        res = await B._try_live_preview_capture(cm, "wsX", url="http://localhost:8080", path=p)
        assert res and res["ok"] and res["source"] == "live_preview"
        assert res["width"] == 64 and res["height"] == 48
        assert cm.written.get(p) == _PNG_RAW
        assert cm.written[p][:8] == bytes.fromhex("89504e470d0a1a0a")  # PNG signature
    finally:
        broker.unregister("wsX", sender)

    # disconnected -> falls back (None) even for a local URL
    assert await B._try_live_preview_capture(cm, "wsX", url="http://localhost:8080", path="/p/s.png") is None


def test_local_preview_url_detection():
    from augmentum.coder import browser as B

    assert B._is_local_preview_url("http://localhost:8080/")
    assert B._is_local_preview_url("http://127.0.0.1:5173")
    assert B._is_local_preview_url("/api/coder/preview/ws/3000/")
    assert not B._is_local_preview_url("https://example.com")
    assert not B._is_local_preview_url("")
