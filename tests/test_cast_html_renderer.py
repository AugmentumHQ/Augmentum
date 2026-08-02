"""Tests for HTMLRenderer.

Tests cover the CDP-message contract — what the renderer sends to
Chrome and how it interprets the response. We don't launch a real
Chrome here; we monkeypatch ``_send`` and ``_subscribe`` so the
render loop runs against an in-memory fake.

Real-Chrome integration is covered by the live container test
(separate file) which is opt-in because it needs the binary present.

Pins:

  - ChromiumUnavailable raised when no binary is discoverable
  - render_html_to_image calls Page.navigate with a data: URL whose
    body is the supplied HTML
  - render_html_to_image sets viewport via Emulation.setDeviceMetrics
  - render_html_to_image waits for Page.loadEventFired before capture
  - the captured base64 PNG is decoded and returned as bytes
  - captureScreenshot returning no data raises HTMLRenderError
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from augmentum.cast.html_renderer import (
    ChromiumUnavailable,
    HTMLRenderer,
    HTMLRenderError,
)


# ── Discovery / lifecycle ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_raises_when_no_chromium(monkeypatch):
    monkeypatch.setattr(
        "augmentum.cast.html_renderer.find_chromium",
        lambda: None,
    )
    renderer = HTMLRenderer()
    with pytest.raises(ChromiumUnavailable):
        await renderer.start()


# ── Render flow (CDP layer mocked) ────────────────────────────────


class _FakeRenderer(HTMLRenderer):
    """Subclass that fakes the CDP layer.

    Records every (method, params) sent. Loads a canned screenshot
    response for Page.captureScreenshot. The subscribe queue is
    populated immediately with a Page.loadEventFired event so the
    wait completes instantly.
    """

    def __init__(self, *, screenshot_b64: str = "") -> None:
        super().__init__(chromium_path="/fake/chrome", port=9999)
        self.sends: list[tuple[str, dict]] = []
        self._fake_screenshot_b64 = screenshot_b64
        self._started = True   # skip real start
        # Pre-stocked queue used by render_html_to_image's wait-for-load.
        self._fake_load_queue: asyncio.Queue = asyncio.Queue()
        self._fake_load_queue.put_nowait({"method": "Page.loadEventFired"})

    async def _send(self, method, params=None, *, timeout=None):  # type: ignore[override]
        self.sends.append((method, dict(params or {})))
        if method == "Page.captureScreenshot":
            return {"data": self._fake_screenshot_b64}
        return {}

    def _subscribe(self, method=None, *, maxsize=16):  # type: ignore[override]
        # Hand back the pre-stocked queue regardless of filter — tests
        # only subscribe to one method.
        return self._fake_load_queue

    def _unsubscribe(self, queue):  # type: ignore[override]
        pass


@pytest.mark.asyncio
async def test_render_navigates_to_data_url_with_supplied_html():
    sample_png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
    r = _FakeRenderer(screenshot_b64=sample_png)

    await r.render_html_to_image("<h1>Hi</h1>")

    methods = [m for m, _ in r.sends]
    assert "Emulation.setDeviceMetricsOverride" in methods
    assert "Page.navigate" in methods
    assert "Page.captureScreenshot" in methods

    nav_params = next(p for m, p in r.sends if m == "Page.navigate")
    assert nav_params["url"].startswith("data:text/html;base64,")
    decoded = base64.b64decode(nav_params["url"].split(",", 1)[1])
    assert decoded == b"<h1>Hi</h1>"


@pytest.mark.asyncio
async def test_render_returns_decoded_png_bytes():
    sample = b"\x89PNG\r\n\x1a\nrendered-bytes"
    r = _FakeRenderer(screenshot_b64=base64.b64encode(sample).decode("ascii"))

    result = await r.render_html_to_image("<p>x</p>")
    assert result == sample


@pytest.mark.asyncio
async def test_render_uses_supplied_viewport():
    sample_png = base64.b64encode(b"PNG").decode("ascii")
    r = _FakeRenderer(screenshot_b64=sample_png)

    await r.render_html_to_image("<p>x</p>", viewport_w=800, viewport_h=600)
    viewport_params = next(
        p for m, p in r.sends if m == "Emulation.setDeviceMetricsOverride"
    )
    assert viewport_params["width"] == 800
    assert viewport_params["height"] == 600


@pytest.mark.asyncio
async def test_render_raises_on_empty_screenshot_data():
    """An empty base64 string from captureScreenshot is suspicious —
    surface it as a clear error, don't silently return b'' which a
    consumer might write to disk as a corrupt PNG."""
    r = _FakeRenderer(screenshot_b64="")  # CDP returns no data
    with pytest.raises(HTMLRenderError):
        await r.render_html_to_image("<p>nope</p>")
