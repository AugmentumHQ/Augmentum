"""Headset-native browser panel renderer.

This module owns host-side Chromium instances that render real local web
pages for immersive XR panels. The XR client maps screenshots onto WebGL
planes and sends UV hits back as pointer/scroll/text input.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from augmentum.tools.application_cdp import BrowserVerifier, ChromiumNotAvailable
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class XRBrowserPanel:
    id: str
    user_id: str
    url: str
    width: int
    height: int
    device_scale_factor: float
    image_format: str
    quality: int
    browser: BrowserVerifier
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    revision: int = 0
    last_frame: bytes = b""
    last_media_type: str = "image/jpeg"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stream_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class XRBrowserPanelManager:
    """Small CDP-backed panel pool.

    One Chromium process per live panel is intentionally simple for the
    first implementation. The routes bound panel ownership by user id and
    callers are expected to close panels when their XR surface closes.
    """

    def __init__(self, *, max_panels_per_user: int = 4, idle_ttl_s: float = 600.0) -> None:
        self.max_panels_per_user = max(1, int(max_panels_per_user))
        self.idle_ttl_s = max(30.0, float(idle_ttl_s))
        self._panels: dict[str, XRBrowserPanel] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        user_id: str,
        url: str,
        width: int = 1440,
        height: int = 900,
        device_scale_factor: float = 1.0,
        image_format: str = "jpeg",
        quality: int = 82,
        auth_headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> XRBrowserPanel:
        await self.cleanup_idle()
        evict_panel_id = ""
        async with self._lock:
            owned = [panel for panel in self._panels.values() if panel.user_id == user_id]
            if len(owned) >= self.max_panels_per_user:
                owned.sort(key=lambda panel: panel.last_used_at)
                evict_panel_id = owned[0].id
        if evict_panel_id:
            await self.close(evict_panel_id, user_id=user_id)

        panel_id = f"xrb_{uuid.uuid4().hex[:16]}"
        browser = BrowserVerifier(
            port=_free_port(),
            connect_timeout=12.0,
            page_timeout=18.0,
            user_data_dir=tempfile.mkdtemp(prefix=f"augmentum-xr-browser-{panel_id}-"),
            cleanup_user_data_dir=True,
            extra_flags=[
                "--allow-insecure-localhost",
                "--ignore-certificate-errors",
                "--remote-allow-origins=*",
            ],
        )
        try:
            await browser.start()
            panel = XRBrowserPanel(
                id=panel_id,
                user_id=user_id,
                url=url,
                width=max(320, min(3840, int(width))),
                height=max(240, min(2160, int(height))),
                device_scale_factor=max(0.5, min(3.0, float(device_scale_factor))),
                image_format="png" if image_format == "png" else "jpeg",
                quality=max(40, min(95, int(quality))),
                browser=browser,
                last_media_type="image/png" if image_format == "png" else "image/jpeg",
            )
            await self._configure_page(panel)
            await self._apply_request_context(
                panel,
                auth_headers=auth_headers or {},
                cookies=cookies or {},
            )
            await self._navigate(panel, url)
            await self._capture_panel(panel)
        except Exception:
            try:
                await browser.stop()
            except Exception as cleanup_exc:
                log.debug(
                    "xr_browser_panel_create_rollback_failed",
                    error=str(cleanup_exc),
                )
            raise

        async with self._lock:
            self._panels[panel.id] = panel
        log.info("xr_browser_panel_created", user_id=user_id, panel_id=panel.id, url=url)
        return panel

    async def _configure_page(self, panel: XRBrowserPanel) -> None:
        await panel.browser._send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": panel.width,
                "height": panel.height,
                "deviceScaleFactor": panel.device_scale_factor,
                "mobile": False,
                "screenWidth": panel.width,
                "screenHeight": panel.height,
            },
        )
        await panel.browser._send(
            "Emulation.setTouchEmulationEnabled",
            {"enabled": True, "maxTouchPoints": 5},
        )

    async def _apply_request_context(
        self,
        panel: XRBrowserPanel,
        *,
        auth_headers: dict[str, str],
        cookies: dict[str, str],
    ) -> None:
        if not auth_headers and not cookies:
            return
        await panel.browser._send("Network.enable")

        authorization = auth_headers.get("Authorization") or auth_headers.get("authorization")
        if authorization:
            await panel.browser._send(
                "Network.setExtraHTTPHeaders",
                {"headers": {"Authorization": authorization}},
            )

        parsed = urlparse(panel.url)
        cookie_url = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else panel.url
        for name, value in cookies.items():
            if not name:
                continue
            result = await panel.browser._send(
                "Network.setCookie",
                {
                    "name": str(name),
                    "value": str(value),
                    "url": cookie_url,
                },
            )
            if result.get("success") is False:
                log.warning("xr_browser_panel_cookie_rejected", panel_id=panel.id, cookie=name)

    async def _navigate(self, panel: XRBrowserPanel, url: str) -> None:
        panel.browser._drain_events()
        await panel.browser._send("Page.navigate", {"url": url})
        await panel.browser._await_event("Page.loadEventFired", timeout=panel.browser.page_timeout)
        await asyncio.sleep(0.25)

    async def get(self, panel_id: str, *, user_id: str) -> XRBrowserPanel | None:
        panel = self._panels.get(panel_id)
        if panel is None or panel.user_id != user_id:
            return None
        panel.last_used_at = time.monotonic()
        return panel

    async def capture(self, panel_id: str, *, user_id: str) -> XRBrowserPanel | None:
        panel = await self.get(panel_id, user_id=user_id)
        if panel is None:
            return None
        await self._capture_panel(panel)
        return panel

    async def _capture_panel(self, panel: XRBrowserPanel) -> None:
        async with panel.lock:
            params: dict[str, Any] = {
                "format": panel.image_format,
                "captureBeyondViewport": False,
            }
            if panel.image_format == "jpeg":
                params["quality"] = panel.quality
            shot = await panel.browser._send("Page.captureScreenshot", params, timeout=panel.browser.page_timeout)
            panel.last_frame = base64.b64decode(shot.get("data") or b"")
            panel.revision += 1
            panel.last_media_type = "image/png" if panel.image_format == "png" else "image/jpeg"
            panel.last_used_at = time.monotonic()

    async def stream_frames(
        self,
        panel: XRBrowserPanel,
        *,
        image_format: str = "jpeg",
        quality: int = 72,
        every_nth_frame: int = 1,
    ):
        """Yield CDP screencast frames for a panel as base64 image payloads."""
        screencast_format = "png" if image_format == "png" else "jpeg"
        media_type = "image/png" if screencast_format == "png" else "image/jpeg"
        started = False
        async with panel.stream_lock:
            queue = panel.browser.subscribe_events("Page.screencastFrame", maxsize=3)
            try:
                params: dict[str, Any] = {
                    "format": screencast_format,
                    "maxWidth": panel.width,
                    "maxHeight": panel.height,
                    "everyNthFrame": max(1, int(every_nth_frame)),
                }
                if screencast_format == "jpeg":
                    params["quality"] = max(35, min(95, int(quality)))
                await panel.browser._send("Page.startScreencast", params)
                started = True
                while True:
                    event = await queue.get()
                    payload = event.get("params") or {}
                    session_id = payload.get("sessionId")
                    if session_id is not None:
                        try:
                            await panel.browser._send(
                                "Page.screencastFrameAck",
                                {"sessionId": session_id},
                                timeout=5.0,
                            )
                        except Exception as ack_exc:
                            log.debug(
                                "xr_browser_screencast_ack_failed",
                                session_id=session_id,
                                error=str(ack_exc),
                            )
                    data = payload.get("data") or ""
                    if not data:
                        continue
                    panel.revision += 1
                    panel.last_media_type = media_type
                    panel.last_used_at = time.monotonic()
                    try:
                        panel.last_frame = base64.b64decode(data)
                    except (ValueError, TypeError) as decode_exc:
                        # Malformed base64 from CDP would be unusual but
                        # not fatal — keep last_frame stale rather than
                        # break the screencast loop.
                        log.debug(
                            "xr_browser_frame_b64_decode_failed",
                            error=str(decode_exc),
                        )
                    yield {
                        "type": "frame",
                        "revision": panel.revision,
                        "media_type": media_type,
                        "data": data,
                        "metadata": payload.get("metadata") or {},
                    }
            finally:
                panel.browser.unsubscribe_events(queue)
                if started:
                    try:
                        await panel.browser._send("Page.stopScreencast", timeout=5.0)
                    except Exception as stop_exc:
                        log.debug(
                            "xr_browser_screencast_stop_failed",
                            error=str(stop_exc),
                        )

    async def input(self, panel_id: str, *, user_id: str, event: dict[str, Any]) -> XRBrowserPanel | None:
        panel = await self.get(panel_id, user_id=user_id)
        if panel is None:
            return None
        event_type = str(event.get("type") or "").strip().lower()
        x = float(event.get("x") or 0)
        y = float(event.get("y") or 0)
        if event.get("normalized", True):
            x *= panel.width
            y *= panel.height
        x = max(0, min(panel.width - 1, int(round(x))))
        y = max(0, min(panel.height - 1, int(round(y))))

        async with panel.lock:
            if event_type == "click":
                await panel.browser._send("Input.dispatchMouseEvent", {
                    "type": "mouseMoved",
                    "x": x,
                    "y": y,
                    "button": "none",
                })
                await panel.browser._send("Input.dispatchMouseEvent", {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                })
                await panel.browser._send("Input.dispatchMouseEvent", {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                })
            elif event_type == "wheel":
                await panel.browser._send("Input.dispatchMouseEvent", {
                    "type": "mouseWheel",
                    "x": x,
                    "y": y,
                    "deltaX": float(event.get("deltaX") or 0),
                    "deltaY": float(event.get("deltaY") or 0),
                })
            elif event_type == "text":
                text = str(event.get("text") or "")
                if text:
                    await panel.browser._send("Input.insertText", {"text": text})
            elif event_type == "key":
                key = str(event.get("key") or "")
                if key:
                    await panel.browser._send("Input.dispatchKeyEvent", {
                        "type": "keyDown",
                        "key": key,
                    })
                    await panel.browser._send("Input.dispatchKeyEvent", {
                        "type": "keyUp",
                        "key": key,
                    })
            elif event_type == "refresh":
                await self._navigate(panel, panel.url)
            else:
                return panel

        await asyncio.sleep(0.06)
        return await self.capture(panel_id, user_id=user_id)

    async def close(self, panel_id: str, *, user_id: str) -> bool:
        async with self._lock:
            panel = self._panels.get(panel_id)
            if panel is None or panel.user_id != user_id:
                return False
            self._panels.pop(panel_id, None)
        try:
            await panel.browser.stop()
        except Exception as stop_exc:
            log.debug(
                "xr_browser_panel_close_stop_failed",
                panel_id=panel_id,
                error=str(stop_exc),
            )
        log.info("xr_browser_panel_closed", user_id=user_id, panel_id=panel_id)
        return True

    async def cleanup_idle(self) -> None:
        now = time.monotonic()
        stale = [
            panel
            for panel in list(self._panels.values())
            if now - panel.last_used_at > self.idle_ttl_s
        ]
        for panel in stale:
            await self.close(panel.id, user_id=panel.user_id)


__all__ = [
    "ChromiumNotAvailable",
    "XRBrowserPanel",
    "XRBrowserPanelManager",
]
