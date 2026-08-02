"""Headless Chrome HTML → PNG renderer.

Drives the system Chrome binary (apt-installed in Dockerfile.gpu) via
the Chrome DevTools Protocol over a WebSocket. Same wire protocol the
App Builder's :class:`BrowserVerifier` uses — we keep a separate class
because verify and render have different lifecycles (verify is one-
shot per build; render is long-lived across many cast jobs) and
distinct API surfaces.

Future refactor: extract the shared launch/connect/CDP-send machinery
into a ``_CDPSession`` base when a third CDP consumer lands. Today
the duplication is private (~120 LOC of launch + read-loop) and
keeping BrowserVerifier's wire-compatible API untouched is worth the
local copy.

Usage:

    async with HTMLRenderer() as r:
        png_bytes = await r.render_html_to_image("<h1>Hi</h1>")

The renderer is NOT thread-safe — one process, one tab, one render
at a time. Pool externally if parallelism is needed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import shutil
import subprocess
import time
from typing import Sequence

from augmentum.tools.application_cdp import find_chromium
from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class HTMLRenderError(RuntimeError):
    """Raised when a render call fails. Subclasses fail closed — the
    dispatcher converts the exception into RenderResult(ok=False) so
    callers never see a raw raise from the renderer."""


class ChromiumUnavailable(HTMLRenderError):
    """No Chrome/Chromium binary discoverable. The dispatcher should
    fall back to the stub executor (or a different node) when this
    fires at start-time."""


class HTMLRenderer:
    """Long-lived headless Chrome → PNG.

    Single Chrome process, single tab. Each ``render_html_to_image``
    call navigates the tab to a fresh ``data:text/html;base64,…``
    document and captures the result via ``Page.captureScreenshot``.
    Reusing the tab keeps per-render latency low (the per-process
    cold-start is ~1s; per-render after that is ~100-300ms for
    simple HTML).
    """

    def __init__(
        self,
        *,
        chromium_path: str | None = None,
        port: int = 9223,                # distinct from BrowserVerifier's 9222
        connect_timeout: float = 10.0,
        page_timeout: float = 15.0,
        user_data_dir: str | None = None,
        cleanup_user_data_dir: bool = False,
        extra_flags: Sequence[str] | None = None,
    ) -> None:
        self.chromium_path = chromium_path or find_chromium()
        self.port = port
        self.connect_timeout = connect_timeout
        self.page_timeout = page_timeout
        self.user_data_dir = user_data_dir
        self.cleanup_user_data_dir = cleanup_user_data_dir
        self.extra_flags = list(extra_flags or [])
        self._process: subprocess.Popen | None = None
        self._ws = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._event_subscribers: list[tuple[str | None, asyncio.Queue]] = []
        self._started = False

    async def __aenter__(self) -> HTMLRenderer:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # --- Lifecycle ----------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        if not self.chromium_path:
            raise ChromiumUnavailable("no Chrome/Chromium binary discoverable")

        flags = [
            "--headless=new",
            # Software WebGL (ANGLE->SwiftShader) + Docker sandbox / -dev-shm
            # flags so WebGL/canvas cast pages composite a captureable frame.
            # Deliberately NOT --disable-gpu — see augmentum/utils/chromium.py.
            *HEADLESS_WEBGL_ARGS,
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            f"--remote-debugging-port={self.port}",
        ]
        if self.user_data_dir:
            flags.append(f"--user-data-dir={self.user_data_dir}")
        flags.extend(self.extra_flags)
        flags.append("about:blank")

        # Offload the fork/exec off the event loop (it briefly blocks).
        # Kept as a subprocess.Popen so lifecycle calls (.pid/.terminate)
        # downstream are unchanged; only the spawn syscall runs in a thread.
        self._process = await asyncio.to_thread(
            lambda: subprocess.Popen(
                [self.chromium_path, *flags],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        log.info("cast_html_renderer_started", pid=self._process.pid, port=self.port)
        await self._connect()
        self._started = True

    async def _connect(self) -> None:
        import httpx

        deadline = time.monotonic() + self.connect_timeout
        target_ws: str | None = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    r = await client.get(
                        f"http://127.0.0.1:{self.port}/json", timeout=0.5,
                    )
                    if r.status_code == 200:
                        pages = [t for t in r.json() if t.get("type") == "page"]
                        if pages:
                            target_ws = pages[0]["webSocketDebuggerUrl"]
                            break
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(0.1)
        if not target_ws:
            raise HTMLRenderError(
                f"chromium did not open CDP endpoint on port {self.port} "
                f"within {self.connect_timeout}s",
            )

        import websockets
        self._ws = await websockets.connect(
            target_ws,
            max_size=64 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self._send("Page.enable")

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                log.debug("cast_html_renderer_ws_close_failed", error=str(exc))
            self._ws = None
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
            log.info("cast_html_renderer_stopped")
        self._process = None
        if self.cleanup_user_data_dir and self.user_data_dir:
            try:
                shutil.rmtree(self.user_data_dir, ignore_errors=True)
            except OSError as exc:
                log.debug(
                    "cast_html_renderer_user_data_dir_cleanup_failed",
                    path=self.user_data_dir,
                    error=str(exc),
                )
        self._started = False

    # --- CDP protocol -------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(str(msg["error"])))
                        else:
                            fut.set_result(msg.get("result", {}))
                else:
                    for method_filter, queue in list(self._event_subscribers):
                        if method_filter is not None and method_filter != msg.get("method"):
                            continue
                        with contextlib.suppress(asyncio.QueueFull):
                            queue.put_nowait(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("cast_html_renderer_reader_error", error=str(exc))

    def _subscribe(self, method: str | None = None, *, maxsize: int = 16) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, maxsize))
        self._event_subscribers.append((method, queue))
        return queue

    def _unsubscribe(self, queue: asyncio.Queue) -> None:
        self._event_subscribers = [
            (mf, q) for mf, q in self._event_subscribers if q is not queue
        ]

    async def _send(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        if not self._ws:
            raise HTMLRenderError("renderer not started")
        self._next_id += 1
        msg_id = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        await self._ws.send(json.dumps({
            "id": msg_id,
            "method": method,
            "params": params or {},
        }))
        return await asyncio.wait_for(fut, timeout=timeout or self.page_timeout)

    # --- Public surface -----------------------------------------------

    async def render_html_to_image(
        self,
        html: str,
        *,
        viewport_w: int = 1920,
        viewport_h: int = 1080,
        wait_for_load_s: float = 5.0,
    ) -> bytes:
        """Render ``html`` and return the captured PNG bytes.

        The HTML is loaded via a ``data:`` URL so each call starts
        from a clean document. ``wait_for_load_s`` bounds the wait
        for Page.loadEventFired; pages that never fire (e.g. blank
        body, immediate render) fall through after the bound and we
        still capture whatever is rendered.
        """
        if not self._started:
            await self.start()

        await self._send("Emulation.setDeviceMetricsOverride", {
            "width": int(viewport_w),
            "height": int(viewport_h),
            "deviceScaleFactor": 1,
            "mobile": False,
        })

        encoded = base64.b64encode((html or "").encode("utf-8")).decode("ascii")
        data_url = f"data:text/html;base64,{encoded}"

        load_q = self._subscribe("Page.loadEventFired", maxsize=2)
        try:
            await self._send("Page.navigate", {"url": data_url})
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(load_q.get(), timeout=wait_for_load_s)
        finally:
            self._unsubscribe(load_q)

        result = await self._send("Page.captureScreenshot", {"format": "png"})
        encoded_png = result.get("data", "") if isinstance(result, dict) else ""
        if not encoded_png:
            raise HTMLRenderError("captureScreenshot returned no data")
        return base64.b64decode(encoded_png)
