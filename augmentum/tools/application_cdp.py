"""Thin Chrome DevTools Protocol client for App Builder verify pass.

Augmentum's verify pass historically used quickjs + a hand-rolled DOM
mock to catch runtime errors. That caught the obvious cases (bare
SyntaxError, calling methods on null) but missed layout problems,
real Canvas rendering, timing bugs, and anything that leans on
browser-specific APIs the mock doesn't simulate.

This module trades that stack for a real headless Chromium driven via
CDP. We deliberately skip Playwright — its 200MB auto-downloaded
browser bundle + its own abstraction layer are overkill for the six
CDP methods we actually need. System chromium (from apt in the GPU
image) gives us ~150MB that's shared across every verify run, and the
client here is ~200 lines we own.

Usage:

    async with BrowserVerifier() as bv:
        errors = await bv.verify_html(assembled_html)

Errors returned match the shape the fix prompt already expects —
runtime exceptions, console.error calls, and any WIRES selectors that
threw during handler attach.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class ChromiumNotAvailable(RuntimeError):
    """Raised when no chromium binary is discoverable on PATH.

    Callers should catch this and fall back to the quickjs verify
    path — the App Builder must continue to work on hosts without a
    system browser installed.
    """


# Chromium binary names we search on PATH, in order. Covers Debian
# (chromium/chromium-browser), Alpine (chromium-browser), the Google-
# branded builds used in some container bases, and Windows executables.
# Edge is Chromium-based and speaks CDP fine, so msedge.exe is a valid
# fallback on Windows hosts that don't ship Chrome.
_CHROMIUM_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "chrome.exe",
    "msedge.exe",
)

# Absolute paths probed after the PATH lookup fails. Order matters:
# prefer Chrome over Edge when both are present since Chrome is the
# upstream Chromium reference for CDP behavior.
_WINDOWS_CHROMIUM_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)

_MACOS_CHROMIUM_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def _looks_like_snap_chromium_stub(path: str) -> bool:
    """Return true for Ubuntu's chromium-browser snap launcher stub.

    Some Ubuntu images expose ``/usr/bin/chromium-browser`` even though
    the actual snap is not installed. Treating that file as a browser
    causes CDP startup to hang until timeout.
    """
    try:
        head = Path(path).read_bytes()[:4096]
    except OSError:
        return False
    return b"requires the chromium snap" in head or b"snap install chromium" in head


def _settings_chromium_override() -> str:
    """Return the user's explicit ``chromium_binary_path`` setting, or "".

    Imported lazily so this module stays a leaf dependency — no cycle
    with ``augmentum.config`` at import time.
    """
    try:
        from augmentum.config import settings  # local import — see docstring
        return (getattr(settings, "chromium_binary_path", "") or "").strip()
    except Exception:
        return ""


def _scan_known_install_paths() -> str | None:
    """Probe platform-specific install locations after the PATH lookup fails.

    Windows scans Program Files / Program Files (x86) and the per-user
    ``%LOCALAPPDATA%\\Google\\Chrome\\Application`` install.
    macOS scans ``/Applications``.
    Linux is intentionally a no-op — apt/snap/yum chromium binaries
    are always on PATH on a properly configured distro.
    """
    if sys.platform == "win32":
        candidates = list(_WINDOWS_CHROMIUM_PATHS)
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            candidates.insert(
                0,
                str(Path(localappdata) / "Google" / "Chrome" / "Application" / "chrome.exe"),
            )
        for path in candidates:
            if Path(path).is_file():
                return path
        return None
    if sys.platform == "darwin":
        for path in _MACOS_CHROMIUM_PATHS:
            if Path(path).is_file():
                return path
        return None
    return None


def _chromium_install_hint() -> str:
    """Platform-aware install instruction surfaced in the not-found error.

    This is one of the load-bearing UX moments for the App Builder
    verify pass and the XR Browser Panel — Windows/macOS users should
    not be told to run apt-get.
    """
    if sys.platform == "win32":
        return (
            "Install Google Chrome from https://www.google.com/chrome/ "
            "or use the bundled Microsoft Edge, then set "
            "`chromium_binary_path` in Settings to the full executable path "
            "(e.g. C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe) "
            "or add the install folder to PATH."
        )
    if sys.platform == "darwin":
        return (
            "Install Google Chrome from https://www.google.com/chrome/, "
            "then set `chromium_binary_path` in Settings to "
            "'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' "
            "if auto-detection misses it."
        )
    return (
        "Install with `apt-get install chromium` (Debian/Ubuntu) or "
        "the equivalent for your distro, and ensure it's on PATH."
    )


def find_chromium() -> str | None:
    """Return the path to a chromium/chrome binary, or ``None``.

    Resolution order:
      1. ``settings.chromium_binary_path`` (explicit override)
      2. PATH lookup for :data:`_CHROMIUM_CANDIDATES`
      3. Platform-specific install locations
         (:data:`_WINDOWS_CHROMIUM_PATHS`, :data:`_MACOS_CHROMIUM_PATHS`)
    """
    override = _settings_chromium_override()
    if override and Path(override).is_file() and not _looks_like_snap_chromium_stub(override):
        return override
    # Fall through to auto-discovery if the override is missing or stale —
    # better to find *some* browser than to fail because a configured
    # path got moved.
    for name in _CHROMIUM_CANDIDATES:
        path = shutil.which(name)
        if path and not _looks_like_snap_chromium_stub(path):
            return path
    return _scan_known_install_paths()


@dataclass
class VerifyResult:
    """Outcome of a single ``verify_html`` call.

    ``errors`` is the list the fix prompt consumes; ``warnings`` is
    collected separately so downstream consumers can decide whether
    console.warn should gate shipping. ``load_ms`` helps diagnose
    hangs during browser verify.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    load_ms: int = 0


class BrowserVerifier:
    """Launches a headless chromium and drives it over the CDP websocket.

    The instance manages one browser process end-to-end: ``start``
    launches chromium and connects, ``verify_html`` reuses the same
    tab for every call, ``stop`` shuts the process down cleanly. Use
    the async-context-manager form when you want those for free.

    Concurrency: the class is NOT thread-safe — one browser per
    verifier, one verifier per task. Pool outside if parallelism is
    needed.
    """

    def __init__(
        self,
        *,
        chromium_path: str | None = None,
        port: int = 9222,
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
        self._ws = None  # websockets.WebSocketClientProtocol when connected
        self._next_id = 0
        # (future_id → asyncio.Future) so response handlers resolve the
        # right pending request. Events (messages with no id) are
        # dispatched into _event_queue instead.
        self._pending: dict[int, asyncio.Future] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_subscribers: list[tuple[str | None, asyncio.Queue]] = []
        self._reader_task: asyncio.Task | None = None

    async def __aenter__(self) -> BrowserVerifier:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Launch chromium and open a CDP websocket connection. Raises
        :class:`ChromiumNotAvailable` when no binary can be found."""
        if not self.chromium_path:
            raise ChromiumNotAvailable(
                f"No chromium/chrome binary found. {_chromium_install_hint()}"
            )
        if _looks_like_snap_chromium_stub(self.chromium_path):
            raise ChromiumNotAvailable(
                f"{self.chromium_path} is Ubuntu's chromium snap launcher, "
                "but the snap is not installed. Install a real chromium/chrome "
                "binary (for example Debian chromium or google-chrome-stable) "
                "to enable browser verify."
            )
        flags = [
            "--headless=new",
            # Software WebGL via ANGLE->SwiftShader (plus --no-sandbox and
            # --disable-dev-shm-usage for Docker) so app-builder / XR-panel
            # pages that use WebGL or canvas actually composite a captureable
            # frame. Deliberately NOT --disable-gpu: that kills the GPU process
            # and leaves WebGL with no compositor to screenshot. Single source
            # of truth: augmentum/utils/chromium.py.
            *HEADLESS_WEBGL_ARGS,
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={self.port}",
        ]
        if self.user_data_dir:
            flags.append(f"--user-data-dir={self.user_data_dir}")
        flags.extend(self.extra_flags)
        flags.append("about:blank")
        self._process = subprocess.Popen(
            [self.chromium_path, *flags],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("app_builder.cdp_started", pid=self._process.pid, port=self.port)
        await self._connect()

    async def _connect(self) -> None:
        """Poll the /json endpoint until chromium is serving CDP, then
        open a websocket to the first page target."""
        import httpx

        deadline = time.monotonic() + self.connect_timeout
        target_ws = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    r = await client.get(f"http://127.0.0.1:{self.port}/json", timeout=0.5)
                    if r.status_code == 200:
                        pages = [t for t in r.json() if t.get("type") == "page"]
                        if pages:
                            target_ws = pages[0]["webSocketDebuggerUrl"]
                            break
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(0.1)
        if not target_ws:
            raise RuntimeError(
                f"chromium did not open CDP endpoint on port {self.port} "
                f"within {self.connect_timeout}s"
            )

        import websockets
        self._ws = await websockets.connect(
            target_ws,
            max_size=64 * 1024 * 1024,  # bump from default 1MB — some pages emit large console payloads
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        # Subscribe to the event streams we actually read.
        await self._send("Runtime.enable")
        await self._send("Page.enable")
        log.info("app_builder.cdp_connected")

    async def stop(self) -> None:
        """Close the websocket and kill the chromium process."""
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                log.debug("cdp_ws_close_failed", error=str(exc))
            self._ws = None
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
            log.info("app_builder.cdp_stopped")
        self._process = None
        if self.cleanup_user_data_dir and self.user_data_dir:
            try:
                shutil.rmtree(self.user_data_dir, ignore_errors=True)
            except OSError as exc:
                log.debug(
                    "cdp_user_data_dir_cleanup_failed",
                    path=self.user_data_dir,
                    error=str(exc),
                )

    # --- CDP protocol ------------------------------------------------------

    async def _reader_loop(self) -> None:
        """Demultiplex incoming CDP messages — responses into ``_pending``,
        events into ``_event_queue``."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(msg["error"]))
                        else:
                            fut.set_result(msg.get("result", {}))
                else:
                    await self._publish_event(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("app_builder.cdp_reader_error", error=str(exc))

    async def _publish_event(self, msg: dict) -> None:
        method = msg.get("method")
        delivered = False
        for method_filter, queue in list(self._event_subscribers):
            if method_filter is not None and method_filter != method:
                continue
            delivered = True
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(msg)
        if delivered and method == "Page.screencastFrame":
            return
        await self._event_queue.put(msg)

    def subscribe_events(self, method: str | None = None, *, maxsize: int = 16) -> asyncio.Queue:
        """Subscribe to CDP events without stealing them from normal waits."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(maxsize)))
        self._event_subscribers.append((method, queue))
        return queue

    def unsubscribe_events(self, queue: asyncio.Queue) -> None:
        self._event_subscribers = [
            (method_filter, subscriber)
            for method_filter, subscriber in self._event_subscribers
            if subscriber is not queue
        ]

    async def _send(self, method: str, params: dict | None = None, *, timeout: float | None = None) -> dict:
        """Send a CDP command and await the response."""
        if not self._ws:
            raise RuntimeError("BrowserVerifier not started")
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

    def _drain_events(self) -> list[dict]:
        """Return all events queued so far and clear the queue."""
        events: list[dict] = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    # --- Verify API --------------------------------------------------------

    async def verify_html(
        self,
        html: str,
        *,
        settle_ms: int = 500,
        click_sequence: list[str] | None = None,
        click_wait_ms: int = 200,
    ) -> VerifyResult:
        """Load ``html`` in the browser and collect console errors + exceptions.

        ``settle_ms`` is an extra wait after load fires to let
        setTimeout-driven initialization complete before we snapshot
        errors. 500ms is enough for most init routines without
        noticeably slowing the pipeline.

        ``click_sequence`` — optional list of CSS selectors to click in
        order after load. Each click waits ``click_wait_ms`` to let
        handlers run; errors raised by the handlers surface in the
        final event drain. This is the smoke-test hook that catches
        semantic bugs (pause/resume state corruption, etc.) that raw
        page-load verification misses.
        """
        self._drain_events()  # clear any stale events from the previous run
        data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(html)

        start = time.monotonic()
        try:
            await self._send("Page.navigate", {"url": data_url})
        except Exception as exc:
            return VerifyResult(errors=[f"RUNTIME: Page.navigate failed: {exc}"])

        load_fired = await self._await_event("Page.loadEventFired", timeout=self.page_timeout)
        if not load_fired:
            return VerifyResult(errors=["RUNTIME: page never fired loadEventFired"])

        await asyncio.sleep(settle_ms / 1000.0)

        if click_sequence:
            for sel in click_sequence:
                try:
                    await self.click(sel, wait_ms=click_wait_ms)
                except Exception as exc:
                    log.info("app_builder.cdp_click_failed", selector=sel, error=str(exc))

        load_ms = int((time.monotonic() - start) * 1000)

        errors, warnings = self._classify_events(self._drain_events())
        return VerifyResult(errors=errors, warnings=warnings, load_ms=load_ms)

    async def click(self, selector: str, *, wait_ms: int = 200) -> None:
        """Synthetically click the element matching ``selector`` via
        Runtime.evaluate. No-op if the selector doesn't match — we
        want smoke tests that target speculative elements to simply
        skip rather than fail. Errors raised by the element's event
        handler are captured as ``Runtime.exceptionThrown`` events and
        surface in the next :meth:`verify_html` drain.
        """
        expr = (
            "(function(){const el=document.querySelector("
            + json.dumps(selector)
            + "); if (el) { el.click(); return true; } return false; })()"
        )
        await self._send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True},
            timeout=self.page_timeout,
        )
        await asyncio.sleep(wait_ms / 1000.0)

    async def capture_screenshot(
        self,
        html: str,
        *,
        viewport_w: int = 800,
        viewport_h: int = 600,
        settle_ms: int = 800,
    ) -> bytes:
        """Load ``html`` and return a PNG screenshot of the viewport.

        ``settle_ms`` defaults higher than verify_html — for library
        thumbnails we want fonts, web fonts, and any setTimeout-driven
        first-paint animations to be on screen, not a blank canvas.
        """
        import base64

        self._drain_events()
        await self._send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": viewport_w,
                "height": viewport_h,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(html)
        await self._send("Page.navigate", {"url": data_url})
        await self._await_event("Page.loadEventFired", timeout=self.page_timeout)
        await asyncio.sleep(settle_ms / 1000.0)

        result = await self._send(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
            timeout=self.page_timeout,
        )
        b64 = (result or {}).get("data") or ""
        if not b64:
            raise RuntimeError("Page.captureScreenshot returned no data")
        return base64.b64decode(b64)

    async def _await_event(self, method: str, *, timeout: float) -> dict | None:
        """Wait for an event with the given ``method`` to appear in the
        event queue. Events that don't match are drained back into the
        queue so other waiters (or the event classifier) still see them."""
        deadline = time.monotonic() + timeout
        bystanders: list[dict] = []
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    ev = await asyncio.wait_for(self._event_queue.get(), timeout=max(remaining, 0.01))
                except TimeoutError:
                    return None
                if ev.get("method") == method:
                    return ev
                bystanders.append(ev)
            return None
        finally:
            # Put bystanders back so _classify_events sees them later.
            for ev in bystanders:
                await self._event_queue.put(ev)

    @staticmethod
    def _classify_events(events: list[dict]) -> tuple[list[str], list[str]]:
        """Turn a CDP event list into error / warning strings.

        Recognised shapes:
        * ``Runtime.exceptionThrown`` → always an error.
        * ``Runtime.consoleAPICalled`` with type ``error`` → error.
        * ``Runtime.consoleAPICalled`` with type ``warning`` → warning.
        * Everything else is ignored.
        """
        errors: list[str] = []
        warnings: list[str] = []
        for ev in events:
            method = ev.get("method", "")
            params = ev.get("params", {})
            if method == "Runtime.exceptionThrown":
                details = params.get("exceptionDetails", {})
                text = details.get("text") or "uncaught exception"
                exc = details.get("exception", {})
                desc = exc.get("description") or exc.get("value")
                url = details.get("url") or ""
                line = details.get("lineNumber")
                loc = f" ({url}:{line})" if url and line is not None else ""
                errors.append(f"RUNTIME: {text}: {desc}{loc}".strip())
            elif method == "Runtime.consoleAPICalled":
                level = params.get("type", "log")
                if level not in ("error", "warning"):
                    continue
                args = params.get("args", [])
                text = " ".join(
                    (a.get("value") or a.get("description") or "")
                    for a in args
                )
                msg = f"CONSOLE.{level.upper()}: {text}".strip()
                if level == "error":
                    errors.append(msg)
                else:
                    warnings.append(msg)
        return errors, warnings


# --- Integration helper ----------------------------------------------------

# Strip chromium's own internal console warnings (devtools-protocol spam,
# cookie warnings unrelated to user code, etc.) so the fix prompt sees
# only real app-level problems.
_CHROMIUM_NOISE = re.compile(
    r"(?:chrome-error://|chrome://|Error with Permissions-Policy header|"
    r"was preloaded using link preload|DevTools failed)",
    re.IGNORECASE,
)


def derive_smoke_sequence(
    planned_files: list[dict],
    description: str = "",
    *,
    max_clicks: int = 8,
) -> list[str]:
    """Derive a CSS-selector click sequence for smoke testing.

    Pulls selectors from two sources in priority order:

    1. WIRES entries on the plan — e.g. a file with
       ``"wires": ["#btn-start click", "#btn-reset click"]`` contributes
       ``["#btn-start", "#btn-reset"]``. These are the elements the
       plan committed to wiring, so they're the right thing to exercise.
    2. Intent-module hints (application_intent.derive_intent_features)
       — the ``wires`` field of each hint is scanned for ``#id``
       and ``.class`` selectors and appended if not already present.

    Duplicates are dropped. Capped at ``max_clicks`` so the smoke test
    stays brief; sequential clicks beyond that add noise without
    catching new failure modes.
    """
    selectors: list[str] = []
    seen: set[str] = set()

    def _add(sel: str) -> None:
        if sel and sel not in seen:
            seen.add(sel)
            selectors.append(sel)

    # 1. Plan WIRES — these are the authoritative interaction points
    for p in planned_files or []:
        for w in p.get("wires") or []:
            token = (w or "").strip().split()
            if token and (token[0].startswith("#") or token[0].startswith(".")):
                _add(token[0])

    # 2. Intent hints — supplemental selectors for features the plan
    #    forgot to WIRE explicitly.
    try:
        from augmentum.tools.application_intent import derive_intent_features
        for feat in derive_intent_features(description):
            hint = feat.get("wires") or ""
            for m in re.finditer(r"(#[\w-]+|\.[\w-]+)", hint):
                _add(m.group(1))
    except Exception:
        # Intent-feature derivation is supplementary — fall back to the
        # plain selectors already harvested from the description.
        log.debug("cdp_smoke_sequence_intent_features_failed", exc_info=True)

    return selectors[:max_clicks]


def filter_browser_errors(errors: list[str]) -> list[str]:
    """Drop chromium-internal noise from the CDP error list.

    Returns a new list with chromium's own notices removed; user-code
    errors (which originate inside the assembled HTML) pass through
    untouched.
    """
    return [e for e in errors if not _CHROMIUM_NOISE.search(e)]


async def capture_html_screenshot(
    html: str,
    *,
    viewport_w: int = 800,
    viewport_h: int = 600,
    settle_ms: int = 800,
) -> bytes | None:
    """One-shot: spin up a verifier, navigate, capture a PNG, shut down.

    Returns the PNG bytes, or ``None`` if chromium isn't installed or the
    capture failed. Used by the library to render static app thumbnails
    instead of hover-driven live iframes.
    """
    try:
        async with BrowserVerifier() as bv:
            return await bv.capture_screenshot(
                html,
                viewport_w=viewport_w,
                viewport_h=viewport_h,
                settle_ms=settle_ms,
            )
    except ChromiumNotAvailable as exc:
        log.info("preview_screenshot.no_chromium", reason=str(exc))
        return None
    except Exception as exc:
        log.warning("preview_screenshot.failed", error=str(exc)[:200])
        return None
