"""ATP browser tools — thin adapters over the agent-browser sidecar.

Exposes the sidecar's browsing verbs (navigate, screenshot, action,
evaluate, wait) as registry Tools so external harnesses (Claude Code,
pi, cursor...) reach them through the agnostic ``/v1/tools`` surface.

Design constraints:

- Sidecar-only: no coder workspace exists on this path, so there is no
  Playwright-in-workspace or HTTP fallback rung. ``health_check`` gates
  listing on the sidecar actually running (a tool that isn't there
  beats a tool that lies).
- Tenant boundary: the sidecar session name is ALWAYS derived
  server-side from the authenticated user (``atp-<user_id>``) — never
  caller-supplied. Same rule as ``browser_sidecar.session_for_workspace``.
- Text-only results: screenshots are saved through the ArtifactStore
  and returned as an artifact download URL — never base64.
- These tools are ATP-only: ``SurfaceExposure`` turns off chat/coder/
  flow reach so registering them does not widen any existing surface.
- ``coder/browser.py`` and ``browser_sidecar.py`` are reused, not
  refactored: we call ``run_cli``/``pull_file`` and the shared
  ``_build_evaluate_wrapper`` directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any

from augmentum.coder import browser_sidecar as _sidecar
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# localhost from the harness's vantage is the HOST machine; the sidecar
# runs in Docker, where the host is reachable as host.docker.internal
# (compose.browser.yaml adds the extra_hosts mapping on Linux).
_LOCALHOST_RE = re.compile(
    r"^(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0)([:/]|$)", re.I
)


def _rewrite_url(url: str) -> str:
    u = (url or "").strip()
    m = _LOCALHOST_RE.match(u)
    if m:
        return u[: m.start(2)] + "host.docker.internal" + u[m.end(2):]
    return u


class _AtpBrowserBase(Tool):
    """Shared plumbing for the ATP browser verb tools."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def surfaces(self) -> SurfaceExposure:
        # ATP-only — do not widen chat/coder/flow surfaces.
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return 90.0

    # -- sidecar access ------------------------------------------------

    def _docker(self):
        cm = getattr(self._app_state, "container_manager", None)
        return getattr(cm, "_docker", None) if cm is not None else None

    def health_check(self) -> bool:
        """Sync, cache-based sidecar liveness.

        The authoritative check is async (``health_check_async``, used
        by the ATP routes). This sync version consults the sidecar
        discovery cache and, when stale, kicks off a background refresh
        so the NEXT check is accurate rather than blocking this one.
        """
        docker = self._docker()
        if docker is None:
            return False
        cached = _sidecar._sidecar_cache
        if cached is not None and time.monotonic() - cached[0] < _sidecar._CACHE_TTL:
            return cached[1] is not None
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_sidecar.find_sidecar(docker))
        return cached[1] is not None if cached is not None else False

    async def health_check_async(self) -> bool:
        docker = self._docker()
        if docker is None:
            return False
        return await _sidecar.is_available(docker)

    # -- session + page state ------------------------------------------

    def _session(self, kwargs: dict) -> str:
        uid = self.extract_user_id(kwargs) or "anon"
        return _sidecar.session_for_workspace(uid, prefix="atp")

    async def _ensure_page(self, docker, session: str, url: str) -> str:
        """Navigate to ``url`` only if the session isn't already there.

        Returns "" on success, or an error string. Mirrors
        ``browser_sidecar._ensure_page`` minus the workspace vantage
        rewrite (ATP callers pass real URLs; localhost is rewritten to
        the Docker host).
        """
        if not url:
            return ""
        target = _rewrite_url(url)
        current = await _sidecar.run_cli(
            docker, ["get", "url"], session=session, timeout=10.0
        )
        if current.get("ok") and _sidecar._urls_equivalent(
            str(current.get("url") or ""), target
        ):
            return ""
        opened = await _sidecar.run_cli(
            docker, ["open", target], session=session, timeout=30.0
        )
        if not opened.get("ok"):
            return str(opened.get("error") or f"failed to open {url}")
        return ""

    async def _page_summary(self, docker, session: str) -> dict[str, Any]:
        title = await _sidecar.run_cli(
            docker, ["get", "title"], session=session, timeout=10.0
        )
        url = await _sidecar.run_cli(
            docker, ["get", "url"], session=session, timeout=10.0
        )
        diag = await _sidecar._page_diagnostics(docker, session)
        return {
            "title": str(title.get("title") or "")[:200],
            "url": str(url.get("url") or "")[:500],
            **diag,
        }

    def _unavailable(self) -> ToolResult:
        return ToolResult(
            success=False,
            error=(
                "browser sidecar is unavailable — the augmentum-browser "
                "container (compose.browser.yaml) is not running"
            ),
        )

    @staticmethod
    def _diag_text(summary: dict[str, Any]) -> str:
        parts = []
        errs = summary.get("console_errors") or []
        fails = summary.get("network_failures") or []
        if errs:
            parts.append(
                "Console errors:\n"
                + "\n".join(f"- [{e.get('type')}] {e.get('text')}" for e in errs[:10])
            )
        if fails:
            parts.append(
                "Network failures:\n"
                + "\n".join(
                    f"- {f.get('method', '')} {f.get('url', '')} "
                    f"({f.get('status') or f.get('failure', '')})"
                    for f in fails[:10]
                )
            )
        return ("\n\n" + "\n\n".join(parts)) if parts else ""


class BrowserNavigateTool(_AtpBrowserBase):
    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return (
            "Open a URL in a persistent server-side browser session and "
            "return the page title and visible text. Session state "
            "(cookies, page) persists across calls. localhost URLs are "
            "rewritten to reach the host machine."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"},
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        url = str(kwargs.get("url") or "").strip()
        if not url:
            return ToolResult(success=False, error="'url' is required", validation_error=True)
        session = self._session(kwargs)
        err = await self._ensure_page(docker, session, url)
        if err:
            return ToolResult(success=False, error=err)
        body = await _sidecar.run_cli(
            docker, ["get", "text", "body"], session=session, timeout=15.0
        )
        summary = await self._page_summary(docker, session)
        text = str(body.get("text") or "")[:4000]
        return ToolResult(
            success=True,
            output=(
                f"Opened {summary['url'] or url}\n"
                f"Title: {summary['title']}\n\n{text}"
                + self._diag_text(summary)
            ),
            metadata=summary,
        )


class BrowserScreenshotTool(_AtpBrowserBase):
    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return (
            "Screenshot the current page (or a URL) in the server-side "
            "browser session. Returns an artifact download URL for the "
            "PNG — fetch it over HTTP; the result itself is text-only."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to open first (omit to shoot the current page)",
                },
                "full_page": {"type": "boolean", "default": False},
                "wait_for_selector": {
                    "type": "string",
                    "description": "CSS selector to wait for before capturing",
                },
            },
        }

    @property
    def produces(self) -> list[str]:
        return ["text", "artifact_url"]

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        store = getattr(self._app_state, "artifact_store", None)
        if store is None:
            return ToolResult(success=False, error="artifact store is unavailable")
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user for screenshot storage")
        session = self._session(kwargs)
        url = str(kwargs.get("url") or "").strip()
        err = await self._ensure_page(docker, session, url)
        if err:
            return ToolResult(success=False, error=err)
        sel = str(kwargs.get("wait_for_selector") or "").strip()
        if sel:
            await _sidecar.run_cli(docker, ["wait", sel], session=session, timeout=15.0)
        remote = f"/tmp/atp_shot_{int(time.time() * 1000)}.png"
        args = ["screenshot"]
        if kwargs.get("full_page"):
            args.append("--full")
        args.append(remote)
        res = await _sidecar.run_cli(docker, args, session=session, timeout=45.0)
        if not res.get("ok"):
            return ToolResult(success=False, error=str(res.get("error") or "screenshot failed"))
        try:
            png = await _sidecar.pull_file(docker, remote)
        except Exception as exc:
            return ToolResult(success=False, error=f"screenshot pull failed: {exc}")
        summary = await self._page_summary(docker, session)
        info = await store.save(
            png,
            f"screenshot_{int(time.time())}.png",
            "png",
            task_id="browser",
            display_name=f"Screenshot: {summary['title'] or summary['url'] or url}"[:120],
            metadata={"source": "atp_browser", "url": summary["url"] or url},
            user_id=user_id,
        )
        return ToolResult(
            success=True,
            output=(
                f"Screenshot captured ({len(png)} bytes).\n"
                f"Page: {summary['title']} — {summary['url'] or url}\n"
                f"Download: {info['download_url']}"
                + self._diag_text(summary)
            ),
            metadata={**summary, "artifact_id": info["id"],
                      "download_url": info["download_url"]},
        )


class BrowserActionTool(_AtpBrowserBase):
    _ACTIONS = ("click", "type", "press", "hover", "scroll", "back", "reload")

    @property
    def name(self) -> str:
        return "browser_action"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def description(self) -> str:
        return (
            "Interact with the current page in the server-side browser "
            "session: click, type (fill), press (a key), hover, scroll, "
            "back, reload. Returns the resulting page title and text."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(self._ACTIONS)},
                "selector": {
                    "type": "string",
                    "description": "CSS selector or snapshot @ref (click/type/hover)",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type, or key name for press (e.g. Enter)",
                },
                "url": {
                    "type": "string",
                    "description": "URL to open first (omit to act on the current page)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        action = str(kwargs.get("action") or "").strip().lower()
        if action not in self._ACTIONS:
            return ToolResult(
                success=False, validation_error=True,
                error=f"unknown action {action!r}; use one of {self._ACTIONS}",
            )
        selector = str(kwargs.get("selector") or "").strip()
        text = str(kwargs.get("text") or "")
        if action in ("click", "hover") and not selector:
            return ToolResult(success=False, validation_error=True,
                              error=f"action={action!r} requires 'selector'")
        if action == "type" and not (selector and text):
            return ToolResult(success=False, validation_error=True,
                              error="action='type' requires 'selector' and 'text'")
        if action == "press" and not text:
            return ToolResult(success=False, validation_error=True,
                              error="action='press' requires 'text' (the key name)")
        session = self._session(kwargs)
        err = await self._ensure_page(docker, session, str(kwargs.get("url") or "").strip())
        if err:
            return ToolResult(success=False, error=err)
        cli = {
            "click": ["click", selector],
            "type": ["fill", selector, text],
            "press": ["press", text],
            "hover": ["hover", selector],
            "scroll": ["scroll", "down"],
            "back": ["back"],
            "reload": ["reload"],
        }[action]
        res = await _sidecar.run_cli(docker, cli, session=session, timeout=20.0)
        if not res.get("ok"):
            return ToolResult(success=False, error=str(res.get("error") or f"{action} failed"))
        body = await _sidecar.run_cli(
            docker, ["get", "text", "body"], session=session, timeout=10.0
        )
        summary = await self._page_summary(docker, session)
        return ToolResult(
            success=True,
            output=(
                f"{action} ok.\nPage: {summary['title']} — {summary['url']}\n\n"
                + str(body.get("text") or "")[:2000]
                + self._diag_text(summary)
            ),
            metadata=summary,
        )


class BrowserEvaluateTool(_AtpBrowserBase):
    @property
    def name(self) -> str:
        return "browser_evaluate"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def description(self) -> str:
        return (
            "Run a JavaScript expression in the current page of the "
            "server-side browser session and return its JSON result. "
            "The expression may be a value ('document.title'), a "
            "function, or an async function."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "url": {
                    "type": "string",
                    "description": "URL to open first (omit to evaluate on the current page)",
                },
                "selector": {
                    "type": "string",
                    "description": "Bind 'el' to the first match of this CSS selector",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        expression = str(kwargs.get("expression") or "").strip()
        if not expression:
            return ToolResult(success=False, validation_error=True,
                              error="'expression' is required")
        from augmentum.coder.browser import _build_evaluate_wrapper

        session = self._session(kwargs)
        err = await self._ensure_page(docker, session, str(kwargs.get("url") or "").strip())
        if err:
            return ToolResult(success=False, error=err)
        selector = str(kwargs.get("selector") or "").strip()
        wrapper = _build_evaluate_wrapper(expression, with_element=bool(selector))
        if selector:
            sel_json = json.dumps(selector)
            code = (
                f"(() => {{ const ___el = document.querySelector({sel_json}); "
                f"if (!___el) return {{__aug_ok: false, error: {{message: "
                f"'selector not found: ' + {sel_json}, name: 'SelectorError'}}}}; "
                f"return ({wrapper})(___el, null); }})()"
            )
        else:
            code = f"({wrapper})(null)"
        res = await _sidecar.run_cli(docker, ["eval", code], session=session, timeout=30.0)
        if not res.get("ok"):
            return ToolResult(success=False, error=str(res.get("error") or "evaluate failed"))
        envelope = res.get("result") if "result" in res else res.get("value")
        if isinstance(envelope, dict) and "__aug_ok" in envelope:
            if not envelope.get("__aug_ok"):
                e = envelope.get("error") or {}
                return ToolResult(
                    success=False,
                    error=f"JS error: {e.get('message', '')}",
                    metadata={"error_detail": e},
                )
            value = envelope.get("value")
        else:
            value = envelope
        try:
            ser = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            ser = json.dumps(str(value))
        truncated = len(ser) > 50_000
        return ToolResult(
            success=True,
            output=ser[:50_000] + ("\n...(truncated)" if truncated else ""),
            metadata={"truncated": truncated},
        )


class BrowserWaitTool(_AtpBrowserBase):
    @property
    def name(self) -> str:
        return "browser_wait"

    @property
    def description(self) -> str:
        return (
            "Wait for a condition on the current page of the server-side "
            "browser session: a CSS selector, a text snippet, or (with "
            "neither) network idle. Returns the page state either way."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
                "timeout_ms": {"type": "integer", "default": 10000},
                "url": {
                    "type": "string",
                    "description": "URL to open first (omit to wait on the current page)",
                },
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        session = self._session(kwargs)
        err = await self._ensure_page(docker, session, str(kwargs.get("url") or "").strip())
        if err:
            return ToolResult(success=False, error=err)
        selector = str(kwargs.get("selector") or "").strip()
        text = str(kwargs.get("text") or "").strip()
        timeout_ms = max(250, min(60_000, int(kwargs.get("timeout_ms") or 10_000)))
        if selector:
            args = ["wait", selector]
        elif text:
            args = ["wait", "--text", text]
        else:
            args = ["wait", "--load", "networkidle"]
        start = time.time()
        res = await _sidecar.run_cli(
            docker, args, session=session, timeout=timeout_ms / 1000.0 + 10.0
        )
        summary = await self._page_summary(docker, session)
        waited_ms = int((time.time() - start) * 1000)
        if not res.get("ok"):
            return ToolResult(
                success=False,
                error=(
                    f"condition not met within {timeout_ms}ms: "
                    f"{str(res.get('error') or '')[:300]}"
                ),
                metadata={**summary, "waited_ms": waited_ms},
            )
        return ToolResult(
            success=True,
            output=(
                f"Condition met after {waited_ms}ms.\n"
                f"Page: {summary['title']} — {summary['url']}"
                + self._diag_text(summary)
            ),
            metadata={**summary, "waited_ms": waited_ms},
        )


class BrowserEnsureAuthTool(_AtpBrowserBase):
    """Authenticate the persistent browser session to THIS Augmentum instance
    without a login form — mint a session server-side from the caller's own
    identity and inject it as the ``augmentum_session`` cookie.

    The harness is already authenticated to ATP (it carries an API key), so
    driving the UI login form by hand — navigate, type user, type password,
    click — is pure ceremony repeated every session, and it forces secrets
    through the transcript. This collapses that whole preamble into one
    idempotent call: if the persistent context already holds a valid cookie,
    it is a no-op; otherwise it mints one and sets it. Reviewing any signed-in
    surface (gallery, library, settings) is then a single ``browser_navigate``.
    """

    # The sidecar reaches the host Augmentum as host.docker.internal; the UI
    # serves HTTPS on 6443 and the leaf cert carries that SAN (compose.yaml).
    _UI_BASE = "https://host.docker.internal:6443"

    @property
    def name(self) -> str:
        return "browser_ensure_auth"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def description(self) -> str:
        return (
            "Sign the persistent browser session in to THIS Augmentum "
            "instance as you (no login form, no password). Idempotent — a "
            "no-op when already authenticated. Call this ONCE before "
            "navigating to any signed-in UI surface; the returned ui_base is "
            "the URL to open (e.g. ui_base + '/ui/'). Removes the "
            "navigate-type-type-click login dance entirely."
        )

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def _session_manager(self):
        return getattr(self._app_state, "session_manager", None)

    async def _existing_cookie(self, docker, session: str) -> str:
        """Return the current augmentum_session cookie value, or ''."""
        res = await _sidecar.run_cli(
            docker, ["cookies", "get"], session=session, timeout=10.0
        )
        if not res.get("ok"):
            return ""
        # agent-browser --json shapes the cookie list under a few possible
        # keys across versions — be permissive.
        jar = res.get("cookies") or res.get("result") or res.get("value") or []
        if isinstance(jar, dict):
            jar = jar.get("cookies") or []
        if not isinstance(jar, list):
            return ""
        for c in jar:
            if isinstance(c, dict) and c.get("name") == "augmentum_session":
                return str(c.get("value") or "")
        return ""

    async def execute(self, **kwargs) -> ToolResult:
        docker = self._docker()
        if docker is None or not await _sidecar.is_available(docker):
            return self._unavailable()
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user to sign in as")
        sm = self._session_manager()
        if sm is None:
            return ToolResult(success=False, error="auth session manager is unavailable")
        session = self._session(kwargs)
        ui_url = f"{self._UI_BASE}/ui/"

        # A cookie only persists once a browsing context exists — with no page
        # open, agent-browser launches an ephemeral context and the cookie
        # evaporates. So establish the page FIRST, then read/inject.
        err = await self._ensure_page(docker, session, ui_url)
        if err:
            return ToolResult(success=False, error=f"could not open UI: {err}")

        # Idempotency: a valid, matching cookie means we're already signed in.
        existing = await self._existing_cookie(docker, session)
        if existing:
            try:
                user = await sm.validate_token(existing)
            except Exception:
                user = None
            if user is not None and getattr(user, "id", "") == user_id:
                return ToolResult(
                    success=True,
                    output=(
                        f"Already authenticated as {user_id}.\n"
                        f"ui_base: {self._UI_BASE} (open {ui_url})"
                    ),
                    metadata={"ui_base": self._UI_BASE, "authenticated": True,
                              "minted": False},
                )

        # Mint a fresh session bound to this user and inject it as the cookie.
        try:
            token = await sm.create_session(
                user_id, ip_address="atp-browser", user_agent="atp-browser",
                source="web",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"could not mint session: {exc}")
        res = await _sidecar.run_cli(
            docker,
            ["cookies", "set", "augmentum_session", token,
             "--url", self._UI_BASE, "--httpOnly", "--secure", "--sameSite", "Lax"],
            session=session, timeout=15.0,
        )
        if not res.get("ok"):
            return ToolResult(
                success=False,
                error=f"cookie injection failed: {res.get('error') or 'unknown'}",
            )
        # Reload so the freshly-set cookie rides the next request and the UI
        # renders signed-in rather than the login page.
        await _sidecar.run_cli(docker, ["reload"], session=session, timeout=20.0)
        return ToolResult(
            success=True,
            output=(
                f"Signed in as {user_id}.\n"
                f"ui_base: {self._UI_BASE} — open {ui_url} (already "
                f"authenticated; no login form needed)."
            ),
            metadata={"ui_base": self._UI_BASE, "authenticated": True,
                      "minted": True},
        )


ATP_BROWSER_TOOL_CLASSES = (
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserActionTool,
    BrowserEvaluateTool,
    BrowserWaitTool,
    BrowserEnsureAuthTool,
)
