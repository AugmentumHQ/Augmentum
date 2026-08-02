"""Service, browser, and profile tools for Coder."""

from __future__ import annotations

import json
from typing import Any

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_MAX_OUTPUT_CHARS = 50_000
_TIMEOUT_MIN_MS = 500
_TIMEOUT_MAX_MS = 60_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(text)} total chars)"


def _clamp_timeout(value, default_ms: int) -> int:
    """Clamp a model-supplied timeout to the [_TIMEOUT_MIN_MS, _TIMEOUT_MAX_MS]
    window. Accepts int, str, or None; falls back to default on any junk."""
    if value is None or value == "":
        return default_ms
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return default_ms
    if ms < _TIMEOUT_MIN_MS:
        return _TIMEOUT_MIN_MS
    if ms > _TIMEOUT_MAX_MS:
        return _TIMEOUT_MAX_MS
    return ms


class _RuntimeCoderTool(Tool):
    # Runtime tools (service/browser/terminal/http/db) drive the Docker
    # container directly via ``self._cm`` — they bypass the WorkspaceExecutor
    # abstraction that the file/shell/git tools use. That means they CANNOT
    # work when there is no container (the ACP "loop in the editor" path, where
    # container_manager is None). ``create_coder_tools`` filters these out in
    # that mode so the model is never offered a tool that would crash on a
    # ``None._cm`` access. Subclasses that touch only profile_store/state
    # (ProfileRead/ProfileUpdate/Observe) override this to False.
    requires_container: bool = True

    def __init__(
        self,
        *,
        container_manager,
        workspace_id: str,
        state,
        executor=None,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        self._cm = container_manager
        self._workspace_id = workspace_id
        # Accept the injected WorkspaceExecutor for parity with _CoderTool.
        # Runtime tools (service/browser/terminal) drive container_manager
        # directly today, so we only store it — but the ctor MUST accept the
        # kwarg because create_coder_tools() passes ``executor`` to EVERY tool
        # when the editor/remote path is active. Without this the whole coder
        # turn crashes before the model's tool loop even starts.
        self._executor = executor
        self._state = state
        self._profile_store = profile_store
        self._service_store = service_store
        self._user_id = user_id or ""
        self._strict_edit_guard = bool(strict_edit_guard)

    @property
    def cacheable(self) -> bool:
        return False


class ServiceStartTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "service_start"

    @property
    def description(self) -> str:
        return (
            "Start a long-running dev service in the workspace, track its pid, "
            "ports, and logs, and make it visible through service_list/logs/probe."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "name": {"type": "string"},
                "cwd": {"type": "string", "default": "/workspace"},
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
                "ports": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(self, *, command: str = "", name: str = "", cwd: str = "/workspace", env: dict | None = None, ports: list | None = None, **_kwargs) -> ToolResult:
        try:
            from augmentum.coder.services import WorkspaceServiceManager, service_result

            svc = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).start(command=command, name=name, cwd=cwd, env=env or {}, ports=ports or [])
            return service_result(svc)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to start service: {exc}")


class ServiceListTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "service_list"

    @property
    def description(self) -> str:
        return "List managed workspace services with status, pids, ports, and log paths."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **_kwargs) -> ToolResult:
        from augmentum.coder.services import WorkspaceServiceManager

        try:
            services = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).list()
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to list services: {exc}")
        if not services:
            return ToolResult(success=True, output="No managed services are registered.", metadata={"services": []})
        lines = ["Managed services:"]
        for svc in services:
            ports = ", ".join(str(p) for p in svc.ports) or "none"
            lines.append(f"- {svc.id} {svc.name}: {svc.status}, pid {svc.pid}, ports {ports}, logs {svc.log_path}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"services": [svc.to_dict() for svc in services]})


class ServiceLogsTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "service_logs"

    @property
    def description(self) -> str:
        return "Tail stdout/stderr logs for a managed workspace service."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "service_id": {"type": "string"},
                "lines": {"type": "integer", "default": 120, "minimum": 1, "maximum": 2000},
            },
            "required": ["service_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, service_id: str = "", lines: int = 120, **_kwargs) -> ToolResult:
        if not service_id:
            return ToolResult(success=False, error="service_logs requires service_id", validation_error=True)
        from augmentum.coder.services import WorkspaceServiceManager

        try:
            output = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).logs(service_id, lines=lines)
            return ToolResult(success=True, output=_truncate(output or "(service log is empty)"))
        except KeyError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to read service logs: {exc}")


class ServiceStopTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "service_stop"

    @property
    def description(self) -> str:
        return "Stop a managed long-running workspace service by service_id."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"service_id": {"type": "string"}},
            "required": ["service_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, service_id: str = "", **_kwargs) -> ToolResult:
        if not service_id:
            return ToolResult(success=False, error="service_stop requires service_id", validation_error=True)
        from augmentum.coder.services import WorkspaceServiceManager, service_result

        try:
            svc = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).stop(service_id)
            return service_result(svc)
        except KeyError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to stop service: {exc}")


class ServiceProbeTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "service_probe"

    @property
    def description(self) -> str:
        return "Probe a managed service, URL, or TCP port for readiness and recent errors."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "service_id": {"type": "string"},
                "url": {"type": "string"},
                "port": {"type": "integer"},
                "timeout": {"type": "number", "default": 5},
            },
            "additionalProperties": False,
        }

    async def execute(self, *, service_id: str = "", url: str = "", port: int | None = None, timeout: float = 5, **_kwargs) -> ToolResult:
        from augmentum.coder.services import WorkspaceServiceManager

        try:
            probe = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).probe(service_id=service_id, url=url, port=port, timeout=timeout)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:
            return ToolResult(success=False, error=f"Service probe failed: {exc}")
        ok = bool(probe.get("ok"))
        return ToolResult(
            success=ok,
            output=f"Probe {'passed' if ok else 'failed'}.\n{json.dumps(probe, indent=2, sort_keys=True)}",
            error="" if ok else str(probe.get("error") or "probe failed"),
            metadata={"probe": probe},
        )


def _render_snapshot_lines(snap: dict, url: str) -> list[str]:
    """Human-readable page summary shared by browser_open and
    browser_snapshot — open returns the page state directly (ledger
    mining: 72% of opens were immediately followed by a look)."""
    lines = [
        f"URL: {snap.get('reachable_url') or url}",
        f"Status: {snap.get('status')}",
        f"Title: {snap.get('title') or '(untitled)'}",
        "Visible elements:",
    ]
    summary = snap.get("summary") or []
    for item in summary[:20]:
        if isinstance(item, dict):
            lines.append(f"- {item.get('tag')}: {item.get('text')}")
    if not summary:
        lines.append("- (no text summary extracted)")
    return lines


class BrowserOpenTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_open"

    @property
    def description(self) -> str:
        return (
            "Open and remember a workspace preview URL for browser "
            "verification. Returns the page snapshot (status, title, "
            "visible elements) directly — no need to call "
            "browser_snapshot right after."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"url": {"type": "string"}}, "additionalProperties": False}

    async def execute(self, *, url: str = "", **_kwargs) -> ToolResult:
        from augmentum.coder.browser import http_snapshot, infer_preview_url, save_browser_session

        url = (url or "").strip() or await infer_preview_url(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No URL provided and no listening preview port was detected.", validation_error=True)
        await save_browser_session(self._cm, self._workspace_id, url)
        snap = await http_snapshot(self._cm, self._workspace_id, url)
        ok = bool(snap.get("ok"))
        return ToolResult(
            success=ok,
            output=f"Opened {url}.\n" + "\n".join(_render_snapshot_lines(snap, url)),
            error="" if ok else str(snap.get("error") or "preview not reachable"),
            metadata={"browser": snap, "url": url},
        )


class BrowserSnapshotTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_snapshot"

    @property
    def description(self) -> str:
        return "Return title, URL, DOM summary, console errors, and network failures for the open preview."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"url": {"type": "string"}}, "additionalProperties": False}

    async def execute(self, *, url: str = "", **_kwargs) -> ToolResult:
        from augmentum.coder.browser import http_snapshot, load_browser_session

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        # Sidecar-first: an accessibility-tree snapshot with stable @refs
        # (click/fill/get accept them directly) beats the regex-over-HTML
        # summary in every way. Falls through to http_snapshot when the
        # sidecar isn't running.
        try:
            from augmentum.coder import browser_sidecar as _bs
            if await _bs.is_available(self._cm._docker):
                a11y = await _bs.snapshot_a11y(self._cm, self._workspace_id, url=url)
                if a11y.get("engine") == "sidecar" and a11y.get("ok"):
                    lines = [
                        f"URL: {a11y.get('url') or url}",
                        f"Title: {a11y.get('title') or '(untitled)'}",
                        "Interactive elements (use @refs with browser_click/"
                        "browser_type/browser_get/browser_interact):",
                    ]
                    refs = a11y.get("refs") or {}
                    for ref, info in list(refs.items())[:60]:
                        if isinstance(info, dict):
                            lines.append(
                                f"- @{ref}: {info.get('role', '?')} "
                                f"{json.dumps(info.get('name', ''))[:120]}"
                            )
                    if not refs:
                        lines.append("- (no interactive elements found)")
                    for err in (a11y.get("console_errors") or [])[:10]:
                        lines.append(f"  [console.{err.get('type')}] {err.get('text', '')}")
                    for fail in (a11y.get("network_failures") or [])[:10]:
                        lines.append(
                            f"  [network {fail.get('status') or fail.get('failure')}] "
                            f"{fail.get('url', '')}"
                        )
                    return ToolResult(
                        success=True,
                        output="\n".join(lines),
                        metadata={"browser": a11y},
                    )
        except Exception:
            log.warning("browser_snapshot_sidecar_failed", exc_info=True)
        snap = await http_snapshot(self._cm, self._workspace_id, url)
        lines = _render_snapshot_lines(snap, url)
        # Fold in console/error events captured from the USER's LIVE preview
        # session. This tool's own snapshot is a fresh headless cold-load and
        # structurally misses interaction/session/auth-state errors; the live
        # buffer is the only place those surface. See preview_console.py.
        live_errors: list[dict] = []
        try:
            from augmentum.coder.preview_console import snapshot as _pc_snapshot
            live_errors = _pc_snapshot(self._workspace_id, limit=20)
        except Exception:
            live_errors = []
        if live_errors:
            lines.append(
                f"\nLive preview console — {len(live_errors)} recent event(s) "
                f"from the user's session:"
            )
            for e in live_errors:
                loc = f" ({e['url']}:{e['line']})" if e.get("line") else ""
                lines.append(f"  [{e.get('type', 'error')}] {e.get('text', '')}{loc}")
        meta: dict = {"browser": snap}
        if live_errors:
            meta["live_console"] = live_errors
        return ToolResult(
            success=bool(snap.get("ok")),
            output="\n".join(lines),
            error="" if snap.get("ok") else str(snap.get("error") or "snapshot failed"),
            metadata=meta,
        )


class BrowserClickTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return "Click a selector in the open preview using Playwright when browser tooling is available."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "url": {"type": "string"},
                "wait_for": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for BEFORE clicking — "
                        "use when the target mounts late (SPA hydration, "
                        "async data). Fixes most 'selector not found' misses."
                    ),
                },
            },
            "required": ["selector"],
            "additionalProperties": False,
        }

    async def execute(
        self, *, selector: str = "", url: str = "", wait_for: str = "", **_kwargs,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="browser_click requires selector", validation_error=True)
        from augmentum.coder.browser import load_browser_session, playwright_action

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        result = await playwright_action(
            self._cm, self._workspace_id, url=url, action="click",
            selector=selector,
            wait_for_selector=(wait_for or "").strip() or selector,
        )
        return ToolResult(
            success=bool(result.get("ok")),
            output=f"Clicked {selector}." if result.get("ok") else "Click failed.",
            error="" if result.get("ok") else str(result.get("error") or "Playwright click failed"),
            metadata={"browser": result},
        )


class BrowserTypeTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_type"

    @property
    def description(self) -> str:
        return "Type text into a selector in the open preview using Playwright when available."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
                "url": {"type": "string"},
                "wait_for": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for BEFORE typing — "
                        "use when the field mounts late. For several fields "
                        "at once, prefer browser_fill_form."
                    ),
                },
            },
            "required": ["selector", "text"],
            "additionalProperties": False,
        }

    async def execute(
        self, *, selector: str = "", text: str = "", url: str = "",
        wait_for: str = "", **_kwargs,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="browser_type requires selector", validation_error=True)
        from augmentum.coder.browser import load_browser_session, playwright_action

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        result = await playwright_action(
            self._cm, self._workspace_id, url=url, action="type",
            selector=selector, text=text,
            wait_for_selector=(wait_for or "").strip() or selector,
        )
        return ToolResult(
            success=bool(result.get("ok")),
            output=f"Typed into {selector}." if result.get("ok") else "Type action failed.",
            error="" if result.get("ok") else str(result.get("error") or "Playwright type failed"),
            metadata={"browser": result},
        )


class BrowserScreenshotTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the open preview URL using Playwright when available. "
            "The default is fast and best-effort for responsive visual iteration: it waits for "
            "DOMContentLoaded, briefly settles fonts/paint, captures full-page when possible, "
            "and falls back to a viewport screenshot with warnings if readiness or full-page "
            "capture is slow. Also collects console errors + failed network requests."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def timeout(self) -> float:
        # Above the helper's own phase-bounded subprocess timeout so the
        # tool returns structured degraded output instead of being preempted.
        return 100.0

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 800},
                "wait_for_selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for before screenshotting. "
                        "If it times out, the tool still captures the current page "
                        "and marks the result degraded."
                    ),
                },
                "wait_until": {
                    "type": "string",
                    "enum": ["domcontentloaded", "load", "networkidle"],
                    "default": "domcontentloaded",
                    "description": (
                        "Navigation readiness signal. Use networkidle only when "
                        "the page has no long-lived sockets/HMR/streaming requests."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": _TIMEOUT_MIN_MS,
                    "maximum": _TIMEOUT_MAX_MS,
                    "default": 15000,
                    "description": "Per-readiness timeout, 500-60000 ms.",
                },
                "full_page": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Capture the full scrollable page when possible. If that "
                        "fails, the tool falls back to a viewport screenshot."
                    ),
                },
                "settle_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5000,
                    "default": 250,
                    "description": "Small post-load paint/font settle delay before capture.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        url: str = "",
        width: int = 1280,
        height: int = 800,
        wait_for_selector: str = "",
        wait_until: str = "domcontentloaded",
        timeout_ms: int | None = None,
        full_page: bool = True,
        settle_ms: int | None = None,
        **_kwargs,
    ) -> ToolResult:
        from augmentum.coder.browser import load_browser_session, playwright_screenshot

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)

        wait_until_norm = (wait_until or "domcontentloaded").strip().lower()
        if wait_until_norm not in {"domcontentloaded", "load", "networkidle"}:
            wait_until_norm = "domcontentloaded"
        timeout_ms = _clamp_timeout(timeout_ms, 15_000)
        try:
            settle_ms_int = int(250 if settle_ms is None else settle_ms)
        except (TypeError, ValueError):
            settle_ms_int = 250
        settle_ms_int = max(0, min(5_000, settle_ms_int))
        if isinstance(full_page, str):
            full_page_bool = full_page.strip().lower() not in ("0", "false", "no", "off")
        else:
            full_page_bool = bool(full_page)

        result = await playwright_screenshot(
            self._cm,
            self._workspace_id,
            url=url,
            viewport={"width": int(width or 1280), "height": int(height or 800)},
            wait_for_selector=(wait_for_selector or "").strip(),
            wait_until=wait_until_norm,
            timeout_ms=timeout_ms,
            full_page=full_page_bool,
            settle_ms=settle_ms_int,
        )
        # Build a richer output string that surfaces runtime errors next
        # to the screenshot path -- without this, the agent sees "ok"
        # plus a file path and assumes the page is healthy even when
        # the console is full of red.
        lines: list[str] = []
        warnings_raw = result.get("warnings") or []
        warning_texts = []
        for w in warnings_raw:
            if isinstance(w, dict):
                phase = str(w.get("phase") or "capture")
                detail = str(w.get("error") or "degraded")
                warning_texts.append(f"{phase}: {detail}")
            else:
                warning_texts.append(str(w))

        if result.get("ok"):
            mode = "full-page" if result.get("full_page") else "viewport"
            lines.append(f"Screenshot captured at {result.get('path')} ({mode}).")
            if result.get("degraded") or warning_texts:
                lines.append("Capture degraded but usable:")
                for w in warning_texts[:5]:
                    lines.append(f"  - {w}")
                if len(warning_texts) > 5:
                    lines.append(f"  - ... +{len(warning_texts) - 5} more")
        else:
            lines.append("Screenshot failed.")
            if result.get("path"):
                lines.append(f"Attempted path: {result.get('path')}")
            for w in warning_texts[:5]:
                lines.append(f"  - {w}")
        ce = result.get("console_errors") or []
        nf = result.get("network_failures") or []
        if ce:
            lines.append(f"\nConsole ({len(ce)} {'errors' if len(ce) != 1 else 'error'}/warnings):")
            for e in ce[:5]:
                lines.append(f"  [{e.get('type', '?')}] {e.get('text', '')}")
            if len(ce) > 5:
                lines.append(f"  ... +{len(ce) - 5} more")
        if nf:
            lines.append(f"\nNetwork ({len(nf)} {'failures' if len(nf) != 1 else 'failure'}):")
            for f in nf[:5]:
                bits = [str(f.get("status", f.get("failure", "?"))), f.get("method", ""), f.get("url", "")]
                lines.append("  " + " ".join(b for b in bits if b))
            if len(nf) > 5:
                lines.append(f"  ... +{len(nf) - 5} more")
        return ToolResult(
            success=bool(result.get("ok")),
            output="\n".join(lines),
            error="" if result.get("ok") else str(result.get("error") or "Playwright screenshot failed"),
            metadata={"browser": result},
            warnings=warning_texts if result.get("ok") else [],
        )


class BrowserEvaluateTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_evaluate"

    @property
    def description(self) -> str:
        return (
            "Run an arbitrary JavaScript expression in the open preview and "
            "return its result. Use for verifying runtime state a screenshot "
            "wouldn't show: hydration completion, store/redux state, "
            "computed DOM properties, fetch results, localStorage values, "
            "feature-flag evaluation. Expression can be a value "
            "(`document.title`), a function (`(arg) => ...`), or async. "
            "JS exceptions return structured {message, name, stack, line, "
            "column}. The value is JSON-serialized with structure-aware "
            "trimming (DOM nodes still serialize to null — return their "
            "properties instead). Bind args with `args`; scope to an "
            "element with `selector` (binds `el` inside the expression)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "JavaScript to run in the page. Shapes:\n"
                        " - value:   `document.title`\n"
                        " - fn:      `(arg) => document.querySelectorAll(arg.sel).length`\n"
                        " - async:   `async () => (await fetch('/api/health')).status`\n"
                        " - scoped:  with `selector` set, `el.textContent.trim()` or "
                        "`(el, arg) => el.dataset[arg.key]`."
                    ),
                },
                "args": {
                    "description": (
                        "Optional JSON value bound to `arg` inside the "
                        "expression. Lets you parameterize the same probe "
                        "without string-concat injecting values into JS."
                    ),
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector. When set the expression is "
                        "evaluated against the first matching element and "
                        "`el` is bound to it. Returns selector_missing=true "
                        "if no element matches within the evaluate timeout."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": "Optional override; defaults to the session URL set by browser_open.",
                },
                "wait_for_selector": {
                    "type": "string",
                    "description": "Optional CSS selector to wait for before evaluating — useful for SPAs that hydrate after networkidle.",
                },
                "goto_timeout_ms": {
                    "type": "integer",
                    "minimum": _TIMEOUT_MIN_MS,
                    "maximum": _TIMEOUT_MAX_MS,
                    "description": "How long page.goto waits for networkidle (default 15000).",
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": _TIMEOUT_MIN_MS,
                    "maximum": _TIMEOUT_MAX_MS,
                    "description": "How long the evaluate itself waits (default 15000).",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        expression: str = "",
        args: Any = None,
        selector: str = "",
        url: str = "",
        wait_for_selector: str = "",
        goto_timeout_ms: int | None = None,
        timeout_ms: int | None = None,
        **_kwargs,
    ) -> ToolResult:
        expr = (expression or "").strip()
        if not expr:
            return ToolResult(
                success=False,
                error="browser_evaluate requires expression",
                validation_error=True,
            )
        goto_ms = _clamp_timeout(goto_timeout_ms, 15_000)
        eval_ms = _clamp_timeout(timeout_ms, 15_000)

        from augmentum.coder.browser import load_browser_session, playwright_evaluate

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(
                success=False,
                error="No browser URL is open. Call browser_open first.",
                validation_error=True,
            )
        result = await playwright_evaluate(
            self._cm,
            self._workspace_id,
            url=url,
            expression=expr,
            args=args,
            selector=(selector or "").strip(),
            wait_for_selector=(wait_for_selector or "").strip(),
            goto_timeout_ms=goto_ms,
            timeout_ms=eval_ms,
        )
        lines: list[str] = []
        if result.get("ok"):
            ser = result.get("result_json") or "null"
            result_type = result.get("result_type") or "unknown"
            if result.get("truncated"):
                lines.append(f"Result ({result_type}, truncated to fit 50KB budget):")
            else:
                lines.append(f"Result ({result_type}):")
            lines.append(ser)
        elif result.get("js_error"):
            # Structured JS error: surface message + line/column + stack.
            detail = result.get("error_detail") or {}
            name = detail.get("name") or "Error"
            msg = detail.get("message") or result.get("error") or "unknown error"
            line = detail.get("line")
            col = detail.get("column")
            loc = ""
            if line is not None:
                loc = f" (line {line}" + (f", col {col})" if col is not None else ")")
            lines.append(f"JS {name}{loc}: {msg}")
            stack = (detail.get("stack") or "").strip()
            if stack:
                lines.append("Stack:")
                for stack_line in stack.splitlines()[:6]:
                    lines.append(f"  {stack_line}")
        elif result.get("selector_missing"):
            lines.append(
                "Selector did not match any element within the evaluate "
                f"timeout: {(selector or '').strip()!r}"
            )
        elif result.get("wrapper_error"):
            lines.append(
                "Expression failed to parse as JS (the wrapper itself "
                "couldn't be compiled). Check syntax — runtime errors "
                "are caught and surfaced separately."
            )
        else:
            lines.append(
                "Evaluation failed: " + str(result.get("error") or "unknown error"),
            )
        # Mirror browser_screenshot's "surface runtime errors next to the
        # primary output" pattern — a JS expression that "succeeded" with
        # console errors firing during goto is misleading otherwise.
        ce = result.get("console_errors") or []
        nf = result.get("network_failures") or []
        if ce:
            lines.append(f"\nConsole ({len(ce)} {'errors' if len(ce) != 1 else 'error'}/warnings):")
            for e in ce[:5]:
                lines.append(f"  [{e.get('type', '?')}] {e.get('text', '')}")
            if len(ce) > 5:
                lines.append(f"  … +{len(ce) - 5} more")
        if nf:
            lines.append(f"\nNetwork ({len(nf)} {'failures' if len(nf) != 1 else 'failure'}):")
            for f in nf[:5]:
                bits = [str(f.get("status", f.get("failure", "?"))), f.get("method", ""), f.get("url", "")]
                lines.append("  " + " ".join(b for b in bits if b))
            if len(nf) > 5:
                lines.append(f"  … +{len(nf) - 5} more")
        # Error string for ToolResult.error: prefer structured JS message
        # when present so the model's error log shows the actual cause,
        # not "Playwright evaluate failed".
        if result.get("ok"):
            err_str = ""
        elif result.get("js_error"):
            detail = result.get("error_detail") or {}
            err_str = f"{detail.get('name', 'Error')}: {detail.get('message', '')}".strip(": ")
        elif result.get("selector_missing"):
            err_str = f"selector not found: {(selector or '').strip()}"
        else:
            err_str = str(result.get("error") or "Playwright evaluate failed")
        return ToolResult(
            success=bool(result.get("ok")),
            output=_truncate("\n".join(lines)),
            error=err_str,
            metadata={"browser_evaluate": result},
        )


class BrowserWaitTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_wait"

    @property
    def description(self) -> str:
        return (
            "Wait for a page condition: a CSS selector (with state "
            "visible/attached/hidden/detached), a text string appearing in "
            "the page, or — with neither — network idle. Use this instead "
            "of setTimeout sleeps inside browser_evaluate. On timeout it "
            "returns the CURRENT page text so you can see what the page "
            "actually shows."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "selector": {"type": "string", "description": "CSS selector to wait for."},
                "text": {"type": "string", "description": "Text to wait for in the page body."},
                "state": {
                    "type": "string",
                    "enum": ["visible", "attached", "hidden", "detached"],
                    "default": "visible",
                    "description": "Selector state to await (with 'selector').",
                },
                "timeout_ms": {
                    "type": "integer", "default": 10000,
                    "description": "Max wait, 250-60000 ms.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self, *, url: str = "", selector: str = "", text: str = "",
        state: str = "visible", timeout_ms: int = 10_000, **_kwargs,
    ) -> ToolResult:
        from augmentum.coder.browser import load_browser_session, playwright_wait

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        result = await playwright_wait(
            self._cm, self._workspace_id, url=url,
            selector=(selector or "").strip(), text=(text or "").strip(),
            state=state, timeout_ms=timeout_ms,
        )
        ok = bool(result.get("ok"))
        cond = (
            f"selector {selector!r} {state}" if (selector or "").strip()
            else (f"text {text!r} present" if (text or "").strip() else "network idle")
        )
        lines = [
            (f"Condition met after {result.get('waited_ms', '?')}ms: {cond}."
             if ok else f"NOT met: {cond}."),
        ]
        if result.get("title"):
            lines.append(f"Title: {result.get('title')}")
        if not ok and result.get("body_preview"):
            lines.append(f"Current page text: {result.get('body_preview')}")
        return ToolResult(
            success=ok,
            output=_truncate("\n".join(lines)),
            error="" if ok else str(result.get("error") or "wait condition not met"),
            metadata={"browser": result},
        )


class BrowserExtractTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_extract"

    @property
    def description(self) -> str:
        return (
            "Extract structured data from the open page as JSON — use this "
            "instead of hand-writing querySelectorAll loops in "
            "browser_evaluate. Kinds: 'text' (element text), 'links' "
            "([{text, href}]), 'table' (headers + rows), 'list' (list "
            "items), 'meta' (title/meta tags/headings), 'attr' (an "
            "attribute's values; pass 'attribute'). Scope any kind with "
            "'selector'."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["text", "links", "table", "list", "meta", "attr"],
                    "default": "text",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to scope extraction (each kind has a sensible default).",
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute name (required for kind='attr').",
                },
                "limit": {"type": "integer", "default": 50, "description": "Max elements/rows (1-200)."},
            },
            "additionalProperties": False,
        }

    async def execute(
        self, *, url: str = "", kind: str = "text", selector: str = "",
        attribute: str = "", limit: int = 50, **_kwargs,
    ) -> ToolResult:
        from augmentum.coder.browser import load_browser_session, playwright_extract

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        result = await playwright_extract(
            self._cm, self._workspace_id, url=url, kind=kind,
            selector=selector, attribute=attribute, limit=limit,
        )
        ok = bool(result.get("ok"))
        if not ok:
            return ToolResult(
                success=False,
                error=str(result.get("error") or "extraction failed"),
                metadata={"browser": result},
            )
        raw = result.get("result_json") or "null"
        try:
            pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = str(raw)
        head = f"Extracted kind={result.get('kind')}"
        if result.get("fallback") == "http":
            head += " (plain-HTTP fallback — JS-rendered content not visible)"
        return ToolResult(
            success=True,
            output=_truncate(f"{head}:\n{pretty}"),
            metadata={"browser": result},
        )


class BrowserFillFormTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_fill_form"

    @property
    def description(self) -> str:
        return (
            "Fill several form fields and optionally submit — ONE call "
            "instead of a browser_type per field. 'fields' maps CSS "
            "selectors to values (string fills inputs/textareas/selects; "
            "boolean checks/unchecks). 'submit' is clicked only when every "
            "field succeeded. Optionally wait afterwards for a selector or "
            "text (e.g. the success message)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "fields": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                    "description": "Map of CSS selector -> value to fill.",
                },
                "submit": {
                    "type": "string",
                    "description": "CSS selector of the submit control to click after filling.",
                },
                "wait_after_selector": {
                    "type": "string",
                    "description": "Selector to wait for after submit (e.g. '.success-toast').",
                },
                "wait_after_text": {
                    "type": "string",
                    "description": "Text to wait for after submit (e.g. 'Saved').",
                },
                "timeout_ms": {"type": "integer", "default": 15000},
            },
            "required": ["fields"],
            "additionalProperties": False,
        }

    async def execute(
        self, *, url: str = "", fields: dict | None = None, submit: str = "",
        wait_after_selector: str = "", wait_after_text: str = "",
        timeout_ms: int = 15_000, **_kwargs,
    ) -> ToolResult:
        if not isinstance(fields, dict) or not fields:
            return ToolResult(
                success=False,
                error=(
                    "browser_fill_form requires 'fields': an object mapping "
                    "CSS selectors to values, e.g. "
                    '{"fields": {"#email": "a@b.c", "#agree": true}, '
                    '"submit": "button[type=submit]"}'
                ),
                validation_error=True,
            )
        from augmentum.coder.browser import load_browser_session, playwright_fill_form

        url = (url or "").strip() or await load_browser_session(self._cm, self._workspace_id)
        if not url:
            return ToolResult(success=False, error="No browser URL is open. Call browser_open first.", validation_error=True)
        result = await playwright_fill_form(
            self._cm, self._workspace_id, url=url, fields=fields,
            submit=(submit or "").strip(),
            wait_after_selector=(wait_after_selector or "").strip(),
            wait_after_text=(wait_after_text or "").strip(),
            timeout_ms=timeout_ms,
        )
        ok = bool(result.get("ok"))
        lines = []
        for f in result.get("fields") or []:
            mark = "ok" if f.get("ok") else f"FAILED: {f.get('error', '')}"
            lines.append(f"  {f.get('selector')}: {mark}")
        head = f"Filled {sum(1 for f in (result.get('fields') or []) if f.get('ok'))}/{len(fields)} fields."
        if (submit or "").strip():
            head += " Submitted." if result.get("submitted") else f" Submit NOT clicked ({result.get('submit_error', '')})."
        if result.get("wait_error"):
            lines.append(f"  after-wait: {result.get('wait_error')}")
        if result.get("body_preview"):
            lines.append(f"Page now shows: {str(result.get('body_preview'))[:600]}")
        return ToolResult(
            success=ok,
            output=_truncate("\n".join([head] + lines)),
            error="" if ok else str(result.get("error") or result.get("submit_error") or result.get("wait_error") or "fill_form failed"),
            metadata={"browser": result},
        )


class HttpRequestTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return (
            "Make a structured HTTP request from inside the workspace. "
            "Use this instead of hand-rolling curl in shell_exec when you "
            "need to probe an API endpoint, verify auth, or check that a "
            "service you started is responding correctly. Returns status, "
            "headers, body (truncated), and final URL after redirects. "
            "Supports GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS with headers "
            "+ body + redirect control."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {
                    "type": "string",
                    "default": "GET",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Request headers as key/value pairs.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Request body as a string (JSON should be pre-serialized). "
                        "Ignored for GET/HEAD."
                    ),
                },
                "follow_redirects": {"type": "boolean", "default": True},
                "verify_tls": {
                    "type": "boolean",
                    "default": True,
                    "description": "Set false ONLY for local self-signed dev certs.",
                },
                "timeout": {"type": "number", "default": 15.0},
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        url: str = "",
        method: str = "GET",
        headers: dict | None = None,
        body: str = "",
        follow_redirects: bool = True,
        verify_tls: bool = True,
        timeout: float = 15.0,
        **_kwargs,
    ) -> ToolResult:
        if not (url or "").strip():
            return ToolResult(
                success=False,
                error="http_request requires url",
                validation_error=True,
            )
        from augmentum.coder.http_probe import run_http_request

        result = await run_http_request(
            self._cm,
            self._workspace_id,
            url=url,
            method=method,
            headers=headers or {},
            body=body or "",
            timeout=float(timeout or 15.0),
            follow_redirects=bool(follow_redirects),
            verify_tls=bool(verify_tls),
        )
        if result.get("validation_error"):
            return ToolResult(
                success=False,
                error=str(result.get("error") or "validation failed"),
                validation_error=True,
            )
        ok = bool(result.get("ok"))
        status = result.get("status") or 0
        lines = [
            f"{method.upper()} {result.get('final_url', url)} → {status} "
            f"{result.get('reason', '') or ''} ({result.get('latency_ms', 0)}ms)",
        ]
        # Header preview — full headers in metadata for callers that need
        # them, top few in output so the model can see content-type /
        # location / set-cookie at a glance.
        hdrs = result.get("headers") or {}
        if hdrs:
            preview_keys = (
                "content-type", "content-length", "location",
                "www-authenticate", "set-cookie", "cache-control",
            )
            lower = {k.lower(): (k, v) for k, v in hdrs.items()}
            shown = []
            for k in preview_keys:
                if k in lower:
                    raw_k, raw_v = lower[k]
                    shown.append(f"  {raw_k}: {raw_v}")
            if shown:
                lines.append("Headers:")
                lines.extend(shown)
        body_text = result.get("body") or ""
        if body_text:
            lines.append("")
            lines.append("Body" + (" (truncated)" if result.get("body_truncated") else "") + ":")
            lines.append(body_text)
        if result.get("error") and not ok:
            lines.append("")
            lines.append(f"Error: {result['error']}")
        return ToolResult(
            success=ok,
            output=_truncate("\n".join(lines)),
            error="" if ok else str(result.get("error") or f"HTTP {status}"),
            metadata={"http_request": result},
        )


class DbInspectTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "db_inspect"

    @property
    def description(self) -> str:
        return (
            "Read-only inspection of a SQLite database file in the "
            "workspace. Use to verify migrations landed, sample app "
            "data, check integrity, or run a focused SELECT. Actions: "
            "schema (CREATE statements), tables (name + row count), "
            "sample (top N rows from a table), query (caller-supplied "
            "SELECT with row cap), integrity (PRAGMA integrity_check). "
            "Writes are refused — use shell_exec / file_write for those."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "description": "Absolute path to the .db / .sqlite file inside the workspace.",
                },
                "action": {
                    "type": "string",
                    "enum": ["schema", "tables", "sample", "query", "integrity"],
                    "default": "schema",
                },
                "table": {
                    "type": "string",
                    "description": "Required when action=sample.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Row cap for sample/query (max 200).",
                },
                "query": {
                    "type": "string",
                    "description": "SELECT/WITH/EXPLAIN/PRAGMA only. Required when action=query.",
                },
            },
            "required": ["db_path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        db_path: str = "",
        action: str = "schema",
        table: str = "",
        limit: int = 5,
        query: str = "",
        **_kwargs,
    ) -> ToolResult:
        from augmentum.coder.db_probe import run_db_inspect

        result = await run_db_inspect(
            self._cm,
            self._workspace_id,
            db_path=db_path,
            action=action,
            table=table,
            limit=int(limit or 5),
            query=query,
        )
        if result.get("validation_error"):
            return ToolResult(
                success=False,
                error=str(result.get("error") or "validation failed"),
                validation_error=True,
            )
        ok = bool(result.get("ok"))
        lines = [f"db: {db_path}  action: {action}"]
        if action == "schema":
            items = result.get("schema") or []
            lines.append(f"Objects: {len(items)}")
            for item in items[:20]:
                sql = (item.get("sql") or "").strip()
                lines.append(f"\n-- {item.get('type')} {item.get('name')}")
                if sql:
                    lines.append(sql)
            if len(items) > 20:
                lines.append(f"\n… +{len(items) - 20} more objects")
        elif action == "tables":
            tables = result.get("tables") or []
            lines.append(f"Tables: {len(tables)}")
            for t in tables[:30]:
                rows = t.get("rows", -1)
                rows_s = "?" if rows < 0 else str(rows)
                lines.append(f"  {t.get('name')}  ({rows_s} rows)")
            if len(tables) > 30:
                lines.append(f"  … +{len(tables) - 30} more")
        elif action in ("sample", "query"):
            cols = result.get("columns") or []
            rows = result.get("rows") or []
            lines.append(f"Columns: {', '.join(cols) if cols else '(none)'}")
            lines.append(f"Rows: {len(rows)}" + (" (truncated)" if result.get("truncated") else ""))
            for r in rows:
                lines.append("  " + json.dumps(r, default=str))
        elif action == "integrity":
            integ = result.get("integrity") or []
            lines.append(f"integrity_check: {', '.join(integ) if integ else '(no rows)'}")
            lines.append(f"Tables: {result.get('table_count', 0)}")
        if not ok and result.get("error"):
            lines.append(f"Error: {result['error']}")
        return ToolResult(
            success=ok,
            output=_truncate("\n".join(lines)),
            error="" if ok else str(result.get("error") or "db_inspect failed"),
            metadata={"db_inspect": result},
        )


class BrowserVerifyTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "browser_verify"

    @property
    def description(self) -> str:
        return "Run a browser smoke check across desktop and mobile viewports when possible, with HTTP fallback."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"url": {"type": "string"}}, "additionalProperties": False}

    async def execute(self, *, url: str = "", **_kwargs) -> ToolResult:
        from augmentum.coder.browser import verify_preview

        result = await verify_preview(self._cm, self._workspace_id, url=url)
        ok = bool(result.get("ok"))
        lines = [
            f"Browser verification {'passed' if ok else 'failed'} for {result.get('url') or url or '(no url)'}.",
            f"Mode: {result.get('mode') or 'unknown'}",
        ]
        for check in result.get("checks") or []:
            lines.append(
                f"- {check.get('viewport')}: {'ok' if check.get('ok') else 'failed'} "
                f"status={check.get('status', '')} title={check.get('title', '')}"
            )
        # Aggregated runtime errors (union across viewports). Surfacing
        # these in the visible output is the whole point of the
        # 2026-05-26 listener wiring — without it the agent sees
        # "verification passed" even when 12 JS errors fired during load.
        ce = result.get("console_errors") or []
        nf = result.get("network_failures") or []
        if ce:
            lines.append(f"\nConsole ({len(ce)}):")
            for e in ce[:8]:
                viewport = f"{e.get('viewport', '')}: " if e.get("viewport") else ""
                lines.append(f"  {viewport}[{e.get('type', '?')}] {e.get('text', '')}")
            if len(ce) > 8:
                lines.append(f"  … +{len(ce) - 8} more")
        if nf:
            lines.append(f"\nNetwork failures ({len(nf)}):")
            for f in nf[:8]:
                viewport = f"{f.get('viewport', '')}: " if f.get("viewport") else ""
                bits = [str(f.get("status", f.get("failure", "?"))), f.get("method", ""), f.get("url", "")]
                lines.append(f"  {viewport}" + " ".join(b for b in bits if b))
            if len(nf) > 8:
                lines.append(f"  … +{len(nf) - 8} more")
        return ToolResult(
            success=ok,
            output="\n".join(lines),
            error="" if ok else str(result.get("error") or "browser verification failed"),
            metadata={"browser_verify": result},
        )


class ProfileReadTool(_RuntimeCoderTool):
    # Reads profile_store only — no container needed; stays available in the
    # editor/remote path where container_manager is None.
    requires_container = False

    @property
    def name(self) -> str:
        return "profile_read"

    @property
    def description(self) -> str:
        return "Read concise learned workspace profile facts for this workspace."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 12},
            },
            "additionalProperties": False,
        }

    async def execute(self, *, category: str = "", query: str = "", limit: int = 12, **_kwargs) -> ToolResult:
        if self._profile_store is None or not self._user_id:
            return ToolResult(success=True, output="No workspace profile store is available.", metadata={"entries": []})
        try:
            entries = await self._profile_store.query_for_workspace(
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                category=(category or None),
            )
            from augmentum.coder.profile import render_profile_block

            block = render_profile_block(entries, query=query, max_entries=limit)
            return ToolResult(
                success=True,
                output=block or "No profile facts are stored for this workspace yet.",
                metadata={"entries": [entry.to_dict() for entry in entries]},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to read workspace profile: {exc}")


class ProfileUpdateTool(_RuntimeCoderTool):
    # Writes profile_store only — no container needed.
    requires_container = False

    @property
    def name(self) -> str:
        return "profile_update"

    @property
    def description(self) -> str:
        return (
            "Propose or write high-confidence workspace profile facts. "
            "Use only for stable commands, framework, conventions, recurring "
            "failures, and explicit user preferences."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["upsert", "delete"]},
                            "category": {"type": "string"},
                            "key": {"type": "string"},
                            "value": {},
                            "confidence": {"type": "number", "default": 0.5},
                            "workspace_scoped": {"type": "boolean", "default": True},
                        },
                        "required": ["action", "category", "key"],
                    },
                },
            },
            "required": ["entries"],
            "additionalProperties": False,
        }

    async def execute(self, *, entries: list | None = None, **_kwargs) -> ToolResult:
        if self._profile_store is None or not self._user_id:
            return ToolResult(success=False, error="No workspace profile store is available.")
        if not isinstance(entries, list) or not entries:
            return ToolResult(success=False, error="profile_update requires entries array", validation_error=True)
        from augmentum.coder.profile import ACTIVE_PROFILE_CATEGORIES

        written: list[dict] = []
        proposed: list[dict] = []
        deleted: list[dict] = []
        for idx, raw in enumerate(entries):
            if not isinstance(raw, dict):
                return ToolResult(success=False, error=f"entries[{idx}] must be an object", validation_error=True)
            action = str(raw.get("action") or "upsert").strip().lower()
            category = str(raw.get("category") or "").strip().lower()
            key = str(raw.get("key") or "").strip()
            if not category or not key or category not in ACTIVE_PROFILE_CATEGORIES:
                return ToolResult(success=False, error=f"entries[{idx}] has invalid category/key", validation_error=True)
            workspace_id = self._workspace_id if raw.get("workspace_scoped", True) else ""
            if action == "delete":
                ok = await self._profile_store.delete(
                    user_id=self._user_id,
                    workspace_id=workspace_id,
                    category=category,
                    key=key,
                )
                deleted.append({"category": category, "key": key, "deleted": ok})
                continue
            if action != "upsert":
                return ToolResult(success=False, error=f"entries[{idx}].action must be upsert or delete", validation_error=True)
            confidence = float(raw.get("confidence") or 0.5)
            payload = {
                "category": category,
                "key": key,
                "value": raw.get("value"),
                "confidence": confidence,
                "workspace_id": workspace_id,
            }
            if confidence < 0.75:
                proposed.append(payload)
                continue
            entry = await self._profile_store.upsert(
                user_id=self._user_id,
                workspace_id=workspace_id,
                category=category,
                key=key,
                value=raw.get("value"),
                confidence=confidence,
            )
            written.append(entry.to_dict())
        lines = [
            f"Profile update complete: {len(written)} written, "
            f"{len(proposed)} proposed, {len(deleted)} deleted."
        ]
        if proposed:
            lines.append("Low-confidence entries were not written; treat them as proposals until confirmed.")
        return ToolResult(success=True, output="\n".join(lines), metadata={"written": written, "proposed": proposed, "deleted": deleted})


class ObserveTool(_RuntimeCoderTool):
    """Append a durable cross-session fact to the observation ledger.

    Distinct from ``profile_update`` (which writes structured per-user
    profile entries) and ``code_edit`` (which mutates project source):
    this writes one line to ``/workspace/.augmentum/observations.jsonl``
    capturing something the agent LEARNED about the workspace that's
    worth remembering next session.

    Examples the model should reach for this tool:
      - "pytest is the test runner; tests live in tests/" (build)
      - "auth tokens are read from /workspace/.env.local" (env)
      - "node 18 is locked; do not require node 20+ features" (constraint)
      - "the /v1/messages endpoint expects a list under 'messages' not
         a single string" (api)

    Examples the model should NOT use this for:
      - One-off intermediate findings ("file_read returned 200 lines")
        — those belong in prose, not durable memory
      - User-confidential information (we're not using this for secrets)
      - Things already in identity.toml's [detected] (auto-refreshed)
    """

    # Writes the observation ledger (state/store) only — no container needed.
    requires_container = False

    @property
    def name(self) -> str:
        return "observe"

    @property
    def description(self) -> str:
        return (
            "Record a durable fact about this workspace so future "
            "turns and future sessions don't have to re-discover it. "
            "Persists to /workspace/.augmentum/observations.jsonl. "
            "Categories: build, test, deploy, api, data, env, "
            "constraint, gotcha, style, other. Use confidence='tentative' "
            "for things you inferred but haven't verified; 'confirmed' "
            "once a tool result or test backed it up. Dedup-by-fact: "
            "recording the same fact twice in the same category "
            "updates the timestamp + confidence rather than appending "
            "a duplicate."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "build", "test", "deploy", "api", "data",
                        "env", "constraint", "gotcha", "style", "other",
                    ],
                },
                "fact": {
                    "type": "string",
                    "description": (
                        "One sentence stating the durable fact. Concrete, "
                        "verifiable, scoped to this workspace."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "enum": ["tentative", "confirmed"],
                    "default": "confirmed",
                    "description": (
                        "'confirmed' when a tool result or test backed "
                        "this up; 'tentative' when inferred from context."
                    ),
                },
            },
            "required": ["category", "fact"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        category: str = "",
        fact: str = "",
        confidence: str = "confirmed",
        **_kwargs,
    ) -> ToolResult:
        # Validate at the tool boundary so the kernel call gets clean
        # inputs. The kernel ALSO defensively normalizes, but a clear
        # validation error here teaches the model the schema.
        from augmentum.coder.observations import CATEGORIES, CONFIDENCES

        cat = (category or "").strip().lower()
        if cat not in CATEGORIES:
            return ToolResult(
                success=False,
                error=(
                    f"category must be one of {sorted(CATEGORIES)}; "
                    f"got {category!r}"
                ),
                validation_error=True,
            )
        conf = (confidence or "confirmed").strip().lower()
        if conf not in CONFIDENCES:
            return ToolResult(
                success=False,
                error=(
                    f"confidence must be one of {sorted(CONFIDENCES)}; "
                    f"got {confidence!r}"
                ),
                validation_error=True,
            )
        fact_text = (fact or "").strip()
        if not fact_text:
            return ToolResult(
                success=False,
                error="fact is required",
                validation_error=True,
            )
        if len(fact_text) > 500:
            return ToolResult(
                success=False,
                error=(
                    "fact must be ≤ 500 chars — observations are one-"
                    "sentence facts, not paragraphs. Split into multiple "
                    "observations if you have multiple things to record."
                ),
                validation_error=True,
            )

        # Source provenance: which turn produced this. Useful when
        # auditing the ledger later ("who recorded that pytest fact?").
        turn_count = int(getattr(self._state, "tool_calls_made", 0) or 0)
        source = f"observe tool, turn (tool_calls={turn_count})"

        # Construct the WorkspaceKernel on the fly — the tool doesn't
        # keep a reference because it's stateless across calls. Kernel
        # construction is cheap (no I/O at init).
        from augmentum.coder.workspace_kernel import WorkspaceKernel

        kernel = WorkspaceKernel(self._cm, self._workspace_id)
        ok = await kernel.record_observation(
            category=cat,
            fact=fact_text,
            source=source,
            confidence=conf,
        )
        if not ok:
            return ToolResult(
                success=False,
                error=(
                    "Failed to persist observation. The workspace "
                    "kernel directory may be unreachable; try again "
                    "or proceed without persisting."
                ),
            )
        return ToolResult(
            success=True,
            output=(
                f"Recorded: [{cat}] {fact_text}"
                + (" (tentative)" if conf == "tentative" else "")
            ),
            metadata={
                "observation": {
                    "category": cat,
                    "fact": fact_text,
                    "confidence": conf,
                    "source": source,
                },
            },
        )
