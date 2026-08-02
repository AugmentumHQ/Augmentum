"""Sidecar-native browser tools (agent-browser service, compose.browser.yaml).

These expose the wider verb surface of the persistent sidecar browser —
interaction verbs, navigation, element introspection, tabs, console,
semantic locators. They REQUIRE the sidecar (no in-workspace Playwright
fallback: they exist precisely because the throwaway-browser path couldn't
hold page state between calls). Selectors accept CSS or snapshot refs
(@e1 from browser_snapshot) interchangeably.

Everything dispatches through ``browser_sidecar.command`` which enforces
the subcommand allow-list — the auth vault, plugins, chat, dashboard, and
install surfaces are NOT reachable from the model.
"""

from __future__ import annotations

import json

from augmentum.coder.runtime_tools import _RuntimeCoderTool
from augmentum.tools.base import ToolCategory, ToolResult


class _SidecarBrowserTool(_RuntimeCoderTool):
    """Shared plumbing: availability check + command dispatch + rendering."""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    async def _run(self, args: list, *, url: str = "", timeout: float = 30.0):
        from augmentum.coder import browser_sidecar as bs

        if not await bs.is_available(self._cm._docker):
            return None
        return await bs.command(self._cm, self._workspace_id, args, url=url, timeout=timeout)

    @staticmethod
    def _unavailable() -> ToolResult:
        return ToolResult(
            success=False,
            error=(
                "The browser sidecar service is not running. This tool needs "
                "the persistent browser (compose.browser.yaml overlay). Use "
                "browser_open/browser_snapshot/browser_evaluate for the "
                "fallback browser path."
            ),
        )

    @staticmethod
    def _render(result: dict, verb: str) -> ToolResult:
        ok = bool(result.get("ok"))
        payload = {
            k: v for k, v in result.items()
            if k not in ("ok", "engine", "playwright", "error", "latency_ms")
            and v not in ("", [], {}, None)
        }
        body = json.dumps(payload, indent=1, sort_keys=True, default=str)[:8000]
        return ToolResult(
            success=ok,
            output=f"{verb} {'succeeded' if ok else 'failed'}.\n{body}",
            error="" if ok else str(result.get("error") or f"{verb} failed"),
            metadata={"browser": result},
        )


class BrowserInteractTool(_SidecarBrowserTool):
    _SIMPLE_VERBS = ("hover", "dblclick", "focus", "check", "uncheck",
                     "scrollintoview", "highlight")

    @property
    def name(self) -> str:
        return "browser_interact"

    @property
    def description(self) -> str:
        return (
            "Interact with the open page in the persistent browser: hover, "
            "dblclick, focus, check/uncheck, select, press (keyboard), "
            "scroll, scrollintoview, drag, highlight. Selector accepts CSS "
            "or a snapshot ref (@e1 from browser_snapshot). Page state "
            "persists across calls."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["hover", "dblclick", "focus", "check", "uncheck",
                             "select", "press", "scroll", "scrollintoview",
                             "drag", "highlight"],
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector or @ref (target element; drag source)",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "press: the key (Enter, Tab, Control+a). select: the "
                        "option value. scroll: direction up/down/left/right. "
                        "drag: the DESTINATION selector."
                    ),
                },
                "amount": {"type": "integer", "description": "scroll distance in px (optional)"},
                "url": {"type": "string", "description": "Optional: open this URL first"},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, *, action: str = "", selector: str = "",
                      value: str = "", amount: int = 0, url: str = "",
                      **_kwargs) -> ToolResult:
        action = (action or "").strip().lower()
        selector = (selector or "").strip()
        value = (value or "").strip()
        if action in self._SIMPLE_VERBS:
            if not selector:
                return ToolResult(success=False, error=f"{action} requires selector", validation_error=True)
            args = [action, selector]
        elif action == "select":
            if not selector or not value:
                return ToolResult(success=False, error="select requires selector and value", validation_error=True)
            args = ["select", selector, value]
        elif action == "press":
            if not value:
                return ToolResult(success=False, error="press requires value (the key, e.g. Enter)", validation_error=True)
            args = ["press", value]
        elif action == "scroll":
            direction = value or "down"
            if direction not in ("up", "down", "left", "right"):
                return ToolResult(success=False, error="scroll value must be up/down/left/right", validation_error=True)
            args = ["scroll", direction] + ([str(int(amount))] if amount else [])
        elif action == "drag":
            if not selector or not value:
                return ToolResult(
                    success=False,
                    error="drag requires selector (source) and value (destination selector)",
                    validation_error=True,
                )
            args = ["drag", selector, value]
        else:
            return ToolResult(success=False, error=f"unknown action {action!r}", validation_error=True)
        result = await self._run(args, url=url)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_interact:{action}")


class BrowserNavigateTool(_SidecarBrowserTool):
    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return (
            "History navigation in the persistent browser: back, forward, "
            "reload, or pushstate (SPA client-side navigation to a path)."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["back", "forward", "reload", "pushstate"]},
                "url": {"type": "string", "description": "pushstate target path/URL"},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, *, action: str = "", url: str = "", **_kwargs) -> ToolResult:
        action = (action or "").strip().lower()
        if action not in ("back", "forward", "reload", "pushstate"):
            return ToolResult(success=False, error=f"unknown action {action!r}", validation_error=True)
        if action == "pushstate":
            if not url:
                return ToolResult(success=False, error="pushstate requires url", validation_error=True)
            args = ["pushstate", url]
        else:
            args = [action]
        result = await self._run(args)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_navigate:{action}")


class BrowserGetTool(_SidecarBrowserTool):
    _WHATS = ("text", "html", "value", "attr", "title", "url", "count", "box", "styles")

    @property
    def name(self) -> str:
        return "browser_get"

    @property
    def description(self) -> str:
        return (
            "Read element/page info from the persistent browser: text, html, "
            "value, attr, title, url, count, box (bounding box), styles. "
            "Selector accepts CSS or @ref."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "what": {"type": "string",
                         "enum": ["text", "html", "value", "attr", "title",
                                  "url", "count", "box", "styles"]},
                "selector": {"type": "string"},
                "attribute": {"type": "string", "description": "for what=attr"},
                "url": {"type": "string", "description": "Optional: open this URL first"},
            },
            "required": ["what"],
            "additionalProperties": False,
        }

    async def execute(self, *, what: str = "", selector: str = "",
                      attribute: str = "", url: str = "", **_kwargs) -> ToolResult:
        what = (what or "").strip().lower()
        if what not in self._WHATS:
            return ToolResult(success=False, error=f"unknown what {what!r}", validation_error=True)
        args = ["get", what]
        if what == "attr":
            if not selector or not attribute:
                return ToolResult(success=False, error="what=attr requires selector and attribute", validation_error=True)
            args += [selector, attribute]
        elif what not in ("title", "url"):
            if not selector:
                return ToolResult(success=False, error=f"what={what} requires selector", validation_error=True)
            args.append(selector)
        result = await self._run(args, url=url)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_get:{what}")


class BrowserConsoleTool(_SidecarBrowserTool):
    @property
    def name(self) -> str:
        return "browser_console"

    @property
    def description(self) -> str:
        return (
            "Read the persistent browser's console log or uncaught page "
            "errors (accumulated across interactions, not just page load). "
            "Optionally clear the buffer."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["console", "errors"], "default": "console"},
                "clear": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        }

    async def execute(self, *, source: str = "console", clear: bool = False, **_kwargs) -> ToolResult:
        source = source if source in ("console", "errors") else "console"
        args = [source] + (["--clear"] if clear else [])
        result = await self._run(args)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_console:{source}")


class BrowserTabsTool(_SidecarBrowserTool):
    @property
    def name(self) -> str:
        return "browser_tabs"

    @property
    def description(self) -> str:
        return (
            "Manage tabs in the persistent browser: list, new (with URL), "
            "switch (by tab id like t2), close."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "new", "switch", "close"], "default": "list"},
                "tab": {"type": "string", "description": "tab id (t1, t2...) for switch/close"},
                "url": {"type": "string", "description": "URL for action=new"},
            },
            "additionalProperties": False,
        }

    async def execute(self, *, action: str = "list", tab: str = "", url: str = "", **_kwargs) -> ToolResult:
        action = (action or "list").strip().lower()
        if action == "list":
            args = ["tab", "list"]
        elif action == "new":
            args = ["tab", "new"] + ([url] if url else [])
        elif action == "switch":
            if not tab:
                return ToolResult(success=False, error="switch requires tab id", validation_error=True)
            args = ["tab", tab]
        elif action == "close":
            args = ["tab", "close"] + ([tab] if tab else [])
        else:
            return ToolResult(success=False, error=f"unknown action {action!r}", validation_error=True)
        result = await self._run(args)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_tabs:{action}")


class BrowserFindTool(_SidecarBrowserTool):
    _LOCATORS = ("role", "text", "label", "placeholder", "alt", "title", "testid")
    _ACTIONS = ("click", "fill", "type", "hover", "check", "uncheck")

    @property
    def name(self) -> str:
        return "browser_find"

    @property
    def description(self) -> str:
        return (
            "Semantic element lookup in the persistent browser (accessibility-"
            "based, more robust than CSS): find by role/text/label/placeholder/"
            "alt/title/testid, optionally performing click/fill/type/hover/"
            "check/uncheck on the match."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "locator": {"type": "string",
                            "enum": ["role", "text", "label", "placeholder",
                                     "alt", "title", "testid"]},
                "value": {"type": "string", "description": "what to find (e.g. the role name or visible text)"},
                "action": {"type": "string",
                           "enum": ["click", "fill", "type", "hover", "check", "uncheck"],
                           "description": "optional action on the match"},
                "text": {"type": "string", "description": "text for action=fill/type"},
                "url": {"type": "string", "description": "Optional: open this URL first"},
            },
            "required": ["locator", "value"],
            "additionalProperties": False,
        }

    async def execute(self, *, locator: str = "", value: str = "",
                      action: str = "", text: str = "", url: str = "",
                      **_kwargs) -> ToolResult:
        locator = (locator or "").strip().lower()
        if locator not in self._LOCATORS:
            return ToolResult(success=False, error=f"unknown locator {locator!r}", validation_error=True)
        if not value:
            return ToolResult(success=False, error="browser_find requires value", validation_error=True)
        args = ["find", locator, value]
        if action:
            action = action.strip().lower()
            if action not in self._ACTIONS:
                return ToolResult(success=False, error=f"unknown action {action!r}", validation_error=True)
            args.append(action)
            if action in ("fill", "type"):
                if not text:
                    return ToolResult(success=False, error=f"action={action} requires text", validation_error=True)
                args.append(text)
        result = await self._run(args, url=url)
        if result is None:
            return self._unavailable()
        return self._render(result, f"browser_find:{locator}")


SIDECAR_BROWSER_TOOLS = (
    BrowserInteractTool,
    BrowserNavigateTool,
    BrowserGetTool,
    BrowserConsoleTool,
    BrowserTabsTool,
    BrowserFindTool,
)
