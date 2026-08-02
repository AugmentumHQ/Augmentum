"""Claude Code driver — runs the Claude Agent SDK headless and maps its
streamed messages onto Augmentum's normalized CoderEvent model.

Auth: the Agent SDK uses the host's own Claude credential (a Pro/Max
subscription via the logged-in Claude CLI, or an API key). It is an OPTIONAL
dependency — ``pip install claude-agent-sdk``; if it's absent or unauthenticated
the driver reports ``is_available() == False`` and Augmentum falls back to the
native coder. We never bundle it as a hard dep (keeps the local-first install
clean and the proprietary path strictly opt-in).

This module imports cleanly WITHOUT the SDK installed — the SDK is imported
lazily inside ``run``/``is_available``. ``_translate`` (the SDK-message →
CoderEvent mapper) is pure and duck-typed so it's unit-testable with fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from augmentum.coder.external.base import (
    CoderEvent,
    ExternalCoderDriver,
    ExternalTask,
    tool_use_event,
)
from augmentum.coder.external.claude_auth import auth_env
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# task.permission → Claude Agent SDK permission_mode. "confirm_mutations" maps
# to the SDK's default mode, where a ``can_use_tool`` callback (wired to
# Augmentum's consent gate in the live slice) decides each mutating tool;
# "auto" auto-accepts edits (still container-sandboxed); "plan" forbids
# execution entirely.
_PERMISSION_MAP = {
    "confirm_mutations": "default",
    "auto": "acceptEdits",
    "plan": "plan",
}


def _as_dict(obj) -> dict:
    """Best-effort dict view of an SDK block/message (handles dataclass-ish and
    dict-ish shapes)."""
    if isinstance(obj, dict):
        return obj
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def _block_to_event(block) -> CoderEvent | None:
    """Map one content block of an AssistantMessage to a CoderEvent."""
    name = type(block).__name__
    d = _as_dict(block)
    # Text prose
    if name == "TextBlock" or "text" in d and "name" not in d and "thinking" not in d:
        text = (d.get("text") or "").strip()
        return CoderEvent(kind="message", text=text, raw=d) if text else None
    # Reasoning
    if name == "ThinkingBlock" or "thinking" in d:
        return CoderEvent(kind="thinking", text=(d.get("thinking") or "")[:200], raw=d)
    # Tool use
    if name == "ToolUseBlock" or ("name" in d and "input" in d):
        return tool_use_event(d.get("name") or "", d.get("input") or {}, raw=d)
    return None


def _translate(msg) -> list[CoderEvent]:
    """Map one streamed SDK message to zero or more CoderEvents. Duck-typed so a
    test can pass simple namespaces mimicking the SDK shapes."""
    name = type(msg).__name__
    d = _as_dict(msg)

    # Terminal result
    if name == "ResultMessage" or "is_error" in d or d.get("type") == "result":
        if d.get("is_error"):
            return [CoderEvent(kind="failed", text=str(d.get("result") or "error"), raw=d)]
        return [CoderEvent(
            kind="completed",
            text=str(d.get("result") or "")[:200],
            raw=d,
        )]

    # Init / system
    if name == "SystemMessage" or d.get("subtype") == "init":
        return [CoderEvent(kind="started", raw=d)]

    # Assistant turn — iterate content blocks
    content = d.get("content")
    if isinstance(content, list):
        out: list[CoderEvent] = []
        for block in content:
            ev = _block_to_event(block)
            if ev is not None:
                out.append(ev)
        return out

    return []


class ClaudeCodeDriver(ExternalCoderDriver):
    id = "claude_code"
    label = "Claude Code"

    def __init__(
        self,
        *,
        cwd: str = "/workspace",
        oauth_token: str = "",
        api_key: str = "",
    ) -> None:
        self._cwd = cwd
        # Subscription OAuth token (sk-ant-oat01-…) from the browser login flow
        # (claude_auth.SETUP_TOKEN_CMD) — preferred for personal/subscription
        # use. ``api_key`` is the per-token-billed fallback.
        self._oauth_token = oauth_token
        self._api_key = api_key
        self._client = None  # set during run() for interrupt()

    def _has_credential(self) -> bool:
        """A usable Claude credential is present (token, key, or logged-in CLI).
        Factored out so it's testable without the SDK installed."""
        if self._oauth_token or self._api_key:
            return True
        return _claude_cli_logged_in()

    async def is_available(self) -> bool:
        try:
            import claude_agent_sdk  # noqa: F401
        except Exception:
            return False
        return self._has_credential()

    async def run(self, task: ExternalTask) -> AsyncIterator[CoderEvent]:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except Exception as exc:  # pragma: no cover — gated by is_available
            yield CoderEvent(kind="failed", text=f"claude-agent-sdk unavailable: {exc!r}")
            return

        # Make the subscription OAuth token (or API key) visible to the SDK.
        # Set in-process env so the bundled Claude Code binary inherits it; a
        # logged-in CLI credential is used when neither is provided.
        import os
        for k, v in auth_env(oauth_token=self._oauth_token, api_key=self._api_key).items():
            os.environ[k] = v

        opts_kwargs = {
            "cwd": task.workspace or self._cwd,
            "permission_mode": _PERMISSION_MAP.get(task.permission, "default"),
        }
        if task.allowed_tools:
            opts_kwargs["allowed_tools"] = list(task.allowed_tools)
        if task.mcp_servers:
            opts_kwargs["mcp_servers"] = task.mcp_servers
        if task.model:
            opts_kwargs["model"] = task.model

        try:
            options = ClaudeAgentOptions(**opts_kwargs)
        except TypeError:
            # SDK version drift on option names — fall back to the minimal set.
            options = ClaudeAgentOptions(cwd=task.workspace or self._cwd)

        yield CoderEvent(kind="started", text=task.prompt[:120])
        try:
            agen = query(prompt=task.prompt, options=options)
            self._client = agen
            async for msg in agen:
                for ev in _translate(msg):
                    # We already emit one explicit "started"; the SDK's init
                    # SystemMessage also maps to "started" — drop the duplicate.
                    if ev.kind == "started":
                        continue
                    yield ev
        except Exception as exc:  # noqa: BLE001 — surface as a normalized failure
            log.warning("claude_code_run_failed", error=repr(exc))
            yield CoderEvent(kind="failed", text=repr(exc))
        finally:
            self._client = None

    async def interrupt(self) -> None:
        client = self._client
        if client is not None:
            for meth in ("interrupt", "stopTask", "stop_task", "aclose", "close"):
                fn = getattr(client, meth, None)
                if fn is None:
                    continue
                try:
                    res = fn()
                    if hasattr(res, "__await__"):
                        await res
                    return
                except Exception:  # noqa: BLE001
                    continue


def _claude_cli_logged_in() -> bool:
    """Heuristic: is there a logged-in Claude CLI credential on this host?
    Checks the standard credential locations without invoking the CLI."""
    import os
    from pathlib import Path

    candidates = [
        Path.home() / ".claude" / ".credentials.json",
        Path.home() / ".config" / "claude" / ".credentials.json",
    ]
    if any(p.exists() for p in candidates):
        return True
    # macOS stores it in Keychain; env override covers headless/container.
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
