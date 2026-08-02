"""Interactive terminal session tools for Coder.

Agent-facing wrappers over ``terminal_sessions.py`` — persistent PTY
sessions with rendered-screen snapshots. These close the "working blind on
anything terminal-interactive" gap: TUIs, REPLs, curses installers,
``watch`` dashboards, pagers. Every state-changing call returns the
rendered screen so one call = act + observe.
"""
from __future__ import annotations

from augmentum.coder.runtime_tools import _RuntimeCoderTool, _truncate
from augmentum.coder.terminal_sessions import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    encode_keys,
    get_terminal_manager,
)
from augmentum.tools.base import ToolCategory, ToolResult


def _clamp(value, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


class TermOpenTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "term_open"

    @property
    def description(self) -> str:
        return (
            "Start a command in an interactive terminal (PTY) session that "
            "persists across turns. Returns the RENDERED screen — what a "
            "user would see, not raw escape codes. Use for TUIs, REPLs, "
            "curses installers, watch dashboards, or any program that needs "
            "keystrokes; use shell_exec for one-shot commands and "
            "service_start for headless daemons. Drive it with term_send, "
            "inspect with term_snapshot, and term_close when done."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run on the PTY (via bash -lc).",
                },
                "name": {
                    "type": "string",
                    "description": "Optional session name (default term1, term2, …).",
                },
                "cols": {"type": "integer", "default": DEFAULT_COLS,
                         "minimum": 20, "maximum": 240},
                "rows": {"type": "integer", "default": DEFAULT_ROWS,
                         "minimum": 5, "maximum": 80},
                "cwd": {"type": "string", "default": "/workspace"},
                "wait_ms": {
                    "type": "integer", "default": 1200,
                    "minimum": 0, "maximum": 15000,
                    "description": "How long to let the program draw before "
                                   "the snapshot (returns early once output "
                                   "goes quiet).",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(self, *, command: str = "", name: str = "",
                      cols=None, rows=None, cwd: str = "/workspace",
                      wait_ms=None, **_kwargs) -> ToolResult:
        if not (command or "").strip():
            return ToolResult(
                success=False, validation_error=True,
                error="term_open requires a 'command' to run.",
            )
        try:
            session = await get_terminal_manager().open(
                self._cm,
                self._workspace_id,
                command,
                name=(name or "").strip(),
                cols=_clamp(cols, 20, 240, DEFAULT_COLS),
                rows=_clamp(rows, 5, 80, DEFAULT_ROWS),
                cwd=cwd or "/workspace",
            )
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except RuntimeError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(
                success=False, error=f"Failed to open terminal session: {exc}",
            )
        await session.settle(_clamp(wait_ms, 0, 15000, 1200), baseline_bytes=0)
        return ToolResult(
            success=True,
            output=_truncate(session.snapshot()),
            metadata={"session": session.describe()},
        )


class TermSendTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "term_send"

    @property
    def description(self) -> str:
        return (
            "Send keystrokes to a terminal session opened with term_open: "
            "literal text and/or named keys (enter, tab, escape, up/down/"
            "left/right, backspace, page_up/page_down, home/end, f1-f12, "
            "ctrl+<letter> like ctrl+c, alt+<key>). Text is sent first, "
            "then keys in order — e.g. text='hello' keys=['enter'] types "
            "hello and submits it. Returns the rendered screen after the "
            "program reacts."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "text": {
                    "type": "string",
                    "description": "Literal text to type (no implicit enter).",
                },
                "keys": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Named keys sent after text, in order.",
                },
                "wait_ms": {
                    "type": "integer", "default": 800,
                    "minimum": 0, "maximum": 15000,
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, session_id: str = "", text: str = "",
                      keys: list | None = None, wait_ms=None,
                      **_kwargs) -> ToolResult:
        session = get_terminal_manager().get(self._workspace_id, session_id)
        if session is None:
            return ToolResult(
                success=False, validation_error=True,
                error=f"No terminal session '{session_id}' — see term_list, "
                      "or open one with term_open.",
            )
        if not text and not keys:
            return ToolResult(
                success=False, validation_error=True,
                error="term_send needs 'text' and/or 'keys'.",
            )
        try:
            payload = (text or "").encode() + encode_keys(list(keys or []))
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        baseline = session.bytes_seen
        try:
            await session.send(payload)
        except RuntimeError as exc:
            # Session died between turns — final screen is still useful.
            return ToolResult(
                success=False,
                error=str(exc),
                metadata={"final_screen": session.snapshot()},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to send input: {exc}")
        await session.settle(
            _clamp(wait_ms, 0, 15000, 800), baseline_bytes=baseline,
        )
        return ToolResult(
            success=True,
            output=_truncate(session.snapshot()),
            metadata={"session": session.describe()},
        )


class TermSnapshotTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "term_snapshot"

    @property
    def description(self) -> str:
        return (
            "Capture the rendered screen of a terminal session (plus "
            "optional scrollback history). Read-only — use to re-check a "
            "TUI/REPL between actions or after waiting for slow output."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "wait_ms": {
                    "type": "integer", "default": 0,
                    "minimum": 0, "maximum": 30000,
                    "description": "Optionally wait for output to settle first.",
                },
                "history_lines": {
                    "type": "integer", "default": 0,
                    "minimum": 0, "maximum": 2000,
                    "description": "Scrolled-off lines to include above the screen.",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, session_id: str = "", wait_ms=None,
                      history_lines=None, **_kwargs) -> ToolResult:
        session = get_terminal_manager().get(self._workspace_id, session_id)
        if session is None:
            return ToolResult(
                success=False, validation_error=True,
                error=f"No terminal session '{session_id}' — see term_list.",
            )
        budget = _clamp(wait_ms, 0, 30000, 0)
        if budget:
            await session.settle(budget)
        return ToolResult(
            success=True,
            output=_truncate(
                session.snapshot(history_lines=_clamp(history_lines, 0, 2000, 0))
            ),
            metadata={"session": session.describe()},
        )


class TermListTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "term_list"

    @property
    def description(self) -> str:
        return "List this workspace's terminal sessions (id, status, command)."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **_kwargs) -> ToolResult:
        sessions = get_terminal_manager().list(self._workspace_id)
        if not sessions:
            return ToolResult(
                success=True,
                output="No terminal sessions are open.",
                metadata={"sessions": []},
            )
        lines = ["Terminal sessions:"]
        for s in sessions:
            d = s.describe()
            lines.append(
                f"- {d['session_id']}: {d['status']}, {d['cols']}x{d['rows']}, "
                f"command: {d['command']}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"sessions": [s.describe() for s in sessions]},
        )


class TermCloseTool(_RuntimeCoderTool):
    @property
    def name(self) -> str:
        return "term_close"

    @property
    def description(self) -> str:
        return (
            "Close a terminal session: interrupts the foreground program "
            "(SIGINT + EOF), detaches, and frees the session slot. Close "
            "sessions you're done with — a workspace holds at most a few."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        }

    async def execute(self, *, session_id: str = "", **_kwargs) -> ToolResult:
        try:
            closed = await get_terminal_manager().close(
                self._workspace_id, session_id
            )
        except Exception as exc:
            return ToolResult(
                success=False, error=f"Failed to close session: {exc}",
            )
        if not closed:
            return ToolResult(
                success=False, validation_error=True,
                error=f"No terminal session '{session_id}' — see term_list.",
            )
        return ToolResult(
            success=True, output=f"Closed terminal session '{session_id}'.",
        )
