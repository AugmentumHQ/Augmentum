"""Cross-turn interactive PTY sessions for the coder agent.

The agent's shell tools are one-shot (``shell_exec``) or fire-and-forget
(``service_start``) — neither can drive a program that needs keystrokes or
paints the screen: TUIs, REPLs, curses installers, ``watch`` dashboards,
pagers, vim. This module gives the agent named PTY sessions that persist
across turns, built on the same docker-exec PTY infrastructure the browser
terminal uses (``ContainerManager.exec_shell``).

The load-bearing piece is *rendering*: raw PTY output is ANSI escape soup,
useless for judging a TUI layout. Each session pumps its output through a
headless VT100 emulator (``pyte``) and snapshots return the RENDERED screen
grid — what a human would see in the terminal — plus scrollback history.

Sessions live in a module-level manager (one server process), keyed by
``(workspace_id, session_id)``. They are best-effort ephemeral state: a
server restart drops them (the exec'd processes die with their attach
connections closing or linger until container stop — the pids limit bounds
the damage), and the tools report a missing session with a clear error so
the agent just re-opens.

Consumed by ``augmentum/coder/terminal_tools.py`` (term_open / term_send /
term_snapshot / term_list / term_close).
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from typing import Any

from augmentum.utils.logging import get_logger

try:  # pure-python, tiny; declared in pyproject — guard for stripped envs
    import pyte
except ImportError:  # pragma: no cover - exercised only when dep missing
    pyte = None  # type: ignore[assignment]

log = get_logger(__name__)

MAX_SESSIONS_PER_WORKSPACE = 4
DEFAULT_COLS = 100
DEFAULT_ROWS = 30
_SCROLLBACK_LINES = 2000
# A snapshot taken while output is mid-burst renders a half-drawn frame.
# ``settle()`` polls until the stream has been quiet this long (or the
# caller's wait budget runs out).
_QUIET_S = 0.25
_POLL_S = 0.05

# Named keys → PTY byte sequences (xterm-compatible; docker exec ttys get
# TERM=xterm). Modifier combos are handled in encode_keys, not here.
_NAMED_KEYS: dict[str, str] = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "escape": "\x1b",
    "esc": "\x1b",
    "backspace": "\x7f",
    "space": " ",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "page_up": "\x1b[5~",
    "page_down": "\x1b[6~",
    "insert": "\x1b[2~",
    "delete": "\x1b[3~",
    "f1": "\x1bOP",
    "f2": "\x1bOQ",
    "f3": "\x1bOR",
    "f4": "\x1bOS",
    "f5": "\x1b[15~",
    "f6": "\x1b[17~",
    "f7": "\x1b[18~",
    "f8": "\x1b[19~",
    "f9": "\x1b[20~",
    "f10": "\x1b[21~",
    "f11": "\x1b[23~",
    "f12": "\x1b[24~",
}


def valid_key_names() -> list[str]:
    """Key names accepted by :func:`encode_keys` (plus ctrl+X / alt+X combos)."""
    return sorted(_NAMED_KEYS) + ["ctrl+<letter>", "alt+<key>"]


def encode_keys(keys: list[str]) -> bytes:
    """Translate named keys into PTY byte sequences.

    Accepts the names in ``_NAMED_KEYS`` plus ``ctrl+<x>`` (control chars,
    e.g. ``ctrl+c`` → 0x03) and ``alt+<key>`` (ESC-prefixed). Raises
    ``ValueError`` naming the offending key so the tool layer can surface a
    validation error the model can correct.
    """
    out: list[str] = []
    for raw in keys:
        key = str(raw).strip().lower().replace("-", "+").replace(" ", "_")
        if key in _NAMED_KEYS:
            out.append(_NAMED_KEYS[key])
            continue
        if key.startswith("ctrl+") and len(key) == 6:
            ch = key[5]
            code = ord(ch.upper()) - 64
            if 1 <= code <= 31:
                out.append(chr(code))
                continue
        if key.startswith("alt+"):
            rest = key[4:]
            base = _NAMED_KEYS.get(rest, rest if len(rest) == 1 else "")
            if base:
                out.append("\x1b" + base)
                continue
        raise ValueError(
            f"Unknown key {raw!r}. Valid: {', '.join(valid_key_names())}"
        )
    return "".join(out).encode()


class TerminalSession:
    """One PTY session: docker exec + pyte screen + background pump."""

    def __init__(
        self,
        session_id: str,
        workspace_id: str,
        command: str,
        *,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        cwd: str = "/workspace",
    ) -> None:
        if pyte is None:
            raise RuntimeError(
                "Terminal sessions require the 'pyte' package in the "
                "Augmentum server environment (pip install pyte)."
            )
        self.id = session_id
        self.workspace_id = workspace_id
        self.command = command
        self.cols = int(cols)
        self.rows = int(rows)
        self.cwd = cwd
        self.opened_at = time.time()
        self.exited = False
        self.last_output_at: float = 0.0
        self.bytes_seen = 0
        self._screen = pyte.HistoryScreen(
            self.cols, self.rows, history=_SCROLLBACK_LINES
        )
        self._vt = pyte.ByteStream(self._screen)
        self._exec_id: str = ""
        self._stream: Any = None
        self._pump_task: asyncio.Task | None = None

    async def start(self, container_manager) -> None:
        """Create the docker exec, attach, and start pumping output."""
        exec_obj = await container_manager.exec_shell(
            self.workspace_id, command=self.command, cwd=self.cwd,
        )
        self._exec_id = getattr(exec_obj, "id", "") or ""
        self._stream = exec_obj.start(detach=False)
        self._pump_task = asyncio.create_task(self._pump())
        # Resize AFTER the pump attaches — Docker rejects a resize on an
        # exec that hasn't started, and the exec only starts when the
        # stream's first read/write triggers the attach. The pump's first
        # read_out fires within a tick; a short grace covers scheduling.
        if self._exec_id:
            await asyncio.sleep(0.2)
            try:
                await container_manager.resize_exec(
                    self._exec_id, self.rows, self.cols
                )
            except Exception:
                log.warning(
                    "terminal_session_resize_failed",
                    session=self.id, workspace_id=self.workspace_id,
                )

    async def _pump(self) -> None:
        try:
            while True:
                msg = await self._stream.read_out()
                if msg is None:
                    break
                data = getattr(msg, "data", msg)
                if isinstance(data, str):
                    data = data.encode()
                self._vt.feed(data)
                self.bytes_seen += len(data)
                self.last_output_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "terminal_session_pump_error",
                session=self.id, workspace_id=self.workspace_id,
                error=str(exc),
            )
        finally:
            self.exited = True
            self.last_output_at = time.monotonic()

    async def send(self, data: bytes) -> None:
        if self.exited or self._stream is None:
            raise RuntimeError(
                f"terminal session '{self.id}' has exited — re-open with term_open"
            )
        await self._stream.write_in(data)

    async def settle(
        self, wait_ms: int, *, baseline_bytes: int | None = None,
    ) -> None:
        """Wait until output goes quiet (or the budget elapses).

        Snapshotting mid-burst renders a half-drawn frame; polling for a
        quiet gap gives programs time to finish painting without paying
        the full budget when they're already idle.

        ``baseline_bytes`` (a prior ``bytes_seen`` reading) makes the wait
        mean "quiet AFTER the program reacted": the stream having been
        quiet since a PREVIOUS burst doesn't count, so a snapshot taken
        right after term_send can't race ahead of the echo. If the program
        never reacts, the full budget is spent — bounded, and the snapshot
        honestly shows an unchanged screen.
        """
        deadline = time.monotonic() + max(0, int(wait_ms)) / 1000.0
        while time.monotonic() < deadline and not self.exited:
            if baseline_bytes is not None and self.bytes_seen <= baseline_bytes:
                await asyncio.sleep(_POLL_S)
                continue
            if (
                self.last_output_at
                and time.monotonic() - self.last_output_at >= _QUIET_S
            ):
                return
            await asyncio.sleep(_POLL_S)

    def snapshot(self, *, history_lines: int = 0) -> str:
        """Render the current screen as plain text (what a user would see)."""
        lines = [line.rstrip() for line in self._screen.display]
        trimmed = 0
        while len(lines) > 1 and not lines[-1]:
            lines.pop()
            trimmed += 1
        cursor = self._screen.cursor
        status = "exited" if self.exited else "running"
        header = (
            f"[{self.id}] {status} — {self.command} "
            f"({self.cols}x{self.rows}, cursor row {cursor.y + 1} "
            f"col {cursor.x + 1}"
            + (f", {trimmed} blank rows trimmed" if trimmed else "")
            + ")"
        )
        ruler = "─" * min(self.cols, 100)
        parts = [header]
        if history_lines > 0:
            hist = self.history_tail(history_lines)
            if hist:
                parts.append(f"── scrollback (last {len(hist)} lines) ──")
                parts.extend(hist)
        parts.append(ruler)
        parts.extend(lines)
        parts.append(ruler)
        return "\n".join(parts)

    def history_tail(self, n: int) -> list[str]:
        """Last ``n`` scrolled-off lines above the visible screen."""
        try:
            rows = list(self._screen.history.top)[-max(0, int(n)):]
            return [
                "".join(row[x].data for x in range(self.cols)).rstrip()
                for row in rows
            ]
        except Exception:
            log.warning(
                "terminal_session_history_render_failed",
                session=self.id, exc_info=True,
            )
            return []

    async def close(self) -> None:
        """Best-effort graceful kill, then detach.

        Docker keeps an exec'd process running after its attach connection
        drops, so just closing the stream would leak the process until the
        container stops. SIGINT the foreground job, EOF the shell, then
        cancel the pump and close the stream.
        """
        if self._stream is not None and not self.exited:
            try:
                await self._stream.write_in(b"\x03")
                await asyncio.sleep(0.1)
                await self._stream.write_in(b"\x04")
            except Exception:
                log.debug("terminal_session_close_signal_failed", session=self.id)
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
        if self._stream is not None:
            try:
                res = self._stream.close()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                log.debug("terminal_session_stream_close_failed", session=self.id)
        self.exited = True

    def describe(self) -> dict:
        return {
            "session_id": self.id,
            "workspace_id": self.workspace_id,
            "command": self.command,
            "status": "exited" if self.exited else "running",
            "cols": self.cols,
            "rows": self.rows,
            "opened_at": self.opened_at,
            "bytes_seen": self.bytes_seen,
        }


class TerminalSessionManager:
    """Registry of live sessions, keyed by (workspace_id, session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], TerminalSession] = {}
        self._counter = 0

    def get(self, workspace_id: str, session_id: str) -> TerminalSession | None:
        return self._sessions.get((workspace_id, session_id))

    def list(self, workspace_id: str) -> list[TerminalSession]:
        return [
            s for (ws, _), s in self._sessions.items() if ws == workspace_id
        ]

    async def open(
        self,
        container_manager,
        workspace_id: str,
        command: str,
        *,
        name: str = "",
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        cwd: str = "/workspace",
    ) -> TerminalSession:
        # An exited session under the requested name is dead weight —
        # replace it. A RUNNING one is a real conflict the model must
        # resolve (close it or pick a new name).
        if name:
            existing = self.get(workspace_id, name)
            if existing is not None:
                if not existing.exited:
                    raise ValueError(
                        f"terminal session '{name}' is already running — "
                        "term_close it or choose another name"
                    )
                await self.close(workspace_id, name)
        running = [s for s in self.list(workspace_id) if not s.exited]
        if len(running) >= MAX_SESSIONS_PER_WORKSPACE:
            ids = ", ".join(s.id for s in running)
            raise ValueError(
                f"Workspace already has {len(running)} running terminal "
                f"sessions ({ids}) — term_close one first."
            )
        if not name:
            self._counter += 1
            name = f"term{self._counter}"
        session = TerminalSession(
            name, workspace_id, command, cols=cols, rows=rows, cwd=cwd,
        )
        await session.start(container_manager)
        self._sessions[(workspace_id, name)] = session
        log.info(
            "terminal_session_opened",
            session=name, workspace_id=workspace_id, command=command,
        )
        return session

    async def close(self, workspace_id: str, session_id: str) -> bool:
        session = self._sessions.pop((workspace_id, session_id), None)
        if session is None:
            return False
        await session.close()
        log.info(
            "terminal_session_closed",
            session=session_id, workspace_id=workspace_id,
        )
        return True


_MANAGER: TerminalSessionManager | None = None


def get_terminal_manager() -> TerminalSessionManager:
    """Process-wide session registry (sessions must outlive a single turn)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TerminalSessionManager()
    return _MANAGER
