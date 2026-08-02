"""Shared consumer for Claude Code's ``--output-format stream-json`` protocol.

Two places drive a Claude run and need the exact same thing from its JSONL byte
stream: the external-coder run route (``_execute_run``) and the self-edit edit
driver. Rather than maintain two consumers that drift, the line→event→outcome
logic lives here once:

* normalize each line via ``parse_cli_event`` (skipping the per-subagent
  ``started`` spam), collecting file changes / ok / error;
* capture run metadata (session id, cost, turns, duration) from the ``result``;
* keep the verbatim raw lines for the full-fidelity transcript;
* hand each normalized event to an optional ``emit`` callback (the caller decides
  whether to persist it, publish it to a live bus, or both).

The collector holds NO I/O of its own — ``emit`` and ``on_session_id`` are
injected — so it's a pure stream→state machine, testable by feeding it bytes.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Awaitable, Callable

from augmentum.coder.external.claude_cli import parse_cli_event

EmitFn = Callable[[dict], Awaitable[None]]
SessionFn = Callable[[str], Awaitable[None]]


def summary_from_raw(raw_lines: list[str]) -> str:
    """Claude's own completion text from the terminal ``result`` event (else '')."""
    for line in reversed(raw_lines):
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if obj.get("type") == "result" and not obj.get("is_error"):
            return str(obj.get("result") or "")[:2000]
    return ""


class ClaudeStreamCollector:
    """Accumulates a Claude stream-json byte stream into run state + events."""

    def __init__(self, *, emit: EmitFn | None = None,
                 on_session_id: SessionFn | None = None) -> None:
        self._emit = emit
        self._on_session_id = on_session_id
        self.files: list[str] = []
        self.ok = False
        self.err = ""
        self.meta = {"session_id": "", "cost_usd": 0.0, "num_turns": 0,
                     "duration_ms": 0, "model": ""}
        self.raw_lines: list[str] = []
        self._buf = ""

    async def on_chunk(self, chunk: bytes) -> None:
        """Feed a stdout chunk; parses + emits any newly-completed lines."""
        self._buf += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.raw_lines.append(line)  # verbatim, for full-fidelity raw_jsonl
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                await self._do_emit({"kind": "message", "text": line[:200]})
                continue
            await self._capture_meta(obj)
            for ev in parse_cli_event(obj):
                # One synthetic "started" is enough; the parser yields one per
                # system/init (every subagent spawn → "▶ started" spam).
                if ev.kind == "started":
                    continue
                if ev.kind == "file_change" and ev.path:
                    self.files.append(ev.path)
                if ev.kind == "completed":
                    self.ok = True
                if ev.kind == "failed":
                    self.err = ev.text
                await self._do_emit({"kind": ev.kind, "text": ev.text, "tool": ev.tool, "path": ev.path})

    async def flush(self, reason: str = "") -> None:
        """Emit any trailing partial line and settle the terminal err default.
        Call once after the stream ends.

        ``reason``: what the CALLER knows about why the stream ended — a
        wall-clock kill, an idle-kill, a non-zero exit. The collector can only
        observe "no result event arrived", which on its own is a symptom, not a
        cause; reporting that symptom as the error told the user strictly less
        than nothing (it reads like Claude misbehaved when in fact Augmentum
        terminated it). When the caller supplies a real cause, it wins.
        """
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            await self._do_emit({"kind": "message", "text": tail[:200]})
        if not self.ok and not self.err:
            self.err = reason or (
                "Claude ended without a result event — the process stopped "
                "before reporting an outcome, and Augmentum did not terminate "
                "it. Check the workspace container logs for a crash."
            )

    @property
    def status(self) -> str:
        return "done" if self.ok else "failed"

    @property
    def outcome(self) -> str:
        return summary_from_raw(self.raw_lines) if self.ok else ""

    @property
    def raw_jsonl(self) -> str:
        return "\n".join(self.raw_lines)

    async def _capture_meta(self, obj: dict) -> None:
        sid = obj.get("session_id")
        if sid and not self.meta["session_id"]:
            self.meta["session_id"] = sid
            if self._on_session_id:
                await self._on_session_id(sid)
        # The model rides the `system/init` handshake (and often the result).
        # Capture the first one seen so we record what the engine actually used,
        # even when the user left the choice on "Account default".
        if not self.meta["model"]:
            model = obj.get("model")
            if model:
                self.meta["model"] = str(model)
        if obj.get("type") == "result":
            with contextlib.suppress(Exception):
                self.meta["cost_usd"] = float(obj.get("total_cost_usd") or self.meta["cost_usd"])
                self.meta["num_turns"] = int(obj.get("num_turns") or self.meta["num_turns"])
                self.meta["duration_ms"] = int(obj.get("duration_ms") or self.meta["duration_ms"])

    async def _do_emit(self, item: dict) -> None:
        if self._emit:
            await self._emit(item)
