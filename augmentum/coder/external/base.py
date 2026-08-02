"""External-coder driver abstraction + normalized event model.

Both Claude Code (Agent SDK) and Codex (`codex exec`) run headless, stream
structured events, sandbox, and resume. This module defines the ONE shape the
rest of Augmentum sees, so the trail / surface / consent gate / engineering_log
never branch on which engine ran.

Mutation awareness is first-class: ``CoderEvent.mutating`` lets the consent gate
intercept exactly the events that change workspace state, regardless of engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

CoderEventKind = Literal[
    "started",       # run accepted, agent loop beginning
    "thinking",      # reasoning / plan-in-progress (non-actionable)
    "message",       # assistant prose
    "plan",          # an explicit plan/todo update
    "tool_call",     # the agent invoked a tool
    "tool_result",   # what a tool returned (per-write syntax feedback, reads, errors)
    "file_change",   # a file was written/edited (mutating)
    "command_exec",  # a shell command ran (potentially mutating)
    "mcp_call",      # an MCP tool was invoked
    "needs_approval",  # the agent is waiting on a permission decision
    "completed",     # run finished successfully
    "failed",        # run errored / aborted
]

# Tool names that change workspace state. Used to set CoderEvent.mutating so the
# consent gate fires on exactly these. Engine-neutral (covers Claude Code +
# Codex tool vocabularies); unknown tools default to NON-mutating (read-safe).
_MUTATING_TOOLS: frozenset[str] = frozenset({
    # Claude Code
    "Write", "Edit", "MultiEdit", "NotebookEdit", "Bash",
    # Codex / generic
    "apply_patch", "shell", "exec", "write_file", "edit_file",
})


def is_mutating_tool(name: str) -> bool:
    return (name or "") in _MUTATING_TOOLS


# Tool-input keys that carry a file path (Write/Edit/etc.).
_FILE_PATH_KEYS = ("file_path", "notebook_path", "path")

# Tool-input keys (priority order) that carry a human-meaningful "what" for a
# tool call — so the transcript reads "Read main.py" / "Grep TODO" instead of
# the bare, duplicated tool name.
_TOOL_TARGET_KEYS = (
    "file_path", "path", "pattern", "query", "url", "command", "description",
    "prompt", "notebook_path",
)


def _tool_target(ti: dict) -> str:
    """A concise descriptor of a tool call's subject, pulled from its input."""
    for k in _TOOL_TARGET_KEYS:
        v = ti.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:160]
    return ""


def tool_use_event(name: str, tool_input: object, raw: dict | None = None) -> CoderEvent:
    """Classify a tool invocation into a normalized CoderEvent. Shared by the
    SDK path (claude_code) and the CLI/stream-json path (claude_cli) so both
    agree on what counts as a file_change / command_exec / mcp_call and what's
    mutating (→ consent gate)."""
    name = name or ""
    ti = tool_input if isinstance(tool_input, dict) else {}
    raw = raw or {}
    if name.startswith("mcp__"):
        return CoderEvent(kind="mcp_call", tool=name, text=_tool_target(ti), raw=raw)
    if name == "Bash":
        cmd = (ti.get("command") or "")[:200] if isinstance(ti.get("command"), str) else ""
        return CoderEvent(kind="command_exec", tool=name, text=cmd, mutating=True, raw=raw)
    if is_mutating_tool(name):
        path = ""
        for k in _FILE_PATH_KEYS:
            if ti.get(k):
                path = str(ti[k])
                break
        return CoderEvent(kind="file_change", tool=name, path=path, mutating=True, raw=raw)
    # Non-mutating tool (Read/Grep/Glob/Task/WebFetch/…): show its subject, not
    # the tool name twice.
    return CoderEvent(kind="tool_call", tool=name, text=_tool_target(ti), raw=raw)


@dataclass
class CoderEvent:
    """One normalized event from any external coder run."""
    kind: CoderEventKind
    text: str = ""                 # human-readable summary (trail/surface/narration)
    tool: str = ""                 # tool name for tool_call/file_change/command_exec
    path: str = ""                 # file path for file_change
    mutating: bool = False         # changes workspace state → consent gate intercepts
    raw: dict[str, Any] = field(default_factory=dict)  # original event (debug/telemetry)


@dataclass
class ExternalTask:
    """A unit of work handed to an external coder."""
    prompt: str
    workspace: str = "/workspace"
    # "confirm_mutations" (default for companion-initiated — user approves writes)
    # or "auto" (run unattended within budget). Mapped per-engine in the driver.
    permission: Literal["confirm_mutations", "auto", "plan"] = "confirm_mutations"
    allowed_tools: tuple[str, ...] = ()      # empty = engine default
    mcp_servers: dict[str, Any] = field(default_factory=dict)  # Augmentum-as-toolbox
    model: str = ""                          # engine-specific model override
    max_tokens: int = 0                      # 0 = engine default
    budget_s: float = 0.0                    # wall-clock cap; 0 = driver default
    # Free-text the user/companion attached about WHY this matters — threaded
    # into the engineering_log so recall feels personal.
    framing: str = ""


@dataclass
class ExternalRunResult:
    """Terminal outcome of an external coder run."""
    ok: bool
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    session_ref: str = ""          # engine session/thread id — enables resume + recall
    error: str = ""


class ExternalCoderDriver(ABC):
    """Drive one external agentic coder headlessly, emitting normalized events.

    Implementations: ClaudeCodeDriver (Agent SDK), CodexDriver (`codex exec`).
    The driver owns engine-specifics (auth, wire format, permission mapping); the
    caller (companion verb OR coder UI) only ever sees CoderEvents + a result.
    """

    id: str = ""        # stable key: "claude_code" | "codex"
    label: str = ""     # human label: "Claude Code" | "Codex"

    @abstractmethod
    async def is_available(self) -> bool:
        """True iff this engine can actually run here (SDK/binary importable AND
        a usable credential present). Drivers that aren't available are never
        offered — Augmentum degrades to the native coder."""
        ...

    @abstractmethod
    def run(self, task: ExternalTask) -> AsyncIterator[CoderEvent]:
        """Run ``task`` and stream normalized events. The final event is
        ``completed`` or ``failed``. Implemented as an async generator."""
        ...

    async def interrupt(self) -> None:
        """Best-effort stop of the in-flight run (maps to the engine's
        interrupt/stop). Default no-op; drivers override."""
        return None


def to_engineering_record(
    task: ExternalTask,
    result: ExternalRunResult,
    *,
    engine_label: str,
) -> dict[str, str]:
    """Project a finished run into the args for
    ``engineering_log.record_engineering_outcome`` — the seam that closes the
    persistence loop so the companion carries the work across sessions. Pure +
    testable; the caller does the actual write."""
    n = len(result.files_changed)
    if result.ok:
        outcome = "done"
        if n:
            outcome += f" — {n} file{'s' if n != 1 else ''} changed"
        if result.summary.strip():
            outcome = result.summary.strip()[:200]
    else:
        outcome = f"didn't finish: {result.error.strip()[:160]}" if result.error else "didn't finish"
    return {
        "task": (task.prompt or "").strip()[:200],
        "outcome": outcome,
        "engine": engine_label,
        "framing": (task.framing or "").strip()[:160],
        "resume_ref": result.session_ref or "",
    }
