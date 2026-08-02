"""NativeModelDriver — a LOCAL model as a coder/self-edit backend (the sovereign
path).

Claude Code and Codex are external *authenticable platforms*. A local model
(Augmentum's model list / bundled llama-server) becomes a coder backend by running
Augmentum's OWN agentic tool-loop over it — no token, your box, your model. It
implements the SAME ``ExternalCoderDriver`` contract, so it registers in the
registry next to Claude/Codex and flows through the same self-edit bridge
(``selfedit.external_edit_driver``) and the same normalized ``CoderEvent`` stream.
One path, three backend kinds: external platform, local model, (and the classifier
uses the lightweight model-source layer).

The agentic loop is INJECTED as ``run_loop`` — the coder's native tool-loop
(harness-selected per model, `coder/harness.py`) driving ``resolve_model_for_role``
with file-edit tools over the workspace. This module owns ONLY the normalization
(loop events → CoderEvents, reusing ``base.tool_use_event`` so file-change/mutating
detection is identical to the SDK/CLI paths). Pure and testable with a fake loop.

Quality scales with the local model — a strong served coder model can self-edit;
the small classifier slot can't. The self-edit gate + verifier are exactly what
make a weaker local agent's output SAFE (verify catches its mistakes), so local
self-edit is feasible and the verification spine is the guardrail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from augmentum.coder.external.base import (
    CoderEvent,
    ExternalCoderDriver,
    ExternalTask,
    tool_use_event,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Injected agentic loop: a task → a stream of generic loop events (dicts). Shapes:
#   {"kind": "message",   "text": ...}                 assistant prose
#   {"kind": "tool_call", "tool": ..., "args": {...}}  a tool invocation (classified here)
#   {"kind": "completed", "text": ..., "session_id": ...}
#   {"kind": "failed",    "text": ...}
NativeLoop = Callable[[ExternalTask], AsyncIterator[dict]]


class NativeModelDriver(ExternalCoderDriver):
    """Wrap Augmentum's native agentic loop (over a local model) as a driver."""

    def __init__(self, *, run_loop: NativeLoop,
                 available: Callable[[], Awaitable[bool]] | None = None,
                 model: str = "", did: str = "native",
                 label: str = "Local model (native)"):
        self.id = did
        self.label = label
        self.model = model
        self._run_loop = run_loop
        self._available = available

    async def is_available(self) -> bool:
        if self._available is None:
            return True
        try:
            return await self._available()
        except Exception:  # noqa: BLE001 — a probe error means "not available"
            return False

    async def run(self, task: ExternalTask) -> AsyncIterator[CoderEvent]:
        try:
            async for ev in self._run_loop(task):
                kind = str(ev.get("kind", ""))
                if kind == "thinking":
                    text = str(ev.get("text", ""))
                    if text:
                        yield CoderEvent(kind="thinking", text=text, raw=ev)
                elif kind in ("message", "prose", "text"):
                    text = str(ev.get("text", ""))
                    if text:
                        yield CoderEvent(kind="message", text=text)
                elif kind == "tool_call":
                    # Reuse the shared classifier → file_change/command_exec/mcp/tool_call.
                    yield tool_use_event(str(ev.get("tool", "")), ev.get("args") or {}, ev)
                elif kind == "tool_result":
                    # The tool's RETURN (per-write syntax feedback, reads, errors) —
                    # persisted so the transcript shows what each call produced.
                    yield CoderEvent(kind="tool_result", text=str(ev.get("text", "")),
                                     tool=str(ev.get("tool", "")),
                                     path=str(ev.get("path", "")), raw=ev)
                elif kind in ("completed", "done"):
                    yield CoderEvent(kind="completed", text=str(ev.get("text", "")), raw=ev)
                    return
                elif kind in ("failed", "error"):
                    yield CoderEvent(kind="failed",
                                     text=str(ev.get("text", "")) or "run failed", raw=ev)
                    return
            # Loop ended without an explicit terminal → treat as completed.
            yield CoderEvent(kind="completed", text="")
        except Exception as exc:  # noqa: BLE001 — normalize to a failed event, never raise
            log.warning("native_model_driver_loop_error", error=repr(exc))
            yield CoderEvent(kind="failed", text=repr(exc))
