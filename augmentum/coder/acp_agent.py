"""AugmentumACPAgent — Augmentum's coder brain exposed as an ACP agent.

This is the "loop in the editor" endpoint: an ``acp.Agent`` whose ``prompt``
runs Augmentum's own coder loop against a workspace it does NOT own (the user's
editor), reached through the ACP client methods (fs/terminal) via
:class:`ACPEditorChannel` + :class:`RemoteEditorExecutor`.

Design decision (in-process): the agent runs INSIDE Augmentum's single-process
asyncio server, so it shares model residency, memory, identity, and one auth
plane with chat/voice — it is just one more cooperative async consumer. The
concurrency invariants that keep it from disturbing other users (a second chat,
an image gen) are baked in here:

  * Per-session isolation — each ACP session owns its OWN RemoteEditorExecutor,
    ACPEditorChannel and cancel Event, keyed by session_id. No shared mutable
    state across sessions; the only shared resource is the model slot, which the
    engine already serialises fairly.
  * A turn semaphore — coder turns hold a model slot far longer than a "hey
    there" chat, so concurrent editor turns are bounded to avoid starving
    interactive chat. Chat/voice keep their own lighter paths.
  * Cancellation — an ACP ``cancel`` sets the session's Event, which the loop
    checks between hops and stops promptly, freeing the slot.
  * Fully async — the agent only ever ``await``s (the loop, session_update); it
    never blocks the event loop. (The one hard rule for editor-path TOOLS is the
    same: no unbounded sync CPU work on the loop — offload with asyncio.to_thread.)

The actual coder loop is injected as ``loop_runner`` (an async generator of
``(kind, payload)`` events, exactly what ``native_loop_events`` yields). That
keeps this agent fully unit-testable without a running app or a real editor; the
production runner (which wires ``native_loop_events`` with the live
backend/runtime/registry/app_state) is the remaining integration seam.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.coder.acp_bridge import ACPEditorChannel
from augmentum.coder.executors import RemoteEditorExecutor
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

try:
    import acp
except ImportError:  # pragma: no cover - the agent is only used with the SDK
    acp = None  # type: ignore[assignment]

# (kind, payload) — the native_loop event shape. kinds: text/tool_call/
# tool_result/metrics/ui_effects.
LoopEvent = tuple[str, dict]
LoopRunner = Callable[["EditorSession", str], AsyncIterator[LoopEvent]]

_DEFAULT_MAX_CONCURRENT_TURNS = 4


@dataclass
class EditorSession:
    """Isolated per-session state — one live editor window/tab."""

    session_id: str
    cwd: str
    executor: RemoteEditorExecutor
    channel: ACPEditorChannel
    user_id: str = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    # FIFO of (tool_name, acp_tool_call_id) so a tool_result can be correlated
    # back to its tool_call for the ACP start/update pairing.
    pending_tools: deque[tuple[str, str]] = field(default_factory=deque)


def _pop_tool_id(sess: EditorSession, tool: str) -> str:
    """Match a tool_result to a pending tool_call id (by name, else FIFO)."""
    for i, (name, call_id) in enumerate(sess.pending_tools):
        if name == tool:
            del sess.pending_tools[i]
            return call_id
    if sess.pending_tools:
        return sess.pending_tools.popleft()[1]
    return f"tc-{uuid.uuid4().hex[:8]}"


class AugmentumACPAgent(acp.Agent if acp is not None else object):  # type: ignore[misc]
    """Augmentum coder loop as an ACP agent (see module docstring)."""

    def __init__(
        self,
        *,
        loop_runner: LoopRunner,
        max_concurrent_turns: int = _DEFAULT_MAX_CONCURRENT_TURNS,
        output_byte_limit: int = 1_000_000,
        default_user_id: str = "",
    ) -> None:
        if acp is None:  # pragma: no cover - guarded at construction
            raise RuntimeError(
                "the ACP SDK is required to run the editor agent — "
                "pip install agent-client-protocol",
            )
        self._loop_runner = loop_runner
        self._conn: Any = None
        self._sessions: dict[str, EditorSession] = {}
        # Connection-scoped tenant: the WS carrier authenticates the sk-aug key
        # ONCE at connect and binds the resulting user_id here. The ACP client
        # (Zed) never sends a user_id on new_session, so every session this
        # agent creates inherits this — the isolation guarantee for the
        # in-process endpoint (a tenant's editor can only touch their own data).
        self._default_user_id = default_user_id
        # Fairness cap: bounds CONCURRENT coder turns so a burst of editor
        # agents can't crowd interactive chat off the model slots.
        self._turn_sem = asyncio.Semaphore(max_concurrent_turns)
        self._output_byte_limit = output_byte_limit

    # -- connection ----------------------------------------------------------

    def on_connect(self, conn: Any) -> None:
        # The AgentSideConnection we call the client (editor) through.
        self._conn = conn

    # -- lifecycle -----------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **_: Any,
    ) -> Any:
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=acp.schema.AgentCapabilities(
                prompt_capabilities=acp.schema.PromptCapabilities(
                    image=False, audio=False, embedded_context=True,
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        *,
        user_id: str = "",
        **_: Any,
    ) -> Any:
        session_id = f"acp-{uuid.uuid4().hex[:12]}"
        channel = ACPEditorChannel(
            self._conn, session_id, output_byte_limit=self._output_byte_limit,
        )
        executor = RemoteEditorExecutor(channel, workspace_root=cwd or "/workspace")
        self._sessions[session_id] = EditorSession(
            session_id=session_id, cwd=cwd or "/workspace",
            executor=executor, channel=channel,
            user_id=user_id or self._default_user_id,
        )
        log.info("acp_new_session", session_id=session_id, cwd=cwd)
        return acp.NewSessionResponse(session_id=session_id)

    async def close_session(self, session_id: str, **_: Any) -> Any:
        self._sessions.pop(session_id, None)
        return None

    async def cancel(self, session_id: str, **_: Any) -> None:
        sess = self._sessions.get(session_id)
        if sess is not None:
            sess.cancel.set()

    # -- the turn ------------------------------------------------------------

    async def prompt(self, session_id: str, prompt: Any, **_: Any) -> Any:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise acp.RequestError.invalid_params(f"unknown session {session_id}")
        sess.cancel.clear()
        sess.pending_tools.clear()
        text = _extract_prompt_text(prompt)

        stop_reason = "end_turn"
        # Bound concurrent coder turns so editor agents don't starve chat.
        async with self._turn_sem:
            async for kind, payload in self._loop_runner(sess, text):
                if sess.cancel.is_set():
                    stop_reason = "cancelled"
                    break
                await self._emit(sess, kind, payload or {})
        return acp.PromptResponse(stop_reason=stop_reason)

    # -- event translation (native_loop -> ACP session_update) ---------------

    async def _emit(self, sess: EditorSession, kind: str, payload: dict) -> None:
        conn = self._conn
        sid = sess.session_id
        if kind == "text":
            txt = str(payload.get("text", "") or "")
            if txt:
                await conn.session_update(sid, update=acp.update_agent_message_text(txt))
        elif kind == "thought":
            txt = str(payload.get("text", "") or "")
            if txt:
                await conn.session_update(sid, update=acp.update_agent_thought_text(txt))
        elif kind == "tool_call":
            tool = str(payload.get("tool", "tool"))
            call_id = f"tc-{uuid.uuid4().hex[:8]}"
            sess.pending_tools.append((tool, call_id))
            await conn.session_update(
                sid,
                update=acp.start_tool_call(
                    call_id, title=tool, status="in_progress",
                    raw_input=payload.get("args"),
                ),
            )
        elif kind == "tool_result":
            tool = str(payload.get("tool", "tool"))
            call_id = _pop_tool_id(sess, tool)
            ok = bool(payload.get("ok", True))
            summary = str(payload.get("snippet") or payload.get("payload_summary") or "")
            await conn.session_update(
                sid,
                update=acp.update_tool_call(
                    call_id,
                    status="completed" if ok else "failed",
                    raw_output=summary or None,
                ),
            )
        # "metrics" and "ui_effects" are Augmentum-surface events, not editor
        # updates — intentionally dropped on the editor path.


def _extract_prompt_text(prompt: Any) -> str:
    """Concatenate the text of a PromptRequest's content blocks."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for block in prompt or []:
        txt = getattr(block, "text", None)
        if txt is None and isinstance(block, dict):
            txt = block.get("text")
        if txt:
            parts.append(str(txt))
    return "\n".join(parts)


async def run_stdio(loop_runner: LoopRunner, **kwargs: Any) -> None:  # pragma: no cover
    """Serve the agent over stdio (an editor spawns this as a subprocess).

    Thin wrapper over ``acp.run_agent``. Used by the co-located stdio-bridge
    deployment; the native in-process WSS deployment reuses the same
    :class:`AugmentumACPAgent` with a different transport.
    """
    if acp is None:
        raise RuntimeError("pip install agent-client-protocol to run the ACP agent")
    agent = AugmentumACPAgent(loop_runner=loop_runner, **kwargs)
    await acp.run_agent(agent)
