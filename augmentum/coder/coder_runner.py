"""Coder-loop runner for the ACP editor agent.

The bridge between :class:`AugmentumACPAgent` (which consumes a ``loop_runner``
yielding ``(kind, payload)`` events) and Augmentum's real CODER loop
(``CoderHandler.handle_stream`` — NOT the companion ``native_loop``). It:

  * builds a ``CoderHandler`` bound to the session's ``RemoteEditorExecutor``
    (so the ~45 coder tools act on the editor, not a Docker container), and
  * streams ``handler.handle_stream(request)`` and translates each
    ``InternalStreamChunk`` into the agent's event tuples.

Design note — why tool VISUALS are polish, not function: the file edits an
editor turn makes happen through the executor (``RemoteEditorExecutor`` ->
``fs/write_text_file`` -> the editor applies), NOT through the ACP tool-call
updates. So the verified stream fields (``content_delta`` -> text,
``thinking_delta`` -> thought, ``done``) are all that's needed for a working
turn; the ``tool_call``/``tool_result`` ACP updates are chat-panel decoration
that ride in the ``InternalStreamChunk.augmentum`` metadata and are wired
separately once their shape is confirmed against a live turn.

Dependency injection: the factory takes ``build_handler`` / ``build_request``
callables rather than reaching into ``app_state`` itself, so it is unit-testable
with fakes and honest about exactly what the ACP endpoint must supply (a backend,
tool_registry, provider_registry, state_manager, etc., assembled the same way
``handler_factory`` does for chat/coder mode).
"""
from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any

from augmentum.coder.acp_agent import EditorSession, LoopEvent
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def coder_chunk_to_events(chunk: Any) -> list[LoopEvent]:
    """Translate one ``InternalStreamChunk`` into agent event tuples.

    Pure + total: reads only the confirmed fields (``thinking_delta``,
    ``content_delta``) and never raises on a partial chunk. Thinking is emitted
    before content so the editor shows the thought stream ahead of the answer,
    mirroring the chat surface.
    """
    events: list[LoopEvent] = []
    thinking = getattr(chunk, "thinking_delta", None)
    if thinking:
        events.append(("thought", {"text": str(thinking)}))
    content = getattr(chunk, "content_delta", None)
    if content:
        events.append(("text", {"text": str(content)}))
    return events


def make_coder_loop_runner(
    *,
    build_handler: Callable[[EditorSession], Any],
    build_request: Callable[[EditorSession, str], Any],
) -> Callable[[EditorSession, str], AsyncIterator[LoopEvent]]:
    """Build a ``loop_runner`` that drives the real coder loop for a session.

    Parameters
    ----------
    build_handler:
        ``(session) -> CoderHandler`` — constructs a handler bound to
        ``session.executor`` (the RemoteEditorExecutor) and the session's
        workspace/user, with the live backend + registries. This is where the
        ACP endpoint injects app_state (same assembly as ``handler_factory``).
        MAY be a coroutine function: resolving the model backend
        (``provider_registry.resolve_backend_with_fabric``) is async, so the
        runner awaits the result if ``build_handler`` returns an awaitable. Unit
        tests pass a plain sync fake and it works unchanged.
    build_request:
        ``(session, prompt_text) -> InternalChatRequest`` — wraps the user's
        turn text into the request the handler streams. Takes the session so it
        can derive the coder KV-affinity keys (``kv_session_key`` / ``kv_mode``)
        that keep an editor turn on the same warm slot as prior turns.
    """

    async def loop_runner(session: EditorSession, prompt_text: str) -> AsyncIterator[LoopEvent]:
        handler = build_handler(session)
        if inspect.isawaitable(handler):
            handler = await handler
        request = build_request(session, prompt_text)
        stream = handler.handle_stream(request)
        try:
            async for chunk in stream:
                for ev in coder_chunk_to_events(chunk):
                    yield ev
                if getattr(chunk, "done", False):
                    break
        finally:
            # Break-on-cancel (the agent stops consuming) must close the
            # underlying coder stream so an in-flight turn doesn't leak.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception as exc:  # noqa: BLE001 — cleanup best-effort
                    log.debug("coder_stream_aclose_failed", error=str(exc))

    return loop_runner
