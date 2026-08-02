"""Production wiring — assemble the REAL coder loop for an ACP editor session.

:mod:`coder_runner` is transport-agnostic and injects the loop via
``build_handler`` / ``build_request`` callables. This module supplies the
PRODUCTION versions of those callables, wired to the live ``app_state`` exactly
the way :func:`augmentum.proxy.handler_factory.get_handler_for_mode` and the
headless coder-background job assemble a real ``CoderHandler``. Swap
:func:`make_app_loop_runner` in for the smoke runner in ``acp_stdio`` (or mount
it on the in-process WSS endpoint) and the editor drives Augmentum's actual
Plan/Act coder brain — same models, tools, memory, and identity as chat.

WHERE THIS RUNS (the one open deployment fork): the assembly needs ``app_state``
(provider registry, model backends, state manager, container/tool registries),
which lives in the running Augmentum process. Two carriers, both consuming THIS
module unchanged:

  * **In-process WSS endpoint** — mount :class:`AugmentumACPAgent` on a WS route
    in the main server; a thin stdio<->WSS bridge is the ``Command`` Zed spawns.
    Preserves single-process model residency (chat/voice/editor share one slot
    queue). This is the production path.
  * **Standalone stdio boot** — the ``acp_stdio`` subprocess boots its own
    ``app_state``. Simpler, but a second runtime that contends for model slots
    if the server is also up. Fine for a single-user local session.

Editor-path deltas from the container assembly (only two):
  1. ``handler._executor = session.executor`` — the ~45 coder tools act on the
     user's editor (RemoteEditorExecutor -> ACP fs/terminal), NOT a Docker
     container Augmentum owns.
  2. ``handler._permission_callback`` — the EDITOR owns approval (Zed prompts on
     the fs/write it applies), so the loop-level gate defaults to permissive
     rather than the container permission-policy modal (which no one would
     answer from inside an editor). Routing to ACP ``session/request_permission``
     is the documented follow-up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.coder.coder_runner import make_coder_loop_runner
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from augmentum.coder.acp_agent import EditorSession

log = get_logger(__name__)

# Coder mode runs its OWN loop (_act_native); "native" selects it over the
# retired legacy strategy. See modes/coder/phase_act.py.
DEFAULT_CODER_STRATEGY = "native"

# Permission callback signature: ``async (tool_name, tool_input) -> bool``.


async def _auto_allow(_tool_name: str, _tool_input: dict) -> bool:
    """Loop-level auto-approve — the editor is the approval authority.

    Zed (and any ACP editor) prompts the user on the actual mutating fs op it
    applies, so gating again at the loop level would either double-prompt or,
    worse, block on a container-policy modal that has no surface inside the
    editor. Read-only tools need no approval; mutations are gated editor-side.
    """
    return True


def build_coder_request(session: EditorSession, prompt_text: str, *, model: str = "") -> Any:
    """Wrap an editor turn's text into the ``InternalChatRequest`` the loop streams.

    Mirrors the headless coder-background job: single user message + the coder
    KV-affinity keys so an editor turn shares the same warm slot as prior turns
    on this session (without them, ``kv_tier=unmanaged`` -> full re-prefill every
    iteration). Conversation-history continuity is a follow-up; a fresh editor
    session starts from the current turn.
    """
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.proxy.session import derive_kv_session_key

    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content=prompt_text)],
        stream=True,
        kv_session_key=derive_kv_session_key(session.user_id, session.session_id),
        kv_mode="coder",
    )


async def build_coder_handler(
    app_state: Any,
    session: EditorSession,
    *,
    model: str = "",
    coder_strategy: str = DEFAULT_CODER_STRATEGY,
    permission_callback: Any = None,
) -> Any:
    """Assemble a real ``CoderHandler`` bound to the editor session.

    Reuses the canonical factory (no assembly drift), then applies the two
    editor-path deltas. Raises if the model is unavailable or the factory falls
    back to passthrough (a tool-less passthrough "coder" would burn tokens
    without ever touching the editor).
    """
    from augmentum.classifier.router import Mode
    from augmentum.proxy.handler_factory import get_handler_for_mode

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        raise RuntimeError("provider registry unavailable — cannot serve the coder loop")
    backend, resolved_model = await registry.resolve_backend_with_fabric(
        model, user_id=session.user_id,
    )
    if backend is None:
        raise RuntimeError(f"model unavailable: {model!r}")

    # The ACP session id is the workspace identity for an editor session — stable
    # across turns in one Zed thread, isolated from other windows and from
    # container workspaces. It seeds both CoderState continuity and KV affinity.
    handler = get_handler_for_mode(
        Mode.CODER,
        backend,
        session.session_id,
        app_state,
        workspace_id=session.session_id,
        user_id=session.user_id,
        coder_strategy=coder_strategy,
    )
    if type(handler).__name__ != "CoderHandler":
        raise RuntimeError(
            f"coder handler unavailable (factory fell back to {type(handler).__name__})",
        )

    # Editor-path deltas (see module docstring).
    handler._executor = session.executor
    handler._permission_callback = permission_callback or _auto_allow
    log.info(
        "acp_coder_handler_built",
        session_id=session.session_id,
        model=resolved_model or model,
    )
    return handler


def make_app_loop_runner(
    app_state: Any,
    *,
    model: str = "",
    coder_strategy: str = DEFAULT_CODER_STRATEGY,
    permission_callback: Any = None,
) -> Callable[[EditorSession, str], Any]:
    """Build the production ``loop_runner`` for :class:`AugmentumACPAgent`.

    Closes over ``app_state`` + the chosen model so each editor turn assembles a
    fresh handler bound to its session's ``RemoteEditorExecutor``. Drop-in
    replacement for the smoke runner in ``acp_stdio``.
    """

    async def _build_handler(session: EditorSession) -> Any:
        return await build_coder_handler(
            app_state, session,
            model=model, coder_strategy=coder_strategy,
            permission_callback=permission_callback,
        )

    def _build_request(session: EditorSession, prompt_text: str) -> Any:
        return build_coder_request(session, prompt_text, model=model)

    return make_coder_loop_runner(
        build_handler=_build_handler, build_request=_build_request,
    )
