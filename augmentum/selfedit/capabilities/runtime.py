"""Production ``model_invoke`` for capability synthesis — DIRECT by construction.

The live runs taught us that synthesis must NOT go through the user-chat pipeline
(classification + intent-dispatch + memory injection mangled the structured
prompts — deepseek-v4-flash 400'd and got re-routed to the classifier slot; the
Qwen runs were silently contaminated by memory injection). This builds a
``model_invoke`` that calls the backend DIRECTLY — ``resolve_backend_for_model``
→ ``backend.chat`` — so there is no pipeline to contaminate it.

It also doesn't force a ``temperature`` (DeepSeek V4 and other reasoning models'
providers 400 on a temperature override): the backend uses its own defaults.
``think`` stays on for reasoning quality.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ModelInvoke = Callable[[str], Awaitable[str]]


def build_direct_model_invoke(app_state: Any, *, model: str = "") -> ModelInvoke:
    """Return an ``async (prompt) -> text`` that calls ``model`` (or the install
    default) directly, bypassing the chat pipeline. Raises on backend failure —
    the synthesis/triage callers already safe-degrade on a raising model_invoke."""

    async def _invoke(prompt: str) -> str:
        registry = getattr(app_state, "provider_registry", None)
        if registry is None:
            raise RuntimeError("provider_registry unavailable")
        backend, clean = await registry.resolve_backend_for_model(model or "")
        from augmentum.models.base import InternalChatRequest, Message
        resp = await backend.chat(InternalChatRequest(
            model=clean,
            messages=[Message(role="user", content=prompt)],
            think=True,
            stream=False,
        ))
        return (getattr(getattr(resp, "message", None), "content", "") or "")

    return _invoke
