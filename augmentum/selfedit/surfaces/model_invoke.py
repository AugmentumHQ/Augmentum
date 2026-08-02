"""Wire the reshape classifier's model call to Augmentum's role-model layer.

The classifier needs ONE fast prompt→JSON call — not a Claude Code agentic
session. The right reuse is the same seam architect/voice/companion already use:
``resolve_model_for_role("classifier")`` → a resident, fast backend. It is
model-AGNOSTIC: it resolves to whatever the user configured for the classifier
role — the local resident slot by DEFAULT (sovereign, free, no token), OR a Claude
model via the real Anthropic backend (``models/adapters/claude.py``) if one is
set. So "use Claude" becomes a config choice, never a hardcode.

(The coder-mode Claude CLI integration — ``coder/external/claude_cli.py`` — is the
right reuse for the *code-edit driver*: a full agentic session in a container with
a token. It is the wrong tool for this one-shot call.)

The registry is injected, so this is testable with a fake backend; any failure
returns "" → the classifier reads that as an honest *unmapped*, never a crash.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.models.base import InternalChatRequest, Message
from augmentum.selfedit.surfaces.classify import ModelInvoke
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SYSTEM = "You are a precise classifier. Respond with JSON only, no prose."


def build_role_model_invoke(registry: Any, settings: Any = None, *,
                            role: str = "classifier", timeout_s: float = 8.0,
                            max_tokens: int = 384) -> ModelInvoke:
    """Return a ``ModelInvoke`` backed by ``resolve_model_for_role(role)``. Mirrors
    the architect/voice fast-call recipe (greedy, no-thinking, small budget)."""

    async def invoke(prompt: str) -> str:
        try:
            backend, model = await registry.resolve_model_for_role(role, settings=settings)
        except Exception as exc:  # noqa: BLE001 — degrade to unmapped, never raise out
            log.warning("reshape_invoke_resolve_failed", role=role, error=repr(exc))
            return ""
        if backend is None:
            log.info("reshape_invoke_no_backend", role=role)
            return ""

        req = InternalChatRequest(
            model=model or "",
            messages=[Message(role="system", content=_SYSTEM),
                      Message(role="user", content=prompt)],
            stream=False,
            temperature=0.0,
            # Same no-thinking contract as the voice/architect classifier hop —
            # without it, thinking-mode families burn the budget on CoT before JSON.
            chat_template_kwargs={"enable_thinking": False},
            max_tokens=max_tokens,
        )
        try:
            resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 — timeout/backend error → unmapped
            log.warning("reshape_invoke_chat_failed", role=role, error=repr(exc))
            return ""
        return getattr(getattr(resp, "message", None), "content", "") or ""

    return invoke
