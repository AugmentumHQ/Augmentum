"""Pluggable model SOURCES for the self-edit / reshape model call.

Matt's shape: Augmentum's own model list is the DEFAULT, and users can *sub in* an
authenticable platform — Claude today (reusing the coder-mode login), Codex and
other token/OAuth platforms later — for the classifier/self-edit reasoning. Each
source is a named builder that yields a ``ModelInvoke`` (or None if it isn't
available for this user). Selection is a per-user/global setting; an unavailable
choice falls back to the model list, so picking a platform can never strand the
feature.

Honest auth reality (verified): a coder Claude credential is either an
``sk-ant-api…`` API key (works with the real Anthropic backend via ``x-api-key``)
or an ``sk-ant-oat01…`` subscription OAuth token (only usable through the Claude
Code CLI/Agent SDK — NOT the messages API; we do not fake it). So the Claude
source serves the API-key case directly and honestly skips the OAuth case (that's
the CLI driver's domain — the same one the code-edit path already uses).

All heavy deps (provider registry, settings store, the Anthropic backend factory)
are passed via ``SourceContext`` and injectable, so this is testable with fakes
and constructs nothing at import time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from augmentum.selfedit.surfaces.classify import ModelInvoke
from augmentum.selfedit.surfaces.model_invoke import build_role_model_invoke
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Setting key the route/UI reads to choose a source (default = the model list).
SOURCE_SETTING_KEY = "selfedit_model_source"
DEFAULT_SOURCE = "augmentum"

# A fast, cheap default for one-shot classification when routing to Claude's API.
_DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_CLASSIFIER_SYSTEM = "You are a precise classifier. Respond with JSON only, no prose."


@dataclass
class SourceContext:
    """Everything a source builder might need, all injectable."""

    user_id: str = ""
    provider_registry: Any = None         # Augmentum's model list (role layer)
    settings_store: Any = None            # per-user secrets (coder login lives here)
    app_settings: Any = None
    claude_model: str = ""                # override the one-shot Claude model
    # token -> Anthropic backend; default builds the real ClaudeBackend lazily.
    claude_backend_factory: Callable[[str], Any] | None = None
    # (settings_store, user_id) -> credential; default = the coder token store.
    claude_token_loader: Callable[[Any, str], Awaitable[str]] | None = None


# A source builds a ModelInvoke for a context, or None if unavailable here.
SourceBuilder = Callable[[SourceContext], Awaitable[ModelInvoke | None]]


@dataclass
class ModelSource:
    id: str
    label: str
    build: SourceBuilder


_SOURCES: dict[str, ModelSource] = {}


def register_source(source: ModelSource) -> None:
    _SOURCES[source.id] = source
    log.info("reshape_model_source_registered", source=source.id)


def get_source(source_id: str) -> ModelSource | None:
    return _SOURCES.get(source_id)


def list_sources() -> list[dict]:
    return [{"id": s.id, "label": s.label} for s in _SOURCES.values()]


def clear_sources() -> None:  # for tests
    _SOURCES.clear()


async def resolve_invoke(source_id: str, ctx: SourceContext, *,
                         fallback: str = DEFAULT_SOURCE) -> ModelInvoke | None:
    """Build the ModelInvoke for ``source_id``; if that source is unknown or
    unavailable for this user, fall back to the model list so the feature never
    strands on a bad platform choice."""
    src = get_source(source_id)
    if src is not None:
        inv = await src.build(ctx)
        if inv is not None:
            return inv
        log.info("reshape_model_source_unavailable", source=source_id, falling_back_to=fallback)
    if source_id != fallback:
        fb = get_source(fallback)
        if fb is not None:
            return await fb.build(ctx)
    return None


# --- default sources -------------------------------------------------------

async def _build_augmentum(ctx: SourceContext) -> ModelInvoke | None:
    if ctx.provider_registry is None:
        return None
    return build_role_model_invoke(ctx.provider_registry, ctx.app_settings)


def _default_claude_backend(token: str) -> Any:
    # Lazy import so the adapter isn't pulled at module load.
    from augmentum.models.adapters.claude import ClaudeBackend
    return ClaudeBackend(api_key=token)


def _claude_api_invoke(backend: Any, model: str) -> ModelInvoke:
    from augmentum.models.base import InternalChatRequest, Message

    async def invoke(prompt: str) -> str:
        req = InternalChatRequest(
            model=model,
            messages=[Message(role="system", content=_CLASSIFIER_SYSTEM),
                      Message(role="user", content=prompt)],
            stream=False, temperature=0.0, max_tokens=384)
        try:
            resp = await backend.chat(req)
        except Exception as exc:  # noqa: BLE001 — degrade to unmapped
            log.warning("reshape_claude_invoke_failed", error=repr(exc))
            return ""
        return getattr(getattr(resp, "message", None), "content", "") or ""

    return invoke


async def _build_claude(ctx: SourceContext) -> ModelInvoke | None:
    """Reuse the coder-mode Claude login. API key → one-shot via the Anthropic
    backend. Subscription OAuth → skip (CLI-only; not faked) → caller falls back."""
    if not ctx.user_id or ctx.settings_store is None:
        return None
    from augmentum.coder.external.claude_auth import is_oauth_token

    loader = ctx.claude_token_loader
    if loader is None:
        from augmentum.coder.external.claude_token_store import load_token as loader

    token = await loader(ctx.settings_store, ctx.user_id)
    if not token:
        return None
    if is_oauth_token(token):
        # Subscription OAuth can't hit the x-api-key messages API; that's the
        # CLI driver's job (same one the code-edit path uses). Fall back honestly.
        log.info("reshape_claude_source_oauth_needs_cli", user_id=ctx.user_id)
        return None
    factory = ctx.claude_backend_factory or _default_claude_backend
    backend = factory(token)
    model = ctx.claude_model or _DEFAULT_CLAUDE_MODEL
    return _claude_api_invoke(backend, model)


def register_default_sources() -> None:
    """Register the built-in sources. Codex / other authenticable platforms slot
    in here later with the same shape (reuse their coder-side auth)."""
    register_source(ModelSource("augmentum", "Augmentum model list", _build_augmentum))
    register_source(ModelSource("claude", "Claude (your coder login)", _build_claude))
