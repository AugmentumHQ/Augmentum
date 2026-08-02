"""Model-tier resolution for Becca's pipeline (Lane 1 §6).

Becca speaks on the user's primary model — when they upgrade their chat
model, she upgrades too. That's the dogfooding promise. Utility tier
handles tool-result wraps; classifier tier handles intent triage and
post-turn labeling. Subagents/channels run at their existing tiers and
are not touched here.

Thin facade over ``provider_registry.resolve_model_for_role`` so the
voice pipeline doesn't reach across packages to do model selection.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


async def _resolve(runtime: CompanionRuntime, role: str) -> tuple[Any, str]:
    """Return (backend, model_name) for ``role``. ``role`` is one of
    "primary_chat", "utility", "classifier".

    Raises ``RuntimeError`` if the runtime has no app_state — the
    voice pipeline cannot speak without backend access.
    """
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        raise RuntimeError(f"tiers._resolve({role}): runtime has no app_state")
    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        raise RuntimeError(f"tiers._resolve({role}): app_state has no provider_registry")
    # resolve_model_for_role returns (backend, clean_model) per the existing
    # contract used by analytical / coder / agentic.
    resolved = await registry.resolve_model_for_role(
        role=role, override="", settings=settings,
    )
    if not resolved:
        raise RuntimeError(f"tiers._resolve({role}): no backend resolved")
    return resolved


async def primary(runtime: CompanionRuntime) -> tuple[Any, str]:
    """The model Becca speaks with — the user's primary chat model by
    default, or the utility-tier model when ``companion_speak_tier`` is
    pinned to "utility" (low-latency small model kept separate from a
    heavier primary chat model). Utility passthrough-defaults to primary
    when no distinct utility model is configured, so the pin is safe."""
    speak_role = "primary_chat"
    if (getattr(settings, "companion_speak_tier", "primary") or "primary") == "utility":
        speak_role = "utility"
    backend, model_name = await _resolve(runtime, speak_role)
    # Stamp for telemetry — becca_act_gap and the consistency harness
    # (scripts/companion_eval.py) compare behavior PER MODEL; this is
    # the one choke-point every primary-tier caller passes through.
    # Suppress narrowly: frozen test doubles reject attr assignment.
    with contextlib.suppress(AttributeError, TypeError):
        runtime.last_primary_model = model_name
    return backend, model_name


async def utility(runtime: CompanionRuntime) -> tuple[Any, str]:
    """Short rephrasing, tool-result synth wraps, post-turn labeler input."""
    return await _resolve(runtime, "utility")


async def classifier(runtime: CompanionRuntime) -> tuple[Any, str]:
    """Triage, labeling, the cheap structured-output stuff."""
    return await _resolve(runtime, "classifier")


__all__ = ["primary", "utility", "classifier"]
