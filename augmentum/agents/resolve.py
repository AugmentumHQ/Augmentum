"""Multi-provider model resolution with fallback chains.

A role's ``model`` block looks like::

    model:
      preferred: "claude-sonnet-4-6@anthropic"
      fallbacks:
        - "qwen3-32b-instruct@local"
        - "gpt-4o-mini@openai"
        - "llama-3-70b@ollama:tower"   # fabric peer

This module turns a preferred + fallback list into a concrete
``ResolvedModel(backend, model_id, spec_string)`` by walking the chain
and using the first one that successfully resolves. Resolution itself
delegates to ``ProviderRegistry.resolve_backend_with_fabric`` so the
``@<backend>`` and ``@fabric:<peer>`` suffix handling is unified across
the codebase.

Spec grammar accepted at parse time:

* ``model_id``                       — registry default resolution
* ``model_id@provider``              — explicit backend (existing @-suffix)
* ``model_id@fabric:peer_id``        — pinned fabric peer
* ``model_id@provider:peer_id``      — sugar that rewrites to fabric pin
  with the provider hint preserved upstream; today routes through the
  same fabric pin path (peers carry their own backend identity).
* ``model_id@local`` or ``...@host`` — treated as default-backend.

Failure semantics: ``resolve_subagent_model`` walks candidates in order,
catching ``ModelUnavailableError`` and falling through. If every
candidate fails, raises ``SubagentModelUnavailableError`` with the full
list of tried specs + their failure reasons so the calling tool surface
can render an actionable message.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.models.base import ModelBackend
from augmentum.models.provider_registry import (
    ModelUnavailableError,
    ProviderRegistry,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedModel:
    """Concrete (backend, model_id) pair after walking the fallback chain."""

    backend: ModelBackend
    """The resolved backend object, ready for ``backend.chat(...)``."""

    model_id: str
    """The cleaned model id to pass to ``backend.chat``. Strips any
    ``@provider`` / ``@fabric:peer`` suffix from the spec."""

    spec: str
    """The original spec string that won the chain (e.g.
    ``"claude-sonnet-4-6@anthropic"``). Recorded for audit / UI."""


class SubagentModelUnavailableError(Exception):
    """No candidate in the preferred + fallback chain resolved."""

    def __init__(self, role: str, attempts: list[tuple[str, str]]) -> None:
        self.role = role
        self.attempts = attempts
        msg = (
            f"role {role!r} has no available model. Tried: "
            + "; ".join(f"{spec} -> {reason}" for spec, reason in attempts)
        )
        super().__init__(msg)


def parse_model_spec(spec: str) -> str:
    """Normalize a spec string for ``resolve_backend_with_fabric``.

    Right now this is a passthrough — ``resolve_backend_with_fabric`` already
    handles the ``@<backend>`` and ``@fabric:<peer>`` suffix forms, and the
    sugar form ``@provider:peer`` collapses to ``@fabric:peer`` at routing
    time (the provider hint is informational only — peers advertise their
    own backend identity in handshake).

    Kept as a function so the dispatcher has one place to add normalization
    (e.g. user-defined provider aliases) without touching the registry.
    """
    return (spec or "").strip()


async def resolve_subagent_model(
    *,
    role: str,
    preferred: str,
    fallbacks: list[str] | None,
    registry: ProviderRegistry,
    user_id: str = "",
    session_id: str = "",
    override: str = "",
) -> ResolvedModel:
    """Walk the fallback chain, returning the first model that resolves.

    Order: ``override`` (if non-empty, single attempt) → ``preferred`` →
    each entry in ``fallbacks``. The first that returns a non-null backend
    wins; the rest are skipped.

    Raises :class:`SubagentModelUnavailableError` if every candidate fails,
    with the per-spec failure reasons attached.
    """
    candidates: list[str] = []
    if override:
        # Explicit override from a tool call bypasses the chain entirely.
        candidates = [override]
    else:
        # ``preferred = ""`` is the "use the registry's default backend's
        # first model" sentinel (the resolver expands "" via
        # primary_chat_model or the default backend). Pass it through
        # rather than bailing — that's the documented contract for
        # built-in roles that opt out of pinning a specific model.
        candidates.append(preferred or "")
        for f in fallbacks or []:
            f = (f or "").strip()
            if f and f not in candidates:
                candidates.append(f)

    attempts: list[tuple[str, str]] = []
    for spec in candidates:
        # NB: empty ``spec`` is a valid "use registry default" sentinel
        # so resolve_backend_with_fabric handles it via the
        # primary_chat_model / default-backend-first-model expansion.
        normalized = parse_model_spec(spec)
        try:
            backend, clean_model = await registry.resolve_backend_with_fabric(
                normalized,
                user_id=user_id,
                session_id=session_id,
            )
        except ModelUnavailableError as exc:
            attempts.append((spec, str(exc)[:160]))
            log.info(
                "subagent_model_unavailable",
                role=role,
                spec=spec,
                reason=str(exc)[:160],
            )
            continue
        except Exception as exc:
            attempts.append((spec, f"{type(exc).__name__}: {exc}"[:160]))
            log.warning(
                "subagent_model_resolution_error",
                role=role,
                spec=spec,
                exc_info=True,
            )
            continue

        if backend is None:
            attempts.append((spec, "resolved to no backend"))
            continue

        return ResolvedModel(backend=backend, model_id=clean_model, spec=spec)

    raise SubagentModelUnavailableError(role, attempts)
