"""BalancerBackend — a ModelBackend facade that fans a request across a load
balancer's members with automatic fallback.

Step 1 of the first-class load-balancer work (see
``docs/load-balancer-first-class-fallback.md``). Returning this facade from
``ProviderRegistry.resolve_backend_for_model`` for an ``lb/<name>`` model gives
EVERY call site (all chat modes, companion tasks, aux routes) fallback for free —
they keep calling ``backend.chat_stream(req)`` / ``backend.chat(req)`` unchanged.

Built (steps 1-3):
  * fallback loop over members (strategy pick, then priority fallback order)
  * streaming fallback is PRE-FIRST-TOKEN only (we cannot un-send streamed tokens)
  * empty completion (clean stream, zero content) counts as a FAILED attempt
  * retryable-error check (429/5xx/timeout/connection; auth/context are not)
  * per-member cooldown (LoadBalancer) so exhausted members are skipped, using
    the provider's parsed Retry-After/reset (step 2, via BackendError) or a
    blind exponential backoff
  * informative, classifiable exhaustion error (never a silent empty turn)
  * capability/passthrough methods proxied to a stable representative member

Deferred (step 4): rich user-facing fallback-visibility metadata (a "served via
fallback" badge) and a fully-formatted exhaustion toast. Also deferred: unifying
``_is_retryable`` with proxy/streaming.py's classifier into one definition.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from augmentum.models.backend_errors import retry_after_from_body
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    ModelBackend,
    ModelDetails,
    ModelInfo,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.load_balancer import LoadBalancer
    from augmentum.state.balancer_store import BalancerMember

log = get_logger(__name__)


# Substrings that mark an error as "the same request would fail identically on
# every member" — never worth burning the pool on. Everything else (429, 5xx,
# timeout, connection reset) is treated as retryable → fall to the next member.
# NOTE: deliberately minimal for step 1; step 2 replaces this with the shared
# classifier extracted from proxy/streaming.py so the facade and the SSE layer
# agree on exactly one definition.
_NON_RETRYABLE_SUBSTRINGS = (
    "context length",
    "context window",
    "maximum context",
    "authentication",
    "invalid api key",
    "unauthorized",
    "no mmproj",
    "vision projector",
)


def _is_retryable(exc: BaseException) -> bool:
    """True iff falling over to another member could plausibly succeed."""
    # Never retry client-cancellation / generator teardown.
    if isinstance(exc, asyncio.CancelledError | GeneratorExit):
        return False
    msg = str(exc).lower()
    return not any(s in msg for s in _NON_RETRYABLE_SUBSTRINGS)


def _member_cooldown_hint(exc: BaseException) -> float | None:
    """Seconds to cool the failing member for.

    Prefers the STRUCTURED reset the adapter parsed from response headers /
    body (``BackendError.retry_after`` — precise, e.g. OpenAI-compat
    ``Retry-After``); falls back to scraping a hint out of the error text
    (covers any exception that isn't a BackendError yet). None → the
    LoadBalancer applies its blind exponential backoff.
    """
    ra = getattr(exc, "retry_after", None)
    if isinstance(ra, int | float) and ra > 0:
        return float(ra)
    return retry_after_from_body(str(exc))


class BalancerBackend(ModelBackend):
    """Fans chat requests across a LoadBalancer's members with fallback.

    Stateless per call: each ``chat``/``chat_stream`` re-selects the primary via
    the balancer's strategy, then walks the priority-ordered remainder on
    retryable failure. Capability queries use a fixed representative member so
    they don't perturb round-robin state.
    """

    # A balancer spans heterogeneous members; the datetime-placement decision
    # (modes/base.py::_ensure_datetime) reads this. Report the representative
    # member's value — refined per-attempt in a later step if members diverge.
    @property
    def supports_mid_conversation_system(self) -> bool:  # type: ignore[override]
        b = self._representative_backend()
        return bool(getattr(b, "supports_mid_conversation_system", False)) if b else False

    def __init__(self, lb: LoadBalancer, registry: object) -> None:
        self._lb = lb
        self._registry = registry

    # ---- member plumbing -------------------------------------------------

    def _backend_for(self, member: BalancerMember) -> ModelBackend | None:
        return getattr(self._registry, "_backends", {}).get(member.backend_key)

    def _representative_backend(self) -> ModelBackend | None:
        """A stable member backend for capability queries (no RR side effect)."""
        for m in self._lb.members:
            b = self._backend_for(m)
            if b is not None:
                return b
        return None

    def _ordered_members(self) -> list[BalancerMember]:
        """Primary (strategy pick) first, then priority fallback order.

        When fallback is disabled the list is just the primary — behavior
        identical to the pre-facade single-shot path.
        """
        primary = self._lb.select()
        if not self._lb.config.fallback_enabled:
            return [primary]
        return [primary, *self._lb.fallback_order(primary)]

    @staticmethod
    def _clone_with_model(request: InternalChatRequest, model: str) -> InternalChatRequest:
        """Shallow request clone pinned to ``model``; never mutate the caller's
        object, and give each attempt its own messages LIST so an in-place
        normalizer on a failed attempt can't corrupt the retry."""
        req = copy.copy(request)
        req.model = model
        if getattr(request, "messages", None) is not None:
            req.messages = list(request.messages)
        return req

    # ---- the fallback loops ---------------------------------------------

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        members = self._ordered_members()
        last_exc: BaseException | None = None
        for i, member in enumerate(members):
            backend = self._backend_for(member)
            if backend is None:
                log.warning(
                    "balancer_member_backend_missing",
                    balancer=self._lb.config.name, backend=member.backend_key,
                )
                continue
            req = self._clone_with_model(request, member.model_name)
            # Track VISIBLE CONTENT, not just "any chunk". A member can end its
            # stream cleanly having emitted only metadata/finish chunks and zero
            # content — Gemini does this on a safety/quota block, and a mid-stream
            # error swallowed by an adapter looks the same. That is a FAILED
            # attempt, not a success: returning here would be a silent empty
            # turn (observed 2026-07-30: gemini-2.5-flash, ttft=0, gen_tokens=0).
            got_content = False
            try:
                async for chunk in backend.chat_stream(req):
                    if getattr(chunk, "content_delta", None):
                        if not got_content and i > 0:
                            log.info(
                                "balancer_fallback_served",
                                balancer=self._lb.config.name,
                                selected=members[0].model_name,
                                served=member.model_name,
                                backend=member.backend_key,
                                attempt=i + 1,
                            )
                        got_content = True
                    yield chunk
            except BaseException as exc:  # noqa: BLE001 — classified below
                # Once visible content has streamed we are committed to this
                # member — a mid-stream failure surfaces (can't un-send output).
                if got_content or not _is_retryable(exc):
                    raise
                last_exc = exc
                self._lb.note_failure(member, _member_cooldown_hint(exc))
                log.warning(
                    "balancer_member_failed",
                    balancer=self._lb.config.name,
                    member=member.model_name, backend=member.backend_key,
                    error=repr(exc), will_fall_over=(i + 1 < len(members)),
                    cooldown_s=round(self._lb.cooldown_remaining(member), 1),
                )
                continue
            if got_content:
                self._lb.note_success(member)
                return
            # Clean stream, but zero content — treat as a failed attempt and
            # fall over. Only empty/metadata chunks (if any) were yielded, which
            # render as nothing, so switching members is safe. Cool it (blind
            # default) — an empty-returning member is usually blocked/quota'd.
            last_exc = RuntimeError(
                f"member '{member.model_name}' returned an empty response"
            )
            self._lb.note_failure(member, None)
            log.warning(
                "balancer_member_empty",
                balancer=self._lb.config.name, member=member.model_name,
                backend=member.backend_key, will_fall_over=(i + 1 < len(members)),
                cooldown_s=round(self._lb.cooldown_remaining(member), 1),
            )
        raise self._exhausted_error(members, last_exc)

    def _exhausted_error(
        self, members: list[BalancerMember], last_exc: BaseException | None
    ) -> Exception:
        """A visible, classifiable error when every member failed/emptied.

        Preserves the last cause's text so ``streaming.py::_classify_backend_error``
        still routes 429/503 to the right friendly toast, and so the user sees a
        real reason instead of a silent empty turn.
        """
        detail = str(last_exc) if last_exc else "no member produced a response"
        return RuntimeError(
            f"Load balancer '{self._lb.config.name}': all {len(members)} "
            f"member(s) failed or returned empty. Last: {detail}"
        )

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        members = self._ordered_members()
        last_exc: BaseException | None = None
        for i, member in enumerate(members):
            backend = self._backend_for(member)
            if backend is None:
                continue
            req = self._clone_with_model(request, member.model_name)
            try:
                resp = await backend.chat(req)
                self._lb.note_success(member)
                if i > 0:
                    log.info(
                        "balancer_fallback_served",
                        balancer=self._lb.config.name,
                        selected=members[0].model_name,
                        served=member.model_name, attempt=i + 1,
                    )
                return resp
            except BaseException as exc:  # noqa: BLE001
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                self._lb.note_failure(member, _member_cooldown_hint(exc))
                log.warning(
                    "balancer_member_failed",
                    balancer=self._lb.config.name,
                    member=member.model_name, backend=member.backend_key,
                    error=repr(exc), will_fall_over=(i + 1 < len(members)),
                )
                continue
        raise self._exhausted_error(members, last_exc)

    # ---- capability / passthrough (delegate to representative member) ----

    def pre_stream_validate(self, request: InternalChatRequest) -> None:
        b = self._representative_backend()
        if b is not None:
            b.pre_stream_validate(request)

    def is_local_engine(self) -> bool:
        b = self._representative_backend()
        return b.is_local_engine() if b else False

    def supported_sampler_params(self, model: str = "") -> set[str]:
        b = self._representative_backend()
        return b.supported_sampler_params(model) if b else set(self.SAMPLER_PARAM_KEYS)

    def is_vision_paired(self, model: str = "") -> bool:
        b = self._representative_backend()
        return b.is_vision_paired(model) if b else True

    async def list_models(self) -> list[ModelInfo]:
        b = self._representative_backend()
        return await b.list_models() if b else []

    async def show_model(self, name: str) -> ModelDetails:
        b = self._representative_backend()
        if b is None:
            raise ValueError(f"Balancer '{self._lb.config.name}' has no members")
        return await b.show_model(name)

    async def get_context_length(self, model: str) -> int:
        b = self._representative_backend()
        return await b.get_context_length(model) if b else 0
