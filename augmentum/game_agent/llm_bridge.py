"""LLM bridge between :mod:`augmentum.game_agent` and Augmentum's
:class:`augmentum.models.provider_registry.ProviderRegistry`.

The game-agent engine is provider-agnostic; it only requires a
:data:`SlowPathLLM`-compatible async callable. This module supplies
one that delegates to Augmentum's registry (Augmentum Engine, Ollama,
OpenAI, Claude, Gemini, llama.cpp).

Model selection is per-lane and role-based. The planning lane resolves
through the ``primary`` role, the fast lane through ``classifier``;
``game_agent_planner_model`` / ``game_agent_fast_model`` override each
when set. Neither lane resolves an empty model name any more — that
path ends at "first model hosted by the default backend", which picked
the game agent's brain by dict iteration order.

Frame attachment
----------------
For backends whose default model is vision-capable
(:func:`augmentum.models.base.is_vision_model_name`), an optional
PNG frame is base64-encoded and attached as the user message's
``images`` list. Non-vision backends silently drop the frame -- the
game-agent prompt is designed to function from log entries alone, so
this degrades gracefully.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

import structlog

from augmentum.config import settings
from augmentum.game_agent.agent import SlowPathLLM
from augmentum.models.base import (
    InternalChatRequest,
    Message,
    is_vision_model_name,
)

if TYPE_CHECKING:
    from augmentum.models.provider_registry import ProviderRegistry

log = structlog.get_logger(__name__)


# Default token budget for the slow-path reply. Plans are strict JSON
# bounded by the PlanPayload schema -- 1024 is plenty for the
# observations + state + actions fields and leaves headroom for
# verbose models that fill the bound.
_DEFAULT_MAX_TOKENS = 1024

# Temperature for the slow-path reply. The prompt is strict-JSON;
# higher temperatures make models prone to commentary or fenced
# markdown. 0.2 is hot enough for varied planning, cold enough that
# the schema holds.
_DEFAULT_TEMPERATURE = 0.2


def make_game_agent_llm(
    registry_or_getter: ProviderRegistry | Callable[[], ProviderRegistry],
    *,
    pinned_model: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> SlowPathLLM:
    """Build a :data:`SlowPathLLM` callable backed by a provider registry.

    Use when:
    - Wiring ``app.state.game_agent_llm`` in
      :func:`augmentum.proxy.server.create_app` so the game-agent
      route layer has a working LLM.

    Expects:
    - ``registry_or_getter`` is either the live
      :class:`ProviderRegistry` or a zero-arg callable that returns
      it. The callable form lets the bridge be wired *before* the
      registry exists (e.g. inside ``create_app`` while the registry
      is built by a startup event), with resolution deferred to the
      first slow-path call.

    Returns:
    - An async callable ``(prompt, frame) -> raw_model_output`` that the
      :class:`augmentum.game_agent.SlowPathAgent` can call directly.
      Errors bubble up as exceptions; the orchestrator's slow-path
      loop catches them and writes ``agent_error`` log entries.

    @param pinned_model:
        If set, force the bridge to use this model name (and whichever
        backend resolves for it) instead of the registry's default.
        Useful when the operator wants game-agent calls to go to a
        specific cheaper/faster model than the rest of the app.
    @param max_tokens:
        Upper bound for the model's reply. The :class:`PlanPayload`
        schema caps the meaningful output well below most model
        ceilings, but we still set a budget so misbehaving models do
        not run away.
    @param temperature:
        Sampling temperature. Lower is safer because the prompt is
        strict-JSON.
    """

    def _resolve_registry() -> ProviderRegistry:
        if callable(registry_or_getter):
            return registry_or_getter()
        return registry_or_getter

    async def _call(prompt: str, frames: Sequence[bytes]) -> str:
        # Resolve registry (possibly lazy), then a backend + clean
        # model name. Resolution order for the PLANNER model:
        #   1. explicit ``pinned_model`` arg (caller override)
        #   2. ``settings.game_agent_planner_model`` — the user-chosen
        #      planning model, read LIVE per call so swapping the
        #      planner is a settings write, no restart.
        #   3. the PRIMARY role.
        #
        # Step 3 used to be ``resolve_backend_with_fabric("")``, whose
        # bottom rung is "first model hosted by the default backend" —
        # i.e. dict iteration order. That is a silent auto-pick, and it
        # is how an unpinned planner ended up running on a 2B classifier
        # model: not because anyone chose it, but because it happened to
        # be first in the map. Going through the role chain instead means
        # an empty pin resolves to the model the user is actually
        # chatting with, and the choice stays theirs either way.
        registry = _resolve_registry()
        planner_pin = pinned_model or str(
            getattr(settings, "game_agent_planner_model", "") or ""
        )
        backend, model_name = await registry.resolve_model_for_role(
            "primary", override=planner_pin, settings=settings,
        )

        # Vision capability — combine three signals so we don't silently
        # drop frames for models the name-pattern doesn't recognize:
        #
        # 1. ``info.vision`` from backend.list_models — the runtime truth
        #    for llama-cpp (set when an mmproj projector is paired) and
        #    for cloud providers that declare vision in /v1/models.
        # 2. ``is_vision_model_name`` — name pattern fallback for backends
        #    that don't populate the flag at listing time.
        #
        # Either signal is sufficient.
        accepts_vision = is_vision_model_name(model_name)
        if not accepts_vision and frames:
            try:
                listed = await backend.list_models()
                for info in listed:
                    if info.name == model_name and info.vision:
                        accepts_vision = True
                        break
            except Exception as exc:  # noqa: BLE001
                # Listing failure shouldn't kill the turn; fall back to
                # the name-pattern decision.
                log.debug(
                    "game_agent.llm.list_models_failed",
                    backend=type(backend).__name__,
                    error=str(exc),
                )

        images: list[str] | None = None
        if frames and accepts_vision:
            # llama-server (and OpenAI-shape backends generally) reject
            # bare base64 strings on image_url.url with "Invalid url
            # value". The full data URL form is the universally-accepted
            # shape; chat-side image plumbing already does this via
            # resolve_chat_image_urls. Frames are always PNG from the
            # iframe canvas snapshot. We attach every frame in the
            # caller's order so the prompt's "oldest first" contract
            # holds end-to-end.
            images = [
                f"data:image/png;base64,{base64.b64encode(f).decode('ascii')}"
                for f in frames
            ]
        elif frames:
            # Frames were offered but the resolved model can't use them.
            # Drop silently AND patch the prompt so the model isn't told
            # FRAMES are attached when nothing is -- otherwise it
            # hallucinates "can't parse pixels" or fabricates content.
            # Replace whichever variant of the FRAMES line the prompt
            # builder emitted; the count is encoded in the prefix.
            for old in (
                "FRAMES: <1 attached>",
                f"FRAMES (oldest -> newest, ~1s apart, read in order): "
                f"<{len(frames)} attached>",
            ):
                if old in prompt:
                    prompt = prompt.replace(
                        old,
                        "FRAMES: <not provided this turn (active model is text-only)>",
                    )
                    break
            log.warning(
                "game_agent.llm.frame_dropped_non_vision_model",
                model=model_name,
                n_frames=len(frames),
            )

        # Thinking + budget are read live from settings so they're tunable
        # without a restart. Small local vision models (the Gemma-4 E2B
        # classifier) plan markedly better WITH reasoning on -- it stops
        # them fixating on one button -- but only if the budget is large
        # enough that the chain-of-thought AND the JSON both fit (a tight
        # cap truncates the reasoning and the answer comes back empty).
        # Reasoning models also want non-greedy sampling, so we lift the
        # temperature when thinking is on. ``think`` is a no-op on models
        # that don't consume enable_thinking, so this is safe globally.
        think = bool(getattr(settings, "game_agent_thinking_enabled", True))
        call_max_tokens = int(getattr(settings, "game_agent_max_tokens", 0)) or max_tokens
        call_temp = 0.6 if think else temperature
        # Grammar-lock the plan to the PlanPayload schema — but ONLY
        # when thinking is off. With thinking on, the grammar forbids
        # the chat template's reasoning channel tokens and llama-server
        # 500s every call ("output does not match the expected
        # peg-gemma4 format") — live-observed run 20, where it silenced
        # the planner for an entire session. Thinking planners keep the
        # lenient parser as their net; non-thinking ones get the
        # structural guarantee.
        raw_options = None
        if not think:
            from augmentum.game_agent.schema import PlanPayload
            raw_options = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "game_plan",
                        "schema": PlanPayload.model_json_schema(),
                    },
                }
            }
        request = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="user", content=prompt, images=images),
            ],
            stream=False,
            max_tokens=call_max_tokens,
            temperature=call_temp,
            think=think,
            raw_options=raw_options,
        )
        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(user_id="", session_id="", mode="game")
        try:
            response = await backend.chat(request)
        except Exception:
            end_capture(_cap_ctx, _cap_tok, error="game_agent_error")
            raise
        end_capture(_cap_ctx, _cap_tok)
        # backend.chat returns InternalChatResponse; the agent prompt
        # asks for strict JSON in the assistant message body.
        return response.message.content or ""

    return _call


def make_game_agent_chat_llm(
    registry_or_getter: ProviderRegistry | Callable[[], ProviderRegistry],
    *,
    pinned_model: str | None = None,
    temperature: float = 0.35,
) -> Callable[[Sequence[dict]], Awaitable[dict]]:
    """Build the multi-message ("call mode") sibling of :func:`make_game_agent_llm`.

    Use when:
    - The orchestrator's fast-turn loop needs a rolling chat window
      (system prompt + user/assistant exchanges, frames attached per
      user message) instead of one flat prompt string. The append-only
      message shape is what keeps llama-server's KV prefix cache hot
      across turns — the entire point of the fast path.

    Expects:
    - ``messages``: sequence of ``{"role": str, "content": str,
      "images": Sequence[bytes] | None}`` dicts, oldest first. Images
      are raw PNG bytes; encoding to data URLs happens here.

    Returns:
    - An async callable ``(messages) -> {"text", "latency_ms",
      "completion_tokens", "cached_tokens", "tok_s"}``. Timing fields
      are 0/None when the backend doesn't report them; ``latency_ms``
      is always populated (wall clock around the call).
    """

    def _resolve_registry() -> ProviderRegistry:
        if callable(registry_or_getter):
            return registry_or_getter()
        return registry_or_getter

    async def _call(messages: Sequence[dict], options: dict | None = None) -> dict:
        import time

        registry = _resolve_registry()
        # Fast-lane pin: survive other tenants flipping the shared
        # default model (which broke a live run with 400 vision errors
        # when the default became a text-only model). Read LIVE so it
        # is switchable without restart.
        #
        # Empty falls through to the CLASSIFIER role, not to "first model
        # on the default backend". The fast lane is a ~30-token micro
        # contract that needs vision and low latency, which is precisely
        # what Slot C is (resident, vision-capable when its model carries
        # an mmproj) — and the role chain then cascades classifier →
        # sidecar → utility → primary, so it still lands somewhere real
        # when no classifier is configured.
        fast_pin = pinned_model or str(
            getattr(settings, "game_agent_fast_model", "") or ""
        )
        backend, model_name = await registry.resolve_model_for_role(
            "classifier", override=fast_pin, settings=settings,
        )

        accepts_vision = is_vision_model_name(model_name)
        if not accepts_vision and any(m.get("images") for m in messages):
            try:
                listed = await backend.list_models()
                for info in listed:
                    if info.name == model_name and info.vision:
                        accepts_vision = True
                        break
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "game_agent.chat_llm.list_models_failed",
                    backend=type(backend).__name__,
                    error=str(exc),
                )

        wire: list[Message] = []
        for m in messages:
            frames = m.get("images") or ()
            images: list[str] | None = None
            if frames and accepts_vision:
                images = [
                    f"data:image/png;base64,{base64.b64encode(f).decode('ascii')}"
                    for f in frames
                ]
            wire.append(
                Message(
                    role=str(m.get("role") or "user"),
                    content=str(m.get("content") or ""),
                    images=images,
                )
            )

        # Fast turns never think — the micro contract is ~30 tokens and
        # the whole point is latency. The budget is a live setting so it
        # can be tuned mid-experiment without a restart.
        max_tokens = int(getattr(settings, "game_agent_fast_max_tokens", 0)) or 192
        # Per-call decoding constraints. The OpenAI ``response_format``
        # json_schema shape is what llama-server's CHAT endpoint
        # actually compiles to a grammar — a bare top-level
        # ``json_schema`` field is silently ignored there (live-verified
        # 2026-07-02: prose prompt + response_format → enum-locked JSON;
        # prose prompt + bare json_schema → free prose). Rides
        # raw_options through the llama_cpp passthrough; other backends
        # ignore it and the caller's lenient parser stays the safety
        # net. Per-call (not baked in) because this same callable
        # serves the scene narrator, whose output is prose.
        raw_options: dict | None = None
        if options and options.get("json_schema") is not None:
            raw_options = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fast_plan",
                        "schema": options["json_schema"],
                    },
                }
            }
        request = InternalChatRequest(
            model=model_name,
            messages=wire,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            think=False,
            raw_options=raw_options,
        )
        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(user_id="", session_id="", mode="game")
        t0 = time.monotonic()
        try:
            response = await backend.chat(request)
        except Exception:
            end_capture(_cap_ctx, _cap_tok, error="game_agent_error")
            raise
        end_capture(_cap_ctx, _cap_tok)
        latency_ms = (time.monotonic() - t0) * 1000.0

        usage = getattr(response, "usage", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "cache_hit_tokens", 0) or 0)
        eval_ms = float(getattr(usage, "eval_duration_ms", 0.0) or 0.0)
        tok_s: float | None = None
        if completion_tokens and eval_ms > 0:
            tok_s = round(completion_tokens / (eval_ms / 1000.0), 1)
        return {
            "text": response.message.content or "",
            "latency_ms": round(latency_ms, 1),
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "tok_s": tok_s,
        }

    return _call


__all__ = ["make_game_agent_llm", "make_game_agent_chat_llm"]
