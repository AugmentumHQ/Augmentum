"""Architect translation layer — reshape raw user intent into well-
formed tool input.

Where ``inference`` fills missing defaults from observation history,
``translation`` runs a final transform pass that takes the post-
inference args and makes them ready for the underlying tool. The
canonical example: image generation. The intent matcher captures
``prompt="a dog"`` from "Hey there, generate an image of a dog please".
"a dog" is a terrible image-model prompt — three tokens with no
subject detail, no lighting, no composition. The translator gets
one LLM turn to expand it into a scene-rich description ("a
golden retriever puppy in soft sunlight, professional photography,
shallow depth of field") before the image surface receives the
generate event.

Translators are per-primitive, opt-in via ``Action.arg_transformer``.
They MUST be tolerant of failure — a slow / errored translator must
not block dispatch; the dispatcher falls back to the untransformed
args.

This module exposes generic helpers + the specific image translator.
Adding a new translator: write an async ``(args, session, runtime) ->
args`` function, register it via ``arg_transformer=...`` on the
primitive's ``register_action(...)`` call.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.intent.action import SessionContext

log = get_logger(__name__)


# Image prompt expansion — the core translator for image.generate.
#
# System prompt is deliberately tight: the model gets ONE job (expand
# the user's casual ask into a paintable scene description). No
# meta-commentary, no caveats, no "I am an AI" disclaimers — just
# the expanded prompt. Returns plain text, single paragraph.
_IMAGE_PROMPT_EXPANSION_SYSTEM = (
    "You expand a casual user request into a vivid image-generation "
    "prompt. The user has spoken a short phrase aloud and a downstream "
    "image model needs richer detail to produce a good result.\n\n"
    "Take the request and produce ONE paragraph (40-80 words) that "
    "describes the scene visually: subject + setting + lighting + mood "
    "+ composition. Match the style the user seems to want — if they "
    "say 'draw a cat' assume natural/photographic, if they say "
    "'generate art of a dragon' assume painterly. Stay grounded in "
    "what they asked; do not invent specific named people or places "
    "they didn't mention.\n\n"
    "Output ONLY the expanded prompt. No preamble, no quotes, no "
    "explanation. Just the description."
)


def _build_image_expansion_user_prompt(
    raw: str,
    *,
    style_hints: dict[str, Any] | None = None,
) -> str:
    """Compose the per-call user prompt with optional style anchors
    from the user's history (last preset, recent successful prompts).
    """
    parts: list[str] = [f"User said: {raw.strip()!r}"]
    if style_hints:
        last_preset = style_hints.get("last_preset") or ""
        last_prompt = style_hints.get("last_prompt") or ""
        if last_preset:
            parts.append(f"Their usual preset: {last_preset}")
        if last_prompt:
            parts.append(
                "Their most recent successful prompt (for style continuity, "
                f"not content): {last_prompt[:200]!r}"
            )
    parts.append("\nExpanded prompt:")
    return "\n".join(parts)


def _parse_expanded_prompt(raw: str) -> str:
    """Normalize the model's response.

    Strip wrapping quotes, leading "Expanded prompt:" / "Here is..."
    preambles, trailing whitespace, and code fences. Returns the
    cleanest scene description we can extract.
    """
    if not raw:
        return ""
    text = raw.strip()
    # Strip common preamble phrases
    for prefix in (
        "expanded prompt:", "here is the expanded prompt:",
        "here's the expanded prompt:", "prompt:", "scene:",
        "here is", "here's",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].lstrip(" :\n")
    # Strip wrapping quotes
    for q in ('"', "'", "`"):
        if text.startswith(q) and text.endswith(q) and len(text) > 2:
            text = text[1:-1]
    # Strip code fences
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text[3:]
        text = text.strip()
    return text.strip()


async def expand_image_prompt(
    raw_prompt: str,
    *,
    app_state: Any,
    user_id: str = "",
    session_id: str = "",
    style_hints: dict[str, Any] | None = None,
    timeout_s: float = 4.0,
    model_override: str = "",
) -> str:
    """Run the LLM expansion. Returns the expanded prompt, or the raw
    prompt unchanged if anything fails.

    The expansion is bounded by ``timeout_s`` so a slow backend never
    blocks the image dispatch chain. Image generation itself takes
    5-30s; a 4s upper bound on expansion is comfortably fast.
    """
    raw = (raw_prompt or "").strip()
    if not raw:
        return raw

    # Don't bother expanding prompts that are already long / detailed.
    # 60+ chars roughly means the user already gave a richer phrase
    # ("a cyberpunk samurai in neon-lit rain"); expansion adds risk.
    if len(raw) >= 60:
        log.debug("image_prompt_expansion_skipped_already_rich", chars=len(raw))
        return raw

    registry = getattr(app_state, "provider_registry", None) if app_state else None
    if registry is None:
        return raw

    try:
        from augmentum.models.base import InternalChatRequest, Message
    except Exception:
        return raw

    try:
        backend, resolved_model = await registry.resolve_backend_with_fabric(
            model_override or "",
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as exc:
        log.debug("image_prompt_expansion_resolve_failed", error=str(exc)[:160])
        return raw

    if backend is None:
        return raw

    req = InternalChatRequest(
        model=resolved_model or model_override or "",
        messages=[
            Message(role="system", content=_IMAGE_PROMPT_EXPANSION_SYSTEM),
            Message(
                role="user",
                content=_build_image_expansion_user_prompt(
                    raw, style_hints=style_hints,
                ),
            ),
        ],
        stream=False,
        temperature=0.7,  # some variety in scene composition
        max_tokens=180,    # ~80 words + slack
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.info("image_prompt_expansion_timeout", model=resolved_model)
        return raw
    except Exception as exc:  # noqa: BLE001 — degrade safely
        log.warning(
            "image_prompt_expansion_error",
            model=resolved_model, error=str(exc)[:200],
        )
        return raw

    content = getattr(getattr(resp, "message", None), "content", "") or ""
    expanded = _parse_expanded_prompt(content)
    if not expanded or len(expanded) < len(raw):
        # Model failed to expand or returned something shorter than
        # the original — usage hint or garbage. Fall back.
        log.info(
            "image_prompt_expansion_short_response",
            raw_len=len(raw), expanded_len=len(expanded),
        )
        return raw

    log.info(
        "image_prompt_expanded",
        model=resolved_model, raw_len=len(raw), expanded_len=len(expanded),
        raw_preview=raw[:60], expanded_preview=expanded[:100],
    )
    return expanded


async def translate_image_args(
    args: dict[str, Any],
    session: "SessionContext",
    runtime: Any,
) -> dict[str, Any]:
    """Action.arg_transformer for image.generate_with_defaults.

    Expands the user's raw prompt into a scene-rich description and
    records both the original + expanded form. The handler emits the
    expanded prompt in the surface payload; the spoken acknowledgment
    references the user's original phrasing so the UX feels natural
    ("Generating a cat" not "Generating: golden retriever puppy in
    soft sunlight…").
    """
    if not getattr(settings, "companion_image_prompt_expansion_enabled", True):
        return args

    raw = (args.get("prompt") or "").strip()
    if not raw:
        return args

    # SessionContext carries the FastAPI app_state when the call site
    # has it (architect dispatch does). The runtime is the secondary
    # path used by some background invokers.
    app_state = getattr(session, "app_state", None)
    if app_state is None and runtime is not None:
        app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return args

    style_hints = {
        "last_preset": args.get("preset") or "",
        # ``inferred_from_image_id`` was set by the image inferrer when
        # we pulled from history; we don't have the prompt directly
        # here but the preset is the main style anchor.
    }

    expanded = await expand_image_prompt(
        raw,
        app_state=app_state,
        user_id=session.user_id,
        session_id=session.session_id,
        style_hints=style_hints,
        timeout_s=float(
            getattr(settings, "companion_image_expansion_timeout_ms", 4000)
        ) / 1000.0,
    )

    if expanded and expanded != raw:
        # Stash both forms — the handler can choose which to speak vs.
        # which to send to the model.
        args["prompt_raw"] = raw
        args["prompt"] = expanded

    return args
