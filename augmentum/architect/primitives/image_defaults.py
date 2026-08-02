"""image.generate_with_defaults — architect-callable image gen with
context-inferred defaults.

When the user says "generate a cat" or "make me an image of a sunset
on Mars", this action:

  1. Matches via Tier 1 patterns (or Tier 3 LLM tool call).
  2. Runs ``_infer_image_args`` to pull the user's last image
     generation and fill missing model + settings (steps, cfg_scale,
     width, height, preset, loras, negative_prompt).
  3. Emits the ``image.generate`` surface event so the frontend's
     image module starts the generation in its own surface.
  4. Speaks a short confirmation.

If the user has no image history yet, the inferrer fills in nothing
beyond the prompt — the frontend's image module then applies its own
default model + settings as it would for any fresh user.

This primitive is intentionally a thin handoff: it does NOT call the
image pipeline directly. The image surface owns generation; the
architect just supplies a context-aware request.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)

# Tier-3-only: LLM picks based on intent + context. The open-slot
# regex/templates ("generate X", "draw X") would eat the entire
# remainder of an utterance — exactly the failure mode that hit
# search.web. See [[no-regex-switchboard]].
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _arg_transformer(args: dict[str, Any], session: SessionContext, runtime: Any) -> dict[str, Any]:
    """Thin wrapper that delegates to the translation module. Kept
    local so register_action's ``arg_transformer`` lookup doesn't
    create an import cycle at decorator time.
    """
    from augmentum.architect.translation import translate_image_args
    return await translate_image_args(args, session, runtime)


async def _conn_from_runtime(runtime: Any) -> Any:
    """Mirror of grove_match._conn_from_runtime — pull the aiosqlite
    connection from the companion runtime. Kept local so the two
    primitives don't share an undeclared dependency.
    """
    if runtime is None:
        return None
    sm = getattr(runtime, "state_manager", None)
    if sm is None:
        app_state = getattr(runtime, "_app_state", None)
        if app_state is not None:
            sm = getattr(app_state, "state_manager", None)
    if sm is None:
        return None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None) if backend else None


async def _resolve_model_name(conn: Any, model_value: str) -> str:
    """Translate the model column from ``image_generations`` to the
    canonical short name used by the UI dropdown + downstream APIs.

    ``image_generations.model`` stores whatever the original generation
    call wrote — historically a full filesystem path like
    ``/data/image_models/Lumina``. The dropdown option ``value`` is
    just the short name (``Lumina``). Path-shaped values silently
    no-op the dropdown set, so the form keeps its previous selection.

    Lookup ``image_models`` (``name`` PK, ``path`` column) and return
    the short name when we can match. Pass through unchanged for
    cloud / peer / already-short identifiers.
    """
    if conn is None or not model_value:
        return model_value or ""
    raw = str(model_value).strip()
    # Already short (no path separator) — passthrough
    if "/" not in raw and "\\" not in raw and not raw.startswith("cloud:"):
        return raw
    try:
        # Exact path match first
        cur = await conn.execute(
            "SELECT name FROM image_models WHERE path = ? LIMIT 1", (raw,),
        )
        row = await cur.fetchone()
        if row and row[0]:
            return row[0]
        # Suffix match — handles minor path normalization differences
        # (trailing slash, double slashes, etc.)
        for sep in ("/", "\\"):
            if sep in raw:
                tail = raw.rsplit(sep, 1)[1]
                cur = await conn.execute(
                    "SELECT name FROM image_models WHERE name = ? LIMIT 1", (tail,),
                )
                row = await cur.fetchone()
                if row and row[0]:
                    return row[0]
    except Exception as exc:  # noqa: BLE001 — lookup is best-effort
        log.warning("image_model_resolve_failed", model=raw[:120], error=str(exc)[:160])
    return raw


async def _infer_image_args(
    partial_args: dict[str, Any],
    session: SessionContext,
    runtime: Any,
) -> dict[str, Any]:
    """Fill model + settings from the user's most-recent image generation.

    Strategy: take the freshest image_generations row and copy over any
    field the matcher didn't supply. The user can override any of these
    via an explicit LLM tool call, but for natural-language requests
    ("generate a cat") the implicit "use what I used last time"
    expectation wins almost every time.

    Model field gets path→name resolution (see ``_resolve_model_name``)
    so the downstream dropdown receives a value it actually has an
    option for. Without this translation, the architect's "use your
    last model" intent silently degrades to "use whatever the form
    happens to have selected right now".
    """
    from augmentum.architect.inference import query_image_history

    args = dict(partial_args)
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return args

    conn = await _conn_from_runtime(runtime)
    if conn is None or not session.user_id:
        return args

    history = await query_image_history(conn, session.user_id, limit=1)
    if not history:
        return args

    last = history[0]

    # Copy every non-prompt field over when the caller didn't specify.
    # Prompt is always the user's current ask, never inherited.
    inheritable = (
        "model", "negative_prompt", "width", "height",
        "steps", "cfg_scale", "preset", "loras",
    )
    for key in inheritable:
        if key not in args or args[key] in (None, "", []):
            if last.get(key) not in (None, "", []):
                args[key] = last[key]

    # Translate model path → canonical name so the surface payload
    # carries a value the dropdown can actually accept.
    if args.get("model"):
        args["model"] = await _resolve_model_name(conn, args["model"])

    args["inferred_from_image_id"] = last.get("image_id", "")
    return args


# Courtesy fillers the regex didn't catch — stripped defensively before
# the prompt reaches the image surface. Order matters: longer matches
# first so "for me" isn't half-stripped to ", me".
_TAIL_FILLERS = (
    ", please", " please", " for me", " thanks", " thank you",
    ", thanks", ", thank you",
)


def _strip_tail_fillers(text: str) -> str:
    """Repeatedly strip courtesy fillers from the end of ``text``."""
    s = text.strip(" .,!?")
    changed = True
    while changed:
        changed = False
        lower = s.lower()
        for filler in _TAIL_FILLERS:
            if lower.endswith(filler):
                s = s[: -len(filler)].strip(" .,!?")
                changed = True
                break
    return s


async def _image_generate_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    """Emit the image.generate surface event so the image surface starts
    the generation. The architect doesn't drive the pipeline itself —
    it hands off to the surface that owns it.
    """
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't generate images for a signed-out session.",
        )

    # ``prompt_raw`` is set by the translator when expansion happens —
    # the spoken acknowledgment uses the user's original phrasing so
    # the UX feels natural ("Generating: a cat" rather than reading
    # back the 80-word scene description). ``prompt`` carries the
    # expanded form when available; falls back to raw when expansion
    # was disabled / failed / skipped (already-rich input).
    raw_label = _strip_tail_fillers(args.get("prompt_raw") or args.get("prompt") or "")
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return ActionResult(
            short_circuit=True,
            speak="What would you like me to generate?",
        )

    # Truncated label for the spoken ack — use the user's RAW phrasing
    # (e.g. "a cat") rather than the expanded scene description so the
    # ack stays conversational. Strip filler words and cap at 60 chars.
    short_label = raw_label[:60].rstrip() if raw_label else prompt[:60].rstrip()
    if raw_label and len(raw_label) > 60:
        short_label += "…"
    elif not raw_label and len(prompt) > 60:
        short_label += "…"

    model = args.get("model") or ""
    inferred_from = args.get("inferred_from_image_id") or ""

    log.info(
        "image_generate_dispatch",
        user_id=session.user_id,
        prompt_len=len(prompt),
        model=model or "(default)",
        inferred_from=inferred_from,
    )

    # Build the surface payload. Image surface reads this and applies
    # the architect-supplied settings to its generation form; any field
    # we don't set is filled by the surface's own defaults.
    payload: dict[str, Any] = {"prompt": prompt}
    for key in (
        "model", "negative_prompt", "width", "height",
        "steps", "cfg_scale", "preset", "loras",
    ):
        if args.get(key) not in (None, "", []):
            payload[key] = args[key]

    speak = (
        f"Generating: {short_label}."
        if not inferred_from
        else f"Generating {short_label} using your last settings."
    )

    # Record the prompt + model on the referent cache so a follow-up
    # like "another one" or "make a variant" has anchors to resolve
    # against. The final image_id is set later by the image surface
    # when generation completes (existing path).
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_image_prompt = prompt[:200]

    return ActionResult(
        short_circuit=True,
        speak=speak,
        surface_emit={
            "channel": "image.generate",
            "payload": payload,
        },
    )


register_action(
    id="image.generate_with_defaults",
    summary=(
        "Generate an image with the user's last-used model + settings. "
        "The matcher extracts the prompt; inference fills in model, "
        "sampler, steps, cfg_scale, dimensions, preset, and loras from "
        "the freshest image_generations row. New users get the image "
        "surface's own defaults."
    ),
    examples=[
        "generate a cat",
        "generate an image of a sunset",
        "make me a picture of a robot",
        "create an image of new york at night",
        "draw a dragon",
        "make an image of mountains",
    ],
    handler=_image_generate_handler,
    fanout=_TIER3_ONLY,
    arg_schema={
        "prompt": {
            "type": "string",
            "description": "What to draw — the subject matter.",
        },
        "model": {
            "type": "string",
            "description": "Optional model override (defaults to last-used).",
        },
        "negative_prompt": {
            "type": "string",
            "description": "Optional negative prompt — what to avoid.",
        },
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "steps": {"type": "integer"},
        "cfg_scale": {"type": "number"},
        "preset": {"type": "string"},
    },
    required=["prompt"],
    # Scoped to Becca's companion mode + chat, NOT the full-screen voice
    # call. Image generation produces a visual result that needs an
    # accessible surface — the call modal occupies the screen so the
    # generated image would land behind it. Becca-ptt fires from the
    # widget while the rest of the UI is reachable, so the user
    # actually sees the image surface open + the generation appear.
    surfaces=["becca", "chat"],
    stakes="costly",
    arg_inferrer=_infer_image_args,
    # Translation pass: expand raw user prompts ("a dog") into scene-
    # rich descriptions before the image model sees them. The handler
    # uses ``prompt_raw`` for the spoken ack and ``prompt`` (expanded)
    # for the surface emit. Bounded by a 4s timeout — image gen takes
    # 5-30s anyway, so the round-trip is invisible to the user.
    arg_transformer=lambda a, s, r: _arg_transformer(a, s, r),
)
