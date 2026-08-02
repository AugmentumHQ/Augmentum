"""Architect API routes.

Two endpoints today:

  * ``POST /api/architect/observe`` — client bridge for surface
    observations the server can't see (AudioBus state changes,
    foreground/blur transitions, scroll-depth, etc.). The handler
    validates user_id and republishes the event on the
    CompanionRuntime bus under ``surface.<X>.<Y>`` topics so the
    observer's recent deque picks them up.

  * ``GET /api/architect/capabilities`` — list architect-callable
    primitives filtered by client surface. Drives discovery UI
    ("here's what you can ask the companion to do").

Both endpoints are user-scoped — auth-protected, user_id extracted
from the request scope. Anon requests (no user) are rejected with
401 rather than leaking to the empty-user sentinel.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/architect", tags=["architect"])


# ---------------------------------------------------------------------------
# Allow-list of observation topics the client bridge may publish.
#
# We restrict to known prefixes so a compromised browser can't flood the
# runtime bus with arbitrary topics. The set mirrors what becca-observer.js
# emits today; new client-emitted surfaces add their prefix here when
# they ship.
# ---------------------------------------------------------------------------
_ALLOWED_TOPIC_PREFIXES: tuple[str, ...] = (
    "surface.audio.",       # AudioBus state changes (music/narration/etc.)
    "surface.attention.",   # foreground/blur transitions across surfaces
    "surface.browse.",      # client-side browse activity (page-turn, scroll-depth)
    "surface.media.",       # client-side media controls (seek, speed)
    "surface.comic.",       # comic page transitions
    "surface.commands.",    # palette agent-action catalog sync (app menu)
    "surface.narrative.",   # active story scene (character select/deselect)
    "surface.coder.",       # workspace file open/close
)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request scope."""
    user = request.scope.get("user")
    return user.id if user else ""


def _is_allowed_topic(topic: str) -> bool:
    """Topic-prefix allow-list — see _ALLOWED_TOPIC_PREFIXES."""
    return any(topic.startswith(prefix) for prefix in _ALLOWED_TOPIC_PREFIXES)


# ---------------------------------------------------------------------------
# POST /api/architect/observe — client bridge
# ---------------------------------------------------------------------------


@router.post("/observe")
async def observe(request: Request) -> JSONResponse:
    """Forward a client-side observation onto the runtime bus.

    Body shape:
      {
        "topic": "surface.audio.kind_changed",       # required, allow-listed
        "payload": {                                  # required, dict
          "kind": "music",
          "active": true,
          ...
        }
      }

    The handler injects ``user_id`` into the payload before publishing so
    every downstream consumer can filter by it. Returns 204 on success
    (no body — the publish is fire-and-forget).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    topic = (body.get("topic") or "").strip()
    payload = body.get("payload") or {}

    if not topic or not isinstance(payload, dict):
        return JSONResponse(
            {"error": "Missing topic or payload"}, status_code=400,
        )
    if not _is_allowed_topic(topic):
        return JSONResponse(
            {"error": f"Topic '{topic[:80]}' not allowed"}, status_code=400,
        )

    # Feed the companion's presence organ BEFORE the runtime check — the
    # attention store is useful (router deixis, prompt now-context) even
    # when companion_runtime is disabled and the bus publish is skipped.
    from augmentum.companion_runtime.presence_context import observe_attention
    observe_attention(uid, topic, payload)

    # App-menu catalog sync rides the same bridge (app.act candidates).
    from augmentum.intent.app_menu import observe_commands
    observe_commands(uid, topic, payload)

    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        # Runtime disabled — accept and drop. Returning 204 keeps the
        # client bridge from retrying forever in environments where
        # companion_runtime_enabled is False.
        return JSONResponse({"status": "ignored"}, status_code=200)

    bus = getattr(runtime, "bus", None)
    if bus is None:
        log.warning("architect_observe_no_bus")
        return JSONResponse({"status": "ignored"}, status_code=200)

    # user_id so downstream consumers can filter by caller; client (the
    # auth session's source: web / android / cast_receiver) so the
    # topical aggregator can honor `companion_attention_sources` and the
    # provenance chip can say which device produced the signal. Client
    # bridges can't be trusted to self-report, so both come from the
    # server-side session, overwriting anything in the body.
    _user = request.scope.get("user")
    _client = getattr(_user, "session_source", "web") if _user else "web"
    enriched = {**payload, "user_id": uid, "client": _client}

    try:
        # FACTUAL_ONLY: surfaces in the recent deque without forcing a
        # journal write or PAD update. Promote per-topic later if a
        # specific stream proves journal-worthy.
        await bus.publish_topic(
            topic,
            enriched,
            propagation="FACTUAL_ONLY",
        )
    except Exception as exc:  # noqa: BLE001 — log and degrade
        log.warning(
            "architect_observe_publish_failed",
            topic=topic, error=str(exc)[:200],
        )
        return JSONResponse(
            {"error": "Publish failed"}, status_code=500,
        )

    return JSONResponse({"status": "ok"}, status_code=200)


# ---------------------------------------------------------------------------
# POST /api/architect/load_context — "Read this …" widget handoff
# ---------------------------------------------------------------------------


@router.post("/load_context")
async def load_context(request: Request) -> JSONResponse:
    """Hand the companion the FULL content of what the user is looking at.

    The perception bridge (``/observe``) only carries index/digest
    fidelity — a page title, a file name. This is the opt-in deep channel:
    when the user presses the widget's "Read this page / chat / file"
    button, the surface posts the full text here. We stash it in the
    ``LoadedContextStore`` (ephemeral, per-user, latest-per-kind); the
    companion's prompt then carries a digest of it while the full body
    waits behind ``context_peek('loaded')`` — so the prompt budget pays
    only for the digest, not the whole article.

    Body:
      {"kind": "page"|"chat"|"file"|"scene"|"book", "label": str,
       "content": str, "ref": str?}

    Returns ``{status, chars}`` — chars is how much was stored (capped).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    kind = (body.get("kind") or "").strip()
    label = (body.get("label") or "").strip()
    content = body.get("content") or ""
    ref = (body.get("ref") or "").strip()

    if not kind or not isinstance(content, str):
        return JSONResponse(
            {"error": "Missing kind or content"}, status_code=400,
        )
    if not content.strip():
        return JSONResponse(
            {"error": "Empty content — nothing to load"}, status_code=400,
        )

    # Screen-read ingest (Android assistant Slice 2) is opt-in: the role and
    # presence are harmless, but capturing whatever's on screen is sensitive,
    # so kind="screen" is gated behind companion_assist_enabled (default OFF).
    # Other kinds (page/chat/file — the web "Read this" widget, an explicit
    # per-item user action) are unaffected. Always 200 so the phone's
    # fire-and-forget post never surfaces an error.
    if kind == "screen":
        from augmentum.config import settings
        if not getattr(settings, "companion_assist_enabled", False):
            return JSONResponse(
                {"status": "disabled", "stored": False}, status_code=200,
            )

    from augmentum.companion_runtime.presence_context import LOADED
    chars = LOADED.load(uid, kind, label=label, content=content, ref=ref)
    log.info("companion_context_loaded", kind=kind, chars=chars, label=label[:60])
    return JSONResponse({"status": "ok", "chars": chars}, status_code=200)


# ---------------------------------------------------------------------------
# GET /api/architect/capabilities — discovery
# ---------------------------------------------------------------------------


@router.get("/capabilities")
async def list_capabilities(request: Request) -> JSONResponse:
    """Return architect-callable actions filtered by ``surface`` query param.

    ``?surface=voice`` returns voice-eligible actions; ``?surface=chat``
    returns chat-eligible. Omitting the param returns all registered
    actions across surfaces (UI uses this for the discovery card).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    surface = (request.query_params.get("surface") or "").strip().lower()

    from augmentum.intent.registry import list_actions

    actions: list[dict[str, Any]] = []
    for action in list_actions():
        if surface and not action.surfaces_for(surface):
            continue
        actions.append({
            "id": action.id,
            "summary": action.summary,
            "examples": list(action.examples),
            "surfaces": list(action.surfaces) or ["*"],
            "modes": list(action.modes) or ["*"],
            "required_args": list(action.required_args),
            "has_inferrer": action.arg_inferrer is not None,
            "tier1": action.fanout.tier1,
            "tier3": action.fanout.tier3,
        })

    return JSONResponse({"actions": actions, "count": len(actions)})
