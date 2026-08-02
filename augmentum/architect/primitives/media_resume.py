"""media.resume — pick up where the user left off in their last media.

User: "resume my audiobook", "continue my book", "play where I left
off", "pick up the audiobook". Imperative-only.

The substrate moat: Augmentum tracks per-user playback history via
``device_play_history`` (the MRU log). The architect queries the
freshest media row matching ``content_kind='audiobook'`` or
``'video'``, then emits a surface event the media-player picks up.

If no recent media exists, the handler returns a clarifying speak
rather than dispatching — better than picking something arbitrary.
The inferrer also handles "resume the [series_name]" forms by
substring-matching content_label against the user's history.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionResult,
    SessionContext,
)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _conn_from_runtime(runtime: Any) -> Any:
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


async def _infer_media_resume_args(
    partial_args: dict[str, Any],
    session: SessionContext,
    runtime: Any,
) -> dict[str, Any]:
    """Find the freshest media item to resume.

    Strategy:
      1. If the user named a specific item ("resume Dune"), filter
         device_play_history by content_label substring match.
      2. Otherwise pick the most-recent media-kind play (audiobook
         > video, but newest-first across both).
      3. Fill the surface-emit args with file_id + content_label +
         content_kind + capability_id.
    """
    from augmentum.architect.inference import query_play_history

    args = dict(partial_args)
    conn = await _conn_from_runtime(runtime)
    if conn is None or not session.user_id:
        return args

    title_hint = (args.get("title") or "").strip().lower()

    # Pull a generous window so the title-substring search has range.
    # Audiobooks first because "resume my audiobook" / "continue my
    # book" is the canonical phrasing; the templates focus there.
    rows = await query_play_history(
        conn, session.user_id,
        content_kind="audiobook",
        limit=20,
        favourites_first=False,  # most-recent ordering for "resume"
    )
    if not rows:
        # Fall back to any media kind (audiobook OR video OR podcast).
        rows = await query_play_history(
            conn, session.user_id,
            limit=20,
            favourites_first=False,
        )

    if not rows:
        return args

    pick = None
    if title_hint:
        for r in rows:
            label = (r.get("content_label") or "").lower()
            if title_hint in label:
                pick = r
                break
    if pick is None:
        pick = rows[0]

    args["file_id"] = pick.get("file_id") or ""
    args["content_label"] = pick.get("content_label") or ""
    args["content_kind"] = "audiobook"  # narrow default — handler will override on video lookup
    args["capability_id"] = pick.get("capability_id") or ""

    # Best-effort detect content_kind from the capability_id since
    # query_play_history doesn't carry it back through. Names like
    # "media.audio_play@1" → audiobook; "media.video_play@1" → video.
    cap = (pick.get("capability_id") or "").lower()
    if "video" in cap:
        args["content_kind"] = "video"
    elif "audio" in cap:
        args["content_kind"] = "audiobook"
    return args


async def _media_resume_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't resume media for a signed-out session.",
        )

    file_id = (args.get("file_id") or "").strip()
    label = (args.get("content_label") or "").strip()
    kind = (args.get("content_kind") or "").strip() or "audiobook"

    if not file_id and not label:
        return ActionResult(
            short_circuit=True,
            speak=(
                "I don't see anything recent to resume. "
                "Play something first and I'll pick up from there next time."
            ),
        )

    short = label[:60] if label else "your last session"
    log.info(
        "architect_media_resume",
        user_id=session.user_id, file_id=file_id, kind=kind, label=label[:80],
    )

    return ActionResult(
        short_circuit=True,
        speak=f"Resuming {short}.",
        surface_emit={
            "channel": "media.resume",
            "payload": {
                "file_id": file_id,
                "content_label": label,
                "content_kind": kind,
            },
        },
    )


register_action(
    id="media.resume",
    summary=(
        "Continue what they were LAST playing — no title needed. "
        "Resumes the user's most recent audiobook, podcast, or video "
        "from the latest position tracked in device_play_history; an "
        "optional title narrows to a specific item. Sibling: when "
        "they NAME a new item to start, use media.play."
    ),
    examples=[
        "resume my audiobook",
        "continue my book",
        "play where I left off",
        "pick up the audiobook",
        "resume Dune",
        "continue listening",
    ],
    handler=_media_resume_handler,
    delivery="artifact",
    arg_schema={
        "title": {
            "type": "string",
            "description": "Optional title or series to resume.",
        },
    },
    surfaces=["becca", "chat"],
    stakes="disruptive",
    arg_inferrer=_infer_media_resume_args,
    templates=[
        # Resume + optional title slot
        "(resume|continue) [my] [(book|audiobook|podcast|video|listening|reading)] [{title}]",
        # "play where I left off" / "pick up where I left off"
        "(play|pick up) where i left off",
        # "pick up my audiobook" / "pick up the audiobook [title]"
        "pick up [(my|the)] (audiobook|book|podcast|video|series) [{title}]",
        # "keep listening" / "keep watching" — single-verb resume
        "(keep|continue) (listening|watching|reading)",
    ],
)
