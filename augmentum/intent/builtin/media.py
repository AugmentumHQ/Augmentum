"""Media verbs — discovery (play).

LLM-orchestrated capabilities (Tier 3 only). The model picks these
verbs based on intent understanding plus the ReferentCache; we do NOT
pattern-match common words like "play" or "find" from the raw
transcript. That switchboard pattern misfires on conversational uses
("I played piano last weekend") and is brittle to STT noise. Becca
chooses the right verb from context.

``media.play`` resolves the query against the user's library via
``augmentum/media/resolver.py`` and DOES the thing: a clear winner
starts playing in the background mini-player (``media.resume``
channel, no surface yanking); near-ties surface as clickable cards in
the companion widget (``companion.candidates`` channel); a true miss
gets an honest "you don't have that" with external offers — never a
tag-coincidence auto-play. Companion Direct Action spec, 2026-06-10.

Music by genre/artist/mood is ``grove.play_matching``'s job (its tier
ladder — favourites → frontend favourites → clarify — is exactly the
right shape for mood asks). This verb is for KNOWN ITEMS: audiobooks,
podcasts, videos, comics, books the user owns.

Transport verbs (pause / next / previous) are owned by
``augmentum/architect/primitives/media_control.py`` and are not
redefined here.

Surface IDs in dispatches MUST match the channels in
``ui/scripts/intent-action-router.js``.
"""

from __future__ import annotations

import time as _time
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _offer_now() -> float:
    """Timestamp for a fresh offer parked in ``pending_candidates`` — read
    back against a TTL by the architect router so a stale offer isn't reused
    for a later, unrelated request."""
    return _time.time()

# Tier-3 only — the LLM picks via tool exposure, never a regex match.
# Latency-critical control (stop/pause/bye) lives in control.py; these
# discovery verbs happily pay the ~1-2s round-trip for understanding.
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


# ── Discovery: play / search ────────────────────────────────────────

def _speak_for_play(top) -> str:
    """One spoken line confirming what started."""
    bits = [f"Starting {top.title}"]
    if top.subtitle:
        bits.append(f"by {top.subtitle.split(' · ')[0]}")
    line = " ".join(bits)
    if top.in_progress:
        line += " — picking up where you left off"
    return line + "."


def _speak_for_offer(query: str, candidates) -> str:
    """Spoken candidate list — voice surfaces have no cards to click."""
    names = []
    for i, c in enumerate(candidates[:3], start=1):
        label = c.title if not c.subtitle else f"{c.title} ({c.subtitle})"
        names.append(f"{i}: {label}")
    return (
        f"I found a few matches for {query} — {'; '.join(names)}. "
        "Which one?"
    )


async def _direct_play_by_file_id(
    session: SessionContext, file_id: str,
) -> ActionResult | None:
    """Accept path for an offered pick: the router (or a parked
    candidate) hands us the exact file_id, so we skip the resolver and
    play it — after an ownership check, never on trust. Returns None
    when the id doesn't resolve in THIS user's library (caller falls
    back to the query path)."""
    conn = getattr(
        getattr(
            getattr(
                getattr(session, "app_state", None), "state_manager", None,
            ),
            "backend", None,
        ),
        "conn", None,
    )
    if conn is None:
        return None
    try:
        cur = await conn.execute(
            """SELECT name, json_extract(source_metadata, '$.entity_kind')
               FROM file_index WHERE id = ? AND user_id = ?""",
            (file_id, session.user_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:  # noqa: BLE001
        log.warning("media_play_file_id_lookup_failed", file_id=file_id[:40])
        return None
    if row is None:
        return None
    title = str(row[0] or "the pick")
    kind = str(row[1] or "")
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_file_id = file_id
        refs.pending_candidates = []
        refs.pending_candidates_at = 0.0
    log.info(
        "media_play_offered_pick",
        user_id=session.user_id, file_id=file_id, title=title[:80],
    )
    return ActionResult(
        short_circuit=True,
        speak=f"Starting {title}.",
        toast=f"Playing {title}"[:80],
        surface_emit={
            "channel": "media.resume",
            "payload": {
                "file_id": file_id,
                "content_label": title,
                "content_kind": kind,
            },
        },
    )


async def _media_play(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    query = (args.get("query") or "").strip()
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your library for a signed-out session.",
        )
    # Exact accept of an offered pick — the router resolves "the
    # second one" / "yeah that one" against parked candidates and
    # passes the chosen file_id straight through.
    file_id = str(args.get("file_id") or "").strip()
    if file_id:
        direct = await _direct_play_by_file_id(session, file_id)
        if direct is not None:
            return direct
        # Stale/foreign id — fall through to the query path.
    if not query:
        # No query extracted — ask, don't guess and don't yank panels.
        # Park so the answer fills the slot (see ReferentCache.pending_intent).
        import time as _t
        refs = getattr(session, "referents", None)
        if refs is not None:
            refs.pending_intent = {
                "action_id": "media.play",
                "args": {},
                "missing": ["query"],
                "question": "What should I play?",
                "asked_at": _t.time(),
            }
        return ActionResult(
            short_circuit=True,
            fulfilled=False,   # parked a question — keep it, don't voice "Playing…"
            speak="What should I play?",
        )

    from augmentum.media.resolver import resolve_media

    result = await resolve_media(
        getattr(session, "app_state", None),
        user_id=session.user_id,
        query=query,
    )
    refs = getattr(session, "referents", None)

    if result.decision == "play" and result.top is not None:
        top = result.top
        if refs is not None:
            refs.last_file_id = top.file_id
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        log.info(
            "media_play_direct",
            user_id=session.user_id, file_id=top.file_id,
            title=top.title[:80], score=round(top.score, 3),
        )
        return ActionResult(
            short_circuit=True,
            speak=_speak_for_play(top),
            toast=f"Playing {top.title}"[:80],
            surface_emit={
                "channel": "media.resume",
                "payload": {
                    "file_id": top.file_id,
                    "content_label": top.title,
                    "content_kind": top.content_kind,
                },
            },
        )

    if result.decision == "offer" and result.candidates:
        payloads = [c.to_payload() for c in result.candidates]
        if refs is not None:
            # Park for follow-up turns ("the second one") — replaced
            # wholesale on each offer, cleared on a successful play.
            refs.pending_candidates = payloads
            refs.pending_candidates_at = _offer_now()
        log.info(
            "media_play_offer",
            user_id=session.user_id, query=query[:80],
            n=len(payloads),
        )
        return ActionResult(
            short_circuit=True,
            speak=_speak_for_offer(result.query or query, result.candidates),
            surface_emit={
                "channel": "companion.candidates",
                "payload": {
                    "intent": "media.play",
                    "query": query,
                    "candidates": payloads,
                },
            },
        )

    # Cross-shelf check BEFORE claiming absence. Game titles routinely
    # land here (the router's play-anything bias sent "play Pokemon
    # Emerald" to media.play, live-test 2026-07-19), and "you don't
    # have that" is a lie when the title sits on the games shelf. A
    # clear game match launches (the user NAMED it); near-ties offer
    # the same candidate cards game.play uses.
    try:
        from augmentum.intent.builtin.games import (
            _launch_result,
            match_games_for_query,
        )
        from augmentum.intent.builtin.games import (
            _speak_for_offer as _speak_game_offer,
        )
        conn = getattr(
            getattr(
                getattr(
                    getattr(session, "app_state", None), "state_manager", None,
                ),
                "backend", None,
            ),
            "conn", None,
        )
        if conn is not None:
            g_scored = await match_games_for_query(
                conn, session.user_id, query, min_score=0.5,
            )
            if g_scored and (
                len(g_scored) == 1
                or (g_scored[0][0] >= 0.85 and g_scored[0][0] - g_scored[1][0] > 0.2)
            ):
                cand = g_scored[0][1]
                if refs is not None:
                    refs.pending_candidates = []
                    refs.pending_candidates_at = 0.0
                log.info(
                    "media_play_game_handoff",
                    user_id=session.user_id, query=query[:80],
                    artifact_id=cand["artifact_id"], title=cand["title"][:80],
                )
                return _launch_result(cand)
            if g_scored:
                payloads = [c for _, c in g_scored[:4]]
                if refs is not None:
                    refs.pending_candidates = payloads
                    refs.pending_candidates_at = _offer_now()
                log.info(
                    "media_play_game_offer",
                    user_id=session.user_id, query=query[:80], n=len(payloads),
                )
                return ActionResult(
                    short_circuit=True,
                    speak=_speak_game_offer(query, payloads),
                    surface_emit={
                        "channel": "companion.candidates",
                        "payload": {
                            "intent": "game.play",
                            "query": query,
                            "candidates": payloads,
                        },
                    },
                )
    except Exception:  # noqa: BLE001 — cross-shelf peek must never break the miss path
        log.warning("media_play_game_check_failed", query=query[:80])

    # Honest miss — no panel yank, no tag-coincidence gamble. Becca
    # owns the miss and offers the external paths (youtube / web
    # search verbs are in her roster; the user's next word decides).
    log.info(
        "media_play_miss",
        user_id=session.user_id, query=query[:80],
    )
    return ActionResult(
        short_circuit=True,
        # Not actuated — keep THIS honest line; the router must not voice the
        # model's optimistic "Playing X" over a library miss (the lying bug).
        fulfilled=False,
        speak=(
            f"I don't see {(result.query or query)[:60]} in your library. "
            "I can search YouTube or the web for it if you'd like."
        ),
    )


register_action(
    id="media.play",
    summary=(
        "Play a NAMED item they own — audiobook, podcast, video, "
        "comic, book — resolved by title. Starts background playback "
        "when one library match is clear; shows pickable candidates "
        "when several are close. Siblings: music by genre, artist, or "
        "mood is grove.play_matching; 'continue what I was on' with "
        "no title is media.resume; 'what should I listen to' with no "
        "title is media.recommend. Call when the user names a known "
        "item they don't already have on screen, or accepts an "
        "offered pick (pass its file_id)."
    ),
    examples=[
        "play the dune audiobook", "play the foundation",
        "play the expanse", "play that podcast about rome",
        "start playing the watchmen comic", "put on project hail mary",
        "yeah play the second one",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "Title (plus optional kind word like 'audiobook' / "
                "'comic') of the item to play. The user's phrasing — "
                "preserve their words."
            ),
        },
        "file_id": {
            "type": "string",
            "description": (
                "Exact library file id, ONLY when accepting one of "
                "the picks just offered (cards/candidates) — copy its "
                "file_id verbatim. Skips title resolution."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_media_play,
    delivery="artifact",
)


# ── Recommendation: the consumption-entity ladder, offered ──────────
#
# "What should I listen to next?" — Gate 1 picks from the user's OWN
# library (continuation / new arrivals / same author / same shelf),
# spoken naturally AND surfaced as tappable cards on the same
# companion.candidates dock media.play's offer path uses. The picks
# park in refs.pending_candidates, so the accept can be a tap, "the
# second one", or a title fragment — all three land on media.play's
# file_id fast-path. Spec: 2026-06-12-consumption-entity-discovery.

_RELATION_SPOKEN = {
    "continuation": "the next one in",
    "new_arrival": "new since the last sync in",
    "same_creator": "more by the same author",
    "same_genre": "same shelf, never started",
    "for_you": "fits what you've been into",
    "fresh": "new in your library",
}


def _speak_for_recommend(payloads: list[dict]) -> str:
    bits = []
    for i, c in enumerate(payloads[:3], start=1):
        why = (c.get("subtitle") or "").strip()
        label = c["title"] if not why else f"{c['title']} — {why}"
        bits.append(f"{i}: {label}")
    lead = (
        "A few things from your own library: " if len(bits) > 1
        else "One thing from your library stands out: "
    )
    return lead + "; ".join(bits) + ". Want me to start one?"


async def _media_recommend(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your library for a signed-out session.",
        )
    conn = getattr(
        getattr(
            getattr(
                getattr(session, "app_state", None), "state_manager", None,
            ),
            "backend", None,
        ),
        "conn", None,
    )
    if conn is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't see the library right now.",
        )

    from augmentum.discovery.catalog_recommender import (
        kind_word_to_filters,
        recommend_picks,
    )

    kind_arg = str(args.get("kind") or "").strip().lower()
    file_kind, entity_kind = kind_word_to_filters(kind_arg)
    genre = str(args.get("genre") or "").strip().lower()
    query = str(args.get("query") or "").strip()

    picks = await recommend_picks(
        conn, getattr(session, "app_state", None) and
        getattr(session.app_state, "file_index", None),
        user_id=session.user_id,
        kind=file_kind or None, entity_kind=entity_kind or None,
        genre=genre, query=query, limit=4,
    )

    payloads: list[dict] = []
    for p in picks:
        if not p.file_id or any(c["file_id"] == p.file_id for c in payloads):
            continue
        payloads.append({
            "file_id": p.file_id,
            "title": p.title,
            "subtitle": p.why,
            "kind": p.kind,
            "content_kind": p.kind,
            "in_progress": p.relation == "continuation",
            "relation": p.relation,
        })

    refs = getattr(session, "referents", None)
    if not payloads:
        if refs is not None:
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        what = " ".join(w for w in (genre, kind_arg) if w).strip()
        what = f"{what} " if what else ""
        return ActionResult(
            short_circuit=True,
            fulfilled=False,   # nothing to recommend — don't voice a confirmation
            speak=(
                f"I went through your library and nothing {what}turned "
                "up — either it's not in there or it's all finished. "
                "Want me to widen it, or look outside your library?"
            ),
        )

    if refs is not None:
        # Same parking contract as media.play's offer — a follow-up
        # ("the second one") resolves through the router's offered-
        # picks block onto media.play's file_id fast-path.
        refs.pending_candidates = payloads
        refs.pending_candidates_at = _offer_now()
    log.info(
        "media_recommend_offer",
        user_id=session.user_id, n=len(payloads),
        relations=[c["relation"] for c in payloads],
    )
    return ActionResult(
        short_circuit=True,
        speak=_speak_for_recommend(payloads),
        surface_emit={
            "channel": "companion.candidates",
            "payload": {
                "intent": "media.recommend",
                "query": " ".join(w for w in (genre, kind_arg) if w).strip(),
                "candidates": payloads,
            },
        },
    )


register_action(
    id="media.recommend",
    summary=(
        "Recommend something from their OWN library when they ask "
        "WITHOUT naming a title. Draws from their whole catalog — "
        "movies, shows, audiobooks, podcasts, comics, books — filtered "
        "by whatever they specify (a type, a genre/mood, a theme) and "
        "ranked by what they've been into. Works even if they've never "
        "played anything: it reads the catalog itself, not just history. "
        "Picks the next unfinished volume, an unread book by an author "
        "they like, or a never-started title that fits the ask. Speaks "
        "the options and shows tappable cards; accepting one (tap or "
        "'the second one') starts playback. Siblings: a NAMED item is "
        "media.play; music by mood is grove.play_matching. Pass kind / "
        "genre / query from what they said — leave blank for 'surprise "
        "me'."
    ),
    examples=[
        "what should I listen to next", "recommend me something",
        "any good movies I haven't seen", "what comic should I read",
        "got anything new for me", "pick something for me",
        "recommend a funny movie", "something scary to watch tonight",
        "a sci-fi audiobook", "what should I watch",
    ],
    arg_schema={
        "kind": {
            "type": "string",
            "description": (
                "Content type when they name one: movie | show | "
                "audiobook | podcast | comic | manga | book | music | "
                "video. Omit for everything."
            ),
        },
        "genre": {
            "type": "string",
            "description": (
                "Single genre or mood word when they specify one — "
                "'comedy', 'horror', 'sci-fi', 'romance'. Omit when "
                "they didn't."
            ),
        },
        "query": {
            "type": "string",
            "description": (
                "Free-text theme/subject when the ask isn't a clean "
                "genre — 'something about space travel', 'a heist'. "
                "Their words. Omit for a plain browse of the type/genre."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_media_recommend,
    delivery="artifact",
)


# ── Mid-experience controls (wiring program Phase 1) ───────────────
#
# Volume rides the receiver resolution ladder server-side (see
# augmentum/intent/media_devices.py): a named device or an active cast
# session is controlled through DeviceRegistry.invoke; otherwise the
# verb falls through to a surface emit and the in-tab foreground
# player (media-player / Grove) handles it in the browser. Speed and
# sleep-timer are in-tab-only concepts (playbackRate and the player's
# own timer), so they're pure surface emits on the media.adjust
# channel.

_VOLUME_DIRECTIONS = ("up", "down", "set", "mute", "unmute")


async def _media_volume(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach playback for a signed-out session.",
        )
    direction = str(args.get("direction") or "").strip().lower()
    level = args.get("level")
    if direction not in _VOLUME_DIRECTIONS:
        # A bare level implies "set"; otherwise ask rather than guess
        # a direction on something that changes the room's loudness.
        if level is not None:
            direction = "set"
        else:
            return ActionResult(
                short_circuit=True,
                speak="Louder or quieter?",
                clarify={"missing": ["direction"], "args": dict(args)},
            )

    from augmentum.intent.media_devices import (
        resolve_playback_target,
        volume_on_target,
    )

    ladder = await resolve_playback_target(
        getattr(session, "app_state", None),
        session.user_id,
        device_hint=str(args.get("device") or ""),
    )
    if ladder.miss:
        return ActionResult(
            short_circuit=True,
            speak=f"I don't see a device called {ladder.miss[:60]}.",
        )
    if ladder.clarify:
        return ActionResult(
            short_circuit=True,
            speak=ladder.clarify,
            clarify={"missing": ["device"], "args": dict(args)},
        )
    if ladder.target is not None:
        ok, detail = await volume_on_target(
            getattr(session, "app_state", None), session.user_id,
            ladder.target, direction=direction,
            level=int(level) if level is not None else None,
        )
        label = ladder.target.device_label
        if not ok:
            return ActionResult(short_circuit=True, speak=f"{detail}.")
        if direction == "mute":
            spoken = f"Muted {label}."
        elif direction == "unmute":
            spoken = f"Unmuted {label}."
        else:
            spoken = f"{label} is at {detail} percent now."
        log.info(
            "media_volume_receiver",
            user_id=session.user_id, device_id=ladder.target.device_id,
            direction=direction,
        )
        return ActionResult(
            short_circuit=True,
            speak=spoken,
            digest=f"volume {direction} on {label}",
        )

    # Nothing cast — the in-tab foreground player owns it. The change
    # itself is audible feedback; a toast names it without TTS noise.
    toast = {
        "up": "Volume up", "down": "Volume down", "mute": "Muted",
        "unmute": "Unmuted",
    }.get(direction, f"Volume {level}%" if level is not None else "Volume")
    return ActionResult(
        short_circuit=True,
        toast=toast,
        digest=f"volume {direction} (in-tab player)",
        surface_emit={
            "channel": "media.volume",
            "payload": {
                "direction": direction,
                "level": int(level) if level is not None else None,
            },
        },
    )


register_action(
    id="media.volume",
    summary=(
        "Change playback volume or mute — on the casting device when "
        "something is playing on a TV/receiver (named device wins), "
        "otherwise on the in-tab player on the user's screen. "
        "Siblings: pausing or skipping is media.pause/next/previous; "
        "playback speed is media.speed. Call for 'turn it up/down', "
        "'set volume to 30', 'mute the TV'."
    ),
    examples=[
        "turn it up", "turn the volume down", "set the volume to 40",
        "mute the tv", "quieter please", "unmute the living room tv",
    ],
    arg_schema={
        "direction": {
            "type": "string",
            "enum": list(_VOLUME_DIRECTIONS),
            "description": "up | down | set | mute | unmute.",
        },
        "level": {
            "type": "integer",
            "description": "Target volume 0-100 (with direction=set).",
        },
        "device": {
            "type": "string",
            "description": (
                "Device name if the user names one ('the living room "
                "TV'). Omit to control whatever is actively playing."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_media_volume,
    delivery="artifact",
    stakes="disruptive",
)


async def _media_speed(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    rate = args.get("rate")
    step = str(args.get("step") or "").strip().lower()
    payload: dict[str, Any] = {"action": "speed"}
    if rate is not None:
        try:
            payload["rate"] = max(0.5, min(3.0, float(rate)))
        except (TypeError, ValueError):
            rate = None
    if rate is None:
        if step not in ("faster", "slower", "reset"):
            return ActionResult(
                short_circuit=True,
                speak="Faster or slower?",
                clarify={"missing": ["step"], "args": dict(args)},
            )
        payload["step"] = step
    label = f"{payload['rate']}x" if "rate" in payload else step
    return ActionResult(
        short_circuit=True,
        toast=f"Speed: {label}",
        digest=f"playback speed {label} (in-tab player)",
        surface_emit={"channel": "media.adjust", "payload": payload},
    )


register_action(
    id="media.speed",
    summary=(
        "Change playback speed of the in-tab player on the user's "
        "screen — audiobooks/podcasts at 1.5x, 'slower', back to "
        "normal. In-tab only; cast devices don't take a rate. "
        "Siblings: loudness is media.volume."
    ),
    examples=[
        "speed it up to one point five", "play this at 2x",
        "slow it down a bit", "back to normal speed",
    ],
    arg_schema={
        "rate": {
            "type": "number",
            "description": "Exact playback rate (0.5-3.0), e.g. 1.5.",
        },
        "step": {
            "type": "string",
            "enum": ["faster", "slower", "reset"],
            "description": "Relative nudge when no exact rate was named.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_media_speed,
    delivery="artifact",
    stakes="disruptive",
)


async def _media_sleep_timer(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    cancel = bool(args.get("cancel"))
    end_of_chapter = bool(args.get("end_of_chapter"))
    minutes = args.get("minutes")
    if cancel:
        payload: dict[str, Any] = {"action": "sleep_timer", "cancel": True}
        spoken = "Sleep timer's off."
    elif end_of_chapter:
        payload = {"action": "sleep_timer", "end_of_chapter": True}
        spoken = "I'll stop it at the end of this chapter."
    else:
        try:
            mins = int(minutes)
        except (TypeError, ValueError):
            mins = 0
        if mins <= 0:
            return ActionResult(
                short_circuit=True,
                speak="How long should it keep playing?",
                clarify={"missing": ["minutes"], "args": dict(args)},
            )
        payload = {"action": "sleep_timer", "minutes": mins}
        spoken = f"Stopping in {mins} minutes."
    return ActionResult(
        short_circuit=True,
        speak=spoken,
        digest="sleep timer set (in-tab player)" if not cancel
        else "sleep timer cancelled",
        surface_emit={"channel": "media.adjust", "payload": payload},
    )


register_action(
    id="media.sleep_timer",
    summary=(
        "Set a sleep timer on the in-tab player on the user's screen — "
        "stop playback in N minutes or at the end of the current "
        "chapter. Call for 'stop in 30 minutes', 'sleep timer', "
        "'turn off after this chapter'. Siblings: a general reminder "
        "with no playback involved is time.set_timer."
    ),
    examples=[
        "stop playing in 30 minutes", "sleep timer for an hour",
        "turn it off after this chapter", "cancel the sleep timer",
    ],
    arg_schema={
        "minutes": {
            "type": "integer",
            "description": "Stop after this many minutes.",
        },
        "end_of_chapter": {
            "type": "boolean",
            "description": "Stop when the current chapter ends.",
        },
        "cancel": {
            "type": "boolean",
            "description": "Cancel an existing sleep timer.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_media_sleep_timer,
    delivery="artifact",
    stakes="disruptive",
)


# ── media.search — retired ──────────────────────────────────────────
# Retired 2026-06-11: ``media.search`` duplicated ``search.local``
# byte-for-byte in effect (same files.search_open surface emit, same
# payload, same empty-query open-Files fallback — the Files panel IS
# the media library browser). Two names for one capability fragmented
# model behavior across families, the same failure mode that retired
# ``search.web`` earlier the same day. ``search.local`` (in
# augmentum/intent/builtin/search.py) absorbed the media-flavored
# examples; browse-before-play asks land there.
