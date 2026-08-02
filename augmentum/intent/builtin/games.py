"""Game verbs — launch and recommend from the user's own game library.

Part of the 2026-07-19 companion content-coverage pass: games were the
largest content family with ZERO companion-reachable path — pinned js13k
games, emulator ROMs, and app builds all had working launch surfaces
(game-surface, emulator-stage, workspace) that only the Library's Open
button could reach. These verbs give the companion the same reach
through the shared frontend dispatcher (``ui/scripts/library/
open-item.js``), emitted on the ``game.launch`` channel.

Same design contract as media.py (Companion Direct Action, 2026-06-10):
LLM-orchestrated Tier 3 only — no regex switchboard on "play"; a clear
winner launches, near-ties surface as tappable candidate cards
(``companion.candidates``), a miss is owned honestly. Candidates park in
``refs.pending_candidates`` carrying ``artifact_id`` so a follow-up
("the second one") routes back through ``game.play``'s artifact_id
fast-path — the game sibling of media.play's file_id accept.

``game.recommend`` reads the catalog plus ``title_runs`` play history:
unplayed titles first, then least-recently-played, so "give me
something to play" surfaces the backlog instead of the same favourite.
"""

from __future__ import annotations

import json
import re
import time as _time
from difflib import SequenceMatcher
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

# Library artifact kinds that count as "a game" for these verbs. App
# builds are deliberately included — "play my snake build" is a real
# ask and the dispatcher routes them to the workspace player.
_GAME_KINDS = ("game", "emulator_rom", "app_build")


def _conn_of(session: SessionContext):
    return getattr(
        getattr(
            getattr(
                getattr(session, "app_state", None), "state_manager", None,
            ),
            "backend", None,
        ),
        "conn", None,
    )


def _row_to_candidate(row) -> dict[str, Any]:
    """artifacts row (id, display_name, filename, metadata-json[, runs,
    last_played, playtime_s]) → the companion.candidates payload shape.
    artifact_id (not file_id) marks it as a library-item candidate for
    the dock and the accept path. Play-history columns are optional so
    both query shapes share this mapper; when present they ride into
    the payload so the detail panel can show full metadata (author,
    system, source, times played, total playtime, last played)."""
    aid, display_name, filename, meta_raw = row[0], row[1], row[2], row[3]
    try:
        meta = json.loads(meta_raw or "{}")
    except (TypeError, ValueError):
        meta = {}
    kind = str(meta.get("kind") or "game")
    title = str(display_name or filename or "Untitled game")
    sub_bits = [b for b in (meta.get("author"), meta.get("source")) if b]
    system = str(meta.get("system") or "")
    if kind == "emulator_rom" and system:
        sub_bits.append(system.upper())
    cand = {
        "artifact_id": aid,
        "title": title,
        "subtitle": " · ".join(str(b) for b in sub_bits),
        "kind": kind,
        "content_kind": kind,
        "cover_url": str(meta.get("thumbnail_url") or meta.get("cover_url") or ""),
        "system": system,
        "author": str(meta.get("author") or ""),
        "source": str(meta.get("source") or ""),
        "description": str(meta.get("description") or ""),
    }
    if len(row) > 6:
        cand["runs"] = int(row[4] or 0)
        cand["last_played"] = str(row[5] or "")
        cand["playtime_s"] = int(row[6] or 0)
    return cand


async def _fetch_game_rows(conn, user_id: str) -> list:
    """All of this user's game-kind artifacts, with play history joined
    in (runs / last played / total playtime from title_runs) so every
    candidate payload carries full metadata. The kind lives inside the
    metadata JSON, so filter with json_extract — the catalog is small
    (pins, ROMs, builds), not the whole-table scan it looks like."""
    cur = await conn.execute(
        """SELECT a.id, a.display_name, a.filename, a.metadata,
                  COUNT(r.id), MAX(r.started_at),
                  COALESCE(SUM(r.duration_s), 0)
           FROM artifacts a
           LEFT JOIN title_runs r
             ON r.artifact_id = a.id AND r.user_id = a.user_id
           WHERE a.user_id = ?
             AND json_extract(a.metadata, '$.kind') IN (?, ?, ?)
           GROUP BY a.id
           ORDER BY a.created_at DESC LIMIT 200""",
        (user_id, *_GAME_KINDS),
    )
    rows = await cur.fetchall()
    await cur.close()
    return list(rows or [])


# Filler that voice asks carry ("one of the pokemon games", "play some
# tetris please") — stripped before scoring so the resolver matches the
# TITLE the user meant, not their phrasing. Live-test 2026-07-19: "One
# of the Pokemon games" was scored verbatim and missed both Pokémon
# ROMs; too-literal matching is a recurring class across resolvers.
_STOPWORDS = frozenset({
    "a", "an", "the", "one", "of", "some", "any", "that", "this",
    "my", "me", "please", "game", "games", "play", "playing",
    "version", "like",
})


def _norm_tokens(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [t for t in s.split() if t and t not in _STOPWORDS]


def _score(query: str, title: str) -> float:
    """Normalized fuzzy title match in [0, 1]: stopword-stripped exact >
    substring > per-token best-of (exact / prefix / SequenceMatcher)."""
    q = _norm_tokens(query)
    t = _norm_tokens(title)
    if not q or not t:
        return 0.0
    qs, ts = " ".join(q), " ".join(t)
    if qs == ts:
        return 1.0
    if qs in ts or ts in qs:
        return 0.9
    total = 0.0
    for qt in q:
        best = 0.0
        for tt in t:
            if qt == tt:
                r = 1.0
            elif tt.startswith(qt) or qt.startswith(tt):
                r = 0.85
            else:
                r = SequenceMatcher(None, qt, tt).ratio()
            best = max(best, r)
        total += best
    return 0.75 * (total / len(q))


async def match_games_for_query(
    conn, user_id: str, query: str, *, min_score: float = 0.35,
) -> list[tuple[float, dict[str, Any]]]:
    """Score the user's game shelf against a free-text query. Shared by
    game.play and by media.play's cross-shelf miss check (a game title
    routinely lands on media.play via the router's play-anything bias,
    and "you don't have that" is a lie when it sits on the games shelf)."""
    rows = await _fetch_game_rows(conn, user_id)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        cand = _row_to_candidate(row)
        s = _score(query, cand["title"])
        if s > min_score:
            scored.append((s, cand))
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored


def _launch_result(cand: dict[str, Any]) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        speak=f"Launching {cand['title']}.",
        toast=f"Launching {cand['title']}"[:80],
        surface_emit={
            "channel": "game.launch",
            "payload": {
                "artifact_id": cand["artifact_id"],
                "title": cand["title"],
                "game_kind": cand["kind"],
            },
        },
    )


def _speak_for_offer(query: str, cands: list[dict]) -> str:
    names = []
    for i, c in enumerate(cands[:3], start=1):
        label = c["title"] if not c["subtitle"] else f"{c['title']} ({c['subtitle']})"
        names.append(f"{i}: {label}")
    return (
        f"I found a few games matching {query} — {'; '.join(names)}. "
        "Which one?"
    )


async def _game_play(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your game library for a signed-out session.",
        )
    conn = _conn_of(session)
    if conn is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't see the library right now.",
        )

    refs = getattr(session, "referents", None)

    # Accept path for an offered pick — ownership-checked, never on trust.
    artifact_id = str(args.get("artifact_id") or "").strip()
    if artifact_id:
        cur = await conn.execute(
            """SELECT id, display_name, filename, metadata
               FROM artifacts WHERE id = ? AND user_id = ?""",
            (artifact_id, session.user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        if row is not None:
            cand = _row_to_candidate(row)
            if refs is not None:
                refs.pending_candidates = []
                refs.pending_candidates_at = 0.0
            log.info(
                "game_play_offered_pick",
                user_id=session.user_id, artifact_id=artifact_id,
                title=cand["title"][:80],
            )
            return _launch_result(cand)
        # Stale/foreign id — fall through to the query path.

    query = (args.get("query") or "").strip()
    if not query:
        if refs is not None:
            refs.pending_intent = {
                "action_id": "game.play",
                "args": {},
                "missing": ["query"],
                "question": "Which game?",
                "asked_at": _time.time(),
            }
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="Which game?",
        )

    scored = await match_games_for_query(conn, session.user_id, query)

    if scored and (
        len(scored) == 1 or scored[0][0] >= 0.85 and scored[0][0] - scored[1][0] > 0.2
    ):
        cand = scored[0][1]
        if refs is not None:
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        log.info(
            "game_play_direct",
            user_id=session.user_id, artifact_id=cand["artifact_id"],
            title=cand["title"][:80], score=round(scored[0][0], 3),
        )
        return _launch_result(cand)

    if scored:
        payloads = [c for _, c in scored[:4]]
        if refs is not None:
            refs.pending_candidates = payloads
            refs.pending_candidates_at = _time.time()
        log.info(
            "game_play_offer",
            user_id=session.user_id, query=query[:80], n=len(payloads),
        )
        return ActionResult(
            short_circuit=True,
            speak=_speak_for_offer(query, payloads),
            surface_emit={
                "channel": "companion.candidates",
                "payload": {
                    "intent": "game.play",
                    "query": query,
                    "candidates": payloads,
                },
            },
        )

    log.info("game_play_miss", user_id=session.user_id, query=query[:80])
    return ActionResult(
        short_circuit=True,
        fulfilled=False,
        speak=(
            f"I don't see {query[:60]} in your game library. "
            "You can pin new games from the Games browser — want me to "
            "open it?"
        ),
    )


register_action(
    id="game.play",
    summary=(
        "Launch a NAMED game they own — a pinned web game, an emulator "
        "ROM, or one of their own app builds — resolved by title from "
        "their library. A clear match launches in its player (game "
        "surface / emulator / workspace); near-ties show pickable "
        "cards. Call when the user names a game, or accepts an offered "
        "game pick (pass its artifact_id). Sibling: 'what should I "
        "play' with no title is game.recommend."
    ),
    examples=[
        "play tetris", "launch pokemon red", "start that racing game",
        "open my snake build", "play the second one",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "Title of the game to launch — the user's phrasing, "
                "preserve their words."
            ),
        },
        "artifact_id": {
            "type": "string",
            "description": (
                "Exact library artifact id, ONLY when accepting one of "
                "the game picks just offered — copy its artifact_id "
                "verbatim. Skips title resolution."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_game_play,
    delivery="artifact",
)


# ── Recommendation: the unplayed-backlog ladder, offered ────────────

def _speak_for_recommend(payloads: list[dict]) -> str:
    bits = []
    for i, c in enumerate(payloads[:3], start=1):
        why = (c.get("subtitle") or "").strip()
        label = c["title"] if not why else f"{c['title']} — {why}"
        bits.append(f"{i}: {label}")
    lead = (
        "A few from your game shelf: " if len(bits) > 1
        else "One game stands out: "
    )
    return lead + "; ".join(bits) + ". Want me to start one?"


async def _game_recommend(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your game library for a signed-out session.",
        )
    conn = _conn_of(session)
    if conn is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't see the library right now.",
        )

    # Catalog + play history in one pass (shared query): unplayed first
    # (the backlog is the point of asking), then least-recently-played.
    rows = await _fetch_game_rows(conn, session.user_id)
    rows.sort(key=lambda r: (0 if int(r[4] or 0) == 0 else 1, str(r[5] or "")))
    rows = rows[:4]

    payloads: list[dict] = []
    for row in rows or []:
        cand = _row_to_candidate(row)
        runs = int(row[4] or 0)
        why = "never played" if runs == 0 else "been a while"
        cand["subtitle"] = (
            f"{cand['subtitle']} · {why}" if cand["subtitle"] else why
        )
        payloads.append(cand)

    refs = getattr(session, "referents", None)
    if not payloads:
        if refs is not None:
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak=(
                "Your game shelf is empty right now — nothing pinned "
                "and no ROMs. Want me to open the Games browser so you "
                "can grab something?"
            ),
        )

    if refs is not None:
        refs.pending_candidates = payloads
        refs.pending_candidates_at = _time.time()
    log.info(
        "game_recommend_offer",
        user_id=session.user_id, n=len(payloads),
    )
    return ActionResult(
        short_circuit=True,
        speak=_speak_for_recommend(payloads),
        surface_emit={
            "channel": "companion.candidates",
            "payload": {
                "intent": "game.recommend",
                "query": "",
                "candidates": payloads,
            },
        },
    )


register_action(
    id="game.recommend",
    summary=(
        "Recommend a game from their OWN shelf when they ask to play "
        "WITHOUT naming a title. Reads the library (pinned games, "
        "emulator ROMs, app builds) plus play history — unplayed "
        "backlog first, then least-recently-played. Speaks the options "
        "and shows tappable cards; accepting one (tap or 'the second "
        "one') launches it. Sibling: a NAMED game is game.play."
    ),
    examples=[
        "what should I play", "recommend me a game",
        "got any games I haven't tried", "pick a game for me",
        "something to play for a bit", "find me a game",
        "find a game for me",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_game_recommend,
    delivery="artifact",
)
