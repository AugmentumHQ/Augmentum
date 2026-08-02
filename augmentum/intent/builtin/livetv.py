"""Live TV companion verbs — tune to a channel and browse what's on.

Part of the 2026-07-19 companion content-coverage pass: live TV was the
largest content family with ZERO companion-reachable path after games.
The Files panel's Live TV chip has a working rail browser + HLS player,
but only the Library's tile-click could reach it. These verbs give the
companion the same reach through the shared stream proxy
(``/api/livetv/play``), emitted on the ``livetv.tune`` channel.

Same design contract as media.py and games.py (Companion Direct Action,
2026-06-10): LLM-orchestrated Tier 3 only — no regex switchboard on
"tune to" / "watch"; a clear channel match tunes via the HLS overlay;
near-ties surface as tappable candidate cards; a miss is owned honestly.

Channel resolution is by name (substring, fuzzy) and by channel number
(exact on major or major.minor). Matches are scoped to the user's own
Emby/JF servers — the same set the Files panel's Live TV chip sees.
"""

from __future__ import annotations

import re
import time as _time
from difflib import SequenceMatcher
from typing import Any

import httpx

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.media.livetv_rails import categorize_channels
from augmentum.media.providers.base import CatalogItem
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.emby_compat import EmbyCompatBase
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.store import MediaServerStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


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


def _http_of(session: SessionContext) -> httpx.AsyncClient | None:
    return getattr(session.app_state, "http_client", None) if session.app_state else None


def _provider_for(name: str, http: httpx.AsyncClient) -> EmbyCompatBase | None:
    if name == "emby":
        return EmbyProvider(http)
    if name == "jellyfin":
        return JellyfinProvider(http)
    return None


# ── Channel fetching (mirrors livetv_routes._fetch_for_server) ──────

async def _fetch_channels_for_server(
    server, http: httpx.AsyncClient,
) -> list[CatalogItem]:
    """Per-server channel fetch with isolated error handling.

    A broken server (DNS gone, token expired) must not take out the
    whole resolution — other servers still contribute channels.
    """
    if not server.access_token:
        return []
    client = _provider_for(server.provider, http)
    if client is None:
        return []
    try:
        items = await client.fetch_live_channels(server.base_url, server.access_token)
    except Exception:
        log.warning(
            "livetv_verb_fetch_failed",
            server_id=server.id, provider=server.provider, exc_info=True,
        )
        return []
    # Tag with originating server so the play path can route back.
    for item in items:
        if isinstance(item.extra, dict):
            item.extra["server_id"] = server.id
    return items


async def _fetch_all_channels(
    conn, user_id: str, http: httpx.AsyncClient,
) -> list[CatalogItem]:
    """Fetch live channels from every Emby/JF server the user can see."""
    store = MediaServerStore(conn)
    servers = await store.list_visible(user_id=user_id)
    live_capable = [s for s in servers if s.provider in ("emby", "jellyfin")]
    if not live_capable:
        return []

    channels: list[CatalogItem] = []
    for s in live_capable:
        batch = await _fetch_channels_for_server(s, http)
        channels.extend(batch)
    return channels


# ── Channel matching ────────────────────────────────────────────────

# Filler words voice asks carry ("watch the news channel please") —
# stripped before scoring. Deliberately conservative: "movie" / "sports"
# are real channel-name words (ESPN is literally a sports channel).
# This set mirrors the resolver's _FILLER_TOKENS but adds TV-specific
# filler ("watch", "channel", "tune", "turn").
_TV_STOPWORDS = frozenset({
    "a", "an", "the", "one", "of", "some", "any", "that", "this",
    "my", "me", "please", "play", "playing", "version", "like",
    "watch", "watching", "channel", "channels", "tune", "turn",
    "to", "on", "for", "put",
})


def _norm_tv_tokens(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [t for t in s.split() if t and t not in _TV_STOPWORDS]


def _score_channel(query: str, channel: CatalogItem) -> float:
    """Normalized fuzzy match in [0, 1] against channel name + number.

    Name match: stopword-stripped exact > substring > per-token fuzzy.
    Number match: exact major or full channel-number match is a hard
    1.0 — "tune to 6" means channel 6, zero ambiguity.
    """
    name = channel.name
    extra = channel.extra if isinstance(channel.extra, dict) else {}
    ch_num = str(extra.get("channel_number") or "").strip()

    q = _norm_tv_tokens(query)
    if not q:
        return 0.0

    # --- Number match ---
    # Exact match on bare channel number (e.g. "6" → channel 6, "6.2" → 6.2)
    # or on the query AS a number when the user literally said "channel six point two".
    qs = " ".join(q)
    if ch_num:
        if qs == ch_num:
            return 1.0
        # Major-only match: "6" matches channel 6.1, 6.2, etc.
        if ch_num.startswith(qs + ".") or (len(q) == 1 and q[0].isdigit() and ch_num.split(".")[0] == q[0]):
            return 0.95

    # --- Name match ---
    n = _norm_tv_tokens(name)
    if not n:
        return 0.0
    ns = " ".join(n)

    if qs == ns:
        return 1.0
    if qs in ns or ns in qs:
        return 0.92

    # Short number-only queries already handled above; if we're here
    # with a short numeric query and no name match, it's not this channel.
    if all(t.isdigit() for t in q) and len(q) <= 2:
        return 0.0

    # Per-token best-of (exact / prefix / SequenceMatcher)
    total = 0.0
    for qt in q:
        best = 0.0
        for nt in n:
            if qt == nt:
                r = 1.0
            elif nt.startswith(qt) or qt.startswith(nt):
                r = 0.88
            else:
                r = SequenceMatcher(None, qt, nt).ratio()
            best = max(best, r)
        total += best
    return 0.75 * (total / len(q))


def _candidate_from_channel(ch: CatalogItem) -> dict[str, Any]:
    """Build a companion.candidates card payload from a CatalogItem.

    Carries server_id + channel_id + name so the frontend can POST to
    /api/livetv/play without re-querying. The dock's artifact_id gate
    doesn't apply here — this is a file_id-free candidate, so the card
    renders with the channel name + number badge.
    """
    extra = ch.extra if isinstance(ch.extra, dict) else {}
    ch_num = str(extra.get("channel_number") or "").strip()
    prog = extra.get("current_program") or {}
    prog_name = str(prog.get("name") or "") if isinstance(prog, dict) else ""
    return {
        "channel_id": ch.external_id,
        "server_id": str(extra.get("server_id") or ""),
        "title": ch.name,
        "subtitle": f"Ch {ch_num}" + (f" — {prog_name}" if prog_name else ""),
        "kind": "live_video",
        "content_kind": "live_tv",
        "channel_number": ch_num,
        "current_program": prog_name,
    }


# ── Verb handlers ───────────────────────────────────────────────────

def _tune_result(ch: CatalogItem) -> ActionResult:
    """Emit a livetv.tune surface event so the frontend opens the HLS
    player overlay. Same flow as clicking a channel tile in Files."""
    extra = ch.extra if isinstance(ch.extra, dict) else {}
    return ActionResult(
        short_circuit=True,
        speak=f"Tuning to {ch.name}.",
        toast=f"Tuning to {ch.name}"[:80],
        surface_emit={
            "channel": "livetv.tune",
            "payload": {
                "channel_id": ch.external_id,
                "server_id": str(extra.get("server_id") or ""),
                "name": ch.name,
            },
        },
    )


def _speak_for_offer(query: str, cands: list[dict]) -> str:
    names = []
    for i, c in enumerate(cands[:3], start=1):
        num = c.get("channel_number", "")
        label = f"{c['title']} (Ch {num})" if num else c["title"]
        names.append(f"{i}: {label}")
    return (
        f"I found a few channels matching {query} — {'; '.join(names)}. "
        "Which one?"
    )


async def _livetv_play(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't tune to a live channel for a signed-out session.",
        )
    conn = _conn_of(session)
    http = _http_of(session)
    if conn is None or http is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach the TV tuner right now.",
        )

    refs = getattr(session, "referents", None)
    query = (args.get("query") or "").strip()

    # Accept path for an offered pick — the candidate payload carries
    # channel_id and server_id verbatim so "the second one" resolves.
    channel_id = str(args.get("channel_id") or "").strip()
    server_id = str(args.get("server_id") or "").strip()
    if channel_id and server_id:
        # Ownership check: re-fetch and confirm this channel exists.
        channels = await _fetch_all_channels(conn, session.user_id, http)
        for ch in channels:
            extra = ch.extra if isinstance(ch.extra, dict) else {}
            if ch.external_id == channel_id and str(extra.get("server_id") or "") == server_id:
                if refs is not None:
                    refs.pending_candidates = []
                    refs.pending_candidates_at = 0.0
                log.info(
                    "livetv_play_offered_pick",
                    user_id=session.user_id, channel=ch.name[:60],
                )
                return _tune_result(ch)
        # Stale/foreign id — fall through to the query path.

    if not query:
        if refs is not None:
            refs.pending_intent = {
                "action_id": "livetv.play",
                "args": {},
                "missing": ["query"],
                "question": "Which channel?",
                "asked_at": _time.time(),
            }
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="Which channel?",
        )

    channels = await _fetch_all_channels(conn, session.user_id, http)
    if not channels:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak=(
                "You don't have any live TV set up yet. Connect an Emby "
                "or Jellyfin server with Live TV configured and I'll be "
                "able to tune in."
            ),
        )

    # Score and sort
    scored: list[tuple[float, CatalogItem]] = []
    for ch in channels:
        s = _score_channel(query, ch)
        if s > 0.35:
            scored.append((s, ch))
    scored.sort(key=lambda p: p[0], reverse=True)

    if not scored:
        log.info("livetv_play_miss", user_id=session.user_id, query=query[:80])
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak=(
                f"I don't see a channel matching {query[:60]}. "
                "Try a channel name or number — you can browse all "
                "channels from the Files panel under Live TV."
            ),
        )

    # Clear winner? Tune directly.
    if len(scored) == 1 or (
        scored[0][0] >= 0.85 and scored[0][0] - scored[1][0] > 0.2
    ):
        ch = scored[0][1]
        if refs is not None:
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        log.info(
            "livetv_play_direct",
            user_id=session.user_id, channel=ch.name[:60],
            score=round(scored[0][0], 3),
        )
        return _tune_result(ch)

    # Near-ties → candidate cards.
    payloads = [_candidate_from_channel(ch) for _, ch in scored[:4]]
    if refs is not None:
        refs.pending_candidates = payloads
        refs.pending_candidates_at = _time.time()
    log.info(
        "livetv_play_offer",
        user_id=session.user_id, query=query[:80], n=len(payloads),
    )
    return ActionResult(
        short_circuit=True,
        speak=_speak_for_offer(query, payloads),
        surface_emit={
            "channel": "companion.candidates",
            "payload": {
                "intent": "livetv.play",
                "query": query,
                "candidates": payloads,
            },
        },
    )


register_action(
    id="livetv.play",
    summary=(
        "Tune to a live TV channel by NAME or NUMBER from the user's "
        "Emby or Jellyfin server. A clear match opens the live HLS "
        "player; near-ties show pickable cards. Call when the user "
        "names a channel, asks to watch something live, or accepts an "
        "offered channel pick (pass its channel_id + server_id). "
        "Sibling: 'what's on TV' with no channel is livetv.browse."
    ),
    examples=[
        "tune to ESPN", "watch channel 6", "put on NBC",
        "watch the news", "play channel 6.2", "watch the second one",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "Channel name or number to tune to — the user's "
                "phrasing, preserve their words."
            ),
        },
        "channel_id": {
            "type": "string",
            "description": (
                "Exact channel external_id, ONLY when accepting an "
                "offered pick — copy it verbatim. Skips resolution."
            ),
        },
        "server_id": {
            "type": "string",
            "description": (
                "Server id that owns the channel, ONLY with channel_id "
                "— copy both verbatim from the offered pick."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_livetv_play,
    delivery="artifact",
)


# ── Browse: "what's on TV right now" ────────────────────────────────

def _speak_for_browse(rails_summary: list[str], total: int) -> str:
    if not rails_summary:
        return (
            f"You have {total} live channels. Try asking for a channel "
            "by name or number."
        )
    joined = "; ".join(rails_summary[:5])
    return (
        f"You've got {total} live channels. Here's what's on: {joined}. "
        "Want me to tune to one?"
    )


async def _livetv_browse(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't browse live TV for a signed-out session.",
        )
    conn = _conn_of(session)
    http = _http_of(session)
    if conn is None or http is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach the TV tuner right now.",
        )

    channels = await _fetch_all_channels(conn, session.user_id, http)
    refs = getattr(session, "referents", None)

    if not channels:
        if refs is not None:
            refs.pending_candidates = []
            refs.pending_candidates_at = 0.0
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak=(
                "You don't have any live TV set up yet. Connect an Emby "
                "or Jellyfin server with Live TV configured and I'll be "
                "able to browse what's on."
            ),
        )

    # Build rails same as the Files panel, then summarize what's airing.
    rails = categorize_channels(channels)
    rail_bits: list[str] = []
    highlighted: list[dict] = []
    seen_ids: set[str] = set()

    for rail in rails:
        if rail.kind == "all":
            continue  # summary only — All Channels is noise for voice
        active = []
        for ch in rail.channels[:3]:
            extra = ch.extra if isinstance(ch.extra, dict) else {}
            prog = extra.get("current_program") or {}
            prog_name = str(prog.get("name") or "") if isinstance(prog, dict) else ""
            if prog_name:
                active.append(f"{ch.name} — {prog_name}")
            if ch.external_id not in seen_ids:
                seen_ids.add(ch.external_id)
                highlighted.append(_candidate_from_channel(ch))
        if active:
            rail_bits.append(f"{rail.title}: {', '.join(active[:2])}")

    # Trim to a sane voice length — more than 5 rails is a monologue.
    top_cards = highlighted[:8] if highlighted else [
        _candidate_from_channel(ch) for ch in channels[:4]
    ]

    if refs is not None:
        refs.pending_candidates = top_cards
        refs.pending_candidates_at = _time.time()

    log.info(
        "livetv_browse",
        user_id=session.user_id, n_channels=len(channels),
        n_rails=len(rails),
    )
    return ActionResult(
        short_circuit=True,
        speak=_speak_for_browse(rail_bits, len(channels)),
        surface_emit={
            "channel": "companion.candidates",
            "payload": {
                "intent": "livetv.browse",
                "query": "",
                "candidates": top_cards,
            },
        },
    )


register_action(
    id="livetv.browse",
    summary=(
        "Browse what's on live TV right now — no channel name needed. "
        "Reads the user's Emby/JF live channels, groups by what's "
        "airing (news, sports, movies, etc.), speaks a summary, and "
        "shows tappable cards for the highlights. Accepting one tunes "
        "to it. Sibling: a NAMED channel is livetv.play."
    ),
    examples=[
        "what's on TV", "what's on right now", "browse live TV",
        "what's airing", "show me what's on",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_livetv_browse,
    delivery="artifact",
)
