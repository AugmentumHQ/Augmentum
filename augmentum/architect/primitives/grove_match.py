"""grove.play_matching — architect-callable music playback by query.

Proof primitive for the architect pattern. When the user says "play
jazz" or "put on some Miles Davis", this action:

  1. Matches via Tier 1 patterns (or Tier 3 LLM tool call).
  2. Runs ``_infer_grove_args`` to find a matching track from the
     user's device_play_history (favourites first, then recents).
  3. Emits ``grove.play_matching`` surface event with the chosen track.
  4. Speaks a short confirmation: "Putting on <label>."

If no history matches the query, the action returns a clarifying
ActionResult rather than picking a random fallback — better to ask
than to surprise.

Wiring:
  * surfaces=['voice', 'chat'] — voice command or text "/play jazz".
  * fanout: tier1 + tier3 — discoverable via direct phrase, also
    exposed to the LLM as a callable tool.
  * arg_schema: query (required), source (optional), track_id (optional).
"""

from __future__ import annotations

import random
from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Tier-3-only: LLM picks this verb + extracts the query. Previous
# Tier-1 templates with an open ``{query}`` slot after a permissive
# opener ("play X", "put on X") ate the rest of the user's utterance
# ("play Dune for me on the speakers" → query="Dune for me on the
# speakers"). The LLM understands the request shape and calls this
# verb with query="Dune", routing the surface hint separately. See
# [[no-regex-switchboard]].
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


# Keyword → genre/mood tag mapping. Bare keywords ("jazz", "lo-fi")
# match against content_label and the extra JSON. Multi-word queries
# ("miles davis") match as substring on content_label.
#
# Keys mirror Grove's discovery-panel chip set (``GENRE_CHIPS`` in
# ``ui/scripts/grove.js`` line 68): Ambient, Lo-fi, Electronic,
# Classical, Jazz, Focus, Nature, Synthwave. Plus a handful of
# user-natural genres that Grove doesn't surface as chips but may
# appear in favorites or external libraries (Rock, Hip-Hop, Folk).
#
# Lookup is normalized: lowercase + collapse hyphens/spaces so
# "Lo-fi", "lo fi", and "lofi" all hit the same key.
_GENRE_SYNONYMS: dict[str, list[str]] = {
    # Grove chip set — exact + close variants
    "jazz":       ["jazz", "swing", "bebop", "fusion", "smooth jazz", "bossa nova"],
    "lofi":       ["lofi", "lo-fi", "lo fi", "chillhop", "study beats", "chill beats"],
    "electronic": ["electronic", "edm", "techno", "house", "drum and bass", "dnb", "trance"],
    "classical":  ["classical", "baroque", "symphony", "orchestral", "piano", "chamber"],
    "ambient":    ["ambient", "drone", "soundscape", "soundscapes", "atmospheric"],
    "focus":      ["focus", "study", "concentration", "deep work", "productivity", "work music"],
    "nature":     ["nature", "rain", "ocean", "forest", "birds", "water", "thunderstorm", "white noise"],
    "synthwave":  ["synthwave", "vaporwave", "retrowave", "outrun", "80s synth", "darksynth"],
    # User-natural extras (not Grove chips but commonly requested)
    "rock":       ["rock", "metal", "punk", "indie rock", "alternative"],
    "hiphop":     ["hip hop", "hip-hop", "hiphop", "rap", "trap"],
    "folk":       ["folk", "acoustic", "singer-songwriter", "americana", "bluegrass"],
}


def _normalize_genre_key(query: str) -> str:
    """Normalize a query to look up in ``_GENRE_SYNONYMS``.

    Collapse hyphens + ALL whitespace + lowercase so "Lo-fi", "lo fi",
    "Lofi", and "lofi" all hit the same key. "Hip Hop", "hip-hop", and
    "hiphop" likewise. Returns empty when the query is empty.
    """
    s = (query or "").strip().lower()
    if not s:
        return ""
    # Drop hyphens / underscores / ALL whitespace so multi-word
    # genres normalize to a single token ("lo fi" -> "lofi",
    # "hip hop" -> "hiphop", "drum and bass" -> "drumandbass").
    s = s.replace("-", "").replace("_", "")
    s = "".join(s.split())
    return s


# Reverse-lookup map built at import: every synonym -> the canonical
# key it expands FROM. So when the user says "rain" (a synonym under
# the ``nature`` key), we still expand to the full nature list.
# Computed once; module-level so re-import cost is paid only at
# startup (or on test reload).
_SYNONYM_TO_KEY: dict[str, str] = {}
for _canon, _syns in _GENRE_SYNONYMS.items():
    _SYNONYM_TO_KEY[_canon] = _canon
    for _syn in _syns:
        _normed = _syn.lower().replace("-", "").replace("_", "")
        _normed = "".join(_normed.split())
        # First-write-wins so a synonym that appears under two
        # canonical keys binds to the earlier one (registration
        # order is intentional in ``_GENRE_SYNONYMS``).
        _SYNONYM_TO_KEY.setdefault(_normed, _canon)


# Preference-class queries — the user is delegating the choice. "Play
# something random" isn't a search term, it's an invitation: her pick.
# Matching is token-based after stripping the media nouns, so "random
# music", "anything", "whatever you like", "surprise me", and "your
# choice" all land here. A companion that answers "I can't do random"
# is failing the brief — there are hundreds of stations; taste is the
# feature (Matt, 2026-06-10).
_PREFERENCE_TOKENS = frozenset({
    "random", "anything", "something", "whatever", "surprise",
    "dealers", "dealer's", "your", "choice", "pick", "choose",
    "you", "like", "want", "me", "good", "nice", "fun",
})
_MEDIA_NOUNS = frozenset({
    "music", "song", "songs", "track", "tracks", "tune", "tunes",
    "station", "stations", "radio", "playlist", "some", "a", "an",
    "the", "to", "listen",
})


def _is_preference_query(query: str) -> bool:
    """True when the query delegates the choice to her."""
    tokens = [
        t for t in query.lower().replace("-", " ").split()
        if t and t not in _MEDIA_NOUNS
    ]
    if not tokens:
        # "some music" / "a song" — pure media noun ask, no constraint.
        return True
    return all(t.strip(".,!?'") in _PREFERENCE_TOKENS for t in tokens)


def _her_genre_pick() -> str:
    """Pick a genre as her own preference — used when the user
    delegates and history has nothing to sample from. Random over the
    canonical genre keys; variety IS the point."""
    return random.choice(list(_GENRE_SYNONYMS.keys()))


def _expand_query(query: str) -> list[str]:
    """Return search terms for the query — including synonyms for
    common genre tags. Used to filter device_play_history results.

    Three-pass lookup:
      1. Normalize the WHOLE query and check the synonym reverse-map
         ("lo fi" → lofi's full synonym list).
      2. Scan individual TOKENS for genre words — the model often
         wraps the genre in mood language ("warm grays single
         instrument ambient jazz", observed live 2026-06-10); the
         genre buried inside must still match a favourite whose label
         merely contains "jazz".
      3. Always include the raw query for artist/title substring
         fallback ("miles davis", "deftones").
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    expanded: list[str] = [q]
    seen: set[str] = {q}

    def _extend(canonical: str) -> None:
        for term in _GENRE_SYNONYMS.get(canonical, ()):
            if term not in seen:
                expanded.append(term)
                seen.add(term)

    whole = _SYNONYM_TO_KEY.get(_normalize_genre_key(q))
    if whole:
        _extend(whole)
    for token in q.replace("-", " ").split():
        token_key = _SYNONYM_TO_KEY.get(_normalize_genre_key(token))
        if token_key:
            _extend(token_key)
    return expanded


def genre_tokens_in(query: str) -> list[str]:
    """Canonical genre keys mentioned anywhere in *query* — used by
    the frontend-facing payload so station discovery can retry with
    the clean genre when the full phrase finds nothing."""
    q = (query or "").strip().lower()
    if not q:
        return []
    out: list[str] = []
    whole = _SYNONYM_TO_KEY.get(_normalize_genre_key(q))
    if whole:
        out.append(whole)
    for token in q.replace("-", " ").split():
        key = _SYNONYM_TO_KEY.get(_normalize_genre_key(token))
        if key and key not in out:
            out.append(key)
    return out


async def _conn_from_runtime(runtime: Any) -> Any:
    """Best-effort: pull the aiosqlite connection from the runtime.

    The runtime exposes ``state_manager.backend.conn`` when the SQLite
    backend is in use. Returns None when the runtime is missing or the
    backend isn't SQLite.
    """
    if runtime is None:
        return None
    sm = getattr(runtime, "state_manager", None)
    if sm is None:
        # Some runtimes hang state_manager off app.state instead.
        app_state = getattr(runtime, "_app_state", None)
        if app_state is not None:
            sm = getattr(app_state, "state_manager", None)
    if sm is None:
        return None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None) if backend else None


async def _infer_grove_args(
    partial_args: dict[str, Any],
    session: SessionContext,
    runtime: Any,
) -> dict[str, Any]:
    """Fill source + track from device_play_history when missing.

    Strategy:
      1. Pull the user's recent music plays (favourites first).
      2. Filter by query terms against content_label.
      3. Pick the top match → fill source + track_id + content_label.

    If history has nothing matching, leaves args partial — the handler
    will return a clarifying response asking how the user wants the
    request resolved (search providers, pick a different track, etc.).
    """
    from augmentum.architect.inference import query_play_history

    args = dict(partial_args)
    query = (args.get("query") or "").strip()
    if not query:
        return args

    # Explicit track_id wins — caller already knows what they want.
    if args.get("track_id"):
        return args

    conn = await _conn_from_runtime(runtime)
    if conn is None or not session.user_id:
        return args

    history = await query_play_history(
        conn,
        session.user_id,
        content_kind="music",
        limit=40,
        favourites_first=True,
    )

    if not history:
        return args

    # Preference-class query — the user delegated the choice. Sample
    # from their history (favourites first, but rotate among the top
    # handful so repeat asks don't always land on the same track).
    if _is_preference_query(query):
        pool = [r for r in history[:8] if (r.get("content_label") or "").strip()]
        if pool:
            pick = random.choice(pool)
            args["source"] = "device_play_history"
            args["track_id"] = pick.get("file_id") or pick.get("content_key") or ""
            args["content_label"] = pick.get("content_label") or ""
            args["capability_id"] = pick.get("capability_id") or ""
            args["assistant_pick"] = True
        return args

    terms = _expand_query(query)
    if not terms:
        return args

    # Filter for any row whose content_label contains any expanded term.
    # First favourites, then recents — query_play_history already orders.
    matches = []
    for row in history:
        label = (row.get("content_label") or "").lower()
        if not label:
            continue
        if any(term in label for term in terms):
            matches.append(row)

    if not matches:
        return args

    pick = matches[0]
    args["source"] = "device_play_history"
    args["track_id"] = pick.get("file_id") or pick.get("content_key") or ""
    args["content_label"] = pick.get("content_label") or query
    args["capability_id"] = pick.get("capability_id") or ""
    return args


# Courtesy fillers the regex didn't catch — stripped defensively before
# the query reaches grove. Mirrors image_defaults._strip_tail_fillers;
# duplicated here so each primitive stays self-contained.
_TAIL_FILLERS = (
    ", please", " please", " for me", " thanks", " thank you",
    ", thanks", ", thank you",
)


def _strip_tail_fillers(text: str) -> str:
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


async def _grove_play_matching_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    """Run the play_matching primitive.

    Refuses anon users (empty user_id) per the multi-tenant invariant.
    Always emits a surface event for the frontend grove module to act
    on; the actual playback happens client-side.
    """
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,   # refused — keep the honest line, not a play confirmation
            speak="I can't play music for a signed-out session.",
        )

    query = _strip_tail_fillers(args.get("query") or "")
    if not query:
        # Park the intent so the ANSWER fills the slot — the next turn
        # ("jazz") dispatches this verb directly instead of re-deriving.
        import time as _t
        refs = getattr(session, "referents", None)
        if refs is not None:
            refs.pending_intent = {
                "action_id": "grove.play_matching",
                "args": {},
                "missing": ["query"],
                "question": "What would you like to play?",
                "asked_at": _t.time(),
            }
        return ActionResult(
            short_circuit=True,
            fulfilled=False,   # parked a question — don't let the router voice "Playing…"
            speak="What would you like to play?",
        )

    label = args.get("content_label") or query
    track_id = args.get("track_id") or ""
    source = args.get("source") or ""

    # No server-side history match. Two flavors:
    #
    # Preference-class ("random", "whatever you like") — the user
    # delegated. She PICKS: a genre of her own, spoken as her call.
    # The frontend resolves it against favourites first, then falls
    # through to station discovery — there are hundreds of stations,
    # so this virtually always plays something. Never "I can't do
    # random": taste is the feature.
    #
    # Specific query — emit the surface event so grove.js can search
    # the user's local favourites (localStorage — invisible to
    # device_play_history), then station discovery. Speak commits to
    # the search; the frontend handles the resolve.
    if not track_id:
        if _is_preference_query(query):
            genre = _her_genre_pick()
            log.info(
                "grove_play_matching_assistant_pick",
                user_id=session.user_id, query=query[:80], genre=genre,
            )
            refs = getattr(session, "referents", None)
            if refs is not None:
                refs.last_played_track = genre
                refs.last_played_query = query[:80]
            return ActionResult(
                short_circuit=True,
                speak=f"My pick — let's do some {genre}.",
                surface_emit={
                    "channel": "grove.play",
                    "payload": {
                        "track_id": "",
                        "label": genre,
                        "source": "assistant_pick",
                        "query": genre,
                        "discover_ok": True,
                    },
                },
            )
        genres = genre_tokens_in(query)
        log.info(
            "grove_play_matching_no_server_match",
            user_id=session.user_id, query=query[:80], genres=genres,
        )
        return ActionResult(
            short_circuit=True,
            speak=f"Looking for {query}.",
            surface_emit={
                "channel": "grove.play",
                "payload": {
                    "track_id": "",
                    "label": query,
                    "source": "frontend_favorites_only",
                    "query": query,
                    "discover_ok": True,
                    # Clean genre keys buried in the phrasing — the
                    # frontend retries discovery with these when the
                    # full phrase finds nothing (station search is
                    # literal-minded; "warm grays ambient jazz" ≠ a
                    # station name, but "jazz" matches plenty).
                    "genre_hints": genres,
                },
            },
        )

    log.info(
        "grove_play_matching",
        user_id=session.user_id,
        query=query[:80],
        label=label[:80],
        source=source,
    )

    # Record the played track on the referent cache so follow-up
    # references ("next track", "louder", "what's playing") resolve
    # against it without re-asking the user.
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_played_track = label[:200]
        refs.last_played_query = query[:80]

    # Delegated choices get owned as hers — "my pick" reads as taste,
    # "putting on <your query>" reads as fetch-execution.
    spoken = (
        f"My pick — putting on {label}."
        if args.get("assistant_pick")
        else f"Putting on {label}."
    )

    return ActionResult(
        short_circuit=True,
        speak=spoken,
        surface_emit={
            "channel": "grove.play",
            "payload": {
                "track_id": track_id,
                "label": label,
                "source": source,
                "query": query,
            },
        },
    )


register_action(
    id="grove.play_matching",
    summary=(
        "Play MUSIC — by genre, artist, mood, or song title. Picks "
        "from the user's favourites and recent plays when possible; "
        "falls back to asking when nothing matches. Siblings: a named "
        "non-music library item (audiobook, podcast, video, comic) is "
        "media.play; 'continue what I was on' is media.resume."
    ),
    # First 6 examples feed the roster relevance ranker at 2x weight --
    # keep the high-traffic phrasings ("music", "throw/put on") inside
    # that window. "throw in some music and open a note" deferred this
    # verb below five note.* verbs because only ONE example carried
    # the word "music" (2026-06-11).
    examples=[
        "put on some music",
        "throw on some music",
        "play jazz",
        "put on something jazzy",
        "put on some lofi",
        "play miles davis",
        # Outside the 2x window from here.
        "play classical music",
    ],
    handler=_grove_play_matching_handler,
    delivery="artifact",
    fanout=_TIER3_ONLY,
    arg_schema={
        "query": {
            "type": "string",
            "description": "Genre, artist, or title to play.",
        },
        "source": {
            "type": "string",
            "description": (
                "Optional source override — 'device_play_history' for "
                "favourites, 'search' to query providers."
            ),
        },
        "track_id": {
            "type": "string",
            "description": "Optional explicit track id (file_id or content_key).",
        },
    },
    required=["query"],
    # Scoped to Becca's companion mode (widget PTT + wake) and chat,
    # NOT the full-screen voice call modal. During a voice call the
    # avatar + transcript modal is up; starting Grove playback would
    # fight the call's TTS and the user wouldn't see the now-playing
    # chip anyway. The companion widget IS the right surface for this.
    surfaces=["becca", "chat"],
    stakes="disruptive",
    arg_inferrer=_infer_grove_args,
)
