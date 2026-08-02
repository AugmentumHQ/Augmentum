"""Catalog-grounded recommendations — the always-available base.

``entity_recommender`` (Gate 1) anchors picks to what the user has
*played*: the next unfinished volume, more by an author they're into.
That's the strongest signal when it exists — but it's empty for a fresh
library, and an ask like "recommend a funny movie I haven't seen" needs
the *catalog itself*, not playback history, as the substrate.

This module makes the user's OWN catalog (``file_index``) the base every
recommendation draws from, filtered by the criteria the model extracted
from the request (type / genre / year / free-text theme) and RE-RANKED
by play-history taste when any exists. Play-history becomes a ranking
*boost*, never a *gate*: she always has something real to offer from day
one, and it sharpens the more the user consumes.

``recommend_picks`` is the unified entry point both the
``media_recommendations`` tool and the ``media.recommend`` verb call:
play-history continuations first (highest value when they fit the ask),
then catalog-grounded taste-ranked picks fill the rest. Every pick shares
the :class:`~augmentum.discovery.entity_recommender.EntityPick` shape so
the companion renders them identically to Gate-1 picks.

Picks remain catalog-grounded: the ``why`` is composed AT QUERY TIME from
real metadata (series / author / shared genres), never free-associated.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.discovery.entity_recommender import EntityPick, top_entity_picks
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_FINISHED_PCT = 95.0

# Playable file-index kinds — the only rows the player stack can start.
# Images, archives, code are never recommendation targets.
_PLAYABLE_KINDS = frozenset({"audio", "video", "comic", "document"})

# When the request constrains a broad file ``kind`` but not a specific
# ``entity_kind``, widen to the entity kinds that live under it so a
# "recommend a movie" (kind=video) still catches both movies and shows
# when the model didn't disambiguate.
_ENTITY_KINDS_BY_KIND: dict[str, list[str]] = {
    "audio": ["book", "podcast"],
    "video": ["movie", "series"],
    "comic": ["manga", "comic"],
    "document": ["book"],
}

# Natural content word the model puts in the ``kind`` arg → (file-index
# ``kind`` family, ``entity_kind``). entity_kind is "" when the word
# doesn't disambiguate within its family (e.g. "video" spans movie+
# series). Search-filter selection on an LLM-extracted arg, NOT transcript
# pattern matching ([[no-regex-switchboard]]).
_KIND_WORDS: dict[str, tuple[str, str]] = {
    "audiobook": ("audio", "book"),
    "audiobooks": ("audio", "book"),
    "podcast": ("audio", "podcast"),
    "podcasts": ("audio", "podcast"),
    "music": ("audio", ""),
    "comic": ("comic", "comic"),
    "comics": ("comic", "comic"),
    "manga": ("comic", "manga"),
    "video": ("video", ""),
    "movie": ("video", "movie"),
    "movies": ("video", "movie"),
    "film": ("video", "movie"),
    "films": ("video", "movie"),
    "show": ("video", "series"),
    "shows": ("video", "series"),
    "tv": ("video", "series"),
    "series": ("video", "series"),
    "book": ("document", "book"),
    "books": ("document", "book"),
    "ebook": ("document", "book"),
    "novel": ("document", "book"),
}


def kind_word_to_filters(word: str) -> tuple[str, str]:
    """(file_index kind, entity_kind) for a natural content word; ('', '')
    when unknown. Both callers map the model's ``kind`` arg through here so
    the vocabulary stays in one place."""
    return _KIND_WORDS.get((word or "").strip().lower(), ("", ""))


# entity_kind → the file ``kind`` family it lives under, for filtering
# play-history picks (which carry a file kind, not an entity_kind) by a
# requested entity_kind. ``book`` is ambiguous (audio vs document), so it
# maps to None — don't filter play-history on it.
_ENTITY_KIND_FAMILY: dict[str, str | None] = {
    "movie": "video",
    "series": "video",
    "manga": "comic",
    "comic": "comic",
    "podcast": "audio",
    "book": None,
}


def _meta(row_meta: Any) -> dict:
    if isinstance(row_meta, dict):
        return row_meta
    try:
        return json.loads(row_meta or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _consumed(meta: dict) -> bool:
    if meta.get("is_finished"):
        return True
    try:
        return float(meta.get("progress_pct") or 0.0) >= _FINISHED_PCT
    except (TypeError, ValueError):
        return False


def _started(meta: dict) -> bool:
    try:
        return float(meta.get("progress_pct") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


async def _q(conn: Any, sql: str, params: tuple) -> list:
    try:
        cur = await conn.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return list(rows)
    except Exception:
        log.warning("catalog_recommender_query_failed", exc_info=True)
        return []


async def _user_taste(conn: Any, *, user_id: str) -> dict[str, set]:
    """Series / creators / genres the user has engaged with, drawn from
    play-history entity clusters. Cheap (one indexed read); powers the
    catalog re-rank boost. Empty sets on a fresh library — the caller
    degrades to recency-ordered catalog picks."""
    taste = {"series": set(), "creators": set(), "genres": set()}
    if not user_id or conn is None:
        return taste
    rows = await _q(
        conn,
        """SELECT entity_ref FROM interest_clusters
           WHERE user_id = ? AND kind = 'entity' AND entity_ref != ''
           ORDER BY (frecency_short * 0.6 + frecency_long * 0.4) DESC,
                    updated_at DESC
           LIMIT 40""",
        (user_id,),
    )
    for (ref_raw,) in rows:
        ent = _meta(ref_raw)
        sn = str(ent.get("series_name") or "").strip().lower()
        if sn:
            taste["series"].add(sn)
        for c in ent.get("creators") or []:
            cs = str(c).strip().lower()
            if cs:
                taste["creators"].add(cs)
        for g in ent.get("genres") or []:
            gs = str(g).strip().lower()
            if gs:
                taste["genres"].add(gs)
    return taste


def _catalog_why(meta: dict, taste: dict[str, set], *, started: bool) -> tuple[str, str]:
    """(relation, speakable why) for a catalog pick — grounded in real
    metadata, personalized when taste overlaps, honest when it doesn't."""
    series = str(meta.get("series_name") or "").strip()
    author = str(meta.get("author") or "").strip()
    genres = [str(g) for g in (meta.get("genres") or []) if str(g).strip()]
    if started:
        where = f"in {series}" if series else "in your library"
        return "continuation", f"you're partway into this — {where}, ready to resume"
    series_l = series.lower()
    author_l = (
        str(meta.get("author_normalized") or "").strip().lower() or author.lower()
    )
    egenres = {g.lower() for g in genres}
    if series_l and series_l in taste["series"]:
        return "for_you", f"in {series}, which you've been into — never started"
    if author_l and author_l in taste["creators"]:
        disp = author or "the same author"
        return "same_creator", f"by {disp}, who you've enjoyed — unread in the library"
    overlap = sorted(egenres & taste["genres"])
    if overlap:
        return "for_you", (
            f"matches your taste — {', '.join(overlap[:2])} — and you "
            "haven't started it"
        )
    if genres:
        return "fresh", f"new in your library — {', '.join(genres[:2])}"
    return "fresh", "ready in your library, never started"


def _taste_score(meta: dict, taste: dict[str, set], *, started: bool) -> float:
    """Re-rank weight. Taste overlap dominates; in-progress is a strong
    nudge (resuming is almost always wanted). All zero → recency order
    (the SQL already returns newest-first) carries the result."""
    score = 0.0
    series_l = str(meta.get("series_name") or "").strip().lower()
    author_l = (
        str(meta.get("author_normalized") or "").strip().lower()
        or str(meta.get("author") or "").strip().lower()
    )
    egenres = {str(g).strip().lower() for g in (meta.get("genres") or [])}
    if series_l and series_l in taste["series"]:
        score += 3.0
    if author_l and author_l in taste["creators"]:
        score += 2.0
    score += min(len(egenres & taste["genres"]), 3) * 1.0
    if started:
        score += 1.5
    return score


async def catalog_picks(
    file_index: Any,
    *,
    user_id: str,
    kind: str | None = None,
    entity_kind: str | None = None,
    genre: str = "",
    year_from: int = 0,
    year_to: int = 0,
    query: str = "",
    taste: dict[str, set] | None = None,
    limit: int = 4,
) -> list[EntityPick]:
    """Catalog-grounded picks straight from ``file_index``, filtered by the
    request criteria and re-ranked by ``taste``. Never raises — degrades to
    an empty list on any failure. Prefers never-started items; broadens to
    the whole (unfinished) catalog only if the unwatched slice is empty."""
    if file_index is None or not user_id:
        return []
    taste = taste or {"series": set(), "creators": set(), "genres": set()}
    pool = max(limit * 8, 24)

    if entity_kind:
        ek_single: str | None = entity_kind
        ek_list: list[str] | None = None
    elif kind:
        ek_single = None
        ek_list = _ENTITY_KINDS_BY_KIND.get(kind)
    else:
        ek_single = None
        ek_list = None

    async def _fetch(status: str | None) -> list:
        kw = dict(
            user_id=user_id,
            kind=kind or None,
            entity_kind=ek_single,
            entity_kinds=ek_list,
            genre=genre or "",
            year_from=year_from,
            year_to=year_to,
            media_status=status,
            limit=pool,
        )
        try:
            if query:
                return await file_index.search(query, **kw) or []
            return await file_index.list_recent(sort="newest", **kw) or []
        except Exception:
            log.warning("catalog_picks_fetch_failed", exc_info=True)
            return []

    entries = await _fetch("not_started")
    if not entries:
        # Maybe everything matching is already in-progress (not finished) —
        # broaden and let the python-side _consumed filter drop finished.
        entries = await _fetch(None)

    scored: list[tuple[float, Any, dict, str, bool]] = []
    for e in entries:
        fid = (getattr(e, "id", "") or "").strip()
        if not fid or getattr(e, "is_directory", False):
            continue
        ekind = (getattr(e, "kind", "") or "").strip()
        if ekind not in _PLAYABLE_KINDS:
            continue
        meta = getattr(e, "source_metadata", None) or {}
        if not isinstance(meta, dict):
            meta = _meta(meta)
        if _consumed(meta):
            continue
        started = _started(meta)
        scored.append((_taste_score(meta, taste, started=started), e, meta, ekind, started))

    # Stable sort: taste score desc, recency (SQL newest-first) breaks ties.
    scored.sort(key=lambda t: -t[0])

    picks: list[EntityPick] = []
    for _score, e, meta, ekind, started in scored[:limit]:
        relation, why = _catalog_why(meta, taste, started=started)
        picks.append(EntityPick(
            relation=relation, gate=1,
            file_id=(getattr(e, "id", "") or ""),
            title=(getattr(e, "name", "") or "")[:120],
            kind=ekind, why=why,
            creator=str(meta.get("author") or ""),
            series_name=str(meta.get("series_name") or ""),
        ))
    return picks


def _pick_matches_request(
    pick: EntityPick,
    *,
    kind: str | None,
    entity_kind: str | None,
    genre: str,
) -> bool:
    """Whether a play-history pick fits an explicitly-constrained request.

    Play-history picks carry a file ``kind`` but no entity_kind/genre, so a
    genre-specific ask ("a comedy") can't be verified against them — defer
    those entirely to the catalog leg (which filters genre in SQL). Kind /
    entity_kind family is checkable, so enforce it."""
    if genre:
        return False
    if kind and pick.kind and pick.kind != kind:
        return False
    if entity_kind:
        family = _ENTITY_KIND_FAMILY.get(entity_kind)
        if family and pick.kind and pick.kind != family:
            return False
    return True


async def recommend_picks(
    conn: Any,
    file_index: Any,
    *,
    user_id: str,
    kind: str | None = None,
    entity_kind: str | None = None,
    genre: str = "",
    year_from: int = 0,
    year_to: int = 0,
    query: str = "",
    limit: int = 4,
) -> list[EntityPick]:
    """Unified recommendation: play-history continuations first (when they
    fit the request), then catalog-grounded taste-ranked picks fill the
    rest. Always draws from the real catalog when anything matches — the
    "nothing found" case now means the *filtered catalog* is genuinely
    empty, not merely that the user hasn't played anything yet."""
    if not user_id:
        return []
    taste = await _user_taste(conn, user_id=user_id)
    picks: list[EntityPick] = []
    seen: set[str] = set()

    # 1. Play-history continuations / new arrivals — strongest when they
    #    match the requested kind. Skipped wholesale on a genre-specific
    #    ask (see _pick_matches_request).
    if conn is not None:
        try:
            groups = await top_entity_picks(
                conn, user_id=user_id, max_entities=4, limit_per_entity=3,
            )
        except Exception:
            log.warning("recommend_picks_entity_failed", exc_info=True)
            groups = []
        for _entity, eps in groups:
            for p in eps:
                if not p.file_id or p.file_id in seen:
                    continue
                if not _pick_matches_request(
                    p, kind=kind, entity_kind=entity_kind, genre=genre,
                ):
                    continue
                seen.add(p.file_id)
                picks.append(p)
                if len(picks) >= limit:
                    break
            if len(picks) >= limit:
                break

    # 2. Catalog base — fills the rest, taste-ranked. Over-fetch so dedup
    #    against the play-history leg still leaves a full dock.
    if len(picks) < limit:
        cat = await catalog_picks(
            file_index, user_id=user_id, kind=kind, entity_kind=entity_kind,
            genre=genre, year_from=year_from, year_to=year_to, query=query,
            taste=taste, limit=limit * 3,
        )
        for p in cat:
            if not p.file_id or p.file_id in seen:
                continue
            seen.add(p.file_id)
            picks.append(p)
            if len(picks) >= limit:
                break

    return picks[:limit]
