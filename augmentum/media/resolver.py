"""Media resolver — turn "the dune audiobook" into a playable file_id.

Companion Direct Action spec (docs/superpowers/specs/2026-06-10-
companion-direct-action-design.md, layer L3). The ``media.play`` verb
hands the user's phrasing here; the resolver searches the unified
file index and returns a confidence-gated decision:

  * ``play``  — one clear winner. The handler emits ``media.resume``
    and playback starts in the background mini-player. No surface
    yanking.
  * ``offer`` — plausible candidates but no dominant one. The handler
    emits ``companion.candidates`` so the widget renders clickable
    cards (and Becca speaks the options on voice surfaces).
  * ``none``  — nothing in the library plausibly matches. The handler
    says so honestly and offers external options — it must NOT
    auto-play a tag coincidence.

Precision over recall (load-bearing, per Matt 2026-06-10): a wrong
auto-play costs trust; an options card costs one tap. Thresholds tune
toward abstain-and-offer. Title similarity dominates the score — an
FTS hit on description/tags alone cannot reach the play threshold.

Music/genre asks are NOT this module's job — ``grove.play_matching``
already owns that tier ladder (favourites → frontend favourites →
clarify). The ``media.play`` verb routes music-ish queries there via
its handler; this resolver covers known-item library asks
(audiobooks, podcasts, videos, comics, books).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Decision thresholds ───────────────────────────────────────────────
#
# PLAY: the top candidate's score must clear this AND lead the runner-
# up by MARGIN. OFFER: anything above this is worth showing as a card.
# Tuned conservative — see the precision rule in the module docstring.
PLAY_THRESHOLD = 0.72
PLAY_MARGIN = 0.15
OFFER_THRESHOLD = 0.35
MAX_CANDIDATES = 4

# Resume boost — an in-progress item with a matching title is almost
# always what "play X" means. Favorites and recent plays nudge too.
_BOOST_IN_PROGRESS = 0.15
_BOOST_RECENT_PLAY = 0.05
_BOOST_FAVORITE = 0.05
_RECENT_PLAY_WINDOW = timedelta(days=30)


# Kind words the user folds into the query ("the dune AUDIOBOOK").
# They select the file-index ``kind`` filter and the player's
# content_kind; the remaining words are the title query. This is
# search-filter selection on an LLM-extracted arg — not transcript
# pattern matching (see [[no-regex-switchboard]]).
_KIND_HINTS: dict[str, tuple[str, str]] = {
    # word → (file_index kind, player content_kind)
    "audiobook": ("audio", "audiobook"),
    "audiobooks": ("audio", "audiobook"),
    "podcast": ("audio", "podcast"),
    "episode": ("audio", "podcast"),
    "song": ("audio", "music"),
    "album": ("audio", "music"),
    "track": ("audio", "music"),
    "video": ("video", "video"),
    "movie": ("video", "video"),
    "film": ("video", "video"),
    "show": ("video", "video"),
    "comic": ("comic", "comic"),
    "comics": ("comic", "comic"),
    "manga": ("comic", "comic"),
    "book": ("document", "book"),
    "ebook": ("document", "book"),
    "novel": ("document", "book"),
}

# Default content_kind per file-index kind when the query carried no
# explicit hint — the player treats generic audio as audiobook-style
# resume, which is the right default for library audio.
_KIND_TO_CONTENT: dict[str, str] = {
    "audio": "audiobook",
    "video": "video",
    "comic": "comic",
    "document": "book",
}

_FILLER_TOKENS = frozenset({
    "the", "a", "an", "my", "me", "that", "this", "some", "of", "on",
    "for", "to", "please", "play", "put", "start", "listen", "watch",
    "read", "again", "by",
    # 2026-07-19: "one" and "any" were missing — "one of the dune
    # books" kept "one" in the title query, dragging the containment
    # score down. "any" same class ("any good sci-fi"). Both are
    # quantity/filler, never a title discriminator.
    "one", "any",
    # "find" / "search" / "show" — common command verbs the LLM or
    # voice routing may leave in the query. Not a title word.
    "find", "search", "show", "get", "give",
})

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Candidate:
    """One playable match, shaped for both the ``media.resume`` payload
    and the ``companion.candidates`` card renderer."""

    file_id: str
    title: str
    subtitle: str = ""        # author / series / source detail
    kind: str = ""            # file_index kind (audio/video/comic/document)
    content_kind: str = ""    # player kind (audiobook/podcast/video/…)
    source: str = ""
    score: float = 0.0
    in_progress: bool = False
    progress_pct: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "kind": self.kind,
            "content_kind": self.content_kind,
            "source": self.source,
            "score": round(self.score, 3),
            "in_progress": self.in_progress,
        }


@dataclass
class ResolveResult:
    decision: str                       # "play" | "offer" | "none"
    top: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    query: str = ""                     # cleaned title query actually searched
    kind_hint: str = ""                 # content_kind extracted from phrasing


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall((text or "").lower()) if t]


def _strip_extension(name: str) -> str:
    return re.sub(r"\.[a-z0-9]{1,5}$", "", name or "", flags=re.IGNORECASE)


def extract_kind_hint(query: str) -> tuple[str, str, str]:
    """Split a raw query into (title_query, index_kind, content_kind).

    "the dune audiobook" → ("dune", "audio", "audiobook")
    "foundation"         → ("foundation", "", "")
    """
    toks = _tokens(query)
    index_kind = ""
    content_kind = ""
    title_toks: list[str] = []
    for t in toks:
        hint = _KIND_HINTS.get(t)
        if hint and not index_kind:
            index_kind, content_kind = hint
            continue
        if t in _FILLER_TOKENS:
            continue
        title_toks.append(t)
    # All-filler queries ("play something") keep nothing — the caller
    # treats an empty title query as "ask, don't guess".
    return " ".join(title_toks), index_kind, content_kind


def _title_similarity(query_toks: list[str], name: str) -> float:
    """Title-dominant score in [0, 1].

    Containment first: every query token appearing in the name (prefix
    match allowed — "found" hits "Foundation") is the strongest signal
    a user-typed title can give. Sequence ratio fills in for partial /
    misspelled queries.
    """
    if not query_toks:
        return 0.0
    name_clean = _strip_extension(name)
    name_toks = _tokens(name_clean)
    if not name_toks:
        return 0.0

    contained = 0
    for qt in query_toks:
        if any(nt == qt or nt.startswith(qt) for nt in name_toks):
            contained += 1
    containment = contained / len(query_toks)

    ratio = SequenceMatcher(
        None, " ".join(query_toks), " ".join(name_toks),
    ).ratio()

    # Containment dominates; ratio refines. Full containment of a
    # multi-token query is near-certain intent even when the name has
    # extra words ("Dune" ⊂ "Dune — Frank Herbert (unabridged)").
    score = 0.7 * containment + 0.3 * ratio
    # Penalize one-token queries slightly — "it" shouldn't auto-play
    # anything on containment alone.
    if len(query_toks) == 1 and len(query_toks[0]) <= 3:
        score *= 0.6
    return min(1.0, score)


def _recency_boost(entry: Any) -> tuple[float, bool, float]:
    """(boost, in_progress, progress_pct) from playback metadata."""
    boost = 0.0
    meta = getattr(entry, "source_metadata", None) or {}
    progress = float(meta.get("progress_pct") or 0.0)
    finished = bool(meta.get("is_finished") or False)
    in_progress = progress > 0 and not finished
    if in_progress:
        boost += _BOOST_IN_PROGRESS
    if getattr(entry, "is_favorite", False):
        boost += _BOOST_FAVORITE
    last_played = (getattr(entry, "last_played_at", "") or "").strip()
    if last_played:
        try:
            ts = datetime.fromisoformat(last_played.replace("Z", "+00:00"))
            if datetime.now(UTC) - ts <= _RECENT_PLAY_WINDOW:
                boost += _BOOST_RECENT_PLAY
        except ValueError:
            pass
    return boost, in_progress, progress


def _subtitle_for(entry: Any) -> str:
    meta = getattr(entry, "source_metadata", None) or {}
    author = (meta.get("author") or "").strip()
    series = ((meta.get("extra") or {}).get("series") or "").strip() \
        if isinstance(meta.get("extra"), dict) else ""
    source = (getattr(entry, "source", "") or "").strip()
    bits = [b for b in (author, series) if b]
    if not bits and source:
        bits = [source]
    return " · ".join(bits)[:80]


async def resolve_media(
    app_state: Any,
    *,
    user_id: str,
    query: str,
    limit: int = 8,
) -> ResolveResult:
    """Resolve a known-item media query against the user's library.

    Never raises — failures return ``decision="none"`` with the error
    logged, so the calling verb can degrade to an honest miss.
    """
    title_query, index_kind, content_kind = extract_kind_hint(query)
    result = ResolveResult(
        decision="none", query=title_query, kind_hint=content_kind,
    )
    if not user_id or not title_query:
        return result

    file_index = getattr(app_state, "file_index", None) if app_state else None
    if file_index is None:
        log.warning("media_resolver_no_file_index")
        return result

    try:
        hits = await file_index.search(
            title_query,
            user_id=user_id,
            kind=index_kind or None,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 — resolver must not break the verb
        log.warning(
            "media_resolver_search_failed",
            error=str(exc)[:200], query=title_query[:80],
        )
        return result

    query_toks = _tokens(title_query)
    candidates: list[Candidate] = []
    for entry in hits or []:
        if getattr(entry, "is_directory", False):
            continue
        name = (getattr(entry, "name", "") or "").strip()
        file_id = (getattr(entry, "id", "") or "").strip()
        if not name or not file_id:
            continue
        entry_kind = (getattr(entry, "kind", "") or "").strip()
        # Playability filter: when the user didn't constrain kind, only
        # consider kinds the player stack can actually start (audio /
        # video / comic / document-reader). Images, archives, code —
        # never "play" targets.
        if entry_kind not in _KIND_TO_CONTENT:
            continue
        sim = _title_similarity(query_toks, name)
        if sim <= 0.0:
            continue
        boost, in_progress, progress = _recency_boost(entry)
        candidates.append(Candidate(
            file_id=file_id,
            title=_strip_extension(name)[:120],
            subtitle=_subtitle_for(entry),
            kind=entry_kind,
            content_kind=content_kind or _KIND_TO_CONTENT.get(entry_kind, ""),
            source=(getattr(entry, "source", "") or ""),
            # Uncapped: boosts must stay able to order two perfect-title
            # matches (the in-progress copy outranks the fresh one).
            # Thresholds all sit below 1.0 so the cap bought nothing.
            score=sim + boost,
            in_progress=in_progress,
            progress_pct=progress,
        ))

    if not candidates:
        return result

    candidates.sort(key=lambda c: -c.score)
    shortlist = [c for c in candidates if c.score >= OFFER_THRESHOLD]
    shortlist = shortlist[:MAX_CANDIDATES]
    if not shortlist:
        return result

    top = shortlist[0]
    runner_up = shortlist[1].score if len(shortlist) > 1 else 0.0
    if top.score >= PLAY_THRESHOLD and (top.score - runner_up) >= PLAY_MARGIN:
        decision = "play"
    else:
        decision = "offer"

    log.info(
        "media_resolver_decision",
        decision=decision,
        query=title_query[:80],
        kind_hint=content_kind,
        top_title=top.title[:80],
        top_score=round(top.score, 3),
        runner_up=round(runner_up, 3),
        n_candidates=len(shortlist),
    )
    result.decision = decision
    result.top = top
    result.candidates = shortlist
    return result
