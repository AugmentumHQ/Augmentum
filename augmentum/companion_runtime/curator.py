"""Curator — Becca as editorial layer over the user's information substrate.

Replaces ``_perform_journal`` (activity_selector.py) as the primary writer
for the notes drawer surface. Where _perform_journal asked a small utility
model to "write a noticing" from interior state — producing AI-poetry
tropes like "the weight of silence" — this module pulls actual signal from
the world:

  * Derived interests from ``interest_clusters`` (already populated by the
    discovery engine; no new substrate needed).
  * Pinned topics + explicit feed subscriptions from
    ``companion_tracked_topics`` (migration 238 — unified row shape, a row
    with feed_url=NULL is a topic-pin, a row with feed_url set is an
    RSS/feed subscription).
  * Existing feed fetchers in :mod:`augmentum.discovery.feeds`
    (HN / Reddit / arxiv / generic RSS).

For each tick (rate-limited), the curator:
  1. Picks the next due topic (oldest last_polled_at + weight tiebreak).
  2. Routes to a feed source — explicit feed_url when present, else
     programmatic source-by-shape (cs.* category → arxiv, GH/code-shape →
     HN, generic → SearXHN via existing browse routes' search wrappers).
  3. Scores returned items vs. topic keywords; rejects below threshold.
  4. Dedups against URLs Becca has already noted in the last week.
  5. Writes ONE journal entry via safe_journal — structured prose with
     a URL ref, NOT a poem. Auto-flagged quiet_share_ready by the
     memory-layer policy because affect_tag is in the surfaceable set.

Per-tick budget: at most ONE note written, even if multiple topics are
due. Volume kills meaning; deliberate cadence is the design.

Privacy class: 'local_only' for the optional LLM digest step. Feed fetches
hit the public internet via the existing http_client (which already lives
on app.state via the FastAPI lifespan).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiosqlite

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ─── Tunables (defaults; overridable via settings) ───────────────────────


# Per-runtime minimum interval between curator writes. Volume kills meaning
# — even useful notes feel meh when they arrive every minute. 30min default
# makes each note a deliberate gesture.
_STEP_DEBOUNCE_SECONDS: float = 1800.0

# Per-runtime minimum interval between curator ATTEMPTS — independent of
# whether the attempt produced a note. The write-debounce above only fires
# AFTER a successful write, so in steady state (everything already
# journaled, low-value, or safety-blocked) the expensive gather_feeds +
# SearXNG recommender pipeline would otherwise re-run every tick (5s in
# 'present' state) and burn through upstream engine rate-limit budget in
# minutes — see the 2026-06-10 cascade where qwant / brave / wikipedia /
# mojeek / karmasearch / presearch were all suspended_time=180. 10 min
# default is a compromise: short enough to pick up freshly-indexed items,
# long enough that we don't fingerprint as a bot.
_STEP_ATTEMPT_DEBOUNCE_SECONDS: float = 600.0

# Per-topic poll cooldown — don't hammer the same feed every tick.
_TOPIC_POLL_COOLDOWN_SECONDS: float = 3600.0

# Minimum relevance score for an item to qualify. Score = keyword-overlap /
# topic-keyword-count, so 0.15 means roughly one keyword overlap is enough
# for topics up to ~6 tokens. Tuned downward in 2026-06 when arxiv queries
# started passing topic terms server-side — arxiv pre-filters to candidates
# that already mention SOME topic word, so the local score is now a sanity
# check ("is this paper coherent with the topic at all?") rather than the
# only relevance signal. The old 0.3 threshold was reflexively rejecting
# long-topic matches that did pass arxiv's index, defeating the upgrade.
_MIN_RELEVANCE_SCORE: float = 0.15

# How far back to scan for URL dedup. Past a week, the same URL becoming
# relevant again is fair game (something new might have happened with it).
_DEDUP_LOOKBACK_DAYS: int = 7

# How many derived interest_clusters to consider as topics in addition to
# explicit pins. Sorted by recent frecency.
_DERIVED_TOPIC_LIMIT: int = 6


# Topic-shape heuristics for programmatic source routing.
_ARXIV_CATEGORY_RE = re.compile(r"^(cs|math|stat|physics|q-bio|q-fin|econ|eess)\.\w+$")
_GITHUB_URL_RE = re.compile(r"^https?://(?:www\.)?github\.com/", re.I)
# An explicit "r/<sub>" reference is an unambiguous subreddit signal — route
# it to Reddit instead of the HN default. Requires the "r/" prefix so a bare
# topic word ("localllama") is NOT mistaken for a subreddit.
_SUBREDDIT_RE = re.compile(r"^\s*/?r/([A-Za-z0-9_]{2,40})\s*$", re.IGNORECASE)


# ─── Row dataclass ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TrackedTopic:
    id: int
    user_id: str
    companion_id: str
    topic: str
    feed_url: str | None
    feed_kind: str | None
    weight: float
    last_polled_at: str | None
    error_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "feed_url": self.feed_url,
            "feed_kind": self.feed_kind,
            "weight": self.weight,
            "last_polled_at": self.last_polled_at,
            "error_count": self.error_count,
        }


# ─── Store CRUD ─────────────────────────────────────────────────────────


def _looks_like_url(s: str) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    # rsshub:// shorthands count as feeds: the per-topic fetch routes
    # through discovery.feeds.fetch_rss, which expands them against
    # the compose.rsshub overlay's base URL.
    return s.startswith(("http://", "https://", "rsshub://"))


def _detect_feed_kind(url: str) -> str:
    """Cheap routing tag based on URL shape. Best effort — curator's
    poll_for_topic uses this as a HINT, not a contract."""
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("ycombinator.com") or "news.ycombinator.com" in host:
        return "hn"
    if host.endswith("reddit.com") or host.endswith("redd.it"):
        return "reddit"
    if host.endswith("arxiv.org"):
        return "arxiv"
    return "rss"


def _detect_subreddit(topic: str) -> str:
    """Return the bare subreddit name if ``topic`` is an ``r/<sub>``
    reference (e.g. 'r/LocalLLaMA', '/r/LocalLLaMA'), else "". Lets a
    natural "what's new on r/LocalLLaMA" pin route to Reddit rather than
    falling through to the HN default."""
    m = _SUBREDDIT_RE.match(topic or "")
    return m.group(1) if m else ""


async def add_topic(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
    topic: str,
    feed_url: str | None = None,
    feed_kind: str | None = None,
) -> TrackedTopic | None:
    """Pin a topic, with optional feed_url. If the topic input looks like
    a URL and no explicit feed_url was given, treat the URL as the feed
    and derive a display topic from its host. Auto-detects feed_kind when
    feed_url is set but kind isn't.

    Returns the inserted row, or None on UNIQUE collision (caller should
    fetch existing).
    """
    if not user_id or not companion_id or not topic:
        raise ValueError("add_topic requires user_id, companion_id, topic")
    topic = topic.strip()
    if not topic:
        return None

    if feed_url is None and _looks_like_url(topic):
        feed_url = topic
        host = (urlparse(feed_url).hostname or "").removeprefix("www.")
        topic = host or topic

    if feed_url and not feed_kind:
        feed_kind = _detect_feed_kind(feed_url)

    try:
        cur = await conn.execute(
            """INSERT INTO companion_tracked_topics
               (user_id, companion_id, topic, feed_url, feed_kind, weight)
               VALUES (?, ?, ?, ?, ?, 1.0)""",
            (user_id, companion_id, topic, feed_url, feed_kind),
        )
        await conn.commit()
        row_id = cur.lastrowid
        await cur.close()
    except aiosqlite.IntegrityError:
        return None  # UNIQUE collision — caller fetches existing

    return TrackedTopic(
        id=int(row_id or 0), user_id=user_id, companion_id=companion_id,
        topic=topic, feed_url=feed_url, feed_kind=feed_kind, weight=1.0,
        last_polled_at=None, error_count=0,
    )


async def list_topics(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
) -> list[TrackedTopic]:
    if not user_id or not companion_id:
        raise ValueError("list_topics requires user_id, companion_id")
    cur = await conn.execute(
        """SELECT id, user_id, companion_id, topic, feed_url, feed_kind,
                  weight, last_polled_at, error_count
           FROM companion_tracked_topics
           WHERE user_id = ? AND companion_id = ?
           ORDER BY created_at DESC""",
        (user_id, companion_id),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        TrackedTopic(
            id=int(r[0]), user_id=r[1], companion_id=r[2], topic=r[3],
            feed_url=r[4], feed_kind=r[5], weight=float(r[6] or 1.0),
            last_polled_at=r[7], error_count=int(r[8] or 0),
        )
        for r in rows
    ]


async def remove_topic(
    conn: aiosqlite.Connection,
    *,
    topic_id: int,
    user_id: str,
    companion_id: str,
) -> bool:
    if not user_id or not companion_id:
        raise ValueError("remove_topic requires user_id, companion_id")
    cur = await conn.execute(
        """DELETE FROM companion_tracked_topics
           WHERE id = ? AND user_id = ? AND companion_id = ?""",
        (int(topic_id), user_id, companion_id),
    )
    affected = cur.rowcount or 0
    await cur.close()
    await conn.commit()
    return affected > 0


async def _mark_polled(
    conn: aiosqlite.Connection,
    *,
    topic_id: int,
    error: str | None = None,
) -> None:
    if error:
        await conn.execute(
            """UPDATE companion_tracked_topics
               SET last_polled_at = datetime('now'),
                   error_count = error_count + 1,
                   last_error = ?
               WHERE id = ?""",
            (error[:200], int(topic_id)),
        )
    else:
        await conn.execute(
            """UPDATE companion_tracked_topics
               SET last_polled_at = datetime('now'),
                   error_count = 0,
                   last_error = NULL
               WHERE id = ?""",
            (int(topic_id),),
        )
    await conn.commit()


# ─── Interest derivation ────────────────────────────────────────────────


async def _derived_topics(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    limit: int = _DERIVED_TOPIC_LIMIT,
) -> list[str]:
    """Pull recent-frecency cluster names from interest_clusters. The
    discovery engine maintains these — we just read."""
    try:
        # kind filter: consumption entities (audiobooks, comics, shows —
        # clusters minted from media_play signals) are NOT pollable
        # topics. Routing one into the HN/feed loop is how "My Quiet
        # Blacksmith Life in Another World" got matched to a systems
        # blog post on the shared token "life" (2026-06-12). Entity
        # clusters get the catalog-first recommendation ladder instead
        # (discovery/entity_recommender.py).
        cur = await conn.execute(
            """SELECT name FROM interest_clusters
               WHERE user_id = ?
                 AND COALESCE(dampened, 0) = 0
                 AND COALESCE(kind, 'topic') = 'topic'
                 AND name != ''
               ORDER BY frecency_short DESC
               LIMIT ?""",
            (user_id, int(limit)),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        # interest_clusters table missing or schema drift — degrade
        # gracefully to pinned-only.
        return []
    return [str(r[0]).strip() for r in rows if r and r[0]]


# ─── URL dedup via existing journal content_refs ────────────────────────


def _url_hash(url: str) -> str:
    return hashlib.sha256((url or "").strip().lower().encode("utf-8")).hexdigest()[:16]


async def _seen_url_recently(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
    url: str,
) -> bool:
    """Has Becca already written a journal entry referencing this URL in
    the last week? Dedup via the existing content_refs blob — no separate
    table needed."""
    if not url:
        return False
    h = _url_hash(url)
    days = _DEDUP_LOOKBACK_DAYS
    try:
        cur = await conn.execute(
            f"""SELECT 1 FROM companion_journal
                WHERE user_id = ? AND companion_id = ?
                  AND created_at > datetime('now', '-{int(days)} days')
                  AND content_refs LIKE ?
                LIMIT 1""",
            (user_id, companion_id, f'%"{h}"%'),
        )
        row = await cur.fetchone()
        await cur.close()
        return row is not None
    except Exception:
        return False


# ─── Relevance scoring ──────────────────────────────────────────────────


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "into", "your",
    "you", "are", "was", "were", "have", "has", "had", "but", "not",
    "all", "any", "can", "will", "would", "could", "should", "what",
    "which", "when", "where", "how", "why", "who", "about", "more",
    "some", "such", "than", "also",
    # High-traffic verbs whose lexical overlap produces near-zero signal
    # about topical relatedness. Their inclusion in token sets allowed
    # incoherent picks like cluster "Bitcoin Price Check" → result
    # "CHECK Definition & Meaning - Merriam-Webster" to pass the
    # coherence floor on the shared word "check".
    "check", "see", "watch", "find", "get", "open", "sign", "make",
    "use", "look", "know", "take", "come", "give", "tell", "show",
    "say", "try", "keep", "let", "set", "put", "run", "call", "move",
    "play", "read", "turn", "start", "help", "click", "login",
    "page", "site", "site's", "web", "online", "video", "videos",
    "post", "posts", "guide", "guides", "intro", "introduction",
    "definition", "meaning", "synonym", "synonyms", "transcript",
})


def _tokens(s: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]{3,}", (s or "").lower())
        if t not in _STOPWORDS
    }


def score_relevance(item: dict, topic: str) -> float:
    """Crude keyword-overlap score in [0, 1]. Item = {title, snippet, ...}.
    Returns fraction of topic tokens that appear in item title+snippet.
    Conservative bias — surface only when the connection is obvious; the
    Discovery surface handles fuzzy/semantic matches.

    Multi-token topics require ≥2 overlapping tokens. One shared word is
    never an "obvious connection": a six-token topic crossing the 0.15
    floor on a single generic token ("life", "world") is the exact
    failure that paired an audiobook with a `main()`-internals blog
    post (2026-06-12) — same class as the commemorated "Bitcoin Price
    Check" → Merriam-Webster bug. Single-token topics keep the ratio
    rule (their one token IS the topic)."""
    topic_toks = _tokens(topic)
    if not topic_toks:
        return 0.0
    item_text = f"{item.get('title', '')} {item.get('snippet', '')}"
    item_toks = _tokens(item_text)
    if not item_toks:
        return 0.0
    overlap = topic_toks & item_toks
    if len(topic_toks) >= 2 and len(overlap) < 2:
        return 0.0
    return len(overlap) / max(1, len(topic_toks))


# ─── Feed polling ───────────────────────────────────────────────────────


async def poll_for_topic(
    runtime: CompanionRuntime,
    *,
    topic: TrackedTopic | str,
    http_client: Any,
) -> list[dict]:
    """Fetch candidate items for one topic. Routing:
      - feed_url + feed_kind set: pull directly from that source (RSS / HN /
        Reddit / arxiv). Topic tokens go into arxiv as a search filter.
      - bare topic matching an arxiv category like 'cs.AI' / 'math.NT' /
        'physics.gen-ph': arxiv with that category as the filter.
      - everything else: HN top as the default broad-tech source. arxiv
        is added ONLY if the topic string mentions an arxiv category
        (e.g. "cs.CL papers on tokenization") — we don't assume every
        user wants academic ML noise mixed in.
    All arxiv calls cache results for 20min (success) / 60s (failure).
    """
    from augmentum.discovery import feeds as _feeds

    topic_str = topic if isinstance(topic, str) else topic.topic
    feed_url = None if isinstance(topic, str) else topic.feed_url
    feed_kind = None if isinstance(topic, str) else topic.feed_kind

    results: list[dict] = []

    # Tokens from the topic string are passed as `terms=` into arxiv calls
    # so the request itself is topic-targeted rather than firehose-then-
    # filter. The keyword overlap done locally (score_relevance) becomes
    # a sanity check on top of arxiv's full-text index, not the only
    # relevance signal. For bare-category topics like "cs.AI" the token
    # set is empty (no 3+ char alphanumerics survive after stopword strip)
    # so fetch_arxiv falls through to a category-only query — matching
    # historic behaviour for those.
    topic_terms = list(_tokens(topic_str))

    try:
        if feed_url:
            kind = (feed_kind or _detect_feed_kind(feed_url)).lower()
            if kind == "rss":
                results = await _feeds.fetch_rss(http_client, [feed_url], per_feed=8)
            elif kind == "hn":
                results = await _feeds.fetch_hn_top(http_client, limit=15)
            elif kind == "reddit":
                # feed_url for reddit is like https://old.reddit.com/r/X/.rss —
                # fetch as RSS rather than going through the curated subs path.
                results = await _feeds.fetch_rss(http_client, [feed_url], per_feed=8)
            elif kind == "arxiv":
                # feed_url stores the category here; fetch_arxiv takes a list.
                cat = _extract_arxiv_category(feed_url) or topic_str
                results = await _feeds.fetch_arxiv(
                    http_client, [cat], limit=50, terms=topic_terms,
                )
            else:
                results = await _feeds.fetch_rss(http_client, [feed_url], per_feed=8)
        elif (_sub := _detect_subreddit(topic_str)):
            # "r/LocalLLaMA" — pull the subreddit's hot listing directly
            # (public JSON, no auth) instead of the HN default.
            results = await _feeds.fetch_reddit(http_client, [_sub], limit=15)
        elif _ARXIV_CATEGORY_RE.match(topic_str):
            results = await _feeds.fetch_arxiv(
                http_client, [topic_str], limit=50, terms=topic_terms,
            )
        else:
            # Programmatic fan-out for topics without an explicit feed.
            # HN top is the default "any topic" source — broad enough to
            # be useful across tech / news / startup / culture-of-the-net
            # adjacent topics. arxiv is added ONLY when the topic string
            # itself signals "I want academic content" (mentions a real
            # arxiv category like cs.AI / physics.gen-ph / q-bio.PE).
            #
            # Earlier versions hardcoded `cs.AI + cs.LG` (later widened to
            # 5 ML categories) as the default arxiv fan-out — that was a
            # baked-in assumption that every user is an ML researcher.
            # For a chef tracking "sourdough hydration" or a photographer
            # tracking "off-camera flash", the ML papers were pure noise
            # that the local relevance score correctly rejected (after
            # paying the network cost). Better to not poll arxiv unless
            # the user has signaled they want it.
            #
            # The topic-tokens-as-query upgrade still applies for the
            # arxiv leg when it does fire — pinned to the category the
            # user named.
            hn = await _feeds.fetch_hn_top(http_client, limit=20)
            cat_hint = _extract_arxiv_category(topic_str)
            if cat_hint:
                arxiv = await _feeds.fetch_arxiv(
                    http_client, [cat_hint], limit=50, terms=topic_terms,
                )
                results = hn + arxiv
            else:
                results = hn
    except Exception as exc:
        log.warning(
            "curator_feed_poll_failed",
            topic=topic_str[:60], error=str(exc)[:200],
        )
        return []

    return results


def _extract_arxiv_category(text: str) -> str | None:
    """Pull a category like 'cs.AI' out of an arxiv-shaped feed url or
    arbitrary string. Falls back to None when no match."""
    m = re.search(r"\b((?:cs|math|stat|physics|q-bio|q-fin|econ|eess)\.\w+)\b", text)
    return m.group(1) if m else None


# ─── Note composition ──────────────────────────────────────────────────


def compose_note(topic: str, item: dict, *, affect_tag: str = "curious") -> tuple[str, list[dict]]:
    """Structured prose, not poetry. Two-line shape:

        {topic}
        {title} — {snippet head}

    Plus a URL ref in content_refs (and a hash ref for dedup).
    """
    title = (item.get("title") or "").strip()
    snippet = (item.get("snippet") or "").strip()
    url = (item.get("url") or "").strip()

    # Trim snippet to a tight summary — long ones read as feed-scrape spam.
    if snippet:
        snippet = snippet[:160].rstrip()
        if len(item.get("snippet", "")) > 160:
            snippet = snippet.rstrip(".,;:") + "…"

    body_parts = [topic, f"{title}"]
    if snippet and snippet != title:
        body_parts[1] += f" — {snippet}"
    content = "\n".join(body_parts).strip()

    refs: list[dict] = []
    if url:
        try:
            host = urlparse(url).hostname or ""
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        refs.append({
            "kind": "url",
            "url": url,
            "id": _url_hash(url),
            "title": title,
            "snippet": snippet,
            "domain": host,
        })

    return content, refs


def compose_entity_note(entity: dict, pick: Any) -> tuple[str, list[dict]]:
    """Note for a catalog-grounded entity pick (Gate 1 of the
    consumption-entity ladder). Same two-line shape as compose_note,
    but the second line is the pick's structured ``why`` — composed
    from catalog facts at query time, never free-associated.

    Dedup ref: a synthetic ``augm:rec:`` URL hashed like a web URL, so
    the existing content_refs LIKE dedup covers (entity, candidate,
    relation) without a new table.
    """
    head = (entity.get("series_name") or entity.get("title") or "").strip()
    title = (getattr(pick, "title", "") or "").strip()
    why = (getattr(pick, "why", "") or "").strip()
    if not head or not (title or why):
        return "", []

    line2 = title if title != head else ""
    if why:
        line2 = f"{line2} — {why}" if line2 else why
    content = f"{head}\n{line2}".strip()

    key = (
        f"augm:rec:{getattr(pick, 'relation', '')}:"
        f"{getattr(pick, 'file_id', '') or title}"
    )
    refs: list[dict] = [{
        "kind": "library_pick",
        "id": _url_hash(key),
        "url": key,
        "file_id": getattr(pick, "file_id", "") or "",
        "title": title or head,
        "relation": getattr(pick, "relation", ""),
        "gate": getattr(pick, "gate", 1),
        "snippet": why,
    }]
    return content, refs


async def _curator_entity_pick(
    runtime: CompanionRuntime,
    conn: aiosqlite.Connection,
    *,
    user_id: str,
) -> tuple[dict, Any] | None:
    """Phase 0.5 — the catalog-first ladder, before any web source.

    Returns the first (entity, pick) not yet journaled, strongest
    relation first (top_entity_picks orders continuation > new_arrival
    > same_creator > same_genre per entity). None when everything
    fresh has been surfaced already.
    """
    from augmentum.discovery.entity_recommender import top_entity_picks

    groups = await top_entity_picks(conn, user_id=user_id)
    for entity, picks in groups:
        for pick in picks:
            key = f"augm:rec:{pick.relation}:{pick.file_id or pick.title}"
            if await _seen_url_recently(
                conn, user_id=user_id,
                companion_id=runtime.companion_id, url=key,
            ):
                continue
            return entity, pick
    return None


# ─── Curator safety + quality floor ─────────────────────────────────────
#
# Browse history is private. Even if the user visits adult / sensitive
# sites, the COMPANION must not summarize that back to them as a
# deliberate "I noticed this for you" note — that's a trust violation
# and an embarrassment vector. Discovery's For-You panel is a different
# context (the user browsed there voluntarily); the curator surfaces a
# single deliberate share that needs much higher safety + quality than
# a 15-result UI panel.
#
# The three filters are layered: safety > quality > framing. A rec must
# pass all three before becoming a journal note.

# Explicit adult-content domains. Conservative list — only sites with
# unambiguously adult-only primary content. Generic social/imageboards
# (4chan, reddit, etc.) are NOT on this list; reddit subreddit subs the
# user explicitly tracks may legitimately surface adult content but the
# user opted in. This list prevents the curator from EVER linking to
# these regardless of how they reached For-You's candidate pool.
_ADULT_DOMAIN_BLOCKLIST: frozenset[str] = frozenset({
    # Porn tubes / studios
    "pornhub.com", "xvideos.com", "xnxx.com", "redtube.com",
    "youporn.com", "tube8.com", "brazzers.com", "spankbang.com",
    "xhamster.com", "youjizz.com", "porn.com", "porntrex.com",
    "eporner.com", "thumbzilla.com", "txxx.com", "hclips.com",
    "motherless.com", "gotporn.com", "pornone.com",
    # Cams / creator-monetization
    "stripchat.com", "chaturbate.com", "bongacams.com", "cam4.com",
    "myfreecams.com", "livejasmin.com", "camsoda.com",
    "onlyfans.com", "fansly.com", "manyvids.com", "iwantclips.com",
    "clips4sale.com", "fancentro.com",
    # Hentai / image-board adult
    "nhentai.net", "nhentai.com", "hanime.tv", "hentaihaven.xxx",
    "hentai2read.com", "rule34.xxx", "e621.net", "e926.net",
    "gelbooru.com", "danbooru.donmai.us", "sankakucomplex.com",
    # Adult fiction
    "literotica.com", "asstr.org",
    # Adult-toy / merchandise retailers (catches the
    # "I took his whole ..." → edenfantasys case)
    "edenfantasys.com", "adameve.com", "adamandeve.com",
    "lovehoney.com", "lovehoney.co.uk", "lovehoney.com.au",
    "lelo.com", "tantus.com", "fleshlight.com", "fleshlightgirls.com",
    "doc-johnson.com", "njoytoys.com", "extremerestraints.com",
    "babeland.com", "goodvibes.com", "mysexshop.com",
    "shevibe.com", "betterhalf.com", "spencersonline.com",
})

# Hostnames whose suffix flags an adult TLD or adult-only ecosystem.
_ADULT_SUFFIX_BLOCKLIST: tuple[str, ...] = (
    ".xxx", ".adult", ".porn", ".sex", ".sexy",
)

# URL-path fragments that indicate an adult-content page even on a
# generic domain. Catches /sex-toy/ on retail sites, /xxx/ on cdn
# subdirectories, etc. Lowercased; matched as substring of the path.
_ADULT_URL_PATH_FRAGMENTS: tuple[str, ...] = (
    "/sex-toy", "/sex-toys", "/sextoy", "/sextoys",
    "/masturbator", "/dildo", "/vibrator", "/fleshlight",
    "/porn/", "/xxx/", "/adult/", "/nsfw/",
    "/cam/", "/webcam/",
    "/escort", "/escorts/",
)

# NSFW token set lives in augmentum.discovery.safety so the recommender
# can apply the same gate to outbound search queries before they ever
# reach upstream engines. We re-export under the historical local name
# so the rest of curator.py (and its tests) keep working without
# rewiring.
from augmentum.discovery.safety import NSFW_TOKENS as _NSFW_CLUSTER_TOKENS  # noqa: E402

# Quality thresholds for the editorial pick. These are HIGHER than the
# legacy per-topic loop's `_MIN_RELEVANCE_SCORE` because that loop
# operates on items already filtered by topic terms server-side; the
# For-You pool is much broader and needs harder gates.
_CURATOR_MIN_REC_SCORE: float = 0.4     # rec must have decent quality signal
_CURATOR_MIN_COHERENCE: float = 0.15    # token overlap of cluster_name vs rec
_CURATOR_MIN_CLUSTER_SIGNALS: int = 3   # don't journal from single-signal clusters

def _curator_low_value_block(rec: dict) -> str | None:
    """Return a short reason when the rec is a low-value generic landing
    page (dictionary def / login page / Reddit 403 / etc.). Returns
    None when the rec is content-worthy. Layer 1.5 — between safety
    and quality. Reuses safety's domain normalization.

    Shares its domain + title-pattern lists with ``filter_for_llm`` via
    ``discovery.quality`` so the same SearXNG firehose results get
    rejected on both the curator pick path and the LLM web_search path.
    """
    from augmentum.discovery.quality import (
        _LOW_VALUE_LANDING_DOMAINS,
        _LOW_VALUE_LANDING_TITLE_PATTERNS,
    )
    domain = (rec.get("domain") or "").lower().lstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    if domain in _LOW_VALUE_LANDING_DOMAINS:
        return f"low_value_domain:{domain}"
    title = (rec.get("title") or "").strip()
    if not title:
        return None
    for pat in _LOW_VALUE_LANDING_TITLE_PATTERNS:
        if pat.search(title):
            return f"low_value_title:{pat.pattern[:40]}"
    return None


def _curator_safety_block(rec: dict, cluster_name: str) -> str | None:
    """Return a short reason string when this rec must NOT become a
    curator note (safety filter, layer 1). Returns None when safe.

    Trips on:
    - rec's domain in the explicit adult blocklist
    - rec's domain ending in an adult suffix (.xxx / .adult / .porn / .sex)
    - rec's URL path containing an adult-content fragment (covers cases
      like a generic retail domain hosting an /sex-toy/ category page)
    - cluster name containing an NSFW standalone token
    - rec's title containing an NSFW standalone token (catches the case
      where the cluster name is innocent but the surfaced rec's title
      is explicit, e.g. a Discovery search that bled into adult results)
    """
    url = (rec.get("url") or "").lower()
    domain = (rec.get("domain") or "").lower().lstrip(".")
    # Derive the domain from the URL when the rec didn't carry one — a
    # blocklisted host reachable only via ``url`` otherwise bypassed the
    # explicit blocklist (audit 2026-06-17). Phase 2 items + For-You recs
    # don't always populate ``domain``.
    if not domain and url:
        try:
            domain = (urlparse(url).hostname or "").lower().lstrip(".")
        except Exception:
            domain = ""
    # Strip leading www. so blocklist entries don't need both forms.
    if domain.startswith("www."):
        domain = domain[4:]
    # Suffix containment (not exact) so subdomains of a blocked host are
    # caught — m.pornhub.com / cdn.xvideos.com bypassed exact match. The
    # "." guard keeps notpornhub.com from matching pornhub.com (audit
    # 2026-06-17).
    for blocked in _ADULT_DOMAIN_BLOCKLIST:
        if domain == blocked or domain.endswith("." + blocked):
            return f"adult_domain:{blocked}"
    for suffix in _ADULT_SUFFIX_BLOCKLIST:
        if domain.endswith(suffix):
            return f"adult_suffix:{suffix}"
    if url:
        # Compare just the path+query portion to avoid false matches on
        # the domain itself.
        path_q = url.split("://", 1)[-1].split("/", 1)
        path = "/" + path_q[1] if len(path_q) > 1 else "/"
        for frag in _ADULT_URL_PATH_FRAGMENTS:
            if frag in path:
                return f"adult_url_path:{frag}"
    if _cluster_name_is_nsfw(cluster_name):
        return "nsfw_cluster_name"
    # Also scan the rec title — Discovery's SearXNG search can return
    # adult-content titles even from generic retail domains.
    if _cluster_name_is_nsfw(rec.get("title") or ""):
        return "nsfw_rec_title"
    return None


def _cluster_name_is_nsfw(name: str) -> bool:
    """Whole-word match of NSFW tokens in cluster name. Lowercased; word
    boundaries enforced so 'naked truth' / 'sex education' don't trip.
    Thin wrapper over augmentum.discovery.safety.is_nsfw_text so this
    surface, the recommender's outbound-query gate, and any future
    consumer of the same policy share one implementation."""
    from augmentum.discovery.safety import is_nsfw_text

    return is_nsfw_text(name)


def _curator_quality_pass(rec: dict, cluster_name: str) -> str | None:
    """Return None when the rec passes the quality bar; otherwise the
    reason for rejection. Layer 2 of the editorial pick filter."""
    # Score from Discovery's quality pipeline (set by _score_rec).
    score = float(rec.get("_score") or 0.0)
    if score < _CURATOR_MIN_REC_SCORE:
        return f"score_below_floor:{score:.2f}"

    # Semantic coherence — the rec should share at least one meaningful
    # token with the cluster it's supposedly "about". Without this, the
    # SearXNG firehose can surface "Future hip-hop" → "Python coinkit"
    # because both contain the word 'future'. (Fresh-zone recs from
    # external feeds skip this — they have no cluster_name by design.)
    zone = (rec.get("zone") or "").lower()
    if zone != "fresh" and cluster_name:
        coherence = score_relevance(rec, cluster_name)
        if coherence < _CURATOR_MIN_COHERENCE:
            return f"low_coherence:{coherence:.2f}"

    return None


def _sanitize_cluster_name_for_framing(name: str) -> str | None:
    """Return a cluster name suitable for the note framing ("On X"), or
    None if it's too mangled to read as a phrase. The auto-generated
    cluster name is the first 8 words of the originating signal title;
    that often truncates mid-phrase ("Future - Feds Did a Sweep (Official
    Music"), produces unbalanced parens, or ends on a dash.
    """
    if not name:
        return None
    s = name.strip()
    if len(s) < 3:
        return None
    # Drop trailing junk that signals a truncated phrase.
    s = s.rstrip("-—–:;,. ").strip()
    # Unbalanced opening brackets → truncated; not safe to frame.
    if s.count("(") != s.count(")"):
        return None
    if s.count("[") != s.count("]"):
        return None
    if s.count('"') % 2 != 0:
        return None
    # All-caps full string reads as shouty in note framing.
    if s.isupper() and len(s) > 4:
        return None
    return s if len(s) >= 3 else None


def compose_note_from_rec(rec: dict) -> tuple[str, list[dict]]:
    """Note composer for For-You picks — uses zone + cluster_name as the
    "why this matters to you" framing.

    Discovery's recommender tags each item with ``zone`` (core / adjacent
    / frontier / fresh) and ``cluster_name`` (the user's interest the
    item connects to). Both come through unchanged, so the note can say
    "you've been exploring X" instead of just dropping a title — the
    user instantly knows WHY this surfaced for them. The cluster name
    is sanitized first to avoid mangled-phrase framing like "On Future -
    Feds Did a Sweep (Official Music".
    """
    title = (rec.get("title") or "").strip()
    snippet = (rec.get("snippet") or "").strip()
    url = (rec.get("url") or "").strip()
    raw_cluster_name = (rec.get("cluster_name") or "").strip()
    cluster_name = _sanitize_cluster_name_for_framing(raw_cluster_name) or ""
    zone = (rec.get("zone") or "").strip().lower()

    if zone == "core" and cluster_name:
        topic_line = f"On {cluster_name}"
    elif zone == "adjacent" and cluster_name:
        topic_line = f"Adjacent to {cluster_name}"
    elif zone == "frontier" and cluster_name:
        topic_line = f"Gap in {cluster_name}"
    elif zone == "fresh":
        topic_line = "Fresh in your feeds"
    else:
        # Cluster name was unusable (truncated, mangled, missing). Fall
        # back to a generic-but-still-coherent framing rather than
        # printing the mangled name verbatim.
        topic_line = "Caught my eye"

    if snippet:
        snippet = snippet[:160].rstrip()
        if len(rec.get("snippet", "")) > 160:
            snippet = snippet.rstrip(".,;:") + "…"

    body_parts = [topic_line, title]
    if snippet and snippet != title:
        body_parts[1] += f" — {snippet}"
    content = "\n".join(body_parts).strip()

    refs: list[dict] = []
    if url:
        # Rich ref payload — the UI can render a real article preview
        # (domain badge + clickable title + snippet) instead of just a
        # `url:<hash>` chip. Older notes without these fields still
        # work via the frontend's fall-back rendering.
        try:
            host = urlparse(url).hostname or ""
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        refs.append({
            "kind": "url",
            "url": url,
            "id": _url_hash(url),
            "title": title,
            "snippet": snippet,
            "domain": host,
            "zone": zone,
            "cluster_name": cluster_name,
        })

    return content, refs


async def _curator_for_you_pick(
    runtime: CompanionRuntime,
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
) -> dict | None:
    """Try Discovery's For-You generator as the curator's primary candidate
    source. Returns the chosen rec (with zone + cluster_name fields) or
    None if nothing eligible.

    Editorial pick: walks the recs in zone-priority order (core → adjacent
    → frontier → fresh), skipping URLs the curator has journaled in its
    dedup window. The zone priority encodes "what's most likely to feel
    like a deliberate share" — core (about user's known interests) over
    fresh (about external feeds the user subscribed to).
    """
    from augmentum.config import settings

    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return None
    discovery_store = getattr(app_state, "discovery_store", None)
    http_client = getattr(app_state, "http_client", None)
    if discovery_store is None or http_client is None:
        return None
    settings_store = getattr(app_state, "settings_store", None)

    # Mirror /api/discovery/for-you's feed_config resolution so the Fresh
    # zone reads from the same per-user settings the UI exposes. Failures
    # degrade to "HN only" defaults rather than collapsing the whole call.
    feed_config: dict = {"hn": True, "reddit_subs": [], "arxiv_cats": [], "rss_urls": []}
    if settings_store is not None:
        try:
            async def _get(k: str, d: str) -> str:
                v = await settings_store.get(k)
                return v if v is not None else d
            hn = await _get("discovery_feeds_hn", "1")
            reddit = await _get("discovery_feeds_reddit", "")
            arxiv = await _get("discovery_feeds_arxiv", "")
            rss = await _get("discovery_feeds_rss", "")
            feed_config = {
                "hn": str(hn).strip() in ("1", "true", "True", "yes"),
                "reddit_subs": [s.strip() for s in (reddit or "").split(",") if s.strip()],
                "arxiv_cats": [s.strip() for s in (arxiv or "").split(",") if s.strip()],
                "rss_urls": [s.strip() for s in (rss or "").split(",") if s.strip()],
            }
        except Exception:
            log.warning("curator_feed_config_load_failed", exc_info=True)

    hidden_urls: set[str] = set()
    try:
        hidden_urls = set(await discovery_store.list_hidden_urls(user_id=user_id))
    except Exception:
        log.warning("curator_hidden_urls_load_failed", exc_info=True)

    try:
        from augmentum.discovery.recommender import generate_recommendations
        recs = await generate_recommendations(
            discovery_store,
            searxng_base=settings.searxng_base_url,
            total=15,
            http_client=http_client,
            domain_scores=None,
            hidden_urls=hidden_urls,
            feed_config=feed_config,
            user_id=user_id,
            allow_non_latin=bool(getattr(settings, "discovery_allow_non_latin", False)),
            # Autonomous path — subject to companion_autonomous_web_search_enabled.
            autonomous=True,
        )
    except Exception:
        log.warning("curator_foryou_recs_failed", exc_info=True)
        return None

    if not recs:
        return None

    zone_order = ("core", "adjacent", "frontier", "fresh")
    by_zone: dict[str, list[dict]] = {z: [] for z in zone_order}
    for r in recs:
        z = (r.get("zone") or "").lower()
        if z in by_zone:
            by_zone[z].append(r)

    for zone in zone_order:
        for r in by_zone[zone]:
            url = (r.get("url") or "").strip()
            if not url:
                continue

            cluster_name = (r.get("cluster_name") or "").strip()

            # Layer 1 — safety. Adult/sensitive domains or NSFW cluster
            # names never become curator notes regardless of how they
            # reached For-You's candidate pool. The browse history that
            # produced the cluster is private; surfacing it back as a
            # deliberate share is a trust violation.
            block = _curator_safety_block(r, cluster_name)
            if block is not None:
                log.info(
                    "curator_foryou_safety_block",
                    reason=block,
                    domain=(r.get("domain") or "")[:80],
                    cluster=cluster_name[:60],
                )
                continue

            # Layer 1.5 — low-value landing pages. Dictionary defs,
            # login pages, Reddit 403s, "What is X?" definitional
            # landings. The SearXNG firehose loves these because they
            # rank high on shared single-word queries; the curator
            # should never deliberately surface one as a finding.
            low_value = _curator_low_value_block(r)
            if low_value is not None:
                log.info(
                    "curator_foryou_low_value_skip",
                    reason=low_value,
                    domain=(r.get("domain") or "")[:80],
                    title=(r.get("title") or "")[:80],
                )
                continue

            # Layer 2 — quality. Reject low-score recs and recs that
            # don't actually relate to their cluster (the "music video
            # cluster → unrelated GitHub wordlist" failure mode).
            quality = _curator_quality_pass(r, cluster_name)
            if quality is not None:
                log.debug(
                    "curator_foryou_quality_skip",
                    reason=quality,
                    url=url[:120],
                    cluster=cluster_name[:60],
                )
                continue

            try:
                if await _seen_url_recently(
                    conn, user_id=user_id, companion_id=companion_id, url=url,
                ):
                    continue
            except Exception as exc:
                # Don't fail the whole pick because one url's dedup check
                # broke — but log so a real regression in _seen_url_recently
                # (schema drift, etc.) doesn't stay invisible. Debug, not
                # warning, because intermittent transient sqlite contention
                # would otherwise spam the log.
                log.debug(
                    "curator_seen_url_check_failed",
                    url=url[:120], error=str(exc)[:160],
                )
                continue
            return r

    return None


# ─── The step: one orchestrated iteration ──────────────────────────────


async def step(runtime: CompanionRuntime) -> int | None:
    """One curator iteration. Called from the companion tick.

    Returns the journal_id of a written note, or None if no note was
    written (debounced, no eligible topic, no relevant item, dedup hit,
    etc — all are normal outcomes).
    """
    from augmentum.config import settings

    if not getattr(settings, "companion_curator_enabled", True):
        return None

    user_id = getattr(runtime, "owner_user_id", "") or ""
    if not user_id:
        return None

    from augmentum.companion_runtime import presence_mode as _pm
    if not _pm.autonomy_allowed():
        return None

    # Per-runtime debounce. Two layers:
    #
    # 1. Write-debounce: minimum interval between successful note writes.
    #    Volume kills meaning — even useful notes feel meh every minute.
    #    Set on each successful note.
    #
    # 2. Attempt-debounce: minimum interval between expensive attempts,
    #    regardless of whether they produced a note. Closes the bug where
    #    step() re-ran the gather_feeds + SearXNG recommender pipeline
    #    every 5s in 'present' state because nothing qualified for writing
    #    (steady-state: most candidates already journaled). That cascade
    #    burned through engine rate-limit budget in minutes.
    write_interval = float(
        getattr(settings, "companion_curator_interval_s", _STEP_DEBOUNCE_SECONDS),
    )
    attempt_interval = float(
        getattr(
            settings, "companion_curator_attempt_interval_s",
            _STEP_ATTEMPT_DEBOUNCE_SECONDS,
        ),
    )
    now = time.time()
    last_write = float(getattr(runtime, "_last_curator_at", 0.0) or 0.0)
    last_attempt = float(getattr(runtime, "_last_curator_attempt_at", 0.0) or 0.0)
    if now - last_write < write_interval:
        return None
    if now - last_attempt < attempt_interval:
        return None

    # Mark the attempt NOW (before any expensive work) so a slow
    # gather_feeds / SearXNG call doesn't leave the next tick free to
    # re-enter while this one's still in flight. Set first, work second.
    runtime._last_curator_attempt_at = now

    backend = runtime.backend
    conn = backend.conn

    # ── Pick eligible topics: explicit pins + top derived clusters ──
    try:
        topics = await list_topics(
            conn, user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception:
        log.warning("curator_topic_list_failed", exc_info=True)
        return None

    derived_names = await _derived_topics(conn, user_id=user_id)

    # Build the candidate list. Pinned topics carry their feed_url; derived
    # ones are bare names that will route programmatically.
    candidates: list[TrackedTopic | str] = []
    poll_cutoff_ago = _TOPIC_POLL_COOLDOWN_SECONDS
    for t in topics:
        if t.last_polled_at:
            # Only re-poll if cooldown expired.
            try:
                # SQLite datetime() string parse — naive but adequate.
                _t = _parse_sqlite_dt(t.last_polled_at)
                if _t and (now - _t) < poll_cutoff_ago:
                    continue
            except (ValueError, TypeError) as exc:
                # Malformed last_polled_at — treat as "never polled"
                # rather than silently skipping the topic.
                log.debug(
                    "curator_cooldown_parse_failed",
                    topic_id=getattr(t, "id", None),
                    raw=str(t.last_polled_at)[:40],
                    error=str(exc)[:120],
                )
        candidates.append(t)

    for name in derived_names:
        # Skip derived names that duplicate a pinned topic.
        if any(t.topic.lower() == name.lower() for t in topics):
            continue
        candidates.append(name)

    if not candidates:
        log.debug("curator_no_candidates", user_id=user_id)
        return None

    # ── http_client from app.state (already lives there for discovery) ──
    http_client = _resolve_http_client(runtime)
    if http_client is None:
        log.debug("curator_no_http_client")
        return None

    written_id: int | None = None

    # ── Phase 0.5: consumption-entity picks (catalog-first ladder) ──
    # The most groundable note she can write costs zero network: the
    # next volume sitting in the library, new chapters a sync landed,
    # the same author's unread book. Gate 1 of the entity ladder
    # (spec 2026-06-12); runs before any web-derived source. The
    # setting turns off the OFFERING side only — the companion's
    # media_recommendations tool answers on request regardless.
    if bool(getattr(settings, "companion_entity_recs_enabled", True)):
        try:
            found = await _curator_entity_pick(runtime, conn, user_id=user_id)
        except Exception:
            log.warning("curator_entity_pick_failed", exc_info=True)
            found = None
        if found is not None:
            entity, pick = found
            content, refs = compose_entity_note(entity, pick)
            # Safety filter — the entity/catalog path bypassed it entirely
            # before (only For-You was gated), so a catalog title/why with
            # adult content reached the drawer (audit 2026-06-17). Scan the
            # composed text; fall through to Phase 1 when blocked.
            entity_block = _curator_safety_block(
                {"title": content,
                 "url": (refs[0].get("url") if refs else "")},
                str(entity.get("series_name") or entity.get("title") or ""),
            ) if content else None
            if entity_block:
                log.info(
                    "curator_entity_safety_blocked",
                    reason=entity_block, user_id=user_id,
                )
            if content and not entity_block:
                try:
                    written_id = await runtime.memory.safe_journal(
                        content,
                        source="curator",
                        user_id=user_id,
                        entry_type="curator_note",
                        affect_tag="curious",
                        content_refs=refs,
                        confidence_numeric=0.9,   # catalog facts, not guesses
                        origin={
                            "source": "curator",
                            "detail": (
                                f"library {pick.relation}: "
                                f"{(pick.title or entity.get('title') or '')[:80]}"
                            ),
                        },
                    )
                except Exception:
                    log.warning("curator_entity_write_failed", exc_info=True)
                    written_id = None
                if written_id:
                    log.info(
                        "curator_note_written",
                        source="entity",
                        user_id=user_id, journal_id=written_id,
                        relation=pick.relation,
                        title=(pick.title or "")[:60],
                    )
                    runtime._last_curator_at = now
                    return written_id

    # ── Phase 1: try For-You as the primary candidate source ──
    # Discovery's recommender already does the heavy lift (cluster-driven
    # queries, quality pipeline, dedup-vs-history, Core/Adjacent/Frontier
    # zone segmentation). The curator's job is editorial — pick ONE rec
    # not yet journaled, frame the note around its zone, write. The legacy
    # per-topic loop below stays as fallback for users without enough
    # clusters for For-You to fire (and to honor any explicit feed_urls
    # tracked in companion_tracked_topics that aren't user-set RSS subs).
    try:
        rec = await _curator_for_you_pick(
            runtime, conn,
            user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception:
        log.warning("curator_foryou_pick_failed", exc_info=True)
        rec = None

    if rec is not None:
        content, refs = compose_note_from_rec(rec)
        if content:
            try:
                written_id = await runtime.memory.safe_journal(
                    content,
                    source="curator",
                    user_id=user_id,
                    entry_type="curator_note",
                    affect_tag="curious",
                    content_refs=refs,
                    confidence_numeric=0.6,
                    origin={
                        "source": "curator",
                        "detail": (
                            f"for-you {rec.get('zone') or 'pick'}: "
                            f"{(rec.get('cluster_name') or rec.get('url') or '')[:80]}"
                        ),
                    },
                )
            except Exception:
                log.warning("curator_foryou_write_failed", exc_info=True)
                written_id = None

            if written_id:
                log.info(
                    "curator_note_written",
                    source="foryou",
                    user_id=user_id, journal_id=written_id,
                    zone=rec.get("zone", ""),
                    cluster=(rec.get("cluster_name") or "")[:60],
                    url=(rec.get("url") or "")[:100],
                )
                runtime._last_curator_at = now
                return written_id

    # ── Phase 2: legacy per-topic polling ──
    for candidate in candidates:
        topic_str = candidate if isinstance(candidate, str) else candidate.topic
        topic_id = None if isinstance(candidate, str) else candidate.id

        try:
            items = await poll_for_topic(
                runtime, topic=candidate, http_client=http_client,
            )
        except Exception as exc:
            log.warning(
                "curator_poll_exception",
                topic=topic_str[:60], error=str(exc)[:200],
            )
            if topic_id:
                try:
                    await _mark_polled(conn, topic_id=topic_id, error=str(exc)[:200])
                except Exception as mark_exc:
                    log.warning(
                        "curator_mark_polled_failed",
                        phase="error",
                        topic_id=topic_id,
                        error=str(mark_exc)[:200],
                    )
            continue

        if topic_id:
            try:
                await _mark_polled(conn, topic_id=topic_id)
            except Exception as mark_exc:
                log.warning(
                    "curator_mark_polled_failed",
                    phase="success",
                    topic_id=topic_id,
                    error=str(mark_exc)[:200],
                )

        if not items:
            continue

        # Rank items by relevance to the topic — only consider qualifying ones.
        scored = sorted(
            ((score_relevance(it, topic_str), it) for it in items),
            key=lambda p: p[0], reverse=True,
        )

        chosen = None
        for score, it in scored:
            if score < _MIN_RELEVANCE_SCORE:
                break
            url = (it.get("url") or "").strip()
            if not url:
                continue
            if await _seen_url_recently(
                conn, user_id=user_id, companion_id=runtime.companion_id, url=url,
            ):
                continue
            chosen = it
            break

        if chosen is None:
            continue

        # ── Compose + write ────────────────────────────────────────
        content, refs = compose_note(topic_str, chosen, affect_tag="curious")
        if not content:
            continue

        # Safety filter — the legacy per-topic path bypassed it entirely
        # before (only For-You was gated), so a tracked feed item with
        # adult content went straight to the drawer (audit 2026-06-17).
        topic_block = _curator_safety_block(chosen, topic_str)
        if topic_block:
            log.info(
                "curator_topic_safety_blocked",
                reason=topic_block, user_id=user_id, topic=topic_str[:60],
            )
            continue

        try:
            written_id = await runtime.memory.safe_journal(
                content,
                source="curator",
                user_id=user_id,
                entry_type="curator_note",
                affect_tag="curious",
                content_refs=refs,
                confidence_numeric=0.6,
                origin={
                    "source": "curator",
                    "detail": f"feed: {topic_str[:80]}",
                },
            )
        except Exception:
            log.warning("curator_write_failed", exc_info=True)
            continue

        if written_id:
            log.info(
                "curator_note_written",
                user_id=user_id, journal_id=written_id,
                topic=topic_str[:60], url=(chosen.get("url") or "")[:100],
            )
            runtime._last_curator_at = now
            return written_id

    return None


# ─── Helpers ────────────────────────────────────────────────────────────


def _parse_sqlite_dt(s: str) -> float | None:
    """Parse a SQLite datetime() string ('YYYY-MM-DD HH:MM:SS' UTC) to
    a Unix timestamp. Returns None on parse failure."""
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        s = s.strip()
        if "T" not in s:
            s = s.replace(" ", "T")
        if not s.endswith("Z"):
            s = s + "Z"
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _resolve_http_client(runtime: CompanionRuntime) -> Any | None:
    """Find the shared httpx client. The FastAPI lifespan stashes it on
    app.state, and the companion runtime has a back-reference."""
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return None
    client = getattr(app_state, "http_client", None)
    return client


__all__ = [
    "TrackedTopic",
    "add_topic",
    "list_topics",
    "remove_topic",
    "score_relevance",
    "compose_note",
    "poll_for_topic",
    "step",
]
