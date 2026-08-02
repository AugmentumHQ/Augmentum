"""Shared content-safety primitives for the Discovery + Curator surfaces.

Two surfaces independently need the same "is this text NSFW?" check:

  * **Recommender** (``recommender.py::build_search_query``) — must drop a
    cluster name before it becomes an outbound SearXNG query. Without
    this gate the curator's safety filter only runs on already-fetched
    results, which means the unsafe *query string* has already been
    transmitted to upstream search engines. That's a privacy leak (the
    search engine sees what we searched for, even if we then discard
    the response) and a reputation leak (the augmentum-deployment IP
    gets flagged as an adult-search source).

  * **Curator** (``companion_runtime/curator.py``) — second line of
    defense on incoming results, also blocks by cluster name and
    result title.

Keeping the token set in one place + giving it a public name + adding
tests at this layer means a future surface (voice search, agentic
web_search, browse autocomplete) can opt in by importing one helper
rather than re-deriving the list.

The token set is conservative on false positives: whole-word match
only, so "sex education", "naked truth", "hot take" don't trigger.
``naked``, ``sex``, ``hot`` are intentionally NOT on the list — they
appear in too many neutral contexts. The list focuses on industry
terms, kink-vocabulary, and explicit-act tokens that have essentially
no neutral usage.
"""

from __future__ import annotations

import re

# Industry / community / explicit-act tokens that, as standalone words,
# essentially never appear in neutral content. Each entry is a literal
# whole-word match (``\bX\b``) so a token like ``anal`` doesn't catch
# ``anally``-as-substring inside ``analysis``.
#
# When adding entries: only add tokens that are unambiguous as
# standalone words. If a token has a neutral meaning (e.g. ``naked``,
# ``adult``, ``mature``) it belongs in the more nuanced URL-path /
# domain checks in curator.py, not here.
NSFW_TOKENS: frozenset[str] = frozenset({
    # Industry terms (unambiguous)
    "porn", "xxx", "nsfw", "hentai", "doujin", "doujinshi", "ahegao",
    "ecchi", "rule34", "r34", "futa", "futanari", "lewd",
    # Kink / fetish (largely unambiguous in this context)
    "bdsm", "fetish", "kink", "kinky", "milf", "bbw",
    "camgirl", "camgirls", "camboy", "stripper", "stripping",
    # Slang for masturbation / industry roles
    "fap", "fapping", "wank", "wanking",
    "pornstar", "pornstars", "camwhore", "camwhores",
    # Explicit sexual acts (mostly unambiguous as single tokens)
    "blowjob", "blowjobs", "handjob", "handjobs",
    "rimjob", "rimming", "footjob", "tittyfuck",
    "creampie", "bukkake", "facials", "deepthroat",
    "anal", "anally", "cumshot", "cumshots", "cuckold",
    # Adult-toy product nouns (unambiguous in cluster names)
    "fleshlight", "dildo", "vibrator", "buttplug", "buttplugs",
    "masturbator", "masturbators", "onahole", "onaholes",
    "strapon", "strap-on",
    # Strong sexual slang as standalone tokens
    "cocksucking", "cocksucker", "pussylicking",
    "gangbang", "gangbangs", "threesome", "foursome", "orgy",
    # Sex-work language
    "escort", "escorts", "callgirl",
    # Adult-tube / cam-site domain stems. Cluster names get scraped
    # from browse-history page titles which often carry the source
    # site name verbatim ("X — Pornhub", "Free videos | XVideos.com").
    # These are tokenized as single words (e.g. "pornhub", "xvideos")
    # so the bare "porn"/"xxx" entries above don't match — they need
    # explicit entries. All listed sites have no neutral usage as
    # standalone words.
    "pornhub", "xvideos", "xnxx", "xhamster", "youporn", "redtube",
    "spankbang", "motherless", "tnaflix", "porntrex", "porntube",
    "tubegalore", "porn7", "thumbzilla", "iceporn", "porndoe",
    "porndig", "pornone", "drtuber", "yespornplease", "eporner",
    "fapality", "fapper", "fapdu", "faphouse", "fapcat",
    # Cam sites (live-stream adult)
    "chaturbate", "stripchat", "bongacams", "myfreecams", "camsoda",
    "livejasmin", "flirt4free", "imlive", "streamate",
    # Hentai / animation-specific
    "nhentai", "hanime", "fakku", "hentaihaven", "pururin",
})

# Korean explicit-content substrings. CJK has no reliable word
# boundaries (no spaces between morphemes, particles fuse onto nouns:
# 성인 → 성인물 / 성인용), so the whole-word token-split semantics used
# for English don't work here — these are matched as *substrings*.
#
# Listing a stem (e.g. ``성인``) therefore also catches its derivatives
# (``성인물``, ``성인용``). This is deliberately tuned for the family /
# SFW gate where over-blocking a borderline card is the safe error
# direction — unlike :data:`NSFW_TOKENS`, which stays conservative to
# avoid dropping neutral discovery content. Used by character-import
# search (chub.ai / RisuRealm) where the upstream SFW flag is the only
# other line of defense and Korean cards are common.
KOREAN_NSFW_SUBSTRINGS: frozenset[str] = frozenset({
    "성인",      # "adult" — dominant usage on card sites is adult-content tagging
    "19금",      # "19+" age-restricted marker
    "야한",      # "lewd / dirty"
    "야설",      # "erotic fiction"
    "야동",      # "porn video"
    "에로",      # "ero(tic)"
    "섹스",      # "sex"
    "자위",      # "masturbation"
    "페티시",    # "fetish"
    "페티쉬",    # "fetish" (alt spelling)
    "변태",      # "pervert / deviant"
    "거유",      # slang: "large breasts"
    "보지",      # explicit genital slang
    "자지",      # explicit genital slang
    "강간",      # "rape" (NSFL)
    "근친",      # "incest"
    "로리",      # "loli" — adult-content marker in this context
    "헨타이",    # "hentai" (Korean transliteration)
})

# Multi-word adult phrases that whole-word token matching can't catch
# because each token is innocuous alone ("barely"/"legal"). Matched as a
# substring against separator-normalized lowercased text, so "barely-legal"
# and "barely_legal" also trip. Keep this set TINY and unambiguous — these
# phrases have essentially no neutral usage (audit 2026-06-17).
NSFW_PHRASES: frozenset[str] = frozenset({
    "barely legal",
    "barely 18",
    "jail bait",
})

# Splits on any non-word character (spaces, punctuation, parens,
# dashes, hyphens). Lowercased input so the token-set match is
# case-insensitive without sprinkling .lower() everywhere.
_TOKEN_SPLIT_RE = re.compile(r"\W+")

# Collapses runs of whitespace / hyphen / underscore to a single space so
# phrase matching is separator-insensitive.
_PHRASE_NORM_RE = re.compile(r"[\s_\-]+")


def is_korean_nsfw_text(text: str | None) -> bool:
    """Return True when *text* contains a Korean explicit-content substring.

    Substring match (not whole-word) because CJK lacks reliable word
    boundaries. Tuned to over-block for the family / SFW gate. Returns
    ``False`` for empty input.
    """
    if not text:
        return False
    return any(sub in text for sub in KOREAN_NSFW_SUBSTRINGS)


def is_unsafe_card_text(text: str | None) -> bool:
    """Return True when *text* trips the English OR Korean explicit check.

    Backstop for character-import search results when SFW is enforced:
    the upstream ``nsfw=false`` flag is trusted but not verified, so a
    mistagged-SFW card (or one whose name/description/tags carry explicit
    English or Korean terms) would otherwise pass straight through.

    English uses whole-word semantics (:func:`is_nsfw_text`); Korean uses
    substring semantics (:func:`is_korean_nsfw_text`).
    """
    return is_nsfw_text(text) or is_korean_nsfw_text(text)


def is_nsfw_text(text: str | None) -> bool:
    """Return True when *text* contains an NSFW token as a whole word.

    Conservative on false positives: only fires when a tokenized,
    lowercased word matches an entry in :data:`NSFW_TOKENS`. Substring
    matches (e.g. ``anal`` inside ``analyze``) do NOT trigger because
    the split-and-set-intersect semantics enforce word boundaries.

    Used at two layers:

      * before the recommender sends an outbound SearXNG query
        (``build_search_query``) — drops the query entirely so the
        unsafe string never reaches upstream search engines;
      * inside the curator's editorial-pick filter — second line of
        defense against unsafe text in result titles / cluster names.

    Returns ``False`` for empty input rather than raising so callers
    can pass user-supplied / DB-derived strings without nil-checks.
    """
    if not text:
        return False
    low = text.lower()
    tokens = {t for t in _TOKEN_SPLIT_RE.split(low) if t}
    if tokens & NSFW_TOKENS:
        return True
    # Phrase pass — separator-normalized substring match for multi-word
    # adult phrases the token pass can't catch.
    norm = _PHRASE_NORM_RE.sub(" ", low).strip()
    return any(p in norm for p in NSFW_PHRASES)
