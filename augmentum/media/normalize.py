"""Name normalisation for author / narrator / series matching.

Real media libraries are messy: the same person shows up as "Brandon
Sanderson" / "Sanderson, Brandon" / "B. Sanderson" / "brandon sanderson"
across different books (sometimes even in the same library) because
audiobook tagging is a manual human process. Naive string equality
fails for all of these.

``normalize_name`` collapses these variants to a canonical form so that
equality on the normalised string answers "are these the same person?"
for the >90% case. ``fuzzy_match_score`` handles the long tail (typos,
missing spaces like "JFBrink") with token intersection.

Design:

1. Lowercase the string.
2. Strip apostrophes (ASCII + curly), periods, commas.
3. Tokenize on whitespace / remaining punctuation.
4. Drop single-letter tokens ("B.", "J.F." → just the surnames).
5. Drop common stopwords ("the", "a", "an").
6. Sort the remaining tokens alphabetically + rejoin with single spaces.

So "Brandon Sanderson" and "Sanderson, Brandon" both normalise to
``"brandon sanderson"``. "J.F. Brink" normalises to ``"brink"`` (single-
letter J and F drop). That's actually *too* lossy for "J.F. Brink" vs
"Jane Frida Brink", but since the user is clicking an explicit author
tag on a known book, they're asking "other books tagged identically or
near-identically" — which is the behaviour we want.
"""

from __future__ import annotations

import re

# Apostrophes + curly quotes + backtick are *removed* (empty replacement)
# so "O'Brien" → "obrien" matches "OBrien". Treating them as separators
# would split "O'Brien" into ["o", "brien"] and (combined with the single-
# letter drop below) lose the "o" entirely, producing "brien" — which
# fails to match "OBrien" = "obrien".
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u201C\u201D'`]")

# Everything else that's a natural word break — periods in titles
# ("Mr." / "J.F."), commas in reverse-name conventions ("Sanderson,
# Brandon"), hyphens in compound names — is *replaced with space* so the
# tokenizer sees distinct words. Single letters that pop out of this
# (the J and F from "J.F.") get dropped later.
_PUNCTUATION_RE = re.compile(r'[".,;:!?\-_/\\()\[\]{}]')

_WHITESPACE_RE = re.compile(r"\s+")

# Very small stopword list — name-specific, not general-English. Adding
# "the" here risks collapsing "The The" (a real band) but avoids "The
# Rock Says" vs "Rock Says" mismatching. Keep deliberately tiny.
_STOPWORDS = frozenset({"the", "a", "an"})


def normalize_name(raw: str) -> str:
    """Return a canonical form suitable for equality matching."""
    if not raw:
        return ""
    lowered = raw.lower()
    # Remove apostrophes silently (no separator) — keeps "O'Brien" as a
    # single "obrien" token.
    no_apos = _APOSTROPHE_RE.sub("", lowered)
    # Replace separator punctuation with space so multi-word names
    # tokenize into their constituent words.
    stripped = _PUNCTUATION_RE.sub(" ", no_apos)
    tokens = [
        t for t in _WHITESPACE_RE.split(stripped)
        if t and len(t) > 1 and t not in _STOPWORDS
    ]
    tokens.sort()
    return " ".join(tokens)


def normalize_list(names: list[str] | None) -> list[str]:
    """Normalise a list of names (authors/narrators arrays from ABS)."""
    if not names:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        norm = normalize_name(n)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# Non-author credits that publishers/ABS sometimes bake into the author
# field as a " - <role>" suffix on a contributor's name (e.g. ABS stores
# an author entry literally named "Jenny McKeon - translator"). These are
# NOT authors — keeping them pollutes the author credit line AND breaks
# related-book matching across a series whose volumes credit *different*
# translators/illustrators (each gets unique junk tokens, so the shared
# real author no longer token-subset-matches). We drop them for matching.
_NON_AUTHOR_ROLES = frozenset({
    "translator", "translators", "illustrator", "illustrators", "narrator",
    "narrators", "editor", "editors", "foreword", "afterword", "introduction",
    "contributor", "contributors", "reader", "cover", "adaptation",
})

# A trailing " - <role>" annotation. Spaces around the dash are REQUIRED so
# compound names ("Jean-Paul Sartre", no surrounding spaces) are never
# mistaken for a role suffix.
_ROLE_SUFFIX_RE = re.compile(r"\s+[-–—]\s+([A-Za-z][A-Za-z ]{1,30})$")


def _is_non_author_credit(segment: str) -> bool:
    """True when a single author-list segment is a non-author role credit
    like ``"Jenny McKeon - translator"``."""
    m = _ROLE_SUFFIX_RE.search((segment or "").strip())
    return bool(m) and m.group(1).strip().lower() in _NON_AUTHOR_ROLES


def author_for_match(raw_author: str) -> str:
    """Drop non-author role credits from a joined author string.

    ``"Okina Baba, Jenny McKeon - translator"`` -> ``"Okina Baba"`` so a
    series whose volumes credit different translators/illustrators still
    matches on the shared real author. Never empties the field: if every
    segment looks like a non-author credit (degenerate data), the original
    is returned unchanged rather than producing an authorless book.
    """
    if not raw_author:
        return ""
    parts = [p.strip() for p in raw_author.split(",")]
    kept = [p for p in parts if p and not _is_non_author_credit(p)]
    return ", ".join(kept) if kept else raw_author


def tokens_match_as_related(seed_normalized: str, other_normalized: str) -> bool:
    """True when two already-normalised names should match as "same person"
    for related-item queries (``Also by X`` / ``Also narrated by X``).

    Equality on the canonical string is too brittle for real libraries:
    a single uploader concatenating a book title into the author field
    (``"JF Brink TheFirstDefier"``) produces different tokens from the
    same author's other books (``"JF Brink"``), and a co-author join
    (``"Jane Austen, Charles Dickens"``) never matches either author's
    solo titles.

    Treat one side's tokens being a subset of the other's as "same
    person". Same tokens are a subset both ways, so this covers exact
    match too. Rejects ``"Jane Smith"`` ↔ ``"John Smith"`` (share only
    ``smith`` — neither is a subset) which is the precision we want.
    """
    if not seed_normalized or not other_normalized:
        return False
    a = set(seed_normalized.split())
    b = set(other_normalized.split())
    if not a or not b:
        return False
    return a <= b or b <= a


def fuzzy_match_score(a: str, b: str) -> float:
    """Token-intersection score between two already-normalised names.

    Returns a 0..1 ratio = (shared tokens) / (union tokens). 1.0 means
    identical, 0.0 means nothing in common. Used as a fallback when the
    exact-normalised equality lookup returns nothing and we want to
    offer "probably the same person" suggestions.
    """
    if a == b and a:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    union = sa | sb
    return len(inter) / len(union)
