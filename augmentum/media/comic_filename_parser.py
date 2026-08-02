"""Comic archive filename parser.

~80% of real-world comic archives have no ComicInfo.xml. Their only metadata
lives in the filename — usually following one of a handful of scanlation/
digital-release conventions. This module extracts what it can and produces a
confidence score so downstream callers (the scan worker, the series-identity
store, the library review queue) know how much to trust each field.

Design:
  - Pure function. No I/O, no DB, no network. Deterministic.
  - Iterative bracket/paren extraction, not a recursive grammar — real
    filenames use flat token lists, not nested grammar. Nested parens in
    scan-group names ("(danke-Empire)") work because we only recurse when
    classification fails.
  - Confidence is a weighted sum of recognized fields, capped at 1.0.
  - Preserves Unicode cleanly (Japanese titles, full-width volume markers).

Output feeds:
  - ``ComicSeriesStore.create_or_resolve_series(name=parsed.series, ...)``
  - ``file_index.mtime / metadata_confidence`` columns via the scan worker
  - Low-confidence results surface in the Library Health review queue
    (Phase A.8, Phase F).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# --- Recognized patterns -----------------------------------------------------

# Year: 1900-2099, only considered a year when it appears inside brackets/parens.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# Volume markers. Ordered longest-first so ``volume`` beats ``vol`` beats ``v``.
_VOLUME_PATTERNS = [
    # Japanese: 第01巻 or 第1巻
    re.compile(r"第(?P<n>\d{1,4})巻"),
    # Omnibus range: v01-03, Vol. 1-3, Volume 1-3
    re.compile(r"(?i)\b(?:volume|vol\.?|v)\s*(?P<a>\d{1,4})\s*[-–]\s*(?P<b>\d{1,4})\b"),
    # Single volume: Volume 1 / Vol. 1 / Vol 1 / v01
    re.compile(r"(?i)\bvolume\s*(?P<n>\d{1,4})\b"),
    re.compile(r"(?i)\bvol\.?\s*(?P<n>\d{1,4})\b"),
    re.compile(r"(?i)\bv(?P<n>\d{1,4})\b"),
]

# Chapter markers. Decimal chapters (12.5) matter for webtoons / bonus chapters.
_CHAPTER_PATTERNS = [
    # Japanese: 第01話
    re.compile(r"第(?P<n>\d{1,5}(?:\.\d{1,2})?)話"),
    # Chapter 1 / Ch. 1 / Ch 1
    re.compile(r"(?i)\bchapter\s*(?P<n>\d{1,5}(?:\.\d{1,2})?)\b"),
    re.compile(r"(?i)\bch\.?\s*(?P<n>\d{1,5}(?:\.\d{1,2})?)\b"),
    # c001 / c1050
    re.compile(r"(?i)\bc(?P<n>\d{1,5}(?:\.\d{1,2})?)\b"),
    # #1 (Western comic issue convention)
    re.compile(r"#(?P<n>\d{1,5}(?:\.\d{1,2})?)"),
]

# Special-issue markers — flip ``is_special`` without being a chapter number.
_SPECIAL_RE = re.compile(
    r"(?i)\b(?:extra|bonus|special|omake|side[\s-]?story|sp|prologue|epilogue)\b"
)

# Same keywords used for bracket classification: a ``(Bonus)`` shouldn't
# be misread as a scan-group name.
_SPECIAL_KEYWORDS = {
    "extra", "bonus", "special", "omake", "side-story", "sidestory",
    "sp", "prologue", "epilogue",
}

# Bracketed-content classifiers. Ordered by specificity.
_SOURCE_KEYWORDS = {
    "digital", "web", "webrip", "webtoon", "scan", "scanlation",
    "official", "physical", "magazine", "tankobon", "tank",
}
_QUALITY_KEYWORDS = {"hq", "lq", "hd", "uhd", "1080p", "720p", "4k", "raw"}
_LANGUAGE_CODES = {
    # ISO 639-1 two-letter subset seen in scan-group filenames
    "jp", "ja", "en", "us", "uk", "de", "fr", "es", "it", "pt", "br",
    "ru", "cn", "zh", "kr", "ko", "nl", "pl", "tr", "id", "vi", "th",
}
_LANGUAGE_NAMES = {
    "japanese", "english", "german", "french", "spanish", "italian",
    "portuguese", "russian", "chinese", "korean", "dutch", "polish",
    "turkish", "indonesian", "vietnamese", "thai",
}

# --- Output shape ------------------------------------------------------------


@dataclass(slots=True)
class ParsedFilename:
    """Structured metadata extracted from a comic archive filename.

    All fields except ``series``, ``confidence``, ``raw_stem`` can be absent.
    ``confidence`` is a float in [0.0, 1.0]; callers use it to gate whether
    to trust the parse or surface the item for human review.
    """

    series: str = ""
    volume: int | None = None
    volume_end: int | None = None          # for omnibus ranges (v1-3)
    chapter: float | None = None           # float supports decimals (12.5)
    year: int | None = None
    scan_group: str | None = None
    source: str | None = None              # Digital | Web | Scan | Official | ...
    language: str | None = None            # ISO code or full name, lowercased
    quality: str | None = None             # HD / HQ / LQ / ...
    is_special: bool = False               # Extra / Bonus / SP / Omake
    confidence: float = 0.0
    raw_stem: str = ""                     # original filename without extension


# --- Helpers -----------------------------------------------------------------


def _strip_extension(filename: str) -> str:
    """Drop the final extension (``.cbz``, ``.cbr``, ``.zip``, ``.rar``) only."""
    p = Path(filename)
    # Only strip if extension is a known archive format; otherwise the
    # "extension" might actually be part of the title (e.g. "v1.5").
    if p.suffix.lower() in {".cbz", ".cbr", ".zip", ".rar", ".7z", ".cbt", ".tar"}:
        return p.stem
    return filename


def _classify_bracketed(content: str) -> tuple[str, str | None]:
    """Classify a bracketed/parenthesized token.

    Returns ``(category, normalized_value)`` where category is one of:
    ``year``, ``source``, ``language``, ``quality``, ``group``, or ``unknown``.
    ``group`` is the fallback — anything we didn't otherwise recognize.
    """
    s = content.strip()
    if not s:
        return ("unknown", None)
    lower = s.lower()

    if _YEAR_RE.match(s):
        return ("year", s)

    if lower in _SPECIAL_KEYWORDS:
        return ("special", lower)

    if lower in _LANGUAGE_CODES:
        return ("language", lower)
    if lower in _LANGUAGE_NAMES:
        return ("language", lower)

    # Source + quality may coexist in one bracket as "Digital-HD" — split.
    parts = re.split(r"[-/ ]", lower)
    matched_source = next((p for p in parts if p in _SOURCE_KEYWORDS), None)
    matched_quality = next((p for p in parts if p in _QUALITY_KEYWORDS), None)
    if matched_source and not matched_quality:
        return ("source", matched_source)
    if matched_quality and not matched_source:
        return ("quality", matched_quality)
    if matched_source and matched_quality:
        # Caller handles dual classification — report primary as source.
        return ("source_quality", f"{matched_source}|{matched_quality}")

    return ("group", s)


def _extract_brackets(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Pop all bracketed/parenthesized tokens from ``text``.

    Returns ``(text_without_brackets, [(category, value), ...])``.
    Processes ``()``, ``[]``, ``{}`` uniformly. Does not support nested
    brackets — real filenames rarely nest, and treating nested as flat
    still yields correct classification for the common cases.
    """
    # Capture groups in order of appearance. Non-greedy so we match shortest
    # balanced pair per opening.
    pattern = re.compile(r"[(\[{]([^()\[\]{}]*?)[)\]}]")
    collected: list[tuple[str, str]] = []

    def _pop(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        category, value = _classify_bracketed(inner)
        if value is not None:
            if category == "source_quality":
                src, qual = value.split("|", 1)
                collected.append(("source", src))
                collected.append(("quality", qual))
            else:
                collected.append((category, value))
        return " "  # replace match with space to preserve surrounding tokens

    stripped = pattern.sub(_pop, text)
    return stripped, collected


def _extract_volume(text: str) -> tuple[str, int | None, int | None]:
    """Find the first volume marker. Returns (remaining, volume, volume_end)."""
    for pat in _VOLUME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        remaining = (text[:m.start()] + " " + text[m.end():])
        groups = m.groupdict()
        if "a" in groups and groups.get("a") and groups.get("b"):
            return remaining, int(groups["a"]), int(groups["b"])
        if "n" in groups and groups.get("n"):
            return remaining, int(groups["n"]), None
    return text, None, None


def _extract_chapter(text: str) -> tuple[str, float | None]:
    """Find the first chapter marker. Returns (remaining, chapter_number)."""
    for pat in _CHAPTER_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        remaining = (text[:m.start()] + " " + text[m.end():])
        n = m.group("n")
        return remaining, float(n)
    return text, None


def _clean_title(text: str) -> str:
    """Normalize whitespace, separator characters, trailing dashes.

    Real filenames use ``.`` and ``_`` as word separators almost as often
    as space. Convert them to spaces, collapse runs, strip trailing
    punctuation that's left after meta extraction.
    """
    # Underscore → space
    text = text.replace("_", " ")
    # Dots between words → space (but keep decimal-number dots; we already
    # extracted chapters, so remaining dots are usually separators)
    text = re.sub(r"\.(?=\s|[A-Z\u3040-\u30ff\u4e00-\u9fff])", " ", text)
    # Dots everywhere else → space (conservative — titles like "Mr." become "Mr")
    text = text.replace(".", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip trailing connector chars
    text = re.sub(r"[-–—:;,.\s]+$", "", text)
    # Strip leading junk
    text = re.sub(r"^[-–—:;,.\s]+", "", text)
    return text.strip()


def _compute_confidence(
    *,
    has_title: bool,
    has_volume: bool,
    has_chapter: bool,
    has_year: bool,
    has_group: bool,
    has_source: bool,
) -> float:
    """Weighted sum of recognized fields, capped at 1.0.

    Weights are tuned so that a bare title scores 0.3 (barely usable),
    a title + volume scores 0.55 (typical scan), and a title + volume +
    year scores 0.75 (typical high-quality scan).
    """
    score = 0.0
    if has_title:
        score += 0.30
    if has_volume:
        score += 0.25
    if has_chapter:
        score += 0.15
    if has_year:
        score += 0.20
    if has_group:
        score += 0.05
    if has_source:
        score += 0.05
    return min(1.0, score)


# --- Public API --------------------------------------------------------------


def parse_filename(filename: str) -> ParsedFilename:
    """Parse a comic archive filename into structured metadata.

    Examples
    --------
    >>> parse_filename("Berserk v01 (2003) (Digital) (LuCaZ).cbz").series
    'Berserk'
    >>> parse_filename("Berserk v01 (2003) (Digital) (LuCaZ).cbz").volume
    1
    >>> parse_filename("Berserk v01 (2003) (Digital) (LuCaZ).cbz").year
    2003
    >>> parse_filename("[Hox] 呪術廻戦 第01巻.cbz").volume
    1
    >>> parse_filename("Berserk Deluxe v1-3 (2019).cbz").volume_end
    3
    >>> parse_filename("Berserk Ch. 12.5 (Bonus).cbz").chapter
    12.5
    """
    result = ParsedFilename(raw_stem=_strip_extension(filename))
    # Underscores are separators in "Series_Name_Vol_1" style filenames, but
    # ``\b`` treats ``_`` as a word character, so volume/chapter regexes fail
    # against them. Normalize to spaces up front. Dots stay — they're
    # meaningful in decimal chapters (12.5) and we strip them in title cleanup.
    working = result.raw_stem.replace("_", " ")

    # 1. Pop bracketed/parenthesized tokens, classify them
    working, tokens = _extract_brackets(working)

    # First ``group`` token is the scan group; subsequent groups are ignored
    # (users who want fine-grained release tracking need a separate field).
    group_seen = False
    for category, value in tokens:
        if category == "year" and result.year is None:
            try:
                y = int(value)
                if 1900 <= y <= 2099:
                    result.year = y
            except ValueError:
                pass
        elif category == "source" and result.source is None:
            result.source = value
        elif category == "language" and result.language is None:
            result.language = value
        elif category == "quality" and result.quality is None:
            result.quality = value
        elif category == "special":
            result.is_special = True
        elif category == "group" and not group_seen:
            result.scan_group = value
            group_seen = True

    # 2. Extract volume / chapter markers
    working, result.volume, result.volume_end = _extract_volume(working)
    working, result.chapter = _extract_chapter(working)

    # 3. Detect special-issue markers
    if _SPECIAL_RE.search(working):
        result.is_special = True
        working = _SPECIAL_RE.sub(" ", working)

    # 4. Whatever's left is the title
    result.series = _clean_title(working)

    # 5. Compute confidence
    result.confidence = _compute_confidence(
        has_title=bool(result.series),
        has_volume=result.volume is not None,
        has_chapter=result.chapter is not None,
        has_year=result.year is not None,
        has_group=result.scan_group is not None,
        has_source=result.source is not None,
    )

    return result
