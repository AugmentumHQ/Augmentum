"""File tag normalisation + autocomplete helpers.

Tags on file_index rows are stored as a JSON array of display strings,
but the matching/dedup story should be based on a canonical form so
"Photo", "photo", "PHOTO " all collapse to the same bucket.

`normalize_tag()` produces that canonical form; `normalize_tags()`
applies it to a list and dedups while preserving the first-seen display
spelling. `suggest_tags()` powers the autocomplete endpoint.
"""

from __future__ import annotations

import json
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


# --- Canonicalisation ---------------------------------------------------

def canonical(tag: str) -> str:
    """Lowercase, NFKC-normalised, whitespace-collapsed form of a tag.

    "Photo " and "photo" collapse to "photo"; full-width "ＰＨＯＴＯ"
    NFKC-normalises to "photo".  Used for dedup + autocomplete matching;
    callers preserve the original display form for rendering.
    """
    if not tag:
        return ""
    t = unicodedata.normalize("NFKC", str(tag))
    return " ".join(t.lower().split())


def normalize_tag(tag: str, *, max_len: int = 100) -> str:
    """Trim + length-cap a tag for storage.  Keeps the original casing
    (display form), strips control bytes, drops the tag entirely if it
    only contains whitespace.
    """
    if not tag:
        return ""
    t = unicodedata.normalize("NFKC", str(tag))
    # Drop control chars + zero-width separators that would render badly.
    t = "".join(ch for ch in t if ch.isprintable() or ch == " ")
    t = " ".join(t.split())
    if not t:
        return ""
    return t[:max_len]


def normalize_tags(
    raw: list | None, *, max_count: int = 50, max_len: int = 100,
) -> list[str]:
    """Clean + dedup a list of tags.

    Dedup is by canonical form so "Photo" and "photo" collapse to one
    entry — the first display spelling wins.  Drops empties.  Caps the
    count at `max_count` (matches the existing route validation).
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        cleaned = normalize_tag(item, max_len=max_len)
        if not cleaned:
            continue
        key = canonical(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_count:
            break
    return out


# --- Autocomplete -------------------------------------------------------

async def suggest_tags(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    prefix: str = "",
    limit: int = 20,
) -> list[str]:
    """Return up to `limit` distinct tags for the user, ordered by usage
    frequency. When `prefix` is given, restrict to canonical-form matches.

    Tags are stored as JSON arrays in file_index.tags so we can't push
    the prefix filter into SQL — we read all rows for the user and
    aggregate in Python. Cheap at file counts up to ~10k; revisit with a
    proper tag table if the catalogue grows past that.
    """
    if not user_id:
        return []
    cursor = await conn.execute(
        "SELECT tags FROM file_index WHERE user_id = ? AND is_trashed = 0",
        (user_id,),
    )
    rows = await cursor.fetchall()

    counts: dict[str, tuple[str, int]] = {}  # canonical -> (display, count)
    needle = canonical(prefix)
    for (raw,) in rows:
        if not raw:
            continue
        try:
            tags = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(tags, list):
            continue
        for t in tags:
            cleaned = normalize_tag(t)
            if not cleaned:
                continue
            key = canonical(cleaned)
            if needle and not key.startswith(needle):
                continue
            display, n = counts.get(key, (cleaned, 0))
            counts[key] = (display, n + 1)

    # Sort by count desc, then display asc for stable pagination.
    ordered = sorted(counts.values(), key=lambda x: (-x[1], x[0].lower()))
    return [display for display, _ in ordered[: max(1, int(limit))]]
