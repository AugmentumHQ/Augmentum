"""Language-pack lookup — queries the ``vocab`` / ``sentences`` tables of
a ``pack_kind=language`` ``.augpack``.

Read-only. The connection is owned by :class:`augmentum.knowledge.packs.PackManager`'s
language-pack registry — get it via ``pack_manager.get_language_pack(lang_code).conn``.
The language-learning HTTP routes call these helpers directly.

Japanese has no word spaces, so click-to-define resolves the word at a
click position by *longest-prefix match* against the dictionary (the
approach Yomitan/Yomichan use) — no tokenizer dependency. Space-delimited
languages can pass the already-isolated token as ``text`` with ``pos=0``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Word-segmenter for space-delimited languages: returns word runs, whitespace
# runs, and single non-word characters as separate tokens. \w in Python's
# default ``re`` covers Unicode letters/digits/_, so this works for Spanish,
# French, Korean (eojeol-level), etc.
_WORD_RE = re.compile(r"\w+|\s+|.", re.UNICODE)

# Japanese headwords are short; this caps how far past the click position
# we look for the longest matching dictionary entry.
_MAX_PREFIX_CHARS = 16

_COLS = ("word_id", "surface", "reading", "pos", "glosses", "freq_rank", "jlpt", "level")
_REQUIRED_COLS = ("word_id", "surface", "reading", "pos", "glosses", "freq_rank", "jlpt")
_COL_CACHE: dict[int, tuple[str, ...]] = {}


async def _vocab_columns(conn: aiosqlite.Connection) -> tuple[str, ...]:
    """Columns available in this pack's ``vocab`` table.

    Newer packs carry a generic ``level`` band (A1/N5/HSK1/etc.), but older
    local packs may predate it. Detect once per connection so adding level
    metadata does not strand existing installs.
    """
    key = id(conn)
    cached = _COL_CACHE.get(key)
    if cached:
        return cached
    try:
        cursor = await conn.execute("PRAGMA table_info(vocab)")
        rows = await cursor.fetchall()
        available = {str(r[1]) for r in rows}
    except aiosqlite.Error:
        available = set(_REQUIRED_COLS)
    cols = tuple(c for c in _COLS if c in available)
    if not cols:
        cols = _REQUIRED_COLS
    _COL_CACHE[key] = cols
    return cols


def _row_to_entry(row: tuple, cols: list[str]) -> dict[str, Any]:
    d = dict(zip(cols, row, strict=True))
    raw = d.get("glosses")
    if isinstance(raw, str) and raw:
        try:
            d["glosses"] = json.loads(raw)
        except json.JSONDecodeError:
            d["glosses"] = [raw]
    elif raw is None:
        d["glosses"] = []
    return d


def _fts_phrase(query: str) -> str:
    """Wrap a user string as a literal FTS5 phrase (escapes embedded quotes)."""
    return '"' + query.replace('"', '""') + '"'


async def _query_in(conn: aiosqlite.Connection, column: str, values: list[str]) -> list[dict]:
    if not values:
        return []
    placeholders = ",".join("?" * len(values))
    cols = await _vocab_columns(conn)
    select = ", ".join(cols)
    cursor = await conn.execute(
        f"SELECT {select} FROM vocab WHERE {column} IN ({placeholders})", values
    )
    rows = await cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    return [_row_to_entry(r, cols) for r in rows]


async def lookup_at(
    conn: aiosqlite.Connection,
    text: str,
    pos: int = 0,
    *,
    max_len: int = _MAX_PREFIX_CHARS,
) -> list[dict]:
    """Longest-prefix dictionary lookup at character offset ``pos`` in
    ``text``. Returns the entries matching at the longest length (usually
    one; more for a homograph headword), or ``[]``.

    Tries the kanji ``surface`` column first, then the kana ``reading``,
    so a click in Japanese text resolves whether the run is kanji or kana.
    """
    if not text or pos < 0 or pos >= len(text):
        return []
    span = text[pos : pos + max_len]
    candidates = [span[:k] for k in range(1, len(span) + 1)]
    candset = set(candidates)
    for column in ("surface", "reading"):
        hits = [h for h in await _query_in(conn, column, candidates) if h[column] in candset]
        if not hits:
            continue
        best = max(len(h[column]) for h in hits)
        return [h for h in hits if len(h[column]) == best]
    return []


async def resolve_surfaces(
    conn: aiosqlite.Connection, surfaces: list[str],
) -> dict[str, str]:
    """Resolve surface forms to ``word_id`` values using seeder semantics.

    The returned mapping is keyed by the caller's original surface string.
    Exact-case matches win, then unresolved items retry lowercased. This is the
    explainable form of :func:`lookup_surfaces`: seeders need only IDs, while
    content audits need to say which unit surfaces missed the pack.
    """
    if not surfaces:
        return {}
    rows = await _query_in(conn, "surface", surfaces)
    by_surface: dict[str, str] = {}
    for row in rows:
        surf = row.get("surface")
        wid = row.get("word_id")
        if surf and wid and surf not in by_surface:
            by_surface[surf] = wid

    unresolved = [s for s in surfaces if s and s not in by_surface]
    lowered_to_originals: dict[str, list[str]] = {}
    for s in unresolved:
        lo = s.lower()
        if lo != s:
            lowered_to_originals.setdefault(lo, []).append(s)
    if lowered_to_originals:
        lo_rows = await _query_in(conn, "surface", list(lowered_to_originals.keys()))
        for row in lo_rows:
            surf = row.get("surface")
            wid = row.get("word_id")
            if not surf or not wid:
                continue
            for original in lowered_to_originals.get(surf, []):
                by_surface.setdefault(original, wid)

    return {s: by_surface[s] for s in surfaces if s in by_surface}


async def lookup_surfaces(
    conn: aiosqlite.Connection, surfaces: list[str],
) -> list[str]:
    """Resolve a list of surface forms to the pack's ``word_id`` values.

    Order-preserving - the returned list mirrors the input order, with misses
    (multi-word expressions, surfaces the dictionary doesn't index) silently
    dropped. Used by the curated-path seeder to map path surfaces to pack IDs.
    """
    resolved = await resolve_surfaces(conn, surfaces)
    return [resolved[s] for s in surfaces if s in resolved]

async def lookup_text(conn: aiosqlite.Connection, query: str, *, limit: int = 10) -> list[dict]:
    """Free-text lookup: exact surface/reading match first, then an FTS5
    MATCH over surface/reading/glosses. De-duplicated, capped at ``limit``."""
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 50))
    seen: set[str] = set()
    out: list[dict] = []
    for h in (
        await _query_in(conn, "surface", [query]) + await _query_in(conn, "reading", [query])
    ):
        if h["word_id"] not in seen:
            seen.add(h["word_id"])
            out.append(h)
    try:
        cols = await _vocab_columns(conn)
        select_v = ", ".join(f"v.{c}" for c in cols)
        cursor = await conn.execute(
            f"SELECT {select_v} FROM vocab_fts f JOIN vocab v ON v.rowid = f.rowid "
            "WHERE vocab_fts MATCH ? LIMIT ?",
            (_fts_phrase(query), limit * 3),
        )
        rows = await cursor.fetchall()
        cols = [c[0] for c in cursor.description]
        for r in rows:
            h = _row_to_entry(r, cols)
            if h["word_id"] not in seen:
                seen.add(h["word_id"])
                out.append(h)
    except aiosqlite.Error as exc:
        # FTS table missing or malformed query — exact matches still stand.
        log.debug("lang_pack_fts_failed", query=query, error=str(exc))
    return out[:limit]


async def top_frequency(conn: aiosqlite.Connection, n: int = 30) -> list[dict]:
    """The ``n`` most-frequent entries that carry a corpus-derived
    ``freq_rank`` (rank 1 = most frequent). Empty if the pack was built
    without the Tatoeba sentence corpus (no frequency data). Used to seed
    a beginner's starter queue with high-utility vocabulary.
    """
    n = max(1, min(int(n), 1000))
    cols = await _vocab_columns(conn)
    select = ", ".join(cols)
    cursor = await conn.execute(
        f"SELECT {select} FROM vocab WHERE freq_rank IS NOT NULL "
        "ORDER BY freq_rank ASC LIMIT ?",
        (n,),
    )
    rows = await cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    return [_row_to_entry(r, cols) for r in rows]


async def get_entry(conn: aiosqlite.Connection, word_id: str) -> dict | None:
    """Fetch a single vocab entry by ``word_id`` (JMdict ``<ent_seq>``)."""
    if not word_id:
        return None
    cols = await _vocab_columns(conn)
    select = ", ".join(cols)
    cursor = await conn.execute(f"SELECT {select} FROM vocab WHERE word_id = ?", (word_id,))
    row = await cursor.fetchone()
    if not row:
        return None
    return _row_to_entry(row, [c[0] for c in cursor.description])


async def tokenize_segment(
    conn: aiosqlite.Connection,
    text: str,
    *,
    max_token_len: int = _MAX_PREFIX_CHARS,
    mode: str | None = None,
) -> list[dict]:
    """Segment ``text`` into dictionary tokens (the "break down a
    highlighted phrase" path). Returns a list of token dicts: matched
    tokens carry the full vocab entry plus ``text`` (the matched slice)
    and ``matched: True``; raw tokens are ``{"text": ..., "matched": False}``.

    ``mode``:
      * ``"longest_prefix"`` — CJK path (no word spaces). At each position,
        take the longest dictionary surface/reading that's a prefix.
      * ``"whitespace"`` — space-delimited path (Spanish, French, Korean
        eojeol). Splits on word boundaries first, then exact-matches each
        word against the dictionary. Punctuation + spaces are emitted as
        raw tokens.
      * ``None`` (default) — read the pack's ``meta.tokenization`` and
        dispatch. Falls back to ``"longest_prefix"`` for packs predating
        that field.
    """
    if not text:
        return []
    if mode is None:
        mode = await pack_tokenization(conn)
    if mode == "whitespace":
        return await _tokenize_whitespace(conn, text)
    return await _tokenize_longest_prefix(conn, text, max_token_len=max_token_len)


async def _tokenize_longest_prefix(
    conn: aiosqlite.Connection, text: str, *, max_token_len: int = _MAX_PREFIX_CHARS,
) -> list[dict]:
    out: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        hits = await lookup_at(conn, text, i, max_len=min(max_token_len, n - i))
        if not hits:
            out.append({"text": text[i], "matched": False})
            i += 1
            continue
        entry = hits[0]
        # Advance by the length of whichever surface/reading form is the
        # prefix at this position (lookup_at returned the longest match).
        mlen = 0
        for form in (entry.get("surface") or "", entry.get("reading") or ""):
            if form and text.startswith(form, i) and len(form) > mlen:
                mlen = len(form)
        if mlen == 0:
            out.append({"text": text[i], "matched": False})
            i += 1
            continue
        out.append({**entry, "text": text[i : i + mlen], "matched": True})
        i += mlen
    return out


async def _tokenize_whitespace(conn: aiosqlite.Connection, text: str) -> list[dict]:
    """Word-boundary tokeniser. Looks up each \\w+ run exactly (no prefix
    search — the boundary already isolated the word). Surrounding spaces
    and punctuation become raw tokens, preserving the original layout so
    the UI can re-render the span verbatim."""
    out: list[dict] = []
    parts = _WORD_RE.findall(text)
    for part in parts:
        if part.isspace() or (len(part) == 1 and not part.isalnum() and not part.isalpha()):
            out.append({"text": part, "matched": False})
            continue
        # Whole-word exact lookup. Case-insensitive — surface is stored as
        # the canonical form in the pack; users hit it however they typed.
        hits = await _query_in(conn, "surface", [part])
        if not hits:
            # Try lowercased — Spanish dictionaries are typically lowercase.
            lower = part.lower()
            if lower != part:
                hits = await _query_in(conn, "surface", [lower])
        if not hits:
            out.append({"text": part, "matched": False})
            continue
        out.append({**hits[0], "text": part, "matched": True})
    return out


async def read_sentences(
    conn: aiosqlite.Connection,
    *,
    n: int = 20,
    max_difficulty: int = 3,
    contains: str | None = None,
    require_translation: bool = True,
) -> list[dict]:
    """A batch of sentences for the reading surface.

    Returns ``{sent_id, lang_text, en_text, difficulty}`` dicts. By
    default: short-to-medium sentences that carry an English translation,
    in random order. ``contains`` restricts to sentences containing that
    surface form in context. For space-delimited scripts this means an
    isolated word, not a substring inside another word.
    Real i+1 vocab-coverage grading is a later phase; for v1 this is just
    "give me some readable sentences".
    """
    n = max(1, min(int(n), 100))
    contains = (contains or "").strip()
    substring_safe = bool(contains and _surface_is_substring_safe(contains))
    needs_boundary_filter = bool(contains and not substring_safe)
    candidate_limit = min(500, max(n * 25, 50)) if needs_boundary_filter else n

    where = ["difficulty <= ?"]
    params: list = [int(max_difficulty)]
    if require_translation:
        where.append("en_text IS NOT NULL AND en_text != ''")
    if contains:
        where.append("lang_text LIKE '%' || ? || '%'")
        params.append(contains)
    params.append(candidate_limit)
    cursor = await conn.execute(
        "SELECT sent_id, lang_text, en_text, difficulty FROM sentences "
        f"WHERE {' AND '.join(where)} ORDER BY RANDOM() LIMIT ?",
        params,
    )
    rows = await cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    out = [dict(zip(cols, r, strict=True)) for r in rows]
    if needs_boundary_filter:
        out = [row for row in out if _surface_is_isolated(row.get("lang_text") or "", contains)]
    return out[:n]


async def count_sentences(
    conn: aiosqlite.Connection,
    *,
    max_difficulty: int = 3,
    require_translation: bool = True,
) -> int:
    """Count sentences matching the same coarse filters as ``read_sentences``.

    Used by readiness surfaces so games that need translated sentence material
    can be gated before launch without fetching full sentence payloads.
    """
    where = ["difficulty <= ?"]
    params: list = [int(max_difficulty)]
    if require_translation:
        where.append("en_text IS NOT NULL AND en_text != ''")
    cursor = await conn.execute(
        f"SELECT COUNT(*) FROM sentences WHERE {' AND '.join(where)}", params,
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0

# CJK character ranges where substring match against a *sentence* is
# safe: a single Japanese kanji like 歯 doesn't appear as a coincidental
# letter pair inside another word. Latin script needs word-boundary
# discipline — otherwise Spanish "la" matches every sentence containing
# "Ho**la**", "ha**bla**r", "esca**la**ra"…
_SUBSTRING_SAFE_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
)


def _surface_is_substring_safe(surface: str) -> bool:
    if not surface:
        return False
    for ch in surface:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _SUBSTRING_SAFE_RANGES):
            return True
    return False


def _has_latin_letter(text: str) -> bool:
    return any(
        ch.isalpha() and "LATIN" in unicodedata.name(ch, "")
        for ch in text
    )


def is_gameworthy_surface(surface: str) -> bool:
    """True when a vocab surface is useful as a standalone game card.

    Single Latin letters are usually alphabet names, abbreviations, or very
    low-value function words. They make visual games feel broken and produce
    noisy sentence examples, so games hide them while dictionary lookup still
    supports them.
    """
    surface = (surface or "").strip()
    if not surface:
        return False
    return not (len(surface) == 1 and _has_latin_letter(surface))


_BOUNDARY_CHARS = " \t\n　.,;:!?¡¿\"'()[]{}—–-…«»\"“”‘’/"


def _surface_is_isolated(text: str, surface: str) -> bool:
    """``True`` if any occurrence of ``surface`` in ``text`` sits next
    to word-boundary characters (whitespace, punctuation, line edges).
    Used to filter out coincidental substring hits like 'la' inside 'Hola'.
    """
    if not surface or not text:
        return False
    haystack = text.lower()
    needle = surface.lower()
    n = len(haystack)
    sn = len(needle)
    i = 0
    while True:
        idx = haystack.find(needle, i)
        if idx < 0:
            return False
        left_ok = idx == 0 or text[idx - 1] in _BOUNDARY_CHARS
        end = idx + sn
        right_ok = end == n or text[end] in _BOUNDARY_CHARS
        if left_ok and right_ok:
            return True
        i = idx + 1


async def get_example(
    conn: aiosqlite.Connection, surface: str, *, en_required: bool = False
) -> dict | None:
    """An example sentence demonstrating ``surface``, easiest first.

    For CJK kanji/kana the bare substring ``LIKE`` is exact — a single
    ideograph doesn't appear coincidentally inside another word. For
    space-segmented scripts (es/fr/ko/en) substring matching is wrong:
    Spanish "la" appears inside "Hola", "hablar", "escalera"… without
    *being* the word "la". So we widen the candidate window, then
    Python-filter for word-boundary isolation. The candidate fetch is
    capped — sentence corpora are large enough that the boundary pass
    almost always finds a match in the first batch.
    """
    surface = (surface or "").strip()
    if not surface:
        return None
    substring_safe = _surface_is_substring_safe(surface)
    candidate_limit = 1 if substring_safe else 200
    sql = "SELECT lang_text, en_text FROM sentences WHERE lang_text LIKE ? "
    if en_required:
        sql += "AND en_text IS NOT NULL "
    sql += "ORDER BY difficulty ASC, sent_id ASC LIMIT ?"
    cursor = await conn.execute(sql, (f"%{surface}%", candidate_limit))
    rows = await cursor.fetchall()
    if not rows:
        return None
    if substring_safe:
        row = rows[0]
        return {"lang_text": row[0], "en_text": row[1]}
    # Non-CJK: require word-boundary isolation. Returning no example is
    # better than showing a misleading substring hit like "a" inside "casa".
    for row in rows:
        if _surface_is_isolated(row[0], surface):
            return {"lang_text": row[0], "en_text": row[1]}
    return None

async def pack_meta(conn: aiosqlite.Connection) -> dict[str, str]:
    """Read the pack's ``meta`` key/value table into a dict."""
    cursor = await conn.execute("SELECT key, value FROM meta")
    return {k: v for k, v in await cursor.fetchall()}


async def pack_pos_labels(conn: aiosqlite.Connection) -> dict[str, str]:
    """Read the pack's POS code → human label map from ``meta.pos_labels``.

    Returns ``{}`` if the pack predates this metadata (callers should fall
    back to a per-language default — e.g. the JMdict map for ja packs)."""
    cursor = await conn.execute("SELECT value FROM meta WHERE key = 'pos_labels'")
    row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    try:
        parsed = json.loads(row[0])
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if k and v}


async def pack_tokenization(conn: aiosqlite.Connection) -> str:
    """Read ``meta.tokenization`` — ``"longest_prefix"`` (CJK) or
    ``"whitespace"`` (space-delimited). Defaults to ``"longest_prefix"``
    for packs that predate this field (all pre-existing packs are JP)."""
    cursor = await conn.execute("SELECT value FROM meta WHERE key = 'tokenization'")
    row = await cursor.fetchone()
    if row and row[0] in ("longest_prefix", "whitespace"):
        return row[0]
    return "longest_prefix"
