"""Per-voice TTS pronunciation lexicon (migration 261).

One row = "when speaking with <voice>, say <term> as <phonetics>".
Voice '' = every voice; a voice-specific row beats a '' row on the same
term. Layered ABOVE the global ``voice_tts_lexicon`` setting (which the
text-cleaning pass applies separately): DB entries rewrite the text
before ``clean_for_tts`` runs, so they take precedence over both the
setting and every built-in normalization rule.

Application is regex word-boundary replacement, longest term first (so
"SQL Server" wins over "SQL"), with literal — never re.sub-template —
replacement strings: a phonetics value containing a backslash must not
crash synthesis.

Rows with empty phonetics are skipped by apply() in v1 — the schema
reserves '' = shield-from-normalization (the setting's semantic), but
shielding requires coordination inside clean_for_tts and is deferred.

Cache: compiled patterns per (user_id, voice) pair, invalidated by a
module-global generation counter bumped on every write. Reads on the
hot speech path cost one dict lookup + int compare after warm-up.
"""

from __future__ import annotations

import re
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# (user_id, voice) → (generation, [(pattern, phonetics), ...])
_cache: dict[tuple[str, str], tuple[int, list[tuple[re.Pattern, str]]]] = {}
_generation: int = 0


def _bump() -> None:
    global _generation
    _generation += 1
    _cache.clear()


def normalize_voice(voice: str) -> str:
    """The voice string as the lexicon keys it: plain name, no
    'provider::' prefix, trimmed."""
    v = (voice or "").strip()
    if "::" in v:
        v = v.split("::")[-1].strip()
    return v


# ─── CRUD ───────────────────────────────────────────────────────────────


async def list_entries(
    conn: aiosqlite.Connection, *, user_id: str, voice: str | None = None,
) -> list[dict[str, Any]]:
    """All entries for the user, optionally filtered to one voice
    (plus the '' every-voice rows, which always apply)."""
    if voice is None:
        cur = await conn.execute(
            """SELECT id, voice, term, phonetics, created_at
               FROM tts_lexicon_entries WHERE user_id = ?
               ORDER BY voice, term COLLATE NOCASE""",
            (user_id,),
        )
    else:
        cur = await conn.execute(
            """SELECT id, voice, term, phonetics, created_at
               FROM tts_lexicon_entries
               WHERE user_id = ? AND voice IN ('', ?)
               ORDER BY voice, term COLLATE NOCASE""",
            (user_id, normalize_voice(voice)),
        )
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"id": int(r[0]), "voice": r[1], "term": r[2],
         "phonetics": r[3], "created_at": r[4]}
        for r in rows
    ]


async def add_entry(
    conn: aiosqlite.Connection, *,
    user_id: str, voice: str, term: str, phonetics: str,
) -> dict[str, Any] | None:
    """Insert or update (upsert on user+voice+term). Returns the row,
    or None on invalid input."""
    term = (term or "").strip()
    phonetics = (phonetics or "").strip()
    voice = normalize_voice(voice)
    if not user_id or not term:
        return None
    if len(term) > 80 or len(phonetics) > 200:
        return None
    await conn.execute(
        """INSERT INTO tts_lexicon_entries (user_id, voice, term, phonetics)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id, voice, term)
           DO UPDATE SET phonetics = excluded.phonetics""",
        (user_id, voice, term, phonetics),
    )
    await conn.commit()
    _bump()
    cur = await conn.execute(
        """SELECT id, voice, term, phonetics, created_at
           FROM tts_lexicon_entries
           WHERE user_id = ? AND voice = ? AND term = ?""",
        (user_id, voice, term),
    )
    r = await cur.fetchone()
    await cur.close()
    if r is None:
        return None
    return {"id": int(r[0]), "voice": r[1], "term": r[2],
            "phonetics": r[3], "created_at": r[4]}


async def remove_entry(
    conn: aiosqlite.Connection, *, entry_id: int, user_id: str,
) -> bool:
    cur = await conn.execute(
        "DELETE FROM tts_lexicon_entries WHERE id = ? AND user_id = ?",
        (int(entry_id), user_id),
    )
    affected = cur.rowcount or 0
    await cur.close()
    await conn.commit()
    if affected:
        _bump()
    return affected > 0


# ─── Application (hot speech path) ──────────────────────────────────────


async def _compiled_for(
    conn: aiosqlite.Connection, *, user_id: str, voice: str,
) -> list[tuple[re.Pattern, str]]:
    key = (user_id, voice)
    hit = _cache.get(key)
    if hit is not None and hit[0] == _generation:
        return hit[1]

    entries = await list_entries(conn, user_id=user_id, voice=voice)
    # Voice-specific beats '' on the same term (case-insensitive).
    by_term: dict[str, dict[str, Any]] = {}
    for e in entries:                      # '' rows sort first; specific
        key_term = e["term"].lower()       # rows overwrite them
        prev = by_term.get(key_term)
        if prev is None or (prev["voice"] == "" and e["voice"] != ""):
            by_term[key_term] = e
    compiled: list[tuple[re.Pattern, str]] = []
    for e in sorted(by_term.values(), key=lambda x: -len(x["term"])):
        if not e["phonetics"]:
            continue                       # shield semantic reserved (v2)
        compiled.append((
            re.compile(rf"(?<!\w){re.escape(e['term'])}(?!\w)", re.IGNORECASE),
            e["phonetics"],
        ))
    _cache[key] = (_generation, compiled)
    return compiled


async def apply(
    conn: aiosqlite.Connection | None, text: str, *,
    user_id: str, voice: str,
) -> str:
    """Rewrite ``text`` with the user's pronunciation entries for
    ``voice``. Never raises — a lexicon problem must not take down
    speech synthesis."""
    if conn is None or not text or not user_id:
        return text
    try:
        compiled = await _compiled_for(
            conn, user_id=user_id, voice=normalize_voice(voice),
        )
        for pattern, phonetics in compiled:
            # Literal replacement via lambda — phonetics containing a
            # backslash must not be parsed as a sub() template.
            text = pattern.sub(lambda _m, _p=phonetics: _p, text)
    except Exception:
        log.warning("tts_lexicon_apply_failed", exc_info=True)
    return text
