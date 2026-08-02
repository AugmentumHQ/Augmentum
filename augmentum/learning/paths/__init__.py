"""Hand-curated learning paths per language.

Replaces the old "top 30 by raw frequency" seeding strategy with
pedagogically-sequenced thematic units. Frequency-sorted vocab works
*after* a learner has hundreds of words of footing — for first contact
it dumps articles and pronouns. Curated paths drop content words
first: greetings, family, food, time. CEFR/HSK/JLPT-aligned.

Files live alongside this module as ``{lang}.json``. See ``_README.md``
for the schema and the authoring rules.

Lookup contract:
    available_langs()                       -> ["es", "zh", ...]
    load_path(lang_code)                    -> dict | None
    starter_surfaces(lang_code, n=30)       -> list[str]   (first-unit surfaces, capped)
    unit_surfaces(lang_code, unit_id)       -> list[str]   (one unit's vocab)

The loader caches each JSON on first access. Paths are read-only; edits
take effect after process restart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_PATHS_DIR = Path(__file__).parent
_CACHE: dict[str, dict[str, Any] | None] = {}
_AUX_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}

# Aux files use `{lang}-{kind}.json` naming. Recognised aux kinds:
#   grammar    — grammar drill curriculum (sequenced morphology / syntax)
#   assessment — 4-skill rubric + LLM grader prompts
#   kanji      — kanji decomposition (ja-only)
#   tones      — tone-pair drill content (zh-only)
#   characters — character decomposition (zh-only)
_AUX_KINDS = ("grammar", "assessment", "kanji", "tones", "characters")


def _path_file(lang_code: str) -> Path:
    return _PATHS_DIR / f"{lang_code}.json"


def _aux_file(lang_code: str, kind: str) -> Path:
    return _PATHS_DIR / f"{lang_code}-{kind}.json"


def available_langs() -> list[str]:
    """ISO lang codes that ship a curated **vocab** path file.

    Only returns stems WITHOUT a hyphen — auxiliary files like
    ``es-grammar.json`` / ``ja-kanji.json`` / ``zh-assessment.json``
    use the ``{lang}-{kind}.json`` naming convention and are reached
    via :func:`load_aux` instead. Without this filter, downstream
    consumers that iterate ``available_langs()`` and assume each entry
    is a vocab path (with ``.levels[].units[]`` shape) would break on
    the aux files (which have completely different schemas).
    """
    out: list[str] = []
    for entry in _PATHS_DIR.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        if entry.name.startswith("_"):
            continue
        stem = entry.stem
        if "-" in stem:
            continue   # auxiliary file
        out.append(stem)
    return sorted(out)


def load_aux(lang_code: str, kind: str) -> dict[str, Any] | None:
    """Load (and cache) an auxiliary curriculum file: grammar, assessment,
    kanji, tones, or characters. Returns ``None`` if the file doesn't
    exist for this language — callers should treat missing aux as "no
    such content yet" and degrade gracefully.

    Aux files have lang-specific schemas (the grammar drill schema
    differs from the assessment rubric schema), so callers must know
    which kind they're requesting and how to interpret the response.
    """
    if not lang_code or kind not in _AUX_KINDS:
        return None
    key = (lang_code, kind)
    if key in _AUX_CACHE:
        return _AUX_CACHE[key]
    file = _aux_file(lang_code, kind)
    if not file.exists():
        _AUX_CACHE[key] = None
        return None
    try:
        with file.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("learning_aux_load_failed", lang=lang_code, kind=kind, error=str(exc))
        _AUX_CACHE[key] = None
        return None
    _AUX_CACHE[key] = data
    return data


def available_aux(lang_code: str) -> list[str]:
    """Which aux kinds exist for this lang. ``['grammar','assessment']``
    means both files ship and ``load_aux(lang, kind)`` will return data."""
    if not lang_code:
        return []
    out = []
    for kind in _AUX_KINDS:
        if _aux_file(lang_code, kind).exists():
            out.append(kind)
    return out


def load_path(lang_code: str) -> dict[str, Any] | None:
    """Load (and cache) the curated path for ``lang_code``. Returns
    ``None`` if no path file is shipped for this language — callers
    should fall back to frequency-based selection in that case.
    """
    if not lang_code:
        return None
    if lang_code in _CACHE:
        return _CACHE[lang_code]
    file = _path_file(lang_code)
    if not file.exists():
        _CACHE[lang_code] = None
        return None
    try:
        with file.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("learning_path_load_failed", lang=lang_code, error=str(exc))
        _CACHE[lang_code] = None
        return None
    _CACHE[lang_code] = data
    return data


def _iter_units(path: dict) -> list[dict]:
    out: list[dict] = []
    for level in path.get("levels") or []:
        for unit in level.get("units") or []:
            out.append(unit)
    return out


def starter_surfaces(lang_code: str, n: int = 30) -> list[str]:
    """First ``n`` vocab surface forms from the path's opening units —
    used as the seed source. Walks units in order so a 30-word seed
    pulls roughly the first 1-2 units' worth of vocab. Surface forms
    are returned as-is; the caller resolves them against the pack's
    ``vocab`` table to get ``word_id`` values (and silently drops
    expressions / phrases the pack dictionary doesn't index, like
    'buenos días' or 'me llamo' which aren't JMdict/kaikki headwords).
    """
    path = load_path(lang_code)
    if not path:
        return []
    n = max(1, min(int(n), 500))
    seen: set[str] = set()
    out: list[str] = []
    for unit in _iter_units(path):
        for vocab in unit.get("vocab") or []:
            s = (vocab.get("surface") or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= n:
                return out
    return out


def unit_surfaces(lang_code: str, unit_id: str) -> list[str]:
    """All vocab surfaces from one specific unit."""
    path = load_path(lang_code)
    if not path:
        return []
    for unit in _iter_units(path):
        if unit.get("id") == unit_id:
            return [
                (v.get("surface") or "").strip()
                for v in (unit.get("vocab") or [])
                if (v.get("surface") or "").strip()
            ]
    return []


def path_summary(lang_code: str) -> dict[str, Any] | None:
    """Lightweight metadata for the API — drops the per-vocab arrays so
    the response stays small. Use ``load_path`` when you need full
    content (a specific unit's vocab + phrases)."""
    path = load_path(lang_code)
    if not path:
        return None
    levels_out = []
    for level in path.get("levels") or []:
        units_out = []
        for unit in level.get("units") or []:
            units_out.append({
                "id": unit.get("id"),
                "title": unit.get("title"),
                "theme": unit.get("theme"),
                "goal": unit.get("goal"),
                "vocab_count": len(unit.get("vocab") or []),
                "phrase_count": len(unit.get("phrases") or []),
                "estimated_minutes": unit.get("estimated_minutes"),
            })
        levels_out.append({
            "code": level.get("code"),
            "name": level.get("name"),
            "can_do": level.get("can_do"),
            "units": units_out,
        })
    return {
        "lang": path.get("lang"),
        "name": path.get("name"),
        "credits": path.get("credits"),
        "license": path.get("license"),
        "level_system": path.get("level_system"),
        "levels": levels_out,
    }
