"""Learning-content QA audit.

This module ties together the existing catalog, curated paths, installed
language packs, and game material requirements. It deliberately does not own
curriculum or pack logic; it reads the same helpers used by learning routes and
turns them into an explainable report for development and release checks.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import aiosqlite

from augmentum.learning import lang_pack_catalog as catalog
from augmentum.learning import lang_packs
from augmentum.learning import paths as learning_paths

_UNIT_MIN_VOCAB = 15
_UNIT_MAX_VOCAB = 25
_SAMPLE_LIMIT = 20

_GAME_REQUIREMENTS: dict[str, dict[str, int]] = {
    "bubble_pop": {"words": 4},
    "word_garden": {"words": 1},
    "echo_chamber": {"words": 4},
    "whisper_race": {"words": 4},
    "story_weaver": {"words_or_discovery": 6},
    "word_forge": {"words": 8, "settled_words_recommended": 4},
    "constellation": {"words": 6},
    "mirror": {"words": 6, "translated_sentences": 4},
    "vocab_quest": {"words": 6},
}


def _finding(severity: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    out = {"severity": severity, "code": code, "message": message}
    out.update(extra)
    return out


def _ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 1.0
    return round(part / whole, 4)


def _sample(items: list[Any], limit: int = _SAMPLE_LIMIT) -> list[Any]:
    return items[: max(0, int(limit))]


def _unit_iter(path: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for level in path.get("levels") or []:
        for unit in level.get("units") or []:
            out.append((level, unit))
    return out


def _collect_path(lang: str) -> dict[str, Any]:
    path = learning_paths.load_path(lang)
    if not path:
        return {
            "present": False,
            "items": [],
            "stats": {
                "present": False,
                "level_system": "",
                "level_count": 0,
                "unit_count": 0,
                "vocab_count": 0,
                "unique_vocab_count": 0,
                "phrase_count": 0,
                "estimated_minutes": 0,
                "levels": [],
            },
            "findings": [_finding("warn", "path_missing", f"No curated path for {lang}")],
        }

    items: list[dict[str, Any]] = []
    unit_summaries: list[dict[str, Any]] = []
    level_summaries: list[dict[str, Any]] = []
    surface_counts: Counter[str] = Counter()
    missing_phrase_target: list[dict[str, str]] = []
    missing_phrase_translation: list[dict[str, str]] = []
    unit_size_warnings: list[dict[str, Any]] = []
    phrase_count = 0
    estimated_minutes = 0

    for level in path.get("levels") or []:
        level_vocab = 0
        level_phrases = 0
        units = level.get("units") or []
        for unit in units:
            unit_id = str(unit.get("id") or "")
            unit_title = str(unit.get("title") or "")
            vocab = unit.get("vocab") or []
            phrases = unit.get("phrases") or []
            level_vocab += len(vocab)
            level_phrases += len(phrases)
            phrase_count += len(phrases)
            estimated_minutes += int(unit.get("estimated_minutes") or 0)
            if len(vocab) < _UNIT_MIN_VOCAB or len(vocab) > _UNIT_MAX_VOCAB:
                unit_size_warnings.append({
                    "unit_id": unit_id,
                    "title": unit_title,
                    "vocab_count": len(vocab),
                    "expected": f"{_UNIT_MIN_VOCAB}-{_UNIT_MAX_VOCAB}",
                })
            for vocab_item in vocab:
                surface = str(vocab_item.get("surface") or "").strip()
                if surface:
                    surface_counts[surface] += 1
                items.append({
                    "surface": surface,
                    "level_code": str(level.get("code") or ""),
                    "unit_id": unit_id,
                    "unit_title": unit_title,
                    "pos": str(vocab_item.get("pos") or ""),
                    "gloss": str(vocab_item.get("gloss") or ""),
                })
            for phrase in phrases:
                target = str(phrase.get("target") or "").strip()
                en = str(phrase.get("en") or "").strip()
                if not target:
                    missing_phrase_target.append({"unit_id": unit_id, "en": en})
                if not en:
                    missing_phrase_translation.append({"unit_id": unit_id, "target": target})
            unit_summaries.append({
                "id": unit_id,
                "title": unit_title,
                "level_code": str(level.get("code") or ""),
                "vocab_count": len(vocab),
                "phrase_count": len(phrases),
                "estimated_minutes": int(unit.get("estimated_minutes") or 0),
            })
        level_summaries.append({
            "code": str(level.get("code") or ""),
            "name": str(level.get("name") or ""),
            "unit_count": len(units),
            "vocab_count": level_vocab,
            "phrase_count": level_phrases,
        })

    duplicates = sorted([surface for surface, n in surface_counts.items() if n > 1])
    single_latin = sorted({
        item["surface"] for item in items
        if item["surface"] and not lang_packs.is_gameworthy_surface(item["surface"])
    })
    findings: list[dict[str, Any]] = []
    if single_latin:
        findings.append(_finding(
            "warn",
            "single_latin_surfaces",
            "Single Latin-letter vocab surfaces appear in the curated path",
            sample=_sample(single_latin),
            count=len(single_latin),
        ))
    if duplicates:
        findings.append(_finding(
            "info",
            "duplicate_path_surfaces",
            "Some vocab surfaces repeat across units or levels",
            sample=_sample(duplicates),
            count=len(duplicates),
        ))
    if missing_phrase_target:
        findings.append(_finding(
            "warn",
            "missing_phrase_target",
            "Some path phrases are missing target-language text",
            sample=_sample(missing_phrase_target),
            count=len(missing_phrase_target),
        ))
    if missing_phrase_translation:
        findings.append(_finding(
            "warn",
            "missing_phrase_translation",
            "Some path phrases are missing English translations",
            sample=_sample(missing_phrase_translation),
            count=len(missing_phrase_translation),
        ))
    if unit_size_warnings:
        findings.append(_finding(
            "info",
            "unit_vocab_size_outside_guidance",
            "Some units fall outside the authoring guidance for vocab count",
            sample=_sample(unit_size_warnings),
            count=len(unit_size_warnings),
        ))

    return {
        "present": True,
        "items": items,
        "stats": {
            "present": True,
            "name": str(path.get("name") or ""),
            "level_system": str(path.get("level_system") or ""),
            "level_count": len(path.get("levels") or []),
            "unit_count": len(unit_summaries),
            "vocab_count": len(items),
            "unique_vocab_count": len(surface_counts),
            "phrase_count": phrase_count,
            "estimated_minutes": estimated_minutes,
            "levels": level_summaries,
            "units": unit_summaries,
            "duplicate_surface_count": len(duplicates),
            "single_latin_surface_count": len(single_latin),
            "missing_phrase_target_count": len(missing_phrase_target),
            "missing_phrase_translation_count": len(missing_phrase_translation),
        },
        "findings": findings,
    }


async def _count_rows(
    conn: aiosqlite.Connection,
    table: str,
    where: str = "",
    params: tuple[Any, ...] = (),
) -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    try:
        cursor = await conn.execute(sql, params)
        row = await cursor.fetchone()
    except aiosqlite.Error:
        return 0
    return int(row[0]) if row else 0


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    try:
        cursor = await conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
    except aiosqlite.Error:
        return set()
    return {str(row[1]) for row in rows}


async def _level_distribution(conn: aiosqlite.Connection) -> dict[str, int]:
    cols = await _table_columns(conn, "vocab")
    column = "level" if "level" in cols else ("jlpt" if "jlpt" in cols else "")
    if not column:
        return {}
    try:
        cursor = await conn.execute(
            f"SELECT {column}, COUNT(*) FROM vocab "
            f"WHERE {column} IS NOT NULL AND {column} != '' GROUP BY {column}"
        )
        rows = await cursor.fetchall()
    except aiosqlite.Error:
        return {}
    return {str(level): int(count) for level, count in rows if level}


async def _pack_audit(pack: Any | None, path_items: list[dict[str, Any]], include_examples: bool) -> dict[str, Any]:
    if pack is None:
        return {
            "pack": {"installed": False},
            "coverage": {"checked": 0, "resolved": 0, "coverage": None, "missing_sample": []},
            "examples": {"checked": 0, "with_example": 0, "coverage": None, "missing_sample": []},
            "discovery_candidates": 0,
            "findings": [_finding("info", "pack_not_installed", "Language pack is not installed")],
        }

    conn = pack.conn
    meta = await lang_packs.pack_meta(conn)
    tokenization = await lang_packs.pack_tokenization(conn)
    sentence_total = await _count_rows(conn, "sentences")
    translated_total = await _count_rows(conn, "sentences", "en_text IS NOT NULL AND en_text != ''")
    translated_easy = await lang_packs.count_sentences(conn, max_difficulty=3, require_translation=True)
    level_distribution = await _level_distribution(conn)
    frequency_entries = await lang_packs.top_frequency(conn, 80)
    discovery_candidates = sum(
        1 for entry in frequency_entries
        if lang_packs.is_gameworthy_surface(entry.get("surface") or "")
    )

    surfaces = [item["surface"] for item in path_items if item.get("surface")]
    resolved = await lang_packs.resolve_surfaces(conn, surfaces)
    resolved_items = [item for item in path_items if item.get("surface") in resolved]
    missing_items = [item for item in path_items if item.get("surface") and item["surface"] not in resolved]

    by_unit: dict[str, dict[str, Any]] = {}
    for item in path_items:
        unit_id = item.get("unit_id") or ""
        row = by_unit.setdefault(unit_id, {
            "unit_id": unit_id,
            "title": item.get("unit_title") or "",
            "level_code": item.get("level_code") or "",
            "checked": 0,
            "resolved": 0,
        })
        if not item.get("surface"):
            continue
        row["checked"] += 1
        if item["surface"] in resolved:
            row["resolved"] += 1
    by_unit_rows = []
    for row in by_unit.values():
        row = dict(row)
        row["coverage"] = _ratio(int(row["resolved"]), int(row["checked"]))
        by_unit_rows.append(row)

    examples = {"checked": 0, "with_example": 0, "coverage": None, "missing_sample": []}
    if include_examples:
        seen_word_ids: set[str] = set()
        missing_examples: list[dict[str, str]] = []
        with_example = 0
        for item in resolved_items:
            word_id = resolved[item["surface"]]
            if word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            entry = await lang_packs.get_entry(conn, word_id)
            if not entry:
                missing_examples.append({
                    "surface": item["surface"],
                    "word_id": word_id,
                    "unit_id": item.get("unit_id") or "",
                })
                continue
            example = await lang_packs.get_example(conn, entry.get("surface") or "")
            if example:
                with_example += 1
            else:
                missing_examples.append({
                    "surface": item["surface"],
                    "word_id": word_id,
                    "unit_id": item.get("unit_id") or "",
                })
        examples = {
            "checked": len(seen_word_ids),
            "with_example": with_example,
            "coverage": _ratio(with_example, len(seen_word_ids)) if seen_word_ids else None,
            "missing_sample": _sample(missing_examples),
        }

    coverage = {
        "checked": len(surfaces),
        "resolved": len(resolved_items),
        "coverage": _ratio(len(resolved_items), len(surfaces)) if surfaces else None,
        "missing_count": len(missing_items),
        "missing_sample": _sample([
            {
                "surface": item["surface"],
                "unit_id": item.get("unit_id") or "",
                "level_code": item.get("level_code") or "",
            }
            for item in missing_items
        ]),
        "by_unit": by_unit_rows,
    }
    findings: list[dict[str, Any]] = []
    if surfaces and coverage["coverage"] is not None and coverage["coverage"] < 0.80:
        findings.append(_finding(
            "warn",
            "low_path_pack_coverage",
            "Installed pack resolves less than 80% of curated path surfaces",
            coverage=coverage["coverage"],
            missing_count=len(missing_items),
        ))
    if include_examples and examples["coverage"] is not None and examples["coverage"] < 0.50:
        findings.append(_finding(
            "warn",
            "low_example_coverage",
            "Less than half of resolved path vocabulary has example sentences",
            coverage=examples["coverage"],
        ))
    if sentence_total == 0:
        findings.append(_finding("warn", "sentence_bank_empty", "Installed pack has no sentences"))
    elif translated_easy == 0:
        findings.append(_finding(
            "warn",
            "translated_easy_sentences_empty",
            "Installed pack has no easy translated sentences for game/readiness flows",
        ))

    return {
        "pack": {
            "installed": True,
            "name": getattr(pack.meta, "name", ""),
            "lang_code": getattr(pack.meta, "lang_code", ""),
            "vocab_count": int(getattr(pack.meta, "vocab_count", 0) or 0),
            "path": str(getattr(pack, "path", "")),
            "tokenization": tokenization,
            "meta_level_system": meta.get("level_system", ""),
            "sentences": {
                "total": sentence_total,
                "translated_total": translated_total,
                "translated_easy": translated_easy,
            },
            "level_distribution": level_distribution,
        },
        "coverage": coverage,
        "examples": examples,
        "discovery_candidates": discovery_candidates,
        "findings": findings,
    }


def _game_material(
    *,
    installed: bool,
    path_vocab: int,
    resolved_vocab: int,
    translated_easy: int,
    path_phrases: int,
    discovery_candidates: int,
) -> dict[str, Any]:
    word_material = resolved_vocab if installed else path_vocab
    translated_material = translated_easy if installed else path_phrases
    basis = "pack" if installed else "path"
    out: dict[str, Any] = {"basis": basis, "games": {}}
    for game_id, req in _GAME_REQUIREMENTS.items():
        if "words_or_discovery" in req:
            available = word_material + (discovery_candidates if installed else 0)
            needed = req["words_or_discovery"]
            ready = available >= needed
            label = f"{available}/{needed} words"
            progress = _ratio(min(available, needed), needed)
        else:
            needed = req.get("words", 0)
            ready = word_material >= needed
            label = f"{word_material}/{needed} words" if needed else "Ready"
            progress = _ratio(min(word_material, needed), needed)
        if ready and req.get("translated_sentences"):
            sent_needed = req["translated_sentences"]
            ready = translated_material >= sent_needed
            if not ready:
                label = f"{translated_material}/{sent_needed} sentences"
                progress = min(progress, _ratio(min(translated_material, sent_needed), sent_needed))
        out["games"][game_id] = {
            "ready": ready,
            "label": "Ready" if ready and game_id != "story_weaver" else ("Explore" if ready else label),
            "progress": progress,
            "requirements": req,
        }
    return out

async def audit_learning_content(
    *,
    pack_manager: Any | None = None,
    lang_codes: list[str] | None = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """Audit supported learning content and installed pack coverage.

    ``pack_manager`` is optional. Without it, the report still audits catalog
    and curated-path distribution and marks pack-specific checks as skipped.
    """
    supported = [spec.lang_code for spec in catalog.available_packs()]
    selected = lang_codes or supported
    languages: dict[str, Any] = {}
    severity_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()

    for lang in selected:
        spec = catalog.get(lang)
        path_info = _collect_path(lang)
        path_stats = path_info["stats"]
        pack = pack_manager.get_language_pack(lang) if pack_manager is not None else None
        pack_info = await _pack_audit(pack, path_info["items"], include_examples)
        resolved_vocab = int(pack_info["coverage"].get("resolved") or 0)
        translated_easy = int(pack_info["pack"].get("sentences", {}).get("translated_easy") or 0)
        findings = [*path_info["findings"], *pack_info["findings"]]
        for finding in findings:
            severity_counts[finding["severity"]] += 1
            code_counts[finding["code"]] += 1

        languages[lang] = {
            "lang": lang,
            "name": spec.name if spec else lang,
            "catalog": spec.to_public_dict() if spec else {"status": "unknown"},
            "path": path_stats,
            "aux_available": learning_paths.available_aux(lang),
            "pack": pack_info["pack"],
            "coverage": pack_info["coverage"],
            "examples": pack_info["examples"],
            "game_material": _game_material(
                installed=bool(pack_info["pack"].get("installed")),
                path_vocab=int(path_stats.get("vocab_count") or 0),
                resolved_vocab=resolved_vocab,
                translated_easy=translated_easy,
                path_phrases=int(path_stats.get("phrase_count") or 0),
                discovery_candidates=int(pack_info.get("discovery_candidates") or 0),
            ),
            "findings": findings,
        }

    totals = {
        "languages": len(selected),
        "supported_languages": supported,
        "path_languages": sum(1 for row in languages.values() if row["path"].get("present")),
        "installed_packs": [lang for lang, row in languages.items() if row["pack"].get("installed")],
        "path_vocab": sum(int(row["path"].get("vocab_count") or 0) for row in languages.values()),
        "path_phrases": sum(int(row["path"].get("phrase_count") or 0) for row in languages.values()),
        "findings_by_severity": dict(sorted(severity_counts.items())),
        "findings_by_code": dict(sorted(code_counts.items())),
    }
    return {"summary": totals, "languages": languages}
