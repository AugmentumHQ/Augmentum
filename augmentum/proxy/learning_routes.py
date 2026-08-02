"""HTTP endpoints for the language-learning system.

All under ``/api/learning``:

* ``GET  /state``                — toggle + profile + installed packs
* ``POST /state``                — save the toggle / profile (per user)
* ``GET  /packs/catalog``        — installable language packs + status
* ``POST /packs/{lang}/install`` — enqueue a pack-build job
* ``GET  /lookup``               — dictionary lookup (longest-prefix or free-text)
* ``POST /vocab/add``            — queue a word for SRS (idempotent)
* ``GET  /srs/due``              — cards whose review is due, with example sentence
* ``POST /srs/grade``            — record a review grade, advance FSRS state

Auth: every handler resolves ``user_id`` from ``request.scope["user"].id``
(the raw-ASGI auth middleware populates it). All persistence calls thread
that ``user_id`` through, per the Augmentum multi-tenancy contract.

See ``docs/superpowers/specs/2026-05-11-language-learning-system.md``.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from augmentum.knowledge.lang_pack_builder import JMDICT_POS_LABELS
from augmentum.learning import fsrs, lang_packs
from augmentum.learning import lang_pack_catalog as catalog
from augmentum.learning import partners as lang_partners
from augmentum.learning import paths as learning_paths
from augmentum.state import vocab_store as vs
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/learning", tags=["learning"])

# Settings keys (per-user via SettingsStore.set_user / get_user).
_K_TOGGLE = "language_learning.toggle_state"        # "on" | "off"
_K_NATIVE = "language_learning.native_lang"         # ISO code, e.g. "en"
_K_TARGETS = "language_learning.target_langs"       # JSON array of ISO codes
_K_LEVEL_PREFIX = "language_learning.level."        # + lang_code -> level
_K_TTS_VOICE = "language_learning.tts_voice"        # Kokoro voice id, or "off"

_DEFAULT_TTS_VOICE = ""   # Off until the user picks during onboarding.
                          # Old setups that stored "jf_alpha" keep working
                          # because the stored value wins over the default.

# Per-language fallback POS label maps, used when an already-installed
# pack predates the `meta.pos_labels` field (the builder writes them
# going forward, but in-place packs from earlier builds don't carry the
# blob). Add a fallback row per lang once that lang's builder ships.
_FALLBACK_POS_LABELS: dict[str, dict[str, str]] = {
    "ja": JMDICT_POS_LABELS,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _require_user(request: Request) -> str:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "Unauthorized")
    return uid


def _settings_store(request: Request):
    return getattr(request.app.state, "settings_store", None)


def _vocab_store(request: Request):
    store = getattr(request.app.state, "vocab_store", None)
    if not store:
        raise HTTPException(503, "Language-learning store not initialized")
    return store


def _pack_manager(request: Request):
    mgr = getattr(request.app.state, "pack_manager", None)
    if not mgr:
        raise HTTPException(503, "Knowledge packs not initialized")
    return mgr


def _sqlite_conn(request: Request):
    """The aiosqlite connection backing ui_characters etc.

    Returns None if state_manager isn't a SQLiteBackend (in-memory tests,
    legacy fixtures). Callers decide whether that's fatal.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return None
    return sm.backend.conn


def _require_lang_pack(request: Request, lang_code: str):
    """Return the active LanguagePack for ``lang_code`` or 404."""
    pack = _pack_manager(request).get_language_pack(lang_code)
    if pack is None:
        raise HTTPException(404, f"No active language pack for '{lang_code}'")
    return pack


# ── /state ───────────────────────────────────────────────────────────


class StateBody(BaseModel):
    toggle: str | None = None                  # "on" | "off"
    native_lang: str | None = None
    target_langs: list[str] | None = None
    levels: dict[str, str] | None = None       # {"ja": "beginner", ...}
    tts_voice: str | None = None               # Kokoro voice id, or "off"


async def _read_state(
    store, uid: str, packs: list[dict], pack_manager=None,
) -> dict[str, Any]:
    toggle = (await store.get_user(uid, _K_TOGGLE)) if store else None
    native = (await store.get_user(uid, _K_NATIVE)) if store else None
    targets_raw = (await store.get_user(uid, _K_TARGETS)) if store else None
    tts_voice = (await store.get_user(uid, _K_TTS_VOICE)) if store else None
    targets: list[str] = []
    if targets_raw:
        try:
            parsed = json.loads(targets_raw)
            if isinstance(parsed, list):
                targets = [str(x) for x in parsed if x]
        except json.JSONDecodeError:
            pass
    levels: dict[str, str] = {}
    if store:
        for pack in packs:
            lc = pack.get("lang_code") or ""
            if not lc:
                continue
            lvl = await store.get_user(uid, _K_LEVEL_PREFIX + lc)
            if lvl:
                levels[lc] = lvl
    pos_labels_by_lang: dict[str, dict[str, str]] = {}
    if pack_manager is not None:
        for pack in packs:
            lc = pack.get("lang_code") or ""
            if not lc:
                continue
            lp = pack_manager.get_language_pack(lc)
            labels: dict[str, str] = {}
            if lp is not None:
                try:
                    labels = await lang_packs.pack_pos_labels(lp.conn)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("pos_labels_read_failed", lang=lc, error=str(exc))
            if not labels:
                labels = _FALLBACK_POS_LABELS.get(lc, {})
            if labels:
                pos_labels_by_lang[lc] = labels
    return {
        "toggle": toggle or "off",
        "native_lang": native or "",
        "target_langs": targets,
        "levels": levels,
        "tts_voice": tts_voice or _DEFAULT_TTS_VOICE,
        "packs": packs,
        "pos_labels_by_lang": pos_labels_by_lang,
    }


@router.get("/state")
async def get_state(request: Request) -> dict[str, Any]:
    """Combined profile + installed-pack snapshot for the UI."""
    uid = _require_user(request)
    mgr = _pack_manager(request)
    packs = mgr.list_language_packs()
    return await _read_state(_settings_store(request), uid, packs, pack_manager=mgr)


@router.post("/state")
async def set_state(request: Request, body: StateBody) -> dict[str, Any]:
    """Patch any subset of {toggle, native_lang, target_langs, levels}."""
    uid = _require_user(request)
    store = _settings_store(request)
    if not store:
        raise HTTPException(503, "Settings store not initialized")

    if body.toggle is not None:
        if body.toggle not in ("on", "off"):
            raise HTTPException(400, "toggle must be 'on' or 'off'")
        await store.set_user(uid, _K_TOGGLE, body.toggle)
    if body.native_lang is not None:
        await store.set_user(uid, _K_NATIVE, body.native_lang.strip())
    if body.target_langs is not None:
        clean = [s.strip() for s in body.target_langs if s and s.strip()]
        await store.set_user(uid, _K_TARGETS, json.dumps(clean))
    if body.levels is not None:
        for lc, lvl in body.levels.items():
            if not lc:
                continue
            await store.set_user(uid, _K_LEVEL_PREFIX + lc.strip(), str(lvl))
    if body.tts_voice is not None:
        v = body.tts_voice.strip() or "off"
        await store.set_user(uid, _K_TTS_VOICE, v)

    mgr = _pack_manager(request)
    packs = mgr.list_language_packs()
    return await _read_state(store, uid, packs, pack_manager=mgr)


# ── /packs (catalog + install) ───────────────────────────────────────


_INSTALL_JOB_TYPE = "lang_pack_install"
_ACTIVE_JOB_STATUSES = ("pending", "running")


@router.get("/packs/catalog")
async def packs_catalog(request: Request) -> dict[str, Any]:
    """The language-pack catalog (available + planned), annotated with
    which packs are already installed and which have an install job in
    flight, so the picker UI can render the right action per row."""
    uid = _require_user(request)
    installed: set[str] = set()
    mgr = getattr(request.app.state, "pack_manager", None)
    if mgr is not None:
        installed = {p["lang_code"] for p in mgr.list_language_packs()}

    in_progress: dict[str, str] = {}   # lang_code -> job_id
    store = getattr(request.app.state, "jobs_store", None)
    if store is not None:
        for status in _ACTIVE_JOB_STATUSES:
            for job in await store.list_for_user(
                user_id=uid, job_type=_INSTALL_JOB_TYPE, status=status,
            ):
                lc = (job.get("payload") or {}).get("lang_code")
                if lc and lc not in in_progress:
                    in_progress[lc] = job["id"]

    packs = []
    for spec in catalog.all_packs():
        d = spec.to_public_dict()
        d["installed"] = spec.lang_code in installed
        d["install_job_id"] = in_progress.get(spec.lang_code)
        d["installable"] = spec.status == "available" and not d["installed"]
        packs.append(d)
    return {"packs": packs}


@router.post("/packs/{lang}/install")
async def install_pack(request: Request, lang: str) -> dict[str, Any]:
    """Enqueue a ``lang_pack_install`` background job for ``lang``.

    Returns ``{job_id, lang, status}``. Polled via ``GET /api/jobs/{id}``.
    Idempotent-ish: if the pack is already installed → 409; if an install
    is already running for this language → returns that job's id.
    """
    uid = _require_user(request)
    spec = catalog.get(lang)
    if spec is None:
        raise HTTPException(404, f"unknown language '{lang}'")
    if spec.status != "available":
        raise HTTPException(400, f"language '{lang}' is not installable yet")

    mgr = getattr(request.app.state, "pack_manager", None)
    if mgr is not None and mgr.has_language_pack(lang):
        raise HTTPException(409, f"language pack '{lang}' is already installed")

    store = getattr(request.app.state, "jobs_store", None)
    if store is None:
        raise HTTPException(503, "Job queue not available")

    # Coalesce: if an install for this lang is already pending/running, reuse it.
    for status in _ACTIVE_JOB_STATUSES:
        for job in await store.list_for_user(
            user_id=uid, job_type=_INSTALL_JOB_TYPE, status=status,
        ):
            if (job.get("payload") or {}).get("lang_code") == lang:
                return {"job_id": job["id"], "lang": lang, "status": job["status"]}

    job_id = await store.create(
        user_id=uid, job_type=_INSTALL_JOB_TYPE,
        payload={"lang_code": lang}, priority=1,
    )
    return {"job_id": job_id, "lang": lang, "status": "pending"}


# ── /lookup ──────────────────────────────────────────────────────────


@router.get("/lookup")
async def lookup(
    request: Request,
    lang: str,
    q: str,
    pos: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Dictionary lookup.

    When ``pos`` is supplied, runs a longest-prefix match starting at
    character offset ``pos`` of ``q`` — the click-to-define path for
    languages without word spaces (the UI passes the click index).
    When ``pos`` is omitted, runs an exact + FTS5 free-text match
    (``/lookup?q=breakfast`` finds 朝ごはん via its English gloss).

    For space-delimited languages the pack also ships an ``inflections``
    table (kaikki Wiktionary form-of metadata). If a direct lookup
    misses we resolve the surface through that table to its lemma and
    look up the lemma instead — so a click on ``hablé`` returns
    ``hablar`` annotated with the inflection tags (preterite, 1st person).
    """
    _require_user(request)
    pack = _require_lang_pack(request, lang)
    if not q:
        return {"entries": []}
    if pos is not None:
        entries = await lang_packs.lookup_at(pack.conn, q, pos)
    else:
        entries = await lang_packs.lookup_text(pack.conn, q, limit=limit)
    # Lemma fallback for inflected forms. Only fires when the direct
    # lookup returned nothing, and only when the pack carries an
    # `inflections` table (V2+ schema — old packs silently skip).
    if not entries and pos is None:
        try:
            cur = await pack.conn.execute(
                "SELECT lemma, tags FROM inflections WHERE form = ? LIMIT 5",
                (q,),
            )
            rows = await cur.fetchall()
            seen: set[str] = set()
            for lemma, tags_json in rows:
                if lemma in seen:
                    continue
                seen.add(lemma)
                sub = await lang_packs.lookup_text(pack.conn, lemma, limit=1)
                if not sub:
                    continue
                try:
                    tag_list = json.loads(tags_json or "[]")
                except (ValueError, TypeError):
                    tag_list = []
                for s in sub:
                    s["inflected_from"] = q
                    s["inflection_tags"] = tag_list
                entries.extend(sub)
                if len(entries) >= limit:
                    break
        except Exception:
            log.debug("inflection_lookup_failed", lang=lang, q=q, exc_info=True)
    return {"entries": entries}


# ── /packs/{lang}/seed ───────────────────────────────────────────────


class SeedBody(BaseModel):
    count: int = 30


@router.post("/packs/{lang}/seed")
async def seed_pack(request: Request, lang: str, body: SeedBody) -> dict[str, Any]:
    """Seed the user's queue with high-value starter vocabulary, due
    immediately so a fresh learner has something to drill on day one.

    Priority:
      1. A hand-curated learning path (``augmentum/learning/paths/{lang}.json``)
         — pedagogically sequenced, content words first. This is the
         right shape for first contact across languages.
      2. Fallback to the pack's top-N corpus-frequency words. (For
         Chinese this returns nothing because CC-CEDICT ships no
         frequency data — that's exactly the hole the curated path
         fills.)

    Path surfaces are resolved against the pack ``vocab`` table — any
    surface the dict doesn't index as a headword (e.g. multi-word
    expressions like "buenos días" or "我 们") is silently dropped.
    The path's ``phrases`` are *not* added to the SRS queue; they're
    exposure material surfaced separately by ``/paths/{lang}``.

    Idempotent. Returns ``{seeded, due, source}`` where ``source`` is
    ``"path"`` or ``"frequency"``.
    """
    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    n = max(1, min(int(body.count), 200))
    store = _vocab_store(request)
    seeded, due, source = await _do_seed(
        store=store, pack_conn=pack.conn, user_id=uid, lang_code=lang, n=n,
    )
    return {"seeded": seeded, "due": due, "source": source}


async def _do_seed(*, store, pack_conn, user_id: str, lang_code: str, n: int):
    """Shared seed logic used by both /seed and /reseed.

    For languages **with** a curated path: walk path units until ``n``
    pack-resolvable word_ids are collected. NEVER falls through to raw
    corpus frequency — frequency-ordered articles/pronouns make a
    pedagogically useless starter queue (the failure mode that produced
    queues of ``y / Y / no / si / Si`` for Spanish learners). If the path
    can't supply ``n`` word_ids even after walking everything, we seed
    what we got and surface that to the caller.

    For languages **without** a curated path: fall back to top-N corpus
    frequency (the legacy behavior — better than nothing).

    Returns ``(seeded, due_count, source)`` where ``source`` is one of
    ``"path"`` | ``"path_partial"`` | ``"frequency"`` | ``"none"``.
    """
    # Ask for a generous candidate pool so we don't undercount when many
    # path surfaces are multi-word expressions (which lookup_surfaces
    # silently drops because they aren't dict headwords).
    path_surfaces = learning_paths.starter_surfaces(lang_code, n=500)
    if path_surfaces:
        resolved = await lang_packs.lookup_surfaces(pack_conn, path_surfaces)
        seen: set[str] = set()
        word_ids: list[str] = []
        for wid in resolved:
            if wid in seen:
                continue
            seen.add(wid)
            word_ids.append(wid)
            if len(word_ids) >= n:
                break
        if word_ids:
            source = "path" if len(word_ids) >= n else "path_partial"
        else:
            # Path exists but resolved zero word_ids — surface this rather
            # than masking the bug with frequency garbage. Caller (UI) can
            # show a real error instead of silently queueing junk.
            log.warning(
                "learning_seed_path_resolved_empty",
                lang=lang_code,
                path_candidates=len(path_surfaces),
            )
            source = "none"
    else:
        entries = await lang_packs.top_frequency(pack_conn, n)
        word_ids = [e["word_id"] for e in entries]
        source = "frequency" if word_ids else "none"
    seeded = await store.seed_words(
        user_id=user_id, lang_code=lang_code, word_ids=word_ids,
    )
    due = await store.count_due(user_id=user_id, lang_code=lang_code)
    return seeded, due, source


class ReseedBody(BaseModel):
    count: int = 30
    confirm: bool = False


@router.post("/packs/{lang}/reseed")
async def reseed_pack(request: Request, lang: str, body: ReseedBody) -> dict[str, Any]:
    """Wipe the user's queue for this language and reseed from the
    curated path. Use when a queue has accumulated low-value entries
    — e.g. the old frequency seeder put 8 different ja kanji that all
    read 'は' into the same starter queue, which is unusable.

    Requires ``confirm: true`` in the body to avoid accidental wipes
    via misclick. Returns ``{cleared, seeded, due, source}``.
    """
    if not body.confirm:
        raise HTTPException(400, "reseed requires confirm: true")
    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    n = max(1, min(int(body.count), 200))
    store = _vocab_store(request)
    cleared = await store.clear_lang(user_id=uid, lang_code=lang)
    seeded, due, source = await _do_seed(
        store=store, pack_conn=pack.conn, user_id=uid, lang_code=lang, n=n,
    )
    return {"cleared": cleared, "seeded": seeded, "due": due, "source": source}


# ── /paths/{lang} ────────────────────────────────────────────────────


@router.get("/paths/{lang}")
async def get_path(request: Request, lang: str) -> dict[str, Any]:
    """Curated learning path for the language — levels → units →
    vocab + phrases + grammar notes. Powers the "Continue Unit N"
    framing on the hub and the in-app curriculum view.

    Returns 404 if no path is shipped for the language yet — the
    frontend falls back to plain SRS review without a path overlay.
    """
    _require_user(request)
    summary = learning_paths.path_summary(lang)
    if summary is None:
        raise HTTPException(404, f"No curated path for '{lang}' yet")
    return summary


@router.get("/paths/{lang}/unit/{unit_id}")
async def get_path_unit(
    request: Request, lang: str, unit_id: str,
) -> dict[str, Any]:
    """One unit's full content — vocab, phrases, grammar note. The
    summary endpoint above is heavy enough for the hub overview;
    callers fetch a specific unit only when the learner opens it."""
    _require_user(request)
    full = learning_paths.load_path(lang)
    if full is None:
        raise HTTPException(404, f"No curated path for '{lang}' yet")
    for level in full.get("levels") or []:
        for unit in level.get("units") or []:
            if unit.get("id") == unit_id:
                return {
                    "lang": lang,
                    "level_code": level.get("code"),
                    "unit": unit,
                }
    raise HTTPException(404, f"Unit '{unit_id}' not found in '{lang}' path")


@router.get("/paths/{lang}/aux/{kind}")
async def get_path_aux(
    request: Request, lang: str, kind: str,
) -> dict[str, Any]:
    """Auxiliary curriculum content: grammar drills, assessment rubrics,
    kanji decomposition (ja), tones (zh), character decomposition (zh).

    ``kind`` ∈ {grammar, assessment, kanji, tones, characters}. Returns
    the raw aux JSON — schema varies per kind. The grammar/assessment
    schemas are common across languages; kanji/tones/characters are
    language-specific. 404 if the lang doesn't ship that aux file
    (e.g. there's no `ja-tones.json` because Japanese isn't tonal).
    """
    _require_user(request)
    data = learning_paths.load_aux(lang, kind)
    if data is None:
        raise HTTPException(404, f"No '{kind}' content for '{lang}'")
    return data


@router.get("/paths/{lang}/manifest")
async def get_path_manifest(request: Request, lang: str) -> dict[str, Any]:
    """One call that tells the UI everything available for this language:
    base path summary + which aux kinds ship. Lets the hub render a
    'Grammar drills · Kanji · Assessment' menu without per-aux probes."""
    _require_user(request)
    summary = learning_paths.path_summary(lang)
    aux = learning_paths.available_aux(lang)
    if summary is None and not aux:
        raise HTTPException(404, f"No curriculum for '{lang}' yet")
    return {
        "lang": lang,
        "path": summary,
        "aux_available": aux,
    }


# ── /breakdown/{lang} ────────────────────────────────────────────────


@router.get("/breakdown/{lang}")
async def breakdown(request: Request, lang: str, q: str) -> dict[str, Any]:
    """Tokenise an arbitrary span of target-language text against the pack
    dictionary — for the "highlight a phrase, see it broken down word by
    word" surface. Returns ``{text, tokens: [...]}`` where matched tokens
    carry the vocab entry (surface/reading/pos/glosses/word_id) and raw
    (unmatched) tokens are just ``{text, matched: false}``."""
    _require_user(request)
    pack = _require_lang_pack(request, lang)
    q = (q or "").strip()[:200]
    if not q:
        return {"text": "", "tokens": []}
    tokens = await lang_packs.tokenize_segment(pack.conn, q)
    return {"text": q, "tokens": tokens}


# ── /read/{lang} ─────────────────────────────────────────────────────


@router.get("/read/{lang}")
async def read_pack(
    request: Request,
    lang: str,
    count: int = 20,
    q: str | None = None,
) -> dict[str, Any]:
    """A batch of target-language sentences for the reading surface —
    short, translated, randomised. ``q`` restricts to sentences containing
    that text (e.g. a word you're studying, to see it in context)."""
    _require_user(request)
    pack = _require_lang_pack(request, lang)
    count = max(1, min(int(count), 100))
    sentences = await lang_packs.read_sentences(
        pack.conn, n=count, contains=(q.strip() if q and q.strip() else None),
    )
    return {"sentences": sentences}


# ── /vocab/add ───────────────────────────────────────────────────────


class AddVocabBody(BaseModel):
    lang: str
    word_id: str
    source_surface: str = "browse"
    source_ref: str = ""


@router.post("/vocab/add")
async def add_vocab(request: Request, body: AddVocabBody) -> dict[str, Any]:
    """Queue a word for SRS. Idempotent — re-adding a queued word returns
    ``{added: False}``."""
    uid = _require_user(request)
    pack = _require_lang_pack(request, body.lang)
    # Validate that the word_id actually exists in the pack so we don't
    # accumulate orphan rows. Cheap (PK lookup).
    entry = await lang_packs.get_entry(pack.conn, body.word_id)
    if entry is None:
        raise HTTPException(404, f"word_id '{body.word_id}' not in pack '{body.lang}'")
    store = _vocab_store(request)
    added = await store.add_word(
        user_id=uid,
        lang_code=body.lang,
        word_id=body.word_id,
        source_surface=body.source_surface or "browse",
        source_ref=body.source_ref or "",
    )
    return {"added": added, "word_id": body.word_id, "lang": body.lang}


# ── /srs/due ─────────────────────────────────────────────────────────


# ── /games/pool ──────────────────────────────────────────────────────

_POOL_MODES = ("mixed", "drill", "consolidate", "explore", "garden")


@router.get("/games/pool")
async def games_pool(
    request: Request,
    lang: str,
    count: int = 30,
    mode: str = "mixed",
    allow_discovery: bool = False,
    focus: list[str] | None = None,
) -> dict[str, Any]:
    """Mixed pool of vocab cards for game modes.

    The ``mode`` parameter tilts the mix toward what the game wants:

    * ``mixed`` (default) — due cards → other known → top-frequency seed.
      One-stop fetch for games that want broad variety.
    * ``drill`` — leech + learning + due first, then due. For speed /
      pressure games (Bubble Pop, Whisper Race) where the goal is to
      hammer weak words.
    * ``consolidate`` — mature + reviewing first. For confidence-building
      surfaces (Word Garden detail, Companion small-talk).
    * ``explore`` — top-frequency words the user does NOT yet have, then
      filled in with known words. For games that grow the user's pool
      as a side effect (Word Forge, Story Weaver).

    Each card is enriched with the dictionary entry + an example sentence
    so a single fetch fuels a whole round.
    """
    import random as _random

    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    store = _vocab_store(request)
    mode = mode if mode in _POOL_MODES else "mixed"
    discovery_enabled = bool(allow_discovery or mode == "explore")
    # Garden visualises the user's whole collection — let the cap stretch
    # so a deep learner sees every plant, not a 100-row sample. Other modes
    # stay capped at 100 to keep the games' wire payload predictable.
    if mode == "garden":
        n = max(4, min(int(count), 1000))
        candidate_n = max(n, min(1000, max(n * 3, n + 20)))
    else:
        n = max(4, min(int(count), 100))
        candidate_n = max(n, min(300, max(n * 3, n + 20)))

    cards: list[dict] = []
    seen: set[str] = set()

    def _add(card: dict) -> None:
        if card.get("word_id") and card["word_id"] not in seen:
            seen.add(card["word_id"])
            cards.append(card)

    known = await store.list_all(user_id=uid, lang_code=lang, limit=max(400, candidate_n))
    due = await store.get_due(user_id=uid, lang_code=lang, limit=candidate_n)

    if mode == "drill":
        # Weakest first: leech > learning > due > everything else.
        ordered = (
            [w for w in known if w.get("mastery_state") == "leech"]
            + [w for w in known if w.get("mastery_state") == "learning"]
            + due
            + [w for w in known if w.get("mastery_state") not in ("leech", "learning")]
        )
        _random.shuffle(ordered[:0])  # no-op; explicit
        for w in ordered:
            _add(w)
            if len(cards) >= candidate_n:
                break
    elif mode == "consolidate":
        ordered = (
            [w for w in known if w.get("mastery_state") == "mature"]
            + [w for w in known if w.get("mastery_state") == "reviewing"]
            + due
            + [w for w in known if w.get("mastery_state") in ("learning", "new")]
        )
        for w in ordered:
            _add(w)
            if len(cards) >= candidate_n:
                break
    elif mode == "explore":
        # Start with high-frequency pack words the user doesn't have yet.
        have = {w["word_id"] for w in known}
        freq = await lang_packs.top_frequency(pack.conn, candidate_n * 2)
        for f in freq:
            if f["word_id"] in have:
                continue
            _add({
                "word_id": f["word_id"],
                "fsrs_difficulty": 5.0, "fsrs_stability": 0.0,
                "fsrs_due_at": "", "fsrs_reps": 0, "fsrs_lapses": 0,
                "mastery_state": "new", "first_seen_at": "",
                "last_reviewed_at": None,
                "_discovery": True,
            })
            if len(cards) >= candidate_n:
                break
        # Pad with known words for distractors.
        if len(cards) < candidate_n:
            for w in known:
                _add(w)
                if len(cards) >= candidate_n:
                    break
    elif mode == "garden":
        # Show every owned word, no due-tilt, no shuffle — the user's
        # collection visualised as a stable space, not a play surface.
        # Backfill below is suppressed so the count reflects only what
        # the user actually owns.
        for w in known:
            _add(w)
            if len(cards) >= candidate_n:
                break
    else:   # mixed
        for c in due:
            _add(c)
        extras = [w for w in known if w["word_id"] not in seen]
        _random.shuffle(extras)
        for w in extras:
            _add(w)
            if len(cards) >= candidate_n:
                break

    if mode != "garden" and discovery_enabled and len(cards) < candidate_n:
        # Discovery backfill is opt-in. Practice games should surface a
        # readiness state instead of grading or hiding ghost words.
        freq = await lang_packs.top_frequency(pack.conn, candidate_n - len(cards))
        for f in freq:
            _add({
                "word_id": f["word_id"],
                "fsrs_difficulty": 5.0, "fsrs_stability": 0.0,
                "fsrs_due_at": "", "fsrs_reps": 0, "fsrs_lapses": 0,
                "mastery_state": "new", "first_seen_at": "",
                "last_reviewed_at": None,
                "_discovery": True,
            })

    # Focus-word bias: the language partner's `suggest_drill` tool can
    # prescribe a focused round on specific word_ids the learner has
    # been struggling with. We reorder the pool so the requested words
    # appear first; non-focus words still follow so distractor games
    # (Echo Chamber) still have material to work with. Focus words not
    # already in the pool are NOT injected — the partner suggested
    # them because they were already in the learner's queue.
    focus_ids = [f for f in (focus or []) if f]
    if focus_ids:
        focus_set = set(focus_ids)
        focused = [c for c in cards if c["word_id"] in focus_set]
        rest = [c for c in cards if c["word_id"] not in focus_set]
        cards = focused + rest

    enriched: list[dict] = []
    for card in cards:
        entry = await lang_packs.get_entry(pack.conn, card["word_id"])
        if entry is None:
            continue
        if not lang_packs.is_gameworthy_surface(entry["surface"]):
            continue
        example = await lang_packs.get_example(pack.conn, entry["surface"])
        enriched.append({
            "word_id": card["word_id"],
            "surface": entry["surface"],
            "reading": entry["reading"],
            "pos": entry["pos"],
            "glosses": entry["glosses"],
            "level": entry.get("level"),
            "example": example,
            "mastery_state": card.get("mastery_state", "new"),
            "in_queue": not bool(card.get("_discovery")),
            "is_discovery": bool(card.get("_discovery")),
        })
        if len(enriched) >= n:
            break
    return {
        "pool": enriched,
        "lang": lang,
        "mode": mode,
        "requested_count": n,
        "candidate_count": len(cards),
        "owned_count": len(known),
        "due_count": len(due),
        "discovery_count": sum(1 for c in enriched if c.get("is_discovery")),
        "allow_discovery": discovery_enabled,
    }


# ── /games/result + /games/best ──────────────────────────────────────


def _progress_ratio(current: int, required: int) -> float:
    if required <= 0:
        return 1.0
    return max(0.0, min(1.0, float(current) / float(required)))


def _game_status(
    *,
    ready: bool,
    label: str,
    progress: float,
    reason: str = "",
    requirements: dict[str, int] | None = None,
    recommended: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "ready": bool(ready),
        "label": label,
        "progress": 1.0 if ready else max(0.0, min(1.0, float(progress))),
        "reason": reason,
        "requirements": requirements or {},
        "recommended": bool(ready if recommended is None else recommended),
        "note": note,
    }


@router.get("/games/readiness")
async def games_readiness(request: Request, lang: str) -> dict[str, Any]:
    """Game-launch snapshot for the hub.

    This is the single readiness spine for game cards: word counts, due
    workload, sentence availability, curriculum metadata, and per-game launch
    gates. Top-level counts stay backward-compatible for older hub code.
    """
    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    store = _vocab_store(request)

    known = await store.list_all(user_id=uid, lang_code=lang, limit=10000)
    known_ids = {w["word_id"] for w in known if w.get("word_id")}
    due_count = await store.count_due(user_id=uid, lang_code=lang)
    counts = {"new": 0, "learning": 0, "reviewing": 0, "mature": 0, "leech": 0}
    for row in known:
        state = row.get("mastery_state") or "new"
        if state in counts:
            counts[state] += 1

    weak_count = counts["learning"] + counts["leech"]
    settled_count = counts["reviewing"] + counts["mature"]
    total = len(known)

    try:
        translated_sentences = await lang_packs.count_sentences(
            pack.conn, max_difficulty=3, require_translation=True,
        )
    except Exception as exc:  # pragma: no cover - defensive for legacy packs
        log.debug("learning_readiness_sentence_count_failed", lang=lang, error=str(exc))
        translated_sentences = 0

    try:
        tokenization = await lang_packs.pack_tokenization(pack.conn)
    except Exception as exc:  # pragma: no cover - defensive for legacy packs
        log.debug("learning_readiness_tokenization_failed", lang=lang, error=str(exc))
        tokenization = "longest_prefix"

    try:
        meta = await lang_packs.pack_meta(pack.conn)
    except Exception as exc:  # pragma: no cover - defensive for legacy packs
        log.debug("learning_readiness_pack_meta_failed", lang=lang, error=str(exc))
        meta = {}

    path_summary = learning_paths.path_summary(lang)
    aux_available = learning_paths.available_aux(lang)

    try:
        top_entries = await lang_packs.top_frequency(pack.conn, 80)
    except Exception as exc:  # pragma: no cover - defensive for legacy packs
        log.debug("learning_readiness_frequency_failed", lang=lang, error=str(exc))
        top_entries = []
    discovery_candidates = [
        e for e in top_entries
        if e.get("word_id") not in known_ids
        and lang_packs.is_gameworthy_surface(e.get("surface") or "")
    ]
    discovery_count = len(discovery_candidates)
    available_for_story = total + discovery_count

    def _word_label(required: int) -> str:
        return f"{total}/{required} words"

    def _sentence_label(required: int) -> str:
        return f"{translated_sentences}/{required} sentences"

    core_label = f"{due_count} due" if due_count else "Ready"
    word_forge_note = "Better after a few settled words" if 0 < total < 8 else ""
    mirror_ready = total >= 6 and translated_sentences >= 4
    mirror_reason = "needs sentence examples" if total >= 6 else "needs queued vocabulary"
    mirror_label = "Ready" if mirror_ready else (
        _sentence_label(4) if total >= 6 else _word_label(6)
    )

    games = {
        "bubble_pop": _game_status(
            ready=total >= 4,
            label=core_label if total >= 4 else _word_label(4),
            progress=_progress_ratio(total, 4),
            reason="" if total >= 4 else "needs queued vocabulary",
            requirements={"words": 4},
        ),
        "word_garden": _game_status(
            ready=total >= 1,
            label="Ready" if total >= 1 else _word_label(1),
            progress=_progress_ratio(total, 1),
            reason="" if total >= 1 else "needs at least one saved word",
            requirements={"words": 1},
        ),
        "echo_chamber": _game_status(
            ready=total >= 4,
            label=core_label if total >= 4 else _word_label(4),
            progress=_progress_ratio(total, 4),
            reason="" if total >= 4 else "needs queued vocabulary",
            requirements={"words": 4},
            note="Examples are limited" if total >= 4 and translated_sentences < 4 else "",
        ),
        "whisper_race": _game_status(
            ready=total >= 4,
            label=core_label if total >= 4 else _word_label(4),
            progress=_progress_ratio(total, 4),
            reason="" if total >= 4 else "needs queued vocabulary",
            requirements={"words": 4},
        ),
        "story_weaver": _game_status(
            ready=available_for_story >= 6,
            label="Explore" if available_for_story >= 6 else f"{available_for_story}/6 words",
            progress=_progress_ratio(available_for_story, 6),
            reason="" if available_for_story >= 6 else "needs vocabulary or frequency candidates",
            requirements={"words_or_discovery": 6},
            recommended=True,
        ),
        "word_forge": _game_status(
            ready=total >= 8,
            label="Ready" if total >= 8 else _word_label(8),
            progress=_progress_ratio(total, 8),
            reason="" if total >= 8 else "needs queued vocabulary",
            requirements={"words": 8, "settled_words_recommended": 4},
            recommended=settled_count >= 4,
            note=word_forge_note,
        ),
        "constellation": _game_status(
            ready=total >= 6,
            label="Ready" if total >= 6 else _word_label(6),
            progress=_progress_ratio(total, 6),
            reason="" if total >= 6 else "needs queued vocabulary",
            requirements={"words": 6},
        ),
        "mirror": _game_status(
            ready=mirror_ready,
            label=mirror_label,
            progress=min(_progress_ratio(total, 6), _progress_ratio(translated_sentences, 4)),
            reason="" if mirror_ready else mirror_reason,
            requirements={"words": 6, "translated_sentences": 4},
        ),
        "vocab_quest": _game_status(
            ready=total >= 6,
            label=core_label if total >= 6 else _word_label(6),
            progress=_progress_ratio(total, 6),
            reason="" if total >= 6 else "needs queued vocabulary",
            requirements={"words": 6},
        ),
    }

    level_system = (
        meta.get("level_system")
        or ((path_summary or {}).get("level_system") if path_summary else "")
        or ""
    )
    return {
        "lang": lang,
        "total": total,
        "due": due_count,
        "counts": counts,
        "weak": weak_count,
        "settled": settled_count,
        "sentences": {"translated_easy": translated_sentences},
        "path": path_summary,
        "aux_available": aux_available,
        "capabilities": {
            "tokenization": tokenization,
            "level_system": level_system,
            "has_level_metadata": bool(level_system),
        },
        "discovery": {"frequency_candidates": discovery_count},
        "games": games,
    }

class GameResultBody(BaseModel):
    game_id: str
    lang: str
    score: int = 0
    words_played: int = 0
    words_correct: int = 0
    duration_sec: int = 0
    metadata: dict | None = None


@router.post("/games/result")
async def post_game_result(request: Request, body: GameResultBody) -> dict[str, Any]:
    """Record a finished game session. Powers hub-card best-scores +
    future adaptive-difficulty + history graphs."""
    uid = _require_user(request)
    store = _vocab_store(request)
    conn = store._conn  # noqa: SLF001 — single-store package, narrow access OK
    await conn.execute(
        """INSERT INTO game_results
               (user_id, game_id, lang_code, score, words_played, words_correct,
                duration_sec, ended_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)""",
        (
            uid, body.game_id.strip()[:64], body.lang.strip()[:16],
            int(body.score), int(body.words_played), int(body.words_correct),
            int(body.duration_sec),
            json.dumps(body.metadata or {}, ensure_ascii=False),
        ),
    )
    await conn.commit()
    return {"recorded": True, "game_id": body.game_id, "score": body.score}


@router.get("/games/best")
async def games_best(request: Request, lang: str) -> dict[str, Any]:
    """Per-game best scores + recent-7-day session count for the user.
    The hub fetches this once to decorate game cards with "best 1240" etc."""
    uid = _require_user(request)
    store = _vocab_store(request)
    conn = store._conn  # noqa: SLF001
    cursor = await conn.execute(
        """SELECT game_id,
                  MAX(score)        AS best,
                  COUNT(*)          AS plays,
                  MAX(ended_at)     AS last_played
             FROM game_results
            WHERE user_id = ? AND lang_code = ?
            GROUP BY game_id""",
        (uid, lang),
    )
    rows = await cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    by_game = {}
    for r in rows:
        d = dict(zip(cols, r, strict=True))
        by_game[d["game_id"]] = {
            "best": int(d["best"] or 0),
            "plays": int(d["plays"] or 0),
            "last_played": d["last_played"],
        }
    return {"by_game": by_game, "lang": lang}


@router.get("/srs/due")
async def srs_due(
    request: Request,
    lang: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Cards whose review is due, enriched with dictionary entry + an
    example sentence per card."""
    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    store = _vocab_store(request)
    rows = await store.get_due(user_id=uid, lang_code=lang, limit=limit)
    out: list[dict[str, Any]] = []
    for row in rows:
        entry = await lang_packs.get_entry(pack.conn, row["word_id"])
        if entry is None:
            continue
        example = await lang_packs.get_example(pack.conn, entry["surface"])
        card = fsrs.CardState(
            difficulty=float(row["fsrs_difficulty"]),
            stability=float(row["fsrs_stability"]),
            reps=int(row["fsrs_reps"]),
            lapses=int(row["fsrs_lapses"]),
        )
        elapsed = vs.elapsed_days_since(row.get("last_reviewed_at") or row.get("first_seen_at"))
        out.append({
            "word_id": row["word_id"],
            "surface": entry["surface"],
            "reading": entry["reading"],
            "pos": entry["pos"],
            "glosses": entry["glosses"],
            "level": entry.get("level"),
            "example": example,
            # Predicted next interval (whole days) per grade — labels the
            # Again/Hard/Good/Easy buttons in the review UI.
            "preview_intervals": fsrs.preview_intervals(card, elapsed_days=elapsed),
            "fsrs": {
                "difficulty": row["fsrs_difficulty"],
                "stability": row["fsrs_stability"],
                "reps": row["fsrs_reps"],
                "lapses": row["fsrs_lapses"],
                "due_at": row["fsrs_due_at"],
                "last_reviewed_at": row["last_reviewed_at"],
            },
            "mastery_state": row["mastery_state"],
        })
    total = await store.count_due(user_id=uid, lang_code=lang)
    return {"due": out, "total": total}


# ── /vocab/leeches + /vocab/progress ─────────────────────────────────

# Approximate word-count→CEFR thresholds. Receptive vocab is generally
# larger than productive; these numbers are calibrated to the receptive
# side because that's what SRS measures (you can recognise the word).
# Per-language tweaks live in _CEFR_BANDS overrides.
_CEFR_BANDS_DEFAULT = (
    (600,   "A1"),
    (1500,  "A2"),
    (2500,  "B1"),
    (4000,  "B2"),
    (8000,  "C1"),
    (16000, "C2"),
)
_CEFR_BANDS: dict[str, tuple] = {
    # CJK scripts ramp into reading comprehension on lower word counts
    # because each "word" often carries more semantic load than a Latin-
    # script word — but conversational fluency requires more. The bands
    # below reflect Defense Language Institute / JLPT-style estimates.
    "ja": (
        (300,   "A1 / N5"),
        (1500,  "A2 / N4"),
        (3750,  "B1 / N3"),
        (6000,  "B2 / N2"),
        (10000, "C1 / N1"),
        (20000, "C2"),
    ),
    "zh": (
        (300,   "A1 / HSK1"),
        (600,   "A2 / HSK2"),
        (1200,  "B1 / HSK3"),
        (2500,  "B2 / HSK4-5"),
        (5000,  "C1 / HSK6"),
        (8000,  "C2"),
    ),
}


def _cefr_for(lang: str, mature_count: int) -> str:
    bands = _CEFR_BANDS.get(lang, _CEFR_BANDS_DEFAULT)
    label = "—"
    for threshold, name in bands:
        if mature_count < threshold:
            return label or name
        label = name
    return label


@router.get("/vocab/leeches")
async def vocab_leeches(
    request: Request,
    lang: str,
    limit: int = 30,
) -> dict[str, Any]:
    """Words the learner keeps getting wrong — FSRS leech bucket plus
    high-lapse cards from the "learning" state. Sorted weakest first
    (most lapses, then oldest first-seen). Enriched with dictionary
    entry + example so the review surface has everything it needs.
    """
    uid = _require_user(request)
    pack = _require_lang_pack(request, lang)
    store = _vocab_store(request)
    n = max(1, min(int(limit), 100))
    known = await store.list_all(user_id=uid, lang_code=lang, limit=1000)
    candidates = [
        w for w in known
        if w.get("mastery_state") == "leech"
        or (w.get("mastery_state") == "learning" and int(w.get("fsrs_lapses") or 0) >= 2)
    ]
    candidates.sort(
        key=lambda w: (
            -int(w.get("fsrs_lapses") or 0),
            w.get("first_seen_at") or "",
        )
    )
    out: list[dict[str, Any]] = []
    for w in candidates[:n]:
        entry = await lang_packs.get_entry(pack.conn, w["word_id"])
        if entry is None:
            continue
        example = await lang_packs.get_example(pack.conn, entry["surface"])
        out.append({
            "word_id": w["word_id"],
            "surface": entry["surface"],
            "reading": entry["reading"],
            "pos": entry["pos"],
            "glosses": entry["glosses"],
            "level": entry.get("level"),
            "example": example,
            "mastery_state": w["mastery_state"],
            "lapses": int(w.get("fsrs_lapses") or 0),
            "reps": int(w.get("fsrs_reps") or 0),
            "last_reviewed_at": w.get("last_reviewed_at"),
        })
    return {"leeches": out, "lang": lang}


@router.get("/vocab/progress")
async def vocab_progress(request: Request, lang: str) -> dict[str, Any]:
    """Aggregate snapshot of the learner's vocabulary state in ``lang``:
    counts by mastery, recent-7-day grade volume, day-streak (consecutive
    days with at least one grade), and a coarse CEFR estimate. Powers the
    progress card on the learning surface.
    """
    uid = _require_user(request)
    _ = _require_lang_pack(request, lang)
    store = _vocab_store(request)
    conn = store._conn  # noqa: SLF001

    known = await store.list_all(user_id=uid, lang_code=lang, limit=10000)
    counts = {"new": 0, "learning": 0, "reviewing": 0, "mature": 0, "leech": 0}
    for w in known:
        m = w.get("mastery_state") or "new"
        if m in counts:
            counts[m] += 1

    # Approximate activity + day streak from `last_reviewed_at` on each
    # vocab_state row. This undercounts (only the latest review per card
    # is recorded, not each individual grade), but it's the right shape
    # for streaks: at least one card touched on day X = day X counts.
    last_7 = 0
    streak = 0
    try:
        cur = await conn.execute(
            """SELECT DATE(last_reviewed_at) AS d, COUNT(*) AS n
                 FROM vocab_state
                WHERE user_id = ? AND lang_code = ?
                  AND last_reviewed_at IS NOT NULL
                  AND last_reviewed_at >= datetime('now', '-60 days')
                GROUP BY DATE(last_reviewed_at)
                ORDER BY d DESC""",
            (uid, lang),
        )
        rows = await cur.fetchall()
        from datetime import date, timedelta
        days_with = {r[0]: int(r[1]) for r in rows if r[0]}
        seven_ago = (date.today() - timedelta(days=7)).isoformat()
        last_7 = sum(n for d, n in days_with.items() if d >= seven_ago)
        today = date.today()
        for i in range(0, 60):
            d = (today - timedelta(days=i)).isoformat()
            if d in days_with:
                streak += 1
            elif i == 0:
                # Today empty is OK — streak counts "consecutive days
                # ending yesterday or today", not "today must be active".
                continue
            else:
                break
    except Exception:
        log.debug("vocab_progress_activity_query_failed", exc_info=True)

    total = sum(counts.values())
    # Mature + reviewing is the "settled" vocabulary the CEFR estimate
    # leans on; learning/leech are in-progress so they don't count fully
    # toward fluency thresholds.
    settled = counts["mature"] + counts["reviewing"]
    return {
        "lang": lang,
        "counts": counts,
        "total": total,
        "settled": settled,
        "last_7_days_reviews": last_7,
        "day_streak": streak,
        "cefr_estimate": _cefr_for(lang, settled),
    }


# ── /srs/grade ───────────────────────────────────────────────────────


class GradeBody(BaseModel):
    lang: str
    word_id: str
    grade: int   # 1..4 (Again / Hard / Good / Easy)


@router.post("/srs/grade")
async def srs_grade(request: Request, body: GradeBody) -> dict[str, Any]:
    """Record a review grade and advance the card's FSRS state.

    Returns the new mastery state, the new due timestamp, and the
    intervals (days) the user would have gotten for *each* of the four
    grades — so the next card the UI shows can label its buttons before
    the user clicks. (For the *just-graded* card this is informational.)
    """
    uid = _require_user(request)
    if body.grade not in fsrs.GRADES:
        raise HTTPException(400, "grade must be 1..4")
    _require_lang_pack(request, body.lang)  # validates the language

    store = _vocab_store(request)
    row = await store.get_word(user_id=uid, lang_code=body.lang, word_id=body.word_id)
    if row is None:
        raise HTTPException(404, "word not in your queue")

    card = fsrs.CardState(
        difficulty=float(row["fsrs_difficulty"]),
        stability=float(row["fsrs_stability"]),
        reps=int(row["fsrs_reps"]),
        lapses=int(row["fsrs_lapses"]),
    )
    elapsed = vs.elapsed_days_since(row.get("last_reviewed_at") or row.get("first_seen_at"))
    result = fsrs.schedule(card, body.grade, elapsed_days=elapsed)
    mastery = fsrs.mastery_for(result.reps, result.stability, result.lapses)
    due_at = vs.future_ts(result.interval_days)

    await store.update_after_grade(
        user_id=uid,
        lang_code=body.lang,
        word_id=body.word_id,
        difficulty=result.difficulty,
        stability=result.stability,
        due_at=due_at,
        reps=result.reps,
        lapses=result.lapses,
        grade=body.grade,
        mastery_state=mastery,
    )

    # Preview of next intervals for *this* card if the user re-graded.
    # The review UI uses this on the *next* card's buttons — but it's
    # cheap, so we always include it for the card just acted on too.
    next_intervals = fsrs.preview_intervals(
        fsrs.CardState(
            difficulty=result.difficulty,
            stability=result.stability,
            reps=result.reps,
            lapses=result.lapses,
        ),
        elapsed_days=0.0,
    )
    return {
        "word_id": body.word_id,
        "lang": body.lang,
        "mastery_state": mastery,
        "fsrs_due_at": due_at,
        "interval_days": result.interval_days,
        "next_intervals": next_intervals,
    }


# ── /partner ─────────────────────────────────────────────────────────


async def _get_or_create_partner(
    conn, *, user_id: str, lang_code: str,
) -> dict[str, Any]:
    """Return the user's partner card for ``lang_code``.

    First access for a (user, lang) pair materialises the bundled seed
    from ``lang_partners`` into ui_characters. Subsequent calls return
    the existing row (which the user may have edited). The deterministic
    id keeps URLs/bookmarks stable across rebuilds and lets the partial
    UNIQUE index in migration 171 protect against races.
    """
    seed = lang_partners.get_seed(lang_code)
    if seed is None:
        raise HTTPException(404, f"No bundled partner for language '{lang_code}'")

    card_id = lang_partners.build_card_id(user_id, lang_code)

    cur = await conn.execute(
        "SELECT id, name, data, avatar, lang_code, is_language_partner, "
        "created_at, updated_at "
        "FROM ui_characters "
        "WHERE user_id = ? AND is_language_partner = 1 AND lang_code = ? "
        "LIMIT 1",
        (user_id, lang_code),
    )
    row = await cur.fetchone()
    if row is not None:
        data = json.loads(row[2] or "{}")
        data, upgraded = lang_partners.upgrade_card_data(seed, data)
        if upgraded:
            await conn.execute(
                "UPDATE ui_characters SET data = ? WHERE id = ? AND user_id = ?",
                (json.dumps(data), row[0], user_id),
            )
            await conn.commit()
        return {
            "id": row[0],
            "name": row[1],
            "data": data,
            "avatar": row[3] or "",
            "lang_code": row[4],
            "is_language_partner": bool(row[5]),
            "createdAt": row[6],
            "updatedAt": row[7],
            "bundled": True,
        }

    # Materialise from seed. INSERT OR IGNORE so a concurrent request
    # that won the race doesn't 500 us — the next SELECT picks up
    # whichever row was committed first.
    data_json = lang_partners.card_data_json(seed)
    from datetime import UTC, datetime
    now = datetime.now(UTC).isoformat()
    try:
        await conn.execute(
            "INSERT OR IGNORE INTO ui_characters "
            "(id, name, data, avatar, created_at, updated_at, user_id, "
            " lang_code, is_language_partner) "
            "VALUES (?, ?, ?, '', ?, ?, ?, ?, 1)",
            (card_id, seed.name, data_json, now, now, user_id, lang_code),
        )
        await conn.commit()
    except Exception as exc:
        log.warning(
            "partner_seed_insert_failed",
            user_id=user_id, lang_code=lang_code, error=str(exc),
        )
        raise HTTPException(500, "Failed to materialise partner card") from exc

    cur = await conn.execute(
        "SELECT id, name, data, avatar, lang_code, is_language_partner, "
        "created_at, updated_at "
        "FROM ui_characters "
        "WHERE user_id = ? AND is_language_partner = 1 AND lang_code = ? "
        "LIMIT 1",
        (user_id, lang_code),
    )
    row = await cur.fetchone()
    if row is None:
        # Shouldn't happen — INSERT OR IGNORE succeeded but SELECT misses.
        raise HTTPException(500, "Partner card vanished after insert")
    return {
        "id": row[0],
        "name": row[1],
        "data": json.loads(row[2] or "{}"),
        "avatar": row[3] or "",
        "lang_code": row[4],
        "is_language_partner": bool(row[5]),
        "createdAt": row[6],
        "updatedAt": row[7],
        "bundled": True,
    }


@router.get("/partner")
async def get_partner(request: Request, lang: str) -> dict[str, Any]:
    """Return the language-learning partner card for ``lang``.

    Creates the card from the bundled seed on first access per user.
    The card lives in ``ui_characters`` and behaves like any other
    character: editable in the character UI, openable in narrative
    chat, voice-enabled. Deleting it via DELETE /api/characters/{id}
    causes the next /partner call to recreate from seed — by design,
    so users can "reset" without losing access to the partner system.
    """
    uid = _require_user(request)
    conn = _sqlite_conn(request)
    if conn is None:
        raise HTTPException(503, "Character store not initialized")
    return await _get_or_create_partner(conn, user_id=uid, lang_code=lang)


@router.get("/partners")
async def list_partners(request: Request) -> dict[str, Any]:
    """All language partners the user has materialised so far.

    Powers the hub picker AND the homepage re-entry card. The card
    surface needs ``dueCount`` per partner so it can render "3 words
    due for review" without a second hop; vocab_store outages degrade
    to 0 (card still shows, just with the quiet copy).
    Does NOT auto-materialise — pickers list installed languages
    (from /state) and let the user pick one to open via /partner.
    """
    uid = _require_user(request)
    conn = _sqlite_conn(request)
    if conn is None:
        return {"partners": []}
    cur = await conn.execute(
        "SELECT id, name, lang_code, avatar, updated_at "
        "FROM ui_characters "
        "WHERE user_id = ? AND is_language_partner = 1 "
        "ORDER BY updated_at DESC",
        (uid,),
    )
    rows = await cur.fetchall()

    # Best-effort due counts. The card is a re-entry pull, not a
    # source of truth — if the vocab store is mid-init or the user
    # has no pack installed for this lang, fall back to 0 rather
    # than 503 the whole picker.
    store = getattr(request.app.state, "vocab_store", None)
    due_counts: dict[str, int] = {}
    if store is not None:
        for r in rows:
            lang = r[2]
            try:
                due_counts[lang] = await store.count_due(
                    user_id=uid, lang_code=lang,
                )
            except Exception as exc:
                log.debug(
                    "partner_due_count_failed",
                    user_id=uid, lang_code=lang, error=str(exc),
                )
                due_counts[lang] = 0

    return {
        "partners": [
            {
                "id": r[0],
                "name": r[1],
                "lang_code": r[2],
                "avatar": r[3] or "",
                "updatedAt": r[4],
                "dueCount": due_counts.get(r[2], 0),
            }
            for r in rows
        ],
        "supported_langs": lang_partners.supported_langs(),
    }
