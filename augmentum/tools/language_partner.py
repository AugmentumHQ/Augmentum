"""Language-partner tools.

These five tools are surfaced to the language-partner character cards
seeded by ``augmentum/learning/partners.py``. They wrap existing
learning-system endpoints / store functions so the partner can:

  - look a word up mid-sentence (``vocab_lookup``)
  - save a word the learner asked about to the SRS queue (``vocab_add``)
  - break a target-language phrase down word-by-word (``vocab_breakdown``)
  - silently check what the learner is struggling with right now
    (``vocab_queue_status``) — informs vocabulary choice without being
    read back to the learner
  - prescribe a short focused drill in one of the language games
    (``suggest_drill``) — UI renders as a chip the learner can launch

Everything routes through ``app.state.vocab_store`` / ``pack_manager``,
so persistence, user-scoping, and pack lifecycle stay consistent with
``learning_routes.py``. Nothing here re-implements those.

Tools take ``user_id`` via ``Tool.extract_user_id`` (the same path the
artifact tools use), keeping the multi-tenant contract intact.
"""

from __future__ import annotations

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default to the first available language pack when the partner forgets
# to specify lang. Set during tool registration in server.py.
_DEFAULT_LANG = "en"


def _no_user() -> ToolResult:
    return ToolResult(
        success=False,
        error="Authentication required (no user_id in tool context)",
    )


def _no_pack(lang: str) -> ToolResult:
    return ToolResult(
        success=False,
        error=f"No language pack installed for '{lang}'",
    )


# ── vocab_lookup ─────────────────────────────────────────────────────


class VocabLookupTool(Tool):
    """Dictionary lookup for a single word in the target language."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "vocab_lookup"

    @property
    def description(self) -> str:
        return (
            "Look up a word in the learner's target-language dictionary. "
            "Returns reading, part-of-speech, and English glosses. "
            "Use when the learner asks 'what does X mean?' or you want "
            "to give an accurate definition."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def cache_ttl(self) -> float:
        return 600.0

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "Word or short phrase to look up",
                },
                "lang": {
                    "type": "string",
                    "description": "ISO 639-1 language code (e.g. 'ja', 'es')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["word", "lang"],
        }

    async def execute(
        self,
        *,
        word: str = "",
        lang: str = "",
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        from augmentum.learning import lang_packs

        word = (word or "").strip()
        if not word:
            return ToolResult(success=False, error="word is required")
        lang = (lang or _DEFAULT_LANG).strip()

        mgr = getattr(self._app_state, "pack_manager", None)
        if mgr is None:
            return ToolResult(success=False, error="Knowledge packs not initialized")
        pack = mgr.get_language_pack(lang)
        if pack is None:
            return _no_pack(lang)

        limit = max(1, min(int(limit), 20))
        entries = await lang_packs.lookup_text(pack.conn, word, limit=limit)
        if not entries:
            return ToolResult(
                success=True,
                output=f"No dictionary entries for '{word}' in {lang}.",
                metadata={"word": word, "lang": lang, "entries": []},
            )

        # Compact human-readable summary for the model to chain on.
        lines = []
        for e in entries[:3]:
            surface = e.get("surface", "")
            reading = e.get("reading", "")
            pos = ", ".join(e.get("pos", []) or [])
            glosses = "; ".join(e.get("glosses", []) or [])
            tag = f" [{pos}]" if pos else ""
            reading_part = f" ({reading})" if reading and reading != surface else ""
            lines.append(f"{surface}{reading_part}{tag}: {glosses}")
        summary = "\n".join(lines)

        return ToolResult(
            success=True,
            output=summary,
            metadata={"word": word, "lang": lang, "entries": entries},
            card={
                "kind": "vocab_lookup",
                "title": word,
                "subtitle": lang.upper(),
                "summary": summary,
                "preview": {"entries": entries[:5]},
            },
        )


# ── vocab_add ────────────────────────────────────────────────────────


class VocabAddTool(Tool):
    """Add a word to the learner's SRS queue."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "vocab_add"

    @property
    def description(self) -> str:
        return (
            "Add a target-language word to the learner's spaced-repetition "
            "queue. Call when the learner says 'I want to remember that' or "
            "you notice them stumbling on a useful word. Idempotent — adding "
            "an already-queued word returns added=false."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE  # writes user-scoped data

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "word_id": {
                    "type": "string",
                    "description": "Pack vocab entry id (from vocab_lookup result)",
                },
                "lang": {
                    "type": "string",
                    "description": "ISO 639-1 language code",
                },
            },
            "required": ["word_id", "lang"],
        }

    async def execute(
        self,
        *,
        word_id: str = "",
        lang: str = "",
        **kwargs,
    ) -> ToolResult:
        from augmentum.learning import lang_packs

        uid = Tool.extract_user_id(kwargs)
        if not uid:
            return _no_user()
        word_id = (word_id or "").strip()
        lang = (lang or _DEFAULT_LANG).strip()
        if not word_id:
            return ToolResult(success=False, error="word_id is required")

        mgr = getattr(self._app_state, "pack_manager", None)
        store = getattr(self._app_state, "vocab_store", None)
        if mgr is None or store is None:
            return ToolResult(success=False, error="Language-learning subsystem not initialized")
        pack = mgr.get_language_pack(lang)
        if pack is None:
            return _no_pack(lang)

        entry = await lang_packs.get_entry(pack.conn, word_id)
        if entry is None:
            return ToolResult(
                success=False,
                error=f"word_id '{word_id}' not in pack '{lang}'",
            )

        added = await store.add_word(
            user_id=uid,
            lang_code=lang,
            word_id=word_id,
            source_surface="partner",
            source_ref="",
        )
        surface = entry.get("surface", word_id)
        msg = f"Saved '{surface}' to your queue." if added else f"'{surface}' was already in your queue."
        return ToolResult(
            success=True,
            output=msg,
            metadata={"added": added, "word_id": word_id, "lang": lang,
                      "surface": surface},
        )


# ── vocab_breakdown ──────────────────────────────────────────────────


class VocabBreakdownTool(Tool):
    """Segment a target-language sentence into glossed tokens."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "vocab_breakdown"

    @property
    def description(self) -> str:
        return (
            "Tokenise a target-language phrase or sentence and return each "
            "word with its dictionary entry. Use when the learner pastes "
            "something they read or heard and asks for help understanding "
            "it word-by-word."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sentence": {
                    "type": "string",
                    "description": "Target-language sentence (≤200 chars)",
                },
                "lang": {
                    "type": "string",
                    "description": "ISO 639-1 language code",
                },
            },
            "required": ["sentence", "lang"],
        }

    async def execute(
        self,
        *,
        sentence: str = "",
        lang: str = "",
        **kwargs,
    ) -> ToolResult:
        from augmentum.learning import lang_packs

        sentence = (sentence or "").strip()[:200]
        lang = (lang or _DEFAULT_LANG).strip()
        if not sentence:
            return ToolResult(success=False, error="sentence is required")

        mgr = getattr(self._app_state, "pack_manager", None)
        if mgr is None:
            return ToolResult(success=False, error="Knowledge packs not initialized")
        pack = mgr.get_language_pack(lang)
        if pack is None:
            return _no_pack(lang)

        tokens = await lang_packs.tokenize_segment(pack.conn, sentence)

        # Compact summary: surface → first gloss, joined.
        parts = []
        for t in tokens:
            if not t.get("matched"):
                continue
            surface = t.get("surface") or t.get("text", "")
            glosses = t.get("glosses") or []
            gloss = glosses[0] if glosses else ""
            parts.append(f"{surface} = {gloss}" if gloss else surface)
        summary = " · ".join(parts) if parts else "No matched tokens."

        return ToolResult(
            success=True,
            output=summary,
            metadata={"sentence": sentence, "lang": lang, "tokens": tokens},
            card={
                "kind": "vocab_breakdown",
                "title": sentence,
                "subtitle": lang.upper(),
                "summary": summary,
                "preview": {"tokens": tokens},
            },
        )


# ── vocab_queue_status ───────────────────────────────────────────────


class VocabQueueStatusTool(Tool):
    """Snapshot of the learner's SRS queue — for partner introspection."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "vocab_queue_status"

    @property
    def description(self) -> str:
        return (
            "Get a snapshot of the learner's current vocabulary state for a "
            "language: counts by mastery (new/learning/reviewing/mature/leech), "
            "due-now count, and a sample of leech words (those they keep "
            "forgetting). Use silently to inform what vocabulary you bring "
            "into conversation. Do NOT recite the status back to the learner."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def cache_ttl(self) -> float:
        return 60.0   # state moves fast during an active session

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "lang": {
                    "type": "string",
                    "description": "ISO 639-1 language code",
                },
                "sample_leeches": {
                    "type": "integer",
                    "description": "How many leech words to include (default 5, max 15)",
                    "default": 5,
                },
            },
            "required": ["lang"],
        }

    async def execute(
        self,
        *,
        lang: str = "",
        sample_leeches: int = 5,
        **kwargs,
    ) -> ToolResult:
        uid = Tool.extract_user_id(kwargs)
        if not uid:
            return _no_user()
        lang = (lang or _DEFAULT_LANG).strip()
        sample_leeches = max(0, min(int(sample_leeches), 15))

        store = getattr(self._app_state, "vocab_store", None)
        if store is None:
            return ToolResult(success=False, error="Vocab store not initialized")

        # Reuse list_all for the counts; cheaper than ad-hoc SQL per call.
        all_rows = await store.list_all(user_id=uid, lang_code=lang, limit=2000)
        counts = {"new": 0, "learning": 0, "reviewing": 0, "mature": 0, "leech": 0}
        leech_words: list[dict] = []
        for r in all_rows:
            state = r.get("mastery_state") or "new"
            if state in counts:
                counts[state] += 1
            if state == "leech" and len(leech_words) < sample_leeches:
                leech_words.append({
                    "word_id": r.get("word_id"),
                    "surface": r.get("surface"),
                    "lapses": r.get("fsrs_lapses"),
                })
        try:
            due_count = await store.count_due(user_id=uid, lang_code=lang)
        except Exception:
            due_count = 0

        summary = (
            f"{lang.upper()} queue: {sum(counts.values())} words "
            f"({counts['mature']} mature, {counts['reviewing']} reviewing, "
            f"{counts['learning']} learning, {counts['leech']} leeches, "
            f"{counts['new']} new). Due now: {due_count}."
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "lang": lang,
                "counts": counts,
                "due_now": due_count,
                "leech_sample": leech_words,
            },
        )


# ── suggest_drill ────────────────────────────────────────────────────


# Game ids the partner is allowed to prescribe. Matches the GAMES table
# in ui/scripts/learning_games/hub.js — keep in sync when adding games.
_DRILLABLE_GAMES: dict[str, str] = {
    "bubble_pop": "Bubble Pop — pop bubbles matching the spoken word",
    "whisper_race": "Whisper Race — pronounce target words against a timer",
    "echo_chamber": "Echo Chamber — hear and pick the right word",
    "mirror": "Mirror — translate sentences in both directions",
    "constellation": "Constellation — connect words into sentences",
    "word_forge": "Word Forge — combine morphemes into new words",
    "vocab_quest": "Vocab Quest — multiple-choice in a narrative scene",
    "story_weaver": "Story Weaver — guide a chapter using new words",
    "word_garden": "Word Garden — review your vocabulary collection",
}


class SuggestDrillTool(Tool):
    """Propose a short focused drill in one of the language games."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "suggest_drill"

    @property
    def description(self) -> str:
        descs = "\n".join(f"  - {k}: {v}" for k, v in _DRILLABLE_GAMES.items())
        return (
            "Propose a short focused exercise in one of the language games. "
            "The UI renders the suggestion as a 'Launch drill?' chip the "
            "learner can tap. Use sparingly — only when you notice a real, "
            "specific weakness worth a 60-second drill.\n\nAvailable games:\n"
            + descs
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT  # produces a UI-actionable artifact

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "enum": list(_DRILLABLE_GAMES.keys()),
                    "description": "Which language game to suggest",
                },
                "lang": {
                    "type": "string",
                    "description": "ISO 639-1 language code",
                },
                "focus_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of word_ids to bias the round toward",
                    "default": [],
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason shown to the learner ('You mixed these up')",
                },
            },
            "required": ["game_id", "lang", "reason"],
        }

    async def execute(
        self,
        *,
        game_id: str = "",
        lang: str = "",
        focus_words: list | None = None,
        reason: str = "",
        **kwargs,
    ) -> ToolResult:
        game_id = (game_id or "").strip()
        if game_id not in _DRILLABLE_GAMES:
            return ToolResult(
                success=False,
                error=f"Unknown game_id '{game_id}'. Choose from: "
                      + ", ".join(_DRILLABLE_GAMES),
            )
        lang = (lang or _DEFAULT_LANG).strip()
        reason = (reason or "").strip()[:200]
        focus_words = [str(w) for w in (focus_words or [])][:20]

        summary = f"Suggested drill: {_DRILLABLE_GAMES[game_id]}"
        if reason:
            summary = f"{summary}. {reason}"

        # The card payload is what the frontend renders as the launch chip.
        # Event name matches the convention used by other actionable cards
        # (artifact:edit, image:open) — the chat surface dispatches on it.
        payload = {
            "game_id": game_id,
            "lang": lang,
            "focus_words": focus_words,
        }
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "game_id": game_id,
                "lang": lang,
                "focus_words": focus_words,
                "reason": reason,
            },
            card={
                "kind": "drill_suggestion",
                "title": f"Practice with {game_id.replace('_', ' ').title()}?",
                "subtitle": lang.upper(),
                "summary": reason or _DRILLABLE_GAMES[game_id],
                "preview": {
                    "game_id": game_id,
                    "focus_words": focus_words,
                },
                "actions": [
                    {
                        "label": "Launch drill",
                        "event": "learning:launch_drill",
                        "payload": payload,
                        "icon": "play",
                    },
                ],
            },
        )


# ── factory ──────────────────────────────────────────────────────────


def all_language_partner_tools(app_state) -> list[Tool]:
    """Instantiate the language-partner tool set bound to ``app_state``.

    Called from ``server.py`` after the vocab_store + pack_manager are
    on app.state. Returns an empty list if neither is wired (the tools
    would just 503 on every call — better to leave the registry slot
    free for that case).
    """
    if (getattr(app_state, "vocab_store", None) is None
            and getattr(app_state, "pack_manager", None) is None):
        return []
    return [
        VocabLookupTool(app_state),
        VocabAddTool(app_state),
        VocabBreakdownTool(app_state),
        VocabQueueStatusTool(app_state),
        SuggestDrillTool(app_state),
    ]
