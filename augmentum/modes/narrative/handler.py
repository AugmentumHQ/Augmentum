"""Narrative mode handler — processes requests through the narrative engine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
)
from augmentum.modes.base import ModeHandler
from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.macro_expander import expand_messages
from augmentum.modes.v_command import extract_v_command
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

    from augmentum.image.queue import GenerationQueue
    from augmentum.modes.narrative.engine import NarrativeResult
    from augmentum.state.manager import StateManager

log = get_logger(__name__)


def _on_refresh_error(task: asyncio.Task) -> None:
    """Log unhandled exceptions from background summary refresh tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        log.warning("narrative_memory_refresh_task_failed", error=str(exc))


class NarrativeHandler(ModeHandler):
    """Processes requests through the narrative engine before sending to backend."""

    # Narrative mode is in-world fiction — a real-world server timestamp is
    # immersion-breaking (the character isn't in 2026 UTC, they're in
    # whatever world the chat established) AND it mutates the system prefix
    # every turn, killing llama-server's KV-cache reuse for the prompt.
    # If a card needs in-world time, that belongs in the card itself or in
    # narrative state, not in a real-clock injection.
    _INJECT_DATETIME = False

    def __init__(
        self,
        backend: ModelBackend,
        engine: NarrativeEngine,
        state_manager: StateManager | None = None,
        session_id: str = "",
        image_queue: GenerationQueue | None = None,
        image_enabled: bool = False,
        app_state: State | None = None,
        user_id: str = "",
    ) -> None:
        self._backend = backend
        self._engine = engine
        self._state_manager = state_manager
        self._session_id = session_id
        # Scopes every persistence read/write to the session owner. Blocks
        # the "send another user's X-Augmentum-Session" spoofing path.
        self._user_id = user_id
        self._state_loaded = False
        self._image_queue = image_queue
        self._image_enabled = image_enabled
        self._app_state = app_state
        self._previous_bg_prompts: list[str] = []
        self._last_model: str = ""  # Model from last request, used as fallback for background tasks
        self._refresh_in_flight = False  # Guard against duplicate refresh triggers
        self._last_archived_history_idx: int = 0  # Pointer into _message_history up to which we've archived
        self._detected_context_length: int = 0  # Auto-detected from backend (cached per model)
        self._detected_for_model: str = ""  # Which model the detection was for
        self._group_speaker_card: dict | None = None  # Loaded on demand for group chat
        self._cached_persona_name: str = ""  # Cached user persona name
        # Pending checkpoint saves we fired off async after the user's
        # response stream closed. Strong-referenced so they're not GC'd
        # mid-flight; each task removes itself via add_done_callback.
        # Concurrency: prepare_stable_checkpoint takes _slot_lock inside
        # the backend, so a fresh chat request that races a pending save
        # serializes naturally on the lock — no extra coordination here.
        self._pending_checkpoints: set[asyncio.Task] = set()

        # Wire branch-tagged persistence into the engine so STATE/LEDGER
        # refreshes + branch creation shadow-write to the new tables
        # (migrations 115-117). While unwired, the engine behaves exactly
        # as before. Skipped when state_manager / user_id absent (test
        # contexts that construct handlers without DB scaffolding).
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            from augmentum.state.narrative_persistence import NarrativePersistence
            backend = getattr(state_manager, "_backend", None) if state_manager else None
            if user_id and isinstance(backend, SQLiteBackend):
                self._engine.attach_persistence(
                    NarrativePersistence(backend.conn), user_id,
                )
        except Exception:
            log.warning("narrative_persistence_attach_failed",
                        session_id=session_id, exc_info=True)

    def _schedule_checkpoint(
        self,
        augmented_request,
        full_response: str,
    ) -> None:
        """Fire prepare_stable_checkpoint in the background.

        The save itself runs under the backend's slot lock so the next chat
        request can't accidentally race it. Moving it off the response path
        means the user perceives turn completion as soon as the last token
        arrives — the 5-10 s prewarm + disk-write happens while they're
        reading the reply, not while they wait on the spinner. If the user
        types fast enough to outrun the save, the next request blocks on
        the slot lock for the remaining few seconds — same total cost,
        better distribution.
        """
        if not hasattr(self._backend, "prepare_stable_checkpoint"):
            return

        async def _run() -> None:
            try:
                await self._backend.prepare_stable_checkpoint(
                    augmented_request, full_response,
                )
            except Exception:
                log.warning("stable_checkpoint_prepare_failed", exc_info=True)

        task = asyncio.create_task(_run())
        self._pending_checkpoints.add(task)
        task.add_done_callback(self._pending_checkpoints.discard)

    async def _refresh_card_extensions(self) -> None:
        """One-shot per session: resolve the card's ``extensions`` dict so
        the world manifest is available BEFORE any card parse happens.

        The engine's ``_character_card`` is parsed from the system prompt
        text on the first turn (no extensions survive that), and some
        sessions never get a ``character_cards`` row at all — so the
        reliable source is the UI session record's ``characterId``
        pointing into ``ui_characters`` (live-verified 2026-07-15: the
        cyraeth session had characterId but no card row and an empty
        ``character_card_name``). Name lookup stays as the fallback for
        sessions predating characterId. Result (possibly ``{}``) lands in
        ``self._world_extensions``; ``_world_manifest`` consults it when
        the parsed card carries no extensions of its own."""
        if getattr(self, "_world_extensions", None) is not None:
            return
        ext: dict = {}
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            backend = getattr(self._state_manager, "_backend", None) \
                if self._state_manager else None
            if isinstance(backend, SQLiteBackend) and self._user_id:
                import json as _json
                cursor = await backend.conn.execute(
                    "SELECT data FROM ui_sessions WHERE id = ? AND user_id = ?",
                    (self._session_id, self._user_id),
                )
                row = await cursor.fetchone()
                char_id = ""
                if row:
                    try:
                        char_id = (_json.loads(dict(row)["data"] or "{}")
                                   .get("characterId") or "")
                    except (ValueError, TypeError):
                        char_id = ""
                if char_id:
                    cursor = await backend.conn.execute(
                        "SELECT data FROM ui_characters "
                        "WHERE id = ? AND user_id = ?",
                        (char_id, self._user_id),
                    )
                    row = await cursor.fetchone()
                    if row:
                        try:
                            card_dict = _json.loads(dict(row)["data"] or "{}")
                            found = card_dict.get("extensions") or (
                                card_dict.get("data") or {}
                            ).get("extensions")
                            if isinstance(found, dict):
                                ext = found
                        except (ValueError, TypeError):
                            pass
        except Exception:
            log.warning("world_extensions_resolve_failed",
                        session_id=self._session_id, exc_info=True)
        if not ext:
            # Fallback: name-based lookup (older sessions, group members)
            card = getattr(self._engine, "_character_card", None)
            name = (getattr(card, "name", "") or
                    self._engine.state.character_card_name)
            if name:
                card_dict = await self._load_character_card_by_name(name)
                found = (card_dict or {}).get("extensions")
                if isinstance(found, dict):
                    ext = found
        self._world_extensions = ext
        self._world_manifest_cache = None

    def _world_manifest(self):
        """Parse the card-declared world-system manifest (cached per card).

        Returns None for cards without ``extensions.world_system``, when
        the feature toggle is off, or when the session/user is missing —
        None means the whole feature is invisible (spec: 2026-07-15
        world-system-manifest design). Group sessions bind the PRIMARY
        card's manifest for v1 (the world belongs to the session).
        """
        from augmentum.config import settings as _cfg
        if not (_cfg.narrative_world_systems_enabled
                and self._session_id and self._user_id):
            return None
        card = getattr(self._engine, "_character_card", None)
        raw = getattr(card, "raw_data", None) if card else None
        cache_key = id(card)
        cached = getattr(self, "_world_manifest_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        from augmentum.modes.narrative.world_system import parse_manifest
        manifest = parse_manifest(raw)
        if manifest is None:
            # Session-resolved extensions (ui_sessions.characterId path —
            # covers text-parsed cards and sessions with no card row).
            ext = getattr(self, "_world_extensions", None)
            if ext:
                manifest = parse_manifest({"extensions": ext})
        self._world_manifest_cache = (cache_key, manifest)
        if manifest is not None:
            log.info(
                "world_manifest_active", session_id=self._session_id,
                world=manifest.name, modules=manifest.modules,
            )
        return manifest

    def _world_store(self, manifest):
        """WorldStore over the persisted state dict (mutations stick)."""
        from augmentum.modes.narrative.world_system import WorldStore
        if not isinstance(self._engine.state.world_state, dict):
            self._engine.state.world_state = {}
        return WorldStore(manifest, self._engine.state.world_state)

    def _mem_setting(self, key: str):
        """Resolve a narrative memory setting: session override -> global fallback."""
        from augmentum.modes.narrative.memory_settings import resolve_memory_setting
        session_settings = getattr(self._engine.state, "memory_settings", None)
        return resolve_memory_setting(session_settings, key)

    async def _user_model_setting(self, key: str) -> str:
        """Resolve a per-user narrative model preference.

        The user's own override (``user_settings``, migration 308) wins;
        falls back to the install-wide value. Per-user because one user's
        model choice (local vs API vs a fabric peer) must not silently
        change another user's narrative background calls.
        """
        from augmentum.config import settings as cfg
        store = (
            getattr(self._app_state, "settings_store", None)
            if self._app_state else None
        )
        if store is not None and self._user_id:
            try:
                val = await store.get_user(self._user_id, key)
                if val is not None:
                    return val
            except Exception:
                log.warning(
                    "narrative_user_setting_failed", key=key, exc_info=True,
                )
        return str(getattr(cfg, key, "") or "")

    async def _ensure_state_loaded(self) -> None:
        """Lazy-load persisted state on first request."""
        if self._state_loaded or not self._state_manager or not self._session_id:
            return
        self._state_loaded = True

        # Only load if the engine hasn't processed anything yet
        if self._engine.state.message_count > 0:
            return

        try:
            saved = await self._state_manager.load_narrative_state(
                self._session_id, user_id=self._user_id,
            )
            if saved:
                self._engine.load_state(saved)
                # Loaded history is already archived in SQLite — advance the
                # pointer so continuous archiving only covers NEW exchanges.
                self._last_archived_history_idx = len(self._engine._message_history)
                log.info(
                    "narrative_state_restored",
                    session_id=self._session_id,
                    message_count=saved.message_count,
                    entities=len(saved.entities),
                    memory_summary_len=len(saved.memory_summary),
                )
        except Exception as exc:
            log.warning("narrative_state_load_failed", session_id=self._session_id, error=str(exc))

        # Rehydrate the character card from ui_characters if the engine
        # knows the name but hasn't seen a system prompt yet. Without this,
        # side actions that run before any chat turn (scene image, inspector,
        # portrait) see character=(no card) and fall back to persona-only.
        await self._ensure_character_card_loaded()

        # Load group context if active
        await self._load_group_context()

    async def _ensure_character_card_loaded(self) -> None:
        """Restore ``engine._character_card`` from the ui_characters table
        when the session knows its character name but the engine hasn't
        processed a chat turn yet (and therefore hasn't parsed a card from
        the system prompt).

        Safe to call repeatedly: no-op if a card is already loaded, or if
        the session has no character name, or if the DB lookup misses.
        """
        engine = self._engine
        if engine._character_card is not None:
            return
        name = (engine.state.character_card_name or "").strip()
        if not name or not self._state_manager:
            return
        # Group sessions don't have a single "main" card — each speaker's
        # card is loaded per-turn by _load_group_context / refresh_member_cards.
        if engine.state.group_id:
            return

        card_dict = await self._load_character_card_by_name(name)
        if not card_dict:
            return

        # Build CharacterCard directly from the UI's camelCase dict schema.
        # Avoids round-tripping through the string-based parser and preserves
        # fields the parsers don't extract (image_model, image_style).
        from augmentum.modes.narrative.card_parser import CharacterCard
        from augmentum.modes.narrative.memory import detect_card_type
        card = CharacterCard(
            name=card_dict.get("name", name),
            personality=card_dict.get("personality", ""),
            appearance=card_dict.get("appearance", ""),
            visual_traits=(
                card_dict.get("visualTraits", "")
                or card_dict.get("visual_traits", "")
            ),
            description=card_dict.get("description", ""),
            greeting=card_dict.get("greeting", ""),
            scenario=card_dict.get("scenario", ""),
            example_dialogue=(
                card_dict.get("examples", "")
                or card_dict.get("example_dialogue", "")
            ),
            system_prompt=(
                card_dict.get("systemPrompt", "")
                or card_dict.get("system_prompt", "")
            ),
            creator_notes=(
                card_dict.get("creatorNotes", "")
                or card_dict.get("creator_notes", "")
            ),
            image_model=(
                card_dict.get("imageModel", "")
                or card_dict.get("image_model", "")
            ),
            image_style=(
                card_dict.get("imageStyle", "")
                or card_dict.get("image_style", "")
            ),
            source_format="ui_characters",
            raw_data=card_dict,
        )
        engine._character_card = card
        # Card type drives summary-focus heuristics; keep state in sync.
        try:
            engine._state.card_type = detect_card_type(card).value
        except Exception as exc:
            log.warning("card_type_detect_failed", name=name, error=str(exc))
        log.info(
            "character_card_rehydrated_from_db",
            name=name,
            has_visual_traits=bool(card.visual_traits),
            has_system_prompt=bool(card.system_prompt),
            source="ui_characters",
        )

    # ------------------------------------------------------------------
    # Group chat support
    # ------------------------------------------------------------------

    _group_turn_manager: object | None = None  # GroupTurnManager
    _active_group: object | None = None  # CharacterGroup
    _group_member_cards: dict[str, dict] | None = None  # name → card dict for all members

    async def _load_group_context(self) -> None:
        """Load group + turn manager if a group is active for this session.

        This runs once per engine lifecycle (or whenever group_id changes).
        Member cards are also loaded here for the initial warm cache, but
        ``_refresh_group_member_cards`` re-pulls them every request so card
        edits go live without a restart.
        """
        group_id = self._engine.state.group_id
        if not group_id or not self._state_manager:
            return
        try:
            from augmentum.modes.narrative.group_manager import GroupStore, GroupTurnManager
            from augmentum.state.backends.sqlite import SQLiteBackend
            backend = getattr(self._state_manager, "_backend", None)
            if not isinstance(backend, SQLiteBackend):
                return
            store = GroupStore(backend.conn)
            # user-scoped lookup: a spoofed X-Augmentum-Group-Id belonging to
            # another user returns None instead of loading their group.
            group = await store.get_group(group_id, user_id=self._user_id)
            if not group or not group.member_names:
                return
            tm = GroupTurnManager(group)
            tm._current_index = self._engine.state.group_speaker_index
            self._group_turn_manager = tm
            self._active_group = group

            await self._refresh_group_member_cards()
            log.info("group_context_loaded", group=group.name, speaker=tm.current_speaker,
                     cards_loaded=len(self._group_member_cards or {}))
        except Exception:
            log.warning("group_context_load_failed", exc_info=True)

    async def _resolve_group_speaker(self, request: InternalChatRequest) -> None:
        """Resolve who speaks this turn. Priority:

        1. Manual override (``request.speaker_override`` from X-Augmentum-Speaker).
           Wins over everything. Allowed even on muted members — explicit
           user intent trumps the mute flag; we log a warning so the
           inconsistency is visible.
        2. ``generation_mode == "llm_decide"`` — ask the backend LLM to pick
           a speaker from unmuted members. Falls back to rotation on error,
           timeout, or unparseable response.
        3. Rotation (round_robin / random) — ``current_speaker`` already
           reflects the index set by the previous turn's ``advance()``,
           which skips muted members. Nothing more to do here.
        """
        if not self._group_turn_manager or not self._active_group:
            return

        override = (getattr(request, "speaker_override", "") or "").strip()
        if override:
            if self._group_turn_manager.set_speaker(override):
                log.info("group_speaker_manual_override", speaker=override)
                if self._active_group.is_muted(override):
                    log.warning(
                        "group_speaker_manual_override_muted",
                        speaker=override,
                        note="user explicitly pinned a muted member",
                    )
            else:
                log.warning(
                    "group_speaker_override_unknown",
                    speaker=override,
                    members=self._active_group.member_names,
                )
            return

        if self._active_group.generation_mode == "llm_decide":
            chosen = await self._llm_pick_speaker(request)
            if chosen:
                self._group_turn_manager.set_speaker(chosen)
                log.info("group_speaker_llm_chose", speaker=chosen)
            else:
                log.info("group_speaker_llm_fallback_to_rotation")

    async def _llm_pick_speaker(self, request: InternalChatRequest) -> str | None:
        """Ask the chat backend to pick the next unmuted speaker.

        Returns a valid member name on success, ``None`` when the response
        is empty, unparseable, or times out. Callers fall back to rotation.
        """
        if not self._active_group:
            return None
        eligible = self._active_group.unmuted_members()
        if len(eligible) <= 1:
            return eligible[0] if eligible else None

        # Compact dialogue window — last 6 non-system messages, each capped at 300 chars
        recent = [m for m in request.messages if m.role in ("user", "assistant")][-6:]
        dialogue_lines = []
        for m in recent:
            who = "User" if m.role == "user" else "Assistant"
            text = (m.content or "").strip()
            if len(text) > 300:
                text = text[:297] + "..."
            dialogue_lines.append(f"{who}: {text}")

        system_prompt = (
            "You are a group-chat director. Given the recent conversation and the "
            "list of available speakers, reply with EXACTLY the name of the "
            "character who should speak next. Reply with only the name — no "
            "punctuation, no explanation."
        )
        user_content = (
            f"Available speakers: {', '.join(eligible)}\n\n"
            "Recent conversation:\n" + "\n".join(dialogue_lines) + "\n\n"
            "Next speaker:"
        )

        director_req = InternalChatRequest(
            model=request.model or self._last_model or "",
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ],
            max_tokens=16,
            temperature=0.3,
            think=False,
        )
        try:
            resp = await asyncio.wait_for(self._backend.chat(director_req), timeout=10.0)
        except TimeoutError:
            log.warning("group_speaker_llm_timeout")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("group_speaker_llm_error", error=str(exc))
            return None

        if not resp or not getattr(resp, "message", None) or not resp.message.content:
            return None
        # Tolerant parse: strip trailing punctuation and take the first line
        raw = resp.message.content.strip().strip(".,!?:;\"'*`").split("\n")[0].strip()

        lower = raw.lower()
        # 1) exact match
        for name in eligible:
            if name.lower() == lower:
                return name
        # 2) prefix tolerance ("Alice says:" → Alice, "Alice." → Alice).
        # Sort by descending length so prefix-sharing names disambiguate
        # correctly — e.g. for [Anna, Annabelle], raw="Annabelle says" must
        # check "annabelle " before "anna ".
        for name in sorted(eligible, key=lambda n: -len(n)):
            lo = name.lower()
            if lower.startswith(lo + " ") or lower.startswith(lo + "."):
                return name
        # 3) contains tolerance (final fallback — last resort). Longest-first
        # so "I think Annabelle should go" doesn't match Anna just because
        # member_names lists Anna ahead of Annabelle.
        for name in sorted(eligible, key=lambda n: -len(n)):
            if name.lower() in lower:
                return name
        log.warning("group_speaker_llm_unparseable", raw=raw[:80], eligible=eligible)
        return None

    async def _refresh_group_member_cards(self) -> None:
        """Re-pull every group member's card from the DB into the cache.

        Called on every request (cheap — a handful of SELECTs) so mid-chat
        card edits flow through without a server restart. Mirrors the
        ``_sync_ui_lorebook`` pattern.
        """
        group = self._active_group
        if not group or not group.member_names:
            return
        fresh: dict[str, dict] = {}
        for name in group.member_names:
            card = await self._load_character_card_by_name(name)
            if card:
                fresh[name] = card
        self._group_member_cards = fresh

    async def _load_character_card_by_name(self, name: str) -> dict | None:
        """Load a character card dict from the database by name.

        Always filtered by ``self._user_id``. Without scoping, a user could
        read another user's character data by sending that user's card name
        in a group (cross-tenant leak). Bails out with a warning if uid is
        missing rather than degrading to an unscoped lookup — auth
        middleware should never let an unauth request reach this code.
        """
        if not self._state_manager:
            log.warning("group_card_load_skip", reason="no state_manager", name=name)
            return None
        if not self._user_id:
            log.warning("group_card_load_skip", reason="no user_id", name=name)
            return None
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            backend = getattr(self._state_manager, "_backend", None)
            if not isinstance(backend, SQLiteBackend):
                log.warning("group_card_load_skip", reason="not SQLiteBackend",
                            name=name, backend_type=type(backend).__name__)
                return None
            import json as _json
            cursor = await backend.conn.execute(
                "SELECT data FROM ui_characters "
                "WHERE name = ? COLLATE NOCASE AND user_id = ? LIMIT 1",
                (name, self._user_id),
            )
            row = await cursor.fetchone()
            if row:
                card = _json.loads(dict(row)["data"])
                log.info("group_card_loaded", name=name,
                         desc_len=len(card.get("description", "")),
                         has_personality=bool(card.get("personality", "").strip()))
                return card
            log.warning("group_card_not_in_db", name=name, user_id=self._user_id)
        except Exception:
            log.warning("group_card_load_failed", name=name, exc_info=True)
        return None

    @staticmethod
    def _expand_card_macros(text: str, char_name: str, user_name: str = "User") -> str:
        """Expand {{char}}, {{user}}, and other common macros in card text."""
        if not text or "{{" not in text:
            return text
        import re
        text = re.sub(r"\{\{char\}\}", char_name, text, flags=re.IGNORECASE)
        text = re.sub(r"\{\{user\}\}", user_name, text, flags=re.IGNORECASE)
        text = re.sub(r"\{\{persona\}\}", "", text, flags=re.IGNORECASE)
        return text

    def _build_group_system_prompt(
        self, speaker_card: dict, speaker_name: str, other_names: list[str],
    ) -> str:
        """Build a system prompt for the current group chat speaker.

        Includes the speaker's full card (with macros expanded) and compact
        mentions of other group members, plus an instruction to stay in character.
        """
        parts = []

        # Resolve user name from cached persona (set by _cache_persona_name before handle/handle_stream)
        user_name = getattr(self, "_cached_persona_name", "") or "User"

        expand = lambda t: self._expand_card_macros(t, speaker_name, user_name)

        # Speaker's system prompt (if they have one)
        sys_prompt = speaker_card.get("systemPrompt", "")
        if sys_prompt:
            parts.append(expand(sys_prompt))

        # Speaker's character definition
        desc = speaker_card.get("description", "")
        personality = speaker_card.get("personality", "")
        scenario = speaker_card.get("scenario", "")

        if desc:
            parts.append(f"[Character: {speaker_name}]\n{expand(desc)}")
        if personality and personality.strip() not in ("", "<p></p>"):
            parts.append(f"[Personality]\n{expand(personality)}")
        if scenario:
            parts.append(f"[Scenario]\n{expand(scenario)}")

        # Example dialogues (helps the model match the character's voice)
        examples = speaker_card.get("examples", "") or speaker_card.get("exampleDialogue", "")
        if examples and examples.strip() not in ("", "<p></p>"):
            expanded_examples = expand(examples.strip())
            # Trim to ~500 chars to stay within budget for group prompts
            if len(expanded_examples) > 500:
                expanded_examples = expanded_examples[:497] + "..."
            parts.append(f"[Example Dialogue]\n{expanded_examples}")

        # Creator notes (author guidance for the model)
        creator_notes = speaker_card.get("creatorNotes", "") or speaker_card.get("creator_notes", "")
        if creator_notes and creator_notes.strip() not in ("", "<p></p>"):
            parts.append(f"[Author's Note]\n{expand(creator_notes.strip())}")

        # Compact summaries of other characters present
        if other_names:
            # Get custom summaries from the group definition (user-edited or AI-generated)
            custom_summaries = self._active_group.member_summaries if self._active_group else {}

            other_lines = []
            for name in other_names:
                # Prefer custom summary over auto-generated
                if custom_summaries.get(name):
                    brief = self._expand_card_macros(custom_summaries[name], name, user_name)
                    other_lines.append(f"[{name}]: {brief}")
                    continue

                # Fallback: auto-generate from card with smart truncation
                card = (self._group_member_cards or {}).get(name)
                if card:
                    brief = (card.get("personality") or card.get("description") or "").strip()
                    if len(brief) > 150:
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', brief)
                        if sentences and len(sentences[0]) <= 150:
                            trimmed = sentences[0]
                            for s in sentences[1:]:
                                if len(trimmed) + 1 + len(s) > 150:
                                    break
                                trimmed += " " + s
                            brief = trimmed
                        else:
                            brief = brief[:150].rsplit(" ", 1)[0] + "..."
                    if brief:
                        brief = self._expand_card_macros(brief, name, user_name)
                        other_lines.append(f"[{name}]: {brief}")
                    else:
                        other_lines.append(f"[{name}]")
                else:
                    other_lines.append(f"[{name}]")
            parts.append("[Other characters present]\n" + "\n".join(other_lines))

        # Inject character_dynamics from STATE if available
        if self._mem_setting("memory_enabled") and self._mem_setting("memory_state_enabled"):
            state_fields = self._engine._state_snapshot.fields if self._engine._state_snapshot else {}
            dynamics = state_fields.get("character_dynamics") or state_fields.get("key_relationships") or ""
            if dynamics:
                parts.append(f"[Current Relationships]\n{dynamics}")

        # Strict instruction to stay in character
        parts.append(
            f"You are {speaker_name}. Write ONLY as {speaker_name}. "
            f"Do not write dialogue or actions for {', '.join(other_names) if other_names else 'other characters'}. "
            f"Do not prefix your response with your name."
        )

        return "\n\n".join(parts)

    def _apply_group_to_request(self, request: InternalChatRequest) -> str | None:
        """If a group is active, swap the system prompt to the current speaker's
        card and add stop strings. Returns the speaker name or None."""
        if not self._group_turn_manager or not self._active_group:
            log.info("group_swap_skip", reason="no turn manager or group")
            return None

        speaker = self._group_turn_manager.current_speaker
        if not speaker:
            log.info("group_swap_skip", reason="no current speaker")
            return None

        # The card should have been loaded already — check cache
        if not hasattr(self, "_group_speaker_card") or not self._group_speaker_card:
            log.warning("group_swap_skip", reason="no card loaded", speaker=speaker)
            return None

        other_names = [n for n in self._active_group.member_names if n != speaker]
        group_prompt = self._build_group_system_prompt(
            self._group_speaker_card, speaker, other_names,
        )

        # Swap the system prompt in the request
        swapped = False
        for msg in request.messages:
            if msg.role == "system":
                msg.content = group_prompt
                swapped = True
                break
        if not swapped:
            # No system message exists — prepend one
            request.messages.insert(0, Message(role="system", content=group_prompt))

        log.info("group_prompt_swapped",
                 speaker=speaker,
                 prompt_len=len(group_prompt),
                 stop_strings=[f"\\n{n}:" for n in other_names])
        log.debug("group_prompt_preview",
                  speaker=speaker,
                  prompt_preview=group_prompt[:150])

        # Add stop strings for other character names
        stop_strings = [f"\n{name}:" for name in other_names]
        if request.stop:
            request.stop = list(request.stop) + stop_strings
        else:
            request.stop = stop_strings

        return speaker

    def _postprocess_group_response(self, text: str, speaker: str) -> str:
        """Clean up a group chat response — strip other-character dialogue."""
        if not text or not self._active_group:
            return text

        import re
        other_names = [n for n in self._active_group.member_names if n != speaker]

        # Strip lines where another character speaks.
        # Match only at the start of a line (after optional whitespace) with the
        # name followed immediately by a colon — this prevents false positives
        # when a character's name appears mid-sentence in dialogue.
        other_patterns = []
        for name in other_names:
            escaped = re.escape(name)
            # Match "Name:" or "**Name**:" at line start
            other_patterns.append(rf"^\s*(?:\*\*{escaped}\*\*|{escaped})\s*:")
        other_re = re.compile("|".join(other_patterns)) if other_patterns else None

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if other_re and other_re.match(line):
                continue
            cleaned.append(line)

        result = "\n".join(cleaned).rstrip()

        # Remove the speaker's own name prefix if present (we add it in the UI)
        if result.startswith(f"{speaker}: "):
            result = result[len(f"{speaker}: "):]
        elif result.startswith(f"**{speaker}**: "):
            result = result[len(f"**{speaker}**: "):]
        elif result.startswith(f"{speaker}:"):
            result = result[len(f"{speaker}:"):]

        return result

    def _advance_group_turn(self) -> None:
        """Advance the group turn and persist the new speaker index."""
        if not self._group_turn_manager:
            return
        old_speaker = self._group_turn_manager.current_speaker
        new_speaker = self._group_turn_manager.advance()
        self._engine.state.group_speaker_index = self._group_turn_manager._current_index
        log.info(
            "group_turn_advanced",
            old_speaker=old_speaker,
            new_speaker=new_speaker,
            index=self._group_turn_manager._current_index,
            mode=self._active_group.generation_mode if self._active_group else "",
        )

    async def _detect_context_length(self, model: str) -> None:
        """Auto-detect the model's context window size.

        Used as a fallback for non-llama backends (Claude/Gemini/ollama).
        For the local llama-server path, ``_effective_context_limit`` reads
        live from ``LlamaServerManager.current_ctx_size`` directly, which
        is always the source of truth for the loaded model.

        Re-runs on every request rather than caching by model name —
        ctx can change without the model name changing (e.g. user adjusts
        n_ctx in Advanced settings), and a stale cached value silently
        sends prompts that exceed the model's actual capacity.
        """
        if not model:
            return
        try:
            ctx = await self._backend.get_context_length(model)
            if ctx > 0:
                if ctx != self._detected_context_length:
                    log.info("context_length_detected", model=model, context_length=ctx)
                self._detected_context_length = ctx
                self._detected_for_model = model
            else:
                self._detected_context_length = 0
        except Exception:
            self._detected_context_length = 0

    def _effective_context_limit(self) -> int:
        """Return the trim target — adaptive to whatever model is currently loaded.

        Source of truth (in priority order):
        1. ``LlamaServerManager.current_ctx_size`` — live value for the running
           llama-server, updated on every model load/swap. Always current,
           no cache to go stale.
        2. ``self._detected_context_length`` — fallback for non-llama backends
           where we probe via ``get_context_length``.

        The manual ``narrative_context_limit`` setting clamps DOWNWARD only:
        it can lower the trim target below the model's max (useful for
        testing tight windows or saving VRAM), but never raise it above —
        exceeding the model's loaded context physically can't work and
        produces 400 ``exceed_context_size_error`` from llama-server.
        """
        from augmentum.config import settings as cfg

        manager = getattr(self._backend, "_manager", None)
        live = int(getattr(manager, "current_ctx_size", 0) or 0) if manager else 0
        actual = live or self._detected_context_length

        if cfg.narrative_context_limit > 0:
            if actual > 0:
                return min(cfg.narrative_context_limit, actual)
            return cfg.narrative_context_limit
        return actual

    async def _refine_trim_with_real_tokens(
        self,
        messages: list[Message],
        budget: int,
        label: str = "messages",
    ) -> list[Message]:
        """Verify the engine's approximate trim using the model's real tokenizer.

        The engine's trim uses tiktoken (cl100k_base), which under-counts
        non-English content, special tokens, and chat-template overhead by
        5-15% on long contexts. This method renders the candidate prompt
        via ``apply_template`` and tokenizes it via ``/tokenize`` — both
        speak to the actual loaded model — and drops oldest non-system
        messages iteratively until the real token count fits ``budget``.

        Returns the original list unchanged when the backend doesn't expose
        ``apply_template`` + ``tokenize`` (Claude, Gemini, ollama) — those
        backends don't have local render+tokenize endpoints, so we trust
        the engine's tiktoken approximation.
        """
        if not messages or budget <= 0:
            return messages

        apply_template = getattr(self._backend, "apply_template", None)
        tokenize = getattr(self._backend, "tokenize", None)
        if not callable(apply_template) or not callable(tokenize):
            return messages

        chat_msgs = [m for m in messages if m.role != "system"]
        candidate_chat = list(chat_msgs)

        # ORDER-PRESERVING candidate build. The previous ``sys_msgs +
        # candidate_chat`` partition looked harmless but silently hoisted
        # every system message to the front — including the dynamic
        # STATE/MEMORY block that ``_augment_request`` deliberately placed
        # just before the latest user turn for prefix-cache friendliness
        # (and the datetime block from ``apply_datetime_context``). Because
        # this refine pass runs on EVERY turn (it returns ``candidate`` even
        # when nothing was dropped), the per-turn-changing block landed at
        # position ~0, the token prefix diverged at message 0, and llama-server
        # re-prefilled the entire context every turn (observed: 12-15 min
        # TTFT on a 61k narrative session that should have warm-started).
        # Trimming must only DROP oldest chat messages, never reorder.
        def _candidate_in_order() -> list[Message]:
            chat_keep = {id(m) for m in candidate_chat}
            return [
                m for m in messages
                if m.role == "system" or id(m) in chat_keep
            ]

        n_tokens = -1
        for iteration in range(3):
            candidate = _candidate_in_order()
            if not candidate:
                break
            try:
                payload = [{"role": m.role, "content": m.content or ""} for m in candidate]
                rendered = await apply_template(payload)
                if not rendered:
                    return messages
                tokens = await tokenize(rendered)
                if not tokens:
                    return messages
                n_tokens = len(tokens)
                if n_tokens <= budget:
                    if iteration > 0:
                        log.info(
                            "trim_refinement_succeeded",
                            label=label,
                            tokens=n_tokens,
                            budget=budget,
                            iterations=iteration + 1,
                            extra_dropped=len(chat_msgs) - len(candidate_chat),
                        )
                    # Nothing dropped → hand back the ORIGINAL list object so
                    # the caller's ``refined_msgs is augmented.messages``
                    # identity fast-path recognizes the no-op.
                    if len(candidate_chat) == len(chat_msgs):
                        return messages
                    return candidate
                excess = n_tokens - budget
                avg = max(1, n_tokens // max(1, len(candidate)))
                n_drop = max(1, excess // avg + 1)  # +1 safety
                if n_drop >= len(candidate_chat):
                    log.warning(
                        "trim_refinement_unable_to_fit",
                        label=label,
                        tokens=n_tokens,
                        budget=budget,
                    )
                    candidate_chat = candidate_chat[-1:] if candidate_chat else []
                    return _candidate_in_order()
                candidate_chat = candidate_chat[n_drop:]
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "trim_refinement_failed",
                    label=label,
                    error=repr(exc),
                    iteration=iteration,
                )
                return messages

        log.warning(
            "trim_refinement_exhausted",
            label=label,
            final_tokens=n_tokens,
            budget=budget,
        )
        return _candidate_in_order()

    async def _refine_augmented_request(
        self,
        result: NarrativeResult,
        request: InternalChatRequest,
        ctx_limit: int,
    ) -> NarrativeResult:
        """Re-trim ``result.augmented_request`` against real model tokens.

        Targets two budgets:
        - chat path: ``ctx_limit - max_tokens``
        - kv_stable_messages: ``ctx_limit - 4096`` (mirror of engine's
          INJECTION_HEADROOM — leaves room for next turn's fresh
          STATE/MEMORY block)

        Returns the same result unchanged if no trim drops occurred.
        """
        if ctx_limit <= 0:
            return result
        augmented = result.augmented_request
        chat_budget = max(0, ctx_limit - (request.max_tokens or 0))
        refined_msgs = await self._refine_trim_with_real_tokens(
            augmented.messages, chat_budget, label="chat",
        )
        refined_stable = augmented.kv_stable_messages
        if refined_stable:
            stable_budget = max(0, ctx_limit - 4096)
            refined_stable = await self._refine_trim_with_real_tokens(
                refined_stable, stable_budget, label="stable",
            )
        if refined_msgs is augmented.messages and refined_stable is augmented.kv_stable_messages:
            return result
        from dataclasses import replace as dataclass_replace
        augmented = dataclass_replace(
            augmented,
            messages=refined_msgs,
            kv_stable_messages=refined_stable,
        )
        return result.__class__(
            augmented_request=augmented,
            context=result.context,
            state=result.state,
            contradictions=result.contradictions,
            new_facts=result.new_facts,
            branch_detected=result.branch_detected,
            is_regeneration=result.is_regeneration,
        )

    def _sync_ui_lorebook(self, request: InternalChatRequest) -> None:
        """Refresh the engine's LoreEngine from the UI-provided lorebook.

        The frontend owns lorebook editing, so the backend copy (populated
        once from character_book at session init) drifts the moment a user
        adds or changes an entry. Every request carries the live session
        lorebook; feed it through ``replace_entries_preserving_state`` so
        sticky/cooldown counters survive across edits.

        When the request carries no ``lorebook`` (e.g. non-UI clients), we
        leave the engine's current entries alone — that preserves the
        character_book path.
        """
        if request.lorebook is None:
            return
        try:
            self._engine._lore_engine.replace_entries_preserving_state(request.lorebook)
        except Exception as exc:  # noqa: BLE001
            log.warning("lorebook_sync_failed", error=str(exc), count=len(request.lorebook or []))

    async def _persist_state(self) -> None:
        """Auto-persist state after each request."""
        if not self._state_manager or not self._session_id:
            return
        try:
            self._engine.sync_to_state()
            await self._state_manager.save_narrative_state(
                self._session_id, self._engine.state, user_id=self._user_id,
            )
            log.info("narrative_state_persisted",
                     session_id=self._session_id,
                     message_count=self._engine.state.message_count,
                     ledger_len=len(self._engine._memory_ledger),
                     has_memory_settings=getattr(self._engine.state, 'memory_settings', None) is not None)
        except Exception as exc:
            log.warning("narrative_state_persist_failed", session_id=self._session_id, error=str(exc))

    async def _memory_model_or_alert(self, *, context: str) -> str | None:
        """The configured ``narrative_memory_model`` if it still resolves.

        Returns ``""`` when no dedicated model is configured (callers use the
        chat model), the model name when it resolves, or ``None`` when the
        configured reference is STALE — in that case a ``model_alert`` is
        stashed on the engine (surfaced by the narrative panel poll as an
        actionable toast: skip / use chat model / use engine model) and the
        caller should skip the refresh rather than silently reroute to
        whatever the default backend has loaded (never auto-select).
        """
        memory_model = await self._user_model_setting("narrative_memory_model")
        if not memory_model:
            return ""
        registry = getattr(self._app_state, "provider_registry", None) if self._app_state else None
        if registry is None or not hasattr(registry, "model_is_resolvable"):
            return memory_model
        try:
            if await registry.model_is_resolvable(memory_model):
                # Model came back (or was fine) — clear any stale alert.
                if getattr(self._engine, "pending_model_alert", None):
                    self._engine.pending_model_alert = None
                return memory_model
        except Exception as exc:  # noqa: BLE001 — resolvability check must not kill the refresh
            log.warning("narrative_memory_model_check_failed", error=str(exc))
            return memory_model
        alert = {
            "setting": "narrative_memory_model",
            "requested": memory_model,
            "context": context,
        }
        if getattr(self._engine, "pending_model_alert", None) != alert:
            self._engine.pending_model_alert = alert
            log.warning(
                "narrative_memory_model_unresolvable",
                requested=memory_model, session_id=self._session_id, context=context,
            )
        return None

    async def _refresh_state_memory(self) -> None:
        """Generate new STATE snapshot + MEMORY entries via LLM call (background task).

        Acquires processing_lock to safely read/write engine state.
        Uses ``narrative_memory_model`` if configured, otherwise the default backend.
        """
        try:
            from augmentum.modes.narrative.memory import CardType, parse_state_memory_response

            # Resolve summary backend — use dedicated model if configured.
            # ``resolve_backend_for_model`` returns ``tuple[ModelBackend, str]``
            # (the backend + the cleaned model name with any ``@backend``
            # suffix stripped); we only need the backend here.
            summary_backend = self._backend
            memory_model = await self._memory_model_or_alert(context="memory_refresh")
            if memory_model is None:
                return  # stale configured model — alert raised, user decides
            if memory_model and self._app_state:
                registry = getattr(self._app_state, "provider_registry", None)
                if registry:
                    resolved = await registry.resolve_backend_with_fabric(memory_model)
                    if resolved and resolved[0] is not None:
                        summary_backend = resolved[0]

            async with self._engine.processing_lock:
                batch_start = max(1, (self._engine.state.last_summary_at or 0) + 1)
                batch_end = self._engine.state.message_count
                summary_request = self._engine.build_state_memory_request(
                    batch_start, batch_end,
                    model=memory_model or self._last_model,
                )
                # Route off slot 0 so this state/memory refresh runs in
                # parallel with the user's next chat turn instead of queueing
                # behind it.
                summary_request.is_background_task = True
                card_type_str = self._engine.state.card_type

            response = await summary_backend.chat(summary_request)

            async with self._engine.processing_lock:
                if response.message and response.message.content:
                    try:
                        card_type = CardType(card_type_str)
                    except ValueError:
                        card_type = CardType.CHARACTER
                    snapshot, entries = parse_state_memory_response(
                        response.message.content, card_type, batch_start, batch_end,
                    )
                    # Respect per-layer toggles — discard unwanted portions
                    if not self._mem_setting("memory_state_enabled"):
                        snapshot = None
                    if not self._mem_setting("memory_ledger_enabled"):
                        entries = []
                    self._engine.apply_state_memory_response(snapshot, entries, batch_end=batch_end)

                    await self._persist_state()
        except Exception:
            log.warning(
                "narrative_memory_refresh_failed",
                session_id=self._session_id,
                exc_info=True,
            )

    async def _compact_ledger(self) -> None:
        """Compact the memory ledger by merging oldest entries via LLM (background task)."""
        try:
            from augmentum.config import settings as cfg
            from augmentum.modes.narrative.memory import (
                CardType,
                build_compaction_prompt,
            )

            async with self._engine.processing_lock:
                ledger = self._engine.memory_ledger
                if not ledger or not self._engine.needs_compaction:
                    return

                ratio = cfg.narrative_memory_compaction_ratio
                compact_count = max(1, int(len(ledger) * ratio))
                entries_to_compact = ledger[:compact_count]
                entries_to_keep = ledger[compact_count:]

                try:
                    card_type = CardType(self._engine.state.card_type)
                except ValueError:
                    card_type = CardType.CHARACTER

            system_content, user_content = build_compaction_prompt(entries_to_compact, card_type)
            # Scale token budget: ~32 tokens per entry output, floor at 600.
            # The old hard-coded 400 caused silent truncation for batches > ~16 entries.
            scaled_tokens = max(600, len(entries_to_compact) * 32)
            memory_model = await self._memory_model_or_alert(context="memory_compaction")
            if memory_model is None:
                return  # stale configured model — alert raised, user decides
            compact_request = InternalChatRequest(
                model=memory_model or self._last_model,
                messages=[
                    Message(role="system", content=system_content),
                    Message(role="user", content=user_content),
                ],
                stream=False,
                temperature=0.0,  # mechanical task — maximum determinism
                max_tokens=scaled_tokens,
                is_background_task=True,  # off slot 0 — see chat() docstring
            )
            response = await self._backend.chat(compact_request)

            if response.message and response.message.content:
                import re

                from augmentum.modes.narrative.memory import MemoryEntry

                compacted: list[MemoryEntry] = []
                entry_pattern = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
                for line in response.message.content.strip().split("\n"):
                    line = line.strip().lstrip("-* ")
                    m = entry_pattern.match(line)
                    if m:
                        compacted.append(MemoryEntry(
                            round_num=int(m.group(1)),
                            category=m.group(2).strip().lower().replace(" ", "_"),
                            content=m.group(3).strip(),
                        ))

                if compacted:
                    # Rescue any input entries whose R# is missing from the output.
                    # This handles silent truncation (token limit hit mid-response) and
                    # LLM drift — whichever dropped entries, we put the originals back.
                    output_rounds = {e.round_num for e in compacted}
                    rescued = [e for e in entries_to_compact if e.round_num not in output_rounds]
                    if rescued:
                        log.warning(
                            "narrative_compaction_rescued_entries",
                            session_id=self._session_id,
                            rescued=[e.round_num for e in rescued],
                        )
                        compacted = sorted(
                            compacted + rescued,
                            key=lambda e: e.round_num,
                        )

                    async with self._engine.processing_lock:
                        # Re-slice from the live ledger — a concurrent refresh may
                        # have extended it while the LLM was compacting.  Using the
                        # stale `entries_to_keep` would silently drop those entries.
                        entries_to_keep = self._engine._memory_ledger[compact_count:]
                        self._engine._memory_ledger = compacted + entries_to_keep
                        self._engine.needs_compaction = False
                        # Re-sync regen snapshot so it doesn't point past the compacted ledger
                        self._engine._pre_refresh_ledger_len = len(self._engine._memory_ledger)
                        await self._persist_state()
                    log.info(
                        "narrative_ledger_compacted",
                        session_id=self._session_id,
                        before=len(entries_to_compact),
                        after=len(compacted),
                    )
        except Exception:
            log.warning(
                "narrative_ledger_compaction_failed",
                session_id=self._session_id,
                exc_info=True,
            )

    async def _archive_exchange(self, user_msg: str, assistant_msg: str, turn_number: int) -> None:
        """Archive a single exchange pair continuously (background task)."""
        if not user_msg or not assistant_msg:
            return
        await self._archive_and_embed([(user_msg, assistant_msg)])

    def _maybe_refresh_summary(self) -> asyncio.Task | None:
        """Fire a background state+memory refresh if the interval has been reached."""
        mem_enabled = self._mem_setting("memory_enabled")
        state_enabled = self._mem_setting("memory_state_enabled")
        ledger_enabled = self._mem_setting("memory_ledger_enabled")
        interval = self._mem_setting("memory_interval")

        if not mem_enabled:
            log.info("narrative_memory_skip", reason="memory_enabled=False",
                     session_id=self._session_id)
            return None
        if not state_enabled and not ledger_enabled:
            log.info("narrative_memory_skip", reason="no_active_layers",
                     session_id=self._session_id)
            return None  # Neither consumer active — skip the LLM call
        if self._refresh_in_flight:
            return None
        if not self._engine.should_refresh(interval):
            return None
        # Use in-memory flag to prevent duplicate triggers from rapid message
        # sends (e.g. group chat).  Do NOT mutate last_summary_at here — that
        # is persisted to SQLite, and if the server restarts before the
        # background task completes the refresh would be permanently skipped.
        self._refresh_in_flight = True

        log.debug(
            "narrative_memory_refresh_triggered",
            session_id=self._session_id,
            message_count=self._engine.state.message_count,
        )
        task = asyncio.create_task(self._refresh_state_memory())
        task.add_done_callback(self._on_refresh_done)
        return task

    def _on_refresh_done(self, task: asyncio.Task) -> None:
        """Clear in-flight flag and log errors from background refresh."""
        self._refresh_in_flight = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.warning("narrative_memory_refresh_task_failed", error=str(exc))

    def _maybe_compact_ledger(self) -> asyncio.Task | None:
        """Fire ledger compaction if ceiling reached."""
        if not self._mem_setting("memory_enabled"):
            return None
        if not self._mem_setting("memory_ledger_enabled"):
            return None
        if not self._mem_setting("memory_compaction_enabled"):
            return None
        if not self._engine.needs_compaction:
            return None

        task = asyncio.create_task(self._compact_ledger())
        task.add_done_callback(_on_refresh_error)
        return task

    def _maybe_archive_exchange(self, response_text: str) -> asyncio.Task | None:
        """Fire continuous archiving — batched every 5 complete exchanges.

        Archives ALL exchanges since the last archive in a single batched LLM call.
        msg_count increments twice per exchange (user + assistant), so we use
        exchange_count = msg_count // 2 for the interval check.
        """
        if not self._mem_setting("memory_enabled"):
            return None
        if not self._mem_setting("memory_continuous_archive"):
            return None
        if not self._state_manager:
            return None

        msg_count = self._engine.state.message_count
        # Convert raw message count to complete exchange count (2 messages per exchange)
        exchange_count = msg_count // 2
        archive_every = 5  # exchanges
        if exchange_count < archive_every or exchange_count % archive_every != 0:
            return None

        # Gather ALL unarchived exchanges since the last archive run.
        # _message_history alternates user/assistant messages.
        history = self._engine._message_history
        unarchived = history[self._last_archived_history_idx:]
        pairs: list[tuple[str, str]] = []
        for i in range(0, len(unarchived) - 1, 2):
            u, a = unarchived[i], unarchived[i + 1]
            # Skip pairs where the assistant response is a refusal
            if u and a and not self._engine._is_refusal(a):
                pairs.append((u, a))

        if not pairs:
            return None

        # Capture message_count NOW — the background task runs later when
        # message_count may have advanced, producing wrong turn_number values.
        msg_count_at_capture = self._engine.state.message_count

        # Advance pointer before the task runs to prevent double-archiving
        self._last_archived_history_idx += len(pairs) * 2

        task = asyncio.create_task(self._archive_and_embed(pairs, msg_count_at_capture))
        task.add_done_callback(_on_refresh_error)
        return task

    def _maybe_world_reconcile(self, response_text: str) -> asyncio.Task | None:
        """Background drift check: did the narration establish tracker
        changes the model never applied via world.track.shift? Produces
        SUGGESTIONS only (spec D1 — no silent writes); the drawer offers
        them as tap-to-accept chips."""
        manifest = self._world_manifest()
        if manifest is None or not manifest.has("trackers"):
            return None
        task = asyncio.create_task(self._run_world_reconcile(response_text))
        task.add_done_callback(_on_refresh_error)
        return task

    async def _run_world_reconcile(self, response_text: str) -> None:
        try:
            from augmentum.config import settings as cfg
            from augmentum.modes.narrative.world_system import (
                extract_drift_suggestions,
            )
            manifest = self._world_manifest()
            if manifest is None:
                return
            registry = (
                getattr(self._app_state, "provider_registry", None)
                if self._app_state else None
            )
            if registry:
                backend, model = await registry.resolve_model_for_role(
                    "utility", override="", settings=cfg,
                )
            else:
                backend, model = self._backend, self._last_model
            suggestions = await extract_drift_suggestions(
                backend, model, self._world_store(manifest), response_text,
            )
            self._world_suggestions = {
                "turn": self._engine.state.message_count,
                "items": suggestions,
            }
            if suggestions:
                log.info(
                    "world_drift_suggestions", session_id=self._session_id,
                    count=len(suggestions),
                    trackers=[s["tracker"] for s in suggestions],
                )
        except Exception:
            log.warning("world_reconcile_failed",
                        session_id=self._session_id, exc_info=True)

    def _maybe_llm_extract(self, response_text: str) -> asyncio.Task | None:
        """Fire background LLM extraction if enabled and interval reached.

        LLM extraction runs independently of regex state tracking.
        Even when narrative_state_tracking_enabled is False, LLM extraction
        can accumulate characters, relationships, plots, and facts —
        these are higher quality than regex and safe to accumulate.

        Respects ``narrative_extraction_interval`` — only fires every N
        messages to reduce LLM call volume.
        """
        from augmentum.config import settings
        if not self._mem_setting("memory_enabled"):
            return None
        if not settings.narrative_llm_extraction:
            return None

        interval = max(1, settings.narrative_extraction_interval)
        msg_count = self._engine.state.message_count
        # Always run on the first few messages to seed initial state,
        # then respect the interval after that
        if interval > 1 and msg_count > interval and msg_count % interval != 0:
            return None

        task = asyncio.create_task(self._run_llm_extraction(response_text))
        task.add_done_callback(_on_refresh_error)
        return task

    async def _run_llm_extraction(self, response_text: str) -> None:
        """Run LLM-based narrative extraction and merge results (background)."""
        try:
            from augmentum.config import settings as cfg
            from augmentum.modes.narrative.llm_extractor import extract_narrative_state

            # Gather known character names
            known_chars = [
                e.name for e in self._engine.state.entities.values()
                if e.entity_type.value == "character"
            ]

            # Resolve extraction backend via role system
            extraction_model = await self._user_model_setting("narrative_extraction_model")
            registry = getattr(self._app_state, "provider_registry", None) if self._app_state else None
            if registry:
                extraction_backend, effective_model = await registry.resolve_model_for_role(
                    "utility",
                    override=extraction_model,
                    settings=cfg,
                )
            else:
                extraction_backend = self._backend
                effective_model = extraction_model or self._last_model

            extraction = await extract_narrative_state(
                backend=extraction_backend,
                text=response_text,
                known_characters=known_chars,
                model=effective_model,
            )
            if extraction:
                async with self._engine.processing_lock:
                    message_index = self._engine.state.message_count - 1
                    self._engine.merge_llm_extraction(extraction, message_index)
                    await self._persist_state()
        except Exception:
            log.debug(
                "narrative_llm_extraction_task_failed",
                session_id=self._session_id,
                exc_info=True,
            )

    def _maybe_handle_overflow(self) -> asyncio.Task | None:
        """Legacy — overflow is now handled by continuous archiving. Returns None."""
        return None

    def _maybe_generate_background(self, request: InternalChatRequest) -> asyncio.Task | None:
        """Fire background image generation for scene backgrounds if interval reached."""
        from augmentum.config import settings
        if not settings.narrative_auto_background:
            return None
        if not self._image_enabled and not self._app_state:
            return None

        interval = settings.narrative_auto_background_interval
        msg_count = self._engine.state.message_count
        # Only trigger on assistant turns (even message indices) and at the interval
        if msg_count < interval or msg_count % interval != 0:
            return None

        log.debug(
            "narrative_auto_background_triggered",
            session_id=self._session_id,
            message_count=msg_count,
        )
        task = asyncio.create_task(self._generate_background(request))
        task.add_done_callback(_on_refresh_error)
        return task

    async def _generate_background(self, request: InternalChatRequest) -> None:
        """Generate a scene background and store the URL for the session."""
        try:
            overrides: dict = {"width": 1280, "height": 720}

            # Use dedicated image model if configured
            bg_image_model = await self._user_model_setting("narrative_auto_bg_image_model")
            if bg_image_model:
                overrides["model"] = bg_image_model

            # Use dedicated distiller LLM if configured
            bg_distiller_model = await self._user_model_setting("narrative_auto_bg_distiller_model")
            if bg_distiller_model:
                overrides["distiller_model"] = bg_distiller_model

            image_url = await self._generate_scene_image(
                user_instruction="Generate a wide atmospheric background image of the current scene. "
                "Focus on environment, lighting, and mood. No text or UI elements.",
                request=request,
                image_overrides=overrides,
                distill_mode="background",
                previous_prompts=self._previous_bg_prompts,
                # Auto-triggered per-turn background — routes to the
                # corner badge surface, not the in-message loader.
                category="auto_bg",
            )
            if image_url and self._app_state:
                # Store the background URL on the engine for the UI to poll
                backgrounds = getattr(self._app_state, "narrative_backgrounds", None)
                if backgrounds is None:
                    self._app_state.narrative_backgrounds = {}
                    backgrounds = self._app_state.narrative_backgrounds
                backgrounds[self._session_id] = image_url
                log.info(
                    "narrative_auto_background_generated",
                    session_id=self._session_id,
                    image_url=image_url,
                )
        except Exception:
            log.debug(
                "narrative_auto_background_failed",
                session_id=self._session_id,
                exc_info=True,
            )

    async def _archive_and_embed(
        self, pairs: list[tuple[str, str]],
        msg_count_at_capture: int | None = None,
    ) -> None:
        """Generate micro-summaries, embed, and store exchange pairs in the archive.

        Each pair gets a one-line LLM summary embedded for retrieval.
        The full exchange text is stored for injection.

        ``msg_count_at_capture`` is the message_count when the pairs were built.
        Using it (instead of the live value) produces accurate turn_number values
        even when the background task runs after additional exchanges have advanced
        the counter.
        """
        if not pairs or not self._state_manager:
            return

        import uuid

        from augmentum.models.base import InternalChatRequest, Message

        try:
            # Build micro-summary request for all pairs at once
            pair_texts = []
            for i, (user_msg, asst_msg) in enumerate(pairs):
                pair_texts.append(
                    f"[Exchange {i + 1}]\n"
                    f"User: {user_msg[:300]}\n"
                    f"Assistant: {asst_msg[:300]}"
                )

            summary_prompt = (
                "Summarize each exchange below into ONE sentence for a story memory index.\n"
                "RULES: (1) Always use the character's actual name — never pronouns like "
                "'she' or 'he'. (2) Include any specific location, object, or event name "
                "mentioned. (3) Lead with the most important subject (character name or "
                "key event). Format: one numbered line per exchange.\n\n"
                + "\n\n".join(pair_texts)
            )

            # Decide whether to spend an LLM call on summarization. The
            # summary feeds the embedder for vector retrieval; truncated
            # raw text embeds nearly as well as an LLM-written summary,
            # and the LLM call here was loading the user's 27B chat model
            # to write 5×one-sentence summaries — which on a single-GPU
            # single-model setup halves the user's chat tok/s while it
            # runs. We use the LLM path ONLY when the user has explicitly
            # configured a *separate* utility model (so loading it doesn't
            # cost their chat-model context window). Otherwise, fall back
            # to a heuristic summary that needs no GPU.
            summaries: list[str] = []
            use_llm_summary = False
            summary_backend = self._backend
            summary_model = self._last_model
            registry = getattr(self._app_state, "provider_registry", None) if self._app_state else None
            if registry:
                try:
                    resolved = await registry.resolve_model_for_role("utility")
                    if resolved and resolved[0] is not None:
                        utility_backend, utility_model_name = resolved
                        # Only use the LLM summary when the resolved utility
                        # model is genuinely different from the chat model.
                        # When they're the same (utility_model unset →
                        # falls back to primary_chat_model), the LLM call
                        # would hit the same model and starve the chat.
                        if utility_model_name and utility_model_name != self._last_model:
                            summary_backend = utility_backend
                            summary_model = utility_model_name
                            use_llm_summary = True
                except Exception:
                    log.debug("archive_summary_role_resolve_failed", exc_info=True)

            if use_llm_summary:
                summary_request = InternalChatRequest(
                    model=summary_model,
                    messages=[
                        Message(role="system", content="You are a story memory indexer. Write dense, entity-rich one-sentence summaries."),
                        Message(role="user", content=summary_prompt),
                    ],
                    stream=False,
                    temperature=0.1,
                    max_tokens=max(150, len(pairs) * 45),
                    is_background_task=True,
                )

                response = await summary_backend.chat(summary_request)
                if response.message and response.message.content:
                    import re
                    for line in response.message.content.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                        if cleaned:
                            summaries.append(cleaned)

            # Heuristic fallback (also pads when the LLM returned fewer
            # summaries than expected). For single-model setups this is
            # the entire path — no GPU work, archive runs free of the
            # chat slot's compute budget.
            while len(summaries) < len(pairs):
                idx = len(summaries)
                u, a = pairs[idx]
                u_clip = (u or "").strip().replace("\n", " ")[:120]
                a_clip = (a or "").strip().replace("\n", " ")[:120]
                summaries.append(f"User: {u_clip} | Assistant: {a_clip}")

            # Embed summaries
            from augmentum.memory.embeddings import EmbeddingService

            summary_embeddings = await asyncio.to_thread(EmbeddingService.embed, summaries)

            # Build exchange dicts for persistence
            effective_count = msg_count_at_capture if msg_count_at_capture is not None else self._engine.state.message_count
            turn_base = max(0, effective_count - len(pairs) * 2)
            exchanges = []
            for i, (user_msg, asst_msg) in enumerate(pairs):
                emb_blob = EmbeddingService.to_blob(summary_embeddings[i])
                exchanges.append({
                    "id": str(uuid.uuid4()),
                    "user_content": user_msg,
                    "assistant_content": asst_msg,
                    "summary": summaries[i],
                    "turn_number": turn_base + i * 2,
                    "embedding_blob": emb_blob,
                })

            # Store via persistence layer
            from augmentum.state.backends.sqlite import SQLiteBackend
            from augmentum.state.narrative_persistence import NarrativePersistence
            backend = getattr(self._state_manager, "_backend", None)
            if isinstance(backend, SQLiteBackend):
                # Ensure sessions row exists — narrative_archive has FK to sessions(id)
                await self._state_manager.get_or_create_session(
                    self._session_id, mode="narrative",
                )
                persistence = NarrativePersistence(backend.conn)
                # Branch-tag every archive row at write time (Phase 3 chunk 1).
                # For sessions that pre-date migrations 115-119, the legacy
                # path's default 'main' is correct; for newly-created sessions
                # the engine's attach_persistence seeded 'main' too.
                current_branch = self._engine._branch_tracker.current_branch  # noqa: SLF001
                await persistence.store_archive_exchanges_for_branch(
                    self._session_id, exchanges,
                    user_id=self._user_id, branch_id=current_branch,
                )
                log.info(
                    "narrative_archive_stored",
                    session_id=self._session_id,
                    exchanges=len(exchanges),
                    branch_id=current_branch,
                )
        except Exception:
            log.warning(
                "narrative_archive_embed_failed",
                session_id=self._session_id,
                exc_info=True,
            )

    async def _prune_archive_for_branch(self) -> None:
        """Remove vectorized archive entries from the abandoned branch path.

        Phase 3 chunk 2: this destructive prune is a no-op when persistence
        is attached to the engine — branch-aware retrieval (ancestry-filtered
        SQL) makes the prune unnecessary AND the prune itself is the source
        of the silent-archive-corruption bug (deletes by turn_number with no
        branch_id awareness, eating the abandoned branch's content forever).

        The legacy path (no persistence attached) keeps the prune so behavior
        is identical for sessions that haven't been touched since the upgrade.
        """
        if not self._state_manager or not self._session_id:
            return
        # Skip the destructive prune when branch-aware retrieval is in play.
        if getattr(self._engine, "_persistence", None) is not None:
            log.debug("branch_archive_prune_skipped_branch_aware",
                      session_id=self._session_id)
            return
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            from augmentum.state.narrative_persistence import NarrativePersistence
            backend = getattr(self._state_manager, "_backend", None)
            if isinstance(backend, SQLiteBackend):
                persistence = NarrativePersistence(backend.conn)
                turn = self._engine.state.message_count
                await persistence.prune_archive_after_turn(
                    self._session_id, turn, user_id=self._user_id,
                )
        except Exception:
            log.debug("branch_archive_prune_failed", exc_info=True)

    async def _prune_archive_for_regen(self) -> None:
        """Remove the last archived exchange so it can be re-archived with the new response.

        The archive fired after the first generation stored (user_N, asst_N_old).
        On regen, asst_N_old is discarded. We prune that entry so future retrievals
        don't surface the discarded response as historical context.
        """
        if not self._state_manager or not self._session_id:
            return
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            from augmentum.state.narrative_persistence import NarrativePersistence
            backend = getattr(self._state_manager, "_backend", None)
            if isinstance(backend, SQLiteBackend):
                persistence = NarrativePersistence(backend.conn)
                # The last archived exchange's turn_number = message_count - 2
                # (archive assigns turn_base + (N-1)*2 for the last of N pairs).
                # Prune it, then roll back the archive pointer so the exchange
                # gets re-archived (with the new response) in the next batch.
                msg_count = self._engine.state.message_count
                last_exchange_turn = max(0, msg_count - 2)
                pruned = await persistence.prune_archive_after_turn(
                    self._session_id, last_exchange_turn - 1,
                    user_id=self._user_id,
                )
                if pruned > 0:
                    # Roll back archive pointer so the last exchange is re-archived
                    self._last_archived_history_idx = max(
                        0, self._last_archived_history_idx - 2
                    )
                    log.info(
                        "narrative_regen_archive_pruned",
                        session_id=self._session_id,
                        pruned=pruned,
                        last_exchange_turn=last_exchange_turn,
                    )
        except Exception:
            log.warning("regen_archive_prune_failed", exc_info=True)

    async def _prefetch_branch_state_snapshot(self, request) -> None:
        """If the upcoming process_request will create a branch, fetch the
        most recent STATE snapshot from history and stash it on the engine.

        Phase 3 chunk 3: closes the empty-STATE-on-first-turn-after-branch
        hole. Without this, ``rollback_to`` wipes STATE to None and the very
        first generation on the new branch loses STATE injection until the
        post-response background refresh fires.

        Read-only operation: peeks at branch detection without mutating the
        tracker. The actual branch application still happens inside
        ``process_request``. If anything fails (no persistence, no branch,
        no snapshot in history), the legacy wipe-to-None path runs unchanged.
        """
        if not self._state_manager or not self._session_id:
            return
        try:
            from augmentum.state.backends.sqlite import SQLiteBackend
            backend = getattr(self._state_manager, "_backend", None)
            if not isinstance(backend, SQLiteBackend) or not self._user_id:
                return

            detection = self._engine.peek_branch_detection(request)
            if not detection.is_branch:
                return

            from augmentum.state.narrative_persistence import NarrativePersistence
            persistence = NarrativePersistence(backend.conn)

            # Make sure the new branch is registered so ancestry includes it.
            # Idempotent — INSERT OR IGNORE.
            await persistence.upsert_branch(
                self._session_id,
                detection.new_branch_id,
                detection.parent_branch_id,
                detection.branch_point,
                user_id=self._user_id,
            )
            ancestry = await persistence.get_branch_ancestry(
                self._session_id, detection.new_branch_id, user_id=self._user_id,
            )
            snapshot_data = await persistence.get_state_snapshot_at(
                self._session_id, ancestry, detection.branch_point,
                user_id=self._user_id,
            )
            self._engine.prepare_branch_snapshot(snapshot_data)
            if snapshot_data is not None:
                log.info(
                    "branch_state_prefetched",
                    session_id=self._session_id,
                    new_branch=detection.new_branch_id,
                    branch_point=detection.branch_point,
                )
        except Exception:
            log.debug("prefetch_branch_state_failed",
                      session_id=self._session_id, exc_info=True)

    async def _retrieve_from_archive(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant archived exchanges via vector similarity, bounded
        by the current branch's ancestry.

        Phase 3 chunk 2: prefers the branch-aware retrieval path
        (retrieve_archive_for_branch with ancestry). Falls back to the legacy
        unfiltered path only for sessions that have no narrative_branches row
        yet (pre-migration sessions that haven't been touched since the
        upgrade) — that path returns identical results to before.

        Returns list of {user_content, assistant_content, summary, turn_number,
        branch_id, distance}.
        """
        if not self._state_manager or not self._session_id:
            return []

        try:
            from augmentum.memory.embeddings import EmbeddingService
            from augmentum.state.backends.sqlite import SQLiteBackend

            backend = getattr(self._state_manager, "_backend", None)
            if not isinstance(backend, SQLiteBackend):
                return []

            query_emb = await asyncio.to_thread(EmbeddingService.embed_query, query)
            query_blob = EmbeddingService.to_blob(query_emb)

            from augmentum.state.narrative_persistence import NarrativePersistence
            persistence = NarrativePersistence(backend.conn)

            # Branch-aware path: walk ancestry from current branch, filter rows
            # by `branch_id IN ancestry AND turn_number < descendant.branch_point`
            # so we never see content from other branches' divergent paths.
            current_branch = self._engine._branch_tracker.current_branch  # noqa: SLF001
            ancestry = await persistence.get_branch_ancestry(
                self._session_id, current_branch, user_id=self._user_id,
            )

            # If the session has zero branch rows in narrative_branches, the
            # ancestry walk falls back to a single-element [(<branch>, 0)].
            # In that pre-migration case, route through the legacy retrieval
            # so we behave identically to before.
            branches = await persistence.list_branches(
                self._session_id, user_id=self._user_id,
            )
            if not branches:
                return await persistence.retrieve_archive_exchanges(
                    self._session_id, query_blob, limit=limit,
                    user_id=self._user_id,
                )

            return await persistence.retrieve_archive_for_branch(
                self._session_id, query_blob, ancestry,
                user_id=self._user_id, limit=limit,
            )
        except Exception:
            log.debug(
                "narrative_archive_retrieve_failed",
                session_id=self._session_id,
                exc_info=True,
            )
            return []

    async def _retrieve_archive_for_request(
        self, request: InternalChatRequest,
    ) -> list[dict]:
        """Extract query context from request and retrieve relevant archive exchanges.

        Uses the last 4 non-system messages (2 turns) as the query rather than
        just the final user message.  Short or ambiguous messages like "continue"
        or "what did she say?" produce near-random embeddings in isolation; the
        surrounding context anchors the semantic search to the right topic.
        """
        from augmentum.config import settings as _cfg
        if not self._mem_setting("memory_enabled"):
            return []
        if not self._mem_setting("smart_retrieval"):
            return []

        # Don't inject archive context until enough messages have accumulated.
        # Before the threshold the full conversation is still in the context
        # window, so archive injection just duplicates what the model can see.
        # Archive continues to BUILD during this period so it's ready when needed.
        min_msgs = _cfg.narrative_archive_min_messages
        if min_msgs > 0 and self._engine.state.message_count < min_msgs:
            return []

        # Build a context-enriched query from the last 2 turns (4 messages).
        # Role prefixes help the embedding model distinguish speaker voices.
        non_system = [m for m in request.messages if m.role != "system" and m.content]
        if not non_system:
            return []
        context_parts = []
        for msg in non_system[-4:]:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            context_parts.append(f"{prefix} {msg.content[:250]}")
        query = "\n".join(context_parts)

        return await self._retrieve_from_archive(
            query, limit=self._mem_setting("smart_retrieval_count"),
        )

    async def _resolve_user_persona(self):
        """Load the authenticated user's default persona from the database.

        Without ``self._user_id`` filtering, every tenant previously saw the
        first is_default row anywhere in the table (almost always the admin's
        persona) — hence the cross-account persona leak the user reported.
        """
        if not self._app_state:
            return None
        state_mgr = getattr(self._app_state, "state_manager", None)
        if not state_mgr:
            return None
        from augmentum.state.backends.sqlite import SQLiteBackend
        if not isinstance(state_mgr.backend, SQLiteBackend):
            return None
        try:
            query = (
                "SELECT name, appearance, description FROM user_personas "
                "WHERE is_default = 1"
            )
            params: list = []
            if self._user_id:
                query += " AND user_id = ?"
                params.append(self._user_id)
            query += " LIMIT 1"
            cursor = await state_mgr.backend.conn.execute(query, params)
            row = await cursor.fetchone()
            if row:
                from augmentum.image.distiller import UserPersona
                return UserPersona(name=row[0], appearance=row[1], description=row[2])
        except Exception:
            log.debug("persona_resolve_failed", exc_info=True)
        return None

    async def _resolve_core_profile(self) -> str:
        """Get the core memory profile text if available."""
        if not self._app_state:
            return ""
        profile_mgr = getattr(self._app_state, "core_profile_manager", None)
        if not profile_mgr:
            return ""
        try:
            return await profile_mgr.get_profile(self._user_id or "default")
        except Exception:
            return ""

    async def _generate_scene_image(
        self,
        user_instruction: str,
        request: InternalChatRequest,
        image_overrides: dict | None = None,
        distill_mode: str = "auto",
        previous_prompts: list[str] | None = None,
        category: str = "user",
    ) -> str | None:
        """Generate an image from current scene state via the distiller.

        Sends character card, user persona, core profile, scene state,
        and recent conversation rounds to the distiller LLM, then forwards
        the resulting prompt to the image generation backend.

        The image model is resolved as: overrides > card.image_model > narrative config > global default.

        Args:
            distill_mode: "auto" (detect from instruction), "scene", "portrait",
                          or "background" (environment-only, no characters).

        Returns the image URL or None on failure.
        """
        if not self._image_enabled and not self._app_state:
            return None

        # Defensive rehydrate: a handler cached from before this fix may have
        # loaded state without populating the card. Covers that path plus any
        # future entry point that doesn't go through _ensure_state_loaded.
        await self._ensure_character_card_loaded()

        overrides = image_overrides or {}

        # Read the user's current image panel settings as base defaults, so
        # narrative image gen matches what the user configured in the image
        # panel (model, resolution, steps, CFG, …). PER-USER — reading the
        # process-global mirror ignored the authenticated user's selection and
        # fell back to the install default. See image/active_settings.py.
        from augmentum.image.active_settings import resolve_active_settings
        ui = await resolve_active_settings(self._app_state, self._user_id)
        log.info(
            "scene_image_ui_settings",
            has_app_state=self._app_state is not None,
            ui_keys=list(ui.keys()),
            ui_width=ui.get("width"),
            ui_height=ui.get("height"),
            ui_model=ui.get("model"),
            overrides_keys=list((image_overrides or {}).keys()),
        )

        try:
            from augmentum.config import settings
            from augmentum.image.distiller import (
                _extract_conversation_rounds,
                build_scene_context,
                distill_scene,
            )
            from augmentum.image.queue import GenerationJob

            # Resolve image model FIRST so the distiller can adapt its prompt style
            # Priority: explicit override > card setting > UI panel > narrative config > global default
            image_model = overrides.get("model", "")
            if not image_model:
                card = self._engine.character_card
                if card and card.image_model:
                    image_model = card.image_model
            if not image_model:
                image_model = ui.get("model", "")
            if not image_model:
                image_model = settings.narrative_scene_image_model
            if not image_model:
                image_model = settings.image_default_model

            # Build a read-only SceneContext for the distiller. The engine
            # is NEVER mutated. If the UI sent character data that disagrees
            # with the engine's cached card, we strip the engine's narrative
            # state from the context as well — that state belongs to a
            # different character (cross-session/cross-character chimera
            # protection). Engine remains intact for the actual chat path.
            from dataclasses import replace as _dc_replace

            from augmentum.modes.narrative.card_parser import CharacterCard

            ui_visual_traits = overrides.pop("visual_traits", "")
            ui_char_name = overrides.pop("character_name", "")

            engine_card = self._engine.character_card
            card_override: CharacterCard | None = None
            trust_engine_state = True

            if ui_visual_traits or ui_char_name:
                if engine_card is None:
                    card_override = CharacterCard(
                        name=ui_char_name,
                        visual_traits=ui_visual_traits,
                        source_format="ui_override",
                    )
                    log.info(
                        "scene_card_built_from_ui",
                        name=ui_char_name,
                        has_visual_traits=bool(ui_visual_traits),
                    )
                else:
                    engine_name = (engine_card.name or "").strip().lower()
                    ui_name = (ui_char_name or "").strip().lower()
                    if ui_name and engine_name and ui_name != engine_name:
                        # Mismatch: UI is asserting a different character than
                        # the engine has cached. Treat UI as authoritative AND
                        # drop the engine's narrative state — it belongs to
                        # the other character.
                        log.warning(
                            "scene_card_identity_mismatch",
                            engine_card_name=engine_card.name,
                            ui_card_name=ui_char_name,
                            session_id=self._session_id,
                            hint="UI character disagrees with cached engine. "
                                 "Building payload-only context (no engine state).",
                        )
                        card_override = CharacterCard(
                            name=ui_char_name,
                            visual_traits=ui_visual_traits,
                            source_format="ui_override",
                        )
                        trust_engine_state = False
                    else:
                        # Same character (or UI has no name to compare). Patch
                        # in fresher UI traits/name without touching the engine.
                        new_traits = ui_visual_traits or engine_card.visual_traits
                        new_name = engine_card.name or ui_char_name
                        if new_traits != engine_card.visual_traits or new_name != engine_card.name:
                            card_override = _dc_replace(
                                engine_card, name=new_name, visual_traits=new_traits,
                            )

            scene_ctx = build_scene_context(
                engine=self._engine,
                card_override=card_override,
                trust_engine_state=trust_engine_state,
            )

            # Gather context for the distiller
            conversation_messages = _extract_conversation_rounds(
                request, rounds=settings.narrative_scene_context_rounds,
            )
            persona = await self._resolve_user_persona()
            core_profile = await self._resolve_core_profile()

            # Build group member visual data for the distiller (if group chat)
            group_members_for_distiller = None
            if self._active_group and self._group_member_cards:
                group_members_for_distiller = []
                for name in self._active_group.member_names:
                    card = (self._group_member_cards or {}).get(name, {})
                    group_members_for_distiller.append({
                        "name": name,
                        "visual_traits": card.get("visual_traits", "") or card.get("visualTraits", ""),
                        "appearance": card.get("appearance", ""),
                        "description": card.get("description", ""),
                        "species": card.get("species", ""),
                    })
                log.info("distiller_group_context",
                         members=len(group_members_for_distiller),
                         with_traits=sum(1 for m in group_members_for_distiller if m["visual_traits"]))

            # Distill narrative state into image prompt (model-aware)
            distiller_model = overrides.pop("distiller_model", "") or settings.narrative_scene_distiller_model
            # Route to the correct backend for the distiller model
            distill_backend = self._backend
            if distiller_model and self._app_state:
                registry = getattr(self._app_state, "provider_registry", None)
                if registry:
                    try:
                        resolved_backend, distiller_model = await registry.resolve_backend_with_fabric(
                            distiller_model
                        )
                        distill_backend = resolved_backend
                    except Exception:
                        log.debug("distiller_backend_resolve_fallback", model=distiller_model, exc_info=True)
            # Pre-queue phase indicator. The distiller LLM runs for
            # 3-10s BEFORE we have a GenerationJob to attach progress
            # to. Without surfacing this window the UI loader sits
            # silent for a third of the total time on small models.
            # /api/image/generation-status reads this dict scoped by
            # session_id and reports it as ``pre_queue`` to surfaces
            # that poll. Cleared in a finally block below so a failure
            # in the distill path doesn't leave a phantom indicator.
            import time as _t
            if self._app_state is not None:
                pre_q = getattr(self._app_state, "scene_image_pre_queue", None)
                if pre_q is None:
                    self._app_state.scene_image_pre_queue = {}
                    pre_q = self._app_state.scene_image_pre_queue
                pre_q[self._session_id] = {
                    "phase": "distilling",
                    "stage": "Composing scene prompt",
                    "started_at": _t.monotonic(),
                    # Mirror the category through to the pre-queue
                    # record so endpoint filters can hide it from the
                    # wrong surface (e.g. the auto-bg badge doesn't
                    # surface the user-initiated illustrate's distill).
                    "category": category,
                }
            try:
                distilled = await distill_scene(
                    ctx=scene_ctx,
                    backend=distill_backend,
                    model="",
                    user_instruction=user_instruction,
                    conversation_messages=conversation_messages,
                    persona=persona,
                    core_profile=core_profile,
                    distiller_model=distiller_model,
                    image_model=image_model,
                    mode=distill_mode,
                    previous_prompts=previous_prompts,
                    group_members=group_members_for_distiller,
                    registry=getattr(self._app_state, "provider_registry", None),
                )
            finally:
                if self._app_state is not None:
                    pre_q = getattr(self._app_state, "scene_image_pre_queue", None)
                    if pre_q:
                        pre_q.pop(self._session_id, None)

            # Track background prompts for deduplication
            if distill_mode == "background" and distilled.get("positive"):
                self._previous_bg_prompts.append(distilled["positive"])
                # Keep only last 5 to bound memory
                self._previous_bg_prompts = self._previous_bg_prompts[-5:]

            # Resolution: explicit override > UI panel > config default
            base_w = overrides.get("width", 0) or ui.get("width", 0) or settings.image_default_width
            base_h = overrides.get("height", 0) or ui.get("height", 0) or settings.image_default_height
            aspect = distilled.get("aspect", "square")
            # Only aspect-adjust if the user didn't set explicit dimensions
            if not overrides.get("width") and not overrides.get("height"):
                if aspect == "portrait":
                    base_w, base_h = base_w, int(base_h * 1.5)
                elif aspect == "landscape":
                    base_w, base_h = int(base_w * 1.5), base_h

            # Check if image_model belongs to a cloud provider
            cloud_provider = None
            if self._app_state:
                from augmentum.proxy.cloud_image_routes import (
                    CloudGenerateRequest,
                    _fetch_cloud_models,
                )
                from augmentum.state.backends.sqlite import SQLiteBackend

                sm = getattr(self._app_state, "state_manager", None)
                conn = None
                if sm and isinstance(sm.backend, SQLiteBackend):
                    conn = sm.backend.conn
                if conn:
                    try:
                        cursor = await conn.execute(
                            "SELECT id, name, base_url, api_key, default_model, default_quality "
                            "FROM image_providers WHERE is_enabled = 1 "
                            "ORDER BY is_default DESC, name"
                        )
                        rows = await cursor.fetchall()
                        for r in rows:
                            prov = {
                                "id": r[0], "name": r[1], "base_url": r[2],
                                "api_key": r[3], "default_model": r[4], "default_quality": r[5],
                            }
                            catalog = await _fetch_cloud_models(prov)
                            if any(cm["name"] == image_model for cm in catalog):
                                cloud_provider = prov
                                break
                    except Exception:
                        log.debug("cloud_provider_lookup_failed", image_model=image_model, exc_info=True)

            if cloud_provider:
                # Route to cloud generation
                from augmentum.proxy.cloud_image_routes import (
                    CloudGenerateRequest,
                    _build_headers,
                    _detect_provider_type,
                    _generate_bfl,
                    _generate_fal,
                    _generate_openai_compat,
                    _generate_stability,
                )
                from augmentum.utils.http_client import normalize_base_url
                cloud_base = normalize_base_url(cloud_provider["base_url"])
                headers = _build_headers(cloud_provider["api_key"], cloud_base)
                ptype = _detect_provider_type(cloud_base)
                quality = cloud_provider.get("default_quality", "standard")

                cloud_req = CloudGenerateRequest(
                    prompt=distilled["positive"],
                    negative_prompt=distilled["negative"],
                    provider_id=cloud_provider["id"],
                    model=image_model,
                    width=base_w,
                    height=base_h,
                    quality=quality,
                    seed=overrides.get("seed", -1),
                )

                if ptype == "stability":
                    resp = await _generate_stability(cloud_base, headers, image_model, cloud_req, quality)
                elif ptype == "bfl":
                    resp = await _generate_bfl(cloud_base, headers, image_model, cloud_req)
                elif ptype == "fal":
                    resp = await _generate_fal(cloud_base, headers, image_model, cloud_req)
                else:
                    resp = await _generate_openai_compat(cloud_base, headers, image_model, cloud_req, quality)

                # Persist to gallery
                from augmentum.proxy.cloud_image_routes import _persist_cloud_generation
                persistence = getattr(self._app_state, "image_persistence", None) if self._app_state else None
                if persistence:
                    await _persist_cloud_generation(persistence, resp)

                import json
                data = json.loads(resp.body)
                return data.get("url", "")

            # Local GPU generation
            if not self._image_queue:
                return None

            raw_steps = overrides.get("steps", 0) or ui.get("steps", 0) or settings.image_default_steps
            raw_cfg = overrides.get("cfg", 0.0) or ui.get("cfg_scale", 0.0) or settings.image_default_cfg
            sampler = overrides.get("sampler", "") or ui.get("sampler", "")
            preset = overrides.get("preset", "") or ui.get("preset", "")
            negative = distilled["negative"] or ui.get("negative_prompt", "")

            # Apply distilled-model-aware defaults (e.g. FLUX uses fewer steps)
            from augmentum.image.distilled import apply_distilled_defaults
            steps, cfg_scale = apply_distilled_defaults(image_model, raw_steps, raw_cfg)

            job = GenerationJob(
                prompt=distilled["positive"],
                negative_prompt=negative,
                model=image_model,
                preset=preset,
                width=base_w,
                height=base_h,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=overrides.get("seed", -1),
                sampler=sampler,
                session_id=self._session_id,
                user_id=self._user_id,
                # ``category`` routes the progress indicator: explicit
                # user clicks → in-message loader, auto_bg → corner
                # badge. Without this both would light up for the same
                # job.
                category=category,
            )

            job = await self._image_queue.submit(job)
            result = await self._image_queue.wait_for_result(
                job, timeout=settings.image_generation_timeout,
            )
            image_id = result["image_id"]
            return f"/api/image/{image_id}"

        except Exception:
            log.warning("narrative_image_generation_failed", exc_info=True)
            return None

    async def _resolve_preset(self):
        """Load the active (default) prompt preset, merging card-level fields as fallbacks.

        Character cards may specify ``post_history_instructions`` and
        ``depth_prompt`` / ``depth_prompt_depth`` (V2 / V2.1 spec).  When the
        user hasn't set those fields in a DB preset we fall back to the card
        values so that imported cards work out-of-the-box.
        """
        from augmentum.modes.narrative.prompt_presets import PromptPreset

        preset = None
        if self._app_state:
            state_mgr = getattr(self._app_state, "state_manager", None)
            if state_mgr:
                from augmentum.state.backends.sqlite import SQLiteBackend
                if isinstance(state_mgr.backend, SQLiteBackend):
                    try:
                        from augmentum.modes.narrative.prompt_presets import PromptPresetStore
                        store = PromptPresetStore(state_mgr.backend.conn)
                        preset = await store.get_default(user_id=self._user_id)
                    except Exception:
                        log.debug("preset_resolve_failed", exc_info=True)

        # Merge card-level injection fields as fallbacks
        card = self._engine.character_card
        if card:
            needs_preset = False
            post_history = ""
            author_note = ""
            author_note_depth = 4

            if card.post_history_instructions and (not preset or not preset.post_history):
                post_history = card.post_history_instructions
                needs_preset = True
            if card.depth_prompt and (not preset or not preset.author_note):
                author_note = card.depth_prompt
                author_note_depth = card.depth_prompt_depth
                needs_preset = True

            if needs_preset:
                if preset:
                    # Fill empty slots from card
                    if not preset.post_history and post_history:
                        preset.post_history = post_history
                    if not preset.author_note and author_note:
                        preset.author_note = author_note
                        preset.author_note_depth = author_note_depth
                else:
                    preset = PromptPreset(
                        name="__card__",
                        post_history=post_history,
                        author_note=author_note,
                        author_note_depth=author_note_depth,
                    )

        return preset

    async def _resolve_regex_scripts(self, character_name: str | None = None):
        """Load regex scripts from the database."""
        if not self._app_state:
            return []
        state_mgr = getattr(self._app_state, "state_manager", None)
        if not state_mgr:
            return []
        from augmentum.state.backends.sqlite import SQLiteBackend
        if not isinstance(state_mgr.backend, SQLiteBackend):
            return []
        try:
            from augmentum.modes.narrative.regex_transformer import RegexScriptStore
            store = RegexScriptStore(state_mgr.backend.conn)
            return await store.list_scripts(
                character_name=character_name, user_id=self._user_id,
            )
        except Exception:
            log.debug("regex_scripts_resolve_failed", exc_info=True)
        return []

    def _apply_macros(self, request: InternalChatRequest) -> InternalChatRequest:
        """Expand macros in all messages."""
        char_name = self._engine.state.character_card_name or "Character"
        # Resolve user/persona name from cached persona or engine state
        user_name = getattr(self, "_cached_persona_name", "") or "User"
        expand_messages(
            request.messages,
            char_name=char_name,
            user_name=user_name,
            message_count=self._engine.state.message_count,
        )
        return request

    async def _cache_persona_name(self) -> None:
        """Cache the persona name for macro expansion (avoids async in _apply_macros)."""
        persona = await self._resolve_user_persona()
        self._cached_persona_name = persona.name if persona and persona.name else "User"

    def _build_request_log(
        self,
        result: object,
        augmented_request: InternalChatRequest,
        preset: object | None,
    ) -> dict:
        """Build a request log entry for the inspector.

        Captures the full context decomposition so users can see exactly
        what was sent to the model (NovelAI-style context viewer).
        """
        from augmentum.utils.tokenizer import count_tokens

        # Context blocks from the builder (already has included/excluded).
        # Full content is kept so the inspector shows exactly what the model saw;
        # the UI side scrolls expanded blocks (max-height: 60vh; overflow-y: auto).
        context_blocks = []
        for bd in result.context.blocks_detail:
            context_blocks.append({
                "label": bd.label,
                "content": bd.content,
                "token_estimate": bd.token_estimate,
                "included": bd.included,
            })

        # Preset blocks (if a preset was applied)
        preset_info: dict = {}
        if preset:
            for field_name in ("system_prompt", "jailbreak", "post_history", "author_note"):
                val = getattr(preset, field_name, "") or ""
                if val.strip():
                    tokens = count_tokens(val)
                    preset_info[field_name] = True
                    context_blocks.append({
                        "label": f"preset:{field_name}",
                        "content": val,
                        "token_estimate": tokens,
                        "included": True,
                    })
                else:
                    preset_info[field_name] = False

        # Message breakdown
        msg_counts: dict[str, int] = {"system": 0, "user": 0, "assistant": 0}
        msg_tokens = 0
        for msg in augmented_request.messages:
            role = msg.role or "user"
            msg_counts[role] = msg_counts.get(role, 0) + 1
            msg_tokens += count_tokens(msg.content or "")

        return {
            "message_index": self._engine.state.message_count,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": augmented_request.model or "",
            "context_blocks": context_blocks,
            "context_tokens_total": result.context.total_tokens_estimate,
            "context_budget": result.context.total_tokens_estimate + result.context.budget_remaining,
            "context_budget_remaining": result.context.budget_remaining,
            "preset_applied": preset_info,
            "message_counts": msg_counts,
            "total_messages": sum(msg_counts.values()),
            "total_token_estimate": msg_tokens,
        }

    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Process a non-streaming narrative request."""
        self._last_model = request.model or ""
        await self._detect_context_length(request.model or "")
        async with self._engine.processing_lock:
            await self._ensure_state_loaded()
            # Ensure session has memory settings (inherit globals on first use)
            if getattr(self._engine.state, "memory_settings", None) is None:
                from augmentum.modes.narrative.memory_settings import SessionMemorySettings
                self._engine.state.memory_settings = SessionMemorySettings.init_from_globals()
            await self._cache_persona_name()
            # World manifest lives in the DB card's extensions; a card
            # parsed from the system prompt text doesn't carry it.
            await self._refresh_card_extensions()

            # Apply macro expansion to incoming messages
            self._apply_macros(request)

            # Apply input regex scripts
            char_name = self._engine.state.character_card_name
            regex_scripts = await self._resolve_regex_scripts(character_name=char_name)
            if regex_scripts:
                from augmentum.modes.narrative.regex_transformer import apply_regex_scripts_safe
                for msg in request.messages:
                    if msg.role == "user" and msg.content:
                        msg.content = await apply_regex_scripts_safe(msg.content, regex_scripts, "input")

            # Check for /v command
            has_v, v_instruction, cleaned_request = extract_v_command(request, fallback_text="Continue the scene.")
            if has_v and self._image_enabled:
                # /v is image-only — generate image without LLM narrative response
                image_url = await self._generate_scene_image(v_instruction, request)
                img_content = f"![Scene]({image_url})" if image_url else "*Image generation failed.*"
                return InternalChatResponse(
                    message=Message(role="assistant", content=img_content),
                    model=request.model,
                )

            # Group chat: sync group_id from request (UI carries it in the
            # X-Augmentum-Group-Id header). If it's new or changed, rebuild
            # the turn manager and member-cards cache.
            req_group_id = getattr(request, "group_id", "") or ""
            if req_group_id and req_group_id != self._engine.state.group_id:
                self._engine.state.group_id = req_group_id
                self._group_turn_manager = None  # force reload for new group
            if self._engine.state.group_id and not self._group_turn_manager:
                await self._load_group_context()

            # Group chat: refresh member cards every turn so edits go live.
            if self._group_turn_manager and self._active_group:
                await self._refresh_group_member_cards()
                # Apply override/llm_decide BEFORE reading current_speaker
                # so the selection reflects this turn's intent.
                await self._resolve_group_speaker(request)

            # Group chat: determine speaker and swap prompt
            group_speaker = None
            if self._group_turn_manager and self._active_group:
                group_speaker = self._group_turn_manager.current_speaker
                log.info("group_chat_turn", speaker=group_speaker,
                         index=self._group_turn_manager._current_index,
                         mode=self._active_group.generation_mode)
                # Load speaker's card for system prompt swap. Fall back to
                # the refreshed member cache if the DB lookup misses (name
                # collision, case drift, transient read). Without the fallback
                # the prompt swap silently skips and the model speaks as
                # whoever was last set — worst-case "wrong character" output.
                self._group_speaker_card = await self._load_character_card_by_name(group_speaker)
                if not self._group_speaker_card and self._group_member_cards:
                    cached = self._group_member_cards.get(group_speaker)
                    if cached:
                        self._group_speaker_card = cached
                        log.warning("group_speaker_card_db_miss_cache_hit",
                                    speaker=group_speaker)
                if self._group_speaker_card:
                    self._apply_group_to_request(request)
                else:
                    log.warning("group_speaker_card_not_found",
                                speaker=group_speaker,
                                cached_members=list((self._group_member_cards or {}).keys()))

            # Retrieve relevant archived exchanges for context
            retrieved_archive = await self._retrieve_archive_for_request(request)

            # Sync archive pointer to engine so branch save captures it
            cur_branch = self._engine._branch_tracker.current_branch
            self._engine._branch_archive_idx[cur_branch] = self._last_archived_history_idx

            # Phase 3 chunk 3: pre-fetch a STATE snapshot from history if a
            # branch is imminent. Closes the empty-STATE-on-first-turn-after-
            # branch hole. Read-only peek — process_request still owns the
            # actual branch application.
            await self._prefetch_branch_state_snapshot(request)

            # Refresh lorebook entries from UI before the pipeline scans them
            self._sync_ui_lorebook(request)

            # Run through narrative pipeline. ``supports_mid_system``
            # selects the dynamic-context placement strategy in the engine
            # — see ``NarrativeEngine._augment_request`` for the rationale.
            # Offload to thread for the same reason as the streaming path:
            # process_request is sync + CPU-bound (lorebook regex scan +
            # N×count_tokens for budget enforcement) and would otherwise
            # block the event loop for the duration of the pipeline pass.
            ctx_limit = self._effective_context_limit()
            supports_mid_system = getattr(
                self._backend, "supports_mid_conversation_system", False
            )
            result = await asyncio.to_thread(
                self._engine.process_request,
                request,
                retrieved_archive=retrieved_archive,
                context_limit=ctx_limit,
                supports_mid_system=supports_mid_system,
            )

            # If a branch was detected, prune vectorized archive entries
            # from the abandoned path so retrieval doesn't surface stale context.
            if result.branch_detected:
                await self._prune_archive_for_branch()
                # Restore archive pointer from the branch we switched to (if saved),
                # otherwise clamp to current history length.
                new_branch = self._engine._branch_tracker.current_branch
                restored_idx = self._engine._branch_archive_idx.get(new_branch)
                hist_len = len(self._engine._message_history)
                if restored_idx is not None:
                    self._last_archived_history_idx = min(restored_idx, hist_len)
                elif self._last_archived_history_idx > hist_len:
                    self._last_archived_history_idx = hist_len

                # Sync the cached turn manager to whatever the restored branch
                # said about group rotation. Without this, the current turn
                # already streams as the OLD branch's speaker (system prompt
                # was swapped before process_request ran), but at least the
                # NEXT turn picks up from the restored index instead of
                # continuing the abandoned timeline's rotation.
                if self._group_turn_manager and self._engine.state.group_id:
                    self._group_turn_manager._current_index = (
                        self._engine.state.group_speaker_index
                    )

            # Apply prompt preset (jailbreak, author's note, post-history)
            preset = await self._resolve_preset()
            if preset:
                from augmentum.modes.narrative.macro_expander import expand_messages
                from augmentum.modes.narrative.prompt_presets import apply_preset
                augmented = apply_preset(result.augmented_request, preset)
                # Expand macros in the injected preset fields
                expand_messages(
                    augmented.messages,
                    char_name=char_name or "Character",
                    message_count=self._engine.state.message_count,
                )
                # The checkpoint snapshot carries the same preset system
                # prompt (apply_preset prepends it to both lists) — it
                # must get the SAME macro expansion or the prewarmed
                # prefix diverges from the real payload at the first
                # macro and the checkpoint is unmatchable.
                if augmented.kv_stable_messages:
                    expand_messages(
                        augmented.kv_stable_messages,
                        char_name=char_name or "Character",
                        message_count=self._engine.state.message_count,
                    )
                result = result.__class__(
                    augmented_request=augmented,
                    context=result.context,
                    state=result.state,
                    contradictions=result.contradictions,
                    new_facts=result.new_facts,
                    branch_detected=result.branch_detected,
                    is_regeneration=result.is_regeneration,
                )

            # Refine the engine's approximate trim with the model's real tokenizer.
            result = await self._refine_augmented_request(result, request, ctx_limit)

            # Capture request log for inspector
            self._engine.add_request_log(
                self._build_request_log(result, result.augmented_request, preset)
            )

            # Send augmented request to backend
            response = await self._backend.chat(result.augmented_request)

            # Apply output regex scripts
            response_text = ""
            if response.message and response.message.content:
                response_text = response.message.content

                # Group chat: clean up response (strip other-character dialogue)
                if group_speaker:
                    response_text = self._postprocess_group_response(response_text, group_speaker)
                    response.message.content = response_text

                if regex_scripts:
                    response.message.content = await apply_regex_scripts_safe(
                        response.message.content, regex_scripts, "output"
                    )
                    response_text = response.message.content
                self._engine.process_response(response_text)
                self._schedule_checkpoint(result.augmented_request, response_text)
            else:
                # No valid response — undo the user message that process_request
                # appended to maintain the alternating user/assistant pattern.
                self._engine.undo_last_request()

            # Group chat: advance turn only for a real new turn. On regen the
            # index would double-advance, skipping a member: Alice→Bob, regen
            # Alice→Carol, and Bob never speaks.
            if group_speaker and not result.is_regeneration:
                self._advance_group_turn()

            if not result.is_regeneration and response_text:
                # Background tasks (ordered: refresh → archive → compact → extract → bg)
                self._maybe_llm_extract(response_text)
                self._maybe_world_reconcile(response_text)

                # Refresh state+memory if needed (non-blocking background task)
                self._maybe_refresh_summary()

                # Continuous archiving (every exchange)
                self._maybe_archive_exchange(response_text)

                # KG is now derived from STATE+MEMORY during _refresh_state_memory
                # (no per-turn LLM extraction needed)

                # Ledger compaction (after refresh, if ceiling reached)
                self._maybe_compact_ledger()

                # Auto-generate scene background if interval reached
                self._maybe_generate_background(request)
            elif result.is_regeneration:
                log.info(
                    "narrative_regen_skipped_background_tasks",
                    session_id=self._session_id,
                    message_count=self._engine.state.message_count,
                )
                # Prune any archive entry that was stored for the discarded
                # generation so future retrievals don't surface stale content.
                await self._prune_archive_for_regen()

            # Auto-persist state
            await self._persist_state()

            return response

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Process a streaming narrative request."""
        self._last_model = request.model or ""
        await self._detect_context_length(request.model or "")
        # Yielding while holding processing_lock keeps the lock alive across
        # the consumer-side suspension and stalls every coroutine waiting on
        # this engine — including the next request from the same client.
        # Pre-stream chunks are buffered and emitted after the lock releases.
        pre_stream_chunks: list[InternalStreamChunk] = []
        early_return = False
        group_speaker: str | None = None
        regex_scripts: list = []
        char_name = ""
        result = None
        augmented_for_stream = None
        async with self._engine.processing_lock:
            await self._ensure_state_loaded()
            # Ensure session has memory settings (inherit globals on first use)
            if getattr(self._engine.state, "memory_settings", None) is None:
                from augmentum.modes.narrative.memory_settings import SessionMemorySettings
                self._engine.state.memory_settings = SessionMemorySettings.init_from_globals()
            await self._cache_persona_name()
            # World manifest lives in the DB card's extensions; a card
            # parsed from the system prompt text doesn't carry it.
            await self._refresh_card_extensions()

            # Apply macro expansion to incoming messages
            self._apply_macros(request)

            # Apply input regex scripts
            char_name = self._engine.state.character_card_name
            regex_scripts = await self._resolve_regex_scripts(character_name=char_name)
            if regex_scripts:
                from augmentum.modes.narrative.regex_transformer import apply_regex_scripts_safe
                for msg in request.messages:
                    if msg.role == "user" and msg.content:
                        msg.content = await apply_regex_scripts_safe(msg.content, regex_scripts, "input")

            # Check for /v command
            has_v, v_instruction, cleaned_request = extract_v_command(request, fallback_text="Continue the scene.")
            if has_v and self._image_enabled:
                # /v is image-only — generate image without LLM narrative response
                image_url = await self._generate_scene_image(v_instruction, request)
                if image_url:
                    pre_stream_chunks.append(
                        InternalStreamChunk(content_delta=f"![Scene]({image_url})"),
                    )
                else:
                    pre_stream_chunks.append(
                        InternalStreamChunk(content_delta="*Image generation failed.*"),
                    )
                early_return = True

            # Sheet commands (/s /status /inv /loc) — tier-1 intercept.
            # Renders from the tracker store WITHOUT a model call (spec D2:
            # zero tokens, cannot be hallucinated). Only when the card
            # declares a sheet; plain cards fall through to normal chat so
            # a user typing "/status" at a non-manifest card still gets a
            # story reply, not a dead command.
            if not early_return:
                _wm = self._world_manifest()
                if _wm is not None and _wm.has("sheet"):
                    from augmentum.modes.narrative.world_system import (
                        match_sheet_command,
                        sheet_text,
                    )
                    _last_user = next(
                        (m.content for m in reversed(request.messages)
                         if m.role == "user" and m.content), "",
                    )
                    _section = match_sheet_command(_last_user)
                    if _section is not None:
                        _sheet = self._world_store(_wm).sheet()
                        if _section:
                            _sheet["sections"] = [
                                s for s in _sheet["sections"]
                                if s["id"] == _section
                            ] or _sheet["sections"]
                        pre_stream_chunks.append(InternalStreamChunk(
                            content_delta="```\n" + sheet_text(_sheet) + "\n```",
                            augmentum={
                                "world_events": [
                                    {"kind": "sheet", "sheet": _sheet},
                                ],
                            },
                        ))
                        early_return = True

            if not early_return:
                # Group chat: sync group_id from request (UI carries it in the
                # X-Augmentum-Group-Id header). If it's new or changed, rebuild
                # the turn manager and member-cards cache.
                req_group_id = getattr(request, "group_id", "") or ""
                if req_group_id and req_group_id != self._engine.state.group_id:
                    self._engine.state.group_id = req_group_id
                    self._group_turn_manager = None  # force reload for new group
                if self._engine.state.group_id and not self._group_turn_manager:
                    await self._load_group_context()

                # Group chat: refresh member cards every turn so edits go live,
                # then apply override/llm_decide BEFORE reading current_speaker.
                if self._group_turn_manager and self._active_group:
                    await self._refresh_group_member_cards()
                    await self._resolve_group_speaker(request)

                # Group chat: determine speaker and swap prompt
                if self._group_turn_manager and self._active_group:
                    group_speaker = self._group_turn_manager.current_speaker
                    log.info("group_chat_turn", speaker=group_speaker,
                             index=self._group_turn_manager._current_index,
                             mode=self._active_group.generation_mode)
                    # Load speaker's card with cache-fallback so a DB miss
                    # (rename, case drift, transient read) doesn't silently
                    # leave the prompt unchanged.
                    self._group_speaker_card = await self._load_character_card_by_name(group_speaker)
                    if not self._group_speaker_card and self._group_member_cards:
                        cached = self._group_member_cards.get(group_speaker)
                        if cached:
                            self._group_speaker_card = cached
                            log.warning("group_speaker_card_db_miss_cache_hit",
                                        speaker=group_speaker)
                    if self._group_speaker_card:
                        self._apply_group_to_request(request)
                    else:
                        log.warning("group_speaker_card_not_found",
                                    speaker=group_speaker,
                                    cached_members=list((self._group_member_cards or {}).keys()))

                    # Buffer an EARLY speaker-selected meta chunk with the
                    # speaker's voice. This lets the voice pipeline switch
                    # session.character_voice BEFORE the first TTS sentence,
                    # so each turn in a group voice call uses the actual
                    # responding character's voice instead of whatever was
                    # set at call start. The chat UI also gets an earlier
                    # signal to swap the avatar viewport (instead of waiting
                    # for the done=True meta at the end of the stream).
                    if group_speaker:
                        pre_stream_chunks.append(InternalStreamChunk(
                            content_delta="",
                            augmentum={
                                "group_speaker": group_speaker,
                                "group_speaker_voice": (self._group_speaker_card or {}).get("voice", ""),
                            },
                        ))

                # Retrieve relevant archived exchanges for context
                retrieved_archive = await self._retrieve_archive_for_request(request)

                # Sync archive pointer to engine so branch save captures it
                cur_branch = self._engine._branch_tracker.current_branch
                self._engine._branch_archive_idx[cur_branch] = self._last_archived_history_idx

                # Phase 3 chunk 3: pre-fetch a STATE snapshot from history if a
                # branch is imminent. See the non-streaming call site for details.
                await self._prefetch_branch_state_snapshot(request)

                # Refresh lorebook entries from UI before the pipeline scans them
                self._sync_ui_lorebook(request)

                # Run through narrative pipeline (streaming path). See the
                # non-streaming call site for ``supports_mid_system`` rationale.
                ctx_limit = self._effective_context_limit()
                supports_mid_system = getattr(
                    self._backend, "supports_mid_conversation_system", False
                )
                # process_request is sync and CPU-bound for long narrative chats:
                # lorebook regex scan + N×count_tokens for budget enforcement +
                # context building. Hundreds of ms to ~1-2s on a long history.
                # Running it directly on the event loop blocks every other
                # coroutine (auth, sync, healthchecks) for the duration — that
                # was the "narrative wedges all clients" symptom. The
                # processing_lock around this block keeps engine-state access
                # safe across the to_thread hop (no other coroutine on this
                # engine can race; the worker thread is the sole writer).
                result = await asyncio.to_thread(
                    self._engine.process_request,
                    request,
                    retrieved_archive=retrieved_archive,
                    context_limit=ctx_limit,
                    supports_mid_system=supports_mid_system,
                )

                # If a branch was detected, prune vectorized archive entries
                if result.branch_detected:
                    await self._prune_archive_for_branch()
                    new_branch = self._engine._branch_tracker.current_branch
                    restored_idx = self._engine._branch_archive_idx.get(new_branch)
                    hist_len = len(self._engine._message_history)
                    if restored_idx is not None:
                        self._last_archived_history_idx = min(restored_idx, hist_len)
                    elif self._last_archived_history_idx > hist_len:
                        self._last_archived_history_idx = hist_len

                    # Sync cached turn manager to restored branch state — see
                    # the non-streaming path for the full rationale.
                    if self._group_turn_manager and self._engine.state.group_id:
                        self._group_turn_manager._current_index = (
                            self._engine.state.group_speaker_index
                        )

                # Apply prompt preset (jailbreak, author's note, post-history)
                preset = await self._resolve_preset()
                if preset:
                    from augmentum.modes.narrative.macro_expander import expand_messages
                    from augmentum.modes.narrative.prompt_presets import apply_preset
                    augmented = apply_preset(result.augmented_request, preset)
                    expand_messages(
                        augmented.messages,
                        char_name=char_name or "Character",
                        message_count=self._engine.state.message_count,
                    )
                    # Mirror of the non-streaming path's checkpoint expansion
                    # (see handle()): the checkpoint snapshot must get the
                    # SAME macro expansion as the live messages or the
                    # prewarmed prefix diverges at the first macro and the
                    # checkpoint is unmatchable. This path missing that block
                    # was the root cause of every narrative turn logging
                    # kv_tier=cold_no_checkpoint (prewarm baseline violated
                    # at message 0 char 44 — literal {{user}}/{{char}} in the
                    # checkpoint vs expanded names in the live payload).
                    if augmented.kv_stable_messages:
                        expand_messages(
                            augmented.kv_stable_messages,
                            char_name=char_name or "Character",
                            message_count=self._engine.state.message_count,
                        )
                    result = result.__class__(
                        augmented_request=augmented,
                        context=result.context,
                        state=result.state,
                        contradictions=result.contradictions,
                        new_facts=result.new_facts,
                        branch_detected=result.branch_detected,
                        is_regeneration=result.is_regeneration,
                    )

                # Refine the engine's approximate trim with the model's real tokenizer.
                result = await self._refine_augmented_request(result, request, ctx_limit)

                # Capture request log for inspector
                self._engine.add_request_log(
                    self._build_request_log(result, result.augmented_request, preset)
                )
                # End of pre-stream phase — close the lock here. Holding it across
                # the backend stream below blocks every other coroutine on this
                # engine (concurrent same-session reads, regen requests, persist)
                # for the entire generation window. The streaming await is
                # backend-bound; engine state isn't mutated until process_response
                # below, which re-acquires the lock.
                augmented_for_stream = result.augmented_request

        # Emit any pre-stream chunks accumulated under the lock. Doing this
        # AFTER releasing the lock avoids the consumer-suspension-holds-lock
        # stall that wedges every other coroutine waiting on this engine.
        for chunk in pre_stream_chunks:
            yield chunk
        if early_return:
            return

        # ---- STREAM phase (NO lock) -----------------------------------------
        # Backend stream can run for tens of seconds; releasing the lock here
        # is what stops a chat from wedging concurrent same-session requests.
        collected_content: list[str] = []
        # Recall-tools wrap: when ``narrative_recall_tools_enabled`` is on
        # AND we have a persistence handle, swap the bare backend stream
        # for the tool-execution loop (``recall_loop.stream_with_recall_tools``).
        # The loop attaches the recall schemas to the request and dispatches
        # ``recall_*`` tool_calls against the live narrative store. Feature
        # flag default is False so existing sessions are unchanged until
        # the user opts in.
        from augmentum.config import settings as _cfg
        # Public accessor on NarrativeEngine — beats reaching into the
        # underscore-prefixed ``_persistence`` attribute across module
        # boundaries. Returns None when no persistence is attached
        # (no-DB test contexts), in which case the recall loop is
        # skipped via the guard below.
        _engine_persistence = self._engine.persistence
        _lorebook_mutations: list[dict] = []
        _has_recall = (
            _cfg.narrative_recall_tools_enabled
            and _engine_persistence is not None
            and self._user_id
            and self._session_id
        )
        _has_lorebook_native = (
            _cfg.narrative_lorebook_native_tools_enabled
            and self._session_id
            and self._user_id
        )
        # Underscore (legacy) family only supplements when native is the
        # chosen surface but unavailable — it must NEVER resurrect when the
        # user disabled lorebook tools. The UI exposes a single "Lorebook
        # tools" toggle wired to the NATIVE key, so that key is the master
        # switch: with it OFF, both surfaces stay off. Without this gate,
        # disabling the only visible control silently activated the legacy
        # tools whenever ``narrative_lorebook_tools_enabled`` lingered True
        # in the store (its default is False, but old installs/API writes
        # can leave it set). Fix-the-class: gate legacy on the master.
        _has_lorebook = (
            _cfg.narrative_lorebook_tools_enabled
            and _cfg.narrative_lorebook_native_tools_enabled
            and not _has_lorebook_native
            and self._session_id
        )
        # World-system tools — gated on the card's manifest (None for the
        # 99% of cards without one; the feature is absent, not disabled).
        _world_manifest_obj = self._world_manifest()
        _has_world = _world_manifest_obj is not None
        _world_events: list[dict] = []
        _world_store_obj = (
            self._world_store(_world_manifest_obj) if _has_world else None
        )

        # [World State] line block — engine-authoritative tracker values,
        # inserted as a system line just before the latest user turn (same
        # cache-friendly placement as the STATE injection; deliberately NOT
        # in kv_stable_messages, so it's per-turn suffix cost only, ~30-60
        # tokens). The model narrates FROM these values (spec D1).
        if _has_world and augmented_for_stream is not None:
            _ws_block = _world_store_obj.state_block()
            if _ws_block:
                _msgs = augmented_for_stream.messages
                _last_user_idx = next(
                    (i for i in range(len(_msgs) - 1, -1, -1)
                     if _msgs[i].role == "user"), None,
                )
                _ws_msg = Message(role="system", content=_ws_block)
                if _last_user_idx is not None:
                    _msgs.insert(_last_user_idx, _ws_msg)
                else:
                    _msgs.append(_ws_msg)
        # _has_lorebook_native already computed above (before _has_lorebook,
        # since the underscore family defers to native when both are on).
        if _has_recall or _has_lorebook or _has_lorebook_native or _has_world:
            from augmentum.modes.narrative.internal_tools import (
                append_conduct_directive,
                with_silent_suffix,
            )
            from augmentum.modes.narrative.recall_loop import (
                stream_with_recall_tools,
            )

            existing_tools = list(augmented_for_stream.tools or [])
            all_internal_names: set[str] = set()
            # Mutating internal tools — the recall loop's pre-call gate
            # flushes held prose before a WRITE (story that established the
            # fact) and drops it before a READ (plan-narration preamble).
            all_write_names: set[str] = set()

            if _has_recall:
                from augmentum.modes.narrative.recall_schemas import (
                    RECALL_TOOL_NAMES,
                    RECALL_TOOL_SCHEMAS,
                    dispatch_recall_tool,
                )
                existing_tools += with_silent_suffix(RECALL_TOOL_SCHEMAS)
                all_internal_names |= RECALL_TOOL_NAMES

            if _has_lorebook:
                from augmentum.modes.narrative.lorebook_schemas import (
                    LOREBOOK_MUTATING_TOOLS,
                    LOREBOOK_TOOL_NAMES,
                    LOREBOOK_TOOL_SCHEMAS,
                    dispatch_lorebook_tool,
                )
                existing_tools += with_silent_suffix(LOREBOOK_TOOL_SCHEMAS)
                all_internal_names |= LOREBOOK_TOOL_NAMES
                all_write_names |= LOREBOOK_MUTATING_TOOLS

            if _has_lorebook_native:
                from augmentum.modes.narrative.lorebook_native_schemas import (
                    LOREBOOK_NATIVE_MUTATING_TOOLS,
                    LOREBOOK_NATIVE_TOOL_NAMES,
                    LOREBOOK_NATIVE_TOOL_SCHEMAS,
                    dispatch_lorebook_native_tool,
                )
                existing_tools += with_silent_suffix(LOREBOOK_NATIVE_TOOL_SCHEMAS)
                all_internal_names |= LOREBOOK_NATIVE_TOOL_NAMES
                all_write_names |= LOREBOOK_NATIVE_MUTATING_TOOLS

            if _has_world:
                from augmentum.modes.narrative.world_native_schemas import (
                    WORLD_NATIVE_MUTATING_TOOLS,
                    WORLD_NATIVE_TOOL_NAMES,
                    dispatch_world_native_tool,
                    schemas_for_manifest,
                )
                _world_schemas = schemas_for_manifest(_world_manifest_obj)
                existing_tools += with_silent_suffix(_world_schemas)
                all_internal_names |= {
                    s["function"]["name"] for s in _world_schemas
                }
                all_write_names |= WORLD_NATIVE_MUTATING_TOOLS

            augmented_for_stream.tools = existing_tools
            # The conduct contract — tools are silent bookkeeping, call
            # before prose, never referenced in the visible story. Without
            # it, agentic-register models (DeepSeek) announce their tool
            # plans as story content (2026-07-15 live report). The
            # mechanical backstop is recall_loop's pre-call gate.
            append_conduct_directive(augmented_for_stream)

            _persistence = _engine_persistence
            _sid = self._session_id
            _uid = self._user_id
            _lore_engine = self._engine._lore_engine
            # Current branch — tags model-authored lore (migration 304) so
            # branch retrieval scopes it. ``state.branch_id`` is the live
            # engine branch (defaults to "main").
            _branch_id = getattr(self._engine.state, "branch_id", "main") or "main"

            async def _combined_dispatcher(name: str, raw_args):
                if _has_recall and name in RECALL_TOOL_NAMES:
                    return await dispatch_recall_tool(
                        _persistence, _sid,
                        user_id=_uid,
                        tool_name=name,
                        raw_arguments=raw_args,
                    )
                if _has_lorebook_native and name in LOREBOOK_NATIVE_TOOL_NAMES:
                    result_text, mutations = dispatch_lorebook_native_tool(
                        _lore_engine, _sid,
                        user_id=_uid,
                        branch_id=_branch_id,
                        tool_name=name,
                        raw_arguments=raw_args,
                    )
                    if mutations:
                        _lorebook_mutations.extend(mutations)
                    return result_text
                if _has_lorebook and name in LOREBOOK_TOOL_NAMES:
                    result_text, mutations = dispatch_lorebook_tool(
                        _lore_engine, _sid,
                        tool_name=name,
                        raw_arguments=raw_args,
                    )
                    if mutations:
                        _lorebook_mutations.extend(mutations)
                    return result_text
                if _has_world and name in WORLD_NATIVE_TOOL_NAMES:
                    result_text, w_events = dispatch_world_native_tool(
                        _world_store_obj,
                        turn=self._engine.state.message_count,
                        tool_name=name,
                        raw_arguments=raw_args,
                    )
                    if w_events:
                        _world_events.extend(w_events)
                    return result_text
                return f"Unknown tool '{name}'."

            stream_source = stream_with_recall_tools(
                augmented_for_stream,
                backend=self._backend,
                dispatcher=_combined_dispatcher,
                max_iters=int(_cfg.narrative_recall_tools_max_iters or 3),
                internal_tool_names=frozenset(all_internal_names),
                internal_write_names=frozenset(all_write_names),
            )
        else:
            stream_source = self._backend.chat_stream(augmented_for_stream)

        async for chunk in stream_source:
            yield chunk
            if chunk.content_delta:
                collected_content.append(chunk.content_delta)

        if _lorebook_mutations:
            yield InternalStreamChunk(
                content_delta="",
                augmentum={
                    "status": "lorebook_mutations",
                    "mutations": _lorebook_mutations,
                },
            )

        if _world_events:
            # UI sync: inline event cards (rolls, tracker shifts) + drawer
            # refresh. Persisted on the message node client-side so cards
            # survive reload (same pattern as generation stats).
            yield InternalStreamChunk(
                content_delta="",
                augmentum={"world_events": _world_events},
            )

        full_response = "".join(collected_content)

        # Group chat: clean up response (pure string transform, lock-free)
        if group_speaker and full_response:
            full_response = self._postprocess_group_response(full_response, group_speaker)

        # Apply output regex scripts (regex transform + yield, no engine state)
        if full_response and regex_scripts:
            transformed = await apply_regex_scripts_safe(full_response, regex_scripts, "output")
            if transformed != full_response:
                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={"regex_transformed": transformed},
                )
                full_response = transformed

        # ---- POST-STREAM phase (re-acquire lock for state mutation) ---------
        # Same buffering pattern as pre-stream: collect chunks under the lock,
        # emit them after release.
        post_stream_chunks: list[InternalStreamChunk] = []
        async with self._engine.processing_lock:
            if full_response:
                self._engine.process_response(full_response)
                self._schedule_checkpoint(augmented_for_stream, full_response)
            else:
                # No valid response — undo the user message that process_request
                # appended to maintain the alternating user/assistant pattern.
                self._engine.undo_last_request()

            # Group chat: advance turn only for real new turns (not regens —
            # see non-streaming handler for rationale). Metadata is still
            # emitted so the UI tags the new node with the speaker.
            if group_speaker:
                if not result.is_regeneration:
                    self._advance_group_turn()
                post_stream_chunks.append(InternalStreamChunk(
                    content_delta="",
                    done=True,
                    augmentum={
                        "group_turn": self._group_turn_manager.to_dict() if self._group_turn_manager else {},
                        "group_speaker": group_speaker,
                    },
                ))

            if not result.is_regeneration and full_response:
                # Background tasks (ordered: refresh → archive → compact → extract → bg)
                self._maybe_llm_extract(full_response)
                self._maybe_world_reconcile(full_response)

                # Refresh state+memory if needed (non-blocking background task)
                self._maybe_refresh_summary()

                # Continuous archiving (every exchange)
                self._maybe_archive_exchange(full_response)

                # KG derived from STATE+MEMORY during _refresh_state_memory

                # Ledger compaction (after refresh, if ceiling reached)
                self._maybe_compact_ledger()

                # Auto-generate scene background if interval reached
                self._maybe_generate_background(request)
            elif result.is_regeneration:
                log.info(
                    "narrative_regen_skipped_background_tasks",
                    session_id=self._session_id,
                    message_count=self._engine.state.message_count,
                )
                # Prune any archive entry that was stored for the discarded
                # generation so future retrievals don't surface stale content.
                await self._prune_archive_for_regen()

            # Auto-persist state
            await self._persist_state()

        # Lock released — emit any post-stream chunks (group-chat done=True).
        for chunk in post_stream_chunks:
            yield chunk
