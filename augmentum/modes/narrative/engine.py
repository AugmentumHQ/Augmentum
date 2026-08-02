"""Narrative engine — core orchestrator for narrative mode processing.

Processing pipeline per message:
1. Parse incoming message → detect character card on first message
2. Track message in DAG (branch detection)
3. Extract state changes (characters, world, plots)
4. Run consistency checks
5. Trigger lorebook entries
6. Build enhanced context (inject state into system prompt)
7. Forward to model backend
8. Parse response → extract state changes from AI output
9. Update all trackers with AI response
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.narrative.branch_tracker import BranchTracker
from augmentum.modes.narrative.card_parser import CardParser, CharacterCard
from augmentum.modes.narrative.character_tracker import CharacterTracker, CharacterUpdate
from augmentum.modes.narrative.context_builder import BuiltContext, ContextBuilder
from augmentum.modes.narrative.llm_extractor import NarrativeExtraction
from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.modes.narrative.memory import (
    CardType,
    MemoryEntry,
    StateSnapshot,
    SummaryMode,
    build_state_memory_prompt,
    detect_card_type,
    format_ledger_for_context,
    format_state_for_context,
)
from augmentum.modes.narrative.plot_tracker import PlotTracker
from augmentum.modes.narrative.relationship_tracker import RelationshipTracker
from augmentum.modes.narrative.world_tracker import SceneState, WorldTracker
from augmentum.state.narrative_state import (
    Contradiction,
    Entity,
    EntityState,
    EntityType,
    Fact,
    NarrativeSessionState,
    _new_id,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class NarrativeResult:
    """Result of narrative processing for a single message."""

    augmented_request: InternalChatRequest
    context: BuiltContext
    state: NarrativeSessionState
    contradictions: list[Contradiction] = field(default_factory=list)
    new_facts: list[Fact] = field(default_factory=list)
    branch_detected: bool = False
    is_regeneration: bool = False


class NarrativeEngine:
    """Core narrative engine — orchestrates all narrative processing."""

    def __init__(
        self,
        session_id: str = "",
        context_budget: int = 4000,
        character_pct: float = 0.25,
        scene_pct: float = 0.15,
        plot_pct: float = 0.15,
        lore_pct: float = 0.25,
        consistency_pct: float = 0.10,
    ) -> None:
        self._session_id = session_id
        self._state = NarrativeSessionState(session_id=session_id)
        self._card_parser = CardParser()
        self._character_tracker = CharacterTracker()
        self._world_tracker = WorldTracker()
        self._plot_tracker = PlotTracker()
        self._context_builder = ContextBuilder(
            token_budget=context_budget,
            character_pct=character_pct,
            scene_pct=scene_pct,
            plot_pct=plot_pct,
            lore_pct=lore_pct,
            consistency_pct=consistency_pct,
        )
        self._lore_engine = LoreEngine()
        self._relationship_tracker = RelationshipTracker()
        self._branch_tracker = BranchTracker(session_id)
        self._character_card: CharacterCard | None = None
        self._initialized = False
        self._message_history: list[str] = []
        self._message_summaries: list[str] = []
        self._request_logs: list[dict] = []
        self.processing_lock = asyncio.Lock()
        self._state_snapshot: StateSnapshot | None = None
        self._memory_ledger: list[MemoryEntry] = []
        self._refresh_ran_this_session: bool = False  # True after first apply_state_memory_response
        # Set by the handler when a user-configured refresh model no longer
        # resolves (stale card/setting). Surfaced by the narrative panel poll
        # as an actionable toast; cleared once the model resolves again or the
        # user picks a resolution. Not persisted — it's a live-runtime signal.
        self.pending_model_alert: dict | None = None
        self._needs_compaction: bool = False
        # Snapshot of ledger length taken just before each refresh batch is applied.
        # Used to roll back entries added by a generation the user discarded (regen).
        self._pre_refresh_ledger_len: int = 0
        # Flag set by process_request on regen so process_response replaces
        # the last history entry instead of appending a second response.
        self._pending_regen: bool = False
        # Per-branch state cache — lets users swap between branches
        # without losing accumulated state/memory for each path.
        # Key: branch_id, Value: saved state dict
        self._branch_states: dict[str, dict] = {}
        # Per-branch archive pointer — managed by handler, shuttled by engine.
        # Key: branch_id, Value: int (archive history index)
        self._branch_archive_idx: dict[str, int] = {}

        # Branch-tagged persistence (Phase 3 chunk 1: shadow writes).
        # Wired by handler.attach_persistence() at request time. While unset,
        # the engine behaves exactly as before — no SQL writes to the new
        # tables. Once set, every state/ledger refresh + branch creation also
        # writes a row to the corresponding migration-115/116/117 table so
        # later chunks can flip read paths without losing data.
        self._persistence = None  # type: NarrativePersistence | None
        self._persist_user_id: str = ""
        # Strong refs to in-flight shadow writes — keep them alive until the
        # event loop runs them; each task removes itself on completion.
        self._shadow_persist_tasks: set[asyncio.Task] = set()
        # Loop reference captured at attach time so shadow-persist coroutines
        # scheduled from a worker thread (e.g. process_request running under
        # asyncio.to_thread) can submit back to the main loop via
        # run_coroutine_threadsafe instead of silently dropping.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Phase 3 chunk 3: branch snapshot recovery. Handler pre-fetches the
        # most recent snapshot from history (async) when it detects a branch
        # is imminent and stashes it here. rollback_to consults this slot
        # and, if populated, restores STATE instead of wiping it to None.
        # Cleared after each consumption so a stale snapshot can't leak into
        # a subsequent unrelated rollback.
        self._pending_branch_snapshot: dict | None = None

    def _mem_setting(self, key: str):
        """Resolve a narrative memory setting: session override -> global fallback."""
        from augmentum.modes.narrative.memory_settings import resolve_memory_setting
        session_settings = getattr(self.state, "memory_settings", None)
        return resolve_memory_setting(session_settings, key)

    # ------------------------------------------------------------------
    # Branch-tagged persistence (Phase 3 chunk 1: shadow writes)
    # ------------------------------------------------------------------

    @property
    def persistence(self):
        """Public accessor for the attached NarrativePersistence (or None).

        Cross-module callers (e.g. the recall-tools wiring in
        ``handler.py``) should read this instead of the underscore-
        prefixed ``_persistence`` attribute so the coupling survives a
        future rename. Returns None on engines that haven't been
        attached yet — call ``attach_persistence`` first or guard the
        consumer.
        """
        return self._persistence

    def attach_persistence(self, persistence, user_id: str) -> None:
        """Wire a NarrativePersistence + user_id for shadow writes to the
        branch-tagged tables (narrative_branches, narrative_state_snapshots,
        narrative_ledger_entries). Called by the handler at request time.

        While unattached, the engine writes only to its in-memory structures
        and the legacy JSON columns — identical to pre-Phase-3 behavior.

        Also seeds a 'main' branch row for sessions that don't yet have one
        (new sessions created after Phase 3 ships, plus any that migration 119
        couldn't backfill). Idempotent via INSERT OR IGNORE.
        """
        self._persistence = persistence
        self._persist_user_id = user_id
        # Capture the running loop so worker-thread shadow scheduling has a
        # target. Falls back to None in sync test contexts where attach is
        # called outside any loop.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if user_id:
            self._schedule_shadow_persist(
                self._shadow_upsert_branch("main", None, 0),
            )

    def _schedule_shadow_persist(self, coro) -> None:
        """Fire-and-forget a persistence coroutine.

        Three call contexts:
          1. Main event loop — schedule with create_task and track a strong ref.
          2. Worker thread (process_request under asyncio.to_thread) — submit
             via run_coroutine_threadsafe against the captured loop. The loop
             holds the strong ref, so we don't track here.
          3. No loop reachable (sync test paths) — close the coroutine.
        """
        # Case 1: a loop is running on this thread.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(coro)
            self._shadow_persist_tasks.add(task)
            task.add_done_callback(self._shadow_persist_tasks.discard)
            return

        # Case 2: called from a worker thread; submit to the captured loop.
        captured = self._loop
        if captured is not None and not captured.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(coro, captured)
                return
            except Exception:
                log.warning("shadow_persist_threadsafe_submit_failed",
                            exc_info=True)

        # Case 3: nothing reachable — close to avoid "never awaited" warnings.
        try:
            coro.close()
        except Exception as exc:
            log.debug("shadow_persist_coro_close_failed", error=str(exc))

    async def _shadow_write_state_memory(
        self, snapshot, entries, batch_end: int, branch_id: str,
    ) -> None:
        """Background: persist STATE snapshot + LEDGER entries from a refresh.

        Errors are swallowed (logged) — shadow writes must never bubble up
        and break the main path. Empty parses are gated client-side: empty
        snapshot.fields means we DON'T write a snapshot row (preserves prior).
        """
        persistence = self._persistence
        user_id = self._persist_user_id
        if persistence is None or not user_id:
            return
        try:
            if snapshot is not None and snapshot.fields:
                await persistence.store_state_snapshot(
                    self._session_id, branch_id, batch_end,
                    snapshot.to_dict(), user_id=user_id,
                )
            if entries:
                await persistence.store_ledger_entries(
                    self._session_id, branch_id,
                    [{"round_num": e.round_num, "category": e.category,
                      "content": e.content} for e in entries],
                    user_id=user_id,
                )
        except Exception:
            log.warning("shadow_persist_state_memory_failed",
                        session_id=self._session_id, branch_id=branch_id,
                        exc_info=True)

    async def _shadow_upsert_branch(
        self, branch_id: str, parent_branch_id: str | None, branch_point: int,
    ) -> None:
        """Background: register a (possibly new) branch in narrative_branches."""
        persistence = self._persistence
        user_id = self._persist_user_id
        if persistence is None or not user_id:
            return
        try:
            await persistence.upsert_branch(
                self._session_id, branch_id, parent_branch_id, branch_point,
                user_id=user_id,
            )
        except Exception:
            log.warning("shadow_persist_upsert_branch_failed",
                        session_id=self._session_id, branch_id=branch_id,
                        exc_info=True)

    def peek_branch_detection(self, request) -> object:
        """Read-only peek at what would happen if process_request were called.

        Used by the handler (async) to pre-fetch a STATE snapshot from history
        before process_request (sync) actually fires rollback_to. Does not
        mutate engine state.
        """
        return self._branch_tracker.detect_branch(request)

    def prepare_branch_snapshot(self, snapshot_data: dict | None) -> None:
        """Stash an async-fetched STATE snapshot for the next rollback.

        rollback_to consumes this on its next call. If None, rollback_to
        falls back to its legacy wipe-to-None behavior.
        """
        self._pending_branch_snapshot = snapshot_data

    @property
    def state(self) -> NarrativeSessionState:
        return self._state

    @property
    def character_card(self) -> CharacterCard | None:
        return self._character_card

    @property
    def world_state(self) -> SceneState:
        return self._world_tracker.state

    @property
    def relationship_tracker(self) -> RelationshipTracker:
        return self._relationship_tracker

    @property
    def request_logs(self) -> list[dict]:
        return self._request_logs

    @property
    def last_request_log(self) -> dict | None:
        """Backward compat: return the most recent log."""
        return self._request_logs[-1] if self._request_logs else None

    def add_request_log(self, entry: dict) -> None:
        """Append a log entry, evicting oldest if over the configured limit."""
        from augmentum.config import settings
        limit = settings.narrative_request_log_limit
        self._request_logs.append(entry)
        if len(self._request_logs) > limit:
            self._request_logs = self._request_logs[-limit:]

    def sync_to_state(self) -> None:
        """Sync ephemeral engine data back to NarrativeSessionState for persistence.

        Called before _persist_state() to ensure relationships, three-layer
        memory, message history, and all engine state survive a restart.
        """
        self._state.relationships = self._relationship_tracker.to_dict_list()

        # Three-layer memory — always persist current state so branch rollbacks
        # (which clear snapshot/ledger) are reflected in the DB.  Writing {}
        # when snapshot is None prevents a stale snapshot from surviving a
        # branch switch + server restart.
        self._state.state_snapshot_data = (
            self._state_snapshot.to_dict() if self._state_snapshot else {}
        )
        if self._memory_ledger:
            self._state.memory_ledger_data = [e.to_dict() for e in self._memory_ledger]

        # Per-branch state cache + group chat state + engine metadata
        branch_data = dict(self._branch_states)
        if self._state.group_id:
            branch_data["__group__"] = {
                "group_id": self._state.group_id,
                "speaker_index": self._state.group_speaker_index,
            }
        branch_data["__meta__"] = {
            "pre_refresh_ledger_len": self._pre_refresh_ledger_len,
        }
        self._state.branch_states_data = branch_data

        # Message history — persisted so summary refresh works after restart
        self._state.message_history_data = list(self._message_history)

        # Scene context from world tracker
        self._state.scene_context = self._world_tracker.state.to_dict()

        # Compaction flag
        self._state.needs_compaction = self._needs_compaction

        # Request logs (context viewer history)
        self._state.request_logs = list(self._request_logs)

        # Overflow summaries — always sync (not just when populated)
        self._state.overflow_summaries = list(self._message_summaries)

        # Lorebook runtime counters — sticky/cooldown/delay survive restart
        self._state.lorebook_runtime_state = self._lore_engine.to_state_dict()

        # Lorebook entries themselves — the LoreEngine is the live source of
        # truth once a session is running (it absorbs character-book imports,
        # UI edits via replace_entries_preserving_state, AND model-authored
        # entries from the ``lorebook.create`` tool). ``save_session_state``
        # persists ``state.lorebook``, so without this sync any entry the
        # model establishes mid-narrative would never reach SQLite and would
        # vanish on the next restart/reload. Mirror the engine's entries back
        # into state so the user-scoped persist path writes them.
        self._state.lorebook = list(self._lore_engine.entries.values())

    def process_request(
        self,
        request: InternalChatRequest,
        retrieved_archive: list[dict] | None = None,
        context_limit: int = 0,
        supports_mid_system: bool = False,
    ) -> NarrativeResult:
        """Process an incoming request through the narrative pipeline.

        This is the main entry point for narrative mode.

        Args:
            request: The incoming chat request.
            retrieved_archive: Pre-retrieved archive exchanges from the handler's
                vec query. Each dict has user_content, assistant_content, summary.
            supports_mid_system: Whether the target backend tolerates system
                messages injected mid-conversation. True for llama-server /
                Ollama / the in-house engine (slot reuse benefits from a
                stable mid-conversation injection point); False for cloud
                OpenAI-compat providers, which often reject the shape with a
                400. When False, dynamic STATE/MEMORY is folded into the
                leading system block instead.
        """
        # Step 1: Initialize on first message (parse character card)
        if not self._initialized:
            self._initialize(request)

        # Step 1.5: Branch detection — check if the incoming messages
        # diverge from our tracked history (user deleted messages, swiped,
        # or took a different path). If so, save the current branch's state
        # and either restore the target branch's saved state or roll back
        # to the branch point.
        branch_detected = False
        detection = self._branch_tracker.detect_branch(request)
        if detection.is_branch:
            branch_detected = True

            # Save current branch state before switching away
            old_branch = detection.parent_branch_id
            self._save_branch_state(old_branch)

            self._branch_tracker.apply_branch(detection)
            new_branch = detection.new_branch_id

            # Try to restore a previously saved state for this branch.
            # This happens when the user swaps back to an earlier branch
            # they've already visited.
            restored = self._restore_branch_state(new_branch)

            if not restored:
                # No saved state — this is a genuinely new path.
                # Roll back to the branch point so stale memory doesn't leak.
                self.rollback_to(detection.branch_point)

            # Discard pre-fetched archive context — it was retrieved before we
            # knew a branch would occur and may reference the abandoned path.
            # Next exchange will retrieve correctly against the new branch.
            retrieved_archive = None

            log.info(
                "narrative_branch_switch",
                branch_point=detection.branch_point,
                old_branch=old_branch,
                new_branch=new_branch,
                restored=restored,
                session=self._session_id,
            )

            # Shadow write: register branch metadata in narrative_branches.
            # Idempotent at the SQL level via INSERT OR IGNORE.
            if self._persistence is not None:
                self._schedule_shadow_persist(
                    self._shadow_upsert_branch(
                        new_branch, old_branch, detection.branch_point,
                    ),
                )

        # Track all messages from the request (skips already-tracked ones).
        # If nothing new is tracked AND no branch was detected, the client is
        # replaying the exact same user message — this is a regeneration.
        tracked_new = self._branch_tracker.track_request_messages(request)
        is_regeneration = (
            not branch_detected
            and len(tracked_new) == 0
            and len(self._branch_tracker.active_messages) > 0
        )

        if is_regeneration:
            # Signal process_response to replace the last history entry rather
            # than appending — prevents [user5, asst5_old, asst5_new] stacking.
            # Only set the flag if a prior response actually exists to replace
            # (even-length history ends on an assistant message). If the first
            # attempt failed before any response was recorded, history ends on
            # the user message — replacing it would corrupt the user/assistant
            # alternating order and cause archive swap bugs.
            if self._message_history and len(self._message_history) % 2 == 0:
                self._pending_regen = True
            # Roll back any ledger entries that were added by the background
            # refresh triggered after the previous (discarded) generation.
            # This prevents the model from seeing "what it already decided"
            # and replicating it — restoring genuine creative freedom.
            # The boundary (_pre_refresh_ledger_len) is persisted across
            # restarts so this guard works even after a server reboot.
            # It also works in-session when _refresh_ran_this_session is True.
            if ((self._refresh_ran_this_session
                    or self._pre_refresh_ledger_len > 0)
                    and len(self._memory_ledger) > self._pre_refresh_ledger_len):
                stripped = len(self._memory_ledger) - self._pre_refresh_ledger_len
                self._memory_ledger = self._memory_ledger[:self._pre_refresh_ledger_len]
                log.info(
                    "narrative_regen_ledger_restored",
                    session_id=self._session_id,
                    stripped_entries=stripped,
                    ledger_remaining=len(self._memory_ledger),
                )

        # Step 2: Extract the latest user message
        user_message = self._get_last_user_message(request)
        if not user_message:
            return NarrativeResult(
                augmented_request=request,
                context=BuiltContext(),
                state=self._state,
            )

        message_index = self._state.message_count
        if not is_regeneration:
            self._state.message_count += 1

        # Step 3: Track the message content (skip on regen — avoids duplicate in history)
        if not is_regeneration:
            self._message_history.append(user_message)

        # Step 4: Extract state changes from user message (if enabled)
        from augmentum.config import settings as _cfg
        if _cfg.narrative_state_tracking_enabled:
            self._extract_state_from_message(user_message, message_index)

        # Step 5: Consistency checks — disabled (regex-based checker was unreliable).
        # Contradiction dataclass + context_builder injection remain for future
        # LLM-based checking.
        contradictions: list[Contradiction] = []

        # Step 6: Trigger lorebook entries (if enabled)
        # When lorebook tools are on, skip keyword-triggered injection entirely —
        # the model pulls what it needs via lorebook.check. Only constant entries
        # still inject (they're always-on by definition). Keywords serve as
        # relevance ranking for tool search results, not injection triggers.
        triggered_lore: list = []
        _lorebook_tools_on = _cfg.narrative_lorebook_native_tools_enabled
        if _cfg.narrative_backend_lorebook and not _lorebook_tools_on:
            _card = self._character_card
            _char_desc = _card.description if _card else ""
            _char_pers = _card.personality if _card else ""
            _scenario = _card.scenario if _card else ""
            _creator = _card.creator_notes if _card else ""
            triggered_lore = self._lore_engine.scan_and_trigger(
                messages=list(reversed(self._message_history)),
                scan_depth=_cfg.narrative_lorebook_scan_depth,
                message_index=message_index,
                recursive=_cfg.world_info_recursive,
                max_recursion=_cfg.world_info_max_recursion_steps,
                min_activations=_cfg.world_info_min_activations,
                token_budget=_cfg.world_info_budget_cap,
                char_description=_char_desc,
                char_personality=_char_pers,
                persona_description="",
                scenario=_scenario,
                creator_notes=_creator,
            )
        elif _lorebook_tools_on:
            triggered_lore = [
                e for e in self._lore_engine.entries.values()
                if e.enabled and e.constant
            ]
            # Don't advance lorebook timers on regen — sticky/cooldown/delay
            # counters would double-decrement for the same semantic turn, so
            # a sticky=3 entry effectively becomes 2-turn after one regen, etc.
            if not is_regeneration:
                self._lore_engine.advance_turn()

        # Step 6.5: Split lore into tiers (2026-07-15 stable-prefix design).
        # CORE = constants + entries that keep re-triggering (hysteresis in
        # LoreEngine.update_core_membership). Core renders as ONE system
        # message right after the card in BOTH the live payload and the
        # kv_stable_messages snapshot — so the KV checkpoint covers it and
        # a big book stops re-prefilling every turn. REACTIVE = the rest of
        # this turn's triggers; they stay in the per-turn injected block
        # (tail placement, suffix-only cost). at_depth entries are always
        # reactive (their whole point is mid-history splicing).
        core_lore_text = ""
        if triggered_lore or self._lore_engine.entries:
            from augmentum.state.narrative_state import LorebookPosition
            _trig_ids = [
                e.id for e in triggered_lore
                if e.position != LorebookPosition.AT_DEPTH
            ]
            _core_ids = self._lore_engine.update_core_membership(
                _trig_ids, message_index,
            )
            _core_entries = [
                e for e in self._lore_engine.entries.values()
                if e.enabled and (e.constant or e.id in _core_ids)
                and e.position != LorebookPosition.AT_DEPTH
            ]
            if _core_entries:
                # Deterministic order (priority, then id) — byte-stable
                # between turns is the whole contract.
                _core_entries.sort(key=lambda e: (e.priority, e.id))
                _CORE_CHAR_CAP = 60_000  # ~15k tokens; whole entries only
                _kept, _used = [], 0
                for e in _core_entries:
                    if _used + len(e.content) > _CORE_CHAR_CAP and _kept:
                        log.warning(
                            "lore_core_budget_dropped", entry=e.id,
                            entry_chars=len(e.content), used=_used,
                        )
                        continue
                    _kept.append(e)
                    _used += len(e.content)
                core_lore_text = (
                    "[World Canon — established setting reference]\n\n"
                    + "\n\n".join(e.content for e in _kept)
                )
                _core_kept_ids = {e.id for e in _kept}
                triggered_lore = [
                    e for e in triggered_lore if e.id not in _core_kept_ids
                ]

        # Step 7: Inject retrieved archive context (vec-based, from handler)
        # Dedup: skip archive entries whose turn_number is within the chat
        # history window — those exchanges are already in the message array
        # the model will see.
        # When memory is disabled, suppress injection entirely — the data is
        # still persisted and resumes if re-enabled, but costs zero context.
        if not self._mem_setting("memory_enabled"):
            state_text = ""
            memory_text = ""
            retrieved_archive = None
        else:
            state_text = self.get_state_text() if self._mem_setting("memory_state_enabled") else ""
            memory_text = self.get_memory_text() if self._mem_setting("memory_ledger_enabled") else ""
        if retrieved_archive:
            # Recency decay: gently penalise older exchanges so recent
            # semantically-similar context isn't displaced by stale material.
            # decay=0.15 → round-1 entry in a 200-round chat gets distance×1.15.
            # A near-perfect match (distance≈0) still wins regardless of age.
            _decay = 0.15
            for _ex in retrieved_archive:
                _age = max(0, message_index - _ex.get("turn_number", 0))
                _age_pct = _age / max(1, message_index)
                _ex["_adj_dist"] = _ex.get("distance", 0.0) * (1.0 + _decay * _age_pct)
            retrieved_archive = sorted(retrieved_archive, key=lambda x: x["_adj_dist"])

            # Count non-system messages to determine the history window.
            # Exchanges within this window are redundant (already in context).
            history_window = sum(1 for m in request.messages if m.role != "system")

            # Build set of ledger content for dedup against archive summaries
            ledger_texts = set()
            for entry in self._memory_ledger:
                if entry.content:
                    ledger_texts.add(entry.content.lower().strip())

            hits: list[tuple[int, str]] = []  # (turn_number, formatted_line)
            for ex in retrieved_archive:
                turn = ex.get("turn_number", 0)
                # Skip if this exchange is within the current chat history
                if turn > 0 and turn >= (message_index - history_window):
                    continue
                summary = ex.get("summary", "")
                # Skip if the summary substantially overlaps with a ledger entry
                if summary and summary.lower().strip() in ledger_texts:
                    continue
                turn_label = f"[R{turn}] " if turn > 0 else ""
                if summary:
                    hits.append((turn, f"- {turn_label}{summary}"))
                else:
                    hits.append((turn,
                        f"- {turn_label}User: {ex['user_content'][:150]} → "
                        f"Assistant: {ex['assistant_content'][:150]}"
                    ))
            if hits:
                # Chronological order — oldest first so context reads naturally
                hits.sort(key=lambda x: x[0])
                retrieval_block = "Relevant earlier exchanges:\n" + "\n".join(h[1] for h in hits)
                memory_text = (
                    memory_text + "\n---\n" + retrieval_block
                    if memory_text else retrieval_block
                )

        # Step 8: Build enhanced context
        # Only include blocks that are enabled — UI handles card, lorebook, examples
        # Dedup: skip relationship tracker summary when STATE has character_dynamics
        state_fields = self._state_snapshot.fields if self._state_snapshot else {}
        has_dynamics = bool(state_fields.get("character_dynamics") or state_fields.get("key_relationships"))
        # (dynamics field already contains per-character relationship annotations)
        rel_summary = ""
        if _cfg.narrative_state_tracking_enabled and not has_dynamics:
            rel_summary = self._relationship_tracker.get_context_summary()

        # In group chat mode, skip card summary/examples injection — the handler
        # already swapped the system prompt to the current speaker's card.
        is_group = bool(self._state.group_id)

        context = self._context_builder.build(
            characters=list(self._state.entities.values()) if _cfg.narrative_state_tracking_enabled else None,
            scene=self._world_tracker.state if _cfg.narrative_state_tracking_enabled else None,
            active_plots=self._plot_tracker.active_threads if _cfg.narrative_state_tracking_enabled else None,
            recent_facts=self._state.get_recent_facts(10) if _cfg.narrative_state_tracking_enabled else None,
            lorebook_entries=triggered_lore if _cfg.narrative_backend_lorebook else None,
            contradictions=None,  # Consistency checking removed (was regex-based, unreliable)
            character_card_summary="" if is_group else (self._character_card.trait_summary if (_cfg.narrative_backend_card_summary and self._character_card) else ""),
            state_text=state_text,
            example_dialogue="" if is_group else (self._character_card.example_dialogue if (_cfg.narrative_backend_examples and self._character_card) else ""),
            creator_notes="" if is_group else (self._character_card.creator_notes if (_cfg.narrative_backend_examples and self._character_card) else ""),
            memory_text=memory_text,
            relationship_summary=rel_summary,
            token_budget=_cfg.narrative_context_budget,
            recall_tools_enabled=_cfg.narrative_recall_tools_enabled,
            lorebook_tools_enabled=_cfg.narrative_lorebook_native_tools_enabled,
        )

        # Step 8: Augment the request (+ enforce context budget)
        augmented = self._augment_request(
            request, context,
            context_limit=context_limit,
            supports_mid_system=supports_mid_system,
            core_lore_text=core_lore_text,
        )

        log.info(
            "narrative_processed",
            message_index=message_index,
            entities=len(self._state.entities),
            facts=len(self._state.facts),
            contradictions=len(contradictions),
            lore_triggered=len(triggered_lore),
            context_tokens=context.total_tokens_estimate,
        )

        return NarrativeResult(
            augmented_request=augmented,
            context=context,
            state=self._state,
            contradictions=contradictions,
            branch_detected=branch_detected,
            is_regeneration=is_regeneration,
        )

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Detect AI refusal/safety responses that shouldn't enter memory.

        Delegates to the shared detection in memory.py which uses compound
        phrase matching to avoid false positives on normal dialogue.
        """
        from augmentum.modes.narrative.memory import _is_refusal_text
        return _is_refusal_text(text)

    def process_response(self, response_text: str) -> None:
        """Process an AI response to extract state changes.

        Refusal/safety responses are detected and excluded from
        message history to prevent contaminating the memory ledger.
        """
        if not response_text:
            return

        if self._is_refusal(response_text):
            self.undo_last_request()
            log.info("narrative_refusal_filtered", length=len(response_text))
            return

        message_index = self._state.message_count
        self._state.message_count += 1

        if self._pending_regen and self._message_history:
            # Replace the old response in-place — keeps the user/assistant
            # alternating pattern intact after a regen.
            self._message_history[-1] = response_text
            self._pending_regen = False
        else:
            self._pending_regen = False
            self._message_history.append(response_text)

        from augmentum.config import settings as _cfg
        if _cfg.narrative_state_tracking_enabled:
            self._extract_state_from_message(response_text, message_index)
            self._state.scene_context = self._world_tracker.state.to_dict()

    def undo_last_request(self) -> None:
        """Undo state changes from process_request when no valid response follows.

        Called by the handler when the backend returns empty content or errors.
        Without this, the orphan user message breaks the alternating user/assistant
        pattern in _message_history, corrupting archive pairing.
        """
        if self._message_history and len(self._message_history) % 2 == 1:
            self._message_history.pop()
            self._state.message_count = max(0, self._state.message_count - 1)
        self._pending_regen = False

    def _initialize(self, request: InternalChatRequest) -> None:
        """Initialize the engine from the first request."""
        self._initialized = True

        # In group chat mode, skip card parsing — the handler manages character
        # cards per-turn. The engine shouldn't lock onto one character's card.
        if self._state.group_id:
            self._state.card_type = "character"
            return

        # Parse character card from system prompt
        system_prompt = self._extract_system_prompt(request)
        if system_prompt:
            self._character_card = self._card_parser.parse(system_prompt)
            if self._character_card:
                # Store the raw parsed name (empty for narrator/freeform cards),
                # not display_name — display_name falls back to "Unknown Character"
                # which would otherwise leak into memory prompts, {{char}} macros,
                # and regex-script scoping as if it were the real character name.
                self._state.character_card_name = self._character_card.name
                self._register_character_from_card(self._character_card)

                # Detect card type for memory summary focus
                self._state.card_type = detect_card_type(self._character_card).value

                # Load lorebook from V2 character book if present
                char_book = self._character_card.raw_data.get("character_book")
                if char_book:
                    entries = self._lore_engine.load_from_character_book(char_book)
                    self._state.lorebook = entries

                log.info(
                    "narrative_initialized",
                    character=self._character_card.display_name,
                    format=self._character_card.source_format,
                    card_type=self._state.card_type,
                    lorebook_entries=len(self._state.lorebook),
                )

    def _register_character_from_card(self, card: CharacterCard) -> None:
        """Create an entity from a parsed character card."""
        if not card.name:
            return

        entity = Entity(
            id=_new_id(),
            session_id=self._session_id,
            entity_type=EntityType.CHARACTER,
            name=card.name,
            aliases=card.aliases,
            state=EntityState(
                emotional_state="neutral",
            ),
            branch_id=self._state.branch_id,
        )
        self._state.entities[entity.id] = entity

    def _extract_state_from_message(self, text: str, message_index: int) -> None:
        """Extract all state changes from a message."""
        known_chars = [
            e for e in self._state.entities.values()
            if e.entity_type == EntityType.CHARACTER
        ]

        # Character state changes
        char_updates = self._character_tracker.extract_updates(text, known_chars)
        for update in char_updates:
            entity = self._state.get_entity_by_name(update.name)
            if entity:
                self._character_tracker.apply_update(entity, update, message_index)

        # World state changes
        world_delta = self._world_tracker.extract_world_changes(text)
        if world_delta:
            self._world_tracker.apply_delta(world_delta, message_index, self._state.branch_id)
            self._state.scene_context = self._world_tracker.state.to_dict()

        # Relationship signals
        char_names = [e.name for e in known_chars]
        rel_deltas = self._relationship_tracker.extract_signals(text, char_names)
        for rd in rel_deltas:
            self._relationship_tracker.apply_delta(rd, message_index)

        # Plot signals
        plot_signals = self._plot_tracker.extract_plot_signals(text)
        if plot_signals and self._plot_tracker.detect_resolutions(text):
                for thread in self._plot_tracker.active_threads:
                    # Simple heuristic: if resolution detected and thread is active, check
                    # if it seems related. Full implementation would use LLM.
                    self._plot_tracker.progress_thread(
                        thread.id, f"Resolution signal at message {message_index}", message_index
                    )

    def _augment_request(
        self, request: InternalChatRequest, context: BuiltContext,
        context_limit: int = 0,
        supports_mid_system: bool = False,
        core_lore_text: str = "",
    ) -> InternalChatRequest:
        """Create an augmented copy of the request with injected context.

        Also enforces the total context budget.  System messages (card +
        injected STATE/MEMORY) are treated as fixed cost; chat history is
        trimmed oldest-first to make everything fit.

        Args:
            context_limit: Total token budget (0 = unlimited).  Determined
                by the handler from auto-detection or manual config.
            supports_mid_system: When True, dynamic context is injected as
                a system message just before the latest user turn (the
                pattern that benefits llama-server / Ollama slot reuse).
                When False, it's folded into the leading system block —
                a universally accepted shape that avoids 400s from cloud
                providers like NVIDIA NIM, DeepSeek, Mistral, and Cohere.
        """
        # Deep copy messages
        new_messages = [
            Message(role=m.role, content=m.content, images=m.images, tool_calls=m.tool_calls)
            for m in request.messages
        ]

        # Stable head block: ONE system message directly after the card,
        # in the STABLE region — the identical insertion happens in the
        # kv_stable_messages snapshot below, so live payload and checkpoint
        # stay byte-aligned and the content prefills once, not every turn.
        # Carries the turn-stable context (card summary / example dialogue
        # / tool guidance — context.stable_text) followed by canon-core
        # lore. Before stable_text existed these static blocks floated in
        # the per-turn injection, re-prefilling their full length every
        # turn (live-measured 2026-07-18: 4-6.5k wasted prefill tokens
        # per narrative turn, reuse 44% vs 80% expected).
        stable_head_text = "\n\n".join(
            t for t in (context.stable_text, core_lore_text) if t
        )

        def _insert_stable_head(msgs: list[Message]) -> None:
            # Fold the stable head INTO the leading system message rather
            # than inserting a SECOND consecutive system message. Two leading
            # systems ([system(card)][system(stable_head)]) are a shape the
            # backend's _normalize_system_messages merges on every live turn
            # (spamming oai_payload_late_system_normalized) — and, critically,
            # prepare_stable_checkpoint prewarms the RAW snapshot (two system
            # blocks) while the live turn renders the MERGED one, so the
            # prewarmed KV prefix diverges from the next turn at the
            # stable-head boundary ([Example Dialogue]) and every narrative
            # turn re-prefills cold (kv_prefix_stability baseline=prewarm,
            # contract=violated). Folding here yields ONE leading system in
            # both the live payload and the checkpoint snapshot, byte-identical
            # to the old backend-merged output, so the prefix stays stable.
            # Mirrors prompt_presets._prepend_system.
            if msgs and msgs[0].role == "system":
                head = msgs[0]
                msgs[0] = Message(
                    role="system",
                    content=(
                        f"{head.content}\n\n{stable_head_text}"
                        if head.content else stable_head_text
                    ),
                    images=head.images,
                    tool_calls=head.tool_calls,
                )
            else:
                msgs.insert(0, Message(role="system", content=stable_head_text))

        if stable_head_text:
            _insert_stable_head(new_messages)

        # Step 1: Inject dynamic context (STATE/MEMORY/archive snippets).
        #
        # Two placement strategies, selected by ``supports_mid_system``:
        #
        # (a) Mid-conversation injection — placed as a ``system`` message
        # just before the latest user turn. This is the cache-friendly
        # pattern: llama-server's slot cache and Ollama's runner both
        # prefix-match at the token level, so a stable injection point
        # with a stable head means everything up to the injection is a
        # cache hit on every turn:
        #
        #   [system prompt — STABLE, cached] → [turn 1 — cached] →
        #   [turn 2 — cached] → ... → [dynamic context — new each turn]
        #   → [latest user message — new]
        #
        # (b) Leading-system fold — appended onto the first system
        # message (or a new leading system message if none exists).
        # Universally accepted shape; required for cloud OpenAI-compat
        # providers like NVIDIA NIM that reject system messages after
        # position 0 with a 400. We trade one-time prefix-cache friction
        # (no shared cache to preserve for cloud anyway) for never
        # producing a malformed request.
        if context.injected_text:
            if supports_mid_system:
                last_user_idx = None
                for i in range(len(new_messages) - 1, -1, -1):
                    if new_messages[i].role == "user":
                        last_user_idx = i
                        break

                injection_msg = Message(role="system", content=context.injected_text)
                if last_user_idx is not None:
                    new_messages.insert(last_user_idx, injection_msg)
                else:
                    new_messages.append(injection_msg)
            else:
                for i, msg in enumerate(new_messages):
                    if msg.role == "system":
                        new_messages[i] = Message(
                            role="system",
                            content=f"{msg.content}\n\n{context.injected_text}",
                            images=msg.images,
                            tool_calls=msg.tool_calls,
                        )
                        break
                else:
                    new_messages.insert(
                        0, Message(role="system", content=context.injected_text)
                    )

        # Step 1.5: Splice at-depth lorebook entries into the messages array.
        #
        # This mirrors SillyTavern's populationInjectionPrompts algorithm so
        # Author's-Note-style injection, position-tagged scene anchors, and
        # reactive state beats land in the model's recent window instead of
        # the system block.
        #
        # Semantics: depth 0 = appended after the newest message; depth N =
        # inserted N messages back from the end. Same (depth, role) entries
        # are joined with "\n" into one message; role buckets are emitted in
        # stable system→user→assistant order within each depth.
        if context.depth_entries:
            # Bucket by depth
            by_depth: dict[int, list] = {}
            for de in context.depth_entries:
                by_depth.setdefault(int(de.depth), []).append(de)

            # When the backend rejects mid-conversation system messages,
            # demote system-role lorebook entries to user role. The
            # at-depth positioning (the lorebook feature's whole point)
            # is preserved; only the role label changes. Mirrors
            # SillyTavern's handling for cloud targets.
            def _resolved_role(role: str) -> str:
                if role == "system" and not supports_mid_system:
                    return "user"
                return role

            new_messages.reverse()  # newest first
            inserted = 0
            role_order = ("system", "user", "assistant")
            for depth in sorted(by_depth.keys()):
                bucket = by_depth[depth]
                for role in role_order:
                    role_entries = sorted(
                        (e for e in bucket if e.role == role),
                        key=lambda e: e.order,
                    )
                    if not role_entries:
                        continue
                    content = "\n".join(e.content for e in role_entries)
                    # Python list.insert clamps to end on large indices
                    # (same as JS splice) — no explicit bound check needed.
                    new_messages.insert(
                        depth + inserted,
                        Message(role=_resolved_role(role), content=content),
                    )
                    inserted += 1
            new_messages.reverse()  # restore chronological order

        # Step 2: Enforce total context budget — system messages are fixed,
        # chat history fills whatever remains.  Walk backward from newest
        # chat message, keeping as many as fit within the remaining budget.
        # Reserve space for the response (max_tokens) so prompt + response
        # doesn't exceed the model's actual context window.
        ctx_limit = context_limit
        if ctx_limit > 0 and request.max_tokens:
            ctx_limit = max(0, ctx_limit - request.max_tokens)
        if ctx_limit > 0:
            from augmentum.utils.tokenizer import count_tokens as _count

            system_msgs = [m for m in new_messages if m.role == "system"]
            chat_msgs = [m for m in new_messages if m.role != "system"]

            # Fixed cost: system prompt + card + injected memory
            system_cost = sum(_count(m.content or "") + 4 for m in system_msgs)
            history_budget = max(0, ctx_limit - system_cost)

            kept: list[Message] = []
            remaining = history_budget
            for m in reversed(chat_msgs):
                cost = _count(m.content or "") + 4  # +4 per-message overhead
                if remaining - cost < 0 and kept:
                    break
                kept.append(m)
                remaining -= cost
            kept.reverse()
            if len(kept) < len(chat_msgs):
                # ORDER-PRESERVING drop. ``system_msgs + kept`` hoisted every
                # system message to the front — including the dynamic
                # STATE/MEMORY injection that Step 1 deliberately placed just
                # before the latest user turn (supports_mid_system placement).
                # A per-turn-changing block at the head diverges the token
                # prefix at message 0 and forces a full re-prefill of the
                # entire context every turn on long (trim-triggering)
                # sessions. Drop the oldest chat messages IN PLACE instead —
                # identical budget math, positions untouched.
                dropped = {id(m) for m in chat_msgs[: len(chat_msgs) - len(kept)]}
                new_messages = [m for m in new_messages if id(m) not in dropped]

        # Build the stable-prefix snapshot for KV checkpointing. Mirrors
        # the chat-path trim above so late-game checkpoints don't overflow
        # ctx_size and 400 the prewarm. Differs from ``new_messages`` by
        # excluding per-turn injections (STATE/MEMORY block, depth-entry
        # lorebook splices) so the prefix stays stable across turns;
        # reserves extra headroom so next turn's fresh injection block
        # can land on top without pushing past the slot's context.
        stable_messages: list[Message] = [
            Message(
                role=m.role,
                content=m.content,
                images=list(m.images) if m.images else None,
                tool_calls=list(m.tool_calls) if m.tool_calls else None,
                thinking=m.thinking,
                tool_call_id=m.tool_call_id,
            )
            for m in request.messages
        ]
        if stable_head_text:
            # Mirror of the live-payload insertion above — the stable head
            # IS part of the stable prefix (that's the whole point).
            _insert_stable_head(stable_messages)
        if ctx_limit > 0:
            from augmentum.utils.tokenizer import count_tokens as _count

            INJECTION_HEADROOM = 4096
            stable_sys = [m for m in stable_messages if m.role == "system"]
            stable_chat = [m for m in stable_messages if m.role != "system"]
            stable_sys_cost = sum(_count(m.content or "") + 4 for m in stable_sys)
            stable_budget = max(0, ctx_limit - stable_sys_cost - INJECTION_HEADROOM)
            stable_kept: list[Message] = []
            stable_remaining = stable_budget
            for m in reversed(stable_chat):
                cost = _count(m.content or "") + 4
                if stable_remaining - cost < 0 and stable_kept:
                    break
                stable_kept.append(m)
                stable_remaining -= cost
            stable_kept.reverse()
            if len(stable_kept) < len(stable_chat):
                # Same order-preserving drop as the chat-path trim above —
                # the checkpoint prefix must match the real request's message
                # order or the prewarmed KV can never be a token-level prefix
                # of the next turn.
                stable_dropped = {
                    id(m) for m in stable_chat[: len(stable_chat) - len(stable_kept)]
                }
                stable_messages = [
                    m for m in stable_messages if id(m) not in stable_dropped
                ]

        # Use ``dataclass_replace`` so every field on the source request
        # flows through automatically — only ``messages`` and the
        # narrative-specific ``kv_stable_messages`` snapshot need to
        # change here. The previous explicit-field-list pattern silently
        # dropped any field added to ``InternalChatRequest`` after this
        # code was written; ``apply_preset`` (commit 731a96d) was bitten
        # by exactly that and dropped the kv fields downstream.
        return dataclass_replace(
            request,
            messages=new_messages,
            kv_stable_messages=stable_messages,
        )

    def _get_last_user_message(self, request: InternalChatRequest) -> str | None:
        """Get the content of the last user message."""
        for msg in reversed(request.messages):
            if msg.role == "user":
                return msg.content
        return None

    def _extract_system_prompt(self, request: InternalChatRequest) -> str:
        """Extract combined system prompt text."""
        return "\n".join(
            msg.content for msg in request.messages if msg.role == "system"
        )

    # --- Memory management (three-layer architecture) ---

    def should_refresh(self, interval: int) -> bool:
        """Check if it's time to refresh the state+memory."""
        if self._state.message_count == 0:
            return False
        return (self._state.message_count - self._state.last_summary_at) >= interval

    def should_refresh_summary(self, interval: int) -> bool:
        """Backward compat — delegates to should_refresh."""
        return self.should_refresh(interval)

    def build_state_memory_request(
        self, batch_start: int, batch_end: int, *, model: str = "",
    ) -> InternalChatRequest:
        """Build an InternalChatRequest for the STATE+MEMORY LLM call.

        ``model`` should be the resolved STATE/MEMORY model (the handler passes
        ``narrative_memory_model`` or the session's active model). It's a
        keyword arg with an empty default so older callers still work, but a
        blank model will be rejected by the backend — always pass one.
        """
        card_type = CardType(self._state.card_type)

        from augmentum.config import settings as _cfg

        try:
            mode = SummaryMode(self._mem_setting("memory_mode"))
        except ValueError:
            mode = SummaryMode.STANDARD

        system_content, user_content = build_state_memory_prompt(
            card_type=card_type,
            current_state=self._state_snapshot,
            memory_ledger=self._memory_ledger,
            recent_messages=self._message_history,
            char_name=self._state.character_card_name,
            batch_start=batch_start,
            batch_end=batch_end,
            custom_prompt=_cfg.narrative_memory_prompt,
            mode=mode,
        )

        temp = 0.0 if mode == SummaryMode.LITE else 0.3

        return InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=system_content),
                Message(role="user", content=user_content),
            ],
            stream=False,
            temperature=temp,
            # Auto-scale by mode when not manually overridden.
            # STANDARD: STATE ~330 tok + up to 10 MEMORY entries ~300 tok + headers = ~640.
            # LITE: STATE ~175 tok + up to 3 MEMORY entries ~90 tok = ~265.
            # Old hard-coded 400 silently truncated STANDARD output after ~2 entries.
            max_tokens=_cfg.narrative_memory_max_tokens or (400 if mode == SummaryMode.LITE else 700),
        )

    def apply_state_memory_response(
        self, snapshot: StateSnapshot | None, entries: list[MemoryEntry],
        batch_end: int | None = None,
    ) -> None:
        """Apply parsed STATE+MEMORY response — overwrite snapshot, append entries.

        ``batch_end`` is the message_count at the time the refresh was planned.
        Using it (instead of the current message_count) prevents skipping any
        exchanges that arrived while the background LLM call was in-flight.

        ``snapshot`` may be ``None`` when the STATE layer is disabled — in that
        case the existing snapshot is preserved unchanged.
        """
        if snapshot is not None:
            self._state_snapshot = snapshot
        # Record the boundary before adding new entries so regen can roll back.
        self._pre_refresh_ledger_len = len(self._memory_ledger)
        self._refresh_ran_this_session = True
        self._memory_ledger.extend(entries)
        self._state.last_summary_at = batch_end if batch_end is not None else self._state.message_count

        # Also update legacy memory_summary for backward compat
        state_text = format_state_for_context(self._state_snapshot) if self._state_snapshot else ""
        memory_text = format_ledger_for_context(self._memory_ledger) if self._memory_ledger else ""
        self._state.memory_summary = (state_text + "\n\n" + memory_text).strip() or self._state.memory_summary

        # Check compaction ceiling (0 = unlimited, never compact)
        if self._mem_setting("memory_ledger_enabled"):
            ceiling = self._mem_setting("memory_ledger_ceiling")
            if ceiling > 0 and len(self._memory_ledger) >= ceiling:
                self._needs_compaction = True

        log.info(
            "narrative_state_memory_updated",
            session_id=self._session_id,
            state_fields=len(snapshot.fields) if snapshot else 0,
            new_entries=len(entries),
            total_ledger=len(self._memory_ledger),
            at_message=self._state.last_summary_at,
        )

        # Shadow write to branch-tagged tables (Phase 3 chunk 1). Identity if
        # persistence isn't attached or no event loop is running.
        if self._persistence is not None:
            branch_id = self._branch_tracker.current_branch
            self._schedule_shadow_persist(
                self._shadow_write_state_memory(
                    snapshot, list(entries),
                    batch_end if batch_end is not None else self._state.message_count,
                    branch_id,
                ),
            )

    def get_state_text(self) -> str:
        """Get formatted state snapshot for context injection."""
        if not self._state_snapshot:
            return ""
        return format_state_for_context(self._state_snapshot)

    def get_memory_text(self) -> str:
        """Get formatted memory ledger for context injection."""
        if not self._memory_ledger:
            return ""
        return format_ledger_for_context(self._memory_ledger)

    @property
    def state_snapshot(self) -> StateSnapshot | None:
        return self._state_snapshot

    @property
    def memory_ledger(self) -> list[MemoryEntry]:
        return self._memory_ledger

    @property
    def needs_compaction(self) -> bool:
        return self._needs_compaction

    @needs_compaction.setter
    def needs_compaction(self, value: bool) -> None:
        self._needs_compaction = value

    def apply_edited_state(self, fields: dict[str, str]) -> None:
        """Apply user-edited state fields."""
        card_type = CardType(self._state.card_type)
        if self._state_snapshot:
            self._state_snapshot.fields = fields
        else:
            self._state_snapshot = StateSnapshot(fields=fields, card_type=card_type)

    def apply_edited_ledger(self, entries_data: list[dict]) -> None:
        """Apply user-edited ledger (full replacement)."""
        self._memory_ledger = [MemoryEntry.from_dict(e) for e in entries_data]
        # Re-check compaction
        ceiling = self._mem_setting("memory_ledger_ceiling")
        self._needs_compaction = ceiling > 0 and len(self._memory_ledger) >= ceiling

    # --- LLM extraction merge ---

    def merge_llm_extraction(self, extraction: NarrativeExtraction, message_index: int) -> None:
        """Merge LLM-extracted state into existing tracker state.

        LLM wins on conflicts (higher-quality extraction).
        """
        # Merge character updates
        for char_ext in extraction.characters:
            entity = self._state.get_entity_by_name(char_ext.name)
            if not entity:
                # LLM discovered a new character — register it
                entity = Entity(
                    id=_new_id(),
                    session_id=self._session_id,
                    entity_type=EntityType.CHARACTER,
                    name=char_ext.name,
                    state=EntityState(),
                    branch_id=self._state.branch_id,
                )
                self._state.entities[entity.id] = entity

            update = CharacterUpdate(
                name=char_ext.name,
                emotional_state=char_ext.emotion,
                emotional_confidence=char_ext.emotion_confidence,
                physical_state=char_ext.physical_state,
                location=char_ext.location,
                inventory_add=char_ext.inventory_add or None,
                inventory_remove=char_ext.inventory_remove or None,
                relationship_updates=char_ext.relationship_changes or None,
            )
            self._character_tracker.apply_update(entity, update, message_index)

            # Merge relationship changes into the relationship graph
            if char_ext.relationship_changes:
                self._relationship_tracker.merge_llm_relationships(
                    char_ext.name,
                    char_ext.relationship_changes,
                    message_index,
                )

        # Merge world state (LLM overrides regex results)
        if extraction.world:
            delta: dict = {}
            if extraction.world.location:
                delta["location"] = extraction.world.location
            if extraction.world.time_of_day:
                delta["time_of_day"] = extraction.world.time_of_day
            if extraction.world.weather:
                delta["weather"] = extraction.world.weather
            if extraction.world.atmosphere:
                delta["atmosphere"] = extraction.world.atmosphere
            if delta:
                self._world_tracker.apply_delta(delta, message_index, self._state.branch_id)
                self._state.scene_context = self._world_tracker.state.to_dict()

        # Merge plot signals
        if extraction.plots:
            for thread_desc in extraction.plots.new_threads:
                self._plot_tracker.add_thread(
                    session_id=self._session_id,
                    title=thread_desc[:100],
                    description=thread_desc,
                    message_index=message_index,
                    branch_id=self._state.branch_id,
                )
            for progression in extraction.plots.progressions:
                # Apply progression to most relevant active thread
                for thread in self._plot_tracker.active_threads:
                    self._plot_tracker.progress_thread(
                        thread.id, progression, message_index,
                    )
                    break  # Only progress the first active thread per signal
            for _resolution in extraction.plots.resolutions:
                for thread in self._plot_tracker.active_threads:
                    self._plot_tracker.resolve_thread(thread.id, message_index)
                    break

        # Merge facts
        for fact_ext in extraction.facts:
            self._state.facts.append(Fact(
                id=_new_id(),
                session_id=self._session_id,
                content=fact_ext.content,
                source="llm_extraction",
                confidence=fact_ext.confidence,
                established_at=message_index,
                branch_id=self._state.branch_id,
            ))

        log.info(
            "narrative_llm_merge",
            characters=len(extraction.characters),
            world=bool(extraction.world),
            plots=bool(extraction.plots),
            facts=len(extraction.facts),
        )

    # --- State management ---

    def _save_branch_state(self, branch_id: str) -> None:
        """Save the current engine state for a branch so it can be restored later."""
        from dataclasses import asdict
        self._branch_states[branch_id] = {
            "message_count": self._state.message_count,
            "message_history": list(self._message_history),
            "state_snapshot": self._state_snapshot.to_dict() if self._state_snapshot else None,
            "memory_ledger": [e.to_dict() for e in self._memory_ledger],
            "pre_refresh_ledger_len": self._pre_refresh_ledger_len,
            "memory_summary": self._state.memory_summary,
            "last_summary_at": self._state.last_summary_at,
            "facts": [asdict(f) for f in self._state.facts],
            "contradictions": [asdict(c) for c in self._state.contradictions],
            "entities": {eid: e.to_db_dict() for eid, e in self._state.entities.items()},
            "relationships": self._relationship_tracker.to_dict_list(),
            "archive_history_idx": self._branch_archive_idx.get(branch_id, 0),
            # Per-branch group rotation. Without these, switching branches
            # in a group chat carries the old branch's speaker_index forward
            # and the wrong character speaks next on the restored timeline.
            "group_id": self._state.group_id,
            "group_speaker_index": self._state.group_speaker_index,
        }
        log.debug("branch_state_saved", branch_id=branch_id, message_count=self._state.message_count)

    def _restore_branch_state(self, branch_id: str) -> bool:
        """Restore saved state for a branch. Returns True if state was found."""
        saved = self._branch_states.get(branch_id)
        if not saved:
            return False

        self._state.message_count = saved["message_count"]
        self._message_history = list(saved["message_history"])
        self._state.memory_summary = saved["memory_summary"]
        self._state.last_summary_at = saved["last_summary_at"]

        # Restore state snapshot
        snap_data = saved.get("state_snapshot")
        if snap_data:
            self._state_snapshot = StateSnapshot.from_dict(snap_data)
        else:
            self._state_snapshot = None

        # Restore memory ledger — treat as canonical (this branch's confirmed history)
        self._memory_ledger = [MemoryEntry.from_dict(e) for e in saved.get("memory_ledger", [])]
        saved_boundary = saved.get("pre_refresh_ledger_len")
        self._pre_refresh_ledger_len = (
            min(saved_boundary, len(self._memory_ledger))
            if isinstance(saved_boundary, int)
            else len(self._memory_ledger)
        )

        # Restore facts — handle both dict and Fact objects
        self._state.facts = []
        for f in saved.get("facts", []):
            if isinstance(f, dict):
                # Enum fields need conversion
                self._state.facts.append(Fact(
                    id=f.get("id", _new_id()),
                    session_id=f.get("session_id", ""),
                    content=f.get("content", ""),
                    source=f.get("source", "extracted"),
                    confidence=f.get("confidence", 0.8),
                    domain=f.get("domain", "general"),
                    established_at=f.get("established_at", 0),
                    superseded_by=f.get("superseded_by"),
                    branch_id=f.get("branch_id", "main"),
                    tags=f.get("tags", []),
                ))
            else:
                self._state.facts.append(f)

        # Restore contradictions
        self._state.contradictions = []
        for c in saved.get("contradictions", []):
            if isinstance(c, dict):
                severity = c.get("severity", "minor")
                if isinstance(severity, str):
                    from augmentum.state.narrative_state import ContradictionSeverity
                    severity = ContradictionSeverity(severity)
                self._state.contradictions.append(Contradiction(
                    session_id=c.get("session_id", ""),
                    message_index=c.get("message_index", 0),
                    contradiction_type=c.get("contradiction_type", ""),
                    description=c.get("description", ""),
                    severity=severity,
                    resolution=c.get("resolution"),
                    fact_ids=c.get("fact_ids", []),
                    branch_id=c.get("branch_id", "main"),
                ))
            else:
                self._state.contradictions.append(c)

        # Restore entities (if saved — backward compat with older branch states)
        saved_entities = saved.get("entities")
        if saved_entities and isinstance(saved_entities, dict):
            self._state.entities = {}
            for eid, edata in saved_entities.items():
                if not isinstance(edata, dict):
                    continue
                etype = edata.get("entity_type", "character")
                if isinstance(etype, str):
                    etype = EntityType(etype)
                state_raw = edata.get("state", {})
                if isinstance(state_raw, str):
                    state_raw = json.loads(state_raw)
                aliases = edata.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = json.loads(aliases)
                self._state.entities[eid] = Entity(
                    id=eid,
                    session_id=edata.get("session_id", ""),
                    entity_type=etype,
                    name=edata.get("name", ""),
                    aliases=aliases,
                    state=EntityState.from_dict(state_raw),
                    branch_id=edata.get("branch_id", "main"),
                )

        # Restore relationships (if saved — backward compat with older branch states)
        saved_rels = saved.get("relationships")
        if saved_rels and isinstance(saved_rels, list):
            self._relationship_tracker.load_from_dict_list(saved_rels)

        # Restore archive pointer for this branch (handler reads this after restore)
        saved_archive_idx = saved.get("archive_history_idx")
        if isinstance(saved_archive_idx, int):
            self._branch_archive_idx[branch_id] = saved_archive_idx

        # Restore per-branch group rotation (backward-compat: older branch
        # states predate these keys, so absence means "leave state alone").
        if "group_id" in saved:
            self._state.group_id = saved["group_id"] or ""
        if "group_speaker_index" in saved:
            idx = saved["group_speaker_index"]
            if isinstance(idx, int):
                self._state.group_speaker_index = idx

        log.info(
            "branch_state_restored",
            branch_id=branch_id,
            message_count=self._state.message_count,
            ledger_entries=len(self._memory_ledger),
        )
        return True

    def rollback_to(self, message_index: int) -> None:
        """Roll back all state to a specific message index (for branching).

        Rolls back: message history, memory ledger, state snapshot,
        facts, contradictions, world tracker, plot tracker, and
        forces a state+memory refresh on the next interval.
        """
        self._world_tracker.rollback_to(message_index, self._state.branch_id)
        self._plot_tracker.rollback_to(message_index, self._state.branch_id)

        # Roll back facts
        self._state.facts = [
            f for f in self._state.facts
            if f.established_at <= message_index
        ]

        # Roll back contradictions
        self._state.contradictions = [
            c for c in self._state.contradictions
            if c.message_index <= message_index
        ]

        # Roll back message history
        if message_index < len(self._message_history):
            self._message_history = self._message_history[:message_index]

        # Roll back three-layer memory system
        # Memory ledger: discard entries from rounds after the branch point
        old_ledger_count = len(self._memory_ledger)
        self._memory_ledger = [
            e for e in self._memory_ledger
            if e.round_num <= message_index
        ]

        # State snapshot: prefer a snapshot from history (Phase 3 chunk 3) if
        # the handler pre-fetched one for this branch_point. Without history
        # available, fall back to wiping — the next summary refresh regenerates.
        # The wipe-then-refresh path is what shipped pre-Phase-3, so behavior
        # is identical when persistence isn't attached.
        pending_snap = self._pending_branch_snapshot
        self._pending_branch_snapshot = None  # consume regardless of outcome
        if pending_snap and isinstance(pending_snap, dict) and pending_snap.get("fields"):
            try:
                from augmentum.modes.narrative.memory import StateSnapshot
                self._state_snapshot = StateSnapshot.from_dict(pending_snap)
                log.info(
                    "narrative_state_recovered_from_history",
                    to_index=message_index,
                    fields=len(self._state_snapshot.fields),
                )
            except Exception:
                log.warning("snapshot_recovery_failed", exc_info=True)
                self._state_snapshot = None
        else:
            self._state_snapshot = None

        # Force a summary refresh on the very next message by resetting
        # last_summary_at so should_refresh() returns True immediately.
        # Even when STATE was recovered from history, the recovered snapshot
        # is from the pre-fork timeline; a refresh on the new branch quickly
        # makes it current.
        self._state.last_summary_at = 0

        # Clear the formatted summary so stale text isn't injected
        self._state.memory_summary = ""

        # Sync regen boundary to the post-rollback ledger length
        self._pre_refresh_ledger_len = len(self._memory_ledger)

        # Reset entity states to neutral — cumulative state mutations from the
        # abandoned path can't be surgically undone.  The forced refresh
        # (last_summary_at=0) will re-extract correct state from the surviving
        # message history on the next exchange.  Entity identities (name, aliases)
        # are preserved; only mutable state fields are cleared.
        for entity in self._state.entities.values():
            entity.state = EntityState()

        # Group chat: reset speaker rotation index. After rollback the next
        # turn must not generate as whoever was speaking at the abandoned
        # head of the conversation. Resetting to 0 lets the group_turn_manager
        # resume cleanly from the start of the rotation; if a particular
        # speaker_index needs to be preserved (revisit of a known branch),
        # _restore_branch_state handles that path separately and runs BEFORE
        # rollback_to via the saved __group__ blob.
        if self._state.group_id:
            self._state.group_speaker_index = 0

        # Clear relationship scores — like entity states, these are cumulative
        # deltas with no undo history.  Refresh will rebuild from context.
        self._relationship_tracker = RelationshipTracker()

        self._state.message_count = message_index
        self._state.scene_context = self._world_tracker.state.to_dict()

        log.info(
            "narrative_rolled_back",
            to_index=message_index,
            ledger_pruned=old_ledger_count - len(self._memory_ledger),
            ledger_remaining=len(self._memory_ledger),
        )

    def load_state(self, state: NarrativeSessionState) -> None:
        """Load state from persistence (used when resuming a session)."""
        self._state = state
        # Resumed state means _initialize() already ran. Previously this keyed
        # off character_card_name, which breaks for narrator/freeform cards
        # that legitimately have no name — they'd get re-initialized every
        # request, re-registering entities and spamming logs. Use message
        # count as the real "has this session processed anything" proxy, and
        # keep the name check as a backstop for older persisted rows.
        self._initialized = state.message_count > 0 or bool(state.character_card_name)

        # Restore trackers
        if state.scene_context:
            self._world_tracker.set_state(SceneState.from_dict(state.scene_context))
        self._plot_tracker.set_threads(state.plot_threads)
        self._lore_engine.set_entries(state.lorebook)
        # Restore lorebook timed-effect counters (survives server bounce)
        runtime = getattr(state, "lorebook_runtime_state", None)
        if runtime:
            self._lore_engine.load_state_dict(runtime)

        # Restore relationship tracker
        if state.relationships:
            self._relationship_tracker.load_from_dict_list(state.relationships)
            log.info("relationships_restored", count=len(state.relationships))

        # Restore three-layer memory
        snapshot_data = getattr(state, 'state_snapshot_data', None)
        if snapshot_data and isinstance(snapshot_data, dict) and snapshot_data.get("fields"):
            self._state_snapshot = StateSnapshot.from_dict(snapshot_data)
        ledger_data = getattr(state, 'memory_ledger_data', None)
        if ledger_data and isinstance(ledger_data, list) and len(ledger_data) > 0:
            self._memory_ledger = [MemoryEntry.from_dict(e) for e in ledger_data]
            # Treat loaded ledger as fully canonical — no refresh happened yet this session.
            self._pre_refresh_ledger_len = len(self._memory_ledger)

        # Restore per-branch state cache + group chat state + engine metadata
        if hasattr(state, 'branch_states_data') and state.branch_states_data:
            branch_data = dict(state.branch_states_data)
            # Extract group state if embedded
            group_data = branch_data.pop("__group__", None)
            if group_data:
                state.group_id = group_data.get("group_id", "")
                state.group_speaker_index = group_data.get("speaker_index", 0)
            # Extract engine metadata — currently stores the regen ledger boundary
            meta_data = branch_data.pop("__meta__", None)
            if meta_data:
                saved_len = meta_data.get("pre_refresh_ledger_len")
                if saved_len is not None and isinstance(saved_len, int):
                    self._pre_refresh_ledger_len = min(
                        saved_len, len(self._memory_ledger)
                    )
            self._branch_states = branch_data
            log.info("branch_states_restored", count=len(self._branch_states))

        # Restore message history
        if hasattr(state, 'message_history_data') and state.message_history_data:
            self._message_history = list(state.message_history_data)
            log.info("message_history_restored", count=len(self._message_history))

        # Restore compaction flag
        if hasattr(state, 'needs_compaction') and state.needs_compaction:
            self._needs_compaction = True

        # Restore request logs (context viewer history)
        raw_logs = getattr(state, 'request_logs', None)
        if raw_logs:
            self._request_logs = list(raw_logs)
        elif getattr(state, 'last_request_log', None):
            # Backward compat: old single-dict format → wrap in list
            self._request_logs = [state.last_request_log]

        # Restore overflow summaries (backward compat)
        if state.overflow_summaries:
            self._message_summaries = list(state.overflow_summaries)
            log.info("overflow_summaries_restored", count=len(self._message_summaries))
