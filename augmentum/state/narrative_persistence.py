"""Narrative state persistence — save/load NarrativeSessionState to SQLite."""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass, field

import aiosqlite

from augmentum.state.narrative_state import (
    Contradiction,
    ContradictionSeverity,
    Entity,
    EntityState,
    EntityType,
    Fact,
    LorebookEntry,
    LorebookPosition,
    NarrativeSessionState,
    PlotStatus,
    PlotThread,
    SelectiveLogic,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------
# Persistence-layer DTOs for the branch-tagged tiers (migrations 115-119)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BranchAncestor:
    """One link in a branch's ancestry chain.

    For the leaf branch, ``branch_point`` is 0 (no descendant in the chain).
    For each strict ancestor, ``branch_point`` is the divergence point of its
    descendant in the chain — used to bound row visibility on retrieval.
    """
    branch_id: str
    branch_point: int


@dataclass(frozen=True)
class BranchInfo:
    """Row shape returned by list_branches."""
    branch_id: str
    parent_branch_id: str | None
    branch_point: int
    status: str  # 'active' | 'stale' | 'archived'
    created_at: str
    last_visited_at: str


@dataclass
class SessionStorage:
    """Per-branch + total storage observability for a session."""
    session_id: str
    branches: dict[str, dict[str, int]] = field(default_factory=dict)
    # branches[branch_id] = {"archive_rows": N, "ledger_entries": N,
    #                        "snapshots": N, "approx_bytes": N}
    total_archive_rows: int = 0
    total_ledger_entries: int = 0
    total_snapshots: int = 0
    total_branches: int = 0
    total_approx_bytes: int = 0


_VALID_BRANCH_STATUSES = frozenset({"active", "stale", "archived"})


def _parse_request_logs(raw: str | None) -> list[dict]:
    """Parse request log column — handles old single-dict and new list formats."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        return [parsed] if parsed else []
    if isinstance(parsed, list):
        return parsed
    return []


def _parse_memory_settings(raw: str | None):
    """Parse memory_settings JSON column back to SessionMemorySettings."""
    if not raw:
        return None
    try:
        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        return SessionMemorySettings.from_dict(json.loads(raw))
    except Exception:
        log.warning("memory_settings_parse_failed", exc_info=True)
        return None


def _serialize_memory_settings(ms: object | None) -> str | None:
    """Serialize SessionMemorySettings to JSON string, or None."""
    if ms is None:
        return None
    try:
        return json.dumps(ms.to_dict())
    except Exception:
        log.warning("memory_settings_serialize_failed", exc_info=True)
        return None


# Per-entry ``context_blocks`` carries the full augmented context for a turn
# (~135 KB). With a 50-entry ring buffer that compounds to a 4.5 MB JSON blob
# that has to be serialized + INSERT-OR-REPLACE'd into one SQLite row on
# every ``_persist_state`` call — and json.dumps + the SQLite write are both
# synchronous on the asyncio loop, so every narrative turn was blocking the
# loop for several seconds. Auth, polling, other tabs all stalled.
#
# The inspector UI only needs ``context_blocks`` on the most recent few
# entries (you scrub through prev/next from the latest turn outward). Keep
# it on the last ``_REQUEST_LOG_FULL_TAIL`` entries; strip it from older
# ones before serialize.
_REQUEST_LOG_FULL_TAIL = 3


def _slim_request_logs_for_persist(logs: list[dict]) -> list[dict]:
    """Strip the bulky ``context_blocks`` field from older request logs.

    Preserves the most recent ``_REQUEST_LOG_FULL_TAIL`` entries verbatim so
    the inspector can show their context. Older entries keep all the small
    fields (timestamp, model, token totals, etc.) — only the heavy field
    is dropped, with a sentinel left behind so the UI can render a
    "context elided" placeholder rather than crash.
    """
    if not logs:
        return logs
    keep_full_from = max(0, len(logs) - _REQUEST_LOG_FULL_TAIL)
    slim: list[dict] = []
    for i, entry in enumerate(logs):
        if i >= keep_full_from or "context_blocks" not in entry:
            slim.append(entry)
            continue
        # Shallow-copy + replace the heavy field. Don't mutate the original
        # because ``state.request_logs`` is the live in-memory ring buffer
        # the inspector reads from.
        copy = dict(entry)
        copy["context_blocks"] = None
        copy["context_blocks_elided"] = True
        slim.append(copy)
    return slim


class NarrativePersistence:
    """Reads and writes NarrativeSessionState to/from a SQLite database.

    Expects that the database already has the schema from migration
    002_narrative_state.sql applied.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_session_state(
        self, session_id: str, state: NarrativeSessionState, *, user_id: str = ""
    ) -> None:
        """Persist a full NarrativeSessionState to the database.

        Uses INSERT OR REPLACE (upsert) so this is safe to call repeatedly.
        Requires user_id — every narrative_* row is user-scoped.
        """
        if not user_id:
            raise ValueError("narrative state save requires user_id")
        await self._save_entities(session_id, state.entities, user_id=user_id)
        await self._save_facts(session_id, state.facts, user_id=user_id)
        await self._save_plot_threads(session_id, state.plot_threads, user_id=user_id)
        await self._save_contradictions(session_id, state.contradictions, user_id=user_id)
        await self._save_lorebook_entries(session_id, state.lorebook, user_id=user_id)
        await self._save_memory(session_id, state, user_id=user_id)
        await self._save_relationships(session_id, state, user_id=user_id)

        await self._conn.commit()
        log.debug(
            "narrative_state_saved",
            session_id=session_id,
            entities=len(state.entities),
            facts=len(state.facts),
            plot_threads=len(state.plot_threads),
            contradictions=len(state.contradictions),
            lorebook=len(state.lorebook),
        )

    async def load_session_state(
        self, session_id: str, *, user_id: str = ""
    ) -> NarrativeSessionState | None:
        """Load a NarrativeSessionState from the database.

        Returns None if no data exists for this session.
        """
        entities = await self._load_entities(session_id, user_id=user_id)
        facts = await self._load_facts(session_id, user_id=user_id)
        plot_threads = await self._load_plot_threads(session_id, user_id=user_id)
        contradictions = await self._load_contradictions(session_id, user_id=user_id)
        lorebook = await self._load_lorebook_entries(session_id, user_id=user_id)

        # Load long-term memory (three-layer: STATE + MEMORY + archive)
        memory_data = await self._load_memory(session_id, user_id=user_id)

        # Load character card name from character_cards table
        character_card_name = await self._load_character_card_name(session_id, user_id=user_id)

        # Load relationships
        relationships = await self._load_relationships(session_id, user_id=user_id)

        # If nothing was found across ALL stores, this session has no narrative state
        has_tracker_data = (entities or facts or plot_threads
                           or contradictions or lorebook)
        has_memory_data = memory_data is not None
        has_relationships = bool(relationships)

        if not has_tracker_data and not has_memory_data and not character_card_name and not has_relationships:
            return None

        # Determine branch_id and message_count from persisted data
        branch_id = "main"
        message_count = 0

        # Try trackers first (legacy path)
        if facts:
            branch_id = facts[0].branch_id
            message_count = max(message_count, max(f.established_at for f in facts) + 1)
        if plot_threads:
            branch_id = plot_threads[0].branch_id
            for pt in plot_threads:
                at = pt.resolved_at if pt.resolved_at is not None else pt.established_at
                message_count = max(message_count, at + 1)
        if contradictions:
            message_count = max(
                message_count, max(c.message_index for c in contradictions) + 1
            )
        if entities:
            first_entity = next(iter(entities.values()))
            branch_id = first_entity.branch_id

        state = NarrativeSessionState(
            session_id=session_id,
            branch_id=branch_id,
            message_count=message_count,
            character_card_name=character_card_name,
            entities=entities,
            facts=facts,
            plot_threads=plot_threads,
            contradictions=contradictions,
            lorebook=lorebook,
            relationships=relationships,
        )

        if memory_data:
            md = memory_data  # now a dict
            state.card_type = md["card_type"]
            state.memory_summary = md["memory_summary"]
            state.last_summary_at = md["last_summary_at"]
            state.overflow_summaries = md["overflow_summaries"]
            state.state_snapshot_data = md["state_snapshot_data"]
            state.memory_ledger_data = md["memory_ledger_data"]
            state.branch_states_data = md["branch_states_data"]
            state.message_history_data = md["message_history_data"]
            state.needs_compaction = md["needs_compaction"]
            state.request_logs = md.get("request_logs") or []
            state.memory_settings = md.get("memory_settings")
            state.world_state = md.get("world_state") or {}

            # Recover message_count — use the most reliable source
            # Priority: persisted count > last_summary_at > ledger max round > tracker-derived
            persisted_count = md["message_count"]
            if persisted_count > state.message_count:
                state.message_count = persisted_count
            if md["last_summary_at"] > state.message_count:
                state.message_count = md["last_summary_at"]
            ledger = md["memory_ledger_data"]
            if ledger:
                max_round = max(e.get("round_num", 0) for e in ledger)
                if max_round > state.message_count:
                    state.message_count = max_round

        return state

    async def save_incremental(
        self,
        session_id: str,
        state: NarrativeSessionState,
        message_index: int,
        *,
        user_id: str = "",
    ) -> None:
        """Save only state changes from the given message index onward.

        For entities, we always upsert all of them (they may have been
        mutated).  For facts, plot threads, contradictions, etc., we only
        write records whose relevant timestamp is >= message_index.
        Requires user_id — every narrative_* row is user-scoped.
        """
        if not user_id:
            raise ValueError("narrative incremental save requires user_id")
        # Entities may have been modified at any point, upsert all
        await self._save_entities(session_id, state.entities, user_id=user_id)

        # Facts: only new or modified since message_index
        new_facts = [f for f in state.facts if f.established_at >= message_index]
        if new_facts:
            await self._save_facts(session_id, new_facts, user_id=user_id)

        # Plot threads: only those established or resolved since message_index
        new_plots = [
            p
            for p in state.plot_threads
            if p.established_at >= message_index
            or (p.resolved_at is not None and p.resolved_at >= message_index)
        ]
        if new_plots:
            await self._save_plot_threads(session_id, new_plots, user_id=user_id)

        # Contradictions: delete any at/after the index, then insert new ones
        new_contradictions = [
            c for c in state.contradictions if c.message_index >= message_index
        ]
        if new_contradictions:
            query = "DELETE FROM contradictions WHERE session_id = ? AND message_index >= ?"
            params: list = [session_id, message_index]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            await self._conn.execute(query, params)
            await self._insert_contradictions(new_contradictions, user_id=user_id)

        # Lorebook: upsert all (trigger counts may have changed)
        await self._save_lorebook_entries(session_id, state.lorebook, user_id=user_id)

        await self._conn.commit()
        log.debug(
            "narrative_state_incremental_saved",
            session_id=session_id,
            from_index=message_index,
            facts=len(new_facts),
            plots=len(new_plots),
            contradictions=len(new_contradictions),
        )

    # ------------------------------------------------------------------
    # Entity persistence
    # ------------------------------------------------------------------

    async def _save_entities(
        self, session_id: str, entities: dict[str, Entity], *, user_id: str = ""
    ) -> None:
        if not entities:
            return
        if user_id:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO entities
                   (id, session_id, entity_type, name, aliases, state, branch_id, user_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                [
                    (
                        entity.id,
                        session_id,
                        entity.entity_type.value,
                        entity.name,
                        json.dumps(entity.aliases),
                        json.dumps(entity.state.to_dict()),
                        entity.branch_id,
                        user_id,
                    )
                    for entity in entities.values()
                ],
            )
        else:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO entities
                   (id, session_id, entity_type, name, aliases, state, branch_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                [
                    (
                        entity.id,
                        session_id,
                        entity.entity_type.value,
                        entity.name,
                        json.dumps(entity.aliases),
                        json.dumps(entity.state.to_dict()),
                        entity.branch_id,
                    )
                    for entity in entities.values()
                ],
            )

    async def _load_entities(self, session_id: str, *, user_id: str = "") -> dict[str, Entity]:
        query = "SELECT * FROM entities WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        entities: dict[str, Entity] = {}
        for row in rows:
            row_dict = dict(row)
            entity = Entity(
                id=row_dict["id"],
                session_id=row_dict["session_id"],
                entity_type=EntityType(row_dict["entity_type"]),
                name=row_dict["name"],
                aliases=json.loads(row_dict["aliases"]),
                state=EntityState.from_dict(json.loads(row_dict["state"])),
                branch_id=row_dict["branch_id"],
            )
            entities[entity.id] = entity
        return entities

    # ------------------------------------------------------------------
    # Fact persistence
    # ------------------------------------------------------------------

    async def _save_facts(self, session_id: str, facts: list[Fact], *, user_id: str = "") -> None:
        if not facts:
            return
        # Batch upsert all facts
        if user_id:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO facts
                   (id, session_id, content, source, confidence, domain,
                    established_at, superseded_by, branch_id, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        fact.id,
                        session_id,
                        fact.content,
                        fact.source,
                        fact.confidence,
                        fact.domain,
                        fact.established_at,
                        fact.superseded_by,
                        fact.branch_id,
                        user_id,
                    )
                    for fact in facts
                ],
            )
        else:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO facts
                   (id, session_id, content, source, confidence, domain,
                    established_at, superseded_by, branch_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        fact.id,
                        session_id,
                        fact.content,
                        fact.source,
                        fact.confidence,
                        fact.domain,
                        fact.established_at,
                        fact.superseded_by,
                        fact.branch_id,
                    )
                    for fact in facts
                ],
            )
        # Batch delete existing tags for all facts
        fact_ids = [fact.id for fact in facts]
        placeholders = ",".join("?" * len(fact_ids))
        await self._conn.execute(
            f"DELETE FROM fact_tags WHERE fact_id IN ({placeholders})",
            fact_ids,
        )
        # Batch insert all tags
        tag_rows = [
            (fact.id, tag)
            for fact in facts
            for tag in fact.tags
        ]
        if tag_rows:
            await self._conn.executemany(
                "INSERT INTO fact_tags (fact_id, tag) VALUES (?, ?)",
                tag_rows,
            )

    async def _load_facts(self, session_id: str, *, user_id: str = "") -> list[Fact]:
        # Use LEFT JOIN to fetch facts and their tags in a single query
        query = (
            "SELECT f.*, ft.tag FROM facts f "
            "LEFT JOIN fact_tags ft ON ft.fact_id = f.id "
            "WHERE f.session_id = ?"
        )
        params: list = [session_id]
        if user_id:
            query += " AND f.user_id = ?"
            params.append(user_id)
        query += " ORDER BY f.id"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        # Group tags by fact id
        facts_dict: dict[str, tuple[dict, list[str]]] = {}
        fact_order: list[str] = []
        for row in rows:
            row_dict = dict(row)
            fact_id = row_dict["id"]
            tag = row_dict.pop("tag", None)
            if fact_id not in facts_dict:
                facts_dict[fact_id] = (row_dict, [])
                fact_order.append(fact_id)
            if tag is not None:
                facts_dict[fact_id][1].append(tag)

        facts: list[Fact] = []
        for fact_id in fact_order:
            row_dict, tags = facts_dict[fact_id]
            facts.append(
                Fact(
                    id=row_dict["id"],
                    session_id=row_dict["session_id"],
                    content=row_dict["content"],
                    source=row_dict["source"],
                    confidence=row_dict["confidence"],
                    domain=row_dict["domain"],
                    established_at=row_dict["established_at"],
                    superseded_by=row_dict["superseded_by"],
                    branch_id=row_dict["branch_id"],
                    tags=tags,
                )
            )
        return facts

    # ------------------------------------------------------------------
    # Plot thread persistence
    # ------------------------------------------------------------------

    async def _save_plot_threads(
        self, session_id: str, threads: list[PlotThread], *, user_id: str = ""
    ) -> None:
        # Batch via executemany — one queue entry on aiosqlite's worker thread
        # for the whole list, instead of N. The previous per-row loop pinned
        # the worker thread for the duration of the chat's narrative save and
        # serialized auth/state queries from other clients behind it.
        if not threads:
            return
        if user_id:
            params = [
                (
                    t.id, session_id, t.title, t.description,
                    t.status.value, t.established_at, t.resolved_at,
                    t.branch_id, json.dumps(t.state), user_id,
                )
                for t in threads
            ]
            await self._conn.executemany(
                """INSERT OR REPLACE INTO plot_threads
                   (id, session_id, title, description, status,
                    established_at, resolved_at, branch_id, state, user_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                params,
            )
        else:
            params = [
                (
                    t.id, session_id, t.title, t.description,
                    t.status.value, t.established_at, t.resolved_at,
                    t.branch_id, json.dumps(t.state),
                )
                for t in threads
            ]
            await self._conn.executemany(
                """INSERT OR REPLACE INTO plot_threads
                   (id, session_id, title, description, status,
                    established_at, resolved_at, branch_id, state, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                params,
            )

    async def _load_plot_threads(self, session_id: str, *, user_id: str = "") -> list[PlotThread]:
        query = "SELECT * FROM plot_threads WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        threads: list[PlotThread] = []
        for row in rows:
            row_dict = dict(row)
            threads.append(
                PlotThread(
                    id=row_dict["id"],
                    session_id=row_dict["session_id"],
                    title=row_dict["title"],
                    description=row_dict["description"],
                    status=PlotStatus(row_dict["status"]),
                    established_at=row_dict["established_at"],
                    resolved_at=row_dict["resolved_at"],
                    branch_id=row_dict["branch_id"],
                    state=json.loads(row_dict["state"]),
                )
            )
        return threads

    # ------------------------------------------------------------------
    # Contradiction persistence
    # ------------------------------------------------------------------

    async def _save_contradictions(
        self, session_id: str, contradictions: list[Contradiction], *, user_id: str = ""
    ) -> None:
        # Contradictions use autoincrement PK so we delete-and-reinsert
        # to avoid duplicates on repeated saves.
        query = "DELETE FROM contradictions WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        await self._conn.execute(query, params)
        await self._insert_contradictions(contradictions, user_id=user_id)

    async def _insert_contradictions(
        self, contradictions: list[Contradiction], *, user_id: str = ""
    ) -> None:
        # Batched via executemany — see _save_plot_threads for why.
        if not contradictions:
            return
        if user_id:
            params = [
                (
                    c.session_id, c.message_index, c.contradiction_type,
                    c.description, c.severity.value, c.resolution,
                    json.dumps(c.fact_ids), c.branch_id, user_id,
                )
                for c in contradictions
            ]
            await self._conn.executemany(
                """INSERT INTO contradictions
                   (session_id, message_index, contradiction_type, description,
                    severity, resolution, fact_ids, branch_id, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                params,
            )
        else:
            params = [
                (
                    c.session_id, c.message_index, c.contradiction_type,
                    c.description, c.severity.value, c.resolution,
                    json.dumps(c.fact_ids), c.branch_id,
                )
                for c in contradictions
            ]
            await self._conn.executemany(
                """INSERT INTO contradictions
                   (session_id, message_index, contradiction_type, description,
                    severity, resolution, fact_ids, branch_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                params,
            )

    async def _load_contradictions(self, session_id: str, *, user_id: str = "") -> list[Contradiction]:
        query = "SELECT * FROM contradictions WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        contradictions: list[Contradiction] = []
        for row in rows:
            row_dict = dict(row)
            contradictions.append(
                Contradiction(
                    session_id=row_dict["session_id"],
                    message_index=row_dict["message_index"],
                    contradiction_type=row_dict["contradiction_type"],
                    description=row_dict["description"],
                    severity=ContradictionSeverity(row_dict["severity"]),
                    resolution=row_dict["resolution"],
                    fact_ids=json.loads(row_dict["fact_ids"]),
                    branch_id=row_dict["branch_id"],
                )
            )
        return contradictions

    # ------------------------------------------------------------------
    # Lorebook entry persistence
    # ------------------------------------------------------------------

    async def _save_lorebook_entries(
        self, session_id: str, entries: list[LorebookEntry], *, user_id: str = ""
    ) -> None:
        """Persist lorebook entries.

        With 84-entry default lorebooks (default RPG world ships ~84) and
        ``_persist_state`` running 2-3× per turn, the prior per-entry
        ``await execute`` loop did 84 × 3 = 250+ aiosqlite round-trips per
        turn — each adding worker-thread queue + context-switch latency on
        the shared connection, blocking unrelated coroutines (auth, polling
        endpoints) waiting on the same connection. Now: one ``executemany``
        for the v2 schema; on schema-version failure, fall back to the
        legacy per-entry path (rare — only triggers on a never-migrated
        DB).
        """
        if not entries:
            return

        def _v2_row(entry: "LorebookEntry") -> tuple:
            row = [
                entry.id,
                session_id,
                json.dumps(entry.keywords),
                entry.content,
                entry.priority,
                entry.source,
                1 if entry.enabled else 0,
                1 if entry.constant else 0,
                entry.position.value,
                entry.scan_depth,
                1 if entry.case_sensitive else 0,
                entry.sticky_turns,
                entry.cooldown_turns,
                entry.last_triggered_at,
                entry.trigger_count,
                json.dumps(entry.secondary_keywords),
                1 if entry.selective else 0,
                entry.selective_logic.value,
                entry.group,
                1 if entry.group_override else 0,
                entry.group_weight,
                entry.probability,
                1 if entry.use_probability else 0,
                1 if entry.ignore_budget else 0,
                None if entry.match_whole_words is None else (1 if entry.match_whole_words else 0),
                None if entry.use_group_scoring is None else (1 if entry.use_group_scoring else 0),
                1 if entry.exclude_recursion else 0,
                1 if entry.prevent_recursion else 0,
                entry.delay_until_recursion,
                1 if entry.match_persona else 0,
                1 if entry.match_char_description else 0,
                1 if entry.match_char_personality else 0,
                1 if entry.match_scenario else 0,
                1 if entry.match_creator_notes else 0,
                entry.delay_turns,
                entry.outlet_name,
                entry.comment,
                entry.branch_id,
            ]
            if user_id:
                row.append(user_id)
            return tuple(row)

        v2_cols = (
            "id, session_id, keywords, content, priority, source,"
            " enabled, constant, position, scan_depth, case_sensitive,"
            " sticky_turns, cooldown_turns, last_triggered_at, trigger_count,"
            " secondary_keywords, selective, selective_logic,"
            " group_name, group_override, group_weight,"
            " probability, use_probability, ignore_budget,"
            " match_whole_words, use_group_scoring,"
            " exclude_recursion, prevent_recursion, delay_until_recursion,"
            " match_persona, match_char_description, match_char_personality,"
            " match_scenario, match_creator_notes,"
            " delay_turns, outlet_name, comment, branch_id"
        )
        if user_id:
            v2_cols += ", user_id"
        v2_rows = [_v2_row(e) for e in entries]
        v2_placeholders = ",".join("?" * len(v2_rows[0]))

        try:
            await self._conn.executemany(
                f"INSERT OR REPLACE INTO lorebook_entries ({v2_cols}) VALUES ({v2_placeholders})",
                v2_rows,
            )
            return
        except Exception:
            # v2 schema not present on this DB — fall back to per-entry
            # legacy path. This was the original behaviour; keep it for
            # any never-migrated install.
            log.warning("lorebook_save_v2_batch_failed",
                        session_id=session_id, count=len(entries), exc_info=True)

        legacy_cols = (
            "id, session_id, keywords, content, priority, source,"
            " enabled, constant, position, scan_depth, case_sensitive,"
            " sticky_turns, cooldown_turns, last_triggered_at, trigger_count"
        )
        if user_id:
            legacy_cols += ", user_id"
        legacy_placeholders = ",".join("?" * (16 if user_id else 15))

        legacy_rows = []
        for entry in entries:
            row = [
                entry.id,
                session_id,
                json.dumps(entry.keywords),
                entry.content,
                entry.priority,
                entry.source,
                1 if entry.enabled else 0,
                1 if entry.constant else 0,
                entry.position.value,
                entry.scan_depth,
                1 if entry.case_sensitive else 0,
                entry.sticky_turns,
                entry.cooldown_turns,
                entry.last_triggered_at,
                entry.trigger_count,
            ]
            if user_id:
                row.append(user_id)
            legacy_rows.append(tuple(row))

        try:
            await self._conn.executemany(
                f"INSERT OR REPLACE INTO lorebook_entries ({legacy_cols}) VALUES ({legacy_placeholders})",
                legacy_rows,
            )
        except Exception:
            log.warning("lorebook_save_legacy_batch_failed",
                        session_id=session_id, count=len(entries), exc_info=True)

    async def _load_lorebook_entries(self, session_id: str, *, user_id: str = "") -> list[LorebookEntry]:
        query = "SELECT * FROM lorebook_entries WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        entries: list[LorebookEntry] = []
        for row in rows:
            d = dict(row)

            # Parse optional bool columns (None = inherit global)
            def _opt_bool(val: object) -> bool | None:
                if val is None:
                    return None
                return bool(val)

            # Secondary keywords — graceful if column missing (pre-migration)
            try:
                secondary_keywords = json.loads(d.get("secondary_keywords") or "[]")
            except (json.JSONDecodeError, TypeError):
                secondary_keywords = []

            try:
                selective_logic = SelectiveLogic(d.get("selective_logic", 0))
            except (ValueError, TypeError):
                selective_logic = SelectiveLogic.AND_ANY

            entries.append(
                LorebookEntry(
                    id=d["id"],
                    session_id=d.get("session_id", ""),
                    keywords=json.loads(d["keywords"]),
                    content=d["content"],
                    priority=d["priority"],
                    source=d["source"],
                    enabled=bool(d["enabled"]),
                    constant=bool(d["constant"]),
                    position=LorebookPosition(d["position"]),
                    scan_depth=d["scan_depth"],
                    case_sensitive=bool(d["case_sensitive"]),
                    sticky_turns=d["sticky_turns"],
                    cooldown_turns=d["cooldown_turns"],
                    last_triggered_at=d["last_triggered_at"],
                    trigger_count=d["trigger_count"],
                    # V2 fields — use .get() for graceful migration handling
                    secondary_keywords=secondary_keywords,
                    selective=bool(d.get("selective", 1)),
                    selective_logic=selective_logic,
                    group=d.get("group_name", ""),  # DB column is group_name
                    group_override=bool(d.get("group_override", 0)),
                    group_weight=d.get("group_weight", 100),
                    probability=d.get("probability", 100),
                    use_probability=bool(d.get("use_probability", 1)),
                    ignore_budget=bool(d.get("ignore_budget", 0)),
                    match_whole_words=_opt_bool(d.get("match_whole_words")),
                    use_group_scoring=_opt_bool(d.get("use_group_scoring")),
                    exclude_recursion=bool(d.get("exclude_recursion", 0)),
                    prevent_recursion=bool(d.get("prevent_recursion", 0)),
                    delay_until_recursion=d.get("delay_until_recursion", 0),
                    match_persona=bool(d.get("match_persona", 0)),
                    match_char_description=bool(d.get("match_char_description", 0)),
                    match_char_personality=bool(d.get("match_char_personality", 0)),
                    match_scenario=bool(d.get("match_scenario", 0)),
                    match_creator_notes=bool(d.get("match_creator_notes", 0)),
                    delay_turns=d.get("delay_turns", 0),
                    outlet_name=d.get("outlet_name", ""),
                    comment=d.get("comment", ""),
                    # Branch tag — graceful for pre-migration-304 rows.
                    branch_id=d.get("branch_id") or "main",
                )
            )
        return entries

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Character card persistence
    # ------------------------------------------------------------------

    async def _load_character_card_name(self, session_id: str, *, user_id: str = "") -> str:
        query = "SELECT name FROM character_cards WHERE session_id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if row is None:
            return ""
        return dict(row)["name"]

    # ------------------------------------------------------------------
    # Narrative memory persistence
    # ------------------------------------------------------------------

    async def _save_memory(
        self, session_id: str, state: NarrativeSessionState, *, user_id: str = ""
    ) -> None:
        """Persist long-term memory summary + overflow data.  Graceful if table missing."""
        try:
            cols = (
                "session_id, card_type, memory_summary, last_summary_at,"
                " overflow_summaries, archived_messages, state_snapshot, memory_ledger,"
                " branch_states, message_count, message_history, graph_summary,"
                " needs_compaction, last_request_log, memory_settings, world_state,"
                " updated_at"
            )
            vals: list = [
                session_id,
                state.card_type,
                state.memory_summary,
                state.last_summary_at,
                json.dumps(state.overflow_summaries),
                "[]",
                json.dumps(getattr(state, 'state_snapshot_data', {}) or {}),
                json.dumps(getattr(state, 'memory_ledger_data', []) or []),
                json.dumps(getattr(state, 'branch_states_data', {}) or {}),
                state.message_count,
                json.dumps(getattr(state, 'message_history_data', []) or []),
                '',  # graph_summary column (legacy, KG removed)
                1 if getattr(state, 'needs_compaction', False) else 0,
                json.dumps(_slim_request_logs_for_persist(
                    getattr(state, 'request_logs', []) or [])),
                _serialize_memory_settings(getattr(state, 'memory_settings', None)),
                json.dumps(getattr(state, 'world_state', {}) or {}),
            ]
            if user_id:
                cols += ", user_id"
                vals.append(user_id)
            # updated_at is datetime('now') in the SQL expression, handled via column default
            placeholders = ",".join(["?"] * 16 + ["datetime('now')"])
            if user_id:
                placeholders += ",?"
            await self._conn.execute(
                f"INSERT OR REPLACE INTO narrative_memory ({cols}) VALUES ({placeholders})",
                vals,
            )
        except Exception:
            # Table may not exist yet or may lack new columns — fall back
            log.warning("narrative_memory_save_full_failed", session_id=session_id, exc_info=True)
            try:
                fb_cols = "session_id, card_type, memory_summary, last_summary_at, updated_at"
                fb_vals: list = [
                    session_id,
                    state.card_type,
                    state.memory_summary,
                    state.last_summary_at,
                ]
                if user_id:
                    fb_cols += ", user_id"
                    fb_ph = "?, ?, ?, ?, datetime('now'), ?"
                    fb_vals.append(user_id)
                else:
                    fb_ph = "?, ?, ?, ?, datetime('now')"
                await self._conn.execute(
                    f"INSERT OR REPLACE INTO narrative_memory ({fb_cols}) VALUES ({fb_ph})",
                    fb_vals,
                )
            except Exception:
                log.warning("narrative_memory_save_fallback_failed", session_id=session_id, exc_info=True)

    async def _load_memory(
        self, session_id: str, *, user_id: str = ""
    ) -> dict | None:
        """Load long-term memory. Returns a dict with all fields or None."""
        try:
            query = (
                "SELECT card_type, memory_summary, last_summary_at, "
                "overflow_summaries, state_snapshot, memory_ledger, "
                "branch_states, message_count, message_history, "
                "graph_summary, needs_compaction, last_request_log, memory_settings, "
                "world_state "
                "FROM narrative_memory WHERE session_id = ?"
            )
            params: list = [session_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            d = dict(row)
            return {
                "card_type": d["card_type"],
                "memory_summary": d["memory_summary"],
                "last_summary_at": d["last_summary_at"],
                "overflow_summaries": json.loads(d.get("overflow_summaries") or "[]"),
                "state_snapshot_data": json.loads(d.get("state_snapshot") or "{}"),
                "memory_ledger_data": json.loads(d.get("memory_ledger") or "[]"),
                "branch_states_data": json.loads(d.get("branch_states") or "{}"),
                "message_count": d.get("message_count", 0) or 0,
                "message_history_data": json.loads(d.get("message_history") or "[]"),
                "needs_compaction": bool(d.get("needs_compaction", 0)),
                "request_logs": _parse_request_logs(d.get("last_request_log")),
                "memory_settings": _parse_memory_settings(d.get("memory_settings")),
                "world_state": json.loads(d.get("world_state") or "{}"),
            }
        except Exception:
            # Table may not exist yet or lacks new columns — try legacy
            try:
                fb_query = (
                    "SELECT card_type, memory_summary, last_summary_at "
                    "FROM narrative_memory WHERE session_id = ?"
                )
                fb_params: list = [session_id]
                if user_id:
                    fb_query += " AND user_id = ?"
                    fb_params.append(user_id)
                cursor = await self._conn.execute(fb_query, fb_params)
                row = await cursor.fetchone()
                if row is None:
                    return None
                row_dict = dict(row)
                return {
                    "card_type": row_dict["card_type"],
                    "memory_summary": row_dict["memory_summary"],
                    "last_summary_at": row_dict["last_summary_at"],
                    "overflow_summaries": [],
                    "state_snapshot_data": {},
                    "memory_ledger_data": [],
                    "branch_states_data": {},
                    "message_count": 0,
                    "message_history_data": [],
                    "needs_compaction": False,
                    "memory_settings": None,
                }
            except Exception:
                return None

    # ------------------------------------------------------------------
    # Relationship persistence
    # ------------------------------------------------------------------

    async def _save_relationships(
        self, session_id: str, state: NarrativeSessionState, *, user_id: str = ""
    ) -> None:
        """Persist relationship tracker data to character_relationships table.

        Each ``await self._conn.execute(...)`` is a full round-trip through
        aiosqlite's per-connection worker thread; the previous DELETE + 12×
        sequential INSERT loop accounted for ~13 round-trips per call (~100-
        200ms wall time), and ``_persist_state`` runs this twice per turn
        (post-stream + post-extraction). Now: 1 DELETE + 1 ``executemany``
        = 2 round-trips, regardless of relationship count.
        """
        relationships = state.relationships
        if not relationships:
            return
        try:
            # Stale-cleanup DELETE (kept — the new state may have FEWER
            # relationships than the prior snapshot, so we can't rely on
            # INSERT OR REPLACE alone).
            del_query = "DELETE FROM character_relationships WHERE session_id = ?"
            del_params: list = [session_id]
            if user_id:
                del_query += " AND user_id = ?"
                del_params.append(user_id)
            await self._conn.execute(del_query, del_params)

            # Batch insert via executemany — single thread round-trip.
            if user_id:
                rows = [
                    (
                        session_id,
                        rel["source"],
                        rel["target"],
                        rel["trust"],
                        rel["affection"],
                        rel["tension"],
                        rel.get("label", ""),
                        rel.get("last_updated_at", 0),
                        user_id,
                    )
                    for rel in relationships
                ]
                await self._conn.executemany(
                    """INSERT INTO character_relationships
                       (session_id, source_entity, target_entity, trust, affection,
                        tension, label, last_updated_at, user_id, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    rows,
                )
            else:
                rows = [
                    (
                        session_id,
                        rel["source"],
                        rel["target"],
                        rel["trust"],
                        rel["affection"],
                        rel["tension"],
                        rel.get("label", ""),
                        rel.get("last_updated_at", 0),
                    )
                    for rel in relationships
                ]
                await self._conn.executemany(
                    """INSERT INTO character_relationships
                       (session_id, source_entity, target_entity, trust, affection,
                        tension, label, last_updated_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    rows,
                )
        except Exception:
            log.warning("relationship_save_failed", session_id=session_id, exc_info=True)

    async def _load_relationships(self, session_id: str, *, user_id: str = "") -> list[dict]:
        """Load relationships from character_relationships table."""
        try:
            query = (
                "SELECT source_entity, target_entity, trust, affection, "
                "tension, label, last_updated_at "
                "FROM character_relationships WHERE session_id = ?"
            )
            params: list = [session_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                {
                    "source": dict(r)["source_entity"],
                    "target": dict(r)["target_entity"],
                    "trust": dict(r)["trust"],
                    "affection": dict(r)["affection"],
                    "tension": dict(r)["tension"],
                    "label": dict(r)["label"],
                    "last_updated_at": dict(r)["last_updated_at"],
                }
                for r in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Character card persistence
    # ------------------------------------------------------------------

    async def save_character_card(
        self,
        session_id: str,
        card_id: str,
        name: str,
        data: dict,
        source_format: str = "unknown",
        *,
        user_id: str = "",
    ) -> None:
        """Persist character card info for a session. Requires user_id."""
        if not user_id:
            raise ValueError("character_cards insert requires user_id")
        await self._conn.execute(
            """INSERT OR REPLACE INTO character_cards
               (id, session_id, name, data, source_format, user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (card_id, session_id, name, json.dumps(data), source_format, user_id),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Narrative archive persistence (pair-aware embedded exchanges)
    # ------------------------------------------------------------------

    async def store_archive_exchanges(
        self,
        session_id: str,
        exchanges: list[dict],
        *,
        user_id: str = "",
    ) -> None:
        """Store archived user+assistant exchange pairs with embeddings.

        Each exchange dict: {id, user_content, assistant_content, summary,
        turn_number, embedding_blob}. Requires user_id.
        """
        if not exchanges:
            return
        if not user_id:
            raise ValueError("narrative_archive insert requires user_id")
        # Batch into 2 round-trips (was 1 + N + M for N exchanges + M with
        # embeddings). Archive batches typically run every 5 exchanges, so
        # the per-exchange loop was 5-10 round-trips per archive run on
        # the shared aiosqlite worker thread.
        archive_rows = [
            (
                ex["id"], session_id,
                ex["user_content"], ex["assistant_content"],
                ex["summary"], ex["turn_number"], user_id,
            )
            for ex in exchanges
        ]
        vec_rows = [
            (ex["id"], ex["embedding_blob"])
            for ex in exchanges
            if ex.get("embedding_blob")
        ]
        try:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO narrative_archive
                   (id, session_id, user_content, assistant_content,
                    summary, turn_number, user_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                archive_rows,
            )
            if vec_rows:
                try:
                    await self._conn.executemany(
                        "INSERT OR REPLACE INTO narrative_archive_vec(id, embedding) VALUES (?, ?)",
                        vec_rows,
                    )
                except Exception:
                    # Vec extension may be unavailable or schema mismatch;
                    # the archive rows are already written, so this is
                    # non-fatal — log once for the whole batch.
                    log.warning("archive_vec_insert_batch_failed",
                                session_id=session_id, count=len(vec_rows),
                                exc_info=True)
            await self._conn.commit()
        except Exception:
            log.warning("archive_store_failed", session_id=session_id, exc_info=True)

    async def archive_etag(
        self,
        session_id: str,
        *,
        user_id: str = "",
    ) -> str:
        """Return a cache validator for the session's archive.

        Computed from ``MAX(turn_number)`` + ``COUNT(*)`` so an unchanged
        archive produces the same string between polls. Archive rows are
        append-only and delete-only — no in-place updates — so this pair is
        sufficient: an append bumps both MAX and COUNT; a delete drops the
        COUNT (and possibly MAX). Both aggregates resolve from existing
        indexes (``idx_narrative_archive_session_turn`` /
        ``idx_narrative_archive_user``), so the cost is dominated by index
        descent, not row I/O — orders of magnitude cheaper than the full
        ``list_archive_exchanges`` fetch this validator gates.

        Returns the empty string on any failure so the caller falls through
        to a normal (uncached) response rather than crashing the poll.
        """
        try:
            query = (
                "SELECT COALESCE(MAX(turn_number), -1) AS mx, COUNT(*) AS n "
                "FROM narrative_archive WHERE session_id = ?"
            )
            params: list = [session_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()
            if row is None:
                return ""
            return f"{row['mx']}-{row['n']}"
        except Exception:
            log.warning("archive_etag_failed", session_id=session_id, exc_info=True)
            return ""

    async def list_archive_exchanges(
        self,
        session_id: str,
        limit: int = 100,
        *,
        user_id: str = "",
    ) -> list[dict]:
        """List all archived exchanges for a session, ordered by turn number.

        Returns list of {id, user_content, assistant_content, summary, turn_number, created_at}.
        """
        try:
            query = (
                "SELECT id, user_content, assistant_content, summary, turn_number, created_at "
                "FROM narrative_archive WHERE session_id = ?"
            )
            params: list = [session_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY turn_number LIMIT ?"
            params.append(limit)
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            log.warning("archive_list_failed", session_id=session_id, exc_info=True)
            return []

    async def delete_archive_exchange(self, exchange_id: str, *, user_id: str = "") -> bool:
        """Delete a single archived exchange by ID."""
        if not user_id:
            raise ValueError("narrative_archive delete requires user_id")
        try:
            # Vec table is keyed by exchange id only — gate via the parent's
            # ownership so cross-tenant id collisions can't trigger deletes.
            try:
                await self._conn.execute(
                    "DELETE FROM narrative_archive_vec WHERE id = ("
                    "SELECT id FROM narrative_archive WHERE id = ? AND user_id = ?)",
                    (exchange_id, user_id),
                )
            except Exception:
                log.warning("archive_vec_delete_failed", id=exchange_id, exc_info=True)
            cursor = await self._conn.execute(
                "DELETE FROM narrative_archive WHERE id = ? AND user_id = ?",
                (exchange_id, user_id),
            )
            await self._conn.commit()
            return cursor.rowcount > 0
        except Exception:
            log.warning("archive_delete_exchange_failed", id=exchange_id, exc_info=True)
            return False

    async def retrieve_archive_exchanges(
        self,
        session_id: str,
        query_embedding_blob: bytes,
        limit: int = 5,
        *,
        user_id: str = "",
    ) -> list[dict]:
        """Retrieve archived exchanges by vector similarity for a session.

        Returns list of {user_content, assistant_content, summary, turn_number, distance}.
        """
        try:
            query = (
                "SELECT na.user_content, na.assistant_content, na.summary,"
                " na.turn_number, v.distance"
                " FROM narrative_archive_vec v"
                " JOIN narrative_archive na ON na.id = v.id"
                " WHERE na.session_id = ?"
                " AND v.embedding MATCH ?"
                " AND k = ?"
            )
            params: list = [session_id, query_embedding_blob, limit * 3]
            if user_id:
                query += " AND na.user_id = ?"
                params.append(user_id)
            query += " ORDER BY v.distance"
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            # Filter to session and take top N
            results = []
            for r in rows:
                rd = dict(r)
                results.append({
                    "user_content": rd["user_content"],
                    "assistant_content": rd["assistant_content"],
                    "summary": rd["summary"],
                    "turn_number": rd["turn_number"],
                    "distance": rd["distance"],
                })
                if len(results) >= limit:
                    break
            return results
        except Exception:
            log.warning("archive_retrieve_failed", session_id=session_id, exc_info=True)
            return []

    async def prune_archive_after_turn(self, session_id: str, turn_number: int, *, user_id: str = "") -> int:
        """Delete archived exchanges with turn_number > the given value.

        Used during branch rollback to remove archive entries from the
        deleted conversation path.  Returns count deleted.
        """
        if not user_id:
            raise ValueError("narrative_archive prune requires user_id")
        try:
            cursor = await self._conn.execute(
                "SELECT id FROM narrative_archive "
                "WHERE session_id = ? AND turn_number > ? AND user_id = ?",
                (session_id, turn_number, user_id),
            )
            rows = await cursor.fetchall()
            ids = [dict(r)["id"] for r in rows]

            for aid in ids:
                try:
                    await self._conn.execute(
                        "DELETE FROM narrative_archive_vec WHERE id = ?", (aid,),
                    )
                except Exception:
                    log.warning("archive_vec_delete_failed", id=aid, exc_info=True)

            cursor = await self._conn.execute(
                "DELETE FROM narrative_archive "
                "WHERE session_id = ? AND turn_number > ? AND user_id = ?",
                (session_id, turn_number, user_id),
            )
            await self._conn.commit()
            log.info("archive_pruned_for_branch", session_id=session_id, after_turn=turn_number, deleted=len(ids))
            return len(ids)
        except Exception:
            log.warning("archive_prune_failed", session_id=session_id, exc_info=True)
            return 0

    async def delete_session_archive(self, session_id: str, *, user_id: str = "") -> int:
        """Delete all archived exchanges for a session. Returns count deleted."""
        if not user_id:
            raise ValueError("narrative_archive session delete requires user_id")
        try:
            cursor = await self._conn.execute(
                "SELECT id FROM narrative_archive WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            rows = await cursor.fetchall()
            ids = [dict(r)["id"] for r in rows]

            # Delete from vec table
            for aid in ids:
                try:
                    await self._conn.execute(
                        "DELETE FROM narrative_archive_vec WHERE id = ?", (aid,),
                    )
                except Exception:
                    log.warning("archive_vec_delete_failed", id=aid, exc_info=True)

            cursor = await self._conn.execute(
                "DELETE FROM narrative_archive WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception:
            log.warning("archive_delete_failed", session_id=session_id, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Branch metadata (migration 115) — first-class branch graph
    # ------------------------------------------------------------------

    async def upsert_branch(
        self,
        session_id: str,
        branch_id: str,
        parent_branch_id: str | None,
        branch_point: int,
        *,
        user_id: str = "",
        status: str = "active",
    ) -> None:
        """INSERT-OR-IGNORE the branch row, then UPDATE last_visited_at.

        Called every time a branch is entered (created or visited).
        """
        if not user_id:
            raise ValueError("narrative_branches upsert requires user_id")
        try:
            # Self-defend against FK violations. ``narrative_branches``
            # has a FOREIGN KEY (session_id) REFERENCES sessions(id);
            # if the parent row in ``sessions`` doesn't exist yet, the
            # insert below 4xxs and the engine's first-frame state
            # never gets persisted. Many call paths into this method
            # don't have a chance to call StateManager.get_or_create_session
            # first (the shadow-persist task runs out-of-band on
            # engine attach, before the handler has a chance to
            # touch the sessions table). Cheap fix: idempotent
            # parent-row creation here so the upsert is unconditional.
            await self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, mode) VALUES (?, ?)",
                (session_id, "narrative"),
            )
            await self._conn.execute(
                """INSERT OR IGNORE INTO narrative_branches
                   (branch_id, session_id, parent_branch_id, branch_point, status, user_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (branch_id, session_id, parent_branch_id, branch_point, status, user_id),
            )
            await self._conn.execute(
                "UPDATE narrative_branches SET last_visited_at = datetime('now') "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            await self._conn.commit()
        except Exception:
            log.warning("narrative_branch_upsert_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)

    async def get_branch_ancestry(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> list[BranchAncestor]:
        """Walk parent_branch_id chain from leaf to main.

        Returns ordered list with the leaf first. Each entry's ``branch_point``
        is THIS branch's divergence point from its parent — matching the
        ``narrative_branches.branch_point`` schema convention.

        Retrieval functions read ``ancestry[i-1].branch_point`` to bound
        row visibility on ancestor i (the descendant's divergence point).
        """
        try:
            chain: list[BranchAncestor] = []
            current = branch_id
            # Bound the walk to prevent infinite loops on corrupt data
            for _ in range(64):
                query = ("SELECT parent_branch_id, branch_point FROM narrative_branches "
                         "WHERE session_id = ? AND branch_id = ?")
                params: list = [session_id, current]
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                cursor = await self._conn.execute(query, params)
                row = await cursor.fetchone()
                if row is None:
                    # Branch not found — degrade to a single-element chain
                    if not chain:
                        chain.append(BranchAncestor(branch_id=branch_id, branch_point=0))
                    break
                parent, my_branch_point = row[0], row[1]
                chain.append(BranchAncestor(branch_id=current, branch_point=my_branch_point))
                if parent is None:
                    break
                current = parent
            return chain
        except Exception:
            log.warning("narrative_branch_ancestry_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return [BranchAncestor(branch_id=branch_id, branch_point=0)]

    async def list_branches(
        self,
        session_id: str,
        *,
        user_id: str = "",
        include_stale: bool = True,
    ) -> list[BranchInfo]:
        """List all branches for a session ordered by created_at."""
        try:
            query = ("SELECT branch_id, parent_branch_id, branch_point, status, "
                     "created_at, last_visited_at FROM narrative_branches "
                     "WHERE session_id = ?")
            params: list = [session_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if not include_stale:
                query += " AND status != 'stale'"
            query += " ORDER BY created_at"
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                BranchInfo(
                    branch_id=r[0],
                    parent_branch_id=r[1],
                    branch_point=r[2],
                    status=r[3],
                    created_at=r[4],
                    last_visited_at=r[5],
                )
                for r in rows
            ]
        except Exception:
            log.warning("narrative_branch_list_failed", session_id=session_id, exc_info=True)
            return []

    async def set_branch_status(
        self,
        session_id: str,
        branch_id: str,
        status: str,
        *,
        user_id: str = "",
    ) -> bool:
        """Update branch lifecycle status. Returns True on success."""
        if not user_id:
            raise ValueError("narrative_branch status update requires user_id")
        if status not in _VALID_BRANCH_STATUSES:
            raise ValueError(f"invalid status: {status} (must be one of {_VALID_BRANCH_STATUSES})")
        try:
            cursor = await self._conn.execute(
                "UPDATE narrative_branches SET status = ? "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (status, session_id, branch_id, user_id),
            )
            await self._conn.commit()
            return cursor.rowcount > 0
        except Exception:
            log.warning("narrative_branch_status_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return False

    async def mark_stale_branches(
        self,
        session_id: str,
        *,
        user_id: str = "",
        threshold_days: int = 30,
    ) -> int:
        """Auto-mark branches as 'stale' when last_visited_at older than threshold.

        Never marks 'archived' branches (user-pinned); never marks 'main'.
        Returns count updated.
        """
        if not user_id:
            raise ValueError("mark_stale_branches requires user_id")
        try:
            cursor = await self._conn.execute(
                "UPDATE narrative_branches "
                "   SET status = 'stale' "
                " WHERE session_id = ? "
                "   AND user_id = ? "
                "   AND branch_id != 'main' "
                "   AND status = 'active' "
                "   AND last_visited_at < datetime('now', ?)",
                (session_id, user_id, f"-{int(threshold_days)} days"),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception:
            log.warning("mark_stale_branches_failed", session_id=session_id, exc_info=True)
            return 0

    async def has_branch_descendants(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> bool:
        """True if any branch has parent_branch_id = branch_id. Used by DELETE
        endpoint to enforce ?cascade=true requirement."""
        try:
            # user_id clause MUST come before LIMIT 1 — SQLite parses LIMIT
            # as a final clause; trailing AND lands in unscoped parser context.
            query = ("SELECT 1 FROM narrative_branches "
                     "WHERE session_id = ? AND parent_branch_id = ?")
            params: list = [session_id, branch_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " LIMIT 1"
            cursor = await self._conn.execute(query, params)
            return (await cursor.fetchone()) is not None
        except Exception:
            log.warning("has_branch_descendants_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # State snapshots (migration 116) — append-only STATE history
    # ------------------------------------------------------------------

    async def store_state_snapshot(
        self,
        session_id: str,
        branch_id: str,
        message_index: int,
        snapshot_data: str | dict,
        *,
        user_id: str = "",
    ) -> str:
        """Append a STATE snapshot row. Returns the new row's id (or '' on failure)."""
        if not user_id:
            raise ValueError("store_state_snapshot requires user_id")
        if isinstance(snapshot_data, dict):
            snapshot_json = json.dumps(snapshot_data)
        else:
            snapshot_json = snapshot_data
        snapshot_id = uuid.uuid4().hex
        try:
            await self._conn.execute(
                """INSERT INTO narrative_state_snapshots
                   (id, session_id, branch_id, message_index, snapshot_data, user_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (snapshot_id, session_id, branch_id, message_index, snapshot_json, user_id),
            )
            await self._conn.commit()
            return snapshot_id
        except Exception:
            log.warning("state_snapshot_store_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return ""

    async def get_state_snapshot_at(
        self,
        session_id: str,
        branch_ancestry: list[BranchAncestor],
        message_index: int,
        *,
        user_id: str = "",
    ) -> dict | None:
        """Most recent snapshot with message_index < N on the ancestry path.

        Walks ancestry: each ancestor contributes snapshots with message_index <
        its descendant's branch_point. The leaf has no upper bound beyond N.
        """
        if not branch_ancestry:
            return None
        try:
            clauses: list[str] = []
            params: list = [session_id]
            leaf = branch_ancestry[0]
            clauses.append("(branch_id = ? AND message_index < ?)")
            params.extend([leaf.branch_id, message_index])
            for i in range(1, len(branch_ancestry)):
                ancestor = branch_ancestry[i]
                descendant = branch_ancestry[i - 1]
                upper = min(message_index, descendant.branch_point)
                if upper <= 0:
                    continue
                clauses.append("(branch_id = ? AND message_index < ?)")
                params.extend([ancestor.branch_id, upper])
            if not clauses:
                return None
            query = (
                "SELECT snapshot_data FROM narrative_state_snapshots "
                "WHERE session_id = ? AND (" + " OR ".join(clauses) + ")"
            )
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY message_index DESC LIMIT 1"
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                log.warning("state_snapshot_json_parse_failed",
                            session_id=session_id, branch_id=leaf.branch_id)
                return None
        except Exception:
            log.warning("state_snapshot_get_failed",
                        session_id=session_id, exc_info=True)
            return None

    async def prune_state_snapshots_for_branch(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> int:
        """Delete all state snapshots for a single branch."""
        if not user_id:
            raise ValueError("prune_state_snapshots_for_branch requires user_id")
        try:
            cursor = await self._conn.execute(
                "DELETE FROM narrative_state_snapshots "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception:
            log.warning("state_snapshots_prune_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Ledger entries (migration 117) — branch-tagged LEDGER rows
    # ------------------------------------------------------------------

    async def store_ledger_entries(
        self,
        session_id: str,
        branch_id: str,
        entries: list[dict],
        *,
        user_id: str = "",
    ) -> int:
        """Bulk INSERT ledger entries. Each entry: {round_num, category, content}.

        Returns count inserted (0 on failure).
        """
        if not user_id:
            raise ValueError("store_ledger_entries requires user_id")
        if not entries:
            return 0
        try:
            rows = [
                (uuid.uuid4().hex, session_id, branch_id,
                 int(e.get("round_num", 0)),
                 str(e.get("category", "")),
                 str(e.get("content", "")),
                 user_id)
                for e in entries
            ]
            await self._conn.executemany(
                """INSERT INTO narrative_ledger_entries
                   (id, session_id, branch_id, round_num, category, content, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await self._conn.commit()
            return len(rows)
        except Exception:
            log.warning("ledger_entries_store_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return 0

    async def list_ledger_entries(
        self,
        session_id: str,
        branch_ancestry: list[BranchAncestor],
        *,
        user_id: str = "",
        max_round: int | None = None,
    ) -> list[dict]:
        """Ancestry-filtered ledger entries ordered by round_num ASC.

        Returns list of {round_num, category, content, branch_id}.
        """
        if not branch_ancestry:
            return []
        try:
            clauses: list[str] = []
            params: list = [session_id]
            leaf = branch_ancestry[0]
            if max_round is not None:
                clauses.append("(branch_id = ? AND round_num < ?)")
                params.extend([leaf.branch_id, max_round])
            else:
                clauses.append("(branch_id = ?)")
                params.append(leaf.branch_id)
            for i in range(1, len(branch_ancestry)):
                ancestor = branch_ancestry[i]
                descendant = branch_ancestry[i - 1]
                upper = descendant.branch_point
                if max_round is not None:
                    upper = min(upper, max_round)
                if upper <= 0:
                    continue
                clauses.append("(branch_id = ? AND round_num < ?)")
                params.extend([ancestor.branch_id, upper])
            if not clauses:
                return []
            query = (
                "SELECT round_num, category, content, branch_id "
                "FROM narrative_ledger_entries "
                "WHERE session_id = ? AND (" + " OR ".join(clauses) + ")"
            )
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY round_num ASC"
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                {"round_num": r[0], "category": r[1], "content": r[2], "branch_id": r[3]}
                for r in rows
            ]
        except Exception:
            log.warning("ledger_entries_list_failed",
                        session_id=session_id, exc_info=True)
            return []

    async def count_ledger_entries(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> int:
        """Count entries on a single branch (no ancestry walk).
        Used for compaction trigger."""
        try:
            query = ("SELECT COUNT(*) FROM narrative_ledger_entries "
                     "WHERE session_id = ? AND branch_id = ?")
            params: list = [session_id, branch_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            log.warning("ledger_entries_count_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return 0

    async def compact_ledger_entries(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
        keep_after_round: int,
        replacement_entries: list[dict],
    ) -> bool:
        """Atomically replace entries on this branch with round_num <
        keep_after_round by replacement_entries. True on success.

        Used by compaction: oldest 50% replaced with LLM-summarized versions.
        """
        if not user_id:
            raise ValueError("compact_ledger_entries requires user_id")
        try:
            await self._conn.execute("BEGIN IMMEDIATE")
            await self._conn.execute(
                "DELETE FROM narrative_ledger_entries "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ? AND round_num < ?",
                (session_id, branch_id, user_id, keep_after_round),
            )
            if replacement_entries:
                rows = [
                    (uuid.uuid4().hex, session_id, branch_id,
                     int(e.get("round_num", 0)),
                     str(e.get("category", "")),
                     str(e.get("content", "")),
                     user_id)
                    for e in replacement_entries
                ]
                await self._conn.executemany(
                    """INSERT INTO narrative_ledger_entries
                       (id, session_id, branch_id, round_num, category, content, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
            await self._conn.commit()
            return True
        except Exception:
            log.warning("ledger_entries_compact_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            try:
                await self._conn.rollback()
            except Exception as rb_exc:
                log.debug(
                    "ledger_entries_compact_rollback_failed",
                    session_id=session_id,
                    branch_id=branch_id,
                    error=str(rb_exc),
                )
            return False

    async def prune_ledger_entries_for_branch(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> int:
        """Delete all ledger entries for a single branch."""
        if not user_id:
            raise ValueError("prune_ledger_entries_for_branch requires user_id")
        try:
            cursor = await self._conn.execute(
                "DELETE FROM narrative_ledger_entries "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception:
            log.warning("ledger_entries_prune_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Branch-aware archive (extends migration 027 + 118)
    # ------------------------------------------------------------------

    async def store_archive_exchanges_for_branch(
        self,
        session_id: str,
        exchanges: list[dict],
        *,
        user_id: str = "",
        branch_id: str = "main",
    ) -> None:
        """Like store_archive_exchanges but tags rows with branch_id.

        Kept as a separate function to preserve backward compatibility for
        callers that haven't migrated to branch-aware archiving yet.
        """
        if not exchanges:
            return
        if not user_id:
            raise ValueError("narrative_archive insert requires user_id")
        # Batched — see store_archive_exchanges for rationale.
        archive_rows = [
            (
                ex["id"], session_id,
                ex["user_content"], ex["assistant_content"],
                ex["summary"], ex["turn_number"], branch_id, user_id,
            )
            for ex in exchanges
        ]
        vec_rows = [
            (ex["id"], ex["embedding_blob"])
            for ex in exchanges
            if ex.get("embedding_blob")
        ]
        try:
            await self._conn.executemany(
                """INSERT OR REPLACE INTO narrative_archive
                   (id, session_id, user_content, assistant_content,
                    summary, turn_number, branch_id, user_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                archive_rows,
            )
            if vec_rows:
                try:
                    await self._conn.executemany(
                        "INSERT OR REPLACE INTO narrative_archive_vec(id, embedding) VALUES (?, ?)",
                        vec_rows,
                    )
                except Exception:
                    log.warning("archive_vec_insert_batch_failed",
                                session_id=session_id, branch_id=branch_id,
                                count=len(vec_rows), exc_info=True)
            await self._conn.commit()
        except Exception:
            log.warning("archive_store_branch_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)

    async def retrieve_archive_for_branch(
        self,
        session_id: str,
        query_embedding_blob: bytes,
        branch_ancestry: list[BranchAncestor],
        *,
        user_id: str = "",
        limit: int = 5,
        max_turn: int | None = None,
    ) -> list[dict]:
        """Vector search bounded by branch ancestry.

        For the leaf branch, returns rows with branch_id=leaf bounded by
        max_turn (if set). For each strict ancestor, returns rows with
        branch_id=ancestor AND turn_number < descendant.branch_point.

        Returns list of {id, user_content, assistant_content, summary,
        turn_number, branch_id, distance}.
        """
        if not branch_ancestry:
            return []
        try:
            query = (
                "SELECT na.id, na.user_content, na.assistant_content, na.summary,"
                " na.turn_number, na.branch_id, v.distance"
                " FROM narrative_archive_vec v"
                " JOIN narrative_archive na ON na.id = v.id"
                " WHERE na.session_id = ?"
                " AND v.embedding MATCH ?"
                " AND k = ?"
            )
            k = max(limit * 3, limit + len(branch_ancestry) * 5)
            params: list = [session_id, query_embedding_blob, k]
            if user_id:
                query += " AND na.user_id = ?"
                params.append(user_id)
            query += " ORDER BY v.distance"
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()

            leaf = branch_ancestry[0]
            bounds: dict[str, int | None] = {leaf.branch_id: None}
            for i in range(1, len(branch_ancestry)):
                ancestor = branch_ancestry[i]
                descendant = branch_ancestry[i - 1]
                bounds[ancestor.branch_id] = descendant.branch_point

            results: list[dict] = []
            for r in rows:
                row_branch = r[5]
                row_turn = r[4]
                if row_branch not in bounds:
                    continue
                upper = bounds[row_branch]
                if upper is not None and row_turn >= upper:
                    continue
                if max_turn is not None and row_branch == leaf.branch_id and row_turn >= max_turn:
                    continue
                results.append({
                    "id": r[0],
                    "user_content": r[1],
                    "assistant_content": r[2],
                    "summary": r[3],
                    "turn_number": row_turn,
                    "branch_id": row_branch,
                    "distance": r[6],
                })
                if len(results) >= limit:
                    break
            return results
        except Exception:
            log.warning("archive_retrieve_branch_failed",
                        session_id=session_id, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Branch deletion cascade
    # ------------------------------------------------------------------

    async def delete_branch_cascade(
        self,
        session_id: str,
        branch_id: str,
        *,
        user_id: str = "",
    ) -> dict[str, int]:
        """Atomically delete a branch + its archive + ledger + snapshots.

        Returns count deleted per tier. Forbidden for branch_id='main' (caller
        should reject before calling). Vec-table cleanup is included.
        """
        if not user_id:
            raise ValueError("delete_branch_cascade requires user_id")
        if branch_id == "main":
            raise ValueError("cannot delete main branch")
        deleted = {"archive_rows": 0, "archive_vec_rows": 0,
                   "ledger_entries": 0, "snapshots": 0, "branches": 0}
        try:
            await self._conn.execute("BEGIN IMMEDIATE")

            # Vec rows first (they reference archive ids)
            cursor = await self._conn.execute(
                "DELETE FROM narrative_archive_vec WHERE id IN ("
                "SELECT id FROM narrative_archive "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?)",
                (session_id, branch_id, user_id),
            )
            deleted["archive_vec_rows"] = cursor.rowcount

            cursor = await self._conn.execute(
                "DELETE FROM narrative_archive "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            deleted["archive_rows"] = cursor.rowcount

            cursor = await self._conn.execute(
                "DELETE FROM narrative_ledger_entries "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            deleted["ledger_entries"] = cursor.rowcount

            cursor = await self._conn.execute(
                "DELETE FROM narrative_state_snapshots "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            deleted["snapshots"] = cursor.rowcount

            cursor = await self._conn.execute(
                "DELETE FROM narrative_branches "
                "WHERE session_id = ? AND branch_id = ? AND user_id = ?",
                (session_id, branch_id, user_id),
            )
            deleted["branches"] = cursor.rowcount

            await self._conn.commit()
            log.info("narrative_branch_deleted",
                     session_id=session_id, branch_id=branch_id, **deleted)
            return deleted
        except Exception:
            log.warning("delete_branch_cascade_failed",
                        session_id=session_id, branch_id=branch_id, exc_info=True)
            try:
                await self._conn.rollback()
            except Exception as rb_exc:
                log.debug(
                    "delete_branch_cascade_rollback_failed",
                    session_id=session_id,
                    branch_id=branch_id,
                    error=str(rb_exc),
                )
            return deleted

    # ------------------------------------------------------------------
    # Storage observability
    # ------------------------------------------------------------------

    async def get_session_storage(
        self,
        session_id: str,
        *,
        user_id: str = "",
    ) -> SessionStorage:
        """Per-branch + total storage counts for a session.

        Used by GET /api/narrative/session/{id}/storage to surface what the user
        has stored, supporting informed cleanup decisions.
        """
        storage = SessionStorage(session_id=session_id)
        try:
            branches = await self.list_branches(session_id, user_id=user_id, include_stale=True)
            storage.total_branches = len(branches)

            for b in branches:
                counts: dict[str, int] = {
                    "archive_rows": 0,
                    "ledger_entries": 0,
                    "snapshots": 0,
                    "approx_bytes": 0,
                }

                # Archive count + approx bytes
                q = ("SELECT COUNT(*), COALESCE(SUM(LENGTH(user_content) "
                     "+ LENGTH(assistant_content) + LENGTH(summary)), 0) "
                     "FROM narrative_archive WHERE session_id = ? AND branch_id = ?")
                p: list = [session_id, b.branch_id]
                if user_id:
                    q += " AND user_id = ?"
                    p.append(user_id)
                row = await (await self._conn.execute(q, p)).fetchone()
                if row:
                    counts["archive_rows"] = int(row[0])
                    counts["approx_bytes"] += int(row[1])
                # Vec rows: ~3KB each (768 floats × 4 bytes)
                counts["approx_bytes"] += counts["archive_rows"] * 3072

                # Ledger
                q = ("SELECT COUNT(*), COALESCE(SUM(LENGTH(content) + LENGTH(category)), 0) "
                     "FROM narrative_ledger_entries WHERE session_id = ? AND branch_id = ?")
                p = [session_id, b.branch_id]
                if user_id:
                    q += " AND user_id = ?"
                    p.append(user_id)
                row = await (await self._conn.execute(q, p)).fetchone()
                if row:
                    counts["ledger_entries"] = int(row[0])
                    counts["approx_bytes"] += int(row[1])

                # Snapshots
                q = ("SELECT COUNT(*), COALESCE(SUM(LENGTH(snapshot_data)), 0) "
                     "FROM narrative_state_snapshots WHERE session_id = ? AND branch_id = ?")
                p = [session_id, b.branch_id]
                if user_id:
                    q += " AND user_id = ?"
                    p.append(user_id)
                row = await (await self._conn.execute(q, p)).fetchone()
                if row:
                    counts["snapshots"] = int(row[0])
                    counts["approx_bytes"] += int(row[1])

                storage.branches[b.branch_id] = counts
                storage.total_archive_rows += counts["archive_rows"]
                storage.total_ledger_entries += counts["ledger_entries"]
                storage.total_snapshots += counts["snapshots"]
                storage.total_approx_bytes += counts["approx_bytes"]

            return storage
        except Exception:
            log.warning("session_storage_failed", session_id=session_id, exc_info=True)
            return storage
