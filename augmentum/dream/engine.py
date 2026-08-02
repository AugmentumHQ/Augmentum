"""Dream generation engine — orchestrates the full dream cycle."""
from __future__ import annotations

import json
import uuid
import time
from datetime import datetime, timezone

import structlog

from augmentum.dream.models import DreamEntry, DreamEntryType, DreamCycle, ContextSegment
from augmentum.dream.context import DreamContextBuilder
from augmentum.dream.prompts import build_dream_prompt, DREAM_ANTI_PATTERNS

log = structlog.get_logger(__name__)


class DreamEngine:
    def __init__(
        self, journal, memory_store, state_manager, embedding_service,
        portrait_manager, settings, provider_registry=None,
        settings_store=None,
    ):
        if journal is None:
            raise ValueError("DreamEngine requires a journal (DreamJournal)")
        if memory_store is None:
            raise ValueError("DreamEngine requires a memory_store")
        self._journal = journal
        self._memory_store = memory_store
        self._state_manager = state_manager
        self._embedding_service = embedding_service
        self._portrait_manager = portrait_manager
        self._settings = settings
        self._provider_registry = provider_registry
        # settings_store is the per-user personalization source of truth since
        # Stage D. The engine reads the caller's ``ui.aiName`` /
        # ``ui.aiInstructions`` / ``ui.responseStyle`` from here so each
        # tenant's dreams speak in their own persona's voice. Optional to
        # accommodate test fixtures; when absent, foundation is the minimal
        # default string and persona_name is "Assistant".
        self._settings_store = settings_store
        self._context_builder = DreamContextBuilder()

    async def run_cycle(
        self,
        persona_id: str,
        trigger_reason: str,
        *,
        user_id: str = "",
    ) -> DreamCycle:
        """Run a full dream cycle: select → context → generate → store → portrait.

        ``user_id`` is required — every step scopes to that user (their
        approved memories, their entries/portraits/cycles, their dreamed-
        memory filter). The journal write rejects empty user_id; pass a
        real id from the auth scope or scheduler.
        """
        cycle_id = uuid.uuid4().hex[:16]
        cycle = DreamCycle(id=cycle_id, persona_id=persona_id, trigger_reason=trigger_reason)
        cycle.started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()

        try:
            # 1. Select dream material
            memories = await self._select_dream_material(persona_id, user_id=user_id)
            if not memories:
                cycle.status = "completed"
                cycle.completed_at = datetime.now(timezone.utc).isoformat()
                cycle.duration_ms = int((time.monotonic() - start_time) * 1000)
                await self._persist_cycle(cycle, user_id=user_id)
                return cycle

            cycle.memories_count = len(memories)

            # 2. Build context segments
            segments = self._context_builder.cluster_by_proximity(memories)

            # 3. Generate dreams for each segment
            all_entries = []
            persona_name, foundation = await self._load_persona(user_id=user_id)
            portrait = (
                await self._portrait_manager.get_current(persona_id, user_id=user_id)
                if self._portrait_manager else None
            )
            # Last 5 reflections seed the prompt's "recent thoughts" frame
            recent_dreams = await self._load_recent_dreams(persona_id, limit=5, user_id=user_id)

            for segment in segments:
                entries = await self._generate_for_segment(
                    segment, persona_name, foundation, portrait, recent_dreams,
                    cycle_id, persona_id,
                )
                all_entries.extend(entries)

            # 4. Store entries in journal (scoped to user)
            for entry in all_entries:
                await self._journal.store_entry(
                    persona_id=entry.persona_id,
                    content=entry.content,
                    entry_type=entry.entry_type,
                    source_memories=entry.source_memories,
                    source_sessions=entry.source_sessions,
                    context_window=entry.context_window,
                    dream_cycle_id=entry.dream_cycle_id,
                    user_id=user_id,
                )

            # 5. Record dreamed memories
            memory_ids = [m["id"] for m in memories]
            await self._record_dreamed(memory_ids, cycle_id, persona_id, user_id=user_id)

            # 6. Regenerate portrait (scoped)
            if self._portrait_manager and all_entries:
                try:
                    await self._portrait_manager.synthesize(persona_id, foundation, None, user_id=user_id)
                except Exception:
                    log.warning("portrait_synthesis_failed", exc_info=True)

            cycle.entries_count = len(all_entries)
            cycle.status = "completed"

        except Exception as e:
            cycle.status = "failed"
            cycle.error = str(e)
            log.error("dream_cycle_failed", error=str(e), exc_info=True)

        cycle.duration_ms = int((time.monotonic() - start_time) * 1000)
        cycle.completed_at = datetime.now(timezone.utc).isoformat()
        await self._persist_cycle(cycle, user_id=user_id)
        return cycle

    async def _persist_cycle(self, cycle: DreamCycle, *, user_id: str = "") -> None:
        """Write a DreamCycle to dream_cycles for status / history queries.

        Uses ``transactional_write`` so a transient ``database is locked``
        on the INSERT can't leave the journal's persistent connection in
        in_transaction=True state — that exact silent failure here on
        2026-05-22 caused the next dream-journal SELECT to pin a WAL
        snapshot at frame ~10 and held it for 8 hours, growing the WAL
        to 28 MB and producing a cascading lock storm. The connection
        itself also has ``install_safe_rollback`` as a second line of
        defense; this helper makes the atomicity intent explicit.
        """
        if not user_id:
            raise ValueError("dream_cycles insert requires user_id")
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return
        from augmentum.state.backends.sqlite import transactional_write

        try:
            async with transactional_write(conn) as db:
                await db.execute(
                    """INSERT OR REPLACE INTO dream_cycles
                       (id, persona_id, trigger_reason, memories_count, entries_count,
                        model_used, tokens_used, duration_ms, status, error,
                        started_at, completed_at, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cycle.id, cycle.persona_id, cycle.trigger_reason,
                        cycle.memories_count, cycle.entries_count, cycle.model_used,
                        cycle.tokens_used, cycle.duration_ms, cycle.status, cycle.error,
                        cycle.started_at, cycle.completed_at, user_id,
                    ),
                )
        except Exception:
            log.warning("dream_cycle_persist_failed", exc_info=True)

    async def _load_recent_dreams(
        self, persona_id: str, limit: int = 5, *, user_id: str = "",
    ) -> list[str]:
        """Fetch the last N reflection contents for the prompt 'recent thoughts' frame."""
        try:
            entries, _ = await self._journal.list_entries(
                persona_id=persona_id, limit=limit, user_id=user_id,
            )
            return [e.content for e in entries if e.content]
        except Exception:
            return []

    async def _select_dream_material(self, persona_id: str, *, user_id: str = "") -> list[dict]:
        """Select undreamed, dream-eligible memories.

        Eligibility = made it past the LLM extractor's confidence gate
        AND landed in the active/core tier AND not already dreamed about.
        We deliberately do NOT require ``memories.user_approved = 1``:
        that flag is only flipped by the manual notification-approval UI
        (memory_routes.py), which most users never touch — and auto-
        approve users don't get it set either (auto-approve only updates
        tier). Gating on it meant the engine produced empty cycles for
        ~everyone, regardless of how much memory material existed. The
        tier filter below is the real quality gate: provisional/expired
        rows never reach this point.
        """
        dreamed_ids = await self._dreamed_memory_ids(persona_id, user_id=user_id)

        all_memories = await self._get_dream_eligible_memories(user_id=user_id)

        result = []
        for mem in all_memories:
            if mem["id"] in dreamed_ids:
                continue
            tier = mem.get("tier", "")
            if isinstance(tier, str):
                tier_val = tier
            else:
                tier_val = tier.value if hasattr(tier, "value") else str(tier)
            if tier_val not in ("active", "core"):
                continue
            if not mem.get("source_message_id") and not mem.get("evidence"):
                continue
            result.append(mem)

        # Sort chronologically. `created_at` has NOT NULL DEFAULT in the
        # schema, but defensive None-coercion costs nothing and prevents
        # the same TypeError that bit cluster_by_proximity if the column
        # is ever NULL for any reason (legacy rows, future schema drift).
        result.sort(key=lambda m: m.get("created_at") or "")
        return result

    async def _get_dream_eligible_memories(self, *, user_id: str = "") -> list[dict]:
        """Pull all unexpired memories for ``user_id`` for dream selection.

        See ``_select_dream_material`` for why we no longer filter by
        ``user_approved``. The caller applies tier and dreamed-id
        filters; this method only enforces user scoping and validity.
        """
        try:
            conn = getattr(self._memory_store, "_conn", None)
            if conn is None:
                return []
            query = (
                "SELECT id, content, evidence, tier, session_id, source_message_id, "
                "user_approved, created_at FROM memories "
                "WHERE (valid_until IS NULL OR valid_until > datetime('now'))"
            )
            params: tuple = ()
            if user_id:
                query += " AND user_id = ?"
                params = (user_id,)
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "content": r[1], "evidence": r[2], "tier": r[3],
                    "session_id": r[4], "source_message_id": r[5],
                    "user_approved": r[6], "created_at": r[7],
                }
                for r in rows
            ]
        except Exception:
            log.warning("get_dream_eligible_memories_failed", exc_info=True)
            return []

    async def _dreamed_memory_ids(self, persona_id: str, *, user_id: str = "") -> set[str]:
        """Get IDs of memories already dreamed about for this persona+user."""
        try:
            conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
            if conn is None:
                return set()
            query = "SELECT memory_id FROM dream_memory_log WHERE persona_id = ?"
            params: list = [persona_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    async def _record_dreamed(
        self,
        memory_ids: list[str],
        cycle_id: str,
        persona_id: str,
        *,
        user_id: str = "",
    ) -> None:
        """Record that these memories have been dreamed about."""
        if not user_id:
            raise ValueError("dream_memory_log insert requires user_id")
        try:
            conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
            if conn is None:
                return
            for mid in memory_ids:
                await conn.execute(
                    """INSERT OR IGNORE INTO dream_memory_log
                       (memory_id, dream_cycle_id, persona_id, user_id)
                       VALUES (?, ?, ?, ?)""",
                    (mid, cycle_id, persona_id, user_id),
                )
            await conn.commit()
        except Exception:
            log.warning("record_dreamed_failed", exc_info=True)

    async def _generate_for_segment(
        self,
        segment,
        persona_name: str,
        foundation,
        portrait,
        recent_dreams,
        cycle_id,
        persona_id: str,
    ):
        """Generate dream entries for a single context segment (LLM call).

        ``persona_name`` is the resolved per-user ``ui.aiName`` passed down
        from :meth:`run_cycle`. Taking it as an explicit parameter (rather
        than re-reading ``self._settings.ai_name`` inline) is the whole
        point of the persona-loader refactor — the legacy inline read
        pulled the install-wide default and ignored every user's
        personalisation.
        """
        # Build the prompt
        memories = segment.get("memories", [])

        memory_content = "\n".join(m.get("content", "") for m in memories)
        memory_evidence = "\n".join(m.get("evidence", "") or "" for m in memories)

        # For now, conversation_messages would come from context window retrieval.
        # Simplified: use evidence as context.
        conversation_messages = memory_evidence

        portrait_text = ""
        if portrait:
            portrait_text = (
                f"Voice: {portrait.voice_notes}\n"
                f"Threads: {portrait.active_threads}\n"
                f"Impressions: {portrait.impressions}"
            )

        previous_dreams_text = "\n".join(f"- {d}" for d in recent_dreams) if recent_dreams else ""

        system_msg, user_msg = build_dream_prompt(
            persona_name=persona_name,
            persona_foundation=foundation,
            memory_content=memory_content,
            memory_evidence=memory_evidence,
            conversation_messages=conversation_messages,
            relative_age="recently",
            absolute_timestamp=memories[0].get("created_at", "") if memories else "",
            current_portrait=portrait_text,
            previous_dreams=previous_dreams_text,
        )

        # Call LLM (simplified — in production, use the model backend)
        response = await self._call_llm(system_msg, user_msg)

        source_memory_ids = [m["id"] for m in memories]
        source_session_ids = list({m.get("session_id", "") for m in memories if m.get("session_id")})

        return self._parse_dream_response(response, cycle_id, persona_id, source_memory_ids, source_session_ids)

    async def _call_llm(self, system_msg: str, user_msg: str) -> str:
        """Call the LLM backend for dream generation.

        Resolves the backend via provider_registry using the dream_model
        setting (or first available model as fallback).
        """
        backend = None
        model = None

        if self._provider_registry is not None:
            try:
                from augmentum.config import settings as _cfg
                backend, model = await self._provider_registry.resolve_model_for_role(
                    "utility",
                    override=_cfg.dream_model or "",
                    settings=_cfg,
                )
            except Exception:
                log.warning("dream_backend_resolve_failed", exc_info=True)

        if backend is None:
            log.warning("no_model_backend_for_dreaming")
            return '{"reflections": []}'

        try:
            from augmentum.models.base import InternalChatRequest, Message
            req = InternalChatRequest(
                model=model or "",
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
            )
            resp = await backend.chat(req)
            return resp.message.content if hasattr(resp, "message") else str(resp)
        except Exception:
            log.warning("dream_llm_call_failed", exc_info=True)
            return '{"reflections": []}'

    async def _load_persona(
        self, *, user_id: str = "",
    ) -> tuple[str, str]:
        """Resolve ``(persona_name, foundation)`` for this user.

        Both values come from the same per-user personalization block:

        * ``persona_name`` = ``ui.aiName`` → used as the speaker identity in
          the dream prompt's system message ("You are {persona_name}…").
        * ``foundation`` = ``"Your name is {name}. {aiInstructions} Your
          communication style is {responseStyle}."`` — the immutable-base
          persona string the prompt grounds every reflection in.

        Previously the engine read settings via
        ``self._state_manager.settings_store`` (an attribute StateManager
        never had), silently falling back to "A helpful assistant." and
        the install-wide config's ``ai_name``. Every user's dreams
        therefore spoke in the generic default voice regardless of how
        they'd personalised. This method now reads ``self._settings_store``
        directly and returns both values together so the prompt layer
        can't accidentally re-diverge.
        """
        name = "Assistant"
        parts: list[str]

        if self._settings_store is None:
            return name, "A helpful assistant."

        try:
            # Per-user ONLY — identity keys are personal state; a global
            # fallback would dream in the OWNER's persona for every user
            # (the multi-tenant pref-leak class).
            resolved_name = await self._settings_store.get_user(user_id, "ui.aiName")
            instructions = await self._settings_store.get_user(user_id, "ui.aiInstructions")
            style = await self._settings_store.get_user(user_id, "ui.responseStyle")
        except Exception:
            log.warning("dream_engine.persona_lookup_failed", user_id=user_id, exc_info=True)
            return name, "A helpful assistant."

        if resolved_name:
            name = resolved_name
        parts = [f"Your name is {name}."]
        if instructions:
            parts.append(instructions)
        if style:
            parts.append(f"Your communication style is {style}.")
        return name, " ".join(parts)

    def _parse_dream_response(
        self,
        response: str,
        cycle_id: str,
        persona_id: str,
        source_memories: list[str],
        source_sessions: list[str],
    ) -> list[DreamEntry]:
        """Parse LLM response into DreamEntry objects. Falls back to single reflection on JSON failure."""
        entries = []

        try:
            data = json.loads(response)
            reflections = data.get("reflections", [])
            for ref in reflections:
                entry_type_str = ref.get("type", "reflection")
                content = ref.get("content", "").strip()
                if not content:
                    continue

                try:
                    entry_type = DreamEntryType(entry_type_str)
                except ValueError:
                    entry_type = DreamEntryType.REFLECTION

                entries.append(DreamEntry(
                    id=uuid.uuid4().hex[:16],
                    persona_id=persona_id,
                    content=content,
                    entry_type=entry_type,
                    source_memories=source_memories,
                    source_sessions=source_sessions,
                    context_window={},
                    embedding=None,
                    dream_cycle_id=cycle_id,
                ))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: treat entire response as a single reflection
            content = response.strip()
            if content:
                entries.append(DreamEntry(
                    id=uuid.uuid4().hex[:16],
                    persona_id=persona_id,
                    content=content,
                    entry_type=DreamEntryType.REFLECTION,
                    source_memories=source_memories,
                    source_sessions=source_sessions,
                    context_window={},
                    embedding=None,
                    dream_cycle_id=cycle_id,
                ))

        # Filter anti-patterns
        entries = self._filter_anti_patterns(entries)
        return entries

    def _filter_anti_patterns(self, entries: list[DreamEntry]) -> list[DreamEntry]:
        """Remove entries containing AI-self-referential phrases."""
        filtered = []
        for entry in entries:
            content_lower = entry.content.lower()
            if any(pattern in content_lower for pattern in DREAM_ANTI_PATTERNS):
                log.debug("dream_entry_filtered", pattern_match=True, content_preview=entry.content[:50])
                continue
            filtered.append(entry)
        return filtered
