"""Dream portrait synthesis and management."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import structlog

from augmentum.dream.models import DreamEntry, DreamPortrait

log = structlog.get_logger(__name__)


class PortraitManager:
    def __init__(self, journal, settings_store, model_backend=None, provider_registry=None):
        self._journal = journal
        self._settings_store = settings_store
        self._model_backend = model_backend
        self._provider_registry = provider_registry

    async def get_current(self, persona_id: str, *, user_id: str = "") -> DreamPortrait | None:
        """Load the current portrait for a persona (optionally scoped to user)."""
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return None
        try:
            query = (
                "SELECT id, persona_id, voice_notes, active_threads, impressions, "
                "source_entries, is_current, checkpoint_name, created_at "
                "FROM dream_portraits WHERE persona_id = ? AND is_current = 1"
            )
            params: list = [persona_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY created_at DESC LIMIT 1"
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            if not row:
                return None
            return DreamPortrait(
                id=row[0], persona_id=row[1],
                voice_notes=row[2], active_threads=row[3], impressions=row[4],
                source_entries=json.loads(row[5]) if row[5] else [],
                is_current=bool(row[6]),
                checkpoint_name=row[7], created_at=row[8],
            )
        except Exception:
            log.warning("get_current_portrait_failed", exc_info=True)
            return None

    async def synthesize(
        self,
        persona_id: str,
        foundation: str,
        journal_entries: list[DreamEntry] | None = None,
        *,
        user_id: str = "",
    ) -> DreamPortrait | None:
        """Generate a new portrait from foundation + journal entries."""
        # Load entries from journal if not provided
        if journal_entries is None:
            entries, _ = await self._journal.list_entries(persona_id, limit=100, user_id=user_id)
            journal_entries = entries

        if not journal_entries:
            return None

        # Weight and sort entries
        now = datetime.now(UTC)
        weighted = self._weight_entries(journal_entries, now)

        # Build journal text for the prompt
        journal_text = "\n".join(
            f"[{e.created_at}] ({e.entry_type.value if hasattr(e.entry_type, 'value') else e.entry_type}): {e.content}"
            for e in journal_entries
        )

        # Resolve speaker name from ui.aiName (per-user with global fallback);
        # mirrors DreamEngine._load_persona so the portrait voice matches the
        # journal voice instead of defaulting to a generic "Assistant".
        persona_name = "Assistant"
        if self._settings_store is not None:
            try:
                # Per-user ONLY — see the pref-leak class (identity keys).
                resolved = await self._settings_store.get_user(user_id, "ui.aiName")
                if resolved:
                    persona_name = resolved
            except Exception:
                log.warning("portrait_persona_name_lookup_failed", user_id=user_id, exc_info=True)

        from augmentum.dream.prompts import build_portrait_prompt
        system_msg, user_msg = build_portrait_prompt(
            persona_name=persona_name,
            persona_foundation=foundation,
            journal_entries_text=journal_text,
        )

        response = await self._call_llm(system_msg, user_msg)
        source_entry_ids = [e.id for e in journal_entries]
        portrait = self._parse_portrait_response(response, persona_id, source_entry_ids)

        if portrait:
            portrait = self._enforce_token_budget(portrait)
            await self._store_portrait(portrait, user_id=user_id)

        return portrait

    def _weight_entries(self, entries: list[DreamEntry], now: datetime) -> list[tuple[str, float]]:
        """Apply time decay + pin bonus to journal entries. Returns [(entry_id, score)]."""
        weighted = []
        for entry in entries:
            base_weight = entry.weight

            # Pin bonus
            if entry.pinned:
                base_weight *= 1.5

            # Time decay
            try:
                created = datetime.fromisoformat(entry.created_at.replace("Z", "+00:00"))
                age_days = (now - created).days
            except (ValueError, TypeError, AttributeError):
                age_days = 0

            if age_days < 7:
                decay = 1.0
            elif age_days < 30:
                decay = 0.7
            else:
                decay = 0.4

            score = base_weight * decay
            weighted.append((entry.id, score))

        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted

    @staticmethod
    def _coerce_section(value) -> str:
        """Normalize an LLM portrait field to a stripped string.

        The prompt asks for strings but the active_threads section is
        described as "2-3 short items", which most models interpret as
        a JSON list. Both shapes are semantically valid responses; we
        accept either rather than break on a reasonable LLM choice.
        Lists are joined with newlines so the bullet structure is
        preserved when displayed. Other shapes (numbers, dicts, None)
        get coerced via str() then stripped — better than dropping into
        the textual fallback path which loses all section structure.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(str(item).strip() for item in value if item).strip()
        return str(value).strip()

    def _parse_portrait_response(self, response: str, persona_id: str, source_entry_ids: list[str]) -> DreamPortrait | None:
        """Parse LLM response into a DreamPortrait. Falls back on JSON failure."""
        try:
            data = json.loads(response)
            return DreamPortrait(
                id=uuid.uuid4().hex[:16],
                persona_id=persona_id,
                voice_notes=self._coerce_section(data.get("voice_notes")),
                active_threads=self._coerce_section(data.get("active_threads")),
                impressions=self._coerce_section(data.get("impressions")),
                source_entries=source_entry_ids,
                is_current=True,
                created_at=datetime.now(UTC).isoformat(),
            )
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            # Fallback: use entire response as voice_notes
            content = response.strip()
            if not content:
                return None
            return DreamPortrait(
                id=uuid.uuid4().hex[:16],
                persona_id=persona_id,
                voice_notes=content,
                active_threads="",
                impressions="",
                source_entries=source_entry_ids,
                is_current=True,
                created_at=datetime.now(UTC).isoformat(),
            )

    def _enforce_token_budget(self, portrait: DreamPortrait) -> DreamPortrait:
        """Truncate portrait sections to token budget using tiktoken cl100k_base."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return portrait  # Skip if tiktoken not available

        portrait.voice_notes = self._truncate_by_sentence(portrait.voice_notes, 150, enc)
        portrait.active_threads = self._truncate_by_sentence(portrait.active_threads, 150, enc)
        portrait.impressions = self._truncate_by_sentence(portrait.impressions, 100, enc)
        return portrait

    def _truncate_by_sentence(self, text: str, max_tokens: int, enc) -> str:
        """Truncate text to max_tokens by sentence boundary."""
        if not text:
            return text
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Decode truncated tokens and find last sentence boundary
        truncated = enc.decode(tokens[:max_tokens])
        last_period = truncated.rfind(". ")
        if last_period > 0:
            return truncated[:last_period + 1]
        # No sentence boundary found — just truncate
        return truncated.rstrip()

    async def _store_portrait(self, portrait: DreamPortrait, *, user_id: str = "") -> None:
        """Store portrait in DB, marking previous as non-current."""
        if not user_id:
            raise ValueError("dream_portraits insert requires user_id")
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return
        try:
            await conn.execute(
                "UPDATE dream_portraits SET is_current = 0 "
                "WHERE persona_id = ? AND is_current = 1 AND user_id = ?",
                (portrait.persona_id, user_id),
            )
            await conn.execute(
                """INSERT INTO dream_portraits
                   (id, persona_id, voice_notes, active_threads, impressions,
                    source_entries, is_current, checkpoint_name, created_at,
                    user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    portrait.id, portrait.persona_id, portrait.voice_notes,
                    portrait.active_threads, portrait.impressions,
                    json.dumps(portrait.source_entries), 1, None,
                    portrait.created_at, user_id,
                ),
            )
            await conn.commit()
        except Exception:
            log.warning("store_portrait_failed", exc_info=True)

    async def save_checkpoint(self, persona_id: str, name: str, *, user_id: str = "") -> str | None:
        """Save current portrait as a named checkpoint."""
        if not user_id:
            raise ValueError("dream_portraits checkpoint insert requires user_id")
        current = await self.get_current(persona_id, user_id=user_id)
        if not current:
            return None

        checkpoint_id = uuid.uuid4().hex[:16]
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return None
        try:
            await conn.execute(
                """INSERT INTO dream_portraits
                   (id, persona_id, voice_notes, active_threads, impressions,
                    source_entries, is_current, checkpoint_name, created_at,
                    user_id)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (
                    checkpoint_id, persona_id, current.voice_notes,
                    current.active_threads, current.impressions,
                    json.dumps(current.source_entries), name, current.created_at,
                    user_id,
                ),
            )
            await conn.commit()
            return checkpoint_id
        except Exception:
            log.warning("save_checkpoint_failed", exc_info=True)
            return None

    async def restore_checkpoint(
        self,
        persona_id: str,
        checkpoint_id: str,
        *,
        user_id: str = "",
    ) -> DreamPortrait | None:
        """Restore a checkpointed portrait as current."""
        if not user_id:
            raise ValueError("dream_portraits restore requires user_id")
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return None
        try:
            await conn.execute(
                "UPDATE dream_portraits SET is_current = 0 "
                "WHERE persona_id = ? AND user_id = ?",
                (persona_id, user_id),
            )
            # Promote checkpoint — caller can't restore another user's checkpoint.
            await conn.execute(
                "UPDATE dream_portraits SET is_current = 1 "
                "WHERE id = ? AND persona_id = ? AND user_id = ?",
                (checkpoint_id, persona_id, user_id),
            )
            await conn.commit()
            return await self.get_current(persona_id, user_id=user_id)
        except Exception:
            log.warning("restore_checkpoint_failed", exc_info=True)
            return None

    async def reset_to_foundation(self, persona_id: str, *, user_id: str = "") -> None:
        """Delete all dream data for a persona (scoped to caller's user_id)."""
        if not user_id:
            raise ValueError("dream reset requires user_id")
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return
        try:
            for table in ("dream_entries", "dream_portraits", "dream_memory_log", "dream_cycles"):
                await conn.execute(
                    f"DELETE FROM {table} WHERE persona_id = ? AND user_id = ?",
                    (persona_id, user_id),
                )
            await conn.commit()
        except Exception:
            log.warning("reset_to_foundation_failed", exc_info=True)

    async def list_checkpoints(self, persona_id: str, *, user_id: str = "") -> list[DreamPortrait]:
        """List saved checkpoints for a persona (scoped to user when provided)."""
        conn = getattr(self._journal, "_conn", None) or getattr(self._journal, "_db", None)
        if conn is None:
            return []
        try:
            query = (
                "SELECT id, persona_id, voice_notes, active_threads, impressions, "
                "source_entries, is_current, checkpoint_name, created_at "
                "FROM dream_portraits WHERE persona_id = ? AND checkpoint_name IS NOT NULL"
            )
            params: list = [persona_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY created_at DESC"
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                DreamPortrait(
                    id=r[0], persona_id=r[1], voice_notes=r[2], active_threads=r[3],
                    impressions=r[4], source_entries=json.loads(r[5]) if r[5] else [],
                    is_current=bool(r[6]), checkpoint_name=r[7], created_at=r[8],
                )
                for r in rows
            ]
        except Exception:
            log.warning("list_checkpoints_failed", exc_info=True)
            return []

    async def _call_llm(self, system_msg: str, user_msg: str) -> str:
        """Call the LLM for portrait synthesis.

        Resolves the backend dynamically via provider_registry using the
        dream_model setting (or first available model as fallback).
        """
        backend = self._model_backend
        model = None

        # Resolve via registry if no static backend
        if backend is None and self._provider_registry is not None:
            try:
                from augmentum.config import settings as _cfg
                backend, model = await self._provider_registry.resolve_model_for_role(
                    "utility",
                    override=_cfg.dream_portrait_model or _cfg.dream_model or "",
                    settings=_cfg,
                )
            except Exception:
                log.warning("portrait_backend_resolve_failed", exc_info=True)

        if backend is None:
            log.debug("portrait_no_backend_available")
            return '{"voice_notes": "", "active_threads": "", "impressions": ""}'

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
            log.warning("portrait_llm_call_failed", exc_info=True)
            return '{"voice_notes": "", "active_threads": "", "impressions": ""}'
