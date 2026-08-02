"""Core memory profile manager — always-in-context user summary.

No LLM required. Builds a compact profile by ranking existing memories
by importance, access frequency, and recency, then greedily packing
them into a token budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta
from math import exp
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore

log = get_logger(__name__)

# Token counting for accurate budget enforcement
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # Fallback if tiktoken not installed
    def _count_tokens(text: str) -> int:
        return len(text) // 4

# Recency half-life: memories lose 50% of their recency weight after this many days.
# Using 60 days (gentler than store's 30-day recall decay) since core profile
# facts like "I'm a software engineer" should persist longer.
_PROFILE_RECENCY_HALF_LIFE_DAYS = 60.0


def _recency_weight(updated_at: str | None) -> float:
    """Exponential decay weight based on memory age. Returns 0.3–1.0.

    Floor of 0.3 ensures very old but important facts still appear
    (e.g., "My name is Alice" shouldn't fully decay).
    """
    if not updated_at:
        return 0.5
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_days = (datetime.now(UTC) - updated).total_seconds() / 86400
        weight = exp(-0.693 * age_days / _PROFILE_RECENCY_HALF_LIFE_DAYS)
        return max(weight, 0.3)  # floor — old facts still matter
    except (ValueError, TypeError):
        return 0.5


class CoreProfileManager:
    """Manages a compact, always-in-context user profile derived from memories."""

    _PERSIST_KEY_PREFIX = "core_profile:"
    # Single key holding per-user counter + last-rebuild timestamp. JSON
    # blob keyed by user_id. Persisted so restarts don't reset progress
    # toward the rebuild threshold and so the age-based trigger has a
    # reference point. Same shape as DreamScheduler._counters.
    _STATE_KEY = "core_profile._state"
    # Rebuild even when extraction count hasn't reached threshold if the
    # current persisted profile is older than this. Catches the case where
    # the user keeps restarting (counter resets) or chats infrequently
    # but still wants a fresh profile when they come back.
    _MAX_AGE = timedelta(hours=24)

    def __init__(
        self,
        store: MemoryStore,
        max_tokens: int | None = None,
        rebuild_interval: int | None = None,
        app_state: object | None = None,
    ) -> None:
        self._store = store
        self._max_tokens = max_tokens or settings.memory_core_profile_max_tokens
        self._rebuild_interval = rebuild_interval or settings.memory_core_profile_rebuild_interval
        self._app_state = app_state  # for lazy backend resolution during LLM profile synthesis
        self._cache: dict[str, str] = {}             # user_id -> profile text
        self._stale: set[str] = set()
        self._extractions_since_rebuild: dict[str, int] = {}
        # Per-user last-rebuild timestamp (UTC). Hydrated from persisted
        # state in initialize(); written by _rebuild on success. Used by
        # the age trigger in get_profile.
        self._last_rebuilt_at: dict[str, datetime] = {}
        # Lock prevents concurrent flushes from racing each other when
        # multiple notify_extraction calls fire in flight.
        self._flush_lock = asyncio.Lock()
        # Per-user rebuild locks. Collapse concurrent rebuild callers
        # into a single LLM call: the second caller waits and then
        # short-circuits when the first has already produced fresh
        # cache+stale state. Without this, two parallel hits on the
        # cold path (e.g., two browser tabs firing /v1/memory/
        # context-preview at once on a post-restart cache) both run
        # _synthesize_profile end-to-end and the second overwrites
        # the first while burning ~130 s of engine time.
        self._rebuild_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Hydrate per-user counters + last-rebuild timestamps from persistence.

        Without this, every container restart would reset the extraction
        counter to 0, meaning a chronically-restarting user (or one mid-
        development) might never accumulate enough extractions between
        restarts to trigger a rebuild. Idempotent — safe to call multiple
        times. Failures are non-fatal: in-memory state stays at empty
        defaults and the system continues to work, just losing the
        "remember progress across restarts" benefit.
        """
        store = self._get_settings_store()
        if store is None:
            return
        try:
            raw = await store.get(self._STATE_KEY)
        except Exception:
            log.debug("core_profile.state_load_failed", exc_info=True)
            return
        if not raw:
            return
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            log.warning("core_profile.state_corrupt_skipped")
            return
        for uid, fields in (data or {}).items():
            with contextlib.suppress(TypeError, ValueError):
                self._extractions_since_rebuild[uid] = int(fields.get("extractions", 0))
            last_rebuilt = fields.get("last_rebuilt_at")
            if last_rebuilt:
                try:
                    self._last_rebuilt_at[uid] = datetime.fromisoformat(last_rebuilt)
                except ValueError:
                    pass
        log.info("core_profile.state_loaded", users=len(data or {}))

    async def _flush_state(self) -> None:
        """Write per-user counters + timestamps back to persistence.

        Called best-effort after notify_extraction and after rebuild.
        Single-key JSON blob (not row-per-user) to match dream scheduler
        and minimize write amplification. Locked to prevent concurrent
        notify_extraction floods from corrupting the blob.
        """
        store = self._get_settings_store()
        if store is None:
            return
        async with self._flush_lock:
            data: dict[str, dict] = {}
            users = (
                set(self._extractions_since_rebuild.keys())
                | set(self._last_rebuilt_at.keys())
            )
            for uid in users:
                last_rebuilt = self._last_rebuilt_at.get(uid)
                data[uid] = {
                    "extractions": self._extractions_since_rebuild.get(uid, 0),
                    "last_rebuilt_at": last_rebuilt.isoformat() if last_rebuilt else None,
                }
            try:
                await store.set(self._STATE_KEY, json.dumps(data))
            except Exception:
                log.debug("core_profile.state_flush_failed", exc_info=True)

    async def get_profile(self, user_id: str = "default") -> str:
        """Return the cached profile, rebuilding if stale or missing.

        On first access after restart, loads the persisted profile from
        SQLite instead of calling the LLM.  Only rebuilds (and re-synthesizes
        via LLM) when explicitly marked stale.

        Validates persisted profiles against actual memory count — if no
        memories exist, clears any stale/hallucinated profile.

        Stale check runs FIRST, before any cache/persisted load. The
        previous order was: cache → persisted → fallthrough-rebuild.
        That meant the manual "rebuild" button (which calls invalidate
        then get_profile) silently returned the stale persisted version
        instead of rebuilding — invalidate cleared the in-memory cache
        but persisted-load short-circuited the rebuild path. Treating
        ``_stale`` as authoritative regardless of cache/persisted state
        is the only correctness-preserving fix.
        """
        # Age trigger — covers the case where the extraction counter
        # never reaches threshold (infrequent chats, frequent restarts
        # losing in-memory progress) but the persisted profile is
        # genuinely stale relative to recent activity. Cheap O(1) check
        # on every get_profile call; only fires the rebuild when both
        # (a) we have a recorded last-rebuild and (b) it's older than
        # _MAX_AGE. First-ever access still goes through the cold path
        # below since _last_rebuilt_at is unset.
        last_rebuilt = self._last_rebuilt_at.get(user_id)
        if last_rebuilt is not None:
            age = datetime.now(UTC) - last_rebuilt
            if age >= self._MAX_AGE:
                self._stale.add(user_id)
                log.debug(
                    "core_profile.aged_stale",
                    user_id=user_id, age_hours=age.total_seconds() / 3600,
                )

        if user_id in self._stale:
            await self._rebuild(user_id)
            return self._cache.get(user_id, "")

        if user_id in self._cache:
            return self._cache[user_id]

        # Cold path — first access for this user (or after invalidate).
        has_memories = await self._has_memories(user_id)

        if not has_memories:
            # No memories — clear any stale persisted profile
            await self._clear_persisted(user_id)
            self._cache[user_id] = ""
            return ""

        loaded = await self._load_persisted(user_id)
        if loaded is not None:
            self._cache[user_id] = loaded
            log.debug("core_profile_loaded_from_db", user_id=user_id, chars=len(loaded))
            return loaded

        # No persisted profile — rebuild from scratch
        await self._rebuild(user_id)
        return self._cache.get(user_id, "")

    async def get_profile_cached_only(self, user_id: str = "default") -> str:
        """Return the cached or persisted profile WITHOUT calling the LLM.

        Use when:
        - A lightweight UI indicator needs the profile fast and a
          stale-or-empty result is acceptable (the chat memory glow,
          /v1/memory/context-preview).
        - Blocking the request on a 30–130 s LLM rebuild would degrade
          UX cascadingly (the engine queue is single-slot, so a rebuild
          here also stalls the game-agent slow path).

        Expects:
        - Same arguments as :meth:`get_profile`.

        Returns:
        - The in-memory cached profile if present.
        - Else the persisted profile from the settings store, promoted
          into the cache for subsequent reads.
        - Else empty string, with a background rebuild scheduled so the
          next call (after the rebuild completes) returns the fresh
          profile without ever blocking a caller. Empty does NOT mean
          "no memories exist"; it means "no profile available yet."
        """
        if user_id in self._cache:
            # If stale (age- or extraction-triggered) we still return
            # the cached version immediately and let the rebuild run
            # in the background. Better stale than stalling the UI.
            if user_id in self._stale:
                self._schedule_background_rebuild(user_id)
            return self._cache[user_id]

        loaded = await self._load_persisted(user_id)
        if loaded is not None:
            self._cache[user_id] = loaded
            return loaded

        # Nothing cached and nothing persisted. Schedule a rebuild for
        # later callers and return empty now.
        self._schedule_background_rebuild(user_id)
        return ""

    def _schedule_background_rebuild(self, user_id: str) -> None:
        """Fire-and-forget rebuild. Best-effort, never raises to caller.

        Marks the user stale and schedules ``_rebuild`` on the running
        loop. The per-user rebuild lock collapses overlapping schedule
        calls into a single LLM round-trip, so calling this repeatedly
        is safe.
        """
        self._stale.add(user_id)
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(self._rebuild(user_id))

    async def _has_memories(self, user_id: str) -> bool:
        """Check whether any non-expired memories exist for this user."""
        try:
            memories = await self._store.list_all(
                user_id=user_id, limit=1, include_expired=False,
            )
            return bool(memories)
        except Exception:
            log.debug("core_profile_memory_check_failed", user_id=user_id, exc_info=True)
            return False

    async def _clear_persisted(self, user_id: str) -> None:
        """Remove a stale persisted profile from SQLite."""
        try:
            settings_store = self._get_settings_store()
            if settings_store is None:
                return
            await settings_store.set(self._PERSIST_KEY_PREFIX + user_id, "")
            log.info("core_profile_cleared_stale", user_id=user_id)
        except Exception:
            log.debug("core_profile_clear_failed", user_id=user_id, exc_info=True)

    async def _load_persisted(self, user_id: str) -> str | None:
        """Load profile from SQLite settings store. Returns None if not found."""
        try:
            settings_store = self._get_settings_store()
            if settings_store is None:
                return None
            value = await settings_store.get(self._PERSIST_KEY_PREFIX + user_id)
            return value if value else None
        except Exception:
            log.debug("core_profile_load_failed", user_id=user_id, exc_info=True)
            return None

    async def _persist(self, user_id: str, profile: str) -> None:
        """Save profile to SQLite settings store."""
        try:
            settings_store = self._get_settings_store()
            if settings_store is None:
                return
            await settings_store.set(self._PERSIST_KEY_PREFIX + user_id, profile)
        except Exception:
            log.debug("core_profile_persist_failed", user_id=user_id, exc_info=True)

    def _get_settings_store(self):
        """Resolve the settings store from app state."""
        return getattr(self._app_state, "settings_store", None)

    def mark_stale(self, user_id: str = "default") -> None:
        """Mark the profile as needing rebuild (e.g., after high-importance fact stored)."""
        self._stale.add(user_id)

    def notify_extraction(self, user_id: str = "default") -> None:
        """Increment extraction counter; triggers rebuild every N extractions.

        Schedules a fire-and-forget flush so the counter survives a
        restart. Without persistence, a chronically-restarting user
        accumulates 0 → 1 → 2 → restart → 0 → 1 → ... and never reaches
        the threshold. With persistence the counter resumes where it
        left off.
        """
        count = self._extractions_since_rebuild.get(user_id, 0) + 1
        self._extractions_since_rebuild[user_id] = count
        if count >= self._rebuild_interval:
            self._stale.add(user_id)
            self._extractions_since_rebuild[user_id] = 0
        # Fire-and-forget flush — don't block the caller (extraction
        # path runs in finally blocks where blocking would matter).
        # Failures are best-effort; the worst case is the next restart
        # loses ≤1 increment of progress.
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(self._flush_state())

    def invalidate(self, user_id: str = "default") -> None:
        """Force a rebuild on next ``get_profile`` for this user.

        Drops the in-memory cache AND marks the user stale, so the next
        access takes the rebuild path even when a persisted version
        exists in the settings store. Previously this method discarded
        the stale flag, which made the manual rebuild button (and the
        orphan-reassignment flow) silently no-op when persisted data
        was present — the persisted-load branch in ``get_profile``
        short-circuited the rebuild.
        """
        self._cache.pop(user_id, None)
        self._stale.add(user_id)

    async def _rebuild(self, user_id: str) -> None:
        """Rebuild the profile from top-ranked memories.

        Per-user mutex collapses concurrent callers: after acquiring,
        we re-check whether another caller already populated the cache
        and cleared the stale flag (double-checked locking). This
        matters because ``_synthesize_profile`` is the expensive LLM
        call; two parallel callers without the lock both run it.
        """
        lock = self._rebuild_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            # Another caller may have finished while we were queued.
            # Treat ``stale flag discarded AND cache populated`` as
            # proof of a successful rebuild — we can return without
            # touching the LLM. The empty-store branch also sets cache
            # to "" and clears stale, so it qualifies too.
            if user_id not in self._stale and user_id in self._cache:
                return
            await self._rebuild_locked(user_id)

    async def _rebuild_locked(self, user_id: str) -> None:
        """Rebuild body. Must be called with the per-user lock held."""
        try:
            memories = await self._store.list_all(
                user_id=user_id, limit=50, include_expired=False,
            )
        except Exception:
            log.warning("core_profile_rebuild_failed", user_id=user_id, exc_info=True)
            return

        if not memories:
            self._cache[user_id] = ""
            self._stale.discard(user_id)
            return

        # Rank by tier_boost * effective_importance * recency_weight.
        # effective_importance already includes access boost + time decay,
        # so we don't need a separate access_count multiplier here.
        # CORE tier memories get a 2x boost — they've earned their place.
        from augmentum.memory.models import Memory, MemoryTier
        from augmentum.memory.store import MemoryStore

        def _rank_score(m: Memory) -> float:
            tier_val = m.tier if isinstance(m.tier, str) else m.tier.value
            tier_boost = 2.0 if tier_val == MemoryTier.CORE else 1.0
            eff_imp = MemoryStore._effective_importance(m)
            return tier_boost * eff_imp * _recency_weight(m.updated_at)

        ranked = sorted(memories, key=_rank_score, reverse=True)

        # Subtractive profile (docs/superpowers/specs/2026-06-20-memory-
        # subtractive-design.md): the ALWAYS-ON Layer-3 slice should carry only
        # EARNED facts — CORE tier, tightly capped — so she shapes tone from a
        # handful of anchors instead of reciting the user's whole life every
        # turn. Passive/uncorroborated facts still reach her, but via the
        # relevance-gated recall lane (Layer 5.5), not this unconditional dump.
        # Revertible: companion_profile_tone_only=False restores the old
        # top-50 life-story synthesis for one release.
        tone_only = bool(getattr(settings, "companion_profile_tone_only", True))
        if tone_only:
            _TONE_ONLY_MAX_TOKENS = 160
            _TONE_ONLY_MAX_FACTS = 6
            core = [
                m for m in ranked
                if (m.tier if isinstance(m.tier, str) else m.tier.value)
                == MemoryTier.CORE
            ]
            if core:
                ranked = core
            else:
                # No earned-CORE facts yet (new relationship, or an install
                # from before the write-side bar lands) — a thin bridge of the
                # top few so Layer 3 isn't abruptly blank. Fades on its own as
                # corroboration promotes facts into CORE.
                ranked = ranked[:3]
            max_tokens = min(self._max_tokens, _TONE_ONLY_MAX_TOKENS)
            max_facts = _TONE_ONLY_MAX_FACTS
        else:
            max_tokens = self._max_tokens
            max_facts = len(ranked)

        # Collect top facts for the profile (reserve space for header + footer)
        facts: list[str] = []
        tokens_used = 20  # reserve for [user_context] header + footer instruction
        for mem in ranked:
            if len(facts) >= max_facts:
                break
            mem_tokens = _count_tokens(mem.content)
            if tokens_used + mem_tokens + 2 > max_tokens:
                break
            facts.append(mem.content)
            tokens_used += mem_tokens + 2

        if not facts:
            self._cache[user_id] = ""
            self._stale.discard(user_id)
            return

        # Build a narrative profile via LLM synthesis.
        # Falls back to bullet list if no backend available.
        profile = await self._synthesize_profile(facts)

        self._cache[user_id] = profile
        self._stale.discard(user_id)
        # Stamp the rebuild time so the age-based trigger has a
        # reference point. Persisted alongside the counter so it
        # survives restarts. Failures non-fatal — just means the next
        # session won't have an age trigger until the next rebuild.
        self._last_rebuilt_at[user_id] = datetime.now(UTC)
        await self._persist(user_id, profile)
        try:
            await self._flush_state()
        except Exception as exc:
            # Failures non-fatal — next session won't have the age-based
            # trigger until the next rebuild. Comment above explains intent.
            log.debug("core_profile_state_flush_failed", error=str(exc))
        log.debug(
            "core_profile_rebuilt",
            user_id=user_id,
            facts=len(facts),
            chars=len(profile),
        )

    _PROFILE_SYNTHESIS_PROMPT = """\
Rewrite these facts about a user into a short, natural-sounding briefing \
(2-4 sentences). Write in third person. Combine related details into \
flowing prose — do not use bullet points or numbered lists. Omit the \
word "user" — use "they" or weave details naturally.

CRITICAL: Use ONLY the facts listed below. Do NOT infer, extrapolate, \
or add any details not explicitly stated. If a fact says "lives in Austin", \
do not add a job title, certification, or anything else not in the list.

Facts:
{facts}

Write ONLY the briefing, nothing else."""

    async def _synthesize_profile(self, facts: list[str]) -> str:
        """Synthesize facts into a natural-language profile via LLM.

        Falls back to a simple bullet list if no backend is available
        or the LLM call fails.
        """
        bullet_fallback = (
            "[user_context]\n"
            + "\n".join(f"- {f}" for f in facts)
            + "\n\nUse this context to shape your voice, tone, and depth — not to introduce subjects. Only reference these topics if the user brings them up first. Never list them back."
        )

        # Try LLM synthesis
        registry = getattr(self._app_state, "provider_registry", None) if self._app_state else None
        if not registry:
            return bullet_fallback

        try:
            from augmentum.models.base import InternalChatRequest, Message

            # Resolve backend via role chain: extraction model override → utility_model → primary
            # (no chat model fallback — profile rebuilds happen outside request context)
            backend = None
            model_name = ""
            try:
                backend, model_name = await registry.resolve_model_for_role(
                    "utility",
                    override=settings.memory_llm_extraction_model,
                    settings=settings,
                )
            except (ValueError, KeyError, Exception):
                pass

            if not backend or not model_name:
                return bullet_fallback

            prompt = self._PROFILE_SYNTHESIS_PROMPT.format(
                facts="\n".join(f"- {f}" for f in facts),
            )

            request = InternalChatRequest(
                model=model_name,
                messages=[Message(role="user", content=prompt)],
                stream=False,
                temperature=0.3,
                max_tokens=300,
            )
            response = await backend.chat(request)
            text = (response.message.content or "").strip()

            if len(text) < 20:
                return bullet_fallback

            profile = (
                "[user_context]\n"
                + text
                + "\n\nUse this context to shape your voice, tone, and depth — not to introduce subjects. Only reference these topics if the user brings them up first. Never list them back."
            )
            log.info("core_profile_synthesized", facts=len(facts), chars=len(profile))
            return profile

        except Exception:
            log.warning("core_profile_synthesis_failed", exc_info=True)
            return bullet_fallback
