"""KV resume ladder — one capability-selected session-resume path.

Design: docs/superpowers/specs/2026-06-26-kv-resume-ladder-low-latency-design.md

A returning session should never pay full prefill at the keyboard. There
are exactly two ways to get a KV cache back and they are rungs of one
ladder, not competitors:

1. RESTORE — load saved K/V tensors from a slot file. Cheap (I/O only),
   but structurally single-slot: llama-server's per-slot save/restore
   API 501s under ``--kv-unified``, so multi-slot configs never have
   slot files.
2. REPLAY  — recompute the KV by prefilling the session's stored prefix
   with ``n_predict=0`` (``prewarm_context``). Universal — works in
   every slot config and even across model swaps (it's just tokens) —
   but costs a real prefill, so it only ever runs in *free* windows
   (post-boot, session-open) and yields to live traffic.
3. COLD    — today's behavior; the first real request prefills.

Every rung failure falls through to the next; the cold floor is always
correct. No resume path may ever block or error a real request.

The replay *source* is captured below the mode layer: ``_manage_slot``
records the exact (post-augmentation) message list each keyed engine
request serves, so replay is byte-identical by construction — no mode
has to re-derive its trimmed history, and no reverse mapping from the
opaque ``kv_session_key`` to per-mode stores is needed. Narrative's
stable checkpoint upgrades the row with the assistant reply after each
turn. Rows live in the KV session manifest DB (``kv_replay_sources``).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.llama_cpp import LlamaCppBackend

log = get_logger(__name__)

# Upper bound on how many resume outcomes we retain for the status
# endpoint. Purely diagnostic; oldest entries drop first.
_RESULTS_KEEP = 32


class KVResumeLadder:
    """Capability-selected resume for one engine backend.

    Owned lazily by :class:`LlamaCppBackend` (``backend.resume_ladder``);
    the server manager reaches it through its attached backend for the
    post-READY boot warm. All state here is advisory — losing it costs
    at most a redundant prewarm, never correctness.
    """

    def __init__(self, backend: LlamaCppBackend) -> None:
        self._backend = backend
        # Per-session dedup so a rapid session-flip in the UI (or boot
        # warm racing an on-open trigger) never stacks duplicate
        # prefills of the same prefix.
        self._inflight: set[str] = set()
        # session_key -> outcome dict, insertion-ordered, bounded.
        self.last_results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def _manager(self):
        return self._backend._manager

    def _ready(self) -> bool:
        mgr = self._manager
        if mgr is None:
            return False
        from augmentum.models.llama_server_manager import ProcessState
        return mgr.state == ProcessState.READY

    def _record(self, session_key: str, outcome: dict[str, Any]) -> dict[str, Any]:
        outcome["ts"] = time.time()
        self.last_results[session_key] = outcome
        while len(self.last_results) > _RESULTS_KEEP:
            self.last_results.pop(next(iter(self.last_results)))
        return outcome

    @staticmethod
    def _replay_payload(row: dict[str, Any]) -> list[dict] | None:
        try:
            messages = json.loads(row.get("messages_json") or "")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(messages, list) or not messages:
            return None
        return messages

    # ------------------------------------------------------------------
    # the ladder
    # ------------------------------------------------------------------

    async def resume_session(self, session_key: str, *, source: str) -> dict[str, Any]:
        """Try to make ``session_key`` hot; return the outcome.

        ``source`` labels the warm window that fired this ("boot",
        "open", ...) — telemetry only. Returns a dict with at least
        ``rung`` ∈ {hot, restore, replay, cold, inflight, none} and,
        for cold, a ``reason``.
        """
        session_key = (session_key or "").strip()
        if not session_key:
            return {"rung": "none", "reason": "no_session_key"}
        if not self._ready():
            return {"rung": "none", "reason": "engine_not_ready"}
        if session_key in self._inflight:
            return {"rung": "inflight"}

        self._inflight.add(session_key)
        started = time.monotonic()
        try:
            outcome = await self._resume_inner(session_key)
        except Exception as exc:  # never break a caller over warm work
            log.warning("kv_resume_error", session=session_key, error=repr(exc))
            outcome = {"rung": "cold", "reason": f"error:{type(exc).__name__}"}
        finally:
            self._inflight.discard(session_key)

        outcome["source"] = source
        outcome["ms"] = round((time.monotonic() - started) * 1000.0, 1)
        log.info("kv_resume", session=session_key, **outcome)
        return self._record(session_key, outcome)

    async def _resume_inner(self, session_key: str) -> dict[str, Any]:
        backend = self._backend
        mgr = self._manager

        # Rung 0 — already hot in a live slot: nothing to do.
        if backend._get_slot_for_session(session_key) is not None:
            return {"rung": "hot"}

        multislot = backend._multislot_enabled()

        # Rung 1 — RESTORE (tensor load). Only exists where slot files
        # exist: single-slot with --slot-save-path live. The busy gate
        # matters because restore ERASES the target slot first — doing
        # that under a live request on slot 0 would destroy the KV that
        # request is using.
        slot_save = bool(getattr(mgr, "_slot_save_supported", False))
        if slot_save and not multislot and backend._slot_state_exists(session_key):
            if mgr.is_busy:
                return {"rung": "cold", "reason": "busy"}
            async with backend._get_slot_lock(0):
                displaced = backend._get_session_for_slot(0)
                if displaced and displaced != session_key:
                    # Mirror _manage_slot's displacement discipline: the
                    # slot's live KV may be newer than its disk copy.
                    await backend.save_session_state(displaced, slot_id=0)
                restored = await backend.restore_session_state(
                    session_key, slot_id=0,
                )
                if restored:
                    backend._claim_slot(0, session_key)
                    return {"rung": "restore", "slot": 0}
            # fall through to replay

        # Rung 2 — REPLAY (recompute). Universal, budget-priced.
        from augmentum.config import settings
        if not getattr(settings, "engine_kv_replay_enabled", True):
            return {"rung": "cold", "reason": "replay_disabled"}
        manifest = backend._kv_manifest()
        if manifest is None:
            return {"rung": "cold", "reason": "no_manifest"}
        row = await manifest.get_replay_source_async(session_key)
        if not row:
            return {"rung": "cold", "reason": "no_replay_source"}
        payload = self._replay_payload(row)
        if payload is None:
            return {"rung": "cold", "reason": "replay_source_unreadable"}

        if multislot:
            # Unpinned: llama-server's LCP similarity router lands the
            # prewarm on the best-matching slot, or an idle LRU slot
            # when nothing matches. It never displaces a busy slot, so
            # this is safe even while a chat is streaming elsewhere.
            target: int | None = None
        else:
            if mgr.is_busy:
                # One slot total — a pinned prewarm would queue the
                # user's own traffic behind a background prefill.
                return {"rung": "cold", "reason": "busy"}
            target = 0

        if target is None:
            warmed_slot = await backend.prewarm_context(payload, slot_id=None)
        else:
            async with backend._get_slot_lock(target):
                displaced = backend._get_session_for_slot(target)
                if displaced and displaced != session_key:
                    await backend.save_session_state(displaced, slot_id=target)
                    backend._release_slot(target)
                warmed_slot = await backend.prewarm_context(payload, slot_id=target)

        if warmed_slot is None:
            return {"rung": "cold", "reason": "replay_failed"}

        if warmed_slot >= 0:
            backend._claim_slot(warmed_slot, session_key)
        elif target is not None:
            # Pinned prewarm whose response didn't echo id_slot — trust
            # the pin.
            backend._claim_slot(target, session_key)
        # Tag for _manage_slot's tier telemetry: the next real request
        # for this session in multi-slot has no occupancy/checkpoint to
        # see, but its prefill will land on the replayed tokens — label
        # it so acceptance runs can tell "cold" from "replay-warmed".
        if mgr is not None:
            getattr(mgr, "_replay_warmed_keys", set()).add(session_key)
        return {
            "rung": "replay",
            "slot": warmed_slot if warmed_slot >= 0 else None,
            "messages": int(row.get("message_count") or 0),
        }

    # ------------------------------------------------------------------
    # boot warm (post-READY window)
    # ------------------------------------------------------------------

    async def warm_recent_sessions(self) -> None:
        """Warm the MRU sessions after boot, budget-bounded, preemptible.

        Candidates merge two ledgers: manifest rows (restore-capable,
        single-slot) and replay rows (universal). Only *successful*
        warms consume the session budget; every skip logs its reason
        and anything dropped by budget is counted — never a silent cap.
        """
        if not self._ready():
            return
        backend = self._backend
        mgr = self._manager
        manifest = backend._kv_manifest()
        if manifest is None:
            return

        from augmentum.config import settings
        max_sessions = int(getattr(settings, "engine_kv_replay_warm_sessions", 2) or 0)
        budget_s = float(getattr(settings, "engine_kv_replay_budget_s", 90.0) or 0.0)
        if not backend._multislot_enabled():
            # One physical slot holds one prefix — warming a second
            # session would just evict the first.
            max_sessions = min(max_sessions, 1)
        if max_sessions <= 0:
            log.info("kv_warm_summary", warmed=0, reason="budget_zero")
            return

        # Merge candidates MRU-first across both ledgers.
        now = time.time()
        candidates: dict[str, float] = {}
        model_key = backend._current_model_key()
        if model_key:
            for row in await asyncio.to_thread(
                manifest.list_model_sessions, model_key,
            ):
                key = str(row.get("session_key") or "")
                if key:
                    ts = float(row.get("last_accessed") or row.get("last_saved") or 0.0)
                    candidates[key] = max(candidates.get(key, 0.0), ts)
        for row in await manifest.list_replay_sources_async(limit=64):
            expires = float(row.get("expires_at") or 0.0)
            if expires and expires <= now:
                continue
            key = str(row.get("session_key") or "")
            if key:
                ts = float(row.get("updated_at") or 0.0)
                candidates[key] = max(candidates.get(key, 0.0), ts)

        ordered = sorted(candidates, key=lambda k: candidates[k], reverse=True)
        if not ordered:
            return

        started = time.monotonic()
        warmed: list[str] = []
        rungs: dict[str, int] = {}
        skipped: dict[str, int] = {}
        dropped = 0
        preempted = False
        for idx, key in enumerate(ordered):
            if len(warmed) >= max_sessions:
                dropped = len(ordered) - idx
                break
            if budget_s and (time.monotonic() - started) >= budget_s:
                dropped = len(ordered) - idx
                skipped["budget_time"] = skipped.get("budget_time", 0) + 1
                break
            if mgr is not None and mgr.is_busy:
                # A real request arrived — its own prefill IS the warm
                # for that session; burning GPU on the rest now would
                # compete with the user.
                preempted = True
                dropped = len(ordered) - idx
                break
            outcome = await self.resume_session(key, source="boot")
            rung = str(outcome.get("rung") or "cold")
            rungs[rung] = rungs.get(rung, 0) + 1
            if rung in ("restore", "replay", "hot"):
                warmed.append(key)
            else:
                reason = str(outcome.get("reason") or rung)
                skipped[reason] = skipped.get(reason, 0) + 1

        log.info(
            "kv_warm_summary",
            warmed=len(warmed),
            sessions=warmed,
            rungs=rungs,
            skipped=skipped,
            dropped_by_budget=dropped,
            preempted=preempted,
            candidates=len(ordered),
            elapsed_s=round(time.monotonic() - started, 1),
        )
