"""Speculative turn generation — rung 3 of the KV resume ladder.

While the user is mid-draft (typing pause, STT partial), the engine can
do better than warming the prefix: it can run the *whole turn* against
the draft on idle GPU cycles. When the real request arrives:

- byte-identical to the speculation → the finished answer streams with
  zero engine work (0ms TTFT);
- edited before send → the speculation is discarded, but its prefill
  already sits in a slot, so llama-server's LCP routing still skips the
  shared prefix — the user pays only the diff;
- the GPU got busy → speculation preempted instantly; nothing happened.

Hard rules, in priority order:

1. LOCAL ONLY. Speculation exists only on the managed llama-server
   backend. Draft text is unsent text — it must never leave the box,
   and cloud tokens cost money. This is a gate, not a default.
2. Real traffic always wins. Any non-speculative request entering the
   engine preempts an in-flight speculation before doing anything else.
3. Never serve a truncated answer. A speculation that stopped on a
   length cap the real request wouldn't have hit is *not servable* —
   its prefix warmth is kept, its text is dropped.
4. Drafts never touch disk. Entries live in process memory, TTL- and
   count-bounded, and the speculative request is excluded from the
   replay-source capture so no draft lands in sqlite.

Byte-exactness comes from the same place as the replay rung: the
predicted request is ``replay_source + [prior assistant] + [user:
draft]``, with sampling replayed from the previous captured request.
Whether a mode is exactly predictable is *measured*, not asserted —
``kv_speculate_result`` logs hit/miss per serve attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest, Usage
    from augmentum.models.llama_cpp import LlamaCppBackend

log = get_logger(__name__)

_RESULTS_KEEP = 32
_MAX_ENTRIES = 4
_DRAFT_MAX_CHARS = 32_768
_PREDICTED_MAX_BYTES = 4_000_000  # mirror replay-source skip-not-truncate

# Request fields that shape the completion. Symmetric between capture
# (previous turn) and serve (incoming turn): unset/default values are
# DROPPED so "absent" and "explicit default" fingerprint identically.
_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "max_tokens", "seed", "stop",
    "format", "think", "reasoning_effort", "preserve_thinking",
    "chat_template_kwargs",
)


def sampling_snapshot(request: Any) -> dict[str, Any]:
    """The completion-shaping fields of a request, defaults dropped."""
    out: dict[str, Any] = {}
    for name in _SAMPLING_FIELDS:
        value = getattr(request, name, None)
        if value is None or value is False:
            continue
        if isinstance(value, str | list | dict) and not value:
            continue
        out[name] = value
    return out


def serialize_text_messages(messages: Any) -> list[dict] | None:
    """Strict text-only [{role, content}] serialization, or None.

    Same eligibility contract as the replay-source capture: images and
    tool turns render through template branches a prediction can't
    reproduce, so those requests are never speculation candidates.
    """
    if not messages:
        return None
    out: list[dict] = []
    for m in messages:
        role = getattr(m, "role", None) or (
            m.get("role") if isinstance(m, dict) else ""
        )
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if role not in ("system", "user", "assistant"):
            return None
        if isinstance(m, dict):
            if m.get("tool_calls") or m.get("tool_call_id") or m.get("images"):
                return None
        else:
            if getattr(m, "tool_calls", None) or getattr(m, "tool_call_id", None):
                return None
            if getattr(m, "images", None):
                return None
        if not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    return out


def compute_fingerprint(
    model_id: str, messages: list[dict], sampling: dict[str, Any],
) -> str:
    """Deterministic digest of everything that must match for a serve."""
    blob = json.dumps(
        {"model": model_id, "messages": messages, "sampling": sampling},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


@dataclass
class SpecEntry:
    """One completed speculation, held in memory only."""

    session_key: str
    model_id: str
    fingerprint: str
    # Recorded stream deltas, replayed verbatim for fidelity.
    deltas: list[tuple[str, str]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Any = None
    created: float = 0.0  # time.monotonic()
    completion_chars: int = 0


class TurnSpeculator:
    """Draft-driven speculation for one managed engine backend.

    Owned lazily by :class:`LlamaCppBackend` (``backend.turn_speculator``).
    All state is advisory: losing it costs a redundant generation at
    worst, never correctness — the cold path is always right.
    """

    def __init__(self, backend: LlamaCppBackend) -> None:
        self._backend = backend
        self._entries: dict[str, SpecEntry] = {}
        self._task: asyncio.Task | None = None
        self._task_key = ""
        self._task_fp = ""
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

    def _store(self, entry: SpecEntry) -> None:
        self._entries[entry.session_key] = entry
        while len(self._entries) > _MAX_ENTRIES:
            self._entries.pop(next(iter(self._entries)))

    # ------------------------------------------------------------------
    # preemption — rule 2
    # ------------------------------------------------------------------

    async def preempt(self, reason: str) -> None:
        """Cancel any in-flight speculation and wait for it to unwind.

        Called at every real-request entry; must be near-free when no
        speculation is running (one attr check).
        """
        task = self._task
        if task is None or task.done():
            self._task = None
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # unwind errors are diagnostic only
            log.warning("kv_speculate_preempt_error", error=repr(exc))
        finally:
            self._task = None
            self._task_key = ""
            self._task_fp = ""
        log.info("kv_speculate_preempted", reason=reason)

    # ------------------------------------------------------------------
    # speculation (draft side)
    # ------------------------------------------------------------------

    async def speculate(
        self,
        session_key: str,
        *,
        draft: str,
        prior_assistant: str = "",
        source: str = "typing",
    ) -> dict[str, Any]:
        """Speculate one turn for ``session_key`` against ``draft``."""
        session_key = (session_key or "").strip()
        if not session_key:
            return {"status": "skip", "reason": "no_session_key"}
        started = time.monotonic()
        try:
            outcome = await self._speculate_inner(
                session_key, draft=draft, prior_assistant=prior_assistant,
            )
        except asyncio.CancelledError:
            outcome = {"status": "preempted"}
            self._record_outcome(session_key, outcome, source, started)
            raise
        except Exception as exc:  # speculation must never surface errors
            log.warning("kv_speculate_error", session=session_key, error=repr(exc))
            outcome = {"status": "error", "reason": type(exc).__name__}
        return self._record_outcome(session_key, outcome, source, started)

    def _record_outcome(
        self, session_key: str, outcome: dict[str, Any],
        source: str, started: float,
    ) -> dict[str, Any]:
        outcome["source"] = source
        outcome["ms"] = round((time.monotonic() - started) * 1000.0, 1)
        log.info("kv_speculate", session=session_key, **outcome)
        return self._record(session_key, outcome)

    async def _speculate_inner(
        self, session_key: str, *, draft: str, prior_assistant: str,
    ) -> dict[str, Any]:
        from augmentum.config import settings

        if not getattr(settings, "engine_speculation_enabled", False):
            return {"status": "skip", "reason": "disabled"}
        backend = self._backend
        mgr = self._manager
        # Rule 1 — LOCAL ONLY. No manager means no bundled llama-server;
        # a cloud/sidecar backend never sees a draft.
        if mgr is None or not self._ready():
            return {"status": "skip", "reason": "engine_not_ready"}
        if not (draft or "").strip():
            return {"status": "skip", "reason": "empty_draft"}
        if len(draft) > _DRAFT_MAX_CHARS:
            return {"status": "skip", "reason": "draft_oversize"}
        if mgr.is_busy:
            return {"status": "skip", "reason": "busy"}

        manifest = backend._kv_manifest()
        if manifest is None:
            return {"status": "skip", "reason": "no_manifest"}
        row = await manifest.get_replay_source_async(session_key)
        if not row:
            return {"status": "skip", "reason": "no_replay_source"}
        try:
            prefix = json.loads(row.get("messages_json") or "")
        except (json.JSONDecodeError, TypeError):
            return {"status": "skip", "reason": "replay_source_unreadable"}
        if not isinstance(prefix, list) or not prefix:
            return {"status": "skip", "reason": "replay_source_unreadable"}

        # Predicted next request: captured prefix, plus the assistant
        # reply the client saw (rows captured pre-reply lack it; rows
        # upgraded by a stable checkpoint already carry it), plus draft.
        predicted = list(prefix)
        tail = predicted[-1] if predicted else None
        prior_assistant = prior_assistant or ""
        if prior_assistant.strip() and not (
            isinstance(tail, dict)
            and tail.get("role") == "assistant"
            and tail.get("content") == prior_assistant
        ):
            predicted.append({"role": "assistant", "content": prior_assistant})
        predicted.append({"role": "user", "content": draft})
        if len(json.dumps(predicted, ensure_ascii=False)) > _PREDICTED_MAX_BYTES:
            return {"status": "skip", "reason": "oversize"}

        try:
            sampling = json.loads(row.get("sampling_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            sampling = {}
        if not isinstance(sampling, dict):
            sampling = {}

        model_id = str(getattr(mgr, "model_id", "") or "")
        fp = compute_fingerprint(model_id, predicted, sampling)

        existing = self._entries.get(session_key)
        if existing is not None and existing.fingerprint == fp:
            return {"status": "cached"}
        task = self._task
        if task is not None and not task.done() and self._task_fp == fp:
            return {"status": "inflight"}
        # A different draft supersedes the running speculation.
        await self.preempt("superseded")
        # The draft invalidates whatever we finished earlier for this
        # session — one entry per session, newest draft wins.
        self._entries.pop(session_key, None)

        prefill_only = bool(getattr(settings, "engine_speculation_prefill_only", False))
        if prefill_only or not sampling:
            # Without a captured sampling snapshot the fingerprint can
            # never match a real request — generating would be waste.
            # Prefix warmth is still the whole Enter-tax.
            return await self._prefill(session_key, predicted)

        return await self._generate(
            session_key, predicted, sampling=sampling,
            model_id=model_id, fingerprint=fp, mode=str(row.get("mode") or ""),
        )

    async def _prefill(
        self, session_key: str, predicted: list[dict],
    ) -> dict[str, Any]:
        """Level 1 — draft-aware prefix warmth (``n_predict=0``)."""
        backend = self._backend
        mgr = self._manager
        self._task = asyncio.current_task()
        self._task_key = session_key
        try:
            if backend._multislot_enabled():
                warmed_slot = await backend.prewarm_context(predicted, slot_id=None)
            else:
                async with backend._get_slot_lock(0):
                    displaced = backend._get_session_for_slot(0)
                    if displaced and displaced != session_key:
                        await backend.save_session_state(displaced, slot_id=0)
                        backend._release_slot(0)
                    warmed_slot = await backend.prewarm_context(predicted, slot_id=0)
        finally:
            self._task = None
            self._task_key = ""
        if warmed_slot is None:
            return {"status": "skip", "reason": "prefill_failed"}
        if warmed_slot >= 0:
            backend._claim_slot(warmed_slot, session_key)
        if mgr is not None:
            getattr(mgr, "_replay_warmed_keys", set()).add(session_key)
        return {"status": "prefix", "slot": warmed_slot if warmed_slot >= 0 else None}

    async def _generate(
        self,
        session_key: str,
        predicted: list[dict],
        *,
        sampling: dict[str, Any],
        model_id: str,
        fingerprint: str,
        mode: str,
    ) -> dict[str, Any]:
        """Level 2 — full speculative turn through the normal stream path.

        The request carries the real ``kv_session_key`` so slot
        affinity, displacement discipline, and tier telemetry all apply
        unchanged; ``_augmentum_speculative`` keeps it out of the
        replay-source capture (rule 4) and the preemption hook.
        """
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        cap = int(getattr(settings, "engine_speculation_max_new_tokens", 2048) or 0)
        capped = "max_tokens" not in sampling
        kwargs: dict[str, Any] = {
            k: v for k, v in sampling.items() if k in _SAMPLING_FIELDS
        }
        if capped and cap > 0:
            kwargs["max_tokens"] = cap
        request = InternalChatRequest(
            model=model_id,
            messages=[
                Message(role=m["role"], content=m["content"]) for m in predicted
            ],
            stream=True,
            kv_session_key=session_key,
            kv_mode=mode,
            **kwargs,
        )
        request._augmentum_speculative = True  # ad-hoc marker; see chat_stream

        deltas: list[tuple[str, str]] = []
        finish_reason: str | None = None
        usage: Usage | None = None
        self._task = asyncio.current_task()
        self._task_key = session_key
        self._task_fp = fingerprint
        started = time.monotonic()
        try:
            async for chunk in self._backend.chat_stream(request):
                content = getattr(chunk, "content_delta", "") or ""
                thinking = getattr(chunk, "thinking_delta", "") or ""
                if content or thinking:
                    deltas.append((content, thinking))
                if getattr(chunk, "finish_reason", None):
                    finish_reason = chunk.finish_reason
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
        finally:
            self._task = None
            self._task_key = ""
            self._task_fp = ""

        completion_chars = sum(len(c) for c, _ in deltas)
        gen_ms = round((time.monotonic() - started) * 1000.0, 1)
        # Rule 3 — never serve a truncated answer. A "length" stop on
        # our own injected cap is a speculation artifact the real
        # request wouldn't reproduce. (A cap the session itself carries
        # is fine — the real request truncates identically.)
        if finish_reason != "stop" and not (
            finish_reason == "length" and not capped
        ):
            return {
                "status": "prefix",
                "reason": f"not_servable:{finish_reason or 'no_finish'}",
                "gen_ms": gen_ms,
            }
        if completion_chars == 0:
            return {"status": "prefix", "reason": "empty_completion", "gen_ms": gen_ms}

        self._store(SpecEntry(
            session_key=session_key,
            model_id=model_id,
            fingerprint=fingerprint,
            deltas=deltas,
            finish_reason=finish_reason or "stop",
            usage=usage,
            created=time.monotonic(),
            completion_chars=completion_chars,
        ))
        return {
            "status": "ready",
            "chars": completion_chars,
            "chunks": len(deltas),
            "gen_ms": gen_ms,
        }

    # ------------------------------------------------------------------
    # serve (real-request side)
    # ------------------------------------------------------------------

    async def on_real_request(self, request: InternalChatRequest) -> SpecEntry | None:
        """Preempt speculation; return a byte-matching entry if one exists.

        Called at the head of every non-speculative ``chat_stream``.
        Returning an entry means the caller streams it verbatim and
        skips the engine entirely; the entry is consumed (one-shot).
        """
        inflight_key = self._task_key
        await self.preempt("real_traffic")

        session_key = (request.kv_session_key or "").strip()
        if not session_key:
            return None
        entry = self._entries.get(session_key)
        if entry is None:
            if inflight_key == session_key:
                log.info(
                    "kv_speculate_result", session=session_key,
                    outcome="miss", reason="inflight_preempted",
                )
            return None

        reason = self._serve_mismatch(entry, request)
        if reason is not None:
            self._entries.pop(session_key, None)
            log.info(
                "kv_speculate_result", session=session_key,
                outcome="miss", reason=reason,
            )
            return None

        self._entries.pop(session_key, None)
        log.info(
            "kv_speculate_result", session=session_key, outcome="hit",
            chars=entry.completion_chars, chunks=len(entry.deltas),
        )
        return entry

    def _serve_mismatch(
        self, entry: SpecEntry, request: InternalChatRequest,
    ) -> str | None:
        """Why ``entry`` cannot serve ``request``, or None if it can."""
        from augmentum.config import settings

        if not getattr(settings, "engine_speculation_enabled", False):
            return "disabled"
        ttl = float(getattr(settings, "engine_speculation_ttl_s", 180.0) or 0.0)
        if ttl and (time.monotonic() - entry.created) > ttl:
            return "stale"
        mgr = self._manager
        model_id = str(getattr(mgr, "model_id", "") or "") if mgr else ""
        if not model_id or model_id != entry.model_id:
            return "model_changed"
        if request.tools or request.tool_choice:
            return "tools"
        payload = serialize_text_messages(request.messages)
        if payload is None:
            return "non_text_request"
        sampling = sampling_snapshot(request)
        fp = compute_fingerprint(model_id, payload, sampling)
        if fp != entry.fingerprint:
            return "draft_mismatch"
        return None

    async def replay_chunks(self, entry: SpecEntry, request: InternalChatRequest):
        """Yield the recorded stream verbatim, labeled as speculative."""
        from augmentum.models.base import InternalStreamChunk

        first = True
        for content_delta, thinking_delta in entry.deltas:
            chunk = InternalStreamChunk(
                content_delta=content_delta,
                thinking_delta=thinking_delta,
                model=request.model,
            )
            if first:
                chunk.role = "assistant"
                chunk.augmentum = {"speculative": True}
                first = False
            yield chunk
            # Let the event loop interleave the writer — a recorded
            # stream otherwise floods the socket in one scheduling turn.
            await asyncio.sleep(0)
        yield InternalStreamChunk(
            content_delta="",
            model=request.model,
            finish_reason=entry.finish_reason,
            usage=entry.usage,
        )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        task = self._task
        return {
            "inflight": bool(task is not None and not task.done()),
            "inflight_session": self._task_key,
            "entries": {
                key: {
                    "chars": e.completion_chars,
                    "chunks": len(e.deltas),
                    "age_s": round(time.monotonic() - e.created, 1),
                    "model": e.model_id,
                }
                for key, e in self._entries.items()
            },
            "results": self.last_results,
        }
