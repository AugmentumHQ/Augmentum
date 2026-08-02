"""Backend for llama.cpp's built-in HTTP server (llama-server).

llama-server exposes an OpenAI-compatible API, so this backend reuses much
of the OpenAI conversion logic but adds llama.cpp-specific endpoints
(slots, tokenize, detokenize, LoRA, health, router-mode model management).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import httpx

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.models.base import (
    v1_entry_is_vision as _v1_entry_is_vision,
)
from augmentum.models.kv_reuse_audit import KvReuseAuditMixin
from augmentum.proxy.status_bus import Stage
from augmentum.utils.logging import get_logger
from augmentum.utils.thinking import (
    ThinkingStreamBuffer,
    detect_reasoning_family,
    normalize_thinking,
)


@dataclass
class SlotOccupancy:
    """Per-slot occupancy record: which session_key currently has its
    KV state in this physical slot.

    Slot IDs are opaque physical handles owned by llama-server; sessions
    are content-addressed identifiers owned by Augmentum. The mapping
    is maintained lazily — observed from completion responses' id_slot
    field (Phase 2+) and updated via _claim_slot. Multi-slot
    architecture spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md
    """
    slot_id: int
    session_key: str
    last_observed_mono: float

log = get_logger(__name__)


# Process-lifetime dedup for warnings emitted from the model-discovery
# path. ``list_models()`` is polled (every UI refresh / dropdown
# render / model-map sync), so any per-model warning emitted from it
# would otherwise repeat every few seconds for the lifetime of the
# process. In a 2h sample, this cluster accounted for ~93% of all
# warnings, drowning real ones. Each key collapses identical events
# to a single emission; the underlying condition is still surfaced
# once, just not re-spammed on every poll.
_collision_logged: set[tuple[str, str, tuple[str, ...]]] = set()

# llama.cpp sampling parameters that live in raw_options and map directly
# to the /v1/chat/completions payload (no name translation needed).
_LLAMACPP_PASSTHROUGH_PARAMS = frozenset({
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "dynatemp_range",
    "dynatemp_exponent",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "xtc_probability",
    "xtc_threshold",
    "grammar",
    "json_schema",
    "samplers",
    "logit_bias",
    "n_probs",
    "min_keep",
    "ignore_eos",
    "cache_prompt",
    "response_format",
})


def kv_restore_skip_reason(record: dict, runtime: dict) -> str | None:
    """Pure function: should a stored KV slot be rejected on restore?

    Compares a manifest ``record`` against the live ``runtime``
    signature (as returned by ``LlamaServerManager.current_runtime_signature``).
    Returns a short human-readable reason when the slot is incompatible,
    or ``None`` when it's safe to restore.

    Lives at module scope so the manager's restart-warm path can use
    it without importing or holding a backend instance.
    """
    stored_model_id = (record.get("model_id") or "").strip()
    runtime_model_id = (runtime.get("model_id") or "").strip()
    if stored_model_id and runtime_model_id and stored_model_id != runtime_model_id:
        return "model changed"

    stored_model_path = (record.get("model_path") or "").strip()
    runtime_model_path = (runtime.get("model_path") or "").strip()
    if stored_model_path and runtime_model_path and stored_model_path != runtime_model_path:
        return "model path changed"

    stored_model_mtime = float(record.get("model_mtime") or 0.0)
    runtime_model_mtime = float(runtime.get("model_mtime") or 0.0)
    if stored_model_mtime and runtime_model_mtime and stored_model_mtime != runtime_model_mtime:
        return "model file changed"

    stored_ctx = int(record.get("ctx_size") or 0)
    runtime_ctx = int(runtime.get("ctx_size") or 0)
    if stored_ctx and runtime_ctx and stored_ctx != runtime_ctx:
        return "context size changed"

    stored_kv_type = (record.get("kv_cache_type") or "").strip()
    runtime_kv_type = (runtime.get("kv_cache_type") or "").strip()
    if stored_kv_type != runtime_kv_type:
        return "KV cache type changed"

    # KV-layout-affecting load options. We compare strictly here: any
    # mismatch (including a 0/'' default vs. a real value) fails the
    # restore. This intentionally invalidates pre-2.5 manifest rows
    # that predate these columns — one-time cold prefill is cheaper
    # than restoring an incompatible KV.
    if bool(record.get("flash_attn")) != bool(runtime.get("flash_attn")):
        return "flash_attn changed"

    runtime_mode = str(runtime.get("gpu_layers_mode") or "").strip()
    stored_mode = str(record.get("gpu_layers_mode") or "").strip()
    if stored_mode != runtime_mode:
        return "gpu_layers_mode changed"

    # In auto mode the layer count is reshuffled per load; tolerate
    # mismatches there. With an explicit/manual mode, any change of
    # layer count means a different KV layout.
    if runtime_mode != "auto":
        stored_layers = int(record.get("gpu_layers") or 0)
        runtime_layers = int(runtime.get("gpu_layers") or 0)
        if stored_layers != runtime_layers:
            return "gpu_layers changed"

    stored_batch = int(record.get("batch_size") or 0)
    runtime_batch = int(runtime.get("batch_size") or 0)
    if stored_batch != runtime_batch:
        return "batch_size changed"

    # Speculative decoding swap: invalidate as cheap insurance
    # against subtle generation-time mismatches.
    stored_draft = (record.get("draft_model") or "").strip()
    runtime_draft = (runtime.get("draft_model") or "").strip()
    if stored_draft != runtime_draft:
        return "draft_model changed"

    stored_dmax = int(record.get("draft_max") or 0)
    runtime_dmax = int(runtime.get("draft_max") or 0)
    if stored_dmax != runtime_dmax:
        return "draft_max changed"

    # Architectural fingerprints (llama.cpp Discussion #15569 must-match
    # list). Compared symmetrically: a 0/missing on EITHER side is
    # tolerated so manifest rows that predate these columns (or were
    # written while no model was loaded) still get a chance to restore.
    # Both sides being non-zero AND mismatched means the underlying
    # model architecture differs — reject.
    stored_embed = int(record.get("n_embed") or 0)
    runtime_embed = int(runtime.get("n_embed") or 0)
    if stored_embed and runtime_embed and stored_embed != runtime_embed:
        return "n_embed changed"

    stored_layers_total = int(record.get("n_layers_total") or 0)
    runtime_layers_total = int(runtime.get("n_layers_total") or 0)
    if stored_layers_total and runtime_layers_total and stored_layers_total != runtime_layers_total:
        return "n_layers_total changed"

    stored_heads_kv = int(record.get("n_heads_kv") or 0)
    runtime_heads_kv = int(runtime.get("n_heads_kv") or 0)
    if stored_heads_kv and runtime_heads_kv and stored_heads_kv != runtime_heads_kv:
        return "n_heads_kv changed"

    return None


class LlamaCppBackend(KvReuseAuditMixin, ModelBackend):
    """Backend for llama.cpp's built-in HTTP server."""

    # llama-server's slot reuse prefix-matches at the token level, so
    # stable mid-conversation system injection (narrative STATE/MEMORY
    # right before the latest user turn) lets the slot cache hit every
    # turn. See ``narrative/engine.py::_augment_request`` for why this
    # injection point was chosen.
    supports_mid_conversation_system = True

    # Per-call timeout for inference HTTP requests to llama-server. The
    # shared ``http_client`` uses ``http_read_timeout`` (default 600s)
    # which is fine for memory extraction / provider polling / web
    # fetches but too short for cold prefill on large contexts:
    # 90k tokens on Qwen3.6-35B-A3B (24GB VRAM) takes 5-7 minutes
    # of llama-server silence before the first SSE byte arrives, and
    # even longer on next-gen 200k-context models. A user hitting the
    # 600s default saw httpx raise ReadTimeout mid-prefill at exactly
    # 10 minutes, augmentum logged ``backend_timeout_during_stream``,
    # Caddy then logged ``i/o timeout`` reading from augmentum, and
    # the chat errored. ``read=3600.0`` (1 hour) bounds genuinely-stuck
    # llama-server processes while not interrupting any plausible
    # prefill window. ``connect`` / ``write`` / ``pool`` stay tight so
    # a dead upstream still fails fast at connection time.
    _INFERENCE_TIMEOUT = httpx.Timeout(
        connect=5.0, read=3600.0, write=30.0, pool=5.0,
    )

    # OOM retry policy — multiplicative GPU-layer backoff. Each attempt
    # past the first reduces the GPU layer count by
    # ``_OOM_RETRY_BACKOFF_STEP`` (×0.15 = 15%) relative to the manager's
    # autofit baseline, capped at ``_OOM_RETRY_MAX_ATTEMPTS`` retries.
    # Replaces an earlier single 75%-drop retry that overshot the common
    # case (a model that almost-fits at autofit usually fits at 90%, not
    # 75%). Mirrors the multiplicative pattern in Ollama's
    # ``llm/server.go::Load``.
    _OOM_RETRY_MAX_ATTEMPTS = 5
    _OOM_RETRY_BACKOFF_STEP = 0.15

    # TTL for the list_models() cache (see __init__). Short enough that
    # operators dropping a new GGUF into a model_dir see it appear in
    # the UI within ~15s; long enough that the 4-5 callers polling
    # /api/tags every few seconds collapse to one disk-scan per window.
    _MODELS_CACHE_TTL_S = 15.0

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None = None,
        server_manager: object | None = None,
    ) -> None:
        self._client = http_client
        self._base_url = base_url.rstrip("/")
        # Strip /v1 suffix — this backend hardcodes /v1/ in all paths
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._api_key = api_key
        self._manager = server_manager  # LlamaServerManager for lazy model loading
        if server_manager is not None:
            # Reverse link so the manager's post-READY boot warm can run
            # the resume ladder (restore→replay→cold) instead of the
            # legacy restore-only walk. Same process lifecycle — a plain
            # attribute, no weakref ceremony needed.
            server_manager._engine_backend = self
        # Slot occupancy tracking — multi-slot architecture foundation.
        # Phase 1 keeps single-slot semantics (everything operates on
        # slot 0) but uses dict-based storage so Phase 2 can add
        # multi-slot routing without a structural refactor.
        # Forward index: slot_id → SlotOccupancy (which session is in this slot)
        # Inverse index: session_key → slot_id (which slot has this session)
        # Both indexes are kept in sync via _claim_slot / _release_slot.
        self._slot_occupancy: dict[int, SlotOccupancy] = {}
        self._session_to_slot: dict[str, int] = {}
        # Latch: set once the server answers 501 to a slot save/restore — that
        # means it was started WITHOUT ``--slot-save-path``, so KV-cache slot
        # I/O isn't available at all. The capability can't change without a
        # server restart (which builds a fresh backend), so we stop attempting
        # and stop logging it every turn. See save_slot / restore_slot.
        self._slot_io_unsupported = False
        self._ensure_lock: asyncio.Lock | None = None
        # Per-slot lifecycle lock dict. Phase 1: only slot 0 is used.
        # Phase 2+: each slot gets independent locking so concurrent
        # requests on different slots don't serialize. Held from
        # _manage_slot through generation through save_session_state
        # so no other request can mutate the slot mid-stream. Also
        # guards prepare_stable_checkpoint so checkpoint prep can't
        # race a regular request on the same slot.
        self._slot_locks: dict[int, asyncio.Lock] = {}
        # In-flight tokenize coalescing for the full-prompt cache.
        # Keyed by ``full:<hash>``; concurrent requests with the same
        # rendered prompt share one tokenize round-trip and one cache
        # write instead of racing duplicate work.
        self._tokenize_inflight: dict[str, asyncio.Future[list[int] | None]] = {}
        # Short-lived cache for the managed-server ``list_models()``
        # output. /api/tags is polled by Open WebUI, the Model Manager,
        # and the Settings modal — each call drives discover_gguf_files
        # (filesystem walk) + per-model _find_paired_mmproj (sidecar
        # reads + GGUF metadata reads). Pre-cache this was 600-820ms
        # per /api/tags hit. The file set rarely changes, so a 15s TTL
        # is a clean speed win with no UX regression on model add/remove
        # (just a small delay before they appear in the dropdown).
        # Currently-loaded model is spliced AFTER the cache lookup so
        # it always reflects live state. Lock is lazy (asyncio.Lock
        # needs a running loop, mirrors _ensure_lock pattern above).
        self._models_cache: tuple[float, list[ModelInfo]] | None = None
        self._models_cache_lock: asyncio.Lock | None = None
        self._list_models_warned = False
        # Stable-prefix contract tracking. Keyed by kv_session_key (the
        # per-conversation key — NOT the per-turn stable-checkpoint
        # fingerprint). Holds the previous turn's (role, content) list so
        # each turn can measure WHERE its payload diverged from the last
        # one. KV reuse (in-slot prefix match, RAM-cache restore f_keep,
        # hybrid-model checkpoint validity) is bounded by this divergence
        # point, so a mode that mutates history mid-prefix silently
        # forfeits all reuse — this metric makes that visible per-mode
        # instead of surfacing as unexplained cold prefills.
        # Request-side prefix-stability tracking + joined contract verdict
        # (see KvReuseAuditMixin). Shared with OpenAICompatBackend so the
        # same reuse audit runs for remote providers.
        self._init_kv_audit()

    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth if configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _ensure_server(self, model: str = "") -> None:
        """Ensure llama-server is running, auto-loading a model if needed.

        Handles:
        - Lazy loading: starts server on first request
        - Model switching: swaps when a different model is requested
        - Crash recovery: detects dead process, restarts automatically
        - OOM retry: if startup fails, retries with fewer GPU layers
        """
        if self._manager is None:
            return

        if self._ensure_lock is None:
            self._ensure_lock = asyncio.Lock()

        async with self._ensure_lock:
            await self._ensure_server_locked(model)

    async def _ensure_server_locked(self, model: str = "") -> None:
        """Inner ensure implementation guarded by ``_ensure_lock``."""
        if self._manager is None:
            return

        from augmentum.models.llama_server_manager import ProcessState

        # Treat 'default' or empty as "use whatever is loaded"
        is_generic = not model or model.lower() == "default"

        # Check if process is still alive (handles crash detection)
        if self._manager.state == ProcessState.READY:
            if not self._manager.check_alive():
                log.warning("engine_v2_crash_detected", model=model)
                # Process died — fall through to restart logic below.
                # All slots' KV is gone with the subprocess; clear the
                # occupancy tracker so subsequent restore attempts
                # don't trust a stale snapshot of who-was-where.
                self._slot_occupancy.clear()
                self._session_to_slot.clear()
            else:
                # Server running and healthy
                if not is_generic and model != self._manager.model_id:
                    path = self._manager._resolve_model_path(model)
                    if path:
                        log.info("engine_v2_model_swap",
                                 from_model=self._manager.model_id, to_model=model)
                        await self._manager.swap(path)
                return

        # Server not running — resolve model to a GGUF path
        path = None
        if not is_generic:
            path = self._manager._resolve_model_path(model)

        # If we crashed, try to restart the same model
        if not path and self._manager._last_crashed_model:
            path = self._manager._last_crashed_model
            log.info("engine_v2_restarting_after_crash", model=path)

        if not path:
            files = self._manager.discover_gguf_files()
            available = [f["filename"] for f in files[:10]]
            if available:
                model_list = ", ".join(available)
                raise RuntimeError(
                    f"No model selected. Available models: {model_list}. "
                    f"Select one from the model dropdown."
                )
            raise RuntimeError(
                "No GGUF models found. Download a model or add a model "
                "directory in Settings > Manage Providers."
            )

        # Try to start with multiplicative GPU-layer backoff on OOM.
        await self._start_with_oom_backoff(path)

    # Fraction of currently-available host RAM a single model load may
    # claim. Deliberately below 1.0: the engine is not the only consumer
    # in the container, and an admission gate that hands out the last
    # byte has not gated anything.
    _HOST_RAM_SPILL_BUDGET_FRAC = 0.80

    @staticmethod
    def _spill_bytes(profile: Any, gpu_layers: int) -> int:
        """Bytes of model weight that will live in HOST RAM at this offload.

        Weights are apportioned by layer count, which is approximate (the
        embedding/output tensors are not per-layer) but correct in the
        direction that matters: it never *under*-estimates the spill for
        the low-gpu-layers cases where the risk is real.
        """
        total = int(getattr(profile, "total_size_bytes", 0) or 0)
        n_layers = int(getattr(profile, "n_layers", 0) or 0)
        if total <= 0 or n_layers <= 0:
            return 0
        on_cpu = max(0, n_layers - max(0, int(gpu_layers)))
        return int(total * (on_cpu / n_layers))

    async def _host_ram_can_absorb_spill(
        self, *, profile: Any, gpu_layers: int,
    ) -> tuple[bool, str]:
        """Would offloading to ``gpu_layers`` fit in available host RAM?

        Returns ``(ok, reason)``. Unknown data admits (``True``) — the same
        conservatism as ``check_engine_fit``, since a model with no profile
        yet must be allowed its first load.
        """
        from augmentum.resource import hostmem

        needed = self._spill_bytes(profile, gpu_layers)
        if needed <= 0:
            return True, ""
        info = hostmem.memory_info()
        needed_mib = needed // (1024 * 1024)
        budget = int(info.available_mib * self._HOST_RAM_SPILL_BUDGET_FRAC)
        if needed_mib <= budget:
            return True, ""
        scope = (
            f", container limit {info.total_mib / 1024:.1f} GB"
            if info.limited
            else ""
        )
        return False, (
            f"~{needed_mib / 1024:.1f} GB would spill to host RAM but only "
            f"~{info.available_mib / 1024:.1f} GB is available{scope}"
        )

    async def _warn_if_host_ram_tight(self) -> None:
        """Surface low host memory BEFORE a load, not after the box dies.

        Advisory only: this does not block. The blocking decision belongs
        to the spill check, which knows the actual size involved.
        """
        try:
            from augmentum.resource import hostmem

            info = hostmem.memory_info()
            if info.limited and info.available_mib < 2048:
                log.warning(
                    "engine_v2_host_ram_tight",
                    available_mib=info.available_mib,
                    total_mib=info.total_mib,
                    source=info.source,
                    note="Model load starting with little host RAM left.",
                )
        except Exception:  # pragma: no cover - advisory path only
            pass

    @staticmethod
    def _is_oom_class_error(exc: BaseException) -> bool:
        """Heuristic: is this RuntimeError an OOM-class startup failure?

        llama-server emits varied messages when it exits during startup
        from insufficient VRAM/RAM (``exited``, ``oom``, ``out of
        memory``); match the union and stay conservative on other
        RuntimeErrors. Reused by tests to assert the OOM gate matches
        production strings.
        """
        if not isinstance(exc, RuntimeError):
            return False
        err = str(exc).lower()
        return "exited" in err or "oom" in err or "memory" in err

    async def _start_with_oom_backoff(self, path: str) -> None:
        """Start the model with multiplicative GPU-layer backoff on OOM.

        First attempt uses the manager's autofit. Each subsequent attempt
        re-uses the autofit baseline scaled by
        ``1.0 - _OOM_RETRY_BACKOFF_STEP * attempt`` (×0.85, ×0.70, …).
        Stops when start succeeds, the layer count reaches 0, or
        ``_OOM_RETRY_MAX_ATTEMPTS`` retries are exhausted.

        Mirrors Ollama's ``llm/server.go::Load`` pattern: finer-grained
        than the previous single 75%-drop retry, which overshot
        almost-fits-at-autofit cases. Non-OOM RuntimeErrors propagate
        immediately on the first failure — backoff only applies to
        memory-class crashes.
        """
        if self._manager is None:
            # Should be unreachable: callers gate on this. Defensive
            # raise so callers see a clear error rather than an
            # AttributeError later.
            raise RuntimeError(
                "_start_with_oom_backoff invoked without an attached "
                "LlamaServerManager"
            )

        await self._warn_if_host_ram_tight()

        autofit_layers: int | None = None

        for attempt in range(self._OOM_RETRY_MAX_ATTEMPTS + 1):
            try:
                if attempt == 0:
                    log.info("engine_v2_lazy_load", model_path=path)
                    await self._manager.start(path)
                else:
                    # On first retry, capture the autofit baseline.
                    # Cached for subsequent retries since profile +
                    # detected VRAM don't change between attempts; the
                    # T1-7 GPU-info cache means recomputing autofit is
                    # also cheap, but caching keeps logs deterministic.
                    if autofit_layers is None:
                        profile = self._manager._last_profile
                        if profile is None or profile.n_layers <= 0:
                            log.error(
                                "engine_v2_oom_retry_no_profile",
                                attempt=attempt,
                            )
                            raise RuntimeError(
                                "OOM retry needs a profile to compute "
                                f"backoff; none cached after attempt {attempt}"
                            )
                        autofit_layers = self._manager._autofit_gpu_layers(profile)

                    factor = max(0.0, 1.0 - self._OOM_RETRY_BACKOFF_STEP * attempt)
                    reduced = max(0, int(autofit_layers * factor))
                    # Reducing GPU layers does not make the weights smaller —
                    # it MOVES them into host RAM. Without this check the
                    # ladder silently converts VRAM pressure into host-memory
                    # pressure, one rung at a time, and on 2026-07-25 that
                    # walked a 128 GB machine into a forced restart (bug B2).
                    # Refuse the spill we cannot afford instead of taking the
                    # whole box down for one model load.
                    spill_ok, spill_reason = await self._host_ram_can_absorb_spill(
                        profile=self._manager._last_profile,
                        gpu_layers=reduced,
                    )
                    if not spill_ok:
                        log.error(
                            "engine_v2_oom_retry_refused_host_ram",
                            attempt=attempt,
                            reduced_layers=reduced,
                            reason=spill_reason,
                        )
                        raise RuntimeError(
                            "Model doesn't fit in VRAM, and offloading the "
                            f"remainder to host RAM would not fit either: "
                            f"{spill_reason}. Free memory, use a smaller "
                            "quantization, or reduce the context size."
                        )
                    log.info(
                        "engine_v2_oom_retry",
                        attempt=attempt,
                        autofit_layers=autofit_layers,
                        reduced_layers=reduced,
                        factor=round(factor, 2),
                    )
                    await self._manager.start(
                        path, gpu_layers_override=reduced,
                    )
                return  # success
            except RuntimeError as exc:
                if not self._is_oom_class_error(exc):
                    # Non-OOM failure — propagate immediately, no retry.
                    raise
                # Partial-offload incompatibility (Qwen 3.5/3.6 Gated Delta Net
                # hybrid-SSM blocks) is latched by the status parser when it
                # observes the sched_reserve warning. No layer-count reduction
                # can rescue this — the architecture requires layer 0 on the
                # same device as the fused GDN tensor. Bail with a useful
                # message instead of cascading 6 doomed retries (each ~5s of
                # subprocess spin-up + abort).
                if getattr(self._manager, "_partial_offload_incompatible", False):
                    log.error(
                        "engine_v2_partial_offload_incompatible",
                        attempts=attempt + 1,
                        note=(
                            "Architecture cannot partial-offload. "
                            "Reduce ctx_size or load a smaller quant."
                        ),
                    )
                    raise RuntimeError(
                        "Model architecture requires full-GPU or full-CPU "
                        "offload (Gated Delta Net fused tensor). The model "
                        "doesn't fit at the current ctx_size/quant — try "
                        "a smaller context or a smaller quantization."
                    ) from exc
                if attempt >= self._OOM_RETRY_MAX_ATTEMPTS:
                    log.error(
                        "engine_v2_oom_retry_exhausted",
                        attempts=attempt + 1,
                        last_error=str(exc)[:200],
                    )
                    raise
                log.warning(
                    "engine_v2_start_failed_retrying",
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                )

    # ------------------------------------------------------------------
    # Segment-level token caching (all modes)
    # ------------------------------------------------------------------

    def _get_cache(self):
        """Get the token cache, or None if unavailable.

        Reads the manager's public ``token_cache`` property. Uses
        ``getattr`` with a ``None`` default so duck-typed test fakes
        that don't implement the property still work — callers all
        tolerate ``None`` by falling back to uncached paths.
        """
        if self._manager is None:
            return None
        return getattr(self._manager, "token_cache", None)

    def _claim_slot(self, slot_id: int, session_key: str) -> None:
        """Atomically update occupancy: ``slot_id`` now holds ``session_key``.

        Prunes stale entries from BOTH indexes when the relationship has
        changed:

        - If slot_id was previously holding a different session, that
          session's inverse-index entry is dropped (it's no longer in
          this slot).
        - If session_key was previously in a different slot, that slot's
          forward-index entry is dropped (slot is now empty from our
          POV; engine may still have the KV in --cache-ram).

        Empty session_key is a no-op (opaque external requests).
        """
        if not session_key:
            return

        prev_at_slot = self._slot_occupancy.get(slot_id)
        if prev_at_slot is not None and prev_at_slot.session_key != session_key:
            self._session_to_slot.pop(prev_at_slot.session_key, None)

        old_slot = self._session_to_slot.get(session_key)
        if old_slot is not None and old_slot != slot_id:
            self._slot_occupancy.pop(old_slot, None)

        self._slot_occupancy[slot_id] = SlotOccupancy(
            slot_id=slot_id,
            session_key=session_key,
            last_observed_mono=time.monotonic(),
        )
        self._session_to_slot[session_key] = slot_id

    def _release_slot(self, slot_id: int) -> None:
        """Mark slot_id as unoccupied. Both indexes pruned. Used on
        engine crash, explicit eviction, or before a re-prewarm
        sequence where the prior occupant is no longer authoritative.
        """
        occ = self._slot_occupancy.pop(slot_id, None)
        if occ is not None:
            self._session_to_slot.pop(occ.session_key, None)

    def _get_session_for_slot(self, slot_id: int) -> str:
        """Return the session_key currently in ``slot_id``, or empty
        string if the slot is unoccupied (or never observed).
        """
        occ = self._slot_occupancy.get(slot_id)
        return occ.session_key if occ is not None else ""

    def _get_slot_for_session(self, session_key: str) -> int | None:
        """Return the slot_id currently holding ``session_key``, or
        ``None`` if the session isn't known to be in any slot.
        """
        if not session_key:
            return None
        return self._session_to_slot.get(session_key)

    def _slot_count(self) -> int:
        """Effective parallel-slot count.

        When ``engine_multislot_enabled`` is on we run llama-server with
        ``--parallel -1`` which the upstream binary at b8935 hardcodes
        to 4 (verified — see Phase 0 spec). If the user pinned
        ``engine_parallel_slots`` we honour that.

        When ``engine_multislot_enabled`` is off, we run with
        ``--parallel 1`` so only slot 0 exists.
        """
        from augmentum.config import settings
        if not getattr(settings, "engine_multislot_enabled", False):
            return 1
        pinned = getattr(settings, "engine_parallel_slots", None)
        if pinned is not None and pinned > 0:
            return int(pinned)
        return 4  # llama-server's auto default at b8935

    def _pick_restore_target_slot(self) -> int:
        """Pick a slot to restore an on-disk checkpoint into.

        Strategy:
          1. First unoccupied slot in [0, slot_count). Empty slots cost
             nothing to overwrite.
          2. If all occupied, evict the LRU one (oldest
             ``last_observed_mono``). The displaced KV survives in
             ``--cache-ram`` (with ``--cache-idle-slots`` default-on),
             and the displaced session has a disk checkpoint regardless,
             so the cost is bounded — at worst we pay a ghost-slot →
             live-slot restore on its next access.
        """
        n = self._slot_count()
        for sid in range(n):
            if sid not in self._slot_occupancy:
                return sid
        # All occupied — pick LRU.
        lru = min(
            self._slot_occupancy.values(),
            key=lambda occ: occ.last_observed_mono,
        )
        return lru.slot_id

    def _pick_checkpoint_target_slot(self, *, avoid: int | None = None) -> int:
        """Pick a slot for ``prepare_stable_checkpoint``'s prewarm.

        The whole point of moving prewarm off slot 0 is to stop blocking
        user-facing chat behind a 5-10 s background prefill. So we
        prefer a slot OTHER than the one currently serving chat
        (``avoid``).

        Phase 1 single-slot fallback returns 0 (only slot exists);
        Phase 2 multi-slot picks a non-avoid slot, preferring unoccupied.
        """
        n = self._slot_count()
        if n <= 1:
            return 0
        # Tier 1: unoccupied non-avoid slot.
        for sid in range(n):
            if sid == avoid:
                continue
            if sid not in self._slot_occupancy:
                return sid
        # Tier 2: LRU non-avoid slot.
        candidates = [
            occ for occ in self._slot_occupancy.values()
            if occ.slot_id != avoid
        ]
        if candidates:
            return min(candidates, key=lambda occ: occ.last_observed_mono).slot_id
        # Tier 3: fall back to avoid (only one slot is available — better
        # to block briefly than to fail). Or 0 if nothing claimed.
        return avoid if avoid is not None else 0

    def _multislot_enabled(self) -> bool:
        """Resolve the multi-slot tri-state setting to a concrete bool.

        ``None`` (the default for users who haven't toggled the
        setting) means "auto" — fall through to the codebase's current
        recommended default. ``True`` and ``False`` are explicit user
        overrides and pass through unchanged. See
        ``augmentum.proxy.status_bus.MULTISLOT_DEFAULT_ENABLED`` for
        the recommendation history and how to flip it.
        """
        from augmentum.config import settings
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        val = getattr(settings, "engine_multislot_enabled", None)
        if val is None:
            return MULTISLOT_DEFAULT_ENABLED
        return bool(val)

    def _get_slot_lock(self, slot_id: int = 0) -> asyncio.Lock:
        """Per-slot lifecycle lock. Lazily constructed per slot_id.

        Lazy because ``asyncio.Lock()`` historically required a running
        event loop, and this class is sometimes constructed in a sync
        startup path before the loop spins up.

        Phase 1 callers default slot_id=0 and behave identically to the
        prior single-lock implementation. Phase 2+ callers pass real
        slot IDs so concurrent requests on different slots don't
        serialize. Invariant: a single async operation holds at most
        one slot lock simultaneously (lint-tested in Phase 2 tests).
        """
        lock = self._slot_locks.get(slot_id)
        if lock is None:
            lock = asyncio.Lock()
            self._slot_locks[slot_id] = lock
        return lock

    # ------------------------------------------------------------------
    # Pre-tokenized prompt assembly (managed server, eligible requests)
    # ------------------------------------------------------------------

    def _can_use_completion(self, request: InternalChatRequest) -> bool:
        """Check if this request can use the /completion endpoint.

        /completion with token arrays works for plain chat. Falls back
        to /v1/chat/completions for tool calling, JSON mode, or tool
        call results (which need special template handling).
        """
        if request.tools:
            return False
        if request.format == "json":
            return False
        if any(m.tool_call_id or m.tool_calls for m in request.messages):
            return False
        # Multimodal requests must use /v1/chat/completions.
        #
        # NOTICE:
        # The /completion fast path tokenizes the rendered chat template
        # and sends a flat int[] prompt to llama-server -- which never
        # invokes mtmd. The template's image-marker tokens (e.g.,
        # ``<__media_HASH__>``) become literal characters in the prompt,
        # and the model reads them back as text, producing "I see a
        # placeholder, not the image" hallucinations.
        # Source/context: observed with Qwen3-VL + paired mmproj; the
        # error log line "This feature is not supported by multimodal"
        # came from the post-stream slot_save side-channel, but the
        # primary failure was this fast-path bypass of mtmd.
        # The /v1/chat/completions path passes the full multimodal
        # content array (image_url parts) through, and llama-server's
        # mtmd middleware substitutes image embeddings at the marker
        # positions before tokenization.
        if any(getattr(m, "images", None) for m in request.messages):
            return False
        family = self.reasoning_family(request.model)
        if request.think and family and (
            family.startswith("qwen") or family.startswith("deepseek")
        ):
            # Native reasoning separation is more reliable on
            # /v1/chat/completions. The /completion fast path can surface the
            # model's planning prose directly in visible content for these
            # families, which breaks the user-facing thinking channel.
            return False
        return True

    async def _build_token_prompt(self, request: InternalChatRequest) -> list[int] | None:
        """Assemble a pre-tokenized prompt from cached segments.

        Uses llama-server's /apply-template to render the chat template,
        then tokenizes with caching. Returns the full token array, or None
        to fall back to /v1/chat/completions.
        """
        if self._manager is None:
            return None

        cache = self._get_cache()
        if cache is None:
            return None

        model_id = self._manager.model_id
        if not model_id:
            return None

        if not self._can_use_completion(request):
            return None

        try:
            return await self._build_token_prompt_inner(request, cache, model_id)
        except Exception as exc:
            log.warning("token_prompt_build_failed", error=str(exc)[:200])
            return None  # fall back to /v1/chat/completions

    async def _build_token_prompt_inner(
        self, request: InternalChatRequest, cache, model_id: str,
    ) -> list[int] | None:
        """Inner implementation — may raise on cache errors."""
        # Build message dicts for apply_template
        msg_dicts = self._request_messages_for_template(request)

        # Render full prompt via server's chat template
        rendered = await self.apply_template(
            msg_dicts,
            chat_template_kwargs=self._chat_template_kwargs(request),
        )
        if not rendered:
            return None

        # Check full prompt cache
        full_key = f"full:{hashlib.sha256(rendered.encode()).hexdigest()[:24]}"
        cached_full = await cache.get_tokens(model_id, full_key)
        if cached_full is not None:
            log.debug("token_cache_full_hit", model=model_id, tokens=len(cached_full))
            return cached_full

        # Coalesce: if another request is already tokenizing this same
        # rendered prompt, wait on its future instead of duplicating
        # the /tokenize round-trip and the cache write.
        inflight = self._tokenize_inflight.get(full_key)
        if inflight is not None:
            try:
                return await inflight
            except Exception as exc:
                # Whoever was tokenizing failed — fall through and try ourselves.
                log.debug("tokenize_inflight_leader_failed", model=model_id, error=str(exc))

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[int] | None] = loop.create_future()
        self._tokenize_inflight[full_key] = future
        try:
            tokens = await self.tokenize(rendered)
            if not tokens:
                future.set_result(None)
                return None

            await cache.store_tokens(model_id, full_key, tokens)
            log.debug("token_cache_tokenized", model=model_id, tokens=len(tokens))
            future.set_result(tokens)
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._tokenize_inflight.pop(full_key, None)
        return tokens

    async def prewarm_context(
        self,
        messages: list[dict],
        *,
        slot_id: int | None = None,
    ) -> int | None:
        """Pre-warm llama-server's KV cache with conversation context.

        Called while the user is typing to prime the prefix cache, from
        ``prepare_stable_checkpoint`` after a response to cache the
        post-response state for the next turn, and by the resume
        ladder's replay rung (``kv_resume.py``) to recompute a stored
        session prefix in a free window.

        Sends the system prompt + conversation (and, in the checkpoint
        case, the assistant response we just generated) to /completion
        with n_predict=0. This forces prefill without generating any
        tokens. The next real request will match the cached prefix and
        skip prefill for all the pre-warmed tokens.

        ``slot_id`` (optional): when provided, pins the prewarm to that
        physical slot via the request's ``id_slot`` field. Phase 2
        multi-slot uses this so checkpoint prewarms target a non-chat
        slot. ``None`` (default) lets llama-server's auto-routing pick
        — typically slot 0 in single-slot mode, or whichever slot has
        the best LCP match in multi-slot.

        Returns the slot id the prewarm landed on (``-1`` when the
        server didn't echo ``id_slot``), or ``None`` on failure. NOTE
        slot 0 is a valid success — callers must test ``is None``, not
        truthiness.
        """
        if self._manager is None:
            return None
        from augmentum.models.llama_server_manager import ProcessState
        if self._manager.state != ProcessState.READY:
            return None

        try:
            # Render via apply-template with thinking disabled and no
            # generation-prompt marker. Two reasons:
            # 1. Qwen3.6 / DeepSeek default chat templates ship with
            #    ``enable_thinking=true``. When the message list ends in
            #    an assistant turn (the checkpoint case — we save the
            #    post-response state), llama-server raises
            #    400 "Assistant response prefill is incompatible with
            #    enable_thinking" because the template can't combine an
            #    assistant prefill with a thinking-mode continuation.
            #    The prewarm itself never generates thinking tokens
            #    (n_predict=0), so disabling here is safe.
            # 2. ``add_generation_prompt=False`` tells the template
            #    "format what's there, don't append a 'now-generate'
            #    suffix." The next real chat turn rebuilds the template
            #    with its own continuation context, so we don't want a
            #    stale next-turn marker baked into the cached prefix.
            render_kwargs = {
                "enable_thinking": False,
                "add_generation_prompt": False,
            }
            tokens: list[int] | None
            if messages and messages[-1].get("role") == "assistant":
                # Two problems when the prewarm ends in an assistant
                # message (the stable-checkpoint case), both solved by the
                # dual-render boundary cut below:
                #
                # 1. History-framing: chat templates render the FINAL
                #    assistant message in continuation/prefill framing —
                #    Qwen3.5 injects an empty ``<think>`` block and omits
                #    the closing ``<|im_end|>`` — while the next real turn
                #    renders that same message in HISTORY framing (plain,
                #    closed). A checkpoint prewarmed in continuation
                #    framing diverges at the assistant header and the
                #    whole cached tail is unreusable. Appending a dummy
                #    user message pushes the assistant into history
                #    framing.
                #
                # 2. Boundary precision: hybrid-attention models (Qwen3.5+)
                #    can't rewind past a context checkpoint, and llama-
                #    server drops its checkpoint at END of prefill. If the
                #    prewarm prefills even a few tokens past what the next
                #    turn can match (dummy-content junk), the checkpoint
                #    lands past the divergence point, gets invalidated,
                #    and the next turn re-evaluates everything (verified
                #    live: f_keep 0.97+ yet full re-prefill).
                #
                # So: render with TWO different dummy user contents. The
                # token-level common prefix of the two renders is exactly
                # the maximal prefix ANY next-turn payload shares —
                # history + assistant in history framing + the user-turn
                # header, ending right where real user content would
                # diverge. Prefill exactly that, so the end-of-prefill
                # checkpoint sits AT the stable boundary.
                r_a, r_b = await asyncio.gather(
                    self.apply_template(
                        [*messages, {"role": "user", "content": ""}],
                        chat_template_kwargs=render_kwargs,
                    ),
                    self.apply_template(
                        [*messages, {"role": "user", "content": "⌘"}],
                        chat_template_kwargs=render_kwargs,
                    ),
                )
                if not r_a or not r_b:
                    return None
                t_a, t_b = await asyncio.gather(
                    self.tokenize(r_a), self.tokenize(r_b),
                )
                if not t_a or not t_b:
                    return None
                boundary = 0
                for x, y in zip(t_a, t_b, strict=False):
                    if x != y:
                        break
                    boundary += 1
                if boundary == 0:
                    return None
                tokens = t_a[:boundary]
            else:
                rendered = await self.apply_template(
                    messages, chat_template_kwargs=render_kwargs,
                )
                if not rendered:
                    return None
                tokens = await self.tokenize(rendered)
            if not tokens:
                return None

            # Defensive backstop: skip if the rendered prompt would
            # overflow the slot's context. Upstream callers (narrative
            # engine) budget for this, but a render-time delta can still
            # push us over by a few hundred tokens on long contexts.
            ctx_size = int(getattr(self._manager, "current_ctx_size", 0) or 0)
            if ctx_size and len(tokens) >= ctx_size:
                log.warning(
                    "prewarm_skipped_oversize",
                    tokens=len(tokens),
                    ctx_size=ctx_size,
                )
                return None

            # Send with n_predict=0 — prefill only, no generation. 300s
            # timeout: long prefills (30k+ tokens on CPU/iGPU) routinely
            # exceed the previous 30s ceiling.
            payload: dict = {
                "prompt": tokens,
                "n_predict": 0,
                "cache_prompt": True,
            }
            if slot_id is not None:
                # Pin the prewarm to the chosen slot. Upstream's task
                # handler routes via ``task.id_slot``: if the slot is
                # busy the task defers (queues) on that slot; if it's
                # idle it runs there. See the design doc Phase 0
                # verification section.
                payload["id_slot"] = slot_id
            resp = await self._client.post(
                f"{self._base_url}/completion",
                json=payload,
                headers=self._headers(),
                timeout=300.0,
            )
            if resp.status_code < 400:
                # llama-server's completion result echoes the serving
                # slot. Surface it so callers (the resume ladder) can
                # claim occupancy for an unpinned prewarm; -1 = server
                # didn't say (older builds / parse miss) but the warm
                # itself succeeded.
                served_slot = -1
                try:
                    served_slot = int(resp.json().get("id_slot", -1))
                except Exception:
                    served_slot = slot_id if slot_id is not None else -1
                log.debug("prewarm_success", tokens=len(tokens), slot=served_slot)
                return served_slot
            log.warning(
                "prewarm_rejected",
                status=resp.status_code,
                body=(resp.text or "")[:200],
                tokens=len(tokens),
            )
            return None
        except Exception as exc:
            log.warning("prewarm_failed", error=repr(exc))
            return None

    def _to_completion_payload(
        self, request: InternalChatRequest, tokens: list[int]
    ) -> dict:
        """Build a /completion payload from a pre-tokenized prompt.

        ``stream`` is deliberately omitted — each caller (``chat`` /
        ``_stream_completion``) sets it explicitly so a streaming caller
        can never leak ``stream=True`` into a non-streaming call.
        """
        payload: dict = {
            "prompt": tokens,
            "cache_prompt": True,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["n_predict"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            payload["seed"] = request.seed

        # llama.cpp-specific params
        if request.raw_options:
            for key in _LLAMACPP_PASSTHROUGH_PARAMS:
                if key in request.raw_options:
                    payload[key] = request.raw_options[key]

        return payload

    def _loaded_template_thinking(self) -> bool:
        """Whether the loaded model's GGUF chat-template consumes a thinking kwarg.

        Ground truth parsed from the embedded jinja at profile time
        (``ModelProfile.template_thinking``). Used as an authoritative fallback
        when the name/arch reasoning-family regex misses a renamed SFT/merged
        model. Safe by construction: reads the CURRENTLY loaded model's profile
        via the manager, and only ever adds an ``enable_thinking`` kwarg that a
        non-branching template ignores. Returns ``False`` when no profile is
        cached (cloud backends, pre-load) so callers keep their name-based path.
        """
        mgr = getattr(self, "_manager", None)
        prof = getattr(mgr, "_last_profile", None) if mgr else None
        return bool(getattr(prof, "template_thinking", False)) if prof else False

    def reasoning_family(self, model: str | None) -> str | None:
        """Arch-aware reasoning-family lookup for the loaded model.

        Name-only detection misses renamed SFT/merged models entirely
        (an ``alethia-9b`` GGUF carries no ``qwen`` substring), which
        routed Qwen3.5 finetunes to the symmetric parser and leaked the
        prefilled-``<think>`` interior into visible content. The GGUF
        ``general.architecture`` from the loaded profile is the model
        author's declared identity — pass it so arch wins over the name
        needle. Falls back to name-only when no profile is cached
        (cloud backends, pre-load), preserving prior behavior.
        """
        mgr = getattr(self, "_manager", None)
        prof = getattr(mgr, "_last_profile", None) if mgr else None
        arch = (getattr(prof, "architecture", "") or "") if prof else ""
        return detect_reasoning_family(model=model, arch=arch or None)

    def _chat_template_kwargs(self, request: InternalChatRequest) -> dict | None:
        """Build llama.cpp chat-template kwargs for reasoning-capable models.

        Forwarded to llama-server as ``chat_template_kwargs`` so the model's
        Jinja template (rendered via ``--jinja``) can branch on the per-turn
        thinking flag. Covers families whose template consumes an
        ``enable_thinking`` boolean: Qwen 3.x, DeepSeek (R1/V2/V3 + V3.2/V4),
        GLM-4.x, LG EXAONE 4.x, and NVIDIA Nemotron 3 Nano (incl. the
        nemotron_h_moe Omni variant). The UI's thinking button sets
        ``request.think`` for these families (see
        ``ui/scripts/settings.js::detectThinkingSupport``).

        When ``settings.engine_reasoning_budget`` is non-zero AND thinking
        is on for this turn, we also forward ``reasoning_budget`` (and
        ``grace_period`` when set). Templates that don't branch on these
        keys silently ignore them — adding them is therefore safe for
        every reasoning family, not just the ones that document the kwarg.
        """
        from augmentum.config import settings  # local import keeps cycles out

        family = self.reasoning_family(request.model)
        family_ok = bool(family) and (
            family.startswith("qwen")
            or family.startswith("deepseek")
            or family.startswith("glm")
            or family.startswith("exaone")
            or family.startswith("nemotron")
            or family.startswith("gemma4")
            or family.startswith("kimi")
            or family.startswith("mimo")
            or family == "chatglm"
        )
        # GGUF chat-template ground truth for the loaded model. Authoritative
        # for SFT/merged models whose display name or arch was renamed off the
        # upstream family — the name/arch regex misses them, but the template
        # actually consumes the thinking kwarg. When the template says so, ship
        # the kwarg even if no family matched. A template that DOESN'T branch on
        # the kwarg ignores it, so this only ever adds a working knob, never a
        # broken one. See model_profile_cache._extract_gguf_capabilities.
        tmpl_thinking = self._loaded_template_thinking()
        if not family_ok and not tmpl_thinking:
            return None
        # Qwen3-Coder (incl. Qwen3-Coder-Next) is non-thinking by design
        # per the official model cards. Skip the kwarg entirely so we don't
        # ship a knob the chat template ignores. Mirrors the UI gate in
        # ``ui/scripts/settings.js::detectThinkingSupport``.
        if family_ok and family.startswith("qwen"):
            normalized_name = "".join(
                ch for ch in (request.model or "").lower() if ch.isalnum()
            )
            if "coder" in normalized_name and "thinking" not in normalized_name:
                return None
        # Per-family kwarg name. Most families consume ``enable_thinking``
        # (Qwen / GLM / DeepSeek / EXAONE / Nemotron / Gemma 4 / MiMo);
        # Moonshot Kimi K2.6 uses bare ``thinking`` (different name, same
        # semantics). Sending the wrong name is harmless on tolerant
        # templates but a no-op on strict ones — get it right per
        # family so the toggle actually reaches the model.
        kwarg_name = "thinking" if (family or "").startswith("kimi") else "enable_thinking"
        kwargs: dict = {kwarg_name: bool(request.think)}
        # DeepSeek V3.2 / V4 (flash/pro) templates additionally consume a
        # ``reasoning_effort`` string ("high" / "max") next to
        # ``enable_thinking`` (Unsloth DS4 docs; llama.cpp #24162). Mirror
        # the cloud adapter's nested ``thinking:{reasoning_effort}`` here so
        # the UI's Off/High/Max picker works identically on local GGUFs.
        # Only the values the template documents pass through — the shared
        # mode hints ("low"/"medium") are dropped, "xhigh" (the OpenAI-enum
        # spelling of the top tier) maps to "max". An explicit "off" is a
        # thinking toggle, not an effort: force the think kwarg false as a
        # belt-and-braces mirror of the UI mapping.
        effort = str(getattr(request, "reasoning_effort", "") or "").strip().lower()
        if (family or "").startswith("deepseek"):
            if effort == "off":
                kwargs[kwarg_name] = False
            elif request.think and effort in ("high", "max", "xhigh"):
                kwargs["reasoning_effort"] = "max" if effort == "xhigh" else effort
        if request.think:
            budget = int(getattr(settings, "engine_reasoning_budget", 0) or 0)
            grace = int(getattr(settings, "engine_reasoning_grace_period", 0) or 0)
            if budget > 0:
                kwargs["reasoning_budget"] = budget
            if grace > 0:
                kwargs["grace_period"] = grace
            # Qwen 3.6 only documents preserve_thinking; we gate to that
            # family to match the UI's preserve popover (other families'
            # templates silently ignore the kwarg, but we'd rather not
            # ship a knob that pretends to work).
            if request.preserve_thinking:
                normalized = "".join(
                    ch for ch in (request.model or "").lower() if ch.isalnum()
                )
                if "qwen36" in normalized:
                    kwargs["preserve_thinking"] = True
        return kwargs

    def _apply_reasoning_request_options(
        self, payload: dict, request: InternalChatRequest,
    ) -> None:
        """Attach llama.cpp reasoning controls to an OpenAI chat payload."""
        chat_template_kwargs: dict = {}
        auto_kwargs = self._chat_template_kwargs(request)
        if auto_kwargs:
            chat_template_kwargs.update(auto_kwargs)
        if request.chat_template_kwargs:
            # Explicit per-request kwargs come from higher-level loops
            # such as coder native mode. Let them override the automatic
            # think-button mapping so a tool loop can force thinking off
            # even when the model family normally supports it.
            chat_template_kwargs.update(request.chat_template_kwargs)
        if request.continue_last_assistant:
            # Continue-the-trailing-assistant: tell the chat template to
            # format the message list as-is without appending a fresh
            # "now-generate" suffix. The model just keeps emitting tokens
            # from where the partial ended. Also force ``enable_thinking``
            # off — llama-server returns 400 "Assistant response prefill
            # is incompatible with enable_thinking" when both are set
            # (the template can't combine a trailing-assistant turn with
            # a fresh think block). Same constraint the prewarm path
            # documents in this file.
            chat_template_kwargs["add_generation_prompt"] = False
            chat_template_kwargs["enable_thinking"] = False
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        family = self.reasoning_family(request.model)
        if family and (family.startswith("qwen") or family.startswith("deepseek")):
            effective_think = bool(request.think)
            if "enable_thinking" in chat_template_kwargs:
                effective_think = bool(chat_template_kwargs["enable_thinking"])
            payload["reasoning_format"] = "deepseek" if effective_think else "none"

    # ------------------------------------------------------------------
    # Core chat
    # ------------------------------------------------------------------

    def _context_usage_payload(
        self, prompt_tokens: int, cache_n: int = 0,
    ) -> dict | None:
        """Build the context-usage payload for a terminal stream chunk.

        Frontends render a context-percentage bar from these two numbers.
        Returns ``None`` when either piece is unavailable so callers can
        attach the payload conditionally (and the bar gracefully stays
        empty for non-llama-server backends).

        ``prompt_tokens`` from llama-server is the count of tokens
        *freshly evaluated* this turn (i.e., not served from the slot's
        KV cache). For real context occupancy we want the cumulative
        prompt size, which is ``prompt_tokens + cache_n`` when timings
        carry ``cache_n``. Pre-fix the percent-ctx bar collapsed to 0%
        the moment cache reuse kicked in, even though the conversation
        was 90% of the way through the window.
        """
        if prompt_tokens <= 0 or self._manager is None:
            return None
        ctx_size = int(getattr(self._manager, "current_ctx_size", 0) or 0)
        if ctx_size <= 0:
            return None
        cumulative = int(prompt_tokens) + max(0, int(cache_n))
        return {
            "context_used": cumulative,
            "context_length": ctx_size,
            "prompt_tokens_evaluated": int(prompt_tokens),
            "prompt_tokens_cached": max(0, int(cache_n)),
        }

    def _log_performance(
        self, timings: dict, t_start: float, t_first_token: float | None,
    ) -> None:
        """Log inference performance metrics from llama-server timings.

        Stamps the request_id and the kv_tier from the proxy's
        ContextVars so this event self-contains the routing decision
        and its perf payoff for distribution analysis without a
        post-hoc log JOIN. Tier values are documented on
        ``augmentum.proxy.status_bus.kv_tier_var``.
        """
        from augmentum.proxy.status_bus import kv_tier_var, request_id_var

        prompt_n = timings.get("prompt_n", 0)
        predicted_n = timings.get("predicted_n", 0)
        prompt_ms = timings.get("prompt_ms", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        prompt_tps = round(prompt_n / (prompt_ms / 1000), 1) if prompt_ms > 0 else 0
        gen_tps = round(predicted_n / (predicted_ms / 1000), 1) if predicted_ms > 0 else 0
        total_s = round(time.monotonic() - t_start, 2)
        ttft_ms = round((t_first_token - t_start) * 1000) if t_first_token else 0

        model_id = self._manager.model_id if self._manager else "unknown"
        # MTP acceptance counters (cumulative for the subprocess
        # lifetime — see LlamaServerManager._mtp_last_log doc). When MTP
        # isn't active or hasn't logged a stats line yet the dict is
        # empty and we omit the fields entirely to keep the
        # ``engine_perf`` event tidy.
        mtp_log = (
            getattr(self._manager, "_mtp_last_log", {})
            if self._manager else {}
        )
        extras: dict[str, float | int] = {}
        if mtp_log:
            extras["mtp_accept_rate"] = mtp_log.get("rate", 0.0)
            extras["mtp_accepted"] = mtp_log.get("accepted", 0)
            extras["mtp_generated"] = mtp_log.get("generated", 0)
        log.info(
            "engine_perf",
            model=model_id,
            prompt_tokens=prompt_n,
            gen_tokens=predicted_n,
            prompt_tps=prompt_tps,
            gen_tps=gen_tps,
            ttft_ms=ttft_ms,
            total_s=total_s,
            request_id=request_id_var.get() or "",
            kv_tier=kv_tier_var.get() or "",
            **extras,
        )

    def _session_fingerprint(self, request: InternalChatRequest) -> str:
        """Return the slot-affinity key for this request, or empty.

        Used by ``_manage_slot`` to decide whether to save/restore KV state.
        Sources, in order:

        1. Stable narrative-checkpoint key (when the request is a checkpoint
           replay — the key already encodes the canonical message tail).
        2. ``request.kv_session_key`` set by the route layer when the caller
           provided a *trustworthy* session source: the in-app
           ``X-Augmentum-Session`` header or a coder workspace ID.
        3. Empty — meaning the caller is an external API client without our
           session header, and the slot manager should not try to save or
           restore against any guessed identifier. llama-server's per-slot
           token-prefix cache covers within-conversation reuse without it.

        Note: a previous fallback hashed the system message when (1)/(2) were
        absent. That collided across unrelated branches and regenerations
        that shared a system prompt, leaking KV state between conversations.
        Removed deliberately — see chat in commit history.
        """
        stable_messages = self._stable_restore_messages(request)
        if stable_messages is not None:
            checkpoint_key = self._stable_checkpoint_key(request, stable_messages)
            if checkpoint_key:
                return checkpoint_key
        if request.kv_session_key:
            return request.kv_session_key.strip()
        return ""

    @staticmethod
    def _slot_storage_name(session_id: str) -> str:
        """Map a logical session id to a filesystem-safe slot filename."""
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        return f"session_{digest}"

    def _slot_state_exists(self, session_id: str) -> bool:
        """Whether llama-server has a saved KV slot file for this session.

        Used to gate the user-facing ``"restoring"`` status chunk so we
        don't claim "restoring session" for a brand-new chat that has no
        prior state on disk — only the *active session id* changed, which
        is not the same as actually pulling cached KV back into the slot.
        Probes by filename prefix because llama-server appends its own
        extension (``.bin``) and that's an upstream-controlled detail.
        """
        if self._manager is None or not session_id:
            return False
        slot_dir = getattr(self._manager, "_slot_dir", "") or ""
        if not slot_dir:
            return False
        prefix = self._slot_storage_name(session_id)
        try:
            return any(name.startswith(prefix) for name in os.listdir(slot_dir))
        except OSError:
            return False

    @staticmethod
    def _system_prompt_hash(request: InternalChatRequest) -> str:
        system_parts = [m.content for m in request.messages if m.role == "system" and m.content]
        if not system_parts:
            return ""
        joined = "\n\n".join(system_parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _template_fingerprint(request: InternalChatRequest) -> str:
        system_parts = [
            {
                "role": m.role,
                "content": m.content or "",
                "images": len(m.images or []),
            }
            for m in request.messages
            if m.role == "system"
        ]
        if not system_parts:
            return ""
        payload = json.dumps(system_parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _prompt_fingerprint(request: InternalChatRequest) -> str:
        message_meta = [
            {
                "role": m.role,
                "content": m.content or "",
                "images": len(m.images or []),
                "tool_calls": len(m.tool_calls or []),
                "tool_call_id": bool(m.tool_call_id),
            }
            for m in request.messages
        ]
        if not message_meta:
            return ""
        payload = json.dumps(message_meta, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _clone_messages(messages: list[Message] | None) -> list[Message]:
        if not messages:
            return []
        return [
            Message(
                role=m.role,
                content=m.content,
                images=list(m.images) if m.images else None,
                tool_calls=list(m.tool_calls) if m.tool_calls else None,
                thinking=m.thinking,
                tool_call_id=m.tool_call_id,
            )
            for m in messages
        ]

    @staticmethod
    def _messages_fingerprint(messages: list[Message] | None) -> str:
        if not messages:
            return ""
        message_meta = [
            {
                "role": m.role,
                "content": m.content or "",
                "images": len(m.images or []),
                "tool_calls": len(m.tool_calls or []),
                "tool_call_id": bool(m.tool_call_id),
            }
            for m in messages
        ]
        payload = json.dumps(message_meta, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    # Number of trailing messages whose content drives the slot key. Five is
    # enough variety to avoid collisions on regeneration of the same prompt
    # (recent assistant content typically varies turn-to-turn) without being
    # so large that head trim drift propagates back into the digest.
    _TAIL_FINGERPRINT_K = 5

    @classmethod
    def _messages_tail_fingerprint(
        cls,
        messages: list[Message] | None,
        k: int | None = None,
    ) -> str:
        """Digest only the last ``k`` messages — invariant to head trim drift.

        The full ``_messages_fingerprint`` is sensitive to every drop at
        the head: when narrative-mode trim shifts by a single oldest
        message between turns, the digest changes completely and a slot
        saved on turn N can no longer be looked up on turn N+1.

        Tail-only digests sidestep this — the trailing messages don't
        shift between save and lookup (only the head does, when trim
        pressure varies), so the slot key stays stable across turns
        even under budget pressure.
        """
        if not messages:
            return ""
        n = int(k if k is not None else cls._TAIL_FINGERPRINT_K)
        tail = messages[-n:] if n > 0 else messages
        return cls._messages_fingerprint(tail)

    @classmethod
    def _stable_restore_messages(cls, request: InternalChatRequest) -> list[Message] | None:
        stable = cls._clone_messages(request.kv_stable_messages)
        if not stable:
            return None
        if stable[-1].role == "user":
            stable = stable[:-1]
        return stable

    @classmethod
    def _stable_checkpoint_key(
        cls,
        request: InternalChatRequest,
        messages: list[Message] | None,
    ) -> str:
        digest = cls._messages_tail_fingerprint(messages) or "root"
        owner = (request.kv_session_key or "").strip()
        if owner:
            return f"{owner}::stable::{digest}"
        return f"stable::{digest}"

    @classmethod
    def _checkpoint_request_from_messages(
        cls,
        request: InternalChatRequest,
        messages: list[Message],
        checkpoint_key: str,
    ) -> InternalChatRequest:
        """Build a checkpoint-save request from an existing one.

        Uses ``dataclass_replace`` so every field on the source
        request flows through automatically — the explicit-list
        pattern that originally lived here would silently drop any
        field added to ``InternalChatRequest`` after this code was
        written. See the ``apply_preset`` fix (commit 731a96d) for
        the bug class this guards against.
        """
        return dataclass_replace(
            request,
            messages=cls._clone_messages(messages),
            stream=False,
            kv_session_key=checkpoint_key,
            kv_stable_messages=cls._clone_messages(messages),
        )

    def _kv_manifest(self):
        if self._manager is None:
            return None
        return getattr(self._manager, "_session_manifest", None)

    def _current_model_key(self) -> str:
        if self._manager is None:
            return ""
        runtime = self._manager.current_runtime_signature()
        return runtime.get("model_key", "")

    def _restore_skip_reason(self, record: dict) -> str | None:
        if self._manager is None:
            return None
        runtime = self._manager.current_runtime_signature()
        return kv_restore_skip_reason(record, runtime)

    async def _record_manifest_save(
        self,
        session_id: str,
        request: InternalChatRequest | None = None,
    ) -> None:
        if self._manager is None or not self._manager._slot_dir:
            return
        manifest = self._kv_manifest()
        if manifest is None:
            return

        model_key = self._current_model_key()
        if not model_key:
            return

        existing = await manifest.get_session_async(model_key, session_id) or {}
        mode = (request.kv_mode or "").strip().lower() if request else str(existing.get("mode", "")).strip().lower()
        ttl_days = self._manager.kv_ttl_days_for_mode(mode)
        runtime = self._manager.current_runtime_signature()

        await manifest.record_save_async(
            model_key=model_key,
            session_key=session_id,
            mode=mode,
            slot_dir=self._manager._slot_dir,
            slot_filename=self._slot_storage_name(session_id),
            model_id=runtime.get("model_id", ""),
            model_path=runtime.get("model_path", ""),
            model_mtime=float(runtime.get("model_mtime", 0.0) or 0.0),
            ctx_size=int(runtime.get("ctx_size", 0) or 0),
            kv_cache_type=runtime.get("kv_cache_type", "") or "",
            template_fingerprint=self._template_fingerprint(request) if request else str(existing.get("template_fingerprint", "")),
            system_prompt_hash=self._system_prompt_hash(request) if request else str(existing.get("system_prompt_hash", "")),
            prompt_fingerprint=self._prompt_fingerprint(request) if request else str(existing.get("prompt_fingerprint", "")),
            prompt_message_count=len(request.messages) if request else int(existing.get("prompt_message_count", 0) or 0),
            ttl_days=ttl_days,
            pinned=self._manager.session_is_pinned(session_id, mode),
            flash_attn=bool(runtime.get("flash_attn", False)),
            gpu_layers=int(runtime.get("gpu_layers", 0) or 0),
            gpu_layers_mode=str(runtime.get("gpu_layers_mode", "") or ""),
            batch_size=int(runtime.get("batch_size", 0) or 0),
            draft_model=str(runtime.get("draft_model", "") or ""),
            draft_max=int(runtime.get("draft_max", 0) or 0),
            n_embed=int(runtime.get("n_embed", 0) or 0),
            n_layers_total=int(runtime.get("n_layers_total", 0) or 0),
            n_heads_kv=int(runtime.get("n_heads_kv", 0) or 0),
        )

    async def _record_manifest_touch(
        self,
        session_id: str,
        request: InternalChatRequest | None = None,
        *,
        restored: bool | None = None,
    ) -> None:
        if self._manager is None:
            return
        manifest = self._kv_manifest()
        if manifest is None:
            return
        model_key = self._current_model_key()
        if not model_key:
            return
        existing = await manifest.get_session_async(model_key, session_id)
        if not existing:
            return
        mode = (request.kv_mode or "").strip().lower() if request else str(existing.get("mode", "")).strip().lower()
        await manifest.touch_session_async(
            model_key=model_key,
            session_key=session_id,
            ttl_days=self._manager.kv_ttl_days_for_mode(mode),
            mode=mode,
            pinned=self._manager.session_is_pinned(session_id, mode),
            restored=restored,
        )

    # ------------------------------------------------------------------
    # Replay-source capture (KV resume ladder — rung 2's durable input)
    # ------------------------------------------------------------------

    # Ceiling on one serialized replay source. A 90k-token conversation
    # is ~400KB of JSON; anything past this is runaway content. Policy
    # is skip-not-truncate: a cut prefix would replay tokens the next
    # real request can't match, poisoning the slot it lands in.
    _REPLAY_SOURCE_MAX_BYTES = 4_000_000

    @staticmethod
    def _replay_source_payload(request: InternalChatRequest) -> list[dict] | None:
        """Serialize this request's replayable prefix, or None.

        Prefers the declared stable prefix (``kv_stable_messages``) so a
        mode's volatile per-turn tail never enters the replay. Text-only
        by design: images can't be reconstructed by a text prefill, and
        tool-call turns render through template branches the prewarm
        path doesn't reproduce — those sessions stay cold rather than
        warm wrong.
        """
        source = request.kv_stable_messages or request.messages
        if not source:
            return None
        out: list[dict] = []
        for m in source:
            role = getattr(m, "role", "")
            content = getattr(m, "content", None)
            if role not in ("system", "user", "assistant"):
                return None
            if getattr(m, "tool_calls", None) or getattr(m, "tool_call_id", None):
                return None
            if getattr(m, "images", None):
                return None
            if not isinstance(content, str):
                return None
            out.append({"role": role, "content": content})
        return out

    def _schedule_replay_capture(
        self,
        request: InternalChatRequest,
        *,
        session_key_override: str = "",
    ) -> None:
        """Fire-and-forget persistence of this request's replay source.

        Called from the chat entry points after ``_manage_slot`` and
        from ``prepare_stable_checkpoint`` (which overrides the key back
        to the BASE session key — checkpoint requests carry a per-turn
        ``::stable::<digest>`` key that an on-open resume could never
        derive). Never blocks or errors the request path.
        """
        if getattr(request, "_augmentum_speculative", False):
            # Rule 4 of kv_speculate: drafts never touch disk. The
            # speculative request must not overwrite the session's real
            # replay row with unsent text.
            return
        session_key = (session_key_override or request.kv_session_key or "").strip()
        if not session_key:
            return
        from augmentum.config import settings as _cfg
        if not getattr(_cfg, "engine_kv_replay_enabled", True):
            return
        if self._kv_manifest() is None:
            return
        task = asyncio.create_task(
            self._record_replay_source(session_key, request)
        )
        # Surface capture failures instead of losing them to the void.
        def _log_capture_failure(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                log.warning("kv_replay_capture_failed", error=repr(exc))
        task.add_done_callback(_log_capture_failure)

    async def _record_replay_source(
        self, session_key: str, request: InternalChatRequest,
    ) -> None:
        manifest = self._kv_manifest()
        if manifest is None or self._manager is None:
            return
        payload = self._replay_source_payload(request)
        if payload is None:
            log.debug("kv_replay_capture_skipped", session=session_key,
                      reason="non_replayable_content")
            return
        messages_json = json.dumps(payload, ensure_ascii=False)
        if len(messages_json) > self._REPLAY_SOURCE_MAX_BYTES:
            log.info(
                "kv_replay_capture_skipped",
                session=session_key,
                reason="oversize",
                bytes=len(messages_json),
            )
            return
        mode = (request.kv_mode or "").strip().lower()
        fingerprint = hashlib.sha256(
            messages_json.encode("utf-8")
        ).hexdigest()[:24]
        # Sampling snapshot rides along so the speculation rung can
        # fingerprint-match the *next* request's completion-shaping
        # fields against this turn's (the best available predictor).
        from augmentum.models.kv_speculate import sampling_snapshot
        sampling_json = json.dumps(
            sampling_snapshot(request), sort_keys=True, ensure_ascii=False,
        )
        await manifest.record_replay_source_async(
            session_key=session_key,
            mode=mode,
            messages_json=messages_json,
            fingerprint=fingerprint,
            message_count=len(payload),
            ttl_days=self._manager.kv_ttl_days_for_mode(mode),
            sampling_json=sampling_json,
        )
        from augmentum.config import settings as _cfg
        max_rows = int(getattr(_cfg, "engine_kv_replay_max_rows", 64) or 0)
        expired, evicted = await manifest.prune_replay_sources_async(
            max_rows=max_rows,
        )
        if expired or evicted:
            log.info(
                "kv_replay_sources_pruned",
                expired=expired,
                evicted=evicted,
                cap=max_rows,
            )

    @property
    def resume_ladder(self):
        """Lazily-built :class:`~augmentum.models.kv_resume.KVResumeLadder`."""
        ladder = getattr(self, "_resume_ladder", None)
        if ladder is None:
            from augmentum.models.kv_resume import KVResumeLadder
            ladder = KVResumeLadder(self)
            self._resume_ladder = ladder
        return ladder

    @property
    def turn_speculator(self):
        """Lazily-built :class:`~augmentum.models.kv_speculate.TurnSpeculator`.

        Built on first use of the speculate endpoint; until then the
        ``_speculator`` attr is absent and the per-request hooks in
        ``chat``/``chat_stream`` cost one ``getattr``.
        """
        spec = getattr(self, "_speculator", None)
        if spec is None:
            from augmentum.models.kv_speculate import TurnSpeculator
            spec = TurnSpeculator(self)
            self._speculator = spec
        return spec

    @staticmethod
    def _system_content_to_text(content: str | list[dict] | None) -> str:
        """Flatten structured system content into text for strict templates."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)

    @classmethod
    def _late_system_context_carrier(cls, content: str | list[dict] | None) -> dict | None:
        """Convert a late system block into a trailing narrative context carrier."""
        text = cls._system_content_to_text(content)
        if not text:
            return None
        return {
            "role": "user",
            "content": (
                "[Augmentum narrative context - not user dialogue]\n"
                "Treat the following block as authoritative background for the next reply. "
                "Do not answer it separately.\n\n"
                f"{text}"
            ),
        }

    @classmethod
    def _normalize_system_messages(cls, messages: list[dict]) -> list[dict]:
        """Collapse multiple system messages into one leading system block.

        Two distinct template hazards motivate this:
          * Qwen variants reject system messages that appear after the
            conversation has started ("System message must be at the
            beginning").
          * Gemma's chat template (and several other strict-alternation
            templates) reject *consecutive* system messages even when both
            are leading — `[sys, sys, user]` 500s with "Conversation roles
            must alternate user/assistant/user/assistant/...". The
            knowledge-pack and tool-context injectors in the passthrough
            handler can produce this shape when they prepend a second
            system message instead of merging into the existing one.

        Leading systems coalesce into one leading block. Late systems do
        NOT merge to the front: relocating a message from the payload
        tail to position 0 rewrites the token stream's head, which
        invalidates the entire KV prefix (slot LCP, RAM checkpoint
        restore) on every turn — the exact regression this merge used to
        cause with the ``<current_time>`` block (contract=violated,
        cold_no_checkpoint, 2026-07-09). Instead each late system is
        converted IN PLACE into a user-role context carrier
        (``_late_system_context_carrier``), which every strict template
        accepts and which keeps its tokens at the position they held.
        """
        if not messages:
            return messages

        seen_non_system = False
        late_system_count = 0  # systems after a non-system msg
        leading_system_count = 0  # consecutive systems at the start
        for msg in messages:
            if msg.get("role") == "system":
                if seen_non_system:
                    late_system_count += 1
                else:
                    leading_system_count += 1
            else:
                seen_non_system = True

        # No-op cases: zero systems, or exactly one leading system + no
        # late systems. Anything else requires reshaping.
        if late_system_count == 0 and leading_system_count <= 1:
            return messages

        merged_parts: list[str] = []
        dropped_non_text = 0
        normalized: list[dict] = []
        seen_non_system = False
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                if seen_non_system:
                    # Late system → in-place user-role carrier.
                    carrier = cls._late_system_context_carrier(content)
                    if carrier is not None:
                        normalized.append(carrier)
                    elif content not in (None, ""):
                        dropped_non_text += 1
                    continue
                text = cls._system_content_to_text(content)
                if text:
                    merged_parts.append(text)
                elif content is not None and content != "":
                    # System message had content but text extraction
                    # yielded empty — most likely a list-typed payload
                    # whose only parts were images or unknown types.
                    # Silently dropping would lose model-relevant
                    # instructions and produce confusing "model
                    # ignores my system prompt" reports. Track + log
                    # so the operator knows their system content was
                    # reshaped. Real fix (image-bearing system messages
                    # → vision-aware merge) is bigger surgery; this
                    # makes the loss visible until then.
                    dropped_non_text += 1
                continue
            seen_non_system = True
            normalized.append(msg)

        if merged_parts:
            normalized.insert(0, {"role": "system", "content": "\n\n".join(merged_parts)})

        if dropped_non_text:
            log.warning(
                "llamacpp_system_message_non_text_dropped",
                count=dropped_non_text,
                note=(
                    "System message(s) with non-text content (likely "
                    "images or unknown structured parts) were dropped "
                    "during normalization. Vision-bearing system "
                    "messages aren't currently handled — convert to a "
                    "user-role message if you need the model to see them."
                ),
            )
        log.info(
            "llamacpp_system_messages_normalized",
            merged=leading_system_count + late_system_count,
            late=late_system_count,
            leading=leading_system_count,
        )
        return normalized

    @classmethod
    def _merge_consecutive_same_role(cls, messages: list[dict]) -> list[dict]:
        """Collapse adjacent same-role messages.

        Strict chat templates (Gemma, Mistral, Llama 3.x) raise "Conversation
        roles must alternate user/assistant/user/assistant/..." when two
        same-role messages land back-to-back. The text-tier tool path
        produces this shape legitimately: the tool result for the model's
        prior turn gets persisted as a user-role message, and the user's
        next prompt then lands as another user-role message — turn 2's
        history is ``[system, user, assistant, user(tool_result), user]``.

        We coalesce by concatenating ``content`` with a blank-line
        separator. Messages carrying structured fields (``tool_calls``,
        ``tool_call_id``, list-typed content for vision/multimodal) are
        treated as non-mergeable boundaries so we don't flatten an
        assistant tool-call into surrounding chat text.
        """
        if len(messages) < 2:
            return messages

        def _structured(msg: dict) -> bool:
            if msg.get("tool_calls") or msg.get("tool_call_id"):
                return True
            content = msg.get("content")
            # list-typed content = multimodal parts; never merge.
            return isinstance(content, list)

        merged: list[dict] = []
        coalesced = 0
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                merged.append(msg)
                continue
            if (
                merged
                and merged[-1].get("role") == role
                and not _structured(msg)
                and not _structured(merged[-1])
            ):
                prev = merged[-1]
                prev_text = prev.get("content") or ""
                next_text = msg.get("content") or ""
                if not isinstance(prev_text, str):
                    prev_text = str(prev_text)
                if not isinstance(next_text, str):
                    next_text = str(next_text)
                joined = (
                    f"{prev_text}\n\n{next_text}"
                    if prev_text and next_text
                    else (prev_text or next_text)
                )
                prev["content"] = joined
                coalesced += 1
                continue
            merged.append(msg)

        if coalesced:
            log.info(
                "llamacpp_consecutive_same_role_merged",
                count=coalesced,
                # Sequence reconstructed from the merged result so
                # operators can see the shape that survived.
                final_roles=[m.get("role") for m in merged[:16]],
            )
        return merged

    @classmethod
    def _ensure_user_first_after_system(cls, messages: list[dict]) -> list[dict]:
        """Insert a synthetic user turn when the first non-system message is assistant.

        Strict chat templates (Llama 3.x family, several Mistral variants)
        require user/assistant alternation starting with user, and 500 with
        "Conversation roles must alternate user/assistant/user/assistant/..."
        when the first turn is from the assistant. Narrative mode produces
        this shape legitimately when a session opens with the character
        setting the scene before the user has typed.
        """
        first_non_system = next(
            (i for i, m in enumerate(messages) if m.get("role") != "system"),
            -1,
        )
        if first_non_system == -1:
            return messages
        if messages[first_non_system].get("role") != "assistant":
            return messages
        carrier = {
            "role": "user",
            "content": (
                "[Augmentum narrative context - scene opens]\n"
                "Treat the next message as the scene's opening beat. "
                "Continue the narrative from there."
            ),
        }
        log.info("llamacpp_assistant_first_user_carrier_inserted")
        return messages[:first_non_system] + [carrier] + messages[first_non_system:]

    @classmethod
    def _rewrite_late_system_messages_for_checkpoint(
        cls,
        messages: list[dict],
    ) -> list[dict]:
        """Rewrite late system messages into tail context carriers.

        Checkpoint-aware narrative requests rely on a stable prefix. Turning
        dynamic lore/archive/state blocks into trailing carriers preserves that
        prefix while still satisfying strict llama.cpp chat templates.
        """
        if not messages:
            return messages

        seen_non_system = False
        converted = 0
        rewritten: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system" and seen_non_system:
                carrier = cls._late_system_context_carrier(msg.get("content"))
                if carrier is not None:
                    rewritten.append(carrier)
                converted += 1
                continue
            if msg.get("role") != "system":
                seen_non_system = True
            rewritten.append(msg)

        if converted:
            log.info("llamacpp_late_system_context_rewritten", converted=converted)
        return cls._merge_consecutive_same_role(
            cls._ensure_user_first_after_system(rewritten),
        )

    _CURRENT_TIME_RE = re.compile(r"<current_time>.*?</current_time>\n*", re.DOTALL)

    @classmethod
    def _relocate_leading_datetime(cls, messages: list[dict]) -> list[dict]:
        """Move a ``<current_time>`` block out of the leading system message
        to the payload tail.

        Several composers (agentic planner, tool chain, companion,
        custom flows) still prepend the minute-resolution datetime block
        to the top of their system prompt. At position ~0 it rewrites
        the head of the token stream every minute, invalidating the
        whole KV prefix on every turn — the class fixed for coder mode
        ("datetime LAST in carrier") applied here at the backend choke
        point so every current and future call site is covered. The
        relocated block lands as a trailing system message, which
        ``_normalize_system_messages`` then converts into a user-role
        carrier for strict templates. No-op when the leading system has
        no block (modes that already place it late, e.g. coder).
        """
        if not messages or messages[0].get("role") != "system":
            return messages
        content = messages[0].get("content")
        if not isinstance(content, str) or "<current_time>" not in content:
            return messages
        blocks = cls._CURRENT_TIME_RE.findall(content)
        if not blocks:
            return messages
        stripped = cls._CURRENT_TIME_RE.sub("", content).strip()
        out = list(messages)
        if stripped:
            out[0] = {**messages[0], "content": stripped}
        else:
            out = out[1:]
        out.append({"role": "system", "content": "\n".join(b.strip() for b in blocks)})
        log.info("llamacpp_datetime_relocated_to_tail", blocks=len(blocks))
        return out

    @classmethod
    def _request_messages_for_template(cls, request: InternalChatRequest) -> list[dict]:
        """Build request messages in the safest shape for llama.cpp templates."""
        messages = []
        for m in request.messages:
            content: str | list[dict] = m.content
            if m.images:
                parts: list[dict] = []
                if m.content:
                    parts.append({"type": "text", "text": m.content})
                for img in m.images:
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                content = parts
            msg_dict: dict = {"role": m.role, "content": content}
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            messages.append(msg_dict)

        messages = cls._relocate_leading_datetime(messages)

        if request.kv_stable_messages:
            return cls._rewrite_late_system_messages_for_checkpoint(messages)
        return cls._merge_consecutive_same_role(
            cls._ensure_user_first_after_system(
                cls._normalize_system_messages(messages),
            ),
        )

    @staticmethod
    def _should_defer_session_save(request: InternalChatRequest) -> bool:
        """Checkpoint-aware narrative requests save via prepare_stable_checkpoint."""
        return bool(request.kv_stable_messages)

    def _kv_reuse_trackable(self) -> bool:
        """Audit only when a live slot manager is present.

        Preserves the pre-extraction guard exactly: a remote llama.cpp
        backend (no ``server_manager``) has no slot/tier telemetry to judge,
        so it skips the audit just as it did when the guard read
        ``self._manager is None`` inline."""
        return self._manager is not None


    async def _manage_slot(self, request: InternalChatRequest) -> None:
        """Save/restore KV slot state for session continuity.

        Two paths, gated by ``engine_multislot_enabled``:

        Single-slot (flag off, current production):
          Slot 0 is the only slot. If the session changed since the
          last request, save slot 0's current KV under the prior
          session, then attempt to restore the new session's checkpoint
          into slot 0. Always claim slot 0 for the new session
          afterwards so post-stream save targets it.

        Multi-slot (flag on):
          Engine has N slots and routes requests via prefix-LCP/LRU.
          We don't pre-claim — the response's id_slot tells us which
          slot the engine actually picked. _manage_slot's job becomes
          narrower: ensure a disk-backed cold session is warmed into
          SOME slot before the engine runs prefill on it. If the
          session is already in our occupancy map (any slot), no-op
          — engine routes to it via prefix match. If not in
          occupancy AND we have a disk checkpoint, restore into a
          picked target slot. The actual served slot may differ from
          our pick (engine may route based on its own LCP); the
          response observation reconciles via _claim_slot.
        """
        # Bind a sentinel up front so the three early-return paths below
        # (no manager, not ready, opaque request) don't leak whatever
        # value the calling task's parent context left in `kv_tier_var`.
        # The hot/cold_with_checkpoint/cold_no_checkpoint paths below
        # each rebind to the real tier.
        from augmentum.proxy.status_bus import bind_kv_tier
        bind_kv_tier("unmanaged")

        if self._manager is None:
            return
        from augmentum.models.llama_server_manager import ProcessState
        if self._manager.state != ProcessState.READY:
            return

        # Stable-prefix contract measurement — every user-facing engine
        # request with a session key, all modes, both stream paths.
        try:
            self.track_prefix_stability(request)
        except Exception:
            log.warning("kv_prefix_stability_failed", exc_info=True)

        # Pick up restart-warmed session from the manager (single-slot
        # path). The manager's _warm_top_session loaded slot 0 with the
        # MRU compatible session before the first request arrived;
        # promoting it to occupancy here means a matching request skips
        # the redundant restore round-trip. Multi-slot path warms its
        # own slots via observation; this path is harmless then because
        # the response will overwrite occupancy on first hit.
        if not self._get_session_for_slot(0):
            warm = getattr(self._manager, "_warm_session_key", "") or ""
            if warm:
                self._claim_slot(0, warm)
                self._manager._warm_session_key = ""

        session = self._session_fingerprint(request)
        if not session:
            return  # opaque request, no slot affinity

        # Per-phase timing for kv_tier_decided. Lets us see in production
        # whether save-displaced, restore, or manifest/disk lookups
        # dominate the slot-management latency on cross-session switches.
        # Each phase records its elapsed_ms; absent keys mean the phase
        # didn't run for this request (e.g. no displaced session to save).
        phases: dict[str, float] = {}

        def _ms_since(start: float) -> float:
            return round((time.monotonic() - start) * 1000.0, 2)

        if self._multislot_enabled():
            # Multi-slot occupancy-driven routing.
            from augmentum.proxy.status_bus import bind_kv_tier, request_id_var

            existing_slot = self._get_slot_for_session(session)
            if existing_slot is not None:
                # Hot path: engine has this session's prefix in some
                # live slot. Engine's prefix matcher will route to it.
                # No restore needed; no pre-claim (response will
                # confirm via id_slot).
                bind_kv_tier("hot")
                log.info(
                    "kv_tier_decided",
                    session=session,
                    tier="hot",
                    slot=existing_slot,
                    multislot=True,
                    request_id=request_id_var.get() or "",
                    phases=phases,
                )
                return
            # Cold path: not tracked. If we have a disk checkpoint,
            # restore into a target slot before sending. Engine's
            # auto-routing should then pick that slot via LCP match.
            # The actual served slot is authoritative — set on response
            # observation, not here.
            _t_state = time.monotonic()
            has_checkpoint = self._slot_state_exists(session)
            phases["slot_state_check_ms"] = _ms_since(_t_state)
            if has_checkpoint:
                target = self._pick_restore_target_slot()
                # If target slot currently holds a different session,
                # save it first (don't lose KV that's only in --cache-ram
                # if our disk checkpoint for it has drifted).
                displaced = self._get_session_for_slot(target)
                if displaced and displaced != session:
                    _t_save = time.monotonic()
                    await self.save_session_state(displaced, slot_id=target)
                    phases["save_displaced_ms"] = _ms_since(_t_save)
                _t_restore = time.monotonic()
                restored = await self.restore_session_state(
                    session, slot_id=target, request=request,
                )
                phases["restore_ms"] = _ms_since(_t_restore)
                bind_kv_tier("cold_with_checkpoint")
                log.info(
                    "kv_tier_decided",
                    session=session,
                    tier="cold_with_checkpoint",
                    slot=target,
                    restored=bool(restored),
                    multislot=True,
                    request_id=request_id_var.get() or "",
                    phases=phases,
                )
                if restored:
                    log.info(
                        "session_kv_restored",
                        session=session, slot=target,
                    )
            else:
                # No occupancy record and no disk checkpoint — but the
                # resume ladder may have replayed this session's prefix
                # into a slot already (occupancy isn't tracked for
                # unpinned replays; llama-server's LCP router finds the
                # tokens regardless). Label it so acceptance runs can
                # tell replay-warmed from genuinely cold. One-shot tag.
                warmed_keys = getattr(self._manager, "_replay_warmed_keys", None)
                base_key = (request.kv_session_key or "").strip()
                replay_warmed = bool(
                    warmed_keys and base_key and base_key in warmed_keys
                )
                if replay_warmed:
                    warmed_keys.discard(base_key)
                tier = "cold_replay_warmed" if replay_warmed else "cold_no_checkpoint"
                bind_kv_tier(tier)
                log.info(
                    "kv_tier_decided",
                    session=session,
                    tier=tier,
                    slot=None,
                    multislot=True,
                    request_id=request_id_var.get() or "",
                    phases=phases,
                )
            return

        # Single-slot path: keep existing behavior unchanged.
        from augmentum.proxy.status_bus import bind_kv_tier, request_id_var

        current_session_in_slot_0 = self._get_session_for_slot(0)
        if session == current_session_in_slot_0:
            bind_kv_tier("hot")
            log.info(
                "kv_tier_decided",
                session=session,
                tier="hot",
                slot=0,
                multislot=False,
                request_id=request_id_var.get() or "",
                phases=phases,
            )
            return  # same session — nothing to do

        # Save current session's KV state before switching
        if current_session_in_slot_0:
            _t_save = time.monotonic()
            await self.save_session_state(current_session_in_slot_0, slot_id=0)
            phases["save_displaced_ms"] = _ms_since(_t_save)

        # Try to restore the new session's KV state
        _t_restore = time.monotonic()
        restored = await self.restore_session_state(session, slot_id=0, request=request)
        phases["restore_ms"] = _ms_since(_t_restore)
        self._claim_slot(0, session)
        single_slot_tier = "cold_with_checkpoint" if restored else "cold_no_checkpoint"
        bind_kv_tier(single_slot_tier)
        log.info(
            "kv_tier_decided",
            session=session,
            tier=single_slot_tier,
            slot=0,
            restored=bool(restored),
            multislot=False,
            request_id=request_id_var.get() or "",
            phases=phases,
        )
        if restored:
            log.info("session_kv_restored", session=session)

    async def prepare_stable_checkpoint(
        self,
        request: InternalChatRequest,
        assistant_content: str,
    ) -> bool:
        """Rebuild slot 0 to a stable checkpoint for the next narrative turn.

        Called by the narrative handler as a background task AFTER the
        chat-stream consumer closes — i.e. outside the original chat
        request's ``request_in_flight()`` scope. The prewarm itself can
        take 5-10 s on long narrative contexts (90k+ tokens), and if the
        idle monitor's countdown started the moment the chat stream
        ended, ``stop()`` could fire mid-prewarm and yank the subprocess
        out from under us.

        Wrap the prewarm + save in ``request_in_flight()`` so the idle
        monitor refuses to unload until the checkpoint completes —
        same protection chat() / chat_stream() apply to inference.
        """
        if not assistant_content or not request.kv_stable_messages:
            return False
        if self._manager is None:
            return False

        checkpoint_messages = self._clone_messages(request.kv_stable_messages)
        checkpoint_messages.append(Message(role="assistant", content=assistant_content))
        checkpoint_key = self._stable_checkpoint_key(request, checkpoint_messages)
        checkpoint_request = self._checkpoint_request_from_messages(
            request,
            checkpoint_messages,
            checkpoint_key,
        )
        checkpoint_payload = [
            {"role": m.role, "content": m.content or ""}
            for m in checkpoint_messages
        ]

        # Record the prewarm content for the stable-prefix metric — the
        # next real turn is compared against this, so "the prewarm
        # prefilled content the next turn can't match" is measured live
        # (kv_prefix_stability baseline=prewarm) instead of assumed.
        # Keyed by the BASE conversation key (checkpoint_request carries
        # the per-turn ``::stable::<digest>`` key, which the next turn
        # would never match).
        try:
            self.track_prefix_stability(
                dataclass_replace(
                    checkpoint_request,
                    kv_session_key=(request.kv_session_key or ""),
                ),
                source="prewarm",
            )
        except Exception:
            log.warning("kv_prefix_stability_failed", exc_info=True)

        # Choose which slot to prewarm into.
        #
        # Single-slot mode: only slot 0 exists; that's the user-facing
        # chat slot AND the checkpoint slot. Prewarm acquires slot 0's
        # lock so chat queues behind it — known UX bug, ~29s of dead
        # air on regenerate after a long-context response (measured
        # 2026-05-05). Phase 2 multi-slot mode is the fix.
        #
        # Multi-slot mode: target the SAME slot the chat just ran on.
        # The checkpoint content is a byte-extension of that slot's live
        # KV (clean stable prefix + the assistant reply), so prewarming
        # in place prefills only the delta AND drops llama-server's
        # end-of-prefill context checkpoint exactly at the stable
        # boundary — which is what makes next-turn reuse work on
        # hybrid-attention models (Qwen3.5+) that can't rewind past a
        # checkpoint. The previous ``avoid=chat_slot`` policy paid a
        # FULL prefill of the whole context on a second slot after
        # every turn (pure GPU churn), and under upstream's
        # cache_idle_slots default its launch evicted the just-idled
        # chat slot — the direct cause of the 12-15 min narrative TTFTs
        # (2026-07-02 trace). When the chat slot is unknown (occupancy
        # empty after restart), send unpinned: with idle slots retaining
        # KV (--no-cache-idle-slots), llama-server's LCP similarity
        # router lands the prewarm on whichever slot holds the matching
        # prefix — same outcome, self-correcting.
        prior_chat_session = (request.kv_session_key or "").strip()
        prior_chat_slot = (
            self._get_slot_for_session(prior_chat_session)
            if prior_chat_session else None
        )
        if self._multislot_enabled():
            target_slot = prior_chat_slot if prior_chat_slot is not None else None
        else:
            target_slot = 0

        async with self._manager.request_in_flight():
            if target_slot is None:
                # Chat slot unknown (occupancy empty — e.g. first turn
                # after restart). Send the prewarm UNPINNED and let
                # llama-server's LCP similarity router land it on
                # whichever slot holds the matching prefix. No slot
                # lock / occupancy bookkeeping — the next turn's
                # response observation reconciles occupancy via
                # id_slot, and routing correctness doesn't depend on
                # our map (the router matches tokens, not bookkeeping).
                warmed = await self.prewarm_context(checkpoint_payload, slot_id=None)
                if warmed is None:
                    log.warning(
                        "stable_checkpoint_prewarm_failed",
                        session=checkpoint_key,
                        slot=None,
                        messages=len(checkpoint_messages),
                    )
                    return False
                log.info(
                    "stable_checkpoint_prepared",
                    session=checkpoint_key,
                    slot=None,
                    messages=len(checkpoint_messages),
                    saved=False,
                    multislot=self._multislot_enabled(),
                )
                # Upgrade the replay source with the post-response
                # prefix (stable history + the reply we just generated)
                # under the BASE session key — the checkpoint key is a
                # per-turn digest an on-open resume could never derive.
                self._schedule_replay_capture(
                    checkpoint_request,
                    session_key_override=(request.kv_session_key or ""),
                )
                return True

            async with self._get_slot_lock(target_slot):
                # Rebind the slot's occupancy to the new checkpoint key.
                # Targeting the chat slot means the "displaced" session
                # is normally this conversation's own previous stable
                # key — the prewarm byte-EXTENDS that KV rather than
                # overwriting it. A genuinely different displaced
                # session (slot got re-routed between turns) still gets
                # its state saved before we extend over it.
                displaced_session = self._get_session_for_slot(target_slot)
                if displaced_session and displaced_session != checkpoint_key:
                    same_conversation = displaced_session.startswith(
                        prior_chat_session.split("::stable::")[0]
                    ) if prior_chat_session else False
                    if not same_conversation:
                        # Save before evicting so we don't lose KV that's
                        # newer than the on-disk version.
                        await self.save_session_state(displaced_session, slot_id=target_slot)
                self._release_slot(target_slot)

                # Pin the prewarm to the chat slot so it extends the
                # live KV in place (delta-only prefill + end-of-prefill
                # checkpoint at the stable boundary).
                warmed = await self.prewarm_context(
                    checkpoint_payload, slot_id=target_slot,
                )
                if warmed is None:
                    log.warning(
                        "stable_checkpoint_prewarm_failed",
                        session=checkpoint_key,
                        slot=target_slot,
                        messages=len(checkpoint_messages),
                    )
                    return False

                self._claim_slot(target_slot, checkpoint_key)
                saved = await self.save_session_state(
                    checkpoint_key, slot_id=target_slot, request=checkpoint_request,
                )
                log.info(
                    "stable_checkpoint_prepared",
                    session=checkpoint_key,
                    slot=target_slot,
                    messages=len(checkpoint_messages),
                    saved=saved,
                    multislot=self._multislot_enabled(),
                )
                # Same upgrade as the unpinned branch: replay source =
                # post-response prefix, keyed by the BASE session key.
                self._schedule_replay_capture(
                    checkpoint_request,
                    session_key_override=(request.kv_session_key or ""),
                )
                return True

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Send a chat request via the OpenAI-compatible endpoint.

        Slot routing:
          - Default (``request.is_background_task == False``): runs on slot 0
            with slot 0's lifecycle lock — preserves user-chat KV affinity and
            keeps single-slot mode working unchanged.
          - Background task (``is_background_task == True``) AND multi-slot
            enabled: picks a non-zero slot via ``_pick_checkpoint_target_slot
            (avoid=0)``, holds THAT slot's lock, and stamps ``id_slot`` into
            the payload via ``_to_openai_payload`` / ``_to_completion_payload``
            so llama-server doesn't re-route. Lets memory refresh, ledger
            compaction, and narrative extraction run in parallel with the
            user's next turn instead of queueing behind slot 0.
          - Background task BUT single-slot mode: degrades to slot 0
            (unchanged behavior — there's only one slot to use).
        """
        # Real traffic preempts any in-flight speculation (kv_speculate
        # rule 2). Non-stream requests don't serve from speculation —
        # every UI chat surface streams — but they must still win the
        # GPU immediately.
        speculator = getattr(self, "_speculator", None)
        if speculator is not None and not getattr(
            request, "_augmentum_speculative", False,
        ):
            await speculator.preempt("real_traffic")

        t_start = time.monotonic()
        target_slot = self._pick_request_slot(request)
        # In-flight tracking: keeps the manager's idle monitor from
        # killing the subprocess mid-request. ``request_in_flight()`` is
        # exception-safe — counter decrements via ``finally`` even on
        # error/cancel. See LlamaServerManager.request_in_flight docstring.
        if self._manager is not None:
            async with self._manager.request_in_flight():
                await self._ensure_server(request.model)
                self.pre_stream_validate(request)
                async with self._get_slot_lock(target_slot):
                    return await self._chat_with_slot(request, t_start, target_slot=target_slot)
        # No manager — external sidecar (classifier container) or test
        # config. The slot lock exists to serialize KV save/restore and
        # lifecycle work on the MANAGED server; with no manager,
        # _manage_slot is a no-op and the server owns its own slot
        # routing + concurrency (-np N). Holding the lock here only
        # serialized independent lanes (game-agent fast/scene/planner,
        # voice router) behind one another — measured as fast-turn p90
        # 12-15s while planner turns ran (2026-07-02).
        await self._ensure_server(request.model)
        return await self._chat_with_slot(request, t_start, target_slot=target_slot)

    def _pick_request_slot(self, request: InternalChatRequest) -> int:
        """Pick the slot for this request. Background tasks get a non-zero
        slot when multi-slot is enabled; everything else stays on slot 0."""
        if not getattr(request, "is_background_task", False):
            return 0
        if not self._multislot_enabled():
            return 0
        return self._pick_checkpoint_target_slot(avoid=0)

    async def _chat_with_slot(
        self, request: InternalChatRequest, t_start: float,
        *, target_slot: int = 0,
    ) -> InternalChatResponse:
        """Chat body running under the target slot's lifecycle lock.

        ``target_slot=0`` keeps the historical single-slot behavior. Non-zero
        means the caller (chat()) routed a background task off slot 0; we
        skip _manage_slot's KV-affinity work (background tasks don't have a
        kv_session_key) and stamp ``id_slot`` into the outbound payload so
        llama-server doesn't re-route based on prefix LCP.
        """
        # Save/restore KV slot state for session switching. Skip for
        # background tasks routed off slot 0 — they don't have a stable
        # session fingerprint to restore, and the slot-0 occupancy state
        # would be wrong for them anyway.
        if target_slot == 0:
            await self._manage_slot(request)
            # Persist this request's exact prefix so the resume ladder
            # can replay it after a restart (rung 2 — the only recovery
            # that exists under --kv-unified). Fire-and-forget.
            self._schedule_replay_capture(request)
        else:
            # Background task on a non-zero slot didn't go through
            # _manage_slot, so nothing has bound this task's `kv_tier_var`.
            # ContextVars inherit values from the parent task at
            # ``asyncio.create_task`` time — without an explicit re-bind
            # here, `engine_perf` reads the spawning chat request's tier
            # (e.g. ``cold_no_checkpoint`` from the user-facing turn that
            # ran on slot 0) and reports it for THIS background inference,
            # which actually ran on a separate warm slot. Misleading
            # telemetry — see 2026-05-20 KV tier leak investigation.
            from augmentum.proxy.status_bus import bind_kv_tier
            bind_kv_tier("unmanaged")

        # Try pre-tokenized path for managed server (eligible requests only)
        tokens = await self._build_token_prompt(request)
        if tokens is not None:
            payload = self._to_completion_payload(request, tokens)
            payload["stream"] = False
            if target_slot != 0:
                payload["id_slot"] = target_slot
            resp = await self._client.post(
                f"{self._base_url}/completion",
                json=payload,
                headers=self._headers(),
                timeout=self._INFERENCE_TIMEOUT,
            )
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = (resp.text or "").strip()[:500]
                    log.warning(
                        "completion_endpoint_invalid_json",
                        status=resp.status_code,
                        body=body or "(empty body)",
                        fallback="chat/completions",
                    )
                else:
                    content = data.get("content", "")
                    raw_thinking = None
                    clean_content, thinking_text = normalize_thinking(
                        content,
                        raw_thinking,
                        model=request.model or data.get("model"),
                        thinking_enabled=request.think,
                        preserve_thinking=bool(request.preserve_thinking),
                    )
                    usage_data = data.get("usage", data.get("timings", {}))
                    timings = data.get("timings", {})
                    if timings:
                        self._log_performance(timings, t_start, None)
                    # Newer llama-server returns OpenAI-style ``usage``
                    # (prompt_tokens/completion_tokens/total_tokens);
                    # older builds only return ``timings`` (prompt_n/
                    # predicted_n). Resolve from either, and compute the
                    # total from the resolved locals so UARF's budget
                    # accounting (analytical/engine.py) stays accurate.
                    prompt_tokens = usage_data.get(
                        "prompt_n", usage_data.get("prompt_tokens", 0),
                    )
                    completion_tokens = usage_data.get(
                        "predicted_n", usage_data.get("completion_tokens", 0),
                    )
                    total_tokens = usage_data.get(
                        "total_tokens", prompt_tokens + completion_tokens,
                    )
                    # /completion (raw, non-OAI) returns cache_n on the
                    # top-level dict, not nested under timings — different
                    # path than /v1/chat/completions.
                    cache_n_raw = int(data.get("cache_n", 0) or 0)
                    self._audit_kv_reuse(
                        request,
                        evaluated_n=int(prompt_tokens or 0),
                        cache_n=cache_n_raw,
                        endpoint="completion",
                    )
                    return InternalChatResponse(
                        message=Message(
                            role="assistant",
                            content=clean_content,
                            thinking=thinking_text or None,
                        ),
                        model=data.get("model", request.model),
                        finish_reason="stop" if data.get("stop", True) else "length",
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cache_hit_tokens=cache_n_raw,
                            cache_miss_tokens=int(prompt_tokens) if cache_n_raw else 0,
                        ),
                    )
            # Fall through to the OpenAI-compatible path on error.
            if resp.status_code >= 400:
                log.warning("completion_endpoint_failed", status=resp.status_code, fallback="chat/completions")

        payload = self._to_openai_payload(request)
        payload["stream"] = False
        self._apply_reasoning_request_options(payload, request)
        if self._manager is not None:
            payload["cache_prompt"] = True
        if target_slot != 0:
            payload["id_slot"] = target_slot

        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self._INFERENCE_TIMEOUT,
        )
        if resp.status_code >= 400:
            body = resp.text[:500]
            log.error(
                "llamacpp_chat_error",
                status=resp.status_code,
                url=f"{self._base_url}/v1/chat/completions",
                body=body,
            )
            raise RuntimeError(
                f"llama.cpp returned {resp.status_code}: {body}"
            )
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            body = (resp.text or "").strip()[:500]
            log.error(
                "llamacpp_chat_invalid_json",
                status=resp.status_code,
                url=f"{self._base_url}/v1/chat/completions",
                body=body or "(empty body)",
            )
            raise RuntimeError(
                f"llama.cpp returned non-JSON body on 2xx: {body or '(empty)'}"
            ) from exc

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage_data = data.get("usage", {})

        raw_content = msg.get("content", "")
        native_thinking = msg.get("reasoning_content")
        clean_content, thinking_text = normalize_thinking(
            raw_content,
            native_thinking,
            model=request.model or data.get("model"),
            thinking_enabled=request.think,
        )

        completion_details = (usage_data or {}).get("completion_tokens_details") or {}
        timings = data.get("timings") or {}
        cache_n_nonstream = int(timings.get("cache_n", 0) or 0)
        prompt_tokens_nonstream = int(usage_data.get("prompt_tokens", 0) or 0)
        if timings:
            # Same evaluated-only contract as the chat_stream site above —
            # usage.prompt_tokens includes cache hits and must not be fed
            # to the audit as the evaluated count.
            self._audit_kv_reuse(
                request,
                evaluated_n=int(
                    timings.get("prompt_n")
                    or max(0, prompt_tokens_nonstream - cache_n_nonstream)
                ),
                cache_n=cache_n_nonstream,
                endpoint="chat",
            )
        response = InternalChatResponse(
            message=Message(
                role=msg.get("role", "assistant"),
                content=clean_content,
                # llama-server emits native tool_calls when --jinja is on
                # and the chat template formats tool schemas. Without
                # surfacing them here the field stays None, which silently
                # drops every non-streaming tool call — the streaming path
                # at line ~2694 already handles them. Pre-2026-05-26 this
                # silently broke any non-stream caller; OpenAI-shape and
                # Anthropic-shape external clients hit it first.
                tool_calls=msg.get("tool_calls"),
                thinking=thinking_text or None,
            ),
            model=data.get("model", request.model),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=Usage(
                prompt_tokens=prompt_tokens_nonstream,
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cache_hit_tokens=cache_n_nonstream or int(
                    (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                ),
                cache_miss_tokens=prompt_tokens_nonstream if cache_n_nonstream else 0,
                reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            ),
        )
        # Save KV state after successful non-streaming response.
        # See _save_after_stream for the multi-slot lookup rationale.
        await self._save_after_stream(request)
        return response

    def _needs_model_load(self) -> bool:
        """Check if a model load/restart will be needed on next request.

        Used by ``chat_stream`` to decide whether to emit a ``loading``
        status chunk before the real work begins. We keep the ``state !=
        READY`` arm explicit so STARTING / STOPPING / DRAINING all signal
        "please show a loading indicator" even though the subprocess may
        technically be alive during those transitions.

        For the ``READY`` arm we defer to ``check_alive()`` rather than
        reading ``process.returncode`` directly: ``check_alive()``
        performs the same check AND resets the manager to ``IDLE`` +
        records ``_last_crashed_model`` on a detected crash, so the
        subsequent ``_ensure_server`` call sees the correct state and
        can trigger an OOM-aware restart.
        """
        if self._manager is None:
            return False
        from augmentum.models.llama_server_manager import ProcessState
        if self._manager.state != ProcessState.READY:
            return True
        return not self._manager.check_alive()

    def _ensure_server_stage_for(self, requested_model: str) -> Stage | None:
        """Classify the upcoming ``_ensure_server`` call as a stage.

        Returns a ``Stage`` describing what's about to happen — model
        load (cold start), model swap (hot replace), or ``None`` if
        ``_ensure_server`` will be a no-op (server ready, same model).

        The conditions mirror the dispatch inside
        ``_ensure_server_locked`` so the stage event matches what the
        engine actually does. We compute this externally rather than
        inside _ensure_server so chat_stream can yield a stage_start
        chunk BEFORE the work begins; the alternative — yielding from
        inside _ensure_server — would require routing chunks through a
        ContextVar+queue, which is overkill for the three stages we're
        instrumenting today.
        """
        if self._manager is None:
            return None
        from augmentum.models.llama_server_manager import ProcessState

        is_generic = not requested_model or requested_model.lower() == "default"

        if self._manager.state == ProcessState.READY and self._manager.check_alive():
            # Server is up. A swap is required only if a non-generic
            # model name was requested AND it differs from the loaded
            # one. Otherwise _ensure_server returns immediately — no
            # observable stage to surface.
            if is_generic or requested_model == self._manager.model_id:
                return None
            return Stage(
                "model_swap",
                label="Switching model",
                detail=f"to {requested_model}",
            )

        # Server not ready (IDLE / STARTING / STOPPING / DRAINING / crashed).
        # _ensure_server will load. Detail is the resolved model name when
        # available — otherwise the manager will pick the last-crashed or
        # error out, both of which the user shouldn't have to read.
        target = requested_model or (self._manager.model_id or "model")
        return Stage("model_load", label="Loading model", detail=target)

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream a chat response via SSE."""
        # Speculative-turn interception (kv_speculate — ladder rung 3).
        # A real request always preempts an in-flight speculation; if a
        # *finished* speculation byte-matches this request, its recorded
        # stream serves with zero engine work. The attr is only set once
        # the speculate endpoint has been used — until then this is one
        # getattr per request.
        speculator = getattr(self, "_speculator", None)
        if speculator is not None and not getattr(
            request, "_augmentum_speculative", False,
        ):
            entry = await speculator.on_real_request(request)
            if entry is not None:
                # The served turn still becomes the next turn's replay
                # row — the session's warm lineage continues normally.
                self._schedule_replay_capture(request)
                async for chunk in speculator.replay_chunks(entry, request):
                    yield chunk
                return

        # Classify what _ensure_server is about to do (model_load,
        # model_swap, or no-op) so we can yield richly-typed stage
        # events around it — see _ensure_server_stage_for. The legacy
        # single-string ``status`` field is emitted alongside for
        # backwards compatibility with frontends that haven't picked up
        # ``stage_start`` yet.
        ensure_stage = self._ensure_server_stage_for(request.model)
        if ensure_stage is not None:
            legacy_status = (
                "loading" if ensure_stage.name == "model_load" else "swapping"
            )
            yield InternalStreamChunk(
                content_delta="", role="assistant", model=request.model,
                augmentum={"status": legacy_status, **ensure_stage.start_payload()},
            )

        # In-flight tracking. Wraps the full generation lifecycle —
        # model load, slot save/restore, prefill, thinking, generation,
        # post-response checkpoint save. The manager's idle monitor
        # refuses to unload while this counter is >0, fixing the
        # "5-minute prefill on a 90k context tripped the 10-min idle
        # timeout and killed the request mid-stream" bug. Counter
        # decrements via ``finally`` even on cancel/error/generator
        # close. See LlamaServerManager.request_in_flight docstring.
        if self._manager is not None:
            async with self._manager.request_in_flight():
                async for chunk in self._run_with_ensure_stage(request, ensure_stage):
                    yield chunk
        else:
            # No manager — tests and standalone configs.
            async for chunk in self._run_with_ensure_stage(request, ensure_stage):
                yield chunk

    async def _run_with_ensure_stage(
        self,
        request: InternalChatRequest,
        ensure_stage: Stage | None,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Run ``_ensure_server`` then ``_chat_stream_with_slot``,
        emitting ``stage_complete`` for the ensure step.

        Extracted so the manager and no-manager branches in
        ``chat_stream`` share one code path; without this, every
        instrumentation change had to be duplicated and one branch
        would inevitably drift.
        """
        try:
            await self._ensure_server(request.model)
        except Exception as exc:
            if ensure_stage is not None:
                yield InternalStreamChunk(
                    model=request.model,
                    augmentum=ensure_stage.complete_payload(
                        success=False, error_text=str(exc)[:200],
                    ),
                )
            raise

        if ensure_stage is not None:
            yield InternalStreamChunk(
                model=request.model,
                augmentum=ensure_stage.complete_payload(success=True),
            )

        # Refuse image-bearing requests routed to a text-only model.
        # Runs post-swap (after _ensure_server) so we validate against
        # the model that will actually serve the request. See
        # pre_stream_validate on the base class for why this gate exists.
        self.pre_stream_validate(request)

        if self._manager is None:
            # External sidecar: no managed KV/lifecycle to protect, and
            # the server handles its own slot routing (-np N). Same
            # rationale as the non-stream path — the app-side lock only
            # serializes independent callers.
            async for chunk in self._chat_stream_with_slot(request):
                yield chunk
            return
        async with self._get_slot_lock(0):
            async for chunk in self._chat_stream_with_slot(request):
                yield chunk

    async def _chat_stream_with_slot(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Streaming body running under the slot 0 lifecycle lock."""
        # Save/restore KV slot state for session switching. Comparison
        # against the session currently in slot 0 is lock-protected so
        # the status chunk reflects committed state. We also probe the
        # slot dir before claiming "restoring" — a session change with
        # no saved state on disk just prefills cold, which isn't a
        # restore.
        session_fp = self._session_fingerprint(request)
        slot_0_session = self._get_session_for_slot(0)
        will_restore = (
            session_fp
            and session_fp != slot_0_session
            and slot_0_session
            and self._slot_state_exists(session_fp)
        )
        restore_stage = (
            Stage("slot_restore", label="Restoring session", detail="")
            if will_restore else None
        )
        if restore_stage is not None:
            yield InternalStreamChunk(
                content_delta="", model=request.model,
                augmentum={
                    "status": "restoring",
                    **restore_stage.start_payload(),
                },
            )

        try:
            await self._manage_slot(request)
        except Exception as exc:
            if restore_stage is not None:
                yield InternalStreamChunk(
                    model=request.model,
                    augmentum=restore_stage.complete_payload(
                        success=False, error_text=str(exc)[:200],
                    ),
                )
            raise
        # Persist this request's exact prefix for the resume ladder's
        # replay rung (fire-and-forget; see _schedule_replay_capture).
        self._schedule_replay_capture(request)

        if restore_stage is not None:
            yield InternalStreamChunk(
                model=request.model,
                augmentum=restore_stage.complete_payload(success=True),
            )

        # Prefill stage — covers prompt processing on llama-server
        # before the first token arrives. On long-context chats (e.g.
        # 90k tokens) prefill is 30+ seconds; without a stage event
        # the UI shows the previous label the whole time even though
        # the model was ready 25 seconds ago. The legacy ``tokenizing``
        # status maps to "Preparing context…" in renderer.js for older
        # frontends; new frontends use ``stage_start.label``. We don't
        # emit ``stage_complete`` for prefill — the first content delta
        # arriving is the implicit completion signal, and asking the
        # frontend to read both events doubles the wire complexity.
        prefill_detail = ""
        # Surface a context-utilization hint when we know the prompt
        # size — useful for "is my 90k chat about to be slow?" awareness.
        if request.kv_stable_messages:
            prefill_detail = f"{len(request.kv_stable_messages)} messages"
        prefill_stage = Stage(
            "prefill", label="Preparing context", detail=prefill_detail,
        )
        yield InternalStreamChunk(
            content_delta="", model=request.model,
            augmentum={
                "status": "tokenizing",
                **prefill_stage.start_payload(),
            },
        )

        # Try pre-tokenized /completion path (eligible requests only)
        tokens = await self._build_token_prompt(request)
        if tokens is not None:
            async for chunk in self._stream_completion(request, tokens):
                yield chunk
            # Save KV state after successful completion. Lookup the
            # actually-served slot via the session fingerprint:
            #   - Single-slot (flag off): _manage_slot claimed slot 0
            #     for this session, so the lookup returns 0.
            #   - Multi-slot (flag on): _stream_completion observed
            #     id_slot from the response and claimed it, so the
            #     lookup returns the engine-picked slot.
            # Either way we save against the slot that actually holds
            # the KV we want persisted.
            await self._save_after_stream(request)
            return

        # Fall back to standard /v1/chat/completions
        async for chunk in self._stream_chat_completions(request):
            yield chunk

        # Save KV state after successful completion (see comment above)
        await self._save_after_stream(request)

    async def _save_after_stream(self, request: InternalChatRequest) -> None:
        """Save the served slot's KV after a streaming chat completes.

        Locates the served slot via session→slot inverse index. If the
        request has no session affinity (opaque external client) or the
        engine never observed a claimable slot, this is a no-op.
        """
        if self._should_defer_session_save(request):
            return
        session_fp = self._session_fingerprint(request)
        if not session_fp:
            return
        slot_id = self._get_slot_for_session(session_fp)
        if slot_id is None:
            # No occupancy for this session — either single-slot mode
            # didn't claim (e.g. _manage_slot early-returned because
            # session matched), or multi-slot mode never observed
            # id_slot (e.g. /v1/chat/completions fallback path which
            # uses OAI compat that doesn't return id_slot). In either
            # case, default to slot 0 for backward compat.
            slot_id = 0
        await self.save_session_state(session_fp, slot_id=slot_id, request=request)

    async def _stream_completion(
        self, request: InternalChatRequest, tokens: list[int],
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream from /completion with pre-tokenized prompt.

        Emits ``done=True`` exactly once — from the ``[DONE]`` SSE
        marker (or the EOF fallback). The intermediate data chunk that
        carries ``stop: true`` is the model's last token + its timings;
        we capture its Usage but DON'T set ``done=True`` on it. If we
        did, an outer SSE wrapper that returns on the first ``done``
        chunk would terminate before the ``[DONE]`` flush ran, dropping
        any partial-tag content held in the thinking buffer at stream
        end. Mirrors the rule in ``_stream_chat_completions`` (see its
        ``[DONE]`` branch's docstring).
        """
        payload = self._to_completion_payload(request, tokens)
        payload["stream"] = True

        thinking_buf = ThinkingStreamBuffer(
            family=self.reasoning_family(request.model),
            model=request.model, thinking_enabled=request.think,
            preserve_thinking=bool(request.preserve_thinking),
        )
        t_start = time.monotonic()
        t_first_token: float | None = None
        # Usage extracted from the final data chunk's ``timings`` field.
        # Propagated to the [DONE] (or EOF) terminator, never emitted
        # on an intermediate chunk.
        final_usage: Usage | None = None
        # Cached-prompt tokens from the same timings block — added to
        # ``final_usage.prompt_tokens`` (the freshly-evaluated count) so
        # the context-usage payload reports cumulative occupancy of the
        # window rather than just this turn's delta.
        final_cache_n: int = 0
        # KV reuse-audit aug fields from the stop:true timings, merged
        # into the terminal chunk's augmentum payload.
        kv_aug: dict | None = None
        # Whether we observed a ``stop: true`` chunk before stream end.
        # llama-server's raw /completion endpoint does NOT emit a
        # ``data: [DONE]`` sentinel — that's an OpenAI-compat
        # convention only — so a clean termination ends with a
        # ``stop: true`` chunk followed by socket close. We treat
        # EOF-after-stop-true as a normal end (debug log) and
        # EOF-without-stop-true as a real truncation (warning).
        saw_stop_true: bool = False
        # Phase 0 observation (multi-slot KV design): /completion
        # streaming chunks include ``id_slot`` per
        # llama.cpp@b8935 ``server_task_result_cmpl_partial::to_json_non_oaicompat``.
        # We capture it from the first chunk that carries it (typically
        # all chunks, but consistent with the source we treat any non-
        # negative observation as authoritative). Logged once per
        # request to verify the wire contract before Phase 2 acts on it.
        observed_slot_id: int = -1

        async with self._client.stream(
            "POST",
            f"{self._base_url}/completion",
            json=payload,
            headers=self._headers(),
            timeout=self._INFERENCE_TIMEOUT,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning("completion_stream_error", status=resp.status_code, body=body)
                raise RuntimeError(f"llama.cpp /completion returned {resp.status_code}: {body}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    flush_content, flush_thinking = thinking_buf.flush()
                    # Slot-id fallback: under llama.cpp's ``--kv-unified`` /
                    # ``--parallel -1`` config the response chunks don't
                    # always carry ``id_slot``, so observed_slot_id can
                    # stay -1 even though the prefill clearly ran on some
                    # slot (visible in the ``slot print_timing: id N``
                    # log line our status parser scrapes into
                    # _prefill_progress). Without this, _claim_slot never
                    # fires and the next turn reports cold_no_checkpoint
                    # despite the slot's KV being live and warm.
                    if (
                        observed_slot_id < 0
                        and self._multislot_enabled()
                        and self._manager is not None
                    ):
                        fallback = (
                            getattr(self._manager, "_prefill_progress", None)
                            or {}
                        ).get("slot_id")
                        if isinstance(fallback, int) and fallback >= 0:
                            observed_slot_id = fallback
                            session_fp = self._session_fingerprint(request)
                            if session_fp:
                                self._claim_slot(observed_slot_id, session_fp)
                                log.debug(
                                    "slot_claim_via_prefill_log",
                                    slot=observed_slot_id,
                                    session=session_fp,
                                    note="id_slot missing from response chunks; "
                                         "claimed via prefill_progress.slot_id",
                                )
                    if observed_slot_id >= 0:
                        log.info(
                            "slot_observation",
                            endpoint="completion",
                            id_slot=observed_slot_id,
                            session=self._get_session_for_slot(0),
                            prompt_tokens=len(tokens) if tokens else 0,
                        )
                    merged = {
                        **(self._context_usage_payload(
                            final_usage.prompt_tokens if final_usage else 0,
                            cache_n=final_cache_n,
                        ) or {}),
                        **(kv_aug or {}),
                    }
                    ctx_payload = merged or None
                    yield InternalStreamChunk(
                        done=True,
                        model=request.model,
                        content_delta=flush_content,
                        thinking_delta=flush_thinking,
                        usage=final_usage,
                        augmentum=ctx_payload,
                    )
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # /completion streaming format: {"content": "token", "stop": false,
                # "id_slot": N, ...}. id_slot is in every chunk per upstream
                # at b8935; we record the first non-negative observation.
                if observed_slot_id < 0:
                    raw_id_slot = data.get("id_slot")
                    if isinstance(raw_id_slot, int) and raw_id_slot >= 0:
                        observed_slot_id = raw_id_slot
                        # Phase 2: claim the served slot for this session.
                        # In single-slot mode (multislot flag off) the
                        # engine always reports slot 0 and _manage_slot
                        # already claimed slot 0 → this is a no-op
                        # confirmation. In multi-slot mode this is the
                        # authoritative occupancy update — the engine's
                        # auto-routing may pick a different slot than the
                        # one we restored into, and the response is the
                        # only source of truth on what actually served us.
                        if self._multislot_enabled():
                            session_fp = self._session_fingerprint(request)
                            if session_fp:
                                self._claim_slot(observed_slot_id, session_fp)

                raw_content = data.get("content", "")
                clean_content, thinking = thinking_buf.process(raw_content, "")

                if t_first_token is None and raw_content:
                    t_first_token = time.monotonic()

                # Capture timings from the model's final chunk. We do
                # NOT propagate ``done=True`` here — the [DONE] (or EOF)
                # branch is the single source of stream termination.
                if data.get("stop", False):
                    saw_stop_true = True
                    timings = data.get("timings", {})
                    if timings:
                        cache_n = int(timings.get("cache_n", 0) or 0)
                        prompt_n = int(timings.get("prompt_n", 0) or 0)
                        final_usage = Usage(
                            prompt_tokens=prompt_n,
                            completion_tokens=timings.get("predicted_n", 0),
                            total_tokens=prompt_n + timings.get("predicted_n", 0),
                            # llama-server splits the prompt into freshly
                            # evaluated tokens (``prompt_n``) and KV-cache
                            # hits (``cache_n``). Mirror the DeepSeek
                            # cache-hit/miss shape so downstream consumers
                            # have one typed surface to read.
                            cache_hit_tokens=cache_n,
                            cache_miss_tokens=prompt_n,
                            # Authoritative decode wall-time — lets the
                            # proxy report a true tok/s even when this
                            # build dumps the whole completion at once.
                            eval_duration_ms=float(timings.get("predicted_ms") or 0.0),
                        )
                        final_cache_n = cache_n
                        self._log_performance(timings, t_start, t_first_token)
                        kv_aug = self._audit_kv_reuse(
                            request,
                            evaluated_n=prompt_n,
                            cache_n=cache_n,
                            endpoint="completion_stream",
                        )

                yield InternalStreamChunk(
                    content_delta=clean_content,
                    thinking_delta=thinking,
                    role="assistant",
                    model=data.get("model", request.model),
                    done=False,
                )

            # Stream ended without ``[DONE]`` — emit a terminal chunk so
            # the outer SSE wrapper sees ``chunk.done`` and closes
            # cleanly. Carries whatever Usage we captured from the
            # stop:true chunk earlier; if no stop:true arrived either,
            # the frontend falls back to the _StreamTimer estimate.
            #
            # Severity split:
            #   - saw_stop_true → normal /completion termination
            #     (the raw endpoint never sends [DONE], unlike the
            #     OAI-compat /v1/chat/completions endpoint). Debug-only.
            #   - !saw_stop_true → real truncation: model crashed,
            #     upstream socket reset, or context-window overflow
            #     killed the request before producing any output.
            #     Worth a warning so it surfaces in dashboards.
            flush_content, flush_thinking = thinking_buf.flush()
            if saw_stop_true:
                log.debug("completion_stream_clean_eof")
            else:
                log.warning(
                    "completion_stream_truncated",
                    detail="stream ended before stop:true (model truncation or upstream reset)",
                )
            # Same id_slot fallback as the [DONE] branch above —
            # see that comment for the --kv-unified rationale.
            if (
                observed_slot_id < 0
                and self._multislot_enabled()
                and self._manager is not None
            ):
                fallback = (
                    getattr(self._manager, "_prefill_progress", None) or {}
                ).get("slot_id")
                if isinstance(fallback, int) and fallback >= 0:
                    observed_slot_id = fallback
                    session_fp = self._session_fingerprint(request)
                    if session_fp:
                        self._claim_slot(observed_slot_id, session_fp)
            if observed_slot_id >= 0:
                log.info(
                    "slot_observation",
                    endpoint="completion",
                    id_slot=observed_slot_id,
                    session=self._get_session_for_slot(0),
                    prompt_tokens=len(tokens) if tokens else 0,
                    eof_no_done=True,
                    saw_stop_true=saw_stop_true,
                )
            merged = {
                **(self._context_usage_payload(
                    final_usage.prompt_tokens if final_usage else (len(tokens) if tokens else 0),
                    cache_n=final_cache_n,
                ) or {}),
                **(kv_aug or {}),
            }
            ctx_payload = merged or None
            yield InternalStreamChunk(
                done=True,
                model=request.model,
                content_delta=flush_content,
                thinking_delta=flush_thinking,
                usage=final_usage,
                augmentum=ctx_payload,
            )

    async def _stream_chat_completions(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream from /v1/chat/completions (standard OpenAI path)."""
        payload = self._to_openai_payload(request)
        payload["stream"] = True
        self._apply_reasoning_request_options(payload, request)
        if self._manager is not None:
            payload["cache_prompt"] = True
        # Ask llama-server to include per-request usage AND its
        # ``timings`` extension (prompt_n / prompt_ms / predicted_n /
        # predicted_ms / cache_n) in the final pre-[DONE] chunk.
        # Without this we'd have to fall back to GET /slots and read
        # ``slot[0]`` — which is wrong under multi-slot (the slot that
        # actually served this request may be anything), AND adds a
        # second HTTP roundtrip per request whose metrics may already
        # have been overwritten by a concurrent generation. Inline
        # timings are authoritative AND free.
        existing_stream_opts = payload.get("stream_options") or {}
        existing_stream_opts["include_usage"] = True
        payload["stream_options"] = existing_stream_opts

        thinking_buf = ThinkingStreamBuffer(
            family=self.reasoning_family(request.model),
            model=request.model, thinking_enabled=request.think,
            preserve_thinking=bool(request.preserve_thinking),
        )
        t_start = time.monotonic()
        t_first_token: float | None = None
        # Whether we observed any chunk with a non-null ``finish_reason``
        # before stream end. The OAI-compat /v1/chat/completions
        # endpoint normally emits one such chunk (``finish_reason="stop"``
        # / "length" / "tool_calls" / etc.) immediately before the
        # ``data: [DONE]`` sentinel. EOF-after-finish-reason is a clean
        # stream that lost only the [DONE] marker; EOF-without is a
        # real truncation.
        saw_finish_reason: bool = False

        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self._INFERENCE_TIMEOUT,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning(
                    "llamacpp_stream_error",
                    status=resp.status_code,
                    url=f"{self._base_url}/v1/chat/completions",
                    body=body,
                )
                raise RuntimeError(
                    f"llama.cpp returned {resp.status_code}: {body}"
                )
            # Emit ``done=True`` exactly once — from the ``[DONE]`` branch
            # (or the EOF fallback). If we also flipped ``done=True`` on the
            # in-loop delta whose ``finish_reason`` is non-null, the outer
            # SSE wrapper (streaming.py:_handler_sse_with_extraction) would
            # return on that first done chunk and the ``[DONE]`` branch's
            # usage (captured below from include_usage) would never reach
            # the consumer.
            inline_usage: Usage | None = None
            inline_cache_n: int = 0
            kv_aug: dict | None = None
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    flush_content, flush_thinking = thinking_buf.flush()
                    merged = {
                        **(self._context_usage_payload(
                            inline_usage.prompt_tokens if inline_usage else 0,
                            cache_n=inline_cache_n,
                        ) or {}),
                        **(kv_aug or {}),
                    }
                    ctx_payload = merged or None
                    yield InternalStreamChunk(
                        done=True,
                        model=request.model,
                        content_delta=flush_content,
                        thinking_delta=flush_thinking,
                        usage=inline_usage,
                        augmentum=ctx_payload,
                    )
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    log.warning("invalid_sse_data", data=data_str[:200])
                    continue
                # Stream-options.include_usage final chunk: empty choices,
                # populated ``usage`` and (llama-server extension)
                # ``timings``. Replaces the legacy GET /slots fallback —
                # those metrics are per-request and accurate even under
                # multi-slot routing.
                choices_list = data.get("choices") or []
                if not choices_list and data.get("usage"):
                    usage_data = data["usage"]
                    timings = data.get("timings") or {}
                    prompt_tokens = int(usage_data.get("prompt_tokens", 0))
                    completion_tokens = int(usage_data.get("completion_tokens", 0))
                    cache_n_inline = int((timings or {}).get("cache_n", 0) or 0)
                    completion_details = usage_data.get("completion_tokens_details") or {}
                    inline_usage = Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=int(usage_data.get(
                            "total_tokens", prompt_tokens + completion_tokens,
                        )),
                        # Mirror llama-server's cache_n into the typed
                        # cache fields (same shape DeepSeek emits). When
                        # the upstream is actually a strict OAI-compat
                        # backend that landed on this path, fall back to
                        # the spec'd ``prompt_tokens_details.cached_tokens``.
                        cache_hit_tokens=cache_n_inline or int(
                            (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens")
                            or 0
                        ),
                        cache_miss_tokens=prompt_tokens if cache_n_inline else 0,
                        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
                        eval_duration_ms=float((timings or {}).get("predicted_ms") or 0.0),
                    )
                    if timings:
                        inline_cache_n = cache_n_inline
                        if self._manager:
                            self._log_performance(timings, t_start, t_first_token)
                            # evaluated_n must be the freshly-EVALUATED
                            # count only — OAI-compat usage.prompt_tokens
                            # is the FULL prompt (cache hits included);
                            # passing it here double-counted the cache in
                            # the audit's denominator and halved every
                            # hot turn's reported reuse (live 2026-07-18:
                            # real 96% logged as 49% "partial_reuse" with
                            # ~78k phantom wasted tokens). timings.prompt_n
                            # is evaluated-only; derive it when absent.
                            kv_aug = self._audit_kv_reuse(
                                request,
                                evaluated_n=int(
                                    timings.get("prompt_n")
                                    or max(0, prompt_tokens - cache_n_inline)
                                ),
                                cache_n=cache_n_inline,
                                endpoint="chat_stream",
                            )
                    saw_finish_reason = True  # usage chunk implies clean end
                    continue
                choice = choices_list[0] if choices_list else {}
                delta = choice.get("delta", {})
                if choice.get("finish_reason") is not None:
                    saw_finish_reason = True

                raw_content = delta.get("content", "")
                native_thinking = delta.get("reasoning_content", "")
                clean_content, thinking = thinking_buf.process(
                    raw_content, native_thinking
                )

                if t_first_token is None and raw_content:
                    t_first_token = time.monotonic()

                yield InternalStreamChunk(
                    content_delta=clean_content,
                    thinking_delta=thinking,
                    role=delta.get("role"),
                    model=data.get("model", request.model),
                    augmentum=(
                        {"tool_calls": delta["tool_calls"]}
                        if delta.get("tool_calls") else None
                    ),
                    done=False,
                )

            # Stream ended without ``[DONE]`` — emit a terminal chunk so
            # the outer wrapper sees ``chunk.done`` and closes the SSE
            # cleanly. If the include_usage chunk DID land before EOF,
            # use it; otherwise prompt_tokens=0 leaves augmentum empty
            # and the frontend falls back to the _StreamTimer estimate.
            #
            # Severity split (mirrors _stream_completion):
            #   - saw_finish_reason → upstream sent the per-choice
            #     terminator but the connection dropped before the
            #     [DONE] sentinel. Stream content is intact; debug only.
            #   - !saw_finish_reason → real truncation: model crashed or
            #     upstream socket reset mid-generation. Worth a warning.
            flush_content, flush_thinking = thinking_buf.flush()
            if saw_finish_reason:
                log.debug("chat_completions_stream_clean_eof")
            else:
                log.warning(
                    "chat_completions_stream_truncated",
                    detail="stream ended before finish_reason (model truncation or upstream reset)",
                )
            merged = {
                **(self._context_usage_payload(
                    inline_usage.prompt_tokens if inline_usage else 0,
                    cache_n=inline_cache_n,
                ) or {}),
                **(kv_aug or {}),
            }
            ctx_payload = merged or None
            yield InternalStreamChunk(
                done=True,
                model=request.model,
                content_delta=flush_content,
                thinking_delta=flush_thinking,
                usage=inline_usage,
                augmentum=ctx_payload,
            )

    # ------------------------------------------------------------------
    # Model listing / details
    # ------------------------------------------------------------------

    def pre_stream_validate(self, request: InternalChatRequest) -> None:
        """Pre-flight gate for llama-server-backed chat requests.

        Currently only screens for image-without-projector; future checks
        (context-length overflow, unsupported tool combos) compose here so
        they share one call site at the streaming entry point. See the
        base-class docstring for the why-this-exists rationale.
        """
        self._reject_if_images_without_vision(request)

    def is_local_engine(self) -> bool:
        """Always True — this backend IS the local llama-server whose
        ``--jinja`` template injects the bare reasoning opener into the
        prompt prefix. See ``ModelBackend.is_local_engine``.
        """
        return True

    def is_vision_paired(self, model: str = "") -> bool:
        """True iff the loaded primary has an mmproj projector paired.

        ``model`` is ignored — llama-server has a single loaded model, so
        the projector pairing IS the per-request answer.

        When False, the route-layer caption fallback rewrites image
        attachments to text via the SmolVLM sibling before the request
        reaches us, so :meth:`_reject_if_images_without_vision` only
        fires when no fallback is available (vision_router empty).
        """
        if self._manager is None:
            # Standalone llama-server — defer to the runtime check. The
            # reject path still fires if it can't actually serve vision.
            return True
        return bool(getattr(self._manager, "current_mmproj_path", ""))

    def _reject_if_images_without_vision(self, request: InternalChatRequest) -> None:
        """Refuse a request that attaches images to a text-only loaded model.

        Why:
          When the loaded llama-server has no ``--mmproj`` paired, its
          chat template still renders the image marker (``<__media_N__>``
          / similar) into the prompt — but with no projector to consume
          it, the marker becomes literal text. The model reads the
          placeholder and produces a "can't see this image" hallucination.

          We catch this earlier and surface a clear, actionable error so
          the user knows to either (a) pair a projector for this model
          via Model Manager > Pair vision, or (b) switch to a model
          that's already vision-paired.

        Raises:
          ValueError when the request carries any image attachment and
          the manager reports no current mmproj. Callers (route layer)
          convert to a 400.
        """

        has_images = any(getattr(m, "images", None) for m in request.messages)
        if not has_images:
            return
        if self._manager is None:
            # Standalone llama-server without an Augmentum manager — can't
            # introspect, defer to the runtime. (Operator configured this.)
            return
        if getattr(self._manager, "current_mmproj_path", "") :
            return
        raise ValueError(
            "This model has no vision projector (mmproj) paired, so it "
            "cannot read attached images. Pair an mmproj via "
            "Model Manager > Pair vision, or switch to a vision-capable "
            "model."
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available models.

        If connected to a managed server, returns discovered GGUF models
        (even if the server isn't running yet). Otherwise queries the
        server's /v1/models endpoint.

        Vision flag is set when the manager finds a sibling mmproj/CLIP
        projector via :meth:`LlamaServerManager._find_paired_mmproj`. The
        pairing logic combines projector_type metadata, base-model name
        claims, filename, and family fallback — see that method's
        docstring for the matching contract. We honor its result here so
        models with auto-paired projectors (e.g., Qwen3-VL family with a
        sibling ``mmproj.gguf``) surface as multimodal in /v1/models and
        downstream consumers (chat dropdown, game-agent frame attach).

        Managed-server path is cached for ``_MODELS_CACHE_TTL_S`` to
        absorb the polling pattern from /api/tags consumers (Open WebUI,
        Settings modal, Model Manager). The currently-loaded model is
        spliced in AFTER the cache lookup so live state always wins.
        """
        # Managed server: list discovered GGUFs so they appear in model map
        if self._manager is not None:
            models = await self._cached_managed_models()
            # Splice the live loaded model outside the cache so swaps
            # show immediately regardless of TTL.
            if self._manager.model_id:
                loaded = self._manager.model_id
                if not any(m.name == loaded for m in models):
                    loaded_vision = bool(
                        getattr(self._manager, "current_mmproj_path", "")
                    )
                    # Copy to avoid mutating the cached list.
                    models = [
                        ModelInfo(
                            name=loaded, model=loaded, size=0, modified_at="",
                            vision=loaded_vision,
                        ),
                        *models,
                    ]
            return models

        try:
            resp = await self._client.get(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            # Some llama-server builds carry vision capability in a
            # parallel ollama-style ``models`` array (``capabilities:
            # ["multimodal"]``) while the openai-style ``data`` entries
            # omit it entirely (only ``meta`` with token counts). Build a
            # name->vision map from ``models`` so the cross-reference can
            # rescue ``data`` entries that look text-only on their own.
            cap_vision: dict[str, bool] = {}
            for mm in data.get("models", []):
                if not isinstance(mm, dict):
                    continue
                name = mm.get("name") or mm.get("model") or ""
                if name:
                    cap_vision[name] = _v1_entry_is_vision(
                        {"id": name, "capabilities": mm.get("capabilities")}
                    )
            return [
                ModelInfo(
                    name=m.get("id", "unknown"),
                    model=m.get("id", "unknown"),
                    size=0,
                    modified_at="",
                    vision=cap_vision.get(m.get("id", ""), False)
                    or _v1_entry_is_vision(m),
                )
                for m in data.get("data", [])
            ]
        except httpx.HTTPError:
            if not self._list_models_warned:
                log.warning("llamacpp_list_models_failed", base_url=self._base_url, exc_info=True)
                self._list_models_warned = True
            else:
                log.debug("llamacpp_list_models_failed", base_url=self._base_url)
            return []

    def invalidate_models_cache(self) -> None:
        """Drop the cached managed-server model list.

        Called by ProviderRegistry.invalidate_model_map() when a backend's
        on-disk model set changes (download completes, file deleted, dir
        added). Without this hook the 15s TTL kept a stale list alive even
        after the registry's own model_map was invalidated — so a
        freshly-downloaded model was invisible to /api/tags consumers
        (chat dropdown, settings modal) for up to 15s.
        """
        self._models_cache = None

    async def _cached_managed_models(self) -> list[ModelInfo]:
        """Cache-fronted version of the managed-server discovery scan.

        Singleflight via ``_models_cache_lock``: concurrent callers
        arriving on a cache miss collapse to a single disk scan + GGUF
        metadata read pass instead of racing each other (pre-fix, all
        4-5 concurrent /api/tags hits would each do the full ~600ms
        walk). Lazy lock construction mirrors ``_ensure_lock`` —
        ``asyncio.Lock()`` historically required a running loop.
        """
        now = time.monotonic()
        cached = self._models_cache
        if cached is not None and (now - cached[0]) < self._MODELS_CACHE_TTL_S:
            return cached[1]
        if self._models_cache_lock is None:
            self._models_cache_lock = asyncio.Lock()
        async with self._models_cache_lock:
            # Re-check under the lock — another caller may have just
            # populated the cache while we were waiting.
            cached = self._models_cache
            now = time.monotonic()
            if cached is not None and (now - cached[0]) < self._MODELS_CACHE_TTL_S:
                return cached[1]
            # Discovery + per-model pairing involves multiple disk reads
            # (filesystem walk, sidecar JSONs, GGUF metadata). Push it
            # off the event loop so other awaitables (chat token
            # streaming, voice, etc.) aren't starved while we scan.
            models = await asyncio.to_thread(self._scan_managed_models)
            self._models_cache = (time.monotonic(), models)
            return models

    def _scan_managed_models(self) -> list[ModelInfo]:
        """Synchronous filesystem scan + per-model dim-checked mmproj
        pairing. Called from ``_cached_managed_models`` via
        ``asyncio.to_thread`` so the loop stays responsive.
        """
        from pathlib import Path

        assert self._manager is not None
        files = self._manager.discover_gguf_files()

        # Group same-stemmed GGUFs so collisions across overlapping
        # scan dirs surface as one entry — but we pick the winner
        # deliberately (MTP-capable copy preferred over a same-named
        # non-MTP older build) and log the shadowing instead of
        # silently dropping the loser. The previous first-seen-wins
        # rule could surface the wrong copy and gave operators no
        # signal that a duplicate existed.
        groups: dict[str, list[dict]] = {}
        for f in files:
            lower = Path(f["filename"]).stem.lower()
            # Skip vision projector files — not loadable standalone
            if lower.startswith("mmproj") or lower.startswith("clip-") or "-mmproj" in lower:
                continue
            groups.setdefault(lower, []).append(f)

        models: list[ModelInfo] = []
        for _stem, entries in groups.items():
            annotated = [
                (e, self._manager.profile_cache.get(e["path"]))
                for e in entries
            ]
            # Pick the first MTP-headed copy if any; else first-seen
            # (which inherits model_dirs ordering — the operator's
            # preferred dir wins by default).
            chosen_entry, chosen_profile = next(
                (
                    (e, p) for e, p in annotated
                    if p and p.has_mtp_heads
                ),
                annotated[0],
            )
            name = Path(chosen_entry["filename"]).stem
            # Discovery context — pass quiet=True so dim-mismatch
            # rejections downgrade to debug.
            paired_mmproj = self._manager._find_paired_mmproj(
                chosen_entry["path"], chosen_profile, quiet=True,
            )
            mtp_capable = bool(chosen_profile and chosen_profile.has_mtp_heads)

            shadowed_paths = [
                e["path"] for e, _ in annotated
                if e["path"] != chosen_entry["path"]
            ]
            details: dict | None = None
            if shadowed_paths:
                # Dedup by (name, chosen, shadowed-set). The collision
                # is a static fact of the filesystem layout — re-warning
                # on every poll adds zero new information.
                dedup_key = (
                    name, chosen_entry["path"], tuple(sorted(shadowed_paths)),
                )
                if dedup_key not in _collision_logged:
                    _collision_logged.add(dedup_key)
                    log.warning(
                        "llama_gguf_name_collision",
                        name=name,
                        chosen=chosen_entry["path"],
                        shadowed=shadowed_paths,
                        chosen_has_mtp=mtp_capable,
                    )
                details = {"shadowed_paths": shadowed_paths}

            models.append(ModelInfo(
                name=name,
                model=name,
                size=chosen_entry.get("size", 0),
                modified_at="",
                vision=bool(paired_mmproj),
                mtp=mtp_capable,
                details=details,
            ))
        return models

    async def show_model(self, name: str) -> ModelDetails:
        """Get server properties (llama.cpp-specific)."""
        try:
            resp = await self._client.get(
                f"{self._base_url}/props",
                headers=self._headers(),
            )
            resp.raise_for_status()
            props = resp.json()
            return ModelDetails(
                format="gguf",
                family=props.get("default_generation_settings", {}).get("model", name),
                parameter_size="",
                quantization_level="",
                system_prompt=props.get("system_prompt", ""),
                template="",
            )
        except httpx.HTTPError:
            return ModelDetails(
                format="gguf", family=name, parameter_size="", quantization_level="",
            )

    async def get_context_length(self, model: str) -> int:
        """Read n_ctx from /props — the actual loaded context size."""
        try:
            props = await self.get_props()
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            if n_ctx and isinstance(n_ctx, (int, float)):
                return int(n_ctx)
        except Exception as exc:
            # Returns 0 when /props is unreachable; callers treat
            # 0 as "unknown" and fall back to model-profile cache.
            log.debug("llama_cpp_get_context_length_failed", model=model, error=str(exc))
        return 0

    # ------------------------------------------------------------------
    # llama.cpp-specific endpoints
    # ------------------------------------------------------------------

    async def get_props(self) -> dict:
        """Get server properties (model path, generation settings, slot count, etc.)."""
        try:
            resp = await self._client.get(
                f"{self._base_url}/props",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            log.warning("llamacpp_get_props_failed", exc_info=True)
            return {}

    async def get_slots(self) -> list[dict]:
        """Get llama.cpp slot/cache status."""
        try:
            resp = await self._client.get(
                f"{self._base_url}/slots",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return []

    async def save_slot(self, slot_id: int, session_id: str) -> bool:
        """Save a slot's KV cache state to disk.

        Persists the full KV state so a conversation can be resumed
        instantly without re-prefilling the entire context.
        Uses a derived, filesystem-safe name for easy lookup.

        Multimodal models: llama.cpp's slot save/restore is not supported
        when an mmproj is loaded (server returns "This feature is not
        supported by multimodal" and bumps log noise on every turn). We
        no-op cleanly in that case -- the chat itself is unaffected, only
        cross-request KV-cache persistence is lost.
        """
        if self._slot_io_unsupported:
            return False
        # Skip cleanly when the loaded model wasn't launched with slot save
        # available (multi-slot/--kv-unified models persist via ctx-checkpoints
        # instead — attempting the per-slot API would just 501). Default True
        # so an unknown/older manager still attempts (the 501 latch backstops).
        if self._manager is not None and not getattr(
            self._manager, "_slot_save_supported", True
        ):
            return False
        if self._manager is not None and getattr(self._manager, "current_mmproj_path", ""):
            return False
        try:
            filename = self._slot_storage_name(session_id)
            resp = await self._client.post(
                f"{self._base_url}/slots/{slot_id}?action=save",
                json={"filename": filename},
                headers=self._headers(),
                timeout=30.0,
            )
            if resp.status_code < 400:
                log.info("slot_saved", slot=slot_id, session=session_id)
                return True
            # 501 = server started without --slot-save-path; the feature is
            # unavailable for this server's lifetime. Latch + log once at info
            # so we stop hammering it (and the log) every turn.
            if resp.status_code == 501:
                self._slot_io_unsupported = True
                log.info("slot_io_unsupported_disabling", slot=slot_id)
                return False
            log.warning("slot_save_failed", slot=slot_id, status=resp.status_code)
            return False
        except Exception as exc:
            # repr() so message-less exceptions (httpx ConnectTimeout,
            # ReadError, etc.) still surface their type in the log.
            log.warning("slot_save_error", slot=slot_id, error=repr(exc))
            return False

    async def restore_slot(self, slot_id: int, session_id: str) -> bool:
        """Restore a slot's KV cache state from disk.

        Loads a previously saved KV state, allowing instant context
        resumption without re-processing the conversation history.

        Multimodal models: short-circuits like :meth:`save_slot` -- the
        upstream ``/slots/.../restore`` endpoint refuses with "This
        feature is not supported by multimodal" when mmproj is loaded.

        We always erase the slot first. ``action=restore`` in upstream
        llama.cpp doesn't wipe before loading — it expects free cells —
        so a previously-occupied slot 0 (from the just-completed chat or
        a prior checkpoint prewarm) will reject the restore with
        ``state_read_meta: failed to find available cells in kv cache``
        and a 400. The erase is a near-zero-cost in-memory operation;
        any state we'd be discarding was saved to disk on the previous
        step, so nothing is actually lost.
        """
        if self._slot_io_unsupported:
            return False
        if self._manager is not None and not getattr(
            self._manager, "_slot_save_supported", True
        ):
            return False
        if self._manager is not None and getattr(self._manager, "current_mmproj_path", ""):
            return False
        try:
            filename = self._slot_storage_name(session_id)
            try:
                await self._client.post(
                    f"{self._base_url}/slots/{slot_id}?action=erase",
                    headers=self._headers(),
                    timeout=10.0,
                )
            except Exception as exc:  # noqa: BLE001
                # Best-effort: a failed erase shouldn't block the restore
                # attempt. If the slot is genuinely full, the restore
                # will surface its own clear error.
                log.debug("slot_erase_before_restore_failed", slot=slot_id, error=repr(exc))
            resp = await self._client.post(
                f"{self._base_url}/slots/{slot_id}?action=restore",
                json={"filename": filename},
                headers=self._headers(),
                timeout=30.0,
            )
            if resp.status_code < 400:
                log.info("slot_restored", slot=slot_id, session=session_id)
                return True
            # 404/400 = no saved state for this session (normal for first use,
            # also fires on cross-turn digest drift). Logged at info so the
            # body surfaces the actual cause without needing global debug.
            if resp.status_code in (404, 400):
                body = (resp.text or "").strip()[:200]
                log.info(
                    "slot_restore_miss",
                    slot=slot_id,
                    session=session_id,
                    status=resp.status_code,
                    body=body,
                )
                return False
            if resp.status_code == 501:
                self._slot_io_unsupported = True
                log.info("slot_io_unsupported_disabling", slot=slot_id)
                return False
            log.warning("slot_restore_failed", slot=slot_id, status=resp.status_code)
            return False
        except Exception as exc:
            # repr() so message-less exceptions surface their type.
            log.warning("slot_restore_error", slot=slot_id, error=repr(exc))
            return False

    async def erase_slot(self, slot_id: int) -> bool:
        """Erase a slot's KV cache."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/slots/{slot_id}?action=erase",
                headers=self._headers(),
            )
            return resp.status_code < 400
        except Exception:
            return False

    async def save_session_state(
        self,
        session_id: str,
        *,
        slot_id: int = 0,
        request: InternalChatRequest | None = None,
    ) -> bool:
        """Save KV state for ``session_id`` from physical slot ``slot_id``.

        Called after each response to persist the conversation's KV
        cache to disk. On next message in this session,
        ``restore_session_state`` brings it back without re-prefilling
        thousands of tokens.

        Phase 1 keeps the default slot_id=0 so existing callers operate
        unchanged. Phase 2 callers pass the actual served slot id from
        the completion response so save targets the slot that produced
        the KV being saved.
        """
        saved = await self.save_slot(slot_id, session_id)
        if saved:
            await self._record_manifest_save(session_id, request=request)
        return saved

    async def restore_session_state(
        self,
        session_id: str,
        *,
        slot_id: int = 0,
        request: InternalChatRequest | None = None,
    ) -> bool:
        """Restore KV state for ``session_id`` into physical slot ``slot_id``.

        Called before processing a request when switching back to a
        previously-active session. If successful, llama-server skips
        prefill entirely and goes straight to generation.

        Phase 1 keeps the default slot_id=0. Phase 2+ callers pick the
        target slot based on occupancy and the slot-affinity strategy
        documented in the multi-slot spec.
        """
        manifest = self._kv_manifest()
        model_key = self._current_model_key()

        # Overlap the manifest compatibility check with the on-disk
        # slot-file existence check. The manifest call dispatches to a
        # worker thread (SQLite); _slot_state_exists is sync (os.listdir)
        # and runs on the main thread while the manifest task is queued.
        # Sequential, these are ~a few ms each; concurrent, the total
        # is max(ms_a, ms_b) rather than ms_a + ms_b. Tiny per-turn win
        # but free.
        manifest_task = None
        if manifest is not None and model_key:
            manifest_task = asyncio.create_task(
                manifest.get_session_async(model_key, session_id)
            )
        file_exists = self._slot_state_exists(session_id)

        if manifest_task is not None:
            record = await manifest_task
            if record:
                reason = self._restore_skip_reason(record)
                if reason:
                    await manifest.mark_restore_skip_async(model_key, session_id, reason)
                    log.info("session_kv_restore_skipped", session=session_id, reason=reason)
                    return False

        # No on-disk checkpoint for this fingerprint — don't call
        # restore_slot. ``restore_slot`` always erases the target slot
        # before attempting the load (mitigation for an upstream
        # "failed to find available cells" error), so a guaranteed-
        # miss restore would destroy whatever KV is currently in the
        # slot. That KV is exactly what ``cache_prompt: true`` walks
        # for prefix matching on the new prompt — destroying it forces
        # a full re-prefill. The common case this rescues: regenerate,
        # where the new request's prompt is a strict prefix of the
        # slot's existing tokens (the assistant turn being regenerated
        # has been dropped). Keeping the slot lets cache_prompt skip
        # prefill of the shared prefix entirely. Pre-fix, regenerate
        # was ~8x slower than continue on the same conversation.
        if not file_exists:
            log.info("session_kv_restore_no_checkpoint", session=session_id)
            return False

        restored = await self.restore_slot(slot_id, session_id)
        await self._record_manifest_touch(session_id, request=request, restored=restored)
        return restored

    async def tokenize(self, text: str) -> list[int]:
        """Tokenize text using llama.cpp's endpoint."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/tokenize",
                json={"content": text},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json().get("tokens", [])
        except httpx.HTTPError:
            return []

    async def detokenize(self, tokens: list[int]) -> str:
        """Convert tokens back to text using llama.cpp's detokenize endpoint."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/detokenize",
                json={"tokens": tokens},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json().get("content", "")
        except httpx.HTTPError:
            log.warning("llamacpp_detokenize_failed", exc_info=True)
            return ""

    async def embeddings(self, input: str | list[str], model: str = "") -> dict:
        """Generate embeddings via the OpenAI-compatible embeddings endpoint.

        Args:
            input: A string or list of strings to embed.
            model: Model identifier (optional, llama-server uses the loaded model).

        Returns:
            The raw response JSON, or an empty dict on failure.
        """
        payload: dict = {"input": input}
        if model:
            payload["model"] = model
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            log.warning("llamacpp_embeddings_failed", exc_info=True)
            return {}

    async def health(self) -> dict:
        """Check server health.

        Returns a dict with at least a ``status`` key (``ok``, ``loading model``,
        ``error``, or ``no slot available``).
        """
        try:
            resp = await self._client.get(
                f"{self._base_url}/health",
                headers=self._headers(),
            )
            # llama-server returns 200 for ok, 503 for loading, 500 for error
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                return {"status": "ok" if resp.status_code == 200 else "error"}
        except httpx.HTTPError:
            log.warning("llamacpp_health_failed", exc_info=True)
            return {"status": "unreachable"}

    async def apply_template(
        self,
        messages: list[dict],
        *,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        """Apply the server's chat template to messages.

        Args:
            messages: List of message dicts (role/content).

        Returns:
            The formatted prompt string, or empty string on failure.
        """
        normalized = self._merge_consecutive_same_role(
            self._ensure_user_first_after_system(
                self._normalize_system_messages(messages),
            ),
        )
        try:
            payload: dict = {"messages": normalized}
            if chat_template_kwargs:
                payload["chat_template_kwargs"] = chat_template_kwargs
            resp = await self._client.post(
                f"{self._base_url}/apply-template",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json().get("prompt", "")
        except httpx.HTTPError as exc:
            # Surface the message-role sequence on 4xx/5xx so the next
            # "System message must be at the beginning" Jinja error has a
            # concrete trail. Q3.6/Qwen3.5 templates raise that exception
            # whenever a system message lands past position 0; capturing
            # the role/content-len shape makes the normalization bug
            # easier to chase than a bare httpx traceback.
            resp_obj = getattr(exc, "response", None)
            status = getattr(resp_obj, "status_code", 0)
            body = (getattr(resp_obj, "text", "") or "")[:300] if resp_obj else ""
            role_shape = [
                {
                    "role": m.get("role"),
                    "len": len(
                        m.get("content") if isinstance(m.get("content"), str) else ""
                    ),
                }
                for m in normalized[:24]
            ]
            log.warning(
                "llamacpp_apply_template_failed",
                status=status,
                body=body,
                role_shape=role_shape,
                msg_count=len(normalized),
            )
            return ""

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    async def list_lora_adapters(self) -> list[dict]:
        """List currently loaded LoRA adapters.

        Returns a list of dicts with ``id`` and ``scale`` keys.
        """
        try:
            resp = await self._client.get(
                f"{self._base_url}/lora-adapters",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            log.warning("llamacpp_list_lora_failed", exc_info=True)
            return []

    async def set_lora_adapters(self, adapters: list[dict]) -> bool:
        """Set active LoRA adapters and their scales.

        Args:
            adapters: List of ``{"id": <int>, "scale": <float>}`` dicts.

        Returns:
            True if the server accepted the request.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/lora-adapters",
                json=adapters,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError:
            log.warning("llamacpp_set_lora_failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Router-mode model management
    # ------------------------------------------------------------------

    async def is_router_mode(self) -> bool:
        """Check if the server is running in router (multi-model) mode.

        Returns True if the ``/models`` endpoint returns model objects with
        a ``status`` field (router mode), as opposed to the standard
        OpenAI-style list (single-model mode).
        """
        try:
            resp = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                # Router mode returns a list of model objects with status
                if isinstance(data, list) and data and "status" in data[0]:
                    return True
            return False
        except httpx.HTTPError:
            return False

    async def list_router_models(self) -> list[dict]:
        """List all models in router mode with their load status.

        Each dict has at least ``model`` and ``status`` keys.
        Status values: ``unloaded``, ``loading``, ``loaded``, ``failed``.
        """
        try:
            resp = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return []
        except httpx.HTTPError:
            log.warning("llamacpp_list_router_models_failed", exc_info=True)
            return []

    async def load_model(self, name: str) -> bool:
        """Load a model in router mode.

        Args:
            name: The model name/path to load.

        Returns:
            True if the server accepted the load request.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/models/load",
                json={"model": name},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError:
            log.warning("llamacpp_load_model_failed", model=name, exc_info=True)
            return False

    async def unload_model(self, name: str) -> bool:
        """Unload a model in router mode.

        Args:
            name: The model name/path to unload.

        Returns:
            True if the server accepted the unload request.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/models/unload",
                json={"model": name},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError:
            log.warning("llamacpp_unload_model_failed", model=name, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vision_content(msg: Message) -> str | list[dict]:
        """Build content field, using vision array format when images present."""
        if not msg.images:
            return msg.content
        parts: list[dict] = []
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
        for img in msg.images:
            parts.append({"type": "image_url", "image_url": {"url": img}})
        return parts

    def _local_max_tokens_floor(self, *, requested: int) -> int:
        """Raise the per-response output cap to a healthy fraction of
        the loaded model's context window, for local backends only.

        The previous flat 8192 mode-hint default (see
        ``modes/inference_hints.py::_MODE_HINTS["coder"]``) is 14×
        smaller than the available output headroom on a 128K-ctx Qwen
        model. Reasoning models routinely consume 3-5K tokens on
        thinking before emitting a tool-call body, so a single
        moderately-large ``file_write`` JSON args block trips
        ``finish_reason="length"`` and the ``path`` field gets chopped
        off the tail. This helper ratchets the cap UP to
        ``ctx * coder_local_max_tokens_pct / 100`` (bounded by
        ``coder_local_max_tokens_cap``) so the budget reflects the
        model's actual room to write.

        NEVER lowers the requested value — purely a floor. Caller
        already filtered ``max_tokens is None``. Cloud backends do not
        call this (they have their own server-side defaults and some
        APIs reject "non-standard" values).

        IMPORTANT — threshold gate. Only applies the floor when the
        requested cap is itself a "large output" value (>= 4096
        tokens). Smaller asks signal deliberate per-step caps from
        modes that NEED tight bounds — analytical's 512-tok UARF
        phases, agentic's 1024-tok plan steps, reflexion's 220-tok
        self-critique. Bumping those to ctx*25% would break the
        contract those pipelines rely on (UARF phases get rambly,
        agentic plans bloat, reflexion stops being terse). The 4096
        threshold matches inference_hints.py's coder / passthrough /
        narrative cluster — modes where long output is a feature, not
        a bug.
        """
        if requested < 4096:
            return requested
        from augmentum.config import settings as _settings
        pct = int(getattr(_settings, "coder_local_max_tokens_pct", 0) or 0)
        if pct <= 0:
            return requested
        ctx_size = int(getattr(self._manager, "current_ctx_size", 0) or 0)
        if ctx_size <= 0:
            return requested
        floor = (ctx_size * pct) // 100
        abs_cap = int(getattr(_settings, "coder_local_max_tokens_cap", 0) or 0)
        if abs_cap > 0:
            floor = min(floor, abs_cap)
        return max(requested, floor)

    def _to_openai_payload(self, request: InternalChatRequest) -> dict:
        """Convert internal request to OpenAI-compatible payload.

        Maps standard InternalChatRequest fields directly, then merges any
        llama.cpp-specific sampling parameters from ``raw_options``.

        ``stream`` is deliberately omitted — each caller (``chat`` /
        ``_stream_chat_completions``) sets it explicitly so a streaming
        caller can never leak ``stream=True`` into a non-streaming call.
        """
        rewritten = LlamaCppBackend._request_messages_for_template(request)
        # Belt-and-braces: even when carrier rewrite ran for the
        # checkpoint path, force a final normalization pass so any
        # message that didn't get converted (e.g. structured content
        # whose first system block was non-string) can't slip a late
        # system role through to a strict Qwen/Qwen3.x template, which
        # would 500 with "System message must be at the beginning".
        # Late systems become in-place carriers (never front-merged),
        # so a fired guard can leave adjacent same-role users — re-run
        # the alternation fixes in that case.
        messages = LlamaCppBackend._normalize_system_messages(rewritten)
        # ``_normalize_system_messages`` returns the input list
        # unchanged when there were no late systems to merge — so an
        # identity mismatch means the guard actually fired and we just
        # papered over a missed case in either the carrier rewrite or
        # the engine-level message construction. Bubble that up as a
        # warning so it's visible in logs rather than silently fixed;
        # without this, the underlying late-system source stays
        # invisible and the workaround becomes permanent dependence.
        if messages is not rewritten:
            messages = LlamaCppBackend._merge_consecutive_same_role(
                LlamaCppBackend._ensure_user_first_after_system(messages),
            )
            log.warning(
                "oai_payload_late_system_normalized",
                original_count=len(rewritten),
                normalized_count=len(messages),
                kv_checkpoint=bool(request.kv_stable_messages),
                note=(
                    "Defensive normalize fired — carrier rewrite or "
                    "upstream message construction left a late system "
                    "message that would have 500'd a strict Qwen-class "
                    "template. Trace request.messages roles + the "
                    "_request_messages_for_template branch taken."
                ),
            )
        payload: dict = {
            "model": request.model,
            "messages": messages,
        }

        # --- Standard fields from InternalChatRequest ---
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = self._local_max_tokens_floor(
                requested=request.max_tokens,
            )
        if request.stop:
            payload["stop"] = request.stop
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        # JSON mode via format field
        if request.format == "json":
            payload["response_format"] = {"type": "json_object"}

        # --- llama.cpp-specific params from raw_options ---
        if request.raw_options:
            for key in _LLAMACPP_PASSTHROUGH_PARAMS:
                if key in request.raw_options:
                    # response_format from raw_options overrides the format-based one
                    payload[key] = request.raw_options[key]

        return payload
