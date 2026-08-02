"""ClassifierSlot — a managed, runtime-switchable resident llama-server
for the ``classifier`` / ``utility`` (and, when VL+mmproj, ``vision``)
roles ("Slot C").

This is the managed counterpart to the external ``compose.classifier.yaml``
container. It lets the user pick the small workhorse model (Gemma-4-E2B /
E4B / a bigger utility-class model) and **swap it from the UI without a
container recreate** — the slot owns its own ``LlamaServerManager``
subprocess, so a swap is a stop+start of that subprocess, not a Docker
operation.

Relationship to the other slots
-------------------------------
- **Slot B** (``secondary_slot.py``) holds an arbitrary user-chosen CHAT
  model, reached via an explicit registry PIN. This module mirrors its
  lifecycle (own port, own manager, lean single-slot, per-model load
  options, stranded-subprocess reconcile) almost verbatim.
- **Slot C** (this) serves a ROLE, so it registers under the backend key
  ``"classifier"`` that ``resolve_model_for_role("classifier"/"utility")``
  already consults — meaning role routing needs ZERO changes. It is
  RESIDENT (``idle_timeout = 0``) because the voice/architect routers run
  on a hard ~2.5s budget and a cold reload would blow it.
- It supersedes the start-only ``architect/classifier_sibling.py`` (which
  could not swap models and was registered only as a best-effort fallback).

Precedence: an EXTERNAL Docker classifier (``AUGMENTUM_CLASSIFIER_BASE_URL``)
already registers ``"classifier"`` at registry construction. This slot
registers only if that key is empty — the external sidecar wins, so
existing installs are untouched.

Vision: when the loaded model is VL and was launched WITH its mmproj
projector, :meth:`is_vision_capable` returns True and the vision substrate
routes captioning here (retiring the SmolVLM sibling on GPU boxes).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.llama_cpp import LlamaCppBackend
    from augmentum.models.llama_server_manager import LlamaServerManager
    from augmentum.models.token_count_cache import TokenCountCache
    from augmentum.state.settings_store import SettingsStore

log = get_logger(__name__)

# Backend key under which the slot registers. Deliberately the SAME key
# resolve_model_for_role checks for the classifier/utility roles, so
# registering here wires role routing for free.
CLASSIFIER_BACKEND_KEY = "classifier"


@dataclass
class ClassifierSlotConfig:
    """Construction inputs for the classifier slot.

    ``model_dirs`` is shared with the primary engine so the same GGUFs are
    loadable here. Per-model runtime config (gpu-layer cap, ctx) is NOT
    here — it travels with the model via load options at load time, exactly
    like Slot B.
    """

    backend_port: int = 8093
    model_dirs: list[str] = field(default_factory=list)
    llama_server_path: str = "/usr/local/bin/llama-server"


class ClassifierSlot:
    """Lifecycle wrapper for the managed classifier llama-server.

    Owns one :class:`LlamaServerManager` and one :class:`LlamaCppBackend`.
    The manager is built once (cheap — no subprocess until a model is
    loaded); the backend is registered under ``"classifier"`` by the
    lifespan wiring so role resolution reaches it.
    """

    def __init__(
        self,
        config: ClassifierSlotConfig,
        http_client: Any,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self._manager: LlamaServerManager | None = None
        self._backend: LlamaCppBackend | None = None
        self._token_cache: TokenCountCache | None = None
        self._settings_store: SettingsStore | None = None
        self._load_lock = asyncio.Lock()

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        """Construct the manager + backend if not already built.

        Synchronous and cheap — does NOT start a subprocess. The manager
        sits IDLE (zero VRAM/RAM) until :meth:`load`.
        """
        if self._manager is not None:
            return
        import os
        import shutil

        from augmentum.models.llama_cpp import LlamaCppBackend
        from augmentum.models.llama_server_manager import LlamaServerManager

        binary = self.config.llama_server_path
        if not os.path.isfile(binary):
            binary = shutil.which("llama-server") or binary

        dirs = [d for d in self.config.model_dirs if d]
        primary_dir = dirs[0] if dirs else "/data/models"

        self._manager = LlamaServerManager(
            llama_server_path=binary,
            backend_port=self.config.backend_port,
            model_dir=primary_dir,
            extra_model_dirs=dirs[1:] or None,
            kv_warm_on_start=False,
            # Classifier verdicts are short, sequential, one-prompt JSON —
            # the multi-slot warm tier never helps and budgets a large host
            # RAM pool. Run lean; gpu-layer cap / ctx travel per-model.
            force_single_slot=True,
        )
        # RESIDENT: the voice/architect routers have a ~2.5s budget; a cold
        # reload after idle eviction would blow it. Never auto-unload.
        self._manager.idle_timeout = 0.0

        if self._token_cache is not None:
            self._manager.set_token_cache(self._token_cache)
        if self._settings_store is not None:
            self._manager.set_settings_store(self._settings_store)

        # LlamaCppBackend (not the generic OpenAI backend) so chat-template
        # kwargs / enable_thinking forwarding is preserved for Gemma — the
        # same reason the external classifier registration uses it.
        self._backend = LlamaCppBackend(
            http_client=self._http_client,
            base_url=self._manager.base_url,
            server_manager=self._manager,
        )

    @property
    def manager(self) -> LlamaServerManager | None:
        return self._manager

    @property
    def backend(self) -> LlamaCppBackend | None:
        """The slot's backend. Built on demand so the lifespan can grab it
        for registration before any model is loaded."""
        if self._backend is None:
            self._build()
        return self._backend

    @property
    def base_url(self) -> str:
        return self._manager.base_url if self._manager else ""

    def set_token_cache(self, cache: TokenCountCache | None) -> None:
        self._token_cache = cache
        if self._manager is not None:
            self._manager.set_token_cache(cache)

    def set_settings_store(self, store: SettingsStore | None) -> None:
        self._settings_store = store
        if self._manager is not None:
            self._manager.set_settings_store(store)

    # -- operations -----------------------------------------------------

    def resolve_model_path(self, name_or_path: str) -> str | None:
        """Resolve a bare model name/stem to an absolute GGUF path."""
        self._build()
        assert self._manager is not None
        import os

        if os.path.isabs(name_or_path):
            return name_or_path
        return self._manager._resolve_model_path(name_or_path)

    async def load(
        self,
        model_path: str,
        *,
        load_options: dict[str, Any] | None = None,
        merge_saved: bool = False,
    ) -> str:
        """Start or swap the slot to ``model_path``. Returns the model_id.

        ``merge_saved=True`` treats the caller's ``load_options`` as
        DEFAULTS and lets the per-model saved options
        (``engine.last_load.<model_id>``) override them — the boot path
        uses this so a user's per-model ctx/mmproj/kv config survives a
        restart instead of being clobbered by the global
        ``classifier_slot_ctx_size``/``gpu_layers`` settings. Explicit
        (UI/API) loads keep caller-wins semantics.

        Raises (TimeoutError / RuntimeError) on load failure — the caller
        surfaces it; the slot does not silently no-op a failed load. During
        the swap window the manager is not READY, so a concurrent classifier
        resolve falls through to the next tier (primary) — never blocks.
        """
        async with self._load_lock:
            self._build()
            assert self._manager is not None
            import os

            from augmentum.models.llama_server_manager import ProcessState

            # Config travels with the model: replay per-model defaults the
            # user set (``engine.last_load.<model_id>``) when the caller
            # passes none. start() does this; swap() does not — resolve here
            # so both paths land the right gpu-layer cap / ctx / mmproj.
            stem = os.path.splitext(os.path.basename(model_path))[0]
            if not load_options:
                saved = await self._manager._load_saved_options(stem)
                if saved:
                    load_options = saved
            elif merge_saved:
                saved = await self._manager._load_saved_options(stem)
                if saved:
                    load_options = {**load_options, **saved}
                # Resident-slot invariant on the boot path: a saved profile
                # written by a primary-engine load of the same GGUF may carry
                # a nonzero idle_timeout — never let it evict Slot C.
                load_options["idle_timeout"] = 0

            if self._manager.state == ProcessState.READY:
                await self._manager.swap(model_path, load_options=load_options)
            else:
                await self._manager.start(model_path, load_options=load_options)
            if load_options:
                await self._manager.persist_load_options(
                    self._manager.model_id, load_options,
                )
            return self._manager.model_id

    async def unload(self) -> None:
        """Stop the subprocess, freeing the slot's VRAM/RAM. Idempotent.

        Serialized against :meth:`load` via the same lock. Reconciles a
        stranded subprocess (WSL2+CUDA D-state) so unload really frees the
        GPU — parity with Slot B / the primary engine.
        """
        async with self._load_lock:
            mgr = self._manager
            if mgr is None:
                return
            try:
                await mgr.stop()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort
                log.warning("classifier_slot_unload_error", error=str(exc)[:200])
            try:
                reclaim = getattr(mgr, "reconcile_stranded_subprocess", None)
                if callable(reclaim):
                    await reclaim()
            except Exception as exc:  # noqa: BLE001
                log.warning("classifier_slot_reconcile_error", error=str(exc)[:200])

    async def is_ready(self) -> bool:
        if self._manager is None:
            return False
        return self._manager.state.name == "READY"

    def is_vision_capable(self) -> bool:
        """True iff the slot is serving a VL model launched WITH its mmproj
        projector — i.e. it can caption images/frames.

        This is the single source of truth the vision substrate keys off:
        when True, the classifier IS the captioner and the SmolVLM CPU
        fallback must NOT run; when False (text-only model, or not loaded),
        the no-GPU fallback may activate. Reads the manager's
        ``current_mmproj_path`` — the same signal PrimaryVisionProvider uses.
        """
        mgr = self._manager
        if mgr is None or mgr.state.name != "READY":
            return False
        return bool(getattr(mgr, "current_mmproj_path", ""))

    def status(self) -> dict[str, Any]:
        """Slot status. ``{enabled: True, loaded: False}`` when built but
        empty; the manager's full status once a model is loaded, plus the
        vision-capability bit."""
        if self._manager is None:
            return {"enabled": True, "loaded": False, "state": "idle",
                    "vision_capable": False}
        st = self._manager.status()
        st["enabled"] = True
        st["loaded"] = bool(self._manager.model_id)
        st["vision_capable"] = self.is_vision_capable()
        return st
