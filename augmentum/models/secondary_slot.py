"""SecondarySlot — the UTILITY tier ("Slot B").

One slot per role tier: Slot A is the primary chat engine, Slot B is
utility, Slot C is the classifier. Slot B holds whatever GGUF the user
loads into it from the model manager and routes to it via an explicit
registry pin (see ``ProviderRegistry.pin_model``), so it doubles as a
second chat model you can pin per-conversation, LM Studio style.

Its role job: when ``role == "utility"`` and no explicit
``utility_model`` / per-feature override is set,
``resolve_model_for_role`` resolves ``engine_secondary_model`` here —
memory consolidation, compaction, reflection, chat titles, the
narrative distiller. Empty or disabled slot degrades to
``primary_chat_model``; it never blocks. Utility deliberately does NOT
share Slot C, whose classifier work runs on a ~2.5s voice/architect
budget that chunky summarisation would sit in front of.

Reuses the proven sibling pattern — own port, own ``LlamaServerManager``
subprocess, never competing with the primary engine for a process slot.
The only shared resource is the GPU/host memory itself; per-model load
config (gpu-layer cap, ctx, idle timeout) travels with the model via
``engine.last_load.<model_id>`` so two different models land their
resource footprints exactly where the user configured them.

Lifecycle:
- Constructed at boot when ``engine_secondary_enabled`` is True, with an
  IDLE manager (no model loaded — no memory cost until first load).
- ``load(model_path)`` starts/swaps the slot to a model and the caller
  pins it.
- ``unload()`` stops the subprocess (frees all of its VRAM/RAM) and the
  caller drops the pin.
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

# Registry key under which the slot's backend registers. The router pins
# the loaded model to this key and excludes it from catalog probing.
SECONDARY_BACKEND_KEY = "engine_secondary"


@dataclass
class SecondarySlotConfig:
    """Construction inputs for the secondary slot.

    ``model_dirs`` is shared with the primary engine so the same GGUFs
    are loadable into either slot. Per-model runtime config is NOT here —
    it comes from the per-model load-options store at load time.
    """

    backend_port: int = 8094
    model_dirs: list[str] = field(default_factory=list)
    llama_server_path: str = "/usr/local/bin/llama-server"


class SecondarySlot:
    """Lifecycle wrapper for the secondary resident llama-server.

    Owns one :class:`LlamaServerManager` and one :class:`LlamaCppBackend`.
    The manager is built once (cheap — no subprocess until a model is
    loaded); the backend is registered in the provider registry by the
    lifespan wiring so resolution can reach the slot via its pin.
    """

    def __init__(
        self,
        config: SecondarySlotConfig,
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

        Synchronous and cheap — it does NOT start a subprocess. The
        manager sits IDLE (zero VRAM/RAM) until :meth:`load`.
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
            # Two resident models means memory is the scarce resource. The
            # multi-slot warm tier budgets a large host-RAM pool per
            # process; for a second model that's wasteful, so the slot runs
            # lean (single slot) and leans on the per-model load options
            # the user set for gpu-layer cap / ctx.
            force_single_slot=True,
        )
        if self._token_cache is not None:
            self._manager.set_token_cache(self._token_cache)
        if self._settings_store is not None:
            self._manager.set_settings_store(self._settings_store)

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
        """The slot's backend. Built on demand so the lifespan can grab
        it for registration before any model is loaded."""
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
    ) -> str:
        """Start or swap the slot to ``model_path``.

        Returns the resulting ``model_id``. Raises (TimeoutError /
        RuntimeError) on load failure — the caller surfaces it; the slot
        does not silently no-op a failed load.
        """
        async with self._load_lock:
            self._build()
            assert self._manager is not None
            import os

            from augmentum.models.llama_server_manager import ProcessState

            # Config travels with the model: when the caller passes no
            # explicit options, replay the per-model defaults the user set
            # in the model manager (``engine.last_load.<model_id>``). start()
            # does this itself, but swap() does not — so we resolve here to
            # cover both, ensuring two different models in/out of Slot B each
            # land their own configured gpu-layer cap / ctx / idle timeout.
            if not load_options:
                stem = os.path.splitext(os.path.basename(model_path))[0]
                saved = await self._manager._load_saved_options(stem)
                if saved:
                    load_options = saved

            if self._manager.state == ProcessState.READY:
                await self._manager.swap(model_path, load_options=load_options)
            else:
                await self._manager.start(model_path, load_options=load_options)
            # Persist per-model options as the install-wide default so a
            # later lazy reload (idle evict, crash) replays the same shape.
            if load_options:
                await self._manager.persist_load_options(
                    self._manager.model_id, load_options,
                )
            return self._manager.model_id

    async def unload(self) -> None:
        """Stop the subprocess, freeing all of the slot's VRAM/RAM. Idempotent.

        Serialized against :meth:`load` via the same lock so an unload can't
        race a concurrent (possibly long) load and leave the manager in a
        half-started state.
        """
        async with self._load_lock:
            mgr = self._manager
            if mgr is None:
                return
            try:
                await mgr.stop()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort
                log.warning("secondary_slot_unload_error", error=str(exc)[:200])
            # Self-heal: if stop()'s SIGKILL window elapsed and the subprocess
            # survived (WSL2+CUDA D-state), manager bookkeeping reads IDLE while
            # the strand still holds VRAM. Reclaim now so "unload" actually frees
            # the GPU instead of leaving Slot B resident — mirrors the primary's
            # idle-evict self-heal.
            try:
                reclaim = getattr(mgr, "reconcile_stranded_subprocess", None)
                if callable(reclaim):
                    await reclaim()
            except Exception as exc:  # noqa: BLE001
                log.warning("secondary_slot_reconcile_error", error=str(exc)[:200])

    async def is_ready(self) -> bool:
        if self._manager is None:
            return False
        return self._manager.state.name == "READY"

    def status(self) -> dict[str, Any]:
        """Slot status. ``{enabled: True, loaded: False}`` when built but
        empty; the manager's full status (incl. actual_memory + gpu free)
        once a model is loaded."""
        if self._manager is None:
            return {"enabled": True, "loaded": False, "state": "idle"}
        st = self._manager.status()
        st["enabled"] = True
        st["loaded"] = bool(self._manager.model_id)
        return st
