"""Classifier sibling — a second llama-server subprocess hosting a
small text model dedicated to fast classification and utility tasks.

Mirrors the SmolVLM-vision-sibling pattern (``augmentum/vision/
provider.py``) but for text. Owns its own ``LlamaServerManager``
on a different port from the primary engine (default 8093) so the
two coexist freely. The sibling is normally a 1-3B class model
suitable for sub-second JSON / one-word verdict responses; the
primary engine carries whatever heavy reasoning model the user
has selected for chat.

Why a sibling instead of multi-slot on the primary
--------------------------------------------------
``resolve_model_for_role("classifier"/"utility")`` already falls
back to the user's ``primary_chat_model`` when no dedicated model
is configured. The problem with that fallback is structural: when
the user switches chat models to something heavy (DeepSeek-V4,
Qwen3.6-35B), every voice classifier call inherits the same
latency budget and starts timing out. Concurrent slots on the
primary don't help — the model and template are the same; only
the conversation slot differs.

A dedicated small-model sibling sidesteps this entirely: the
voice router resolves to a backend that's always-ready, always
fast (~200-500ms for the JSON shape), and never competes with
chat for GPU/CPU/memory budget.

Lifecycle
---------
The lifespan hook in ``augmentum/proxy/server.py`` starts the
sibling when EITHER:

  * ``classifier_engine_enabled`` is True (manual master switch
    for users who want it for utility tasks regardless of voice
    mode), OR
  * ``companion_activation_mode == "always_listening"`` (the only
    voice mode that genuinely benefits from the latency floor —
    PTT and wake-word fire intentionally so a 2s classifier hop
    on a heavy model is acceptable).

Stops on lifespan teardown. Mode changes during runtime require
a server restart today; making the lifecycle truly reactive is a
follow-on (would need a settings-change subscriber).

Resource cost
-------------
~700 MB host RAM for a 1.5B Q4_K_M model on CPU. Negligible idle
CPU (llama-server idles waiting for requests). VRAM: zero by
default (``classifier_engine_gpu_layers = 0``). Voice classifier
fires <10 times/minute on typical traffic, so CPU is plenty.

Failure handling
----------------
The sibling is fail-open: if the model file is missing, the port
is in use, or the subprocess refuses to start, the sibling logs a
warning and stays down. ``resolve_model_for_role`` falls through
to the next tier (utility_model -> primary_chat_model -> default
backend) so voice classification still works, just slower.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.llama_server_manager import LlamaServerManager

log = get_logger(__name__)


@dataclass(slots=True)
class ClassifierConfig:
    """Configuration for the classifier sibling subprocess.

    Empty ``model_path`` means "sibling disabled" — :meth:`start`
    returns False without touching the subprocess.
    """

    model_path: str = ""                  # path to small text GGUF (1-3B)
    backend_port: int = 8093              # primary uses 8091, vision 8092
    gpu_layers: int = 0                   # CPU-only by default
    ctx_size: int = 4096                  # classifier prompts are short
    batch_size: int = 512
    llama_server_path: str = "/usr/local/bin/llama-server"


class ClassifierSibling:
    """Lifecycle wrapper for the classifier-serving llama-server instance.

    Owns one :class:`LlamaServerManager`. Distinct from the primary
    engine's manager and from the vision sibling — three independent
    subprocesses on three ports, three model slots, never competing.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        self._manager: LlamaServerManager | None = None
        self._start_lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """Empty string when not started; OpenAI-compatible base URL
        otherwise. Callers should check :meth:`is_ready` first."""
        if self._manager is None:
            return ""
        return self._manager.base_url

    @property
    def manager(self) -> LlamaServerManager | None:
        """Direct access to the underlying manager. May be None
        before :meth:`start` succeeds."""
        return self._manager

    async def start(self) -> bool:
        """Bring up the sibling subprocess. Idempotent — returns True
        if the sibling is running. Returns False if the model file is
        missing or the process refuses to start. Logs the reason in
        either case so a quiet start failure is debuggable.
        """
        cfg = self.config
        if not cfg.model_path:
            log.info("classifier_sibling_disabled", reason="no model path configured")
            return False
        from pathlib import Path
        if not Path(cfg.model_path).is_file():
            log.warning(
                "classifier_sibling_missing_model",
                path=cfg.model_path,
            )
            return False

        async with self._start_lock:
            if self._manager is not None and self._manager.state.name == "READY":
                return True

            # Fall back to PATH if the configured llama-server binary
            # isn't present at the dataclass default (e.g. Apple Silicon
            # Homebrew lands in /opt/homebrew/bin, not /usr/local/bin).
            import os
            import shutil

            from augmentum.models.llama_server_manager import LlamaServerManager
            _binary = cfg.llama_server_path
            if not os.path.isfile(_binary):
                _binary = shutil.which("llama-server") or _binary

            self._manager = LlamaServerManager(
                llama_server_path=_binary,
                backend_port=cfg.backend_port,
                model_dir=str(Path(cfg.model_path).parent),
                gpu_layers=cfg.gpu_layers,
                ctx_size=cfg.ctx_size,
                batch_size=cfg.batch_size,
                kv_warm_on_start=False,
                # Classifier calls are short, sequential, one-prompt JSON
                # verdicts — multi-slot warm tier never helps. Pin single
                # slot so the sibling drops the heavy KV / cache flags
                # from its CLI and the small process footprint stays small.
                force_single_slot=True,
            )
            # Stay loaded once started — the whole point of a sibling is
            # consistent low latency. A cold reload after idle eviction
            # would defeat the architecture.
            self._manager.idle_timeout = 0.0

            try:
                await self._manager.start(cfg.model_path)
            except Exception as exc:  # noqa: BLE001 — fail-open
                log.warning(
                    "classifier_sibling_start_failed",
                    error=str(exc)[:200],
                )
                self._manager = None
                return False

            log.info(
                "classifier_sibling_started",
                port=cfg.backend_port,
                gpu_layers=cfg.gpu_layers,
                model=Path(cfg.model_path).stem,
            )
            return True

    async def stop(self) -> None:
        """Gracefully stop the sibling. Idempotent."""
        mgr = self._manager
        if mgr is None:
            return
        try:
            await mgr.stop()
        finally:
            self._manager = None
            log.info("classifier_sibling_stopped")

    async def is_ready(self) -> bool:
        """True iff the sibling subprocess is up and accepting requests.
        Cheap check — no HTTP round-trip."""
        if self._manager is None:
            return False
        return self._manager.state.name == "READY"
