"""Vision provider implementations.

See :mod:`augmentum.vision` for the high-level design. This file
ships three classes:

- :class:`VisionProvider` — abstract base. Two methods: ``caption``
  and ``is_available``. Implementations may add OCR/VQA later.

- :class:`SmolVLMSibling` — lifecycle wrapper around a
  :class:`LlamaServerManager` instance pointed at SmolVLM 256M. Lives
  on a different port from the primary engine (default 8092) so the
  two coexist freely. Default CPU-only (``gpu_layers=0``) per the
  always-on substrate philosophy — opt-in GPU for users with VRAM
  headroom.

- :class:`SmolVLMProvider` — wraps a :class:`SmolVLMSibling` and
  speaks the OpenAI vision Chat Completions shape against the
  sibling's ``base_url``. Reuses the proxy's http_client so
  request/response observability is uniform.

- :class:`PrimaryVisionProvider` — uses the currently-loaded primary
  model when it's VL-capable. Routes through the existing provider
  registry rather than reimplementing the chat call.

Fail-open philosophy: every method that can't complete returns a
sensible empty/None and logs at WARNING. The caller decides whether
to retry, fallback, or give up.
"""

from __future__ import annotations

import asyncio
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from augmentum.models.llama_server_manager import LlamaServerManager

log = get_logger(__name__)


# stb_image (what mtmd_helper uses) decodes these magic prefixes natively.
# Anything else gets a Pillow round-trip to PNG before being sent.
_STB_NATIVE_MAGICS = (
    b"\x89PNG\r\n\x1a\n",   # PNG
    b"\xff\xd8\xff",        # JPEG
    b"GIF87a", b"GIF89a",   # GIF
    b"BM",                  # BMP (2-byte; cheap collision risk but stb-accepted)
)


# DEDICATED live-caption sampling profile for ALL captioner models (SmolVLM
# sibling + Gemma classifier). This is deliberately ISOLATED from the
# classifier's other roles: the routing/verdict sampler lives in the env
# (AUGMENTUM_CLASSIFIER_SAMPLING_* → applied to voice/architect-router calls)
# and any future low-latency chat use of the small model would carry its own
# sampler — captioning never shares mutable sampler state with them, because
# llama.cpp applies sampling per-request. A captioner whose text is inlined
# into the live conversation AND can flow into memory must be as DETERMINISTIC
# as possible — at Gemma's model-card temp 1.0 it confabulated
# settings/brands/backstory that then poison downstream context.
#
# Tuning history (captioner consistency bake-off, 2026-06-18): run-to-run
# stability went 15% (temp 1.0) → 61% (temp 0.2 + top_p/top_k below). The
# residual variance at temp 0.2 was the RANDOM per-request seed (llama.cpp
# default -1) — so we PIN ``seed`` here: identical frames now yield identical
# captions (the strongest no-poison guarantee), while temp 0.2 keeps a hair of
# escape from greedy's repetition traps. Any key added here is honoured by
# ``_caption_via_openai_endpoint`` below, so this dict is the single, isolated
# place to personalise live captioning.
_CAPTION_SAMPLING: dict[str, float | int] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 0,
}


def _ensure_stb_decodable(data: bytes) -> bytes | None:
    """Return bytes mtmd_helper/stb_image can decode.

    Pass-through when the magic prefix is already a stb-native format
    (PNG/JPEG/GIF/BMP). Otherwise transcode through Pillow to PNG.
    Returns None on unrecoverable decode failure.
    """
    if any(data.startswith(m) for m in _STB_NATIVE_MAGICS):
        return data
    try:
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


async def _caption_via_openai_endpoint(
    http: httpx.AsyncClient,
    base_url: str,
    image_bytes: bytes,
    *,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    model: str = "vision",
    extra_frames: list[bytes] | None = None,
    sampling: dict[str, float | int] | None = None,
    enable_thinking: bool | None = None,
    out_meta: dict | None = None,
) -> str:
    """POST image(s)+prompt to an OpenAI-compatible vision endpoint.

    Shared by every llama-server-backed vision provider — the SmolVLM
    sibling and the multimodal classifier sidecar both speak the same
    ``/v1/chat/completions`` vision shape, so the transcode + request +
    parse logic lives here once instead of per provider. ``base_url`` may
    or may not already carry a ``/v1`` suffix (the sibling exposes a bare
    root; the classifier base_url from env ends in ``/v1``) — it's
    normalised either way. Returns '' on any failure (caller falls back).

    ``extra_frames`` (live-camera path): additional frames sent in the
    SAME message as a sequence, so a video-capable model (Gemma 4) reasons
    over motion/continuity instead of N independent stills. The primary
    ``image_bytes`` is frame 0.

    ``sampling`` overrides temperature/top_p/top_k (Gemma wants its own
    distribution — see ``_GEMMA_VISION_SAMPLING``); default is steady
    temp 0.2. ``enable_thinking`` forwards a ``chat_template_kwargs``
    thinking flag (only when not None) — captioning is an instruct,
    high-repetition role so callers pass ``False`` to keep latency low;
    omitted (None) for templates that don't branch on it (SmolVLM).

    ``out_meta``, when given, is filled with ``finish_reason``,
    ``completion_tokens`` and ``reasoning_chars``. It exists because this
    function reports every failure as ``''`` — fine for best-effort captioning,
    but indistinguishable from a legitimately empty result for callers where
    the difference is load-bearing. A reasoning model that burns its whole
    ``max_tokens`` budget thinking returns ``finish_reason='length'`` with NO
    content; without this, an OCR caller records that truncation as "this page
    has no text". Optional so the existing ''-on-failure contract is unchanged
    for every caller that doesn't care.
    """
    # mtmd_helper in llama.cpp uses stb_image, which only decodes
    # PNG/JPEG/BMP/TGA/PSD/GIF/HDR/PIC/PNM. WebP/AVIF/HEIC raise
    # 400 "Failed to load image". Transcode to PNG so any Pillow-readable
    # format works. Each frame is transcoded independently.
    frames_in = [image_bytes, *(extra_frames or [])]
    image_parts: list[dict] = []
    for fb in frames_in:
        decoded = await asyncio.to_thread(_ensure_stb_decodable, fb)
        if decoded is None:
            continue
        b64 = base64.b64encode(decoded).decode("ascii")
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    if not image_parts:
        log.warning("vision_caption_transcode_failed", bytes=len(image_bytes))
        return ""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    try:
        payload: dict = {
            "model": model,   # llama-server is permissive on model name
            "messages": [
                {
                    "role": "user",
                    "content": [*image_parts, {"type": "text", "text": prompt}],
                },
            ],
            "max_tokens": max_tokens,
        }
        if sampling:
            # Forward the whole caption profile — temp/top_p/top_k shape the
            # distribution, min_p trims the tail, and ``seed`` pins the RNG so
            # identical frames yield identical captions (no-poison determinism).
            for key in ("temperature", "top_p", "top_k", "min_p", "seed"):
                if sampling.get(key) is not None:
                    payload[key] = sampling[key]
        else:
            payload["temperature"] = 0.2   # captions want to be steady
        # Instruct vs thinking — forwarded to the model's --jinja template.
        # Only set when the caller opted in (Gemma branches on it; SmolVLM's
        # template doesn't, so we omit the kwarg there rather than ship a
        # no-op key a strict template might reject).
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        resp = await http.post(
            f"{root}/v1/chat/completions", json=payload, timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if out_meta is not None:
            out_meta["finish_reason"] = choices[0].get("finish_reason") or ""
            out_meta["completion_tokens"] = int(
                (data.get("usage") or {}).get("completion_tokens") or 0,
            )
            out_meta["reasoning_chars"] = len(msg.get("reasoning_content") or "")
        return str(content).strip()
    except Exception as exc:
        log.warning(
            "vision_caption_endpoint_failed", base_url=root, error=str(exc)[:200],
        )
        return ""


# ── Abstract base ────────────────────────────────────────────────────


class VisionProvider(ABC):
    """Common interface for any source of vision inference.

    Implementations must be safe to call concurrently from multiple
    coroutines — the captioner pipeline fan-outs assume this.
    """

    @abstractmethod
    async def is_available(self) -> bool:
        """True iff this provider can serve a caption right now.

        Should be cheap — called per-request to decide routing. Network
        calls are acceptable as long as they're short-timeout-bounded.
        """

    @abstractmethod
    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt: str = "Describe this image in one short sentence.",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        frames: list[bytes] | None = None,
    ) -> str:
        """Return a short text description of the image.

        ``frames`` are additional frames (live-camera sequence) understood
        as one clip by video-capable providers; single-image providers may
        ignore them. Returns empty string on any failure. Implementations
        should log at WARNING (not ERROR) — caption failure is recoverable;
        the caller may fall back to another provider or skip enrichment.
        """


# ── SmolVLM sibling subprocess ───────────────────────────────────────


@dataclass(slots=True)
class SmolVLMConfig:
    """Configuration for the SmolVLM sibling subprocess."""

    base_model_path: str = ""        # /models/vision/SmolVLM-256M-Instruct-Q8_0.gguf
    mmproj_path: str = ""            # /models/vision/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
    backend_port: int = 8092         # different from primary (8091)
    gpu_layers: int = 0              # CPU-only by default
    ctx_size: int = 8192             # captions don't need much
    batch_size: int = 256
    llama_server_path: str = "/usr/local/bin/llama-server"


class SmolVLMSibling:
    """Lifecycle wrapper for the SmolVLM-serving llama-server instance.

    Owns one :class:`LlamaServerManager`. Distinct from the primary
    engine's manager — they run on different ports, manage different
    subprocesses, and never compete for the same model slot.

    Lifecycle is independent of the primary engine: this sibling can
    be running while the primary is unloaded, and vice versa.
    """

    def __init__(self, config: SmolVLMConfig) -> None:
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
        """Direct access to the underlying manager. Used by
        :class:`SmolVLMProvider` for ``is_busy`` checks and graceful
        shutdown. May be None before :meth:`start` succeeds."""
        return self._manager

    async def start(self) -> bool:
        """Bring up the sibling subprocess. Idempotent — returns True
        if the sibling is running (whether or not we started it this
        call). Returns False if the model files are missing or the
        process refuses to start.
        """
        cfg = self.config
        if not cfg.base_model_path:
            log.info("smolvlm_sibling_disabled", reason="no model path configured")
            return False
        from pathlib import Path
        if not Path(cfg.base_model_path).is_file():
            log.warning(
                "smolvlm_sibling_missing_model",
                base=cfg.base_model_path,
            )
            return False
        if cfg.mmproj_path and not Path(cfg.mmproj_path).is_file():
            log.warning(
                "smolvlm_sibling_missing_mmproj",
                mmproj=cfg.mmproj_path,
            )
            return False

        async with self._start_lock:
            if self._manager is not None and self._manager.state.name == "READY":
                return True

            # Lazy import — LlamaServerManager pulls in a lot of state
            # (KV manifest, token cache) we don't need for vision.
            from augmentum.models.llama_server_manager import LlamaServerManager

            # Validate the projector and write the operator sidecar so
            # status/diagnostic surfaces see a paired projector. The
            # sidecar alone isn't enough to make ``LlamaServerManager``
            # actually pass ``--mmproj`` — its auto-pair logic only runs
            # when ``engine_auto_pair_mmproj`` is True (default False).
            # We pass the projector explicitly via ``load_options``
            # below so the sibling boots vision-capable regardless.
            mmproj_load_path = ""
            if cfg.mmproj_path:
                from augmentum.models.llama_server_manager import (
                    validate_mmproj_pair,
                    write_projector_sidecar,
                )
                ok, reason = validate_mmproj_pair(cfg.base_model_path, cfg.mmproj_path)
                if not ok:
                    log.warning(
                        "smolvlm_sibling_pair_rejected",
                        reason=reason,
                    )
                    return False
                write_projector_sidecar(cfg.base_model_path, cfg.mmproj_path)
                mmproj_load_path = cfg.mmproj_path

            self._manager = LlamaServerManager(
                llama_server_path=cfg.llama_server_path,
                backend_port=cfg.backend_port,
                model_dir=str(Path(cfg.base_model_path).parent),
                gpu_layers=cfg.gpu_layers,
                ctx_size=cfg.ctx_size,
                batch_size=cfg.batch_size,
                kv_warm_on_start=False,
                # Captions are sync, single-image requests — the multi-
                # slot warm tier never helps and otherwise budgets ~16 GiB
                # of host RAM (auto-sized from system total) for a 256M
                # auxiliary that never evicts a slot. Pin single-slot so
                # the sibling drops --kv-unified / --cache-ram /
                # --cache-idle-slots / --ctx-checkpoints from its CLI.
                force_single_slot=True,
            )
            # Vision sibling stays loaded; never auto-unload. Caption
            # bursts are exactly the workload that suffers most from
            # cold-start latency.
            self._manager.idle_timeout = 0.0

            load_options: dict[str, object] = {}
            if mmproj_load_path:
                # Explicit projector pairing wins over the global
                # auto-pair toggle — this is the load that actually
                # benefits from vision (SmolVLM is useless without it).
                load_options["mmproj_path"] = mmproj_load_path

            try:
                await self._manager.start(
                    cfg.base_model_path,
                    load_options=load_options or None,
                )
            except Exception as exc:
                log.warning("smolvlm_sibling_start_failed", error=str(exc)[:200])
                self._manager = None
                return False

            log.info(
                "smolvlm_sibling_started",
                port=cfg.backend_port,
                gpu_layers=cfg.gpu_layers,
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
            log.info("smolvlm_sibling_stopped")

    async def is_ready(self) -> bool:
        """True iff the sibling subprocess is up and accepting
        requests. Cheap check — no HTTP round-trip."""
        if self._manager is None:
            return False
        # ProcessState.READY is the post-warmup state where /v1/* is
        # accepting requests. Any other state means we'd block or fail.
        return self._manager.state.name == "READY"

    def can_serve(self) -> bool:
        """True iff the model files exist so the sibling COULD lazily start.
        Cheap (no subprocess). Lets the provider advertise availability as a
        CPU fallback without keeping a process resident on GPU boxes — it
        only actually starts when the router selects it (no VL classifier)."""
        from pathlib import Path
        cfg = self.config
        if not cfg.base_model_path or not Path(cfg.base_model_path).is_file():
            return False
        return not (cfg.mmproj_path and not Path(cfg.mmproj_path).is_file())


# ── SmolVLM-backed provider ──────────────────────────────────────────


class SmolVLMProvider(VisionProvider):
    """CPU-only vision FALLBACK via the SmolVLM sibling subprocess.

    Retired from the default path (2026-06-19): the Gemma classifier slot is
    the primary captioner. This provider exists ONLY for no-GPU deployments
    whose classifier is text-only. It LAZILY starts the sibling on first real
    use — the router selects it only when neither the primary nor the
    classifier can serve vision, so on GPU boxes the subprocess never starts.
    Speaks OpenAI vision Chat Completions to the sibling's base URL.
    """

    def __init__(
        self,
        sibling: SmolVLMSibling,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._sibling = sibling
        self._http = http_client

    async def is_available(self) -> bool:
        # Ready, OR able to lazily start (model files present). Advertising
        # can-start lets the router pick this fallback without a resident
        # process; caption() cold-starts it on first selection.
        if await self._sibling.is_ready():
            return True
        return self._sibling.can_serve()

    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt: str = "Describe this image in one short sentence.",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        frames: list[bytes] | None = None,
    ) -> str:
        # Lazy cold-start: only reached when the router fell back to the CPU
        # tier (no VL classifier/primary). Idempotent; returns "" if it
        # can't start so the router's fallback chain continues. (start() is
        # short-circuited when already ready.)
        if not await self._sibling.is_ready() and not await self._sibling.start():
            return ""
        # SmolVLM-256M is a single-image model, not a video model — extra
        # frames would confuse it more than inform it, so we caption the
        # most-recent frame only (frame 0). Multi-frame clips are the
        # classifier (Gemma) provider's job.
        return await _caption_via_openai_endpoint(
            self._http,
            self._sibling.base_url,
            image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            model="smolvlm",
            # Same low-variance sampling as the Gemma path so the two
            # captioners behave consistently (bake-off 2026-06-18).
            sampling=_CAPTION_SAMPLING,
        )


# ── Classifier-sidecar-backed provider ───────────────────────────────


class ClassifierVisionProvider(VisionProvider):
    """Vision via the classifier sidecar, when it's a natively-multimodal
    model (e.g. Gemma 4 E2B/E4B).

    The classifier is already a small, GPU-resident llama-server doing the
    voice/architect routing verdicts — reusing it for frame captioning and
    video understanding avoids standing up a second SmolVLM subprocess, and
    matches Gemma's own guidance that small VL models at a low visual-token
    budget are ideal for captioning / many-frame video. It plays the same
    "dedicated small VL that keeps the primary KV cache clean" role the
    SmolVLM sibling does, so the router treats it as a sibling-class
    provider.

    REQUIRES the slot's model to be VL and launched WITH its mmproj projector.
    A text-only classifier can't read images; ``capability_fn`` (wired to the
    managed Slot C's live ``is_vision_capable()``) gates availability so a
    text-only slot doesn't claim the captioner role — the router then falls
    back to the CPU sibling. ``caption`` also fails soft (empty → fallback)
    as a backstop.
    """

    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        *,
        model: str = "classifier",
        capability_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._base_url = (base_url or "").strip()
        self._http = http_client
        self._model = model
        # Live vision-capability probe. True only when the classifier is
        # ACTUALLY serving vision now (Slot C model is VL+mmproj) — reflects
        # async loads and runtime model swaps. None (external Docker
        # classifier we can't introspect) → assume capable; a text-only one
        # returns empty captions and the router's fallback chain handles it.
        self._capability_fn = capability_fn

    async def is_available(self) -> bool:
        if not self._base_url:
            return False
        if self._capability_fn is not None:
            try:
                return bool(self._capability_fn())
            except Exception:  # noqa: BLE001 — never break routing on a probe
                return True
        return True

    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt: str = "Describe this image in one short sentence.",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        frames: list[bytes] | None = None,
    ) -> str:
        if not self._base_url:
            return ""
        # Gemma's intended distribution + instruct mode. Captioning is the
        # high-repetition role, so thinking is OFF here for low latency
        # (the user-response reasoning happens on the primary brain). Extra
        # frames ride along as one clip for native video understanding.
        return await _caption_via_openai_endpoint(
            self._http,
            self._base_url,
            image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            model=self._model,
            extra_frames=frames,
            sampling=_CAPTION_SAMPLING,
            enable_thinking=False,
        )


# ── Primary-model-backed provider ────────────────────────────────────


class PrimaryVisionProvider(VisionProvider):
    """Vision via the currently-loaded primary model, when VL-capable.

    The primary engine already pairs base + mmproj when the model has
    a sibling projector file or operator-declared sidecar. We piggyback
    on that — no new infra needed. If the primary slot is text-only,
    :meth:`is_available` returns False and the router falls back to
    SmolVLM.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    async def is_available(self) -> bool:
        if self._app_state is None:
            return False
        mgr = getattr(self._app_state, "llama_manager", None)
        if mgr is None:
            return False
        if mgr.state.name != "READY":
            return False
        # ``current_mmproj_path`` is set by ``_find_paired_mmproj`` when
        # the loaded base has a sibling projector. Empty string means
        # text-only.
        return bool(getattr(mgr, "current_mmproj_path", ""))

    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt: str = "Describe this image in one short sentence.",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        frames: list[bytes] | None = None,
    ) -> str:
        if not await self.is_available():
            return ""
        # Single-image only on this path: it runs as a router FALLBACK
        # captioner, and the keep-KV-clean invariant means we don't push a
        # multi-frame clip through the primary slot. Direct VL-primary
        # reading of frames happens on the chat/voice request, not here.
        mgr = self._app_state.llama_manager
        try:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            http = getattr(self._app_state, "http_client", None)
            if http is None:
                return ""
            payload = {
                "model": mgr.model_id or "primary",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            url = f"{mgr.base_url}/v1/chat/completions"
            resp = await http.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            content = (choices[0].get("message") or {}).get("content") or ""
            return str(content).strip()
        except Exception as exc:
            log.warning("primary_vision_caption_failed", error=str(exc)[:200])
            return ""


__all__ = [
    "PrimaryVisionProvider",
    "SmolVLMConfig",
    "SmolVLMProvider",
    "SmolVLMSibling",
    "VisionProvider",
]
