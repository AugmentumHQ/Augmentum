"""Capability extractors: pure-read adapters from in-process state.

One extractor per capability kind. Each takes its data source as a
constructor dependency (so tests can supply fakes) and exposes a
single async ``collect()`` returning a list of capability dataclasses.

Performance budget: <10ms p99 each. None of these should do I/O,
shell out, or hit SQLite. Everything they need is already in RAM
from the existing resource ledger / managers / registries.

Extractors run on every fabric heartbeat tick (every 5s) plus
opportunistically on state-change events in higher phases. They are
*pure observation*: they never modify the source. A bug in any
extractor must not break the corresponding subsystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.fabric.capabilities import (
    CapabilityBase,
    CastRenderCapability,
    ImageGenerationCapability,
    KnowledgeSearchCapability,
    LLMInferenceCapability,
    STTTranscribeCapability,
    TTSSynthesizeCapability,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.knowledge.packs import PackManager
    from augmentum.models.llama_server_manager import LlamaServerManager
    from augmentum.models.provider_registry import ProviderRegistry

log = get_logger(__name__)


class LLMCapabilityExtractor:
    """Surfaces all LLM models the local node can serve.

    For the bundled engine, we advertise EVERY GGUF discovered on disk
    (via ``llama_manager.discover_models()``) — not just the one that
    happens to be loaded right now. The currently-loaded model is
    flagged ``loaded=True`` with full slot/device detail; the rest
    appear with ``loaded=False`` so a fabric routing director can
    decide whether to ask this node to load-and-serve. That mirrors
    the image extractor's "everything on disk" model — a peer caring
    about routing decisions should see the full inventory, not just
    whatever's resident this exact second.

    For non-engine backends (ollama on LAN, llama.cpp subprocess, vllm,
    sglang), we list whatever models the backend itself advertises
    via ``provider_registry.refresh_model_map``.

    **Cloud-backed providers (openai, anthropic, deepseek, cerebras,
    groq, mistral, gemini, etc.) are deliberately NOT advertised over
    fabric.** Sharing them would let a peer rent another peer's
    third-party API credentials — a compromised peer could exhaust the
    operator's API budget at the speed of the rate limiter. The local
    operator retains full access via their own UI; only the cross-peer
    surface is closed. Locality is determined by ``is_local_engine_url``
    (loopback / RFC1918 / known docker-compose hostnames).
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry | None = None,
        llama_manager: LlamaServerManager | None = None,
    ) -> None:
        self._registry = provider_registry
        self._llama_manager = llama_manager

    async def collect(self) -> list[CapabilityBase]:
        out: list[CapabilityBase] = []

        # Augmentum engine: enumerate ALL discovered GGUFs from disk,
        # flag the currently-loaded one with rich detail.
        if self._llama_manager is not None:
            out.extend(await self._engine_inventory_capabilities())

        # Other backends: list whatever models they expose.
        if self._registry is not None:
            try:
                model_map = await self._registry.refresh_model_map()
            except Exception:
                log.debug("fabric_llm_extractor_model_map_failed", exc_info=True)
                model_map = {}

            from augmentum.fabric.cost_table import lookup_cost
            from augmentum.models.openai_compat import is_local_engine_url

            for model_name, backend_key in model_map.items():
                # Skip the engine -- we already reported it above with
                # richer detail.
                if backend_key == "engine":
                    continue
                # Locality gate: cloud-backed providers (anthropic, openai,
                # gemini, deepseek, cerebras, groq, mistral, ...) are not
                # advertised over fabric. Letting a peer route to them
                # would burn the local operator's API key from someone
                # else's box — full rationale in the class docstring.
                backend = self._registry._backends.get(backend_key)
                base_url = getattr(backend, "_base_url", "") if backend else ""
                if not is_local_engine_url(base_url):
                    log.debug(
                        "fabric_llm_extractor_skipping_cloud_backend",
                        backend=backend_key, model=model_name,
                    )
                    continue
                # Local non-engine backends (ollama, llamacpp via subprocess,
                # vllm, sglang) get advertised. lookup_cost returns (0, 0)
                # for these — kept in the payload for forward-compat in case
                # we later support advertising paid-but-local inference too.
                in_cost, out_cost = lookup_cost(model_name)
                out.append(
                    LLMInferenceCapability(
                        backend=backend_key,
                        model_id=model_name,
                        loaded=True,
                        input_cost_per_token=in_cost,
                        output_cost_per_token=out_cost,
                    )
                )

        return out

    async def _engine_inventory_capabilities(self) -> list[CapabilityBase]:
        """Build one capability per GGUF on disk + flag the loaded one.

        ``discover_models()`` walks the model dir, enriches each entry
        with its cached profile (architecture, context_length, etc.).
        We emit one ``LLMInferenceCapability`` per discovered file.
        The currently-loaded model gets ``loaded=True`` with slot/GPU
        detail; the rest get ``loaded=False`` so a peer's routing
        director knows the node *could* serve them on request.

        Returns ``[]`` on any inventory failure rather than partial
        output — heartbeats are idempotent; a single failed cycle just
        means stale data on the receiver for ~5s.
        """
        try:
            inventory = await self._llama_manager.discover_models()
        except Exception:
            log.debug("fabric_llm_extractor_inventory_failed", exc_info=True)
            return []

        try:
            status = self._llama_manager.status() or {}
        except Exception:
            status = {}

        loaded_state = str(status.get("state", "") or "")
        loaded_model_id = status.get("model_id") or ""
        is_loaded_state = loaded_state in {"ready", "starting", "draining"}
        loaded_profile = status.get("profile") or {}
        loaded_gpu = status.get("gpu") or {}

        out: list[CapabilityBase] = []
        for entry in inventory:
            filename = str(entry.get("filename", "") or "")
            if not filename:
                continue
            # model_id contract: filename without the ``.gguf`` suffix.
            # That's also what the engine's ``status.model_id`` reports
            # for the currently-loaded model, so the equality check
            # below lights up the loaded row cleanly.
            model_id = filename
            if model_id.endswith(".gguf"):
                model_id = model_id[: -len(".gguf")]

            is_loaded = (
                is_loaded_state and bool(loaded_model_id)
                and model_id == loaded_model_id
            )

            family = str(entry.get("architecture", "") or "")
            ctx_max = int(entry.get("context_length", 0) or 0)
            # ``params_b`` is a rough proxy — total_size_bytes is the file
            # size, not the parameter count. For Q4/Q5 GGUFs the param
            # count is ~2x the byte count in billions, for FP16 it's ~0.5x.
            # Receivers using this for routing should treat it as an
            # order-of-magnitude hint, not an exact value.
            size_bytes = int(entry.get("total_size_bytes", 0) or 0)
            params_b = round(size_bytes / 1_000_000_000.0, 2) if size_bytes else 0.0

            cap_kwargs: dict[str, Any] = {
                "backend": "engine",
                "model_id": model_id,
                "model_family": family,
                "params_b": params_b,
                "ctx_max": ctx_max,
                "loaded": is_loaded,
            }
            if is_loaded:
                # Override ctx_max with the actual loaded-time setting (the
                # operator may have loaded with a smaller-than-profile context).
                cap_kwargs["ctx_max"] = int(status.get("ctx_size", 0) or ctx_max)
                cap_kwargs["free_slots"] = _safe_int(status.get("free_slots", 0))
                cap_kwargs["device"] = {
                    "gpu_name": str(loaded_gpu.get("name", "") or ""),
                    "vram_free_mb": _safe_int(loaded_gpu.get("vram_free_mib", 0)),
                    "vram_total_mb": _safe_int(loaded_gpu.get("vram_total_mib", 0)),
                }
                if loaded_profile.get("architecture") and not family:
                    cap_kwargs["model_family"] = str(loaded_profile["architecture"])
            out.append(LLMInferenceCapability(**cap_kwargs))
        return out


class ImageCapabilityExtractor:
    """Surfaces installed local image-generation models.

    Inventory source is the union of two authoritative lists:

    1. ``ModelManager.list_local_models()`` — disk scan of the user
       model_dir + the optional system_dir hook (currently empty —
       no models are pre-baked). This is the same source the
       ``/api/image/models`` dropdown reads, so what we advertise
       across fabric matches exactly what the operator sees and
       can pick.
    2. ``ImagePersistence.list_models()`` — SQLite ``image_models``
       table. Populated by the download path (huggingface / civitai)
       which also writes the row. Models registered here but no longer
       on disk are filtered out by the disk scan; models on disk but
       missing from SQLite (hand-dropped folders, future bundled
       models) are caught by the scan that would otherwise be lost.

    Pre-fix the extractor read only from ``ImagePersistence``.
    Hand-dropped models never get a SQLite row, so they were
    advertised by nobody — yet the dropdown showed them locally,
    leading to "fabric thinks local can't serve this; bounce to
    peer" misroutes.

    Cloud image providers (openai, stability, together, etc.) are
    skipped here for the same reason cloud LLM backends are — they
    represent the operator's API budget, not local compute, and
    sharing them lets a peer rent the operator's quota. See
    LLMCapabilityExtractor docstring.
    """

    def __init__(
        self,
        *,
        persistence: Any | None = None,
        pipeline_registry: Any | None = None,
        model_manager: Any | None = None,
    ) -> None:
        self._persistence = persistence
        self._registry = pipeline_registry
        self._model_manager = model_manager

    async def collect(self) -> list[CapabilityBase]:
        # Currently-loaded model name, if any, so we can mark exactly one
        # cap as loaded=True. Best-effort; if the registry isn't wired or
        # nothing is loaded we just emit everything as loaded=False.
        loaded_name = ""
        if self._registry is not None:
            try:
                loaded_name = str(getattr(self._registry, "current_model", "") or "")
            except Exception:
                loaded_name = ""

        # name -> family. Disk scan wins on collision since it's the
        # source of truth for "can we actually load this right now".
        merged: dict[str, str] = {}

        if self._persistence is not None:
            try:
                models = await self._persistence.list_models()
                for m in models:
                    name = str(getattr(m, "name", "") or "")
                    if not name:
                        continue
                    ptype = getattr(m, "pipeline_type", None)
                    # PipelineType is an enum; ``.value`` gives the
                    # canonical family string ("sd_1.5", "sdxl", ...).
                    family = str(getattr(ptype, "value", ptype) or "")
                    merged[name] = family
            except Exception:
                log.debug("fabric_image_extractor_list_failed", exc_info=True)

        if self._model_manager is not None:
            try:
                disk_models = self._model_manager.list_local_models()
                for m in disk_models:
                    name = str(m.get("name", "") or "")
                    if not name:
                        continue
                    ptype = m.get("pipeline_type", None)
                    family = str(getattr(ptype, "value", ptype) or "")
                    merged[name] = family  # disk scan wins on collision
            except Exception:
                log.debug("fabric_image_extractor_disk_failed", exc_info=True)

        out: list[CapabilityBase] = []
        for name, family in merged.items():
            out.append(
                ImageGenerationCapability(
                    backend="diffusers",
                    model_id=name,
                    family=family,
                    loaded=(name == loaded_name and bool(loaded_name)),
                )
            )
        return out


class KnowledgeSearchExtractor:
    """Surfaces installed knowledge packs.

    Uses the pack_manager.installed() listing, which already returns
    a normalised dict per pack (augpack + zim merged when both are
    present). One capability per installed pack.
    """

    def __init__(self, *, pack_manager: PackManager | None = None) -> None:
        self._packs = pack_manager

    async def collect(self) -> list[CapabilityBase]:
        if self._packs is None:
            return []

        out: list[CapabilityBase] = []
        try:
            installed = self._packs.installed()
        except Exception:
            log.debug("fabric_knowledge_extractor_installed_failed", exc_info=True)
            return []

        for pack in installed:
            pid = str(pack.get("pack_id", ""))
            if not pid:
                continue
            out.append(
                KnowledgeSearchCapability(
                    pack_id=pid,
                    pack_name=str(pack.get("name", "")),
                    chunk_count=_safe_int(pack.get("chunk_count", 0)),
                    embedding_dim=_safe_int(pack.get("embedding_dim", 0)),
                    active=bool(pack.get("active", True)),
                    pack_format=_derive_pack_format(pack),
                )
            )
        return out


class AudioCapabilityExtractor:
    """Surfaces TTS + STT engines this node can serve for fabric peers.

    Single extractor for both kinds because most code paths inspect them
    together (the voice WS picks a TTS + an STT in the same handshake).
    Each call emits one capability per engine, never per voice: voices
    travel inline on the TTSSynthesizeCapability so the receiver builds
    its voice→provider map from heartbeats with no extra round-trip.

    Sources:
      - bundled in-process engines (Kokoro, Pocket TTS, Moonshine) read
        via attribute inspection on their singleton modules — gated by
        the same settings flags that gate their warmup paths.
      - external HTTP providers read from ``audio_providers`` on the
        open aiosqlite connection. SELECT of a ~5-20 row table on the
        warm conn is sub-ms; the heartbeat cadence (5s) tolerates it
        comfortably. For external providers we ship ``default_voice``
        only — full per-voice fanout would require an HTTP fetch
        against the provider, which violates the no-I/O extractor
        budget. Operators routing to peer external providers use the
        explicit ``fabric:<node>::<voice>`` prefix.
    """

    def __init__(self, *, db_conn: aiosqlite.Connection | None = None) -> None:
        self._db_conn = db_conn

    async def collect(self) -> list[CapabilityBase]:
        out: list[CapabilityBase] = []
        out.extend(self._collect_builtin_tts())
        out.extend(self._collect_builtin_stt())
        if self._db_conn is not None:
            out.extend(await self._collect_external())
        return out

    def _collect_builtin_tts(self) -> list[CapabilityBase]:
        from augmentum.config import settings

        out: list[CapabilityBase] = []

        # Kokoro: in-process built-in xor external sidecar by URL. The
        # URL setting overrides built-in; we advertise whichever is
        # active so a peer sees the actual provider, not the absence
        # of one path.
        if settings.tts_kokoro_url:
            try:
                from augmentum.voice.kokoro_tts import VOICE_META as KOKORO_META
                voices = sorted(KOKORO_META.keys())
                languages = sorted({m.get("lang", "") for m in KOKORO_META.values() if m.get("lang")})
            except Exception:
                voices, languages = [], []
            out.append(
                TTSSynthesizeCapability(
                    engine="kokoro",
                    provider_id="kokoro-url",
                    provider_name="Kokoro (sidecar)",
                    default_voice="af_heart",
                    voices=voices,
                    languages=languages,
                    streaming=True,
                    in_process=False,
                )
            )
        elif settings.tts_kokoro_builtin:
            try:
                from augmentum.voice.kokoro_tts import VOICE_META as KOKORO_META
                voices = sorted(KOKORO_META.keys())
                languages = sorted({m.get("lang", "") for m in KOKORO_META.values() if m.get("lang")})
                out.append(
                    TTSSynthesizeCapability(
                        engine="kokoro",
                        provider_id="kokoro-builtin",
                        provider_name="Kokoro (built-in)",
                        default_voice="af_heart",
                        voices=voices,
                        languages=languages,
                        streaming=True,
                        in_process=True,
                    )
                )
            except Exception:
                log.debug("fabric_audio_extractor_kokoro_failed", exc_info=True)

        if settings.tts_pocket_builtin:
            try:
                from augmentum.voice.pocket_tts import (
                    _DEFAULT_VOICE as _POCKET_DEFAULT_VOICE,
                )
                from augmentum.voice.pocket_tts import (
                    _DEFAULT_VOICE_NAMES as _POCKET_DEFAULT_VOICE_NAMES,
                )
                # Language from the user's setting — Pocket supports 6
                # (english + french/german/italian/portuguese/spanish 24l).
                # Advertise just the active one; switching languages
                # requires a model reload, so peers see one at a time.
                _pocket_lang = (
                    settings.tts_pocket_language or "english"
                ).split("_", 1)[0]
                out.append(
                    TTSSynthesizeCapability(
                        engine="pockettts",
                        provider_id="pockettts-builtin",
                        provider_name="Pocket TTS (built-in)",
                        default_voice=_POCKET_DEFAULT_VOICE,
                        voices=list(_POCKET_DEFAULT_VOICE_NAMES),
                        languages=[_pocket_lang],
                        streaming=True,
                        in_process=True,
                    )
                )
            except Exception:
                log.debug("fabric_audio_extractor_pocket_failed", exc_info=True)

        # Other URL-mode TTS sidecars the project ships overlays for.
        # The receiver doesn't get per-voice fanout here (would need an
        # HTTP fetch against the sidecar, which violates the no-I/O
        # extractor budget). Empty voice list signals "ask the sidecar
        # directly via OpenAI-compat /v1/audio/voices when routing".
        for engine_key, setting_name, friendly in (
            ("chatterbox", "tts_chatterbox_url", "Chatterbox"),
            ("chatterbox-turbo", "tts_chatterbox_turbo_url", "Chatterbox Turbo"),
            ("qwen", "tts_qwen_url", "Qwen TTS"),
            ("fish", "tts_fish_url", "Fish Speech"),
        ):
            if getattr(settings, setting_name, ""):
                out.append(
                    TTSSynthesizeCapability(
                        engine=engine_key,
                        provider_id=f"{engine_key}-url",
                        provider_name=f"{friendly} (sidecar)",
                        default_voice="",
                        voices=[],
                        languages=[],
                        streaming=True,
                        in_process=False,
                    )
                )

        return out

    def _collect_builtin_stt(self) -> list[CapabilityBase]:
        from augmentum.config import settings

        out: list[CapabilityBase] = []

        if settings.voice_moonshine_enabled:
            out.append(
                STTTranscribeCapability(
                    engine="moonshine",
                    provider_id="moonshine-builtin",
                    provider_name="Moonshine (built-in)",
                    default_model=settings.voice_moonshine_model or "",
                    languages=["en"],
                    streaming=True,
                    in_process=True,
                )
            )

        return out

    async def _collect_external(self) -> list[CapabilityBase]:
        out: list[CapabilityBase] = []
        try:
            # Explicit close prevents cursor accumulation — this runs
            # every fabric heartbeat (~5s) and aiosqlite's cursor isn't
            # released until either ``close()`` or GC. Wrapping in async
            # with — closes deterministically on every path including
            # exception, so the underlying sqlite statement handle is
            # returned promptly.
            async with self._db_conn.execute(
                "SELECT id, provider_type, name, default_model, default_voice, base_url "
                "FROM audio_providers WHERE is_enabled = 1 AND base_url != 'builtin'"
            ) as cursor:
                rows = await cursor.fetchall()
        except Exception:
            log.warning("fabric_audio_extractor_db_query_failed", exc_info=True)
            return out

        for row in rows:
            pid, ptype, name, default_model, default_voice = row[0], row[1], row[2], row[3] or "", row[4] or ""
            # Belt-and-suspenders: the bundled engines (Kokoro/Pocket/Moonshine)
            # are seeded as audio_providers rows with base_url='builtin' so they
            # show in the UI list, but _collect_builtin_* already advertises them
            # in-process with full voice metadata. Skipping them here avoids a
            # duplicate capability with the SAME provider_id but impoverished
            # (1-voice) metadata that would clobber the rich in-process entry on
            # the receiver's provider map.
            if (row[5] or "") == "builtin":
                continue
            if ptype == "tts":
                out.append(
                    TTSSynthesizeCapability(
                        engine=_infer_external_engine(pid),
                        provider_id=pid,
                        provider_name=name or pid,
                        default_model=default_model,
                        default_voice=default_voice,
                        voices=[default_voice] if default_voice else [],
                        languages=[],
                        streaming=True,
                        in_process=False,
                    )
                )
            elif ptype == "stt":
                out.append(
                    STTTranscribeCapability(
                        engine=_infer_external_engine(pid),
                        provider_id=pid,
                        provider_name=name or pid,
                        default_model=default_model,
                        languages=[],
                        streaming=False,
                        in_process=False,
                    )
                )
        return out


class CastRenderCapabilityExtractor:
    """Reports this node's cast-render capability.

    Always emits exactly one capability (this node IS the rendering
    surface for itself). Single-machine deployments produce a lite-tier
    cap with just CPU info — no UX overhead, no missing-dep noise.

    Hardware detection runs once on first ``collect()`` and is cached.
    Subsequent ticks are sub-microsecond returns of the cached cap. We
    don't redetect on every heartbeat because hardware doesn't change at
    runtime; live load tracking belongs in a separate channel
    (coordinator-side, future work).

    Detection sources, all optional:
      - CPU threads: ``os.cpu_count()`` — always works
      - NVIDIA GPU: pynvml if importable; nothing if not
      - Headless browser: presence of the ``playwright`` package

    None of these are required for the extractor to function. Missing
    dependencies are silent — the capability simply has fewer flags
    set, which the routing director reads correctly as "this node
    can't do that, look elsewhere or stay local."
    """

    def __init__(self) -> None:
        self._cap: CastRenderCapability | None = None

    async def collect(self) -> list[CapabilityBase]:
        if self._cap is None:
            self._cap = self._detect_once()
        return [self._cap]

    def _detect_once(self) -> CastRenderCapability:
        import os

        cpu_threads = os.cpu_count() or 0

        gpu_vendor, gpu_model, gpu_vram_gb, hw_encoder = _detect_gpu()
        max_streams = _max_concurrent_streams_for(gpu_vendor, gpu_vram_gb)

        has_browser = _detect_headless_browser()

        can_render_html = has_browser
        can_render_vrm = bool(gpu_vendor) and has_browser
        can_encode_video = bool(hw_encoder)
        can_stream_webrtc = False  # gated until aiortc lands in deps

        tier = _classify_tier(
            gpu_vendor=gpu_vendor,
            gpu_vram_gb=gpu_vram_gb,
            cpu_threads=cpu_threads,
            has_browser=has_browser,
        )

        return CastRenderCapability(
            tier=tier,
            cpu_threads=cpu_threads,
            gpu_vendor=gpu_vendor,
            gpu_model=gpu_model,
            gpu_vram_gb=gpu_vram_gb,
            hw_encoder=hw_encoder,
            max_concurrent_streams=max_streams,
            can_render_html=can_render_html,
            can_render_vrm=can_render_vrm,
            can_encode_video=can_encode_video,
            can_stream_webrtc=can_stream_webrtc,
        )


def _detect_gpu() -> tuple[str, str, float, str]:
    """Return (vendor, model, vram_gb, hw_encoder). All "" / 0.0 on miss.

    Tries NVIDIA via pynvml. AMD / Intel detection is forward-reserved
    for a follow-up; today their hardware encoder routing is a no-op so
    we don't surface a cap when we'd misroute jobs at them.
    """
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return ("", "", 0.0, "")

    try:
        pynvml.nvmlInit()
    except Exception:
        return ("", "", 0.0, "")

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return ("", "", 0.0, "")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode("utf-8", "replace") if isinstance(name_raw, bytes) else str(name_raw)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = round(mem.total / (1024 ** 3), 1)
        # NVENC has been standard on every NVIDIA GPU since Kepler (2012).
        # Anything modern enough to run augmentum has it.
        return ("nvidia", name, vram_gb, "nvenc")
    except Exception:
        return ("", "", 0.0, "")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:
            log.debug("fabric_gpu_pynvml_shutdown_failed", error=str(exc))


def _detect_headless_browser() -> bool:
    """True iff a Chromium/Chrome binary is reachable for headless render.

    Augmentum already owns a CDP client wrapper (BrowserVerifier in
    augmentum/tools/application_cdp.py); the only question is whether
    a binary is present to drive. We reuse find_chromium() — same
    discovery the App Builder verify path uses — so the diagnostic
    surface and the actual renderer share one source of truth.

    The find_chromium() call walks PATH + platform-specific install
    paths; cheap (no subprocess) and matches the extractor's no-I/O
    budget.
    """
    try:
        from augmentum.tools.application_cdp import find_chromium
        return find_chromium() is not None
    except Exception:
        return False


def _max_concurrent_streams_for(gpu_vendor: str, vram_gb: float) -> int:
    """Heuristic max concurrent rendered streams this node can sustain.

    Conservative — better to under-promise and have the director route
    extra jobs elsewhere than to overcommit and stutter all streams. The
    director compares this against its current active-stream count.
    """
    if gpu_vendor == "nvidia":
        # Stock NVENC session limit is ~3 on consumer cards. VRAM is the
        # secondary constraint for the WebGL renderer.
        if vram_gb >= 16:
            return 3
        if vram_gb >= 8:
            return 2
        return 1
    return 1  # CPU-only path: one stream tops


def _classify_tier(
    *,
    gpu_vendor: str,
    gpu_vram_gb: float,
    cpu_threads: int,
    has_browser: bool,
) -> str:
    """Map hardware signals to a coarse routing tier.

    heavy   = NVIDIA GPU with ≥12 GB VRAM + headless browser
    standard = any GPU + headless browser, OR strong CPU + headless browser
    lite    = everything else (includes single-board / NUC-class boxes)

    The director uses tier as a coarse first-pass filter. It doesn't
    decide routing alone — fine-grained checks against can_* flags
    happen after tier-matching.
    """
    if gpu_vendor == "nvidia" and gpu_vram_gb >= 12 and has_browser:
        return "heavy"
    if (gpu_vendor or cpu_threads >= 8) and has_browser:
        return "standard"
    return "lite"


def _infer_external_engine(provider_id: str) -> str:
    """Best-effort mapping from a provider_id slug to a canonical engine
    label. Used only for UI grouping ("which engine family is this");
    routing keys on provider_id, not engine, so misclassification is
    cosmetic.
    """
    pid = provider_id.lower()
    for needle, label in (
        ("kokoro", "kokoro"),
        ("pocket", "pockettts"),
        ("chatterbox", "chatterbox"),
        ("qwen", "qwen"),
        ("fish", "fish"),
        ("csm", "csm"),
        ("sesame", "csm"),
        ("deepgram", "deepgram"),
        ("whisper", "whisper"),
        ("moonshine", "moonshine"),
        ("speaches", "whisper"),
    ):
        if needle in pid:
            return label
    return "other"


# ── Internal helpers ──────────────────────────────────────────────


def _safe_int(value: Any) -> int:
    """Coerce to int, defaulting to 0 on anything unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _getattr_or_key(obj: Any, key: str, default: Any = "") -> Any:
    """Read ``key`` from either an attribute or a dict-style item.

    Tolerates the inconsistency between dataclass-style model
    descriptors and raw dicts -- some Augmentum subsystems return
    one, some return the other, and the extractor shouldn't care.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_coroutinefunction(fn: Any) -> bool:
    """asyncio.iscoroutinefunction without forcing an import on hot path."""
    import asyncio

    return asyncio.iscoroutinefunction(fn)


def _derive_pack_format(pack: dict[str, Any]) -> str:
    """Infer the on-disk format from the metadata dict.

    Packs ship as ``.augpack`` (sqlite-vec + FTS5) OR ``.zim`` (kiwix)
    OR both ("augpack+zim" when a small pack keeps the source .zim
    alongside the converted .augpack for in-app browseability).
    """
    has_chunks = bool(pack.get("chunk_count", 0))
    has_zim = bool(pack.get("main_entry_path") or pack.get("zim_main_path"))
    if has_chunks and has_zim:
        return "augpack+zim"
    if has_zim:
        return "zim"
    return "augpack"
