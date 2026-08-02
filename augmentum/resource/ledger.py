"""Resource ledger — unified view of GPU/RAM usage across all subsystems."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import aiosqlite

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.model_manager import ModelManager
    from augmentum.models.provider_registry import ProviderRegistry

log = get_logger(__name__)


# SQL constants — shared by the atomic persist path and the legacy
# helpers (kept for direct test access). Defined once so the schema
# changes in only one place.
_UPSERT_PROFILES_SQL = """
    INSERT INTO resource_profiles
        (model_name, subsystem, backend, vram_mb, ram_mb, device,
         quantization, parameter_size, family, pipeline_type,
         times_seen, first_seen, last_seen)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'), datetime('now'))
    ON CONFLICT(model_name, backend) DO UPDATE SET
        vram_mb = CASE WHEN excluded.vram_mb > 0 THEN excluded.vram_mb
                       ELSE resource_profiles.vram_mb END,
        ram_mb = CASE WHEN excluded.ram_mb > 0 THEN excluded.ram_mb
                      ELSE resource_profiles.ram_mb END,
        device = CASE WHEN excluded.device != '' AND excluded.device != 'unknown'
                      THEN excluded.device ELSE resource_profiles.device END,
        quantization = CASE WHEN excluded.quantization != ''
                            THEN excluded.quantization ELSE resource_profiles.quantization END,
        parameter_size = CASE WHEN excluded.parameter_size != ''
                              THEN excluded.parameter_size ELSE resource_profiles.parameter_size END,
        family = CASE WHEN excluded.family != ''
                      THEN excluded.family ELSE resource_profiles.family END,
        pipeline_type = CASE WHEN excluded.pipeline_type != ''
                             THEN excluded.pipeline_type ELSE resource_profiles.pipeline_type END,
        times_seen = resource_profiles.times_seen + 1,
        last_seen = datetime('now')
"""

_INSERT_SNAPSHOT_SQL = """
    INSERT INTO resource_snapshots
        (timestamp, gpu_total_mb, gpu_used_mb, gpu_free_mb,
         ram_total_mb, ram_used_mb, ram_free_mb,
         loaded_model_count, loaded_models_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class TrackedModel:
    """A model currently loaded somewhere."""
    name: str
    subsystem: str  # "llm" | "image" | "tts" | "stt"
    backend: str    # "ollama" | "llamacpp" | "diffusers" | provider key
    device: str     # "gpu" | "cpu" | "gpu+cpu" | "remote" | "unknown"
    vram_mb: int = 0
    ram_mb: int = 0
    quantization: str = ""
    parameter_size: str = ""
    family: str = ""
    pipeline_type: str = ""
    expires_at: str = ""
    active: bool = True
    status: str = "ready"
    pid: int = 0
    # How the memory figures were obtained (spec §4.6): "measured" (read from
    # the process / device), "declared" (a known model-card / config constant),
    # "estimated" (formula or residual split). Drives the UI confidence chip.
    confidence: str = "measured"


@dataclass
class ModelProfile:
    """Accumulated knowledge about a model's resource needs from past observations."""
    model_name: str
    subsystem: str
    backend: str
    vram_mb: int = 0
    ram_mb: int = 0
    device: str = ""
    quantization: str = ""
    parameter_size: str = ""
    family: str = ""
    pipeline_type: str = ""
    times_seen: int = 0
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class GpuProcess:
    """A process using the GPU, from nvidia-smi."""
    pid: int
    name: str           # process executable name
    vram_mb: int
    label: str = ""     # friendly label (e.g. "LM Studio", "Ollama")


@dataclass
class DiskDestination:
    """One filesystem destination where models can land.

    Each download target (engine_model_dir, llamacpp_model_dir,
    image_model_dir, knowledge pack dir, etc.) may live on a different
    bind mount — possibly a different volume entirely. We probe each
    distinct path separately so the UI can show "this 30GB GGUF won't
    fit on the LLM volume" without conflating it with disk free on
    the image volume.
    """
    dir: str
    modality: str          # 'llm' | 'image' | 'knowledge' | 'voice' | 'other'
    free_bytes: int
    total_bytes: int
    error: str = ""


@dataclass
class JobStatus:
    """A background job currently in flight.

    Sourced from JobsStore's in-memory ``_active`` registry — pure
    RAM read, no DB query on the snapshot collect path. Job state
    transitions write to the DB via the existing JobsStore code
    paths; we just surface what's already known.
    """
    job_id: str
    user_id: str
    kind: str              # 'gguf_download' | 'image_model_pull' | ...
    target_id: str         # model name being downloaded
    progress_pct: float
    stage: str             # human-readable status from the handler
    started_at: int        # unix seconds


@dataclass
class InventoryEntry:
    """A model on disk that COULD be loaded — superset of TrackedModel.

    TrackedModel describes what's loaded right now; InventoryEntry
    describes everything installed (loaded or not). The intersection
    is marked ``loaded=True``; entries on disk but not currently
    loaded are ``loaded=False``.

    ``capable`` is the key cross-peer field — the 8GB peer marks the
    13B GGUF entry ``capable=False`` so a cross-peer UI can grey out
    "load on peer-laptop" while still showing the model is installed.
    For local rendering ``capable`` is computed against this node's
    GPU; for fabric publishing it's still computed against the
    publishing peer's GPU.
    """
    name: str
    modality: str          # 'llm' | 'image' | 'tts' | 'stt' | 'embedder' | 'reranker' | 'knowledge'
    backend: str           # 'gguf' | 'ollama' | 'engine' | 'diffusers' | 'kokoro' | 'speaches' | ...
    size_bytes: int
    location: str          # filesystem path or registry id
    loaded: bool
    capable: bool
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResourceSnapshot:
    """Complete resource state at a moment in time."""
    timestamp: datetime
    gpu_name: str = ""
    gpu_total_mb: int = 0
    gpu_used_mb: int = 0
    gpu_free_mb: int = 0
    ram_total_mb: int = 0
    ram_used_mb: int = 0
    ram_free_mb: int = 0
    models: list[TrackedModel] = field(default_factory=list)
    gpu_processes: list[GpuProcess] = field(default_factory=list)
    unattributed_vram_mb: int = 0
    # Sprint A additions — transient, never persisted to SQLite.
    disk_destinations: list[DiskDestination] = field(default_factory=list)
    active_jobs: list[JobStatus] = field(default_factory=list)
    inventory: list[InventoryEntry] = field(default_factory=list)
    # Etag derived from sorted_dir_mtimes + per-modality action clock.
    # Stable string used by UI clients (and Sprint B's fabric peer
    # transport) to decide whether to re-render the inventory list.
    inventory_etag: str = ""


def _probe_gpu() -> tuple[str, int, int, int]:
    """Return (gpu_name, total_mb, used_mb, free_mb)."""
    # Prefer nvidia-smi for host/global GPU usage. torch.cuda can be accurate
    # for the current process, but the resource panel wants the whole device,
    # including separate llama-server subprocesses.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 4:
                return parts[0].strip(), int(parts[1]), int(parts[2]), int(parts[3])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        log.debug("gpu_probe_nvidia_smi_failed", error=str(exc))

    # Fall back to torch.cuda when nvidia-smi isn't available.
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = props.name
            total = getattr(props, "total_memory", 0) // (1024 * 1024)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free = free_bytes // (1024 * 1024)
            return name, total, total - free, free
    except (ImportError, RuntimeError) as exc:
        log.debug("gpu_probe_torch_failed", error=str(exc))

    return "", 0, 0, 0


def _probe_ram() -> tuple[int, int, int]:
    """Return (total_mb, used_mb, free_mb).

    Container-aware via ``hostmem``: under a cgroup limit this reports the
    limit and our working set within it, not the host's numbers. ``used``
    excludes reclaimable page cache (``usage - inactive_file``), so a box
    holding 17 GB of droppable cache is not reported as under pressure.
    """
    try:
        from augmentum.resource import hostmem

        info = hostmem.memory_info()
        return (info.total_mib, info.used_mib, info.available_mib)
    except Exception:
        return 0, 0, 0


def _get_torch_allocated_mb() -> int:
    """Return MB of VRAM currently allocated by PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except (ImportError, RuntimeError) as exc:
        log.debug("torch_allocated_mb_probe_failed", error=str(exc))
    return 0


# Known GPU process names → friendly labels
_PROCESS_LABELS: dict[str, str] = {
    # LM Studio
    "lms.exe": "LM Studio",
    "lm studio.exe": "LM Studio",
    "lm-studio": "LM Studio",
    "lms": "LM Studio",
    # Ollama
    "ollama_llama_server": "Ollama",
    "ollama_llama_server.exe": "Ollama",
    "ollama-runner": "Ollama",
    # llama.cpp
    "llama-server": "llama.cpp",
    "llama-server.exe": "llama.cpp",
    "server": "llama.cpp",  # generic name in some builds
    # vLLM / TGI / Aphrodite
    "vllm": "vLLM",
    "text-generation-launcher": "TGI",
    "aphrodite-engine": "Aphrodite",
    # koboldcpp
    "koboldcpp": "KoboldCpp",
    "koboldcpp.exe": "KoboldCpp",
    # ComfyUI / A1111 / Forge
    "comfyui": "ComfyUI",
    "webui.py": "Stable Diffusion WebUI",
}


def _probe_gpu_processes() -> list[GpuProcess]:
    """Query nvidia-smi for per-process GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []

        import os
        my_pid = os.getpid()
        processes: list[GpuProcess] = []

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                proc_name = parts[1]
                vram_mb = int(parts[2])
            except (ValueError, IndexError):
                continue

            # Derive a friendly label from the process name
            base = proc_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
            label = _PROCESS_LABELS.get(base, "")

            # Tag our own process
            if pid == my_pid:
                label = "Augmentum"

            processes.append(GpuProcess(
                pid=pid, name=proc_name, vram_mb=vram_mb, label=label,
            ))

        return processes
    except Exception:
        return []


async def _probe_lmstudio_models(registry) -> list[TrackedModel]:
    """Probe LM Studio for LOADED models only via native /api/v0/models.

    Only returns models where state != 'not-loaded'.  Skips cloud APIs
    and non-LM-Studio OpenAI-compat backends entirely — those don't
    consume local resources and clutter the resource panel.
    """
    from augmentum.models.openai_compat import OpenAIBackend

    models: list[TrackedModel] = []
    seen_urls: set[str] = set()

    for key, backend in registry._backends.items():
        if not isinstance(backend, OpenAIBackend):
            continue
        url = backend._base_url
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Only probe local servers — skip cloud APIs entirely
        is_local = any(h in url for h in (
            "localhost", "127.0.0.1", "0.0.0.0", "[::1]",
            "192.168.", "10.", "172.16.",
        ))
        if not is_local:
            continue

        # Only probe LM Studio (has /api/v0/models with state field)
        url_lower = url.lower()
        if not ("lmstudio" in url_lower or ":1234" in url_lower):
            continue

        try:
            # Strip /v1 suffix to get the base origin for native API
            base = url
            if base.endswith("/v1"):
                base = base[:-3]
            elif base.endswith("/v1/"):
                base = base[:-4]

            resp = await backend._client.get(
                f"{base}/api/v0/models", timeout=3,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            for m in data.get("data", []):
                model_id = m.get("id", "")
                state = m.get("state", "not-loaded")
                if not model_id or state == "not-loaded":
                    continue

                models.append(TrackedModel(
                    name=model_id,
                    subsystem="llm",
                    backend="LM Studio",
                    device="gpu",
                    quantization=m.get("quantization", ""),
                    family=m.get("arch", ""),
                ))
        except Exception:
            log.debug("probe_lmstudio_failed", url=url, exc_info=True)

    return models


def _probe_inprocess_models() -> list[TrackedModel]:
    """Detect in-process ML models (ONNX, FastEmbed, etc.).

    These live inside the augmentum process, so their individual footprint
    isn't separable from the main-process RSS via psutil. The ``ram_mb`` values
    are therefore **declared** model-card constants (spec §4.6 rung A), not
    measurements — flagged ``confidence="declared"`` so the UI is honest about
    it. (Replacing these with learned profiles is a later slice.)
    """
    models: list[TrackedModel] = []

    # FastEmbed embedding model
    try:
        from augmentum.memory.embeddings import EmbeddingService
        if EmbeddingService._model is not getattr(EmbeddingService, "_UNLOADED", None):
            models.append(TrackedModel(
                name=EmbeddingService.MODEL_NAME,
                subsystem="embeddings",
                backend="fastembed",
                device="cpu",
                ram_mb=45,  # bge-small-en-v1.5 is ~45MB
                confidence="declared",
            ))
    except (ImportError, AttributeError) as exc:
        log.debug("inprocess_probe_embeddings_failed", error=str(exc))

    # Silero VAD
    try:
        from augmentum.voice.vad import VadProcessor
        # VadProcessor stores model as class-level; check if any instance exists
        if hasattr(VadProcessor, "_model") and VadProcessor._model is not None:
            models.append(TrackedModel(
                name="silero-vad-v6",
                subsystem="vad",
                backend="onnx",
                device="cpu",
                ram_mb=3,
                confidence="declared",
            ))
    except (ImportError, AttributeError) as exc:
        log.debug("inprocess_probe_vad_failed", error=str(exc))

    # WeSpeaker speaker verification
    try:
        from augmentum.voice.speaker import SpeakerVerifier
        if hasattr(SpeakerVerifier, "_session") and SpeakerVerifier._session is not None:
            models.append(TrackedModel(
                name="wespeaker-resnet34",
                subsystem="speaker",
                backend="onnx",
                device="cpu",
                ram_mb=28,
                confidence="declared",
            ))
    except (ImportError, AttributeError) as exc:
        log.debug("inprocess_probe_speaker_failed", error=str(exc))

    # Reranker model
    try:
        from augmentum.memory.reranker import RerankService
        if hasattr(RerankService, "_model") and RerankService._model is not None:
            models.append(TrackedModel(
                name=getattr(RerankService, "MODEL_NAME", "reranker"),
                subsystem="reranker",
                backend="fastembed",
                device="cpu",
                ram_mb=35,
                confidence="declared",
            ))
    except (ImportError, AttributeError) as exc:
        log.debug("inprocess_probe_reranker_failed", error=str(exc))

    return models


def invalidate(app_state, modality: str, *, disk: bool = False) -> None:
    """Convenience for action sites: bump the modality's inventory
    clock and optionally invalidate the disk cache.

    Safe when no ledger is wired (no-op) — tests + bare-bones
    deployments don't need to set up a ledger before they can call
    action routes. Call with ``disk=True`` after operations that
    add/remove files on disk (downloads, deletes); leave default
    False for in-process events that don't touch the filesystem
    (load/unload, provider CRUD).
    """
    ledger = getattr(app_state, "resource_ledger", None)
    if ledger is None:
        return
    if modality:
        ledger.invalidate_inventory(modality)
    if disk:
        ledger.invalidate_disk()


def _probe_disk_batch(
    targets: list[tuple[str, str]],
) -> list[tuple[str, str, int, int, str]]:
    """Threadpool worker: statfs each ``(modality, dir)`` pair.

    Returns ``(modality, dir, free_bytes, total_bytes, error)``. Runs
    in a worker thread so a degraded mount stalling for seconds
    doesn't freeze the event loop.
    """
    import shutil

    results: list[tuple[str, str, int, int, str]] = []
    for modality, path in targets:
        try:
            usage = shutil.disk_usage(path)
            results.append((modality, path, int(usage.free), int(usage.total), ""))
        except OSError as exc:
            results.append((modality, path, 0, 0, str(exc)))
    return results


def _dir_mtimes(dirs: list[str]) -> dict[str, float]:
    """Best-effort dict of {dir: mtime}. Missing dirs are skipped.

    The mtime of a directory advances when a file is added, removed,
    or renamed in it — but NOT when a file's contents change. That's
    fine: we care about inventory shape, not content drift. Download
    completion creates a new file → bumps mtime → inventory refreshes.
    """
    import os

    out: dict[str, float] = {}
    for d in dirs:
        if not d:
            continue
        try:
            st = os.stat(d)
            out[d] = float(st.st_mtime)
        except (OSError, FileNotFoundError):
            continue
    return out


def _format_mtimes(label: str, mtimes: dict[str, float]) -> str:
    """Stringify a mtimes dict for inclusion in the inventory_etag."""
    return f"{label}:" + "|".join(
        f"{k}@{v:.3f}" for k, v in sorted(mtimes.items())
    )


def _dir_size(path: str, *, max_depth: int = 2) -> int:
    """Sum of file sizes under ``path``, bounded by ``max_depth``.

    Used for image-model folders. Image models live in directories
    of ~5-10 files; max_depth=2 covers typical diffusers structure
    without risking a symlink loop into the void.
    """
    import os

    total = 0

    def _walk(d: str, depth: int) -> None:
        nonlocal total
        if depth > max_depth:
            return
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            _walk(entry.path, depth + 1)
                    except (OSError, FileNotFoundError):
                        continue
        except (OSError, FileNotFoundError):
            return

    _walk(path, 0)
    return total


def _can_fit(size_bytes: int, gpu_free_mb: int) -> bool:
    """Crude capability check: model size in MB <= GPU free MB.

    A 13B Q4 GGUF is ~7 GB. On an 8 GB VRAM peer with 7.2 GB free this
    returns True; with 6 GB free it returns False. We don't try to
    model partial offload — the answer to "can this load?" is
    intentionally conservative. Cross-peer routing already handles
    "can fit but won't be fast" via the existing scoring layer.

    Returns True when either operand is 0 (unknown) — we prefer
    surfacing the model + letting the operator try than greying out
    something they could have loaded.
    """
    if size_bytes <= 0 or gpu_free_mb <= 0:
        return True
    size_mb = size_bytes // (1024 * 1024)
    # Leave ~10% headroom for inference workspace; the model file
    # itself isn't the whole runtime memory footprint.
    return size_mb < int(gpu_free_mb * 0.9)


def _infer_device(size_vram: int, size_ram: int) -> str:
    """Infer device from VRAM/RAM split."""
    if size_vram and size_ram:
        if size_vram > size_ram:
            return "gpu"
        return "gpu+cpu"
    if size_vram and not size_ram:
        return "gpu"
    if size_ram and not size_vram:
        return "cpu"
    return "unknown"


def _extract_engine_actual_memory(status: dict) -> tuple[int, int]:
    """Return llama.cpp-reported VRAM/RAM totals from manager status."""
    actual = status.get("actual_memory")
    if not isinstance(actual, dict):
        return 0, 0
    try:
        vram_mb = int(actual.get("vram_total_mib") or 0)
    except (TypeError, ValueError):
        vram_mb = 0
    try:
        ram_mb = int(actual.get("ram_total_mib") or 0)
    except (TypeError, ValueError):
        ram_mb = 0
    return max(0, vram_mb), max(0, ram_mb)


class ResourceLedger:
    """Aggregates resource state across all Augmentum subsystems."""

    # Concurrent UI polls share one underlying collection while the
    # cached snapshot is fresher than this window. 8s matches the UI's
    # 15s background poll cadence — most polls hit the cache, only the
    # explicit refresh button forces a fresh nvidia-smi. The popover's
    # 3s polls are kept cheap by the same cache. ``force=True`` still
    # bypasses for callers that need fresh data (admin refresh,
    # capability fingerprint changes).
    _COLLECT_TTL_S: float = 8.0

    # Sprint A — disk probe TTL. Disk usage rarely changes between
    # collects; the underlying statfs is wrapped in to_thread but a
    # degraded mount can still stall for seconds. Cache results for
    # this long so successive collects within the window reuse them
    # without re-probing. Distinct from _COLLECT_TTL_S because the
    # overall snapshot may need fresher GPU/model data while disk
    # is fine for longer.
    _DISK_PROBE_TTL_S: float = 15.0

    def __init__(self, db: aiosqlite.Connection | None = None) -> None:
        self._db = db
        self._model_manager: ModelManager | None = None
        self._provider_registry: ProviderRegistry | None = None
        self._llama_manager = None
        # Secondary local engine ("Slot B") — a second resident llama-server
        # whose VRAM/RAM must show up in the snapshot too, else it lands in
        # ``unattributed_vram_mb`` and admission checks under-count usage.
        # Stored as the slot wrapper; its ``.manager`` is read live each
        # collect (None until a model is loaded).
        self._secondary_slot = None
        # Managed classifier slot ("Slot C") — a RESIDENT llama-server
        # (idle_timeout=0) that permanently holds VRAM/RAM once loaded, so it
        # MUST show up in the snapshot too. Being the always-on slot, leaving
        # it out is worse than for Slot B: its footprint silently lands in
        # ``unattributed_vram_mb`` and every admission check (primary AND
        # Slot B loads) over-counts free VRAM by the classifier's resident
        # size → green-lit loads that OOM. Same live-``.manager`` read as B.
        self._classifier_slot = None
        self._pipeline_registry = None
        self._jobs_store = None  # JobsStore — set by lifespan after init
        self._last_snapshot: ResourceSnapshot | None = None
        # Disk probe cache — per-dir, last (free_bytes, total_bytes,
        # error) tuple plus the monotonic timestamp it was probed at.
        # The cache is bypassed when ``invalidate_disk()`` was called
        # since the last probe (a download just completed; old free
        # space figure is misleading).
        self._disk_cache: dict[str, tuple[float, int, int, str]] = {}
        self._disk_invalidated_at: float = 0.0
        # Inventory cache — per-modality. Three parallel dicts indexed
        # by modality slug ('llm', 'image', etc.):
        #   _inventory_cache: cached enumeration result
        #   _inventory_mtimes: {dir: mtime} fingerprint at the time of
        #     the cache write — refresh when any watched dir mtime has
        #     advanced
        #   _inventory_action_clock: monotonic counter bumped by
        #     ``invalidate_inventory(modality)``; the cached entry
        #     records the clock value it was built against. Mismatch
        #     forces a refresh even if mtime is unchanged (covers
        #     load/unload events that don't touch the filesystem).
        self._inventory_cache: dict[str, list[InventoryEntry]] = {}
        self._inventory_mtimes: dict[str, dict[str, float]] = {}
        self._inventory_clock: dict[str, int] = {}
        self._inventory_cache_clock: dict[str, int] = {}
        # Wall-clock (monotonic) of last successful collect — paired
        # with ``_last_snapshot`` for the TTL cache.
        self._last_collect_at: float = 0.0
        # Single-flight handle: the one in-flight ``_collect_uncached`` task.
        # Every concurrent caller (forced or stale) awaits THIS instead of
        # starting its own probe, so N simultaneous requests cost one collect,
        # not N. Cleared by a done-callback so a cancelled awaiter can't strand
        # it. Replaces the serialize-behind-``_collect_lock`` behavior where
        # each forced caller ran a full probe in turn.
        self._collect_in_flight: asyncio.Task[ResourceSnapshot] | None = None
        # Persistence gate: write a history row + bump profiles only
        # when the set of loaded models changes (load/unload events).
        # Steady-state polls — by far the common case — produce no DB
        # writes at all. ``_last_persisted_models`` is the model set we
        # last successfully wrote; ``_persist_in_flight`` ensures at
        # most one persist task can exist at any moment, so even bursty
        # load events can't queue up the way the prior fire-on-every-
        # collect path did (that produced the +15s/persist pile-up
        # observed on the live system).
        self._last_persisted_models: frozenset[tuple[str, str]] | None = None
        self._persist_in_flight: bool = False

    def set_model_manager(self, mm: ModelManager) -> None:
        self._model_manager = mm

    def set_provider_registry(self, pr: ProviderRegistry) -> None:
        self._provider_registry = pr

    def set_llama_manager(self, mgr) -> None:
        self._llama_manager = mgr

    def set_secondary_slot(self, slot) -> None:
        """Track the secondary engine slot ("Slot B") for resource accounting.

        Optional — when unset the snapshot only covers the primary engine.
        The slot's underlying manager is read live each collect so a model
        loaded into Slot B after this call is still picked up.
        """
        self._secondary_slot = slot

    def set_classifier_slot(self, slot) -> None:
        """Track the managed classifier slot ("Slot C") for resource accounting.

        Parity with :meth:`set_secondary_slot`. Optional — when unset the
        snapshot only covers the primary engine (+ Slot B). The slot's
        underlying manager is read live each collect so a model loaded into
        Slot C after this call is still picked up. Slot C is RESIDENT, so its
        VRAM is effectively permanent once loaded — tracking it here is what
        keeps admission control from over-committing the GPU.
        """
        self._classifier_slot = slot

    def set_image_subsystem(self, pipeline_reg, hw_profile=None) -> None:
        self._pipeline_registry = pipeline_reg

    def set_jobs_store(self, jobs_store) -> None:
        """Wire the JobsStore for active-jobs reporting.

        Optional — when missing, snapshots have ``active_jobs=[]``.
        Set during server startup; never re-set at runtime.
        """
        self._jobs_store = jobs_store

    # ── Invalidation hooks (called by action sites) ───────────────

    def invalidate_inventory(self, modality: str) -> None:
        """Force a re-enumeration of ``modality``'s inventory on the
        next collect.

        Called from action sites (model load/unload, download
        completion, provider CRUD, etc.) so the inventory list and
        ``inventory_etag`` advance immediately, without waiting for
        the next directory mtime tick. Idempotent + cheap — just
        bumps an integer counter.
        """
        if not modality:
            return
        self._inventory_clock[modality] = (
            self._inventory_clock.get(modality, 0) + 1
        )

    def invalidate_disk(self) -> None:
        """Force re-probing of disk usage on the next collect.

        Called after a download completes or a model is deleted so
        the free-space figure in the UI reflects the change without
        waiting for the disk-probe TTL to expire. Idempotent.
        """
        self._disk_invalidated_at = time.monotonic()

    # ── Probe helpers (sprint A) ─────────────────────────────────

    def _enumerate_disk_destinations(self) -> list[tuple[str, str]]:
        """Return ``(modality, dir)`` pairs for every distinct
        download destination, deduplicated.

        Reads from settings every call — cheap (dict lookup) and
        means a settings change is picked up on the next collect
        without restart. Empty strings (operator hasn't configured
        a path) are skipped silently.
        """
        from augmentum.config import settings

        candidates: list[tuple[str, str]] = []

        def _add(modality: str, path: str | None) -> None:
            if not path:
                return
            path = str(path).strip()
            if not path:
                return
            candidates.append((modality, path))

        _add("llm", getattr(settings, "engine_model_dir", "") or "")
        _add("llm", getattr(settings, "llamacpp_model_dir", "") or "")
        _add("image", getattr(settings, "image_model_dir", "") or "")
        _add("knowledge", getattr(settings, "knowledge_packs_dir", "") or "")

        # Dedup while preserving order. Same path can serve multiple
        # modalities on a tight install; keep the first-seen modality
        # tag to match the operator's first configuration intent.
        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for modality, path in candidates:
            if path in seen:
                continue
            seen.add(path)
            deduped.append((modality, path))
        return deduped

    async def _probe_disk_destinations(self) -> list[DiskDestination]:
        """Sample free / total bytes for every configured download
        destination. Threadpool-wrapped (degraded mount can stall).

        Cache: per-dir, ``_DISK_PROBE_TTL_S`` window unless
        ``invalidate_disk()`` was called between this call and the
        last probe of that dir.
        """
        destinations = self._enumerate_disk_destinations()
        if not destinations:
            return []

        now = time.monotonic()
        results: list[DiskDestination] = []
        to_probe: list[tuple[str, str]] = []
        for modality, path in destinations:
            cached = self._disk_cache.get(path)
            if (
                cached is not None
                and cached[0] > self._disk_invalidated_at
                and (now - cached[0]) < self._DISK_PROBE_TTL_S
            ):
                _, free, total, err = cached
                results.append(DiskDestination(
                    dir=path, modality=modality,
                    free_bytes=free, total_bytes=total, error=err,
                ))
            else:
                to_probe.append((modality, path))

        if to_probe:
            # Run statfs syscalls in a worker thread — a degraded mount
            # can hang seconds.
            probed = await asyncio.to_thread(_probe_disk_batch, to_probe)
            for modality, path, free, total, err in probed:
                self._disk_cache[path] = (now, free, total, err)
                results.append(DiskDestination(
                    dir=path, modality=modality,
                    free_bytes=free, total_bytes=total, error=err,
                ))

        return results

    def _probe_active_jobs(self) -> list[JobStatus]:
        """Snapshot of currently-running jobs from JobsStore.

        Pure RAM read — pulls from JobsStore's ``_active`` runtime
        registry. Returns ``[]`` when no jobs store is wired
        (degraded gracefully — local panel works either way) or
        when the store hasn't implemented the runtime registry yet
        (the registry is added in this same sprint).
        """
        store = self._jobs_store
        if store is None:
            return []
        list_active = getattr(store, "list_active", None)
        if list_active is None:
            return []
        try:
            active = list_active()
        except Exception:
            log.warning("active_jobs_probe_failed", exc_info=True)
            return []
        out: list[JobStatus] = []
        for entry in active:
            try:
                payload = entry.get("payload") or {}
                target = (
                    payload.get("model_id")
                    or payload.get("name")
                    or payload.get("pack_id")
                    or ""
                )
                out.append(JobStatus(
                    job_id=str(entry.get("id", "")),
                    user_id=str(entry.get("user_id", "")),
                    kind=str(entry.get("job_type", "")),
                    target_id=str(target),
                    progress_pct=float(entry.get("progress") or 0.0),
                    stage=str(entry.get("stage", "")),
                    started_at=int(entry.get("started_at") or 0),
                ))
            except (ValueError, TypeError, KeyError) as exc:
                log.debug("ledger_job_entry_parse_failed", entry_id=entry.get("id") if isinstance(entry, dict) else None, error=str(exc))
                continue
        return out

    async def _probe_inventory(
        self, models_in_snapshot: list[TrackedModel], gpu_free_mb: int,
    ) -> tuple[list[InventoryEntry], str]:
        """Enumerate every model on disk + compute inventory_etag.

        Per-modality mtime + action-clock cache. ``capable`` is
        computed against current ``gpu_free_mb`` so the UI can
        grey out entries that won't fit *right now*. ``loaded`` is
        joined against ``models_in_snapshot`` so the entry knows
        whether it's currently in service.

        Per-modality cache keyed by directory mtime fingerprint AND
        the action-clock counter (so a load/unload that doesn't
        change the filesystem still forces a refresh).
        """
        loaded_names = {m.name for m in models_in_snapshot}

        # Cache key per modality is (dir_mtimes_dict, action_clock).
        # If either changed, re-enumerate; otherwise reuse cached entries.
        all_entries: list[InventoryEntry] = []
        mtime_parts: list[str] = []

        # LLM — from model_manager + filesystem scan of GGUF dirs.
        llm_entries, llm_mtimes = await self._enum_llm_inventory(
            loaded_names=loaded_names, gpu_free_mb=gpu_free_mb,
        )
        all_entries.extend(llm_entries)
        mtime_parts.append(_format_mtimes("llm", llm_mtimes))

        # Image — pipeline registry + image_model_dir scan.
        image_entries, image_mtimes = self._enum_image_inventory(
            loaded_names=loaded_names, gpu_free_mb=gpu_free_mb,
        )
        all_entries.extend(image_entries)
        mtime_parts.append(_format_mtimes("image", image_mtimes))

        # Audio providers (TTS + STT) — single DB SELECT, cheap; gated
        # on action-clock so the SELECT only runs after a provider CRUD.
        audio_entries = await self._enum_audio_inventory(
            loaded_names=loaded_names,
        )
        all_entries.extend(audio_entries)

        # Knowledge packs — single DB SELECT, gated the same way.
        kp_entries = await self._enum_knowledge_inventory(
            loaded_names=loaded_names,
        )
        all_entries.extend(kp_entries)

        # Etag covers mtimes + action clocks. SHA-1 (not crypto here, just
        # change detection) keeps it short for the wire format Sprint B
        # will publish over fabric.
        import hashlib
        clock_part = "|".join(
            f"{k}={v}" for k, v in sorted(self._inventory_clock.items())
        )
        etag = hashlib.sha1(
            ("|".join(mtime_parts) + ";;" + clock_part).encode("utf-8"),
        ).hexdigest()[:16]
        return all_entries, etag

    async def _enum_llm_inventory(
        self, *, loaded_names: set[str], gpu_free_mb: int,
    ) -> tuple[list[InventoryEntry], dict[str, float]]:
        """LLM inventory: Ollama list + filesystem scan of GGUF dirs.

        Mtime-cached: if no watched dir's mtime has advanced AND no
        invalidate_inventory('llm') has fired, return the cached list
        verbatim (no syscalls, no DB hits).
        """
        from augmentum.config import settings

        # Build the list of watched dirs first — mtime cache key.
        gguf_dirs = [
            getattr(settings, "engine_model_dir", "") or "",
            getattr(settings, "llamacpp_model_dir", "") or "",
        ]
        gguf_dirs = [d for d in gguf_dirs if d]

        # Capture current mtimes (cheap — one stat() per dir).
        current_mtimes = await asyncio.to_thread(_dir_mtimes, gguf_dirs)

        # Cache hit? Both mtime fingerprint AND action clock unchanged.
        modality = "llm"
        cached = self._inventory_cache.get(modality)
        last_clock = self._inventory_cache_clock.get(modality, -1)
        current_clock = self._inventory_clock.get(modality, 0)
        if (
            cached is not None
            and self._inventory_mtimes.get(modality) == current_mtimes
            and last_clock == current_clock
        ):
            # Reuse cached entries; still need to refresh ``loaded`` +
            # ``capable`` fields though, since they depend on transient
            # state (current loaded models / GPU free).
            return (
                [self._refresh_runtime_fields(e, loaded_names, gpu_free_mb) for e in cached],
                current_mtimes,
            )

        # Cache miss — enumerate.
        entries: list[InventoryEntry] = []

        # (a) Local GGUF files — filesystem glob.
        if self._model_manager is not None and gguf_dirs:
            for d in gguf_dirs:
                try:
                    files = await asyncio.to_thread(
                        self._model_manager.list_local_gguf, d,
                    )
                except Exception:
                    log.debug("inventory_llm_glob_failed", dir=d, exc_info=True)
                    continue
                for f in files or []:
                    size = int(f.get("size") or 0)
                    name = str(f.get("name") or "")
                    if not name:
                        continue
                    entries.append(InventoryEntry(
                        name=name,
                        modality="llm",
                        backend="gguf",
                        size_bytes=size,
                        location=str(f.get("path") or d),
                        loaded=name in loaded_names,
                        capable=_can_fit(size, gpu_free_mb),
                        metadata={
                            "quantization": str(f.get("quantization", "") or ""),
                            "family": str(f.get("family", "") or ""),
                        },
                    ))

        # (b) Ollama list — model_manager already proxies this.
        if self._model_manager is not None:
            try:
                ollama_models = await self._model_manager.list_all_models()
            except Exception:
                log.debug("inventory_llm_ollama_failed", exc_info=True)
                ollama_models = []
            for m in ollama_models or []:
                # ModelInfo dataclass attrs are inconsistent across backends;
                # do attribute-safe reads.
                backend = getattr(m, "backend", "ollama")
                if backend == "ollama":
                    name = getattr(m, "name", "") or ""
                    size_bytes = int(getattr(m, "size", 0) or 0)
                    if not name:
                        continue
                    entries.append(InventoryEntry(
                        name=name,
                        modality="llm",
                        backend="ollama",
                        size_bytes=size_bytes,
                        location="ollama",
                        loaded=name in loaded_names,
                        # Ollama models are runnable on partial offload —
                        # we mark capable=True conservatively; the
                        # 0-byte case (unknown size) stays True too.
                        capable=size_bytes == 0 or _can_fit(size_bytes // 2, gpu_free_mb),
                        metadata={},
                    ))

        # Stash cache for next call.
        self._inventory_cache[modality] = entries
        self._inventory_mtimes[modality] = current_mtimes
        self._inventory_cache_clock[modality] = current_clock
        return entries, current_mtimes

    def _enum_image_inventory(
        self, *, loaded_names: set[str], gpu_free_mb: int,
    ) -> tuple[list[InventoryEntry], dict[str, float]]:
        """Image inventory: image_model_dir scan + pipeline registry.

        Lightweight — image models are large but few. mtime-cached.
        """
        from augmentum.config import settings

        image_dir = getattr(settings, "image_model_dir", "") or ""
        watched = [image_dir] if image_dir else []
        current_mtimes = _dir_mtimes(watched)

        modality = "image"
        cached = self._inventory_cache.get(modality)
        last_clock = self._inventory_cache_clock.get(modality, -1)
        current_clock = self._inventory_clock.get(modality, 0)
        if (
            cached is not None
            and self._inventory_mtimes.get(modality) == current_mtimes
            and last_clock == current_clock
        ):
            return (
                [self._refresh_runtime_fields(e, loaded_names, gpu_free_mb) for e in cached],
                current_mtimes,
            )

        entries: list[InventoryEntry] = []
        if image_dir:
            try:
                import os
                for entry in os.scandir(image_dir):
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    # Sum the size of files within the model dir (image
                    # models are folder-shaped). Bounded by max depth 2
                    # so we never traverse forever on a misconfigured
                    # symlink loop.
                    size = _dir_size(entry.path, max_depth=2)
                    entries.append(InventoryEntry(
                        name=name,
                        modality="image",
                        backend="diffusers",
                        size_bytes=size,
                        location=entry.path,
                        loaded=name in loaded_names,
                        capable=_can_fit(size // 4, gpu_free_mb),
                        metadata={},
                    ))
            except (OSError, FileNotFoundError):
                log.debug("inventory_image_scan_failed", dir=image_dir, exc_info=True)

        self._inventory_cache[modality] = entries
        self._inventory_mtimes[modality] = current_mtimes
        self._inventory_cache_clock[modality] = current_clock
        return entries, current_mtimes

    async def _enum_audio_inventory(
        self, *, loaded_names: set[str],
    ) -> list[InventoryEntry]:
        """TTS + STT provider list from the audio_providers SQLite table.

        Single SELECT, gated by action clock — runs only after provider
        CRUD has fired ``invalidate_inventory('tts')`` /
        ``invalidate_inventory('stt')``. On idle systems this is a
        cache hit and never hits the DB.
        """
        if self._db is None:
            return []

        modality_tts = "tts"
        modality_stt = "stt"
        clock_tts = self._inventory_clock.get(modality_tts, 0)
        clock_stt = self._inventory_clock.get(modality_stt, 0)
        cached_tts = self._inventory_cache.get(modality_tts)
        cached_stt = self._inventory_cache.get(modality_stt)
        if (
            cached_tts is not None
            and cached_stt is not None
            and self._inventory_cache_clock.get(modality_tts, -1) == clock_tts
            and self._inventory_cache_clock.get(modality_stt, -1) == clock_stt
        ):
            return [
                self._refresh_runtime_fields(e, loaded_names, 0)
                for e in (cached_tts + cached_stt)
            ]

        entries_tts: list[InventoryEntry] = []
        entries_stt: list[InventoryEntry] = []
        try:
            cursor = await self._db.execute(
                "SELECT id, kind, name, base_url FROM audio_providers"
            )
            rows = await cursor.fetchall()
        except Exception:
            # Table missing (very fresh schema) — surface no entries
            # without crashing.
            log.debug("inventory_audio_select_failed", exc_info=True)
            rows = []

        for row in rows or []:
            provider_id = str(row[0]) if row[0] is not None else ""
            kind = str(row[1] or "").lower()
            name = str(row[2] or "")
            base_url = str(row[3] or "")
            if not name or not kind:
                continue
            entry = InventoryEntry(
                name=name,
                modality="tts" if "tts" in kind else "stt",
                backend=base_url or "provider",
                size_bytes=0,  # remote providers have no on-disk size
                location=base_url or "remote",
                loaded=False,   # provider is configured, not "loaded"
                capable=True,   # remote — always callable
                metadata={"provider_id": provider_id, "kind": kind},
            )
            if entry.modality == "tts":
                entries_tts.append(entry)
            else:
                entries_stt.append(entry)

        self._inventory_cache[modality_tts] = entries_tts
        self._inventory_cache[modality_stt] = entries_stt
        self._inventory_cache_clock[modality_tts] = clock_tts
        self._inventory_cache_clock[modality_stt] = clock_stt
        return entries_tts + entries_stt

    async def _enum_knowledge_inventory(
        self, *, loaded_names: set[str],
    ) -> list[InventoryEntry]:
        """Knowledge packs from the knowledge_packs SQLite table.

        Same shape as ``_enum_audio_inventory`` — action-clock gated,
        single SELECT on miss.
        """
        if self._db is None:
            return []

        modality = "knowledge"
        clock = self._inventory_clock.get(modality, 0)
        cached = self._inventory_cache.get(modality)
        if (
            cached is not None
            and self._inventory_cache_clock.get(modality, -1) == clock
        ):
            return [
                self._refresh_runtime_fields(e, loaded_names, 0)
                for e in cached
            ]

        entries: list[InventoryEntry] = []
        try:
            cursor = await self._db.execute(
                "SELECT id, name, pack_format, install_path, size_bytes "
                "FROM knowledge_packs"
            )
            rows = await cursor.fetchall()
        except Exception:
            log.debug("inventory_knowledge_select_failed", exc_info=True)
            rows = []

        for row in rows or []:
            pack_id = str(row[0]) if row[0] is not None else ""
            name = str(row[1] or "")
            pack_format = str(row[2] or "")
            install_path = str(row[3] or "")
            size_bytes = int(row[4] or 0)
            if not name:
                continue
            entries.append(InventoryEntry(
                name=name,
                modality="knowledge",
                backend=pack_format or "augpack",
                size_bytes=size_bytes,
                location=install_path or pack_id,
                loaded=False,
                capable=True,
                metadata={"pack_id": pack_id, "pack_format": pack_format},
            ))

        self._inventory_cache[modality] = entries
        self._inventory_cache_clock[modality] = clock
        return entries

    @staticmethod
    def _refresh_runtime_fields(
        entry: InventoryEntry, loaded_names: set[str], gpu_free_mb: int,
    ) -> InventoryEntry:
        """Return a copy of ``entry`` with ``loaded`` + ``capable``
        recomputed against current snapshot state.

        Inventory caching is keyed by mtime / action clock — those
        cover what's on disk. ``loaded`` (current models) and
        ``capable`` (current VRAM headroom) are transient and refreshed
        on every collect even when the cache otherwise hits.
        """
        new_loaded = entry.name in loaded_names
        new_capable = entry.capable
        if gpu_free_mb > 0 and entry.size_bytes > 0:
            new_capable = _can_fit(entry.size_bytes, gpu_free_mb)
        if new_loaded == entry.loaded and new_capable == entry.capable:
            return entry
        return InventoryEntry(
            name=entry.name, modality=entry.modality, backend=entry.backend,
            size_bytes=entry.size_bytes, location=entry.location,
            loaded=new_loaded, capable=new_capable,
            metadata=entry.metadata,
        )

    async def collect(
        self, force: bool = False, *, cache_only: bool = False,
    ) -> ResourceSnapshot:
        """Poll all backends and build a unified snapshot.

        Cache + coalesce: a snapshot fresher than ``_COLLECT_TTL_S``
        returns immediately without re-probing. Concurrent calls find
        the cache stale at the same time, take a single async lock,
        and only the first one through performs the underlying probe;
        subsequent callers receive the freshly-cached snapshot. Pass
        ``force=True`` to bypass both checks (e.g. for an explicit
        admin refresh).

        ``cache_only=True`` is the read-path contract (``GET /status``):
        return the last snapshot regardless of age and NEVER take the
        slow nvidia-smi/model-manager path inline — the background sampler
        owns the refresh. Only a true cold start (no snapshot yet) falls
        through to one live collect so the first poll isn't empty.

        Net effect under typical UI load (4-6 surfaces polling
        ``/resource/status`` concurrently): one nvidia-smi shell-out
        every ``_COLLECT_TTL_S`` seconds instead of one per request.
        """
        # Fast path — fresh-enough cache, no awaiting. cache_only also serves a
        # stale snapshot here (read path never probes).
        if self._last_snapshot is not None:
            age = time.monotonic() - self._last_collect_at
            if cache_only:
                return self._last_snapshot
            if not force and age < self._COLLECT_TTL_S:
                return self._last_snapshot

        # Single-flight: collapse every concurrent caller onto ONE in-flight
        # collect. A burst of ?fresh=1 requests that arrive DURING a probe all
        # await the same task instead of each running a full ~9s probe behind a
        # lock (the decreasing-staircase stall in the logs). A forced caller
        # with NOTHING in flight still runs a fresh probe, preserving force's
        # bypass-the-cache contract. The check-and-set is atomic — there is no
        # await between reading ``_collect_in_flight`` and assigning it on this
        # single event loop, so two coroutines can't both start a probe.
        inflight = self._collect_in_flight
        if inflight is not None and not inflight.done():
            return await asyncio.shield(inflight)

        task: asyncio.Task[ResourceSnapshot] = asyncio.ensure_future(
            self._collect_uncached()
        )
        self._collect_in_flight = task
        # Clear via callback (not finally) so a cancelled awaiter — e.g. a
        # client that disconnects mid-probe — can't strand the handle while the
        # shielded task is still running and would-be sharers start a second.
        task.add_done_callback(self._clear_inflight_collect)
        return await asyncio.shield(task)

    def _clear_inflight_collect(self, task: asyncio.Task[ResourceSnapshot]) -> None:
        """Done-callback: drop the single-flight handle when its task ends."""
        if self._collect_in_flight is task:
            self._collect_in_flight = None

    async def _collect_uncached(self) -> ResourceSnapshot:
        """Internal: do the actual probe + persistence work.

        Always takes the slow path (nvidia-smi + model manager +
        provider registry probes + DB writes). Callers should normally
        go through ``collect()`` which fronts this with the TTL cache
        and coalescing lock.
        """
        models: list[TrackedModel] = []

        # _probe_gpu / _probe_gpu_processes shell out to nvidia-smi (5s
        # timeout each). Running them inline blocks the event loop for the
        # duration — bad for the dashboards that call collect() on a UI
        # poll. Hand off to a worker thread; _probe_ram is /proc-only and
        # cheap enough to leave inline.
        gpu_name, gpu_total, gpu_used, gpu_free = await asyncio.to_thread(_probe_gpu)
        ram_total, ram_used, ram_free = _probe_ram()
        gpu_processes = await asyncio.to_thread(_probe_gpu_processes)

        # --- LLM models from ModelManager (Ollama /api/ps, llama.cpp /slots) ---
        if self._model_manager:
            try:
                for rm in await self._model_manager.get_running_models():
                    svram = rm.size_vram or 0
                    sram = rm.size_ram or 0
                    models.append(TrackedModel(
                        name=rm.name,
                        subsystem="llm",
                        backend=rm.backend,
                        device=_infer_device(svram, sram),
                        vram_mb=svram // (1024 * 1024) if svram else 0,
                        ram_mb=sram // (1024 * 1024) if sram else 0,
                        quantization=rm.details.get("quantization_level", ""),
                        parameter_size=rm.details.get("parameter_size", ""),
                        family=rm.details.get("family", ""),
                        expires_at=rm.expires_at,
                    ))
            except Exception:
                log.warning("resource_collect_llm_failed", exc_info=True)

        # --- LM Studio (only loaded models via native /api/v0/models) ---
        if self._provider_registry:
            try:
                known_names = {m.name for m in models}
                for lms_model in await _probe_lmstudio_models(self._provider_registry):
                    if lms_model.name not in known_names:
                        models.append(lms_model)
                        known_names.add(lms_model.name)
            except Exception:
                log.warning("resource_collect_lmstudio_failed", exc_info=True)

        # --- Augmentum Engine managed llama-server(s): primary + Slot B ---
        # Surface each resident engine with per-model VRAM that NEVER sums
        # past the device total. Two passes:
        #   1. Take each model's RELIABLE per-process VRAM — its own
        #      actual_memory (llama-server log totals), else nvidia-smi
        #      per-pid. RAM is always per-process (psutil), so it's exact.
        #   2. Split the REMAINING device VRAM (gpu_used minus external GPU
        #      processes minus the reliable sum) among models that had no
        #      reliable number, proportional to their load-plan estimate.
        # This bounds attributed VRAM to physical reality: two models can't
        # report more VRAM than the GPU is actually using. The old approach
        # gave a model lacking real data its *planned* budget — which had no
        # relation to usage and overflowed the device total.
        engine_managers = []
        if self._llama_manager:
            engine_managers.append(self._llama_manager)
        _sec_mgr = getattr(self._secondary_slot, "manager", None) if self._secondary_slot else None
        if _sec_mgr is not None:
            engine_managers.append(_sec_mgr)
        _cls_mgr = getattr(self._classifier_slot, "manager", None) if self._classifier_slot else None
        if _cls_mgr is not None:
            engine_managers.append(_cls_mgr)

        engine_entries: list[dict] = []
        known_names = {m.name for m in models}
        for _engine_mgr in engine_managers:
            try:
                status = _engine_mgr.status()
            except Exception:
                log.warning("resource_collect_engine_failed", exc_info=True)
                continue
            state = str(status.get("state", "") or "")
            model_name = status.get("model_id") or ""
            if not model_name and status.get("model_path"):
                model_name = str(status["model_path"]).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if model_name.lower().endswith(".gguf"):
                    model_name = model_name[:-5]
            if not model_name or state not in {"starting", "draining", "ready"}:
                continue
            if model_name in known_names:
                continue
            known_names.add(model_name)

            pid = int(status.get("pid") or 0)
            actual_vram_mb, actual_ram_mb = _extract_engine_actual_memory(status)
            vram_mb = actual_vram_mb
            vram_reliable = actual_vram_mb > 0
            if not vram_reliable:
                proc = next((p for p in gpu_processes if p.pid == pid), None)
                if proc and proc.vram_mb > 0:
                    vram_mb = proc.vram_mb
                    vram_reliable = True

            ram_mb = actual_ram_mb
            if ram_mb <= 0:
                ram_info = status.get("ram") or {}
                if isinstance(ram_info, dict):
                    ram_mb = int(ram_info.get("rss_mb") or 0)
                if ram_mb <= 0 and pid:
                    try:
                        import psutil

                        ram_mb = int(psutil.Process(pid).memory_info().rss // (1024 * 1024))
                    except Exception:
                        ram_mb = 0

            plan_mem = (status.get("load_plan") or {}).get("memory") or {}
            est_vram = int(
                plan_mem.get("estimated_vram_mb")
                or plan_mem.get("steady_vram_mb")
                or 0
            )
            # During load the model is still ramping up; project to the plan's
            # steady footprint (VRAM capped at device-used) so the card shows
            # where it WILL land, not the partial current value. Marked
            # reliable so pass 2 treats the projection as fixed, not residual.
            if state in {"starting", "draining"}:
                planned_vram = (
                    min(est_vram, gpu_used) if est_vram > 0 and gpu_used > 0 else est_vram
                )
                if planned_vram > vram_mb:
                    vram_mb = planned_vram
                    vram_reliable = True
                planned_ram = int(
                    plan_mem.get("estimated_ram_mb")
                    or plan_mem.get("steady_ram_mb")
                    or 0
                )
                if planned_ram > ram_mb:
                    ram_mb = planned_ram
            load_cfg = status.get("load_config") or {}
            engine_entries.append({
                "name": model_name,
                "pid": pid,
                "vram_mb": int(max(0, vram_mb)),
                "reliable": vram_reliable,
                "est_vram": est_vram,
                "ram_mb": int(max(0, ram_mb)),
                "family": (status.get("profile") or {}).get("architecture", ""),
                "quant": (load_cfg.get("kv_cache_type") or "") or "",
                "state": state,
            })

        if engine_entries:
            # Pass 2 — distribute the device VRAM the reliable models DIDN'T
            # claim among the models that lacked a per-process number.
            external_vram = sum(
                p.vram_mb for p in gpu_processes
                if not any(e["pid"] and e["pid"] == p.pid for e in engine_entries)
            )
            reliable_sum = sum(e["vram_mb"] for e in engine_entries if e["reliable"])
            unknown = [e for e in engine_entries if not e["reliable"]]
            if unknown:
                remaining = (
                    max(0, gpu_used - external_vram - reliable_sum)
                    if gpu_used > 0 else 0
                )
                total_est = sum(e["est_vram"] for e in unknown)
                for e in unknown:
                    if gpu_used <= 0:
                        # No device signal at all — the plan estimate is the
                        # only number available.
                        e["vram_mb"] = e["est_vram"]
                    elif remaining <= 0:
                        # Reliable models + external already account for all
                        # used VRAM; this model's share is unmeasurable here.
                        e["vram_mb"] = 0
                    elif total_est > 0:
                        e["vram_mb"] = int(remaining * e["est_vram"] / total_est)
                    else:
                        e["vram_mb"] = remaining // len(unknown)

            for e in engine_entries:
                models.append(TrackedModel(
                    name=e["name"],
                    subsystem="llm",
                    backend="engine",
                    device=_infer_device(e["vram_mb"] * 1024 * 1024, e["ram_mb"] * 1024 * 1024),
                    vram_mb=e["vram_mb"],
                    ram_mb=e["ram_mb"],
                    family=e["family"],
                    quantization=e["quant"],
                    status="ready" if e["state"] == "ready" else "loading",
                    pid=e["pid"],
                    # VRAM is measured when it came from the model's own
                    # actual_memory / per-pid figure; the proportional residual
                    # split (pass 2) is an estimate. RAM is always per-process.
                    confidence="measured" if e["reliable"] else "estimated",
                ))

        # --- Image model from diffusers pipeline registry ---
        if self._pipeline_registry and getattr(self._pipeline_registry, "is_loaded", False):
            try:
                img_vram = _get_torch_allocated_mb()
                models.append(TrackedModel(
                    name=getattr(self._pipeline_registry, "current_model", "unknown"),
                    subsystem="image",
                    backend="diffusers",
                    device="gpu" if img_vram > 100 else "cpu",
                    vram_mb=img_vram,
                    pipeline_type=getattr(self._pipeline_registry, "pipeline_type", ""),
                ))
            except Exception:
                log.warning("resource_collect_image_failed", exc_info=True)

        # --- In-process models (FastEmbed, Silero VAD, WeSpeaker, reranker) ---
        try:
            models.extend(_probe_inprocess_models())
        except Exception:
            log.warning("resource_collect_inprocess_failed", exc_info=True)

        # LM Studio's /api/v0/models doesn't expose per-model VRAM, so attribute
        # the lms.exe process VRAM (from nvidia-smi) across loaded LMS models.
        lms_vram_mb = sum(p.vram_mb for p in gpu_processes if p.label == "LM Studio")
        lms_models = [m for m in models if m.backend == "LM Studio"]
        if lms_vram_mb and lms_models:
            per_model = lms_vram_mb // len(lms_models)
            for m in lms_models:
                m.vram_mb = per_model

        # Calculate unattributed VRAM: total used minus sum of all process VRAM
        attributed_vram = sum(p.vram_mb for p in gpu_processes)
        unattributed = max(0, gpu_used - attributed_vram) if gpu_used > 0 else 0

        # Sprint A — disk destinations, active jobs, inventory. None of
        # these write to the DB. Each has its own cache layer so idle
        # collects spend nothing beyond a few dict reads.
        disk_destinations: list[DiskDestination] = []
        active_jobs: list[JobStatus] = []
        inventory: list[InventoryEntry] = []
        inventory_etag = ""
        try:
            disk_destinations = await self._probe_disk_destinations()
        except Exception:
            log.warning("resource_collect_disk_failed", exc_info=True)
        try:
            active_jobs = self._probe_active_jobs()
        except Exception:
            log.warning("resource_collect_jobs_failed", exc_info=True)
        try:
            inventory, inventory_etag = await self._probe_inventory(
                models_in_snapshot=models, gpu_free_mb=gpu_free,
            )
        except Exception:
            log.warning("resource_collect_inventory_failed", exc_info=True)

        snapshot = ResourceSnapshot(
            timestamp=datetime.utcnow(),
            gpu_name=gpu_name,
            gpu_total_mb=gpu_total,
            gpu_used_mb=gpu_used,
            gpu_free_mb=gpu_free,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_free_mb=ram_free,
            models=models,
            gpu_processes=gpu_processes,
            unattributed_vram_mb=unattributed,
            disk_destinations=disk_destinations,
            active_jobs=active_jobs,
            inventory=inventory,
            inventory_etag=inventory_etag,
        )

        self._last_snapshot = snapshot
        # Stamp completion time so the TTL cache can serve subsequent
        # callers from this snapshot.
        self._last_collect_at = time.monotonic()

        if self._db:
            # Persist only when the set of loaded models changed (load/unload
            # events). Steady-state polls write nothing. Single-flight gate
            # inside the helper guarantees at most one persist task ever
            # exists, so bursty events can't pile up.
            self._maybe_schedule_persist(snapshot)

        return snapshot

    def _persist_worth_doing(self, snap: ResourceSnapshot) -> bool:
        """True iff something happened that history needs to capture.

        The only signal worth a write is a change in the set of loaded
        models. Inference-time VRAM wiggle is uninteresting for the
        history chart and the live UI already serves real-time numbers
        from ``_last_snapshot``. With this gate, idle systems produce
        zero DB writes regardless of poll frequency.
        """
        current = frozenset((m.name, m.backend) for m in snap.models)
        return current != self._last_persisted_models

    def _maybe_schedule_persist(self, snap: ResourceSnapshot) -> None:
        """Schedule one persist task if state changed and none is in flight.

        Drops overlapping schedules — when a persist is already running,
        any concurrent change just relies on the next ``collect()`` to
        retrigger after the in-flight one clears the gate. We can't lose
        a meaningful event because ``_last_persisted_models`` only
        advances on successful commit, so a still-pending change keeps
        triggering until it's been written.

        Skipped entirely while the engine is mid-load (``ProcessState.
        STARTING``): the load path holds the augmentum.db writer lock
        for several minutes on big GGUFs, so a persist attempt here
        just burns the 30s busy_timeout, fails with ``database is
        locked``, and retries on the next snapshot. The model-set
        transition that triggered this call will fire again from the
        next ``collect()`` after the load completes, so no event is
        lost. Tracked as the 2026-05-15 ledger-storm incident.
        """
        if not self._persist_worth_doing(snap):
            return
        if self._persist_in_flight:
            return
        # Skip while ANY engine (primary, Slot B, or Slot C) is mid-load — a
        # big GGUF load holds augmentum.db's writer lock for minutes, so a
        # persist here just burns the busy_timeout and retries next collect.
        _loading_managers = [self._llama_manager]
        _sec_mgr = getattr(self._secondary_slot, "manager", None) if self._secondary_slot else None
        if _sec_mgr is not None:
            _loading_managers.append(_sec_mgr)
        _cls_mgr = getattr(self._classifier_slot, "manager", None) if self._classifier_slot else None
        if _cls_mgr is not None:
            _loading_managers.append(_cls_mgr)
        for _mgr in _loading_managers:
            if _mgr is None:
                continue
            state = str(getattr(_mgr, "state", "") or "")
            if state in ("starting", "draining", "stopping"):
                return
        self._persist_in_flight = True
        # ``track`` keeps a ref so Python's GC doesn't drop the persist
        # mid-execution — the failure mode is silently lost snapshots
        # (the ``_persist_in_flight`` flag still clears in the inner
        # ``finally`` so the next collect() proceeds, but THIS snapshot
        # vanishes if the task is collected before it runs).
        from augmentum.utils.bg_tasks import track
        track(self._persist_and_clear(snap))

    async def _persist_and_clear(self, snap: ResourceSnapshot) -> None:
        """Run one atomic persist; always release the in-flight gate."""
        try:
            await self._persist_atomic(snap)
            self._last_persisted_models = frozenset(
                (m.name, m.backend) for m in snap.models
            )
        except Exception:
            log.warning("resource_snapshot_persist_failed", exc_info=True)
        finally:
            self._persist_in_flight = False

    async def _persist_atomic(self, snap: ResourceSnapshot) -> None:
        """Profiles UPSERT + snapshot INSERT in a single transaction.

        ``BEGIN IMMEDIATE`` acquires the writer lock upfront so we fail
        fast on contention rather than discovering it mid-statement.
        Combining both writes halves writer-lock acquisitions versus
        the previous two-commit pattern.
        """
        if not self._db:
            return
        rows = [
            (
                m.name, m.subsystem, m.backend, m.vram_mb, m.ram_mb,
                m.device, m.quantization, m.parameter_size, m.family,
                m.pipeline_type,
            )
            for m in snap.models
        ]
        snap_row = (
            snap.timestamp.isoformat(),
            snap.gpu_total_mb, snap.gpu_used_mb, snap.gpu_free_mb,
            snap.ram_total_mb, snap.ram_used_mb, snap.ram_free_mb,
            len(snap.models),
            json.dumps([
                {"name": m.name, "backend": m.backend, "vram_mb": m.vram_mb}
                for m in snap.models
            ]),
        )

        await self._db.execute("BEGIN IMMEDIATE")
        try:
            if rows:
                await self._db.executemany(_UPSERT_PROFILES_SQL, rows)
            await self._db.execute(_INSERT_SNAPSHOT_SQL, snap_row)
            await self._db.commit()
        except Exception:
            try:
                await self._db.rollback()
            except aiosqlite.Error as rollback_exc:
                # Best-effort rollback inside an outer exception handler;
                # the outer raise still propagates the original failure.
                log.debug(
                    "resource_ledger_rollback_failed",
                    error=str(rollback_exc),
                )
            raise

        await self._maybe_prune_snapshots()

    @property
    def last_snapshot(self) -> ResourceSnapshot | None:
        return self._last_snapshot

    async def check_engine_fit(
        self, model_name: str, *, size_bytes: int = 0,
    ) -> tuple[bool, str, int, int]:
        """Decide whether ``model_name`` fits in current free VRAM.

        Returns ``(ok, reason, needed_mb, free_mb)``. Used as an admission
        gate before loading a model into a second resident slot, so the
        load is rejected with a clear message instead of OOM-crashing the
        llama-server subprocess.

        Conservative by design: when either the expected footprint or the
        free VRAM is unknown (0) it returns ``ok=True`` — mirroring
        :func:`_can_fit`, we'd rather let a possibly-valid load proceed than
        block on missing data (e.g. a model never loaded before has no
        profile yet; its first load is always allowed and teaches the
        profile for next time). Prefers the accumulated ``resource_profiles``
        footprint (what the model actually used last time) over the GGUF
        file size, which over-counts for partial-offload and under-counts
        for KV/compute workspace.

        ``free_mb`` already excludes whatever the primary engine holds —
        nvidia-smi reports true device-wide free VRAM — so this is the real
        headroom for a second model.
        """
        snap = await self.collect()  # TTL-cached; cheap on the hot path
        free_mb = int(snap.gpu_free_mb or 0)
        needed_mb = 0
        profile = await self.get_model_profile(model_name)
        if profile and profile.vram_mb > 0:
            needed_mb = int(profile.vram_mb)
        elif size_bytes > 0:
            needed_mb = int(size_bytes // (1024 * 1024))
        if free_mb <= 0 or needed_mb <= 0:
            return True, "", needed_mb, free_mb
        # Same ~10% workspace headroom as _can_fit — the weights aren't the
        # whole runtime footprint (KV cache + compute buffers on top).
        ok = needed_mb < int(free_mb * 0.9)
        reason = ""
        if not ok:
            reason = (
                f"needs ~{needed_mb / 1024:.1f} GB VRAM but only "
                f"~{free_mb / 1024:.1f} GB is free"
            )
        return ok, reason, needed_mb, free_mb

    async def check_ram_fit(
        self,
        *,
        needed_mb: int,
        label: str = "",
        headroom_frac: float = 0.85,
    ) -> tuple[bool, str, int, int]:
        """Decide whether ``needed_mb`` of HOST RAM can be admitted.

        Returns ``(ok, reason, needed_mb, available_mb)``.

        This is the missing counterpart to :meth:`check_engine_fit`. Until
        2026-07-25 Augmentum enforced a VRAM budget and had **no host-RAM
        budget at all**: RAM was probed, persisted to ``resource_snapshots``,
        and never consulted. Every consumer sized itself against the whole
        machine, independently, and the sum was free to exceed it — which is
        how a 128 GB box was driven into a forced restart.

        Deliberately asymmetric with the VRAM gate in two ways:

        * **Availability is container-aware** (``hostmem``), so under Docker
          this is the cgroup ceiling minus our working set, not the host's
          headroom. Reclaimable page cache is excluded from "used".
        * **Unknown data does NOT auto-approve.** ``check_engine_fit``
          returns ok on missing numbers because an over-refusal there costs a
          load and an under-refusal costs a subprocess crash. Here an
          under-refusal costs the user's whole machine, so a missing
          ``needed_mb`` is admitted (we know nothing) but a *known* need
          against *known* scarcity is refused.
        """
        from augmentum.resource import hostmem

        info = hostmem.memory_info()
        available_mb = int(info.available_mib)
        needed_mb = int(max(0, needed_mb))
        if needed_mb <= 0 or available_mb <= 0:
            return True, "", needed_mb, available_mb

        budget_mb = int(available_mb * max(0.0, min(1.0, headroom_frac)))
        ok = needed_mb <= budget_mb
        reason = ""
        if not ok:
            what = f"{label} " if label else ""
            # Real numbers, and the fact that this is OUR ceiling — a user
            # staring at 80 GB free in Task Manager deserves to know why we
            # said no (§7: refusals explain, never fail silently).
            scope = (
                f" (container limit {info.total_mib / 1024:.1f} GB)"
                if info.limited
                else ""
            )
            reason = (
                f"{what}needs ~{needed_mb / 1024:.1f} GB host RAM but only "
                f"~{available_mb / 1024:.1f} GB is available{scope}"
            )
            log.warning(
                "ram_admission_refused",
                label=label or "unknown",
                needed_mb=needed_mb,
                available_mb=available_mb,
                budget_mb=budget_mb,
                source=info.source,
                limited=info.limited,
            )
        return ok, reason, needed_mb, available_mb

    async def get_model_profile(self, model_name: str) -> ModelProfile | None:
        """Look up stored profile for a model."""
        if not self._db:
            return None
        cursor = await self._db.execute(
            "SELECT * FROM resource_profiles WHERE model_name = ? LIMIT 1",
            (model_name,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        return ModelProfile(
            model_name=d["model_name"],
            subsystem=d["subsystem"],
            backend=d["backend"],
            vram_mb=d.get("vram_mb", 0),
            ram_mb=d.get("ram_mb", 0),
            device=d.get("device", ""),
            quantization=d.get("quantization", ""),
            parameter_size=d.get("parameter_size", ""),
            family=d.get("family", ""),
            pipeline_type=d.get("pipeline_type", ""),
            times_seen=d.get("times_seen", 0),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
        )

    async def list_profiles(self) -> list[ModelProfile]:
        """All known model profiles."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT * FROM resource_profiles ORDER BY last_seen DESC LIMIT 500"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            result.append(ModelProfile(
                model_name=d["model_name"],
                subsystem=d["subsystem"],
                backend=d["backend"],
                vram_mb=d.get("vram_mb", 0),
                ram_mb=d.get("ram_mb", 0),
                device=d.get("device", ""),
                quantization=d.get("quantization", ""),
                parameter_size=d.get("parameter_size", ""),
                family=d.get("family", ""),
                pipeline_type=d.get("pipeline_type", ""),
                times_seen=d.get("times_seen", 0),
                first_seen=d.get("first_seen", ""),
                last_seen=d.get("last_seen", ""),
            ))
        return result

    async def get_history(self, hours: int = 24, limit: int = 100) -> list[ResourceSnapshot]:
        """Recent snapshots for charting."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT * FROM resource_snapshots "
            "WHERE timestamp > datetime('now', ? || ' hours') "
            "ORDER BY timestamp DESC LIMIT ?",
            (f"-{hours}", limit),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            models_json = json.loads(d.get("loaded_models_json", "[]"))  # noqa: F841
            result.append(ResourceSnapshot(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                gpu_total_mb=d.get("gpu_total_mb", 0),
                gpu_used_mb=d.get("gpu_used_mb", 0),
                gpu_free_mb=d.get("gpu_free_mb", 0),
                ram_total_mb=d.get("ram_total_mb", 0),
                ram_used_mb=d.get("ram_used_mb", 0),
                ram_free_mb=d.get("ram_free_mb", 0),
                models=[],  # Don't reconstruct TrackedModel from summary
            ))
        return result

    async def can_fit_model(self, model_name: str) -> tuple[bool, int]:
        """Check if a model can fit in available VRAM.

        Returns (can_fit, estimated_vram_mb). Uses stored profile if available,
        otherwise returns (True, 0) -- optimistic when unknown.
        """
        profile = await self.get_model_profile(model_name)
        if not profile or profile.vram_mb == 0:
            return True, 0  # Unknown -- let Ollama decide

        snap = self._last_snapshot
        if not snap or snap.gpu_free_mb == 0:
            snap = await self.collect()

        if snap.gpu_free_mb == 0:
            return True, profile.vram_mb  # No GPU info -- optimistic

        fits = profile.vram_mb <= snap.gpu_free_mb
        return fits, profile.vram_mb

    async def _update_profiles(self, models: list[TrackedModel]) -> None:
        """Standalone profile UPSERT — used by tests for direct semantic
        coverage of the UPSERT (preserve-non-zero, increment times_seen).

        The production persist path goes through ``_persist_atomic``,
        which writes profiles + snapshot in one transaction. Both paths
        share ``_UPSERT_PROFILES_SQL``.
        """
        if not self._db:
            return
        if not models:
            return
        rows = [
            (
                m.name, m.subsystem, m.backend, m.vram_mb, m.ram_mb,
                m.device, m.quantization, m.parameter_size, m.family,
                m.pipeline_type,
            )
            for m in models
        ]
        await self._db.executemany(_UPSERT_PROFILES_SQL, rows)
        await self._db.commit()

    # Lazy hourly pruner — set on first ``_store_snapshot`` call. Was a
    # per-snapshot DELETE + COMMIT, which doubled the fsync count on every
    # snapshot persist (each fsync is ~50-100ms on the live disk). At a
    # snapshot every 2-30s, that DELETE was running thousands of times a
    # day to remove a handful of expired rows. Now: prune once an hour.
    _last_prune_at: float = 0.0
    _PRUNE_INTERVAL_S: float = 3600.0  # 1 hour

    async def _store_snapshot(self, snap: ResourceSnapshot) -> None:
        """Standalone snapshot INSERT — used by tests for direct coverage
        of the snapshot row + history-readback round-trip.

        The production persist path goes through ``_persist_atomic``,
        which writes profiles + snapshot in one transaction. Both paths
        share ``_INSERT_SNAPSHOT_SQL``.
        """
        if not self._db:
            return
        models_summary = [
            {"name": m.name, "backend": m.backend, "vram_mb": m.vram_mb}
            for m in snap.models
        ]
        await self._db.execute(_INSERT_SNAPSHOT_SQL, (
            snap.timestamp.isoformat(),
            snap.gpu_total_mb, snap.gpu_used_mb, snap.gpu_free_mb,
            snap.ram_total_mb, snap.ram_used_mb, snap.ram_free_mb,
            len(snap.models), json.dumps(models_summary),
        ))
        await self._db.commit()

        await self._maybe_prune_snapshots()

    async def _maybe_prune_snapshots(self) -> None:
        """Run the >7d prune at most once per ``_PRUNE_INTERVAL_S`` seconds.

        The DELETE itself is cheap when called occasionally (small range
        scan on the timestamp index), but doing it on every snapshot
        persist meant a second COMMIT/fsync per call. Hourly is plenty for
        a 7-day rolling window.
        """
        if not self._db:
            return
        now = time.monotonic()
        if now - self._last_prune_at < self._PRUNE_INTERVAL_S:
            return
        self._last_prune_at = now
        try:
            await self._db.execute(
                "DELETE FROM resource_snapshots WHERE timestamp < datetime('now', '-7 days')"
            )
            await self._db.commit()
        except Exception:
            log.warning("resource_snapshot_prune_failed", exc_info=True)
