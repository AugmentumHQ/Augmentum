"""LlamaServerManager — subprocess lifecycle for llama-server.

Manages a single llama-server child process with start/stop/swap/health
operations, plus GGUF file discovery across configurable directories.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import psutil

from augmentum.config import settings
from augmentum.models.kv_session_manifest import KVSessionManifest
from augmentum.models.model_profile_cache import (
    ModelProfile,
    ModelProfileCache,
    peek_gguf_string_keys,
    peek_gguf_uint_keys,
    scan_gguf_header,
)
from augmentum.models.workspace_calibration import WorkspaceCalibration
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    # Annotation-only: the token cache lives outside the manager's
    # construction so its lifecycle can outlive a given subprocess.
    # Imported under TYPE_CHECKING to keep the runtime surface minimal.
    from augmentum.models.token_count_cache import TokenCountCache
    from augmentum.state.settings_store import SettingsStore


_LAST_LOAD_KEY_PREFIX = "engine.last_load."


# Filenames matching these patterns are vision projector GGUFs, not base
# models. Used to identify mmproj/CLIP companion files in :meth:`LlamaServerManager._find_paired_mmproj`.
_MMPROJ_FILENAME_RE = re.compile(r"^(mmproj|clip-)|[-_]mmproj[-_.]", re.IGNORECASE)


# Operator-declared projector pairings live in a sidecar JSON next to
# each base GGUF. This is the contract our ``/api/models/.../projector``
# endpoint writes, and it's what the load path reads to find the right
# projector. Mirrors Jan's ``model.json`` / Ollama's manifest layer
# approach -- pairing is operator-declared, not heuristically guessed.
_PROJECTOR_SIDECAR_SUFFIX = ".augmentum-projector.json"


def _projector_sidecar_path(base_path: str) -> Path:
    """Return the sidecar path for ``base_path`` (sibling, suffix swap)."""

    return Path(base_path).with_suffix(_PROJECTOR_SIDECAR_SUFFIX)


class ModelPinnedError(RuntimeError):
    """Raised when a model swap is requested while the currently-loaded
    model is pinned by another consumer.

    Long-running consumers (bug_finder with thinking enabled, variance
    benches, long agentic loops) acquire a pin via
    ``LlamaServerManager.pin_model`` to prevent sibling services from
    evicting the in-use model mid-flight. The pin is refcounted so
    multiple concurrent consumers of the same model share the lock
    cleanly; the swap path raises this exception when a *different*
    model is requested while the pinned id is still loaded.

    Caller policy: most sibling services should treat this as a
    transient backoff signal (try again later when the pin is released);
    a few may want to surface it to the user as "the local model is busy
    with another task, try again in a few minutes."
    """

    def __init__(
        self, *, pinned_model: str, requested_model: str, refcount: int,
    ) -> None:
        super().__init__(
            f"llama-server model {pinned_model!r} is pinned "
            f"(refcount={refcount}); swap to {requested_model!r} blocked",
        )
        self.pinned_model = pinned_model
        self.requested_model = requested_model
        self.refcount = refcount


def read_projector_sidecar(base_path: str) -> str:
    """Read the operator-declared mmproj path for ``base_path``, if any.

    Returns ``""`` when no sidecar exists, the sidecar is malformed, or
    the declared path no longer exists on disk. Callers should treat
    missing/invalid sidecars as "no pairing" -- never crash on them.
    """

    side = _projector_sidecar_path(base_path)
    try:
        with open(side, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    mmproj = str(data.get("mmproj_path") or "").strip()
    if not mmproj or not Path(mmproj).is_file():
        return ""
    return mmproj


def write_projector_sidecar(base_path: str, mmproj_path: str) -> None:
    """Persist (or clear) the operator-declared mmproj for ``base_path``.

    Pass an empty ``mmproj_path`` to delete the sidecar (unpair).

    Use when:
    - The UI's "Pair projector" affordance has confirmed a candidate.
      Caller must run :func:`validate_mmproj_pair` first to ensure the
      pairing won't crash llama-server at load time.
    """

    side = _projector_sidecar_path(base_path)
    cleaned = str(mmproj_path or "").strip()
    if not cleaned:
        with contextlib.suppress(FileNotFoundError):
            side.unlink()
        return
    payload = {"mmproj_path": cleaned, "set_at_unix_ms": int(time.time() * 1000)}
    side.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Invalid operator-declared projector pairings that have already been
# logged. The catalog / resource-profile evaluation re-resolves projectors
# on every scan, so a STABLE mismatch (e.g. a stale UI pairing whose dims
# don't match) would re-warn every few seconds forever. Warn ONCE per
# (model, mmproj, reason); repeats drop to debug. Bounded: one entry per
# distinct bad pairing on disk. Cleared only on process restart, which is
# also when a corrected pairing should re-surface if still wrong.
_warned_invalid_sidecars: set[tuple[str, str, str]] = set()


def _read_base_embedding_length(base_path: str, profile: ModelProfile | None) -> int:
    """Return the base model's ``<arch>.embedding_length``, or 0.

    Prefer the cached profile (already paid the full-scan cost during
    discovery). Fall back to a targeted GGUF header peek so unprofiled
    models can still be validated.
    """

    if profile and profile.n_embed:
        return int(profile.n_embed)
    # No profile yet -- read general.architecture, then the matching
    # ``<arch>.embedding_length``.
    strs = peek_gguf_string_keys(base_path, {"general.architecture"})
    arch = (strs.get("general.architecture") or "").strip()
    if not arch:
        return 0
    nums = peek_gguf_uint_keys(base_path, {f"{arch}.embedding_length"})
    return int(nums.get(f"{arch}.embedding_length", 0))


def _read_mmproj_projection_dim(mmproj_path: str) -> int:
    """Return the mmproj's ``clip.vision.projection_dim`` (== expected base n_embd), or 0."""

    nums = peek_gguf_uint_keys(mmproj_path, {"clip.vision.projection_dim"})
    return int(nums.get("clip.vision.projection_dim", 0))


def validate_mmproj_pair(
    base_path: str,
    mmproj_path: str,
    profile: ModelProfile | None = None,
) -> tuple[bool, str]:
    """Verify a base/mmproj pair will not crash llama-server at startup.

    Use when:
    - About to write a projector sidecar (operator confirm step).
    - About to launch llama-server with ``--mmproj``.
    - Enumerating candidates for the UI's projector picker.

    The check compares the base model's ``embedding_length`` (or
    cached ``profile.n_embed``) to the mmproj's
    ``clip.vision.projection_dim``. A mismatch is exactly the condition
    that triggers llama-server's ``mtmd_init_from_file: mismatch
    between text model (n_embd=X) and mmproj (n_embd=Y)`` fatal error.

    Returns:
    - ``(True, "")`` when the dims agree (or one side is unreadable and
      we'd rather let the runtime decide than block a possibly-valid
      pair -- conservative path documented in the reason string).
    - ``(False, reason)`` when we can confidently say the load will
      fail. Reason is human-readable and surfaced to the UI.
    """

    if not mmproj_path or not Path(mmproj_path).is_file():
        return (False, "projector file does not exist")
    base_dim = _read_base_embedding_length(base_path, profile)
    proj_dim = _read_mmproj_projection_dim(mmproj_path)
    if not base_dim or not proj_dim:
        # One side unreadable -- don't block; the runtime will catch
        # any real mismatch.
        return (True, "")
    if base_dim != proj_dim:
        return (
            False,
            f"dim mismatch: base embedding_length={base_dim} but "
            f"projector projection_dim={proj_dim}",
        )
    return (True, "")

# Strips trailing quant / format tags so two stems that differ only by quant
# (e.g. ``Qwen3.6-35B-A3B-UD-Q4_K_XL`` and ``Qwen3.6-35B-A3B``) compare equal.
_QUANT_SUFFIX_RE = re.compile(
    r"[._-](ud[._-])?(i?q\d[a-z_0-9]*|fp?16|bf16|f32)\b.*$",
    re.IGNORECASE,
)


def _normalize_stem_for_match(s: str) -> str:
    """Lowercase, strip quant/format suffixes, collapse separators to '-'."""
    s = _QUANT_SUFFIX_RE.sub("", s.lower())
    return re.sub(r"[._-]+", "-", s).strip("-")


def _looks_like_specific_model_claim(name: str) -> bool:
    """True when ``name`` looks like a concrete model identifier.

    A specific claim has a parameter-size token (``31B``, ``8b``, ``1.2b``)
    and isn't an obvious tooling-generated anonymous name. We use this to
    avoid treating ``general.name = "Unsloth_Gguf_Jmqiwvw4"`` (which
    Unsloth's quant tools emit) as a real claim about which base model the
    projector pairs with.
    """
    if not name or len(name) < 4:
        return False
    n = name.lower()
    if "unsloth_gguf" in n or "unsloth-gguf" in n:
        return False
    # Param-size pattern: digits (optionally decimal) followed by 'b'
    return bool(re.search(r"\d+(?:\.\d+)?\s*b\b|\d+(?:\.\d+)?b[-_]?[a-z]", n))


def _model_family_key(s: str) -> str:
    """Reduce an arch / name / projector_type string to a coarse family key.

    Examples
    --------
    >>> _model_family_key("qwen35moe")
    'qwen3'
    >>> _model_family_key("qwen3vl_merger")
    'qwen3'
    >>> _model_family_key("gemma3")
    'gemma3'
    >>> _model_family_key("Gemma-4-31B-It")
    'gemma4'

    A Qwen 3 VL projector and a Qwen 3.5 / 3.6 base both reduce to
    ``qwen3``, which is what we want when deciding "are these from the
    same architecture family." We deliberately take the FIRST digit
    after the alpha prefix — VL/MoE/version tags after that point
    don't change projector compatibility within the family.
    """
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    match = re.match(r"([a-z]+?)(\d)", s)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return s[:16]

# Common GGUF storage locations — checked in addition to configured model_dirs.
_COMMON_GGUF_PATHS: list[str] = []

def _build_common_paths() -> list[str]:
    """Build list of common GGUF paths for this platform.

    Only includes paths that are likely to contain user-managed GGUF
    files. Skips generic system paths and the primary model_dir
    (which is already in the scan list).
    """
    paths = []
    if os.name == "nt":
        # Windows: LM Studio and HuggingFace Hub cache
        paths.extend([
            os.path.expandvars(r"%USERPROFILE%\.cache\lm-studio\models"),
            os.path.expandvars(r"%USERPROFILE%\.cache\huggingface\hub"),
        ])
    else:
        # Linux/macOS: only user-specific cache dirs
        paths.extend([
            os.path.expanduser("~/.cache/lm-studio/models"),
            os.path.expanduser("~/.cache/huggingface/hub"),
        ])
    return paths

_COMMON_GGUF_PATHS = _build_common_paths()

# Prefill-progress canary from llama-server's `slot print_timing` log
# line. Format: ``slot print_timing: id 3 | task 0 | prompt processing,
# n_tokens = 4096, progress = 0.05, t = 42.52 s / 96.33 tokens per second``.
# Five captured groups: slot_id, tokens_done, progress (0-1), elapsed_s, tps.
#
# slot_id matters for KV affinity tracking — under ``--kv-unified`` /
# ``--parallel -1`` the response stream chunks don't always carry an
# id_slot field, so we fall back to this log-line slot id to drive
# _claim_slot. Without that, the chat backend can't track which slot
# holds which session, and every subsequent turn reports cold even
# though llama-server's KV cache is actually warm.
_PREFILL_PROGRESS_RE = re.compile(
    r"slot\s+print_timing:\s*id\s+(\d+)\s*\|.*?"
    r"prompt processing,\s+n_tokens\s*=\s*(\d+),\s+progress\s*=\s*([0-9.]+),"
    r"\s+t\s*=\s*([0-9.]+)\s*s\s*/\s*([0-9.]+)\s+tokens per second"
)

# Architectures whose fused tensors (e.g. Gated Delta Net in the
# Qwen-Next hybrid-SSM family) require either layer 0 + the fused
# tensor co-located on GPU OR a full-CPU load. Intermediate
# ``--n-gpu-layers`` values place layer 0 on CPU while the GDN tensor
# stays on CUDA0, sched_reserve aborts, and the subprocess exits 1.
# The runtime parser at line ~1122 already catches this from stderr
# AFTER the failed boot; this set lets us emit the same actionable
# guidance at load-plan time, before the wasted subprocess spin-up.
#
# Match against the lowercased ``profile.architecture`` GGUF field.
# Add other architectures here as upstream lands more
# hybrid-SSM / fused-tensor layouts (e.g. Mamba-Llama hybrids).
_PARTIAL_OFFLOAD_INCOMPATIBLE_ARCHS: frozenset[str] = frozenset({
    "qwen3next",
})

# Filesystem types that indicate a path goes through a host-bridge
# layer (Docker Desktop) or a network protocol. These load GGUFs
# significantly slower than native ext4/xfs/btrfs because each
# read crosses a translation layer:
#   9p           — WSL2 ↔ Windows host
#   fuse.grpcfuse, virtiofs — Docker Desktop (older / newer)
#   osxfs        — Docker Desktop on macOS (legacy)
#   drvfs        — WSL1 ↔ Windows
#   cifs/smbfs/nfs — network filesystems
# A path on `ext4`/`overlay`/`xfs`/`btrfs`/`tmpfs` is full speed,
# regardless of which container path prefix it lives under.
_SLOW_FS_TYPES: frozenset[str] = frozenset({
    "9p", "fuse.grpcfuse", "virtiofs", "osxfs", "drvfs",
    "cifs", "smbfs", "smb3", "nfs", "nfs4",
})

# Cache the parsed mountinfo for this many seconds. Bind mounts don't
# come and go during a normal Augmentum session and we don't want to
# re-parse a 50-line file on every dashboard refresh.
_MOUNTINFO_CACHE_TTL_S = 30.0
_mountinfo_cache: tuple[float, list[tuple[str, str]]] | None = None


def _read_mountinfo() -> list[tuple[str, str]]:
    """Parse /proc/self/mountinfo into [(mount_point, fs_type), ...].

    Sorted by descending mount-point length so the longest-prefix
    match wins in classify_mount_fs (e.g. /data/host-models/foo
    matches its bind mount before falling through to /). Returns
    an empty list if mountinfo isn't readable (e.g. running outside
    Linux or with /proc unavailable).

    Format reference (see proc(5) mountinfo):
      36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,…
                              ^mount point         ^fs type
    """
    global _mountinfo_cache
    now = time.monotonic()
    if _mountinfo_cache is not None and now - _mountinfo_cache[0] < _MOUNTINFO_CACHE_TTL_S:
        return _mountinfo_cache[1]

    entries: list[tuple[str, str]] = []
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                # Split on " - " — mount-options field can contain spaces
                # in theory, but the kernel guarantees the " - " separator
                # is unique.
                parts = line.split(" - ", 1)
                if len(parts) != 2:
                    continue
                left = parts[0].split()
                right = parts[1].split()
                if len(left) < 5 or not right:
                    continue
                entries.append((left[4], right[0]))
    except OSError:
        return []

    entries.sort(key=lambda e: len(e[0]), reverse=True)
    _mountinfo_cache = (now, entries)
    return entries


def classify_mount_fs(path: str) -> str:
    """Return the filesystem type that backs ``path``.

    Walks /proc/self/mountinfo and returns the fs type of the
    longest-prefix mount entry containing the path. Empty string if
    mountinfo isn't readable. Used to label a directory as slow only
    when its actual fs is bridged/network — not based on whether the
    container path prefix happens to look host-mount-ish.
    """
    if not path:
        return ""
    norm = path.rstrip("/") or "/"
    for mnt, fs in _read_mountinfo():
        mnt_norm = mnt.rstrip("/") or "/"
        if norm == mnt_norm or norm.startswith(mnt_norm + "/") or mnt_norm == "/":
            return fs
    return ""


def is_slow_filesystem(fs: str) -> bool:
    """True iff the named filesystem goes through a host bridge or
    network protocol — see _SLOW_FS_TYPES for the full set.
    """
    return fs in _SLOW_FS_TYPES


log = get_logger(__name__)

# Matches a llama.cpp memory-report line:
#   "load_tensors:   CUDA0 model buffer size =  4096.00 MiB"
# Deliberately tolerant: ANY "<words> buffer" component (not a fixed list)
# and any binary/SI unit. The previous tight regex enumerated only
# model/KV/RS/compute/output buffers in MiB, so a model whose buffers were
# named differently (e.g. some hybrid-SSM architectures) matched NOTHING —
# its VRAM then went unattributed and the resource ledger fell back to a
# (often wildly wrong) plan estimate. "buffer" is kept INSIDE the component
# group because downstream lookups key on e.g. ``compute_buffer``.
_MEMORY_COMPONENT_LINE_RE = re.compile(
    r":\s+(?P<location>[A-Za-z0-9_ ()+./-]+?)\s+"
    r"(?P<component>(?:[A-Za-z0-9_]+[ _]){1,4}buffer)\s+"
    r"size\s*=\s*(?P<size>[0-9.]+)\s+(?P<unit>[KMGT]i?B)\b",
    re.IGNORECASE,
)

# llama.cpp prints MiB, but normalize defensively so a build that switches
# to GiB for large buffers doesn't silently under/over-count.
_MEM_UNIT_TO_MIB: dict[str, float] = {
    "kib": 1.0 / 1024, "mib": 1.0, "gib": 1024.0, "tib": 1024.0 * 1024,
    "kb": 1.0 / 1024, "mb": 1.0, "gb": 1024.0, "tb": 1024.0 * 1024,
}


def parse_llama_memory_from_lines(lines) -> tuple[int, int]:
    """Sum ``(vram_mib, ram_mib)`` from llama.cpp buffer-size log lines.

    Pure + reusable so there's ONE source of truth for the llama-server memory
    banner format. The engine manager parses these lines from the subprocess it
    owns (``_capture_actual_memory_from_line``); the resource panel parses the
    SAME lines from a sidecar llama-server container's logs to attribute the
    classifier / vision siblings' VRAM (spec §4.6 rung B — ride data already
    produced, capture once at load).

    Mirrors ``_actual_memory_snapshot`` exactly: dedup by (component, location)
    — last wins, matching a reload within the same window — then sum per
    location across components and bucket by VRAM/RAM scope.
    """
    comp_loc: dict[tuple[str, str], float] = {}
    for line in lines:
        match = _MEMORY_COMPONENT_LINE_RE.search(str(line))
        if not match:
            continue
        try:
            size = float(match.group("size"))
        except (TypeError, ValueError):
            continue
        if size < 0:
            continue
        unit = (match.group("unit") or "MiB").lower()
        size_mib = size * _MEM_UNIT_TO_MIB.get(unit, 1.0)
        location = LlamaServerManager._normalize_memory_location(match.group("location"))
        component = LlamaServerManager._normalize_memory_component(match.group("component"))
        if not location or not component:
            continue
        comp_loc[(component, location)] = size_mib  # overwrite dup → last wins

    loc_totals: dict[str, float] = {}
    for (_component, location), size_mib in comp_loc.items():
        loc_totals[location] = loc_totals.get(location, 0.0) + size_mib

    vram = ram = 0.0
    for location, size_mib in loc_totals.items():
        scope = LlamaServerManager._memory_location_scope(location)
        if scope == "vram":
            vram += size_mib
        elif scope == "ram":
            ram += size_mib
    return int(round(vram)), int(round(ram))


class ProcessState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPING = "stopping"


class LlamaServerManager:
    """Manages a single llama-server child process."""

    # Sync GPU info cache TTL. Short enough that ``status()`` polls
    # don't show stale VRAM numbers, long enough that consecutive
    # load-plan helpers in a single ``start()`` invocation reuse the
    # warm cache populated by ``_query_gpu_info_async``.
    _GPU_INFO_TTL_S: float = 2.0

    # Hard ceiling on the auto-sized ``--cache-ram`` host KV cache.
    # This cache only accelerates SESSION SWITCHING (one memcpy instead
    # of a re-prefill); it is pure anonymous host memory that nothing
    # reclaims under pressure. On 2026-07-25 the container-blind 25%
    # heuristic sized it at 23.6 GiB and helped drive a 128 GB machine
    # into a forced restart. 8 GiB holds several full session states of
    # any realistic model; beyond that the marginal hit-rate gain does
    # not justify being the largest allocation on the box. Override
    # deliberately with the ``engine_cache_ram_mib`` setting.
    _CACHE_RAM_ABSOLUTE_CAP_MIB: int = 8192

    # Baseline VRAM reservation for OS / driver / small driver-side
    # allocations llama.cpp doesn't pre-account for. Subtracted from
    # the autofit budget BEFORE per-layer fit so we never claim the
    # last byte of VRAM. 1 GiB is conservative on consumer cards
    # (24 GB has 4% overhead, 48-80 GB cards have <2%); going-over
    # VRAM cliff (5-10× slowdown via PCIe shared memory fallback)
    # is much worse than leaving 1 GiB on the table.
    _FIRST_FIT_BASELINE_RESERVE_BYTES: int = 1024 * 1024 * 1024

    # Fused expert tensors per MoE layer. Modern GGUFs (Qwen3, Mixtral,
    # DeepSeek, GLM, Kimi, …) stack ALL of a layer's experts into a small
    # set of tensors — ``ffn_gate_exps`` / ``ffn_up_exps`` / ``ffn_down_exps``
    # (~3 per MoE layer) — instead of one tensor per expert. The GGUF scanner
    # only matches those fused names, so ``n_expert_tensors`` scales with the
    # MoE-LAYER count (~3×), NEVER with ``expert_count``. Used to recover the
    # MoE-layer count from the aggregate tensor tally. Variants that fuse
    # gate+up (2 per layer) divide slightly low → a few fewer experts on GPU,
    # which errs on the safe (no-overflow) side.
    _FUSED_EXPERT_TENSORS_PER_LAYER: int = 3

    # Compute-buffer reservation for prefill / decode workspace,
    # branched by Flash Attention. With FA enabled, llama.cpp doesn't
    # materialize the full attention scores matrix — the graph runs
    # tiled and the working set is much smaller. With FA disabled the
    # softmax(QK^T) scratch is allocated in full per layer, roughly
    # doubling the workspace under typical batch sizes.
    #
    # Mirrors the asymmetry already used in ``build_load_plan`` (384
    # vs 640 MiB) so the autofit allocator and the displayed plan
    # agree on what they're counting. Pre-T2-6 the allocator used a
    # single 512 MiB blend that under-reserved on FA-off loads (a
    # frequent OOM trigger on Pascal cards where FA isn't supported)
    # and over-reserved on FA-on loads (leaving ~150 MiB on the table
    # that the plan would have given back to layers).
    #
    # Reference: Ollama's ``llm/memory.go`` uses a similar two-estimate
    # model — graphPartialOffload vs graphFullOffload — though the
    # exact byte budgets differ since their accounting also covers the
    # PyTorch path. Numbers here come from llama.cpp peak-residency
    # measurements at batch 512 across Q4_K_M / Q8_0 dense models.
    _FIRST_FIT_COMPUTE_RESERVE_FA_BYTES: int = 384 * 1024 * 1024
    _FIRST_FIT_COMPUTE_RESERVE_NO_FA_BYTES: int = 640 * 1024 * 1024

    @classmethod
    def _compute_reserve_baseline_bytes(cls, flash_attn: bool) -> int:
        """Uncalibrated compute workspace budget keyed off FA.

        Pure function — same input always yields the same output.
        Used as the BASELINE that the calibration multiplies. Tests
        and callers that want the raw constant (e.g. for a parity
        check between autofit and the displayed plan) read from
        here.
        """
        return (
            cls._FIRST_FIT_COMPUTE_RESERVE_FA_BYTES
            if flash_attn
            else cls._FIRST_FIT_COMPUTE_RESERVE_NO_FA_BYTES
        )

    @staticmethod
    def _calibration_bucket_for(flash_attn: bool) -> str:
        """Stable bucket key for the workspace calibration store."""
        return "fa_on" if flash_attn else "fa_off"

    def _compute_reserve_bytes(self, flash_attn: bool) -> int:
        """Per-load compute workspace budget keyed off FA, with the
        T2-7 self-calibrating multiplier applied.

        Once we have ``MIN_SAMPLES_TO_TRUST`` real-load observations
        the calibration scales the baseline. Below that threshold
        ``get_factor`` returns 1.0 — the displayed plan and autofit
        reserve match the original constants exactly. Single source
        of truth for both the autofit allocator and
        ``build_load_plan``'s displayed peak so the two never drift.
        """
        baseline = self._compute_reserve_baseline_bytes(flash_attn)
        bucket = self._calibration_bucket_for(flash_attn)
        factor = self._workspace_calibration.get_factor(bucket)
        return int(baseline * factor)

    def __init__(
        self,
        llama_server_path: str = "/usr/local/bin/llama-server",
        backend_port: int = 8091,
        model_dir: str = "/models",
        extra_model_dirs: list[str] | None = None,
        gpu_layers: int = 99,
        ctx_size: int = 32384,
        batch_size: int = 512,
        profile_cache_dir: str = "",
        kv_manifest_db: str = "",
        kv_ttl_days: int = 2,
        kv_narrative_ttl_days: int = 7,
        kv_max_snapshots_per_model: int = 8,
        kv_auto_pin_narrative: bool = False,
        kv_warm_on_start: bool = True,
        force_single_slot: bool = False,
    ) -> None:
        self._llama_server_path = llama_server_path
        self._backend_port = backend_port
        self._model_dir = model_dir
        self._gpu_layers = gpu_layers
        self._ctx_size = ctx_size
        self._batch_size = batch_size
        # Manager-local opt-out from the multi-slot warm-tier args.
        # Sibling subprocesses (vision aux, future helpers) don't
        # benefit from multi-slot because they handle sync single-
        # request workloads — multi-slot just budgets a large
        # ``--cache-ram`` pool for evicted KV that never gets used.
        # The primary engine leaves this False and follows
        # ``settings.engine_multislot_enabled`` like before.
        self._force_single_slot = force_single_slot
        self.current_ctx_size: int = ctx_size
        self.current_gpu_layers: int = gpu_layers
        self.current_batch_size: int = batch_size
        self.current_kv_cache_type: str = ""
        self.current_flash_attn: bool = True
        self.current_draft_model: str = ""
        self.current_draft_max: int = 5
        self.current_draft_ctx_size: int = 2048
        self.current_draft_gpu_layers: int = 999
        self.current_draft_min: int = 1
        self.current_draft_p_min: float = 0.75
        self.current_gpu_layers_mode: str = "auto"
        self.kv_ttl_days: int = kv_ttl_days
        self.kv_narrative_ttl_days: int = kv_narrative_ttl_days
        self.kv_max_snapshots_per_model: int = kv_max_snapshots_per_model
        self.kv_auto_pin_narrative: bool = kv_auto_pin_narrative
        self.kv_warm_on_start: bool = kv_warm_on_start
        # Set by ``_warm_top_session`` after a successful restart-hydration
        # so the backend can short-circuit its own restore on the first
        # matching request. Cleared on stop/swap.
        self._warm_session_key: str = ""
        # Reverse link to the LlamaCppBackend serving this manager (set
        # by the backend's __init__). Lets the post-READY boot warm run
        # the resume ladder (restore→replay→cold) instead of the legacy
        # restore-only walk. None in tests / before the backend exists.
        self._engine_backend = None
        # Session keys the ladder replay-warmed without occupancy
        # bookkeeping (unpinned multi-slot prewarms). Consumed one-shot
        # by _manage_slot's tier telemetry so a replay-warmed first turn
        # isn't misreported as genuinely cold.
        self._replay_warmed_keys: set[str] = set()
        # Handle for the background boot-warm task so stop() can cancel
        # a warm loop that's still replaying when the model goes away.
        self._kv_warm_task: asyncio.Task | None = None

        # Per-model pin refcounts. While ``_pinned_models[model_id] > 0``,
        # any swap request *away from* that model raises
        # ``ModelPinnedError`` instead of unloading it. Used by long-
        # running consumers (bug_finder runs with thinking enabled,
        # variance benches, long agentic loops) that can't survive a
        # mid-flight cold-load: a sibling service requesting a different
        # model would otherwise evict the in-use model, killing the
        # consumer's budget on the reload. Refcounted so multiple
        # concurrent consumers of the same model share a pin cleanly.
        # Keyed by ``model_id`` (the GGUF basename llama-server reports,
        # NOT a friendly name) — comparison happens against
        # ``self.model_id`` in the swap path.
        self._pinned_models: dict[str, int] = {}

        # Latched when the subprocess emits a sched_reserve warning about
        # a fused Gated Delta Net tensor being mis-placed across CPU/CUDA
        # devices. These architectures (Qwen 3.5/3.6 hybrid SSM blocks)
        # don't tolerate partial GPU offload — reducing `n_gpu_layers`
        # leaves layer 0 on CPU while the fused tensor stays on CUDA0
        # and the model aborts during sched_reserve.
        #
        # `_start_with_oom_backoff` reads this between retry attempts to
        # short-circuit the OOM cascade (no amount of layer reduction
        # will rescue an incompatible-with-partial-offload model). Reset
        # to False at the start of every ``start()`` so a fresh attempt
        # gets a clean slate.
        self._partial_offload_incompatible: bool = False

        # Most-recent prefill progress parsed from llama-server's
        # ``slot print_timing: ... prompt processing, n_tokens = X,
        # progress = Y, t = Z s / TPS tokens per second`` log line.
        # Updated by the status parser as new lines arrive; consumed by
        # the frontend's poll loop to render a live progress bar during
        # the prefill stage (long-context turns can spend 30-180s here
        # with no other visible indication of progress).
        #
        # Schema: { "tokens_done": int, "progress": float (0-1),
        # "elapsed_s": float, "tps": float, "updated_at": float }.
        # ``None`` until the first progress line of a load arrives, or
        # after the response completes / a new model loads.
        self._prefill_progress: dict[str, Any] | None = None

        # Most-recent model-load progress snapshot, set at the start of
        # ``start()`` and cleared when the server reaches READY (or the
        # attempt fails). llama.cpp doesn't emit a reliable percentage
        # during the tensor-mmap + GPU-upload phase, so we surface an
        # elapsed/expected pair where ``expected_s`` is the median of
        # recent successful loads for the same model id (falling back
        # to a coarse file-size estimate on first load). The frontend's
        # poll loop renders a soft progress bar from this snapshot so
        # the user sees "Loading deepseek-v3 · 14s of ~30s" instead of
        # the "stream stalled" banner during a normal 30-60s cold start.
        #
        # Schema: { "model_id": str, "model_path": str, "started_at":
        # float, "elapsed_s": float, "expected_s": float, "size_bytes":
        # int, "stage_label": str }.
        self._load_progress: dict[str, Any] | None = None

        # Rolling history of recent successful load durations per model
        # id, used to estimate ``expected_s`` for the next load of that
        # model. Capped per id so the median stays representative of
        # the current install rather than absorbing a long tail of one-
        # off slow loads from an unrelated past condition.
        self._load_duration_history: dict[str, list[float]] = {}

        # Instance attributes
        self.kv_cache_type: str = ""
        self.draft_model: str = ""
        self.draft_max: int = 5
        self.draft_ctx_size: int = 2048
        self.draft_gpu_layers: int = 999
        self.draft_min: int = 1
        self.draft_p_min: float = 0.75
        # MTP (multi-token prediction) self-speculation — upstream
        # llama.cpp PR #22673 (merged 2026-05-16). When True the runtime
        # passes ``--spec-type draft-mtp --spec-draft-n-max N`` to
        # llama-server, using the model's own MTP heads as the
        # speculation source (no separate draft model). Mutually
        # exclusive with ``draft_model`` — if both are set, MTP wins
        # (the model has MTP heads, so the external draft is redundant).
        # Requires ``--parallel 1`` at the binary level (enforced in
        # the argv assembly). Requires a binary at commit 2555826+ or
        # llama-server rejects ``--spec-type`` as an unknown arg.
        self.mtp_enabled: bool = False
        self.mtp_n_max: int = 2
        # Cumulative MTP acceptance counters scraped from llama-server
        # stderr (PR #22673 emits ``draft acceptance rate = X
        # (A accepted / G generated)`` per stats interval). Read by
        # ``LlamaCppBackend._log_performance`` to stamp each
        # ``engine_perf`` event with the latest counters so operators
        # can see MTP health per turn without separately tailing
        # llama-server logs. ``rate`` mirrors the most recent line;
        # ``accepted`` / ``generated`` are running totals over the
        # current subprocess lifetime — useful for trends, but not a
        # per-request number (upstream doesn't emit one). Reset on
        # each ``start()`` so a model swap doesn't carry over stats.
        self._mtp_last_log: dict[str, float | int] = {}
        self.flash_attn: bool = True
        self.cont_batching: bool = True
        self.process: asyncio.subprocess.Process | None = None
        self.state: ProcessState = ProcessState.IDLE
        self.model_id: str = ""
        self.model_path: str = ""
        self._pinned_sessions: set[str] = set()
        self._slot_dir: str = ""
        # Whether the CURRENTLY-loaded model was launched with per-slot
        # save/restore available (single-slot, non-unified KV cache). Set in
        # _build_cli_args; the backend checks it to skip slot I/O cleanly on
        # multi-slot/--kv-unified models (which use --ctx-checkpoints instead).
        self._slot_save_supported: bool = False
        self._drain_tasks: list[asyncio.Task] = []
        self._last_profile: ModelProfile | None = None
        self._last_load_plan: dict[str, Any] | None = None
        # Settings store handle (attached after construction via
        # ``set_settings_store``). The lazy-load path consults
        # ``app_settings["engine.last_load.<model_id>"]`` so an auto-restart
        # replays the same ctx/GPU layout the user last confirmed via the UI,
        # instead of falling back to engine defaults. Optional — when unset
        # (tests, standalone usage) the lazy-load path simply uses defaults.
        self._settings_store: SettingsStore | None = None
        self._last_crashed_model: str = ""
        self._last_crash_code: int = 0
        self._actual_memory_components: dict[str, dict[str, float]] = {}
        self._actual_memory_loaded: bool = False
        # Buffer-ish log lines the memory regex did NOT match — surfaced in
        # the empty-capture warning so an unparseable build/arch format is
        # visible (and fixable) rather than silently degrading to estimates.
        self._mem_capture_misses: list[str] = []
        # Effective ``--n-gpu-layers`` of the in-flight start attempt, and
        # the value llama-server reported when it aborted ``common_fit_params``
        # ("n_gpu_layers already set by user to N, abort"). Used to retry a
        # failed load with a stepped-down offload instead of just failing.
        self._last_effective_gpu_layers: int = 0
        self._fit_abort_layers: int | None = None
        self._session_manifest: KVSessionManifest | None = None
        if kv_manifest_db:
            try:
                self._session_manifest = KVSessionManifest(kv_manifest_db)
            except Exception as exc:
                log.warning("kv_manifest_init_failed", path=kv_manifest_db, error=str(exc))
        # Segment-level token count cache — attached by the proxy layer
        # after construction via ``set_token_cache``. Read by the
        # backend through the ``token_cache`` property. Optional because
        # some tests and standalone usage don't configure one; callers
        # already treat ``None`` as "caching disabled".
        self._token_cache: TokenCountCache | None = None
        self._load_pinned_sessions()
        self._start_time: float | None = None

        # Timeouts
        self.health_timeout: float = 120.0  # seconds to wait for model load
        self.idle_timeout: float = 600.0  # seconds before auto-unload (10 min default)
        self._last_request_time: float = 0.0
        self._idle_task: asyncio.Task | None = None

        # In-flight request tracking. The idle monitor uses this to skip
        # unloads while requests are active. Prior to 2026-04-27 the monitor
        # only checked elapsed time since the last ``touch()`` call, which
        # fires once at request entry — meaning a long prefill (e.g. 90k
        # context taking 5+ minutes) could exceed the idle window mid-
        # generation, kill the subprocess out from under the live request,
        # and surface as ``peer closed connection`` to the user. Backends
        # wrap ``chat()`` / ``chat_stream()`` in ``request_in_flight()``
        # below so the counter goes >0 for the entire request lifecycle —
        # model load, slot save/restore, prefill, thinking, generation, and
        # post-response checkpoint save. Only when the last in-flight
        # request finishes does the 10-minute idle countdown actually start.
        self._in_flight_count: int = 0

        # Start idempotency. ``manager.start(path)`` can be called
        # concurrently from two paths: the explicit
        # ``POST /api/engine/v2/models/load`` route AND the chat path's
        # automatic ``backend._ensure_server`` recovery. Without
        # coalescing, the second caller would see ``state != READY``
        # and decide to start its own subprocess, killing the first
        # in-flight load mid-flight and forcing a full re-load.
        # Symptom observed 2026-05-06: user clicks "Load model", sends
        # a message before the load completes, sees the model "unload"
        # and "reload" — actually the chat path's ensure_server
        # racing the explicit load.
        # ``_starting_future`` is the in-flight start's promise.
        # Concurrent callers requesting the same model_path await it
        # instead of restarting; different model_path falls through to
        # kill-and-restart (the caller wanted a different model).
        self._starting_future: asyncio.Future | None = None
        self._starting_path: str = ""

        # GPU info cache. ``_query_gpu_info`` shells out to nvidia-smi,
        # which blocks the event loop for ~100-500 ms (5 s timeout
        # ceiling). ``start()`` pre-warms this via
        # ``_query_gpu_info_async`` BEFORE the synchronous load-plan
        # helpers (``_get_vram_bytes``, ``_autofit_gpu_layers_for``,
        # ``_cap_ctx_for_vram``, ``build_load_plan``) run, so the
        # subprocess executes off-loop. Periodic sync callers
        # (``status()``) honor a short TTL so they don't re-shell on
        # every poll.
        self._gpu_info_cache: dict | None = None
        self._gpu_info_cached_at: float = 0.0

        # Model directories
        self.model_dirs: list[str] = [model_dir]
        if extra_model_dirs:
            self.model_dirs.extend(extra_model_dirs)

        # Profile cache. Construction is fail-soft: it tries the requested
        # path, then ``/data/model_profiles``, then a tmp fallback, and
        # finally falls back to in-memory only if every path is unwritable.
        # Never raises — anything that crashed init here would cascade up
        # to ``engine_v2_init_failed`` and silently break the model
        # manager UI.
        cache_dir = profile_cache_dir or os.path.join(model_dir, ".profiles")
        self.profile_cache = ModelProfileCache(cache_dir=cache_dir)

        # T2-7: self-calibrating workspace estimator. Persist next to
        # the profile cache's *resolved* directory so they share one
        # writable location. When the profile cache is disk-disabled
        # entirely we point calibration at the original requested path —
        # it'll fail-soft on save (logs and continues) and the in-memory
        # EMA still works for the lifetime of the process.
        calibration_dir: Path = (
            self.profile_cache.cache_dir
            if self.profile_cache.cache_dir is not None
            else Path(cache_dir)
        )
        self._workspace_calibration = WorkspaceCalibration(
            calibration_dir / "workspace_calibration.json"
        )
        # Compute reserve we used on the most recent successful load —
        # captured at plan time so the "model loaded" handler can
        # compute observed/predicted without re-running build_load_plan.
        self._last_predicted_compute_reserve_bytes: int = 0
        self._last_predicted_compute_bucket: str = ""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._backend_port}"

    @property
    def token_cache(self) -> TokenCountCache | None:
        """Segment-level token count cache, or ``None`` if unattached.

        Backends reading this must tolerate ``None`` — the cache is an
        optional accelerator for repeat tokenization, not a correctness
        dependency.
        """
        return self._token_cache

    def set_token_cache(self, cache: TokenCountCache | None) -> None:
        """Attach (or clear) the segment-level token count cache.

        The cache is constructed by the proxy layer at startup (its DB
        is shared across restarts) and handed to the manager so that
        backends can reach it without threading it through every call
        site. Passing ``None`` detaches, which disables the /completion
        fast path without affecting chat functionality.
        """
        self._token_cache = cache

    def set_settings_store(self, store: SettingsStore | None) -> None:
        """Attach (or clear) the settings store used for lazy-load fallback.

        When attached, ``start()`` reads the previously-applied load options
        for a model from ``app_settings["engine.last_load.<model_id>"]`` if
        the caller didn't pass any. The route handler that performs explicit
        loads writes to the same key, so a request that triggers a lazy
        re-load (after an unload, idle timeout, or crash) reuses whatever
        ctx/GPU layout the user last confirmed.
        """
        self._settings_store = store

    async def _load_saved_options(self, model_id: str) -> dict[str, Any] | None:
        """Read the install-wide saved load options for ``model_id``.

        Returns the dict written by :meth:`persist_load_options`, or
        ``None`` when no settings store is attached, no entry exists, or
        the stored value isn't a JSON object. Failures are swallowed —
        the lazy-load path falls back to defaults rather than blocking on
        a settings-store hiccup.
        """
        if self._settings_store is None or not model_id:
            return None
        try:
            raw = await self._settings_store.get(_LAST_LOAD_KEY_PREFIX + model_id)
        except Exception as exc:
            log.warning("engine_last_load_read_failed", model=model_id, error=str(exc)[:200])
            return None
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("engine_last_load_parse_failed", model=model_id)
            return None
        return parsed if isinstance(parsed, dict) else None

    async def persist_load_options(self, model_id: str, load_options: dict[str, Any]) -> None:
        """Persist ``load_options`` as the install-wide default for ``model_id``.

        Called by the engine load route after a successful explicit load
        so that subsequent lazy-loads (after unload, idle timeout, or
        crash) replay the same configuration. No-ops without a settings
        store or on serialization failure — persistence is best-effort.
        """
        if self._settings_store is None or not model_id or not load_options:
            return
        try:
            payload = json.dumps(load_options)
        except (TypeError, ValueError) as exc:
            log.warning("engine_last_load_encode_failed", model=model_id, error=str(exc)[:200])
            return
        try:
            await self._settings_store.set(_LAST_LOAD_KEY_PREFIX + model_id, payload)
        except Exception as exc:
            log.warning("engine_last_load_write_failed", model=model_id, error=str(exc)[:200])

    def kv_ttl_days_for_mode(self, mode: str = "") -> int:
        """Return the sliding TTL for a mode's warm KV snapshot."""
        if (mode or "").strip().lower() == "narrative":
            return max(0, int(self.kv_narrative_ttl_days))
        return max(0, int(self.kv_ttl_days))

    def session_is_pinned(self, session_key: str, mode: str = "") -> bool:
        """Return True when a session should be protected from eviction."""
        if session_key in self._pinned_sessions:
            return True
        return bool(self.kv_auto_pin_narrative and (mode or "").strip().lower() == "narrative")

    def current_runtime_signature(self) -> dict[str, Any]:
        """Describe the loaded runtime shape for KV compatibility checks.

        All fields here participate in ``_restore_skip_reason``: a saved
        slot whose recorded signature differs from the live runtime is
        rejected on restore. Add new dimensions to both the dict here
        AND the comparison in ``llama_cpp.py:_restore_skip_reason`` so
        we never accept a slot built with incompatible KV layout.
        """
        try:
            model_mtime = os.path.getmtime(self.model_path) if self.model_path else 0.0
        except OSError:
            model_mtime = 0.0
        model_key = self.model_id or (Path(self.model_path).stem if self.model_path else "")
        # Architectural fingerprints (llama.cpp Discussion #15569 must-match
        # list). Pulled from the cached profile when available; zero when
        # the manager hasn't yet loaded a model (manifest entries seeded
        # before the first ``start()`` will get zeros, which the
        # restore-skip checker tolerates symmetrically).
        profile = self._last_profile
        n_embed = int(profile.n_embed or 0) if profile else 0
        n_layers_total = int(profile.n_layers or 0) if profile else 0
        n_heads_kv = int(profile.n_heads_kv or 0) if profile else 0
        return {
            "model_key": model_key,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_mtime": model_mtime,
            "ctx_size": self.current_ctx_size,
            "kv_cache_type": self.current_kv_cache_type or self.kv_cache_type or "",
            # KV-layout-affecting load options. flash_attn changes K/V
            # tensor strides; gpu_layers changes which device holds
            # which layer's KV; both make a saved slot wire-incompatible
            # with the new runtime.
            "flash_attn": bool(self.current_flash_attn),
            "gpu_layers": int(self.current_gpu_layers or 0),
            "gpu_layers_mode": str(self.current_gpu_layers_mode or ""),
            "batch_size": int(self.current_batch_size or 0),
            # Speculative decoding doesn't strictly affect main-slot KV,
            # but invalidating on draft-model changes is cheap insurance
            # against subtle generation-time mismatches.
            "draft_model": str(self.current_draft_model or ""),
            "draft_max": int(self.current_draft_max or 0),
            "draft_ctx_size": int(self.current_draft_ctx_size or 0),
            "draft_gpu_layers": int(self.current_draft_gpu_layers or 0),
            "draft_min": int(self.current_draft_min or 0),
            "draft_p_min": float(self.current_draft_p_min or 0.0),
            # Architectural fingerprints — protect against the rare case
            # where ``model_id`` + ``model_mtime`` happen to coincide
            # across two different models (e.g. user overwrites a GGUF
            # with a same-mtime variant). All sourced from the cached
            # profile so adding them costs only a dict assignment.
            "n_embed": n_embed,
            "n_layers_total": n_layers_total,
            "n_heads_kv": n_heads_kv,
        }

    @staticmethod
    def _coerce_int(
        value: Any,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """Parse integer-ish values safely, then clamp them."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        if minimum is not None:
            number = max(minimum, number)
        if maximum is not None:
            number = min(maximum, number)
        return number

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        """Parse booleans from JSON, strings, and form-ish values."""
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _normalize_kv_cache_type(value: Any) -> str:
        """Normalize KV cache values while keeping llama.cpp default as empty."""
        cache_type = str(value or "").strip().lower()
        if cache_type in {"", "auto", "default"}:
            return ""
        if cache_type in {"f16", "q8_0", "q4_0"}:
            return cache_type
        return ""

    @staticmethod
    def _query_process_ram_mb(pid: int | None) -> int:
        """Best-effort RSS lookup for the managed llama-server process."""
        if not pid:
            return 0
        try:
            import psutil

            proc = psutil.Process(pid)
            return int(proc.memory_info().rss // (1024 * 1024))
        except Exception:
            return 0

    @staticmethod
    def _query_system_memory_info() -> dict[str, int]:
        """Best-effort system memory snapshot in MiB.

        Container-aware: reports the cgroup ceiling when one applies, not
        the host/WSL-VM total. Fit checks that consume this were otherwise
        comparing model sizes against memory the container cannot have.
        """
        try:
            from augmentum.resource import hostmem

            info = hostmem.memory_info()
            return {
                "total_mib": info.total_mib,
                "available_mib": info.available_mib,
                "used_mib": info.used_mib,
                "source": info.source,
                "limited": int(info.limited),
            }
        except Exception:
            return {}

    def _reset_actual_memory(self) -> None:
        """Forget the most recently parsed llama.cpp memory breakdown."""
        self._actual_memory_components = {}
        self._actual_memory_loaded = False
        self._mem_capture_misses = []

    @staticmethod
    def _normalize_memory_location(location: str) -> str:
        """Normalize llama.cpp memory location labels."""
        return re.sub(r"\s+", " ", str(location or "").strip())

    @staticmethod
    def _normalize_memory_component(component: str) -> str:
        """Normalize llama.cpp component labels to snake_case."""
        return str(component or "").strip().lower().replace(" ", "_")

    @staticmethod
    def _memory_location_scope(location: str) -> str:
        """Classify a llama.cpp memory location as VRAM or host RAM."""
        lowered = str(location or "").strip().lower()
        if not lowered:
            return "unknown"
        if "host" in lowered or lowered.startswith("cpu") or "mapped" in lowered:
            return "ram"
        if lowered.startswith(("cuda", "metal", "vulkan", "rocm", "hip", "sycl", "opencl")):
            return "vram"
        return "ram"

    def _capture_actual_memory_from_line(self, line: str) -> None:
        """Parse a llama.cpp memory component line into structured totals."""
        match = _MEMORY_COMPONENT_LINE_RE.search(line)
        if not match:
            # Stash near-miss buffer lines so the "no memory captured"
            # warning can show the exact format we failed to parse — the
            # fastest way to extend the regex for a new build/architecture
            # instead of guessing. Capped so a chatty log can't grow it.
            low = line.lower()
            if "buffer" in low and "size" in low and len(self._mem_capture_misses) < 12:
                self._mem_capture_misses.append(line.strip()[:200])
            return
        try:
            size = float(match.group("size"))
        except (TypeError, ValueError):
            return
        if size < 0:
            return
        unit = (match.group("unit") or "MiB").lower()
        size_mib = size * _MEM_UNIT_TO_MIB.get(unit, 1.0)
        location = self._normalize_memory_location(match.group("location"))
        component = self._normalize_memory_component(match.group("component"))
        if not location or not component:
            return
        by_location = self._actual_memory_components.setdefault(component, {})
        by_location[location] = size_mib

    def _actual_memory_snapshot(self) -> dict[str, Any] | None:
        """Return parsed llama.cpp memory usage, or None if unavailable."""
        if not self._actual_memory_components:
            return None

        location_totals: dict[str, dict[str, Any]] = {}
        for by_location in self._actual_memory_components.values():
            for location, size_mib in by_location.items():
                entry = location_totals.setdefault(
                    location,
                    {
                        "scope": self._memory_location_scope(location),
                        "total_mib": 0.0,
                    },
                )
                entry["total_mib"] += float(size_mib)

        vram_total = 0.0
        ram_total = 0.0
        for entry in location_totals.values():
            if entry["scope"] == "vram":
                vram_total += entry["total_mib"]
            elif entry["scope"] == "ram":
                ram_total += entry["total_mib"]

        return {
            "source": "llama_server_logs",
            "complete": self._actual_memory_loaded,
            "vram_total_mib": int(round(vram_total)),
            "ram_total_mib": int(round(ram_total)),
            "locations": {
                location: {
                    "scope": entry["scope"],
                    "total_mib": int(round(entry["total_mib"])),
                }
                for location, entry in sorted(location_totals.items())
            },
            "components": {
                component: {
                    location: round(size_mib, 2)
                    for location, size_mib in sorted(by_location.items())
                }
                for component, by_location in sorted(self._actual_memory_components.items())
            },
        }

    def _record_workspace_calibration_sample(
        self, snapshot: dict[str, Any]
    ) -> None:
        """Update the calibration EMA from a freshly-observed load.

        Reads the ``compute buffer`` component from the actual-memory
        snapshot, sums the VRAM-scoped contribution, and records the
        ratio against the BASELINE compute reserve we used for this
        load. The baseline (not the calibrated value) is recorded so
        successive samples are independent estimates of the true
        ratio rather than residuals after applied calibration —
        otherwise the EMA would self-reinforce toward 1.0 instead of
        converging on real hardware behavior.

        Silently no-ops in CPU-only loads (no compute reserve to
        calibrate), missing-component snapshots (older llama-server
        builds that didn't log compute buffers), and any case where
        the predicted baseline wasn't captured (build_load_plan
        skipped, e.g. a forced ad-hoc start).
        """
        if self._last_predicted_compute_reserve_bytes <= 0:
            return
        if not self._last_predicted_compute_bucket:
            return

        components = snapshot.get("components") or {}
        # ``_normalize_memory_component`` lowercases + snake_cases the
        # llama.cpp label, so "compute buffer" lands at the
        # ``compute_buffer`` key.
        compute_components = components.get("compute_buffer") or {}
        if not compute_components:
            return

        observed_mib = 0.0
        for location, size_mib in compute_components.items():
            if self._memory_location_scope(location) == "vram":
                try:
                    observed_mib += float(size_mib)
                except (TypeError, ValueError):
                    continue

        if observed_mib <= 0:
            return

        self._workspace_calibration.record(
            self._last_predicted_compute_bucket,
            observed_bytes=observed_mib * 1024 * 1024,
            predicted_bytes=float(self._last_predicted_compute_reserve_bytes),
        )

    def _ingest_server_line(self, line: str, stream_name: str) -> None:
        """Process one llama-server log line for state, metrics, and logging."""
        if not line:
            return

        self._capture_actual_memory_from_line(line)

        # ``common_fit_params`` abort: llama.cpp's VRAM autofit wanted to
        # reduce n_gpu_layers but it was pinned via ``--n-gpu-layers`` (which
        # we always pass), so it bails instead of shrinking. Capture the
        # reported layer count so the start path can retry with a lower
        # offload. Example line:
        #   "common_fit_params: failed to fit params to free device memory:
        #    n_gpu_layers already set by user to 40, abort"
        if "n_gpu_layers already set by user" in line:
            m = re.search(r"n_gpu_layers already set by user to (\d+)", line)
            self._fit_abort_layers = (
                int(m.group(1)) if m else (self._last_effective_gpu_layers or 0)
            )

        # Slot-router visibility: llama-server logs WHICH slot it picked for
        # each unpinned request and WHY ("selected slot by LCP similarity" vs
        # "by LRU") plus per-request prefix reuse ("prompt done, n_past").
        # These are the ground truth for KV session-restore behavior and were
        # previously dropped to debug, which is how the multi-slot zero-reuse
        # regression stayed invisible.
        if ("selected slot by" in line
                or "launch_slot_" in line
                or "prompt done" in line
                or "kv cache rm" in line):
            log.info("llama_server_slot_trace", line=line)

        lower = line.lower()
        if "model loaded" in lower and not self._actual_memory_loaded:
            self._actual_memory_loaded = True
            snapshot = self._actual_memory_snapshot()
            if snapshot:
                log.info(
                    "llama_server_memory_loaded",
                    vram_mib=snapshot.get("vram_total_mib", 0),
                    ram_mib=snapshot.get("ram_total_mib", 0),
                )
                self._record_workspace_calibration_sample(snapshot)
            else:
                # No buffer lines parsed → the resource ledger will ESTIMATE
                # this model's VRAM instead of reporting it exactly. Surface
                # the unparsed format so the regex can be extended for this
                # build/architecture rather than silently degrading.
                log.warning(
                    "llama_server_memory_capture_empty",
                    model=self.model_id,
                    note="no VRAM/RAM buffer lines parsed; ledger will estimate this model",
                    sample=self._mem_capture_misses[:6],
                )

        if "unable to restore slot" in lower:
            log.debug("llama_server_slot_miss", line=line)
            return

        # Prefill progress — llama-server emits ``slot print_timing: id N |
        # task M | prompt processing, n_tokens = X, progress = Y, t = Z s /
        # TPS tokens per second`` every chunk of prefill (n_tokens chunks
        # sized by --batch-size). On a 100K-token narrative turn this fires
        # ~25-40 times across 30-180s — perfect cadence for a "preparing
        # context… 47%" bar that tells the user how close they are to TTFT.
        # We just stash the latest values; the frontend polls for updates
        # via /api/engine/v2/prefill_progress.
        if "prompt processing" in lower and "progress" in lower:
            m = _PREFILL_PROGRESS_RE.search(line)
            if m:
                self._prefill_progress = {
                    "slot_id": int(m.group(1)),
                    "tokens_done": int(m.group(2)),
                    "progress": float(m.group(3)),
                    "elapsed_s": float(m.group(4)),
                    "tps": float(m.group(5)),
                    "updated_at": time.time(),
                }
                log.info(
                    "llama_prefill_progress",
                    **self._prefill_progress,
                )
                return

        # Gated Delta Net + partial offload incompatibility. Qwen 3.5 / 3.6
        # hybrid-SSM models fuse layer 0's GDN tensors onto whichever device
        # holds them; if `n_gpu_layers` is reduced (OOM backoff) and layer 0
        # ends up on CPU while the fused tensor stays on CUDA0, sched_reserve
        # aborts and the subprocess exits with code 1. Latch the incompatibility
        # so `_start_with_oom_backoff` stops cascading retries that can't help.
        if "sched_reserve" in lower and "gated delta net" in lower:
            self._partial_offload_incompatible = True
            log.warning(
                "llama_server_partial_offload_incompatible",
                line=line,
                note=(
                    "Architecture requires full-GPU or full-CPU offload. "
                    "OOM backoff retries with reduced n_gpu_layers will "
                    "not rescue this load — reduce ctx_size or use a "
                    "smaller quant instead."
                ),
            )
            return

        # MTP acceptance-rate scraper. PR #22673 emits two lines we care
        # about each stats interval:
        #   "draft acceptance rate = 0.79728 ( 4169 accepted / 5229 generated)"
        #   "statistics mtp: #calls(b,g,a) = 2 2272 1976, #gen drafts = 2272"
        # Counters are cumulative for the lifetime of the subprocess —
        # see _mtp_last_log doc in __init__.
        if "draft acceptance rate" in lower:
            m = re.search(
                r"draft acceptance rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated",
                line,
            )
            if m:
                self._mtp_last_log = {
                    "rate": float(m.group(1)),
                    "accepted": int(m.group(2)),
                    "generated": int(m.group(3)),
                }
                log.info(
                    "llama_mtp_acceptance",
                    rate=self._mtp_last_log["rate"],
                    accepted=self._mtp_last_log["accepted"],
                    generated=self._mtp_last_log["generated"],
                )
                return
        if any(
            kw in lower
            for kw in (
                # ``error:`` was the original anchor but llama.cpp emits
                # ``error loading model: error loading model architecture:``
                # which has no ``error:`` form. The previous classifier
                # then matched ``loading`` below and demoted the real
                # diagnostic to DEBUG. ``error loading`` + the explicit
                # ``unknown/unsupported architecture`` patterns recover it.
                "error:", "error loading", "fatal", "failed", "oom", "out of memory",
                "segfault", "exception", "abort",
                "unknown model architecture", "unsupported architecture",
                "unknown architecture", "unrecognized",
            )
        ):
            if "seq_load_file" in lower or "no such file" in lower:
                log.debug("llama_server_slot_miss", line=line)
            else:
                log.error("llama_server_error", line=line)
        elif any(
            kw in lower
            for kw in (
                "model loaded", "warming", "ready", "listening", "cuda",
                "device", "system_info", "load_tensors", "kv_cache",
                "compute buffer", "n_gpu_layers",
            )
        ):
            log.info("llama_server_status", line=line)
        elif "%" in line or "loading" in lower:
            log.debug("llama_server_progress", line=line)
        elif any(
            kw in lower
            for kw in (
                "eval time", "prompt eval", "sample time", "total time",
                "tokens per second",
            )
        ):
            log.info("llama_server_perf", line=line)
        else:
            log.debug("llama_server_output", stream=stream_name, line=line[:200])

    def _model_gpu_bytes(self, profile: ModelProfile, gpu_layers: int) -> int:
        """Estimate how many model bytes live in VRAM for the chosen offload."""
        if gpu_layers <= 0 or profile.total_size_bytes <= 0 or profile.n_layers <= 0:
            return 0
        if gpu_layers >= profile.n_layers:
            return int(profile.total_size_bytes)
        return int(profile.total_size_bytes * (gpu_layers / profile.n_layers))

    def _kv_bytes_per_token(self, profile: ModelProfile, kv_cache_type: str) -> int:
        """Estimate KV bytes per token for the selected cache precision.

        Modern LLMs (Qwen3, Llama 3, Mistral, …) use Grouped Query Attention
        — K/V tensors hold only ``n_heads_kv`` heads, not the full
        ``n_heads``. Treating KV as a full ``n_embed``-wide tensor (the old
        formula) overestimates by the GQA ratio (typically 4–8×), which
        made autofit pick conservative GPU layer counts and tiny ctxs even
        on cards that could hold the real configuration comfortably.

        Falls back to the full-embed estimate when GQA metadata is missing
        (older non-GQA models, non-standard GGUFs).
        """
        if profile.n_layers <= 0 or profile.n_embed <= 0:
            return 0
        n_heads = profile.n_heads or 0
        n_heads_kv = profile.n_heads_kv or n_heads
        if n_heads > 0 and n_heads_kv > 0:
            head_dim = profile.n_embed // n_heads
            kv_per_token = 2 * profile.n_layers * head_dim * n_heads_kv * 2
        else:
            kv_per_token = 2 * profile.n_layers * profile.n_embed * 2
        if kv_cache_type == "q8_0":
            kv_per_token //= 2
        elif kv_cache_type == "q4_0":
            kv_per_token //= 4
        return kv_per_token

    def _estimate_prompt_workspace_bytes(
        self,
        profile: ModelProfile,
        batch_size: int,
        flash_attn: bool,
        gpu_layers: int,
    ) -> tuple[int, int]:
        """Estimate prompt-eval workspace split across GPU and system RAM.

        LM Studio's public docs explicitly account for eval batch size and
        flash attention in its load estimator. We mirror that behavior here
        with a conservative heuristic so long prompts are less surprising.
        """
        if batch_size <= 0 or profile.n_embed <= 0:
            return (0, 0)

        token_batch = max(32, int(batch_size))
        hidden_bytes = token_batch * max(profile.n_embed, 1024) * 2
        attention_factor = 4.0 if flash_attn else 8.0
        head_factor = min(2.0, max(1.0, (profile.n_heads or 8) / 16.0))
        moe_factor = 1.0 + (0.2 * max(0, min(profile.expert_used_count or 1, 8) - 1))
        workspace_total = int(hidden_bytes * attention_factor * head_factor * moe_factor)

        if profile.n_layers <= 0 or gpu_layers <= 0:
            return (0, workspace_total)
        if gpu_layers >= profile.n_layers:
            return (workspace_total, 0)

        gpu_fraction = max(0.0, min(1.0, gpu_layers / profile.n_layers))
        gpu_workspace = int(workspace_total * gpu_fraction)
        cpu_workspace = max(0, workspace_total - gpu_workspace)
        return (gpu_workspace, cpu_workspace)

    def _cap_ctx_for_vram(
        self,
        profile: ModelProfile,
        requested_ctx: int,
        kv_cache_type: str,
        gpu_layers: int,
    ) -> int:
        """Cap context conservatively so the chosen load profile still fits."""
        ctx = max(2048, int(requested_ctx or self._ctx_size or 2048))
        vram_bytes = self._get_vram_bytes()
        if vram_bytes <= 0:
            return ctx

        model_gpu_bytes = self._model_gpu_bytes(profile, gpu_layers)
        kv_per_token = self._kv_bytes_per_token(profile, kv_cache_type)
        if model_gpu_bytes <= 0 or kv_per_token <= 0:
            return ctx

        vram_after_model = vram_bytes - model_gpu_bytes - 2 * 1024**3
        if vram_after_model <= 0:
            return 2048

        max_ctx = int(vram_after_model / kv_per_token) if kv_per_token > 0 else ctx
        if max_ctx < ctx:
            old_ctx = ctx
            ctx = max(2048, max_ctx)
            log.info(
                "autofit_ctx_capped",
                original=old_ctx,
                capped=ctx,
                vram_gb=round(vram_bytes / 1e9, 1),
            )
        return ctx

    def _autofit_gpu_layers_for(
        self,
        profile: ModelProfile,
        ctx_size: int,
        kv_cache_type: str,
        gpu_layers_cap: int | None = None,
        flash_attn: bool | None = None,
        extra_reserve_bytes: int = 0,
    ) -> int:
        """First-fit GPU layer allocator with per-layer KV proration.

        Walks layers 0 → ``min(n_layers, gpu_cap)`` in order, tracking
        a running VRAM total of ``layer_weight + layer_kv_share +
        layer_compute_share``. Stops at the first layer whose addition
        would exceed ``vram - _FIRST_FIT_BASELINE_RESERVE_BYTES``.

        ``flash_attn`` selects the compute-reserve branch (T2-6):
        FA-on uses the smaller 384 MiB pool, FA-off uses the larger
        640 MiB pool to cover the full attention-scores scratch.
        ``None`` falls back to ``self.flash_attn`` (the manager's
        configured default) for backwards compatibility with callers
        that haven't been updated.

        Replaces the previous flat-margin formula:

            available = vram - kv_total - compute_total - safety_margin
            fit = available / (total_size / n_layers)

        That formula reserved KV for ALL layers up front, regardless
        of how many were actually offloaded. In practice KV only
        consumes GPU memory for GPU-resident layers — CPU-resident
        layers' KV stays in system RAM. The flat-margin formula was
        therefore over-reserving VRAM proportional to how few layers
        were offloaded, leaving usable VRAM on the table. Per-layer
        proration attributes KV correctly to each offloaded layer,
        freeing budget that the old formula wasted.

        Mirrors the pattern in Ollama's
        ``llm/memory.go::EstimateGPULayers``: per-layer cost includes
        a fair share of KV + compute, walk layer-by-layer, stop at
        the first overflow rather than averaging globally. On a 24 GB
        card with a 30 GB / 32-layer model and 8 GB max KV, the new
        path typically fits ~5 more layers than the old formula.
        """
        gpu_cap = self._gpu_layers if gpu_layers_cap is None else gpu_layers_cap
        if gpu_cap == 0:
            return 0

        vram_bytes = self._get_vram_bytes()
        if vram_bytes <= 0 or profile.n_layers == 0 or profile.total_size_bytes == 0:
            # Can't probe VRAM or no layer info — fall back to "all
            # layers up to cap." llama-server's mmap will sort out the
            # actual fit at load time; this path is rare in practice.
            if profile.n_layers > 0 and gpu_cap > 0:
                return min(gpu_cap, profile.n_layers)
            return gpu_cap

        # Per-layer cost components.
        bytes_per_layer = profile.total_size_bytes / profile.n_layers

        kv_per_token = self._kv_bytes_per_token(profile, kv_cache_type)
        budget_ctx = max(2048, min(ctx_size, profile.context_length or ctx_size))
        kv_total = kv_per_token * budget_ctx if kv_per_token > 0 else 0
        kv_per_layer = kv_total / profile.n_layers

        # Compute reserve distributed per offloaded layer. The total
        # pool depends on Flash Attention — see ``_compute_reserve_bytes``
        # for the rationale.
        fa_for_reserve = self.flash_attn if flash_attn is None else flash_attn
        compute_total = self._compute_reserve_bytes(fa_for_reserve)
        compute_per_layer = compute_total / profile.n_layers

        per_layer_cost = bytes_per_layer + kv_per_layer + compute_per_layer
        # extra_reserve_bytes carves out room for *other* GPU consumers the
        # target doesn't know about — the speculative-decoding draft model
        # is the current one. Without this the target greedily fills VRAM
        # and the draft hits OOM at load time. Caller computes the
        # reservation (draft size + draft KV + small compute headroom).
        budget = (
            vram_bytes
            - self._FIRST_FIT_BASELINE_RESERVE_BYTES
            - max(0, int(extra_reserve_bytes))
        )

        if budget <= 0 or per_layer_cost <= 0:
            log.warning(
                "autofit_no_vram",
                vram_gb=round(vram_bytes / 1e9, 1),
                baseline_reserve_gb=round(self._FIRST_FIT_BASELINE_RESERVE_BYTES / 1e9, 2),
                per_layer_mb=round(per_layer_cost / 1e6),
            )
            return 0

        # First-fit walk. ``int(budget // per_layer_cost)`` is
        # arithmetically equivalent to a layer-by-layer loop and far
        # cheaper; preserved as the canonical interpretation since
        # all per-layer terms are uniform across the chain.
        fit_layers = int(budget // per_layer_cost)
        fit_layers = max(0, min(fit_layers, profile.n_layers, gpu_cap))

        log.info(
            "autofit_first_fit",
            vram_gb=round(vram_bytes / 1e9, 1),
            baseline_reserve_gb=round(self._FIRST_FIT_BASELINE_RESERVE_BYTES / 1e9, 2),
            compute_reserve_mb=round(compute_total / 1e6),
            flash_attn=fa_for_reserve,
            kv_total_gb=round(kv_total / 1e9, 2),
            model_gb=profile.size_gb,
            per_layer_mb=round(per_layer_cost / 1e6),
            kv_per_layer_mb=round(kv_per_layer / 1e6),
            layers_total=profile.n_layers,
            layers_fit=fit_layers,
            is_moe=profile.is_moe,
            non_expert_gb=round(profile.non_expert_tensor_bytes / 1e9, 2) if profile.is_moe else 0,
            expert_gb=round(profile.expert_tensor_bytes / 1e9, 2) if profile.is_moe else 0,
        )

        if fit_layers < min(profile.n_layers, gpu_cap):
            log.info(
                "autofit_partial",
                vram_gb=round(vram_bytes / 1e9, 1),
                model_gb=profile.size_gb,
                kv_total_gb=round(kv_total / 1e9, 2),
                layers_total=profile.n_layers,
                layers_fit=fit_layers,
            )
        else:
            log.info(
                "autofit_full",
                vram_gb=round(vram_bytes / 1e9, 1),
                model_gb=profile.size_gb,
                layers=fit_layers,
            )

        return fit_layers

    def _autofit_moe_cpu_layers(
        self,
        profile: ModelProfile,
        ctx_size: int,
        kv_cache_type: str,
        flash_attn: bool,
        batch_size: int,
        extra_reserve_bytes: int = 0,
    ) -> int:
        """Pick N for ``--n-cpu-moe`` that maximises VRAM use without spill.

        Strategy: all non-expert tensors stay on GPU (attention, norms,
        embeddings, shared FFN) along with KV cache and compute reserve.
        Then fit as many *expert-layer-worth-of-bytes* on GPU as
        possible. Whatever can't fit goes to CPU.

        Returns the integer N for ``--n-cpu-moe`` — the count of
        layers whose experts go to CPU. ``N = 0`` means every expert
        stays on GPU (only fits for tiny MoEs); ``N = n_layers`` means
        every expert goes to CPU (same effect as ``--cpu-moe``).

        Falls back to ``profile.n_layers`` (all experts to CPU) when
        VRAM can't be probed or the budget math goes underwater —
        always preferable to a VRAM overflow which would spill into
        shared system memory and tank perf.
        """
        if not profile.is_moe or profile.n_layers <= 0:
            return profile.n_layers

        # Use FREE VRAM, not total — other processes (image-gen models,
        # other CUDA contexts, the desktop compositor on a connected
        # display, a still-loaded earlier model) routinely sit on
        # multiple GB. Budgeting against total leads to "common_params
        # _fit_impl: cannot meet free memory target" + spill to shared
        # memory (which CUDA does silently — observable as 3-5× slower
        # generation, not an OOM). Observed 2026-05-15 on the 122B-A10B
        # load: total=25.8 GB but free=22.7 GB, so total-based budget
        # overshot by ~3 GB and llama.cpp aborted its fit step.
        gpu_info = self._query_gpu_info()
        free_bytes = int(gpu_info.get("free_bytes", 0) or 0)
        total_bytes = int(gpu_info.get("total_bytes", 0) or 0)
        # When we can't probe free directly, fall back conservatively
        # to (total - 2 GB) as a "something is using VRAM" estimate.
        if free_bytes <= 0:
            free_bytes = max(0, total_bytes - 2 * 1024**3)
        if free_bytes <= 0:
            return profile.n_layers

        expert_total = int(profile.expert_tensor_bytes or 0)
        non_expert_total = int(profile.non_expert_tensor_bytes or 0)
        if expert_total <= 0:
            return profile.n_layers

        kv_per_token = self._kv_bytes_per_token(profile, kv_cache_type)
        kv_total = kv_per_token * ctx_size if kv_per_token > 0 else 0
        compute_reserve = self._compute_reserve_bytes(flash_attn)
        gpu_workspace, _ = self._estimate_prompt_workspace_bytes(
            profile, batch_size=batch_size, flash_attn=flash_attn,
            gpu_layers=profile.n_layers,
        )

        # VRAM budget for expert tensors after carving out the must-be-
        # resident bits. The 1.07 GiB baseline reserve covers llama
        # .cpp's own ``free memory target of 1024 MiB`` + a tiny cushion
        # for drift between this probe and the subprocess actually
        # allocating (other CUDA contexts can shift VRAM 100-300 MiB in
        # the seconds it takes llama-server to start).
        budget = (
            free_bytes
            - non_expert_total
            - kv_total
            - compute_reserve
            - gpu_workspace
            - self._FIRST_FIT_BASELINE_RESERVE_BYTES
            - max(0, int(extra_reserve_bytes))
        )

        if budget <= 0:
            log.info(
                "autofit_moe_no_vram",
                free_gb=round(free_bytes / 1e9, 1),
                non_expert_gb=round(non_expert_total / 1e9, 2),
                kv_gb=round(kv_total / 1e9, 2),
            )
            return profile.n_layers

        # For pure-transformer MoEs (Mixtral, Qwen3-MoE, DeepSeek-V3),
        # every block carries experts so dividing by ``n_layers`` gives
        # an accurate per-layer cost. For Mamba2-Transformer hybrids
        # (Nemotron 3 Nano Omni's ``nemotron_h_moe``, future Falcon-H1,
        # Jamba, Granite 4.0), experts live only in the transformer
        # blocks while Mamba blocks carry zero expert tensors. Dividing
        # ``expert_total`` by ``n_layers`` then underestimates per-MoE-
        # layer cost, causes the autofit to over-allocate experts to
        # GPU, and llama-server aborts with "cannot meet free memory
        # target" → silent fallback to ``--cpu-moe`` (leaves a lot of
        # VRAM idle).
        #
        # Estimate the actual MoE-bearing layer count from the tensor
        # summary the GGUF scanner already collected. Typical MoE FFN
        # carries ``expert_count × 3`` tensors per layer (gate / up /
        # down). Topologies with shared experts or gated-linear FFN
        # vary, but a wrong K errs in the safe direction (smaller
        # ``n_moe_layers`` → larger per-layer cost → fewer experts to
        # GPU → no overflow).
        n_moe_layers = profile.n_layers
        if profile.n_expert_tensors:
            # Fused-expert layout: n_expert_tensors ≈ 3 × (MoE layers), NOT
            # 3 × expert_count × (MoE layers). The retired ``expert_count * 3``
            # denominator assumed one-tensor-per-expert and collapsed
            # n_moe_layers to 1 for high-expert models — e.g. Qwen3.5-122B-A10B
            # scanned 144 fused tensors, 144 / (128*3) → round(0.375) → clamp 1,
            # so all 70 GB of experts were treated as ONE unsplittable layer,
            # nothing fit the budget, and moe_auto_vram silently degraded to
            # --cpu-moe (every expert on CPU — the slowest config, worse than a
            # sane hand-tuned N). Dividing by the fused triple restores the real
            # per-layer cost, so the budget math can place experts on GPU again.
            est_moe_layers = round(
                profile.n_expert_tensors / self._FUSED_EXPERT_TENSORS_PER_LAYER
            )
            n_moe_layers = max(1, min(profile.n_layers, est_moe_layers))

        expert_per_layer = expert_total / n_moe_layers
        if expert_per_layer <= 0:
            return profile.n_layers

        # How many MoE-layer-worths of experts fit on GPU within budget?
        # Bounded by ``n_moe_layers`` (can't fit more than exist).
        fit_expert_layers = int(budget // expert_per_layer)
        fit_expert_layers = max(0, min(fit_expert_layers, n_moe_layers))

        # ``--n-cpu-moe N`` semantics: llama.cpp moves the LAST N
        # layers' expert tensors to CPU (counts from highest layer
        # downward). Non-MoE layers in that range are no-ops. We pass
        # ``n_layers - fit_expert_layers`` so at least the bottom
        # ``fit_expert_layers`` worth of MoE layers stay on GPU — this
        # over-counts (includes any Mamba blocks at the top) but the
        # over-count is harmless to llama.cpp and conservative for VRAM.
        n_cpu_moe = profile.n_layers - fit_expert_layers

        log.info(
            "autofit_moe_cpu_layers",
            free_gb=round(free_bytes / 1e9, 1),
            total_gb=round(total_bytes / 1e9, 1),
            non_expert_gb=round(non_expert_total / 1e9, 2),
            expert_gb=round(expert_total / 1e9, 2),
            kv_gb=round(kv_total / 1e9, 2),
            budget_gb=round(budget / 1e9, 2),
            expert_per_layer_mb=round(expert_per_layer / 1e6),
            n_layers=profile.n_layers,
            n_moe_layers=n_moe_layers,
            experts_on_gpu=fit_expert_layers,
            n_cpu_moe=n_cpu_moe,
        )
        return n_cpu_moe

    def _is_slow_mount(self, path: str) -> bool:
        """Check if a path lives on a bridged/network filesystem.

        Reads /proc/self/mountinfo to determine the *actual* fs type
        backing the path, then classifies based on whether that fs
        crosses a translation layer (9p, virtiofs, osxfs, drvfs) or
        a network protocol (cifs, nfs). This matters because GGUF
        loads are large sequential reads — the bridge layer is where
        the ~10× slowdown comes from.

        Falls back to a path-prefix heuristic if mountinfo isn't
        readable. The fallback over-reports on native-Linux setups
        (where /models/host/* would actually be plain ext4), but
        under-reporting would silently make Localize unavailable for
        users who actually need it, so we err on the safe side.
        """
        fs = classify_mount_fs(path)
        if fs:
            return is_slow_filesystem(fs)
        # Fallback for environments without /proc/self/mountinfo
        return (
            path.startswith("/models/")
            or path.startswith("/mnt/host/")
            or path.startswith("/data/host-models/")
        )

    def _local_cache_path(self, model_path: str) -> str:
        """Get local cache path for a host-mounted model."""
        filename = Path(model_path).name
        return os.path.join(self._model_dir, ".host-cache", filename)

    async def _ensure_local_copy(self, model_path: str) -> str:
        """Use a local copy of a host-mounted model if one exists.

        Does NOT auto-copy. Users must explicitly cache models via the
        /v2/models/cache endpoint (since models can be 100+ GB).
        If a cached copy exists and matches the original size, uses it.
        Otherwise falls back to the slow mount path.
        """
        if not self._is_slow_mount(model_path):
            return model_path

        local_path = self._local_cache_path(model_path)

        # Use cached copy if it exists and matches size
        if os.path.isfile(local_path):
            try:
                original_size = os.path.getsize(model_path)
                local_size = os.path.getsize(local_path)
                if local_size == original_size:
                    log.info("host_cache_hit", model=Path(model_path).name,
                             size_gb=round(local_size / 1e9, 1))
                    return local_path
                else:
                    os.unlink(local_path)  # stale
            except OSError:
                pass

        return model_path

    async def cache_host_model(self, model_path: str) -> str:
        """Explicitly copy a host-mounted model to local storage.

        Called by the /v2/models/cache endpoint. Returns the local path
        on success, or raises on failure.
        """
        if not self._is_slow_mount(model_path):
            return model_path

        import shutil
        local_path = self._local_cache_path(model_path)
        cache_dir = os.path.dirname(local_path)
        os.makedirs(cache_dir, exist_ok=True)

        size = os.path.getsize(model_path)
        log.info("host_cache_copying", model=Path(model_path).name,
                 size_gb=round(size / 1e9, 1))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, model_path, local_path)
        log.info("host_cache_complete", model=Path(model_path).name)
        return local_path

    async def start(
        self,
        model_path: str,
        gpu_layers_override: int | None = None,
        load_options: dict[str, Any] | None = None,
    ) -> None:
        """Start llama-server with the given model.

        Gets/builds a profile, constructs CLI args, starts the subprocess,
        drains output pipes, and waits for the health endpoint to report ready.

        Idempotency: concurrent ``start()`` calls for the SAME model
        coalesce — the second caller awaits the in-flight start's
        future instead of killing it and restarting. Different model
        falls through to the existing kill+restart path. See
        ``_starting_future`` in ``__init__`` for the full rationale.
        """
        # Resolve path early so we can match against in-flight start's
        # path. Pushed off-loop because ``_resolve_model_path`` calls
        # ``discover_gguf_files`` which walks every model dir with
        # os.scandir; on a 9p / virtiofs mount that can take seconds
        # and would freeze chat / health checks / DB queries for the
        # duration. Worker thread keeps the loop responsive.
        resolved = await asyncio.to_thread(self._resolve_model_path, model_path)
        if resolved:
            model_path = resolved

        # Coalesce concurrent starts for the same model. Caller wins
        # if they want a different model (kill + restart).
        if (
            self.state == ProcessState.STARTING
            and self._starting_future is not None
            and not self._starting_future.done()
            and self._starting_path == model_path
        ):
            log.info(
                "engine_v2_start_coalesced",
                model_path=model_path,
                pid=self.process.pid if self.process is not None else None,
            )
            try:
                await self._starting_future
            except asyncio.CancelledError:
                # Awaited future cancelled by stop() / shutdown — propagate.
                raise
            except Exception:
                # In-flight start failed; do NOT silently fall through.
                # The caller should know the start failed (and decide
                # whether to retry with different params, e.g. OOM
                # backoff in ``LlamaCppBackend._start_with_oom_backoff``).
                raise
            return

        # Claim the start SYNCHRONOUSLY — no awaits between the gate
        # above and this block. Any await in that window (the reconcile
        # probe and stop() used to sit here) lets a concurrent same-
        # model start() read state != STARTING and double-spawn the
        # subprocess — the exact race coalescing exists to prevent.
        # Hold a LOCAL reference (``my_future``) — the instance
        # attribute can be replaced by a fresher start() call mid-
        # execution (e.g. a different-model swap that kills our
        # in-flight start). Operating on the local reference ensures
        # we set our own result/exception even if the instance attr
        # has moved on, AND prevents the finally cleanup from
        # clobbering a fresher start's future.
        loop = asyncio.get_event_loop()
        my_future = loop.create_future()
        self._starting_future = my_future
        self._starting_path = model_path
        self.state = ProcessState.STARTING

        # Defensive reconcile: if we think we own nothing but a
        # llama-server is alive on our backend port (uvicorn worker
        # swap, prior stop() that didn't actually terminate, etc.),
        # reclaim the port BEFORE we try to spawn — otherwise our
        # spawn races with the strand for the port and we end up
        # hoarding VRAM behind an unrecognized PID. No-op on the
        # common case (port is free, /health doesn't answer).
        if self.process is None:
            await self.reconcile_stranded_subprocess()

        # Kill any existing process first (prevents zombie accumulation).
        # Note: this also kills an in-flight DIFFERENT-model start, by
        # design — the caller wants a different model loaded.
        if self.process is not None:
            await self.stop()
            # stop() resets state to IDLE as it finishes — reassert our
            # claim so same-model callers arriving mid-swap coalesce
            # instead of spawning a third start.
            self.state = ProcessState.STARTING

        # Reset the partial-offload incompatibility latch — each fresh
        # ``start()`` invocation gets a clean slate. The flag is set by
        # the status parser if the subprocess emits the sched_reserve
        # warning during this attempt.
        self._partial_offload_incompatible = False
        # Reset prefill-progress snapshot so the next subprocess's first
        # progress line gets a clean baseline.
        self._prefill_progress = None

        # Seed the load-progress snapshot so the chat dispatch path can
        # surface "Loading model · X · 14s of ~30s" while this start()
        # is in flight. expected_s is the median of recent successful
        # loads for this model_id; falls back to a coarse file-size
        # estimate (~25 MB/s effective throughput including mmap +
        # GPU upload) on first load.
        self._load_progress = self._build_load_progress_snapshot(model_path)

        try:
            await self._start_impl(model_path, gpu_layers_override, load_options)
            if not my_future.done():
                my_future.set_result(None)
        except BaseException as exc:
            if not my_future.done():
                # Use set_exception for Exception subclasses; Cancelled
                # propagates via cancel().
                if isinstance(exc, asyncio.CancelledError):
                    my_future.cancel()
                else:
                    my_future.set_exception(exc)
            self._finalize_load_progress(success=False)
            raise
        finally:
            # Only clear the instance attr if it's still pointing at
            # our future — a different-model start that took over
            # has its own future installed and we mustn't clobber it.
            if self._starting_future is my_future:
                self._starting_future = None
                self._starting_path = ""

    def _build_load_progress_snapshot(self, model_path: str) -> dict[str, Any]:
        """Seed snapshot for the ``self._load_progress`` field.

        ``expected_s`` prefers the median of recent successful loads
        for this model_id (sharp ETA after a few warm runs), falling
        back to a coarse file-size heuristic (~25 MB/s effective
        throughput including mmap + GPU upload) so the very first
        load still shows a meaningful bar instead of "no idea yet".
        """
        model_id = Path(model_path).stem
        size_bytes = 0
        try:
            size_bytes = Path(model_path).stat().st_size
        except OSError:
            pass
        history = self._load_duration_history.get(model_id) or []
        if history:
            sorted_hist = sorted(history)
            expected_s = sorted_hist[len(sorted_hist) // 2]
        else:
            # 25 MB/s is a deliberately conservative estimate so the bar
            # finishes early (jumping to 95%-cap → stage_complete) rather
            # than appearing stalled if the actual load is faster. Bare
            # minimum 5s so a tiny model doesn't show a 0s bar.
            est = (size_bytes / (25 * 1024 * 1024)) if size_bytes else 30.0
            expected_s = max(5.0, est)
        return {
            "model_id": model_id,
            "model_path": model_path,
            "started_at": time.monotonic(),
            "size_bytes": size_bytes,
            "expected_s": float(expected_s),
            "stage_label": "Loading model",
        }

    def _finalize_load_progress(self, *, success: bool) -> None:
        """Record duration on success, then clear the snapshot."""
        snapshot = self._load_progress
        if snapshot is not None:
            if success:
                duration = time.monotonic() - float(snapshot.get("started_at", 0.0))
                if duration > 0:
                    model_id = snapshot.get("model_id", "") or ""
                    history = self._load_duration_history.setdefault(model_id, [])
                    history.append(duration)
                    # Cap so the median tracks the current install,
                    # not a long tail of one-off slow loads from a
                    # transient past condition (cold disk, throttled GPU).
                    if len(history) > 12:
                        del history[: len(history) - 12]
        self._load_progress = None

    async def _start_impl(
        self,
        model_path: str,
        gpu_layers_override: int | None = None,
        load_options: dict[str, Any] | None = None,
        _fit_retries_left: int = 3,
    ) -> None:
        """The actual start work. Wrapped by ``start()`` for idempotency.

        ``_fit_retries_left`` is internal: when llama-server aborts
        ``common_fit_params`` because the (always-passed) ``--n-gpu-layers``
        was pinned too high for the free VRAM, we re-enter with a
        stepped-down ``gpu_layers_override`` rather than surfacing a load
        failure. Each retry halves the offload; after the budget is spent
        the failure propagates.
        """

        if not load_options:
            saved = await self._load_saved_options(Path(model_path).stem)
            if saved:
                load_options = saved
                log.info(
                    "lazy_load_from_settings_store",
                    model=Path(model_path).stem,
                    ctx_size=saved.get("ctx_size"),
                    gpu_layers_mode=saved.get("gpu_layers_mode"),
                    gpu_layers=saved.get("gpu_layers"),
                )

        # Copy host-mounted models to local storage for faster loading
        model_path = await self._ensure_local_copy(model_path)
        self._reset_actual_memory()
        self._fit_abort_layers = None  # cleared per attempt; set by the drainer
        # Forget the prior model's MTP acceptance counters — they're
        # subprocess-scoped and meaningless across a swap.
        self._mtp_last_log = {}

        # Get or build profile
        profile = self.profile_cache.get(model_path)
        if profile is None:
            log.info("No cached profile, scanning GGUF header", model_path=model_path)
            profile = scan_gguf_header(model_path)
            self.profile_cache.save(profile)

        self._last_profile = profile
        self.model_path = model_path
        self.model_id = Path(model_path).stem

        # Decide whether to attach an mmproj this load. Priority:
        #   1. Explicit ``mmproj_path`` in load_options — always wins
        #      (programmatic override, e.g. from a tool that knows it
        #      needs vision and which projector to use).
        #   2. ``vision_mode`` in load_options — explicit per-load:
        #         True  = pair (run sidecar+heuristic search)
        #         False = skip entirely (KV restore stays available)
        #   3. Global ``engine_auto_pair_mmproj`` setting when neither
        #      of the above is set. Defaults False so text-only chats
        #      — the common case — keep KV save/restore working. Vision
        #      is opt-in via the Load Setup toggle.
        #
        # Upstream llama.cpp returns 501 on /slots/.../save+restore the
        # instant --mmproj is loaded, so attaching one silently disables
        # session restore for the lifetime of the subprocess. Honoring
        # the user's explicit "no vision this load" intent keeps that
        # cost off the default path.
        existing_mmproj = str((load_options or {}).get("mmproj_path") or "").strip()
        if not existing_mmproj:
            vision_opt = (load_options or {}).get("vision_mode")
            if vision_opt is True:
                attempt_pair = True
                pair_reason = "vision_mode=true"
            elif vision_opt is False:
                attempt_pair = False
                pair_reason = "vision_mode=false"
            else:
                attempt_pair = bool(getattr(settings, "engine_auto_pair_mmproj", False))
                pair_reason = (
                    "engine_auto_pair_mmproj=true" if attempt_pair
                    else "engine_auto_pair_mmproj=false (default)"
                )

            if attempt_pair:
                auto_mmproj = await asyncio.to_thread(
                    self._find_paired_mmproj, model_path, profile
                )
                if auto_mmproj:
                    load_options = {**(load_options or {}), "mmproj_path": auto_mmproj}
                    log.info(
                        "mmproj_auto_paired",
                        model=Path(model_path).stem,
                        mmproj=auto_mmproj,
                        reason=pair_reason,
                    )
            else:
                log.info(
                    "mmproj_auto_pair_skipped",
                    model=Path(model_path).stem,
                    reason=pair_reason,
                )

        # Pre-warm GPU info off the event loop before the synchronous
        # load-plan path. _build_cli_args → build_load_plan calls
        # _query_gpu_info / _get_vram_bytes 3-4 times across helpers
        # (autofit, ctx cap, plan summary); without pre-warming each
        # invocation would shell out to nvidia-smi inline and block the
        # loop for ~100-500 ms × N. The async pre-warm runs once via
        # asyncio.to_thread so subsequent sync helpers hit the cache.
        await self._query_gpu_info_async()

        args = self._build_cli_args(
            profile,
            model_path,
            gpu_layers_override=gpu_layers_override,
            load_options=load_options,
        )
        log.info("Starting llama-server", model=model_path,
                 args=" ".join(args))

        self.process = await asyncio.create_subprocess_exec(
            self._llama_server_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Start background pipe drainers — prevents pipe buffer from filling
        # and blocking the subprocess (a silent deadlock)
        self._drain_tasks = [
            asyncio.create_task(self._drain_pipe(self.process.stdout, "stdout")),
            asyncio.create_task(self._drain_pipe(self.process.stderr, "stderr")),
        ]

        # Scale health timeout with model size — 70GB through a bind mount
        # can take 10+ minutes. Base: configured timeout. Add 10s per GB
        # for models on slow mounts, 3s per GB for local models.
        timeout = self.health_timeout
        if profile and profile.total_size_bytes > 0:
            size_gb = profile.total_size_bytes / 1e9
            if self._is_slow_mount(model_path):
                timeout = max(timeout, size_gb * 10 + 60)
            else:
                timeout = max(timeout, size_gb * 3 + 30)
            log.info("health_timeout_scaled", timeout_s=round(timeout),
                     size_gb=round(size_gb, 1),
                     slow_mount=self._is_slow_mount(model_path))

        try:
            await self._wait_for_health(timeout=timeout)
        except (RuntimeError, TimeoutError):
            # Health-wait failed. The abort line may still be in flight on
            # the stderr drainer when the process has just exited — give it
            # a brief moment so the fit-abort signal isn't missed by a race.
            if self._fit_abort_layers is None and self.process and self.process.returncode is not None:
                await asyncio.sleep(0.25)
            fit_layers = self._fit_abort_layers

            # Terminate the orphaned subprocess so it doesn't keep holding
            # VRAM. stop() handles process kill, drain-task cleanup, state
            # reset (it does not touch _fit_abort_layers).
            await self.stop()

            if fit_layers and _fit_retries_left > 0:
                # llama.cpp wanted to shrink n_gpu_layers but it was pinned.
                # Step down (halve, floor 1) and retry — giving its autofit
                # room to also fit ctx/batch. Worst case we walk down to a
                # CPU-heavy split that loads slowly but works, which beats a
                # hard load failure.
                reduced = max(1, fit_layers // 2)
                if reduced >= fit_layers:  # degenerate (fit_layers <= 1) — give up
                    raise
                log.warning(
                    "llama_server_fit_retry",
                    model=Path(model_path).stem,
                    rejected_n_gpu_layers=fit_layers,
                    retry_n_gpu_layers=reduced,
                    retries_left=_fit_retries_left - 1,
                )
                await self._start_impl(
                    model_path,
                    gpu_layers_override=reduced,
                    load_options=load_options,
                    _fit_retries_left=_fit_retries_left - 1,
                )
                return
            raise

        self.state = ProcessState.READY
        self._start_time = time.monotonic()
        self.touch()  # reset idle timer
        self.start_idle_monitor()
        log.info("llama-server ready", model_id=self.model_id, pid=self.process.pid)
        # Record this load's duration so the next load of the same
        # model can offer a sharper ETA, then clear the in-flight
        # snapshot so the chat dispatch path stops surfacing progress.
        self._finalize_load_progress(success=True)

        # Opportunistically warm recent sessions. With a backend
        # attached, this runs the resume ladder (restore where slot
        # files exist, REPLAY everywhere else — including --kv-unified,
        # which has no slot files at all) as a background task so a
        # multi-session replay never delays start() returning. Without
        # a backend (tests, early lifespan), fall back to the legacy
        # single-slot restore walk. Failure is non-fatal — log, serve
        # cold.
        if self.kv_warm_on_start:
            backend = self._engine_backend
            if backend is not None:
                self._kv_warm_task = asyncio.create_task(
                    self._kv_warm_start(backend)
                )
            else:
                try:
                    await self._warm_top_session()
                except Exception as exc:
                    log.warning("kv_warm_on_start_failed", error=str(exc)[:200])

    # ------------------------------------------------------------------
    # Pinning — keep a model resident across sibling-model requests
    # ------------------------------------------------------------------

    def pin_model(self, model_id: str) -> None:
        """Increment the pin refcount for ``model_id``.

        While the count is positive, ``swap()`` raises ``ModelPinnedError``
        when the currently-loaded model matches. Callers MUST pair this
        with ``unpin_model`` in a try/finally — a leaked pin permanently
        blocks model swaps for that id.
        """
        if not model_id:
            return
        self._pinned_models[model_id] = self._pinned_models.get(model_id, 0) + 1
        log.info(
            "llama_server_model_pinned",
            model_id=model_id,
            refcount=self._pinned_models[model_id],
        )

    def unpin_model(self, model_id: str) -> None:
        """Decrement the pin refcount for ``model_id``.

        Drops the key from the dict when the count hits zero. Calling
        with an unknown id is a no-op (defensive — easier than tracking
        whether a pair was paired)."""
        if not model_id:
            return
        cur = self._pinned_models.get(model_id, 0)
        if cur <= 1:
            self._pinned_models.pop(model_id, None)
            log.info("llama_server_model_unpinned", model_id=model_id)
        else:
            self._pinned_models[model_id] = cur - 1
            log.info(
                "llama_server_model_unpin_refcount",
                model_id=model_id,
                refcount=cur - 1,
            )

    def is_pinned(self, model_id: str) -> bool:
        """True when ``model_id`` has a positive pin refcount."""
        return self._pinned_models.get(model_id, 0) > 0

    async def swap(self, new_model_path: str, load_options: dict[str, Any] | None = None) -> None:
        """Drain, stop current model, then start the new one.

        Raises ``ModelPinnedError`` when the currently-loaded model has
        a positive pin refcount and the new path doesn't match it.
        """
        if self.model_id and self.is_pinned(self.model_id):
            new_basename = (
                new_model_path.rsplit("/", 1)[-1]
                if new_model_path else ""
            )
            if new_basename != self.model_id:
                raise ModelPinnedError(
                    pinned_model=self.model_id,
                    requested_model=new_basename or new_model_path,
                    refcount=self._pinned_models[self.model_id],
                )
        self.state = ProcessState.DRAINING
        log.info("Swapping model", current=self.model_id, new=new_model_path)
        await self.stop()
        await self.start(new_model_path, load_options=load_options)

    async def stop(self) -> None:
        """Terminate the subprocess (10s timeout), kill if needed.

        Emits ``vram_release`` telemetry after teardown when the model
        had GPU layers — INFO when the driver returned a reasonable
        chunk, WARNING when it lagged. The WSL2 + CUDA-in-Docker stack
        is known to occasionally hold onto VRAM across subprocess exit
        (driver bug class); this gives us in-process evidence of how
        often that fires for a given operator before deciding whether
        a heavier mitigation (sacrificial cudaDeviceReset helper) is
        worth the complexity. No automatic recovery is attempted —
        telemetry only.
        """
        self.stop_idle_monitor()
        # Slot 0 dies with the process — drop the warm hint so the next
        # backend that observes the manager doesn't think a stale slot
        # is still warm. Same for the ladder's replay-warmed tags, and
        # cancel any boot-warm loop still replaying into slots that are
        # about to disappear.
        self._warm_session_key = ""
        self._replay_warmed_keys.clear()
        if self._kv_warm_task is not None and not self._kv_warm_task.done():
            self._kv_warm_task.cancel()
        self._kv_warm_task = None

        if self.process is None:
            self._reset_actual_memory()
            self.state = ProcessState.IDLE
            return

        self.state = ProcessState.STOPPING
        pid = self.process.pid
        log.info("Stopping llama-server", pid=pid)

        # Snapshot pre-stop VRAM + the gpu_layers we're about to release.
        # Skip the nvidia-smi probe entirely on CPU-only loads — there's
        # nothing to verify, and the shell-out is wasted work on every
        # CPU-only sibling stop (vision aux, embedding helper, etc.).
        # Best-effort — nvidia-smi may not be present on a CPU-only host.
        pre_vram_used_mib = 0
        pre_model_id = self.model_id
        # Both are mmap'd by the child, so both ratchet the page cache.
        pre_model_path = self.model_path
        pre_mmproj_path = self.current_mmproj_path or ""
        pre_gpu_layers = self.current_gpu_layers
        if pre_gpu_layers > 0:
            try:
                pre_info = await self._sample_vram_fresh_async()
                pre_vram_used_mib = int(pre_info.get("used_mib", 0))
            except Exception as exc:
                # nvidia-smi missing / driver hiccup — telemetry is
                # best-effort and never blocks teardown.
                log.debug("vram_pre_sample_failed", error=str(exc)[:120])

        try:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10.0)
            except TimeoutError:
                log.warning("llama-server did not exit in 10s, killing", pid=pid)
                self.process.kill()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except TimeoutError:
                    # asyncio.wait() can hang past SIGKILL on WSL2+CUDA
                    # when the kernel hasn't reaped the process yet
                    # (D-state, waiting on GPU driver release). Bypass
                    # the asyncio.subprocess machinery: poll psutil for
                    # actual reap and re-send SIGKILL via os.kill if
                    # the PID is genuinely still alive. If escalation
                    # also fails, log loudly — manager state will still
                    # be cleared below, but reconcile_stranded_subprocess
                    # at the next idle tick / start() / boot picks up
                    # the strand. The new ``stranded`` flag in status()
                    # surfaces it to the UI in the meantime.
                    await self._escalate_kill(pid)
        except ProcessLookupError:
            pass  # Already dead

        await self._cancel_drain_tasks()
        self.process = None
        self.model_id = ""
        self.model_path = ""
        self._start_time = None
        self.current_ctx_size = self._ctx_size
        self.current_gpu_layers = self._gpu_layers
        self.current_batch_size = self._batch_size
        self.current_kv_cache_type = self.kv_cache_type or ""
        self.current_flash_attn = self.flash_attn
        self.current_draft_model = self.draft_model
        self.current_draft_max = self.draft_max
        self.current_gpu_layers_mode = "auto"
        self.current_mmproj_path = ""
        self._last_load_plan = None
        self._reset_actual_memory()
        self.state = ProcessState.IDLE

        # Only meaningful for loads that had GPU layers AND a working
        # nvidia-smi probe; skip otherwise so CPU-only deploys don't
        # spam telemetry. Give the driver up to 3s of poll wall-time
        # before declaring a lag — measured release of well-behaved
        # unloads is sub-second on bare metal, ~1-2s under WSL.
        if pre_gpu_layers > 0 and pre_vram_used_mib > 0:
            await self._emit_vram_release_telemetry(
                pre_used_mib=pre_vram_used_mib,
                model_id=pre_model_id,
                gpu_layers=pre_gpu_layers,
            )

        # Return our own freed pages to the kernel. The child process exiting
        # hands back everything IT held, but the harness side (buffers, parsed
        # GGUF metadata, telemetry) sits in our heap, where free() returns to
        # the allocator arena and not to the OS. Until this was shared out of
        # image/vram.py, image teardown was the only path that did it — which
        # is exactly why host RAM ratcheted across model swaps (spec §5.5.1 H2).
        try:
            from augmentum.resource.reclaim import trim_allocator

            trim_allocator()
        except Exception:  # never let a hygiene step fail an unload
            log.debug("stop_allocator_trim_failed", exc_info=True)

        # Evict the unloaded model's GGUF from the page cache. llama.cpp
        # mmaps the whole file, so without this every swap leaves its full
        # size resident — the ratchet that put the WSL VM at 88 GB of cache
        # and 2 GB free on 2026-07-26, at which point CUDA could no longer
        # get a host-memory backing allocation and reported the failure as
        # VRAM exhaustion on an idle GPU. See ``reclaim.drop_file_cache``.
        # Safe here specifically because the child process has already
        # exited: pages another process still maps are left alone.
        if pre_model_path:
            try:
                from augmentum.resource.reclaim import drop_file_cache

                advised_mib = sum(
                    drop_file_cache(p) for p in (pre_model_path, pre_mmproj_path) if p
                )
                if advised_mib > 0:
                    log.info(
                        "model_page_cache_dropped",
                        model=pre_model_id,
                        advised_mib=advised_mib,
                    )
            except Exception:
                log.debug("stop_page_cache_drop_failed", exc_info=True)

    async def _escalate_kill(self, pid: int) -> None:
        """Re-SIGKILL via ``os.kill`` when ``asyncio.wait()`` hangs past
        the SIGKILL window.

        ``Process.wait()`` blocks until the kernel reaps the child. Under
        WSL2+CUDA-in-Docker the child can sit in uninterruptible sleep
        (D-state) after SIGKILL while the GPU driver releases the
        device, so the asyncio waiter never returns within its 5s
        timeout even though a follow-up ``os.kill`` would do the job.

        We poll ``psutil.pid_exists`` (no signal, just /proc lookup)
        and re-send SIGKILL up to a few times. If the PID is still
        alive after escalation, log ``stop_subprocess_unkillable`` —
        the strand will be caught by ``reconcile_stranded_subprocess``
        on the next start() / idle-monitor self-heal / boot.
        """
        # signal.SIGKILL is POSIX-only; production runs in Linux Docker
        # but the test suite runs on Windows. Defensive fallback to 9
        # (the wire value) keeps the test path callable; Linux runtime
        # is unaffected.
        sigkill = getattr(signal, "SIGKILL", 9)
        for _ in range(6):  # ~3s of polling at 0.5s steps
            try:
                if not psutil.pid_exists(pid):
                    return
                os.kill(pid, sigkill)
            except ProcessLookupError:
                return
            except PermissionError as exc:
                log.error(
                    "stop_escalate_kill_permission_denied",
                    pid=pid,
                    error=str(exc),
                )
                return
            await asyncio.sleep(0.5)

        if psutil.pid_exists(pid):
            log.error(
                "stop_subprocess_unkillable",
                pid=pid,
                note=(
                    "llama-server survived SIGTERM + SIGKILL + escalation. "
                    "Manager bookkeeping will be cleared; reconcile_stranded"
                    "_subprocess on next idle tick / start() / boot will "
                    "retry the kill. Resource panel will show stranded=true."
                ),
            )

    async def _emit_vram_release_telemetry(
        self,
        *,
        pre_used_mib: int,
        model_id: str,
        gpu_layers: int,
    ) -> None:
        """Sample VRAM up to 3s post-teardown; log release delta.

        WARN threshold: a GPU-resident load that returns less than
        500 MiB after subprocess exit is almost certainly leaking —
        even a single embedded layer plus the CUDA context dwarfs
        that. The threshold is intentionally conservative; tighten
        once we have a few days of INFO baselines from real loads.
        """
        for delay in (0.5, 1.0, 1.5):
            await asyncio.sleep(delay)
            try:
                post_info = await self._sample_vram_fresh_async()
            except Exception:
                return
            post_used_mib = int(post_info.get("used_mib", 0))
            if post_used_mib <= 0:
                return
            released_mib = pre_used_mib - post_used_mib
            # Driver may take a tick to settle; keep polling if the
            # release is still growing toward the conservative floor.
            if released_mib >= 500:
                log.info(
                    "vram_release",
                    model=model_id,
                    gpu_layers=gpu_layers,
                    pre_used_mib=pre_used_mib,
                    post_used_mib=post_used_mib,
                    released_mib=released_mib,
                )
                return
        # 3s window elapsed without the driver returning a meaningful
        # chunk of VRAM. Surface for the operator — this is the WSL/
        # CUDA-in-Docker leak signature.
        log.warning(
            "vram_release_lagged",
            model=model_id,
            gpu_layers=gpu_layers,
            pre_used_mib=pre_used_mib,
            post_used_mib=post_used_mib,
            released_mib=pre_used_mib - post_used_mib,
            note=(
                "VRAM did not return within 3s of llama-server exit. "
                "Repeated occurrences on WSL2/Docker indicate a driver "
                "leak across subprocess teardown — restart the container "
                "to reclaim if VRAM ceiling becomes a problem."
            ),
        )

    async def reconcile_stranded_subprocess(self) -> bool:
        """Detect and reclaim a llama-server subprocess that is alive on
        our backend port but not tracked by this manager.

        Strands happen when the manager's Python state resets without
        the subprocess actually exiting. Three known paths:
          1. uvicorn worker swap (worker dies, a new worker constructs
             a fresh manager, but the prior worker's llama-server child
             was reparented to PID 1 and survived);
          2. ``stop()``'s SIGTERM+SIGKILL window elapses without the
             subprocess dying, while the manager's state already moved
             to IDLE (existing ``vram_release_lagged`` telemetry catches
             driver-side sticky VRAM, but not "the process is literally
             still alive");
          3. A previous manager instance constructed before this one
             and left a subprocess holding our port.

        We can't ``adopt`` a subprocess we never spawned — there's no
        asyncio.subprocess.Process handle to track — so the reclaim
        path is "terminate it cleanly; verify the port frees". Foreign
        processes we don't recognize (no ``llama-server`` in cmdline,
        no ``--port <ours>`` arg) are left alone and logged; the next
        ``start()`` will fail with a clearer EADDRINUSE signal than a
        silent VRAM hoard.

        Returns True if a strand was found and reclaimed. Never raises —
        startup paths must not crash because reconcile probing failed.
        """
        try:
            # base_url inside the guard: the never-raises contract must
            # hold even on a partially-initialized manager.
            base = self.base_url
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(base + "/health")
                if r.status_code != 200:
                    return False
        except Exception:
            return False  # nothing on the port — common case

        port_arg = f"--port {self._backend_port}"
        candidate_pid: int | None = None
        candidate_cmdline = ""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline_parts = proc.info.get("cmdline") or []
                    cmdline = " ".join(cmdline_parts)
                    if "llama-server" not in cmdline:
                        continue
                    if port_arg not in cmdline:
                        continue
                    candidate_pid = int(proc.info["pid"])
                    candidate_cmdline = cmdline
                    break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:
            log.warning(
                "reconcile_psutil_scan_failed",
                port=self._backend_port,
                error=str(exc)[:200],
            )
            return False

        if candidate_pid is None:
            log.warning(
                "reconcile_port_busy_owner_unknown",
                port=self._backend_port,
                note=(
                    "health responds on our backend port but no "
                    "llama-server process owns it — likely a foreign "
                    "service; subsequent start() will surface EADDRINUSE"
                ),
            )
            return False

        stranded_model = ""
        try:
            parts = candidate_cmdline.split()
            idx = parts.index("--model")
            if idx + 1 < len(parts):
                stranded_model = Path(parts[idx + 1]).stem
        except (ValueError, IndexError):
            pass

        log.warning(
            "reclaiming_stranded_llama_server",
            pid=candidate_pid,
            port=self._backend_port,
            model=stranded_model,
            reason="alive on our backend port but not tracked by this manager",
        )

        try:
            os.kill(candidate_pid, signal.SIGTERM)
        except ProcessLookupError:
            return True  # raced; already gone
        except PermissionError as exc:
            log.error(
                "reconcile_kill_permission_denied",
                pid=candidate_pid,
                error=str(exc),
            )
            return False

        # Up to 10s for SIGTERM to take effect — matches stop()'s window.
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    r = await client.get(base + "/health")
                    if r.status_code != 200:
                        log.info(
                            "reconcile_stranded_released",
                            pid=candidate_pid,
                            model=stranded_model,
                            signal="sigterm",
                        )
                        return True
            except Exception:
                log.info(
                    "reconcile_stranded_released",
                    pid=candidate_pid,
                    model=stranded_model,
                    signal="sigterm",
                )
                return True

        # SIGTERM ignored — escalate.
        try:
            os.kill(candidate_pid, signal.SIGKILL)
        except ProcessLookupError:
            return True

        for _ in range(10):  # up to 5s post-SIGKILL
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    r = await client.get(base + "/health")
                    if r.status_code != 200:
                        log.warning(
                            "reconcile_stranded_released",
                            pid=candidate_pid,
                            model=stranded_model,
                            signal="sigkill",
                        )
                        return True
            except Exception:
                log.warning(
                    "reconcile_stranded_released",
                    pid=candidate_pid,
                    model=stranded_model,
                    signal="sigkill",
                )
                return True

        log.error(
            "reconcile_stranded_unkillable",
            pid=candidate_pid,
            port=self._backend_port,
            model=stranded_model,
            note=(
                "subprocess survived SIGTERM and SIGKILL; manual cleanup "
                "required (docker exec ... kill -9 <pid>)"
            ),
        )
        return False

    def check_alive(self) -> bool:
        """Check if the subprocess is still running.

        Returns False if the process has exited (crash, OOM, etc.).
        Resets state to IDLE so the next request triggers a restart.
        """
        if self.state != ProcessState.READY:
            return self.state != ProcessState.IDLE
        if self.process is None:
            self._reset_actual_memory()
            self.state = ProcessState.IDLE
            return False
        if self.process.returncode is not None:
            code = self.process.returncode
            log.error("llama_server_died", pid=self.process.pid,
                      exit_code=code, model=self.model_id)
            # Reset state so next request triggers restart
            self.process = None
            old_model = self.model_path
            self.model_id = ""
            self.model_path = ""
            self._start_time = None
            self._reset_actual_memory()
            self.state = ProcessState.IDLE
            # Store for potential OOM retry
            self._last_crashed_model = old_model
            self._last_crash_code = code
            return False
        return True

    # ── Pipe draining ─────────────────────────────────────────────────

    async def _drain_pipe(self, pipe, name: str) -> None:
        """Read from a subprocess pipe continuously to prevent buffer fill.

        Logs important lines (errors, loading progress), discards the rest.
        Runs as a background task for the lifetime of the subprocess.
        """
        if pipe is None:
            return
        try:
            async for line_bytes in pipe:
                line = line_bytes.decode(errors="replace").rstrip()
                self._ingest_server_line(line, name)
        except (asyncio.CancelledError, ValueError):
            pass  # Pipe closed or task cancelled

    async def _cancel_drain_tasks(self) -> None:
        """Cancel background pipe drain tasks."""
        for task in getattr(self, "_drain_tasks", []):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._drain_tasks = []

    # ── Health check ──────────────────────────────────────────────────

    async def _wait_for_health(self, timeout: float = 120.0) -> None:
        """Poll /health until status=ok or process dies."""
        url = f"{self.base_url}/health"
        deadline = time.monotonic() + timeout

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                # Check if process died during startup
                if self.process and self.process.returncode is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup with code "
                        f"{self.process.returncode}"
                    )

                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "ok":
                            return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(0.5)

        raise TimeoutError(f"llama-server did not become healthy within {timeout}s")

    def _autofit_gpu_layers(self, profile: ModelProfile) -> int:
        """Compute optimal GPU layer count based on available VRAM.

        For MoE models, uses separate expert vs non-expert weight sizes
        to avoid the VRAM cliff — going even 1GB over physical VRAM
        causes a 5-10x slowdown due to PCIe shared memory fallback.

        Strategy for MoE:
        - Non-expert weights (attention, norm, embedding) are needed for
          every token — prioritize these for GPU
        - Expert weights are large but only a subset activate per token —
          these can stay on CPU with acceptable performance
        - Budget VRAM conservatively: it's much better to be 2GB under
          than 1GB over

        Returns the number of layers to place on GPU.
        """
        return self._autofit_gpu_layers_for(
            profile,
            ctx_size=self._ctx_size,
            kv_cache_type=self.kv_cache_type,
            gpu_layers_cap=self._gpu_layers,
            flash_attn=self.flash_attn,
        )

    def _get_vram_bytes(self) -> int:
        """Query total VRAM from nvidia-smi. Returns 0 if unavailable.

        Routes through the instance-level GPU info cache so multiple
        load-plan helpers in a single ``start()`` reuse one nvidia-smi
        subprocess instead of each blocking the loop on a fresh one.
        """
        info = self._query_gpu_info()
        return info.get("total_bytes", 0)

    def _query_gpu_info(self) -> dict:
        """GPU memory stats. Sync surface; uses the warm cache if set.

        ``start()`` pre-warms the cache via :meth:`_query_gpu_info_async`
        BEFORE the synchronous load-plan helpers run, so the
        nvidia-smi subprocess executes off the event loop. Outside
        that warm window (e.g. periodic ``status()`` calls) the cache
        respects a short TTL to bound staleness without re-shelling
        on every poll.

        Returns {total_bytes, used_bytes, free_bytes, gpu_name,
        total_mib/used_mib/free_mib} or {} when nvidia-smi is
        unavailable.
        """
        cached_at = self._gpu_info_cached_at
        if (
            self._gpu_info_cache is not None
            and cached_at > 0
            and (time.monotonic() - cached_at) < self._GPU_INFO_TTL_S
        ):
            return self._gpu_info_cache
        info = self._query_gpu_info_blocking()
        self._gpu_info_cache = info
        self._gpu_info_cached_at = time.monotonic() if info else 0.0
        return info

    async def _query_gpu_info_async(self) -> dict:
        """Async-safe GPU stats query.

        Wraps the blocking nvidia-smi subprocess in
        :func:`asyncio.to_thread` so async callers — primarily
        ``start()`` priming the load-plan path — don't stall the event
        loop. Result is written into the same instance cache the sync
        ``_query_gpu_info`` reads, so all subsequent sync helpers in
        the same ``start()`` invocation get the cached value.
        """
        info = await asyncio.to_thread(self._query_gpu_info_blocking)
        self._gpu_info_cache = info
        self._gpu_info_cached_at = time.monotonic() if info else 0.0
        return info

    async def _sample_vram_fresh_async(self) -> dict:
        """Uncached, async-safe GPU stats sample for tight before/after
        deltas. Does NOT touch ``_gpu_info_cache`` — call sites that
        need a release-verification sample within the cache TTL window
        (post-stop()) must bypass to see the true post-teardown value.
        """
        return await asyncio.to_thread(self._query_gpu_info_blocking)

    @staticmethod
    def _query_gpu_info_blocking() -> dict:
        """Direct blocking nvidia-smi call. Don't call from async paths.

        The 5-second subprocess timeout BLOCKS the event loop when
        invoked from an async context. Callers in async code must go
        through :meth:`_query_gpu_info_async` (which off-loads to a
        worker thread) or rely on the warm cache populated there.

        Returns aggregated stats (total/used/free across all GPUs) PLUS
        a ``gpu_count`` and a ``devices`` list of per-GPU dicts so
        callers can gate multi-GPU UI surface on actual device count.
        Single-GPU output is identical to the prior shape — the
        aggregated fields equal the device-0 fields.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.total,memory.used,memory.free,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return {}
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            if not lines:
                return {}
            devices: list[dict] = []
            for ln in lines:
                parts = [p.strip() for p in ln.split(", ")]
                if len(parts) < 3:
                    continue
                devices.append({
                    "total_mib": int(parts[0]),
                    "used_mib": int(parts[1]),
                    "free_mib": int(parts[2]),
                    "gpu_name": parts[3] if len(parts) > 3 else "",
                })
            if not devices:
                return {}
            # Aggregate so single-GPU callers keep working unchanged.
            # The primary device (index 0) defines the "name" + back-
            # compat byte fields.
            total_mib = sum(d["total_mib"] for d in devices)
            used_mib = sum(d["used_mib"] for d in devices)
            free_mib = sum(d["free_mib"] for d in devices)
            primary = devices[0]
            return {
                "total_bytes": primary["total_mib"] * 1024 * 1024,
                "used_bytes": primary["used_mib"] * 1024 * 1024,
                "free_bytes": primary["free_mib"] * 1024 * 1024,
                "total_mib": primary["total_mib"],
                "used_mib": primary["used_mib"],
                "free_mib": primary["free_mib"],
                "gpu_name": primary["gpu_name"],
                "gpu_count": len(devices),
                "devices": devices,
                # Aggregate fields are kept distinct from the primary-
                # device fields so existing single-GPU autofit math
                # (which uses ``total_bytes``/``free_bytes``) doesn't
                # silently over-count VRAM on multi-GPU hosts.
                "total_mib_all": total_mib,
                "used_mib_all": used_mib,
                "free_mib_all": free_mib,
            }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return {}

    def build_load_plan(
        self,
        model_path: str,
        load_options: dict[str, Any] | None = None,
        profile: ModelProfile | None = None,
    ) -> dict[str, Any]:
        """Build a user-facing plan for loading a model into the engine."""
        opts = load_options or {}
        if profile is None:
            profile = self.profile_cache.get(model_path)
            if profile is None:
                profile = scan_gguf_header(model_path)
                self.profile_cache.save(profile)

        model_max_ctx = int(profile.context_length or self._ctx_size or 8192)
        default_ctx = min(max(2048, self._ctx_size or 8192), model_max_ctx)
        requested_ctx = self._coerce_int(
            opts.get("ctx_size"),
            default=default_ctx,
            minimum=2048,
            maximum=model_max_ctx,
        )
        user_requested_ctx = opts.get("ctx_size") not in (None, "")
        batch_size = self._coerce_int(
            opts.get("batch_size"),
            default=self._batch_size,
            minimum=32,
            maximum=8192,
        )
        # Physical (per-step) batch — llama-server's --ubatch-size. Defaults
        # to 0 = "don't pass the flag" (server default 512). For CPU-MoE
        # offload prefill this is THE dominant lever: each ubatch of prompt
        # tokens pays one CPU→GPU expert-weight copy, so ub 2048-4096 cuts
        # the copy count 4-8x. Costs compute-buffer VRAM at load time.
        ubatch_size = self._coerce_int(
            opts.get("ubatch_size"),
            default=0,
            minimum=0,
            maximum=8192,
        )
        if ubatch_size:
            # ub > b is rejected by llama-server; clamp instead of failing.
            ubatch_size = min(ubatch_size, batch_size)
        gpu_layers_mode = str(opts.get("gpu_layers_mode") or "auto").strip().lower()
        if gpu_layers_mode not in {"auto", "cpu", "custom", "moe_cpu", "moe_first_n_cpu", "moe_auto_vram"}:
            gpu_layers_mode = "auto"
        # MoE modes are meaningful only for MoE models; silently fall back
        # to auto for dense models so a saved profile shared between
        # machines or imported alongside the wrong model can't wedge
        # the load.
        if gpu_layers_mode in {"moe_cpu", "moe_first_n_cpu", "moe_auto_vram"} and not profile.is_moe:
            gpu_layers_mode = "auto"
        # Default when the caller omits ``moe_cpu_layers``: all experts on
        # CPU (``profile.n_layers``). Previously defaulted to
        # ``n_layers // 2``, which assumed "half on GPU" — safe for
        # Mixtral-class models with 30-50 GB expert pools but
        # catastrophic for Qwen3.5-122B-A10B-class (70 GB expert pool),
        # because half-on-GPU = 35 GB targeted at a 24 GB card →
        # guaranteed OOM during ``fitting params to device memory``.
        # All-on-CPU is always loadable; users who want to claw back
        # VRAM should set ``gpu_layers_mode = moe_auto_vram`` (which
        # runs ``_autofit_moe_cpu_layers`` to compute the actual fit)
        # or supply ``moe_cpu_layers`` explicitly. ``minimum=0`` lets
        # an explicit ``0`` (all experts on GPU — fits for tiny MoEs)
        # round-trip cleanly; previously the ``minimum=1`` clamp
        # silently turned ``0`` into ``1`` which was an undocumented
        # off-by-one for the rare power-user case.
        moe_cpu_layers = self._coerce_int(
            opts.get("moe_cpu_layers"),
            default=max(1, profile.n_layers) if profile.n_layers else 16,
            minimum=0,
            maximum=profile.n_layers if profile.n_layers > 0 else 999,
        )
        kv_cache_type = self._normalize_kv_cache_type(opts.get("kv_cache_type", self.kv_cache_type))
        flash_attn = self._coerce_bool(opts.get("flash_attn"), self.flash_attn)
        idle_timeout = float(
            self._coerce_int(
                opts.get("idle_timeout"),
                default=int(self.idle_timeout),
                minimum=0,
                maximum=86_400,
            )
        )
        draft_model = str(opts.get("draft_model", self.draft_model) or "").strip()
        draft_max = self._coerce_int(opts.get("draft_max"), default=self.draft_max, minimum=1, maximum=32)
        draft_ctx_size = self._coerce_int(
            opts.get("draft_ctx_size"), default=self.draft_ctx_size, minimum=512, maximum=32_768,
        )
        draft_gpu_layers = self._coerce_int(
            opts.get("draft_gpu_layers"), default=self.draft_gpu_layers, minimum=0, maximum=999,
        )
        draft_min = self._coerce_int(
            opts.get("draft_min"), default=self.draft_min, minimum=0, maximum=32,
        )
        try:
            draft_p_min = float(opts.get("draft_p_min", self.draft_p_min))
        except (TypeError, ValueError):
            draft_p_min = self.draft_p_min
        draft_p_min = max(0.0, min(1.0, draft_p_min))

        # MTP self-speculation per-load overrides. When the body omits a
        # key, we fall back to the engine-wide setting (instance attr
        # hydrated from settings.engine_mtp_*). Explicit per-model
        # values let a user enable MTP only on the GGUFs that have the
        # heads without flipping the global toggle.
        mtp_enabled = self._coerce_bool(opts.get("mtp_enabled"), self.mtp_enabled)
        mtp_n_max = self._coerce_int(
            opts.get("mtp_n_max"), default=int(self.mtp_n_max or 2), minimum=1, maximum=16,
        )

        # CPU thread pool. ``0`` means "let llama-server pick its
        # default" (half the available hardware threads). Surfaced as a
        # power-user knob because the default leaves significant
        # throughput on the table for partial-offload + MoE workloads
        # where expert eval lives on CPU.
        cpu_threads = self._coerce_int(opts.get("cpu_threads"), default=0, minimum=0, maximum=256)
        cpu_threads_batch = self._coerce_int(
            opts.get("cpu_threads_batch"), default=0, minimum=0, maximum=256,
        )

        # Memory-locking knobs. ``mlock`` pins resident weights so the
        # OS can't swap them out — important for partial-offload runs
        # where the CPU-resident portion is large enough to be a swap
        # target during long sessions. Default off because mlock can
        # fail without privilege on some hosts.
        mlock = self._coerce_bool(opts.get("mlock"), False)

        # Optional V-cache quantization override. When empty the V cache
        # uses the same type as K — preserves the previous behaviour for
        # every saved profile that pre-dates this field. Set explicitly
        # for the power-user pattern of "q4_0 K + q8_0 V" (K compresses
        # cleanly, V suffers more from aggressive quant).
        kv_cache_type_v = self._normalize_kv_cache_type(opts.get("kv_cache_type_v", ""))

        # LoRA hot-load. Path to a GGUF/safetensors LoRA + optional
        # scale weight. Mirrors the draft_model validation pattern —
        # absolute path required, file-not-found surfaces as a clean
        # ValueError before any subprocess work. ``scale = 1.0`` emits
        # ``--lora <path>``; any other value uses ``--lora-scaled``.
        lora_model = str(opts.get("lora_model", "") or "").strip()
        try:
            lora_scale = float(opts.get("lora_scale", 1.0))
        except (TypeError, ValueError):
            lora_scale = 1.0
        lora_scale = max(0.0, min(2.0, lora_scale))
        if lora_model and not os.path.isfile(lora_model):
            raise ValueError(
                f"LoRA model not found at path: {lora_model}. "
                "Pick a different LoRA or clear the field."
            )

        # Sampler seed. ``-1`` (or any negative) means "random per
        # request"; non-negative pins for reproducibility. llama-server
        # accepts the flag at startup and uses it as the seed for the
        # process-wide RNG.
        seed = self._coerce_int(opts.get("seed"), default=-1, minimum=-1, maximum=2**31 - 1)

        # Multi-GPU placement. ``tensor_split`` is a comma-separated
        # list of relative weights ("24,16" splits the layer count 60/40
        # across two GPUs). ``main_gpu`` picks which device gets the
        # non-distributable bits (output layer, KV reductions).
        # ``split_mode`` controls how individual tensors are sliced:
        # ``layer`` distributes whole layers; ``row`` splits tensor
        # rows; ``none`` keeps everything on main_gpu.
        tensor_split = str(opts.get("tensor_split", "") or "").strip()
        main_gpu = self._coerce_int(opts.get("main_gpu"), default=0, minimum=0, maximum=15)
        split_mode = str(opts.get("split_mode", "") or "").strip().lower()
        if split_mode not in {"", "layer", "row", "none"}:
            split_mode = ""

        # Chat template override (per-model). Modes:
        #   "embedded" — use the GGUF's tokenizer_config.json template (--jinja)
        #   "builtin"  — let llama-server pick its hard-coded default (no --jinja)
        #   "custom"   — user-provided Jinja content; written to disk + --chat-template-file
        # ``reasoning_format`` overrides the global engine_reasoning_format
        # ("deepseek" extracts <think> to reasoning_content; "none" preserves inline).
        valid_modes = {"embedded", "builtin", "custom"}
        chat_template_mode = str(opts.get("chat_template_mode") or "embedded").strip().lower()
        if chat_template_mode not in valid_modes:
            chat_template_mode = "embedded"
        chat_template_content = str(opts.get("chat_template_content") or "")
        if chat_template_mode == "custom" and not chat_template_content.strip():
            # Custom mode without content is meaningless — fall back to embedded.
            chat_template_mode = "embedded"
            chat_template_content = ""
        valid_reasoning = {"deepseek", "none", "auto", ""}
        reasoning_format = str(opts.get("reasoning_format") or "").strip().lower()
        if reasoning_format not in valid_reasoning:
            reasoning_format = ""
        # Chat template kwargs — forwarded as `--chat-template-kwargs` to
        # llama-server. Several reasoning models branch on these (most notably
        # GLM-4.x's `enable_thinking` and `clear_thinking`, Qwen3's
        # `enable_thinking`). Stored as a JSON string so users can paste the
        # exact form llama-server expects (`{"enable_thinking": false}`).
        chat_template_kwargs = str(opts.get("chat_template_kwargs") or "").strip()
        if chat_template_kwargs:
            try:
                json.loads(chat_template_kwargs)
            except (json.JSONDecodeError, ValueError):
                # Reject malformed JSON rather than silently ignoring — better
                # to fail loudly so the user sees their typo.
                raise ValueError(
                    f"chat_template_kwargs must be valid JSON, got: {chat_template_kwargs!r}"
                ) from None

        # Pre-load draft profile so we can both (a) reserve its VRAM in
        # autofit below — without this, autofit fills the GPU with target
        # layers and the draft hits OOM at load time — and (b) reuse the
        # same loaded profile for the VRAM accounting block further down.
        # File-existence and missing-path errors raise here so the user
        # sees a clean message before any download/load work happens.
        draft_profile: ModelProfile | None = None
        if draft_model:
            if not os.path.isfile(draft_model):
                raise ValueError(
                    f"Draft model not found at path: {draft_model}. "
                    "Pick a different draft or unset speculative decoding."
                )
            try:
                draft_profile = self.profile_cache.get(draft_model)
                if draft_profile is None:
                    draft_profile = scan_gguf_header(draft_model)
                    self.profile_cache.save(draft_profile)
            except Exception as exc:
                log.warning(
                    "draft_profile_scan_failed", path=draft_model, error=str(exc),
                )
                draft_profile = None

        # VRAM the autofit must hold back from the target. Only meaningful
        # when the draft is actually GPU-resident; CPU drafts don't compete.
        # Compute reserve fudge factor (~64 MiB) covers the draft's prompt
        # workspace, which we don't model precisely.
        draft_vram_reservation = 0
        if draft_profile is not None and draft_gpu_layers > 0:
            draft_size_pre = int(draft_profile.total_size_bytes or 0)
            draft_kv_bpt_pre = self._kv_bytes_per_token(draft_profile, kv_cache_type)
            draft_vram_reservation = (
                draft_size_pre
                + draft_kv_bpt_pre * draft_ctx_size
                + 64 * 1024 * 1024
            )

        if gpu_layers_mode == "cpu":
            gpu_layers = 0
        elif gpu_layers_mode == "custom":
            gpu_layers = self._coerce_int(
                opts.get("gpu_layers"),
                default=min(self._gpu_layers, profile.n_layers or self._gpu_layers),
                minimum=0,
                maximum=profile.n_layers if profile.n_layers > 0 else None,
            )
        elif gpu_layers_mode in {"moe_cpu", "moe_first_n_cpu", "moe_auto_vram"}:
            # All non-expert tensors (attention, norms, embeddings, shared
            # FFN) go on GPU; only expert weights are offloaded — that's
            # the whole point of these modes. ``--n-gpu-layers`` is set to
            # ``profile.n_layers`` so the layer-level offload is "all in",
            # then ``--cpu-moe`` / ``--n-cpu-moe N`` strips expert tensors
            # out per-tensor. See _build_cli_args for the flag emission.
            gpu_layers = profile.n_layers if profile.n_layers > 0 else self._gpu_layers
            if gpu_layers_mode == "moe_auto_vram":
                # Compute N to maximise GPU utilisation: keep as many
                # experts on GPU as VRAM allows, push the rest to CPU.
                # Closer to LM Studio's behaviour for big MoE models —
                # gives you back the ~17 GB of headroom that pure
                # --cpu-moe leaves idle.
                moe_cpu_layers = self._autofit_moe_cpu_layers(
                    profile,
                    ctx_size=requested_ctx,
                    kv_cache_type=kv_cache_type,
                    flash_attn=flash_attn,
                    batch_size=batch_size,
                    extra_reserve_bytes=draft_vram_reservation,
                )
        else:
            gpu_layers = self._autofit_gpu_layers_for(
                profile,
                ctx_size=requested_ctx,
                kv_cache_type=kv_cache_type,
                gpu_layers_cap=self._gpu_layers,
                flash_attn=flash_attn,
                extra_reserve_bytes=draft_vram_reservation,
            )
            # Auto-promote to VRAM-balanced MoE expert offload when
            # whole-layer autofit can't fit the full model AND the model
            # is MoE. Layer-level partial offload (e.g. 10/49 layers)
            # thrashes for big MoE models because every CPU-resident
            # layer's attention path crosses PCIe per token. Keeping
            # all attention on GPU and tuning ``--n-cpu-moe`` to fill
            # the remaining VRAM with as many experts as possible is
            # close to what LM Studio does — gives ~3× the throughput
            # of pure ``--cpu-moe`` (which leaves 15+ GB idle on a 24 GB
            # consumer GPU). Observed on Qwen3.5-122B-A10B: 3.2 → 11 tok/s.
            if (
                profile.is_moe
                and profile.n_layers > 0
                and gpu_layers < profile.n_layers
            ):
                gpu_layers_mode = "moe_auto_vram"
                gpu_layers = profile.n_layers
                moe_cpu_layers = self._autofit_moe_cpu_layers(
                    profile,
                    ctx_size=requested_ctx,
                    kv_cache_type=kv_cache_type,
                    flash_attn=flash_attn,
                    batch_size=batch_size,
                    extra_reserve_bytes=draft_vram_reservation,
                )

        effective_ctx = requested_ctx
        if not user_requested_ctx:
            effective_ctx = self._cap_ctx_for_vram(profile, requested_ctx, kv_cache_type, gpu_layers)

        if gpu_layers_mode == "moe_cpu":
            # All experts on CPU, everything else (attention/norms/
            # embeddings/shared FFN) on GPU. ``non_expert_tensor_bytes``
            # is precomputed by the GGUF profile scan from per-tensor
            # name classification.
            model_gpu_bytes = int(profile.non_expert_tensor_bytes or 0)
            model_cpu_bytes = int(profile.expert_tensor_bytes or 0)
        elif gpu_layers_mode in {"moe_first_n_cpu", "moe_auto_vram"} and profile.n_layers > 0:
            # Experts of the first N layers go to CPU; remaining layers'
            # experts stay on GPU alongside all non-expert tensors. Pro-
            # rated assuming experts are uniformly distributed across
            # layers (true for every MoE we've shipped support for).
            expert_total = int(profile.expert_tensor_bytes or 0)
            non_expert_total = int(profile.non_expert_tensor_bytes or 0)
            cpu_share = min(moe_cpu_layers, profile.n_layers) / profile.n_layers
            model_cpu_bytes = int(expert_total * cpu_share)
            model_gpu_bytes = non_expert_total + (expert_total - model_cpu_bytes)
        else:
            model_gpu_bytes = self._model_gpu_bytes(profile, gpu_layers)
            model_cpu_bytes = max(0, int(profile.total_size_bytes or 0) - model_gpu_bytes)
        kv_bytes = self._kv_bytes_per_token(profile, kv_cache_type) * effective_ctx
        # KV cache lives on GPU when ANY layer offloads to GPU, including
        # the MoE modes (which set ``--n-gpu-layers`` to all layers).
        kv_gpu_bytes = kv_bytes if gpu_layers > 0 else 0
        kv_cpu_bytes = 0 if gpu_layers > 0 else kv_bytes
        gpu_workspace_bytes, cpu_workspace_bytes = self._estimate_prompt_workspace_bytes(
            profile,
            batch_size=batch_size,
            flash_attn=flash_attn,
            gpu_layers=gpu_layers,
        )
        compute_reserve = self._compute_reserve_bytes(flash_attn) if gpu_layers > 0 else 0
        process_overhead = 256 * 1024**2

        # Speculative decoding adds a second model + its own KV cache to the
        # working set. Account for it here so the user sees the real VRAM
        # cost in the plan card (eliminates the "draft memory not included"
        # warning we used to ship with). Vocab/architecture mismatch fails
        # fast: llama-server would otherwise hard-error at startup with
        # ``draft model vocab does not match`` and the user only sees a
        # generic load failure.
        draft_vocab_warning = ""
        draft_gpu_bytes = 0
        draft_cpu_bytes = 0
        draft_kv_gpu_bytes = 0
        draft_kv_cpu_bytes = 0
        if draft_profile is not None:
            target_vocab = int(profile.n_vocab or 0)
            draft_vocab = int(draft_profile.n_vocab or 0)
            if target_vocab and draft_vocab and target_vocab != draft_vocab:
                raise ValueError(
                    f"Draft model vocab ({draft_vocab:,}) does not match target ({target_vocab:,}). "
                    f"Speculative decoding requires identical tokenizers — pick a draft from the same model family."
                )
            # Architecture compatibility: allow prefix-matched variants so
            # MTP / EAGLE / draft-suffix pairings work. Google ships
            # gemma-4-31b-it-assistant as `gemma4_mtp` (vs target's
            # `gemma4`); DeepSeek V3's MTP head and Qwen's spec drafts
            # follow the same convention. Vocab equality (above) is the
            # only hard requirement — llama-server enforces that at
            # startup. If neither arch is a prefix of the other, the pair
            # is almost certainly wrong (different model families) and we
            # surface a clean error before the runtime fails.
            target_arch = (profile.architecture or "").lower()
            draft_arch = (draft_profile.architecture or "").lower()
            if (
                target_arch and draft_arch
                and target_arch != draft_arch
                and not target_arch.startswith(draft_arch)
                and not draft_arch.startswith(target_arch)
            ):
                raise ValueError(
                    f"Draft architecture '{draft_profile.architecture}' is not "
                    f"compatible with target '{profile.architecture}'. "
                    f"Pair drafts with their matching family (e.g. gemma-4-31b-it-assistant for gemma-4-31b)."
                )
            if (
                profile.chat_template_hash
                and draft_profile.chat_template_hash
                and profile.chat_template_hash != draft_profile.chat_template_hash
            ):
                draft_vocab_warning = (
                    "Draft model uses a different chat template than the target. "
                    "Spec decoding will work, but output sampling may behave subtly differently."
                )

            draft_size_bytes = int(draft_profile.total_size_bytes or 0)
            if draft_gpu_layers > 0:
                draft_gpu_bytes = draft_size_bytes
            else:
                draft_cpu_bytes = draft_size_bytes
            draft_kv_bpt = self._kv_bytes_per_token(draft_profile, kv_cache_type)
            draft_kv_total = draft_kv_bpt * draft_ctx_size
            if draft_gpu_layers > 0:
                draft_kv_gpu_bytes = draft_kv_total
            else:
                draft_kv_cpu_bytes = draft_kv_total

        steady_vram_mb = int((model_gpu_bytes + kv_gpu_bytes + draft_gpu_bytes + draft_kv_gpu_bytes) // (1024 * 1024))
        steady_ram_mb = int((model_cpu_bytes + kv_cpu_bytes + draft_cpu_bytes + draft_kv_cpu_bytes + process_overhead) // (1024 * 1024))
        estimated_vram_mb = int((model_gpu_bytes + kv_gpu_bytes + draft_gpu_bytes + draft_kv_gpu_bytes + compute_reserve + gpu_workspace_bytes) // (1024 * 1024))
        estimated_ram_mb = int((model_cpu_bytes + kv_cpu_bytes + draft_cpu_bytes + draft_kv_cpu_bytes + process_overhead + cpu_workspace_bytes) // (1024 * 1024))

        gpu = self._query_gpu_info()
        gpu_total_mib = int(gpu.get("total_mib", 0) or 0)
        gpu_free_mib = int(gpu.get("free_mib", 0) or 0)
        # GPU count drives the multi-GPU section visibility in the UI.
        # Populated by ``_query_gpu_info_blocking`` (one row per device
        # in nvidia-smi output). When the probe fails (no nvidia-smi,
        # CPU-only host) the field is 0 and the UI section stays hidden.
        gpu_count = int(gpu.get("gpu_count", 0) or 0)
        ram = self._query_system_memory_info()
        ram_total_mib = int(ram.get("total_mib", 0) or 0)
        ram_available_mib = int(ram.get("available_mib", 0) or 0)
        gpu_budget_mib = gpu_free_mib or gpu_total_mib
        fits_gpu = gpu_budget_mib <= 0 or estimated_vram_mb <= gpu_budget_mib
        fits_ram = ram_available_mib <= 0 or estimated_ram_mb <= ram_available_mib

        warnings: list[str] = []
        if draft_vocab_warning:
            warnings.append(draft_vocab_warning)
        if effective_ctx < requested_ctx:
            warnings.append(
                f"Current memory settings are likely to load around {effective_ctx:,} tokens instead of the requested {requested_ctx:,}."
            )
        elif user_requested_ctx and (not fits_gpu or not fits_ram):
            warnings.append(
                f"{requested_ctx:,} tokens will be requested as-is. If the load fails, lower context manually or move more of the model/KV cache off the GPU."
            )
        if gpu_layers_mode == "cpu":
            warnings.append("This profile keeps the model in system RAM, which is slower but uses less VRAM.")
        elif gpu_layers_mode == "custom" and profile.n_layers > 0 and gpu_layers < profile.n_layers:
            warnings.append(f"Only {gpu_layers} of {profile.n_layers} layers will be placed on the GPU.")
        elif gpu_layers_mode == "moe_cpu":
            warnings.append(
                "MoE expert offload: attention + shared weights on GPU, all experts on system RAM. "
                "First reply will be slower while experts page in; later replies are fast."
            )
        elif gpu_layers_mode == "moe_first_n_cpu" and profile.n_layers > 0:
            warnings.append(
                f"MoE expert offload: experts of the first {moe_cpu_layers} of {profile.n_layers} layers "
                f"go to system RAM; remaining experts stay on GPU."
            )
        elif gpu_layers_mode == "moe_auto_vram" and profile.n_layers > 0:
            on_gpu = profile.n_layers - moe_cpu_layers
            warnings.append(
                f"MoE expert offload (VRAM-balanced): {on_gpu} layers' experts on GPU, "
                f"{moe_cpu_layers} layers' experts on system RAM. Adjust manually via the "
                f"\"experts of first N layers on CPU\" mode if you want to leave VRAM headroom."
            )
        if gpu_budget_mib > 0 and estimated_vram_mb > gpu_budget_mib:
            if gpu_free_mib > 0:
                warnings.append("Estimated peak VRAM is above what is currently free on the GPU. Lower context, GPU layers, batch size, or use a smaller KV cache.")
            else:
                warnings.append("Estimated peak VRAM is above the GPU capacity. Lower context, GPU layers, batch size, or use a smaller KV cache.")
        if ram_available_mib > 0 and estimated_ram_mb > ram_available_mib:
            warnings.append("Estimated peak RAM is above what is currently available in system memory.")
        if not flash_attn and gpu_layers > 0:
            warnings.append("Turning Flash Attention off increases prompt-processing memory needs.")
        if batch_size > 1024:
            warnings.append("Large batch sizes can improve prompt processing speed, but they also increase peak memory use.")

        return {
            "model_id": Path(model_path).stem,
            "model_path": model_path,
            "profile": {
                "architecture": profile.architecture,
                "n_layers": profile.n_layers,
                "size_gb": profile.size_gb,
                "context_length": model_max_ctx,
                "is_moe": profile.is_moe,
            },
            "requested": {
                "ctx_size": requested_ctx,
                "batch_size": batch_size,
                "gpu_layers_mode": gpu_layers_mode,
                "gpu_layers": gpu_layers if gpu_layers_mode == "custom" else None,
                "moe_cpu_layers": moe_cpu_layers if gpu_layers_mode == "moe_first_n_cpu" else None,
                "kv_cache_type": kv_cache_type or "auto",
                "flash_attn": flash_attn,
                "idle_timeout": idle_timeout,
                "draft_model": draft_model,
                "draft_max": draft_max,
                "draft_ctx_size": draft_ctx_size,
                "draft_gpu_layers": draft_gpu_layers,
                "draft_min": draft_min,
                "draft_p_min": draft_p_min,
                "mtp_enabled": mtp_enabled,
                "mtp_n_max": mtp_n_max,
            },
            "applied": {
                "ctx_size": effective_ctx,
                "batch_size": batch_size,
                "ubatch_size": ubatch_size,
                "gpu_layers_mode": gpu_layers_mode,
                "gpu_layers": gpu_layers,
                "moe_cpu_layers": moe_cpu_layers,
                "kv_cache_type": kv_cache_type,
                "kv_cache_type_v": kv_cache_type_v,
                "flash_attn": flash_attn,
                "idle_timeout": idle_timeout,
                "cpu_threads": cpu_threads,
                "cpu_threads_batch": cpu_threads_batch,
                "mlock": mlock,
                "lora_model": lora_model,
                "lora_scale": lora_scale,
                "seed": seed,
                "tensor_split": tensor_split,
                "main_gpu": main_gpu,
                "split_mode": split_mode,
                "draft_model": draft_model,
                "draft_max": draft_max,
                "draft_ctx_size": draft_ctx_size,
                "draft_gpu_layers": draft_gpu_layers,
                "draft_min": draft_min,
                "draft_p_min": draft_p_min,
                "mtp_enabled": mtp_enabled,
                "mtp_n_max": mtp_n_max,
                "chat_template_mode": chat_template_mode,
                "chat_template_content": chat_template_content,
                "chat_template_kwargs": chat_template_kwargs,
                "reasoning_format": reasoning_format,
            },
            "memory": {
                "steady_vram_mb": steady_vram_mb,
                "steady_ram_mb": steady_ram_mb,
                "estimated_vram_mb": estimated_vram_mb,
                "estimated_ram_mb": estimated_ram_mb,
                "workspace_vram_mb": int(gpu_workspace_bytes // (1024 * 1024)),
                "workspace_ram_mb": int(cpu_workspace_bytes // (1024 * 1024)),
                "gpu_total_mib": gpu_total_mib,
                "gpu_free_mib": gpu_free_mib,
                "gpu_budget_mib": gpu_budget_mib,
                "gpu_count": gpu_count,
                "fits_gpu": fits_gpu,
                "ram_total_mib": ram_total_mib,
                "ram_available_mib": ram_available_mib,
                "fits_ram": fits_ram,
            },
            "warnings": warnings,
        }

    def _auto_cache_ram_mib(
        self,
        profile: ModelProfile | None = None,
        ctx_size: int = 0,
        kv_cache_type: str = "",
    ) -> int:
        """Auto-size the upstream ``--cache-ram`` host-memory cache for
        displaced slot KV.

        The cache's job is SESSION SWITCHING: when a request displaces
        another conversation's slot, the old state is saved here and
        restored on return (one memcpy each way instead of a full
        re-prefill). Sizing it right therefore means "hold a few full
        session states of the LOADED MODEL", not a blind fraction of
        RAM: a model-blind 16 GiB holds 100+ states of a 4B (waste) but
        only ~2 of a 122B at long context (evicts under light use).

        Model-aware: ``3 × ctx_size × kv_bytes_per_token``, clamped to
        [1 GiB, 25% of available RAM]. Three states = the active session
        plus two switched-away sessions before eviction starts. Falls
        back to the old ``min(16 GiB, 25% RAM)`` heuristic when no
        profile is available (tests, early boot).

        **The fraction is of memory we may ACTUALLY use.** This used
        ``psutil.virtual_memory().total``, which is container-blind: in
        Docker/WSL2 it reports the whole VM, so 25% of a 94 GiB VM sized
        this cache at 23.6 GiB of anonymous, unreclaimable host RAM that
        the container was never entitled to. That is bug B1 of the
        2026-07-25 incident. ``hostmem`` reads the cgroup limit instead.

        A hard ``_CACHE_RAM_ABSOLUTE_CAP_MIB`` ceiling also applies: this
        cache is a *session-switching* optimisation, and no plausible
        gain justifies it becoming the largest allocation on the box.
        Set ``engine_cache_ram_mib`` to pin it explicitly.
        """
        from augmentum.resource import hostmem

        info = hostmem.memory_info()
        # Under a real ceiling, size against what is still free — memory
        # already spoken for is not ours to hand out a second time.
        base_mib = info.available_mib if info.limited else info.total_mib
        ram_cap = max(1024, min(int(base_mib * 0.25), self._CACHE_RAM_ABSOLUTE_CAP_MIB))
        if not info.limited:
            # No cgroup bound: nothing will stop this cache from growing
            # into the host. Stay conservative and say so once.
            log.debug(
                "llama_cache_ram_unbounded_host",
                source=info.source,
                total_mib=info.total_mib,
                note="No cgroup memory limit detected; using conservative cap.",
            )
        if profile is None or ctx_size <= 0:
            return min(16384, ram_cap)
        try:
            bytes_per_token = self._kv_bytes_per_token(profile, kv_cache_type or "f16")
        except Exception:
            return min(16384, ram_cap)
        state_mib = (ctx_size * bytes_per_token) // (1024 * 1024)
        return max(1024, min(ram_cap, int(state_mib * 3)))

    @staticmethod
    def _cache_reuse_args() -> list[str]:
        """Optional ``--cache-reuse`` (mid-prompt KV chunk salvage).

        Prefix reuse forfeits everything after the FIRST divergent token;
        ``--cache-reuse N`` additionally shifts identical chunks (>= N
        tokens) after a mid-history divergence back into position —
        exactly the shape of a message edit, a regeneration, or a
        deliberately deep Author's Note, where one small change orphans
        an otherwise-identical multi-thousand-token tail.

        Safe to pass unconditionally: llama-server self-gates on
        ``llama_memory_can_shift`` and IGNORES the flag on models whose
        memory can't shift (hybrid-attention families like Qwen3.5+),
        logging "cache reuse is not supported" once per affected request.
        Verified in b9181 tools/server/server-context.cpp.
        """
        raw = getattr(settings, "engine_cache_reuse_min", 256)
        try:
            min_chunk = int(raw)
        except (TypeError, ValueError):
            min_chunk = 256
        if min_chunk <= 0:
            return []
        return ["--cache-reuse", str(min_chunk)]

    def _build_slot_scheduling_args(
        self,
        mtp_active: bool = False,
        profile: ModelProfile | None = None,
        ctx_size: int = 0,
        kv_cache_type: str = "",
    ) -> list[str]:
        """Build the slot-scheduling portion of the llama-server CLI.

        See the comment at the call site in ``_build_cli_args`` for
        the rationale on each flag and the single/multi paths.

        Resolves the tri-state ``engine_multislot_enabled`` setting:
          - ``None`` (default) → codebase recommendation
            (``MULTISLOT_DEFAULT_ENABLED``)
          - ``True`` / ``False`` → explicit user override

        ``mtp_active`` is the *effective* MTP decision from
        ``_build_cli_args`` — already gated by ``profile.has_mtp_heads``
        so we never force ``--parallel 1`` for a model that can't
        actually use the spec-decode path.
        """
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        # MTP self-speculation requires ``--parallel 1`` per the upstream
        # PR. Force single-slot only when MTP is actually being applied
        # (mtp_enabled + the loaded GGUF has built-in MTP heads). If
        # both MTP and an external draft are set, MTP wins at the
        # dispatch site — the draft is dropped, so we still want
        # ``--parallel 1`` here.
        if mtp_active:
            return ["--parallel", "1"]
        # Manager-level opt-out takes precedence over the global setting.
        # Sibling subprocesses (vision aux, etc.) pin single-slot so they
        # don't budget a multi-GiB ``--cache-ram`` warm tier for their
        # sync single-request workload.
        if self._force_single_slot:
            return ["--parallel", "1"]
        explicit = getattr(settings, "engine_multislot_enabled", None)
        multislot = MULTISLOT_DEFAULT_ENABLED if explicit is None else bool(explicit)
        if not multislot:
            return ["--parallel", "1"]

        # Multi-slot path. Pin or auto-resolve --parallel.
        pinned_parallel = getattr(settings, "engine_parallel_slots", None)
        if pinned_parallel is not None and pinned_parallel > 0:
            parallel_arg = str(int(pinned_parallel))
        else:
            # ``--parallel -1`` resolves to 4 with kv_unified=true at b8935.
            # Verified in upstream source via Phase 0.
            parallel_arg = "-1"

        # Auto-size cache-ram if the user hasn't pinned it.
        pinned_cache_ram = getattr(settings, "engine_cache_ram_mib", None)
        if pinned_cache_ram is not None and pinned_cache_ram > 0:
            cache_ram_mib = int(pinned_cache_ram)
        else:
            cache_ram_mib = self._auto_cache_ram_mib(
                profile=profile, ctx_size=ctx_size, kv_cache_type=kv_cache_type,
            )

        return [
            "--parallel", parallel_arg,
            "--kv-unified",
            "--cache-ram", str(cache_ram_mib),
            # ``--no-cache-idle-slots`` MUST be passed explicitly:
            # ``cache_idle_slots`` DEFAULTS TO TRUE at b9181 (paired
            # flag; mere absence enables it) whenever kv-unified +
            # cache-ram are active. With it on, EVERY task launch
            # saves-and-clears all idle slots into the RAM prompt cache,
            # so slots are always empty at selection time and the LCP
            # similarity router never matches anything ("selected slot
            # by LRU" 100% of the time — verified live 2026-07-02, and
            # A/B-verified that this flag restores similarity routing).
            # Reuse then hinges entirely on prompt-cache restore, which
            # (a) skips candidates below f_keep 0.25, (b) is a multi-GB
            # state memcpy per save/restore at large-model scale, and
            # (c) got poisoned every turn by the narrative prewarm's
            # then-unmatchable state — net effect: every narrative turn
            # paid a full cold prefill (12-15 min at 61k on
            # Qwen3.5-122B). With the flag OFF, idle slots keep their
            # KV in place (zero-copy reuse within a session) and the
            # prompt cache still handles cross-session displacement
            # lazily via get_available_slot's save-on-displace path.
            # Trade-off accepted: under KV-pool pressure an idle slot
            # is purged (try_clear_idle_slots) instead of pre-saved.
            "--no-cache-idle-slots",
            # Per-slot context checkpoints: 32 lets a long narrative
            # carry a deep stack of SWA/recurrent snapshots without
            # eviction-thrashing on context shift. b8935 default.
            "--ctx-checkpoints", "32",
            # Mid-prefill checkpoint cadence — drop a snapshot every N
            # tokens so an interrupted prefill (e.g. cancelled mid-
            # request) doesn't lose the work. Upstream renamed
            # ``--checkpoint-every-n-tokens`` → ``--checkpoint-min-step``
            # between b9181 and b9931 (same value semantics: minimum
            # token spacing between context checkpoints); passing the
            # old name is a hard startup error ("invalid argument"), so
            # this must track the pinned LLAMA_SERVER_VERSION.
            "--checkpoint-min-step", "8192",
        ]

    def _build_cli_args(
        self,
        profile: ModelProfile,
        model_path: str,
        gpu_layers_override: int | None = None,
        load_options: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build llama-server CLI flags from config + profile."""
        # Determine ctx size — llama-server allocates KV cache upfront at this size.
        # Start with configured max, then cap based on VRAM if needed.
        plan = self.build_load_plan(model_path, load_options=load_options, profile=profile)
        applied = dict(plan["applied"])

        ctx = int(applied["ctx_size"])
        batch_size = int(applied["batch_size"])
        ubatch_size = int(applied.get("ubatch_size") or 0)
        gpu_layers = int(applied["gpu_layers"])
        gpu_layers_mode = str(applied.get("gpu_layers_mode") or "auto")
        moe_cpu_layers = int(applied.get("moe_cpu_layers") or 0)
        cpu_threads = int(applied.get("cpu_threads") or 0)
        cpu_threads_batch = int(applied.get("cpu_threads_batch") or 0)
        mlock_enabled = bool(applied.get("mlock") or False)
        kv_cache_type_v = str(applied.get("kv_cache_type_v") or "")
        lora_model = str(applied.get("lora_model") or "")
        lora_scale = float(applied.get("lora_scale") or 1.0)
        seed = int(applied.get("seed") if applied.get("seed") is not None else -1)
        tensor_split = str(applied.get("tensor_split") or "").strip()
        main_gpu_idx = int(applied.get("main_gpu") or 0)
        split_mode = str(applied.get("split_mode") or "")
        if gpu_layers_override is not None:
            gpu_layers = max(0, int(gpu_layers_override))
            applied["gpu_layers"] = gpu_layers
            # The fit-abort retry path overrides gpu_layers downward.
            # If that demotion crosses below n_layers AND we were in a
            # MoE-offload mode, we lose the "experts off but everything
            # else on GPU" guarantee — flip mode back to whole-layer
            # offload (no --cpu-moe) so the runtime is internally
            # consistent with what we pass on the CLI.
            if gpu_layers_mode in {"moe_cpu", "moe_first_n_cpu", "moe_auto_vram"} and gpu_layers < profile.n_layers:
                gpu_layers_mode = "custom"
                moe_cpu_layers = 0
        # Remember what we're actually passing as --n-gpu-layers this attempt;
        # the fit-abort retry path steps down from here.
        self._last_effective_gpu_layers = gpu_layers

        # Pre-flight warning for partial-offload-incompatible architectures.
        # If we're about to launch with an intermediate ``--n-gpu-layers``
        # value (0 < x < n_layers) on one of these architectures, the load
        # will sched_reserve-abort during startup — the runtime parser
        # catches that and latches ``_partial_offload_incompatible``, but
        # we already burned a 5-30s subprocess spin-up to learn it. Emit
        # the actionable guidance here so the operator sees the right
        # course of action (full-GPU or full-CPU) before the failure.
        # Don't auto-override: user offload config is sacred — this is
        # diagnostic only.
        arch = (profile.architecture or "").lower()
        n_layers = int(getattr(profile, "n_layers", 0) or 0)
        if (
            arch in _PARTIAL_OFFLOAD_INCOMPATIBLE_ARCHS
            and n_layers > 0
            and 0 < gpu_layers < n_layers
        ):
            log.warning(
                "llama_server_partial_offload_preflight_warning",
                architecture=arch,
                gpu_layers=gpu_layers,
                n_layers=n_layers,
                gpu_layers_mode=gpu_layers_mode,
                model=Path(model_path).stem,
                note=(
                    "Architecture requires full-GPU or full-CPU offload. "
                    "Intermediate --n-gpu-layers will sched_reserve-abort. "
                    "Set gpu_layers_mode=auto for full-GPU autofit or 0 "
                    "for full-CPU."
                ),
            )
        kv_cache_type = str(applied["kv_cache_type"] or "")
        flash_attn = bool(applied["flash_attn"])
        # Suppress flash-attn on CPU-only loads. Flash-attn is a CUDA
        # kernel optimization for attention; with all layers on CPU
        # there is no GPU attention to accelerate, but passing
        # ``--flash-attn on`` still forces llama-server to initialize
        # a full CUDA context (~400 MB VRAM) — VRAM that the chat
        # model on the GPU sibling needs for offload. Visible in
        # SmolVLM's footprint on a single-GPU box: it runs CPU-only
        # but used to claim ~400 MB of the sibling GPU's VRAM purely for
        # an unused CUDA context, which dragged the main 35B-A3B model's
        # max ``--n-gpu-layers`` down by several layers.
        if gpu_layers == 0 and flash_attn:
            log.info(
                "flash_attn_suppressed_cpu_only_load",
                model=Path(model_path).stem,
                note=(
                    "--flash-attn forces CUDA context init even on "
                    "CPU-only loads; ~400 MB VRAM waste with no perf "
                    "benefit when attention runs on CPU"
                ),
            )
            flash_attn = False
            applied["flash_attn"] = False
        draft_model = str(applied["draft_model"] or "")
        draft_max = int(applied["draft_max"])
        draft_ctx_size = int(applied.get("draft_ctx_size", self.draft_ctx_size))
        draft_gpu_layers = int(applied.get("draft_gpu_layers", self.draft_gpu_layers))
        draft_min = int(applied.get("draft_min", self.draft_min))
        draft_p_min = float(applied.get("draft_p_min", self.draft_p_min))
        # MTP per-load overrides. ``applied`` carries the merged value
        # from build_load_plan (per-model body wins over the engine-wide
        # ``self.mtp_*`` instance attrs). Keep the fallback so a caller
        # that bypasses build_load_plan still sees the global setting.
        mtp_enabled_eff = bool(applied.get("mtp_enabled", self.mtp_enabled))
        mtp_n_max_eff = int(applied.get("mtp_n_max", self.mtp_n_max or 6))
        self.idle_timeout = float(applied["idle_timeout"])
        chat_template_mode = str(applied.get("chat_template_mode") or "embedded")
        chat_template_content = str(applied.get("chat_template_content") or "")
        chat_template_kwargs = str(applied.get("chat_template_kwargs") or "")
        reasoning_format_override = str(applied.get("reasoning_format") or "")

        # MTP capability gate. Compute the effective decision once so
        # both the slot-scheduler (forces --parallel 1) and the
        # spec-decode CLI block downstream see the same answer.
        #
        # Two preconditions for MTP to actually run:
        #   1. user opted in (``mtp_enabled``)
        #   2. the loaded GGUF advertises built-in MTP heads
        #
        # Without (2), passing ``--spec-type draft-mtp`` is a strict
        # loss: llama-server either no-ops the spec path or fails it,
        # while we still pay the ``--parallel 1`` cost (multi-slot KV
        # warm-tier disabled, no idle-slot eviction, no prefill
        # checkpoints — see Phase 5 design doc).
        #
        # When BOTH MTP heads + an external ``draft_model`` are set,
        # MTP wins (they share the same spec-decode slot in
        # llama-server and only one can be active). Operator intent for
        # a model with built-in heads is "use them"; the external draft
        # is treated as a stale leftover from before the heads existed.
        # The override is logged at the dispatch site below so the user
        # can see their draft choice was ignored.
        mtp_active = (
            mtp_enabled_eff
            and profile.has_mtp_heads
        )
        if mtp_enabled_eff and not profile.has_mtp_heads:
            log.warning(
                "llama_mtp_skipped_no_heads",
                architecture=profile.architecture,
                note=(
                    "MTP enabled but the loaded GGUF has no built-in "
                    "next-N predict heads. Skipping --spec-type "
                    "draft-mtp to avoid forcing --parallel 1 for no "
                    "speculation benefit. Disable engine_mtp_enabled "
                    "or load a model with MTP heads (DeepSeek V3/V4, "
                    "Qwen 3.6, Gemma 4 MTP builds)."
                ),
            )

        # Slot scheduling — single vs multi gated on a feature flag.
        #
        # Single-slot path (default, ``engine_multislot_enabled=False``):
        #   ``--parallel 1`` puts the entire ctx behind one queue. The
        #   original mitigation for an upstream "starved slots → failed
        #   to find available cells" error class that surfaced under
        #   ``--parallel 4`` with one big narrative conversation.
        #
        # Multi-slot path (``engine_multislot_enabled=True``):
        #   ``--parallel -1`` (auto, hardcoded to 4 at b8935) plus
        #   ``--kv-unified`` removes the static KV partition that caused
        #   the original failure — slots share one pool and allocate
        #   cells dynamically. ``--cache-ram`` provides a host-RAM warm
        #   tier of "ghost slots" for evicted KV state;
        #   ``--cache-idle-slots`` auto-evicts idle live slots into that
        #   tier when new tasks arrive. ``--ctx-checkpoints`` and
        #   ``--checkpoint-every-n-tokens`` capture per-slot SWA/recurrent
        #   state mid-prefill so context-shift doesn't tank prefix
        #   reuse. Full design + verification:
        #   docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md.
        slot_scheduling_args = self._build_slot_scheduling_args(
            mtp_active=mtp_active,
            profile=profile,
            ctx_size=ctx,
            kv_cache_type=kv_cache_type,
        )

        args: list[str] = [
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(self._backend_port),
            "--ctx-size", str(ctx),
            "--batch-size", str(batch_size),
            *(["--ubatch-size", str(ubatch_size)] if ubatch_size else []),
            "--n-gpu-layers", str(gpu_layers),
            *slot_scheduling_args,
            *self._cache_reuse_args(),
            "--metrics",
            "--perf",
        ]

        # MoE expert offload — keep attention + shared weights on GPU,
        # push expert tensors to CPU. ``--cpu-moe`` does all experts;
        # ``--n-cpu-moe N`` does experts of the first N layers only.
        # build_load_plan has already set ``--n-gpu-layers`` to
        # profile.n_layers in these modes so the layer-level decision
        # is "all in" and the tensor-level filter only strips out the
        # expert weights. Both flags require the b8733+ binary.
        #
        # ``moe_auto_vram`` is identical to ``moe_first_n_cpu`` on the
        # CLI — the only difference is where ``moe_cpu_layers`` came
        # from (computed by ``_autofit_moe_cpu_layers`` vs supplied by
        # the user). When the autofit returns N == n_layers (no expert
        # fits in remaining VRAM) we shortcut to ``--cpu-moe`` so the
        # CLI is canonical.
        if gpu_layers_mode == "moe_cpu" or (
            gpu_layers_mode in {"moe_first_n_cpu", "moe_auto_vram"}
            and profile.n_layers > 0
            and moe_cpu_layers >= profile.n_layers
        ):
            args.append("--cpu-moe")
        elif gpu_layers_mode in {"moe_first_n_cpu", "moe_auto_vram"} and moe_cpu_layers > 0:
            args.extend(["--n-cpu-moe", str(moe_cpu_layers)])

        # Skip the synchronous warmup pass when any model weights live
        # on CPU. llama-server's default warmup runs one empty forward
        # pass that has to fault every page of every CPU-resident
        # tensor — for a 122B at 10/49 layers on GPU that's ~59 GB of
        # CPU-side tensors, and on a tight system RAM budget the OS
        # page cache evicts faster than warmup can read, so the warmup
        # never completes. Observed 2026-05-15: load reached "warming
        # up the model with an empty run" then hung past the 900s
        # health_timeout, leaving a stuck process holding RAM/VRAM
        # that needed manual eviction. Skipping warmup defers the
        # cost to the first real prompt (one slow TTFT on the first
        # request) but lets /health flip to OK in seconds instead of
        # minutes. Full GPU offload keeps default warmup — it's near-
        # instant on-device and gives normal first-request latency.
        any_cpu_weights = (
            gpu_layers < profile.n_layers
            or gpu_layers_mode in {"moe_cpu", "moe_first_n_cpu", "moe_auto_vram"}
        )
        if any_cpu_weights:
            args.append("--no-warmup")

        # CPU thread pool. ``0`` (default) means "let llama-server pick"
        # — same default behaviour as before this knob existed. Both
        # flags are only emitted when set explicitly so the previous
        # behaviour is preserved for users who never touch the field.
        if cpu_threads > 0:
            args.extend(["--threads", str(cpu_threads)])
        if cpu_threads_batch > 0:
            args.extend(["--threads-batch", str(cpu_threads_batch)])

        # Memory locking — pins resident weights so the OS can't swap
        # them out. Only useful when some weights live on CPU; for
        # full-GPU loads this is a no-op but harmless.
        if mlock_enabled:
            args.append("--mlock")
            # User-set mlock is honoured (offload config is theirs), but
            # under partial offload it pins model weights in host RAM as
            # unreclaimable anonymous memory — the B2 hazard. Warn loudly
            # rather than override: this is a real, diagnosable cause of
            # host-memory exhaustion and the user needs to be able to see it.
            if n_layers > 0 and gpu_layers < n_layers:
                log.warning(
                    "llama_mlock_pins_host_weights",
                    gpu_layers=gpu_layers,
                    n_layers=n_layers,
                    note=(
                        "mlock is ON with a partial GPU offload: "
                        f"{n_layers - gpu_layers} layers will be pinned in "
                        "host RAM and cannot be reclaimed under memory "
                        "pressure. Disable mlock or fully offload."
                    ),
                )

        # LoRA hot-load. ``--lora`` for scale=1.0; ``--lora-scaled``
        # otherwise. The validation in build_load_plan already verified
        # the file exists, so we can pass the path directly.
        if lora_model:
            if abs(lora_scale - 1.0) < 1e-6:
                args.extend(["--lora", lora_model])
            else:
                args.extend(["--lora-scaled", lora_model, str(lora_scale)])

        # Sampler seed. Negative = random per request (omit flag);
        # non-negative pins the process-wide RNG.
        if seed >= 0:
            args.extend(["--seed", str(seed)])

        # Multi-GPU placement. Only meaningful on hosts with ≥2 GPUs;
        # the UI hides this section on single-GPU hosts but the backend
        # still accepts the fields (a saved profile from a 2-GPU host
        # carried over to a 1-GPU host shouldn't crash — llama-server
        # ignores extraneous values gracefully).
        if tensor_split:
            args.extend(["--tensor-split", tensor_split])
        if main_gpu_idx > 0:
            args.extend(["--main-gpu", str(main_gpu_idx)])
        if split_mode and split_mode in {"layer", "row", "none"}:
            args.extend(["--split-mode", split_mode])

        mmproj_path = str((load_options or {}).get("mmproj_path") or "").strip()
        # mmproj + MTP coexist fine — verified by direct CLI bench
        # (scripts/mtp_bench.py, B2 vs A3 configs 2026-05-18): identical
        # acceptance rate, identical gen tok/s, only the ~1 GB projector
        # weight cost on top of MTP. The Unsloth model-card note about
        # "--mmproj not yet supported with MTP" likely refers to passing
        # image INPUTS through the vision path, not loading the
        # projector for text-only generation. Keep both flags.
        if mmproj_path:
            args.extend(["--mmproj", mmproj_path])
            # Surface the KV-restore trade explicitly. Upstream llama.cpp
            # returns 501 on /slots/.../save+restore whenever --mmproj is
            # present (verified on pin b9181, 2026-05-19), so every load
            # with a projector attached pays a cold-prefill cost on every
            # follow-up turn. This log makes the cause grep-able when
            # users see ``cold_no_checkpoint`` in kv_tier_decided.
            log.info(
                "kv_restore_disabled_due_to_mmproj",
                model=Path(model_path).stem,
                mmproj=mmproj_path,
                note="llama.cpp 501s on /slots/save+restore when --mmproj is loaded",
            )
        self.current_mmproj_path = mmproj_path

        if flash_attn:
            args.extend(["--flash-attn", "on"])

        # Note: --cont-batching removed — enabled by default in llama.cpp b5000+

        # Chat template selection — three modes, surfaced in the load sheet UI:
        #   embedded (default): --jinja, use the GGUF's tokenizer_config.json
        #     template. Required for correct thinking-mode behavior on newer
        #     reasoning models (GLM-4.7, Qwen3, DeepSeek-R1).
        #   builtin: omit --jinja so llama-server picks its hard-coded default.
        #     Escape hatch for GGUFs with broken embedded templates.
        #   custom: write user-provided Jinja content to disk + pass
        #     --chat-template-file. Power-user override when neither built-in
        #     option works.
        # Global ``engine_use_jinja_template=False`` forces builtin everywhere.
        global_jinja_enabled = getattr(settings, "engine_use_jinja_template", True)
        if not global_jinja_enabled and chat_template_mode == "embedded":
            chat_template_mode = "builtin"

        if chat_template_mode == "custom" and chat_template_content.strip():
            tmpl_dir = Path(self._model_dir) / ".chat_templates"
            tmpl_path = tmpl_dir / f"{Path(model_path).stem}.jinja"
            try:
                tmpl_dir.mkdir(parents=True, exist_ok=True)
                tmpl_path.write_text(chat_template_content, encoding="utf-8")
                # llama.cpp silently ignores --chat-template-file unless --jinja
                # is also set. (Confirmed in upstream docs + tested b8935.)
                args.extend(["--jinja", "--chat-template-file", str(tmpl_path)])
            except OSError:
                # Filesystem unavailable — log + fall through to embedded so
                # the user isn't left with no template at all.
                log.warning(
                    "chat_template_write_failed_falling_back",
                    path=str(tmpl_path), exc_info=True,
                )
                args.append("--jinja")
        elif chat_template_mode == "embedded":
            args.append("--jinja")
        # builtin: emit nothing — llama-server handles it.

        # Reasoning format: per-model override > global setting. Only meaningful
        # when --jinja or --chat-template-file is in play; ignored otherwise.
        if chat_template_mode != "builtin":
            reasoning_fmt = reasoning_format_override or (
                getattr(settings, "engine_reasoning_format", "deepseek") or ""
            ).strip()
            if reasoning_fmt and reasoning_fmt != "auto":
                args.extend(["--reasoning-format", reasoning_fmt])
            # Pass the model's template kwargs (e.g. `enable_thinking`,
            # `clear_thinking` for GLM-4.x; `enable_thinking` for Qwen3).
            # Only meaningful with --jinja / --chat-template-file.
            if chat_template_kwargs:
                args.extend(["--chat-template-kwargs", chat_template_kwargs])

        # K/V cache quantization. When ``kv_cache_type_v`` is empty (the
        # default — preserves prior behaviour for every saved profile
        # that pre-dates this field), V uses the same type as K. When
        # set explicitly, V overrides independently. Lets power users
        # run e.g. ``q4_0`` K + ``q8_0`` V (K compresses cleanly, V
        # tends to suffer more from aggressive quant).
        if kv_cache_type:
            args.extend(["--cache-type-k", kv_cache_type])
            args.extend(["--cache-type-v", kv_cache_type_v or kv_cache_type])
        elif kv_cache_type_v:
            # Edge case: V override set but K left on default. Apply V
            # only; llama-server keeps its default for K.
            args.extend(["--cache-type-v", kv_cache_type_v])

        # MTP self-speculation takes precedence over an external draft
        # model — they share the spec-decoding code path and llama-server
        # only honors one. Operator intent for a model with built-in
        # heads is "use them instead of a separate draft", so MTP wins
        # and the external draft is dropped from the CLI. ``mtp_active``
        # was already computed above (mtp_enabled + has_mtp_heads).
        if mtp_active:
            if draft_model:
                log.info(
                    "llama_draft_model_overridden_by_mtp",
                    draft_model=draft_model,
                    note=(
                        "Both MTP heads and an external draft model "
                        "are configured. Using the model's own MTP "
                        "heads; the external draft is ignored. Clear "
                        "engine_mtp_enabled to use the external draft."
                    ),
                )
            # PR #22673: model's own MTP heads as the speculation source.
            n_max = max(1, int(mtp_n_max_eff or 2))
            args.extend(["--spec-type", "draft-mtp"])
            args.extend(["--spec-draft-n-max", str(n_max)])
            # MTP cache-spill prevention. Without these, a long first
            # message can grow KV cache past available VRAM, evict
            # mmap-loaded weights from page cache, and drop generation
            # to ~2 tok/s permanently — verified empirically. fit-target
            # leaves VRAM headroom for KV growth; mlock + no-mmap pin
            # weights so they never get evicted as cache scales. See
            # carteakey.dev/blog/running-qwen3-6-mtp-locally for the
            # recipe these defaults are pulled from.
            args.extend(["--fit", "on", "--fit-target", "1536"])
            # ...but ONLY when the weights are actually going to the GPU.
            # The pinning rationale above assumes a full offload. Under a
            # partial offload (autofit on tight VRAM, or the OOM-backoff
            # ladder stepping layers down) the *host* holds those weights,
            # and --no-mmap --mlock then makes them anonymous, unswappable
            # and unreclaimable — no pressure level, no governor, and no
            # page-cache drop can ever get that memory back. That is how
            # VRAM pressure silently converted into permanent host-RAM
            # pressure in the 2026-07-25 incident (bug B2). mmap'd weights
            # under partial offload are file-backed and reclaimable, which
            # is strictly the safer failure mode.
            full_offload = n_layers > 0 and gpu_layers >= n_layers
            if full_offload:
                args.extend(["--no-mmap", "--mlock"])
            else:
                log.warning(
                    "llama_mtp_mlock_skipped_partial_offload",
                    gpu_layers=gpu_layers,
                    n_layers=n_layers,
                    note=(
                        "Partial offload: leaving weights mmap'd so host RAM "
                        "stays reclaimable. MTP may run slower; that is "
                        "preferable to pinning weights in host memory."
                    ),
                )
            log.info("llama_mtp_enabled", spec_draft_n_max=n_max)
        elif draft_model:
            args.extend(["--model-draft", draft_model])
            args.extend(["--draft-max", str(draft_max)])
            # --gpu-layers-draft and --ctx-size-draft are critical: without
            # them llama-server inherits the parent's --n-gpu-layers (which
            # may push the draft to CPU on tight-VRAM autofit) and sizes the
            # draft KV to the main ctx_size (wastes VRAM the draft can't use).
            args.extend(["--gpu-layers-draft", str(draft_gpu_layers)])
            args.extend(["--ctx-size-draft", str(draft_ctx_size)])
            if draft_min > 0:
                args.extend(["--draft-min", str(draft_min)])
            if draft_p_min > 0:
                args.extend(["--draft-p-min", f"{draft_p_min:.3f}"])

        # Enable slot save/restore for KV state persistence (scoped by model),
        # but ONLY when applicable. llama.cpp's per-slot ``/slots/{id}?action=
        # save|restore`` API is INCOMPATIBLE with a unified KV cache: under
        # ``--kv-unified`` (our default multi-slot path) the slots share one
        # cache region that can't be isolated per-slot, so those endpoints
        # return 501. Multi-slot models instead persist context via
        # ``--ctx-checkpoints`` (added in _build_slot_scheduling_args), so
        # nothing is lost — attempting slot-save there is just dead weight +
        # log noise. Pass --slot-save-path and mark the capability live only in
        # single-slot (non-unified) mode, where slot save/restore actually works.
        self._slot_save_supported = False
        # Derive once up front: both the slot-dir path (else branch) AND the
        # observation lookup-cache log lines below reference it. Defining it
        # only inside the non-unified slot branch left it unbound under
        # --kv-unified, so the cache-attached log raised UnboundLocalError
        # (swallowed + mislabeled observation_lookup_cache_probe_failed).
        model_stem = Path(model_path).stem
        kv_unified_active = "--kv-unified" in slot_scheduling_args
        if kv_unified_active:
            self._slot_dir = ""
            log.info(
                "slot_save_skipped_kv_unified",
                note="multi-slot/--kv-unified uses --ctx-checkpoints for KV "
                     "persistence; per-slot save/restore is N/A",
            )
        else:
            # Place .slots beside the actual model file rather than under the
            # primary _model_dir — multi-mount setups (host bind-mount + Docker
            # volume) often have the primary dir read-only to the container's
            # uid while extra dirs (where the model actually lives) are
            # writable. Deriving from model_path.parent guarantees the slot
            # dir is on a mount we can write to: if we just loaded a model
            # from there, we can write slot files there too.
            self._slot_dir = os.path.join(str(Path(model_path).parent), ".slots", model_stem)
            try:
                os.makedirs(self._slot_dir, exist_ok=True)
                self._evict_old_slots()
                args.extend(["--slot-save-path", self._slot_dir])
                self._slot_save_supported = True
                log.info("slot_save_enabled", slot_dir=self._slot_dir)
            except PermissionError:
                log.warning("slot_dir_not_writable", path=self._slot_dir)
                self._slot_dir = ""  # disable slot persistence, non-fatal

        # Observation Substrate (BOM) — lookup-cache hint for the loaded
        # model. Probe-only: if the operator has both enabled the gate
        # AND a cache file already exists for the configured primary
        # user against this model, append --lookup-cache-static. We DO
        # NOT trigger the export here — that's an explicit operation
        # via /api/observation/rebuild-cache so the model-start hot path
        # never blocks on a subprocess. Missing cache = silent skip.
        try:
            from augmentum.config import settings as _cfg
            if (
                getattr(_cfg, "observation_substrate_enabled", False)
                and getattr(_cfg, "observation_lookup_cache_enabled", False)
            ):
                primary_user = (
                    getattr(_cfg, "observation_primary_user_id", "") or ""
                ).strip()
                if primary_user:
                    from augmentum.observation.exporter import cache_path_for
                    cache_path = cache_path_for(primary_user, model_path)
                    if cache_path.exists() and cache_path.stat().st_size > 0:
                        args.extend(["--lookup-cache-static", str(cache_path)])
                        log.info(
                            "observation_lookup_cache_attached",
                            user_id=primary_user,
                            model_stem=model_stem,
                            cache_path=str(cache_path),
                            cache_bytes=cache_path.stat().st_size,
                        )
                    else:
                        log.info(
                            "observation_lookup_cache_skipped_not_built",
                            user_id=primary_user,
                            model_stem=model_stem,
                            expected_path=str(cache_path),
                        )
        except Exception:
            # Never let observation-substrate wiring break model start.
            log.warning("observation_lookup_cache_probe_failed", exc_info=True)

        self.current_ctx_size = ctx
        self.current_gpu_layers = gpu_layers
        self.current_batch_size = batch_size
        self.current_kv_cache_type = kv_cache_type
        self.current_flash_attn = flash_attn
        # When MTP wins, the external draft was dropped from the CLI —
        # reflect that in current_* so the KV signature + status surface
        # match what's actually running.
        self.current_draft_model = "" if mtp_active else draft_model
        self.current_draft_max = draft_max
        self.current_draft_ctx_size = draft_ctx_size
        self.current_draft_gpu_layers = draft_gpu_layers
        self.current_draft_min = draft_min
        self.current_draft_p_min = draft_p_min
        self.current_gpu_layers_mode = str(applied["gpu_layers_mode"])
        plan["applied"] = applied
        self._last_load_plan = plan

        # T2-7: snapshot the BASELINE compute reserve we're about to
        # use, so the model-loaded handler can compute
        # observed/baseline once llama-server reports its peak.
        # Baseline (not calibrated) makes each sample an independent
        # estimate of "what should the multiplier be?" — calibrated-
        # against-calibrated would self-reinforce and converge to 1.0.
        # Skip CPU-only loads: there's no GPU compute reserve to
        # calibrate against.
        if gpu_layers > 0:
            self._last_predicted_compute_reserve_bytes = (
                self._compute_reserve_baseline_bytes(flash_attn)
            )
            self._last_predicted_compute_bucket = (
                self._calibration_bucket_for(flash_attn)
            )
        else:
            self._last_predicted_compute_reserve_bytes = 0
            self._last_predicted_compute_bucket = ""
        return args

    async def _kv_warm_start(self, backend) -> None:
        """Background boot warm via the resume ladder.

        Runs detached from start() so replaying several long sessions
        never delays the load call returning READY. The ladder itself
        preempts on live traffic (``is_busy``) and every prewarm
        re-checks ProcessState, so a swap/stop mid-warm degrades to
        logged no-ops.
        """
        try:
            await backend.resume_ladder.warm_recent_sessions()
        except asyncio.CancelledError:
            log.info("kv_warm_start_cancelled")
            raise
        except Exception as exc:
            log.warning("kv_warm_on_start_failed", error=str(exc)[:200])

    async def _warm_top_session(self) -> None:
        """Restore the most-recently-used compatible session into slot 0.

        Called once after the model reaches READY. Walks the manifest
        from MRU down, picks the first row whose stored signature matches
        the live runtime, and POSTs ``/slots/0?action=restore``. Sets
        ``_warm_session_key`` so the backend knows which session is live
        in slot 0 without doing its own restore on the first request.

        Mirrors the erase-before-restore sequencing in
        ``LlamaCppBackend.restore_slot``: upstream llama.cpp's
        ``action=restore`` rejects with ``state_read_meta: failed to
        find available cells`` (HTTP 400) when the slot has any
        existing cells. On the first iteration after a fresh start
        slot 0 is empty, but if that first attempt 4xx-fails for any
        OTHER reason the slot may end up partially populated and
        reject every subsequent candidate. Erase between attempts is
        defensive insurance, near-zero cost (in-memory cell wipe), and
        loses nothing — manifest rows already correspond to disk-saved
        slot files.

        No-ops cleanly when there's no manifest, no compatible row, or
        the slot directory is missing.
        """
        from collections import Counter

        from augmentum.models.llama_cpp import kv_restore_skip_reason

        if not self._session_manifest or not self._slot_dir:
            return
        runtime = self.current_runtime_signature()
        model_key = runtime.get("model_key", "")
        if not model_key:
            return

        rows = self._session_manifest.list_model_sessions(model_key)

        # Track why each candidate was rejected so the no-warm exit can
        # explain the silence. Without this, when warm-on-start finds
        # candidates but skips every one (most commonly: ctx_size or
        # kv_cache_type changed since the saves), the user sees a slow
        # first-turn cold prefill with no log entry pointing at the
        # cause. Verified at 2026-05-20 — user hit this after a
        # ctx_size change, manifest had 5 candidates for the model, all
        # rejected as "context size changed", and the only signal was
        # the absent ``kv_session_warmed`` log.
        skip_reasons: Counter[str] = Counter()
        attempted = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            for row in rows:
                reason = kv_restore_skip_reason(row, runtime)
                if reason:
                    skip_reasons[reason] += 1
                    continue
                slot_filename = row.get("slot_filename") or ""
                if not slot_filename:
                    skip_reasons["no_slot_filename"] += 1
                    continue
                slot_path = os.path.join(self._slot_dir, slot_filename)
                if not os.path.isfile(slot_path):
                    # Manifest row points at a deleted slot file — skip it
                    # but leave the row alone (eviction handles cleanup).
                    skip_reasons["slot_file_missing"] += 1
                    continue

                attempted += 1
                # Erase first. Best-effort — a failed erase shouldn't
                # block the restore attempt; if the slot is genuinely
                # full the restore will surface its own clear 400.
                try:
                    await client.post(
                        f"{self.base_url}/slots/0?action=erase",
                        timeout=10.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "kv_warm_erase_before_restore_failed",
                        session=row.get("session_key"),
                        error=repr(exc),
                    )

                try:
                    resp = await client.post(
                        f"{self.base_url}/slots/0?action=restore",
                        json={"filename": slot_filename},
                    )
                except Exception as exc:
                    log.warning(
                        "kv_warm_on_start_post_failed",
                        session=row.get("session_key"),
                        error=str(exc)[:200],
                    )
                    return

                if resp.status_code < 400:
                    self._warm_session_key = row.get("session_key", "") or ""
                    log.info(
                        "kv_session_warmed",
                        model=model_key,
                        session=self._warm_session_key,
                        last_accessed=row.get("last_accessed", 0),
                    )
                    # Mark the manifest row as restored so list_sessions
                    # reflects the live ordering.
                    self._session_manifest.touch_session(
                        model_key=model_key,
                        session_key=self._warm_session_key,
                        ttl_days=self.kv_ttl_days_for_mode(row.get("mode", "")),
                        mode=str(row.get("mode", "")),
                        pinned=bool(row.get("pinned", 0)),
                        restored=True,
                    )
                    return

                skip_reasons[f"restore_status_{resp.status_code}"] += 1
                log.info(
                    "kv_warm_on_start_skipped",
                    session=row.get("session_key"),
                    status=resp.status_code,
                )
                # Fall through and try the next candidate. The next
                # iteration's erase undoes any partial state this 4xx
                # restore might have left in the slot.

        # Reached end of loop without warming. If the manifest had rows
        # at all, log why so the first slow turn after a layout change
        # (ctx_size, kv_cache_type, model_path) is greppable.
        if rows:
            log.info(
                "kv_warm_on_start_no_candidate",
                model=model_key,
                rows_considered=len(rows),
                attempted=attempted,
                skip_reasons=dict(skip_reasons),
            )

    def _cleanup_expired_slots(self) -> None:
        """Remove expired slot saves before the model starts."""
        if not self._session_manifest or not os.path.isdir(self._slot_dir):
            return
        for row in self._session_manifest.list_expired_sessions(
            self._slot_dir,
            pinned_sessions=self._pinned_sessions,
        ):
            slot_path = os.path.join(self._slot_dir, row.get("slot_filename", ""))
            try:
                if slot_path and os.path.isfile(slot_path):
                    os.unlink(slot_path)
            except OSError:
                pass
            self._session_manifest.delete_session(row["model_key"], row["session_key"])
            log.info(
                "slot_expired",
                session=row["session_key"],
                model=row["model_key"],
                path=slot_path,
            )

    def _evict_old_slots(self, max_slots: int | None = None) -> None:
        """Keep only the N most recently modified slot saves.

        Each slot save can be 200-500MB (full KV cache state).
        Without eviction, disk usage grows unbounded.

        Pinned sessions (those in _pinned_sessions) are protected from
        eviction and don't count toward the max_slots limit.
        """
        if not os.path.isdir(self._slot_dir):
            return
        self._cleanup_expired_slots()

        if max_slots is None:
            max_slots = max(0, int(self.kv_max_snapshots_per_model))
        if max_slots <= 0:
            return

        manifest_by_filename: dict[str, dict[str, Any]] = {}
        if self._session_manifest:
            manifest_by_filename = {
                row.get("slot_filename", ""): row
                for row in self._session_manifest.list_sessions(self._slot_dir)
            }

        pinned: list[tuple[float, str]] = []
        unpinned: list[tuple[float, str]] = []
        orphan_count = 0

        for entry in os.scandir(self._slot_dir):
            if not entry.is_file():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue

            # Check if this save belongs to a pinned session
            meta = manifest_by_filename.get(entry.name)
            if meta is None:
                orphan_count += 1
            session_key = meta.get("session_key", "") if meta else ""
            last_accessed = float(meta.get("last_accessed", 0) or 0) if meta else 0.0
            sort_ts = last_accessed or mtime
            is_pinned = False
            if session_key:
                is_pinned = self.session_is_pinned(session_key, meta.get("mode", ""))
            if is_pinned:
                pinned.append((sort_ts, entry.path))
            else:
                unpinned.append((sort_ts, entry.path))

        if orphan_count:
            log.info(
                "slot_orphans_present",
                slot_dir=self._slot_dir,
                count=orphan_count,
                note="slot files with no manifest row; will evict via LRU",
            )

        if len(unpinned) <= max_slots:
            return

        # Sort oldest first, delete excess unpinned only
        unpinned.sort()
        to_delete = unpinned[: len(unpinned) - max_slots]
        for _mtime, path in to_delete:
            try:
                os.unlink(path)
                if self._session_manifest:
                    meta = manifest_by_filename.get(Path(path).name)
                    if meta:
                        self._session_manifest.delete_session(
                            meta["model_key"],
                            meta["session_key"],
                        )
                log.info("slot_evicted", path=path)
            except OSError:
                pass

    def _pinned_sessions_path(self) -> str:
        """Path to the pinned sessions JSON file."""
        return os.path.join(self._model_dir, ".slots", "pinned_sessions.json")

    def _load_pinned_sessions(self) -> None:
        """Load pinned sessions from disk."""
        path = self._pinned_sessions_path()
        if os.path.isfile(path):
            try:
                import json
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._pinned_sessions = set(data)
                    log.info("pinned_sessions_loaded", count=len(self._pinned_sessions))
            except Exception as exc:
                log.warning("pinned_sessions_load_failed", error=str(exc))

    def _save_pinned_sessions(self) -> None:
        """Persist pinned sessions to disk atomically.

        Pinned-session state guards KV slots from eviction. A torn
        write here (kill -9 / container OOM mid-flush) would orphan
        slots that the manager thinks are persisted but aren't —
        every pin would silently fail until the operator manually
        repaired the file. atomic_write_json gives us tmp + fsync +
        replace so a crash either preserves the prior state or
        commits the new state, never both halfway.
        """
        path = self._pinned_sessions_path()
        try:
            from augmentum.utils.atomic_io import atomic_write_json
            atomic_write_json(path, sorted(self._pinned_sessions))
        except Exception as exc:
            log.warning("pinned_sessions_save_failed", error=str(exc))

    def pin_session(self, session_fingerprint: str) -> None:
        """Pin a session so its KV slot save is protected from eviction."""
        self._pinned_sessions.add(session_fingerprint)
        self._save_pinned_sessions()
        if self._session_manifest:
            self._session_manifest.set_pinned(session_fingerprint, True)
        log.info("session_pinned", session=session_fingerprint)

    def unpin_session(self, session_fingerprint: str) -> None:
        """Unpin a session, allowing its slot save to be evicted."""
        self._pinned_sessions.discard(session_fingerprint)
        self._save_pinned_sessions()
        if self._session_manifest:
            self._session_manifest.set_pinned(session_fingerprint, False)
        log.info("session_unpinned", session=session_fingerprint)

    # Split-GGUF naming convention from llama.cpp's gguf-split tool, e.g.
    # "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf". llama-server takes the
    # first part as -m and auto-discovers the rest, so we collapse the
    # whole set down to one entry pointing at part 1.
    _MULTIPART_RE = re.compile(r"^(.+?)-(\d{1,5})-of-(\d{1,5})\.gguf$", re.IGNORECASE)

    def discover_gguf_files(self) -> list[dict[str, Any]]:
        """Scan model_dirs for .gguf files (2 levels deep).

        Multi-part GGUFs (``-NNNNN-of-NNNNN.gguf``) collapse to a single
        entry pointing at part 1 with ``size`` summed across all present
        parts. Caller treats the entry as one model.

        Returns list of {filename, path, size, modified[, parts, expected_parts, part_paths]}.
        """
        raw: list[dict[str, Any]] = []
        seen: set[str] = set()

        for base_dir in self.model_dirs:
            if not os.path.isdir(base_dir):
                continue
            self._scan_dir(base_dir, base_dir, raw, seen, depth=0)

        # Group multi-part files by (dirname, basename, expected_parts).
        # Singletons pass through unchanged.
        groups: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
        singles: list[dict[str, Any]] = []
        for entry in raw:
            m = self._MULTIPART_RE.match(entry["filename"])
            if not m:
                singles.append(entry)
                continue
            basename = m.group(1)
            idx = int(m.group(2))
            of = int(m.group(3))
            dirname = os.path.dirname(entry["path"])
            groups.setdefault((dirname, basename, of), []).append((idx, entry))

        merged: list[dict[str, Any]] = list(singles)
        for (_dirname, _basename, of), members in groups.items():
            members.sort(key=lambda t: t[0])
            indices_present = {idx for idx, _ in members}
            part1 = next((e for idx, e in members if idx == 1), None)
            if part1 is None:
                # Part 1 missing — surface every part as its own row so the
                # user can see the broken state instead of silently hiding it.
                for _idx, e in members:
                    merged.append(e)
                continue
            total_size = sum(int(e.get("size") or 0) for _idx, e in members)
            latest_mtime = max(float(e.get("modified") or 0.0) for _idx, e in members)
            entry = dict(part1)
            entry["size"] = total_size
            entry["modified"] = latest_mtime
            entry["parts"] = len(members)
            entry["expected_parts"] = of
            entry["part_paths"] = [e["path"] for _idx, e in members]
            entry["is_multipart"] = True
            entry["missing_parts"] = sorted(set(range(1, of + 1)) - indices_present)
            merged.append(entry)

        return merged

    def _scan_dir(
        self,
        base_dir: str,
        current_dir: str,
        results: list[dict[str, Any]],
        seen: set[str],
        depth: int,
    ) -> None:
        """Recursive scanner, max depth 2, skip dotdirs."""
        if depth > 2:
            return

        try:
            entries = os.scandir(current_dir)
        except OSError:
            return

        with entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue

                if entry.is_file() and entry.name.lower().endswith(".gguf"):
                    # Skip vision projector files
                    lower_name = entry.name.lower()
                    if lower_name.startswith("mmproj") or lower_name.startswith("clip-") or "-mmproj" in lower_name:
                        continue
                    real = os.path.realpath(entry.path)
                    if real not in seen:
                        seen.add(real)
                        try:
                            stat = entry.stat()
                            results.append({
                                "filename": entry.name,
                                "path": entry.path,
                                "size": stat.st_size,
                                "modified": stat.st_mtime,
                            })
                        except OSError:
                            pass

                elif entry.is_dir():
                    self._scan_dir(base_dir, entry.path, results, seen, depth + 1)

    def _find_paired_mmproj(
        self,
        model_path: str,
        profile: ModelProfile | None = None,
        *,
        quiet: bool = False,
    ) -> str:
        """Return absolute path to the mmproj/CLIP projector that pairs
        with ``model_path``, or ``""`` when no unambiguous match exists.

        Resolution order, mirroring how Ollama / Jan / cortex handle
        projector pairing (operator-declared, not heuristically guessed):

        1. **Sidecar** (``<base>.augmentum-projector.json``): an
           operator-declared pairing written by the UI's "Pair projector"
           affordance. Wins unconditionally when present and the path
           still exists on disk. This is the authoritative source.

        2. **Strict-signal sibling heuristic**: scans sibling mmproj
           candidates in the same directory and pairs only when there's
           a SPECIFIC signal (filename contains the base stem OR mmproj
           metadata names this base explicitly). Family-level match is a
           GATE — required for compatibility but never a positive score
           on its own. This avoids the prior bug where family alone
           paired dimensionally-incompatible projectors and crashed
           llama-server at startup.

        3. **Pre-load dim guard**: whichever candidate passes (1) or (2),
           we run :func:`validate_mmproj_pair` to confirm
           ``base.embedding_length == mmproj.clip.vision.projection_dim``.
           A mismatch returns ``""`` (text-only fallback) so the load
           proceeds without crashing.

        Policy: "Better text-only than wrong-pairing." If you have a
        projector that should pair but doesn't here, declare it via the
        UI; the heuristic deliberately refuses ambiguous cases.
        """

        # 1. Operator-declared sidecar wins, with dim guard.
        sidecar = read_projector_sidecar(model_path)
        if sidecar:
            ok, reason = validate_mmproj_pair(model_path, sidecar, profile)
            if ok:
                return sidecar
            # Warn ONCE per stable mismatch — this resolver re-runs on every
            # catalog scan, so an uncorrected pairing must not spam the log.
            _key = (Path(model_path).stem, sidecar, str(reason))
            if _key not in _warned_invalid_sidecars:
                _warned_invalid_sidecars.add(_key)
                log.warning(
                    "projector_sidecar_invalid",
                    model=Path(model_path).stem,
                    mmproj=sidecar,
                    reason=reason,
                    note="logged once per process; clear the bad pairing in the UI to fix",
                )
            else:
                log.debug(
                    "projector_sidecar_invalid_repeat",
                    model=Path(model_path).stem, mmproj=sidecar,
                )
            # Fall through to heuristic; operator may have a typo.
        try:
            model_dir = Path(model_path).parent
            base_stem = Path(model_path).stem
        except (TypeError, ValueError):
            return ""

        base_norm = _normalize_stem_for_match(base_stem)
        base_arch = (profile.architecture if profile else "") or ""
        base_family = _model_family_key(base_arch) if base_arch else _model_family_key(base_stem)

        candidates: list[str] = []
        try:
            for entry in os.scandir(model_dir):
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.lower().endswith(".gguf"):
                    continue
                if _MMPROJ_FILENAME_RE.search(name):
                    candidates.append(entry.path)
        except OSError:
            return ""

        if not candidates:
            return ""

        matches: list[tuple[int, str]] = []
        for cand in candidates:
            cand_stem = Path(cand).stem
            cand_norm = _normalize_stem_for_match(cand_stem)
            peeked = peek_gguf_string_keys(
                cand,
                {
                    "general.name",
                    "general.basename",
                    "general.base_model.0.name",
                    "clip.projector_type",
                },
            )
            score = 0
            disqualified = False
            family_ok = False

            # Signal 1: projector_type → family GATE (not a positive score on its own).
            #
            # NOTICE:
            # Family-level match (e.g., projector_type "qwen3vl_merger" + base
            # "qwen35moe" → both reduce to "qwen3") is necessary but NOT
            # sufficient: the same family can span variants with very different
            # n_embd (e.g., Qwen3.6-35B-A3B has n_embd=2048 while a different
            # Qwen3-VL has n_embd=4096), and a mismatched projector crashes
            # llama-server at startup with "mtmd_init_from_file: error: mismatch
            # between text model (n_embd=X) and mmproj (n_embd=Y)". We treat
            # family as a prerequisite gate and require a SECOND positive signal
            # (specific name claim or filename match) before pairing. The
            # docstring's "Better text-only than wrong-pairing" policy is now
            # actually enforced.
            # Source/context: observed Qwen3.6-35B-A3B + /models/host/mmproj.gguf
            # auto-pair followed by llama-server exit code 1 → all
            # /v1/chat/completions 500s.
            projector_type = str(peeked.get("clip.projector_type", "")).strip()
            if projector_type:
                cand_family = _model_family_key(projector_type)
                if cand_family and base_family:
                    if cand_family == base_family:
                        family_ok = True
                    else:
                        # Family explicitly named and different — hard skip.
                        continue

            # Signal 2: metadata model claim
            for key in ("general.name", "general.basename", "general.base_model.0.name"):
                claim = str(peeked.get(key, "") or "").strip()
                if not claim or not _looks_like_specific_model_claim(claim):
                    continue
                claim_norm = _normalize_stem_for_match(claim)
                claim_family = _model_family_key(claim)
                if claim_norm and (claim_norm in base_norm or base_norm in claim_norm):
                    score += 80
                elif claim_family and base_family and claim_family != base_family:
                    disqualified = True
                    break
                # Same family but different specific size — no score, no skip
            if disqualified:
                continue

            # Signal 3: filename
            if base_norm and base_norm in cand_norm:
                score += 50
            else:
                stripped = re.sub(r"^(mmproj-|clip-)", "", cand_norm)
                if _looks_like_specific_model_claim(stripped):
                    # Filename names a different specific model.
                    continue

            # Family alone is no longer enough; require at least one specific
            # signal (name claim or filename match). Anonymous projectors with
            # generic filenames (``mmproj.gguf``) will only pair when an
            # explicit sidecar declares the pairing.
            if score > 0 and (family_ok or not projector_type):
                matches.append((score, cand))

        if not matches:
            return ""

        matches.sort(key=lambda x: x[0], reverse=True)
        top_score = matches[0][0]
        tied = [p for s, p in matches if s == top_score]
        if len(tied) > 1:
            bare = [p for p in tied if Path(p).stem.lower() == "mmproj"]
            if bare:
                picked = bare[0]
            else:
                # Otherwise prefer the longest stem (most specific filename).
                picked = max(tied, key=lambda p: len(Path(p).stem))
        else:
            picked = tied[0]

        # Final gate: dim compatibility. Even strict-signal pairs can be
        # dimensionally wrong (e.g., two Qwen3-VL releases at different
        # sizes). Refuse the load rather than letting llama-server crash.
        ok, reason = validate_mmproj_pair(model_path, picked, profile)
        if not ok:
            # Discovery path (``quiet=True``) iterates every GGUF on
            # every UI refresh and hits this branch once per
            # incompatible neighbour. Drop those to debug so we don't
            # spam the logs. Explicit-load callers leave ``quiet=False``
            # so a real load-time mismatch still surfaces.
            if quiet:
                log.debug(
                    "projector_heuristic_rejected_dim_mismatch",
                    model=Path(model_path).stem,
                    mmproj=Path(picked).name,
                    reason=reason,
                )
            else:
                log.warning(
                    "projector_heuristic_rejected_dim_mismatch",
                    model=Path(model_path).stem,
                    mmproj=Path(picked).name,
                    reason=reason,
                )
            return ""
        return picked

    def suggest_mmproj_candidates(
        self,
        base_path: str,
        profile: ModelProfile | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate mmproj candidates for the UI's projector picker.

        Use when:
        - The chat dropdown's "Pair projector" affordance opens. The UI
          shows each candidate with its filename, source directory, and
          a compatibility verdict so the operator can pick correctly.

        Searches the base model's directory AND every other configured
        model directory (``model_dir`` + ``extra_model_dirs``), so a
        projector in ``/models/host`` can be picked for a base in
        ``/data/models``. Each candidate is dim-checked via
        :func:`validate_mmproj_pair`; mismatches are returned with
        ``compatible=False`` and a reason so the UI can disable them
        without hiding their existence.

        Returns a list of dicts with keys:
            ``path`` (absolute), ``filename``, ``compatible`` (bool),
            ``reason`` (str, empty when compatible), ``projector_type``
            (str, may be empty), ``projection_dim`` (int, 0 if unreadable),
            ``is_current`` (bool, True iff this is the currently-paired
            projector for the base).
        """

        current = read_projector_sidecar(base_path)
        scan_dirs: list[str] = [str(Path(base_path).parent)]
        # Every other configured model directory, so a projector in
        # /models/host can be picked for a base in /data/models, etc.
        for d in self.model_dirs:
            if d and d not in scan_dirs:
                scan_dirs.append(d)

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for d in scan_dirs:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for entry in entries:
                if not entry.is_file():
                    continue
                if not entry.name.lower().endswith(".gguf"):
                    continue
                if not _MMPROJ_FILENAME_RE.search(entry.name):
                    continue
                ap = os.path.abspath(entry.path)
                if ap in seen:
                    continue
                seen.add(ap)

                peeked = peek_gguf_string_keys(ap, {"clip.projector_type"})
                proj_type = str(peeked.get("clip.projector_type", "")).strip()
                proj_dim = _read_mmproj_projection_dim(ap)
                ok, reason = validate_mmproj_pair(base_path, ap, profile)
                out.append({
                    "path": ap,
                    "filename": entry.name,
                    "compatible": ok,
                    "reason": reason,
                    "projector_type": proj_type,
                    "projection_dim": proj_dim,
                    "is_current": (ap == current),
                })
        # Compatible candidates first, then current pair, then by filename.
        out.sort(key=lambda c: (not c["compatible"], not c["is_current"], c["filename"]))
        return out

    async def discover_models(self) -> list[dict[str, Any]]:
        """Discover files and enrich with cached profiles.

        The whole body is synchronous filesystem work — a recursive
        ``os.scandir`` walk (``discover_gguf_files``) plus a per-file mmproj
        probe that reads GGUF headers and a profile-cache lookup that can hit
        disk — so it runs in a worker thread. The fabric heartbeat calls this
        every cycle; running it inline froze the event loop for 15s+ on a
        large/slow model mount, stalling every request (``/api/resources/
        status`` hit 60-70s and tripped the event_loop_stall watchdog).
        """
        return await asyncio.to_thread(self._discover_models_sync)

    def _discover_models_sync(self) -> list[dict[str, Any]]:
        files = self.discover_gguf_files()
        enriched: list[dict[str, Any]] = []

        for f in files:
            entry: dict[str, Any] = dict(f)
            profile = self.profile_cache.get(f["path"])
            if profile:
                entry["architecture"] = profile.architecture
                entry["is_moe"] = profile.is_moe
                entry["context_length"] = profile.context_length
                entry["n_layers"] = profile.n_layers
                entry["n_tensors"] = profile.n_tensors
                entry["total_size_bytes"] = profile.total_size_bytes
                entry["has_mtp_heads"] = profile.has_mtp_heads
                # Chat-template ground truth (parsed from the GGUF's embedded
                # jinja). ``template_thinking`` is authoritative for the UI's
                # thinking-toggle detection on SFT/merged models whose display
                # name and arch were renamed away from the upstream family —
                # the name/arch regex misses them, but the template doesn't lie.
                entry["template_thinking"] = profile.template_thinking
                entry["reasoning_family"] = profile.reasoning_family
                entry["has_profile"] = True
            else:
                entry["has_mtp_heads"] = False
                entry["has_profile"] = False
            mmproj = self._find_paired_mmproj(f["path"], profile, quiet=True)
            entry["mmproj_path"] = mmproj
            entry["supports_vision"] = bool(mmproj)
            enriched.append(entry)

        return enriched

    async def scan_and_cache_profiles(self) -> int:
        """Scan all discovered GGUFs, cache profiles, return count of new.

        Same blocking-FS class as ``discover_models`` — the scandir walk plus
        ``scan_gguf_header`` byte reads per file — so it runs in a worker
        thread to keep the event loop free. The profile cache is thread-safe
        (see ``ModelProfileCache._lock``).
        """
        return await asyncio.to_thread(self._scan_and_cache_profiles_sync)

    def _scan_and_cache_profiles_sync(self) -> int:
        files = self.discover_gguf_files()
        new_count = 0

        for f in files:
            existing = self.profile_cache.get(f["path"])
            if existing is not None:
                continue

            try:
                profile = scan_gguf_header(f["path"])
                self.profile_cache.save(profile)
                new_count += 1
            except Exception as exc:
                log.warning("Failed to scan %s: %s", f["path"], exc)

        return new_count

    def _resolve_model_path(self, name: str) -> str | None:
        """Resolve filename/name to absolute path, supports fuzzy matching."""
        # Ensure .gguf extension
        if not name.lower().endswith(".gguf"):
            name = name + ".gguf"

        # Exact match in each model dir
        for d in self.model_dirs:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate

        # Fuzzy: scan discovered files for stem match (case-insensitive)
        search_stem = Path(name).stem.lower()
        for f in self.discover_gguf_files():
            file_stem = Path(f["filename"]).stem.lower()
            if file_stem == search_stem:
                return f["path"]

        return None

    def add_common_model_dirs(self) -> list[str]:
        """Add commonly-used GGUF directories that exist on this system.

        Only adds directories that actually exist AND contain at least
        one .gguf file (skips empty dirs to avoid clutter).
        Returns list of directories that were added.
        """
        added = []
        for path in _COMMON_GGUF_PATHS:
            if not path or not os.path.isdir(path) or path in self.model_dirs:
                continue
            # Only add if it contains at least one GGUF (check 2 levels)
            if self._dir_has_gguf(path):
                self.model_dirs.append(path)
                added.append(path)
        if added:
            log.info("engine_v2_common_dirs_added", dirs=added)
        return added

    @staticmethod
    def _dir_has_gguf(path: str, max_depth: int = 2) -> bool:
        """Check if a directory contains any .gguf files (up to max_depth)."""
        try:
            for entry in os.scandir(path):
                if entry.is_file() and entry.name.lower().endswith(".gguf"):
                    return True
                if entry.is_dir() and max_depth > 0 and not entry.name.startswith("."):
                    if LlamaServerManager._dir_has_gguf(entry.path, max_depth - 1):
                        return True
        except (PermissionError, OSError):
            pass
        return False

    # ── Idle timeout ────────────────────────────────────────────────

    def touch(self) -> None:
        """Record that a request was made (resets idle timer)."""
        self._last_request_time = time.monotonic()

    @property
    def is_busy(self) -> bool:
        """True iff at least one request is currently in flight.

        Companion runtime gates read this to avoid firing autonomous LLM
        calls while the primary model has the user's request mid-stream
        (KV-cache thrash, audible latency spike). See
        :func:`augmentum.companion_runtime.gates.is_primary_busy`.
        """
        return self._in_flight_count > 0

    @contextlib.asynccontextmanager
    async def request_in_flight(self):
        """Mark a request as active for the duration of the ``async with``.

        While at least one request is in flight, the idle monitor refuses
        to unload the model regardless of how long ``_last_request_time``
        has been stale. The 10-minute idle countdown only starts after
        the LAST in-flight request exits.

        Touches on entry and exit so the timer is fresh when the request
        begins (handles the rare case where a request arrives just as
        the monitor was about to fire) and again on exit (so the idle
        clock starts now, not when the request originally entered).

        Exception-safe: ``finally`` always decrements the counter, even
        on cancellation, generator close, or upstream errors.
        """
        self._in_flight_count += 1
        self.touch()
        try:
            yield
        finally:
            self._in_flight_count -= 1
            self.touch()

    def start_idle_monitor(self) -> None:
        """Start the background idle-timeout monitor.

        Checks periodically if the model has been idle longer than
        idle_timeout. If so, saves the KV slot state and stops the
        server, freeing VRAM for other tasks. The next request will
        lazy-load the model again.

        Set idle_timeout=0 to disable (model stays loaded forever).
        Set idle_timeout=-1 to disable (same as 0, Ollama convention).
        """
        if self._idle_task and not self._idle_task.done():
            return  # already running
        if self.idle_timeout <= 0:
            return  # disabled
        self._idle_task = asyncio.create_task(self._idle_monitor_loop())

    def stop_idle_monitor(self) -> None:
        """Stop the idle monitor.

        Self-call safe. ``stop()``'s very first statement is
        ``self.stop_idle_monitor()``, but ``stop()`` is itself reached
        from inside the monitor's own task body (the
        idle-timeout-fires path in :meth:`_idle_monitor_loop`).
        Naively calling ``self._idle_task.cancel()`` from there
        schedules a ``CancelledError`` against the currently-running
        task; it fires on ``stop()``'s very next ``await`` (the VRAM
        pre-sample), unwinds out of ``stop()`` BEFORE ``terminate()``
        is called, and the monitor loop catches the cancel and
        returns. The result: ``state=STOPPING`` set synchronously at
        the top of ``stop()``, ``self.process`` still points at the
        live subprocess, no further engine logs ever appear, and the
        model stays in GPU memory until container restart. This was
        the actual cause of the "Stopping llama-server, then silence"
        log pattern operators were hitting.

        Detach the task reference instead of cancelling when the
        caller IS the monitor — the body will exit naturally once
        ``stop()`` returns. External callers (a user-initiated swap,
        shutdown, etc.) still get the normal cancel-and-clear path.
        """
        task = self._idle_task
        self._idle_task = None
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if current is task:
            # Self-call from inside the monitor body — don't cancel
            # ourselves; the body returns to the loop naturally.
            return
        task.cancel()

    async def _idle_monitor_loop(self) -> None:
        """Background loop that unloads the model after idle timeout.

        Skips the unload check while any request is in flight — see
        :meth:`request_in_flight`. Idle countdown only starts after the
        last active request exits.

        Resilience: an exception raised by any iteration's body
        (transient ``time.monotonic`` weirdness, a stop() that surfaces
        an OSError, anything else) is logged with stack trace and the
        loop continues after a short backoff. Pre-2026-05 only
        ``CancelledError`` was caught — anything else silently killed
        the task. Since the task is fire-and-forget
        (``asyncio.create_task`` with nobody checking ``.exception()``),
        a single transient bug would disable idle-unload until the next
        explicit ``start_idle_monitor`` call, leaving the model
        loaded indefinitely.

        Note on the "Save KV state before unloading" behaviour: the
        per-turn ``prepare_stable_checkpoint`` call (LlamaCppBackend)
        already writes the active session's slot to disk after every
        response, so by the time idle-unload fires we have a recent
        on-disk checkpoint. The next reload restores from that record.
        """
        check_interval = min(30.0, self.idle_timeout / 2)
        while True:
            try:
                await asyncio.sleep(check_interval)

                if self.state != ProcessState.READY:
                    continue
                if self._last_request_time <= 0:
                    continue
                if self._in_flight_count > 0:
                    # Request active. Don't even consider idle. Touch so
                    # the clock isn't already ticking against this request
                    # when it eventually exits.
                    self.touch()
                    continue

                idle_secs = time.monotonic() - self._last_request_time
                if idle_secs >= self.idle_timeout:
                    log.info("idle_timeout_unloading",
                             model=self.model_id,
                             idle_s=round(idle_secs, 1),
                             timeout=self.idle_timeout,
                             in_flight=self._in_flight_count)
                    await self.stop()
                    # Self-heal: if stop()'s SIGKILL window elapsed and
                    # the subprocess survived (WSL2+CUDA D-state, etc.),
                    # manager bookkeeping is already cleared to IDLE but
                    # the strand is still holding VRAM. Reclaim now
                    # rather than waiting for the next user chat to
                    # trigger the start()-path reconcile.
                    try:
                        await self.reconcile_stranded_subprocess()
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "idle_self_heal_reconcile_failed",
                            exc_info=True,
                        )
                    return  # stop monitoring; reload will restart it
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                # Don't let one transient failure kill the monitor task.
                # Log with stack trace, brief backoff to avoid a hot
                # error loop, then continue. Persistent errors will keep
                # logging — surface in diagnostics rather than going
                # silently dark.
                log.error("idle_monitor_iteration_failed", exc_info=True)
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    return

    def _probe_backend_port_sync(self) -> bool:
        """Return True if a llama-server is answering on our port.

        Used by :meth:`status` to surface a stranded subprocess when
        manager bookkeeping says IDLE but the port still answers
        ``/health`` — gives the resource panel a visible signal
        instead of a phantom "nothing loaded." Sync because status()
        is called via ``asyncio.to_thread`` from the route handler.

        Cheap in the common case: TCP connection-refused on localhost
        returns in <1ms when no strand exists.
        """
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(self.base_url + "/health")
                return r.status_code == 200
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        """Return current status including process health."""
        uptime = None
        if self._start_time is not None and self.state == ProcessState.READY:
            uptime = round(time.monotonic() - self._start_time, 1)

        alive = self.check_alive() if self.state == ProcessState.READY else None
        pid = self.process.pid if self.process else None

        idle_s = None
        if self._last_request_time > 0 and self.state == ProcessState.READY:
            idle_s = round(time.monotonic() - self._last_request_time, 1)

        result: dict[str, Any] = {
            "state": self.state.value,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "backend_url": self.base_url,
            "uptime_s": uptime,
            "idle_s": idle_s,
            "idle_timeout": self.idle_timeout,
            "pid": pid,
            "alive": alive,
            "pinned_sessions": len(self._pinned_sessions),
            "model_dirs": len(self.model_dirs),
            "load_config": {
                "ctx_size": self.current_ctx_size,
                "gpu_layers": self.current_gpu_layers,
                "gpu_layers_mode": self.current_gpu_layers_mode,
                "batch_size": self.current_batch_size,
                "kv_cache_type": self.current_kv_cache_type or "",
                "flash_attn": self.current_flash_attn,
                "draft_model": self.current_draft_model,
                "draft_max": self.current_draft_max,
                "draft_ctx_size": self.current_draft_ctx_size,
                "draft_gpu_layers": self.current_draft_gpu_layers,
                "draft_min": self.current_draft_min,
                "draft_p_min": self.current_draft_p_min,
                "idle_timeout": self.idle_timeout,
            },
        }

        if self._last_load_plan:
            result["load_plan"] = self._last_load_plan
        actual_memory = self._actual_memory_snapshot()
        if actual_memory:
            result["actual_memory"] = actual_memory

        # Include last crash info if relevant
        if self._last_crashed_model:
            result["last_crash"] = {
                "model": self._last_crashed_model,
                "exit_code": self._last_crash_code,
            }

        # Include profile info if loaded
        if self._last_profile:
            result["profile"] = {
                "architecture": self._last_profile.architecture,
                "n_layers": self._last_profile.n_layers,
                "size_gb": self._last_profile.size_gb,
                "is_moe": self._last_profile.is_moe,
            }

        # GPU memory stats (only query when model is loaded — nvidia-smi has overhead)
        if self.state in {ProcessState.STARTING, ProcessState.DRAINING, ProcessState.READY}:
            gpu = self._query_gpu_info()
            if gpu:
                result["gpu"] = {
                    "name": gpu.get("gpu_name", ""),
                    "vram_total_mib": gpu.get("total_mib", 0),
                    "vram_used_mib": gpu.get("used_mib", 0),
                    "vram_free_mib": gpu.get("free_mib", 0),
                }
        if pid:
            ram_mb = self._query_process_ram_mb(pid)
            if ram_mb > 0:
                result["ram"] = {"rss_mb": ram_mb}

        # Strand detection: when bookkeeping says IDLE but the backend
        # port still answers /health, a llama-server subprocess survived
        # teardown (typically stop()'s SIGKILL window elapsing under
        # WSL2+CUDA driver lag). Surface so the resource panel renders
        # something other than a phantom "nothing loaded" — the GPU
        # really is occupied. Reclaim happens at the next idle tick,
        # next start(), or container reboot.
        if self.state == ProcessState.IDLE and self._probe_backend_port_sync():
            result["stranded"] = True

        return result
