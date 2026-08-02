"""Model profile cache — GGUF header parser and metadata cache.

Parses GGUF file headers to extract model architecture, layer counts,
MoE classification, and other metadata needed to configure llama-server.
Caches results to JSON on disk with an LRU-capped in-memory layer.

This is a pure-Python module with no llama-cpp-python dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# MoE expert tensor detection patterns
_DEFAULT_MOE_PATTERNS = [
    "_exps.",
    "_exps_",
    ".ffn_gate_exps.",
    ".ffn_up_exps.",
    ".ffn_down_exps.",
]

# ---------------------------------------------------------------------------
# GGUF value types (from gguf spec)
# ---------------------------------------------------------------------------

_GGUF_TYPE_SIZE = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    12: 8,  # FLOAT64
}

# GGUF tensor type -> bytes per element. Type IDs match ggml's
# ``enum ggml_type`` (see ggml/include/ggml.h). Quant block sizes come
# from ggml/src/ggml-common.h. Bytes/element = block_size_in_bytes /
# elements_per_block (QK_K=256 for K-quants and most IQ-quants;
# QK4_0=QK4_1=QK5_0=QK5_1=QK8_0=QK8_1=QK4_NL=32 for non-K).
#
# Entries 24-25, 28, 30-31 in the original table were keyed at the
# wrong IDs (collided with I8/I16/F64/BF16); profile_version was bumped
# in concert with this fix so any cached profile with those wrong
# byte-counts gets re-scanned.
_GGUF_TENSOR_TYPE_BPE = {
    0: 4.0,        # F32
    1: 2.0,        # F16
    2: 0.5625,     # Q4_0   (block 18 / 32)
    3: 0.625,      # Q4_1   (block 20 / 32)
    6: 0.6875,     # Q5_0   (block 22 / 32)
    7: 0.75,       # Q5_1   (block 24 / 32)
    8: 1.0625,     # Q8_0   (block 34 / 32)
    9: 1.125,      # Q8_1   (block 36 / 32)
    10: 0.65625,   # Q2_K   (block 168 / 256)  K-quants below
    11: 0.4297,    # Q3_K   (block 110 / 256)
    12: 0.5625,    # Q4_K   (block 144 / 256)
    13: 0.6875,    # Q5_K   (block 176 / 256)
    14: 0.8125,    # Q6_K   (block 208 / 256)
    15: 1.0625,    # Q8_K   (block 272 / 256)
    16: 0.25,      # IQ2_XXS (block 64 / 256)
    17: 0.3125,    # IQ2_XS  (block 80 / 256)
    18: 0.3125,    # IQ3_XXS (block 80 / 256)
    19: 0.21875,   # IQ1_S   (block 56 / 256)
    20: 0.5625,    # IQ4_NL  (block 18 / 32)
    21: 0.4375,    # IQ3_S   (block 112 / 256)
    22: 0.34375,   # IQ2_S   (block 88 / 256)
    23: 0.53125,   # IQ4_XS  (block 136 / 256)
    24: 1.0,       # I8
    25: 2.0,       # I16
    26: 4.0,       # I32
    27: 8.0,       # I64
    28: 8.0,       # F64
    29: 0.21875,   # IQ1_M   (block 56 / 256)
    30: 2.0,       # BF16
    34: 0.21875,   # TQ1_0   (block 56 / 256)
    35: 0.25,      # TQ2_0   (block 64 / 256)
}

# Fallback bytes/element for unknown type IDs. Lowered from F16 (2.0)
# to ~Q4 territory so a forward-incompat type ID doesn't badly
# overestimate model size and starve autofit's GPU layer count.
# Most "unknown" IDs in practice will be future IQ-class quants in
# the 3-5 bpw range, not new fp16 variants.
_GGUF_TENSOR_TYPE_BPE_FALLBACK = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShardInfo:
    """Metadata for one GGUF shard file."""

    path: str
    size_bytes: int
    mtime: float
    n_tensors: int
    data_offset: int  # byte offset where tensor data starts


@dataclass
class ModelProfile:
    """Complete cached profile for a model."""

    model_path: str  # path to shard 1 (or single file)
    model_name: str
    profile_version: int = 7  # bump → invalidates older cached profiles
    # v7: capability signals parsed from the chat_template content + tokens
    # (template_thinking / template_tool_calling / template_system /
    # reasoning_family / eos_token_ids / finetune_base) — ground truth for
    # name/arch-independent detection, the reliable signal for custom SFT.
    # v6: retain ``nextn`` metadata keys so the MTP capability gate can
    # detect built-in MTP heads (DeepSeek V3/V4, Qwen 3.6, Gemma 4 MTP
    # builds) without re-reading the GGUF header.
    # v5: corrected _GGUF_TENSOR_TYPE_BPE keys (IQ2_XXS / IQ2_XS / IQ1_S /
    # TQ1_0 / TQ2_0 were keyed at the wrong ggml type IDs; added missing
    # IQ2_S / IQ2_M / IQ3_S / IQ3_XXS / IQ3_XS / IQ4_NL / IQ4_XS / IQ1_M;
    # lowered unknown-type fallback from F16 (2.0) to ~Q4 (0.5)).
    # v4: scalar-coerce n_layers/n_heads/n_heads_kv/n_embed/context_length
    # against Gemma-4 / hybrid-arch GGUFs that publish list-typed metadata.
    created_at: float = 0.0

    # Architecture
    architecture: str = ""
    expert_count: int = 0
    expert_used_count: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_heads_kv: int = 0  # GQA: K/V heads (often << n_heads on modern models)
    n_embed: int = 0
    n_vocab: int = 0
    context_length: int = 0
    chat_template_hash: str = ""

    # Capability signals derived from the GGUF (chat_template content + tokens).
    # GROUND TRUTH: name/arch-independent, so they're correct for custom SFT
    # models whose name is meaningless and whose template may diverge from the
    # base architecture's defaults. All default to a safe "off"/empty so a
    # missing/unreadable template (or an old cache) never asserts a capability.
    template_thinking: bool = False       # template consumes a thinking kwarg
    template_tool_calling: bool = False   # template supports native tool/function calls
    template_system: bool = False         # template handles a system role
    reasoning_family: str = ""            # reasoning-delimiter style (think_tag/channel/bracket)
    eos_token_ids: list[int] = field(default_factory=list)  # stop tokens (SFT often adds)
    finetune_base: str = ""               # general.base_model.*/finetune — SFT lineage

    # Shards
    shards: list[dict] = field(default_factory=list)
    total_size_bytes: int = 0

    # Tensor summary (no individual tensor tracking)
    n_tensors: int = 0
    n_expert_tensors: int = 0
    expert_tensor_bytes: int = 0
    non_expert_tensor_bytes: int = 0

    # Computed flags
    is_moe: bool = False
    recommended_n_ctx: int = 8192

    # Extra metadata from GGUF header (scalar values only)
    metadata: dict = field(default_factory=dict)

    @property
    def has_mtp_heads(self) -> bool:
        """True iff the GGUF advertises built-in MTP / next-N predict heads.

        Used by the llama-server CLI builder to gate ``--spec-type
        draft-mtp``: enabling MTP on a model without heads is a strict
        loss (forces ``--parallel 1`` while llama-server no-ops or
        rejects the spec-decode path).

        Conservative detection — looks for any retained metadata key
        containing ``nextn`` (e.g. ``deepseek3.nextn_predict_layers``,
        ``qwen3moe.nextn_layer_count``). A positive integer value is
        required; key-present-but-zero is treated as no-heads.
        """
        for key, val in self.metadata.items():
            if "nextn" in key.lower():
                try:
                    if int(val) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    @property
    def recommended_cli_flags(self) -> dict[str, Any]:
        """Compute recommended llama-server CLI flags from profile data."""
        flags: dict[str, Any] = {
            "--ctx-size": self.recommended_n_ctx,
        }
        if self.n_layers > 0:
            flags["--n-gpu-layers"] = self.n_layers + 1  # all layers + output
        if self.is_moe:
            # MoE models: suggest smaller batch for memory
            flags["--batch-size"] = 512
        return flags

    @property
    def size_gb(self) -> float:
        return round(self.total_size_bytes / 1e9, 1) if self.total_size_bytes else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelProfile:
        d = dict(d)  # shallow copy — don't mutate caller's dict
        d.pop("profile_version", None)
        # Belt-and-suspenders: even if profile_version hadn't been bumped,
        # any cached JSON predating the v4 fix could carry list-typed
        # block_count etc. Coerce on load so the offload math never sees
        # a list. _as_int is defined later in this module — late-import
        # via globals() to avoid a circular forward reference.
        _coerce = globals().get("_as_int")
        if _coerce is not None:
            for key in (
                "n_layers", "n_heads", "n_heads_kv", "n_embed",
                "context_length", "n_vocab", "expert_count", "expert_used_count",
            ):
                if key in d:
                    d[key] = _coerce(d[key])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def cached_version(cls, d: dict) -> int:
        """Return the ``profile_version`` recorded in a cached dict, or 0."""
        try:
            return int(d.get("profile_version", 0))
        except (TypeError, ValueError):
            return 0


# ---------------------------------------------------------------------------
# GGUF capability extraction (chat_template content + tokens → ground truth)
# ---------------------------------------------------------------------------

def _detect_reasoning_family(tmpl: str) -> str:
    """Reasoning-delimiter style used by a chat template, or '' if none.

    Grounds ``utils/thinking.py``'s parser dispatch in the ACTUAL template
    rather than the arch/name (custom SFT can diverge).
    """
    if "<|channel|>" in tmpl or "<|channel>" in tmpl:
        return "channel"
    if "[THINK]" in tmpl or "[/THINK]" in tmpl:
        return "bracket"
    if "<think>" in tmpl or "</think>" in tmpl:
        return "think_tag"
    return ""


def _extract_gguf_capabilities(metadata: dict) -> dict:
    """Derive name/arch-independent capability signals from GGUF metadata.

    Ground truth for custom SFT models whose name is meaningless and whose
    chat_template may diverge from the base architecture's defaults. Never
    raises — returns safe "off"/empty defaults on any malformed input.
    """
    caps: dict[str, Any] = {
        "template_thinking": False,
        "template_tool_calling": False,
        "template_system": False,
        "reasoning_family": "",
        "eos_token_ids": [],
        "finetune_base": "",
    }
    try:
        # chat_template may be a plain string OR a list of {name, template}
        # dicts (multiple named templates); some GGUFs also publish extra
        # ``tokenizer.chat_template.<name>`` keys. Concatenate every body.
        parts: list[str] = []
        raw = metadata.get("tokenizer.chat_template", "")
        if isinstance(raw, str):
            parts.append(raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("template", "")))
        for k, v in metadata.items():
            if (
                isinstance(k, str)
                and k.startswith("tokenizer.chat_template")
                and k != "tokenizer.chat_template"
                and isinstance(v, str)
            ):
                parts.append(v)
        tmpl = "\n".join(p for p in parts if p)

        if tmpl:
            low = tmpl.lower()
            # Thinking toggle: most families read ``enable_thinking``; Kimi
            # uses a bare ``thinking`` variable. Require a jinja context for
            # the bare word so ordinary prose "thinking" doesn't false-positive.
            caps["template_thinking"] = bool(
                "enable_thinking" in low
                or (re.search(r"\bthinking\b", low) and "{%" in tmpl)
            )
            # Native tool/function calling: the template renders tool inputs
            # and/or tool-call outputs.
            caps["template_tool_calling"] = bool(
                "tool_calls" in low
                or "tool_call" in low
                or "tools" in low
                or "function_call" in low
            )
            # System role handling.
            caps["template_system"] = bool(
                re.search(r"""['"]system['"]|==\s*['"]?system""", low)
            )
            caps["reasoning_family"] = _detect_reasoning_family(tmpl)

        # EOS / stop tokens (single int or list; fold in EOT). Custom SFT
        # frequently adds these — missing them causes run-on generation.
        eos: list[int] = []
        for key in (
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
        ):
            v = metadata.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                eos.append(v)
            elif isinstance(v, list):
                eos.extend(
                    int(x) for x in v
                    if isinstance(x, int) and not isinstance(x, bool)
                )
        seen: set[int] = set()
        caps["eos_token_ids"] = [x for x in eos if not (x in seen or seen.add(x))]

        # Finetune lineage — the base a custom SFT was built on, under an
        # opaque HF name. Lets family detection resolve the base.
        for key in (
            "general.base_model.0.name",
            "general.finetune",
            "general.basename",
        ):
            v = metadata.get(key)
            if isinstance(v, str) and v.strip():
                caps["finetune_base"] = v.strip()
                break
    except Exception:  # noqa: BLE001 — extraction must never break a scan
        pass
    return caps


# ---------------------------------------------------------------------------
# GGUF header parser
# ---------------------------------------------------------------------------

def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a GGUF metadata value to a scalar int.

    Most architecture fields (``block_count``, ``head_count``, etc.)
    are scalars, but Gemma 4 / hybrid-architecture GGUFs publish them
    as per-block-type *lists*. The downstream offload math (autofit
    layer count, KV-cache sizing) treats these as plain ints and
    crashes with ``'>' not supported between instances of 'list' and
    'int'`` if a list slips through. For a list, take the max — that
    matches the layer-count semantics for hybrid SSM/attention models
    where the list represents per-layer-type counts and the largest
    is the actual number of transformer blocks.
    """
    if isinstance(value, list):
        if not value:
            return default
        try:
            return max(int(v) for v in value if v is not None)
        except (TypeError, ValueError):
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def peek_gguf_string_keys(model_path: str, keys: set[str]) -> dict[str, str]:
    """Read just the named string-valued KV pairs from a GGUF header.

    Returns a dict mapping requested keys to their string values for any
    that were found; missing keys are simply absent. Faster than
    :func:`scan_gguf_header` when only a handful of metadata strings are
    needed (e.g. ``general.name``, ``general.base_model.0.name``,
    ``clip.projector_type``) and we don't care about tensor layout.
    """
    found: dict[str, str] = {}
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return found
            struct.unpack("<I", f.read(4))  # version
            struct.unpack("<Q", f.read(8))  # n_tensors
            n_kv = struct.unpack("<Q", f.read(8))[0]
            remaining = set(keys)
            for _ in range(n_kv):
                if not remaining:
                    break
                key = _read_gguf_string(f)
                val_type = struct.unpack("<I", f.read(4))[0]
                if key in remaining and val_type == 8:  # STRING
                    found[key] = _read_gguf_string(f)
                    remaining.discard(key)
                else:
                    _skip_gguf_value(f, val_type)
    except (OSError, struct.error, ValueError, UnicodeDecodeError) as exc:
        log.debug("peek_gguf_string_keys_failed", path=model_path, error=str(exc)[:200])
    return found


def peek_gguf_uint_keys(model_path: str, keys: set[str]) -> dict[str, int]:
    """Read named integer-valued KV pairs from a GGUF header.

    Use when:
    - Validating a base GGUF / mmproj pair before passing ``--mmproj``
      to llama-server. The base's ``<arch>.embedding_length`` must equal
      the mmproj's ``clip.vision.projection_dim`` or the runtime errors
      with ``mtmd_init_from_file: mismatch between text model (n_embd=X)
      and mmproj (n_embd=Y)`` and llama-server crashes at startup.

    Expects:
    - ``keys`` is a set of GGUF metadata key names whose values are
      stored as one of the integer scalar types (UINT8/INT8/UINT16/
      INT16/UINT32/INT32/UINT64). Non-integer values are skipped.

    Returns:
    - A dict mapping each requested key found in the header to its int
      value. Missing keys are simply absent — the caller decides how to
      treat that (typically: refuse to declare compatibility).
    """

    _INT_TYPES = {0, 1, 2, 3, 4, 5, 10}  # uint/int 8/16/32/64
    _SCALAR_FORMATS: dict[int, tuple[str, int]] = {
        0: ("<B", 1), 1: ("<b", 1),
        2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4),
        10: ("<Q", 8),
    }

    found: dict[str, int] = {}
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return found
            struct.unpack("<I", f.read(4))  # version
            struct.unpack("<Q", f.read(8))  # n_tensors
            n_kv = struct.unpack("<Q", f.read(8))[0]
            remaining = set(keys)
            for _ in range(n_kv):
                if not remaining:
                    break
                key = _read_gguf_string(f)
                val_type = struct.unpack("<I", f.read(4))[0]
                if key in remaining and val_type in _INT_TYPES:
                    fmt, size = _SCALAR_FORMATS[val_type]
                    found[key] = int(struct.unpack(fmt, f.read(size))[0])
                    remaining.discard(key)
                else:
                    _skip_gguf_value(f, val_type)
    except (OSError, struct.error, ValueError, UnicodeDecodeError) as exc:
        log.debug("peek_gguf_uint_keys_failed", path=model_path, error=str(exc)[:200])
    return found


def _read_gguf_string(f) -> str:
    """Read a GGUF string (uint64 length + raw bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _skip_gguf_value(f, val_type: int) -> None:
    """Skip a GGUF metadata value based on its type."""
    if val_type in _GGUF_TYPE_SIZE:
        f.read(_GGUF_TYPE_SIZE[val_type])
    elif val_type == 8:  # STRING
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(length)
    elif val_type == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_type in _GGUF_TYPE_SIZE:
            f.read(arr_len * _GGUF_TYPE_SIZE[arr_type])
        elif arr_type == 8:
            for _ in range(arr_len):
                _skip_gguf_value(f, 8)
        else:
            raise ValueError(f"Unknown array element type: {arr_type}")
    else:
        raise ValueError(f"Unknown GGUF value type: {val_type}")


def _read_gguf_value(f, val_type: int) -> Any:
    """Read a GGUF metadata value."""
    _SCALAR_FORMATS: dict[int, tuple[str, int]] = {
        0: ("<B", 1),   # UINT8
        1: ("<b", 1),   # INT8
        2: ("<H", 2),   # UINT16
        3: ("<h", 2),   # INT16
        4: ("<I", 4),   # UINT32
        5: ("<i", 4),   # INT32
        6: ("<f", 4),   # FLOAT32
        10: ("<Q", 8),  # UINT64
        12: ("<d", 8),  # FLOAT64
    }

    if val_type in _SCALAR_FORMATS:
        fmt, size = _SCALAR_FORMATS[val_type]
        return struct.unpack(fmt, f.read(size))[0]
    if val_type == 7:  # BOOL
        return bool(struct.unpack("<B", f.read(1))[0])
    if val_type == 8:  # STRING
        return _read_gguf_string(f)
    if val_type == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len > 1000:
            # Skip large arrays (e.g. tokenizer vocab)
            if arr_type in _GGUF_TYPE_SIZE:
                f.read(arr_len * _GGUF_TYPE_SIZE[arr_type])
            elif arr_type == 8:
                for _ in range(arr_len):
                    _skip_gguf_value(f, 8)
            return f"<array[{arr_len}]>"
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    else:
        raise ValueError(f"Unknown GGUF value type: {val_type}")


def scan_gguf_header(
    model_path: str,
    *,
    moe_patterns: list[str] | None = None,
) -> ModelProfile:
    """Parse GGUF header(s) to build a ModelProfile.

    For sharded models, reads shard 1's metadata (full architecture info)
    and scans tensor descriptors from all shards.

    Parameters
    ----------
    model_path:
        Path to the GGUF file (or first shard).
    moe_patterns:
        Tensor name patterns that indicate MoE expert weights.
        Defaults to the standard patterns if not provided.
    """
    if moe_patterns is None:
        moe_patterns = _DEFAULT_MOE_PATTERNS

    t0 = time.time()
    model_path = str(model_path)

    # Determine shards
    shard_paths = [model_path]
    shard_match = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", model_path)
    if shard_match:
        base = shard_match.group(1)
        total = int(shard_match.group(3))
        shard_paths = []
        for i in range(1, total + 1):
            sp = f"{base}-{i:05d}-of-{total:05d}.gguf"
            if os.path.isfile(sp):
                shard_paths.append(sp)

    profile = ModelProfile(
        model_path=model_path,
        model_name=Path(model_path).stem,
        created_at=time.time(),
    )

    _ARCH_KEYS = {
        "expert_count", "expert_used_count", "block_count",
        "attention.head_count", "attention.head_count_kv",
        "embedding_length", "context_length",
        # MTP-head metadata. DeepSeek V3/V4 publish
        # ``<arch>.nextn_predict_layers``; Qwen 3.6 / Gemma 4 MTP builds
        # use the same family of ``nextn_*`` keys. Retain so the
        # capability gate in llama_server_manager can detect built-in
        # MTP heads without re-reading the GGUF header.
        "nextn",
    }

    total_expert_tensors = 0
    total_expert_bytes = 0
    total_non_expert_bytes = 0
    total_tensor_count = 0

    for shard_idx, shard_path in enumerate(shard_paths):
        try:
            stat = os.stat(shard_path)
            shard_info = ShardInfo(
                path=shard_path,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                n_tensors=0,
                data_offset=0,
            )

            with open(shard_path, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    log.warning("Not a GGUF file: %s", shard_path)
                    continue

                _version = struct.unpack("<I", f.read(4))[0]
                n_tensors = struct.unpack("<Q", f.read(8))[0]
                n_kv = struct.unpack("<Q", f.read(8))[0]
                shard_info.n_tensors = n_tensors

                # Read metadata
                metadata: dict[str, Any] = {}
                for _ in range(n_kv):
                    key = _read_gguf_string(f)
                    val_type = struct.unpack("<I", f.read(4))[0]
                    val = _read_gguf_value(f, val_type)
                    metadata[key] = val

                if shard_idx == 0:
                    arch = metadata.get("general.architecture", "")
                    profile.architecture = arch
                    profile.metadata = {
                        k: v for k, v in metadata.items()
                        if isinstance(v, int | float | str | bool)
                        and any(ak in k for ak in _ARCH_KEYS)
                    }

                    prefix = f"{arch}." if arch else ""
                    profile.expert_count = _as_int(metadata.get(f"{prefix}expert_count", 0))
                    profile.expert_used_count = _as_int(metadata.get(f"{prefix}expert_used_count", 0))
                    # Gemma 4 / some hybrid-arch GGUFs emit list-typed values
                    # for block_count, attention.head_count, etc. — one entry
                    # per block type. Downstream offload math expects scalar
                    # ints and crashes with "'>' not supported between list
                    # and int" when comparing the raw list. Coerce defensively
                    # via the same pattern n_vocab uses.
                    profile.n_layers = _as_int(metadata.get(f"{prefix}block_count", 0))
                    profile.n_heads = _as_int(metadata.get(f"{prefix}attention.head_count", 0))
                    # GQA: KV head count. Falls back to n_heads when missing
                    # (older non-GQA models or non-standard metadata).
                    profile.n_heads_kv = _as_int(metadata.get(
                        f"{prefix}attention.head_count_kv", profile.n_heads,
                    ))
                    profile.n_embed = _as_int(metadata.get(f"{prefix}embedding_length", 0))
                    profile.context_length = _as_int(metadata.get(f"{prefix}context_length", 0))
                    profile.n_vocab = metadata.get(
                        f"{prefix}vocab_size",
                        metadata.get("tokenizer.ggml.tokens", 0),
                    )
                    if isinstance(profile.n_vocab, list):
                        profile.n_vocab = len(profile.n_vocab)
                    profile.n_vocab = _as_int(profile.n_vocab)

                    # Chat template hash
                    chat_template = metadata.get("tokenizer.chat_template", "")
                    if isinstance(chat_template, str) and chat_template:
                        profile.chat_template_hash = hashlib.sha256(
                            chat_template.encode("utf-8")
                        ).hexdigest()[:16]

                    # Capability ground truth from the template + tokens
                    # (name/arch-independent — correct for custom SFT). Purely
                    # additive: populates the new profile fields, no behavior
                    # change here — consumers read them.
                    _caps = _extract_gguf_capabilities(metadata)
                    profile.template_thinking = _caps["template_thinking"]
                    profile.template_tool_calling = _caps["template_tool_calling"]
                    profile.template_system = _caps["template_system"]
                    profile.reasoning_family = _caps["reasoning_family"]
                    profile.eos_token_ids = _caps["eos_token_ids"]
                    profile.finetune_base = _caps["finetune_base"]

                # Read tensor descriptors (aggregate stats only)
                for _ in range(n_tensors):
                    tname = _read_gguf_string(f)
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                    dtype = struct.unpack("<I", f.read(4))[0]
                    _offset = struct.unpack("<Q", f.read(8))[0]

                    n_elements = 1
                    for d in dims:
                        n_elements *= d
                    bpe = _GGUF_TENSOR_TYPE_BPE.get(dtype, _GGUF_TENSOR_TYPE_BPE_FALLBACK)
                    size_bytes = int(n_elements * bpe)

                    is_expert = any(p in tname for p in moe_patterns)
                    total_tensor_count += 1
                    if is_expert:
                        total_expert_tensors += 1
                        total_expert_bytes += size_bytes
                    else:
                        total_non_expert_bytes += size_bytes

                shard_info.data_offset = f.tell()
                profile.shards.append(asdict(shard_info))

        except Exception as exc:
            log.warning("Failed to scan shard %s: %s", shard_path, exc)
            continue

    # Aggregate
    profile.n_tensors = total_tensor_count
    profile.n_expert_tensors = total_expert_tensors
    profile.expert_tensor_bytes = total_expert_bytes
    profile.non_expert_tensor_bytes = total_non_expert_bytes
    profile.total_size_bytes = sum(s["size_bytes"] for s in profile.shards)

    # Derived flags
    profile.is_moe = profile.expert_count > 0

    # Context recommendation based on model size
    if profile.total_size_bytes > 100 * 1024**3:
        profile.recommended_n_ctx = 4096
    elif profile.total_size_bytes > 50 * 1024**3:
        profile.recommended_n_ctx = 8192

    elapsed = time.time() - t0
    log.info(
        "GGUF scan complete: %s — %d tensors (%d expert), "
        "%d shards, %.1f GB total in %.2fs",
        profile.model_name,
        profile.n_tensors,
        profile.n_expert_tensors,
        len(profile.shards),
        profile.total_size_bytes / 1e9,
        elapsed,
    )

    return profile


# ---------------------------------------------------------------------------
# Profile cache manager
# ---------------------------------------------------------------------------

class ModelProfileCache:
    """Manages cached model profiles with on-disk LRU + in-memory layer.

    The on-disk cache is an *accelerator*, not load-bearing. If no
    writable directory is available — locked-down container, read-only
    bind-mount, restrictive WSL host permissions, foreign-UID mount,
    etc. — we silently disable disk persistence and keep the
    in-memory layer. Profiles re-scan on each restart but the engine
    still works.

    Construction is guaranteed never to raise on path/permission
    failure; this class is wired into ``LlamaServerManager.__init__``
    and any exception there cascades into ``engine_v2_init_failed``,
    which prevents the engine backend from being registered with the
    provider registry and silently breaks the model-manager UI.
    """

    _MAX_MEMORY_ENTRIES = 64

    # Standard fallback directories tried in order if the requested
    # ``cache_dir`` isn't writable. ``/data`` is the canonical Docker
    # volume mount and is writable by the container's UID by design.
    _FALLBACK_DIRS: tuple[str, ...] = ("/data/model_profiles",)

    def __init__(self, cache_dir: str = "/data/model_profiles"):
        self._memory: OrderedDict[str, ModelProfile] = OrderedDict()
        # Guards every mutation of ``_memory``. The cache is read/written
        # from the event-loop thread AND from worker threads (model discovery
        # / profile scans run via asyncio.to_thread to keep the loop free), so
        # the OrderedDict's non-atomic move_to_end/__setitem__/popitem must be
        # serialized or it can corrupt under concurrent access.
        self._lock = threading.RLock()
        self._cache_dir: Path | None = self._select_writable_dir(cache_dir)

    @property
    def cache_dir(self) -> Path | None:
        """Resolved cache directory, or ``None`` if disk persistence is disabled.

        Callers that want to colocate their own state files (e.g.
        WorkspaceCalibration) should use this resolved path so they
        share the same writable location instead of independently
        rediscovering it.
        """
        return self._cache_dir

    @property
    def disk_enabled(self) -> bool:
        return self._cache_dir is not None

    @staticmethod
    def _select_writable_dir(preferred: str) -> Path | None:
        """Try the preferred dir, then standard fallbacks. Return None if all fail.

        ``mkdir`` succeeding isn't proof of writability — quirky mounts
        (FAT/exFAT, restrictive ACLs, network filesystems mid-failover)
        can pass mkdir and still reject file creation. We probe with a
        touch+unlink so the resolved path is actually usable.
        """
        import tempfile

        candidates: list[str] = [preferred]
        for fallback in ModelProfileCache._FALLBACK_DIRS:
            if fallback not in candidates:
                candidates.append(fallback)
        tmp_fallback = str(
            Path(tempfile.gettempdir()) / "augmentum_model_profiles"
        )
        if tmp_fallback not in candidates:
            candidates.append(tmp_fallback)

        for candidate in candidates:
            try:
                p = Path(candidate)
                p.mkdir(parents=True, exist_ok=True)
                # Probe actual writability — mkdir + write are different
                # syscalls and can disagree on some filesystem configs.
                probe = p / ".augmentum_write_test"
                probe.touch()
                probe.unlink()
            except OSError:
                continue

            if str(p) != preferred:
                log.warning(
                    "profile_cache_dir_fallback requested=%s using=%s",
                    preferred,
                    str(p),
                )
            return p

        # Every candidate failed. Disk persistence is off; in-memory cache
        # still works and we fail open to a fresh GGUF re-scan after each
        # restart. This is the production fail-safe — the engine MUST be
        # able to start even when no writable cache directory exists.
        log.warning(
            "profile_cache_disabled requested=%s — all fallback paths failed; "
            "profiles will be re-scanned on every restart",
            preferred,
        )
        return None

    def _cache_key(self, model_path: str) -> str:
        """Generate a stable cache key from model path + file identity.

        Uses file sizes and mtimes so the cache auto-invalidates when
        model files change.
        """
        model_path = str(model_path)

        paths = [model_path]
        shard_match = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", model_path)
        if shard_match:
            base = shard_match.group(1)
            total = int(shard_match.group(3))
            paths = []
            for i in range(1, total + 1):
                sp = f"{base}-{i:05d}-of-{total:05d}.gguf"
                if os.path.isfile(sp):
                    paths.append(sp)

        identity = ""
        for p in sorted(paths):
            try:
                st = os.stat(p)
                identity += f"{p}|{st.st_size}|{st.st_mtime:.0f}\n"
            except OSError:
                identity += f"{p}|missing\n"

        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    def _profile_path(self, cache_key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{cache_key}.json"

    def get(self, model_path: str) -> ModelProfile | None:
        """Load a cached profile, or None if not cached / stale."""
        key = self._cache_key(model_path)

        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key]

        path = self._profile_path(key)
        if path is None or not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_v = ModelProfile.cached_version(data)
            if cached_v < ModelProfile.profile_version:
                # Schema bumped (e.g. new field used by autofit). Drop the
                # stale file so the next caller re-scans the GGUF instead
                # of operating on incomplete/wrong derived values.
                log.info(
                    "Profile cache version stale, re-scanning: %s (cached=%d, current=%d)",
                    path.name, cached_v, ModelProfile.profile_version,
                )
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
            profile = ModelProfile.from_dict(data)
            with self._lock:
                self._memory[key] = profile
                while len(self._memory) > self._MAX_MEMORY_ENTRIES:
                    self._memory.popitem(last=False)
            log.info("Loaded cached profile: %s (key=%s)", profile.model_name, key)
            return profile
        except Exception as exc:
            log.warning("Failed to load cached profile %s: %s", path, exc)
            return None

    def save(self, profile: ModelProfile) -> None:
        """Save a profile to disk and memory cache.

        Always updates the in-memory layer. Disk write is best-effort —
        skipped silently when persistence is disabled, logged-but-ignored
        on transient I/O failures.
        """
        key = self._cache_key(profile.model_path)

        # In-memory layer first — works even when disk is disabled, and
        # is the bound that keeps us from re-scanning identical GGUFs
        # back-to-back within a single process lifetime.
        with self._lock:
            self._memory[key] = profile
            while len(self._memory) > self._MAX_MEMORY_ENTRIES:
                self._memory.popitem(last=False)

        path = self._profile_path(key)
        if path is None:
            return  # disk persistence disabled — memory-only

        try:
            path.write_text(
                json.dumps(profile.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Saved model profile: %s (key=%s)", profile.model_name, key)
        except Exception as exc:
            log.warning("Failed to save profile %s: %s", path, exc)

    def list_profiles(self) -> list[dict]:
        """List all cached profiles (summary info only)."""
        if self._cache_dir is None:
            return []
        profiles = []
        for path in sorted(self._cache_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append({
                    "model_name": data.get("model_name", ""),
                    "model_path": data.get("model_path", ""),
                    "architecture": data.get("architecture", ""),
                    "is_moe": data.get("is_moe", False),
                    "expert_count": data.get("expert_count", 0),
                    "total_size_gb": round(data.get("total_size_bytes", 0) / 1e9, 1),
                    "n_tensors": data.get("n_tensors", 0),
                    "created_at": data.get("created_at", 0),
                    "cache_key": path.stem,
                })
            except Exception as exc:
                log.warning("Failed to read profile %s: %s", path, exc)
                continue
        return profiles

    def delete(self, model_path: str) -> bool:
        """Delete a cached profile.

        Returns True if a disk artifact was removed. Memory is always
        cleared regardless of disk state.
        """
        key = self._cache_key(model_path)
        with self._lock:
            self._memory.pop(key, None)
        path = self._profile_path(key)
        if path is None or not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False
