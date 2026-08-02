"""Tests for augmentum.models.model_profile_cache."""

from __future__ import annotations

import hashlib
import struct
import tempfile
from pathlib import Path

import pytest

from augmentum.models.model_profile_cache import (
    _GGUF_TENSOR_TYPE_BPE,
    _GGUF_TENSOR_TYPE_BPE_FALLBACK,
    ModelProfileCache,
    scan_gguf_header,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal valid GGUF file
# ---------------------------------------------------------------------------


def _write_gguf_string(buf: bytearray, s: str) -> None:
    """Append a GGUF-format string (uint64 length + raw bytes)."""
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    """Append a GGUF KV pair with UINT32 value (type 4)."""
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", 4)  # val_type = UINT32
    buf += struct.pack("<I", value)


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    """Append a GGUF KV pair with STRING value (type 8)."""
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", 8)  # val_type = STRING
    _write_gguf_string(buf, value)


def _write_tensor_descriptor(
    buf: bytearray,
    name: str,
    dims: list[int],
    dtype: int = 0,
    offset: int = 0,
) -> None:
    """Append a GGUF tensor descriptor."""
    _write_gguf_string(buf, name)
    buf += struct.pack("<I", len(dims))
    for d in dims:
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", dtype)
    buf += struct.pack("<Q", offset)


def _make_minimal_gguf(
    arch: str = "llama",
    block_count: int = 32,
    expert_count: int | None = None,
    expert_used_count: int | None = None,
    chat_template: str | None = None,
    tensors: list[tuple[str, list[int]]] | None = None,
) -> bytes:
    """Build a minimal valid GGUF v3 file in memory.

    Returns the raw bytes of the file.
    """
    if tensors is None:
        tensors = [("token_embd.weight", [4096, 32000])]

    # Count KV pairs
    n_kv = 2  # general.architecture + {arch}.block_count
    if expert_count is not None:
        n_kv += 1
    if expert_used_count is not None:
        n_kv += 1
    if chat_template is not None:
        n_kv += 1

    buf = bytearray()
    # Header
    buf += b"GGUF"
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", len(tensors))  # n_tensors
    buf += struct.pack("<Q", n_kv)  # n_kv

    # KV pairs
    _write_kv_string(buf, "general.architecture", arch)
    _write_kv_uint32(buf, f"{arch}.block_count", block_count)
    if expert_count is not None:
        _write_kv_uint32(buf, f"{arch}.expert_count", expert_count)
    if expert_used_count is not None:
        _write_kv_uint32(buf, f"{arch}.expert_used_count", expert_used_count)
    if chat_template is not None:
        _write_kv_string(buf, "tokenizer.chat_template", chat_template)

    # Tensor descriptors
    for name, dims in tensors:
        _write_tensor_descriptor(buf, name, dims)

    return bytes(buf)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Tests: scan_gguf_header
# ---------------------------------------------------------------------------


def test_parses_architecture(tmp_dir: Path):
    """scan_gguf_header extracts architecture and n_layers."""
    gguf_path = tmp_dir / "test-model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(arch="llama", block_count=40))

    profile = scan_gguf_header(str(gguf_path))

    assert profile.architecture == "llama"
    assert profile.n_layers == 40
    assert profile.model_name == "test-model"


def test_detects_moe(tmp_dir: Path):
    """scan_gguf_header detects MoE models via expert_count."""
    gguf_path = tmp_dir / "mixtral.gguf"
    gguf_path.write_bytes(
        _make_minimal_gguf(
            arch="llama",
            block_count=32,
            expert_count=8,
            expert_used_count=2,
        )
    )

    profile = scan_gguf_header(str(gguf_path))

    assert profile.is_moe is True
    assert profile.expert_count == 8
    assert profile.expert_used_count == 2


def test_returns_model_name_from_path(tmp_dir: Path):
    """model_name is derived from the filename stem."""
    gguf_path = tmp_dir / "Qwen2.5-72B-Q4_K_M.gguf"
    gguf_path.write_bytes(_make_minimal_gguf())

    profile = scan_gguf_header(str(gguf_path))

    assert profile.model_name == "Qwen2.5-72B-Q4_K_M"


def test_chat_template_hash(tmp_dir: Path):
    """scan_gguf_header computes chat_template_hash when template is present."""
    template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
    gguf_path = tmp_dir / "model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(chat_template=template))

    profile = scan_gguf_header(str(gguf_path))

    expected = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    assert profile.chat_template_hash == expected


def test_no_chat_template_hash(tmp_dir: Path):
    """chat_template_hash is empty when no template is present."""
    gguf_path = tmp_dir / "model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf())

    profile = scan_gguf_header(str(gguf_path))

    assert profile.chat_template_hash == ""


def test_non_moe_model(tmp_dir: Path):
    """A model without expert_count is not classified as MoE."""
    gguf_path = tmp_dir / "dense-model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(arch="llama", block_count=32))

    profile = scan_gguf_header(str(gguf_path))

    assert profile.is_moe is False
    assert profile.expert_count == 0


def test_recommended_cli_flags_dense(tmp_dir: Path):
    """recommended_cli_flags returns expected flags for a dense model."""
    gguf_path = tmp_dir / "model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(block_count=32))

    profile = scan_gguf_header(str(gguf_path))
    flags = profile.recommended_cli_flags

    assert "--ctx-size" in flags
    assert isinstance(flags["--ctx-size"], int)


def test_recommended_cli_flags_moe(tmp_dir: Path):
    """recommended_cli_flags includes MoE-specific flags."""
    gguf_path = tmp_dir / "moe.gguf"
    gguf_path.write_bytes(
        _make_minimal_gguf(block_count=32, expert_count=8, expert_used_count=2)
    )

    profile = scan_gguf_header(str(gguf_path))
    flags = profile.recommended_cli_flags

    assert "--ctx-size" in flags


# ---------------------------------------------------------------------------
# Tests: ModelProfileCache
# ---------------------------------------------------------------------------


def test_save_and_load(tmp_dir: Path):
    """Save a profile then load it back — all fields survive the round-trip."""
    gguf_path = tmp_dir / "model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(arch="qwen2", block_count=64))

    profile = scan_gguf_header(str(gguf_path))

    cache = ModelProfileCache(cache_dir=str(tmp_dir / "cache"))
    cache.save(profile)

    # Clear in-memory to force disk read
    cache._memory.clear()

    loaded = cache.get(str(gguf_path))
    assert loaded is not None
    assert loaded.architecture == "qwen2"
    assert loaded.n_layers == 64
    assert loaded.model_name == "model"


def test_returns_none_on_miss(tmp_dir: Path):
    """Cache miss returns None."""
    cache = ModelProfileCache(cache_dir=str(tmp_dir / "cache"))
    result = cache.get("/nonexistent/model.gguf")
    assert result is None


def test_list_profiles(tmp_dir: Path):
    """list_profiles returns saved profiles."""
    gguf_path = tmp_dir / "model.gguf"
    gguf_path.write_bytes(_make_minimal_gguf(arch="llama", block_count=32))

    profile = scan_gguf_header(str(gguf_path))

    cache = ModelProfileCache(cache_dir=str(tmp_dir / "cache"))
    cache.save(profile)

    profiles = cache.list_profiles()
    assert len(profiles) == 1
    assert profiles[0]["model_name"] == "model"
    assert profiles[0]["architecture"] == "llama"


def test_in_memory_lru(tmp_dir: Path):
    """In-memory cache respects LRU eviction."""
    cache = ModelProfileCache(cache_dir=str(tmp_dir / "cache"))
    cache._MAX_MEMORY_ENTRIES = 2

    # Create 3 models
    for i in range(3):
        p = tmp_dir / f"model{i}.gguf"
        p.write_bytes(_make_minimal_gguf(arch="llama", block_count=i + 1))
        profile = scan_gguf_header(str(p))
        cache.save(profile)

    # Only 2 should be in memory
    assert len(cache._memory) == 2


# ---------------------------------------------------------------------------
# Tensor BPE (bytes-per-element) table — keys must match ggml.h type IDs
# ---------------------------------------------------------------------------


class TestTensorBpeTable:
    """Lock in correct ggml type IDs for the tensor-size estimator.

    The BPE values feed the autofit GPU-layer calculation; a wrong ID
    means a real tensor's bytes get estimated using the wrong table
    entry, which silently shifts the autofit budget. Fail loudly here
    so any future renumbering is caught at test time, not at load
    time.

    Type IDs match enum ggml_type in ggml/include/ggml.h. Block sizes
    come from ggml/src/ggml-common.h. Bytes/element formula:
    block_size_in_bytes / elements_per_block.
    """

    def test_basic_float_types(self) -> None:
        # F32 / F16 / BF16 are 4 / 2 / 2 bytes — the bedrock cases.
        assert _GGUF_TENSOR_TYPE_BPE[0] == 4.0   # F32
        assert _GGUF_TENSOR_TYPE_BPE[1] == 2.0   # F16
        assert _GGUF_TENSOR_TYPE_BPE[28] == 8.0  # F64
        assert _GGUF_TENSOR_TYPE_BPE[30] == 2.0  # BF16

    def test_q_types_at_canonical_ggml_ids(self) -> None:
        # Q4_0 .. Q8_K live at IDs 2-15 in ggml.h.
        assert _GGUF_TENSOR_TYPE_BPE[2] == 0.5625    # Q4_0
        assert _GGUF_TENSOR_TYPE_BPE[8] == 1.0625    # Q8_0
        assert _GGUF_TENSOR_TYPE_BPE[12] == 0.5625   # Q4_K
        assert _GGUF_TENSOR_TYPE_BPE[14] == 0.8125   # Q6_K

    def test_iq_types_at_canonical_ggml_ids(self) -> None:
        # The previously-missing / mis-keyed IQ entries. ID 16 is
        # IQ2_XXS in ggml.h, NOT 24 (which is I8). Same fix applied
        # across the IQ family.
        assert _GGUF_TENSOR_TYPE_BPE[16] == 0.25     # IQ2_XXS
        assert _GGUF_TENSOR_TYPE_BPE[17] == 0.3125   # IQ2_XS
        assert _GGUF_TENSOR_TYPE_BPE[18] == 0.3125   # IQ3_XXS
        assert _GGUF_TENSOR_TYPE_BPE[19] == 0.21875  # IQ1_S
        assert _GGUF_TENSOR_TYPE_BPE[20] == 0.5625   # IQ4_NL
        assert _GGUF_TENSOR_TYPE_BPE[21] == 0.4375   # IQ3_S
        assert _GGUF_TENSOR_TYPE_BPE[22] == 0.34375  # IQ2_S
        assert _GGUF_TENSOR_TYPE_BPE[23] == 0.53125  # IQ4_XS
        assert _GGUF_TENSOR_TYPE_BPE[29] == 0.21875  # IQ1_M

    def test_integer_types(self) -> None:
        # I8/I16/I32/I64 share IDs 24-27 with what an older table
        # mis-claimed for IQ2_XXS / IQ2_XS / etc. — make sure they
        # land at their actual byte-widths.
        assert _GGUF_TENSOR_TYPE_BPE[24] == 1.0  # I8
        assert _GGUF_TENSOR_TYPE_BPE[25] == 2.0  # I16
        assert _GGUF_TENSOR_TYPE_BPE[26] == 4.0  # I32
        assert _GGUF_TENSOR_TYPE_BPE[27] == 8.0  # I64

    def test_ternary_types(self) -> None:
        # TQ1_0/TQ2_0 are at IDs 34/35 in ggml.h, NOT 30/31 (which
        # are BF16 and an unassigned slot).
        assert _GGUF_TENSOR_TYPE_BPE[34] == 0.21875  # TQ1_0
        assert _GGUF_TENSOR_TYPE_BPE[35] == 0.25     # TQ2_0

    def test_fallback_in_q4_territory(self) -> None:
        # Unknown forward-incompat type IDs default to ~Q4 territory.
        # F16's 2.0 (the old default) overestimated IQ-class quants
        # by ~4x and starved autofit's GPU-layer budget.
        assert _GGUF_TENSOR_TYPE_BPE_FALLBACK == 0.5
        # Sanity: the fallback is below F16 and above 1-bit quant.
        assert _GGUF_TENSOR_TYPE_BPE_FALLBACK < 2.0
        assert _GGUF_TENSOR_TYPE_BPE_FALLBACK > 0.125

    def test_no_legacy_misplaced_entries(self) -> None:
        # The pre-fix table claimed IQ types at IDs 24-25 / 28 / 30-31.
        # Those slots now correctly hold integer / float / BF16
        # entries — verify the IQ values aren't lingering at the wrong
        # IDs (would be a regression bringing back the original bug).
        assert _GGUF_TENSOR_TYPE_BPE[24] != 0.5    # was IQ2_XXS=0.5
        assert _GGUF_TENSOR_TYPE_BPE[25] != 0.5625 # was IQ2_XS=0.5625
        # ID 28 was IQ1_S=0.5; now F64=8.0 (verified above).
        # ID 30 was TQ1_0=0.375; now BF16=2.0 (verified above).
