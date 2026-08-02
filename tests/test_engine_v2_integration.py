"""Integration tests for Engine v2: manager → discovery → profile caching → token cache → resolve."""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

from augmentum.models.llama_server_manager import LlamaServerManager, ProcessState
from augmentum.models.model_profile_cache import ModelProfileCache, scan_gguf_header
from augmentum.models.token_count_cache import TokenCountCache

# ---------------------------------------------------------------------------
# Helper: write a minimal valid GGUF v3 file
# ---------------------------------------------------------------------------

def _make_minimal_gguf(
    path: str | Path,
    arch: str = "llama",
    n_layers: int = 32,
    expert_count: int = 0,
) -> None:
    """Write a minimal valid GGUF v3 file to *path*.

    Encodes:
      - Magic / version / n_tensors / n_kv header
      - general.architecture (string, type 8)
      - {arch}.block_count (uint32, type 4)
      - {arch}.expert_count (uint32, type 4) — only when expert_count > 0
    """

    def _pack_string(s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def _kv_string(key: str, value: str) -> bytes:
        return _pack_string(key) + struct.pack("<I", 8) + _pack_string(value)

    def _kv_uint32(key: str, value: int) -> bytes:
        return _pack_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)

    kv_pairs: list[bytes] = [
        _kv_string("general.architecture", arch),
        _kv_uint32(f"{arch}.block_count", n_layers),
    ]
    if expert_count > 0:
        kv_pairs.append(_kv_uint32(f"{arch}.expert_count", expert_count))

    n_kv = len(kv_pairs)
    kv_data = b"".join(kv_pairs)

    with open(path, "wb") as f:
        f.write(b"GGUF")                        # magic
        f.write(struct.pack("<I", 3))            # version 3
        f.write(struct.pack("<Q", 0))            # n_tensors = 0
        f.write(struct.pack("<Q", n_kv))         # n_kv
        f.write(kv_data)                         # KV metadata


# ---------------------------------------------------------------------------
# Test 1: Full pipeline — discover → scan → enrich → resolve
# ---------------------------------------------------------------------------

async def test_discover_scan_resolve():
    """Full pipeline: manager → discovery → profile caching → resolve."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create fake GGUF files
        _make_minimal_gguf(os.path.join(tmp, "llama-7b-q4.gguf"), arch="llama", n_layers=32)
        _make_minimal_gguf(os.path.join(tmp, "mistral-7b-q4.gguf"), arch="mistral", n_layers=24)
        # Subdirectory (2 levels deep is supported)
        subdir = os.path.join(tmp, "moe_models")
        os.makedirs(subdir)
        _make_minimal_gguf(
            os.path.join(subdir, "mixtral-8x7b.gguf"),
            arch="llama",
            n_layers=32,
            expert_count=8,
        )

        profile_cache_dir = os.path.join(tmp, ".profiles")
        manager = LlamaServerManager(
            model_dir=tmp,
            profile_cache_dir=profile_cache_dir,
        )

        # --- discover_gguf_files ---
        files = manager.discover_gguf_files()
        assert len(files) == 3, f"Expected 3 files, got {len(files)}: {[f['filename'] for f in files]}"

        # --- scan_and_cache_profiles ---
        new_count = await manager.scan_and_cache_profiles()
        assert new_count == 3, f"Expected 3 new profiles, got {new_count}"

        # A second scan should yield 0 new (all cached)
        second_count = await manager.scan_and_cache_profiles()
        assert second_count == 0, f"Expected 0 new on second scan, got {second_count}"

        # --- discover_models enriches with profile data ---
        models = await manager.discover_models()
        assert len(models) == 3

        enriched = [m for m in models if m.get("has_profile")]
        assert len(enriched) == 3, "All models should have profiles after scan"

        # Verify architecture was captured
        archs = {m["filename"]: m.get("architecture") for m in models}
        assert archs["llama-7b-q4.gguf"] == "llama"
        assert archs["mistral-7b-q4.gguf"] == "mistral"
        assert archs["mixtral-8x7b.gguf"] == "llama"

        # Verify MoE model is flagged
        moe_model = next(m for m in models if m["filename"] == "mixtral-8x7b.gguf")
        assert moe_model["is_moe"] is True

        non_moe = next(m for m in models if m["filename"] == "llama-7b-q4.gguf")
        assert non_moe["is_moe"] is False

        # --- _resolve_model_path: exact match ---
        resolved = manager._resolve_model_path("llama-7b-q4.gguf")
        assert resolved is not None
        assert resolved.endswith("llama-7b-q4.gguf")

        # --- _resolve_model_path: stem without extension ---
        resolved_no_ext = manager._resolve_model_path("mistral-7b-q4")
        assert resolved_no_ext is not None
        assert resolved_no_ext.endswith("mistral-7b-q4.gguf")

        # --- _resolve_model_path: fuzzy stem match (case-insensitive) ---
        resolved_fuzzy = manager._resolve_model_path("LLAMA-7B-Q4")
        assert resolved_fuzzy is not None
        assert "llama-7b-q4" in resolved_fuzzy.lower()

        # --- _resolve_model_path: miss → None ---
        resolved_miss = manager._resolve_model_path("nonexistent-model.gguf")
        assert resolved_miss is None


# ---------------------------------------------------------------------------
# Test 2: Token cache standalone
# ---------------------------------------------------------------------------

async def test_token_cache_standalone():
    """Token cache stores and retrieves counts independently."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "token_counts.db")
        cache = TokenCountCache(db_path=db_path)
        await cache.init_db()

        # Miss before any store
        result = await cache.get_count("model-a", "hello world")
        assert result is None

        # Store and retrieve
        await cache.store_count("model-a", "hello world", 3)
        result = await cache.get_count("model-a", "hello world")
        assert result == 3

        # Same text, different model → different entry
        await cache.store_count("model-b", "hello world", 5)
        result_b = await cache.get_count("model-b", "hello world")
        assert result_b == 5
        # model-a unchanged
        result_a = await cache.get_count("model-a", "hello world")
        assert result_a == 3

        # Store a few more entries
        await cache.store_count("model-a", "longer text here", 4)
        await cache.store_count("model-b", "completely different text", 8)

        # Stats verify totals
        stats = await cache.stats()
        assert stats["total_entries"] == 4
        assert stats["distinct_models"] == 2

        # Purge model-a
        deleted = await cache.purge_model("model-a")
        assert deleted == 2  # 2 entries for model-a

        stats_after = await cache.stats()
        assert stats_after["total_entries"] == 2
        assert stats_after["distinct_models"] == 1

        # model-a entries are gone
        result_purged = await cache.get_count("model-a", "hello world")
        assert result_purged is None

        # model-b entries survive
        result_b_after = await cache.get_count("model-b", "hello world")
        assert result_b_after == 5


# ---------------------------------------------------------------------------
# Test 3: Manager state machine
# ---------------------------------------------------------------------------

async def test_manager_state_machine():
    """State transitions: initial IDLE, status dict correct, stop from IDLE is safe."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = LlamaServerManager(
            model_dir=tmp,
            profile_cache_dir=os.path.join(tmp, ".profiles"),
        )

        # Initial state is IDLE
        assert manager.state == ProcessState.IDLE

        # status() returns correct dict
        s = manager.status()
        assert s["state"] == "idle"
        assert s["model_id"] == ""
        assert s["model_path"] == ""
        assert "backend_url" in s
        assert s["uptime_s"] is None

        # stop() from IDLE is safe (no process, no exception)
        await manager.stop()
        assert manager.state == ProcessState.IDLE

        # Calling stop() multiple times is safe
        await manager.stop()
        assert manager.state == ProcessState.IDLE

        # status() still returns sane values
        s2 = manager.status()
        assert s2["state"] == "idle"
        assert s2["uptime_s"] is None


# ---------------------------------------------------------------------------
# Test 4: scan_gguf_header directly
# ---------------------------------------------------------------------------

async def test_scan_gguf_header_direct():
    """scan_gguf_header returns correct profile data from minimal GGUF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test-model.gguf")
        _make_minimal_gguf(path, arch="llama", n_layers=32, expert_count=0)

        profile = scan_gguf_header(path)

        assert profile.architecture == "llama"
        assert profile.n_layers == 32
        assert profile.is_moe is False
        assert profile.model_name == "test-model"
        assert profile.model_path == path
        assert len(profile.shards) == 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "moe-model.gguf")
        _make_minimal_gguf(path, arch="llama", n_layers=32, expert_count=8)

        profile = scan_gguf_header(path)

        assert profile.architecture == "llama"
        assert profile.n_layers == 32
        assert profile.expert_count == 8
        assert profile.is_moe is True


# ---------------------------------------------------------------------------
# Test 5: ProfileCache round-trip
# ---------------------------------------------------------------------------

async def test_profile_cache_round_trip():
    """Save and retrieve a profile; memory and disk layers both work."""
    with tempfile.TemporaryDirectory() as tmp:
        gguf_path = os.path.join(tmp, "round-trip.gguf")
        cache_dir = os.path.join(tmp, ".profiles")
        _make_minimal_gguf(gguf_path, arch="mistral", n_layers=24)

        cache = ModelProfileCache(cache_dir=cache_dir)

        # Nothing cached yet
        assert cache.get(gguf_path) is None

        # Scan and save
        profile = scan_gguf_header(gguf_path)
        cache.save(profile)

        # Retrieve from memory
        cached = cache.get(gguf_path)
        assert cached is not None
        assert cached.architecture == "mistral"
        assert cached.n_layers == 24

        # Create a fresh cache (empty memory), load from disk
        fresh_cache = ModelProfileCache(cache_dir=cache_dir)
        disk_profile = fresh_cache.get(gguf_path)
        assert disk_profile is not None
        assert disk_profile.architecture == "mistral"
        assert disk_profile.n_layers == 24

        # list_profiles returns our entry
        listed = fresh_cache.list_profiles()
        assert len(listed) == 1
        assert listed[0]["architecture"] == "mistral"

        # delete removes it
        deleted = fresh_cache.delete(gguf_path)
        assert deleted is True
        assert fresh_cache.get(gguf_path) is None
