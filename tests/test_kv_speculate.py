"""Speculative turn generation (KV resume ladder rung 3) — unit tests.

Covers the four hard rules from ``augmentum/models/kv_speculate.py``:
local-only gating, real-traffic preemption, the truncation guard
(never serve a capped answer), and drafts-never-touch-disk (the
speculative request is excluded from replay capture). Style mirrors
``tests/test_kv_resume_ladder.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
    Usage,
)
from augmentum.models.kv_session_manifest import KVSessionManifest
from augmentum.models.kv_speculate import (
    SpecEntry,
    TurnSpeculator,
    compute_fingerprint,
    sampling_snapshot,
    serialize_text_messages,
)
from augmentum.models.llama_cpp import LlamaCppBackend

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_request(**kwargs) -> InternalChatRequest:
    base = {
        "model": "test-model.gguf",
        "messages": [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
        ],
        "stream": True,
    }
    base.update(kwargs)
    return InternalChatRequest(**base)


def _ready_manager() -> MagicMock:
    from augmentum.models.llama_server_manager import ProcessState

    mgr = MagicMock()
    mgr.state = ProcessState.READY
    mgr.is_busy = False
    mgr.model_id = "test-model"
    mgr._replay_warmed_keys = set()
    mgr.kv_ttl_days_for_mode = MagicMock(return_value=2)
    return mgr


def _spec_backend(manifest: KVSessionManifest | None = None) -> MagicMock:
    """Mock backend exposing exactly the surface the speculator touches."""
    backend = MagicMock()
    backend._manager = _ready_manager()
    backend._kv_manifest = MagicMock(return_value=manifest)
    backend._multislot_enabled = MagicMock(return_value=True)
    backend._get_slot_lock = MagicMock(return_value=asyncio.Lock())
    backend._get_session_for_slot = MagicMock(return_value="")
    backend._claim_slot = MagicMock()
    backend._release_slot = MagicMock()
    backend.save_session_state = AsyncMock(return_value=True)
    backend.prewarm_context = AsyncMock(return_value=1)
    return backend


@contextlib.contextmanager
def _setting(name: str, value):
    from augmentum.config import settings
    prev = getattr(settings, name)
    setattr(settings, name, value)
    try:
        yield
    finally:
        setattr(settings, name, prev)


_PREFIX = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hello"},
]


def _seed_replay(
    manifest: KVSessionManifest, key: str, *,
    sampling: dict | None = None, mode: str = "",
) -> None:
    manifest.record_replay_source(
        session_key=key,
        mode=mode,
        messages_json=json.dumps(_PREFIX),
        fingerprint="fp_" + key,
        message_count=len(_PREFIX),
        ttl_days=2,
        sampling_json=json.dumps(sampling, sort_keys=True) if sampling else "",
    )


def _predicted(draft: str, prior_assistant: str = "hi") -> list[dict]:
    return [
        *_PREFIX,
        {"role": "assistant", "content": prior_assistant},
        {"role": "user", "content": draft},
    ]


# ---------------------------------------------------------------------------
# fingerprint primitives
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_sampling_snapshot_drops_defaults(self):
        req = _make_request(temperature=0.7, stop=[], think=False)
        snap = sampling_snapshot(req)
        assert snap == {"temperature": 0.7}

    def test_sampling_snapshot_keeps_meaningful_falsy(self):
        req = _make_request(temperature=0.0, seed=0, think=True)
        snap = sampling_snapshot(req)
        assert snap == {"temperature": 0.0, "seed": 0, "think": True}

    def test_fingerprint_deterministic_and_sensitive(self):
        msgs = _predicted("draft")
        fp1 = compute_fingerprint("m", msgs, {"temperature": 0.7})
        fp2 = compute_fingerprint("m", msgs, {"temperature": 0.7})
        assert fp1 == fp2
        assert fp1 != compute_fingerprint("m", _predicted("other"), {"temperature": 0.7})
        assert fp1 != compute_fingerprint("m", msgs, {"temperature": 0.8})
        assert fp1 != compute_fingerprint("m2", msgs, {"temperature": 0.7})

    def test_serialize_rejects_tool_and_image_messages(self):
        assert serialize_text_messages(
            [Message(role="user", content="x", tool_calls=[{"id": "1"}])]
        ) is None
        assert serialize_text_messages(
            [Message(role="user", content="x", images=["b64"])]
        ) is None
        assert serialize_text_messages(
            [Message(role="tool", content="x")]
        ) is None
        assert serialize_text_messages(
            [Message(role="user", content="x")]
        ) == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# speculation gates
# ---------------------------------------------------------------------------


class TestSpeculateGates:
    @pytest.mark.asyncio
    async def test_disabled(self):
        spec = TurnSpeculator(_spec_backend())
        with _setting("engine_speculation_enabled", False):
            out = await spec.speculate("kv_a", draft="d")
        assert out["status"] == "skip" and out["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_busy_skips(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a")
        backend = _spec_backend(manifest)
        backend._manager.is_busy = True
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="d")
        assert out["reason"] == "busy"

    @pytest.mark.asyncio
    async def test_no_replay_source(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        spec = TurnSpeculator(_spec_backend(manifest))
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="d")
        assert out["reason"] == "no_replay_source"

    @pytest.mark.asyncio
    async def test_no_manager_is_local_only_gate(self):
        backend = _spec_backend()
        backend._manager = None
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="d")
        assert out["reason"] == "engine_not_ready"


# ---------------------------------------------------------------------------
# level 1 — draft-aware prefill
# ---------------------------------------------------------------------------


class TestPrefill:
    @pytest.mark.asyncio
    async def test_no_sampling_row_prefills_instead_of_generating(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling=None)  # pre-column row
        backend = _spec_backend(manifest)
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="draft", prior_assistant="hi")
        assert out["status"] == "prefix"
        backend.prewarm_context.assert_awaited_once()
        payload = backend.prewarm_context.await_args.args[0]
        assert payload == _predicted("draft")
        backend._claim_slot.assert_called_once_with(1, "kv_a")
        assert "kv_a" in backend._manager._replay_warmed_keys
        assert spec._entries == {}

    @pytest.mark.asyncio
    async def test_prefill_only_setting_forces_level1(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling={"temperature": 0.7})
        backend = _spec_backend(manifest)
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True), \
                _setting("engine_speculation_prefill_only", True):
            out = await spec.speculate("kv_a", draft="draft")
        assert out["status"] == "prefix"
        backend.prewarm_context.assert_awaited_once()


# ---------------------------------------------------------------------------
# level 2 — generation + serve
# ---------------------------------------------------------------------------


def _wire_stream(backend, chunks, captured: dict):
    async def fake_stream(req):
        captured["req"] = req
        for c in chunks:
            yield c
    backend.chat_stream = fake_stream


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generates_and_stores_entry(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling={"temperature": 0.7}, mode="direct")
        backend = _spec_backend(manifest)
        captured: dict = {}
        _wire_stream(backend, [
            InternalStreamChunk(content_delta="Hello ", model="m"),
            InternalStreamChunk(content_delta="world", thinking_delta="hmm", model="m"),
            InternalStreamChunk(
                finish_reason="stop", usage=Usage(completion_tokens=2), model="m",
            ),
        ], captured)
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate(
                "kv_a", draft="draft", prior_assistant="hi",
            )
        assert out["status"] == "ready"
        req = captured["req"]
        assert getattr(req, "_augmentum_speculative", False) is True
        assert req.kv_session_key == "kv_a"
        assert req.kv_mode == "direct"
        assert req.temperature == 0.7
        assert req.max_tokens > 0  # injected cap
        entry = spec._entries["kv_a"]
        assert entry.deltas == [("Hello ", ""), ("world", "hmm")]
        assert entry.finish_reason == "stop"
        assert entry.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_capped_generation_never_servable(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling={"temperature": 0.7})
        backend = _spec_backend(manifest)
        _wire_stream(backend, [
            InternalStreamChunk(content_delta="truncated answ", model="m"),
            InternalStreamChunk(finish_reason="length", model="m"),
        ], {})
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="draft")
        assert out["status"] == "prefix"
        assert "not_servable" in out["reason"]
        assert spec._entries == {}

    @pytest.mark.asyncio
    async def test_session_own_cap_is_servable(self, tmp_path):
        # max_tokens captured FROM the session: the real request would
        # truncate identically, so a length stop is honest output.
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling={"max_tokens": 4})
        backend = _spec_backend(manifest)
        _wire_stream(backend, [
            InternalStreamChunk(content_delta="four token answer", model="m"),
            InternalStreamChunk(finish_reason="length", model="m"),
        ], {})
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            out = await spec.speculate("kv_a", draft="draft")
        assert out["status"] == "ready"
        assert spec._entries["kv_a"].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_same_fingerprint_cached(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        _seed_replay(manifest, "kv_a", sampling={"temperature": 0.7})
        backend = _spec_backend(manifest)
        _wire_stream(backend, [
            InternalStreamChunk(content_delta="x", model="m"),
            InternalStreamChunk(finish_reason="stop", model="m"),
        ], {})
        spec = TurnSpeculator(backend)
        with _setting("engine_speculation_enabled", True):
            first = await spec.speculate("kv_a", draft="draft")
            second = await spec.speculate("kv_a", draft="draft")
        assert first["status"] == "ready"
        assert second["status"] == "cached"


class TestServe:
    def _entry_for(self, spec: TurnSpeculator, draft: str = "draft") -> SpecEntry:
        msgs = _predicted(draft)
        fp = compute_fingerprint("test-model", msgs, {"temperature": 0.7})
        entry = SpecEntry(
            session_key="kv_a", model_id="test-model", fingerprint=fp,
            deltas=[("Hello ", ""), ("world", "")], finish_reason="stop",
            usage=Usage(completion_tokens=2), created=time.monotonic(),
            completion_chars=11,
        )
        spec._store(entry)
        return entry

    def _real_request(self, draft: str = "draft") -> InternalChatRequest:
        return InternalChatRequest(
            model="whatever-alias",
            messages=[
                Message(role=m["role"], content=m["content"])
                for m in _predicted(draft)
            ],
            stream=True,
            temperature=0.7,
            kv_session_key="kv_a",
        )

    @pytest.mark.asyncio
    async def test_hit_pops_entry(self):
        spec = TurnSpeculator(_spec_backend())
        entry = self._entry_for(spec)
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(self._real_request())
        assert got is entry
        assert spec._entries == {}

    @pytest.mark.asyncio
    async def test_draft_mismatch_misses(self):
        spec = TurnSpeculator(_spec_backend())
        self._entry_for(spec, draft="draft")
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(self._real_request(draft="edited"))
        assert got is None
        assert spec._entries == {}  # consumed either way

    @pytest.mark.asyncio
    async def test_stale_entry_misses(self):
        spec = TurnSpeculator(_spec_backend())
        entry = self._entry_for(spec)
        entry.created = time.monotonic() - 99_999
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(self._real_request())
        assert got is None

    @pytest.mark.asyncio
    async def test_model_swap_misses(self):
        backend = _spec_backend()
        spec = TurnSpeculator(backend)
        self._entry_for(spec)
        backend._manager.model_id = "different-model"
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(self._real_request())
        assert got is None

    @pytest.mark.asyncio
    async def test_tool_request_misses(self):
        spec = TurnSpeculator(_spec_backend())
        self._entry_for(spec)
        req = self._real_request()
        req.tools = [{"type": "function", "function": {"name": "t"}}]
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(req)
        assert got is None

    @pytest.mark.asyncio
    async def test_real_request_preempts_inflight(self):
        spec = TurnSpeculator(_spec_backend())
        started = asyncio.Event()

        async def long_speculation():
            started.set()
            await asyncio.sleep(30)

        task = asyncio.create_task(long_speculation())
        spec._task = task
        spec._task_key = "kv_a"
        await started.wait()
        with _setting("engine_speculation_enabled", True):
            got = await spec.on_real_request(self._real_request())
        assert got is None
        assert task.cancelled()
        assert spec._task is None

    @pytest.mark.asyncio
    async def test_replay_chunks_fidelity(self):
        spec = TurnSpeculator(_spec_backend())
        entry = self._entry_for(spec)
        req = self._real_request()
        chunks = [c async for c in spec.replay_chunks(entry, req)]
        assert [(c.content_delta, c.thinking_delta) for c in chunks[:-1]] == [
            ("Hello ", ""), ("world", ""),
        ]
        assert chunks[0].role == "assistant"
        assert getattr(chunks[0], "augmentum", None) == {"speculative": True}
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.usage is entry.usage


# ---------------------------------------------------------------------------
# capture integration (rule 4 — drafts never touch disk)
# ---------------------------------------------------------------------------


class TestCaptureIntegration:
    def test_speculative_request_skips_replay_capture(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        req = _make_request(kv_session_key="kv_a")
        req._augmentum_speculative = True
        # Early-returns before touching any instance attribute — a bare
        # backend would raise on anything past the marker check.
        assert backend._schedule_replay_capture(req) is None

    @pytest.mark.asyncio
    async def test_capture_records_sampling_snapshot(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "m.db"))
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._kv_manifest = MagicMock(return_value=manifest)
        backend._manager = _ready_manager()
        req = _make_request(
            kv_session_key="kv_a", kv_mode="direct",
            temperature=0.6, top_p=0.9, think=False,
        )
        await backend._record_replay_source("kv_a", req)
        row = manifest.get_replay_source("kv_a")
        assert row is not None
        sampling = json.loads(row["sampling_json"])
        assert sampling == {"temperature": 0.6, "top_p": 0.9}


# ---------------------------------------------------------------------------
# manifest schema upgrade
# ---------------------------------------------------------------------------


class TestManifestUpgrade:
    def test_alter_adds_sampling_column_to_old_db(self, tmp_path):
        path = str(tmp_path / "old.db")
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE kv_replay_sources (
                    session_key TEXT PRIMARY KEY,
                    mode TEXT NOT NULL DEFAULT '',
                    messages_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    approx_chars INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0,
                    expires_at REAL NOT NULL DEFAULT 0
                )
                """
            )
        manifest = KVSessionManifest(path)
        _seed_replay(manifest, "kv_a", sampling={"temperature": 0.5})
        row = manifest.get_replay_source("kv_a")
        assert row is not None
        assert json.loads(row["sampling_json"]) == {"temperature": 0.5}
