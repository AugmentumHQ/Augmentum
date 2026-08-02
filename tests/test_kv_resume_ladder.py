"""KV resume ladder tests (restore→replay→cold).

Covers the three layers shipped for the ladder:

  - ``kv_replay_sources`` store in ``KVSessionManifest`` — round-trip,
    MRU ordering, TTL + cap pruning (never a silent cap: counts return).
  - Replay-source capture in ``LlamaCppBackend`` — prefers the declared
    stable prefix, refuses non-replayable content (images / tool calls /
    structured content) instead of storing something replay would warm
    wrong, skips oversize instead of truncating.
  - ``KVResumeLadder`` rung selection — hot short-circuit, single-slot
    restore with displacement save, universal replay (the only
    cross-restart recovery under --kv-unified), cold floor with a
    reason, busy preemption, in-flight dedup, boot-warm budgets.

Spec: docs/superpowers/specs/2026-06-26-kv-resume-ladder-low-latency-design.md
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.kv_resume import KVResumeLadder
from augmentum.models.kv_session_manifest import KVSessionManifest
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
    mgr._slot_dir = ""
    mgr._slot_save_supported = False
    mgr._warm_session_key = ""
    mgr._replay_warmed_keys = set()
    mgr._session_manifest = None
    mgr.model_id = "test-model"
    mgr.current_ctx_size = 8192
    mgr.kv_ttl_days_for_mode = MagicMock(return_value=2)
    mgr.session_is_pinned = MagicMock(return_value=False)
    return mgr


def _ladder_backend(manifest: KVSessionManifest | None = None) -> MagicMock:
    """A mock backend exposing exactly the surface the ladder touches."""
    backend = MagicMock()
    backend._manager = _ready_manager()
    backend._get_slot_for_session = MagicMock(return_value=None)
    backend._get_session_for_slot = MagicMock(return_value="")
    backend._multislot_enabled = MagicMock(return_value=True)
    backend._slot_state_exists = MagicMock(return_value=False)
    backend._get_slot_lock = MagicMock(return_value=asyncio.Lock())
    backend._claim_slot = MagicMock()
    backend._release_slot = MagicMock()
    backend._kv_manifest = MagicMock(return_value=manifest)
    backend._current_model_key = MagicMock(return_value="test-model")
    backend.save_session_state = AsyncMock(return_value=True)
    backend.restore_session_state = AsyncMock(return_value=True)
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


def _seed_replay(manifest: KVSessionManifest, key: str, *, mode: str = "",
                 ttl_days: float = 2) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"hello from {key}"},
    ]
    manifest.record_replay_source(
        session_key=key,
        mode=mode,
        messages_json=json.dumps(messages),
        fingerprint="fp_" + key,
        message_count=len(messages),
        ttl_days=ttl_days,
    )


# ---------------------------------------------------------------------------
# replay-source store
# ---------------------------------------------------------------------------


class TestReplaySourceStore:
    def test_round_trip(self, tmp_path):
        m = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(m, "kv_abc", mode="narrative")
        row = m.get_replay_source("kv_abc")
        assert row is not None
        assert row["mode"] == "narrative"
        assert row["message_count"] == 2
        payload = json.loads(row["messages_json"])
        assert payload[1]["content"] == "hello from kv_abc"

    def test_upsert_replaces(self, tmp_path):
        m = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(m, "kv_abc")
        m.record_replay_source(
            session_key="kv_abc",
            mode="chat",
            messages_json="[]",
            fingerprint="fp2",
            message_count=0,
            ttl_days=2,
        )
        rows = m.list_replay_sources()
        assert len(rows) == 1
        assert rows[0]["fingerprint"] == "fp2"

    def test_list_is_mru_ordered(self, tmp_path):
        m = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(m, "kv_old")
        _seed_replay(m, "kv_new")
        # Bump kv_old so it becomes most recent.
        _seed_replay(m, "kv_old")
        rows = m.list_replay_sources()
        assert [r["session_key"] for r in rows][0] == "kv_old"

    def test_prune_reports_expired_and_evicted(self, tmp_path):
        m = KVSessionManifest(str(tmp_path / "manifest.db"))
        # ttl_days<=0 means "no expiry", so use a tiny positive TTL and
        # prune with a future 'now' to exercise the expiry leg.
        _seed_replay(m, "kv_expired", ttl_days=0.0001)
        for i in range(4):
            _seed_replay(m, f"kv_keep_{i}")
        import time as _time
        expired, evicted = m.prune_replay_sources(
            max_rows=2, now=_time.time() + 60,
        )
        assert expired == 1
        assert evicted == 2
        assert len(m.list_replay_sources()) == 2

    def test_delete(self, tmp_path):
        m = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(m, "kv_abc")
        m.delete_replay_source("kv_abc")
        assert m.get_replay_source("kv_abc") is None


# ---------------------------------------------------------------------------
# capture (backend side)
# ---------------------------------------------------------------------------


class TestReplaySourcePayload:
    def test_prefers_stable_messages(self):
        req = _make_request(
            kv_stable_messages=[Message(role="user", content="stable")],
        )
        payload = LlamaCppBackend._replay_source_payload(req)
        assert payload == [{"role": "user", "content": "stable"}]

    def test_falls_back_to_messages(self):
        req = _make_request()
        payload = LlamaCppBackend._replay_source_payload(req)
        assert payload is not None
        assert [m["role"] for m in payload] == ["system", "user"]

    def test_rejects_images(self):
        req = _make_request(
            messages=[Message(role="user", content="look", images=["b64..."])],
        )
        assert LlamaCppBackend._replay_source_payload(req) is None

    def test_rejects_tool_turns(self):
        req = _make_request(
            messages=[
                Message(role="user", content="run it"),
                Message(role="assistant", content="", tool_calls=[{"id": "x"}]),
            ],
        )
        assert LlamaCppBackend._replay_source_payload(req) is None

    def test_rejects_non_string_content(self):
        req = _make_request(
            messages=[Message(role="user", content=[{"type": "text", "text": "hi"}])],
        )
        assert LlamaCppBackend._replay_source_payload(req) is None

    @pytest.mark.asyncio
    async def test_record_writes_through(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        backend._manager._session_manifest = manifest

        req = _make_request(kv_session_key="kv_abc", kv_mode="narrative")
        await backend._record_replay_source("kv_abc", req)

        row = manifest.get_replay_source("kv_abc")
        assert row is not None
        assert row["mode"] == "narrative"
        assert json.loads(row["messages_json"])[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_record_skips_oversize_without_truncating(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        backend._manager._session_manifest = manifest

        huge = "x" * (LlamaCppBackend._REPLAY_SOURCE_MAX_BYTES + 10)
        req = _make_request(
            kv_session_key="kv_huge",
            messages=[Message(role="user", content=huge)],
        )
        await backend._record_replay_source("kv_huge", req)
        assert manifest.get_replay_source("kv_huge") is None

    def test_schedule_noops_without_session_key(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        # No running loop → create_task would raise; the empty-key guard
        # must return before ever getting there.
        backend._schedule_replay_capture(_make_request(kv_session_key=""))


# ---------------------------------------------------------------------------
# ladder rung selection
# ---------------------------------------------------------------------------


class TestLadderRungSelection:
    @pytest.mark.asyncio
    async def test_empty_key_is_none_rung(self):
        ladder = KVResumeLadder(_ladder_backend())
        out = await ladder.resume_session("", source="test")
        assert out == {"rung": "none", "reason": "no_session_key"}

    @pytest.mark.asyncio
    async def test_not_ready_is_none_rung(self):
        backend = _ladder_backend()
        backend._manager.state = "stopped"
        ladder = KVResumeLadder(backend)
        out = await ladder.resume_session("kv_abc", source="test")
        assert out["rung"] == "none"
        assert out["reason"] == "engine_not_ready"

    @pytest.mark.asyncio
    async def test_hot_short_circuits(self):
        backend = _ladder_backend()
        backend._get_slot_for_session = MagicMock(return_value=2)
        ladder = KVResumeLadder(backend)
        out = await ladder.resume_session("kv_abc", source="test")
        assert out["rung"] == "hot"
        backend.prewarm_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_slot_restore_with_displacement_save(self):
        backend = _ladder_backend()
        backend._multislot_enabled = MagicMock(return_value=False)
        backend._manager._slot_save_supported = True
        backend._slot_state_exists = MagicMock(return_value=True)
        backend._get_session_for_slot = MagicMock(return_value="kv_other")
        ladder = KVResumeLadder(backend)

        out = await ladder.resume_session("kv_abc", source="test")

        assert out["rung"] == "restore"
        backend.save_session_state.assert_awaited_once_with("kv_other", slot_id=0)
        backend.restore_session_state.assert_awaited_once_with("kv_abc", slot_id=0)
        backend._claim_slot.assert_called_with(0, "kv_abc")

    @pytest.mark.asyncio
    async def test_single_slot_busy_stays_cold(self):
        backend = _ladder_backend()
        backend._multislot_enabled = MagicMock(return_value=False)
        backend._manager._slot_save_supported = True
        backend._manager.is_busy = True
        backend._slot_state_exists = MagicMock(return_value=True)
        ladder = KVResumeLadder(backend)

        out = await ladder.resume_session("kv_abc", source="test")

        assert out["rung"] == "cold"
        assert out["reason"] == "busy"
        backend.restore_session_state.assert_not_awaited()
        backend.prewarm_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multislot_replays_unpinned(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_abc")
        backend = _ladder_backend(manifest)
        backend.prewarm_context = AsyncMock(return_value=3)
        ladder = KVResumeLadder(backend)

        out = await ladder.resume_session("kv_abc", source="test")

        assert out["rung"] == "replay"
        assert out["slot"] == 3
        # Unpinned: llama-server routes; we pass slot_id=None.
        assert backend.prewarm_context.await_args.kwargs["slot_id"] is None
        backend._claim_slot.assert_called_with(3, "kv_abc")
        assert "kv_abc" in backend._manager._replay_warmed_keys

    @pytest.mark.asyncio
    async def test_replay_disabled_stays_cold(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_abc")
        backend = _ladder_backend(manifest)
        ladder = KVResumeLadder(backend)
        with _setting("engine_kv_replay_enabled", False):
            out = await ladder.resume_session("kv_abc", source="test")
        assert out["rung"] == "cold"
        assert out["reason"] == "replay_disabled"

    @pytest.mark.asyncio
    async def test_no_replay_source_stays_cold(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        backend = _ladder_backend(manifest)
        ladder = KVResumeLadder(backend)
        out = await ladder.resume_session("kv_missing", source="test")
        assert out["rung"] == "cold"
        assert out["reason"] == "no_replay_source"

    @pytest.mark.asyncio
    async def test_replay_failure_stays_cold(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_abc")
        backend = _ladder_backend(manifest)
        backend.prewarm_context = AsyncMock(return_value=None)
        ladder = KVResumeLadder(backend)
        out = await ladder.resume_session("kv_abc", source="test")
        assert out["rung"] == "cold"
        assert out["reason"] == "replay_failed"

    @pytest.mark.asyncio
    async def test_slot_zero_is_a_valid_replay_success(self, tmp_path):
        """Regression guard for the bool→int|None contract change:
        slot 0 is falsy but must count as success everywhere."""
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_abc")
        backend = _ladder_backend(manifest)
        backend.prewarm_context = AsyncMock(return_value=0)
        ladder = KVResumeLadder(backend)
        out = await ladder.resume_session("kv_abc", source="test")
        assert out["rung"] == "replay"
        assert out["slot"] == 0

    @pytest.mark.asyncio
    async def test_inflight_dedup(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_abc")
        backend = _ladder_backend(manifest)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_prewarm(*a, **k):
            started.set()
            await release.wait()
            return 1

        backend.prewarm_context = AsyncMock(side_effect=_slow_prewarm)
        ladder = KVResumeLadder(backend)

        first = asyncio.create_task(ladder.resume_session("kv_abc", source="a"))
        await started.wait()
        second = await ladder.resume_session("kv_abc", source="b")
        assert second == {"rung": "inflight"}
        release.set()
        out = await first
        assert out["rung"] == "replay"


# ---------------------------------------------------------------------------
# boot warm (budget + preemption)
# ---------------------------------------------------------------------------


class TestBootWarm:
    @pytest.mark.asyncio
    async def test_warms_up_to_session_budget(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        for i in range(5):
            _seed_replay(manifest, f"kv_s{i}")
        backend = _ladder_backend(manifest)
        ladder = KVResumeLadder(backend)

        with _setting("engine_kv_replay_warm_sessions", 2):
            await ladder.warm_recent_sessions()

        assert backend.prewarm_context.await_count == 2

    @pytest.mark.asyncio
    async def test_preempts_when_busy(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        for i in range(3):
            _seed_replay(manifest, f"kv_s{i}")
        backend = _ladder_backend(manifest)
        mgr = backend._manager

        # First candidate warms; then a "real request" arrives.
        async def _prewarm_then_busy(*a, **k):
            mgr.is_busy = True
            return 1

        backend.prewarm_context = AsyncMock(side_effect=_prewarm_then_busy)
        ladder = KVResumeLadder(backend)

        with _setting("engine_kv_replay_warm_sessions", 3):
            await ladder.warm_recent_sessions()

        assert backend.prewarm_context.await_count == 1

    @pytest.mark.asyncio
    async def test_single_slot_caps_at_one(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        for i in range(3):
            _seed_replay(manifest, f"kv_s{i}")
        backend = _ladder_backend(manifest)
        backend._multislot_enabled = MagicMock(return_value=False)
        ladder = KVResumeLadder(backend)

        with _setting("engine_kv_replay_warm_sessions", 3):
            await ladder.warm_recent_sessions()

        assert backend.prewarm_context.await_count == 1

    @pytest.mark.asyncio
    async def test_ttl_zero_rows_never_expire(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed_replay(manifest, "kv_forever", ttl_days=0)  # ttl<=0 → expires_at=0 → never
        backend = _ladder_backend(manifest)
        ladder = KVResumeLadder(backend)
        await ladder.warm_recent_sessions()
        # Genuinely-expired rows are covered by the store's prune test;
        # here: expires_at=0 must not read as "expired at epoch".
        assert backend.prewarm_context.await_count == 1

    @pytest.mark.asyncio
    async def test_no_candidates_is_quiet(self, tmp_path):
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        backend = _ladder_backend(manifest)
        ladder = KVResumeLadder(backend)
        await ladder.warm_recent_sessions()
        backend.prewarm_context.assert_not_awaited()


# ---------------------------------------------------------------------------
# prewarm_context return contract
# ---------------------------------------------------------------------------


class TestPrewarmReturnContract:
    @pytest.mark.asyncio
    async def test_success_returns_served_slot(self):
        async def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/apply-template"):
                return httpx.Response(200, json={"prompt": "hello"})
            if url.endswith("/tokenize"):
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            if url.endswith("/completion"):
                return httpx.Response(200, json={"id_slot": 2})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        out = await backend.prewarm_context([{"role": "user", "content": "hi"}])
        assert out == 2

    @pytest.mark.asyncio
    async def test_success_without_id_slot_returns_minus_one(self):
        async def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/apply-template"):
                return httpx.Response(200, json={"prompt": "hello"})
            if url.endswith("/tokenize"):
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            if url.endswith("/completion"):
                return httpx.Response(200, json={})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        out = await backend.prewarm_context([{"role": "user", "content": "hi"}])
        assert out == -1

    @pytest.mark.asyncio
    async def test_rejection_returns_none(self):
        async def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/apply-template"):
                return httpx.Response(200, json={"prompt": "hello"})
            if url.endswith("/tokenize"):
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            if url.endswith("/completion"):
                return httpx.Response(500, text="boom")
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()
        out = await backend.prewarm_context([{"role": "user", "content": "hi"}])
        assert out is None
