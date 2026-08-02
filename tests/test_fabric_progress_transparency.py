"""Cross-peer state transparency: a peer's model-load + prefill progress
surfaces to the originator through the SAME progress endpoints + UI poller
as a local load (2026-06-12).

Covers the four seams:
  1. shared progress-payload builders (models/load_progress.py)
  2. coordinator peer-progress cache (record/read/expire)
  3. receiver /api/fabric/load_status progress enrichment
  4. sender FabricBackend: load -> coordinator cache + model_load stage
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import InternalChatRequest, Message

# -- 1. Shared builders --------------------------------------------

def test_build_load_progress_payload_shapes_and_caps():
    from augmentum.models.load_progress import build_load_progress_payload

    assert build_load_progress_payload(None) == {"active": False}

    snap = {
        "model_id": "deepseek-v3", "size_bytes": 1234,
        "stage_label": "Loading model", "started_at": 100.0, "expected_s": 30.0,
    }
    # 15s into a 30s load -> 0.5
    p = build_load_progress_payload(snap, now_monotonic=115.0)
    assert p["active"] is True
    assert p["model_id"] == "deepseek-v3"
    assert p["progress"] == 0.5
    assert p["elapsed_s"] == 15.0
    assert p["expected_s"] == 30.0
    # Past expected -> capped at 0.95, never 1.0 while still loading.
    capped = build_load_progress_payload(snap, now_monotonic=1000.0)
    assert capped["progress"] == 0.95


def test_build_prefill_progress_payload_staleness():
    from augmentum.models.load_progress import build_prefill_progress_payload

    assert build_prefill_progress_payload(None) == {"active": False}

    snap = {"tokens_done": 4096, "progress": 0.47, "elapsed_s": 14.5,
            "tps": 282.1, "updated_at": 1000.0}
    fresh = build_prefill_progress_payload(snap, now_wall=1002.0)
    assert fresh["active"] is True
    assert fresh["tokens_done"] == 4096
    assert fresh["progress"] == 0.47
    # Older than the 8s staleness ceiling -> inactive (bar clears).
    stale = build_prefill_progress_payload(snap, now_wall=1100.0)
    assert stale["active"] is False
    assert "age_s" in stale


# -- 2. Coordinator cache ------------------------------------------

def _coordinator():
    # The progress cache is pure in-memory; identity/db are unused by it,
    # so mocks are sufficient for these unit tests.
    from augmentum.fabric.coordinator import FabricCoordinator
    return FabricCoordinator(MagicMock(), MagicMock())


def test_coordinator_progress_cache_roundtrip_and_isolation():
    coord = _coordinator()
    coord.record_peer_load_progress("model-a", {"active": True, "progress": 0.3})
    coord.record_peer_load_progress("model-b", {"active": True, "progress": 0.7})

    a = coord.peer_load_progress("model-a")
    b = coord.peer_load_progress("model-b")
    assert a["progress"] == 0.3
    assert b["progress"] == 0.7
    # Internal bookkeeping key is not leaked to the wire shape.
    assert "recorded_at" not in a
    # Unknown model -> None (not a stale/empty dict).
    assert coord.peer_load_progress("model-c") is None


def test_coordinator_progress_cache_expires():
    coord = _coordinator()
    # Record at t=100, read past the staleness ceiling -> expired -> None.
    with patch("augmentum.fabric.coordinator.time.monotonic", return_value=100.0):
        coord.record_peer_load_progress("m", {"active": True, "progress": 0.5})
    later = 100.0 + coord._PEER_PROGRESS_STALE_AFTER_S + 1.0
    with patch("augmentum.fabric.coordinator.time.monotonic", return_value=later):
        assert coord.peer_load_progress("m") is None


def test_coordinator_progress_cache_rejects_empty():
    coord = _coordinator()
    coord.record_peer_load_progress("", {"active": True})
    coord.record_peer_load_progress("m", None)  # type: ignore[arg-type]
    assert coord.peer_load_progress("m") is None


# -- 3. Receiver enrichment ----------------------------------------

@pytest.mark.asyncio
async def test_load_status_embeds_progress_while_loading():
    from augmentum.models.llama_server_manager import ProcessState
    from augmentum.proxy.fabric_routes import fabric_load_status

    manager = MagicMock()
    manager.state = ProcessState.STARTING
    manager.model_id = "other"
    manager.check_alive = MagicMock(return_value=True)
    manager._load_progress = {
        "model_id": "m", "size_bytes": 10, "stage_label": "Loading model",
        "started_at": 0.0, "expected_s": 30.0,
    }
    manager._prefill_progress = None

    request = MagicMock()
    request.app.state.llama_manager = manager
    # An in-flight (not-done) load task for model "m".
    task = MagicMock()
    task.done = MagicMock(return_value=False)
    request.app.state._fabric_load_tasks = {"m": task}

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        result = await fabric_load_status(request, model_id="m")

    assert result["status"] == "loading"
    assert "load_progress" in result
    assert result["load_progress"]["active"] is True
    assert result["load_progress"]["model_id"] == "m"


# -- 4. Sender: load drives cache + model_load stage ---------------

def _llm_cap():
    from augmentum.fabric.capabilities import LLMInferenceCapability
    return LLMInferenceCapability(
        backend="peer", model_id="m", model_family="qwen3",
        params_b=7.0, ctx_max=8192,
    )


@pytest.mark.asyncio
async def test_chat_stream_emits_model_load_stage_and_records_cache():
    """A cold peer load yields a model_load stage_start before the
    inference SSE (suspends the UI watchdog + starts the load poller),
    a stage_complete when ready, and records the peer's progress into
    the coordinator cache so /api/engine/v2/load_progress can surface it.
    """
    import httpx

    from augmentum.models.fabric_backend import FabricBackend

    coord = MagicMock()

    # Drive: kick -> loading, poll #1 -> loading + progress, poll #2 -> ready.
    kick_resp = MagicMock(spec=httpx.Response)
    kick_resp.status_code = 200
    kick_resp.json.return_value = {"status": "loading", "current_model": "old"}

    poll1 = MagicMock(spec=httpx.Response)
    poll1.status_code = 200
    poll1.json.return_value = {
        "status": "loading", "model_id": "m",
        "load_progress": {"active": True, "model_id": "m", "progress": 0.4},
    }
    poll2 = MagicMock(spec=httpx.Response)
    poll2.status_code = 200
    poll2.json.return_value = {"status": "ready", "model_id": "m"}

    # Inference stream after load: a single content chunk + done.
    async def fake_aiter_lines():
        import json as _json
        yield "data: " + _json.dumps({
            "model": "m",
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]"

    stream_resp = MagicMock(spec=httpx.Response)
    stream_resp.status_code = 200
    stream_resp.aiter_lines = fake_aiter_lines

    class _StreamCM:
        async def __aenter__(self):
            return stream_resp
        async def __aexit__(self, *a, **k):
            return False

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=kick_resp)
    fake_client.get = AsyncMock(side_effect=[poll1, poll2])
    fake_client.stream = MagicMock(return_value=_StreamCM())

    backend = FabricBackend(
        http_client=fake_client, peer_node_id="peer-x",
        peer_addr="10.0.0.2:6443", advertised_capability=_llm_cap(),
        coordinator=coord,
    )
    # Make the poll loop instant.
    with patch("augmentum.models.fabric_backend._LOAD_POLL_INTERVAL_S", 0.0):
        request = InternalChatRequest(
            model="m", messages=[Message(role="user", content="hi")], stream=True,
        )
        chunks = [c async for c in backend.chat_stream(request)]

    # Peer progress reached the coordinator cache.
    coord.record_peer_load_progress.assert_any_call(
        "m", {"active": True, "model_id": "m", "progress": 0.4},
    )

    # A model_load stage_start preceded the inference content, and a
    # stage_complete(success=True) closed it.
    stage_starts = [
        c.augmentum["stage_start"] for c in chunks
        if c.augmentum and "stage_start" in c.augmentum
    ]
    stage_completes = [
        c.augmentum["stage_complete"] for c in chunks
        if c.augmentum and "stage_complete" in c.augmentum
    ]
    assert len(stage_starts) == 1
    assert stage_starts[0]["stage"] == "model_load"
    assert stage_starts[0]["detail"] == "m"
    assert len(stage_completes) == 1
    assert stage_completes[0]["success"] is True
    # Inference content still arrives after the stage.
    assert any(c.content_delta == "hi" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_warm_peer_emits_no_stage():
    """A warm peer (kick -> ready) must NOT emit a model_load stage --
    no flicker for the common already-loaded case."""
    import httpx

    from augmentum.models.fabric_backend import FabricBackend

    kick_resp = MagicMock(spec=httpx.Response)
    kick_resp.status_code = 200
    kick_resp.json.return_value = {"status": "ready"}

    async def fake_aiter_lines():
        import json as _json
        yield "data: " + _json.dumps({
            "model": "m",
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]"

    stream_resp = MagicMock(spec=httpx.Response)
    stream_resp.status_code = 200
    stream_resp.aiter_lines = fake_aiter_lines

    class _StreamCM:
        async def __aenter__(self):
            return stream_resp
        async def __aexit__(self, *a, **k):
            return False

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=kick_resp)
    fake_client.stream = MagicMock(return_value=_StreamCM())

    backend = FabricBackend(
        http_client=fake_client, peer_node_id="peer-x",
        peer_addr="10.0.0.2:6443", advertised_capability=_llm_cap(),
    )
    request = InternalChatRequest(
        model="m", messages=[Message(role="user", content="hi")], stream=True,
    )
    chunks = [c async for c in backend.chat_stream(request)]
    assert not any(
        c.augmentum and ("stage_start" in c.augmentum or "stage_complete" in c.augmentum)
        for c in chunks
    )
    assert any(c.content_delta == "hi" for c in chunks)


# -- 5. Prefill transparency (P2): stage crosses the wire + cache fed ----


def test_receiver_sse_dict_forwards_prefill_stage_augmentum():
    """The receiver embeds the local backend's augmentum stage metadata
    (notably the ``prefill`` stage_start it emits before blocking on
    prompt processing) as a top-level ``augmentum`` block, so the sender
    can relay it. tool_calls already ride in delta.tool_calls — they must
    NOT be double-forwarded into the augmentum block."""
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.fabric_routes import _internal_chunk_to_openai_sse_dict

    chunk = InternalStreamChunk(
        content_delta="", model="m",
        augmentum={
            "status": "tokenizing",
            "stage_start": {"id": "stg1", "stage": "prefill",
                            "label": "Preparing context"},
            "tool_calls": [{"index": 0, "id": "call_1"}],
        },
    )
    sse = _internal_chunk_to_openai_sse_dict(chunk, "cid", "m")
    assert "augmentum" in sse
    assert sse["augmentum"]["stage_start"]["stage"] == "prefill"
    assert sse["augmentum"]["status"] == "tokenizing"
    # tool_calls forwarded via delta, stripped from the augmentum block.
    assert sse["choices"][0]["delta"]["tool_calls"] == [{"index": 0, "id": "call_1"}]
    assert "tool_calls" not in sse["augmentum"]


def test_receiver_sse_dict_no_augmentum_when_none():
    """A plain content chunk carries no augmentum block."""
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.fabric_routes import _internal_chunk_to_openai_sse_dict

    sse = _internal_chunk_to_openai_sse_dict(
        InternalStreamChunk(content_delta="hi", model="m"), "cid", "m",
    )
    assert "augmentum" not in sse


@pytest.mark.asyncio
async def test_sender_parser_relays_top_level_augmentum():
    """The sender's SSE parser lifts the receiver's top-level
    ``augmentum`` block back into chunk.augmentum — so a relayed prefill
    stage_start reaches the UI stage handler and starts the prefill bar,
    even on an empty-content chunk."""
    import json as _json
    from unittest.mock import MagicMock as _MM

    import httpx

    from augmentum.models.fabric_backend import _parse_openai_sse_stream

    async def fake_aiter_lines():
        yield "data: " + _json.dumps({
            "model": "m",
            "choices": [{"delta": {}, "finish_reason": None}],
            "augmentum": {
                "status": "tokenizing",
                "stage_start": {"id": "s1", "stage": "prefill",
                                "label": "Preparing context"},
            },
        })
        yield "data: [DONE]"

    resp = _MM(spec=httpx.Response)
    resp.aiter_lines = fake_aiter_lines

    chunks = [c async for c in _parse_openai_sse_stream(resp, "m")]
    staged = [c for c in chunks if c.augmentum and "stage_start" in c.augmentum]
    assert len(staged) == 1
    assert staged[0].augmentum["stage_start"]["stage"] == "prefill"
    assert staged[0].content_delta == ""


@pytest.mark.asyncio
async def test_poll_prefill_into_cache_records_progress():
    """The prefill side-poll reads /load_status (which embeds
    prefill_progress while the model is READY + prompt processing) and
    mirrors it into the coordinator cache under the model_id, so the
    originator's /api/engine/v2/prefill_progress surfaces it."""
    import asyncio as _asyncio
    import contextlib
    from unittest.mock import MagicMock as _MM

    import httpx

    from augmentum.models.fabric_backend import FabricBackend

    coord = _MM()
    status_resp = _MM(spec=httpx.Response)
    status_resp.status_code = 200
    status_resp.json.return_value = {
        "status": "ready", "model_id": "m",
        "prefill_progress": {"active": True, "progress": 0.47, "tps": 282.1},
    }
    fake_client = _MM()
    fake_client.get = AsyncMock(return_value=status_resp)

    backend = FabricBackend(
        http_client=fake_client, peer_node_id="peer-x",
        peer_addr="10.0.0.2:6443", advertised_capability=_llm_cap(),
        coordinator=coord,
    )

    with patch("augmentum.models.fabric_backend._PREFILL_POLL_INTERVAL_S", 0.0):
        task = _asyncio.create_task(backend._poll_prefill_into_cache("m"))
        # Let the poll loop run a few iterations, then stop it.
        for _ in range(5):
            await _asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(_asyncio.CancelledError):
            await task

    coord.record_peer_prefill_progress.assert_any_call(
        "m", {"active": True, "progress": 0.47, "tps": 282.1},
    )


@pytest.mark.asyncio
async def test_poll_prefill_noop_without_coordinator():
    """No coordinator → the side-poll is a no-op (legacy/test paths)."""
    from unittest.mock import MagicMock as _MM

    from augmentum.models.fabric_backend import FabricBackend

    fake_client = _MM()
    fake_client.get = AsyncMock()
    backend = FabricBackend(
        http_client=fake_client, peer_node_id="peer-x",
        peer_addr="10.0.0.2:6443", advertised_capability=_llm_cap(),
    )
    await backend._poll_prefill_into_cache("m")
    fake_client.get.assert_not_called()
