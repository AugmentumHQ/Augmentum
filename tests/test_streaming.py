"""Tests for streaming support — NDJSON and SSE formats."""

from __future__ import annotations

import json

import pytest


def test_ollama_chat_streaming(client):
    """POST /api/chat with stream=true returns NDJSON."""
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers.get("content-type", "")

        chunks = []
        for line in resp.iter_lines():
            if line.strip():
                chunks.append(json.loads(line))

        # Should have content chunks + final done chunk
        assert len(chunks) >= 2
        assert chunks[-1]["done"] is True
        assert chunks[-1].get("done_reason") == "stop"

        # Reassemble content
        content = "".join(c["message"]["content"] for c in chunks)
        assert content == "Hello from mock Ollama!"


def test_ollama_generate_streaming(client):
    """POST /api/generate with stream=true returns NDJSON."""
    with client.stream(
        "POST",
        "/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": "Hello",
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200

        chunks = []
        for line in resp.iter_lines():
            if line.strip():
                chunks.append(json.loads(line))

        assert len(chunks) >= 2
        assert chunks[-1]["done"] is True

        content = "".join(c["response"] for c in chunks)
        assert content == "Hello from mock Ollama!"


def test_openai_chat_streaming(client):
    """POST /v1/chat/completions with stream=true returns SSE."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        chunks = []
        done_received = False
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                done_received = True
                break
            if line.startswith("data: "):
                data = json.loads(line[6:])
                chunks.append(data)

        assert done_received
        assert len(chunks) >= 2

        # All chunks should have SSE structure
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]

        # Reassemble content
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert content == "Hello from mock Ollama!"


def test_streaming_ndjson_format(client):
    """Verify NDJSON chunks are proper JSON with newline delimiter."""
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        raw = resp.read().decode()
        lines = [line for line in raw.split("\n") if line.strip()]
        for line in lines:
            # Each line should be valid JSON
            parsed = json.loads(line)
            assert "model" in parsed
            assert "done" in parsed


def test_streaming_sse_format(client):
    """Verify SSE chunks follow 'data: {json}\\n\\n' format."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        raw = resp.read().decode()
        # Every non-empty line should start with "data: "
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped:
                assert stripped.startswith("data: "), f"Invalid SSE line: {stripped}"



# ── prime_stream (2026-05-26) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_prime_stream_passes_first_chunk_through():
    """Happy path: primer pulls the first chunk + returns an iterator
    that yields it back, then forwards the rest.
    """
    from augmentum.proxy.streaming import prime_stream

    async def _gen():
        yield b"first"
        yield b"second"
        yield b"third"

    primed = await prime_stream(_gen())
    out = [chunk async for chunk in primed]
    assert out == [b"first", b"second", b"third"]


@pytest.mark.asyncio
async def test_prime_stream_surfaces_connect_error_before_headers():
    """If the first chunk raises (httpx.ConnectError shape), the primer
    raises StreamPrimeError — the caller can map to 502 BEFORE handing
    anything to StreamingResponse, so http.response.start is never
    sent. This is the whole point of the helper.
    """
    import httpx

    from augmentum.proxy.streaming import StreamPrimeError, prime_stream

    async def _gen():
        raise httpx.ConnectError("All connection attempts failed")
        yield b""  # unreachable; keeps the function an async-generator

    with pytest.raises(StreamPrimeError) as ei:
        await prime_stream(_gen())
    assert isinstance(ei.value.cause, httpx.ConnectError)


@pytest.mark.asyncio
async def test_prime_stream_empty_stream_raises():
    """Generator yields nothing then returns cleanly. Primer treats
    this as an upstream misconfiguration worth reporting (vs handing
    StreamingResponse a 0-byte body silently).
    """
    from augmentum.proxy.streaming import StreamPrimeError, prime_stream

    async def _gen():
        if False:
            yield b""  # async-generator marker; never executed

    with pytest.raises(StreamPrimeError) as ei:
        await prime_stream(_gen())
    assert isinstance(ei.value.cause, StopAsyncIteration)


@pytest.mark.asyncio
async def test_prime_stream_propagates_cancellation():
    """CancelledError is preserved (NOT wrapped in StreamPrimeError) —
    the asyncio loop needs the raw cancellation to unwind correctly.
    """
    import asyncio as _asyncio

    from augmentum.proxy.streaming import prime_stream

    async def _gen():
        raise _asyncio.CancelledError()
        yield b""

    with pytest.raises(_asyncio.CancelledError):
        await prime_stream(_gen())


@pytest.mark.asyncio
async def test_prime_stream_mid_stream_error_propagates_normally():
    """Errors AFTER the first chunk passed through cleanly are not
    primer's problem — they raise inside the StreamingResponse body
    iterator, the existing route-handler try/excepts handle them.
    """
    from augmentum.proxy.streaming import prime_stream

    async def _gen():
        yield b"first"
        raise RuntimeError("boom mid-stream")

    primed = await prime_stream(_gen())
    chunks = []
    with pytest.raises(RuntimeError, match="boom mid-stream"):
        async for chunk in primed:
            chunks.append(chunk)
    assert chunks == [b"first"]


# ── _with_heartbeat: chat_dispatch stage suppresses the stall watchdog ──
#
# Before this change, the wrapper opened with a generic heartbeat that
# the frontend treats as "TCP alive but no watchdog reset." Result:
# the "Stream stalled. Abort & retry." banner fired during a normal
# 30-60s model load because the inner handler can take many seconds
# before yielding its first model_load stage_start. The wrapper now
# opens with a proper stage_start: chat_dispatch event so the JS
# watchdog suspends from byte 0; matching stage_complete fires on
# the first inner chunk (or stream end).


@pytest.mark.asyncio
async def test_with_heartbeat_first_chunk_is_chat_dispatch_stage_start():
    """The very first emitted chunk must be a stage_start: chat_dispatch
    so the client's content-watchdog suspends before its 15s timer ticks.
    The dispatch stage also carries a stable id for the matching
    stage_complete to pair with later."""
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.streaming import _with_heartbeat

    async def _empty():
        return
        yield  # pragma: no cover  — make this an async generator

    chunks: list[InternalStreamChunk] = []
    async for c in _with_heartbeat(_empty()):
        chunks.append(c)

    assert chunks, "wrapper must emit at least the dispatch stage_start"
    first = chunks[0].augmentum or {}
    assert "stage_start" in first
    assert first["stage_start"]["stage"] == "chat_dispatch"
    assert first["stage_start"]["label"] == "Preparing"
    assert first["stage_start"]["id"].startswith("stg_dispatch_")
    # Backwards-compat: the legacy heartbeat fields still ride along on
    # the first chunk so any consumer that was watching for them still
    # has a wire signal.
    assert first.get("heartbeat") is True
    assert first.get("phase") == "starting"


@pytest.mark.asyncio
async def test_with_heartbeat_completes_dispatch_on_first_inner_chunk():
    """When the wrapped stream yields its first chunk, the wrapper
    must yield a stage_complete: chat_dispatch BEFORE forwarding the
    inner chunk so the JS active-stages set transitions cleanly into
    whatever stage the inner stream is starting (model_load, etc.)."""
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.streaming import _with_heartbeat

    async def _stream():
        yield InternalStreamChunk(
            augmentum={"stage_start": {
                "id": "stg_model_load_1", "stage": "model_load",
                "label": "Loading model", "detail": "deepseek-v3",
                "started_at": 0.0, "request_id": "",
            }},
        )

    chunks: list[InternalStreamChunk] = []
    async for c in _with_heartbeat(_stream()):
        chunks.append(c)

    # Three chunks: dispatch start (opening), dispatch complete (because
    # inner yielded), then the inner model_load stage_start.
    assert len(chunks) == 3
    assert (chunks[0].augmentum or {}).get("stage_start", {}).get("stage") == "chat_dispatch"
    assert (chunks[1].augmentum or {}).get("stage_complete", {}).get("stage") == "chat_dispatch"
    assert (chunks[2].augmentum or {}).get("stage_start", {}).get("stage") == "model_load"
    # Same dispatch id on the matching pair.
    start_id = chunks[0].augmentum["stage_start"]["id"]
    complete_id = chunks[1].augmentum["stage_complete"]["id"]
    assert start_id == complete_id


@pytest.mark.asyncio
async def test_with_heartbeat_dispatch_complete_on_empty_stream():
    """If the wrapped stream yields nothing at all, the wrapper must
    still emit a matching stage_complete so the frontend's set of
    active stages doesn't leak the dispatch entry across the request
    boundary."""
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.streaming import _with_heartbeat

    async def _empty():
        return
        yield  # pragma: no cover

    chunks: list[InternalStreamChunk] = []
    async for c in _with_heartbeat(_empty()):
        chunks.append(c)

    assert len(chunks) == 2
    assert (chunks[0].augmentum or {}).get("stage_start", {}).get("stage") == "chat_dispatch"
    assert (chunks[1].augmentum or {}).get("stage_complete", {}).get("stage") == "chat_dispatch"


# ── Model-load progress snapshot ───────────────────────────────────


def test_load_progress_snapshot_uses_history_median_when_present():
    """expected_s = median of recent successful loads for this model."""
    from augmentum.models.llama_server_manager import LlamaServerManager
    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {"my-model": [10.0, 20.0, 30.0]}
    snap = mgr._build_load_progress_snapshot("/models/my-model.gguf")
    assert snap["model_id"] == "my-model"
    assert snap["expected_s"] == 20.0  # median of [10, 20, 30]


def test_load_progress_snapshot_falls_back_to_file_size_estimate():
    """First-load (empty history) — derive expected_s from file size
    using a conservative 25 MB/s throughput heuristic, floor 5s."""
    import os
    import tempfile

    from augmentum.models.llama_server_manager import LlamaServerManager

    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {}
    # 250 MB → 250 / 25 = 10s
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"\0" * (250 * 1024 * 1024))
        path = f.name
    try:
        snap = mgr._build_load_progress_snapshot(path)
        # Wide tolerance — the math is `size / 25MB`; allow ~10% wiggle
        # for filesystem block-size rounding.
        assert 9.0 <= snap["expected_s"] <= 11.0
    finally:
        os.unlink(path)


def test_load_progress_snapshot_floor_for_tiny_models():
    """Even a 1MB model gets a 5s floor so the bar isn't 0s."""
    import os
    import tempfile

    from augmentum.models.llama_server_manager import LlamaServerManager

    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {}
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"\0" * 1024)
        path = f.name
    try:
        snap = mgr._build_load_progress_snapshot(path)
        assert snap["expected_s"] == 5.0
    finally:
        os.unlink(path)


def test_finalize_load_progress_records_duration_on_success():
    import time

    from augmentum.models.llama_server_manager import LlamaServerManager
    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {}
    mgr._load_progress = {
        "model_id": "x", "model_path": "/m/x.gguf",
        "started_at": time.monotonic() - 7.5,
        "size_bytes": 0, "expected_s": 30.0, "stage_label": "Loading model",
    }
    mgr._finalize_load_progress(success=True)
    assert mgr._load_progress is None
    hist = mgr._load_duration_history["x"]
    assert len(hist) == 1
    assert 7.0 < hist[0] < 8.5


def test_finalize_load_progress_no_record_on_failure():
    import time

    from augmentum.models.llama_server_manager import LlamaServerManager
    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {}
    mgr._load_progress = {
        "model_id": "x", "model_path": "/m/x.gguf",
        "started_at": time.monotonic() - 3.0,
        "size_bytes": 0, "expected_s": 30.0, "stage_label": "Loading model",
    }
    mgr._finalize_load_progress(success=False)
    assert mgr._load_progress is None
    # Failed loads don't pollute the median — keep the history clean.
    assert mgr._load_duration_history.get("x", []) == []


def test_load_progress_history_capped_at_12_entries():
    """Median must track the current install, not absorb a long tail
    of one-off slow loads from past transient conditions."""
    import time

    from augmentum.models.llama_server_manager import LlamaServerManager
    mgr = LlamaServerManager.__new__(LlamaServerManager)
    mgr._load_duration_history = {"x": [1.0] * 15}  # over the cap
    mgr._load_progress = {
        "model_id": "x", "model_path": "/m/x.gguf",
        "started_at": time.monotonic() - 5.0,
        "size_bytes": 0, "expected_s": 30.0, "stage_label": "Loading model",
    }
    mgr._finalize_load_progress(success=True)
    hist = mgr._load_duration_history["x"]
    assert len(hist) == 12
