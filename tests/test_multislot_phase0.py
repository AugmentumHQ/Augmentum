"""Phase 0 tests for the multi-slot KV architecture.

Verifies the diagnostic ``slot_observation`` log fires correctly when
``/completion`` SSE chunks include ``id_slot`` per the upstream wire
format at llama.cpp@b8935. No behavior change is introduced by this
phase — only observation. These tests lock in the parser's ability to
read the field so Phase 2 can confidently act on it.

See ``docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    Message,
)
from augmentum.models.llama_cpp import LlamaCppBackend


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


class _StreamingMockTransport(httpx.AsyncBaseTransport):
    """Minimal SSE-capable mock transport.

    The default MockTransport in test_model_backends.py only returns
    JSON bodies. For the streaming /completion path we need to send
    chunked SSE bytes that match upstream's wire format.
    """

    def __init__(self, completion_chunks: list[bytes]) -> None:
        self._chunks = completion_chunks
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("/completion"):
            return httpx.Response(
                status_code=200,
                content=b"".join(self._chunks),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        if url.endswith("/tokenize"):
            return httpx.Response(
                status_code=200,
                content=b'{"tokens":[1,2,3]}',
                headers={"content-type": "application/json"},
                request=request,
            )
        if url.endswith("/apply-template"):
            return httpx.Response(
                status_code=200,
                content=b'{"prompt":"<test>"}',
                headers={"content-type": "application/json"},
                request=request,
            )
        return httpx.Response(404, request=request, text="not in mock")


def _make_request() -> InternalChatRequest:
    return InternalChatRequest(
        model="test-model.gguf",
        messages=[Message(role="user", content="hello")],
        stream=True,
    )


class TestSlotObservation:
    """The /completion stream contract per b8935: id_slot is present
    in every partial chunk AND the final stop chunk. Phase 0 only
    needs to read it from any chunk and log it once. Phase 2 uses
    the same parser to drive occupancy tracking.
    """

    @pytest.mark.asyncio
    async def test_id_slot_logged_when_present_in_first_chunk(self, capfd):
        """Standard happy path: id_slot in first content chunk → log fires
        once with the correct value. Augmentum's structlog config emits
        rendered console output to stdout/stderr, so we capture via capfd
        rather than caplog (which only sees pre-render LogRecords).
        """
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False, "id_slot": 2}),
            _sse_chunk({"content": " there", "stop": False, "id_slot": 2}),
            _sse_chunk({
                "content": "",
                "stop": True,
                "id_slot": 2,
                "timings": {"prompt_n": 10, "predicted_n": 2},
            }),
            _sse_done(),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        tokens_in: list[int] = [1, 2, 3, 4, 5]
        stream = backend._stream_completion(_make_request(), tokens_in)
        async for _ in stream:
            pass

        out, err = capfd.readouterr()
        combined = out + err
        observations = [
            line for line in combined.splitlines() if "slot_observation" in line
        ]
        assert len(observations) == 1, (
            f"expected exactly one slot_observation log, got {len(observations)}: "
            f"{observations}"
        )
        assert "id_slot=2" in observations[0]
        assert "endpoint=completion" in observations[0]

    @pytest.mark.asyncio
    async def test_no_log_when_id_slot_absent(self, capfd):
        """Older llama-server builds (pre-id_slot) or unexpected response
        shapes: parser must not emit a misleading observation.
        """
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False}),
            _sse_chunk({
                "content": "",
                "stop": True,
                "timings": {"prompt_n": 5, "predicted_n": 1},
            }),
            _sse_done(),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        stream = backend._stream_completion(_make_request(), [1, 2])
        async for _ in stream:
            pass

        out, err = capfd.readouterr()
        combined = out + err
        assert "slot_observation" not in combined, (
            "slot_observation should not fire when id_slot is absent, "
            f"got output: {combined!r}"
        )

    @pytest.mark.asyncio
    async def test_negative_id_slot_treated_as_absent(self, capfd):
        """Defensive: if id_slot is present but -1 (auto / invalid),
        don't claim we observed a real assignment. -1 means
        "unassigned" in upstream's task model.
        """
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False, "id_slot": -1}),
            _sse_chunk({"content": "", "stop": True, "id_slot": -1}),
            _sse_done(),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        stream = backend._stream_completion(_make_request(), [1])
        async for _ in stream:
            pass

        out, err = capfd.readouterr()
        combined = out + err
        assert "slot_observation" not in combined, (
            "negative id_slot should be treated as absent (unassigned)"
        )

    @pytest.mark.asyncio
    async def test_logs_only_once_even_with_many_chunks(self, capfd):
        """Performance: streaming a long generation should log exactly
        one observation, not one per chunk. Avoids log-flood on long
        responses while still capturing the slot assignment.
        """
        chunks = [
            _sse_chunk({"content": "tok", "stop": False, "id_slot": 0})
            for _ in range(50)
        ]
        chunks.append(
            _sse_chunk({"content": "", "stop": True, "id_slot": 0})
        )
        chunks.append(_sse_done())

        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        stream = backend._stream_completion(_make_request(), [1])
        async for _ in stream:
            pass

        out, err = capfd.readouterr()
        combined = out + err
        observations = [
            line for line in combined.splitlines() if "slot_observation" in line
        ]
        assert len(observations) == 1, (
            f"expected exactly one log even with 50 chunks, got {len(observations)}"
        )

    @pytest.mark.asyncio
    async def test_logs_on_eof_no_done_path(self, capfd):
        """Edge case: stream EOFs without [DONE] marker (network blip,
        backend cut). Observation still fires from the EOF fallback so
        we don't lose the record. The eof_no_done=True flag distinguishes
        it from clean termination for diagnostics.
        """
        # No [DONE] marker — stream just ends after stop:true.
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False, "id_slot": 1}),
            _sse_chunk({"content": "", "stop": True, "id_slot": 1}),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        stream = backend._stream_completion(_make_request(), [1])
        async for _ in stream:
            pass

        out, err = capfd.readouterr()
        combined = out + err
        observations = [
            line for line in combined.splitlines() if "slot_observation" in line
        ]
        assert len(observations) == 1
        assert "id_slot=1" in observations[0]
        assert "eof_no_done=True" in observations[0]
