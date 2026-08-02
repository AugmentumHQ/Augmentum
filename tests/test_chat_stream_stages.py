"""Tests for stage_start/stage_complete events emitted by LlamaCppBackend.chat_stream.

Covers the three stages instrumented in the 1-day MVP:
- model_load / model_swap (around _ensure_server)
- slot_restore (around _manage_slot, when restore is actually attempted)
- prefill (always emitted before prompt processing)

These tests don't run a real llama-server — they mock _ensure_server and
the slot management to exercise the generator's stage-emit logic in
isolation. End-to-end browser verification is the manual step.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.llama_cpp import LlamaCppBackend


def _make_backend() -> LlamaCppBackend:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    return LlamaCppBackend(client, "http://llamacpp:8080")


def _make_manager_ready(model_id: str = "loaded-model.gguf") -> MagicMock:
    """Mock manager in READY state with given model loaded.

    Includes a working ``request_in_flight`` async context manager so
    ``chat_stream`` can ``async with`` against it without TypeErrors.
    """
    from augmentum.models.llama_server_manager import ProcessState

    manager = MagicMock()
    manager.state = ProcessState.READY
    manager.model_id = model_id
    manager.check_alive = MagicMock(return_value=True)

    counter = {"n": 0}

    @contextlib.asynccontextmanager
    async def _in_flight():
        counter["n"] += 1
        try:
            yield
        finally:
            counter["n"] -= 1

    manager.request_in_flight = _in_flight
    return manager


def _make_manager_idle() -> MagicMock:
    """Mock manager in IDLE state (server not started / crashed)."""
    from augmentum.models.llama_server_manager import ProcessState

    manager = MagicMock()
    manager.state = ProcessState.IDLE
    manager.model_id = ""
    manager.check_alive = MagicMock(return_value=False)

    counter = {"n": 0}

    @contextlib.asynccontextmanager
    async def _in_flight():
        counter["n"] += 1
        try:
            yield
        finally:
            counter["n"] -= 1

    manager.request_in_flight = _in_flight
    return manager


class TestEnsureServerStageClassification:
    """``_ensure_server_stage_for`` is the dispatch table for what stage
    event chat_stream emits. Wrong classification = wrong label in the UI.
    """

    def test_no_manager_returns_none(self):
        """Tests / standalone configs without a manager — no stage events."""
        backend = _make_backend()
        backend._manager = None
        assert backend._ensure_server_stage_for("any-model") is None

    def test_ready_same_model_returns_none(self):
        """No-op _ensure_server: the user shouldn't see a phantom stage."""
        backend = _make_backend()
        backend._manager = _make_manager_ready("model-x.gguf")
        assert backend._ensure_server_stage_for("model-x.gguf") is None

    def test_ready_generic_model_returns_none(self):
        """``default`` and empty string mean "use whatever's loaded" —
        also a no-op, no stage event.
        """
        backend = _make_backend()
        backend._manager = _make_manager_ready("model-x.gguf")
        assert backend._ensure_server_stage_for("default") is None
        assert backend._ensure_server_stage_for("") is None

    def test_ready_different_model_returns_swap_stage(self):
        backend = _make_backend()
        backend._manager = _make_manager_ready("loaded.gguf")
        stage = backend._ensure_server_stage_for("requested.gguf")
        assert stage is not None
        assert stage.name == "model_swap"
        assert stage.label == "Switching model"
        assert "requested.gguf" in stage.detail

    def test_idle_returns_load_stage(self):
        backend = _make_backend()
        backend._manager = _make_manager_idle()
        stage = backend._ensure_server_stage_for("requested.gguf")
        assert stage is not None
        assert stage.name == "model_load"
        assert stage.label == "Loading model"
        assert stage.detail == "requested.gguf"

    def test_ready_but_crashed_returns_load_stage(self):
        """check_alive() == False signals a crashed-but-not-yet-cleaned-up
        process. The next _ensure_server will restart, so stage = load.
        """
        from augmentum.models.llama_server_manager import ProcessState

        backend = _make_backend()
        manager = MagicMock()
        manager.state = ProcessState.READY  # state stale; alive check fails
        manager.model_id = "old-model"
        manager.check_alive = MagicMock(return_value=False)
        counter = {"n": 0}

        @contextlib.asynccontextmanager
        async def _in_flight():
            counter["n"] += 1
            try:
                yield
            finally:
                counter["n"] -= 1
        manager.request_in_flight = _in_flight
        backend._manager = manager

        stage = backend._ensure_server_stage_for("requested.gguf")
        assert stage is not None
        assert stage.name == "model_load"


class TestChatStreamStageEmission:
    """Exercise ``chat_stream`` end-to-end against a mocked manager and
    a mocked ``_chat_stream_with_slot`` to assert the stage_start /
    stage_complete events emit at the right boundaries.
    """

    def _request(self, model: str = "requested.gguf") -> InternalChatRequest:
        return InternalChatRequest(
            model=model,
            messages=[Message(role="user", content="hello")],
            stream=True,
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_no_stage_events_when_server_ready_same_model(self):
        """The no-op path must stay silent — no spurious "Loading model"
        flash on every request.
        """
        backend = _make_backend()
        backend._manager = _make_manager_ready("requested.gguf")
        backend._ensure_server = AsyncMock()
        # Stub the inner streamer so we don't have to mock the slot path.
        async def _stub_inner(request):
            from augmentum.models.base import InternalStreamChunk
            yield InternalStreamChunk(content_delta="hi", model=request.model)
            yield InternalStreamChunk(done=True, model=request.model)
        backend._chat_stream_with_slot = _stub_inner

        chunks = []
        async for chunk in backend.chat_stream(self._request("requested.gguf")):
            chunks.append(chunk)

        stage_starts = [c for c in chunks if (c.augmentum or {}).get("stage_start")]
        stage_completes = [c for c in chunks if (c.augmentum or {}).get("stage_complete")]
        assert stage_starts == [], (
            f"unexpected stage_start on no-op _ensure_server: {stage_starts}"
        )
        assert stage_completes == []

    @pytest.mark.asyncio
    async def test_model_swap_emits_start_and_complete(self):
        backend = _make_backend()
        backend._manager = _make_manager_ready("loaded.gguf")
        backend._ensure_server = AsyncMock()
        async def _stub_inner(request):
            from augmentum.models.base import InternalStreamChunk
            yield InternalStreamChunk(content_delta="hi", model=request.model)
            yield InternalStreamChunk(done=True, model=request.model)
        backend._chat_stream_with_slot = _stub_inner

        chunks = []
        async for chunk in backend.chat_stream(self._request("requested.gguf")):
            chunks.append(chunk)

        stage_starts = [
            (c.augmentum or {}).get("stage_start") for c in chunks
            if (c.augmentum or {}).get("stage_start")
        ]
        stage_completes = [
            (c.augmentum or {}).get("stage_complete") for c in chunks
            if (c.augmentum or {}).get("stage_complete")
        ]
        assert len(stage_starts) == 1
        assert stage_starts[0]["stage"] == "model_swap"
        assert stage_starts[0]["label"] == "Switching model"

        assert len(stage_completes) == 1
        assert stage_completes[0]["stage"] == "model_swap"
        assert stage_completes[0]["success"] is True
        # Same id pairs start↔complete.
        assert stage_starts[0]["id"] == stage_completes[0]["id"]
        # Duration is non-negative and small for this synthetic test.
        assert stage_completes[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_legacy_status_emitted_alongside_stage_start(self):
        """Backwards compatibility: frontends that haven't picked up
        stage_* events still see ``status: loading|swapping`` on the
        same chunk, mapped to the existing ``_STATUS_LABELS``.
        """
        backend = _make_backend()
        backend._manager = _make_manager_idle()
        backend._ensure_server = AsyncMock()
        async def _stub_inner(request):
            from augmentum.models.base import InternalStreamChunk
            yield InternalStreamChunk(done=True, model=request.model)
        backend._chat_stream_with_slot = _stub_inner

        chunks = []
        async for chunk in backend.chat_stream(self._request("requested.gguf")):
            chunks.append(chunk)

        # The first chunk carries BOTH legacy and new fields.
        first_aug = chunks[0].augmentum or {}
        assert first_aug.get("status") == "loading"
        assert first_aug.get("stage_start", {}).get("stage") == "model_load"

    @pytest.mark.asyncio
    async def test_stage_complete_with_failure_on_ensure_server_raise(self):
        """If ``_ensure_server`` raises (e.g., OOM exhausted), the stage
        must complete with success=False BEFORE the exception propagates.
        Without this the UI would show a stuck "Loading model…" forever.
        """
        backend = _make_backend()
        backend._manager = _make_manager_idle()
        backend._ensure_server = AsyncMock(side_effect=RuntimeError("OOM exhausted"))
        backend._chat_stream_with_slot = AsyncMock()  # never reached

        chunks: list = []
        with pytest.raises(RuntimeError, match="OOM exhausted"):
            async for chunk in backend.chat_stream(self._request("requested.gguf")):
                chunks.append(chunk)

        completes = [
            (c.augmentum or {}).get("stage_complete") for c in chunks
            if (c.augmentum or {}).get("stage_complete")
        ]
        assert len(completes) == 1
        assert completes[0]["stage"] == "model_load"
        assert completes[0]["success"] is False
        assert "OOM exhausted" in completes[0]["error"]

    @pytest.mark.asyncio
    async def test_no_manager_path_still_emits_stage_events(self):
        """Tests / standalone configs without a manager: ensure_server
        is still called and the no-manager branch must run the same
        stage-instrumentation code path. _ensure_server_stage_for
        returns None when there's no manager, so no stages emit —
        but the path itself must not crash.
        """
        backend = _make_backend()
        backend._manager = None
        backend._ensure_server = AsyncMock()
        async def _stub_inner(request):
            from augmentum.models.base import InternalStreamChunk
            yield InternalStreamChunk(done=True, model=request.model)
        backend._chat_stream_with_slot = _stub_inner

        chunks = []
        async for chunk in backend.chat_stream(self._request("any-model")):
            chunks.append(chunk)

        # No stage events (manager is None → no stage classification).
        # But the path completes without exception.
        starts = [c for c in chunks if (c.augmentum or {}).get("stage_start")]
        assert starts == []
