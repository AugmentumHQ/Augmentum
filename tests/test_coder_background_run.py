"""Tests for the ``coder_background_run`` job handler (queued headless missions).

The handler's contract, locked in here:
  1. Malformed payloads fail loudly (ValueError) — never a silent no-op.
  2. Happy path drives the coder handler stream, appends the prompt +
     final answer to the persisted conversation, publishes a
     ``coder.run.complete`` notification, and returns run metadata.
  3. A stream failure notifies ``coder.run.failed`` BEFORE re-raising.
  4. A busy workspace (active broker run) requeues via JobRetryable
     instead of double-running the workspace.
  5. A factory fallback to a non-coder handler is a hard error (a
     passthrough "mission" would burn tokens with zero tools).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import augmentum.jobs.handlers.coder_background_run as mod
from augmentum.jobs.context import JobContext, JobRetryable
from augmentum.models.base import InternalStreamChunk


class _StubJobStore:
    def __init__(self):
        self.progress: list[tuple[float, str]] = []
        self.cancel = False

    async def update_progress(self, job_id, *, progress, stage=""):
        self.progress.append((progress, stage))

    async def is_cancel_requested(self, job_id):
        return self.cancel


def _ctx(payload, store=None):
    return JobContext(
        job_id="job_test1",
        user_id="usr_test",
        job_type=mod.JOB_TYPE,
        payload=payload,
        store=store or _StubJobStore(),
    )


class _StubPersistence:
    """Stands in for CoderPersistence — records saves."""

    instances: list[_StubPersistence] = []

    def __init__(self, conn):
        self.saved = None
        _StubPersistence.instances.append(self)

    async def load_conversation(self, workspace_id, *, user_id=""):
        return [
            {"id": "m1", "role": "user", "content": "earlier prompt"},
            {"id": "m2", "role": "assistant", "content": "earlier answer"},
            {"id": "m3", "role": "tool", "tool": "file_read", "input": {}},
        ]

    async def save_conversation(self, workspace_id, messages, *, user_id=""):
        self.saved = (workspace_id, messages, user_id)


class _StubRegistry:
    def __init__(self, backend="backend"):
        self.backend = backend
        self.calls = []

    async def resolve_backend_with_fabric(self, model, *, user_id="", session_id=""):
        self.calls.append((model, user_id))
        return self.backend, model


def _chunks_ok():
    return [
        InternalStreamChunk(augmentum={"run_id": "run_abc", "phase": "act", "status": "executing"}),
        InternalStreamChunk(content_delta="did the "),
        InternalStreamChunk(
            augmentum={"status": "tool_call", "tool_call": {"id": "t1", "tool": "code_edit"}},
        ),
        InternalStreamChunk(content_delta="thing."),
        InternalStreamChunk(
            done=True,
            augmentum={"status": "complete", "review_turn_id": "rev_9"},
        ),
    ]


class CoderHandler:  # noqa: N801 — name checked by the handler under test
    def __init__(self, chunks=None, error=None):
        self._chunks = chunks if chunks is not None else _chunks_ok()
        self._error = error

    async def handle_stream(self, request):
        for c in self._chunks:
            yield c
        if self._error is not None:
            raise self._error


class _WrongHandler:
    async def handle_stream(self, request):
        yield InternalStreamChunk(content_delta="hi")


def _app(registry=None, broker=None):
    return SimpleNamespace(
        state=SimpleNamespace(
            provider_registry=registry or _StubRegistry(),
            coder_run_broker=broker,
            state_manager=SimpleNamespace(backend=SimpleNamespace(conn=object())),
            notification_hub=None,
        ),
    )


@pytest.fixture(autouse=True)
def _patch_collaborators(monkeypatch):
    _StubPersistence.instances = []
    monkeypatch.setattr(
        "augmentum.state.coder_persistence.CoderPersistence", _StubPersistence,
    )
    notifications: list[dict] = []

    async def _fake_notify(app, **kw):
        notifications.append(kw)

    monkeypatch.setattr(mod, "_notify", _fake_notify)
    yield notifications


_PAYLOAD = {"workspace_id": "ws1", "prompt": "fix the bug", "model": "m-test"}


class TestValidation:
    @pytest.mark.parametrize("missing", ["workspace_id", "prompt", "model"])
    async def test_missing_field_raises(self, missing, monkeypatch):
        payload = dict(_PAYLOAD)
        payload[missing] = ""
        handler = mod.make_coder_background_run_handler(_app())
        with pytest.raises(ValueError):
            await handler(_ctx(payload))


class TestHappyPath:
    async def test_full_run(self, monkeypatch, _patch_collaborators):
        notifications = _patch_collaborators
        registry = _StubRegistry()
        app = _app(registry)
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            lambda *a, **k: CoderHandler(),
        )
        handler = mod.make_coder_background_run_handler(app)
        result = await handler(_ctx(dict(_PAYLOAD)))

        # Run metadata surfaced on the job row.
        assert result["run_id"] == "run_abc"
        assert result["review_turn_id"] == "rev_9"
        assert result["tool_calls"] == 1
        assert result["answer_chars"] == len("did the thing.")

        # Model resolved with the user's provider visibility.
        assert registry.calls == [("m-test", "usr_test")]

        # Conversation got the prompt + answer appended (fresh re-read:
        # the stub returns 3 rows, we append 2).
        saved_ws, saved_msgs, saved_uid = _StubPersistence.instances[-1].saved
        assert saved_ws == "ws1" and saved_uid == "usr_test"
        assert saved_msgs[-2]["role"] == "user"
        assert saved_msgs[-2]["content"] == "fix the bug"
        assert saved_msgs[-1]["role"] == "assistant"
        assert saved_msgs[-1]["content"] == "did the thing."

        # Completion notification with the deep-link payload.
        assert len(notifications) == 1
        n = notifications[0]
        assert n["ok"] is True
        assert n["payload"]["workspace_id"] == "ws1"
        assert n["payload"]["run_id"] == "run_abc"
        assert n["payload"]["review_turn_id"] == "rev_9"

    async def test_history_tool_rows_excluded_from_llm_context(self, monkeypatch):
        seen = {}

        def _capture_factory(mode, backend, session_id, state, **kw):
            class CoderHandler:  # noqa: N801
                async def handle_stream(self, request):
                    seen["messages"] = request.messages
                    yield InternalStreamChunk(content_delta="ok", done=True)
            return CoderHandler()

        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            _capture_factory,
        )
        handler = mod.make_coder_background_run_handler(_app())
        await handler(_ctx(dict(_PAYLOAD)))
        roles = [m.role for m in seen["messages"]]
        # user + assistant history (tool row dropped) + the new prompt.
        assert roles == ["user", "assistant", "user"]
        assert seen["messages"][-1].content == "fix the bug"


class TestFailurePath:
    async def test_stream_error_notifies_failed_and_raises(
        self, monkeypatch, _patch_collaborators,
    ):
        notifications = _patch_collaborators
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            lambda *a, **k: CoderHandler(error=RuntimeError("backend exploded")),
        )
        handler = mod.make_coder_background_run_handler(_app())
        with pytest.raises(RuntimeError, match="backend exploded"):
            await handler(_ctx(dict(_PAYLOAD)))
        assert len(notifications) == 1
        assert notifications[0]["ok"] is False
        assert "backend exploded" in notifications[0]["body"]

    async def test_non_coder_fallback_is_hard_error(self, monkeypatch):
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            lambda *a, **k: _WrongHandler(),
        )
        handler = mod.make_coder_background_run_handler(_app())
        with pytest.raises(RuntimeError, match="coder handler unavailable"):
            await handler(_ctx(dict(_PAYLOAD)))


class TestWorkspaceSerialization:
    async def test_busy_workspace_requeues(self, monkeypatch):
        class _BusyBroker:
            def get_active_for_workspace(self, *, user_id, workspace_id):
                return object()  # always busy

        # Zero out the wait so the test doesn't sleep for real.
        monkeypatch.setattr(mod, "_BUSY_WAIT_MAX_S", 0)
        handler = mod.make_coder_background_run_handler(
            _app(broker=_BusyBroker()),
        )
        with pytest.raises(JobRetryable):
            await handler(_ctx(dict(_PAYLOAD)))


class _IdleBroker:
    """Free workspace; records cancel() calls."""

    def __init__(self):
        self.cancelled: list[tuple[str, str]] = []

    def get_active_for_workspace(self, *, user_id, workspace_id):
        return None

    def cancel(self, run_id, *, reason=""):
        self.cancelled.append((run_id, reason))
        return True


class TestCancellation:
    async def test_user_cancel_stops_broker_and_never_notifies_failure(
        self, monkeypatch, _patch_collaborators,
    ):
        notifications = _patch_collaborators
        broker = _IdleBroker()
        store = _StubJobStore()

        chunks = [
            InternalStreamChunk(augmentum={"run_id": "run_c1"}),
        ] + [InternalStreamChunk(content_delta="x")] * 60  # crosses one heartbeat

        def _factory(*a, **k):
            return CoderHandler(chunks=chunks)

        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode", _factory,
        )
        store.cancel = True  # cancel flag already set → first heartbeat trips
        handler = mod.make_coder_background_run_handler(_app(broker=broker))
        from augmentum.jobs.context import JobCancelled
        with pytest.raises(JobCancelled):
            await handler(_ctx(dict(_PAYLOAD), store=store))
        # Broker run stopped with the user_cancel reason...
        assert broker.cancelled == [("run_c1", "user_cancel")]
        # ...and the user did NOT get a "mission failed" notification for
        # something they asked for (the original bug this test pins).
        assert notifications == []


class TestWallclockCeiling:
    async def test_timeout_cancels_broker_and_notifies(
        self, monkeypatch, _patch_collaborators,
    ):
        notifications = _patch_collaborators
        broker = _IdleBroker()
        # Let the payload's max_seconds go arbitrarily small.
        monkeypatch.setattr(mod, "_MISSION_MIN_S", 0.01)

        class _HangingHandler:
            async def handle_stream(self, request):
                yield InternalStreamChunk(augmentum={"run_id": "run_t1"})
                import asyncio as _a
                await _a.sleep(5)  # wedged stream — never yields again
                yield InternalStreamChunk(content_delta="never")

        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            lambda *a, **k: type("CoderHandler", (), {
                "handle_stream": _HangingHandler().handle_stream,
            })(),
        )
        payload = dict(_PAYLOAD)
        payload["max_seconds"] = 0.05
        handler = mod.make_coder_background_run_handler(_app(broker=broker))
        with pytest.raises(RuntimeError, match="wallclock ceiling"):
            await handler(_ctx(payload))
        assert broker.cancelled == [("run_t1", "background_timeout")]
        assert len(notifications) == 1
        assert notifications[0]["ok"] is False
        assert "timed out" in notifications[0]["title"]


class TestActionRegistration:
    def test_open_action_handler_registered_for_coder_run_channels(self):
        from augmentum.notifications.actions import resolve_handler
        assert resolve_handler("coder.run.complete") is mod._handle_open_action
        assert resolve_handler("coder.run.failed") is mod._handle_open_action
