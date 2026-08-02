"""Tests for ``LlamaServerManager.start()`` idempotency.

When a model load is in flight (``state == STARTING``), a second
concurrent ``start()`` call for the SAME model must coalesce — wait
for the in-flight load instead of killing it and restarting. A
DIFFERENT-model start is a real swap and must kill+restart.

Original symptom (2026-05-06): user clicked "Load model", sent a chat
message before the load finished, observed the model "unload then
reload". The chat path's ``backend._ensure_server`` was racing the
explicit load route, both calling ``manager.start(path)``; the second
call killed the first's in-flight subprocess. Multi-slot's longer
startup made the race more reachable.

See ``LlamaServerManager._starting_future`` for the design.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from augmentum.models.llama_server_manager import LlamaServerManager, ProcessState


def _make_manager() -> LlamaServerManager:
    """A bare manager with the minimum wiring for start() coalescing tests.

    We patch out ``_start_impl`` (the actual subprocess work) so each
    test controls when the in-flight start "completes."
    """
    m = LlamaServerManager.__new__(LlamaServerManager)
    m.state = ProcessState.IDLE
    m.process = None
    m._starting_future = None
    m._starting_path = ""
    m.model_id = ""
    m.model_path = ""
    # start()-path state that __init__ would normally seed. The bare
    # __new__ fixture must track new start() dependencies or the task
    # dies mid-flight and the coalescing assertions read a reset state
    # (state=IDLE instead of STARTING) — that failure mode looks like a
    # coalescing bug but is fixture drift.
    m._load_duration_history = {}
    m._load_progress = None
    m._prefill_progress = None
    m._partial_offload_incompatible = False
    # _resolve_model_path: pass-through. Real implementation does
    # symlink resolution + alias lookup; tests don't need that.
    m._resolve_model_path = lambda p: p
    return m


async def _spin_until(cond, timeout: float = 2.0) -> None:
    """Poll ``cond`` until true — the awaits in start() ahead of the
    condition (off-loop path resolve, reconcile probe) make fixed
    yield counts nondeterministic."""
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(0.001)


class TestStartCoalescing:
    """Two concurrent ``start()`` calls for the same model must result
    in a single ``_start_impl`` invocation. The second caller awaits
    the first's future.
    """

    @pytest.mark.asyncio
    async def test_concurrent_same_model_coalesces(self):
        """The race the user hit live: explicit load + chat-path
        ensure_server hitting concurrently with the same model.
        Without coalescing, two _start_impl calls fire and the second
        kills the first's subprocess. With coalescing, exactly one
        runs.
        """
        m = _make_manager()
        impl_calls: list[str] = []
        impl_can_finish = asyncio.Event()

        async def slow_impl(path, *_args, **_kwargs):
            impl_calls.append(path)
            await impl_can_finish.wait()
            m.state = ProcessState.READY

        m._start_impl = slow_impl

        # Caller A — explicit load route
        task_a = asyncio.create_task(m.start("/models/x.gguf"))
        # start() awaits an off-loop (to_thread) path resolve before the
        # coalescing gate, so bare sleep(0) yields don't cover it — wait
        # on the observable claim instead of counting yields.
        await _spin_until(lambda: m.state == ProcessState.STARTING)
        assert len(impl_calls) == 1, "first start should be in-flight"

        # Caller B — chat path racing
        task_b = asyncio.create_task(m.start("/models/x.gguf"))
        # Cover B's own off-loop path resolve so it reaches the gate
        # while A is still in-flight.
        await asyncio.sleep(0.05)

        # B must NOT have called impl — coalesced behind A's future
        assert len(impl_calls) == 1, (
            f"second start spawned a new _start_impl ({len(impl_calls)} calls); "
            "coalescing failed"
        )

        # Release A's impl; both tasks should complete successfully
        impl_can_finish.set()
        await task_a
        await task_b

        assert m.state == ProcessState.READY
        assert len(impl_calls) == 1  # only one actual subprocess start

    @pytest.mark.asyncio
    async def test_three_concurrent_same_model_all_coalesce(self):
        """N concurrent callers all wait on the same future. Common
        deployment pattern: explicit load + chat + memory extraction
        racing on cold start.
        """
        m = _make_manager()
        impl_calls: list[str] = []
        impl_can_finish = asyncio.Event()

        async def slow_impl(path, *_args, **_kwargs):
            impl_calls.append(path)
            await impl_can_finish.wait()
            m.state = ProcessState.READY

        m._start_impl = slow_impl

        a = asyncio.create_task(m.start("/models/x.gguf"))
        await _spin_until(lambda: m.state == ProcessState.STARTING)
        b = asyncio.create_task(m.start("/models/x.gguf"))
        c = asyncio.create_task(m.start("/models/x.gguf"))
        # Cover B/C's off-loop path resolves so both reach the
        # coalescing gate while A is still in-flight.
        await asyncio.sleep(0.05)

        assert len(impl_calls) == 1

        impl_can_finish.set()
        await asyncio.gather(a, b, c)
        assert len(impl_calls) == 1

    @pytest.mark.asyncio
    async def test_different_model_kills_in_flight(self):
        """If a caller wants a different model, the in-flight start is
        killed via stop() and a fresh start begins for the new model.
        Production semantics: stop() kills the subprocess, the
        in-flight ``_start_impl`` sees its subprocess die during
        health-wait and raises (typically ``RuntimeError`` /
        ``TimeoutError``). The displaced caller sees that error so
        they know their start was preempted.
        """
        m = _make_manager()
        impl_calls: list[str] = []
        # ``killed`` is the simulated equivalent of "subprocess was
        # killed during health-wait" — when the in-flight impl wakes
        # up, it raises if the kill happened.
        kill_flag = {"killed": False}
        impl_started = asyncio.Event()
        impl_can_finish = asyncio.Event()

        async def slow_impl(path, *_args, **_kwargs):
            impl_calls.append(path)
            impl_started.set()
            await impl_can_finish.wait()
            if kill_flag["killed"] and path == "/models/x.gguf":
                # Simulate "subprocess died during health-wait."
                raise RuntimeError("simulated kill mid-load")
            m.state = ProcessState.READY

        m._start_impl = slow_impl

        stop_called = []
        async def fake_stop():
            stop_called.append(True)
            m.process = None
            m.state = ProcessState.IDLE
            kill_flag["killed"] = True
            impl_can_finish.set()
        m.stop = fake_stop

        # First caller — model X, will run impl
        task_a = asyncio.create_task(m.start("/models/x.gguf"))
        await impl_started.wait()
        m.process = MagicMock()  # simulate "process is running"

        # Reset for B's impl to set on its own start
        impl_started.clear()
        impl_can_finish.clear()

        # Second caller — model Y, must kill X and start Y
        task_b = asyncio.create_task(m.start("/models/y.gguf"))
        # Wait for B's impl to start (means stop() ran and B's
        # _start_impl is now waiting on the (re-cleared) finish event).
        await impl_started.wait()
        # stop() ran during B's path-mismatch handling
        assert stop_called == [True]

        # Release B's impl
        impl_can_finish.set()
        await task_b

        # A's task should have raised — it was preempted by B
        with pytest.raises(RuntimeError, match="simulated kill mid-load"):
            await task_a

        assert "/models/x.gguf" in impl_calls
        assert "/models/y.gguf" in impl_calls
        assert m.state == ProcessState.READY


class TestStartFailurePropagation:
    """When the in-flight start FAILS, awaiters must see the failure
    — they shouldn't silently succeed against a server that didn't
    actually start. Otherwise the chat path would proceed thinking
    the engine is up.
    """

    @pytest.mark.asyncio
    async def test_in_flight_failure_propagates_to_awaiter(self):
        m = _make_manager()
        impl_can_finish = asyncio.Event()
        boom = RuntimeError("simulated load failure")

        async def failing_impl(*_args, **_kwargs):
            await impl_can_finish.wait()
            raise boom

        m._start_impl = failing_impl

        task_a = asyncio.create_task(m.start("/models/x.gguf"))
        await asyncio.sleep(0); await asyncio.sleep(0)
        # B coalesces behind A
        task_b = asyncio.create_task(m.start("/models/x.gguf"))
        await asyncio.sleep(0)

        # Release impl — both raise
        impl_can_finish.set()

        with pytest.raises(RuntimeError, match="simulated load failure"):
            await task_a
        with pytest.raises(RuntimeError, match="simulated load failure"):
            await task_b

    @pytest.mark.asyncio
    async def test_post_failure_state_allows_retry(self):
        """After a failed start, the manager must be ready to accept
        a fresh start() call (no leftover _starting_future blocking).
        """
        m = _make_manager()

        async def failing_impl(*_args, **_kwargs):
            raise RuntimeError("first try fails")

        m._start_impl = failing_impl

        with pytest.raises(RuntimeError):
            await m.start("/models/x.gguf")

        # _starting_future cleared after failure
        assert m._starting_future is None

        # Second attempt — different impl, succeeds
        async def ok_impl(*_args, **_kwargs):
            m.state = ProcessState.READY

        m._start_impl = ok_impl
        await m.start("/models/x.gguf")
        assert m.state == ProcessState.READY


class TestStartingFutureCleanup:
    """``_starting_future`` must always be cleared after start() returns
    or raises. A leaked future would prevent future starts from
    coalescing correctly (or worse, awaiters could get stuck on a
    completed future that's been replaced).
    """

    @pytest.mark.asyncio
    async def test_future_cleared_on_success(self):
        m = _make_manager()
        async def ok_impl(*_args, **_kwargs):
            m.state = ProcessState.READY
        m._start_impl = ok_impl

        await m.start("/models/x.gguf")

        assert m._starting_future is None
        assert m._starting_path == ""

    @pytest.mark.asyncio
    async def test_future_cleared_on_exception(self):
        m = _make_manager()
        async def failing_impl(*_args, **_kwargs):
            raise RuntimeError("boom")
        m._start_impl = failing_impl

        with pytest.raises(RuntimeError):
            await m.start("/models/x.gguf")

        assert m._starting_future is None
        assert m._starting_path == ""
