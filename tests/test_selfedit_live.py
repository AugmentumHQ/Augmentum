"""Live self-edit run tests — the streamed-theater backbone: the in-memory run
(buffer + bus + terminal), the context-scoped progress sink, and the background
launcher. No model, no DB — pure mechanics, so a regression here is loud."""

from __future__ import annotations

import asyncio

from augmentum.selfedit import live as L


class _State:
    """Stand-in for app.state (just an attribute bag)."""


# --- the run object ---------------------------------------------------------

def test_emit_assigns_seq_and_buffers():
    run = L.LiveRun("r1", user_id="u")
    run.emit({"kind": "phase", "phase": "target"})
    run.emit({"kind": "agent", "sub": "message", "text": "hi"})
    assert [e["seq"] for e in run.events] == [1, 2]
    assert run.events[0]["kind"] == "phase"


async def test_subscribe_receives_live_events():
    run = L.LiveRun("r1", user_id="u")
    q = run.subscribe()
    run.emit({"kind": "agent", "sub": "tool_call", "tool": "search"})
    item = await asyncio.wait_for(q.get(), timeout=1)
    assert item["kind"] == "agent" and item["tool"] == "search" and item["seq"] == 1


def test_finish_broadcasts_done_and_is_idempotent():
    run = L.LiveRun("r1", user_id="u")
    q = run.subscribe()
    run.finish({"status": "done", "ok": True})
    assert run.finished.is_set()
    done = q.get_nowait()
    assert done["kind"] == "done" and done["ok"] is True
    # second finish is a no-op (no extra event)
    run.finish({"status": "failed"})
    assert run.status == "done"


def test_snapshot_filters_by_since():
    run = L.LiveRun("r1", user_id="u", target="code_quality.silent_catches")
    for i in range(3):
        run.emit({"kind": "agent", "sub": "message", "text": str(i)})
    snap = run.snapshot(since=2)
    assert [e["seq"] for e in snap["events"]] == [3]
    assert snap["target"] == "code_quality.silent_catches"


def test_buffer_is_capped():
    run = L.LiveRun("r1", user_id="u")
    for _ in range(L._MAX_BUFFER + 50):
        run.emit({"kind": "agent", "sub": "message"})
    assert len(run.events) == L._MAX_BUFFER
    # the newest survive; the oldest dropped
    assert run.events[-1]["seq"] == L._MAX_BUFFER + 50


# --- the context-scoped progress sink ---------------------------------------

def test_emit_progress_is_noop_without_a_sink():
    # No scope set → must not raise, must not do anything.
    L.emit_progress({"kind": "phase", "phase": "target"})


async def test_launch_scopes_sink_so_pipeline_events_land():
    """The whole point: deep code calls the bare ``emit_progress`` and it reaches
    THIS run's bus because the launcher scoped the contextvar to its task."""
    state = _State()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _coro() -> dict:
        started.set()
        # a "deep" call with no handle to the run — relies on the scoped sink
        L.emit_progress({"kind": "phase", "phase": "agent", "text": "working"})
        await release.wait()
        return {"status": "done", "ok": True, "tier": "verified"}

    run = L.launch_live_run(state, user_id="u", run_id="r1", title="t",
                            target="x", ladder=["a", "b"], coro_factory=_coro)
    await asyncio.wait_for(started.wait(), timeout=1)
    # the opening "run" frame + the scoped "phase" event are both buffered
    kinds = [e["kind"] for e in run.events]
    assert kinds[0] == "run"
    assert any(e.get("phase") == "agent" for e in run.events)
    release.set()
    await asyncio.wait_for(run.finished.wait(), timeout=1)
    assert run.result["tier"] == "verified"
    assert run.events[-1]["kind"] == "done"


async def test_launch_failure_finishes_as_failed():
    state = _State()

    async def _coro() -> dict:
        raise RuntimeError("boom")

    run = L.launch_live_run(state, user_id="u", run_id="r2", title="t",
                            target="", ladder=[], coro_factory=_coro)
    await asyncio.wait_for(run.finished.wait(), timeout=1)
    assert run.status == "failed"
    assert "boom" in (run.result.get("error") or "")


async def test_emit_progress_does_not_leak_across_runs():
    """After a run's task ends, the sink must reset — a later bare emit_progress
    on the main task must not land on the finished run."""
    state = _State()

    async def _coro() -> dict:
        return {"status": "done", "ok": True}

    run = L.launch_live_run(state, user_id="u", run_id="r3", title="t",
                            target="", ladder=[], coro_factory=_coro)
    await asyncio.wait_for(run.finished.wait(), timeout=1)
    before = len(run.events)
    L.emit_progress({"kind": "phase", "phase": "stray"})
    assert len(run.events) == before  # the stray event went nowhere


# --- the manager ------------------------------------------------------------

async def test_manager_get_and_stop():
    state = _State()
    mgr = L.get_live_run_manager(state)
    assert mgr is L.get_live_run_manager(state)  # cached on app_state

    forever = asyncio.Event()

    async def _coro() -> dict:
        await forever.wait()
        return {"status": "done", "ok": True}

    run = L.launch_live_run(state, user_id="u", run_id="r4", title="t",
                            target="", ladder=[], coro_factory=_coro)
    assert mgr.get("r4") is run
    assert mgr.list_active(user_id="u") and not mgr.list_active(user_id="other")
    stopped = await mgr.stop("r4", user_id="u")
    assert stopped is True
    await asyncio.wait_for(run.finished.wait(), timeout=1)
    assert run.status == "cancelled"


async def test_manager_stop_rejects_other_user():
    state = _State()
    mgr = L.get_live_run_manager(state)
    forever = asyncio.Event()

    async def _coro() -> dict:
        await forever.wait()
        return {}

    L.launch_live_run(state, user_id="owner", run_id="r5", title="t",
                      target="", ladder=[], coro_factory=_coro)
    assert await mgr.stop("r5", user_id="intruder") is False
    forever.set()
