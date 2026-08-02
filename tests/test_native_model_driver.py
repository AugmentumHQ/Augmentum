"""Tests for NativeModelDriver — a local model as a coder/self-edit backend.

Injected loop (fake), so deterministic. Load-bearing:
  - loop events normalize to CoderEvents (tool_call → file_change via the shared
    classifier; message; completed);
  - a failed loop event → failed; a raising loop → failed (never propagates);
  - it composes with the self-edit bridge: local-model loop → driver → EditResult
    (the sovereign self-edit path, end to end).
"""

from __future__ import annotations

from augmentum.coder.external.base import ExternalTask
from augmentum.coder.external.native_model_driver import NativeModelDriver
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.external_edit_driver import run_external_edit_driver
from augmentum.selfedit.orchestrator import EditRequest


def _loop_from(events, *, raise_at_end=False):
    async def loop(_task: ExternalTask):
        for ev in events:
            yield ev
        if raise_at_end:
            raise RuntimeError("loop boom")
    return loop


async def _collect(driver, task=None):
    out = []
    async for ev in driver.run(task or ExternalTask(prompt="x")):
        out.append(ev)
    return out


async def test_loop_events_normalize_to_coder_events():
    loop = _loop_from([
        {"kind": "message", "text": "planning"},
        {"kind": "tool_call", "tool": "edit_file", "args": {"file_path": "augmentum/x.py"}},
        {"kind": "tool_call", "tool": "Read", "args": {"file_path": "augmentum/y.py"}},
        {"kind": "completed", "text": "done", "session_id": "s1"},
    ])
    evs = await _collect(NativeModelDriver(run_loop=loop))
    kinds = [e.kind for e in evs]
    assert kinds == ["message", "file_change", "tool_call", "completed"]
    fc = next(e for e in evs if e.kind == "file_change")
    assert fc.path == "augmentum/x.py" and fc.mutating is True   # mutating tool → file_change
    assert evs[-1].raw.get("session_id") == "s1"                  # resume ref carried


async def test_failed_event_and_raising_loop_are_normalized():
    failed = await _collect(NativeModelDriver(run_loop=_loop_from(
        [{"kind": "failed", "text": "model errored"}])))
    assert failed[-1].kind == "failed" and failed[-1].text == "model errored"

    raised = await _collect(NativeModelDriver(run_loop=_loop_from([], raise_at_end=True)))
    assert raised[-1].kind == "failed" and "loop boom" in raised[-1].text  # never propagates


async def test_is_available_default_and_probe():
    assert await NativeModelDriver(run_loop=_loop_from([])).is_available() is True

    async def _down():
        return False
    assert await NativeModelDriver(run_loop=_loop_from([]), available=_down).is_available() is False


async def test_local_model_drives_self_edit_end_to_end():
    """The sovereign path: a local-model loop → driver → bridge → EditResult."""
    loop = _loop_from([
        {"kind": "message", "text": "editing"},
        {"kind": "tool_call", "tool": "write_file", "args": {"file_path": "helper.py"}},
        {"kind": "completed", "text": "Added helper.", "session_id": "s2"},
    ])
    driver = NativeModelDriver(run_loop=loop, did="native")
    drive = run_external_edit_driver(conn=None, driver=driver)   # the self-edit bridge
    res = await drive(EditRequest(
        candidate=Candidate(name="a1", path="/tmp/wt", branch="selfedit/a1",
                            base_ref="HEAD", base_sha="abc"),
        objective="add helper", attempt_id="a1", user_id="u1"))
    assert res.ok is True and res.final_text == "Added helper."  # local model, no token
