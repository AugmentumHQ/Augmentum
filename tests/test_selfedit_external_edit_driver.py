"""Tests for the backend-agnostic self-edit driver (any ExternalCoderDriver).

A fake ``ExternalCoderDriver`` yields normalized ``CoderEvent``s, proving the
bridge consumes ANY backend through one consumer. Load-bearing:
  - a completed stream → ok, final_text, file_change paths collected + persisted;
  - a failed stream → ok=False with the error;
  - a driver that raises → normalized to ok=False (never propagates);
  - build_selected_edit_driver wraps a selected backend, and returns None when no
    backend is available (caller falls back).
"""

from __future__ import annotations

from augmentum.coder.external.base import CoderEvent, ExternalCoderDriver
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.external_edit_driver import (
    build_selected_edit_driver,
    run_external_edit_driver,
)
from augmentum.selfedit.orchestrator import EditRequest


class FakeDriver(ExternalCoderDriver):
    def __init__(self, events=(), *, fail=False, did="fake"):
        self.id = did
        self.label = did
        self._events = list(events)
        self._fail = fail

    async def is_available(self) -> bool:
        return True

    async def run(self, task):
        if self._fail:
            raise RuntimeError("kaput")
        for ev in self._events:
            yield ev


class FakeConn:
    def __init__(self):
        self.executes: list = []

    async def execute(self, sql, params=()):
        self.executes.append((sql, params))

    async def commit(self):
        return None


def _cand() -> Candidate:
    return Candidate(name="att1", path="/tmp/wt/att1", branch="selfedit/att1",
                     base_ref="HEAD", base_sha="abc")


def _req() -> EditRequest:
    return EditRequest(candidate=_cand(), objective="add a helper",
                       attempt_id="att1", user_id="u1")


def _finish_call(conn: FakeConn):
    return next(c for c in conn.executes if "files_changed" in c[0])


async def test_completed_stream_collects_files_and_persists():
    conn = FakeConn()
    events = [
        CoderEvent(kind="message", text="working on it"),
        CoderEvent(kind="file_change", path="helper.py", tool="Write", mutating=True),
        CoderEvent(kind="file_change", path="helper.py"),         # dup → collected once
        CoderEvent(kind="completed", text="Added helper.", raw={"session_id": "s1"}),
    ]
    drive = run_external_edit_driver(conn=conn, driver=FakeDriver(events))
    res = await drive(_req())
    assert res.ok is True and res.final_text == "Added helper."
    status, _outcome, _err, files_json, *_rest = _finish_call(conn)[1]
    assert status == "done"
    assert "helper.py" in files_json and files_json.count("helper.py") == 1  # de-duped


async def test_failed_stream_sets_error():
    conn = FakeConn()
    drive = run_external_edit_driver(conn=conn, driver=FakeDriver([
        CoderEvent(kind="message", text="trying"),
        CoderEvent(kind="failed", text="exploded"),
    ]))
    res = await drive(_req())
    assert res.ok is False and res.error == "exploded"
    assert _finish_call(conn)[1][0] == "failed"


async def test_raising_driver_is_normalized_not_propagated():
    drive = run_external_edit_driver(conn=None, driver=FakeDriver(fail=True))
    res = await drive(_req())                       # must not raise
    assert res.ok is False and "kaput" in res.error


async def test_build_selected_wraps_available_backend():
    async def _select(prefer, *, cwd, claude_oauth_token, claude_api_key):
        return FakeDriver([CoderEvent(kind="completed", text="ok")], did="claude_code")

    drive = await build_selected_edit_driver(conn=None, prefer="claude_code", _select=_select)
    assert drive is not None
    res = await drive(_req())
    assert res.ok is True and res.final_text == "ok"


async def test_build_selected_none_when_no_backend():
    async def _select(prefer, *, cwd, claude_oauth_token, claude_api_key):
        return None

    drive = await build_selected_edit_driver(conn=None, _select=_select)
    assert drive is None                            # caller falls back to native
