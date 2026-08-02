"""Edit driver + shared stream collector tests.

The collector is the consumer shared by the live coder route and the self-edit
driver; the driver wires it to run_store via an injected command_runner (the
container/subprocess seam), so a fake runner feeding canned JSONL exercises the
whole path without a container or a Claude token.
"""

from __future__ import annotations

import json
import pathlib

import aiosqlite

from augmentum.coder.external import run_store
from augmentum.coder.external.stream import ClaudeStreamCollector, summary_from_raw
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.edit_driver import run_engine_edit_driver
from augmentum.selfedit.orchestrator import EditRequest

_MIG_287 = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "287_claude_runs.sql"
)

# A realistic claude --output-format stream-json transcript.
_INIT = {"type": "system", "subtype": "init", "session_id": "sess-1"}
_WRITE = {"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Write", "input": {"file_path": "helper.py", "content": "x=1"}}]}}
_RESULT = {"type": "result", "subtype": "success", "is_error": False,
           "result": "Added helper.", "session_id": "sess-1",
           "total_cost_usd": 0.02, "num_turns": 3, "duration_ms": 1500}
_FAIL_RESULT = {"type": "result", "subtype": "error_max_turns", "is_error": True,
                "result": "ran out of turns"}


def _jsonl(*objs) -> bytes:
    return ("\n".join(json.dumps(o) for o in objs) + "\n").encode()


# ---------------------------------------------------------------------------
# ClaudeStreamCollector
# ---------------------------------------------------------------------------

async def test_collector_parses_full_run():
    events: list[dict] = []
    async def _emit(e):
        events.append(e)
    c = ClaudeStreamCollector(emit=_emit)
    await c.on_chunk(_jsonl(_INIT, _WRITE, _RESULT))
    await c.flush()
    assert c.ok is True and c.status == "done"
    assert c.files == ["helper.py"]
    assert c.meta["session_id"] == "sess-1" and c.meta["num_turns"] == 3
    assert c.outcome == "Added helper."
    # the "started" event is skipped; file_change + completed surface
    kinds = [e["kind"] for e in events]
    assert "started" not in kinds and "file_change" in kinds and "completed" in kinds


async def test_collector_buffers_across_chunk_boundaries():
    c = ClaudeStreamCollector()
    raw = _jsonl(_INIT, _WRITE, _RESULT)
    # feed one byte at a time — lines must still parse
    for i in range(len(raw)):
        await c.on_chunk(raw[i:i + 1])
    await c.flush()
    assert c.ok is True and c.files == ["helper.py"]


async def test_collector_failure_result():
    c = ClaudeStreamCollector()
    await c.on_chunk(_jsonl(_INIT, _FAIL_RESULT))
    await c.flush()
    assert c.ok is False and c.status == "failed" and "turns" in c.err


async def test_collector_empty_stream_settles_error():
    c = ClaudeStreamCollector()
    await c.flush()
    assert c.ok is False and c.err == "claude ended without a result"


def test_summary_from_raw():
    raw = [json.dumps(_INIT), json.dumps(_RESULT)]
    assert summary_from_raw(raw) == "Added helper."
    assert summary_from_raw([json.dumps(_FAIL_RESULT)]) == ""  # is_error → no summary


# ---------------------------------------------------------------------------
# run_engine_edit_driver
# ---------------------------------------------------------------------------

async def _db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)")
    await conn.executescript(_MIG_287.read_text())
    await conn.commit()
    return conn


def _cand() -> Candidate:
    return Candidate(name="att1", path="/tmp/wt/att1", branch="selfedit/att1",
                     base_ref="HEAD", base_sha="abc")


def _runner_feeding(*objs):
    captured = {"argv": None, "env": None}
    async def _run(*, request, argv, on_chunk, environment):
        captured["argv"] = argv
        captured["env"] = environment
        await on_chunk(_jsonl(*objs))
    return _run, captured


async def test_driver_records_run_and_returns_result():
    conn = await _db()
    try:
        runner, captured = _runner_feeding(_INIT, _WRITE, _RESULT)
        # Realistic OAuth subscription token — auth_env routes it to the OAuth env
        # var (an sk-ant-api… key would route to ANTHROPIC_API_KEY instead).
        tok = "sk-ant-oat01-tok"
        drive = run_engine_edit_driver(conn=conn, command_runner=runner, token=tok)
        res = await drive(EditRequest(candidate=_cand(), objective="add a helper",
                                      attempt_id="att1", user_id="u1"))
        assert res.ok is True and res.run_id and res.final_text == "Added helper."
        # the token went into the env, not the argv
        assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == tok
        assert tok not in " ".join(captured["argv"])
        # persisted as a claude_run
        row = await run_store.get_run(conn, run_id=res.run_id, user_id="u1")
        assert row["status"] == "done" and row["files_changed"] == ["helper.py"]
        assert row["session_id"] == "sess-1"
    finally:
        await conn.close()


async def test_driver_runner_crash_is_failed_not_raised():
    conn = await _db()
    try:
        async def _boom(*, request, argv, on_chunk, environment):
            raise RuntimeError("container died")
        drive = run_engine_edit_driver(conn=conn, command_runner=_boom)
        res = await drive(EditRequest(candidate=_cand(), objective="x",
                                      attempt_id="att1", user_id="u1"))
        assert res.ok is False and "container died" in res.error
        row = await run_store.get_run(conn, run_id=res.run_id, user_id="u1")
        assert row["status"] == "failed"
    finally:
        await conn.close()
