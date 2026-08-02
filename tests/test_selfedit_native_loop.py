"""Tests for the local-model agentic edit loop (the sovereign self-edit engine).

The model turn is injected (a scripted fake), so the loop mechanics are fully
testable with a temp workspace — no model needed. Load-bearing:
  - tools really edit files in the workspace (write/edit), and only there
    (path-traversal refused); read/list work;
  - it terminates: finish tool, no-tool turn, or the iteration cap — never loops;
  - a tool error / unknown tool / model error never crashes (normalized);
  - it composes with NativeModelDriver + the bridge → an EditResult (the full
    sovereign path: local-loop → driver → self-edit, no token).
"""

from __future__ import annotations

from augmentum.coder.external.base import ExternalTask
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.external_edit_driver import run_external_edit_driver
from augmentum.selfedit.native_loop import (
    _normalize_calls,
    make_native_loop,
)
from augmentum.selfedit.orchestrator import EditRequest


def _scripted(turns):
    state = {"i": 0}

    async def chat(_messages, _specs):
        i = state["i"]
        state["i"] += 1
        return turns[i] if i < len(turns) else {"content": "", "tool_calls": []}
    return chat


def _call(name, args, cid="1"):
    return {"name": name, "args": args, "id": cid}


async def _run(loop, ws, prompt="do it"):
    return [e async for e in loop(ExternalTask(prompt=prompt, workspace=ws))]


# --- tools really edit, confined to the workspace --------------------------

async def test_write_then_finish_creates_file(tmp_path):
    chat = _scripted([
        {"content": "writing", "tool_calls": [_call("write_file", {"path": "x.py", "content": "hi"})]},
        {"content": "done", "tool_calls": [_call("finish", {"summary": "added x"})]},
    ])
    evs = await _run(make_native_loop(chat=chat), str(tmp_path))
    assert [e["kind"] for e in evs][-1] == "completed" and evs[-1]["text"] == "added x"
    assert any(e["kind"] == "tool_call" and e["tool"] == "write_file" for e in evs)
    assert (tmp_path / "x.py").read_text() == "hi"


async def test_edit_file_replaces_unique(tmp_path):
    (tmp_path / "f.py").write_text("alpha BETA gamma")
    chat = _scripted([
        {"content": "", "tool_calls": [_call("edit_file",
            {"path": "f.py", "old_string": "BETA", "new_string": "DELTA"})]},
    ])  # next turn defaults to no-tool → completed
    await _run(make_native_loop(chat=chat), str(tmp_path))
    assert (tmp_path / "f.py").read_text() == "alpha DELTA gamma"


async def test_edit_rejects_nonunique_and_missing(tmp_path):
    (tmp_path / "f.py").write_text("x x")
    chat = _scripted([{"content": "", "tool_calls": [
        _call("edit_file", {"path": "f.py", "old_string": "x", "new_string": "y"})]}])
    await _run(make_native_loop(chat=chat), str(tmp_path))
    assert (tmp_path / "f.py").read_text() == "x x"          # not-unique → refused, no change


async def test_path_traversal_is_refused(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    chat = _scripted([{"content": "", "tool_calls": [
        _call("write_file", {"path": "../escape.py", "content": "pwned"})]}])
    await _run(make_native_loop(chat=chat), str(ws))
    assert not (tmp_path / "escape.py").exists()             # never escapes the workspace


# --- termination + robustness ---------------------------------------------

async def test_no_tool_turn_completes(tmp_path):
    chat = _scripted([{"content": "nothing to do", "tool_calls": []}])
    evs = await _run(make_native_loop(chat=chat), str(tmp_path))
    assert evs[-1]["kind"] == "completed"


async def test_iteration_cap_terminates(tmp_path):
    # never calls finish — would loop forever without the cap. Vary the call each
    # turn (distinct paths) so it's the CAP that stops it, not the stagnation
    # breaker (which handles the identical-call case separately).
    chat = _scripted([{"content": "", "tool_calls": [_call("list_dir", {"path": f"d{k}"})]}
                      for k in range(50)])
    evs = await _run(make_native_loop(chat=chat, max_iters=3), str(tmp_path))
    assert evs[-1]["kind"] == "completed" and "cap" in evs[-1]["text"]
    assert sum(1 for e in evs if e["kind"] == "tool_call") == 3   # bounded


async def test_tool_and_model_errors_are_normalized(tmp_path):
    # unknown tool → ERROR result, loop continues to completion (no crash)
    ok = await _run(make_native_loop(chat=_scripted([
        {"content": "", "tool_calls": [_call("nope", {})]}])), str(tmp_path))
    assert ok[-1]["kind"] == "completed"

    async def boom(_m, _s):
        raise RuntimeError("model down")
    bad = await _run(make_native_loop(chat=boom), str(tmp_path))
    assert bad[-1]["kind"] == "failed" and "model down" in bad[-1]["text"]


# --- full sovereign path: native loop → driver → self-edit -----------------

async def test_composes_with_driver_to_editresult(tmp_path):
    from augmentum.coder.external.native_model_driver import NativeModelDriver

    chat = _scripted([
        {"content": "", "tool_calls": [_call("write_file", {"path": "h.py", "content": "x"})]},
        {"content": "", "tool_calls": [_call("finish", {"summary": "added h"})]},
    ])
    driver = NativeModelDriver(run_loop=make_native_loop(chat=chat))
    drive = run_external_edit_driver(conn=None, driver=driver)
    res = await drive(EditRequest(
        candidate=Candidate(name="a1", path=str(tmp_path), branch="selfedit/a1",
                            base_ref="HEAD", base_sha="abc"),
        objective="add helper", attempt_id="a1", user_id="u1"))
    assert res.ok is True and res.final_text == "added h"     # local model, no token
    assert (tmp_path / "h.py").read_text() == "x"


# --- the live adapter's pure normalizer ------------------------------------

def test_normalize_openai_tool_calls():
    calls = _normalize_calls([
        {"id": "c1", "function": {"name": "write_file", "arguments": '{"path":"a.py","content":"z"}'}},
        {"id": "c2", "function": {"name": "finish", "arguments": {"summary": "ok"}}},
    ])
    assert calls[0] == {"name": "write_file", "args": {"path": "a.py", "content": "z"}, "id": "c1"}
    assert calls[1]["name"] == "finish" and calls[1]["args"] == {"summary": "ok"}


# --- W1/W2/W4: per-write syntax feedback, safe auto-fix, stagnation ---------

from augmentum.selfedit.native_loop import _autofix, _syntax_check  # noqa: E402


def test_syntax_check_locates_python_break():
    err = _syntax_check("server.py", "def f():\n  x = 1\n    y = 2\n")
    assert "line" in err and ("Indent" in err or "indent" in err)


def test_syntax_check_clean_python_is_empty():
    assert _syntax_check("m.py", "x = 1\ndef f():\n    return x\n") == ""


def test_syntax_check_json():
    assert "JSON" in _syntax_check("c.json", "{bad}")
    assert _syntax_check("c.json", '{"ok": 1}') == ""


def test_autofix_tabs_and_trailing_space():
    fixed, note = _autofix("m.py", "def f():\n\treturn 1   \n")
    assert "\t" not in fixed and fixed == "def f():\n    return 1\n"
    assert "tabs" in note and "trailing" in note


def test_autofix_leaves_clean_code_untouched():
    src = "x = 1\ndef f():\n    return x\n"
    fixed, note = _autofix("m.py", src)
    assert fixed == src and note == ""


async def test_write_file_flags_syntax_error_immediately(tmp_path):
    # the server.py-stall scenario in miniature: a broken write is flagged AT
    # WRITE TIME, located, so the model can fix it in the same loop.
    from augmentum.selfedit.native_loop import _write_file
    broken = "def f():\n  a = 1\n    b = 2\n"
    res = await _write_file(str(tmp_path), {"path": "b.py", "content": broken})
    assert "SYNTAX ERROR" in res and "line" in res


async def test_edit_file_flags_break_it_introduces(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    from augmentum.selfedit.native_loop import _edit_file
    # insert a mis-indented line
    res = await _edit_file(str(tmp_path), {
        "path": "m.py", "old_string": "    return 1",
        "new_string": "    return 1\n  broken_indent = 2"})
    assert "SYNTAX ERROR" in res


async def test_stagnation_breaker_halts_repeated_calls(tmp_path):
    (tmp_path / "f.py").write_text("x = 1")
    # model repeats the SAME read_file call forever
    same = _call("read_file", {"path": "f.py"})
    chat = _scripted([{"content": "", "tool_calls": [same]} for _ in range(6)])
    evs = await _run(make_native_loop(chat=chat, max_iters=10), str(tmp_path))
    assert evs[-1]["kind"] == "completed"
    assert "repeated the same" in evs[-1]["text"]


# --- transparency: tool results, edit content, and reasoning are EMITTED ----

async def test_tool_result_event_is_emitted(tmp_path):
    # the loss we fixed: the tool's RETURN (incl. W1 syntax feedback) is now an
    # event, not only appended to the model's private context.
    chat = _scripted([
        {"content": "", "tool_calls": [_call("write_file", {"path": "b.py", "content": "def f(:\n"})]},
        {"content": "done", "tool_calls": [_call("finish", {"summary": "x"})]},
    ])
    evs = await _run(make_native_loop(chat=chat), str(tmp_path))
    results = [e for e in evs if e["kind"] == "tool_result"]
    assert results, "no tool_result event emitted"
    assert "SYNTAX ERROR" in results[0]["text"]      # the W1 feedback is now visible
    assert results[0]["tool"] == "write_file" and results[0]["path"] == "b.py"


async def test_reasoning_is_emitted_as_thinking(tmp_path):
    async def chat(_m, _s):
        # a thinking model returns reasoning alongside the tool call
        return {"content": "", "reasoning": "I should create x.py because …",
                "tool_calls": [_call("finish", {"summary": "done"})]}
    evs = await _run(make_native_loop(chat=chat), str(tmp_path))
    thinking = [e for e in evs if e["kind"] == "thinking"]
    assert thinking and "because" in thinking[0]["text"]


async def test_no_reasoning_no_thinking_event(tmp_path):
    # models that don't emit reasoning shouldn't produce empty thinking noise
    chat = _scripted([{"content": "hi", "tool_calls": [_call("finish", {"summary": "d"})]}])
    evs = await _run(make_native_loop(chat=chat), str(tmp_path))
    assert not any(e["kind"] == "thinking" for e in evs)
