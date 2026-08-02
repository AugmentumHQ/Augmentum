"""Native-loop guardrails — the scanner write-lockout (can't edit the judge) and
the step-budget nudge (commit before the cap, not explore-until-cut-off). Both
land directly from a real run that (a) tried to edit runtime_suppressions.json and
(b) burned 40 turns reading with zero edits on the large workspace."""

from __future__ import annotations

import augmentum.selfedit.native_loop as N
from augmentum.coder.external.base import ExternalTask

# --- scanner write-lockout (integrity) --------------------------------------

def test_is_audit_infra_flags_the_judge():
    assert N._is_audit_infra(".claude/skills/augmentum-dev/scripts/runtime_suppressions.json")
    assert N._is_audit_infra("augmentum/selfedit/.claude/x.py")
    assert N._is_audit_infra("ui/quality_suppressions.json")
    assert N._is_audit_infra("security_exceptions.json")
    # the real code the agent SHOULD edit is not flagged
    assert not N._is_audit_infra("augmentum/proxy/server.py")
    assert not N._is_audit_infra("ui/styles/workshop.css")


async def test_write_file_refuses_audit_infra(tmp_path):
    ws = str(tmp_path)
    out = await N._write_file(ws, {"path": ".claude/skills/augmentum-dev/scripts/runtime_suppressions.json",
                                   "content": "{}"})
    assert "refused" in out.lower() and "judge" in out.lower()
    # ...and it really didn't write
    assert not (tmp_path / ".claude").exists()


async def test_edit_file_refuses_audit_infra(tmp_path):
    ws = str(tmp_path)
    out = await N._edit_file(ws, {"path": "x_suppressions.json", "old_string": "a", "new_string": "b"})
    assert "refused" in out.lower()


async def test_write_file_allows_real_code(tmp_path):
    ws = str(tmp_path)
    out = await N._write_file(ws, {"path": "augmentum/foo.py", "content": "x = 1\n"})
    assert "wrote" in out.lower()
    assert (tmp_path / "augmentum" / "foo.py").read_text() == "x = 1\n"


# --- budget nudge (commit before the cap) -----------------------------------

async def test_budget_nudge_fires_before_cap():
    """Once ~70% of the step budget is spent, the loop injects a 'stop exploring,
    make the edit' message — captured here via a fake chat that records messages."""
    seen_nudge = {"hit": False}
    n = {"i": 0}

    async def fake_chat(messages, specs):
        # a model that only ever reads (never edits, never finishes) — the failure
        # mode. Vary the path each turn so the BUDGET nudge is what engages, not the
        # identical-call stagnation breaker (a separate guard).
        if any("running low" in (m.get("content") or "") for m in messages if m["role"] == "user"):
            seen_nudge["hit"] = True
        n["i"] += 1
        return {"content": "looking...", "tool_calls": [
            {"name": "read_file", "args": {"path": f"seed{n['i']}.py"}, "id": "1"}]}

    loop = N.make_native_loop(chat=fake_chat, max_iters=10)
    events = [ev async for ev in loop(ExternalTask(prompt="fix it", workspace="."))]
    assert seen_nudge["hit"], "budget nudge never fired before the cap"
    assert events[-1]["kind"] == "completed"  # ends on the cap, gracefully


async def test_system_prompt_states_the_budget():
    captured = {}

    async def fake_chat(messages, specs):
        captured["system"] = messages[0]["content"]
        return {"content": "done", "tool_calls": []}   # finish immediately

    loop = N.make_native_loop(chat=fake_chat, max_iters=42)
    _ = [ev async for ev in loop(ExternalTask(prompt="x", workspace="."))]
    assert "42 steps" in captured["system"]
    assert "never edit the scanner" in captured["system"].lower()
