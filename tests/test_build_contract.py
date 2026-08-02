"""Unit tests for the behavior contract + gate pure helpers."""

from __future__ import annotations

import pytest

from augmentum.builds import verify as bv
from augmentum.builds.contract import (
    extract_json_object,
    normalize_behaviors,
    render_behaviors_for_build,
    render_failures_for_fix,
)
from augmentum.builds.verify import (
    build_gate_payload,
    gate_summary,
    parse_gate_output,
)


def test_extract_json_object_tolerates_fences_and_prose():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('thinking...\n{"behaviors": []} trailing') == {"behaviors": []}
    assert extract_json_object("not json") == {}
    assert extract_json_object("") == {}


def test_normalize_behaviors_shapes_dedupes_and_caps():
    raw = {"behaviors": [
        {"id": "Tip Basic", "description": "shows tip"},
        {"description": "shows tip"},          # id derived from desc -> dup -> suffixed
        "clear resets everything",              # bare string
        {"description": ""},                    # dropped (empty)
        42,                                      # dropped (junk)
    ]}
    out = normalize_behaviors(raw)
    assert [b["id"] for b in out] == ["tip-basic", "shows-tip", "clear-resets-everything"]
    assert all(b["status"] == "untested" and b["evidence"] == "" for b in out)


def test_normalize_behaviors_caps_at_ten():
    raw = {"behaviors": [{"description": f"behavior {i}"} for i in range(20)]}
    assert len(normalize_behaviors(raw)) == 10


def test_render_blocks():
    behaviors = [
        {"id": "a", "description": "does X", "status": "pass", "evidence": "ok"},
        {"id": "b", "description": "does Y", "status": "fail", "evidence": "got NaN"},
    ]
    build_block = render_behaviors_for_build(behaviors)
    assert "does X" in build_block and "does Y" in build_block
    fix_block = render_failures_for_fix(behaviors)
    assert "does Y" in fix_block and "got NaN" in fix_block
    assert "does X" not in fix_block  # only failures
    assert render_failures_for_fix([{"id": "a", "status": "pass"}]) == ""


def test_build_gate_payload_shape():
    payload = build_gate_payload({"a": {"steps": [], "assert": "true"}}, port=9000)
    assert payload["url"] == "http://localhost:9000"
    assert payload["assertions"] == [{"id": "a", "steps": [], "assert": "true"}]


def test_parse_gate_output():
    ok = parse_gate_output('{"results":[{"id":"a","passed":true,"evidence":"ok"}]}')
    assert ok["a"]["passed"] is True
    # fatal -> None so caller falls back
    assert parse_gate_output('{"fatal":"playwright unavailable: x"}') is None
    assert parse_gate_output("garbage") is None


def test_gate_summary():
    behaviors = [
        {"id": "a", "status": "pass"},
        {"id": "b", "status": "fail"},
        {"id": "c", "status": "untested"},
    ]
    s = gate_summary(behaviors)
    assert s == {
        "total": 3, "checked": 2, "passed": 1, "failed": 1,
        "all_passed": False, "failed_ids": ["b"],
    }
    # all checked passing -> all_passed True
    s2 = gate_summary([{"id": "a", "status": "pass"}, {"id": "b", "status": "pass"}])
    assert s2["all_passed"] is True
    # nothing checked -> not all_passed (can't claim verified)
    assert gate_summary([{"id": "a", "status": "untested"}])["all_passed"] is False


# ---------------------------------------------------------------------------
# Sidecar gate runner (2026-07-17 — behavior gate rides the browser sidecar)
# ---------------------------------------------------------------------------


class _GateCM:
    """Just enough container_manager for _run_gate_sidecar."""

    _docker = object()


def _payload():
    return {
        "url": "http://localhost:8129",
        "assertions": [
            {"id": "b1", "steps": [{"action": "fill", "selector": "#x", "value": "5"},
                                   {"action": "click", "selector": "#go"}],
             "assert": "document.querySelector('#out').textContent.includes('5')"},
            {"id": "b2", "steps": [], "assert": "true"},
        ],
    }


def _patch_sidecar(monkeypatch, run_cli, available=True):
    from augmentum.coder import browser_sidecar as bs

    async def _avail(_docker):
        return available

    async def _reach(_cm, _ws, url):
        return url.replace("localhost", "172.28.0.9")

    monkeypatch.setattr(bs, "is_available", _avail)
    monkeypatch.setattr(bs, "reachable_url", _reach)
    monkeypatch.setattr(bs, "run_cli", run_cli)


@pytest.mark.asyncio
async def test_sidecar_gate_fresh_page_per_behavior(monkeypatch):
    calls = []

    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        calls.append((session, list(args)))
        if args[0] == "eval":
            return {"ok": True, "sidecar": True, "result": True}
        if args[0] == "console" and len(args) == 1:
            return {"ok": True, "sidecar": True, "messages": []}
        return {"ok": True, "sidecar": True}

    _patch_sidecar(monkeypatch, _run)
    results = await bv._run_gate_sidecar(_GateCM(), "ws1", _payload())
    assert results["b1"]["passed"] and results["b2"]["passed"]
    opens = [a for _s, a in calls if a[0] == "open"]
    # One fresh open per behavior — deliberate state reset between behaviors.
    assert len(opens) == 2
    assert opens[0][1] == "http://172.28.0.9:8129"
    # Dedicated gate session, never the agent's ws- session.
    assert all(s.startswith("gate-") for s, _a in calls)
    # Session closed at the end.
    assert calls[-1][1] == ["close"]


@pytest.mark.asyncio
async def test_sidecar_gate_step_failure_is_evidence_not_fatal(monkeypatch):
    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "click":
            return {"ok": False, "sidecar": True, "error": "no such element"}
        if args[0] == "eval":
            return {"ok": True, "sidecar": True, "result": True}
        if args[0] == "console" and len(args) == 1:
            return {"ok": True, "sidecar": True, "messages": []}
        return {"ok": True, "sidecar": True}

    _patch_sidecar(monkeypatch, _run)
    results = await bv._run_gate_sidecar(_GateCM(), "ws1", _payload())
    assert results["b1"]["passed"] is False
    assert "step failed (click #go)" in results["b1"]["evidence"]
    assert results["b2"]["passed"] is True


@pytest.mark.asyncio
async def test_sidecar_gate_assert_js_error_surfaces(monkeypatch):
    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "eval":
            return {"ok": True, "sidecar": True, "result": {"__e": "ReferenceError: x"}}
        if args[0] == "console" and len(args) == 1:
            return {"ok": True, "sidecar": True, "messages": []}
        return {"ok": True, "sidecar": True}

    _patch_sidecar(monkeypatch, _run)
    results = await bv._run_gate_sidecar(_GateCM(), "ws1", _payload())
    assert results["b2"]["passed"] is False
    assert "assert error: ReferenceError" in results["b2"]["evidence"]


@pytest.mark.asyncio
async def test_sidecar_gate_unavailable_returns_none(monkeypatch):
    async def _run(*a, **k):  # pragma: no cover — should not be called
        raise AssertionError("run_cli must not run when sidecar is down")

    _patch_sidecar(monkeypatch, _run, available=False)
    assert await bv._run_gate_sidecar(_GateCM(), "ws1", _payload()) is None


@pytest.mark.asyncio
async def test_sidecar_gate_midrun_death_falls_back_whole_gate(monkeypatch):
    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "open":
            return {"ok": False, "sidecar": False, "error": "gone"}
        return {"ok": True, "sidecar": True}

    _patch_sidecar(monkeypatch, _run)
    assert await bv._run_gate_sidecar(_GateCM(), "ws1", _payload()) is None


@pytest.mark.asyncio
async def test_run_behavior_gate_ladder(monkeypatch):
    """Sidecar results skip the legacy script entirely; sidecar-None stages
    and runs the in-container Playwright script."""

    class _FullCM:
        def __init__(self):
            self.writes = []
            self.commands = []

        async def run_command(self, ws, cmd, timeout=30.0):
            self.commands.append(cmd[-1])
            if "gate.py" in cmd[-1]:
                return ('{"results":[{"id":"b1","passed":true,"evidence":"ok"}]}'
                        "\n__AUGMENTUM_GATE_EXIT__:0")
            return "\n__AUGMENTUM_GATE_EXIT__:0"

        async def file_write(self, ws, path, content):
            self.writes.append(path)

    async def _compile(_backend, *, model, behaviors, source):
        return {"b1": {"steps": [], "assert": "true"}}

    monkeypatch.setattr(bv, "compile_assertions", _compile)
    behaviors = [{"id": "b1", "description": "works"}]

    # Rung 1: sidecar answers — legacy never staged.
    async def _sidecar_ok(cm, ws, payload):
        return {"b1": {"id": "b1", "passed": True, "evidence": "ok"}}

    monkeypatch.setattr(bv, "_run_gate_sidecar", _sidecar_ok)
    cm = _FullCM()
    merged, ran = await bv.run_behavior_gate(
        container_manager=cm, workspace_id="w", backend=None, model="m",
        behaviors=behaviors)
    assert ran and merged[0]["status"] == "pass"
    assert cm.writes == []

    # Rung 2: sidecar unavailable — legacy script staged and executed.
    async def _sidecar_none(cm, ws, payload):
        return None

    monkeypatch.setattr(bv, "_run_gate_sidecar", _sidecar_none)
    cm = _FullCM()
    merged, ran = await bv.run_behavior_gate(
        container_manager=cm, workspace_id="w", backend=None, model="m",
        behaviors=behaviors)
    assert ran and merged[0]["status"] == "pass"
    assert any("gate.py" in p for p in cm.writes)
    assert any("gate.py" in c for c in cm.commands)
