"""Tests for the independent cross-model coder-run verifier.

Protects the honesty contract:
  * No heavyweight pinned / disabled / self-verify → 'unchecked' (never a
    self-graded pass)
  * A different model's positive judgment → 'probable' (never 'verified' —
    that tier is reserved for mechanical test execution)
  * ok=false → 'failed'; needs_human → 'human_required'
  * Backend / parse failure degrades to 'unchecked', never raises
  * The judge is fed the actual diff, not just the driver's answer
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeBackend:
    def __init__(self, reply="", raise_exc=False):
        self._reply = reply
        self._raise = raise_exc
        self.last_prompt = ""

    async def chat(self, req):
        if self._raise:
            raise RuntimeError("backend down")
        self.last_prompt = req.messages[0].content
        return SimpleNamespace(message=SimpleNamespace(content=self._reply, thinking=""))


class _FakeRegistry:
    def __init__(self, backend, resolved="deepseek-v4-pro"):
        self._backend = backend
        self._resolved = resolved

    async def resolve_backend_with_fabric(self, model, *, user_id=""):
        return self._backend, self._resolved


def _bundle(diff="--- a.py ---\n+print('hi')\n"):
    return SimpleNamespace(files=[
        SimpleNamespace(path="a.py", status="modified", unified_diff=diff),
    ])


class _FakeReviewRegistry:
    def __init__(self, bundle):
        self._bundle = bundle

    def get(self, turn_id):
        return self._bundle


def _app(backend=None, *, resolved="deepseek-v4-pro", bundle=None):
    return SimpleNamespace(
        provider_registry=_FakeRegistry(backend, resolved) if backend else None,
        review_registry=_FakeReviewRegistry(bundle if bundle is not None else _bundle()),
    )


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_verify_enabled", True, raising=False)
    monkeypatch.setattr(settings, "heavyweight_model", "deepseek-v4-pro", raising=False)
    return settings


async def _run(app, driver="qwen3.6-35b", prompt="add a greeting", answer="done"):
    from augmentum.coder.run_verifier import verify_coder_run
    return await verify_coder_run(
        app, user_id="u1", review_turn_id="ctr_1",
        prompt=prompt, answer=answer, driver_model=driver,
    )


@pytest.mark.asyncio
async def test_no_heavyweight_is_unchecked(_settings, monkeypatch):
    monkeypatch.setattr(_settings, "heavyweight_model", "", raising=False)
    v = await _run(_app(_FakeBackend('{"ok": true}')))
    assert v.tier == "unchecked"
    assert "heavyweight" in v.reason.lower()


@pytest.mark.asyncio
async def test_disabled_is_unchecked(_settings, monkeypatch):
    monkeypatch.setattr(_settings, "coder_verify_enabled", False, raising=False)
    v = await _run(_app(_FakeBackend('{"ok": true}')))
    assert v.tier == "unchecked"


@pytest.mark.asyncio
async def test_self_verification_is_unchecked():
    # Heavyweight resolves to the SAME model that drove the run.
    backend = _FakeBackend('{"ok": true}')
    app = _app(backend, resolved="qwen3.6-35b")
    v = await _run(app, driver="qwen3.6-35b")
    assert v.tier == "unchecked"
    assert v.self_verified is True
    # It must NOT have even asked the model to grade itself.
    assert backend.last_prompt == ""


@pytest.mark.asyncio
async def test_no_diff_is_unchecked():
    app = _app(_FakeBackend('{"ok": true}'), bundle=SimpleNamespace(files=[]))
    v = await _run(app)
    assert v.tier == "unchecked"


@pytest.mark.asyncio
async def test_positive_judgment_is_probable_not_verified():
    backend = _FakeBackend('{"ok": true, "reason": "greeting added"}')
    v = await _run(_app(backend))
    assert v.tier == "probable"          # never 'verified' — no mechanical proof
    assert v.oracle == "judgment"
    assert v.verifier_model == "deepseek-v4-pro"
    # The judge saw the actual diff, not just the answer.
    assert "print('hi')" in backend.last_prompt


@pytest.mark.asyncio
async def test_negative_judgment_is_failed():
    v = await _run(_app(_FakeBackend('{"ok": false, "reason": "does nothing"}')))
    assert v.tier == "failed"
    assert "does nothing" in v.reason


@pytest.mark.asyncio
async def test_needs_human_is_human_required():
    v = await _run(_app(_FakeBackend('{"ok": true, "needs_human": true, "reason": "autosave or button?"}')))
    assert v.tier == "human_required"


@pytest.mark.asyncio
async def test_garbage_response_is_unchecked():
    v = await _run(_app(_FakeBackend("not json at all")))
    assert v.tier == "unchecked"


@pytest.mark.asyncio
async def test_backend_error_is_unchecked_not_raised():
    v = await _run(_app(_FakeBackend(raise_exc=True)))
    assert v.tier == "unchecked"


@pytest.mark.asyncio
async def test_envelope_shape():
    v = await _run(_app(_FakeBackend('{"ok": true, "reason": "ok"}')))
    env = v.to_envelope()
    assert set(env) == {
        "tier", "oracle", "reason", "verifier_model", "self_verified",
        "contract_unmet",
    }
    assert env["contract_unmet"] == []  # no inbound contract on the fake app


# ── Mechanical tier (re-run the recorded tests) ─────────────────────────────

class _FakeCM:
    def __init__(self, mode="pass"):  # pass | fail | timeout
        self.mode = mode
        self.calls = []

    async def run_command(self, workspace_id, cmd, timeout=None, *, idle_timeout=None, strict=False):
        self.calls.append(cmd)
        from augmentum.coder.containers import ExecAborted
        if self.mode == "fail":
            raise ExecAborted("exit_code", "1 failed")
        if self.mode == "timeout":
            raise ExecAborted("wall_clock", "timed out")
        return "3 passed"


def _app_mech(reply, cm, *, resolved="deepseek-v4-pro"):
    app = _app(_FakeBackend(reply), resolved=resolved)
    app.container_manager = cm
    app.state_manager = SimpleNamespace(backend=SimpleNamespace(conn=SimpleNamespace()))
    return app


def _patch_ledger(monkeypatch, tests):
    class _FakeLedger:
        def __init__(self, conn):
            pass

        async def get_run(self, run_id, *, user_id=""):
            return {"tests_run": list(tests)}
    import augmentum.coder.ledger as ledger
    monkeypatch.setattr(ledger, "CoderTurnLedgerStore", _FakeLedger)


async def _run_mech(app):
    from augmentum.coder.run_verifier import verify_coder_run
    return await verify_coder_run(
        app, user_id="u1", review_turn_id="ctr_1",
        prompt="add a greeting", answer="done", driver_model="qwen3.6-35b",
        run_id="run_1", workspace_id="ws_1",
    )


@pytest.mark.asyncio
async def test_mechanical_pass_plus_judgment_is_verified(monkeypatch):
    _patch_ledger(monkeypatch, ["pytest -x"])
    cm = _FakeCM("pass")
    v = await _run_mech(_app_mech('{"ok": true, "reason": "greeting added"}', cm))
    assert v.tier == "verified"
    assert v.oracle == "mechanical"
    # Re-ran through a login shell so the workspace venv is on PATH.
    assert cm.calls and cm.calls[0][:2] == ["bash", "-lc"]


@pytest.mark.asyncio
async def test_mechanical_fail_overrides_positive_judgment(monkeypatch):
    _patch_ledger(monkeypatch, ["pytest -x"])
    v = await _run_mech(_app_mech('{"ok": true, "reason": "looks fine"}', _FakeCM("fail")))
    assert v.tier == "failed"
    assert v.oracle == "mechanical"


@pytest.mark.asyncio
async def test_mechanical_timeout_is_inconclusive_keeps_judgment(monkeypatch):
    _patch_ledger(monkeypatch, ["pytest -x"])
    v = await _run_mech(_app_mech('{"ok": true, "reason": "greeting added"}', _FakeCM("timeout")))
    assert v.tier == "probable"          # judgment preserved
    assert "couldn't re-run" in v.reason


@pytest.mark.asyncio
async def test_tests_pass_but_reviewer_failed_is_human_required(monkeypatch):
    _patch_ledger(monkeypatch, ["pytest -x"])
    v = await _run_mech(_app_mech('{"ok": false, "reason": "misses the edge case"}', _FakeCM("pass")))
    assert v.tier == "human_required"    # disagreement, never laundered to verified


@pytest.mark.asyncio
async def test_no_recorded_tests_stays_probable(monkeypatch):
    _patch_ledger(monkeypatch, [])       # driver ran no tests
    v = await _run_mech(_app_mech('{"ok": true, "reason": "greeting added"}', _FakeCM("pass")))
    assert v.tier == "probable"
    assert v.oracle == "judgment"


# ── Inbound-contract gate (P3) ──────────────────────────────────────────────

class _FakeState:
    def __init__(self, mission=(), contract=None):
        self.mission = [SimpleNamespace(description=d) for d in mission]
        self.pending_objective_contract = contract or {}


class _FakeStateManager:
    def __init__(self, state):
        self._state = state

    async def load_coder_state(self, session_id, *, user_id=""):
        return self._state


def _app_contract(reply, state, *, resolved="deepseek-v4-pro"):
    app = _app(_FakeBackend(reply), resolved=resolved)
    app.state_manager = _FakeStateManager(state)
    return app


async def _run_contract(app, workspace_id="ws_1"):
    from augmentum.coder.run_verifier import verify_coder_run
    return await verify_coder_run(
        app, user_id="u1", review_turn_id="ctr_1",
        prompt="add a greeting", answer="done", driver_model="qwen3.6-35b",
        workspace_id=workspace_id,
    )


@pytest.mark.asyncio
async def test_no_inbound_contract_behaves_as_before():
    # Empty mission + no contract → judge sees no contract, verdict unchanged.
    app = _app_contract('{"ok": true, "reason": "ok"}', _FakeState())
    v = await _run_contract(app)
    assert v.tier == "probable"
    assert v.contract_unmet == ()


@pytest.mark.asyncio
async def test_contract_unmet_forces_human_required_even_if_ok():
    # Judge says ok=true but lists an unmet load-bearing item → not laundered.
    app = _app_contract(
        '{"ok": true, "unmet": ["remote URL must be verified"], "reason": "mostly"}',
        _FakeState(mission=["ship the endpoint"],
                   contract={"must_prove": ["remote URL must be verified"]}),
    )
    v = await _run_contract(app)
    assert v.tier == "human_required"
    assert "remote URL must be verified" in v.contract_unmet


@pytest.mark.asyncio
async def test_contract_fed_to_judge_prompt():
    backend = _FakeBackend('{"ok": true, "unmet": [], "reason": "all met"}')
    app = _app(backend)
    app.state_manager = _FakeStateManager(_FakeState(mission=["build the parser"]))
    v = await _run_contract(app)
    assert v.tier == "probable"
    assert "build the parser" in backend.last_prompt
    assert "inbound_contract" in backend.last_prompt


@pytest.mark.asyncio
async def test_contract_unmet_with_ok_false_is_failed():
    app = _app_contract(
        '{"ok": false, "unmet": ["tests must pass"], "reason": "no tests"}',
        _FakeState(mission=["tests must pass"]),
    )
    v = await _run_contract(app)
    assert v.tier == "failed"
    assert "tests must pass" in v.contract_unmet


# ── Durable persistence ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_persist_roundtrip_and_user_scope():
    import aiosqlite
    from augmentum.coder.run_verifier import (
        ORACLE_MECHANICAL, TIER_VERIFIED, RunVerdict, load_verdict, save_verdict,
    )
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE coder_run_verifications (run_id TEXT PRIMARY KEY, "
            "user_id TEXT, workspace_id TEXT, tier TEXT, oracle TEXT, reason TEXT, "
            "verifier_model TEXT, self_verified INTEGER)",
        )
        await conn.commit()
        v = RunVerdict(TIER_VERIFIED, ORACLE_MECHANICAL, "tests passed", "deepseek-v4-pro", False)
        await save_verdict(conn, run_id="run_1", user_id="u1", workspace_id="ws_1", verdict=v)
        loaded = await load_verdict(conn, run_id="run_1", user_id="u1")
        assert loaded["tier"] == "verified"
        assert loaded["oracle"] == "mechanical"
        assert loaded["verifier_model"] == "deepseek-v4-pro"
        # Another user cannot read it.
        assert await load_verdict(conn, run_id="run_1", user_id="stranger") is None
    finally:
        await conn.close()
