"""Test-first capability authoring — the SOTA self-evolution seam.

Load-bearing:
  - synthesize_acceptance_test turns a request into a valid pytest oracle
    (strips fences, repairs once, rejects junk);
  - acceptance_test_verifier injects OUR canonical test into the candidate at
    verify-time (un-gameable) and runs it as a mechanical confirm oracle;
  - author_capability composes synth → objective → the live engine's
    run_self_edit with the authored test plugged in as extra_verifiers.
"""

from __future__ import annotations

from augmentum.selfedit.capabilities import (
    acceptance_test_verifier,
    author_capability,
    build_authoring_objective,
    synthesize_acceptance_test,
)
from augmentum.selfedit.verifier import ORACLE_MECHANICAL, Verifier, VerifierResult

_GOOD_TEST = (
    "from augmentum.intent import REGISTRY\n"
    "def test_verb_registers():\n"
    "    assert REGISTRY.get('x.y') is not None\n"
)


# --- synthesize the acceptance test ---------------------------------------

async def test_synthesize_happy_path():
    async def mi(_p: str) -> str:
        return _GOOD_TEST
    rel, src = await synthesize_acceptance_test("a verb that does X", model_invoke=mi)
    assert rel is not None
    assert rel.startswith("tests/test_authored_") and rel.endswith(".py")
    assert "def test_" in src


async def test_synthesize_strips_code_fence():
    async def mi(_p: str) -> str:
        return f"```python\n{_GOOD_TEST}```"
    rel, src = await synthesize_acceptance_test("x", model_invoke=mi)
    assert rel is not None and not src.startswith("```") and "def test_" in src


async def test_synthesize_repairs_then_succeeds():
    calls: list[str] = []

    async def mi(prompt: str) -> str:
        calls.append(prompt)
        return "def nope(): pass" if len(calls) == 1 else _GOOD_TEST  # 1st has no test_

    rel, src = await synthesize_acceptance_test("x", model_invoke=mi)
    assert rel is not None and "def test_" in src
    assert len(calls) == 2 and "not usable" in calls[1]


async def test_synthesize_rejects_syntax_error():
    async def mi(_p: str) -> str:
        return "def test_x(:\n  pass"   # broken twice
    rel, err = await synthesize_acceptance_test("x", model_invoke=mi)
    assert rel is None and "syntax" in err.lower()


# --- the acceptance-test verifier (un-gameable oracle injection) -----------

async def test_acceptance_verifier_injects_and_runs(tmp_path):
    captured = {}

    def fake_factory(paths, *, cwd):
        captured["paths"] = paths
        captured["cwd"] = cwd

        async def _run(_ctx):
            return VerifierResult("inner", ORACLE_MECHANICAL, "pass", confirms_intent=True)
        return Verifier("inner", ORACLE_MECHANICAL, _run, confirms_intent=True)

    v = acceptance_test_verifier(
        "tests/test_authored_x.py", _GOOD_TEST, pytest_confirm_factory=fake_factory,
    )
    res = await v.run({"candidate_dir": str(tmp_path)})

    # the canonical test was written INTO the candidate (the engine can't weaken it)
    written = tmp_path / "tests" / "test_authored_x.py"
    assert written.read_text() == _GOOD_TEST
    # delegated to the engine's pytest confirm verifier with our path + candidate cwd
    assert captured["paths"] == ["tests/test_authored_x.py"]
    assert captured["cwd"] == str(tmp_path)
    # a passing run confirms the intent (mechanical → VERIFIED upstream)
    assert res.status == "pass" and res.confirms_intent is True
    assert res.name == "acceptance_test"


async def test_acceptance_verifier_fails_on_unwritable_dir():
    # no candidate_dir that exists + a path that can't be made → fail, not crash
    def fake_factory(paths, *, cwd):
        async def _run(_ctx):
            return VerifierResult("inner", ORACLE_MECHANICAL, "pass", confirms_intent=True)
        return Verifier("inner", ORACLE_MECHANICAL, _run, confirms_intent=True)

    v = acceptance_test_verifier(
        "tests/t.py", _GOOD_TEST, pytest_confirm_factory=fake_factory,
    )
    res = await v.run({"candidate_dir": "\x00/bad"})  # NUL byte → os write fails
    assert res.status == "fail" and "inject" in res.detail


# --- objective ------------------------------------------------------------

def test_objective_carries_test_and_no_edit_rule():
    obj = build_authoring_objective("do X", "tests/test_authored_x.py", _GOOD_TEST)
    assert "tests/test_authored_x.py" in obj
    assert "def test_verb_registers" in obj          # the test is shown as the contract
    assert "Do NOT edit the acceptance test" in obj


# --- end-to-end composition (injected engine) -----------------------------

async def test_author_capability_drives_engine_with_oracle():
    seen = {}

    async def mi(_p: str) -> str:
        return _GOOD_TEST

    async def fake_run(*, repo_dir, objective, user_id, conn, driver, extra_verifiers, **kw):
        seen["objective"] = objective
        seen["verifiers"] = extra_verifiers
        seen["repo_dir"] = repo_dir
        return {"status": "gated"}   # sentinel outcome

    outcome, err = await author_capability(
        "a verb that does X", repo_dir="/repo", conn=None, driver=object(),
        model_invoke=mi, run=fake_run,
    )
    assert err == "" and outcome == {"status": "gated"}
    assert "def test_verb_registers" in seen["objective"]      # contract handed over
    assert len(seen["verifiers"]) == 1
    assert seen["verifiers"][0].name == "acceptance_test"      # oracle plugged in
    assert seen["verifiers"][0].confirms_intent is True


async def test_author_capability_aborts_when_test_synthesis_fails():
    called = {"run": False}

    async def mi(_p: str) -> str:
        return "garbage, no test"

    async def fake_run(**_kw):
        called["run"] = True
        return {"status": "gated"}

    outcome, err = await author_capability(
        "x", repo_dir="/repo", conn=None, driver=object(), model_invoke=mi, run=fake_run,
    )
    assert outcome is None and err
    assert called["run"] is False    # never drove the engine on an un-authorable request
