"""Grounded synthesis + the oracle gate — fixes the live-run failure modes.

Load-bearing:
  - scaffolded synthesis renders the CORRECT harness (async dispatch, real
    imports) around the model's id/args/assertions, so mechanics can't be wrong;
  - the oracle gate classifies a synthesized test valid / vacuous / broken by how
    it fails against the unbuilt code (only 'valid' = right reason = trustworthy).
"""

from __future__ import annotations

import json

from augmentum.selfedit.capabilities import (
    oracle_verdict,
    render_scaffolded_test,
    syn_module_stem_for,
    synthesize_verb_acceptance,
)

_GOOD_OBJ = {
    "id": "unit.c_to_f",
    "args": {"celsius": 100},
    "assertions": ['"212" in res.speak'],
    "summary": "convert Celsius to Fahrenheit and speak it",
}


# --- scaffolded synthesis: mechanics correct by construction ---------------

def test_render_scaffold_has_correct_mechanics():
    rel, src = render_scaffolded_test(_GOOD_OBJ)
    assert rel == "tests/test_authored_unit_c_to_f.py"
    compile(src, "<scaffold>", "exec")                     # valid Python
    # the harness the model kept getting wrong, now ours + correct:
    assert "import asyncio" in src
    assert "from augmentum.intent import REGISTRY" in src
    assert "from augmentum.intent.action import SessionContext" in src
    assert "asyncio.run(action.handler(" in src            # async, via .handler
    assert "REGISTRY.get('unit.c_to_f')" in src
    # boot-path verification: no DIRECT module import (that would mask a skipped
    # intent/__init__ registration line — the verb would pass yet be dead at boot).
    assert "import augmentum.intent.builtin." not in src
    assert "from augmentum.intent import REGISTRY" in src   # runs __init__ = boot wiring
    assert 'assert "212" in res.speak' in src              # model's behavior, our assert


async def test_synthesize_verb_acceptance_happy():
    async def mi(_p: str) -> str:
        return f"```json\n{json.dumps(_GOOD_OBJ)}\n```"
    rel, src = await synthesize_verb_acceptance("c to f converter", model_invoke=mi)
    assert rel is not None and "asyncio.run(action.handler(" in src


async def test_synthesize_repairs_bad_then_good():
    calls: list[str] = []

    async def mi(p: str) -> str:
        calls.append(p)
        bad = json.dumps({"id": "BadId", "args": {}, "assertions": []})
        return bad if len(calls) == 1 else json.dumps(_GOOD_OBJ)

    rel, src = await synthesize_verb_acceptance("x", model_invoke=mi)
    assert rel is not None and len(calls) == 2 and "invalid" in calls[1].lower()


async def test_synthesize_rejects_assertion_without_res():
    async def mi(_p: str) -> str:                          # assertion ignores `res` (always-true junk)
        return json.dumps({"id": "x.y", "args": {}, "assertions": ["1 == 1"]})
    rel, err = await synthesize_verb_acceptance("x", model_invoke=mi)
    assert rel is None and "res" in err


def test_module_stem_matches_scaffold_import():
    assert syn_module_stem_for("unit.c_to_f") == "syn_unit_c_to_f"


# --- the oracle gate: valid / vacuous / broken -----------------------------

async def test_oracle_valid_when_impl_module_missing():
    async def runner(_src: str):
        return False, "E   ModuleNotFoundError: No module named 'augmentum.intent.builtin.syn_unit_c_to_f'"
    verdict, _ = await oracle_verdict(
        "...", expected_missing_module="augmentum.intent.builtin.syn_unit_c_to_f", run_pytest=runner,
    )
    assert verdict == "valid"


async def test_oracle_valid_on_registration_assert():
    async def runner(_src: str):
        return False, "AssertionError: verb unit.c_to_f must be registered"
    verdict, _ = await oracle_verdict("...", run_pytest=runner)
    assert verdict == "valid"


async def test_oracle_vacuous_when_passes_already():
    async def runner(_src: str):
        return True, "1 passed"
    verdict, detail = await oracle_verdict("...", run_pytest=runner)
    assert verdict == "vacuous" and "nothing real" in detail


async def test_oracle_broken_on_wrong_import():
    # the exact Case-B failure: importing a real-looking module that doesn't exist
    async def runner(_src: str):
        return False, "E   ModuleNotFoundError: No module named 'augmentum.models'"
    verdict, detail = await oracle_verdict(
        "...", expected_missing_module="augmentum.intent.builtin.syn_x", run_pytest=runner,
    )
    assert verdict == "broken" and "wrong reason" in detail
