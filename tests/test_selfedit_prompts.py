"""Overridable prompt registry tests — the bridge that lets Evolve change the app.

Locks: a registered prompt resolves to the override when set (else the default),
the override layer never breaks a prompt site (store hiccup → default), and only
registered keys are overridable. Also that the debt advisor honors an override
without breaking on stray braces in an evolved prompt.
"""

from __future__ import annotations

import json

import pytest

from augmentum.selfedit import prompts


class _FakeStore:
    def __init__(self, data=None):
        self.data = data or {}

    async def get_user(self, user_id, key):
        return self.data.get((user_id, key))


@pytest.fixture(autouse=True)
def _clean():
    prompts.clear_registry()
    yield
    prompts.clear_registry()


def test_register_and_key_namespacing():
    spec = prompts.register_prompt("demo", "DEFAULT TEXT", label="Demo", description="d")
    assert spec.key == prompts.KEY_PREFIX + "demo"
    assert prompts.get_spec("demo").default == "DEFAULT TEXT"
    assert prompts.spec_for_key(spec.key) is spec
    assert spec.to_dict()["overridden"] is False


async def test_resolves_override_then_default():
    spec = prompts.register_prompt("demo", "DEFAULT")
    store = _FakeStore({("u1", spec.key): "EVOLVED"})
    assert await prompts.resolved_prompt("demo", settings_store=store, user_id="u1") == "EVOLVED"
    # a different user has no override → the default
    assert await prompts.resolved_prompt("demo", settings_store=store, user_id="u2") == "DEFAULT"
    # empty override falls back to the default (so revert == set empty)
    store.data[("u1", spec.key)] = ""
    assert await prompts.resolved_prompt("demo", settings_store=store, user_id="u1") == "DEFAULT"


async def test_override_layer_never_breaks_a_site():
    prompts.register_prompt("demo", "DEFAULT")

    class _Boom:
        async def get_user(self, *_):
            raise RuntimeError("store down")
    # a store failure must not raise — the site still gets the default
    assert await prompts.resolved_prompt("demo", settings_store=_Boom(), user_id="u1") == "DEFAULT"
    # no store / no user → default
    assert await prompts.resolved_prompt("demo", settings_store=None, user_id="") == "DEFAULT"


async def test_unregistered_slug_uses_passed_default():
    assert await prompts.resolved_prompt("nope", settings_store=None, user_id="u1",
                                         default="FALLBACK") == "FALLBACK"


async def test_debt_advisor_honors_override_without_brace_crash():
    # an evolved advisor prompt with stray braces + no {findings} slot must still
    # run (findings appended) and not raise on .format-style braces.
    from augmentum.selfedit import debt
    from augmentum.selfedit.debt_advisor import advise
    from augmentum.selfedit.scanners import parse_audit_json
    triage = debt.triage(parse_audit_json(json.dumps(
        {"score": 80.0, "metrics": {"code_quality": {"silent_catches": 3}}})))

    seen = {}

    async def chat(prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps({"summary": "s", "recommendations": [
            {"id": "code_quality.silent_catches", "rationale": "x", "approach": "y"}]})

    evolved = "Rank the findings. Use your judgment {with braces}. Return JSON."  # no {findings}
    adv = await advise(triage, chat=chat, prompt=evolved)
    assert adv.available is True and len(adv.recommendations) == 1
    assert "FINDINGS:" in seen["prompt"] and "silent_catches" in seen["prompt"]  # appended
