"""Capability synthesis — spec validation, deterministic render, LLM synth.

Load-bearing:
  - validate_spec enforces the closed-world contract (id shape, safe behavior +
    stakes palette, arg types, required⊆declared);
  - the rendered module is REAL: exec'd against the live registry it actually
    registers the verb and its handler dispatches as specified;
  - the rendered smoke test is valid Python (it's the acceptance oracle);
  - synthesize is closed-world: valid JSON → spec, invalid → one repair, an
    "unsupported" ask → no spec (falls back to human authoring).
"""

from __future__ import annotations

import json

import pytest

from augmentum.selfedit.capabilities import (
    CapabilitySpec,
    build_capability_objective,
    render_verb_module,
    render_verb_test,
    synthesize_capability_spec,
    validate_spec,
)


def _surface_spec(verb_id: str = "navigate.open_browse") -> CapabilitySpec:
    return CapabilitySpec(
        id=verb_id,
        summary="Open the Browse surface.",
        examples=["open the browse panel", "take me to browse"],
        behavior="surface_emit",
        channel="navigate.open_surface",
        payload={"surface": "browse"},   # a real navigate target (router_catalog)
        toast="Opening Browse",
    )


def _speak_spec(verb_id: str = "fun.coin_flip_quip") -> CapabilitySpec:
    return CapabilitySpec(
        id=verb_id,
        summary="Say a short coin-flip quip.",
        examples=["give me a coin flip quip"],
        behavior="speak",
        speak="Heads you win, tails you learn.",
    )


# --- validation ------------------------------------------------------------

def test_validate_accepts_good_specs():
    assert validate_spec(_surface_spec()) == []
    assert validate_spec(_speak_spec()) == []


def test_validate_rejects_bad_id():
    errs = validate_spec(_surface_spec("BadId"))
    assert any("surface.action" in e for e in errs)


def test_validate_rejects_unknown_behavior():
    s = _surface_spec()
    s.behavior = "shell_exec"
    assert any("safe palette" in e for e in validate_spec(s))


def test_validate_rejects_unsafe_stakes():
    s = _surface_spec()
    s.stakes = "irrevocable"
    assert any("synthesizable" in e for e in validate_spec(s))


def test_validate_rejects_bad_arg_and_dangling_required():
    s = _surface_spec()
    s.arg_schema = {"city": {"type": "place"}}      # bad type
    s.required = ["nope"]                            # not declared
    errs = validate_spec(s)
    assert any("type" in e for e in errs)
    assert any("required arg 'nope'" in e for e in errs)


def test_validate_surface_emit_needs_channel():
    s = _surface_spec()
    s.channel = ""
    assert any("channel" in e for e in validate_spec(s))


# --- deterministic render (the generated code is REAL) ---------------------

def test_rendered_surface_module_registers_and_dispatches():
    import asyncio

    from augmentum.intent import REGISTRY
    from augmentum.intent.action import SessionContext

    spec = _surface_spec("navigate.open_testlab")  # unique id; additive to REGISTRY
    src = render_verb_module(spec)
    compile(src, "<syn-module>", "exec")            # valid Python (syntax)
    exec(src, {})                                    # runs register_action live

    action = REGISTRY.get(spec.id)
    assert action is not None and action.id == spec.id
    assert action.stakes == "trivial_reversible"
    ctx = SessionContext(user_id="u", session_id="s")
    res = asyncio.run(action.handler("", ctx, {"surface": "workshop"}))
    assert res is not None and res.short_circuit is True
    assert res.surface_emit["channel"] == "navigate.open_surface"
    # user args flow through into the payload alongside the static slot
    assert res.surface_emit["payload"]["surface"] == "workshop"


def test_rendered_speak_module_registers_and_dispatches():
    import asyncio

    from augmentum.intent import REGISTRY
    from augmentum.intent.action import SessionContext

    spec = _speak_spec("fun.testquip")
    exec(render_verb_module(spec), {})
    action = REGISTRY.get(spec.id)
    assert action is not None
    res = asyncio.run(action.handler("", SessionContext(user_id="u", session_id="s"), {}))
    assert res.short_circuit is True and res.speak == spec.speak


def test_rendered_test_is_valid_python():
    # the smoke test is the acceptance oracle — it must at least be valid Python
    compile(render_verb_test(_surface_spec()), "<syn-test>", "exec")
    compile(render_verb_test(_speak_spec()), "<syn-test>", "exec")


def test_smoke_test_verifies_via_boot_path_not_direct_import():
    # The oracle must catch a skipped intent/__init__ registration (OpenRoom Step
    # 2.5: "without registration, actions won't route"). It verifies through the
    # boot path (REGISTRY.get after the package import), NOT a direct module import
    # that would register the verb regardless and mask the missing wiring.
    spec = _surface_spec("navigate.open_unwired_probe")  # never wired into __init__
    src = render_verb_test(spec)
    assert "import augmentum.intent.builtin" not in src   # no direct-import bypass
    assert "from augmentum.intent import REGISTRY" in src

    # A verb that isn't wired into the boot path is NOT registered → its smoke test
    # MUST fail. (Before the fix, the direct module import made this pass anyway.)
    ns: dict = {}
    exec(compile(src, "<syn-test>", "exec"), ns)
    with pytest.raises(AssertionError):
        ns["test_navigate_open_unwired_probe_registered"]()


# --- objective packaging ---------------------------------------------------

def test_build_objective_packages_two_files_and_registration():
    build = build_capability_objective(_surface_spec())
    assert build.module_path == "augmentum/intent/builtin/syn_navigate_open_browse.py"
    assert build.test_path == "tests/test_syn_navigate_open_browse.py"
    assert "register_action" in build.module_source
    assert "navigate.open_surface" in build.module_source
    assert build.registration_line in build.objective
    assert set(build.files()) == {build.module_path, build.test_path}


def test_build_rejects_invalid_spec():
    bad = CapabilitySpec(id="bad", summary="", examples=[], behavior="speak")
    with pytest.raises(ValueError):
        build_capability_objective(bad)


# --- synthesis (closed-world, injected model) ------------------------------

# A genuinely net-new verb (a quip with no existing equivalent) — note we do NOT
# enshrine a navigate.open_* example here: navigation is already covered by the
# universal navigate.open_surface verb, so such a verb collides (see the dedicated
# collision tests). These success-path tests disable the live matcher/registry for
# determinism; the gates themselves are exercised in their own tests.
_VALID_JSON = json.dumps({
    "id": "fun.coin_flip_quip",
    "summary": "Say a short coin-flip quip.",
    "examples": ["give me a coin flip quip"],
    "behavior": "speak",
    "speak": "Heads you win, tails you learn.",
    "stakes": "trivial_reversible",
})

def _NO_MATCH(_t):   # disable the collision gate deterministically
    return None


async def test_synthesize_happy_path():
    async def mi(_prompt: str) -> str:
        return f"sure, here you go:\n```json\n{_VALID_JSON}\n```"   # fenced + prose
    spec, errs = await synthesize_capability_spec(
        "give me a coin flip quip", model_invoke=mi, known_verbs=[], match_fn=_NO_MATCH,
    )
    assert errs == [] and spec is not None and spec.id == "fun.coin_flip_quip"


async def test_synthesize_repairs_on_second_try():
    calls: list[str] = []

    async def mi(prompt: str) -> str:
        calls.append(prompt)
        return '{"id": "bad id", "behavior": "nope"}' if len(calls) == 1 else _VALID_JSON

    spec, errs = await synthesize_capability_spec(
        "a coin flip quip", model_invoke=mi, known_verbs=[], match_fn=_NO_MATCH,
    )
    assert spec is not None and errs == []
    assert len(calls) == 2 and "invalid" in calls[1].lower()  # repair prompt carried the errors


async def test_synthesize_unsupported_returns_no_spec():
    async def mi(_p: str) -> str:
        return '{"unsupported": "needs to send an email"}'
    spec, errs = await synthesize_capability_spec("email my mom", model_invoke=mi)
    assert spec is None and any("unsupported" in e for e in errs)


async def test_synthesize_junk_returns_no_spec():
    async def mi(_p: str) -> str:
        return "I cannot help with that."
    spec, errs = await synthesize_capability_spec("do a thing", model_invoke=mi)
    assert spec is None and errs
