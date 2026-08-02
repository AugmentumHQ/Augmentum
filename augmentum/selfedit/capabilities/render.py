"""Deterministic codegen — CapabilitySpec → verb module + smoke-test oracle.

Nothing here calls a model. The spec's behavior selects a fixed handler template;
the spec's data slots are embedded as Python *literals* via ``repr()`` (so a
payload/speak string can never become executable code). That's why a synthesized
verb is safe by construction: the only model-authored thing is *data*.

Synthesized verbs live in their own ``builtin/syn_<id>.py`` files so they never
collide with hand-authored modules and are visually obvious as machine-authored.
"""

from __future__ import annotations

from augmentum.selfedit.capabilities.spec import CapabilitySpec


def syn_module_stem(spec: CapabilitySpec) -> str:
    """Collision-proof builtin file stem for a synthesized verb."""
    return "syn_" + spec.id.replace(".", "_")


def _fanout_expr(spec: CapabilitySpec) -> str:
    return (
        f"ActionFanout(tier1={spec.tier1!r}, tier2={spec.tier2!r}, "
        f"tier3={spec.tier3!r})"
    )


def _handler_src(spec: CapabilitySpec) -> tuple[str, str]:
    """Return (handler_source, args_param_name). Args is named ``_args`` when the
    template doesn't read it (keeps the generated module lint-clean)."""
    fn = spec.func_name
    if spec.behavior == "surface_emit":
        body = (
            f"async def {fn}(_text, _session, args):\n"
            f"    payload = {{**{spec.payload!r}, **(args or {{}})}}\n"
            f"    return ActionResult(\n"
            f"        short_circuit=True,\n"
            f"        surface_emit={{\"channel\": {spec.channel!r}, \"payload\": payload}},\n"
            f"        toast={spec.toast!r},\n"
            f"        speak={spec.speak!r},\n"
            f"    )\n"
        )
        return body, "args"
    # behavior == "speak"
    body = (
        f"async def {fn}(_text, _session, _args):\n"
        f"    return ActionResult(short_circuit=True, speak={spec.speak!r})\n"
    )
    return body, "_args"


def render_verb_module(spec: CapabilitySpec) -> str:
    """The full source of the new builtin verb module."""
    delivery = "artifact" if spec.behavior == "surface_emit" else "verbal"
    handler_src, _ = _handler_src(spec)
    reg = (
        "register_action(\n"
        f"    id={spec.id!r},\n"
        f"    summary={spec.summary!r},\n"
        f"    examples={list(spec.examples)!r},\n"
        f"    handler={spec.func_name},\n"
        f"    arg_schema={dict(spec.arg_schema)!r},\n"
        f"    required={list(spec.required)!r},\n"
        f"    surfaces={list(spec.surfaces)!r},\n"
        f"    stakes={spec.stakes!r},\n"
        f"    fanout={_fanout_expr(spec)},\n"
        f"    delivery={delivery!r},\n"
        ")\n"
    )
    return (
        f'"""{spec.id} - synthesized verb (capability synthesis).\n'
        "\n"
        f"{spec.summary}\n"
        "\n"
        "Machine-authored from a CapabilitySpec via augmentum/selfedit/capabilities.\n"
        "Handler is template-rendered (bounded behavior), not free-form code.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "\n"
        "from augmentum.intent.action import ActionFanout, ActionResult\n"
        "from augmentum.intent.registry import register_action\n"
        "\n"
        "\n"
        f"{handler_src}"
        "\n"
        "\n"
        f"{reg}"
    )


def render_registration_line(spec: CapabilitySpec) -> str:
    """The import line to append to augmentum/intent/__init__.py so the verb's
    register_action() runs at package import (explicit, not side-effect discovery)."""
    stem = syn_module_stem(spec)
    return (
        f"from augmentum.intent.builtin import {stem} as _{stem}  # noqa: F401"
    )


def _sample_args(spec: CapabilitySpec) -> dict:
    """Plausible args for the dispatch smoke test, one per declared arg."""
    sample: dict = {}
    for name, meta in spec.arg_schema.items():
        t = (meta or {}).get("type", "string")
        sample[name] = {
            "string": "x", "integer": 1, "number": 1.0, "boolean": True,
        }.get(t, "x")
    return sample


def render_verb_test(spec: CapabilitySpec) -> str:
    """A pytest smoke test = the confirm-oracle. Asserts the verb registered with
    the specified shape AND that its handler dispatches as specified. A passing
    run earns the verifier's VERIFIED tier (not a bare 'didn't break').

    Crucially it verifies via the BOOT PATH — ``from augmentum.intent import
    REGISTRY`` runs the package ``__init__`` (which is what wires builtins at
    startup) and then asserts ``REGISTRY.get(id)``. It does NOT import the new
    module directly. A direct import would register the verb regardless of whether
    the ``intent/__init__.py`` import line was added, masking a skipped
    registration step — the verb would pass its oracle yet be dead at real boot
    (OpenRoom's Step 2.5 "without registration, actions won't route"). No direct
    import means the test fails unless the verb is wired the way production loads
    it."""
    safe = spec.id.replace(".", "_")
    sample = _sample_args(spec)

    if spec.behavior == "surface_emit":
        dispatch_asserts = (
            "    assert res is not None and res.short_circuit is True\n"
            "    assert res.surface_emit is not None\n"
            f"    assert res.surface_emit.get(\"channel\") == {spec.channel!r}\n"
        )
    else:
        dispatch_asserts = (
            "    assert res is not None and res.short_circuit is True\n"
            f"    assert res.speak == {spec.speak!r}\n"
        )

    return (
        f'"""Smoke test (confirm-oracle) for synthesized verb {spec.id}.\n'
        "\n"
        "Verifies via the boot path (the package import runs intent/__init__, which\n"
        "wires builtins at startup) - NOT a direct module import. So this fails\n"
        'unless the verb is registered the way production loads it."""\n'
        "from __future__ import annotations\n"
        "\n"
        "import asyncio\n"
        "\n"
        "from augmentum.intent import REGISTRY  # runs intent/__init__ = the boot wiring\n"
        "\n"
        "\n"
        f"def test_{safe}_registered():\n"
        f"    action = REGISTRY.get({spec.id!r})\n"
        "    assert action is not None\n"
        f"    assert action.id == {spec.id!r}\n"
        f"    assert action.stakes == {spec.stakes!r}\n"
        f"    assert set(action.arg_schema.keys()) == {set(spec.arg_schema.keys())!r}\n"
        f"    assert action.required_args == {list(spec.required)!r}\n"
        "\n"
        "\n"
        f"def test_{safe}_dispatches():\n"
        "    from augmentum.intent.action import SessionContext\n"
        f"    action = REGISTRY.get({spec.id!r})\n"
        "    ctx = SessionContext(user_id=\"u\", session_id=\"s\")\n"
        f"    res = asyncio.run(action.handler(\"\", ctx, {sample!r}))\n"
        f"{dispatch_asserts}"
    )
