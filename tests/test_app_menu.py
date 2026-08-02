"""App menu — catalog store, filtering funnel, and the app.act verb.

Pins the 2026-06-10 foundation: client-synced palette commands as
companion action space, matched closed-world and stakes-capped.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import augmentum.intent  # noqa: F401 — registers app.act
from augmentum.intent import app_menu
from augmentum.intent.app_menu import MenuStore, observe_commands
from augmentum.intent.registry import REGISTRY


@pytest.fixture(autouse=True)
def _fresh_menu():
    app_menu.MENU.reset()
    yield
    app_menu.MENU.reset()


def _entries(**overrides):
    base = {
        "id": "grove.favorite-current",
        "description": "Add the currently playing station to favorites",
        "keywords": "favorite save like",
        "stakes": "trivial_reversible",
        "speak": "Saved — it's in your favorites.",
        "live": True,
    }
    base.update(overrides)
    return [base]


# ── Store + filtering funnel ──────────────────────────────────────────

def test_update_and_catalog_roundtrip():
    store = MenuStore()
    assert store.update("u1", _entries()) == 1
    cat = store.catalog("u1")
    assert len(cat) == 1
    assert cat[0]["id"] == "grove.favorite-current"
    assert cat[0]["speak"].startswith("Saved")


def test_dead_entries_filtered_from_candidates():
    store = MenuStore()
    store.update("u1", _entries(live=False))
    assert store.catalog("u1") == []
    # ...but still visible in the full synced list (honest-miss copy).
    assert len(store.all_entries("u1")) == 1


def test_stakes_cap_excludes_non_trivial():
    store = MenuStore()
    store.update("u1", _entries(id="files.delete", stakes="irrevocable"))
    assert store.catalog("u1") == []


def test_entries_without_id_or_description_dropped():
    store = MenuStore()
    assert store.update("u1", [{"id": "", "description": "x", "live": True}]) == 0
    assert store.update("u1", [{"id": "a.b", "description": "", "live": True}]) == 0


def test_catalog_caps_and_truncates():
    store = MenuStore()
    flood = [
        {"id": f"a.{i}", "description": "d" * 500, "live": True}
        for i in range(200)
    ]
    accepted = store.update("u1", flood)
    assert accepted == 64  # _MAX_ENTRIES
    assert all(len(e["description"]) <= 160 for e in store.catalog("u1"))


def test_per_user_isolation():
    store = MenuStore()
    store.update("u1", _entries())
    assert store.catalog("u2") == []


def test_observe_topic_mapper():
    observe_commands("u1", "surface.commands.catalog", {"entries": _entries()})
    assert len(app_menu.MENU.catalog("u1")) == 1
    # Wrong topic is a no-op, not an error.
    observe_commands("u1", "surface.audio.kind_changed", {"entries": []})
    assert len(app_menu.MENU.catalog("u1")) == 1


# ── Matcher (closed world, soft-fail) ─────────────────────────────────

@pytest.mark.asyncio
async def test_match_intent_empty_inputs():
    assert await app_menu.match_intent(
        "", _entries(), app_state=SimpleNamespace(provider_registry=None),
    ) is None
    assert await app_menu.match_intent(
        "favorite this", [], app_state=SimpleNamespace(provider_registry=None),
    ) is None


@pytest.mark.asyncio
async def test_match_intent_no_registry_soft_fails():
    state = SimpleNamespace(provider_registry=None)
    assert await app_menu.match_intent("favorite this", _entries(), app_state=state) is None


class _FakeBackend:
    def __init__(self, content):
        self._content = content

    async def chat(self, req):
        return SimpleNamespace(message=SimpleNamespace(content=self._content))


def _state_with_backend(content):
    async def _resolve(override, *, user_id="", session_id=""):
        return _FakeBackend(content), "fake-model"
    return SimpleNamespace(
        provider_registry=SimpleNamespace(resolve_backend_with_fabric=_resolve),
    )


@pytest.mark.asyncio
async def test_match_intent_picks_from_menu():
    state = _state_with_backend(
        '{"choice": "grove.favorite-current", "confidence": 0.9}'
    )
    entry = await app_menu.match_intent("love this song, save it", _entries(), app_state=state)
    assert entry is not None
    assert entry["id"] == "grove.favorite-current"


@pytest.mark.asyncio
async def test_match_intent_rejects_offlist_choice():
    # Closed world: an id we never showed it cannot execute.
    state = _state_with_backend('{"choice": "files.delete_all", "confidence": 0.99}')
    assert await app_menu.match_intent("delete everything", _entries(), app_state=state) is None


@pytest.mark.asyncio
async def test_match_intent_honors_none_and_low_confidence():
    assert await app_menu.match_intent(
        "what station is this?", _entries(),
        app_state=_state_with_backend('{"choice": "none"}'),
    ) is None
    assert await app_menu.match_intent(
        "hm maybe", _entries(),
        app_state=_state_with_backend(
            '{"choice": "grove.favorite-current", "confidence": 0.3}'
        ),
    ) is None


@pytest.mark.asyncio
async def test_match_intent_garbage_response_soft_fails():
    assert await app_menu.match_intent(
        "favorite this", _entries(),
        app_state=_state_with_backend("I think you should press the button!"),
    ) is None


# ── app.act verb ──────────────────────────────────────────────────────

def _get_action(action_id):
    for action in REGISTRY.all():
        if action.id == action_id:
            return action
    raise AssertionError(f"action {action_id} not registered")


def test_app_act_registered_tier3_artifact():
    action = _get_action("app.act")
    assert action.delivery == "artifact"
    assert action.fanout.tier3 and not action.fanout.tier1
    assert "intent" in action.required_args


def _session(user_id="u1", app_state=None):
    from augmentum.intent.action import SessionContext
    return SessionContext(
        user_id=user_id, session_id="s1", mode=None,
        app_state=app_state or SimpleNamespace(provider_registry=None),
    )


@pytest.mark.asyncio
async def test_app_act_empty_catalog_honest_miss():
    action = _get_action("app.act")
    result = await action.handler("", _session(), {"intent": "favorite this"})
    assert result.short_circuit
    assert "quick actions" in result.speak
    assert result.surface_emit is None


@pytest.mark.asyncio
async def test_app_act_match_emits_palette_run():
    app_menu.MENU.update("u1", _entries())
    state = _state_with_backend(
        '{"choice": "grove.favorite-current", "confidence": 0.9}'
    )
    action = _get_action("app.act")
    result = await action.handler(
        "", _session(app_state=state), {"intent": "add this to favorites"},
    )
    assert result.surface_emit["channel"] == "palette.run"
    assert result.surface_emit["payload"]["command_id"] == "grove.favorite-current"
    assert result.speak == "Saved — it's in your favorites."


@pytest.mark.asyncio
async def test_app_act_miss_names_live_options():
    app_menu.MENU.update("u1", _entries())
    state = _state_with_backend('{"choice": "none"}')
    action = _get_action("app.act")
    result = await action.handler(
        "", _session(app_state=state), {"intent": "order me a pizza"},
    )
    assert result.surface_emit is None
    assert "favorites" in result.speak  # nearest option named honestly


@pytest.mark.asyncio
async def test_app_act_anon_refused():
    action = _get_action("app.act")
    result = await action.handler("", _session(user_id=""), {"intent": "x"})
    assert "signed-out" in result.speak


# ── ActionTool user-context injection (both conventions) ──────────────

@pytest.mark.asyncio
async def test_action_tool_reads_user_id_from_both_conventions():
    """chain.py injects ``_user_id`` top-level; the native loop (via
    passthrough's executor) nests it in ``_context``. Both must reach
    the handler — reading only the first ran every loop-executed
    registry verb as anonymous ("signed-out" refusals, 2026-06-11).
    """
    from augmentum.intent.tool_adapter import ActionTool

    seen = []

    async def _handler(text, session, args):
        seen.append(session.user_id)
        from augmentum.intent.action import ActionResult
        return ActionResult(short_circuit=True, speak="ok")

    action = SimpleNamespace(
        id="test.echo_user", summary="t", arg_schema={}, required_args=[],
        handler=_handler,
    )
    tool = ActionTool(action, app_state=None)

    await tool.execute(_user_id="u_chain")
    await tool.execute(_context={"user_id": "u_loop", "session_id": "s1"})
    assert seen == ["u_chain", "u_loop"]
