"""Tests for the reshape verb + the unified ask→present handler (one verb, every
surface).

Fakes for the model + store, so deterministic. Load-bearing:
  - handle_reshape_ask runs the engine and returns the one presentation;
  - the verb classifies via the model SOURCE, applies, and emits the
    `reshape.result` payload every surface renders;
  - the verb refuses the anon user (never writes the shared row).
"""

from __future__ import annotations

from augmentum.intent.action import SessionContext
from augmentum.intent.builtin.reshape import _reshape
from augmentum.selfedit.surfaces import (
    CLASS_ADAPTATION,
    ReshapeChange,
    build_config_surface,
    clear_schemas,
    clear_sources,
    clear_surfaces,
    example_adaptation_schema,
    handle_reshape_ask,
    register_default_sources,
    register_schema,
    register_surface,
)


class _Store:
    def __init__(self):
        self.data: dict = {}

    async def read(self, uid, key):
        return self.data.get((uid, key))

    async def write(self, uid, key, val):
        self.data[(uid, key)] = val


def _register_config(store: _Store):
    clear_surfaces()
    register_surface(build_config_surface(read=store.read, write=store.write))


# --- the unified handler ---------------------------------------------------

async def test_handle_reshape_ask_applies_and_presents():
    store = _Store()
    _register_config(store)

    async def classify(req, _surfaces):
        return ReshapeChange(surface="config", change_class=CLASS_ADAPTATION,
                             payload={"key": "theme", "value": "dark"}, actor=req.actor)

    pres = await handle_reshape_ask("make it dark", "u1", classify=classify)
    assert pres.status == "applied" and "theme" in pres.detail
    assert store.data[("u1", "theme")] == "dark"
    assert pres.actions and pres.actions[0].id == "undo"   # reversible


# --- the verb (model-source classify → emit reshape.result) ----------------

class _Msg:
    def __init__(self, c):
        self.content = c


class _Resp:
    def __init__(self, c):
        self.message = _Msg(c)


class _Backend:
    def __init__(self, c):
        self._c = c

    async def chat(self, _req):
        return _Resp(self._c)


class _Registry:
    def __init__(self, c):
        self._c = c

    async def resolve_model_for_role(self, role, override="", settings=None):
        return _Backend(self._c), "m"


class _AppState:
    def __init__(self, registry):
        self.provider_registry = registry
        self.settings = None
        self.state_manager = None


async def test_reshape_verb_classifies_applies_and_emits():
    store = _Store()
    _register_config(store)
    clear_schemas()
    register_schema(example_adaptation_schema())   # config: theme/density/accent
    clear_sources()
    register_default_sources()                     # augmentum (model list) + claude

    registry = _Registry('{"surface":"config","key":"theme","value":"dark"}')
    session = SessionContext(user_id="u1", session_id="s1", app_state=_AppState(registry))
    res = await _reshape("make it dark", session, {})

    assert res.surface_emit and res.surface_emit["channel"] == "reshape.result"
    payload = res.surface_emit["payload"]
    assert payload["status"] == "applied"          # one payload, every surface renders it
    assert res.fulfilled is True and res.speak
    assert store.data[("u1", "theme")] == "dark"


async def test_reshape_verb_refuses_anon():
    res = await _reshape("make it dark", SessionContext(user_id="", session_id="s1"), {})
    assert res.fulfilled is False and "signed-in" in res.speak
    assert res.surface_emit is None                # nothing applied for the anon row
