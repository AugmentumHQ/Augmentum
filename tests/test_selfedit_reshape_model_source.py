"""Tests for pluggable model sources (model list default + sub-in Claude).

All deps injected. Load-bearing:
  - the model-list ("augmentum") source is the default and builds from the registry;
  - a Claude API-key credential builds a working one-shot invoke (reusing the
    coder login) via an injected backend;
  - a Claude *subscription OAuth* credential is NOT used via the API (honest skip)
    → resolve falls back to the model list;
  - no token / unknown source → graceful fallback, never strands the feature.
"""

from __future__ import annotations

from augmentum.selfedit.surfaces import (
    ModelSource,
    SourceContext,
    clear_sources,
    get_source,
    list_sources,
    register_default_sources,
    register_source,
    resolve_invoke,
)


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.message = _Msg(content)


class FakeBackend:
    def __init__(self, content):
        self._content = content

    async def chat(self, req):
        return _Resp(self._content)


class FakeRegistry:
    async def resolve_model_for_role(self, role, override="", settings=None):
        return FakeBackend('{"ok":true}'), "utility-model"


def _ctx(**kw):
    return SourceContext(**kw)


async def _loader_returning(token):
    async def load(_store, _uid):
        return token
    return load


# --- registry --------------------------------------------------------------

def test_register_default_sources_lists_both():
    clear_sources()
    register_default_sources()
    ids = {s["id"] for s in list_sources()}
    assert ids == {"augmentum", "claude"}
    assert get_source("augmentum") is not None


# --- augmentum (model list) default ---------------------------------------

async def test_augmentum_source_builds_from_registry():
    clear_sources()
    register_default_sources()
    inv = await resolve_invoke("augmentum", _ctx(provider_registry=FakeRegistry()))
    assert inv is not None
    out = await inv("prompt")
    assert out == '{"ok":true}'                      # routed through the role layer


# --- claude (sub-in via coder login) --------------------------------------

async def test_claude_api_key_builds_working_invoke():
    clear_sources()
    register_default_sources()
    ctx = _ctx(user_id="u1", settings_store=object(),
               claude_token_loader=await _loader_returning("sk-ant-api-test123"),
               claude_backend_factory=lambda _tok: FakeBackend('{"surface":"config"}'))
    inv = await resolve_invoke("claude", ctx)
    assert inv is not None
    assert await inv("prompt") == '{"surface":"config"}'   # one-shot via the Anthropic backend


async def test_claude_oauth_token_falls_back_to_model_list():
    clear_sources()
    register_default_sources()
    ctx = _ctx(user_id="u1", settings_store=object(), provider_registry=FakeRegistry(),
               claude_token_loader=await _loader_returning("sk-ant-oat01-subscription"))
    inv = await resolve_invoke("claude", ctx)            # OAuth → API path skipped
    assert inv is not None
    assert await inv("prompt") == '{"ok":true}'          # fell back to the model list


async def test_no_token_falls_back():
    clear_sources()
    register_default_sources()
    ctx = _ctx(user_id="u1", settings_store=object(), provider_registry=FakeRegistry(),
               claude_token_loader=await _loader_returning(""))
    inv = await resolve_invoke("claude", ctx)
    assert inv is not None and await inv("p") == '{"ok":true}'


async def test_unknown_source_falls_back_to_model_list():
    clear_sources()
    register_default_sources()
    inv = await resolve_invoke("does-not-exist", _ctx(provider_registry=FakeRegistry()))
    assert inv is not None and await inv("p") == '{"ok":true}'


async def test_custom_source_can_register():
    clear_sources()

    async def build(_ctx):
        async def inv(_p):
            return "custom"
        return inv
    register_source(ModelSource("codex", "Codex (future)", build))
    inv = await resolve_invoke("codex", _ctx())
    assert await inv("p") == "custom"                    # the same shape extends to Codex etc.
