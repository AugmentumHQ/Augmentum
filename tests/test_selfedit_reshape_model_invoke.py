"""Tests for the role-model invoke adapter (classifier model call → role layer).

Registry + backend are injected fakes. Load-bearing:
  - invoke routes through resolve_model_for_role and returns the model's content;
  - the no-thinking + greedy recipe is on the request;
  - resolve failure / no backend / chat error all degrade to "" (→ honest
    unmapped downstream, never a crash);
  - composes: build_role_model_invoke → build_model_classifier → a valid change.
"""

from __future__ import annotations

from augmentum.selfedit.surfaces import (
    ReshapeRequest,
    build_model_classifier,
    build_role_model_invoke,
    clear_schemas,
    example_adaptation_schema,
    register_schema,
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
        self.last_req = None

    async def chat(self, req):
        self.last_req = req
        return _Resp(self._content)


class FakeRegistry:
    def __init__(self, backend, model="utility-model"):
        self._backend = backend
        self._model = model
        self.resolved_role = None

    async def resolve_model_for_role(self, role, override="", settings=None):
        self.resolved_role = role
        return self._backend, self._model


async def test_invoke_routes_through_role_and_returns_content():
    backend = FakeBackend('{"surface":"config","key":"theme","value":"dark"}')
    reg = FakeRegistry(backend)
    invoke = build_role_model_invoke(reg, role="classifier")
    out = await invoke("the prompt")
    assert out.startswith("{")
    assert reg.resolved_role == "classifier"
    # the fast-call recipe: no-thinking + greedy + prompt as the user message
    assert backend.last_req.chat_template_kwargs == {"enable_thinking": False}
    assert backend.last_req.temperature == 0.0
    assert backend.last_req.messages[-1].content == "the prompt"


class _RaisingRegistry:
    async def resolve_model_for_role(self, role, override="", settings=None):
        raise RuntimeError("no model")


class _NoneRegistry:
    async def resolve_model_for_role(self, role, override="", settings=None):
        return None, ""


class _ChatErrBackend:
    async def chat(self, req):
        raise TimeoutError("slow")


async def test_failures_degrade_to_empty_string():
    assert await build_role_model_invoke(_RaisingRegistry())("p") == ""      # resolve raised
    assert await build_role_model_invoke(_NoneRegistry())("p") == ""         # no backend
    assert await build_role_model_invoke(FakeRegistry(_ChatErrBackend()))("p") == ""  # chat raised


async def test_composes_with_classifier_end_to_end():
    clear_schemas()
    register_schema(example_adaptation_schema())
    backend = FakeBackend('{"surface":"config","key":"theme","value":"Dark"}')
    invoke = build_role_model_invoke(FakeRegistry(backend))
    classifier = build_model_classifier(invoke)
    change = await classifier(ReshapeRequest(ask="make it dark", actor="u1"), ["config"])
    assert change is not None
    assert change.payload == {"key": "theme", "value": "dark"}   # normalized, end-to-end
