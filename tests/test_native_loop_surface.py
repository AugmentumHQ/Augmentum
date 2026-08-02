"""Voice surface routing + long-task detach in the native loop (2026-06-15).

Two behaviors, voice-only (``drain_surface_events=False``):

* **C** — visual tool results (image_search / youtube) are parked as
  ``intent_action`` surface events so the route drain opens the native
  panel instead of narrating URLs out loud.
* **D** — long tasks (image generation) hand off to their client panel
  and the turn completes, so the user can keep talking while it renders
  (the long-horizon detach-and-notify contract). Chat (drain=True) is
  unaffected: it runs these inline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.models.base import InternalChatRequest, InternalChatResponse, Message
from augmentum.modes.passthrough.handler import PassthroughHandler


class _FakeBackend:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests_seen = 0

    async def chat(self, request):
        self.requests_seen += 1
        return self._responses.pop(0) if self._responses else _resp("")


def _resp(text):
    return InternalChatResponse(
        message=Message(role="assistant", content=text), model="fake",
    )


class _Bus:
    async def publish_topic(self, *a, **kw):
        return None


@pytest.fixture
def _patched_helper(monkeypatch):
    """Stub passthrough primitives; record any tool that actually executed."""
    executed = []

    def _inject(self, request, tools):
        from augmentum.modes.analytical.tool_calling import ToolCallingTier
        return ToolCallingTier.NATIVE

    def _parse(self, response, tools):
        return []

    async def _execute(self, request, response, calls, *, on_tool_start=None,
                       on_tool_result=None, text_tier=False, progress_queue=None):
        request.messages.append(Message(
            role="assistant",
            content=response.message.content if response.message else "",
        ))
        succeeded = set()
        for name, args, tc_id in calls:
            executed.append((name, dict(args)))
            request.messages.append(Message(
                role="tool", content=f"{name} ok", tool_call_id=tc_id,
            ))
            succeeded.add(name)
            if on_tool_result:
                await on_tool_result(name, True, f"{name} ok", _META.get(name, {}), tc_id, 5)
        return succeeded

    monkeypatch.setattr(PassthroughHandler, "_inject_tool_schemas", _inject)
    monkeypatch.setattr(PassthroughHandler, "_parse_tool_calls", _parse)
    monkeypatch.setattr(PassthroughHandler, "_execute_and_append", _execute)
    return executed


# Tool result metadata the fake executor hands back per tool.
_META = {
    "image_search": {"images": [
        {"embed_url": "/api/artifacts/1/download", "title": "red panda"},
    ]},
}


def _fake_tool(name):
    return SimpleNamespace(name=name, cacheable=True)


def _registry(names):
    tools = {n: _fake_tool(n) for n in names}
    return SimpleNamespace(resolve=lambda n: tools.get(n))


def _runtime():
    return SimpleNamespace(bus=_Bus(), companion_id="becca", _app_state=None)


@pytest.fixture
def _fake_refs(monkeypatch):
    """Fake ReferentCache so surface-event pushes are observable, and a
    no-op roster so selection doesn't hit the real registry/embedder."""
    import importlib

    from augmentum.companion_runtime import tools as tool_bridge

    # augmentum.intent.__init__ exposes a `dispatch` FUNCTION that shadows
    # the submodule attribute, so import the real module object directly.
    intent_dispatch = importlib.import_module("augmentum.intent.dispatch")

    refs = SimpleNamespace(pending_surface_events=[])
    # native_loop imports get_referent_cache from intent.dispatch at call time.
    monkeypatch.setattr(intent_dispatch, "get_referent_cache", lambda *a, **k: refs)
    monkeypatch.setattr(tool_bridge, "enumerate_tools", lambda text, pin=None: [])
    monkeypatch.setattr(tool_bridge, "pending_pin", lambda *a, **k: None)
    # ring.record also reaches for the cache — keep it a no-op.
    from augmentum.companion_runtime import ring
    monkeypatch.setattr(ring, "record", lambda *a, **k: None)
    return refs


async def _collect(gen):
    return [ev async for ev in gen]


def _run(backend, registry, *, initial_calls, drain):
    from augmentum.companion_runtime import native_loop
    return native_loop.native_loop_events(
        InternalChatRequest(model="fake", messages=[
            Message(role="user", content="go"),
        ]),
        backend=backend,
        runtime=_runtime(),
        intent=SimpleNamespace(text="go", user_id="u1", metadata={}),
        registry=registry,
        user_id="u1",
        session_id="s1",
        app_state=object(),
        initial_calls=initial_calls,
        drain_surface_events=drain,
    )


# ── D: long-task detach (image generation) ────────────────────────────


@pytest.mark.asyncio
async def test_image_generation_detaches_to_panel(_patched_helper, _fake_refs):
    """Voice: an image_generation call parks an image.generate surface
    event, does NOT run the server tool, and the turn still completes
    with a spoken ack."""
    backend = _FakeBackend([_resp("Okay — putting that together now.")])
    events = await _collect(_run(
        backend, _registry(["image_generation"]),
        initial_calls=[("image_generation", {"prompt": "a red panda"})],
        drain=False,
    ))

    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "text"]
    # The server-side tool never ran — it was handed off.
    assert _patched_helper == []
    # A surface event opened the image panel with the prompt.
    assert len(_fake_refs.pending_surface_events) == 1
    sev = _fake_refs.pending_surface_events[0]
    assert sev["surface"]["channel"] == "image.generate"
    assert sev["surface"]["payload"]["prompt"] == "a red panda"
    # She acknowledged without claiming it's done.
    assert "putting that together" in events[2][1]["text"].lower()


@pytest.mark.asyncio
async def test_chat_path_runs_image_generation_inline(_patched_helper, _fake_refs):
    """Chat (drain=True): NO detach — the tool runs inline so the image
    lands in the thread, and nothing is parked as a surface event."""
    backend = _FakeBackend([_resp("Here's what I made.")])
    await _collect(_run(
        backend, _registry(["image_generation"]),
        initial_calls=[("image_generation", {"prompt": "a red panda"})],
        drain=True,
    ))
    assert _patched_helper == [("image_generation", {"prompt": "a red panda"})]
    assert _fake_refs.pending_surface_events == []


# ── C: visual results → native panel ──────────────────────────────────


@pytest.mark.asyncio
async def test_image_search_parks_surface_event(_patched_helper, _fake_refs):
    """Voice: image_search runs (it returns data she narrates), AND its
    results are parked as an image.search surface event for the viewer."""
    backend = _FakeBackend([_resp("Found a few — here they are.")])
    events = await _collect(_run(
        backend, _registry(["image_search"]),
        initial_calls=[("image_search", {"query": "red panda"})],
        drain=False,
    ))
    # The tool DID run (unlike image gen — search returns data).
    assert _patched_helper == [("image_search", {"query": "red panda"})]
    assert any(k == "tool_result" for k, _ in events)
    # And its images were routed to the viewer.
    assert len(_fake_refs.pending_surface_events) == 1
    sev = _fake_refs.pending_surface_events[0]
    assert sev["surface"]["channel"] == "image.search"
    assert sev["surface"]["payload"]["images"][0]["title"] == "red panda"
