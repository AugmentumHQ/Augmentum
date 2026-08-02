"""Seeded native loop — voice handoff of sieve-parsed calls.

Pins the best-universal-system adaptation (2026-06-11): calls parsed
from a streamed first hop execute through the loop's machinery, their
results land in request.messages in native format, and continuation
hops run the normal parse→execute cycle until the final synthesis.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.models.base import InternalChatRequest, InternalChatResponse, Message
from augmentum.modes.passthrough.handler import PassthroughHandler


class _FakeBackend:
    """Returns canned responses in order; records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests_seen = 0

    async def chat(self, request):
        self.requests_seen += 1
        return self._responses.pop(0)


def _resp(text):
    return InternalChatResponse(
        message=Message(role="assistant", content=text), model="fake",
    )


class _Bus:
    async def publish_topic(self, *a, **kw):
        return None


@pytest.fixture
def _patched_helper(monkeypatch):
    """Stub the passthrough primitives the loop borrows."""
    executed = []

    def _inject(self, request, tools):
        from augmentum.modes.analytical.tool_calling import ToolCallingTier
        return ToolCallingTier.NATIVE

    def _parse(self, response, tools):
        return []  # continuation hop finds no further calls

    async def _execute(self, request, response, calls, *, on_tool_start=None,
                       on_tool_result=None, text_tier=False, progress_queue=None):
        # Mirror the real contract: append assistant + tool messages.
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
                await on_tool_result(name, True, f"{name} ok", {}, tc_id, 5)
        return succeeded

    monkeypatch.setattr(PassthroughHandler, "_inject_tool_schemas", _inject)
    monkeypatch.setattr(PassthroughHandler, "_parse_tool_calls", _parse)
    monkeypatch.setattr(PassthroughHandler, "_execute_and_append", _execute)
    return executed


def _fake_tool(name):
    return SimpleNamespace(name=name, cacheable=True)


def _registry(names):
    tools = {n: _fake_tool(n) for n in names}
    return SimpleNamespace(resolve=lambda n: tools.get(n))


def _runtime():
    return SimpleNamespace(bus=_Bus(), companion_id="becca", _app_state=None)


async def _collect(gen):
    return [ev async for ev in gen]


@pytest.mark.asyncio
async def test_seeded_calls_execute_then_loop_synthesizes(_patched_helper, monkeypatch):
    from augmentum.companion_runtime import native_loop
    from augmentum.companion_runtime import tools as tool_bridge

    monkeypatch.setattr(
        tool_bridge, "enumerate_tools",
        lambda text, pin=None: [{"name": "note.append"}],
    )

    backend = _FakeBackend([_resp("Added the CPI numbers to your note.")])
    request = InternalChatRequest(
        model="fake",
        messages=[Message(role="system", content="sys"),
                  Message(role="user", content="look it up and add to the note")],
    )
    events = await _collect(native_loop.native_loop_events(
        request,
        backend=backend,
        runtime=_runtime(),
        intent=SimpleNamespace(text="look it up", user_id="u1", metadata={}),
        registry=_registry(["note.append", "web_search", "web_fetch",
                            "image_generation", "memory_recall", "wikipedia"]),
        user_id="u1",
        session_id="s1",
        app_state=None,
        initial_calls=[("note.append", {"content": "CPI rose 2.9%"})],
        initial_assistant_text="Let me add that.",
        drain_surface_events=False,
    ))

    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "text"]
    assert events[0][1]["tool"] == "note.append"
    assert events[1][1]["ok"] is True
    assert "CPI" in events[2][1]["text"]
    # The seeded call executed exactly once, before any model hop.
    assert _patched_helper == [("note.append", {"content": "CPI rose 2.9%"})]
    # The continuation hop saw the tool result in the conversation.
    assert backend.requests_seen == 1
    roles = [m.role for m in request.messages]
    assert "tool" in roles and "assistant" in roles


@pytest.mark.asyncio
async def test_cancel_before_start_yields_nothing(_patched_helper, monkeypatch):
    from augmentum.companion_runtime import native_loop
    from augmentum.companion_runtime import tools as tool_bridge

    monkeypatch.setattr(tool_bridge, "enumerate_tools", lambda text, pin=None: [])

    class _Cancelled:
        def is_set(self):
            return True

    backend = _FakeBackend([])
    request = InternalChatRequest(
        model="fake", messages=[Message(role="user", content="x")],
    )
    events = await _collect(native_loop.native_loop_events(
        request,
        backend=backend,
        runtime=_runtime(),
        intent=SimpleNamespace(text="x", user_id="u1", metadata={}),
        registry=_registry(["web_search", "web_fetch", "image_generation",
                            "memory_recall", "wikipedia"]),
        user_id="u1",
        session_id="s1",
        app_state=None,
        initial_calls=[("web_search", {"query": "q"})],
        cancel=_Cancelled(),
        drain_surface_events=False,
    ))
    assert events == []
    assert backend.requests_seen == 0
    assert _patched_helper == []


@pytest.mark.asyncio
async def test_no_initial_calls_runs_plain_loop(_patched_helper, monkeypatch):
    from augmentum.companion_runtime import native_loop
    from augmentum.companion_runtime import tools as tool_bridge

    monkeypatch.setattr(tool_bridge, "enumerate_tools", lambda text, pin=None: [])

    backend = _FakeBackend([_resp("Just a chat answer.")])
    request = InternalChatRequest(
        model="fake", messages=[Message(role="user", content="hi")],
    )
    events = await _collect(native_loop.native_loop_events(
        request,
        backend=backend,
        runtime=_runtime(),
        intent=SimpleNamespace(text="hi", user_id="u1", metadata={}),
        registry=_registry(["web_search", "web_fetch", "image_generation",
                            "memory_recall", "wikipedia"]),
        user_id="u1",
        session_id="s1",
        app_state=None,
        drain_surface_events=False,
    ))
    assert [k for k, _ in events] == ["text"]
    assert events[0][1]["text"] == "Just a chat answer."


# ── Roster family diversity ───────────────────────────────────────────

def test_roster_family_cap_keeps_multi_intent_breadth():
    """A two-intent ask must not let one verb family monopolize the
    roster: 'throw in some music and open a note' scored five note.*
    verbs above any music verb and the budget filled with notes
    (2026-06-11). The family cap guarantees breadth: other families'
    top verbs rank ahead of any family's 4th verb, so the music half
    of the ask is represented.
    """
    import augmentum.architect.primitives  # noqa: F401 — grove et al.
    from augmentum.companion_runtime.tools import enumerate_tools

    roster = enumerate_tools(
        "Hey there. can you throw in some music and open a note for me?"
    )
    dotted = [t["name"] for t in roster if "." in t["name"]]
    fams = {}
    for n in dotted:
        fam = n.split(".", 1)[0]
        fams[fam] = fams.get(fam, 0) + 1
    # Breadth: the ranked window spans multiple families, not one.
    assert len(fams) >= 3, fams
    # The music half of the ask is represented.
    assert any(
        n in ("grove.play_matching", "media.play") for n in dotted
    ), dotted
