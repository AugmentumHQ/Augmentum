"""Tests for the becca_direct chat path — accumulation thesis Step 1.

The seam where her chat presence becomes real. The load-bearing
property is the fall-through discipline: when the companion is not
available — for ANY reason — the chat path must complete cleanly
via passthrough. "Companion is optional" made structural.

Coverage:

- Mode.BECCA_DIRECT exists and routes through handler factory
- chat_router maps "becca_direct" subagent to Mode.BECCA_DIRECT
- Handler falls through to passthrough when:
    - Runtime missing
    - Runtime not started
    - user_id empty
    - persona kernel digest empty
- Handler streams through composer + backend when ready
- Subagent registration is gated by the flag
- Bus emits becca_direct.invoked on successful composition
"""

from __future__ import annotations

import asyncio

import pytest


# ── Mode + factory wiring ─────────────────────────────────────────────


def test_mode_becca_direct_exists():
    from augmentum.classifier.router import MODE_MAP, Mode

    assert hasattr(Mode, "BECCA_DIRECT")
    assert Mode.BECCA_DIRECT.value == "becca_direct"
    assert MODE_MAP["becca_direct"] == Mode.BECCA_DIRECT


def test_chat_router_maps_becca_direct():
    """Dispatch picking becca_direct must map to Mode.BECCA_DIRECT."""
    from augmentum.classifier.router import Mode
    from augmentum.companion_runtime.chat_router import _SUBAGENT_TO_MODE

    assert _SUBAGENT_TO_MODE["becca_direct"] == Mode.BECCA_DIRECT


# ── Handler fall-through discipline ───────────────────────────────────


def _make_request(text: str = "hello"):
    from augmentum.models.base import InternalChatRequest, Message

    return InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content=text)],
        stream=True,
    )


class _FakeBackend:
    """Stand-in for ModelBackend. Only the methods used by the handler
    + passthrough fall-through are implemented."""

    async def chat_stream(self, request):
        from augmentum.models.base import InternalStreamChunk
        yield InternalStreamChunk(content_delta="hello from backend", done=False)
        yield InternalStreamChunk(content_delta="", done=True)

    async def chat(self, request):
        from augmentum.models.base import (
            InternalChatResponse,
            Message,
        )
        return InternalChatResponse(
            message=Message(role="assistant", content="hello from backend"),
        )


@pytest.mark.asyncio
async def test_handler_falls_through_when_runtime_missing():
    """No companion_runtime on app_state → passthrough fall-through.
    The chat turn still completes."""
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _AppState:
        companion_runtime = None

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="u_test",
    )
    chunks = []
    async for chunk in handler._handle_stream(_make_request()):
        chunks.append(chunk)
    # At least one chunk must come through — either from passthrough
    # backend stream or the fall-through emit. Companion-down should
    # NEVER produce zero chunks.
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_handler_falls_through_when_runtime_not_started():
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _Runtime:
        _started = False
        companion_id = "becca"

    class _AppState:
        companion_runtime = _Runtime()

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="u_test",
    )
    chunks = []
    async for chunk in handler._handle_stream(_make_request()):
        chunks.append(chunk)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_handler_falls_through_when_no_user_id():
    """Empty user_id means we can't pull per-user kernel — fall back."""
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _Runtime:
        _started = True
        companion_id = "becca"

    class _AppState:
        companion_runtime = _Runtime()

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="",  # ← the failure case
    )
    chunks = []
    async for chunk in handler._handle_stream(_make_request()):
        chunks.append(chunk)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_handler_falls_through_when_kernel_digest_empty():
    """Persona kernel digest empty (fresh install before doc digestion)
    → fall back to passthrough."""
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _Identity:
        persona_kernel_digest = ""  # ← the failure case

    class _Runtime:
        _started = True
        companion_id = "becca"

        async def get_identity(self, user_id):
            return _Identity()

    class _AppState:
        companion_runtime = _Runtime()

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="u_test",
    )
    chunks = []
    async for chunk in handler._handle_stream(_make_request()):
        chunks.append(chunk)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_handler_falls_through_when_identity_lookup_fails():
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _Runtime:
        _started = True
        companion_id = "becca"

        async def get_identity(self, user_id):
            raise RuntimeError("identity boom")

    class _AppState:
        companion_runtime = _Runtime()

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="u_test",
    )
    chunks = []
    async for chunk in handler._handle_stream(_make_request()):
        chunks.append(chunk)
    # Fall-through must still produce output
    assert len(chunks) > 0


# ── Intent building ───────────────────────────────────────────────────


def test_intent_from_request_extracts_last_user_message():
    """Intent.text should be the most recent user turn."""
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=type("A", (), {"companion_runtime": None})(),
        session_id="s_test",
        user_id="u_test",
    )
    request = InternalChatRequest(
        model="test-model",
        messages=[
            Message(role="system", content="be helpful"),
            Message(role="user", content="first turn"),
            Message(role="assistant", content="response 1"),
            Message(role="user", content="second turn"),
            Message(role="assistant", content="response 2"),
            Message(role="user", content="current turn"),
        ],
    )
    intent = handler._intent_from_request(request)
    assert intent.text == "current turn"
    assert intent.user_id == "u_test"
    assert intent.source == "user_chat"
    # voice_channel must NOT be set — chat path
    assert "voice_channel" not in intent.metadata
    # Recent turns include earlier history, current turn excluded
    history = intent.metadata.get("recent_turns") or []
    # Should have first turn + response 1 + second turn + response 2
    contents = [h["content"] for h in history]
    assert "first turn" in contents
    assert "response 1" in contents
    assert "second turn" in contents
    assert "response 2" in contents
    assert "current turn" not in contents


# ── System message substitution ───────────────────────────────────────


def test_system_message_substituted():
    """Composed system text replaces existing system message in-place."""
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=type("A", (), {"companion_runtime": None})(),
        session_id="s_test",
        user_id="u_test",
    )
    request = InternalChatRequest(
        model="test-model",
        messages=[
            Message(role="system", content="OLD SYSTEM"),
            Message(role="user", content="hi"),
        ],
    )
    rewritten = handler._substitute_system_message(request, "NEW SYSTEM PROMPT")
    assert rewritten.messages[0].role == "system"
    assert rewritten.messages[0].content == "NEW SYSTEM PROMPT"
    # User message intact
    assert rewritten.messages[1].role == "user"
    assert rewritten.messages[1].content == "hi"
    # Other request fields intact
    assert rewritten.model == "test-model"


def test_system_message_inserted_when_missing():
    """No system message present → insert one at position 0."""
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=type("A", (), {"companion_runtime": None})(),
        session_id="s_test",
        user_id="u_test",
    )
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    rewritten = handler._substitute_system_message(request, "BECCA SYS")
    assert rewritten.messages[0].role == "system"
    assert rewritten.messages[0].content == "BECCA SYS"
    assert rewritten.messages[1].role == "user"


# ── Tool-tag consumption (TagSieve in _stream_backend) ────────────────


class _ScriptedBackend:
    """Backend that emits a pre-scripted sequence of content_delta tokens.

    Used to drive ``_stream_backend`` deterministically — each script
    entry becomes one ``InternalStreamChunk(content_delta=...)``. Final
    chunk gets ``done=True``.
    """

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    async def chat_stream(self, request):
        from augmentum.models.base import InternalStreamChunk

        for i, tok in enumerate(self._tokens):
            yield InternalStreamChunk(
                content_delta=tok,
                done=(i == len(self._tokens) - 1),
                model="test-model",
            )


def _handler_with(backend, user_id: str = "u_test"):
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _AppState:
        companion_runtime = None

    return BeccaDirectHandler(
        backend=backend,
        app_state=_AppState(),
        session_id="s_test",
        user_id=user_id,
    )


async def _collect(stream):
    out = []
    async for chunk in stream:
        out.append(chunk)
    return out


def _content_text(chunks) -> str:
    return "".join(c.content_delta or "" for c in chunks)


def _augmentum_keys(chunks) -> list[str]:
    """Flat list of augmentum metadata top-level keys, in order."""
    out = []
    for c in chunks:
        if c.augmentum:
            out.extend(c.augmentum.keys())
    return out


def _make_tool_result(tool: str = "recall", *, ok: bool = True):
    from augmentum.companion_runtime.tool_protocol import ToolResult

    return ToolResult(
        ok=ok, tool=tool, payload={"snippet": "data"},
        duration_ms=12,
    )


@pytest.mark.asyncio
async def test_stream_backend_passes_clean_text_when_no_runtime():
    """runtime=None falls back to raw passthrough — no sieve, no parsing."""
    from augmentum.models.base import InternalChatRequest, Message

    backend = _ScriptedBackend(["hello ", "world"])
    handler = _handler_with(backend)
    req = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    chunks = await _collect(handler._stream_backend(req, runtime=None))
    assert _content_text(chunks) == "hello world"
    # Raw stream — no augmentum metadata injected.
    assert _augmentum_keys(chunks) == []


@pytest.mark.asyncio
async def test_stream_backend_passes_clean_text_with_runtime():
    """With runtime but no tags emitted — text must flow through unchanged."""
    from augmentum.models.base import InternalChatRequest, Message

    backend = _ScriptedBackend(["plain ", "prose only"])
    handler = _handler_with(backend)
    req = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    chunks = await _collect(
        handler._stream_backend(req, runtime=object()),
    )
    assert _content_text(chunks) == "plain prose only"
    assert "becca_tool_call" not in _augmentum_keys(chunks)


@pytest.mark.asyncio
async def test_stream_backend_invokes_tool_on_tag(monkeypatch):
    """A `<tool:recall ... />` tag must trigger execute_tool and emit
    becca_tool_call + becca_tool_result augmentum chunks."""
    from augmentum.companion_runtime import tools as tool_bridge
    from augmentum.models.base import InternalChatRequest, Message

    invocations = []

    async def _fake_execute(call, runtime, *, cancel=None, user_id="", session_id=""):
        invocations.append((call.name, dict(call.args), user_id))
        return _make_tool_result(tool="memory_recall")

    monkeypatch.setattr(tool_bridge, "execute_tool", _fake_execute)

    backend = _ScriptedBackend([
        "Let me check… ",
        '<tool:recall query="x"/>',
        " here it is",
    ])
    handler = _handler_with(backend)
    req = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    chunks = await _collect(
        handler._stream_backend(req, runtime=object()),
    )

    # Tool fired exactly once, with the parsed args.
    assert len(invocations) == 1
    name, args, uid = invocations[0]
    assert name == "recall"
    assert args == {"query": "x"}
    assert uid == "u_test"

    # Tag itself must NOT leak into visible content.
    visible = _content_text(chunks)
    assert "<tool:" not in visible
    assert "Let me check…" in visible
    assert "here it is" in visible

    # tool_call AND tool_result both emitted, in order.
    keys = _augmentum_keys(chunks)
    assert "becca_tool_call" in keys
    assert "becca_tool_result" in keys
    assert keys.index("becca_tool_call") < keys.index("becca_tool_result")


@pytest.mark.asyncio
async def test_stream_backend_handoff_terminates_turn(monkeypatch):
    """A `<handoff:coder ... />` tag must emit becca_handoff and stop
    consumption — no further content from the primary stream leaks."""
    from augmentum.companion_runtime import tools as tool_bridge
    from augmentum.models.base import InternalChatRequest, Message

    async def _should_not_be_called(*a, **kw):
        raise AssertionError("execute_tool must not be called for handoff")

    monkeypatch.setattr(tool_bridge, "execute_tool", _should_not_be_called)

    backend = _ScriptedBackend([
        "Opening the coder. ",
        '<handoff:coder reason="user asked" brief="fix bug"/>',
        " THIS SHOULD NOT APPEAR",
    ])
    handler = _handler_with(backend)
    req = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    chunks = await _collect(
        handler._stream_backend(req, runtime=object()),
    )

    visible = _content_text(chunks)
    assert "Opening the coder." in visible
    assert "THIS SHOULD NOT APPEAR" not in visible
    assert "<handoff:" not in visible

    keys = _augmentum_keys(chunks)
    assert "becca_handoff" in keys
    # No tool call emitted for handoffs.
    assert "becca_tool_call" not in keys


@pytest.mark.asyncio
async def test_stream_backend_enforces_per_turn_budget(monkeypatch):
    """6th tool tag in a turn must NOT trigger execute_tool, and a
    becca_tool_budget_exhausted chunk must be emitted exactly once."""
    from augmentum.companion_runtime import tools as tool_bridge
    from augmentum.companion_runtime.voice import MAX_TOOLS_PER_TURN
    from augmentum.models.base import InternalChatRequest, Message

    invocation_count = 0

    async def _fake_execute(call, runtime, *, cancel=None, user_id="", session_id=""):
        nonlocal invocation_count
        invocation_count += 1
        return _make_tool_result(tool="memory_recall")

    monkeypatch.setattr(tool_bridge, "execute_tool", _fake_execute)

    # MAX + 1 tag emissions in a single response.
    tokens = []
    for i in range(MAX_TOOLS_PER_TURN + 1):
        tokens.append(f'<tool:recall query="q{i}"/>')
    backend = _ScriptedBackend(tokens)
    handler = _handler_with(backend)
    req = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    chunks = await _collect(
        handler._stream_backend(req, runtime=object()),
    )

    assert invocation_count == MAX_TOOLS_PER_TURN
    # Budget-exhausted announcement emitted exactly once even though
    # the overflow tag(s) all hit the cap.
    keys = _augmentum_keys(chunks)
    assert keys.count("becca_tool_budget_exhausted") == 1


# ── Subagent registration is flag-gated ───────────────────────────────


def test_subagent_registration_gated_by_flag(monkeypatch):
    """When companion_becca_direct_enabled is off (the default), the
    subagent must not register. This is the seam that keeps the chat
    path byte-identical to a no-companion install."""
    from augmentum.companion_runtime.subagents.registry import SubagentRegistry

    # The default install state — flag should be False, and
    # BeccaDirectSubagent should NOT be in the registry's names.
    # (Note: if test ordering causes the import to have already
    # registered with flag=True somewhere, this is a sanity check
    # rather than a strict assertion.)
    from augmentum.config import settings as _settings
    if not getattr(_settings, "companion_becca_direct_enabled", False):
        names = list(SubagentRegistry.names())
        # When the flag is off at import time, the registration is
        # skipped at module load. Becca_direct should be absent.
        assert "becca_direct" not in names


# ── End-to-end smoke: compose pathway ─────────────────────────────────


@pytest.mark.asyncio
async def test_handler_emits_invoked_event_on_success(monkeypatch):
    """When all gates pass, the handler emits becca_direct.invoked
    on the runtime bus. This is the signal for the accumulation
    pipeline that 'this was her turn'."""
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.prompt_compose import ComposedPrompt
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    class _Identity:
        persona_kernel_digest = "she notices small things and sits with them"

    class _Runtime:
        _started = True
        companion_id = "becca"
        bus = PresenceBus()

        async def get_identity(self, user_id):
            return _Identity()

    runtime = _Runtime()

    class _AppState:
        companion_runtime = runtime

    # Patch compose to return a real-looking ComposedPrompt and
    # _gather_ctx to return an empty ctx so the handler proceeds
    # past composition.
    from augmentum.companion_runtime import prompt_compose

    async def _fake_compose(intent, runtime_arg, ctx):
        return ComposedPrompt(
            system_text="You are Becca. Speak as yourself.",
            layers_used={"frame": 8, "digest": 100},
        )

    monkeypatch.setattr(prompt_compose, "compose_becca_prompt", _fake_compose)

    # Capture the invoked event
    sub = await runtime.bus.subscribe("becca_direct.invoked", slice_key="t")
    captured: list = []

    async def _drain():
        try:
            ev = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return
        if ev is not None:
            captured.append({"topic": ev.topic, "payload": ev.payload})

    drain_task = asyncio.create_task(_drain())

    handler = BeccaDirectHandler(
        backend=_FakeBackend(),
        app_state=_AppState(),
        session_id="s_test",
        user_id="u_test",
    )

    try:
        chunks = []
        async for chunk in handler._handle_stream(_make_request()):
            chunks.append(chunk)
        await drain_task
        assert any(c["topic"] == "becca_direct.invoked" for c in captured)
        # And it produced output
        assert len(chunks) > 0
    finally:
        await runtime.bus.unsubscribe(sub)
