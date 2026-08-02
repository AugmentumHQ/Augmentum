"""Tests for passthrough mode handler and SSOS orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    Usage,
)
from augmentum.modes.passthrough.handler import PassthroughHandler
from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator


def _make_request(content: str = "Hello", model: str = "test-model", stream: bool = False):
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content=content)],
        stream=stream,
    )


def _make_backend():
    """Create a mock backend with chat and chat_stream methods."""
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content="Mock response"),
        model="test-model",
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))

    async def _mock_stream(request):
        yield InternalStreamChunk(content_delta="Mock", role="assistant", model=request.model, done=False)
        yield InternalStreamChunk(content_delta=" response", model=request.model, done=False)
        yield InternalStreamChunk(content_delta="", model=request.model, done=True, finish_reason="stop")

    backend.chat_stream = _mock_stream
    return backend


class TestPassthroughHandlerBasic:
    """Basic passthrough handler behavior."""

    async def test_handle_simple_request(self):
        backend = _make_backend()
        handler = PassthroughHandler(backend=backend)
        request = _make_request("Hello")
        response = await handler.handle(request)
        assert response.message.content == "Mock response"
        assert response.finish_reason == "stop"

    async def test_handle_stream_yields_chunks(self):
        backend = _make_backend()
        handler = PassthroughHandler(backend=backend)
        request = _make_request("Hello", stream=True)
        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)
        content = "".join(c.content_delta for c in chunks if c.content_delta)
        assert "Mock" in content
        assert "response" in content

    async def test_stream_preserves_chunk_order(self):
        backend = _make_backend()
        handler = PassthroughHandler(backend=backend)
        request = _make_request("Hello", stream=True)
        deltas = []
        async for chunk in handler.handle_stream(request):
            if chunk.content_delta:
                deltas.append(chunk.content_delta)
        assert deltas == ["Mock", " response"]

    async def test_handler_without_tools_no_injection(self):
        backend = _make_backend()
        handler = PassthroughHandler(backend=backend)
        assert handler._tool_registry is None
        assert handler._enabled_tools == []

    async def test_handler_with_empty_tools(self):
        backend = _make_backend()
        registry = MagicMock()
        handler = PassthroughHandler(backend=backend, tool_registry=registry, enabled_tools=[])
        resolved = handler._resolve_tools()
        assert resolved == []


class TestPassthroughToolInjection:
    """Tool injection and resolution behavior."""

    async def test_resolve_tools_returns_matching(self):
        backend = _make_backend()
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        registry = MagicMock()
        registry.resolve.return_value = mock_tool
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        tools = handler._resolve_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    async def test_resolve_tools_skips_unknown(self):
        backend = _make_backend()
        registry = MagicMock()
        registry.resolve.return_value = None
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["nonexistent_tool"],
        )
        tools = handler._resolve_tools()
        assert tools == []


class TestAutoInvokeGating:
    """Direct ("auto-invoke when enabled") tool execution must be suppressible.

    Chat keeps it ON (per-tool button toggle = intent). Voice flips it OFF
    because it enables tools via a blanket ['all'] sentinel — otherwise an
    auto-invoke tool like youtube fires on every spoken turn (the reported
    "agent returns a video every message" bug).
    """

    def _handler_with_auto_invoke_tool(self):
        backend = _make_backend()
        tool = MagicMock()
        tool.name = "youtube"
        tool.auto_invoke_when_enabled = True
        registry = MagicMock()
        registry.resolve.return_value = tool
        return PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["youtube"],
        )

    async def test_auto_invoke_on_by_default(self):
        """Chat path: an enabled auto-invoke tool is collected for direct run."""
        handler = self._handler_with_auto_invoke_tool()
        assert handler._auto_invoke_enabled is True
        direct = handler._collect_direct_invoke_tools()
        assert [t.name for t in direct] == ["youtube"]

    async def test_auto_invoke_off_suppresses_direct_tools(self):
        """Voice path: with auto-invoke disabled, nothing is auto-fired even
        though the tool is enabled (it stays available for LLM selection)."""
        handler = self._handler_with_auto_invoke_tool()
        handler._auto_invoke_enabled = False
        assert handler._collect_direct_invoke_tools() == []
        # The tool is still resolvable for the LLM to choose explicitly.
        assert [t.name for t in handler._resolve_tools()] == ["youtube"]


class TestAutoCapabilityTools:
    """Auto mode (no explicit tools) exposes the SSOS lookup capabilities as
    NATIVE Tool objects, replacing the [[tool:NAME]] soft-trigger text protocol
    that native-tool-trained models reliably missed."""

    def _registry_resolving_anything(self):
        registry = MagicMock()

        def _resolve(name):
            tool = MagicMock()
            tool.name = name
            return tool

        registry.resolve.side_effect = _resolve
        return registry

    def test_auto_tools_text_runs_image_inline_builders_propose(self):
        """Text chat (default, _gate_heavy_tools=False): the request was typed,
        so image_generation is exposed as the REAL tool (runs inline → live
        card + inline result). The multi-step builders stay propose-only
        proxies so an inferred multi-minute build still confirms first."""
        from augmentum.modes.passthrough.gated_proxy import GatedProxyTool
        backend = _make_backend()
        registry = self._registry_resolving_anything()
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )
        tools = handler._resolve_auto_capability_tools()
        names = {t.name for t in tools}
        # Every capability is on the menu (lookups + all 5 gated).
        expected = (
            {c.tool for c in SSOSOrchestrator.lookup_capabilities()}
            | {c.tool for c in SSOSOrchestrator.gated_capabilities()}
        )
        assert names == expected
        assert "web_search" in names and "image_search" in names
        assert "image_generation" in names
        # image_generation is a REAL tool here, not a propose-only proxy.
        proxies = {t.name for t in tools if isinstance(t, GatedProxyTool)}
        assert "image_generation" not in proxies
        # The heavy builders remain proxies (inferred build confirms first).
        assert proxies == {
            "build_application", "create_ebook",
            "create_presentation", "create_document",
        }

    def test_auto_tools_voice_keeps_image_as_proxy(self):
        """Voice (_gate_heavy_tools=True): STT can misfire, so image_generation
        is exposed as a propose-only proxy and confirmed before it runs.

        Chart/spreadsheet are the exception: they stay INLINE even on voice.
        The reason to gate on voice is the cost of acting on a mis-heard
        request, and these are second-scale, in-process and non-destructive —
        a wrongly-drawn chart costs a moment, while a confirm chip mid-
        conversation costs the flow. See _NEVER_GATED_CAPABILITIES.
        """
        from augmentum.modes.passthrough.gated_proxy import GatedProxyTool
        from augmentum.modes.passthrough.handler import (
            _NEVER_GATED_CAPABILITIES,
        )
        backend = _make_backend()
        registry = self._registry_resolving_anything()
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )
        handler._gate_heavy_tools = True
        tools = handler._resolve_auto_capability_tools()
        proxies = {t.name for t in tools if isinstance(t, GatedProxyTool)}
        expected = {
            c.tool for c in SSOSOrchestrator.gated_capabilities()
        } - _NEVER_GATED_CAPABILITIES
        assert proxies == expected
        assert "image_generation" in proxies
        assert not (proxies & _NEVER_GATED_CAPABILITIES)
        # …and they're still on the menu, just as the real tool.
        assert {t.name for t in tools} >= _NEVER_GATED_CAPABILITIES

    def test_auto_tools_empty_without_registry(self):
        """No registry → no SSOS → no auto tools (pure passthrough)."""
        handler = PassthroughHandler(backend=_make_backend())
        assert handler._resolve_auto_capability_tools() == []

    def test_auto_tools_skip_unresolvable(self):
        """Capabilities whose tool isn't registered are dropped, not faked."""
        backend = _make_backend()
        registry = MagicMock()
        registry.resolve.return_value = None  # nothing resolves
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )
        assert handler._resolve_auto_capability_tools() == []


class TestGatedToolIntercept:
    """Gating PROPOSES a heavy tool (confirmation chip) instead of running it,
    to guard against MISREAD intent. Policy (``_should_gate_capability``):

    * Explicitly enabled tool → never gate (the button is consent).
    * image_generation → gate only under voice (uncertain STT); text runs it
      inline (live card + inline result, the pre-offer-substrate behavior).
    * Multi-step builders → always gate an INFERRED call (multi-minute work).
    """

    def _handler(self, *, gate_heavy=False, enabled_tools=None):
        backend = _make_backend()
        registry = MagicMock()

        def _resolve(name):
            tool = MagicMock()
            tool.name = name
            tool.description = "real description"
            tool.model_hint = "real hint"
            return tool

        registry.resolve.side_effect = _resolve
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry,
            enabled_tools=enabled_tools or [],
        )
        handler._gate_heavy_tools = gate_heavy
        return handler

    # --- arg-parsing mechanics (policy-independent: builders always gate) ---

    def test_first_gated_matches_canonical_arg(self):
        handler = self._handler()
        out = handler._first_gated(
            [("build_application", {"description": "a todo app"}, "tc1")]
        )
        assert out is not None
        cap, brief = out
        assert cap.tool == "build_application"
        assert brief == "a todo app"

    def test_first_gated_falls_back_to_first_string_arg(self):
        """Model used a non-canonical arg name — still recover the brief."""
        handler = self._handler()
        cap, brief = handler._first_gated(
            [("build_application", {"text": "a notes app"}, "tc1")]
        )
        assert cap.tool == "build_application"
        assert brief == "a notes app"

    def test_first_gated_ignores_lookups_and_empty(self):
        handler = self._handler()
        assert handler._first_gated([("web_search", {"query": "x"}, "t")]) is None
        assert handler._first_gated([]) is None

    # --- the modality policy itself ---

    def test_image_gen_not_gated_in_text(self):
        """Typed request = explicit intent → image_generation runs inline."""
        handler = self._handler(gate_heavy=False)
        assert handler._first_gated(
            [("image_generation", {"prompt": "a red fox"}, "tc1")]
        ) is None

    def test_image_gen_gated_in_voice(self):
        """Uncertain STT → image_generation is proposed, not run."""
        handler = self._handler(gate_heavy=True)
        out = handler._first_gated(
            [("image_generation", {"prompt": "a red fox"}, "tc1")]
        )
        assert out is not None and out[0].tool == "image_generation"

    def test_explicitly_enabled_tool_never_gated(self):
        """The user ticked the tool — consent given, run inline even in voice."""
        handler = self._handler(
            gate_heavy=True, enabled_tools=["image_generation"],
        )
        assert handler._first_gated(
            [("image_generation", {"prompt": "a red fox"}, "tc1")]
        ) is None

    def test_builders_gate_on_inferred_call_regardless_of_modality(self):
        """Multi-minute builds confirm first even in text when only inferred."""
        handler = self._handler(gate_heavy=False)
        out = handler._first_gated(
            [("build_application", {"description": "a shop"}, "tc1")]
        )
        assert out is not None and out[0].tool == "build_application"

    async def test_gated_response_proposes_offer_not_execution(self):
        """The gated path surfaces a confirmation offer and returns a warm
        lead-in — it does NOT execute the heavy tool."""
        handler = self._handler()
        handler._ssos.propose_gated = AsyncMock(return_value=True)
        cap = next(
            c for c in SSOSOrchestrator.gated_capabilities()
            if c.tool == "image_generation"
        )
        resp = await handler._gated_response(cap, "a red fox", "m")
        # Offer surfaced; tool never run.
        handler._ssos.propose_gated.assert_awaited_once()
        assert "image" in (resp.message.content or "").lower()
        assert resp.finish_reason == "stop"

    def test_proxy_schema_is_single_primary_arg(self):
        from augmentum.modes.passthrough.gated_proxy import (
            build_gated_proxy_tools,
        )
        registry = MagicMock()

        def _resolve(name):
            t = MagicMock()
            t.name = name
            t.description = "d"
            t.model_hint = ""
            return t

        registry.resolve.side_effect = _resolve
        proxies = build_gated_proxy_tools(
            SSOSOrchestrator.gated_capabilities(), registry,
        )
        names = {p.name for p in proxies}
        assert names == {c.tool for c in SSOSOrchestrator.gated_capabilities()}
        assert all(getattr(p, "is_gated_proxy", False) for p in proxies)
        img = next(p for p in proxies if p.name == "image_generation")
        props = img.input_schema["properties"]
        assert list(props.keys()) == ["prompt"]
        assert img.input_schema["required"] == ["prompt"]

    def test_proxy_skipped_when_tool_unavailable(self):
        """No image provider → image_generation tool absent → no proxy (we
        don't advertise what can't run)."""
        from augmentum.modes.passthrough.gated_proxy import (
            build_gated_proxy_tools,
        )
        registry = MagicMock()
        registry.resolve.return_value = None
        assert build_gated_proxy_tools(
            SSOSOrchestrator.gated_capabilities(), registry,
        ) == []

    async def test_native_streaming_intercepts_gated_call(self, monkeypatch):
        """Voice path (uncertain STT, _gate_heavy_tools=True): a model emits a
        NATIVE image_generation tool_call → it is intercepted into a
        confirmation offer, the lead-in streams, and the heavy tool is NEVER
        executed (guarding against a stray STT-triggered generation)."""
        from augmentum.config import settings as _settings
        from augmentum.modes.passthrough.gated_proxy import (
            build_gated_proxy_tools,
        )
        monkeypatch.setattr(
            _settings, "uarf_tool_tier_override", "native", raising=False,
        )

        async def _stream(req):
            yield InternalStreamChunk(
                content_delta="", model="m", done=False,
                augmentum={"tool_calls": [{
                    "index": 0, "id": "tc1", "type": "function",
                    "function": {
                        "name": "image_generation",
                        "arguments": '{"prompt": "a red fox at dusk"}',
                    },
                }]},
            )
            yield InternalStreamChunk(
                content_delta="", model="m", done=True,
                finish_reason="tool_calls",
            )

        backend = _make_backend()
        backend.chat_stream = _stream
        registry = MagicMock()

        def _resolve(name):
            tool = MagicMock()
            tool.name = name
            tool.description = "d"
            tool.model_hint = ""
            tool.execute = AsyncMock()  # must NOT be awaited
            return tool

        registry.resolve.side_effect = _resolve
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )
        handler._gate_heavy_tools = True  # voice modality
        handler._ssos.propose_gated = AsyncMock(return_value=True)

        proxies = build_gated_proxy_tools(
            [c for c in SSOSOrchestrator.gated_capabilities()
             if c.tool == "image_generation"],
            registry,
        )
        req = _make_request("make me an image of a fox", stream=True)
        out_q: asyncio.Queue = asyncio.Queue()
        await handler._resolve_tool_calls_streaming(
            req, proxies, output_queue=out_q,
        )
        chunks = []
        while not out_q.empty():
            chunks.append(out_q.get_nowait())
        content = "".join(c.content_delta for c in chunks if c.content_delta)

        # The offer surfaced and the heavy tool was never run.
        handler._ssos.propose_gated.assert_awaited_once()
        assert content.strip()  # a lead-in was streamed
        assert "confirm" in content.lower() or "image" in content.lower()
        # Stream terminated cleanly.
        assert any(c.done for c in chunks)


class TestCallDontNarrateDirectives:
    """Rendering tools share one failure mode: the model NARRATES the content
    (an image "she paints a sunrise…", a chart as a markdown table) instead of
    CALLING the tool, so nothing renders. When such a tool is on the menu we
    inject a call-don't-describe directive — originally the fix for the
    deepseek-v4-flash 'described the image, never called the tool' case, now
    table-driven so every rendering tool gets the same treatment."""

    _IMAGE = "[IMAGE TOOL]"
    _CHART = "[CHART TOOL]"

    def _tool(self, name):
        t = MagicMock()
        t.name = name
        return t

    def _req(self, system="You are Becca, a companion."):
        return InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content=system),
                Message(role="user", content="show me the sunrise"),
            ],
        )

    @pytest.mark.parametrize(("tool_name", "mark"), [
        ("image_generation", _IMAGE),
        ("create_chart", _CHART),
    ])
    def test_directive_added_when_tool_present(self, tool_name, mark):
        req = self._req()
        PassthroughHandler._inject_call_dont_narrate_directives(
            req, [self._tool("web_search"), self._tool(tool_name)],
        )
        sys_msg = next(m for m in req.messages if m.role == "system")
        assert mark in sys_msg.content
        assert "You are Becca" in sys_msg.content  # appended, not replaced

    def test_only_the_present_tools_directive_is_added(self):
        """A chart turn must not carry the image directive (or vice versa) —
        every unused directive is wasted prefix on every turn."""
        req = self._req()
        PassthroughHandler._inject_call_dont_narrate_directives(
            req, [self._tool("create_chart")],
        )
        sys_msg = next(m for m in req.messages if m.role == "system")
        assert self._CHART in sys_msg.content
        assert self._IMAGE not in sys_msg.content

    def test_both_added_when_both_present(self):
        req = self._req()
        PassthroughHandler._inject_call_dont_narrate_directives(
            req, [self._tool("image_generation"), self._tool("create_chart")],
        )
        sys_msg = next(m for m in req.messages if m.role == "system")
        assert self._IMAGE in sys_msg.content
        assert self._CHART in sys_msg.content

    def test_not_added_without_a_rendering_tool(self):
        req = self._req()
        PassthroughHandler._inject_call_dont_narrate_directives(
            req, [self._tool("web_search"), self._tool("calculator")],
        )
        content = req.messages[0].content or ""
        assert self._IMAGE not in content
        assert self._CHART not in content

    @pytest.mark.parametrize(("tool_name", "mark"), [
        ("image_generation", _IMAGE),
        ("create_chart", _CHART),
    ])
    def test_idempotent_across_tool_loop_iterations(self, tool_name, mark):
        req = self._req()
        tools = [self._tool(tool_name)]
        PassthroughHandler._inject_call_dont_narrate_directives(req, tools)
        PassthroughHandler._inject_call_dont_narrate_directives(req, tools)
        sys_msg = next(m for m in req.messages if m.role == "system")
        assert sys_msg.content.count(mark) == 1

    def test_inserts_system_message_when_none_exists(self):
        req = InternalChatRequest(
            model="m", messages=[Message(role="user", content="show me")],
        )
        PassthroughHandler._inject_call_dont_narrate_directives(
            req, [self._tool("image_generation")],
        )
        assert req.messages[0].role == "system"
        assert self._IMAGE in req.messages[0].content

    def test_every_never_gated_renderer_has_a_directive(self):
        """create_chart is exposed inline in Auto, so it needs the directive.

        create_spreadsheet deliberately does NOT: a spreadsheet is a file the
        user asked to keep, not something the model should proactively produce,
        so its description carries the routing rule and no prefix is spent.
        """
        from augmentum.modes.passthrough.handler import (
            _CALL_DONT_NARRATE_DIRECTIVES,
        )
        named = {n for n, _ in _CALL_DONT_NARRATE_DIRECTIVES}
        assert "create_chart" in named
        assert "create_spreadsheet" not in named
        # No duplicate entries — a repeat would double the injected prefix.
        assert len(named) == len(_CALL_DONT_NARRATE_DIRECTIVES)


class TestAutoCapabilitySelfModel:
    """Auto mode injects a STABLE capability self-model so the model never
    denies a capability just because the per-turn relevance filter dropped that
    tool's schema this turn (the "I can't make images on a non-image turn"
    flicker Matt hit when asking 'surprise me')."""

    def _handler(self):
        backend = _make_backend()
        registry = MagicMock()

        def _resolve(name):
            tool = MagicMock()
            tool.name = name
            tool.description = "d"
            tool.model_hint = ""
            return tool

        registry.resolve.side_effect = _resolve
        return PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )

    def test_self_model_advertises_image_gen_and_preserves_prompt(self):
        handler = self._handler()
        req = InternalChatRequest(
            model="m",
            messages=[Message(role="system", content="You are Becca.")],
        )
        tools = handler._resolve_auto_capability_tools()
        handler._inject_auto_capability_self_model(req, tools)
        sys = next(m for m in req.messages if m.role == "system")
        assert "generate images" in sys.content.lower()
        assert "Becca" in sys.content  # original persona preserved

    def test_self_model_creates_system_message_when_absent(self):
        handler = self._handler()
        req = InternalChatRequest(
            model="m", messages=[Message(role="user", content="hi")],
        )
        tools = handler._resolve_auto_capability_tools()
        handler._inject_auto_capability_self_model(req, tools)
        assert req.messages[0].role == "system"
        assert "generate images" in req.messages[0].content.lower()

    def test_self_model_not_doubled_on_repeat(self):
        handler = self._handler()
        req = InternalChatRequest(
            model="m", messages=[Message(role="system", content="x")],
        )
        tools = handler._resolve_auto_capability_tools()
        handler._inject_auto_capability_self_model(req, tools)
        handler._inject_auto_capability_self_model(req, tools)
        sys = next(m for m in req.messages if m.role == "system")
        assert sys.content.lower().count("generate images") == 1

    def test_self_model_omits_image_gen_without_provider(self):
        """No image tool resolves → the self-model must NOT claim image gen."""
        backend = _make_backend()
        registry = MagicMock()

        def _resolve(name):
            if name == "image_generation":
                return None  # no image provider on this install
            tool = MagicMock()
            tool.name = name
            tool.description = "d"
            tool.model_hint = ""
            return tool

        registry.resolve.side_effect = _resolve
        handler = PassthroughHandler(
            backend=backend, tool_registry=registry, enabled_tools=[],
        )
        req = InternalChatRequest(
            model="m", messages=[Message(role="system", content="x")],
        )
        tools = handler._resolve_auto_capability_tools()
        handler._inject_auto_capability_self_model(req, tools)
        sys = next(m for m in req.messages if m.role == "system")
        assert "generate images" not in sys.content.lower()


class TestPassthroughNativeToolFiltering:
    """Native tool_calls must be filtered by the user's enabled tool set.

    Regression: capable models hallucinate well-known tool names (web_search,
    image_generation) even when the injected schema only lists a curated
    subset. Without filtering, the registry resolved any name globally and
    the phantom call ran anyway — bypassing the user's button selection.
    """

    def _native_response(self, *names: str) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": f"call_{i}", "function": {"name": n, "arguments": "{}"}}
                    for i, n in enumerate(names)
                ],
            ),
            model="test-model",
            finish_reason="tool_calls",
        )

    async def test_native_call_for_disabled_tool_is_dropped(self):
        from augmentum.modes.analytical.tool_calling import ToolCallingTier

        backend = _make_backend()
        registry = MagicMock()
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["image_search"],
        )
        image_search = MagicMock()
        image_search.name = "image_search"

        response = self._native_response("image_generation")

        with patch(
            "augmentum.modes.analytical.tool_calling.select_tier",
            return_value=ToolCallingTier.NATIVE,
        ):
            calls = handler._parse_tool_calls(response, [image_search])

        assert calls == [], (
            "image_generation must be dropped — only image_search is enabled"
        )

    async def test_native_call_for_enabled_tool_is_kept(self):
        from augmentum.modes.analytical.tool_calling import ToolCallingTier

        backend = _make_backend()
        registry = MagicMock()
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["image_search"],
        )
        image_search = MagicMock()
        image_search.name = "image_search"

        response = self._native_response("image_search")

        with patch(
            "augmentum.modes.analytical.tool_calling.select_tier",
            return_value=ToolCallingTier.NATIVE,
        ):
            calls = handler._parse_tool_calls(response, [image_search])

        assert len(calls) == 1
        assert calls[0][0] == "image_search"

    async def test_mixed_native_calls_keep_only_enabled(self):
        from augmentum.modes.analytical.tool_calling import ToolCallingTier

        backend = _make_backend()
        registry = MagicMock()
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["image_search"],
        )
        image_search = MagicMock()
        image_search.name = "image_search"

        response = self._native_response(
            "web_search", "image_search", "image_generation",
        )

        with patch(
            "augmentum.modes.analytical.tool_calling.select_tier",
            return_value=ToolCallingTier.NATIVE,
        ):
            calls = handler._parse_tool_calls(response, [image_search])

        assert [c[0] for c in calls] == ["image_search"], (
            "Phantom web_search and image_generation must be filtered out"
        )


class TestSSOSStructuredSearch:
    """SSOS web_search merge uses structured metadata, not formatted text.

    Regression: the old implementation re-parsed the rendered output
    (`URL: ...` line prefixes) and renumbered with regex. If web_search
    ever changed its formatter, dedup silently broke. The new path reads
    `result.metadata["results"]` directly.
    """

    def _search_result(self, *entries):
        """Build a ToolResult mimicking web_search output."""
        from augmentum.tools.base import ToolResult

        # Render text from entries (parallel to _render_search_results_text)
        lines = []
        for i, e in enumerate(entries, 1):
            lines.append(f"[{i}] {e['title']}")
            lines.append(f"    URL: {e['url']}")
            lines.append(f"    {e.get('snippet', '')}")
            lines.append("")
        return ToolResult(
            success=True,
            output="\n".join(lines).rstrip(),
            metadata={"results": list(entries)},
        )

    async def test_search_dedupes_by_url_via_metadata(self):
        from augmentum.tools.intent import QueryIntent
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator

        # Two parallel queries return overlapping URLs.
        result_a = self._search_result(
            {"title": "A1", "url": "https://example.com/a", "snippet": "first"},
            {"title": "B1", "url": "https://example.com/b", "snippet": "second"},
        )
        result_b = self._search_result(
            {"title": "A2-dupe", "url": "https://example.com/a", "snippet": "dup"},
            {"title": "C1", "url": "https://example.com/c", "snippet": "third"},
        )

        web_search = MagicMock()
        web_search.name = "web_search"
        web_search.execute = AsyncMock(side_effect=[result_a, result_b])
        registry = MagicMock()
        registry.get.return_value = web_search

        orch = SSOSOrchestrator(registry)
        intent = QueryIntent(action="search", confidence=0.9)

        # Patch query formulator to return 2 queries (forcing 2 parallel calls)
        with patch(
            "augmentum.modes.passthrough.orchestrator.formulate_queries",
            return_value=["q1", "q2"],
        ):
            text, merged = await orch._execute_search(intent, "user msg")

        # Dedup by URL
        urls = [r["url"] for r in merged]
        assert urls == [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ], "URLs must be deduped to first-seen order"

        # First entry is the FIRST result for example.com/a, not the dupe.
        assert merged[0]["title"] == "A1"
        # Renumbering is contiguous from 1.
        assert "[1] A1" in text
        assert "[2] B1" in text
        assert "[3] C1" in text
        # No phantom [4] from the duplicate.
        assert "[4]" not in text

    async def test_search_emits_tool_events(self):
        from augmentum.tools.intent import QueryIntent
        from augmentum.modes.passthrough.orchestrator import (
            SSOSOrchestrator, _EventEmitter,
        )

        result = self._search_result(
            {"title": "T", "url": "https://x.test/1", "snippet": "s"},
        )
        web_search = MagicMock()
        web_search.execute = AsyncMock(return_value=result)
        registry = MagicMock()
        registry.get.return_value = web_search

        orch = SSOSOrchestrator(registry)
        intent = QueryIntent(action="search", confidence=0.9)

        starts: list[tuple] = []
        completes: list[tuple] = []

        async def _on_start(tc_id, name, args):
            starts.append((tc_id, name, args))

        async def _on_complete(name, success, snippet, meta, tc_id, dur_ms):
            completes.append((name, success, meta))

        emit = _EventEmitter(
            on_tool_start=_on_start, on_tool_complete=_on_complete,
        )

        with patch(
            "augmentum.modes.passthrough.orchestrator.formulate_queries",
            return_value=["q1"],
        ):
            await orch._execute_search(intent, "user msg", emit=emit)

        assert len(starts) == 1
        assert starts[0][1] == "web_search"

        assert len(completes) == 1
        name, success, meta = completes[0]
        assert name == "web_search"
        assert success is True
        # Structured results MUST surface to the UI via result_metadata
        # so source cards can render without re-parsing prose.
        assert "results" in meta
        assert meta["results"][0]["url"] == "https://x.test/1"
        assert meta["result_count"] == 1


class TestPassthroughAutoTools:
    """SSOS orchestrator and auto-tools suppression."""

    async def test_ssos_created_when_registry_provided(self):
        backend = _make_backend()
        registry = MagicMock()
        handler = PassthroughHandler(backend=backend, tool_registry=registry)
        assert handler._ssos is not None

    async def test_ssos_not_created_without_registry(self):
        backend = _make_backend()
        handler = PassthroughHandler(backend=backend)
        assert handler._ssos is None


def _make_app_state(autoTools_value: str | None = "true"):
    """Build an app_state stub with a settings_store that returns
    ``autoTools_value`` for ``ui.autoTools`` lookups and None for everything
    else. Pass ``None`` to simulate an unset preference.
    """
    store = MagicMock()
    async def _get(uid, key):
        if key == "ui.autoTools":
            return autoTools_value
        return None
    store.get_user_or_global = AsyncMock(side_effect=_get)
    state = MagicMock()
    state.settings_store = store
    return state


class TestSSOSEnablement:
    """Per-user gating of SSOS via ui.autoTools."""

    async def test_disabled_when_no_user_id(self):
        orch = SSOSOrchestrator(MagicMock(), app_state=_make_app_state("true"))
        assert await orch.is_enabled() is False

    async def test_disabled_when_no_app_state(self):
        orch = SSOSOrchestrator(MagicMock(), user_id="u1")
        assert await orch.is_enabled() is False

    async def test_disabled_when_pref_unset(self):
        orch = SSOSOrchestrator(
            MagicMock(), user_id="u1", app_state=_make_app_state(None),
        )
        assert await orch.is_enabled() is False

    async def test_disabled_when_pref_false(self):
        orch = SSOSOrchestrator(
            MagicMock(), user_id="u1", app_state=_make_app_state("false"),
        )
        assert await orch.is_enabled() is False

    async def test_enabled_when_pref_true(self):
        orch = SSOSOrchestrator(
            MagicMock(), user_id="u1", app_state=_make_app_state("true"),
        )
        assert await orch.is_enabled() is True

    async def test_two_users_independent(self):
        """A user with the pref off must not see SSOS fire even when another
        user on the same install has it on."""
        store = MagicMock()
        async def _get(uid, key):
            if key != "ui.autoTools":
                return None
            return "true" if uid == "u_on" else "false"
        store.get_user_or_global = AsyncMock(side_effect=_get)
        state = MagicMock()
        state.settings_store = store
        on = SSOSOrchestrator(MagicMock(), user_id="u_on", app_state=state)
        off = SSOSOrchestrator(MagicMock(), user_id="u_off", app_state=state)
        assert await on.is_enabled() is True
        assert await off.is_enabled() is False

    async def test_try_orchestrate_short_circuits_when_disabled(self):
        """When the pref is off, try_orchestrate must not even classify intent."""
        registry = MagicMock()
        orch = SSOSOrchestrator(
            registry, user_id="u1", app_state=_make_app_state("false"),
        )
        with patch("augmentum.modes.passthrough.orchestrator.classify_intent") as cls:
            request = InternalChatRequest(
                model="test-model",
                messages=[Message(role="user", content="search for cats")],
            )
            result = await orch.try_orchestrate(request)
            assert result is None
            cls.assert_not_called()


class TestSSOSOrchestrator:
    """SSOS orchestrator intent-driven behavior."""

    async def test_try_orchestrate_returns_none_on_empty_message(self):
        registry = MagicMock()
        orch = SSOSOrchestrator(
            registry, user_id="u1", app_state=_make_app_state("true"),
        )
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="")],
        )
        result = await orch.try_orchestrate(request)
        assert result is None

    async def test_try_orchestrate_returns_none_on_no_user_message(self):
        registry = MagicMock()
        orch = SSOSOrchestrator(
            registry, user_id="u1", app_state=_make_app_state("true"),
        )
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="system", content="You are helpful")],
        )
        result = await orch.try_orchestrate(request)
        assert result is None

    async def test_build_synthesis_request_includes_search_tag(self):
        """When action is search, the synthesis request contains <search_results> tags."""
        registry = MagicMock()
        orch = SSOSOrchestrator(tool_registry=registry)
        original = _make_request("What is Python?")
        from augmentum.tools.intent import QueryIntent
        intent = QueryIntent(action="search", confidence=0.9)
        result = orch._build_synthesis_request(original, "Some results", intent)
        assert result.tools is None
        last_msg = result.messages[-1].content
        assert "<search_results>" in last_msg

    async def test_build_synthesis_request_strips_tools(self):
        """Synthesis request should have tools=None."""
        registry = MagicMock()
        orch = SSOSOrchestrator(tool_registry=registry)
        original = _make_request("Calculate 2+2")
        from augmentum.tools.intent import QueryIntent
        intent = QueryIntent(action="calculate", confidence=0.9)
        result = orch._build_synthesis_request(original, "2+2 = 4", intent)
        assert result.tools is None

    async def test_build_synthesis_request_preserves_model(self):
        registry = MagicMock()
        orch = SSOSOrchestrator(tool_registry=registry)
        original = _make_request("query", model="custom-model")
        from augmentum.tools.intent import QueryIntent
        intent = QueryIntent(action="datetime", confidence=0.9)
        result = orch._build_synthesis_request(original, "2024-01-01", intent)
        assert result.model == "custom-model"


class TestSoftTriggerParsing:
    """parse_trigger + hint building for model-initiated capabilities."""

    def test_parse_valid_marker(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        cap, args = SSOSOrchestrator.parse_trigger("[[tool:web_search]] best keyboards 2026")
        assert cap.name == "web_search"
        assert args == "best keyboards 2026"

    def test_parse_leading_whitespace(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        parsed = SSOSOrchestrator.parse_trigger("   [[tool:wikipedia]]   Ada Lovelace  ")
        assert parsed is not None
        cap, args = parsed
        assert cap.name == "wikipedia"
        assert args == "Ada Lovelace"

    def test_parse_no_marker_returns_none(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        assert SSOSOrchestrator.parse_trigger("Sure, here's the answer.") is None

    def test_parse_unknown_tool_returns_none(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        assert SSOSOrchestrator.parse_trigger("[[tool:format_drive]] do it") is None

    def test_parse_empty_args_returns_none(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        assert SSOSOrchestrator.parse_trigger("[[tool:web_search]]") is None
        assert SSOSOrchestrator.parse_trigger("[[tool:web_search]]   ") is None

    def test_hint_lists_tools_and_protocol(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        registry = MagicMock()
        registry.get.return_value = None  # force fallback_hint
        hint = SSOSOrchestrator(registry).build_soft_trigger_hint()
        assert "[[tool:NAME]]" in hint
        for name in ("web_search", "wikipedia", "youtube", "image_search"):
            assert name in hint

    def test_hint_prefers_tool_model_hint(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        tool = MagicMock()
        tool.model_hint = "CANARY-HINT for searching"
        registry = MagicMock()
        registry.get.return_value = tool
        hint = SSOSOrchestrator(registry).build_soft_trigger_hint()
        assert "CANARY-HINT for searching" in hint


class TestSoftTriggerExecution:
    """run_named_tool + synthesis request building."""

    def _tool(self, result):
        tool = MagicMock()
        tool.model_hint = "h"
        tool.timeout = 30.0
        tool.execute = AsyncMock(return_value=result)
        return tool

    async def test_run_named_tool_success_emits_events(self):
        from augmentum.tools.base import ToolResult
        from augmentum.modes.passthrough.orchestrator import (
            SSOSOrchestrator, ModelCapability, _EventEmitter,
        )
        result = ToolResult(
            success=True, output="results text",
            metadata={"results": [{"url": "u"}], "_private": "x"},
        )
        registry = MagicMock()
        registry.get.return_value = self._tool(result)
        orch = SSOSOrchestrator(registry)
        cap = ModelCapability(
            name="web_search", tool="web_search", kind="lookup",
            primary_arg="query", fallback_hint="h", synthesis="search",
        )

        completes = []
        emit = _EventEmitter(
            on_tool_start=AsyncMock(),
            on_tool_complete=AsyncMock(side_effect=lambda *a, **k: completes.append((a, k))),
        )
        text, meta = await orch.run_named_tool(cap, "best keyboards", emit=emit)
        assert text == "results text"
        # Underscore-prefixed plumbing keys are stripped from UI metadata.
        assert "results" in meta and "_private" not in meta

    async def test_run_named_tool_failure_returns_none(self):
        from augmentum.tools.base import ToolResult
        from augmentum.modes.passthrough.orchestrator import (
            SSOSOrchestrator, ModelCapability,
        )
        registry = MagicMock()
        registry.get.return_value = self._tool(ToolResult(success=False, error="boom"))
        orch = SSOSOrchestrator(registry)
        cap = ModelCapability(
            name="wikipedia", tool="wikipedia", kind="lookup",
            primary_arg="query", fallback_hint="h", synthesis="wikipedia",
        )
        text, meta = await orch.run_named_tool(cap, "Ada Lovelace")
        assert text is None
        assert meta == {}

    async def test_tool_synthesis_request_keys_on_capability(self):
        from augmentum.modes.passthrough.orchestrator import (
            SSOSOrchestrator, ModelCapability,
        )
        orch = SSOSOrchestrator(MagicMock())
        original = _make_request("who was Ada Lovelace?")
        cap = ModelCapability(
            name="wikipedia", tool="wikipedia", kind="lookup",
            primary_arg="query", fallback_hint="h", synthesis="wikipedia",
        )
        req = orch.build_tool_synthesis_request(original, cap, "article text")
        assert req.tools is None
        assert "<wikipedia>" in req.messages[-1].content

    async def test_search_intent_no_longer_orchestrated(self):
        """Retired regex search: try_orchestrate returns None for a search
        intent so the model-initiated pass takes over."""
        from augmentum.tools.intent import QueryIntent
        registry = MagicMock()
        orch = SSOSOrchestrator(
            registry, user_id="u1", app_state=_make_app_state("true"),
        )
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="latest news on mars")],
        )
        with patch(
            "augmentum.modes.passthrough.orchestrator.classify_intent",
            return_value=QueryIntent(action="search", confidence=0.9),
        ):
            assert await orch.try_orchestrate(request) is None


class TestSoftTriggerStream:
    """Handler streaming soft-trigger pass: marker vs no-marker."""

    def _handler(self, stream_fn):
        backend = _make_backend()
        backend.chat_stream = stream_fn
        registry = MagicMock()
        handler = PassthroughHandler(backend=backend, tool_registry=registry)
        return handler

    async def test_no_marker_streams_verbatim_single_call(self):
        calls = []

        async def _stream(req):
            calls.append(req)
            yield InternalStreamChunk(content_delta="Just a plain answer.", model="m", done=False)
            yield InternalStreamChunk(content_delta="", model="m", done=True, finish_reason="stop")

        handler = self._handler(_stream)
        req = _make_request("how are you?", stream=True)
        chunks = [c async for c in handler._soft_trigger_stream(req)]
        content = "".join(c.content_delta for c in chunks if c.content_delta)
        assert "Just a plain answer." in content
        # No tool events fired, and exactly ONE backend call (no synthesis pass).
        assert not any((c.augmentum or {}).get("tool_start") for c in chunks)
        assert len(calls) == 1

    async def test_marker_runs_tool_then_synthesizes(self):
        from augmentum.tools.base import ToolResult

        async def _stream(req):
            last = req.messages[-1].content
            if "<search_results>" in last:
                yield InternalStreamChunk(content_delta="Synthesized answer.", model="m", done=True, finish_reason="stop")
            else:
                yield InternalStreamChunk(content_delta="[[tool:web_search]] best keyboards\n", model="m", done=False)
                yield InternalStreamChunk(content_delta="LEAK-should-be-suppressed", model="m", done=True, finish_reason="stop")

        handler = self._handler(_stream)
        # web_search tool returns a result
        tool = MagicMock()
        tool.model_hint = "h"
        tool.timeout = 30.0
        tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="result block", metadata={"results": [{"url": "u"}]},
        ))
        handler._tool_registry.get.return_value = tool

        req = _make_request("best mechanical keyboards 2026", stream=True)
        chunks = [c async for c in handler._soft_trigger_stream(req)]
        content = "".join(c.content_delta for c in chunks if c.content_delta)

        # Marker suppressed; trailing decide tokens dropped; synthesis streamed.
        assert "[[tool:web_search]]" not in content
        assert "LEAK-should-be-suppressed" not in content
        assert "Synthesized answer." in content
        # Tool events surfaced to the UI under the "auto" phase.
        starts = [c.augmentum["tool_start"] for c in chunks if (c.augmentum or {}).get("tool_start")]
        assert starts and starts[0]["tool"] == "web_search"
        assert starts[0]["phase"] == "auto"
        completes = [c.augmentum["tool_complete"] for c in chunks if (c.augmentum or {}).get("tool_complete")]
        assert completes and completes[0]["success"] is True


class TestVCommandIntegration:
    """Verify /v command detection in passthrough context."""

    def test_v_command_detected(self):
        from augmentum.modes.v_command import extract_v_command
        request = _make_request("/v a sunset")
        has_v, instruction, cleaned = extract_v_command(request)
        assert has_v is True
        assert instruction == "a sunset"
        assert cleaned.messages[0].content == "a sunset"

    def test_v_command_not_detected(self):
        from augmentum.modes.v_command import extract_v_command
        request = _make_request("Hello there")
        has_v, instruction, cleaned = extract_v_command(request)
        assert has_v is False
        assert instruction == ""
        assert cleaned is request

    def test_v_command_empty_instruction_uses_fallback(self):
        from augmentum.modes.v_command import extract_v_command
        request = _make_request("/v")
        has_v, instruction, cleaned = extract_v_command(request)
        assert has_v is True
        assert instruction == ""
        assert cleaned.messages[0].content == "Continue the scene."
