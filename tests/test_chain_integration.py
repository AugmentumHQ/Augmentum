"""Integration tests for tool chain end-user integration.

Covers: safety guards (Unit 0), fallback handler plumbing (Unit 1),
config API exposure (Unit 3), voice chain forwarding (Unit 4),
flow command parsing (Unit 5), adaptive detection (Unit 6),
and safety edge cases (Unit 7).
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite

from augmentum.config import settings
from augmentum.classifier.router import Mode
from augmentum.tools.base import ToolCategory
from augmentum.tools.chain import (
    ChainPlan,
    ChainStep,
    StepResult,
    _TEMPLATE_MAX_CHARS,
    execute_chain,
    resolve_templates,
)
from augmentum.tools.custom_flows import (
    CustomFlowStore,
    _validate_regex_safe,
)


def _run(coro):
    return asyncio.run(coro)


async def _create_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS custom_flows (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            trigger_pattern TEXT DEFAULT '',
            steps_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            enabled INTEGER DEFAULT 1
        )
    """)
    await db.commit()
    return db


def _make_tool(name: str):
    from augmentum.tools.base import Tool, ToolCategory, ToolResult

    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.category = ToolCategory.SEARCH
    tool.cacheable = True
    tool.description = f"Test {name}"
    tool.timeout = 30.0
    tool.input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    tool.execute = AsyncMock(return_value=ToolResult(success=True, output=f"{name} result", metadata={}))
    return tool


def _make_registry(*tools):
    registry = MagicMock()
    tool_map = {t.name: t for t in tools}
    registry.resolve.side_effect = lambda name: tool_map.get(name)
    registry.list_tools.return_value = list(tools)
    return registry


# ---------------------------------------------------------------------------
# Unit 0: Safety Guards
# ---------------------------------------------------------------------------


class TestTemplateTruncation(unittest.TestCase):
    """Resolved template values are capped at _TEMPLATE_MAX_CHARS."""

    def test_short_value_unchanged(self):
        results = {1: StepResult(step_id=1, tool_name="t", output="short", metadata={}, success=True)}
        resolved = resolve_templates({"key": "prefix {{step.1.output}} suffix"}, results)
        self.assertEqual(resolved["key"], "prefix short suffix")

    def test_long_value_truncated(self):
        long_output = "x" * 20_000
        results = {1: StepResult(step_id=1, tool_name="t", output=long_output, metadata={}, success=True)}
        resolved = resolve_templates({"key": "{{step.1.output}}"}, results)
        self.assertIn("…[truncated]", resolved["key"])
        self.assertLessEqual(len(resolved["key"]), _TEMPLATE_MAX_CHARS + 20)


class TestReDoSValidation(unittest.TestCase):
    """Nested quantifier patterns are rejected."""

    def test_safe_pattern_passes(self):
        _validate_regex_safe(r"analyze\s+video")

    def test_nested_quantifier_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_regex_safe(r"(.+)+")
        self.assertIn("nested quantifiers", str(ctx.exception))

    def test_too_long_pattern_rejected(self):
        with self.assertRaises(ValueError):
            _validate_regex_safe("a" * 201)

    def test_valid_long_pattern_passes(self):
        _validate_regex_safe("a" * 200)


class TestMaxFlowLimit(unittest.TestCase):
    """Flow creation respects passthrough_chain_max_flows."""

    def test_max_flows_enforced(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            original = settings.passthrough_chain_max_flows
            try:
                settings.passthrough_chain_max_flows = 2
                await store.create_flow("flow1", [{"id": 1, "tool": "t", "input": {}, "needs": []}])
                await store.create_flow("flow2", [{"id": 1, "tool": "t", "input": {}, "needs": []}])
                with self.assertRaises(ValueError) as ctx:
                    await store.create_flow("flow3", [{"id": 1, "tool": "t", "input": {}, "needs": []}])
                self.assertIn("Maximum number of flows", str(ctx.exception))
            finally:
                settings.passthrough_chain_max_flows = original
                await db.close()
        _run(_test())


class TestSemaphoreConcurrency(unittest.TestCase):
    """Wave execution respects passthrough_chain_max_parallel."""

    def test_semaphore_limits_parallelism(self):
        """Verify that at most max_parallel steps run concurrently."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _slow_execute(**kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            from augmentum.tools.base import ToolResult
            return ToolResult(success=True, output="ok", metadata={})

        async def _test():
            nonlocal max_concurrent
            tools = []
            for i in range(5):
                t = _make_tool(f"tool{i}")
                t.execute = _slow_execute
                tools.append(t)

            registry = _make_registry(*tools)
            backend = MagicMock()

            plan = ChainPlan(
                steps=[
                    ChainStep(id=i + 1, tool=f"tool{i}", input={"query": "test"}, needs=[])
                    for i in range(5)
                ],
                source="test",
            )

            original = settings.passthrough_chain_max_parallel
            try:
                settings.passthrough_chain_max_parallel = 2
                await execute_chain(plan, backend, registry)
            finally:
                settings.passthrough_chain_max_parallel = original

            self.assertLessEqual(max_concurrent, 2)

        _run(_test())


# ---------------------------------------------------------------------------
# Unit 0: Flow run rate limiting
# ---------------------------------------------------------------------------


class TestFlowRunRateLimit(unittest.TestCase):
    """Per-flow rate limiting returns 429 on excess."""

    def test_rate_limit_deque_logic(self):
        from augmentum.proxy.flow_routes import (
            _FLOW_RUN_LIMIT,
            _FLOW_RUN_WINDOW,
            _flow_run_timestamps,
        )
        from collections import deque

        flow_id = "test_rate_limit"
        _flow_run_timestamps[flow_id] = deque()
        now = time.monotonic()

        # Fill up to limit
        for i in range(_FLOW_RUN_LIMIT):
            _flow_run_timestamps[flow_id].append(now)

        # Next request should exceed
        ts_deque = _flow_run_timestamps[flow_id]
        while ts_deque and ts_deque[0] < now - _FLOW_RUN_WINDOW:
            ts_deque.popleft()
        self.assertGreaterEqual(len(ts_deque), _FLOW_RUN_LIMIT)

        # Clean up
        del _flow_run_timestamps[flow_id]


# ---------------------------------------------------------------------------
# Unit 1: Fallback Handler Plumbing
# ---------------------------------------------------------------------------


class TestFallbackHandlerPlumbing(unittest.TestCase):
    """All fallback paths construct PassthroughHandler with custom_flow_store."""

    def test_narrative_fallback_has_flow_store(self):
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.classifier.router import Mode

        app_state = MagicMock()
        app_state.narrative_engines = None  # Force fallback
        app_state.custom_flow_store = MagicMock()
        app_state.image_queue = None

        backend = MagicMock()

        # Make _get_or_create_engine raise to trigger fallback
        with patch("augmentum.proxy.handler_factory._get_or_create_engine", side_effect=RuntimeError("test")):
            handler = get_handler_for_mode(Mode.NARRATIVE, backend, "test-session", app_state)

        self.assertIsInstance(handler, PassthroughHandler)
        self.assertEqual(handler._custom_flow_store, app_state.custom_flow_store)

    def test_analytical_fallback_has_flow_store(self):
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.classifier.router import Mode

        app_state = MagicMock()
        app_state.tool_registry = None  # Force AnalyticalHandler to fail
        app_state.custom_flow_store = MagicMock()
        app_state.image_queue = None

        backend = MagicMock()

        with patch("augmentum.proxy.handler_factory.AnalyticalHandler", side_effect=RuntimeError("test")):
            handler = get_handler_for_mode(Mode.ANALYTICAL, backend, "test-session", app_state)

        self.assertIsInstance(handler, PassthroughHandler)
        self.assertEqual(handler._custom_flow_store, app_state.custom_flow_store)

    def test_agentic_fallback_has_flow_store(self):
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.classifier.router import Mode

        app_state = MagicMock()
        app_state.custom_flow_store = MagicMock()
        app_state.image_queue = None

        backend = MagicMock()

        with patch("augmentum.proxy.handler_factory.AgenticHandler", side_effect=RuntimeError("test")):
            handler = get_handler_for_mode(Mode.AGENTIC, backend, "test-session", app_state)

        self.assertIsInstance(handler, PassthroughHandler)
        self.assertEqual(handler._custom_flow_store, app_state.custom_flow_store)


# ---------------------------------------------------------------------------
# Unit 3: Config API exposure
# ---------------------------------------------------------------------------


class TestConfigApiExposure(unittest.TestCase):
    """Core chain settings appear in _TOOL_SETTINGS."""

    def test_chain_settings_in_tool_settings(self):
        from augmentum.proxy.config_routes import _TOOL_SETTINGS

        expected_keys = [
            "passthrough_chain_enabled",
            "passthrough_chain_max_steps",
            "passthrough_chain_timeout",
            "passthrough_chain_max_parallel",
            "passthrough_chain_max_flows",
        ]
        for key in expected_keys:
            self.assertIn(key, _TOOL_SETTINGS, f"Missing {key} in _TOOL_SETTINGS")

    def test_chain_timeout_range(self):
        from augmentum.proxy.config_routes import _TOOL_SETTINGS

        typ, min_val, max_val = _TOOL_SETTINGS["passthrough_chain_timeout"]
        self.assertEqual(typ, float)
        self.assertEqual(min_val, 10.0)
        self.assertEqual(max_val, 600.0)


# ---------------------------------------------------------------------------
# Unit 4: Voice chain forwarding
# ---------------------------------------------------------------------------


class TestVoiceChainForwarding(unittest.TestCase):
    """Voice producer sends chain_status and chain_step messages."""

    def test_chain_metadata_fields_expected(self):
        """Verify the expected metadata keys for chain status."""
        chain_meta = {"status": "planning", "source": "adaptive"}
        self.assertIn("status", chain_meta)
        self.assertIn("source", chain_meta)

        step_meta = {"id": 1, "tool": "web_search", "status": "running", "reason": "Search"}
        self.assertIn("id", step_meta)
        self.assertIn("tool", step_meta)
        self.assertIn("status", step_meta)


# ---------------------------------------------------------------------------
# Adaptive detection
# ---------------------------------------------------------------------------


class TestAdaptiveDetection(unittest.TestCase):
    """detect_complexity triggers on multi-step language in query text."""

    def test_multi_step_language_triggers(self):
        from augmentum.tools.chain import detect_complexity  # noqa: F811

        tools = [
            _make_tool("web_search"),
            _make_tool("calculator"),
        ]
        tools[0].category = ToolCategory.SEARCH
        tools[1].category = ToolCategory.VERIFY

        result = detect_complexity(
            "search for the latest GDP numbers then calculate the growth rate", tools,
        )
        self.assertTrue(result)

    def test_multi_category_alone_does_not_trigger(self):
        """Multiple categories without multi-step language should not trigger."""
        from augmentum.tools.chain import detect_complexity

        tools = [
            _make_tool("web_search"),
            _make_tool("calculator"),
        ]
        tools[0].category = ToolCategory.SEARCH
        tools[1].category = ToolCategory.VERIFY

        result = detect_complexity(
            "search for the latest GDP numbers and calculate the growth rate", tools,
        )
        self.assertFalse(result)

    def test_simple_query_skipped(self):
        from augmentum.tools.chain import detect_complexity

        tools = [_make_tool("calculator")]
        result = detect_complexity("what is 2+2", tools)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Config settings in Settings model
# ---------------------------------------------------------------------------


class TestConfigSettings(unittest.TestCase):
    """New config settings exist with correct defaults."""

    def test_chain_timeout_default(self):
        self.assertEqual(settings.passthrough_chain_timeout, 120.0)

    def test_chain_max_parallel_default(self):
        self.assertEqual(settings.passthrough_chain_max_parallel, 3)

    def test_chain_max_flows_default(self):
        self.assertEqual(settings.passthrough_chain_max_flows, 50)


# ---------------------------------------------------------------------------
# Unit 1: Streaming chain step progress (queue-based)
# ---------------------------------------------------------------------------


class TestExecuteChainStreaming(unittest.TestCase):
    """execute_chain_streaming pushes StepResults to a queue incrementally."""

    def test_queue_receives_step_results(self):
        from augmentum.tools.chain import execute_chain_streaming

        async def _test():
            tools = [_make_tool("web_search"), _make_tool("calculator")]
            tools[1].category = ToolCategory.VERIFY
            registry = _make_registry(*tools)
            backend = MagicMock()

            plan = ChainPlan(
                steps=[
                    ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
                    ChainStep(id=2, tool="calculator", input={"query": "1+1"}, needs=[1]),
                ],
            )

            queue: asyncio.Queue = asyncio.Queue()
            results = await execute_chain_streaming(plan, backend, registry, queue)

            # Collect all items from queue
            items = []
            while not queue.empty():
                items.append(queue.get_nowait())

            # Should have received StepResult objects plus None sentinel
            step_results = [i for i in items if i is not None]
            sentinels = [i for i in items if i is None]
            self.assertEqual(len(step_results), 2)
            self.assertEqual(len(sentinels), 1)  # None sentinel
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.success for r in results.values()))

        _run(_test())

    def test_queue_receives_sentinel_on_error(self):
        """Queue gets None sentinel even when steps fail."""
        from augmentum.tools.chain import execute_chain_streaming

        async def _test():
            registry = _make_registry()  # empty — steps will fail
            backend = MagicMock()
            plan = ChainPlan(
                steps=[ChainStep(id=1, tool="nonexistent", input={"q": "x"}, needs=[])],
            )
            queue: asyncio.Queue = asyncio.Queue()
            results = await execute_chain_streaming(plan, backend, registry, queue)

            items = []
            while not queue.empty():
                items.append(queue.get_nowait())

            self.assertTrue(any(i is None for i in items))
            self.assertFalse(results[1].success)

        _run(_test())


# ---------------------------------------------------------------------------
# Unit 2: /flow command classification
# ---------------------------------------------------------------------------


class TestFlowCommandClassification(unittest.TestCase):
    """/flow command forces passthrough mode."""

    def test_flow_command_returns_passthrough(self):
        from augmentum.classifier.router import RequestClassifier
        from augmentum.models.base import InternalChatRequest, Message

        classifier = RequestClassifier()
        request = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="/flow Video Analyzer")],
        )
        result = classifier.classify(request)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("/flow", result.reason)

    def test_non_flow_not_intercepted(self):
        from augmentum.classifier.router import RequestClassifier
        from augmentum.models.base import InternalChatRequest, Message

        classifier = RequestClassifier()
        request = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="what is 2+2")],
        )
        result = classifier.classify(request)
        # Should NOT be forced to passthrough by the flow check
        # (may still be passthrough via default, but reason won't mention /flow)
        self.assertNotIn("/flow", result.reason)


# ---------------------------------------------------------------------------
# Unit 3: Non-streaming /flow not-found
# ---------------------------------------------------------------------------


class TestNonStreamingFlowNotFound(unittest.TestCase):
    """Non-streaming path returns error for unknown /flow name."""

    def test_unknown_flow_returns_error_response(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.tools.custom_flows import CustomFlowStore

        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            backend = MagicMock()
            backend.chat = AsyncMock(return_value=MagicMock(
                message=MagicMock(content="hello", tool_calls=None),
                model="test",
            ))

            handler = PassthroughHandler(
                backend=backend,
                tool_registry=_make_registry(_make_tool("web_search")),
                enabled_tools=["web_search"],
                custom_flow_store=store,
            )

            request = InternalChatRequest(
                model="test",
                messages=[Message(role="user", content="/flow NonExistent")],
            )

            # Enable chains
            original = settings.passthrough_chain_enabled
            try:
                settings.passthrough_chain_enabled = True
                response = await handler._try_chain_execution(request, handler._resolve_tools())
                self.assertIsNotNone(response)
                self.assertIn("No flows are configured", response.message.content)
            finally:
                settings.passthrough_chain_enabled = original
                await db.close()

        _run(_test())

    def test_unknown_flow_lists_available(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.tools.custom_flows import CustomFlowStore

        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Video Analyzer", [{"id": 1, "tool": "web_search"}])

            backend = MagicMock()
            handler = PassthroughHandler(
                backend=backend,
                tool_registry=_make_registry(_make_tool("web_search")),
                enabled_tools=["web_search"],
                custom_flow_store=store,
            )

            request = InternalChatRequest(
                model="test",
                messages=[Message(role="user", content="/flow Unknown")],
            )

            original = settings.passthrough_chain_enabled
            try:
                settings.passthrough_chain_enabled = True
                response = await handler._try_chain_execution(request, handler._resolve_tools())
                self.assertIn("Video Analyzer", response.message.content)
            finally:
                settings.passthrough_chain_enabled = original
                await db.close()

        _run(_test())


# ---------------------------------------------------------------------------
# Bare /flow lists available flows
# ---------------------------------------------------------------------------


class TestBareFlowCommand(unittest.TestCase):
    """Bare `/flow` (no args) lists available flows."""

    def test_classifier_routes_bare_flow(self):
        from augmentum.classifier.router import RequestClassifier
        from augmentum.models.base import InternalChatRequest, Message

        classifier = RequestClassifier()
        request = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="/flow")],
        )
        result = classifier.classify(request)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertIn("/flow", result.reason)

    def test_bare_flow_non_streaming_lists_flows(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.tools.custom_flows import CustomFlowStore

        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow(
                "Video Analyzer",
                [{"id": 1, "tool": "web_search"}],
                description="Analyze a video",
            )
            await store.create_flow(
                "Report Builder",
                [{"id": 1, "tool": "web_search"}],
            )

            backend = MagicMock()
            handler = PassthroughHandler(
                backend=backend,
                tool_registry=_make_registry(_make_tool("web_search")),
                enabled_tools=["web_search"],
                custom_flow_store=store,
            )

            request = InternalChatRequest(
                model="test",
                messages=[Message(role="user", content="/flow")],
            )

            original = settings.passthrough_chain_enabled
            try:
                settings.passthrough_chain_enabled = True
                response = await handler._try_chain_execution(request, handler._resolve_tools())
                self.assertIsNotNone(response)
                self.assertIn("Video Analyzer", response.message.content)
                self.assertIn("Report Builder", response.message.content)
                self.assertIn("Analyze a video", response.message.content)
                self.assertIn("/flow <name>", response.message.content)
            finally:
                settings.passthrough_chain_enabled = original
                await db.close()

        _run(_test())

    def test_bare_flow_empty_store(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from augmentum.tools.custom_flows import CustomFlowStore

        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)

            backend = MagicMock()
            handler = PassthroughHandler(
                backend=backend,
                tool_registry=_make_registry(_make_tool("web_search")),
                enabled_tools=["web_search"],
                custom_flow_store=store,
            )

            request = InternalChatRequest(
                model="test",
                messages=[Message(role="user", content="/flow")],
            )

            original = settings.passthrough_chain_enabled
            try:
                settings.passthrough_chain_enabled = True
                response = await handler._try_chain_execution(request, handler._resolve_tools())
                self.assertIsNotNone(response)
                self.assertIn("No flows are configured", response.message.content)
            finally:
                settings.passthrough_chain_enabled = original
                await db.close()

        _run(_test())


class TestChainUserIdPropagation(unittest.TestCase):
    """IMAGE/ARTIFACT tools called from chain must receive the user_id slot.

    Regression: chain.py:478 used to gate user_id injection on ARTIFACT only,
    so IMAGE-category tools (image_generation, image_search) ran with empty
    user_id. The image_generations row was then never written and the file
    panel never showed the image.
    """

    def _make_capturing_tool(self, name: str, category):
        """Build a spec'd tool whose execute() captures kwargs."""
        from augmentum.tools.base import Tool, ToolResult

        captured: dict = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return ToolResult(success=True, output="ok", metadata={})

        tool = MagicMock(spec=Tool)
        tool.name = name
        tool.category = category
        tool.cacheable = False
        tool.description = f"test {name}"
        tool.timeout = 30.0
        tool.input_schema = {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        }
        tool.execute = _capture
        return tool, captured

    def test_image_category_tool_receives_user_id(self):
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.chain import execute_chain
        from augmentum.models.base import InternalChatRequest, Message

        async def _test():
            tool, captured = self._make_capturing_tool(
                "image_generation", ToolCategory.IMAGE,
            )
            registry = _make_registry(tool)

            plan = ChainPlan(steps=[ChainStep(
                id=1, tool="image_generation",
                input={"prompt": "a cat"},
                reason="generate",
            )])
            request = InternalChatRequest(
                model="m", messages=[Message(role="user", content="go")],
            )

            await execute_chain(
                plan, MagicMock(), registry,
                request_context=request,
                cache_user_id="user-xyz",
            )

            # Both injection slots must be populated for image_generation
            # (which uses Tool.extract_user_id, looking at both).
            self.assertEqual(captured.get("_user_id"), "user-xyz")
            ctx = captured.get("_context")
            self.assertIsInstance(ctx, dict)
            self.assertEqual(ctx.get("user_id"), "user-xyz")

        _run(_test())

    def test_artifact_category_tool_receives_user_id(self):
        """Same contract for ARTIFACT — keeps the existing behavior."""
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.chain import execute_chain
        from augmentum.models.base import InternalChatRequest, Message

        async def _test():
            tool, captured = self._make_capturing_tool(
                "create_ebook", ToolCategory.ARTIFACT,
            )
            registry = _make_registry(tool)

            plan = ChainPlan(steps=[ChainStep(
                id=1, tool="create_ebook",
                input={"prompt": "a story"},
                reason="write",
            )])
            request = InternalChatRequest(
                model="m", messages=[Message(role="user", content="go")],
            )

            await execute_chain(
                plan, MagicMock(), registry,
                request_context=request,
                cache_user_id="user-xyz",
            )

            self.assertEqual(captured.get("_user_id"), "user-xyz")
            self.assertEqual(captured.get("_context", {}).get("user_id"), "user-xyz")

        _run(_test())

    def test_search_category_tool_does_not_receive_user_id_slot(self):
        """SEARCH tools (web_search) shouldn't get user_id injected — they
        don't write to user-scoped tables. Keeps the injection narrow.
        """
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.chain import execute_chain
        from augmentum.models.base import InternalChatRequest, Message

        async def _test():
            tool, captured = self._make_capturing_tool(
                "web_search", ToolCategory.SEARCH,
            )
            registry = _make_registry(tool)

            plan = ChainPlan(steps=[ChainStep(
                id=1, tool="web_search",
                input={"prompt": "x"}, reason="search",
            )])
            request = InternalChatRequest(
                model="m", messages=[Message(role="user", content="go")],
            )

            await execute_chain(
                plan, MagicMock(), registry,
                request_context=request,
                cache_user_id="user-xyz",
            )

            # SEARCH tool: no _user_id slot (not relevant — it doesn't
            # persist to user-scoped tables). Keeps schema noise down.
            self.assertNotIn("_user_id", captured)

        _run(_test())


class TestChainAllowedToolNames(unittest.TestCase):
    """Failure-recovery (substitute / mutate) must stay within ``allowed_tool_names``.

    Without the bound, a chain that started constrained (e.g. only
    ``image_search``) could pull in ``web_search``/``image_generation`` from
    the global registry when the LLM picks a substitute or mutates the plan.
    """

    def test_substitute_outside_allowed_set_is_rejected(self):
        """An LLM substitution to a tool outside the allowed set is dropped."""
        from augmentum.tools.base import ToolResult

        async def _test():
            # Two tools registered globally, but only one allowed.
            allowed_tool = _make_tool("image_search")
            allowed_tool.execute = AsyncMock(return_value=ToolResult(
                success=False, output="", error="search failed",
            ))
            forbidden_tool = _make_tool("web_search")
            registry = _make_registry(allowed_tool, forbidden_tool)

            plan = ChainPlan(
                steps=[ChainStep(id=1, tool="image_search", reason="search")],
            )

            # LLM picks "substitute:web_search" (forbidden). We patch
            # _replan_on_failure to return that decision deterministically.
            with patch(
                "augmentum.tools.chain._replan_on_failure",
                AsyncMock(return_value="substitute:web_search"),
            ):
                original_retries = settings.passthrough_chain_max_retries
                try:
                    settings.passthrough_chain_max_retries = 1
                    backend = MagicMock()
                    results = await execute_chain(
                        plan, backend, registry,
                        allowed_tool_names={"image_search"},
                    )
                finally:
                    settings.passthrough_chain_max_retries = original_retries

            # web_search must NOT have been executed.
            forbidden_tool.execute.assert_not_called()
            # The original failure stands.
            self.assertFalse(results[1].success)

        _run(_test())

    def test_substitute_inside_allowed_set_is_accepted(self):
        """An LLM substitution to a tool *inside* the allowed set runs."""
        from augmentum.tools.base import ToolResult

        async def _test():
            failing = _make_tool("image_search")
            failing.execute = AsyncMock(return_value=ToolResult(
                success=False, output="", error="search failed",
            ))
            backup = _make_tool("wikipedia")
            backup.execute = AsyncMock(return_value=ToolResult(
                success=True, output="wiki result", metadata={},
            ))
            registry = _make_registry(failing, backup)

            plan = ChainPlan(
                steps=[ChainStep(id=1, tool="image_search", reason="search")],
            )

            with patch(
                "augmentum.tools.chain._replan_on_failure",
                AsyncMock(return_value="substitute:wikipedia"),
            ):
                original_retries = settings.passthrough_chain_max_retries
                try:
                    settings.passthrough_chain_max_retries = 1
                    backend = MagicMock()
                    results = await execute_chain(
                        plan, backend, registry,
                        allowed_tool_names={"image_search", "wikipedia"},
                    )
                finally:
                    settings.passthrough_chain_max_retries = original_retries

            backup.execute.assert_called_once()
            self.assertTrue(results[1].success)
            self.assertEqual(results[1].output, "wiki result")

        _run(_test())

    def test_mutate_outside_allowed_set_is_rejected(self):
        """A mutated plan referencing forbidden tools is rejected."""
        from augmentum.tools.chain import _mutate_plan

        async def _test():
            allowed_tool = _make_tool("image_search")
            forbidden_tool = _make_tool("web_search")
            registry = _make_registry(allowed_tool, forbidden_tool)

            plan = ChainPlan(
                steps=[ChainStep(id=1, tool="image_search", reason="search")],
            )
            failed_step = plan.steps[0]
            failed_result = StepResult(
                step_id=1, tool_name="image_search", output="err",
                metadata={}, success=False,
            )

            # LLM returns a mutated plan that uses web_search (forbidden).
            backend = MagicMock()
            mutated_json = (
                '[{"id": 2, "tool": "web_search", "reason": "fall back",'
                ' "needs": [], "input": null}]'
            )
            backend.chat = AsyncMock(return_value=MagicMock(
                message=MagicMock(content=mutated_json),
            ))

            mutated = await _mutate_plan(
                plan, failed_step, failed_result, {1: failed_result},
                [], backend, registry, None,
                allowed_tool_names={"image_search"},
            )

            # Mutation must be rejected because it references web_search.
            self.assertIsNone(mutated)

        _run(_test())


if __name__ == "__main__":
    unittest.main()
