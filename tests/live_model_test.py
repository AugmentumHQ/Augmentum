"""Live model integration test harness.

Tests the full Augmentum tool calling and chain execution pipelines against
real LLM backends (LM Studio, Ollama, cloud APIs).  Not part of the regular
pytest suite — run manually:

    .venv/Scripts/python tests/live_model_test.py [OPTIONS]

Discovery:
    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model instead of discovering all
    --ollama URL        Also test Ollama models at this URL (e.g. http://localhost:11434)

Pipeline selection (test all by default):
    --tier-only         Only test tool calling tiers (fastest)
    --chain-only        Only test chain planning + execution
    --tool-only         Only test single tool call loop
    --full              Run everything (default)

Options:
    --think             Include think=True variants (doubles test count)
    --verbose / -v      Show full model outputs
    --json              Output results as JSON
    --timeout SECS      Per-call timeout (default: 60)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows (avoids cp1252 encoding errors)
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from augmentum.models.base import (
        InternalChatRequest,
        InternalChatResponse,
        InternalStreamChunk,
        Message,
        Usage,
    )
    from augmentum.modes.analytical.tool_calling import (
        ToolCallingTier,
        build_structured_output_schema,
        coerce_tool_params,
        parse_json_tool_calls,
        parse_native_tool_calls_all,
        parse_python_style_tool_call,
        parse_react_tool_call,
        parse_structured_output,
        parse_xml_tool_call,
        tools_to_native_format,
    )
    from augmentum.tools.base import Tool, ToolCategory, ToolResult
    from augmentum.tools.chain import (
        ChainPlan,
        ChainStep,
        StepResult,
        ToolChainPlanner,
        build_synthesis_prompt,
        detect_complexity,
        execute_step,
        parse_plan_from_json,
        parse_plan_from_response,
    )
    from augmentum.tools.registry import ToolRegistry
    from augmentum.utils.logging import get_logger
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"live_model_test deps not importable in this build: {_import_exc}", allow_module_level=True)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stub tools — real tool schemas, deterministic execute
# ---------------------------------------------------------------------------


class StubWebSearch(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns search results with titles, URLs, and snippets."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        return ToolResult(
            success=True,
            output=(
                f'Search results for "{query}":\n'
                "1. Example Result - https://example.com/1 - First result snippet about the topic\n"
                "2. Wikipedia - https://en.wikipedia.org/wiki/Topic - Wikipedia article overview\n"
                "3. Research Paper - https://arxiv.org/abs/1234 - Academic perspective on the subject"
            ),
            metadata={"urls": ["https://example.com/1", "https://en.wikipedia.org/wiki/Topic"]},
        )


class StubYouTubeTranscript(Tool):
    @property
    def name(self) -> str:
        return "youtube_transcript"

    @property
    def description(self) -> str:
        return "Fetch the transcript (captions/subtitles) of a YouTube video. Provide a YouTube URL or video ID."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "YouTube URL or video ID"},
                "language": {"type": "string", "description": "Language code", "default": "en"},
            },
            "required": ["video"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        video = kwargs.get("video", "")
        return ToolResult(
            success=True,
            output=(
                f"Transcript for video {video}:\n"
                "[0:00] Welcome to this video about artificial intelligence.\n"
                "[0:15] Today we'll explore the fundamentals of machine learning.\n"
                "[0:30] Neural networks are inspired by biological brains.\n"
                "[1:00] Deep learning has revolutionized many fields.\n"
                "[1:30] Thank you for watching!"
            ),
            metadata={"video_id": "M02Q5IC-YRs", "segments": 5},
        )


class StubTextAnalysis(Tool):
    @property
    def name(self) -> str:
        return "text_analysis"

    @property
    def description(self) -> str:
        return "Analyze text content: word count, readability score, sentence count, and reading time."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        words = len(text.split())
        return ToolResult(
            success=True,
            output=f"Word count: {words}, Sentences: {max(1, words // 15)}, Reading time: {max(1, words // 200)} min, Readability: Grade 8",
            metadata={"word_count": words},
        )


class StubCalculator(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions. Returns the numeric result."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        expr = kwargs.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return ToolResult(success=True, output=str(result))
        except Exception:
            return ToolResult(success=True, output="42")


class StubWebFetch(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch and extract text content from a URL. Returns the page text."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        return ToolResult(
            success=True,
            output=f"Content from {url}:\nThis is the extracted text content from the web page. "
            "It contains information about the topic including key facts, analysis, and references.",
            metadata={"url": url, "content_length": 150},
        )


class StubWikipedia(Tool):
    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return "Look up a topic on Wikipedia. Returns article summary."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to search"},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        return ToolResult(
            success=True,
            output=f"Wikipedia: {query}\n{query} is a well-documented topic. "
            "Key aspects include historical development, current applications, and future directions.",
            metadata={"title": query, "url": f"https://en.wikipedia.org/wiki/{query}"},
        )


class StubPythonExec(Tool):
    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed environment. Returns stdout output."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")
        return ToolResult(
            success=True,
            output=f">>> Executed:\n{code[:200]}\n\nOutput: [execution result]",
        )


def _build_registry() -> ToolRegistry:
    """Build a registry with all stub tools."""
    registry = ToolRegistry()
    for tool_cls in (
        StubWebSearch, StubYouTubeTranscript, StubTextAnalysis,
        StubCalculator, StubWebFetch, StubWikipedia, StubPythonExec,
    ):
        registry.register(tool_cls())
    return registry


# ---------------------------------------------------------------------------
# Backend wrapper — directly uses httpx like OpenAIBackend
# ---------------------------------------------------------------------------


try:
    from augmentum.models.openai_compat import OpenAIBackend as _OpenAIBackend
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"augmentum.models.openai_compat not importable in this build: {_import_exc}", allow_module_level=True)


class LiveTestBackend(_OpenAIBackend):
    """Minimal backend that calls an OpenAI-compatible API directly.

    Extends OpenAIBackend so the tier selector recognizes it and applies
    the correct tool calling tier (NATIVE for local servers).
    Overrides chat/chat_stream to avoid using OpenAIBackend's internal
    payload builder (which doesn't handle all test scenarios).
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout
        self._api_key = None  # Satisfy OpenAIBackend attribute access

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Send a chat completion request."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "stream": False,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Tool calling (Tier 1)
        if request.tools:
            payload["tools"] = request.tools

        # Format constraint (Tier 2 / JSON mode)
        if request.format:
            if isinstance(request.format, dict) or request.format == "json":
                payload["response_format"] = {"type": "json_object"}

        # Think control
        if hasattr(request, "think") and request.think is False:
            # Some servers support this, others ignore it
            pass

        resp = await self._client.post("/chat/completions", json=payload)

        if resp.status_code != 200:
            # Format rejection — raise so callers can fall back
            raise RuntimeError(
                f"Backend returned {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = msg.get("tool_calls")

        return InternalChatResponse(
            message=Message(
                role=msg.get("role", "assistant"),
                content=msg.get("content") or "",
                tool_calls=tool_calls,
                thinking=msg.get("reasoning_content"),
            ),
            model=data.get("model", request.model),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )

    async def chat_stream(self, request: InternalChatRequest):
        """Streaming chat — yields InternalStreamChunks."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "stream": True,
        }
        if request.tools:
            payload["tools"] = request.tools

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                line = line[6:].strip()
                if line == "[DONE]":
                    yield InternalStreamChunk(done=True)
                    return
                chunk_data = json.loads(line)
                delta = chunk_data["choices"][0].get("delta", {})
                yield InternalStreamChunk(
                    content_delta=delta.get("content", ""),
                    thinking_delta=delta.get("reasoning_content", ""),
                    model=chunk_data.get("model", ""),
                )

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class TestResult:
    test_name: str
    model: str
    status: Status
    elapsed_ms: float = 0.0
    detail: str = ""
    tier: str = ""
    think: bool = False
    raw_output: str = ""


@dataclass
class ModelReport:
    model: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == Status.WARN)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


# --- Tier 1: Native function calling ---

SINGLE_TOOL_QUERIES = [
    {
        "name": "simple_search",
        "query": "Search the web for recent quantum computing news",
        "expected_tool": "web_search",
        "expected_param": "query",
        "description": "Explicit search request -> web_search call",
    },
    {
        "name": "youtube_video",
        "query": "Get the transcript of https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "expected_tool": "youtube_transcript",
        "expected_param": "video",
        "description": "YouTube URL → youtube_transcript call",
    },
    {
        "name": "math_expression",
        "query": "What is 15 * 23 + 7?",
        "expected_tool": "calculator",
        "expected_param": "expression",
        "description": "Math question → calculator call",
    },
    {
        "name": "analyze_text",
        "query": "Analyze this text for readability: 'The quick brown fox jumps over the lazy dog.'",
        "expected_tool": "text_analysis",
        "expected_param": "text",
        "description": "Text analysis request → text_analysis call",
    },
]

MULTI_STEP_QUERIES = [
    {
        "name": "search_then_fetch",
        "query": "Search for quantum computing and then fetch the top result for more details.",
        "expected_tools": ["web_search", "web_fetch"],
        "min_steps": 2,
        "description": "Two-step: search → fetch",
    },
    {
        "name": "video_then_analyze",
        "query": "Get the transcript of this YouTube video https://www.youtube.com/watch?v=dQw4w9WgXcQ and analyze its readability.",
        "expected_tools": ["youtube_transcript", "text_analysis"],
        "min_steps": 2,
        "description": "Two-step: transcript → text_analysis",
    },
    {
        "name": "research_pipeline",
        "query": "Search for 'machine learning applications', check Wikipedia for background, then fetch the top search result.",
        "expected_tools": ["web_search", "wikipedia", "web_fetch"],
        "min_steps": 3,
        "description": "Three-step: search + wikipedia (parallel) → fetch",
    },
]

MULTI_TOOL_QUERIES = [
    {
        "name": "search_and_calc",
        "query": "I need two things done at once: search the web for 'gold price per ounce' AND calculate 15 * 23 + 7.",
        "expected_tools": {"web_search", "calculator"},
        "min_calls": 2,
        "description": "Two independent tools in one response",
    },
    {
        "name": "search_and_wiki",
        "query": "Search the web for 'quantum computing applications' and also look up 'quantum computing' on Wikipedia. Do both now.",
        "expected_tools": {"web_search", "wikipedia"},
        "min_calls": 2,
        "description": "Two search-category tools in parallel",
    },
    {
        "name": "triple_tool",
        "query": "Do all three of these right now: search the web for 'Python 3.12', calculate 2^10 - 24, and analyze this text for readability: 'The quick brown fox jumps over the lazy dog.'",
        "expected_tools": {"web_search", "calculator", "text_analysis"},
        "min_calls": 3,
        "description": "Three independent tools in one response",
    },
]

NO_TOOL_QUERIES = [
    {
        "name": "greeting",
        "query": "Hello, how are you?",
        "description": "Greeting — model should NOT call tools",
    },
    {
        "name": "simple_fact",
        "query": "What color is the sky?",
        "description": "Simple factual — model should NOT call tools",
    },
]


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


class LiveModelTester:
    """Runs the full test suite against a live model backend."""

    def __init__(
        self,
        backend: LiveTestBackend,
        model: str,
        registry: ToolRegistry,
        *,
        verbose: bool = False,
        timeout: float = 60.0,
        test_think: bool = False,
    ) -> None:
        self.backend = backend
        self.model = model
        self.registry = registry
        self.verbose = verbose
        self.timeout = timeout
        self.test_think = test_think
        self.results: list[TestResult] = []
        self._tools = registry.list_tools()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def _add_result(self, **kwargs) -> None:
        r = TestResult(model=self.model, **kwargs)
        self.results.append(r)
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "→", "WARN": "⚠"}[r.status.value]
        detail = f" — {r.detail}" if r.detail else ""
        tier_info = f" [{r.tier}]" if r.tier else ""
        think_info = " [think]" if r.think else ""
        elapsed = f" ({r.elapsed_ms:.0f}ms)" if r.elapsed_ms else ""
        print(f"  {icon} {r.test_name}{tier_info}{think_info}{elapsed}{detail}")

    # ---------------------------------------------------------------
    # Tier tests — tool calling format compliance
    # ---------------------------------------------------------------

    async def test_tier_native(self) -> None:
        """Test Tier 1: native function calling via tools parameter."""
        for scenario in SINGLE_TOOL_QUERIES:
            await self._run_tier_test(
                scenario, ToolCallingTier.NATIVE, think=False,
            )
            if self.test_think:
                await self._run_tier_test(
                    scenario, ToolCallingTier.NATIVE, think=True,
                )

    async def test_tier_structured(self) -> None:
        """Test Tier 2: structured JSON output."""
        for scenario in SINGLE_TOOL_QUERIES:
            await self._run_tier_test(
                scenario, ToolCallingTier.STRUCTURED, think=False,
            )
            if self.test_think:
                await self._run_tier_test(
                    scenario, ToolCallingTier.STRUCTURED, think=True,
                )

    async def test_tier_text(self) -> None:
        """Test Tier 3: text-based TOOL_CALL parsing."""
        for scenario in SINGLE_TOOL_QUERIES:
            await self._run_tier_test(
                scenario, ToolCallingTier.TEXT, think=False,
            )
            if self.test_think:
                await self._run_tier_test(
                    scenario, ToolCallingTier.TEXT, think=True,
                )

    async def _run_tier_test(
        self,
        scenario: dict,
        tier: ToolCallingTier,
        *,
        think: bool = False,
    ) -> None:
        """Run a single tool calling tier test."""
        test_name = f"tier_{tier.value}_{scenario['name']}"
        start = time.monotonic()

        try:
            request = InternalChatRequest(
                model=self.model,
                messages=[Message(role="user", content=scenario["query"])],
                stream=False,
                think=think,
            )

            # Inject schemas based on tier
            if tier == ToolCallingTier.NATIVE:
                request.tools = tools_to_native_format(self._tools)
            elif tier == ToolCallingTier.STRUCTURED:
                schema = build_structured_output_schema(self._tools)
                request.format = schema  # type: ignore[assignment]
            else:
                # Text tier — inject tool description prompt
                tool_lines = []
                for t in self._tools:
                    params = t.input_schema.get("properties", {})
                    param_str = ", ".join(
                        f"{k}: {v.get('type', 'string')}" for k, v in params.items()
                    )
                    tool_lines.append(f"- {t.name}({param_str}): {t.description}")
                prompt = (
                    "\n\nAvailable tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "To call a tool, respond with:\n"
                    "TOOL_CALL: tool_name\n"
                    'TOOL_INPUT: {"param": "value"}\n\n'
                    "If no tool is needed, respond normally."
                )
                request.messages[-1] = Message(
                    role="user",
                    content=request.messages[-1].content + prompt,
                )

            try:
                response = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )
            except RuntimeError as exc:
                if "400" in str(exc):
                    elapsed = (time.monotonic() - start) * 1000
                    self._add_result(
                        test_name=test_name, status=Status.SKIP,
                        elapsed_ms=elapsed, tier=tier.value, think=think,
                        detail=f"Backend rejected format — {exc}",
                    )
                    return
                raise

            elapsed = (time.monotonic() - start) * 1000

            # Parse tool calls based on tier
            tool_name, tool_args = None, {}
            raw = response.message.content or ""

            if tier == ToolCallingTier.NATIVE:
                calls = parse_native_tool_calls_all(response)
                if calls:
                    tool_name, tool_args = calls[0]
            elif tier == ToolCallingTier.STRUCTURED:
                parsed = parse_structured_output(raw)
                if parsed:
                    tool_name, tool_args = parsed
            else:
                # Text tier — try multiple parsers
                from augmentum.modes.analytical.engine import AnalyticalEngine
                tc_name, tc_input = AnalyticalEngine._parse_tool_call(raw)
                if tc_name:
                    tool_name, tool_args = tc_name, tc_input
                else:
                    json_calls = parse_json_tool_calls(raw, {t.name for t in self._tools})
                    if json_calls:
                        tool_name, tool_args = json_calls[0]
                    else:
                        py_call = parse_python_style_tool_call(raw, {t.name for t in self._tools})
                        if py_call:
                            tool_name, tool_args = py_call
                        else:
                            xml_call = parse_xml_tool_call(raw, {t.name for t in self._tools})
                            if xml_call:
                                tool_name, tool_args = xml_call
                            else:
                                react_call = parse_react_tool_call(raw, {t.name for t in self._tools})
                                if react_call:
                                    tool_name, tool_args = react_call

            self._log(f"Raw: {raw[:300]}")
            self._log(f"Parsed: tool={tool_name}, args={tool_args}")

            # Validate
            expected_tool = scenario["expected_tool"]
            expected_param = scenario["expected_param"]

            if not tool_name:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, tier=tier.value, think=think,
                    detail=f"No tool call parsed (expected {expected_tool})",
                    raw_output=raw[:500],
                )
                return

            # Fuzzy tool name match
            resolved = self.registry.resolve(tool_name)
            actual_name = resolved.name if resolved else tool_name

            if actual_name != expected_tool:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, tier=tier.value, think=think,
                    detail=f"Wrong tool: got {actual_name}, expected {expected_tool}",
                    raw_output=raw[:500],
                )
                return

            # Check parameter name
            if expected_param not in tool_args:
                # Check if any arg has content (might be wrong name but right intent)
                if tool_args:
                    self._add_result(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed, tier=tier.value, think=think,
                        detail=f"Wrong param name: got {list(tool_args.keys())}, expected '{expected_param}'",
                        raw_output=raw[:500],
                    )
                else:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, tier=tier.value, think=think,
                        detail=f"No args (expected '{expected_param}')",
                        raw_output=raw[:500],
                    )
                return

            self._add_result(
                test_name=test_name, status=Status.PASS,
                elapsed_ms=elapsed, tier=tier.value, think=think,
                detail=f"{actual_name}({expected_param}={str(tool_args[expected_param])[:60]})",
            )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed, tier=tier.value, think=think,
                detail=f"Exception: {exc}",
                raw_output=traceback.format_exc()[:500],
            )

    # ---------------------------------------------------------------
    # No-tool tests — model should NOT call tools
    # ---------------------------------------------------------------

    async def test_no_tool_calls(self) -> None:
        """Verify the model doesn't hallucinate tool calls for simple queries."""
        for tier in (ToolCallingTier.NATIVE, ToolCallingTier.TEXT):
            for scenario in NO_TOOL_QUERIES:
                test_name = f"no_tool_{tier.value}_{scenario['name']}"
                start = time.monotonic()
                try:
                    request = InternalChatRequest(
                        model=self.model,
                        messages=[Message(role="user", content=scenario["query"])],
                        stream=False,
                    )
                    if tier == ToolCallingTier.NATIVE:
                        request.tools = tools_to_native_format(self._tools)

                    response = await asyncio.wait_for(
                        self.backend.chat(request), timeout=self.timeout,
                    )
                    elapsed = (time.monotonic() - start) * 1000

                    # Check if model called a tool
                    calls = parse_native_tool_calls_all(response)
                    raw = response.message.content or ""
                    text_call = parse_json_tool_calls(raw, {t.name for t in self._tools})

                    if calls or text_call:
                        self._add_result(
                            test_name=test_name, status=Status.WARN,
                            elapsed_ms=elapsed, tier=tier.value,
                            detail=f"Unnecessary tool call: {calls or text_call}",
                        )
                    else:
                        self._add_result(
                            test_name=test_name, status=Status.PASS,
                            elapsed_ms=elapsed, tier=tier.value,
                            detail="Correctly answered without tools",
                        )
                except Exception as exc:
                    elapsed = (time.monotonic() - start) * 1000
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, tier=tier.value,
                        detail=f"Exception: {exc}",
                    )

    # ---------------------------------------------------------------
    # Tool loop test — multi-turn tool usage
    # ---------------------------------------------------------------

    async def test_tool_loop(self) -> None:
        """Test the full tool call → execute → feed back → respond loop."""
        for tier in (ToolCallingTier.NATIVE, ToolCallingTier.TEXT):
            test_name = f"tool_loop_{tier.value}"
            start = time.monotonic()
            try:
                messages = [
                    Message(role="user", content="Search the web for 'artificial intelligence breakthroughs 2025'"),
                ]
                request = InternalChatRequest(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )

                if tier == ToolCallingTier.NATIVE:
                    request.tools = tools_to_native_format(self._tools)
                else:
                    tool_lines = []
                    for t in self._tools:
                        params = t.input_schema.get("properties", {})
                        param_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in params.items())
                        tool_lines.append(f"- {t.name}({param_str}): {t.description}")
                    prompt = (
                        "\n\nAvailable tools:\n" + "\n".join(tool_lines) + "\n\n"
                        "To call a tool, respond with:\nTOOL_CALL: tool_name\n"
                        'TOOL_INPUT: {"param": "value"}\n\n'
                        "If no tool is needed, respond normally."
                    )
                    request.messages[-1] = Message(
                        role="user",
                        content=request.messages[-1].content + prompt,
                    )

                # Iteration 1: expect tool call
                response = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )

                calls = parse_native_tool_calls_all(response)
                raw = response.message.content or ""
                tool_name = None
                tool_args = {}

                if calls:
                    tool_name, tool_args = calls[0]
                else:
                    from augmentum.modes.analytical.engine import AnalyticalEngine
                    tc_name, tc_input = AnalyticalEngine._parse_tool_call(raw)
                    if tc_name:
                        tool_name, tool_args = tc_name, tc_input

                if not tool_name:
                    elapsed = (time.monotonic() - start) * 1000
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, tier=tier.value,
                        detail="No tool call in iteration 1",
                        raw_output=raw[:500],
                    )
                    continue

                # Execute tool
                resolved = self.registry.resolve(tool_name)
                if resolved:
                    tool_args = coerce_tool_params(resolved, tool_args)
                    result = await resolved.execute(**tool_args)
                    tool_output = result.output
                else:
                    tool_output = f"Error: Unknown tool '{tool_name}'"

                # Feed result back
                request.messages.append(Message(
                    role="assistant", content=raw,
                    tool_calls=response.message.tool_calls,
                ))

                if tier == ToolCallingTier.NATIVE and response.message.tool_calls:
                    tc_id = response.message.tool_calls[0].get("id", "call_1")
                    request.messages.append(Message(
                        role="tool", content=tool_output,
                        tool_call_id=tc_id,
                    ))
                else:
                    request.messages.append(Message(
                        role="user",
                        content=f"Tool result from {tool_name}:\n{tool_output}",
                    ))

                # Clear tool schemas for final response
                request.tools = None
                request.format = None  # type: ignore[assignment]

                # Iteration 2: expect final text response
                response2 = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                final_text = response2.message.content or ""

                if len(final_text.strip()) > 10:
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed, tier=tier.value,
                        detail=f"Tool: {tool_name} → final response ({len(final_text)} chars)",
                    )
                else:
                    self._add_result(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed, tier=tier.value,
                        detail=f"Final response too short ({len(final_text)} chars)",
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, tier=tier.value,
                    detail=f"Exception: {exc}",
                    raw_output=traceback.format_exc()[:500],
                )

    # ---------------------------------------------------------------
    # Multi-tool tests — multiple tool calls in one response
    # ---------------------------------------------------------------

    async def test_multi_tool(self) -> None:
        """Test that models can produce multiple tool calls in a single response.

        Tests both native tier (tool_calls array) and text tier (multiple
        TOOL_CALL blocks or JSON arrays in content).
        """
        for scenario in MULTI_TOOL_QUERIES:
            for tier in (ToolCallingTier.NATIVE, ToolCallingTier.TEXT):
                await self._run_multi_tool_test(scenario, tier)

    async def _run_multi_tool_test(
        self, scenario: dict, tier: ToolCallingTier,
    ) -> None:
        test_name = f"multi_tool_{tier.value}_{scenario['name']}"
        start = time.monotonic()

        try:
            request = InternalChatRequest(
                model=self.model,
                messages=[Message(role="user", content=scenario["query"])],
                stream=False,
            )

            if tier == ToolCallingTier.NATIVE:
                request.tools = tools_to_native_format(self._tools)
            else:
                tool_lines = []
                for t in self._tools:
                    params = t.input_schema.get("properties", {})
                    param_str = ", ".join(
                        f"{k}: {v.get('type', 'string')}" for k, v in params.items()
                    )
                    tool_lines.append(f"- {t.name}({param_str}): {t.description}")
                prompt = (
                    "\n\nAvailable tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "You may call MULTIPLE tools in a single response.\n"
                    "For each tool call, use this format on its own line:\n"
                    "TOOL_CALL: tool_name\n"
                    'TOOL_INPUT: {"param": "value"}\n\n'
                    "Call all requested tools now."
                )
                request.messages[-1] = Message(
                    role="user",
                    content=request.messages[-1].content + prompt,
                )

            try:
                response = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )
            except RuntimeError as exc:
                if "400" in str(exc):
                    elapsed = (time.monotonic() - start) * 1000
                    self._add_result(
                        test_name=test_name, status=Status.SKIP,
                        elapsed_ms=elapsed, tier=tier.value,
                        detail=f"Backend rejected format — {exc}",
                    )
                    return
                raise

            elapsed = (time.monotonic() - start) * 1000
            raw = response.message.content or ""
            self._log(f"Raw ({len(raw)} chars): {raw[:500]}")

            # Use the universal parser — same code path as production
            from augmentum.tools.parsing import parse_tool_calls as _parse_all
            parsed = _parse_all(response, self._tools, self.backend)
            all_calls: list[tuple[str, dict]] = [(p.name, p.args) for p in parsed]

            # Deduplicate by tool name (same tool called twice = 1 unique)
            found_tools = {name for name, _ in all_calls}

            expected = scenario["expected_tools"]
            min_calls = scenario["min_calls"]

            self._log(f"Found {len(all_calls)} call(s): {[n for n, _ in all_calls]}")

            if len(all_calls) >= min_calls and found_tools >= expected:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed, tier=tier.value,
                    detail=f"{len(all_calls)} calls: {sorted(found_tools)}",
                )
            elif len(all_calls) >= 1:
                missing = expected - found_tools
                self._add_result(
                    test_name=test_name, status=Status.WARN,
                    elapsed_ms=elapsed, tier=tier.value,
                    detail=f"Only {len(all_calls)}/{min_calls} calls, got {sorted(found_tools)}, missing {sorted(missing)}",
                    raw_output=raw[:500],
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, tier=tier.value,
                    detail=f"No tool calls parsed (expected {min_calls}: {sorted(expected)})",
                    raw_output=raw[:500],
                )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed, tier=tier.value,
                detail=f"Exception: {exc}",
                raw_output=traceback.format_exc()[:500],
            )

    # ---------------------------------------------------------------
    # Chain planning tests
    # ---------------------------------------------------------------

    async def test_chain_planning(self) -> None:
        """Test chain plan generation with JSON and text fallback."""
        for scenario in MULTI_STEP_QUERIES:
            for method in ("json", "text"):
                for think in ([False, True] if self.test_think else [False]):
                    await self._run_plan_test(scenario, method, think=think)

    async def _run_plan_test(
        self, scenario: dict, method: str, *, think: bool = False,
    ) -> None:
        test_name = f"plan_{method}_{scenario['name']}"
        start = time.monotonic()

        try:
            planner = ToolChainPlanner(self.backend, self.registry)
            plan_messages = planner._build_plan_prompt(
                InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=scenario["query"])],
                    stream=False,
                ),
                self._tools,
            )

            request = InternalChatRequest(
                model=self.model,
                messages=plan_messages,
                stream=False,
                think=think,
            )

            if method == "json":
                request.format = "json"

            try:
                response = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )
            except RuntimeError as exc:
                if "400" in str(exc) and method == "json":
                    elapsed = (time.monotonic() - start) * 1000
                    self._add_result(
                        test_name=test_name, status=Status.SKIP,
                        elapsed_ms=elapsed, think=think,
                        detail="Backend rejected json format — expected for this server",
                    )
                    return
                raise

            elapsed = (time.monotonic() - start) * 1000
            raw = response.message.content or ""
            thinking = getattr(response.message, "thinking", None) or ""

            self._log(f"Raw ({len(raw)} chars): {raw[:400]}")
            if thinking:
                self._log(f"Thinking ({len(thinking)} chars): {thinking[:200]}")

            # Try JSON parse first, then text
            plan = parse_plan_from_json(raw, self.registry)
            parse_method = "json"
            if not plan and thinking:
                plan = parse_plan_from_json(thinking, self.registry)
                parse_method = "json_from_thinking"
            if not plan:
                plan = parse_plan_from_response(response, self.registry)
                parse_method = "text"
            if not plan and thinking:
                # Try text parse on thinking
                from augmentum.models.base import InternalChatResponse as ICR
                think_resp = ICR(
                    message=Message(role="assistant", content=thinking),
                    model=self.model,
                )
                plan = parse_plan_from_response(think_resp, self.registry)
                parse_method = "text_from_thinking"

            if not plan:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, think=think,
                    detail="No plan parsed (tried json+text on content+thinking)",
                    raw_output=raw[:500],
                )
                return

            # Validate plan
            plan_tools = [s.tool for s in plan.steps if s.tool]
            expected = set(scenario["expected_tools"])
            found = set(plan_tools)
            min_steps = scenario.get("min_steps", 2)

            issues = []
            if len(plan.steps) < min_steps:
                issues.append(f"too few steps ({len(plan.steps)}, need {min_steps})")
            missing = expected - found
            if missing:
                issues.append(f"missing tools: {missing}")

            if issues:
                self._add_result(
                    test_name=test_name, status=Status.WARN,
                    elapsed_ms=elapsed, think=think,
                    detail=f"Plan issues: {'; '.join(issues)} (parsed via {parse_method}, got {plan_tools})",
                    raw_output=raw[:500],
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed, think=think,
                    detail=f"Plan OK: {plan_tools} (via {parse_method})",
                )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed, think=think,
                detail=f"Exception: {exc}",
                raw_output=traceback.format_exc()[:500],
            )

    # ---------------------------------------------------------------
    # Chain execution tests — full pipeline
    # ---------------------------------------------------------------

    async def test_chain_execution(self) -> None:
        """Test full chain: plan → execute steps → synthesize."""
        for scenario in MULTI_STEP_QUERIES[:2]:  # First two for speed
            await self._run_chain_test(scenario)

    async def _run_chain_test(self, scenario: dict) -> None:
        test_name = f"chain_exec_{scenario['name']}"
        start = time.monotonic()

        try:
            planner = ToolChainPlanner(self.backend, self.registry)
            request = InternalChatRequest(
                model=self.model,
                messages=[Message(role="user", content=scenario["query"])],
                stream=False,
            )

            result = await asyncio.wait_for(
                planner.plan_and_execute(request, self._tools),
                timeout=self.timeout * 2,  # chains take longer
            )

            elapsed = (time.monotonic() - start) * 1000

            if result is None:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail="plan_and_execute returned None (planning failed)",
                )
                return

            results, plan = result
            succeeded = sum(1 for r in results.values() if r.success)
            total = len(results)

            if succeeded == total:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed,
                    detail=f"All {total} steps succeeded: {[r.tool_name for r in results.values()]}",
                )
            elif succeeded > 0:
                failed_tools = [r.tool_name for r in results.values() if not r.success]
                self._add_result(
                    test_name=test_name, status=Status.WARN,
                    elapsed_ms=elapsed,
                    detail=f"{succeeded}/{total} steps succeeded. Failed: {failed_tools}",
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"All {total} steps failed",
                )

        except TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed,
                detail=f"Chain timed out after {self.timeout * 2:.0f}s",
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed,
                detail=f"Exception: {exc}",
                raw_output=traceback.format_exc()[:500],
            )

    # ---------------------------------------------------------------
    # Arg resolution tests — LLM determines tool parameters
    # ---------------------------------------------------------------

    async def test_arg_resolution(self) -> None:
        """Test LLM's ability to determine correct tool arguments from context."""
        scenarios = [
            {
                "name": "arg_from_query",
                "step": ChainStep(
                    id=1, tool="web_search", needs=[],
                    reason="Search for information about quantum computing",
                ),
                "query": "Tell me about quantum computing applications in healthcare",
                "expected_param": "query",
                "prior_results": {},
            },
            {
                "name": "arg_from_prior_step",
                "step": ChainStep(
                    id=2, tool="text_analysis", needs=[1],
                    reason="Analyze the fetched content",
                ),
                "query": "Analyze this article",
                "expected_param": "text",
                "prior_results": {
                    1: StepResult(
                        step_id=1, tool_name="web_fetch",
                        output="This is a long article about quantum computing and its applications in various fields.",
                        metadata={}, success=True,
                    ),
                },
            },
        ]

        for sc in scenarios:
            test_name = f"arg_resolve_{sc['name']}"
            start = time.monotonic()
            try:
                ctx = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=sc["query"])],
                    stream=False,
                )
                sr = await asyncio.wait_for(
                    execute_step(
                        sc["step"], sc["prior_results"],
                        self.backend, self.registry,
                        request_context=ctx,
                    ),
                    timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                if sr.success:
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"Tool {sr.tool_name} executed successfully",
                    )
                else:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed,
                        detail=f"Step failed: {sr.output[:200]}",
                    )
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                )

    # ---------------------------------------------------------------
    # Synthesis test — final response from tool results
    # ---------------------------------------------------------------

    async def test_synthesis(self) -> None:
        """Test the synthesis prompt — LLM generates final answer from step results."""
        test_name = "synthesis"
        start = time.monotonic()
        try:
            plan = ChainPlan(steps=[
                ChainStep(id=1, tool="web_search", reason="Search for the topic"),
                ChainStep(id=2, tool="web_fetch", reason="Fetch detailed content", needs=[1]),
            ])
            results = {
                1: StepResult(
                    step_id=1, tool_name="web_search",
                    output='Search results for "AI safety":\n1. AI Safety Institute - https://example.com\n2. Wikipedia - https://en.wikipedia.org/wiki/AI_safety',
                    metadata={"urls": ["https://example.com"]}, success=True,
                ),
                2: StepResult(
                    step_id=2, tool_name="web_fetch",
                    output="The AI Safety Institute was established to research and mitigate risks from advanced AI systems. Key areas include alignment research, interpretability, and evaluation frameworks.",
                    metadata={}, success=True,
                ),
            }

            synth_prompt = build_synthesis_prompt(plan, results)
            request = InternalChatRequest(
                model=self.model,
                messages=[
                    Message(role="user", content="Tell me about AI safety organizations"),
                    Message(role="user", content=synth_prompt),
                ],
                stream=False,
            )

            response = await asyncio.wait_for(
                self.backend.chat(request), timeout=self.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            content = response.message.content or ""

            if len(content.strip()) > 50:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed,
                    detail=f"Synthesis produced {len(content)} chars",
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.WARN,
                    elapsed_ms=elapsed,
                    detail=f"Synthesis too short ({len(content)} chars)",
                    raw_output=content[:500],
                )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                elapsed_ms=elapsed,
                detail=f"Exception: {exc}",
            )

    # ---------------------------------------------------------------
    # Complexity detection test
    # ---------------------------------------------------------------

    async def test_complexity_detection(self) -> None:
        """Test that detect_complexity correctly identifies multi-step queries.

        Uses a small tool set (2 tools from 1 category) for simple queries
        to avoid false positives from having too many tool categories.
        """
        test_name = "complexity_detection"
        # Small tool set: single category (both SEARCH)
        small_tools = [t for t in self._tools if t.name in ("web_search", "wikipedia")]
        # Multi-category tool set for complex queries
        all_tools = self._tools

        cases = [
            ("Search for cats", False, "simple single-tool", small_tools),
            ("Hello!", False, "greeting", small_tools),
            ("What is 2+2?", False, "simple math", [t for t in self._tools if t.name == "calculator"]),
            ("Search for cats then fetch the article", True, "explicit then", all_tools),
            ("First search for AI, then get Wikipedia, then analyze the text", True, "three step", all_tools),
        ]

        all_pass = True
        for query, expected, desc, tools in cases:
            result = detect_complexity(query, tools)
            if result != expected:
                all_pass = False
                self._add_result(
                    test_name=f"{test_name}_{desc.replace(' ', '_')}",
                    status=Status.FAIL,
                    detail=f"'{query}' -> {result}, expected {expected}",
                )

        if all_pass:
            self._add_result(
                test_name=test_name, status=Status.PASS,
                detail=f"All {len(cases)} cases correct",
            )

    # ---------------------------------------------------------------
    # Parser affinity test — measures "last-successful-wins" hit rate
    # ---------------------------------------------------------------

    # Extra queries beyond SINGLE_TOOL_QUERIES to increase sample size
    _AFFINITY_EXTRA_QUERIES = [
        {"query": "Search the web for Python 3.12 release notes", "name": "affinity_search2"},
        {"query": "What is 2^10 - 24?", "name": "affinity_math2"},
        {"query": "Get the Wikipedia article about quantum computing", "name": "affinity_wiki"},
        {"query": "Fetch the web page at https://example.com", "name": "affinity_fetch"},
        {"query": "Search for latest AI research papers 2026", "name": "affinity_search3"},
        {"query": "Calculate the square root of 144", "name": "affinity_math3"},
        {"query": "Analyze this text for sentiment: 'I love sunny days but hate traffic.'", "name": "affinity_analyze2"},
        {"query": "Search the web for climate change solutions", "name": "affinity_search4"},
    ]

    # Named parser functions for affinity tracking
    _PARSER_KEYS = ["tool_call_text", "json_array", "python_style", "xml_function", "react"]

    def _identify_parser(self, text: str) -> str | None:
        """Run all text-tier parsers and return the key of the first match."""
        from augmentum.modes.analytical.engine import AnalyticalEngine

        known = {t.name for t in self._tools}

        # Same order as handler waterfall
        tc_name, _ = AnalyticalEngine._parse_tool_call(text)
        if tc_name and (not known or tc_name in known):
            return "tool_call_text"

        if parse_json_tool_calls(text, known):
            return "json_array"

        if parse_python_style_tool_call(text, known):
            return "python_style"

        if parse_xml_tool_call(text, known):
            return "xml_function"

        if parse_react_tool_call(text, known):
            return "react"

        return None

    async def test_parser_affinity(self) -> None:
        """Measure how consistent a model's text-tier format is.

        Sends multiple tool-requiring queries with text-tier prompts, tracks
        which parser wins each time, and simulates the affinity cache to
        measure how many calls would be cache hits vs misses.
        """
        # Build query list: SINGLE_TOOL_QUERIES + extra affinity queries
        queries = [
            {"query": s["query"], "name": s["name"]}
            for s in SINGLE_TOOL_QUERIES
        ] + self._AFFINITY_EXTRA_QUERIES

        # Build text-tier prompt suffix
        tool_lines = []
        for t in self._tools:
            params = t.input_schema.get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in params.items())
            tool_lines.append(f"- {t.name}({param_str}): {t.description}")
        prompt_suffix = (
            "\n\nAvailable tools:\n" + "\n".join(tool_lines) + "\n\n"
            "To call a tool, respond with:\nTOOL_CALL: tool_name\n"
            'TOOL_INPUT: {"param": "value"}\n\n'
            "If no tool is needed, respond normally."
        )

        # Track results
        parser_hits: list[str | None] = []  # which parser won each query
        cache_key: str | None = None  # simulated affinity cache
        cache_hits = 0
        cache_misses = 0
        total_parsed = 0

        for q in queries:
            test_name = f"affinity_{q['name']}"
            start = time.monotonic()
            try:
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(
                        role="user",
                        content=q["query"] + prompt_suffix,
                    )],
                    stream=False,
                )

                response = await asyncio.wait_for(
                    self.backend.chat(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                raw = response.message.content or ""

                winner = self._identify_parser(raw)
                parser_hits.append(winner)

                if winner:
                    total_parsed += 1
                    if cache_key and cache_key == winner:
                        cache_hits += 1
                    elif cache_key:
                        cache_misses += 1
                    # First success doesn't count as hit or miss
                    cache_key = winner

                    self._log(f"  {q['name']}: parser={winner}")
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed, tier="text",
                        detail=f"parser={winner}",
                    )
                else:
                    self._log(f"  {q['name']}: NO PARSE — {raw[:200]}")
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, tier="text",
                        detail="No parser matched",
                        raw_output=raw[:500],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                parser_hits.append(None)
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, tier="text",
                    detail=f"Exception: {exc}",
                )

        # Summary
        parser_counts: dict[str, int] = {}
        for p in parser_hits:
            if p:
                parser_counts[p] = parser_counts.get(p, 0) + 1

        total_cacheable = cache_hits + cache_misses  # excludes first call
        hit_rate = (cache_hits / total_cacheable * 100) if total_cacheable else 0

        dominant = max(parser_counts, key=parser_counts.get) if parser_counts else "none"  # type: ignore[arg-type]
        dominant_pct = (parser_counts.get(dominant, 0) / total_parsed * 100) if total_parsed else 0

        summary = (
            f"Parsed: {total_parsed}/{len(queries)} | "
            f"Dominant: {dominant} ({dominant_pct:.0f}%) | "
            f"Affinity hits: {cache_hits}/{total_cacheable} ({hit_rate:.0f}%) | "
            f"Parsers used: {dict(parser_counts)}"
        )
        print(f"\n  Parser Affinity Summary: {summary}")

        status = Status.PASS if hit_rate >= 70 else (Status.WARN if hit_rate >= 40 else Status.FAIL)
        self._add_result(
            test_name="affinity_summary", status=status,
            detail=summary,
        )


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


async def discover_models(base_url: str) -> list[str]:
    """Fetch available models from an OpenAI-compatible server."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/models")
            if resp.status_code != 200:
                print(f"  Warning: {base_url}/models returned {resp.status_code}")
                return []
            data = resp.json()
            models = [
                m["id"] for m in data.get("data", [])
                if "embed" not in m["id"].lower()
            ]
            return models
    except Exception as exc:
        print(f"  Warning: Failed to discover models at {base_url}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_report(reports: list[ModelReport]) -> None:
    """Print a summary report of all test results."""
    print("\n" + "=" * 70)
    print("LIVE MODEL TEST REPORT")
    print("=" * 70)

    for report in reports:
        total = len(report.results)
        print(f"\n{'─' * 70}")
        print(f"  Model: {report.model}")
        print(f"  Results: {report.passed}/{total} passed, "
              f"{report.failed} failed, {report.warnings} warnings")
        print(f"{'─' * 70}")

        # Group by test category
        categories: dict[str, list[TestResult]] = {}
        for r in report.results:
            cat = r.test_name.split("_")[0]
            categories.setdefault(cat, []).append(r)

        for cat, results in categories.items():
            passed = sum(1 for r in results if r.status == Status.PASS)
            failed = sum(1 for r in results if r.status == Status.FAIL)
            warns = sum(1 for r in results if r.status == Status.WARN)
            skips = sum(1 for r in results if r.status == Status.SKIP)
            print(f"\n  {cat}: {passed}✓ {failed}✗ {warns}⚠ {skips}→")

            # Show failures and warnings
            for r in results:
                if r.status in (Status.FAIL, Status.WARN):
                    icon = "✗" if r.status == Status.FAIL else "⚠"
                    print(f"    {icon} {r.test_name}: {r.detail}")

    # Overall summary
    total_pass = sum(r.passed for r in reports)
    total_fail = sum(r.failed for r in reports)
    total_warn = sum(r.warnings for r in reports)
    total = sum(len(r.results) for r in reports)
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_pass}/{total} passed, {total_fail} failed, {total_warn} warnings")
    print(f"{'=' * 70}")


def export_json(reports: list[ModelReport], path: str) -> None:
    """Export results as JSON."""
    data = []
    for report in reports:
        data.append({
            "model": report.model,
            "passed": report.passed,
            "failed": report.failed,
            "warnings": report.warnings,
            "results": [
                {
                    "test": r.test_name,
                    "status": r.status.value,
                    "elapsed_ms": r.elapsed_ms,
                    "detail": r.detail,
                    "tier": r.tier,
                    "think": r.think,
                }
                for r in report.results
            ],
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_tests(args: argparse.Namespace) -> list[ModelReport]:
    """Run the test suite against discovered models."""
    base_url = args.url.rstrip("/")
    registry = _build_registry()
    reports: list[ModelReport] = []

    # Discover models
    if args.model:
        models = [args.model]
    else:
        print(f"\nDiscovering models at {base_url}...")
        models = await discover_models(base_url)
        if not models:
            print("No models found!")
            return reports
        print(f"Found {len(models)} model(s): {', '.join(models)}")

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"  Testing: {model}")
        print(f"{'=' * 70}")

        backend = LiveTestBackend(base_url, timeout=args.timeout)
        tester = LiveModelTester(
            backend, model, registry,
            verbose=args.verbose,
            timeout=args.timeout,
            test_think=args.think,
        )

        run_all = not (args.tier_only or args.chain_only or args.tool_only or args.affinity_only or args.multi_only)

        # Complexity detection (pure logic, no LLM — always run unless affinity-only)
        if not args.affinity_only:
            await tester.test_complexity_detection()

        if run_all or args.tier_only:
            print("\n  --- Tier 1: Native Function Calling ---")
            await tester.test_tier_native()
            print("\n  --- Tier 2: Structured JSON Output ---")
            await tester.test_tier_structured()
            print("\n  --- Tier 3: Text-Based Parsing ---")
            await tester.test_tier_text()

        if run_all or args.tool_only:
            print("\n  --- No-Tool Queries ---")
            await tester.test_no_tool_calls()
            print("\n  --- Tool Loop (call → execute → respond) ---")
            await tester.test_tool_loop()

        if run_all or args.multi_only:
            print("\n  --- Multi-Tool (parallel calls in one response) ---")
            await tester.test_multi_tool()

        if run_all or args.chain_only:
            print("\n  --- Chain Planning ---")
            await tester.test_chain_planning()
            print("\n  --- Arg Resolution ---")
            await tester.test_arg_resolution()
            print("\n  --- Chain Execution ---")
            await tester.test_chain_execution()
            print("\n  --- Synthesis ---")
            await tester.test_synthesis()

        if run_all or args.affinity_only:
            print("\n  --- Parser Affinity (last-successful-wins) ---")
            await tester.test_parser_affinity()

        report = ModelReport(model=model, results=tester.results)
        reports.append(report)
        await backend.close()

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live model integration test harness for Augmentum pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--url", default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL (default: %(default)s)")
    parser.add_argument("--model", help="Test a single model")
    parser.add_argument("--ollama", help="Also test Ollama at this URL")
    parser.add_argument("--tier-only", action="store_true", help="Only test tiers")
    parser.add_argument("--chain-only", action="store_true", help="Only test chains")
    parser.add_argument("--tool-only", action="store_true", help="Only test tool loop")
    parser.add_argument("--affinity-only", action="store_true", help="Only test parser affinity")
    parser.add_argument("--multi-only", action="store_true", help="Only test multi-tool calls")
    parser.add_argument("--think", action="store_true", help="Include think=True variants")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", help="Export results as JSON to this path")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-call timeout (default: %(default)s)")

    args = parser.parse_args()

    print("Augmentum Live Model Test Harness")
    print(f"Backend: {args.url}")
    if args.think:
        print("Think mode: enabled (testing both think=True and think=False)")

    reports = asyncio.run(run_tests(args))

    if reports:
        print_report(reports)
        if args.json:
            export_json(reports, args.json)
    else:
        print("\nNo test results generated.")


if __name__ == "__main__":
    main()
