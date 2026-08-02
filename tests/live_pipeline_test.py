"""Live pipeline integration test — exercises the PRODUCTION PassthroughHandler.

Unlike live_model_test.py which builds raw requests and calls parsers directly,
this test goes through the real handler code path:

    PassthroughHandler.handle() / handle_stream()
        → _resolve_tools()
        → _inject_tool_schemas()   [from augmentum.tools.parsing]
        → backend.chat()
        → _parse_tool_calls()      [universal parser with affinity cache]
        → _execute_and_append()    [coerce_and_execute]
        → (loop until done)

Each tool has a **prompt pool** (6-10 phrasings: explicit, implicit, question,
terse, polite, etc.). Use ``--rounds N`` to test N random prompts per tool
instead of just one. This catches phrasing-dependent failures that a single
"Search the web for X" prompt would miss.

Usage:
    .venv/Scripts/python tests/live_pipeline_test.py [OPTIONS]

Discovery:
    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model instead of discovering all

Pipeline selection (test all by default):
    --tool-only         Only test individual tool triggering
    --multi-only        Only test multi-tool scenarios
    --chain-only        Only test chain planning + execution
    --stream-only       Only test streaming path
    --error-only        Only test error scenarios
    --synthesis-only    Only test synthesis quality
    --ambiguous-only    Only test ambiguous/tricky prompts
    --full              Run everything (default)

Options:
    --verbose / -v      Show full model outputs
    --json PATH         Export results as JSON
    --timeout SECS      Per-call timeout (default: 60)
    --skip-artifacts    Skip artifact tool tests (faster)
    --rounds N          Prompts per tool (default: 1). Higher = broader coverage.
    --seed N            Random seed for reproducible prompt selection.

Examples:
    # Quick smoke test — 1 prompt per tool on a specific model
    .venv/Scripts/python tests/live_pipeline_test.py --model qwen3.5-27b

    # Thorough test — 4 prompt variations per tool, reproducible
    .venv/Scripts/python tests/live_pipeline_test.py --model qwen3.5-27b --rounds 4 --seed 42

    # Full battery on all models
    .venv/Scripts/python tests/live_pipeline_test.py --rounds 3 --json results.json
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

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    Usage,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.parsing import clear_parser_affinity
from augmentum.tools.registry import ToolRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


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
# LiveTestBackend — calls real LLM via OpenAI-compatible API
# ---------------------------------------------------------------------------


class LiveTestBackend:
    """Minimal ModelBackend that calls an OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout
        # Pretend to be an OpenAI backend so select_tier returns NATIVE
        self.__class__.__qualname__ = "OpenAIBackend"

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
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
        if request.tools:
            payload["tools"] = request.tools

        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Backend {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        return InternalChatResponse(
            message=Message(
                role=msg.get("role", "assistant"),
                content=msg.get("content") or "",
                tool_calls=msg.get("tool_calls"),
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
                raw = line[6:].strip()
                if raw == "[DONE]":
                    yield InternalStreamChunk(done=True)
                    return
                chunk_data = json.loads(raw)
                delta = chunk_data["choices"][0].get("delta", {})
                yield InternalStreamChunk(
                    content_delta=delta.get("content", ""),
                    thinking_delta=delta.get("reasoning_content", ""),
                    model=chunk_data.get("model", ""),
                )

    async def list_models(self):
        resp = await self._client.get("/models")
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Call-tracking wrapper — wraps REAL tools with call counters
# ---------------------------------------------------------------------------


class _TrackedTool(Tool):
    """Base for stub-only tools that have no real implementation."""

    def __init__(self) -> None:
        self._calls: list[dict] = []

    def record_call(self, **kwargs: Any) -> None:
        self._calls.append(kwargs)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_args(self) -> dict:
        return self._calls[-1] if self._calls else {}

    def reset(self) -> None:
        self._calls.clear()


class _TrackedWrapper(_TrackedTool):
    """Wraps a REAL tool with call tracking.

    Delegates ``name``, ``description``, ``category``, ``input_schema``
    to the wrapped tool, but intercepts ``execute()`` to record calls
    before forwarding to the real implementation.
    """

    def __init__(self, real_tool: Tool) -> None:
        super().__init__()
        self._real = real_tool

    @property
    def name(self) -> str:
        return self._real.name

    @property
    def description(self) -> str:
        return self._real.description

    @property
    def category(self):
        return self._real.category

    @property
    def input_schema(self) -> dict:
        return self._real.input_schema

    @property
    def cacheable(self) -> bool:
        return getattr(self._real, "cacheable", True)

    @property
    def timeout(self) -> float:
        return getattr(self._real, "timeout", 30.0)

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        return await self._real.execute(**kwargs)


# ---------------------------------------------------------------------------
# Stub tools — only for tools that need complex infrastructure
# ---------------------------------------------------------------------------


class StubWebSearch(_TrackedTool):
    name = "web_search"
    description = "Search the web using SearXNG"
    category = ToolCategory.SEARCH
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {"type": "integer", "description": "Maximum number of results", "default": 5},
            "categories": {"type": "string", "description": "SearXNG search categories", "default": "general"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        q = kwargs.get("query", "")
        return ToolResult(
            success=True,
            output=(
                f'Search results for "{q}":\n'
                "1. Example Result — https://example.com/1 — First result about the topic\n"
                "2. Wikipedia — https://en.wikipedia.org/wiki/Topic — Overview article\n"
                "3. Research Paper — https://arxiv.org/abs/1234 — Academic perspective"
            ),
        )


class StubWeb(_TrackedTool):
    name = "web"
    description = "Look up information on the web. Accepts a URL to fetch directly, or a search query."
    category = ToolCategory.SEARCH
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A URL to fetch directly, or a search query"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        q = kwargs.get("query", "")
        return ToolResult(
            success=True,
            output=f"Web results for '{q}': Found 3 relevant pages with information about the topic.",
        )


class StubMemoryRecall(_TrackedTool):
    name = "memory_recall"
    description = "Search the user's cross-session memory for relevant context"
    category = ToolCategory.SEARCH
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query to find relevant memories"},
            "limit": {"type": "integer", "description": "Maximum memories to return", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        return ToolResult(
            success=True,
            output="- [preference] User prefers dark mode and concise responses\n"
                   "- [fact] User works on the Augmentum project",
        )


class StubPythonExec(_TrackedTool):
    name = "python_exec"
    description = "Execute Python code in a sandboxed environment"
    category = ToolCategory.EXECUTE
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30},
        },
        "required": ["code"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        code = kwargs.get("code", "")
        return ToolResult(success=True, output=f">>> {code[:100]}\n\nOutput: 45")


class StubMathVerify(_TrackedTool):
    name = "math_verify"
    description = "Verify mathematical expressions and equations"
    category = ToolCategory.VERIFY
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            "expected": {"type": "string", "description": "Expected result to compare against"},
        },
        "required": ["expression"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        expr = kwargs.get("expression", "")
        expected = kwargs.get("expected", "")
        return ToolResult(
            success=True,
            output=f"Expression: {expr}\nResult: verified" + (f" (matches {expected})" if expected else ""),
        )


class StubImageGeneration(_TrackedTool):
    name = "image_generation"
    description = "Generate an image from a text prompt"
    category = ToolCategory.IMAGE
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the image to generate"},
            "style": {"type": "string", "description": "Genre preset", "default": ""},
            "aspect": {"type": "string", "description": "Aspect ratio", "default": "square"},
        },
        "required": ["prompt"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        return ToolResult(
            success=True,
            output="Image generated successfully",
            metadata={"image_id": "img_test123", "image_url": "/api/image/img_test123"},
        )


class StubCreateDocument(_TrackedTool):
    name = "create_document"
    description = "Create a professional document (PDF or DOCX)"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title"},
            "format": {"type": "string", "enum": ["pdf", "docx"], "default": "pdf"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["heading", "body"],
                },
            },
        },
        "required": ["title", "sections"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        title = kwargs.get("title", "Untitled")
        return ToolResult(success=True, output=f"Document '{title}' created: /downloads/{title}.pdf")


class StubCreatePresentation(_TrackedTool):
    name = "create_presentation"
    description = "Create a PowerPoint presentation (.pptx)"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Presentation title"},
            "slides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string", "default": ""},
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["title", "slides"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        title = kwargs.get("title", "Untitled")
        return ToolResult(success=True, output=f"Presentation '{title}' created: /downloads/{title}.pptx")


class StubCreateSpreadsheet(_TrackedTool):
    name = "create_spreadsheet"
    description = "Create an Excel spreadsheet (.xlsx)"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Spreadsheet filename"},
            "sheets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "headers": {"type": "array", "items": {"type": "string"}},
                        "rows": {"type": "array", "items": {"type": "array"}},
                    },
                    "required": ["name", "headers", "rows"],
                },
            },
        },
        "required": ["title", "sheets"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        title = kwargs.get("title", "Untitled")
        return ToolResult(success=True, output=f"Spreadsheet '{title}' created: /downloads/{title}.xlsx")


class StubCreateChart(_TrackedTool):
    name = "create_chart"
    description = "Create a chart image (PNG) from structured data"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Chart title"},
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"], "default": "bar"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["values"],
                },
            },
        },
        "required": ["title", "labels", "datasets"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        title = kwargs.get("title", "Untitled")
        return ToolResult(
            success=True,
            output=f"Chart '{title}' created",
            metadata={"image_url": "/api/image/chart_test123"},
        )


class StubFailingTool(_TrackedTool):
    name = "failing_tool"
    description = "A tool that always fails (for testing error handling)"
    category = ToolCategory.EXECUTE
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Input query"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.record_call(**kwargs)
        return ToolResult(success=False, error="Simulated failure: service unavailable")


# ---------------------------------------------------------------------------
# Registry builder — real tools where possible, stubs for heavy infra
# ---------------------------------------------------------------------------
#
# Tier 1 (always real): calculator, datetime, unit_converter, hash,
#         json_tool, text_analysis — zero external deps
# Tier 2 (real with network): wikipedia, youtube_transcript, web_fetch
#         — need only httpx
# Tier 3 (stub): web_search (SearXNG), python_exec / math_verify (executor),
#         file_ops, document_parse — need containers or disk setup
# Tier 4 (stub): image_generation, memory_recall, artifacts — need complex
#         app state (queues, stores, etc.)
#

_STUB_ONLY = [
    StubWebSearch, StubWeb, StubMemoryRecall,
    StubPythonExec, StubMathVerify,
    StubImageGeneration,
]

_ARTIFACT_STUB_CLASSES = [
    StubCreateDocument, StubCreatePresentation,
    StubCreateSpreadsheet, StubCreateChart,
]


def _build_real_tools() -> list[Tool]:
    """Instantiate real tool implementations (Tier 1 + 2)."""
    import httpx

    tools: list[Tool] = []

    # Tier 1 — zero-dep utility tools
    from augmentum.tools.calculator import CalculatorTool
    from augmentum.tools.datetime_tool import DateTimeTool
    from augmentum.tools.hash_tool import HashTool
    from augmentum.tools.json_tool import JsonTool
    from augmentum.tools.text_analysis import TextAnalysisTool
    from augmentum.tools.unit_converter import UnitConverterTool

    tools.extend([
        CalculatorTool(),
        DateTimeTool(),
        UnitConverterTool(),
        HashTool(),
        JsonTool(),
        TextAnalysisTool(),
    ])

    # Tier 2 — network tools (real Wikipedia API, real YouTube, real web fetch)
    http_client = httpx.AsyncClient(timeout=20.0)

    from augmentum.tools.wikipedia import WikipediaTool
    tools.append(WikipediaTool(http_client))

    from augmentum.tools.youtube_transcript import YouTubeTranscriptTool
    tools.append(YouTubeTranscriptTool())

    from augmentum.tools.web_fetch import WebFetchTool
    tools.append(WebFetchTool())

    # Tier 3 — file ops with a temp dir (real disk I/O, sandboxed)
    import tempfile
    _workdir = tempfile.mkdtemp(prefix="augmentum_test_")
    # Seed a test file so file_ops read prompts work
    _test_file = Path(_workdir) / "notes.txt"
    _test_file.write_text(
        "Meeting notes from March 10:\n"
        "- Discussed Q1 results: revenue up 12%\n"
        "- New product launch scheduled for April\n"
        "- Action items: finalize budget, hire 2 engineers\n"
    )
    from augmentum.tools.file_ops import FileOpsTool
    tools.append(FileOpsTool(base_dir=_workdir))

    # Tier 3 — document parse (real parser, but files may not exist)
    from augmentum.tools.document_parse import DocumentParseTool
    tools.append(DocumentParseTool(base_dir=_workdir))

    return tools


def _build_registry(*, include_artifacts: bool = True, include_failing: bool = False) -> ToolRegistry:
    """Build tool registry: real tools (tracked) + stubs for infrastructure-heavy ones."""
    registry = ToolRegistry()

    # Register real tools wrapped in call tracking
    real_tools = _build_real_tools()
    for tool in real_tools:
        registry.register(_TrackedWrapper(tool))

    # Register stub-only tools (need SearXNG, executor, memory store, etc.)
    for cls in _STUB_ONLY:
        registry.register(cls())

    if include_artifacts:
        for cls in _ARTIFACT_STUB_CLASSES:
            registry.register(cls())

    if include_failing:
        registry.register(StubFailingTool())

    return registry


def _get_stub(registry: ToolRegistry, name: str) -> _TrackedTool:
    """Get a tracked tool from the registry for call tracking."""
    tool = registry.get(name)
    assert isinstance(tool, (_TrackedTool, _TrackedWrapper)), f"Tool '{name}' is not tracked"
    return tool


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt pools — multiple phrasings per tool for randomised selection
# ---------------------------------------------------------------------------
# Each tool gets a list of (query, description) tuples. The test runner
# picks ``rounds`` prompts per tool (default 1, use --rounds N for more).

import random as _random

TOOL_PROMPT_POOLS: dict[str, list[tuple[str, str]]] = {
    # ── Search / Fetch ────────────────────────────────────────────────
    "web_search": [
        ("Search the web for recent quantum computing breakthroughs", "explicit search request"),
        ("Look up the current price of gold online", "implicit search — 'look up'"),
        ("Find me news articles about the Mars rover", "implicit search — 'find me'"),
        ("What are the top programming languages in 2026?", "question implying search"),
        ("I need web results about climate change policies", "direct need statement"),
        ("Can you search for healthy dinner recipes?", "polite request"),
        ("Search the internet for open-source LLM benchmarks", "uses 'internet' not 'web'"),
        ("Google renewable energy statistics for 2025", "uses 'google' as verb"),
    ],
    "web_fetch": [
        ("Fetch the content from https://example.com/article", "explicit URL fetch"),
        ("Read this webpage for me: https://news.example.org/story", "implicit fetch — 'read'"),
        ("Get the text from https://docs.python.org/3/tutorial/index.html", "implicit fetch — 'get text'"),
        ("What does this page say? https://blog.example.com/post-42", "question with URL"),
        ("Scrape https://example.com/pricing for me", "uses 'scrape'"),
        ("Open https://example.com/about and show me what it says", "conversational"),
        ("Download the page at https://example.net/data.html", "uses 'download'"),
        ("Extract the main content from https://medium.com/example-article", "uses 'extract'"),
    ],
    "wikipedia": [
        ("Look up photosynthesis on Wikipedia", "explicit Wikipedia mention"),
        ("What does Wikipedia say about the Roman Empire?", "question format"),
        ("Give me the Wikipedia summary for CRISPR gene editing", "summary request"),
        ("I want to read about the Napoleonic Wars on Wikipedia", "want statement"),
        ("Check Wikipedia for information on black holes", "uses 'check'"),
        ("Pull up the Wikipedia article on the Great Barrier Reef", "uses 'pull up'"),
        ("Wikipedia entry for Alan Turing please", "terse command"),
        ("What's the Wikipedia page about dark matter?", "casual question"),
    ],
    "youtube_transcript": [
        ("Get the transcript of https://www.youtube.com/watch?v=dQw4w9WgXcQ", "explicit transcript request"),
        ("What is said in this video? https://youtube.com/watch?v=abc123", "question about video"),
        ("Grab the subtitles from https://www.youtube.com/watch?v=xyz789", "uses 'subtitles'"),
        ("Can you get captions for https://youtu.be/test456", "uses 'captions' with short URL"),
        ("I need the text from this YouTube video: https://www.youtube.com/watch?v=demo999", "need statement"),
        ("Transcribe this YouTube link: https://youtube.com/watch?v=vid001", "uses 'transcribe'"),
        ("Show me what they say in https://www.youtube.com/watch?v=talk42", "casual request"),
        ("Pull the transcript from https://www.youtube.com/watch?v=lec007", "uses 'pull'"),
    ],
    "memory_recall": [
        ("What do you remember about my preferences?", "explicit memory query"),
        ("Do you have any memories about me?", "question about self"),
        ("Recall anything you know about my projects", "uses 'recall'"),
        ("Check your memory for my name", "uses 'check your memory'"),
        ("What have I told you before?", "past conversation reference"),
        ("Search your memory for my work preferences", "explicit search memory"),
        ("Do you remember what language I prefer coding in?", "specific recall"),
        ("Any saved memories about my setup?", "terse query"),
    ],
    # ── Verify / Compute ──────────────────────────────────────────────
    "calculator": [
        ("Calculate 15 * 23 + 7", "explicit calculate"),
        ("What is 3.14 * 25 squared?", "question with math"),
        ("Solve: (100 + 250) / 7", "uses 'solve'"),
        ("How much is 18% of 450?", "percentage question"),
        ("Compute the result of 2^10 - 24", "uses 'compute'"),
        ("I need the answer to 999 * 1001", "need statement"),
        ("Do the math: 144 / 12 + 8 * 3", "uses 'do the math'"),
        ("Work out 5 factorial for me", "uses 'work out'"),
        ("What's the square root of 625?", "sqrt question"),
        ("Add up 12.5, 33.7, and 99.8", "summation in natural language"),
    ],
    "datetime": [
        ("What is the current date and time?", "explicit datetime"),
        ("What day is today?", "simple day question"),
        ("Tell me the current time", "time only"),
        ("What day of the week was January 1, 2000?", "day-of-week historical"),
        ("What's today's date?", "casual date question"),
        ("How many days until December 25?", "date diff"),
        ("What time is it in Tokyo right now?", "timezone question"),
        ("Is it a weekday or weekend today?", "day type question"),
        ("Give me the current UTC timestamp", "technical request"),
        ("When is 30 days from now?", "date addition"),
    ],
    "unit_converter": [
        ("Convert 100 kilometers to miles", "explicit conversion"),
        ("How many pounds is 50 kilograms?", "question format"),
        ("What's 72°F in Celsius?", "temperature conversion"),
        ("Convert 5 liters to gallons", "volume conversion"),
        ("How many inches are in 2.5 meters?", "length conversion"),
        ("What is 16 ounces in grams?", "weight — small units"),
        ("Turn 3.5 acres into square meters", "uses 'turn into'"),
        ("Express 120 km/h in miles per hour", "speed conversion"),
    ],
    "hash": [
        ("Compute the SHA-256 hash of 'hello world'", "explicit hash"),
        ("What's the MD5 hash of 'password123'?", "question with algorithm"),
        ("Hash the string 'test data' using SHA-512", "specific algorithm"),
        ("Generate a SHA-256 digest for 'openai'", "uses 'digest'"),
        ("I need the hash of 'my secret text'", "need statement"),
        ("Calculate the SHA-1 checksum of 'verify me'", "uses 'checksum'"),
        ("Get the hash value of 'benchmark' in SHA-256", "uses 'hash value'"),
        ("Create a hash for 'input data'", "uses 'create a hash'"),
    ],
    "json_tool": [
        ('Validate this JSON: {"name": "test", "value": 42}', "explicit validate"),
        ('Is this valid JSON? {"users": [1, 2, 3]}', "question format"),
        ('Pretty-print this JSON: {"a":1,"b":2,"c":3}', "format request"),
        ('Check if this JSON is correct: {"key": "value", "nested": {"x": true}}', "uses 'check'"),
        ("Minify this JSON: { \"hello\" : \"world\" , \"count\" : 5 }", "minify request"),
        ('What are the keys in this JSON: {"name":"Alice","age":30,"city":"NYC"}?', "keys query"),
        ('Parse and validate: [{"id":1},{"id":2}]', "array JSON"),
        ('Format this JSON data: {"status":"ok","data":{"items":[1,2,3]}}', "nested format"),
    ],
    "math_verify": [
        ("Verify that the square root of 144 equals 12", "explicit verify"),
        ("Is it true that 7! = 5040?", "question format"),
        ("Check if 2^8 equals 256", "uses 'check'"),
        ("Prove that sin(90°) = 1", "uses 'prove'"),
        ("Confirm the equation: 3x + 5 = 20 when x = 5", "algebraic check"),
        ("Validate that log base 2 of 1024 is 10", "uses 'validate'"),
        ("Is the derivative of x^3 equal to 3x^2?", "calculus verification"),
        ("Double-check: the sum from 1 to 100 should be 5050", "uses 'double-check'"),
    ],
    "text_analysis": [
        ("Analyze this text for readability: 'The quick brown fox jumps over the lazy dog.'", "explicit analysis"),
        ("How readable is this paragraph? 'Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.'", "readability question"),
        ("Count the words in: 'To be or not to be, that is the question.'", "word count"),
        ("What's the reading level of this text? 'The mitochondria is the powerhouse of the cell.'", "reading level"),
        ("Analyze: 'Four score and seven years ago our fathers brought forth on this continent a new nation.'", "terse command"),
        ("Give me readability scores for: 'Python is a high-level programming language known for its simplicity and versatility.'", "scores request"),
        ("How long would it take to read this? 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.'", "reading time"),
        ("Check the grade level of: 'Quantum entanglement is a phenomenon where two particles become interconnected.'", "grade level"),
    ],
    "python_exec": [
        ("Run this Python code: print(sum(range(10)))", "explicit run"),
        ("Execute this Python script: for i in range(5): print(i**2)", "uses 'execute'"),
        ("Run Python: import math; print(math.factorial(10))", "import + function"),
        ("Can you run this code? print([x*2 for x in range(8)])", "polite request"),
        ("Execute: print(sorted([3,1,4,1,5,9,2,6]))", "sorting task"),
        ("Write and run a Python one-liner that prints the first 10 Fibonacci numbers", "generate + run"),
        ("Run this: print(len('Hello, World!'))", "string operation"),
        ("Python: print({k: v for k, v in zip('abc', [1,2,3])})", "dict comprehension"),
    ],
    "file_ops": [
        ("Read the file at /data/notes.txt", "explicit read"),
        ("What's in the file /home/user/config.yaml?", "question format"),
        ("List all files in the /data directory", "list operation"),
        ("Show me the contents of /logs/app.log", "uses 'show me'"),
        ("Check if /data/output.csv exists", "existence check"),
        ("Open and read /documents/readme.md", "uses 'open and read'"),
        ("Write 'hello world' to /tmp/test.txt", "write operation"),
        ("Display the file /etc/hosts", "uses 'display'"),
    ],
    "document_parse": [
        ("Parse the PDF file at /data/report.pdf", "explicit parse"),
        ("Extract text from /documents/whitepaper.pdf", "uses 'extract text'"),
        ("Read this PDF: /home/user/invoice.pdf", "read PDF"),
        ("What does the document /data/thesis.docx contain?", "question about docx"),
        ("Parse /data/slides.pptx and show the content", "pptx parse"),
        ("Open and extract /reports/quarterly.pdf", "uses 'open and extract'"),
        ("Get the text from this document: /files/manual.pdf", "get text from doc"),
        ("Can you read this PDF for me? /uploads/contract.pdf", "polite request"),
    ],
    "image_generation": [
        ("Generate an image of a sunset over mountains", "explicit generate"),
        ("Create a picture of a cat wearing a top hat", "uses 'create a picture'"),
        ("Draw a futuristic cityscape at night", "uses 'draw'"),
        ("Make me an image of a cozy cabin in the snow", "uses 'make me'"),
        ("Paint a portrait of a medieval knight", "uses 'paint'"),
        ("I want an image of a dragon flying over a castle", "want statement"),
        ("Generate art: an underwater coral reef scene", "uses 'art'"),
        ("Create a photorealistic image of a golden retriever puppy", "style hint"),
    ],
    # ── Artifacts ─────────────────────────────────────────────────────
    "create_document": [
        ("Create a PDF report titled 'AI Overview' with a section about machine learning", "explicit create PDF"),
        ("Write me a document about climate change with sections on causes and effects", "uses 'write me'"),
        ("Generate a report titled 'Q1 Results' with an executive summary and financial data", "business report"),
        ("Make a PDF called 'User Guide' with an introduction and getting started section", "user guide"),
        ("Create a Word document outlining our company's remote work policy", "docx request"),
        ("Draft a document titled 'Project Proposal' with background and methodology sections", "uses 'draft'"),
    ],
    "create_presentation": [
        ("Make a PowerPoint presentation titled 'AI Intro' with 3 slides about machine learning", "explicit pptx"),
        ("Create a slide deck called 'Team Update' with slides for progress, blockers, and next steps", "business deck"),
        ("Build a presentation titled 'Solar System' with a slide for each planet", "educational"),
        ("Make a pitch deck titled 'Startup XYZ' with problem, solution, and market slides", "pitch deck"),
        ("Create presentation slides about the history of the internet", "casual request"),
        ("Put together a PowerPoint called 'Onboarding' with welcome, tools, and resources slides", "onboarding"),
    ],
    "create_spreadsheet": [
        ("Create an Excel spreadsheet titled 'Sales Data' with monthly revenue data", "explicit xlsx"),
        ("Make a spreadsheet tracking employee hours for the week", "tracking sheet"),
        ("Build an Excel file called 'Budget 2026' with income and expenses columns", "budget"),
        ("Create a spreadsheet with student names and their test scores", "grades sheet"),
        ("Generate an Excel file called 'Inventory' with product names, quantities, and prices", "inventory"),
        ("Make a spreadsheet comparing features of 5 different laptops", "comparison"),
    ],
    "create_chart": [
        ("Create a bar chart titled 'Quarterly Revenue' with Q1-Q4 data", "explicit bar chart"),
        ("Make a pie chart showing market share of top 5 browsers", "pie chart"),
        ("Plot a line chart of temperature changes over 12 months", "line chart"),
        ("Create a chart comparing sales across 4 regions", "comparison chart"),
        ("Generate a scatter plot of height vs weight data", "scatter plot"),
        ("Visualize this data as a bar chart: apples 30, bananas 45, oranges 25", "inline data"),
    ],
}

# Flattened legacy-compatible views (first prompt per tool)
INDIVIDUAL_TOOL_SCENARIOS = [
    (tool, prompts[0][0], prompts[0][1])
    for tool, prompts in TOOL_PROMPT_POOLS.items()
    if tool not in {"create_document", "create_presentation", "create_spreadsheet", "create_chart"}
]

ARTIFACT_SCENARIOS = [
    (tool, TOOL_PROMPT_POOLS[tool][0][0], TOOL_PROMPT_POOLS[tool][0][1])
    for tool in ["create_document", "create_presentation", "create_spreadsheet", "create_chart"]
]

# ---------------------------------------------------------------------------
# Multi-tool prompt pool
# ---------------------------------------------------------------------------

MULTI_TOOL_POOL: list[dict] = [
    # Search + Compute
    {
        "name": "search_and_calc",
        "query": "I need two things: search the web for 'gold price' AND calculate 15 * 23 + 7",
        "expected_tools": {"web_search", "calculator"},
    },
    {
        "name": "search_and_convert",
        "query": "Look up the distance from New York to London online, and convert 5500 km to miles",
        "expected_tools": {"web_search", "unit_converter"},
    },
    {
        "name": "wiki_and_calc",
        "query": "Look up the speed of light on Wikipedia, and calculate 299792458 * 3.5",
        "expected_tools": {"wikipedia", "calculator"},
    },
    # Search + Search
    {
        "name": "web_and_wiki",
        "query": "Search the web for 'quantum computing' and also look it up on Wikipedia. Do both now.",
        "expected_tools": {"web_search", "wikipedia"},
    },
    {
        "name": "dual_search",
        "query": "Find news about SpaceX launches and also check Wikipedia for the Falcon 9 rocket",
        "expected_tools": {"web_search", "wikipedia"},
    },
    # Verify + Verify
    {
        "name": "convert_and_hash",
        "query": "Convert 100 km to miles AND compute the SHA-256 hash of 'test'",
        "expected_tools": {"unit_converter", "hash"},
    },
    {
        "name": "calc_and_verify",
        "query": "Calculate 2^16 and also verify that 17 * 19 = 323",
        "expected_tools": {"calculator", "math_verify"},
    },
    {
        "name": "datetime_and_calc",
        "query": "What's today's date and also calculate how many seconds are in 365 days (365*24*3600)?",
        "expected_tools": {"datetime", "calculator"},
    },
    # Search + File
    {
        "name": "search_and_read",
        "query": "Search the web for 'Python best practices' and read the file at /data/notes.txt",
        "expected_tools": {"web_search", "file_ops"},
    },
    # Text tools together
    {
        "name": "hash_and_analyze",
        "query": "Hash the text 'important data' with SHA-256 and analyze 'The quick brown fox jumps over the lazy dog' for readability",
        "expected_tools": {"hash", "text_analysis"},
    },
    {
        "name": "json_and_hash",
        "query": "Validate this JSON: {\"key\": 123} and compute the SHA-256 hash of 'json_data'",
        "expected_tools": {"json_tool", "hash"},
    },
    # Cross-category combos
    {
        "name": "time_and_convert",
        "query": "What time is it right now and convert 98.6°F to Celsius",
        "expected_tools": {"datetime", "unit_converter"},
    },
    {
        "name": "wiki_and_analyze",
        "query": "Get the Wikipedia article on Shakespeare and analyze its readability",
        "expected_tools": {"wikipedia", "text_analysis"},
    },
    {
        "name": "search_and_python",
        "query": "Search the web for 'fibonacci sequence' and run Python code: print([1,1,2,3,5,8,13])",
        "expected_tools": {"web_search", "python_exec"},
    },
    {
        "name": "calc_and_convert",
        "query": "Calculate 3.14159 * 10^2 and convert the result from square centimeters to square inches",
        "expected_tools": {"calculator", "unit_converter"},
    },
]

# Legacy view — pick first N for backwards compat
MULTI_TOOL_SCENARIOS = MULTI_TOOL_POOL[:3]

# ---------------------------------------------------------------------------
# Chain prompt pool (multi-step with sequencing language)
# ---------------------------------------------------------------------------

CHAIN_POOL: list[dict] = [
    {
        "name": "search_then_fetch",
        "query": "Search for quantum computing breakthroughs and then fetch the top result for more details",
        "expected_tools": ["web_search", "web_fetch"],
        "min_steps": 2,
    },
    {
        "name": "video_then_analyze",
        "query": "Get the transcript of https://www.youtube.com/watch?v=dQw4w9WgXcQ and then analyze its readability",
        "expected_tools": ["youtube_transcript", "text_analysis"],
        "min_steps": 2,
    },
    {
        "name": "wiki_then_summarize",
        "query": "Look up machine learning on Wikipedia, then analyze the text for reading level",
        "expected_tools": ["wikipedia", "text_analysis"],
        "min_steps": 2,
    },
    {
        "name": "fetch_then_hash",
        "query": "Fetch the page at https://example.com/data and then compute a SHA-256 hash of the content",
        "expected_tools": ["web_fetch", "hash"],
        "min_steps": 2,
    },
    {
        "name": "search_then_wiki_then_analyze",
        "query": "First search the web for CRISPR, then look it up on Wikipedia, and finally analyze the Wikipedia text",
        "expected_tools": ["web_search", "wikipedia", "text_analysis"],
        "min_steps": 3,
    },
    {
        "name": "calc_then_verify",
        "query": "Calculate 17 * 23 and then verify the result equals 391",
        "expected_tools": ["calculator", "math_verify"],
        "min_steps": 2,
    },
    {
        "name": "search_and_calc_then_convert",
        "query": "Search for the Earth's circumference, then calculate 40075 divided by 24, and finally convert the result from km to miles",
        "expected_tools": ["web_search", "calculator", "unit_converter"],
        "min_steps": 3,
    },
    {
        "name": "read_then_analyze",
        "query": "Read the file at /data/notes.txt, then analyze the text for readability",
        "expected_tools": ["file_ops", "text_analysis"],
        "min_steps": 2,
    },
    {
        "name": "parse_then_analyze",
        "query": "Parse the PDF at /data/report.pdf, then analyze its readability scores",
        "expected_tools": ["document_parse", "text_analysis"],
        "min_steps": 2,
    },
    {
        "name": "video_then_hash",
        "query": "Get the YouTube transcript from https://youtube.com/watch?v=abc123, then hash the transcript text",
        "expected_tools": ["youtube_transcript", "hash"],
        "min_steps": 2,
    },
    {
        "name": "python_then_verify",
        "query": "Run this Python code: print(2**10), then verify that the result equals 1024",
        "expected_tools": ["python_exec", "math_verify"],
        "min_steps": 2,
    },
    {
        "name": "search_then_document",
        "query": "Search the web for AI trends in 2026, then create a PDF report with the findings",
        "expected_tools": ["web_search", "create_document"],
        "min_steps": 2,
    },
]

# Legacy view
CHAIN_SCENARIOS = CHAIN_POOL[:2]

# ---------------------------------------------------------------------------
# No-tool prompt pool (model should NOT call any tools)
# ---------------------------------------------------------------------------

NO_TOOL_POOL: list[tuple[str, str]] = [
    # Greetings / social
    ("greeting", "Hello, how are you?"),
    ("greeting_casual", "Hey! What's up?"),
    ("greeting_formal", "Good morning, I hope you're doing well."),
    # Simple factual knowledge (no search needed)
    ("simple_fact", "What color is the sky?"),
    ("common_knowledge", "How many legs does a spider have?"),
    ("basic_science", "What is H2O?"),
    ("geography", "What continent is Brazil in?"),
    # Opinions / subjective
    ("opinion", "Do you prefer cats or dogs?"),
    ("advice_general", "What should I name my new puppy?"),
    ("creative_writing", "Write me a haiku about autumn"),
    # Meta / about the AI
    ("self_aware", "What model are you?"),
    ("capability", "Can you help me write code?"),
    ("identity", "Who created you?"),
    # Simple tasks that don't need tools
    ("explain", "Explain what a linked list is in simple terms"),
    ("translate", "How do you say 'thank you' in Japanese?"),
    ("definition", "What does the word 'ephemeral' mean?"),
    # Trick prompts that COULD tempt tool use but shouldn't need it
    ("math_trivial", "What is 2 + 2?"),
    ("rhetorical", "Why is the sky blue?"),
    ("story_request", "Tell me a short joke"),
    ("code_explain", "What does 'def' mean in Python?"),
]

# Legacy view
NO_TOOL_SCENARIOS = NO_TOOL_POOL[:3]

# ---------------------------------------------------------------------------
# Synthesis prompt pool — verify model answers the question, not just echoes
# ---------------------------------------------------------------------------

SYNTHESIS_POOL: list[dict] = [
    {
        "name": "search_answer",
        "query": "What are the latest breakthroughs in quantum computing?",
        "expected_tool": "web_search",
        "answer_keywords": ["quantum", "computing"],
    },
    {
        "name": "video_analysis",
        "query": "Analyze this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "expected_tool": "youtube_transcript",
        "answer_keywords": ["video", "transcript"],
    },
    {
        "name": "wiki_explain",
        "query": "Using Wikipedia, explain what CRISPR is and why it matters",
        "expected_tool": "wikipedia",
        "answer_keywords": ["crispr", "gene"],
    },
    {
        "name": "fetch_summarize",
        "query": "Fetch https://example.com/article and summarize the key points",
        "expected_tool": "web_fetch",
        "answer_keywords": ["quantum", "computing"],
    },
    {
        "name": "calc_explain",
        "query": "Calculate 365 * 24 * 60 and explain what the result represents",
        "expected_tool": "calculator",
        "answer_keywords": ["minutes", "year"],
    },
    {
        "name": "text_interpret",
        "query": "Analyze this text and tell me if it's suitable for a 5th grader: 'Photosynthesis is the process by which green plants convert sunlight into chemical energy.'",
        "expected_tool": "text_analysis",
        "answer_keywords": ["readability", "grade"],
    },
    {
        "name": "hash_present",
        "query": "Compute the SHA-256 hash of 'hello world' and explain what hash functions are used for",
        "expected_tool": "hash",
        "answer_keywords": ["hash", "sha"],
    },
    {
        "name": "file_summarize",
        "query": "Read the file at /data/notes.txt and give me a summary of what's in it",
        "expected_tool": "file_ops",
        "answer_keywords": ["file", "content"],
    },
]

# Legacy view
SYNTHESIS_SCENARIOS = SYNTHESIS_POOL[:2]

# ---------------------------------------------------------------------------
# Streaming prompt pool
# ---------------------------------------------------------------------------

STREAMING_POOL: list[tuple[str, str, bool]] = [
    # (test_name, query, expect_tool_call)
    ("stream_search", "Search the web for quantum computing", True),
    ("stream_wiki", "Look up black holes on Wikipedia", True),
    ("stream_calc", "Calculate 42 * 58", True),
    ("stream_hash", "Compute the SHA-256 hash of 'streaming test'", True),
    ("stream_datetime", "What's the current date and time?", True),
    ("stream_fetch", "Fetch https://example.com/page for me", True),
    ("stream_no_tool_hello", "Hello, how are you today?", False),
    ("stream_no_tool_joke", "Tell me a funny programming joke", False),
    ("stream_no_tool_explain", "Explain recursion in one sentence", False),
    ("stream_python", "Run this Python code: print('hello from stream')", True),
]

# ---------------------------------------------------------------------------
# Error scenario pool
# ---------------------------------------------------------------------------

ERROR_SCENARIOS: list[tuple[str, str]] = [
    ("error_explicit", "Use the failing_tool to process 'test data'"),
    ("error_polite", "Please run the failing_tool on 'my input'"),
    ("error_verbose", "I need you to use the failing_tool with query 'complex data processing'"),
]

# ---------------------------------------------------------------------------
# Ambiguous / tricky prompts (expected tool might be one of several)
# ---------------------------------------------------------------------------

AMBIGUOUS_POOL: list[dict] = [
    {
        "name": "math_vs_python",
        "query": "What is the sum of all prime numbers less than 50?",
        "acceptable_tools": {"calculator", "python_exec", "math_verify"},
        "desc": "math problem — calculator or python both valid",
    },
    {
        "name": "search_vs_wiki",
        "query": "Tell me about the history of the Internet",
        "acceptable_tools": {"web_search", "wikipedia", "web"},
        "desc": "knowledge query — search or wiki both valid",
    },
    {
        "name": "verify_vs_calc",
        "query": "Is 997 a prime number?",
        "acceptable_tools": {"calculator", "math_verify", "python_exec"},
        "desc": "primality — verify, calc, or python all valid",
    },
    {
        "name": "fetch_vs_search",
        "query": "What can you find about https://example.com ?",
        "acceptable_tools": {"web_fetch", "web_search", "web"},
        "desc": "URL present — fetch or search both reasonable",
    },
    {
        "name": "unit_vs_calc",
        "query": "How many centimeters are in 6 feet 2 inches?",
        "acceptable_tools": {"unit_converter", "calculator"},
        "desc": "compound conversion — converter or calc both work",
    },
    {
        "name": "code_vs_math",
        "query": "Generate the first 20 numbers in the Fibonacci sequence",
        "acceptable_tools": {"python_exec", "calculator"},
        "desc": "sequence generation — python or repeated calc",
    },
    {
        "name": "doc_vs_file",
        "query": "Read the document at /data/report.pdf",
        "acceptable_tools": {"document_parse", "file_ops"},
        "desc": "PDF read — parser preferred but file_ops acceptable",
    },
    {
        "name": "memory_vs_nothing",
        "query": "What's my favorite programming language?",
        "acceptable_tools": {"memory_recall"},
        "desc": "personal question — memory recall or direct answer",
        "allow_no_tool": True,
    },
]

# ---------------------------------------------------------------------------
# Helpers for prompt selection
# ---------------------------------------------------------------------------


def select_prompts(
    pool: dict[str, list[tuple[str, str]]],
    rounds: int = 1,
    *,
    seed: int | None = None,
) -> list[tuple[str, str, str]]:
    """Select ``rounds`` prompts per tool from the pool.

    Returns list of (tool_name, query, description).
    """
    rng = _random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for tool, prompts in pool.items():
        chosen = rng.sample(prompts, min(rounds, len(prompts)))
        for query, desc in chosen:
            out.append((tool, query, desc))
    return out


def select_from_list(
    pool: list,
    rounds: int = 1,
    *,
    seed: int | None = None,
) -> list:
    """Select ``rounds`` items from a flat list, shuffled."""
    rng = _random.Random(seed)
    if rounds >= len(pool):
        items = list(pool)
        rng.shuffle(items)
        return items
    return rng.sample(pool, rounds)


# ---------------------------------------------------------------------------
# Synthesis quality checker
# ---------------------------------------------------------------------------


_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should can could may might must need dare ought to of in for on with "
    "at by from it its this that these those i you he she we they my your his her "
    "our their what which who whom and but or nor not so yet both either neither "
    "each every all any few more most other some such no only same than too very "
    "just about also how when where why if then else".split()
)


def check_synthesis(query: str, response_text: str, tool_outputs: dict[str, str]) -> tuple[Status, str]:
    """Check whether the synthesis actually answers the query."""
    if not response_text or len(response_text.strip()) < 30:
        return Status.FAIL, f"Response too short ({len(response_text)} chars)"

    response_lower = response_text.lower()

    # Check query keyword incorporation
    query_words = {w.lower().strip("?.,!") for w in query.split()} - _STOP_WORDS
    keyword_hits = sum(1 for w in query_words if w in response_lower and len(w) > 2)
    if keyword_hits == 0 and query_words:
        return Status.WARN, "Response doesn't reference query keywords"

    # Check that response isn't just raw tool output verbatim
    for tool_name, output in tool_outputs.items():
        if output and len(output) > 50:
            # If >80% of the response is identical to tool output, it's just echoing
            if output.strip() in response_text.strip():
                return Status.WARN, f"Response is verbatim copy of {tool_name} output"

    return Status.PASS, f"Synthesis OK ({len(response_text)} chars, {keyword_hits} keyword hits)"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class LivePipelineTester:
    """Runs the full pipeline test suite via PassthroughHandler."""

    def __init__(
        self,
        backend: LiveTestBackend,
        model: str,
        registry: ToolRegistry,
        *,
        verbose: bool = False,
        timeout: float = 60.0,
        skip_artifacts: bool = False,
        rounds: int = 1,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.registry = registry
        self.verbose = verbose
        self.timeout = timeout
        self.skip_artifacts = skip_artifacts
        self.rounds = rounds
        self.seed = seed
        self.results: list[TestResult] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def _add(self, **kwargs: Any) -> None:
        r = TestResult(model=self.model, **kwargs)
        self.results.append(r)
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "→", "WARN": "⚠"}[r.status.value]
        detail = f" — {r.detail}" if r.detail else ""
        elapsed = f" ({r.elapsed_ms:.0f}ms)" if r.elapsed_ms else ""
        print(f"  {icon} {r.test_name}{elapsed}{detail}")

    def _make_handler(self, tool_names: list[str] | None = None):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        names = tool_names or [t.name for t in self.registry.list_tools()]
        return PassthroughHandler(
            backend=self.backend,
            tool_registry=self.registry,
            enabled_tools=names,
        )

    def _reset_stubs(self) -> None:
        """Reset call counters on all stub tools."""
        for tool in self.registry.list_tools():
            if isinstance(tool, _TrackedTool):
                tool.reset()

    # ---------------------------------------------------------------
    # Individual tool tests
    # ---------------------------------------------------------------

    async def test_individual_tools(self) -> None:
        """Test each tool individually via the production handler.

        Uses the prompt pool with ``self.rounds`` prompts per tool for
        broader coverage across phrasings and edge cases.
        """
        # Build tool pool excluding artifacts if skipped
        pool = {k: v for k, v in TOOL_PROMPT_POOLS.items()}
        if self.skip_artifacts:
            for art in ("create_document", "create_presentation",
                        "create_spreadsheet", "create_chart"):
                pool.pop(art, None)

        scenarios = select_prompts(pool, rounds=self.rounds, seed=self.seed)

        # Track how many prompts per tool for unique test names
        tool_counters: dict[str, int] = {}

        for expected_tool, query, desc in scenarios:
            tool_counters[expected_tool] = tool_counters.get(expected_tool, 0) + 1
            idx = tool_counters[expected_tool]
            test_name = f"tool_{expected_tool}" if idx == 1 else f"tool_{expected_tool}_{idx}"
            self._reset_stubs()
            clear_parser_affinity()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=query)],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                # Check if the expected tool was called
                stub = _get_stub(self.registry, expected_tool)
                final_text = response.message.content if response.message else ""

                if stub.call_count > 0:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"{desc} — tool called {stub.call_count}x, response {len(final_text)} chars",
                    )
                else:
                    # Check if any other tool was called instead
                    called = [
                        t.name for t in self.registry.list_tools()
                        if isinstance(t, _TrackedTool) and t.call_count > 0
                    ]
                    if called:
                        self._add(
                            test_name=test_name, status=Status.WARN,
                            elapsed_ms=elapsed,
                            detail=f"Wrong tool called: {called} instead of {expected_tool}",
                            raw_output=final_text[:300],
                        )
                    else:
                        self._add(
                            test_name=test_name, status=Status.FAIL,
                            elapsed_ms=elapsed,
                            detail=f"No tool called. Response: {final_text[:200]}",
                            raw_output=final_text[:500],
                        )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                    raw_output=traceback.format_exc()[:500],
                )

    # ---------------------------------------------------------------
    # Multi-tool tests
    # ---------------------------------------------------------------

    async def test_multi_tool(self) -> None:
        """Test multi-tool scenarios via the handler."""
        scenarios = select_from_list(
            MULTI_TOOL_POOL,
            rounds=max(3, self.rounds * 3),
            seed=self.seed,
        )
        for scenario in scenarios:
            test_name = f"multi_{scenario['name']}"
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=scenario["query"])],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                called = {
                    t.name for t in self.registry.list_tools()
                    if isinstance(t, _TrackedTool) and t.call_count > 0
                }
                expected = scenario["expected_tools"]
                final_text = response.message.content if response.message else ""

                if expected <= called:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"All expected tools called: {called}",
                    )
                elif called & expected:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"Partial: called {called}, expected {expected}",
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed,
                        detail=f"No expected tools called. Got {called}, wanted {expected}",
                        raw_output=final_text[:500],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                )

    # ---------------------------------------------------------------
    # Chain execution tests
    # ---------------------------------------------------------------

    async def test_chain_execution(self) -> None:
        """Test chain planning + execution via the handler.

        Uses the ToolChainPlanner directly to avoid needing the
        complexity detector to trigger (which depends on query heuristics).
        """
        from augmentum.tools.chain import ToolChainPlanner

        scenarios = select_from_list(
            CHAIN_POOL,
            rounds=max(2, self.rounds * 2),
            seed=self.seed,
        )
        for scenario in scenarios:
            test_name = f"chain_{scenario['name']}"
            self._reset_stubs()
            start = time.monotonic()

            try:
                planner = ToolChainPlanner(self.backend, self.registry)
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=scenario["query"])],
                    stream=False,
                )
                tools = self.registry.list_tools()

                result = await asyncio.wait_for(
                    planner.plan_and_execute(request, tools),
                    timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                if result is None:
                    self._add(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, detail="Planning returned None",
                    )
                    continue

                results, plan = result
                step_count = len(plan.steps)
                succeeded = sum(1 for r in results.values() if r.success)
                tools_used = [r.tool_name for r in results.values()]

                if step_count >= scenario["min_steps"] and succeeded == step_count:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"{step_count} steps, all succeeded: {tools_used}",
                    )
                elif step_count >= scenario["min_steps"]:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"{step_count} steps, {succeeded} succeeded: {tools_used}",
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"Only {step_count} steps (expected {scenario['min_steps']}): {tools_used}",
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                    raw_output=traceback.format_exc()[:500],
                )

    # ---------------------------------------------------------------
    # Synthesis quality tests
    # ---------------------------------------------------------------

    async def test_synthesis(self) -> None:
        """Test that the model answers the user's question, not just echoes tool output."""
        scenarios = select_from_list(
            SYNTHESIS_POOL,
            rounds=max(2, self.rounds * 2),
            seed=self.seed,
        )
        for scenario in scenarios:
            test_name = f"synthesis_{scenario['name']}"
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=scenario["query"])],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                final_text = response.message.content if response.message else ""

                # Collect tool outputs
                tool_outputs = {}
                for tool in self.registry.list_tools():
                    if isinstance(tool, _TrackedTool) and tool.call_count > 0:
                        tool_outputs[tool.name] = tool.last_args.get("_output", "")

                # Check answer keywords
                response_lower = final_text.lower()
                keyword_hits = sum(
                    1 for kw in scenario.get("answer_keywords", [])
                    if kw.lower() in response_lower
                )
                total_keywords = len(scenario.get("answer_keywords", []))

                if keyword_hits == total_keywords and len(final_text) > 50:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"All {total_keywords} keywords found, {len(final_text)} chars",
                    )
                elif keyword_hits > 0:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"{keyword_hits}/{total_keywords} keywords, {len(final_text)} chars",
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed,
                        detail=f"No keywords found in response ({len(final_text)} chars)",
                        raw_output=final_text[:500],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                )

    # ---------------------------------------------------------------
    # No-tool tests
    # ---------------------------------------------------------------

    async def test_no_tool(self) -> None:
        """Verify the model doesn't hallucinate tool calls for simple queries."""
        scenarios = select_from_list(
            NO_TOOL_POOL,
            rounds=max(3, self.rounds * 3),
            seed=self.seed,
        )
        for name, query in scenarios:
            test_name = f"no_tool_{name}"
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=query)],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                called = [
                    t.name for t in self.registry.list_tools()
                    if isinstance(t, _TrackedTool) and t.call_count > 0
                ]
                final_text = response.message.content if response.message else ""

                if not called and len(final_text) > 5:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"No tools, response {len(final_text)} chars",
                    )
                elif called:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"Unnecessary tool call: {called}",
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"Empty response ({len(final_text)} chars)",
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                )

    # ---------------------------------------------------------------
    # Streaming tests
    # ---------------------------------------------------------------

    async def test_streaming(self) -> None:
        """Test the streaming path through the production handler."""
        scenarios = select_from_list(
            STREAMING_POOL,
            rounds=max(2, self.rounds * 2),
            seed=self.seed,
        )

        for test_name, query, expect_tool in scenarios:
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=query)],
                    stream=True,
                )
                chunks: list[InternalStreamChunk] = []
                async for chunk in handler.handle_stream(request):
                    chunks.append(chunk)

                elapsed = (time.monotonic() - start) * 1000
                content = "".join(c.content_delta or "" for c in chunks)
                has_done = any(c.done for c in chunks)

                called = [
                    t.name for t in self.registry.list_tools()
                    if isinstance(t, _TrackedTool) and t.call_count > 0
                ]

                issues = []
                if not chunks:
                    issues.append("no chunks")
                if not has_done:
                    issues.append("no done marker")
                if len(content) < 10:
                    issues.append(f"short content ({len(content)} chars)")
                if expect_tool and not called:
                    issues.append("expected tool call, none happened")

                if not issues:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"{len(chunks)} chunks, {len(content)} chars"
                              + (f", tools: {called}" if called else ""),
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.WARN if len(content) > 0 else Status.FAIL,
                        elapsed_ms=elapsed,
                        detail="; ".join(issues),
                        raw_output=content[:300],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                    raw_output=traceback.format_exc()[:500],
                )

    # ---------------------------------------------------------------
    # Error scenario tests
    # ---------------------------------------------------------------

    async def test_error_scenarios(self) -> None:
        """Test graceful handling when tools fail."""
        # Register failing tool temporarily
        failing = StubFailingTool()
        self.registry.register(failing)

        scenarios = select_from_list(
            ERROR_SCENARIOS,
            rounds=max(1, self.rounds),
            seed=self.seed,
        )

        for test_name, query in scenarios:
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler(tool_names=["failing_tool"])
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=query)],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                final_text = response.message.content if response.message else ""

                if failing.call_count > 0 and len(final_text) > 10:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"Tool failed, model recovered ({len(final_text)} chars)",
                    )
                elif failing.call_count > 0:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"Tool failed but response too short ({len(final_text)} chars)",
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail="Model didn't call the failing tool",
                        raw_output=final_text[:300],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                    raw_output=traceback.format_exc()[:500],
                )

        # Unregister failing tool
        self.registry._tools.pop("failing_tool", None)

    # ---------------------------------------------------------------
    # Ambiguous / tricky prompt tests
    # ---------------------------------------------------------------

    async def test_ambiguous(self) -> None:
        """Test prompts where multiple tools are acceptable answers."""
        scenarios = select_from_list(
            AMBIGUOUS_POOL,
            rounds=max(3, self.rounds * 3),
            seed=self.seed,
        )

        for scenario in scenarios:
            test_name = f"ambiguous_{scenario['name']}"
            self._reset_stubs()
            start = time.monotonic()

            try:
                handler = self._make_handler()
                request = InternalChatRequest(
                    model=self.model,
                    messages=[Message(role="user", content=scenario["query"])],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    handler.handle(request), timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                called = {
                    t.name for t in self.registry.list_tools()
                    if isinstance(t, _TrackedTool) and t.call_count > 0
                }
                acceptable = scenario["acceptable_tools"]
                final_text = response.message.content if response.message else ""
                allow_no_tool = scenario.get("allow_no_tool", False)

                if called & acceptable:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"{scenario['desc']} — called {called}",
                    )
                elif not called and allow_no_tool and len(final_text) > 10:
                    self._add(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed,
                        detail=f"{scenario['desc']} — answered without tools ({len(final_text)} chars)",
                    )
                elif not called:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"{scenario['desc']} — no tool called, expected one of {acceptable}",
                        raw_output=final_text[:300],
                    )
                else:
                    self._add(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed,
                        detail=f"{scenario['desc']} — called {called}, expected one of {acceptable}",
                        raw_output=final_text[:300],
                    )

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._add(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed,
                    detail=f"Exception: {exc}",
                )

    # ---------------------------------------------------------------
    # Run all
    # ---------------------------------------------------------------

    async def run_all(
        self,
        *,
        tool: bool = True,
        multi: bool = True,
        chain: bool = True,
        synthesis: bool = True,
        stream: bool = True,
        error: bool = True,
        ambiguous: bool = True,
    ) -> ModelReport:
        """Run selected test groups."""
        pool = {k: v for k, v in TOOL_PROMPT_POOLS.items()}
        if self.skip_artifacts:
            for art in ("create_document", "create_presentation",
                        "create_spreadsheet", "create_chart"):
                pool.pop(art, None)
        tool_count = sum(min(self.rounds, len(v)) for v in pool.values())
        multi_count = min(len(MULTI_TOOL_POOL), max(3, self.rounds * 3))
        chain_count = min(len(CHAIN_POOL), max(2, self.rounds * 2))
        synth_count = min(len(SYNTHESIS_POOL), max(2, self.rounds * 2))
        stream_count = min(len(STREAMING_POOL), max(2, self.rounds * 2))
        error_count = min(len(ERROR_SCENARIOS), max(1, self.rounds))
        no_tool_count = min(len(NO_TOOL_POOL), max(3, self.rounds * 3))
        ambig_count = min(len(AMBIGUOUS_POOL), max(3, self.rounds * 3))

        if tool:
            print(f"\n  ── Individual Tools ({tool_count} tests, {self.rounds} round{'s' if self.rounds > 1 else ''}) ──")
            await self.test_individual_tools()

        if multi:
            print(f"\n  ── Multi-Tool ({multi_count} tests) ──")
            await self.test_multi_tool()

        if chain:
            print(f"\n  ── Chain Execution ({chain_count} tests) ──")
            await self.test_chain_execution()

        if synthesis:
            print(f"\n  ── Synthesis Quality ({synth_count} tests) ──")
            await self.test_synthesis()

        if stream:
            print(f"\n  ── Streaming ({stream_count} tests) ──")
            await self.test_streaming()

        if error:
            print(f"\n  ── Error Handling ({error_count} tests) ──")
            await self.test_error_scenarios()

        if ambiguous:
            print(f"\n  ── Ambiguous Prompts ({ambig_count} tests) ──")
            await self.test_ambiguous()

        # Always test no-tool
        print(f"\n  ── No-Tool ({no_tool_count} tests) ──")
        await self.test_no_tool()

        return ModelReport(model=self.model, results=self.results)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


async def discover_models(backend: LiveTestBackend) -> list[str]:
    """Discover available models from the backend."""
    try:
        models = await backend.list_models()
        return [m["id"] for m in models if isinstance(m, dict) and "id" in m]
    except Exception as exc:
        print(f"  Model discovery failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(reports: list[ModelReport]) -> None:
    """Print final summary report."""
    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY")
    print("=" * 70)

    total_pass = total_fail = total_warn = 0

    for report in reports:
        p, f, w = report.passed, report.failed, report.warnings
        total_pass += p
        total_fail += f
        total_warn += w
        total = len(report.results)
        pct = (p / total * 100) if total else 0

        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\n  {report.model}")
        print(f"    {bar} {pct:.0f}%")
        print(f"    ✓ {p} passed  ✗ {f} failed  ⚠ {w} warnings  ({total} total)")

        # Show failures
        failures = [r for r in report.results if r.status == Status.FAIL]
        if failures:
            print("    Failures:")
            for r in failures:
                print(f"      ✗ {r.test_name}: {r.detail}")

    print(f"\n  TOTAL: ✓ {total_pass}  ✗ {total_fail}  ⚠ {total_warn}")
    print("=" * 70)


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
                }
                for r in report.results
            ],
        })
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"\n  Results exported to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Live pipeline integration test")
    parser.add_argument("--url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", help="Test a single model")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-call timeout")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="Export results as JSON")
    parser.add_argument("--skip-artifacts", action="store_true", help="Skip artifact tool tests")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Prompts per tool (1=one prompt each, 3=three random variations, etc.)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible prompt selection")

    # Pipeline selection
    parser.add_argument("--tool-only", action="store_true")
    parser.add_argument("--multi-only", action="store_true")
    parser.add_argument("--chain-only", action="store_true")
    parser.add_argument("--stream-only", action="store_true")
    parser.add_argument("--error-only", action="store_true")
    parser.add_argument("--synthesis-only", action="store_true")
    parser.add_argument("--ambiguous-only", action="store_true")
    parser.add_argument("--full", action="store_true")

    args = parser.parse_args()

    # Determine which tests to run
    any_filter = any([
        args.tool_only, args.multi_only, args.chain_only,
        args.stream_only, args.error_only, args.synthesis_only,
        args.ambiguous_only,
    ])
    run_tool = args.tool_only or args.full or not any_filter
    run_multi = args.multi_only or args.full or not any_filter
    run_chain = args.chain_only or args.full or not any_filter
    run_stream = args.stream_only or args.full or not any_filter
    run_error = args.error_only or args.full or not any_filter
    run_synthesis = args.synthesis_only or args.full or not any_filter
    run_ambiguous = args.ambiguous_only or args.full or not any_filter

    # Configure settings for testing
    settings.passthrough_tool_max_iterations = 5
    settings.tool_result_max_chars = 8000

    print("=" * 70)
    print("AUGMENTUM LIVE PIPELINE TEST")
    print(f"  Backend: {args.url}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Rounds: {args.rounds}" + (f"  Seed: {args.seed}" if args.seed else ""))
    print("=" * 70)

    backend = LiveTestBackend(args.url, timeout=args.timeout)
    registry = _build_registry(
        include_artifacts=not args.skip_artifacts,
        include_failing=run_error,
    )

    tool_count = len(registry.list_tools())
    print(f"  Registered {tool_count} stub tools")

    # Discover models
    if args.model:
        models = [args.model]
    else:
        models = await discover_models(backend)
        if not models:
            print("  No models found. Use --model NAME to specify.")
            return

    print(f"  Models: {', '.join(models)}")

    reports: list[ModelReport] = []

    for model_name in models:
        print(f"\n{'─' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'─' * 70}")

        tester = LivePipelineTester(
            backend=backend,
            model=model_name,
            registry=registry,
            verbose=args.verbose,
            timeout=args.timeout,
            skip_artifacts=args.skip_artifacts,
            rounds=args.rounds,
            seed=args.seed,
        )

        report = await tester.run_all(
            tool=run_tool,
            multi=run_multi,
            chain=run_chain,
            synthesis=run_synthesis,
            stream=run_stream,
            error=run_error,
            ambiguous=run_ambiguous,
        )
        reports.append(report)

    # Final report
    print_report(reports)

    if args.json:
        export_json(reports, args.json)

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
