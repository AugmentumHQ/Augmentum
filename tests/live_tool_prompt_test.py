"""Live tool prompt effectiveness benchmark.

Measures how well different models interpret tool instructions and extract
parameters across a variety of prompting styles.  Tests three dimensions:

1. **Tool selection** — does the model pick the right tool for implicit vs
   explicit queries, ambiguous requests, and negative cases?
2. **Parameter quality** — does the model generate concise, effective params
   (e.g. short Wikipedia queries, focused search terms)?
3. **Instruction following** — how sensitive is the model to tool description
   wording, schema format, and prompt framing?

Usage:
    .venv/Scripts/python tests/live_tool_prompt_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model instead of discovering all
    --verbose / -v      Show full model outputs
    --json PATH         Export results as JSON
    --timeout SECS      Per-call timeout (default: 60)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
    Usage,
)
from augmentum.modes.analytical.tool_calling import tools_to_native_format  # noqa: F401
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.parsing import parse_tool_calls
from augmentum.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Stub tools — match production schemas
# ---------------------------------------------------------------------------


class StubWebSearch(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using SearXNG"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        }

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(success=True, output="stub")


class StubWikipedia(Tool):
    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Search Wikipedia and retrieve article summaries. "
            "Use for factual lookups, definitions, historical events, "
            "biographies, and general knowledge. "
            "IMPORTANT: Use short topic names as queries (e.g. 'Iran–United States relations'), "
            "NOT long sentences or full questions."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise topic name to look up (e.g. 'Photosynthesis', "
                        "'World War II', 'Python programming language'). "
                        "Use short noun phrases, not full sentences."
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(success=True, output="stub")


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

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(success=True, output="42")


class StubImageGen(Tool):
    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return (
            "Generate an image from a text prompt. Supports style presets "
            "(fantasy_rpg, anime, scifi, horror, realism) and aspect ratios "
            "(portrait, landscape, square)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.IMAGE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Detailed text description of the image"},
                "negative_prompt": {"type": "string", "description": "Things to exclude", "default": ""},
                "style": {"type": "string", "description": "Genre preset", "default": ""},
                "aspect": {"type": "string", "description": "Aspect ratio: portrait, landscape, square", "default": "square"},
            },
            "required": ["prompt"],
        }

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(success=True, output="stub")


class StubWebFetch(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch and extract text content from a URL. This tool retrieves "
            "the page on your behalf — use it whenever the user shares a link."
        )

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

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(success=True, output="stub")


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in (StubWebSearch, StubWikipedia, StubCalculator, StubImageGen, StubWebFetch):
        reg.register(cls())
    return reg


# ---------------------------------------------------------------------------
# Backend (reuses LiveTestBackend pattern)
# ---------------------------------------------------------------------------

from augmentum.models.openai_compat import OpenAIBackend


class _Backend(OpenAIBackend):
    """Minimal OpenAI-compat backend for prompt testing."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout
        self._api_key = None

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.format:
            payload["response_format"] = (
                {"type": "json_object"} if isinstance(request.format, (dict, str)) else None
            )

        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Backend returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        return InternalChatResponse(
            message=Message(
                role=msg.get("role", "assistant"),
                content=msg.get("content") or "",
                tool_calls=msg.get("tool_calls"),
            ),
            model=data.get("model", request.model),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single prompt effectiveness test case."""
    name: str
    category: str           # selection, parameter, negative, ambiguous
    query: str
    expected_tool: str | None   # None = should NOT call any tool
    param_key: str = ""         # Which param to check
    param_validator: str = ""   # Regex or "short" / "has_url" / "math_expr"
    description: str = ""


# --- Tool Selection: Implicit vs Explicit ---
SELECTION_SCENARIOS = [
    # Explicit — model is told which tool to use
    Scenario("explicit_search", "selection",
             "Use web_search to find the latest news about SpaceX launches.",
             "web_search", "query", "",
             "Explicit tool name in query"),
    Scenario("explicit_wiki", "selection",
             "Look up 'Quantum entanglement' on Wikipedia.",
             "wikipedia", "query", "short",
             "Explicit Wikipedia mention"),
    Scenario("explicit_calc", "selection",
             "Use the calculator to compute 2^16 - 1.",
             "calculator", "expression", "math_expr",
             "Explicit calculator request"),
    Scenario("explicit_image", "selection",
             "Generate an image of a sunset over mountains.",
             "image_generation", "prompt", "",
             "Explicit image generation"),

    # Implicit — model must infer the right tool from context
    Scenario("implicit_search", "selection",
             "What are the latest developments in fusion energy?",
             "web_search", "query", "",
             "Current events → web_search (not wikipedia)"),
    Scenario("implicit_wiki", "selection",
             "Who was Alan Turing?",
             "wikipedia", "query", "short",
             "Historical person → wikipedia"),
    Scenario("implicit_calc", "selection",
             "If I have 3 items at $12.50 each plus 8% tax, what's the total?",
             "calculator", "expression", "math_expr",
             "Math word problem → calculator"),
    Scenario("implicit_image", "selection",
             "I'd love to see a cyberpunk city at night with neon lights.",
             "image_generation", "prompt", "",
             "Visual description → image_generation"),
    Scenario("implicit_fetch", "selection",
             "Can you read this article? https://example.com/article",
             "web_fetch", "url", "has_url",
             "URL in message → web_fetch"),
]

# --- Parameter Quality ---
PARAMETER_SCENARIOS = [
    Scenario("wiki_concise_topic", "parameter",
             "Tell me about the history of the Roman Empire",
             "wikipedia", "query", "short",
             "Should extract concise topic, not full sentence"),
    Scenario("wiki_person", "parameter",
             "I'd like to know about Marie Curie's contributions to science",
             "wikipedia", "query", r"(?i)marie\s+curie",
             "Should extract person name"),
    Scenario("wiki_not_sentence", "parameter",
             "Can you look up information about photosynthesis and how it works in plants?",
             "wikipedia", "query", "short",
             "Should be a noun phrase, not a question"),
    Scenario("search_focused", "parameter",
             "Find recent articles about the effects of AI on employment in 2026",
             "web_search", "query", "",
             "Search query should be focused"),
    Scenario("search_not_verbose", "parameter",
             "I was wondering if you could help me find some information about how electric vehicles compare to gasoline cars in terms of total cost of ownership over five years",
             "web_search", "query", "short",
             "Should condense verbose request into concise search query"),
    Scenario("calc_extract_math", "parameter",
             "What's the square root of 144 multiplied by 3?",
             "calculator", "expression", "math_expr",
             "Should extract a valid math expression"),
    Scenario("image_descriptive", "parameter",
             "Make me a picture of a cat wearing a top hat sitting in a library",
             "image_generation", "prompt", r"cat.*(?:hat|library)|(?:hat|library).*cat",
             "Should preserve key visual details in prompt"),
]

# --- Negative Cases (should NOT call tools) ---
NEGATIVE_SCENARIOS = [
    Scenario("greeting", "negative",
             "Hello! How are you today?",
             None, description="Simple greeting — no tool needed"),
    Scenario("opinion", "negative",
             "What do you think about pineapple on pizza?",
             None, description="Opinion question — no tool needed"),
    Scenario("simple_fact", "negative",
             "What color is the sky?",
             None, description="Common knowledge — no tool needed"),
    Scenario("followup", "negative",
             "Thanks, that's really helpful!",
             None, description="Acknowledgement — no tool needed"),
    Scenario("creative_writing", "negative",
             "Write me a short poem about autumn.",
             None, description="Creative task — no tool needed"),
]

# --- Ambiguous Cases (multiple tools could work) ---
AMBIGUOUS_SCENARIOS = [
    Scenario("search_or_wiki", "ambiguous",
             "Tell me about quantum computing.",
             None, "query", "",  # Either web_search or wikipedia is acceptable
             "Could be web_search or wikipedia — either is fine"),
    Scenario("calc_or_none", "ambiguous",
             "How much is a dozen eggs times 3?",
             None, "expression", "",  # calculator or inline answer
             "Simple enough to answer directly, but calculator is also valid"),
]

ALL_SCENARIOS = SELECTION_SCENARIOS + PARAMETER_SCENARIOS + NEGATIVE_SCENARIOS + AMBIGUOUS_SCENARIOS


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    scenario: str
    category: str
    status: str       # PASS, FAIL, WARN
    tool_called: str   # Actual tool called (or "none")
    expected_tool: str
    param_value: str
    param_ok: bool
    elapsed_ms: float
    detail: str
    raw_output: str = ""


@dataclass
class ModelReport:
    model: str
    results: list[TestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_param(value: str, validator: str) -> tuple[bool, str]:
    """Check if a parameter value meets the quality bar.

    Returns (ok, reason).
    """
    if not validator:
        return (True, "any value accepted") if value else (False, "empty value")

    if validator == "short":
        if not value:
            return False, "empty"
        if len(value) > 80:
            return False, f"too long ({len(value)} chars) — should be concise"
        # Penalize full sentences (starts with question word, ends with ?)
        if re.match(r"^(?:what|how|can|could|tell|find|search|i was)\b", value, re.I):
            return False, f"looks like a sentence, not a topic: '{value[:60]}'"
        return True, f"concise ({len(value)} chars)"

    if validator == "has_url":
        if re.search(r"https?://", value):
            return True, "contains URL"
        return False, f"no URL found in '{value[:60]}'"

    if validator == "math_expr":
        if not value:
            return False, "empty expression"
        # Should contain digits and operators
        if re.search(r"\d", value) and re.search(r"[+\-*/^%().]", value):
            return True, f"valid expression: {value[:40]}"
        # Might be a single number (e.g. sqrt function)
        if re.search(r"\d", value):
            return True, f"numeric: {value[:40]}"
        return False, f"doesn't look like math: '{value[:40]}'"

    # Treat as regex
    if re.search(validator, value, re.IGNORECASE):
        return True, f"matches /{validator}/"
    return False, f"'{value[:60]}' doesn't match /{validator}/"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class PromptTester:
    def __init__(
        self, backend: _Backend, model: str, registry: ToolRegistry,
        *, verbose: bool = False, timeout: float = 60.0,
    ) -> None:
        self.backend = backend
        self.model = model
        self.registry = registry
        self.tools = registry.list_tools()
        self.verbose = verbose
        self.timeout = timeout
        self.results: list[TestResult] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    async def run_all(self) -> None:
        for scenario in ALL_SCENARIOS:
            await self._run_scenario(scenario)

    async def _run_scenario(self, s: Scenario) -> None:
        start = time.monotonic()

        try:
            request = InternalChatRequest(
                model=self.model,
                messages=[Message(role="user", content=s.query)],
                stream=False,
            )
            # Use the full production injection path (schemas + restraint prompt)
            from augmentum.tools.parsing import inject_tool_schemas
            inject_tool_schemas(request, self.tools, self.backend)

            response = await asyncio.wait_for(
                self.backend.chat(request), timeout=self.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            # Parse with the full production waterfall
            parsed = parse_tool_calls(response, self.tools, self.backend)
            raw = response.message.content or ""

            tool_called = parsed[0].name if parsed else "none"
            tool_args = parsed[0].args if parsed else {}

            self._log(f"Query: {s.query[:70]}")
            self._log(f"Called: {tool_called}, Args: {tool_args}")
            if self.verbose and raw:
                self._log(f"Raw: {raw[:200]}")

            # --- Evaluate ---
            if s.category == "negative":
                # Should NOT call any tool
                if parsed:
                    status, detail = "FAIL", f"called {tool_called} (expected none)"
                else:
                    status, detail = "PASS", "correctly avoided tool use"
                self.results.append(TestResult(
                    scenario=s.name, category=s.category, status=status,
                    tool_called=tool_called, expected_tool="none",
                    param_value="", param_ok=True, elapsed_ms=elapsed,
                    detail=detail, raw_output=raw[:300],
                ))

            elif s.category == "ambiguous":
                # Either calling a reasonable tool or not calling is fine
                acceptable = {"web_search", "wikipedia", "calculator", "none"}
                if tool_called in acceptable:
                    status = "PASS"
                    detail = f"chose {tool_called} (acceptable)"
                else:
                    status = "WARN"
                    detail = f"unusual choice: {tool_called}"
                self.results.append(TestResult(
                    scenario=s.name, category=s.category, status=status,
                    tool_called=tool_called, expected_tool=s.expected_tool or "any",
                    param_value=str(tool_args.get(s.param_key, ""))[:100],
                    param_ok=True, elapsed_ms=elapsed, detail=detail,
                    raw_output=raw[:300],
                ))

            else:
                # selection or parameter — must call the right tool
                if not parsed:
                    self.results.append(TestResult(
                        scenario=s.name, category=s.category, status="FAIL",
                        tool_called="none", expected_tool=s.expected_tool or "none",
                        param_value="", param_ok=False, elapsed_ms=elapsed,
                        detail=f"no tool called (expected {s.expected_tool})",
                        raw_output=raw[:300],
                    ))
                elif tool_called != s.expected_tool:
                    self.results.append(TestResult(
                        scenario=s.name, category=s.category, status="FAIL",
                        tool_called=tool_called, expected_tool=s.expected_tool or "",
                        param_value="", param_ok=False, elapsed_ms=elapsed,
                        detail=f"wrong tool: {tool_called} (expected {s.expected_tool})",
                        raw_output=raw[:300],
                    ))
                else:
                    # Right tool — check parameter quality
                    param_val = str(tool_args.get(s.param_key, "")) if s.param_key else ""
                    param_ok, param_reason = _validate_param(param_val, s.param_validator)

                    if s.category == "parameter" and not param_ok:
                        status = "FAIL"
                        detail = f"param quality: {param_reason}"
                    elif not param_ok:
                        status = "WARN"
                        detail = f"right tool but param issue: {param_reason}"
                    else:
                        status = "PASS"
                        detail = f"{tool_called}({s.param_key}={param_val[:50]}) — {param_reason}"

                    self.results.append(TestResult(
                        scenario=s.name, category=s.category, status=status,
                        tool_called=tool_called, expected_tool=s.expected_tool or "",
                        param_value=param_val[:100], param_ok=param_ok,
                        elapsed_ms=elapsed, detail=detail, raw_output=raw[:300],
                    ))

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self.results.append(TestResult(
                scenario=s.name, category=s.category, status="FAIL",
                tool_called="error", expected_tool=s.expected_tool or "",
                param_value="", param_ok=False, elapsed_ms=elapsed,
                detail=f"exception: {exc}",
            ))

        # Print inline
        r = self.results[-1]
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[r.status]
        print(f"  {icon} [{r.category:10s}] {r.scenario:30s} ({r.elapsed_ms:.0f}ms) {r.detail}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(reports: list[ModelReport]) -> None:
    print("\n" + "=" * 80)
    print("TOOL PROMPT EFFECTIVENESS REPORT")
    print("=" * 80)

    for report in reports:
        total = len(report.results)
        passed = sum(1 for r in report.results if r.status == "PASS")
        failed = sum(1 for r in report.results if r.status == "FAIL")
        warned = sum(1 for r in report.results if r.status == "WARN")

        print(f"\n{'─' * 80}")
        print(f"  Model: {report.model}")
        print(f"  Overall: {passed}/{total} passed ({passed/total*100:.0f}%), "
              f"{failed} failed, {warned} warnings")
        print(f"{'─' * 80}")

        # Per-category breakdown
        categories = {}
        for r in report.results:
            categories.setdefault(r.category, []).append(r)

        for cat, results in sorted(categories.items()):
            cat_pass = sum(1 for r in results if r.status == "PASS")
            cat_fail = sum(1 for r in results if r.status == "FAIL")
            cat_warn = sum(1 for r in results if r.status == "WARN")
            pct = cat_pass / len(results) * 100 if results else 0
            print(f"\n  {cat:12s}: {cat_pass}/{len(results)} ({pct:.0f}%) "
                  f"{'✓' * cat_pass}{'✗' * cat_fail}{'⚠' * cat_warn}")

            for r in results:
                if r.status != "PASS":
                    icon = "✗" if r.status == "FAIL" else "⚠"
                    print(f"    {icon} {r.scenario}: {r.detail}")

        # Parameter quality stats (for scenarios that tested params)
        param_tests = [r for r in report.results if r.category == "parameter"]
        if param_tests:
            good_params = sum(1 for r in param_tests if r.param_ok)
            print(f"\n  Parameter quality: {good_params}/{len(param_tests)} "
                  f"({good_params/len(param_tests)*100:.0f}%) produced good params")

    # Cross-model summary
    if len(reports) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'=' * 80}")
        print(f"  {'Model':40s} {'Pass':>5s} {'Fail':>5s} {'Warn':>5s} {'Score':>6s}")
        for r in sorted(reports, key=lambda r: sum(1 for t in r.results if t.status == "PASS"), reverse=True):
            passed = sum(1 for t in r.results if t.status == "PASS")
            failed = sum(1 for t in r.results if t.status == "FAIL")
            warned = sum(1 for t in r.results if t.status == "WARN")
            pct = passed / len(r.results) * 100 if r.results else 0
            print(f"  {r.model:40s} {passed:5d} {failed:5d} {warned:5d} {pct:5.0f}%")


def export_json(reports: list[ModelReport], path: str) -> None:
    data = []
    for report in reports:
        data.append({
            "model": report.model,
            "results": [
                {
                    "scenario": r.scenario,
                    "category": r.category,
                    "status": r.status,
                    "tool_called": r.tool_called,
                    "expected_tool": r.expected_tool,
                    "param_value": r.param_value,
                    "param_ok": r.param_ok,
                    "elapsed_ms": r.elapsed_ms,
                    "detail": r.detail,
                }
                for r in report.results
            ],
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to {path}")


# ---------------------------------------------------------------------------
# Model discovery (shared with live_model_test)
# ---------------------------------------------------------------------------


async def discover_models(base_url: str) -> list[str]:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/models")
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [
                m["id"] for m in data.get("data", [])
                if "embed" not in m["id"].lower()
            ]
    except Exception as exc:
        print(f"  Warning: Failed to discover models: {exc}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_tests(args: argparse.Namespace) -> list[ModelReport]:
    base_url = args.url.rstrip("/")
    registry = _build_registry()
    reports: list[ModelReport] = []

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
        print(f"\n{'=' * 80}")
        print(f"  Model: {model}")
        print(f"  Scenarios: {len(ALL_SCENARIOS)} "
              f"({len(SELECTION_SCENARIOS)} selection, "
              f"{len(PARAMETER_SCENARIOS)} parameter, "
              f"{len(NEGATIVE_SCENARIOS)} negative, "
              f"{len(AMBIGUOUS_SCENARIOS)} ambiguous)")
        print(f"{'=' * 80}\n")

        backend = _Backend(base_url, timeout=args.timeout)
        tester = PromptTester(
            backend, model, registry,
            verbose=args.verbose, timeout=args.timeout,
        )

        await tester.run_all()
        reports.append(ModelReport(model=model, results=tester.results))
        await backend.close()

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool prompt effectiveness benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--model", help="Test a single model")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", help="Export results as JSON")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    print("Augmentum Tool Prompt Effectiveness Benchmark")
    print(f"Backend: {args.url}")

    reports = asyncio.run(run_tests(args))
    if reports:
        print_report(reports)
        if args.json:
            export_json(reports, args.json)


if __name__ == "__main__":
    main()
