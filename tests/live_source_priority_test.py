"""Live test for web_search source_type parameter.

Tests whether models correctly pick fresh/reference/data source types when
calling web_search, and verifies the scoring logic ranks URLs appropriately.

Usage:
    .venv/Scripts/python tests/live_source_priority_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model (repeatable for multiple)
    --verbose / -v      Show full model outputs
    --timeout SECS      Per-call timeout (default: 60)
    --scoring-only      Only run scoring logic verification (no LLM)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
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
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.parsing import inject_tool_schemas, parse_tool_calls
from augmentum.tools.registry import ToolRegistry
from augmentum.tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class _Backend(OpenAIBackend):
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
# Stub web_search tool (real schema, no execution)
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
        import httpx
        real_tool = WebSearchTool(http_client=httpx.AsyncClient(), base_url="http://stub")
        return real_tool.input_schema

    @property
    def timeout(self) -> float:
        return 30.0

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="stub")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    query: str
    expected: str  # "fresh", "reference", or "data"
    acceptable: tuple[str, ...] = ()


SCENARIOS = [
    # Fresh
    Scenario("breaking_news", "What's happening in the news today?", "fresh"),
    Scenario("weather", "What's the weather forecast for this week?", "fresh", ("data",)),
    Scenario("local_news", "Show me recent local news near me", "fresh"),
    Scenario("stock_price", "What is Tesla's stock price right now?", "fresh", ("data",)),
    Scenario("sports_scores", "What were the NFL scores this weekend?", "fresh"),
    # Reference
    Scenario("biography", "Who was Napoleon Bonaparte?", "reference"),
    Scenario("definition", "What is quantum entanglement?", "reference"),
    Scenario("history", "When did World War 2 end and what were the key events?", "reference"),
    Scenario("how_it_works", "How does photosynthesis work?", "reference"),
    # Data
    Scenario("population_stats", "What is the population of each US state?", "data", ("reference",)),
    Scenario("cpu_specs", "Compare the specs of AMD Ryzen 9 7950X vs Intel i9-13900K", "data"),
    Scenario("benchmark_scores", "What are the latest GPU benchmark scores for RTX 4090?", "data", ("fresh",)),
]


# ---------------------------------------------------------------------------
# Scoring verification (no LLM)
# ---------------------------------------------------------------------------


def test_scoring_logic():
    from augmentum.tools.preferred_sources import get_source_info

    print("\n" + "=" * 80)
    print("  SCORING LOGIC VERIFICATION (no LLM)")
    print("=" * 80)

    test_urls = [
        ("https://weather.gov/forecast", "realtime data"),
        ("https://en.wikipedia.org/wiki/Weather", "reference"),
        ("https://reddit.com/r/news", "forum"),
        ("https://data.gov/dataset", "government data"),
        ("https://example.com/article", "unknown domain"),
    ]

    for stype in ("fresh", "reference", "data"):
        print(f"\n  source_type: {stype}")
        print(f"  {'URL':<45} {'Info':<25} {'Score':>6}")
        print(f"  {'-' * 45} {'-' * 25} {'-' * 6}")

        scored: list[tuple[float, str, str]] = []
        for url, desc in test_urls:
            score = WebSearchTool._score_url_for_source_type(url, stype)
            info = get_source_info(url)
            info_str = f"{info.content_type}/{info.freshness}" if info else "unknown"
            scored.append((score, url, f"{desc} ({info_str})"))

        scored.sort(key=lambda x: -x[0])
        for i, (score, url, desc) in enumerate(scored):
            marker = " <-- TOP" if i == 0 else ""
            print(f"  {url:<45} {desc:<25} {score:6.1f}{marker}")

    print()


# ---------------------------------------------------------------------------
# LLM test runner
# ---------------------------------------------------------------------------


async def run_scenario(
    backend: _Backend,
    model: str,
    registry: ToolRegistry,
    scenario: Scenario,
    *,
    verbose: bool = False,
) -> tuple[str, str | None, float]:
    """Run a single scenario. Returns (status, source_type, elapsed)."""
    tools = registry.list_tools()
    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a helpful assistant with access to tools."),
            Message(role="user", content=scenario.query),
        ],
        stream=False,
    )

    inject_tool_schemas(request, tools, backend)
    start = time.monotonic()

    try:
        response = await asyncio.wait_for(backend.chat(request), timeout=60.0)
    except asyncio.TimeoutError:
        return "TIMEOUT", None, 60.0
    except Exception as exc:
        return f"ERROR: {exc}", None, time.monotonic() - start

    elapsed = time.monotonic() - start
    calls = parse_tool_calls(response, tools, backend)

    if not calls:
        if verbose:
            content = (response.message.content or "")[:150] if response.message else ""
            print(f"    No tool call. Response: {content}")
        return "NO_TOOL", None, elapsed

    call = calls[0]
    if call.name != "web_search":
        return f"WRONG_TOOL:{call.name}", None, elapsed

    stype = call.args.get("source_type")

    if verbose:
        print(f"    query={call.args.get('query', '')!r}  source_type={stype!r}")

    if not stype:
        return "NO_TYPE", None, elapsed

    if stype == scenario.expected:
        return "PASS", stype, elapsed
    if stype in scenario.acceptable:
        return "OK", stype, elapsed
    return f"WRONG:{stype}", stype, elapsed


async def run_tests(args: argparse.Namespace) -> None:
    base_url = args.url.rstrip("/")

    if args.model:
        models = args.model
    else:
        import httpx
        print(f"\nDiscovering models at {base_url}...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/models")
                data = resp.json()
                all_models = [
                    m["id"] for m in data.get("data", [])
                    if "embed" not in m["id"].lower()
                ]
        except Exception as exc:
            print(f"  Failed: {exc}")
            return
        if not all_models:
            print("No models found!")
            return
        # Pick 4 models across different sizes
        print(f"Found {len(all_models)} model(s), selecting up to 4...")
        models = all_models[:4]

    print(f"Testing: {', '.join(models)}\n")

    registry = ToolRegistry()
    registry.register(StubWebSearch())

    for model in models:
        print(f"{'=' * 70}")
        print(f"  {model}  ({len(SCENARIOS)} scenarios)")
        print(f"{'=' * 70}")

        backend = _Backend(base_url, timeout=args.timeout)
        counts = {"PASS": 0, "OK": 0, "NO_TYPE": 0, "fail": 0}

        for sc in SCENARIOS:
            status, stype, elapsed = await run_scenario(
                backend, model, registry, sc, verbose=args.verbose,
            )

            if status == "PASS":
                icon, key = "PASS", "PASS"
            elif status == "OK":
                icon, key = "OK  ", "OK"
            elif status in ("NO_TYPE", "NO_TOOL"):
                icon, key = "SKIP", "NO_TYPE"
            else:
                icon, key = "FAIL", "fail"

            stype_str = stype or "-"
            print(f"  [{icon}] {sc.name:<20} expect={sc.expected:<10} got={stype_str:<10} {elapsed:5.1f}s")
            if key == "fail" and args.verbose:
                print(f"         {status}")

            counts[key] += 1

        total = len(SCENARIOS)
        effective = (counts["PASS"] + counts["OK"]) / max(total, 1) * 100
        print(f"\n  {counts['PASS']} pass, {counts['OK']} ok, "
              f"{counts['NO_TYPE']} skip, {counts['fail']} fail  "
              f"({effective:.0f}%)\n")

        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Source type benchmark")
    parser.add_argument("--url", default="http://localhost:1234/v1")
    parser.add_argument("--model", action="append", help="Model name (repeatable, max 4)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--scoring-only", action="store_true")
    args = parser.parse_args()

    print("Augmentum Source Type Benchmark")
    print(f"Backend: {args.url}\n")

    test_scoring_logic()

    if not args.scoring_only:
        asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
