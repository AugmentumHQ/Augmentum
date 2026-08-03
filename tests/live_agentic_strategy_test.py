"""Live agentic strategy benchmark — tests execution strategies against real models.

Compares three execution strategies across real LLM backends:

1. **Single-shot** — One LLM call with tools, parse and execute, return. No feedback loop.
   (Baseline: what the current agentic handler does.)

2. **ReAct loop** — Model calls tools, sees results, decides next action, until done.
   (The standard agentic pattern used by Claude Code, OpenAI Agents SDK, etc.)

3. **REWOO (planned execution)** — Model plans all tool calls upfront with variable
   references (#E1, #E2...). Tools execute without the model in the loop. Model
   synthesizes final answer from collected results.
   (Token-efficient alternative for weaker models.)

Each strategy runs identical tasks against identical models with real tool execution
(Wikipedia API, calculator, web search if SearXNG available).

Usage:
    .venv/Scripts/python tests/live_agentic_strategy_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model (default: discover all loaded models)
    --verbose / -v      Show full model outputs and tool results
    --json PATH         Export results as JSON
    --timeout SECS      Per-call timeout (default: 90)
    --searxng URL       SearXNG URL (default: http://localhost:8080)
    --skip-search       Skip tasks that require web search
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
)
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    select_tier,
    tools_to_native_format,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.parsing import (
    build_text_tool_prompt,
    coerce_and_execute,
    parse_tool_calls,
)

# ---------------------------------------------------------------------------
# Real tools (hit real APIs)
# ---------------------------------------------------------------------------


class RealCalculator(Tool):
    """Calculator — evaluates math expressions safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. Supports basic arithmetic, "
            "sqrt, log, sin, cos, etc. Input must be a valid math expression."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g. '347 * 892')",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        from augmentum.tools.calculator import CalculatorTool
        real = CalculatorTool()
        return await real.execute(**kwargs)


class RealWikipedia(Tool):
    """Wikipedia — hits real MediaWiki API."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Search Wikipedia and retrieve article summaries. "
            "Use for factual lookups, definitions, historical events, biographies. "
            "IMPORTANT: Use short topic names as queries, NOT full questions."
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
                    "description": "Short topic name to look up (e.g. 'Battle of Hastings')",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        from augmentum.tools.wikipedia import WikipediaTool
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        real = WikipediaTool(self._client)
        return await real.execute(**kwargs)


class RealWebSearch(Tool):
    """Web search — hits real SearXNG instance."""

    def __init__(self, searxng_url: str = "http://localhost:8080"):
        self._searxng_url = searxng_url

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information. Returns titles, URLs, and snippets."

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
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="No query provided")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self._searxng_url}/search",
                    params={"q": query, "format": "json", "categories": "general"},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])[:5]
                if not results:
                    return ToolResult(success=True, output="No results found.")
                lines = []
                for r in results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("content", "")[:200]
                    lines.append(f"- {title}\n  {url}\n  {snippet}")
                return ToolResult(success=True, output="\n\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}")


# ---------------------------------------------------------------------------
# Test tasks — each has a query, required tools, and verification
# ---------------------------------------------------------------------------


@dataclass
class TaskVerdict:
    correct: bool
    explanation: str
    partial_credit: float = 0.0  # 0.0 - 1.0


@dataclass
class TestTask:
    id: str
    name: str
    query: str
    required_tools: list[str]  # tools that SHOULD be called
    requires_search: bool = False  # skip if SearXNG unavailable

    def verify(self, answer: str, tool_calls_made: list[dict]) -> TaskVerdict:
        """Override in subclasses for task-specific verification."""
        raise NotImplementedError


class MathTask(TestTask):
    """Pure calculator task with known answer."""

    def __init__(self, id: str, name: str, query: str, expected: float, tolerance: float = 0.01):
        super().__init__(id=id, name=name, query=query, required_tools=["calculator"])
        self.expected = expected
        self.tolerance = tolerance

    def verify(self, answer: str, tool_calls_made: list[dict]) -> TaskVerdict:
        used_calc = any(tc["name"] == "calculator" for tc in tool_calls_made)
        # Extract numbers from the answer
        numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        for num_str in numbers:
            try:
                val = float(num_str)
                if abs(val - self.expected) / max(abs(self.expected), 1) < self.tolerance:
                    return TaskVerdict(
                        correct=True,
                        explanation=f"Found {val} (expected {self.expected}), calc_used={used_calc}",
                        partial_credit=1.0 if used_calc else 0.7,
                    )
            except ValueError:
                continue
        # Check if model got close
        for num_str in numbers:
            try:
                val = float(num_str)
                if abs(val - self.expected) / max(abs(self.expected), 1) < 0.1:
                    return TaskVerdict(
                        correct=False,
                        explanation=f"Close ({val}) but not within tolerance of {self.expected}",
                        partial_credit=0.5,
                    )
            except ValueError:
                continue
        return TaskVerdict(correct=False, explanation=f"Expected ~{self.expected}, not found in answer")


class KeywordTask(TestTask):
    """Task verified by presence of keywords in the answer."""

    def __init__(self, id: str, name: str, query: str, required_tools: list[str],
                 keywords: list[str], requires_search: bool = False,
                 min_keywords: int = 0):
        super().__init__(id=id, name=name, query=query,
                         required_tools=required_tools, requires_search=requires_search)
        self.keywords = [k.lower() for k in keywords]
        self.min_keywords = min_keywords or max(1, len(keywords) // 2)

    def verify(self, answer: str, tool_calls_made: list[dict]) -> TaskVerdict:
        answer_lower = answer.lower()
        found = [k for k in self.keywords if k in answer_lower]
        missing = [k for k in self.keywords if k not in answer_lower]
        tools_used = {tc["name"] for tc in tool_calls_made}
        tools_ok = all(t in tools_used for t in self.required_tools)

        if len(found) >= self.min_keywords:
            credit = len(found) / len(self.keywords)
            if tools_ok:
                credit = min(1.0, credit + 0.1)
            return TaskVerdict(
                correct=True,
                explanation=f"Found {len(found)}/{len(self.keywords)} keywords, tools_ok={tools_ok}",
                partial_credit=credit,
            )
        return TaskVerdict(
            correct=False,
            explanation=f"Found {len(found)}/{len(self.keywords)}: {found}. Missing: {missing}",
            partial_credit=len(found) / len(self.keywords),
        )


class MultiStepMathTask(TestTask):
    """Task requiring both information lookup and calculation."""

    def __init__(self, id: str, name: str, query: str, required_tools: list[str],
                 check_fn, requires_search: bool = False):
        super().__init__(id=id, name=name, query=query,
                         required_tools=required_tools, requires_search=requires_search)
        self.check_fn = check_fn

    def verify(self, answer: str, tool_calls_made: list[dict]) -> TaskVerdict:
        return self.check_fn(answer, tool_calls_made)


# ---------------------------------------------------------------------------
# Task suite
# ---------------------------------------------------------------------------


def build_task_suite() -> list[TestTask]:
    """Build the test task suite."""
    return [
        # --- Tier 1: Single tool, verifiable answer ---
        MathTask(
            "calc_1", "Basic arithmetic",
            "What is 347 multiplied by 892?",
            expected=309524.0,
        ),
        MathTask(
            "calc_2", "Multi-step math",
            "What is the square root of 144 plus 37 times 5?",
            expected=197.0,  # sqrt(144) + 37*5 = 12 + 185 = 197
        ),
        KeywordTask(
            "wiki_1", "Wikipedia lookup",
            "What year was the Battle of Hastings and who won?",
            required_tools=["wikipedia"],
            keywords=["1066", "william", "norman", "harold"],
            min_keywords=2,
        ),
        KeywordTask(
            "wiki_2", "Wikipedia entity",
            "What is the chemical symbol for gold and what is its atomic number?",
            required_tools=["wikipedia"],
            keywords=["au", "79"],
            min_keywords=2,
        ),

        # --- Tier 2: Multi-tool, requires chaining ---
        MultiStepMathTask(
            "chain_1", "Wikipedia + Calculator",
            "The speed of light is approximately 299,792 km/s. "
            "How many kilometers does light travel in one hour? Use the calculator.",
            required_tools=["calculator"],
            check_fn=lambda answer, tc: TaskVerdict(
                correct=any(
                    abs(float(n.replace(",", "")) - 1_079_251_200) / 1_079_251_200 < 0.01
                    for n in re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
                    if _safe_float(n.replace(",", "")) is not None
                ),
                explanation="Expected ~1,079,251,200 km",
                partial_credit=0.5 if any(tc["name"] == "calculator" for tc in tc) else 0.0,
            ),
        ),
        KeywordTask(
            "chain_2", "Wikipedia + reasoning",
            "Look up Albert Einstein on Wikipedia. What year was he born, "
            "and what is his most famous equation?",
            required_tools=["wikipedia"],
            keywords=["1879", "e=mc", "relativity", "mass", "energy"],
            min_keywords=2,
        ),

        # --- Tier 3: Search-dependent (skipped if no SearXNG) ---
        KeywordTask(
            "search_1", "Web search factual",
            "Search the web for the current population of Tokyo.",
            required_tools=["web_search"],
            keywords=["tokyo", "million", "population"],
            min_keywords=2,
            requires_search=True,
        ),
        KeywordTask(
            "search_2", "Search + synthesis",
            "Search the web to find who won the most recent Super Bowl and what the final score was.",
            required_tools=["web_search"],
            keywords=["super bowl"],
            min_keywords=1,
            requires_search=True,
        ),

        # --- Tier 4: Complex multi-tool ---
        MultiStepMathTask(
            "complex_1", "Multi-tool chain",
            "Look up the height of Mount Everest in meters on Wikipedia, "
            "then use the calculator to convert it to feet (multiply by 3.28084).",
            required_tools=["wikipedia", "calculator"],
            check_fn=lambda answer, tc: _verify_everest_feet(answer, tc),
        ),
        KeywordTask(
            "complex_2", "Multi-source synthesis",
            "Look up both 'Python programming language' and 'Rust programming language' "
            "on Wikipedia. Compare when each was first released.",
            required_tools=["wikipedia"],
            keywords=["python", "rust", "1991", "2015", "guido"],
            min_keywords=3,
        ),
    ]


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _verify_everest_feet(answer: str, tool_calls: list[dict]) -> TaskVerdict:
    """Verify Everest height conversion: 8848.86m * 3.28084 = ~29031.7 ft."""
    used_wiki = any(tc["name"] == "wikipedia" for tc in tool_calls)
    used_calc = any(tc["name"] == "calculator" for tc in tool_calls)
    numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
    for n in numbers:
        val = _safe_float(n)
        if val and 28800 < val < 29200:  # reasonable range for Everest in feet
            return TaskVerdict(
                correct=True,
                explanation=f"Found {val} ft, wiki={used_wiki}, calc={used_calc}",
                partial_credit=1.0 if (used_wiki and used_calc) else 0.6,
            )
    # Check if they got meters right at least
    for n in numbers:
        val = _safe_float(n)
        if val and 8840 < val < 8860:
            return TaskVerdict(
                correct=False,
                explanation=f"Found meters ({val}) but not feet conversion",
                partial_credit=0.3,
            )
    return TaskVerdict(correct=False, explanation="Everest height in feet not found")


# ---------------------------------------------------------------------------
# Execution strategies
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    strategy: str
    task_id: str
    model: str
    answer: str
    tool_calls: list[dict]
    verdict: TaskVerdict
    llm_calls: int
    tool_executions: int
    total_tokens: int
    wall_time_ms: float
    error: str = ""


async def _call_llm(
    backend: OpenAIBackend, model: str, messages: list[Message],
    tools: list[Tool] | None = None, timeout: float = 90.0,
) -> InternalChatResponse:
    """Single LLM call with optional native tool schemas."""
    native_tools = None
    fmt = None
    tier = select_tier(backend, model)

    if tools:
        if tier == ToolCallingTier.NATIVE:
            native_tools = tools_to_native_format(tools)
        elif tier == ToolCallingTier.STRUCTURED:
            from augmentum.modes.analytical.tool_calling import build_structured_output_schema
            fmt = build_structured_output_schema(tools)

    req = InternalChatRequest(
        model=model,
        messages=messages,
        stream=False,
        tools=native_tools,
        format=fmt,
    )

    # For text-tier, inject tool prompt into last user message
    if tools and tier == ToolCallingTier.TEXT:
        prompt = build_text_tool_prompt(tools)
        last = messages[-1]
        req.messages[-1] = Message(
            role=last.role,
            content=f"{last.content}\n\n{prompt}",
        )

    resp = await asyncio.wait_for(backend.chat(req), timeout=timeout)
    return resp


def _extract_tokens(resp: InternalChatResponse) -> int:
    if resp.usage:
        return (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
    return 0


# --- Strategy 1: Single-shot ---

async def run_single_shot(
    backend: OpenAIBackend, model: str, task: TestTask,
    tools: list[Tool], timeout: float, verbose: bool,
) -> StrategyResult:
    """One LLM call → parse tool calls → execute → return."""
    start = time.monotonic()
    all_tool_calls: list[dict] = []
    total_tokens = 0
    llm_calls = 0
    tool_execs = 0

    messages = [
        Message(role="system", content=(
            "You are a helpful assistant with access to tools. "
            "Use the tools to answer the user's question accurately. "
            "After getting tool results, provide a clear final answer."
        )),
        Message(role="user", content=task.query),
    ]

    try:
        resp = await _call_llm(backend, model, messages, tools, timeout)
        llm_calls += 1
        total_tokens += _extract_tokens(resp)
        output = resp.message.content if resp.message else ""

        # Parse and execute tool calls
        parsed = parse_tool_calls(resp, tools, backend)
        tool_results = []
        for tc in parsed:
            tool = next((t for t in tools if t.name == tc.name), None)
            if not tool:
                continue
            result_text, _ = await coerce_and_execute(tool, tc.args, timeout=30.0)
            tool_execs += 1
            all_tool_calls.append({"name": tc.name, "args": tc.args, "result": result_text[:500]})
            tool_results.append(f"[{tc.name}]: {result_text}")

        # If we got tool results, combine with model output
        if tool_results:
            answer = output + "\n\n" + "\n".join(tool_results)
        else:
            answer = output

    except Exception as e:
        answer = ""
        return StrategyResult(
            strategy="single_shot", task_id=task.id, model=model,
            answer="", tool_calls=all_tool_calls, verdict=TaskVerdict(False, f"Error: {e}"),
            llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
            wall_time_ms=(time.monotonic() - start) * 1000, error=str(e),
        )

    verdict = task.verify(answer, all_tool_calls)
    elapsed = (time.monotonic() - start) * 1000

    if verbose:
        print(f"    [single_shot] Answer: {answer[:200]}...")
        for tc in all_tool_calls:
            print(f"    Tool: {tc['name']}({tc['args']}) → {tc['result'][:100]}")

    return StrategyResult(
        strategy="single_shot", task_id=task.id, model=model,
        answer=answer, tool_calls=all_tool_calls, verdict=verdict,
        llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
        wall_time_ms=elapsed,
    )


# --- Strategy 2: ReAct loop ---

async def run_react(
    backend: OpenAIBackend, model: str, task: TestTask,
    tools: list[Tool], timeout: float, verbose: bool,
    max_turns: int = 6,
) -> StrategyResult:
    """Standard ReAct: model calls tools, sees results, loops until done."""
    start = time.monotonic()
    all_tool_calls: list[dict] = []
    total_tokens = 0
    llm_calls = 0
    tool_execs = 0
    answer = ""

    messages = [
        Message(role="system", content=(
            "You are a helpful assistant with access to tools. "
            "Use the tools to research and calculate as needed. "
            "You can call tools multiple times. When you have enough information "
            "to answer the question fully, provide your final answer WITHOUT "
            "calling any more tools."
        )),
        Message(role="user", content=task.query),
    ]

    try:
        for turn in range(max_turns):
            resp = await _call_llm(backend, model, messages, tools, timeout)
            llm_calls += 1
            total_tokens += _extract_tokens(resp)
            output = resp.message.content if resp.message else ""

            parsed = parse_tool_calls(resp, tools, backend)

            if not parsed:
                # Model is done — no more tool calls
                answer = output
                break

            # Execute tools, feed results back
            messages.append(Message(role="assistant", content=output))

            tool_result_parts = []
            for tc in parsed:
                tool = next((t for t in tools if t.name == tc.name), None)
                if not tool:
                    continue
                result_text, _ = await coerce_and_execute(tool, tc.args, timeout=30.0)
                tool_execs += 1
                all_tool_calls.append({"name": tc.name, "args": tc.args, "result": result_text[:500]})
                tool_result_parts.append(
                    f"## Tool Result: {tc.name}\n{result_text}"
                )

            if tool_result_parts:
                messages.append(Message(
                    role="user",
                    content=(
                        "\n\n".join(tool_result_parts)
                        + "\n\nUse these results to continue. Call more tools if needed, "
                        "or provide your final answer."
                    ),
                ))
            else:
                answer = output
                break
        else:
            # Max turns reached — whatever we have is the answer
            answer = output

    except Exception as e:
        return StrategyResult(
            strategy="react", task_id=task.id, model=model,
            answer="", tool_calls=all_tool_calls, verdict=TaskVerdict(False, f"Error: {e}"),
            llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
            wall_time_ms=(time.monotonic() - start) * 1000, error=str(e),
        )

    verdict = task.verify(answer, all_tool_calls)
    elapsed = (time.monotonic() - start) * 1000

    if verbose:
        print(f"    [react] {llm_calls} LLM calls, {tool_execs} tool execs")
        print(f"    Answer: {answer[:200]}...")
        for tc in all_tool_calls:
            print(f"    Tool: {tc['name']}({tc['args']}) → {tc['result'][:100]}")

    return StrategyResult(
        strategy="react", task_id=task.id, model=model,
        answer=answer, tool_calls=all_tool_calls, verdict=verdict,
        llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
        wall_time_ms=elapsed,
    )


# --- Strategy 3: REWOO (Planned execution) ---

_REWOO_PLAN_PROMPT = """\
You are a planning assistant. Given a question and available tools, plan ALL the \
tool calls needed to answer it. Output a structured plan with variable references.

Format:
Plan: <your reasoning>
#E1 = tool_name(param="value")
Plan: <reasoning for next step>
#E2 = tool_name(param="value using #E1 if needed")
...

Rules:
- Reference earlier results with #E1, #E2, etc.
- Use ONLY these tools: {tool_list}
- Plan 1-5 tool calls. Don't over-plan.
- If no tools are needed, write: Plan: Answer directly.\
"""

_REWOO_SYNTH_PROMPT = """\
You are a helpful assistant. The user asked a question and tool calls have been \
executed to gather information. Use the results below to provide a clear, complete answer.

## Tool Results
{results}

Answer the user's question based on these results. Be precise and cite the data.\
"""


async def run_rewoo(
    backend: OpenAIBackend, model: str, task: TestTask,
    tools: list[Tool], timeout: float, verbose: bool,
) -> StrategyResult:
    """REWOO: Plan all tool calls upfront → execute → synthesize."""
    start = time.monotonic()
    all_tool_calls: list[dict] = []
    total_tokens = 0
    llm_calls = 0
    tool_execs = 0

    tool_list = ", ".join(
        f"{t.name}({_param_summary(t)})" for t in tools
    )

    # Step 1: Plan
    plan_messages = [
        Message(role="system", content=_REWOO_PLAN_PROMPT.format(tool_list=tool_list)),
        Message(role="user", content=task.query),
    ]

    try:
        plan_resp = await _call_llm(backend, model, plan_messages, timeout=timeout)
        llm_calls += 1
        total_tokens += _extract_tokens(plan_resp)
        plan_text = plan_resp.message.content if plan_resp.message else ""

        if verbose:
            print(f"    [rewoo] Plan:\n    {plan_text[:300]}...")

        # Step 2: Parse planned tool calls
        planned_calls = _parse_rewoo_plan(plan_text, {t.name for t in tools})

        # Step 3: Execute sequentially, substituting results
        evidence: dict[str, str] = {}  # #E1 -> result text

        for var, tc_name, tc_args_raw in planned_calls:
            # Substitute variable references in args
            tc_args = {}
            for k, v in tc_args_raw.items():
                if isinstance(v, str):
                    for evar, eresult in evidence.items():
                        v = v.replace(evar, eresult[:200])
                tc_args[k] = v

            tool = next((t for t in tools if t.name == tc_name), None)
            if not tool:
                evidence[var] = f"Error: tool '{tc_name}' not found"
                continue

            result_text, _ = await coerce_and_execute(tool, tc_args, timeout=30.0)
            tool_execs += 1
            evidence[var] = result_text
            all_tool_calls.append({"name": tc_name, "args": tc_args, "result": result_text[:500]})

        # Step 4: Synthesize
        results_text = "\n\n".join(
            f"### {var}: {tc_name}\n{evidence.get(var, 'No result')}"
            for var, tc_name, _ in planned_calls
        )
        if not results_text:
            results_text = "(No tool calls were planned)"

        synth_messages = [
            Message(role="system", content=_REWOO_SYNTH_PROMPT.format(results=results_text)),
            Message(role="user", content=task.query),
        ]

        synth_resp = await _call_llm(backend, model, synth_messages, timeout=timeout)
        llm_calls += 1
        total_tokens += _extract_tokens(synth_resp)
        answer = synth_resp.message.content if synth_resp.message else ""

    except Exception as e:
        return StrategyResult(
            strategy="rewoo", task_id=task.id, model=model,
            answer="", tool_calls=all_tool_calls, verdict=TaskVerdict(False, f"Error: {e}"),
            llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
            wall_time_ms=(time.monotonic() - start) * 1000, error=str(e),
        )

    verdict = task.verify(answer, all_tool_calls)
    elapsed = (time.monotonic() - start) * 1000

    if verbose:
        print(f"    [rewoo] {llm_calls} LLM calls, {tool_execs} tool execs")
        print(f"    Answer: {answer[:200]}...")

    return StrategyResult(
        strategy="rewoo", task_id=task.id, model=model,
        answer=answer, tool_calls=all_tool_calls, verdict=verdict,
        llm_calls=llm_calls, tool_executions=tool_execs, total_tokens=total_tokens,
        wall_time_ms=elapsed,
    )


def _param_summary(tool: Tool) -> str:
    """Short parameter summary for REWOO plan prompt."""
    schema = tool.input_schema or {}
    props = schema.get("properties", {})
    return ", ".join(f'{k}="..."' for k in props)


def _parse_rewoo_plan(plan_text: str, known_tools: set[str]) -> list[tuple[str, str, dict]]:
    """Parse REWOO plan into [(variable, tool_name, args), ...]."""
    results = []
    # Match: #E1 = tool_name(param="value", ...)
    pattern = re.compile(
        r'(#E\d+)\s*=\s*(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    )
    for match in pattern.finditer(plan_text):
        var = match.group(1)
        tool_name = match.group(2)
        args_str = match.group(3)

        if tool_name not in known_tools:
            continue

        # Parse args: key="value" or key='value' or key=value
        args: dict[str, str] = {}
        arg_pattern = re.compile(r'(\w+)\s*=\s*["\']?([^"\',$)]+)["\']?')
        for arg_match in arg_pattern.finditer(args_str):
            args[arg_match.group(1)] = arg_match.group(2).strip()

        results.append((var, tool_name, args))

    return results


# ---------------------------------------------------------------------------
# Discovery & orchestration
# ---------------------------------------------------------------------------


async def discover_models(base_url: str) -> list[str]:
    """Discover available models from LM Studio / OpenAI-compat endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return models
        except Exception as e:
            print(f"  Failed to discover models: {e}")
            return []


async def check_searxng(url: str) -> bool:
    """Check if SearXNG is available."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{url}/healthz")
            return resp.status_code == 200
        except Exception:
            pass
        # Fallback: try a search
        try:
            resp = await client.get(
                f"{url}/search",
                params={"q": "test", "format": "json"},
            )
            return resp.status_code == 200
        except Exception:
            return False


def build_tools(searxng_available: bool, searxng_url: str) -> list[Tool]:
    """Build the real tool list."""
    tools: list[Tool] = [RealCalculator(), RealWikipedia()]
    if searxng_available:
        tools.append(RealWebSearch(searxng_url))
    return tools


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: list[StrategyResult], models: list[str]):
    """Print a formatted results table."""
    # Group by model
    for model in models:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue

        print(f"\n{'=' * 90}")
        print(f"  MODEL: {model}")
        print(f"{'=' * 90}")
        print(f"  {'Task':<25} {'Strategy':<14} {'Pass':>5} {'Credit':>7} "
              f"{'LLM#':>5} {'Tool#':>6} {'Tokens':>7} {'Time':>8}")
        print(f"  {'-' * 25} {'-' * 14} {'-' * 5} {'-' * 7} "
              f"{'-' * 5} {'-' * 6} {'-' * 7} {'-' * 8}")

        task_ids_seen = []
        for r in model_results:
            if r.task_id not in task_ids_seen:
                task_ids_seen.append(r.task_id)

        for task_id in task_ids_seen:
            task_results = [r for r in model_results if r.task_id == task_id]
            for i, r in enumerate(task_results):
                task_name = r.task_id if i == 0 else ""
                passed = "Y" if r.verdict.correct else "N"
                credit = f"{r.verdict.partial_credit:.1%}"
                time_str = f"{r.wall_time_ms / 1000:.1f}s"
                print(
                    f"  {task_name:<25} {r.strategy:<14} {passed:>5} {credit:>7} "
                    f"{r.llm_calls:>5} {r.tool_executions:>6} {r.total_tokens:>7} {time_str:>8}"
                )
                if r.error:
                    print(f"    ERROR: {r.error[:70]}")
            if len(task_results) > 0:
                print()

    # Summary by strategy
    print(f"\n{'=' * 90}")
    print("  STRATEGY SUMMARY")
    print(f"{'=' * 90}")
    strategies = ["single_shot", "react", "rewoo"]
    for strat in strategies:
        strat_results = [r for r in results if r.strategy == strat]
        if not strat_results:
            continue
        passed = sum(1 for r in strat_results if r.verdict.correct)
        total = len(strat_results)
        avg_credit = sum(r.verdict.partial_credit for r in strat_results) / max(total, 1)
        avg_calls = sum(r.llm_calls for r in strat_results) / max(total, 1)
        avg_tools = sum(r.tool_executions for r in strat_results) / max(total, 1)
        avg_tokens = sum(r.total_tokens for r in strat_results) / max(total, 1)
        avg_time = sum(r.wall_time_ms for r in strat_results) / max(total, 1)
        print(
            f"  {strat:<14} pass={passed}/{total} ({passed/max(total,1):.0%})  "
            f"credit={avg_credit:.1%}  "
            f"avg_llm={avg_calls:.1f}  avg_tools={avg_tools:.1f}  "
            f"avg_tokens={avg_tokens:.0f}  avg_time={avg_time/1000:.1f}s"
        )

    # Summary by model
    print(f"\n{'=' * 90}")
    print("  MODEL SUMMARY")
    print(f"{'=' * 90}")
    for model in models:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue
        passed = sum(1 for r in model_results if r.verdict.correct)
        total = len(model_results)
        avg_credit = sum(r.verdict.partial_credit for r in model_results) / max(total, 1)
        print(f"  {model:<50} pass={passed}/{total} ({passed/max(total,1):.0%})  credit={avg_credit:.1%}")

    # Best strategy per model
    print(f"\n{'=' * 90}")
    print("  BEST STRATEGY PER MODEL")
    print(f"{'=' * 90}")
    for model in models:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue
        by_strat = {}
        for strat in strategies:
            sr = [r for r in model_results if r.strategy == strat]
            if sr:
                by_strat[strat] = sum(r.verdict.partial_credit for r in sr) / len(sr)
        if by_strat:
            best = max(by_strat, key=by_strat.get)
            print(f"  {model:<50} → {best} ({by_strat[best]:.1%} avg credit)")
            for s, c in sorted(by_strat.items(), key=lambda x: -x[1]):
                marker = " <<<" if s == best else ""
                print(f"    {s:<14} {c:.1%}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="Agentic strategy benchmark")
    parser.add_argument("--url", default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="Test a single model")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", default=None, help="Export results as JSON")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--searxng", default="http://localhost:8080")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("  AUGMENTUM AGENTIC STRATEGY BENCHMARK")
    print("  Testing: Single-Shot vs ReAct vs REWOO")
    print("=" * 90)

    # Discover models
    print(f"\n  Endpoint: {args.url}")
    if args.model:
        models = [args.model]
        print(f"  Model: {args.model}")
    else:
        print("  Discovering models...")
        models = await discover_models(args.url)
        if not models:
            print("  ERROR: No models found. Is LM Studio running?")
            return
        print(f"  Found {len(models)} model(s): {', '.join(models)}")

    # Check SearXNG
    searxng_available = False
    if not args.skip_search:
        print(f"\n  Checking SearXNG at {args.searxng}...")
        searxng_available = await check_searxng(args.searxng)
    if searxng_available:
        print("  SearXNG: available")
    else:
        print("  SearXNG: unavailable (search tasks will be skipped)")

    # Build tools and tasks
    tools = build_tools(searxng_available, args.searxng)
    tasks = build_task_suite()
    tasks = [t for t in tasks if not t.requires_search or searxng_available]

    print(f"\n  Tools: {', '.join(t.name for t in tools)}")
    print(f"  Tasks: {len(tasks)}")
    print("  Strategies: single_shot, react, rewoo")
    print(f"  Total runs: {len(models) * len(tasks) * 3}")

    # Create backend
    client = httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=10.0))
    backend = OpenAIBackend(client, args.url)

    strategies = [
        ("single_shot", run_single_shot),
        ("react", run_react),
        ("rewoo", run_rewoo),
    ]

    all_results: list[StrategyResult] = []

    for model in models:
        print(f"\n{'─' * 90}")
        print(f"  Testing model: {model}")
        print(f"{'─' * 90}")

        for task in tasks:
            print(f"\n  Task: {task.name} [{task.id}]")
            print(f"  Query: {task.query[:80]}...")

            task_tools = [t for t in tools if t.name in task.required_tools or
                          t.name == "calculator"]  # always include calculator
            # Also include all tools so the model can choose
            task_tools = tools

            for strat_name, strat_fn in strategies:
                try:
                    result = await strat_fn(
                        backend, model, task, task_tools, args.timeout, args.verbose,
                    )
                except Exception as e:
                    result = StrategyResult(
                        strategy=strat_name, task_id=task.id, model=model,
                        answer="", tool_calls=[], verdict=TaskVerdict(False, f"Fatal: {e}"),
                        llm_calls=0, tool_executions=0, total_tokens=0,
                        wall_time_ms=0, error=str(e),
                    )

                mark = "PASS" if result.verdict.correct else "FAIL"
                credit = f"{result.verdict.partial_credit:.0%}"
                print(
                    f"    {strat_name:<14} [{mark}] credit={credit:>4}  "
                    f"llm={result.llm_calls} tools={result.tool_executions} "
                    f"tokens={result.total_tokens} time={result.wall_time_ms/1000:.1f}s"
                )
                if result.error and args.verbose:
                    print(f"      Error: {result.error[:80]}")
                if not result.verdict.correct and args.verbose:
                    print(f"      Reason: {result.verdict.explanation[:80]}")

                all_results.append(result)

    await client.aclose()

    # Print summary tables
    print_results_table(all_results, models)

    # Export JSON
    if args.json:
        export = []
        for r in all_results:
            export.append({
                "strategy": r.strategy,
                "task_id": r.task_id,
                "model": r.model,
                "correct": r.verdict.correct,
                "partial_credit": r.verdict.partial_credit,
                "explanation": r.verdict.explanation,
                "llm_calls": r.llm_calls,
                "tool_executions": r.tool_executions,
                "total_tokens": r.total_tokens,
                "wall_time_ms": r.wall_time_ms,
                "tool_calls": r.tool_calls,
                "error": r.error,
            })
        Path(args.json).write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"\n  Results exported to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
