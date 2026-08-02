#!/usr/bin/env python3
"""Automated testing harness for the coder agent across multiple models.

Tests the full pipeline: plan → strategy selection → tool execution → verification.
Run from inside the Augmentum container or from the host with Docker access.

Usage:
    # Test all available models
    python tests/test_coder_agent.py

    # Test specific models
    python tests/test_coder_agent.py --models "gemma-3-4b-it,nemo:latest"

    # Test only single-call (Direct) strategy
    python tests/test_coder_agent.py --strategy direct

    # Verbose output
    python tests/test_coder_agent.py -v
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://127.0.0.1:6100"

_REACT_FAMILIES = {"qwen", "deepseek", "llama3", "mistral", "gpt-4", "claude", "gemini"}

def _guess_strategy(model: str) -> str:
    """Fallback strategy guess when augmentum package not importable."""
    ml = model.lower()
    for fam in _REACT_FAMILIES:
        if fam in ml:
            return "react"
    return "rewoo"


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------

@dataclass
class TestScenario:
    """A test case for the coder agent."""
    name: str
    request: str
    expect_tool_calls: list[str]  # Tool names that should be called
    expect_file_exists: str = ""  # File that should exist after
    expect_output_contains: str = ""  # String in terminal/tool output
    expect_question: bool = False  # Should the agent ask a question?
    difficulty: str = "simple"  # simple, medium, complex
    max_time_s: int = 60


SCENARIOS_BASIC = [
    TestScenario(
        name="create_simple_file",
        request="create a file called hello.py that contains print('hello world')",
        expect_tool_calls=["file_write"],
        expect_file_exists="/workspace/hello.py",
        difficulty="simple",
    ),
    TestScenario(
        name="list_workspace",
        request="list all files in the workspace",
        expect_tool_calls=["file_list"],
        difficulty="simple",
    ),
    TestScenario(
        name="create_and_run",
        request="create a python script called add.py that prints 2+2, then run it to verify",
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/add.py",
        expect_output_contains="4",
        difficulty="medium",
    ),
    TestScenario(
        name="read_and_edit",
        request="read hello.py and add a comment at the top saying '# greeting script'",
        expect_tool_calls=["file_read", "code_edit"],
        difficulty="medium",
    ),
    TestScenario(
        name="run_failing_command",
        request="run 'python3 nonexistent.py' and explain what happened",
        expect_tool_calls=["shell_exec"],
        difficulty="simple",
    ),
    TestScenario(
        name="create_with_test",
        request="create a function in math_utils.py that calculates factorial, then write a test for it in test_math.py and run the test",
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/math_utils.py",
        difficulty="complex",
        max_time_s=90,
    ),
    TestScenario(
        name="vague_request_should_ask",
        request="create a file",
        expect_question=True,
        expect_tool_calls=[],
        difficulty="simple",
    ),
    TestScenario(
        name="search_codebase",
        request="find all files that contain the word 'print'",
        expect_tool_calls=["code_grep"],
        difficulty="simple",
    ),
]

# ---------------------------------------------------------------------------
# Complex Scenarios — multi-step reasoning, cross-file, integration patterns
# ---------------------------------------------------------------------------

SCENARIOS_COMPLEX = [
    # --- Multi-file project generation ---
    TestScenario(
        name="flask_api",
        request=(
            "Create a minimal Flask REST API with two endpoints: "
            "GET /api/items returns a JSON list of items, "
            "POST /api/items adds a new item. "
            "Store items in a Python list. "
            "Include a requirements.txt with flask. "
            "Then run 'python3 -c \"import flask; print(flask.__version__)\"' to verify flask is importable."
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/app.py",
        difficulty="complex",
        max_time_s=120,
    ),

    # --- Read + multi-edit refactor ---
    TestScenario(
        name="refactor_module",
        request=(
            "First create math_lib.py with a function 'add(a, b)' that returns a+b. "
            "Then read it back, add a 'subtract(a, b)' function to the same file. "
            "Then create test_lib.py that tests both functions using assert statements. "
            "Then run test_lib.py to verify."
        ),
        expect_tool_calls=["file_write", "file_read", "shell_exec"],
        expect_file_exists="/workspace/math_lib.py",
        difficulty="complex",
        max_time_s=120,
    ),

    # --- CLI tool with argparse ---
    TestScenario(
        name="cli_tool",
        request=(
            "Create a CLI tool called 'converter.py' that converts between Celsius and Fahrenheit. "
            "It should use argparse with: converter.py --to-f 100  (outputs 212.0) and "
            "converter.py --to-c 212  (outputs 100.0). "
            "Then run 'python3 converter.py --to-f 100' to verify it outputs 212"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/converter.py",
        expect_output_contains="212",
        difficulty="complex",
        max_time_s=90,
    ),

    # --- Data processing pipeline ---
    TestScenario(
        name="data_pipeline",
        request=(
            "Create a data processing script called 'pipeline.py' that: "
            "1) creates a CSV file 'data.csv' with columns: name,score (5 rows of sample data) "
            "2) reads the CSV and calculates the average score "
            "3) writes the result to 'report.txt' with the format 'Average score: X.X' "
            "Then run the script and show the contents of report.txt"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/pipeline.py",
        expect_output_contains="Average",
        difficulty="complex",
        max_time_s=90,
    ),

    # --- Git workflow ---
    TestScenario(
        name="git_workflow",
        request=(
            "Check the git status, then create a .gitignore file that ignores "
            "__pycache__/, *.pyc, .env, and node_modules/. "
            "Then run 'git add .gitignore && git commit -m \"Add gitignore\"' "
            "and show the git log"
        ),
        expect_tool_calls=["shell_exec", "file_write"],
        expect_file_exists="/workspace/.gitignore",
        expect_output_contains="gitignore",
        difficulty="medium",
        max_time_s=60,
    ),

    # --- Debug a broken file ---
    TestScenario(
        name="debug_broken_code",
        request=(
            "Create a file called 'broken.py' with this content:\n"
            "def greet(name)\n"
            "    return f'Hello {name}'\n"
            "print(greet('World'))\n\n"
            "Then run it. It will fail. Fix the syntax error and run it again "
            "until it prints 'Hello World'"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_output_contains="Hello World",
        difficulty="complex",
        max_time_s=120,
    ),

    # --- Install package + use it ---
    TestScenario(
        name="install_and_use_package",
        request=(
            "Install the 'requests' library using pip, then create a script called 'check_http.py' "
            "that imports requests and prints requests.__version__. Run it to verify."
        ),
        expect_tool_calls=["shell_exec", "file_write"],
        expect_file_exists="/workspace/check_http.py",
        difficulty="complex",
        max_time_s=180,
    ),

    # --- Multi-file project with imports ---
    TestScenario(
        name="multi_module_project",
        request=(
            "Create a small project with 3 files:\n"
            "1) 'models.py' with a dataclass called 'User' with fields: name (str), email (str), age (int)\n"
            "2) 'utils.py' with a function 'validate_email(email)' that returns True if '@' is in the email\n"
            "3) 'main.py' that imports User and validate_email, creates a user, validates the email, and prints the result\n"
            "Then run main.py"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/models.py",
        expect_output_contains="True",
        difficulty="complex",
        max_time_s=90,
    ),

    # --- Glob + grep exploration ---
    TestScenario(
        name="explore_and_summarize",
        request=(
            "Find all Python files in the workspace, then grep for any function definitions (lines starting with 'def '). "
            "List what you find."
        ),
        expect_tool_calls=["find_files"],
        difficulty="medium",
        max_time_s=60,
    ),

    # --- Shell scripting ---
    TestScenario(
        name="bash_script",
        request=(
            "Create a bash script called 'setup.sh' that: "
            "1) creates a directory called 'project' "
            "2) creates project/__init__.py (empty) "
            "3) creates project/config.json with {\"version\": \"1.0\"} "
            "4) prints 'Setup complete!' "
            "Make it executable and run it"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/setup.sh",
        expect_output_contains="Setup complete",
        difficulty="complex",
        max_time_s=90,
    ),

    # --- JSON manipulation ---
    TestScenario(
        name="json_transform",
        request=(
            "Create a script called 'transform.py' that: "
            "1) Creates a JSON file 'users.json' with an array of 3 users (each with name, role) "
            "2) Reads it, filters to only users with role='admin' "
            "3) Writes the filtered list to 'admins.json' "
            "4) Prints how many admins were found "
            "Run it."
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/transform.py",
        difficulty="complex",
        max_time_s=90,
    ),

    # --- Web scraper skeleton ---
    TestScenario(
        name="async_http_server",
        request=(
            "Create a simple HTTP server in 'server.py' using only the built-in http.server module. "
            "It should serve on port 8888 and return JSON {\"status\": \"ok\"} for all requests. "
            "Don't start it — just verify it parses correctly with 'python3 -c \"import server\"'"
        ),
        expect_tool_calls=["file_write", "shell_exec"],
        expect_file_exists="/workspace/server.py",
        difficulty="complex",
        max_time_s=90,
    ),
]

# Default: basic scenarios only. Use --complex or --all for more.
SCENARIOS = SCENARIOS_BASIC


# ---------------------------------------------------------------------------
# Test Results
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    scenario: str
    model: str
    strategy: str
    success: bool
    duration_ms: int
    tool_calls_made: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    response_preview: str = ""
    question_asked: bool = False


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

async def get_available_models(session: aiohttp.ClientSession) -> list[str]:
    """Fetch available models from the API."""
    async with session.get(f"{BASE_URL}/api/tags") as r:
        if r.status != 200:
            return []
        data = await r.json()
        models = data.get("models", [])
        # Filter out prefixed models
        return [
            m["name"] for m in models
            if not m["name"].startswith(("a/", "n/", "p/", "g/", "c/"))
        ]


async def ensure_workspace(session: aiohttp.ClientSession) -> str:
    """Ensure a test workspace exists and is running. Returns workspace ID."""
    async with session.get(f"{BASE_URL}/api/coder/workspaces") as r:
        data = await r.json()
        workspaces = data.get("workspaces", [])

    # Find or create test workspace
    for ws in workspaces:
        if ws["name"] == "agent-test" and ws["status"] == "running":
            return ws["id"]

    # Create new workspace
    async with session.post(f"{BASE_URL}/api/coder/workspaces", json={"name": "agent-test"}) as r:
        data = await r.json()
        ws_id = data.get("id", "")

    if ws_id:
        print(f"  Created workspace {ws_id[:8]}, waiting for setup...")
        # Wait for container to be ready (apt-get + git init)
        # Check for .git directory — it's created AFTER apt-get finishes
        for i in range(30):
            await asyncio.sleep(5)
            try:
                async with session.get(
                    f"{BASE_URL}/api/coder/files/{ws_id}",
                    params={"path": "/workspace"},
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json()
                    files = data.get("files", [])
                    names = [f.get("name", "") for f in files]
                    if ".git" in names:
                        print(f"  Workspace ready after {(i+1)*5}s (git init complete)")
                        return ws_id
            except Exception:
                pass
        print("  WARNING: Workspace may not be fully ready")
    return ws_id


async def clean_workspace(session: aiohttp.ClientSession, ws_id: str):
    """Remove test files from workspace."""
    test_files = ["hello.py", "add.py", "math_utils.py", "test_math.py", "fib.py", "test_fib.py"]
    for f in test_files:
        # Use shell to remove
        pass  # Files accumulate — that's OK for testing


async def run_scenario(
    session: aiohttp.ClientSession,
    scenario: TestScenario,
    model: str,
    ws_id: str,
    verbose: bool = False,
) -> TestResult:
    """Run a single test scenario against a model."""
    t0 = time.monotonic()
    tool_calls = []
    errors = []
    full_response = ""
    question_asked = False

    try:
        # Send the request
        resp = await session.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": scenario.request}],
                "stream": True,
            },
            headers={
                "X-Augmentum-Mode": "coder",
                "X-Augmentum-Workspace": ws_id,
            },
            timeout=aiohttp.ClientTimeout(total=scenario.max_time_s),
        )

        # Parse streaming NDJSON response
        async for line in resp.content:
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            try:
                chunk = json.loads(line_str)
                # Content — check both Ollama and OpenAI formats
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    # Try OpenAI format
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                if content:
                    full_response += content

                # Augmentum metadata
                aug = chunk.get("augmentum", {})
                if aug.get("status") == "tool_call":
                    tc = aug.get("tool_call", {})
                    tool_name = tc.get("tool", "")
                    if tool_name:
                        tool_calls.append(tool_name)
                        if verbose:
                            print(f"      >> {tool_name}: {json.dumps(tc.get('input', {}))[:100]}")
                if aug.get("status") == "tool_result":
                    tr = aug.get("tool_result", {})
                    if verbose:
                        print(f"      << {tr.get('tool', '')}: success={tr.get('success')} preview={tr.get('output_preview', '')[:80]}")
                    if not tr.get("success", True):
                        errors.append(f"{tr.get('tool', '')}: {tr.get('output_preview', '')[:100]}")
                    # Capture tool output for expect_output_contains checks
                    tool_output = tr.get("output_preview", "")
                    if tool_output:
                        full_response += tool_output
                if aug.get("status") == "error":
                    errors.append("Stream error")
                if aug.get("strategy"):
                    strategy_used = aug["strategy"]

            except json.JSONDecodeError:
                continue

    except TimeoutError:
        errors.append(f"Timed out after {scenario.max_time_s}s")
    except Exception as exc:
        errors.append(f"Request failed: {str(exc)[:200]}")

    duration_ms = int((time.monotonic() - t0) * 1000)

    # Check if the agent asked a question
    if "?" in full_response and not tool_calls:
        question_asked = True

    # Determine success
    success = True
    check_errors = []

    # Check expected tool calls
    if scenario.expect_tool_calls:
        for expected in scenario.expect_tool_calls:
            if expected not in tool_calls:
                check_errors.append(f"Expected tool '{expected}' not called. Called: {tool_calls}")
                success = False

    # Check expected question
    if scenario.expect_question:
        if not question_asked:
            check_errors.append("Expected a clarifying question but agent acted directly")
            success = False
    elif scenario.expect_tool_calls and not tool_calls and not question_asked:
        check_errors.append("No tool calls made and no question asked")
        success = False

    # Check file exists
    if scenario.expect_file_exists and success:
        try:
            async with session.get(
                f"{BASE_URL}/api/coder/files/{ws_id}/read",
                params={"path": scenario.expect_file_exists},
            ) as r:
                if r.status != 200:
                    check_errors.append(f"Expected file {scenario.expect_file_exists} not found")
                    success = False
        except Exception:
            pass

    # Check output contains
    if scenario.expect_output_contains and success:
        if scenario.expect_output_contains not in full_response:
            check_errors.append(f"Expected '{scenario.expect_output_contains}' in output")
            success = False

    errors.extend(check_errors)

    # Determine strategy used
    strategy = "unknown"
    try:
        from augmentum.coder.harness import select_harness
        strategy = select_harness(model)
    except ImportError:
        strategy = _guess_strategy(model)

    result = TestResult(
        scenario=scenario.name,
        model=model,
        strategy=strategy,
        success=success,
        duration_ms=duration_ms,
        tool_calls_made=tool_calls,
        errors=errors,
        response_preview=full_response[:200].replace("\n", " "),
        question_asked=question_asked,
    )

    if verbose:
        status = "✓" if success else "✗"
        print(f"    {status} {scenario.name} ({duration_ms}ms) tools={tool_calls}")
        if errors:
            for e in errors:
                print(f"      ERROR: {e}")
        if full_response:
            print(f"      Response: {full_response[:150]}...")
        else:
            print("      Response: <empty>")

    return result


async def run_all_tests(
    models: list[str] | None = None,
    strategy_filter: str | None = None,
    verbose: bool = False,
):
    """Run all test scenarios against all (or specified) models."""
    async with aiohttp.ClientSession() as session:
        # Get available models
        if models:
            available = models
        else:
            available = await get_available_models(session)
            if not available:
                print("ERROR: No models available. Check LM Studio / Ollama.")
                return

        print(f"\n{'='*60}")
        print("CODER AGENT TEST HARNESS")
        print(f"{'='*60}")
        print(f"Models: {', '.join(available)}")
        print(f"Scenarios: {len(SCENARIOS)}")
        print(f"Strategy filter: {strategy_filter or 'all'}")

        # Ensure test workspace
        print("\nSetting up workspace...")
        ws_id = await ensure_workspace(session)
        if not ws_id:
            print("ERROR: Could not create workspace")
            return
        print(f"Workspace: {ws_id[:8]}")

        # Build codebase index
        print("Building codebase index...")
        async with session.post(f"{BASE_URL}/api/coder/index/{ws_id}") as r:
            data = await r.json()
            print(f"  Index: {data.get('total_chunks', 0)} chunks")

        # Run tests
        all_results: list[TestResult] = []

        for model in available:
            try:
                from augmentum.coder.harness import select_harness as _sh  # noqa: F811
                strategy = _sh(model)
            except ImportError:
                strategy = _guess_strategy(model)

            if strategy_filter and strategy != strategy_filter:
                print(f"\n  Skipping {model} (strategy: {strategy}, filter: {strategy_filter})")
                continue

            print(f"\n{'-'*60}")
            print(f"Model: {model}")
            print(f"Strategy: {strategy}")
            print(f"{'-'*60}")

            for scenario in SCENARIOS:
                # Skip complex scenarios for speed (can be enabled with --all)
                result = await run_scenario(session, scenario, model, ws_id, verbose)
                all_results.append(result)

                if not verbose:
                    status = "✓" if result.success else "✗"
                    q = " (asked question)" if result.question_asked else ""
                    print(f"  {status} {scenario.name}: {result.duration_ms}ms "
                          f"tools={result.tool_calls_made}{q}")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        # Per-model summary
        for model in available:
            model_results = [r for r in all_results if r.model == model]
            if not model_results:
                continue
            passed = sum(1 for r in model_results if r.success)
            total = len(model_results)
            avg_ms = sum(r.duration_ms for r in model_results) // max(total, 1)
            strategy = model_results[0].strategy
            pct = (passed / total * 100) if total else 0

            bar = "#" * int(pct / 10) + "." * (10 - int(pct / 10))
            print(f"  {model:30s} {bar} {passed}/{total} ({pct:.0f}%) avg={avg_ms}ms [{strategy}]")

        # Overall
        total_passed = sum(1 for r in all_results if r.success)
        total_tests = len(all_results)
        print(f"\n  Overall: {total_passed}/{total_tests} passed")

        # Per-scenario summary
        print("\n  Per scenario:")
        for scenario in SCENARIOS:
            s_results = [r for r in all_results if r.scenario == scenario.name]
            if not s_results:
                continue
            passed = sum(1 for r in s_results if r.success)
            total = len(s_results)
            print(f"    {scenario.name:30s} {passed}/{total}")

        # Failed tests detail
        failed = [r for r in all_results if not r.success]
        if failed:
            print("\n  Failed tests:")
            for r in failed:
                print(f"    {r.model} / {r.scenario}: {r.errors[0] if r.errors else 'unknown'}")

        # Save results to JSON
        results_path = Path("tests/eval_results/coder_agent_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": available,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "results": [
                {
                    "scenario": r.scenario,
                    "model": r.model,
                    "strategy": r.strategy,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "tool_calls": r.tool_calls_made,
                    "errors": r.errors,
                    "question_asked": r.question_asked,
                }
                for r in all_results
            ],
        }
        results_path.write_text(json.dumps(results_data, indent=2))
        print(f"\n  Results saved to {results_path}")

        # Cleanup
        print("\nCleaning up test workspace...")
        async with session.delete(f"{BASE_URL}/api/coder/workspaces/{ws_id}") as r:
            print(f"  Deleted: {r.status}")

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="Test coder agent across models")
    parser.add_argument("--models", "-m", help="Comma-separated model names")
    parser.add_argument("--strategy", "-s", choices=["direct", "architect", "react"],
                        help="Only test models using this strategy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--complex", action="store_true", help="Run complex scenarios only")
    parser.add_argument("--all", action="store_true", help="Run all scenarios (basic + complex)")
    parser.add_argument("--base-url", default=BASE_URL, help="Augmentum API base URL")
    args = parser.parse_args()

    BASE_URL = args.base_url

    # Select scenario set
    global SCENARIOS
    if args.all:
        SCENARIOS = SCENARIOS_BASIC + SCENARIOS_COMPLEX
    elif args.complex:
        SCENARIOS = SCENARIOS_COMPLEX

    models = args.models.split(",") if args.models else None
    asyncio.run(run_all_tests(models=models, strategy_filter=args.strategy, verbose=args.verbose))


if __name__ == "__main__":
    main()
