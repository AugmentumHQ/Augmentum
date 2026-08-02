"""Systematic prompt testing for the app builder PLAN + GENERATE pipeline.

Tests the full build prompts (not edit/fix) against real models to measure:
1. Plan quality (valid file list, reasonable file count, good descriptions)
2. Generate quality (valid code, correct format, complete implementation)
3. Cross-file consistency (IDs referenced in JS exist in HTML, etc.)

Usage:
    python tests/test_build_prompts.py                        # Default model
    python tests/test_build_prompts.py --model qwen3.5:7b
    python tests/test_build_prompts.py --prompt-variants       # Test all prompt styles
    python tests/test_build_prompts.py --test plan_only        # Only test plan phase
    python tests/test_build_prompts.py --test generate_only    # Only test generate phase
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Build Descriptions (what users actually ask to build)
# ---------------------------------------------------------------------------

BUILD_DESCRIPTIONS = [
    {
        "id": "simple_counter",
        "description": "a simple click counter with increment, decrement, and reset buttons",
        "expected_files": 3,  # HTML + CSS + JS
        "expected_roles": ["entry", "style", "script"],
        "code_patterns": [r"addEventListener", r"getElementById", r"textContent"],
    },
    {
        "id": "todo_app",
        "description": "a todo list app where you can add, complete, and delete tasks with localStorage persistence",
        "expected_files": 3,
        "expected_roles": ["entry", "style", "script"],
        "code_patterns": [r"localStorage", r"addEventListener", r"createElement|innerHTML"],
    },
    {
        "id": "calculator",
        "description": "a calculator with basic math operations (add, subtract, multiply, divide) and a clean display",
        "expected_files": 3,
        "expected_roles": ["entry", "style", "script"],
        "code_patterns": [r"\+|\-|\*|\/|eval|calculate|operate"],
    },
    {
        "id": "snake_game",
        "description": "a classic snake game on an HTML canvas with arrow key controls, food, growing snake, and game over",
        "expected_files": 3,
        "expected_roles": ["entry", "style", "script"],
        "code_patterns": [r"canvas|getContext|requestAnimationFrame", r"keydown|ArrowUp|ArrowDown"],
    },
    {
        "id": "weather_dashboard",
        "description": "a weather dashboard that shows temperature, humidity, wind speed with a search bar for city names (use fake/mock data)",
        "expected_files": 3,
        "expected_roles": ["entry", "style", "script"],
        "code_patterns": [r"temperature|humidity|wind", r"search|input|city"],
    },
]

# ---------------------------------------------------------------------------
# Plan Prompt Variants
# ---------------------------------------------------------------------------

PLAN_VARIANTS = {
    "current": {
        "name": "Current (production)",
        # Uses the actual build_plan_prompt from scaffolds.py
        "use_production": True,
    },
    "role_architect": {
        "name": "Role: Software Architect",
        "system": (
            "You are a senior software architect planning a web application. "
            "Your file plans are known for their clarity and precision — every file "
            "has a single, well-defined responsibility.\n\n"
            "Output a file list. For EACH file:\n"
            "FILE: <path> | ROLE: <entry|style|script|module|data> | LANG: <language> | DESCRIPTION: <specific purpose>\n\n"
            "Rules:\n"
            "- 3 files for simple apps (HTML + CSS + JS)\n"
            "- Descriptions must be specific: 'Manages task CRUD, localStorage sync' not 'main logic'\n"
            "- Don't over-split — a 3-file app is better than a 7-file one with pointless abstraction\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "minimal": {
        "name": "Minimal instructions",
        "system": (
            "Plan files for a web app.\n"
            "Format: FILE: path | ROLE: entry|style|script | LANG: lang | DESCRIPTION: purpose\n"
            "End with __PASS_COMPLETE__"
        ),
    },
}

# ---------------------------------------------------------------------------
# Generate Prompt Variants
# ---------------------------------------------------------------------------

GENERATE_VARIANTS = {
    "current": {
        "name": "Current (production)",
        "use_production": True,
    },
    "role_craftsman": {
        "name": "Role: Code Craftsman",
        "system": (
            "You are a master frontend developer known for writing beautiful, production-ready code. "
            "Your code is clean, well-structured, and works flawlessly on the first try.\n\n"
            "Generate the requested file as a fenced code block:\n"
            "```<filename>\n"
            "your code\n"
            "```\n\n"
            "Quality standards:\n"
            "- Complete, working code — no stubs, no TODOs, no placeholders\n"
            "- CSS: custom properties for colors, hover states, transitions, responsive\n"
            "- JS: IIFE with 'use strict', addEventListener (no inline onclick), escape user text\n"
            "- Share data between files via window.X — never duplicate const/let declarations\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "concise": {
        "name": "Concise",
        "system": (
            "Generate the file as ```filename\\ncode\\n```.\n"
            "Complete working code, no stubs. Use window.X for cross-file sharing.\n"
            "End with __PASS_COMPLETE__"
        ),
    },
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    test_id: str
    model: str
    variant: str
    # Format
    has_file_lines: bool = False
    has_pass_complete: bool = False
    num_files: int = 0
    has_entry: bool = False
    has_style: bool = False
    has_script: bool = False
    # Quality
    descriptions_specific: bool = False  # Not generic
    reasonable_file_count: bool = False
    # Meta
    latency_ms: int = 0
    raw_response: str = ""

    @property
    def score(self) -> float:
        parts = [
            self.has_file_lines,
            self.has_pass_complete,
            self.num_files >= 2,
            self.has_entry,
            self.has_style,
            self.has_script,
            self.descriptions_specific,
            self.reasonable_file_count,
        ]
        return sum(parts) / len(parts)


@dataclass
class GenerateResult:
    test_id: str
    model: str
    variant: str
    file_path: str = ""
    # Format
    has_code_fence: bool = False
    has_pass_complete: bool = False
    code_length: int = 0
    # Quality
    looks_like_code: bool = False
    no_stubs: bool = True
    has_expected_patterns: bool = False
    # Meta
    latency_ms: int = 0
    raw_response: str = ""

    @property
    def score(self) -> float:
        parts = [
            self.has_code_fence,
            self.has_pass_complete,
            self.code_length > 50,
            self.looks_like_code,
            self.no_stubs,
            self.has_expected_patterns,
        ]
        return sum(parts) / len(parts)


def score_plan(response: str, test: dict) -> PlanResult:
    r = PlanResult(test_id=test["id"], model="", variant="")
    r.raw_response = response

    # Parse FILE lines
    file_re = re.compile(r"FILE:\s*(.+?)\s*\|\s*ROLE:\s*(\w+)", re.IGNORECASE)
    files = file_re.findall(response)
    r.has_file_lines = len(files) > 0
    r.num_files = len(files)
    r.has_pass_complete = "__PASS_COMPLETE__" in response

    roles = [f[1].lower() for f in files]
    r.has_entry = "entry" in roles
    r.has_style = "style" in roles
    r.has_script = "script" in roles

    # Check descriptions aren't generic
    desc_re = re.compile(r"DESCRIPTION:\s*(.+)", re.IGNORECASE)
    descs = desc_re.findall(response)
    generic_words = ["main logic", "handles logic", "main file", "primary file", "core functionality"]
    r.descriptions_specific = len(descs) > 0 and not any(
        any(g in d.lower() for g in generic_words) for d in descs
    )

    # Reasonable file count
    expected = test.get("expected_files", 3)
    r.reasonable_file_count = expected - 1 <= r.num_files <= expected + 3

    return r


def score_generate(response: str, test: dict, file_path: str, role: str) -> GenerateResult:
    r = GenerateResult(test_id=test["id"], model="", variant="", file_path=file_path)
    r.raw_response = response

    # Code fence
    fence_re = re.compile(r"```\S*\n([\s\S]*?)```")
    fences = fence_re.findall(response)
    r.has_code_fence = len(fences) > 0
    r.has_pass_complete = "__PASS_COMPLETE__" in response

    code = fences[0] if fences else response
    r.code_length = len(code)

    # Looks like code?
    if role == "entry":
        r.looks_like_code = "<" in code and ">" in code
    elif role == "style":
        r.looks_like_code = "{" in code and ":" in code
    elif role == "script":
        r.looks_like_code = any(kw in code for kw in ["function", "const", "let", "=>", "class"])
    else:
        r.looks_like_code = len(code) > 50

    # No stubs
    stub_patterns = ["TODO", "implement", "your code here", "placeholder", "// ..."]
    r.no_stubs = not any(s.lower() in code.lower() for s in stub_patterns)

    # Expected patterns
    r.has_expected_patterns = any(
        re.search(p, code, re.IGNORECASE) for p in test.get("code_patterns", [])
    )

    return r


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def call_model(base_url, model, system, user, max_tokens=4096):
    start = time.time()
    try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"] or ""
        return content, int((time.time() - start) * 1000)
    except Exception as e:
        return f"ERROR: {e}", int((time.time() - start) * 1000)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def discover_models(base_url):
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["id"] for m in data.get("data", []) if "embed" not in m["id"].lower()]
        return models
    except Exception as e:
        print(f"Could not connect to {base_url}: {e}")
        return []


def run_plan_tests(base_url, models, variants, tests):
    results = []
    total = len(models) * len(variants) * len(tests)
    done = 0

    for model in models:
        for vkey, variant in variants.items():
            for test in tests:
                done += 1
                print(f"[{done}/{total}] PLAN | {model[:30]} | {variant['name']} | {test['id']}...", end=" ", flush=True)

                if variant.get("use_production"):
                    system = (
                        "You are a senior frontend architect planning a production-quality web application.\n"
                        "Scaffold: 'Static Web App' — Default files: index.html, styles.css, app.js\n"
                        "No CDN libraries.\n\n"
                        "Plan a well-architected file structure. Each file should have a clear, single responsibility.\n"
                        "Output a file list. For EACH file:\n"
                        "FILE: <path> | ROLE: <entry|style|script|module|data> | LANG: <language> | DESCRIPTION: <specific purpose>\n\n"
                        "Descriptions should be specific: 'Manages task CRUD, localStorage persistence' not 'main logic'.\n"
                        "Choose the right number of files for the complexity. Don't over-split.\n"
                        "End with __PASS_COMPLETE__"
                    )
                else:
                    system = variant["system"]

                user = f"Build: {test['description']}"
                response, latency = call_model(base_url, model, system, user, max_tokens=2048)

                r = score_plan(response, test)
                r.model = model
                r.variant = vkey
                r.latency_ms = latency

                status = "PASS" if r.score >= 0.8 else "PARTIAL" if r.score >= 0.5 else "FAIL"
                print(f"{status} (score={r.score:.2f}, files={r.num_files}, {latency}ms)")

                if r.score < 0.8:
                    issues = []
                    if not r.has_file_lines: issues.append("no FILE lines")
                    if not r.has_pass_complete: issues.append("no __PASS_COMPLETE__")
                    if not r.has_entry: issues.append("no entry file")
                    if not r.has_style: issues.append("no style file")
                    if not r.has_script: issues.append("no script file")
                    if not r.descriptions_specific: issues.append("generic descriptions")
                    if not r.reasonable_file_count: issues.append(f"file count {r.num_files} (expected ~{test['expected_files']})")
                    print(f"  Issues: {', '.join(issues)}")
                    if not r.has_file_lines:
                        preview = response[:200].encode('ascii', 'replace').decode('ascii')
                        print(f"  Preview: {preview}...")

                results.append(r)

    return results


def run_generate_tests(base_url, models, variants, tests):
    results = []
    total = len(models) * len(variants) * len(tests)
    done = 0

    for model in models:
        for vkey, variant in variants.items():
            for test in tests:
                done += 1
                print(f"[{done}/{total}] GEN | {model[:30]} | {variant['name']} | {test['id']}...", end=" ", flush=True)

                if variant.get("use_production"):
                    system = (
                        "You are a senior frontend developer generating production-quality code.\n"
                        "Your output should be indistinguishable from a hand-crafted professional application.\n\n"
                        "Output the file as a fenced code block with the filename:\n"
                        "  ```<filename>\n"
                        "  file content here\n"
                        "  ```\n\n"
                        "End with __PASS_COMPLETE__\n\n"
                        "QUALITY STANDARD:\n"
                        "- Complete, polished code — no stubs, no placeholders, no TODO\n"
                        "- CSS: custom properties for colors, hover states, transitions, responsive\n"
                        "- JS: IIFE with 'use strict', addEventListener, escape user text\n"
                        "- Wrap JS in (function(){'use strict'; ...})();\n"
                        "- Share between files with window.X — no duplicate const/let\n"
                        "- Responsive: works on mobile (320px+)"
                    )
                else:
                    system = variant["system"]

                # Simulate generating the JS file (the hardest one)
                user = (
                    f"Project: {test['description']}\n\n"
                    f"Generate this file: ```app.js```\n"
                    f"- app.js (script): Main application logic, event handlers, state management\n\n"
                    f"The HTML file has these elements:\n"
                    f"- #app (main container)\n"
                    f"- Buttons and inputs relevant to the app\n"
                    f"- IDs follow the pattern: feature-action (e.g. #add-btn, #task-input)\n"
                )

                response, latency = call_model(base_url, model, system, user, max_tokens=8192)

                r = score_generate(response, test, "app.js", "script")
                r.model = model
                r.variant = vkey
                r.latency_ms = latency

                status = "PASS" if r.score >= 0.8 else "PARTIAL" if r.score >= 0.5 else "FAIL"
                print(f"{status} (score={r.score:.2f}, len={r.code_length}, {latency}ms)")

                if r.score < 0.8:
                    issues = []
                    if not r.has_code_fence: issues.append("no code fence")
                    if not r.has_pass_complete: issues.append("no __PASS_COMPLETE__")
                    if not r.looks_like_code: issues.append("doesn't look like code")
                    if not r.no_stubs: issues.append("contains stubs/TODOs")
                    if not r.has_expected_patterns: issues.append("missing expected patterns")
                    print(f"  Issues: {', '.join(issues)}")

                results.append(r)

    return results


def print_summary(plan_results, gen_results):
    print("\n" + "=" * 80)
    print("BUILD PROMPT EVALUATION SUMMARY")
    print("=" * 80)

    for label, results in [("PLAN", plan_results), ("GENERATE", gen_results)]:
        if not results:
            continue
        print(f"\n--- {label} PHASE ---")
        models = sorted(set(r.model for r in results))
        variants = sorted(set(r.variant for r in results))

        for model in models:
            print(f"\n  {model[:50]}")
            for v in variants:
                vr = [r for r in results if r.model == model and r.variant == v]
                if not vr:
                    continue
                vname = PLAN_VARIANTS.get(v, GENERATE_VARIANTS.get(v, {})).get("name", v)
                avg = sum(r.score for r in vr) / len(vr)
                passes = sum(1 for r in vr if r.score >= 0.8)
                print(f"    {vname:40s} score={avg:.2f}  pass={passes}/{len(vr)}")

    # Save
    out_path = Path("tests/eval_results/build_prompt_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "plan": [{"test_id": r.test_id, "model": r.model, "variant": r.variant, "score": r.score, "files": r.num_files, "latency": r.latency_ms} for r in plan_results],
        "generate": [{"test_id": r.test_id, "model": r.model, "variant": r.variant, "score": r.score, "code_length": r.code_length, "latency": r.latency_ms} for r in gen_results],
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test build prompt variants against real models")
    parser.add_argument("--base-url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--model", help="Specific model to test")
    parser.add_argument("--prompt-variants", action="store_true", help="Test all prompt variants")
    parser.add_argument("--test", choices=["plan_only", "generate_only", "both"], default="both")
    args = parser.parse_args()

    models = [args.model] if args.model else discover_models(args.base_url)
    if not models:
        print("No models found.")
        sys.exit(1)
    print(f"Models: {', '.join(m[:30] for m in models[:5])}{'...' if len(models) > 5 else ''}")

    plan_variants = PLAN_VARIANTS if args.prompt_variants else {"current": PLAN_VARIANTS["current"]}
    gen_variants = GENERATE_VARIANTS if args.prompt_variants else {"current": GENERATE_VARIANTS["current"]}

    plan_results = []
    gen_results = []

    if args.test in ("plan_only", "both"):
        print(f"\n{'='*60}\nPLAN PHASE ({len(plan_variants)} variants × {len(BUILD_DESCRIPTIONS)} tests)\n{'='*60}")
        plan_results = run_plan_tests(args.base_url, models, plan_variants, BUILD_DESCRIPTIONS)

    if args.test in ("generate_only", "both"):
        print(f"\n{'='*60}\nGENERATE PHASE ({len(gen_variants)} variants × {len(BUILD_DESCRIPTIONS)} tests)\n{'='*60}")
        gen_results = run_generate_tests(args.base_url, models, gen_variants, BUILD_DESCRIPTIONS)

    print_summary(plan_results, gen_results)


if __name__ == "__main__":
    main()
