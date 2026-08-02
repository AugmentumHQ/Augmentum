"""Systematic prompt testing for the workspace quick-edit pipeline.

Sends real edit requests to LM Studio models and scores the responses on:
1. Format compliance (valid SEARCH/REPLACE blocks)
2. Patch applicability (do the patches actually apply?)
3. Correctness (did the edit do what was asked?)
4. Minimality (did it change only what was needed?)

Usage:
    python tests/test_edit_prompts.py                    # Run all tests with default model
    python tests/test_edit_prompts.py --model qwen3.5:7b # Specific model
    python tests/test_edit_prompts.py --all-models       # Test all available models
    python tests/test_edit_prompts.py --prompt-variants   # Test prompt variations
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

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Test Projects (realistic multi-file apps)
# ---------------------------------------------------------------------------

SIMPLE_PROJECT = [
    {
        "path": "index.html",
        "role": "entry",
        "content": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Counter App</title>
</head>
<body>
  <main id="app">
    <h1>Counter</h1>
    <div id="count">0</div>
    <button id="inc-btn">+</button>
    <button id="dec-btn">-</button>
    <button id="reset-btn">Reset</button>
  </main>
</body>
</html>""",
    },
    {
        "path": "styles.css",
        "role": "style",
        "content": """body {
  margin: 0;
  font-family: system-ui, sans-serif;
  background: #ffffff;
  color: #1a1a1a;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

#app {
  text-align: center;
}

#count {
  font-size: 4rem;
  font-weight: 700;
  margin: 1rem 0;
}

button {
  padding: 0.5rem 1.5rem;
  font-size: 1.2rem;
  border: 2px solid #1a1a1a;
  background: transparent;
  cursor: pointer;
  margin: 0 0.25rem;
  border-radius: 6px;
  transition: all 0.15s ease;
}

button:hover {
  background: #1a1a1a;
  color: #ffffff;
}""",
    },
    {
        "path": "app.js",
        "role": "script",
        "content": """(function() {
  'use strict';
  let count = 0;

  function render() {
    document.getElementById('count').textContent = count;
  }

  document.getElementById('inc-btn').addEventListener('click', () => {
    count++;
    render();
  });

  document.getElementById('dec-btn').addEventListener('click', () => {
    count--;
    render();
  });

  document.getElementById('reset-btn').addEventListener('click', () => {
    count = 0;
    render();
  });

  render();
})();""",
    },
]

# ---------------------------------------------------------------------------
# Test Cases — (description, expected_file, expected_pattern)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # CSS-only changes
    {
        "id": "css_bg_color",
        "description": "change the background to dark blue",
        "expected_file": "styles.css",
        "expected_patterns": [r"background.*#[0-9a-fA-F]|background.*dark|background.*blue|background.*rgb"],
        "expected_unchanged": ["app.js"],  # Should NOT modify JS
    },
    {
        "id": "css_add_shadow",
        "description": "add a box shadow to the buttons",
        "expected_file": "styles.css",
        "expected_patterns": [r"box-shadow"],
        "expected_unchanged": ["app.js", "index.html"],
    },
    {
        "id": "css_dark_theme",
        "description": "make this a dark theme with light text",
        "expected_file": "styles.css",
        "expected_patterns": [r"background.*#[0-2]|background.*dark|color.*#[dDeEfF]|color.*white|color.*light"],
        "expected_unchanged": ["app.js"],
    },
    # JS-only changes
    {
        "id": "js_step_size",
        "description": "make the increment button add 5 instead of 1",
        "expected_file": "app.js",
        "expected_patterns": [r"5|five|\+\s*5|count\s*\+\s*=\s*5"],
        "expected_unchanged": ["styles.css"],
    },
    {
        "id": "js_add_double",
        "description": "add a double button that multiplies the count by 2",
        "expected_file": "app.js",
        "expected_patterns": [r"\*\s*2|count\s*\*=?\s*2|double"],
        "expected_unchanged": ["styles.css"],
    },
    # HTML changes
    {
        "id": "html_add_title",
        "description": "change the title to 'My Awesome Counter'",
        "expected_file": "index.html",
        "expected_patterns": [r"My Awesome Counter"],
        "expected_unchanged": ["app.js", "styles.css"],
    },
    # Multi-file changes
    {
        "id": "multi_dark_mode_toggle",
        "description": "add a dark mode toggle button",
        "expected_file": None,
        "expected_patterns": [r"dark|theme|toggle"],
        "expected_unchanged": [],
    },
    # --- Multi-file: coordinated changes across HTML + CSS + JS ---
    {
        "id": "multi_add_history",
        "description": "add a history display that shows the last 5 counter values below the buttons",
        "expected_file": None,
        "expected_patterns": [r"history|log|record|previous"],
        "expected_unchanged": [],
    },
    {
        "id": "multi_color_on_negative",
        "description": "make the counter text turn red when the value is negative",
        "expected_file": None,
        "expected_patterns": [r"red|negative|color.*#|classList|style\.color"],
        "expected_unchanged": [],
    },
    {
        "id": "multi_keyboard_shortcuts",
        "description": "add keyboard shortcuts: press + to increment, - to decrement, 0 to reset",
        "expected_file": "app.js",
        "expected_patterns": [r"keydown|keypress|event\.key|addEventListener.*key"],
        "expected_unchanged": ["styles.css"],
    },
    {
        "id": "multi_animation",
        "description": "add a smooth scale animation to the count number when it changes",
        "expected_file": None,
        "expected_patterns": [r"animation|@keyframes|transform.*scale|transition"],
        "expected_unchanged": [],
    },
    {
        "id": "multi_max_min_limit",
        "description": "add a max limit of 100 and min limit of -100, disable the buttons when at the limit",
        "expected_file": "app.js",
        "expected_patterns": [r"100|limit|max|min|disabled"],
        "expected_unchanged": [],
    },
    {
        "id": "multi_localstorage",
        "description": "save the counter value to localStorage and restore it on page load",
        "expected_file": "app.js",
        "expected_patterns": [r"localStorage|setItem|getItem|storage"],
        "expected_unchanged": ["styles.css"],
    },
]

# ---------------------------------------------------------------------------
# Prompt Variants to Test
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "current": {
        "name": "Current (structured, production)",
        "system": (
            "You are editing code files. Follow these steps exactly:\n"
            "1. Identify which file(s) need changes\n"
            "2. For each change, output a SEARCH/REPLACE block\n"
            "3. The SEARCH section must contain EXACT lines from the current file\n"
            "4. The REPLACE section contains the new lines\n\n"
            "Output format (no other text):\n"
            "=== FILE: <filename> ===\n"
            "<<<<<<< SEARCH\n"
            "exact existing lines\n"
            "=======\n"
            "new replacement lines\n"
            ">>>>>>> REPLACE\n\n"
            "Important:\n"
            "- Copy the SEARCH lines EXACTLY from the file (whitespace matters)\n"
            "- Only change what's needed\n"
            "- All files share ONE global scope when assembled. Use window.X for cross-file access\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "concise": {
        "name": "Concise (minimal instructions)",
        "system": (
            "Output SEARCH/REPLACE patches. No explanation.\n\n"
            "=== FILE: name ===\n"
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n\n"
            "End with __PASS_COMPLETE__"
        ),
    },
    "structured": {
        "name": "Structured (explicit steps)",
        "system": (
            "You are editing code files. Follow these steps exactly:\n"
            "1. Identify which file(s) need changes\n"
            "2. For each change, output a SEARCH/REPLACE block\n"
            "3. The SEARCH section must contain EXACT lines from the current file\n"
            "4. The REPLACE section contains the new lines\n\n"
            "Output format (no other text):\n"
            "=== FILE: <filename> ===\n"
            "<<<<<<< SEARCH\n"
            "exact existing lines\n"
            "=======\n"
            "new replacement lines\n"
            ">>>>>>> REPLACE\n\n"
            "Important:\n"
            "- Copy the SEARCH lines EXACTLY from the file (whitespace matters)\n"
            "- Only change what's needed\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "example_driven": {
        "name": "Example-driven (show before asking)",
        "system": (
            "You edit code by outputting SEARCH/REPLACE blocks.\n\n"
            "Example — changing a color:\n"
            "=== FILE: styles.css ===\n"
            "<<<<<<< SEARCH\n"
            "  background: #ffffff;\n"
            "=======\n"
            "  background: #1a1a2e;\n"
            ">>>>>>> REPLACE\n\n"
            "Rules:\n"
            "- SEARCH must match the file exactly (copy-paste from the file content)\n"
            "- REPLACE contains the new version\n"
            "- Only output SEARCH/REPLACE blocks, no explanation\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "role_play": {
        "name": "Role-play (senior dev persona)",
        "system": (
            "You are a senior full-stack developer performing a surgical code edit. "
            "You understand that precision matters — the SEARCH block must match the "
            "existing code character-for-character or the patch will fail.\n\n"
            "Output ONLY SEARCH/REPLACE blocks in this exact format:\n"
            "=== FILE: <filename> ===\n"
            "<<<<<<< SEARCH\n"
            "exact lines from the file\n"
            "=======\n"
            "your replacement\n"
            ">>>>>>> REPLACE\n\n"
            "- Do not add commentary or explanation\n"
            "- Keep changes minimal — touch only what's needed\n"
            "- Preserve all existing functionality\n"
            "- End with __PASS_COMPLETE__"
        ),
    },
    "negative_examples": {
        "name": "Negative examples (show what NOT to do)",
        "system": (
            "You are a code editor. Output SEARCH/REPLACE patches.\n\n"
            "Format:\n"
            "=== FILE: <filename> ===\n"
            "<<<<<<< SEARCH\n"
            "exact lines to find\n"
            "=======\n"
            "replacement lines\n"
            ">>>>>>> REPLACE\n\n"
            "WRONG (do NOT do these):\n"
            "- Do NOT add explanation text before or after patches\n"
            "- Do NOT use approximate/modified SEARCH lines — they must be EXACT copies\n"
            "- Do NOT change files that don't need changes\n"
            "- Do NOT output the entire file — only the changed sections\n"
            "- Do NOT forget __PASS_COMPLETE__ at the end\n\n"
            "End with __PASS_COMPLETE__"
        ),
    },
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_id: str
    model: str
    prompt_variant: str
    # Format
    has_file_header: bool = False
    has_search_replace: bool = False
    has_pass_complete: bool = False
    num_patches: int = 0
    # Applicability
    patches_applied: int = 0
    # Correctness
    correct_file_targeted: bool = False
    expected_pattern_found: bool = False
    unchanged_files_respected: bool = True
    # Minimality
    files_touched: int = 0
    lines_changed: int = 0
    # Meta
    response_tokens: int = 0
    latency_ms: int = 0
    raw_response: str = ""
    error: str = ""

    @property
    def format_score(self) -> float:
        """0-1 score for format compliance."""
        parts = [self.has_file_header, self.has_search_replace, self.has_pass_complete, self.num_patches > 0]
        return sum(parts) / len(parts)

    @property
    def quality_score(self) -> float:
        """0-1 score for overall quality."""
        parts = [
            self.format_score,
            1.0 if self.patches_applied > 0 else 0.0,
            1.0 if self.correct_file_targeted else 0.0,
            1.0 if self.expected_pattern_found else 0.0,
            1.0 if self.unchanged_files_respected else 0.0,
        ]
        return sum(parts) / len(parts)


def score_response(response: str, test_case: dict, files: list[dict]) -> TestResult:
    """Score an LLM response against a test case."""
    result = TestResult(test_id=test_case["id"], model="", prompt_variant="")
    result.raw_response = response
    result.response_tokens = len(response.split())  # Approximate

    # Format checks
    result.has_file_header = bool(re.search(r"===\s*FILE:", response, re.IGNORECASE))
    result.has_search_replace = bool(re.search(
        r"<<<<<<<?\.?\s*SEARCH\n[\s\S]*?\n?={3,}\n[\s\S]*?\n?>>>>>>>?\.?\s*REPLACE",
        response, re.IGNORECASE
    ))
    result.has_pass_complete = "__PASS_COMPLETE__" in response

    # Count patches
    sr_re = re.compile(
        r"<<<<<<<?\.?\s*SEARCH\n([\s\S]*?)\n?={3,}\n([\s\S]*?)\n?>>>>>>>?\.?\s*REPLACE",
        re.IGNORECASE,
    )
    patches = list(sr_re.finditer(response))
    result.num_patches = len(patches)

    # Try applying patches
    test_files = [{"path": f["path"], "role": f["role"], "content": f["content"]} for f in files]
    applied = 0
    files_touched = set()

    # Parse FILE sections
    section_re = re.compile(r"===\s*FILE:\s*(.+?)\s*===\s*\n([\s\S]*?)(?=\n===\s*FILE:|$)", re.IGNORECASE)
    for m in section_re.finditer(response):
        filename = m.group(1).strip()
        section = m.group(2).strip()
        target = next((f for f in test_files if f["path"] == filename), None)
        if not target:
            continue

        for patch in sr_re.finditer(section):
            search = patch.group(1).rstrip()
            replace = patch.group(2).rstrip()
            if search in target["content"]:
                target["content"] = target["content"].replace(search, replace, 1)
                applied += 1
                files_touched.add(filename)
            else:
                # Try trimmed matching
                s_lines = [l.strip() for l in search.split("\n")]
                c_lines = target["content"].split("\n")
                for i in range(len(c_lines) - len(s_lines) + 1):
                    if all(s_lines[j] == c_lines[i + j].strip() for j in range(len(s_lines))):
                        applied += 1
                        files_touched.add(filename)
                        break

    # Also try without FILE wrappers
    if applied == 0:
        for patch in sr_re.finditer(response):
            search = patch.group(1).rstrip()
            replace = patch.group(2).rstrip()
            for f in test_files:
                if search in f["content"]:
                    f["content"] = f["content"].replace(search, replace, 1)
                    applied += 1
                    files_touched.add(f["path"])
                    break

    result.patches_applied = applied
    result.files_touched = len(files_touched)

    # Correctness: right file targeted?
    if test_case["expected_file"]:
        result.correct_file_targeted = test_case["expected_file"] in files_touched
    else:
        result.correct_file_targeted = len(files_touched) > 0

    # Expected pattern in patched content?
    all_content = " ".join(f["content"] for f in test_files)
    result.expected_pattern_found = any(
        re.search(p, all_content, re.IGNORECASE) for p in test_case["expected_patterns"]
    )

    # Unchanged files respected?
    for unchanged_file in test_case.get("expected_unchanged", []):
        orig = next((f for f in files if f["path"] == unchanged_file), None)
        curr = next((f for f in test_files if f["path"] == unchanged_file), None)
        if orig and curr and orig["content"] != curr["content"]:
            result.unchanged_files_respected = False
            break

    return result


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def call_model(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str, int]:
    """Call an LLM and return (response_text, latency_ms)."""
    start = time.time()
    try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
        latency = int((time.time() - start) * 1000)
        return content, latency
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return f"ERROR: {e}", latency


def build_user_prompt(files: list[dict], description: str, prompt_variant: dict) -> str:
    """Build the user prompt using the same heuristic file filtering as production."""
    desc_lower = description.lower()
    full_files = []
    sig_files = []

    for f in files:
        path = f["path"].lower()
        is_relevant = False
        if any(kw in desc_lower for kw in ["style", "color", "background", "font", "css", "theme", "dark", "layout", "margin", "padding", "border"]):
            is_relevant = path.endswith(".css")
        if any(kw in desc_lower for kw in ["function", "click", "event", "logic", "score", "speed", "timer", "game", "state", "button", "input"]):
            is_relevant = is_relevant or path.endswith(".js")
        if any(kw in desc_lower for kw in ["html", "element", "div", "text", "title", "heading", "structure", "add", "remove"]):
            is_relevant = is_relevant or path.endswith(".html") or path.endswith(".htm")
        if not any(kw in desc_lower for kw in ["style", "color", "background", "font", "css", "function", "click", "html", "element"]):
            is_relevant = True
        if is_relevant:
            full_files.append(f)
        else:
            sig_files.append(f)

    if not full_files:
        full_files = files
        sig_files = []

    file_context = ""
    for f in full_files:
        file_context += f"\n=== {f['path']} (FULL) ===\n{f['content']}\n"
    for f in sig_files:
        lines = f["content"].count("\n") + 1
        exports = []
        for line in f["content"].split("\n")[:30]:
            stripped = line.strip()
            if stripped.startswith("function ") or stripped.startswith("const ") or stripped.startswith("class "):
                exports.append(stripped.split("(")[0].split("=")[0].strip())
        sig = ", ".join(exports[:15]) if exports else f"{lines} lines"
        file_context += f"\n=== {f['path']} ({sig}) ===\n"

    return f"Files:\n{file_context}\n\n---\nRequested change: {description}\n\nApply now."


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_tests(
    base_url: str,
    models: list[str],
    prompt_variants: list[str],
    test_cases: list[dict] | None = None,
) -> list[TestResult]:
    """Run all test combinations and return results."""
    if test_cases is None:
        test_cases = TEST_CASES

    results = []

    # Discover models if --all-models
    if not models:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
            if not models:
                print("No models found. Load a model in LM Studio first.")
                return []
            print(f"Found {len(models)} model(s): {', '.join(models)}")
        except Exception as e:
            print(f"Could not connect to {base_url}: {e}")
            return []

    total = len(models) * len(prompt_variants) * len(test_cases)
    done = 0

    for model in models:
        for variant_key in prompt_variants:
            variant = PROMPT_VARIANTS[variant_key]
            for tc in test_cases:
                done += 1
                label = f"[{done}/{total}] {model} | {variant['name']} | {tc['id']}"
                print(f"{label}...", end=" ", flush=True)

                user_prompt = build_user_prompt(SIMPLE_PROJECT, tc["description"], variant)
                response, latency = call_model(
                    base_url, model, variant["system"], user_prompt
                )

                result = score_response(response, tc, SIMPLE_PROJECT)
                result.model = model
                result.prompt_variant = variant_key
                result.latency_ms = latency

                status = "PASS" if result.quality_score >= 0.8 else "PARTIAL" if result.quality_score >= 0.4 else "FAIL"
                print(f"{status} (quality={result.quality_score:.2f}, patches={result.patches_applied}, {latency}ms)")

                if result.quality_score < 0.8:
                    # Show why it failed
                    issues = []
                    if not result.has_file_header:
                        issues.append("no FILE header")
                    if not result.has_search_replace:
                        issues.append("no SEARCH/REPLACE")
                    if not result.has_pass_complete:
                        issues.append("no __PASS_COMPLETE__")
                    if result.patches_applied == 0:
                        issues.append("patches didn't apply")
                    if not result.correct_file_targeted:
                        issues.append(f"wrong file (expected {tc['expected_file']})")
                    if not result.expected_pattern_found:
                        issues.append("expected pattern not found")
                    if not result.unchanged_files_respected:
                        issues.append("modified files that shouldn't change")
                    print(f"  Issues: {', '.join(issues)}")
                    if not result.has_search_replace:
                        preview = response[:200].encode('ascii', 'replace').decode('ascii')
                        print(f"  Response preview: {preview}...")

                results.append(result)

    return results


def print_summary(results: list[TestResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Group by model
    models = sorted(set(r.model for r in results))
    variants = sorted(set(r.prompt_variant for r in results))

    for model in models:
        print(f"\n--- {model} ---")
        for variant in variants:
            vr = [r for r in results if r.model == model and r.prompt_variant == variant]
            if not vr:
                continue
            avg_quality = sum(r.quality_score for r in vr) / len(vr)
            avg_format = sum(r.format_score for r in vr) / len(vr)
            avg_latency = sum(r.latency_ms for r in vr) / len(vr)
            pass_count = sum(1 for r in vr if r.quality_score >= 0.8)
            total_patches = sum(r.patches_applied for r in vr)
            print(
                f"  {PROMPT_VARIANTS[variant]['name']:40s} "
                f"quality={avg_quality:.2f}  format={avg_format:.2f}  "
                f"pass={pass_count}/{len(vr)}  patches={total_patches}  "
                f"latency={avg_latency:.0f}ms"
            )

    # Best prompt per model
    print("\n--- BEST PROMPT PER MODEL ---")
    for model in models:
        best_variant = None
        best_score = -1
        for variant in variants:
            vr = [r for r in results if r.model == model and r.prompt_variant == variant]
            if not vr:
                continue
            avg = sum(r.quality_score for r in vr) / len(vr)
            if avg > best_score:
                best_score = avg
                best_variant = variant
        if best_variant:
            print(f"  {model:40s} -> {PROMPT_VARIANTS[best_variant]['name']} (score={best_score:.2f})")

    # Save detailed results
    out_path = Path("tests/eval_results/edit_prompt_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = []
    for r in results:
        out_data.append({
            "test_id": r.test_id,
            "model": r.model,
            "prompt_variant": r.prompt_variant,
            "quality_score": r.quality_score,
            "format_score": r.format_score,
            "patches_applied": r.patches_applied,
            "num_patches": r.num_patches,
            "correct_file": r.correct_file_targeted,
            "pattern_found": r.expected_pattern_found,
            "unchanged_respected": r.unchanged_files_respected,
            "latency_ms": r.latency_ms,
            "response_tokens": r.response_tokens,
        })
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nDetailed results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test edit prompt variants against real models")
    parser.add_argument("--base-url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--model", help="Specific model to test")
    parser.add_argument("--all-models", action="store_true", help="Test all available models")
    parser.add_argument("--prompt-variants", action="store_true", help="Test all prompt variants")
    parser.add_argument("--variant", help="Test a specific prompt variant (current, concise, structured, example_driven, role_play, negative_examples)")
    parser.add_argument("--test", help="Run a specific test case by ID")
    args = parser.parse_args()

    models = []
    if args.model:
        models = [args.model]
    elif not args.all_models:
        models = []  # Will auto-discover

    if args.prompt_variants:
        variants = list(PROMPT_VARIANTS.keys())
    elif args.variant:
        variants = [args.variant]
    else:
        variants = ["current"]

    test_cases = TEST_CASES
    if args.test:
        test_cases = [tc for tc in TEST_CASES if tc["id"] == args.test]
        if not test_cases:
            print(f"Test case '{args.test}' not found. Available: {', '.join(tc['id'] for tc in TEST_CASES)}")
            sys.exit(1)

    print(f"Testing {len(variants)} prompt variant(s) × {len(test_cases)} test case(s)")
    print(f"LM Studio: {args.base_url}")
    print()

    results = run_tests(args.base_url, models, variants, test_cases)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
