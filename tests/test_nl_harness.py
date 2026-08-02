"""Live A/B test: Natural Language intent extraction vs ReWOO JSON pipeline.

Sends identical coding tasks to the running model using two different
prompts, then compares what each approach produces.

Usage:
    python tests/test_nl_harness.py [--base-url URL] [--model MODEL]

Defaults to Ollama at localhost:11434. Adjust for LM Studio (localhost:1234).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── Intent Extractor (the novel NL approach) ────────────────────────────

@dataclass
class ExtractedAction:
    tool: str
    input: dict
    confidence: float  # 0.0 - 1.0
    source_text: str   # the phrase that triggered extraction

def extract_intents(text: str, workspace: str = "/workspace") -> list[ExtractedAction]:
    """Extract tool intents from natural language model output.

    Scans prose for action phrases and code blocks, translating them
    into structured tool calls without the model needing to know
    about the tool API.
    """
    actions: list[ExtractedAction] = []

    # ── Step 1: Extract ALL fenced code blocks with their context ─
    # This is the most reliable signal — models consistently use code
    # fences for actionable content.
    block_re = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    blocks: list[tuple[int, int, str, str]] = []  # (start, end, lang, content)
    for m in block_re.finditer(text):
        blocks.append((m.start(), m.end(), m.group(1).lower(), m.group(2)))

    # Build a set of ranges covered by code blocks (to avoid phrase
    # matches inside them)
    block_ranges = [(s, e) for s, e, _, _ in blocks]

    for start, end, lang, content in blocks:
        content_stripped = content.strip()
        if not content_stripped:
            continue

        # Look backwards from the code block for a filename mention
        # within the preceding 200 chars of prose
        preceding = text[max(0, start - 200):start]
        filename = None

        # Pattern: `filename.ext` or 'filename.ext' or "filename.ext" nearby
        fn_match = re.search(
            r'[`"\']([^\s`"\']+\.\w{1,5})[`"\']',
            preceding[max(0, len(preceding) - 150):],  # last 150 chars before block
        )
        if fn_match:
            filename = fn_match.group(1)

        # ── Bash/shell blocks → shell_exec ────────────────────
        if lang in ('bash', 'sh', 'shell', 'console', 'zsh', ''):
            # Multi-line bash: each non-comment line is a command
            # Single-line: one command
            cmd_lines = [
                ln.strip()
                for ln in content_stripped.splitlines()
                if ln.strip() and not ln.strip().startswith('#')
            ]
            # Only treat unlabeled blocks as bash if they look like commands
            if lang == '' and not _looks_like_shell(content_stripped):
                continue
            for cmd in cmd_lines:
                actions.append(ExtractedAction(
                    tool="shell_exec",
                    input={"command": cmd},
                    confidence=0.95 if lang else 0.7,
                    source_text=f"```{lang}\\n{cmd}",
                ))

        # ── Code blocks with filename → file_write ────────────
        elif lang in ('python', 'py', 'javascript', 'js', 'typescript', 'ts',
                       'html', 'css', 'json', 'yaml', 'yml', 'rust', 'go',
                       'java', 'c', 'cpp', 'rb', 'ruby', 'jsx', 'tsx',
                       'sql', 'toml', 'xml', 'swift', 'kotlin', 'php'):
            if filename:
                path = filename if filename.startswith("/") else f"{workspace}/{filename}"
                actions.append(ExtractedAction(
                    tool="file_write",
                    input={"path": path, "content": content},
                    confidence=0.9,
                    source_text=f"{filename}: ```{lang}",
                ))
            else:
                # Check if preceding text mentions creating a file
                create_match = re.search(
                    r'(?:creat|writ|sav|mak|add|put)\w*\s+(?:a\s+)?(?:file\s+)?(?:called\s+|named\s+)?'
                    r'[`"\']?([^\s`"\',:]+\.\w{1,5})',
                    preceding, re.IGNORECASE,
                )
                if create_match:
                    path = create_match.group(1)
                    if not path.startswith("/"):
                        path = f"{workspace}/{path}"
                    actions.append(ExtractedAction(
                        tool="file_write",
                        input={"path": path, "content": content},
                        confidence=0.85,
                        source_text=f"create {path}: ```{lang}",
                    ))

        # ── tool_code blocks (gemma-style) → shell_exec ──────
        elif lang == 'tool_code':
            actions.append(ExtractedAction(
                tool="shell_exec",
                input={"command": f"python3 -c {json.dumps(content_stripped)}"},
                confidence=0.8,
                source_text="```tool_code",
            ))

    # ── Step 2: Phrase-based intent matching (outside code blocks) ─
    def _in_code_block(pos: int) -> bool:
        return any(s <= pos <= e for s, e in block_ranges)

    phrase_patterns = [
        # Read file: "let me read `main.py`", "look at main.py", "examine the file"
        (
            r'\b(?:read|look\s+at|open|view|examin\w*|inspect|check(?:ing)?)\s+(?:the\s+)?(?:file\s+)?(?:contents?\s+of\s+)?[`"\']?([^\s`"\',:()]+\.\w{1,5})[`"\']?',
            "file_read",
            lambda m: {"path": m.group(1) if m.group(1).startswith("/") else f"{workspace}/{m.group(1)}"},
            0.85,
        ),
        # Search: "search for 'pattern'", "grep for 'term'"
        (
            r'\b(?:search|grep|find|look)\s+(?:for\s+)?[`"\']([^`"\']{2,40})[`"\']',
            "code_grep",
            lambda m: {"pattern": m.group(1), "path": workspace},
            0.8,
        ),
        # Run tests
        (
            r'\b(?:run|execute)\s+(?:the\s+)?tests?\b',
            "test_run",
            lambda m: {},
            0.8,
        ),
        # Check structure / directory
        (
            r'\b(?:check|look\s+at|view|show|see|list)\s+(?:the\s+)?(?:project\s+)?(?:structure|director\w*|files|tree|folder)\b',
            "dir_tree",
            lambda m: {"path": workspace, "depth": 3},
            0.85,
        ),
        # Git operations
        (
            r'\b(?:check|show|get)\s+(?:the\s+)?git\s+(status|log|diff)\b',
            "git",
            lambda m: {"action": m.group(1)},
            0.8,
        ),
        # Check environment
        (
            r'\b(?:check|see|find\s+out)\s+(?:if\s+)?(?:what\s+)?(?:python|node|npm|pip|environment|runtime|version)',
            "env_info",
            lambda m: {},
            0.75,
        ),
    ]

    for pattern, tool, build_input, confidence in phrase_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if _in_code_block(m.start()):
                continue
            inp = build_input(m)
            actions.append(ExtractedAction(
                tool=tool,
                input=inp,
                confidence=confidence,
                source_text=m.group(0)[:80],
            ))

    # ── Deduplicate ──────────────────────────────────────────────
    seen: set[str] = set()
    deduped: list[ExtractedAction] = []
    for a in actions:
        key = (a.tool, json.dumps(a.input, sort_keys=True))
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    return deduped


def _looks_like_shell(text: str) -> bool:
    """Heuristic: does this unlabeled code block look like a shell command?"""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    # Common shell patterns
    shell_starts = (
        'ls', 'cd', 'cat', 'grep', 'find', 'pip', 'npm', 'yarn',
        'python', 'python3', 'node', 'git', 'curl', 'wget', 'apt',
        'brew', 'cargo', 'go ', 'make', 'cmake', 'docker', 'kubectl',
        'mkdir', 'rm ', 'cp ', 'mv ', 'chmod', 'touch', 'echo ',
        'export', 'source', 'which', 'whoami', 'pwd', 'wc ',
        'head', 'tail', 'sort', 'uniq', 'awk', 'sed', 'test ',
    )
    return any(first.startswith(s) for s in shell_starts)


# ── ReWOO JSON parser (existing approach) ────────────────────────────

def parse_rewoo_json(text: str) -> list[dict]:
    """Parse tool calls from ReWOO JSON array output."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```[\w]*\n?", "", text).replace("```", "").strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        # Try individual objects
        calls = []
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text):
            try:
                obj = json.loads(m.group())
                if obj.get("tool") and obj.get("input") is not None:
                    calls.append(obj)
            except json.JSONDecodeError:
                continue
        return calls

    try:
        arr = json.loads(text[start:end + 1])
        if isinstance(arr, list):
            return [item for item in arr if isinstance(item, dict) and "tool" in item]
    except json.JSONDecodeError:
        pass
    return []


# ── Prompts ──────────────────────────────────────────────────────────

NL_SYSTEM = """\
You are a coding assistant working in a project at /workspace.
Solve the user's request step by step. Be direct and take action.

When you want to do something, say it clearly:
- To run a command, put it in a ```bash block
- To create or edit a file, name it with backticks then show code in a fenced block
- To read a file, say "let me read `filename`"
- To search, say "let me search for 'pattern'"

Example:
---
First let me check what we have:
```bash
ls -la /workspace
```

I'll create `hello.py`:
```python
print("Hello, World!")
```

Let me run it:
```bash
python3 /workspace/hello.py
```
---

Be concise. Every code block is an action I will execute for you.\
"""

REWOO_SYSTEM = """\
You are a coding agent. Read the user's task and output a JSON array of tool calls.

Output ONLY a valid JSON array. No explanation. No markdown. No text before or after.

Tools (use exact names and input fields):
- dir_tree: {"path": "/workspace", "depth": 3}
- file_read: {"path": "/workspace/file.py"}
- file_write: {"path": "/workspace/file.py", "content": "full content"}
- code_edit: {"path": "/workspace/file.py", "search": "old", "replace": "new"}
- code_grep: {"pattern": "term", "path": "/workspace"}
- shell_exec: {"command": "python3 script.py"}
- test_run: {"command": "pytest -x"} or {}
- env_info: {}
- git: {"action": "status"}

Rules:
- ALWAYS include a final verification step (test_run or shell_exec)
- file_read before code_edit (always required)
- file_write for new files, code_edit for existing files

Example:
[{"tool": "dir_tree", "input": {"path": "/workspace"}}, {"tool": "file_write", "input": {"path": "/workspace/hello.py", "content": "print('Hello')\\n"}}, {"tool": "shell_exec", "input": {"command": "python3 /workspace/hello.py"}}]

Output the JSON array:\
"""


# ── Test cases ───────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "Explore codebase",
        "prompt": "What files are in this project? Give me an overview.",
    },
    {
        "name": "Create a script",
        "prompt": "Create a Python script called calculator.py that can add, subtract, multiply and divide two numbers. Include a simple test.",
    },
    {
        "name": "Debug/investigate",
        "prompt": "There's a bug in main.py where the login function crashes. Can you look into it?",
    },
    {
        "name": "Install and test",
        "prompt": "Install flask and create a minimal web app, then verify it starts correctly.",
    },
    {
        "name": "Simple question",
        "prompt": "Is Python installed? What version?",
    },
]


# ── LLM caller ───────────────────────────────────────────────────────

def _http_post(url: str, payload: dict, timeout: int = 120) -> tuple[dict | None, float]:
    """POST JSON, return (parsed_response, elapsed) or (None, elapsed) on error."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return body, time.time() - start
    except Exception:
        return None, time.time() - start


def call_llm(base_url: str, model: str, system: str, user: str, timeout: int = 120) -> tuple[str, float]:
    """Call the LLM and return (response_text, elapsed_seconds)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Try OpenAI-compatible endpoint first
    body, elapsed = _http_post(f"{base_url}/v1/chat/completions", {
        "model": model, "messages": messages, "stream": False, "temperature": 0.7,
    }, timeout)
    if body:
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, elapsed

    # Fall back to Ollama /api/chat
    body, elapsed = _http_post(f"{base_url}/api/chat", {
        "model": model, "messages": messages, "stream": False,
    }, timeout)
    if body:
        content = body.get("message", {}).get("content", "")
        return content, elapsed

    return "", elapsed


# ── Main ─────────────────────────────────────────────────────────────

def run_test(base_url: str, model: str, test: dict) -> dict:
    """Run one test case with both approaches."""
    name = test["name"]
    prompt = test["prompt"]
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"  PROMPT: {prompt}")
    print(f"{'='*70}")

    # ── Natural Language approach ──
    print("\n  [NL] Calling model...")
    try:
        nl_response, nl_time = call_llm(base_url, model, NL_SYSTEM, prompt)
    except Exception as e:
        print(f"  [NL] ERROR: {e}")
        nl_response, nl_time = "", 0.0

    print(f"  [NL] Got {len(nl_response)} chars in {nl_time:.1f}s")
    nl_actions = extract_intents(nl_response)

    # ── ReWOO JSON approach ──
    print("\n  [ReWOO] Calling model...")
    try:
        rewoo_response, rewoo_time = call_llm(base_url, model, REWOO_SYSTEM, prompt)
    except Exception as e:
        print(f"  [ReWOO] ERROR: {e}")
        rewoo_response, rewoo_time = "", 0.0

    print(f"  [ReWOO] Got {len(rewoo_response)} chars in {rewoo_time:.1f}s")
    rewoo_calls = parse_rewoo_json(rewoo_response)

    # ── Display results ──
    print("\n  ── Natural Language Response ──")
    # Show first 500 chars
    preview = nl_response[:500]
    if len(nl_response) > 500:
        preview += "..."
    for line in preview.split("\n"):
        print(f"  │ {line}")

    print(f"\n  ── NL Extracted Actions ({len(nl_actions)}) ──")
    for a in nl_actions:
        inp_preview = json.dumps(a.input)
        if len(inp_preview) > 80:
            inp_preview = inp_preview[:77] + "..."
        print(f"  │ {a.tool:12s} conf={a.confidence:.0%}  {inp_preview}")
        print(f"  │   source: \"{a.source_text}\"")

    print("\n  ── ReWOO Response ──")
    preview = rewoo_response[:500]
    if len(rewoo_response) > 500:
        preview += "..."
    for line in preview.split("\n"):
        print(f"  │ {line}")

    print(f"\n  ── ReWOO Parsed Calls ({len(rewoo_calls)}) ──")
    for tc in rewoo_calls:
        inp_preview = json.dumps(tc.get("input", {}))
        if len(inp_preview) > 80:
            inp_preview = inp_preview[:77] + "..."
        print(f"  │ {tc.get('tool', '?'):12s}  {inp_preview}")

    # ── Metrics ──
    nl_tools = {a.tool for a in nl_actions}
    rewoo_tools = {tc.get("tool", "") for tc in rewoo_calls}
    rewoo_valid = len(rewoo_calls) > 0

    print("\n  ── Comparison ──")
    print(f"  │ NL time:      {nl_time:.1f}s")
    print(f"  │ ReWOO time:   {rewoo_time:.1f}s")
    print(f"  │ NL actions:   {len(nl_actions)} ({', '.join(a.tool for a in nl_actions) or 'none'})")
    print(f"  │ ReWOO calls:  {len(rewoo_calls)} ({', '.join(tc.get('tool','') for tc in rewoo_calls) or 'none'})")
    print(f"  │ ReWOO valid:  {'✓' if rewoo_valid else '✖ FAILED TO PARSE'}")
    print(f"  │ Tool overlap: {nl_tools & rewoo_tools or 'none'}")
    print(f"  │ NL-only:      {nl_tools - rewoo_tools or 'none'}")
    print(f"  │ ReWOO-only:   {rewoo_tools - nl_tools or 'none'}")

    return {
        "name": name,
        "nl_time": nl_time,
        "rewoo_time": rewoo_time,
        "nl_actions": len(nl_actions),
        "rewoo_calls": len(rewoo_calls),
        "rewoo_valid": rewoo_valid,
        "nl_tools": nl_tools,
        "rewoo_tools": rewoo_tools,
        "nl_response_len": len(nl_response),
        "rewoo_response_len": len(rewoo_response),
        "nl_avg_confidence": sum(a.confidence for a in nl_actions) / max(len(nl_actions), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="A/B test: NL intent extraction vs ReWOO JSON")
    parser.add_argument("--base-url", default="http://localhost:1234",
                        help="LLM API base URL (default: localhost:1234 for LM Studio)")
    parser.add_argument("--model", default="",
                        help="Model name (empty = use whatever is loaded)")
    parser.add_argument("--test", type=int, default=-1,
                        help="Run only test N (0-indexed, -1=all)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  NL Intent Extraction vs ReWOO JSON — Live A/B Test            ║")
    print(f"║  Backend: {args.base_url:53s}║")
    print(f"║  Model:   {(args.model or '(auto)'):53s}║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Quick connectivity check
    try:
        with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())
            models = data.get("data", [])
            if models and not args.model:
                args.model = models[0].get("id", "")
            print(f"\n  Connected. Model: {args.model or '(unknown)'}")
    except Exception:
        try:
            with urllib.request.urlopen(f"{args.base_url}/api/tags", timeout=5) as r:
                data = json.loads(r.read().decode())
                models = data.get("models", [])
                if models and not args.model:
                    args.model = models[0].get("name", "")
                print(f"\n  Connected (Ollama). Model: {args.model or '(unknown)'}")
        except Exception as e:
            print(f"\n  ✖ Cannot connect to {args.base_url}: {e}")
            sys.exit(1)

    cases = TEST_CASES if args.test < 0 else [TEST_CASES[args.test]]
    results = []
    for test in cases:
        result = run_test(args.base_url, args.model, test)
        results.append(result)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    nl_total_actions = sum(r["nl_actions"] for r in results)
    rewoo_total_calls = sum(r["rewoo_calls"] for r in results)
    rewoo_parse_fails = sum(1 for r in results if not r["rewoo_valid"])
    nl_avg_time = sum(r["nl_time"] for r in results) / max(len(results), 1)
    rewoo_avg_time = sum(r["rewoo_time"] for r in results) / max(len(results), 1)
    nl_avg_conf = sum(r["nl_avg_confidence"] for r in results) / max(len(results), 1)

    print(f"\n  Tests run:           {len(results)}")
    print(f"  NL total actions:    {nl_total_actions}")
    print(f"  ReWOO total calls:   {rewoo_total_calls}")
    print(f"  ReWOO parse fails:   {rewoo_parse_fails}/{len(results)}")
    print(f"  NL avg time:         {nl_avg_time:.1f}s")
    print(f"  ReWOO avg time:      {rewoo_avg_time:.1f}s")
    print(f"  NL avg confidence:   {nl_avg_conf:.0%}")

    print("\n  Per-test breakdown:")
    for r in results:
        status = "✓" if r["rewoo_valid"] else "✖"
        print(f"    {r['name']:25s}  NL:{r['nl_actions']}actions  ReWOO:{status}{r['rewoo_calls']}calls  "
              f"NL:{r['nl_time']:.1f}s  ReWOO:{r['rewoo_time']:.1f}s")

    print()


if __name__ == "__main__":
    main()
