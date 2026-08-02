"""Subagent delegation probe — does a model ELECT to fan out?

Phase-0 forcing experiment for the subagent-professionalization program
(``docs/superpowers/specs/2026-06-19-subagent-professionalization.md``).

THE QUESTION: the live db shows 0 `coder_subagent_runs` across 661 coder
turns, yet the full wiring is correct and `task_dispatch` IS offered to the
model on every native-tier turn. A prior session already rewrote the prompt
to v2.1 trigger-shaped guidance (2026-06-06) and it's STILL 0. Hypothesis:
local/open models don't spontaneously delegate. This harness tests that
hypothesis directly, across the local models we actually have — WITHOUT
needing a workspace or the full coder loop.

HOW: for each (model, scenario) it sends ONE OpenAI-compat chat completion
offering the REAL `task_dispatch` tool (verbatim description + input_schema
from ``augmentum/coder/tools.py``) ALONGSIDE decoy tools (read_file /
grep_search / code_edit) so the model has a genuine choice, plus a task
engineered to hit one of the shipped v2.1 triggers. It then classifies the
model's FIRST move:

  * DELEGATED — called ``task_dispatch`` (the behavior we want)
  * SELF      — called a decoy tool (chose to grind itself)
  * NONE      — emitted no tool call (just talked)

Two conditions bracket the answer:
  * default ("shipped"): system prompt carries the real delegation coaching
    block from ``prompts.py`` — production-like propensity.
  * ``--bare``: coaching lives ONLY in the tool description — baseline
    propensity.

Usage
-----
    python scripts/subagent_delegation_probe.py \
        --models Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL,Qwen3.6-27B-Q4_K_S \
        --out docs/superpowers/specs/data/delegation-probe-2026-06-19.json

Endpoint + key auto-load from ``~/.claude/review-config.json`` (or pass
``--endpoint`` / ``--api-key``). NOTE: each distinct model triggers a GPU
load/swap — keep the model list to what fits, expect minutes for cold loads.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# --- REAL task_dispatch tool, copied verbatim from augmentum/coder/tools.py
# (TaskDispatchTool.description + .input_schema). Kept inline so the probe is
# a standalone script with no app-boot dependency; if the tool changes,
# re-sync this block (it's the whole point that the probe is faithful).
_TASK_DISPATCH_DESCRIPTION = (
    "Spawn a focused subagent (Claude Code Task-style) when "
    "delegating beats grinding. Use ANY of:\n"
    "  - About to read 5+ files searching for one thing -> role=explore\n"
    "  - Stuck between 2-3 approaches for a non-trivial change -> role=plan\n"
    "  - Just made a complex multi-file change worth a second opinion -> role=review\n"
    "  - Need an API/library answer beyond your training -> role=research\n"
    "  - Auditing a file/diff for vulnerabilities -> role=security_review\n"
    "  - Need a threat model document for downstream security tooling -> role=threat_model\n\n"
    "The subagent runs in its own context budget - its file_reads "
    "don't crowd yours. Result is a structured tool output; "
    "treat it like any other tool's answer and continue.\n\n"
    "Don't dispatch for single-file edits or work the user explicitly asked "
    "you to do - those belong to YOU."
)

_TASK_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "description": (
                "Registered role name. Built-ins: explore, plan, review, "
                "research, security_review, threat_model."
            ),
        },
        "prompt": {
            "type": "string",
            "description": "The focused task to hand to the subagent.",
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Definition-of-done: concrete checkable conditions.",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hard limits the subagent must respect.",
        },
        "model": {"type": "string", "description": "Optional model override."},
        "context": {
            "type": "string",
            "enum": ["slim", "workspace", "hot"],
            "description": "How much parent context to inherit.",
        },
    },
    "required": ["role", "prompt"],
    "additionalProperties": False,
}

# Decoy tools — realistic alternatives so "do it myself" is a genuine option.
_DECOY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace by path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search the workspace for a regex pattern; returns matching file:line.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_edit",
            "description": "Apply an edit to a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]

_TASK_DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "task_dispatch",
        "description": _TASK_DISPATCH_DESCRIPTION,
        "parameters": _TASK_DISPATCH_SCHEMA,
    },
}

# --- REFRAME condition: the same subagent roles, but dressed as CONCRETE,
# in-distribution tools the model is comfortable calling. No "subagent",
# "delegate", "role", or "success_criteria" in sight — just plain verbs with
# one obvious arg and a predictable "returns findings" contract. Underneath,
# each maps 1:1 to dispatch(role=...). Tests Matt's hypothesis: the model
# won't call an abstract meta-tool but WILL call a familiar concrete one.
_REFRAME_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore_codebase",
            "description": (
                "Search the codebase to find where and how something is "
                "implemented or used. Reads across as many files as needed and "
                "returns a structured summary of every relevant location "
                "(file:line + what it does). Use for 'find all callers of X', "
                "'where is Y handled', 'how does Z work across the repo'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string",
                               "description": "What to find, in plain words."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_changes",
            "description": (
                "Get a thorough second-opinion review of code for "
                "correctness, edge cases, and security. Returns findings with "
                "severity. Use after a complex or multi-file change."
            ),
            "parameters": {
                "type": "object",
                "properties": {"scope": {"type": "string",
                               "description": "What to review (files/area)."}},
                "required": ["scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_topic",
            "description": (
                "Look up current external/library/API information beyond your "
                "training. Returns a sourced summary. Use when you need "
                "up-to-date or authoritative facts you can't be sure of."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approach",
            "description": (
                "Produce a considered implementation plan that weighs 2-3 "
                "alternatives with tradeoffs before any code is written. "
                "Returns a recommended approach. Use for non-trivial changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
    },
]

# Names that count as "delegated" in each condition.
_DELEGATE_NAMES_DEFAULT = {"task_dispatch"}
_DELEGATE_NAMES_REFRAME = {"explore_codebase", "review_changes",
                           "research_topic", "plan_approach"}

# The shipped delegation coaching from prompts.py:872-889 (default condition).
_COACHING = (
    "You also have `task_dispatch` to spawn a focused subagent in its own "
    "context budget. Concrete triggers - dispatch when ANY of:\n"
    "  * About to read 5+ files looking for one thing -> role=explore\n"
    "  * Stuck between 2-3 approaches for a non-trivial change -> role=plan\n"
    "  * Just made a complex multi-file change worth a second look -> role=review\n"
    "  * Need an API/library answer beyond your training -> role=research\n"
    "  * Auditing a file/diff for vulnerabilities -> role=security_review\n"
    "Keep each subagent's job NARROW. Don't dispatch for single-file edits or "
    "work the user explicitly asked YOU to do."
)

_SYSTEM_BASE = (
    "You are a coding agent operating inside a workspace container. You have "
    "tools to read, search, and edit code, run shell commands, and delegate "
    "focused sub-tasks. Work efficiently: use the right tool for the job. "
    "When you take an action, call a tool."
)

# Bait scenarios — each engineered to hit one shipped trigger. The RIGHT move
# for each is to dispatch the named role; a model that instead calls a decoy
# (or just talks) is declining to delegate.
_SCENARIOS = [
    {
        "key": "explore_many_files",
        "trigger": "explore",
        "user": (
            "I need to change how `resolve_backend_for_model` works. Before "
            "touching it, find EVERY call site of that function across this "
            "large codebase (dozens of modules) and summarize what each one "
            "passes and expects back. There are many files to look through."
        ),
    },
    {
        "key": "plan_approaches",
        "trigger": "plan",
        "user": (
            "We need to add per-tenant rate limiting to the API. There are "
            "2-3 viable approaches (middleware token bucket, per-route "
            "decorator, gateway-level) with real tradeoffs. Think it through "
            "and come back with a recommended plan before we write code."
        ),
    },
    {
        "key": "review_multifile",
        "trigger": "review",
        "user": (
            "I just finished a complex refactor that touched auth across 6 "
            "files (session tokens, middleware, the user store, and two route "
            "handlers). Get a thorough second opinion on whether it's correct "
            "and secure before I ship."
        ),
    },
    {
        "key": "research_api",
        "trigger": "research",
        "user": (
            "What are the current request/response parameters and rate limits "
            "for the Anthropic Messages API as of 2026? I need accurate, "
            "up-to-date details beyond what you might remember."
        ),
    },
]


def _load_cfg() -> dict[str, Any]:
    p = Path(os.path.expanduser("~/.claude/review-config.json"))
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _post_chat(
    endpoint: str, api_key: str, body: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _warmup(endpoint: str, api_key: str, model: str, *, attempts: int, timeout: float) -> bool:
    """Force a cold model into memory before probing.

    A model swap routed through the gateway returns 502 (empty body) while
    the load is in flight. Send a trivial completion and retry with backoff
    until it answers 200 — so the real scenario calls aren't scored as
    refusals when they were really load timeouts.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    for i in range(attempts):
        try:
            _post_chat(endpoint, api_key, body, timeout=timeout)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504):
                print(f"  (warming {model}: {e.code}, attempt {i+1}/{attempts})")
                time.sleep(15)
                continue
            return False
        except Exception:
            time.sleep(15)
    return False


def _classify(resp: dict[str, Any], delegate_names: set[str]) -> tuple[str, str]:
    """Return (verdict, detail). verdict in DELEGATED|SELF|NONE|ERROR.

    A first tool call whose name is in ``delegate_names`` counts as
    DELEGATED — whether that's the abstract ``task_dispatch`` (default
    condition) or a concrete reframed role-tool like ``explore_codebase``.
    """
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "ERROR", "no message in response"
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # Some models emit a single function_call (legacy) instead.
        fc = msg.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            tool_calls = [{"function": fc}]
    if not tool_calls:
        text = (msg.get("content") or "").strip().replace("\n", " ")
        return "NONE", text[:120]
    first = tool_calls[0]
    fn = first.get("function") if isinstance(first.get("function"), dict) else first
    name = str(fn.get("name") or "")
    if name in delegate_names:
        if name == "task_dispatch":
            return "DELEGATED", f"role={_extract_role(fn.get('arguments'))}"
        return "DELEGATED", name
    return "SELF", name


def _extract_role(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return "?"
    if isinstance(args, dict):
        return str(args.get("role", "?"))
    return "?"


def _run(
    *, endpoint: str, api_key: str, models: list[str], bare: bool,
    reframe: bool, temperature: float, max_tokens: int, timeout: float,
    verbose: bool, warmup_attempts: int,
) -> dict[str, Any]:
    if reframe:
        # Concrete role-tools, no coaching needed (that's the whole point):
        # the surface is in-distribution. Keep read_file/code_edit as genuine
        # "do it myself" alternatives; drop grep (explore_codebase IS search).
        system = _SYSTEM_BASE
        tools = [*_REFRAME_TOOLS, _DECOY_TOOLS[0], _DECOY_TOOLS[2]]
        delegate_names = _DELEGATE_NAMES_REFRAME
        condition = "reframe"
    else:
        system = _SYSTEM_BASE if bare else (_SYSTEM_BASE + "\n\n" + _COACHING)
        tools = [_TASK_DISPATCH_TOOL, *_DECOY_TOOLS]
        delegate_names = _DELEGATE_NAMES_DEFAULT
        condition = "bare" if bare else "shipped"
    results: dict[str, Any] = {
        "condition": condition,
        "models": {},
    }
    for model in models:
        print(f"\n### {model}")
        warm = _warmup(endpoint, api_key, model, attempts=warmup_attempts, timeout=timeout)
        if not warm:
            print("  !! could not warm model (load failed / OOM / gateway) - SKIPPED")
            results["models"][model] = {
                "delegated": 0, "total": len(_SCENARIOS),
                "available": False, "scenarios": {},
            }
            continue
        per_scenario: dict[str, Any] = {}
        for sc in _SCENARIOS:
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": sc["user"]},
                ],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            verdict, detail = "ERROR", "unset"
            for attempt in range(3):
                try:
                    resp = _post_chat(endpoint, api_key, body, timeout=timeout)
                    verdict, detail = _classify(resp, delegate_names)
                    break
                except urllib.error.HTTPError as e:
                    detail = f"HTTP {e.code}"
                    if e.code in (502, 503, 504) and attempt < 2:
                        time.sleep(10)
                        continue
                    verdict = "ERROR"
                    break
                except Exception as e:  # noqa: BLE001 — never crash the matrix
                    verdict, detail = "ERROR", f"{type(e).__name__}: {e}"
                    break
            mark = {"DELEGATED": "[OK]", "SELF": "[--]", "NONE": "[  ]",
                    "ERROR": "[ER]"}.get(verdict, "[??]")
            line = f"  {mark} {sc['key']:20s} (want {sc['trigger']:14s}) -> {verdict}"
            if verbose or verdict in ("DELEGATED", "ERROR"):
                line += f"  {detail}"
            print(line)
            per_scenario[sc["key"]] = {
                "trigger": sc["trigger"], "verdict": verdict, "detail": detail,
            }
        errored = sum(1 for v in per_scenario.values() if v["verdict"] == "ERROR")
        delegated = sum(1 for v in per_scenario.values() if v["verdict"] == "DELEGATED")
        results["models"][model] = {
            "delegated": delegated,
            "errored": errored,
            "total": len(_SCENARIOS),
            "available": errored < len(_SCENARIOS),
            "scenarios": per_scenario,
        }
        suffix = f" ({errored} errored)" if errored else ""
        print(f"  => delegated {delegated}/{len(_SCENARIOS)}{suffix}")
    return results


def main(argv: list[str] | None = None) -> int:
    cfg = _load_cfg()
    ap = argparse.ArgumentParser(description="Does a model elect to call task_dispatch?")
    ap.add_argument("--endpoint", default=cfg.get("endpoint", "https://localhost:6443"))
    ap.add_argument("--api-key", default=cfg.get("apiKey", ""))
    ap.add_argument("--models", default="", help="comma-separated model ids")
    ap.add_argument("--bare", action="store_true",
                    help="drop the system-prompt coaching (tool-description-only)")
    ap.add_argument("--reframe", action="store_true",
                    help="offer concrete role-tools (explore_codebase/...) "
                         "instead of abstract task_dispatch")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--warmup-attempts", type=int, default=12,
                    help="retries (15s apart) to load a cold model before probing")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not args.api_key:
        ap.error("no api key (set --api-key or ~/.claude/review-config.json)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        ap.error("pass --models id1,id2,...")

    results = _run(
        endpoint=args.endpoint, api_key=args.api_key, models=models,
        bare=args.bare, reframe=args.reframe, temperature=args.temperature,
        max_tokens=args.max_tokens, timeout=args.timeout, verbose=args.verbose,
        warmup_attempts=args.warmup_attempts,
    )

    print("\n=== delegation matrix ({}) ===".format(results["condition"]))
    for model, mr in results["models"].items():
        if not mr.get("available", True):
            print(f"  {model:48s} UNAVAILABLE (could not load)")
        else:
            err = f"  ({mr.get('errored', 0)} errored)" if mr.get("errored") else ""
            print(f"  {model:48s} {mr['delegated']}/{mr['total']}{err}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
