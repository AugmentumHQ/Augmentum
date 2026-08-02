"""The local-model agentic edit loop — the sovereign self-edit engine.

Drives a CAPABLE local model (your model list / a served coder model) with a
small, WORKSPACE-CONFINED file-edit toolset, iterating tool-calls until the model
finishes — yielding the generic event dicts ``NativeModelDriver`` normalizes. No
external platform, no token: this is what makes ``engine="native"`` debt-paydown /
self-edit fully sovereign.

Design for professionalism + testability:
  - the MODEL TURN is injected (``chat``: messages + tool specs → content +
    structured tool_calls), so the loop is model-agnostic and unit-testable with a
    scripted fake — no model needed to verify the loop mechanics;
  - the TOOLS are real file operations, **confined to the candidate worktree**
    (path-traversal refused) and named so ``base.tool_use_event`` classifies
    ``edit_file``/``write_file`` as mutating → ``file_change``;
  - the loop is BOUNDED (``max_iters``), tolerant (a tool error becomes a tool
    result the model can recover from, never a crash), and terminates on an
    explicit ``finish`` tool, a no-tool turn, or the iteration cap.

``make_native_loop(chat=…)`` returns the ``native_loop`` callable that
``engine_select.wire_selfedit_driver(engine="native", native_loop=…)`` consumes.
``build_local_chat(registry)`` is the live adapter (resolve_model_for_role +
native function-calling) — the one model-coupled piece.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from augmentum.coder.external.base import ExternalTask
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

FINISH_TOOL = "finish"
_MAX_READ = 20000  # cap file reads fed back to the model


def _resolve_in(workspace: str, rel: str) -> str:
    """Resolve ``rel`` inside ``workspace``; refuse anything that escapes it."""
    base = os.path.realpath(workspace)
    target = os.path.realpath(os.path.join(base, rel or "."))
    if target != base and not target.startswith(base + os.sep):
        raise ValueError(f"path escapes workspace: {rel!r}")
    return target


def _is_audit_infra(rel: str) -> bool:
    """True for the audit/scanner infrastructure — the JUDGE that grades the edit.
    The agent may READ it (to understand findings) but must never WRITE it: a real
    run was caught editing ``runtime_suppressions.json`` + the scanner scripts to
    make a finding 'disappear' instead of fixing the bug. Editing the judge is
    gaming, not fixing."""
    n = (rel or "").replace("\\", "/").strip()
    if n.startswith("./"):
        n = n[2:]
    parts = [p for p in n.rstrip("/").split("/") if p]
    base = parts[-1] if parts else ""
    return (
        ".claude" in parts
        or base.endswith("_suppressions.json")
        or base in ("security_exceptions.json", "audit_history.jsonl")
    )


_AUDIT_REFUSAL = (
    "ERROR: refused — that path is audit/scanner infrastructure (the judge that "
    "grades this change). Fix the REAL code in augmentum/ or ui/, never the scanner, "
    "its suppressions, or audit history."
)


# --- per-write validation + safe auto-fix (W1/W2) --------------------------
# The coder parse-checks and lints after EVERY write and the workspace auto-fixes
# syntax; the self-edit agent used to write blind and only learn at the final
# gate (the server.py indentation stall). These give the model IMMEDIATE,
# located feedback in the same loop, and clean up mechanical slips it can't be
# bothered to get right (tabs, trailing space).


def _syntax_check(rel: str, content: str) -> str:
    """A located syntax verdict for the just-written file, or '' if clean/n/a.
    Python via ``ast.parse`` (catches the IndentationError class), JSON via
    ``json.loads`` — the cheap in-process checks that make a break fixable at
    write time instead of at the gate."""
    low = rel.lower()
    if low.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:  # SyntaxError covers IndentationError/TabError
            loc = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
            src = (e.text or "").strip()
            tail = f" · {src[:100]}" if src else ""
            return f"{type(e).__name__} at {loc}: {e.msg}{tail}"
    elif low.endswith(".json"):
        try:
            json.loads(content)
        except ValueError as e:
            return f"JSON error: {e}"
    return ""


def _autofix(rel: str, text: str) -> tuple[str, str]:
    """Deterministic, SAFE cleanups on the model's own text (never other lines):
    leading tabs → 4 spaces (Python forbids mixed tabs/spaces anyway) and trailing
    whitespace. Returns (fixed_text, note). Conservative on purpose — it never
    reflows or guesses indentation depth (that's the model's job, now that W1
    tells it exactly where it's wrong)."""
    if not rel.lower().endswith((".py", ".pyi")):
        return text, ""
    notes: list[str] = []
    lines = text.split("\n")
    tabs = trail = 0
    out: list[str] = []
    for ln in lines:
        fixed = ln
        stripped_lead = fixed.lstrip("\t")
        if len(stripped_lead) != len(fixed):  # had leading tabs
            fixed = "    " * (len(fixed) - len(stripped_lead)) + stripped_lead
            tabs += 1
        rstripped = fixed.rstrip()
        if rstripped != fixed:
            trail += 1
            fixed = rstripped
        out.append(fixed)
    if tabs:
        notes.append(f"tabs→spaces on {tabs} line(s)")
    if trail:
        notes.append(f"trailing space on {trail} line(s)")
    return "\n".join(out), "; ".join(notes)


def _write_feedback(rel: str, written: str, fix_note: str) -> str:
    """Compose the tool result: the write, any auto-fix note, and — the point —
    a LOCATED syntax error the model must fix before finishing."""
    msg = f"wrote {rel}"
    if fix_note:
        msg += f" (auto-fixed: {fix_note})"
    err = _syntax_check(rel, written)
    if err:
        msg += (f"\n⚠ SYNTAX ERROR introduced — {err}\n"
                "The file will FAIL verification as-is. Fix this now (re-read the "
                "region and correct the indentation/syntax) before you finish.")
    return msg


# --- the workspace-confined toolset ---------------------------------------

async def _read_file(ws: str, args: dict) -> str:
    p = _resolve_in(ws, str(args.get("path", "")))
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read(_MAX_READ)


async def _list_dir(ws: str, args: dict) -> str:
    p = _resolve_in(ws, str(args.get("path", ".")))
    return "\n".join(sorted(os.listdir(p))) or "(empty)"


async def _write_file(ws: str, args: dict) -> str:
    rel = str(args.get("path", ""))
    if not rel:
        return "ERROR: path required"
    if _is_audit_infra(rel):
        return _AUDIT_REFUSAL
    p = _resolve_in(ws, rel)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    content, fix_note = _autofix(rel, str(args.get("content", "")))
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return _write_feedback(rel, content, fix_note)


async def _edit_file(ws: str, args: dict) -> str:
    rel = str(args.get("path", ""))
    old = str(args.get("old_string", ""))
    new = str(args.get("new_string", ""))
    if _is_audit_infra(rel):
        return _AUDIT_REFUSAL
    p = _resolve_in(ws, rel)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    if old and old not in text:
        return f"ERROR: old_string not found in {rel} (no change made)"
    if old and text.count(old) > 1:
        return f"ERROR: old_string is not unique in {rel} ({text.count(old)}×) — add context"
    # auto-fix only the model's NEW text (never other lines of the file)
    new_fixed, fix_note = _autofix(rel, new)
    updated = text.replace(old, new_fixed, 1) if old else new_fixed
    with open(p, "w", encoding="utf-8") as f:
        f.write(updated)
    # syntax-check the RESULTING file so an edit that breaks the whole module
    # (e.g. a mis-indented insert) is caught at write time, located.
    return _write_feedback(rel, updated, fix_note)


async def _search(ws: str, args: dict) -> str:
    """Recursively search the workspace for a literal substring → up to 50
    ``path:line: text`` hits. This is what lets the agent CONFIRM a symbol is
    unused (e.g. a CSS class with no JS/HTML reference) instead of reading the
    whole tree file by file. Skips vcs/build dirs; confined to the workspace."""
    needle = str(args.get("query", ""))
    if not needle:
        return "ERROR: query required"
    base = os.path.realpath(ws)
    skip = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    hits: list[str] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            p = os.path.join(root, fn)
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if needle in line:
                            hits.append(f"{os.path.relpath(p, base)}:{i}: "
                                        f"{line.strip()[:160]}")
                            if len(hits) >= 50:
                                return "\n".join(hits) + "\n…(truncated at 50)"
            except OSError:
                continue
    return "\n".join(hits) if hits else f"NO MATCHES for {needle!r}"


async def _finish(_ws: str, args: dict) -> str:
    return str(args.get("summary", ""))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[str, dict], Awaitable[str]]

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


DEFAULT_TOOLS: list[Tool] = [
    Tool("read_file", "Read a file in the workspace.",
         _p({"path": {"type": "string"}}, ["path"]), _read_file),
    Tool("list_dir", "List a directory in the workspace.",
         _p({"path": {"type": "string"}}, []), _list_dir),
    Tool("search", "Search the whole workspace for a literal string (e.g. a CSS "
         "class or symbol) to find every reference — use it to CONFIRM something "
         "is unused before removing it.",
         _p({"query": {"type": "string"}}, ["query"]), _search),
    Tool("write_file", "Create or overwrite a file with full content.",
         _p({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
         _write_file),
    Tool("edit_file", "Replace a unique old_string with new_string in a file.",
         _p({"path": {"type": "string"}, "old_string": {"type": "string"},
             "new_string": {"type": "string"}}, ["path", "old_string", "new_string"]),
         _edit_file),
    Tool(FINISH_TOOL, "Call when the task is complete, with a one-line summary.",
         _p({"summary": {"type": "string"}}, []), _finish),
]

_SYSTEM = (
    "You are an autonomous engineer editing the repository at {workspace} to "
    "accomplish ONE objective. You have a budget of ~{budget} steps — spend most of "
    "it EDITING, not exploring. Work in small, verifiable steps: use search to "
    "CONFIRM facts (e.g. that a symbol/class is unused) instead of reading the "
    "whole tree, read a file before you edit it, prefer edit_file (unique "
    "old_string) over rewriting, and keep the change minimal and focused. Don't "
    "over-explore: once you've confirmed the target, MAKE THE EDIT, then call "
    "{finish}. You MUST actually edit a file before finishing. NEVER edit the "
    "scanner, its suppressions, or anything under .claude/ — that's the judge; fix "
    "the real code. Do not ask questions — act."
)

# Injected model turn: (messages, tool_specs) → {"content": str,
# "tool_calls": [{"name": str, "args": dict, "id": str}]}.
ChatTurn = Callable[[list[dict], list[dict]], Awaitable[dict]]


def make_native_loop(*, chat: ChatTurn, tools: list[Tool] | None = None,
                     max_iters: int = 64) -> Callable[[ExternalTask], AsyncIterator[dict]]:
    """Build the ``native_loop`` (ExternalTask → CoderEvent-dict stream) over an
    injected model turn. Bounded by ``max_iters`` model turns (a turn may batch
    several tool calls); tools confined to the task workspace. A budget nudge fires
    in the last ~30% so the model commits to an edit instead of exploring until the
    cap (the observed cap-with-zero-edits failure mode on a large workspace)."""
    toolset = {t.name: t for t in (tools or DEFAULT_TOOLS)}
    specs = [t.spec() for t in toolset.values()]
    nudge_at = max(1, int(max_iters * 0.7))

    async def native_loop(task: ExternalTask) -> AsyncIterator[dict]:
        ws = task.workspace
        messages: list[dict] = [
            {"role": "system",
             "content": _SYSTEM.format(workspace=ws, finish=FINISH_TOOL, budget=max_iters)},
            {"role": "user", "content": task.prompt},
        ]
        nudged = False
        seen_calls: dict[str, int] = {}   # (tool+args) fingerprint → count, for stagnation
        for i in range(max_iters):
            # Budget nudge: once most of the step budget is spent, stop exploring and
            # commit — a weak model on a big tree will otherwise read/search until
            # it's cut off with no edit. Injected once.
            if not nudged and i >= nudge_at:
                nudged = True
                messages.append({"role": "user", "content": (
                    f"You are at step {i} of {max_iters} — running low. STOP exploring "
                    f"now: make the single edit you've identified (edit_file), then call "
                    f"{FINISH_TOOL}. If unsure, edit your best candidate now rather than "
                    "finishing with no change.")})
            try:
                turn = await chat(messages, specs)
            except Exception as exc:  # noqa: BLE001 — a model error ends the run, normalized
                log.warning("native_loop_chat_error", iter=i, error=repr(exc))
                yield {"kind": "failed", "text": f"model turn failed: {exc!r}"}
                return

            content = str(turn.get("content", "") or "")
            calls = list(turn.get("tool_calls") or [])
            reasoning = str(turn.get("reasoning", "") or "")
            if reasoning:                       # the model's "why" — captured for the transcript
                yield {"kind": "thinking", "text": reasoning}
            if content:
                yield {"kind": "message", "text": content}
            # Echo the assistant turn back in OpenAI WIRE shape (id + type +
            # function{name, JSON-string arguments}). The loop's internal
            # {name, args, id} shape is NOT what the chat backend / llama.cpp
            # template expects — feeding it back raw fails the next turn with
            # "Missing tool call type". Dispatch below still uses ``calls``.
            wire_calls = [{"id": c.get("id") or c.get("name") or "",
                           "type": "function",
                           "function": {"name": c.get("name", ""),
                                        "arguments": json.dumps(c.get("args") or {})}}
                          for c in calls]
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": wire_calls})

            if not calls:                       # the model stopped acting → done
                yield {"kind": "completed", "text": content or "done"}
                return

            for call in calls:
                name = str(call.get("name", ""))
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                cid = str(call.get("id", "") or name)
                if name == FINISH_TOOL:
                    summary = str(args.get("summary", "")) or content or "done"
                    yield {"kind": "completed", "text": summary}
                    return
                # Stagnation breaker (coder's identical-call pattern): the same
                # tool+args repeated with no progress means the model is wedged —
                # halt so it doesn't burn the whole budget looping (and, in a
                # self-heal pass, hand off to escalation instead of spinning).
                fp = f"{name}:{json.dumps(args, sort_keys=True, default=str)[:200]}"
                seen_calls[fp] = seen_calls.get(fp, 0) + 1
                if seen_calls[fp] >= 3:
                    log.info("native_loop_stagnation", tool=name, repeats=seen_calls[fp])
                    yield {"kind": "completed",
                           "text": f"stopped: repeated the same {name} call "
                                   f"{seen_calls[fp]}× with no progress"}
                    return
                yield {"kind": "tool_call", "tool": name, "args": args}
                tool = toolset.get(name)
                if tool is None:
                    result = f"ERROR: unknown tool {name!r}"
                else:
                    try:
                        result = await tool.run(ws, args)
                    except Exception as exc:  # noqa: BLE001 — tool error → result, model recovers
                        result = f"ERROR: {exc!r}"
                # The repeat nudge rides INSIDE the tool result — never as a separate
                # user message, which would split the assistant-tool_call → tool
                # pairing that strict backends (DeepSeek) 400 on.
                if seen_calls[fp] == 2:
                    result += ("\n\n(You just repeated this exact call and it didn't move "
                               "things forward — do something DIFFERENT next: re-read the "
                               "real current state or change approach. Don't repeat it.)")
                messages.append({"role": "tool", "tool_call_id": cid, "content": result})
                # Emit the RESULT too (not just the call) so the transcript captures
                # what came back — the per-write syntax feedback (W1), file reads,
                # tool errors — the record we need to actually see the loop iterate.
                yield {"kind": "tool_result", "tool": name,
                       "path": str(args.get("path", "") or ""), "text": result}

        log.info("native_loop_max_iters", iters=max_iters)
        yield {"kind": "completed", "text": f"stopped at the {max_iters}-step cap"}

    return native_loop


def build_local_chat(registry: Any, *, role: str = "utility", model: str = "",
                     temperature: float = 0.0, max_tokens: int = 2048) -> ChatTurn:
    """LIVE adapter: a ``ChatTurn`` backed by the provider registry + native
    function-calling. When ``model`` is set it pins that exact model via
    ``resolve_backend_with_fabric`` (the same resolver the /v1 chat path uses, so
    remote/fabric models resolve too); otherwise it falls back to the ``role``'s
    model via ``resolve_model_for_role``. Prefers the model's structured
    ``tool_calls``; falls back to the tolerant text parser. Needs a CAPABLE served
    coder model to be useful — the one model-coupled piece."""
    async def chat(messages: list[dict], specs: list[dict]) -> dict:
        from augmentum.models.base import InternalChatRequest, Message

        if model and hasattr(registry, "resolve_backend_with_fabric"):
            backend, resolved = await registry.resolve_backend_with_fabric(model)
        else:
            backend, resolved = await registry.resolve_model_for_role(role)
        if backend is None:
            return {"content": "", "tool_calls": []}
        req = InternalChatRequest(
            model=resolved or model,
            messages=[Message(role=m["role"], content=m.get("content", ""),
                              tool_calls=m.get("tool_calls"),
                              tool_call_id=m.get("tool_call_id")) for m in messages],
            tools=specs, tool_choice="auto", temperature=temperature,
            max_tokens=max_tokens, stream=False)
        resp = await backend.chat(req)
        msg = getattr(resp, "message", None)
        content = getattr(msg, "content", "") or ""
        # capture the model's REASONING too (thinking models emit it separately) —
        # otherwise the "why" behind each edit is dropped from the transcript.
        reasoning = (getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "")
                     or "")
        calls = _normalize_calls(getattr(msg, "tool_calls", None))
        if not calls and content:
            calls = _parse_fallback(resp, specs, backend)
        return {"content": content, "tool_calls": calls, "reasoning": reasoning}

    return chat


def _normalize_calls(raw: Any) -> list[dict]:
    """OpenAI-style ``tool_calls`` → the loop's {"name","args","id"} shape."""
    out: list[dict] = []
    for tc in raw or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", fn.get("args", {}))
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (ValueError, TypeError):
                raw_args = {}
        out.append({"name": name, "args": raw_args if isinstance(raw_args, dict) else {},
                    "id": tc.get("id", name) if isinstance(tc, dict) else name})
    return out


def _parse_fallback(resp: Any, specs: list[dict], backend: Any) -> list[dict]:
    """Tolerant text parser for models that emit tool calls as content."""
    try:
        from augmentum.tools.parsing import Tool as PTool
        from augmentum.tools.parsing import parse_tool_calls
        ptools = [PTool(name=s["function"]["name"], description="",
                        parameters=s["function"].get("parameters", {})) for s in specs]
        parsed = parse_tool_calls(resp, ptools, backend)
        return [{"name": getattr(p, "name", ""), "args": getattr(p, "arguments", {}) or {},
                 "id": getattr(p, "name", "")} for p in parsed]
    except Exception as exc:  # noqa: BLE001 — no parse → no calls (loop completes)
        log.debug("native_loop_parse_fallback_failed", error=repr(exc))
        return []
