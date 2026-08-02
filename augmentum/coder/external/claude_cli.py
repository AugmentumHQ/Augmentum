"""Claude CLI driver — the DEDICATED-CONTAINER path.

Per the 2026-06-21 decision, the external coder runs inside a persistent,
per-user Claude-home container (node + the ``claude`` CLI baked in), NOT the
Python SDK in the app process. Augmentum drives it by exec'ing
``claude -p <task> --output-format stream-json`` inside that container and
parsing the JSONL event stream into normalized ``CoderEvent``s — the same shape
the SDK path emits, so the trail / consent gate / engineering_log are unchanged.

Decoupled from the container infra for testability: the driver takes a
``runner`` — an async callable that execs an argv inside the container and yields
stdout lines. Tests inject a fake runner streaming canned JSONL; the live path
wires it to ``ContainerManager.run_command`` against the Claude-home container.
The parser (``parse_cli_event``) is pure and fully unit-tested.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import AsyncIterator, Callable

from augmentum.coder.external.base import (
    CoderEvent,
    ExternalCoderDriver,
    ExternalTask,
    tool_use_event,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# task.permission → claude CLI --permission-mode value.
_PERMISSION_MAP = {
    "confirm_mutations": "default",
    "auto": "acceptEdits",
    "plan": "plan",
}

# An async exec runner: given argv, yields decoded stdout lines from a process
# running INSIDE the Claude-home container.
Runner = Callable[[list[str]], AsyncIterator[str]]


def build_claude_argv(task: ExternalTask, *, resume_session_id: str = "") -> list[str]:
    """Build the ``claude`` CLI argv for a headless streaming run. The OAuth
    token is delivered via the container env (CLAUDE_CODE_OAUTH_TOKEN), NOT on
    the command line, so it never lands in process listings or logs.

    ``resume_session_id``: when set, ``--resume <id>`` is prepended so Claude
    Code reloads that session's full transcript + TODO state and CONTINUES it
    (its native resume). Requires the same exec CWD as the original run — we
    always exec from ``/workspace`` — and the session transcript still on disk
    (we redirect ``CLAUDE_CONFIG_DIR`` under the persistent ``/workspace`` volume
    so it survives container recreate)."""
    argv = ["claude"]
    if resume_session_id:
        argv += ["--resume", resume_session_id]
    argv += [
        "-p", task.prompt,
        "--output-format", "stream-json",
        "--verbose",  # stream-json requires verbose to emit intermediate events
        "--permission-mode", _PERMISSION_MAP.get(task.permission, "default"),
    ]
    if task.model:
        argv += ["--model", task.model]
    # Scope the agent to the project dir (the home container holds many).
    if task.workspace:
        argv += ["--add-dir", task.workspace]
    if task.allowed_tools:
        argv += ["--allowed-tools", ",".join(task.allowed_tools)]
    return argv


def _cli_block_to_event(block: dict) -> CoderEvent | None:
    """Map one content block of a stream-json assistant message to a CoderEvent.
    Blocks carry a ``type`` field ("text" | "thinking" | "tool_use")."""
    btype = block.get("type")
    if btype == "text":
        text = (block.get("text") or "").strip()
        return CoderEvent(kind="message", text=text, raw=block) if text else None
    if btype == "thinking":
        return CoderEvent(kind="thinking", text=(block.get("thinking") or "")[:200], raw=block)
    if btype == "tool_use":
        return tool_use_event(block.get("name") or "", block.get("input") or {}, raw=block)
    return None


def parse_cli_event(obj: dict) -> list[CoderEvent]:
    """Map one stream-json JSONL object to zero or more CoderEvents. Pure +
    fully testable. Shapes (claude --output-format stream-json):
      {"type":"system","subtype":"init",...}
      {"type":"assistant","message":{"content":[{type:text|thinking|tool_use}]}}
      {"type":"user","message":{...tool_result...}}       (echo — ignored)
      {"type":"result","subtype":"success"|...,"is_error":bool,"result":...}
    """
    if not isinstance(obj, dict):
        return []
    t = obj.get("type")
    if t == "system":
        return [CoderEvent(kind="started", raw=obj)]
    if t == "assistant":
        content = (obj.get("message") or {}).get("content")
        if isinstance(content, list):
            return [e for e in (_cli_block_to_event(b) for b in content if isinstance(b, dict)) if e]
        return []
    if t == "result":
        if obj.get("is_error") or obj.get("subtype") not in (None, "success"):
            return [CoderEvent(kind="failed", text=str(obj.get("result") or obj.get("subtype") or "error"), raw=obj)]
        return [CoderEvent(kind="completed", text=str(obj.get("result") or "")[:200], raw=obj)]
    # "user" tool-result echoes and anything else: nothing to surface.
    return []


class ClaudeCliDriver(ExternalCoderDriver):
    id = "claude_code"
    label = "Claude Code"

    def __init__(self, *, runner: Runner, has_credential: bool = True) -> None:
        # ``runner`` execs argv inside the Claude-home container and yields
        # stdout lines. ``has_credential`` is resolved by the caller (token
        # present in the container env / volume).
        self._runner = runner
        self._has_credential = has_credential

    async def is_available(self) -> bool:
        # Availability here = a runner (a live home container) + a credential.
        # The CLI/node presence is guaranteed by the home image.
        return self._runner is not None and self._has_credential

    async def run(self, task: ExternalTask) -> AsyncIterator[CoderEvent]:
        argv = build_claude_argv(task)
        yield CoderEvent(kind="started", text=task.prompt[:120])
        saw_terminal = False
        try:
            async for line in self._runner(argv):
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    # Non-JSON noise on stdout — surface as prose, don't crash.
                    yield CoderEvent(kind="message", text=line[:200])
                    continue
                for ev in parse_cli_event(obj):
                    if ev.kind == "started":
                        continue  # we already emitted one
                    if ev.kind in ("completed", "failed"):
                        saw_terminal = True
                    yield ev
        except Exception as exc:  # noqa: BLE001 — normalize to a failure
            log.warning("claude_cli_run_failed", error=repr(exc))
            yield CoderEvent(kind="failed", text=repr(exc))
            return
        if not saw_terminal:
            # Process ended without a result event (killed / crashed).
            yield CoderEvent(kind="failed", text="run ended without a result event")


def _cmdline(argv: list[str]) -> str:
    """Debug helper — shell-quoted argv for logs (token is in env, not here)."""
    return " ".join(shlex.quote(a) for a in argv)
