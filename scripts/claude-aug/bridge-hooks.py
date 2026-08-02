#!/usr/bin/env python3
"""Claude Code hook scripts for the Augmentum agent bridge.

Hooks: SessionStart (checkin), Stop (done + review), PreToolUse (approve gate).
All curl the /v1/tools/call endpoint directly — NOT the MCP bridge (hooks run
before MCP servers are live). Stdlib only; no dependencies beyond the Augmentum
proxy being up.

Env: inherit ANTHROPIC_BASE_URL + AUGMENTUM_API_KEY from the parent shell
(claude-aug.sh sets both). CLAUDE_SESSION_ID is set by Claude Code.

Agent session state lives at ~/.augmentum/agent-session-{session_id}.json so
Stop can read what SessionStart wrote — hooks share no memory otherwise.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

SESSION_DIR = os.path.join(os.path.expanduser("~"), ".augmentum", "agent-sessions")
BASE = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:6100").rstrip("/")
KEY = os.environ.get("AUGMENTUM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
SESSION_ID = os.environ.get("CLAUDE_SESSION_ID", "unknown")
HARNESS = os.environ.get("AUGMENTUM_HARNESS", "claude_code")
PROJECT = os.environ.get("AUGMENTUM_PROJECT_SLUG", "")


def api(path, data=None):
    """Call the Augmentum proxy. Returns parsed JSON, or {} on failure."""
    headers = {
        "x-api-key": KEY,
        "Content-Type": "application/json",
        "X-Augmentum-Harness": HARNESS,
    }
    if PROJECT:
        headers["X-Augmentum-Project"] = PROJECT
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)
    except Exception:
        return {}


def call_tool(name, arguments=None):
    """Thin wrapper around POST /v1/tools/call."""
    r = api("/v1/tools/call", {"tool": name, "arguments": arguments or {}})
    return r if r.get("ok") else {}


def session_path():
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, f"{SESSION_ID}.json")


def read_session():
    try:
        with open(session_path(), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def write_session(data):
    with open(session_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def delete_session():
    try:
        os.remove(session_path())
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Hook: SessionStart — register presence at task start
# ---------------------------------------------------------------------------

def on_session_start(event):
    # Grab the first user prompt as the agent title. The event structure has
    # "transcript" or "prompt" depending on version; be defensive.
    prompt = ""
    if isinstance(event, dict):
        prompt = str(
            event.get("prompt") or
            event.get("message", {}).get("content") or
            ""
        )[:120].strip()
    if not prompt:
        prompt = "claude-aug session"

    # Check in — server mints the agent_id on first call.
    r = call_tool("agent_checkin", {
        "title": prompt,
        "status": "working",
        "summary": "Session started",
    })
    if not r:
        write_session({"agent_id": "", "checkin_failed": True})
        return {}
    meta = r.get("metadata") or {}
    agent_id = meta.get("agent_id", "")
    answered = meta.get("answered_requests") or []
    assignments = meta.get("assignments") or []
    write_session({"agent_id": agent_id, "title": prompt, "answered": answered})

    # Reverse channel: a task the user/companion queued for this agent while
    # it was away. Surface it as session context so a fresh run picks it up.
    if assignments:
        tasks = "\n".join(
            f"- {a.get('task') or a.get('title') or ''}" for a in assignments)
        ctx = ("You have been assigned the following task(s) through Augmentum "
               "while you were away. Treat them as the user's request:\n" + tasks)
        return {"hookSpecificOutput": {
            "hookEventName": "SessionStart", "additionalContext": ctx}}
    return {}


# ---------------------------------------------------------------------------
# Hook: Stop — mark done + offer review
# ---------------------------------------------------------------------------

def on_stop(event):
    s = read_session()
    agent_id = s.get("agent_id", "")
    if not agent_id:
        return {}

    # Grab stop reason from the event if available, else generic.
    reason = ""
    if isinstance(event, dict):
        reason = str(event.get("reason", "") or event.get("stopReason", ""))
    summary = (f"Session ended{f': {reason}' if reason else ''}. "
               f"Review the transcript and decide what to do next.")

    # Mark done.
    call_tool("agent_checkin", {
        "agent_id": agent_id,
        "title": s.get("title", ""),
        "status": "done",
        "summary": summary,
    })

    # Offer review.
    call_tool("ask_user", {
        "agent_id": agent_id,
        "kind": "review",
        "title": "Session complete",
        "body": summary,
    })

    delete_session()
    return {}


# ---------------------------------------------------------------------------
# Hook: PreToolUse — approve gate via phone notification
# ---------------------------------------------------------------------------

PERMISSION_TIMEOUT_S = int(os.environ.get(
    "AUGMENTUM_APPROVE_TIMEOUT", "300"))  # 5 min default
POLL_INTERVAL_S = 10
ESCAPE_HATCH_PHRASE = "approve this session"
ESCAPE_HATCH_SESSION_FILE = os.path.join(SESSION_DIR, "auto_approve")


def _can_auto_approve():
    """Session-wide escape hatch: if the user approved ANY permission in this
    session through the bridge, auto-approve subsequent ones. Prevents
    notification spam during long coding runs."""
    try:
        return os.path.exists(ESCAPE_HATCH_SESSION_FILE)
    except OSError:
        return False


def _set_auto_approve():
    try:
        open(ESCAPE_HATCH_SESSION_FILE, "w").close()
    except OSError:
        pass


def _clear_auto_approve():
    try:
        os.remove(ESCAPE_HATCH_SESSION_FILE)
    except OSError:
        pass


def on_pretooluse(event):
    """Route permission gates to the phone. Only acts when the agent is
    actively checked in (SessionStart ran and got an agent_id)."""
    s = read_session()
    agent_id = s.get("agent_id", "")
    if not agent_id:
        return {}  # SessionStart never ran — let default UI handle it.

    # If the user already blessed this session, auto-approve.
    if _can_auto_approve():
        try:
            return {"decision": "approve"}
        except Exception:
            return {}

    # Build a concise permission question from the tool event.
    tool_name = ""
    if isinstance(event, dict):
        tool_name = str(event.get("tool_name") or event.get("name") or "")
    title = f"Approve {tool_name}?" if tool_name else "Permission needed"
    body = ""
    if isinstance(event, dict):
        args = event.get("tool_input") or event.get("arguments") or {}
        # Summarize arguments for context (truncate to avoid giant bodies).
        args_str = json.dumps(args, indent=0, default=str)[:300]
        if args_str:
            body = f"Tool: {tool_name}\nArgs: {args_str}"

    # Fire the approve notification.
    r = call_tool("ask_user", {
        "agent_id": agent_id,
        "kind": "approve",
        "title": title,
        "body": body,
    })
    if not r:
        return {}  # Bridge down — fall back to default permission prompt.
    request_id = (r.get("metadata") or {}).get("request_id", "")

    # Poll for answer.
    deadline = time.time() + PERMISSION_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_S)
        poll = call_tool("check_reply", {"request_id": request_id})
        if not poll:
            continue
        meta = poll.get("metadata") or {}
        if meta.get("status") != "answered":
            continue
        action = (meta.get("reply_action") or "").lower()
        text = (meta.get("reply_text") or "").lower()
        if action == "approve" or "yes" in text:
            _set_auto_approve()
            try:
                return {"decision": "approve"}
            except Exception:
                return {}
        elif action == "deny" or "no" in text:
            try:
                return {"decision": "deny"}
            except Exception:
                return {}
        else:
            # Free-text without clear signal — treat as deny to be safe.
            try:
                return {"decision": "deny"}
            except Exception:
                return {}

    # Timed out — let the default prompt appear in-terminal.
    return {}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

HOOKS = {
    "SessionStart": on_session_start,
    "Stop": on_stop,
    "PreToolUse": on_pretooluse,
}


def main():
    hook_name = os.environ.get("CLAUDE_HOOK_NAME", "")
    if not hook_name:
        # Fallback: --hook flag on command line
        args = sys.argv[1:]
        for i, a in enumerate(args):
            if a == "--hook" and i + 1 < len(args):
                hook_name = args[i + 1]
                break
    handler = HOOKS.get(hook_name)
    if handler is None:
        sys.stderr.write(f"bridge-hooks: unknown hook {hook_name!r}\n")
        sys.stderr.flush()
        print("{}")
        return 1

    # Read event from stdin (Claude Code pipes the hook event as JSON).
    event = {}
    try:
        raw = sys.stdin.read()
        if raw.strip():
            event = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        pass

    result = handler(event)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
