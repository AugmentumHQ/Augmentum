#!/usr/bin/env python3
"""MCP stdio bridge: exposes Augmentum's ATP tools (/v1/tools) to Claude Code.

Registered in ~/.augmentum/claude-config/.claude.json, so it only exists
in claude-aug sessions. Stdlib only — no dependencies.
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:6100").rstrip("/")
KEY = os.environ.get("AUGMENTUM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")


def load_env_fallback():
    """If launched outside claude-aug (no env), read claude.env directly."""
    global BASE, KEY
    if KEY:
        return
    path = os.path.expanduser("~/.augmentum/claude.env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("export AUGMENTUM_API_KEY="):
                    KEY = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("export ANTHROPIC_BASE_URL="):
                    BASE = line.split("=", 1)[1].strip().strip("\"'").rstrip("/")
    except OSError:
        pass


def api(path, data=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={
            "x-api-key": KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            # Identity for server-side memory scoping ({harness}:{project}).
            # The bridge runs with cwd = the session's project directory.
            "X-Augmentum-Harness": "claude_code",
            "X-Augmentum-Project": os.path.basename(os.getcwd()) or "default",
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        # ATP reports tool errors as HTTP 4xx/5xx with a JSON body
        # ({ok:false, error:...} or FastAPI {detail:...}) — surface the
        # real message instead of an opaque status code.
        try:
            body = json.loads(exc.read())
        except Exception:
            raise exc
        if isinstance(body, dict):
            if "error" in body:
                return {"ok": False, "error": body["error"],
                        "warnings": body.get("warnings")}
            if "detail" in body:
                return {"ok": False, "error": str(body["detail"])}
        raise exc


# Large outputs (youtube transcripts, research dumps) are spilled to a
# file instead of truncated: the model gets a preview + the path, and can
# Grep/Read the full text on demand. Spill files are pruned by age.
MAX_OUTPUT_CHARS = 20_000
SPILL_DIR = os.path.join(os.path.expanduser("~"), ".augmentum", "atp-outputs")
SPILL_MAX_AGE_S = 24 * 3600
SPILL_MAX_FILES = 200


def spill(tool_name: str, text: str) -> str:
    """Write full output to a spill file, prune old ones, return the path."""
    os.makedirs(SPILL_DIR, exist_ok=True)
    try:  # prune: by age, then by count (oldest first)
        entries = sorted(
            (os.path.join(SPILL_DIR, f) for f in os.listdir(SPILL_DIR)),
            key=os.path.getmtime,
        )
        import time
        now = time.time()
        for p in entries[:-SPILL_MAX_FILES] + [
            p for p in entries if now - os.path.getmtime(p) > SPILL_MAX_AGE_S
        ]:
            try:
                os.remove(p)
            except OSError:
                pass
    except OSError:
        pass
    import hashlib
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:10]
    path = os.path.join(SPILL_DIR, f"{tool_name}-{digest}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(text)
    return path


# Hard-won usage hints injected into tool descriptions — models weight the
# schema at selection time far more than session guidance. Keep these STATIC
# (they're part of the KV-cache prefix hash; churn = re-prefill).
USAGE_HINTS = {
    "math_verify": "Numeric expressions only (e.g. expression='sqrt(2)*sqrt(2)', "
                   "expected='2'). Does NOT solve equations — use python_exec + sympy.",
    "python_exec": "STATELESS: each call is a fresh interpreter — no variables or "
                   "imports persist between calls. Bundle related work in one call.",
    "context_peek": "slot must be one of: page, note, playing, working, recent, "
                    "referents, abilities, loaded.",
    "document_parse": "Parses files on the AUGMENTUM SERVER or artifacts — NOT the "
                      "local filesystem (use the Read tool for local files).",
    "memory_store": "Writes a STAGED candidate needing human approval in the "
                    "Augmentum UI — memory_recall won't see it until promoted. "
                    "Tell the user what you staged.",
    "memory_recall": "Empty result can mean 'stored but not yet human-approved', "
                     "not just 'never stored'.",
    "search_files": "Searches files indexed by Augmentum server-side — for the "
                    "local repo use Grep/Glob.",
    "wikipedia": "Relevance can be loose — verify the returned title matches "
                 "intent before citing.",
    "research": "Multi-source with citations; can take 30-120s on hard queries — "
                "prefer one research call over chaining web_search yourself.",
}


def mcp_tools_list():
    tools = []
    for t in api("/v1/tools/list").get("tools", []):
        schema = t.get("parameters") or {"type": "object", "properties": {}}
        desc = t.get("description", "")
        hint = USAGE_HINTS.get(t["name"])
        if hint:
            desc += f"\nUSAGE: {hint}"
        hints = t.get("error_hints")
        if hints:
            desc += f"\nError hints: {hints}"
        tools.append({"name": t["name"], "description": desc, "inputSchema": schema})
    return tools


def mcp_tools_call(name, arguments):
    r = api("/v1/tools/call", {"tool": name, "arguments": arguments or {}})
    if r.get("ok"):
        out = r.get("output", "")
        if not isinstance(out, str):
            out = json.dumps(out, indent=2)
        if len(out) > MAX_OUTPUT_CHARS:
            try:
                path = spill(name, out)
                out = out[:MAX_OUTPUT_CHARS] + (
                    f"\n\n[output is {len(out)} chars; full text saved to "
                    f"{path} - Grep it for specifics or Read it with "
                    "offset/limit instead of re-calling this tool]"
                )
            except OSError:
                out = out[:MAX_OUTPUT_CHARS] + (
                    f"\n\n[... truncated at {MAX_OUTPUT_CHARS} chars; "
                    "narrow the request (max_chars/num_results/limit)]"
                )
        return {"content": [{"type": "text", "text": out or "(empty result)"}]}
    err = r.get("error") or "unknown error (tool returned an empty error message)"
    warnings = r.get("warnings") or []
    text = f"ATP error: {err}"
    if warnings:
        text += "\nWarnings: " + "; ".join(map(str, warnings))
    return {"content": [{"type": "text", "text": text}], "isError": True}


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    load_env_fallback()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method", "")

        if method == "initialize":
            result = {
                "protocolVersion": msg.get("params", {}).get(
                    "protocolVersion", "2024-11-05"
                ),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "atp", "version": "1.0.0"},
            }
        elif method == "tools/list":
            try:
                result = {"tools": mcp_tools_list()}
            except Exception as exc:
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32000,
                    "message": f"ATP unreachable at {BASE}: {exc}",
                }})
                continue
        elif method == "tools/call":
            params = msg.get("params", {})
            try:
                result = mcp_tools_call(params.get("name"), params.get("arguments"))
            except Exception as exc:
                result = {
                    "content": [{"type": "text", "text": f"ATP bridge error: {exc}"}],
                    "isError": True,
                }
        elif method == "ping":
            result = {}
        elif mid is None:
            continue  # notification (e.g. notifications/initialized)
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": f"Method not found: {method}",
            }})
            continue

        send({"jsonrpc": "2.0", "id": mid, "result": result})


if __name__ == "__main__":
    main()
