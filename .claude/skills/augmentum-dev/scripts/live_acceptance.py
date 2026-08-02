#!/usr/bin/env python3
"""Live ground-truth acceptance checks against the RUNNING Augmentum stack.

Where audit.py asserts code shape, this asserts observed reality: real
HTTP routes, real containers, real browser sessions, real DOM state.
Built from the 2026-07-17 browser-sidecar acceptance pass — the two bugs
that pass caught (Docker bridge isolation, the page-reopen state reset)
were invisible to every static scanner and unit test.

Usage:
    python live_acceptance.py                       # all non-disruptive suites
    python live_acceptance.py --suite browser       # one suite
    python live_acceptance.py --allow-disruption    # include stop/start drills
    python live_acceptance.py --model <id>          # model for LLM checks
    python live_acceptance.py --list                # show registry
    python live_acceptance.py --init                # write config template
    python live_acceptance.py --format=json         # machine-readable

Checks that need an LLM read the model from --model or
live_acceptance.local.json; they SKIP (never auto-pick) when unset.
Disruptive checks (they stop/start services) run only with
--allow-disruption. Every run cleans up after itself: bench token
revoked, test workspaces deleted, sidecar sessions closed.

Exit code: number of FAILED checks (0 = all pass/skip).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import bench_harness as bh
from _common import cyan, green, red, yellow

# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    suite: str
    fn: Callable[[dict], tuple[bool, str]]
    disruptive: bool = False
    needs_model: bool = False
    depends_on: tuple[str, ...] = ()


@dataclass
class Result:
    name: str
    status: str  # pass / fail / skip
    evidence: str
    seconds: float = 0.0


CHECKS: list[Check] = []


def check(suite: str, *, disruptive: bool = False, needs_model: bool = False,
          depends_on: tuple[str, ...] = ()):
    def _wrap(fn):
        CHECKS.append(Check(fn.__name__, suite, fn, disruptive,
                            needs_model, depends_on))
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# In-container probe scripts (run inside augmentum-main via stdin; results
# come back as one JSON line on stdout — see bench_harness docstring for why)
# ---------------------------------------------------------------------------

_PROBE_PRELUDE = """
import asyncio, json, os, sys
sys.path.insert(0, "/app")

async def _cm_for(name):
    import aiodocker
    docker = aiodocker.Docker()
    cid = None
    for c in await docker.containers.list():
        info = await c.show()
        if info.get("Name", "").lstrip("/") == name:
            cid = info["Id"]; target = c
    if cid is None:
        print(json.dumps({"ok": False, "error": f"container {name} not found"}))
        raise SystemExit(0)

    class _Info: container_id = cid

    class _CM:
        _docker = docker
        async def _get_workspace(self, ws): return _Info()
        async def run_command(self, ws, cmd, timeout=30.0):
            ex = await target.exec(cmd=cmd, stdout=True, stderr=True)
            stream = ex.start(detach=False)
            chunks = []
            async with stream:
                while True:
                    msg = await stream.read_out()
                    if msg is None: break
                    if msg.data: chunks.append(msg.data)
            return b"".join(chunks).decode("utf-8", "replace")
    return docker, _CM()
"""

_LADDER_PROBE = _PROBE_PRELUDE + """
async def main():
    from augmentum.coder import browser
    ws = os.environ["WS_ID"]
    docker, cm = await _cm_for(os.environ["WS_CONTAINER"])
    url = "http://localhost:5199/"
    r1 = await browser.playwright_action(cm, ws, url=url, action="click", selector="#b")
    r2 = await browser.playwright_action(cm, ws, url=url, action="click", selector="#b")
    r3 = await browser.playwright_evaluate(
        cm, ws, url=url, expression="document.querySelector('#b').textContent")
    print(json.dumps({
        "ok": bool(r1.get("ok") and r2.get("ok") and r3.get("ok")),
        "engines": [r.get("engine") for r in (r1, r2, r3)],
        "counter": r3.get("result_json"),
        "error": r1.get("error") or r2.get("error") or r3.get("error") or "",
    }))
    await docker.close()

asyncio.run(main())
"""

_GATE_PROBE = _PROBE_PRELUDE + """
import urllib.request

async def main():
    from augmentum.builds.verify import gate_summary, run_behavior_gate
    ws = os.environ["WS_ID"]
    docker, cm = await _cm_for(os.environ["WS_CONTAINER"])
    staged_legacy = []
    _orig_fw = getattr(cm, "file_write", None)
    async def _file_write(w, path, content):
        staged_legacy.append(path)
    cm.file_write = _file_write

    class _HTTPBackend:
        async def chat(self, req):
            body = json.dumps({
                "model": req.model,
                "messages": [{"role": m.role, "content": m.content}
                             for m in req.messages],
                "temperature": req.temperature, "max_tokens": 2000,
            }).encode()
            def _call():
                r = urllib.request.Request(
                    "http://localhost:6100/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Cookie": "augmentum_session=" + os.environ["BENCH_TOKEN"],
                             "Origin": "http://localhost:6100"})
                with urllib.request.urlopen(r, timeout=300) as resp:
                    return json.load(resp)
            data = await asyncio.to_thread(_call)
            import types
            text = data["choices"][0]["message"]["content"] or ""
            return types.SimpleNamespace(message=types.SimpleNamespace(
                content=text, reasoning_content=""))

    behaviors = [
        {"id": "add-item", "description":
         "Typing a todo and clicking Add appends it to the list"},
        {"id": "count-updates", "description":
         "The item counter updates to reflect the number of items"},
        {"id": "empty-ignored", "description":
         "Clicking Add with an empty input does not add an item"},
    ]
    merged, ran = await run_behavior_gate(
        container_manager=cm, workspace_id=ws, backend=_HTTPBackend(),
        model=os.environ["BENCH_MODEL"], behaviors=behaviors)
    print(json.dumps({
        "ok": bool(ran), "summary": gate_summary(merged),
        "staged_legacy": staged_legacy,
        "statuses": {b["id"]: b.get("status") for b in merged},
    }))
    await docker.close()

asyncio.run(main())
"""

_ISOLATION_PROBE = _PROBE_PRELUDE + """
async def main():
    from augmentum.coder import browser_sidecar as bs
    ws = os.environ["WS_ID"]
    docker, cm = await _cm_for(os.environ["WS_CONTAINER"])
    s_a = bs.session_for_workspace(ws)
    s_b = bs.session_for_workspace(ws, prefix="gate")
    url = await bs.reachable_url(cm, ws, "http://localhost:5199/")

    async def drive(session, n):
        await bs.run_cli(docker, ["open", url], session=session, timeout=30.0)
        for i in range(n):
            await bs.run_cli(docker, ["click", "#b"], session=session, timeout=10.0)
        r = await bs.run_cli(docker, ["get", "text", "#b"], session=session, timeout=10.0)
        return r.get("text")

    a, b = await asyncio.gather(drive(s_a, 3), drive(s_b, 5))
    for s in (s_a, s_b):
        await bs.run_cli(docker, ["close"], session=s, timeout=10.0)
    print(json.dumps({"ok": a == "3" and b == "5", "a": a, "b": b}))
    await docker.close()

asyncio.run(main())
"""

_FALLBACK_PROBE = _PROBE_PRELUDE + """
async def main():
    from augmentum.builds.verify import _run_gate_sidecar
    ws = os.environ["WS_ID"]
    docker, cm = await _cm_for(os.environ["WS_CONTAINER"])
    payload = {"url": "http://localhost:5199",
               "assertions": [{"id": "x", "steps": [], "assert": "true"}]}
    res = await _run_gate_sidecar(cm, ws, payload)
    # With the sidecar STOPPED the correct answer is None (fall to legacy).
    print(json.dumps({"ok": res is None, "result": res}))
    await docker.close()

asyncio.run(main())
"""

_COUNTER_PAGE = (
    "<!doctype html><title>LiveAccept</title>"
    "<button id=b onclick=\"this.textContent=(+this.textContent||0)+1\">0</button>"
)

_TODO_PAGE = """<!doctype html><html><head><title>Todo Live</title></head><body>
<h1>Todos</h1>
<input id="new-todo" placeholder="what to do">
<button id="add">Add</button>
<ul id="list"></ul>
<div id="count">0 items</div>
<script>
document.getElementById("add").addEventListener("click", () => {
  const inp = document.getElementById("new-todo");
  if (!inp.value.trim()) return;
  const li = document.createElement("li");
  li.textContent = inp.value;
  document.getElementById("list").appendChild(li);
  inp.value = "";
  document.getElementById("count").textContent =
    document.querySelectorAll("#list li").length + " items";
});
</script></body></html>"""


# ---------------------------------------------------------------------------
# Browser suite
# ---------------------------------------------------------------------------

@check("browser")
def sidecar_running(ctx) -> tuple[bool, str]:
    name = bh.sidecar_container()
    ctx["sidecar"] = name
    return (bool(name),
            f"sidecar container: {name}" if name
            else "no running container with label " + bh.SIDECAR_LABEL)


@check("browser", depends_on=("sidecar_running",))
def https_ca_trust(ctx) -> tuple[bool, str]:
    """Sidecar Chrome must trust caddy's local root CA: navigation to the
    Augmentum UI over https must succeed WITHOUT ignore-https-errors
    (locks in the CA-import entrypoint replacing the old
    AGENT_BROWSER_IGNORE_HTTPS_ERRORS=1 blanket bypass)."""
    name = ctx["sidecar"]
    session = "live-accept-ca"
    url = "https://host.docker.internal:6443/ui/"
    try:
        _, out = bh.docker("exec", name, "agent-browser", "--session", session,
                           "--json", "open", url, timeout=60.0)
        res = bh.last_json(out) or {}
        ok = bool(res.get("success"))
        err = str(res.get("error") or "")
        _, tout = bh.docker("exec", name, "agent-browser", "--session", session,
                            "--json", "get", "title")
        tres = bh.last_json(tout) or {}
        data = tres.get("data") if isinstance(tres.get("data"), dict) else {}
        title = str((data or {}).get("title") or "")
    finally:
        bh.docker("exec", name, "agent-browser", "--session", session, "close")
    passed = ok and "ERR_CERT" not in err and "ERR_CERT" not in title
    return passed, (f"open ok={ok}, title={title[:60]!r}"
                    + (f", error={err[:120]}" if err else ""))


@check("browser")
def workspace_create_slim(ctx) -> tuple[bool, str]:
    """Create a Browser/Test workspace via the real API: must be fast, on
    the prebaked image, and contain NO in-workspace Playwright."""
    sess: bh.BenchSession = ctx["session"]
    with bh.Timer() as t:
        status, body = sess.api("POST", "/api/coder/workspaces",
                                {"name": "live-accept",
                                 "tooling_profile": "browser"})
    if isinstance(body, dict) and body.get("id"):
        # Record for cleanup even if a later assertion fails.
        ctx["ws"] = body["id"]
        ctx["ws_container"] = bh.workspace_container_name(body["id"])
    if status not in (200, 201) or not ctx.get("ws"):
        return False, f"create failed: HTTP {status} {str(body)[:200]}"
    ws = ctx["ws"]
    code, image = bh.docker("inspect", ctx["ws_container"],
                            "--format", "{{.Config.Image}}")
    image = image.strip()
    _, pw = bh.workspace_shell(ws, "python3 -c 'import playwright' 2>&1 | tail -1")
    no_playwright = "ModuleNotFoundError" in pw
    ok = t.seconds < 30 and code == 0 and no_playwright
    return ok, (f"created in {t.seconds:.2f}s, image={image}, "
                f"in-workspace playwright: {'absent' if no_playwright else 'PRESENT'}")


@check("browser", depends_on=("workspace_create_slim",))
def ladder_engine_sidecar(ctx) -> tuple[bool, str]:
    """browser.py ladder must choose the sidecar and page state must
    persist across separate tool calls (the reopen-bug regression)."""
    ws = ctx["ws"]
    code, out = bh.workspace_shell(
        ws,
        f"printf %s '{_COUNTER_PAGE}' > /workspace/index.html && cd /workspace "
        f"&& (nohup python3 -m http.server 5199 >/dev/null 2>&1 &) && sleep 1")
    if code != 0:
        return False, f"page staging failed: {out[:200]}"
    _, out = bh.docker_py(bh.APP_CONTAINER, _LADDER_PROBE,
                          env={"WS_ID": ws, "WS_CONTAINER": ctx["ws_container"]})
    res = bh.last_json(out)
    if not res:
        return False, f"probe produced no JSON: {out[:300]}"
    engines = res.get("engines") or []
    ok = (res.get("ok") and all(e == "sidecar" for e in engines)
          and res.get("counter") == '"2"')
    return ok, (f"engines={engines}, counter={res.get('counter')}"
                + (f", error={res.get('error')}" if res.get("error") else ""))


@check("browser", needs_model=True, depends_on=("ladder_engine_sidecar",))
def gate_real_llm_binding(ctx) -> tuple[bool, str]:
    """builds behavior gate end-to-end with REAL LLM assertion binding —
    all behaviors pass on the sidecar engine, legacy script never staged."""
    ws = ctx["ws"]
    code, out = bh.workspace_shell(
        ws, "cat > /workspace/index.html <<'HTML'\n" + _TODO_PAGE + "\nHTML")
    if code != 0:
        return False, f"todo page staging failed: {out[:200]}"
    _, out = bh.docker_py(
        bh.APP_CONTAINER, _GATE_PROBE, timeout=600.0,
        env={"WS_ID": ws, "WS_CONTAINER": ctx["ws_container"],
             "BENCH_TOKEN": ctx["session"].token,
             "BENCH_MODEL": ctx["model"]})
    res = bh.last_json(out)
    if not res:
        return False, f"probe produced no JSON: {out[:300]}"
    summary = res.get("summary") or {}
    ok = (res.get("ok") and summary.get("all_passed")
          and not res.get("staged_legacy"))
    return ok, (f"gate_ran={res.get('ok')}, "
                f"passed={summary.get('passed')}/{summary.get('total')}, "
                f"legacy_staged={res.get('staged_legacy')}, "
                f"statuses={res.get('statuses')}")


@check("browser", depends_on=("ladder_engine_sidecar",))
def dual_session_isolation(ctx) -> tuple[bool, str]:
    """Concurrent ws-/gate- sessions on the same workspace must not
    share page state."""
    # Stage our own page — earlier checks may have replaced index.html.
    code, out = bh.workspace_shell(
        ctx["ws"], f"printf %s '{_COUNTER_PAGE}' > /workspace/index.html")
    if code != 0:
        return False, f"page staging failed: {out[:200]}"
    _, out = bh.docker_py(bh.APP_CONTAINER, _ISOLATION_PROBE,
                          env={"WS_ID": ctx["ws"],
                               "WS_CONTAINER": ctx["ws_container"]})
    res = bh.last_json(out)
    if not res:
        return False, f"probe produced no JSON: {out[:300]}"
    return bool(res.get("ok")), f"counters: a={res.get('a')} b={res.get('b')} (want 3/5)"


@check("browser", disruptive=True, depends_on=("ladder_engine_sidecar",))
def sidecar_fallback_drill(ctx) -> tuple[bool, str]:
    """Stop the sidecar: the gate runner must answer None (= fall to the
    legacy rung), and the sidecar must come back up after."""
    name = ctx.get("sidecar") or bh.sidecar_container()
    if not name:
        return False, "no sidecar to stop"
    bh.docker("stop", name, timeout=60.0)
    try:
        time.sleep(11)  # discovery cache TTL
        _, out = bh.docker_py(bh.APP_CONTAINER, _FALLBACK_PROBE,
                              env={"WS_ID": ctx["ws"],
                                   "WS_CONTAINER": ctx["ws_container"]})
        res = bh.last_json(out)
    finally:
        bh.docker("start", name, timeout=60.0)
    if not res:
        return False, f"probe produced no JSON: {out[:300]}"
    back = bool(bh.sidecar_container())
    return (bool(res.get("ok")) and back,
            f"gate fell through: {res.get('ok')}, sidecar restarted: {back}")


@check("browser", depends_on=("workspace_create_slim",))
def session_cleanup_on_delete(ctx) -> tuple[bool, str]:
    """Deleting the workspace must close its sidecar browser sessions.
    Runs LAST in the suite (it consumes the workspace)."""
    sess: bh.BenchSession = ctx["session"]
    ws = ctx["ws"]
    status, _body = sess.api("DELETE", f"/api/coder/workspaces/{ws}")
    ctx["ws_deleted"] = status == 200
    if status != 200:
        return False, f"delete failed: HTTP {status}"
    time.sleep(2)
    name = ctx.get("sidecar") or bh.sidecar_container()
    code, out = bh.docker("exec", name, "agent-browser", "session", "list", "--json")
    res = bh.last_json(out) or {}
    sessions = ((res.get("data") or {}).get("sessions")
                if isinstance(res.get("data"), dict) else res.get("sessions")) or []
    leaked = [s for s in sessions if ws[:8] in str(s)]
    return not leaked, f"workspace deleted, leaked sessions: {leaked or 'none'}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(suites: list[str], *, allow_disruption: bool, model: str,
        fmt: str) -> int:
    is_text = fmt == "text"
    selected = [c for c in CHECKS if not suites or c.suite in suites]
    if not selected:
        print(f"no checks match suites {suites}; known: "
              f"{sorted({c.suite for c in CHECKS})}")
        return 1

    results: list[Result] = []
    passed_names: set[str] = set()

    if not bh.stack_up():
        for c in selected:
            results.append(Result(c.name, "skip", "stack not running on "
                                  + bh.load_config()["base_url"]))
        return _report(results, fmt)

    # Headroom guard (2026-07-17 incident): live checks spin containers,
    # Chrome, and possibly a model — refuse when the host is already tight
    # so we never stack onto a user's model load and thrash the machine.
    free_gb = bh.host_free_memory_gb()
    min_gb = float(bh.load_config().get("min_free_memory_gb", 8))
    if free_gb is not None and free_gb < min_gb:
        for c in selected:
            results.append(Result(
                c.name, "skip",
                f"host memory headroom too low ({free_gb:.1f}GB free < "
                f"{min_gb:.0f}GB) — rerun when the machine is idle"))
        return _report(results, fmt)

    ctx: dict[str, Any] = {"model": model}
    try:
        ctx["session"] = bh.BenchSession().__enter__()
    except RuntimeError as exc:
        for c in selected:
            results.append(Result(c.name, "skip", str(exc)))
        return _report(results, fmt)

    try:
        for c in selected:
            if c.disruptive and not allow_disruption:
                results.append(Result(c.name, "skip",
                                      "disruptive — pass --allow-disruption"))
                continue
            if c.needs_model and not model:
                results.append(Result(
                    c.name, "skip",
                    "needs an LLM — set --model or 'model' in "
                    + bh.CONFIG_FILE.name + " (never auto-selected)"))
                continue
            missing = [d for d in c.depends_on if d not in passed_names]
            if missing:
                results.append(Result(c.name, "skip",
                                      f"dependency did not pass: {missing}"))
                continue
            if is_text:
                print(cyan(f"  -> {c.name}"))
            start = time.monotonic()
            try:
                ok, evidence = c.fn(ctx)
            except Exception as exc:  # noqa: BLE001 — a crashed check is a failed check
                ok, evidence = False, f"check crashed: {exc!r}"
            secs = time.monotonic() - start
            results.append(Result(c.name, "pass" if ok else "fail",
                                  evidence, secs))
            if ok:
                passed_names.add(c.name)
            if is_text:
                mark = green("PASS") if ok else red("FAIL")
                print(f"     {mark} ({secs:.1f}s) {evidence}")
    finally:
        # Belt-and-braces cleanup: workspace (if a check left it), token.
        try:
            if ctx.get("ws") and not ctx.get("ws_deleted"):
                ctx["session"].api("DELETE", f"/api/coder/workspaces/{ctx['ws']}")
        finally:
            ctx["session"].__exit__(None, None, None)

    return _report(results, fmt)


def _report(results: list[Result], fmt: str) -> int:
    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]
    passed = [r for r in results if r.status == "pass"]
    if fmt == "json":
        print(json.dumps({
            "metrics": {
                "live_checks_passed": len(passed),
                "live_checks_failed": len(failed),
                "live_checks_skipped": len(skipped),
            },
            "results": [r.__dict__ for r in results],
        }, indent=1))
    else:
        print()
        print(f"live acceptance: {green(str(len(passed)) + ' passed')}, "
              f"{red(str(len(failed)) + ' failed') if failed else '0 failed'}, "
              f"{yellow(str(len(skipped)) + ' skipped') if skipped else '0 skipped'}")
        for r in failed:
            print(red(f"  FAIL {r.name}: {r.evidence}"))
        for r in skipped:
            print(yellow(f"  skip {r.name}: {r.evidence}"))
        # audit.py-parseable metric lines
        print(f"live_checks_passed={len(passed)}")
        print(f"live_checks_failed={len(failed)}")
        print(f"live_checks_skipped={len(skipped)}")
    return len(failed)


def main() -> int:
    args = sys.argv[1:]
    if "--init" in args:
        path = bh.write_config_template()
        print(f"config: {path} (set 'model' there for LLM-dependent checks)")
        return 0
    if "--list" in args:
        for c in CHECKS:
            flags = []
            if c.disruptive:
                flags.append("disruptive")
            if c.needs_model:
                flags.append("needs-model")
            print(f"  {c.suite}/{c.name}"
                  + (f"  [{', '.join(flags)}]" if flags else ""))
        return 0
    suites = []
    if "--suite" in args:
        suites = [args[args.index("--suite") + 1]]
    model = bh.load_config().get("model") or ""
    if "--model" in args:
        model = args[args.index("--model") + 1]
    fmt = "json" if "--format=json" in args else "text"
    return run(suites, allow_disruption="--allow-disruption" in args,
               model=model, fmt=fmt)


if __name__ == "__main__":
    sys.exit(main())
