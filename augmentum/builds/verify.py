"""Behavior gate — run the spec-derived contract against the real app.

This is the "tested" half of outcome-based verification. Given the frozen
behavior contract (``builds/contract.py``) and a built workspace, it:

  1. binds each behavior INTENT to the actual DOM (one LLM call: behaviors +
     the app source -> concrete ``{steps, assert}`` specs);
  2. serves the workspace, launches headless chromium ONCE, and for each
     behavior drives the UI and evaluates a boolean assertion in the page;
  3. returns per-behavior ``pass`` / ``fail`` + evidence.

The check is run by the SYSTEM, in a real browser, and a failing assertion is
a real defect — not the agent's say-so. Selector binding adapts to the
implementation, but the behaviors themselves were frozen from the objective
before the build, so the agent can't weaken the contract.

Browser engine ladder (mirrors coder/browser.py): the shared agent-browser
sidecar service runs the gate first (dedicated ``gate-<workspace>`` session,
fresh page per behavior); when the sidecar isn't running, the legacy
in-container Playwright script covers pre-2026-07-17 workspaces that still
have Playwright installed. Reuses the ``run_shell`` sentinel-exit pattern
from coder mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from augmentum.builds.contract import extract_json_object
from augmentum.models.base import InternalChatRequest, Message, response_text
from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

GATE_DIR = "/workspace/.augmentum_gate"
GATE_PORT = 8129  # dedicated, isolated from whatever port the agent used
_SOURCE_BUDGET = 14_000  # chars of app source handed to the binding call


# ---------------------------------------------------------------------------
# Workspace shell runner (sentinel-exit pattern, mirrors coder _legacy)
# ---------------------------------------------------------------------------

def make_workspace_shell_runner(container_manager: Any, workspace_id: str):
    """Return ``async run_shell(cmd, timeout) -> (exit_code, output)`` bound to
    the workspace. Captures the exit code via a trailing sentinel line because
    ``_run_command`` only returns stdout."""
    sentinel = "__AUGMENTUM_GATE_EXIT__:"

    async def run_shell(cmd: str, timeout: float) -> tuple[int, str]:
        if container_manager is None:
            return 1, "no container manager"
        wrapped = f"({cmd}) 2>&1; printf '\\n{sentinel}%s\\n' \"$?\""
        try:
            output = await container_manager.run_command(
                workspace_id, ["bash", "-lc", wrapped], timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return 1, f"shell error: {exc}"
        lines = (output or "").splitlines()
        exit_code, body = 1, lines
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(sentinel):
                try:
                    exit_code = int(lines[i][len(sentinel):].strip())
                except ValueError:
                    exit_code = 1
                body = lines[:i]
                break
        return exit_code, "\n".join(body)

    return run_shell


# ---------------------------------------------------------------------------
# Assertion binding (behaviors + app source -> concrete steps/assert specs)
# ---------------------------------------------------------------------------

_BIND_SYSTEM = (
    "You are a browser-test author. You translate acceptance criteria into "
    "concrete Playwright-style interaction steps and a JavaScript boolean "
    "assertion, using the ACTUAL selectors present in the app's source."
)


def _bind_prompt(behaviors: list[dict], source: str) -> str:
    blist = "\n".join(f'  - id "{b["id"]}": {b["description"]}' for b in behaviors)
    return (
        "App source (HTML/JS/CSS), possibly truncated:\n"
        "-----\n" + source[:_SOURCE_BUDGET] + "\n-----\n\n"
        "Behaviors to turn into runnable checks:\n" + blist + "\n\n"
        "For EACH behavior, produce: a list of setup `steps` that drive the UI, "
        "and one `assert` — a JavaScript boolean expression evaluated in the "
        "page that is true IFF the behavior holds. Rules:\n"
        "- Use selectors that actually exist in the source above (ids, classes, "
        "names). Prefer ids.\n"
        "- step actions: {\"action\":\"fill\",\"selector\":\"#x\",\"value\":\"100\"}, "
        "{\"action\":\"click\",\"selector\":\"#y\"}, "
        "{\"action\":\"select\",\"selector\":\"#z\",\"value\":\"2024\"}, "
        "{\"action\":\"wait\",\"ms\":150}.\n"
        "- The `assert` is a single JS expression (no statements, no return). "
        "Be tolerant: compare with .includes() / parseFloat over exact string "
        "equality where reasonable. Example: "
        "\"document.querySelector('#total').textContent.includes('115')\".\n"
        "- If a behavior genuinely cannot be checked from the DOM, omit it.\n\n"
        "Return ONLY JSON, no prose, no fence:\n"
        '{"assertions":[{"id":"<behavior-id>","steps":[...],"assert":"<js>"}]}'
    )


async def compile_assertions(
    backend: Any, *, model: str, behaviors: list[dict], source: str,
) -> dict[str, dict]:
    """Bind behavior intents to concrete ``{steps, assert}`` specs, keyed by
    behavior id. Returns {} on failure (caller marks behaviors untestable)."""
    if not behaviors:
        return {}
    req = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_BIND_SYSTEM),
            Message(role="user", content=_bind_prompt(behaviors, source)),
        ],
        temperature=0.1,
        chat_template_kwargs={"enable_thinking": False},
    )
    try:
        resp = await backend.chat(req)
    except Exception:  # noqa: BLE001
        log.warning("build_gate.bind_failed", exc_info=True)
        return {}
    obj = extract_json_object(response_text(resp))
    out: dict[str, dict] = {}
    for a in (obj.get("assertions") or []):
        if not isinstance(a, dict):
            continue
        bid = str(a.get("id") or "").strip()
        assertion = str(a.get("assert") or "").strip()
        if not bid or not assertion:
            continue
        steps = a.get("steps") if isinstance(a.get("steps"), list) else []
        out[bid] = {"steps": steps, "assert": assertion}
    return out


# ---------------------------------------------------------------------------
# The in-container batch gate script
# ---------------------------------------------------------------------------

# One chromium launch, a fresh page per behavior (no state bleed). Always exits
# 0 with a JSON object on stdout so the shell runner can parse it; per-behavior
# failures live inside the JSON, not in the exit code.
_GATE_SCRIPT = r'''
import json, sys
try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print(json.dumps({"fatal": "playwright unavailable: " + str(e)})); sys.exit(0)
data = json.load(open("__ASSERTIONS_PATH__"))
url = data["url"]; items = data["assertions"]
out = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=__CHROME_ARGS__)
        for it in items:
            res = {"id": it.get("id", ""), "passed": False, "evidence": ""}
            errs = []
            try:
                page = browser.new_page()
                page.on("console", lambda m, e=errs: e.append(m.text) if m.type == "error" else None)
                page.goto(url, wait_until="networkidle", timeout=15000)
                for s in (it.get("steps") or []):
                    a = s.get("action")
                    try:
                        if a == "fill": page.fill(s["selector"], str(s.get("value", "")), timeout=4000)
                        elif a == "click": page.click(s["selector"], timeout=4000)
                        elif a == "select": page.select_option(s["selector"], str(s.get("value", "")), timeout=4000)
                        elif a == "wait": page.wait_for_timeout(int(s.get("ms", 200)))
                    except Exception as se:
                        res["evidence"] = "step failed (" + str(a) + " " + str(s.get("selector", "")) + "): " + str(se)[:120]
                        raise
                page.wait_for_timeout(150)
                val = page.evaluate("(()=>{try{return !!(" + it["assert"] + ");}catch(e){return {__e:String(e)};}})()")
                if isinstance(val, dict):
                    res["evidence"] = "assert error: " + str(val.get("__e"))[:140]
                else:
                    res["passed"] = bool(val)
                    res["evidence"] = "ok" if val else "assertion was false"
                if errs:
                    res["console_errors"] = errs[:3]
                    if res["passed"]:
                        res["evidence"] = "ok (console errors: " + "; ".join(errs[:2])[:100] + ")"
                page.close()
            except Exception as e:
                if not res["evidence"]:
                    res["evidence"] = "run error: " + str(e)[:140]
            out.append(res)
        browser.close()
except Exception as e:
    print(json.dumps({"fatal": "browser launch failed: " + str(e)})); sys.exit(0)
print(json.dumps({"results": out}))
'''


def build_gate_payload(assertions: dict[str, dict], *, port: int = GATE_PORT) -> dict:
    """Assemble the assertions.json payload the gate script reads."""
    return {
        "url": f"http://localhost:{port}",
        "assertions": [{"id": bid, **spec} for bid, spec in assertions.items()],
    }


def parse_gate_output(raw: str) -> dict[str, dict] | None:
    """Parse the gate script's stdout. Returns {behavior_id: result} or None on
    a fatal (playwright/browser unavailable) so the caller can fall back."""
    obj = extract_json_object(raw)
    if not obj or obj.get("fatal"):
        if obj.get("fatal"):
            log.warning("build_gate.fatal", reason=obj.get("fatal"))
        return None
    results = obj.get("results")
    if not isinstance(results, list):
        return None
    return {str(r.get("id")): r for r in results if isinstance(r, dict) and r.get("id")}


# ---------------------------------------------------------------------------
# Sidecar gate runner — same semantics as the in-container script: one
# browser, a FRESH page per behavior (unconditional open resets state so
# behaviors can't bleed into each other), always returns per-behavior
# results; infra failure returns None so the caller falls to the legacy rung.
# ---------------------------------------------------------------------------

_ASSERT_WRAPPER = "(()=>{{try{{return !!({expr});}}catch(e){{return {{__e:String(e)}};}}}})()"


async def _run_gate_sidecar(
    container_manager: Any, workspace_id: str, payload: dict,
) -> dict[str, dict] | None:
    """Drive the behavior gate through the browser sidecar.

    Returns ``{behavior_id: {id, passed, evidence, console_errors?}}`` —
    the exact shape ``parse_gate_output`` produces — or None when the
    sidecar is unavailable / dies mid-run (caller falls back to the
    legacy in-container Playwright script).
    """
    from augmentum.coder import browser_sidecar as bs

    docker = getattr(container_manager, "_docker", None)
    if docker is None or not await bs.is_available(docker):
        return None
    # Dedicated session: never clobbers the coder agent's page state.
    session = bs.session_for_workspace(workspace_id, prefix="gate")
    url = await bs.reachable_url(container_manager, workspace_id, payload["url"])
    results: dict[str, dict] = {}
    try:
        for it in payload.get("assertions") or []:
            bid = str(it.get("id") or "")
            res = {"id": bid, "passed": False, "evidence": ""}
            # Fresh page per behavior — deliberate state reset.
            opened = await bs.run_cli(docker, ["open", url], session=session, timeout=30.0)
            if not opened.get("sidecar", True):
                return None  # sidecar died mid-run — fall back whole-gate
            await bs.run_cli(docker, ["console", "clear"], session=session, timeout=10.0)
            if not opened.get("ok"):
                res["evidence"] = f"open failed: {str(opened.get('error'))[:140]}"
                results[bid] = res
                continue
            step_failed = False
            for s in it.get("steps") or []:
                a = s.get("action")
                sel = str(s.get("selector") or "")
                val = str(s.get("value", ""))
                if a == "fill":
                    r = await bs.run_cli(docker, ["fill", sel, val], session=session, timeout=10.0)
                elif a == "click":
                    r = await bs.run_cli(docker, ["click", sel], session=session, timeout=10.0)
                elif a == "select":
                    r = await bs.run_cli(docker, ["select", sel, val], session=session, timeout=10.0)
                elif a == "wait":
                    await asyncio.sleep(min(5.0, int(s.get("ms", 200)) / 1000.0))
                    continue
                else:
                    continue
                if not r.get("sidecar", True):
                    return None
                if not r.get("ok"):
                    res["evidence"] = (
                        f"step failed ({a} {sel}): {str(r.get('error'))[:120]}"
                    )
                    step_failed = True
                    break
            if not step_failed:
                await asyncio.sleep(0.15)
                code = _ASSERT_WRAPPER.format(expr=it.get("assert") or "false")
                ev = await bs.run_cli(docker, ["eval", code], session=session, timeout=15.0)
                if not ev.get("sidecar", True):
                    return None
                val = ev.get("result") if "result" in ev else ev.get("value")
                if not ev.get("ok"):
                    res["evidence"] = f"assert error: {str(ev.get('error'))[:140]}"
                elif isinstance(val, dict):
                    res["evidence"] = "assert error: " + str(val.get("__e"))[:140]
                else:
                    res["passed"] = bool(val)
                    res["evidence"] = "ok" if val else "assertion was false"
                errs = [e["text"] for e in await bs._console_errors(docker, session)
                        if e.get("type") == "error"]
                if errs:
                    res["console_errors"] = errs[:3]
                    if res["passed"]:
                        res["evidence"] = (
                            "ok (console errors: " + "; ".join(errs[:2])[:100] + ")"
                        )
            results[bid] = res
    finally:
        try:
            await bs.run_cli(docker, ["close"], session=session, timeout=10.0)
        except Exception:  # noqa: BLE001
            log.warning("build_gate.sidecar_session_close_failed", exc_info=True)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _read_source(run_shell) -> str:
    """Concatenate the app's text source for the binding call."""
    cmd = (
        "for f in /workspace/index.html /workspace/*.html /workspace/*.js "
        "/workspace/*.css /workspace/js/*.js /workspace/src/*.js; do "
        "[ -f \"$f\" ] && { echo \"===== $f =====\"; cat \"$f\"; }; done 2>/dev/null"
    )
    _code, out = await run_shell(cmd, 20.0)
    return out or ""


async def run_behavior_gate(
    *,
    container_manager: Any,
    workspace_id: str,
    backend: Any,
    model: str,
    behaviors: list[dict],
    port: int = GATE_PORT,
) -> tuple[list[dict], bool]:
    """Run the contract against the live app.

    Returns ``(behaviors, gate_ran)``: each behavior dict gets its ``status``
    (``pass`` / ``fail`` / ``untested``) + ``evidence`` updated in place-ish
    (a new list is returned). ``gate_ran`` is False when the gate could not run
    at all (no behaviors, browser/playwright unavailable) — the caller then
    falls back to the trail-based verdict instead of failing the build.
    """
    if not behaviors or container_manager is None or not workspace_id:
        return behaviors, False

    run_shell = make_workspace_shell_runner(container_manager, workspace_id)

    # 1. Bind intents -> concrete asserts against the real source.
    source = await _read_source(run_shell)
    assertions = await compile_assertions(
        backend, model=model, behaviors=behaviors, source=source,
    )
    if not assertions:
        log.info("build_gate.no_assertions", behaviors=len(behaviors))
        return behaviors, False

    # 2. Serve the workspace on the dedicated gate port.
    payload = build_gate_payload(assertions, port=port)
    await run_shell(f"pkill -f 'http.server {port}' 2>/dev/null; sleep 0.2", 8.0)
    await run_shell(
        f"cd /workspace && (nohup python3 -m http.server {port} "
        f">/dev/null 2>&1 &) ; sleep 1", 12.0,
    )

    # 3. Engine ladder: sidecar first; legacy in-container Playwright script
    #    for pre-sidecar workspaces that still have it installed.
    try:
        results = await _run_gate_sidecar(container_manager, workspace_id, payload)
        if results is None:
            script = _GATE_SCRIPT.replace(
                "__ASSERTIONS_PATH__", f"{GATE_DIR}/assertions.json"
            ).replace("__CHROME_ARGS__", repr(list(HEADLESS_WEBGL_ARGS)))
            try:
                await container_manager.file_write(
                    workspace_id, f"{GATE_DIR}/assertions.json", json.dumps(payload))
                await container_manager.file_write(
                    workspace_id, f"{GATE_DIR}/gate.py", script)
            except Exception:  # noqa: BLE001
                log.warning("build_gate.stage_failed", exc_info=True)
                return behaviors, False
            _code, raw = await run_shell(f"python3 {GATE_DIR}/gate.py", 120.0)
            results = parse_gate_output(raw)
        else:
            log.info("build_gate.engine", engine="sidecar",
                     behaviors=len(payload.get("assertions") or []))
    finally:
        await run_shell(f"pkill -f 'http.server {port}' 2>/dev/null", 8.0)

    if results is None:
        return behaviors, False  # fatal (no sidecar AND no playwright) -> fall back

    # 4. Merge results back into the behavior list.
    merged: list[dict] = []
    for b in behaviors:
        r = results.get(b["id"])
        nb = dict(b)
        if r is None:
            nb["status"] = "untested"
            nb["evidence"] = nb.get("evidence") or "no assertion was compiled for this behavior"
        else:
            nb["status"] = "pass" if r.get("passed") else "fail"
            nb["evidence"] = (r.get("evidence") or "")[:240]
        merged.append(nb)
    return merged, True


def gate_summary(behaviors: list[dict]) -> dict[str, Any]:
    """Reduce a checked behavior list to counts + the failing ids."""
    passed = [b for b in behaviors if b.get("status") == "pass"]
    failed = [b for b in behaviors if b.get("status") == "fail"]
    checked = passed + failed
    return {
        "total": len(behaviors),
        "checked": len(checked),
        "passed": len(passed),
        "failed": len(failed),
        "all_passed": bool(checked) and not failed,
        "failed_ids": [b["id"] for b in failed],
    }
