"""Browser-style verification helpers for Coder workspaces.

Engine ladder per helper: the shared agent-browser sidecar service
(compose.browser.yaml — persistent sessions, real Chrome) → legacy
in-workspace Playwright (pre-2026-07-17 workspaces that still have it
installed; new workspaces don't) → plain HTTP, so the tools stay useful
even with no browser at all.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import shlex
import time
from typing import Any

from augmentum.coder import browser_sidecar as _sidecar
from augmentum.coder.services import _container_reachable_url
from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _try_sidecar(coro_factory, cm) -> dict[str, Any] | None:
    """Sidecar-first rung of the engine ladder (sidecar → in-workspace
    Playwright → HTTP). Returns None to fall through to the legacy path:
    when the sidecar service isn't running, or when the call failed for
    infrastructure reasons (exec/discovery) rather than page reasons.
    Page-level failures ARE the answer — no double-run."""
    try:
        if not await _sidecar.is_available(cm._docker):
            return None
        result = await coro_factory()
    except Exception:
        log.warning("browser_sidecar_call_failed_falling_back", exc_info=True)
        return None
    if result.get("engine") == "sidecar":
        return result
    return None

_SESSION_PATH = "/workspace/.augmentum/browser-session.json"
_SCREENSHOT_DIR = "/workspace/.augmentum/browser-screenshots"
_DEFAULT_PORTS = (5173, 3000, 8000, 8080)


async def save_browser_session(cm, workspace_id: str, url: str) -> None:
    payload = json.dumps({"url": url, "updated_at": time.time()}, sort_keys=True)
    await cm.run_command(
        workspace_id,
        ["bash", "-lc", "mkdir -p /workspace/.augmentum"],
        timeout=3.0,
    )
    await cm.file_write(workspace_id, _SESSION_PATH, payload)


async def load_browser_session(cm, workspace_id: str) -> str:
    try:
        raw = await cm.file_read(workspace_id, _SESSION_PATH)
        data = json.loads(raw or "{}")
        return str(data.get("url") or "")
    except Exception:
        return ""


async def infer_preview_url(cm, workspace_id: str) -> str:
    try:
        ports = await cm.list_ports(workspace_id)
    except Exception:
        ports = []
    ready = [
        int(p.get("container_port") or 0)
        for p in ports
        if p.get("listening") and int(p.get("container_port") or 0) in _DEFAULT_PORTS
    ]
    if ready:
        return f"/api/coder/preview/{workspace_id}/{ready[0]}/"
    return ""


async def http_snapshot(cm, workspace_id: str, url: str, *, timeout: float = 8.0) -> dict[str, Any]:
    target = _container_reachable_url(url, workspace_id)
    script = (
        "import json,re,time,urllib.request\n"
        f"url={target!r}; timeout={float(timeout)!r}; start=time.time()\n"
        "try:\n"
        "    req=urllib.request.Request(url, headers={'User-Agent':'Augmentum-Coder-Browser'})\n"
        "    with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
        "        raw=resp.read(200000).decode('utf-8','replace')\n"
        "        status=getattr(resp,'status',0) or resp.getcode(); final=resp.geturl(); err=''\n"
        "except Exception as exc:\n"
        "    raw=''; status=0; final=url; err=str(exc)\n"
        "title=''\n"
        "m=re.search(r'<title[^>]*>(.*?)</title>', raw, re.I|re.S)\n"
        "if m: title=re.sub(r'\\s+',' ',m.group(1)).strip()[:200]\n"
        "texts=[]\n"
        "for tag in ('h1','h2','button','a','label','input'):\n"
        "    for mm in re.finditer(r'<%s\\b[^>]*>(.*?)</%s>' % (tag, tag), raw, re.I|re.S):\n"
        "        txt=re.sub(r'<[^>]+>',' ',mm.group(1)); txt=re.sub(r'\\s+',' ',txt).strip()\n"
        "        if txt: texts.append({'tag':tag,'text':txt[:160]})\n"
        "inputs=re.findall(r'<(?:input|textarea|select)\\b[^>]*(?:id|name|aria-label)=[\"\\']([^\"\\']+)', raw, re.I)\n"
        "buttons=re.findall(r'<button\\b[^>]*(?:id|class|aria-label)=[\"\\']([^\"\\']+)', raw, re.I)\n"
        "print(json.dumps({'url':url,'reachable_url':final,'status':status,"
        "'ok':200 <= int(status or 0) < 400,'title':title,"
        "'summary':texts[:30],'inputs':inputs[:30],'buttons':buttons[:30],"
        "'console_errors':[],'network_failures':[],'error':err,"
        "'fallback':'http','latency_ms':int((time.time()-start)*1000)}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=timeout + 4.0,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {
            "url": target,
            "status": 0,
            "ok": False,
            "title": "",
            "summary": [],
            "console_errors": [],
            "network_failures": [],
            "error": out,
            "fallback": "http",
        }


async def playwright_action(
    cm,
    workspace_id: str,
    *,
    url: str,
    action: str,
    selector: str = "",
    text: str = "",
    viewport: dict[str, int] | None = None,
    wait_for_selector: str = "",
) -> dict[str, Any]:
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.action(
            cm, workspace_id, url=url, action=action, selector=selector,
            text=text, wait_for_selector=wait_for_selector,
        ),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    target = _container_reachable_url(url, workspace_id)
    viewport = viewport or {"width": 1280, "height": 800}
    # Console + network listeners: register BEFORE page.goto so we
    # don't miss errors fired during the initial load. ``console_errors``
    # only collects level=error / warning (info/log/debug are noise);
    # ``network_failures`` collects requestfailed + responses with HTTP
    # 4xx/5xx (the agent cares about both — a broken API call AND a
    # 404 image both signal something to fix).
    script = (
        "import json,sys,time\n"
        f"url={target!r}; action={action!r}; selector={selector!r}; text={text!r}; "
        f"viewport={viewport!r}; wait_for_selector={wait_for_selector!r}\n"
        f"chrome_args={list(HEADLESS_WEBGL_ARGS)!r}\n"
        "start=time.time()\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':False,'error':str(exc)})); raise SystemExit(0)\n"
        "console_errors=[]; network_failures=[]\n"
        "def _on_console(msg):\n"
        "    if msg.type in ('error','warning'):\n"
        "        console_errors.append({'type':msg.type,'text':msg.text[:300]})\n"
        "def _on_request_failed(req):\n"
        "    network_failures.append({'url':req.url[:200],'method':req.method,"
        "'failure':(req.failure or '')[:200]})\n"
        "def _on_response(resp):\n"
        "    if resp.status >= 400:\n"
        "        network_failures.append({'url':resp.url[:200],'method':resp.request.method,"
        "'status':resp.status})\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        browser=p.chromium.launch(headless=True, args=chrome_args)\n"
        "        page=browser.new_page(viewport=viewport)\n"
        "        page.on('console', _on_console)\n"
        "        page.on('requestfailed', _on_request_failed)\n"
        "        page.on('response', _on_response)\n"
        # 'load', NOT 'networkidle': a live Vite/HMR dev server (or any page with
        # a persistent websocket / polling) never reaches network idle, so
        # 'networkidle' hangs to timeout. 'load' fires reliably and Playwright's
        # per-action auto-wait handles element readiness after.
        "        resp=page.goto(url, wait_until='load', timeout=15000)\n"
        "        if wait_for_selector:\n"
        "            page.wait_for_selector(wait_for_selector, timeout=10000)\n"
        "        if action == 'click': page.click(selector, timeout=5000)\n"
        "        elif action == 'type': page.fill(selector, text, timeout=5000)\n"
        "        title=page.title(); body=page.locator('body').inner_text(timeout=5000)[:2000]\n"
        "        browser.close()\n"
        "        status=resp.status if resp else 0\n"
        "        print(json.dumps({'ok':True,'playwright':True,'status':status,"
        "'title':title,'body_preview':body,'latency_ms':int((time.time()-start)*1000),"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':True,'error':str(exc),"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=25.0,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "playwright": False, "error": out}


def _is_local_preview_url(url: str) -> bool:
    """A URL that resolves to the workspace's own dev server / preview — i.e.
    what the user has open in the live preview. We only substitute a live GPU
    capture for these, never for an external site the model navigated to."""
    u = (url or "").lower()
    return (
        "localhost" in u
        or "127.0.0.1" in u
        or "://0.0.0.0" in u
        or "/api/coder/preview/" in u
    )


async def _try_live_preview_capture(
    cm, workspace_id: str, *, url: str, path: str, timeout: float = 8.0
) -> dict[str, Any] | None:
    """When the user's coder preview is open, capture the frame their real GPU
    already rendered (via the preview-capture broker/WS) and land it at ``path``
    so the normal screenshot vision feed picks it up — instead of re-rendering a
    heavy WebGL page in the GPU-less headless workspace (6-45s+ or a timeout).

    Best-effort at every step: not a local preview URL, no live socket, a
    slow/absent frame, a non-canvas page, or an oversized image all return None
    so the caller falls through to the (graceful) headless path.
    """
    if not _is_local_preview_url(url):
        return None
    try:
        from augmentum.coder.preview_capture import broker
    except Exception:
        return None
    if not broker.is_connected(workspace_id):
        return None
    try:
        result = await broker.capture(workspace_id, url=url, timeout=timeout)
    except Exception:
        return None
    if not result:
        return None
    data_url = str(result.get("data_url") or "")
    if not data_url.startswith("data:image") or "," not in data_url:
        return None
    b64 = data_url.split(",", 1)[1]
    if not b64:
        return None
    import base64 as _b64
    try:
        png = _b64.b64decode(b64)
    except Exception:
        return None
    # Sanity cap — a screenshot PNG is rarely more than a couple MB; a giant
    # canvas falls back to headless rather than shipping a huge blob.
    if not png or len(png) > 12_000_000:
        return None
    try:
        await cm.file_write_bytes(workspace_id, path, png)
    except Exception:
        return None
    return {
        "ok": True,
        "playwright": False,
        "source": "live_preview",
        "path": path,
        "title": "",
        "status": 200,
        "url": url,
        "wait_until": "live",
        "full_page": False,
        "requested_full_page": False,
        "degraded": False,
        "warnings": [],
        "width": result.get("width"),
        "height": result.get("height"),
        "phase_ms": {},
        "latency_ms": 0,
        "console_errors": [],
        "network_failures": [],
        "note": "captured from the live GPU preview (your browser) — not headless",
    }


async def playwright_screenshot(
    cm,
    workspace_id: str,
    *,
    url: str,
    viewport: dict[str, int] | None = None,
    wait_for_selector: str = "",
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 15_000,
    full_page: bool = True,
    settle_ms: int = 250,
) -> dict[str, Any]:
    target = _container_reachable_url(url, workspace_id)
    viewport = viewport or {"width": 1280, "height": 800}
    if wait_until not in {"domcontentloaded", "load", "networkidle"}:
        wait_until = "domcontentloaded"
    try:
        timeout_ms = int(timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = 15_000
    timeout_ms = max(1_000, min(60_000, timeout_ms))
    try:
        settle_ms = int(settle_ms)
    except (TypeError, ValueError):
        settle_ms = 250
    settle_ms = max(0, min(5_000, settle_ms))

    launch_timeout_ms = min(10_000, max(5_000, timeout_ms))
    # Cap goto INDEPENDENTLY of timeout_ms: a model may pass a large timeout_ms
    # hoping for a slow capture, but domcontentloaded/load on a local dev server
    # is ~1s — a 60s goto budget just eats the subprocess wall for nothing.
    goto_timeout_ms = min(timeout_ms, 20_000)
    selector_timeout_ms = min(timeout_ms, 10_000) if wait_for_selector else 0
    # Heavy WebGL software renders (SDF ray-march, large Three.js) swing 6-45s+
    # with HOST LOAD, not mainly with canvas size (measured: the same 1280x800
    # page took 11s idle and >45s under coder load; shrinking the viewport did
    # NOT reliably help, and a same-page retry after a timeout is poisoned).
    # So: one BOUNDED generous attempt that covers the common range and stays
    # under the subprocess wall, then degrade GRACEFULLY with a clear message —
    # never a hard subprocess kill (what a cranked-up timeout_ms used to cause).
    # A reliable frame of a heavy WebGL page comes from the live GPU preview,
    # not more headless patience.
    screenshot_timeout_ms = min(35_000, max(25_000, timeout_ms))
    fallback_timeout_ms = 12_000
    networkidle_grace_ms = 0 if wait_until == "networkidle" else min(1_500, timeout_ms)
    subprocess_timeout = min(
        92.0,
        max(
            30.0,
            (
                launch_timeout_ms
                + goto_timeout_ms
                + selector_timeout_ms
                + screenshot_timeout_ms
                + (fallback_timeout_ms if full_page else 0)
                + networkidle_grace_ms
                + settle_ms
            ) / 1000.0 + 8.0,
        ),
    )
    path = f"{_SCREENSHOT_DIR}/shot_{int(time.time())}.png"
    # Prefer the user's live GPU preview when it's open: a heavy WebGL page
    # renders there in a few seconds vs 6-45s+ (or a timeout) in the headless
    # GPU-less workspace. Falls through to the headless capture below when no
    # preview is connected, the frame doesn't arrive in time, or there's no
    # canvas to grab.
    live = await _try_live_preview_capture(cm, workspace_id, url=target, path=path, timeout=8.0)
    if live is not None:
        return live
    # Second preference: the persistent sidecar browser (compose.browser.
    # yaml) — real Chrome, warm daemon, no per-call cold start. Falls
    # through to the legacy in-workspace headless path when not running.
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.screenshot(
            cm, workspace_id, url=url, full_page=full_page,
            wait_for_selector=wait_for_selector,
        ),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    # Capture is best-effort. Pages with WebSockets, HMR, WebGL loops, or
    # slow third-party assets often never reach networkidle; design iteration
    # still needs a visual artifact plus clear diagnostics about degradation.
    script = (
        "import json,time\n"
        f"url={target!r}; viewport={viewport!r}; path={path!r}; "
        f"wait_for_selector={wait_for_selector!r}; wait_until={wait_until!r}; "
        f"launch_timeout_ms={launch_timeout_ms!r}; goto_timeout_ms={goto_timeout_ms!r}; "
        f"selector_timeout_ms={selector_timeout_ms!r}; screenshot_timeout_ms={screenshot_timeout_ms!r}; "
        f"fallback_timeout_ms={fallback_timeout_ms!r}; full_page={bool(full_page)!r}; "
        f"settle_ms={settle_ms!r}; networkidle_grace_ms={networkidle_grace_ms!r}\n"
        f"chrome_args={list(HEADLESS_WEBGL_ARGS)!r}\n"
        "start=time.time(); phases={}; warnings=[]; console_errors=[]; network_failures=[]\n"
        "status=0; title=''; captured=False; captured_full_page=False; final_error=''\n"
        "def _elapsed_ms(t): return int((time.time()-t)*1000)\n"
        "def _warn(phase, exc): warnings.append({'phase':phase,'error':str(exc)[:500]})\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':False,'error':str(exc)})); raise SystemExit(0)\n"
        "def _on_console(msg):\n"
        "    if msg.type in ('error','warning'):\n"
        "        console_errors.append({'type':msg.type,'text':msg.text[:300]})\n"
        "def _on_request_failed(req):\n"
        "    network_failures.append({'url':req.url[:200],'method':req.method,"
        "'failure':(req.failure or '')[:200]})\n"
        "def _on_response(resp):\n"
        "    if resp.status >= 400:\n"
        "        network_failures.append({'url':resp.url[:200],'method':resp.request.method,"
        "'status':resp.status})\n"
        "browser=None\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        t=time.time(); browser=p.chromium.launch(headless=True, args=chrome_args, timeout=launch_timeout_ms); phases['launch_ms']=_elapsed_ms(t)\n"
        "        page=browser.new_page(viewport=viewport)\n"
        "        page.set_default_timeout(min(goto_timeout_ms, 10000))\n"
        "        page.on('console', _on_console)\n"
        "        page.on('requestfailed', _on_request_failed)\n"
        "        page.on('response', _on_response)\n"
        "        t=time.time()\n"
        "        try:\n"
        "            resp=page.goto(url, wait_until=wait_until, timeout=goto_timeout_ms)\n"
        "            status=resp.status if resp else 0\n"
        "        except Exception as exc:\n"
        "            _warn('goto_' + wait_until, exc)\n"
        "            final_error=str(exc)[:500]\n"
        "        phases['goto_ms']=_elapsed_ms(t)\n"
        "        if wait_for_selector:\n"
        "            t=time.time()\n"
        "            try:\n"
        "                page.wait_for_selector(wait_for_selector, timeout=selector_timeout_ms)\n"
        "            except Exception as exc:\n"
        "                _warn('wait_for_selector', exc)\n"
        "            phases['selector_ms']=_elapsed_ms(t)\n"
        "        if settle_ms:\n"
        "            page.wait_for_timeout(settle_ms)\n"
        "        try:\n"
        "            page.wait_for_function(\"() => !document.fonts || document.fonts.status === 'loaded'\", timeout=1000)\n"
        "        except Exception:\n"
        "            pass\n"
        "        if networkidle_grace_ms:\n"
        "            try:\n"
        "                page.wait_for_load_state('networkidle', timeout=networkidle_grace_ms)\n"
        "                phases['networkidle_grace']='ready'\n"
        "            except Exception:\n"
        "                phases['networkidle_grace']='timed_out'\n"
        "        t=time.time()\n"
        "        try:\n"
        "            page.screenshot(path=path, full_page=full_page, timeout=screenshot_timeout_ms)\n"
        "            captured=True; captured_full_page=bool(full_page)\n"
        "        except Exception as exc:\n"
        "            _warn('screenshot_full_page' if full_page else 'screenshot', exc)\n"
        "            final_error=str(exc)[:500]\n"
        "            if full_page:\n"
        "                try:\n"
        "                    page.screenshot(path=path, full_page=False, timeout=fallback_timeout_ms)\n"
        "                    captured=True; captured_full_page=False\n"
        "                    warnings.append({'phase':'screenshot_fallback','error':'full-page screenshot failed; viewport screenshot captured'})\n"
        "                except Exception as exc2:\n"
        "                    _warn('screenshot_viewport', exc2)\n"
        "                    final_error=str(exc2)[:500]\n"
        "        phases['screenshot_ms']=_elapsed_ms(t)\n"
        # On a timeout with no capture, hand the model a clear, actionable
        # reason instead of a bare Playwright TimeoutError — heavy WebGL is
        # slow to software-render headless; the live GPU preview is reliable.
        "        if not captured and 'imeout' in (final_error or ''):\n"
        "            final_error=('screenshot timed out after %dms - this looks like a heavy WebGL/canvas page, which renders slowly headless (no GPU). '%screenshot_timeout_ms) + 'It renders fine in the live GPU preview; try a smaller width/height, or view the preview directly.'\n"
        "        try:\n"
        "            title=page.title()\n"
        "        except Exception:\n"
        "            title=''\n"
        "        browser.close(); browser=None\n"
        "        print(json.dumps({'ok':captured,'playwright':True,'path':path,'title':title,"
        "'status':status,'url':url,'wait_until':wait_until,'full_page':captured_full_page,"
        "'requested_full_page':bool(full_page),'degraded':bool(warnings),'warnings':warnings[:10],"
        "'phase_ms':phases,'latency_ms':int((time.time()-start)*1000),"
        "'error':'' if captured else final_error,"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
        "except Exception as exc:\n"
        "    try:\n"
        "        browser.close() if browser else None\n"
        "    except Exception:\n"
        "        pass\n"
        "    print(json.dumps({'ok':False,'playwright':True,'error':str(exc),'path':path,"
        "'phase_ms':phases,'warnings':warnings[:10],"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
    )
    await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"mkdir -p {shlex.quote(_SCREENSHOT_DIR)}"],
        timeout=3.0,
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=subprocess_timeout,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "playwright": False, "error": out, "path": path}


def _build_evaluate_wrapper(user_expression: str, *, with_element: bool) -> str:
    """Build the JS wrapper that runs the user expression and returns a
    structured envelope.

    Always emits an arrow function so Playwright's ``page.evaluate(fn, arg)``
    path is exercised — that's what makes ``arg`` reach the page. The wrapper
    catches JS exceptions and surfaces ``{message, name, stack, line, column}``
    instead of letting Playwright bubble a generic Error string.

    Two shapes:

    - page-scoped: ``(arg) => ...``. ``arg`` is bound to the user-passed args.
    - element-scoped: ``(el, arg) => ...``. ``el`` is the matched locator's
      element handle; ``arg`` is the user-passed args.

    The user expression can be:

    - A value (``document.title``) — captured as-is.
    - A function (``() => x``, ``(arg) => x``, ``(el, arg) => x``) — called
      with the appropriate args. The wrapper awaits the result, so async
      functions work too.

    A JS syntax error in the user expression breaks the wrapper itself —
    Playwright raises a parse error which surfaces via the outer Python
    except. JS runtime errors are caught inside the wrapper and routed
    through the structured ``error`` envelope.
    """
    sig = "(el, arg)" if with_element else "(arg)"
    call_args = "el, arg" if with_element else "arg"
    return (
        "(async " + sig + " => {"
        "  try {"
        "    const ___v = (" + user_expression + ");"
        "    const ___result = (typeof ___v === 'function')"
        "      ? await ___v(" + call_args + ")"
        "      : await ___v;"
        "    let ___t;"
        "    if (Array.isArray(___result)) ___t = 'array';"
        "    else if (___result === null) ___t = 'null';"
        "    else if (___result === undefined) ___t = 'undefined';"
        "    else if (typeof ___result === 'object') ___t = 'object';"
        "    else ___t = typeof ___result;"
        "    return { __aug_ok: true, value: ___result, type: ___t };"
        "  } catch (e) {"
        "    const stack = String((e && e.stack) || '');"
        "    const m = stack.match(/<anonymous>:(\\d+):(\\d+)/) "
        "             || stack.match(/at [^\\n]*:(\\d+):(\\d+)/);"
        "    return { __aug_ok: false, error: {"
        "      message: String((e && e.message) || e),"
        "      name: String((e && e.name) || 'Error'),"
        "      stack: stack.slice(0, 2000),"
        "      line: m ? parseInt(m[1], 10) : null,"
        "      column: m ? parseInt(m[2], 10) : null,"
        "    }};"
        "  }"
        "})"
    )


# Truncation budgets — exposed as module constants so tests can pin them
# and a future call-site can override per request if needed.
EVALUATE_RESULT_BYTE_BUDGET = 50_000
EVALUATE_STRING_CAP = 2_000
EVALUATE_ARRAY_CAP = 50
EVALUATE_OBJECT_KEYS_CAP = 50
EVALUATE_DEPTH_CAP = 8


def _trim_evaluate_result(v, depth, str_cap, arr_cap, obj_cap, depth_cap):
    """Structure-aware trim of a JSON-shaped value.

    Replaces the previous "byte-slice at 50KB" strategy. Walking the parsed
    value lets us:

    - Cap strings without breaking them mid-codepoint or producing an
      unterminated JSON string literal.
    - Show array head + a "...(N more items)" sentinel so the agent sees
      both the shape and the size.
    - Cap object key count similarly, with an ``__augmentum_truncated_keys``
      sentinel so the model knows keys were elided.
    - Guard recursion depth so a circular-ish structure can't run away.

    The function is intentionally pure + small so the inline subprocess
    script can be reconstructed from its source via ``inspect.getsource``.
    Keep it dependency-free (stdlib types only) for that reason.
    """
    if depth > depth_cap:
        return f"...(depth>{depth_cap} truncated)"
    if isinstance(v, str):
        return v if len(v) <= str_cap else v[:str_cap] + f"...({len(v)} chars)"
    if isinstance(v, list):
        if len(v) <= arr_cap:
            return [_trim_evaluate_result(x, depth + 1, str_cap, arr_cap, obj_cap, depth_cap) for x in v]
        head_n = max(1, arr_cap // 2)
        head = [_trim_evaluate_result(x, depth + 1, str_cap, arr_cap, obj_cap, depth_cap) for x in v[:head_n]]
        return head + [f"...({len(v) - head_n} more items)"]
    if isinstance(v, dict):
        items = list(v.items())
        if len(items) <= obj_cap:
            return {k: _trim_evaluate_result(val, depth + 1, str_cap, arr_cap, obj_cap, depth_cap) for k, val in items}
        head = {k: _trim_evaluate_result(val, depth + 1, str_cap, arr_cap, obj_cap, depth_cap) for k, val in items[:obj_cap]}
        head["__augmentum_truncated_keys"] = len(items) - obj_cap
        return head
    return v


# Inlined-into-subprocess source of the trim function. Computed once at
# import. The subprocess script can then call ``_trim_evaluate_result(...)``
# by name, with no parallel implementation to drift.
_TRIM_SOURCE = inspect.getsource(_trim_evaluate_result)


async def playwright_evaluate(
    cm,
    workspace_id: str,
    *,
    url: str,
    expression: str,
    args: Any = None,
    selector: str = "",
    viewport: dict[str, int] | None = None,
    wait_for_selector: str = "",
    goto_timeout_ms: int = 15_000,
    timeout_ms: int = 15_000,
) -> dict[str, Any]:
    """Run an arbitrary JS expression in the open preview and return its result.

    Wraps ``page.evaluate(wrapper, arg)`` (or ``locator.evaluate`` when
    ``selector`` is given) so that:

    - JS exceptions are caught and returned as structured
      ``{message, name, stack, line, column}`` instead of a bare string.
    - The user expression can be a value, a sync function, or an async
      function — the wrapper awaits as needed.
    - When ``selector`` is provided the wrapper binds ``el`` to the
      matched element; expression can use ``el.textContent`` directly or
      be a function ``(el, arg) => ...``.
    - When ``args`` is provided it's JSON-serialized and bound to ``arg``
      inside the wrapper, so the agent can parameterize the same probe
      across calls without string-concat injecting values into the JS.
    - Output is structure-aware-trimmed (strings, arrays, objects, depth)
      so the JSON stays parseable instead of being byte-sliced mid-token.
    - The result envelope includes ``result_type`` so the caller knows
      what was returned without re-parsing.

    Timeouts:

    - ``goto_timeout_ms`` — how long page.goto waits (default 15s)
    - ``timeout_ms`` — how long the evaluate itself waits (default 15s)

    The Python subprocess wall-clock is sized to fit both with a small
    safety margin.
    """
    # Sidecar rung — full parity: selector binding and arg injection ride
    # the same _build_evaluate_wrapper contract as the legacy path.
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.evaluate(
            cm, workspace_id, url=url, expression=expression,
            args=args, selector=selector, timeout_ms=timeout_ms,
        ),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    target = _container_reachable_url(url, workspace_id)
    viewport = viewport or {"width": 1280, "height": 800}
    selector = (selector or "").strip()
    wrapped_expr = _build_evaluate_wrapper(expression, with_element=bool(selector))
    # args travels Python -> JSON string -> JS literal so the JS layer never
    # sees user-controlled values interpolated raw. None becomes JS null;
    # any non-JSON-serializable value falls back to str() (Playwright will
    # have its own marshalling on the JS side).
    try:
        args_json = json.dumps(args) if args is not None else ""
    except (TypeError, ValueError):
        args_json = json.dumps(str(args))
    script = (
        "import json,sys,time\n"
        f"url={target!r}; "
        f"wrapped_expr={wrapped_expr!r}; "
        f"args_json={args_json!r}; "
        f"selector={selector!r}; "
        f"viewport={viewport!r}; "
        f"wait_for_selector={wait_for_selector!r}; "
        f"goto_timeout_ms={int(goto_timeout_ms)!r}; "
        f"timeout_ms={int(timeout_ms)!r}\n"
        f"chrome_args={list(HEADLESS_WEBGL_ARGS)!r}\n"
        f"STR_CAP={EVALUATE_STRING_CAP!r}; "
        f"ARR_CAP={EVALUATE_ARRAY_CAP!r}; "
        f"OBJ_CAP={EVALUATE_OBJECT_KEYS_CAP!r}; "
        f"DEPTH_CAP={EVALUATE_DEPTH_CAP!r}; "
        f"BYTE_BUDGET={EVALUATE_RESULT_BYTE_BUDGET!r}\n"
        "start=time.time()\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':False,'error':str(exc)})); sys.exit(0)\n"
        "args_val = json.loads(args_json) if args_json else None\n"
        "console_errors=[]; network_failures=[]\n"
        "def _on_console(msg):\n"
        "    if msg.type in ('error','warning'):\n"
        "        console_errors.append({'type':msg.type,'text':msg.text[:300]})\n"
        "def _on_request_failed(req):\n"
        "    network_failures.append({'url':req.url[:200],'method':req.method,"
        "'failure':(req.failure or '')[:200]})\n"
        "def _on_response(resp):\n"
        "    if resp.status >= 400:\n"
        "        network_failures.append({'url':resp.url[:200],'method':resp.request.method,"
        "'status':resp.status})\n"
        + _TRIM_SOURCE + "\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        browser=p.chromium.launch(headless=True, args=chrome_args)\n"
        "        page=browser.new_page(viewport=viewport)\n"
        "        page.on('console', _on_console)\n"
        "        page.on('requestfailed', _on_request_failed)\n"
        "        page.on('response', _on_response)\n"
        # 'load', NOT 'networkidle' — see playwright_action: a live HMR dev
        # server never goes network-idle, which timed out browser_evaluate.
        "        page.goto(url, wait_until='load', timeout=goto_timeout_ms)\n"
        "        if wait_for_selector:\n"
        "            page.wait_for_selector(wait_for_selector, timeout=10000)\n"
        "        page.set_default_timeout(timeout_ms)\n"
        "        if selector:\n"
        "            target_obj = page.locator(selector)\n"
        "            try:\n"
        "                target_obj.wait_for(state='attached', timeout=min(timeout_ms, 10000))\n"
        "            except Exception as locator_exc:\n"
        "                browser.close()\n"
        "                print(json.dumps({'ok':False,'playwright':True,"
        "'error':'selector not found: ' + selector,"
        "'selector_missing':True,"
        "'console_errors':console_errors[:25],"
        "'network_failures':network_failures[:25]})); sys.exit(0)\n"
        "            envelope = target_obj.evaluate(wrapped_expr, args_val)\n"
        "        else:\n"
        "            envelope = page.evaluate(wrapped_expr, args_val)\n"
        "        browser.close()\n"
        "        if not isinstance(envelope, dict) or '__aug_ok' not in envelope:\n"
        "            # Wrapper didn't produce its envelope — happens if the\n"
        "            # user expression couldn't even parse. Surface the raw\n"
        "            # value so the caller has *something* to debug from.\n"
        "            print(json.dumps({'ok':False,'playwright':True,"
        "'error':'wrapper produced no envelope (likely a JS syntax error in expression)',"
        "'wrapper_error':True,'raw':envelope,"
        "'console_errors':console_errors[:25],"
        "'network_failures':network_failures[:25]})); sys.exit(0)\n"
        "        if not envelope.get('__aug_ok'):\n"
        "            err = envelope.get('error') or {}\n"
        "            print(json.dumps({'ok':False,'playwright':True,'js_error':True,"
        "'error': err.get('message', '') or 'JS error',"
        "'error_detail': err,"
        "'latency_ms':int((time.time()-start)*1000),"
        "'console_errors':console_errors[:25],"
        "'network_failures':network_failures[:25]})); sys.exit(0)\n"
        "        raw_value = envelope.get('value')\n"
        "        result_type = envelope.get('type') or 'unknown'\n"
        "        trimmed = _trim_evaluate_result(raw_value, 0, STR_CAP, ARR_CAP, OBJ_CAP, DEPTH_CAP)\n"
        "        try:\n"
        "            ser = json.dumps(trimmed, default=str, ensure_ascii=False)\n"
        "        except Exception as enc_exc:\n"
        "            ser = json.dumps({'__unserializable__': str(enc_exc)})\n"
        "        truncated = False\n"
        "        if len(ser) > BYTE_BUDGET:\n"
        "            # Re-trim with tighter limits. Output stays parseable\n"
        "            # JSON because the trim returns Python objects, never\n"
        "            # raw string slices of the encoded form.\n"
        "            ser = json.dumps(\n"
        "                _trim_evaluate_result(raw_value, 0, STR_CAP // 4, ARR_CAP // 5, OBJ_CAP // 2, DEPTH_CAP - 3),\n"
        "                default=str, ensure_ascii=False,\n"
        "            )\n"
        "            truncated = True\n"
        "        print(json.dumps({'ok':True,'playwright':True,'result_json':ser,"
        "'result_type':result_type,'truncated':truncated,"
        "'latency_ms':int((time.time()-start)*1000),"
        "'console_errors':console_errors[:25],"
        "'network_failures':network_failures[:25]}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':True,'error':str(exc),"
        "'console_errors':console_errors[:25],"
        "'network_failures':network_failures[:25]}))\n"
    )
    # Subprocess wall-clock: goto + evaluate + a 5s safety margin, never
    # less than 30s (Chromium cold start + page load on a busy host).
    subprocess_timeout = max(30.0, (goto_timeout_ms + timeout_ms) / 1000.0 + 5.0)
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=subprocess_timeout,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "playwright": False, "error": out}


# ---------------------------------------------------------------------------
# Wave-2 primitives: wait / extract / fill_form
#
# Ledger mining (2026-07-02, 20.7k tool calls) showed browser_evaluate was
# the #3 tool overall because these primitives were missing: 24% of its
# expressions embedded hand-rolled setTimeout waits, 38% were DOM-extraction
# loops, 14% called .click() directly. Each helper below replaces one of
# those improvisation classes with a declarative verb.
# ---------------------------------------------------------------------------

# Structured extraction, implemented ON TOP of playwright_evaluate so the
# error envelope, arg binding, and structure-aware trimming are all reused
# (no parallel subprocess script to drift). ``arg`` carries
# {kind, selector, attribute, limit}.
_EXTRACT_JS = """(arg) => {
  const kind = arg.kind, sel = arg.selector, attr = arg.attribute, lim = arg.limit;
  const pick = (s, d) => Array.from(document.querySelectorAll(s || d)).slice(0, lim);
  const txt = (el) => ((el.innerText !== undefined ? el.innerText : el.textContent) || '')
    .replace(/\\s+/g, ' ').trim();
  if (kind === 'links') {
    return pick(sel, 'a[href]').map(a => ({text: txt(a).slice(0, 200), href: a.href}));
  }
  if (kind === 'meta') {
    const metas = {};
    document.querySelectorAll('meta[name],meta[property]').forEach(m => {
      const k = m.getAttribute('name') || m.getAttribute('property');
      if (k && !(k in metas)) metas[k] = (m.getAttribute('content') || '').slice(0, 300);
    });
    return {
      title: document.title,
      lang: document.documentElement.lang || '',
      headings: pick(sel, 'h1,h2,h3').map(h => ({tag: h.tagName.toLowerCase(), text: txt(h).slice(0, 200)})),
      meta: metas,
    };
  }
  if (kind === 'table') {
    return pick(sel, 'table').map(t => {
      const cap = t.querySelector('caption');
      return {
        caption: cap ? txt(cap).slice(0, 200) : '',
        headers: Array.from(t.querySelectorAll('thead th, tr:first-child th')).map(th => txt(th).slice(0, 120)),
        rows: Array.from(t.querySelectorAll('tbody tr, tr')).slice(0, lim).map(tr =>
          Array.from(tr.querySelectorAll('td,th')).map(td => txt(td).slice(0, 200))),
      };
    });
  }
  if (kind === 'list') {
    return pick(sel, 'ul,ol').map(l =>
      Array.from(l.querySelectorAll(':scope > li')).slice(0, lim).map(li => txt(li).slice(0, 300)));
  }
  if (kind === 'attr') {
    return pick(sel, '[' + attr + ']').map(el => ({text: txt(el).slice(0, 120), value: el.getAttribute(attr)}));
  }
  return pick(sel, 'body').map(el => txt(el).slice(0, 4000));
}"""

EXTRACT_KINDS = ("text", "links", "table", "list", "meta", "attr")

# Kinds the plain-HTTP fallback can serve (regex over static HTML). The
# DOM-dependent kinds need a real browser and fail with a clear message.
_HTTP_EXTRACT_KINDS = ("text", "links", "meta")


async def playwright_extract(
    cm,
    workspace_id: str,
    *,
    url: str,
    kind: str = "text",
    selector: str = "",
    attribute: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Structured page extraction — the first-class version of the DOM
    loops agents previously hand-wrote in browser_evaluate."""
    kind = (kind or "text").strip().lower()
    if kind not in EXTRACT_KINDS:
        return {"ok": False, "error": f"unknown kind {kind!r}; use one of {EXTRACT_KINDS}"}
    if kind == "attr" and not (attribute or "").strip():
        return {"ok": False, "error": "kind='attr' requires the 'attribute' argument"}
    limit = max(1, min(200, int(limit or 50)))
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.extract(
            cm, workspace_id, url=url, kind=kind, selector=selector,
            attribute=attribute, limit=limit,
        ),
        cm,
    )
    if sidecar_result is not None:
        sidecar_result["kind"] = kind
        return sidecar_result
    result = await playwright_evaluate(
        cm,
        workspace_id,
        url=url,
        expression=_EXTRACT_JS,
        args={
            "kind": kind,
            "selector": (selector or "").strip(),
            "attribute": (attribute or "").strip(),
            "limit": limit,
        },
    )
    if not result.get("playwright") and kind in _HTTP_EXTRACT_KINDS:
        return await _http_extract(cm, workspace_id, url=url, kind=kind, limit=limit)
    if not result.get("playwright"):
        result["error"] = (
            f"kind={kind!r} extraction needs a real browser — start the "
            f"browser sidecar service (compose.browser.yaml); without it "
            f"only plain-HTTP kinds {_HTTP_EXTRACT_KINDS} work here. "
            + str(result.get("error") or "")
        )
    result["kind"] = kind
    return result


async def _http_extract(
    cm, workspace_id: str, *, url: str, kind: str, limit: int,
) -> dict[str, Any]:
    """Regex-over-static-HTML fallback for workspaces without Playwright.
    Same spirit as ``http_snapshot`` — crude but useful for server-rendered
    pages; JS-rendered content won't appear."""
    target = _container_reachable_url(url, workspace_id)
    script = (
        "import json,re,urllib.request,html\n"
        f"url={target!r}; kind={kind!r}; limit={int(limit)!r}\n"
        "try:\n"
        "    req=urllib.request.Request(url, headers={'User-Agent':'Augmentum-Coder-Browser'})\n"
        "    with urllib.request.urlopen(req, timeout=8) as resp:\n"
        "        raw=resp.read(400000).decode('utf-8','replace')\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'error':str(exc),'fallback':'http'})); raise SystemExit(0)\n"
        "def strip(s): return re.sub(r'\\s+',' ',re.sub(r'<[^>]+>',' ',s)).strip()\n"
        "if kind=='links':\n"
        "    data=[{'text':html.unescape(strip(m.group(2)))[:200],'href':html.unescape(m.group(1))[:500]}\n"
        "          for m in re.finditer(r'<a\\b[^>]*href=[\"\\']([^\"\\']+)[\"\\'][^>]*>(.*?)</a>', raw, re.I|re.S)][:limit]\n"
        "elif kind=='meta':\n"
        "    title=''\n"
        "    m=re.search(r'<title[^>]*>(.*?)</title>', raw, re.I|re.S)\n"
        "    if m: title=html.unescape(strip(m.group(1)))[:200]\n"
        "    metas={}\n"
        "    for mm in re.finditer(r'<meta\\b[^>]*(?:name|property)=[\"\\']([^\"\\']+)[\"\\'][^>]*content=[\"\\']([^\"\\']*)[\"\\']', raw, re.I):\n"
        "        metas.setdefault(mm.group(1), html.unescape(mm.group(2))[:300])\n"
        "    heads=[{'tag':mm.group(1).lower(),'text':html.unescape(strip(mm.group(2)))[:200]}\n"
        "           for mm in re.finditer(r'<(h[1-3])\\b[^>]*>(.*?)</h[1-3]>', raw, re.I|re.S)][:limit]\n"
        "    data={'title':title,'lang':'','headings':heads,'meta':metas}\n"
        "else:\n"
        "    body=re.sub(r'<(script|style)[^>]*>.*?</\\1>','',raw,flags=re.I|re.S)\n"
        "    data=[html.unescape(strip(body))[:4000]]\n"
        "print(json.dumps({'ok':True,'fallback':'http','result_json':json.dumps(data,ensure_ascii=False),"
        "'result_type':'array' if isinstance(data,list) else 'object'}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=15.0,
    )
    try:
        parsed = json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        parsed = {"ok": False, "error": out, "fallback": "http"}
    parsed["kind"] = kind
    parsed["playwright"] = False
    return parsed


async def playwright_wait(
    cm,
    workspace_id: str,
    *,
    url: str,
    selector: str = "",
    text: str = "",
    state: str = "visible",
    timeout_ms: int = 10_000,
) -> dict[str, Any]:
    """Declarative wait — replaces the hand-rolled ``setTimeout`` sleeps
    that made up 24% of measured browser_evaluate expressions.

    Exactly one condition: ``selector`` (waits for its ``state``), ``text``
    (waits for it to appear in body innerText), or neither (waits for
    network idle). On timeout the CURRENT page text is returned so the
    model sees what the page actually shows instead of guessing.
    """
    timeout_ms = max(250, min(60_000, int(timeout_ms or 10_000)))
    state = state if state in ("visible", "attached", "hidden", "detached") else "visible"
    # Sidecar rung — all states: visible maps to native `wait <sel>`,
    # attached/hidden/detached poll via `wait --fn`.
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.wait(
            cm, workspace_id, url=url, selector=selector, text=text,
            state=state, timeout_ms=timeout_ms,
        ),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    target = _container_reachable_url(url, workspace_id)
    script = (
        "import json,time\n"
        f"url={target!r}; selector={selector!r}; text={text!r}; "
        f"state={state!r}; timeout_ms={int(timeout_ms)!r}\n"
        f"chrome_args={list(HEADLESS_WEBGL_ARGS)!r}\n"
        "start=time.time()\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':False,'error':str(exc)})); raise SystemExit(0)\n"
        "console_errors=[]\n"
        "def _on_console(msg):\n"
        "    if msg.type in ('error','warning'):\n"
        "        console_errors.append({'type':msg.type,'text':msg.text[:300]})\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        browser=p.chromium.launch(headless=True, args=chrome_args)\n"
        "        page=browser.new_page()\n"
        "        page.on('console', _on_console)\n"
        # domcontentloaded, NOT networkidle: the whole point is waiting
        # for the caller's condition, not double-waiting on the network.
        "        page.goto(url, wait_until='domcontentloaded', timeout=15000)\n"
        "        met=False; err=''\n"
        "        try:\n"
        "            if selector:\n"
        "                page.wait_for_selector(selector, state=state, timeout=timeout_ms)\n"
        "            elif text:\n"
        "                page.wait_for_function(\n"
        "                    't => document.body && document.body.innerText.includes(t)',\n"
        "                    arg=text, timeout=timeout_ms)\n"
        "            else:\n"
        "                page.wait_for_load_state('networkidle', timeout=timeout_ms)\n"
        "            met=True\n"
        "        except Exception as wexc:\n"
        "            err=str(wexc)[:300]\n"
        "        title=page.title()\n"
        "        try: body=page.locator('body').inner_text(timeout=3000)[:800]\n"
        "        except Exception: body=''\n"
        "        browser.close()\n"
        "        print(json.dumps({'ok':met,'playwright':True,"
        "'waited_ms':int((time.time()-start)*1000),'title':title,"
        "'body_preview':body,'error':'' if met else ('condition not met within %sms: %s' % (timeout_ms, err)),"
        "'console_errors':console_errors[:25]}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':True,'error':str(exc),"
        "'console_errors':console_errors[:25]}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=timeout_ms / 1000.0 + 25.0,
    )
    try:
        result = json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "playwright": False, "error": out}
    if not result.get("playwright"):
        return await _http_wait(
            cm, workspace_id, url=url, text=text, selector=selector,
            timeout_ms=timeout_ms,
        )
    return result


async def _http_wait(
    cm, workspace_id: str, *, url: str, text: str, selector: str,
    timeout_ms: int,
) -> dict[str, Any]:
    """No-Playwright fallback: poll the static HTML for ``text``.
    Selector waits genuinely need a DOM — fail with the reason."""
    if selector:
        return {
            "ok": False, "playwright": False, "fallback": "http",
            "error": (
                "selector waits need a real browser — start the browser "
                "sidecar service (compose.browser.yaml). Without it use "
                "text=... (polls the static HTML) instead."
            ),
        }
    start = time.time()
    deadline = start + timeout_ms / 1000.0
    last: dict[str, Any] = {}
    while True:
        last = await http_snapshot(cm, workspace_id, url)
        if not text:
            # No condition — a reachable page is the condition.
            if last.get("ok"):
                break
        else:
            hay = " ".join(
                [str(last.get("title") or "")]
                + [str(i.get("text") or "") for i in (last.get("summary") or [])
                   if isinstance(i, dict)]
            )
            if text in hay:
                break
        if time.time() >= deadline:
            return {
                "ok": False, "playwright": False, "fallback": "http",
                "waited_ms": int((time.time() - start) * 1000),
                "title": last.get("title") or "",
                "error": f"condition not met within {timeout_ms}ms (HTTP polling; "
                         f"JS-rendered content is invisible to this fallback)",
            }
        await asyncio.sleep(1.5)
    return {
        "ok": True, "playwright": False, "fallback": "http",
        "waited_ms": int((time.time() - start) * 1000),
        "title": last.get("title") or "",
    }


async def playwright_fill_form(
    cm,
    workspace_id: str,
    *,
    url: str,
    fields: dict[str, Any],
    submit: str = "",
    wait_after_selector: str = "",
    wait_after_text: str = "",
    timeout_ms: int = 15_000,
) -> dict[str, Any]:
    """Fill several fields and optionally submit — one call instead of a
    browser_type round-trip per field.

    Field values: strings fill text inputs/textareas (falling back to
    ``select_option`` for <select>); booleans check/uncheck. Submit only
    fires when EVERY field succeeded — never submits a half-filled form.
    """
    if not isinstance(fields, dict) or not fields:
        return {"ok": False, "error": "fields must be a non-empty object of {selector: value}"}
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.fill_form(
            cm, workspace_id, url=url, fields=fields, submit=submit,
            wait_after_selector=wait_after_selector,
            wait_after_text=wait_after_text, timeout_ms=timeout_ms,
        ),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    target = _container_reachable_url(url, workspace_id)
    timeout_ms = max(1_000, min(60_000, int(timeout_ms or 15_000)))
    try:
        fields_json = json.dumps(fields)
    except (TypeError, ValueError):
        return {"ok": False, "error": "fields must be JSON-serializable (string/number/boolean values)"}
    script = (
        "import json,time\n"
        f"url={target!r}; fields=json.loads({fields_json!r}); submit={submit!r}; "
        f"wait_sel={wait_after_selector!r}; wait_text={wait_after_text!r}; "
        f"timeout_ms={int(timeout_ms)!r}\n"
        f"chrome_args={list(HEADLESS_WEBGL_ARGS)!r}\n"
        "start=time.time()\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':False,'error':str(exc)})); raise SystemExit(0)\n"
        "console_errors=[]; network_failures=[]\n"
        "def _on_console(msg):\n"
        "    if msg.type in ('error','warning'):\n"
        "        console_errors.append({'type':msg.type,'text':msg.text[:300]})\n"
        "def _on_request_failed(req):\n"
        "    network_failures.append({'url':req.url[:200],'method':req.method,"
        "'failure':(req.failure or '')[:200]})\n"
        "def _on_response(resp):\n"
        "    if resp.status >= 400:\n"
        "        network_failures.append({'url':resp.url[:200],'method':resp.request.method,"
        "'status':resp.status})\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        browser=p.chromium.launch(headless=True, args=chrome_args)\n"
        "        page=browser.new_page()\n"
        "        page.on('console', _on_console)\n"
        "        page.on('requestfailed', _on_request_failed)\n"
        "        page.on('response', _on_response)\n"
        # 'load', NOT 'networkidle' — see playwright_action (HMR dev servers).
        "        page.goto(url, wait_until='load', timeout=15000)\n"
        "        results=[]\n"
        "        for sel, val in fields.items():\n"
        "            try:\n"
        "                if isinstance(val, bool):\n"
        "                    page.set_checked(sel, val, timeout=5000)\n"
        "                else:\n"
        "                    try:\n"
        "                        page.fill(sel, str(val), timeout=5000)\n"
        "                    except Exception:\n"
        "                        page.select_option(sel, str(val), timeout=5000)\n"
        "                results.append({'selector':sel,'ok':True})\n"
        "            except Exception as fexc:\n"
        "                results.append({'selector':sel,'ok':False,'error':str(fexc)[:200]})\n"
        "        all_ok=all(r['ok'] for r in results)\n"
        "        submitted=False; submit_error=''\n"
        "        if submit and all_ok:\n"
        "            try:\n"
        "                page.click(submit, timeout=5000); submitted=True\n"
        "                try: page.wait_for_load_state('networkidle', timeout=8000)\n"
        "                except Exception: pass\n"
        "            except Exception as sexc:\n"
        "                submit_error=str(sexc)[:200]\n"
        "        elif submit and not all_ok:\n"
        "            submit_error='skipped: not all fields filled (never submits a half-filled form)'\n"
        "        wait_error=''\n"
        "        try:\n"
        "            if wait_sel: page.wait_for_selector(wait_sel, timeout=timeout_ms)\n"
        "            elif wait_text:\n"
        "                page.wait_for_function(\n"
        "                    't => document.body && document.body.innerText.includes(t)',\n"
        "                    arg=wait_text, timeout=timeout_ms)\n"
        "        except Exception as wexc:\n"
        "            wait_error=str(wexc)[:200]\n"
        "        title=page.title()\n"
        "        try: body=page.locator('body').inner_text(timeout=3000)[:1200]\n"
        "        except Exception: body=''\n"
        "        browser.close()\n"
        "        ok=all_ok and (submitted or not submit) and not wait_error\n"
        "        print(json.dumps({'ok':ok,'playwright':True,'fields':results,"
        "'submitted':submitted,'submit_error':submit_error,'wait_error':wait_error,"
        "'title':title,'body_preview':body,"
        "'latency_ms':int((time.time()-start)*1000),"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'playwright':True,'error':str(exc),"
        "'console_errors':console_errors[:25],'network_failures':network_failures[:25]}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=timeout_ms / 1000.0 + 35.0,
    )
    try:
        result = json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "playwright": False, "error": out}
    if not result.get("playwright"):
        result["error"] = (
            "browser_fill_form needs a real browser — start the browser "
            "sidecar service (compose.browser.yaml); there is no "
            "plain-HTTP way to fill a form. "
            + str(result.get("error") or "")
        )
    return result


async def verify_preview(
    cm,
    workspace_id: str,
    *,
    url: str = "",
) -> dict[str, Any]:
    url = url or await load_browser_session(cm, workspace_id) or await infer_preview_url(cm, workspace_id)
    if not url:
        return {
            "ok": False,
            "error": "No preview URL is open and no listening common dev port was detected.",
            "checks": [],
        }
    sidecar_result = await _try_sidecar(
        lambda: _sidecar.verify_preview(cm, workspace_id, url=url),
        cm,
    )
    if sidecar_result is not None:
        return sidecar_result
    checks: list[dict[str, Any]] = []
    desktop = await playwright_action(
        cm,
        workspace_id,
        url=url,
        action="open",
        viewport={"width": 1440, "height": 900},
    )
    if desktop.get("playwright"):
        mobile = await playwright_action(
            cm,
            workspace_id,
            url=url,
            action="open",
            viewport={"width": 390, "height": 844},
        )
        checks.extend([
            {"viewport": "desktop", **desktop},
            {"viewport": "mobile", **mobile},
        ])
        # Aggregate per-viewport errors so the top-level result carries
        # the union — callers that don't drill into per-viewport detail
        # still see "page had 3 JS errors and 2 failed network calls"
        # without having to walk the checks array themselves.
        agg_console: list[dict] = []
        agg_network: list[dict] = []
        for c in checks:
            for err in (c.get("console_errors") or []):
                agg_console.append({"viewport": c.get("viewport", ""), **err})
            for fail in (c.get("network_failures") or []):
                agg_network.append({"viewport": c.get("viewport", ""), **fail})
        ok = all(bool(c.get("ok")) for c in checks)
        return {
            "ok": ok,
            "url": url,
            "mode": "playwright",
            "checks": checks,
            "console_errors": agg_console[:50],
            "network_failures": agg_network[:50],
        }
    snap = await http_snapshot(cm, workspace_id, url)
    checks.append({"viewport": "http", **snap})
    return {
        "ok": bool(snap.get("ok")),
        "url": url,
        "mode": "http",
        "checks": checks,
        "console_errors": snap.get("console_errors") or [],
        "network_failures": snap.get("network_failures") or [],
        "error": snap.get("error") or "",
    }
