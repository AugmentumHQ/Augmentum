"""Client for the agent-browser sidecar service (compose.browser.yaml).

The sidecar runs vercel-labs/agent-browser (pinned via AGENT_BROWSER_VERSION,
vendored-binary model like llama-server) — a persistent Rust CDP daemon +
Chrome for Testing in its own container. Augmentum drives it by exec'ing the
``agent-browser`` CLI *inside* that container through the existing
aiodocker/docker-proxy machinery (the CLI↔daemon IPC is a local socket, so
no ports are exposed).

Multi-tenant boundary: agent-browser sessions are named, not authenticated.
Session names are ALWAYS derived server-side here (``ws-<workspace_id>``;
workspace ids are user-scoped) — never accept a caller-supplied session name.

Wave-1 scope (see docs/superpowers/specs/2026-07-16-agent-browser-sidecar-
design.md): stateless per-call mapping onto the existing browser.py result
envelopes. Known gaps, closed in wave 2: ``network_failures`` is always []
on the sidecar path (agent-browser exposes ``network requests`` but the
per-call plumb isn't wired yet); the a11y snapshot-with-refs tool surface.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shlex
import tarfile
import time
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SIDECAR_LABEL = "augmentum.browser_sidecar=true"
_SCREENSHOT_DIR = "/workspace/.augmentum/browser-screenshots"

# Short discovery cache: (monotonic_ts, container_or_none). Sidecar
# presence rarely changes; a 10s TTL keeps the common no-sidecar install
# from paying a containers.list per browser tool call.
_CACHE_TTL = 10.0
_sidecar_cache: tuple[float, Any] | None = None
_cache_lock = asyncio.Lock()


def session_for_workspace(workspace_id: str, *, prefix: str = "ws") -> str:
    """Derived, never caller-supplied — this is our tenant boundary.

    ``prefix`` distinguishes server-side consumers sharing a workspace
    (e.g. the builds behavior gate uses ``gate`` so it never clobbers the
    agent's page state); it is a code-level constant, never user input.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", str(workspace_id))[:80]
    return f"{prefix}-{safe}"


async def find_sidecar(docker) -> Any | None:
    """Locate the running browser sidecar container, or None."""
    global _sidecar_cache
    now = time.monotonic()
    cached = _sidecar_cache
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    async with _cache_lock:
        cached = _sidecar_cache
        if cached and time.monotonic() - cached[0] < _CACHE_TTL:
            return cached[1]
        container = None
        try:
            containers = await docker.containers.list(
                filters=json.dumps({"label": [_SIDECAR_LABEL], "status": ["running"]})
            )
            if containers:
                container = containers[0]
        except Exception as exc:
            log.warning("browser_sidecar_discovery_failed", error=str(exc))
        _sidecar_cache = (time.monotonic(), container)
        return container


async def is_available(docker) -> bool:
    return await find_sidecar(docker) is not None


def invalidate_cache() -> None:
    global _sidecar_cache
    _sidecar_cache = None


async def _exec_collect(container, cmd: list[str], *, timeout: float) -> str:
    """Run a command in the sidecar container, return combined output."""
    exec_obj = await container.exec(
        cmd=cmd, stdin=False, stdout=True, stderr=True, tty=False,
    )
    stream = exec_obj.start(detach=False)
    chunks: list[bytes] = []

    async def _read() -> None:
        async with stream:
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break
                if msg.data:
                    chunks.append(msg.data)

    await asyncio.wait_for(_read(), timeout=timeout)
    return b"".join(chunks).decode("utf-8", "replace")


async def run_cli(
    docker,
    args: list[str],
    *,
    session: str,
    timeout: float = 30.0,
    json_output: bool = True,
) -> dict[str, Any]:
    """Exec ``agent-browser --session <session> [--json] <args...>`` in the
    sidecar and parse the result.

    Returns ``{"ok": bool, ...}``. Non-JSON output lands under ``raw``.
    """
    container = await find_sidecar(docker)
    if container is None:
        return {
            "ok": False,
            "sidecar": False,
            "error": "browser sidecar is not running (enable compose.browser.yaml)",
        }
    cmd = ["agent-browser", "--session", session]
    if json_output:
        cmd.append("--json")
    cmd.extend(str(a) for a in args)
    try:
        out = await _exec_collect(container, cmd, timeout=timeout)
    except TimeoutError:
        return {
            "ok": False,
            "sidecar": True,
            "error": f"agent-browser command timed out after {timeout:.0f}s: "
                     f"{' '.join(shlex.quote(a) for a in args)[:200]}",
        }
    except Exception as exc:
        # A dead exec target usually means the container was recreated —
        # drop the discovery cache so the next call re-finds it.
        invalidate_cache()
        return {"ok": False, "sidecar": True, "error": str(exc)}
    if not json_output:
        return {"ok": True, "sidecar": True, "raw": out}
    # agent-browser --json prints one JSON document per call, shaped
    # {"success": bool, "data": {...}, "error": str|null} (verified live
    # against 0.32.1). We lift data's keys (title/text/result/messages/
    # refs/...) to the top level for the callers, dropping the noisy
    # per-call `lifecycle` block. Scan from the last line up to tolerate
    # stray daemon-spawn noise on earlier lines.
    for line in reversed((out or "").strip().splitlines()):
        line = line.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            result: dict[str, Any] = {
                "ok": bool(parsed.get("success", not parsed.get("error"))),
                "sidecar": True,
                "error": parsed.get("error") or "",
            }
            data = parsed.get("data")
            if isinstance(data, dict):
                for k, v in data.items():
                    if k != "lifecycle":
                        result.setdefault(k, v)
            return result
        return {"ok": True, "sidecar": True, "value": parsed}
    return {
        "ok": False,
        "sidecar": True,
        "error": f"no JSON in agent-browser output: {out[:400]!r}",
        "raw": out[:2000],
    }


# ---------------------------------------------------------------------------
# Vantage rewrite
# ---------------------------------------------------------------------------

_LOCAL_URL_RE = re.compile(
    r"^(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0)(:(\d+))?(/.*)?$", re.I
)
_PREVIEW_URL_RE = re.compile(r"/api/coder/preview/([^/]+)/(\d+)(/.*)?$")


# Networks the sidecar is already confirmed to share, keyed by the
# sidecar CONTAINER ID — a recreated/restarted-with-recreate sidecar
# loses its `docker network connect` attachments, so a bare per-process
# name set would go stale and silently break workspace reachability.
_connected_networks: set[tuple[str, str]] = set()


async def _ensure_sidecar_on_network(docker, network_name: str) -> None:
    """Connect the sidecar to ``network_name`` if it isn't already.

    Workspace containers live on Docker's DEFAULT bridge
    (``coder_workspace_network_mode`` default), while the sidecar joins
    the compose networks — Docker iptables-isolates bridges from each
    other, so without this the sidecar gets ERR_CONNECTION_REFUSED
    toward workspace dev servers (found live: bench run on ws f51e22fc,
    2026-07-17; earlier successes were WSL2 isolation leaking, not
    design). Idempotent; failures are logged and left to the caller's
    normal error surface."""
    if not network_name:
        return
    sidecar = await find_sidecar(docker)
    if sidecar is None:
        return
    try:
        details = await sidecar.show()
        sidecar_id = str(details.get("Id") or "")
        if (sidecar_id, network_name) in _connected_networks:
            return
        have = set((details.get("NetworkSettings") or {}).get("Networks") or {})
        if network_name not in have:
            network = await docker.networks.get(network_name)
            await network.connect({"Container": sidecar_id})
            log.info("browser_sidecar_network_connected", network=network_name)
        _connected_networks.add((sidecar_id, network_name))
    except Exception as exc:
        log.warning("browser_sidecar_network_connect_failed",
                    network=network_name, error=str(exc))


async def _workspace_ip(cm, workspace_id: str) -> str:
    """The workspace container's IP on a network the sidecar can reach —
    connecting the sidecar to that network first when needed."""
    info = await cm._get_workspace(workspace_id)
    if not info.container_id:
        return ""
    try:
        container = await cm._docker.containers.get(info.container_id)
        details = await container.show()
    except Exception as exc:
        log.warning("browser_sidecar_workspace_inspect_failed",
                    workspace_id=workspace_id, error=str(exc))
        return ""
    networks = ((details.get("NetworkSettings") or {}).get("Networks") or {})
    # Prefer the shared workspace network; fall back to any attached net.
    for name, net in networks.items():
        if "workspace" in name and net.get("IPAddress"):
            await _ensure_sidecar_on_network(cm._docker, name)
            return str(net["IPAddress"])
    for name, net in networks.items():
        if net.get("IPAddress"):
            await _ensure_sidecar_on_network(cm._docker, name)
            return str(net["IPAddress"])
    return ""


async def reachable_url(cm, workspace_id: str, url: str) -> str:
    """Rewrite a workspace-local URL to the sidecar's vantage.

    ``localhost:5173`` (in-workspace vantage) and ``/api/coder/preview/
    {ws}/{port}/...`` (user vantage) both become ``http://<workspace_ip>:
    <port>/...``. External URLs pass through untouched.
    """
    u = (url or "").strip()
    m = _PREVIEW_URL_RE.search(u)
    if m:
        ip = await _workspace_ip(cm, m.group(1) or workspace_id)
        if ip:
            return f"http://{ip}:{m.group(2)}{m.group(3) or '/'}"
        return u
    m = _LOCAL_URL_RE.match(u)
    if m:
        ip = await _workspace_ip(cm, workspace_id)
        if ip:
            port = m.group(4) or "80"
            return f"{m.group(1)}{ip}:{port}{m.group(5) or '/'}"
    return u


# ---------------------------------------------------------------------------
# File extraction (screenshots)
# ---------------------------------------------------------------------------

async def pull_file(docker, path: str) -> bytes:
    """Read one file out of the sidecar container via the archive API."""
    container = await find_sidecar(docker)
    if container is None:
        raise RuntimeError("browser sidecar is not running")
    tar_obj = await container.get_archive(path)
    # aiodocker returns a tarfile.TarFile (or raw bytes on some versions).
    if isinstance(tar_obj, bytes | bytearray):
        tar_obj = tarfile.open(fileobj=io.BytesIO(bytes(tar_obj)))  # noqa: SIM115 — closed below
    try:
        for member in tar_obj.getmembers():
            if member.isfile():
                f = tar_obj.extractfile(member)
                if f is not None:
                    return f.read()
    finally:
        tar_obj.close()
    raise RuntimeError(f"no file in archive for {path}")


# ---------------------------------------------------------------------------
# High-level ops — mapped onto browser.py's result envelope.
#
# ``playwright: True`` is a DEPRECATED alias meaning "a real browser ran"
# (runtime_truth / oracle_telemetry / prompts still read it); ``engine``
# is the truthful key. Sweep in wave 4.
# ---------------------------------------------------------------------------

async def _console_errors(docker, session: str) -> list[dict[str, Any]]:
    res = await run_cli(docker, ["console"], session=session, timeout=10.0)
    entries = res.get("messages") or res.get("value") or []
    out = []
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and e.get("type") in ("error", "warning"):
                out.append({"type": e.get("type"), "text": str(e.get("text", ""))[:300]})
    return out[:25]


async def _network_failures(docker, session: str) -> list[dict[str, Any]]:
    """HTTP 4xx/5xx + failed requests from the daemon's request log —
    entry shape verified live against 0.32.1: {url, method, status, ...}."""
    res = await run_cli(docker, ["network", "requests"], session=session, timeout=10.0)
    entries = res.get("requests") or []
    out = []
    if isinstance(entries, list):
        for r in entries:
            if not isinstance(r, dict):
                continue
            status = int(r.get("status") or 0)
            failure = str(r.get("failure") or r.get("errorText") or "")
            if status >= 400 or failure:
                item = {"url": str(r.get("url", ""))[:200],
                        "method": r.get("method", "")}
                if status:
                    item["status"] = status
                if failure:
                    item["failure"] = failure[:200]
                out.append(item)
    return out[:25]


def _envelope(res: dict[str, Any], **extra) -> dict[str, Any]:
    env = {
        "ok": bool(res.get("ok")),
        # ``sidecar: False`` in a run_cli result means the service itself
        # was unreachable — signal the ladder to fall through instead of
        # presenting an infra failure as a page-level answer.
        "engine": "sidecar" if res.get("sidecar", True) else "sidecar_unavailable",
        "playwright": True,  # deprecated alias — real browser ran
        "error": str(res.get("error") or "") if not res.get("ok") else "",
    }
    env.setdefault("network_failures", [])
    env.update(extra)
    return env


async def _page_diagnostics(docker, session: str) -> dict[str, Any]:
    """Console errors + network failures in one envelope-ready dict."""
    return {
        "console_errors": await _console_errors(docker, session),
        "network_failures": await _network_failures(docker, session),
    }


def _urls_equivalent(a: str, b: str) -> bool:
    """Same page modulo trailing slash and scheme-noise — used to decide
    whether an ``open`` is needed at all."""
    def norm(u: str) -> str:
        u = (u or "").strip().rstrip("/")
        return u.split("#", 1)[0]
    return bool(norm(a)) and norm(a) == norm(b)


async def _ensure_page(cm, docker, session: str, workspace_id: str, url: str) -> dict[str, Any] | None:
    """Navigate the session to ``url`` ONLY if it isn't already there.

    Found the hard way (bench run ctr_724dd1e6..., 2026-07-17): an
    unconditional ``open`` per tool call resets page state and
    invalidates snapshot @refs — the exact persistence the sidecar
    exists to provide. A click that re-opens first wipes the counter it
    just incremented; a ref from browser_snapshot dies before
    browser_click can use it.

    Returns None on success (already there, or navigated ok); returns
    the failed-open envelope dict on navigation failure.
    """
    if not url:
        return None
    target = await reachable_url(cm, workspace_id, url)
    current = await run_cli(docker, ["get", "url"], session=session, timeout=10.0)
    if current.get("ok") and _urls_equivalent(str(current.get("url") or ""), target):
        return None
    opened = await run_cli(docker, ["open", target], session=session, timeout=30.0)
    if not opened.get("ok"):
        return _envelope(opened, url=url,
                         **await _page_diagnostics(docker, session))
    return None


async def action(
    cm,
    workspace_id: str,
    *,
    url: str,
    action: str,
    selector: str = "",
    text: str = "",
    wait_for_selector: str = "",
) -> dict[str, Any]:
    """open / click / type against the persistent session."""
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    if wait_for_selector:
        await run_cli(docker, ["wait", wait_for_selector], session=session, timeout=15.0)
    if action == "click" and selector:
        res = await run_cli(docker, ["click", selector], session=session, timeout=15.0)
    elif action == "type" and selector:
        res = await run_cli(docker, ["fill", selector, text], session=session, timeout=15.0)
    else:
        res = {"ok": True}
    title_res = await run_cli(docker, ["get", "title"], session=session, timeout=10.0)
    body_res = await run_cli(docker, ["get", "text", "body"], session=session, timeout=10.0)
    return _envelope(
        res,
        url=url,
        title=str(title_res.get("title") or "")[:200],
        body_preview=str(body_res.get("text") or "")[:2000],
        latency_ms=int((time.time() - start) * 1000),
        **await _page_diagnostics(docker, session),
    )


async def screenshot(
    cm,
    workspace_id: str,
    *,
    url: str,
    full_page: bool = True,
    wait_for_selector: str = "",
) -> dict[str, Any]:
    """Capture in the sidecar, land the PNG in the workspace so the
    existing vision feed picks it up unchanged."""
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    if wait_for_selector:
        await run_cli(docker, ["wait", wait_for_selector], session=session, timeout=15.0)
    remote = f"/tmp/aug_shot_{int(time.time() * 1000)}.png"
    args = ["screenshot"]
    if full_page:
        args.append("--full")
    args.append(remote)
    res = await run_cli(docker, args, session=session, timeout=45.0)
    if not res.get("ok"):
        return _envelope(res, url=url,
                         **await _page_diagnostics(docker, session))
    try:
        png = await pull_file(docker, remote)
    except Exception as exc:
        return _envelope({"ok": False, "error": f"screenshot pull failed: {exc}"}, url=url)
    path = f"{_SCREENSHOT_DIR}/shot_{int(time.time())}.png"
    await cm.run_command(
        workspace_id, ["bash", "-lc", f"mkdir -p {shlex.quote(_SCREENSHOT_DIR)}"],
        timeout=3.0,
    )
    await cm.file_write_bytes(workspace_id, path, png)
    title_res = await run_cli(docker, ["get", "title"], session=session, timeout=10.0)
    return _envelope(
        res,
        path=path,
        url=url,
        title=str(title_res.get("title") or "")[:200],
        full_page=bool(full_page),
        requested_full_page=bool(full_page),
        degraded=False,
        warnings=[],
        latency_ms=int((time.time() - start) * 1000),
        **await _page_diagnostics(docker, session),
    )


async def evaluate(
    cm,
    workspace_id: str,
    *,
    url: str,
    expression: str,
    args: Any = None,
    selector: str = "",
    timeout_ms: int = 15_000,
) -> dict[str, Any]:
    """Full-parity evaluate: reuses browser.py's ``_build_evaluate_wrapper``
    so values / sync fns / async fns, ``arg`` binding, and element scoping
    (``el`` = first ``selector`` match) all behave identically to the legacy
    path. agent-browser's ``eval`` awaits promises (verified live 0.32.1),
    so the async wrapper resolves before returning."""
    # Local import — browser.py imports this module at load time.
    from augmentum.coder.browser import _build_evaluate_wrapper

    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    selector = (selector or "").strip()
    try:
        args_json = json.dumps(args) if args is not None else "null"
    except (TypeError, ValueError):
        args_json = json.dumps(str(args))
    wrapper = _build_evaluate_wrapper(expression, with_element=bool(selector))
    if selector:
        sel_json = json.dumps(selector)
        code = (
            f"(() => {{ const ___el = document.querySelector({sel_json}); "
            f"if (!___el) return {{__aug_ok: false, error: {{message: "
            f"'selector not found: ' + {sel_json}, name: 'SelectorError'}}, "
            f"__aug_selector_missing: true}}; "
            f"return ({wrapper})(___el, {args_json}); }})()"
        )
    else:
        code = f"({wrapper})({args_json})"
    res = await run_cli(
        docker, ["eval", code], session=session,
        timeout=max(10.0, timeout_ms / 1000.0 + 5.0),
    )
    envelope_val = res.get("result") if "result" in res else res.get("value")
    # Unpack the wrapper's structured envelope (same contract as the
    # legacy subprocess path).
    if isinstance(envelope_val, dict) and "__aug_ok" in envelope_val:
        if not envelope_val.get("__aug_ok"):
            err = envelope_val.get("error") or {}
            return _envelope(
                {"ok": False, "sidecar": True,
                 "error": err.get("message", "") or "JS error"},
                js_error=True,
                selector_missing=bool(envelope_val.get("__aug_selector_missing")),
                error_detail=err,
                latency_ms=int((time.time() - start) * 1000),
                **await _page_diagnostics(docker, session),
            )
        value = envelope_val.get("value")
    else:
        value = envelope_val
    try:
        ser = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        ser = json.dumps(str(value))
    return _envelope(
        res,
        result_json=ser[:50_000],
        result_type=type(value).__name__ if value is not None else "null",
        truncated=len(ser) > 50_000,
        latency_ms=int((time.time() - start) * 1000),
        **await _page_diagnostics(docker, session),
    )


async def wait(
    cm,
    workspace_id: str,
    *,
    url: str,
    selector: str = "",
    text: str = "",
    state: str = "visible",
    timeout_ms: int = 10_000,
) -> dict[str, Any]:
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    if selector and state == "visible":
        args = ["wait", selector]
    elif selector:
        # Non-visible states map onto `wait --fn` polling.
        sel_json = json.dumps(selector)
        fns = {
            "attached": f"!!document.querySelector({sel_json})",
            "detached": f"!document.querySelector({sel_json})",
            "hidden": (
                f"(el => !el || el.offsetParent === null)"
                f"(document.querySelector({sel_json}))"
            ),
        }
        args = ["wait", "--fn", fns.get(state, fns["attached"])]
    elif text:
        args = ["wait", "--text", text]
    else:
        args = ["wait", "--load", "networkidle"]
    res = await run_cli(docker, args, session=session,
                        timeout=timeout_ms / 1000.0 + 10.0)
    title_res = await run_cli(docker, ["get", "title"], session=session, timeout=10.0)
    return _envelope(
        res,
        waited_ms=int((time.time() - start) * 1000),
        title=str(title_res.get("title") or "")[:200],
        **await _page_diagnostics(docker, session),
    )


async def extract(
    cm,
    workspace_id: str,
    *,
    url: str,
    kind: str = "text",
    selector: str = "",
    attribute: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Structured extraction — same _EXTRACT_JS + arg contract as the
    legacy path, executed in the persistent sidecar page."""
    from augmentum.coder.browser import _EXTRACT_JS

    return await evaluate(
        cm, workspace_id, url=url, expression=_EXTRACT_JS,
        args={
            "kind": kind,
            "selector": (selector or "").strip(),
            "attribute": (attribute or "").strip(),
            "limit": limit,
        },
    )


async def fill_form(
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
    """Fill several fields and optionally submit. Same never-submit-a-
    half-filled-form rule as the legacy path."""
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    results: list[dict[str, Any]] = []
    for sel, val in fields.items():
        if isinstance(val, bool):
            r = await run_cli(
                docker, ["check" if val else "uncheck", sel],
                session=session, timeout=10.0,
            )
        else:
            r = await run_cli(docker, ["fill", sel, str(val)], session=session, timeout=10.0)
            if not r.get("ok"):
                # <select> elements reject fill — retry as a dropdown.
                r = await run_cli(docker, ["select", sel, str(val)], session=session, timeout=10.0)
        results.append({"selector": sel, "ok": bool(r.get("ok")),
                        **({"error": str(r.get("error"))[:200]} if not r.get("ok") else {})})
    all_ok = all(r["ok"] for r in results)
    submitted = False
    submit_error = ""
    if submit and all_ok:
        s = await run_cli(docker, ["click", submit], session=session, timeout=10.0)
        submitted = bool(s.get("ok"))
        if not submitted:
            submit_error = str(s.get("error") or "")[:200]
    elif submit and not all_ok:
        submit_error = "skipped: not all fields filled (never submits a half-filled form)"
    wait_error = ""
    if wait_after_selector or wait_after_text:
        w_args = (["wait", wait_after_selector] if wait_after_selector
                  else ["wait", "--text", wait_after_text])
        w = await run_cli(docker, w_args, session=session,
                          timeout=timeout_ms / 1000.0 + 10.0)
        if not w.get("ok"):
            wait_error = str(w.get("error") or "")[:200]
    title_res = await run_cli(docker, ["get", "title"], session=session, timeout=10.0)
    body_res = await run_cli(docker, ["get", "text", "body"], session=session, timeout=10.0)
    ok = all_ok and (submitted or not submit) and not wait_error
    return _envelope(
        {"ok": ok, "sidecar": True,
         "error": submit_error or wait_error or
                  ("; ".join(r.get("error", "") for r in results if not r["ok"]) if not all_ok else "")},
        fields=results,
        submitted=submitted,
        submit_error=submit_error,
        wait_error=wait_error,
        title=str(title_res.get("title") or "")[:200],
        body_preview=str(body_res.get("text") or "")[:1200],
        latency_ms=int((time.time() - start) * 1000),
        **await _page_diagnostics(docker, session),
    )


async def set_viewport(docker, session: str, width: int, height: int) -> bool:
    res = await run_cli(
        docker, ["set", "viewport", str(int(width)), str(int(height))],
        session=session, timeout=10.0,
    )
    return bool(res.get("ok"))


async def verify_preview(
    cm,
    workspace_id: str,
    *,
    url: str,
) -> dict[str, Any]:
    """Desktop + mobile verification against the persistent sidecar page —
    same checks/aggregation shape as the legacy dual-Playwright-run path."""
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    checks: list[dict[str, Any]] = []
    for label, (w, h) in (("desktop", (1440, 900)), ("mobile", (390, 844))):
        await set_viewport(docker, session, w, h)
        result = await action(cm, workspace_id, url=url, action="open")
        checks.append({"viewport": label, **result})
    # Restore a sane default so later calls aren't stuck mobile-sized.
    await set_viewport(docker, session, 1280, 800)
    agg_console: list[dict] = []
    agg_network: list[dict] = []
    for c in checks:
        for err in (c.get("console_errors") or []):
            agg_console.append({"viewport": c.get("viewport", ""), **err})
        for fail in (c.get("network_failures") or []):
            agg_network.append({"viewport": c.get("viewport", ""), **fail})
    return {
        "ok": all(bool(c.get("ok")) for c in checks),
        "url": url,
        "mode": "sidecar",
        "engine": "sidecar",
        "playwright": True,  # deprecated alias — real browser ran
        "checks": checks,
        "console_errors": agg_console[:50],
        "network_failures": agg_network[:50],
    }


async def snapshot_a11y(
    cm,
    workspace_id: str,
    *,
    url: str = "",
    interactive: bool = True,
    compact: bool = True,
) -> dict[str, Any]:
    """Accessibility-tree snapshot with stable element refs (@e1, @e2...) —
    the agent-first alternative to CSS-selector guessing. Refs stay valid
    for subsequent click/fill/get calls in the same session."""
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    args = ["snapshot"]
    if interactive:
        args.append("-i")
    if compact:
        args.append("-c")
    res = await run_cli(docker, args, session=session, timeout=20.0)
    title_res = await run_cli(docker, ["get", "title"], session=session, timeout=10.0)
    url_res = await run_cli(docker, ["get", "url"], session=session, timeout=10.0)
    return _envelope(
        res,
        refs=res.get("refs") or {},
        snapshot=res.get("snapshot") or res.get("tree") or "",
        title=str(title_res.get("title") or "")[:200],
        url=str(url_res.get("url") or url)[:500],
        latency_ms=int((time.time() - start) * 1000),
        **await _page_diagnostics(docker, session),
    )


# Allow-listed subcommand heads for the generic passthrough. Everything a
# coder tool exposes goes through here; anything not listed (auth vault,
# plugins, chat, dashboard, install...) is NOT reachable from the model.
COMMAND_ALLOWLIST = (
    "click", "dblclick", "hover", "focus", "press", "keyboard",
    "check", "uncheck", "select", "drag", "scroll", "scrollintoview",
    "back", "forward", "reload", "pushstate",
    "get", "is", "find", "tab", "console", "errors",
    "wait", "highlight", "read", "diff", "vitals",
)


async def command(
    cm,
    workspace_id: str,
    args: list[str],
    *,
    url: str = "",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Generic allow-listed passthrough for the wider verb surface
    (interact/navigate/get/tabs/console/find tools). Opens ``url`` first
    when given; otherwise acts on the session's current page."""
    if not args or args[0] not in COMMAND_ALLOWLIST:
        return {"ok": False, "engine": "sidecar",
                "error": f"command not allowed: {args[:1]!r}"}
    docker = cm._docker
    session = session_for_workspace(workspace_id)
    start = time.time()
    failed = await _ensure_page(cm, docker, session, workspace_id, url)
    if failed is not None:
        return failed
    res = await run_cli(docker, [str(a) for a in args], session=session, timeout=timeout)
    payload = {k: v for k, v in res.items() if k not in ("ok", "sidecar", "error")}
    return _envelope(
        res,
        latency_ms=int((time.time() - start) * 1000),
        **payload,
    )


async def close_workspace_session(cm, workspace_id: str) -> None:
    """Best-effort cleanup when a workspace is deleted."""
    try:
        await run_cli(
            cm._docker, ["close"],
            session=session_for_workspace(workspace_id), timeout=10.0,
        )
    except Exception:
        log.warning("browser_sidecar_session_close_failed",
                    workspace_id=workspace_id, exc_info=True)


# ---------------------------------------------------------------------------
# Real-browser search escalation (retrieval scarcity fallback)
# ---------------------------------------------------------------------------
# When SearXNG's httpx-scraped engines are rate-limited/suspended (Google,
# Brave, Startpage, DDG all return "too many requests"/CAPTCHA under load —
# the recurring 2026-08 scarcity that left the pool full of junk), a REAL
# Chrome fingerprint sails past the same bot-detection. We render
# DuckDuckGo's HTML endpoint (stable, parseable markup) through the existing
# agent-browser sidecar and return results in SearXNG's {url,title,content}
# shape so the caller's normal ranker (filter_for_docs / filter_for_llm)
# handles them uniformly.
#
# Shared, not coder-specific: any SearXNG consumer (doc_search, web_search,
# research) can escalate to this on a thin/degraded pool. Runs on a SEPARATE
# browser session (prefix="search") so it never disturbs the agent's own
# page state. Best-effort: returns [] on any failure — the caller keeps
# whatever SearXNG gave it.

# Bing is the SERP we render: tested against the sidecar it's the one that
# returns parseable results (classic ``li.b_algo`` markup) — DuckDuckGo/
# lite block our datacenter IP even through a real browser, Google serves a
# consent interstitial, and Brave renders results client-side (no links in
# the serialized HTML). Bing still occasionally challenges under load, so
# this stays best-effort: [] on any miss, never a hard dependency.
_BING_URL = "https://www.bing.com/search?q={q}"


def _decode_bing_href(href: str) -> str:
    """Bing wraps outbound links as bing.com/ck/a?...&u=a1<base64url>.
    Return the decoded target; pass through already-direct links."""
    import base64
    from urllib.parse import parse_qs, urlparse

    if not href or "/ck/a" not in href:
        return href
    try:
        u = parse_qs(urlparse(href).query or "").get("u", [""])[0]
        if u.startswith("a1"):
            u = u[2:]
        pad = "=" * (-len(u) % 4)
        decoded = base64.urlsafe_b64decode(u + pad).decode("utf-8", "replace")
        return decoded if decoded.startswith(("http://", "https://")) else href
    except Exception:
        return href


def _parse_bing_html(html: str, max_results: int) -> list[dict]:
    """Parse a Bing SERP into {url,title,content} rows."""
    if not html:
        return []
    try:
        from lxml import html as lxml_html
    except ImportError:
        return []
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return []
    out: list[dict] = []
    for li in doc.xpath('//li[contains(@class, "b_algo")]'):
        a = li.xpath(".//h2/a")
        if not a:
            continue
        url = _decode_bing_href(a[0].get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        title = " ".join(a[0].text_content().split()).strip()
        sn = li.xpath('.//div[contains(@class, "b_caption")]//p | .//p')
        content = " ".join(sn[0].text_content().split()).strip() if sn else ""
        out.append({"url": url, "title": title, "content": content, "engine": "browser-bing"})
        if len(out) >= max_results:
            break
    return out


async def search_serp(
    cm, workspace_id: str, query: str, *, max_results: int = 15, timeout: float = 30.0
) -> list[dict]:
    """Render a Bing SERP via the real-browser sidecar and return
    SearXNG-shaped result rows. Returns [] if the sidecar is unavailable or
    anything fails — a best-effort scarcity escalation, never a hard dep."""
    if cm is None or not (query or "").strip():
        return []
    docker = getattr(cm, "_docker", None)
    if docker is None:
        return []
    try:
        if not await is_available(docker):
            return []
        from urllib.parse import quote_plus

        session = session_for_workspace(workspace_id, prefix="search")
        url = _BING_URL.format(q=quote_plus(query.strip()))
        opened = await run_cli(docker, ["open", url], session=session, timeout=timeout)
        if not opened.get("ok"):
            return []
        got = await run_cli(docker, ["get", "html", "body"], session=session, timeout=15.0)
        if not got.get("ok"):
            return []
        html = str(got.get("html") or got.get("text") or got.get("value")
                   or got.get("result") or got.get("content") or "")
        rows = _parse_bing_html(html, max_results)
        log.info("browser_serp", query=query[:80], results=len(rows))
        return rows
    except Exception:
        log.warning("browser_serp_failed", query=query[:80], exc_info=True)
        return []
