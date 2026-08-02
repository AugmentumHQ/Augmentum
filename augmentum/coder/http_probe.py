"""HTTP probe helper — structured request/response for the ``http_request`` tool.

Workspace images ship ``curl``, ``httpie``, and the Python ``requests`` library
in the base image (see Dockerfile.workspace). Before this module the agent
hand-rolled ``curl`` flags through ``shell_exec`` and re-parsed the output
each time — error-prone and slow. ``run_http_request`` executes a small
Python script inside the workspace that uses ``requests`` (rich + structured)
with a ``urllib`` fallback in case the wheel is missing from a custom image.

The script prints a single JSON line so the caller can parse it cleanly. All
captured bytes (body, headers) are capped to keep tool output budgeted.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from augmentum.coder.services import _container_reachable_url

# Max bytes of response body the script returns. Larger responses are
# truncated and flagged via ``body_truncated=True`` so the agent knows
# the cut happened. 50 KB is enough for typical JSON API responses while
# keeping a single tool output under the global tool-output cap.
_MAX_BODY_BYTES = 50_000

# Permitted HTTP methods. Locked down to common REST verbs — exotic
# verbs (TRACE, CONNECT) almost always indicate the agent took a wrong
# turn rather than a legitimate need.
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})


async def run_http_request(
    cm,
    workspace_id: str,
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: float = 15.0,
    follow_redirects: bool = True,
    verify_tls: bool = True,
) -> dict[str, Any]:
    """Make an HTTP request from inside the workspace container.

    Returns a dict with: ``ok``, ``status``, ``reason``, ``headers``,
    ``body`` (truncated), ``body_truncated``, ``final_url`` (after
    redirects), ``latency_ms``, ``error``. ``ok`` is True iff a response
    was received and ``status`` is 2xx/3xx.

    ``url`` accepts the same shapes ``browser_open`` does — a workspace
    preview path like ``/api/coder/preview/{ws_id}/{port}/`` is
    rewritten to ``http://127.0.0.1:{port}/...`` before the request, so
    the same URL works from both the host UI and inside the container.
    """
    method = (method or "GET").strip().upper()
    if method not in _METHODS:
        return {
            "ok": False,
            "error": f"method must be one of {sorted(_METHODS)}; got {method!r}",
            "validation_error": True,
        }
    target = _container_reachable_url(url or "", workspace_id)
    if not target:
        return {
            "ok": False,
            "error": "url is required",
            "validation_error": True,
        }
    headers = dict(headers or {})

    # Use ``requests`` when available (richer headers handling, automatic
    # redirect tracking with full final URL). Fall back to ``urllib`` so
    # this still works on slimmed-down images where requests was pruned.
    script = (
        "import json,time,sys\n"
        f"url={target!r}; method={method!r}; headers={headers!r}; body={body!r}; "
        f"timeout={float(timeout)!r}; follow={bool(follow_redirects)!r}; "
        f"verify_tls={bool(verify_tls)!r}; max_body={int(_MAX_BODY_BYTES)!r}\n"
        "start=time.time()\n"
        "use_requests=True\n"
        "try:\n"
        "    import requests\n"
        "except Exception:\n"
        "    use_requests=False\n"
        "try:\n"
        "    if use_requests:\n"
        "        resp=requests.request(method, url, headers=headers,"
        " data=(body.encode('utf-8') if body else None),"
        " timeout=timeout, allow_redirects=follow, verify=verify_tls)\n"
        "        raw=resp.content[:max_body]\n"
        "        body_truncated=bool(len(resp.content) > max_body)\n"
        "        try:\n"
        "            body_text=raw.decode('utf-8','replace')\n"
        "        except Exception:\n"
        "            body_text=repr(raw[:200])\n"
        "        hdrs={k:v[:600] for k,v in dict(resp.headers).items()}\n"
        "        out=dict(status=int(resp.status_code), reason=str(getattr(resp,'reason','') or ''),"
        " headers=hdrs, body=body_text, body_truncated=body_truncated,"
        " final_url=str(resp.url), latency_ms=int((time.time()-start)*1000),"
        " ok=(200 <= resp.status_code < 400), error='')\n"
        "    else:\n"
        "        import urllib.request, urllib.error\n"
        "        req=urllib.request.Request(url, method=method, headers=headers,"
        " data=(body.encode('utf-8') if body else None))\n"
        "        try:\n"
        "            with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
        "                raw=resp.read(max_body+1); body_truncated=bool(len(raw) > max_body); raw=raw[:max_body]\n"
        "                hdrs={k:v[:600] for k,v in dict(resp.headers.items()).items()}\n"
        "                out=dict(status=int(resp.status), reason='',"
        " headers=hdrs, body=raw.decode('utf-8','replace'), body_truncated=body_truncated,"
        " final_url=resp.geturl(), latency_ms=int((time.time()-start)*1000),"
        " ok=(200 <= resp.status < 400), error='')\n"
        "        except urllib.error.HTTPError as exc:\n"
        "            raw=exc.read(max_body+1); body_truncated=bool(len(raw) > max_body); raw=raw[:max_body]\n"
        "            hdrs={k:v[:600] for k,v in dict(exc.headers.items()).items()} if exc.headers else {}\n"
        "            out=dict(status=int(exc.code), reason=str(exc.reason or ''),"
        " headers=hdrs, body=raw.decode('utf-8','replace'), body_truncated=body_truncated,"
        " final_url=exc.geturl(), latency_ms=int((time.time()-start)*1000),"
        " ok=False, error='')\n"
        "    print(json.dumps(out))\n"
        "except Exception as exc:\n"
        "    print(json.dumps(dict(ok=False, status=0, reason='', headers={},"
        " body='', body_truncated=False, final_url=url,"
        " latency_ms=int((time.time()-start)*1000), error=str(exc))))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=float(timeout) + 5.0,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {
            "ok": False, "status": 0, "headers": {}, "body": "",
            "body_truncated": False, "final_url": target,
            "latency_ms": 0, "error": out or "no output from probe",
        }
