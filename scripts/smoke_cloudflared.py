"""Live smoke test for the cloudflared quick-tunnel path (NOT a CI test).

Spawns a REAL cloudflared quick tunnel against a tiny local HTTP server, then
hits the assigned *.trycloudflare.com URL from the public internet and checks:

  1. cloudflared spawns and prints an assignable URL,
  2. CloudflaredEngine.parse_url extracts it from the real output,
  3. a request through the public URL actually reaches the local server,
  4. --http-host-header rewrites the Host to the sentinel the middleware guard
     path-scopes on (augmentum-invite-gate.internal) — the whole security model
     of the public tier depends on this.

Usage:  python scripts/smoke_cloudflared.py /path/to/cloudflared[.exe]
Exit 0 = all checks passed. Requires network egress to cloudflare.
"""
from __future__ import annotations

import asyncio
import contextlib
import http.server
import socket
import sys
import threading
import urllib.request

from augmentum.connect.reachability import INVITE_TUNNEL_HOST, CloudflaredEngine


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        host = self.headers.get("Host", "")
        body = f"ok path={self.path} host={host}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


async def _spawn(cloudflared: str, target: str):
    return await asyncio.create_subprocess_exec(
        cloudflared, "tunnel", "--no-autoupdate",
        "--http-host-header", INVITE_TUNNEL_HOST,
        "--url", target,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )


async def _capture_url(proc, timeout=40.0) -> str:
    async def _loop() -> str:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return ""
            line = raw.decode("utf-8", errors="replace")
            url = CloudflaredEngine.parse_url(line)
            if url:
                return url
    try:
        return await asyncio.wait_for(_loop(), timeout)
    except TimeoutError:
        return ""


async def main() -> int:
    cloudflared = sys.argv[1] if len(sys.argv) > 1 else "cloudflared"
    port = _free_port()
    server = http.server.HTTPServer(("127.0.0.1", port), _EchoHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[local] echo server on http://127.0.0.1:{port}")

    proc = await _spawn(cloudflared, f"http://127.0.0.1:{port}")
    try:
        url = await _capture_url(proc)
        if not url:
            print("[FAIL] no trycloudflare URL captured")
            return 1
        print(f"[tunnel] up at {url}  (parse_url OK)")

        def _fetch(path: str) -> str:
            # The freshly-assigned hostname can take 10-30s to become globally
            # resolvable; retry through DNS/connection errors before giving up.
            last = None
            for _ in range(20):
                try:
                    req = urllib.request.Request(url + path, headers={"User-Agent": "smoke"})
                    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
                        return r.read().decode()
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    import time as _t
                    _t.sleep(4)
            raise RuntimeError(f"public fetch never succeeded: {last}")

        body = await asyncio.get_event_loop().run_in_executor(None, _fetch, "/ui/portal/")
        print(f"[public] GET /ui/portal/ -> {body!r}")
        ok_reach = "ok path=/ui/portal/" in body
        ok_host = f"host={INVITE_TUNNEL_HOST}" in body
        print(f"[check] reaches local server: {ok_reach}")
        print(f"[check] Host rewritten to sentinel '{INVITE_TUNNEL_HOST}': {ok_host}")
        return 0 if (ok_reach and ok_host) else 1
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
