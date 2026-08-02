"""Dump the exact Host / X-Forwarded-Host cloudflared delivers to the origin,
to confirm which header the tunnel-guard must key on. Run:
    python scripts/smoke_cf_headers.py /path/to/cloudflared[.exe]
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
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class _H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        line = (f"Host={self.headers.get('Host')!r} "
                f"XFH={self.headers.get('X-Forwarded-Host')!r}")
        b = line.encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


async def main() -> int:
    cf = sys.argv[1] if len(sys.argv) > 1 else "cloudflared"
    port = _free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    proc = await asyncio.create_subprocess_exec(
        cf, "tunnel", "--no-autoupdate", "--http-host-header", INVITE_TUNNEL_HOST,
        "--url", f"http://127.0.0.1:{port}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        url = ""
        for _ in range(400):
            raw = await proc.stdout.readline()
            if not raw:
                break
            url = CloudflaredEngine.parse_url(raw.decode("utf-8", "replace"))
            if url:
                break
        if not url:
            print("no url"); return 1

        def _f() -> str:
            import time
            last = None
            for _ in range(20):
                try:
                    with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310
                        return r.read().decode()
                except Exception as e:  # noqa: BLE001
                    last = e; time.sleep(4)
            return f"FAIL {last}"
        print("ORIGIN SAW:", await asyncio.get_event_loop().run_in_executor(None, _f))
        return 0
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
        srv.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
