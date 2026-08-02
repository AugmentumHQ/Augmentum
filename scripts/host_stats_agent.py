#!/usr/bin/env python3
"""Host stats agent — exposes the host machine's RAM/CPU to Augmentum.

When Augmentum runs inside a Docker container, the in-app resource
monitor can only see the container's view of the system. On Docker
Desktop (Windows/macOS) that's actually the Linux/WSL2 VM, whose RAM
total and CPU usage diverge from what the host OS's Task Manager /
Activity Monitor reports. There is no way for code inside the container
to read the real host numbers — the host OS is opaque to it.

Run this tiny agent on the host and Augmentum will fetch its readings,
so the resource panel can show "host" and "container" side by side.

Usage:

    pip install psutil
    python scripts/host_stats_agent.py            # binds 127.0.0.1:6109

On Docker Desktop (Windows/macOS) that's all you need — the container
reaches a loopback service on the host via ``host.docker.internal``.

On plain Linux Docker, ``host.docker.internal`` resolves to the host's
bridge IP, so loopback isn't reachable; bind to the bridge instead:

    python scripts/host_stats_agent.py --bind 0.0.0.0 --token SECRET

(``--bind 0.0.0.0`` exposes RAM/CPU readings on your LAN — set ``--token``
and ``AUGMENTUM_HOST_STATS_TOKEN`` in the container's environment if that
matters to you.)

Custom port / non-default host: set ``AUGMENTUM_HOST_STATS_URL`` in the
augmentum container, e.g. ``http://host.docker.internal:7000/stats``.

To keep it running across reboots, register it as a startup item:
  - Windows: Task Scheduler → "At log on" → ``pythonw scripts/host_stats_agent.py``
  - macOS:   a LaunchAgent plist
  - Linux:   a systemd user service
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

try:
    import psutil
except ImportError:  # pragma: no cover - import guard for the standalone script
    sys.stderr.write(
        "host_stats_agent: psutil is required.  Install it with:\n"
        "    pip install psutil\n"
    )
    raise SystemExit(1) from None

DEFAULT_PORT = 6109


def _collect() -> dict:
    """One reading of host RAM + CPU, formatted for Augmentum."""
    vm = psutil.virtual_memory()
    # Short blocking sample — matches what the in-app container probe does
    # (psutil.cpu_percent(interval=0.1)) so the two numbers are comparable.
    cpu_pct = psutil.cpu_percent(interval=0.1)
    return {
        "ram": {
            "total_mb": vm.total // (1024 * 1024),
            "used_mb": vm.used // (1024 * 1024),
            "free_mb": vm.available // (1024 * 1024),
        },
        "cpu_pct": round(cpu_pct, 1),
        "cpu_count": psutil.cpu_count(logical=True) or 0,
        "os": platform.system(),       # "Windows" | "Darwin" | "Linux"
        "hostname": socket.gethostname(),
        "agent": "augmentum-host-stats/1",
    }


def _make_handler(token: str):
    class Handler(BaseHTTPRequestHandler):
        # Quieter default logging — one line per request is noisy for a poller.
        def log_message(self, *_args) -> None:  # noqa: D401
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parsed = urlsplit(self.path)
            path = parsed.path.rstrip("/") or "/"

            if token:
                supplied = parse_qs(parsed.query).get("token", [""])[0]
                if supplied != token:
                    self._send_json(403, {"error": "bad or missing token"})
                    return

            if path == "/stats":
                try:
                    self._send_json(200, _collect())
                except Exception as exc:  # pragma: no cover - defensive
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return

            if path == "/":
                self._send_json(200, {"ok": True, "agent": "augmentum-host-stats/1",
                                      "endpoint": "/stats"})
                return

            self._send_json(404, {"error": "not found"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augmentum host stats agent")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1; use 0.0.0.0 "
                             "for plain Linux Docker — this exposes stats on the LAN)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--token", default="",
                        help="optional shared secret; clients must pass ?token=...")
    args = parser.parse_args(argv)

    handler = _make_handler(args.token.strip())
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    where = f"http://{args.bind}:{args.port}/stats"
    sys.stderr.write(f"host_stats_agent: serving {where}"
                     + (" (token required)\n" if args.token.strip() else "\n"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
