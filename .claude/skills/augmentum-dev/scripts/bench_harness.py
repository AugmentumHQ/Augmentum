#!/usr/bin/env python3
"""Live ground-truth harness primitives for augmentum-dev.

Everything here talks to the REAL running stack: the HTTP API on
localhost, containers via the docker CLI, and the app DB inside the
augmentum container. Built from the 2026-07-17 browser-sidecar
acceptance pass, wrapping the gotchas that burned time doing it by hand:

- ``docker exec`` heredocs silently no-op without ``-i`` → all container
  scripts go through ``docker_py()`` which pipes stdin explicitly.
- MSYS/git-bash mangles absolute container paths and /tmp is not shared
  with the system Python → no host temp files, no path args; scripts
  travel via stdin, results via stdout JSON.
- ``auth_sessions.expires_at`` is compared as an ISO string; inserting
  an epoch float mints an already-expired token → mint uses isoformat.

Config: ``live_acceptance.local.json`` next to this file (git-ignored;
see live_acceptance.py --init). Holds the bench username and the model
for LLM-dependent checks — the model is NEVER auto-selected; checks that
need one SKIP with an actionable message until it is configured.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from _common import ROOT  # noqa: F401  (import applies console UTF-8 safety)

BASE_URL = "http://localhost:6100"
APP_CONTAINER = "augmentum-augmentum-1"
SIDECAR_LABEL = "augmentum.browser_sidecar=true"
CONFIG_FILE = Path(__file__).with_name("live_acceptance.local.json")

_DEFAULT_CONFIG = {
    "bench_username": "bench",
    # Model for checks that call an LLM (e.g. the builds behavior gate's
    # assertion binding). Deliberately empty: the user picks, we never do.
    "model": "",
    "base_url": BASE_URL,
    # Refuse to run live checks below this much free host RAM (GB) — see
    # host_free_memory_gb() for the incident this guards against.
    "min_free_memory_gb": 8,
}


def load_config() -> dict[str, Any]:
    cfg = dict(_DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    return cfg


def write_config_template() -> Path:
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(_DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
    return CONFIG_FILE


# --- docker primitives ------------------------------------------------------

def docker(*args: str, timeout: float = 60.0) -> tuple[int, str]:
    """Run a docker CLI command, return (exit_code, combined output)."""
    try:
        p = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, f"docker unavailable: {exc}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def docker_py(container: str, script: str, *, timeout: float = 300.0,
              env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a python3 script inside ``container`` via stdin (never a temp
    file, never a heredoc — see module docstring)."""
    cmd = ["docker", "exec", "-i"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [container, "python3", "-"]
    try:
        # Bytes in/out: text=True would encode stdin with the Windows
        # locale (cp1252) and any non-ASCII char in the script becomes a
        # SyntaxError inside the container.
        p = subprocess.run(cmd, input=script.encode("utf-8"),
                           capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, f"docker exec failed: {exc}"
    out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
    return p.returncode, out


def last_json(output: str) -> dict[str, Any] | None:
    """Parse the last JSON object line from mixed output (log noise above)."""
    for line in reversed((output or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
    return None


def stack_up() -> bool:
    """True when the augmentum API answers on BASE_URL."""
    try:
        req = urllib.request.Request(load_config()["base_url"] + "/api/auth/me")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except urllib.error.HTTPError:
        return True  # 401 = up, just unauthenticated
    except Exception:
        return False


def sidecar_container() -> str:
    """Name of the running browser sidecar container, or ''."""
    code, out = docker("ps", "--filter", f"label={SIDECAR_LABEL}",
                       "--filter", "status=running", "--format", "{{.Names}}")
    return out.strip().splitlines()[0].strip() if code == 0 and out.strip() else ""


# --- bench session ----------------------------------------------------------

_MINT_SCRIPT = """
import sqlite3, hashlib, json, os
from datetime import datetime, timedelta
tok = os.environ["BENCH_TOKEN"]
username = os.environ["BENCH_USERNAME"]
con = sqlite3.connect("/data/augmentum.db")
row = con.execute("select id from users where username=?", (username,)).fetchone()
if not row:
    print(json.dumps({"ok": False, "error": f"no user {username!r}"})); raise SystemExit(0)
h = hashlib.sha256(tok.encode()).hexdigest()
# expires_at is ISO-string compared by the session manager — epoch floats
# read as already-expired.
exp = (datetime.utcnow() + timedelta(hours=2)).isoformat()
con.execute(
    "insert into auth_sessions (token, user_id, created_at, expires_at,"
    " last_activity, ip_address, user_agent, source)"
    " values (?,?,datetime('now'),?,datetime('now'),?,?,?)",
    (h, row[0], exp, "127.0.0.1", "live-acceptance", "web"))
con.commit()
print(json.dumps({"ok": True, "user_id": row[0]}))
"""

_REVOKE_SCRIPT = """
import sqlite3, hashlib, json, os
h = hashlib.sha256(os.environ["BENCH_TOKEN"].encode()).hexdigest()
con = sqlite3.connect("/data/augmentum.db")
con.execute("delete from auth_sessions where token=?", (h,))
con.commit()
print(json.dumps({"ok": True, "revoked": con.total_changes}))
"""


class BenchSession:
    """Minted bench auth session. Use as a context manager — the token is
    always revoked on exit, pass or fail."""

    def __init__(self, username: str | None = None):
        cfg = load_config()
        self.base_url: str = cfg["base_url"]
        self.username: str = username or cfg["bench_username"]
        self.token: str = ""
        self.user_id: str = ""

    def __enter__(self) -> BenchSession:
        self.token = secrets.token_urlsafe(32)
        code, out = docker_py(
            APP_CONTAINER, _MINT_SCRIPT,
            env={"BENCH_TOKEN": self.token, "BENCH_USERNAME": self.username})
        res = last_json(out)
        if code != 0 or not res or not res.get("ok"):
            raise RuntimeError(
                f"bench token mint failed: {(res or {}).get('error') or out[:300]}")
        self.user_id = res["user_id"]
        return self

    def __exit__(self, *exc) -> None:
        if self.token:
            docker_py(APP_CONTAINER, _REVOKE_SCRIPT,
                      env={"BENCH_TOKEN": self.token})
            self.token = ""

    def api(self, method: str, path: str, body: dict | None = None,
            *, timeout: float = 120.0) -> tuple[int, Any]:
        """Authenticated JSON call against the live API."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={
                "Cookie": f"augmentum_session={self.token}",
                "Origin": self.base_url,  # CSRF check needs it
                "Content-Type": "application/json",
            })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            status = e.code
        except Exception as exc:
            return 0, {"error": str(exc)}
        with contextlib.suppress(json.JSONDecodeError):
            return status, json.loads(raw)
        return status, {"raw": raw[:2000]}


# --- workspace helpers ------------------------------------------------------

def workspace_container_name(workspace_id: str) -> str:
    return f"augmentum-ws-{workspace_id[:8]}"


def workspace_shell(workspace_id: str, cmd: str, *,
                    timeout: float = 60.0) -> tuple[int, str]:
    """bash -lc inside the workspace container."""
    try:
        p = subprocess.run(
            ["docker", "exec", workspace_container_name(workspace_id),
             "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, f"docker exec failed: {exc}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def host_free_memory_gb() -> float | None:
    """Free physical memory on the HOST, or None when undeterminable.

    Exists because of the 2026-07-17 incident: an acceptance run (image
    build + workspace + sidecar Chrome + LLM call) stacked on top of a
    user model load pushed vmmemWSL to 112GB virtual and took the whole
    machine down. Callers should refuse to start expensive live work
    without comfortable headroom.
    """
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass
    try:  # Windows without psutil
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and p.stdout.strip():
            return float(p.stdout.strip()) / 1024**2  # KB -> GB
    except Exception:
        pass
    try:  # Linux/WSL
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
    except Exception:
        pass
    return None


class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.seconds = time.monotonic() - self.start
