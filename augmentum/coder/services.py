"""Managed long-running services for Coder workspaces.

The service layer gives the agent a small process API instead of forcing
it to hand-roll ``npm run dev &`` shells every turn. Records persist in
SQLite when the app state is available and also mirror into
``/workspace/.augmentum/services.json`` so tools still have a workspace
local source of truth in stripped-down test or dev environments.
"""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import aiosqlite

from augmentum.coder.executors import ContainerExecutor
from augmentum.tools.base import ToolResult

_SERVICE_FILE = "/workspace/.augmentum/services.json"
_SERVICE_LOG_DIR = "/workspace/.augmentum/services"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(slots=True)
class WorkspaceService:
    id: str
    user_id: str
    workspace_id: str
    name: str
    command: str
    cwd: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    pid: int = 0
    ports: list[int] = field(default_factory=list)
    log_path: str = ""
    status: str = "unknown"
    last_probe: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(data: Any) -> str:
    return json.dumps(data, default=str, sort_keys=True)


def _row_to_service(row: aiosqlite.Row | dict[str, Any]) -> WorkspaceService:
    def _get(key: str, default: Any = "") -> Any:
        try:
            return row[key]
        except Exception:
            return default

    try:
        env = json.loads(_get("env_json", "{}") or "{}")
    except Exception:
        env = {}
    try:
        ports = json.loads(_get("ports_json", "[]") or "[]")
    except Exception:
        ports = []
    try:
        last_probe = json.loads(_get("last_probe", "{}") or "{}")
    except Exception:
        last_probe = {}
    return WorkspaceService(
        id=str(_get("id")),
        user_id=str(_get("user_id", "") or ""),
        workspace_id=str(_get("workspace_id", "") or ""),
        name=str(_get("name", "") or ""),
        command=str(_get("command", "") or ""),
        cwd=str(_get("cwd", "/workspace") or "/workspace"),
        env={str(k): str(v) for k, v in (env or {}).items()},
        pid=int(_get("pid", 0) or 0),
        ports=[int(p) for p in (ports or []) if str(p).isdigit()],
        log_path=str(_get("log_path", "") or ""),
        status=str(_get("status", "unknown") or "unknown"),
        last_probe=last_probe if isinstance(last_probe, dict) else {},
        error=str(_get("error", "") or ""),
        created_at=float(_get("created_at", 0.0) or 0.0),
        updated_at=float(_get("updated_at", 0.0) or 0.0),
    )


def _normalize_cwd(cwd: str | None) -> str:
    clean = (cwd or "/workspace").strip() or "/workspace"
    if clean == ".":
        return "/workspace"
    if clean.startswith("./"):
        clean = clean[2:]
    if not clean.startswith("/"):
        clean = f"/workspace/{clean}"
    if clean != "/workspace" and not clean.startswith("/workspace/"):
        raise ValueError("Service cwd must be inside /workspace")
    return clean


def _normalize_ports(ports: list[int] | list[str] | None) -> list[int]:
    out: list[int] = []
    for raw in ports or []:
        try:
            port = int(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535 and port not in out:
            out.append(port)
    return out[:12]


def _normalize_env(env: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (env or {}).items():
        key_s = str(key).strip()
        if not _ENV_KEY_RE.match(key_s):
            continue
        out[key_s] = str(value)
    return out


class CoderServiceStore:
    """SQLite CRUD for ``coder_workspace_services``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def upsert(self, svc: WorkspaceService) -> WorkspaceService:
        now = time.time()
        svc.updated_at = now
        if not svc.created_at:
            svc.created_at = now
        await self._conn.execute(
            """
            INSERT INTO coder_workspace_services
                (id, user_id, workspace_id, name, command, cwd, env_json,
                 pid, ports_json, log_path, status, last_probe, error,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id=excluded.user_id,
                workspace_id=excluded.workspace_id,
                name=excluded.name,
                command=excluded.command,
                cwd=excluded.cwd,
                env_json=excluded.env_json,
                pid=excluded.pid,
                ports_json=excluded.ports_json,
                log_path=excluded.log_path,
                status=excluded.status,
                last_probe=excluded.last_probe,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (
                svc.id,
                svc.user_id or "",
                svc.workspace_id,
                svc.name,
                svc.command,
                svc.cwd,
                _json(svc.env),
                int(svc.pid or 0),
                _json(svc.ports),
                svc.log_path,
                svc.status,
                _json(svc.last_probe),
                svc.error,
                svc.created_at,
                svc.updated_at,
            ),
        )
        await self._conn.commit()
        return svc

    async def list(
        self,
        *,
        user_id: str,
        workspace_id: str,
    ) -> list[WorkspaceService]:
        self._conn.row_factory = aiosqlite.Row
        cursor = await self._conn.execute(
            """
            SELECT * FROM coder_workspace_services
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
            """,
            (user_id or "", workspace_id),
        )
        return [_row_to_service(row) for row in await cursor.fetchall()]

    async def get(
        self,
        service_id: str,
        *,
        user_id: str = "",
        workspace_id: str = "",
    ) -> WorkspaceService | None:
        self._conn.row_factory = aiosqlite.Row
        sql = "SELECT * FROM coder_workspace_services WHERE id = ?"
        params: list[Any] = [service_id]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if workspace_id:
            sql += " AND workspace_id = ?"
            params.append(workspace_id)
        sql += " LIMIT 1"
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        return _row_to_service(row) if row else None

    async def delete(
        self,
        service_id: str,
        *,
        user_id: str = "",
        workspace_id: str = "",
    ) -> bool:
        sql = "DELETE FROM coder_workspace_services WHERE id = ?"
        params: list[Any] = [service_id]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if workspace_id:
            sql += " AND workspace_id = ?"
            params.append(workspace_id)
        cursor = await self._conn.execute(sql, params)
        await self._conn.commit()
        return cursor.rowcount > 0


class WorkspaceServiceManager:
    """Controller used by routes and native Coder tools."""

    def __init__(
        self,
        container_manager,
        workspace_id: str,
        *,
        store: CoderServiceStore | None = None,
        user_id: str = "",
    ) -> None:
        self._cm = container_manager
        self._executor = ContainerExecutor(container_manager, workspace_id)
        self._workspace_id = workspace_id
        self._store = store
        self._user_id = user_id or ""

    async def start(
        self,
        *,
        command: str,
        name: str = "",
        cwd: str = "/workspace",
        env: dict[str, Any] | None = None,
        ports: list[int] | list[str] | None = None,
    ) -> WorkspaceService:
        command = (command or "").strip()
        if not command:
            raise ValueError("service_start requires a command")
        cwd = _normalize_cwd(cwd)
        env_norm = _normalize_env(env)
        ports_norm = _normalize_ports(ports)
        service_id = "svc_" + uuid.uuid4().hex[:14]
        clean_name = (name or "").strip() or command.split(None, 1)[0][:48]
        log_path = f"{_SERVICE_LOG_DIR}/{service_id}.log"
        env_pairs = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(env_norm.items())
        )
        env_prefix = f"env {env_pairs} " if env_pairs else "env "
        launch = (
            f"mkdir -p {shlex.quote(_SERVICE_LOG_DIR)} && "
            f"cd {shlex.quote(cwd)} && "
            f"( nohup {env_prefix}bash -lc {shlex.quote(command)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! )"
        )
        output = await self._executor.run_command(["bash", "-lc", launch],
            timeout=10.0,
        )
        pid = 0
        for token in re.findall(r"\b\d+\b", output or ""):
            pid = int(token)
            break
        if pid <= 0:
            raise RuntimeError(
                "Service command was launched but no process id was returned. "
                f"Output: {(output or '').strip()}"
            )
        now = time.time()
        svc = WorkspaceService(
            id=service_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            name=clean_name,
            command=command,
            cwd=cwd,
            env=env_norm,
            pid=pid,
            ports=ports_norm,
            log_path=log_path,
            status="running",
            created_at=now,
            updated_at=now,
        )
        await self._save(svc)
        return svc

    async def list(self) -> list[WorkspaceService]:
        services = await self._load_services()
        refreshed: list[WorkspaceService] = []
        for svc in services:
            refreshed.append(await self.refresh_status(svc))
        return refreshed

    async def restart(self, service_id: str) -> WorkspaceService:
        """Re-launch a stopped service with its original configuration.

        Counterpart to :meth:`stop` — closes the user-facing toggle loop:
        a service the user (or agent) registered once stays as a durable
        row that can be turned off and back on without reissuing the
        command. Reuses the existing service_id so the UI keeps a stable
        handle rather than accumulating phantom rows on each toggle.

        Returns the refreshed service row with a new pid + status=running.
        Raises ``KeyError`` if the service is not found and ``RuntimeError``
        if the launch produced no pid (container crashed, command exited
        immediately, etc.).

        Idempotent when the service is already running — re-probes status,
        returns the live row unchanged.
        """
        svc = await self.get(service_id)
        if svc is None:
            raise KeyError(f"Service {service_id} not found")
        # Re-probe in case status drifted (container restart, manual kill,
        # OOM). Already-running paths exit here so the toggle is safe to
        # call from a UI that polls + clicks.
        refreshed = await self.refresh_status(svc)
        if refreshed.status == "running" and refreshed.pid > 0:
            return refreshed
        # Mirror start()'s launch command exactly so behavior matches the
        # very first run. Logs stream into the same path as before — the
        # logs endpoint shows accumulated history across restarts.
        log_path = svc.log_path or f"{_SERVICE_LOG_DIR}/{svc.id}.log"
        env_pairs = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(svc.env.items())
        )
        env_prefix = f"env {env_pairs} " if env_pairs else "env "
        launch = (
            f"mkdir -p {shlex.quote(_SERVICE_LOG_DIR)} && "
            f"cd {shlex.quote(svc.cwd)} && "
            f"( nohup {env_prefix}bash -lc {shlex.quote(svc.command)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! )"
        )
        output = await self._executor.run_command(["bash", "-lc", launch],
            timeout=10.0,
        )
        pid = 0
        for token in re.findall(r"\b\d+\b", output or ""):
            pid = int(token)
            break
        if pid <= 0:
            raise RuntimeError(
                "Service command was launched but no process id was returned. "
                f"Output: {(output or '').strip()}"
            )
        svc.pid = pid
        svc.status = "running"
        svc.log_path = log_path
        svc.error = ""
        svc.updated_at = time.time()
        await self._save(svc)
        return svc

    async def logs(self, service_id: str, *, lines: int = 120) -> str:
        svc = await self.get(service_id)
        if svc is None:
            raise KeyError(f"Service {service_id} not found")
        limit = max(1, min(int(lines or 120), 2000))
        cmd = (
            f"test -f {shlex.quote(svc.log_path)} && "
            f"tail -n {limit} {shlex.quote(svc.log_path)} || true"
        )
        return await self._executor.run_command(["bash", "-lc", cmd],
            timeout=10.0,
        )

    async def stop(self, service_id: str) -> WorkspaceService:
        svc = await self.get(service_id)
        if svc is None:
            raise KeyError(f"Service {service_id} not found")
        if svc.pid > 0:
            cmd = (
                # SIGTERM the process group (PGID ≠ PID when nohup reshuffles).
                "pgid=$(ps -o pgid= -p " + str(int(svc.pid)) + " 2>/dev/null | tr -d ' '); "
                "[ -n \"$pgid\" ] && kill -TERM -- -\"$pgid\" 2>/dev/null || true; "
                "sleep 1; "
                # If anything still alive (not zombie/stopped), escalate to SIGKILL.
                "ps -o state= -p " + str(int(svc.pid)) + " 2>/dev/null | grep -qv '[ZT]' && { "
                "  [ -n \"$pgid\" ] && kill -KILL -- -\"$pgid\" 2>/dev/null || true; "
                "  sleep 0.5; "
                "}; "
                # Final check: Z (zombie), T (stopped), and gone are all "stopped".
                "ps -o state= -p " + str(int(svc.pid)) + " 2>/dev/null | grep -qv '[ZT]' && echo running || echo stopped"
            )
            out = await self._executor.run_command(["bash", "-lc", cmd],
                timeout=5.0,
            )
            svc.status = "running" if "running" in (out or "") else "stopped"
        else:
            svc.status = "stopped"
        svc.updated_at = time.time()
        await self._save(svc)
        return svc

    async def probe(
        self,
        *,
        service_id: str = "",
        url: str = "",
        port: int | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        svc = await self.get(service_id) if service_id else None
        if not url and port is None and svc and svc.ports:
            port = svc.ports[0]
        probe: dict[str, Any]
        if url:
            probe = await self._probe_url(url, timeout=timeout)
        elif port is not None:
            probe = await self._probe_port(int(port), timeout=timeout)
        else:
            raise ValueError("service_probe requires url, port, or a service with ports")
        if svc is not None:
            svc.last_probe = probe
            svc.status = "running" if probe.get("ok") else svc.status
            await self._save(svc)
        return probe

    async def get(self, service_id: str) -> WorkspaceService | None:
        if self._store is not None:
            return await self._store.get(
                service_id,
                user_id=self._user_id,
                workspace_id=self._workspace_id,
            )
        for svc in await self._load_fallback():
            if svc.id == service_id:
                return svc
        return None

    async def refresh_status(self, svc: WorkspaceService) -> WorkspaceService:
        if svc.pid <= 0:
            return svc
        out = await self._executor.run_command(["bash", "-lc", f"ps -o state= -p {int(svc.pid)} 2>/dev/null | grep -qv '[ZT]' && echo running || echo stopped"],
            timeout=3.0,
        )
        status = "running" if "running" in (out or "") else "stopped"
        if status != svc.status:
            svc.status = status
            svc.updated_at = time.time()
            await self._save(svc)
        return svc

    async def _probe_port(self, port: int, *, timeout: float) -> dict[str, Any]:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        script = (
            "import json,socket,sys,time\n"
            f"port={int(port)}; timeout={float(timeout)!r}\n"
            "start=time.time(); s=socket.socket(); s.settimeout(timeout)\n"
            "try:\n"
            "    s.connect(('127.0.0.1', port)); ok=True; err=''\n"
            "except Exception as exc:\n"
            "    ok=False; err=str(exc)\n"
            "finally:\n"
            "    s.close()\n"
            "print(json.dumps({'kind':'tcp','port':port,'ok':ok,'error':err,"
            "'latency_ms':int((time.time()-start)*1000)}))\n"
        )
        out = await self._executor.run_command(["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
            timeout=timeout + 3.0,
        )
        try:
            return json.loads((out or "").strip().splitlines()[-1])
        except Exception:
            return {"kind": "tcp", "port": port, "ok": False, "error": out}

    async def _probe_url(self, url: str, *, timeout: float) -> dict[str, Any]:
        target = _container_reachable_url(url, self._workspace_id)
        script = (
            "import json,sys,time,urllib.request\n"
            f"url={target!r}; timeout={float(timeout)!r}\n"
            "start=time.time()\n"
            "try:\n"
            "    req=urllib.request.Request(url, headers={'User-Agent':'Augmentum-Coder-Probe'})\n"
            "    with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
            "        body=resp.read(2048).decode('utf-8','replace')\n"
            "        status=getattr(resp,'status',0) or resp.getcode()\n"
            "        ok=200 <= int(status) < 400\n"
            "        err=''\n"
            "except Exception as exc:\n"
            "    body=''; status=0; ok=False; err=str(exc)\n"
            "print(json.dumps({'kind':'http','url':url,'status':status,'ok':ok,"
            "'error':err,'latency_ms':int((time.time()-start)*1000),"
            "'preview':body[:300]}))\n"
        )
        out = await self._executor.run_command(["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
            timeout=timeout + 3.0,
        )
        try:
            return json.loads((out or "").strip().splitlines()[-1])
        except Exception:
            return {"kind": "http", "url": target, "ok": False, "error": out}

    async def _save(self, svc: WorkspaceService) -> None:
        if self._store is not None:
            await self._store.upsert(svc)
        await self._save_fallback(svc)

    async def _load_services(self) -> list[WorkspaceService]:
        if self._store is not None:
            return await self._store.list(
                user_id=self._user_id,
                workspace_id=self._workspace_id,
            )
        return await self._load_fallback()

    async def _load_fallback(self) -> list[WorkspaceService]:
        try:
            raw = await self._executor.read_file(_SERVICE_FILE)
            data = json.loads(raw or "[]")
        except Exception:
            data = []
        services: list[WorkspaceService] = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                services.append(WorkspaceService(**{
                    **item,
                    "env": _normalize_env(item.get("env") or {}),
                    "ports": _normalize_ports(item.get("ports") or []),
                }))
        return services

    async def _save_fallback(self, svc: WorkspaceService) -> None:
        try:
            services = await self._load_fallback()
            by_id = {item.id: item for item in services}
            by_id[svc.id] = svc
            payload = json.dumps(
                [item.to_dict() for item in by_id.values()],
                indent=2,
                sort_keys=True,
            )
            await self._executor.run_command(["bash", "-lc", "mkdir -p /workspace/.augmentum"],
                timeout=3.0,
            )
            await self._executor.write_file(_SERVICE_FILE, payload)
        except Exception:
            # The SQLite record is authoritative when present. The fallback
            # is best-effort so a permissions or image quirk does not make
            # service_start fail after the process is already running.
            return


def _container_reachable_url(url: str, workspace_id: str) -> str:
    """Map same-origin preview URLs to a URL reachable inside the container."""
    clean = (url or "").strip()
    if clean.startswith("/api/coder/preview/"):
        parts = clean.split("/", 6)
        if len(parts) >= 6 and parts[4] == workspace_id:
            port = parts[5]
            suffix = "/" + parts[6] if len(parts) > 6 else "/"
            if port.isdigit():
                return f"http://127.0.0.1:{int(port)}{suffix}"
    return clean


def service_result(svc: WorkspaceService) -> ToolResult:
    ports = ", ".join(str(p) for p in svc.ports) or "none declared"
    return ToolResult(
        success=True,
        output=(
            f"Service {svc.name} is {svc.status} (id {svc.id}, pid {svc.pid}).\n"
            f"Command: {svc.command}\n"
            f"Cwd: {svc.cwd}\n"
            f"Ports: {ports}\n"
            f"Logs: {svc.log_path}"
        ),
        metadata={"service": svc.to_dict()},
    )
