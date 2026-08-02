"""Dynamic HTTPS front doors for provisioned media servers.

Provisioned media containers (Jellyfin, Suwayomi, Audiobookshelf, Komga)
speak plain HTTP on the internal Docker network. To make them reachable
over *real* HTTPS — in a browser (HSTS-correct) and from native TV/phone
apps — we give each one a dedicated TLS front door: a Caddy reverse-proxy
listener on a reserved port, terminated with the same trusted leaf cert
Caddy already serves for the main app, proxying to the container's
internal HTTP.

Mechanism (no DB, no migration):
  - The front-door ports are published once on the caddy service in
    compose.yaml (a reserved range). Docker port bindings are fixed at
    container creation, so the range is reserved up front; the snippets
    below fill it dynamically.
  - Augmentum writes one Caddy *site block* per server into a volume
    shared with the caddy container (``caddy_sites``). The Caddyfile
    ``import``s ``/etc/caddy/sites/*.caddy``, so a cold caddy start picks
    up every front door with no action from Augmentum — restart-safe.
  - After writing/removing a snippet, Augmentum triggers a graceful
    ``caddy reload`` via ``docker exec`` in the caddy container. The
    default admin endpoint lives on caddy's container-localhost:2019, so
    nothing is exposed on the network.

Safety: a single malformed imported snippet aborts the WHOLE Caddyfile
parse — which would take down the main app on the next caddy restart. So
snippets are machine-generated from a fixed template with only an int
port and a ``[a-z0-9_-]`` service id (never user input), written
atomically (tmp + ``os.replace``), and ``caddy validate``'d before every
reload. On any failure the just-written snippet is removed and the last
good config is restored, and the error is surfaced to the caller — which
treats front-door failure as non-fatal (the container + the internal
auto-connect still work).
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiodocker

log = get_logger(__name__)

# Reserved TLS front-door port range — MUST match the range published on
# the caddy service in compose.yaml. The catalog assigns each media entry
# an ``https_port`` inside this range.
FRONT_DOOR_PORT_MIN = 6800
FRONT_DOOR_PORT_MAX = 6809

# Augmentum-side mount of the volume shared with caddy (caddy sees it at
# /etc/caddy/sites). Overridable for tests.
SITES_DIR = os.environ.get("AUGMENTUM_CADDY_SITES_DIR", "/caddy-sites")

# Cert/key paths INSIDE the caddy container (caddy_data:/data).
_CADDY_CERT = "/data/cert.pem"
_CADDY_KEY = "/data/key.pem"
_CADDY_CONFIG = "/etc/caddy/Caddyfile"

# Defensive: only ever interpolate a safe service id into a snippet.
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
# Defensive: only ever interpolate a safe hostname into a Caddy site address.
_SAFE_DOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,253}[a-z0-9])?$")

# The published HTTPS port the gate site blocks listen on. Reuses the main
# app's TLS listener (compose publishes :6443), so the existing host-catch-all
# ``:6443 {}`` block already serves the gate apex (main app on the gate domain,
# where login sets the parent-domain cookie) — no separate apex block needed.
GATE_LISTEN_PORT = 6443

# Gate modes — how the identity gate at ``<svc>.<gate_domain>:6443`` treats an
# upstream. All modes share the same unbounded listener (one :6443, unlimited
# subdomains), so unlike the dedicated-port door they are NOT capped by the
# reserved 6800-6809 range.
#
#   off    — no gate block; the service is reachable only on its raw port.
#   access — forward_auth gates ACCESS (is this a trusted Augmentum user?),
#            then proxies straight through. The app keeps its OWN login for
#            identity. This is the door for form-login apps (n8n) that can't
#            take an injected credential.
#   basic  — forward_auth + server-side Basic-credential injection, so the
#            app's login DISSOLVES for trusted users (managed-auth media
#            servers). The browser never sees the credential.
#
# Natural extension (not yet built): ``sso`` — like ``access`` but the gate
# ALSO seeds the app's own session (form-login POST / cookie mint) so its
# login dissolves too. It slots in as one more branch in ``_gate_block_text``
# + ``gate_routes.gate_verify`` beside ``access``/``basic``.
GATE_MODE_OFF = "off"
GATE_MODE_ACCESS = "access"
GATE_MODE_BASIC = "basic"
# Straight TLS proxy on the gate subdomain — NO forward_auth. For apps with
# their OWN login (Open WebUI, n8n, Gitea, …): forward_auth breaks their SPA/
# websockets (every XHR gets 302'd to the Augmentum login → the app receives
# HTML/403 instead of JSON), and the app's own auth already protects it. Apps
# with NO auth (CyberChef, drawio) use ACCESS instead, where forward_auth is
# the only thing guarding them.
GATE_MODE_PROXY = "proxy"

# Serialize snippet-write + reload so two concurrent installs (or an
# install racing restore-on-startup) can't interleave a half-written
# config with a reload.
_LOCK = asyncio.Lock()

# Regex to extract the listen port from a caddy snippet file header.
# Snippets start with ":PORT {" — this is the ground truth for what
# caddy actually binds, regardless of what in-memory state claims.
_SNIPPET_PORT_RE = re.compile(r"^:(\d+)\s*\{", re.MULTILINE)


def claimed_snippet_ports(exclude_service_id: str = "") -> dict[int, str]:
    """Return ``{port: service_id}`` for every snippet on disk whose port
    falls in the front-door range. This is the **disk-level ground truth**
    for port allocation — it reads actual snippet files, not in-memory
    state, so pre-persistence installs whose ``config_json`` has no
    ``https_port`` are still visible to the allocator.

    Pass ``exclude_service_id`` to ignore a specific snippet (used when
    re-writing the same service's front door so it doesn't collide with
    itself).
    """
    out: dict[int, str] = {}
    sites_dir = Path(SITES_DIR)
    if not sites_dir.is_dir():
        return out
    for f in sites_dir.glob("*.caddy"):
        sid = f.stem
        if sid == exclude_service_id:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        m = _SNIPPET_PORT_RE.search(text)
        if m:
            port = int(m.group(1))
            if front_door_port_ok(port):
                out[port] = sid
    return out


def snippet_port_for(service_id: str) -> int:
    """Return the front-door port encoded in a service's caddy snippet,
    or 0 when the snippet is absent / unreadable / out of range.

    This is the **disk-level truth** — it survives restarts and predates
    ``config_json`` persistence. Used by boot rehydration to recover ports
    for pre-persistence installs whose in-memory definitions have
    ``https_port=0`` but whose snippet files still bind real ports.
    """
    if not _SAFE_ID.match(service_id or ""):
        return 0
    path = _snippet_path(service_id)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    m = _SNIPPET_PORT_RE.search(text)
    if not m:
        return 0
    port = int(m.group(1))
    return port if front_door_port_ok(port) else 0


def allocate_front_door_port(used: set[int]) -> int:
    """Pick a free port in the reserved front-door range, or 0 when the
    range is exhausted (caller degrades to no front door, loudly).

    Runtime service manifests don't hand-pick https_ports — collisions
    between community-authored manifests would be inevitable. The
    installer allocates against everything already claimed (catalog
    entries + running managed services)."""
    for port in range(FRONT_DOOR_PORT_MIN, FRONT_DOOR_PORT_MAX + 1):
        if port not in used:
            return port
    return 0


def front_door_port_ok(https_port: int) -> bool:
    return FRONT_DOOR_PORT_MIN <= int(https_port) <= FRONT_DOOR_PORT_MAX


def _snippet_path(service_id: str) -> Path:
    return Path(SITES_DIR) / f"{service_id}.caddy"


def gate_domain() -> str:
    """The configured front-gate domain (e.g. "aug.lan"), or "" when disabled."""
    try:
        from augmentum.config import settings
        return (settings.gate_domain or "").strip().lower()
    except Exception:  # noqa: BLE001 — never let config access break a reconcile
        return ""


# Served by every front-door block's ``handle_errors`` while the upstream
# service container is up but not yet answering (still doing first-time setup
# — DB migrations, model/data downloads). Without this, a booting service
# returns a raw Caddy 502 (or, during an app restart, the misleading "Waking
# Augmentum" page) even though the app is "installed". The page is a static
# file on the caddy container (written by the compose entrypoint) so Caddy's
# {placeholder} expansion never touches its CSS/JS braces; it's service-aware
# (reads the subdomain) and polls until the real service answers, keyed off the
# ``X-Augmentum-Service-Starting`` marker header this block sets.
_STARTING_ERRORS = (
    "\thandle_errors 502 503 {\n"
    "\t\troot * /srv\n"
    "\t\trewrite * /service-starting.html\n"
    "\t\tfile_server\n"
    '\t\theader Content-Type "text/html; charset=utf-8"\n'
    '\t\theader Cache-Control "no-store"\n'
    '\t\theader X-Augmentum-Service-Starting "1"\n'
    "\t}\n"
)


def _snippet_text(
    service_id: str,
    https_port: int,
    internal_port: int,
    gate_domain: str = "",
    gate_mode: str = GATE_MODE_OFF,
) -> str:
    """The Caddy site block(s) for one provisioned service.

    Always emits the dedicated-port front door (the "use it on your network /
    native apps" door — server's own login). When ``gate_domain`` is set AND
    ``gate_mode`` isn't ``off``, the service is ALSO published at
    ``<service>.<gate_domain>`` behind the identity gate — ``access`` (gate
    access, app keeps its login) or ``basic`` (dissolve login via injected
    credential). Caddy upgrades websockets transparently for both.
    """
    target = f"augmentum-{service_id}:{int(internal_port)}"
    blocks: list[str] = []
    # Dedicated-port door — only when the service won a port from the bounded
    # 6800-6809 pool. Past the 10th install the pool is exhausted (https_port
    # 0); that must NOT suppress the gate door below, or the service falls
    # through Caddy's catch-all to the Augmentum app ("Ollama is running").
    if front_door_port_ok(https_port):
        blocks.append(
            f":{int(https_port)} {{\n"
            f"\ttls {_CADDY_CERT} {_CADDY_KEY}\n"
            f"\treverse_proxy {target} {{\n"
            f"\t\tflush_interval -1\n"
            f"\t\ttransport http {{\n"
            f"\t\t\tread_timeout 600s\n"
            f"\t\t\twrite_timeout 600s\n"
            f"\t\t}}\n"
            f"\t}}\n"
            f"{_STARTING_ERRORS}"
            f"}}\n"
        )
    # Gate-subdomain door — the UNBOUNDED path (one shared :6443 listener).
    # Independent of the port pool, so every installed service is reachable
    # via <svc>.<gate_domain> however many are installed.
    if gate_domain and gate_mode != GATE_MODE_OFF and _SAFE_DOMAIN.match(gate_domain):
        blocks.append(_gate_block_text(
            service_id, internal_port, gate_domain, gate_mode,
        ))
    return "\n".join(blocks)


def _gate_block_text(
    service_id: str, internal_port: int, domain: str, mode: str = GATE_MODE_BASIC,
) -> str:
    """Identity-gated site block, dispatched by ``mode``.

    ``access`` gates access and passes through (app keeps its own login);
    ``basic`` injects a Basic credential to dissolve login. New modes (e.g.
    ``sso``) add a branch here + a matching arm in ``gate_routes.gate_verify``.
    """
    if mode == GATE_MODE_PROXY:
        return _gate_block_proxy(service_id, internal_port, domain)
    if mode == GATE_MODE_ACCESS:
        return _gate_block_access(service_id, internal_port, domain)
    return _gate_block_basic(service_id, internal_port, domain)


def _gate_block_proxy(service_id: str, internal_port: int, domain: str) -> str:
    """Straight TLS proxy on the gate subdomain — NO forward_auth (websockets
    upgrade transparently). For apps that authenticate their own users; their
    login is the sole gate, and forward_auth would break their SPA."""
    target = f"augmentum-{service_id}:{int(internal_port)}"
    return (
        f"{service_id}.{domain}:{GATE_LISTEN_PORT} {{\n"
        f"\ttls {_CADDY_CERT} {_CADDY_KEY}\n"
        f"\treverse_proxy {target} {{\n"
        f"\t\tflush_interval -1\n"
        f"\t\ttransport http {{\n"
        f"\t\t\tread_timeout 600s\n"
        f"\t\t\twrite_timeout 600s\n"
        f"\t\t}}\n"
        f"\t}}\n"
        f"{_STARTING_ERRORS}"
        f"}}\n"
    )


def _gate_block_access(service_id: str, internal_port: int, domain: str) -> str:
    """Access-gate site block: forward_auth gates ACCESS, then proxies straight
    through so the app keeps its OWN login for identity.

    This is the unbounded door for form-login apps (n8n) that can't take an
    injected Basic credential. The app's own cookies pass through untouched so
    its session persists; the Augmentum ``forward_auth`` only decides *whether*
    the request reaches the app. The natural extension is a ``sso`` mode that
    ALSO seeds the app's session so its login dissolves too — until then this
    keeps the app's login, which is correct and safe.
    """
    target = f"augmentum-{service_id}:{int(internal_port)}"
    return (
        f"{service_id}.{domain}:{GATE_LISTEN_PORT} {{\n"
        f"\ttls {_CADDY_CERT} {_CADDY_KEY}\n"
        f"\troute {{\n"
        f"\t\tforward_auth augmentum:6100 {{\n"
        f"\t\t\turi /api/gate/verify?svc={service_id}\n"
        f"\t\t}}\n"
        f"\t\treverse_proxy {target} {{\n"
        f"\t\t\tflush_interval -1\n"
        f"\t\t\ttransport http {{\n"
        f"\t\t\t\tread_timeout 600s\n"
        f"\t\t\t\twrite_timeout 600s\n"
        f"\t\t\t}}\n"
        f"\t\t}}\n"
        f"\t}}\n"
        f"{_STARTING_ERRORS}"
        f"}}\n"
    )


def _gate_block_basic(service_id: str, internal_port: int, domain: str) -> str:
    """Identity-gated site block: forward_auth → inject upstream Basic cred.

    The gate validates the Augmentum session via /api/gate/verify; on success
    Caddy copies the returned ``X-Aug-Gate-Authz`` onto the request, we map it
    to ``Authorization`` and strip both the helper header and the inbound
    ``Cookie`` (so the upstream never sees the Augmentum session token). Only
    emitted for Basic-auth (managed_auth) upstreams — form-login apps use the
    ``access`` mode instead.
    """
    target = f"augmentum-{service_id}:{int(internal_port)}"
    # route{} forces handler order: strip any client-supplied gate/auth headers
    # FIRST, so the injected credential can only originate from
    # /api/gate/verify (a LAN client can't forge X-Aug-Gate-Authz / Authorization
    # to reach the upstream un-gated). Then forward_auth, then proxy.
    return (
        f"{service_id}.{domain}:{GATE_LISTEN_PORT} {{\n"
        f"\ttls {_CADDY_CERT} {_CADDY_KEY}\n"
        f"\troute {{\n"
        f"\t\trequest_header -X-Aug-Gate-Authz\n"
        f"\t\trequest_header -Authorization\n"
        f"\t\tforward_auth augmentum:6100 {{\n"
        f"\t\t\turi /api/gate/verify?svc={service_id}\n"
        f"\t\t\tcopy_headers X-Aug-Gate-Authz\n"
        f"\t\t}}\n"
        f"\t\treverse_proxy {target} {{\n"
        f"\t\t\theader_up Authorization {{http.request.header.X-Aug-Gate-Authz}}\n"
        f"\t\t\theader_up -X-Aug-Gate-Authz\n"
        f"\t\t\theader_up -Cookie\n"
        f"\t\t\tflush_interval -1\n"
        f"\t\t\ttransport http {{\n"
        f"\t\t\t\tread_timeout 600s\n"
        f"\t\t\t\twrite_timeout 600s\n"
        f"\t\t\t}}\n"
        f"\t\t}}\n"
        f"\t}}\n"
        f"{_STARTING_ERRORS}"
        f"}}\n"
    )




def write_snippet(
    service_id: str,
    https_port: int,
    internal_port: int,
    gate_domain: str = "",
    gate_mode: str = GATE_MODE_OFF,
) -> Path:
    """Atomically write a front-door snippet. Returns its path.

    Raises ValueError on an unsafe id or out-of-range port — callers must
    not let either reach the Caddyfile.
    """
    if not _SAFE_ID.match(service_id or ""):
        raise ValueError(f"unsafe service id for caddy snippet: {service_id!r}")
    _gate_ok = bool(gate_domain) and gate_mode != GATE_MODE_OFF and bool(
        _SAFE_DOMAIN.match(gate_domain or ""))
    if not front_door_port_ok(https_port) and not _gate_ok:
        # No dedicated port AND no gate door → nothing to write. (A valid
        # https_port alone, or a gate door alone, is enough.)
        raise ValueError(
            f"no front door possible for {service_id!r}: https_port "
            f"{https_port} outside {FRONT_DOOR_PORT_MIN}-{FRONT_DOOR_PORT_MAX} "
            f"and no gate domain configured",
        )
    sites_dir = Path(SITES_DIR)
    sites_dir.mkdir(parents=True, exist_ok=True)
    final = _snippet_path(service_id)
    tmp = sites_dir / f".{service_id}.caddy.tmp"
    tmp.write_text(
        _snippet_text(service_id, https_port, internal_port, gate_domain, gate_mode),
        encoding="utf-8",
    )
    os.replace(tmp, final)  # atomic within the volume
    return final


def remove_snippet(service_id: str) -> bool:
    """Delete a front-door snippet. Returns True if one was removed."""
    if not _SAFE_ID.match(service_id or ""):
        return False
    try:
        _snippet_path(service_id).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        log.warning("front_door_snippet_remove_failed", service=service_id, exc_info=True)
        return False


def snippet_exists(service_id: str) -> bool:
    return _SAFE_ID.match(service_id or "") is not None and _snippet_path(service_id).exists()


async def _find_caddy_container(docker: aiodocker.Docker):
    """Resolve the caddy container by name (compose may suffix -1/-2)."""
    try:
        matches = await docker.containers.list(all=True, filters={"name": ["caddy"]})
    except Exception:
        log.warning("front_door_caddy_lookup_failed", exc_info=True)
        return None
    for c in matches:
        names = []
        try:
            names = c._container.get("Names", []) or []  # noqa: SLF001 — aiodocker dict
        except Exception:
            names = []
        if any("caddy" in str(n).lower() for n in names):
            return c
    return matches[0] if matches else None


async def _exec(docker: aiodocker.Docker, cmd: list[str]) -> tuple[bool, str]:
    """Run a command in the caddy container; return (ok, combined output)."""
    container = await _find_caddy_container(docker)
    if container is None:
        return False, "caddy container not found"
    try:
        exec_obj = await container.exec(
            cmd=cmd, stdin=False, stdout=True, stderr=True, tty=False,
        )
        stream = exec_obj.start(detach=False)
        chunks: list[bytes] = []
        while True:
            msg = await stream.read_out()
            if msg is None:
                break
            if getattr(msg, "data", None):
                chunks.append(msg.data)
        info = await exec_obj.inspect()
        exit_code = info.get("ExitCode")
        output = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return exit_code == 0, output
    except Exception as exc:  # noqa: BLE001 — surface as a clean failure
        return False, str(exc)


def _caddy_unavailable(output: str) -> bool:
    """True when validate/reload failed because caddy isn't reachable (not a
    config error). On a cold start caddy boots AFTER augmentum (depends_on),
    so the startup reconcile can't exec into it yet — in that case we KEEP the
    snippet (caddy imports it when it starts) instead of rolling it back."""
    o = (output or "").lower()
    return (
        "not running" in o
        or "no such container" in o
        or "caddy container not found" in o
        or "409" in o
        or "is restarting" in o
        or "connection refused" in o
    )


async def _validate(docker: aiodocker.Docker) -> tuple[bool, str]:
    return await _exec(docker, ["caddy", "validate", "--config", _CADDY_CONFIG])


async def _reload(docker: aiodocker.Docker) -> tuple[bool, str]:
    return await _exec(docker, ["caddy", "reload", "--config", _CADDY_CONFIG])


async def apply_front_door(
    docker: aiodocker.Docker | None,
    service_id: str,
    https_port: int,
    internal_port: int,
    *,
    gate_mode: str = GATE_MODE_OFF,
) -> bool:
    """Write + activate a service's HTTPS front door.

    Returns True if the front door is live (snippet written and Caddy
    reloaded). On validation/reload failure the snippet is rolled back and
    the previous config restored, returning False. Never raises for an
    operational failure — front doors are additive and non-fatal.

    ``gate_mode`` (``access`` | ``basic``) additionally publishes the service
    at ``<id>.<gate_domain>`` behind the identity gate, but only when a gate
    domain is actually configured — otherwise just the dedicated-port door is
    written. ``docker is None`` still writes the snippet (cold-start import
    picks it up) and returns False.
    """
    gd = gate_domain() if gate_mode != GATE_MODE_OFF else ""
    try:
        async with _LOCK:
            # Idempotent: if the snippet already matches, it's live (or will
            # be on caddy's next import) — skip validate+reload so the
            # startup reconcile doesn't hammer caddy with redundant reloads.
            if not _SAFE_ID.match(service_id or ""):
                raise ValueError(f"unsafe service id for caddy snippet: {service_id!r}")
            _gate_ok = bool(gd) and gate_mode != GATE_MODE_OFF
            if not front_door_port_ok(https_port) and not _gate_ok:
                # Neither a dedicated port nor a gate door → nothing to do.
                # (Port-pool exhaustion alone is fine: the gate door below is
                # unbounded and carries the service on its own.)
                raise ValueError(
                    f"no front door possible for {service_id!r}: https_port "
                    f"{https_port} out of range and no gate domain",
                )
            desired = _snippet_text(service_id, https_port, internal_port, gd, gate_mode)
            try:
                current = _snippet_path(service_id).read_text(encoding="utf-8")
            except OSError:
                current = None
            if current == desired:
                return True

            write_snippet(service_id, https_port, internal_port, gd, gate_mode)
            if docker is None:
                return False
            ok, out = await _validate(docker)
            if not ok:
                if _caddy_unavailable(out):
                    # Cold-start ordering: caddy isn't up yet. KEEP the snippet
                    # (it imports on caddy's start); don't roll back.
                    log.info("front_door_deferred", service=service_id, reason=out[:120])
                    return False
                remove_snippet(service_id)
                log.warning(
                    "front_door_validate_failed",
                    service=service_id, port=https_port, output=out[:500],
                )
                return False
            ok, out = await _reload(docker)
            if not ok:
                if _caddy_unavailable(out):
                    log.info("front_door_deferred", service=service_id, reason=out[:120])
                    return False
                # Roll back: drop the snippet and reload to restore last-good.
                remove_snippet(service_id)
                await _reload(docker)
                log.warning(
                    "front_door_reload_failed",
                    service=service_id, port=https_port, output=out[:500],
                )
                return False
            log.info("front_door_up", service=service_id, port=https_port)
            return True
    except ValueError:
        # Unsafe id / out-of-range port — programmer/catalog error.
        log.warning("front_door_rejected", service=service_id, port=https_port, exc_info=True)
        return False
    except Exception:  # noqa: BLE001 — never let a front door break install
        log.warning("front_door_apply_failed", service=service_id, exc_info=True)
        return False


async def remove_front_door(docker: aiodocker.Docker | None, service_id: str) -> None:
    """Remove a media server's front door (snippet + reload best-effort)."""
    try:
        async with _LOCK:
            removed = remove_snippet(service_id)
            if removed and docker is not None:
                await _reload(docker)
    except Exception:  # noqa: BLE001
        log.warning("front_door_remove_failed", service=service_id, exc_info=True)


# ── Workspace service gate ──────────────────────────────────────────
# Workspace containers run on Docker's default bridge, not the compose
# network — Caddy can't reach them by container name. Instead, the
# snippet proxies to host.docker.internal:<host_port>, which is the
# Docker-assigned host port for the published container port.

_WS_ID_PREFIX = "ws-"
_SAFE_WS_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def _ws_snippet_id(workspace_slug: str) -> str:
    return f"{_WS_ID_PREFIX}{workspace_slug}"


def _ws_gate_snippet_text(
    workspace_slug: str,
    host_port: int,
    domain: str,
) -> str:
    """Caddy site block for a workspace service behind the identity gate."""
    svc_id = _ws_snippet_id(workspace_slug)
    target = f"host.docker.internal:{int(host_port)}"
    return (
        f"{svc_id}.{domain}:{GATE_LISTEN_PORT} {{\n"
        f"\ttls {_CADDY_CERT} {_CADDY_KEY}\n"
        f"\troute {{\n"
        f"\t\tforward_auth augmentum:6100 {{\n"
        f"\t\t\turi /api/gate/verify?svc={svc_id}\n"
        f"\t\t}}\n"
        f"\t\treverse_proxy {target} {{\n"
        f"\t\t\tflush_interval -1\n"
        f"\t\t\ttransport http {{\n"
        f"\t\t\t\tread_timeout 600s\n"
        f"\t\t\t\twrite_timeout 600s\n"
        f"\t\t\t}}\n"
        f"\t\t}}\n"
        f"\t}}\n"
        f"}}\n"
    )


async def apply_workspace_gate(
    docker: aiodocker.Docker | None,
    workspace_slug: str,
    host_port: int,
) -> bool:
    """Write + activate a Caddy gate snippet for a workspace service.

    The snippet proxies ``<ws-slug>.<gate_domain>:6443`` to
    ``host.docker.internal:<host_port>`` with HTTPS + Augmentum auth.
    Returns True if live. Non-fatal — workspace services work without
    the gate (direct host port access on LAN).
    """
    gd = gate_domain()
    if not gd:
        return False
    slug = (workspace_slug or "").strip().lower()
    if not _SAFE_WS_SLUG.match(slug):
        log.warning("workspace_gate_unsafe_slug", slug=slug)
        return False

    svc_id = _ws_snippet_id(slug)
    desired = _ws_gate_snippet_text(slug, host_port, gd)

    try:
        async with _LOCK:
            try:
                current = _snippet_path(svc_id).read_text(encoding="utf-8")
            except OSError:
                current = None
            if current == desired:
                return True

            sites_dir = Path(SITES_DIR)
            sites_dir.mkdir(parents=True, exist_ok=True)
            final = _snippet_path(svc_id)
            tmp = sites_dir / f".{svc_id}.caddy.tmp"
            tmp.write_text(desired, encoding="utf-8")
            os.replace(tmp, final)

            if docker is None:
                return False
            ok, out = await _validate(docker)
            if not ok:
                if _caddy_unavailable(out):
                    log.info("workspace_gate_deferred", slug=slug, reason=out[:120])
                    return False
                remove_snippet(svc_id)
                log.warning("workspace_gate_validate_failed", slug=slug, output=out[:500])
                return False
            ok, out = await _reload(docker)
            if not ok:
                if _caddy_unavailable(out):
                    log.info("workspace_gate_deferred", slug=slug, reason=out[:120])
                    return False
                remove_snippet(svc_id)
                await _reload(docker)
                log.warning("workspace_gate_reload_failed", slug=slug, output=out[:500])
                return False
            log.info("workspace_gate_up", slug=slug, host_port=host_port, domain=gd)
            return True
    except Exception:  # noqa: BLE001
        log.warning("workspace_gate_apply_failed", slug=slug, exc_info=True)
        return False


async def remove_workspace_gate(
    docker: aiodocker.Docker | None,
    workspace_slug: str,
) -> None:
    """Remove a workspace service's gate snippet."""
    slug = (workspace_slug or "").strip().lower()
    if not _SAFE_WS_SLUG.match(slug):
        return
    svc_id = _ws_snippet_id(slug)
    try:
        async with _LOCK:
            removed = remove_snippet(svc_id)
            if removed and docker is not None:
                await _reload(docker)
                log.info("workspace_gate_removed", slug=slug)
    except Exception:  # noqa: BLE001
        log.warning("workspace_gate_remove_failed", slug=slug, exc_info=True)


