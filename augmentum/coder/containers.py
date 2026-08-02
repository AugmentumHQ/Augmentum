"""ContainerManager — async Docker workspace container lifecycle management.

Manages per-user Docker containers as isolated development workspaces.
Containers are labelled with augmentum.workspace=true for easy enumeration
and are constrained by resource limits and capability drops for security.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import posixpath
import re
import shlex
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.coder import profiles as _profiles
from augmentum.coder.models import ContainerInfo, FileEntry
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class ExecAborted(RuntimeError):
    """A ``run_command`` exec ended abnormally (raised only when ``strict=True``).

    Exists because the historic contract — return the partial output with a
    ``[Command timed out after Ns]`` marker appended — is silently lossy for
    STREAMING callers, which consume bytes via ``on_chunk`` and discard the
    return value entirely. That is how a 900s wall-clock kill of a Claude Code
    run surfaced to the user as the generic "claude ended without a result":
    the one string naming the real cause was thrown away by the caller.

    ``kind`` is the machine-readable cause so callers can branch without
    string-sniffing:

    * ``wall_clock`` — the absolute ``timeout`` budget expired
    * ``idle``       — no output for ``idle_timeout`` seconds
    * ``exit_code``  — the process exited non-zero

    ``partial`` carries whatever output had accumulated, so a caller that wants
    the old behaviour can still recover it.
    """

    def __init__(self, kind: str, detail: str, *, partial: str = "") -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.partial = partial


# Environment/build noise to exclude from the review-diff untracked list
# (workspaces frequently don't gitignore these, which would bury real changes).
_NOISE_PATH = re.compile(
    r"(^|/)(\.venv|venv|node_modules|__pycache__|\.git|\.augmentum|\.mypy_cache"
    r"|\.pytest_cache|\.ruff_cache|dist|build|\.next|target|\.tox|site-packages)(/|$)"
)


def _fmt_bytes(n: int | float) -> str:
    """Human-readable byte count. Used in download-progress heartbeats
    emitted by :meth:`ContainerManager._run_command` so the user sees
    '[download progress: 482 MB (+45 MB in 30s)]' rather than a raw
    integer. Binary units (MiB etc.) are avoided because mainstream
    progress tools like ``wget`` show decimal MB, and matching that
    expectation is more useful than strict adherence to IEC prefixes.
    """
    n = float(max(n, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# Docker container labels used to identify augmentum workspace containers
_LABEL_WORKSPACE = "augmentum.workspace"
_LABEL_NAME = "augmentum.name"
_LABEL_ID = "augmentum.id"


def _is_container_gone(exc: BaseException) -> bool:
    """True when the exception indicates Docker no longer has the container.

    aiodocker raises ``DockerError`` with HTTP-style messages; we string-match
    against the cohort of "the container you referenced doesn't exist anymore"
    responses (404 from inspect/get, "No such container" from the daemon).
    Used by lifecycle methods to reconcile a stale ``container_id`` rather
    than re-issuing the same doomed call on every tick.
    """
    msg = str(exc).lower()
    return (
        "404" in msg
        or "no such container" in msg
        or "not found" in msg
    )


def _is_container_not_running(exc: BaseException) -> bool:
    """True when the exception indicates Docker's there but in the wrong state.

    409 from pause/unpause/stop when the container has already left the
    target state (SIGKILL'd, daemon-restarted, externally stopped, etc.).
    The DB row is stale — caller should reconcile to ``stopped`` instead
    of bubbling the error to the user.
    """
    msg = str(exc).lower()
    return "not running" in msg or "is not running" in msg

# Capabilities: drop all then add back the minimum needed for package management
# and basic development (apt-get needs SETUID/SETGID/CHOWN/FOWNER/DAC_OVERRIDE)
_CAP_DROP = ["ALL"]
_CAP_ADD = ["SETUID", "SETGID", "CHOWN", "FOWNER", "DAC_OVERRIDE"]


def _resolve_cap_add(profile_id: str) -> list[str]:
    """Return base CapAdd plus the profile's extra_caps.

    Per-profile caps come from ``ToolingProfile.extra_caps`` in
    profiles.py — e.g. pentest contributes ``NET_RAW`` so nmap SYN
    scans / arp / raw capture work without root. Returns a fresh
    list every call so callers can mutate without affecting the
    next workspace.
    """
    extras: tuple[str, ...] = ()
    try:
        resolved = _profiles.resolve(profile_id)
        extras = resolved.extra_caps
    except (ValueError, AttributeError):
        extras = ()
    return list(_CAP_ADD) + list(extras)

# Common dev-server ports pre-published at workspace creation so a user's
# `npm run dev` / `pytest --http` / etc. is reachable from the host
# browser. Docker auto-assigns a random high host port per binding; the
# mapping is exposed via the /ports endpoint. Bound to 127.0.0.1 only
# so the dev server isn't exposed to the LAN.
_DEV_PORTS = [3000, 3001, 4200, 5000, 5173, 8000, 8080, 8888]
_GIT_AUTHOR_NAME = "Augmentum Workspace"
_GIT_AUTHOR_EMAIL = "workspace@augmentum.local"


def _workspace_slug(name: str, workspace_id: str) -> str:
    """Sanitize a workspace name into a safe slug for Caddy snippets."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "workspace").lower()).strip("-")
    slug = slug[:30] if slug else "ws"
    return f"{slug}-{workspace_id[:6]}"

# Fallback bootstrap for plain ubuntu:24.04 workspaces. This is the
# guide-critical subset of Dockerfile.workspace: enough for the coder
# workspace guide to stay truthful even when the prebaked image is
# missing. Deliberately excludes heavier extras like rustup, aider, and
# provider CLIs so fallback startup remains bounded.
_FALLBACK_APT_PACKAGES = (
    "nano vim-tiny less tree jq httpie "
    "git git-lfs "
    "curl wget ca-certificates openssh-client "
    "build-essential pkg-config "
    "libffi-dev libsqlite3-dev zlib1g-dev libbz2-dev libreadline-dev "
    "libssl-dev libxml2-dev libxslt1-dev "
    "python3 python3-pip python3-venv python3-dev python3-tk "
    "xvfb x11-utils "
    "nodejs npm "
    "golang-go "
    "sqlite3 postgresql-client redis-tools "
    "imagemagick "
    "unzip file sudo ripgrep fd-find bash-completion"
)
_FALLBACK_PIP_PACKAGES = (
    "setuptools wheel "
    "pytest pytest-asyncio ruff black mypy "
    "requests httpx flask fastapi uvicorn"
)
_FALLBACK_NPM_PACKAGES = "typescript ts-node eslint prettier"

# Marker files the keep-alive writes inside the (persistent) /workspace
# volume so the frontend / turn loop can tell a still-provisioning or
# degraded container apart from a healthy one — and so a failed install is
# diagnosable AFTER the fact (the container stays alive instead of dying).
_PROVISION_LOG = "/workspace/.augmentum/provision.log"
_PROVISION_EXIT = "/workspace/.augmentum/provision.exit"

# Persistent dependency layer inside the /workspace named volume.
#
# The workspace FILESYSTEM survives container recreation (ports published,
# LAN toggled, image bumped) because /workspace is a named volume — but pip
# and npm -g installs land in image layers and vanish, so agents lost
# mid-project dependencies and burned turns re-installing + re-diagnosing.
# Fix: a venv INSIDE the volume (--system-site-packages so the baked
# profile packages stay visible; new installs shadow them) plus an npm
# prefix in the volume, both wired into every login shell via
# /etc/profile.d — which `bash -lc` (shell_exec, service_start, test_run,
# terminal sessions) sources. Same canonical-env-extension pattern as
# augmentum-tools.sh in Dockerfile.workspace. profile.d lives in the image
# layer, but these setup lines run on EVERY container start, so it is
# rewritten on recreate.
_WORKSPACE_VENV = "/workspace/.venv"
_WORKSPACE_NPM_PREFIX = "/workspace/.augmentum/npm-global"
# apt system packages install into the container ROOT fs (image layers), which
# recreate wipes — unlike pip/npm, which the venv/prefix above keep in the
# volume. So without capture, every recreate (LAN toggle, port publish, image
# bump) silently reverts a workspace to its birth profile's system tooling. We
# close that with (a) a persisted manifest replayed on every start and (b) an
# apt/apt-get SHIM on PATH that records install targets transparently — no
# agent cooperation and no command-syntax lock-in. The manifest is a plain,
# user-editable file (one package per line), symmetric with requirements.txt.
_WORKSPACE_BIN = "/workspace/.augmentum/bin"
_APT_MANIFEST = "/workspace/.augmentum/apt-packages"
_PERSIST_ENV_SH = f"""\
# Augmentum: persistent workspace dependency layer. Lives in the /workspace
# volume so pip/npm installs survive container recreation. Rewritten on
# every container start by the workspace bootstrap.
export VIRTUAL_ENV={_WORKSPACE_VENV}
export NPM_CONFIG_PREFIX={_WORKSPACE_NPM_PREFIX}
export PATH="{_WORKSPACE_BIN}:{_WORKSPACE_VENV}/bin:{_WORKSPACE_NPM_PREFIX}/bin:$PATH"
"""

# apt/apt-get capture shim. Installed as both names in _WORKSPACE_BIN (ahead of
# /usr/bin on PATH). It forwards verbatim to the real binary and, on a
# SUCCESSFUL ``install``, appends the named packages to the manifest so they
# replay on recreate. Records after success (not before) so failed installs
# don't poison the manifest; dedupe happens at replay via ``sort -u``. Resolves
# the real binary by its own name so one script serves both ``apt`` and
# ``apt-get``. Non-install invocations (update/list/remove) pass straight
# through. Limitation: ``sudo apt-get`` resets PATH and bypasses the shim, but
# workspace agents run as root, so direct apt-get is the norm.
_APT_SHIM_SH = f"""\
#!/bin/sh
REAL="/usr/bin/$(basename "$0")"
"$REAL" "$@"; rc=$?
if [ "$1" = install ] && [ $rc -eq 0 ]; then
  shift
  for a in "$@"; do
    case "$a" in -*) ;; *) echo "$a" >> {_APT_MANIFEST} ;; esac
  done
fi
exit $rc
"""


def _persistence_setup_lines() -> list[str]:
    """Bootstrap lines for the persistent dependency layer.

    Idempotent — runs on every container start/recreate. Creates (or
    heals) the venv, wires the env into login shells, then runs the two
    user-facing hooks: ``/workspace/requirements.txt`` (auto pip install)
    and ``/workspace/.augmentum/setup.sh`` (arbitrary project bootstrap,
    runs under ``bash -l`` so it sees the venv PATH). Every line is
    ``|| true``-guarded — a broken hook degrades provisioning (visible in
    provision.log) but never blocks the workspace.
    """
    env_b64 = base64.b64encode(_PERSIST_ENV_SH.encode()).decode()
    apt_shim_b64 = base64.b64encode(_APT_SHIM_SH.encode()).decode()
    venv = _WORKSPACE_VENV
    return [
        f"mkdir -p {_WORKSPACE_NPM_PREFIX} {_WORKSPACE_BIN}",
        f"echo '{env_b64}' | base64 -d > /etc/profile.d/augmentum-persist.sh",
        "chmod 0644 /etc/profile.d/augmentum-persist.sh",
        # apt/apt-get capture shim (both names → same script; resolves the real
        # binary by $0). Written every start so it heals if the volume copy is
        # removed. Chmod +x so it's directly executable on PATH.
        f"echo '{apt_shim_b64}' | base64 -d > {_WORKSPACE_BIN}/apt-get",
        f"cp {_WORKSPACE_BIN}/apt-get {_WORKSPACE_BIN}/apt",
        f"chmod 0755 {_WORKSPACE_BIN}/apt-get {_WORKSPACE_BIN}/apt",
        # Replay captured apt packages BEFORE the pip hook (pip builds may need
        # system libs). Calls the REAL apt-get directly (not the shim) to avoid
        # re-recording. Dedupe via sort -u; whole step is || true so a stale or
        # unavailable package degrades provisioning (visible in provision.log)
        # without ever blocking the workspace.
        (
            f"( [ -f {_APT_MANIFEST} ] && command -v /usr/bin/apt-get >/dev/null 2>&1 "
            "&& /usr/bin/apt-get update -qq 2>/dev/null "
            f"&& sort -u {_APT_MANIFEST} | xargs -r /usr/bin/apt-get install "
            "-y -qq --no-install-recommends ) || true"
        ),
        # Create the venv on first boot; recreate it when an image bump
        # changed the python minor version (the old venv's site-packages
        # would silently vanish from sys.path — a broken venv is worse
        # than an empty one, and requirements.txt repopulates it below).
        (
            "( PYV=$(python3 -c \"import sys; print('python%d.%d' % "
            "sys.version_info[:2])\" 2>/dev/null); "
            f'[ -n "$PYV" ] && [ -d {venv}/lib/$PYV/site-packages ] '
            f"|| python3 -m venv --clear --system-site-packages {venv} ) "
            "|| true"
        ),
        (
            f"( [ -f /workspace/requirements.txt ] && [ -x {venv}/bin/pip ] "
            f"&& {venv}/bin/pip install --no-cache-dir "
            "-r /workspace/requirements.txt ) || true"
        ),
        (
            "( [ -f /workspace/.augmentum/setup.sh ] "
            "&& bash -l /workspace/.augmentum/setup.sh ) || true"
        ),
    ]


def _assemble_keepalive_cmd(setup_lines: list[str]) -> list[str]:
    """Wrap workspace provisioning so it can NEVER kill the container.

    PID 1 of a workspace container IS this command. Historically it was
    ``sh -c "<step> && <step> && … && tail -f /dev/null"`` — the keep-alive
    was the last link of an ``&&`` chain, so ANY provisioning failure (a
    flaky apt mirror, ``python3``/``npm`` not yet on PATH, disk pressure)
    aborted the chain before ``tail -f /dev/null`` and PID 1 exited 127,
    tearing the container down out from under a running agent turn. Every
    subsequent tool call then 409'd "is not running" and the terminal WS
    closed. See the 2026-06-20 workspace-container-death incident.

    The fix decouples liveness from provisioning. The install chain runs
    best-effort inside a subshell whose combined output is captured to
    ``provision.log`` and whose exit code lands in ``provision.exit``; then
    we ``exec tail -f /dev/null`` UNCONDITIONALLY. A failed install now
    yields a live, degraded container (diagnosable via the markers) instead
    of a dead one. ``mkdir`` is hoisted OUT of the redirected subshell so
    the log's target directory exists before the shell opens the redirect.
    """
    setup = " && ".join(setup_lines)
    script = (
        "mkdir -p /workspace/.augmentum 2>/dev/null; "
        f"( {setup} ) > {_PROVISION_LOG} 2>&1; "
        f"echo $? > {_PROVISION_EXIT} 2>/dev/null || true; "
        "exec tail -f /dev/null"
    )
    return ["sh", "-c", script]


_TOOLING_PROFILE_LABEL = "augmentum.tooling_profile"

# Profile catalog + install-line emitter live in augmentum/coder/profiles.py
# (single source of truth — also consumed by the
# /api/coder/tooling-profiles route and the UI).


def _normalize_tooling_profile(value: str | None) -> str:
    # When no profile is explicitly passed, fall through to the configured
    # ``coder_default_tooling_profile`` setting (default "browser") so the
    # caller doesn't have to know about settings. Existing callers passing
    # a real value still win — only empty/None falls through.
    if not value:
        try:
            from augmentum.config import settings as _cfg
            value = getattr(_cfg, "coder_default_tooling_profile", "browser")
        except Exception:  # noqa: BLE001 — degrade to hardcoded default
            value = "browser"
    profile = value.strip().lower() or "browser"
    if not _profiles.has_profile(profile):
        valid = ", ".join(sorted(p.id for p in _profiles.all_profiles()))
        raise ValueError(
            f"Unknown tooling_profile. Expected one of: {valid}"
        )
    return profile


def _tooling_profile_metadata(profile: str) -> dict:
    """Compatibility shim — delegates to profiles.metadata()."""
    return _profiles.metadata(profile)


def _profile_image_is_prebaked(image: str, profile_id: str) -> bool:
    """Return True when the resolved image already contains the profile's
    full install set (i.e. matches the profile's declared ``image_tag``).

    Workspaces that boot on a matching prebake skip the post-create
    install pass — the packages are already there. Unknown profile ids
    return False (we'd rather over-install than skip and leave the
    workspace half-configured).
    """
    try:
        resolved = _profiles.resolve(profile_id)
    except ValueError:
        return False
    return image == resolved.image_tag

_GIT_CREDENTIAL_HELPER = r"""#!/bin/sh
# augmentum-git-credential — proxy to Augmentum server for git auth
WORKSPACE_ID="__WORKSPACE_ID__"
HOST=""
PROTO="https"
while IFS='=' read -r key value; do
  case "$key" in
    host) HOST="$value" ;;
    protocol) PROTO="$value" ;;
  esac
done
# Try host.docker.internal (Docker Desktop), fall back to 172.17.0.1 (Linux Docker)
TOKEN=$(curl -sf "http://host.docker.internal:6100/api/coder/git-credential?host=$HOST&workspace_id=$WORKSPACE_ID" 2>/dev/null)
if [ -z "$TOKEN" ]; then
  TOKEN=$(curl -sf "http://172.17.0.1:6100/api/coder/git-credential?host=$HOST&workspace_id=$WORKSPACE_ID" 2>/dev/null)
fi
[ -n "$TOKEN" ] && printf 'protocol=%s\nhost=%s\nusername=oauth2\npassword=%s\n' "$PROTO" "$HOST" "$TOKEN"
"""

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

# Floor for the workspace PID limit. Below this Docker rejects the
# value AND the workspace genuinely can't run useful work — a vite
# dev server alone burns ~10 PIDs. Above 16K is the cgroup pids.max
# ceiling on most kernels; we cap the setting at the registered max.
_WORKSPACE_PIDS_FLOOR = 256
_WORKSPACE_PIDS_CEILING = 16_384
_WORKSPACE_PIDS_FALLBACK = 1024  # used when settings is unavailable

# One-shot guard so the "host-pivot block disabled" operator warning fires
# once per process instead of on every container create.
_host_pivot_unblocked_warned = False


def _workspace_block_host_pivot() -> bool:
    """Honor ``settings.coder_workspace_block_host_pivot`` with safe default.

    Default True (block) so a fresh install gets the hardened posture
    without operator action. Operators who legitimately need their
    workspace to reach back to host services (Plex/Jellyfin on the host,
    bare-metal services) can flip the setting off via the admin UI.
    """
    try:
        from augmentum.config import settings
        blocked = bool(getattr(settings, "coder_workspace_block_host_pivot", True))
    except Exception:
        return True
    if not blocked:
        global _host_pivot_unblocked_warned
        if not _host_pivot_unblocked_warned:
            _host_pivot_unblocked_warned = True
            log.warning(
                "coder_host_pivot_block_disabled",
                detail=(
                    "coder_workspace_block_host_pivot is OFF — workspace "
                    "containers can resolve host.docker.internal / "
                    "gateway.docker.internal and reach back to host services "
                    "(including the Augmentum proxy). Intended only when a "
                    "workspace legitimately needs host services; re-enable for "
                    "the hardened default."
                ),
            )
    return blocked


_NETWORK_MODES = ("bridge", "none")


def _resolve_workspace_network_mode() -> str:
    """Honor ``settings.coder_workspace_network_mode`` with safe default.

    ``bridge`` (default) keeps internet access for apt/pip/npm/git.
    ``none`` airgaps the workspace entirely — for untrusted code or a
    no-egress policy. Anything else (typo'd setting, out-of-band write)
    falls back to ``bridge`` so a bad value can't brick workspace
    creation. Applies to NEW containers only; an existing workspace
    keeps its mode until recreated.
    """
    try:
        from augmentum.config import settings
        mode = str(
            getattr(settings, "coder_workspace_network_mode", "bridge") or "bridge",
        ).strip().lower()
        return mode if mode in _NETWORK_MODES else "bridge"
    except Exception:
        return "bridge"


def _resolve_workspace_pids_limit(explicit: int | None = None) -> int:
    """Return the effective workspace PidsLimit.

    Precedence:
      1. ``explicit`` (when > 0) — caller override; used by tests and
         legacy paths that already had a value.
      2. ``settings.coder_workspace_pids_limit`` — operator-tunable
         live; validated by ``_TOOL_SETTINGS`` to the (256, 16_384)
         range so an out-of-band write can't break us.
      3. ``_WORKSPACE_PIDS_FALLBACK`` (1024) — last-resort default
         when settings isn't loaded (early-init paths, test harness).
    """
    if explicit is not None and explicit > 0:
        return max(_WORKSPACE_PIDS_FLOOR, min(_WORKSPACE_PIDS_CEILING, int(explicit)))
    try:
        from augmentum.config import settings
        v = int(getattr(settings, "coder_workspace_pids_limit", 0) or 0)
        if v > 0:
            return max(_WORKSPACE_PIDS_FLOOR, min(_WORKSPACE_PIDS_CEILING, v))
    except Exception:
        pass
    return _WORKSPACE_PIDS_FALLBACK


def _parse_memory(memory_str: str) -> int:
    """Parse Docker-style memory string to bytes.

    Examples::

        _parse_memory("2g")   → 2147483648
        _parse_memory("512m") → 536870912
        _parse_memory("1024k") → 1048576
    """
    s = memory_str.strip().lower()
    if s.endswith("g"):
        return int(s[:-1]) * 1024 * 1024 * 1024
    if s.endswith("m"):
        return int(s[:-1]) * 1024 * 1024
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def _resolve_memory_swap(memory_bytes: int) -> int:
    """Total memory+swap allowance for a workspace container.

    Docker's ``MemorySwap`` is the COMBINED memory+swap ceiling. Pinning it
    equal to ``Memory`` (the old behavior) gives ZERO swap headroom, so any
    transient spike past the limit OOM-kills the container — a prime suspect
    for workspaces that die mid-turn during heavy apt/pip/npm/playwright
    provisioning or a big test run (the 2026-06-20 incident's ``python3: not
    found`` was most likely a dpkg child OOM-killed mid-configure). A small
    swap cushion lets a brief overshoot swap instead of dying. Tunable via
    ``coder_workspace_swap_ratio``; 0 restores the no-swap behavior.
    """
    try:
        from augmentum.config import settings as _cfg
        ratio = float(getattr(_cfg, "coder_workspace_swap_ratio", 0.5) or 0.0)
    except Exception:
        ratio = 0.5
    if ratio <= 0:
        return memory_bytes
    return memory_bytes + int(memory_bytes * ratio)


def _parse_ls_output(text: str, path: str) -> list[FileEntry]:
    """Parse output of ``ls -la --time-style=+%s`` into FileEntry objects.

    Expects GNU ls long format lines::

        drwxr-xr-x 2 root root 4096 1711234567 subdir
        -rw-r--r-- 1 root root  512 1711234568 file.txt

    Skips the ``total`` header line and ``.`` / ``..`` entries.
    """
    entries: list[FileEntry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("total"):
            continue
        parts = line.split(None, 8)
        # Need at least: perms, links, user, group, size, timestamp, name
        if len(parts) < 7:
            continue
        perms = parts[0]
        size_str = parts[4]
        timestamp_str = parts[5]
        name = parts[6]
        if name in (".", ".."):
            continue
        is_dir = perms.startswith("d")
        try:
            size = int(size_str)
        except ValueError:
            size = 0
        try:
            modified = float(timestamp_str)
        except ValueError:
            modified = 0.0
        entry_path = f"{path.rstrip('/')}/{name}"
        entries.append(FileEntry(
            name=name,
            path=entry_path,
            is_dir=is_dir,
            size=size,
            modified=modified,
        ))
    return entries


class ContainerManager:
    """Manages Docker workspace containers for Coder mode.

    Parameters
    ----------
    docker:
        An aiodocker ``Docker`` client instance (or mock for tests).
    db:
        An ``aiosqlite.Connection`` for persisting workspace metadata.
        May be ``None`` in tests.
    """

    def __init__(self, docker: object, db: aiosqlite.Connection | None) -> None:
        self._docker = docker
        self._db = db
        # Active execs per workspace, so a user cancel (Ctrl+C / Esc)
        # can kill the underlying docker ``exec`` — otherwise the
        # Python task exits but ``apt-get install`` / ``cargo build``
        # keeps churning inside the container, wasting CPU and
        # confusing the NEXT turn's shell_exec calls (port in use,
        # disk contention, etc.). Entries are removed when the exec
        # finishes naturally; ``cancel_workspace_execs`` walks the
        # set and signals each one.
        self._active_execs: dict[str, set[tuple[object, str]]] = {}
        # Per-workspace activity bump debounce. Process-local cache of
        # the last wall-clock time we touched ``last_active`` for each
        # workspace, used by ``mark_active`` to coalesce same-second
        # writes from polling endpoints. Survives only for the life of
        # the process — the reaper reads ``last_active`` from the DB
        # directly so a restart picks up the persisted value.
        self._activity_last_seen: dict[str, float] = {}
        # Per-workspace throttle for the stale-always_on warning. Logged
        # at most once per process-lifetime + per 24h since sweep_idle
        # runs every minute and we don't want one warning per workspace
        # per tick. Cleared on workspace activity.
        self._stale_always_on_warned: dict[str, float] = {}
        # Short-lived cache of {workspace_id: raw_docker_state} parsed
        # from ``docker containers.list(all=True, label=workspace=true)``.
        # The IPC round-trip dominates ``list_workspaces`` cost (~100-
        # 500ms on a busy daemon); coalescing it across hot-path callers
        # (coder-mode entry, 10s status poll, reconcile_with_docker)
        # drops repeat-fetch latency near zero. Lifecycle methods (start
        # /stop/pause/unpause/create/delete) call
        # ``_invalidate_docker_state_cache`` so user-initiated state
        # changes appear in the very next list. TTL is the safety net
        # for out-of-band changes the manager didn't drive.
        self._docker_state_cache: tuple[float, dict[str, str]] | None = None
        self._docker_state_lock = asyncio.Lock()
        # Resolved-once cache for the volume name backing {data_dir}.
        # None = not yet resolved; "" is a valid resolution (mount
        # disabled). See _resolve_bare_repo_volume.
        self._bare_repo_volume_resolved: str | None = None

    _DOCKER_STATE_CACHE_TTL_S: float = 2.0

    async def _docker_state_map(self) -> dict[str, str]:
        """Return {workspace_id: raw_docker_state} for every labelled
        workspace container, served from a small TTL cache.

        Raw Docker states (running/paused/exited/created/restarting/dead/
        removing) are surfaced as-is; callers map them into the DB
        vocabulary (``_docker_to_db`` in reconcile, the literal state in
        ``list_workspaces`` enrichment) as needed.

        Raises whatever ``docker containers.list`` raises so callers can
        distinguish "Docker unreachable" from "Docker reports no
        containers" — both produce an empty dict but only the former is
        a safety reason to skip writes.
        """
        now = time.monotonic()
        cached = self._docker_state_cache
        if cached and now - cached[0] < self._DOCKER_STATE_CACHE_TTL_S:
            return cached[1]
        async with self._docker_state_lock:
            # Re-check after acquiring — a concurrent caller may have
            # populated the cache while we were queuing.
            cached = self._docker_state_cache
            if cached and time.monotonic() - cached[0] < self._DOCKER_STATE_CACHE_TTL_S:
                return cached[1]
            containers = await self._docker.containers.list(
                all=True,
                filters={"label": [f"{_LABEL_WORKSPACE}=true"]},
            )
            states: dict[str, str] = {}
            for c in containers:
                labels = getattr(c, "_container", {}).get("Labels", {}) or {}
                ws_id = labels.get(_LABEL_ID)
                state = getattr(c, "_container", {}).get("State", "")
                if ws_id:
                    states[ws_id] = state
            self._docker_state_cache = (time.monotonic(), states)
            return states

    def _invalidate_docker_state_cache(self) -> None:
        """Drop the cached Docker state so the next read goes to Docker.

        Called by lifecycle methods that just mutated container state
        (start/stop/pause/unpause/create/delete) — without invalidation,
        a 10s status-poll tick could show the workspace in its prior
        state for up to TTL seconds, which the user reads as "the UI
        is lying about my action."
        """
        self._docker_state_cache = None

    def _workspace_container_name(self, workspace_id: str) -> str:
        return f"augmentum-ws-{workspace_id[:8]}"

    def _workspace_volume_name(self, workspace_id: str) -> str:
        return f"augmentum-ws-{workspace_id[:12]}"

    def _container_has_published_ports(self, details: dict) -> bool:
        net = (details.get("NetworkSettings") or {}).get("Ports") or {}
        return any(bindings for bindings in net.values())

    async def _resolve_workspace_image(
        self, base_image: str, profile_id: str,
    ) -> str:
        """Find a usable image, preferring the profile's prebaked tag.

        Fallback chain (first hit wins):
          1. ``base_image`` if the caller passed a non-default value
             (legacy callers pinning to a custom tag / ``ubuntu:24.04``).
          2. The profile's declared ``image_tag``
             (e.g. ``augmentum-workspace:browser``).
          3. ``augmentum-workspace:standard`` — a v2 tag that the
             post-create install path can layer extras on.
          4. ``augmentum-workspace`` (the v1 generic tag, still around
             on installs that haven't rebuilt yet).
          5. ``ubuntu:24.04`` (bare base — the runtime bootstrap path
             in ``_build_setup_lines`` covers this).

        Logs at INFO when any fallback past the requested image
        triggers, so a slow first boot ("why is it taking 10 minutes?")
        is always explained.
        """
        try:
            resolved = _profiles.resolve(profile_id)
        except ValueError:
            # Caller passed an unknown profile — let normalize raise
            # downstream. Use the v1 generic tag as a safe stand-in.
            resolved = None

        candidates: list[str] = []
        # The v1 default was the generic ``augmentum-workspace`` tag. When
        # we see it (and a profile is resolved), upgrade to the profile's
        # preferred tag so existing callers get prebake speedup without a
        # code change.
        if base_image == "augmentum-workspace":
            if resolved is not None:
                candidates.append(resolved.image_tag)
            candidates.append("augmentum-workspace")
        else:
            candidates.append(base_image)
            if resolved is not None and resolved.image_tag not in candidates:
                candidates.append(resolved.image_tag)
        for fallback in (
            "augmentum-workspace:standard",
            "augmentum-workspace",
            "ubuntu:24.04",
        ):
            if fallback not in candidates:
                candidates.append(fallback)

        requested = candidates[0]
        for candidate in candidates:
            try:
                await self._docker.images.inspect(candidate)
            except Exception:
                continue
            if candidate != requested:
                log.info(
                    "workspace_image_fallback",
                    requested=requested,
                    resolved=candidate,
                    profile=profile_id,
                )
            return candidate
        # Absolute last resort — ubuntu:24.04 wasn't even pullable, but
        # we'll return it anyway; the caller's runtime install path will
        # fail with a clearer error than "no candidate matched".
        return "ubuntu:24.04"

    def _build_workspace_setup_lines(
        self,
        *,
        workspace_id: str,
        name: str,
        actual_image: str,
        git_url: str | None = None,
        git_branch: str | None = None,
        packages: list[str] | None = None,
        tooling_profile: str = "browser",
        initialize_workspace: bool = True,
    ) -> list[str]:
        """Build bootstrap commands for initial workspace start/recreate."""
        from augmentum.coder.prompts import workspace_guide

        tooling_profile = _normalize_tooling_profile(tooling_profile)
        guide_b64 = base64.b64encode(
            workspace_guide(tooling_profile).encode()
        ).decode()
        profile_b64 = base64.b64encode(
            json.dumps(
                _tooling_profile_metadata(tooling_profile),
                separators=(",", ":"),
            ).encode()
        ).decode()
        helper_script = _GIT_CREDENTIAL_HELPER.strip().replace(
            "__WORKSPACE_ID__", workspace_id,
        )
        cred_b64 = base64.b64encode(helper_script.encode()).decode()

        setup_lines = [
            "mkdir -p /workspace/.augmentum",
            f"echo '{guide_b64}' | base64 -d > /workspace/.augmentum/workspace.md",
            f"echo '{profile_b64}' | base64 -d > /workspace/.augmentum/tooling-profile.json",
            f"echo '{cred_b64}' | base64 -d > /usr/local/bin/git-credential-augmentum",
            "chmod +x /usr/local/bin/git-credential-augmentum",
        ]

        if actual_image == "ubuntu:24.04":
            setup_lines.extend([
                "export DEBIAN_FRONTEND=noninteractive",
                (
                    # Tolerate a flaky `apt-get update` (transient mirror/DNS):
                    # `;` not `&&` so a stale-but-cached index can still install.
                    "apt-get update -qq 2>/dev/null; apt-get install -y -qq "
                    f"--no-install-recommends {_FALLBACK_APT_PACKAGES}"
                ),
                "ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true",
                "rm -f /usr/lib/python3*/EXTERNALLY-MANAGED 2>/dev/null || true",
                (
                    "python3 -m pip install --no-cache-dir --ignore-installed "
                    f"{_FALLBACK_PIP_PACKAGES}"
                ),
                f"npm install -g {_FALLBACK_NPM_PACKAGES}",
                "ln -sf /usr/bin/fdfind /usr/local/bin/fd 2>/dev/null || true",
            ])

        # Profile-driven install lines come from the catalog so adding a
        # new profile is a single data-model change, not edits here.
        # Skipped when the matched image is the profile's prebake — the
        # packages are already baked in, and re-running pip/apt would
        # just waste ~30s reaffirming installed packages.
        if not _profile_image_is_prebaked(actual_image, tooling_profile):
            setup_lines.extend(_profiles.emit_install_lines(tooling_profile))

        setup_lines.append(
            "git config --global credential.helper 'git-credential-augmentum'"
        )
        setup_lines.append(
            f"git config --global user.name {shlex.quote(_GIT_AUTHOR_NAME)}"
        )
        setup_lines.append(
            f"git config --global user.email {shlex.quote(_GIT_AUTHOR_EMAIL)}"
        )
        # Trust every repo path inside the container. The bare repo at
        # /augmentum-bare and the /workspace tree are bind-mounted from
        # the host and owned by a different UID than the in-container git
        # user, so git's CVE-2022-24765 "dubious ownership" guard
        # otherwise aborts the seed clone (leaving the workspace EMPTY)
        # and every later checkpoint push/fetch to origin. Blanket-trust
        # is safe here: the container is ephemeral and fully sandboxed.
        setup_lines.append(
            "git config --global --add safe.directory '*'"
        )
        setup_lines.append(
            f"echo 'export PS1=\"\\[\\033[1;32m\\]{name}\\[\\033[0m\\]:\\[\\033[1;34m\\]\\w\\[\\033[0m\\]\\$ \"' >> /root/.bashrc"
        )

        if initialize_workspace:
            if git_url:
                # Escape user-supplied URL/branch for safe inclusion in bash -c.
                # Without shlex.quote, a value containing `'` would break out
                # of the single-quoted context and execute arbitrary commands.
                branch_flag = (
                    f"-b {shlex.quote(git_branch)}" if git_branch else ""
                )
                url_arg = shlex.quote(git_url)
                setup_lines.extend([
                    "cd /tmp",
                    f"( export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true; "
                    f"git clone --depth 1 {branch_flag} {url_arg} /tmp/_clone "
                    f"> /workspace/.augmentum/clone.log 2>&1 "
                    f"&& cp -a /tmp/_clone/. /workspace/ "
                    f"&& rm -rf /tmp/_clone "
                    f"&& touch /workspace/.augmentum/clone_ok ) "
                    f"|| touch /workspace/.augmentum/clone_failed",
                    "cd /workspace",
                    "[ -f /workspace/.augmentum/clone_failed ] && git init -q 2>/dev/null || true",
                    (
                        "[ -f /workspace/.augmentum/clone_failed ] && git add -A && "
                        f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                        f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                        "commit -q -m 'Initial workspace state' --allow-empty "
                        "2>/dev/null || true"
                    ),
                ])
            else:
                setup_lines.extend([
                    "cd /workspace && git init -q 2>/dev/null || true",
                    (
                        "cd /workspace && git add -A && "
                        f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                        f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                        "commit -q -m 'Initial workspace state' --allow-empty "
                        "2>/dev/null || true"
                    ),
                ])

        if packages:
            pkg_list = " ".join(packages)
            setup_lines.append(
                "apt-get update -qq 2>/dev/null; "
                f"apt-get install -y -qq --no-install-recommends {pkg_list}"
            )

        # Persistent dependency layer LAST (after the optional git clone so
        # a cloned requirements.txt is installable on the very first boot).
        setup_lines.extend(_persistence_setup_lines())

        setup_lines.append("touch /workspace/.augmentum/ready")
        return setup_lines

    def _build_workspace_container_config(
        self,
        *,
        workspace_id: str,
        name: str,
        actual_image: str,
        cmd: list[str],
        cpu: float,
        memory_bytes: int,
        pids: int,
        publish_ports: bool,
        tooling_profile: str = "browser",
        lan_accessible: bool = False,
    ) -> dict:
        # Resolve the PID limit through the live setting so an
        # operator can tune without code change. Negative / zero
        # values flow through to the settings default; explicit
        # positive values win for tests + legacy callers.
        pids = _resolve_workspace_pids_limit(pids)
        tooling_profile = _normalize_tooling_profile(tooling_profile)
        labels = {
            _LABEL_WORKSPACE: "true",
            _LABEL_NAME: name,
            _LABEL_ID: workspace_id,
            _TOOLING_PROFILE_LABEL: tooling_profile,
        }

        host_ip = "0.0.0.0" if lan_accessible else "127.0.0.1"
        if publish_ports:
            host_config_ports = {
                "ExposedPorts": {f"{p}/tcp": {} for p in _DEV_PORTS},
                "PortBindings": {
                    f"{p}/tcp": [{"HostIp": host_ip, "HostPort": ""}]
                    for p in _DEV_PORTS
                },
            }
        else:
            host_config_ports = {}

        return {
            "Image": actual_image,
            "Cmd": cmd,
            "Labels": labels,
            "Tty": True,
            "OpenStdin": True,
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            **({"ExposedPorts": host_config_ports["ExposedPorts"]}
               if publish_ports else {}),
            "HostConfig": {
                "CapDrop": _CAP_DROP,
                "CapAdd": _resolve_cap_add(tooling_profile),
                "SecurityOpt": ["no-new-privileges:true"],
                "RestartPolicy": {"Name": "on-failure", "MaximumRetryCount": 2},
                "NanoCpus": int(cpu * 1e9),
                "Memory": memory_bytes,
                "MemorySwap": _resolve_memory_swap(memory_bytes),
                "PidsLimit": pids,
                "NetworkMode": _resolve_workspace_network_mode(),
                "Binds": [f"{self._workspace_volume_name(workspace_id)}:/workspace"],
                **(
                    {"ExtraHosts": [
                        "host.docker.internal:0.0.0.0",
                        "gateway.docker.internal:0.0.0.0",
                    ]}
                    if _workspace_block_host_pivot() else {}
                ),
                **({"PortBindings": host_config_ports["PortBindings"]}
                   if publish_ports else {}),
            },
        }

    # ------------------------------------------------------------------
    # Workspace lifecycle
    # ------------------------------------------------------------------

    async def create_workspace(
        self,
        name: str,
        base_image: str = "augmentum-workspace",
        packages: list[str] | None = None,
        git_url: str | None = None,
        git_branch: str | None = None,
        cpu: float = 2.0,
        memory: str = "2g",
        # PID limit (cgroup pids.max). 256 (pre-2026-05-31 default)
        # saturates on workspaces that run a dev server (vite + esbuild
        # = ~10 PIDs) plus any accumulation of orphaned bash --login
        # processes from shell_exec ``cmd &`` backgrounding. The
        # saturation symptom is ``procReady not received`` on any new
        # docker exec — runc can't fork inside a maxed pids.max cgroup
        # even though the container reports "running".
        #
        # ``-1`` (or any value <= 0) means "read from settings.coder_workspace_pids_limit"
        # so an operator can tune without redeploying. Direct ints win
        # over the setting — tests and explicit overrides stay in
        # control. See [[project_coder_workspace_pids_limit]].
        pids: int = -1,
        publish_ports: bool = False,
        tooling_profile: str = "browser",
        *,
        user_id: str = "",
        kind: str = "regular",
        project_id: str = "",
        lan_accessible: bool = False,
    ) -> ContainerInfo:
        """Create and start a new workspace container.

        Uses the pre-baked ``augmentum-workspace`` image (Dockerfile.workspace)
        which has dev tools, language runtimes, and AI coding CLI prerequisites
        already installed.  Falls back to ``ubuntu:24.04`` with runtime install
        if the workspace image isn't available.

        ``project_id`` (Phase 1 / PR-1.2): when provided alongside a
        non-empty ``user_id``, the checkout is linked to a Project. The
        Project's bare repo on host (``{data_dir}/projects/{user_id}/
        {project_id}.git/``) is ensured, then bind-mounted at
        ``/augmentum-bare`` inside the container via a Docker named-volume
        subpath. The setup script clones from there instead of running
        ``git init`` locally — so a recycled container inherits the
        prior checkout's history. Empty ``project_id`` keeps legacy
        behaviour (a new bare repo is created lazily under an
        auto-generated Project, but not yet bind-mounted).

        Returns a :class:`ContainerInfo` with ``status="running"``.
        """
        workspace_id = str(uuid.uuid4())
        container_name = f"augmentum-ws-{workspace_id[:8]}"
        memory_bytes = _parse_memory(memory)
        tooling_profile = _normalize_tooling_profile(tooling_profile)

        # Phase 1 / PR-1.2: ensure the durable Project + bare repo exist
        # before the container starts. The bare repo is the source of
        # truth that survives container recycle; the workspace volume
        # is a disposable checkout.
        bare_repo_subpath = ""
        if user_id and self._db is not None:
            try:
                project_id = await self._ensure_project_for_checkout(
                    project_id=project_id,
                    user_id=user_id,
                    name=name,
                )
                if project_id:
                    bare_repo_subpath = (
                        f"projects/{user_id}/{project_id}.git"
                    )
            except Exception:
                # Substrate failure shouldn't block container creation
                # in Phase 1 — log and fall back to legacy git-init.
                log.warning(
                    "project_substrate_unavailable_falling_back",
                    user_id=user_id, error="ensure_project_failed",
                    exc_info=True,
                )
                project_id = ""

        # Resolve image with a profile-aware fallback chain. ``base_image``
        # is honored when the caller supplied a non-default value; otherwise
        # the profile's preferred tag wins. See _resolve_workspace_image.
        actual_image = await self._resolve_workspace_image(
            base_image, tooling_profile,
        )

        # Write workspace guide for the agent
        # Encode file contents as base64 to avoid heredoc/quoting issues in &&-chained commands
        import base64

        from augmentum.coder.prompts import workspace_guide
        guide_b64 = base64.b64encode(
            workspace_guide(tooling_profile).encode()
        ).decode()
        profile_b64 = base64.b64encode(
            json.dumps(
                _tooling_profile_metadata(tooling_profile),
                separators=(",", ":"),
            ).encode()
        ).decode()
        helper_script = _GIT_CREDENTIAL_HELPER.strip().replace(
            "__WORKSPACE_ID__", workspace_id,
        )
        cred_b64 = base64.b64encode(helper_script.encode()).decode()

        setup_lines = [
            "mkdir -p /workspace/.augmentum",
            f"echo '{guide_b64}' | base64 -d > /workspace/.augmentum/workspace.md",
            f"echo '{profile_b64}' | base64 -d > /workspace/.augmentum/tooling-profile.json",
            f"echo '{cred_b64}' | base64 -d > /usr/local/bin/git-credential-augmentum",
            "chmod +x /usr/local/bin/git-credential-augmentum",
        ]

        if actual_image == "ubuntu:24.04":
            # Fallback: bootstrap a guide-compatible dev stack at runtime.
            # Slower than the prebaked image, but it keeps env_info and the
            # workspace guide aligned so the model doesn't assume tools that
            # are absent.
            setup_lines.extend([
                "export DEBIAN_FRONTEND=noninteractive",
                (
                    # Tolerate a flaky `apt-get update` (transient mirror/DNS):
                    # `;` not `&&` so a stale-but-cached index can still install.
                    "apt-get update -qq 2>/dev/null; apt-get install -y -qq "
                    f"--no-install-recommends {_FALLBACK_APT_PACKAGES}"
                ),
                "ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true",
                "rm -f /usr/lib/python3*/EXTERNALLY-MANAGED 2>/dev/null || true",
                (
                    "python3 -m pip install --no-cache-dir --ignore-installed "
                    f"{_FALLBACK_PIP_PACKAGES}"
                ),
                f"npm install -g {_FALLBACK_NPM_PACKAGES}",
                "ln -sf /usr/bin/fdfind /usr/local/bin/fd 2>/dev/null || true",
            ])

        # Profile-driven install lines come from the catalog so adding a
        # new profile is a single data-model change, not edits here.
        # Skipped when the matched image is the profile's prebake — the
        # packages are already baked in, and re-running pip/apt would
        # just waste ~30s reaffirming installed packages.
        if not _profile_image_is_prebaked(actual_image, tooling_profile):
            setup_lines.extend(_profiles.emit_install_lines(tooling_profile))

        # Git config must come after package install (git may not exist on base ubuntu)
        setup_lines.append(
            "git config --global credential.helper 'git-credential-augmentum'"
        )
        setup_lines.append(
            f"git config --global user.name {shlex.quote(_GIT_AUTHOR_NAME)}"
        )
        setup_lines.append(
            f"git config --global user.email {shlex.quote(_GIT_AUTHOR_EMAIL)}"
        )
        # Trust every repo path inside the container. The bare repo at
        # /augmentum-bare and the /workspace tree are bind-mounted from
        # the host and owned by a different UID than the in-container git
        # user, so git's CVE-2022-24765 "dubious ownership" guard
        # otherwise aborts the seed clone (leaving the workspace EMPTY)
        # and every later checkpoint push/fetch to origin. Blanket-trust
        # is safe here: the container is ephemeral and fully sandboxed.
        setup_lines.append(
            "git config --global --add safe.directory '*'"
        )

        setup_lines.extend([
            # Custom prompt showing workspace name
            f"echo 'export PS1=\"\\[\\033[1;32m\\]{name}\\[\\033[0m\\]:\\[\\033[1;34m\\]\\w\\[\\033[0m\\]\\$ \"' >> /root/.bashrc",
        ])

        # Phase 1 / PR-1.2: bare-repo clone path. When a Project bare
        # repo is mounted at /augmentum-bare, clone from it instead of
        # `git init`-ing a fresh repo. This is how a recycled container
        # inherits prior history. Falls through to the legacy branches
        # below if the mount isn't present (older Docker without
        # VolumeOptions.Subpath, or no project link).
        if bare_repo_subpath and not git_url:
            setup_lines.extend([
                # Clone the bare repo into a temp dir. Cloning an empty
                # bare repo succeeds but creates an empty working tree
                # with HEAD on the unborn default branch — that's the
                # correct state for a brand-new project.
                "( git clone /augmentum-bare /tmp/_proj_seed "
                "> /workspace/.augmentum/bare_clone.log 2>&1 "
                "&& cp -a /tmp/_proj_seed/. /workspace/ "
                "&& rm -rf /tmp/_proj_seed "
                "&& touch /workspace/.augmentum/bare_clone_ok ) "
                "|| touch /workspace/.augmentum/bare_clone_failed",
                # Fallback if the clone path errored (mount missing /
                # corrupted bare repo): init locally so checkpoints
                # still work. Bare repo will be reseeded on the first
                # successful push.
                "[ -f /workspace/.augmentum/bare_clone_failed ] && "
                "cd /workspace && git init -q 2>/dev/null || true",
                # Ensure origin points at the bare repo regardless of
                # which branch we took (clone sets it; fallback init
                # doesn't). `git remote set-url` is idempotent; the
                # `|| add` covers the never-set case.
                "cd /workspace && (git remote set-url origin "
                "/augmentum-bare 2>/dev/null || "
                "git remote add origin /augmentum-bare 2>/dev/null) || true",
                # If we cloned a totally-empty bare repo, the working
                # tree has no commits. Make an empty initial commit so
                # auto-checkpoint diffs have a base and push works.
                (
                    "cd /workspace && [ -z \"$(git rev-parse --verify HEAD 2>/dev/null)\" ] && "
                    "git add -A && "
                    f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                    f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                    "commit -q -m 'Initial workspace state' --allow-empty "
                    "2>/dev/null || true"
                ),
            ])
        elif git_url:
            # Clone into a temp dir then move contents into /workspace
            # (git clone refuses to clone into a non-empty directory).
            # GIT_TERMINAL_PROMPT=0 and GIT_ASKPASS=/bin/true make auth
            # failures return immediately instead of blocking forever on
            # an unattached TTY when the credential helper returns
            # nothing (the old behavior left the container's primary
            # process stuck, so the ready marker was never written and
            # the frontend polled until its 120s timeout).
            # Clone output is captured to clone.log so the frontend can
            # surface errors, and we use `;` + explicit markers instead
            # of `&&` so we always reach the ready marker even on
            # failure.
            # Escape user-supplied URL/branch for safe inclusion in bash -c.
            # Without shlex.quote, a value containing `'` would break out
            # of the single-quoted context and execute arbitrary commands.
            branch_flag = (
                f"-b {shlex.quote(git_branch)}" if git_branch else ""
            )
            url_arg = shlex.quote(git_url)
            setup_lines.extend([
                "cd /tmp",
                f"( export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true; "
                f"git clone --depth 1 {branch_flag} {url_arg} /tmp/_clone "
                f"> /workspace/.augmentum/clone.log 2>&1 "
                f"&& cp -a /tmp/_clone/. /workspace/ "
                f"&& rm -rf /tmp/_clone "
                f"&& touch /workspace/.augmentum/clone_ok ) "
                f"|| touch /workspace/.augmentum/clone_failed",
                "cd /workspace",
                # If clone failed, init an empty repo so auto-checkpoints still work
                "[ -f /workspace/.augmentum/clone_failed ] && git init -q 2>/dev/null || true",
                (
                    "[ -f /workspace/.augmentum/clone_failed ] && git add -A && "
                    f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                    f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                    "commit -q -m 'Initial workspace state' --allow-empty "
                    "2>/dev/null || true"
                ),
            ])
        else:
            # No clone — init an empty repo for auto-checkpointing
            setup_lines.extend([
                "cd /workspace && git init -q 2>/dev/null || true",
                (
                    "cd /workspace && git add -A && "
                    f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                    f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                    "commit -q -m 'Initial workspace state' --allow-empty "
                    "2>/dev/null || true"
                ),
            ])

        if packages:
            pkg_list = " ".join(packages)
            setup_lines.append(f"apt-get update -qq 2>/dev/null; apt-get install -y -qq --no-install-recommends {pkg_list}")

        # Persistent dependency layer (venv-in-volume + requirements.txt /
        # setup.sh hooks). NOTE: this create path duplicates most of
        # _build_workspace_setup_lines (used by the recreate paths) — any
        # bootstrap change must land in BOTH until they're unified.
        setup_lines.extend(_persistence_setup_lines())

        # Mark workspace as ready. Unconditional so a clone failure
        # (which now writes clone_failed instead of propagating its exit
        # code) doesn't block the ready probe — the frontend inspects
        # clone_ok/clone_failed to distinguish success from failure.
        setup_lines.append("touch /workspace/.augmentum/ready")

        # Keep-alive is decoupled from provisioning: a failed install can no
        # longer kill PID 1 and tear the container down mid-turn. See
        # _assemble_keepalive_cmd.
        cmd = _assemble_keepalive_cmd(setup_lines)

        labels = {
            _LABEL_WORKSPACE: "true",
            _LABEL_NAME: name,
            _LABEL_ID: workspace_id,
            _TOOLING_PROFILE_LABEL: tooling_profile,
        }

        # Named volume for persistent /workspace — survives container restarts
        volume_name = f"augmentum-ws-{workspace_id[:12]}"

        host_ip = "0.0.0.0" if lan_accessible else "127.0.0.1"
        if publish_ports:
            host_config_ports = {
                "ExposedPorts": {f"{p}/tcp": {} for p in _DEV_PORTS},
                "PortBindings": {
                    f"{p}/tcp": [{"HostIp": host_ip, "HostPort": ""}]
                    for p in _DEV_PORTS
                },
            }
        else:
            host_config_ports = {}

        # Phase 1 / PR-1.2: bind the per-project bare repo into the
        # workspace via a Docker volume Subpath mount. Subpath isolates
        # the workspace to a single project's repo — without it, RW
        # access to the whole augmentum_data volume would let user code
        # touch the application DB, image cache, etc. Subpath requires
        # Docker Engine >= 25.0 (Jan 2024); on older engines this Mount
        # entry will fail container creation and the setup script's
        # bare_clone_failed fallback takes over.
        mounts: list[dict] = []
        if bare_repo_subpath:
            bare_volume = await self._resolve_bare_repo_volume()
            if bare_volume:
                mounts.append({
                    "Type": "volume",
                    "Source": bare_volume,
                    "Target": "/augmentum-bare",
                    "ReadOnly": False,
                    "VolumeOptions": {"Subpath": bare_repo_subpath},
                })

        config = {
            "Image": actual_image,
            "Cmd": cmd,
            "Labels": labels,
            "Tty": True,
            "OpenStdin": True,
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            **({"ExposedPorts": host_config_ports["ExposedPorts"]}
                if publish_ports else {}),
            "HostConfig": {
                "CapDrop": _CAP_DROP,
                "CapAdd": _resolve_cap_add(tooling_profile),
                "SecurityOpt": ["no-new-privileges:true"],
                # Self-heal genuine crashes (OOM kill, daemon restart) that
                # exit PID 1 between turns, when no exec is around to trigger
                # the revive-on-409 path. Bounded so a hard-looping container
                # (e.g. immediate re-OOM) can't thrash. Docker does NOT restart
                # on an explicit `docker stop`, so this never fights the idle
                # reaper. With the keep-alive decoupled, a provisioning failure
                # no longer exits PID 1, so this only fires on true crashes.
                "RestartPolicy": {"Name": "on-failure", "MaximumRetryCount": 2},
                "NanoCpus": int(cpu * 1e9),
                "Memory": memory_bytes,
                # Swap cushion (coder_workspace_swap_ratio) so a transient
                # spike swaps instead of OOM-killing the container mid-turn.
                "MemorySwap": _resolve_memory_swap(memory_bytes),
                "PidsLimit": pids,
                # "bridge" (default) or "none" via the
                # coder_workspace_network_mode setting — see
                # _resolve_workspace_network_mode for the contract.
                "NetworkMode": _resolve_workspace_network_mode(),
                "Binds": [f"{volume_name}:/workspace"],
                **({"Mounts": mounts} if mounts else {}),
                # Host-pivot neutralisation. Mirrors the build_create_config
                # block above — without this the workspace can curl the
                # Augmentum proxy via host.docker.internal / gateway.docker
                # .internal and pivot back into the host. Gated by the same
                # _workspace_block_host_pivot() setting.
                **(
                    {"ExtraHosts": [
                        "host.docker.internal:0.0.0.0",
                        "gateway.docker.internal:0.0.0.0",
                    ]}
                    if _workspace_block_host_pivot() else {}
                ),
                **({"PortBindings": host_config_ports["PortBindings"]}
                    if publish_ports else {}),
            },
        }

        log.info(
            "creating_workspace",
            name=name,
            workspace_id=workspace_id,
            image=actual_image,
            prebaked=actual_image != "ubuntu:24.04",
            tooling_profile=tooling_profile,
        )

        try:
            container = await self._docker.containers.run(
                config=config,
                name=container_name,
            )
        except Exception as exc:
            if not mounts:
                raise
            # The bare-repo Subpath mount is the only optional piece of
            # this config. Wrong volume name, pre-25.0 engine (no
            # Subpath), or a missing repo dir all fail at create/start —
            # and the Phase 1 contract is that substrate failure must
            # not block the workspace. Retry once without the mount;
            # the setup script's bare_clone_failed branch git-inits a
            # local repo instead.
            log.warning(
                "bare_repo_mount_failed_retrying_without",
                workspace_id=workspace_id,
                volume=mounts[0].get("Source"),
                subpath=bare_repo_subpath,
                error=str(exc),
            )
            # docker run = create + start; a start-time mount failure
            # leaves the named container behind, so clear it or the
            # retry hits a 409 name conflict. 404 here just means the
            # failure happened at create time — nothing to clean up.
            try:
                stale = await self._docker.containers.get(container_name)
                await stale.delete(force=True)
            except Exception:
                log.debug("no_stale_container_to_clean", name=container_name)
            config["HostConfig"].pop("Mounts", None)
            container = await self._docker.containers.run(
                config=config,
                name=container_name,
            )

        created_at = time.time()
        info = ContainerInfo(
            id=workspace_id,
            name=name,
            container_id=container.id,
            status="running",
            template_id=None,
            git_url=git_url,
            created_at=created_at,
            last_active=created_at,
            resources_cpu=cpu,
            resources_memory=memory,
            tooling_profile=tooling_profile,
            user_id=user_id,
            kind=(kind or "regular").strip() or "regular",
            project_id=project_id,
            lan_accessible=lan_accessible,
        )

        if self._db is not None:
            await self._persist_workspace(info)
        self._invalidate_docker_state_cache()

        log.info(
            "workspace container started",
            workspace_id=workspace_id,
            container_id=container.id,
        )
        return info

    async def list_workspaces(self) -> list[ContainerInfo]:
        """Return all known workspaces, enriched with live Docker status.

        Reads from the database first, then cross-references running containers
        to update the ``status`` field. When a DB row says ``running`` /
        ``paused`` but the labelled container is missing from Docker entirely
        (``docker rm`` after a crash / manual prune / image rebuild), the row
        is reconciled in place to ``status='stopped'`` — without this the
        ``always_on=1`` exemption would lock the row in a phantom-running
        state forever.
        """
        if self._db is None:
            return []

        rows = await self._db.execute_fetchall(
            "SELECT id, name, container_id, status, template_id, git_url, "
            "created_at, last_active, resources_cpu, resources_memory, "
            "safeguards_enabled, tooling_profile, "
            "kind, bug_finder_verifier_model, project_id, "
            "planning_mode, always_on, lan_accessible "
            "FROM project_checkouts WHERE archived_at IS NULL "
            "ORDER BY created_at DESC"
        )

        workspaces = [
            ContainerInfo(
                id=row[0],
                name=row[1],
                container_id=row[2],
                status=row[3],
                template_id=row[4],
                git_url=row[5],
                created_at=row[6],
                last_active=row[7],
                resources_cpu=row[8],
                resources_memory=row[9],
                safeguards_enabled=bool(row[10]) if row[10] is not None else True,
                tooling_profile=row[11] or "browser",
                kind=row[12] or "regular",
                bug_finder_verifier_model=row[13] or "",
                project_id=row[14] or "",
                planning_mode=row[15] or "default",
                always_on=bool(row[16]) if row[16] is not None else False,
                lan_accessible=bool(row[17]) if row[17] is not None else False,
            )
            for row in rows
        ]

        # Enrich with live Docker status. The state map is served from
        # a short TTL cache (see ``_docker_state_map``) so hot polling
        # paths share one ``docker containers.list`` round-trip.
        try:
            live_ids = await self._docker_state_map()
        except Exception as exc:
            log.warning("workspace_docker_status_failed", error=str(exc))
            return workspaces
        drifted: list[str] = []
        for ws in workspaces:
            if ws.id in live_ids:
                ws.status = live_ids[ws.id]
            elif ws.container_id:
                # Container existed in DB but is gone from Docker
                # (shutdown / pruned / external docker rm). If the DB
                # row still claims running/paused, that's drift — record
                # it for the batch writeback below.
                if ws.status in ("running", "paused"):
                    drifted.append(ws.id)
                ws.status = "stopped"

        if drifted and self._db is not None:
            try:
                qmarks = ",".join("?" * len(drifted))
                await self._db.execute(
                    f"UPDATE project_checkouts SET status='stopped' "
                    f"WHERE id IN ({qmarks})",
                    drifted,
                )
                await self._db.commit()
                log.info(
                    "workspace_drift_reconciled",
                    count=len(drifted),
                    ids=[d[:8] for d in drifted],
                )
            except Exception:
                log.warning("workspace_drift_writeback_failed", exc_info=True)

        return workspaces

    async def reconcile_with_docker(self) -> dict[str, int]:
        """Reconcile DB lifecycle state with live Docker container state.

        Designed to run once at server startup, before the idle watcher
        scans and before any route serves a workspace request. Three
        drift classes that motivated this:

          * **Daemon-restart cohort** — Docker Desktop / dockerd restart
            SIGKILL'd workspace containers but ``status='running'`` rows
            stayed behind. The next sweep tried to ``pause`` them, got
            409s, and (pre-fix) logged false-positive "reaped" lines on
            every tick.
          * **Out-of-band state changes** — operator (or another tool)
            ran ``docker pause/stop/unpause`` directly. DB and Docker
            diverge until the next user action surfaces the mismatch.
          * **Orphaned containers** — a workspace container is alive in
            Docker but no DB row references it (row deleted manually,
            DB restored from backup, etc.). We log + count these but do
            not auto-remove — the bind-mounted workspace volume may
            still hold user data the operator wants to recover.

        Returns counters ``{"reconciled": N, "orphans": N, "ok": N}``.
        Safe to call when Docker is unreachable (returns zeros, logs a
        warning) so startup isn't blocked by a flaky daemon.
        """
        if self._db is None:
            return {"reconciled": 0, "orphans": 0, "ok": 0}

        # Map Docker's lifecycle vocabulary (created/restarting/running/
        # paused/exited/dead/removing) into the three states the DB
        # tracks. Anything that isn't actively running or paused is
        # ``stopped`` for our purposes — the user can ``start`` a row
        # in any of those states.
        def _docker_to_db(state: str) -> str:
            s = (state or "").lower()
            if s == "running":
                return "running"
            if s == "paused":
                return "paused"
            return "stopped"

        # Reconcile reads fresh truth: drop the cache before reading so
        # we never reconcile against a TTL-stale snapshot at startup.
        self._invalidate_docker_state_cache()
        try:
            raw = await self._docker_state_map()
        except Exception as exc:
            log.warning(
                "workspace_reconcile_docker_unreachable",
                error=str(exc)[:160],
            )
            return {"reconciled": 0, "orphans": 0, "ok": 0}
        live: dict[str, str] = {ws_id: _docker_to_db(s) for ws_id, s in raw.items()}

        try:
            rows = await self._db.execute_fetchall(
                "SELECT id, status FROM project_checkouts WHERE archived_at IS NULL"
            )
        except Exception:
            log.warning("workspace_reconcile_db_read_failed", exc_info=True)
            return {"reconciled": 0, "orphans": 0, "ok": 0}

        db_ids = {row[0] for row in rows}
        # Group updates by target status so we can issue one UPDATE per
        # bucket instead of one per row. Mirrors the batch pattern in
        # ``list_workspaces`` above.
        by_status: dict[str, list[str]] = {}
        ok = 0
        for ws_id, db_status in rows:
            truth = live.get(ws_id, "stopped")
            if db_status == truth:
                ok += 1
                continue
            by_status.setdefault(truth, []).append(ws_id)

        orphans = [c_id for c_id in live if c_id not in db_ids]
        if orphans:
            log.warning(
                "workspace_reconcile_orphans_detected",
                count=len(orphans),
                ids=[o[:8] for o in orphans],
            )

        reconciled_total = 0
        if by_status:
            try:
                for status, ids in by_status.items():
                    qmarks = ",".join("?" * len(ids))
                    await self._db.execute(
                        f"UPDATE project_checkouts SET status=? "
                        f"WHERE id IN ({qmarks})",
                        [status, *ids],
                    )
                    reconciled_total += len(ids)
                await self._db.commit()
            except Exception:
                log.warning(
                    "workspace_reconcile_writeback_failed", exc_info=True
                )
                return {
                    "reconciled": 0,
                    "orphans": len(orphans),
                    "ok": ok,
                }

        if reconciled_total or orphans:
            log.info(
                "workspace_reconcile_complete",
                reconciled=reconciled_total,
                orphans=len(orphans),
                ok=ok,
            )

        return {
            "reconciled": reconciled_total,
            "orphans": len(orphans),
            "ok": ok,
        }

    async def start(self, workspace_id: str) -> ContainerInfo:
        """Start a stopped (or paused) workspace container.

        Paused containers thaw via ``unpause`` instead of ``start`` —
        sub-second instead of a full recreate. Stopped containers go
        through the normal ``start`` path.

        If the referenced container is gone from Docker (manual rm,
        prune after image rebuild, host wipe), the DB row's
        ``container_id`` is cleared + status flipped to ``stopped``
        before re-raising — so a follow-up create path doesn't trip
        over a dangling reference.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        try:
            container = await self._docker.containers.get(info.container_id)
        except Exception as exc:
            if _is_container_gone(exc):
                await self._reconcile_to_stopped(
                    workspace_id, clear_container_id=True,
                )
                log.warning(
                    "workspace_start_container_gone",
                    workspace_id=workspace_id,
                    container_id=info.container_id,
                )
                raise RuntimeError(
                    "Workspace container has been removed; recreate the workspace."
                ) from exc
            raise

        # Branch on live Docker state: paused needs unpause, stopped
        # needs start. The DB ``status`` field is authoritative for
        # bookkeeping but Docker is authoritative for what the API
        # call actually needs to be.
        try:
            details = await container.show()
            state = (details.get("State") or {}).get("Status", "")
        except Exception:
            state = ""
        if state == "paused" or info.status == "paused":
            await container.unpause()
        else:
            await container.start()
        info.status = "running"
        info.last_active = time.time()

        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET status=?, last_active=? WHERE id=?",
                ("running", info.last_active, workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()

        log.info("workspace started", workspace_id=workspace_id)
        return info

    async def check_pids_pressure(
        self, *, workspace_id: str | None = None,
    ) -> list[dict]:
        """Inspect workspace PID counts and emit early-warning logs.

        Walks running workspace containers and compares live PID count
        vs. ``pids.max`` cgroup limit. When the ratio crosses
        ``coder_workspace_pids_warn_pct`` (default 0.80) the watchdog
        logs ``coder.workspace_pids_pressure`` with the workspace id +
        ratio so an operator can react BEFORE saturation wedges runc.

        Returns one dict per inspected workspace with keys ``workspace_id``,
        ``pid_count``, ``pid_limit``, ``ratio``, ``warned``. Callers can
        ignore the return value — the side effect (logging) is the
        intended surface. Best-effort: docker errors are absorbed.

        Periodic invocation is the caller's job (a cron-style task in
        the proxy's lifespan, or an admin route). Cheap enough to run
        every 30s — one ``docker inspect`` per workspace.
        """
        try:
            from augmentum.config import settings
            warn_pct = float(getattr(settings, "coder_workspace_pids_warn_pct", 0.80) or 0)
        except Exception:
            warn_pct = 0.80
        warn_pct = max(0.0, min(0.99, warn_pct))
        if warn_pct <= 0:
            return []

        results: list[dict] = []
        try:
            containers = await self._docker.containers.list(
                all=False,  # running only — stopped containers have no PIDs
                filters={"label": [f"{_LABEL_WORKSPACE}=true"]},
            )
        except Exception as exc:
            log.debug("workspace_pids_check_list_failed", error=str(exc)[:160])
            return results

        for c in containers:
            try:
                details = await c.show()
            except Exception:
                continue
            labels = ((details.get("Config") or {}).get("Labels") or {})
            ws_id = labels.get(_LABEL_ID) or ""
            if workspace_id and ws_id != workspace_id:
                continue
            pid_limit = int((details.get("HostConfig") or {}).get("PidsLimit") or 0)
            if pid_limit <= 0:
                continue  # unlimited / unknown — can't compute ratio

            # Pull current PID count from docker stats. Stream=False
            # returns a single snapshot — cheap.
            try:
                stats = await c.stats(stream=False)
                if isinstance(stats, list):
                    stats = stats[0] if stats else {}
                pid_count = int(((stats or {}).get("pids_stats") or {}).get("current") or 0)
            except Exception:
                continue
            if pid_count <= 0:
                continue

            ratio = pid_count / pid_limit
            warned = ratio >= warn_pct
            if warned:
                log.warning(
                    "coder.workspace_pids_pressure",
                    workspace_id=ws_id,
                    pid_count=pid_count,
                    pid_limit=pid_limit,
                    ratio=round(ratio, 3),
                    warn_pct=warn_pct,
                    hint=(
                        "Container near pids.max; new docker exec calls "
                        "will return 'procReady not received' once saturated. "
                        "Restart the workspace to free orphaned PIDs, or "
                        "raise coder_workspace_pids_limit."
                    ),
                )
            results.append({
                "workspace_id": ws_id,
                "pid_count": pid_count,
                "pid_limit": pid_limit,
                "ratio": ratio,
                "warned": warned,
            })
        return results

    async def enable_published_ports(
        self, workspace_id: str,
    ) -> tuple[ContainerInfo, bool]:
        """Recreate a workspace with loopback-published dev-server ports.

        Docker port bindings are fixed at container creation. To expose ports
        after a workspace already exists, recreate the container against the
        same persistent workspace volume so files survive while runtime network
        bindings change.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        container = await self._docker.containers.get(info.container_id)
        details = await container.show()
        if self._container_has_published_ports(details):
            info.status = ((details.get("State") or {}).get("Status") or info.status)
            return info, False

        actual_image = ((details.get("Config") or {}).get("Image") or "augmentum-workspace")
        # Read existing PidsLimit then upgrade through the resolver so
        # recreate paths auto-apply any operator-bumped setting. A
        # workspace stuck at the old 256 default gets the new limit
        # without forcing the user to migrate manually.
        existing_pids = int((details.get("HostConfig") or {}).get("PidsLimit") or 0)
        pids = max(existing_pids, _resolve_workspace_pids_limit())

        try:
            await container.stop()
        except Exception:
            log.warning("workspace_publish_ports_stop_failed", workspace_id=workspace_id, exc_info=True)
        try:
            await container.delete()
        except Exception as exc:
            raise RuntimeError(f"Failed to recreate workspace container: {exc}") from exc

        setup_lines = self._build_workspace_setup_lines(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            tooling_profile=info.tooling_profile,
            initialize_workspace=False,
        )
        # Decoupled keep-alive: re-provisioning on recreate can't kill the box.
        cmd = _assemble_keepalive_cmd(setup_lines)
        config = self._build_workspace_container_config(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            cmd=cmd,
            cpu=info.resources_cpu,
            memory_bytes=_parse_memory(info.resources_memory),
            pids=pids,
            publish_ports=True,
            tooling_profile=info.tooling_profile,
            lan_accessible=info.lan_accessible,
        )
        new_container = await self._docker.containers.run(
            config=config,
            name=self._workspace_container_name(workspace_id),
        )

        info.container_id = new_container.id
        info.status = "running"
        info.last_active = time.time()

        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET container_id=?, status=?, last_active=? WHERE id=?",
                (info.container_id, info.status, info.last_active, workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()

        log.info(
            "workspace_ports_published",
            workspace_id=workspace_id,
            container_id=info.container_id,
        )
        return info, True

    async def set_lan_accessible(
        self, workspace_id: str, enabled: bool,
    ) -> tuple[ContainerInfo, bool]:
        """Toggle LAN accessibility for a workspace.

        Recreates the container so port bindings switch between 127.0.0.1
        (loopback) and 0.0.0.0 (LAN-reachable). The workspace volume
        survives — data is never at risk. Returns (info, changed).
        """
        info = await self._get_workspace(workspace_id)
        if info.lan_accessible == enabled:
            return info, False
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        container = await self._docker.containers.get(info.container_id)
        details = await container.show()
        actual_image = (
            (details.get("Config") or {}).get("Image") or "augmentum-workspace"
        )
        existing_pids = int(
            (details.get("HostConfig") or {}).get("PidsLimit") or 0
        )
        pids = max(existing_pids, _resolve_workspace_pids_limit())
        has_ports = self._container_has_published_ports(details)

        try:
            await container.stop()
        except Exception:
            log.warning(
                "workspace_lan_toggle_stop_failed",
                workspace_id=workspace_id, exc_info=True,
            )
        try:
            await container.delete()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to recreate workspace container: {exc}"
            ) from exc

        setup_lines = self._build_workspace_setup_lines(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            tooling_profile=info.tooling_profile,
            initialize_workspace=False,
        )
        cmd = _assemble_keepalive_cmd(setup_lines)
        config = self._build_workspace_container_config(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            cmd=cmd,
            cpu=info.resources_cpu,
            memory_bytes=_parse_memory(info.resources_memory),
            pids=pids,
            publish_ports=has_ports,
            tooling_profile=info.tooling_profile,
            lan_accessible=enabled,
        )
        new_container = await self._docker.containers.run(
            config=config,
            name=self._workspace_container_name(workspace_id),
        )

        info.container_id = new_container.id
        info.status = "running"
        info.last_active = time.time()
        info.lan_accessible = enabled

        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts "
                "SET container_id=?, status=?, last_active=?, lan_accessible=? "
                "WHERE id=?",
                (
                    info.container_id, info.status, info.last_active,
                    1 if enabled else 0, workspace_id,
                ),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()

        log.info(
            "workspace_lan_accessible_toggled",
            workspace_id=workspace_id,
            lan_accessible=enabled,
            container_id=info.container_id,
        )

        if not enabled:
            try:
                from augmentum.providers.caddy_front_door import (
                    remove_workspace_gate,
                )
                ws_slug = _workspace_slug(info.name, workspace_id)
                await remove_workspace_gate(self._docker, ws_slug)
            except Exception:
                log.debug("workspace_gate_remove_on_lan_off", exc_info=True)

        return info, True

    async def stop(self, workspace_id: str) -> ContainerInfo:
        """Stop a running workspace container.

        Idempotent against drift: if the container is gone (404) or
        already stopped (409 "not running"), reconcile the DB row and
        return success — stopped is the desired state either way.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        try:
            container = await self._docker.containers.get(info.container_id)
            await container.stop()
        except Exception as exc:
            if _is_container_gone(exc):
                await self._reconcile_to_stopped(
                    workspace_id, clear_container_id=True,
                )
                info.status = "stopped"
                info.container_id = None
                log.info(
                    "workspace_stop_reconciled_gone",
                    workspace_id=workspace_id,
                )
                return info
            if _is_container_not_running(exc):
                await self._reconcile_to_stopped(workspace_id)
                info.status = "stopped"
                log.info(
                    "workspace_stop_reconciled_already_stopped",
                    workspace_id=workspace_id,
                )
                return info
            raise

        info.status = "stopped"
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET status=? WHERE id=?",
                ("stopped", workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()

        log.info("workspace stopped", workspace_id=workspace_id)
        return info

    async def pause(self, workspace_id: str) -> ContainerInfo:
        """Suspend every process in the workspace via the cgroup freezer.

        Idempotent (Docker returns 304 on already-paused). CPU drops to 0,
        RAM is held — resume via :meth:`unpause` is sub-second. Used by
        the idle reaper as a softer alternative to ``stop`` so a returning
        user doesn't pay the 30-60s recreate cost.

        Persists ``status='paused'`` on success so list/sweep paths see
        the new state without a Docker probe.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        try:
            container = await self._docker.containers.get(info.container_id)
            await container.pause()
        except Exception as exc:
            # Two drift classes the reaper needs to absorb so it stops
            # re-issuing pause every tick: 404 (container gone — clear
            # the stale id) and 409 "not running" (container exited but
            # row still claims running). Both reconcile to ``stopped``.
            if _is_container_gone(exc):
                await self._reconcile_to_stopped(
                    workspace_id, clear_container_id=True,
                )
                info.status = "stopped"
                info.container_id = None
                log.info(
                    "workspace_pause_reconciled_gone",
                    workspace_id=workspace_id,
                )
                return info
            if _is_container_not_running(exc):
                await self._reconcile_to_stopped(workspace_id)
                info.status = "stopped"
                log.info(
                    "workspace_pause_reconciled_stopped",
                    workspace_id=workspace_id,
                )
                return info
            log.warning(
                "workspace_pause_failed",
                workspace_id=workspace_id, error=str(exc),
            )
            return info

        info.status = "paused"
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET status=? WHERE id=?",
                ("paused", workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()
        log.info("workspace_paused", workspace_id=workspace_id)
        return info

    async def unpause(self, workspace_id: str) -> ContainerInfo:
        """Thaw a paused workspace. Idempotent.

        Bumps ``last_active`` so the reaper doesn't re-pause it on the
        next tick. Callers that resume from the reaper deliberately
        (without user activity) should not use this method.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        try:
            container = await self._docker.containers.get(info.container_id)
            await container.unpause()
        except Exception as exc:
            # mark_active calls this every ~30s while DB still says
            # paused, so if we don't reconcile here a missing-container
            # row spins the loop forever (pre-fix bug exactly mirroring
            # the old pause()-409 loop).
            if _is_container_gone(exc):
                await self._reconcile_to_stopped(
                    workspace_id, clear_container_id=True,
                )
                info.status = "stopped"
                info.container_id = None
                log.info(
                    "workspace_unpause_reconciled_gone",
                    workspace_id=workspace_id,
                )
                return info
            if _is_container_not_running(exc):
                # Race: container exited between DB read and unpause
                # (idle reaper stage-2 stop, OOM, etc.). Reconcile.
                await self._reconcile_to_stopped(workspace_id)
                info.status = "stopped"
                log.info(
                    "workspace_unpause_reconciled_stopped",
                    workspace_id=workspace_id,
                )
                return info
            log.warning(
                "workspace_unpause_failed",
                workspace_id=workspace_id, error=str(exc),
            )
            return info

        info.status = "running"
        info.last_active = time.time()
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET status=?, last_active=? WHERE id=?",
                ("running", info.last_active, workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()
        log.info("workspace_unpaused", workspace_id=workspace_id)
        return info

    async def set_always_on(
        self, workspace_id: str, *, always_on: bool,
    ) -> ContainerInfo:
        """Persist the always-on lifecycle flag for a workspace.

        ``always_on=True`` exempts the workspace from ``sweep_idle``.
        ``always_on=False`` (default for new workspaces) opts into the
        idle reaper. The change is purely a policy bit — no container
        state is mutated here; the reaper picks up the new value on
        its next sweep tick.
        """
        info = await self._get_workspace(workspace_id)
        info.always_on = always_on
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET always_on=? WHERE id=?",
                (1 if always_on else 0, workspace_id),
            )
            await self._db.commit()
        log.info(
            "workspace_always_on_set",
            workspace_id=workspace_id,
            always_on=always_on,
        )
        return info

    # Per-workspace activity debounce. Bumping last_active on every
    # inspector poll / chat completion produces a DB write per second
    # under steady-state UI load. The reaper threshold is measured in
    # minutes, so coalescing same-workspace bumps within a 30s window
    # is harmless and keeps the DB quiet.
    _ACTIVITY_DEBOUNCE_S: float = 30.0

    async def _touch_last_active(self, workspace_id: str) -> None:
        """Lean, debounced ``last_active`` bump — no unpause side-effects.

        Called from the exec hot-path (``_run_command``) so that a workspace
        actively running commands can NEVER be reaped, even with the UI tab
        closed (no inspector polling). Historically ``last_active`` was bumped
        only by route hits + the start of each LLM completion, so a single
        long tool call (a multi-minute build/test, or a stuck command) past
        ``coder_idle_timeout`` let the reaper pause/freeze the container
        mid-command — the in-flight exec then hung on the frozen cgroup. This
        closes that race at the layer where work actually happens. Shares the
        same 30s debounce as ``mark_active`` so a burst of tool calls is one
        DB write. ``mark_active`` keeps the unpause logic for route hits;
        ``_run_command`` already thaws, so this stays minimal. Never raises.
        """
        if self._db is None or not workspace_id:
            return
        now = time.time()
        last = self._activity_last_seen.get(workspace_id, 0.0)
        if now - last < self._ACTIVITY_DEBOUNCE_S:
            return
        self._activity_last_seen[workspace_id] = now
        try:
            await self._db.execute(
                "UPDATE project_checkouts SET last_active=? WHERE id=?",
                (now, workspace_id),
            )
            await self._db.commit()
        except Exception:
            log.warning(
                "touch_last_active_failed", workspace_id=workspace_id, exc_info=True,
            )

    async def mark_active(self, workspace_id: str) -> None:
        """Bump ``last_active`` for activity tracking by the idle reaper.

        Called from coder route hits + chat completions whenever a
        client is doing something against this workspace. Coalesces
        repeated bumps within a 30-second window so high-frequency
        polling endpoints (inspector-state at ~5s cadence) don't
        write the DB every tick.

        Safe to call on a missing / stopped / deleted workspace —
        absent or non-matching IDs simply no-op (UPDATE … WHERE id=?
        matches zero rows). Never raises; reaper-relevant errors are
        logged and swallowed.
        """
        if self._db is None or not workspace_id:
            return
        now = time.time()
        last = self._activity_last_seen.get(workspace_id, 0.0)
        if now - last < self._ACTIVITY_DEBOUNCE_S:
            return
        self._activity_last_seen[workspace_id] = now
        try:
            await self._db.execute(
                "UPDATE project_checkouts SET last_active=? WHERE id=?",
                (now, workspace_id),
            )
            await self._db.commit()
        except Exception:
            log.warning(
                "mark_active_failed", workspace_id=workspace_id, exc_info=True,
            )

        # Auto-thaw if the workspace was paused by the idle reaper.
        # Cheap DB check first (one indexed read) — only round-trip to
        # Docker when we have an actual paused row. The debounce above
        # already coalesces hot polling, so this fires at most once
        # per 30s per workspace.
        try:
            row = await self._db.execute_fetchall(
                "SELECT status FROM project_checkouts WHERE id=?",
                (workspace_id,),
            )
            if row and row[0][0] == "paused":
                await self.unpause(workspace_id)
        except Exception:
            log.warning(
                "mark_active_unpause_failed",
                workspace_id=workspace_id, exc_info=True,
            )

    async def sweep_idle(self, timeout_seconds: int) -> int:
        """Reap workspaces idle past the timeout. Returns count acted on.

        Two-stage lifecycle when ``settings.coder_pause_idle`` is set
        (default True):

          * running → idle for ``timeout_seconds`` → **paused**
            (cgroup freeze: CPU=0, RAM held, sub-second resume)
          * paused  → idle for ``timeout_seconds + coder_pause_stop_after_seconds``
            → **stopped** (RAM freed, recreate cost on next access)

        With ``coder_pause_idle=False`` falls back to the legacy
        single-stage behavior (running → stopped).

        Selection rules across stages:
          * always_on=0     — opt-in to reaping
          * kind != 'bug_finder' (long-running audit workspaces have
            their own lifecycle; reaping them mid-run would corrupt
            their bundle output)
          * last_active is NULL OR < now - threshold

        Errors on any single workspace don't abort the sweep — the
        next tick retries. Called periodically by the lifespan
        background watcher (see ``_coder_idle_watcher`` in server.py).
        """
        if self._db is None or timeout_seconds <= 0:
            return 0
        try:
            from augmentum.config import settings as _cfg
            pause_idle = bool(getattr(_cfg, "coder_pause_idle", True))
            stop_after = int(getattr(_cfg, "coder_pause_stop_after_seconds", 21600) or 0)
        except Exception:
            pause_idle = True
            stop_after = 21600
        now = time.time()
        pause_cutoff = now - timeout_seconds
        stop_cutoff = now - timeout_seconds - max(stop_after, 0)

        # Stage 1: running workspaces past the idle threshold. When
        # pause_idle is True, freeze them; else fall back to legacy stop.
        try:
            running_rows = await self._db.execute_fetchall(
                "SELECT id FROM project_checkouts "
                "WHERE status = 'running' "
                "  AND always_on = 0 "
                "  AND COALESCE(kind, 'regular') != 'bug_finder' "
                "  AND (last_active IS NULL OR last_active < ?)",
                (pause_cutoff,),
            )
        except Exception:
            log.warning("sweep_idle_query_failed", exc_info=True)
            return 0

        acted = 0
        for row in running_rows:
            ws_id = row[0]
            try:
                if pause_idle:
                    result = await self.pause(ws_id)
                    # pause() swallows "container not running" and reconciles
                    # the row to 'stopped' on its own — don't count or log
                    # that as a reap, the row's already accounted for.
                    if result.status == "paused":
                        log.info(
                            "workspace_reaped_paused",
                            workspace_id=ws_id,
                            timeout_s=timeout_seconds,
                        )
                        acted += 1
                else:
                    await self.stop(ws_id)
                    log.info(
                        "workspace_reaped_idle",
                        workspace_id=ws_id,
                        timeout_s=timeout_seconds,
                    )
                    acted += 1
            except Exception:
                log.warning(
                    "sweep_idle_stage1_failed",
                    workspace_id=ws_id,
                    exc_info=True,
                )

        # Stage 2: paused workspaces past the deeper threshold. Only
        # runs when pause_idle is on AND stop_after > 0 (otherwise
        # paused state persists until user activity).
        if pause_idle and stop_after > 0:
            try:
                paused_rows = await self._db.execute_fetchall(
                    "SELECT id FROM project_checkouts "
                    "WHERE status = 'paused' "
                    "  AND always_on = 0 "
                    "  AND COALESCE(kind, 'regular') != 'bug_finder' "
                    "  AND (last_active IS NULL OR last_active < ?)",
                    (stop_cutoff,),
                )
            except Exception:
                log.warning("sweep_idle_stage2_query_failed", exc_info=True)
                paused_rows = []

            for row in paused_rows:
                ws_id = row[0]
                # Stop expects status='running' downstream; the docker
                # container is paused, but stop() calls container.stop()
                # which works on paused containers too (Docker unpauses
                # first internally).
                try:
                    await self.stop(ws_id)
                    log.info(
                        "workspace_reaped_stopped_from_pause",
                        workspace_id=ws_id,
                        timeout_s=timeout_seconds + stop_after,
                    )
                    acted += 1
                except Exception:
                    log.warning(
                        "sweep_idle_stage2_failed",
                        workspace_id=ws_id,
                        exc_info=True,
                    )

        # Stage 3 (passive): surface always_on=1 rows that haven't been
        # touched in 14+ days. These are usually leftover flags from
        # short-lived probe runs (powers-probe-*, viewport-probe) that
        # never had the bit cleared. The reaper deliberately exempts
        # always_on=1 from any stop/pause action — we only warn so the
        # user can decide whether to clear the flag.
        stale_cutoff = now - 14 * 86400
        warn_throttle = now - 86400  # re-warn at most once per workspace per day
        try:
            stale_rows = await self._db.execute_fetchall(
                "SELECT id, name, last_active FROM project_checkouts "
                "WHERE always_on = 1 "
                "  AND archived_at IS NULL "
                "  AND (last_active IS NULL OR last_active < ?)",
                (stale_cutoff,),
            )
        except Exception:
            log.warning("sweep_idle_stage3_query_failed", exc_info=True)
            stale_rows = []

        for row in stale_rows:
            ws_id = row[0]
            last_warned = self._stale_always_on_warned.get(ws_id, 0.0)
            if last_warned > warn_throttle:
                continue
            self._stale_always_on_warned[ws_id] = now
            idle_days = (
                int((now - row[2]) / 86400) if row[2] else None
            )
            # info, not warning: nothing is wrong — the reaper deliberately
            # exempts always_on=1, this is purely a "you have a leftover flag
            # you might want to clear" advisory (usually short-lived probe
            # workspaces). Warning level implied an action was required.
            log.info(
                "workspace_stale_always_on",
                workspace_id=ws_id[:8],
                name=row[1],
                idle_days=idle_days,
                hint="exempt from reaper; clear always_on=0 if no longer needed",
            )

        return acted

    async def delete(self, workspace_id: str, keep_volume: bool = True) -> None:
        """Stop and remove a workspace container + its DB record.

        Phase 1 / PR-1.2 changed the default for ``keep_volume`` from
        ``False`` to ``True``. Rationale: the durable Project bare repo
        on host now carries every checkpoint, so the workspace volume
        is a disposable checkout. Nuking the volume on every recycle
        was the silent data-loss bug the integrated coding nervous
        system spec calls out. Callers that actually want the volume
        gone (e.g. ``archive_project``) opt in explicitly with
        ``keep_volume=False``.
        """
        info = await self._get_workspace(workspace_id)

        # Close the workspace's persistent browser session in the sidecar
        # (best-effort): otherwise it lingers until the daemon idle-timeout,
        # and a recreated workspace with the same id would inherit stale
        # page state (cookies, tabs, whatever the last agent left open).
        try:
            from augmentum.coder import browser_sidecar as _bs
            await _bs.close_workspace_session(self, workspace_id)
        except Exception:
            log.warning(
                "browser_sidecar_session_close_on_delete_failed",
                workspace_id=workspace_id, exc_info=True,
            )

        if info.container_id is not None:
            try:
                container = await self._docker.containers.get(info.container_id)
                await container.stop()
                await container.delete()
            except Exception:
                log.warning(
                    "failed to remove Docker container",
                    workspace_id=workspace_id,
                    container_id=info.container_id,
                )

        # Remove the persistent volume only when the caller opts in.
        if not keep_volume:
            volume_name = f"augmentum-ws-{workspace_id[:12]}"
            try:
                vol = await self._docker.volumes.get(volume_name)
                await vol.delete()
                log.info("workspace_volume_deleted", volume=volume_name)
            except Exception as exc:
                # Volume may not exist (old container, or already removed);
                # debug-log so a permission/daemon issue is findable.
                log.debug(
                    "workspace_volume_delete_skipped",
                    volume=volume_name,
                    error=str(exc),
                )

        if self._db is not None:
            await self._db.execute(
                "DELETE FROM project_checkouts WHERE id=?", (workspace_id,)
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()

        log.info(
            "workspace deleted",
            workspace_id=workspace_id,
            kept_volume=keep_volume,
        )

    async def archive(self, workspace_id: str) -> None:
        """Soft-delete: remove the container, KEEP the /workspace volume, and
        mark the row archived so it drops out of the active list but can be
        restored natively later.

        This is the default the delete button routes to — it reclaims the
        container's runtime footprint (the ~GB image layers + any RAM) while
        preserving the volume (files, deps) and the row (name, mission/task
        progression). The opt-in hard delete is :meth:`delete` with
        ``keep_volume=False``.
        """
        info = await self._get_workspace(workspace_id)

        # Snapshot the volume size while the container is still around — the
        # socket proxy forbids /system/df, so this is our one chance to record
        # what archiving reclaims without spinning a probe on every list load.
        size_bytes = await self._measure_volume_bytes(workspace_id)

        # Close the persistent browser session (best-effort) — same as delete,
        # so a later restore doesn't inherit stale page/cookie state.
        try:
            from augmentum.coder import browser_sidecar as _bs
            await _bs.close_workspace_session(self, workspace_id)
        except Exception:
            log.warning(
                "browser_sidecar_session_close_on_archive_failed",
                workspace_id=workspace_id, exc_info=True,
            )

        if info.container_id is not None:
            try:
                container = await self._docker.containers.get(info.container_id)
                await container.stop()
                await container.delete()
            except Exception:
                log.warning(
                    "archive_container_remove_failed",
                    workspace_id=workspace_id, container_id=info.container_id,
                )

        now = time.time()
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts "
                "SET archived_at=?, archived_size_bytes=?, status='archived', "
                "container_id=NULL, last_active=? WHERE id=?",
                (now, size_bytes, now, workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()
        log.info(
            "workspace_archived",
            workspace_id=workspace_id, size_bytes=size_bytes,
        )

    async def restore(self, workspace_id: str) -> ContainerInfo:
        """Respawn a fresh container onto an archived workspace's surviving
        volume and clear the archived flag.

        The volume (``augmentum-ws-<id>``) still holds the files + persistent
        dep layer, so recreation is nearly free — we rebuild the container the
        same way the port-publish / LAN toggles recreate it, binding the same
        named volume.
        """
        info = await self._get_workspace(workspace_id)

        volume_name = self._workspace_volume_name(workspace_id)
        if not await self._volume_exists(volume_name):
            raise ValueError(
                f"Cannot restore {workspace_id}: data volume '{volume_name}' "
                f"is gone (was it completely removed?)"
            )

        # No prior container to read the image from — resolve the same way the
        # create path does (default base image, profile-aware fallback chain).
        actual_image = await self._resolve_workspace_image(
            "augmentum-workspace", info.tooling_profile,
        )
        pids = _resolve_workspace_pids_limit()

        # Clean up any stale container squatting the deterministic name (a
        # crashed archive that removed the row-pointer but not the container).
        container_name = self._workspace_container_name(workspace_id)
        try:
            stale = await self._docker.containers.get(container_name)
            await stale.delete(force=True)
        except Exception:
            pass

        setup_lines = self._build_workspace_setup_lines(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            tooling_profile=info.tooling_profile,
            initialize_workspace=False,
        )
        cmd = _assemble_keepalive_cmd(setup_lines)
        config = self._build_workspace_container_config(
            workspace_id=workspace_id,
            name=info.name,
            actual_image=actual_image,
            cmd=cmd,
            cpu=info.resources_cpu,
            memory_bytes=_parse_memory(info.resources_memory),
            pids=pids,
            publish_ports=False,
            tooling_profile=info.tooling_profile,
            lan_accessible=info.lan_accessible,
        )
        new_container = await self._docker.containers.run(
            config=config, name=container_name,
        )

        info.container_id = new_container.id
        info.status = "running"
        info.last_active = time.time()
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts "
                "SET archived_at=NULL, container_id=?, status=?, last_active=? "
                "WHERE id=?",
                (info.container_id, info.status, info.last_active, workspace_id),
            )
            await self._db.commit()
        self._invalidate_docker_state_cache()
        log.info(
            "workspace_restored",
            workspace_id=workspace_id, container_id=info.container_id,
        )
        return info

    async def list_archived(self, *, user_id: str = "") -> list[dict]:
        """Archived workspaces for the archive view — name, timestamps, volume
        size, and accumulated task/mission progression (from coder_sessions).

        User-scoped: only rows owned by ``user_id`` (empty = no scoping, for
        internal callers). Rows whose volume has since vanished are still
        listed with ``volume_present=False`` so the user can purge the dead
        row instead of it lingering invisibly.
        """
        if self._db is None:
            return []
        params: list = []
        where = "archived_at IS NOT NULL"
        if user_id:
            where += " AND user_id=?"
            params.append(user_id)
        rows = await self._db.execute_fetchall(
            "SELECT id, name, created_at, last_active, archived_at, "
            "tooling_profile, kind, project_id, git_url, archived_size_bytes "
            f"FROM project_checkouts WHERE {where} ORDER BY archived_at DESC",
            tuple(params),
        )
        out: list[dict] = []
        for r in rows:
            ws_id = r[0]
            volume_name = self._workspace_volume_name(ws_id)
            tasks = await self._archived_task_summary(ws_id)
            out.append({
                "id": ws_id,
                "name": r[1] or "workspace",
                "created_at": r[2],
                "last_active": r[3],
                "archived_at": r[4],
                "tooling_profile": r[5] or "browser",
                "kind": r[6] or "regular",
                "project_id": r[7] or "",
                "git_url": r[8] or "",
                "volume_present": await self._volume_exists(volume_name),
                "size_bytes": int(r[9] or 0),
                "tasks": tasks,
            })
        return out

    async def _archived_task_summary(self, workspace_id: str) -> dict:
        """Mission/task progression for an archived workspace's sessions."""
        if self._db is None:
            return {"total": 0, "done": 0, "items": []}
        rows = await self._db.execute_fetchall(
            "SELECT mission, tasks, plan_steps FROM coder_sessions "
            "WHERE workspace_id=? ORDER BY updated_at DESC LIMIT 1",
            (workspace_id,),
        )
        if not rows:
            return {"total": 0, "done": 0, "items": []}
        import json as _json
        items: list[dict] = []
        for raw in rows[0]:
            if not raw:
                continue
            try:
                parsed = _json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, list):
                for it in parsed:
                    if isinstance(it, dict):
                        text = (
                            it.get("text") or it.get("title")
                            or it.get("description") or it.get("promise") or ""
                        )
                        status = (
                            it.get("status") or it.get("state")
                            or ("done" if it.get("done") or it.get("verified") else "")
                        )
                    else:
                        text, status = str(it), ""
                    if text:
                        items.append({"text": text[:200], "status": status})
                if items:
                    break  # first non-empty source wins (mission > tasks > plan)
        done = sum(1 for it in items if str(it["status"]).lower() in ("done", "complete", "completed", "verified"))
        return {"total": len(items), "done": done, "items": items[:50]}

    async def archive_project(
        self,
        *,
        project_id: str,
        user_id: str,
    ) -> bool:
        """Permanently delete a Project + every checkout it owns.

        This is the *opt-in* destructive path complementing the now-
        non-destructive :meth:`delete`. It:
          1. Looks up every checkout linked to the project
          2. Stops + removes each one, including the checkout volume
          3. Deletes the Project row (cascades to project_repos +
             project_refs)
          4. ``ProjectStore.delete`` rmtrees the on-disk bare repo

        Idempotent: missing project / no checkouts -> returns False.
        """
        if self._db is None or not project_id or not user_id:
            return False

        from augmentum.config import settings as _settings
        from augmentum.projects import ProjectRepoStorage, ProjectStore

        storage = ProjectRepoStorage(
            Path(_settings.data_dir) / "projects",
        )
        store = ProjectStore(self._db, storage)
        project = await store.get(project_id, user_id=user_id)
        if project is None:
            return False

        # Pull every linked checkout so we can clean their containers
        # + volumes before the FK SET NULL fires.
        cursor = await self._db.execute(
            "SELECT id FROM project_checkouts WHERE project_id = ?",
            (project_id,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            checkout_id = row[0]
            try:
                await self.delete(checkout_id, keep_volume=False)
            except Exception:
                log.warning(
                    "archive_project_checkout_delete_failed",
                    project_id=project_id, checkout_id=checkout_id,
                    exc_info=True,
                )

        # Project row + on-disk bare repo go via the store.
        deleted = await store.delete(project_id, user_id=user_id)
        log.info(
            "project_archived",
            project_id=project_id, user_id=user_id,
            checkouts_removed=len(rows), deleted=deleted,
        )
        return deleted

    # ------------------------------------------------------------------
    # Terminal / exec
    # ------------------------------------------------------------------

    async def exec_shell(
        self,
        workspace_id: str,
        *,
        command: str | None = None,
        cwd: str = "/workspace",
    ) -> object:
        """Create a docker exec session for an interactive shell.

        Returns an aiodocker exec object whose ``start()`` can be used with
        a WebSocket for PTY streaming.

        With ``command`` set, runs that command on the PTY via ``bash -lc``
        (login shell, so /etc/profile.d env — venv, cargo, npm prefix — is
        live) instead of an interactive login shell. This is the agent-
        facing terminal-session path (see terminal_sessions.py); the
        browser terminal keeps the default interactive shell.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        container = await self._docker.containers.get(info.container_id)

        # Probe state before exec — a stopped container would make
        # container.exec() fail with a 409 and surface as an unexplained
        # WebSocket close 1011 to the browser. Try to start once if
        # possible so a merely-paused workspace recovers silently.
        try:
            details = await container.show()
            state = (details.get("State") or {}).get("Status", "")
            if state and state != "running":
                log.info(
                    "exec_shell_container_not_running",
                    workspace_id=workspace_id,
                    state=state,
                )
                try:
                    # Paused → unpause (cheap, sub-second). Anything
                    # else (stopped/exited/created) → full start.
                    if state == "paused":
                        await container.unpause()
                    else:
                        await container.start()
                except Exception as exc:
                    raise RuntimeError(
                        f"Container not running (state={state}); recover failed: {exc}"
                    ) from exc
                if self._db is not None:
                    await self._db.execute(
                        "UPDATE project_checkouts SET status=?, last_active=? WHERE id=?",
                        ("running", time.time(), workspace_id),
                    )
                    await self._db.commit()
        except RuntimeError:
            raise
        except Exception:
            log.warning("exec_shell_state_probe_failed", workspace_id=workspace_id, exc_info=True)

        exec_obj = await container.exec(
            cmd=(
                ["bash", "-lc", command] if command
                else ["/bin/bash", "--login"]
            ),
            stdin=True,
            stdout=True,
            stderr=True,
            tty=True,
            workdir=cwd or "/workspace",
        )

        log.info(
            "exec session created",
            workspace_id=workspace_id,
            exec_id=getattr(exec_obj, "id", None),
        )

        # Touch last_active
        if self._db is not None:
            await self._db.execute(
                "UPDATE project_checkouts SET last_active=? WHERE id=?",
                (time.time(), workspace_id),
            )
            await self._db.commit()

        return exec_obj

    async def resize_exec(self, exec_id: str, rows: int, cols: int) -> None:
        """Resize the PTY of an active exec session."""
        try:
            async with self._docker._query(
                f"exec/{exec_id}/resize",
                method="POST",
                params={"h": rows, "w": cols},
            ) as resp:
                if resp.status not in (200, 201):
                    log.warning("exec_resize_bad_status", exec_id=exec_id, status=resp.status)
        except Exception:
            log.warning("failed to resize exec PTY", exec_id=exec_id)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    async def git_checkpoint(self, workspace_id: str, message: str) -> str | None:
        """Auto-commit the current workspace state as a checkpoint.

        Returns the short commit hash, or None if nothing to commit.
        """
        try:
            # HEAD may not exist yet on a freshly-initialized repo.
            before_sha = (
                await self._run_command(
                    workspace_id,
                    ["bash", "-c", "cd /workspace && git rev-parse --verify --short HEAD 2>/dev/null || true"],
                    timeout=5.0,
                )
            ).strip()
            await self._run_command(
                workspace_id,
                ["bash", "-c", "cd /workspace && git add -A"],
                timeout=10.0,
            )
            # Check if there's anything to commit
            status = await self._run_command(
                workspace_id,
                ["bash", "-c", "cd /workspace && git status --porcelain"],
                timeout=5.0,
            )
            if not status.strip():
                return None  # Nothing changed

            commit_msg = shlex.quote((message or "").strip() or "Workspace checkpoint")
            commit_output = (
                await self._run_command(
                    workspace_id,
                    [
                        "bash", "-c",
                        "cd /workspace && "
                        f"git -c user.name={shlex.quote(_GIT_AUTHOR_NAME)} "
                        f"-c user.email={shlex.quote(_GIT_AUTHOR_EMAIL)} "
                        f"commit -q -m {commit_msg}",
                    ],
                    timeout=10.0,
                )
            ).strip()
            # Get the short hash. Validate it so stderr from a failed
            # rev-parse never leaks into the UI as a fake checkpoint.
            sha = (
                await self._run_command(
                    workspace_id,
                    ["bash", "-c", "cd /workspace && git rev-parse --verify --short HEAD 2>/dev/null || true"],
                    timeout=5.0,
                )
            ).strip()
            if not _GIT_SHA_RE.fullmatch(sha):
                log.warning(
                    "git_checkpoint_invalid_head",
                    workspace=workspace_id,
                    head_output=sha[:200],
                    commit_output=commit_output[:200],
                )
                return None
            if sha == before_sha:
                log.warning(
                    "git_checkpoint_head_unchanged",
                    workspace=workspace_id,
                    head=sha,
                    commit_output=commit_output[:200],
                )
                return None

            # Phase 1 / PR-1.2: push to the host bare repo so the
            # checkpoint survives container recycle. The origin remote
            # was configured to /augmentum-bare at workspace creation
            # when a Project bare repo was bind-mounted. If origin
            # isn't set (legacy workspaces without a Project link),
            # skip silently — the local /workspace .git keeps the
            # commit, just without cross-recycle durability.
            push_output = (
                await self._run_command(
                    workspace_id,
                    [
                        "bash", "-c",
                        "cd /workspace && "
                        "git remote get-url origin > /dev/null 2>&1 && "
                        "git push origin HEAD:refs/heads/main 2>&1 || true",
                    ],
                    timeout=20.0,
                )
            ).strip()
            if push_output and (
                "error" in push_output.lower()
                or "fatal" in push_output.lower()
                or "rejected" in push_output.lower()
            ):
                # Non-fatal: commit lives in the workspace volume even
                # if the push lost. Surface as warning so a recurring
                # mount problem doesn't stay invisible.
                log.warning(
                    "git_checkpoint_push_failed",
                    workspace=workspace_id,
                    sha=sha,
                    push_output=push_output[:300],
                )

            return sha
        except Exception:
            # Previously log.debug — per CLAUDE.md, save-path failures
            # must surface as warnings or users silently lose restore
            # points without any indication.
            log.warning("git_checkpoint_failed", workspace=workspace_id, exc_info=True)
            return None

    async def git_log(self, workspace_id: str, limit: int = 20) -> list[dict]:
        """Get the checkpoint log for a workspace.

        Returns list of {hash, message, timestamp} dicts.
        """
        try:
            output = await self._run_command(
                workspace_id,
                ["bash", "-c", f"cd /workspace && git log --oneline --format='%h|%s|%ct' -n {limit}"],
                timeout=5.0,
            )
            entries = []
            for line in output.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    entries.append({
                        "hash": parts[0],
                        "message": parts[1],
                        "timestamp": int(parts[2]) if parts[2].isdigit() else 0,
                    })
            return entries
        except Exception:
            return []

    async def git_revert(self, workspace_id: str, commit_hash: str) -> bool:
        """Revert the workspace to a specific checkpoint.

        Uses ``git read-tree -u --reset <hash>`` to rewrite the working
        tree AND index to exactly match the target commit's tree,
        including removals — then commits the new state.

        Pre-2026-04-21 this used ``git checkout <hash> -- .`` followed
        by ``git add -A && git commit``. That subtly broke: `checkout`
        with a pathspec only restores files THAT EXIST in ``<hash>``;
        files added to HEAD after ``<hash>`` remained in the working
        tree and were bundled into the "Reverted to X" commit — a
        silent correctness bug. ``read-tree --reset`` also removes
        those, so the revert actually reverts.

        HEAD is not moved, so history is preserved (non-destructive).
        """
        # Basic sanitation — commit_hash comes from a user click on a
        # list we built from `git log`, but belt-and-suspenders against
        # shell injection via a hostile API client.
        safe_hash = "".join(c for c in commit_hash if c.isalnum())
        if not safe_hash:
            return False
        try:
            await self._run_command(
                workspace_id,
                [
                    "bash", "-c",
                    f"cd /workspace && git read-tree -u --reset {safe_hash}",
                ],
                timeout=10.0,
            )
            await self._run_command(
                workspace_id,
                [
                    "bash", "-c",
                    f"cd /workspace && git commit -q --allow-empty "
                    f"-m 'Reverted to {safe_hash}'",
                ],
                timeout=10.0,
            )
            return True
        except Exception:
            log.warning("git_revert_failed", workspace=workspace_id, hash=commit_hash, exc_info=True)
            return False

    async def git_diff(self, workspace_id: str, commit_hash: str) -> str:
        """Get the diff between a checkpoint and the current state."""
        try:
            output = await self._run_command(
                workspace_id,
                ["bash", "-c", f"cd /workspace && git diff {commit_hash} HEAD --stat"],
                timeout=10.0,
            )
            return output.strip()
        except Exception:
            return ""

    async def git_review_diff(
        self, workspace_id: str, base_commit: str, *, max_bytes: int = 200_000,
    ) -> dict:
        """Full working-tree diff since ``base_commit`` for the Agents review
        window: the stat summary plus the unified patch (committed AND
        uncommitted changes vs the dispatch-time HEAD), capped to keep the
        payload sane. Returns {stat, patch, truncated, base}."""
        base = (base_commit or "").strip()
        if not base:
            return {"stat": "", "patch": "", "truncated": False, "base": ""}
        try:
            stat = await self._run_command(
                workspace_id,
                ["bash", "-c", f"cd /workspace && git diff {shlex.quote(base)} --stat"],
                timeout=15.0,
            )
            patch = await self._run_command(
                workspace_id,
                ["bash", "-c", f"cd /workspace && git diff {shlex.quote(base)}"],
                timeout=20.0,
            )
        except Exception:
            return {"stat": "", "patch": "", "truncated": False, "base": base}
        # git diff misses untracked-new files; list them so review never
        # hides a file the run created (non-mutating — no index changes).
        # Filter environment noise: workspaces often don't gitignore their
        # venv/deps, which would otherwise bury the run's real new files.
        untracked: list[str] = []
        try:
            u = await self._run_command(
                workspace_id,
                ["bash", "-c",
                 "cd /workspace && git ls-files --others --exclude-standard"],
                timeout=10.0,
            )
            untracked = [
                ln for ln in (u or "").splitlines()
                if ln.strip() and not _NOISE_PATH.search(ln)
            ][:60]
        except Exception:
            untracked = []
        patch = patch or ""
        truncated = len(patch.encode("utf-8", "ignore")) > max_bytes
        if truncated:
            patch = patch.encode("utf-8", "ignore")[:max_bytes].decode("utf-8", "ignore")
        return {"stat": (stat or "").strip(), "patch": patch,
                "untracked": untracked, "truncated": truncated, "base": base}

    async def git_head_short(self, workspace_id: str) -> str:
        """Current HEAD short-hash, or '' if the workspace has no commits."""
        try:
            out = await self._run_command(
                workspace_id,
                ["bash", "-c",
                 "cd /workspace && git rev-parse --verify --short HEAD 2>/dev/null || true"],
                timeout=10.0,
            )
            return (out or "").strip()
        except Exception:
            return ""

    async def file_list(self, workspace_id: str, path: str = "/workspace") -> list[FileEntry]:
        """List files in a directory inside the workspace container."""
        output = await self._run_command(
            workspace_id,
            ["ls", "-la", "--time-style=+%s", path],
        )
        return _parse_ls_output(output, path)

    async def file_read(self, workspace_id: str, path: str) -> str:
        """Read the contents of a file in the workspace container."""
        return await self._run_command(workspace_id, ["cat", path])

    # Emitted by file_write on success. ``_run_command`` returns stdout and
    # NEVER inspects the exit code, so a failed shell redirect is
    # indistinguishable from a successful one by return value alone. The
    # marker is the only proof the write actually happened.
    _WRITE_OK = "__AUG_WRITE_OK__"

    async def file_write(self, workspace_id: str, path: str, content: str) -> None:
        """Write content to a file in the workspace container via sh -c.

        Creates the parent directory if needed, and RAISES if the write did
        not happen.

        Both ``content`` and ``path`` are shell-quoted. ``path`` quoting is
        defense-in-depth: the route layer validates it, but this manager API
        is reachable by other callers, and an unquoted redirect target is a
        shell-injection vector (e.g. ``foo; rm -rf /`` as a path).

        Two failure modes were silent here before 2026-07-26:

        1. **Missing parent.** ``> /workspace/src/new/app.py`` fails outright
           when ``src/new`` doesn't exist — the shell can't create the
           redirect target. ``file_write_bytes`` already ``mkdir -p``'d its
           parent for exactly this reason; the text writer didn't, so the
           same call succeeded or failed depending only on which of the two
           it routed through. Auto-create matches that sibling, matches
           ``file_restore`` below, and matches what every coding agent's
           write tool does.
        2. **No failure detection at all.** This returned ``None`` and
           discarded ``_run_command``'s output, which is where the shell's
           ``No such file or directory`` went. The tool then reported
           "Wrote N bytes" for a file that does not exist — the model
           believes it, moves on, and the next read or import fails somewhere
           unrelated. mkdir alone would NOT have fixed that: a read-only
           mount or a full disk still writes nothing and still claims
           success. So verify explicitly rather than trusting the redirect.
        """
        # Escape single quotes in content for the shell command
        escaped = content.replace("'", "'\\''")
        quoted_path = shlex.quote(path)
        parent = posixpath.dirname(path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        out = await self._run_command(
            workspace_id,
            ["sh", "-c",
             f"{mkdir}printf '%s' '{escaped}' > {quoted_path} "
             f"&& echo {self._WRITE_OK}"],
        )
        if self._WRITE_OK not in out:
            detail = out.replace(self._WRITE_OK, "").strip() or "unknown error"
            log.warning(
                "file_write_failed", workspace_id=workspace_id,
                path=path, detail=detail,
            )
            raise OSError(f"write to {path} failed: {detail}")

    async def file_delete(
        self,
        workspace_id: str,
        path: str,
        *,
        recursive: bool = False,
    ) -> None:
        """Delete a file (or directory tree if ``recursive``).

        Uses ``rm -f`` / ``rm -rf`` via run_command so the call works
        for both files and dirs depending on the flag. Path sanitization
        is the caller's responsibility — see the route handler for the
        rejection rules.

        ``rm -f`` is intentional: a request to delete a path that no
        longer exists succeeds quietly. Concurrent deletes from the
        agent + UI shouldn't surface a 500 to whoever races.
        """
        flag = "-rf" if recursive else "-f"
        await self._run_command(
            workspace_id, ["rm", flag, "--", path], timeout=30.0,
        )

    # -- Soft delete / trash -------------------------------------------------
    #
    # UI deletes route through file_trash (reversible) rather than file_delete
    # (hard rm). Trashed items live at /workspace/.augmentum/trash/<id>/ with a
    # manifest recording their original path, so an accidental delete — the
    # untracked-new-script case git can't recover — is one Undo away. The
    # trash dir is added to .git/info/exclude so it never pollutes git status
    # or a commit. The agent's own file_delete tool is unchanged (hard delete).

    _TRASH_ROOT = "/workspace/.augmentum/trash"
    _TRASH_ID_RE = re.compile(r"[0-9a-f]{8,32}")

    async def file_trash(self, workspace_id: str, path: str) -> str:
        """Move a path into the workspace trash. Returns the trash id.

        Handles files and directories (``mv``). Idempotently ensures the
        trash dir is git-excluded. Raises if the source doesn't exist so
        the caller can surface a real error rather than a phantom Undo.
        """
        trash_id = uuid.uuid4().hex[:16]
        base = path.rstrip("/").rsplit("/", 1)[-1] or "item"
        dest_dir = f"{self._TRASH_ROOT}/{trash_id}"
        manifest = json.dumps({
            "trash_id": trash_id, "original": path, "name": base,
            "deleted_at": int(time.time()),
        })
        manifest_b64 = base64.b64encode(manifest.encode()).decode()
        script = (
            "cd /workspace && "
            # Keep trash out of git without touching the user's .gitignore.
            "if [ -d .git ]; then mkdir -p .git/info && "
            "{ grep -qxF '.augmentum/trash/' .git/info/exclude 2>/dev/null || "
            "echo '.augmentum/trash/' >> .git/info/exclude; }; fi && "
            f"mkdir -p {shlex.quote(dest_dir)} && "
            f"mv -- {shlex.quote(path)} {shlex.quote(dest_dir)}/ && "
            f"echo {shlex.quote(manifest_b64)} | base64 -d "
            f"> {shlex.quote(dest_dir)}/.manifest.json"
        )
        await self._run_command(
            workspace_id, ["bash", "-c", script], timeout=30.0,
        )
        return trash_id

    async def file_restore(self, workspace_id: str, trash_id: str) -> dict:
        """Restore a trashed item to its original path.

        Returns ``{"restored": True, "path": <original>}`` on success, or
        ``{"restored": False, "reason": ...}`` when the trash entry is gone
        or the original path is now occupied (we never clobber on restore).
        """
        if not self._TRASH_ID_RE.fullmatch(trash_id or ""):
            return {"restored": False, "reason": "invalid trash id"}
        dest_dir = f"{self._TRASH_ROOT}/{trash_id}"
        raw = await self._run_command(
            workspace_id,
            ["bash", "-c",
             f"cat {shlex.quote(dest_dir)}/.manifest.json 2>/dev/null || true"],
            timeout=5.0,
        )
        if not raw.strip():
            return {"restored": False, "reason": "trash entry not found"}
        try:
            meta = json.loads(raw)
            original = meta["original"]
            name = meta["name"]
        except (ValueError, KeyError):
            return {"restored": False, "reason": "corrupt trash manifest"}
        src = f"{dest_dir}/{name}"
        parent = original.rsplit("/", 1)[0] or "/workspace"
        script = (
            f"if [ -e {shlex.quote(original)} ]; then echo __OCCUPIED__; "
            f"elif [ ! -e {shlex.quote(src)} ]; then echo __MISSING__; "
            f"else mkdir -p {shlex.quote(parent)} && "
            f"mv -- {shlex.quote(src)} {shlex.quote(original)} && "
            f"rm -rf {shlex.quote(dest_dir)} && echo __OK__; fi"
        )
        out = await self._run_command(
            workspace_id, ["bash", "-c", script], timeout=15.0,
        )
        if "__OCCUPIED__" in out:
            return {"restored": False, "reason": "a file already exists at the original path", "path": original}
        if "__MISSING__" in out:
            return {"restored": False, "reason": "trashed content missing"}
        if "__OK__" in out:
            return {"restored": True, "path": original}
        return {"restored": False, "reason": "restore failed"}

    async def file_list_trash(self, workspace_id: str) -> list[dict]:
        """List trash entries newest-first: [{trash_id, name, original, deleted_at}]."""
        out = await self._run_command(
            workspace_id,
            ["bash", "-c",
             f"for m in {self._TRASH_ROOT}/*/.manifest.json; do "
             "[ -f \"$m\" ] && cat \"$m\" && echo; done 2>/dev/null || true"],
            timeout=8.0,
        )
        entries: list[dict] = []
        for line in (out or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except ValueError:
                continue
            if meta.get("trash_id"):
                entries.append(meta)
        entries.sort(key=lambda e: e.get("deleted_at", 0), reverse=True)
        return entries

    async def file_purge_trash(self, workspace_id: str, trash_id: str = "") -> None:
        """Permanently remove one trash entry, or the whole trash if blank."""
        if trash_id:
            if not self._TRASH_ID_RE.fullmatch(trash_id):
                return
            target = f"{self._TRASH_ROOT}/{trash_id}"
        else:
            target = self._TRASH_ROOT
        await self._run_command(
            workspace_id, ["rm", "-rf", "--", target], timeout=15.0,
        )

    async def git_show(self, workspace_id: str, commit_hash: str) -> str:
        """Return a single commit's own diff (git show), for the history browser.

        Distinct from ``git_diff``, which diffs a commit against current
        HEAD. This is "what changed IN this commit". Caller validates the
        hash shape; we still pass it as a literal arg (no shell interp).
        """
        try:
            return (await self._run_command(
                workspace_id,
                ["bash", "-c",
                 f"cd /workspace && git show --no-color --format="
                 f"'%H%n%an%n%ct%n%s' {shlex.quote(commit_hash)} 2>&1"],
                timeout=15.0,
            )).strip()
        except Exception:
            return ""

    async def file_rename(
        self,
        workspace_id: str,
        old_path: str,
        new_path: str,
    ) -> None:
        """Rename / move a path inside the container.

        Wraps ``mv`` so it handles both files and dirs, and supports
        cross-directory moves the agent might use to reorganize a
        workspace. Caller sanitizes both paths.
        """
        await self._run_command(
            workspace_id, ["mv", "--", old_path, new_path], timeout=10.0,
        )

    async def file_mkdir(self, workspace_id: str, path: str) -> None:
        """Create a directory (and parents) inside the container.

        Idempotent (``mkdir -p``) so repeated calls or races against
        the agent are safe.
        """
        await self._run_command(
            workspace_id, ["mkdir", "-p", "--", path], timeout=5.0,
        )

    async def seed_from_host_paths(
        self,
        workspace_id: str,
        dest_path: str,
        host_root: str,
        rel_paths: list[str],
    ) -> int:
        """Stream files from a host directory into the workspace at ``dest_path``.

        Reads each ``rel_path`` from ``host_root`` on the *augmentum
        container's* filesystem (so the host path must be bind-mounted
        in) and tars the lot into a single put-archive call. Returns the
        count of files actually added — silently skips paths that
        disappear or aren't regular files, which matches git's own
        tolerance for races during ``ls-files``.

        Used by the coder self-test endpoint to seed a workspace with
        the live augmentum source tree, including uncommitted working-
        tree changes (which a ``git_url`` clone can't see).
        """
        import io
        import os
        import tarfile

        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        await self._run_command(workspace_id, ["mkdir", "-p", dest_path])

        buf = io.BytesIO()
        added = 0
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel in rel_paths:
                full = os.path.join(host_root, rel)
                try:
                    if not os.path.isfile(full):
                        continue
                    tar.add(full, arcname=rel, recursive=False)
                    added += 1
                except OSError:
                    continue
        buf.seek(0)

        container = await self._docker.containers.get(info.container_id)
        await container.put_archive(path=dest_path, data=buf.getvalue())
        return added

    async def file_upload(
        self,
        workspace_id: str,
        dest_path: str,
        files: list[tuple[str, bytes]],
    ) -> None:
        """Extract a set of files into the container at ``dest_path``.

        Uses Docker's put-archive API so binary files round-trip cleanly
        (the text-mode file_write path would corrupt anything that isn't
        valid UTF-8 via its shell quoting). Each tuple is
        ``(relative_path, bytes)`` — relative paths may include
        forward-slashed subdirectories; Docker's tar extractor creates
        them.

        Path sanitization is the caller's responsibility; see the route
        handler for the rejection rules.
        """
        import io
        import tarfile
        import time as _time

        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        buf = io.BytesIO()
        now = int(_time.time())
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel_path, data in files:
                ti = tarfile.TarInfo(name=rel_path)
                ti.size = len(data)
                ti.mtime = now
                ti.mode = 0o644
                tar.addfile(ti, io.BytesIO(data))
        buf.seek(0)

        # Docker's put-archive API requires dest_path to exist — it
        # extracts INTO a directory, it won't create the directory for
        # you. Ensure it exists first so callers can upload into fresh
        # sub-paths like /workspace/.augmentum/attachments without a
        # separate mkdir step.
        await self._run_command(
            workspace_id,
            ["mkdir", "-p", dest_path],
        )

        container = await self._docker.containers.get(info.container_id)
        await container.put_archive(path=dest_path, data=buf.getvalue())

    async def file_download(self, workspace_id: str, path: str) -> bytes:
        """Return raw bytes for a single file inside the workspace.

        Uses Docker's get-archive endpoint (returns a tar containing the
        file) rather than ``cat``-through-exec, so binary files aren't
        corrupted by stream decoding.
        """
        import tarfile

        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        container = await self._docker.containers.get(info.container_id)
        tar_obj = await container.get_archive(path=path)
        # aiodocker returns a tarfile.TarFile object. Pull the first
        # (only) regular file out of it.
        try:
            for member in tar_obj.getmembers():
                if member.isfile():
                    extracted = tar_obj.extractfile(member)
                    if extracted is None:
                        continue
                    return extracted.read()
            raise FileNotFoundError(f"No regular file found at {path}")
        finally:
            try:
                tar_obj.close()
            except (OSError, tarfile.TarError):
                # Tar-stream close on an already-broken stream raises; the
                # caller's exception (or FileNotFoundError above) is what
                # we want to surface, not the close failure.
                pass

    async def file_write_bytes(self, workspace_id: str, path: str, data: bytes) -> None:
        """Write raw bytes to a file inside the workspace, binary-safe.

        Uses Docker's put-archive endpoint (upload a tar) — the inverse of
        file_download's get-archive — rather than a base64-through-exec write,
        which blows ARG_MAX for anything but tiny files (a screenshot PNG is
        hundreds of KB). ``put_archive`` extracts INTO a directory, so we tar
        the single file and target its parent.
        """
        import io
        import posixpath
        import tarfile

        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        directory = posixpath.dirname(path) or "/"
        name = posixpath.basename(path)
        if not name:
            raise ValueError(f"file_write_bytes needs a file path, got {path!r}")
        # put_archive won't create the target directory — ensure it exists.
        await self._run_command(workspace_id, ["mkdir", "-p", directory])

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            member = tarfile.TarInfo(name=name)
            member.size = len(data)
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(data))

        container = await self._docker.containers.get(info.container_id)
        await container.put_archive(path=directory, data=buf.getvalue())

    async def list_ports(self, workspace_id: str) -> list[dict]:
        """Report which of the pre-published dev ports are actually
        being listened on inside the container, with the host-side
        port Docker assigned to each.

        Returns a list of dicts::

            [
              {"container_port": 3000, "host_port": 54321, "listening": True},
              {"container_port": 5173, "host_port": 54322, "listening": False},
              ...
            ]

        Reads /proc/net/tcp and /proc/net/tcp6 directly so we don't
        depend on ``ss`` or ``netstat`` being installed (they aren't
        on the fallback ubuntu:24.04 image).
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            return []

        # Host-side mapping from Docker.
        try:
            container = await self._docker.containers.get(info.container_id)
            details = await container.show()
        except Exception:
            return []
        net = (details.get("NetworkSettings") or {}).get("Ports") or {}
        host_map: dict[int, int] = {}
        for key, bindings in net.items():
            if not bindings:
                continue
            # key looks like "3000/tcp"
            try:
                cport = int(key.split("/", 1)[0])
            except ValueError:
                continue
            # Multiple bindings possible (IPv4 + IPv6); take the first
            # 127.0.0.1 one we find, or the first entry.
            picked = None
            for b in bindings:
                if (b.get("HostIp") or "") in ("127.0.0.1", "0.0.0.0"):
                    picked = b
                    break
            if picked is None:
                picked = bindings[0]
            try:
                host_map[cport] = int(picked.get("HostPort") or 0)
            except (TypeError, ValueError):
                host_map[cport] = 0

        # Workspace created without `publish_ports` — skip the /proc
        # read entirely. Saves an exec roundtrip every 5s.
        if not host_map:
            return [{
                "container_port": cport,
                "host_port": 0,
                "listening": False,
            } for cport in _DEV_PORTS]

        # Listening ports inside the container. /proc/net/tcp columns:
        # sl local_address rem_address st tx_queue:rx_queue tr:tm_when
        #    retrnsmt uid timeout inode ...
        # local_address = "IPHEX:PORTHEX"; st == "0A" means LISTEN.
        try:
            proc_output = await self._run_command(
                workspace_id,
                [
                    "sh", "-c",
                    "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null",
                ],
                timeout=5.0,
            )
        except Exception:
            proc_output = ""

        listening: set[int] = set()
        for line in proc_output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            if parts[3] != "0A":
                continue
            local = parts[1]
            if ":" not in local:
                continue
            try:
                port_hex = local.rsplit(":", 1)[1]
                listening.add(int(port_hex, 16))
            except ValueError:
                continue

        result: list[dict] = []
        for cport in _DEV_PORTS:
            result.append({
                "container_port": cport,
                "host_port": host_map.get(cport, 0),
                "listening": cport in listening,
            })
        return result

    async def import_archive_into(
        self, workspace_id: str, archive_bytes: bytes,
    ) -> None:
        """Extract a workspace_archive_stream-format ``.tar.gz`` into the
        target container's filesystem.

        Counterpart to :meth:`workspace_archive_stream` — together they
        round-trip a workspace's file state. The exported archive is
        produced by ``tar -czf - … workspace`` run from ``/``, so its
        top-level path is ``workspace/`` and extracting at ``/`` lands
        files exactly where they were.

        Raises ``ValueError`` for invalid gzip / corrupt archives so the
        route can surface a 400 instead of a 500.
        """
        import gzip

        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(
                f"Workspace {workspace_id} has no associated container"
            )
        if not archive_bytes:
            raise ValueError("Archive is empty")
        # put_archive needs raw tar; the export format is gzipped. Doing
        # the gunzip here in Python rather than relying on Docker's
        # content-type detection makes the contract explicit and easier
        # to test.
        try:
            tar_bytes = gzip.decompress(archive_bytes)
        except (OSError, EOFError) as exc:
            raise ValueError(f"Invalid gzip archive: {exc}") from exc

        container = await self._docker.containers.get(info.container_id)
        # Extract at root: the archive's top-level entry is ``workspace/``
        # so files land at /workspace/... matching the original layout.
        await container.put_archive(path="/", data=tar_bytes)
        log.info(
            "workspace_archive_imported",
            workspace_id=workspace_id,
            tar_bytes=len(tar_bytes),
            gzip_bytes=len(archive_bytes),
        )

    async def workspace_archive_stream(
        self, workspace_id: str, *, excludes: list[str] | None = None,
    ):
        """Yield chunks of a gzipped tar containing /workspace.

        The archive is produced by ``tar -czf -`` run against the
        ``/workspace`` named volume. The volume persists across container
        restarts and even container removal, so export works whether the
        workspace container is running, stopped, paused, or gone — we do
        NOT depend on the workspace's own container being live.

        Historically this ran ``tar`` via ``docker exec`` in the workspace
        container, which 409'd ("container is not running") whenever the
        workspace had been suspended/stopped — and the export route
        swallowed that into a silent 0-byte download. Now: if the
        workspace container is running we exec in it (cheap, no side
        effects); otherwise we tar the volume from a short-lived helper
        container mounting it read-only.

        Pre-baked excludes drop common dep/build dirs that balloon
        archive size — callers can pass ``excludes=[]`` to get everything.
        """
        if excludes is None:
            excludes = [
                "node_modules", ".venv", "venv", "__pycache__",
                ".next", "dist", "build", "target", ".cache",
            ]
        exclude_flags = " ".join(f"--exclude='{e}'" for e in excludes)
        cmd_str = f"cd / && tar -czf - {exclude_flags} workspace 2>/dev/null"

        async with self._exec_container(workspace_id) as container:
            async for chunk in self._exec_archive_stream(container, cmd_str):
                yield chunk

    @contextlib.asynccontextmanager
    async def _exec_container(self, workspace_id: str):
        """Yield a container to exec volume ops (tar/du) against.

        Prefers the workspace's OWN container when it's actually running (no
        helper spin, no side effects). Otherwise runs a throwaway helper that
        mounts the ``/workspace`` volume READ-ONLY — so the op works whether
        the workspace is stopped, paused, archived, or its container is gone.
        The helper uses the always-present workspace image and a long-lived
        shell (so we can exec and read RAW bytes; container logs would
        text-decode and corrupt a gzip stream). Cleans the helper up on exit.
        """
        info = await self._get_workspace(workspace_id)
        container = None
        if info.container_id is not None:
            try:
                c = await self._docker.containers.get(info.container_id)
                details = await c.show()
                if ((details.get("State") or {}).get("Status")) == "running":
                    container = c
            except Exception:
                container = None
        if container is not None:
            yield container
            return

        volume_name = self._workspace_volume_name(workspace_id)
        if not await self._volume_exists(volume_name):
            raise ValueError(
                f"Workspace {workspace_id} has no data volume "
                f"(volume '{volume_name}' not found)"
            )
        helper_name = f"augmentum-export-{workspace_id[:12]}"
        helper = None
        try:
            try:
                stale = await self._docker.containers.get(helper_name)
                await stale.delete(force=True)
            except Exception:
                pass
            helper = await self._docker.containers.run(
                config={
                    "Image": "augmentum-workspace",
                    "Cmd": ["sh", "-c", "sleep 3600"],
                    "HostConfig": {
                        "Binds": [f"{volume_name}:/workspace:ro"],
                        "AutoRemove": False,
                    },
                    "NetworkDisabled": True,
                },
                name=helper_name,
            )
            yield helper
        finally:
            if helper is not None:
                try:
                    await helper.delete(force=True)
                except Exception:
                    log.debug("export_helper_cleanup_failed", name=helper_name)

    async def _measure_volume_bytes(self, workspace_id: str) -> int:
        """Bytes on the ``/workspace`` volume, via ``du`` in the workspace or a
        read-only helper. Best-effort — 0 if it can't be measured."""
        try:
            out = b""
            async with self._exec_container(workspace_id) as container:
                async for chunk in self._exec_archive_stream(
                    container, "du -sb /workspace 2>/dev/null",
                ):
                    out += chunk
            return int(out.split()[0]) if out.split() else 0
        except Exception:
            log.debug("measure_volume_bytes_failed", workspace_id=workspace_id)
            return 0

    async def _volume_exists(self, name: str) -> bool:
        try:
            await self._docker.volumes.get(name)
            return True
        except Exception:
            return False

    @staticmethod
    async def _exec_archive_stream(container, cmd_str: str):
        """Exec ``cmd_str`` in ``container`` and yield raw stdout bytes."""
        exec_obj = await container.exec(
            cmd=["sh", "-c", cmd_str],
            stdin=False,
            stdout=True,
            stderr=False,
            tty=False,
        )
        stream = exec_obj.start(detach=False)
        while True:
            msg = await stream.read_out()
            if msg is None:
                break
            yield msg.data

    # ------------------------------------------------------------------
    # Public tool-facing API
    # ------------------------------------------------------------------

    async def run_command(
        self,
        workspace_id: str,
        cmd: list[str],
        timeout: float | None = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Callable[[bytes], Awaitable[None]] | None = None,
        environment: dict[str, str] | None = None,
        login_shell: bool = False,
        strict: bool = False,
    ) -> str:
        """Run a command in the workspace container and return stdout.

        ``environment``: extra env vars for THIS exec only (not the
        container). Used to hand a credential (e.g. CLAUDE_CODE_OAUTH_TOKEN)
        to the process via env rather than the command line, so it never
        appears in ``ps``/logs.

        ``login_shell``: run ``cmd`` through ``bash -lc`` (login shell) so it
        sources ``/etc/profile.d`` and sees the persistent-volume PATH
        (``.venv/bin``, ``npm-global/bin``) — the SAME environment the
        interactive terminal / shell_exec / test_run get. Required to invoke a
        bare-name tool installed on-demand into the volume prefix (e.g. the
        ``claude`` CLI); a direct exec resolves argv[0] against the image PATH
        only and fails with ``executable file not found in $PATH``.

        Public entry point for tools and other callers. Delegates to
        :meth:`_run_command` to preserve the existing call sites; future
        refactors should migrate internal callers to this name and drop
        the leading-underscore alias entirely.

        ``timeout``: wall-clock cap in seconds, or ``None`` for no cap.
        ``None`` is for genuinely open-ended work (a background agent
        session) where any fixed budget is arbitrary — liveness there is
        ``idle_timeout``'s job, which measures whether the process is still
        DOING anything rather than how long it has been doing it. On expiry
        the remote process is signalled, not merely detached from.

        ``strict``: raise :class:`ExecAborted` on abnormal termination
        (wall-clock expiry, idle-kill, or non-zero exit) instead of returning
        partial output with a bracketed marker appended. Streaming callers —
        anything consuming bytes via ``on_chunk`` and ignoring the return
        value — should pass ``strict=True``, otherwise the explanation of why
        the command died is written into a string nobody reads.

        ``idle_timeout``: optional per-chunk timeout (in seconds). When
        set, the read loop kills the exec if no bytes arrive within
        ``idle_timeout`` even if ``timeout`` (the wall-clock cap)
        hasn't expired. Designed for long-running operations that stream
        progress (``apt-get install``, ``cargo build``, ``npm install``,
        ``docker pull``) — as long as the command is actively producing
        output, it keeps running up to ``timeout``; once it goes quiet,
        idle-kill fires quickly. Omit to use pure wall-clock behaviour.

        ``progress_path``: optional path inside the container. When set
        AND the idle timer is about to fire, the handler first stats
        the file and checks whether it has grown since the previous
        check. If yes, the idle timer resets and a human-readable
        progress line is APPENDED to the returned output (visible in
        the shell_exec result). If no, the stall counter ticks; only
        after the stall sustains across the full idle_timeout window
        is the exec actually killed. This is what lets ``wget -q`` /
        ``curl -s`` downloads that emit nothing on stdout still keep
        running as long as bytes are landing on disk.

        ``on_chunk``: optional async callback invoked with each non-
        empty stdout/stderr chunk AS IT ARRIVES from docker exec. Lets
        callers stream live output to the UI during long commands
        (``pytest``, ``npm install``, ``docker build``) instead of
        waiting for the full result at process exit. Failures inside
        the callback are swallowed so a misbehaving consumer can't
        kill an otherwise-healthy shell run. Synthetic chunks (e.g.,
        download-progress heartbeats) are also forwarded so the user
        sees them live too.
        """
        return await self._run_command(
            workspace_id, cmd, timeout=timeout,
            idle_timeout=idle_timeout, progress_path=progress_path,
            on_chunk=on_chunk, environment=environment,
            login_shell=login_shell, strict=strict,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _revive_container(self, container: object, workspace_id: str) -> None:
        """Start a stopped/exited workspace container and wait for it to run.

        Called by the exec hot-path when Docker reports the container is not
        running. Re-running the entrypoint re-provisions a fallback image, but
        the persistent ``/workspace`` volume and the container's own writable
        layer survive a stop/start — and the keep-alive is now decoupled from
        provisioning, so the revived container stays up regardless. Raises
        ``RuntimeError`` if it can't be brought back so the caller surfaces a
        clear failure instead of looping on 409s.
        """
        # Why did it die? An OOM kill (exit 137 / OOMKilled) under the 2GB,
        # no-swap cap is the prime suspect for a container that dies under a
        # heavy build/test — distinct from a clean stop or a daemon restart.
        # Surface it explicitly so a RECURRING OOM is diagnosable instead of
        # hiding behind a generic revive. (Best-effort; never blocks revival.)
        try:
            pre = await container.show()  # type: ignore[attr-defined]
            state = pre.get("State") or {}
            if state.get("OOMKilled") or state.get("ExitCode") == 137:
                log.warning(
                    "run_command_container_oom",
                    workspace_id=workspace_id,
                    oom_killed=bool(state.get("OOMKilled")),
                    exit_code=state.get("ExitCode"),
                    detail=(
                        "workspace container was killed (likely out of memory). "
                        "The workload may exceed its memory limit — consider a "
                        "larger workspace or lighter concurrent steps."
                    ),
                )
        except Exception:
            pass
        try:
            await container.start()  # type: ignore[attr-defined]
        except Exception as exc:
            # A concurrent revive (another tool call racing us) may have already
            # started it — Docker says "already started"/"not paused". Treat
            # those as benign; anything else is a real, unrecoverable failure.
            msg = str(exc).lower()
            if "already started" not in msg and "not paused" not in msg:
                raise RuntimeError(
                    f"workspace container could not be revived: {exc}"
                ) from exc
        # Wait (briefly) for the daemon to report Running, then reconcile the
        # DB row so later calls skip the not-running branch entirely.
        for _ in range(20):  # ~2s ceiling
            try:
                details = await container.show()  # type: ignore[attr-defined]
            except Exception:
                break
            if (details.get("State") or {}).get("Running"):
                if self._db is not None:
                    await self._db.execute(
                        "UPDATE project_checkouts SET status=?, last_active=? WHERE id=?",
                        ("running", time.time(), workspace_id),
                    )
                    await self._db.commit()
                log.info("run_command_container_revived", workspace_id=workspace_id)
                return
            await asyncio.sleep(0.1)
        # Didn't observe Running within the window. Don't hard-fail here — the
        # immediate exec retry will either succeed (slow daemon) or raise its
        # own clear 409 the caller can report.
        log.warning(
            "run_command_container_revive_unconfirmed", workspace_id=workspace_id,
        )

    async def _run_command(
        self,
        workspace_id: str,
        cmd: list[str],
        timeout: float | None = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Callable[[bytes], Awaitable[None]] | None = None,
        environment: dict[str, str] | None = None,
        login_shell: bool = False,
        strict: bool = False,
    ) -> str:
        """Run a non-interactive command in the workspace container and return stdout.

        Args:
            timeout: Wall-clock ceiling — kill after this many seconds
                regardless of activity. Default 30s. ``None`` disables the
                ceiling entirely, for open-ended background work whose
                duration can't be known up front; pair it with
                ``idle_timeout`` so liveness is still bounded.
            idle_timeout: Optional per-chunk silence ceiling. When set,
                kills the exec if no bytes arrive within this many
                seconds even while ``timeout`` still has budget. Used
                by the shell_exec tool for download-heavy workloads
                (apt-get, cargo, npm) so a 10-minute total cap doesn't
                keep a hung process alive for the full duration once
                it stops making progress.
            strict: Raise :class:`ExecAborted` on abnormal termination
                instead of returning partial output with a bracketed
                marker. Required for correctness in streaming callers,
                which never read the return value.

        Both timeout paths SIGNAL the remote process before returning.
        They previously only stopped reading its stdout, which left the
        process running orphaned inside the container — still writing
        files, with nothing observing it.

        Note: ``workdir="/workspace"`` is passed to every exec so relative
        paths in commands (``cat README.md``, ``ls``) resolve against the
        workspace root. Observed 2026-04-20 without this: the model
        passed a relative ``README.md`` to file_read; cat ran from the
        container's default /, returned ENOENT; every file_read failed
        silently and the agent burned 10 iters reading non-existent
        files before a circuit breaker fired. The interactive-shell
        exec (line 446) has this; _run_command had been inconsistent.

        Prefer :meth:`run_command` in new code — leading underscore
        kept only to avoid churning every existing tool call site in
        one commit.
        """
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            raise ValueError(f"Workspace {workspace_id} has no associated container")

        # An active exec IS activity. Bump last_active (debounced) so the idle
        # reaper can't pause/stop a workspace mid-command when the UI tab is
        # closed and nothing is polling. See _touch_last_active.
        await self._touch_last_active(workspace_id)

        container = await self._docker.containers.get(info.container_id)
        # If the idle reaper paused this workspace between turns,
        # docker exec on a frozen cgroup will hang forever waiting on
        # processes that can't respond. Thaw first; the DB write back
        # to 'running' lets subsequent calls skip this branch.
        if info.status == "paused":
            async def _mark_running():
                info.status = "running"
                if self._db is not None:
                    await self._db.execute(
                        "UPDATE project_checkouts SET status=?, last_active=? WHERE id=?",
                        ("running", time.time(), workspace_id),
                    )
                    await self._db.commit()

            try:
                await container.unpause()
                await _mark_running()
                log.info("run_command_unpause", workspace_id=workspace_id)
            except Exception:
                # DB/cache said "paused" but Docker rejected the unpause —
                # most often the container is actually already running (the
                # idle-reaper's state drifted, or a daemon/host restart
                # cleared the pause), so Docker returns "[500] ... is not
                # paused". Reconcile to the container's real state instead of
                # leaving status="paused": otherwise EVERY subsequent exec on
                # this workspace (file_list on open, etc.) retries unpause and
                # 500s again, spamming tracebacks and adding latency. Only a
                # genuinely non-running container is a real failure.
                real_status = ""
                try:
                    details = await container.show()
                    real_status = (details.get("State") or {}).get("Status", "")
                except Exception:
                    pass
                if real_status == "running":
                    await _mark_running()
                    log.info(
                        "run_command_unpause_reconciled",
                        workspace_id=workspace_id,
                    )
                else:
                    log.warning(
                        "run_command_unpause_failed",
                        workspace_id=workspace_id,
                        real_status=real_status, exc_info=True,
                    )
        _exec_env = [f"{k}={v}" for k, v in (environment or {}).items()]

        # A direct ``docker exec`` resolves argv[0] against the image's default
        # PATH — it does NOT source ``/etc/profile.d``, so tools installed into
        # the persistent volume prefix (``/workspace/.venv/bin``,
        # ``/workspace/.augmentum/npm-global/bin``) are invisible by bare name.
        # ``login_shell=True`` routes the command through ``bash -lc`` so it
        # sees the SAME PATH the interactive terminal / shell_exec / test_run
        # already get. Without this, a bare-name tool installed on-demand into
        # the volume (e.g. the ``claude`` CLI) passes a ``bash -lc`` existence
        # probe yet dies at exec time with ``executable file not found in
        # $PATH``. See _PERSIST_ENV_SH.
        _exec_cmd = ["bash", "-lc", shlex.join(cmd)] if login_shell else cmd

        async def _make_exec():
            kwargs = dict(
                cmd=_exec_cmd,
                stdin=False,
                stdout=True,
                stderr=True,
                tty=False,
                workdir="/workspace",
            )
            if _exec_env:
                kwargs["environment"] = _exec_env
            return await container.exec(**kwargs)

        try:
            exec_obj = await _make_exec()
        except Exception as exc:
            # The container exited out from under us — a crash, an OOM kill,
            # a daemon restart, or (pre the keep-alive fix) a provisioning
            # death. docker exec on a stopped container raises
            # ``[409] … is not running``. Historically this 409'd EVERY
            # subsequent tool call for the rest of the turn (shell_exec,
            # list_files, the snapshot scan), so the agent burned a dozen
            # iterations against a corpse. Revive it once and retry so the
            # turn self-heals; if it can't be revived, surface a clear
            # error instead of looping.
            if not _is_container_not_running(exc):
                raise
            log.warning(
                "run_command_container_not_running_reviving",
                workspace_id=workspace_id, container_id=info.container_id,
            )
            await self._revive_container(container, workspace_id)
            exec_obj = await _make_exec()
        # Register the active exec so a concurrent user-cancel can
        # signal the remote process. ``exec_obj.id`` exposes the
        # docker exec_id; combined with the container reference, we
        # have everything needed to issue a ``kill <pid>`` via the
        # container. Entries are removed in ``finally`` so the set
        # never grows unboundedly.
        exec_id = getattr(exec_obj, "id", None) or getattr(exec_obj, "_id", "")
        self._active_execs.setdefault(workspace_id, set()).add(
            (exec_obj, exec_id),
        )
        stream = exec_obj.start(detach=False)
        chunks: list[bytes] = []
        idle_killed = False

        # Active-download liveness. When ``progress_path`` is set we
        # poll the target file at a shorter tick (max 30s) and treat
        # file growth as "the command is still working" — emitting a
        # heartbeat into ``chunks`` and resetting the stall counter.
        # Only when the file stays at a fixed size across the FULL
        # ``idle_timeout`` window do we declare it hung. This matches
        # user expectation for silent ``wget -q`` / ``curl -s`` downloads
        # whose only observable liveness signal is bytes on disk.
        PROGRESS_TICK = 30.0
        if progress_path and idle_timeout is not None:
            effective_tick = min(idle_timeout, PROGRESS_TICK)
            max_stalled_ticks = max(1, int(idle_timeout / effective_tick))
        else:
            effective_tick = idle_timeout
            max_stalled_ticks = 1  # unused when progress_path is None

        async def _forward_chunk(data: bytes) -> None:
            """Forward live bytes to the on_chunk sink, never raising.

            Consumer errors are logged but never propagated: a broken
            UI subscription must not be allowed to kill an otherwise-
            healthy shell run (think: 45-minute build hijacked by a
            transient WebSocket close).
            """
            if not on_chunk or not data:
                return
            try:
                await on_chunk(data)
            except Exception as exc:
                log.debug(
                    "run_command_on_chunk_failed",
                    workspace=workspace_id, error=str(exc),
                )

        async def _read_stream():
            nonlocal idle_killed
            last_size = -1
            stalled_ticks = 0
            while True:
                # ``effective_tick`` (if set) bounds each individual
                # read. A long-running process streaming progress
                # keeps resetting this budget, so only TRUE stalls
                # (no bytes for ``idle_timeout`` seconds AND, when
                # applicable, no disk growth) kill it. ``timeout``
                # (outer wait_for) is the absolute cap.
                if effective_tick is not None:
                    try:
                        msg = await asyncio.wait_for(
                            stream.read_out(), timeout=effective_tick,
                        )
                    except TimeoutError:
                        # Stdout went quiet. If we're watching a
                        # download target, check whether bytes are
                        # still landing before calling it hung.
                        if progress_path:
                            current_size = await self._stat_path_size(
                                container, progress_path,
                            )
                            if current_size is not None and current_size > last_size:
                                delta = current_size - max(last_size, 0)
                                heartbeat = (
                                    f"[download progress: "
                                    f"{_fmt_bytes(current_size)} "
                                    f"(+{_fmt_bytes(delta)} in "
                                    f"{int(effective_tick)}s) — "
                                    f"{progress_path}]\n"
                                ).encode()
                                chunks.append(heartbeat)
                                await _forward_chunk(heartbeat)
                                last_size = current_size
                                stalled_ticks = 0
                                continue
                            # File didn't grow this tick — might be a
                            # real stall or simply a slow remote. Bump
                            # the stall counter; only kill when we've
                            # accumulated the full idle budget worth of
                            # consecutive no-growth ticks.
                            stalled_ticks += 1
                            if stalled_ticks < max_stalled_ticks:
                                continue
                        idle_killed = True
                        break
                else:
                    msg = await stream.read_out()
                if msg is None:
                    break
                if msg.data:
                    chunks.append(msg.data)
                    await _forward_chunk(msg.data)
                    # Stdout produced bytes — reset download stall
                    # counter so a mostly-silent download that emits
                    # a single chunk every 4 minutes isn't killed.
                    stalled_ticks = 0

        try:
            try:
                await asyncio.wait_for(_read_stream(), timeout=timeout)
            except TimeoutError:
                log.warning("run_command_timeout", workspace=workspace_id,
                            cmd=cmd[:3], timeout=timeout)
                # Terminate the remote process. ``wait_for`` only cancels the
                # PYTHON coroutine reading stdout — detaching from a docker
                # exec does NOT signal the process, exactly like closing a
                # ``docker exec`` client. Without this the exec keeps running
                # orphaned: unobserved, unlogged, still mutating /workspace,
                # and (for an agent) still spending the user's tokens. Only
                # the user-cancel path ever killed; the two timeout paths did
                # not, so a timed-out run kept editing files for as long as it
                # liked after the UI had already marked it failed.
                try:
                    await self._kill_exec(container, exec_obj, exec_id)
                except Exception:
                    log.warning(
                        "run_command_timeout_kill_failed",
                        workspace=workspace_id, exec_id=exec_id, exc_info=True,
                    )
                output = b"".join(chunks).decode("utf-8", errors="replace")
                detail = (
                    f"Command exceeded its {timeout}s wall-clock budget and was "
                    "terminated."
                )
                if strict:
                    raise ExecAborted("wall_clock", detail, partial=output) from None
                return output + f"\n\n[{detail}]"
            except asyncio.CancelledError:
                # User cancel (Ctrl+C / Esc) — kill the remote exec
                # before re-raising so the docker process doesn't keep
                # running after the Python task exits. Best-effort:
                # failures during kill are logged, not raised, so the
                # original CancelledError always surfaces to the
                # caller.
                log.info(
                    "run_command_cancelled", workspace=workspace_id,
                    cmd=cmd[:3],
                )
                try:
                    await self._kill_exec(container, exec_obj, exec_id)
                except Exception:
                    log.warning(
                        "run_command_cancel_kill_failed",
                        workspace=workspace_id, exec_id=exec_id,
                        exc_info=True,
                    )
                raise
        finally:
            self._active_execs.get(workspace_id, set()).discard(
                (exec_obj, exec_id),
            )

        output = b"".join(chunks).decode("utf-8", errors="replace")
        if idle_killed:
            log.warning(
                "run_command_idle_timeout", workspace=workspace_id,
                cmd=cmd[:3], idle_timeout=idle_timeout,
                progress_path=progress_path,
            )
            # The messages below have always SAID "was killed" — but nothing
            # ever signalled the process: ``_read_stream`` just set the flag
            # and broke out of the loop. Same orphan as the wall-clock path.
            # Make the claim true before we make it.
            try:
                await self._kill_exec(container, exec_obj, exec_id)
            except Exception:
                log.warning(
                    "run_command_idle_kill_failed",
                    workspace=workspace_id, exec_id=exec_id, exc_info=True,
                )
            if progress_path:
                # Active-download path: we were watching a file AND it
                # hadn't grown for the full idle window. That's a real
                # stall — remote likely closed the connection, resume
                # with the same command (wget -c / curl -C) is probably
                # what the user wants.
                output += (
                    f"\n\n[Download stalled — {progress_path} stopped "
                    f"growing for {idle_timeout}s and was killed. The "
                    "remote likely dropped the connection. Re-run with "
                    "a resume flag (``wget -c``, ``curl -C -``) or "
                    "``aria2c -c`` to continue from the partial file.]"
                )
            else:
                output += (
                    f"\n\n[Command went silent for {idle_timeout}s and "
                    "was killed — still running after producing no "
                    "output. Re-run with a concrete check of state "
                    "(ps, logs) or pass an explicit longer idle "
                    "timeout if you expected a long pause.]"
                )
            if strict:
                raise ExecAborted(
                    "idle",
                    f"Command produced no output for {idle_timeout}s and was "
                    "terminated as hung.",
                    partial=output,
                )
            return output

        # Normal end-of-stream. Recover the process's exit code so a crash is
        # reportable as a crash. Without this a non-zero exit is indistinguishable
        # from a clean finish at this layer, which is how a dead CLI reached the
        # user as a generic "ended without a result". Best-effort: an inspect
        # failure must not turn a successful command into an error.
        if strict:
            exit_code: int | None = None
            try:
                inspect_fn = getattr(exec_obj, "inspect", None)
                if inspect_fn is not None:
                    info = await inspect_fn()
                    if isinstance(info, dict):
                        raw = info.get("ExitCode")
                        if isinstance(raw, int):
                            exit_code = raw
            except Exception:
                log.debug("run_command_exit_inspect_failed", exc_info=True)
            if exit_code:
                raise ExecAborted(
                    "exit_code",
                    f"Command exited with code {exit_code}.",
                    partial=output,
                )
        return output

    async def _kill_exec(
        self, container: object, exec_obj: object, exec_id: str,
    ) -> None:
        """Signal a specific remote exec so it stops cooperatively.

        Docker doesn't expose a ``kill_exec`` primitive — execs run
        inside the container's PID namespace, so we have to inspect
        the exec to learn its PID then run ``kill`` as a fresh exec.
        SIGTERM first, SIGKILL fallback if ``pid`` can't be recovered.
        Best-effort: callers should treat exceptions as non-fatal.
        """
        if not exec_id:
            return
        pid: int | None = None
        try:
            inspect_fn = getattr(exec_obj, "inspect", None)
            if inspect_fn is not None:
                info = await inspect_fn()
                # aiodocker returns a dict with "Pid" at the top level.
                raw_pid = (
                    (info or {}).get("Pid")
                    if isinstance(info, dict) else None
                )
                if isinstance(raw_pid, int) and raw_pid > 0:
                    pid = raw_pid
        except Exception:
            log.debug("kill_exec_inspect_failed", exc_info=True)

        if pid is not None:
            # Signal the process tree started by the exec. ``kill
            # -TERM -<pid>`` would signal the whole group but process
            # groups aren't guaranteed for every exec; TERM the main
            # pid and any children named similarly instead.
            try:
                kill_exec = await container.exec(
                    cmd=["bash", "-c", f"kill -TERM {pid} 2>/dev/null; kill -TERM -{pid} 2>/dev/null || true"],
                    stdin=False, stdout=True, stderr=True, tty=False,
                )
                # Start + drain quickly; we don't care about output.
                s = kill_exec.start(detach=False)
                try:
                    await asyncio.wait_for(s.read_out(), timeout=2.0)
                except Exception as drain_exc:
                    log.debug("kill_exec_drain_failed_pid", pid=pid, error=str(drain_exc))
            except Exception as exc:
                log.debug("kill_exec_pid_failed", pid=pid, error=str(exc))
        else:
            # No pid available — fall back to a best-effort container-
            # wide ``pkill`` on all children of PID 1 that aren't the
            # container's foreground (``tail -f /dev/null``). Blunt
            # but effective when a user cancel needs to actually stop
            # a build.
            try:
                kill_exec = await container.exec(
                    cmd=["bash", "-c", "pkill -TERM -P 1 -x -v 'tail' 2>/dev/null || true"],
                    stdin=False, stdout=True, stderr=True, tty=False,
                )
                s = kill_exec.start(detach=False)
                try:
                    await asyncio.wait_for(s.read_out(), timeout=2.0)
                except Exception as drain_exc:
                    log.debug("kill_exec_drain_failed_pkill", error=str(drain_exc))
            except Exception as exc:
                log.debug("kill_exec_pkill_failed", error=str(exc))

    async def _stat_path_size(
        self, container: object, path: str,
    ) -> int | None:
        """Return the file size of ``path`` inside ``container``, or None.

        Used by :meth:`_run_command`'s active-download liveness check.
        Runs a tiny ``stat -c %s`` exec; the docker round-trip is
        cheap (<50ms typically). Missing files, permission errors,
        and any other failure return ``None`` — the caller treats
        that as "no growth", which is the correct fail-closed default
        (if we can't verify progress, don't keep a potentially-hung
        exec alive indefinitely).
        """
        if not path:
            return None
        try:
            stat_exec = await container.exec(
                cmd=["stat", "-c", "%s", path],
                stdin=False, stdout=True, stderr=True, tty=False,
            )
            s = stat_exec.start(detach=False)
            raw = await asyncio.wait_for(s.read_out(), timeout=3.0)
            if raw is None or not getattr(raw, "data", b""):
                return None
            text = raw.data.decode("utf-8", errors="replace").strip()
            # stat -c %s prints the size on its own line; strip anything
            # beyond the first integer token defensively.
            first = text.split(None, 1)[0] if text else ""
            return int(first)
        except Exception:
            log.debug("stat_path_size_failed", path=path, exc_info=True)
            return None

    async def cancel_workspace_execs(self, workspace_id: str) -> int:
        """Signal all active execs for a workspace. Returns the count.

        Called by the coder handler in its CancelledError cleanup
        path so a user-initiated cancel actually stops in-flight
        builds / installs inside the container, not just the Python
        task wrapping them.
        """
        execs = list(self._active_execs.get(workspace_id, set()))
        if not execs:
            return 0
        info = await self._get_workspace(workspace_id)
        if info.container_id is None:
            return 0
        container = await self._docker.containers.get(info.container_id)
        killed = 0
        for exec_obj, exec_id in execs:
            try:
                await self._kill_exec(container, exec_obj, exec_id)
                killed += 1
            except Exception:
                log.warning(
                    "cancel_workspace_exec_failed",
                    workspace=workspace_id, exec_id=exec_id,
                    exc_info=True,
                )
        return killed

    async def _reconcile_to_stopped(
        self,
        workspace_id: str,
        *,
        clear_container_id: bool = False,
    ) -> None:
        """Write ``status='stopped'`` (and optionally null out ``container_id``)
        for a workspace whose Docker container has gone out from under us.

        Called by lifecycle methods when they see "container missing" or
        "not in expected state" errors — keeps the DB row from being
        wedged in a phantom-running / phantom-paused state that the
        idle reaper would then bang on every sweep.

        ``clear_container_id=True`` is for the 404-gone case where the
        stale id points at nothing in Docker; start() will then need a
        recreate path on the next user attempt rather than re-trying
        the dead id forever.
        """
        if self._db is None:
            return
        try:
            if clear_container_id:
                await self._db.execute(
                    "UPDATE project_checkouts SET status=?, container_id=NULL "
                    "WHERE id=?",
                    ("stopped", workspace_id),
                )
            else:
                await self._db.execute(
                    "UPDATE project_checkouts SET status=? WHERE id=?",
                    ("stopped", workspace_id),
                )
            await self._db.commit()
            self._invalidate_docker_state_cache()
        except Exception:
            log.warning(
                "workspace_reconcile_writeback_failed",
                workspace_id=workspace_id, exc_info=True,
            )

    async def _get_workspace(self, workspace_id: str) -> ContainerInfo:
        """Fetch a workspace by ID from the database.

        Raises :class:`KeyError` if the workspace is not found.
        """
        if self._db is None:
            raise KeyError(f"Workspace {workspace_id} not found (no DB)")

        row = await self._db.execute_fetchall(
            "SELECT id, name, container_id, status, template_id, git_url, "
            "created_at, last_active, resources_cpu, resources_memory, "
            "safeguards_enabled, tooling_profile, "
            "kind, bug_finder_verifier_model, project_id, "
            "planning_mode, always_on, lan_accessible "
            "FROM project_checkouts WHERE id=?",
            (workspace_id,),
        )
        if not row:
            raise KeyError(f"Workspace {workspace_id} not found")

        r = row[0]
        return ContainerInfo(
            id=r[0],
            name=r[1],
            container_id=r[2],
            status=r[3],
            template_id=r[4],
            git_url=r[5],
            created_at=r[6],
            last_active=r[7],
            resources_cpu=r[8],
            resources_memory=r[9],
            safeguards_enabled=bool(r[10]) if r[10] is not None else True,
            tooling_profile=r[11] or "browser",
            kind=r[12] or "regular",
            bug_finder_verifier_model=r[13] or "",
            project_id=r[14] or "",
            planning_mode=r[15] or "auto",
            always_on=bool(r[16]) if r[16] is not None else False,
            lan_accessible=bool(r[17]) if r[17] is not None else False,
        )

    async def _resolve_bare_repo_volume(self) -> str:
        """Name of the Docker volume backing ``{data_dir}``, resolved once.

        ``coder_bare_repo_volume_name`` defaults to the compose volume
        KEY (``augmentum_data``), but Docker Compose prefixes volume
        names with the project name — a default checkout's real volume
        is ``augmentum_augmentum_data``. Mounting the un-prefixed name
        makes Docker auto-create an EMPTY volume by that name, the
        Subpath validation lstats ``projects/...`` inside it, and every
        project-linked workspace create 500s. The bare repo is written
        to ``{data_dir}/projects`` by definition, so the volume mounted
        at ``{data_dir}`` in THIS container is authoritative — inspect
        our own mounts (container id == hostname) and use that name.
        The setting remains the fallback for deployments where
        self-inspection isn't possible (native dev, mocked Docker).
        """
        if self._bare_repo_volume_resolved is not None:
            return self._bare_repo_volume_resolved
        from augmentum.config import settings as _settings
        detected = ""
        try:
            own = await self._docker.containers.get(socket.gethostname())
            details = await own.show()
            data_dir = (_settings.data_dir or "/data").rstrip("/")
            for m in details.get("Mounts") or []:
                if (
                    m.get("Type") == "volume"
                    and (m.get("Destination") or "").rstrip("/") == data_dir
                    and m.get("Name")
                ):
                    detected = m["Name"]
                    break
        except Exception as exc:
            # Expected when running outside Docker or in tests with a
            # mocked client — the configured name takes over below.
            log.warning(
                "bare_repo_volume_self_inspect_failed",
                fallback=_settings.coder_bare_repo_volume_name,
                error=str(exc),
            )
        configured = _settings.coder_bare_repo_volume_name
        if detected and detected != configured:
            log.info(
                "bare_repo_volume_detected",
                detected=detected,
                configured=configured,
            )
        self._bare_repo_volume_resolved = detected or configured
        return self._bare_repo_volume_resolved

    async def _ensure_project_for_checkout(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
    ) -> str:
        """Ensure a Project + bare repo exist; return the project_id.

        If ``project_id`` is non-empty, validate it belongs to ``user_id``
        and ensure its bare repo. If empty, mint a new Project for this
        checkout and ensure its bare repo. Empty return = caller should
        skip the bare-repo bind.
        """
        from augmentum.config import settings as _settings
        from augmentum.projects import ProjectRepoStorage, ProjectStore

        storage = ProjectRepoStorage(
            Path(_settings.data_dir) / "projects",
        )
        store = ProjectStore(self._db, storage)

        if project_id:
            row = await store.get(project_id, user_id=user_id)
            if row is None:
                log.warning(
                    "project_id_not_found_for_checkout",
                    project_id=project_id, user_id=user_id,
                )
                return ""
        else:
            project = await store.create(
                user_id=user_id,
                name=name,
                kind="coder",
                origin="checkout_auto",
            )
            project_id = project["id"]

        await store.ensure_bare_repo(project_id, user_id=user_id)
        return project_id

    async def _persist_workspace(self, info: ContainerInfo) -> None:
        """Insert a new workspace record into the database.

        ``info.user_id`` is stamped atomically here. When empty (legacy /
        unauthenticated callers), the column is left NULL — defensible
        because all live routes pass a non-empty user_id post-auth-rollout.
        """
        kind = (info.kind or "regular").strip() or "regular"
        verifier_model = (info.bug_finder_verifier_model or "").strip() or None
        project_id = (info.project_id or "").strip() or None
        lan_acc = 1 if info.lan_accessible else 0
        if info.user_id:
            await self._db.execute(
                "INSERT INTO project_checkouts "
                "(id, name, container_id, status, template_id, git_url, "
                "created_at, last_active, resources_cpu, resources_memory, "
                "tooling_profile, user_id, kind, bug_finder_verifier_model, "
                "project_id, lan_accessible) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.id,
                    info.name,
                    info.container_id,
                    info.status,
                    info.template_id,
                    info.git_url,
                    info.created_at,
                    info.last_active,
                    info.resources_cpu,
                    info.resources_memory,
                    info.tooling_profile,
                    info.user_id,
                    kind,
                    verifier_model,
                    project_id,
                    lan_acc,
                ),
            )
        else:
            await self._db.execute(
                "INSERT INTO project_checkouts "
                "(id, name, container_id, status, template_id, git_url, "
                "created_at, last_active, resources_cpu, resources_memory, "
                "tooling_profile, kind, bug_finder_verifier_model, "
                "project_id, lan_accessible) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.id,
                    info.name,
                    info.container_id,
                    info.status,
                    info.template_id,
                    info.git_url,
                    info.created_at,
                    info.last_active,
                    info.resources_cpu,
                    info.resources_memory,
                    info.tooling_profile,
                    kind,
                    verifier_model,
                    project_id,
                    lan_acc,
                ),
            )
        await self._db.commit()
