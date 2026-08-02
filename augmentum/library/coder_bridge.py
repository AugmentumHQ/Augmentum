"""Coder → Library bridge.

The save route needs three things the coder subsystem already knows:

1. **What's the preview running?** (workspace's listening ports +
   matching workspace_service rows → served dir + entry point hint)
2. **Is it static or dynamic?** (probe the running service, classify
   the response — see :mod:`augmentum.library.preview_kind`)
3. **What bytes to snapshot?** (tar the served dir out of the running
   container)

These compose existing primitives — `ContainerManager.list_ports`,
`CoderServiceStore.list`, `Container.get_archive`. The bridge gives
the save route a single coherent contract instead of stitching three
modules together inline.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from augmentum.library.preview_kind import PreviewKind, probe_preview
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


@dataclass
class PreviewSnapshot:
    """Everything the save route needs about a workspace's current preview.

    ``preview_kind`` controls UI gating:

    * ``"static"``  — save button enabled
    * ``"dynamic"`` — save button visible but disabled with hint
    * ``"none"``    — save button hidden (no preview running)
    * ``"unknown"`` — save button hidden (couldn't classify; treat as none)
    """

    preview_kind: Literal["static", "dynamic", "none", "unknown"]
    primary_url: str | None
    served_dir: str | None       # container path, e.g. /workspace/dist
    container_port: int | None   # port the dev server listens on
    host_port: int | None        # docker-mapped port, for the static probe
    service_name: str | None
    file_count: int = 0
    estimated_size_bytes: int = 0

    @property
    def saveable(self) -> bool:
        return self.preview_kind == "static" and bool(self.served_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview_ready": self.saveable,
            "preview_kind": self.preview_kind,
            "served_dir": self.served_dir,
            "primary_url": self.primary_url,
            "container_port": self.container_port,
            "service_name": self.service_name,
            "estimated_size_bytes": self.estimated_size_bytes,
            "file_count": self.file_count,
        }


async def gather_preview_state(
    *,
    request: Request,
    workspace_id: str,
    user_id: str,
) -> PreviewSnapshot:
    """Inspect the workspace's running preview.

    Reads (no mutation):

    * ``container_manager.list_ports`` → which ports are listening
    * ``CoderServiceStore.list`` → which service is serving from where
    * HTTP probe of the listening service → static vs dynamic
    * ``du`` + ``find`` inside the container → size + file count

    Failures degrade gracefully — a probe timeout returns
    ``preview_kind="unknown"`` rather than 500ing the save route.
    """
    from augmentum.coder.services import CoderServiceStore  # local; avoids import cycle

    container_mgr = getattr(request.app.state, "container_manager", None)
    if container_mgr is None:
        return PreviewSnapshot(
            preview_kind="none", primary_url=None, served_dir=None,
            container_port=None, host_port=None, service_name=None,
        )

    try:
        ports = await container_mgr.list_ports(workspace_id)
    except Exception:
        log.warning("library_preview_list_ports_failed", workspace=workspace_id, exc_info=True)
        return PreviewSnapshot(
            preview_kind="none", primary_url=None, served_dir=None,
            container_port=None, host_port=None, service_name=None,
        )

    listening = [
        p for p in (ports or [])
        if bool(p.get("listening")) and int(p.get("host_port") or 0) > 0
    ]
    if not listening:
        return PreviewSnapshot(
            preview_kind="none", primary_url=None, served_dir=None,
            container_port=None, host_port=None, service_name=None,
        )
    chosen = listening[0]
    container_port = int(chosen["container_port"])
    host_port = int(chosen["host_port"])

    # Match the listening port to a registered service so we know what dir.
    conn = _conn_from_request(request)
    served_dir: str | None = None
    service_name: str | None = None
    if conn is not None:
        try:
            services = await CoderServiceStore(conn).list(
                user_id=user_id, workspace_id=workspace_id,
            )
            for svc in services:
                if container_port in (svc.ports or []):
                    served_dir = svc.cwd or "/workspace"
                    service_name = svc.name
                    break
        except Exception:
            log.warning(
                "library_preview_service_lookup_failed",
                workspace=workspace_id, exc_info=True,
            )

    # If we couldn't pin the service down, fall back to /workspace. The
    # user can still save — they'll just snapshot the workspace root.
    if served_dir is None:
        served_dir = "/workspace"

    primary_url = f"/api/coder/preview/{workspace_id}/{container_port}/"

    # Probe the dev server via the docker-host loopback. From inside the
    # Augmentum container, 127.0.0.1 is the container's own loopback —
    # NOT where Docker publishes the workspace's host_port. The existing
    # coder preview-proxy solves this with `_resolve_proxy_host` which
    # probes `host.docker.internal` first, then `172.17.0.1` (the bridge
    # gateway), and caches whichever connects. Reuse it directly so the
    # save-preflight and the iframe preview agree on which host to hit.
    probe_host: str | None = "127.0.0.1"
    try:
        from augmentum.proxy.coder_routes import _resolve_proxy_host
        resolved = await _resolve_proxy_host(request.app, host_port)
        if resolved:
            probe_host = resolved
    except Exception:
        log.debug("library_preview_host_resolve_failed", host_port=host_port, exc_info=True)
    probe_url = f"http://{probe_host}:{host_port}/"
    preview_kind: PreviewKind = await probe_preview(probe_url, timeout_seconds=4.0)
    if preview_kind == "unknown" and not listening:
        # Be explicit about the "no preview" state vs. "preview that
        # couldn't be classified" — they look the same to a UI but
        # only the latter is an error condition.
        kind: Literal["static", "dynamic", "none", "unknown"] = "none"
    else:
        kind = preview_kind

    # Best-effort size + file-count from inside the container. Failure
    # here doesn't gate the save — the UI just shows blanks. Logged at
    # debug so it's observable without producing warnings on every
    # preflight against an idle workspace.
    file_count = 0
    estimated_size_bytes = 0
    try:
        count_str = await _safe_run(container_mgr, workspace_id, [
            "sh", "-c",
            f"find {_quote(served_dir)} -type f 2>/dev/null | wc -l",
        ])
        file_count = int((count_str or "0").strip() or 0)
    except Exception as exc:
        log.debug("library_preview_file_count_failed", workspace=workspace_id, error=str(exc))
    try:
        size_str = await _safe_run(container_mgr, workspace_id, [
            "sh", "-c",
            f"du -sb {_quote(served_dir)} 2>/dev/null | cut -f1",
        ])
        estimated_size_bytes = int((size_str or "0").strip() or 0)
    except Exception as exc:
        log.debug("library_preview_size_estimate_failed", workspace=workspace_id, error=str(exc))

    return PreviewSnapshot(
        preview_kind=kind,
        primary_url=primary_url,
        served_dir=served_dir,
        container_port=container_port,
        host_port=host_port,
        service_name=service_name,
        file_count=file_count,
        estimated_size_bytes=estimated_size_bytes,
    )


async def snapshot_container_path(
    *,
    request: Request,
    workspace_id: str,
    container_path: str,
    host_dest_dir: Path,
) -> Path:
    """Tar a container path out of the workspace and extract to host.

    Returns the host path containing the extracted content (the same
    layout the container had — caller passes this to
    :meth:`LibraryStorage.write_bundle`).

    ``container_path`` is a path INSIDE the container (e.g.
    ``/workspace/dist``). The returned host path will be
    ``host_dest_dir / Path(container_path).name`` because Docker's
    get_archive roots tar members at the leaf name.

    Raises ``FileNotFoundError`` if the path doesn't exist in the
    container, ``RuntimeError`` if Docker isn't configured.
    """
    container_mgr = getattr(request.app.state, "container_manager", None)
    if container_mgr is None:
        raise RuntimeError("container_manager not available")

    info = await container_mgr._get_workspace(workspace_id)
    if not info or info.container_id is None:
        raise RuntimeError(f"workspace {workspace_id} has no container")

    container = await container_mgr._docker.containers.get(info.container_id)
    try:
        tar_obj = await container.get_archive(path=container_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"container path {container_path!r} not found"
        ) from exc

    host_dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        # aiodocker returns a tarfile.TarFile already opened for read.
        _safe_extract_tar(tar_obj, host_dest_dir)
    finally:
        try:
            tar_obj.close()
        except (OSError, tarfile.TarError):
            pass

    # Find the extracted root. get_archive of /workspace/dist roots
    # members at "dist/..." — so the leaf name is what we want.
    leaf = Path(container_path).name or "content"
    candidate = host_dest_dir / leaf
    if candidate.exists():
        return candidate
    # Fallback: if the tar didn't use the expected leaf (e.g. someone
    # passed a path with trailing slash), pick the single top-level
    # entry we extracted.
    children = [p for p in host_dest_dir.iterdir()]
    if len(children) == 1:
        return children[0]
    return host_dest_dir


# ── Helpers ────────────────────────────────────────────────────────────


def _conn_from_request(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


def _quote(path: str) -> str:
    """Single-quote a path for sh -c. Defensive — served_dir comes
    from our own DB rows so this is belt+suspenders."""
    return "'" + path.replace("'", "'\\''") + "'"


async def _safe_run(container_mgr, workspace_id: str, argv: list[str]) -> str:
    """Run a command in the workspace container, swallowing exceptions
    so the caller's path keeps a useful default. ContainerManager's
    private ``_run_command`` is the only API for ad-hoc execs."""
    return await container_mgr._run_command(workspace_id, argv, timeout=5.0)


def _safe_extract_tar(tar_obj: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar archive to ``dest``, rejecting any member whose
    resolved path escapes ``dest``. Mirrors the guard pattern in
    CPython's tarfile.extractall(filter='data') but works on older
    Python without that filter."""
    dest_resolved = dest.resolve()
    for member in tar_obj.getmembers():
        # Symlinks / hardlinks could point outside — refuse them.
        if member.issym() or member.islnk():
            continue
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            log.warning("library_snapshot_skipped_traversal", member=member.name)
            continue
        tar_obj.extract(member, path=dest)


async def write_bundle_zip(
    *,
    source_dir: Path,
    output_path: Path,
) -> int:
    """Zip a publication's content dir for the download endpoint.

    Returns the resulting zip's size in bytes. Output is overwritten if
    it exists. Used by :meth:`LibraryStorage` or directly by the
    download route; kept here next to the snapshot helper since both
    deal with archiving.
    """
    import zipfile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in source_dir.rglob("*"):
            if entry.is_file():
                arcname = entry.relative_to(source_dir).as_posix()
                zf.write(entry, arcname=arcname)
    return output_path.stat().st_size


def find_entry_point(snapshot_dir: Path) -> str:
    """Heuristic: pick the entry HTML file inside a saved bundle.

    Preference order: ``index.html`` at root → any ``*.html`` at root →
    the first ``*.html`` found anywhere → empty string (caller treats
    as ``index.html`` and lets the launcher 404 if absent).
    """
    if (snapshot_dir / "index.html").is_file():
        return "index.html"
    for p in snapshot_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".html":
            return p.name
    for p in snapshot_dir.rglob("*.html"):
        if p.is_file():
            return p.relative_to(snapshot_dir).as_posix()
    return "index.html"


# Re-exported so importers don't have to reach into preview_kind.
__all__ = [
    "PreviewSnapshot",
    "find_entry_point",
    "gather_preview_state",
    "snapshot_container_path",
    "write_bundle_zip",
]
