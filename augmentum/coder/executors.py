"""WorkspaceExecutor — the seam between the coder loop and *where* it acts.

Historically every coder tool reached straight into :class:`ContainerManager`
and ran ``docker exec`` against a workspace container Augmentum owns. That welds
the loop to a workspace it controls — there is no way to point the SAME tools at
a workspace Augmentum does NOT own, which is exactly what a user's editor is
(the ACP / "loop in the editor" work, see
``docs/superpowers/specs/2026-07-21-augmentum-loop-in-editor-design.md``).

This module introduces the abstraction WITHOUT moving any container-lifecycle
code. ``_run_command`` in ``containers.py`` is deliberately NOT extracted: it is
woven into revive-on-crash, unpause, DB status reconciliation, activity bumps,
and the cancel registry — all container-specific concerns a remote editor never
has. Instead we draw the seam one level up, at the *semantic op set* the tools
actually call, and :class:`ContainerExecutor` simply **forwards** each op to the
existing ``ContainerManager`` method. Behavior is byte-identical; the hot path is
untouched. A future ``RemoteEditorExecutor`` will implement the same interface
over an ACP session (``fs/read_text_file`` / ``fs/write_text_file`` / terminal),
and the tools won't know the difference.

The op set here is exactly what the agentic coder tools call today (discovered,
not designed) — nothing speculative. Container-only concerns (interactive TTY
sessions, port publishing, git plumbing used by UI routes) are intentionally
OUT of this interface: they are not tool-facing and do not port to an editor.

ACP mapping (for the eventual RemoteEditorExecutor):
    read_file        -> fs/read_text_file
    write_file       -> fs/write_text_file
    read_file_bytes  -> fs read (binary)
    list_files       -> workspace listing / search
    run_command      -> terminal/create + terminal/output
    upload_files     -> batched fs writes
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from augmentum.coder.containers import ContainerManager
    from augmentum.coder.models import FileEntry

log = get_logger(__name__)


class WorkspaceCapability(str, Enum):
    """Optional backend features the coder controller must gate on.

    The controller asks ``executor.supports(cap)`` — NEVER ``isinstance(executor,
    …)``. Type-sniffing is the anti-pattern that made the container coupling
    leak past the seam; capability negotiation means a NEW backend (editor, a
    remote SSH host, an external harness like pi/claude) just advertises what it
    can do and the loop adapts with zero controller edits.
    """

    CHECKPOINTS = "checkpoints"   # VCS checkpoint before/after edits
    PORTS = "ports"               # publish dev-server ports to a host
    EXEC_CANCEL = "exec_cancel"   # cancel in-flight execs on teardown


class WorkspaceCapabilityError(RuntimeError):
    """A capability method was called on a backend that doesn't advertise it.

    Never expected in normal flow — the controller gates every optional call on
    ``supports(cap)`` first. Raised (not silently no-op'd) so a MISSING gate is
    loud in tests instead of another silent degradation.
    """

    def __init__(self, cap: WorkspaceCapability) -> None:
        super().__init__(f"workspace backend does not support {cap.value!r}")
        self.capability = cap


@dataclass(frozen=True)
class WorkspaceProfile:
    """Backend-agnostic workspace metadata the coder CONTROLLER reads.

    Everything here used to be dug out of ``ContainerManager._get_workspace()``
    — a ``project_checkouts`` DB row that only a container workspace has. Moving
    it behind the executor lets each backend source it appropriately:
    ``ContainerExecutor`` from the DB row; ``RemoteEditorExecutor`` from the ACP
    session / sane defaults; a future harness from wherever it likes. The
    controller calls ``await executor.profile()`` and never names a backend.
    """

    workspace_root: str = "/workspace"
    planning_mode: str = ""          # "" | "auto" | "plan"
    git_url: str = ""
    buddy_model: str = ""            # per-workspace HVY-escalation model, if any
    tooling_profile: str = ""
    safeguards: dict = field(default_factory=dict)


class WorkspaceExecutor(ABC):
    """Workspace-scoped execution surface for the coder tools.

    An instance is bound to a single workspace/session, so — unlike the
    ``ContainerManager`` methods it fronts — the ops here do NOT take a
    ``workspace_id``. Implementations:

    - :class:`ContainerExecutor` — forwards to a ``ContainerManager`` +
      ``workspace_id`` (today's Docker-owned workspace; behavior unchanged).
    - ``RemoteEditorExecutor`` (future) — dispatches over an ACP editor session.

    Signatures mirror the corresponding ``ContainerManager`` methods exactly
    (minus ``workspace_id``) so forwarding is a pure rename with no adaptation.
    """

    @abstractmethod
    async def run_command(
        self,
        cmd: list[str],
        timeout: float = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Callable[[bytes], Awaitable[None]] | None = None,
        environment: dict[str, str] | None = None,
        login_shell: bool = False,
    ) -> str:
        """Run a non-interactive command and return combined stdout/stderr."""
        raise NotImplementedError

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a UTF-8 text file's contents."""
        raise NotImplementedError

    @abstractmethod
    async def read_file_bytes(self, path: str) -> bytes:
        """Read a file's raw bytes (binary-safe)."""
        raise NotImplementedError

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write UTF-8 text to a file (create/overwrite)."""
        raise NotImplementedError

    @abstractmethod
    async def list_files(self, path: str = "/workspace") -> list[FileEntry]:
        """List directory entries at ``path``."""
        raise NotImplementedError

    @abstractmethod
    async def upload_files(
        self, dest_path: str, files: list[tuple[str, bytes]],
    ) -> None:
        """Extract a set of ``(relative_path, bytes)`` files into ``dest_path``."""
        raise NotImplementedError

    # -- capabilities & metadata (the rest of the contract) -----------------
    # These make the executor the COMPLETE workspace surface, so the controller
    # never has to reach past it to a ContainerManager. Defaults describe the
    # minimal backend (fs/exec only, no git/ports/cancel); richer backends
    # override.

    @property
    def workspace_root(self) -> str:
        """The backend's real filesystem root for this workspace.

        The controller must reference THIS instead of a literal ``/workspace``
        so its commands and paths resolve on whatever surface it's running:
        container → ``/workspace``, editor → the ACP session cwd, a future
        harness → wherever it roots. Sync (not ``profile()``) so it's usable
        inside shell-command f-strings. The loop assuming a fixed ``/workspace``
        FS shape — not just a container manager — is what welded it to one
        surface; this convention is what unwelds it.
        """
        return "/workspace"

    @property
    def capabilities(self) -> frozenset[WorkspaceCapability]:
        """The optional features this backend advertises. Default: none."""
        return frozenset()

    def supports(self, cap: WorkspaceCapability) -> bool:
        """Gate optional calls on this — NOT ``isinstance``."""
        return cap in self.capabilities

    async def profile(self) -> WorkspaceProfile:
        """Backend-agnostic workspace metadata. Default: bare root, no DB fields."""
        return WorkspaceProfile()

    async def checkpoint(self, label: str = "") -> str | None:
        """Create a VCS checkpoint of the workspace (CHECKPOINTS capability).

        Returns a checkpoint id/hash, or None if nothing changed. The controller
        must gate on ``supports(CHECKPOINTS)`` first.
        """
        raise WorkspaceCapabilityError(WorkspaceCapability.CHECKPOINTS)

    async def publish_ports(self, ports: list[int]) -> dict[int, int]:
        """Publish container/dev-server ports to the host (PORTS capability)."""
        raise WorkspaceCapabilityError(WorkspaceCapability.PORTS)

    async def list_ports(self) -> list[dict]:
        """List currently-published ports (PORTS capability)."""
        raise WorkspaceCapabilityError(WorkspaceCapability.PORTS)

    async def cancel_execs(self) -> int:
        """Cancel in-flight execs on teardown (EXEC_CANCEL capability)."""
        raise WorkspaceCapabilityError(WorkspaceCapability.EXEC_CANCEL)


class ContainerExecutor(WorkspaceExecutor):
    """WorkspaceExecutor backed by a Docker workspace container.

    A thin, workspace-bound wrapper over :class:`ContainerManager`. Every op
    forwards 1:1 to the existing manager method with ``workspace_id`` injected —
    no behavior change, no container-lifecycle code moved. This is the
    "today's coder mode, expressed through the interface" implementation.
    """

    def __init__(self, container_manager: ContainerManager, workspace_id: str) -> None:
        self._cm = container_manager
        self._workspace_id = workspace_id

    async def run_command(self, cmd: list[str], *args: object, **kwargs: object) -> str:
        # Transparent passthrough — NOT an explicit re-listing of the kwargs.
        # The wrapper must reproduce the caller's exact call shape (only
        # ``workspace_id`` prepended) so it is behavior-neutral: re-listing the
        # params would forward positional-vs-keyword differently and inject
        # defaults (idle_timeout=None, login_shell=False, ...) the caller never
        # passed, which changes the call the ContainerManager sees and breaks
        # narrow-signature test fakes. ``*args/**kwargs`` forwards verbatim.
        return await self._cm.run_command(self._workspace_id, cmd, *args, **kwargs)

    async def read_file(self, path: str) -> str:
        return await self._cm.file_read(self._workspace_id, path)

    async def read_file_bytes(self, path: str) -> bytes:
        # file_download is the binary-safe reader (get-archive), NOT the
        # non-existent file_read_bytes some call sites still reference.
        return await self._cm.file_download(self._workspace_id, path)

    async def write_file(self, path: str, content: str) -> None:
        await self._cm.file_write(self._workspace_id, path, content)

    async def list_files(self, path: str = "/workspace") -> list[FileEntry]:
        return await self._cm.file_list(self._workspace_id, path)

    async def upload_files(
        self, dest_path: str, files: list[tuple[str, bytes]],
    ) -> None:
        await self._cm.file_upload(self._workspace_id, dest_path, files)

    # -- capabilities & metadata (a real Docker workspace has them all) ------

    @property
    def capabilities(self) -> frozenset[WorkspaceCapability]:
        return frozenset({
            WorkspaceCapability.CHECKPOINTS,
            WorkspaceCapability.PORTS,
            WorkspaceCapability.EXEC_CANCEL,
        })

    async def profile(self) -> WorkspaceProfile:
        # Source the metadata the controller needs from the project_checkouts
        # row (the same object the controller used to read via _get_workspace).
        info = await self._cm._get_workspace(self._workspace_id)
        return WorkspaceProfile(
            workspace_root="/workspace",
            planning_mode=(getattr(info, "planning_mode", "") or ""),
            git_url=(getattr(info, "git_url", "") or ""),
            buddy_model=(getattr(info, "buddy_model", "") or ""),
            tooling_profile=(getattr(info, "tooling_profile", "") or ""),
            safeguards=(getattr(info, "safeguards", None) or {}),
        )

    async def checkpoint(self, label: str = "") -> str | None:
        return await self._cm.git_checkpoint(self._workspace_id)

    async def publish_ports(self, ports: list[int]) -> dict[int, int]:
        return await self._cm.enable_published_ports(self._workspace_id, ports)

    async def list_ports(self) -> list[dict]:
        return await self._cm.list_ports(self._workspace_id)

    async def cancel_execs(self) -> int:
        return await self._cm.cancel_workspace_execs(self._workspace_id)


class EditorError(Exception):
    """An editor-side op failed (rejected permission, missing file, etc.).

    Raised by an :class:`EditorChannel` when the editor returns an error for a
    request. Coder tools already wrap executor calls in try/except and surface
    ``ToolResult(success=False, error=...)``, so this maps cleanly onto the
    existing failure path — a rejected edit reads to the model like any other
    failed write.
    """


class EditorChannel(ABC):
    """Transport-agnostic request/response link to a connected editor.

    This is the round-trip primitive the ACP "loop in the editor" work needs
    and the container path never had: the coder loop, mid-tool-call, dispatches
    an op to the editor and **awaits the editor's reply**. Concrete channels:

    - a test double (``FakeEditorChannel`` in the coder tests), and
    - the ACP transport adapter (Phase 2.3) that implements ``request`` by
      driving the Agent Client Protocol over stdio/WebSocket — assigning a
      JSON-RPC id, sending the frame, and parking on a future until the
      matching response arrives.

    Keeping :class:`RemoteEditorExecutor` on top of this narrow interface (one
    method) means it is fully unit-testable without the ACP SDK or a real
    editor, and swaps transports without change.
    """

    @abstractmethod
    async def request(self, method: str, params: dict) -> dict:
        """Send ``method`` + ``params`` to the editor and await its result.

        Returns the editor's result object. Raises :class:`EditorError` (or a
        transport error) on failure. ``method`` names follow ACP where a 1:1
        mapping exists (``fs/read_text_file``, ``fs/write_text_file``); the
        compound terminal op uses the semantic name ``terminal/run`` which the
        ACP adapter fulfils via the create→wait→output handshake.
        """
        raise NotImplementedError


class RemoteEditorExecutor(WorkspaceExecutor):
    """WorkspaceExecutor backed by a connected editor over an EditorChannel.

    The other half of the Phase-1 seam: the SAME coder tools that run against a
    Docker workspace via :class:`ContainerExecutor` run against a user's editor
    when handed one of these instead. Each portable op becomes an outbound
    editor request whose reply the tool awaits inline — so from the loop's point
    of view ``tool.execute()` simply takes a network round-trip longer, and no
    loop surgery is needed.

    Not yet supported (raise ``NotImplementedError`` with a clear message):
    binary ``read_file_bytes`` / ``upload_files`` — stable ACP has no binary fs
    op, so the tools that need them (the binary analyzer, patch upload) degrade
    with an honest error rather than silently misbehaving. A base64 extension is
    the natural follow-up.
    """

    def __init__(self, channel: EditorChannel, *, workspace_root: str = "/workspace") -> None:
        self._channel = channel
        self._root = workspace_root

    async def run_command(
        self,
        cmd: list[str],
        timeout: float = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Callable[[bytes], Awaitable[None]] | None = None,
        environment: dict[str, str] | None = None,
        login_shell: bool = False,
    ) -> str:
        # idle_timeout / progress_path / login_shell are container-terminal
        # concepts with no stable ACP analogue yet; the editor terminal owns its
        # own liveness. We forward the wall-clock timeout + env and return the
        # collected output. Live streaming (on_chunk per chunk) is a later
        # enhancement — for now we forward the full output once so callers that
        # only echo it still work.
        resp = await self._channel.request(
            "terminal/run",
            {
                "command": list(cmd),
                "cwd": self._root,
                "timeout": timeout,
                "environment": dict(environment or {}),
            },
        )
        output = str(resp.get("output", ""))
        if on_chunk and output:
            try:
                await on_chunk(output.encode("utf-8"))
            except Exception as exc:  # noqa: BLE001 — broken sink must not kill the run
                log.debug("remote_run_command_on_chunk_failed", error=str(exc))
        return output

    async def read_file(self, path: str) -> str:
        resp = await self._channel.request("fs/read_text_file", {"path": path})
        return str(resp.get("content", ""))

    async def read_file_bytes(self, path: str) -> bytes:
        raise NotImplementedError(
            "binary read (read_file_bytes) is not supported over the editor "
            "channel yet — stable ACP has no binary fs op",
        )

    async def write_file(self, path: str, content: str) -> None:
        await self._channel.request(
            "fs/write_text_file", {"path": path, "content": content},
        )

    async def list_files(self, path: str = "/workspace") -> list[FileEntry]:
        # ACP has no stable directory-list method, so we list via the terminal
        # and reuse ContainerManager's battle-tested ls parser — identical
        # FileEntry output to the container path.
        from augmentum.coder.containers import _parse_ls_output

        out = await self.run_command(["ls", "-la", "--time-style=+%s", path])
        return _parse_ls_output(out, path)

    async def upload_files(
        self, dest_path: str, files: list[tuple[str, bytes]],
    ) -> None:
        raise NotImplementedError(
            "binary upload (upload_files) is not supported over the editor "
            "channel yet — stable ACP has no binary fs op",
        )

    # -- capabilities & metadata --------------------------------------------
    # An editor workspace advertises NO container-only capabilities: the editor
    # owns its own VCS (no git_checkpoint), publishes no ports, and the ACP
    # terminal manages its own liveness (no exec-cancel registry). The
    # controller gates on supports(...) and cleanly skips them — no isinstance,
    # no silent KeyError from reaching a container that isn't there.

    @property
    def workspace_root(self) -> str:
        # The editor's real root is the ACP session cwd, NOT /workspace.
        return self._root

    @property
    def capabilities(self) -> frozenset[WorkspaceCapability]:
        return frozenset()

    async def profile(self) -> WorkspaceProfile:
        # No project_checkouts row for an editor session; derive what we can
        # from the ACP-provided root and use sane defaults. planning_mode="auto"
        # matches the editor-owns-approval posture (Zed gates mutations itself).
        return WorkspaceProfile(
            workspace_root=self._root,
            planning_mode="auto",
        )
