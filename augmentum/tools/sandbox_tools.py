"""ATP ``sandbox_shell`` — an E2B-class sandbox from any harness.

Reuses the coder workspace substrate wholesale: one lazily-created,
per-user "ATP sandbox" workspace (real Docker container, resource
limits, persistent /workspace) with shell access. The workspace id is
remembered in the settings store (``atp.sandbox.<user_id>``) so repeat
calls land in the same container; if it was deleted out-of-band a fresh
one is created transparently.

No new execution machinery: ``ContainerManager.create_workspace`` /
``start`` / ``run_command`` do all the work.
"""

from __future__ import annotations

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_OUTPUT_CAP = 30_000


class SandboxShellTool(Tool):
    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "sandbox_shell"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in your persistent per-user sandbox — a "
            "real Docker container with /workspace that survives across "
            "calls (install deps, clone repos, build, test). Isolated "
            "from your local machine and from other users. First call "
            "may take ~10s while the sandbox is created."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return 180.0

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command (bash -lc)"},
                "timeout_s": {"type": "integer", "default": 60,
                              "description": "Command timeout, max 150"},
            },
            "required": ["command"],
        }

    def health_check(self) -> bool:
        return getattr(self._app_state, "container_manager", None) is not None

    async def _ensure_sandbox(self, user_id: str) -> str:
        """Existing sandbox workspace id, or create one. Raises on failure."""
        cm = self._app_state.container_manager
        store = getattr(self._app_state, "settings_store", None)
        key = f"atp.sandbox.{user_id}"
        ws_id = ""
        if store is not None:
            try:
                ws_id = str(await store.get(key) or "")
            except Exception:
                ws_id = ""
        if ws_id:
            try:
                info = await cm._get_workspace(ws_id)
                if getattr(info, "status", "") != "running":
                    await cm.start(ws_id)
                return ws_id
            except Exception:
                log.warning("atp_sandbox_stale_recreating",
                            user_id=user_id, workspace_id=ws_id)
        info = await cm.create_workspace(
            "atp-sandbox", tooling_profile="standard", user_id=user_id,
        )
        ws_id = info.id
        if store is not None:
            try:
                await store.set(key, ws_id)
            except Exception:
                log.warning("atp_sandbox_id_persist_failed", user_id=user_id,
                            exc_info=True)
        return ws_id

    async def execute(self, **kwargs) -> ToolResult:
        cm = getattr(self._app_state, "container_manager", None)
        if cm is None:
            return ToolResult(success=False, error="container manager unavailable")
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        command = str(kwargs.get("command") or "").strip()
        if not command:
            return ToolResult(success=False, validation_error=True,
                              error="'command' is required")
        timeout_s = max(1, min(150, int(kwargs.get("timeout_s") or 60)))
        try:
            ws_id = await self._ensure_sandbox(user_id)
        except Exception as exc:
            return ToolResult(success=False, error=f"sandbox unavailable: {exc}")
        try:
            out = await cm.run_command(
                ws_id, ["bash", "-lc", command], timeout=float(timeout_s),
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"command failed: {exc}",
                              metadata={"workspace_id": ws_id})
        out = out or ""
        truncated = len(out) > _OUTPUT_CAP
        return ToolResult(
            success=True,
            output=out[:_OUTPUT_CAP] + ("\n...(truncated)" if truncated else ""),
            metadata={"workspace_id": ws_id, "truncated": truncated},
        )
