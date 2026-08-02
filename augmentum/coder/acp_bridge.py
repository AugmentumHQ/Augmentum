"""ACP transport for the coder loop's editor path.

:class:`ACPEditorChannel` is the concrete :class:`EditorChannel` that makes
:class:`RemoteEditorExecutor` real: it fulfils the executor's ``request(method,
params)`` calls by driving the Agent Client Protocol (ACP) client-side methods
on an ``AgentSideConnection`` — i.e. asking the connected editor to read/write
files and run terminals on the agent's behalf.

Why a separate module: the ACP SDK (``agent-client-protocol``) is an OPTIONAL
dependency. ``executors.py`` (the ABC + Container/Remote executors) stays free
of it; the ``acp`` import lives here and only the terminal path touches ACP
types, so Augmentum's core never pulls the SDK in. Install it only for the
editor bridge::

    pip install agent-client-protocol   # Apache-2.0 (Zed); confirm LICENSE

Method mapping (channel → ACP client call):
    fs/read_text_file  -> connection.read_text_file(session_id, path)
    fs/write_text_file -> connection.write_text_file(session_id, path, content)
    terminal/run       -> create_terminal -> wait_for_terminal_exit
                          -> terminal_output -> release_terminal

The compound ``terminal/run`` honours the wall-clock ``timeout`` by racing the
exit wait and killing the terminal on expiry — mirroring the container path's
run_command timeout semantics.
"""
from __future__ import annotations

import asyncio
from typing import Any

from augmentum.coder.executors import EditorChannel, EditorError
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Default cap on captured terminal output (bytes). Editors require a limit on
# create_terminal; 1 MB matches the container path's generous shell budget.
_DEFAULT_OUTPUT_BYTE_LIMIT = 1_000_000


class ACPEditorChannel(EditorChannel):
    """EditorChannel over an ACP ``AgentSideConnection`` for one session.

    Parameters
    ----------
    connection:
        The ACP connection the agent uses to call the client (editor). Must
        expose the async client methods ``read_text_file`` /
        ``write_text_file`` / ``create_terminal`` / ``wait_for_terminal_exit``
        / ``terminal_output`` / ``kill_terminal`` / ``release_terminal``
        (duck-typed so tests can pass a fake).
    session_id:
        The ACP session this channel is bound to.
    """

    def __init__(
        self,
        connection: Any,
        session_id: str,
        *,
        output_byte_limit: int = _DEFAULT_OUTPUT_BYTE_LIMIT,
    ) -> None:
        self._conn = connection
        self._session_id = session_id
        self._output_byte_limit = output_byte_limit

    async def request(self, method: str, params: dict) -> dict:
        if method == "fs/read_text_file":
            resp = await self._conn.read_text_file(self._session_id, params["path"])
            return {"content": getattr(resp, "content", "") or ""}
        if method == "fs/write_text_file":
            await self._conn.write_text_file(
                self._session_id, params["path"], params["content"],
            )
            return {}
        if method == "terminal/run":
            return await self._run_terminal(params)
        raise EditorError(f"unsupported editor method: {method!r}")

    async def _run_terminal(self, params: dict) -> dict:
        # ACP EnvVariable is the only SDK type we touch — import lazily so the
        # module stays importable without the SDK for the fs-only paths + tests.
        cmd = list(params.get("command") or [])
        if not cmd:
            raise EditorError("terminal/run requires a non-empty command")
        env_map = params.get("environment") or {}
        env_list = self._build_env(env_map)
        cwd = params.get("cwd")
        timeout = params.get("timeout")

        created = await self._conn.create_terminal(
            self._session_id,
            command=cmd[0],
            args=cmd[1:] or None,
            env=env_list,
            cwd=cwd,
            output_byte_limit=self._output_byte_limit,
        )
        terminal_id = created.terminal_id
        timed_out = False
        exit_code: int | None = None
        try:
            wait = self._conn.wait_for_terminal_exit(self._session_id, terminal_id)
            try:
                exit_info = (
                    await asyncio.wait_for(wait, timeout=timeout)
                    if timeout
                    else await wait
                )
                exit_code = getattr(exit_info, "exit_code", None)
            except TimeoutError:
                timed_out = True
                try:
                    await self._conn.kill_terminal(self._session_id, terminal_id)
                except Exception as exc:  # noqa: BLE001 — best-effort kill
                    log.debug("acp_kill_terminal_failed", error=str(exc))
            out = await self._conn.terminal_output(self._session_id, terminal_id)
            output = getattr(out, "output", "") or ""
            if timed_out:
                output += f"\n\n[Command timed out after {timeout}s]"
            return {"output": output, "exit_code": exit_code}
        finally:
            try:
                await self._conn.release_terminal(self._session_id, terminal_id)
            except Exception as exc:  # noqa: BLE001 — release is best-effort cleanup
                log.debug("acp_release_terminal_failed", error=str(exc))

    def _build_env(self, env_map: dict) -> list | None:
        """Convert a plain env dict into ACP EnvVariable objects (or None).

        Imports ``acp`` lazily; a clear error if the SDK is absent points at
        the optional-install path rather than surfacing an opaque ImportError.
        """
        if not env_map:
            return None
        try:
            from acp.schema import EnvVariable
        except ImportError as exc:  # pragma: no cover - only without the SDK
            raise EditorError(
                "the ACP SDK is required to pass terminal env vars — "
                "pip install agent-client-protocol",
            ) from exc
        return [EnvVariable(name=str(k), value=str(v)) for k, v in env_map.items()]
