"""In-process WebSocket carrier for :class:`AugmentumACPAgent`.

The production deployment (see ``acp_app`` module docstring) keeps the coder loop
INSIDE the main Augmentum process so it shares model residency with chat/voice.
Editors speak ACP over stdio, so a thin stdio<->WSS bridge (``acp_stdio --bridge``)
sits on the editor machine and this endpoint receives the tunnelled frames here.

This module is the server half: a PURE BYTE TUNNEL between a Starlette/FastAPI
``WebSocket`` and the ``AgentSideConnection`` the ACP SDK drives. The SDK frames
its own JSON-RPC over an ``asyncio`` ``StreamReader``/``StreamWriter`` pair
(exactly as it does over stdio); we feed inbound WS bytes into the reader and
ship the writer's outbound bytes back over the WS. Because it is byte-transparent,
the framing stays entirely the SDK's concern — we never parse a frame.

Reuses the SDK's own ``_WritePipeProtocol`` for the writer so ``writer.drain()``
flow-control works identically to the stdio path; only the transport sink differs
(a queue drained onto the WS instead of ``sys.stdout``).
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class _WSWriteTransport(asyncio.BaseTransport):
    """A StreamWriter transport whose ``write`` queues bytes for the WS drainer.

    ``write`` is sync (the SDK calls it inline) but WS send is async, so we push
    onto an ordered queue that a single drainer task ships in-order — preserving
    frame order without blocking the loop. Mirrors the SDK's ``_StdoutTransport``
    with a queue sink instead of stdout.
    """

    def __init__(self, out_queue: asyncio.Queue[bytes | None]) -> None:
        self._q = out_queue
        self._closing = False

    def write(self, data: bytes) -> None:  # type: ignore[override]
        if not self._closing:
            self._q.put_nowait(bytes(data))

    def can_write_eof(self) -> bool:  # type: ignore[override]
        return False

    def is_closing(self) -> bool:  # type: ignore[override]
        return self._closing

    def close(self) -> None:  # type: ignore[override]
        if not self._closing:
            self._closing = True
            self._q.put_nowait(None)  # sentinel -> stop the drainer

    def abort(self) -> None:  # type: ignore[override]
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:  # type: ignore[override]
        return default


async def serve_acp_over_websocket(ws: Any, agent: Any) -> None:
    """Drive ``agent`` (an ``acp.Agent``) over an accepted WebSocket ``ws``.

    ``ws`` must expose the Starlette contract: ``await ws.send_bytes(data)`` and
    ``await ws.receive()`` -> ``{"type", "bytes"?, "text"?}``. Returns when the
    client disconnects (reader EOF ends ``conn.listen()``) or the socket errors.
    """
    from acp.agent import AgentSideConnection
    from acp.stdio import _WritePipeProtocol

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    out_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    transport = _WSWriteTransport(out_q)
    writer = asyncio.StreamWriter(transport, _WritePipeProtocol(), None, loop)

    # input_stream=writer (agent writes to client), output_stream=reader
    # (agent reads client). on_connect(conn) fires inside the constructor and
    # wires agent._conn so the loop can call fs/terminal/session_update.
    conn = AgentSideConnection(agent, writer, reader, listening=False)

    async def _drain_outbound() -> None:
        while True:
            data = await out_q.get()
            if data is None:  # transport closed
                return
            await ws.send_bytes(data)

    async def _pump_inbound() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:
                    text = msg.get("text")
                    data = text.encode("utf-8") if text else None
                if data:
                    reader.feed_data(data)
        finally:
            # EOF unblocks conn.listen() so the whole serve returns.
            reader.feed_eof()

    drainer = asyncio.create_task(_drain_outbound())
    pumper = asyncio.create_task(_pump_inbound())
    try:
        await conn.listen()
    finally:
        transport.close()
        for task in (pumper, drainer):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await asyncio.shield(conn.close())
