"""acp_ws — the in-process WebSocket carrier for the ACP agent."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("acp")

from augmentum.coder.acp_agent import AugmentumACPAgent  # noqa: E402
from augmentum.coder.acp_ws import _WSWriteTransport, serve_acp_over_websocket  # noqa: E402


def test_write_transport_queues_and_closes() -> None:
    q: asyncio.Queue = asyncio.Queue()
    t = _WSWriteTransport(q)
    t.write(b"abc")
    assert q.get_nowait() == b"abc"
    t.close()
    assert q.get_nowait() is None  # sentinel stops the drainer
    assert t.is_closing() is True
    t.write(b"after-close")  # ignored once closing
    assert q.empty()


async def _dummy_runner(_session, _text):
    return
    yield  # pragma: no cover - makes this an async generator


class _FakeWS:
    """Feeds queued inbound byte frames, captures outbound, then disconnects.

    The disconnect is deferred until AFTER the agent has sent at least one
    response, so the initialize reply can't be raced by an early EOF.
    """

    def __init__(self, inbound: list[bytes]) -> None:
        self._inbound = list(inbound)
        self.sent: list[bytes] = []
        self._responded = asyncio.Event()

    async def receive(self) -> dict:
        if self._inbound:
            return {"type": "websocket.receive", "bytes": self._inbound.pop(0)}
        await self._responded.wait()
        return {"type": "websocket.disconnect"}

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)
        self._responded.set()


@pytest.mark.asyncio
async def test_carrier_answers_initialize_over_the_websocket() -> None:
    # A real ACP initialize frame (newline-delimited JSON-RPC, as over stdio).
    frame = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
        b'"params":{"protocolVersion":1,"clientCapabilities":{}}}\n'
    )
    ws = _FakeWS([frame])
    agent = AugmentumACPAgent(loop_runner=_dummy_runner, default_user_id="u1")

    await asyncio.wait_for(serve_acp_over_websocket(ws, agent), timeout=5)

    out = b"".join(ws.sent).decode("utf-8")
    assert '"result"' in out
    assert "agentCapabilities" in out or "protocolVersion" in out
