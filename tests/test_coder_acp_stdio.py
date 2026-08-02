"""acp_stdio smoke entrypoint — proves the smoke runner drives real editor ops."""
from __future__ import annotations

import pytest

pytest.importorskip("acp")

from augmentum.coder.acp_agent import AugmentumACPAgent  # noqa: E402
from augmentum.coder.acp_stdio import _join, _smoke_loop_runner  # noqa: E402


def test_join_prefers_cwd_separator() -> None:
    assert _join("/home/me/proj", "X.md") == "/home/me/proj/X.md"
    assert _join("/home/me/proj/", "X.md") == "/home/me/proj/X.md"
    assert _join("C:\\Users\\me\\proj", "X.md") == "C:\\Users\\me\\proj\\X.md"


class FakeConn:
    """A fake ACP AgentSideConnection recording every client call."""

    def __init__(self) -> None:
        self.updates: list = []
        self.files: dict[str, str] = {}
        self.terminals = 0

    async def session_update(self, sid, update, **kw) -> None:
        self.updates.append(update)

    # -- fs client methods ---------------------------------------------------
    async def write_text_file(self, sid, path, content) -> None:
        self.files[path] = content

    async def read_text_file(self, sid, path):
        from types import SimpleNamespace

        return SimpleNamespace(content=self.files.get(path, ""))

    # -- terminal client methods (list_files uses `ls`) ----------------------
    async def create_terminal(self, sid, *, command, args, env, cwd, output_byte_limit):
        from types import SimpleNamespace

        self.terminals += 1
        return SimpleNamespace(terminal_id="t1")

    async def wait_for_terminal_exit(self, sid, tid):
        from types import SimpleNamespace

        return SimpleNamespace(exit_code=0)

    async def terminal_output(self, sid, tid):
        from types import SimpleNamespace

        # empty listing — _parse_ls_output tolerates it; the leg just reports 0
        return SimpleNamespace(output="")

    async def release_terminal(self, sid, tid) -> None:
        pass


@pytest.mark.asyncio
async def test_smoke_runner_writes_and_reads_back_over_acp() -> None:
    conn = FakeConn()
    agent = AugmentumACPAgent(loop_runner=_smoke_loop_runner)
    agent.on_connect(conn)
    r = await agent.new_session(cwd="/proj")
    resp = await agent.prompt(r.session_id, "smoke please")

    assert resp.stop_reason == "end_turn"
    # The write leg actually persisted through fs/write_text_file:
    assert conn.files["/proj/AUGMENTUM_SMOKE.md"].startswith("# Augmentum smoke test")
    # The terminal leg fired (list_files -> ls):
    assert conn.terminals == 1
    # And the turn produced ACP updates (thought + text narration):
    kinds = [type(u).__name__ for u in conn.updates]
    assert "AgentThoughtChunk" in kinds
    assert "AgentMessageChunk" in kinds
