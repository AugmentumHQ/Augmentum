"""Unit tests for the agent-browser sidecar client (wave 1).

Covers the pure/mocked logic: session naming (tenant boundary), the
result-envelope ladder semantics, vantage URL rewriting, and CLI JSON
parsing tolerance. Live sidecar behavior is wave-2 acceptance.
"""

from __future__ import annotations

import json

import pytest

from augmentum.coder import browser_sidecar as bs

# ---------------------------------------------------------------------------
# Session naming — the multi-tenant boundary
# ---------------------------------------------------------------------------

def test_session_name_is_derived_and_sanitized():
    assert bs.session_for_workspace("ws_abc-123") == "ws-ws_abc-123"
    # Shell/CLI metacharacters never survive into a session name.
    hostile = 'x"; rm -rf / #--session evil'
    name = bs.session_for_workspace(hostile)
    assert name.startswith("ws-")
    assert all(c.isalnum() or c in "-_" for c in name[3:])


def test_session_name_length_capped():
    assert len(bs.session_for_workspace("a" * 500)) <= 83


# ---------------------------------------------------------------------------
# Envelope / ladder semantics
# ---------------------------------------------------------------------------

def test_envelope_marks_unavailable_sidecar_for_ladder_fallthrough():
    env = bs._envelope({"ok": False, "sidecar": False, "error": "not running"})
    assert env["engine"] == "sidecar_unavailable"
    env2 = bs._envelope({"ok": True, "sidecar": True})
    assert env2["engine"] == "sidecar"
    # Deprecated alias stays truthy — a real browser ran.
    assert env2["playwright"] is True


def test_envelope_carries_error_only_on_failure():
    ok = bs._envelope({"ok": True, "sidecar": True, "error": "stale"})
    assert ok["error"] == ""
    bad = bs._envelope({"ok": False, "sidecar": True, "error": "boom"})
    assert bad["error"] == "boom"


# ---------------------------------------------------------------------------
# run_cli output parsing
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, payloads: list[bytes]):
        self._payloads = list(payloads)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def read_out(self):
        if not self._payloads:
            return None

        class _Msg:
            def __init__(self, data):
                self.data = data

        return _Msg(self._payloads.pop(0))


class _FakeExec:
    def __init__(self, payloads):
        self._payloads = payloads

    def start(self, detach=False):
        return _FakeStream(self._payloads)


class _FakeContainer:
    def __init__(self, output: bytes):
        self._output = output

    async def exec(self, **kwargs):
        self.last_cmd = kwargs["cmd"]
        return _FakeExec([self._output])


class _FakeDocker:
    def __init__(self, container):
        self._container = container


@pytest.mark.asyncio
async def test_run_cli_parses_last_json_line(monkeypatch):
    # Real 0.32.1 wire shape (verified live 2026-07-16): success/data/error,
    # payload nested under data with a noisy lifecycle block we drop.
    wire = {
        "success": True,
        "data": {"lifecycle": {"launched": True}, "title": "Hi"},
        "error": None,
    }
    container = _FakeContainer(
        b"daemon spawned\n" + json.dumps(wire).encode() + b"\n"
    )

    async def _find(_docker):
        return container

    monkeypatch.setattr(bs, "find_sidecar", _find)
    res = await bs.run_cli(object(), ["get", "title"], session="ws-t")
    assert res["ok"] is True
    assert res["sidecar"] is True
    assert res["title"] == "Hi"
    assert "lifecycle" not in res
    assert res["error"] == ""
    # Session flag always precedes the subcommand.
    assert container.last_cmd[:3] == ["agent-browser", "--session", "ws-t"]
    assert "--json" in container.last_cmd


@pytest.mark.asyncio
async def test_run_cli_no_sidecar_signals_unavailable(monkeypatch):
    async def _find(_docker):
        return None

    monkeypatch.setattr(bs, "find_sidecar", _find)
    res = await bs.run_cli(object(), ["open", "http://x"], session="ws-t")
    assert res["ok"] is False
    assert res["sidecar"] is False


@pytest.mark.asyncio
async def test_run_cli_non_json_output_is_surfaced(monkeypatch):
    container = _FakeContainer(b"garbage output, no json")

    async def _find(_docker):
        return container

    monkeypatch.setattr(bs, "find_sidecar", _find)
    res = await bs.run_cli(object(), ["snapshot"], session="ws-t")
    assert res["ok"] is False
    assert "no JSON" in res["error"]


# ---------------------------------------------------------------------------
# Vantage rewrite
# ---------------------------------------------------------------------------

class _FakeCM:
    """Just enough ContainerManager for reachable_url."""

    def __init__(self, ip="172.28.0.9"):
        self._ip = ip
        self._docker = self

    async def _get_workspace(self, workspace_id):
        class _Info:
            container_id = "cid"

        return _Info()

    @property
    def containers(self):
        return self

    async def get(self, cid):
        return self

    async def show(self):
        return {
            "NetworkSettings": {
                "Networks": {
                    "augmentum_workspace_net": {"IPAddress": self._ip},
                }
            }
        }


@pytest.mark.asyncio
async def test_reachable_url_rewrites_localhost():
    cm = _FakeCM()
    out = await bs.reachable_url(cm, "ws1", "http://localhost:5173/app?x=1")
    assert out == "http://172.28.0.9:5173/app?x=1"


@pytest.mark.asyncio
async def test_reachable_url_rewrites_preview_path():
    cm = _FakeCM()
    out = await bs.reachable_url(cm, "ws1", "/api/coder/preview/ws1/3000/")
    assert out == "http://172.28.0.9:3000/"


@pytest.mark.asyncio
async def test_reachable_url_passes_external_through():
    cm = _FakeCM()
    url = "https://example.com/docs"
    assert await bs.reachable_url(cm, "ws1", url) == url


# ---------------------------------------------------------------------------
# Full-integration additions (wave 2)
# ---------------------------------------------------------------------------

def _cli_recorder(responses):
    """monkeypatch-able run_cli that records arg lists and pops canned
    responses (dict per subcommand-head, default ok)."""
    calls = []

    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        calls.append(list(args))
        return dict(responses.get(args[0], {"ok": True, "sidecar": True}))

    return calls, _run


@pytest.mark.asyncio
async def test_command_allowlist_blocks_unlisted(monkeypatch):
    cm = _FakeCM()
    res = await bs.command(cm, "ws1", ["auth", "list"])
    assert res["ok"] is False
    assert "not allowed" in res["error"]


@pytest.mark.asyncio
async def test_command_dispatches_allowlisted(monkeypatch):
    calls, run = _cli_recorder({"get": {"ok": True, "sidecar": True, "title": "T"}})
    monkeypatch.setattr(bs, "run_cli", run)
    cm = _FakeCM()
    res = await bs.command(cm, "ws1", ["get", "title"])
    assert res["ok"] is True
    assert res["title"] == "T"
    assert calls == [["get", "title"]]


@pytest.mark.asyncio
async def test_wait_states_map_to_fn_polling(monkeypatch):
    calls, run = _cli_recorder({})
    monkeypatch.setattr(bs, "run_cli", run)
    cm = _FakeCM()
    await bs.wait(cm, "ws1", url="", selector="#x", state="detached")
    wait_call = next(c for c in calls if c[0] == "wait")
    assert wait_call[1] == "--fn"
    assert "!document.querySelector" in wait_call[2]


@pytest.mark.asyncio
async def test_fill_form_never_submits_half_filled(monkeypatch):
    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "fill" and args[1] == "#bad":
            return {"ok": False, "sidecar": True, "error": "no such element"}
        if args[0] == "select":
            return {"ok": False, "sidecar": True, "error": "not a select"}
        return {"ok": True, "sidecar": True}

    monkeypatch.setattr(bs, "run_cli", _run)
    cm = _FakeCM()
    res = await bs.fill_form(
        cm, "ws1", url="", fields={"#good": "a", "#bad": "b"},
        submit="button[type=submit]",
    )
    assert res["ok"] is False
    assert res["submitted"] is False
    assert "half-filled" in res["submit_error"]


@pytest.mark.asyncio
async def test_evaluate_selector_scoped_wraps_queryselector(monkeypatch):
    captured = {}

    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "eval":
            captured["code"] = args[1]
            return {"ok": True, "sidecar": True,
                    "result": {"__aug_ok": True, "value": "hi", "type": "string"}}
        return {"ok": True, "sidecar": True}

    monkeypatch.setattr(bs, "run_cli", _run)
    cm = _FakeCM()
    res = await bs.evaluate(cm, "ws1", url="", expression="el.textContent",
                            selector="h1", args={"k": 1})
    assert res["ok"] is True
    assert res["result_json"] == '"hi"'
    assert "document.querySelector(\"h1\")" in captured["code"]
    assert '{"k": 1}' in captured["code"]


@pytest.mark.asyncio
async def test_evaluate_js_error_surfaces_structured(monkeypatch):
    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        if args[0] == "eval":
            return {"ok": True, "sidecar": True,
                    "result": {"__aug_ok": False,
                               "error": {"message": "boom", "name": "TypeError"}}}
        return {"ok": True, "sidecar": True}

    monkeypatch.setattr(bs, "run_cli", _run)
    cm = _FakeCM()
    res = await bs.evaluate(cm, "ws1", url="", expression="x.y")
    assert res["ok"] is False
    assert res["js_error"] is True
    assert res["error"] == "boom"


@pytest.mark.asyncio
async def test_ensure_page_skips_open_when_already_there(monkeypatch):
    """Regression (bench ctr_724dd1e6, 2026-07-17): unconditional open per
    call reset page state and killed snapshot @refs."""
    calls = []

    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        calls.append(list(args))
        if args[0] == "get" and args[1] == "url":
            return {"ok": True, "sidecar": True, "url": "http://172.28.0.9:5173/"}
        return {"ok": True, "sidecar": True}

    monkeypatch.setattr(bs, "run_cli", _run)
    cm = _FakeCM()
    await bs.action(cm, "ws1", url="http://localhost:5173/", action="click", selector="#inc")
    assert ["open", "http://172.28.0.9:5173/"] not in calls
    assert ["click", "#inc"] in calls


@pytest.mark.asyncio
async def test_ensure_page_opens_when_elsewhere(monkeypatch):
    calls = []

    async def _run(_docker, args, *, session, timeout=30.0, json_output=True):
        calls.append(list(args))
        if args[0] == "get" and args[1] == "url":
            return {"ok": True, "sidecar": True, "url": "about:blank"}
        return {"ok": True, "sidecar": True}

    monkeypatch.setattr(bs, "run_cli", _run)
    cm = _FakeCM()
    await bs.action(cm, "ws1", url="http://localhost:5173/", action="open")
    assert ["open", "http://172.28.0.9:5173/"] in calls
