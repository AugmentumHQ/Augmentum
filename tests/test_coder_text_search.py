"""Tests for the workspace text-search backend (Files-panel search pane).

Covers ``augmentum/coder/text_search.py`` (rg --json parsing, grep
fallback, span conversion, windowing, truncation honesty) and the two
route changes in ``coder_routes.py``: the new
``/api/coder/workspaces/{id}/search-text`` endpoint and the rename
route's destination-exists guard (409 instead of silent ``mv`` clobber).
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.coder.text_search import search_workspace_text
from augmentum.proxy.coder_routes import router
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Module-level: rg --json parsing
# ---------------------------------------------------------------------------


class _FakeCM:
    """Container-manager stand-in that records the exec and returns
    canned output — optionally different output per call (fallback)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.commands: list[list[str]] = []

    async def run_command(self, workspace_id, cmd, timeout=30.0, **_kw):
        self.commands.append(cmd)
        return self.outputs.pop(0) if self.outputs else ""


def _rg_match(path, line_no, text, start, end):
    return json.dumps({
        "type": "match",
        "data": {
            "path": {"text": path},
            "lines": {"text": text},
            "line_number": line_no,
            "absolute_offset": 0,
            "submatches": [{"match": {"text": text[start:end]},
                            "start": start, "end": end}],
        },
    })


def test_rg_output_parsed_to_structured_matches():
    out = "\n".join([
        json.dumps({"type": "begin", "data": {"path": {"text": "./app.py"}}}),
        _rg_match("./app.py", 12, "def handle_user(x):\n", 11, 15),
        _rg_match("./lib/util.py", 3, "user = load()\n", 0, 4),
        json.dumps({"type": "summary", "data": {}}),
    ])
    cm = _FakeCM([out])
    res = _run(search_workspace_text(cm, "ws1", "user"))
    assert res["engine"] == "rg"
    assert res["total_returned"] == 2
    assert res["files_with_matches"] == 2
    assert res["truncated"] is False
    first = res["matches"][0]
    assert first["path"] == "/workspace/app.py"
    assert first["line"] == 12
    assert first["text"] == "def handle_user(x):"
    assert first["spans"] == [[11, 15]]


def test_rg_byte_offsets_converted_to_char_offsets():
    # "héllo user" — é is 2 bytes, so byte offsets 7..11 are chars 6..10.
    out = _rg_match("./a.txt", 1, "héllo user\n", 7, 11)
    cm = _FakeCM([out])
    res = _run(search_workspace_text(cm, "ws1", "user"))
    span = res["matches"][0]["spans"][0]
    text = res["matches"][0]["text"]
    assert text[span[0]:span[1]] == "user"


def test_rg_default_flags_literal_and_case_insensitive():
    cm = _FakeCM([""])
    _run(search_workspace_text(cm, "ws1", "a.b(c)"))
    script = cm.commands[0][2]
    assert " -F" in script
    assert " -i" in script
    assert "node_modules" in script
    assert "'!*.min.js'" in script


def test_rg_flags_flip_with_options_and_glob_quoted():
    cm = _FakeCM([""])
    _run(search_workspace_text(
        cm, "ws1", "def .*handler", regex=True, case_sensitive=True,
        glob="*.py",
    ))
    script = cm.commands[0][2]
    assert " -F" not in script
    assert " -i " not in script and not script.endswith(" -i")
    assert "'*.py'" in script


def test_long_line_windowed_with_clipped_flag():
    long_text = ("x" * 900) + "needle" + ("y" * 900) + "\n"
    out = _rg_match("./big.json", 1, long_text, 900, 906)
    cm = _FakeCM([out])
    res = _run(search_workspace_text(cm, "ws1", "needle"))
    m = res["matches"][0]
    assert m["clipped"] is True
    assert len(m["text"]) <= 400
    s, e = m["spans"][0]
    assert m["text"][s:e] == "needle"


def test_leading_indent_stripped_and_spans_shifted():
    # 8 spaces of indentation then "return user"; match "user" at 15..19.
    out = _rg_match("./svc.py", 42, "        return user\n", 15, 19)
    cm = _FakeCM([out])
    res = _run(search_workspace_text(cm, "ws1", "user"))
    m = res["matches"][0]
    # Indent gone → line starts at the code, not blank space.
    assert m["text"] == "return user"
    assert not m["text"].startswith(" ")
    s, e = m["spans"][0]
    assert m["text"][s:e] == "user"


def test_strip_never_cuts_into_a_match():
    # A match that begins inside the indentation (col 4) must keep its
    # highlight — we strip only up to the earliest match start.
    out = _rg_match("./svc.py", 1, "    x = 1\n", 4, 5)
    cm = _FakeCM([out])
    res = _run(search_workspace_text(cm, "ws1", "x"))
    m = res["matches"][0]
    assert m["text"] == "x = 1"
    s, e = m["spans"][0]
    assert m["text"][s:e] == "x"


def test_max_results_cap_sets_truncated():
    lines = [_rg_match("./f.py", i + 1, "hit here\n", 0, 3) for i in range(30)]
    cm = _FakeCM(["\n".join(lines)])
    res = _run(search_workspace_text(cm, "ws1", "hit", max_results=10))
    assert res["total_returned"] == 10
    assert res["truncated"] is True


def test_regex_error_surfaced_not_swallowed():
    cm = _FakeCM(["regex parse error:\n    (unclosed\nerror: unclosed group"])
    res = _run(search_workspace_text(cm, "ws1", "(unclosed", regex=True))
    assert res["matches"] == []
    assert "error" in res
    assert "regex parse error" in res["error"]


def test_non_utf8_line_skipped():
    ev = json.dumps({
        "type": "match",
        "data": {
            "path": {"text": "./bin.dat"},
            "lines": {"bytes": "3q2+7w=="},
            "line_number": 1,
            "submatches": [],
        },
    })
    cm = _FakeCM([ev])
    res = _run(search_workspace_text(cm, "ws1", "x"))
    assert res["matches"] == []


# ---------------------------------------------------------------------------
# Module-level: grep fallback
# ---------------------------------------------------------------------------


def test_grep_fallback_when_rg_missing():
    grep_out = "./src/app.py:7:user = get_user()\n./src/app.py:9:print(USER)"
    cm = _FakeCM(["__AUG_NO_RG__", grep_out])
    res = _run(search_workspace_text(cm, "ws1", "user"))
    assert res["engine"] == "grep"
    assert res["total_returned"] == 2
    m = res["matches"][0]
    assert m["path"] == "/workspace/src/app.py"
    assert m["line"] == 7
    # Literal span computed here (grep gives no offsets); second line
    # matched case-insensitively.
    s, e = m["spans"][0]
    assert m["text"][s:e] == "user"
    s2, e2 = res["matches"][1]["spans"][0]
    assert res["matches"][1]["text"][s2:e2] == "USER"
    # The fallback exec used grep, not rg.
    assert "grep " in cm.commands[1][2]


def test_empty_query_rejected():
    cm = _FakeCM([""])
    res = _run(search_workspace_text(cm, "ws1", ""))
    assert res["matches"] == []
    assert "error" in res
    assert cm.commands == []


# ---------------------------------------------------------------------------
# Route-level
# ---------------------------------------------------------------------------


async def _seed_user(conn, user_id: str, username: str) -> None:
    await conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, username, username.title(), "pw", "user"),
    )
    await conn.commit()


async def _seed_workspace(conn, workspace_id: str, *, user_id: str | None) -> None:
    await conn.execute(
        "INSERT INTO project_checkouts (id, name, status, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (workspace_id, "test", "running", time.time(), user_id),
    )
    await conn.commit()


def _make_app(backend: SQLiteBackend, manager, *, user_id: str | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.container_manager = manager
    app.state.state_manager = StateManager(backend)

    if user_id is not None:
        @app.middleware("http")
        async def _inject_user(request, call_next):
            class _U:
                def __init__(self, uid):
                    self.id = uid
            request.scope["user"] = _U(user_id)
            return await call_next(request)

    return app


@pytest.fixture
def backend(tmp_path):
    be = SQLiteBackend(str(tmp_path / "text-search-routes.db"))
    _run(be.connect())
    _run(_seed_user(be.conn, "alice", "alice"))
    _run(_seed_user(be.conn, "bob", "bob"))
    _run(_seed_workspace(be.conn, "ws1", user_id="alice"))
    try:
        yield be
    finally:
        _run(be.close())


def test_search_text_route_requires_query(backend):
    mgr = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces/ws1/search-text")
    assert resp.status_code == 400


def test_search_text_route_happy_path(backend):
    mgr = AsyncMock()
    mgr.run_command = AsyncMock(
        return_value=_rg_match("./app.py", 5, "user = 1\n", 0, 4),
    )
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.get(
            "/api/coder/workspaces/ws1/search-text",
            params={"q": "user"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "rg"
    assert data["matches"][0]["path"] == "/workspace/app.py"
    assert data["matches"][0]["line"] == 5


def test_search_text_route_cross_tenant_404(backend):
    mgr = AsyncMock()
    app = _make_app(backend, mgr, user_id="bob")
    with TestClient(app) as client:
        resp = client.get(
            "/api/coder/workspaces/ws1/search-text",
            params={"q": "user"},
        )
    assert resp.status_code == 404


def test_rename_blocks_existing_destination(backend):
    mgr = AsyncMock()
    mgr.run_command = AsyncMock(return_value="EXISTS\n")
    mgr.file_rename = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/files/ws1/rename",
            json={"old_path": "/workspace/a.py", "new_path": "/workspace/b.py"},
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "destination_exists"
    mgr.file_rename.assert_not_called()


def test_rename_overwrite_true_skips_probe(backend):
    mgr = AsyncMock()
    mgr.run_command = AsyncMock(return_value="EXISTS\n")
    mgr.file_rename = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/files/ws1/rename",
            json={
                "old_path": "/workspace/a.py",
                "new_path": "/workspace/b.py",
                "overwrite": True,
            },
        )
    assert resp.status_code == 200
    mgr.run_command.assert_not_called()
    mgr.file_rename.assert_called_once_with(
        "ws1", "/workspace/a.py", "/workspace/b.py",
    )


def test_rename_clean_destination_proceeds(backend):
    mgr = AsyncMock()
    mgr.run_command = AsyncMock(return_value="")
    mgr.file_rename = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/files/ws1/rename",
            json={"old_path": "/workspace/a.py", "new_path": "/workspace/dir/a.py"},
        )
    assert resp.status_code == 200
    assert resp.json()["renamed"] is True
    mgr.file_rename.assert_called_once()
