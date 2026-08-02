"""Agent-tool wrapper tests for the wiring + call-graph LLM surface.

Verifies the 7 new tools (4 wiring + 3 call-graph) execute and return
the expected JSON shape so a calling LLM can parse the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.bug_finder.agent_tools import (
    DETERMINISTIC_TOOL_NAMES,
    CalleesOfTool,
    DecoratorsOnTool,
    GetConstantTool,
    IsReachableTool,
    MiddlewareChainTool,
    TraceOriginTool,
    WhoCallsTool,
    _CallGraphHolder,
    build_deterministic_tools,
)


def _mkproject(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Allow-list / registry shape
# ---------------------------------------------------------------------------


def test_new_tools_are_in_deterministic_set() -> None:
    expected = {
        "middleware_chain", "decorators_on", "get_constant",
        "trace_origin", "who_calls", "callees_of", "is_reachable_from",
    }
    assert expected.issubset(DETERMINISTIC_TOOL_NAMES)


def test_agents_tools_allowlist_includes_new_tools() -> None:
    """The role allow-lists in augmentum.agents.tools must include the
    new tool names so the detector / investigator are permitted to call
    them."""
    from augmentum.agents.tools import (
        DETECTOR_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
    )
    from augmentum.agents.tools import (
        DETERMINISTIC_TOOL_NAMES as canonical,
    )
    expected = {
        "middleware_chain", "decorators_on", "get_constant",
        "trace_origin", "who_calls", "callees_of", "is_reachable_from",
    }
    assert expected.issubset(canonical)
    assert expected.issubset(DETECTOR_TOOL_NAMES)
    assert expected.issubset(INVESTIGATOR_TOOL_NAMES)


def test_build_deterministic_tools_includes_new_tools(tmp_path: Path) -> None:
    tools = build_deterministic_tools(tmp_path)
    names = {t.name for t in tools}
    assert "middleware_chain" in names
    assert "trace_origin" in names
    assert "who_calls" in names
    assert "decorators_on" in names
    assert "get_constant" in names
    assert "callees_of" in names
    assert "is_reachable_from" in names


# ---------------------------------------------------------------------------
# MiddlewareChainTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_chain_tool_emits_flow_order(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "server.py": (
            "def create():\n"
            "    app = FastAPI()\n"
            "    app.add_middleware(Auth)\n"
            "    app.add_middleware(Fabric)\n"
            "    return app\n"
        ),
    })
    tool = MiddlewareChainTool(root=root)
    res = await tool.execute()
    assert res.success
    data = json.loads(res.output)
    names = [e["name"] for e in data["registration_order"]]
    assert names == ["Auth", "Fabric"]
    # Last-registered runs first
    assert data["request_flow_order"] == ["Fabric", "Auth"]


@pytest.mark.asyncio
async def test_middleware_chain_tool_empty_project(tmp_path: Path) -> None:
    tool = MiddlewareChainTool(root=tmp_path)
    res = await tool.execute()
    assert res.success
    data = json.loads(res.output)
    assert data["registration_order"] == []
    assert data["request_flow_order"] == []


# ---------------------------------------------------------------------------
# DecoratorsOnTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decorators_on_tool_returns_chain(tmp_path: Path) -> None:
    src = (
        "@router.post('/login')\n"   # 1
        "@require_auth\n"             # 2
        "def login():\n"              # 3
        "    return 1\n"              # 4
    )
    root = _mkproject(tmp_path, {"r.py": src})
    tool = DecoratorsOnTool(root=root)
    res = await tool.execute(file="r.py", line=3)
    assert res.success
    data = json.loads(res.output)
    assert [d["name"] for d in data] == ["router.post", "require_auth"]
    assert data[0]["args"] == ["'/login'"]


@pytest.mark.asyncio
async def test_decorators_on_tool_validates_required(tmp_path: Path) -> None:
    tool = DecoratorsOnTool(root=tmp_path)
    res = await tool.execute()
    assert not res.success
    assert res.validation_error


# ---------------------------------------------------------------------------
# GetConstantTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_constant_tool_resolves_literal(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"c.py": "DEBUG = True\n"})
    tool = GetConstantTool(root=root)
    res = await tool.execute(name="DEBUG")
    assert res.success
    data = json.loads(res.output)
    assert data["found"] is True
    assert data["value_repr"] == "True"
    assert data["confident"] is True


@pytest.mark.asyncio
async def test_get_constant_tool_runtime_value(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "c.py": "import os\nKEY = os.environ['K']\n",
    })
    tool = GetConstantTool(root=root)
    res = await tool.execute(name="KEY")
    data = json.loads(res.output)
    assert data["found"] is True
    assert data["confident"] is False


@pytest.mark.asyncio
async def test_get_constant_tool_missing(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"c.py": "X = 1\n"})
    tool = GetConstantTool(root=root)
    res = await tool.execute(name="NOT_THERE")
    data = json.loads(res.output)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# TraceOriginTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_origin_tool_parameter(tmp_path: Path) -> None:
    src = (
        "def f(token: str):\n"
        "    return token\n"
    )
    root = _mkproject(tmp_path, {"f.py": src})
    tool = TraceOriginTool(root=root)
    res = await tool.execute(file="f.py", line=2, var="token")
    assert res.success
    data = json.loads(res.output)
    assert data["found"] is True
    assert data["origin"]["kind"] == "parameter"


@pytest.mark.asyncio
async def test_trace_origin_tool_unbound_returns_explanation(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"u.py": "def f():\n    return undefined\n"})
    tool = TraceOriginTool(root=root)
    res = await tool.execute(file="u.py", line=2, var="undefined")
    assert res.success
    data = json.loads(res.output)
    assert data["found"] is False
    assert "framework-supplied" in data["note"] or "dynamically" in data["note"]


# ---------------------------------------------------------------------------
# Call-graph tools (WhoCallsTool / CalleesOfTool / IsReachableTool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_who_calls_tool(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def handler():\n"
            "    return danger()\n"
            "\n"
            "def danger():\n"
            "    return 1\n"
        ),
    })
    holder = _CallGraphHolder(root.resolve())
    tool = WhoCallsTool(root=root, holder=holder)
    res = await tool.execute(target="danger")
    assert res.success
    data = json.loads(res.output)
    callers = [c["caller"] for c in data]
    assert "app.handler" in callers


@pytest.mark.asyncio
async def test_callees_of_tool(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def handler():\n"
            "    a()\n"
            "    b()\n"
        ),
    })
    holder = _CallGraphHolder(root.resolve())
    tool = CalleesOfTool(root=root, holder=holder)
    res = await tool.execute(qualified_caller="app.handler")
    assert res.success
    data = json.loads(res.output)
    assert set(data) >= {"a", "b"}


@pytest.mark.asyncio
async def test_is_reachable_tool_true(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def route():\n"
            "    helper()\n"
            "\n"
            "def helper():\n"
            "    eval('1+1')\n"
        ),
    })
    holder = _CallGraphHolder(root.resolve())
    tool = IsReachableTool(root=root, holder=holder)
    res = await tool.execute(source="app.route", sink="eval", max_depth=3)
    data = json.loads(res.output)
    assert data["reachable"] is True


@pytest.mark.asyncio
async def test_is_reachable_tool_false_isolated_sink(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def route():\n"
            "    pass\n"
            "\n"
            "def dead():\n"
            "    eval('1+1')\n"
        ),
    })
    holder = _CallGraphHolder(root.resolve())
    tool = IsReachableTool(root=root, holder=holder)
    res = await tool.execute(source="app.route", sink="eval")
    data = json.loads(res.output)
    assert data["reachable"] is False


@pytest.mark.asyncio
async def test_call_graph_holder_caches(tmp_path: Path) -> None:
    """Building the graph more than once would be a perf bug."""
    root = _mkproject(tmp_path, {"a.py": "def f(): pass\n"})
    holder = _CallGraphHolder(root.resolve())
    g1 = holder.get()
    g2 = holder.get()
    assert g1 is g2
