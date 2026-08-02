"""Tests for augmentum/coder/code_intel.py.

Covers:
  - extract_python / extract_js (pure extraction, no I/O)
  - build_code_intel round-trip using a fake container manager
  - has_index, find_symbol, file_outline
  - reindex_paths (incremental update / deletion)
  - render_repo_map byte-stability contract
  - FindSymbolTool / FileOutlineTool schema and registry membership
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from augmentum.coder.code_intel import (
    build_code_intel,
    extract_js,
    extract_python,
    file_outline,
    find_symbol,
    has_index,
    reindex_paths,
    render_repo_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


class FakeContainerManager:
    """Minimal container manager backed by an in-memory {abs_path: content} dict.

    Implements exactly the two interfaces code_intel uses:
      1.  ``_run_command(workspace_id, argv, timeout=...)`` — used for the
          ``find`` listing.  Returns ``"<mtime>\\t<size>\\t<abs_path>\\n"``
          lines for every file in the store.
      2.  Passed to ``_batch_read_files`` (imported from indexer), which calls
          ``_run_command(workspace_id, ['bash', '-c', script], timeout=60.0)``.
          The script echoes ``\\n@@AUGFILE:<abs_path>@@\\n<base64>`` for each
          path.  We handle this by detecting the ``@@AUGFILE`` sentinel in the
          script and returning the right output.
    """

    def __init__(self, files: dict[str, str]) -> None:
        # abs_path → content
        self.files: dict[str, str] = dict(files)

    async def _run_command(
        self, workspace_id: str, argv: list[str], timeout: float = 30.0, **_kw
    ) -> str:
        script = argv[-1] if len(argv) >= 1 and argv[0] in ("bash",) else ""

        # --- batch-read path (used by _batch_read_files in indexer) ---
        if "@@AUGFILE:" in script or "base64" in script:
            # Reconstruct output: @@AUGFILE:<path>@@\n<base64-content>
            parts: list[str] = []
            for abs_path, content in self.files.items():
                # Only emit files whose path appears in the script.
                if abs_path in script:
                    parts.append(f"\n@@AUGFILE:{abs_path}@@\n{_b64(content)}")
            return "".join(parts)

        # --- find listing path ---
        # Returns "<mtime>\t<size>\t<abs_path>" lines for all files.
        lines: list[str] = []
        for abs_path, content in self.files.items():
            size = len(content.encode())
            # Use a fixed mtime so tests can control change detection.
            lines.append(f"1700000000.0\t{size}\t{abs_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. extract_python
# ---------------------------------------------------------------------------

_PY_SRC = """\
import os
from pathlib import Path

MY_CONST = 42

class MyClass:
    def method_a(self):
        pass

    async def method_b(self):
        pass

    class Inner:
        pass

def top_func(x, y):
    return x + y

async def async_func():
    pass
"""


def test_extract_python_classes_and_methods():
    syms, _ = extract_python(_PY_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    assert kinds["MyClass"] == "class"
    assert kinds["method_a"] == "method"
    assert kinds["method_b"] == "method"


def test_extract_python_method_scope():
    syms, _ = extract_python(_PY_SRC)
    methods = [s for s in syms if s["kind"] == "method"]
    for m in methods:
        assert m["scope"] == "MyClass"


def test_extract_python_nested_class_scope():
    syms, _ = extract_python(_PY_SRC)
    inner = next((s for s in syms if s["name"] == "Inner"), None)
    assert inner is not None
    assert inner["kind"] == "class"
    assert inner["scope"] == "MyClass"


def test_extract_python_top_functions():
    syms, _ = extract_python(_PY_SRC)
    func_names = {s["name"] for s in syms if s["kind"] == "function"}
    assert "top_func" in func_names
    assert "async_func" in func_names


def test_extract_python_const_uppercase_only():
    syms, _ = extract_python(_PY_SRC)
    consts = {s["name"] for s in syms if s["kind"] == "const"}
    assert "MY_CONST" in consts
    # lower-case assignments must NOT appear
    lower_names = {s["name"] for s in syms if s["kind"] == "const" and not s["name"].isupper()}
    assert lower_names == set()


def test_extract_python_imports():
    _, imports = extract_python(_PY_SRC)
    modules = {i["module"] for i in imports}
    assert "os" in modules
    assert "pathlib" in modules


def test_extract_python_from_import_name():
    _, imports = extract_python(_PY_SRC)
    pathlib_imp = next(i for i in imports if i["module"] == "pathlib")
    assert pathlib_imp["name"] == "Path"


def test_extract_python_syntax_error_returns_empty():
    syms, imports = extract_python("def bad(:\n    pass\n")
    assert syms == []
    assert imports == []


def test_extract_python_syntax_error_no_raise():
    # Must not raise — just return empty.
    result = extract_python("this is (not valid python >>>")
    assert result == ([], [])


# ---------------------------------------------------------------------------
# 2. extract_js
# ---------------------------------------------------------------------------

_JS_SRC = """\
import { foo } from './mod';

function regularFunc(x) { return x; }

class MyBar {
  constructor() {}
}

export const baz = 42;

const arrowFn = (x) => x + 1;

export interface MyInterface {}

export type MyType = string;

export enum MyEnum { A, B }
"""


def test_extract_js_function():
    syms, _ = extract_js(_JS_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    assert kinds["regularFunc"] == "function"


def test_extract_js_class():
    syms, _ = extract_js(_JS_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    assert kinds["MyBar"] == "class"


def test_extract_js_export_const():
    syms, _ = extract_js(_JS_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    # export const baz matches _JS_CONST (export const)
    assert "baz" in kinds


def test_extract_js_arrow_function():
    syms, _ = extract_js(_JS_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    assert kinds["arrowFn"] == "function"


def test_extract_js_interface_and_type():
    syms, _ = extract_js(_JS_SRC)
    kinds = {s["name"]: s["kind"] for s in syms}
    assert kinds.get("MyInterface") == "type"
    assert kinds.get("MyType") == "type"


def test_extract_js_import_module():
    _, imports = extract_js(_JS_SRC)
    modules = {i["module"] for i in imports}
    assert "./mod" in modules


# ---------------------------------------------------------------------------
# 3. Build round-trip with a fake container manager
# ---------------------------------------------------------------------------

_WS_ID = "test-workspace-001"

_PY_CONTENT = """\
class Router:
    def dispatch(self, request):
        pass

    def register(self, route):
        pass

def build_router():
    return Router()

MY_VERSION = "1.0"
"""

_JS_CONTENT = """\
export function createHandler(cfg) {
  return cfg;
}

class EventBus {
  emit(event) {}
}
"""


@pytest.fixture
def fake_cm():
    return FakeContainerManager({
        "/workspace/router.py": _PY_CONTENT,
        "/workspace/bus.js": _JS_CONTENT,
    })


async def test_build_code_intel_has_index(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    result = await build_code_intel(fake_cm, _WS_ID, force=True)
    assert result["status"] == "ok"
    assert await has_index(_WS_ID)


async def test_find_symbol_exact_class(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    hits = await find_symbol(_WS_ID, "Router")
    assert hits
    top = hits[0]
    assert top["name"] == "Router"
    assert top["kind"] == "class"
    assert "router.py" in top["path"]
    assert top["exact"] is True


async def test_find_symbol_qualified_method(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    hits = await find_symbol(_WS_ID, "Router.dispatch")
    assert hits
    top = hits[0]
    assert top["name"] == "dispatch"
    assert top["scope"] == "Router"
    assert top["kind"] == "method"
    assert top["exact"] is True


async def test_find_symbol_fuzzy_fallback(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    # "oute" is a substring of "Router" — should fuzzy-match
    hits = await find_symbol(_WS_ID, "oute")
    assert hits
    assert hits[0]["exact"] is False


async def test_find_symbol_js_function(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    hits = await find_symbol(_WS_ID, "createHandler")
    assert hits
    assert hits[0]["kind"] == "function"
    assert "bus.js" in hits[0]["path"]


# ---------------------------------------------------------------------------
# 4. file_outline
# ---------------------------------------------------------------------------


async def test_file_outline_known_path(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    outline = await file_outline(_WS_ID, "router.py")
    assert outline is not None
    assert outline["path"] == "router.py"
    assert outline["lang"] == "python"
    sym_names = {s["name"] for s in outline["symbols"]}
    assert "Router" in sym_names
    assert "build_router" in sym_names
    assert isinstance(outline["imports"], list)


async def test_file_outline_unknown_path_returns_none(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    result = await file_outline(_WS_ID, "nonexistent.py")
    assert result is None


# ---------------------------------------------------------------------------
# 5. Incremental: reindex_paths
# ---------------------------------------------------------------------------

_NEW_PY = """\
class UpdatedRouter:
    def new_method(self):
        pass
"""


async def test_reindex_paths_updates_symbols(tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    ws_id = "incremental-ws-001"
    cm = FakeContainerManager({
        "/workspace/router.py": _PY_CONTENT,
    })
    await build_code_intel(cm, ws_id, force=True)

    # Verify original symbol present
    hits = await find_symbol(ws_id, "Router")
    assert any(h["name"] == "Router" for h in hits)

    # Update in-place: replace content, call reindex_paths
    cm.files["/workspace/router.py"] = _NEW_PY
    updated = await reindex_paths(cm, ws_id, ["/workspace/router.py"])
    assert updated == 1

    # New symbol present
    hits_new = await find_symbol(ws_id, "UpdatedRouter")
    assert hits_new
    assert hits_new[0]["name"] == "UpdatedRouter"


async def test_reindex_paths_deletes_removed_file(tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    ws_id = "delete-ws-001"
    cm = FakeContainerManager({
        "/workspace/router.py": _PY_CONTENT,
    })
    await build_code_intel(cm, ws_id, force=True)

    # Ensure symbol is in the index
    hits = await find_symbol(ws_id, "Router")
    assert hits

    # Simulate file deletion: remove from cm and reindex with the path
    del cm.files["/workspace/router.py"]
    # reindex_paths treats a missing file (no content returned) as deleted
    await reindex_paths(cm, ws_id, ["/workspace/router.py"])

    # Symbol should be gone
    hits_after = await find_symbol(ws_id, "Router")
    # If there's still a fuzzy hit it won't be exact; exact should be gone
    exact_hits = [h for h in hits_after if h.get("exact") is True]
    assert exact_hits == []


# ---------------------------------------------------------------------------
# 6. Byte-stable repo map
# ---------------------------------------------------------------------------


async def test_render_repo_map_is_byte_stable(fake_cm, tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    await build_code_intel(fake_cm, _WS_ID, force=True)

    map1 = await render_repo_map(_WS_ID)
    map2 = await render_repo_map(_WS_ID)
    assert map1 == map2
    assert map1 != ""


async def test_render_repo_map_body_change_does_not_alter_map(tmp_path, monkeypatch):
    """Changing a function BODY without renaming/adding symbols must leave
    the rendered map byte-identical — that is the KV-prefix cache property.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    ws_id = "stable-body-ws"
    original = """\
def compute(x):
    return x + 1
"""
    cm = FakeContainerManager({"/workspace/core.py": original})
    await build_code_intel(cm, ws_id, force=True)
    map_before = await render_repo_map(ws_id)

    # Change the body only — symbol name unchanged
    mutated = """\
def compute(x):
    # Rewritten body: now does multiplication
    return x * 2 + 42
"""
    cm.files["/workspace/core.py"] = mutated
    await reindex_paths(cm, ws_id, ["/workspace/core.py"])
    map_after = await render_repo_map(ws_id)

    assert map_before == map_after


async def test_render_repo_map_changes_on_new_symbol(tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    ws_id = "symbol-change-ws"
    original = "def alpha(): pass\n"
    cm = FakeContainerManager({"/workspace/core.py": original})
    await build_code_intel(cm, ws_id, force=True)
    map_before = await render_repo_map(ws_id)

    # Add a new function
    with_new = "def alpha(): pass\ndef beta(): pass\n"
    cm.files["/workspace/core.py"] = with_new
    await reindex_paths(cm, ws_id, ["/workspace/core.py"])
    map_after = await render_repo_map(ws_id)

    assert map_before != map_after


def test_render_repo_map_no_line_numbers(fake_cm, tmp_path, monkeypatch):
    """The map must contain no line-number annotations or timestamps."""
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)

    _run(build_code_intel(fake_cm, _WS_ID, force=True))
    repo_map = _run(render_repo_map(_WS_ID))

    import re
    # No "line N" style annotations
    assert not re.search(r"\bline\s+\d+", repo_map, re.IGNORECASE)
    # No timestamp-like patterns (YYYY-MM-DD or epoch seconds in the body)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", repo_map)


# ---------------------------------------------------------------------------
# 7. Tool schemas
# ---------------------------------------------------------------------------


def test_find_symbol_tool_schema():
    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import ALL_CODER_TOOLS, READ_ONLY_TOOLS, FindSymbolTool

    # Minimal instantiation (container_manager / workspace_id not exercised
    # in schema-only checks; _CoderTool.__init__ requires them as kwargs).
    class _DummyCM:
        pass

    state = CoderState(session_id="s", workspace_id="ws")
    tool = FindSymbolTool(
        container_manager=_DummyCM(),
        workspace_id="ws",
        state=state,
    )

    assert tool.name == "find_symbol"
    schema = tool.input_schema
    assert schema["type"] == "object"
    required = schema.get("required", [])
    assert "name" in required

    # Membership checks
    tool_classes = {cls.__name__ for cls in ALL_CODER_TOOLS}
    assert "FindSymbolTool" in tool_classes
    assert "find_symbol" in READ_ONLY_TOOLS


def test_file_outline_tool_schema():
    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import ALL_CODER_TOOLS, READ_ONLY_TOOLS, FileOutlineTool

    class _DummyCM:
        pass

    state = CoderState(session_id="s", workspace_id="ws")
    tool = FileOutlineTool(
        container_manager=_DummyCM(),
        workspace_id="ws",
        state=state,
    )

    assert tool.name == "file_outline"
    schema = tool.input_schema
    required = schema.get("required", [])
    assert "paths" in required

    tool_classes = {cls.__name__ for cls in ALL_CODER_TOOLS}
    assert "FileOutlineTool" in tool_classes
    assert "file_outline" in READ_ONLY_TOOLS
