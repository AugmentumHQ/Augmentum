"""Tests for augmentum/coder/ — models, state, harness, indexer, repomap, web_tools."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

import pytest

from augmentum.coder.models import ContainerInfo, FileEntry, WorkspaceConfig
from augmentum.coder.state import CoderPhase, CoderState
from augmentum.coder.harness import select_harness
from augmentum.coder.repomap import _format_definitions, _DEF_PATTERNS
from augmentum.coder.indexer import CodeChunk, SearchResult, _INDEXABLE_EXTS, _SKIP_DIRS

# web_tools has a circular import with tools.py at module level.
# Import individual symbols lazily to avoid collection-time ImportError.
import importlib as _il


def _get_web_tools():
    """Lazy import to avoid circular import at collection time."""
    _il.import_module("augmentum.coder.tools")
    mod = _il.import_module("augmentum.coder.web_tools")
    return mod


_domain_from_url = None
_rerank_results = None
_DOC_DOMAINS = None
_AVOID_DOMAINS = None


def _ensure_web_tools():
    global _domain_from_url, _rerank_results, _DOC_DOMAINS, _AVOID_DOMAINS
    if _domain_from_url is None:
        mod = _get_web_tools()
        _domain_from_url = mod._domain_from_url
        _rerank_results = mod._rerank_results
        _DOC_DOMAINS = mod._DOC_DOMAINS
        _AVOID_DOMAINS = mod._AVOID_DOMAINS


class TestContainerInfo:
    """Verify ContainerInfo dataclass."""

    def test_defaults(self):
        ci = ContainerInfo(id="ws1", name="test")
        assert ci.status == "stopped"
        assert ci.resources_cpu == 2.0
        assert ci.resources_memory == "2g"
        assert ci.container_id is None

    def test_with_all_fields(self):
        ci = ContainerInfo(
            id="ws1", name="test", container_id="cnt_abc",
            status="running", git_url="https://github.com/test/repo",
        )
        assert ci.status == "running"
        assert ci.git_url == "https://github.com/test/repo"


class TestFileEntry:
    """Verify FileEntry dataclass."""

    def test_construction(self):
        fe = FileEntry(name="main.py", path="/workspace/main.py", is_dir=False, size=1024)
        assert fe.name == "main.py"
        assert fe.is_dir is False


class TestWorkspaceConfig:
    """Verify WorkspaceConfig dataclass."""

    def test_defaults(self):
        wc = WorkspaceConfig(name="test")
        # Pre-baked image with the standard coder tool chain already
        # installed (ripgrep, fd, build-essential, etc). Falls back to
        # ubuntu:24.04 with runtime install if the image isn't built.
        # See containers.py::ContainerManager.create_workspace.
        assert wc.base_image == "augmentum-workspace"
        assert wc.cpu == 2.0
        assert wc.memory == "2g"
        assert wc.pids == 256
        assert wc.packages == []


class TestCoderState:
    """Verify CoderState lifecycle and serialization."""

    def test_default_phase(self):
        state = CoderState(session_id="s1", workspace_id="w1")
        assert state.phase == CoderPhase.WAITING

    def test_progress_zero_with_no_steps(self):
        state = CoderState(session_id="s1", workspace_id="w1")
        assert state.progress_pct == 0.0

    def test_progress_after_steps(self):
        state = CoderState(
            session_id="s1", workspace_id="w1",
            plan_steps=["step1", "step2", "step3", "step4"],
        )
        state.advance_step("output1")
        assert state.current_step == 1
        assert state.progress_pct == 25.0

    def test_advance_past_end_is_noop(self):
        state = CoderState(
            session_id="s1", workspace_id="w1",
            plan_steps=["step1"],
        )
        state.advance_step("done")
        state.advance_step("extra")  # should be noop
        assert state.current_step == 1

    def test_record_file_read(self):
        state = CoderState(session_id="s1", workspace_id="w1")
        state.record_file_read("/workspace/main.py")
        assert "/workspace/main.py" in state.files_read
        assert "/workspace/main.py" in state.working_set

    def test_can_edit_requires_read(self):
        state = CoderState(session_id="s1", workspace_id="w1")
        assert state.can_edit("/workspace/main.py") is False
        state.record_file_read("/workspace/main.py")
        assert state.can_edit("/workspace/main.py") is True

    def test_to_dict_round_trip(self):
        # files_read schema is now ``dict[str, float]`` (path → mtime)
        # so the mtime-aware staleness check in ``can_edit`` works —
        # see CoderState docstring. Pre-mtime tests stored a bare set
        # which broke after the schema migration. Use float("inf") here
        # for the never-stale sentinel the schema treats as "no
        # staleness info captured".
        state = CoderState(
            session_id="s1", workspace_id="w1",
            phase=CoderPhase.EXECUTING,
            plan_steps=["a", "b"],
            working_set={"file1.py", "file2.py"},
            files_read={"file1.py": float("inf")},
        )
        d = state.to_dict()
        restored = CoderState.from_row(d)
        assert restored.session_id == "s1"
        assert restored.phase == CoderPhase.EXECUTING
        assert restored.plan_steps == ["a", "b"]
        assert "file1.py" in restored.files_read
        assert "file2.py" in restored.working_set

    def test_to_dict_json_serializable(self):
        state = CoderState(session_id="s1", workspace_id="w1")
        d = state.to_dict()
        # All values should be JSON-serializable
        json.dumps(d)


class TestSelectHarness:
    """Verify harness selection defaults to react."""

    def test_unknown_model_defaults_react(self):
        result = select_harness("some-random-model:7b")
        assert result == "react"

    def test_returns_string(self):
        result = select_harness("llama3.1:8b")
        assert isinstance(result, str)
        assert result in ("rewoo", "react")


class TestIndexerConstants:
    """Verify indexer configuration constants."""

    def test_python_is_indexable(self):
        assert ".py" in _INDEXABLE_EXTS

    def test_node_modules_skipped(self):
        assert "node_modules" in _SKIP_DIRS

    def test_git_skipped(self):
        assert ".git" in _SKIP_DIRS


class TestCodeChunkAndSearchResult:
    """Verify indexer data classes."""

    def test_code_chunk(self):
        chunk = CodeChunk(
            file_path="src/main.py", start_line=1, end_line=50,
            content="def hello(): pass", file_hash="abc123",
        )
        assert chunk.file_path == "src/main.py"

    def test_search_result(self):
        sr = SearchResult(
            file_path="src/main.py", start_line=1, end_line=10,
            content="def hello(): pass", score=0.95,
        )
        assert sr.score == 0.95


class TestWebToolsHelpers:
    """Verify web tools utility functions."""

    def test_domain_from_url(self):
        _ensure_web_tools()
        assert _domain_from_url("https://docs.python.org/3/library/") == "docs.python.org"

    def test_domain_from_url_invalid(self):
        _ensure_web_tools()
        assert _domain_from_url("not-a-url") == ""

    def test_doc_domains_contains_python(self):
        _ensure_web_tools()
        assert "docs.python.org" in _DOC_DOMAINS

    def test_avoid_domains_contains_w3schools(self):
        _ensure_web_tools()
        assert "w3schools.com" in _AVOID_DOMAINS

    def test_rerank_boosts_doc_domains(self):
        _ensure_web_tools()
        results = [
            {"url": "https://w3schools.com/python", "_score": 1.0},
            {"url": "https://docs.python.org/3/tutorial/", "_score": 0.5},
        ]
        ranked = _rerank_results(results)
        # docs.python.org should be ranked first
        assert "docs.python.org" in ranked[0]["url"]


class TestDocFetchTool:
    """Verify coder-mode documentation fetching."""

    @pytest.mark.asyncio
    async def test_doc_fetch_uses_safe_http_fetch(self):
        mod = _get_web_tools()
        tool = mod.DocFetchTool(
            container_manager=MagicMock(),
            workspace_id="ws_test",
            state=MagicMock(),
        )

        mock_client = AsyncMock()
        mock_client.fetch.return_value = (
            "<html><body><h1>Asyncio</h1><p>Hello docs.</p></body></html>",
            {"url": "https://docs.python.org/3/library/asyncio.html"},
        )

        with patch("augmentum.utils.safe_http.SafeHttpClient", return_value=mock_client):
            result = await tool.execute(url="https://docs.python.org/3/library/asyncio.html")

        assert result.success is True
        assert "Hello docs." in result.output
        assert "https://docs.python.org/3/library/asyncio.html" in result.output
        mock_client.fetch.assert_awaited_once_with(
            "https://docs.python.org/3/library/asyncio.html",
            timeout=15.0,
        )


class TestFormatDefinitions:
    """Verify definition formatting."""

    def test_formats_grep_output(self):
        raw = "/workspace/src/main.py:1:def hello():\n/workspace/src/main.py:5:class App:\n"
        result = _format_definitions(raw, 1000)
        assert "main.py" in result
        assert "def hello():" in result

    def test_respects_budget(self):
        raw = "/workspace/a.py:1:def f():\n" * 100
        result = _format_definitions(raw, 50)
        assert len(result) <= 100  # Some overhead, but stays near budget

    def test_empty_input(self):
        result = _format_definitions("", 1000)
        assert result == ""


class TestDefPatterns:
    """Verify definition regex patterns exist for key languages."""

    def test_python_pattern(self):
        assert "py" in _DEF_PATTERNS

    def test_javascript_pattern(self):
        assert "js" in _DEF_PATTERNS

    def test_rust_pattern(self):
        assert "rs" in _DEF_PATTERNS
