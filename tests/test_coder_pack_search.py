"""Unit tests for the coder pack_search tool (2026-07-06).

Offline knowledge packs were chat-injection-only; pack_search gives the
coder loop reach into them. Key contract (Matt's design): the tool
DESCRIPTION carries the installed-pack inventory (name + curation date)
so the model never burns a call discovering coverage.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import augmentum.coder.tools  # noqa: F401 — break tools<->web_tools circularity first
from augmentum.coder.knowledge_tools import PackSearchTool
from augmentum.knowledge.runtime import set_pack_manager


@dataclass
class _FakeResult:
    content: str = "TaskGroup cancels remaining tasks on first failure."
    title: str = "asyncio — Task Groups"
    section: str = "TaskGroup"
    url: str = "devdocs/python~3.12/asyncio-task"
    pack_id: str = "devdocs_en_python"
    source: str = "devdocs"
    score: float = 0.9


@dataclass
class _FakeManager:
    installed_rows: list = field(default_factory=list)
    search_results: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    @property
    def installed(self):
        return self.installed_rows

    async def search(self, query, *, pack_ids, limit=5, rerank=True):
        self.calls.append((query, tuple(pack_ids)))
        return self.search_results


def _tool():
    return PackSearchTool(container_manager=None, workspace_id="w", state=None)


def teardown_function(_fn):
    set_pack_manager(None)


def test_description_lists_installed_packs_with_dates():
    set_pack_manager(_FakeManager(installed_rows=[
        {"pack_id": "devdocs_en_python", "name": "DevDocs Python", "build_date": "2026-03", "active": True},
        {"pack_id": "devdocs_en_javascript", "name": "DevDocs JS", "build_date": "2026-02", "active": True},
        {"pack_id": "old_pack", "name": "Disabled", "build_date": "2020-01", "active": False},
    ]))
    desc = _tool().description
    assert "DevDocs Python (curated 2026-03)" in desc
    assert "DevDocs JS (curated 2026-02)" in desc
    assert "Disabled" not in desc          # inactive packs hidden
    assert "FIRST RESORT" in desc


def test_description_with_no_packs_steers_away():
    set_pack_manager(None)
    desc = _tool().description
    assert "NO PACKS" in desc and "doc_search" in desc


def test_execute_searches_active_packs():
    mgr = _FakeManager(
        installed_rows=[{"pack_id": "devdocs_en_python", "name": "DevDocs Python",
                         "build_date": "2026-03", "active": True}],
        search_results=[_FakeResult()],
    )
    set_pack_manager(mgr)
    r = asyncio.run(_tool().execute(query="asyncio TaskGroup cancellation"))
    assert r.success
    assert "TaskGroup" in r.output and "devdocs_en_python" in r.output
    assert mgr.calls and mgr.calls[0][1] == ("devdocs_en_python",)


def test_execute_without_packs_degrades_with_pointer():
    set_pack_manager(None)
    r = asyncio.run(_tool().execute(query="anything"))
    assert not r.success
    assert "doc_search" in (r.error or "")


def test_no_results_points_to_web_not_rephrase_loop():
    mgr = _FakeManager(
        installed_rows=[{"pack_id": "p", "name": "P", "build_date": "", "active": True}],
        search_results=[],
    )
    set_pack_manager(mgr)
    r = asyncio.run(_tool().execute(query="obscure thing"))
    assert r.success and "doc_search" in r.output
