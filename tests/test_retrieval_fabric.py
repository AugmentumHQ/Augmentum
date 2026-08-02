"""Retrieval fabric P1 — the resolve→act/offer/miss lifecycle.

Spec: docs/superpowers/specs/2026-06-11-retrieval-fabric-design.md.
Pins: single-source confidence gates (media-resolver thresholds),
multi-source never-auto-act + RRF merge, leg soft-failure, kind
filtering, and the files.find consumer contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.retrieval.fabric import resolve


class _Entry(SimpleNamespace):
    """FileIndex entry stand-in (attribute access like FileEntry)."""


def _entry(name, file_id, kind="document", **kw):
    base = dict(
        name=name, id=file_id, kind=kind, is_directory=False,
        source="uploads", mime_type="", size=0,
    )
    base.update(kw)
    return _Entry(**base)


class _FakeIndex:
    def __init__(self, entries):
        self._entries = entries
        self.queries = []

    async def search(self, query, *, user_id="", kind=None, limit=10):
        self.queries.append(query)
        return self._entries


class _FakePackManager:
    def __init__(self, results, packs=None):
        self._results = results
        self._packs = packs or [{"id": "p1"}]

    def installed(self):
        return self._packs

    async def search(self, query, *, pack_ids, limit=5, rerank=True):
        return self._results


def _pack_result(title, score=0.5):
    return SimpleNamespace(
        content="...", title=title, section="History", url=f"zim://{title}",
        pack_id="p1", source="wikipedia", score=score,
    )


# ── Single-source lifecycle (index leg) ───────────────────────────────

@pytest.mark.asyncio
async def test_exact_title_acts():
    state = SimpleNamespace(file_index=_FakeIndex([
        _entry("Dune Part Two.mkv", "f1", kind="video"),
        _entry("Sand Dunes Documentary.mkv", "f2", kind="video"),
    ]), pack_manager=None)
    r = await resolve("dune part two", user_id="u1", app_state=state,
                      sources=("index",))
    assert r.outcome == "act"
    assert r.item.id == "f1"
    assert "play" in r.item.actions


@pytest.mark.asyncio
async def test_ambiguous_titles_offer_not_act():
    state = SimpleNamespace(file_index=_FakeIndex([
        _entry("Project Notes v1.md", "f1"),
        _entry("Project Notes v2.md", "f2"),
    ]), pack_manager=None)
    r = await resolve("project notes", user_id="u1", app_state=state,
                      sources=("index",))
    # Two near-identical scores can't clear the margin — offer, never
    # a confident wrong open.
    assert r.outcome == "offer"
    assert len(r.candidates) == 2


@pytest.mark.asyncio
async def test_no_match_is_honest_miss():
    state = SimpleNamespace(file_index=_FakeIndex([
        _entry("Grocery list.txt", "f1"),
    ]), pack_manager=None)
    r = await resolve("quarterly tax filing", user_id="u1", app_state=state,
                      sources=("index",))
    assert r.outcome == "miss"


@pytest.mark.asyncio
async def test_kind_filter_excludes():
    state = SimpleNamespace(file_index=_FakeIndex([
        _entry("Dune.mkv", "f1", kind="video"),
        _entry("Dune.epub", "f2", kind="document"),
    ]), pack_manager=None)
    r = await resolve("dune", user_id="u1", app_state=state,
                      sources=("index",), kinds=("document",))
    assert all(c.kind == "document" for c in r.candidates)


# ── Multi-source: never auto-act, RRF merge ──────────────────────────

@pytest.mark.asyncio
async def test_multi_source_never_acts_even_on_exact():
    state = SimpleNamespace(
        file_index=_FakeIndex([_entry("Rogue Class Guide.pdf", "f1")]),
        pack_manager=_FakePackManager([_pack_result("Rogue (D&D class)")]),
    )
    r = await resolve("rogue class guide", user_id="u1", app_state=state)
    assert r.outcome == "offer"
    sources = {c.source for c in r.candidates}
    assert sources == {"index", "packs"}
    assert set(r.legs) == {"index", "packs"}


@pytest.mark.asyncio
async def test_failed_leg_soft_fails_other_leg_survives():
    class _BrokenIndex:
        async def search(self, *a, **kw):
            raise RuntimeError("index exploded")

    state = SimpleNamespace(
        file_index=_BrokenIndex(),
        pack_manager=_FakePackManager([_pack_result("Kyoto")]),
    )
    r = await resolve("kyoto", user_id="u1", app_state=state)
    assert r.outcome == "offer"
    assert r.candidates[0].source == "packs"
    assert "infuse" in r.candidates[0].actions


@pytest.mark.asyncio
async def test_disabled_packs_excluded():
    state = SimpleNamespace(
        file_index=_FakeIndex([]),
        pack_manager=_FakePackManager(
            [_pack_result("X")], packs=[{"id": "p1", "disabled": True}],
        ),
    )
    r = await resolve("anything", user_id="u1", app_state=state,
                      sources=("packs",))
    assert r.outcome == "miss"


@pytest.mark.asyncio
async def test_empty_query_and_anon_miss():
    state = SimpleNamespace(file_index=_FakeIndex([]), pack_manager=None)
    assert (await resolve("", user_id="u1", app_state=state)).outcome == "miss"
    assert (await resolve("x", user_id="", app_state=state)).outcome == "miss"


# ── files.find consumer contract ─────────────────────────────────────

@pytest.mark.asyncio
async def test_files_find_acts_on_confident_match():
    import augmentum.architect.primitives  # noqa: F401 — registers verbs
    from augmentum.intent.action import SessionContext
    from augmentum.intent.registry import REGISTRY

    action = REGISTRY.get("files.find")
    state = SimpleNamespace(
        file_index=_FakeIndex([_entry("Resume 2026.pdf", "f9")]),
        pack_manager=None,
        settings_store=None,
    )
    session = SessionContext(user_id="u_ff1", session_id="s_ff1",
                             mode=None, app_state=state)
    result = await action.handler("", session, {"query": "resume 2026"})
    assert result.surface_emit["channel"] == "files.open"
    assert result.surface_emit["payload"]["file_id"] == "f9"


@pytest.mark.asyncio
async def test_files_find_offers_on_ambiguity():
    import augmentum.architect.primitives  # noqa: F401
    from augmentum.intent.action import SessionContext
    from augmentum.intent.registry import REGISTRY

    action = REGISTRY.get("files.find")
    state = SimpleNamespace(
        file_index=_FakeIndex([
            _entry("Budget draft v1.xlsx", "f1"),
            _entry("Budget draft v2.xlsx", "f2"),
        ]),
        pack_manager=None,
        settings_store=None,
    )
    session = SessionContext(user_id="u_ff2", session_id="s_ff2",
                             mode=None, app_state=state)
    result = await action.handler("", session, {"query": "budget draft"})
    assert result.surface_emit["channel"] == "files.search_open"
    assert "Budget draft" in result.speak
