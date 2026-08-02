"""Brief → structure planner for gated structured creators (ebook/deck/doc).

One LLM call expands a one-line brief into the tool's full input, so the
confirmation chip can show the OUTLINE and Accept runs the real tool with the
plan. Bad/garbage model output degrades to None (caller falls back), never raises.

See augmentum/modes/passthrough/gated_planner.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.modes.passthrough.gated_planner import (
    expand_brief,
    is_planned_tool,
    outline_summary,
)


class _FakeBackend:
    """Returns a canned content string from .chat()."""

    def __init__(self, content: str):
        self._content = content

    async def chat(self, _req):
        return SimpleNamespace(message=SimpleNamespace(content=self._content))


_GOOD_EBOOK = """Here is your outline:
```json
{"title": "Bramble the Brave", "author": "Auto",
 "chapters": [
   {"heading": "The Pond", "body": "Bramble lived by a quiet pond.", "illustration": "a turtle by a pond"},
   {"heading": "The Storm", "body": "One night a great storm came.", "illustration": "a storm over water"},
   {"heading": "The Rescue", "body": "Bramble saved a tiny frog.", "illustration": "a turtle and a frog"}
 ]}
```
Hope you like it!"""


def test_is_planned_tool():
    assert is_planned_tool("create_ebook")
    assert is_planned_tool("create_presentation")
    assert not is_planned_tool("image_generation")
    assert not is_planned_tool("nope")


@pytest.mark.asyncio
async def test_expand_brief_parses_json_amid_prose():
    out = await expand_brief("create_ebook", "a kids book about a brave turtle",
                             _FakeBackend(_GOOD_EBOOK))
    assert out is not None
    assert out["title"] == "Bramble the Brave"
    assert len(out["chapters"]) == 3
    assert all("heading" in c and "body" in c for c in out["chapters"])


@pytest.mark.asyncio
async def test_outline_summary_lists_headings():
    out = await expand_brief("create_ebook", "turtle book", _FakeBackend(_GOOD_EBOOK))
    summary = outline_summary("create_ebook", out)
    assert "Bramble the Brave" in summary
    assert "3 chapters" in summary
    assert "The Pond" in summary


@pytest.mark.asyncio
async def test_expand_brief_drops_incomplete_sections():
    bad_sections = (
        '{"title": "T", "chapters": ['
        '{"heading": "ok", "body": "real text"},'
        '{"heading": "", "body": ""},'         # blank → dropped
        '{"heading": "missing body"}]}'         # no body → dropped
    )
    out = await expand_brief("create_ebook", "x", _FakeBackend(bad_sections))
    assert out is not None
    assert len(out["chapters"]) == 1
    assert out["chapters"][0]["heading"] == "ok"


@pytest.mark.asyncio
async def test_expand_brief_none_on_garbage():
    assert await expand_brief("create_ebook", "x", _FakeBackend("no json here")) is None
    assert await expand_brief("create_ebook", "x", _FakeBackend('{"title":"T"}')) is None  # no list
    assert await expand_brief("create_ebook", "x", _FakeBackend("{bad json")) is None


@pytest.mark.asyncio
async def test_expand_brief_unknown_tool_or_empty_brief():
    assert await expand_brief("image_generation", "x", _FakeBackend(_GOOD_EBOOK)) is None
    assert await expand_brief("create_ebook", "", _FakeBackend(_GOOD_EBOOK)) is None


@pytest.mark.asyncio
async def test_expand_brief_presentation_shape():
    deck = '{"title": "Q3", "slides": [{"title": "Intro", "bullets": ["a","b"]}, {"title": "End"}]}'
    out = await expand_brief("create_presentation", "q3 review", _FakeBackend(deck))
    assert out is not None and out["title"] == "Q3"
    assert len(out["slides"]) == 2  # both have a title (the required key)
