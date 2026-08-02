"""Regression: the per-turn relevance filter must NEVER hide the image tools.

Image intent is phrased too many ways for the keyword regex to catch ("make me
a picture of a sunset" doesn't match `make\\s+(an?\\s+)?(image|picture|photo)`),
and when the schema is dropped the model can't see the tool — it denies the
capability and falls back to web search. So image_generation / image_search are
exempt from the filter (``_ALWAYS_INCLUDE``): present every turn, the model
decides via native tool-calling. They are only kept when actually in the
toolset — an unregistered image backend is not conjured.
"""

from __future__ import annotations

from augmentum.tools.filter import filter_tools_for_query


class _FakeTool:
    def __init__(self, name: str, category: str = "search") -> None:
        self.name = name
        self.category = category

    def health_check(self) -> bool:
        return True


def _toolset() -> list[_FakeTool]:
    # A realistic Auto toolset (large enough to trigger the safe-default path).
    return [
        _FakeTool("web", "search"),
        _FakeTool("wikipedia", "search"),
        _FakeTool("youtube", "fetch"),
        _FakeTool("python_exec", "execute"),
        _FakeTool("image_search", "search"),
        _FakeTool("image_generation", "image"),
        _FakeTool("build_application", "artifact"),
        _FakeTool("create_ebook", "artifact"),
    ]


# The phrasings that used to silently drop image_generation. The first is the
# exact shape from the bug report; the rest are common asks the regex misses.
_IMAGE_ASKS_THE_REGEX_MISSES = [
    "make me a picture of a sunset",
    "whip up a logo for my band",
    "I'd love to see a cozy cabin in the snow",
    "can you do a portrait of my dog",
]

# Turns with NO image intent at all — the tool must still be PRESENT (offered),
# because the model is the right arbiter of whether to call it.
_NON_IMAGE_TURNS = [
    "hello how are you",
    "what is the capital of Australia",
    "summarize this text for me",
]


def _names(tools):
    return {t.name for t in tools}


def test_image_generation_survives_phrasings_the_regex_misses():
    for q in _IMAGE_ASKS_THE_REGEX_MISSES:
        out = filter_tools_for_query(q, _toolset(), min_tools=2)
        assert "image_generation" in _names(out), f"dropped on: {q!r}"


def test_image_tools_present_even_on_non_image_turns():
    # The model decides; the schema must be there for it to choose.
    for q in _NON_IMAGE_TURNS:
        out = filter_tools_for_query(q, _toolset(), min_tools=2)
        names = _names(out)
        assert "image_generation" in names, f"dropped on: {q!r}"
        assert "image_search" in names, f"dropped on: {q!r}"


def test_explicit_image_keyword_still_works():
    out = filter_tools_for_query("draw a fox", _toolset(), min_tools=2)
    assert "image_generation" in _names(out)


def test_not_conjured_when_absent_from_toolset():
    # No image backend registered → tool simply isn't in the list → not added.
    toolset = [_FakeTool("web", "search"), _FakeTool("wikipedia", "search")]
    out = filter_tools_for_query("make me a picture of a sunset", toolset, min_tools=2)
    assert "image_generation" not in _names(out)
    assert "image_search" not in _names(out)
