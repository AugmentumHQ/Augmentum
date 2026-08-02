"""Scheduling source upgrades — subreddit routing in the curator, and the
briefing read-through pass that reads top results in full (web_fetch) instead
of synthesizing from 140-char snippets."""
from __future__ import annotations

from augmentum.companion_runtime import curator
from augmentum.companion_runtime import standing_tasks as st
from augmentum.companion_runtime.standing_tasks import _attach_read_through

# ── Subreddit routing ───────────────────────────────────────────────────

def test_detect_subreddit_matches_r_prefix():
    assert curator._detect_subreddit("r/LocalLLaMA") == "LocalLLaMA"
    assert curator._detect_subreddit("/r/LocalLLaMA") == "LocalLLaMA"
    assert curator._detect_subreddit("  r/MachineLearning  ") == "MachineLearning"


def test_detect_subreddit_ignores_bare_and_other_shapes():
    # No "r/" prefix → NOT a subreddit (avoids hijacking plain topic words).
    assert curator._detect_subreddit("localllama") == ""
    assert curator._detect_subreddit("cs.AI") == ""
    # A full reddit URL is handled by the feed_url path, not this detector.
    assert curator._detect_subreddit("https://reddit.com/r/x") == ""
    assert curator._detect_subreddit("") == ""


# ── Briefing read-through ───────────────────────────────────────────────

class _FakeResult:
    def __init__(self, success, output):
        self.success = success
        self.output = output


class _FakeFetch:
    def __init__(self, mapping):
        self._m = mapping
        self.calls = []

    async def execute(self, *, url, max_chars):
        self.calls.append(url)
        out = self._m.get(url)
        return _FakeResult(out is not None, out or "")


class _FakeRegistry:
    def __init__(self, tool):
        self._tool = tool

    def resolve(self, name):
        return self._tool if name == "web_fetch" else None


class _AppState:
    def __init__(self, registry):
        self.tool_registry = registry


async def test_read_through_attaches_excerpts():
    gathered = [{
        "topic": "news",
        "items": [
            {"title": "A", "url": "https://a.example/1", "snippet": "s"},
            {"title": "B", "url": "https://b.example/2", "snippet": "s"},
        ],
    }]
    fetch = _FakeFetch({
        "https://a.example/1": "full article A body",
        "https://b.example/2": "full article B body",
    })
    n = await _attach_read_through(_AppState(_FakeRegistry(fetch)), gathered)
    assert n == 2
    assert gathered[0]["items"][0]["excerpt"] == "full article A body"
    assert gathered[0]["items"][1]["excerpt"] == "full article B body"


async def test_read_through_respects_global_cap():
    # 5 topics x 2 items = 10 candidates; _BRIEFING_READ_MAX caps the fetches.
    gathered = [{
        "topic": f"t{i}",
        "items": [
            {"title": "x", "url": f"https://s{i}.example/a"},
            {"title": "y", "url": f"https://s{i}.example/b"},
        ],
    } for i in range(5)]
    fetch = _FakeFetch({})  # every fetch "fails" (no mapping) → no excerpt
    await _attach_read_through(_AppState(_FakeRegistry(fetch)), gathered)
    assert len(fetch.calls) == st._BRIEFING_READ_MAX


async def test_read_through_caps_per_topic():
    # One topic with 5 items → only _BRIEFING_READ_TOP get fetched.
    gathered = [{
        "topic": "t",
        "items": [{"title": "x", "url": f"https://s.example/{i}"} for i in range(5)],
    }]
    fetch = _FakeFetch({})
    await _attach_read_through(_AppState(_FakeRegistry(fetch)), gathered)
    assert len(fetch.calls) == st._BRIEFING_READ_TOP


async def test_read_through_skips_when_no_fetch_tool():
    gathered = [{"topic": "x", "items": [{"url": "https://a.example/1"}]}]
    n = await _attach_read_through(_AppState(_FakeRegistry(None)), gathered)
    assert n == 0
    assert "excerpt" not in gathered[0]["items"][0]


async def test_read_through_skips_non_http_urls():
    gathered = [{"topic": "x", "items": [{"url": "ftp://nope"}, {"url": ""}]}]
    fetch = _FakeFetch({})
    n = await _attach_read_through(_AppState(_FakeRegistry(fetch)), gathered)
    assert n == 0
    assert fetch.calls == []
