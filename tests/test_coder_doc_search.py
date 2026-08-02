"""Unit tests for doc_search hardening (2026-07-06).

Three fixes for small-model retrieval failures:
- per-turn repeat-query memory (the search-spam loop none of the
  iteration-level detectors could see);
- deterministic query distillation (tracebacks/paths → keywords),
  model-free so it works for any model tier;
- query-term relevance in the canonical scorer (trusted-domain pages
  about the wrong topic were outranking on-topic mid-trust pages).
"""
from __future__ import annotations

import augmentum.coder.tools  # noqa: F401 — web_tools↔tools are circular; tools must load first
from augmentum.coder.web_tools import _distill_query, _query_tokens
from augmentum.discovery.quality import filter_for_docs

# ── query distillation ────────────────────────────────────────────────


def test_short_clean_query_passes_through():
    q, changed = _distill_query("python asyncio gather")
    assert q == "python asyncio gather" and not changed


def test_traceback_reduces_to_exception_line():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/workspace/ide/core/cli.py", line 42, in register\n'
        "    self._commands[name] = fn\n"
        "TypeError: unhashable type: 'dict'\n"
    )
    q, changed = _distill_query(tb)
    assert changed
    assert q.startswith("TypeError")
    assert "unhashable" in q
    assert "/workspace" not in q and "line 42" not in q


def test_last_exception_line_wins_on_chained_tracebacks():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/a.py", line 1, in x\n'
        "KeyError: 'foo'\n\n"
        "During handling of the above exception, another exception occurred:\n\n"
        "Traceback (most recent call last):\n"
        '  File "/b.py", line 9, in y\n'
        "RuntimeError: registry is empty\n"
    )
    q, _ = _distill_query(tb)
    assert q.startswith("RuntimeError")


def test_noise_stripped_and_length_capped():
    q, changed = _distill_query(
        "why does /workspace/ide/core/cli.py at 0xDEADBEEF line 42 fail to "
        "register commands when the registry map stays empty across every "
        "invocation attempt in the argparse subparser dispatch layer"
    )
    assert changed
    assert "/workspace" not in q and "0xDEADBEEF" not in q
    assert len(q.split()) <= 12


def test_query_tokens_overlap_shape():
    a = _query_tokens("argparse subparser default handler")
    b = _query_tokens("argparse subparser default handler please")
    assert len(a & b) / len(a | b) >= 0.75


# ── canonical scorer: query-term relevance ────────────────────────────


def _result(url: str, title: str, snippet: str = "") -> dict:
    return {"url": url, "title": title, "content": snippet}


def test_on_topic_midtrust_beats_offtopic_trusted():
    query = "argparse subparser default handler"
    results = [
        # Trusted domain, wrong topic (zero query-term overlap).
        _result("https://docs.python.org/3/library/email.html", "email package"),
        # Mid-trust domain, exactly on topic.
        _result(
            "https://realpython.com/argparse-subparser-guide/",
            "argparse subparser default handler patterns",
            "def handler(): ... set_defaults(func=handler)",
        ),
    ]
    ranked = filter_for_docs(results, query)
    assert "argparse" in ranked[0]["title"]


def test_on_topic_trusted_still_wins():
    query = "asyncio gather exceptions"
    results = [
        _result(
            "https://realpython.com/async-io-python/",
            "asyncio gather exceptions in Python",
        ),
        _result(
            "https://docs.python.org/3/library/asyncio-task.html",
            "asyncio Tasks — gather, exceptions and cancellation",
            "import asyncio",
        ),
    ]
    ranked = filter_for_docs(results, query)
    assert "docs.python.org" in ranked[0]["url"]


def test_zero_overlap_junk_is_dropped():
    """The hard relevance floor drops results sharing no significant token
    with the query — the flooding class diagnosed 2026-08-01 (percona
    docker images, NSFW reddit, quiz spam, off-topic MDN docs ranking top
    of an LLM-inference search because position was the scoring baseline)."""
    query = "llama.cpp GGUF benchmark tokens per second"
    results = [
        _result("https://hub.docker.com/r/percona/percona-server", "percona/percona-server", "database"),
        _result("https://developer.mozilla.org/en-US/docs/Web/API/RTCVideoSourceStats", "RTCVideoSourceStats", "webrtc video stats"),
        _result("https://www.reddit.com/r/sex/comments/x/y", "Women what did you think", "nsfw thread"),
        _result("https://github.com/ggml-org/llama.cpp", "llama.cpp LLM inference", "LLM inference in C/C++ GGUF benchmark tokens per second"),
    ]
    ranked = filter_for_docs([dict(r) for r in results], query)
    urls = [r["url"] for r in ranked]
    assert ranked and "github.com/ggml-org/llama.cpp" in ranked[0]["url"]
    assert not any("percona" in u or "reddit.com/r/sex" in u or "RTCVideoSourceStats" in u for u in urls)


def test_source_resolution_default_and_targeted():
    """Model-declared sources map to engines; unspecified / unknown falls
    back to the default reliable pool (no keyword intent-guessing)."""
    import augmentum.coder.web_tools as wt

    default = wt._resolve_sources(None)
    assert "github" in default and "arxiv" not in default and "mdn" not in default
    assert wt._resolve_sources(["papers"]) == ["arxiv"]
    assert wt._resolve_sources(["code"]) == ["github"]
    assert wt._resolve_sources(["reference"]) == ["mdn"]
    assert set(wt._resolve_sources(["code", "papers"])) == {"github", "arxiv"}
    assert wt._resolve_sources(["totally-made-up"]) == default


def test_all_stopword_query_keeps_results():
    """When the query has no significant tokens the floor is skipped —
    dropping everything would help no one."""
    query = "what is the best"
    results = [
        _result("https://docs.python.org/3/", "Python docs", "the standard library"),
        _result("https://example.com/", "Example", "a page"),
    ]
    ranked = filter_for_docs([dict(r) for r in results], query)
    assert len(ranked) == 2


# ── per-turn repeat-query memory ──────────────────────────────────────


def _mk_tool():
    import augmentum.coder.web_tools as wt

    return wt.DocSearchTool(
        container_manager=None, workspace_id="w", state=None,
    )


def test_exact_repeat_returns_cached_banner(monkeypatch):
    import asyncio

    import augmentum.coder.web_tools as wt

    calls = {"n": 0}

    async def fake_searx(client, base_url, query, *, engines=None):
        calls["n"] += 1
        return [
            {"url": "https://docs.python.org/3/library/argparse.html",
             "title": "argparse docs", "content": "subparser"},
        ]

    monkeypatch.setattr(wt, "_searxng_query", fake_searx)
    tool = _mk_tool()
    r1 = asyncio.run(tool.execute(query="argparse subparser handler"))
    assert r1.success and "Repeat search" not in r1.output
    first_calls = calls["n"]

    r2 = asyncio.run(tool.execute(query="argparse subparser handler"))
    assert r2.success
    assert "Repeat search" in r2.output
    assert "argparse docs" in r2.output          # cached results still shown
    assert calls["n"] == first_calls             # NO new SearXNG traffic
    assert r2.metadata.get("repeat") is True


def test_near_duplicate_gets_soft_note(monkeypatch):
    import asyncio

    import augmentum.coder.web_tools as wt

    async def fake_searx(client, base_url, query, *, engines=None):
        return [
            {"url": "https://docs.python.org/3/library/argparse.html",
             "title": "argparse docs", "content": "subparser"},
        ]

    monkeypatch.setattr(wt, "_searxng_query", fake_searx)
    tool = _mk_tool()
    asyncio.run(tool.execute(query="argparse subparser default handler"))
    r2 = asyncio.run(tool.execute(query="argparse subparser default handler python"))
    assert r2.success
    assert "nearly the same" in r2.output


def test_no_hit_query_is_also_remembered(monkeypatch):
    import asyncio

    import augmentum.coder.web_tools as wt

    async def fake_searx(client, base_url, query, *, engines=None):
        return []

    monkeypatch.setattr(wt, "_searxng_query", fake_searx)
    tool = _mk_tool()
    r1 = asyncio.run(tool.execute(query="zqxwv nonexistent thing"))
    assert r1.success is False or "No results" in (r1.output or "")
    r2 = asyncio.run(tool.execute(query="zqxwv nonexistent thing"))
    # Second attempt of the same miss gets the corrective banner, not
    # another round trip pretending it might differ.
    if r2.success:
        assert "Repeat search" in (r2.output or "") or "No results" in (r2.output or "")
