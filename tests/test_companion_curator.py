"""Curator unit tests — topic CRUD, scoring, composition, dedup, step gating."""

from __future__ import annotations

import pytest


async def _fresh_backend(user_id: str = "usr_c") -> tuple:
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (user_id, "tester", "x"),
    )
    await backend.conn.commit()
    return backend


# ── Store CRUD ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_topic_bare_phrase():
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    row = await curator.add_topic(
        backend.conn, user_id="usr_c", companion_id="becca",
        topic="rust async runtime",
    )
    assert row is not None
    assert row.topic == "rust async runtime"
    assert row.feed_url is None
    assert row.feed_kind is None


@pytest.mark.asyncio
async def test_add_topic_url_input_extracts_feed():
    """A URL pasted into the topic field becomes a feed_url with a
    host-derived display topic — single-input UX contract."""
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    row = await curator.add_topic(
        backend.conn, user_id="usr_c", companion_id="becca",
        topic="https://drewdevault.com/feed.xml",
    )
    assert row is not None
    assert row.feed_url == "https://drewdevault.com/feed.xml"
    assert row.feed_kind == "rss"
    assert row.topic == "drewdevault.com"


@pytest.mark.asyncio
async def test_add_topic_explicit_feed_url():
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    row = await curator.add_topic(
        backend.conn, user_id="usr_c", companion_id="becca",
        topic="anthropic blog", feed_url="https://www.anthropic.com/news/rss",
    )
    assert row is not None
    assert row.topic == "anthropic blog"
    assert row.feed_url == "https://www.anthropic.com/news/rss"
    assert row.feed_kind == "rss"


@pytest.mark.asyncio
async def test_add_topic_duplicate_returns_none():
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    first = await curator.add_topic(
        backend.conn, user_id="usr_c", companion_id="becca", topic="x",
    )
    assert first is not None
    dup = await curator.add_topic(
        backend.conn, user_id="usr_c", companion_id="becca", topic="x",
    )
    assert dup is None


@pytest.mark.asyncio
async def test_list_and_remove_topics():
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    for t in ("alpha", "beta", "gamma"):
        await curator.add_topic(
            backend.conn, user_id="usr_c", companion_id="becca", topic=t,
        )
    topics = await curator.list_topics(
        backend.conn, user_id="usr_c", companion_id="becca",
    )
    assert {t.topic for t in topics} == {"alpha", "beta", "gamma"}

    target = next(t for t in topics if t.topic == "beta")
    removed = await curator.remove_topic(
        backend.conn, topic_id=target.id,
        user_id="usr_c", companion_id="becca",
    )
    assert removed is True
    remaining = await curator.list_topics(
        backend.conn, user_id="usr_c", companion_id="becca",
    )
    assert {t.topic for t in remaining} == {"alpha", "gamma"}


@pytest.mark.asyncio
async def test_remove_topic_wrong_user_no_op():
    """Another user's topic_id must not be removable — tenant isolation."""
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend("usr_owner")
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_attacker", "att", "x"),
    )
    await backend.conn.commit()
    row = await curator.add_topic(
        backend.conn, user_id="usr_owner", companion_id="becca",
        topic="private interest",
    )
    assert row is not None
    removed = await curator.remove_topic(
        backend.conn, topic_id=row.id,
        user_id="usr_attacker", companion_id="becca",
    )
    assert removed is False


# ── Relevance scoring ────────────────────────────────────────────────


def test_score_relevance_overlap():
    from augmentum.companion_runtime import curator
    item = {
        "title": "Rust async runtime comparison: tokio vs async-std",
        "snippet": "A look at how the two compete on throughput.",
    }
    score = curator.score_relevance(item, "rust async runtime")
    # All three topic tokens (rust, async, runtime) appear in title.
    assert score == pytest.approx(1.0)


def test_score_relevance_partial():
    from augmentum.companion_runtime import curator
    item = {
        "title": "Tokio async runtime v1.40 released",
        "snippet": "New features in this release.",
    }
    score = curator.score_relevance(item, "rust async runtime")
    # "async" + "runtime" appear, "rust" doesn't → 2/3.
    assert score == pytest.approx(2 / 3)


def test_score_relevance_single_shared_token_zero():
    """≥2-token floor (2026-06-12): one shared word on a multi-token
    topic is never an "obvious connection" — the Blacksmith-audiobook /
    "Life before main()" pairing matched on just "life"."""
    from augmentum.companion_runtime import curator
    item = {
        "title": "Tokio v1.40 released",
        "snippet": "New async features in this release.",
    }
    # Only "async" overlaps with "rust async runtime" → floored to 0.
    assert curator.score_relevance(item, "rust async runtime") == 0.0


def test_score_relevance_no_overlap_zero():
    from augmentum.companion_runtime import curator
    item = {
        "title": "Bread baking sourdough notes",
        "snippet": "Hydration ratios and bulk ferment timings.",
    }
    score = curator.score_relevance(item, "rust async runtime")
    assert score == 0.0


# ── Note composition + dedup hash ─────────────────────────────────────


def test_compose_note_shape_and_url_ref():
    from augmentum.companion_runtime import curator
    item = {
        "title": "Continual learning via task-conditional adapters",
        "snippet": "We propose a new way to stack LoRAs for sequential tasks.",
        "url": "https://arxiv.org/abs/2511.01234",
    }
    content, refs = curator.compose_note("persona kernel", item)
    # Two-line shape: topic on its own line, then title — snippet.
    assert content.startswith("persona kernel\n")
    assert "Continual learning" in content
    assert "stack LoRAs" in content
    # No poem-y opener.
    assert not content.lower().startswith("the weight of")
    # URL ref present with stable hash for dedup.
    assert len(refs) == 1
    assert refs[0]["kind"] == "url"
    assert refs[0]["url"] == "https://arxiv.org/abs/2511.01234"
    assert len(refs[0]["id"]) == 16  # truncated sha256


def test_compose_note_no_url_no_ref():
    from augmentum.companion_runtime import curator
    content, refs = curator.compose_note("x", {"title": "y", "snippet": ""})
    assert refs == []
    assert content


def test_compose_note_ref_carries_domain_title_snippet():
    """Drawer renders a clickable article preview (domain badge + title
    + snippet) from the ref alone — without parsing the composed prose.
    Older notes fall back to prose-parsing in the UI; new notes carry
    the full payload here."""
    from augmentum.companion_runtime import curator
    item = {
        "title": "Continual learning via task-conditional adapters",
        "snippet": "We propose a new way to stack LoRAs for sequential tasks.",
        "url": "https://www.arxiv.org/abs/2511.01234",
    }
    _content, refs = curator.compose_note("persona kernel", item)
    assert len(refs) == 1
    r = refs[0]
    # www. stripped — the badge reads as "arxiv.org" not "www.arxiv.org".
    assert r["domain"] == "arxiv.org"
    assert r["title"] == "Continual learning via task-conditional adapters"
    assert r["snippet"].startswith("We propose")
    assert r["url"] == "https://www.arxiv.org/abs/2511.01234"


# ── Curator low-value content filter (2026-06-05) ───────────────────
#
# Pins the named-failure modes from the 2026-06-04 drawer audit, where
# the curator was writing notes for dictionary definitions and 403
# placeholder pages because they shared a single token with the
# cluster name and passed the coherence floor.


def test_low_value_blocks_merriam_webster_definition():
    """Drawer #7283: cluster='World News Today' → title='INTRODUCTION
    Definition & Meaning - Merriam-Webster'. Domain hits the dictionary
    blocklist before the title pattern even runs."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {
        "domain": "merriam-webster.com",
        "url": "https://www.merriam-webster.com/dictionary/introduction",
        "title": "INTRODUCTION Definition & Meaning - Merriam-Webster",
    }
    reason = _curator_low_value_block(rec)
    assert reason is not None
    assert reason.startswith("low_value_domain:")


def test_low_value_blocks_dictionary_definition_title_on_neutral_domain():
    """Defense in depth — if the rec comes from a domain not on the
    explicit dictionary blocklist but the title still reads as a
    definition page, reject on the title pattern."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {
        "domain": "wordsmith.example.com",
        "url": "https://wordsmith.example.com/word/check",
        "title": "CHECK Definition & Meaning - Wordsmith",
    }
    reason = _curator_low_value_block(rec)
    assert reason is not None
    assert reason.startswith("low_value_title:")


def test_low_value_blocks_transcript_landing_page():
    """Drawer #7282: cluster='Google Drive Sign-in' → title='Transcript
    of Google Drive introduction video'. NSW teacher PD transcript;
    semantically off-cluster despite shared 'google drive' tokens."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {
        "domain": "education.nsw.gov.au",
        "title": "Transcript of 'Google Drive introduction' video",
    }
    assert _curator_low_value_block(rec) == "low_value_title:^\\s*transcript\\s+of\\b"


def test_low_value_blocks_reddit_403_placeholder():
    """Drawer #7285: title='Reddit — We would like to show you a
    description here but the site won't allow us.' This is Reddit's
    generic 403 placeholder, not a post."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {
        "domain": "reddit.com",
        "title": "Reddit — We would like to show you a description here but the site won't allow us.",
    }
    reason = _curator_low_value_block(rec)
    assert reason is not None and reason.startswith("low_value_title:")


def test_low_value_blocks_signin_landing_page():
    from augmentum.companion_runtime.curator import _curator_low_value_block
    for title in (
        "Sign-in to Google Drive",
        "Sign In to GitHub",
        "Log in to Reddit",
        "Login to Notion",
    ):
        rec = {"domain": "example.com", "title": title}
        reason = _curator_low_value_block(rec)
        assert reason is not None and reason.startswith("low_value_title:"), \
            f"Failed to reject login landing: {title!r}"


def test_low_value_blocks_what_is_definitional_title():
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {"domain": "tech-explained.example", "title": "What is artificial intelligence?"}
    reason = _curator_low_value_block(rec)
    assert reason is not None and reason.startswith("low_value_title:")


def test_low_value_passes_real_article():
    """Negative case — real articles must NOT trip the low-value block."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    for rec in (
        {"domain": "news.ycombinator.com", "title": "Grok 4 release thread"},
        {"domain": "kdnuggets.com", "title": "Self-Hosted AI Models: A Practical Guide"},
        {"domain": "arxiv.org", "title": "Continual learning via task-conditional adapters"},
        # Substring "check" in title doesn't trigger the definition regex —
        # the regex requires "definition & meaning" right after the word.
        {"domain": "example.com", "title": "Building a fact-check pipeline for LLMs"},
    ):
        assert _curator_low_value_block(rec) is None, \
            f"False positive on real article: {rec['title']!r}"


def test_low_value_strips_www_prefix_on_domain():
    """www. prefix on the rec's domain must not let a blocked domain slip."""
    from augmentum.companion_runtime.curator import _curator_low_value_block
    rec = {"domain": "www.merriam-webster.com", "title": "Anything"}
    assert _curator_low_value_block(rec) == "low_value_domain:merriam-webster.com"


# ── Stopword expansion regression ────────────────────────────────────


def test_stopwords_include_high_traffic_verbs():
    """Drawer #7283-class: cluster 'Bitcoin Price Check' vs title
    'CHECK Definition & Meaning' overlapped on 'check' alone. Adding
    'check' / 'introduction' / 'definition' to the stopword set drops
    coherence on those overlaps to zero, complementing the low-value
    block as a defense-in-depth signal-quality fix."""
    from augmentum.companion_runtime import curator
    # Every named regression-driving verb is now a stopword.
    for word in (
        "check", "see", "watch", "find", "open", "sign",
        "introduction", "definition", "meaning", "transcript",
    ):
        assert word in curator._STOPWORDS, f"Missing stopword: {word!r}"


def test_tokens_helper_drops_stopwords():
    from augmentum.companion_runtime.curator import _tokens
    # "Bitcoin Price Check" — only 'bitcoin' and 'price' should survive
    # the new stopword filter.
    toks = _tokens("Bitcoin Price Check · Seattle")
    assert "check" not in toks
    assert "bitcoin" in toks and "price" in toks


def test_coherence_drops_to_zero_when_only_shared_token_is_stopword():
    """The load-bearing regression — cluster + title share ONLY a
    high-traffic verb. With the verb stopworded, coherence is 0.0 →
    the rec fails the 0.15 floor in _curator_quality_pass."""
    from augmentum.companion_runtime.curator import score_relevance
    item = {
        "title": "CHECK Definition & Meaning - Merriam-Webster",
        "snippet": "The meaning of CHECK is to inspect, examine, or look at appraisingly.",
    }
    # Only shared distinctive token would have been 'check'; now stopworded.
    coherence = score_relevance(item, "Bitcoin Price Check")
    assert coherence == 0.0


def test_compose_note_from_rec_ref_carries_zone_and_cluster():
    """For-You picks additionally carry zone + cluster_name so the UI
    can render the 'On X' / 'Adjacent to X' framing as a structured
    badge rather than re-parsing the composed prose."""
    from augmentum.companion_runtime.curator import compose_note_from_rec
    rec = {
        "title": "Self-Hosted AI Models",
        "snippet": "How to run LLMs locally in 2026.",
        "url": "https://kdnuggets.com/self-hosted-llms-real-world",
        "cluster_name": "self hosted llms",
        "zone": "core",
    }
    _body, refs = compose_note_from_rec(rec)
    assert len(refs) == 1
    r = refs[0]
    assert r["domain"] == "kdnuggets.com"
    assert r["title"] == "Self-Hosted AI Models"
    assert r["zone"] == "core"
    assert r["cluster_name"] == "self hosted llms"


# ── Dedup check against companion_journal ─────────────────────────────


@pytest.mark.asyncio
async def test_seen_url_recently_detects_prior_note():
    """A URL the curator wrote about within the dedup window must be
    recognized — sanity check the content_refs LIKE-match path."""
    from augmentum.companion_runtime import curator
    backend = await _fresh_backend()
    import json
    url = "https://example.com/article-x"
    h = curator._url_hash(url)
    # Insert a journal row referencing this URL — mimics what compose_note
    # + safe_journal would persist.
    await backend.conn.execute(
        """INSERT INTO companion_journal
           (companion_id, user_id, entry_type, content,
            affect_tag, content_refs, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        ("becca", "usr_c", "curator_note", "topic\ntitle — snippet",
         "curious", json.dumps([{"kind": "url", "id": h, "url": url}]),
         "curator"),
    )
    await backend.conn.commit()

    seen = await curator._seen_url_recently(
        backend.conn, user_id="usr_c", companion_id="becca", url=url,
    )
    assert seen is True

    not_seen = await curator._seen_url_recently(
        backend.conn, user_id="usr_c", companion_id="becca",
        url="https://example.com/different",
    )
    assert not_seen is False


# ── Step gating ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_disabled_returns_none(monkeypatch):
    """Master kill switch off → no work, no exceptions."""
    from augmentum.companion_runtime import curator
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_curator_enabled", False)
    backend = await _fresh_backend()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()
    rt.owner_user_id = "usr_c"

    result = await curator.step(rt)
    assert result is None


# ── Safety filter (2026-06-04 curator_safety_pass) ───────────────────
#
# Regression pins for the four notes that escaped to the journal
# before the safety/quality/framing patch landed. Each test names the
# real failure mode that produced the bad note so a future relax of
# the filter has to consciously break a named test, not a silent one.


def test_safety_blocks_adult_tube_domain():
    """#7280 — straight pornhub.com domain."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {"domain": "pornhub.com", "url": "https://pornhub.com/videos"}
    assert _curator_safety_block(rec, "porn videos") is not None


def test_safety_blocks_adult_toy_retailer_with_explicit_path():
    """#7289 — edenfantasys.com /sex-toy/ page reached For-You via a
    SearXNG search that bled into adult retail results. Both the
    domain and the URL path independently trip the filter."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {
        "domain": "edenfantasys.com",
        "url": "https://edenfantasys.com/sex-toys/masturbator/happy-cup/",
        "title": "Happy cup pussy & mouth masturbator",
    }
    reason = _curator_safety_block(rec, "")
    assert reason is not None
    # Domain hits first (deterministic ordering matters for the
    # quarantine_reason audit trail).
    assert reason.startswith("adult_domain:")


def test_safety_blocks_adult_url_path_on_neutral_domain():
    """Hypothetical generic CDN serving an /adult/ path — the domain
    is innocuous but the path is not."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {
        "domain": "cdn.example.com",
        "url": "https://cdn.example.com/adult/preview/123.mp4",
    }
    assert _curator_safety_block(rec, "") == "adult_url_path:/adult/"


def test_safety_blocks_adult_tld():
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {"domain": "something.xxx", "url": "https://something.xxx/"}
    assert _curator_safety_block(rec, "") == "adult_suffix:.xxx"


def test_safety_strips_leading_www():
    """www. prefix must not let a blocked domain slip through."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {"domain": "www.pornhub.com", "url": "https://www.pornhub.com/"}
    assert _curator_safety_block(rec, "") == "adult_domain:pornhub.com"


def test_safety_blocks_nsfw_token_in_cluster_name():
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {"domain": "example.com", "url": "https://example.com/"}
    assert _curator_safety_block(rec, "hentai recommendations") == "nsfw_cluster_name"


def test_safety_nsfw_token_match_is_whole_word_only():
    """'naked truth' / 'sex education' must NOT trip the NSFW filter.
    Whole-word match is the load-bearing property — a substring match
    would over-block legitimate health/education clusters."""
    from augmentum.companion_runtime.curator import _cluster_name_is_nsfw
    assert _cluster_name_is_nsfw("naked truth") is False
    assert _cluster_name_is_nsfw("sex education curriculum") is False
    assert _cluster_name_is_nsfw("cocktail recipes") is False
    # But the unambiguous tokens DO match.
    assert _cluster_name_is_nsfw("blowjob videos") is True
    assert _cluster_name_is_nsfw("dildo reviews") is True


def test_safety_blocks_nsfw_token_in_rec_title_innocent_cluster():
    """The cluster name is innocent but the surfaced rec's title is
    explicit. Discovery's SearXNG firehose can do this — block on
    title as well, not just cluster."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {
        "domain": "blogplatform.example",
        "url": "https://blogplatform.example/p/123",
        "title": "Reviewing the latest fleshlight model",
    }
    assert _curator_safety_block(rec, "tech reviews") == "nsfw_rec_title"


def test_safety_passes_normal_tech_rec():
    """The negative case — a normal recommendation must pass through."""
    from augmentum.companion_runtime.curator import _curator_safety_block
    rec = {
        "domain": "github.com",
        "url": "https://github.com/anthropics/anthropic-sdk-python",
        "title": "Anthropic SDK for Python",
    }
    assert _curator_safety_block(rec, "python sdks") is None


# ── Quality filter ───────────────────────────────────────────────────


def test_quality_rejects_low_score():
    from augmentum.companion_runtime.curator import _curator_quality_pass
    rec = {"_score": 0.1, "title": "Python tutorial", "zone": "core"}
    reason = _curator_quality_pass(rec, "python")
    assert reason is not None and reason.startswith("score_below_floor:")


def test_quality_rejects_incoherent_match():
    """#7286-class — cluster name 'self hosted llms' but the rec is
    about coinkit Bitcoin wordlists. Token overlap is ~0; coherence
    floor catches it."""
    from augmentum.companion_runtime.curator import _curator_quality_pass
    rec = {
        "_score": 0.8,
        "title": "coinkit/words.py at master · mflaxman/coinkit",
        "snippet": "Cryptocurrency wallet interfaces for Bitcoin",
        "zone": "core",
    }
    reason = _curator_quality_pass(rec, "self hosted llms in the real world")
    assert reason is not None and reason.startswith("low_coherence:")


def test_quality_skips_coherence_for_fresh_zone():
    """Fresh-zone recs come from external feeds with no cluster_name —
    coherence check is N/A for them."""
    from augmentum.companion_runtime.curator import _curator_quality_pass
    rec = {
        "_score": 0.7,
        "title": "Totally unrelated arxiv paper",
        "zone": "fresh",
    }
    # Cluster name passed but should be skipped because zone=fresh.
    assert _curator_quality_pass(rec, "python sdks") is None


# ── Framing sanitization ─────────────────────────────────────────────


def test_framing_rejects_unbalanced_parens():
    """#7291 — 'Future - Feds Did a Sweep (Official Music' is mid-phrase.
    Returning None makes compose_note_from_rec fall back to a generic
    framing instead of writing a mangled sentence."""
    from augmentum.companion_runtime.curator import _sanitize_cluster_name_for_framing
    assert _sanitize_cluster_name_for_framing(
        "Future - Feds Did a Sweep (Official Music",
    ) is None


def test_framing_rejects_trailing_dash():
    from augmentum.companion_runtime.curator import _sanitize_cluster_name_for_framing
    # Trailing dash gets stripped, then the remainder is inspected.
    result = _sanitize_cluster_name_for_framing("python programming -")
    assert result == "python programming"


def test_framing_rejects_all_caps():
    from augmentum.companion_runtime.curator import _sanitize_cluster_name_for_framing
    assert _sanitize_cluster_name_for_framing("BREAKING NEWS HEADLINE") is None


def test_framing_rejects_too_short():
    from augmentum.companion_runtime.curator import _sanitize_cluster_name_for_framing
    assert _sanitize_cluster_name_for_framing("a") is None
    assert _sanitize_cluster_name_for_framing("") is None


def test_framing_accepts_clean_cluster_name():
    from augmentum.companion_runtime.curator import _sanitize_cluster_name_for_framing
    assert _sanitize_cluster_name_for_framing(
        "self hosted llms",
    ) == "self hosted llms"


def test_compose_note_falls_back_when_cluster_name_mangled():
    """End-to-end: when sanitization rejects the cluster name, the
    note still composes with a generic-but-honest framing rather
    than dropping the rec or writing a broken sentence."""
    from augmentum.companion_runtime.curator import compose_note_from_rec
    rec = {
        "title": "Some interesting article",
        "url": "https://example.com/article",
        "snippet": "A snippet here",
        "cluster_name": "Future - Feds Did a Sweep (Official Music",
        "zone": "core",
    }
    body, _refs = compose_note_from_rec(rec)
    assert "Future - Feds" not in body  # mangled name not in framing
    assert "Some interesting article" in body
