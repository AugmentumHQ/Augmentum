"""feed_watch kind + feed_resolve tests — the "follow a creator" class:
YouTube channels, podcasts, blogs, subreddits, raw RSS/Atom."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.companion_runtime.feed_resolve import (
    feed_title,
    parse_feed,
    resolve_feed_source,
)

# Trimmed real-shape samples.
_YT_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns="http://www.w3.org/2005/Atom">
 <title>Some Creator</title>
 <entry>
  <id>yt:video:abc123DEF45</id>
  <yt:videoId>abc123DEF45</yt:videoId>
  <title>Newest Video!</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=abc123DEF45"/>
  <published>2026-07-01T15:00:00+00:00</published>
 </entry>
 <entry>
  <id>yt:video:old999XYZ01</id>
  <title>Older Video</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=old999XYZ01"/>
  <published>2026-06-20T15:00:00+00:00</published>
 </entry>
</feed>"""

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
 <title>Some Blog</title>
 <item>
  <title><![CDATA[Post Two]]></title>
  <link>https://blog.example.com/post-two</link>
  <guid>https://blog.example.com/post-two</guid>
  <pubDate>Wed, 01 Jul 2026 10:00:00 GMT</pubDate>
 </item>
 <item>
  <title>Post One</title>
  <link>https://blog.example.com/post-one</link>
  <guid>https://blog.example.com/post-one</guid>
 </item>
</channel></rss>"""


def _resp(text: str, status: int = 200):
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock(
        side_effect=None if status == 200 else RuntimeError("http error"),
    )
    return r


def _client(routes: dict[str, str]):
    """Fake httpx client: url-substring → body."""
    async def _get(url, **_kw):
        for frag, body in routes.items():
            if frag in url:
                return _resp(body)
        raise RuntimeError(f"unexpected fetch: {url}")
    c = MagicMock()
    c.get = AsyncMock(side_effect=_get)
    return c


# ── parse_feed ──────────────────────────────────────────────────────────


def test_parse_feed_atom_youtube_shape():
    entries = parse_feed(_YT_ATOM)
    assert len(entries) == 2
    assert entries[0]["id"] == "yt:video:abc123DEF45"
    assert entries[0]["title"] == "Newest Video!"
    assert entries[0]["url"] == "https://www.youtube.com/watch?v=abc123DEF45"
    assert feed_title(_YT_ATOM) == "Some Creator"


def test_parse_feed_rss_with_cdata():
    entries = parse_feed(_RSS)
    assert len(entries) == 2
    assert entries[0]["title"] == "Post Two"
    assert entries[0]["url"] == "https://blog.example.com/post-two"
    assert feed_title(_RSS) == "Some Blog"


def test_parse_feed_non_feed_returns_empty():
    assert parse_feed("<html><body>not a feed</body></html>") == []


# ── resolve_feed_source ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_youtube_channel_url_direct():
    client = _client({"feeds/videos.xml?channel_id=UCabcdefghij": _YT_ATOM})
    r = await resolve_feed_source(
        client, "https://www.youtube.com/channel/UCabcdefghij",
    )
    assert r.ok and r.source_kind == "youtube"
    assert "channel_id=UCabcdefghij" in r.feed_url
    assert r.label == "Some Creator"
    assert r.latest_title == "Newest Video!"


@pytest.mark.asyncio
async def test_resolve_youtube_handle_via_page_scrape():
    page = '<html>… "channelId":"UChandle12345" …</html>'
    client = _client({
        "youtube.com/@somecreator": page,
        "feeds/videos.xml?channel_id=UChandle12345": _YT_ATOM,
    })
    r = await resolve_feed_source(client, "@somecreator")
    assert r.ok and "UChandle12345" in r.feed_url


@pytest.mark.asyncio
async def test_resolve_subreddit_shorthand():
    client = _client({"reddit.com/r/LocalLLaMA/.rss": _RSS})
    r = await resolve_feed_source(client, "r/LocalLLaMA")
    assert r.ok and r.source_kind == "reddit"
    assert r.label == "r/LocalLLaMA"


@pytest.mark.asyncio
async def test_resolve_site_autodiscovery():
    html = ('<html><head><link rel="alternate" '
            'type="application/rss+xml" href="/feed.xml"></head></html>')
    client = _client({
        "blog.example.com/feed.xml": _RSS,
        "blog.example.com": html,
    })
    r = await resolve_feed_source(client, "https://blog.example.com")
    assert r.ok and r.feed_url == "https://blog.example.com/feed.xml"


@pytest.mark.asyncio
async def test_resolve_direct_feed_url():
    client = _client({"blog.example.com/rss": _RSS})
    r = await resolve_feed_source(client, "https://blog.example.com/rss")
    assert r.ok and r.label == "Some Blog"


@pytest.mark.asyncio
async def test_resolve_failure_is_conversational():
    client = _client({})  # everything 404s
    r = await resolve_feed_source(client, "https://nope.example.com/x")
    assert not r.ok and r.error


# ── feed_watch runner ───────────────────────────────────────────────────


def _runtime_with_feed(body: str):
    runtime = MagicMock()
    client = _client({"": body})  # match any url
    runtime._app_state = MagicMock(http_client=client)
    return runtime


@pytest.mark.asyncio
async def test_feed_watch_first_poll_baselines_silently():
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    runner = _TASK_KINDS["feed_watch"]
    params = {"feed_url": "https://yt.example/feed", "source_label": "Some Creator"}
    result = await runner(_runtime_with_feed(_YT_ATOM), user_id="u1", params=params)
    assert result["noteworthy"] is False
    assert "following" in result["summary"]
    update = result["details"]["params_update"]
    assert update["last_seen_id"] == "yt:video:abc123DEF45"


@pytest.mark.asyncio
async def test_feed_watch_surfaces_new_entries_and_advances_cursor():
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    runner = _TASK_KINDS["feed_watch"]
    params = {
        "feed_url": "https://yt.example/feed",
        "source_label": "Some Creator",
        "last_seen_id": "yt:video:old999XYZ01",  # one entry is newer
    }
    result = await runner(_runtime_with_feed(_YT_ATOM), user_id="u1", params=params)
    assert result["noteworthy"] is True
    assert "1 new post" in result["summary"]
    assert "Newest Video!" in result["details"]["content"]
    assert result["refs"] and result["refs"][0]["url"].endswith("abc123DEF45")
    assert result["details"]["params_update"]["last_seen_id"] == "yt:video:abc123DEF45"


@pytest.mark.asyncio
async def test_feed_watch_nothing_new_is_silent():
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    runner = _TASK_KINDS["feed_watch"]
    params = {
        "feed_url": "https://yt.example/feed",
        "last_seen_id": "yt:video:abc123DEF45",  # already newest
    }
    result = await runner(_runtime_with_feed(_YT_ATOM), user_id="u1", params=params)
    assert result["noteworthy"] is False
    assert "nothing new" in result["summary"]


@pytest.mark.asyncio
async def test_feed_watch_bad_params_raise():
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    runner = _TASK_KINDS["feed_watch"]
    with pytest.raises(ValueError):
        await runner(MagicMock(), user_id="u1", params={"feed_url": "ftp://x"})
