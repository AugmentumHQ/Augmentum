"""Resolve "whatever the user pasted" into a pollable RSS/Atom feed.

The feed_watch kind (standing_tasks.py) polls a feed URL — but users
don't think in feed URLs. They think "watch this YouTuber", "follow
r/LocalLLaMA", "tell me when this blog posts". This module turns those
into concrete feeds:

  * YouTube — channel page / @handle / /c/ / /user/ / watch URL →
    ``youtube.com/feeds/videos.xml?channel_id=UC…`` (public, keyless).
    Playlist URLs → ``…?playlist_id=…``.
  * Reddit — ``r/name`` or a subreddit URL → ``reddit.com/r/name/.rss``.
  * Direct feed URLs — anything that already parses as RSS/Atom.
  * Any other page — HTML feed autodiscovery
    (``<link rel="alternate" type="application/rss+xml|atom+xml">``).

Every resolution ends with a validation fetch of the candidate feed, so
a successful result is a feed we know we can poll — the resolver returns
its title and latest entry as proof the user can react to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_UA = "Augmentum/1.0 (feed-watch)"
_FETCH_TIMEOUT = 15.0


@dataclass(slots=True)
class ResolvedFeed:
    ok: bool
    feed_url: str = ""
    label: str = ""
    source_kind: str = ""       # youtube | reddit | feed | site
    latest_title: str = ""
    error: str = ""
    entries: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "feed_url": self.feed_url, "label": self.label,
            "source_kind": self.source_kind,
            "latest_title": self.latest_title,
            "entries": self.entries, "error": self.error,
        }


# ─── Feed parsing (shared with the feed_watch runner) ───────────────────


_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)


def _strip_cdata(text: str) -> str:
    m = _CDATA_RE.search(text)
    return (m.group(1) if m else text).strip()


def _tag_text(block: str, tag: str) -> str:
    """First <tag>…</tag> text in a feed entry block (namespace + CDATA
    tolerant). Empty string when absent."""
    m = re.search(
        rf"<(?:\w+:)?{tag}(?:\s[^>]*)?>(.*?)</(?:\w+:)?{tag}>",
        block, re.DOTALL | re.IGNORECASE,
    )
    return _strip_cdata(m.group(1)) if m else ""


def _entry_link(block: str) -> str:
    """Entry link: Atom ``<link href="…">`` (prefer rel=alternate),
    falling back to RSS ``<link>text</link>``."""
    best = ""
    for m in re.finditer(r"<link\b([^>]*?)/?>", block, re.IGNORECASE):
        attrs = m.group(1)
        href_m = re.search(r'href="([^"]+)"', attrs)
        if not href_m:
            continue
        rel_m = re.search(r'rel="([^"]+)"', attrs)
        rel = (rel_m.group(1) if rel_m else "alternate").lower()
        if rel == "alternate":
            return href_m.group(1).strip()
        if not best:
            best = href_m.group(1).strip()
    if best:
        return best
    text_link = _tag_text(block, "link")
    return text_link if text_link.startswith(("http://", "https://")) else ""


def parse_feed(body: str, *, max_entries: int = 25) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom into ``[{id, title, url, published}]``,
    newest-first as the feed orders them. Regex-based on purpose — the
    substrate must not gain an XML dependency, and real-world feeds are
    messy enough that a strict parser fails more often than this does.
    Returns [] for non-feed bodies (that's the validation signal)."""
    entries: list[dict[str, str]] = []
    # Atom <entry> first (YouTube), then RSS <item>.
    blocks = re.findall(r"<entry\b.*?</entry>", body, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"<item\b.*?</item>", body, re.DOTALL | re.IGNORECASE)
    for block in blocks[:max_entries]:
        title = _tag_text(block, "title")
        url = _entry_link(block)
        # Stable identity: Atom <id> / RSS <guid>, else the link, else
        # the title — SOMETHING so the last-seen baseline can work.
        entry_id = (
            _tag_text(block, "id") or _tag_text(block, "guid")
            or url or title
        )
        if not entry_id:
            continue
        published = (
            _tag_text(block, "published") or _tag_text(block, "pubDate")
            or _tag_text(block, "updated")
        )
        entries.append({
            "id": entry_id, "title": title, "url": url,
            "published": published,
        })
    return entries


def feed_title(body: str) -> str:
    """The feed's own <title> (channel/blog/podcast name)."""
    # First title OUTSIDE any entry/item: take the first title in the
    # document — feeds put theirs before any entries.
    m = re.search(
        r"<(?:\w+:)?title(?:\s[^>]*)?>(.*?)</(?:\w+:)?title>",
        body, re.DOTALL | re.IGNORECASE,
    )
    return _strip_cdata(m.group(1))[:120] if m else ""


# ─── Source-shape detection ─────────────────────────────────────────────


_YT_CHANNEL_ID_RE = re.compile(r'"channelId"\s*:\s*"(UC[0-9A-Za-z_-]{10,})"')
_YT_CHANNEL_URL_RE = re.compile(r"youtube\.com/channel/(UC[0-9A-Za-z_-]{10,})")
_YT_PLAYLIST_RE = re.compile(r"[?&]list=([0-9A-Za-z_-]{10,})")
_SUBREDDIT_RE = re.compile(r"^(?:https?://(?:www\.|old\.)?reddit\.com/)?r/([A-Za-z0-9_]{2,50})/?", re.IGNORECASE)
_AUTODISCOVER_RE = re.compile(
    r'<link\b[^>]*type="application/(?:rss|atom)\+xml"[^>]*>',
    re.IGNORECASE,
)


def _yt_feed_for_channel(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


def _looks_like_youtube(raw: str) -> bool:
    return (
        "youtube.com" in raw or "youtu.be" in raw
        or raw.startswith("@")
    )


async def _fetch_text(http_client, url: str) -> str | None:
    try:
        resp = await http_client.get(
            url, timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        log.info("feed_resolve_fetch_failed", url=url[:120], error=str(exc)[:120])
        return None


async def _validate_feed(http_client, feed_url: str, source_kind: str,
                         label_hint: str = "") -> ResolvedFeed:
    body = await _fetch_text(http_client, feed_url)
    if body is None:
        return ResolvedFeed(
            ok=False, source_kind=source_kind,
            error="couldn't fetch the feed",
        )
    entries = parse_feed(body)
    if not entries:
        return ResolvedFeed(
            ok=False, source_kind=source_kind,
            error="that address didn't return a readable feed",
        )
    return ResolvedFeed(
        ok=True, feed_url=feed_url,
        label=(label_hint or feed_title(body) or feed_url)[:120],
        source_kind=source_kind,
        latest_title=entries[0].get("title", "")[:160],
        entries=len(entries),
    )


async def resolve_feed_source(http_client, raw: str) -> ResolvedFeed:
    """Turn user input into a validated feed. See module docstring."""
    raw = (raw or "").strip()
    if not raw:
        return ResolvedFeed(ok=False, error="empty source")

    # Subreddit — r/name or a reddit URL.
    sub = _SUBREDDIT_RE.match(raw)
    if sub and ("reddit.com" in raw.lower() or raw.lower().startswith("r/")):
        name = sub.group(1)
        return await _validate_feed(
            http_client, f"https://www.reddit.com/r/{name}/.rss",
            "reddit", label_hint=f"r/{name}",
        )

    # YouTube in any of its shapes.
    if _looks_like_youtube(raw):
        ch = _YT_CHANNEL_URL_RE.search(raw)
        if ch:
            return await _validate_feed(
                http_client, _yt_feed_for_channel(ch.group(1)), "youtube",
            )
        pl = _YT_PLAYLIST_RE.search(raw)
        if pl:
            return await _validate_feed(
                http_client,
                f"https://www.youtube.com/feeds/videos.xml?playlist_id={pl.group(1)}",
                "youtube",
            )
        # Handle / custom / user / video URL → fetch the page and pull
        # the canonical channelId out of the embedded player config.
        page_url = raw
        if raw.startswith("@"):
            page_url = f"https://www.youtube.com/{raw}"
        elif not raw.startswith(("http://", "https://")):
            page_url = f"https://{raw}"
        body = await _fetch_text(http_client, page_url)
        if body:
            m = _YT_CHANNEL_ID_RE.search(body)
            if m:
                return await _validate_feed(
                    http_client, _yt_feed_for_channel(m.group(1)), "youtube",
                )
        return ResolvedFeed(
            ok=False, source_kind="youtube",
            error="couldn't find that YouTube channel — try the full "
                  "channel URL",
        )

    # Everything else needs to at least be a URL.
    url = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
    if not urlparse(url).netloc:
        return ResolvedFeed(ok=False, error="that doesn't look like a URL")

    # Maybe it IS a feed already.
    probe = await _validate_feed(http_client, url, "feed")
    if probe.ok:
        return probe

    # HTML autodiscovery.
    body = await _fetch_text(http_client, url)
    if body is None:
        return ResolvedFeed(ok=False, error="couldn't fetch that page")
    m = _AUTODISCOVER_RE.search(body)
    if m:
        href_m = re.search(r'href="([^"]+)"', m.group(0))
        if href_m:
            feed_url = urljoin(url, href_m.group(1))
            return await _validate_feed(http_client, feed_url, "site")
    return ResolvedFeed(
        ok=False, source_kind="site",
        error="no feed found on that page — paste the RSS/Atom URL "
              "directly if you know it",
    )


__all__ = ["ResolvedFeed", "resolve_feed_source", "parse_feed", "feed_title"]
