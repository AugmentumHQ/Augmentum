"""Live TV rail categorizer.

Takes ``CatalogItem(kind='live_video')`` rows from Emby/Jellyfin (via
:meth:`augmentum.media.providers.emby_compat.EmbyCompatBase.fetch_live_channels`)
and groups them into the YouTube-TV-style rails the UI renders:

    Favorites · Recently Watched · Live News · Live Sports ·
    Movies & Shows · Music · Kids · Local OTA · All Channels

Multi-rail membership is allowed — a channel airing news right now
shows up in *Live News* even if its network would normally classify
elsewhere. Rail order is fixed; empty rails are dropped (so a brand-
new install with no favorites just shows the thematic rails it can
fill).

Within each rail channels sort by parsed channel number (so ``6.2``
sits between ``6`` and ``7``), then alphabetically on name as a
tiebreak. Unparseable numbers go to the end of the rail.

Categorization is signal-driven, in this order:

  1. EPG ``current_program`` flags (``is_news``, ``is_sports``,
     ``is_movie``, ``is_series``, ``is_kids``) — most authoritative.
  2. Conservative hand-curated network-name hints (substring,
     case-insensitive) — fills gaps for channels whose EPG doesn't
     populate those flags. Lists are intentionally short; the goal
     is obvious rail membership, not exhaustive coverage.

The categorizer is pure — no I/O, no provider calls, no caching. The
route layer feeds it pre-fetched ``CatalogItem`` lists (potentially
aggregated across multiple Emby/JF servers per user) and ships the
result back to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from augmentum.media.providers.base import CatalogItem


# ── Network-name hints ────────────────────────────────────────────
# Conservative on purpose — each list short enough to scan, every
# entry a recognizable network identifier. If a channel doesn't
# match a list AND its EPG doesn't flag the program, it lands only
# in *All Channels* — that's the right outcome (the user can still
# find it, just not in a thematic rail it doesn't belong in).
_NEWS_HINTS = (
    "news", "cnn", "msnbc", "fox news", "newsmax", "bbc world",
    "bbcw", "bbc news", "sky news", "al jazeera", "nhk world",
    "dw english", "france 24", "rt ", "bloomberg", "cnbc",
    "newsnation", "cbsn", "abc news", "nbc news", "weather channel",
)

_SPORTS_HINTS = (
    "espn", "sec network", "acc network", "big ten", "fs1", "fs2",
    "fox sports", "sportsnet", "tsn", "nbc sports", "cbs sports",
    "mlb network", "nfl network", "nba tv", "nhl network",
    "golf channel", "tennis channel", "olympic",
)

_MOVIES_HINTS = (
    "amc", "tnt", "tbs", "fx", "fxx", "usa network", "tlc", "hgtv",
    "bravo", "ifc", "sundance", "starz", "showtime", "cinemax",
    "epix", "movies", "tcm", "syfy", "comedy central",
)

_MUSIC_HINTS = (
    "mtv", "vh1", "cmt", "bet jams", "music choice", "fuse",
    "axs tv", "the country network", "rev'n",
)

_KIDS_HINTS = (
    "nickelodeon", "nick jr", "cartoon network", "disney",
    "disney xd", "disney junior", "boomerang", "discovery kids",
    "pbs kids", "universal kids",
)


# ── Public types ──────────────────────────────────────────────────

@dataclass(slots=True)
class Rail:
    """One categorized rail surfaced to the UI."""

    id: str                       # stable slug ('favorites', 'news', …)
    title: str                    # human-readable rail header
    kind: str                     # 'favorites'|'recent'|'thematic'|'ota'|'all'
    channels: list[CatalogItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "title":    self.title,
            "kind":     self.kind,
            "channels": [_channel_to_dict(c) for c in self.channels],
        }


# ── Public entry point ────────────────────────────────────────────

def categorize_channels(channels: Iterable[CatalogItem]) -> list[Rail]:
    """Group live channels into ordered rails. Empty rails are dropped."""
    chans = [c for c in channels if c.kind == "live_video"]
    if not chans:
        return []

    rails: list[Rail] = []

    favs = [c for c in chans if _is_favorite(c)]
    if favs:
        rails.append(Rail(
            id="favorites", title="Favorites", kind="favorites",
            channels=sorted(favs, key=_chan_sort_key),
        ))

    recent = [c for c in chans if _play_count(c) > 0]
    if recent:
        rails.append(Rail(
            id="recent", title="Recently Watched", kind="recent",
            channels=sorted(recent, key=lambda c: (-_play_count(c), _chan_sort_key(c))),
        ))

    # Thematic rails — order matches the UI's expected scan order.
    for rail_id, title, hints, prog_keys in (
        ("news",   "Live News",      _NEWS_HINTS,   ("is_news",)),
        ("sports", "Live Sports",    _SPORTS_HINTS, ("is_sports",)),
        ("movies", "Movies & Shows", _MOVIES_HINTS, ("is_movie", "is_series")),
        ("music",  "Music",          _MUSIC_HINTS,  ()),
        ("kids",   "Kids",           _KIDS_HINTS,   ("is_kids",)),
    ):
        matches = [
            c for c in chans
            if _matches_hints(c, hints) or _current_program_flags(c, prog_keys)
        ]
        if matches:
            rails.append(Rail(
                id=rail_id, title=title, kind="thematic",
                channels=sorted(matches, key=_chan_sort_key),
            ))

    # Local OTA: ATSC sub-channels expose a dotted number (e.g. 6.1,
    # 13.2). A bare integer is usually a cable/Internet channel —
    # those don't qualify here even if the user happens to have OTA
    # configured upstream.
    ota = [c for c in chans if "." in _channel_number(c)]
    if ota:
        rails.append(Rail(
            id="ota", title="Local OTA", kind="ota",
            channels=sorted(ota, key=_chan_sort_key),
        ))

    rails.append(Rail(
        id="all", title="All Channels", kind="all",
        channels=sorted(chans, key=_chan_sort_key),
    ))

    return rails


# ── Internal helpers ──────────────────────────────────────────────

def _extra(c: CatalogItem) -> dict:
    return c.extra if isinstance(c.extra, dict) else {}


def _is_favorite(c: CatalogItem) -> bool:
    return bool(_extra(c).get("is_favorite", False))


def _play_count(c: CatalogItem) -> int:
    try:
        return int(_extra(c).get("play_count", 0) or 0)
    except (ValueError, TypeError):
        return 0


def _channel_number(c: CatalogItem) -> str:
    return str(_extra(c).get("channel_number", "") or "").strip()


def _matches_hints(c: CatalogItem, hints: tuple[str, ...]) -> bool:
    name = c.name.lower()
    return any(hint in name for hint in hints)


def _current_program_flags(c: CatalogItem, keys: tuple[str, ...]) -> bool:
    cp = _extra(c).get("current_program") or {}
    if not isinstance(cp, dict):
        return False
    return any(bool(cp.get(k)) for k in keys)


# Channels with no parseable number sort to the end. We use a large
# sentinel rather than ``None`` so the tuple ordering stays total.
_NO_NUMBER_SENTINEL = 10**9


def _chan_sort_key(c: CatalogItem) -> tuple[int, int, str]:
    """``(major, minor, lowercased name)`` so ``6.2`` slots between
    ``6`` and ``7`` and ties resolve alphabetically."""
    num = _channel_number(c)
    if not num:
        return (_NO_NUMBER_SENTINEL, 0, c.name.lower())
    parts = num.split(".", 1)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (_NO_NUMBER_SENTINEL, 0, c.name.lower())
    minor = 0
    if len(parts) == 2:
        try:
            minor = int(parts[1])
        except ValueError:
            minor = 0
    return (major, minor, c.name.lower())


def _channel_to_dict(c: CatalogItem) -> dict:
    """JSON shape the UI consumes. Preserves the fields the rail tile
    needs to render (name, number, current program, logo variants)
    plus any caller-injected ``server_id`` so the play path knows
    which Emby/JF server to round-trip back to."""
    extra = _extra(c)
    return {
        "external_id":      c.external_id,
        "name":             c.name,
        "cover_url":        c.cover_url,
        "channel_number":   str(extra.get("channel_number") or ""),
        "source_provider":  str(extra.get("source_provider") or ""),
        "server_id":        str(extra.get("server_id") or ""),
        "is_favorite":      bool(extra.get("is_favorite") or False),
        "play_count":       int(extra.get("play_count") or 0),
        "has_logo_primary": bool(extra.get("has_logo_primary") or False),
        "has_logo_light":   bool(extra.get("has_logo_light") or False),
        "has_logo_dark":    bool(extra.get("has_logo_dark") or False),
        "current_program":  extra.get("current_program"),
    }
