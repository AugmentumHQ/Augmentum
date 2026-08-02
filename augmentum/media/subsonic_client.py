"""Async Subsonic API client — music server integration.

Implements the Subsonic REST API (v1.16.1) over httpx. Used by the
``provider_bridge`` hook to register Navidrome/Jellyfin/Airsonic as
music providers, and by ``media.play`` to search and stream.

Auth: token-based — md5(password + random salt) — passed as query
params on every request. All responses are JSON (``f=json``).

The client is protocol-compatible with every Subsonic-speaking server:
Navidrome, Airsonic, Gonic, LMS (with Subsonic plugin), and Jellyfin
(with the Subsonic plugin).
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field
from typing import Any

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_API_VERSION = "1.16.1"
_CLIENT_NAME = "Augmentum"


@dataclass
class SubsonicArtist:
    id: str = ""
    name: str = ""


@dataclass
class SubsonicAlbum:
    id: str = ""
    name: str = ""
    artist: str = ""
    cover_id: str = ""
    song_count: int = 0
    year: int = 0


@dataclass
class SubsonicSong:
    id: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    album_id: str = ""
    cover_id: str = ""
    duration: int = 0   # seconds
    track: int = 0
    year: int = 0
    content_type: str = ""


@dataclass
class SearchResult:
    albums: list[SubsonicAlbum] = field(default_factory=list)
    songs: list[SubsonicSong] = field(default_factory=list)


class SubsonicClient:
    """Async Subsonic client over a single server URL + credentials."""

    def __init__(self, base_url: str, username: str = "",
                 password: str = "", timeout: float = 15.0) -> None:
        self._base = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._timeout = timeout

    def _auth_params(self) -> dict[str, str]:
        """Build auth query params for the next request."""
        salt = "".join(random.choices(string.ascii_letters + string.digits, k=12))
        token = hashlib.md5((self._password + salt).encode()).hexdigest()  # noqa: S324 — Subsonic spec
        return {
            "u": self._username,
            "t": salt,
            "s": token,
            "v": _API_VERSION,
            "c": _CLIENT_NAME,
            "f": "json",
        }

    def _rest_url(self, endpoint: str, **extra) -> str:
        """Build a full REST URL with auth + extra params."""
        params = self._auth_params()
        params.update({k: str(v) for k, v in extra.items() if v})
        qs = "&".join(f"{k}={_q(v)}" for k, v in params.items())
        return f"{self._base}/rest/{endpoint}?{qs}"

    async def _get(self, endpoint: str, **extra) -> dict[str, Any]:
        url = self._rest_url(endpoint, **extra)
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as c:
            resp = await c.get(url)
            resp.raise_for_status()
            data = resp.json()
        status = data.get("subsonic-response", {}).get("status", "")
        if status != "ok":
            err = data.get("subsonic-response", {}).get("error", {}).get("message", status)
            raise SubsonicError(f"Subsonic error: {err}")
        return data.get("subsonic-response", {})

    async def ping(self) -> bool:
        """Health check. Returns True if the server is reachable + auth works."""
        try:
            await self._get("ping")
            return True
        except Exception:
            return False

    async def search(self, query: str, *, album_count: int = 5,
                     song_count: int = 10) -> SearchResult:
        """Search the library by query string."""
        try:
            resp = await self._get(
                "search3",
                query=query,
                artistCount=0,
                albumCount=album_count,
                songCount=song_count,
            )
        except Exception:
            log.warning("subsonic_search_failed", query=query[:80], exc_info=True)
            return SearchResult()

        sr = resp.get("searchResult3", {}) or {}
        result = SearchResult()
        for a in sr.get("album", []):
            result.albums.append(SubsonicAlbum(
                id=a.get("id", ""), name=a.get("name", ""),
                artist=a.get("artist", ""), cover_id=a.get("coverArt", ""),
                song_count=a.get("songCount", 0), year=a.get("year", 0),
            ))
        for s in sr.get("song", []):
            result.songs.append(SubsonicSong(
                id=s.get("id", ""), title=s.get("title", ""),
                artist=s.get("artist", ""), album=s.get("album", ""),
                album_id=s.get("albumId", ""), cover_id=s.get("coverArt", ""),
                duration=s.get("duration", 0), track=s.get("track", 0),
                year=s.get("year", 0), content_type=s.get("contentType", ""),
            ))
        return result

    async def get_album(self, album_id: str) -> SubsonicAlbum | None:
        """Fetch album details with song list."""
        try:
            resp = await self._get("getAlbum", id=album_id)
        except Exception:
            log.warning("subsonic_get_album_failed", album_id=album_id, exc_info=True)
            return None
        a = resp.get("album", {}) or {}
        album = SubsonicAlbum(
            id=a.get("id", ""), name=a.get("name", ""),
            artist=a.get("artist", ""), cover_id=a.get("coverArt", ""),
            song_count=a.get("songCount", 0), year=a.get("year", 0),
        )
        return album

    async def get_random_songs(self, size: int = 20) -> list[SubsonicSong]:
        """Get random songs from the library."""
        try:
            resp = await self._get("getRandomSongs", size=size)
        except Exception:
            log.warning("subsonic_random_failed", exc_info=True)
            return []
        songs: list[SubsonicSong] = []
        for s in resp.get("randomSongs", {}).get("song", []):
            songs.append(SubsonicSong(
                id=s.get("id", ""), title=s.get("title", ""),
                artist=s.get("artist", ""), album=s.get("album", ""),
                album_id=s.get("albumId", ""), cover_id=s.get("coverArt", ""),
                duration=s.get("duration", 0), track=s.get("track", 0),
                year=s.get("year", 0), content_type=s.get("contentType", ""),
            ))
        return songs

    async def get_starred(self) -> list[SubsonicSong]:
        """Get starred/favorited songs."""
        try:
            resp = await self._get("getStarred2")
        except Exception:
            log.warning("subsonic_starred_failed", exc_info=True)
            return []
        songs: list[SubsonicSong] = []
        for s in resp.get("starred2", {}).get("song", []):
            songs.append(SubsonicSong(
                id=s.get("id", ""), title=s.get("title", ""),
                artist=s.get("artist", ""), album=s.get("album", ""),
                album_id=s.get("albumId", ""), cover_id=s.get("coverArt", ""),
                duration=s.get("duration", 0), track=s.get("track", 0),
                year=s.get("year", 0), content_type=s.get("contentType", ""),
            ))
        return songs

    def stream_url(self, song_id: str) -> str:
        """Return a URL that streams audio for a song.

        The caller proxies this URL or passes it to a media player.
        The URL carries auth params — it's valid for a single request.
        """
        return self._rest_url("stream", id=song_id)

    def cover_art_url(self, cover_id: str, size: int = 300) -> str:
        """Return a URL for cover art."""
        return self._rest_url("getCoverArt", id=cover_id, size=size)


class SubsonicError(Exception):
    """The Subsonic server returned an error response."""


def _q(val: str) -> str:
    """URL-encode a query parameter value. Simple version — httpx handles
    encoding in the URL, so this is only for manually building query strings."""
    import urllib.parse
    return urllib.parse.quote(str(val), safe="")
