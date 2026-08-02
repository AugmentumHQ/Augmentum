"""js13k browse + bundle provider.

js13kgames.com runs an annual JS-game contest where every entry must fit
inside a 13KB zip. All submissions are MIT-licensed by the contest rules,
which means redistribution is explicitly permitted -- we can legitimately
download the bundle and serve it from our own origin.

**Catalog source.** The authoritative catalog is the
`js13kGames/games <https://github.com/js13kGames/games>`_ GitHub repo,
which stores every submission as ``games/{slug}/`` with a standard shape:

- ``index.html`` — game entrypoint (often standalone with inline JS)
- ``.t.png`` / ``.c.png`` — thumbnail / cover art (conventionally named)
- ``README.md`` — optional frontmatter with ``directors_cut`` / ``post``
  / ``video`` links and a longform description
- ``.src/`` — optional original uncompressed source directory

**Bundle source.** ``play.js13kgames.com/{slug}.zip`` serves a pre-built
zip (13–40 KB) containing the exact files the game ships with. We fetch,
extract, and store these at pin time — no per-user fetch against
js13k's Cloudflare Worker on every Play.

**Rate limits.** The unauthenticated GitHub API is 60 requests/hour, IP-
pooled. We page the catalog lazily (one API call per browse page) and
cache 24 h so a typical self-hosted deployment never hits the ceiling.
If we ever need headroom, adding a ``GITHUB_TOKEN`` env var lifts it to
5000/hour.

**Sort semantics.** GitHub's directory listing is alphabetical and that
is the only sort we can cheaply offer across the full 14 000-entry
catalog. ``newest`` maps to alphabetical for API-shape parity with other
providers; ``popular`` maps to the same. The UI hides the sort toggle
when the active source is js13k.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from typing import Any
from urllib.parse import urlparse

from augmentum.games.models import GameBrowseResult
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient  # re-exported for API parity

log = get_logger(__name__)

# GitHub API pagination uses ``?per_page=N&page=N``. GitHub enforces a
# 100-entry hard cap regardless of what we send for ``per_page``.
_CATALOG_API = (
    "https://api.github.com/repos/js13kGames/games/contents/games"
)
_CATALOG_PAGE_SIZE = 100

# Raw content URLs. Images work in ``<img>`` tags despite raw GitHub's
# ``x-frame-options: deny`` — XFO only gates iframe/frame loads. Bundle
# downloads use the play.js13kgames.com zip endpoint, which also serves
# with CORS open for server-side fetches.
_RAW_BASE = "https://raw.githubusercontent.com/js13kGames/games/main/games"
_ZIP_BASE = "https://play.js13kgames.com"

# Game-page URL on js13kgames.com (used as ``source_url`` for the "open
# source page" link in the game surface header). The full play URL at
# ``play.js13kgames.com`` is set as ``embed_url`` even though we don't
# actually embed it — it's the canonical offsite playable link for
# users who prefer the hosted version.
_PAGE_BASE = "https://js13kgames.com/games"
_PLAY_BASE = "https://play.js13kgames.com"

# Binary-file extensions we always base64-encode when packing the bundle
# into ``source_json`` (which is a TEXT column). Anything not matching is
# stored as UTF-8 text, which preserves readability in logs/audit while
# still round-tripping cleanly.
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".mp3", ".ogg", ".wav", ".m4a",
    ".ttf", ".otf", ".woff", ".woff2",
    ".zip", ".gz", ".wasm",
}


def _humanize_slug(slug: str) -> str:
    """Turn ``13-curses-of-the-sea`` into ``13 Curses Of The Sea``.

    The GitHub catalog stores slugs but no cleartext titles. README
    frontmatter doesn't carry the title either (it's in the submission
    form which doesn't publish). A slug-to-title transform is the best
    cheap approximation we have for the browse strip; the pin path
    later overrides with the README ``<h1>`` or ``<title>`` tag when one
    exists.
    """
    if not slug:
        return "Untitled"
    words = re.split(r"[-_]", slug)
    return " ".join(w.capitalize() if not w.isupper() else w for w in words if w)


def _thumb_url(slug: str) -> str:
    # Convention across the repo varies by contest year:
    #   - Older years: ``.t.png`` (+ ``.c.png``, ``.cs.png``, ``.ts.png``)
    #   - Newer years: ``.t.jpg`` (+ ``.c.jpg``)
    # We return the ``.t.png`` guess as the primary src; the frontend's
    # ``onerror`` cascade walks through the alternate extensions before
    # falling through to the emoji placeholder. Probing here would burn
    # a GitHub API call per game and we only get 60/hour unauthenticated.
    return f"{_RAW_BASE}/{slug}/.t.png"


def _thumb_candidates(slug: str) -> list[str]:
    """All plausible thumbnail URLs for a js13k game, in preference order.

    The frontend uses this list to drive an ``<img onerror>`` fallback
    chain so the right one loads regardless of which contest year the
    game came from. Ordered most-likely-present first.
    """
    return [
        f"{_RAW_BASE}/{slug}/.t.png",
        f"{_RAW_BASE}/{slug}/.t.jpg",
        f"{_RAW_BASE}/{slug}/.c.png",
        f"{_RAW_BASE}/{slug}/.c.jpg",
    ]


def _source_page(slug: str) -> str:
    return f"{_PAGE_BASE}/{slug}"


async def _github_fetch(url: str, timeout: float = 15.0) -> dict | list | None:
    """Fetch a GitHub API URL with curl_cffi (Chrome TLS) and parse JSON.

    GitHub doesn't gate unauthenticated requests behind Cloudflare
    challenges, but we route through curl_cffi anyway for consistency
    with other providers and because the library's TLS fingerprint is
    friendlier to rate-limit-adjacent API calls.
    """
    try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(
            impersonate="chrome131",
            max_redirects=5,
            timeout=timeout,
        ) as session:
            response = await session.get(url, allow_redirects=True)
            if response.status_code == 403:
                # Rate limit hit — log prominently so support can see it.
                log.warning(
                    "js13k_github_rate_limited",
                    remaining=response.headers.get("x-ratelimit-remaining"),
                    reset=response.headers.get("x-ratelimit-reset"),
                    url=url,
                )
                return None
            if response.status_code != 200:
                log.warning(
                    "js13k_github_non_200",
                    status=response.status_code,
                    url=url,
                )
                return None
            return json.loads(response.text)
    except ImportError:
        log.warning("js13k_curl_cffi_unavailable")
        return None
    except Exception as exc:
        log.warning("js13k_github_fetch_failed", url=url, error=str(exc))
        return None


async def browse(
    sort: str,
    page: int,
    safe_client: SafeHttpClient,
    *,
    page_size: int = _CATALOG_PAGE_SIZE,
) -> list[GameBrowseResult]:
    """Fetch one page of the js13k catalog from GitHub.

    Returns an empty list on any upstream failure. ``sort`` is accepted
    for API parity with other providers but has no effect — GitHub's
    directory listing is alphabetical and that's the only order we can
    offer without a much heavier precomputed index.
    """
    url = f"{_CATALOG_API}?per_page={page_size}&page={max(1, page)}"
    data = await _github_fetch(url)
    if not isinstance(data, list):
        return []

    results: list[GameBrowseResult] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "dir":
            continue
        slug = str(entry.get("name") or "").strip()
        if not slug or slug.startswith("."):
            continue
        results.append(
            GameBrowseResult(
                source="js13k",
                source_id=slug,
                name=_humanize_slug(slug),
                author="",  # Not available without fetching README; lazy-enrich.
                tagline="",
                thumbnail_url=_thumb_url(slug),
                source_url=_source_page(slug),
                embed_url=f"{_PLAY_BASE}/{slug}/",
                play_mode="local",
                # Alternative URLs the ``<img onerror>`` cascade walks
                # through when the default .t.png 404s. Kept in ``extra``
                # so the base model stays provider-agnostic.
                extra={"slug": slug, "thumbnail_candidates": _thumb_candidates(slug)},
            )
        )
    return results


# ─── Details fetch ───────────────────────────────────────────────────
#
# README.md is the only structured metadata source per game. We parse
# the frontmatter (if any) for links to the author's site/video, and
# the first paragraph of body text for a tagline. Title comes from the
# first ``# Heading``; if absent, we humanize the slug.

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+([^\n]+)", re.MULTILINE)


def _parse_readme(body: str) -> dict[str, Any]:
    """Extract title, description, and link frontmatter from a README."""
    out: dict[str, Any] = {
        "title": "",
        "description": "",
        "directors_cut": "",
        "post_url": "",
        "video_url": "",
    }
    if not body:
        return out

    # Frontmatter block — simple key: value pairs; skip anything nested.
    fm_match = _FRONTMATTER_RE.match(body)
    rest = body
    if fm_match:
        for line in fm_match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "directors_cut":
                out["directors_cut"] = val
            elif key == "post" and val.startswith(("http://", "https://")):
                out["post_url"] = val
            elif key == "video":
                out["video_url"] = val
        rest = body[fm_match.end():]

    # Title: first H1 in the body.
    h1 = _H1_RE.search(rest)
    if h1:
        out["title"] = h1.group(1).strip()

    # Description: first paragraph of plain body text after the H1 (if any).
    paragraphs = [p.strip() for p in rest.split("\n\n") if p.strip()]
    for p in paragraphs:
        # Skip heading-only paragraphs.
        if p.startswith("#"):
            continue
        # Strip basic markdown emphasis/link syntax for a readable one-liner.
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", p)
        clean = re.sub(r"[*_`]", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean:
            out["description"] = clean[:600]
            break

    return out


async def fetch_details(
    source_id: str,
    safe_client: SafeHttpClient,
) -> dict[str, Any]:
    """Fetch a js13k game's README + compute the bundle size estimate."""
    readme_url = f"{_RAW_BASE}/{source_id}/README.md"
    try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(impersonate="chrome131", timeout=15.0) as session:
            resp = await session.get(readme_url, allow_redirects=True)
            readme_body = resp.text if resp.status_code == 200 else ""
    except Exception as exc:
        log.warning("js13k_readme_fetch_failed", slug=source_id, error=str(exc))
        readme_body = ""

    parsed = _parse_readme(readme_body)
    title = parsed["title"] or _humanize_slug(source_id)

    return {
        "ok": True,
        "source": "js13k",
        "source_id": source_id,
        "source_url": _source_page(source_id),
        "name": title,
        "description": parsed["description"],
        "cover_url": f"{_RAW_BASE}/{source_id}/.c.png",
        "thumbnail_url": _thumb_url(source_id),
        "thumbnail_candidates": _thumb_candidates(source_id),
        "author_name": "",
        "author_url": "",
        "genre": [],
        "rating_value": 0.0,
        "rating_count": 0,
        "date_published": "",
        "date_modified": "",
        "platforms": ["HTML5"],
        # js13k games are keyboard/mouse for the overwhelming majority;
        # marking touchscreen would require per-game HTML inspection.
        # Leave empty to trigger the "controls unlisted" chip path --
        # honest-unknown rather than a fabricated desktop-only claim.
        "inputs": [],
        "mobile_friendly": False,
        # No embed_src for js13k — play.js13kgames.com sends
        # ``frame-ancestors`` CSP that blocks third-party framing. The
        # game runs locally instead; ``fetch_bundle`` below supplies the
        # files we serve from our own origin.
        "embed_src": "",
        "directors_cut_url": parsed["directors_cut"],
        "post_url": parsed["post_url"],
        "video_url": parsed["video_url"],
    }


# ─── Bundle fetch + pack ─────────────────────────────────────────────
#
# At pin time we download the game's zip, extract all files, and pack
# them into a JSON structure suitable for the ``artifacts.source_json``
# TEXT column. Binary files are base64-encoded and tagged with
# ``encoding: "base64"`` so the frontend's assembler knows to re-hydrate
# before inlining.


def _guess_encoding(path: str, data: bytes) -> str:
    """Return ``'text'`` or ``'base64'`` based on extension + content sniff."""
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in _BINARY_EXTENSIONS:
        return "base64"
    # Sniff: if there are null bytes or >10% high bytes, treat as binary.
    if b"\x00" in data[:2048]:
        return "base64"
    try:
        data.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "base64"


# Refuse to unpack bundles that exceed this size. 13 KB zipped expands
# to ~500 KB worst case; 2 MB gives comfortable headroom while stopping
# a malicious redirect that swaps the zip for something huge.
_MAX_BUNDLE_BYTES = 2 * 1024 * 1024


async def fetch_bundle(source_id: str) -> dict[str, Any]:
    """Download and unpack a js13k game's zip into a pack-ready payload.

    Returns ``{"ok": True, "files": [{path, content, encoding}],
    "size_bytes": N, "single_file": bool, "entry": "index.html" | "..."}``.
    On any failure returns ``{"ok": False, "reason": "..."}``.
    """
    if not source_id or "/" in source_id or ".." in source_id:
        return {"ok": False, "reason": "invalid_source_id"}

    zip_url = f"{_ZIP_BASE}/{source_id}.zip"
    try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(impersonate="chrome131", timeout=30.0) as session:
            resp = await session.get(zip_url, allow_redirects=True)
            if resp.status_code != 200:
                log.warning(
                    "js13k_zip_fetch_non_200",
                    slug=source_id,
                    status=resp.status_code,
                )
                return {"ok": False, "reason": f"upstream_{resp.status_code}"}
            body = resp.content
    except Exception as exc:
        log.warning("js13k_zip_fetch_failed", slug=source_id, error=str(exc))
        return {"ok": False, "reason": "fetch_failed"}

    if len(body) > _MAX_BUNDLE_BYTES:
        return {"ok": False, "reason": "bundle_too_large"}

    files: list[dict[str, Any]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(body))
        total_unpacked = 0
        for name in zf.namelist():
            # Zip-slip guard: reject absolute paths and parent-traversal
            # sequences before the file goes anywhere near storage.
            if name.startswith("/") or ".." in name.split("/"):
                log.warning("js13k_zip_unsafe_path", slug=source_id, path=name)
                continue
            # Skip directory entries and .src/ originals (we ship the
            # built output, not the source tree).
            if name.endswith("/"):
                continue
            if name.startswith(".src/") or "/.src/" in name:
                continue
            try:
                data = zf.read(name)
            except Exception as exc:
                log.warning(
                    "js13k_zip_entry_read_failed",
                    slug=source_id,
                    path=name,
                    error=str(exc),
                )
                continue
            total_unpacked += len(data)
            if total_unpacked > _MAX_BUNDLE_BYTES:
                return {"ok": False, "reason": "bundle_unpacked_too_large"}
            encoding = _guess_encoding(name, data)
            content = (
                base64.b64encode(data).decode("ascii")
                if encoding == "base64"
                else data.decode("utf-8")
            )
            files.append({"path": name, "content": content, "encoding": encoding})
    except zipfile.BadZipFile:
        log.warning("js13k_bad_zip", slug=source_id)
        return {"ok": False, "reason": "bad_zip"}

    if not files:
        return {"ok": False, "reason": "empty_bundle"}

    # Find the entry point. index.html at the root is the overwhelming
    # convention; a rare game nests one subdirectory deep.
    entry = ""
    for f in files:
        if f["path"] == "index.html":
            entry = "index.html"
            break
    if not entry:
        # Fall back to the first .html file.
        for f in files:
            if f["path"].lower().endswith(".html"):
                entry = f["path"]
                break

    return {
        "ok": True,
        "files": files,
        "size_bytes": len(body),
        "unpacked_bytes": total_unpacked,
        "single_file": len(files) == 1,
        "entry": entry,
    }
