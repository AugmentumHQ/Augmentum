"""Image search tool — finds relevant images via SearXNG and stores as artifacts.

Downloaded images are saved through the ArtifactStore (scoped to the
agentic task), NOT the image gallery (image_generations table).  This
keeps web-sourced reference images separate from user-generated artwork.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.tools.artifact_storage import ArtifactStore

log = get_logger(__name__)

# Minimum keyword overlap between query and result title to consider relevant
_MIN_RELEVANCE_WORDS = 1

# Domains to skip (tracking pixels, icons, logos, ads)
_SKIP_DOMAINS = frozenset({
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "pinterest.com", "tiktok.com", "youtube.com",
    "google.com", "gstatic.com", "googleusercontent.com",
    "gravatar.com", "wp.com", "cloudfront.net",
})

# File extensions we'll accept
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"})


class ImageSearchTool(Tool):
    """Search for images on the web and download the best match.

    Results are stored as task-scoped artifacts, separate from the
    image generation gallery.  Returns an artifact download URL that
    DocumentTool and PresentationTool can embed directly.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        artifact_store: ArtifactStore | None = None,
        searxng_url: str = "",
    ) -> None:
        self._client = http_client
        self._store = artifact_store
        self._searxng_url = searxng_url or settings.searxng_base_url

    @property
    def name(self) -> str:
        return "image_search"

    @property
    def description(self) -> str:
        return (
            "Search for an existing image on the web. Downloads and stores it locally. "
            "Use specific queries like 'global temperature anomaly chart NOAA' "
            "not vague terms like 'climate change picture'. "
            "For creating NEW images from scratch, use image_generation instead."
        )

    @property
    def category(self) -> ToolCategory:
        # IMAGE (not SEARCH): image_search writes to user-scoped artifact
        # tables, so chain.py:489 needs to recognize it for user_id /
        # _context injection. The comment at chain.py:496 already names
        # image_search as an IMAGE tool — this aligns the category with
        # that contract.
        return ToolCategory.IMAGE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No images found": "Try more descriptive terms or add context. Include the subject and style: 'temperature anomaly chart NOAA 2024' not just 'climate chart'.",
            "download failed": "The image URL was unreachable. Try a different search query to find alternative sources.",
        }

    @property
    def requires_services(self) -> list[str]:
        return ["searxng"]

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Descriptive search query for the image. Be specific: "
                        "'sea level rise projection chart' not 'ocean image'"
                    ),
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "Number of images to return (default: 4, max: 6). "
                        "Prefer 4–6 for gallery-style display unless the "
                        "user explicitly asked for a single image."
                    ),
                    "default": 4,
                    "minimum": 1,
                    "maximum": 6,
                },
                "prefer_charts": {
                    "type": "boolean",
                    "description": "Prefer charts, graphs, and diagrams over photos (default: false)",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 50.0

    async def execute(
        self,
        *,
        query: str = "",
        count: int = 4,
        prefer_charts: bool = False,
        task_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not query:
            return ToolResult(success=False, error="No search query provided")

        count = max(1, min(count, 6))

        # Enhance query for chart preference
        search_query = query
        if prefer_charts and not any(
            w in query.lower() for w in ("chart", "graph", "diagram", "figure", "plot", "infographic")
        ):
            search_query = f"{query} chart OR graph OR diagram"

        # Search SearXNG images (over-fetch for pipeline headroom)
        candidates = await self._search_images(search_query, max_results=count * 5)

        if not candidates:
            return ToolResult(
                success=False,
                error=f"No images found for '{query}'",
            )

        # Quality pipeline: reject junk domains, noise, score by relevance
        try:
            from augmentum.discovery.quality import filter_for_images
            relevant = filter_for_images(candidates, query, prefer_charts=prefer_charts)
        except Exception:
            relevant = self._filter_relevant(candidates, query, prefer_charts)

        if not relevant:
            relevant = candidates
        # Keep a generous pool — downloads fail often (404, hotlinks), and we
        # want the gallery filled even if half the candidates die.
        pool = relevant if len(relevant) >= count else (relevant + [c for c in candidates if c not in relevant])

        # Cross-round dedup (spec: turn_search_dedup) — skip images already shown
        # to the user earlier this turn so repeated rounds don't re-surface the
        # same picture; the over-fetched pool backfills with fresh ones. Marked
        # only on successful download (a failed fetch stays retryable next round).
        from augmentum.tools.turn_search_dedup import get_turn_dedup
        dedup = get_turn_dedup()

        # Download until we hit `count`, tolerating individual failures.
        results = []
        skipped_dup = 0
        for candidate in pool:
            if len(results) >= count:
                break
            cand_url = candidate.get("url", "")
            if dedup is not None and dedup.seen("image", cand_url):
                skipped_dup += 1
                continue
            stored = await self._download_and_store(
                candidate, task_id=task_id, session_id=session_id, query=query, **kwargs,
            )
            if stored:
                results.append(stored)
                if dedup is not None:
                    dedup.mark("image", cand_url)

        if not results:
            msg = f"Found {len(candidates)} images but failed to download any"
            if skipped_dup:
                msg = (f"All matching images were already shown to the user earlier "
                       f"this turn ({skipped_dup} skipped). Don't search for the same "
                       f"subject again — answer with what's already displayed.")
            return ToolResult(success=False, error=msg)

        # Format output — phrased so the model knows the images are already
        # displayed to the user, rather than trying to reinvent URLs in prose.
        count_word = "image" if len(results) == 1 else "images"
        lines = [
            f"Found {len(results)} {count_word} for '{query}' — already displayed to the user inline.",
            "",
            "Results:",
        ]
        for r in results:
            title = r["title"] or "(untitled)"
            lines.append(f"- {title}  [source: {r['source']}]")
        lines.extend([
            "",
            "The user can see the images above. Briefly acknowledge what was found "
            "(e.g. the subject and source) without restating URLs.",
        ])
        if skipped_dup:
            lines.append(
                f"(Note: {skipped_dup} matching image(s) were already shown earlier "
                f"this turn and omitted. Avoid re-searching the same subject.)"
            )

        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(
            success=True,
            output=wrap_untrusted("web/image_search", "\n".join(lines)),
            metadata={
                "images": results,
                "embed_urls": [r["embed_url"] for r in results],
            },
        )

    # ------------------------------------------------------------------
    # SearXNG image search
    # ------------------------------------------------------------------

    async def _search_images(
        self, query: str, max_results: int = 10,
    ) -> list[dict]:
        """Query SearXNG for images. Returns list of {title, url, source, thumbnail}."""
        try:
            resp = await self._client.get(
                f"{self._searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "images",
                    "safesearch": "1",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("image_search_failed", query=query, error=str(e))
            return []

        results = []
        seen_urls: set[str] = set()

        for item in data.get("results", []):
            img_url = item.get("img_src") or item.get("url", "")
            if not img_url or img_url in seen_urls:
                continue

            # Skip unwanted domains
            domain = _extract_domain(img_url)
            if domain in _SKIP_DOMAINS:
                continue

            # Check file extension
            path_lower = img_url.split("?")[0].lower()
            has_ext = any(path_lower.endswith(ext) for ext in _IMAGE_EXTENSIONS)
            # Some image URLs don't have extensions (CDN URLs) — allow them
            # but prefer those with known extensions
            if not has_ext and "." in path_lower.split("/")[-1]:
                # Has an extension but not an image one — skip
                ext = "." + path_lower.split("/")[-1].rsplit(".", 1)[-1]
                if ext not in _IMAGE_EXTENSIONS and len(ext) < 6:
                    continue

            seen_urls.add(img_url)
            results.append({
                "title": item.get("title", ""),
                "url": img_url,
                "source": item.get("source", domain),
                "thumbnail": item.get("thumbnail_src", ""),
                "resolution": item.get("resolution", ""),
            })

            if len(results) >= max_results:
                break

        log.info("image_search_results", query=query, count=len(results))
        return results

    # ------------------------------------------------------------------
    # Relevance filtering
    # ------------------------------------------------------------------

    def _filter_relevant(
        self,
        candidates: list[dict],
        query: str,
        prefer_charts: bool,
    ) -> list[dict]:
        """Score and filter candidates by relevance to the query."""
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        scored: list[tuple[float, dict]] = []

        for c in candidates:
            title_words = set(re.findall(r"\b\w{3,}\b", c["title"].lower()))
            source_words = set(re.findall(r"\b\w{3,}\b", c["source"].lower()))
            all_words = title_words | source_words

            # Keyword overlap score
            overlap = len(query_words & all_words)
            if overlap < _MIN_RELEVANCE_WORDS:
                continue

            score = overlap / max(len(query_words), 1)

            # Boost charts/diagrams when preferred
            if prefer_charts:
                chart_words = {"chart", "graph", "diagram", "figure", "plot",
                               "data", "statistics", "infographic", "visualization"}
                if all_words & chart_words:
                    score += 0.3

            # Boost higher resolution images
            res = c.get("resolution", "")
            if res:
                try:
                    parts = re.split(r"[x×]", res)
                    if len(parts) == 2:
                        w, h = int(parts[0].strip()), int(parts[1].strip())
                        if w >= 800 and h >= 600:
                            score += 0.2
                        elif w < 200 or h < 200:
                            score -= 0.3  # penalize tiny images
                except (ValueError, IndexError):
                    pass

            # Penalize stock photo sites slightly (generic images)
            stock_domains = {"shutterstock", "istockphoto", "gettyimages",
                             "dreamstime", "depositphotos", "123rf", "alamy"}
            if any(d in c["source"].lower() for d in stock_domains):
                score -= 0.2

            scored.append((score, c))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    # ------------------------------------------------------------------
    # Download and artifact storage
    # ------------------------------------------------------------------

    async def _download_and_store(
        self,
        candidate: dict,
        *,
        task_id: str,
        session_id: str,
        query: str,
        **kwargs,
    ) -> dict | None:
        """Download an image, compress to a WebP thumbnail, store as artifact.

        We keep the ORIGINAL source URL in artifact metadata so the UI can
        link out to full-res on click; locally we only keep a small WebP
        (≤1024px long edge, quality 82) to prevent disk blow-up.  A single
        high-res JPEG can be 8MB+ but users only ever view 96×96 thumbs.
        """
        url = candidate["url"]

        try:
            # Browser-like headers: image CDNs (news sites especially)
            # hotlink-protect aggressively — a bot UA with no Referer gets
            # blanket 403s ("Found 15 images but failed to download any",
            # 2026-07-07 briefing hero-image case). The Referer points at
            # the image's own host, which is what a same-site embed sends.
            dl_headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
            source_domain = (candidate.get("source") or "").strip()
            if source_domain:
                dl_headers["Referer"] = f"https://{source_domain}/"
            resp = await self._client.get(
                url,
                timeout=15.0,
                follow_redirects=True,
                headers=dl_headers,
            )
            resp.raise_for_status()
            raw_data = resp.content

            if len(raw_data) < 1000:
                log.debug("image_too_small", url=url, size=len(raw_data))
                return None

            # Compress to WebP thumbnail. Preserve originals only when PIL
            # can't decode (SVG, rare formats, broken files).
            data, ext = _compress_to_thumbnail(raw_data, url)
            original_size = len(raw_data)

            # Store as artifact (NOT in image_generations gallery)
            if self._store:
                safe_query = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:40]
                filename = f"img_{safe_query}_{uuid.uuid4().hex[:6]}{ext}"

                info = await self._store.save(
                    data=data,
                    filename=filename,
                    fmt=ext.lstrip("."),
                    task_id=task_id,
                    session_id=session_id,
                    display_name=candidate["title"][:100] or filename,
                    metadata={
                        "page_type": "web_image",
                        "source_url": url,
                        "source_domain": candidate["source"],
                        "search_query": query,
                        "original_size_bytes": original_size,
                    },
                    user_id=Tool.extract_user_id(kwargs),
                    transient=True,
                )

                log.info(
                    "image_downloaded",
                    url=url[:100],
                    stored_size=len(data),
                    original_size=original_size,
                    saved_pct=int((1 - len(data) / max(original_size, 1)) * 100),
                    artifact_id=info["id"],
                )

                return {
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "source_url": url,
                    "embed_url": info["download_url"],  # /api/artifacts/{id}/download
                    "download_url": info["download_url"],
                    "artifact_id": info["id"],
                    "size_bytes": len(data),
                }
            else:
                # No artifact store — return the raw URL (can't embed in PDFs)
                return {
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "source_url": url,
                    "embed_url": url,
                    "download_url": url,
                    "artifact_id": "",
                    "size_bytes": len(data),
                }

        except Exception as e:
            log.warning("image_download_failed", url=url[:100], error=str(e), exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract domain from a URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _ext_from_content_type(ct: str) -> str:
    """Map content-type to file extension."""
    ct = ct.split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
    }
    return mapping.get(ct, "")


def _ext_from_url(url: str) -> str:
    """Extract file extension from URL path."""
    path = url.split("?")[0].lower()
    for ext in _IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ""


# Max long-edge in pixels for stored thumbnail. The gallery renders at 96×96
# and the lightbox falls back to the source URL for full-res, so 1024 is plenty.
_THUMB_MAX_EDGE = 1024
_THUMB_WEBP_QUALITY = 82


def _compress_to_thumbnail(raw: bytes, url: str) -> tuple[bytes, str]:
    """Resize + re-encode as WebP. Returns (bytes, extension).

    Falls back to the original bytes + inferred extension when PIL can't
    decode (SVG, animated GIFs we don't want to touch, broken downloads).
    """
    try:
        import io

        from PIL import Image
    except Exception:
        return raw, _ext_from_url(url) or ".jpg"

    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        # Animated GIFs — keep as-is so animation survives
        if getattr(im, "is_animated", False):
            return raw, ".gif"
        # SVG / unsupported — fall back
        if im.format in (None, "SVG"):
            return raw, _ext_from_url(url) or ".jpg"
        # Convert palette/RGBA to RGB for WebP compatibility when no alpha
        if im.mode in ("P", "RGBA", "LA"):
            has_alpha = "A" in im.mode or (im.mode == "P" and "transparency" in im.info)
            if not has_alpha:
                im = im.convert("RGB")
        elif im.mode != "RGB" and "A" not in im.mode:
            im = im.convert("RGB")
        # Resize preserving aspect
        long_edge = max(im.width, im.height)
        if long_edge > _THUMB_MAX_EDGE:
            scale = _THUMB_MAX_EDGE / long_edge
            new_size = (int(im.width * scale), int(im.height * scale))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="WEBP", quality=_THUMB_WEBP_QUALITY, method=4)
        data = out.getvalue()
        # Only use the thumbnail if it actually saves space
        if len(data) >= len(raw):
            return raw, _ext_from_url(url) or ".jpg"
        return data, ".webp"
    except Exception:
        return raw, _ext_from_url(url) or ".jpg"
