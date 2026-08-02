"""EPUB metadata + cover extraction.

EPUBs are ZIP archives containing an OPF manifest. We pull Dublin Core
metadata (title/creator/publisher/language/date/description) and the
cover image referenced by either scheme:

  EPUB 3 — ``<item properties="cover-image" href=.../>``
  EPUB 2 — ``<meta name="cover" content="<manifest-id>"/>``

The cover is piped through Pillow to produce a normalized JPEG data URI
matching the shape of :func:`augmentum.vfs.enrichment._generate_thumbnail`
so it drops straight into ``file_index.thumbnail`` alongside image
thumbnails. Stdlib + Pillow only — no new deps.
"""

from __future__ import annotations

import base64
import io
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree as ET

from augmentum.utils.logging import get_logger
from augmentum.utils.safe_archive import ensure_zip_sane

log = get_logger(__name__)

_CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
_OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass
class EpubMetadata:
    title: str = ""
    author: str = ""
    publisher: str = ""
    language: str = ""
    date: str = ""
    description: str = ""
    cover_thumbnail: str = ""

    def as_source_metadata(self) -> dict:
        """Return the non-empty subset suitable for ``source_metadata`` JSON.

        The cover thumbnail lives on the ``thumbnail`` column, so it's
        intentionally excluded here.
        """
        return {
            k: v
            for k, v in {
                "title": self.title,
                "author": self.author,
                "publisher": self.publisher,
                "language": self.language,
                "date": self.date,
                "description": self.description,
            }.items()
            if v
        }


def extract(path: str, *, max_thumb_px: int = 320) -> EpubMetadata | None:
    """Open an EPUB and return metadata + a normalized cover thumbnail.

    Returns ``None`` if the file isn't a readable EPUB (bad ZIP, missing
    OPF, etc.). Returns a best-effort :class:`EpubMetadata` otherwise —
    individual fields may be empty if the OPF omits them.
    """
    try:
        with zipfile.ZipFile(str(path)) as zf:
            ensure_zip_sane(zf, source="epub_reader")
            opf_path = _find_opf(zf)
            if not opf_path:
                return None
            opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
            opf_root = ET.fromstring(zf.read(opf_path))
            meta = _parse_metadata(opf_root)
            cover_href = _find_cover_href(opf_root)
            if cover_href:
                meta.cover_thumbnail = (
                    _extract_cover_thumbnail(
                        zf, opf_dir + cover_href, max_px=max_thumb_px,
                    )
                    or ""
                )
            return meta
    except zipfile.BadZipFile:
        return None
    except Exception as exc:
        log.warning("epub_extract_failed", path=str(path), error=str(exc))
        return None


def _find_opf(zf: zipfile.ZipFile) -> str:
    try:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", _CONTAINER_NS)
        if rootfile is not None:
            return rootfile.get("full-path", "") or ""
    except Exception as exc:  # malformed/missing container.xml — fall back to scanning for an .opf
        log.debug("epub_container_parse_failed", error=str(exc))
    return next((n for n in zf.namelist() if n.endswith(".opf")), "") or ""


def _parse_metadata(opf: ET.Element) -> EpubMetadata:
    m = EpubMetadata()
    metadata = opf.find("opf:metadata", _OPF_NS)
    if metadata is None:
        return m

    def _dc(tag: str) -> str:
        el = metadata.find(f"dc:{tag}", _OPF_NS)
        return (el.text or "").strip() if el is not None and el.text else ""

    m.title = _dc("title")
    m.author = _dc("creator")
    m.publisher = _dc("publisher")
    m.language = _dc("language")
    m.date = _dc("date")
    m.description = _dc("description")
    return m


def _find_cover_href(opf: ET.Element) -> str:
    """Locate the cover image href relative to the OPF directory.

    Tries EPUB 3 (``properties="cover-image"``) first, falls back to
    EPUB 2 (``<meta name="cover">`` → manifest id), and finally a
    filename heuristic for malformed EPUBs.
    """
    manifest_items = opf.findall("opf:manifest/opf:item", _OPF_NS)
    id_to_item = {it.get("id", ""): it for it in manifest_items}

    for it in manifest_items:
        props = (it.get("properties") or "").split()
        if "cover-image" in props:
            href = it.get("href", "")
            if href:
                return href

    meta_block = opf.find("opf:metadata", _OPF_NS)
    if meta_block is not None:
        for meta in meta_block.findall("opf:meta", _OPF_NS):
            if (meta.get("name") or "").lower() == "cover":
                cover_id = meta.get("content", "")
                if cover_id and cover_id in id_to_item:
                    href = id_to_item[cover_id].get("href", "")
                    if href:
                        return href

    for it in manifest_items:
        media_type = (it.get("media-type") or "").lower()
        href = (it.get("href") or "").lower()
        if media_type.startswith("image/") and "cover" in href:
            return it.get("href", "")

    return ""


_BLOCK_TAG_RE = re.compile(
    r"</(?:p|div|h[1-6]|li|tr|br|section|article|blockquote)\s*>"
    r"|<br\s*/?>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_HEADING_RE = re.compile(
    r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL,
)
_BODY_RE = re.compile(r"<body[^>]*>(.*?)</body>", re.IGNORECASE | re.DOTALL)
_DROP_BLOCKS_RE = re.compile(
    r"<(script|style|head|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL,
)


def _html_to_text(html: str) -> str:
    """Crude but dependency-free HTML → plain text for TTS."""
    html = _DROP_BLOCKS_RE.sub(" ", html)
    html = _BLOCK_TAG_RE.sub("\n", html)
    text = _TAG_RE.sub("", html)
    text = unescape(text)
    # Normalise whitespace: collapse intra-line runs, trim each line,
    # cap consecutive blank lines at one.
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    return _BLANKLINES_RE.sub("\n\n", text).strip()


def _first_heading(html: str) -> str:
    m = _HEADING_RE.search(html)
    if not m:
        return ""
    return _WS_RE.sub(" ", unescape(_TAG_RE.sub("", m.group(1)))).strip()


def chapters_text(path: str, *, max_chapters: int = 500) -> list[dict]:
    """Extract the EPUB's spine as a list of ``{heading, text}`` chapters.

    Plain text only — HTML chrome, scripts and styles are stripped. Used
    by the reader's read-aloud feature; intentionally lossy (no images,
    footnotes flattened). Returns ``[]`` for unreadable EPUBs.
    """
    try:
        with zipfile.ZipFile(str(path)) as zf:
            ensure_zip_sane(zf, source="epub_reader")
            opf_path = _find_opf(zf)
            if not opf_path:
                return []
            opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
            opf = ET.fromstring(zf.read(opf_path))

            manifest: dict[str, dict] = {}
            for item in opf.findall("opf:manifest/opf:item", _OPF_NS):
                manifest[item.get("id", "")] = {
                    "href": item.get("href", ""),
                    "type": item.get("media-type", ""),
                }
            spine_ids = [
                ref.get("idref", "")
                for ref in opf.findall("opf:spine/opf:itemref", _OPF_NS)
            ]

            out: list[dict] = []
            for sid in spine_ids:
                if len(out) >= max_chapters:
                    break
                entry = manifest.get(sid)
                if not entry or "html" not in entry["type"]:
                    continue
                try:
                    raw = zf.read(opf_dir + entry["href"]).decode("utf-8", "replace")
                except KeyError:
                    continue
                body_match = _BODY_RE.search(raw)
                body_html = body_match.group(1) if body_match else raw
                heading = _first_heading(body_html)
                text = _html_to_text(body_html)
                if not text:
                    continue
                out.append({
                    "heading": heading or f"Section {len(out) + 1}",
                    "text": text,
                })
            return out
    except zipfile.BadZipFile:
        return []
    except Exception as exc:
        log.warning("epub_chapters_text_failed", path=str(path), error=str(exc))
        return []


def _extract_cover_thumbnail(
    zf: zipfile.ZipFile, href: str, *, max_px: int,
) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        raw = zf.read(href)
    except KeyError:
        basename = href.rsplit("/", 1)[-1]
        matches = [n for n in zf.namelist() if n.endswith(basename)]
        if not matches:
            return None
        raw = zf.read(matches[0])
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=78)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None
