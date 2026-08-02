"""Async client for a docling-serve OCR sidecar.

POSTs an image to ``/v1/convert/file`` (multipart) and parses the
``DoclingDocument`` JSON into a flat list of boxed text regions. Validated
2026-06-17 against ``docling-serve-cpu:latest`` on real Golden-Age comic
pages: a full page → ~24 regions in ~4.5s on CPU, each carrying a provenance
bounding box.

The sidecar bundles only the layout + OCR models, **not** the Granite-Docling
VLM — so we use the default OCR pipeline (``do_ocr=true``), not ``pipeline=vlm``.

Bounding-box coordinate system: docling reports ``prov[0].bbox`` with
``coord_origin`` usually ``BOTTOMLEFT`` (y grows upward). :class:`Region`
normalizes everything to a **top-down** frame (y grows downward, origin
top-left, all values in 0..1 against the page size) so downstream
reading-order/rendering code never has to think about the flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Region:
    """One OCR'd text span with a normalized top-down bounding box.

    ``x, y, w, h`` are fractions of the page (0..1), origin top-left. ``text``
    is the raw OCR output (rough on stylized lettering — the assembly pass
    cleans it). ``cx, cy`` are the box centre, used for reading-order sorting.
    """

    text: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def as_bbox(self) -> list[float]:
        return [round(self.x, 5), round(self.y, 5), round(self.w, 5), round(self.h, 5)]


def parse_docling_regions(doc: dict) -> list[Region]:
    """Flatten a ``DoclingDocument`` JSON into normalized :class:`Region`s.

    Tolerant of both response shapes: the bare document or the
    ``{"json_content": {...}}`` wrapper that docling-serve returns when
    ``to_formats`` includes ``json``. Regions with empty text or no bbox are
    dropped. Returns regions in docling's native (unordered) sequence — call
    :func:`augmentum.ocr.reading_order.order_regions` to sort.
    """
    if not isinstance(doc, dict):
        return []
    inner = doc.get("json_content")
    if isinstance(inner, dict):
        doc = inner

    pages = doc.get("pages") or {}
    page_w = page_h = 0.0
    if pages:
        first = next(iter(pages.values()), {}) or {}
        size = first.get("size") or {}
        page_w = float(size.get("width") or 0.0)
        page_h = float(size.get("height") or 0.0)

    out: list[Region] = []
    for span in doc.get("texts") or []:
        text = (span.get("text") or "").strip()
        if not text:
            continue
        prov = (span.get("prov") or [{}])
        bbox = (prov[0] or {}).get("bbox") if prov else None
        if not bbox:
            continue
        try:
            l = float(bbox.get("l", 0.0))
            t = float(bbox.get("t", 0.0))
            r = float(bbox.get("r", 0.0))
            b = float(bbox.get("b", 0.0))
        except (TypeError, ValueError):
            continue
        origin = (bbox.get("coord_origin") or "BOTTOMLEFT").upper()
        # Per-span page size if present (multi-page docs), else the doc size.
        pw = page_w
        ph = page_h
        if prov and isinstance(prov[0], dict):
            # docling sometimes stamps the page no.; size still comes from pages map
            pass
        if pw <= 0 or ph <= 0:
            # No page dims — fall back to raw coords so callers at least get
            # relative ordering. Normalize against the bbox extent as a guess.
            pw = max(pw, r, l, 1.0)
            ph = max(ph, t, b, 1.0)

        # Normalize to top-down, top-left origin, 0..1.
        left = min(l, r)
        right = max(l, r)
        if origin == "BOTTOMLEFT":
            top = ph - max(t, b)      # larger y = higher → flip
            bottom = ph - min(t, b)
        else:
            top = min(t, b)
            bottom = max(t, b)
        x = max(0.0, left / pw)
        y = max(0.0, top / ph)
        w = max(0.0, (right - left) / pw)
        h = max(0.0, (bottom - top) / ph)
        out.append(Region(text=text, x=x, y=y, w=w, h=h))
    return out


class DoclingClient:
    """Minimal async wrapper over a docling-serve instance."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self._base}/health")
                return r.status_code == 200
        except Exception:  # noqa: BLE001 — health probe is best-effort
            return False

    async def convert_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "page.jpg",
        force_full_page: bool = True,
    ) -> list[Region]:
        """OCR one image → normalized boxed regions (docling-native order).

        ``force_full_page`` maps to docling's ``force_ocr`` — on comics the
        whole page is lettering, so forcing OCR over the full page (rather
        than trusting layout's text/figure split) extracts far more.
        Raises on transport / non-success so the caller (the synth job) marks
        the page failed rather than silently emitting an empty timeline.
        """
        data = {
            "from_formats": "image",
            "to_formats": "json",
            "do_ocr": "true",
            "force_ocr": "true" if force_full_page else "false",
        }
        files = {"files": (filename, image_bytes, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            resp = await c.post(f"{self._base}/v1/convert/file", data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()
        status = (payload.get("status") or "").lower()
        if status and status not in ("success", "partial_success"):
            errs = payload.get("errors") or []
            raise RuntimeError(f"docling convert {status}: {errs}")
        document = payload.get("document") or {}
        regions = parse_docling_regions(document)
        log.info("docling_convert", regions=len(regions), status=status or "?")
        return regions
