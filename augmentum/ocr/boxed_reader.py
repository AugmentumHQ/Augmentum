"""Bubble-cropped page reading — docling finds the text, the VLM reads it.

The full-page VLM path (:mod:`vlm_reader`) hands one whole comic page to the
vision model. That loses on arithmetic: the Gemma-4 projector declares
``clip.vision.image_size = 224``, so a 1600x2400 page is resampled toward 224px
before the model ever sees it and comic lettering lands a few pixels tall. The
model isn't misreading the page so much as being shown a thumbnail of it.

This module spends the same token budget differently. Docling boxes the text,
the boxes are grouped into bubbles, each bubble is CROPPED, and the crops go to
the model as a numbered image sequence in ONE request. A ~250x150 bubble crop
nearly fills the 224px tower instead of occupying a fiftieth of it — roughly an
order of magnitude more pixels per character, at the same one-call-per-page
cost.

Two structural wins fall out of it:

* **Reading order stops being the model's job.** Order comes from
  :func:`~augmentum.ocr.reading_order.order_regions` — deterministic and
  direction-aware. The failure where a manga page gets swept left-to-right
  can't happen here, because the model never chooses a sweep. It reads crop 1,
  then crop 2.
* **Bounding boxes come back.** The full-page path returns ``bbox=None`` (VLMs
  confabulate coordinates); here the geometry is docling's, so it's real, and
  pan-and-scan playback works again.

The division of labour matters: docling contributes ONLY geometry. Its OCR text
is discarded — that text is the rough lettering the old assembly pass existed to
repair, and it's the part docling is worst at (stylized fonts, non-English).
What docling is genuinely good at is finding where the text is.

Its remaining failure is a *missed* bubble — unusual shapes, hand lettering,
webtoon panels — which geometry can't recover, because nothing was found to
crop. Guard-railed two ways rather than one: too few regions falls back to the
whole-page read, and every empty batch is checked against the endpoint being
alive (an unreachable service must never read as "this page has no dialogue").
"""

from __future__ import annotations

import io
import re

import httpx

from augmentum.ocr.docling_client import Region
from augmentum.ocr.reading_order import order_regions
from augmentum.utils.logging import get_logger
from augmentum.vision.provider import _caption_via_openai_endpoint

log = get_logger(__name__)

__all__ = ["group_regions", "crop_regions", "read_boxed_page", "BOXED_PROMPT"]


def group_regions(regions: list[Region], *, gap: float = 0.012) -> list[Region]:
    """Merge docling's text spans into bubble-sized groups.

    Docling boxes *lines*, not balloons, so cropping its raw output would hand
    the model one line of a sentence at a time — losing exactly the sentence
    context that makes a joined line read as English.

    Grouping is geometric union-find: two boxes join when they overlap after
    being expanded by ``gap``. Lines stacked inside one balloon are within a
    line-height of each other and merge; separate balloons are held apart by
    the balloon border and the art between them. ``gap`` is a fraction of page
    size — ~0.012 is about one line of lettering.
    """
    n = len(regions)
    if n <= 1:
        return list(regions)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def touches(a: Region, b: Region) -> bool:
        return not (
            a.x - gap > b.x + b.w + gap
            or b.x - gap > a.x + a.w + gap
            or a.y - gap > b.y + b.h + gap
            or b.y - gap > a.y + a.h + gap
        )

    for i in range(n):
        for j in range(i + 1, n):
            if touches(regions[i], regions[j]):
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)

    out: list[Region] = []
    for members in buckets.values():
        picked = [regions[i] for i in members]
        x0 = min(r.x for r in picked)
        y0 = min(r.y for r in picked)
        x1 = max(r.x + r.w for r in picked)
        y1 = max(r.y + r.h for r in picked)
        # Top-to-bottom join so the carried text (used only for logging and as
        # a last-resort fallback) reads in the order it was lettered.
        picked.sort(key=lambda r: (r.y, r.x))
        out.append(Region(
            text=" ".join(r.text for r in picked).strip(),
            x=x0, y=y0, w=x1 - x0, h=y1 - y0,
        ))
    return out


def crop_regions(
    image_bytes: bytes,
    regions: list[Region],
    *,
    pad: float = 0.06,
    min_px: int = 32,
) -> list[bytes]:
    """Crop ``regions`` out of the page as PNG bytes, in the given order.

    ``pad`` is a fraction of each box's own size, so a small bubble gets a
    small margin and a big one a big margin. The padding is not cosmetic:
    docling's boxes hug the glyphs, and a hairline crop clips ascenders and
    the balloon tail that says who is speaking.

    A crop that fails is returned as ``b""`` rather than dropped, so the
    caller's index-to-region mapping stays aligned.
    """
    from PIL import Image

    out: list[bytes] = []
    try:
        page = Image.open(io.BytesIO(image_bytes))
        if page.mode not in ("RGB", "L"):
            page = page.convert("RGB")
    except Exception:  # noqa: BLE001 — unreadable page → caller falls back
        log.warning("ocr_boxed_page_decode_failed", bytes=len(image_bytes))
        return [b"" for _ in regions]

    pw, ph = page.size
    for r in regions:
        try:
            px = r.w * pad
            py = r.h * pad
            left = max(0, int((r.x - px) * pw))
            top = max(0, int((r.y - py) * ph))
            right = min(pw, int((r.x + r.w + px) * pw))
            bottom = min(ph, int((r.y + r.h + py) * ph))
            if right - left < min_px or bottom - top < min_px:
                # Too small to carry legible text — pad out to a floor rather
                # than hand the model a sliver.
                cx, cy = (left + right) // 2, (top + bottom) // 2
                left = max(0, cx - min_px // 2)
                top = max(0, cy - min_px // 2)
                right = min(pw, left + min_px)
                bottom = min(ph, top + min_px)
            buf = io.BytesIO()
            page.crop((left, top, right, bottom)).save(buf, format="PNG")
            out.append(buf.getvalue())
        except Exception:  # noqa: BLE001 — one bad crop must not lose the page
            out.append(b"")
    return out


BOXED_PROMPT = """You are transcribing speech bubbles from one page of a comic for a \
text-to-speech reading. Each image is ONE cropped bubble from that page, given \
in reading order.

- Output one line per image, in the same order, formatted exactly: `N: text`
- N is the image's number, starting at 1.
- Read each bubble top to bottom. Text that wraps across several lines is ONE \
sentence — join it, repairing any word split by the line break.
- Every line must read aloud as a grammatical English sentence. If a joined \
line comes out garbled or starts mid-sentence, re-read the bubble and correct it.
- Write in normal sentence case, not ALL CAPS. Keep ? and !, but collapse \
decorative repetition (AAAAHHH -> Aah).
- Transcribe only what is legibly printed. Never invent or continue dialogue \
that is not there.
- If an image has no readable text, output `N: -` for it.

Output only those lines. No commentary, no explanation, no headings."""

_NUMBERED = re.compile(r"^\s*\[?(\d{1,3})\]?\s*[:.)-]\s*(.+?)\s*$")


def _parse_numbered(raw: str, count: int) -> dict[int, str]:
    """``"1: hello"`` lines → ``{0: 'hello'}`` (0-based, in-range only)."""
    out: dict[int, str] = {}
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text or text.startswith("```"):
            continue
        m = _NUMBERED.match(text)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        body = m.group(2).strip()
        if idx < 0 or idx >= count:
            continue
        if len(body) > 1 and body[0] == body[-1] and body[0] in "\"'":
            body = body[1:-1].strip()
        if not body or body in ("-", "—", "–", "(empty)", "[empty]"):
            continue
        out[idx] = body
    return out


async def read_boxed_page(
    image_bytes: bytes,
    regions: list[Region],
    *,
    reader: tuple[str, str],
    reading_direction: str = "",
    batch_size: int = 12,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
    prompt: str = "",
) -> list[dict]:
    """Boxed page → ``[{order, kind, text, bbox}]`` in deterministic order.

    ``regions`` is docling's raw output; grouping, ordering and cropping happen
    here so callers can't get the sequence wrong. Crops are sent in batches
    (``batch_size`` images per request) because a whole page of bubbles in one
    message would blow past the context window on a dense page — and because a
    misnumbered reply costs one batch, not the page.

    Raises on a batch that comes back empty from a dead endpoint; a batch the
    model genuinely read as blank yields no lines and is not an error.
    """
    from augmentum.ocr.vlm_reader import VisionNotConfigured

    base_url, model = reader
    grouped = group_regions(regions)
    ordered = order_regions(grouped, reading_direction=reading_direction)
    if not ordered:
        return []

    crops = crop_regions(image_bytes, ordered)
    instructions = (prompt or "").strip() or BOXED_PROMPT

    lines: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        for start in range(0, len(ordered), max(1, batch_size)):
            chunk = list(range(start, min(start + batch_size, len(ordered))))
            frames = [crops[i] for i in chunk if crops[i]]
            live = [i for i in chunk if crops[i]]
            if not frames:
                continue
            raw = await _caption_via_openai_endpoint(
                http,
                base_url,
                frames[0],
                prompt=instructions,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                model=model,
                extra_frames=frames[1:],
                # Same reason as the whole-page read: thinking shares the
                # max_tokens budget with the answer, and a batch of crops is
                # the densest prompt in this pipeline. This path is currently
                # disabled (ocr_vlm_use_boxes), but the defect is identical and
                # leaving it armed would just reintroduce the bug on the day it
                # gets switched back on.
                enable_thinking=False,
            )
            if not (raw or "").strip():
                # Empty is either "these bubbles were unreadable" or "the
                # service just died". Guessing wrong is how a dead endpoint
                # becomes a chapter that reports no dialogue.
                root = base_url.rstrip("/")
                root = root[:-3].rstrip("/") if root.endswith("/v1") else root
                try:
                    resp = await http.get(f"{root}/v1/models", timeout=10.0)
                    resp.raise_for_status()
                except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
                    raise VisionNotConfigured(
                        f"The service serving '{model}' stopped responding "
                        f"({str(exc)[:120]}). Narration stopped rather than "
                        "transcribing the rest of the comic as blank pages.",
                    ) from exc
                continue

            parsed = _parse_numbered(raw, len(live))
            for local, region_idx in enumerate(live):
                text = parsed.get(local)
                if not text:
                    continue
                r = ordered[region_idx]
                lines.append({
                    "order": len(lines),
                    "kind": "speech",
                    "text": text,
                    "bbox": r.as_bbox(),
                })

    log.info(
        "ocr_boxed_read",
        model=model,
        direction=reading_direction,
        regions=len(regions),
        bubbles=len(ordered),
        out_lines=len(lines),
    )
    return lines
