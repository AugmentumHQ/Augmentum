"""Deterministic geometric reading-order for boxed OCR regions.

Validated 2026-06-17: band-row sorting put a 24-region western comic page in
rough story order ("HOURS LATER" → … → "THE END"). It will still intermix
fragments from adjacent panels within a band — that residual is what the LLM
assembly pass (``assembly.py``) cleans up using semantic coherence. Geometry
gives the cheap reliable *rough* order; the model fixes the local seams.

``ltr`` (western) reads left→right within a top→bottom band. ``rtl`` (manga)
reads right→left. ``band`` is the vertical tolerance (fraction of page height)
within which two boxes are considered the same row — ~0.12 ≈ one panel-row.
"""

from __future__ import annotations

from augmentum.ocr.docling_client import Region


def default_reading_direction() -> str:
    """The install's configured reading direction (``'ltr'`` or ``'rtl'``).

    Every fallback in the comic/narration path routes through here rather than
    carrying its own ``"ltr"`` literal. That is the whole point: the previous
    arrangement had eight copies of the default, so a user who set RTL kept
    finding it undone by whichever call site happened to omit the argument.
    """
    from augmentum.config import settings

    return normalize_reading_direction(
        getattr(settings, "comic_default_reading_direction", ""), fallback="ltr",
    )


def normalize_reading_direction(value: object, *, fallback: str = "") -> str:
    """Coerce ``value`` to ``'ltr'``/``'rtl'``, else ``fallback``.

    ``fallback=""`` means "caller supplied nothing recognizable" and lets the
    caller decide; passing ``default_reading_direction()`` is the usual choice.
    Only the literal ``"ltr"`` bottom of :func:`default_reading_direction`
    hardcodes a direction — nothing else in the pipeline should.
    """
    v = str(value or "").strip().lower()
    return v if v in ("ltr", "rtl") else fallback


def order_regions(
    regions: list[Region],
    *,
    reading_direction: str = "",
    band: float = 0.12,
) -> list[Region]:
    """Return ``regions`` sorted into reading order (does not mutate input).

    An empty ``reading_direction`` means "unspecified" and falls through to the
    install default — not to left-to-right.
    """
    if not regions:
        return []
    rtl = normalize_reading_direction(
        reading_direction, fallback=default_reading_direction(),
    ) == "rtl"
    band = max(band, 1e-3)

    def key(r: Region) -> tuple[int, float]:
        row = round(r.cy / band)
        # ltr: ascending x. rtl: descending x → negate.
        col = (-r.cx) if rtl else r.cx
        return (row, col)

    return sorted(regions, key=key)
