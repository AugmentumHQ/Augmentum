"""OCR with boxes + LLM script-assembly — Augmentum's reusable OCR layer.

The high-level entry point is :func:`extract_page_script`: image bytes in,
a clean reading-ordered list of ``{order, kind, text, bbox}`` out. It composes
the three validated pieces:

1. :mod:`docling_client` — image → boxed text regions (docling-serve sidecar).
2. :mod:`reading_order` — deterministic band-row sort (rough, reliable order).
3. :mod:`assembly` — an LLM pass that cleans the rough lettering, merges split
   sentences, fixes local order by meaning, and tags narration/speech/sfx.

This makes rough comic OCR usable for TTS narration. The same extraction layer
serves PDF/document/knowledge ingestion (without the assembly pass, which is
comic-specific). Validated live 2026-06-17 — see the audio-manga design spec.
"""

from __future__ import annotations

from augmentum.ocr.assembly import assemble_script
from augmentum.ocr.docling_client import DoclingClient, Region, parse_docling_regions
from augmentum.ocr.reading_order import (
    default_reading_direction,
    normalize_reading_direction,
    order_regions,
)
from augmentum.ocr.vlm_reader import read_page_script
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DoclingClient",
    "Region",
    "parse_docling_regions",
    "order_regions",
    "assemble_script",
    "read_page_script",
    "extract_page_script",
    "ocr_enabled",
    "get_docling_client",
]


def ocr_enabled(settings) -> bool:
    return bool(getattr(settings, "ocr_enabled", False))


def get_docling_client(settings) -> DoclingClient:
    base = getattr(settings, "ocr_base_url", "") or "http://ocr:5001"
    timeout = float(getattr(settings, "ocr_timeout_s", 120.0) or 120.0)
    return DoclingClient(base, timeout_s=timeout)


def _union_bbox(regions: list[Region], src_ids: list[int]) -> list[float] | None:
    """Union the boxes of the 1-based source fragment numbers ``src_ids``."""
    picked = [regions[i - 1] for i in src_ids if 1 <= i <= len(regions)]
    if not picked:
        return None
    x0 = min(r.x for r in picked)
    y0 = min(r.y for r in picked)
    x1 = max(r.x + r.w for r in picked)
    y1 = max(r.y + r.h for r in picked)
    return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]


async def _boxed_page_script(
    app,
    settings,
    image_bytes: bytes,
    *,
    reading_direction: str,
    filename: str,
    vlm_reader: tuple[str, str] | None,
) -> list[dict] | None:
    """Docling-boxed + VLM-cropped read, or ``None`` to fall back to full page.

    ``None`` (not ``[]``) is returned when docling contributes nothing usable —
    the sidecar is down, or it found fewer regions than a page with dialogue
    could plausibly have. ``[]`` stays reserved for "this page really is
    textless", which the caller must be able to tell apart.
    """
    from augmentum.ocr.boxed_reader import read_boxed_page
    from augmentum.ocr.vlm_reader import resolve_reader

    min_regions = int(getattr(settings, "ocr_vlm_boxes_min_regions", 2) or 0)
    try:
        client = get_docling_client(settings)
        regions = await client.convert_image(
            image_bytes,
            filename=filename,
            force_full_page=bool(getattr(settings, "ocr_force_full_page", True)),
        )
    except Exception as exc:  # noqa: BLE001 — sidecar down → whole-page read
        log.warning("ocr_boxed_docling_unavailable", error=str(exc)[:200])
        return None

    if len(regions) < max(1, min_regions):
        log.info("ocr_boxed_too_few_regions", regions=len(regions), floor=min_regions)
        return None

    reader = vlm_reader or await resolve_reader(app)
    return await read_boxed_page(
        image_bytes,
        regions,
        reader=reader,
        reading_direction=reading_direction,
        batch_size=int(getattr(settings, "ocr_vlm_batch_images", 12) or 12),
        max_tokens=int(getattr(settings, "ocr_vlm_max_tokens", 1024) or 1024),
        timeout_s=float(getattr(settings, "ocr_vlm_timeout_s", 120.0) or 120.0),
        prompt=getattr(settings, "ocr_vlm_boxes_prompt", "") or "",
    )


async def extract_page_script(
    app,
    image_bytes: bytes,
    *,
    reading_direction: str = "",
    assemble: bool = True,
    filename: str = "page.jpg",
    vlm_reader: tuple[str, str] | None = None,
    refine_reader: tuple[str, str] | None = None,
    glossary: list[str] | None = None,
) -> list[dict]:
    """One comic page image → ``[{order, kind, text, bbox}]`` reading-ordered.

    ``bbox`` is ``[x, y, w, h]`` normalized 0..1 (origin top-left) for pan/scan,
    or ``None`` when the line can't be located (player holds on the full page).
    Returns ``[]`` for a textless page (splash art) — not an error. Raises only
    on OCR transport failure (the synth job marks the page failed and continues).
    """
    from augmentum.config import settings

    # Resolve ONCE, here, at the top of the only entry point — every path below
    # (boxed, whole-page, refine) then reads the same answer. Resolving it
    # per-callee is how the pipeline ended up with `read_page_script` defaulting
    # to rtl while this function defaulted to ltr: two halves of one read
    # disagreeing about which way the page goes.
    reading_direction = normalize_reading_direction(
        reading_direction, fallback=default_reading_direction(),
    )

    # Vision-LLM engine: whole page → transcript, no docling. Only for the
    # assembled path — ``assemble=False`` callers (the ocr_extract tool, doc
    # ingestion) want true geometry, which a VLM can't give.
    engine = (getattr(settings, "ocr_engine", "docling") or "docling").lower()
    if assemble and engine == "vlm":
        from augmentum.ocr.vlm_reader import read_page_script, split_speaker

        def _finalize(lines: list[dict]) -> list[dict]:
            """Split the ``[M]/[F]/[N]`` speaker tag off each line's text.

            The tag rides inline through both reads and the proof-read (so
            dedup/drift/similarity see a stable string); it's peeled here, at
            the one exit, into a ``speaker`` field the synth job reads to pick a
            voice. A narration tag also settles the line's ``kind`` — the
            whole-page reader can't otherwise tell a caption from dialogue.
            """
            out = []
            for ln in lines or []:
                tag, text = split_speaker(ln.get("text") or "")
                if not text:
                    continue
                # An explicit [N] tag settles the kind as narration; an untagged
                # line keeps whatever kind it had and reads in the narrator voice.
                speaker = tag or "narrator"
                kind = "narration" if tag == "narrator" else ln.get("kind", "speech")
                out.append({**ln, "text": text, "speaker": speaker, "kind": kind})
            return out

        async def _refine(draft: list[dict]) -> list[dict]:
            """Pass 2 — proof-read ``draft`` against the same image.

            Applies to BOTH read paths: boxed crops and whole-page share the
            same weakness (a page read with no idea what the chapter has
            established) and so share the same remedy. Skipped when disabled or
            when no refiner was resolved — a single-shot caller shouldn't pay
            for a second model resolution it never asked for.
            """
            if not (bool(getattr(settings, "ocr_vlm_second_pass_enabled", True)) and refine_reader):
                return draft
            from augmentum.ocr.second_pass import refine_page_script

            return await refine_page_script(
                image_bytes,
                draft,
                reader=refine_reader,
                reading_direction=reading_direction,
                glossary=glossary or [],
                prompt=getattr(settings, "ocr_vlm_second_pass_prompt", "") or "",
                max_tokens=int(getattr(settings, "ocr_vlm_max_tokens", 1024) or 1024),
                timeout_s=float(getattr(settings, "ocr_vlm_timeout_s", 120.0) or 120.0),
                max_drift=float(getattr(settings, "ocr_vlm_second_pass_max_drift", 0.5) or 0.5),
                rescue_empty=bool(getattr(settings, "ocr_vlm_rescue_empty", True)),
            )

        # Boxed path: docling finds the text, the VLM reads CROPS of it. Worth
        # the extra sidecar call because the vision tower is 224px — a bubble
        # crop lands near-native there, a whole page lands as a thumbnail. See
        # boxed_reader's module docstring.
        if bool(getattr(settings, "ocr_vlm_use_boxes", True)):
            lines = await _boxed_page_script(
                app, settings, image_bytes,
                reading_direction=reading_direction,
                filename=filename,
                vlm_reader=vlm_reader,
            )
            # None = docling found too little to trust (missed the lettering
            # entirely, unusual balloons, webtoon panels). Fall through to the
            # whole-page read rather than narrate a page as silent.
            if lines is not None:
                return _finalize(await _refine(lines))

        lines = await read_page_script(
            app,
            image_bytes,
            prompt=getattr(settings, "ocr_vlm_prompt", "") or "",
            reading_direction=reading_direction,
            # The chapter's confirmed spellings ride on the FIRST read too, not
            # just the proof-read. Pass 1 is the read that has to recover a name
            # from the lettering; handing it the answer is free (no extra call)
            # and is the only part of the two-pass benefit that survives with
            # the second pass switched off.
            glossary=glossary or [],
            # Resolved once per chapter by the caller (see the narration job);
            # None means single-shot use and resolves per call.
            reader=vlm_reader,
            max_tokens=int(getattr(settings, "ocr_vlm_max_tokens", 1024) or 1024),
            timeout_s=float(getattr(settings, "ocr_vlm_timeout_s", 120.0) or 120.0),
        )

        return _finalize(await _refine(lines))

    client = get_docling_client(settings)
    regions = await client.convert_image(
        image_bytes,
        filename=filename,
        force_full_page=bool(getattr(settings, "ocr_force_full_page", True)),
    )
    if not regions:
        return []

    ordered = order_regions(regions, reading_direction=reading_direction)

    if not assemble or not bool(getattr(settings, "ocr_assembly_enabled", True)):
        return [
            {"order": i, "kind": "speech", "text": r.text, "bbox": r.as_bbox()}
            for i, r in enumerate(ordered)
        ]

    sm = getattr(app.state, "state_manager", None)
    registry = getattr(app.state, "provider_registry", None) or getattr(app.state, "registry", None)
    lines = None
    if registry is not None:
        lines = await assemble_script(
            registry,
            settings,
            ordered,
            role=getattr(settings, "ocr_assembly_role", "classifier") or "classifier",
            override_model=getattr(settings, "ocr_assembly_model", "") or "",
            timeout_s=float(getattr(settings, "ocr_assembly_timeout_s", 60.0) or 60.0),
        )
    if not lines:
        # Assembly unavailable/empty → fall back to the raw geometric order.
        # Still narratable, just rougher lettering and panel seams.
        log.info("ocr_assembly_fallback_raw", regions=len(ordered))
        return [
            {"order": i, "kind": "speech", "text": r.text, "bbox": r.as_bbox()}
            for i, r in enumerate(ordered)
        ]

    out = []
    for i, ln in enumerate(lines):
        out.append({
            "order": i,
            "kind": ln.get("kind", "speech"),
            "text": ln["text"],
            "bbox": _union_bbox(ordered, ln.get("src") or []),
        })
    return out
