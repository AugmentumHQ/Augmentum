"""Tests for the OCR layer + comic-narration store (audio-manga slice 1).

Pure-logic coverage of the pieces validated live 2026-06-17:
  - reading_order.order_regions   — band-row sort, ltr + rtl
  - docling_client.parse_docling_regions — bbox normalize + y-up flip
  - assembly._parse               — model-output parse incl. 'src' fragment ids
  - ocr._union_bbox               — merged-line bbox from source fragments
Plus a ComicNarrationStore round-trip (begin → progress → timeline → done →
delete) with user_id isolation.
"""

from __future__ import annotations

import asyncio

import aiosqlite

from augmentum.ocr import _union_bbox
from augmentum.ocr.assembly import _parse as parse_assembly
from augmentum.ocr.docling_client import Region, parse_docling_regions
from augmentum.ocr.reading_order import order_regions


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ───────────────────────── reading order ─────────────────────────

def test_order_regions_ltr_bands_top_to_bottom_left_to_right():
    # Two rows of two boxes each. cy: row0 ~0.1, row1 ~0.5.
    regs = [
        Region("B", 0.6, 0.10, 0.2, 0.06),   # row0 right
        Region("A", 0.1, 0.10, 0.2, 0.06),   # row0 left
        Region("D", 0.6, 0.50, 0.2, 0.06),   # row1 right
        Region("C", 0.1, 0.50, 0.2, 0.06),   # row1 left
    ]
    out = [r.text for r in order_regions(regs, reading_direction="ltr")]
    assert out == ["A", "B", "C", "D"]


def test_order_regions_rtl_reads_right_first():
    regs = [
        Region("L", 0.1, 0.10, 0.2, 0.06),
        Region("R", 0.6, 0.10, 0.2, 0.06),
    ]
    out = [r.text for r in order_regions(regs, reading_direction="rtl")]
    assert out == ["R", "L"]


def test_order_regions_empty():
    assert order_regions([]) == []


# ───────────────────── docling bbox parsing ──────────────────────

def test_parse_docling_flips_bottomleft_origin():
    # Page 100 tall. A box near the TOP of the page in BOTTOMLEFT coords has a
    # large t/b (y grows upward). After the flip it should land near y≈0.
    doc = {
        "pages": {"1": {"size": {"width": 100.0, "height": 100.0}}},
        "texts": [
            {"text": "TOP", "prov": [{"bbox": {
                "l": 10, "r": 30, "t": 95, "b": 85, "coord_origin": "BOTTOMLEFT"}}]},
            {"text": "BOTTOM", "prov": [{"bbox": {
                "l": 10, "r": 30, "t": 15, "b": 5, "coord_origin": "BOTTOMLEFT"}}]},
        ],
    }
    regs = parse_docling_regions(doc)
    by = {r.text: r for r in regs}
    assert by["TOP"].y < 0.1        # top of page → small y after flip
    assert by["BOTTOM"].y > 0.8     # bottom of page → large y
    # widths normalized to fraction of page
    assert abs(by["TOP"].w - 0.2) < 1e-6


def test_parse_docling_json_content_wrapper_and_skips_empty():
    doc = {"json_content": {
        "pages": {"1": {"size": {"width": 50.0, "height": 50.0}}},
        "texts": [
            {"text": "  ", "prov": [{"bbox": {"l": 0, "r": 1, "t": 1, "b": 0}}]},  # empty
            {"text": "X", "prov": [{"bbox": {"l": 0, "r": 10, "t": 10, "b": 0,
                                             "coord_origin": "TOPLEFT"}}]},
        ],
    }}
    regs = parse_docling_regions(doc)
    assert [r.text for r in regs] == ["X"]


def test_parse_docling_handles_missing_doc():
    assert parse_docling_regions(None) == []
    assert parse_docling_regions({}) == []


# ─────────────────── assembly output parsing ─────────────────────

def test_parse_assembly_extracts_lines_and_src():
    content = (
        'prefix junk {"lines": ['
        '{"order": 0, "kind": "narration", "text": "THE NEWS BURST", "src": [1]},'
        '{"order": 1, "kind": "speech", "text": "It\'s come at last!", "src": [4, 5]}'
        ']} trailing'
    )
    out = parse_assembly(content)
    assert len(out) == 2
    assert out[0]["kind"] == "narration"
    assert out[1]["text"] == "It's come at last!"
    assert out[1]["src"] == [4, 5]


def test_parse_assembly_bad_kind_defaults_speech_and_drops_empty():
    content = '{"lines": [{"kind": "weird", "text": "Hi", "src": []}, {"kind":"speech","text":"  ","src":[]}]}'
    out = parse_assembly(content)
    assert len(out) == 1
    assert out[0]["kind"] == "speech"


def test_parse_assembly_garbage_returns_empty():
    assert parse_assembly("not json at all") == []
    assert parse_assembly("") == []


# ───────────────────────── union bbox ────────────────────────────

def test_union_bbox_merges_source_fragments():
    regs = [
        Region("a", 0.10, 0.10, 0.10, 0.05),   # frag 1
        Region("b", 0.30, 0.20, 0.10, 0.05),   # frag 2
    ]
    bb = _union_bbox(regs, [1, 2])
    # x0=0.10, y0=0.10, x1=0.40, y1=0.25 → w=0.30, h=0.15
    assert bb == [0.10, 0.10, 0.30, 0.15]


def test_union_bbox_out_of_range_ids_return_none():
    regs = [Region("a", 0.1, 0.1, 0.1, 0.1)]
    assert _union_bbox(regs, [9]) is None
    assert _union_bbox(regs, []) is None


# ──────────────────── ComicNarrationStore ────────────────────────

_SCHEMA = """
CREATE TABLE users (id TEXT PRIMARY KEY, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE comic_narrations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    comic_kind TEXT NOT NULL,
    comic_ref TEXT NOT NULL,
    narration_artifact_id TEXT NOT NULL DEFAULT '',
    timeline TEXT NOT NULL DEFAULT '[]',
    pages TEXT NOT NULL DEFAULT '[]',
    voice TEXT NOT NULL DEFAULT '',
    voice_male TEXT NOT NULL DEFAULT '',
    voice_female TEXT NOT NULL DEFAULT '',
    voice_cast TEXT NOT NULL DEFAULT '{}',
    engine_id TEXT NOT NULL DEFAULT '',
    reading_direction TEXT NOT NULL DEFAULT 'ltr',
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    job_id TEXT NOT NULL DEFAULT '',
    processed_pages INTEGER NOT NULL DEFAULT 0,
    total_pages INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX idx_comic_narrations_unique
    ON comic_narrations(user_id, comic_kind, comic_ref);
"""


async def _setup_db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_SCHEMA)
    await conn.execute("INSERT INTO users (id) VALUES ('u1'), ('u2')")
    await conn.commit()
    return conn


def test_comic_narration_store_roundtrip():
    from augmentum.state.comic_narration_store import ComicNarrationStore

    async def _go():
        conn = await _setup_db()
        store = ComicNarrationStore(conn)
        rid = await store.begin("file", "c1", "eve", "job1",
                                engine_id="kokoro-builtin", reading_direction="rtl", user_id="u1")
        row = await store.get("file", "c1", user_id="u1")
        assert row["status"] == "pending"
        assert row["reading_direction"] == "rtl"

        await store.mark_running(rid)
        await store.set_progress(rid, 2, 10)
        tl = [{"page": 0, "order": 0, "bbox": [0.1, 0.1, 0.2, 0.1], "text": "Hi",
               "kind": "speech", "audio_start_ms": 0, "audio_end_ms": 900}]
        await store.set_timeline(rid, tl)
        row = await store.get("file", "c1", user_id="u1")
        assert row["status"] == "running"
        assert row["processed_pages"] == 2 and row["total_pages"] == 10

        # Streaming model: per-page list accumulates and survives a round-trip.
        page_entry = {"page": 0, "artifact_id": "pg0", "duration_ms": 900, "lines": tl}
        await store.set_pages(rid, [page_entry])
        row = await store.get("file", "c1", user_id="u1")
        import json
        assert json.loads(row["pages"])[0]["artifact_id"] == "pg0"

        await store.mark_done(rid, "art123", tl, pages=[page_entry])
        row = await store.get("file", "c1", user_id="u1")
        assert row["status"] == "done"
        assert row["narration_artifact_id"] == "art123"
        assert json.loads(row["timeline"])[0]["text"] == "Hi"
        assert json.loads(row["pages"])[0]["lines"][0]["text"] == "Hi"

        # Streaming-only mark_done (no legacy artifact/timeline) still works.
        await store.mark_done(rid, pages=[page_entry, {**page_entry, "page": 1}])
        row = await store.get("file", "c1", user_id="u1")
        assert len(json.loads(row["pages"])) == 2

        # user isolation
        assert await store.get("file", "c1", user_id="u2") is None

        # delete
        assert await store.delete("file", "c1", user_id="u1") is True
        assert await store.get("file", "c1", user_id="u1") is None
        await conn.close()

    _run(_go())


def test_comic_narration_store_begin_resets_existing():
    from augmentum.state.comic_narration_store import ComicNarrationStore

    async def _go():
        conn = await _setup_db()
        store = ComicNarrationStore(conn)
        rid1 = await store.begin("file", "c1", "eve", "job1", user_id="u1")
        await store.mark_done(rid1, "art1", [{"x": 1}], pages=[{"page": 0}])
        # Re-begin should reset the SAME row to pending + clear artifact/timeline/pages.
        rid2 = await store.begin("file", "c1", "amy", "job2", user_id="u1")
        assert rid1 == rid2
        row = await store.get("file", "c1", user_id="u1")
        assert row["status"] == "pending"
        assert row["narration_artifact_id"] == ""
        assert row["timeline"] == "[]"
        assert row["pages"] == "[]"
        assert row["voice"] == "amy"
        await conn.close()

    _run(_go())
