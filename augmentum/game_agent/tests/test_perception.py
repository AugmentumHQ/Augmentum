"""Tests for the game-agent perception pipeline (dedup + grid overlay).

The dangerous failure mode is dedup dropping a *real* game state (hiding
information from the agent), so the dedup tests assert distinct frames
always survive while only byte-near-identical ones collapse.
"""

from __future__ import annotations

import io

import pytest

from augmentum.game_agent.perception import (
    PreparedFrames,
    dedup_frames,
    draw_grid,
    prepare_frames,
)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _frame(color=(30, 40, 60), marks=()):
    img = Image.new("RGB", (240, 160), color)
    d = ImageDraw.Draw(img)
    for (x, y, s) in marks:
        d.rectangle([x, y, x + s, y + s], fill=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── dedup ─────────────────────────────────────────────────────────────


def test_dedup_collapses_a_fully_static_window_to_one():
    a, b, c = _frame(), _frame(), _frame()  # three identical captures
    assert len(dedup_frames([a, b, c])) == 1


def test_dedup_preserves_a_sprite_appearing():
    static = _frame()
    appeared = _frame(marks=[(100, 80, 16)])
    # The newest frame is a real change — it must survive.
    assert len(dedup_frames([static, static, appeared])) == 2


def test_dedup_preserves_every_distinct_frame_in_motion():
    f0 = _frame()
    f1 = _frame(marks=[(100, 80, 16)])
    f2 = _frame(marks=[(140, 80, 16)])  # sprite stepped one tile
    assert len(dedup_frames([f0, f1, f2])) == 3


def test_dedup_preserves_a_menu_cursor_move():
    # A cursor moving between menu options is a small but real change the
    # agent must see. It must NOT be collapsed.
    opt1 = _frame(marks=[(20, 120, 8)])
    opt2 = _frame(marks=[(20, 136, 8)])
    assert len(dedup_frames([opt1, opt2])) == 2


def test_dedup_collapses_subtle_global_animation_noise():
    # A faint whole-screen shimmer (water-tile animation, re-encode jitter)
    # carries no gameplay information and SHOULD collapse — the max-diff
    # metric ignores low-amplitude global change.
    base = _frame((30, 40, 60))
    shimmer = _frame((30, 42, 62))  # every pixel +2 — below the noise floor
    assert len(dedup_frames([base, shimmer])) == 1


def test_dedup_noop_on_short_inputs():
    assert dedup_frames([]) == []
    one = _frame()
    assert dedup_frames([one]) == [one]


# ── grid overlay ──────────────────────────────────────────────────────


def test_grid_overlay_returns_a_valid_image_of_same_size():
    src = _frame()
    out = draw_grid(src, cols=8, rows=6)
    assert out != src  # something was drawn
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (240, 160)


def test_grid_overlay_fails_open_on_garbage_input():
    junk = b"not a png"
    assert draw_grid(junk) == junk  # returns input unchanged, no raise


# ── prepare_frames orchestration ──────────────────────────────────────


def test_prepare_frames_static_window_collapses_and_notes_it():
    a = _frame()
    out: PreparedFrames = prepare_frames([a, a, a], dedup=True, grid=True)
    assert out.n_in == 3
    assert out.n_unique == 1
    assert len(out.frames) == 1
    assert out.grid_applied is True
    assert "static" in out.note.lower()
    assert "grid" in out.note.lower()


def test_prepare_frames_motion_window_keeps_all():
    f0 = _frame()
    f1 = _frame(marks=[(100, 80, 16)])
    f2 = _frame(marks=[(140, 80, 16)])
    out = prepare_frames([f0, f1, f2], dedup=True, grid=True)
    assert out.n_unique == 3
    assert len(out.frames) == 3


def test_prepare_frames_passthrough_when_disabled():
    a, b = _frame(), _frame()
    out = prepare_frames([a, b], dedup=False, grid=False)
    assert out.frames == [a, b]
    assert out.note == ""
    assert out.grid_applied is False


def test_prepare_frames_empty_is_text_only():
    out = prepare_frames([])
    assert out.frames == []
    assert out.note == ""
    assert out.n_in == 0
