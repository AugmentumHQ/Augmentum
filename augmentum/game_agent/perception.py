"""Frame perception preparation for the game agent.

This is the single seam that turns the raw frame stack a surface adapter
captured into the image set (plus prompt annotations) the slow-path LLM
actually consumes. Every vision-quality improvement composes here rather
than in any one adapter, so it works uniformly for js13k / emulatorjs /
emulator / curated:

    raw frames
      -> dedup       (drop redundant near-identical frames)
      -> grid overlay (Set-of-Marks: labeled cells for spatial grounding)
      -> [future] RAM-derived overlay / tile-aligned grid
      -> PreparedFrames(frames, note)

Why this exists (grounded in the LLM-plays-games literature):

* Turn-based / menu-heavy games (Pokémon) sit on static screens most of
  the time, so a fixed-cadence capture hands the model 3 *identical*
  frames — wasting 2/3 of the vision budget AND lying to it (the prompt
  says "oldest→newest, reason about CHANGE" when nothing changed).
  Dedup collapses a static stack to one frame and lets the prompt tell
  the truth. PokemonRedExperiments V2 made the same move (it replaced
  pixel-frame novelty tracking with state-based tracking); we dedup on a
  perceptual fingerprint here because, lacking RAM, the fingerprint is
  the change signal we have.

* An on-frame labeled grid is the strongest vision scaffold short of RAM
  state (Set-of-Marks prompting; Claude Plays Pokémon's navigator grid;
  lmgame-Bench symbolic modules). VLMs are trained on natural images, not
  action-oriented spatial reasoning, so giving them explicit cell labels
  to reference ("the player is in D4") beats asking them to localize from
  raw pixels. Without RAM we can only draw a *fixed* screen-space grid;
  when RAM lands, the same overlay becomes tile-aligned + walkability-
  colored via :func:`overlay_from_state` (the documented extension point).

Pure and dependency-light: Pillow is the only import, and every step is
individually toggleable so a deployment can dial perception up or down
without touching the orchestrator.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import structlog
from PIL import Image, ImageDraw

log = structlog.get_logger(__name__)

# Fingerprint grid edge — 32 → 1024 sample pixels. Fine enough that a
# small localized element (an 8px menu cursor on a 240px frame) survives
# the downscale and registers in the change metric.
_FP_EDGE = 32

# Change metric is the MAX per-channel pixel difference across the
# downscaled fingerprint, NOT an average. Average SAD dilutes a small
# bright element (a cursor) across the whole frame and would drop it;
# max-diff stays sensitive to localized changes while ignoring global
# re-encode/animation noise. Threshold is in 0–255 units: two frames are
# "the same screen" (collapsed) only when their max per-channel diff is
# <= this. Calibrated on 240×160 GBA frames at a 32×32 fingerprint:
# identical = 0, subtle global animation shimmer (water tiles) = ~2, a
# menu cursor moving one slot = ~156, a sprite appearing = ~225, a dialog
# box = ~210. A threshold of 24 sits far above the noise floor and far
# below any real change — so dedup only ever collapses a genuinely static
# window and never hides a meaningful update from the agent.
_DEFAULT_DEDUP_THRESHOLD = 24

# Default grid shape. 8 columns × 6 rows over a 240×160-native (or
# upscaled) GBA frame gives ~30×27 px cells at native res — fine enough to
# name a position, coarse enough that labels stay legible.
_DEFAULT_GRID_COLS = 8
_DEFAULT_GRID_ROWS = 6

_COL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Default longest-edge cap applied before any other perception step.
# EmulatorJS hands us the *display* canvas — a nearest-neighbor upscale
# of the native framebuffer (GBA is 240×160; the canvas is whatever the
# window made it, often 4-6x that). The upscale carries zero information
# but costs real time at every hop: browser PNG encode, postMessage,
# WebSocket, base64 (+33%), server decode + grid re-encode, and the
# vision tower's image preprocessing. 480 = 2x GBA native keeps dialog
# text and grid labels crisp while shrinking the PNG ~10-20x. Gemma's
# visual token budget resizes the input anyway, so nothing is lost
# model-side. 0 disables (settings: game_agent_frame_max_edge).
_DEFAULT_MAX_EDGE = 480


def downscale_frame(png: bytes, *, max_edge: int = _DEFAULT_MAX_EDGE) -> bytes:
    """Cap a frame's longest edge, preserving aspect ratio.

    NEAREST resampling on purpose: game frames are pixel art that was
    nearest-upscaled by the display path, so NEAREST inverts that
    losslessly-ish and keeps text edges hard (BILINEAR smears 8px GBA
    font glyphs into grey mush at these sizes). Returns the input
    unchanged when it already fits, when ``max_edge`` is 0/negative, or
    on any decode failure (fail-open — a big frame beats no frame).
    """

    if max_edge <= 0:
        return png
    try:
        with Image.open(io.BytesIO(png)) as im:
            w, h = im.size
            if max(w, h) <= max_edge:
                return png
            scale = max_edge / float(max(w, h))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out = im.convert("RGB").resize((nw, nh), Image.NEAREST)
            buf = io.BytesIO()
            out.save(buf, "PNG")
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("game_perception.downscale_failed", error=str(exc))
        return png


@dataclass
class PreparedFrames:
    """Result of :func:`prepare_frames`.

    ``frames`` is the processed PNG set in oldest→newest order, ready to
    hand to the LLM. ``note`` is a short string to splice into the prompt
    near the FRAMES line (grid legend, dedup disclosure) — empty when no
    annotation is warranted.
    """

    frames: list[bytes]
    note: str = ""
    # Diagnostics — handy for tests and logging, never sent to the model.
    n_in: int = 0
    n_unique: int = 0
    grid_applied: bool = False
    meta: dict = field(default_factory=dict)


def _fingerprint_bytes(png: bytes) -> bytes | None:
    """32×32 RGB fingerprint as a flat bytes object (len 3072)."""

    try:
        with Image.open(io.BytesIO(png)) as im:
            small = im.convert("RGB").resize((_FP_EDGE, _FP_EDGE), Image.BILINEAR)
            return small.tobytes()
    except Exception as exc:  # noqa: BLE001
        log.debug("game_perception.fingerprint_failed", error=str(exc))
        return None


def _max_channel_diff(a: bytes, b: bytes) -> int:
    """Max per-channel absolute difference (0–255) between two equal-length
    fingerprint byte strings. Sensitive to a single localized change (a
    cursor), robust to global low-amplitude noise (animation shimmer,
    re-encode jitter). Returns a huge value if the inputs can't be
    compared, so the caller keeps the frame (fail-open)."""

    if a is None or b is None or len(a) != len(b):
        return 1 << 30
    m = 0
    for x, y in zip(a, b, strict=False):
        d = x - y if x > y else y - x
        if d > m:
            m = d
    return m


# Visual-bucket geometry. The fingerprint (32×32) is averaged down to a
# 4×4 block grid and each channel quantized to 8 levels. The point is a
# key that is STABLE across within-scene motion but DISTINCT across
# scenes: a walking sprite or a moving cursor perturbs one block by a few
# units and lands in the same bucket, while a new room/level/screen moves
# most blocks by tens of units and lands in a new one.
#
# Why this exists: it is the only novelty dimension that does not
# saturate on a game we have no RAM probes for. ``screen`` is a closed
# 8-label vocabulary (TITLE/OVERWORLD/BATTLE/…) that runs out within
# minutes, and ``dialog`` only fires on text-bearing screens — so a
# platformer or puzzle game would otherwise report zero progress forever
# and hang the stall watchdog. Bucket count is unbounded and game-agnostic.
#
# Honest limitation: this is an approximation, not an identity. A scene
# whose block average sits exactly on a quantization boundary can flicker
# between two adjacent buckets and inflate the distinct count by a small
# constant. Coarse levels (8) keep boundaries rare relative to real scene
# changes; consumers should treat the count as an ordinal progress signal
# for comparing runs, never as an exact room census.
_BUCKET_EDGE = 4
_BUCKET_LEVELS = 8


def visual_bucket(fp: bytes | None, *, edge: int = _BUCKET_EDGE) -> str | None:
    """Coarse, unbounded visual-scene key derived from a fingerprint.

    Use when:
    - Recording visual novelty for a game with no RAM probes, so
      "have I ever seen this screen before?" is answerable from pixels.

    Expects:
    - ``fp`` is a :func:`_fingerprint_bytes` result (32×32 RGB, len 3072).
      Anything else returns ``None`` (fail-open: no bucket, no novelty
      claim — never a wrong one).

    Returns:
    - A short hex key, or ``None`` when no bucket could be computed.
    """

    if not fp or len(fp) != _FP_EDGE * _FP_EDGE * 3 or edge <= 0:
        return None
    block = _FP_EDGE // edge
    if block <= 0:
        return None
    n = block * block
    parts: list[str] = []
    for by in range(edge):
        for bx in range(edge):
            r = g = b = 0
            for y in range(by * block, (by + 1) * block):
                row = y * _FP_EDGE * 3
                for x in range(bx * block, (bx + 1) * block):
                    i = row + x * 3
                    r += fp[i]
                    g += fp[i + 1]
                    b += fp[i + 2]
            # (avg * levels) >> 8 maps 0–255 onto 0–(levels-1).
            qr = ((r // n) * _BUCKET_LEVELS) >> 8
            qg = ((g // n) * _BUCKET_LEVELS) >> 8
            qb = ((b // n) * _BUCKET_LEVELS) >> 8
            parts.append(f"{(qr << 6) | (qg << 3) | qb:03x}")
    return "".join(parts)


def dedup_frames(
    frames: list[bytes], *, threshold: int = _DEFAULT_DEDUP_THRESHOLD
) -> list[bytes]:
    """Drop frames that are near-identical to the previously-kept frame.

    Walks oldest→newest, keeping a frame only when its fingerprint differs
    from the last kept frame by more than ``threshold``. A fully static
    stack collapses to a single frame; a stack with real motion is left
    intact. Order is preserved. Always returns at least one frame when
    given a non-empty input (the newest survivor stands in for the run).
    """

    if len(frames) <= 1:
        return list(frames)
    kept: list[bytes] = []
    last_fp: bytes | None = None
    for f in frames:
        fp = _fingerprint_bytes(f)
        if not kept:
            kept.append(f)
            last_fp = fp
            continue
        if fp is None or last_fp is None:
            # Can't compare — keep it (fail-open).
            kept.append(f)
            last_fp = fp
            continue
        if _max_channel_diff(fp, last_fp) > threshold:
            kept.append(f)
            last_fp = fp
    return kept


def draw_grid(
    png: bytes,
    *,
    cols: int = _DEFAULT_GRID_COLS,
    rows: int = _DEFAULT_GRID_ROWS,
) -> bytes:
    """Overlay a labeled Set-of-Marks grid on a frame.

    Columns get letters (A, B, …), rows get numbers (1, 2, …); each cell
    is labeled in its top-left corner (e.g. ``C4``). Lines are thin and
    semi-transparent so they orient the model without occluding sprites.
    Returns a fresh PNG; on any failure returns the input unchanged
    (fail-open — a missing grid is better than a dropped frame).
    """

    try:
        with Image.open(io.BytesIO(png)) as base:
            img = base.convert("RGBA")
        w, h = img.size
        if w < cols or h < rows:
            return png
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        line = (255, 64, 200, 110)        # semi-transparent magenta
        label_fg = (255, 255, 255, 235)
        label_bg = (0, 0, 0, 150)
        cell_w = w / cols
        cell_h = h / rows
        for c in range(1, cols):
            x = int(round(c * cell_w))
            d.line([(x, 0), (x, h)], fill=line, width=1)
        for r in range(1, rows):
            y = int(round(r * cell_h))
            d.line([(0, y), (w, y)], fill=line, width=1)
        for c in range(cols):
            for r in range(rows):
                label = f"{_COL_LETTERS[c % len(_COL_LETTERS)]}{r + 1}"
                lx = int(round(c * cell_w)) + 1
                ly = int(round(r * cell_h)) + 1
                # Tiny label chip for legibility on any background.
                d.rectangle([lx, ly, lx + 6 * len(label), ly + 9], fill=label_bg)
                d.text((lx + 1, ly), label, fill=label_fg)
        out = Image.alpha_composite(img, overlay).convert("RGB")
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("game_perception.grid_failed", error=str(exc))
        return png


def _grid_legend(cols: int, rows: int) -> str:
    last_col = _COL_LETTERS[(cols - 1) % len(_COL_LETTERS)]
    return (
        f"A labeled grid is overlaid on each frame: columns A–{last_col} "
        f"(left→right), rows 1–{rows} (top→bottom). Refer to on-screen "
        f"positions by cell, e.g. 'the player is in C4'."
    )


def prepare_frames(
    frames: list[bytes],
    *,
    dedup: bool = True,
    dedup_threshold: int = _DEFAULT_DEDUP_THRESHOLD,
    grid: bool = True,
    grid_cols: int = _DEFAULT_GRID_COLS,
    grid_rows: int = _DEFAULT_GRID_ROWS,
    max_edge: int = _DEFAULT_MAX_EDGE,
) -> PreparedFrames:
    """Turn a raw frame stack into the prepared image set + prompt note.

    Use when:
    - The orchestrator has pulled a temporal frame window from the adapter
      and is about to hand it to the slow-path LLM.

    Pipeline (each step independently toggleable):
    0. ``max_edge`` — cap the longest edge (kill the display upscale).
    1. ``dedup`` — collapse redundant near-identical frames.
    2. ``grid``  — overlay a labeled Set-of-Marks grid on every survivor.

    Returns a :class:`PreparedFrames`; ``.frames`` replaces the raw list
    in the LLM call and ``.note`` is appended to the prompt's FRAMES
    section. Empty input → empty result (text-only turn), no work done.
    """

    n_in = len(frames)
    if n_in == 0:
        return PreparedFrames(frames=[], note="", n_in=0, n_unique=0)

    work = list(frames)
    if max_edge > 0:
        work = [downscale_frame(f, max_edge=max_edge) for f in work]
    if dedup:
        work = dedup_frames(work, threshold=dedup_threshold)
    n_unique = len(work)

    note_parts: list[str] = []
    # Frame-primacy correction (emitted whenever perception is active). The
    # base prompt tells the model to trust the LIVE_LOG over FRAMES —
    # correct when a RAM OVERLAY exists, but in vision-only mode the FRAME
    # is the only ground truth and the log's effect-score is a trap: it is
    # non-zero on animated screens (titles, menus, water tiles) even when
    # the input did nothing, so a model that reasons from it hallucinates
    # movement/progress. Steer it back to the pixels. (TODO(ram-era): drop
    # this once an OVERLAY is present — then log-over-frames is right again.)
    if dedup or grid:
        note_parts.append(
            "Read the FRAMES directly — they are your ground truth for what "
            "is on screen right now. 'effect_score'/'something changed' is "
            "UNRELIABLE: on animated screens (title, menus, water) it is "
            "non-zero even when your input did nothing, so never infer "
            "movement, progress, or which screen you are on from it — "
            "confirm everything against the frame."
        )
    grid_applied = False
    if grid:
        work = [draw_grid(f, cols=grid_cols, rows=grid_rows) for f in work]
        grid_applied = True
        note_parts.append(_grid_legend(grid_cols, grid_rows))

    if dedup and n_unique < n_in:
        # Tell the model the truth about the temporal window so it doesn't
        # hallucinate motion across what are really duplicate ticks.
        if n_unique == 1:
            note_parts.append(
                "The screen was static this window (duplicate frames "
                "collapsed to one) — treat it as a single still."
            )
        else:
            note_parts.append(
                f"{n_in - n_unique} duplicate frame(s) were collapsed; "
                f"the {n_unique} shown are the distinct states."
            )

    return PreparedFrames(
        frames=work,
        note=" ".join(note_parts),
        n_in=n_in,
        n_unique=n_unique,
        grid_applied=grid_applied,
        meta={"dedup": dedup, "grid": grid},
    )


# ── Extension point for the RAM era ───────────────────────────────────────
#
# When the emulator surface gains libretro memory access (a custom core
# build that exports retro_get_memory_data, or the native streamed path),
# the probe layer will produce a decoded world-state dict — player (x,y),
# facing, a collision/walkability grid, map id, party, dialog flags. At
# that point the fixed grid above becomes a *tile-aligned* grid centered on
# the player with walkable/blocked cells colored, which the ecosystem
# (Claude Plays Pokémon, GeminiPlaysPokemonLive) shows is the single
# biggest perception lever. The signature below is the seam; it stays a
# no-op shim until that data exists, so callers can wire it now.


def overlay_from_state(
    png: bytes,
    state: dict | None,
    *,
    cols: int = _DEFAULT_GRID_COLS,
    rows: int = _DEFAULT_GRID_ROWS,
) -> bytes:
    """Tile-aligned, RAM-driven grid overlay (extension point).

    Until RAM state is available (``state`` is ``None`` / lacks player
    coordinates), this falls back to the fixed :func:`draw_grid`. Once a
    decoded collision/position state flows in, this is where tile-aligned
    offsetting and walkability coloring will live — keeping the call site
    in the orchestrator unchanged across the transition.
    """

    if not state or "player_x" not in state or "player_y" not in state:
        return draw_grid(png, cols=cols, rows=rows)
    # TODO(ram-era): compute tile offset from (player_x, player_y), align
    # grid to game tiles, color cells by the RAM collision map.
    return draw_grid(png, cols=cols, rows=rows)


__all__ = [
    "PreparedFrames",
    "prepare_frames",
    "dedup_frames",
    "downscale_frame",
    "draw_grid",
    "overlay_from_state",
]
