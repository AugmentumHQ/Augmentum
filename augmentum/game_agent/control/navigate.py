"""Navigation macro compiler ("navigate_to" quickaction).

The collapsed premise behind every successful Pokémon-playing harness
(Claude Plays Pokémon, Gemini Plays Pokémon): **the model emits spatial
intent; the harness executes it deterministically.** Overworld walking
via individual d-pad presses is the dominant waste class for an LLM
agent — the 71%-dead-press audit here matched what both frontier teams
found independently, and both shipped the same answer: a pathfinding
primitive over the collision map read from RAM.

One plan action ``{"s": "navigate_to", "text": "12,8"}`` (map-tile
coordinates) or ``{"text": "down 5"}`` (relative) compiles into the
primitive press sequence via BFS over the ``walk_grid`` probe — the
same shape as :mod:`text_entry`'s ``type_text``.

Correctness posture: the compiled walk is *best-effort convergent*, not
guaranteed. The first press may spend frames turning the player, NPCs
move, and the grid window is finite — so the worker verifies the end
position afterwards and reports reached/short via the normal fx path;
the model re-issues and converges. Unreachable targets path to the
closest reachable tile instead (progress over perfection).
"""

from __future__ import annotations

import re
from collections import deque

from augmentum.game_agent.schema import PlanAction

# Profiles whose probe preset ships a walk_grid. Mirrors
# text_entry.LAYOUTS: presence here is what unlocks the quickaction.
NAV_PROFILES: frozenset[str] = frozenset({"pokemon_emerald"})

# One tile of Gen-3 walking is 16 frames ≈ 267ms. Per-tile presses keep
# distance deterministic (a hold's tile count drifts with turn frames).
_TILE_PRESS_MS = 240
# Hard cap on compiled path length — beyond this the window data is too
# stale to trust; the model re-issues from the new position.
_MAX_PATH_TILES = 24

_DIR_SEMANTIC = {
    (0, -1): "nav_up",
    (0, 1): "nav_down",
    (-1, 0): "nav_left",
    (1, 0): "nav_right",
}

_REL_RE = re.compile(r"^\s*(up|down|left|right)\s+(\d{1,3})\s*$", re.IGNORECASE)
_ABS_RE = re.compile(r"^\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*$")


def has_navigation(game_profile: str | None) -> bool:
    return bool(game_profile) and game_profile in NAV_PROFILES


def parse_nav_target(text: str, px: int, py: int) -> tuple[int, int] | None:
    """Resolve the action's ``text`` to absolute map-tile coordinates.

    Accepts ``"x,y"`` (absolute, the coordinate space of the player_x/
    player_y probes) or ``"<up|down|left|right> N"`` (relative).
    """

    m = _ABS_RE.match(text or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _REL_RE.match(text or "")
    if m:
        n = int(m.group(2))
        dx, dy = {
            "up": (0, -n), "down": (0, n), "left": (-n, 0), "right": (n, 0),
        }[m.group(1).lower()]
        return px + dx, py + dy
    return None


def extract_landmarks(grid: dict, px: int, py: int) -> dict[str, tuple[int, int]]:
    """Name the walkable exits of the current window — symbols, not math.

    A 2B actor cannot derive "(12,8)" from an ASCII map; it CAN pick
    "exit_north" from a list. For each window side, BFS-reachable
    walkable boundary cells are grouped into contiguous runs and the
    midpoint of the largest run becomes ``exit_<side>``. Pure geometry,
    zero game knowledge; recomputed identically at delta-build time (to
    advertise names) and at action time (to resolve them).
    """

    rows = grid.get("rows") or []
    x0 = int(grid.get("x0", 0))
    y0 = int(grid.get("y0", 0))
    h = len(rows)
    w = len(rows[0]) if h else 0
    if not h or not w:
        return {}

    def walkable(x: int, y: int) -> bool:
        cx, cy = x - x0, y - y0
        return 0 <= cy < h and 0 <= cx < w and rows[cy][cx] == "."

    # Reachability from the player (start tile exempt, as in the walker).
    seen = {(px, py)}
    queue: deque[tuple[int, int]] = deque([(px, py)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DIR_SEMANTIC:
            nxt = (cx + dx, cy + dy)
            if nxt not in seen and walkable(*nxt):
                seen.add(nxt)
                queue.append(nxt)

    sides: dict[str, list[tuple[int, int]]] = {
        "north": [(x0 + i, y0) for i in range(w)],
        "south": [(x0 + i, y0 + h - 1) for i in range(w)],
        "west": [(x0, y0 + j) for j in range(h)],
        "east": [(x0 + w - 1, y0 + j) for j in range(h)],
    }
    out: dict[str, tuple[int, int]] = {}
    for side, cells in sides.items():
        runs: list[list[tuple[int, int]]] = []
        cur: list[tuple[int, int]] = []
        for cell in cells:
            if cell in seen:
                cur.append(cell)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        if runs:
            best = max(runs, key=len)
            out[f"exit_{side}"] = best[len(best) // 2]
    return out


def resolve_nav_target(
    text: str, grid: dict, px: int, py: int,
) -> tuple[int, int] | None:
    """Resolve any accepted target form: "x,y", "down 5", or a landmark."""

    target = parse_nav_target(text, px, py)
    if target is not None:
        return target
    return extract_landmarks(grid, px, py).get((text or "").strip().lower())


def compile_navigation(
    grid: dict, px: int, py: int, tx: int, ty: int,
) -> tuple[list[PlanAction], tuple[int, int]]:
    """BFS a collision-safe path and compile it to per-tile presses.

    ``grid`` is the walk_grid probe value: ``{"x0", "y0", "rows"}`` with
    rows of ``.`` (walkable) / ``#`` (blocked) / ``?`` (out of bounds),
    in the same map-tile coordinate space as the player probes.

    Returns ``(actions, expected_end)``. When the exact target is
    unreachable or outside the window, paths to the reachable tile
    closest to it; when the player already stands there (or nothing is
    reachable), returns ``([], (px, py))``.
    """

    rows = grid.get("rows") or []
    x0 = int(grid.get("x0", 0))
    y0 = int(grid.get("y0", 0))
    h = len(rows)
    w = len(rows[0]) if h else 0

    def walkable(x: int, y: int) -> bool:
        cx, cy = x - x0, y - y0
        return 0 <= cy < h and 0 <= cx < w and rows[cy][cx] == "."

    # BFS from the player over walkable tiles. The start tile itself is
    # exempt from the walkability test (the player is standing on it —
    # e.g. a door tile the grid marks specially).
    start = (px, py)
    prev: dict[tuple[int, int], tuple[int, int]] = {start: start}
    queue: deque[tuple[int, int]] = deque([start])
    best: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None  # (manhattan-to-target, hops)
    hops: dict[tuple[int, int], int] = {start: 0}
    while queue:
        cur = queue.popleft()
        d_target = abs(cur[0] - tx) + abs(cur[1] - ty)
        key = (d_target, hops[cur])
        if best_key is None or key < best_key:
            best, best_key = cur, key
        if cur == (tx, ty):
            break
        if hops[cur] >= _MAX_PATH_TILES:
            continue
        for dx, dy in _DIR_SEMANTIC:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in prev or not walkable(*nxt):
                continue
            prev[nxt] = cur
            hops[nxt] = hops[cur] + 1
            queue.append(nxt)

    if best is None or best == start:
        return [], start

    # Reconstruct start→best.
    path: list[tuple[int, int]] = []
    node = best
    while node != start:
        path.append(node)
        node = prev[node]
    path.reverse()

    actions = [
        PlanAction(
            semantic=_DIR_SEMANTIC[(b[0] - a[0], b[1] - a[1])],
            duration_ms=_TILE_PRESS_MS,
        )
        for a, b in zip([start, *path], path, strict=False)
    ]
    return actions, best


__all__ = [
    "NAV_PROFILES",
    "compile_navigation",
    "extract_landmarks",
    "has_navigation",
    "parse_nav_target",
    "resolve_nav_target",
]
