"""TinyQuest — a real, tiny, deterministic game used as an eval case.

A four-room overworld behind a title screen. The player walks a grid,
bumps into walls, passes through doors into visually distinct rooms, and
can talk to an NPC. It is small, but it is a *game*, not a script: state
advances only in response to input, and the same inputs always produce
the same run.

Why it exists
-------------
The scoring spine in :mod:`augmentum.game_agent.progress` claims to
measure progress on a game it knows nothing about. TinyQuest is how that
claim gets tested for real, offline and unattended:

* **No RAM probes.** No preset decodes its memory, so nothing hands the
  agent coordinates or a screen label. Progress has to be read off the
  pixels or not at all.
* **A real title gate.** The game does not start until ``confirm`` is
  pressed. An agent that mashes the wrong button sits on the title
  screen forever — which is precisely the "loaded but never played"
  case the score is supposed to refuse to reward.
* **Real input effect.** Every input reports a genuine ``effect_score``
  measured by diffing the frame before and after, the same ground-truth
  signal the browser bridge computes. Walking into a wall really does
  score near zero.
* **Visually distinct places.** Each room has its own palette, so
  reaching a new room is a real visual event rather than a labelled one.

Determinism is deliberate: there is no randomness anywhere, so a
before/after comparison reflects the agent's change and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

import structlog
from PIL import Image, ImageDraw

from augmentum.game_agent.schema import EventPayload, SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn

log = structlog.get_logger(__name__)

CELL = 16
COLS, ROWS = 15, 10
WIDTH, HEIGHT = COLS * CELL, ROWS * CELL

# Semantic vocabulary. Intentionally the universal set from
# control/actions.py — TinyQuest teaches nothing game-specific.
SEMANTIC_INPUTS = (
    "nav_up",
    "nav_down",
    "nav_left",
    "nav_right",
    "confirm",
    "cancel",
)

# Door cells, one per edge. Stepping onto one moves you to the linked
# room and lands you at the facing door.
_DOOR_N = (COLS // 2, 0)
_DOOR_S = (COLS // 2, ROWS - 1)
_DOOR_W = (0, ROWS // 2)
_DOOR_E = (COLS - 1, ROWS // 2)

# room_id -> {edge: (target_room, spawn_cell)}
_LINKS: dict[int, dict[str, tuple[int, tuple[int, int]]]] = {
    0: {"E": (1, _DOOR_W), "S": (2, _DOOR_N)},
    1: {"W": (0, _DOOR_E), "S": (3, _DOOR_N)},
    2: {"N": (0, _DOOR_S), "E": (3, _DOOR_W)},
    3: {"W": (2, _DOOR_E), "N": (1, _DOOR_S)},
}

# Palettes chosen to be far apart so a room change is unmistakable in a
# coarse visual bucket, the way a real game's areas differ.
_PALETTES: dict[int, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    0: ((34, 92, 48), (18, 48, 26)),      # green field
    1: ((150, 122, 70), (86, 68, 36)),    # sand
    2: ((44, 62, 140), (22, 32, 78)),     # water cavern
    3: ((110, 44, 110), (60, 22, 60)),    # purple keep
}

_TITLE_BG = (12, 12, 28)
_NPC_ROOM, _NPC_CELL = 1, (7, 4)
_NPC_LINES = (
    "The keep lies south of the sands.",
    "Mind the walls, traveller.",
    "You have come far already.",
)


class TinyQuest:
    """The game itself: state, rules, and rendering. No I/O, no asyncio.

    Kept free of transport concerns so it can be stepped synchronously
    in a unit test and reasoned about like the state machine it is.
    """

    def __init__(self) -> None:
        self.started = False
        self.room = 0
        self.px, self.py = COLS // 2, ROWS // 2
        self.dialog: str = ""
        self.npc_talks = 0
        self.steps = 0
        self.rooms_visited: set[int] = {0}

    # ── rules ─────────────────────────────────────────────────────

    def _is_wall(self, x: int, y: int) -> bool:
        """Border cells are walls unless they are this room's doors."""

        on_border = x in (0, COLS - 1) or y in (0, ROWS - 1)
        if not on_border:
            # One interior pillar per room so there is something to bump
            # into away from the edges.
            return (x, y) == (4, 3)
        return (x, y) not in self._doors()

    def _doors(self) -> set[tuple[int, int]]:
        edges = _LINKS.get(self.room, {})
        cells = set()
        for edge in edges:
            cells.add(
                {"N": _DOOR_N, "S": _DOOR_S, "W": _DOOR_W, "E": _DOOR_E}[edge]
            )
        return cells

    def _edge_of(self, x: int, y: int) -> str | None:
        for edge, cell in (
            ("N", _DOOR_N), ("S", _DOOR_S), ("W", _DOOR_W), ("E", _DOOR_E)
        ):
            if (x, y) == cell and edge in _LINKS.get(self.room, {}):
                return edge
        return None

    def press(self, semantic: str) -> None:
        """Apply one input. The only way state ever changes."""

        # The title gate. Only confirm starts the game; everything else
        # is a genuinely dead press.
        if not self.started:
            if semantic == "confirm":
                self.started = True
            return

        if self.dialog:
            # Any press dismisses an open dialog box.
            self.dialog = ""
            return

        if semantic == "confirm":
            if self.room == _NPC_ROOM and self._adjacent_to_npc():
                self.dialog = _NPC_LINES[self.npc_talks % len(_NPC_LINES)]
                self.npc_talks += 1
            return

        deltas = {
            "nav_up": (0, -1),
            "nav_down": (0, 1),
            "nav_left": (-1, 0),
            "nav_right": (1, 0),
        }
        if semantic not in deltas:
            return  # cancel and anything unknown do nothing in the field
        dx, dy = deltas[semantic]
        nx, ny = self.px + dx, self.py + dy
        if not (0 <= nx < COLS and 0 <= ny < ROWS):
            return
        edge = self._edge_of(nx, ny)
        if edge is not None:
            target, spawn = _LINKS[self.room][edge]
            self.room = target
            self.px, self.py = spawn
            self.rooms_visited.add(target)
            self.steps += 1
            return
        if self._is_wall(nx, ny):
            return  # bump — a real dead press
        self.px, self.py = nx, ny
        self.steps += 1

    def _adjacent_to_npc(self) -> bool:
        nx, ny = _NPC_CELL
        return abs(self.px - nx) + abs(self.py - ny) <= 1

    # ── rendering ─────────────────────────────────────────────────

    def render(self) -> bytes:
        """Current frame as PNG bytes. Deterministic for a given state."""

        if not self.started:
            im = Image.new("RGB", (WIDTH, HEIGHT), _TITLE_BG)
            d = ImageDraw.Draw(im)
            d.rectangle([30, 54, WIDTH - 30, 74], fill=(220, 200, 90))
            d.rectangle([70, 96, WIDTH - 70, 104], fill=(90, 90, 130))
            return _encode(im)

        bg, wall = _PALETTES[self.room]
        im = Image.new("RGB", (WIDTH, HEIGHT), bg)
        d = ImageDraw.Draw(im)
        for y in range(ROWS):
            for x in range(COLS):
                if self._is_wall(x, y):
                    d.rectangle(
                        [x * CELL, y * CELL, x * CELL + CELL - 1, y * CELL + CELL - 1],
                        fill=wall,
                    )
        for cx, cy in self._doors():
            d.rectangle(
                [cx * CELL, cy * CELL, cx * CELL + CELL - 1, cy * CELL + CELL - 1],
                fill=(230, 220, 170),
            )
        if self.room == _NPC_ROOM:
            nx, ny = _NPC_CELL
            d.ellipse(
                [nx * CELL + 2, ny * CELL + 2, nx * CELL + CELL - 3, ny * CELL + CELL - 3],
                fill=(240, 120, 120),
            )
        d.rectangle(
            [
                self.px * CELL + 3, self.py * CELL + 3,
                self.px * CELL + CELL - 4, self.py * CELL + CELL - 4,
            ],
            fill=(250, 250, 250),
        )
        if self.dialog:
            d.rectangle([8, HEIGHT - 46, WIDTH - 8, HEIGHT - 8], fill=(245, 245, 235))
            d.rectangle(
                [8, HEIGHT - 46, WIDTH - 8, HEIGHT - 8], outline=(30, 30, 30), width=2
            )
            d.text((16, HEIGHT - 38), self.dialog[:40], fill=(20, 20, 20))
        return _encode(im)


def _encode(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class TinyQuestAdapter:
    """:class:`SurfaceAdapter` over :class:`TinyQuest`.

    Reports ``frame`` as its only observation modality and emits NO
    probes — the agent gets pixels and nothing else, which is the whole
    point of using this as an eval case.
    """

    def __init__(self, *, log_schema: str = "tinyquest.v1") -> None:
        self.game = TinyQuest()
        self._log_schema = log_schema
        self._resolver = SemanticInputResolver()
        self._emit: EmitEventFn | None = None
        self._inputs: list[tuple[str, int]] = []
        self._tick_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        for semantic in SEMANTIC_INPUTS:
            self._resolver.bind(semantic, self._make_binding(semantic))

    # ── SurfaceAdapter Protocol ───────────────────────────────────

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        return SurfaceCapsPayload(
            semantic_inputs=list(SEMANTIC_INPUTS),
            log_schema=self._log_schema,
            observation_modalities=["log", "frame"],
        )

    async def start(self, emit: EmitEventFn) -> None:
        self._emit = emit
        self._stopped.clear()
        self._tick_task = asyncio.create_task(self._tick_loop(), name="tinyquest-tick")

    async def stop(self) -> None:
        self._stopped.set()
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None

    async def snapshot_frame(self) -> bytes | None:
        return self.game.render()

    # ── inspection (tests / reporting) ────────────────────────────

    @property
    def recorded_inputs(self) -> list[tuple[str, int]]:
        return list(self._inputs)

    # ── internals ─────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Heartbeat so the session has a log channel even while idle."""

        while not self._stopped.is_set():
            if self._emit is not None:
                with contextlib.suppress(Exception):
                    await self._emit(
                        EventPayload(
                            channel="log",
                            data={"event": "tick", "started": self.game.started},
                        )
                    )
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=0.25)

    def _make_binding(self, semantic: str) -> Any:
        async def _apply(duration_ms: int) -> None:
            from augmentum.game_agent.perception import (
                _fingerprint_bytes,
                _max_channel_diff,
            )

            before = _fingerprint_bytes(self.game.render())
            self.game.press(semantic)
            after = _fingerprint_bytes(self.game.render())
            # Real ground truth, measured the same way the browser bridge
            # measures it: did the screen actually change?
            effect = (
                _max_channel_diff(before, after)
                if before is not None and after is not None
                else 0
            )
            effect = min(effect, 255)
            self._inputs.append((semantic, duration_ms))
            if self._emit is not None:
                await self._emit(
                    EventPayload(
                        channel="log",
                        data={
                            "event": "input_ack",
                            "button": semantic,
                            "effect_score": int(effect),
                        },
                    )
                )

        return _apply


__all__ = [
    "COLS",
    "HEIGHT",
    "ROWS",
    "SEMANTIC_INPUTS",
    "WIDTH",
    "TinyQuest",
    "TinyQuestAdapter",
]
