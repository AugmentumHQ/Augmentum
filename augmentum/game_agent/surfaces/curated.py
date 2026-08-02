"""Curated (Cradle-style) surface adapter (scaffold).

Targets any game (or any application, really) that does not expose a
privileged API. The adapter operates the way Cradle operates: screen
capture for observation, synthetic OS-level input events for action,
and optional OCR over the captured frames for cheap text-only
observations.

This is the lowest-common-denominator surface. Use it when nothing
better is available -- the trade-off is that the slow-path agent
carries most of the cognitive load, because the log channel has only
whatever OCR + region-change heuristics yield.

Wire protocol (curated.v1)
--------------------------
Action emission uses platform-native synthetic input:

* Linux: ``xdotool key`` / ``xdotool mousemove`` subprocess.
* Windows: ``pyautogui`` / Win32 ``SendInput`` (TODO).
* macOS: ``pyautogui`` / Quartz event taps (TODO).

Observation:

* Frame: screen-capture region defined by ``capture_rect``.
* OCR (optional): a deterministic OCR pass over the frame; diffs
  against the previous OCR result are emitted as
  ``EventPayload(channel="ocr", data={"added":[...], "removed":[...]})``.

Implementation status
---------------------
Scaffold only. Resolver bindings + caps are correct; ``start()``
raises until the capture + input transports are wired.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from augmentum.game_agent.schema import SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn


@dataclass(frozen=True)
class CaptureRect:
    """Screen region to capture (origin top-left, pixels)."""

    x: int
    y: int
    width: int
    height: int


class CuratedAdapter:
    """Curated (Cradle-style) screen + synthetic-input adapter (scaffold)."""

    def __init__(
        self,
        *,
        semantic_to_input: Mapping[str, str],
        capture_rect: CaptureRect | None = None,
        ocr_enabled: bool = False,
        log_schema: str = "curated.v1",
    ) -> None:
        """Construct a curated adapter.

        Parameters
        ----------
        semantic_to_input:
            Maps semantic ids to wire-format input strings the platform
            adapter understands -- e.g. ``"jump": "space"`` for an
            ``xdotool key`` binding, or ``"talk": "mouse:left:640,360"``
            for a mouse-click at a fixed pixel.
        capture_rect:
            Screen region to capture for frames. ``None`` disables
            vision.
        ocr_enabled:
            If True, run OCR over each captured frame and emit diffs
            on the ``ocr`` channel.
        """

        self._semantic_to_input = dict(semantic_to_input)
        self._capture_rect = capture_rect
        self._ocr_enabled = ocr_enabled
        self._log_schema = log_schema
        self._resolver = SemanticInputResolver()
        for semantic in self._semantic_to_input:
            self._resolver.bind(semantic, self._make_synthetic_input(semantic))

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        modalities: list[str] = []
        if self._capture_rect is not None:
            modalities.append("frame")
        if self._ocr_enabled:
            modalities.append("ocr")
        if not modalities:
            # A surface must declare at least one observation channel;
            # default to "log" even though curated rarely has one.
            modalities.append("log")
        return SurfaceCapsPayload(
            semantic_inputs=list(self._semantic_to_input.keys()),
            log_schema=self._log_schema,
            observation_modalities=modalities,  # type: ignore[arg-type]
        )

    async def start(self, emit: EmitEventFn) -> None:
        if self._capture_rect is None and not self._ocr_enabled:
            raise NotImplementedError(
                "CuratedAdapter: no observation channel configured. "
                "Provide capture_rect and/or set ocr_enabled=True."
            )
        # TODO: start a frame-capture loop; if ocr_enabled, run OCR per
        # frame and emit diffs through emit(...).
        raise NotImplementedError("CuratedAdapter capture + input TODO")

    async def stop(self) -> None:
        # TODO: cancel capture loop.
        return None

    async def snapshot_frame(self) -> bytes | None:
        # TODO: synchronous screen-capture of self._capture_rect, PNG-encode.
        return None

    def _make_synthetic_input(self, semantic: str):  # type: ignore[no-untyped-def]
        async def _resolver(duration_ms: int) -> None:
            # TODO: parse self._semantic_to_input[semantic] and dispatch
            # via xdotool / pyautogui / Win32 / Quartz depending on
            # platform. Honor duration_ms for held-input bindings.
            _ = duration_ms
            _ = self._semantic_to_input[semantic]

        return _resolver
