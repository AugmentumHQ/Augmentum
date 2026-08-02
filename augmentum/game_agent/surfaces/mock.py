"""Scripted mock adapter for tests and offline demos.

The mock implements :class:`SurfaceAdapter` against an in-memory
script: a list of ``(delay_ms_since_start, EventPayload)`` tuples that
get emitted on schedule. Resolver invocations are recorded so tests
can assert action emission.

This is the adapter every other module's unit test runs against. It
also doubles as a reference implementation when authoring a new real
adapter: read this first.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from augmentum.game_agent.schema import EventPayload, SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn


@dataclass
class ScriptedEvent:
    """One scheduled mock observation."""

    delay_ms: int
    payload: EventPayload


_DEFAULT_SEMANTICS = ("noop", "advance", "back", "confirm")


class MockAdapter:
    """Real adapter against a scripted event stream.

    Use when:
    - Unit-testing the orchestrator, fast path, or slow path without
      booting a real game.
    - Demoing the end-to-end pipeline with no external deps.

    Expects:
    - ``script`` is sorted by ``delay_ms`` ascending; the adapter does
      not re-sort.
    - ``semantic_inputs`` lists the semantic ids this mock pretends to
      accept. Resolver bindings are no-ops that just record calls.

    Returns:
    - Recorded inputs are available via :attr:`recorded_inputs` after
      the session; useful for ``assert``s.
    """

    def __init__(
        self,
        *,
        script: list[ScriptedEvent] | None = None,
        semantic_inputs: tuple[str, ...] = _DEFAULT_SEMANTICS,
        frame_bytes: bytes | None = None,
        log_schema: str = "mock.v1",
        observation_modalities: tuple[str, ...] = ("log",),
    ) -> None:
        self._script: list[ScriptedEvent] = list(script or [])
        self._semantic_inputs = list(semantic_inputs)
        self._frame_bytes = frame_bytes
        self._log_schema = log_schema
        self._observation_modalities = list(observation_modalities)
        self._resolver = SemanticInputResolver()
        self._recorded_inputs: list[tuple[str, int]] = []
        for semantic in self._semantic_inputs:
            self._resolver.bind(semantic, self._make_record_resolver(semantic))
        self._emit_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    # ── SurfaceAdapter Protocol implementation ────────────────────

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        return SurfaceCapsPayload(
            semantic_inputs=self._semantic_inputs,
            log_schema=self._log_schema,
            observation_modalities=self._observation_modalities,  # type: ignore[arg-type]
        )

    async def start(self, emit: EmitEventFn) -> None:
        self._stopped.clear()
        self._emit_task = asyncio.create_task(self._emit_loop(emit), name="mock-emit")

    async def stop(self) -> None:
        self._stopped.set()
        if self._emit_task is not None:
            self._emit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._emit_task
            self._emit_task = None

    async def snapshot_frame(self) -> bytes | None:
        return self._frame_bytes

    # ── Test helpers ──────────────────────────────────────────────

    @property
    def recorded_inputs(self) -> list[tuple[str, int]]:
        """All resolver invocations as ``(semantic, duration_ms)`` tuples."""

        return list(self._recorded_inputs)

    def append_to_script(self, event: ScriptedEvent) -> None:
        """Push a new event for the emit loop to pick up.

        Useful when a test wants to react to an agent's action by
        injecting a follow-up observation. Append while the session is
        running and the emit loop will deliver it as long as
        ``delay_ms`` is still in the future.
        """

        self._script.append(event)

    # ── Internals ─────────────────────────────────────────────────

    async def _emit_loop(self, emit: EmitEventFn) -> None:
        # We re-read self._script every iteration so tests can
        # append_to_script() while the session is running.
        emitted: set[int] = set()
        origin = asyncio.get_event_loop().time()
        while not self._stopped.is_set():
            now_ms = int((asyncio.get_event_loop().time() - origin) * 1000)
            next_due_idx: int | None = None
            next_due_delay: int | None = None
            for idx, ev in enumerate(self._script):
                if idx in emitted:
                    continue
                if ev.delay_ms <= now_ms:
                    await emit(ev.payload)
                    emitted.add(idx)
                elif next_due_delay is None or ev.delay_ms < next_due_delay:
                    next_due_delay = ev.delay_ms
                    next_due_idx = idx
            if next_due_idx is None:
                # Nothing more scheduled; idle.
                await asyncio.sleep(0.05)
                continue
            assert next_due_delay is not None
            wait_s = max(0.001, (next_due_delay - now_ms) / 1000.0)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=wait_s)
            except TimeoutError:
                continue

    def _make_record_resolver(self, semantic: str) -> Any:
        async def _resolver(duration_ms: int) -> None:
            self._recorded_inputs.append((semantic, duration_ms))

        return _resolver
