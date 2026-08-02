"""Semantic input resolution.

The agent emits actions in a surface-agnostic vocabulary
(``"jump"``, ``"attack"``, ``"move_north"``, etc.); the surface
adapter binds each semantic id to a callable that produces the
wire-format input (a key event, a gamepad press, a Lua RPC, a
``xdotool key`` invocation, ...).

This module owns the binding registry and provides the small set of
helpers needed to construct, query, and invoke bindings.

Design intent
-------------
Bindings are *flat data*, not code. A binding is conceptually
``semantic_id -> (kind, target, modifiers)`` where the adapter
interprets ``(kind, target, modifiers)``. We keep them as opaque
Python callables here for ergonomics, but each adapter ships a
declarative table that drives those callables -- the agent and the
authoring layer never see code, only the table.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

ResolverFn: TypeAlias = Callable[[int], Awaitable[None]]
"""A bound resolver. Takes ``duration_ms`` and emits the input asynchronously.

The duration is the *intended* hold time for the input; press-and-release
adapters may ignore it and emit a tap, while held-input adapters use it
verbatim.
"""

ChordResolverFn: TypeAlias = Callable[[list[str], int], Awaitable[None]]
"""A chord resolver: several semantics held SIMULTANEOUSLY for one duration.

The first list element is the primary input (it names the chord in logs
and acks); the rest are held alongside it. Adapters that can't press
multiple inputs at once simply never bind one — callers check
:attr:`SemanticInputResolver.supports_chord` and degrade to sequential
presses.
"""


class UnknownSemanticError(KeyError):
    """Raised when the agent emits a semantic id that is not bound."""

    def __init__(self, semantic: str, known: list[str]) -> None:
        self.semantic = semantic
        self.known = known
        super().__init__(
            f"semantic input {semantic!r} is not bound; known: {sorted(known)}"
        )


class SemanticInputResolver:
    """Maps surface-agnostic semantic ids to surface-specific resolvers.

    Use when:
    - A surface adapter is being constructed and needs to declare which
      semantic ids it accepts.
    - The orchestrator is about to apply a :class:`PlanAction` and needs
      to look up the matching resolver.

    Expects:
    - Bindings are registered up-front, typically in the adapter
      constructor. Mid-session re-binding is allowed but rare.

    Returns:
    - From :meth:`semantic_inputs`, the stable list that goes into
      :class:`SurfaceCapsPayload`; this is the *only* vocabulary the
      slow-path agent may emit.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, ResolverFn] = {}
        self._chord_resolver: ChordResolverFn | None = None

    def bind(self, semantic: str, resolver: ResolverFn) -> None:
        """Register or replace a binding.

        ``semantic`` must be ``[a-z0-9_]+`` and non-empty -- it ends up
        in the agent prompt, so we keep it stable and machine-readable.
        """

        if not semantic or not all(c.isalnum() or c == "_" for c in semantic):
            raise ValueError(
                f"semantic id {semantic!r} must be [a-z0-9_]+ and non-empty"
            )
        self._bindings[semantic] = resolver

    def has(self, semantic: str) -> bool:
        return semantic in self._bindings

    def semantic_inputs(self) -> list[str]:
        """The list of bound semantic ids, sorted for stability."""

        return sorted(self._bindings.keys())

    async def apply(self, semantic: str, duration_ms: int) -> None:
        """Resolve and invoke. Raises :class:`UnknownSemanticError` on miss."""

        resolver = self._bindings.get(semantic)
        if resolver is None:
            raise UnknownSemanticError(semantic, list(self._bindings.keys()))
        await resolver(duration_ms)

    # ── chords ────────────────────────────────────────────────────

    def bind_chord(self, resolver: ChordResolverFn) -> None:
        """Register the adapter's simultaneous-press dispatcher."""

        self._chord_resolver = resolver

    @property
    def supports_chord(self) -> bool:
        return self._chord_resolver is not None

    async def apply_chord(self, semantics: list[str], duration_ms: int) -> None:
        """Hold several bound semantics simultaneously for one duration.

        Every member must be individually bound (same vocabulary rule as
        :meth:`apply`). Raises :class:`UnknownSemanticError` on any miss
        and ``RuntimeError`` when the adapter never bound a chord path —
        callers should check :attr:`supports_chord` first and fall back
        to sequential :meth:`apply` calls.
        """

        if self._chord_resolver is None:
            raise RuntimeError("surface adapter does not support chorded input")
        for semantic in semantics:
            if semantic not in self._bindings:
                raise UnknownSemanticError(semantic, list(self._bindings.keys()))
        await self._chord_resolver(list(semantics), duration_ms)
