"""Base class for primitive capabilities.

Primitives are stateless one-call capabilities. Each primitive class
exposes a single ``call(ctx, **kwargs)`` entry point. The kwargs shape
is documented per-primitive — Sprint 2 keeps them loosely typed so
each adapter can match the underlying service's natural call shape.

If a capability is inherently streaming or stateful (TTS chunk pump,
STT realtime feed), the primitive yields async iterators from
``call()``; the registry doesn't care.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.runtime import CompanionRuntime


@dataclass(frozen=True, slots=True)
class PrimitiveContext:
    """Per-call context. Lightweight by design."""
    runtime: CompanionRuntime
    bus: PresenceBus
    companion_id: str
    user_id: str = ""
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class PrimitiveResult:
    """A single primitive call result. ``payload`` is service-shaped;
    callers know the schema per-primitive."""
    ok: bool
    payload: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PrimitiveBase(ABC):
    """Abstract base for a primitive capability."""

    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(f"{cls.__name__}: subclass must set `name`")

    @abstractmethod
    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        """Execute the capability. Implementations should catch and
        return errors via ``PrimitiveResult(ok=False, error=...)`` so
        callers never need to wrap in try/except."""
        ...


__all__ = ["PrimitiveBase", "PrimitiveContext", "PrimitiveResult"]
