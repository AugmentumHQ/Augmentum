"""Base class + protocol for subagent adapters.

A *subagent* is a stateful, conversation-shaped persona that wraps an
existing mode orchestrator (passthrough, analytical, agentic, …). The
runtime dispatches an :class:`augmentum.companion_runtime.runtime.Intent`
to exactly one subagent per turn.

Distinction from primitives:
- Subagents may issue multiple LLM turns and own conversation state
  for the duration of the invocation.
- Primitives are stateless capabilities (TTS, embed, browse) and
  return on a single call.

Lifecycle: subagents are instantiated once at import time (registered
into :class:`SubagentRegistry`), but each ``invoke`` call is independent
— no shared mutable state across calls. Callers serialize through the
runtime, so concurrent ``invoke``s for the same subagent shouldn't
happen except via tests or background tick (Sprint 4a).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent


@dataclass(frozen=True, slots=True)
class SubagentContext:
    """Per-invocation context handed to a subagent.

    Sprint 2 keeps this minimal — Sprint 3's dispatcher will enrich
    with retrieval results, persona kernel digest, focus rank.
    """
    intent: Intent
    runtime: CompanionRuntime
    bus: PresenceBus
    companion_id: str
    invocation_id: str = ""        # filled by dispatcher; stable id for bus events


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """A single invocation's output."""
    content: str
    handled_by: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""                # non-empty when the subagent failed cleanly


class SubagentBase(ABC):
    """Abstract base for a mode-wrapping subagent.

    Subclasses MUST set ``name`` (unique key) and ``description``
    (short — used by Sprint 3 lexical scoring).
    """

    name: str = ""
    description: str = ""
    # Soft compatibility hints used by Sprint 3 dispatch ranking. None of
    # these are enforced; the dispatcher just adds utility for matches.
    role_affinity: tuple[str, ...] = ()        # eg ("collaborator", "host")
    focus_affinity: tuple[str, ...] = ()       # eg ("owner", "world")
    state_affinity: tuple[str, ...] = ()       # eg ("attentive", "working")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(f"{cls.__name__}: subclass must set `name`")

    @abstractmethod
    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        """Handle one intent and return a result.

        Implementations should emit ``subagent.invoked`` and
        ``subagent.completed`` bus events for observability. The
        dispatcher (Sprint 3) emits the outer ``dispatch.*`` envelope;
        adapters only emit their own inner events.
        """
        ...


__all__ = ["SubagentBase", "SubagentContext", "SubagentResult"]
