"""Companion persona — the identity a slow-path agent speaks *as*.

Phase 1 added a generic companion mode: the planner could set a ``say``
field that got routed through a bridge to a TTS provider. The persona
was anonymous (no name, no character, default voice).

This module adds the persona slot. When a session is started with a
character_id, the route layer loads the character card and builds a
:class:`CompanionPersona`, which the orchestrator threads in two
directions:

1. To the agent's prompt: an IDENTITY block is prepended above the
   strict planner rules so the model speaks in first person as the
   named character.
2. To the voice bridge: ``voice`` overrides the default voice on every
   :meth:`VoiceBridge.synthesize` call for this session, so the
   companion has a distinctive timbre rather than the operator's
   default.

The dataclass is intentionally tiny and frozen — it carries identity,
nothing else. Episodic memory, idle behavior, banter timing (Phase 6)
will be separate modules that consume the same persona.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompanionPersona:
    """Identity for a companion-mode session.

    Use when:
    - The route layer has loaded a character card and is constructing
      an :class:`Orchestrator`. Pass the persona alongside
      ``companion=True`` to personalize prompt + voice.

    All fields are optional. An empty persona is still valid — the
    agent falls back to the generic addendum and the default TTS voice,
    which is the Phase 1 behavior.
    """

    #: Display name the companion uses for itself in dialogue. ``""``
    #: means the prompt does not introduce a named identity (generic
    #: addendum only).
    name: str = ""

    #: Short personality / persona summary that gets prepended to the
    #: slow-path prompt. Keep under ~600 characters; the prompt budget
    #: is tight. Empty means no persona block.
    persona: str = ""

    #: TTS voice identifier passed to :meth:`VoiceBridge.synthesize`.
    #: Accepts plain voice names (e.g. ``"af_heart"``) or explicit
    #: provider-prefixed names (e.g. ``"qwen-tts::Vivian"``). Empty
    #: means "use the bridge's default voice", which itself may be
    #: empty (= "use the TTS provider's default").
    voice: str = ""

    @property
    def has_identity(self) -> bool:
        """True iff the persona has anything worth injecting into the prompt."""

        return bool(self.name or self.persona)


def build_identity_prefix(persona: CompanionPersona | None) -> str:
    """Render the IDENTITY block that gets prepended above the strict prompt.

    Returns ``""`` when no persona is supplied or the persona is empty.
    Putting this above the planner rules (rather than appending it
    inside the COMPANION addendum) keeps the planner rules the LAST
    instructions the model reads, which is where strict-JSON contracts
    have to live to survive.

    Before:
    - persona = None or empty
    After:
    - returns ""

    Before:
    - persona = CompanionPersona(name="Aria", persona="Curious, encouraging")
    After:
    - returns "IDENTITY\\nYou are Aria. Persona: Curious, encouraging.\\n..."
    """

    if persona is None or not persona.has_identity:
        return ""

    lines: list[str] = ["IDENTITY"]
    if persona.name:
        lines.append(f"You are {persona.name}.")
    if persona.persona:
        # One blank line of separation lets the model parse the persona
        # paragraph as a single contiguous block of prose without
        # confusing it for an instruction list.
        lines.append(f"Persona: {persona.persona}")
    lines.append(
        "Speak about yourself in first person. The 'say' utterances "
        "(when you choose to speak) should sound like you, not like a "
        "neutral narrator. Stay in character; do not break the fourth "
        "wall to reference 'the model', 'the game-agent', or 'the AI'."
    )
    return "\n".join(lines) + "\n\n"


__all__ = ["CompanionPersona", "build_identity_prefix"]
