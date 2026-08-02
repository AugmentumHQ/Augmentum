"""WorldInfoBuffer — assembles scan text from multiple sources for WI matching."""

from __future__ import annotations

from dataclasses import dataclass, field

_BOUNDARY = "\x01"


@dataclass
class WorldInfoBuffer:
    """Collects chat messages and metadata fields, then produces a single scan
    string for World Info keyword matching."""

    chat_messages: list[str] = field(default_factory=list)
    persona_description: str = ""
    char_description: str = ""
    char_personality: str = ""
    scenario: str = ""
    creator_notes: str = ""
    extension_prompts: list[str] = field(default_factory=list)

    _recursion_buffer: list[str] = field(default_factory=list, repr=False)
    _depth_skew: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scan_text(
        self,
        scan_depth: int,
        *,
        include_persona: bool = False,
        include_char_description: bool = False,
        include_char_personality: bool = False,
        include_scenario: bool = False,
        include_creator_notes: bool = False,
        include_recursion: bool = True,
    ) -> str:
        """Return a boundary-delimited string of all requested scan sources."""
        segments: list[str] = []

        # Chat messages (newest-first, up to effective depth).
        effective_depth = scan_depth + self._depth_skew
        for msg in self.chat_messages[:effective_depth]:
            if msg:
                segments.append(msg)

        # Optional global metadata fields.
        if include_persona and self.persona_description:
            segments.append(self.persona_description)
        if include_char_description and self.char_description:
            segments.append(self.char_description)
        if include_char_personality and self.char_personality:
            segments.append(self.char_personality)
        if include_scenario and self.scenario:
            segments.append(self.scenario)
        if include_creator_notes and self.creator_notes:
            segments.append(self.creator_notes)

        # Extension prompts.
        for ep in self.extension_prompts:
            if ep:
                segments.append(ep)

        # Recursion buffer (content from previously-activated entries).
        if include_recursion:
            for rb in self._recursion_buffer:
                if rb:
                    segments.append(rb)

        return _BOUNDARY.join(segments)

    def add_to_recursion_buffer(self, content: str) -> None:
        """Append activated entry content so later passes can match against it."""
        self._recursion_buffer.append(content)

    def advance_scan(self) -> None:
        """Widen effective scan depth by 1 (used for min-activation retries)."""
        self._depth_skew += 1

    def reset_skew(self) -> None:
        """Reset depth skew back to zero."""
        self._depth_skew = 0
