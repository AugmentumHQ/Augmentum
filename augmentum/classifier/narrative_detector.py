"""Narrative mode detector — identifies character cards and RP patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from augmentum.models.base import InternalChatRequest

# --- Pattern definitions ---

# SillyTavern template variables
SILLYTAVERN_PATTERNS = [
    (re.compile(r"\{\{char\}\}", re.IGNORECASE), 0.35, "sillytavern_char"),
    (re.compile(r"\{\{user\}\}", re.IGNORECASE), 0.25, "sillytavern_user"),
    (re.compile(r"\{\{personality\}\}", re.IGNORECASE), 0.20, "sillytavern_personality"),
    (re.compile(r"\{\{scenario\}\}", re.IGNORECASE), 0.20, "sillytavern_scenario"),
    (re.compile(r"\{\{mesExamples?\}\}", re.IGNORECASE), 0.15, "sillytavern_examples"),
    (re.compile(r"\{\{description\}\}", re.IGNORECASE), 0.15, "sillytavern_description"),
]

# W++ format (SillyTavern character cards)
WPP_PATTERNS = [
    (re.compile(r"\[character\s*=", re.IGNORECASE), 0.40, "wpp_character"),
    (re.compile(r"\[personality\s*=", re.IGNORECASE), 0.30, "wpp_personality"),
    (re.compile(r'\"[^"]+\"\s*\+\s*\"[^"]+\"'), 0.15, "wpp_concat"),
]

# PList format (key: value style character descriptions)
PLIST_PATTERNS = [
    (re.compile(r"^Personality:\s*.+", re.MULTILINE | re.IGNORECASE), 0.25, "plist_personality"),
    (re.compile(r"^Appearance:\s*.+", re.MULTILINE | re.IGNORECASE), 0.20, "plist_appearance"),
    (re.compile(r"^Species:\s*.+", re.MULTILINE | re.IGNORECASE), 0.20, "plist_species"),
    (re.compile(r"^Background:\s*.+", re.MULTILINE | re.IGNORECASE), 0.15, "plist_background"),
    (re.compile(r"^Abilities:\s*.+", re.MULTILINE | re.IGNORECASE), 0.15, "plist_abilities"),
]

# Character.AI / Kobold format
CAI_PATTERNS = [
    (re.compile(r"^Name:\s*.+", re.MULTILINE), 0.15, "cai_name"),
    (re.compile(r"^Greeting:\s*.+", re.MULTILINE | re.IGNORECASE), 0.25, "cai_greeting"),
    (re.compile(r"^Example\s+(?:Dialogue|Messages?):", re.MULTILINE | re.IGNORECASE), 0.20, "cai_examples"),
    (re.compile(r"^Long\s+Description:", re.MULTILINE | re.IGNORECASE), 0.15, "cai_long_desc"),
]

# Character Card V2 JSON markers
V2_JSON_PATTERNS = [
    (re.compile(r'"spec":\s*"chara_card_v2"', re.IGNORECASE), 0.50, "v2_spec"),
    (re.compile(r'"first_mes":', re.IGNORECASE), 0.25, "v2_first_mes"),
    (re.compile(r'"mes_example":', re.IGNORECASE), 0.20, "v2_mes_example"),
    (re.compile(r'"creator_notes":', re.IGNORECASE), 0.15, "v2_creator_notes"),
    (re.compile(r'"character_book":', re.IGNORECASE), 0.20, "v2_character_book"),
    (re.compile(r'"system_prompt":', re.IGNORECASE), 0.10, "v2_system_prompt"),
]

# General RP/narrative structural indicators
RP_STRUCTURAL_PATTERNS = [
    (re.compile(r"\*[^*]+\*"), 0.10, "action_asterisks"),  # *action text*
    (re.compile(r"<[A-Z][a-z]+>"), 0.10, "xml_char_tag"),  # <CharName>
    (re.compile(r"You (?:are|play|roleplay as|act as)\s", re.IGNORECASE), 0.25, "rp_instruction"),
    (re.compile(r"(?:Stay in character|maintain character|don't break character)", re.IGNORECASE), 0.30, "stay_in_char"),
    (re.compile(r"(?:OOC|out of character|parentheses for)", re.IGNORECASE), 0.20, "ooc_reference"),
    (re.compile(r"(?:Write|respond|reply)\s+(?:as|like|in the style of)\s", re.IGNORECASE), 0.20, "write_as"),
    (re.compile(r"(?:setting|scenario|scene):\s*", re.IGNORECASE), 0.15, "scene_setting"),
]

# All pattern groups
ALL_PATTERN_GROUPS = [
    ("sillytavern", SILLYTAVERN_PATTERNS),
    ("wpp", WPP_PATTERNS),
    ("plist", PLIST_PATTERNS),
    ("cai", CAI_PATTERNS),
    ("v2_json", V2_JSON_PATTERNS),
    ("rp_structural", RP_STRUCTURAL_PATTERNS),
]


@dataclass
class NarrativeDetection:
    confidence: float
    reason: str
    metadata: dict = field(default_factory=dict)


class NarrativeDetector:
    """Detects narrative/roleplay content from system prompts and message patterns."""

    def detect(self, request: InternalChatRequest) -> NarrativeDetection:
        """Analyze a request for narrative mode indicators.

        Returns a detection result with confidence 0.0-1.0.
        """
        system_prompt = self._extract_system_prompt(request)
        if not system_prompt:
            return NarrativeDetection(confidence=0.0, reason="no system prompt")

        total_score = 0.0
        matched_patterns: list[str] = []
        matched_groups: set[str] = set()

        for group_name, patterns in ALL_PATTERN_GROUPS:
            for regex, weight, pattern_name in patterns:
                if regex.search(system_prompt):
                    total_score += weight
                    matched_patterns.append(pattern_name)
                    matched_groups.add(group_name)

        # Bonus for matching patterns from multiple format groups
        if len(matched_groups) >= 2:
            total_score += 0.15
        if len(matched_groups) >= 3:
            total_score += 0.10

        # Additional heuristic: very long system prompts (>500 chars) with any RP pattern
        # are more likely narrative
        if len(system_prompt) > 500 and matched_patterns:
            total_score += 0.10

        # Cap at 1.0
        confidence = min(total_score, 1.0)

        if not matched_patterns:
            return NarrativeDetection(confidence=0.0, reason="no narrative patterns found")

        # Build reason string
        top_patterns = matched_patterns[:5]
        reason = f"narrative patterns detected: {', '.join(top_patterns)}"
        if len(matched_patterns) > 5:
            reason += f" (+{len(matched_patterns) - 5} more)"

        return NarrativeDetection(
            confidence=confidence,
            reason=reason,
            metadata={
                "matched_patterns": matched_patterns,
                "matched_groups": sorted(matched_groups),
                "system_prompt_length": len(system_prompt),
            },
        )

    def _extract_system_prompt(self, request: InternalChatRequest) -> str:
        """Extract system prompt(s) from message list."""
        parts = []
        for msg in request.messages:
            if msg.role == "system":
                parts.append(msg.content)
        return "\n".join(parts)
