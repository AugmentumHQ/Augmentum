"""Character card parser — extracts structured character data from all major formats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class CharacterCard:
    """Unified character representation extracted from any card format."""

    name: str = ""
    aliases: list[str] = field(default_factory=list)
    personality: str = ""
    appearance: str = ""
    visual_traits: str = ""  # AI-parsed or user-edited physical descriptors for image generation
    description: str = ""
    background: str = ""
    abilities: str = ""
    species: str = ""
    greeting: str = ""
    scenario: str = ""
    example_dialogue: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    depth_prompt: str = ""
    depth_prompt_depth: int = 4
    creator_notes: str = ""
    tags: list[str] = field(default_factory=list)
    image_model: str = ""
    image_style: str = ""  # User-selected art style for background generation
    source_format: str = "unknown"
    raw_data: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or "Unknown Character"

    @property
    def trait_summary(self) -> str:
        """Short summary of key traits for context injection."""
        parts = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.species:
            parts.append(f"Species: {self.species}")
        if self.personality:
            parts.append(f"Personality: {self.personality[:200]}")
        if self.appearance:
            parts.append(f"Appearance: {self.appearance[:200]}")
        if self.scenario:
            parts.append(f"Scenario: {self.scenario[:200]}")
        return "\n".join(parts)


class CardParser:
    """Parses character cards from system prompts into a unified CharacterCard."""

    def parse(self, system_prompt: str) -> CharacterCard | None:
        """Try all parsers in priority order, return the first successful parse."""
        # Strip Augmentum-injected sections before parsing so they don't
        # confuse format detectors (e.g. PList matching Name: from persona).
        card_text = re.sub(
            r"\[User/Player Character\]\s*\n.+?(?=\n\n|\Z)", "",
            system_prompt, flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        parsers = [
            self._parse_v2_json,
            self._parse_sillytavern_template,
            self._parse_wpp,
            self._parse_plist,
            self._parse_cai_kobold,
        ]

        for parser in parsers:
            result = parser(card_text)
            if result is not None:
                # Extract visual traits from [Visual Traits] section if not already set
                if not result.visual_traits:
                    result.visual_traits = self._extract_visual_traits_section(system_prompt)
                return result

        # No structured format matched — build a minimal card from the raw text
        # so that visual traits, personality, scenario etc. are still available
        # to the image distiller and context builder.
        return self._parse_freeform(system_prompt)

    @staticmethod
    def _extract_visual_traits_section(text: str) -> str:
        """Extract visual traits from a [Visual Traits] section in the prompt."""
        match = re.search(
            r"\[Visual Traits\]\s*\n(.+?)(?:\n\n|\n\[|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def _parse_v2_json(self, text: str) -> CharacterCard | None:
        """Parse Character Card V2 JSON format."""
        # Look for JSON containing spec: chara_card_v2
        if '"chara_card_v2"' not in text.lower() and '"spec"' not in text:
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text
            match = re.search(r'\{[^{}]*"spec"[^{}]*\}', text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        card_data = data.get("data", data)

        card = CharacterCard(
            name=card_data.get("name", ""),
            personality=card_data.get("personality", ""),
            description=card_data.get("description", ""),
            scenario=card_data.get("scenario", ""),
            greeting=card_data.get("first_mes", ""),
            example_dialogue=card_data.get("mes_example", ""),
            system_prompt=card_data.get("system_prompt", ""),
            post_history_instructions=card_data.get("post_history_instructions", ""),
            creator_notes=card_data.get("creator_notes", ""),
            tags=card_data.get("tags", []),
            source_format="v2_json",
            raw_data=data,
        )

        # Extract image_model, visual_traits, image_style from top-level or extensions
        card.image_model = card_data.get("image_model", "")
        card.visual_traits = card_data.get("visual_traits", "")
        card.image_style = card_data.get("image_style", "")

        # Extract additional fields from extensions
        extensions = card_data.get("extensions", {})
        if extensions:
            card.raw_data["extensions"] = extensions
            if not card.image_model:
                card.image_model = extensions.get("image_model", "")
            if not card.visual_traits:
                card.visual_traits = extensions.get("visual_traits", "")
            if not card.image_style:
                card.image_style = extensions.get("image_style", "")
            # depth_prompt lives in extensions (V2.1+)
            if extensions.get("depth_prompt"):
                card.depth_prompt = extensions["depth_prompt"]
                card.depth_prompt_depth = extensions.get("depth_prompt_depth", 4)

        # Character book
        char_book = card_data.get("character_book")
        if char_book:
            card.raw_data["character_book"] = char_book

        # Parse aliases from alternate_greetings or name variations
        alt_greetings = card_data.get("alternate_greetings", [])
        if alt_greetings:
            card.raw_data["alternate_greetings"] = alt_greetings

        return card

    def _parse_sillytavern_template(self, text: str) -> CharacterCard | None:
        """Parse SillyTavern {{char}}/{{user}} template format."""
        if "{{char}}" not in text.lower() and "{{user}}" not in text.lower():
            return None

        card = CharacterCard(source_format="sillytavern")

        # Try to extract character name from patterns like "{{char}} is <Name>"
        # or "{{char}}'s name is <Name>"
        name_patterns = [
            re.compile(r"\{\{char\}\}\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
            re.compile(r"\{\{char\}\}'s\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
            re.compile(r"Name:\s*([^\n]+)", re.IGNORECASE),
        ]
        for pattern in name_patterns:
            match = pattern.search(text)
            if match:
                card.name = match.group(1).strip().rstrip(".")
                break

        # Extract personality from known patterns
        personality_match = re.search(
            r"\{\{char\}\}'s\s+personality\s+(?:is|:)\s*([^\n]+(?:\n[^\n{]+)*)",
            text, re.IGNORECASE,
        )
        if personality_match:
            card.personality = personality_match.group(1).strip()

        # Extract scenario
        scenario_match = re.search(
            r"\{\{scenario\}\}:\s*([^\n]+(?:\n[^\n{]+)*)",
            text, re.IGNORECASE,
        )
        if scenario_match:
            card.scenario = scenario_match.group(1).strip()

        # Store the full template as description
        card.description = text
        card.raw_data["template_text"] = text

        return card

    def _parse_wpp(self, text: str) -> CharacterCard | None:
        """Parse W++ format: [character= "Name"] [personality= "trait" + "trait"]."""
        if "[character" not in text.lower() and "[personality" not in text.lower():
            return None

        card = CharacterCard(source_format="wpp")

        # Extract character name
        char_match = re.search(r'\[character\s*=\s*"([^"]+)"', text, re.IGNORECASE)
        if char_match:
            card.name = char_match.group(1)

        # Extract W++ fields
        field_pattern = re.compile(r'\[(\w+)\s*=\s*([^\]]+)\]', re.IGNORECASE)
        for match in field_pattern.finditer(text):
            field_name = match.group(1).lower()
            raw_value = match.group(2).strip()

            # Parse "value" + "value" format
            values = re.findall(r'"([^"]+)"', raw_value)
            value = ", ".join(values) if values else raw_value

            if field_name == "personality":
                card.personality = value
            elif field_name == "appearance":
                card.appearance = value
            elif field_name == "species":
                card.species = value
            elif field_name == "background":
                card.background = value
            elif field_name == "abilities":
                card.abilities = value
            elif field_name == "description":
                card.description = value

        card.raw_data["wpp_text"] = text
        return card

    def _parse_plist(self, text: str) -> CharacterCard | None:
        """Parse PList format: key: value style character descriptions."""
        # Require at least 3 PList-style fields
        plist_fields = re.findall(
            r"^(Personality|Appearance|Species|Background|Name|Abilities|Gender|Age|Occupation):\s*.+",
            text, re.MULTILINE | re.IGNORECASE,
        )
        if len(plist_fields) < 3:
            return None

        card = CharacterCard(source_format="plist")

        field_map = {
            "name": "name",
            "personality": "personality",
            "appearance": "appearance",
            "species": "species",
            "background": "background",
            "abilities": "abilities",
            "description": "description",
        }

        for match in re.finditer(
            r"^(\w[\w\s]*?):\s*(.+?)(?=\n\w[\w\s]*?:|\Z)",
            text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        ):
            field_name = match.group(1).strip().lower()
            value = match.group(2).strip()

            attr = field_map.get(field_name)
            if attr:
                setattr(card, attr, value)

        card.raw_data["plist_text"] = text
        return card

    def _parse_cai_kobold(self, text: str) -> CharacterCard | None:
        """Parse Character.AI / Kobold format with Name:, Greeting:, etc."""
        has_name = bool(re.search(r"^Name:\s*.+", text, re.MULTILINE))
        has_greeting = bool(re.search(r"^Greeting:\s*.+", text, re.MULTILINE | re.IGNORECASE))
        has_examples = bool(re.search(r"^Example\s+(?:Dialogue|Messages?):", text, re.MULTILINE | re.IGNORECASE))

        if not (has_name and (has_greeting or has_examples)):
            return None

        card = CharacterCard(source_format="cai")

        name_match = re.search(r"^Name:\s*(.+)", text, re.MULTILINE)
        if name_match:
            card.name = name_match.group(1).strip()

        greeting_match = re.search(
            r"^Greeting:\s*(.+?)(?=\n\w[\w\s]*?:|\Z)",
            text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if greeting_match:
            card.greeting = greeting_match.group(1).strip()

        examples_match = re.search(
            r"^Example\s+(?:Dialogue|Messages?):\s*(.+?)(?=\n\w[\w\s]*?:|\Z)",
            text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if examples_match:
            card.example_dialogue = examples_match.group(1).strip()

        # Try long description
        desc_match = re.search(
            r"^(?:Long\s+)?Description:\s*(.+?)(?=\n\w[\w\s]*?:|\Z)",
            text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if desc_match:
            card.description = desc_match.group(1).strip()

        card.raw_data["cai_text"] = text
        return card

    def _parse_freeform(self, text: str) -> CharacterCard | None:
        """Last-resort parser for plain-text system prompts.

        Extracts [Visual Traits], Personality:, Scenario:, and
        [User/Player Character] sections from unstructured text so
        the image distiller and context builder still have data.
        Returns None only if the text is too short to be useful.
        """
        stripped = text.strip()
        if len(stripped) < 20:
            return None

        # Skip generic assistant prompts — only parse if there's character-specific
        # content like visual traits, personality fields, or bracketed sections.
        has_character_signals = (
            "[Visual Traits]" in text
            or re.search(r"^Personality:", text, re.MULTILINE | re.IGNORECASE)
            or re.search(r"^Scenario:", text, re.MULTILINE | re.IGNORECASE)
            or re.search(r"^Appearance:", text, re.MULTILINE | re.IGNORECASE)
        )
        if not has_character_signals:
            return None

        card = CharacterCard(source_format="freeform")

        # Strip Augmentum-injected persona block before extracting fields
        # so Name:/Appearance: from the user persona don't contaminate the card.
        char_text = re.sub(
            r"\[User/Player Character\]\s*\n.+?(?=\n\n|\Z)", "",
            text, flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Visual traits — bracketed section
        card.visual_traits = self._extract_visual_traits_section(text)

        # Personality: line
        m = re.search(
            r"^Personality:\s*(.+?)(?=\n\n|\n\w[\w\s]*?:|\Z)",
            char_text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if m:
            card.personality = m.group(1).strip()

        # Scenario: line
        m = re.search(
            r"^Scenario:\s*(.+?)(?=\n\n|\n\w[\w\s]*?:|\Z)",
            char_text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if m:
            card.scenario = m.group(1).strip()

        # Name: line (if present)
        m = re.search(r"^Name:\s*(.+)", char_text, re.MULTILINE | re.IGNORECASE)
        if m:
            card.name = m.group(1).strip()

        # Strip [Visual Traits] block from description to avoid duplication
        desc = re.sub(
            r"\[Visual Traits\]\s*\n.+?(?=\n\n|\Z)", "", char_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if desc:
            card.description = desc

        card.raw_data["freeform_text"] = text
        return card
