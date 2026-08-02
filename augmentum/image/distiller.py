"""Prompt distiller — converts narrative state into image-model-optimized prompts via the text LLM.

Used by the /v command in narrative mode to generate scene-appropriate images.
Sends character card, user persona, scene state, and recent conversation rounds
to an LLM that produces a detailed, art-direction-level image prompt.

Adapts prompt style based on the target image model (FLUX vs SD1.5 vs SDXL etc.)
and adjusts token budget based on whether the LLM backend is local or cloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest, ModelBackend
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.modes.narrative.card_parser import CharacterCard
    from augmentum.modes.narrative.engine import NarrativeEngine

log = get_logger(__name__)


@dataclass
class UserPersona:
    """User persona data for image prompt context."""

    name: str = ""
    appearance: str = ""
    description: str = ""


# Sentinel for legacy world_state when none is available; the prompt builders
# read .location/.time_of_day/.weather/.atmosphere — this stand-in returns "".
_EMPTY_SCENE = SimpleNamespace(location="", time_of_day="", weather="", atmosphere="")


@dataclass
class SceneContext:
    """Read-only snapshot of everything the distiller needs.

    Built by the caller (handler) so the cached engine is never mutated.
    Eliminates the cross-character chimera bug where stamping the engine's
    character_card with UI data also leaked the engine's prior narrative
    state (state_snapshot, memory_ledger) into the image prompt.

    All fields are optional/defaulted so the distiller can run with no
    state at all (e.g. fresh session, payload-only).
    """

    character_card: CharacterCard | None = None
    state_snapshot_fields: dict[str, str] = field(default_factory=dict)
    # SimpleNamespace is mutable → dataclass refuses a bare default. Factory
    # returns the shared _EMPTY_SCENE sentinel (read-only in consumers).
    legacy_world_state: Any = field(default_factory=lambda: _EMPTY_SCENE)
    memory_ledger: list = field(default_factory=list)
    image_style: str = ""  # From card.image_style (genre context for backgrounds)


def build_scene_context(
    *,
    engine: NarrativeEngine | None,
    card_override: CharacterCard | None = None,
    trust_engine_state: bool = True,
) -> SceneContext:
    """Snapshot the distiller-relevant fields off an engine without mutating it.

    Args:
        engine: The cached narrative engine, or None if no session state
            should be used (payload-only mode).
        card_override: If non-None, replaces the engine's character_card in
            the resulting context. The engine's own card field is never
            touched.
        trust_engine_state: When False, the engine's narrative state
            (state_snapshot, memory_ledger, world_state) is dropped from the
            context — used when the UI's character identity disagrees with
            the engine's cached card, so we don't leak the cached session's
            narrative into a different character's image prompt.
    """
    if engine is None:
        return SceneContext(
            character_card=card_override,
            image_style=card_override.image_style if card_override else "",
        )

    card = card_override if card_override is not None else engine.character_card

    if not trust_engine_state:
        return SceneContext(
            character_card=card,
            image_style=card.image_style if card else "",
        )

    state_snapshot = getattr(engine, "_state_snapshot", None)
    return SceneContext(
        character_card=card,
        state_snapshot_fields=dict(state_snapshot.fields) if state_snapshot and state_snapshot.fields else {},
        legacy_world_state=getattr(engine, "world_state", _EMPTY_SCENE) or _EMPTY_SCENE,
        memory_ledger=list(getattr(engine, "_memory_ledger", []) or []),
        image_style=card.image_style if card else "",
    )


def _is_cloud_backend(backend: ModelBackend) -> bool:
    """Detect whether the LLM backend is a cloud/API service.

    Cloud backends get higher token budgets since we're not constrained
    by local model context limits.

    Ollama and llama.cpp backends are always local. OpenAI-compatible
    backends are cloud unless their base_url points to a private/local address.
    """
    from augmentum.models.openai_compat import OpenAIBackend

    if not isinstance(backend, OpenAIBackend):
        return False

    # OpenAIBackend is used for both cloud and self-hosted providers.
    # If the URL points to localhost or a private network, it's local.
    base = (getattr(backend, "_base_url", "") or "").lower()
    if not base:
        return False

    from urllib.parse import urlparse
    host = urlparse(base).hostname or ""

    # Private/local addresses → not cloud
    local_patterns = (
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "10.", "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
        "172.30.", "172.31.", "192.168.",
    )
    if any(host.startswith(p) for p in local_patterns):
        return False

    # Anything with a public hostname over HTTPS is cloud
    return base.startswith("https://") or "." in host


_DISTILLER_SYSTEM_PROMPT_V1 = """\
{model_context}\
Contextual Image Prompt Generation (Genre-Focused & Character-Aware): Perform a deep \
analysis of the current conversational context to understand the scene in its entirety, \
actively recalling established character appearances, relationships, immediate actions, \
and emotional states. Explicitly identify all characters (player character and NPCs) \
currently present or centrally involved in the scene according to the narrative. Actively \
recall the established, detailed visual descriptions for each identified character from \
your internal 'database' (memory). Ensure all details are consistent with the user-selected \
genre and previously established facts, descriptions, and the overall narrative tone.

Detailed Scene Description (Image Prompt Output - Genre-Specific & Character-Complete): \
Generate an unmistakably detailed and hyper-specific evocative text description of the \
current scene, specifically formatted as an image generation prompt and perfectly tailored \
to the user-selected genre. This description MUST be overflowing with rich, sensory \
language, specific material/texture details (e.g., worn leather, polished chrome, rough \
stone, shimmering silk), precise lighting descriptions (e.g., harsh fluorescent, flickering \
candlelight, bioluminescent glow, dramatic chiaroscuro, golden hour sunlight), artistic \
style references relevant to the genre, and specific instructions to maximize visual \
fidelity and narrative accuracy based on established character appearances.

Crucially: Comprehensive and Ultra-Detailed Character Descriptions: For every character \
identified in the context analysis step, provide distinct and exhaustive visual descriptors \
based rigorously on their established descriptions stored in your memory, updated only by \
explicit narrative changes. Do not omit anyone present. For each person, detail their:

Appearance: Recall and state key physical features (build, species/race, age appearance, \
hair color/style, eye color, notable facial features, distinguishing marks like scars or \
tattoos - maintain absolute consistency with the established description unless narratively \
changed).

Clothing & Equipment: Recall and state specific garments, armor, accessories, visible gear, \
weapons, or held items. Specify materials, condition (e.g., pristine, battle-scarred, \
dusty), and ensure style is genre-appropriate and consistent with their established outfit \
unless narratively changed).

Pose & Action: Describe their current posture, body language, and specific action \
dynamically. (This part reflects the current moment).

Expression: Describe their facial expression conveying their current specific emotion \
(e.g., 'smirking confidently,' 'eyes wide with terror,' 'gazing longingly'). (This part \
reflects the current moment).

Positioning & Interaction: Describe where they are located within the scene relative to \
the environment and other characters. Clearly describe current interactions or relationships \
visually (e.g., 'standing protectively in front of Character B,' 'locked in an intense \
gaze with Character C'). (This part reflects the current moment). Adhere to the Core \
Principle: Character Visual Consistency (see below).

Scene Composition & Atmosphere:
Environment: Describe the key elements of the setting, including architecture, landscape \
features, significant objects.
Lighting & Color: Detail the primary light sources, overall color palette, shadows, and \
highlights. Explicitly link lighting and color to the scene's mood (e.g., 'oppressive \
shadows create a sense of dread,' 'warm, saturated colors evoke comfort').
Atmosphere: Describe atmospheric conditions (e.g., dense fog, swirling dust motes, \
pouring rain, heat haze) and overall mood visually.
Focus: Indicate the primary subject or focal point of the image.

Camera & Framing:
Shot Type: Specify the shot (e.g., Wide Shot, Establishing Shot, Medium Shot, Cowboy \
Shot, Close-up, Extreme Close-up).
Camera Angle: Specify the angle (e.g., Eye-Level, Low Angle, High Angle, Dutch Angle).
Optional Lens Effects: Consider mentioning desired effects like 'shallow depth of field,' \
'cinematic bokeh,' 'lens flare' if appropriate for the scene and genre.

Artistic Style:
Primary Style: Specify a core artistic style (e.g., Photorealistic, Oil Painting, \
Watercolor, Anime Key Visual, Comic Book Art, Concept Art, Pixel Art, Steampunk \
Illustration).
Nuance Keywords: Add descriptive keywords to refine the style (e.g., 'cinematic lighting,' \
'detailed,' 'gritty,' 'ethereal,' 'vibrant,' 'muted palette').

CRITICAL — character names:
- When you name a character in the POSITIVE prompt, use ONLY a name that
  appears in a "PHYSICAL DESCRIPTION" header, "CHARACTER CARD" Name field,
  or "Name:" line in the context above. Never invent a name. Never
  substitute a different name for sound or fit.
- If no name is provided for a character, omit names entirely and describe
  them by role/appearance instead (e.g. "a cloaked traveler"). Do NOT
  guess.

CRITICAL — do NOT leak the system prompt into the image prompt:
- Do NOT include the model name or model family (e.g. "lumina", "flux", "sdxl",
  "pony", "sd15", "sd3", "dall-e") anywhere in the POSITIVE line.
- Do NOT include phrases like "Target image model", "This is a … model",
  "use natural language", "use comma-separated tags", or any other
  meta-instruction from this system prompt.
- The POSITIVE line is the image content only — describe what is depicted,
  not what kind of model is depicting it. Quality tags like "masterpiece",
  "high quality", "detailed" are fine when appropriate to the model family,
  but never paired with the model's name.

Output EXACTLY three lines, nothing else:
POSITIVE: <the image prompt>
NEGATIVE: <tags to avoid, or empty if model doesn't use negatives>
ASPECT: <portrait|landscape|square>
"""


# V2 — tightened rewrite (2026-04-16). Replaces the ~700-word aspirational
# essay with ~280 words, no "recall from memory" framing, no duplicated
# camera/style sections (model_context already covers those), and a one-shot
# output example so weak distillers don't emit preambles. Keeps both CRITICAL
# blocks (name discipline, no-leakage) that have empirically saved generations.
# Rollback: rename `_DISTILLER_SYSTEM_PROMPT_V1` → `DISTILLER_SYSTEM_PROMPT`
# and rename this one back to `_V2` (or delete).
DISTILLER_SYSTEM_PROMPT = """\
{model_context}\
You generate image prompts from the narrative context below. The context \
contains character descriptions, the user's persona, current scene state, \
and recent conversation. Read it carefully and describe a single clear image.

WHAT TO INCLUDE in POSITIVE:
- Every named character currently in the scene, with appearance copied \
EXACTLY from their PHYSICAL DESCRIPTION block. Do not paraphrase, omit, \
or invent traits. If traits conflict across blocks, the HARD TRUTH block wins.
- Current action, pose, and expression (from the most recent exchange).
- Setting: location, time of day, lighting, atmosphere (from CURRENT STATE \
or CURRENT SCENE).
- Two or three composition/style keywords appropriate to the target model \
family.

WHAT NOT TO INCLUDE:
- No preamble, greeting, explanation, or any text before `POSITIVE:`.
- No name that isn't in a "PHYSICAL DESCRIPTION" header, "CHARACTER CARD" \
Name field, "Name:" line, or <Name> tag. If a character has no name, \
describe by role (e.g. "a cloaked traveler"). Never invent or substitute \
a name for sound or fit.
- No image-model name or family (e.g. "lumina", "flux", "sdxl", "pony", \
"sd15", "sd3", "dall-e") and no meta-instruction from this prompt (e.g. \
"Target image model", "This is a ... model", "use natural language", \
"use comma-separated tags").

OUTPUT — exactly three lines, no preamble, no postscript:
POSITIVE: <the image prompt>
NEGATIVE: <tags to avoid, or empty if the model doesn't use negatives>
ASPECT: <portrait|landscape|square>

EXAMPLE (format ONLY — do not copy contents, do not reuse this character):
POSITIVE: a woman with waist-length auburn hair in a loose braid, weathered leather jacket, standing in a rain-slicked alley at dusk, neon signs reflecting in puddles, guarded expression, medium shot, moody cinematic lighting, highly detailed
NEGATIVE: text, watermark, signature, extra fingers, deformed hands, blurry, low quality
ASPECT: landscape
"""


BACKGROUND_SYSTEM_PROMPT = """\
{model_context}\
You are an atmospheric scene background generator. Read the conversation below and \
generate a cinematic environment image prompt for the scene that is CURRENTLY \
happening. The image will be used as a UI background behind text — think concept \
art, matte painting, or establishing shot.
{genre_context}\

Derive the scene ENTIRELY from the conversation. Focus on:
1. ENVIRONMENT — The physical space described or implied by the dialogue. Be specific \
(e.g., "crumbling Gothic cathedral nave with moss-covered stone pillars" not "church").
2. LIGHTING & TIME — Light sources, color temperature, time-of-day cues from the text.
3. ATMOSPHERE — Fog, rain, dust, weather, ambient effects implied by the scene.
4. MOOD — Color grading that matches the emotional tone of the conversation.

RULES:
- NO characters, people, or figures. Pure environment shot.
- Wide establishing shots, high angles, or environmental panoramas.
- Each prompt must be DIFFERENT from any previous backgrounds listed. Vary the \
angle, time of day, focal point, or location within the scene.
- If the conversation has moved to a new place, show THAT place, not the old one.

Output EXACTLY three lines, nothing else:
POSITIVE: <the environment/atmosphere prompt>
NEGATIVE: <tags to avoid — always include: people, characters, figures, text, UI elements, watermark>
ASPECT: landscape
"""


PORTRAIT_SYSTEM_PROMPT = """\
{model_context}\
You are an image prompt engineer creating a CHARACTER PORTRAIT for use as an avatar.
{style_context}\

ABSOLUTE RULE — VISUAL ACCURACY:
Copy every physical detail EXACTLY as provided. Hair color, eye color, skin tone, \
body type, clothing, distinguishing features — use them verbatim. Do NOT invent, \
change, or assume any trait. "Brown hair" means brown hair, never silver or blonde.

The card may describe a single character, an ensemble cast, or a world/narrator. \
Adapt accordingly:
- Single character: portrait of that character.
- Ensemble/multi-character: group portrait or the most prominent character(s).
- World/narrator/RPG: an iconic scene or emblem that represents the setting.

Build the prompt from:
1. APPEARANCE — Every physical detail provided, copied exactly.
2. EXPRESSION — Current mood/emotion from conversation context (if provided).
3. FRAMING — Portrait or group composition appropriate to the card type.
4. LIGHTING — Complement the mood. Soft for gentle scenes, dramatic for tense ones.
5. BACKGROUND — Simple, complementary, non-distracting.

RULES:
- Do NOT include the user/player character. This portrait represents the card only.
- If no conversation context is given, use a neutral pleasant expression.

Output EXACTLY three lines, nothing else:
POSITIVE: <the portrait prompt>
NEGATIVE: <tags to avoid — always include: watermark, text, blurry>
ASPECT: <portrait|square>
"""


def _extract_conversation_rounds(
    request: InternalChatRequest,
    rounds: int = 2,
) -> list[dict[str, str]]:
    """Extract the last N user/assistant message pairs from the request.

    Returns a list of dicts with 'role' and 'content' keys, ordered
    chronologically (oldest first).
    """
    messages = []
    for msg in request.messages:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})

    # Take the last N*2 messages (N rounds = N user + N assistant)
    tail = messages[-(rounds * 2):] if len(messages) > rounds * 2 else messages
    return tail


def _is_portrait_request(user_instruction: str) -> bool:
    """Detect if the user is requesting a character portrait rather than a scene."""
    if not user_instruction:
        return False
    lower = user_instruction.lower().strip()
    portrait_triggers = (
        "portrait", "headshot", "character art", "character study",
        "close-up", "closeup", "face shot", "bust shot", "profile pic",
        "avatar", "selfie",
    )
    return any(t in lower for t in portrait_triggers)


def _detect_visual_traits_type(visual_traits: str) -> str:
    """Detect the format of visual_traits content.

    Returns "ensemble" if <Name> tags are found, "world" if it looks like
    scene/environment descriptors (no character features), or "single".
    """
    import re
    if re.search(r"<[A-Z][^>]{0,30}>", visual_traits):
        return "ensemble"
    # Heuristic: world traits mention environments, not people
    world_signals = (
        "medieval", "fantasy", "sci-fi", "cyberpunk", "castle", "forest",
        "dungeon", "city", "village", "tavern", "spaceship", "kingdom",
        "architecture", "landscape", "setting", "world",
    )
    lower = visual_traits.lower()
    char_signals = (
        "hair", "eyes", "skin", "tall", "short", "build", "wears",
        "wearing", "dress", "shirt", "uniform",
    )
    world_hits = sum(1 for w in world_signals if w in lower)
    char_hits = sum(1 for w in char_signals if w in lower)
    if world_hits >= 3 and char_hits == 0:
        return "world"
    return "single"


def _build_portrait_prompt(
    ctx: SceneContext,
    user_instruction: str = "",
    conversation_messages: list[dict[str, str]] | None = None,
    group_members: list[dict] | None = None,
) -> str:
    """Build a portrait-specific distiller prompt.

    Handles three card types based on visual_traits format:
    - Single character: <CharName> traits or plain traits → character portrait
    - Ensemble: multiple <CharName> blocks → group or featured character portrait
    - World/RPG: environment descriptors → iconic setting image
    - Group: multiple characters from group_members list

    If visual_traits exists, it's the hard truth. Otherwise full card is sent
    for the LLM to parse. No user/persona info — this is the card's avatar.
    """
    sections: list[str] = []

    # Group portrait: all members labeled
    if group_members:
        member_names = ", ".join(m.get("name", "?") for m in group_members)
        sections.append(
            f"GROUP PORTRAIT — portrait all of these characters together: {member_names}\n"
            "Show all characters in a single composition. Each character must match "
            "their physical description exactly."
        )
        for member in group_members:
            name = member.get("name", "Unknown")
            traits = member.get("visual_traits", "")
            appearance = member.get("appearance", "")
            desc = member.get("description", "")
            species = member.get("species", "")

            char_parts = []
            if species:
                char_parts.append(f"Species: {species}")
            if traits:
                char_parts.append(f"Appearance: {traits}")
            elif appearance:
                char_parts.append(f"Appearance: {appearance}")
            elif desc:
                char_parts.append(f"Description: {desc[:300]}")
            sections.append(
                f"<{name}> (use these details EXACTLY):\n" + "\n".join(char_parts)
            )

        if user_instruction.strip():
            sections.append(f"FOCUS: {user_instruction.strip()}")

        return "\n\n".join(sections) if sections else "Generate a group portrait."

    if ctx.character_card:
        card = ctx.character_card

        if card.visual_traits:
            traits_type = _detect_visual_traits_type(card.visual_traits)

            if traits_type == "ensemble":
                # Multi-character: send tagged blocks as-is
                header = "ENSEMBLE CHARACTERS (portrait all named characters):"
                if card.name:
                    header = f"ENSEMBLE CHARACTERS — {card.name} (portrait the cast):"
                sections.append(f"{header}\n{card.visual_traits}")

            elif traits_type == "world":
                # World/RPG: scene-focused
                header = "SETTING VISUAL IDENTITY"
                if card.name:
                    header = f"SETTING VISUAL IDENTITY — {card.name}"
                sections.append(
                    f"{header} (create an iconic image representing this world):\n"
                    f"{card.visual_traits}"
                )
                # Add scenario for richer world context
                if card.scenario:
                    sections.append(f"SCENARIO:\n{card.scenario[:500]}")

            else:
                # Single character: hard truth appearance
                parts = []
                if card.name:
                    parts.append(f"Name: {card.name}")
                if card.species:
                    parts.append(f"Species: {card.species}")
                parts.append(f"Appearance: {card.visual_traits}")
                sections.append(
                    "CHARACTER (use these details EXACTLY as written):\n"
                    + "\n".join(parts)
                )
        else:
            # No curated traits — send full card for the LLM to extract visuals
            card_parts = []
            if card.name:
                card_parts.append(f"Name: {card.name}")
            if card.species:
                card_parts.append(f"Species: {card.species}")
            if card.appearance:
                card_parts.append(f"Appearance: {card.appearance}")
            if card.description:
                card_parts.append(f"Description: {card.description}")
            if card.scenario:
                card_parts.append(f"Scenario: {card.scenario[:300]}")
            if card_parts:
                sections.append(
                    "CHARACTER CARD (extract visual appearance from these details):\n"
                    + "\n".join(card_parts)
                )

    # Conversation — last exchange only, for expression/mood context
    if conversation_messages:
        msgs = conversation_messages[-2:]
        conv_lines = []
        for msg in msgs:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"[{role_label}]: {msg['content'][:400]}")
        sections.append(
            "CURRENT MOOD CONTEXT (use for expression/emotion only):\n"
            + "\n".join(conv_lines)
        )

    if user_instruction.strip():
        sections.append(f"FOCUS: {user_instruction.strip()}")

    if not sections:
        return "Generate a character portrait."

    return "\n\n".join(sections)


def build_distiller_prompt(
    ctx: SceneContext,
    user_instruction: str = "",
    conversation_messages: list[dict[str, str]] | None = None,
    persona: UserPersona | None = None,
    core_profile: str = "",
    is_portrait: bool = False,
    group_members: list[dict] | None = None,
) -> str:
    """Build a distiller prompt from a SceneContext snapshot.

    Delegates to ``_build_portrait_prompt`` for portrait mode (avatar gen).
    Scene mode includes full context for multi-character scene illustration.

    Args:
        group_members: List of dicts with keys: name, visual_traits, appearance,
            description, species. When provided, replaces single character card
            with labeled per-character blocks.
    """
    if is_portrait:
        return _build_portrait_prompt(ctx, user_instruction, conversation_messages,
                                      group_members=group_members)

    sections: list[str] = []

    if group_members:
        # Group chat: inject all member visual data with name labels
        for member in group_members:
            name = member.get("name", "Unknown")
            traits = member.get("visual_traits", "")
            appearance = member.get("appearance", "")
            desc = member.get("description", "")
            species = member.get("species", "")

            if traits:
                sections.append(
                    f"<{name}> PHYSICAL DESCRIPTION — HARD TRUTH FOR APPEARANCE "
                    f"(use these exactly as written):\n{traits}"
                )
            parts = []
            if species:
                parts.append(f"Species: {species}")
            if appearance and appearance != traits:
                parts.append(f"Appearance: {appearance}")
            if desc:
                parts.append(f"Description: {desc[:300]}")
            if parts:
                sections.append(f"<{name}> CHARACTER DETAILS:\n" + "\n".join(parts))
    elif ctx.character_card:
        # Single character mode (original behavior)
        card = ctx.character_card
        char_name = (card.name or "").strip() or "Character"

        # Visual traits as a separate hard-truth block (top priority for the distiller).
        # Bind the name directly into the header so weak distiller LLMs can't lose
        # track of which name owns these traits and invent a substitute.
        if card.visual_traits:
            sections.append(
                f"{char_name.upper()}'S PHYSICAL DESCRIPTION — HARD TRUTH FOR APPEARANCE "
                f"(this is the canonical description of the character named '{char_name}'; "
                "override any other description if conflicting, use these exactly as written):\n"
                f"{card.visual_traits}"
            )

        card_parts = []
        if card.name:
            card_parts.append(f"Name: {card.name}")
        if card.species:
            card_parts.append(f"Species: {card.species}")
        if card.appearance:
            card_parts.append(f"Appearance: {card.appearance}")
        if card.description:
            card_parts.append(f"Description: {card.description}")
        if card.background:
            card_parts.append(f"Background: {card.background}")
        if card.scenario:
            card_parts.append(f"Scenario/Genre: {card.scenario}")
        if card_parts:
            sections.append(
                f"CHARACTER CARD for '{char_name}' (use for visual consistency):\n"
                + "\n".join(card_parts)
            )

    # User persona — the player/user's visual identity
    if persona and (persona.name or persona.appearance or persona.description):
        persona_name = (persona.name or "").strip() or "the player character"
        if persona.appearance:
            sections.append(
                f"{persona_name.upper()}'S PHYSICAL DESCRIPTION — HARD TRUTH FOR APPEARANCE "
                f"(this is the canonical description of the player character named '{persona_name}'; "
                "override any other description if conflicting, use these exactly as written):\n"
                f"{persona.appearance}"
            )

        persona_parts = []
        if persona.name:
            persona_parts.append(f"Name: {persona.name}")
        if persona.description:
            persona_parts.append(f"Description: {persona.description}")
        if persona_parts:
            sections.append("USER/PLAYER CHARACTER (use for visual consistency):\n" + "\n".join(persona_parts))

    # Core profile facts (supplementary user info from memory)
    if core_profile:
        sections.append(f"USER PROFILE FACTS:\n{core_profile}")

    # Scene context — prefer three-layer STATE snapshot (LLM-verified),
    # fall back to legacy world_state tracker (regex-based)
    if ctx.state_snapshot_fields:
        state_lines = []
        for key, value in ctx.state_snapshot_fields.items():
            if value:
                label = key.replace("_", " ").title()
                state_lines.append(f"{label}: {value}")
        if state_lines:
            sections.append("CURRENT STATE (scene snapshot):\n" + "\n".join(state_lines))
    else:
        # Legacy fallback — regex-based world tracker
        scene = ctx.legacy_world_state
        if scene.location or scene.time_of_day or scene.weather or scene.atmosphere:
            scene_parts = []
            if scene.location:
                scene_parts.append(f"Location: {scene.location}")
            if scene.time_of_day:
                scene_parts.append(f"Time: {scene.time_of_day}")
            if scene.weather:
                scene_parts.append(f"Weather: {scene.weather}")
            if scene.atmosphere:
                scene_parts.append(f"Atmosphere: {scene.atmosphere}")
            sections.append("CURRENT SCENE:\n" + "\n".join(scene_parts))

    # Memory ledger — recent key events for richer visual context
    if ctx.memory_ledger:
        recent_entries = ctx.memory_ledger[-8:]
        ledger_lines = [f"- [R{e.round_num}] {e.content}" for e in recent_entries]
        sections.append("RECENT KEY EVENTS:\n" + "\n".join(ledger_lines))

    # Conversation rounds — the actual dialogue for action/context
    if conversation_messages:
        conv_lines = []
        for msg in conversation_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"[{role_label}]: {msg['content']}")
        sections.append("RECENT CONVERSATION (use for current action/emotion/context):\n" + "\n".join(conv_lines))

    # User instruction (anything after /v)
    if user_instruction.strip():
        sections.append(f"USER FOCUS (prioritize this in the image): {user_instruction.strip()}")

    if group_members:
        member_names = ", ".join(m.get("name", "?") for m in group_members)
        intro = (
            f"This is a GROUP SCENE with multiple characters: {member_names}.\n"
            "All named characters should be present in the image with their correct appearances.\n"
            "Analyze the following context and generate a detailed image prompt "
            "that captures this group scene with visual consistency for each character.\n\n"
        )
    else:
        intro = (
            "Analyze the following narrative context and generate a detailed "
            "image prompt that captures this scene with full character visual consistency.\n\n"
        )

    return intro + "\n\n".join(sections)


def build_background_prompt(
    user_instruction: str = "",
    conversation_messages: list[dict[str, str]] | None = None,
    previous_prompts: list[str] | None = None,
) -> str:
    """Build a user prompt for atmospheric background generation.

    The prompt is CONVERSATION-ONLY — no static card info, no world state,
    no facts.  All style/genre guidance lives in the system prompt.
    The distiller LLM derives the scene entirely from the dialogue.

    ``previous_prompts`` lists recent background POSITIVE prompts for dedup.
    """
    sections: list[str] = []

    # Conversation is the ONLY scene source
    if conversation_messages:
        msgs = conversation_messages[-8:]  # 4 rounds for richer context
        conv_lines = []
        for msg in msgs:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:800]
            conv_lines.append(f"[{role_label}]: {content}")
        sections.append("\n".join(conv_lines))

    # Deduplication — show what was already generated
    if previous_prompts:
        dedup_lines = [f"- {p[:150]}" for p in previous_prompts[-3:]]
        sections.append(
            "PREVIOUS BACKGROUNDS (do NOT repeat):\n"
            + "\n".join(dedup_lines)
        )

    if not sections:
        return "Generate a moody atmospheric environment background."

    return "\n\n".join(sections)


_IMAGE_STYLE_MAP: dict[str, str] = {
    "anime": "Anime / manga illustration style backgrounds. Cel-shaded, vibrant colors, clean lines.",
    "painterly": "Painterly concept art. Rich brushwork, atmospheric, like a fantasy book illustration.",
    "photorealistic": "Photorealistic cinematic render. Film grain, shallow depth of field, natural lighting.",
    "watercolor": "Soft watercolor illustration. Gentle washes, muted tones, warm and delicate.",
    "pixel": "Pixel art / retro game aesthetic. Chunky pixels, limited palette, nostalgic.",
    "comic": "Comic book / graphic novel style. Bold outlines, flat colors, dramatic shadows.",
    "dark": "Dark gothic horror atmosphere. Deep shadows, chiaroscuro, muted desaturated palette.",
    "fantasy": "High fantasy epic. Grand scale, magical lighting, rich saturated colors.",
    "scifi": "Sci-fi / cyberpunk. Neon accents, sleek surfaces, technological, cool blue-purple palette.",
    "ukiyoe": "Ukiyo-e / East Asian ink wash. Flowing lines, muted earth tones, elegant composition.",
    "noir": "Film noir. High contrast black and white, dramatic shadows, rain-slicked surfaces.",
    "cozy": "Cozy slice of life. Warm golden light, soft focus, comfortable domestic spaces.",
}


def _build_genre_context(ctx: SceneContext) -> str:
    """Build a style hint for the background system prompt from the card's image_style."""
    style_key = ctx.image_style
    if not style_key or style_key not in _IMAGE_STYLE_MAP:
        return ""
    return f"\nArt style: {_IMAGE_STYLE_MAP[style_key]}\n"


_LEAK_PATTERNS = (
    # System-prompt preamble fragments seen leaking into POSITIVE.
    re.compile(r"^\s*target\s+image\s+model(?:\s+family)?\s*:\s*[^,\n.]+[,.]?\s*",
               re.IGNORECASE),
    re.compile(r"^\s*this\s+is\s+a[n]?\s+[a-z0-9 \-]+?\s+model\s*[.,:]?\s*",
               re.IGNORECASE),
    re.compile(r"^\s*(?:positive|prompt)\s*:\s*", re.IGNORECASE),
)


_FAMILY_NAME_TOKENS = frozenset({
    "lumina", "flux", "sdxl", "sd15", "sd1.5", "sd3", "pony",
    "dall-e", "dalle", "midjourney", "ideogram", "recraft",
    "schnell", "neta-lumina",
})


def _sanitize_positive(positive: str) -> str:
    """Strip distiller-system-prompt leakage from the POSITIVE line.

    Weak distiller LLMs sometimes echo the system-prompt model-context
    preamble (e.g. "Target image model family: lumina, ...") into the
    POSITIVE line, or fuse the model's family name into a tag list
    ("high quality lumina, …"). That text then gets rendered as literal
    characters in the generated image. Strip the known preamble shapes
    AND any bare family-name tokens here so the pipeline only ever sees
    a clean image prompt.
    """
    out = positive.strip()
    # Iterate so we strip stacked prefixes (model-context line + style line).
    for _ in range(4):
        before = out
        for pat in _LEAK_PATTERNS:
            out = pat.sub("", out, count=1).lstrip()
        if out == before:
            break
    # Drop bare family-name tokens that slipped in as comma-separated tags,
    # AND scrub family names embedded as standalone words inside any tag
    # (e.g. "high quality lumina" → "high quality").
    family_word_pat = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in _FAMILY_NAME_TOKENS) + r")\b",
        re.IGNORECASE,
    )
    if "," in out:
        kept = []
        for raw in out.split(","):
            tok = raw.strip()
            if tok.lower() in _FAMILY_NAME_TOKENS:
                continue
            scrubbed = family_word_pat.sub("", tok)
            scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
            if scrubbed:
                kept.append(scrubbed)
        out = ", ".join(kept)
    else:
        scrubbed = family_word_pat.sub("", out)
        out = re.sub(r"\s+", " ", scrubbed).strip()
    return out


def parse_distiller_response(response: str) -> dict:
    """Parse the LLM's distiller output into structured prompt components.

    Returns:
        dict with keys: positive, negative, aspect
    """
    result = {
        "positive": "",
        "negative": "blurry, low quality, deformed, ugly, watermark, text, words, letters",
        "aspect": "square",
    }

    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("POSITIVE:"):
            result["positive"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("NEGATIVE:"):
            result["negative"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ASPECT:"):
            aspect = line.split(":", 1)[1].strip().lower()
            if aspect in ("portrait", "landscape", "square"):
                result["aspect"] = aspect

    result["positive"] = _sanitize_positive(result["positive"])
    return result


async def distill_scene(
    ctx: SceneContext,
    backend: ModelBackend,
    model: str,
    user_instruction: str = "",
    conversation_messages: list[dict[str, str]] | None = None,
    persona: UserPersona | None = None,
    core_profile: str = "",
    distiller_model: str = "",
    image_model: str = "",
    mode: str = "auto",
    previous_prompts: list[str] | None = None,
    group_members: list[dict] | None = None,
    registry: ProviderRegistry | None = None,
) -> dict:
    """Run the full distillation pipeline: build prompt -> LLM call -> parse.

    Adapts behavior based on the target image model and LLM backend:
    - Cloud LLM backends get higher token budgets (1500 vs 800)
    - Portrait requests use a focused character-study prompt
    - Background mode uses environment-focused prompt (no characters)
    - Model-specific style hints guide prompt format (tags vs natural language)

    Args:
        ctx: Read-only snapshot of distiller-relevant fields. Built by the
            caller via ``build_scene_context`` so the cached engine is never
            mutated and never read from after construction.
        backend: The LLM backend to use for distillation.
        model: Fallback model name.
        user_instruction: Text after /v command.
        conversation_messages: Recent conversation rounds.
        persona: User persona data.
        core_profile: Core memory profile text.
        distiller_model: Specific model for distillation (overrides model).
        image_model: Target image generation model (for style-aware prompting).
        mode: Distillation mode — "auto" (detect from instruction), "scene",
              "portrait", or "background".

    Returns:
        dict with keys: positive, negative, aspect
    """
    from augmentum.image.prompt_condenser import _build_model_context
    from augmentum.models.base import InternalChatRequest, Message

    # Resolve mode
    is_background = mode == "background"
    is_portrait = mode == "portrait" or (mode == "auto" and _is_portrait_request(user_instruction))
    is_cloud = _is_cloud_backend(backend)

    # Log what visual data the distiller is working with
    card = ctx.character_card
    card_visual_traits = card.visual_traits if card else ""
    card_name = card.name if card else "(no card)"
    persona_appearance = persona.appearance if persona else ""
    log.info(
        "distiller_input_context",
        character=card_name,
        has_visual_traits=bool(card_visual_traits),
        visual_traits=card_visual_traits[:200] if card_visual_traits else "(none)",
        has_persona_appearance=bool(persona_appearance),
        persona_appearance=persona_appearance[:200] if persona_appearance else "(none)",
        is_portrait=is_portrait,
        is_background=is_background,
        mode=mode,
    )

    # Select prompt builder based on mode
    if is_background:
        distiller_prompt = build_background_prompt(
            user_instruction=user_instruction,
            conversation_messages=conversation_messages,
            previous_prompts=previous_prompts,
        )
    else:
        distiller_prompt = build_distiller_prompt(
            ctx,
            user_instruction=user_instruction,
            conversation_messages=conversation_messages,
            persona=persona,
            core_profile=core_profile,
            is_portrait=is_portrait,
            group_members=group_members,
        )

    log.debug("distiller_full_prompt", prompt=distiller_prompt, mode=mode)

    # Build model-aware system prompt
    model_context = _build_model_context(image_model)
    if is_background:
        system_template = BACKGROUND_SYSTEM_PROMPT
    elif is_portrait:
        system_template = PORTRAIT_SYSTEM_PROMPT
    else:
        system_template = DISTILLER_SYSTEM_PROMPT
    style_context = _build_genre_context(ctx)
    if is_background:
        system = system_template.format(model_context=model_context, genre_context=style_context)
    elif is_portrait:
        system = system_template.format(model_context=model_context, style_context=style_context)
    else:
        system = system_template.format(model_context=model_context)

    # Use distiller-specific model if configured; otherwise use the role-based
    # resolver (utility role → image_prompt_condense_model override).
    use_model = distiller_model or model
    if not use_model:
        try:
            from augmentum.config import settings as _settings

            if registry is not None:
                backend, use_model = await registry.resolve_model_for_role(
                    "utility",
                    override=_settings.image_prompt_condense_model,
                    settings=_settings,
                )
            else:
                # No registry available — fall back to settings string then
                # first model from the already-resolved backend.
                use_model = _settings.image_prompt_condense_model
        except Exception as exc:
            log.debug("distiller_role_resolve_failed", error=str(exc))
    if not use_model:
        try:
            available = await backend.list_models()
            if available:
                use_model = available[0].name
        except Exception as exc:
            log.debug("distiller_list_models_failed", error=str(exc))
    log.debug("distiller_model_resolved", model=use_model or "(none)")

    # Adaptive token budget: cloud LLMs can afford richer output;
    # background mode uses less since it's environment-only
    if is_background:
        max_tokens = 800 if is_cloud else 500
    else:
        max_tokens = 1500 if is_cloud else 800

    request = InternalChatRequest(
        model=use_model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=distiller_prompt),
        ],
        stream=False,
        temperature=0.7 if is_background else 0.3,
        max_tokens=max_tokens,
    )

    try:
        response = await backend.chat(request)
        output = response.message.content if response.message else ""
    except Exception as exc:
        log.warning("distiller_llm_failed", error=str(exc), mode=mode)
        # Fall back to a simple prompt from scene state
        scene = ctx.legacy_world_state
        parts = []
        if scene.location:
            parts.append(scene.location)
        if scene.time_of_day:
            parts.append(scene.time_of_day)
        if scene.atmosphere:
            parts.append(scene.atmosphere)
        if not is_background:
            # Only include character details in non-background fallbacks
            if ctx.character_card and ctx.character_card.appearance:
                parts.append(ctx.character_card.appearance[:200])
            if persona and persona.appearance:
                parts.append(persona.appearance[:200])
        else:
            if scene.weather:
                parts.append(scene.weather)
        if user_instruction:
            parts.append(user_instruction)

        default_negative = "blurry, low quality, deformed, ugly, watermark, text"
        if is_background:
            default_negative += ", people, characters, figures, faces"

        return {
            "positive": ", ".join(parts) if parts else "fantasy scene",
            "negative": default_negative,
            "aspect": "landscape" if is_background else ("portrait" if is_portrait else "square"),
        }

    result = parse_distiller_response(output)

    # Determine whether the target model honors negatives. FLUX/Schnell and
    # cloud APIs effectively run at cfg=1, so negatives are inert there.
    supports_negative = True
    if image_model:
        from augmentum.image.prompt_condenser import detect_image_model_style
        model_info = detect_image_model_style(image_model)
        supports_negative = bool(model_info["supports_negative"])

    if supports_negative:
        # Baseline floor — applied unconditionally so a hallucinated empty
        # NEGATIVE: line from the distiller LLM never ships an unguarded image.
        # Covers the artifacts the user actually sees (garbled text from strong
        # text encoders, fingers/anatomy/watermark/jpeg artifacts).
        baseline = [
            "text", "words", "letters", "signature", "watermark",
            "jumbled text", "garbled text", "logo",
            "low quality", "blurry", "jpeg artifacts",
            "deformed", "bad anatomy", "extra fingers", "missing fingers",
            "extra limbs", "fused fingers", "mutated hands",
        ]
        if is_background:
            # Backgrounds: also suppress people/characters/UI elements.
            baseline.extend(["people", "characters", "figures", "faces", "ui elements"])
        neg_lower = result["negative"].lower()
        for tag in baseline:
            if tag not in neg_lower:
                result["negative"] += f", {tag}"
        # Strip any leading ", " that arises when the LLM returned an empty
        # NEGATIVE line and we appended onto a non-empty default fallback.
        result["negative"] = result["negative"].lstrip(", ").strip()
    else:
        result["negative"] = ""

    # Force landscape for backgrounds regardless of LLM output
    if is_background:
        result["aspect"] = "landscape"

    log.info(
        "scene_distilled",
        positive_len=len(result["positive"]),
        aspect=result["aspect"],
        is_portrait=is_portrait,
        is_background=is_background,
        is_cloud_llm=is_cloud,
        image_model=image_model or "(default)",
    )
    return result
