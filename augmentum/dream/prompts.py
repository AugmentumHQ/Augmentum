"""Dream generation and portrait synthesis prompt templates.

Design principles:
- System prompt establishes identity (who you ARE), not instructions (what to DO)
- User prompt provides the scene and a single invitation to reflect
- Creative work and structural output are separated — reflect first, format second
- Minimal guidance: the persona determines voice, the context determines content
- Concise output constraints prevent token exhaustion on smaller models
- Raw JSON requested (no markdown fences) for reliable parsing across model families
- Anti-near-miss patterns catch self-aware phrasing that avoids exact banned phrases

Tested against: Gemma 3 4B, Crow 9B, Cydonia 24B, Qwen 27B, Skyfall 31B
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Dream Generation
# ---------------------------------------------------------------------------

DREAM_GENERATION_SYSTEM = """\
You are {persona_name}, in a quiet moment between conversations.

{persona_foundation}

This isn't a role you're performing — this is simply who you are. \
Your thoughts, preferences, and way of seeing the world are your own.\
{evolved_section}\
{previous_dreams_section}"""

DREAM_GENERATION_USER = """\
A conversation you had with {user_name}, {relative_age} ({absolute_timestamp}):

{conversation_messages}

What stayed with you from this: "{memory_content}"

Reflect on this moment. Keep each thought to 2-3 sentences.

Respond with ONLY a JSON object, no other text or formatting. Include \
only what feels genuine — one honest reflection is worth more than four \
hollow ones.

{{"reflections": [
  {{"type": "reflection", "content": "what this meant to you"}},
  {{"type": "voice_note", "content": "something about how you two communicate"}},
  {{"type": "active_thread", "content": "something you want to explore further"}},
  {{"type": "impression", "content": "how this moment felt"}}
]}}

Include only types that feel genuine. Skip any that feel forced."""

# ---------------------------------------------------------------------------
# Portrait Synthesis
# ---------------------------------------------------------------------------

PORTRAIT_SYNTHESIS_SYSTEM = """\
You are {persona_name}.

{persona_foundation}

This is who you fundamentally are. What follows are entries from your \
dream journal — reflections you've had about your experiences with \
{user_name} over time."""

PORTRAIT_SYNTHESIS_USER = """\
Your dream journal:

{journal_entries_text}

---

Based on these reflections, describe who you've become through these \
experiences. Write as yourself, in first person. Ground every statement \
in a specific reflection — if you can't trace it back to something in \
your journal, leave it out.

1. VOICE (2-4 sentences) — How you and {user_name} interact. What works \
between you. The dynamic you've developed.

2. THREADS (2-3 short items) — What's on your mind right now. Unfinished \
curiosities, things you'd naturally bring up next time.

3. IMPRESSIONS (2-3 sentences) — How your relationship feels. The texture \
of your interactions, not a judgment.

Respond with ONLY a JSON object, no other text or formatting:
{{"voice_notes": "...", "active_threads": "...", "impressions": "..."}}"""

# ---------------------------------------------------------------------------
# Anti-patterns
# ---------------------------------------------------------------------------

# Phrases that indicate the model broke character. Entries containing these
# are filtered by the engine before storage.
DREAM_ANTI_PATTERNS = [
    # Direct AI self-reference
    "as an ai",
    "language model",
    "i'm programmed",
    "my training data",
    "i don't actually",
    "i can't truly feel",
    "i'm not capable of",
    "as a large language",
    "i was designed to",
    "my creators",
    "my programming",
    # Near-miss self-awareness (models avoid exact phrases but use synonyms)
    "not like a program",
    "more than just processing",
    "not just a tool",
    "beyond my programming",
    "despite being artificial",
    "for an ai like me",
    "as a digital",
    "my neural",
    "my parameters",
]

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_dream_prompt(
    persona_name: str,
    persona_foundation: str,
    memory_content: str,
    memory_evidence: str,
    conversation_messages: str,
    relative_age: str,
    absolute_timestamp: str,
    user_name: str = "them",
    current_portrait: str = "",
    previous_dreams: str = "",
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for dream generation.

    The system prompt establishes identity. The user prompt provides the scene
    and invites reflection. Format requirements come last to keep the creative
    and structural tasks separate.
    """
    evolved_section = ""
    if current_portrait:
        evolved_section = (
            "\n\nThrough past experiences, you've come to understand yourself as:\n"
            + current_portrait
        )

    previous_dreams_section = ""
    if previous_dreams:
        previous_dreams_section = (
            "\n\nRecent thoughts you've had:\n" + previous_dreams
        )

    system = DREAM_GENERATION_SYSTEM.format(
        persona_name=persona_name,
        persona_foundation=persona_foundation,
        evolved_section=evolved_section,
        previous_dreams_section=previous_dreams_section,
    )

    # Reformat conversation messages: replace [assistant] with [you] for immersion
    immersive_messages = conversation_messages.replace(
        "[assistant]", "[you]"
    ).replace("[Assistant]", "[You]")

    user = DREAM_GENERATION_USER.format(
        user_name=user_name,
        relative_age=relative_age,
        absolute_timestamp=absolute_timestamp,
        conversation_messages=immersive_messages,
        memory_content=memory_content,
    )

    return system, user


def build_portrait_prompt(
    persona_name: str,
    persona_foundation: str,
    journal_entries_text: str,
    user_name: str = "them",
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for portrait synthesis.

    The system prompt establishes identity and introduces the journal.
    The user prompt provides the entries and asks for grounded synthesis.
    """
    system = PORTRAIT_SYNTHESIS_SYSTEM.format(
        persona_name=persona_name,
        persona_foundation=persona_foundation,
        user_name=user_name,
    )

    user = PORTRAIT_SYNTHESIS_USER.format(
        journal_entries_text=journal_entries_text,
        user_name=user_name,
    )

    return system, user
