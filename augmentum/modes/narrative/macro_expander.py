"""Macro expansion for narrative mode — replaces template variables in text.

Adapted from SillyTavern's expandMacros() pattern for Augmentum's proxy
architecture.  Macros are expanded in system prompts, character card fields,
lorebook content, and preset prompt fields before they reach the LLM.

Supported macros:
  {{char}}       — Character name (from parsed card or "Character")
  {{user}}       — User/persona name (from active persona or "User")
  {{obj}}        — Alias for {{user}} (JanitorAI compatibility)
  {{persona}}    — Full persona description
  {{time}}       — Current local time (HH:MM)
  {{date}}       — Current date (YYYY-MM-DD)
  {{day}}        — Day of week (Monday, Tuesday, ...)
  {{random}}     — Random float 0.0000–0.9999
  {{random:a,b,c}} — Pick one item at random from the comma-separated list
  {{roll:NdM}}   — Dice roll sum (e.g. {{roll:2d6}} → 7)
  {{idle_duration}} — Messages since user last spoke (approximate)
"""

from __future__ import annotations

import random
import re
from datetime import datetime

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Pre-compiled patterns for performance
_ROLL_RE = re.compile(r"\{\{roll:(\d{1,3})d(\d{1,4})\}\}", re.IGNORECASE)
_RANDOM_LIST_RE = re.compile(r"\{\{random:([^}]+)\}\}", re.IGNORECASE)
_SIMPLE_MACROS = re.compile(
    r"\{\{(char|user|obj|persona|time|date|day|random|idle_duration)\}\}",
    re.IGNORECASE,
)


def expand_macros(
    text: str,
    *,
    char_name: str = "Character",
    user_name: str = "User",
    persona_description: str = "",
    message_count: int = 0,
) -> str:
    """Expand all macros in *text* and return the result.

    Parameters
    ----------
    text:
        Input text with ``{{macro}}`` placeholders.
    char_name:
        Name of the character (from parsed card).
    user_name:
        Name of the user persona.
    persona_description:
        Full text of the user persona description.
    message_count:
        Approximate number of messages in the session (for idle_duration).
    """
    if not text or "{{" not in text:
        return text

    # Strip stray underscores around macros (markdown italic artifacts from card imports)
    text = re.sub(r"_(\{\{[^}]+\}\})_?", r"\1", text)
    text = re.sub(r"(\{\{[^}]+\}\})_", r"\1", text)

    # Use configured timezone so macros match the injected system date
    from augmentum.utils.datetime_context import _get_local_tz

    now = datetime.now(_get_local_tz())

    lookup = {
        "char": char_name,
        "user": user_name,
        "obj": user_name,  # JanitorAI alias for {{user}}
        "persona": persona_description,
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
        "random": f"{random.random():.4f}",
        "idle_duration": str(message_count),
    }

    def _replace_simple(m: re.Match) -> str:
        key = m.group(1).lower()
        return lookup.get(key, m.group(0))

    result = _SIMPLE_MACROS.sub(_replace_simple, text)

    # Random list pick — {{random:a,b,c}}
    def _replace_random_list(m: re.Match) -> str:
        items = [item.strip() for item in m.group(1).split(",") if item.strip()]
        if not items:
            return ""
        return random.choice(items)

    result = _RANDOM_LIST_RE.sub(_replace_random_list, result)

    # Dice rolls — {{roll:NdM}}
    def _replace_roll(m: re.Match) -> str:
        count = min(int(m.group(1)), 100)  # cap at 100 dice
        sides = max(int(m.group(2)), 1)
        total = sum(random.randint(1, sides) for _ in range(count))
        return str(total)

    result = _ROLL_RE.sub(_replace_roll, result)

    return result


def expand_messages(
    messages: list,
    *,
    char_name: str = "Character",
    user_name: str = "User",
    persona_description: str = "",
    message_count: int = 0,
) -> list:
    """Expand macros in all message contents (modifies in-place, returns same list)."""
    for msg in messages:
        if msg.content and "{{" in msg.content:
            msg.content = expand_macros(
                msg.content,
                char_name=char_name,
                user_name=user_name,
                persona_description=persona_description,
                message_count=message_count,
            )
    return messages
