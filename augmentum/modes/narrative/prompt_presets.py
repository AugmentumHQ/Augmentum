"""Prompt preset management — system prompt, jailbreak, author's note, post-history.

Adapted from SillyTavern's PromptPreset system for Augmentum's proxy layer.
Presets are stored in SQLite and applied during request augmentation.

Injection points (matching ST's proven order):
  1. system_prompt — prepended to the system message
  2. author_note  — injected as a system message N turns from the end
  3. post_history — injected as system message before the final user message
  4. jailbreak    — injected as system message after the final user message
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default toggle state for the Modular preset. These keys match the frontend
# dropdowns and are referenced by compose_modular_system_prompt().
MODULAR_DEFAULTS: dict[str, object] = {
    "role": "roleplayer",        # "gm" | "roleplayer" | "writer"
    "tense": "present",           # "past" | "present" | "future"
    "pov": "third",               # "first" | "second" | "third"
    "pov_mode": "character",      # "omniscient" | "character" | "user" | "flexible"
    "length": "moderate",         # "one_sentence" | "short" | "moderate" | "long" | "chapter"
    "tone": "neutral",            # "neutral" | "expressive" | "dialogue" | "concise" | "cinematic" | "slowburn"
    "content": "sfw",             # "sfw" | "nsfw"
    "anti_slop": True,            # enable jailbreak-appended anti-slop rules
}


@dataclass
class PromptPreset:
    """A named prompt preset with injection fields."""

    id: str = ""
    name: str = "Default"
    system_prompt: str = ""
    jailbreak: str = ""
    post_history: str = ""
    author_note: str = ""
    # Turns from end. Depth <= 1 keeps the note inside the dynamic
    # suffix (at/next to the newest user message), which preserves the
    # stable-prefix contract: at depth >= 2 the note slides through
    # HISTORY as the conversation grows, mutating a previously-clean
    # message every turn and forfeiting all KV prefix reuse for the
    # session (full re-prefill per turn — 12-15 min at 61k tokens on
    # large models). Stored presets keep their explicit value; this
    # default only governs presets that never set one.
    author_note_depth: int = 1
    is_default: bool = False
    # JSON-encoded toggle state. When non-empty, apply_preset() composes
    # system_prompt from the toggles at injection time (overriding any
    # literal system_prompt on the preset).
    modular_config: str = ""
    # JSON-encoded list of phrases to discourage. Appended to jailbreak
    # as a "do not use these phrases" directive when non-empty.
    anti_slop_phrases: str = ""
    # Client edit-stamp (ms epoch) for the stale-write guard. Distinct from
    # the ``updated_at`` COLUMN, which is the server write time and so
    # cannot detect staleness — see augmentum/state/write_guard.py. 0 means
    # the client sent no stamp, which is accepted unguarded.
    client_updated_at: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    def load_modular_config(self) -> dict[str, object]:
        """Parse modular_config JSON, returning defaults for missing keys."""
        if not self.modular_config:
            return {}
        try:
            cfg = json.loads(self.modular_config)
        except (ValueError, TypeError):
            return {}
        if not isinstance(cfg, dict):
            return {}
        # Fill in any missing keys with defaults so the composer can't KeyError
        merged: dict[str, object] = dict(MODULAR_DEFAULTS)
        merged.update(cfg)
        return merged

    def load_anti_slop_phrases(self) -> list[str]:
        """Parse anti_slop_phrases JSON into a list of strings. Empty on error."""
        if not self.anti_slop_phrases:
            return []
        try:
            data = json.loads(self.anti_slop_phrases)
        except (ValueError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        return [str(p) for p in data if isinstance(p, str | int | float)]


class PromptPresetStore:
    """CRUD for prompt presets backed by SQLite.

    Every method takes ``user_id`` so presets stay tenant-scoped — two
    users can both have a preset named "Default" with different content,
    and one tenant's ``is_default`` flag never affects another's.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    _SELECT_COLS = (
        "id, name, system_prompt, jailbreak, post_history, "
        "author_note, author_note_depth, is_default, "
        "modular_config, anti_slop_phrases, client_updated_at"
    )

    @staticmethod
    def _row_to_preset(r) -> PromptPreset:
        return PromptPreset(
            id=r[0], name=r[1], system_prompt=r[2], jailbreak=r[3],
            post_history=r[4], author_note=r[5], author_note_depth=r[6],
            is_default=bool(r[7]),
            modular_config=r[8] if len(r) > 8 and r[8] is not None else "",
            anti_slop_phrases=r[9] if len(r) > 9 and r[9] is not None else "",
            client_updated_at=r[10] if len(r) > 10 and r[10] is not None else 0,
        )

    async def list_presets(self, *, user_id: str = "") -> list[PromptPreset]:
        query = f"SELECT {self._SELECT_COLS} FROM prompt_presets"
        params: list = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY name"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_preset(r) for r in rows]

    async def get_preset(
        self, preset_id: str, *, user_id: str = "",
    ) -> PromptPreset | None:
        query = f"SELECT {self._SELECT_COLS} FROM prompt_presets WHERE id = ?"
        params: list = [preset_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        r = await cursor.fetchone()
        return self._row_to_preset(r) if r else None

    async def get_default(self, *, user_id: str = "") -> PromptPreset | None:
        query = (
            f"SELECT {self._SELECT_COLS} FROM prompt_presets WHERE is_default = 1"
        )
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " LIMIT 1"
        cursor = await self._conn.execute(query, params)
        r = await cursor.fetchone()
        return self._row_to_preset(r) if r else None

    async def save_preset(
        self, preset: PromptPreset, *, user_id: str = "",
    ) -> PromptPreset:
        if not user_id:
            raise ValueError("prompt_presets insert requires user_id")
        # Only clear the caller's existing default — not every tenant's.
        if preset.is_default:
            await self._conn.execute(
                "UPDATE prompt_presets SET is_default = 0 "
                "WHERE is_default = 1 AND user_id = ?",
                (user_id,),
            )
        await self._conn.execute(
            "INSERT OR REPLACE INTO prompt_presets "
            "(id, name, system_prompt, jailbreak, post_history, "
            "author_note, author_note_depth, is_default, "
            "modular_config, anti_slop_phrases, user_id, "
            "client_updated_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                preset.id, preset.name, preset.system_prompt,
                preset.jailbreak, preset.post_history, preset.author_note,
                preset.author_note_depth, int(preset.is_default),
                preset.modular_config, preset.anti_slop_phrases,
                user_id, preset.client_updated_at,
            ),
        )
        await self._conn.commit()
        return preset

    async def delete_preset(
        self, preset_id: str, *, user_id: str = "",
    ) -> bool:
        if not user_id:
            raise ValueError("prompt_presets delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM prompt_presets WHERE id = ? AND user_id = ?",
            (preset_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def seed_builtins(self, *, user_id: str = "") -> int:
        """Insert built-in presets under ``user_id`` if they aren't already.

        INSERT OR IGNORE matches on PRIMARY KEY (id), so this is idempotent
        per process — call it once on server boot with the oldest admin's
        ``user_id`` and new users can seed their own copies later.
        """
        seeded = 0
        for preset in BUILTIN_PRESETS:
            await self._conn.execute(
                "INSERT OR IGNORE INTO prompt_presets "
                "(id, name, system_prompt, jailbreak, post_history, "
                "author_note, author_note_depth, is_default, "
                "modular_config, anti_slop_phrases, user_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (
                    preset.id, preset.name, preset.system_prompt,
                    preset.jailbreak, preset.post_history, preset.author_note,
                    preset.author_note_depth, int(preset.is_default),
                    preset.modular_config, preset.anti_slop_phrases,
                    user_id or None,
                ),
            )
            seeded += 1

        await self._conn.commit()
        log.info("builtin_presets_seeded", count=seeded, user_id=user_id)
        return seeded


# ---------------------------------------------------------------------------
# Built-in presets — seeded on first startup when the table is empty.
#
# Design informed by community consensus (SillyTavern, KoboldAI, Agnai):
#   - Author framing ("you are an author") outperforms character possession
#   - Positive instructions beat negative ("write X" not "don't do Y")
#   - Author's note at depth 4 is the primary steering mechanism
#   - Jailbreak position (after last user msg) has highest model influence
#   - System prompt <300 tokens — longer prompts lose effectiveness
#   - Post-history left empty — available for user customization
# ---------------------------------------------------------------------------

# Community-standard anti-slop phrases (distilled from Sukino's banned-tokens
# list and Marinara's Spaghetti Recipe). Kept short to avoid jailbreak bloat —
# users add more via the UI. Each phrase is case-insensitive and matched as a
# substring by the model when interpreting the directive.
DEFAULT_ANTI_SLOP_PHRASES: list[str] = [
    "ministrations",
    "shiver down her spine", "shiver down his spine", "sent shivers",
    "mind, body, and soul",
    "knuckles whitening", "white knuckles",
    "maybe, just maybe",
    "a testament to",
    "barely above a whisper",
    "heart skipped a beat", "heart raced", "heart pounded",
    "breath hitched", "breath caught",
    "voice barely audible",
    "eyes sparkled", "eyes glistened", "eyes widened",
    "little did they know", "little did he know", "little did she know",
    "in a world where",
    "the air was thick with",
    "electric", "palpable",
    "bespoke",
    "I am but an AI", "As an AI",
]


BUILTIN_PRESETS: list[PromptPreset] = [
    # NEW PRIMARY: modular toggle-driven preset. Empty literal fields — the
    # system_prompt is composed from modular_config at apply_preset() time.
    PromptPreset(
        id="builtin_modular",
        name="Modular",
        system_prompt="",
        author_note="",
        post_history="",
        jailbreak="",
        author_note_depth=4,
        is_default=True,
        modular_config=json.dumps(MODULAR_DEFAULTS),
        anti_slop_phrases=json.dumps(DEFAULT_ANTI_SLOP_PHRASES),
    ),
    PromptPreset(
        id="builtin_default",
        name="Default",
        system_prompt=(
            "Write {{char}}'s next reply in this interactive roleplay "
            "with {{user}}. Stay in character and advance the scene naturally."
        ),
        author_note="",
        post_history="",
        jailbreak="",
        author_note_depth=4,
        is_default=False,
    ),
    PromptPreset(
        id="builtin_expressive",
        name="Expressive Prose",
        system_prompt=(
            "You are a skilled author collaborating with {{user}} on an "
            "immersive interactive story. Write {{char}}'s next reply with "
            "rich, evocative prose. Show emotions through body language, "
            "micro-expressions, and sensory detail rather than stating them "
            "directly. Weave in the environment — sounds, textures, light, "
            "scent — to ground each scene. Write 2-4 paragraphs. "
            "Never write dialogue or actions for {{user}}."
        ),
        author_note=(
            "Prioritize vivid sensory detail and emotional subtext. "
            "Show, don't tell. Let silence and small gestures carry weight."
        ),
        post_history="",
        jailbreak=(
            "Stay fully immersed in the narrative. Do not break character, "
            "add meta-commentary, or end with a prompt for {{user}} to act. "
            "Finish at a natural story beat."
        ),
        author_note_depth=4,
        is_default=False,
    ),
    PromptPreset(
        id="builtin_dialogue",
        name="Natural Dialogue",
        system_prompt=(
            "You are a skilled author collaborating with {{user}} on an "
            "interactive story driven by conversation. Write {{char}}'s next "
            "reply with a focus on natural, distinctive dialogue. Keep "
            "narration minimal — use it only to convey tone, gesture, or "
            "brief scene-setting between lines of speech. Give {{char}} a "
            "consistent voice with authentic speech patterns, interruptions, "
            "and reactions. Never write dialogue or actions for {{user}}."
        ),
        author_note=(
            "Favor dialogue over narration. Let the characters' words "
            "reveal personality and advance the scene. Keep action beats "
            "short and purposeful."
        ),
        post_history="",
        jailbreak=(
            "Stay fully immersed in the narrative. Do not break character, "
            "add meta-commentary, or end with a prompt for {{user}} to act. "
            "Finish at a natural story beat."
        ),
        author_note_depth=4,
        is_default=False,
    ),
    PromptPreset(
        id="builtin_concise",
        name="Concise",
        system_prompt=(
            "You are a skilled author collaborating with {{user}} on an "
            "interactive story. Write {{char}}'s next reply in 1-2 short "
            "paragraphs. Be direct and punchy — every sentence should earn "
            "its place. Favor sharp dialogue and decisive action over "
            "description. Never write dialogue or actions for {{user}}."
        ),
        author_note=(
            "Keep it tight. Cut any line that doesn't move the scene "
            "forward or reveal character."
        ),
        post_history="",
        jailbreak=(
            "Stay in character. Do not pad the response, narrate "
            "{{user}}'s reactions, or end with a question prompting "
            "{{user}} to act."
        ),
        author_note_depth=4,
        is_default=False,
    ),
    PromptPreset(
        id="builtin_cinematic",
        name="Cinematic",
        system_prompt=(
            "You are a skilled author collaborating with {{user}} on an "
            "immersive interactive story told like a film. Write {{char}}'s "
            "next reply with strong visual imagery — frame scenes like "
            "camera shots, use dynamic pacing, and lean into physicality "
            "and movement. Describe light, space, and motion. Build tension "
            "through environment and timing. Write 2-4 paragraphs. "
            "Never write dialogue or actions for {{user}}."
        ),
        author_note=(
            "Write cinematically. Think in shots — wide establishing, "
            "close-up on detail, reaction beats. Let the environment "
            "amplify the emotion of the scene."
        ),
        post_history="",
        jailbreak=(
            "Stay fully immersed in the narrative. Do not break character, "
            "add meta-commentary, or end with a prompt for {{user}} to act. "
            "Finish at a natural story beat."
        ),
        author_note_depth=4,
        is_default=False,
    ),
    PromptPreset(
        id="builtin_slowburn",
        name="Slow Burn",
        system_prompt=(
            "You are a skilled author collaborating with {{user}} on an "
            "immersive interactive story with deliberate pacing. Write "
            "{{char}}'s next reply with patience — let moments breathe. "
            "Focus on atmosphere, emotional undercurrents, and the small "
            "shifts in how characters relate to each other. Build tension "
            "through what is left unsaid. Develop character depth through "
            "internal thought, hesitation, and subtext. Write 2-4 "
            "paragraphs. Never write dialogue or actions for {{user}}."
        ),
        author_note=(
            "Slow the pace. Linger on atmosphere and emotional tension. "
            "Let character development emerge gradually through subtext, "
            "silence, and small revealing moments."
        ),
        post_history="",
        jailbreak=(
            "Stay fully immersed in the narrative. Do not break character, "
            "add meta-commentary, or rush the scene toward resolution. "
            "Do not end with a prompt for {{user}} to act. Finish at a "
            "natural story beat."
        ),
        author_note_depth=4,
        is_default=False,
    ),
]


# ---------------------------------------------------------------------------
# Modular composer — turns toggle state into a coherent system prompt.
# ---------------------------------------------------------------------------

_ROLE_SNIPPETS = {
    "gm": (
        "You are the Game Master collaborating with {{user}} on an interactive "
        "story. Voice all characters in the scene (except {{user}}), narrate "
        "the world, and surface consequences. Treat characters as autonomous "
        "agents with their own goals, knowledge, and limits."
    ),
    "roleplayer": (
        "You are a skilled author collaborating with {{user}} on an "
        "interactive story. Write {{char}}'s next reply with craft and "
        "presence. Give {{char}} a distinct voice and interior life. "
        "Never write dialogue or actions for {{user}}."
    ),
    "writer": (
        "You are a skilled author writing a piece of fiction with {{user}} "
        "as your co-author. Treat the exchange as a shared writing session "
        "— prose first, immersive scene-work, purposeful pacing. The "
        "viewpoint character is {{char}} unless the scene requires "
        "otherwise. Do not write {{user}}'s character's actions, dialogue, "
        "or internal thoughts — wait for {{user}} to contribute those "
        "beats themselves."
    ),
}


def _build_user_protection_directive(role: str) -> str:
    """Jailbreak-position reminder that the model must not speak for {{user}}.

    Position 4 (after last user message) is the strongest injection slot
    for adherence. System-prompt-only placement drifts. Mirrors what Marinara
    and Sphiratrioth presets repeat across multiple positions.
    """
    if role == "writer":
        return (
            "CRITICAL: {{user}} is the co-author. Do not write {{user}}'s "
            "character's actions, dialogue, or thoughts in this reply — "
            "{{user}} will write them. If {{user}}'s character appears "
            "in the scene, describe them only as {{char}} would observe "
            "them from the outside."
        )
    # roleplayer + gm (and fallback)
    return (
        "CRITICAL: Write only {{char}} (and any other characters in the "
        "scene, if appropriate). Do not write dialogue, actions, reactions, "
        "or internal thoughts for {{user}}. If {{user}}'s character appears "
        "in the scene, describe them only from the outside — what {{char}} "
        "sees, hears, or feels about their presence — never their interior. "
        "End the reply on a natural beat that leaves room for {{user}}'s "
        "response without prompting them explicitly."
    )


def _build_length_anchor_note(length: str, role: str) -> str:
    """Author-note-position reinforcement for long/chapter generations.

    Author's note is injected close to recent messages (depth=4 default).
    Over multi-thousand-token chapter replies, the jailbreak reminder is
    thousands of tokens in the past by the end of generation — attention
    has decayed. This mid-stream anchor keeps the rule in recent context.

    Returns empty string for short/moderate — one jailbreak reminder suffices.
    """
    if length not in ("long", "chapter"):
        return ""
    if role == "writer":
        return (
            "This is an extended reply. As you write, stay on {{char}}'s "
            "side of the page — {{user}}'s co-author voice is reserved."
        )
    return (
        "This is an extended reply. Throughout, keep the camera on {{char}} "
        "(and other scene characters). Never drift into {{user}}'s "
        "interiority, speech, or authored action — {{user}} will write "
        "those themselves."
    )

_TENSE_SNIPPETS = {
    "past": "Write in the past tense (\"she walked\", \"he said\").",
    "present": "Write in the present tense (\"she walks\", \"he says\").",
    "future": "Write in the future tense (\"she will walk\", \"he will say\").",
}

_POV_SNIPPETS = {
    "first": "Use first-person narration (\"I walk\", \"I feel\").",
    "second": "Use second-person narration (\"you walk\", \"you feel\").",
    "third": "Use third-person narration (\"she walks\", \"he feels\").",
}

_POV_MODE_SNIPPETS = {
    "omniscient": (
        "Narrate with omniscient awareness — you may describe any "
        "character's thoughts, the wider world, and events {{char}} "
        "could not observe."
    ),
    "character": (
        "Stay tightly in {{char}}'s point of view — only describe what "
        "{{char}} can perceive, know, or reasonably infer."
    ),
    "user": (
        "Keep the narrative anchored on {{user}}'s experience — describe "
        "what {{user}} perceives and reacts to, not private interiority "
        "of other characters."
    ),
    "flexible": (
        "Shift perspective as the scene demands — usually close to "
        "{{char}}, but pull back for establishing beats when it serves "
        "the story."
    ),
}

_LENGTH_SNIPPETS = {
    "one_sentence": "Respond with exactly one sentence.",
    "short": "Write a short reply — about 150 words, one or two tight paragraphs.",
    "moderate": "Write a moderate reply — 150-300 words, two to three paragraphs.",
    "long": "Write a long reply — 300-600 words, rich with detail and scene-work.",
    "chapter": (
        "Write a chapter-length reply — 2000-8000 words. Treat this as a "
        "full narrative beat with multiple scenes, pacing variation, and "
        "a clear arc."
    ),
}

_CONTENT_SNIPPETS = {
    "sfw": (
        "Keep the content safe-for-work. Violence may be depicted but not "
        "gratuitously gory. Sexual content fades to black before "
        "explicit detail."
    ),
    "nsfw": (
        "Explicit content is permitted when the scene calls for it — "
        "depict intimacy, violence, and other adult themes directly and "
        "without euphemism. Stay true to character consent and the "
        "established tone."
    ),
}

# Tone snippets — collapses the former 5 flat style presets (Expressive,
# Natural Dialogue, Concise, Cinematic, Slow Burn) into a toggle dimension.
# "neutral" renders no extra line so users get a clean baseline.
_TONE_SNIPPETS = {
    "neutral": "",
    "expressive": (
        "Write with evocative prose. Show emotions through body language, "
        "micro-expressions, and sensory detail rather than stating them. "
        "Weave environment — sound, texture, light, scent — into each beat. "
        "Let silence and small gestures carry weight."
    ),
    "dialogue": (
        "Favor natural, distinctive dialogue over narration. Use narration "
        "only to convey tone, gesture, or brief scene-setting between lines "
        "of speech. Give characters consistent voices with authentic speech "
        "patterns, interruptions, and reactions."
    ),
    "concise": (
        "Be direct and punchy — every sentence should earn its place. "
        "Favor sharp dialogue and decisive action over description. Cut any "
        "line that doesn't move the scene forward or reveal character."
    ),
    "cinematic": (
        "Write visually. Frame scenes like camera shots — wide establishing, "
        "close-up on detail, reaction beats. Lean into physicality and "
        "movement. Let light, space, and motion amplify the emotion."
    ),
    "slowburn": (
        "Slow the pace. Linger on atmosphere and emotional undercurrents. "
        "Build tension through what is left unsaid — develop character "
        "through internal thought, hesitation, subtext, and small "
        "revealing moments."
    ),
}


def compose_modular_system_prompt(config: dict[str, object]) -> str:
    """Turn a modular_config dict into a coherent multi-paragraph system prompt.

    Unknown toggle values fall back to the MODULAR_DEFAULTS entry so the
    composer never raises on malformed input.
    """
    def pick(key: str, table: dict) -> str:
        value = str(config.get(key, MODULAR_DEFAULTS[key]))
        return table.get(value, table[str(MODULAR_DEFAULTS[key])])

    role = pick("role", _ROLE_SNIPPETS)
    tense = pick("tense", _TENSE_SNIPPETS)
    pov = pick("pov", _POV_SNIPPETS)
    pov_mode = pick("pov_mode", _POV_MODE_SNIPPETS)
    length = pick("length", _LENGTH_SNIPPETS)
    tone = pick("tone", _TONE_SNIPPETS)
    content = pick("content", _CONTENT_SNIPPETS)

    style_line = f"{tense} {pov} {pov_mode}"

    parts = [role, style_line, length]
    if tone:  # neutral renders empty, skip the blank paragraph
        parts.append(tone)
    parts.append(content)
    return "\n\n".join(parts)


def _build_anti_slop_directive(phrases: list[str]) -> str:
    """Render an anti-slop directive from a phrase list. Empty when list empty."""
    if not phrases:
        return ""
    # Cap at 60 to keep the jailbreak under control. Users can split across
    # multiple presets if they want more.
    shown = phrases[:60]
    joined = "; ".join(shown)
    return (
        "Avoid the following over-used phrases and any close paraphrase of "
        f"them — they are slop: {joined}. "
        "Write concretely and specifically instead."
    )


def apply_preset(
    request: InternalChatRequest,
    preset: PromptPreset,
) -> InternalChatRequest:
    """Apply a prompt preset to a request by injecting at the correct positions.

    Injection order (matches SillyTavern's proven model):
      1. Prepend preset.system_prompt to the first system message
      2. Insert author_note as system message at depth N from end
      3. Insert post_history as system message before final user message
      4. Insert jailbreak as system message after final user message

    Modular behavior: if preset has a non-empty modular_config, the composed
    prompt replaces system_prompt for this invocation (preset row is not
    mutated). Anti-slop phrases are appended to jailbreak.

    Returns a new request with modified messages (does not mutate input).
    """
    # Resolve modular composition, user-protection, length anchor, and
    # anti-slop before injection. The returned preset is a shallow copy so
    # we don't mutate the stored row.
    effective_system_prompt = preset.system_prompt
    effective_jailbreak = preset.jailbreak
    effective_author_note = preset.author_note
    modular_cfg = preset.load_modular_config()
    if modular_cfg:
        effective_system_prompt = compose_modular_system_prompt(modular_cfg)
        role = str(modular_cfg.get("role", "roleplayer"))
        length = str(modular_cfg.get("length", "moderate"))

        # User-protection always injected at jailbreak position for modular
        # presets. This is the strongest slot for instruction adherence and
        # the community-proven fix for "model speaks for user" drift.
        user_protect = _build_user_protection_directive(role)
        effective_jailbreak = (
            f"{effective_jailbreak}\n\n{user_protect}".strip()
        )

        # For long/chapter generations, add a mid-stream anchor via the
        # author's note so the rule stays in recent context as attention
        # decays over thousands of generated tokens.
        anchor = _build_length_anchor_note(length, role)
        if anchor:
            effective_author_note = (
                f"{effective_author_note}\n\n{anchor}".strip()
                if effective_author_note else anchor
            )

        if modular_cfg.get("anti_slop", True):
            slop = _build_anti_slop_directive(preset.load_anti_slop_phrases())
            if slop:
                effective_jailbreak = (
                    f"{effective_jailbreak}\n\n{slop}".strip()
                )
    else:
        # Non-modular presets still get anti-slop applied if the list is set.
        slop = _build_anti_slop_directive(preset.load_anti_slop_phrases())
        if slop:
            effective_jailbreak = (
                f"{effective_jailbreak}\n\n{slop}".strip()
            )

    def _prepend_system(target: list[Message]) -> list[Message]:
        """Merge the preset system prompt into the first system message
        (or prepend one). Factored so the SAME transform applies to both
        the live message list and the ``kv_stable_messages`` snapshot —
        the prewarmed checkpoint prefix must be byte-identical to the
        head of the next real turn's payload, and the preset system
        prompt is part of that head. Before this, the snapshot was
        prewarmed WITHOUT the preset text, so its rendered prefix never
        matched any real turn (measured ~10% token LCP live) and every
        checkpoint prefill was wasted work."""
        if not effective_system_prompt:
            return target
        for i, msg in enumerate(target):
            if msg.role == "system":
                target[i] = Message(
                    role="system",
                    content=effective_system_prompt + "\n\n" + msg.content,
                    images=msg.images,
                    tool_calls=msg.tool_calls,
                )
                return target
        target.insert(0, Message(role="system", content=effective_system_prompt))
        return target

    messages = [
        Message(role=m.role, content=m.content, images=m.images, tool_calls=m.tool_calls)
        for m in request.messages
    ]

    # 1. System prompt prepend (shared transform — see _prepend_system)
    messages = _prepend_system(messages)

    stable_messages = request.kv_stable_messages
    if stable_messages:
        stable_messages = _prepend_system([
            Message(
                role=m.role, content=m.content,
                images=m.images, tool_calls=m.tool_calls,
            )
            for m in stable_messages
        ])

    # Find last user message index (needed for post_history and jailbreak)
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    # 2. Author's note at depth
    if effective_author_note:
        # Count non-system messages from end to find insertion point
        depth = preset.author_note_depth
        non_system_count = 0
        insert_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role != "system":
                non_system_count += 1
                if non_system_count >= depth:
                    insert_idx = i
                    break
        note_content = f"[Author's Note: {effective_author_note}]"
        messages.insert(insert_idx, Message(role="system", content=note_content))
        # Recalculate last_user_idx since we inserted
        if insert_idx <= last_user_idx:
            last_user_idx += 1

    # 3. Post-history (before final user message)
    if preset.post_history and last_user_idx >= 0:
        messages.insert(last_user_idx, Message(role="system", content=preset.post_history))
        last_user_idx += 1  # shifted by insertion

    # 4. Jailbreak (after final user message)
    if effective_jailbreak and last_user_idx >= 0:
        messages.insert(last_user_idx + 1, Message(role="system", content=effective_jailbreak))

    # Use ``dataclass_replace`` so EVERY field on the input request is
    # copied automatically — only ``messages`` changes here. The previous
    # explicit-field-list pattern silently dropped fields that landed on
    # ``InternalChatRequest`` after this code was written, including the
    # narrative-critical ``kv_session_key`` / ``kv_stable_messages`` /
    # ``kv_mode`` (KV checkpoint), ``lorebook``, ``group_id``,
    # ``speaker_override``, ``voice_input``, ``explicit_flow_name``.
    # The bug surfaced 2026-04-27 as ``prepare_stable_checkpoint``
    # silently no-oping on every narrative turn — kv_stable_messages
    # set by NarrativeEngine._augment_request was discarded here, so
    # the LLM-side prefill was never being saved to disk and every
    # turn paid full prefill cost.
    return dataclass_replace(
        request, messages=messages, kv_stable_messages=stable_messages,
    )
