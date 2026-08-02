"""LLM-based memory extraction with reconciliation against existing memories."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.memory.models import ExtractedFact, MemoryType
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

# Max chars of combined input to avoid overwhelming the LLM.
# Increased from 2000 because we now send full conversation + existing memories.
_MAX_INPUT_CHARS = 8000

# ---------------------------------------------------------------------------
# Unified extraction prompt
#
# Design philosophy: ONE prompt for all model sizes.  The prompt asks for
# 3 core fields (content, type, importance) and the parser gracefully
# accepts anything from a plain string to a full object.  This keeps
# cognitive load low enough for 8B models while preserving the typed/scored
# output that powers the core profile, recall ranking, and tier system.
#
# Fields NOT asked for in the prompt but derived downstream:
#   - confidence: defaults to 0.8 for all LLM extractions (self-rated
#     confidence was never reliable anyway)
#   - evidence: derived by finding the closest user message substring
#     (more reliable than LLM quoting, which often paraphrases)
#   - action/ref: reconciliation is handled by the store's embedding
#     dedup (0.92 cosine), with an optional reconciliation appendix
#     added to the prompt only for large models with existing memories.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You extract personal facts about the USER from conversations.

Extract what the user says about themselves. Do NOT extract from assistant \
messages, greetings, or task requests ("write me X").

CRITICAL — DO NOT EXTRACT properties of things the user is building, making, \
or describing. These are artifact specs, not user facts. The test: would this \
fact still be useful in an unrelated conversation 3 months from now? If it's \
only meaningful in the context of the current project, skip it.

Skip artifact descriptions: "the game must run on a web server", "the level \
kills the player", "the spaceship is invisible", "the API needs auth", \
"the report should be detailed". Subject is the artifact, not the user.

Skip in-progress task state: "currently fixing auth", "working on level 2", \
"deploying to staging tonight". That's session context, not durable identity.

Skip facts about third parties unless the user explicitly states a durable \
relationship: "my wife runs a bakery" (relationship — keep), "the customer \
wants blue" (project context — skip).

CRITICAL — DO NOT extract anything about a topic, character, person, place, \
or thing the user is ASKING ABOUT, or that comes from a search / lookup / \
article in the assistant's reply. "Tell me about X" / "who is X" means X is a \
TOPIC the user is curious about — never a fact about the user, and never \
something the user "is", even if the answer describes X in detail. Only \
extract what the user states about THEIR OWN life in THEIR OWN words.

Return JSON: {"facts": [{"content": "...", "type": "...", "importance": 0.0-1.0, "durability": "..."}]}
Return {"facts": []} if nothing personal.

Types: fact (personal info), preference (likes/dislikes), skill (expertise), \
entity (project/person/org the USER is connected to), relationship (connection to someone)
Importance: 1.0 = core identity, 0.7 = significant, 0.4 = minor detail

Durability — one of "durable" | "transient" | "unknown":
- "durable": still true a year from now (identity, allergies, durable preferences, \
relationships, location, name, language, religion, occupation if stable).
- "transient": true now but likely changes in weeks (current project, sprint focus, \
mood today, what they're cooking tonight, which chapter they're drafting).
- "unknown": you genuinely can't tell — pick this rather than guessing.

USER: Hi, how are you?
→ {"facts": []}

USER: Can you help me write a cover letter?
→ {"facts": []}

USER: The game must run on a web server with a public IP.
→ {"facts": []}

USER: Tell me about the main characters in that book.
→ {"facts": []}

USER: Who is Ada Lovelace and what did she do?
→ {"facts": []}

USER: Walls kill the player on contact and there's a black hole in level 2.
→ {"facts": []}

USER: I live in Seattle and I love spicy food.
→ {"facts": [{"content": "Lives in Seattle", "type": "fact", "importance": 0.8, "durability": "durable"}, \
{"content": "Loves spicy food", "type": "preference", "importance": 0.5, "durability": "durable"}]}

USER: I'm a backend dev at a fintech startup, mostly Go and PostgreSQL.
→ {"facts": [{"content": "Backend developer at a fintech startup", "type": "fact", \
"importance": 0.85, "durability": "durable"}, {"content": "Works with Go and PostgreSQL", "type": "skill", \
"importance": 0.7, "durability": "durable"}]}

USER: I'm building an AI app with voice and memory features for my wife's business.
→ {"facts": [{"content": "Building an AI app with voice and memory features", \
"type": "entity", "importance": 0.6, "durability": "transient"}, {"content": "Wife has a business", \
"type": "relationship", "importance": 0.5, "durability": "durable"}]}

USER: I'm a game developer who's been shipping web games for 10 years, currently \
working on an FPS where walls kill the player.
→ {"facts": [{"content": "Game developer with 10 years of web games experience", \
"type": "fact", "importance": 0.9, "durability": "durable"}]}

USER: Drafting chapter 12 of my novel today, the antagonist finally shows up.
→ {"facts": [{"content": "Is writing a novel", "type": "entity", "importance": 0.7, "durability": "durable"}, \
{"content": "Currently drafting chapter 12", "type": "fact", "importance": 0.3, "durability": "transient"}]}
"""

# Appended to the user prompt when existing memories are available AND the
# model is large enough to handle reconciliation.  Small models skip this
# entirely — the store's cosine dedup handles duplicates instead.
_RECONCILIATION_APPENDIX = """\

Existing memories:
{memories}

If a fact above is already known, omit it. If one is outdated and the user \
explicitly corrected it, include the updated version."""

# Parameter thresholds (by name substring) for detecting small models.
# Small models skip the reconciliation appendix (too much cognitive load)
# but use the same core prompt and output format as large models.
_SMALL_MODEL_PATTERNS = (
    "1b", "1.5b", "2b", "3b", "4b", "7b", "8b", "9b",
    "10b", "11b", "12b", "13b", "14b",
    "mini", "tiny", "nano", "small", "lite",
    "phi-2", "phi-3", "phi-4",
    "gemma-2b", "gemma-7b",
)


def _is_small_model(model: str) -> bool:
    """Heuristic: check if model name suggests a small model."""
    if not model:
        return False
    ml = model.lower().replace("_", "-").replace(" ", "-")
    for pattern in _SMALL_MODEL_PATTERNS:
        if pattern in ml:
            return True
    return False


_MODE_INSTRUCTIONS: dict[str, str] = {
    "passthrough": (
        "Extract personal facts the user states about themselves in this conversation."
    ),
    "analytical": (
        "Extract personal facts the user states about themselves. "
        "Domain expertise and tool preferences count if the user claims them."
    ),
    "agentic": (
        "Extract personal facts the user states about themselves. "
        "Workflow preferences and tools count if the user claims them. "
        "Do not extract individual task steps."
    ),
}

_VALID_TYPES = {"fact", "preference", "entity", "skill", "relationship"}

_TYPE_MAP: dict[str, MemoryType] = {
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "entity": MemoryType.ENTITY,
    "skill": MemoryType.SKILL,
    "relationship": MemoryType.RELATIONSHIP,
}


async def llm_extract(
    user_message: str,
    assistant_response: str,
    backend: ModelBackend,
    model: str,
    mode: str = "passthrough",
    existing_memories: list[dict] | None = None,
) -> list[ExtractedFact]:
    """Extract facts from a single conversation turn using an LLM.

    Returns an empty list on any failure (timeout, bad JSON, backend error).
    """
    return await llm_extract_batch(
        [(user_message, assistant_response)],
        backend,
        model,
        mode,
        existing_memories=existing_memories,
    )


async def llm_extract_batch(
    pairs: list[tuple[str, str]],
    backend: ModelBackend,
    model: str,
    mode: str = "passthrough",
    existing_memories: list[dict] | None = None,
) -> list[ExtractedFact]:
    """Extract facts from multiple conversation turns in a single LLM call.

    Each pair is (user_message, assistant_response). Both sides are included
    in the prompt so the LLM has full context, but assistant messages are
    labeled as CONTEXT ONLY.

    When *existing_memories* is provided, the prompt includes them as a
    numbered list so the LLM can reference them for SKIP/UPDATE actions
    instead of creating duplicates.

    Returns an empty list on any failure.
    """
    from augmentum.models.base import InternalChatRequest, Message

    if not pairs:
        return []

    small = _is_small_model(model)
    mode_instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["passthrough"])

    # Build conversation block with both user and assistant turns.
    turns: list[str] = []
    total_chars = 0
    # Reserve budget for reconciliation appendix (large models only).
    memory_budget = (min(1000, _MAX_INPUT_CHARS // 4) if existing_memories and not small else 0)
    convo_budget = _MAX_INPUT_CHARS - memory_budget

    for user_msg, asst_resp in pairs:
        per_turn = convo_budget // max(len(pairs), 1)
        half = per_turn // 2

        u = user_msg[:half]
        a = asst_resp[:half]
        turns.append(f"USER: {u}")
        turns.append(f"ASSISTANT [CONTEXT ONLY]: {a}")
        total_chars += len(u) + len(a)
        if total_chars > convo_budget:
            break

    conversation = "\n".join(turns)

    # Build user prompt: mode instruction + conversation + optional reconciliation
    user_prompt = f"{mode_instruction}\n\nConversation:\n{conversation}"

    # Large models get reconciliation appendix when existing memories available
    if existing_memories and not small:
        mem_lines: list[str] = []
        for i, mem in enumerate(existing_memories, 1):
            content = str(mem.get("content", ""))[:200]
            mem_lines.append(f"- {content}")
            if sum(len(line) for line in mem_lines) > memory_budget:
                break
        if mem_lines:
            user_prompt += _RECONCILIATION_APPENDIX.format(
                memories="\n".join(mem_lines),
            )

    log.info(
        "llm_extraction_starting",
        model=model,
        small_model=small,
        pairs=len(pairs),
    )

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        stream=False,
        temperature=0.1,
        max_tokens=800,
    )

    user_messages = [user_msg for user_msg, _asst in pairs]

    timeout = settings.tool_execution_timeout
    try:
        response = await asyncio.wait_for(backend.chat(request), timeout=timeout)
        raw = response.message.content
        return _parse_extraction_response(
            raw,
            existing_memories=existing_memories,
            user_messages=user_messages,
        )
    except asyncio.TimeoutError:
        log.warning("llm_extraction_timeout", mode=mode, turns=len(pairs), timeout_s=timeout, model=model)
        return []
    except Exception:
        log.warning("llm_extraction_failed", mode=mode, turns=len(pairs), exc_info=True)
        return []


def _derive_evidence(content: str, user_messages: list[str]) -> str:
    """Find the user message substring that best matches the extracted fact.

    More reliable than LLM-quoted evidence because it matches against the
    actual messages rather than trusting the model to quote accurately.
    """
    if not user_messages:
        return ""

    content_words = {w.lower() for w in content.split() if len(w) > 3}
    if not content_words:
        return ""

    best_msg = ""
    best_overlap = 0
    for msg in user_messages:
        msg_words = {w.lower() for w in msg.split() if len(w) > 3}
        overlap = len(content_words & msg_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_msg = msg

    if not best_msg or best_overlap < 2:
        return ""

    # Return a trimmed snippet (max 120 chars) centered on the matching area
    if len(best_msg) <= 120:
        return best_msg.strip()
    # Find the first matching word's position and extract a window
    for w in content_words:
        pos = best_msg.lower().find(w)
        if pos >= 0:
            start = max(0, pos - 30)
            end = min(len(best_msg), pos + 90)
            snippet = best_msg[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(best_msg):
                snippet = snippet + "..."
            return snippet
    return best_msg[:120].strip()


def _parse_extraction_response(
    raw: str,
    existing_memories: list[dict] | None = None,
    user_messages: list[str] | None = None,
) -> list[ExtractedFact]:
    """Parse LLM extraction response into ExtractedFact list.

    Accepts multiple output shapes:
    - {"facts": ["string", ...]}           — plain strings (any model)
    - {"facts": [{"content": "..."}]}      — partial objects (fills defaults)
    - {"facts": [{"content": "...", "type": "...", "importance": 0.8}]} — full
    - {"memories": [...]}                  — legacy format (backward compat)

    Evidence is derived from user messages rather than trusting LLM quoting.
    Confidence defaults to 0.8 for all LLM extractions.
    """
    text = raw.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("llm_extraction_json_parse_failed", raw=text[:200])
        return []

    if not isinstance(data, dict):
        return []

    # Support "facts" (new unified) and "memories" (legacy) keys.
    raw_items = data.get("facts") or data.get("memories") or []
    if not isinstance(raw_items, list):
        return []

    facts: list[ExtractedFact] = []
    for item in raw_items:
        # Plain string: any model can produce this
        if isinstance(item, str):
            item = item.strip()
            if len(item) >= 5:
                evidence = _derive_evidence(item, user_messages or [])
                facts.append(ExtractedFact(
                    content=item,
                    type=MemoryType.FACT,
                    importance=0.6,
                    confidence=0.8,
                    is_explicit=False,
                    source_context={"extraction": "llm", "action": "add"},
                    evidence=evidence,
                ))
            continue

        if not isinstance(item, dict):
            continue

        action = str(item.get("action", "ADD")).upper()

        # SKIP actions produce no output fact.
        if action == "SKIP":
            continue

        content = str(item.get("content", "")).strip()
        if len(content) < 5:
            continue

        fact_type = str(item.get("type", "fact")).lower()
        if fact_type not in _VALID_TYPES:
            fact_type = "fact"

        importance = _clamp(item.get("importance", 0.6))
        # Confidence always defaults to 0.8 — LLM self-rated confidence is unreliable
        confidence = 0.8
        # Durability — LLM-emitted lifecycle judgment. Unknown is the safe
        # default for parse failures or omitted fields.
        from augmentum.memory.models import Durability
        raw_dur = str(item.get("durability", "unknown")).strip().lower()
        try:
            durability = Durability(raw_dur)
        except ValueError:
            durability = Durability.UNKNOWN
        # Derive evidence from actual user messages (more reliable than LLM quoting)
        evidence = _derive_evidence(content, user_messages or [])

        # Build source_context based on action.
        source_ctx: dict = {"extraction": "llm"}
        if action == "UPDATE":
            ref = item.get("ref")
            if ref is not None:
                try:
                    ref_idx = int(ref)
                except (TypeError, ValueError):
                    ref_idx = None
                if ref_idx is not None:
                    source_ctx["action"] = "update"
                    source_ctx["ref"] = ref_idx
                    # Attach the existing memory id if available.
                    if (
                        existing_memories
                        and 1 <= ref_idx <= len(existing_memories)
                    ):
                        existing_id = existing_memories[ref_idx - 1].get("id")
                        if existing_id:
                            source_ctx["target_memory_id"] = existing_id
        elif action == "ADD":
            source_ctx["action"] = "add"

        facts.append(ExtractedFact(
            content=content,
            type=_TYPE_MAP[fact_type],
            importance=importance,
            confidence=confidence,
            is_explicit=False,
            source_context=source_ctx,
            evidence=evidence,
            durability=durability,
        ))

    return _dedup_batch(facts)


def _dedup_batch(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Remove near-duplicate facts within a single extraction batch.

    Compares normalized content strings. When two facts are very similar,
    keeps the one with higher importance (or first if tied).
    """
    if len(facts) <= 1:
        return facts

    def _normalize(s: str) -> str:
        import re
        s = s.lower().strip()
        # Strip possessives, punctuation, articles
        s = re.sub(r"'s\b", "", s)
        s = re.sub(r"[''\".,!?;:\-()]", "", s)
        s = re.sub(r"\b(a|an|the|some|this|that)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        # Normalize common subject prefixes
        for prefix in ("user is ", "user has ", "user was ", "user ",
                       "interested in ", "familiar with ", "knows about ",
                       "works with ", "uses "):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        return s

    kept: list[ExtractedFact] = []
    seen_normalized: list[str] = []

    for fact in facts:
        norm = _normalize(fact.content)
        # Check against all already-kept facts
        is_dup = False
        for i, existing_norm in enumerate(seen_normalized):
            # Simple overlap check: if one contains most of the other's words
            words_new = set(norm.split())
            words_existing = set(existing_norm.split())
            if not words_new or not words_existing:
                continue
            overlap = len(words_new & words_existing)
            smaller = min(len(words_new), len(words_existing))
            if smaller > 0 and overlap / smaller >= 0.6:
                # Keep the one with higher importance
                if fact.importance > kept[i].importance:
                    kept[i] = fact
                    seen_normalized[i] = norm
                is_dup = True
                break

        if not is_dup:
            kept.append(fact)
            seen_normalized.append(norm)

    return kept


def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi], handling non-numeric gracefully."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(lo, min(hi, v))
