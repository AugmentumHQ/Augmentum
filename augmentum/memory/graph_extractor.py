"""Knowledge graph extraction — LLM-based entity and relationship extraction.

Card-type-aware extraction prompts tailor the graph to narrative context:
CHARACTER cards focus on relationship dynamics and emotional arcs,
NARRATOR cards focus on world state, quests, and faction relationships,
ENSEMBLE cards focus on inter-character dynamics and group structure.
Non-narrative modes use a universal prompt.

Produces structured node/edge updates for the KnowledgeGraph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.graph import KnowledgeGraph
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

_MAX_INPUT_CHARS = 2000

# ---------------------------------------------------------------------------
# Base rules shared by all prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
Return a JSON object with an "updates" array. Each update is one of:
- {"type": "node", "label": "...", "kind": "person|place|thing|concept|event", "properties": {}}
- {"type": "edge", "source": "...", "target": "...", "relation": "...", "weight": 0.0-1.0}

Rules:
- Weight reflects certainty: stated fact = 1.0, implied = 0.7, speculated = 0.4
- Resolve pronouns to named entities using context (don't extract "she" as a node)
- Only extract what is explicitly stated or strongly implied
- If nothing meaningful to extract, return {"updates": []}
- Keep labels concise (1-4 words)
- Return ONLY valid JSON, no markdown fences"""

# ---------------------------------------------------------------------------
# Universal prompt (non-narrative modes)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are a knowledge graph extraction system. Given a conversation message \
and recent context, extract entities and relationships as structured JSON.

{_BASE_RULES}

- Extract entities (people, places, objects, concepts, events) as nodes
- Extract relationships between entities as edges
- Use natural language for relations ("trusts", "located_in", "depends_on", "wrote")"""

# ---------------------------------------------------------------------------
# Card-type-aware prompts (narrative mode)
# ---------------------------------------------------------------------------

_CHARACTER_PROMPT = f"""\
You are a knowledge graph extractor for a character-driven roleplay narrative.
Focus on interpersonal dynamics and the protagonist's world.

{_BASE_RULES}

Priority extractions (most important first):
- **Relationship edges**: trust, affection, tension, loyalty, rivalry, fear, \
attraction, resentment between named characters (weight by intensity)
- **Emotional state edges**: "feels_toward", "jealous_of", "protective_of", \
"angry_at" — capture shifting emotions between characters
- **Shared experiences**: events that bonded or divided characters \
("survived_together", "betrayed", "confided_in", "fought_alongside")
- **Character knowledge**: what each character knows or has learned about \
others ("knows_secret_of", "discovered", "suspects")
- **Personal items/places**: meaningful objects or locations tied to the \
relationship ("gifted", "lives_at", "carries")

Node kinds to prefer: person, emotion, secret, memory, item, place
Relation style: use evocative verbs ("yearns_for" not "has_relationship_with")
Capture CHANGE — if a relationship shifted this turn, extract the new state."""

_NARRATOR_PROMPT = f"""\
You are a knowledge graph extractor for a world-building / RPG narrative.
Focus on world state, factions, quests, and systemic relationships.

{_BASE_RULES}

Priority extractions (most important first):
- **Quest/objective edges**: "assigned_by", "requires", "rewards", \
"blocks", "advances" — track mission dependencies and progress
- **Faction/allegiance edges**: "allied_with", "hostile_to", "member_of", \
"rules", "serves", "rebelling_against" — political and power structures
- **Location edges**: "located_in", "connects_to", "controls", \
"discovered", "traveled_to" — geographic and strategic relationships
- **NPC state edges**: "guards", "sells", "knows_about", "wounded_by", \
"seeking" — NPC roles and current conditions
- **Lore/world rule nodes**: magical systems, laws, prophecies, artifacts \
as concept nodes with edges to who/what they affect
- **Temporal edges**: "happened_before", "caused", "triggered" — event chains

Node kinds to prefer: person, place, faction, quest, item, event, concept
Relation style: use strategic verbs ("controls" not "is_near")
Track POWER shifts — who gained/lost influence, territory, or resources."""

_ENSEMBLE_PROMPT = f"""\
You are a knowledge graph extractor for a multi-character ensemble narrative.
Focus on group dynamics, inter-character relationships, and role distribution.

{_BASE_RULES}

Priority extractions (most important first):
- **Inter-character edges**: every pair of named characters should have \
relationship edges when they interact ("trusts", "rivals_with", \
"mentors", "secretly_loves", "annoyed_by", "protects")
- **Group role edges**: "leads", "follows", "mediates", "comic_relief", \
"strategist" — who plays what role in the group
- **Alliance/conflict subgroups**: "sided_with", "argued_against", \
"excluded_from" — internal faction dynamics
- **Character-to-event edges**: who did what ("proposed_plan", \
"caused_problem", "saved", "discovered", "refused")
- **Dialogue attribution**: "said_to", "confided_in", "lied_to", \
"agreed_with" — track who communicates with whom
- **Status/capability edges**: "injured", "skilled_at", "carrying", \
"missing" — individual character states

Node kinds to prefer: person, group, event, conflict, goal, item
Relation style: capture the social verb ("teased" not "interacted_with")
Prioritize EVERY character mentioned — don't let supporting cast vanish."""

_CARD_TYPE_PROMPTS = {
    "character": _CHARACTER_PROMPT,
    "narrator": _NARRATOR_PROMPT,
    "ensemble": _ENSEMBLE_PROMPT,
}


@dataclass
class GraphUpdate:
    """A single graph update (node or edge)."""

    update_type: str  # "node" or "edge"
    label: str = ""
    kind: str = "thing"
    properties: dict = field(default_factory=dict)
    source: str = ""
    target: str = ""
    relation: str = ""
    weight: float = 0.5


async def extract_graph_updates(
    user_message: str,
    assistant_response: str,
    backend: ModelBackend,
    model: str = "",
    recent_context: list[str] | None = None,
    card_type: str | None = None,
) -> list[GraphUpdate]:
    """Extract knowledge graph updates from a conversation turn.

    Args:
        card_type: Narrative card type ("character", "narrator", "ensemble")
                   or None for non-narrative modes. Selects a card-type-aware
                   extraction prompt that prioritises the right entities and
                   relationship styles for the narrative context.

    Returns a list of GraphUpdate objects (nodes and edges).
    """
    # Select the right system prompt
    system_prompt = _CARD_TYPE_PROMPTS.get(card_type, _SYSTEM_PROMPT) if card_type else _SYSTEM_PROMPT

    # Build the user prompt
    parts = []
    if recent_context:
        context_text = "\n".join(
            f"[{i + 1}] {msg[:300]}" for i, msg in enumerate(recent_context[-3:])
        )
        parts.append(f"Recent context:\n{context_text}")

    # Escape braces to prevent .format() injection
    safe_user = user_message[:_MAX_INPUT_CHARS].replace("{", "{{").replace("}", "}}")
    safe_assistant = assistant_response[:_MAX_INPUT_CHARS].replace("{", "{{").replace("}", "}}")

    parts.append(f"User message:\n{safe_user}")
    parts.append(f"Assistant response:\n{safe_assistant}")

    user_prompt = "\n\n".join(parts)

    try:
        from augmentum.models.base import InternalChatRequest, Message

        request = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            stream=False,
            temperature=0.2,
        )
        response = await backend.chat(request)

        raw = response.get("message", {}).get("content", "")
        if not raw:
            return []

        return _parse_extraction_response(raw)

    except Exception:
        log.debug("kg_extraction_failed", exc_info=True)
        return []


def _parse_extraction_response(raw: str) -> list[GraphUpdate]:
    """Parse the LLM's JSON response into GraphUpdate objects."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.debug("kg_extraction_parse_failed", raw=raw[:200])
        return []

    updates_raw = data.get("updates", [])
    if not isinstance(updates_raw, list):
        return []

    results = []
    for item in updates_raw:
        if not isinstance(item, dict):
            continue

        update_type = item.get("type", "")

        if update_type == "node":
            label = str(item.get("label", "")).strip()
            if len(label) < 2:
                continue
            results.append(GraphUpdate(
                update_type="node",
                label=label,
                kind=str(item.get("kind", "thing")).lower(),
                properties=item.get("properties", {}),
            ))

        elif update_type == "edge":
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "")).strip()
            if not source or not target or not relation:
                continue
            weight = float(item.get("weight", 0.5))
            weight = max(0.0, min(1.0, weight))
            results.append(GraphUpdate(
                update_type="edge",
                source=source,
                target=target,
                relation=relation,
                weight=weight,
            ))

    return results


async def apply_graph_updates(
    updates: list[GraphUpdate],
    graph: KnowledgeGraph,
    chat_id: str | None = None,
    user_id: str = "default",
    message_idx: int | None = None,
) -> dict[str, int]:
    """Apply extracted updates to the knowledge graph.

    Returns stats: {nodes_created, nodes_merged, edges_created, edges_reinforced}.
    """
    stats = {"nodes_created": 0, "nodes_merged": 0, "edges_created": 0, "edges_reinforced": 0}

    # First pass: create/merge all nodes
    node_map: dict[str, str] = {}  # label -> node_id

    for update in updates:
        if update.update_type != "node":
            continue

        existing = await graph.find_node(update.label, chat_id=chat_id, user_id=user_id)
        if existing:
            node_map[update.label] = existing.id
            stats["nodes_merged"] += 1
        else:
            node = await graph.upsert_node(
                label=update.label,
                kind=update.kind,
                chat_id=chat_id,
                user_id=user_id,
                properties=update.properties,
            )
            node_map[update.label] = node.id
            stats["nodes_created"] += 1

    # Second pass: create/reinforce edges
    for update in updates:
        if update.update_type != "edge":
            continue

        # Resolve source and target to node IDs
        source_id = node_map.get(update.source)
        if not source_id:
            source_node = await graph.find_node(update.source, chat_id=chat_id, user_id=user_id)
            if not source_node:
                # Auto-create the source node
                source_node = await graph.upsert_node(
                    label=update.source, chat_id=chat_id, user_id=user_id,
                )
                stats["nodes_created"] += 1
            source_id = source_node.id
            node_map[update.source] = source_id

        target_id = node_map.get(update.target)
        if not target_id:
            target_node = await graph.find_node(update.target, chat_id=chat_id, user_id=user_id)
            if not target_node:
                target_node = await graph.upsert_node(
                    label=update.target, chat_id=chat_id, user_id=user_id,
                )
                stats["nodes_created"] += 1
            target_id = target_node.id
            node_map[update.target] = target_id

        edge = await graph.upsert_edge(
            source_id=source_id,
            target_id=target_id,
            relation=update.relation,
            weight=update.weight,
            chat_id=chat_id,
            message_idx=message_idx,
        )
        if edge.created_at == edge.updated_at:
            stats["edges_created"] += 1
        else:
            stats["edges_reinforced"] += 1

    if any(v > 0 for v in stats.values()):
        log.info("kg_updates_applied", **stats)

    return stats
