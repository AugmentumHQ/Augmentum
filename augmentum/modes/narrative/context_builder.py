"""Context builder — token-budget-aware prompt augmentation for narrative mode.

Assembles enhanced system prompts by injecting relevant state (character info,
scene context, active plots, lorebook entries, consistency flags) within a
configurable token budget.

Budget allocation (inspired by SillyTavern's approach):
- Character state: 25%
- Scene/world state: 15%
- Active plots: 15%
- Lorebook entries: 25%
- Consistency flags: 10%
- Reserve: 10%
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.modes.narrative.world_tracker import SceneState
from augmentum.state.narrative_state import (
    Contradiction,
    Entity,
    EntityType,
    Fact,
    LorebookEntry,
    LorebookPosition,
    PlotThread,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.tokenizer import count_tokens

log = get_logger(__name__)

# Budget allocation percentages — used as defaults, overridable via config
_BUDGET_CHARACTERS = 0.25
_BUDGET_SCENE = 0.15
_BUDGET_PLOTS = 0.15
_BUDGET_LOREBOOK = 0.25
_BUDGET_CONSISTENCY = 0.10
_BUDGET_RESERVE = 0.10


@dataclass
class ContextBlock:
    """A block of context to inject into the prompt."""

    label: str
    content: str
    priority: int = 100  # Lower = higher priority
    token_estimate: int = 0
    # Stable blocks hold content that does not change turn-to-turn
    # (card summary, example dialogue, tool guidance). They render into
    # ``BuiltContext.stable_text`` and the engine places them in the
    # STABLE head region (covered by the KV checkpoint) instead of the
    # per-turn floating injection — a static block that rides in the
    # per-turn tail re-prefills its full length every single turn.
    stable: bool = False

    def __post_init__(self) -> None:
        if not self.token_estimate:
            self.token_estimate = count_tokens(self.content)


@dataclass
class BlockDetail:
    """Detail about a context block for the request log."""

    label: str
    content: str
    token_estimate: int
    included: bool


@dataclass
class DepthEntry:
    """A lorebook entry destined for mid-history injection.

    Held separately from ``injected_text`` because it needs role-aware
    splicing into the messages array rather than being folded into the
    main system block. See ``NarrativeEngine._augment_request`` for the
    SillyTavern-compatible splice algorithm.
    """

    content: str
    depth: int = 4
    role: str = "system"          # "system" | "user" | "assistant"
    order: int = 100              # lower = inserted first within same (depth, role)
    label: str = "lore_at_depth"  # for the request inspector
    token_estimate: int = 0


@dataclass
class BuiltContext:
    """Result of context building."""

    injected_text: str = ""
    # Turn-stable blocks (card summary / example dialogue / tool
    # guidance) — placed in the stable head next to core lore so they
    # prefill once per session instead of once per turn.
    stable_text: str = ""
    blocks_used: list[str] = field(default_factory=list)
    total_tokens_estimate: int = 0
    budget_remaining: int = 0
    blocks_detail: list[BlockDetail] = field(default_factory=list)
    # At-depth entries pulled out so the engine can splice them into the
    # messages array at per-entry depths (matches ST's worldInfoDepth bucket).
    depth_entries: list[DepthEntry] = field(default_factory=list)


_RECALL_TOOLS_GUIDANCE = """\
You have memory tools available. Before writing about a character's current \
state, location, relationships, or any established facts, look them up first \
rather than guessing. Use recall_entity for a specific character or place, \
list_entities to see who is present, recall_facts to verify details, \
recall_plot_thread to check story arcs, and recall_archive to find earlier \
scenes that have left the conversation window."""

_LOREBOOK_TOOLS_GUIDANCE = """\
You have lorebook tools to manage the world's reference material.

CHECK (lorebook.check) before writing about any established location, \
character, faction, item, or rule — if an entry exists, use what it says. \
Do not contradict it.

CREATE (lorebook.create) a new entry when the story establishes something \
worth remembering: a named character and their key traits, a new location, \
a faction, a world rule, a significant event or its outcome, a named item. \
Do not record passing description or atmosphere — only details that would \
break consistency if forgotten later.
- keywords: the names and terms someone would mention when this lore matters \
again (a character's name, a location name, a faction name).
- content: write as a concise factual reference, not prose. \
Example: "Ashwander: main river through the Greyvale, fed by northern \
snowmelt, runs past the capital. Bridged at two points." \
Include current status, relationships, and distinguishing details.

UPDATE (lorebook.update) an existing entry when its facts change — a \
character dies, a location is destroyed, an alliance shifts, new \
information is revealed. Check first to find the entry id, then update. \
Do not create a duplicate.

DELETE (lorebook.delete) only when an entry is completely irrelevant and \
will never matter again. Prefer updating with enabled=false instead."""


def _build_tool_guidance(*, recall: bool, lorebook: bool) -> str:
    parts: list[str] = []
    if recall:
        parts.append(_RECALL_TOOLS_GUIDANCE)
    if lorebook:
        parts.append(_LOREBOOK_TOOLS_GUIDANCE)
    return "\n\n".join(parts)


class ContextBuilder:
    """Builds enhanced prompts for narrative mode with budget-aware context injection."""

    def __init__(
        self,
        token_budget: int = 4000,
        character_pct: float = _BUDGET_CHARACTERS,
        scene_pct: float = _BUDGET_SCENE,
        plot_pct: float = _BUDGET_PLOTS,
        lore_pct: float = _BUDGET_LOREBOOK,
        consistency_pct: float = _BUDGET_CONSISTENCY,
    ) -> None:
        self._budget = token_budget
        self._char_pct = character_pct
        self._scene_pct = scene_pct
        self._plot_pct = plot_pct
        self._lore_pct = lore_pct
        self._consistency_pct = consistency_pct

    def build(
        self,
        *,
        characters: list[Entity] | None = None,
        scene: SceneState | None = None,
        active_plots: list[PlotThread] | None = None,
        recent_facts: list[Fact] | None = None,
        lorebook_entries: list[LorebookEntry] | None = None,
        contradictions: list[Contradiction] | None = None,
        character_card_summary: str = "",
        state_text: str = "",
        example_dialogue: str = "",
        creator_notes: str = "",
        memory_text: str = "",
        relationship_summary: str = "",
        token_budget: int | None = None,
        recall_tools_enabled: bool = False,
        lorebook_tools_enabled: bool = False,
    ) -> BuiltContext:
        """Build the injected context within budget.

        token_budget overrides the instance default for this call only, so
        live changes to ``narrative_context_budget`` take effect without
        rebuilding the cached engine. Pass 0 for unlimited.
        """
        # Apply per-call budget override (None → use instance default)
        budget = self._budget if token_budget is None else token_budget
        blocks: list[ContextBlock] = []

        # Character card summary (highest priority)
        if character_card_summary:
            blocks.append(ContextBlock(
                label="character_card",
                content=character_card_summary,
                priority=10,
                stable=True,
            ))

        # Example dialogue — injected right after character card for few-shot style.
        # No hard truncation: user-authored content. If a budget is set, the
        # per-category allocation or overall _assemble() cap drops it cleanly.
        if example_dialogue:
            blocks.append(ContextBlock(
                label="example_dialogue",
                content=example_dialogue,
                priority=12,
                stable=True,
            ))

        # Creator notes / author's note — guidance for the AI.
        if creator_notes:
            blocks.append(ContextBlock(
                label="authors_note",
                content=creator_notes,
                priority=14,
            ))

        # State snapshot (highest priority after card — current scene)
        if state_text:
            blocks.append(ContextBlock(
                label="state_snapshot",
                content=state_text,
                priority=13,
            ))

        # Memory ledger (append-only historical events)
        if memory_text:
            blocks.append(ContextBlock(
                label="story_memory",
                content=memory_text,
                priority=15,
            ))

        # Character relationships
        if relationship_summary:
            blocks.append(ContextBlock(
                label="character_relationships",
                content=relationship_summary,
                priority=18,
            ))

        # Per-category budgets (0 = unlimited → use large fallback so nothing truncates)
        _unlimited = budget <= 0
        _cat = lambda pct: 999_999 if _unlimited else int(budget * pct)

        # Character states
        char_block = self._build_character_block(characters or [], _cat(self._char_pct))
        if char_block:
            blocks.append(char_block)

        # Scene/world state
        scene_block = self._build_scene_block(scene, _cat(self._scene_pct))
        if scene_block:
            blocks.append(scene_block)

        # Active plots
        plot_block = self._build_plot_block(active_plots or [], _cat(self._plot_pct))
        if plot_block:
            blocks.append(plot_block)

        # Lorebook entries — before_char/after_char go through the block
        # pipeline; at_depth entries bypass _assemble and travel in
        # BuiltContext.depth_entries for mid-history splicing by the engine.
        lore_blocks, lore_depth_entries = self._build_lorebook_blocks(
            lorebook_entries or [], _cat(self._lore_pct),
        )
        blocks.extend(lore_blocks)

        # Recent facts — suppressed when state is available,
        # since state captures the current situation.
        if recent_facts and not state_text:
            fact_block = self._build_facts_block(recent_facts)
            if fact_block:
                blocks.append(fact_block)

        # Consistency flags
        if contradictions:
            flag_block = self._build_consistency_block(contradictions, _cat(self._consistency_pct))
            if flag_block:
                blocks.append(flag_block)

        # Tool guidance — injected when recall or lorebook tools are on
        tool_guidance = _build_tool_guidance(
            recall=recall_tools_enabled,
            lorebook=lorebook_tools_enabled,
        )
        if tool_guidance:
            blocks.append(ContextBlock(
                label="tool_guidance",
                content=tool_guidance,
                priority=16,
                stable=True,
            ))

        # Sort by priority and assemble within budget
        blocks.sort(key=lambda b: b.priority)
        result = self._assemble(blocks, budget=budget)

        # Attach at-depth entries + report them in blocks_detail so the
        # request inspector surfaces them alongside the system-block entries.
        result.depth_entries = lore_depth_entries
        for de in lore_depth_entries:
            result.blocks_detail.append(BlockDetail(
                label=f"{de.label} (@D{de.depth}/{de.role})",
                content=de.content,
                token_estimate=de.token_estimate,
                included=True,
            ))
        return result

    def _assemble(self, blocks: list[ContextBlock], *, budget: int | None = None) -> BuiltContext:
        """Assemble blocks within the total token budget (0 = unlimited)."""
        used_blocks: list[ContextBlock] = []
        all_details: list[BlockDetail] = []
        total_tokens = 0
        budget = self._budget if budget is None else budget
        unlimited = budget <= 0

        for block in blocks:
            fits = unlimited or (total_tokens + block.token_estimate <= budget)
            all_details.append(BlockDetail(
                label=block.label,
                content=block.content,
                token_estimate=block.token_estimate,
                included=fits,
            ))
            if fits:
                used_blocks.append(block)
                total_tokens += block.token_estimate

        # Build final text with XML boundary tags per block. Stable
        # blocks render separately — the engine places them in the
        # checkpoint-covered head region; only genuinely per-turn
        # content rides in the floating injection (see ContextBlock.stable).
        parts = []
        stable_parts = []
        for block in used_blocks:
            rendered = f"<{block.label}>\n{block.content}\n</{block.label}>"
            (stable_parts if block.stable else parts).append(rendered)

        result = BuiltContext(
            injected_text="\n\n".join(parts),
            stable_text="\n\n".join(stable_parts),
            blocks_used=[b.label for b in used_blocks],
            total_tokens_estimate=total_tokens,
            budget_remaining=budget - total_tokens,
            blocks_detail=all_details,
        )

        log.debug(
            "context_built",
            blocks=result.blocks_used,
            tokens=result.total_tokens_estimate,
            remaining=result.budget_remaining,
        )

        return result

    def _build_character_block(
        self, characters: list[Entity], budget_tokens: int,
    ) -> ContextBlock | None:
        """Build character state context block."""
        chars = [c for c in characters if c.entity_type == EntityType.CHARACTER]
        if not chars:
            return None

        lines = []

        for char in chars:
            parts = [f"{char.name}:"]
            if char.state.location:
                parts.append(f"at {char.state.location}")
            if char.state.emotional_state:
                parts.append(f"feeling {char.state.emotional_state}")
            if char.state.physical_state:
                parts.append(f"({char.state.physical_state})")

            line = " ".join(parts)
            if count_tokens("\n".join(lines + [line])) > budget_tokens:
                break
            lines.append(line)

        if not lines:
            return None

        return ContextBlock(
            label="character_states",
            content="\n".join(lines),
            priority=30,
        )

    def _build_scene_block(
        self, scene: SceneState | None, budget_tokens: int,
    ) -> ContextBlock | None:
        """Build scene/world state context block."""
        if not scene:
            return None

        parts = []
        if scene.location:
            parts.append(f"Location: {scene.location}")
        if scene.time_of_day:
            parts.append(f"Time: {scene.time_of_day}")
        if scene.weather:
            parts.append(f"Weather: {scene.weather}")
        if scene.atmosphere:
            parts.append(f"Atmosphere: {scene.atmosphere}")
        if scene.present_characters:
            parts.append(f"Present: {', '.join(scene.present_characters)}")

        if not parts:
            return None

        content = "\n".join(parts)
        if count_tokens(content) > budget_tokens:
            # Trim to approximate length, then verify
            content = content[:budget_tokens * 4]
            while count_tokens(content) > budget_tokens:
                content = content[:int(len(content) * 0.85)]

        return ContextBlock(
            label="current_scene",
            content=content,
            priority=40,
        )

    def _build_plot_block(
        self, plots: list[PlotThread], budget_tokens: int,
    ) -> ContextBlock | None:
        """Build active plots context block."""
        if not plots:
            return None

        lines = []

        for plot in plots:
            line = f"- {plot.title}"
            if plot.description:
                line += f": {plot.description[:80]}"
            if count_tokens("\n".join(lines + [line])) > budget_tokens:
                break
            lines.append(line)

        if not lines:
            return None

        return ContextBlock(
            label="active_plots",
            content="\n".join(lines),
            priority=50,
        )

    def _build_lorebook_blocks(
        self, entries: list[LorebookEntry], budget_tokens: int,
    ) -> tuple[list[ContextBlock], list[DepthEntry]]:
        """Build lorebook entry context blocks.

        Returns a tuple: (system-block entries, at-depth entries).

        - BEFORE_CHAR / AFTER_CHAR → ContextBlock folded into the main
          system message (priority 8 / 60+).
        - AT_DEPTH → returned separately so the engine can splice them
          into the messages array at per-entry (depth, role). Still
          counts against the shared lorebook budget.
        """
        if not entries:
            return [], []

        blocks: list[ContextBlock] = []
        depth_entries: list[DepthEntry] = []
        remaining = budget_tokens

        # Position-based priority offsets for the system-block entries
        position_base = {
            LorebookPosition.BEFORE_CHAR: 8,
            LorebookPosition.AFTER_CHAR: 60,
            LorebookPosition.AT_DEPTH: 45,  # unused here; kept for callers
        }

        # Sort by priority (lower = higher priority)
        sorted_entries = sorted(entries, key=lambda e: e.priority)

        for entry in sorted_entries:
            estimate = count_tokens(entry.content)
            if estimate > remaining and not entry.ignore_budget:
                continue

            if entry.position == LorebookPosition.AT_DEPTH:
                # Emit as a depth entry; engine splices into messages later.
                depth_entries.append(DepthEntry(
                    content=entry.content,
                    depth=max(0, entry.injection_depth),
                    role=entry.injection_role or "system",
                    order=entry.priority,
                    label=f"lore_depth:{','.join(entry.keywords[:3])}",
                    token_estimate=estimate,
                ))
            else:
                base_priority = position_base.get(entry.position, 60)
                blocks.append(ContextBlock(
                    label=f"lore:{','.join(entry.keywords[:3])}",
                    content=entry.content,
                    priority=base_priority + entry.priority,
                    token_estimate=estimate,
                ))

            if not entry.ignore_budget:
                remaining -= estimate

        return blocks, depth_entries

    def _build_facts_block(self, facts: list[Fact]) -> ContextBlock | None:
        """Build recent facts context block."""
        if not facts:
            return None

        lines = [f"- {f.content}" for f in facts[:10]]
        return ContextBlock(
            label="established_facts",
            content="\n".join(lines),
            priority=55,
        )

    def _build_consistency_block(
        self, contradictions: list[Contradiction], budget_tokens: int,
    ) -> ContextBlock | None:
        """Build consistency warning block."""
        if not contradictions:
            return None

        # Only include recent/major contradictions
        recent = sorted(contradictions, key=lambda c: c.message_index, reverse=True)[:3]

        lines = ["WARNING — Consistency issues detected:"]
        for c in recent:
            lines.append(f"- [{c.severity.value}] {c.description}")

        content = "\n".join(lines)
        if count_tokens(content) > budget_tokens:
            content = content[:budget_tokens * 4]
            while count_tokens(content) > budget_tokens:
                content = content[:int(len(content) * 0.85)]

        return ContextBlock(
            label="consistency_warnings",
            content=content,
            priority=20,  # High priority
        )
