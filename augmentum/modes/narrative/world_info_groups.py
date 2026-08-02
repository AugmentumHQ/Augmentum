from __future__ import annotations

import random
from collections import defaultdict

from augmentum.state.narrative_state import LorebookEntry


def filter_by_groups(
    entries: list[LorebookEntry],
    *,
    seed: int | None = None,
) -> list[LorebookEntry]:
    """Filter activated entries through inclusion group logic.

    Rules:
    1. Ungrouped entries (group="") always pass through
    2. For each group name, only one entry survives:
       - If any entry has group_override=True, it wins
       - Otherwise, weighted random selection by group_weight
    3. Entries can belong to multiple groups (comma-separated)
    4. An entry selected by one group stays even if excluded by another

    Args:
        entries: List of already-activated entries to filter
        seed: Optional random seed for deterministic testing
    """
    if not entries:
        return []

    rng = random.Random(seed)

    ungrouped: list[LorebookEntry] = []
    # group_name -> list of entries in that group
    groups: dict[str, list[LorebookEntry]] = defaultdict(list)
    # Track which entries participate in any group
    grouped_ids: set[str] = set()

    for entry in entries:
        raw_group = entry.group.strip()
        if not raw_group:
            ungrouped.append(entry)
            continue

        group_names = [g.strip() for g in raw_group.split(",") if g.strip()]
        if not group_names:
            ungrouped.append(entry)
            continue

        grouped_ids.add(entry.id)
        for name in group_names:
            groups[name].append(entry)

    # Select one winner per group
    selected_ids: set[str] = set()

    for _group_name, members in groups.items():
        # Check for override winners first
        overrides = [e for e in members if e.group_override]
        if overrides:
            # First override wins
            selected_ids.add(overrides[0].id)
            continue

        # Weighted random selection; clamp weight to minimum 1
        weights = [max(1, e.group_weight) for e in members]
        winner = rng.choices(members, weights=weights, k=1)[0]
        selected_ids.add(winner.id)

    # Build result preserving input order
    result: list[LorebookEntry] = []
    for entry in entries:
        if entry.id not in grouped_ids:
            # Ungrouped — always pass
            result.append(entry)
        elif entry.id in selected_ids:
            result.append(entry)

    return result
