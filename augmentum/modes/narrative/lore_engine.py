"""Lorebook engine — parses and triggers World Info entries.

Supports SillyTavern World Info JSON format with:
- Keyword matching (literal and regex)
- Timed effects (sticky, cooldown, delay turns)
- Priority-based budget allocation
- Recursive scanning with token budget
"""

from __future__ import annotations

import random
import re

from augmentum.modes.narrative.world_info_buffer import WorldInfoBuffer
from augmentum.modes.narrative.world_info_groups import filter_by_groups
from augmentum.state.narrative_state import (
    LorebookEntry,
    LorebookPosition,
    SelectiveLogic,
    _new_id,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# AI-authored entries use keywords for search relevance ranking only,
# not for automatic injection. The model fetches them via lorebook.check.
_TOOL_FETCH_ONLY_SOURCES: frozenset[str] = frozenset(
    {"narrative_established", "llm_authored"}
)


def _try_regex(pattern_str: str, text: str) -> bool:
    """Try to match a /pattern/flags style regex."""
    parts = pattern_str.split("/")
    if len(parts) < 3:
        return False

    pattern = "/".join(parts[1:-1])
    flags_str = parts[-1]

    flags = 0
    if "i" in flags_str:
        flags |= re.IGNORECASE
    if "m" in flags_str:
        flags |= re.MULTILINE

    try:
        return bool(re.search(pattern, text, flags))
    except re.error:
        return False


def match_keywords(
    keywords: list[str],
    text: str,
    *,
    case_sensitive: bool = False,
    whole_words: bool = False,
) -> bool:
    """Check if ANY keyword matches the text.

    Each keyword is tried as:
    1. Regex if it looks like /pattern/flags
    2. Whole-word match if whole_words=True and keyword is single word
    3. Substring match otherwise
    """
    if not keywords or not text:
        return False

    check_text = text if case_sensitive else text.lower()

    for keyword in keywords:
        check_keyword = keyword if case_sensitive else keyword.lower()

        # Regex pattern
        if check_keyword.startswith("/") and "/" in check_keyword[1:]:
            if _try_regex(check_keyword, check_text):
                return True
        elif whole_words and " " not in check_keyword:
            # Whole-word match for single-word keywords
            escaped = re.escape(check_keyword)
            if re.search(rf"(?:^|\W){escaped}(?:$|\W)", check_text):
                return True
        else:
            # Substring match (also used for multi-word keys in whole-word mode)
            if check_keyword in check_text:
                return True

    return False


def check_secondary(
    secondary_keywords: list[str],
    text: str,
    logic: SelectiveLogic,
    *,
    case_sensitive: bool = False,
    whole_words: bool = False,
) -> bool:
    """Evaluate secondary keyword logic after primary match.

    AND_ANY (0): any secondary must match
    NOT_ALL (1): NOT all secondaries match (at least one missing)
    NOT_ANY (2): no secondary matches at all
    AND_ALL (3): all secondaries must match

    Empty secondary list = always True.
    """
    if not secondary_keywords:
        return True

    matches = [
        match_keywords([kw], text, case_sensitive=case_sensitive, whole_words=whole_words)
        for kw in secondary_keywords
    ]

    if logic == SelectiveLogic.AND_ANY:
        return any(matches)
    if logic == SelectiveLogic.NOT_ALL:
        return not all(matches)
    if logic == SelectiveLogic.NOT_ANY:
        return not any(matches)
    if logic == SelectiveLogic.AND_ALL:
        return all(matches)
    return True


class LoreEngine:
    """Manages lorebook entries and triggers them based on conversation content."""

    def __init__(self) -> None:
        self._entries: dict[str, LorebookEntry] = {}
        # Track timed effects: entry_id → remaining turns
        self._sticky_counters: dict[str, int] = {}
        self._cooldown_counters: dict[str, int] = {}
        # Track how many advance_turn() calls since engine init (for delay_turns)
        self._delay_counters: dict[str, int] = {}
        self._turn_count: int = 0
        # Canon-core tier (stable-prefix lore placement, 2026-07-15 spec):
        # membership survives restarts via to_state_dict. hit_log holds the
        # recent trigger turns per entry (promotion window evidence).
        self._core_members: dict[str, int] = {}   # id -> turn joined
        self._core_hit_log: dict[str, list[int]] = {}

    @property
    def entries(self) -> dict[str, LorebookEntry]:
        return dict(self._entries)

    def load_from_character_book(self, character_book: dict) -> list[LorebookEntry]:
        """Parse a Character Card V2 character_book into lorebook entries."""
        entries_data = character_book.get("entries", {})
        loaded = []

        # Support both dict-of-dicts (V2 spec) and list-of-dicts (common variant)
        items: list[dict] = []
        if isinstance(entries_data, dict):
            items = [v for v in entries_data.values() if isinstance(v, dict)]
        elif isinstance(entries_data, list):
            items = [v for v in entries_data if isinstance(v, dict)]

        for entry_data in items:
            entry = self._parse_entry(entry_data)
            if entry:
                self._entries[entry.id] = entry
                loaded.append(entry)

        log.info("lorebook_loaded", count=len(loaded))
        return loaded

    def load_from_world_info_json(self, world_info: list[dict]) -> list[LorebookEntry]:
        """Parse SillyTavern World Info JSON export."""
        loaded = []
        for entry_data in world_info:
            entry = self._parse_entry(entry_data)
            if entry:
                self._entries[entry.id] = entry
                loaded.append(entry)

        log.info("world_info_loaded", count=len(loaded))
        return loaded

    def add_entry(self, entry: LorebookEntry) -> None:
        """Add a single lorebook entry."""
        self._entries[entry.id] = entry

    def remove_entry(self, entry_id: str) -> None:
        """Remove a lorebook entry."""
        self._entries.pop(entry_id, None)

    def scan_and_trigger(
        self,
        messages: list[str],
        scan_depth: int = 5,
        message_index: int = 0,
        *,
        recursive: bool = False,
        max_recursion: int = 5,
        token_budget: int = 0,
        min_activations: int = 0,
        char_description: str = "",
        char_personality: str = "",
        persona_description: str = "",
        scenario: str = "",
        creator_notes: str = "",
    ) -> list[LorebookEntry]:
        """Scan recent messages for keyword matches and return triggered entries.

        Args:
            messages: Recent messages (newest first).
            scan_depth: How many messages to scan.
            message_index: Current message index for timed effects.
            recursive: If True, content of activated entries is scanned for more matches.
            max_recursion: Maximum recursion passes (including the first).
            token_budget: Maximum estimated tokens for all activated content (0 = unlimited).
            min_activations: Minimum entries to activate; widens scan depth if not met.
            char_description: Character description for per-entry match flags.
            char_personality: Character personality for per-entry match flags.
            persona_description: User persona for per-entry match flags.
            scenario: Scenario text for per-entry match flags.
            creator_notes: Creator notes for per-entry match flags.
        """
        # Build WorldInfoBuffer from inputs
        buf = WorldInfoBuffer(
            chat_messages=list(messages),
            persona_description=persona_description,
            char_description=char_description,
            char_personality=char_personality,
            scenario=scenario,
            creator_notes=creator_notes,
        )

        activated_ids: set[str] = set()
        activated_entries: list[LorebookEntry] = []
        sticky_activated: set[str] = set()  # entries triggered via sticky (don't reset counter)
        token_used = 0

        # Sort entries by priority (lower = higher priority) for deterministic ordering
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.priority)

        for pass_num in range(max_recursion):
            new_activations: list[LorebookEntry] = []

            for entry in sorted_entries:
                if entry.id in activated_ids:
                    continue
                if not entry.enabled:
                    continue

                # Cooldown check
                if entry.id in self._cooldown_counters and self._cooldown_counters[entry.id] > 0:
                    continue

                # Delay turns check — suppressed until enough turns have passed
                if entry.delay_turns > 0 and self._turn_count < entry.delay_turns:
                    continue

                # Exclude from recursive passes
                if entry.exclude_recursion and pass_num > 0:
                    continue

                # Constant entries always activate
                if entry.constant:
                    new_activations.append(entry)
                    continue

                # Sticky (still active from previous trigger)
                if entry.id in self._sticky_counters and self._sticky_counters[entry.id] > 0:
                    new_activations.append(entry)
                    sticky_activated.add(entry.id)
                    continue

                # Build per-entry scan text using entry's own flags
                scan_text = buf.get_scan_text(
                    entry.scan_depth if entry.scan_depth else scan_depth,
                    include_persona=entry.match_persona,
                    include_char_description=entry.match_char_description,
                    include_char_personality=entry.match_char_personality,
                    include_scenario=entry.match_scenario,
                    include_creator_notes=entry.match_creator_notes,
                    include_recursion=(pass_num > 0),
                )

                # Primary keyword check
                if not match_keywords(
                    entry.keywords,
                    scan_text,
                    case_sensitive=entry.case_sensitive,
                    whole_words=bool(entry.match_whole_words),
                ):
                    continue

                # Secondary keyword check (selective mode)
                if entry.selective and entry.secondary_keywords:
                    if not check_secondary(
                        entry.secondary_keywords,
                        scan_text,
                        entry.selective_logic,
                        case_sensitive=entry.case_sensitive,
                        whole_words=bool(entry.match_whole_words),
                    ):
                        continue

                new_activations.append(entry)

            # Filter through inclusion groups
            new_activations = filter_by_groups(new_activations)

            # Probability roll
            after_prob: list[LorebookEntry] = []
            for entry in new_activations:
                if entry.use_probability and entry.probability < 100:
                    if entry.probability <= 0:
                        continue
                    if random.random() * 100 > entry.probability:  # noqa: S311
                        continue
                after_prob.append(entry)
            new_activations = after_prob

            # Token budget filtering
            entries_to_add: list[LorebookEntry] = []
            for entry in new_activations:
                entry_tokens = len(entry.content) // 4
                if token_budget > 0 and not entry.ignore_budget:
                    if token_used + entry_tokens > token_budget:
                        continue
                token_used += entry_tokens
                entries_to_add.append(entry)

            # Update timed effects and record activations
            for entry in entries_to_add:
                activated_ids.add(entry.id)
                activated_entries.append(entry)
                entry.trigger_count += 1
                entry.last_triggered_at = message_index
                if entry.sticky_turns > 0 and entry.id not in sticky_activated:
                    self._sticky_counters[entry.id] = entry.sticky_turns
                elif entry.cooldown_turns > 0 and entry.id not in sticky_activated:
                    # Non-sticky entries start cooldown immediately after triggering
                    self._cooldown_counters[entry.id] = entry.cooldown_turns

            # Add non-prevent_recursion content to recursion buffer
            if recursive and entries_to_add:
                for entry in entries_to_add:
                    if not entry.prevent_recursion:
                        buf.add_to_recursion_buffer(entry.content)

            # Check termination conditions
            if not entries_to_add:
                # No new entries this pass
                if min_activations > 0 and len(activated_entries) < min_activations:
                    buf.advance_scan()
                    continue
                break

            if not recursive:
                break

            # If recursive but we have enough, check min_activations
            if min_activations > 0 and len(activated_entries) < min_activations:
                buf.advance_scan()
                continue

        # Sort by priority (lower = higher priority)
        activated_entries.sort(key=lambda e: e.priority)

        log.debug(
            "lorebook_scan",
            triggered=len(activated_entries),
            total=len(self._entries),
            passes=min(pass_num + 1, max_recursion) if self._entries else 0,
        )
        return activated_entries

    def advance_turn(self) -> None:
        """Advance timed effects by one turn."""
        self._turn_count += 1

        # Decrement sticky counters
        expired_sticky = []
        for entry_id, remaining in self._sticky_counters.items():
            self._sticky_counters[entry_id] = remaining - 1
            if remaining - 1 <= 0:
                expired_sticky.append(entry_id)

        # Start cooldown for expired sticky entries
        new_cooldowns: set[str] = set()
        for entry_id in expired_sticky:
            del self._sticky_counters[entry_id]
            entry = self._entries.get(entry_id)
            if entry and entry.cooldown_turns > 0:
                self._cooldown_counters[entry_id] = entry.cooldown_turns
                new_cooldowns.add(entry_id)

        # Decrement cooldown counters (skip newly created ones)
        expired_cooldown = []
        for entry_id, remaining in self._cooldown_counters.items():
            if entry_id in new_cooldowns:
                continue
            self._cooldown_counters[entry_id] = remaining - 1
            if remaining - 1 <= 0:
                expired_cooldown.append(entry_id)

        for entry_id in expired_cooldown:
            del self._cooldown_counters[entry_id]

    def _matches_keywords(self, entry: LorebookEntry, text: str) -> bool:
        """Check if text matches the entry's keywords."""
        return match_keywords(
            entry.keywords,
            text,
            case_sensitive=entry.case_sensitive,
            whole_words=bool(getattr(entry, "match_whole_words", None)),
        )

    def _try_regex_match(self, pattern_str: str, text: str) -> bool:
        """Try to match a /pattern/flags style regex (kept for back-compat)."""
        return _try_regex(pattern_str, text)

    def _parse_entry(self, data: dict) -> LorebookEntry | None:
        """Parse a single lorebook entry from JSON data.

        Handles full SillyTavern World Info JSON fields with camelCase aliases.
        """
        content = data.get("content", "")
        if not content:
            return None

        # --- Keywords (primary) ---
        keywords_raw = data.get("key", data.get("keys", data.get("keyword", [])))
        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords = []

        # --- Secondary keywords ---
        sec_raw = data.get("keysecondary", data.get("secondary_keywords", []))
        if isinstance(sec_raw, str):
            secondary_keywords = [k.strip() for k in sec_raw.split(",") if k.strip()]
        elif isinstance(sec_raw, list):
            secondary_keywords = [str(k).strip() for k in sec_raw if str(k).strip()]
        else:
            secondary_keywords = []

        # --- Selective ---
        selective = data.get("selective", True)
        if selective in (False, 0, "false"):
            selective = False
        else:
            selective = True

        # --- Selective logic (int → enum) ---
        sel_logic_raw = data.get("selectiveLogic", data.get("selective_logic", 0))
        try:
            selective_logic = SelectiveLogic(int(sel_logic_raw))
        except (ValueError, TypeError):
            selective_logic = SelectiveLogic.AND_ANY

        # --- Position ---
        position_raw = data.get("position", 0)
        position_map = {
            0: LorebookPosition.BEFORE_CHAR,
            1: LorebookPosition.AFTER_CHAR,
            4: LorebookPosition.AT_DEPTH,
            5: LorebookPosition.EM_TOP,
            6: LorebookPosition.EM_BOTTOM,
            7: LorebookPosition.OUTLET,
            "before_char": LorebookPosition.BEFORE_CHAR,
            "after_char": LorebookPosition.AFTER_CHAR,
            "at_depth": LorebookPosition.AT_DEPTH,
            "em_top": LorebookPosition.EM_TOP,
            "em_bottom": LorebookPosition.EM_BOTTOM,
            "outlet": LorebookPosition.OUTLET,
        }
        position = position_map.get(position_raw, LorebookPosition.BEFORE_CHAR)

        # --- Injection depth + role (only meaningful for position=at_depth) ---
        # Accept both our keys and ST's canonical "depth" / "role".
        injection_depth_raw = data.get(
            "injection_depth",
            data.get("injectionDepth", data.get("depth", 4)),
        )
        try:
            injection_depth = int(injection_depth_raw)
        except (ValueError, TypeError):
            injection_depth = 4

        # Role: accept strings ("system"/"user"/"assistant") or ST's int enum
        # (0=system, 1=user, 2=assistant). Default "system" matches ST's Author's
        # Note convention.
        role_raw = data.get("injection_role", data.get("injectionRole", data.get("role", "system")))
        role_int_map = {0: "system", 1: "user", 2: "assistant"}
        if isinstance(role_raw, int):
            injection_role = role_int_map.get(role_raw, "system")
        elif isinstance(role_raw, str):
            rn = role_raw.strip().lower()
            injection_role = rn if rn in ("system", "user", "assistant") else "system"
        else:
            injection_role = "system"

        # --- Enabled (handle disable/disabled negation) ---
        if "disable" in data or "disabled" in data:
            disable_val = data.get("disable", data.get("disabled", False))
            enabled = disable_val in (False, 0, "false")
        else:
            enabled = data.get("enabled", True) not in (False, 0, "false")

        # --- Bool helpers ---
        def _bool(val: object, default: bool = False) -> bool:
            if val is None:
                return default
            return val not in (False, 0, "false")

        def _opt_bool(val: object) -> bool | None:
            if val is None:
                return None
            return val not in (False, 0, "false")

        return LorebookEntry(
            id=str(data.get("uid", data.get("id", _new_id()))),
            keywords=keywords,
            secondary_keywords=secondary_keywords,
            selective=selective,
            selective_logic=selective_logic,
            content=content,
            comment=data.get("comment", data.get("memo", "")),
            constant=_bool(data.get("constant")),
            priority=data.get("order", data.get("priority", 100)),
            source=data.get("source", "character_book"),
            enabled=enabled,
            position=position,
            scan_depth=data.get("depth", data.get("scan_depth", 5)),
            case_sensitive=_bool(data.get("caseSensitive", data.get("case_sensitive"))),
            probability=data.get("probability", 100),
            use_probability=_bool(data.get("useProbability", data.get("use_probability")), default=True),
            group=data.get("group", ""),
            group_override=_bool(data.get("groupOverride", data.get("group_override"))),
            group_weight=data.get("groupWeight", data.get("group_weight", 100)),
            sticky_turns=data.get("sticky", data.get("sticky_turns", 0)),
            cooldown_turns=data.get("cooldown", data.get("cooldown_turns", 0)),
            delay_turns=data.get("delay", data.get("delay_turns", 0)),
            exclude_recursion=_bool(data.get("excludeRecursion", data.get("exclude_recursion"))),
            prevent_recursion=_bool(data.get("preventRecursion", data.get("prevent_recursion"))),
            delay_until_recursion=data.get("delayUntilRecursion", data.get("delay_until_recursion", 0)),
            match_persona=_bool(data.get("matchPersonaDescription", data.get("match_persona"))),
            match_char_description=_bool(data.get("matchCharacterDescription", data.get("match_char_description"))),
            match_char_personality=_bool(data.get("matchCharacterPersonality", data.get("match_char_personality"))),
            match_scenario=_bool(data.get("matchScenario", data.get("match_scenario"))),
            match_creator_notes=_bool(data.get("matchCreatorNotes", data.get("match_creator_notes"))),
            ignore_budget=_bool(data.get("ignoreBudget", data.get("ignore_budget"))),
            match_whole_words=_opt_bool(data.get("matchWholeWords", data.get("match_whole_words"))),
            use_group_scoring=_opt_bool(data.get("useGroupScoring", data.get("use_group_scoring"))),
            outlet_name=data.get("outletName", data.get("outlet_name", "")),
            injection_depth=injection_depth,
            injection_role=injection_role,
        )

    def set_entries(self, entries: list[LorebookEntry]) -> None:
        """Set entries directly (used when loading from DB)."""
        self._entries = {e.id: e for e in entries}

    def replace_entries_preserving_state(
        self, entries_data: list[dict],
    ) -> list[LorebookEntry]:
        """Replace all entries from a list of raw dicts, preserving timed-effect
        counters for entries that still exist.

        Called per-turn by the narrative handler so the live UI lorebook drives
        the engine — instead of only whatever was embedded in the card at init.
        Entry identity is by ``id``/``uid`` when present, falling back to a
        deterministic hash of keywords+content (so an edit to a keyword is a
        new entry and its sticky timer resets, which matches user expectation).

        Returns the parsed entries.
        """
        # Parse fresh
        parsed: list[LorebookEntry] = []
        seen_ids: set[str] = set()
        for raw in entries_data or []:
            if not isinstance(raw, dict):
                continue
            # Ensure a stable id: if the UI didn't give one, derive a short
            # hash so state survives re-serialisation. The UI does give ids
            # now that sessions persist them, but older sessions may not.
            if not raw.get("id") and not raw.get("uid"):
                keys_part = ",".join(str(k) for k in raw.get("keys", []))
                content_part = (raw.get("content", "") or "")[:200]
                import hashlib
                h = hashlib.sha256(f"{keys_part}|{content_part}".encode()).hexdigest()[:16]
                raw = {**raw, "id": f"ui_{h}"}
            entry = self._parse_entry(raw)
            if entry:
                parsed.append(entry)
                seen_ids.add(entry.id)

        # Build the new entries dict
        new_entries: dict[str, LorebookEntry] = {e.id: e for e in parsed}

        # Preserve existing timed-effect counters for entries that still exist
        self._sticky_counters = {
            eid: n for eid, n in self._sticky_counters.items() if eid in seen_ids
        }
        self._cooldown_counters = {
            eid: n for eid, n in self._cooldown_counters.items() if eid in seen_ids
        }
        self._delay_counters = {
            eid: n for eid, n in self._delay_counters.items() if eid in seen_ids
        }

        self._entries = new_entries
        return parsed

    # -- Canon-core tier (stable-prefix placement) --------------------------

    CORE_PROMOTE_HITS = 2       # >= hits ...
    CORE_PROMOTE_WINDOW = 5     # ... within this many turns -> promote
    CORE_DEMOTE_IDLE = 20       # no trigger for this many turns -> demote

    def update_core_membership(self, triggered_ids: list[str], turn: int) -> set[str]:
        """Hysteresis: entries that keep triggering PROMOTE into the canon
        core (rendered in the stable prefix so the KV checkpoint covers
        them); long-idle members DEMOTE. Membership changes are the only
        head mutations, and the prewarm bridge absorbs them off-turn.

        Returns the current member id set (constants are the caller's
        responsibility — they're core by definition, no hysteresis)."""
        for eid in triggered_ids:
            hits = self._core_hit_log.setdefault(eid, [])
            if not hits or hits[-1] != turn:
                hits.append(turn)
            del hits[:-8]
            recent = [t for t in hits if turn - t < self.CORE_PROMOTE_WINDOW]
            if len(recent) >= self.CORE_PROMOTE_HITS and eid not in self._core_members:
                self._core_members[eid] = turn
                log.info("lore_core_promoted", entry=eid, turn=turn)
        for eid in list(self._core_members):
            hits = self._core_hit_log.get(eid) or []
            last = hits[-1] if hits else self._core_members[eid]
            if turn - last >= self.CORE_DEMOTE_IDLE:
                del self._core_members[eid]
                log.info("lore_core_demoted", entry=eid, turn=turn)
        # Drop members whose entries vanished (book edited mid-session)
        for eid in list(self._core_members):
            if eid not in self._entries:
                del self._core_members[eid]
        return set(self._core_members)

    def to_state_dict(self) -> dict:
        """Snapshot runtime counters for persistence across server restarts."""
        return {
            "turn_count": self._turn_count,
            "sticky": dict(self._sticky_counters),
            "cooldown": dict(self._cooldown_counters),
            "delay": dict(self._delay_counters),
            "core_members": dict(self._core_members),
            "core_hit_log": {k: list(v) for k, v in self._core_hit_log.items()},
        }

    def load_state_dict(self, data: dict) -> None:
        """Restore runtime counters from a prior snapshot. Unknown entry ids
        are kept — if the entry re-appears later, its timer resumes; if not,
        the stale counter is harmless (entry lookups miss)."""
        if not isinstance(data, dict):
            return
        self._turn_count = int(data.get("turn_count", 0) or 0)
        self._sticky_counters = {
            str(k): int(v) for k, v in (data.get("sticky") or {}).items() if isinstance(v, (int, float))
        }
        self._cooldown_counters = {
            str(k): int(v) for k, v in (data.get("cooldown") or {}).items() if isinstance(v, (int, float))
        }
        self._delay_counters = {
            str(k): int(v) for k, v in (data.get("delay") or {}).items() if isinstance(v, (int, float))
        }
        self._core_members = {
            str(k): int(v) for k, v in (data.get("core_members") or {}).items()
            if isinstance(v, int | float)
        }
        self._core_hit_log = {
            str(k): [int(x) for x in v if isinstance(x, int | float)]
            for k, v in (data.get("core_hit_log") or {}).items()
            if isinstance(v, list)
        }
