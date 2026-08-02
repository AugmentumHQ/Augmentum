"""Tests for World Info V2 dataclass extensions."""

from __future__ import annotations

from augmentum.state.narrative_state import (
    LorebookEntry,
    LorebookPosition,
    SelectiveLogic,
)


class TestSelectiveLogicEnum:
    def test_values(self) -> None:
        assert SelectiveLogic.AND_ANY == 0
        assert SelectiveLogic.NOT_ALL == 1
        assert SelectiveLogic.NOT_ANY == 2
        assert SelectiveLogic.AND_ALL == 3

    def test_all_members(self) -> None:
        assert len(SelectiveLogic) == 4

    def test_int_coercion(self) -> None:
        assert int(SelectiveLogic.AND_ANY) == 0
        assert int(SelectiveLogic.AND_ALL) == 3


class TestLorebookPositionExtensions:
    def test_original_values_exist(self) -> None:
        assert LorebookPosition.BEFORE_CHAR == "before_char"
        assert LorebookPosition.AFTER_CHAR == "after_char"
        assert LorebookPosition.AT_DEPTH == "at_depth"

    def test_new_values_exist(self) -> None:
        assert LorebookPosition.EM_TOP == "em_top"
        assert LorebookPosition.EM_BOTTOM == "em_bottom"
        assert LorebookPosition.OUTLET == "outlet"

    def test_total_members(self) -> None:
        assert len(LorebookPosition) == 6


class TestLorebookEntryDefaults:
    """All new fields must have defaults so existing data loads."""

    def test_default_construction(self) -> None:
        entry = LorebookEntry()
        assert entry is not None

    def test_existing_fields_unchanged(self) -> None:
        entry = LorebookEntry()
        assert entry.keywords == []
        assert entry.content == ""
        assert entry.priority == 100
        assert entry.enabled is True
        assert entry.constant is False
        assert entry.position == LorebookPosition.BEFORE_CHAR
        assert entry.scan_depth == 5
        assert entry.case_sensitive is False
        assert entry.sticky_turns == 0
        assert entry.cooldown_turns == 0
        assert entry.trigger_count == 0

    def test_secondary_keywords_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.secondary_keywords == []
        assert entry.selective is True
        assert entry.selective_logic == SelectiveLogic.AND_ANY

    def test_inclusion_group_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.group == ""
        assert entry.group_override is False
        assert entry.group_weight == 100

    def test_probability_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.probability == 100
        assert entry.use_probability is True

    def test_budget_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.ignore_budget is False

    def test_matching_option_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.match_whole_words is None
        assert entry.use_group_scoring is None

    def test_recursion_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.exclude_recursion is False
        assert entry.prevent_recursion is False
        assert entry.delay_until_recursion == 0

    def test_scanning_scope_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.match_persona is False
        assert entry.match_char_description is False
        assert entry.match_char_personality is False
        assert entry.match_scenario is False
        assert entry.match_creator_notes is False

    def test_timed_effect_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.delay_turns == 0

    def test_outlet_defaults(self) -> None:
        entry = LorebookEntry()
        assert entry.outlet_name == ""

    def test_comment_default(self) -> None:
        entry = LorebookEntry()
        assert entry.comment == ""


class TestLorebookEntryCustomValues:
    def test_secondary_keywords_set(self) -> None:
        entry = LorebookEntry(
            secondary_keywords=["dragon", "fire"],
            selective=True,
            selective_logic=SelectiveLogic.AND_ALL,
        )
        assert entry.secondary_keywords == ["dragon", "fire"]
        assert entry.selective_logic == SelectiveLogic.AND_ALL

    def test_group_fields(self) -> None:
        entry = LorebookEntry(group="lore_set_a", group_override=True, group_weight=50)
        assert entry.group == "lore_set_a"
        assert entry.group_override is True
        assert entry.group_weight == 50

    def test_recursion_fields(self) -> None:
        entry = LorebookEntry(
            exclude_recursion=True,
            prevent_recursion=True,
            delay_until_recursion=3,
        )
        assert entry.exclude_recursion is True
        assert entry.prevent_recursion is True
        assert entry.delay_until_recursion == 3

    def test_scanning_flags_set(self) -> None:
        entry = LorebookEntry(
            match_persona=True,
            match_char_description=True,
            match_char_personality=True,
            match_scenario=True,
            match_creator_notes=True,
        )
        assert entry.match_persona is True
        assert entry.match_char_description is True
        assert entry.match_char_personality is True
        assert entry.match_scenario is True
        assert entry.match_creator_notes is True

    def test_outlet_position(self) -> None:
        entry = LorebookEntry(
            position=LorebookPosition.OUTLET,
            outlet_name="world_lore",
        )
        assert entry.position == LorebookPosition.OUTLET
        assert entry.outlet_name == "world_lore"

    def test_probability_disabled(self) -> None:
        entry = LorebookEntry(probability=50, use_probability=False)
        assert entry.probability == 50
        assert entry.use_probability is False

    def test_matching_options_explicit(self) -> None:
        entry = LorebookEntry(match_whole_words=True, use_group_scoring=False)
        assert entry.match_whole_words is True
        assert entry.use_group_scoring is False


# ---------------------------------------------------------------------------
# WorldInfoBuffer
# ---------------------------------------------------------------------------

from augmentum.modes.narrative.world_info_buffer import WorldInfoBuffer


class TestWorldInfoBuffer:
    def _make(self, **kwargs) -> WorldInfoBuffer:
        return WorldInfoBuffer(**kwargs)

    def test_chat_messages_at_depth(self) -> None:
        buf = self._make(chat_messages=["m1", "m2", "m3"])
        text = buf.get_scan_text(2)
        assert "m1" in text
        assert "m2" in text
        assert "m3" not in text

    def test_full_depth_returns_all(self) -> None:
        buf = self._make(chat_messages=["a", "b", "c"])
        text = buf.get_scan_text(10)
        assert "a" in text
        assert "b" in text
        assert "c" in text

    def test_persona_included_when_flagged(self) -> None:
        buf = self._make(persona_description="I am the user persona")
        text = buf.get_scan_text(5, include_persona=True)
        assert "I am the user persona" in text

    def test_persona_excluded_by_default(self) -> None:
        buf = self._make(persona_description="hidden persona")
        text = buf.get_scan_text(5)
        assert "hidden persona" not in text

    def test_char_description_included(self) -> None:
        buf = self._make(char_description="tall and mysterious")
        text = buf.get_scan_text(5, include_char_description=True)
        assert "tall and mysterious" in text

    def test_recursion_buffer_appended(self) -> None:
        buf = self._make(chat_messages=["hello"])
        buf.add_to_recursion_buffer("dragon lore activated")
        text = buf.get_scan_text(5, include_recursion=True)
        assert "dragon lore activated" in text

    def test_boundary_separator(self) -> None:
        buf = self._make(chat_messages=["first", "second"])
        text = buf.get_scan_text(5)
        assert "\x01" in text
        parts = text.split("\x01")
        assert parts[0] == "first"
        assert parts[1] == "second"

    def test_advance_scan_increases_depth(self) -> None:
        buf = self._make(chat_messages=["a", "b", "c", "d"])
        # depth=2 → only a, b
        text1 = buf.get_scan_text(2)
        assert "c" not in text1
        # advance once → effective depth 3
        buf.advance_scan()
        text2 = buf.get_scan_text(2)
        assert "c" in text2

    def test_reset_skew(self) -> None:
        buf = self._make(chat_messages=["a", "b", "c"])
        buf.advance_scan()
        buf.advance_scan()
        buf.reset_skew()
        text = buf.get_scan_text(2)
        assert "c" not in text

    def test_empty_fields_not_included(self) -> None:
        buf = self._make(
            chat_messages=["msg"],
            persona_description="",
            char_description="",
            scenario="",
        )
        text = buf.get_scan_text(
            5,
            include_persona=True,
            include_char_description=True,
            include_scenario=True,
        )
        # Only the message, no extra boundaries from empty fields
        assert text == "msg"


# ---------------------------------------------------------------------------
# Keyword Matching
# ---------------------------------------------------------------------------

from augmentum.modes.narrative.lore_engine import check_secondary, match_keywords


class TestKeywordMatching:
    def test_plain_substring(self) -> None:
        assert match_keywords(["dragon"], "The dragon roars") is True

    def test_plain_no_match(self) -> None:
        assert match_keywords(["unicorn"], "The dragon roars") is False

    def test_case_insensitive(self) -> None:
        assert match_keywords(["DRAGON"], "the dragon roars") is True

    def test_case_sensitive(self) -> None:
        assert match_keywords(["DRAGON"], "the dragon roars", case_sensitive=True) is False
        assert match_keywords(["DRAGON"], "the DRAGON roars", case_sensitive=True) is True

    def test_regex_match(self) -> None:
        assert match_keywords(["/drag.n/i"], "The Dragon roars") is True
        assert match_keywords(["/^hello/"], "hello world") is True
        assert match_keywords(["/^hello/"], "say hello") is False

    def test_whole_word_single(self) -> None:
        # "art" matches "the art of war" but not "the artist paints"
        assert match_keywords(["art"], "the art of war", whole_words=True) is True
        assert match_keywords(["art"], "the artist paints", whole_words=True) is False

    def test_whole_word_multi_word_key(self) -> None:
        # "art of" matches via substring even in whole-word mode
        assert match_keywords(["art of"], "the art of war", whole_words=True) is True

    def test_any_keyword_triggers(self) -> None:
        # Multiple keywords, any match = True
        assert match_keywords(["unicorn", "dragon"], "The dragon roars") is True

    def test_empty_keywords_no_match(self) -> None:
        assert match_keywords([], "The dragon roars") is False

    def test_empty_text_no_match(self) -> None:
        assert match_keywords(["dragon"], "") is False


class TestSecondaryKeywordLogic:
    def test_and_any_matches(self) -> None:
        assert check_secondary(["fire", "ice"], "fire burns", SelectiveLogic.AND_ANY) is True

    def test_and_any_fails(self) -> None:
        assert check_secondary(["fire", "ice"], "water flows", SelectiveLogic.AND_ANY) is False

    def test_not_all_when_partial(self) -> None:
        # Only "fire" matches, not all → True
        assert check_secondary(["fire", "ice"], "fire burns", SelectiveLogic.NOT_ALL) is True

    def test_not_all_when_all_match(self) -> None:
        # Both match → False
        assert check_secondary(["fire", "ice"], "fire and ice", SelectiveLogic.NOT_ALL) is False

    def test_not_any_when_none_match(self) -> None:
        assert check_secondary(["fire", "ice"], "water flows", SelectiveLogic.NOT_ANY) is True

    def test_not_any_when_some_match(self) -> None:
        assert check_secondary(["fire", "ice"], "fire burns", SelectiveLogic.NOT_ANY) is False

    def test_and_all_when_all_match(self) -> None:
        assert check_secondary(["fire", "ice"], "fire and ice", SelectiveLogic.AND_ALL) is True

    def test_and_all_when_partial(self) -> None:
        assert check_secondary(["fire", "ice"], "fire burns", SelectiveLogic.AND_ALL) is False

    def test_empty_secondary_always_true(self) -> None:
        assert check_secondary([], "anything", SelectiveLogic.AND_ANY) is True
        assert check_secondary([], "anything", SelectiveLogic.NOT_ALL) is True
        assert check_secondary([], "anything", SelectiveLogic.NOT_ANY) is True
        assert check_secondary([], "anything", SelectiveLogic.AND_ALL) is True


# ---------------------------------------------------------------------------
# Inclusion Groups
# ---------------------------------------------------------------------------

from augmentum.modes.narrative.world_info_groups import filter_by_groups


def _entry(
    id: str,
    group: str = "",
    override: bool = False,
    weight: int = 100,
) -> LorebookEntry:
    return LorebookEntry(id=id, group=group, group_override=override, group_weight=weight)


class TestInclusionGroups:
    def test_no_groups_passthrough(self) -> None:
        entries = [_entry("a"), _entry("b"), _entry("c")]
        result = filter_by_groups(entries)
        assert [e.id for e in result] == ["a", "b", "c"]

    def test_group_keeps_one(self) -> None:
        entries = [_entry("a", group="faction"), _entry("b", group="faction")]
        result = filter_by_groups(entries, seed=42)
        assert len(result) == 1
        assert result[0].id in ("a", "b")

    def test_override_wins(self) -> None:
        entries = [
            _entry("a", group="faction", weight=1),
            _entry("b", group="faction", override=True, weight=1),
            _entry("c", group="faction", weight=999),
        ]
        # Override always wins regardless of weight
        for _ in range(10):
            result = filter_by_groups(entries)
            assert len(result) == 1
            assert result[0].id == "b"

    def test_multiple_groups_independent(self) -> None:
        entries = [
            _entry("a1", group="faction"),
            _entry("a2", group="faction"),
            _entry("b1", group="race"),
            _entry("b2", group="race"),
        ]
        result = filter_by_groups(entries, seed=0)
        ids = {e.id for e in result}
        assert len(ids) == 2
        assert ids & {"a1", "a2"}  # one from faction
        assert ids & {"b1", "b2"}  # one from race

    def test_ungrouped_always_pass(self) -> None:
        entries = [
            _entry("free1"),
            _entry("a", group="faction"),
            _entry("b", group="faction"),
            _entry("free2"),
        ]
        result = filter_by_groups(entries, seed=7)
        ids = [e.id for e in result]
        assert "free1" in ids
        assert "free2" in ids
        assert len(ids) == 3  # 2 ungrouped + 1 grouped winner

    def test_comma_separated_groups(self) -> None:
        entries = [
            _entry("multi", group="faction, race"),
            _entry("f_only", group="faction"),
            _entry("r_only", group="race"),
        ]
        # 'multi' participates in both groups; if selected by either it stays
        result = filter_by_groups(entries, seed=42)
        ids = {e.id for e in result}
        # At least 1, at most 3 (one winner per group, multi could win both)
        assert 1 <= len(ids) <= 3

    def test_deterministic_with_seed(self) -> None:
        entries = [
            _entry("a", group="g", weight=50),
            _entry("b", group="g", weight=50),
            _entry("c", group="g", weight=50),
        ]
        results = [filter_by_groups(entries, seed=123)[0].id for _ in range(20)]
        assert len(set(results)) == 1  # same seed always same result

    def test_empty_list(self) -> None:
        assert filter_by_groups([]) == []

    def test_single_entry_in_group(self) -> None:
        entries = [_entry("solo", group="exclusive")]
        result = filter_by_groups(entries, seed=0)
        assert len(result) == 1
        assert result[0].id == "solo"

    def test_zero_weight_still_selectable(self) -> None:
        entries = [_entry("z", group="g", weight=0)]
        result = filter_by_groups(entries, seed=0)
        assert len(result) == 1
        assert result[0].id == "z"


# ---------------------------------------------------------------------------
# Recursive Scanning
# ---------------------------------------------------------------------------

from augmentum.modes.narrative.lore_engine import LoreEngine


class TestRecursiveScanning:
    def _engine(self) -> LoreEngine:
        return LoreEngine()

    def test_single_pass_no_recursion(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(id="a", keywords=["dragon"], content="A fire dragon."))
        result = engine.scan_and_trigger(["The dragon roars"], recursive=False)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_recursive_triggers_from_content(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(id="a", keywords=["dragon"], content="The beast breathes fire."))
        engine.add_entry(LorebookEntry(id="b", keywords=["fire"], content="Flames everywhere."))
        # Without recursion only "dragon" triggers
        result_flat = engine.scan_and_trigger(["A dragon appears"], recursive=False)
        assert {e.id for e in result_flat} == {"a"}

        # Reset trigger counts for fresh engine
        engine2 = self._engine()
        engine2.add_entry(LorebookEntry(id="a", keywords=["dragon"], content="The beast breathes fire."))
        engine2.add_entry(LorebookEntry(id="b", keywords=["fire"], content="Flames everywhere."))
        result_rec = engine2.scan_and_trigger(["A dragon appears"], recursive=True)
        assert {e.id for e in result_rec} == {"a", "b"}

    def test_max_recursion_stops(self) -> None:
        """Circular references terminate at max_recursion."""
        engine = self._engine()
        engine.add_entry(LorebookEntry(id="a", keywords=["alpha"], content="mentions beta"))
        engine.add_entry(LorebookEntry(id="b", keywords=["beta"], content="mentions alpha"))
        # Should not infinite-loop; both get activated within max_recursion
        result = engine.scan_and_trigger(
            ["alpha is here"], recursive=True, max_recursion=10,
        )
        assert {e.id for e in result} == {"a", "b"}

    def test_exclude_recursion_skipped(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(id="a", keywords=["dragon"], content="It breathes fire."))
        engine.add_entry(
            LorebookEntry(id="b", keywords=["fire"], content="Hot flames.", exclude_recursion=True),
        )
        result = engine.scan_and_trigger(["A dragon appears"], recursive=True)
        # Entry b has exclude_recursion so it should NOT be found in recursive pass
        assert {e.id for e in result} == {"a"}

    def test_prevent_recursion_content_not_scanned(self) -> None:
        engine = self._engine()
        engine.add_entry(
            LorebookEntry(
                id="a", keywords=["dragon"], content="fire breathing beast",
                prevent_recursion=True,
            ),
        )
        engine.add_entry(LorebookEntry(id="b", keywords=["fire"], content="Hot flames."))
        result = engine.scan_and_trigger(["A dragon appears"], recursive=True)
        # Entry a's content should NOT be added to recursion buffer
        assert {e.id for e in result} == {"a"}


class TestTokenBudget:
    def _engine(self) -> LoreEngine:
        return LoreEngine()

    def test_budget_limits_entries(self) -> None:
        engine = self._engine()
        # Each entry ~25 tokens (100 chars / 4)
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="x" * 100, priority=1,
        ))
        engine.add_entry(LorebookEntry(
            id="b", keywords=["hello"], content="y" * 100, priority=2,
        ))
        result = engine.scan_and_trigger(["hello"], token_budget=30)
        # Budget fits ~30 tokens, first entry uses 25, second would exceed
        assert len(result) == 1
        assert result[0].id == "a"

    def test_ignore_budget_bypasses(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="x" * 100, priority=1,
        ))
        engine.add_entry(LorebookEntry(
            id="b", keywords=["hello"], content="y" * 100, priority=2,
            ignore_budget=True,
        ))
        result = engine.scan_and_trigger(["hello"], token_budget=30)
        assert {e.id for e in result} == {"a", "b"}

    def test_no_budget_unlimited(self) -> None:
        engine = self._engine()
        for i in range(10):
            engine.add_entry(LorebookEntry(
                id=str(i), keywords=["hello"], content="x" * 200,
            ))
        result = engine.scan_and_trigger(["hello"], token_budget=0)
        assert len(result) == 10

    def test_probability_zero_never_triggers(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="content", probability=0,
        ))
        # Run multiple times to be sure
        for _ in range(20):
            result = engine.scan_and_trigger(["hello"])
            assert len(result) == 0

    def test_probability_100_always_triggers(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="content", probability=100,
        ))
        for _ in range(20):
            result = engine.scan_and_trigger(["hello"])
            assert len(result) == 1


class TestDelayTurns:
    def _engine(self) -> LoreEngine:
        return LoreEngine()

    def test_delay_suppresses_initial(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="delayed entry", delay_turns=2,
        ))
        # Turn 0 — should not trigger
        result = engine.scan_and_trigger(["hello"])
        assert len(result) == 0
        engine.advance_turn()
        # Turn 1 — still suppressed
        result = engine.scan_and_trigger(["hello"])
        assert len(result) == 0

    def test_delay_activates_after_threshold(self) -> None:
        engine = self._engine()
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="delayed entry", delay_turns=2,
        ))
        engine.advance_turn()
        engine.advance_turn()
        # Turn 2 — should now trigger
        result = engine.scan_and_trigger(["hello"])
        assert len(result) == 1
        assert result[0].id == "a"


class TestMinActivations:
    def _engine(self) -> LoreEngine:
        return LoreEngine()

    def test_min_activations_widens_depth(self) -> None:
        engine = self._engine()
        # Entry a matches "hello" in recent messages
        engine.add_entry(LorebookEntry(
            id="a", keywords=["hello"], content="greeting", scan_depth=2,
        ))
        # Entry b matches "ancient" which is only in message index 3 (depth 4)
        engine.add_entry(LorebookEntry(
            id="b", keywords=["ancient"], content="old lore", scan_depth=2,
        ))
        messages = ["hello world", "how are you", "fine thanks", "ancient ruins here"]
        # With depth 2, only "a" matches; min_activations=2 should widen to find "b"
        result = engine.scan_and_trigger(
            messages, scan_depth=2, min_activations=2, recursive=True, max_recursion=5,
        )
        assert {e.id for e in result} == {"a", "b"}


# ---------------------------------------------------------------------------
# SillyTavern World Info Import
# ---------------------------------------------------------------------------


class TestSTWorldInfoImport:
    def test_full_st_entry(self):
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        entries = engine.load_from_world_info_json([{
            "uid": 42,
            "key": ["dragon", "wyrm"],
            "keysecondary": ["fire", "scales"],
            "comment": "Dragon lore",
            "content": "Dragons are ancient creatures",
            "constant": False,
            "selective": True,
            "selectiveLogic": 0,
            "order": 50,
            "position": 4,  # AT_DEPTH
            "depth": 3,
            "disable": False,
            "probability": 80,
            "useProbability": True,
            "group": "creatures",
            "groupOverride": False,
            "groupWeight": 100,
            "sticky": 3,
            "cooldown": 2,
            "delay": 1,
            "excludeRecursion": False,
            "preventRecursion": False,
            "matchPersonaDescription": True,
            "matchCharacterDescription": True,
            "ignoreBudget": False,
            "matchWholeWords": True,
        }])
        assert len(entries) == 1
        e = entries[0]
        assert e.keywords == ["dragon", "wyrm"]
        assert e.secondary_keywords == ["fire", "scales"]
        assert e.comment == "Dragon lore"
        assert e.probability == 80
        assert e.group == "creatures"
        assert e.sticky_turns == 3
        assert e.cooldown_turns == 2
        assert e.delay_turns == 1
        assert e.match_persona is True
        assert e.match_char_description is True
        assert e.match_whole_words is True
        assert e.position.value == "at_depth"
        assert e.scan_depth == 3

    def test_disabled_entry(self):
        # disable: True → enabled: False
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        entries = engine.load_from_world_info_json([{
            "content": "test", "key": ["a"], "disable": True,
        }])
        assert entries[0].enabled is False

    def test_string_keywords_split(self):
        # key as comma-separated string
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        entries = engine.load_from_world_info_json([{
            "content": "test", "key": "dragon, wyrm",
        }])
        assert "dragon" in entries[0].keywords
        assert "wyrm" in entries[0].keywords

    def test_empty_content_skipped(self):
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        entries = engine.load_from_world_info_json([
            {"key": ["a"], "content": ""},
            {"key": ["b"], "content": "valid"},
        ])
        assert len(entries) == 1

    def test_character_book_v2(self):
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        entries = engine.load_from_character_book({
            "entries": [
                {"keys": ["sword"], "content": "A magic sword", "enabled": True},
                {"keys": ["shield"], "content": "A strong shield", "enabled": True},
            ]
        })
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Full Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestFullScanPipeline:
    def test_constant_entry_always_triggers(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(content="Always here", constant=True))
        triggered = engine.scan_and_trigger(["random text"])
        assert len(triggered) == 1

    def test_full_pipeline_with_groups_and_budget(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(id="e1", keywords=["war"], content="War is brutal", group="theme"))
        engine.add_entry(LorebookEntry(id="e2", keywords=["war"], content="War is glory", group="theme"))
        engine.add_entry(LorebookEntry(id="e3", keywords=["war"], content="Swords clash"))
        triggered = engine.scan_and_trigger(["The war begins"])
        # One from group + ungrouped = 2
        assert len(triggered) == 2

    def test_secondary_and_any_filters(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(
            keywords=["dragon"],
            secondary_keywords=["fire", "ice"],
            selective=True,
            selective_logic=SelectiveLogic.AND_ANY,
            content="Elemental dragon",
        ))
        assert len(engine.scan_and_trigger(["a fire dragon"])) == 1
        assert len(engine.scan_and_trigger(["a dragon flies"])) == 0

    def test_char_description_scanning(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(
            keywords=["brave"],
            content="Bravery lore",
            match_char_description=True,
        ))
        triggered = engine.scan_and_trigger(
            messages=["hello"],
            char_description="A brave warrior",
        )
        assert len(triggered) == 1

    def test_recursive_chain(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(id="e1", keywords=["forest"], content="The dark forest has ancient ruins"))
        engine.add_entry(LorebookEntry(id="e2", keywords=["ancient ruins"], content="Ruins of the old kingdom"))
        engine.add_entry(LorebookEntry(id="e3", keywords=["old kingdom"], content="The kingdom fell long ago"))
        triggered = engine.scan_and_trigger(["into the forest"], recursive=True, max_recursion=5)
        ids = {e.id for e in triggered}
        assert "e1" in ids
        assert "e2" in ids
        assert "e3" in ids  # Triple chain triggered

    def test_sticky_persists_across_turns(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(keywords=["dragon"], content="Dragon", sticky_turns=3))
        engine.scan_and_trigger(["a dragon"])
        engine.advance_turn()
        # Should still trigger even without keyword match
        triggered = engine.scan_and_trigger(["the weather is nice"])
        assert len(triggered) == 1

    def test_cooldown_suppresses(self) -> None:
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.add_entry(LorebookEntry(keywords=["dragon"], content="Dragon", cooldown_turns=2))
        engine.scan_and_trigger(["a dragon"])
        engine.advance_turn()
        # On cooldown
        triggered = engine.scan_and_trigger(["a dragon"])
        assert len(triggered) == 0
        engine.advance_turn()
        engine.advance_turn()
        # Cooldown expired
        triggered = engine.scan_and_trigger(["a dragon"])
        assert len(triggered) == 1

    def test_st_import_then_scan(self) -> None:
        """Full round trip: import ST data, scan, verify triggers."""
        from augmentum.modes.narrative.lore_engine import LoreEngine
        engine = LoreEngine()
        engine.load_from_world_info_json([
            {"key": ["forest"], "content": "Ancient trees", "selective": False},
            {"key": ["castle"], "content": "Stone walls", "selective": False},
        ])
        triggered = engine.scan_and_trigger(["deep in the forest"])
        assert len(triggered) == 1
        assert triggered[0].content == "Ancient trees"
