"""Tests for narrative context builder, macro expander, lore engine, world info buffer."""

from __future__ import annotations

from augmentum.modes.narrative.context_builder import BuiltContext, ContextBlock, ContextBuilder
from augmentum.modes.narrative.lore_engine import LoreEngine, check_secondary, match_keywords
from augmentum.modes.narrative.macro_expander import expand_macros, expand_messages
from augmentum.modes.narrative.world_info_buffer import WorldInfoBuffer
from augmentum.modes.narrative.world_info_groups import filter_by_groups
from augmentum.state.narrative_state import LorebookEntry, LorebookPosition, SelectiveLogic


class TestContextBuilder:
    """Context assembly within token budget."""

    def test_build_empty_returns_empty(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build()
        assert ctx.injected_text == ""
        assert ctx.total_tokens_estimate == 0

    def test_build_with_character_card(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build(character_card_summary="Luna is a mysterious elf mage.")
        # Card summary is turn-STABLE — it renders into stable_text (the
        # checkpoint-covered head region), never the per-turn injection.
        assert "Luna" in ctx.stable_text
        assert "Luna" not in ctx.injected_text
        assert "character_card" in ctx.blocks_used

    def test_build_with_state_text(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build(state_text="Location: Tavern\nTime: Evening")
        assert "Tavern" in ctx.injected_text

    def test_build_with_memory_text(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build(memory_text="Round 1: Luna arrived at the tavern.")
        assert "Luna arrived" in ctx.injected_text

    def test_build_respects_token_budget(self):
        # Tiny budget forces truncation
        builder = ContextBuilder(token_budget=10)
        ctx = builder.build(
            character_card_summary="A" * 500,
            state_text="B" * 500,
        )
        # Should have used some tokens but not unlimited
        assert ctx.total_tokens_estimate <= 20  # very small budget

    def test_build_multiple_blocks(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build(
            character_card_summary="Luna the elf mage",
            state_text="Location: forest",
            memory_text="Round 1: arrival",
            relationship_summary="Luna trusts the party",
        )
        assert len(ctx.blocks_used) >= 2

    def test_context_block_auto_counts_tokens(self):
        block = ContextBlock(label="test", content="This is a short block of text.")
        assert block.token_estimate > 0


class TestBuiltContext:
    """BuiltContext dataclass."""

    def test_defaults(self):
        ctx = BuiltContext()
        assert ctx.injected_text == ""
        assert ctx.blocks_used == []
        assert ctx.total_tokens_estimate == 0
        assert ctx.budget_remaining == 0


class TestMacroExpander:
    """Macro expansion for narrative text."""

    def test_expand_char_macro(self):
        result = expand_macros("Hello, {{char}}!", char_name="Luna")
        assert result == "Hello, Luna!"

    def test_expand_user_macro(self):
        result = expand_macros("{{user}} enters the room.", user_name="Alex")
        assert result == "Alex enters the room."

    def test_expand_obj_macro_alias(self):
        result = expand_macros("{{obj}} looks around.", user_name="Alex")
        assert result == "Alex looks around."

    def test_expand_persona_macro(self):
        result = expand_macros("{{persona}}", persona_description="A brave warrior")
        assert result == "A brave warrior"

    def test_expand_time_macro(self):
        result = expand_macros("Time: {{time}}")
        assert ":" in result  # Should be HH:MM

    def test_expand_date_macro(self):
        result = expand_macros("Date: {{date}}")
        assert "-" in result  # Should be YYYY-MM-DD

    def test_expand_day_macro(self):
        result = expand_macros("Day: {{day}}")
        # Should be a day name
        days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        day_part = result.replace("Day: ", "")
        assert day_part in days

    def test_expand_random_list(self):
        result = expand_macros("Color: {{random:red,blue,green}}")
        assert result.replace("Color: ", "") in {"red", "blue", "green"}

    def test_expand_roll(self):
        result = expand_macros("Roll: {{roll:2d6}}")
        num = int(result.replace("Roll: ", ""))
        assert 2 <= num <= 12

    def test_no_macros_returns_unchanged(self):
        text = "Just plain text without any macros."
        assert expand_macros(text) == text

    def test_expand_messages_modifies_content(self):
        from augmentum.models.base import Message
        messages = [Message(role="user", content="Hello {{char}}!")]
        expand_messages(messages, char_name="Luna")
        assert messages[0].content == "Hello Luna!"

    def test_expand_messages_skips_no_macros(self):
        from augmentum.models.base import Message
        messages = [Message(role="user", content="Hello world")]
        expand_messages(messages, char_name="Luna")
        assert messages[0].content == "Hello world"


class TestLoreEngineKeywordMatching:
    """Lorebook keyword matching."""

    def test_match_keywords_simple(self):
        assert match_keywords(["dragon"], "The dragon breathes fire.") is True

    def test_match_keywords_case_insensitive(self):
        assert match_keywords(["Dragon"], "the dragon breathes fire.") is True

    def test_match_keywords_no_match(self):
        assert match_keywords(["unicorn"], "The dragon breathes fire.") is False

    def test_match_keywords_empty_list(self):
        assert match_keywords([], "Some text") is False

    def test_match_keywords_empty_text(self):
        assert match_keywords(["test"], "") is False

    def test_match_keywords_regex(self):
        assert match_keywords(["/drag.n/i"], "The Dragon breathes fire.") is True

    def test_match_keywords_whole_words(self):
        assert match_keywords(["art"], "The art gallery is open.", whole_words=True) is True
        assert match_keywords(["art"], "The artist paints.", whole_words=True) is False


class TestLoreEngineSecondaryKeywords:
    """Secondary keyword logic (AND_ANY, NOT_ALL, NOT_ANY, AND_ALL)."""

    def test_and_any_passes_with_one_match(self):
        result = check_secondary(
            ["fire", "ice"], "The fire burns bright.",
            SelectiveLogic.AND_ANY,
        )
        assert result is True

    def test_and_any_fails_with_no_match(self):
        result = check_secondary(
            ["water", "ice"], "The fire burns bright.",
            SelectiveLogic.AND_ANY,
        )
        assert result is False

    def test_not_any_passes_when_none_match(self):
        result = check_secondary(
            ["water", "ice"], "The fire burns bright.",
            SelectiveLogic.NOT_ANY,
        )
        assert result is True

    def test_and_all_passes_when_all_match(self):
        result = check_secondary(
            ["fire", "bright"], "The fire burns bright.",
            SelectiveLogic.AND_ALL,
        )
        assert result is True

    def test_and_all_fails_when_one_missing(self):
        result = check_secondary(
            ["fire", "ice"], "The fire burns bright.",
            SelectiveLogic.AND_ALL,
        )
        assert result is False

    def test_empty_secondary_always_true(self):
        result = check_secondary([], "any text", SelectiveLogic.AND_ANY)
        assert result is True


class TestLoreEngine:
    """LoreEngine construction and entry management."""

    def test_construct_empty(self):
        engine = LoreEngine()
        assert engine.entries == {}

    def test_load_from_character_book(self):
        engine = LoreEngine()
        book = {
            "entries": {
                "0": {
                    "keys": ["dragon", "wyrm"],
                    "content": "Dragons are ancient creatures of immense power.",
                    "enabled": True,
                    "position": "before_char",
                },
            },
        }
        loaded = engine.load_from_character_book(book)
        assert len(loaded) >= 1


class TestLorebookAtDepth:
    """At-depth lorebook support: parsing, context-builder bucketing, UI round-trip."""

    def test_parse_our_field_names(self):
        engine = LoreEngine()
        entries = engine.replace_entries_preserving_state([
            {
                "id": "e1",
                "keys": ["x"],
                "content": "note",
                "position": "at_depth",
                "injection_depth": 3,
                "injection_role": "user",
            },
        ])
        assert len(entries) == 1
        assert entries[0].position == LorebookPosition.AT_DEPTH
        assert entries[0].injection_depth == 3
        assert entries[0].injection_role == "user"

    def test_parse_sillytavern_int_enum(self):
        """ST JSON uses ints for position and role — map them correctly."""
        engine = LoreEngine()
        entries = engine.replace_entries_preserving_state([
            {
                "id": "e_st",
                "keys": ["x"],
                "content": "note",
                "position": 4,            # AT_DEPTH
                "depth": 2,
                "role": 2,                # assistant
            },
        ])
        assert entries[0].position == LorebookPosition.AT_DEPTH
        assert entries[0].injection_depth == 2
        assert entries[0].injection_role == "assistant"

    def test_parse_role_string_uppercase_normalized(self):
        engine = LoreEngine()
        entries = engine.replace_entries_preserving_state([
            {"id": "e", "keys": ["x"], "content": "c", "position": "at_depth", "role": "USER"},
        ])
        assert entries[0].injection_role == "user"

    def test_parse_invalid_role_falls_back_to_system(self):
        engine = LoreEngine()
        entries = engine.replace_entries_preserving_state([
            {"id": "e", "keys": ["x"], "content": "c", "position": "at_depth", "role": "bot"},
        ])
        assert entries[0].injection_role == "system"

    def test_context_builder_splits_at_depth_from_injected_text(self):
        """at_depth entries must NOT appear in injected_text — they travel in depth_entries."""
        builder = ContextBuilder(token_budget=4000)
        entry = LorebookEntry(
            id="d1",
            keywords=["x"],
            content="DEPTH_MARKER_UNIQUE",
            position=LorebookPosition.AT_DEPTH,
            injection_depth=3,
            injection_role="system",
        )
        ctx = builder.build(lorebook_entries=[entry])
        assert "DEPTH_MARKER_UNIQUE" not in ctx.injected_text
        assert len(ctx.depth_entries) == 1
        de = ctx.depth_entries[0]
        assert de.content == "DEPTH_MARKER_UNIQUE"
        assert de.depth == 3
        assert de.role == "system"

    def test_context_builder_before_after_unaffected(self):
        """BEFORE_CHAR / AFTER_CHAR entries still land in injected_text as before."""
        builder = ContextBuilder(token_budget=4000)
        before = LorebookEntry(
            id="b1", keywords=["k"], content="BEFORE_MARKER",
            position=LorebookPosition.BEFORE_CHAR,
        )
        after = LorebookEntry(
            id="a1", keywords=["k"], content="AFTER_MARKER",
            position=LorebookPosition.AFTER_CHAR,
        )
        ctx = builder.build(lorebook_entries=[before, after])
        assert "BEFORE_MARKER" in ctx.injected_text
        assert "AFTER_MARKER" in ctx.injected_text
        assert ctx.depth_entries == []

    def test_engine_splice_places_entry_n_messages_from_end(self):
        """Depth=2 entry lands 2 messages back from the final user turn."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.narrative.engine import NarrativeEngine

        # Enable state tracking + lorebook toggles
        orig_tracking = settings.narrative_state_tracking_enabled
        orig_lore = settings.narrative_backend_lorebook
        object.__setattr__(settings, "narrative_state_tracking_enabled", True)
        object.__setattr__(settings, "narrative_backend_lorebook", True)
        try:
            engine = NarrativeEngine(session_id="depth_test")

            # Inject a constant at_depth entry so it activates without keyword match
            engine._lore_engine.replace_entries_preserving_state([
                {
                    "id": "depth_e",
                    "keys": ["anything"],
                    "content": "AT_DEPTH_MARKER",
                    "position": "at_depth",
                    "injection_depth": 2,
                    "injection_role": "system",
                    "constant": True,
                    "enabled": True,
                },
            ])

            # Multi-turn request: 4 non-system messages + 1 system card
            req = InternalChatRequest(
                model="llama3:8b",
                messages=[
                    Message(role="system", content="Card: Luna the mage."),
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="greetings"),
                    Message(role="user", content="current turn"),
                ],
            )
            result = engine.process_request(req)

            # Find the AT_DEPTH_MARKER in the augmented message array
            msgs = result.augmented_request.messages
            marker_idx = next(
                (i for i, m in enumerate(msgs) if m.content == "AT_DEPTH_MARKER"),
                None,
            )
            assert marker_idx is not None, "at_depth entry should appear as a message"

            # Depth=2 means the marker sits 2 messages back from the end.
            # Counting from the final message (index len-1):
            #   index len-1 = depth 0
            #   index len-2 = depth 1
            #   index len-3 = depth 2 ← marker here
            assert marker_idx == len(msgs) - 3, (
                f"depth=2 entry expected at index {len(msgs) - 3}, got {marker_idx}; "
                f"messages: {[(m.role, m.content[:30]) for m in msgs]}"
            )
            assert msgs[marker_idx].role == "system"
        finally:
            object.__setattr__(settings, "narrative_state_tracking_enabled", orig_tracking)
            object.__setattr__(settings, "narrative_backend_lorebook", orig_lore)

    def test_engine_splice_joins_same_depth_same_role(self):
        """Two entries at (depth=1, role=system) should be joined into one message."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.narrative.engine import NarrativeEngine

        orig_tracking = settings.narrative_state_tracking_enabled
        orig_lore = settings.narrative_backend_lorebook
        object.__setattr__(settings, "narrative_state_tracking_enabled", True)
        object.__setattr__(settings, "narrative_backend_lorebook", True)
        try:
            engine = NarrativeEngine(session_id="depth_join_test")
            engine._lore_engine.replace_entries_preserving_state([
                {"id": "a", "keys": ["x"], "content": "FIRST", "position": "at_depth",
                 "injection_depth": 1, "injection_role": "system",
                 "constant": True, "enabled": True, "priority": 10},
                {"id": "b", "keys": ["x"], "content": "SECOND", "position": "at_depth",
                 "injection_depth": 1, "injection_role": "system",
                 "constant": True, "enabled": True, "priority": 20},
            ])
            req = InternalChatRequest(
                model="llama3:8b",
                messages=[
                    Message(role="system", content="Card."),
                    Message(role="user", content="go"),
                ],
            )
            result = engine.process_request(req)
            msgs = result.augmented_request.messages
            joined = next(
                (m for m in msgs if m.role == "system" and "FIRST" in m.content and "SECOND" in m.content),
                None,
            )
            assert joined is not None, (
                f"same-depth same-role entries should be joined; "
                f"messages: {[(m.role, m.content[:50]) for m in msgs]}"
            )
            # Priority 10 should come first (lower = higher priority).
            assert joined.content.index("FIRST") < joined.content.index("SECOND")
        finally:
            object.__setattr__(settings, "narrative_state_tracking_enabled", orig_tracking)
            object.__setattr__(settings, "narrative_backend_lorebook", orig_lore)


class TestWorldInfoBuffer:
    """WorldInfoBuffer scan text assembly."""

    def test_basic_scan_text(self):
        buf = WorldInfoBuffer(chat_messages=["Hello", "World"])
        text = buf.get_scan_text(scan_depth=5)
        assert "Hello" in text
        assert "World" in text

    def test_scan_depth_limits_messages(self):
        buf = WorldInfoBuffer(chat_messages=["A", "B", "C", "D"])
        text = buf.get_scan_text(scan_depth=2)
        assert "A" in text
        assert "B" in text
        # C and D might or might not be included depending on depth

    def test_include_persona(self):
        buf = WorldInfoBuffer(
            chat_messages=["Hello"],
            persona_description="A brave warrior",
        )
        text = buf.get_scan_text(scan_depth=5, include_persona=True)
        assert "brave warrior" in text

    def test_exclude_persona_by_default(self):
        buf = WorldInfoBuffer(
            chat_messages=["Hello"],
            persona_description="A brave warrior",
        )
        text = buf.get_scan_text(scan_depth=5)
        assert "brave warrior" not in text


class TestWorldInfoGroups:
    """Inclusion group filtering."""

    def test_ungrouped_entries_pass_through(self):
        entry = LorebookEntry(id="1", content="test", group="")
        result = filter_by_groups([entry])
        assert len(result) == 1

    def test_grouped_entries_select_one(self):
        e1 = LorebookEntry(id="1", content="dragon lore", group="mythical", group_weight=100)
        e2 = LorebookEntry(id="2", content="phoenix lore", group="mythical", group_weight=100)
        result = filter_by_groups([e1, e2], seed=42)
        # Only one from the group should survive
        grouped = [r for r in result if r.group == "mythical"]
        assert len(grouped) == 1

    def test_override_wins(self):
        e1 = LorebookEntry(id="1", content="dragon", group="mythical", group_weight=100)
        e2 = LorebookEntry(id="2", content="phoenix", group="mythical", group_override=True)
        result = filter_by_groups([e1, e2], seed=42)
        grouped = [r for r in result if r.group == "mythical"]
        assert len(grouped) == 1
        assert grouped[0].content == "phoenix"

    def test_empty_input_returns_empty(self):
        assert filter_by_groups([]) == []


class TestStableHeadPlacement:
    """Turn-stable context must ride in the checkpoint-covered head.

    Live-measured 2026-07-18: example dialogue / card summary / tool
    guidance floating in the per-turn injection re-prefilled 4-6.5k
    tokens every narrative turn (KV reuse 44% vs 80% expected) because
    the injection point moves every turn while its content doesn't.
    """

    def _engine_and_request(self, example: str = "EXAMPLE_DLG_MARKER"):
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.narrative.card_parser import CharacterCard
        from augmentum.modes.narrative.engine import NarrativeEngine

        engine = NarrativeEngine(session_id="stable_head_test")
        engine._character_card = CharacterCard(
            name="Luna",
            description="An elf mage.",
            example_dialogue=example,
        )
        # _initialize() would re-parse the card from the system prompt and
        # replace the injected one — skip it, we're testing placement.
        engine._initialized = True
        req = InternalChatRequest(
            model="llama3:8b",
            messages=[
                Message(role="system", content="Card: Luna the mage."),
                Message(role="user", content="hello"),
                Message(role="assistant", content="greetings"),
                Message(role="user", content="current turn"),
            ],
        )
        return settings, engine, req

    def test_example_dialogue_lands_in_stable_head_not_injection(self):
        settings, engine, req = self._engine_and_request()
        orig = settings.narrative_backend_examples
        object.__setattr__(settings, "narrative_backend_examples", True)
        try:
            result = engine.process_request(req)
            msgs = result.augmented_request.messages
            head_idx = next(
                (i for i, m in enumerate(msgs)
                 if m.role == "system" and "EXAMPLE_DLG_MARKER" in (m.content or "")),
                None,
            )
            assert head_idx is not None, (
                f"example dialogue missing from payload: "
                f"{[(m.role, (m.content or '')[:40]) for m in msgs]}"
            )
            # Head region = before the first non-system message, NOT the
            # floating per-turn injection near the tail.
            first_chat = next(
                i for i, m in enumerate(msgs) if m.role != "system"
            )
            assert head_idx < first_chat, (
                f"example dialogue at index {head_idx} but chat starts at "
                f"{first_chat} — it's floating in the per-turn tail again"
            )
        finally:
            object.__setattr__(settings, "narrative_backend_examples", orig)

    def test_stable_head_mirrored_into_kv_stable_messages(self):
        settings, engine, req = self._engine_and_request()
        orig = settings.narrative_backend_examples
        object.__setattr__(settings, "narrative_backend_examples", True)
        try:
            result = engine.process_request(req)
            live = result.augmented_request.messages
            stable = result.augmented_request.kv_stable_messages
            assert stable, "kv_stable_messages snapshot missing"
            live_head = next(
                (m.content for m in live
                 if m.role == "system" and "EXAMPLE_DLG_MARKER" in (m.content or "")),
                None,
            )
            stable_head = next(
                (m.content for m in stable
                 if m.role == "system" and "EXAMPLE_DLG_MARKER" in (m.content or "")),
                None,
            )
            assert live_head is not None and stable_head is not None, (
                "stable head must exist in BOTH live payload and checkpoint snapshot"
            )
            # Byte-aligned — that's the whole checkpoint contract.
            assert live_head == stable_head
        finally:
            object.__setattr__(settings, "narrative_backend_examples", orig)

    def test_dynamic_state_stays_in_floating_injection(self):
        builder = ContextBuilder(token_budget=4000)
        ctx = builder.build(
            character_card_summary="Luna the elf mage",
            state_text="Location: Tavern DYNAMIC_STATE_MARKER",
        )
        # State changes turn-to-turn — it must NOT be pinned into the
        # stable head (a per-turn-changing head block would invalidate
        # the whole prefix at message 0).
        assert "DYNAMIC_STATE_MARKER" in ctx.injected_text
        assert "DYNAMIC_STATE_MARKER" not in ctx.stable_text
        assert "Luna the elf mage" in ctx.stable_text
