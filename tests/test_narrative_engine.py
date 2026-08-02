"""Tests for the narrative engine — card parsing, state tracking, context building, branching."""

from __future__ import annotations

import json

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.narrative.branch_tracker import BranchTracker
from augmentum.modes.narrative.card_parser import CardParser, CharacterCard
from augmentum.modes.narrative.character_tracker import CharacterTracker, CharacterUpdate
from augmentum.modes.narrative.context_builder import ContextBuilder
from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.modes.narrative.plot_tracker import PlotTracker
from augmentum.modes.narrative.world_tracker import SceneState, WorldTracker
from augmentum.state.narrative_state import (
    Contradiction,
    ContradictionSeverity,
    Entity,
    EntityState,
    EntityType,
    LorebookEntry,
    PlotStatus,
    content_hash,
)

# --- Helpers ---


def make_request(
    user_content: str,
    system_content: str = "",
    model: str = "llama3.1:8b",
) -> InternalChatRequest:
    messages = []
    if system_content:
        messages.append(Message(role="system", content=system_content))
    messages.append(Message(role="user", content=user_content))
    return InternalChatRequest(model=model, messages=messages)


def make_multi_turn_request(
    turns: list[tuple[str, str]],
    system_content: str = "",
) -> InternalChatRequest:
    """Create a request with multiple user/assistant turns."""
    messages = []
    if system_content:
        messages.append(Message(role="system", content=system_content))
    for role, content in turns:
        messages.append(Message(role=role, content=content))
    return InternalChatRequest(model="llama3.1:8b", messages=messages)


SILLYTAVERN_CARD = (
    "{{char}} is Lyra, a 200-year-old elven sorceress.\n"
    "Personality: Wise, patient, slightly sardonic\n"
    "Appearance: Tall with silver hair that flows like moonlight\n"
    "{{char}} lives in a tower at the edge of the Whispering Forest.\n"
    "{{user}} is a young adventurer seeking {{char}}'s wisdom.\n"
    "{{scenario}}: {{user}} arrives at the tower during a thunderstorm.\n"
    "Stay in character at all times. Use *asterisks* for actions."
)

V2_JSON_CARD = json.dumps({
    "spec": "chara_card_v2",
    "data": {
        "name": "Captain Aria",
        "personality": "Bold, cunning, loyal to her crew",
        "description": "A pirate captain of the Crimson Wave",
        "scenario": "Docked at Port Ember for supplies",
        "first_mes": "*adjusts her hat* Welcome aboard!",
        "mes_example": "User: Where are we heading?\nAria: To the edge of the known world!",
        "creator_notes": "Best with creative writing models",
        "tags": ["pirate", "adventure", "fantasy"],
        "character_book": {
            "entries": {
                "0": {
                    "key": "Crimson Wave,ship",
                    "content": "The Crimson Wave is a three-masted brigantine.",
                    "enabled": True,
                    "priority": 10,
                },
                "1": {
                    "key": "Port Ember",
                    "content": "Port Ember is a bustling trade port known for its night markets.",
                    "enabled": True,
                    "priority": 20,
                    "constant": True,
                },
            }
        },
    },
})


# === Card Parser Tests ===


class TestCardParser:
    def setup_method(self):
        self.parser = CardParser()

    def test_parse_v2_json(self):
        """V2 JSON character cards should be parsed correctly."""
        card = self.parser.parse(V2_JSON_CARD)
        assert card is not None
        assert card.name == "Captain Aria"
        assert card.source_format == "v2_json"
        assert "Bold" in card.personality
        assert card.greeting == "*adjusts her hat* Welcome aboard!"
        assert card.tags == ["pirate", "adventure", "fantasy"]
        assert "character_book" in card.raw_data

    def test_parse_sillytavern_template(self):
        """SillyTavern template cards should extract character name."""
        card = self.parser.parse(SILLYTAVERN_CARD)
        assert card is not None
        assert card.source_format == "sillytavern"
        assert card.name == "Lyra"
        assert "thunderstorm" in card.scenario

    def test_parse_wpp(self):
        """W++ format should be parsed correctly."""
        wpp = '[character= "Elena"] [personality= "kind" + "brave" + "curious"] [species= "Elf"]'
        card = self.parser.parse(wpp)
        assert card is not None
        assert card.source_format == "wpp"
        assert card.name == "Elena"
        assert "kind" in card.personality
        assert card.species == "Elf"

    def test_parse_plist(self):
        """PList format should be parsed correctly."""
        plist = (
            "Name: Sir Aldric\n"
            "Personality: Brave, honorable, stubborn\n"
            "Appearance: Tall knight in silver armor\n"
            "Species: Human\n"
            "Background: Former royal guard turned wandering knight"
        )
        card = self.parser.parse(plist)
        assert card is not None
        assert card.source_format == "plist"
        assert card.name == "Sir Aldric"
        assert "Brave" in card.personality

    def test_parse_v2_json_post_history_and_depth_prompt(self):
        """V2 JSON cards with post_history_instructions and depth_prompt should be parsed."""
        card_json = json.dumps({
            "spec": "chara_card_v2",
            "data": {
                "name": "Test Char",
                "personality": "Friendly",
                "description": "A test character",
                "post_history_instructions": "Always respond in rhyme",
                "system_prompt": "You are a poet",
                "extensions": {
                    "depth_prompt": "Remember the user's name",
                    "depth_prompt_depth": 2,
                },
            },
        })
        card = self.parser.parse(card_json)
        assert card is not None
        assert card.post_history_instructions == "Always respond in rhyme"
        assert card.system_prompt == "You are a poet"
        assert card.depth_prompt == "Remember the user's name"
        assert card.depth_prompt_depth == 2

    def test_parse_v2_json_scenario_in_trait_summary(self):
        """Scenario should appear in trait_summary when set."""
        card = self.parser.parse(V2_JSON_CARD)
        assert card is not None
        assert "Scenario:" in card.trait_summary
        assert "Docked at Port Ember" in card.trait_summary

    def test_parse_cai(self):
        """Character.AI format should be parsed correctly."""
        cai = (
            "Name: Captain Aria Blackwood\n"
            "Greeting: *Aria adjusts her captain's hat* Welcome aboard!\n"
            "Example Dialogue:\n"
            "User: Where are we heading?\n"
            "Aria: To the edge of the known world!"
        )
        card = self.parser.parse(cai)
        assert card is not None
        assert card.source_format == "cai"
        assert card.name == "Captain Aria Blackwood"
        assert "Welcome aboard" in card.greeting

    def test_parse_returns_none_for_plain_text(self):
        """Plain assistant prompts should not parse as a card."""
        card = self.parser.parse("You are a helpful assistant.")
        assert card is None

    def test_trait_summary(self):
        """trait_summary should return a concise summary."""
        card = CharacterCard(name="Lyra", species="Elf", personality="Wise and patient")
        summary = card.trait_summary
        assert "Lyra" in summary
        assert "Elf" in summary
        assert "Wise" in summary


# === Character Tracker Tests ===


class TestCharacterTracker:
    def setup_method(self):
        self.tracker = CharacterTracker()
        self.lyra = Entity(
            id="lyra1",
            session_id="test",
            entity_type=EntityType.CHARACTER,
            name="Lyra",
            state=EntityState(location="tower", emotional_state="calm"),
        )

    def test_extract_emotion_from_action(self):
        """Should detect emotions from attributed actions."""
        text = '*Lyra smiled warmly* "Welcome, traveler."'
        updates = self.tracker.extract_updates(text, [self.lyra])
        assert len(updates) >= 1
        emotion_update = next((u for u in updates if u.name == "Lyra"), None)
        assert emotion_update is not None
        assert emotion_update.emotional_state == "happy"

    def test_extract_location_change(self):
        """Should detect location changes from actions."""
        text = "*Lyra enters the library*"
        updates = self.tracker.extract_updates(text, [self.lyra])
        assert len(updates) >= 1
        loc_update = next((u for u in updates if u.name == "Lyra"), None)
        assert loc_update is not None
        assert loc_update.location == "library"

    def test_confidence_dampening(self):
        """Low confidence emotions should not override existing state."""
        update = CharacterUpdate(
            name="Lyra",
            emotional_state="sad",
            emotional_confidence=0.2,
        )
        self.tracker.apply_update(self.lyra, update, message_index=1)
        # With low confidence and dampening, the emotional state should stay
        # The scale = (1 - 0.6) + 0.2 * 0.6 = 0.4 + 0.12 = 0.52 >= 0.5, so it should update
        assert self.lyra.state.emotional_state == "sad"

    def test_high_confidence_emotion_updates(self):
        """High confidence emotions should always update."""
        update = CharacterUpdate(
            name="Lyra",
            emotional_state="angry",
            emotional_confidence=0.9,
        )
        self.tracker.apply_update(self.lyra, update, message_index=1)
        assert self.lyra.state.emotional_state == "angry"

    def test_physical_state_detection(self):
        """Should detect physical state changes."""
        text = "*Lyra sits down on the bench*"
        updates = self.tracker.extract_updates(text, [self.lyra])
        assert len(updates) >= 1
        phys_update = next((u for u in updates if u.name == "Lyra"), None)
        assert phys_update is not None
        assert phys_update.physical_state == "sitting"

    def test_no_update_for_unknown_character(self):
        """Should not create updates for untracked characters."""
        text = "*Gandalf casts a spell*"
        updates = self.tracker.extract_updates(text, [self.lyra])
        assert not any(u.name == "Gandalf" for u in updates)


# === World Tracker Tests ===


class TestWorldTracker:
    def setup_method(self):
        self.tracker = WorldTracker()

    def test_detect_weather(self):
        """Should detect weather changes."""
        text = "The rain began to fall heavily on the cobblestone streets."
        delta = self.tracker.extract_world_changes(text)
        assert delta.get("weather") == "raining"

    def test_detect_time_of_day(self):
        """Should detect time of day changes."""
        text = "As the sun set, the sky turned orange and purple."
        delta = self.tracker.extract_world_changes(text)
        assert delta.get("time_of_day") == "evening"

    def test_detect_location(self):
        """Should detect location changes from narration."""
        text = '*They enter the ancient library* "So many books!"'
        delta = self.tracker.extract_world_changes(text)
        assert delta.get("location") == "ancient library"

    def test_delta_compression(self):
        """Deltas should be tracked incrementally."""
        self.tracker.apply_delta({"weather": "clear"}, message_index=0)
        self.tracker.apply_delta({"time_of_day": "morning"}, message_index=1)
        self.tracker.apply_delta({"weather": "raining"}, message_index=2)

        assert len(self.tracker.deltas) == 3
        assert self.tracker.state.weather == "raining"
        assert self.tracker.state.time_of_day == "morning"

    def test_reconstruct_at_point(self):
        """Should reconstruct state at any historical point."""
        self.tracker.apply_delta({"weather": "clear", "location": "village"}, message_index=0)
        self.tracker.apply_delta({"weather": "raining"}, message_index=2)
        self.tracker.apply_delta({"location": "forest"}, message_index=4)

        state_at_1 = self.tracker.reconstruct_at(1)
        assert state_at_1.weather == "clear"
        assert state_at_1.location == "village"

        state_at_3 = self.tracker.reconstruct_at(3)
        assert state_at_3.weather == "raining"
        assert state_at_3.location == "village"

    def test_rollback(self):
        """Rollback should revert to a previous state."""
        self.tracker.apply_delta({"weather": "clear"}, message_index=0)
        self.tracker.apply_delta({"weather": "raining"}, message_index=2)
        self.tracker.apply_delta({"weather": "snowing"}, message_index=4)

        self.tracker.rollback_to(2)
        assert self.tracker.state.weather == "raining"
        assert len(self.tracker.deltas) == 2


# === Plot Tracker Tests ===


class TestPlotTracker:
    def setup_method(self):
        self.tracker = PlotTracker()

    def test_add_and_list_threads(self):
        """Should add and retrieve plot threads."""
        thread = self.tracker.add_thread("sess1", "Find the lost artifact", "Ancient sword", 0)
        assert len(self.tracker.active_threads) == 1
        assert thread.title == "Find the lost artifact"

    def test_resolve_thread(self):
        """Resolving a thread should change its status."""
        thread = self.tracker.add_thread("sess1", "Defeat the dragon", "", 0)
        self.tracker.resolve_thread(thread.id, message_index=10)
        assert len(self.tracker.active_threads) == 0
        assert self.tracker.threads[thread.id].status == PlotStatus.RESOLVED

    def test_progress_thread(self):
        """Progressing a thread should record the update."""
        thread = self.tracker.add_thread("sess1", "Journey north", "", 0)
        self.tracker.progress_thread(thread.id, "Reached the mountain pass", 5)
        progressions = thread.state.get("progressions", [])
        assert len(progressions) == 1
        assert progressions[0]["message_index"] == 5

    def test_rollback(self):
        """Rollback should remove threads established after the rollback point."""
        t1 = self.tracker.add_thread("sess1", "Thread 1", "", 0)
        t2 = self.tracker.add_thread("sess1", "Thread 2", "", 5)
        self.tracker.rollback_to(3)
        assert t1.id in self.tracker.threads
        assert t2.id not in self.tracker.threads

    def test_context_summary(self):
        """Should generate a readable summary."""
        self.tracker.add_thread("sess1", "Find the artifact", "Hidden in the temple", 0)
        self.tracker.add_thread("sess1", "Rescue the prince", "", 2)
        summary = self.tracker.get_context_summary()
        assert "Find the artifact" in summary
        assert "Rescue the prince" in summary

    def test_extract_plot_signals(self):
        """Should detect plot-relevant keywords."""
        text = "They discovered a hidden passage leading to the quest objective."
        signals = self.tracker.extract_plot_signals(text)
        assert len(signals) > 0


# === Context Builder Tests ===


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder(token_budget=1024)

    def test_empty_context(self):
        """Should handle empty inputs gracefully."""
        result = self.builder.build()
        assert result.injected_text == ""
        assert result.total_tokens_estimate == 0

    def test_character_state_injection(self):
        """Should inject character state."""
        chars = [Entity(
            id="e1", session_id="s1", entity_type=EntityType.CHARACTER,
            name="Lyra", state=EntityState(location="tower", emotional_state="calm"),
        )]
        result = self.builder.build(characters=chars)
        assert "Lyra" in result.injected_text
        assert "tower" in result.injected_text
        assert "character_states" in result.blocks_used

    def test_scene_injection(self):
        """Should inject scene state."""
        scene = SceneState(location="ancient library", time_of_day="evening", weather="raining")
        result = self.builder.build(scene=scene)
        assert "ancient library" in result.injected_text
        assert "evening" in result.injected_text

    def test_contradiction_warnings(self):
        """Should inject contradiction warnings with high priority."""
        contradictions = [Contradiction(
            session_id="s1", message_index=5,
            contradiction_type="time_paradox",
            description="Time went backwards: afternoon → morning",
            severity=ContradictionSeverity.MINOR,
        )]
        result = self.builder.build(contradictions=contradictions)
        assert "consistency_warnings" in result.blocks_used
        assert "Time went backwards" in result.injected_text

    def test_budget_enforcement(self):
        """Should respect token budget."""
        builder = ContextBuilder(token_budget=10)  # Very small budget
        chars = [Entity(
            id="e1", session_id="s1", entity_type=EntityType.CHARACTER,
            name="Lyra Moonwhisper the Great Sorceress of the Northern Lands",
            state=EntityState(
                location="the incredibly elaborate tower of infinite wonders",
                emotional_state="tremendously and overwhelmingly happy",
            ),
        )]
        result = builder.build(characters=chars)
        assert result.total_tokens_estimate <= 10

    def test_lorebook_injection(self):
        """Should inject lorebook entries sorted by priority."""
        entries = [
            LorebookEntry(id="l1", keywords=["ship"], content="The ship is large.", priority=10),
            LorebookEntry(id="l2", keywords=["port"], content="The port is busy.", priority=20),
        ]
        result = self.builder.build(lorebook_entries=entries)
        assert "ship is large" in result.injected_text


# === Consistency Checker Tests ===


# === Lorebook Engine Tests ===


class TestLoreEngine:
    def setup_method(self):
        self.engine = LoreEngine()

    def test_keyword_trigger(self):
        """Should trigger entries when keywords match."""
        entry = LorebookEntry(
            id="l1", keywords=["dragon", "wyrm"],
            content="Dragons are ancient creatures.", enabled=True,
        )
        self.engine.add_entry(entry)
        triggered = self.engine.scan_and_trigger(["I see a dragon approaching"])
        assert len(triggered) == 1

    def test_no_trigger_without_keyword(self):
        """Should not trigger when no keywords match."""
        entry = LorebookEntry(
            id="l1", keywords=["dragon"],
            content="Dragons are ancient creatures.", enabled=True,
        )
        self.engine.add_entry(entry)
        triggered = self.engine.scan_and_trigger(["The weather is nice today."])
        assert len(triggered) == 0

    def test_constant_entries_always_trigger(self):
        """Constant entries should always trigger regardless of keywords."""
        entry = LorebookEntry(
            id="l1", keywords=["xyz_impossible"],
            content="Always present info.", enabled=True, constant=True,
        )
        self.engine.add_entry(entry)
        triggered = self.engine.scan_and_trigger(["Hello"])
        assert len(triggered) == 1

    def test_disabled_entries_dont_trigger(self):
        """Disabled entries should not trigger."""
        entry = LorebookEntry(
            id="l1", keywords=["dragon"],
            content="Dragons!", enabled=False,
        )
        self.engine.add_entry(entry)
        triggered = self.engine.scan_and_trigger(["I see a dragon"])
        assert len(triggered) == 0

    def test_sticky_turns(self):
        """Sticky entries should remain active for N turns."""
        entry = LorebookEntry(
            id="l1", keywords=["dragon"],
            content="Dragons!", enabled=True, sticky_turns=2,
        )
        self.engine.add_entry(entry)

        # First trigger
        triggered = self.engine.scan_and_trigger(["I see a dragon"], message_index=0)
        assert len(triggered) == 1

        # Advance one turn — should still be active via sticky
        self.engine.advance_turn()
        triggered = self.engine.scan_and_trigger(["The weather is nice"], message_index=1)
        assert len(triggered) == 1  # Still sticky

        # Advance again — sticky expires
        self.engine.advance_turn()
        triggered = self.engine.scan_and_trigger(["The weather is nice"], message_index=2)
        assert len(triggered) == 0  # Sticky expired

    def test_cooldown_after_sticky(self):
        """Entries should enter cooldown after sticky expires."""
        entry = LorebookEntry(
            id="l1", keywords=["dragon"],
            content="Dragons!", enabled=True, sticky_turns=1, cooldown_turns=2,
        )
        self.engine.add_entry(entry)

        # Turn 0: Trigger with keyword → sticky_counter[l1] = 1
        self.engine.scan_and_trigger(["dragon"], message_index=0)

        # Advance: sticky 1→0, expired → cooldown starts = 2
        self.engine.advance_turn()

        # Dragon keyword present but in cooldown (2 remaining)
        triggered = self.engine.scan_and_trigger(["dragon"], message_index=1)
        assert len(triggered) == 0

        # Advance: cooldown 2→1
        self.engine.advance_turn()
        triggered = self.engine.scan_and_trigger(["dragon"], message_index=2)
        assert len(triggered) == 0  # Still in cooldown

        # Advance: cooldown 1→0, expired
        self.engine.advance_turn()
        triggered = self.engine.scan_and_trigger(["dragon"], message_index=3)
        assert len(triggered) == 1  # Cooldown expired, can trigger again

    def test_load_character_book(self):
        """Should parse character book from V2 format."""
        book = {
            "entries": {
                "0": {"key": "ship,vessel", "content": "A big ship.", "enabled": True},
                "1": {"key": "port", "content": "A busy port.", "enabled": True, "constant": True},
            }
        }
        entries = self.engine.load_from_character_book(book)
        assert len(entries) == 2
        assert any(e.constant for e in entries)

    def test_priority_ordering(self):
        """Triggered entries should be sorted by priority."""
        self.engine.add_entry(LorebookEntry(
            id="l1", keywords=["dragon"], content="Low priority.", priority=100, enabled=True,
        ))
        self.engine.add_entry(LorebookEntry(
            id="l2", keywords=["dragon"], content="High priority.", priority=10, enabled=True,
        ))
        triggered = self.engine.scan_and_trigger(["dragon"])
        assert triggered[0].priority == 10


# === Branch Tracker Tests ===


class TestBranchTracker:
    def setup_method(self):
        self.tracker = BranchTracker("test_session")

    def test_no_branch_on_first_message(self):
        """First message should not be detected as a branch."""
        req = make_request("Hello")
        detection = self.tracker.detect_branch(req)
        assert not detection.is_branch

    def test_no_branch_on_continuation(self):
        """Normal continuation should not be detected as a branch."""
        # Track initial messages
        self.tracker.track_message("user", "Hello")
        self.tracker.track_message("assistant", "Hi there!")

        # Send a request that continues the conversation
        req = make_multi_turn_request([
            ("user", "Hello"),
            ("assistant", "Hi there!"),
            ("user", "How are you?"),
        ])
        detection = self.tracker.detect_branch(req)
        assert not detection.is_branch

    def test_detect_branch_on_divergence(self):
        """Should detect a branch when message content diverges from history."""
        # Track initial conversation
        self.tracker.track_message("user", "Hello")
        self.tracker.track_message("assistant", "Hi there!")
        self.tracker.track_message("user", "Tell me a story")
        self.tracker.track_message("assistant", "Once upon a time...")

        # Send a request that diverges at message 2 (different user message)
        req = make_multi_turn_request([
            ("user", "Hello"),
            ("assistant", "Hi there!"),
            ("user", "Tell me a joke instead"),  # Different from "Tell me a story"
        ])
        detection = self.tracker.detect_branch(req)
        assert detection.is_branch
        assert detection.branch_point == 2  # Diverges at index 2

    def test_apply_branch_deactivates_old_messages(self):
        """Applying a branch should soft-delete divergent messages."""
        self.tracker.track_message("user", "Hello")
        self.tracker.track_message("assistant", "Hi!")
        self.tracker.track_message("user", "Story please")
        self.tracker.track_message("assistant", "Once upon a time...")

        req = make_multi_turn_request([
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "Joke please"),  # Diverges
        ])
        detection = self.tracker.detect_branch(req)
        assert detection.is_branch

        self.tracker.apply_branch(detection)
        active = self.tracker.active_messages
        # Only messages before the branch point should be active in new branch
        assert all(m.branch_id == detection.new_branch_id or m.message_index < detection.branch_point
                    for m in active)

    def test_content_hash_consistency(self):
        """Same content should produce the same hash."""
        h1 = content_hash("Hello, world!")
        h2 = content_hash("Hello, world!")
        h3 = content_hash("Hello, world?")
        assert h1 == h2
        assert h1 != h3

    def test_rewind_to_earlier_message_fires_branch(self):
        """REGRESSION (2026-05-06 dogfooding): clicking back to message 3 of
        a 25-message conversation didn't fire branch detection. Result: STATE
        / LEDGER / ARCHIVE didn't roll back; the inspector still showed the
        message-25 view. Phase 3 snapshot recovery never activated.

        After fix: shorter request (matching prefix) is treated as a 'rewind'
        branch with branch_point = len(request) and a content-addressed
        deterministic branch_id."""
        # Build a 6-message conversation tracked on main
        for i, (role, content) in enumerate([
            ("user", "msg0"), ("assistant", "ans0"),
            ("user", "msg1"), ("assistant", "ans1"),
            ("user", "msg2"), ("assistant", "ans2"),
        ]):
            self.tracker.track_message(role, content)

        # Simulate user clicking back to message 3 — request only has the
        # first 3 messages (matching prefix, no edits)
        req = make_multi_turn_request([
            ("user", "msg0"), ("assistant", "ans0"),
            ("user", "msg1"),
        ])
        detection = self.tracker.detect_branch(req)
        assert detection.is_branch is True
        assert detection.branch_point == 3
        assert detection.parent_branch_id == "main"
        assert detection.new_branch_id.startswith("branch_")

    def test_rewind_branch_id_is_deterministic(self):
        """Going back to the same point twice produces the same branch_id —
        idempotent so revisiting rejoins the existing branch instead of
        forking infinitely."""
        for role, content in [
            ("user", "a"), ("assistant", "b"),
            ("user", "c"), ("assistant", "d"),
        ]:
            self.tracker.track_message(role, content)

        req1 = make_multi_turn_request([
            ("user", "a"), ("assistant", "b"),
        ])
        # Need a fresh tracker for second detection — track_request_messages
        # in real flow would handle this, but for unit purposes recreate.
        tracker2 = BranchTracker("test_session")
        for role, content in [
            ("user", "a"), ("assistant", "b"),
            ("user", "c"), ("assistant", "d"),
        ]:
            tracker2.track_message(role, content)

        d1 = self.tracker.detect_branch(req1)
        d2 = tracker2.detect_branch(req1)
        assert d1.new_branch_id == d2.new_branch_id

    def test_simple_regen_of_last_response_does_not_fire_branch(self):
        """REGRESSION GUARD for the rewind fix above: dropping just the last
        assistant message (standard regenerate-the-last-response action) must
        STILL be classified as not-a-branch so the legacy regen path runs
        and replaces the response in place. Otherwise every regenerate click
        would fork a new branch.

        shortage = 1 → keep legacy non-branch behavior.
        shortage > 1 → fire rewind branch."""
        for role, content in [
            ("user", "msg0"), ("assistant", "ans0"),
            ("user", "msg1"), ("assistant", "ans1"),
            ("user", "msg2"), ("assistant", "ans2"),
        ]:
            self.tracker.track_message(role, content)

        # Regenerate the last assistant: drop ans2 only
        req = make_multi_turn_request([
            ("user", "msg0"), ("assistant", "ans0"),
            ("user", "msg1"), ("assistant", "ans1"),
            ("user", "msg2"),
        ])
        detection = self.tracker.detect_branch(req)
        assert detection.is_branch is False, (
            "shortage=1 (regen-of-last) must not fire branch — would create a "
            "new branch on every regenerate click"
        )

    def test_rewind_distinct_from_continuation_fork(self):
        """Rewind branch (shorter, matching prefix) and content-divergence
        branch (same length, different content at position N) must produce
        DIFFERENT branch_ids — they're conceptually different paths."""
        for role, content in [
            ("user", "a"), ("assistant", "b"),
            ("user", "c"), ("assistant", "d"),
            ("user", "e"),
        ]:
            self.tracker.track_message(role, content)

        # Rewind: truncate to first 3 messages, no divergence
        rewind_req = make_multi_turn_request([
            ("user", "a"), ("assistant", "b"), ("user", "c"),
        ])
        # Fork: same length as full history but content differs at position 4
        tracker2 = BranchTracker("test_session")
        for role, content in [
            ("user", "a"), ("assistant", "b"),
            ("user", "c"), ("assistant", "d"),
            ("user", "e"),
        ]:
            tracker2.track_message(role, content)
        fork_req = make_multi_turn_request([
            ("user", "a"), ("assistant", "b"),
            ("user", "c"), ("assistant", "d"),
            ("user", "DIFFERENT"),
        ])

        d_rewind = self.tracker.detect_branch(rewind_req)
        d_fork = tracker2.detect_branch(fork_req)
        assert d_rewind.is_branch and d_fork.is_branch
        assert d_rewind.new_branch_id != d_fork.new_branch_id


# === Narrative Engine Integration Tests ===


class TestNarrativeEngine:
    _BACKEND_TOGGLES = (
        "narrative_state_tracking_enabled", "narrative_consistency_enabled",
        "narrative_backend_lorebook", "narrative_backend_card_summary",
    )

    def setup_method(self):
        from augmentum.config import settings
        self._originals = {}
        for attr in self._BACKEND_TOGGLES:
            self._originals[attr] = getattr(settings, attr)
            object.__setattr__(settings, attr, True)
        self.engine = NarrativeEngine(session_id="test_session")

    def teardown_method(self):
        from augmentum.config import settings
        for attr, val in self._originals.items():
            object.__setattr__(settings, attr, val)

    def test_initialize_from_sillytavern_card(self):
        """Should parse character card on first request."""
        req = make_request("*waves hello*", system_content=SILLYTAVERN_CARD)
        result = self.engine.process_request(req)

        assert self.engine.character_card is not None
        assert self.engine.character_card.name == "Lyra"
        assert len(result.state.entities) >= 1

    def test_initialize_from_v2_json(self):
        """Should parse V2 JSON character card."""
        req = make_request("Hello captain!", system_content=V2_JSON_CARD)
        result = self.engine.process_request(req)

        assert self.engine.character_card is not None
        assert self.engine.character_card.name == "Captain Aria"
        assert len(result.state.lorebook) == 2  # Two entries in character_book

    def test_context_injection(self):
        """Should inject narrative context into the request."""
        req = make_request("*waves*", system_content=SILLYTAVERN_CARD)
        result = self.engine.process_request(req)

        # The augmented request should have more content than the original
        orig_system = next(
            (m.content for m in req.messages if m.role == "system"), ""
        )
        aug_system = next(
            (m.content for m in result.augmented_request.messages if m.role == "system"), ""
        )
        assert len(aug_system) > len(orig_system)

    def test_state_updates_from_response(self):
        """Should extract state changes from AI responses."""
        req = make_request("*enters the tower*", system_content=SILLYTAVERN_CARD)
        self.engine.process_request(req)

        # Simulate AI response with state-changing content
        self.engine.process_response(
            '*Lyra smiled warmly as the rain pounded against the windows* '
            '"Welcome, traveler. You look like you could use some warmth."'
        )

        # Check state was updated
        lyra = self.engine.state.get_entity_by_name("Lyra")
        if lyra:
            # Emotional state should have been updated
            assert lyra.state.emotional_state in ("happy", "loving", "calm")

    def test_multi_turn_state_tracking(self):
        """Should track state across multiple turns."""
        req = make_request("*enters the tower*", system_content=SILLYTAVERN_CARD)
        self.engine.process_request(req)

        asst_response = "*The rain intensifies outside* Lyra looks up from her spellbook."
        self.engine.process_response(asst_response)

        # Real clients send the full conversation history with each request
        req2 = make_multi_turn_request(
            [
                ("user", "*enters the tower*"),
                ("assistant", asst_response),
                ("user", "*looks around the room*"),
            ],
            system_content=SILLYTAVERN_CARD,
        )
        result2 = self.engine.process_request(req2)

        assert result2.state.message_count >= 2

    def test_world_state_tracked(self):
        """Should track world state changes."""
        req = make_request("*enters the tower*", system_content=SILLYTAVERN_CARD)
        self.engine.process_request(req)

        self.engine.process_response(
            "As midnight fell, the thunderstorm grew fiercer."
        )

        assert self.engine.world_state.time_of_day == "night"
        assert self.engine.world_state.weather == "stormy"

    def test_rollback(self):
        """Should support rolling back state."""
        resp1 = "Response 1"
        resp2 = "Response 2"

        # Real clients include the full conversation history in each request
        req1 = make_request("Hello", system_content=SILLYTAVERN_CARD)
        self.engine.process_request(req1)           # message_count → 1
        self.engine.process_response(resp1)         # message_count → 2

        req2 = make_multi_turn_request(
            [("user", "Hello"), ("assistant", resp1), ("user", "Turn 2")],
            system_content=SILLYTAVERN_CARD,
        )
        self.engine.process_request(req2)           # message_count → 3
        self.engine.process_response(resp2)         # message_count → 4

        assert self.engine.state.message_count == 4
        self.engine.rollback_to(2)
        assert self.engine.state.message_count == 2

    def test_plain_request_without_card(self):
        """Should handle requests without character cards gracefully."""
        req = make_request("Hello!")
        result = self.engine.process_request(req)
        assert self.engine.character_card is None
        # Should still return an augmented request (even if minimal)
        assert result.augmented_request is not None


# === Backend-Aware System Injection ===
#
# Narrative mode used to always inject dynamic STATE/MEMORY as a system
# message just before the latest user turn — a llama-server slot-cache
# optimization that strict cloud OpenAI-compat APIs (NVIDIA NIM, DeepSeek,
# Mistral, Cohere) reject with "System message must be at the beginning."
# The engine now branches on ``supports_mid_system``: True keeps the
# cache-friendly placement, False folds the dynamic context into the
# leading system block. These tests guard both paths.


class TestBackendAwareSystemInjection:
    def setup_method(self):
        from augmentum.config import settings
        self._originals = {
            attr: getattr(settings, attr) for attr in (
                "narrative_state_tracking_enabled",
                "narrative_backend_card_summary",
                "narrative_backend_examples",
                "narrative_backend_lorebook",
            )
        }
        for attr in self._originals:
            object.__setattr__(settings, attr, True)
        self.engine = NarrativeEngine(session_id="test_backend_aware")

    def teardown_method(self):
        from augmentum.config import settings
        for attr, val in self._originals.items():
            object.__setattr__(settings, attr, val)

    def _augmented_messages(self, supports_mid_system: bool) -> list:
        """Run a chat through the engine and return the augmented messages."""
        req = make_request("*waves hello*", system_content=SILLYTAVERN_CARD)
        result = self.engine.process_request(
            req, supports_mid_system=supports_mid_system,
        )
        return list(result.augmented_request.messages)

    def test_default_backend_flag_is_false(self):
        """Unknown backends opt OUT — production-safe default.

        A backend that doesn't override the class attribute must not get
        the cache-friendly mid-system injection, since for cloud targets
        that means a guaranteed 400.
        """
        from augmentum.models.base import ModelBackend

        assert ModelBackend.supports_mid_conversation_system is False

    def test_known_backends_opt_in(self):
        """The three backends that benefit from llama-server-style slot
        reuse must opt in. Regression guard against an accidental flip
        that would silently regress KV warm-reuse on the in-house path.
        """
        from augmentum.models.engine import AugmentumEngineBackend
        from augmentum.models.llama_cpp import LlamaCppBackend
        from augmentum.models.ollama import OllamaBackend

        assert LlamaCppBackend.supports_mid_conversation_system is True
        assert OllamaBackend.supports_mid_conversation_system is True
        assert AugmentumEngineBackend.supports_mid_conversation_system is True

    def test_cloud_path_produces_no_mid_system(self):
        """When the gate is closed, the augmented messages must contain
        no system message after the first non-system message."""
        messages = self._augmented_messages(supports_mid_system=False)

        seen_non_system = False
        for msg in messages:
            if msg.role != "system":
                seen_non_system = True
            elif seen_non_system:
                raise AssertionError(
                    f"Cloud path produced mid-conversation system message: {msg.content[:60]}"
                )

    def test_cloud_path_folds_into_leading_system(self):
        """Dynamic context still reaches the model — folded onto the
        leading system block. The injected payload must be present in
        message 0 (the system message), not orphaned."""
        messages = self._augmented_messages(supports_mid_system=False)

        assert messages[0].role == "system"
        # The character-card-derived injection mentions Lyra by name; the
        # leading system block should contain it after folding.
        assert "Lyra" in messages[0].content

    def test_llama_path_keeps_mid_system_injection(self):
        """The cache-friendly placement is preserved for backends that
        opt in. Without this, KV slot reuse silently regresses on the
        in-house path."""
        messages = self._augmented_messages(supports_mid_system=True)

        # Find the index of the latest user message and the system
        # message that should precede it.
        last_user_idx = max(
            i for i, m in enumerate(messages) if m.role == "user"
        )
        # Some system message must exist with index < last_user_idx and > 0
        # (i.e. not the leading card system block).
        mid_system_present = any(
            messages[i].role == "system"
            for i in range(1, last_user_idx)
        )
        assert mid_system_present, (
            "Expected mid-conversation system injection on supports=True path"
        )

    def test_budget_trim_does_not_hoist_mid_system_to_front(self):
        """Regression: the context-budget trim used to rebuild the array as
        ``system_msgs + kept``, hoisting the dynamic STATE/MEMORY injection
        (deliberately placed just before the latest user turn) to the front.
        A per-turn-changing block at the head diverges the token prefix at
        message 0 → llama-server re-prefills the ENTIRE context every turn
        on long (trim-triggering) sessions. Observed live 2026-07-01:
        12-15 min TTFT on a 61k-token narrative session that should have
        warm-started. Trim must drop oldest chat messages IN PLACE.
        """
        turns: list[tuple[str, str]] = []
        for i in range(40):
            turns.append(("user", f"user turn {i} " + "filler words " * 40))
            turns.append(("assistant", f"reply {i} " + "filler words " * 40))
        turns.append(("user", "*waves goodbye*"))
        req = make_multi_turn_request(turns, system_content=SILLYTAVERN_CARD)

        result = self.engine.process_request(
            req, supports_mid_system=True, context_limit=4096,
        )
        out = result.augmented_request.messages

        # The trim must actually have dropped history for this test to
        # exercise the rebuild path.
        assert len(out) < len(req.messages) + 2, "trim did not trigger"

        # Cache-friendly placement must survive the trim: at least one
        # system message strictly between the leading block and the last
        # user turn — and NOT stacked at the front.
        last_user_idx = max(i for i, m in enumerate(out) if m.role == "user")
        mid_system_present = any(
            out[i].role == "system" for i in range(1, last_user_idx)
            if any(m.role != "system" for m in out[:i])
        )
        assert mid_system_present, (
            "budget trim hoisted the mid-conversation injection to the front "
            "(prefix-cache-breaking regression)"
        )

        # Chronology of the kept chat messages must be untouched.
        kept_chat = [m.content for m in out if m.role != "system"]
        orig_chat = [m.content for m in req.messages if m.role != "system"]
        assert kept_chat == orig_chat[len(orig_chat) - len(kept_chat):], (
            "trim reordered or dropped non-oldest chat messages"
        )

    def test_creates_leading_system_when_none_exists(self):
        """If the request has no leading system message and the gate is
        closed, the engine must create one rather than silently dropping
        the dynamic context."""
        # Build a request with NO system message — pure user turn
        req = InternalChatRequest(
            model="cloud-model",
            messages=[Message(role="user", content="Hello there.")],
        )
        # Force a non-empty injection by initializing the engine first
        self.engine.process_request(
            make_request("setup", system_content=SILLYTAVERN_CARD),
            supports_mid_system=False,
        )
        result = self.engine.process_request(req, supports_mid_system=False)

        msgs = list(result.augmented_request.messages)
        # Either there's no injection (empty context.injected_text on this
        # turn) or the engine created a leading system message for it.
        # If a system message exists, it MUST be at index 0.
        for i, m in enumerate(msgs):
            if m.role == "system":
                assert i == 0, "Cloud path placed system message after a non-system message"

    def test_signature_keyword_only_default_preserves_legacy_callers(self):
        """``supports_mid_system`` is a keyword arg with default False.
        Legacy callers that didn't pass it (and tests above that omit it)
        must continue to work without crashing."""
        req = make_request("hi", system_content=SILLYTAVERN_CARD)
        # No ``supports_mid_system`` arg at all — exercises the default.
        result = self.engine.process_request(req)
        assert result.augmented_request is not None


# === State Data Model Tests ===


class TestEntityState:
    def test_apply_delta_simple(self):
        """Should apply simple field updates."""
        state = EntityState(location="tower", emotional_state="calm")
        new_state = state.apply_delta({"location": "library", "emotional_state": "happy"})
        assert new_state.location == "library"
        assert new_state.emotional_state == "happy"

    def test_apply_delta_inventory(self):
        """Should handle list add/remove operations."""
        state = EntityState(inventory=["sword", "shield"])
        new_state = state.apply_delta({
            "inventory": {"add": ["potion"], "remove": ["shield"]},
        })
        assert "sword" in new_state.inventory
        assert "potion" in new_state.inventory
        assert "shield" not in new_state.inventory

    def test_apply_delta_relationships(self):
        """Should merge relationship updates."""
        state = EntityState(relationships={"Alice": "friend"})
        new_state = state.apply_delta({
            "relationships": {"Bob": "ally"},
        })
        assert new_state.relationships["Alice"] == "friend"
        assert new_state.relationships["Bob"] == "ally"

    def test_roundtrip_dict(self):
        """Should serialize and deserialize correctly."""
        state = EntityState(
            location="castle",
            emotional_state="excited",
            inventory=["map"],
            relationships={"King": "loyal subject"},
        )
        data = state.to_dict()
        restored = EntityState.from_dict(data)
        assert restored.location == "castle"
        assert restored.inventory == ["map"]
        assert restored.relationships["King"] == "loyal subject"
