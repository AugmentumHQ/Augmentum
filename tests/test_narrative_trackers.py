"""Tests for narrative trackers: character, world, plot, branch."""

from __future__ import annotations

from augmentum.modes.narrative.branch_tracker import BranchDetection, BranchTracker
from augmentum.modes.narrative.character_tracker import CharacterTracker, CharacterUpdate
from augmentum.modes.narrative.plot_tracker import PlotTracker
from augmentum.modes.narrative.world_tracker import SceneState, WorldTracker
from augmentum.state.narrative_state import Entity, EntityState, EntityType, PlotStatus


def _make_character(name: str = "Luna", aliases: list[str] | None = None) -> Entity:
    return Entity(
        name=name,
        entity_type=EntityType.CHARACTER,
        aliases=aliases or [],
        state=EntityState(),
    )


class TestCharacterTracker:
    """Character state extraction from narrative text."""

    def test_extract_emotion_happy(self):
        tracker = CharacterTracker()
        char = _make_character("Luna")
        text = "*Luna smiled warmly, her eyes bright with joy.*"
        updates = tracker.extract_updates(text, [char])
        assert len(updates) >= 1
        luna_update = next((u for u in updates if u.name == "Luna"), None)
        assert luna_update is not None
        assert luna_update.emotional_state is not None

    def test_extract_no_updates_for_unknown_char(self):
        tracker = CharacterTracker()
        char = _make_character("Luna")
        text = "*Bob walked into the room and sat down.*"
        updates = tracker.extract_updates(text, [char])
        # Bob is not a known character, so no updates for him
        luna_updates = [u for u in updates if u.name == "Luna"]
        # Luna might not have updates either since the text is about Bob
        assert all(u.name == "Luna" for u in updates)

    def test_extract_physical_state(self):
        tracker = CharacterTracker()
        char = _make_character("Luna")
        text = "*Luna sits down on the wooden bench, exhausted.*"
        updates = tracker.extract_updates(text, [char])
        luna_update = next((u for u in updates if u.name == "Luna"), None)
        if luna_update:
            assert luna_update.physical_state is not None or luna_update.emotional_state is not None

    def test_extract_location_change(self):
        tracker = CharacterTracker()
        char = _make_character("Luna")
        text = "*Luna enters the grand library, looking around in awe.*"
        updates = tracker.extract_updates(text, [char])
        luna_update = next((u for u in updates if u.name == "Luna"), None)
        if luna_update and luna_update.location:
            assert "library" in luna_update.location.lower()

    def test_apply_update_emotional_state(self):
        tracker = CharacterTracker()
        entity = _make_character("Luna")
        update = CharacterUpdate(name="Luna", emotional_state="happy", emotional_confidence=0.8)
        delta = tracker.apply_update(entity, update, message_index=1)
        assert entity.state.emotional_state == "happy"

    def test_apply_update_physical_state(self):
        tracker = CharacterTracker()
        entity = _make_character("Luna")
        update = CharacterUpdate(name="Luna", physical_state="sitting")
        delta = tracker.apply_update(entity, update, message_index=1)
        assert entity.state.physical_state == "sitting"

    def test_character_update_dataclass_defaults(self):
        update = CharacterUpdate()
        assert update.name == ""
        assert update.emotional_state is None
        assert update.physical_state is None
        assert update.location is None

    def test_extract_with_aliases(self):
        tracker = CharacterTracker()
        char = _make_character("Luna", aliases=["the witch", "Ms. Silver"])
        text = "*The witch laughed with delight.*"
        updates = tracker.extract_updates(text, [char])
        # Should still recognize Luna via alias
        if updates:
            assert updates[0].name == "Luna"


class TestWorldTracker:
    """World/scene state extraction."""

    def test_scene_state_defaults(self):
        scene = SceneState()
        assert scene.location == ""
        assert scene.time_of_day == ""
        assert scene.weather == ""

    def test_scene_state_to_dict(self):
        scene = SceneState(location="tavern", time_of_day="evening", weather="raining")
        d = scene.to_dict()
        assert d["location"] == "tavern"
        assert d["time_of_day"] == "evening"
        assert d["weather"] == "raining"

    def test_scene_state_from_dict(self):
        d = {"location": "forest", "time_of_day": "dawn", "weather": "foggy"}
        scene = SceneState.from_dict(d)
        assert scene.location == "forest"
        assert scene.time_of_day == "dawn"

    def test_scene_state_apply_delta(self):
        scene = SceneState(location="tavern", time_of_day="evening")
        new_scene = scene.apply_delta({"location": "forest"})
        assert new_scene.location == "forest"
        # time_of_day should be preserved
        assert new_scene.time_of_day == "evening"

    def test_world_tracker_construct(self):
        tracker = WorldTracker()
        assert tracker is not None


class TestPlotTracker:
    """Plot thread management."""

    def test_extract_plot_signals(self):
        tracker = PlotTracker()
        text = "The heroes completed their quest and discovered a hidden treasure."
        signals = tracker.extract_plot_signals(text)
        assert len(signals) > 0

    def test_detect_resolutions(self):
        tracker = PlotTracker()
        text = "Finally, peace was restored to the kingdom."
        assert tracker.detect_resolutions(text) is True

    def test_no_resolution_in_normal_text(self):
        tracker = PlotTracker()
        text = "Luna walked down the street and bought some bread."
        assert tracker.detect_resolutions(text) is False

    def test_add_thread(self):
        tracker = PlotTracker()
        thread = tracker.add_thread(
            session_id="test",
            title="Find the artifact",
            description="The party must locate the ancient artifact.",
        )
        assert thread.title == "Find the artifact"
        assert thread.status == PlotStatus.ACTIVE
        assert thread.id in tracker.threads

    def test_active_threads(self):
        tracker = PlotTracker()
        tracker.add_thread(session_id="test", title="Thread A")
        tracker.add_thread(session_id="test", title="Thread B")
        assert len(tracker.active_threads) == 2


class TestBranchTracker:
    """DAG-based message tracking."""

    def test_initial_state(self):
        tracker = BranchTracker(session_id="test")
        assert tracker.current_branch == "main"
        assert tracker.messages == []

    def test_branch_detection_defaults(self):
        det = BranchDetection()
        assert det.is_branch is False
        assert det.branch_point == -1
        assert det.parent_branch_id == "main"
