"""Tests for character relationship tracker."""

from __future__ import annotations

import pytest

from augmentum.modes.narrative.relationship_tracker import (
    Relationship,
    RelationshipDelta,
    RelationshipTracker,
)


class TestRelationshipBasics:
    def test_empty_tracker(self):
        tracker = RelationshipTracker()
        assert tracker.relationships == []
        assert tracker.get("A", "B") is None

    def test_apply_delta_creates_relationship(self):
        tracker = RelationshipTracker()
        delta = RelationshipDelta(
            source="Lyra", target="Kael",
            trust_delta=0.5, confidence=1.0,
        )
        rel = tracker.apply_delta(delta, message_index=1)
        assert rel.source == "Lyra"
        assert rel.target == "Kael"
        assert rel.trust > 0

    def test_apply_delta_updates_existing(self):
        tracker = RelationshipTracker()
        delta1 = RelationshipDelta(
            source="A", target="B",
            trust_delta=0.5, confidence=1.0,
        )
        tracker.apply_delta(delta1, message_index=1)

        delta2 = RelationshipDelta(
            source="A", target="B",
            trust_delta=0.5, confidence=1.0,
        )
        rel = tracker.apply_delta(delta2, message_index=2)
        # Trust should have increased from both deltas
        assert rel.trust > 0.1

    def test_get_relationship(self):
        tracker = RelationshipTracker()
        delta = RelationshipDelta(
            source="A", target="B",
            affection_delta=0.3, confidence=0.8,
        )
        tracker.apply_delta(delta, message_index=1)

        rel = tracker.get("A", "B")
        assert rel is not None
        assert rel.affection > 0

        # Reverse direction doesn't exist
        assert tracker.get("B", "A") is None

    def test_get_all_for(self):
        tracker = RelationshipTracker()
        tracker.apply_delta(
            RelationshipDelta(source="A", target="B", trust_delta=0.2, confidence=0.5),
            message_index=1,
        )
        tracker.apply_delta(
            RelationshipDelta(source="C", target="A", affection_delta=0.3, confidence=0.5),
            message_index=2,
        )
        tracker.apply_delta(
            RelationshipDelta(source="B", target="C", tension_delta=0.5, confidence=0.5),
            message_index=3,
        )

        a_rels = tracker.get_all_for("A")
        assert len(a_rels) == 2  # A→B and C→A


class TestDampening:
    def test_low_confidence_dampens(self):
        tracker = RelationshipTracker()
        delta = RelationshipDelta(
            source="A", target="B",
            trust_delta=1.0, confidence=0.1,
        )
        rel = tracker.apply_delta(delta, message_index=1)
        # With low confidence and dampening, trust should be small
        assert rel.trust < 0.1

    def test_high_confidence_stronger(self):
        tracker = RelationshipTracker()
        delta = RelationshipDelta(
            source="A", target="B",
            trust_delta=1.0, confidence=1.0,
        )
        rel = tracker.apply_delta(delta, message_index=1)
        # With full confidence, should be larger
        assert rel.trust > 0.2

    def test_tension_clamped_to_positive(self):
        tracker = RelationshipTracker()
        delta = RelationshipDelta(
            source="A", target="B",
            tension_delta=-1.0, confidence=1.0,
        )
        rel = tracker.apply_delta(delta, message_index=1)
        assert rel.tension == 0.0

    def test_scores_clamped_to_range(self):
        tracker = RelationshipTracker()
        # Apply many positive deltas
        for i in range(100):
            tracker.apply_delta(
                RelationshipDelta(source="A", target="B", trust_delta=1.0, confidence=1.0),
                message_index=i,
            )
        rel = tracker.get("A", "B")
        assert rel is not None
        assert rel.trust <= 1.0


class TestSignalExtraction:
    def test_trust_positive_signal(self):
        tracker = RelationshipTracker()
        text = "*Lyra trusts Kael with her life, confiding her deepest secrets.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael"])
        assert len(deltas) == 2  # Bidirectional
        assert all(d.trust_delta > 0 for d in deltas)

    def test_affection_signal(self):
        tracker = RelationshipTracker()
        text = "*Lyra kissed Kael tenderly, embracing him warmly.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael"])
        assert len(deltas) == 2
        assert all(d.affection_delta > 0 for d in deltas)

    def test_tension_signal(self):
        tracker = RelationshipTracker()
        text = "*Lyra argued with Kael, the confrontation growing hostile.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael"])
        assert len(deltas) == 2
        assert all(d.tension_delta > 0 for d in deltas)

    def test_no_signal_when_only_one_character(self):
        tracker = RelationshipTracker()
        text = "*Lyra smiled happily.*"
        deltas = tracker.extract_signals(text, ["Lyra"])
        assert len(deltas) == 0

    def test_no_signal_when_character_not_in_text(self):
        tracker = RelationshipTracker()
        text = "*Lyra walked alone through the forest.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael"])
        assert len(deltas) == 0

    def test_mixed_signals(self):
        tracker = RelationshipTracker()
        text = "*Lyra betrayed Kael's trust, yet embraced him afterward.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael"])
        assert len(deltas) == 2
        # Should have both negative trust and positive affection
        delta = deltas[0]
        assert delta.trust_delta < 0  # betrayal
        assert delta.affection_delta > 0  # embrace

    def test_three_characters_multiple_pairs(self):
        tracker = RelationshipTracker()
        text = "*Lyra and Kael argued while Mira watched nervously.*"
        deltas = tracker.extract_signals(text, ["Lyra", "Kael", "Mira"])
        # Should detect signals between Lyra-Kael (argued) and possibly Lyra-Mira and Kael-Mira
        assert len(deltas) >= 2


class TestLLMMerge:
    def test_merge_trust_label(self):
        tracker = RelationshipTracker()
        tracker.merge_llm_relationships(
            "Lyra",
            {"Kael": "growing trust and respect"},
            message_index=5,
        )
        rel = tracker.get("Lyra", "Kael")
        assert rel is not None
        assert rel.trust > 0
        assert rel.label == "growing trust and respect"

    def test_merge_romantic_label(self):
        tracker = RelationshipTracker()
        tracker.merge_llm_relationships(
            "Lyra",
            {"Kael": "romantic tension"},
            message_index=3,
        )
        rel = tracker.get("Lyra", "Kael")
        assert rel is not None
        assert rel.affection > 0
        assert rel.tension > 0

    def test_merge_unknown_label_stores_only(self):
        tracker = RelationshipTracker()
        tracker.merge_llm_relationships(
            "A",
            {"B": "complicated feelings"},
            message_index=2,
        )
        rel = tracker.get("A", "B")
        assert rel is not None
        assert rel.label == "complicated feelings"
        # No dimension changes for unknown label
        assert rel.trust == 0.0
        assert rel.affection == 0.0
        assert rel.tension == 0.0

    def test_merge_betrayal(self):
        tracker = RelationshipTracker()
        tracker.merge_llm_relationships(
            "A",
            {"B": "betrayal and hatred"},
            message_index=4,
        )
        rel = tracker.get("A", "B")
        assert rel is not None
        assert rel.trust < 0
        assert rel.affection < 0
        assert rel.tension > 0


class TestContextSummary:
    def test_empty_summary(self):
        tracker = RelationshipTracker()
        assert tracker.get_context_summary() == ""

    def test_summary_with_relationships(self):
        tracker = RelationshipTracker()
        tracker.apply_delta(
            RelationshipDelta(
                source="Lyra", target="Kael",
                trust_delta=0.8, affection_delta=0.5,
                confidence=1.0, label="allies",
            ),
            message_index=1,
        )
        summary = tracker.get_context_summary()
        assert "Lyra" in summary
        assert "Kael" in summary
        assert "trust:" in summary

    def test_summary_skips_low_scores(self):
        tracker = RelationshipTracker()
        tracker.apply_delta(
            RelationshipDelta(
                source="A", target="B",
                trust_delta=0.01, confidence=0.1,  # Very small
            ),
            message_index=1,
        )
        # Score too low to appear
        summary = tracker.get_context_summary()
        assert summary == ""


class TestSerialization:
    def test_roundtrip(self):
        tracker = RelationshipTracker()
        tracker.apply_delta(
            RelationshipDelta(
                source="A", target="B",
                trust_delta=0.5, affection_delta=0.3, tension_delta=0.2,
                confidence=1.0, label="friends",
            ),
            message_index=5,
        )

        data = tracker.to_dict_list()
        assert len(data) == 1

        tracker2 = RelationshipTracker()
        tracker2.load_from_dict_list(data)

        rel = tracker2.get("A", "B")
        assert rel is not None
        assert rel.trust == tracker.get("A", "B").trust
        assert rel.affection == tracker.get("A", "B").affection
        assert rel.label == "friends"

    def test_load_clears_existing(self):
        tracker = RelationshipTracker()
        tracker.apply_delta(
            RelationshipDelta(source="X", target="Y", trust_delta=0.5, confidence=1.0),
            message_index=1,
        )
        tracker.load_from_dict_list([])
        assert tracker.relationships == []


class TestEngineIntegration:
    def test_relationship_extraction_in_engine(self):
        from unittest.mock import patch

        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.state.narrative_state import Entity, EntityState, EntityType, _new_id

        engine = NarrativeEngine(session_id="test")
        engine._initialized = True

        # Register two characters
        for name in ["Lyra", "Kael"]:
            entity = Entity(
                id=_new_id(),
                session_id="test",
                entity_type=EntityType.CHARACTER,
                name=name,
                state=EntityState(),
            )
            engine._state.entities[entity.id] = entity

        # State tracking must be enabled for regex extraction to fire
        with patch("augmentum.config.settings.narrative_state_tracking_enabled", True):
            engine.process_response("*Lyra trusted Kael with the secret map, embracing him warmly.*")

        # Check relationships were created
        rels = engine.relationship_tracker.relationships
        assert len(rels) >= 2  # Bidirectional

    def test_relationship_context_injected(self):
        from unittest.mock import patch

        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.modes.narrative.relationship_tracker import RelationshipDelta
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.state.narrative_state import Entity, EntityState, EntityType, _new_id

        engine = NarrativeEngine(session_id="test")
        engine._initialized = True

        # Register character
        entity = Entity(
            id=_new_id(),
            session_id="test",
            entity_type=EntityType.CHARACTER,
            name="Lyra",
            state=EntityState(),
        )
        engine._state.entities[entity.id] = entity

        # Add a strong relationship directly
        engine.relationship_tracker.apply_delta(
            RelationshipDelta(
                source="Lyra", target="Kael",
                trust_delta=1.0, affection_delta=0.8,
                confidence=1.0,
            ),
            message_index=1,
        )

        # State tracking must be enabled for relationship context injection
        with patch("augmentum.config.settings.narrative_state_tracking_enabled", True):
            # Process a request — should include relationship context
            request = InternalChatRequest(
                model="test",
                messages=[
                    Message(role="system", content="You are Lyra."),
                    Message(role="user", content="What do you think of Kael?"),
                ],
                stream=False,
            )
            result = engine.process_request(request)
            # Check that relationship summary is in the context
            assert "character_relationships" in result.context.blocks_used
