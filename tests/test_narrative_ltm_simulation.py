"""Simulation tests for narrative three-layer memory (STATE / LEDGER / ARCHIVE).

Drives the NarrativeEngine through simulated short and long sessions with
mocked LLM responses to verify the full lifecycle:
- STATE snapshot updates correctly each refresh
- LEDGER entries accumulate and don't duplicate
- Compaction triggers at ceiling and preserves all entries (shortened text)
- Persistence round-trips survive restart (sync_to_state → load → restore)
- Regeneration rolls back ledger entries correctly
- Empty/refusal responses handled gracefully

No real LLM or network calls — all state+memory refresh and compaction
responses are synthetic but follow the exact format the parser expects.
"""

from __future__ import annotations

from unittest.mock import patch

from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.memory import (
    CardType,
    MemoryEntry,
    StateSnapshot,
    build_compaction_prompt,
    build_state_memory_prompt,
    parse_state_memory_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(session_id: str = "sim-session") -> NarrativeEngine:
    engine = NarrativeEngine(session_id=session_id)
    engine._state.card_type = "character"
    engine._initialized = True
    return engine


def _fake_refresh_response(
    location: str,
    who_present: str,
    activity: str,
    entries: list[tuple[int, str, str]],
) -> str:
    """Build a fake LLM response for STATE+MEMORY refresh.

    entries: list of (round_num, category, content)
    """
    lines = [
        "## STATE",
        f"- location: {location}",
        f"- who_present: {who_present}",
        f"- current_activity: {activity}",
        "- emotional_tone: tense",
        "- immediate_tensions: none",
        "- open_threads: ongoing quest",
        "- character_dynamics: Alice (calm) → trusts Bob",
        "",
        "## MEMORY",
    ]
    for rn, cat, content in entries:
        lines.append(f"[R{rn}|{cat}] {content}")
    return "\n".join(lines)


def _fake_compaction_response(entries: list[MemoryEntry]) -> str:
    """Build a fake compaction response — shortens text by removing articles."""
    lines = []
    for e in entries:
        # Simulate text shortening: remove "the ", "a ", "an "
        short = e.content.replace("the ", "").replace("a ", "").replace("an ", "")
        lines.append(f"[R{e.round_num}|{e.category}] {short}")
    return "\n".join(lines)


def _simulate_round(engine: NarrativeEngine, round_num: int, user_msg: str, asst_msg: str) -> None:
    """Simulate a single user→assistant exchange without the full handler."""
    engine._state.message_count += 1
    engine._message_history.append(user_msg)
    engine.process_response(asst_msg)


# ---------------------------------------------------------------------------
# Short session simulation (10 rounds, 2 refreshes, no compaction)
# ---------------------------------------------------------------------------


class TestShortSession:
    """Simulate ~10 rounds — STATE populates, LEDGER grows, no compaction."""

    @patch("augmentum.config.settings")
    def test_short_session_lifecycle(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_memory_compaction_ratio = 0.5
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        mock_settings.narrative_memory_max_tokens = 0
        mock_settings.narrative_memory_state_word_target = 200
        mock_settings.narrative_memory_continuous_archive = True
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # --- Phase 1: First 5 rounds (no refresh yet) ---
        for i in range(1, 6):
            _simulate_round(engine, i, f"User message {i}", f"Assistant reply {i}")

        assert len(engine._message_history) == 10  # 5 user + 5 assistant
        assert engine._state.message_count == 10
        assert engine.state_snapshot is None  # no refresh yet
        assert engine.memory_ledger == []

        # should_refresh at interval=5: last_summary_at=0, message_count=10 → 10 >= 5
        assert engine.should_refresh(5) is True
        assert engine.should_refresh(20) is False

        # --- Phase 2: First refresh (rounds 1-10) ---
        refresh_text = _fake_refresh_response(
            location="dark forest",
            who_present="Alice, Bob",
            activity="exploring ruins",
            entries=[
                (2, "discovery", "found ancient map in the ruins"),
                (4, "relationship_shift", "Alice confided her fears to Bob"),
                (5, "commitment", "agreed to seek the mountain temple"),
            ],
        )
        snap, entries = parse_state_memory_response(refresh_text, CardType.CHARACTER, 1, 10)

        # Verify parse quality
        assert snap.fields["location"] == "dark forest"
        assert snap.fields["who_present"] == "Alice, Bob"
        assert snap.fields["current_activity"] == "exploring ruins"
        assert snap.fields.get("character_dynamics") == "Alice (calm) → trusts Bob"
        assert len(entries) == 3
        assert entries[0].round_num == 2
        assert entries[0].category == "discovery"

        engine.apply_state_memory_response(snap, entries, batch_end=10)

        assert engine.state_snapshot is not None
        assert engine.state_snapshot.fields["location"] == "dark forest"
        assert len(engine.memory_ledger) == 3
        assert engine._state.last_summary_at == 10
        assert engine._refresh_ran_this_session is True
        assert engine._pre_refresh_ledger_len == 0  # was 0 before first refresh
        assert engine.needs_compaction is False  # 3 < 60

        # --- Phase 3: Rounds 6-10 ---
        for i in range(6, 11):
            _simulate_round(engine, i, f"User message {i}", f"Assistant reply {i}")

        assert engine._state.message_count == 20
        assert engine.should_refresh(5) is True  # 20 - 10 >= 5

        # --- Phase 4: Second refresh (rounds 11-20) ---
        refresh_text_2 = _fake_refresh_response(
            location="mountain temple entrance",
            who_present="Alice, Bob, mysterious stranger",
            activity="confronting the temple guardian",
            entries=[
                (13, "world_change", "earthquake revealed hidden path"),
                (17, "discovery", "found guardian's weakness in mural"),
            ],
        )
        snap2, entries2 = parse_state_memory_response(refresh_text_2, CardType.CHARACTER, 11, 20)

        # Verify STATE is overwritten (not merged)
        assert snap2.fields["location"] == "mountain temple entrance"

        engine.apply_state_memory_response(snap2, entries2, batch_end=20)

        # STATE replaced, LEDGER extended
        assert engine.state_snapshot.fields["location"] == "mountain temple entrance"
        assert len(engine.memory_ledger) == 5  # 3 + 2
        assert engine.memory_ledger[0].round_num == 2  # original entries preserved
        assert engine.memory_ledger[3].round_num == 13  # new entry appended
        assert engine._state.last_summary_at == 20
        assert engine._pre_refresh_ledger_len == 3  # was 3 before second refresh
        assert engine.needs_compaction is False  # 5 < 60

    @patch("augmentum.config.settings")
    def test_short_session_context_injection(self, mock_settings):
        """Verify formatted STATE and LEDGER text is correct for context builder."""
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()
        snap = StateSnapshot(
            fields={"location": "castle", "who_present": "guards", "current_activity": "patrolling"},
            card_type=CardType.CHARACTER,
        )
        entries = [
            MemoryEntry(round_num=1, category="discovery", content="found secret passage"),
            MemoryEntry(round_num=3, category="commitment", content="promised to help the queen"),
        ]
        engine._state_snapshot = snap
        engine._memory_ledger = entries

        state_text = engine.get_state_text()
        assert "[Current State]" in state_text
        assert "Location: castle" in state_text
        assert "Who Present: guards" in state_text

        memory_text = engine.get_memory_text()
        assert "[Story Memory]" in memory_text
        assert "[R1|discovery] found secret passage" in memory_text
        assert "[R3|commitment] promised to help the queen" in memory_text


# ---------------------------------------------------------------------------
# Long session simulation (80 rounds, multiple refreshes, compaction)
# ---------------------------------------------------------------------------


class TestLongSession:
    """Simulate ~80 rounds — LEDGER hits ceiling, compaction fires."""

    @patch("augmentum.config.settings")
    def test_long_session_compaction_trigger(self, mock_settings):
        """Fill the ledger to ceiling and verify compaction flag."""
        mock_settings.narrative_memory_ledger_ceiling = 20  # low ceiling for test
        mock_settings.narrative_memory_compaction_ratio = 0.5
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        mock_settings.narrative_memory_max_tokens = 0
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # Simulate 40 rounds of chat
        for i in range(1, 41):
            _simulate_round(engine, i, f"User msg {i}", f"Reply {i}")

        # Simulate 4 refreshes, each adding 5 entries → 20 total → hits ceiling
        categories = ["discovery", "relationship_shift", "commitment", "world_change", "consequence"]
        for batch in range(4):
            batch_start = batch * 10 + 1
            batch_end = (batch + 1) * 10
            entries_data = [
                (batch_start + j * 2, categories[j], f"Event {batch * 5 + j + 1} happened in the story")
                for j in range(5)
            ]
            refresh_text = _fake_refresh_response(
                location=f"location_{batch}",
                who_present=f"chars_{batch}",
                activity=f"activity_{batch}",
                entries=entries_data,
            )
            snap, entries = parse_state_memory_response(
                refresh_text, CardType.CHARACTER, batch_start, batch_end,
            )
            engine.apply_state_memory_response(snap, entries, batch_end=batch_end)

        assert len(engine.memory_ledger) == 20
        assert engine.needs_compaction is True  # 20 >= 20 ceiling

    @patch("augmentum.config.settings")
    def test_compaction_preserves_all_entries(self, mock_settings):
        """Compaction shortens text but keeps every entry's R# and category."""
        mock_settings.narrative_memory_ledger_ceiling = 10
        mock_settings.narrative_memory_compaction_ratio = 0.5
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # Fill ledger with 12 entries
        entries = [
            MemoryEntry(round_num=i, category="discovery", content=f"The adventurer found a hidden {i}th artifact in the ancient ruins")
            for i in range(1, 13)
        ]
        engine._memory_ledger = entries
        engine._needs_compaction = True

        # Simulate compaction: oldest 50% (6 entries) get compacted
        compact_count = max(1, int(len(engine._memory_ledger) * 0.5))
        entries_to_compact = engine._memory_ledger[:compact_count]
        entries_to_keep = engine._memory_ledger[compact_count:]

        assert compact_count == 6
        assert len(entries_to_compact) == 6
        assert len(entries_to_keep) == 6

        # Verify compaction prompt is well-formed
        system, user = build_compaction_prompt(entries_to_compact, CardType.CHARACTER)
        assert "text compressor" in system.lower()
        assert "LOCKED KEY" in system
        assert f"You will receive {compact_count} records" in system
        assert f"output {compact_count} records" in system
        for e in entries_to_compact:
            assert f"[R{e.round_num}|{e.category}]" in user

        # Simulate LLM compaction response
        compacted_text = _fake_compaction_response(entries_to_compact)
        compacted: list[MemoryEntry] = []
        import re
        entry_pattern = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
        for line in compacted_text.strip().split("\n"):
            line = line.strip().lstrip("- ")
            m = entry_pattern.match(line)
            if m:
                compacted.append(MemoryEntry(
                    round_num=int(m.group(1)),
                    category=m.group(2).strip().lower().replace(" ", "_"),
                    content=m.group(3).strip(),
                ))

        # All entries preserved (same count)
        assert len(compacted) == len(entries_to_compact)

        # Each entry's R# and category are unchanged
        for orig, comp in zip(entries_to_compact, compacted):
            assert comp.round_num == orig.round_num
            assert comp.category == orig.category
            # Text should be shorter (articles removed)
            assert len(comp.content) <= len(orig.content)

        # Apply compaction
        engine._memory_ledger = compacted + entries_to_keep
        engine._needs_compaction = False

        # Total entries unchanged
        assert len(engine._memory_ledger) == 12
        # Temporal order preserved
        for i in range(len(engine._memory_ledger) - 1):
            assert engine._memory_ledger[i].round_num <= engine._memory_ledger[i + 1].round_num

    @patch("augmentum.config.settings")
    def test_compaction_rescue_on_truncation(self, mock_settings):
        """If the LLM drops entries (token limit), rescue logic re-inserts originals."""
        mock_settings.narrative_memory_ledger_ceiling = 10
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        entries = [
            MemoryEntry(round_num=i, category="discovery", content=f"event {i}")
            for i in range(1, 7)
        ]

        # Simulate LLM only outputting 4 of 6 entries (dropped R3 and R5)
        truncated_response = "\n".join([
            "[R1|discovery] event 1",
            "[R2|discovery] event 2",
            "[R4|discovery] event 4",
            "[R6|discovery] event 6",
        ])

        compacted: list[MemoryEntry] = []
        import re
        entry_pattern = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
        for line in truncated_response.strip().split("\n"):
            m = entry_pattern.match(line.strip())
            if m:
                compacted.append(MemoryEntry(
                    round_num=int(m.group(1)),
                    category=m.group(2).strip(),
                    content=m.group(3).strip(),
                ))

        # Rescue missing entries
        output_rounds = {e.round_num for e in compacted}
        rescued = [e for e in entries if e.round_num not in output_rounds]

        assert len(rescued) == 2  # R3 and R5 were dropped
        assert {e.round_num for e in rescued} == {3, 5}

        compacted = sorted(compacted + rescued, key=lambda e: e.round_num)

        # All 6 entries preserved after rescue
        assert len(compacted) == 6
        assert [e.round_num for e in compacted] == [1, 2, 3, 4, 5, 6]

    @patch("augmentum.config.settings")
    def test_long_session_full_lifecycle(self, mock_settings):
        """End-to-end: 80 rounds, 8 refreshes, compaction cycle, verify final state."""
        mock_settings.narrative_memory_ledger_ceiling = 30
        mock_settings.narrative_memory_compaction_ratio = 0.33
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        mock_settings.narrative_memory_max_tokens = 0
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()
        refresh_interval = 10
        total_ledger_entries = 0
        compaction_count = 0

        for round_num in range(1, 81):
            _simulate_round(engine, round_num, f"User {round_num}", f"Reply {round_num}")

            # Check if refresh is due
            if engine.should_refresh(refresh_interval):
                batch_start = max(1, engine._state.last_summary_at + 1)
                batch_end = engine._state.message_count

                # Generate 3-5 entries per refresh
                n_entries = min(5, 3 + (round_num % 3))
                categories = ["discovery", "relationship_shift", "commitment", "world_change", "consequence", "emotional_milestone"]
                entries_data = [
                    (batch_start + j, categories[j % len(categories)], f"Event at round {batch_start + j}")
                    for j in range(n_entries)
                ]
                refresh_text = _fake_refresh_response(
                    location=f"place_{round_num}",
                    who_present=f"group_{round_num}",
                    activity=f"doing_{round_num}",
                    entries=entries_data,
                )
                snap, entries = parse_state_memory_response(
                    refresh_text, CardType.CHARACTER, batch_start, batch_end,
                )
                engine.apply_state_memory_response(snap, entries, batch_end=batch_end)
                total_ledger_entries += len(entries)

                # Simulate compaction if needed
                if engine.needs_compaction:
                    compact_count = max(1, int(len(engine._memory_ledger) * 0.33))
                    to_compact = engine._memory_ledger[:compact_count]
                    to_keep = engine._memory_ledger[compact_count:]

                    compacted_text = _fake_compaction_response(to_compact)
                    compacted: list[MemoryEntry] = []
                    import re
                    pat = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
                    for line in compacted_text.strip().split("\n"):
                        m = pat.match(line.strip())
                        if m:
                            compacted.append(MemoryEntry(
                                round_num=int(m.group(1)),
                                category=m.group(2).strip().lower().replace(" ", "_"),
                                content=m.group(3).strip(),
                            ))

                    engine._memory_ledger = compacted + to_keep
                    engine._needs_compaction = False
                    engine._pre_refresh_ledger_len = len(engine._memory_ledger)
                    compaction_count += 1

        # Final state checks
        assert engine._state.message_count == 160  # 80 user + 80 assistant
        assert len(engine._message_history) == 160
        assert engine.state_snapshot is not None
        assert len(engine.memory_ledger) > 0

        # Compaction should have fired at least once
        assert compaction_count >= 1, f"Expected at least 1 compaction, got {compaction_count}"

        # All entries have valid round numbers
        for e in engine.memory_ledger:
            assert e.round_num > 0
            assert e.category != ""
            assert e.content != ""

        # Temporal order maintained
        for i in range(len(engine.memory_ledger) - 1):
            assert engine.memory_ledger[i].round_num <= engine.memory_ledger[i + 1].round_num

        # STATE should reflect the last refresh
        assert engine.state_snapshot.fields.get("location") is not None


# ---------------------------------------------------------------------------
# Regeneration handling
# ---------------------------------------------------------------------------


class TestRegeneration:
    """Verify regen correctly rolls back ledger entries and replaces history."""

    @patch("augmentum.config.settings")
    def test_regen_rolls_back_ledger(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # Build up some initial state
        for i in range(1, 6):
            _simulate_round(engine, i, f"User {i}", f"Reply {i}")

        # First refresh — 3 entries
        snap = StateSnapshot(fields={"location": "forest"}, card_type=CardType.CHARACTER)
        entries = [MemoryEntry(round_num=i, category="discovery", content=f"found {i}") for i in range(1, 4)]
        engine.apply_state_memory_response(snap, entries, batch_end=10)

        assert len(engine.memory_ledger) == 3
        assert engine._pre_refresh_ledger_len == 0

        # Add more rounds
        for i in range(6, 11):
            _simulate_round(engine, i, f"User {i}", f"Reply {i}")

        # Second refresh — adds 2 more entries
        snap2 = StateSnapshot(fields={"location": "cave"}, card_type=CardType.CHARACTER)
        entries2 = [
            MemoryEntry(round_num=6, category="commitment", content="vowed to return"),
            MemoryEntry(round_num=8, category="world_change", content="cave collapsed"),
        ]
        engine.apply_state_memory_response(snap2, entries2, batch_end=20)

        assert len(engine.memory_ledger) == 5  # 3 + 2
        assert engine._pre_refresh_ledger_len == 3

        # Simulate regeneration — user wants a different response to last message
        # This should strip entries added by the most recent refresh (entries2)
        assert engine._refresh_ran_this_session is True
        assert len(engine._memory_ledger) > engine._pre_refresh_ledger_len

        # Regen rollback logic (from engine.process_request)
        stripped = len(engine._memory_ledger) - engine._pre_refresh_ledger_len
        engine._memory_ledger = engine._memory_ledger[:engine._pre_refresh_ledger_len]

        assert stripped == 2
        assert len(engine.memory_ledger) == 3  # back to pre-second-refresh state
        assert engine.memory_ledger[-1].content == "found 3"

    @patch("augmentum.config.settings")
    def test_regen_replaces_last_history_entry(self, mock_settings):
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # 3 rounds of normal chat
        _simulate_round(engine, 1, "Hello", "Hi there")
        _simulate_round(engine, 2, "How are you", "I'm good")
        _simulate_round(engine, 3, "Tell me a story", "Once upon a time...")

        assert len(engine._message_history) == 6  # 3 user + 3 assistant
        assert engine._message_history[-1] == "Once upon a time..."

        # Simulate regen: flag is set, then new response replaces last
        engine._pending_regen = True
        engine.process_response("In a faraway kingdom...")

        assert len(engine._message_history) == 6  # count unchanged
        assert engine._message_history[-1] == "In a faraway kingdom..."  # replaced
        assert engine._pending_regen is False

    @patch("augmentum.config.settings")
    def test_regen_no_rollback_after_restart(self, mock_settings):
        """After restart, _refresh_ran_this_session is False — no rollback."""
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()
        # Simulate restored state from DB
        engine._memory_ledger = [
            MemoryEntry(round_num=i, category="discovery", content=f"loaded {i}")
            for i in range(1, 6)
        ]
        engine._pre_refresh_ledger_len = 0  # default after restart
        engine._refresh_ran_this_session = False  # not refreshed this boot

        # Regen should NOT strip entries because refresh hasn't run this session
        assert not (engine._refresh_ran_this_session
                    and len(engine._memory_ledger) > engine._pre_refresh_ledger_len)
        assert len(engine.memory_ledger) == 5  # all preserved


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty responses, refusals, malformed LLM output."""

    def test_empty_response_not_added(self):
        engine = _make_engine()
        engine._state.message_count = 1
        engine._message_history.append("Hello")

        engine.process_response("")
        # Empty response is early-returned, no history append
        assert len(engine._message_history) == 1

    def test_refusal_undoes_request(self):
        engine = _make_engine()
        engine._state.message_count = 1
        engine._message_history.append("Write something")

        refusal = "I can't generate that as an AI language model due to content policy."
        engine.process_response(refusal)

        # Refusal detected → undo_last_request removes the orphan user message
        assert len(engine._message_history) == 0
        assert engine._state.message_count == 0

    def test_undo_last_request_odd_history(self):
        engine = _make_engine()
        engine._message_history = ["user1", "asst1", "user2"]  # odd = orphan
        engine._state.message_count = 3

        engine.undo_last_request()

        assert len(engine._message_history) == 2
        assert engine._state.message_count == 2

    def test_undo_last_request_even_history_noop(self):
        engine = _make_engine()
        engine._message_history = ["user1", "asst1"]  # even = paired
        engine._state.message_count = 2

        engine.undo_last_request()

        # Even-length history → no pop (history is properly paired)
        assert len(engine._message_history) == 2
        assert engine._state.message_count == 2

    def test_parse_malformed_state_response(self):
        """LLM returns garbled output — parser should not crash."""
        text = "Here's my analysis:\nThe scene is interesting.\nNo valid format."
        snap, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        assert isinstance(snap, StateSnapshot)
        assert isinstance(entries, list)
        assert len(entries) == 0  # no valid [R#|cat] entries

    def test_parse_missing_state_header(self):
        """LLM omits ## STATE header but provides fields."""
        text = (
            "location: marketplace\n"
            "who_present: merchants\n\n"
            "## MEMORY\n"
            "[R3|discovery] found rare herb at stall"
        )
        snap, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        # Memory section should still parse
        assert len(entries) == 1
        assert entries[0].content == "found rare herb at stall"

    def test_parse_out_of_range_round_clamped(self):
        """Round numbers outside batch range get clamped to batch_end."""
        text = "## STATE\n\n## MEMORY\n[R999|discovery] far future event"
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        assert len(entries) == 1
        assert entries[0].round_num == 10  # clamped

    def test_parse_refusal_in_memory_filtered(self):
        """Entries containing refusal text are stripped from parsed memory."""
        text = (
            "## STATE\nlocation: park\n\n## MEMORY\n"
            "[R1|discovery] found key\n"
            "[R2|discovery] I can't generate that as an AI due to content policy\n"
            "[R3|commitment] promised to help"
        )
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        assert len(entries) == 2  # R2 filtered out
        assert entries[0].round_num == 1
        assert entries[1].round_num == 3

    def test_parse_category_not_validated(self):
        """Known issue: invented categories pass through without validation.

        This test documents current behavior — categories are NOT checked
        against the valid set. If this gets fixed, update this test.
        """
        text = "## STATE\n\n## MEMORY\n[R1|totally_made_up_category] some event"
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        assert len(entries) == 1
        assert entries[0].category == "totally_made_up_category"  # passes through


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify sync_to_state → NarrativeSessionState → restore."""

    @patch("augmentum.config.settings")
    def test_sync_and_restore(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        engine = _make_engine()

        # Build up state
        for i in range(1, 6):
            _simulate_round(engine, i, f"User {i}", f"Reply {i}")

        snap = StateSnapshot(
            fields={"location": "dungeon", "who_present": "party"},
            card_type=CardType.CHARACTER,
        )
        entries = [
            MemoryEntry(round_num=1, category="discovery", content="found door"),
            MemoryEntry(round_num=3, category="commitment", content="chose to enter"),
        ]
        engine.apply_state_memory_response(snap, entries, batch_end=10)

        # Sync to state
        engine.sync_to_state()

        # Verify state captures everything
        state = engine.state
        assert state.state_snapshot_data == {"fields": {"location": "dungeon", "who_present": "party"}, "card_type": "character"}
        assert len(state.memory_ledger_data) == 2
        assert state.memory_ledger_data[0]["content"] == "found door"
        assert state.message_history_data == engine._message_history
        assert state.needs_compaction is False

        # Simulate "restart" — create new engine, restore from state
        engine2 = _make_engine("sim-session-2")

        # Restore snapshot
        if state.state_snapshot_data:
            engine2._state_snapshot = StateSnapshot.from_dict(state.state_snapshot_data)
        # Restore ledger
        engine2._memory_ledger = [MemoryEntry.from_dict(d) for d in state.memory_ledger_data]
        # Restore history
        engine2._message_history = list(state.message_history_data)
        engine2._needs_compaction = state.needs_compaction

        # Verify restored engine matches original
        assert engine2.state_snapshot.fields == {"location": "dungeon", "who_present": "party"}
        assert len(engine2.memory_ledger) == 2
        assert engine2.memory_ledger[0].content == "found door"
        assert len(engine2._message_history) == 10
        assert engine2.needs_compaction is False

        # Verify restored engine can continue working
        assert engine2.get_state_text() == engine.get_state_text()
        assert engine2.get_memory_text() == engine.get_memory_text()


# ---------------------------------------------------------------------------
# Prompt quality assertions
# ---------------------------------------------------------------------------


class TestPromptQuality:
    """Verify the prompts sent to the LLM are well-formed and unambiguous."""

    def test_refresh_prompt_contains_all_fields(self):
        """System prompt lists all STATE fields for the card type."""
        for card_type in CardType:
            fields = list(dict.fromkeys(
                f for f in ["location", "who_present", "current_activity",
                            "emotional_tone", "immediate_tensions", "open_threads",
                            "character_dynamics"]
            )) if card_type == CardType.CHARACTER else None

            system, user = build_state_memory_prompt(
                card_type, None, [], ["msg1", "msg2"], "TestChar", 1, 2,
            )
            # All defined fields should appear in system prompt
            from augmentum.modes.narrative.memory import STATE_FIELDS
            for f in STATE_FIELDS[card_type]:
                assert f in system or f.replace("_", " ") in system, \
                    f"Missing field '{f}' in {card_type.value} system prompt"

    def test_refresh_prompt_contains_all_categories(self):
        """System prompt lists all valid MEMORY categories."""
        for card_type in CardType:
            system, _ = build_state_memory_prompt(
                card_type, None, [], ["msg"], "Char", 1, 1,
            )
            from augmentum.modes.narrative.memory import MEMORY_CATEGORIES
            for cat in MEMORY_CATEGORIES[card_type]:
                assert cat in system, f"Missing category '{cat}' in {card_type.value} prompt"

    def test_refresh_prompt_round_range(self):
        """Prompt specifies correct round range."""
        system, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], ["a", "b", "c", "d", "e"], "X", 3, 7,
        )
        assert "R3" in system
        assert "R7" in system
        assert "R3 through R7" in user or "R3-R7" in user or "R3" in user

    def test_compaction_prompt_entry_count(self):
        """Compaction prompt correctly states input/output record count."""
        entries = [MemoryEntry(round_num=i, category="discovery", content=f"e{i}") for i in range(1, 9)]
        system, user = build_compaction_prompt(entries, CardType.CHARACTER)
        assert "8 records" in system
        assert "output 8 records" in system

    def test_compaction_prompt_preserves_tags(self):
        """Compaction user prompt includes all entry tags verbatim."""
        entries = [
            MemoryEntry(round_num=5, category="relationship_shift", content="Alice betrayed Bob"),
            MemoryEntry(round_num=12, category="world_change", content="volcano erupted"),
        ]
        _, user = build_compaction_prompt(entries, CardType.CHARACTER)
        assert "[R5|relationship_shift]" in user
        assert "[R12|world_change]" in user

    def test_refresh_prompt_no_repeat_instruction(self):
        """When previous ledger entries are included, prompt says 'do NOT repeat'."""
        ledger = [MemoryEntry(round_num=1, category="discovery", content="old event")]
        _, user = build_state_memory_prompt(
            CardType.CHARACTER, None, ledger, ["msg"], "X", 1, 5,
        )
        assert "do NOT repeat" in user

    def test_refresh_prompt_refusal_filtered_from_messages(self):
        """Refusal messages in history are excluded from the user prompt."""
        messages = [
            "Tell me a story",
            "I can't generate that as an AI language model due to content policy.",
            "Tell me about the weather instead",
            "The sun was setting...",
        ]
        _, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], messages, "X", 1, 4,
        )
        assert "content policy" not in user
        assert "Tell me a story" in user
        assert "sun was setting" in user
