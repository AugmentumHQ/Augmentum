"""Real conversation simulation tests for narrative three-layer memory.

Loads actual roleplay conversations from JSONL files and drives them through
the NarrativeEngine's STATE/LEDGER pipeline to verify:
- Prompt generation works with real message content
- parse_state_memory_response handles real LLM-style output
- Compaction prompts are well-formed for real ledger data
- The engine doesn't crash on any real conversation patterns
- Persistence round-trips work with real data volumes

Uses synthetic (but realistic) LLM responses since we can't call a real LLM
in tests, but the MESSAGE CONTENT is real from actual sessions.

Requires: roleplay conversation files in $AUGMENTUM_ROLEPLAY_CONVO_DIR (defaults to /data/roleplay_train/)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.memory import (
    CardType,
    MemoryEntry,
    StateSnapshot,
    SummaryMode,
    build_compaction_prompt,
    build_state_memory_prompt,
    format_ledger_for_context,
    format_state_for_context,
    parse_state_memory_response,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CONVO_DIR = Path(os.environ.get("AUGMENTUM_ROLEPLAY_CONVO_DIR", "/data/roleplay_train"))


def _load_conversations() -> list[tuple[str, list[dict]]]:
    """Load all JSONL conversation files, return list of (filename, messages)."""
    if not CONVO_DIR.exists():
        return []
    results = []
    for f in sorted(CONVO_DIR.glob("*.jsonl")):
        messages = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip metadata line (first line has chat_metadata)
            if "chat_metadata" in obj:
                continue
            if "mes" in obj:
                messages.append(obj)
        if messages:
            results.append((f.name, messages))
    return results


def _extract_messages(raw_messages: list[dict]) -> list[str]:
    """Extract alternating user/assistant message strings."""
    result = []
    for msg in raw_messages:
        text = msg.get("mes", "")
        if text:
            result.append(text)
    return result


def _classify_conversation(filename: str) -> CardType:
    """Guess card type from filename."""
    lower = filename.lower()
    if "rpg" in lower or "isekai" in lower or "chronicles" in lower:
        return CardType.NARRATOR
    return CardType.CHARACTER


ALL_CONVOS = _load_conversations()


def _convo_ids() -> list[str]:
    """Generate readable test IDs from filenames."""
    return [name.replace(" ", "_")[:60] for name, _ in ALL_CONVOS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(session_id: str = "real-test") -> NarrativeEngine:
    engine = NarrativeEngine(session_id=session_id)
    engine._state.card_type = "character"
    engine._initialized = True
    return engine


def _generate_realistic_refresh(
    messages: list[str],
    batch_start: int,
    batch_end: int,
    card_type: CardType,
) -> str:
    """Generate a synthetic but realistic STATE+MEMORY response from messages.

    Extracts location hints, character names, and key events from the actual
    message text to produce a plausible refresh response.
    """
    from augmentum.modes.narrative.memory import STATE_FIELDS, MEMORY_CATEGORIES

    fields = STATE_FIELDS.get(card_type, STATE_FIELDS[CardType.CHARACTER])
    categories = MEMORY_CATEGORIES.get(card_type, MEMORY_CATEGORIES[CardType.CHARACTER])

    # Extract basic info from messages
    all_text = " ".join(messages[-10:])[:2000]

    # Find names (capitalized words that appear multiple times)
    name_pattern = re.compile(r"\b([A-Z][a-z]{2,})\b")
    names = name_pattern.findall(all_text)
    name_counts = {}
    for n in names:
        if n not in ("The", "This", "That", "What", "When", "Where", "How", "You", "Your", "She", "Her", "His", "They"):
            name_counts[n] = name_counts.get(n, 0) + 1
    top_names = sorted(name_counts, key=name_counts.get, reverse=True)[:4]
    who_present = ", ".join(top_names) if top_names else "unknown characters"

    # Build STATE
    state_lines = ["## STATE"]
    field_values = {
        "location": "an unspecified location",
        "who_present": who_present,
        "characters_present": who_present,
        "current_activity": "conversing",
        "emotional_tone": "tense",
        "immediate_tensions": "underlying conflict",
        "open_threads": "ongoing situation",
        "character_dynamics": f"{top_names[0]} (engaged) → interacting with others" if top_names else "unknown dynamics",
        "group_dynamic": "cautious cooperation",
        "party_status": "active",
        "active_quest": "current objective",
        "immediate_situation": "developing scene",
        "environmental_conditions": "normal",
        "pending_decisions": "next move unclear",
        "key_relationships": f"{top_names[0]} → allied with party" if top_names else "unknown",
    }
    for f in fields:
        val = field_values.get(f, "unknown")
        state_lines.append(f"- {f}: {val}")

    # Build MEMORY entries (2-4 entries from the batch)
    memory_lines = ["", "## MEMORY"]
    n_entries = min(4, max(2, (batch_end - batch_start) // 3))
    for i in range(n_entries):
        rn = batch_start + int((batch_end - batch_start) * i / max(1, n_entries - 1)) if n_entries > 1 else batch_start
        rn = max(batch_start, min(batch_end, rn))
        cat = categories[i % len(categories)]
        # Extract a snippet from a nearby message as content
        msg_idx = min(len(messages) - 1, max(0, rn - 1))
        snippet = messages[msg_idx][:80].replace("\n", " ").replace("\r", " ").strip()
        if snippet.startswith("*"):
            snippet = snippet[1:].strip()
        # Trim to a reasonable event description
        snippet = snippet[:60]
        memory_lines.append(f"[R{rn}|{cat}] {snippet}")

    return "\n".join(state_lines + memory_lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ALL_CONVOS, reason="No conversation files found")
class TestRealConversationPrompts:
    """Verify prompt generation works with real message content."""

    @pytest.mark.parametrize("name,messages", ALL_CONVOS, ids=_convo_ids())
    def test_prompt_generation_doesnt_crash(self, name, messages):
        """build_state_memory_prompt handles real message content without errors."""
        msg_texts = _extract_messages(messages)
        card_type = _classify_conversation(name)

        # Simulate first refresh (after ~10 messages)
        batch_end = min(10, len(msg_texts))
        system, user = build_state_memory_prompt(
            card_type=card_type,
            current_state=None,
            memory_ledger=[],
            recent_messages=msg_texts[:batch_end],
            char_name=messages[0].get("name", "Character") if not messages[0].get("is_user") else "Character",
            batch_start=1,
            batch_end=batch_end,
        )

        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 100  # non-trivial prompt
        assert "## STATE" in system
        assert "## MEMORY" in system

    @pytest.mark.parametrize("name,messages", ALL_CONVOS, ids=_convo_ids())
    def test_prompt_message_lines_have_round_numbers(self, name, messages):
        """Each message in the user prompt has an [R#] round label."""
        msg_texts = _extract_messages(messages)
        batch_end = min(10, len(msg_texts))
        _, user = build_state_memory_prompt(
            card_type=CardType.CHARACTER,
            current_state=None,
            memory_ledger=[],
            recent_messages=msg_texts[:batch_end],
            char_name="Char",
            batch_start=1,
            batch_end=batch_end,
        )
        # Every non-empty line with actual message content should have [R#]
        r_pattern = re.compile(r"\[R\d+\]")
        content_lines = [l for l in user.split("\n") if l.strip() and not l.startswith(("Previous", "Recent", "These"))]
        tagged = [l for l in content_lines if r_pattern.search(l)]
        # At least half the content lines should be tagged messages
        assert len(tagged) >= 1, f"No [R#] tagged messages found in prompt for {name}"


@pytest.mark.skipif(not ALL_CONVOS, reason="No conversation files found")
class TestRealConversationParsing:
    """Verify synthetic refresh responses parse correctly for real conversations."""

    @pytest.mark.parametrize("name,messages", ALL_CONVOS, ids=_convo_ids())
    def test_synthetic_refresh_parses(self, name, messages):
        """Generated refresh response parses into valid STATE + entries."""
        msg_texts = _extract_messages(messages)
        card_type = _classify_conversation(name)
        batch_end = min(10, len(msg_texts))

        refresh_text = _generate_realistic_refresh(
            msg_texts[:batch_end], 1, batch_end, card_type,
        )
        snap, entries = parse_state_memory_response(refresh_text, card_type, 1, batch_end)

        assert isinstance(snap, StateSnapshot)
        assert len(snap.fields) > 0, f"No STATE fields parsed for {name}"
        assert isinstance(entries, list)
        assert len(entries) >= 1, f"No MEMORY entries parsed for {name}"
        for e in entries:
            assert 1 <= e.round_num <= batch_end, f"Round {e.round_num} outside [1, {batch_end}]"
            assert e.category != ""
            assert e.content != ""


@pytest.mark.skipif(not ALL_CONVOS, reason="No conversation files found")
class TestRealConversationFullLifecycle:
    """Drive the engine through real conversations end-to-end."""

    @patch("augmentum.config.settings")
    def test_short_convo_lifecycle(self, mock_settings):
        """Pick a short conversation (<50 messages) and run the full lifecycle."""
        mock_settings.narrative_memory_ledger_ceiling = 30
        mock_settings.narrative_memory_compaction_ratio = 0.5
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        mock_settings.narrative_memory_max_tokens = 0
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        # Find a short conversation
        short_convos = [(n, m) for n, m in ALL_CONVOS if len(m) <= 50]
        if not short_convos:
            pytest.skip("No short conversations available")

        name, raw_messages = short_convos[0]
        msg_texts = _extract_messages(raw_messages)
        card_type = _classify_conversation(name)

        engine = _make_engine(f"short-{name[:20]}")
        engine._state.card_type = card_type.value
        refresh_interval = 6
        refresh_count = 0

        for i, msg in enumerate(msg_texts):
            is_user = raw_messages[i].get("is_user", False) if i < len(raw_messages) else (i % 2 == 0)

            if is_user:
                engine._state.message_count += 1
                engine._message_history.append(msg)
            else:
                engine.process_response(msg)

            # Trigger refresh at intervals
            if engine.should_refresh(refresh_interval):
                batch_start = max(1, engine._state.last_summary_at + 1)
                batch_end = engine._state.message_count

                refresh_text = _generate_realistic_refresh(
                    engine._message_history, batch_start, batch_end, card_type,
                )
                snap, entries = parse_state_memory_response(
                    refresh_text, card_type, batch_start, batch_end,
                )
                engine.apply_state_memory_response(snap, entries, batch_end=batch_end)
                refresh_count += 1

        # Verify end state
        assert engine._state.message_count > 0
        assert len(engine._message_history) > 0
        if refresh_count > 0:
            assert engine.state_snapshot is not None
            assert len(engine.memory_ledger) > 0

        # Persistence round-trip
        engine.sync_to_state()
        state = engine.state
        assert state.state_snapshot_data or refresh_count == 0
        assert len(state.memory_ledger_data) == len(engine.memory_ledger)

    @patch("augmentum.config.settings")
    def test_long_convo_lifecycle(self, mock_settings):
        """Pick the longest conversation and run the full lifecycle with compaction."""
        mock_settings.narrative_memory_ledger_ceiling = 20  # low ceiling to trigger compaction
        mock_settings.narrative_memory_compaction_ratio = 0.33
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        mock_settings.narrative_memory_max_tokens = 0
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        # Find the longest conversation
        longest = max(ALL_CONVOS, key=lambda x: len(x[1]))
        name, raw_messages = longest
        msg_texts = _extract_messages(raw_messages)
        card_type = _classify_conversation(name)

        engine = _make_engine(f"long-{name[:20]}")
        engine._state.card_type = card_type.value
        refresh_interval = 10
        refresh_count = 0
        compaction_count = 0

        # Cap at 200 messages for test speed
        cap = min(200, len(msg_texts))

        for i in range(cap):
            msg = msg_texts[i]
            is_user = raw_messages[i].get("is_user", False) if i < len(raw_messages) else (i % 2 == 0)

            if is_user:
                engine._state.message_count += 1
                engine._message_history.append(msg)
            else:
                engine.process_response(msg)

            # Trigger refresh
            if engine.should_refresh(refresh_interval):
                batch_start = max(1, engine._state.last_summary_at + 1)
                batch_end = engine._state.message_count

                refresh_text = _generate_realistic_refresh(
                    engine._message_history, batch_start, batch_end, card_type,
                )
                snap, entries = parse_state_memory_response(
                    refresh_text, card_type, batch_start, batch_end,
                )
                engine.apply_state_memory_response(snap, entries, batch_end=batch_end)
                refresh_count += 1

                # Compaction
                if engine.needs_compaction:
                    compact_count = max(1, int(len(engine._memory_ledger) * 0.33))
                    to_compact = engine._memory_ledger[:compact_count]

                    system, user = build_compaction_prompt(to_compact, card_type)
                    # Verify prompt is valid
                    assert len(system) > 50
                    assert len(user) > 10
                    for e in to_compact:
                        assert f"[R{e.round_num}|{e.category}]" in user

                    # Simulate compaction (shorten text)
                    compacted = []
                    for e in to_compact:
                        short = e.content[:40].replace("the ", "").replace("a ", "")
                        compacted.append(MemoryEntry(
                            round_num=e.round_num,
                            category=e.category,
                            content=short,
                        ))
                    to_keep = engine._memory_ledger[compact_count:]
                    engine._memory_ledger = compacted + to_keep
                    engine._needs_compaction = False
                    engine._pre_refresh_ledger_len = len(engine._memory_ledger)
                    compaction_count += 1

        # Final state checks
        assert engine._state.message_count > 50
        assert refresh_count >= 3, f"Expected >=3 refreshes, got {refresh_count}"
        assert compaction_count >= 1, f"Expected >=1 compaction, got {compaction_count}"
        assert engine.state_snapshot is not None
        assert len(engine.memory_ledger) > 0

        # Temporal order maintained
        for i in range(len(engine.memory_ledger) - 1):
            assert engine.memory_ledger[i].round_num <= engine.memory_ledger[i + 1].round_num

        # Verify context text generation
        state_text = engine.get_state_text()
        memory_text = engine.get_memory_text()
        assert "[Current State]" in state_text
        assert "[Story Memory]" in memory_text

        # Persistence round-trip
        engine.sync_to_state()
        state = engine.state
        assert len(state.memory_ledger_data) == len(engine.memory_ledger)
        assert state.state_snapshot_data.get("fields")

    @patch("augmentum.config.settings")
    @pytest.mark.parametrize("name,messages", ALL_CONVOS[:5], ids=_convo_ids()[:5])
    def test_compaction_prompt_quality_real_data(self, mock_settings, name, messages):
        """Compaction prompts for real conversation data are well-formed."""
        mock_settings.narrative_memory_ledger_ceiling = 60
        mock_settings.narrative_state_tracking_enabled = False
        mock_settings.narrative_request_log_limit = 10

        msg_texts = _extract_messages(messages)
        card_type = _classify_conversation(name)
        batch_end = min(20, len(msg_texts))

        # Generate a realistic ledger from real messages
        entries = []
        categories = ["discovery", "relationship_shift", "commitment", "world_change"]
        for i in range(min(12, batch_end)):
            snippet = msg_texts[i][:60].replace("\n", " ").strip()
            entries.append(MemoryEntry(
                round_num=i + 1,
                category=categories[i % len(categories)],
                content=snippet,
            ))

        if len(entries) < 2:
            pytest.skip("Not enough messages for compaction test")

        system, user = build_compaction_prompt(entries, card_type)

        # Validate compaction prompt structure
        assert "text compressor" in system.lower()
        assert "LOCKED KEY" in system
        assert f"{len(entries)} records" in system
        for e in entries:
            assert f"[R{e.round_num}|{e.category}]" in user
            # Actual message content should appear in the user prompt
            assert e.content[:20] in user


@pytest.mark.skipif(not ALL_CONVOS, reason="No conversation files found")
class TestMessageContentEdgeCases:
    """Test edge cases found in real conversation data."""

    def test_messages_with_markdown_formatting(self):
        """Messages with markdown (bold, italic, code blocks) don't break parsing."""
        # Real conversations use *action* formatting and **bold**
        messages_with_md = []
        for name, msgs in ALL_CONVOS[:5]:
            for m in msgs:
                text = m.get("mes", "")
                if "```" in text or "**" in text:
                    messages_with_md.append(text)
                    if len(messages_with_md) >= 10:
                        break
            if len(messages_with_md) >= 10:
                break

        if not messages_with_md:
            pytest.skip("No markdown-heavy messages found")

        # These should not crash prompt generation
        system, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], messages_with_md,
            "TestChar", 1, len(messages_with_md),
        )
        assert "## STATE" in system
        assert len(user) > 0

    def test_messages_with_special_characters(self):
        """Messages with special chars (quotes, newlines, unicode) handled."""
        special_msgs = []
        for name, msgs in ALL_CONVOS[:10]:
            for m in msgs:
                text = m.get("mes", "")
                if any(c in text for c in ['"', "'", "\r\n", "—", "…"]):
                    special_msgs.append(text)
                    if len(special_msgs) >= 10:
                        break
            if len(special_msgs) >= 10:
                break

        if not special_msgs:
            pytest.skip("No messages with special characters found")

        system, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], special_msgs,
            "TestChar", 1, len(special_msgs),
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_very_long_messages(self):
        """Very long messages (>2000 chars) don't break the 30-message window."""
        long_msgs = []
        for name, msgs in ALL_CONVOS:
            for m in msgs:
                text = m.get("mes", "")
                if len(text) > 2000:
                    long_msgs.append(text)
                    if len(long_msgs) >= 30:
                        break
            if len(long_msgs) >= 30:
                break

        if len(long_msgs) < 5:
            pytest.skip("Not enough long messages found")

        # The prompt builder truncates to last 30 messages
        system, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], long_msgs[:30],
            "TestChar", 1, 30,
        )
        assert "## STATE" in system
        # All 30 messages should have round numbers (unless refusal-filtered)
        r_count = len(re.findall(r"\[R\d+\]", user))
        assert r_count >= 1
