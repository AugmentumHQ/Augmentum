"""Tests for narrative features — macro expansion, prompt presets, regex scripts, groups."""

from __future__ import annotations

import pytest

from augmentum.models.base import InternalChatRequest, Message

# ── Macro Expansion ────────────────────────────────────────────────────

class TestMacroExpansion:
    def test_basic_char_user(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros(
            "Hello {{char}}, I am {{user}}.",
            char_name="Aria",
            user_name="Alex",
        )
        assert result == "Hello Aria, I am Alex."

    def test_case_insensitive(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros(
            "{{CHAR}} meets {{User}}",
            char_name="Aria",
            user_name="Alex",
        )
        assert result == "Aria meets Alex"

    def test_time_date_macros(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("Now: {{time}} on {{date}}, {{day}}")
        assert ":" in result  # time has colon
        assert "-" in result  # date has dashes
        # day should be a weekday name
        days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        for d in days:
            if d in result:
                break
        else:
            pytest.fail(f"No weekday found in: {result}")

    def test_random_macro(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("Roll: {{random}}")
        # Should be a float
        num_str = result.replace("Roll: ", "")
        val = float(num_str)
        assert 0.0 <= val < 1.0

    def test_dice_roll(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("Result: {{roll:2d6}}")
        num_str = result.replace("Result: ", "")
        val = int(num_str)
        assert 2 <= val <= 12

    def test_dice_roll_1d20(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("{{roll:1d20}}")
        val = int(result)
        assert 1 <= val <= 20

    def test_multiple_rolls(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("{{roll:1d6}} and {{roll:1d6}}")
        parts = result.split(" and ")
        assert len(parts) == 2
        for p in parts:
            assert 1 <= int(p) <= 6

    def test_no_macros_passthrough(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        text = "Just plain text without macros"
        assert expand_macros(text) == text

    def test_empty_string(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        assert expand_macros("") == ""
        assert expand_macros(None) is None

    def test_persona_macro(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros(
            "About you: {{persona}}",
            persona_description="A brave knight",
        )
        assert result == "About you: A brave knight"

    def test_expand_messages(self):
        from augmentum.modes.narrative.macro_expander import expand_messages
        msgs = [
            Message(role="system", content="You are {{char}}"),
            Message(role="user", content="I am {{user}}"),
        ]
        expand_messages(msgs, char_name="Aria", user_name="Alex")
        assert msgs[0].content == "You are Aria"
        assert msgs[1].content == "I am Alex"

    def test_idle_duration(self):
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("Messages: {{idle_duration}}", message_count=42)
        assert result == "Messages: 42"

    def test_roll_cap(self):
        """Large dice count is capped at 100."""
        from augmentum.modes.narrative.macro_expander import expand_macros
        result = expand_macros("{{roll:999d6}}")
        val = int(result)
        assert 100 <= val <= 600  # 100 dice, not 999


# ── Prompt Presets ─────────────────────────────────────────────────────

class TestPromptPresetApplication:
    def _make_request(self, *messages):
        return InternalChatRequest(
            model="test",
            messages=list(messages),
        )

    def test_system_prompt_prepend(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="You are Aria."),
            Message(role="user", content="Hello"),
        )
        preset = PromptPreset(system_prompt="Be creative and descriptive.")
        result = apply_preset(req, preset)
        assert result.messages[0].content.startswith("Be creative and descriptive.")
        assert "You are Aria." in result.messages[0].content

    def test_system_prompt_no_existing(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="user", content="Hello"),
        )
        preset = PromptPreset(system_prompt="Be creative.")
        result = apply_preset(req, preset)
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Be creative."

    def test_jailbreak_after_user(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="You are Aria."),
            Message(role="user", content="Hello"),
        )
        preset = PromptPreset(jailbreak="Stay in character always.")
        result = apply_preset(req, preset)
        # Jailbreak should be after the last user message
        user_idx = None
        jb_idx = None
        for i, m in enumerate(result.messages):
            if m.role == "user":
                user_idx = i
            if m.content == "Stay in character always.":
                jb_idx = i
        assert jb_idx is not None
        assert user_idx is not None
        assert jb_idx == user_idx + 1

    def test_post_history_before_user(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="Card"),
            Message(role="assistant", content="Hello"),
            Message(role="user", content="Hi"),
        )
        preset = PromptPreset(post_history="Continue the scene naturally.")
        result = apply_preset(req, preset)
        # post_history should be right before the last user message
        user_idx = None
        ph_idx = None
        for i, m in enumerate(result.messages):
            if m.content == "Hi":
                user_idx = i
            if m.content == "Continue the scene naturally.":
                ph_idx = i
        assert ph_idx is not None
        assert user_idx is not None
        assert ph_idx == user_idx - 1

    def test_author_note_at_depth(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="Card"),
            Message(role="user", content="Msg 1"),
            Message(role="assistant", content="Reply 1"),
            Message(role="user", content="Msg 2"),
            Message(role="assistant", content="Reply 2"),
            Message(role="user", content="Msg 3"),
            Message(role="assistant", content="Reply 3"),
            Message(role="user", content="Msg 4"),
        )
        preset = PromptPreset(author_note="Focus on emotions", author_note_depth=4)
        result = apply_preset(req, preset)
        # Find the author's note
        note_idx = None
        for i, m in enumerate(result.messages):
            if "Author's Note" in m.content:
                note_idx = i
                break
        assert note_idx is not None
        # Should be 4 non-system messages from the end
        non_system_after = sum(
            1 for m in result.messages[note_idx + 1:] if m.role != "system"
        )
        assert non_system_after >= 4

    def test_all_fields_combined(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="Card"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
        )
        preset = PromptPreset(
            system_prompt="Prefix.",
            jailbreak="JB.",
            post_history="PH.",
            author_note="AN.",
            author_note_depth=2,
        )
        result = apply_preset(req, preset)
        contents = [m.content for m in result.messages]
        # System should be prefixed
        assert contents[0].startswith("Prefix.")
        # All injected fields should be present
        assert any("JB." in c for c in contents)
        assert any("PH." in c for c in contents)
        assert any("Author's Note: AN." in c for c in contents)

    def test_empty_preset_passthrough(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="Card"),
            Message(role="user", content="Hello"),
        )
        preset = PromptPreset()  # All empty
        result = apply_preset(req, preset)
        assert len(result.messages) == len(req.messages)

    def test_original_not_mutated(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset
        req = self._make_request(
            Message(role="system", content="Card"),
            Message(role="user", content="Hello"),
        )
        original_content = req.messages[0].content
        preset = PromptPreset(system_prompt="Prefix.")
        apply_preset(req, preset)
        # Original should not be modified
        assert req.messages[0].content == original_content


# ── Regex Scripts ──────────────────────────────────────────────────────

class TestRegexTransformer:
    def test_basic_replacement(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"\bAI\b", replace_string="assistant", placement="output"),
        ]
        result = apply_regex_scripts("The AI responded.", scripts, "output")
        assert result == "The assistant responded."

    def test_placement_filter(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"test", replace_string="replaced", placement="input"),
        ]
        # Should NOT apply to output
        result = apply_regex_scripts("test", scripts, "output")
        assert result == "test"
        # Should apply to input
        result = apply_regex_scripts("test", scripts, "input")
        assert result == "replaced"

    def test_both_placement(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"foo", replace_string="bar", placement="both"),
        ]
        assert apply_regex_scripts("foo", scripts, "input") == "bar"
        assert apply_regex_scripts("foo", scripts, "output") == "bar"

    def test_ordering(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"a", replace_string="b", order_num=1, placement="output"),
            RegexScript(find_regex=r"b", replace_string="c", order_num=2, placement="output"),
        ]
        # Sort by order_num before applying
        scripts.sort(key=lambda s: s.order_num)
        # order_num=1 runs first: a→b ("ab" → "bb"), then order_num=2: b→c ("bb" → "cc")
        result = apply_regex_scripts("ab", scripts, "output")
        assert result == "cc"

    def test_disabled_scripts_skipped(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"test", replace_string="replaced", enabled=False, placement="output"),
        ]
        result = apply_regex_scripts("test", scripts, "output")
        assert result == "test"

    def test_invalid_regex_skipped(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(find_regex=r"[invalid", replace_string="x", placement="output"),
            RegexScript(find_regex=r"valid", replace_string="replaced", placement="output"),
        ]
        result = apply_regex_scripts("valid text", scripts, "output")
        assert result == "replaced text"

    def test_capture_groups(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [
            RegexScript(
                find_regex=r"\*(.+?)\*",
                replace_string=r"<em>\1</em>",
                placement="output",
            ),
        ]
        result = apply_regex_scripts("She *smiled* warmly.", scripts, "output")
        assert result == "She <em>smiled</em> warmly."

    def test_empty_text_passthrough(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        scripts = [RegexScript(find_regex=r"x", replace_string="y", placement="output")]
        assert apply_regex_scripts("", scripts, "output") == ""

    def test_empty_scripts_passthrough(self):
        from augmentum.modes.narrative.regex_transformer import apply_regex_scripts
        assert apply_regex_scripts("text", [], "output") == "text"


# ── Character Groups ──────────────────────────────────────────────────

class TestGroupTurnManager:
    def test_round_robin(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager
        group = CharacterGroup(
            name="Party",
            member_names=["Aria", "Kael", "Lyra"],
            generation_mode="round_robin",
        )
        mgr = GroupTurnManager(group)
        assert mgr.current_speaker == "Aria"
        assert mgr.advance() == "Kael"
        assert mgr.advance() == "Lyra"
        assert mgr.advance() == "Aria"  # wraps around

    def test_random_no_repeat(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager
        group = CharacterGroup(
            name="Duo",
            member_names=["Aria", "Kael"],
            generation_mode="random",
        )
        mgr = GroupTurnManager(group)
        mgr.advance()  # first advance
        # With 2 members, random should always pick the other one
        last = mgr.last_speaker
        current = mgr.current_speaker
        assert last != current

    def test_manual_set_speaker(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager
        group = CharacterGroup(
            name="Party",
            member_names=["Aria", "Kael", "Lyra"],
            generation_mode="manual",
        )
        mgr = GroupTurnManager(group)
        assert mgr.set_speaker("Lyra")
        assert mgr.current_speaker == "Lyra"
        assert not mgr.set_speaker("Nobody")  # not a member

    def test_to_dict(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager
        group = CharacterGroup(
            name="Party",
            member_names=["Aria", "Kael"],
            generation_mode="round_robin",
        )
        mgr = GroupTurnManager(group)
        d = mgr.to_dict()
        assert d["current_speaker"] == "Aria"
        assert d["generation_mode"] == "round_robin"
        assert d["members"] == ["Aria", "Kael"]

    def test_empty_group(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager
        group = CharacterGroup(name="Empty", member_names=[])
        mgr = GroupTurnManager(group)
        assert mgr.current_speaker == ""
        assert mgr.advance() == ""


# ── Preset Store (Integration, needs SQLite) ──────────────────────────

@pytest.fixture
async def db_conn():
    """Create an in-memory SQLite database with the schema."""
    import aiosqlite
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            jailbreak TEXT NOT NULL DEFAULT '',
            post_history TEXT NOT NULL DEFAULT '',
            author_note TEXT NOT NULL DEFAULT '',
            author_note_depth INTEGER NOT NULL DEFAULT 4,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS regex_scripts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            find_regex TEXT NOT NULL,
            replace_string TEXT NOT NULL DEFAULT '',
            placement TEXT NOT NULL DEFAULT 'output',
            enabled INTEGER NOT NULL DEFAULT 1,
            order_num INTEGER NOT NULL DEFAULT 100,
            character_name TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Schema mirrors migrations 018 + 043 (member_summaries) + 044 (avatar)
    # + 080 (muted_names) + 082 (user_id). Update this when columns land.
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS character_groups (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            member_names TEXT NOT NULL DEFAULT '[]',
            generation_mode TEXT NOT NULL DEFAULT 'round_robin',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            member_summaries TEXT NOT NULL DEFAULT '{}',
            avatar TEXT NOT NULL DEFAULT '',
            muted_names TEXT NOT NULL DEFAULT '[]',
            user_id TEXT
        )
    """)
    await conn.commit()
    yield conn
    await conn.close()


@pytest.mark.asyncio
class TestPresetStore:
    async def test_crud(self, db_conn):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, PromptPresetStore
        store = PromptPresetStore(db_conn)

        # Create
        preset = PromptPreset(
            name="Test",
            system_prompt="Be creative.",
            jailbreak="Stay in character.",
            is_default=True,
        )
        saved = await store.save_preset(preset)
        assert saved.id

        # List
        presets = await store.list_presets()
        assert len(presets) == 1
        assert presets[0].name == "Test"
        assert presets[0].system_prompt == "Be creative."

        # Get default
        default = await store.get_default()
        assert default is not None
        assert default.is_default

        # Get by ID
        fetched = await store.get_preset(saved.id)
        assert fetched is not None
        assert fetched.jailbreak == "Stay in character."

        # Update
        fetched.name = "Updated"
        await store.save_preset(fetched)
        updated = await store.get_preset(saved.id)
        assert updated.name == "Updated"

        # Delete
        deleted = await store.delete_preset(saved.id)
        assert deleted
        assert await store.get_preset(saved.id) is None

    async def test_default_toggle(self, db_conn):
        from augmentum.modes.narrative.prompt_presets import PromptPreset, PromptPresetStore
        store = PromptPresetStore(db_conn)

        p1 = PromptPreset(name="One", is_default=True)
        p2 = PromptPreset(name="Two", is_default=True)
        await store.save_preset(p1)
        await store.save_preset(p2)  # should un-default p1

        fetched_p1 = await store.get_preset(p1.id)
        fetched_p2 = await store.get_preset(p2.id)
        assert not fetched_p1.is_default
        assert fetched_p2.is_default


@pytest.mark.asyncio
class TestRegexScriptStore:
    async def test_crud(self, db_conn):
        from augmentum.modes.narrative.regex_transformer import RegexScript, RegexScriptStore
        store = RegexScriptStore(db_conn)

        script = RegexScript(
            name="Remove OOC",
            find_regex=r"\(OOC:.*?\)",
            replace_string="",
            placement="output",
        )
        saved = await store.save_script(script)

        scripts = await store.list_scripts()
        assert len(scripts) == 1
        assert scripts[0].name == "Remove OOC"

        # Toggle
        await store.toggle_script(saved.id, False)
        scripts = await store.list_scripts()
        assert not scripts[0].enabled

        # Delete
        await store.delete_script(saved.id)
        assert len(await store.list_scripts()) == 0

    async def test_character_filter(self, db_conn):
        from augmentum.modes.narrative.regex_transformer import RegexScript, RegexScriptStore
        store = RegexScriptStore(db_conn)

        await store.save_script(RegexScript(name="Global", find_regex=r"x", placement="output"))
        await store.save_script(RegexScript(
            name="Aria Only", find_regex=r"y", placement="output", character_name="Aria",
        ))
        await store.save_script(RegexScript(
            name="Kael Only", find_regex=r"z", placement="output", character_name="Kael",
        ))

        # Global + Aria
        scripts = await store.list_scripts(character_name="Aria")
        assert len(scripts) == 2
        names = {s.name for s in scripts}
        assert "Global" in names
        assert "Aria Only" in names
        assert "Kael Only" not in names


@pytest.mark.asyncio
class TestGroupStore:
    async def test_crud(self, db_conn):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupStore
        store = GroupStore(db_conn)
        uid = "user-test"

        group = CharacterGroup(
            name="Adventure Party",
            member_names=["Aria", "Kael", "Lyra"],
            generation_mode="round_robin",
        )
        saved = await store.save_group(group, user_id=uid)

        groups = await store.list_groups(user_id=uid)
        assert len(groups) == 1
        assert groups[0].member_names == ["Aria", "Kael", "Lyra"]

        fetched = await store.get_group(saved.id, user_id=uid)
        assert fetched.generation_mode == "round_robin"

        await store.delete_group(saved.id, user_id=uid)
        assert len(await store.list_groups(user_id=uid)) == 0

    async def test_read_paths_require_user_id(self, db_conn):
        """list_groups/get_group must raise rather than silently returning
        rows across tenants when uid is missing — auth middleware guarantees
        a uid on /api/narrative/*, so an empty uid here means a regression."""
        import pytest

        from augmentum.modes.narrative.group_manager import GroupStore
        store = GroupStore(db_conn)

        with pytest.raises(ValueError, match="user_id"):
            await store.list_groups()
        with pytest.raises(ValueError, match="user_id"):
            await store.get_group("some-id")

    async def test_tenant_isolation(self, db_conn):
        """A group owned by user A is invisible to user B even when B knows
        the id — the cross-tenant predicate is what blocks forged
        X-Augmentum-Group-Id headers from leaking another user's data."""
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupStore
        store = GroupStore(db_conn)

        a_group = await store.save_group(
            CharacterGroup(name="A's Party", member_names=["X", "Y"]),
            user_id="user-a",
        )

        # User B can't list user A's groups
        assert await store.list_groups(user_id="user-b") == []
        # User B can't fetch user A's group by id either
        assert await store.get_group(a_group.id, user_id="user-b") is None
        # User A still sees their own
        assert len(await store.list_groups(user_id="user-a")) == 1
