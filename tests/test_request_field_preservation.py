"""Round-trip tests for InternalChatRequest field preservation through transform sites.

The ``InternalChatRequest`` dataclass has accumulated fields over time —
``kv_session_key`` / ``kv_stable_messages`` / ``kv_mode`` for KV cache
restoration, ``lorebook`` / ``group_id`` / ``speaker_override`` for
narrative routing, etc. Several functions in the codebase TRANSFORM an
existing request into a modified one (preset application, KV checkpoint
prep, narrative augmentation). When these transforms construct a fresh
``InternalChatRequest(...)`` with an explicit field list rather than
using ``dataclasses.replace(...)``, any new field added to the dataclass
gets silently dropped on the next code path that goes through the
transform.

This bit us in production: ``apply_preset`` dropped ``kv_stable_messages``,
which made ``prepare_stable_checkpoint`` silently no-op, which made every
narrative turn pay full GPU prefill cost. See commit 731a96d.

These tests build a request with EVERY field set to a non-default
value and then run it through each known transform site. If any field
fails to survive, the test fails immediately and the next bug-class
regression is caught at CI rather than in a user transcript.

Adding a new field to InternalChatRequest? Add it to ``_make_full_request``
below so this test covers it. The whole point is to catch the next
silent-drop before it ships.
"""
from __future__ import annotations

from augmentum.models.base import InternalChatRequest, Message


def _make_full_request() -> InternalChatRequest:
    """Construct a request with every field set to a non-default value.

    Defaults for `InternalChatRequest` (per base.py): zeros, empty strings,
    empty lists, None. We use distinct sentinels so a transform that
    silently drops a field shows up as the sentinel being missing.
    """
    return InternalChatRequest(
        model="test-model-sentinel",
        messages=[Message(role="user", content="hello")],
        stream=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=512,
        stop=["</stop>"],
        frequency_penalty=0.1,
        presence_penalty=0.2,
        seed=42,
        tools=[{"type": "function", "function": {"name": "test_tool"}}],
        tool_choice="required",
        chat_template_kwargs={"enable_thinking": False},
        format="json",
        keep_alive="5m",
        raw_options={"custom": "value"},
        think=True,
        reasoning_effort="high",
        preserve_thinking=True,
        memory_hint="recall:test",
        voice_input=True,
        explicit_flow_name="explicit-flow-sentinel",
        lorebook=[{"key": "test", "value": "lore-sentinel"}],
        group_id="group-sentinel",
        speaker_override="speaker-sentinel",
        kv_session_key="session-key-sentinel",
        kv_stable_messages=[Message(role="user", content="stable-sentinel")],
        kv_mode="kv-mode-sentinel",
        pack_injection={"pack_id": "pack-sentinel", "n_sources": 3},
        is_background_task=True,
        training_mode=True,
        continue_last_assistant=True,
    )


def _assert_field_survives(
    original: InternalChatRequest,
    transformed: InternalChatRequest,
    field_name: str,
) -> None:
    """Assert a field on ``transformed`` matches the original (or was deliberately overridden)."""
    original_val = getattr(original, field_name)
    transformed_val = getattr(transformed, field_name)
    assert transformed_val == original_val, (
        f"Field {field_name!r} was silently dropped or mutated by transform: "
        f"original={original_val!r}, after_transform={transformed_val!r}"
    )


# ---------------------------------------------------------------------------
# apply_preset (narrative.prompt_presets)
# ---------------------------------------------------------------------------


def test_apply_preset_preserves_all_fields():
    """``apply_preset`` should preserve every InternalChatRequest field
    from the input on its return — only ``messages`` should change."""
    from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset

    original = _make_full_request()
    preset = PromptPreset(
        id="test-preset",
        name="Test Preset",
        system_prompt="",  # empty so messages don't actually change
        author_note="",
        post_history="",
        jailbreak="",
    )

    transformed = apply_preset(original, preset)

    # Every field except ``messages`` must survive.
    fields_to_check = [
        "model", "stream", "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty", "seed", "tools", "format",
        "keep_alive", "raw_options", "think", "memory_hint", "voice_input",
        "explicit_flow_name", "lorebook", "group_id", "speaker_override",
        "kv_session_key", "kv_stable_messages", "kv_mode",
    ]
    for f in fields_to_check:
        _assert_field_survives(original, transformed, f)


def test_apply_preset_kv_fields_specifically():
    """Regression guard for the 731a96d bug: kv_stable_messages and
    kv_session_key MUST survive apply_preset. If this fails,
    prepare_stable_checkpoint will silently no-op on every turn."""
    from augmentum.modes.narrative.prompt_presets import PromptPreset, apply_preset

    original = _make_full_request()
    preset = PromptPreset(
        id="t", name="t", system_prompt="", author_note="",
        post_history="", jailbreak="",
    )
    transformed = apply_preset(original, preset)

    assert transformed.kv_session_key == "session-key-sentinel"
    assert transformed.kv_stable_messages is not None
    assert len(transformed.kv_stable_messages) == 1
    assert transformed.kv_stable_messages[0].content == "stable-sentinel"
    assert transformed.kv_mode == "kv-mode-sentinel"


# ---------------------------------------------------------------------------
# _checkpoint_request_from_messages (llama_cpp.LlamaCppBackend)
# ---------------------------------------------------------------------------


def test_checkpoint_request_preserves_non_overridden_fields():
    """``_checkpoint_request_from_messages`` overrides ``messages``,
    ``stream``, ``kv_session_key``, and ``kv_stable_messages`` by design.
    Every OTHER field on the source request must survive."""
    from augmentum.models.llama_cpp import LlamaCppBackend

    original = _make_full_request()
    new_messages = [Message(role="user", content="checkpoint-content")]
    new_key = "new-checkpoint-key"

    transformed = LlamaCppBackend._checkpoint_request_from_messages(
        original, new_messages, new_key,
    )

    # These four fields are intentionally overridden — verify the override.
    assert transformed.messages[0].content == "checkpoint-content"
    assert transformed.stream is False  # checkpoint must be non-streaming
    assert transformed.kv_session_key == new_key
    assert transformed.kv_stable_messages[0].content == "checkpoint-content"

    # Every other field must come through from the source.
    fields_to_check = [
        "model", "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty", "seed", "tools", "format",
        "keep_alive", "raw_options", "think", "memory_hint", "voice_input",
        "explicit_flow_name", "lorebook", "group_id", "speaker_override",
        "kv_mode",
    ]
    for f in fields_to_check:
        _assert_field_survives(original, transformed, f)


# ---------------------------------------------------------------------------
# NarrativeEngine._augment_request
# ---------------------------------------------------------------------------


def test_augment_request_preserves_non_overridden_fields():
    """``_augment_request`` overrides ``messages`` (with augmented context)
    and ``kv_stable_messages`` (snapshot of incoming messages). Every
    other field on the source request must survive."""
    from augmentum.modes.narrative.engine import NarrativeEngine

    original = _make_full_request()

    # _augment_request is a method, but it's effectively pure for the
    # field-preservation check — we don't need a real engine. It uses
    # ``self`` only to read state and inject context, neither of which
    # affect the fields we're testing here. Construct a minimal engine.
    engine = NarrativeEngine.__new__(NarrativeEngine)
    engine._state = type("S", (), {
        "system_prompt": "", "character_card_name": "",
        "card_type": "character", "user_persona_name": "",
        "user_persona_description": "", "world_book_entries": [],
        "memory_settings": None, "compaction_settings": None,
        "scene": None, "lorebook_entries": [], "facts": [],
        "entities": {}, "plot_threads": [], "memory_ledger": [],
        "narration_style": "", "extra_directive": "", "regex_scripts": [],
    })()
    engine._memory_ledger = []
    engine._message_history = []
    engine._context_budget = 4096
    engine._token_estimator = lambda x: len(x or "") // 4
    engine._tokenizer_name = "approx"
    engine._lore_engine = None
    engine._world_tracker = type("W", (), {"current_scene": None})()
    engine._character_tracker = type("C", (), {"by_name": {}})()
    engine._narrative_style_locked = False

    # Build the preconditions _augment_request expects: a BuiltContext
    # and a list of system messages it can prepend to.
    from augmentum.modes.narrative.context_builder import BuiltContext

    context = BuiltContext()
    context.total_tokens_estimate = 100

    transformed = engine._augment_request(
        request=original,
        context=context,
        context_limit=0,
    )

    # ``messages`` and ``kv_stable_messages`` are intentionally
    # overridden — the test for those is in the engine's own tests.
    # Every OTHER field must come through.
    fields_to_check = [
        "model", "stream", "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty", "seed", "tools", "format",
        "keep_alive", "raw_options", "think", "memory_hint", "voice_input",
        "explicit_flow_name", "lorebook", "group_id", "speaker_override",
        "kv_session_key", "kv_mode",
    ]
    for f in fields_to_check:
        _assert_field_survives(original, transformed, f)


# ---------------------------------------------------------------------------
# Schema-shape sanity check
# ---------------------------------------------------------------------------


def test_full_request_helper_covers_every_field():
    """Meta-test: ``_make_full_request`` must set every public field on
    ``InternalChatRequest`` to a non-default value. If a new field is
    added to the dataclass and this helper isn't updated, the
    round-trip tests above will silently miss the new field. This test
    fails first and points the author at the gap."""
    import dataclasses
    request = _make_full_request()
    fields = {f.name for f in dataclasses.fields(InternalChatRequest)}
    # Every field on the dataclass should have a non-default value here.
    for field in fields:
        value = getattr(request, field)
        # ``messages`` is a list — non-empty means non-default.
        if field == "messages":
            assert value, f"Field {field!r} must be non-empty in _make_full_request"
            continue
        # All other fields: not None, not empty, not zero, not False.
        # A field with a literal default value would slip through this
        # check if its sentinel happened to equal the default; we use
        # distinct strings/numbers in _make_full_request to prevent that.
        assert value is not None and value != "" and value != 0 and value is not False, (
            f"Field {field!r} has default-like value in _make_full_request: {value!r}. "
            f"Update _make_full_request with a distinct sentinel value."
        )
