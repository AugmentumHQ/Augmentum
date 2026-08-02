"""Tests for the message converter framework."""

from __future__ import annotations

import pytest

from augmentum.models.converters.base import MessageConverter, PostProcessMode
from augmentum.models.converters.utils import (
    ZWS,
    extract_system_prefix,
    force_alternating,
    merge_consecutive_messages,
    post_process,
    prepend_name,
)


class TestMergeConsecutive:
    def test_no_merge_needed(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = merge_consecutive_messages(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "hi"

    def test_merge_two_user(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "first\n\nsecond"

    def test_merge_preserves_order(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u3"},
        ]
        result = merge_consecutive_messages(msgs)
        assert len(result) == 4
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "u1\n\nu2"}
        assert result[2] == {"role": "assistant", "content": "a1"}
        assert result[3] == {"role": "user", "content": "u3"}


class TestPrependName:
    def test_prepend_with_name(self) -> None:
        msg = {"role": "user", "content": "hello", "name": "Alice"}
        result = prepend_name(msg)
        assert result["content"] == "Alice: hello"
        assert "name" not in result

    def test_no_name_unchanged(self) -> None:
        msg = {"role": "user", "content": "hello"}
        result = prepend_name(msg)
        assert result["content"] == "hello"
        assert result == {"role": "user", "content": "hello"}


class TestForceAlternating:
    def test_insert_placeholder_between_same_role(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = force_alternating(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == ZWS
        assert result[2]["role"] == "user"

    def test_already_alternating(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        result = force_alternating(msgs)
        assert len(result) == 3
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "hi"
        assert result[2]["content"] == "bye"


class TestExtractSystemPrefix:
    def test_extracts_leading_system(self) -> None:
        msgs = [
            {"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        system, rest = extract_system_prefix(msgs)
        assert len(system) == 2
        assert system[0]["content"] == "s1"
        assert system[1]["content"] == "s2"
        assert len(rest) == 2
        assert rest[0]["role"] == "user"

    def test_no_system_prefix(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        system, rest = extract_system_prefix(msgs)
        assert len(system) == 0
        assert len(rest) == 2


class TestPostProcess:
    def test_none_passthrough(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        result = post_process(msgs, PostProcessMode.NONE)
        assert len(result) == 2
        assert result[0]["content"] == "a"
        assert result[1]["content"] == "b"

    def test_merge_mode(self) -> None:
        msgs = [
            {"role": "user", "content": "hello", "name": "Alice"},
            {"role": "user", "content": "world"},
        ]
        result = post_process(msgs, PostProcessMode.MERGE)
        assert len(result) == 1
        assert "Alice: hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_semi_mode(self) -> None:
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "mid-system"},
            {"role": "user", "content": "u2"},
        ]
        result = post_process(msgs, PostProcessMode.SEMI)
        # mid-conversation system becomes user, then consecutive users merge
        assert all(m["role"] != "system" for m in result)

    def test_strict_mode(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        result = post_process(msgs, PostProcessMode.STRICT)
        # strict: semi processing + force alternating
        # After merge: merged-user, assistant
        # Already alternating, so no placeholders needed
        assert result[-1]["role"] == "assistant"
        # Verify alternation holds
        for i in range(1, len(result)):
            assert result[i]["role"] != result[i - 1]["role"]


class TestInputImmutability:
    """Verify that utility functions don't mutate their inputs."""

    def test_merge_no_mutation(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        original = [dict(m) for m in msgs]
        merge_consecutive_messages(msgs)
        assert msgs == original

    def test_prepend_name_no_mutation(self) -> None:
        msg = {"role": "user", "content": "hello", "name": "Alice"}
        original = dict(msg)
        prepend_name(msg)
        assert msg == original
