"""Tests for augmentum/utils/tokenizer.py — token counting."""

from __future__ import annotations

from augmentum.utils.tokenizer import count_tokens, count_tokens_messages


class TestCountTokens:
    """Verify token counting for various inputs."""

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_none_like_empty(self):
        # Empty string explicitly
        assert count_tokens("") == 0

    def test_hello_world(self):
        result = count_tokens("Hello, world!")
        assert result > 0
        # tiktoken cl100k_base: "Hello, world!" is typically 4 tokens
        assert result < 20

    def test_single_word(self):
        result = count_tokens("hello")
        assert result == 1

    def test_long_string_does_not_crash(self):
        text = "word " * 100_000
        result = count_tokens(text)
        assert result > 0

    def test_unicode_text(self):
        result = count_tokens("Bonjour le monde!")
        assert result > 0

    def test_code_snippet(self):
        code = "def hello():\n    return 'world'"
        result = count_tokens(code)
        assert result > 0

    def test_whitespace_only(self):
        result = count_tokens("   ")
        assert result > 0  # Whitespace still has tokens


class TestCountTokensMessages:
    """Verify message-level token counting."""

    def test_empty_messages(self):
        assert count_tokens_messages([]) == 0

    def test_single_dict_message(self):
        result = count_tokens_messages([{"role": "user", "content": "Hello"}])
        # 1 token for "Hello" + 4 overhead
        assert result >= 5

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there."},
        ]
        result = count_tokens_messages(msgs)
        assert result > 8  # At least 2 * 4 overhead

    def test_message_without_content(self):
        result = count_tokens_messages([{"role": "user"}])
        # Just overhead (4 tokens)
        assert result == 4

    def test_object_with_content_attr(self):
        class FakeMsg:
            content = "Hello world"
        result = count_tokens_messages([FakeMsg()])
        assert result > 4
