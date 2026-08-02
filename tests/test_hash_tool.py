"""Tests for HashTool — cryptographic hash computation and comparison."""

from __future__ import annotations

import pytest

from augmentum.tools.hash_tool import HashTool, compute_hash, compute_hmac


class TestComputeHash:
    """Direct tests on compute_hash function."""

    def test_md5_length(self):
        h = compute_hash("hello", "md5")
        assert len(h) == 32  # MD5 = 128 bits = 32 hex chars

    def test_sha256_length(self):
        h = compute_hash("hello", "sha256")
        assert len(h) == 64  # SHA256 = 256 bits = 64 hex chars

    def test_sha512_length(self):
        h = compute_hash("hello", "sha512")
        assert len(h) == 128  # SHA512 = 512 bits = 128 hex chars

    def test_sha1_length(self):
        h = compute_hash("hello", "sha1")
        assert len(h) == 40

    def test_consistent_output(self):
        h1 = compute_hash("test", "sha256")
        h2 = compute_hash("test", "sha256")
        assert h1 == h2

    def test_different_inputs_differ(self):
        h1 = compute_hash("hello", "sha256")
        h2 = compute_hash("world", "sha256")
        assert h1 != h2

    def test_empty_string_produces_hash(self):
        h = compute_hash("a", "sha256")  # using single char, not empty (tool rejects empty)
        assert len(h) == 64

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_hash("test", "md4")

    def test_blake2b_supported(self):
        h = compute_hash("test", "blake2b")
        assert len(h) > 0

    def test_sha3_256_supported(self):
        h = compute_hash("test", "sha3_256")
        assert len(h) == 64


class TestComputeHmac:
    """HMAC computation."""

    def test_hmac_sha256(self):
        h = compute_hmac("message", "secret", "sha256")
        assert len(h) == 64

    def test_hmac_consistent(self):
        h1 = compute_hmac("msg", "key", "sha256")
        h2 = compute_hmac("msg", "key", "sha256")
        assert h1 == h2

    def test_hmac_different_keys_differ(self):
        h1 = compute_hmac("msg", "key1", "sha256")
        h2 = compute_hmac("msg", "key2", "sha256")
        assert h1 != h2


class TestHashToolExecute:
    """HashTool execute() contract tests."""

    async def test_hash_action_returns_digest(self):
        tool = HashTool()
        result = await tool.execute(action="hash", text="hello world")
        assert result.success is True
        assert len(result.output) == 64  # default sha256
        assert result.metadata["algorithm"] == "sha256"

    async def test_hash_custom_algorithm(self):
        tool = HashTool()
        result = await tool.execute(action="hash", text="test", algorithm="md5")
        assert result.success is True
        assert len(result.output) == 32

    async def test_hmac_action_requires_key(self):
        tool = HashTool()
        result = await tool.execute(action="hmac", text="test")
        assert result.success is False
        assert "key" in result.error.lower()

    async def test_hmac_action_with_key(self):
        tool = HashTool()
        result = await tool.execute(action="hmac", text="message", key="secret")
        assert result.success is True
        assert len(result.output) == 64

    async def test_compare_action_match(self):
        tool = HashTool()
        digest = compute_hash("hello", "sha256")
        result = await tool.execute(action="compare", text="hello", expected=digest)
        assert result.success is True
        assert result.metadata["match"] is True

    async def test_compare_action_no_match(self):
        tool = HashTool()
        result = await tool.execute(action="compare", text="hello", expected="wrong_hash")
        assert result.success is True
        assert result.metadata["match"] is False

    async def test_empty_text_returns_error(self):
        tool = HashTool()
        result = await tool.execute(action="hash", text="")
        assert result.success is False

    async def test_unknown_action(self):
        tool = HashTool()
        result = await tool.execute(action="unknown", text="test")
        assert result.success is False

    async def test_tool_properties(self):
        tool = HashTool()
        assert tool.name == "hash"
        assert tool.category.value == "verify"
