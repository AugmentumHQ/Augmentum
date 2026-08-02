"""Tests for auth password hashing."""

from __future__ import annotations

from augmentum.auth.passwords import (
    hash_password,
    verify_password,
    verify_dummy,
    needs_rehash,
)


class TestHashPassword:
    def test_produces_argon2id_hash(self):
        h = hash_password("test-password-123")
        assert h.startswith("$argon2id$")

    def test_different_passwords_different_hashes(self):
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2

    def test_same_password_different_hashes(self):
        """Argon2 uses random salt — same input produces different output."""
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2


class TestVerifyPassword:
    def test_correct_password(self):
        h = hash_password("correct-horse-battery-staple")
        assert verify_password(h, "correct-horse-battery-staple") is True

    def test_wrong_password(self):
        h = hash_password("correct-password")
        assert verify_password(h, "wrong-password") is False

    def test_empty_password(self):
        h = hash_password("real-password")
        assert verify_password(h, "") is False

    def test_corrupted_hash(self):
        assert verify_password("not-a-valid-hash", "password") is False


class TestVerifyDummy:
    def test_always_returns_false(self):
        assert verify_dummy("any-password") is False
        assert verify_dummy("") is False

    def test_takes_measurable_time(self):
        """Dummy verify should take similar time to real verify (Argon2id)."""
        import time
        start = time.monotonic()
        verify_dummy("test")
        elapsed = time.monotonic() - start
        assert elapsed > 0.01  # At least 10ms (Argon2id is intentionally slow)


class TestNeedsRehash:
    def test_current_params_no_rehash(self):
        h = hash_password("test")
        assert needs_rehash(h) is False
