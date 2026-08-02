"""Argon2id password hashing."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# OWASP-recommended Argon2id parameters:
# memory_cost=19456 (19 MiB), time_cost=2, parallelism=1
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=19456,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)

# Precomputed dummy hash for constant-time login on unknown usernames
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing-resistance")


def hash_password(password: str) -> str:
    """Hash a password with Argon2id. Returns encoded hash string."""
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Verify a password against a stored hash. Constant-time on failure."""
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except VerificationError:
        log.warning("password_verify_error", exc_info=True)
        return False


def verify_dummy(password: str) -> bool:
    """Run Argon2id on a dummy hash to prevent timing attacks on unknown usernames."""
    try:
        _hasher.verify(_DUMMY_HASH, password)
    except (VerifyMismatchError, VerificationError):
        pass
    return False


def needs_rehash(password_hash: str) -> bool:
    """Check if a stored hash should be re-hashed (parameters changed)."""
    return _hasher.check_needs_rehash(password_hash)
