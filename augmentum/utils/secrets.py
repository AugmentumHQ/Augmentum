"""API key encryption at rest and error response sanitization.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
Key file is auto-generated on first use and stored alongside the database.
"""

from __future__ import annotations

import re
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_ENC_PREFIX = "enc:"
_fernet_instance = None

# --------------------------------------------------------------------------
# Patterns for sanitizing auth data from error responses
# --------------------------------------------------------------------------
_AUTH_PATTERNS = [
    # Bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    # Generic API key patterns (sk-..., key-..., etc.)
    re.compile(r"\b(?:sk|key|api[_-]?key|token)[_-][A-Za-z0-9]{16,}\b", re.IGNORECASE),
    # Authorization header values in error dumps
    re.compile(r"(?:Authorization|x-api-key|xi-api-key|x-key)\s*[:=]\s*\S+", re.IGNORECASE),
    # Base64-encoded long secrets (40+ chars of base64)
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
]

# Patterns that leak internal filesystem paths or stack traces
_PATH_PATTERNS = [
    # Unix absolute paths (e.g. /home/user/project/augmentum/...)
    re.compile(r"(?:/[\w._-]+){3,}/[\w._-]+\.py\b"),
    # Windows absolute paths (e.g. C:\Users\...\augmentum\...)
    re.compile(r"[A-Za-z]:\\(?:[\w._-]+\\){2,}[\w._-]+\.py\b"),
    # Python traceback file references (File "...", line N)
    re.compile(r'File\s+"[^"]+",\s+line\s+\d+'),
]


def _get_key_path() -> Path:
    """Return the path to the encryption key file."""
    from augmentum.config import settings
    return Path(settings.data_dir) / ".secret_key"


def _load_or_create_key() -> bytes:
    """Load the Fernet key from disk, or generate and save a new one."""
    key_path = _get_key_path()
    if key_path.exists():
        return key_path.read_bytes().strip()

    # Generate a new key
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()

    # Ensure parent directory exists
    key_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with restrictive permissions (owner-only on Unix)
    key_path.write_bytes(key)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass  # Windows doesn't support Unix permissions

    log.info("encryption_key_created", path=str(key_path))
    return key


def derive_secret(label: str, *, length: int = 32) -> str:
    """Deterministically derive a stable secret from this install's key.

    HMAC-SHA256 over the per-install ``.secret_key`` with a caller label.
    Same install + same label → same value, every time, with no extra
    storage — so derived service credentials survive restarts (and even a
    service's own volume being wiped) without a DB round-trip.

    Used to mint managed credentials for provisioned sidecars (e.g. the
    Basic-auth password baked into a fresh Suwayomi), where Augmentum is
    the source of truth and the secret must be reproducible across the
    container's lifecycle. ``length`` is the hex-char count (<= 64).
    """
    import hashlib
    import hmac

    key = _load_or_create_key()
    digest = hmac.new(key, label.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[: max(8, min(64, length))]


def _get_fernet():
    """Get or create the singleton Fernet instance."""
    global _fernet_instance
    if _fernet_instance is None:
        from cryptography.fernet import Fernet
        key = _load_or_create_key()
        _fernet_instance = Fernet(key)
    return _fernet_instance


def encrypt_api_key(plaintext: str | None) -> str | None:
    """Encrypt a secret for storage at rest. Returns None/empty unchanged.

    **Fails CLOSED.** If encryption is unavailable (corrupt or unreadable
    ``data_dir/.secret_key``), this raises rather than silently persisting the
    secret as plaintext — the operator must fix the key file. Reads still work
    (``decrypt_api_key`` passes legacy plaintext through), so the app degrades
    to "can't store new secrets" instead of "secrets stored in the clear
    without anyone noticing". Callers that persist credentials surface this as
    a write error, which is the correct, visible failure mode.
    """
    if not plaintext:
        return plaintext
    # Already encrypted — return as-is
    if plaintext.startswith(_ENC_PREFIX):
        return plaintext
    try:
        f = _get_fernet()
        token = f.encrypt(plaintext.encode("utf-8"))
        return _ENC_PREFIX + token.decode("ascii")
    except Exception as exc:
        log.error("encrypt_api_key_failed", exc_info=True)
        raise RuntimeError(
            "Cannot encrypt secret at rest — check the encryption key file "
            "(data_dir/.secret_key). Refusing to store the secret in plaintext."
        ) from exc


def decrypt_api_key(stored: str | None) -> str | None:
    """Decrypt an API key from storage. Handles plaintext (backward compat)."""
    if not stored:
        return stored
    if not stored.startswith(_ENC_PREFIX):
        # Plaintext (legacy) — return as-is
        return stored
    try:
        f = _get_fernet()
        token = stored[len(_ENC_PREFIX):].encode("ascii")
        return f.decrypt(token).decode("utf-8")
    except Exception:
        log.warning("decrypt_api_key_failed", exc_info=True)
        return None


def sanitize_error_detail(text: str) -> str:
    """Strip auth tokens, API keys, credentials, and internal paths from error text.

    Used before logging upstream error responses or returning them to clients.
    """
    if not text:
        return text
    result = text
    for pattern in _AUTH_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    for pattern in _PATH_PATTERNS:
        result = pattern.sub("[internal]", result)
    return result
