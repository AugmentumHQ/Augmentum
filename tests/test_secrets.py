"""Tests for API key encryption and error sanitization."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_fernet():
    """Reset the singleton Fernet instance between tests."""
    import augmentum.utils.secrets as mod
    mod._fernet_instance = None
    yield
    mod._fernet_instance = None


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Point settings.data_dir to a temp directory."""
    with patch("augmentum.utils.secrets._get_key_path", return_value=tmp_path / ".secret_key"):
        yield tmp_path


# ── Encryption / Decryption ──────────────────────────────────────────────


class TestEncryptDecrypt:
    def test_round_trip(self, tmp_data_dir):
        from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key
        encrypted = encrypt_api_key("sk-test-12345")
        assert encrypted is not None
        assert encrypted.startswith("enc:")
        assert "sk-test-12345" not in encrypted
        assert decrypt_api_key(encrypted) == "sk-test-12345"

    def test_none_passthrough(self, tmp_data_dir):
        from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key
        assert encrypt_api_key(None) is None
        assert decrypt_api_key(None) is None

    def test_empty_passthrough(self, tmp_data_dir):
        from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key
        assert encrypt_api_key("") == ""
        assert decrypt_api_key("") == ""

    def test_already_encrypted_skipped(self, tmp_data_dir):
        from augmentum.utils.secrets import encrypt_api_key
        already = "enc:some-token"
        assert encrypt_api_key(already) == already

    def test_plaintext_backward_compat(self, tmp_data_dir):
        from augmentum.utils.secrets import decrypt_api_key
        assert decrypt_api_key("sk-plain-key") == "sk-plain-key"

    def test_key_file_created(self, tmp_data_dir):
        from augmentum.utils.secrets import encrypt_api_key
        encrypt_api_key("test")
        key_file = tmp_data_dir / ".secret_key"
        assert key_file.exists()
        assert len(key_file.read_bytes().strip()) > 20

    def test_key_file_reused(self, tmp_data_dir):
        from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key
        enc1 = encrypt_api_key("aaa")
        # Reset singleton but keep the key file
        import augmentum.utils.secrets as mod
        mod._fernet_instance = None
        assert decrypt_api_key(enc1) == "aaa"

    def test_wrong_key_returns_none(self, tmp_data_dir):
        from augmentum.utils.secrets import encrypt_api_key
        enc = encrypt_api_key("secret")
        # Overwrite with a different key
        from cryptography.fernet import Fernet
        (tmp_data_dir / ".secret_key").write_bytes(Fernet.generate_key())
        import augmentum.utils.secrets as mod
        mod._fernet_instance = None
        from augmentum.utils.secrets import decrypt_api_key
        assert decrypt_api_key(enc) is None

    def test_encrypt_fails_closed_not_plaintext(self, tmp_data_dir):
        """If encryption is unavailable, encrypt MUST raise — never silently
        return the plaintext (which would persist a secret in the clear).
        And the error must not leak the secret value."""
        import augmentum.utils.secrets as mod
        with patch.object(mod, "_get_fernet", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError) as exc_info:
                mod.encrypt_api_key("sk-super-secret")
        assert "sk-super-secret" not in str(exc_info.value)


# ── Error Sanitization ───────────────────────────────────────────────────


class TestSanitization:
    def test_bearer_token_stripped(self):
        from augmentum.utils.secrets import sanitize_error_detail
        text = 'Authorization: Bearer sk-abc123def456ghijklmnop error 401'
        result = sanitize_error_detail(text)
        assert "sk-abc123def456ghijklmnop" not in result
        assert "[REDACTED]" in result

    def test_api_key_pattern_stripped(self):
        from augmentum.utils.secrets import sanitize_error_detail
        text = 'Invalid key: sk-1234567890abcdef1234567890abcdef'
        result = sanitize_error_detail(text)
        assert "sk-1234567890abcdef1234567890abcdef" not in result

    def test_xi_api_key_stripped(self):
        from augmentum.utils.secrets import sanitize_error_detail
        text = 'xi-api-key: abc123def456ghi789jkl012mno345'
        result = sanitize_error_detail(text)
        assert "abc123def456ghi789jkl012mno345" not in result

    def test_empty_passthrough(self):
        from augmentum.utils.secrets import sanitize_error_detail
        assert sanitize_error_detail("") == ""

    def test_safe_text_unchanged(self):
        from augmentum.utils.secrets import sanitize_error_detail
        text = "Connection refused: could not reach backend at port 8080"
        assert sanitize_error_detail(text) == text

    def test_x_key_header_stripped(self):
        from augmentum.utils.secrets import sanitize_error_detail
        text = 'x-key: mySecretKeyValue123 was invalid'
        result = sanitize_error_detail(text)
        assert "mySecretKeyValue123" not in result
