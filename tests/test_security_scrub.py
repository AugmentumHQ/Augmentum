"""Tests — augmentum.security.scrub secret-leak prevention.

Covers:
  * is_secret_key suffix matching + exact sensitive list + allowlist
  * scrub_dict shallow vs deep recursion
  * Nested dicts / lists / tuples handled
  * Non-secret keys pass through unchanged
  * Allowlist defeats suffix match
  * Mutation safety — input dict is not modified
  * scrub_response convenience wrapper for dict / list / scalar
"""

from __future__ import annotations

import pytest

from augmentum.security.scrub import (
    REDACTED,
    is_secret_key,
    scrub_dict,
    scrub_response,
)


class TestIsSecretKey:
    @pytest.mark.parametrize("key", [
        "openai_api_key", "anthropic_apikey",
        "user_password", "old_passwd", "new_pass", "current_pwd",
        "shared_secret", "stripe_client_secret",
        "session_token", "user_access_token", "oauth_refresh_token",
        "auth_bearer_token",
        "smtp_credential", "imap_credentials",
        "encryption_key", "signing_key", "ssh_private_key",
        "totp_secret", "otp_secret",
        "session_id",
        "request_signature",
        "session_cookie",
    ])
    def test_suffix_match_redacts(self, key):
        assert is_secret_key(key) is True

    @pytest.mark.parametrize("key", [
        "api_key", "apikey", "password", "secret", "token",
    ])
    def test_bare_name_match_redacts(self, key):
        assert is_secret_key(key) is True

    @pytest.mark.parametrize("key", [
        "user_name", "session_label", "monkey_count",
        "broker_url", "tokenized_input", "secretary_id",
        "passcode_required", "keystroke_count",
        # The string "key" appears but not as a suffix marker
        "ascii_keyset",
    ])
    def test_substring_false_positives_pass_through(self, key):
        # These would all match a naive substring "key in name" — they
        # MUST pass through.
        assert is_secret_key(key) is False

    def test_allowlist_overrides_suffix(self):
        # google_pse_cx is a public identifier even though "cx" suffix
        # would be ambiguous; explicitly allowlisted.
        assert is_secret_key("google_pse_cx") is False

    def test_sensitive_exact_list_matches(self):
        # Not secret-shaped but flagged as a capability handle we don't
        # want exposed pre-auth.
        assert is_secret_key("reminder_webhook_integration_id") is True

    @pytest.mark.parametrize("key", ["", None])
    def test_empty_or_none(self, key):
        assert is_secret_key(key) is False

    def test_case_insensitive(self):
        assert is_secret_key("USER_PASSWORD") is True
        assert is_secret_key("ApiKey") is True
        assert is_secret_key("Bearer_Token") is True


class TestScrubDict:
    def test_redacts_top_level_secret(self):
        out = scrub_dict({"openai_api_key": "sk-xxx", "user_name": "matt"})
        assert out["openai_api_key"] == REDACTED
        assert out["user_name"] == "matt"

    def test_deep_recursion_nested_dict(self):
        out = scrub_dict({
            "providers": {
                "openai": {"api_key": "sk-1", "base_url": "https://x"},
                "anthropic": {"api_key": "sk-2"},
            },
        })
        assert out["providers"]["openai"]["api_key"] == REDACTED
        assert out["providers"]["openai"]["base_url"] == "https://x"
        assert out["providers"]["anthropic"]["api_key"] == REDACTED

    def test_list_recursion(self):
        out = scrub_dict({
            "keys": [
                {"api_key": "k1"},
                {"api_key": "k2", "label": "live"},
            ],
        })
        assert out["keys"][0]["api_key"] == REDACTED
        assert out["keys"][1]["api_key"] == REDACTED
        assert out["keys"][1]["label"] == "live"

    def test_tuple_recursion(self):
        out = scrub_dict({"pair": ({"password": "p"}, {"safe": True})})
        # Tuples are preserved as tuples.
        assert isinstance(out["pair"], tuple)
        assert out["pair"][0]["password"] == REDACTED
        assert out["pair"][1]["safe"] is True

    def test_shallow_mode_does_not_recurse(self):
        original = {"providers": {"openai": {"api_key": "sk"}}}
        out = scrub_dict(original, deep=False)
        # Shallow — nested api_key passes through (shallow callers
        # validated structure another way).
        assert out["providers"]["openai"]["api_key"] == "sk"

    def test_input_not_mutated(self):
        original = {"api_key": "secret", "nested": {"token": "t"}}
        snapshot = {"api_key": "secret", "nested": {"token": "t"}}
        scrub_dict(original)
        assert original == snapshot

    def test_none_input_returns_empty(self):
        assert scrub_dict(None) == {}

    def test_non_dict_input_returns_empty(self):
        # Defensive — handler occasionally passes a response object of
        # unexpected shape; fail closed (empty payload) rather than
        # crash.
        assert scrub_dict("a string") == {}
        assert scrub_dict([1, 2, 3]) == {}
        assert scrub_dict(42) == {}

    def test_allowlist_key_passes_through_in_scrub(self):
        out = scrub_dict({
            "google_pse_cx": "xx:yy",
            "google_pse_api_key": "secret_one",
        })
        assert out["google_pse_cx"] == "xx:yy"
        assert out["google_pse_api_key"] == REDACTED


class TestScrubResponse:
    def test_dict_payload(self):
        out = scrub_response({"api_key": "x", "name": "n"})
        assert out["api_key"] == REDACTED
        assert out["name"] == "n"

    def test_list_payload(self):
        out = scrub_response([{"api_key": "x"}, {"api_key": "y"}])
        assert out[0]["api_key"] == REDACTED
        assert out[1]["api_key"] == REDACTED

    def test_tuple_payload(self):
        out = scrub_response(({"password": "p"}, {"safe": True}))
        assert isinstance(out, tuple)
        assert out[0]["password"] == REDACTED

    def test_scalar_passes_through(self):
        assert scrub_response("a string") == "a string"
        assert scrub_response(42) == 42
        assert scrub_response(True) is True
        assert scrub_response(None) is None


class TestRealisticPayloads:
    """Spot-check against the kinds of payloads actually returned by
    pre-auth surfaces (settings reads, provider listings, version info)."""

    def test_provider_listing_payload(self):
        payload = {
            "providers": [
                {
                    "name": "OpenAI",
                    "base_url": "https://api.openai.com",
                    "api_key": "sk-aaa",
                    "default_model": "gpt-5",
                },
                {
                    "name": "Anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key": "sk-bbb",
                },
            ],
            "active": "openai",
        }
        out = scrub_response(payload)
        for p in out["providers"]:
            assert p["api_key"] == REDACTED
            assert p["base_url"].startswith("https://")
        assert out["active"] == "openai"

    def test_settings_export_payload(self):
        payload = {
            "ui": {"theme": "dark", "voice_enabled": True},
            "auth": {"session_secret": "s", "totp_secret": "t"},
            "google_pse_cx": "public-id",
            "google_pse_api_key": "private-key",
            "reminder_webhook_integration_id": "wh_123",
        }
        out = scrub_response(payload)
        assert out["ui"]["theme"] == "dark"
        assert out["auth"]["session_secret"] == REDACTED
        assert out["auth"]["totp_secret"] == REDACTED
        assert out["google_pse_cx"] == "public-id"
        assert out["google_pse_api_key"] == REDACTED
        assert out["reminder_webhook_integration_id"] == REDACTED
