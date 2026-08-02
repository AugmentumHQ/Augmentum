"""Per-user, encrypted-at-rest storage for the Claude credential.

The user logs in once (``claude setup-token`` → ``sk-ant-oat01-…`` subscription
token, or an ``sk-ant-api…`` key) and Augmentum stores it ENCRYPTED, scoped to
the user, in the ``user_settings`` table (via SettingsStore). The driver loads
it at spawn time and hands it to the Claude-home container via env — never on a
command line, never in logs. This is the cred-at-rest requirement from the
external-coder design (§3.6.1).

Pure-ish + testable: takes a SettingsStore-like object with async
``get_user`` / ``set_user``; tests inject a fake.
"""

from __future__ import annotations

from typing import Any

from augmentum.coder.external.claude_auth import is_oauth_token
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

_KEY = "claude_code_oauth_token"


def looks_like_claude_credential(token: str) -> bool:
    """A Claude subscription OAuth token or an Anthropic API key."""
    t = (token or "").strip()
    return t.startswith("sk-ant-")


async def save_token(settings_store: Any, user_id: str, token: str) -> None:
    """Encrypt + persist the credential for ``user_id``. Raises ValueError on
    empty input."""
    token = (token or "").strip()
    if not user_id:
        raise ValueError("user_id required")
    if not token:
        raise ValueError("token required")
    await settings_store.set_user(user_id, _KEY, encrypt_api_key(token) or "")


async def load_token(settings_store: Any, user_id: str) -> str:
    """Decrypt + return the stored credential, or "" if none."""
    if settings_store is None or not user_id:
        return ""
    raw = await settings_store.get_user(user_id, _KEY)
    if not raw:
        return ""
    return decrypt_api_key(raw) or ""


async def clear_token(settings_store: Any, user_id: str) -> None:
    """Forget the stored credential (disconnect)."""
    if settings_store is None or not user_id:
        return
    await settings_store.set_user(user_id, _KEY, None)


def _mask(token: str) -> str:
    """A safe, non-secret hint to show the user it's set."""
    if len(token) > 16:
        return f"{token[:10]}…{token[-4:]}"
    return "set" if token else ""


async def status(settings_store: Any, user_id: str) -> dict:
    """Connection status for the UI — never returns the raw token."""
    tok = await load_token(settings_store, user_id)
    if not tok:
        return {"connected": False, "kind": "", "hint": ""}
    if is_oauth_token(tok):
        kind = "subscription"
    elif tok.startswith("sk-ant-"):
        kind = "api_key"
    else:
        kind = "unknown"
    return {"connected": True, "kind": kind, "hint": _mask(tok)}
