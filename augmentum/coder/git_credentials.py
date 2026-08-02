"""Git credential token storage for coder mode workspaces.

Tokens stored in the app settings store with a 'git_token:' key prefix.
Each token is a JSON object: {host, username, token, created_at}.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    from augmentum.state.settings_store import SettingsStore

log = get_logger(__name__)

_KEY_PREFIX = "git_token:"


class GitTokenStore:
    """CRUD for git host tokens backed by SettingsStore."""

    def __init__(self, conn) -> None:
        from augmentum.state.settings_store import SettingsStore
        self._store = SettingsStore(conn)

    async def set_token(
        self,
        host: str,
        token: str,
        *,
        username: str = "oauth2",
        user_id: str = "",
    ) -> None:
        """Store a git token for a host (creates or overwrites).

        The token value is encrypted at rest via Fernet (AES-128-CBC).
        """
        data = {
            "host": host,
            "username": username,
            "token": encrypt_api_key(token),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        key = f"{_KEY_PREFIX}{host}"
        if user_id:
            await self._store.set_user(user_id, key, json.dumps(data))
        else:
            await self._store.set(key, json.dumps(data))
        log.info("git_token_stored", host=host, user_id=user_id or "(global)")

    async def get_token(self, host: str, *, user_id: str = "") -> dict | None:
        """Return full token data for a host, or None.

        The token value is decrypted from Fernet storage.
        Handles legacy plaintext tokens (backward compat).
        """
        key = f"{_KEY_PREFIX}{host}"
        raw = (
            await self._store.get_user(user_id, key)
            if user_id else None
        )
        if raw is None:
            raw = await self._store.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        # Decrypt the token (handles both encrypted and plaintext legacy)
        if data.get("token"):
            data["token"] = decrypt_api_key(data["token"])
        return data

    async def list_tokens(self, *, user_id: str = "") -> list[dict]:
        """Return all tokens with the actual token value redacted."""
        all_settings = (
            await self._store.get_all_user(user_id)
            if user_id else await self._store.get_all()
        )
        tokens = []
        for key, value in all_settings.items():
            if not key.startswith(_KEY_PREFIX):
                continue
            data = json.loads(value)
            tokens.append({
                "host": data["host"],
                "username": data.get("username", "oauth2"),
                "created_at": data.get("created_at", ""),
            })
        return tokens

    async def delete_token(self, host: str, *, user_id: str = "") -> None:
        """Remove a git token for a host."""
        key = f"{_KEY_PREFIX}{host}"
        if user_id:
            await self._store.set_user(user_id, key, None)
        else:
            await self._store.set(key, None)
        log.info("git_token_deleted", host=host, user_id=user_id or "(global)")
