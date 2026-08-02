"""Session management — FastAPI dependency for extracting/creating sessions."""

from __future__ import annotations

import hashlib
import uuid

from fastapi import Request

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Header name for explicit session identification
SESSION_HEADER = "X-Augmentum-Session"


def get_client_id(request: Request) -> str:
    """Extract client identity from request headers or IP.

    Priority:
    1. X-Augmentum-Client-ID header (explicit)
    2. X-Forwarded-For header (proxy)
    3. Client IP address
    """
    explicit = request.headers.get("X-Augmentum-Client-ID", "")
    if explicit:
        return explicit
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return ""


async def get_session_id(request: Request) -> str:
    """Extract or generate a session ID from the request.

    Priority:
    1. X-Augmentum-Session header (explicit)
    2. Deterministic fingerprint from system prompt + first user message
    3. Random UUID fallback

    When ``session_client_isolation`` is enabled, the client identity is
    prefixed to the hash input so that the same system prompt from different
    clients produces different session IDs.
    """
    from augmentum.config import settings

    # 1. Explicit header
    header_value = request.headers.get(SESSION_HEADER)
    if header_value:
        return header_value

    # 2. Deterministic fingerprint from request body
    try:
        body = await request.json()
        messages = body.get("messages", [])
        if messages:
            # Hash system prompt (if present) + first user message
            parts = []
            for msg in messages:
                if msg.get("role") == "system":
                    parts.append(msg.get("content", ""))
                    break
            for msg in messages:
                if msg.get("role") == "user":
                    parts.append(msg.get("content", ""))
                    break

            if parts:
                content = "|".join(parts)
                if settings.session_client_isolation:
                    client_id = get_client_id(request)
                    if client_id:
                        content = f"{client_id}:{content}"
                fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]
                return f"ses_{fingerprint}"
    except Exception:
        log.debug("session_id_derivation_failed", exc_info=True)

    # 3. Random UUID fallback
    return f"ses_{uuid.uuid4().hex[:16]}"


def derive_kv_session_key(user_id: str, session_id: str) -> str:
    """Opaque, user-scoped key for KV-slot bookkeeping.

    Folding ``user_id`` into the digest scopes the key per-user (so two
    users with the same character card don't collide on a shared slot
    file) without storing any raw identifier in the manifest schema or
    in log lines. Empty ``user_id`` (auth disabled) maps to a single
    "_anon" bucket.
    """
    if not session_id:
        return ""
    owner = user_id or "_anon"
    raw = f"u:{owner}|s:{session_id}"
    return "kv_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
