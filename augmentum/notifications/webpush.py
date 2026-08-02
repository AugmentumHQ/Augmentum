"""Web Push delivery + VAPID key lifecycle.

The browser-side Notification API can subscribe a page to a push
service (FCM / Mozilla autopush / Apple Push). The push service holds
the user's endpoint and forwards encrypted payloads from any server
that signs them with the matching VAPID private key.

This module owns the server-side half:

* ``ensure_vapid_keys`` — lazy-generates and persists a single
  application-wide VAPID keypair in ``app_settings``. The PUBLIC key
  is served to the browser so it can subscribe; the PRIVATE key
  never leaves the server.

* ``send_webpush`` — encrypts + dispatches one push to one
  subscription. Handles the 410 GONE response (subscription expired
  on the push service side) by returning a sentinel so the
  ``NotificationHub`` can prune the row.

Why a single application-wide keypair (not per-user): VAPID identifies
the SERVER, not the user. Browsers cache subscriptions keyed on the
applicationServerKey, so rotating the key invalidates every existing
subscription. One process-wide keypair is the standard model.

Storage: ``app_settings`` because the keys are server-level state and
shouldn't be per-user-scoped. They survive process restart; rotation
is a deliberate operator action, not automatic.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass


log = get_logger(__name__)


# Stored under these keys in app_settings.
SETTING_VAPID_PUBLIC = "webpush.vapid_public_key_b64url"
SETTING_VAPID_PRIVATE = "webpush.vapid_private_key_b64url"
SETTING_VAPID_SUBJECT = "webpush.vapid_subject"


# RFC 8292 says the VAPID ``sub`` claim must be a contact URL or
# mailto. Browsers don't enforce it strictly but push services log
# it for abuse outreach, so a stable mailto-style placeholder beats
# leaving it blank or per-deploy-random.
DEFAULT_SUBJECT = "mailto:webpush@augmentum.local"


@dataclass(frozen=True)
class VapidKeys:
    """One application-wide VAPID keypair.

    Both fields are base64url-encoded (no padding) per RFC 7515 — the
    same encoding ``pywebpush`` and browsers expect. The public form
    is what the browser sends to the push service in
    ``pushManager.subscribe({applicationServerKey})``.
    """

    public_b64url: str
    private_b64url: str
    subject: str = DEFAULT_SUBJECT


# ── Key generation + persistence ─────────────────────────────────


def _generate_keypair() -> tuple[str, str]:
    """Create a fresh P-256 keypair and return ``(public, private)``
    base64url-encoded strings."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())

    # Private key: raw scalar bytes (32 bytes), b64url for pywebpush.
    private_numbers = private_key.private_numbers()
    private_bytes = private_numbers.private_value.to_bytes(32, "big")

    # Public key: uncompressed point (0x04 + X + Y = 65 bytes), which
    # is the form browsers expect for applicationServerKey.
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    return _b64url(public_bytes), _b64url(private_bytes)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


async def _read_setting(conn: Any, key: str) -> str:
    cur = await conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (key,),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return ""
    return str(row[0] or "")


async def _write_setting(conn: Any, key: str, value: str) -> None:
    # app_settings has (key, value) shape with key as primary key —
    # UPSERT to either insert or replace.
    await conn.execute(
        "INSERT INTO app_settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    await conn.commit()


async def ensure_vapid_keys(conn: Any) -> VapidKeys:
    """Read the persisted VAPID keypair, generating one on first use.

    Idempotent: subsequent calls return the existing keys. Generation
    happens at most once per database; rotation is operator-driven
    (delete the rows + restart).
    """

    public = await _read_setting(conn, SETTING_VAPID_PUBLIC)
    private = await _read_setting(conn, SETTING_VAPID_PRIVATE)
    subject = await _read_setting(conn, SETTING_VAPID_SUBJECT) or DEFAULT_SUBJECT

    if public and private:
        return VapidKeys(public_b64url=public, private_b64url=private, subject=subject)

    public, private = _generate_keypair()
    await _write_setting(conn, SETTING_VAPID_PUBLIC, public)
    await _write_setting(conn, SETTING_VAPID_PRIVATE, private)
    await _write_setting(conn, SETTING_VAPID_SUBJECT, subject)
    log.info("webpush_vapid_keys_generated", subject=subject)
    return VapidKeys(public_b64url=public, private_b64url=private, subject=subject)


async def get_public_key(conn: Any) -> str:
    """Return the application's VAPID public key (b64url). Generates
    if absent — safe to call from a route handler."""

    keys = await ensure_vapid_keys(conn)
    return keys.public_b64url


# ── Send path ────────────────────────────────────────────────────


@dataclass
class WebPushSendResult:
    """Outcome of one push send.

    ``status`` is the HTTP status the push service returned.
    ``expired`` is True for 404 / 410 responses — those mean "this
    subscription is dead, prune it." The caller's clean-up loop
    deletes the matching ``notification_subscriptions`` row.
    """

    status: int = 0
    expired: bool = False
    error: str = ""


def send_webpush(
    *,
    endpoint: str,
    p256dh: str,
    auth: str,
    payload: dict[str, Any],
    vapid: VapidKeys,
    ttl_seconds: int = 3600,
) -> WebPushSendResult:
    """Synchronously send one encrypted push.

    Synchronous because ``pywebpush`` uses ``requests`` under the
    hood. Callers from async code should ``run_in_executor`` (or
    ``asyncio.to_thread``) so the FastAPI loop stays responsive.

    The payload is a JSON-serialisable dict. Keep it small — push
    services typically cap at 4 KB encrypted, and the encryption
    adds overhead. For Connect, ship: title, body, icon, url to
    open on click, plus the notification_id so the SW can dedupe.
    """

    try:
        from pywebpush import WebPushException, webpush
    except ImportError as exc:
        return WebPushSendResult(
            status=0, error=f"pywebpush not installed: {exc}",
        )

    subscription_info = {
        "endpoint": endpoint,
        "keys": {"p256dh": p256dh, "auth": auth},
    }
    vapid_claims = {"sub": vapid.subject}

    try:
        resp = webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload, separators=(",", ":")),
            vapid_private_key=vapid.private_b64url,
            vapid_claims=vapid_claims,
            ttl=ttl_seconds,
        )
        status = getattr(resp, "status_code", 201)
        return WebPushSendResult(status=int(status), expired=False)
    except WebPushException as exc:
        status = 0
        response = getattr(exc, "response", None)
        if response is not None:
            status = int(getattr(response, "status_code", 0) or 0)
        # Per RFC 8030: 404 = endpoint never existed; 410 = subscription
        # explicitly invalidated by the user / browser. Both mean "stop
        # sending to this endpoint."
        expired = status in (404, 410)
        return WebPushSendResult(
            status=status,
            expired=expired,
            error=str(exc)[:240],
        )
    except Exception as exc:
        # Network / DNS / TLS / unexpected failure — keep the
        # subscription row, surface the error so the caller can log.
        return WebPushSendResult(status=0, error=str(exc)[:240])
