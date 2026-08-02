"""Guest gateway — the browser-side-E2E envelope layer for portal guests.

The design journey (bug → Funnel → door physics → anonymized-tunnel reframe →
browser envelope) is recorded in
``docs/superpowers/specs/2026-07-16-guest-gateway-anonymous-tunnel-e2e-design.md``.
Read it before changing the trust chain.

Trust chain:

  QR/link fragment pins the instance Ed25519 identity (fabric identity —
  reused, never duplicated)
    → identity SIGNS the instance X25519 seal key (this module mints/holds it)
      → the seal key decrypts request envelopes sealed by the guest's device
        → the guest device's Ed25519 key (registered at portal confirm)
          signs every envelope, making the device key — bound to the guest
          account by the host's explicit confirm — the credential.

Ed25519↔X25519 birational conversion is deliberately NOT used (no supported
API in ``cryptography``; hand-rolling it is a foot-gun). Two durable keys,
one signature binding them.

Envelope wire format (v1) — JSON body with
``Content-Type: application/augmentum-envelope+json``::

    {"v": 1, "device_id": "...", "epk": b64, "nonce": b64, "ct": b64,
     "sig": b64}

* ``epk``   — fresh ephemeral X25519 public key per envelope (sender forward
              secrecy).
* KDF       — ECDH(epk, instance seal key) → HKDF-SHA256(salt=nonce,
              info=``augmentum-guest-env-v1``) → AES-256-GCM key.
* ``sig``   — Ed25519 over ``v|device_id|epk|nonce|ct`` (ascii, pipe-joined):
              request envelopes are signed by the DEVICE key; response
              envelopes by the INSTANCE identity.
* Replay    — server keeps a per-device sliding nonce window; the inner
              payload also carries ``ts`` (unix seconds) checked against
              ``MAX_SKEW_S``.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

log = get_logger(__name__)

ENVELOPE_CONTENT_TYPE = "application/augmentum-envelope+json"
_HKDF_INFO = b"augmentum-guest-env-v1"
_SEAL_SIG_CTX = b"augmentum-guest-seal-v1"
# Settings-store key for the instance X25519 seal key. Namespaced with the
# fabric identity keys so backup/restore tooling that snapshots ``fabric.*``
# carries the full trust chain.
_KEY_SEAL_PRIVATE = "fabric.node_seal_private_key"

MAX_SKEW_S = 120
_REPLAY_WINDOW_PER_DEVICE = 256


class EnvelopeError(ValueError):
    """Any envelope that must be refused (bad sig, replay, skew, format).

    One public error type; the log carries the specific reason. Callers
    return a uniform 400 so the error channel doesn't oracle the crypto.
    """


@dataclass(frozen=True)
class GatewayKeys:
    """The instance-side key material for the guest gateway."""

    identity: "FabricIdentity"
    seal_private: X25519PrivateKey

    @property
    def seal_public_b64(self) -> str:
        raw = self.seal_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")

    def bundle(self) -> dict:
        """The public gateway bundle the portal fetches and verifies.

        ``sig`` = identity over ``_SEAL_SIG_CTX || raw seal pub`` — the link
        in the chain from the QR-pinned identity to the seal key.
        """
        raw = base64.b64decode(self.seal_public_b64)
        sig = self.identity.sign(_SEAL_SIG_CTX + raw)
        return {
            "v": 1,
            "identity_pub": self.identity.public_key_b64,
            "identity_did": self.identity.did_key,
            "seal_pub": self.seal_public_b64,
            "sig": base64.b64encode(sig).decode("ascii"),
        }


async def get_gateway_keys(store: "SettingsStore") -> GatewayKeys:
    """Load (or mint on first use) the instance gateway keys.

    Reuses the durable fabric Ed25519 identity as the signing anchor; mints a
    durable X25519 seal key alongside it (same Fernet-at-rest pattern). Unlike
    the identity, a missing/corrupt seal key is NOT fail-closed: it only
    protects transport (guests re-pin nothing — the identity is the pin), so
    it may be re-minted, invalidating in-flight envelopes at worst.
    """
    from augmentum.fabric.identity import FabricIdentity

    identity = await FabricIdentity.from_settings_store(store)

    encrypted = await store.get(_KEY_SEAL_PRIVATE)
    seal_private: X25519PrivateKey | None = None
    if encrypted:
        raw_b64 = decrypt_api_key(encrypted)
        if raw_b64:
            try:
                seal_private = X25519PrivateKey.from_private_bytes(base64.b64decode(raw_b64))
            except Exception:
                log.warning("guest_gateway_seal_key_unreadable_reminting", exc_info=True)
    if seal_private is None:
        seal_private = X25519PrivateKey.generate()
        raw = seal_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        await store.set(_KEY_SEAL_PRIVATE, encrypt_api_key(base64.b64encode(raw).decode("ascii")))
        log.info("guest_gateway_seal_key_initialised")
    return GatewayKeys(identity=identity, seal_private=seal_private)


# ── envelope primitives ─────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: str, *, what: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except Exception as exc:
        raise EnvelopeError(f"bad base64 in {what}") from exc


def _sig_payload(v: int, device_id: str, epk: str, nonce: str, ct: str) -> bytes:
    return f"{v}|{device_id}|{epk}|{nonce}|{ct}".encode("ascii", errors="replace")


def _derive_key(shared: bytes, nonce: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=nonce, info=_HKDF_INFO,
    ).derive(shared)


def open_envelope(
    envelope: dict, *, keys: GatewayKeys, device_sign_pub_b64: str,
) -> dict:
    """Verify + decrypt a guest request envelope → the inner request dict.

    Raises :class:`EnvelopeError` on ANY failure. Timestamp skew is enforced
    here (the inner payload carries ``ts``); replay is the caller's job (it
    owns the per-device state).
    """
    if not isinstance(envelope, dict) or envelope.get("v") != 1:
        raise EnvelopeError("unsupported envelope version")
    device_id = str(envelope.get("device_id", "") or "")
    epk_s = str(envelope.get("epk", "") or "")
    nonce_s = str(envelope.get("nonce", "") or "")
    ct_s = str(envelope.get("ct", "") or "")
    sig_s = str(envelope.get("sig", "") or "")
    if not (device_id and epk_s and nonce_s and ct_s and sig_s):
        raise EnvelopeError("missing envelope fields")

    sig = _unb64(sig_s, what="sig")
    try:
        pub = Ed25519PublicKey.from_public_bytes(_unb64(device_sign_pub_b64, what="device key"))
        pub.verify(sig, _sig_payload(1, device_id, epk_s, nonce_s, ct_s))
    except EnvelopeError:
        raise
    except Exception as exc:
        raise EnvelopeError("bad envelope signature") from exc

    epk = _unb64(epk_s, what="epk")
    nonce = _unb64(nonce_s, what="nonce")
    ct = _unb64(ct_s, what="ct")
    if len(nonce) != 12:
        raise EnvelopeError("bad nonce length")
    try:
        shared = keys.seal_private.exchange(X25519PublicKey.from_public_bytes(epk))
        plain = AESGCM(_derive_key(shared, nonce)).decrypt(nonce, ct, None)
        inner = json.loads(plain.decode("utf-8"))
    except Exception as exc:
        raise EnvelopeError("envelope decrypt failed") from exc
    if not isinstance(inner, dict):
        raise EnvelopeError("inner payload is not an object")

    ts = inner.get("ts")
    if not isinstance(ts, int | float) or abs(time.time() - float(ts)) > MAX_SKEW_S:
        raise EnvelopeError("envelope timestamp outside window")
    return inner


def seal_to_device(
    inner: dict, *, keys: GatewayKeys, device_seal_pub_b64: str, device_id: str,
) -> dict:
    """Seal a response payload to the guest device; signed by the identity."""
    eph = X25519PrivateKey.generate()
    epk_raw = eph.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    )
    nonce = os.urandom(12)
    shared = eph.exchange(
        X25519PublicKey.from_public_bytes(base64.b64decode(device_seal_pub_b64))
    )
    ct = AESGCM(_derive_key(shared, nonce)).encrypt(
        nonce, json.dumps(inner, separators=(",", ":")).encode("utf-8"), None,
    )
    epk_s, nonce_s, ct_s = _b64(epk_raw), _b64(nonce), _b64(ct)
    sig = keys.identity.sign(_sig_payload(1, device_id, epk_s, nonce_s, ct_s))
    return {
        "v": 1, "device_id": device_id,
        "epk": epk_s, "nonce": nonce_s, "ct": ct_s, "sig": _b64(sig),
    }


class ReplayGuard:
    """Per-device sliding nonce window (in-memory; envelopes also carry ``ts``
    so a restart's forgotten window is bounded by ``MAX_SKEW_S``)."""

    def __init__(self, *, per_device: int = _REPLAY_WINDOW_PER_DEVICE, max_devices: int = 4096) -> None:
        self._per_device = per_device
        self._devices: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
        self._max_devices = max_devices

    def check_and_record(self, device_id: str, nonce_b64: str) -> bool:
        """False (and no record) when the nonce was already seen."""
        seen = self._devices.get(device_id)
        if seen is None:
            seen = OrderedDict()
            self._devices[device_id] = seen
            self._devices.move_to_end(device_id)
            while len(self._devices) > self._max_devices:
                self._devices.popitem(last=False)
        if nonce_b64 in seen:
            return False
        seen[nonce_b64] = None
        while len(seen) > self._per_device:
            seen.popitem(last=False)
        return True


# ── device key record helpers ───────────────────────────────────────────────


def parse_device_public_key(raw: str) -> dict | None:
    """Parse the portal's registered ``device_public_key`` JSON.

    ``{"v": 1, "sign_pub": b64-ed25519, "seal_pub": b64-x25519}``. Returns
    None for empty/legacy (pre-gateway guests registered an empty string —
    they keep working, plain; the Guests panel shows them as legacy).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except Exception:
        return None
    if not isinstance(rec, dict) or rec.get("v") != 1:
        return None
    sign_pub = str(rec.get("sign_pub", "") or "")
    seal_pub = str(rec.get("seal_pub", "") or "")
    if not sign_pub or not seal_pub:
        return None
    try:
        if len(base64.b64decode(sign_pub, validate=True)) != 32:
            return None
        if len(base64.b64decode(seal_pub, validate=True)) != 32:
            return None
    except Exception:
        return None
    return {"sign_pub": sign_pub, "seal_pub": seal_pub}


# The route sub-tree an enveloped request may dispatch to. Deny-by-default:
# the auth middleware's guest allow-list ALSO applies underneath (the inner
# dispatch authenticates as the guest user), so this list is defense-in-depth
# scoping what the envelope endpoint will even attempt.
ENVELOPE_DISPATCH_PREFIXES: tuple[str, ...] = (
    "/api/portal/me",
    "/api/connect/",
    "/api/auth/status",
    "/api/auth/logout",
    "/api/notify/",
)
ENVELOPE_DISPATCH_METHODS: frozenset[str] = frozenset({"GET", "POST", "PUT", "DELETE"})


def dispatch_allowed(method: str, path: str) -> bool:
    if method.upper() not in ENVELOPE_DISPATCH_METHODS:
        return False
    p = (path or "").split("?", 1)[0]
    if p.startswith("/api/portal/env") or p.startswith("/api/portal/gateway"):
        return False  # no envelope-in-envelope
    return any(p == pre.rstrip("/") or p.startswith(pre) for pre in ENVELOPE_DISPATCH_PREFIXES)
