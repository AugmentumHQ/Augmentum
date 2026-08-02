"""Fabric node identity: stable ID + ed25519 keypair.

Loaded lazily from the settings store on first request. Auto-generates
on first call and persists for future restarts. The private key is
encrypted at rest using the same Fernet helper used for provider API
keys.

Identity is shown in the UI as a short SSH-style fingerprint
(``SHA256:abcd...wxyz``); two operators read fingerprints to each other
when pairing peers. Identical to the SSH host-key verification model.

This module is pure Python with no side effects on import. The keypair
is only generated when ``FabricIdentity.from_settings_store(store)`` is
awaited explicitly -- meaning a solo install with ``fabric_enabled =
False`` never instantiates an identity at all.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    from augmentum.state.settings_store import SettingsStore

log = get_logger(__name__)

# Settings-store keys. Namespaced under ``fabric.*`` so they sort
# together in admin tools and don't collide with any future feature.
_KEY_NODE_ID = "fabric.node_id"
_KEY_PRIVATE_KEY = "fabric.node_private_key"


class FabricIdentityCorruptError(RuntimeError):
    """Stored fabric identity exists but is unreadable.

    Raised by :meth:`FabricIdentity.from_settings_store` when prior
    identity state is present in the settings store but cannot be
    decrypted/parsed, OR when only one half of the (node_id, key) pair
    is present (a torn write / partial wipe).

    We deliberately FAIL CLOSED rather than mint a replacement key:
    regenerating under a preserved ``node_id`` would silently rotate the
    instance's federated identity, and every peer that pinned the old
    key would keep trusting an identity the operator never chose. The
    operator must instead restore from the 24-word BIP39 backup
    (``recovery.py``) or an atomic ``data_dir`` snapshot. The lifespan
    hook catches this and degrades fabric, not the whole app.
    """


@dataclass(frozen=True)
class FabricIdentity:
    """Durable identity for this Augmentum instance in a fabric.

    Constructed via :meth:`from_settings_store` -- direct ``__init__``
    is reserved for unit tests that want a deterministic identity. The
    keypair is generated once on first call and persisted; subsequent
    calls return the same identity bit-for-bit.
    """

    node_id: str
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @classmethod
    async def from_settings_store(cls, store: SettingsStore) -> FabricIdentity:
        """Load or generate the fabric identity from the settings store.

        On first call, generates a new node_id + ed25519 keypair,
        encrypts the private key, and persists both. On subsequent
        calls, returns the previously-persisted identity unchanged.
        """
        node_id = await store.get(_KEY_NODE_ID)
        encrypted_priv = await store.get(_KEY_PRIVATE_KEY)

        if node_id and encrypted_priv:
            priv_bytes_b64 = decrypt_api_key(encrypted_priv)
            if priv_bytes_b64:
                try:
                    priv_bytes = base64.b64decode(priv_bytes_b64)
                    private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
                    return cls(
                        node_id=node_id,
                        private_key=private_key,
                        public_key=private_key.public_key(),
                    )
                except Exception as exc:
                    # FAIL CLOSED. Prior identity exists but its key is
                    # unreadable. Regenerating would silently rotate this
                    # instance's federated identity under the SAME node_id
                    # and break every peer that pinned the old key. Halt
                    # loudly; the operator restores from BIP39 backup or a
                    # data_dir snapshot. (v2 finding #5 / RC-4.)
                    log.error(
                        "fabric_identity_corrupt_halting",
                        node_id=node_id,
                        exc_info=True,
                    )
                    raise FabricIdentityCorruptError(
                        "stored fabric identity is present but its key could "
                        "not be parsed; refusing to silently regenerate. "
                        "Restore from the 24-word backup or a data_dir snapshot."
                    ) from exc
            # node_id + ciphertext present, but decrypt returned None
            # (wrong/rotated Fernet key, truncated value). Also corrupt.
            log.error("fabric_identity_undecryptable_halting", node_id=node_id)
            raise FabricIdentityCorruptError(
                "stored fabric private key failed to decrypt; refusing to "
                "regenerate (fail-closed). Restore from backup."
            )

        if node_id or encrypted_priv:
            # Exactly one half present = torn write / partial wipe. This
            # is NOT a clean first boot; minting a fresh key under a stale
            # node_id (or a node_id under an orphan key) is the same
            # silent-rotation hazard. Halt.
            log.error(
                "fabric_identity_half_present_halting",
                has_node_id=bool(node_id),
                has_key=bool(encrypted_priv),
            )
            raise FabricIdentityCorruptError(
                "fabric identity is half-present (torn state); refusing to "
                "regenerate. Restore from backup."
            )

        # TRUE first boot: neither node_id nor key exists. Mint fresh.
        new_node_id = secrets.token_hex(16)
        new_private = Ed25519PrivateKey.generate()
        priv_bytes = new_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        encrypted = encrypt_api_key(base64.b64encode(priv_bytes).decode("ascii"))

        await store.set(_KEY_NODE_ID, new_node_id)
        await store.set(_KEY_PRIVATE_KEY, encrypted)

        log.info(
            "fabric_identity_initialised",
            node_id=new_node_id,
            fingerprint=_fingerprint_from_public_key(new_private.public_key()),
        )

        return cls(
            node_id=new_node_id,
            private_key=new_private,
            public_key=new_private.public_key(),
        )

    @property
    def public_key_bytes(self) -> bytes:
        """Raw 32-byte ed25519 public key."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_b64(self) -> str:
        """Base64-encoded public key, suitable for wire/storage."""
        return base64.b64encode(self.public_key_bytes).decode("ascii")

    @property
    def did_key(self) -> str:
        """Canonical ``did:key:z...`` form of this identity's public key.

        The full-key, byte-comparable federated identifier (vs the
        one-way :attr:`fingerprint`). Peers pin and compare on this via
        :func:`augmentum.fabric.didkey.did_equal`.
        """
        from augmentum.fabric.didkey import encode_ed25519_did

        return encode_ed25519_did(self.public_key_bytes)

    def mnemonic_backup(self) -> str:
        """24-word BIP39 phrase encoding this identity's private key.

        The only recovery path for a lost key. Shown to the operator
        once at fabric-enable; never persisted in plaintext.
        """
        from cryptography.hazmat.primitives import serialization

        from augmentum.fabric.recovery import key_to_mnemonic

        priv_raw = self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return key_to_mnemonic(priv_raw)

    @property
    def fingerprint(self) -> str:
        """Short SSH-style fingerprint for human verification.

        Returns ``"SHA256:<32-char-hex>"``. Operators compare these
        two values out-of-band during pairing.
        """
        return _fingerprint_from_public_key(self.public_key)

    def sign(self, payload: bytes) -> bytes:
        """Sign arbitrary bytes with this identity's private key.

        Used in higher phases for peer handshakes and (optionally)
        per-message signing. Returns a raw 64-byte ed25519 signature.
        """
        return self.private_key.sign(payload)

    @staticmethod
    def verify(payload: bytes, signature: bytes, public_key_b64: str) -> bool:
        """Verify a signature against a base64-encoded public key.

        Returns True iff the signature is valid for the given payload
        under the given public key. Catches all exceptions and returns
        False -- callers should not need to handle cryptography errors.
        """
        try:
            pub_bytes = base64.b64decode(public_key_b64)
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub.verify(signature, payload)
            return True
        except Exception:
            return False


def _fingerprint_from_public_key(public_key: Ed25519PublicKey) -> str:
    """Compute the SSH-style fingerprint for a public key.

    Uses SHA-256 of the raw public key bytes, hex-truncated to 32
    chars. Short enough to read out loud, long enough to make
    collisions infeasible (128 bits).
    """
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    digest = hashlib.sha256(raw).hexdigest()
    return f"SHA256:{digest[:32]}"
