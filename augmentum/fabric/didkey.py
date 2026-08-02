"""did:key encoding for fabric identities (Ed25519).

The fabric identity (``identity.py``) is an Ed25519 keypair. For
federation we need a *canonical, full-key* identifier that two
instances can byte-compare, pin, and block on — the existing
``SHA256:`` fingerprint is a one-way hash (good for humans to read
aloud, useless for re-deriving or verifying a key). did:key is that
identifier: it carries the entire 32-byte public key in a single
self-describing string.

Format (W3C did:key method, base58btc multibase):

    did:key:z<base58btc( <multicodec-varint> || <raw-pubkey-bytes> )>

  * ``z`` is the multibase prefix for base58btc.
  * the multicodec varint binds the *curve* to the bytes:
      - ``0xed01`` → Ed25519 public key (what we mint)
      - ``0xec01`` → X25519 public key (a future sealing key; REJECTED
        by :func:`decode_ed25519_did` so an X25519 did can never be
        mistaken for a signing identity — the curve-confusion class).

Security contract (the second red-team's D1-05 finding):
**trust comparison is ALWAYS on the decoded 32 bytes, never on the
string.** Two different strings must never be treated as two different
identities if they decode to the same key, and one string must never
satisfy a pin/block check for a different key. :func:`did_equal` is the
only correct comparator; callers must not ``==`` the raw strings.

Pure module: stdlib + a ~20-line base58 codec, no new dependency.
"""

from __future__ import annotations

# Multicodec unsigned-varint prefixes. These two are single-byte codes
# whose LEB128 varint encoding is two bytes each.
_ED25519_PUB_PREFIX = b"\xed\x01"  # multicodec 0xed "ed25519-pub"
_X25519_PUB_PREFIX = b"\xec\x01"  # multicodec 0xec "x25519-pub"

_DID_KEY_PREFIX = "did:key:z"

# Bitcoin/IPFS base58btc alphabet (same ordering libp2p/did:key use).
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def _b58encode(data: bytes) -> str:
    """Base58btc-encode raw bytes (leading-zero-byte preserving)."""
    n = int.from_bytes(data, "big")
    out: list[str] = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    # Each leading 0x00 byte encodes as a leading '1'.
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + "".join(reversed(out))


def _b58decode(s: str) -> bytes:
    """Base58btc-decode a string to raw bytes. Raises ValueError on a
    character outside the alphabet."""
    n = 0
    for ch in s:
        idx = _B58_INDEX.get(ch)
        if idx is None:
            raise ValueError(f"invalid base58 character {ch!r}")
        n = n * 58 + idx
    # Recover leading zero bytes from leading '1's.
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    return b"\x00" * pad + body


def encode_ed25519_did(pub_raw: bytes) -> str:
    """32 raw Ed25519 public-key bytes → canonical ``did:key:z...``.

    Single canonical output form for any given key.
    """
    if len(pub_raw) != 32:
        raise ValueError(f"ed25519 public key must be 32 bytes, got {len(pub_raw)}")
    return _DID_KEY_PREFIX + _b58encode(_ED25519_PUB_PREFIX + pub_raw)


def decode_ed25519_did(did: str) -> bytes:
    """``did:key:z...`` → 32 raw Ed25519 public-key bytes.

    Codec→curve bound: a did:key carrying any non-Ed25519 multicodec
    (e.g. an X25519 ``0xec01`` key) is **rejected**, not silently
    returned — closing the curve-confusion / two-strings-one-key
    evasion (D1-05). Raises ValueError on any malformed input.
    """
    if not isinstance(did, str) or not did.startswith(_DID_KEY_PREFIX):
        raise ValueError("not a base58btc did:key")
    decoded = _b58decode(did[len(_DID_KEY_PREFIX):])
    if decoded[:2] == _X25519_PUB_PREFIX:
        raise ValueError("did:key is X25519, not a valid signing identity")
    if decoded[:2] != _ED25519_PUB_PREFIX:
        raise ValueError(f"unsupported did:key multicodec {decoded[:2].hex()!r}")
    raw = decoded[2:]
    if len(raw) != 32:
        raise ValueError(f"ed25519 key body must be 32 bytes, got {len(raw)}")
    return raw


def is_ed25519_did(did: str) -> bool:
    """True iff ``did`` is a well-formed Ed25519 did:key."""
    try:
        decode_ed25519_did(did)
        return True
    except ValueError:
        return False


def did_equal(a: str, b: str) -> bool:
    """Identity comparison on decoded key BYTES, never on strings.

    The ONLY correct way to compare/pin/block two did:keys. Returns
    False (not raises) if either side is malformed, so it is safe to
    call on untrusted wire input.
    """
    try:
        return decode_ed25519_did(a) == decode_ed25519_did(b)
    except ValueError:
        return False
