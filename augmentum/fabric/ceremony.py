"""Out-of-band verification ceremony — safety number + SAS (P1).

A parsed contact card is pinned but *unverified*: a malicious host could
have minted it for a key it controls. The ceremony is how two humans
confirm, over a channel the attacker doesn't control (a voice call, an
in-person QR scan), that the key each side pinned for the other is the
RIGHT key.

This is the Signal "safety number" model rendered two ways:

  * **Safety number** (text/QR channel): a deterministic commitment over
    BOTH sides' identities. Both instances compute the identical string
    from the two did:keys (+ the two author keys); scanning the QR or
    reading the string and finding them equal proves no key was
    substituted in transit. A mismatch = MITM / wrong pin.
  * **SAS words** (voice channel): the same commitment rendered as 4
    BIP39 words for people to read aloud on a call.

Order-independence is load-bearing: each side plugs in (self, peer) in a
different order, so we SORT the two identities before hashing. Both sides
get the same number regardless of who calls whom.

AK-1 fix — the **author key is folded into the commitment.** If a
malicious host substitutes the per-user author key, the safety number /
SAS changes, so the voice/QR check catches it. Omitting the author key
would let the host swap it invisibly after the ceremony.
"""

from __future__ import annotations

import hashlib

from augmentum.fabric.didkey import decode_ed25519_did
from augmentum.fabric.recovery import _wordlist

# Number of BIP39 words in the spoken SAS. 4 words from a 2048-word list
# = 44 bits of commitment — far beyond what a live attacker can grind
# inside a single verification call, while staying short enough to read
# aloud. (Signal uses 60 digits for the full number; the spoken SAS is
# deliberately a shorter human-checkable digest of the same commitment.)
_SAS_WORDS = 4
_SAS_BITS = _SAS_WORDS * 11  # 44


def _identity_blob(did_key: str, author_did_key: str) -> bytes:
    """Canonical bytes for ONE side: raw did:key bytes || raw author bytes.

    Decoding to raw bytes (not trusting the string form) means two
    different string encodings of the same key produce the same blob —
    the commitment is over KEY MATERIAL, not text.
    """
    raw = decode_ed25519_did(did_key)
    author_raw = decode_ed25519_did(author_did_key) if author_did_key else b""
    return raw + b"\x00" + author_raw


def _commitment(
    self_did: str,
    self_author: str,
    peer_did: str,
    peer_author: str,
) -> bytes:
    """Order-independent SHA-256 commitment over both identities."""
    a = _identity_blob(self_did, self_author)
    b = _identity_blob(peer_did, peer_author)
    lo, hi = sorted([a, b])  # sort by raw bytes → same on both sides
    return hashlib.sha256(b"augmentum-fabric-sas-v1" + lo + hi).digest()


def safety_number(
    self_did: str,
    peer_did: str,
    *,
    self_author: str = "",
    peer_author: str = "",
) -> str:
    """Full safety-number string for the text/QR channel.

    60 decimal digits grouped in fives (Signal-style), derived from the
    commitment. Two sides that pinned the same keys produce the same
    string; a substituted key changes it.
    """
    digest = _commitment(self_did, self_author, peer_did, peer_author)
    n = int.from_bytes(digest, "big")
    digits = f"{n % (10 ** 60):060d}"
    return " ".join(digits[i:i + 5] for i in range(0, 60, 5))


def sas_words(
    self_did: str,
    peer_did: str,
    *,
    self_author: str = "",
    peer_author: str = "",
) -> list[str]:
    """4 BIP39 words for the spoken (voice-call) channel.

    Same commitment as :func:`safety_number`, rendered as words people
    read to each other. Equal on both sides ⇔ no key substitution.
    """
    digest = _commitment(self_did, self_author, peer_did, peer_author)
    n = int.from_bytes(digest, "big")
    bits = n % (1 << _SAS_BITS)
    words = _wordlist()
    out: list[str] = []
    for i in range(_SAS_WORDS):
        idx = (bits >> (11 * (_SAS_WORDS - 1 - i))) & 0x7FF
        out.append(words[idx])
    return out


def verify_match(
    self_did: str,
    peer_did: str,
    presented: str,
    *,
    self_author: str = "",
    peer_author: str = "",
) -> bool:
    """True iff ``presented`` (spoken words OR a scanned safety number)
    matches what this side computes — i.e. the ceremony passed.

    Accepts either the space-joined SAS words or the safety-number
    string; normalises whitespace/case so a human-transcribed value
    still compares cleanly.
    """
    presented_norm = " ".join(presented.lower().split())
    expected_words = " ".join(
        sas_words(self_did, peer_did, self_author=self_author, peer_author=peer_author)
    )
    if presented_norm == expected_words:
        return True
    expected_number = safety_number(
        self_did, peer_did, self_author=self_author, peer_author=peer_author
    )
    # Compare digit-only forms so grouping/spacing differences don't fail.
    return _digits(presented) == _digits(expected_number)


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())
