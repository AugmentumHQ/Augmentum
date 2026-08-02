"""Bounded, accessible proof-of-work for cold-contact intake (P4).

PoW is a cost on the abuser's side of cold contact (knocks, directory
PUTs, tip-line intake). The second red-team flagged it as an
**accessibility/exclusion harm** if unbounded: a hard PoW taxes exactly
the old/throttled/battery-constrained phones at-risk and global-south
users carry. So this implementation is deliberately constrained:

  * **Target-bound + expiring.** A challenge names what it's for
    (``target``) and an ``issued_at``; it expires, so a solution can't be
    precomputed or replayed against a different target.
  * **Difficulty-capped.** ``MAX_DIFFICULTY_BITS`` keeps a solve under a
    few seconds on a low-end phone. We never accept a request to mint a
    harder challenge than the cap.
  * **Can't-compute fallback.** ``difficulty=0`` is legal and verifies
    trivially — the policy layer may waive PoW for an accessibility
    path. PoW is one signal, never the sole gate.

PoW only has teeth if the CHALLENGE is server-issued and the server won't
accept a client-minted one (SEC-7): otherwise an attacker just makes a
``difficulty=0`` challenge and "solves" it for free. So the intake path
must use the **signed** challenge helpers — ``sign_challenge`` /
``open_signed_challenge`` — which bind the challenge to the issuing
instance's Ed25519 key. A single-use guard (``ConsumedNonces``) stops one
solved challenge being replayed for many submissions.

Pure stdlib for the PoW math (sha256); the signed wrapper uses the same
pyca Ed25519 the rest of fabric uses. No new dependency.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did, did_equal

# A solve at this difficulty is ~2^18 hashes ≈ well under 3s on a weak
# phone. The cap is a hard ceiling — we refuse to demand more.
MAX_DIFFICULTY_BITS = 20
DEFAULT_TTL_S = 300
_CHALLENGE_CTX = "augmentum-fabric-pow-challenge-v1"  # domain separation


class PowChallengeError(ValueError):
    """Raised when a signed challenge is malformed, forged, expired, or
    over the difficulty cap — i.e. not a genuine server-issued challenge."""


@dataclass(frozen=True)
class PowChallenge:
    target: str          # what this PoW authorises (e.g. a recipient did:key)
    nonce: str           # server-chosen, unguessable
    difficulty: int      # required leading zero BITS of sha256(challenge||sol)
    issued_at: int
    ttl_s: int = DEFAULT_TTL_S

    def _prefix(self) -> bytes:
        return f"{self.target}|{self.nonce}|{self.difficulty}|{self.issued_at}".encode()


def make_challenge(
    *, target: str, issued_at: int, difficulty: int = 12,
) -> PowChallenge:
    """Mint a target-bound challenge. ``difficulty`` is clamped to
    ``MAX_DIFFICULTY_BITS`` (never harder than the accessibility cap)."""
    difficulty = max(0, min(int(difficulty), MAX_DIFFICULTY_BITS))
    return PowChallenge(
        target=target,
        nonce=os.urandom(12).hex(),
        difficulty=difficulty,
        issued_at=int(issued_at),
    )


def _leading_zero_bits(digest: bytes) -> int:
    bits = 0
    for byte in digest:
        if byte == 0:
            bits += 8
            continue
        bits += 8 - byte.bit_length()
        break
    return bits


def solve(challenge: PowChallenge, *, max_iterations: int = 1 << 24) -> str:
    """Find a solution string whose hash meets the difficulty. Returns the
    solution. ``difficulty=0`` returns immediately (accessibility path)."""
    if challenge.difficulty == 0:
        return "0"
    prefix = challenge._prefix()
    for i in range(max_iterations):
        sol = str(i)
        digest = hashlib.sha256(prefix + b"|" + sol.encode()).digest()
        if _leading_zero_bits(digest) >= challenge.difficulty:
            return sol
    raise RuntimeError("PoW not solved within iteration budget (raise the budget)")


def verify(challenge: PowChallenge, solution: str, *, now: int) -> bool:
    """True iff ``solution`` satisfies ``challenge`` and it hasn't expired.

    ``now`` is passed in (no implicit clock). A difficulty-0 challenge
    verifies any solution within the TTL (the waiver path)."""
    if challenge.difficulty > MAX_DIFFICULTY_BITS:
        return False  # someone tried to demand more than the cap
    if now < challenge.issued_at or now > challenge.issued_at + challenge.ttl_s:
        return False
    if challenge.difficulty == 0:
        return True
    digest = hashlib.sha256(challenge._prefix() + b"|" + solution.encode()).digest()
    return _leading_zero_bits(digest) >= challenge.difficulty


# ── signed (server-issued) challenges — the SEC-7 fix ────────────────


def _challenge_statement(challenge: PowChallenge, issuer_did: str) -> dict[str, Any]:
    return {
        "ctx": _CHALLENGE_CTX,
        "issuer_did": issuer_did,
        "target": challenge.target,
        "nonce": challenge.nonce,
        "difficulty": challenge.difficulty,
        "issued_at": challenge.issued_at,
        "ttl_s": challenge.ttl_s,
    }


def sign_challenge(challenge: PowChallenge, *, sign, issuer_did: str) -> dict[str, Any]:
    """Bind a challenge to the issuing instance's identity key.

    ``sign`` is the instance identity's ``bytes -> bytes`` Ed25519 signer;
    ``issuer_did`` is that identity's did:key. The returned dict is what
    the server hands the client; the client solves it and returns it
    verbatim alongside the solution. The server then re-verifies via
    :func:`open_signed_challenge` so a CLIENT-minted (e.g. difficulty-0)
    challenge can never be accepted.
    """
    decode_ed25519_did(issuer_did)
    statement = _challenge_statement(challenge, issuer_did)
    sig = sign(canonical_bytes(statement))
    return {**statement, "sig": base64.b64encode(sig).decode("ascii")}


def open_signed_challenge(
    signed: dict[str, Any], *, expected_issuer_did: str, now: int,
) -> PowChallenge:
    """Verify a signed challenge really came from ``expected_issuer_did``,
    isn't expired, and is within the difficulty cap. Returns the
    :class:`PowChallenge` for solution checking; raises
    :class:`PowChallengeError` on any failure.

    This is what makes PoW meaningful: the verifier only proceeds for a
    challenge IT signed (or a peer it trusts signed), never one supplied
    by the requester.
    """
    if not isinstance(signed, dict) or signed.get("ctx") != _CHALLENGE_CTX:
        raise PowChallengeError("not a signed PoW challenge")
    sig_b64 = signed.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise PowChallengeError("challenge missing signature")

    issuer_did = str(signed.get("issuer_did", ""))
    if not did_equal(issuer_did, expected_issuer_did):
        raise PowChallengeError("challenge issuer does not match the expected instance")

    try:
        challenge = PowChallenge(
            target=str(signed["target"]),
            nonce=str(signed["nonce"]),
            difficulty=int(signed["difficulty"]),
            issued_at=int(signed["issued_at"]),
            ttl_s=int(signed["ttl_s"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PowChallengeError(f"malformed challenge fields: {exc}") from None

    if challenge.difficulty > MAX_DIFFICULTY_BITS:
        raise PowChallengeError("challenge demands more work than the cap")

    statement = _challenge_statement(challenge, issuer_did)
    try:
        pub_raw = decode_ed25519_did(issuer_did)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            base64.b64decode(sig_b64), canonical_bytes(statement)
        )
    except PowChallengeError:
        raise
    except Exception as exc:
        raise PowChallengeError("challenge signature verification failed") from exc

    if now < challenge.issued_at or now > challenge.issued_at + challenge.ttl_s:
        raise PowChallengeError("challenge expired")
    return challenge


def verify_signed_solution(
    signed: dict[str, Any],
    solution: str,
    *,
    expected_issuer_did: str,
    now: int,
    consumed: ConsumedNonces | None = None,
) -> bool:
    """Full server-side intake check: the challenge was issued by us, is
    fresh, the solution satisfies it, and (if ``consumed`` is supplied)
    the nonce hasn't already been spent.

    Returns True only when all hold. Raises :class:`PowChallengeError` if
    the challenge itself is forged/expired (distinct from a merely wrong
    solution, which returns False)."""
    challenge = open_signed_challenge(
        signed, expected_issuer_did=expected_issuer_did, now=now,
    )
    if not verify(challenge, solution, now=now):
        return False
    if consumed is None:
        return True
    # one solved challenge, one use — replay of the (challenge, solution)
    # pair is rejected once the nonce is spent.
    return consumed.spend(challenge.nonce)


class ConsumedNonces:
    """Single-use guard: a challenge nonce may be redeemed once.

    Without this, an attacker solves ONE server-issued challenge and
    replays the (challenge, solution) pair for many submissions, paying
    the PoW cost a single time. In-memory here; a durable store (keyed by
    nonce, pruned past TTL) is needed for restart-safety — same caveat as
    ``relay_seal.ReplayWindow`` (SEC-8)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def spend(self, nonce: str) -> bool:
        """Return True and record the nonce if fresh; False if already
        spent."""
        fresh = nonce not in self._seen
        self._seen.add(nonce)  # idempotent on a set; records the nonce
        return fresh
