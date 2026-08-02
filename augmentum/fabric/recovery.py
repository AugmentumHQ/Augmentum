"""BIP39 seed-phrase backup for the fabric identity key.

The fabric Ed25519 private key (``identity.py``) is the one piece of
state that cannot be recovered if lost: it IS the instance's federated
identity, and the fail-closed loader (``identity.py``) deliberately
refuses to silently mint a replacement. So the operator needs a durable,
offline, human-transcribable backup. That backup is a BIP39 mnemonic.

A raw Ed25519 private key is exactly 32 bytes = 256 bits = BIP39's
256-bit entropy case = a **24-word** phrase. Because the key already IS
the entropy, there is no KDF and no derivation path here — the phrase
encodes the key bytes directly (entropy + 8-bit checksum → 24× 11-bit
word indices). ``mnemonic_to_key(key_to_mnemonic(k)) == k`` for all k.

This is intentionally vanilla BIP39 (standard English wordlist, standard
checksum) so the phrase interoperates with any BIP39 tool an operator
already trusts. The 2048-word list is vendored at
``_bip39_english.txt`` (public domain, SHA-256
``2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda``).

Pure stdlib (``hashlib``); no third-party mnemonic dependency.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_WORDLIST_PATH = Path(__file__).with_name("_bip39_english.txt")
_ENTROPY_BYTES = 32  # Ed25519 private key length
_ENTROPY_BITS = _ENTROPY_BYTES * 8  # 256
_CHECKSUM_BITS = _ENTROPY_BITS // 32  # 8
_TOTAL_BITS = _ENTROPY_BITS + _CHECKSUM_BITS  # 264
_WORD_COUNT = _TOTAL_BITS // 11  # 24


class MnemonicError(ValueError):
    """Raised when a mnemonic phrase is malformed or fails its checksum."""


@lru_cache(maxsize=1)
def _wordlist() -> list[str]:
    words = _WORDLIST_PATH.read_text(encoding="utf-8").split()
    if len(words) != 2048:
        raise MnemonicError(
            f"BIP39 wordlist must have 2048 words, found {len(words)}"
        )
    return words


@lru_cache(maxsize=1)
def _word_index() -> dict[str, int]:
    return {w: i for i, w in enumerate(_wordlist())}


def _bits_from_bytes(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def key_to_mnemonic(priv_raw: bytes) -> str:
    """32-byte Ed25519 private key → 24-word BIP39 phrase."""
    if len(priv_raw) != _ENTROPY_BYTES:
        raise MnemonicError(
            f"key must be {_ENTROPY_BYTES} bytes, got {len(priv_raw)}"
        )
    checksum = hashlib.sha256(priv_raw).digest()
    bits = _bits_from_bytes(priv_raw) + _bits_from_bytes(checksum)[:_CHECKSUM_BITS]
    words = _wordlist()
    out = [
        words[int(bits[i:i + 11], 2)]
        for i in range(0, _TOTAL_BITS, 11)
    ]
    return " ".join(out)


def mnemonic_to_key(phrase: str) -> bytes:
    """24-word BIP39 phrase → 32-byte key. Validates the checksum.

    Raises :class:`MnemonicError` on the wrong word count, an unknown
    word, or a checksum mismatch (i.e. a transcription typo).
    """
    tokens = phrase.strip().lower().split()
    if len(tokens) != _WORD_COUNT:
        raise MnemonicError(
            f"phrase must be {_WORD_COUNT} words, got {len(tokens)}"
        )
    index = _word_index()
    bits_parts: list[str] = []
    for tok in tokens:
        idx = index.get(tok)
        if idx is None:
            raise MnemonicError(f"unknown BIP39 word {tok!r}")
        bits_parts.append(f"{idx:011b}")
    bits = "".join(bits_parts)
    entropy_bits, checksum_bits = bits[:_ENTROPY_BITS], bits[_ENTROPY_BITS:]
    entropy = int(entropy_bits, 2).to_bytes(_ENTROPY_BYTES, "big")
    expected = _bits_from_bytes(hashlib.sha256(entropy).digest())[:_CHECKSUM_BITS]
    if checksum_bits != expected:
        raise MnemonicError(
            "BIP39 checksum mismatch — the phrase has a typo or wrong word order"
        )
    return entropy
