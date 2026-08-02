"""Tests for BIP39 seed-phrase backup of the fabric identity key.

Standard, interoperable BIP39 (English wordlist, standard checksum) so
the phrase works with any BIP39 tool the operator already trusts. Pins:

  - the canonical 256-bit all-zero entropy vector → 23×"abandon" + "art"
    (Trezor/BIP39 reference vector).
  - round-trip for arbitrary keys.
  - a single-word typo is caught by the checksum.
  - wrong word count / unknown word raise.
"""
from __future__ import annotations

import pytest

from augmentum.fabric.recovery import (
    MnemonicError,
    key_to_mnemonic,
    mnemonic_to_key,
)

# BIP39 reference: 32 zero bytes of entropy → these 24 words.
_ZERO_PHRASE = " ".join(["abandon"] * 23 + ["art"])


def test_known_zero_entropy_vector():
    assert key_to_mnemonic(b"\x00" * 32) == _ZERO_PHRASE
    assert mnemonic_to_key(_ZERO_PHRASE) == b"\x00" * 32


def test_round_trip_many_keys():
    # Deterministic spread of keys (no RNG — Math.random is unavailable
    # in some harnesses and determinism makes failures reproducible).
    for i in range(100):
        key = bytes(((i * 37 + j * 13) % 256) for j in range(32))
        phrase = key_to_mnemonic(key)
        assert len(phrase.split()) == 24
        assert mnemonic_to_key(phrase) == key


def test_all_ff_entropy_round_trips():
    key = b"\xff" * 32
    assert mnemonic_to_key(key_to_mnemonic(key)) == key


def test_single_word_typo_fails_checksum():
    words = _ZERO_PHRASE.split()
    # Flip the 5th word to something else valid-in-list; checksum breaks.
    words[5] = "zoo"
    with pytest.raises(MnemonicError):
        mnemonic_to_key(" ".join(words))


def test_wrong_word_count_raises():
    with pytest.raises(MnemonicError):
        mnemonic_to_key("abandon abandon abandon")  # 3 words


def test_unknown_word_raises():
    words = _ZERO_PHRASE.split()
    words[0] = "notabip39word"
    with pytest.raises(MnemonicError):
        mnemonic_to_key(" ".join(words))


def test_whitespace_and_case_tolerant():
    upper = "  " + _ZERO_PHRASE.upper().replace(" ", "   ") + "  "
    assert mnemonic_to_key(upper) == b"\x00" * 32


def test_key_to_mnemonic_rejects_wrong_length():
    with pytest.raises(MnemonicError):
        key_to_mnemonic(b"\x00" * 31)
