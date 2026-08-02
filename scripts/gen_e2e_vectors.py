"""Generate golden cross-language vectors for the client E2E crypto port.

Emits, from the REAL server modules, the canonical bytes + did:key strings
the browser client must reproduce byte-for-byte. The node interop test
(tests/js/test_crypto_interop.mjs) asserts the JS twins match these.

Run: python scripts/gen_e2e_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import encode_ed25519_did

_VECTORS = Path(__file__).resolve().parent.parent / "tests" / "vectors"

# Adversarial canonical corpus — nested, unicode, emoji, escapes, control
# chars, key-order permutations, empty containers, ints/bools/null. If the
# JS canonicalizer diverges on ANY of these, signatures break in the wild.
_CANONICAL_CORPUS = [
    {},
    [],
    {"b": 1, "a": 2, "c": 3},
    {"z": {"y": {"x": 1}}, "a": [3, 2, 1]},
    {"nested": {"keys": {"out": "of", "order": True}}, "alpha": None},
    {"unicode": "café résumé naïve", "emoji": "🌿🔒✓"},
    {"escapes": 'quote " backslash \\ slash / tab \t newline \n'},
    {"control": "\b\f end"},
    {"ints": [0, 1, -1, 9007199254740991], "bool": [True, False], "null": None},
    {"empty_str": "", "empty_obj": {}, "empty_arr": []},
    # The real shapes we actually sign:
    {
        "ctx": "augmentum-fabric-relay-seal-v1",
        "source_did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
        "recipient_seal": "AAAA",
        "seq": 1, "ts": 1718000000,
        "payload": {"text": "hello over the relay"},
    },
    {
        "ctx": "augmentum-fabric-author-binding-v1", "v": 1,
        "master_did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
        "subkey_did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
        "purpose": "device", "issued_at": 1,
    },
]

# did:key vectors from raw public-key bytes (incl. the W3C canonical one).
_DIDKEY_CORPUS = [
    "2e6fcce36701dc791488e0d0b1745cc1e33a4c1c9fcc41c63bd343dbbe0970e6",
    "00" * 32,
    "ff" * 32,
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
]


def main() -> None:
    _VECTORS.mkdir(parents=True, exist_ok=True)

    canon = [
        {"obj": obj, "canonical_hex": canonical_bytes(obj).hex()}
        for obj in _CANONICAL_CORPUS
    ]
    (_VECTORS / "e2e_canonical.json").write_text(
        json.dumps(canon, indent=2), encoding="utf-8"
    )

    didkeys = [
        {"pub_hex": h, "did": encode_ed25519_did(bytes.fromhex(h))}
        for h in _DIDKEY_CORPUS
    ]
    (_VECTORS / "e2e_didkey.json").write_text(
        json.dumps(didkeys, indent=2), encoding="utf-8"
    )

    # Round-trip fixtures: a Python-sealed blob + a Python author binding,
    # so the JS client can prove it OPENS Python's output (and, with its
    # own seal written to a file, that Python opens JS's). Keys are fixed
    # and exported so the JS test is deterministic.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from augmentum.fabric.author_binding import mint_binding
    from augmentum.fabric.relay_seal import generate_sealing_key, seal

    # Recipient X25519 sealing key (export raw priv so JS can rebuild it).
    recip = generate_sealing_key()
    recip_priv_raw = recip.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    recip_pub_b64 = __import__("base64").b64encode(
        recip.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode()

    # Sender device signing key (Ed25519).
    dev = Ed25519PrivateKey.generate()
    dev_pub = dev.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    dev_did = encode_ed25519_did(dev_pub)

    blob = seal(
        payload={"text": "sealed by python, opened by js"},
        recipient_sealing_pub_b64=recip_pub_b64,
        origin_sign=dev.sign, source_did=dev_did, seq=7, ts=1718000000,
    )

    # Author binding: a master vouches for the device subkey.
    master = Ed25519PrivateKey.generate()
    master_did = encode_ed25519_did(master.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ))
    binding = mint_binding(
        master_sign=master.sign, master_did=master_did,
        subkey_did=dev_did, issued_at=1718000000,
    )

    (_VECTORS / "e2e_seal.json").write_text(json.dumps({
        "blob": blob,
        "recipient_x25519_priv_hex": recip_priv_raw.hex(),
        "source_did": dev_did,
        "payload": {"text": "sealed by python, opened by js"},
        "binding": binding,
        "master_did": master_did,
        "subkey_did": dev_did,
    }, indent=2), encoding="utf-8")

    print(
        f"wrote {len(canon)} canonical + {len(didkeys)} did:key + 1 seal/binding "
        f"fixture to {_VECTORS}"
    )


if __name__ == "__main__":
    main()
