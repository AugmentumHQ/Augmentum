"""Guest-gateway envelope layer — the browser-side E2E trust chain.

Covers the crypto contract the portal (``ui/portal/env.js``) and the
``/api/portal/env`` route depend on:

  - the signed seal-key bundle verifies against the pinned identity,
  - a request envelope round-trips (device-signed → instance-sealed),
  - a response envelope round-trips (instance-signed → device-sealed),
  - every failure mode (bad sig, tamper, skew, bad format) raises,
  - the replay guard rejects a reused nonce and bounds its windows,
  - device-key parsing + dispatch allow-list gate correctly,
  - seal-key persistence is stable across loads.

Spec: docs/superpowers/specs/2026-07-16-guest-gateway-anonymous-tunnel-e2e-design.md
"""
from __future__ import annotations

import base64
import json
import time

import aiosqlite
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from augmentum.connect.guest_gateway import (
    _HKDF_INFO,  # noqa: PLC2701 - test mirrors the wire KDF deliberately
    _SEAL_SIG_CTX,  # noqa: PLC2701
    EnvelopeError,
    GatewayKeys,
    ReplayGuard,
    dispatch_allowed,
    get_gateway_keys,
    open_envelope,
    parse_device_public_key,
    seal_to_device,
)
from augmentum.fabric.identity import FabricIdentity
from augmentum.state.settings_store import SettingsStore

# NOTE: the KDF (HKDF-SHA256, imported above) is re-derived locally in
# ``_derive`` so the test is an independent second implementation of the wire
# format, not a call into the code under test.


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _derive(shared: bytes, nonce: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce, info=_HKDF_INFO).derive(shared)


def _raw_pub(key) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    )


def _make_identity() -> FabricIdentity:
    priv = Ed25519PrivateKey.generate()
    return FabricIdentity(node_id="test-node", private_key=priv, public_key=priv.public_key())


def _make_gateway_keys() -> GatewayKeys:
    return GatewayKeys(identity=_make_identity(), seal_private=X25519PrivateKey.generate())


class _Device:
    """A stand-in for the portal's WebCrypto device keys."""

    def __init__(self) -> None:
        self.sign = Ed25519PrivateKey.generate()
        self.seal = X25519PrivateKey.generate()

    @property
    def sign_pub_b64(self) -> str:
        return _b64(_raw_pub(self.sign))

    @property
    def seal_pub_b64(self) -> str:
        return _b64(_raw_pub(self.seal))

    def make_request_envelope(self, keys: GatewayKeys, inner: dict, *, device_id="dev-1") -> dict:
        """Mirror the browser: seal inner to the instance seal key, sign with device."""
        eph = X25519PrivateKey.generate()
        epk_raw = _raw_pub(eph)
        nonce = b"\x01" * 12
        shared = eph.exchange(X25519PublicKey.from_public_bytes(
            base64.b64decode(keys.seal_public_b64)))
        ct = AESGCM(_derive(shared, nonce)).encrypt(
            nonce, json.dumps(inner).encode("utf-8"), None)
        epk_s, nonce_s, ct_s = _b64(epk_raw), _b64(nonce), _b64(ct)
        payload = f"1|{device_id}|{epk_s}|{nonce_s}|{ct_s}".encode("ascii")
        sig = self.sign.sign(payload)
        return {"v": 1, "device_id": device_id, "epk": epk_s,
                "nonce": nonce_s, "ct": ct_s, "sig": _b64(sig)}


# ── bundle: the signed seal-key hand-off ────────────────────────────────────

def test_bundle_signature_verifies_against_identity():
    keys = _make_gateway_keys()
    bundle = keys.bundle()
    assert bundle["identity_pub"] == keys.identity.public_key_b64
    raw_seal = base64.b64decode(bundle["seal_pub"])
    sig = base64.b64decode(bundle["sig"])
    # The portal verifies exactly this before sealing anything.
    assert FabricIdentity.verify(_SEAL_SIG_CTX + raw_seal, sig, bundle["identity_pub"])


def test_bundle_signature_rejects_wrong_seal_key():
    keys = _make_gateway_keys()
    bundle = keys.bundle()
    other_seal = _raw_pub(X25519PrivateKey.generate())
    sig = base64.b64decode(bundle["sig"])
    assert not FabricIdentity.verify(_SEAL_SIG_CTX + other_seal, sig, bundle["identity_pub"])


# ── envelope round-trips ────────────────────────────────────────────────────

def test_request_envelope_roundtrip():
    keys = _make_gateway_keys()
    dev = _Device()
    inner = {"m": "GET", "p": "/api/portal/me", "ts": int(time.time())}
    env = dev.make_request_envelope(keys, inner)
    got = open_envelope(env, keys=keys, device_sign_pub_b64=dev.sign_pub_b64)
    assert got["m"] == "GET" and got["p"] == "/api/portal/me"


def test_response_envelope_roundtrip():
    keys = _make_gateway_keys()
    dev = _Device()
    resp_inner = {"s": 200, "b": "eyJvayI6IHRydWV9", "ts": int(time.time())}
    sealed = seal_to_device(
        resp_inner, keys=keys, device_seal_pub_b64=dev.seal_pub_b64, device_id="dev-1")
    # Client opens: verify instance sig, decrypt with device seal priv.
    payload = f"1|dev-1|{sealed['epk']}|{sealed['nonce']}|{sealed['ct']}".encode("ascii")
    assert FabricIdentity.verify(payload, base64.b64decode(sealed["sig"]), keys.identity.public_key_b64)
    shared = dev.seal.exchange(X25519PublicKey.from_public_bytes(base64.b64decode(sealed["epk"])))
    nonce = base64.b64decode(sealed["nonce"])
    plain = AESGCM(_derive(shared, nonce)).decrypt(nonce, base64.b64decode(sealed["ct"]), None)
    assert json.loads(plain)["s"] == 200


# ── failure modes (all must raise EnvelopeError) ────────────────────────────

def test_bad_device_signature_rejected():
    keys = _make_gateway_keys()
    dev, imposter = _Device(), _Device()
    env = dev.make_request_envelope(keys, {"m": "GET", "p": "/x", "ts": int(time.time())})
    # Verified against a DIFFERENT device's key → signature fails.
    with pytest.raises(EnvelopeError):
        open_envelope(env, keys=keys, device_sign_pub_b64=imposter.sign_pub_b64)


def test_tampered_ciphertext_rejected():
    keys = _make_gateway_keys()
    dev = _Device()
    env = dev.make_request_envelope(keys, {"m": "GET", "p": "/x", "ts": int(time.time())})
    ct = bytearray(base64.b64decode(env["ct"]))
    ct[0] ^= 0xFF
    env["ct"] = _b64(bytes(ct))
    # The sig covers ct, so tampering breaks the signature first.
    with pytest.raises(EnvelopeError):
        open_envelope(env, keys=keys, device_sign_pub_b64=dev.sign_pub_b64)


def test_timestamp_skew_rejected():
    keys = _make_gateway_keys()
    dev = _Device()
    stale = {"m": "GET", "p": "/x", "ts": int(time.time()) - 10_000}
    env = dev.make_request_envelope(keys, stale)
    with pytest.raises(EnvelopeError):
        open_envelope(env, keys=keys, device_sign_pub_b64=dev.sign_pub_b64)


def test_wrong_seal_key_fails_decrypt():
    keys = _make_gateway_keys()
    dev = _Device()
    env = dev.make_request_envelope(keys, {"m": "GET", "p": "/x", "ts": int(time.time())})
    # A gateway with a different seal key can't derive the shared secret; but the
    # device sig still matches, so it gets past sig and fails at decrypt.
    other = GatewayKeys(identity=keys.identity, seal_private=X25519PrivateKey.generate())
    with pytest.raises(EnvelopeError):
        open_envelope(env, keys=other, device_sign_pub_b64=dev.sign_pub_b64)


@pytest.mark.parametrize("bad", [
    {},
    {"v": 2, "device_id": "d", "epk": "a", "nonce": "b", "ct": "c", "sig": "s"},
    {"v": 1, "device_id": "", "epk": "a", "nonce": "b", "ct": "c", "sig": "s"},
    {"v": 1, "device_id": "d", "epk": "!!!", "nonce": "b", "ct": "c", "sig": "s"},
])
def test_malformed_envelope_rejected(bad):
    keys = _make_gateway_keys()
    with pytest.raises(EnvelopeError):
        open_envelope(bad, keys=keys, device_sign_pub_b64=_Device().sign_pub_b64)


# ── replay guard ────────────────────────────────────────────────────────────

def test_replay_guard_rejects_reused_nonce():
    guard = ReplayGuard()
    assert guard.check_and_record("dev", "nonceA") is True
    assert guard.check_and_record("dev", "nonceA") is False   # replay
    assert guard.check_and_record("dev", "nonceB") is True    # fresh ok


def test_replay_guard_evicts_old_nonces():
    guard = ReplayGuard(per_device=4)
    for i in range(4):
        assert guard.check_and_record("dev", f"n{i}") is True
    # n0 falls out of the window once we exceed per_device.
    assert guard.check_and_record("dev", "n4") is True
    assert guard.check_and_record("dev", "n0") is True  # evicted → treated as fresh


def test_replay_guard_isolates_devices():
    guard = ReplayGuard()
    assert guard.check_and_record("A", "shared") is True
    assert guard.check_and_record("B", "shared") is True  # different device, own window


# ── device key parsing ──────────────────────────────────────────────────────

def test_parse_device_public_key_valid():
    dev = _Device()
    rec = json.dumps({"v": 1, "sign_pub": dev.sign_pub_b64, "seal_pub": dev.seal_pub_b64})
    parsed = parse_device_public_key(rec)
    assert parsed and parsed["sign_pub"] == dev.sign_pub_b64


@pytest.mark.parametrize("raw", [
    "", "   ", "not-json", "{}",
    json.dumps({"v": 2, "sign_pub": "x", "seal_pub": "y"}),
    json.dumps({"v": 1, "sign_pub": "", "seal_pub": "y"}),
    json.dumps({"v": 1, "sign_pub": _b64(b"short"), "seal_pub": _b64(b"short")}),
])
def test_parse_device_public_key_rejects(raw):
    assert parse_device_public_key(raw) is None


# ── dispatch allow-list ─────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path,ok", [
    ("GET", "/api/portal/me", True),
    ("POST", "/api/connect/threads", True),
    ("GET", "/api/auth/status", True),
    ("GET", "/api/portal/env", False),       # no envelope-in-envelope
    ("GET", "/api/portal/gateway", False),
    ("GET", "/api/admin/users", False),      # outside the guest surface
    ("PATCH", "/api/portal/me", False),      # method not allowed
])
def test_dispatch_allowed(method, path, ok):
    assert dispatch_allowed(method, path) is ok


# ── seal-key persistence ────────────────────────────────────────────────────

async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))")
    await conn.commit()
    return conn, SettingsStore(conn)


@pytest.mark.asyncio
async def test_seal_key_is_stable_across_loads():
    conn, store = await _make_store()
    try:
        first = await get_gateway_keys(store)
        second = await get_gateway_keys(store)
        assert first.seal_public_b64 == second.seal_public_b64
        assert first.identity.public_key_b64 == second.identity.public_key_b64
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_seal_private_not_stored_plaintext():
    conn, store = await _make_store()
    try:
        keys = await get_gateway_keys(store)
        raw_priv = keys.seal_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        cur = await conn.execute("SELECT key, value FROM app_settings")
        rows = await cur.fetchall()
        await cur.close()
        blob = "\n".join(v for _, v in rows)
        assert _b64(raw_priv) not in blob  # encrypted at rest
    finally:
        await conn.close()
