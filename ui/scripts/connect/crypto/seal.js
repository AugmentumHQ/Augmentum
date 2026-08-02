/* connect/crypto/seal.js — sign-then-seal, the client mirror of relay_seal.py.
 *
 * Byte-for-byte the same ECIES the server speaks: ephemeral X25519 -> ECDH
 * -> HKDF-SHA256(salt=eph_pub, info=relay-seal-v1) -> ChaCha20-Poly1305,
 * with the origin Ed25519 signature carried INSIDE the seal and the
 * recipient's sealing key bound into the signed inner (SEC-1). A blob
 * sealed here opens in Python and vice-versa (proven by the round-trip
 * vectors once noble is vendored).
 */
import { ed25519, x25519, chacha20poly1305, hkdf, sha256 } from '../../../lib/noble/index.js';
import { canonicalBytes } from './canonical.js';
import { b64decode, b64encode } from './b64.js';
import { decodeEd25519Did } from './didkey.js';

const SEAL_VERSION = 1;
const SEAL_CTX = 'augmentum-fabric-relay-seal-v1';
const HKDF_INFO = new TextEncoder().encode('augmentum-fabric-relay-seal-v1');

function randomBytes(n) {
  const b = new Uint8Array(n);
  crypto.getRandomValues(b);
  return b;
}

function x25519SecretKey() {
  const u = x25519.utils;
  return (u.randomSecretKey || u.randomPrivateKey).call(u);
}

/** sign({payload, recipientSealPubB64, sign, sourceDid, seq, ts}) -> outer blob.
 *  `sign` is (bytes -> 64-byte Ed25519 sig) using the sender's signing key. */
export function seal({ payload, recipientSealPubB64, sign, sourceDid, seq, ts }) {
  decodeEd25519Did(sourceDid); // fail loudly on a malformed source did

  const innerSigned = {
    ctx: SEAL_CTX,
    source_did: sourceDid,
    recipient_seal: recipientSealPubB64,
    seq, ts, payload,
  };
  const originSig = sign(canonicalBytes(innerSigned));
  const inner = { ...innerSigned, origin_sig: b64encode(originSig) };
  const innerBytes = canonicalBytes(inner);

  const ephPriv = x25519SecretKey();
  const ephPub = x25519.getPublicKey(ephPriv);
  const shared = x25519.getSharedSecret(ephPriv, b64decode(recipientSealPubB64));
  const key = hkdf(sha256, shared, ephPub, HKDF_INFO, 32);

  const nonce = randomBytes(12);
  const aad = canonicalBytes({ v: SEAL_VERSION, eph: b64encode(ephPub) });
  const ct = chacha20poly1305(key, nonce, aad).encrypt(innerBytes);

  return {
    v: SEAL_VERSION,
    eph_pub: b64encode(ephPub),
    nonce: b64encode(nonce),
    ct: b64encode(ct),
  };
}

/** unseal(blob, sealingPriv) -> {source_did, seq, ts, payload}, authenticated. */
export function unseal(blob, sealingPriv) {
  if (!blob || blob.v !== SEAL_VERSION) throw new Error('unsupported sealed envelope');
  const ephPub = b64decode(blob.eph_pub);
  const nonce = b64decode(blob.nonce);
  const ct = b64decode(blob.ct);

  const shared = x25519.getSharedSecret(sealingPriv, ephPub);
  const key = hkdf(sha256, shared, ephPub, HKDF_INFO, 32);
  const aad = canonicalBytes({ v: SEAL_VERSION, eph: b64encode(ephPub) });
  let innerBytes;
  try {
    innerBytes = chacha20poly1305(key, nonce, aad).decrypt(ct);
  } catch {
    throw new Error('seal decryption failed');
  }
  const inner = JSON.parse(new TextDecoder().decode(innerBytes));

  // SEC-1: confirm WE are the intended recipient before trusting anything.
  const ownPub = b64encode(x25519.getPublicKey(sealingPriv));
  if (inner.recipient_seal !== ownPub) {
    throw new Error('sealed message was addressed to a different recipient');
  }

  const innerSigned = {
    ctx: SEAL_CTX,
    source_did: inner.source_did,
    recipient_seal: inner.recipient_seal,
    seq: inner.seq, ts: inner.ts, payload: inner.payload,
  };
  const pub = decodeEd25519Did(inner.source_did);
  if (!ed25519.verify(b64decode(inner.origin_sig), canonicalBytes(innerSigned), pub)) {
    throw new Error('inner origin signature failed — unauthenticated');
  }
  return { source_did: inner.source_did, seq: inner.seq, ts: inner.ts, payload: inner.payload };
}
