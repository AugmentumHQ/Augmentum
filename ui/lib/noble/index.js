/* ui/lib/noble/index.js — audited crypto primitives facade.
 *
 * Single integration point for the @noble suite used by the client E2E
 * crypto. The bundles below are vendored by `node scripts/vendor_noble.mjs`
 * (pinned versions, hashes recorded in ui/lib/VENDORED.md). Importing this
 * before vendoring throws — only seal.js/binding.js/e2e.js depend on it, so
 * the proven interop spine (canonical/didkey/b64) stays usable regardless.
 *
 * Required exports (noble 2.x):
 *   ed25519  — .sign(msg, priv) -> 64B, .verify(sig, msg, pub) -> bool,
 *              .getPublicKey(priv) -> 32B
 *   x25519   — .getSharedSecret(priv, pub) -> 32B, .getPublicKey(priv) -> 32B,
 *              .utils.randomSecretKey() -> 32B
 *   hkdf(sha256, ikm, salt, info, len) -> Uint8Array
 *   sha256   — hash ctor for hkdf
 *   chacha20poly1305(key, nonce, aad) -> { encrypt(pt), decrypt(ct) }
 */
export { ed25519, x25519 } from './curves-ed25519.mjs';
export { chacha20poly1305 } from './ciphers-chacha.mjs';
export { hkdf } from './hashes-hkdf.mjs';
export { sha256 } from './hashes-sha2.mjs';
