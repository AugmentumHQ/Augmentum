/* connect/crypto/didkey.js — did:key for Ed25519 (mirror of didkey.py).
 *
 * Format: "did:key:z" + base58btc( 0xED 0x01 || <32-byte raw pub> ).
 * Trust comparison is ALWAYS on the decoded 32 bytes, never the string.
 * X25519 dids (codec 0xEC 0x01) are rejected — a signing identity must be
 * Ed25519 (closes curve-confusion). Base58 alphabet + algorithm match the
 * Python so encodings are byte-identical.
 */

const _ED25519_PREFIX = Uint8Array.from([0xed, 0x01]);
const _X25519_PREFIX = Uint8Array.from([0xec, 0x01]);
const _DID_PREFIX = 'did:key:z';
const _B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
const _B58_INDEX = (() => {
  const m = {};
  for (let i = 0; i < _B58.length; i++) m[_B58[i]] = i;
  return m;
})();

function b58encode(bytes) {
  // Bytes -> bigint (big-endian), base58 digits, leading-zero -> '1'.
  let n = 0n;
  for (const b of bytes) n = (n << 8n) | BigInt(b);
  let out = '';
  while (n > 0n) {
    const rem = Number(n % 58n);
    n = n / 58n;
    out = _B58[rem] + out;
  }
  let pad = 0;
  for (const b of bytes) { if (b === 0) pad++; else break; }
  return '1'.repeat(pad) + out;
}

function b58decode(str) {
  let n = 0n;
  for (const ch of str) {
    const v = _B58_INDEX[ch];
    if (v === undefined) throw new Error('invalid base58 character');
    n = n * 58n + BigInt(v);
  }
  const bytes = [];
  while (n > 0n) { bytes.unshift(Number(n & 0xffn)); n >>= 8n; }
  let pad = 0;
  for (const ch of str) { if (ch === '1') pad++; else break; }
  return Uint8Array.from([...new Array(pad).fill(0), ...bytes]);
}

export function encodeEd25519Did(pubRaw) {
  if (pubRaw.length !== 32) throw new Error('ed25519 pubkey must be 32 bytes');
  const body = new Uint8Array(2 + 32);
  body.set(_ED25519_PREFIX, 0);
  body.set(pubRaw, 2);
  return _DID_PREFIX + b58encode(body);
}

export function decodeEd25519Did(did) {
  if (typeof did !== 'string' || !did.startsWith(_DID_PREFIX)) {
    throw new Error('not a base58btc did:key');
  }
  const decoded = b58decode(did.slice(_DID_PREFIX.length));
  if (decoded[0] === _X25519_PREFIX[0] && decoded[1] === _X25519_PREFIX[1]) {
    throw new Error('did:key is X25519, not a valid signing identity');
  }
  if (decoded[0] !== _ED25519_PREFIX[0] || decoded[1] !== _ED25519_PREFIX[1]) {
    throw new Error('unsupported did:key multicodec');
  }
  const raw = decoded.slice(2);
  if (raw.length !== 32) throw new Error('ed25519 key body must be 32 bytes');
  return raw;
}

export function didEqual(a, b) {
  try {
    const ra = decodeEd25519Did(a);
    const rb = decodeEd25519Did(b);
    if (ra.length !== rb.length) return false;
    let diff = 0;
    for (let i = 0; i < ra.length; i++) diff |= ra[i] ^ rb[i];
    return diff === 0;
  } catch {
    return false;
  }
}
