/* connect/crypto/b64.js — standard base64 (NOT url-safe).
 *
 * The seal wire fields (eph_pub, nonce, ct, sig) are Python
 * base64.b64encode — standard alphabet (+ /) with = padding. Must match
 * exactly; url-safe would corrupt signed/AEAD material.
 */

const _A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function b64encode(bytes) {
  let out = '';
  const n = bytes.length;
  for (let i = 0; i < n; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < n ? bytes[i + 1] : 0;
    const b2 = i + 2 < n ? bytes[i + 2] : 0;
    out += _A[b0 >> 2];
    out += _A[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < n ? _A[((b1 & 15) << 2) | (b2 >> 6)] : '=';
    out += i + 2 < n ? _A[b2 & 63] : '=';
  }
  return out;
}

export function b64decode(str) {
  const clean = str.replace(/=+$/, '');
  const out = [];
  let buf = 0;
  let bits = 0;
  for (const ch of clean) {
    const v = _A.indexOf(ch);
    if (v < 0) throw new Error('b64decode: invalid character');
    buf = (buf << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push((buf >> bits) & 0xff);
    }
  }
  return Uint8Array.from(out);
}
