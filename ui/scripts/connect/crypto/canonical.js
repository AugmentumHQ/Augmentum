/* connect/crypto/canonical.js — byte-identical canonical JSON.
 *
 * The interop linchpin for client-side E2E. A signature is taken over
 * canonical_bytes(obj); the browser MUST emit the SAME bytes as the
 * Python server's
 *   json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)
 * or signatures fail to verify across the wire.
 *
 * Why this works with plain JSON.stringify under the hood:
 *   - String escaping is identical between Python json and JS
 *     JSON.stringify (both escape only  "  \  and U+0000..U+001F, with
 *     the same short forms and lowercase \u00xx; everything >= 0x20 incl.
 *     non-ASCII is emitted raw — matching ensure_ascii=False).
 *   - JSON.stringify's default separators are already "," and ":" with no
 *     spaces, matching separators=(",",":").
 *   - The only thing JSON.stringify does NOT do is sort keys, so we
 *     rebuild the value with recursively sorted keys first.
 *
 * Constraints on signed payloads (enforced by the subset, proven by the
 * golden cross-language vectors):
 *   - integers only, within Number.MAX_SAFE_INTEGER (no floats: JS prints
 *     1.0 as "1" but Python prints "1.0").
 *   - object keys are ASCII/BMP identifiers (so JS UTF-16 sort == Python
 *     code-point sort).
 *   - no NaN / Infinity / undefined.
 */

function sortValue(v) {
  if (Array.isArray(v)) return v.map(sortValue);
  if (v && typeof v === 'object') {
    const out = {};
    for (const k of Object.keys(v).sort()) out[k] = sortValue(v[k]);
    return out;
  }
  if (typeof v === 'number' && !Number.isInteger(v)) {
    throw new Error('canonicalize: floats are not allowed in signed payloads');
  }
  return v;
}

/** Serialize to the canonical UTF-8 byte string used for signing. */
export function canonicalString(obj) {
  return JSON.stringify(sortValue(obj));
}

/** Canonical UTF-8 bytes (Uint8Array) for signing / AEAD. */
export function canonicalBytes(obj) {
  return new TextEncoder().encode(canonicalString(obj));
}
