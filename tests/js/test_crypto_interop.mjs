/* Cross-language interop test: the JS client crypto must produce
 * byte-identical canonical bytes and did:key strings to the Python
 * server (the signatures are taken over these bytes). Run:
 *   python scripts/gen_e2e_vectors.py   # regenerate golden vectors
 *   node tests/js/test_crypto_interop.mjs
 * Exits non-zero on any divergence.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { canonicalBytes } from '../../ui/scripts/connect/crypto/canonical.js';
import {
  encodeEd25519Did, decodeEd25519Did, didEqual,
} from '../../ui/scripts/connect/crypto/didkey.js';
import { b64encode, b64decode } from '../../ui/scripts/connect/crypto/b64.js';

const here = dirname(fileURLToPath(import.meta.url));
const vdir = join(here, '..', 'vectors');
const toHex = (u8) => Buffer.from(u8).toString('hex');
const fromHex = (h) => Uint8Array.from(Buffer.from(h, 'hex'));

let pass = 0;
let fail = 0;
const bad = (msg) => { console.error('  FAIL:', msg); fail++; };
const ok = () => { pass++; };

// ── canonical bytes: byte-identical to Python ────────────────────────
const canon = JSON.parse(readFileSync(join(vdir, 'e2e_canonical.json'), 'utf-8'));
for (const v of canon) {
  const got = toHex(canonicalBytes(v.obj));
  if (got === v.canonical_hex) ok();
  else bad(`canonical mismatch for ${JSON.stringify(v.obj).slice(0, 60)}\n    exp ${v.canonical_hex}\n    got ${got}`);
}

// ── did:key: byte-identical + round-trip ─────────────────────────────
const dids = JSON.parse(readFileSync(join(vdir, 'e2e_didkey.json'), 'utf-8'));
for (const v of dids) {
  const did = encodeEd25519Did(fromHex(v.pub_hex));
  if (did !== v.did) { bad(`did encode mismatch\n    exp ${v.did}\n    got ${did}`); continue; }
  if (toHex(decodeEd25519Did(did)) !== v.pub_hex) { bad(`did decode round-trip ${v.did}`); continue; }
  ok();
}

// ── did:key negatives ────────────────────────────────────────────────
// X25519 did must be rejected (curve confusion).
try {
  decodeEd25519Did('did:key:z6LSeu9HkTHSfLLeUs2nnzUSNedgDUevfNQgQjQC23ZCit6F');
  bad('X25519 did was NOT rejected');
} catch { ok(); }
if (!didEqual(encodeEd25519Did(fromHex(dids[0].pub_hex)), dids[0].did)) bad('didEqual same-key false'); else ok();
if (didEqual(dids[0].did, dids[1].did)) bad('didEqual distinct-key true'); else ok();

// ── base64 standard alphabet round-trip + known vector ───────────────
const sample = Uint8Array.from([0, 1, 2, 253, 254, 255, 16, 32]);
if (toHex(b64decode(b64encode(sample))) === toHex(sample)) ok(); else bad('b64 round-trip');
// Known vectors: Python base64.b64encode(b"\0"*32) == 43 'A' + '='; 3 zero bytes == 'AAAA'.
if (b64encode(new Uint8Array(32)) === 'A'.repeat(43) + '=' && b64encode(new Uint8Array(3)) === 'AAAA') ok();
else bad(`b64 known vector: ${b64encode(new Uint8Array(32))}`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
