/* Cross-language seal round-trip: proves the JS client opens Python's
 * sealed output and verifies Python's author binding — and that a JS
 * seal round-trips. Requires the vendored @noble bundles; if they're not
 * present yet it SKIPS cleanly (the crypto modules are complete, just
 * waiting on `node scripts/vendor_noble.mjs`). Run:
 *   node tests/js/test_seal_roundtrip.mjs
 */
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const nobleReady = existsSync(join(here, '..', '..', 'ui', 'lib', 'noble', 'curves-ed25519.mjs'));
if (!nobleReady) {
  console.log('SKIP: @noble not vendored yet — run `node scripts/vendor_noble.mjs`.');
  console.log('      The client crypto modules (seal/binding/e2e) are complete and will');
  console.log('      activate automatically once the bundles are present.');
  process.exit(0);
}

const { unseal, seal } = await import('../../ui/scripts/connect/crypto/seal.js');
const { verifyBinding } = await import('../../ui/scripts/connect/crypto/binding.js');
const { ed25519, x25519 } = await import('../../ui/lib/noble/index.js');
const { encodeEd25519Did } = await import('../../ui/scripts/connect/crypto/didkey.js');
const { b64encode } = await import('../../ui/scripts/connect/crypto/b64.js');

const fromHex = (h) => Uint8Array.from(Buffer.from(h, 'hex'));
let pass = 0, fail = 0;
const ok = () => pass++;
const bad = (m) => { console.error('  FAIL:', m); fail++; };

const fx = JSON.parse(readFileSync(join(here, '..', 'vectors', 'e2e_seal.json'), 'utf-8'));

// 1. Python-sealed blob opens in JS, authenticated.
try {
  const out = unseal(fx.blob, fromHex(fx.recipient_x25519_priv_hex));
  if (JSON.stringify(out.payload) === JSON.stringify(fx.payload) && out.source_did === fx.source_did) ok();
  else bad('python->js seal: payload/source mismatch');
} catch (e) { bad('python->js seal threw: ' + e.message); }

// 2. Python author binding verifies in JS.
try {
  if (verifyBinding(fx.binding, fx.master_did) === fx.subkey_did) ok();
  else bad('binding subkey mismatch');
} catch (e) { bad('binding verify threw: ' + e.message); }

// 3. Pure-JS seal -> unseal round-trip (and recipient-binding rejection).
try {
  const recipPriv = x25519.utils.randomSecretKey ? x25519.utils.randomSecretKey() : x25519.utils.randomPrivateKey();
  const recipPub = b64encode(x25519.getPublicKey(recipPriv));
  const devPriv = ed25519.utils.randomSecretKey ? ed25519.utils.randomSecretKey() : ed25519.utils.randomPrivateKey();
  const devDid = encodeEd25519Did(ed25519.getPublicKey(devPriv));
  const blob = seal({
    payload: { text: 'js round trip' }, recipientSealPubB64: recipPub,
    sign: (m) => ed25519.sign(m, devPriv), sourceDid: devDid, seq: 1, ts: 1,
  });
  const out = unseal(blob, recipPriv);
  if (out.payload.text === 'js round trip' && out.source_did === devDid) ok();
  else bad('js round-trip mismatch');

  // wrong recipient must be rejected (SEC-1)
  const otherPriv = x25519.utils.randomSecretKey ? x25519.utils.randomSecretKey() : x25519.utils.randomPrivateKey();
  try { unseal(blob, otherPriv); bad('wrong-recipient was NOT rejected'); }
  catch { ok(); }
} catch (e) { bad('js seal round-trip threw: ' + e.message); }

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
