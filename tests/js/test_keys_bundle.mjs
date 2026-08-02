/* P2 client custody + the full A->B E2E flow through a device bundle.
 * Requires vendored noble; skips cleanly otherwise. Run:
 *   node tests/js/test_keys_bundle.mjs
 */
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(here, '..', '..', 'ui', 'lib', 'noble', 'curves-ed25519.mjs'))) {
  console.log('SKIP: @noble not vendored — run `node scripts/vendor_noble.mjs`.');
  process.exit(0);
}

const { getOrCreateIdentity, deviceBundle, recipientsFromBundle } =
  await import('../../ui/scripts/connect/crypto/keys.js');
const { sealForRecipients, openForMe } =
  await import('../../ui/scripts/connect/crypto/e2e.js');
const { verifyBinding } = await import('../../ui/scripts/connect/crypto/binding.js');

let pass = 0, fail = 0;
const ok = () => pass++;
const bad = (m) => { console.error('  FAIL:', m); fail++; };

// In-memory storage adapter (browser uses IndexedDB).
const mkStore = () => { const m = new Map(); return { getItem: async (k) => m.get(k) ?? null, setItem: async (k, v) => void m.set(k, v) }; };

// 1. Identity is created, persisted, and stable across reloads.
const store = mkStore();
const a1 = await getOrCreateIdentity(store, 1718000000);
const a2 = await getOrCreateIdentity(store, 9999999999); // later "now" must NOT change anything
if (a1.masterDid === a2.masterDid && a1.deviceDid === a2.deviceDid &&
    a1.binding.sig === a2.binding.sig) ok();
else bad('identity not stable across reloads');

// 2. The self-produced binding verifies against the master.
try {
  if (verifyBinding(a1.binding, a1.masterDid) === a1.deviceDid) ok();
  else bad('binding subkey mismatch');
} catch (e) { bad('binding verify threw: ' + e.message); }

// 3. recipientsFromBundle enforces the pinned-master check.
const bundleA = deviceBundle(a1, "Alice's phone");
try { recipientsFromBundle(bundleA, 'did:key:zWRONGMASTER'); bad('pinned-master mismatch NOT rejected'); }
catch { ok(); }
const recips = recipientsFromBundle(bundleA, a1.masterDid); // correct pin
if (recips.length === 1 && recips[0].device_did === a1.deviceDid) ok(); else bad('recipients shape');

// 4. FULL FLOW: Bob seals to Alice's device from her bundle; Alice opens.
const bob = await getOrCreateIdentity(mkStore(), 1718000001);
const sealed = sealForRecipients({
  payload: { text: 'hi alice, end to end' },
  recipients: recipientsFromBundle(bundleA, a1.masterDid),
  deviceSign: bob.sign, deviceDid: bob.deviceDid, seq: 1, ts: 1718000002,
});
try {
  const out = openForMe(sealed, {
    myDeviceDid: a1.deviceDid, myDevicePriv: a1.sealingPriv,
    senderMasterDid: bob.masterDid, senderDeviceBinding: bob.binding,
  });
  if (out.payload.text === 'hi alice, end to end' && out.device_did === bob.deviceDid) ok();
  else bad('full flow payload/sender mismatch');
} catch (e) { bad('full flow open threw: ' + e.message); }

// 5. A wrong sender binding (not chaining to the claimed master) is rejected.
const mallory = await getOrCreateIdentity(mkStore(), 1718000003);
try {
  openForMe(sealed, {
    myDeviceDid: a1.deviceDid, myDevicePriv: a1.sealingPriv,
    senderMasterDid: mallory.masterDid, senderDeviceBinding: mallory.binding, // wrong pair
  });
  bad('wrong sender chain NOT rejected');
} catch { ok(); }

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
