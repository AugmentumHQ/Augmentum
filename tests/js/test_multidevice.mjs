/* P4: multi-device. A second device is linked under the first's master,
 * the bundle carries both, and a sender's message is sealed to BOTH —
 * each device opens its own copy. Skips cleanly if noble isn't vendored.
 */
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(here, '..', '..', 'ui', 'lib', 'noble', 'curves-ed25519.mjs'))) {
  console.log('SKIP: @noble not vendored.'); process.exit(0);
}

const {
  getOrCreateIdentity, generateDeviceKeys, authorizeDevice,
  deviceEntry, deviceBundle, recipientsFromBundle,
} = await import('../../ui/scripts/connect/crypto/keys.js');
const { sealForRecipients, openForMe } = await import('../../ui/scripts/connect/crypto/e2e.js');
const { verifyBinding } = await import('../../ui/scripts/connect/crypto/binding.js');

let pass = 0, fail = 0;
const ok = () => pass++;
const bad = (m) => { console.error('  FAIL:', m); fail++; };
const mkStore = () => { const m = new Map(); return { getItem: async (k) => m.get(k) ?? null, setItem: async (k, v) => void m.set(k, v) }; };

// Phone holds the master; laptop is a new device to link.
const phoneStore = mkStore(), laptopStore = mkStore();
const phone = await getOrCreateIdentity(phoneStore, 1718000000);
const laptop = await generateDeviceKeys(laptopStore, 1718000100);

// Phone authorizes the laptop (the master signs the laptop's subkey).
const laptopBinding = await authorizeDevice(phoneStore, {
  subkeyDid: laptop.deviceDid, issuedAt: laptop.issuedAt,
});

// That binding chains to the SAME master as the phone's identity.
try {
  if (verifyBinding(laptopBinding, phone.masterDid) === laptop.deviceDid) ok();
  else bad('laptop binding subkey mismatch');
} catch (e) { bad('laptop binding verify threw: ' + e.message); }

// The published bundle carries BOTH devices under one master.
const bundle = deviceBundle(phone, "phone", [{
  subkey_did: laptop.deviceDid, sealing_pub_b64: laptop.sealingPubB64,
  binding: laptopBinding, label: 'laptop',
}]);
if (bundle.master_did === phone.masterDid && bundle.devices.length === 2) ok();
else bad('bundle not 2-device');

// A sender seals to ALL of the user's devices; each opens its own copy.
const sender = await getOrCreateIdentity(mkStore(), 1718000200);
const recipients = recipientsFromBundle(bundle, phone.masterDid);
const sealed = sealForRecipients({
  payload: { text: 'reaches all my devices' }, recipients,
  deviceSign: sender.sign, deviceDid: sender.deviceDid, seq: 1, ts: 1718000300,
});
if (Object.keys(sealed).length === 2) ok(); else bad('did not seal to both devices');

for (const [who, dev, priv] of [['phone', phone, phone.sealingPriv], ['laptop', laptop, laptop.sealingPriv]]) {
  try {
    const out = openForMe(sealed, {
      myDeviceDid: dev.deviceDid, myDevicePriv: priv,
      senderMasterDid: sender.masterDid, senderDeviceBinding: sender.binding,
    });
    if (out.payload.text === 'reaches all my devices') ok();
    else bad(`${who} payload mismatch`);
  } catch (e) { bad(`${who} open threw: ` + e.message); }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
