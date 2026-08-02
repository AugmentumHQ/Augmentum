/* P3: a full two-user E2E conversation through the high-level messaging
 * API + a fake server (bundle store). Proves the orchestration the UI
 * calls: init -> publish bundle -> encryptFor -> decrypt, with the
 * pinned-master guard. Skips cleanly if noble isn't vendored.
 */
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(here, '..', '..', 'ui', 'lib', 'noble', 'curves-ed25519.mjs'))) {
  console.log('SKIP: @noble not vendored — run `node scripts/vendor_noble.mjs`.');
  process.exit(0);
}

const { E2EMessaging } = await import('../../ui/scripts/connect/crypto/messaging.js');

let pass = 0, fail = 0;
const ok = () => pass++;
const bad = (m) => { console.error('  FAIL:', m); fail++; };

// Fake server: an in-memory device-bundle store keyed by user id, with a
// per-client apiFetch that knows which user it is acting as.
const SERVER = new Map(); // userId -> bundle {master_did, devices}
function makeApi(userId) {
  return async (path, opts = {}) => {
    if (path === '/api/fabric/e2e/device-bundle' && opts.method === 'PUT') {
      SERVER.set(userId, opts.body); return { published: true };
    }
    if (path.startsWith('/api/fabric/e2e/device-bundle/') && (opts.method || 'GET') === 'GET') {
      const peer = decodeURIComponent(path.split('/').pop());
      const b = SERVER.get(peer);
      if (!b) throw new Error('404');
      return b; // {master_did, devices}
    }
    throw new Error('unhandled ' + path);
  };
}
const mkStore = () => { const m = new Map(); return { getItem: async (k) => m.get(k) ?? null, setItem: async (k, v) => void m.set(k, v) }; };

// Alice and Bob set up E2E (generate keys + publish bundles).
const alice = new E2EMessaging(mkStore(), makeApi('alice'));
const bob = new E2EMessaging(mkStore(), makeApi('bob'));
const aInfo = await alice.init(1718000000);
const bInfo = await bob.init(1718000001);
if (SERVER.has('alice') && SERVER.has('bob')) ok(); else bad('bundles not published');

// In a real flow each side pins+verifies the other's master via the
// ceremony; here we use the published masters as the pinned values.
const alicePinnedBob = bInfo.masterDid;
const bobPinnedAlice = aInfo.masterDid;

// Bob -> Alice: encrypt, "send" (the body is opaque), Alice decrypts.
const body = await bob.encryptFor('alice', bobPinnedAlice, { text: 'dinner at 7?' }, 1, 1718000002);
if (E2EMessaging.isEncrypted(body) && !JSON.stringify(body).includes('dinner')) ok();
else bad('body not opaque / not flagged');

try {
  const got = alice.decrypt(body, alicePinnedBob);
  if (got.text === 'dinner at 7?') ok(); else bad('decrypt payload mismatch');
} catch (e) { bad('alice decrypt threw: ' + e.message); }

// Pinned-master guard: if Bob's verified master is wrong, Alice refuses to
// be sealed to by him in reverse — and Bob refuses to seal to a wrong pin.
try {
  await alice.encryptFor('bob', 'did:key:zWRONGPIN', { text: 'x' }, 1);
  bad('wrong pinned master was NOT refused');
} catch { ok(); }

// A tampered sender_binding (re-pointed to a different master the recipient
// didn't pin) must fail decryption.
try {
  const evil = { ...body, sender_binding: { ...body.sender_binding, master_did: 'did:key:zEVIL' } };
  alice.decrypt(evil, alicePinnedBob);
  bad('tampered binding was NOT rejected');
} catch { ok(); }

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
