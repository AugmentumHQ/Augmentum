/* connect/crypto/keys.js — on-device key custody (Connect E2E P2).
 *
 * Generates and persists this device's keys; private keys never leave the
 * device. Produces the master-signed device binding and the public bundle
 * to publish. Storage is injectable so the same code runs in the browser
 * (IndexedDB) and under Node tests (in-memory).
 *
 * Key set:
 *   - master signing key (Ed25519)   — the trust anchor verified in the
 *                                       ceremony; signs device bindings.
 *   - device signing subkey (Ed25519)— signs each outgoing message.
 *   - device sealing key (X25519)    — receives (decrypts) incoming.
 *
 * P2 = single device: the master is generated alongside the first device.
 * Multi-device linking (a second device enrolled by the first) is P4.
 *
 * Web-custody residual (disclosed): IndexedDB is not a secure enclave; a
 * compromised page can read these. The native path should migrate device
 * keys into the platform keystore. This is the ceiling of web E2E.
 */
import { ed25519, x25519 } from '../../../lib/noble/index.js';
import { canonicalBytes } from './canonical.js';
import { b64encode } from './b64.js';
import { encodeEd25519Did } from './didkey.js';

const BINDING_CTX = 'augmentum-fabric-author-binding-v1';
const _toHex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
const _fromHex = (h) => Uint8Array.from(h.match(/.{2}/g).map((x) => parseInt(x, 16)));
const _edSecret = () => (ed25519.utils.randomSecretKey || ed25519.utils.randomPrivateKey)();
const _xSecret = () => (x25519.utils.randomSecretKey || x25519.utils.randomPrivateKey)();

async function _loadOrGen(storage, key, gen) {
  const hex = await storage.getItem(key);
  if (hex) return _fromHex(hex);
  const k = gen();
  await storage.setItem(key, _toHex(k));
  return k;
}

/** Load this device's identity, generating + persisting on first run.
 *  `nowSeconds` is the issued_at for the binding (pass Date.now()/1000 in
 *  the browser); persisted on first create so the bundle stays stable. */
export async function getOrCreateIdentity(storage, nowSeconds) {
  const masterPriv = await _loadOrGen(storage, 'e2e.master', _edSecret);
  const signPriv = await _loadOrGen(storage, 'e2e.device_sign', _edSecret);
  const sealPriv = await _loadOrGen(storage, 'e2e.device_seal', _xSecret);

  let issuedAt = await storage.getItem('e2e.issued_at');
  if (!issuedAt) {
    issuedAt = String(Math.floor(nowSeconds ?? 0));
    await storage.setItem('e2e.issued_at', issuedAt);
  }
  issuedAt = parseInt(issuedAt, 10);

  const masterDid = encodeEd25519Did(ed25519.getPublicKey(masterPriv));
  const deviceDid = encodeEd25519Did(ed25519.getPublicKey(signPriv));
  const sealingPubB64 = b64encode(x25519.getPublicKey(sealPriv));

  // Master vouches for this device subkey (mirror author_binding.mint_binding).
  const stmt = {
    ctx: BINDING_CTX, v: 1, master_did: masterDid,
    subkey_did: deviceDid, purpose: 'device', issued_at: issuedAt,
  };
  const binding = { ...stmt, sig: b64encode(ed25519.sign(canonicalBytes(stmt), masterPriv)) };

  return {
    masterDid,
    deviceDid,
    sealingPubB64,
    binding,
    /** sign(bytes) -> 64-byte Ed25519 sig with THIS device's signing key. */
    sign: (msg) => ed25519.sign(msg, signPriv),
    /** the X25519 private key for opening messages sealed to this device. */
    sealingPriv: sealPriv,
  };
}

/** The PUBLIC device entry for this device (for inclusion in a bundle). */
export function deviceEntry(identity, label = '') {
  return {
    subkey_did: identity.deviceDid,
    sealing_pub_b64: identity.sealingPubB64,
    binding: identity.binding,
    label,
  };
}

/** The PUBLIC bundle to publish (PUT /api/fabric/e2e/device-bundle). Pass
 *  extra authorized device entries (P4 multi-device) to seal to them too. */
export function deviceBundle(identity, label = '', extraDevices = []) {
  return {
    master_did: identity.masterDid,
    devices: [deviceEntry(identity, label), ...extraDevices],
  };
}

// ── P4: multi-device linking ─────────────────────────────────────────

/** Generate (and persist) a NEW device's keys WITHOUT a master — for a
 *  device being linked into an existing account. The master-holding device
 *  authorizes it via authorizeDevice(); the returned `binding` is null
 *  until then. */
export async function generateDeviceKeys(storage, nowSeconds) {
  const signPriv = await _loadOrGen(storage, 'e2e.device_sign', _edSecret);
  const sealPriv = await _loadOrGen(storage, 'e2e.device_seal', _xSecret);
  let issuedAt = await storage.getItem('e2e.issued_at');
  if (!issuedAt) { issuedAt = String(Math.floor(nowSeconds ?? 0)); await storage.setItem('e2e.issued_at', issuedAt); }
  return {
    deviceDid: encodeEd25519Did(ed25519.getPublicKey(signPriv)),
    sealingPubB64: b64encode(x25519.getPublicKey(sealPriv)),
    sign: (msg) => ed25519.sign(msg, signPriv),
    sealingPriv: sealPriv,
    issuedAt: parseInt(issuedAt, 10),
  };
}

/** A master-holding device vouches for a new device's subkey. Returns the
 *  signed binding the new device stores + the master publishes in the
 *  bundle. (The QR/SAS confirmation that you're authorizing the RIGHT
 *  device is the ceremony around this call — the crypto is here.) */
export async function authorizeDevice(masterStorage, { subkeyDid, issuedAt }) {
  const masterHex = await masterStorage.getItem('e2e.master');
  if (!masterHex) throw new Error('this device does not hold the master key');
  const masterPriv = _fromHex(masterHex);
  const masterDid = encodeEd25519Did(ed25519.getPublicKey(masterPriv));
  const stmt = {
    ctx: BINDING_CTX, v: 1, master_did: masterDid,
    subkey_did: subkeyDid, purpose: 'device', issued_at: Math.floor(issuedAt ?? 0),
  };
  return { ...stmt, sig: b64encode(ed25519.sign(canonicalBytes(stmt), masterPriv)) };
}

/** Recipients to seal to, from a fetched peer bundle — ONLY after the
 *  pinned-master check. Throws if the published master isn't the one this
 *  user verified in the ceremony (closes the host-swap gap). */
export function recipientsFromBundle(bundle, pinnedMasterDid) {
  if (!bundle || bundle.master_did !== pinnedMasterDid) {
    throw new Error('contact encryption key is not the one you verified — refusing to seal');
  }
  return (bundle.devices || []).map((d) => ({
    device_did: d.subkey_did,
    sealing_pub_b64: d.sealing_pub_b64,
    label: d.label || '',
  }));
}

/** A browser IndexedDB storage adapter (string values). */
export function indexedDbStorage(dbName = 'augmentum-e2e', store = 'keys') {
  function open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(dbName, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(store);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  const tx = async (mode, fn) => {
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode);
      const os = t.objectStore(store);
      const r = fn(os);
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
  };
  return {
    getItem: (k) => tx('readonly', (os) => os.get(k)).then((v) => v ?? null),
    setItem: (k, v) => tx('readwrite', (os) => os.put(v, k)),
  };
}
