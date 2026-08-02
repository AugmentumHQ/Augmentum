/* connect/crypto/messaging.js — high-level E2E for a conversation (P3).
 *
 * The single API the message UI calls so it never touches raw crypto:
 *
 *   const e2e = new E2EMessaging(storage, apiFetch);
 *   await e2e.init();                       // device keys + publish bundle
 *   const body = await e2e.encryptFor(peerUserId, pinnedMasterDid, text, seq);
 *   const text = e2e.decrypt(body, senderMasterDid, senderBinding);
 *
 * encryptFor fetches the peer's published device bundle, enforces the
 * pinned-master check (refuses to seal to a key the user didn't verify),
 * and fans the message out to all of the peer's devices. The returned
 * `body` is what the UI stores/sends in connect_messages.body — opaque to
 * both servers.
 *
 * Outgoing wire body shape: { e2e: 1, v: 1, sealed: { deviceDid: blob, ... } }.
 */
import { getOrCreateIdentity, deviceBundle, recipientsFromBundle } from './keys.js';
import { sealForRecipients, openForMe } from './e2e.js';

export class E2EMessaging {
  /** storage: IndexedDB adapter (keys.js). apiFetch(path, opts) -> json. */
  constructor(storage, apiFetch) {
    this.storage = storage;
    this.api = apiFetch;
    this.identity = null;
  }

  /** Generate/load device keys and publish our public bundle. Idempotent. */
  async init(nowSeconds) {
    this.identity = await getOrCreateIdentity(this.storage, nowSeconds ?? Math.floor(Date.now() / 1000));
    await this.api('/api/fabric/e2e/device-bundle', {
      method: 'PUT', body: deviceBundle(this.identity, 'this device'),
    });
    return { masterDid: this.identity.masterDid, deviceDid: this.identity.deviceDid };
  }

  /** Seal `payload` to every device of `peerUserId`. Throws if the peer's
   *  published master isn't the one the user verified (`pinnedMasterDid`). */
  async encryptFor(peerUserId, pinnedMasterDid, payload, seq, ts) {
    if (!this.identity) throw new Error('call init() first');
    const bundle = await this.api(`/api/fabric/e2e/device-bundle/${encodeURIComponent(peerUserId)}`, { method: 'GET' });
    const recipients = recipientsFromBundle(bundle, pinnedMasterDid); // throws on mismatch
    const sealed = sealForRecipients({
      payload, recipients,
      deviceSign: this.identity.sign,
      deviceDid: this.identity.deviceDid,
      seq, ts: ts ?? Math.floor(Date.now() / 1000),
    });
    // Self-describing: carry the sender's device binding so the recipient
    // can validate device->master WITHOUT a second fetch. The binding is
    // still verified against the recipient's PINNED master, so a forged
    // one can't lie about who the sender is.
    return { e2e: 1, v: 1, sealed, sender_binding: this.identity.binding };
  }

  /** Open an incoming E2E body addressed to this device. `senderMasterDid`
   *  is the master the recipient VERIFIED for the sender (from their pin) —
   *  never taken from the body. */
  decrypt(body, senderMasterDid) {
    if (!this.identity) throw new Error('call init() first');
    if (!body || body.e2e !== 1 || !body.sealed) throw new Error('not an E2E message');
    const out = openForMe(body.sealed, {
      myDeviceDid: this.identity.deviceDid,
      myDevicePriv: this.identity.sealingPriv,
      senderMasterDid, senderDeviceBinding: body.sender_binding,
    });
    return out.payload;
  }

  /** True if a stored body is an E2E envelope (so the UI decrypts vs renders). */
  static isEncrypted(body) {
    return !!(body && body.e2e === 1 && body.sealed);
  }
}
