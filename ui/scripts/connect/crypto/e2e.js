/* connect/crypto/e2e.js — device-to-device E2E (mirror of e2e.py + e2e_session.py).
 *
 * Direct first: seal to the recipient device, open validating the full
 * chain device -> master -> ceremony-verified human. sealForRecipients is
 * the multi-device / (gated) companion generalization — for a 1:1 chat the
 * list is just the peer's device(s).
 */
import { seal, unseal } from './seal.js';
import { verifyBinding } from './binding.js';
import { didEqual } from './didkey.js';

/** Seal one message to each recipient device. Returns
 *  { recipientDeviceDid: sealedBlob }. */
export function sealForRecipients({ payload, recipients, deviceSign, deviceDid, seq, ts }) {
  const out = {};
  for (const r of recipients) {
    out[r.device_did] = seal({
      payload,
      recipientSealPubB64: r.sealing_pub_b64,
      sign: deviceSign,
      sourceDid: deviceDid,
      seq, ts,
    });
  }
  return out;
}

/** Open the blob sealed to THIS device + validate the sender chain. */
export function openForMe(sealedByDevice, {
  myDeviceDid, myDevicePriv, senderMasterDid, senderDeviceBinding,
}) {
  const blob = sealedByDevice[myDeviceDid];
  if (!blob) throw new Error('no E2E envelope was sealed to this device');
  const inner = unseal(blob, myDevicePriv);
  const boundSubkey = verifyBinding(senderDeviceBinding, senderMasterDid);
  if (!didEqual(boundSubkey, inner.source_did)) {
    throw new Error('author binding is for a different device than the signer');
  }
  return { payload: inner.payload, device_did: inner.source_did, seq: inner.seq, ts: inner.ts };
}
