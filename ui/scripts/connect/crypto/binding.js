/* connect/crypto/binding.js — author binding verify (mirror of author_binding.py).
 *
 * Proves a sender's device subkey is vouched for by a master key the user
 * verified in the ceremony. Verification is against the PINNED master
 * (passed in), never the binding's self-asserted one.
 */
import { ed25519 } from '../../../lib/noble/index.js';
import { canonicalBytes } from './canonical.js';
import { b64decode } from './b64.js';
import { decodeEd25519Did, didEqual } from './didkey.js';

const BINDING_VERSION = 1;
const BINDING_CTX = 'augmentum-fabric-author-binding-v1';

export function verifyBinding(binding, expectedMasterDid) {
  if (!binding || binding.v !== BINDING_VERSION) throw new Error('malformed binding');
  if (!binding.sig) throw new Error('binding missing signature');
  if (!didEqual(binding.master_did, expectedMasterDid)) {
    throw new Error('binding master does not match the pinned master');
  }
  const stmt = {
    ctx: BINDING_CTX,
    v: BINDING_VERSION,
    master_did: binding.master_did,
    subkey_did: binding.subkey_did,
    purpose: binding.purpose || 'device',
    issued_at: binding.issued_at,
  };
  const pub = decodeEd25519Did(binding.master_did);
  if (!ed25519.verify(b64decode(binding.sig), canonicalBytes(stmt), pub)) {
    throw new Error('binding signature verification failed');
  }
  return binding.subkey_did;
}
