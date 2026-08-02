/* env.js — the portal's envelope layer (the "browser-side VPN").
 *
 * Trust chain (docs/superpowers/specs/2026-07-16-guest-gateway-anonymous-
 * tunnel-e2e-design.md):
 *   invite QR fragment (#k=) pins the instance Ed25519 identity
 *     -> /api/portal/gateway hands out the X25519 seal key SIGNED by it
 *       -> every request is sealed to the seal key + signed by THIS
 *          device's Ed25519 key (registered at claim, trusted at confirm).
 *
 * The fragment never travels on the wire, so the pin arrives out-of-band
 * from whatever (untrusted, anonymous) tunnel carried the page. Once the
 * host confirms the device, the envelope replaces the cookie as the
 * credential — transport becomes plumbing.
 *
 * Crypto: noble Ed25519/X25519/HKDF (same vendored, audited facade the
 * connect E2E layer uses) + native WebCrypto AES-256-GCM.
 */
import { ed25519, x25519, hkdf, sha256 } from '../lib/noble/index.js';

const INFO = new TextEncoder().encode('augmentum-guest-env-v1');
const SEAL_SIG_CTX = new TextEncoder().encode('augmentum-guest-seal-v1');

const b64e = (u8) => btoa(String.fromCharCode(...u8));
const b64d = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const toHex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
const fromHex = (h) => Uint8Array.from(h.match(/.{2}/g).map((x) => parseInt(x, 16)));

/* ── pinned server identity ─────────────────────────────────────────── */

/** Capture #k= from the invite bundle URL (first load) or return the
 *  stored pin. A DIFFERENT key in the fragment than the stored pin is a
 *  hard error — "the server's identity changed" — never silently re-pin. */
export function capturePin() {
  const m = location.hash.match(/[#&]k=([A-Za-z0-9+/=%]+)/);
  const fromUrl = m ? decodeURIComponent(m[1]) : '';
  const stored = localStorage.getItem('portal.pinned_server_key') || '';
  if (fromUrl && stored && fromUrl !== stored) {
    throw new Error('server identity changed — ask your host for a fresh invite');
  }
  if (fromUrl && !stored) localStorage.setItem('portal.pinned_server_key', fromUrl);
  return fromUrl || stored;
}

export const pinnedKey = () => localStorage.getItem('portal.pinned_server_key') || '';

/* ── this device's keys (generated at claim, sent with register) ────── */

function loadOrGen(key, gen) {
  const hex = localStorage.getItem(key);
  if (hex) return fromHex(hex);
  const k = gen();
  localStorage.setItem(key, toHex(k));
  return k;
}

const _edSecret = () => (ed25519.utils.randomSecretKey || ed25519.utils.randomPrivateKey)();
const _xSecret = () => (x25519.utils.randomSecretKey || x25519.utils.randomPrivateKey)();

export function deviceKeys() {
  const signPriv = loadOrGen('portal.e2e.sign', _edSecret);
  const sealPriv = loadOrGen('portal.e2e.seal', _xSecret);
  return {
    signPriv, sealPriv,
    signPubB64: b64e(ed25519.getPublicKey(signPriv)),
    sealPubB64: b64e(x25519.getPublicKey(sealPriv)),
  };
}

/** The device_public_key record sent at register (fills the P2 slot). */
export function devicePublicKeyRecord() {
  const k = deviceKeys();
  return JSON.stringify({ v: 1, sign_pub: k.signPubB64, seal_pub: k.sealPubB64 });
}

/* ── gateway bundle (the server's seal key, verified against the pin) ─ */

let _sealPub = null; // Uint8Array, verified

export async function loadGateway() {
  if (_sealPub) return true;
  const pin = pinnedKey();
  if (!pin) return false;
  let bundle;
  try {
    const res = await fetch('/api/portal/gateway', { credentials: 'same-origin' });
    if (!res.ok) return false;
    bundle = await res.json();
  } catch { return false; }
  if (!bundle || bundle.v !== 1 || !bundle.seal_pub || !bundle.sig) return false;
  const sealRaw = b64d(bundle.seal_pub);
  const payload = new Uint8Array(SEAL_SIG_CTX.length + sealRaw.length);
  payload.set(SEAL_SIG_CTX, 0); payload.set(sealRaw, SEAL_SIG_CTX.length);
  try {
    if (!ed25519.verify(b64d(bundle.sig), payload, b64d(pin))) return false;
  } catch { return false; }
  _sealPub = sealRaw;
  return true;
}

export const envelopeReady = () => !!(_sealPub && pinnedKey());

/* ── seal / open ────────────────────────────────────────────────────── */

async function aesGcm(keyBytes, nonce, data, mode) {
  const key = await crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false,
    [mode === 'encrypt' ? 'encrypt' : 'decrypt']);
  const buf = await crypto.subtle[mode]({ name: 'AES-GCM', iv: nonce }, key, data);
  return new Uint8Array(buf);
}

const kdf = (shared, nonce) => hkdf(sha256, shared, nonce, INFO, 32);
const sigPayload = (v, deviceId, epk, nonce, ct) =>
  new TextEncoder().encode(`${v}|${deviceId}|${epk}|${nonce}|${ct}`);

/** Enveloped fetch. `opts`: {method, body (object|string), contentType}.
 *  Returns {status, contentType, json, text}. Throws on envelope failure —
 *  callers decide whether to fall back to plain fetch. */
export async function envFetch(path, opts = {}) {
  if (!envelopeReady()) throw new Error('envelope not ready');
  const k = deviceKeys();
  const deviceId = localStorage.getItem('portal.device_id') || '';
  if (!deviceId) throw new Error('no device id');

  const method = (opts.method || 'GET').toUpperCase();
  let bodyB64 = '', innerCt = '';
  if (opts.body !== undefined && opts.body !== null) {
    const raw = typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body);
    bodyB64 = b64e(new TextEncoder().encode(raw));
    innerCt = opts.contentType || 'application/json';
  }
  const inner = { m: method, p: path, b: bodyB64, ct: innerCt, ts: Math.floor(Date.now() / 1000) };

  const eph = _xSecret();
  const epkB64 = b64e(x25519.getPublicKey(eph));
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const shared = x25519.getSharedSecret(eph, _sealPub);
  const ct = await aesGcm(kdf(shared, nonce), nonce,
    new TextEncoder().encode(JSON.stringify(inner)), 'encrypt');
  const nonceB64 = b64e(nonce), ctB64 = b64e(ct);
  const sig = ed25519.sign(sigPayload(1, deviceId, epkB64, nonceB64, ctB64), k.signPriv);

  const res = await fetch('/api/portal/env', {
    method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/augmentum-envelope+json' },
    body: JSON.stringify({
      v: 1, device_id: deviceId, epk: epkB64, nonce: nonceB64, ct: ctB64, sig: b64e(sig),
    }),
  });
  if (!res.ok) throw new Error(`envelope refused (${res.status})`);
  const sealed = await res.json();

  // Response: signed by the PINNED identity, sealed to this device.
  const ok = ed25519.verify(
    b64d(sealed.sig),
    sigPayload(1, sealed.device_id, sealed.epk, sealed.nonce, sealed.ct),
    b64d(pinnedKey()),
  );
  if (!ok) throw new Error('response signature failed — refusing to open');
  const rShared = x25519.getSharedSecret(k.sealPriv, b64d(sealed.epk));
  const rNonce = b64d(sealed.nonce);
  const plain = await aesGcm(kdf(rShared, rNonce), rNonce, b64d(sealed.ct), 'decrypt');
  const innerResp = JSON.parse(new TextDecoder().decode(plain));

  const text = innerResp.b ? new TextDecoder().decode(b64d(innerResp.b)) : '';
  let json = null;
  if ((innerResp.ct || '').includes('json')) { try { json = JSON.parse(text); } catch { /* raw */ } }
  return { status: innerResp.s, contentType: innerResp.ct || '', json, text };
}
