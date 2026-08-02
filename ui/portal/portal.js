/* portal.js — the invited-guest experience.
 *
 * One page, many states: welcome+register (from an invite token) ->
 * pending (admin confirms) -> sign in -> ready (the messenger/dialer
 * gateway) — or waiting / new-location if the host hasn't confirmed this
 * place yet. Warm, non-technical, installable. Vanilla, no deps.
 */
import { PortalComms } from './messenger.js';
import {
  capturePin, devicePublicKeyRecord, envelopeReady, envFetch, loadGateway,
} from './env.js';

const $ = (sel) => document.querySelector(sel);
const show = (state) => {
  document.querySelectorAll('.pt-card').forEach((c) => { c.hidden = c.dataset.state !== state; });
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Show/hide password eye — wired for every [data-eye] (register, confirm, and
 * sign-in), so a guest can check what they typed. No confirm-mismatch surprises
 * and no blind single-shot password entry. */
const EYE_SHOW = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_HIDE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
function _wirePasswordEyes() {
  document.querySelectorAll('.pt-eye[data-eye]').forEach((btn) => {
    const input = btn.parentElement.querySelector('input');
    if (!input) return;
    btn.innerHTML = EYE_SHOW;
    btn.addEventListener('click', () => {
      const reveal = input.type === 'password';
      input.type = reveal ? 'text' : 'password';
      btn.innerHTML = reveal ? EYE_HIDE : EYE_SHOW;
      btn.setAttribute('aria-label', reveal ? 'Hide password' : 'Show password');
      input.focus();
    });
  });
}
_wirePasswordEyes();

const token = new URLSearchParams(location.search).get('token') || location.hash.match(/token=([^&]+)/)?.[1] || '';
let inviterName = 'your host';

/* A stable per-install device id — this is what makes the line reconnect
 * from ANY network (the host trusts the device, not the IP). Saved with
 * the PWA so reopening keeps the same identity. */
function deviceId() {
  let id = localStorage.getItem('portal.device_id');
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : `dev-${Date.now()}-${Math.random()}`);
    localStorage.setItem('portal.device_id', id);
  }
  return id;
}

async function api(path, opts = {}) {
  // Envelope-first (the browser-side E2E layer): once the server identity is
  // pinned + this device's key is trusted, guest API calls travel sealed and
  // device-signed — the transport (LAN, tunnel, relay) becomes untrusted
  // plumbing. Plain fetch remains the graceful fallback (legacy guests,
  // pre-confirm bootstrap, envelope hiccups).
  if (envelopeReady() && !path.startsWith('/api/portal/register/')) {
    try {
      const r = await envFetch(path, opts);
      if (r.status < 400) return r.json;
      throw new Error((r.json && r.json.error) || `${r.status}`);
    } catch (ex) {
      if (String(ex.message || '').includes('refusing to open')) throw ex;
      // fall through to plain fetch (envelope refused ≠ request invalid:
      // e.g. device not confirmed yet)
    }
  }
  const res = await fetch(path, {
    method: opts.method || 'GET',
    headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    credentials: 'same-origin',
  });
  let json = null;
  try { json = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((json && json.error) || `${res.status}`);
  return json;
}

/* ── boot: decide the first screen ─────────────────────────────────── */
async function boot() {
  // Pin the server identity from the invite bundle fragment FIRST — it
  // rides only in the link/QR (fragments never hit the wire). A changed
  // key is a hard stop, never a silent re-pin.
  try {
    capturePin();
  } catch (ex) {
    document.body.innerHTML = `<main class="pt-shell"><p class="pt-sub">${esc(ex.message)}</p></main>`;
    return;
  }
  loadGateway().catch(() => { /* envelope stays off; plain fetch works */ });

  // Already signed in? (returning guest) -> route by portal status.
  try {
    const me = await api('/api/portal/me');
    return routeByStatus(me);
  } catch { /* not signed in — continue */ }

  if (!token) { show('signin'); return; }
  // Show who invited you, warmly.
  try {
    const { invite } = await api(`/api/auth/invite/${encodeURIComponent(token)}`);
    if (invite && invite.inviter_display_name) inviterName = invite.inviter_display_name;
    // Server lifecycle status is one of active|expired|used|revoked (see
    // invite_store.invite_status). Anything other than "active" means the link
    // is dead. (Was comparing against 'valid', a value the server never emits —
    // latent because the fetch path above was also wrong and always threw.)
    if (invite && invite.status && invite.status !== 'active') {
      $('[data-welcome-sub]').textContent = "This invite isn't active anymore — ask your host for a fresh one.";
      $('[data-register-form]').hidden = true;
    } else {
      $('[data-welcome-title]').textContent = `${inviterName} invited you`;
      $('[data-welcome-sub]').textContent =
        `A private place to message and call ${inviterName} — yours, not a big company's.`;
    }
  } catch { /* preview optional */ }
  show('welcome');
}

function routeByStatus(me) {
  if (me.state === 'ready') return showReady();
  if (me.state === 'new_location') return show('new_location');
  show('waiting');
  // gently poll until the host confirms
  setTimeout(async () => {
    try { routeByStatus(await api('/api/portal/me')); } catch { /* stay */ }
  }, 5000);
}

let _comms = null;
function showReady() {
  show('ready');
  if (_comms) return; // already mounted
  _comms = new PortalComms($('[data-comms]'));
  _comms.start().catch((ex) => {
    $('[data-comms]').innerHTML = `<p class="pt-sub">Couldn't open your messenger — ${esc(ex.message)}</p>`;
  });
}

/* ── register ──────────────────────────────────────────────────────── */
$('[data-register-form]').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const err = $('[data-register-err]');
  err.hidden = true;

  // Client-side guards (the form is novalidate, so the browser won't). Catch a
  // too-short or mistyped password HERE, before creating anything, instead of
  // bouncing off the server or — worse — locking the guest out of an account
  // whose password they can't reproduce.
  const pw = f.password.value;
  const confirm = f.confirm_password ? f.confirm_password.value : pw;
  if (pw.length < 8) {
    err.textContent = 'Password must be at least 8 characters.'; err.hidden = false;
    f.password.focus(); return;
  }
  if (pw !== confirm) {
    err.textContent = "Those passwords don't match — please retype them."; err.hidden = false;
    f.confirm_password.focus(); return;
  }

  const btn = $('[data-register-submit]');
  btn.disabled = true; btn.textContent = 'Creating…';
  try {
    const out = await api(`/api/portal/register/${encodeURIComponent(token)}`, {
      method: 'POST',
      body: {
        username: f.username.value.trim(),
        password: f.password.value,
        display_name: f.display_name.value.trim(),
        device_id: deviceId(),
        // This device's public keys (Ed25519 sign + X25519 seal) — the host's
        // confirm binds them to the guest account, enabling the envelope
        // layer (device signature = the credential). Fills the P2 slot.
        device_public_key: devicePublicKeyRecord(),
      },
    });
    $('[data-pending-sub]').textContent = out.message || $('[data-pending-sub]').textContent;
    show('pending');
  } catch (ex) {
    err.textContent = ex.message; err.hidden = false;
  } finally {
    btn.disabled = false; btn.textContent = 'Create my line';
  }
});

/* ── sign in ───────────────────────────────────────────────────────── */
$('[data-goto-signin]').addEventListener('click', () => show('signin'));
$('[data-signin-form]').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const err = $('[data-signin-err]');
  err.hidden = true;
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ username: f.username.value.trim(), password: f.password.value }),
    });
    if (res.ok) {
      routeByStatus(await api('/api/portal/me'));
      return;
    }
    // The hardened guest gate returns a friendly guest_state on 403.
    const data = await res.json().catch(() => ({}));
    if (data.guest_state === 'waiting') return show('waiting');
    if (data.guest_state === 'new_location') return show('new_location');
    err.textContent = data.error || "That didn't work — check your username and password.";
    err.hidden = false;
  } catch {
    // A THROWN fetch is a connection failure, NOT a credential failure — most
    // often the one-time invite link (an ephemeral tunnel) has closed. Saying
    // "wrong password" here sends the guest chasing a problem that isn't theirs.
    err.textContent = "Can't reach your host right now. The temporary invite link may have closed — ask them for a fresh invite or a permanent address.";
    err.hidden = false;
  }
});

/* ── add to home screen (PWA) ──────────────────────────────────────── */
let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  const b = $('[data-install]');
  if (b) { b.hidden = false; b.onclick = async () => { deferredPrompt.prompt(); deferredPrompt = null; b.hidden = true; }; }
});
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => { /* installability is best-effort */ });
}

boot();
