/* connect/incoming-modal.js — full-screen incoming-call ring screen.
 *
 * Substrate-side, EVENT_INVITE arrives on the connect signaling WS.
 * Without this module, the only user-visible signal was the small
 * notification banner — easy to miss when the user is heads-down in
 * another tab. Real comms apps (FaceTime, WhatsApp, Discord) all
 * front-and-center the incoming-call surface; this matches.
 *
 * What it owns:
 *   - A centered modal that appears when EVENT_INVITE arrives.
 *   - Accept / Decline buttons (large, finger-sized).
 *   - Auto-dismiss on hangup, signaling close, or after the invite
 *     lifetime expires.
 *   - The ringtone hook (the audio is wired in ringtone.js — this
 *     module just calls start() / stop()).
 *
 * What it deliberately doesn't own:
 *   - The WebRTC handshake. That stays in incoming.js, kicked off by
 *     the same ``augmentum:notification-action`` event the existing
 *     notification banner uses. The modal dispatches that event on
 *     Accept so the gesture-capture flow there runs unchanged.
 *   - Server-side state. MSG_ACCEPT / MSG_DECLINE go straight through
 *     the connect WS, which already routes the corresponding events
 *     to the caller and stops the missed-call timer.
 */

import { escapeHtml } from '../app.js';
import {
  broadcastCallChanged,
  initBroadcast,
  onBroadcast,
} from './broadcast.js';
import { on as onConnectEvent, send as sendEnvelope } from './client.js';
import { icon } from './icons.js';
import { resolvePeerName } from './messages.js';
import { startRingtone, stopRingtone } from './ringtone.js';

const INVITE_LIFETIME_MS = 60_000;

let _modal = null;
let _modalState = null;  // { callId, callerDid, modalities, expiresAt, timer }
let _initialized = false;

// ── Public API ──────────────────────────────────────────────────

/** Initialize. Idempotent. */
export function initIncomingCallModal() {
  if (_initialized) return;
  _initialized = true;

  // Keyboard dismiss: Escape declines the incoming call (matches the
  // swipe-away gesture on native dialers). Guarded by _modalState so the
  // listener is inert whenever no call is ringing.
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && _modalState) {
      ev.preventDefault();
      _onDecline();
    }
  });

  // Open the modal on invite. Multiple invites in flight collapses
  // to the latest — the previous one is dismissed (treated as
  // declined-by-superseded) so we don't stack a ring tower.
  onConnectEvent('invite', ({ data, peer }) => {
    const callId = String(data?.call_id || '');
    if (!callId) return;
    if (_modalState && _modalState.callId === callId) return;
    _openModal({
      callId,
      callerDid: String(peer || ''),
      modalities: String(data?.modalities || 'audio'),
      lifetimeMs: Number(data?.lifetime) || INVITE_LIFETIME_MS,
    });
  });

  // Caller hung up before we accepted → close the modal.
  onConnectEvent('hangup', ({ data }) => {
    const callId = String(data?.call_id || '');
    if (!callId) return;
    if (_modalState && _modalState.callId === callId) _closeModal('remote_hangup');
  });

  // EVENT_DECLINE arrives in two cases:
  //   1) The remote caller's invite was cancelled (rare — caller
  //      usually sends HANGUP, not DECLINE).
  //   2) A sibling tab on this same user_id declined on our behalf
  //      (server-side echo, ``resolved_by: sibling``). Either way the
  //      modal is now stale — close it silently.
  onConnectEvent('decline', ({ data }) => {
    const callId = String(data?.call_id || '');
    if (_modalState && _modalState.callId === callId) {
      _closeModal('declined_elsewhere');
    }
  });

  // EVENT_ACCEPT here is the sibling-fanout echo: another tab on the
  // same user_id accepted this invite. The other tab owns the call
  // now — close our ringing surface so we don't double-prompt and so
  // the user doesn't accidentally decline (which would kill the
  // accepted call before the server's race guard catches it).
  onConnectEvent('accept', ({ data }) => {
    const callId = String(data?.call_id || '');
    if (_modalState && _modalState.callId === callId) {
      _closeModal('accepted_elsewhere');
    }
  });

  // Signaling died — drop the modal rather than freeze with a
  // call_id the user can't act on.
  onConnectEvent('__closed', () => {
    if (_modalState) _closeModal('signaling_lost');
  });

  // Some other surface (e.g. notification banner) already handled
  // this invite — dismiss our modal so we don't double-prompt.
  window.addEventListener('augmentum:notification-action', (evt) => {
    const detail = evt.detail || {};
    const n = detail.notification || {};
    if (!n.channel_id || !n.channel_id.startsWith('connect.call.')) return;
    const callId = String(n.payload?.call_id || '');
    if (_modalState && _modalState.callId === callId) {
      _closeModal(`handled_by_${detail.actionId || 'banner'}`);
    }
  });

  // Same-browser cross-tab dismiss. BroadcastChannel fires within a
  // few ms — much faster than the server-side echo round-trip — so a
  // user who accepts on tab 1 doesn't see tabs 2/3 ring for the
  // network's worth of latency. Cross-device (laptop ↔ phone) is
  // covered by the server-side EVENT_ACCEPT/EVENT_DECLINE echo
  // because those origins can't share a BroadcastChannel.
  initBroadcast();
  onBroadcast((msg) => {
    if (msg?.type !== 'call-changed') return;
    const callId = String(msg.call_id || '');
    if (!callId) return;
    if (_modalState && _modalState.callId === callId) {
      _closeModal(`sibling_${msg.kind || 'resolved'}`);
    }
  });
}

/** Test seam — close any open modal + reset state. */
export function resetIncomingModal() {
  _closeModal('test_reset');
}

// ── Modal DOM ────────────────────────────────────────────────────

function _ensureModal() {
  if (_modal) return _modal;
  const el = document.createElement('div');
  el.className = 'connect-incoming-overlay hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-modal', 'true');
  el.setAttribute('aria-label', 'Incoming call');
  el.innerHTML = `
    <div class="connect-incoming-card">
      <div class="connect-incoming-avatar" aria-hidden="true"></div>
      <div class="connect-incoming-name"></div>
      <div class="connect-incoming-sub">
        <span class="connect-incoming-modality-glyph" aria-hidden="true"></span>
        <span class="connect-incoming-modality-label"></span>
      </div>
      <div class="connect-incoming-actions">
        <button type="button" class="connect-incoming-decline"
                aria-label="Decline call">
          <span class="connect-incoming-action-glyph" aria-hidden="true">${icon('phone-off', { size: 28 })}</span>
          <span class="connect-incoming-action-label">Decline</span>
        </button>
        <button type="button" class="connect-incoming-accept"
                aria-label="Accept call">
          <span class="connect-incoming-action-glyph" aria-hidden="true">${icon('phone', { size: 28 })}</span>
          <span class="connect-incoming-action-label">Accept</span>
        </button>
      </div>
    </div>
  `;

  // Decline is safe to wire async — no gesture-capture needed.
  el.querySelector('.connect-incoming-decline').addEventListener(
    'click',
    () => _onDecline(),
  );

  // Accept MUST stay synchronous up through the dispatchEvent below.
  // ``incoming.js`` calls getUserMedia inside the event handler and
  // an async hop would consume the click-gesture and trigger a
  // browser permission-denied for the mic prompt.
  el.querySelector('.connect-incoming-accept').addEventListener(
    'click',
    _onAcceptGesture,
    { capture: false },
  );

  document.body.appendChild(el);
  _modal = el;
  return el;
}

function _openModal({ callId, callerDid, modalities, lifetimeMs }) {
  // Close any previous modal — newest invite wins. We treat the
  // displaced one as a passive miss; the missed-call timer on the
  // server will mark it eventually.
  if (_modalState) _closeModal('superseded');

  const el = _ensureModal();
  const wantsVideo = modalities
    .split(',')
    .map((s) => s.trim())
    .includes('video');

  // Resolve the caller's DID to a display name from the directory +
  // contacts cache. Falls back to a Title-Cased local part of the DID
  // when uncached (e.g. caller flipped discoverability off after we
  // hydrated). Never falls back to the raw ``usr_<hash>`` form.
  // No `|| callerDid` fallback: resolvePeerName already guarantees a
  // non-empty, non-raw string for a non-empty DID, so that fallback could
  // only ever fire for an empty DID -- where it would have rendered the
  // empty string anyway. Keeping it invited the raw form back in.
  const displayName = resolvePeerName(callerDid) || 'Unknown caller';
  el.querySelector('.connect-incoming-name').textContent = displayName;
  el.querySelector('.connect-incoming-modality-label').textContent =
    wantsVideo ? 'Video call' : 'Voice call';
  const glyph = el.querySelector('.connect-incoming-modality-glyph');
  glyph.innerHTML = icon(wantsVideo ? 'video' : 'phone', { size: 16 });

  // Initials-derived avatar tint — mirrors the in-call overlay's
  // tinting strategy so the visual identity is consistent across the
  // pre-call → in-call transition. Use the resolved display name as
  // the initials source so the avatar tracks the name shown above it.
  const initials = _initialsFor(displayName || callerDid);
  el.querySelector('.connect-incoming-avatar').textContent = initials;
  const color = _avatarColorFor(callerDid);
  el.style.setProperty('--connect-incoming-peer-primary', color.primary);
  el.style.setProperty('--connect-incoming-peer-secondary', color.secondary);

  el.classList.remove('hidden');

  // Auto-dismiss after invite lifetime. Server-side timer also fires
  // a missed-call notification; this just prevents the modal from
  // dangling visually.
  const timer = window.setTimeout(
    () => _closeModal('timeout'),
    Math.max(5_000, lifetimeMs),
  );

  _modalState = {
    callId,
    callerDid,
    modalities,
    expiresAt: Date.now() + lifetimeMs,
    timer,
  };

  // Kick off the ringtone — fire-and-forget; if autoplay is blocked
  // the visual signal is still present.
  try { startRingtone(); }
  catch (err) { console.warn('connect: ringtone start failed', err); }

  // Focus the accept button so keyboard users can act immediately.
  setTimeout(() => {
    el.querySelector('.connect-incoming-accept')?.focus();
  }, 0);
}

function _closeModal(reason) {
  if (!_modalState) {
    // Defensive — modal may have been opened then resetIncomingModal()
    // called; ensure the DOM hides regardless.
    if (_modal) _modal.classList.add('hidden');
    stopRingtone();
    return;
  }
  if (_modalState.timer) window.clearTimeout(_modalState.timer);
  _modalState = null;
  if (_modal) _modal.classList.add('hidden');
  try { stopRingtone(); } catch (_) { /* defensive */ }
  // Reason is intentionally not exposed via DOM event — surfacing
  // it would invite consumers to build state machines off of it.
  // The wire is the source of truth.
  void reason;
}

// ── Action handlers ──────────────────────────────────────────────

function _onAcceptGesture() {
  const state = _modalState;
  if (!state) return;

  // 1) Dispatch the synthetic notification-action event SYNCHRONOUSLY
  //    so incoming.js's gesture-capture handler can call
  //    getUserMedia() inside the same tick. The synthetic notification
  //    shape mirrors what the server publishes so incoming.js doesn't
  //    need a special code path.
  const syntheticNotification = {
    channel_id: 'connect.call.incoming',
    payload: {
      call_id: state.callId,
      initiator_did: state.callerDid,
      modalities: state.modalities,
    },
  };
  try {
    window.dispatchEvent(new CustomEvent('augmentum:notification-action', {
      detail: { notification: syntheticNotification, actionId: 'accept' },
    }));
  } catch (err) {
    console.warn('connect: incoming-modal accept dispatch failed', err);
  }

  // 2) Send MSG_ACCEPT to the server. The signaling dispatcher
  //    routes EVENT_ACCEPT to the caller and cancels the missed-call
  //    timer. This replaces the /api/notify/.../action/accept POST
  //    that the notification-banner accept would have done.
  try {
    sendEnvelope({
      verb: 'accept',
      peer: state.callerDid,
      data: { call_id: state.callId },
    });
  } catch (err) {
    console.warn('connect: MSG_ACCEPT send failed', err);
  }

  // Same-browser sibling tabs dismiss via BroadcastChannel without
  // waiting for the server's EVENT_ACCEPT echo round-trip. The server
  // echo is still authoritative — it's also what handles cross-device.
  try { broadcastCallChanged(state.callId, 'accepted'); }
  catch (err) { console.warn('connect: broadcast accept failed', err); }

  _closeModal('local_accept');
}

function _onDecline() {
  const state = _modalState;
  if (!state) return;

  try {
    sendEnvelope({
      verb: 'decline',
      peer: state.callerDid,
      data: { call_id: state.callId, reason: 'declined' },
    });
  } catch (err) {
    console.warn('connect: MSG_DECLINE send failed', err);
  }

  try { broadcastCallChanged(state.callId, 'declined'); }
  catch (err) { console.warn('connect: broadcast decline failed', err); }

  _closeModal('local_decline');
}

// ── Helpers (avatar tinting, initials) ───────────────────────────

function _initialsFor(did) {
  const local = String(did || '').split('@')[0] || did || '?';
  // First non-letter splits give us the personal handle's leading
  // glyphs; fall back to first char if the handle is opaque.
  const parts = local.split(/[._-]+/).filter(Boolean);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }
  return (local[0] || '?').toUpperCase();
}

function _avatarColorFor(did) {
  // Deterministic hue from DID — same peer always lands on the same
  // tint, so muscle-memory recognition works even before the contact
  // has a custom avatar.
  let h = 0;
  for (const ch of String(did || '')) {
    h = (h * 31 + ch.charCodeAt(0)) & 0xffffffff;
  }
  const hue = Math.abs(h) % 360;
  return {
    primary: `hsl(${hue} 65% 52%)`,
    secondary: `hsl(${(hue + 30) % 360} 65% 38%)`,
  };
}

// Test seam — exposed for unit-level inspection.
export function _getModalStateForTest() {
  return _modalState && { ..._modalState, timer: null };
}

// Re-export `escapeHtml` consumer to keep linters quiet about the
// import once we add user-controlled name rendering paths.
void escapeHtml;
