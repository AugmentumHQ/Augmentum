/* connect/broadcast.js — cross-tab state sync for Connect.
 *
 * The ConnectHub already fans incoming WS frames to every active
 * connection a user has open, so an inbound EVENT_TEXT_RECEIVED
 * lands on all tabs. But OUTGOING actions on one tab (sendMessage,
 * markRead, react, edit, delete) don't get echoed back to the
 * sender's own WS — only the recipient sees them. That leaves
 * a second tab on the same account with stale state.
 *
 * BroadcastChannel is the standard fix: a same-origin pub/sub bus
 * between tabs/windows of one user agent. We broadcast a tiny
 * invalidation event after each successful outgoing action; sibling
 * tabs subscribe and re-fetch the affected thread.
 *
 * Deliberately minimal:
 *   - We do NOT mirror full message payloads through the channel.
 *     The DB is authoritative; sibling tabs fetch what they need
 *     via the existing catch-up endpoint.
 *   - We do NOT try to support cross-device sync — different origins
 *     can't share a BroadcastChannel. Real federation is the
 *     fabric-routing wedge.
 *
 * Wire shape:
 *   { type: 'thread-changed', thread_id: '…', kind: 'send'|'read'|… }
 *   { type: 'contact-changed', contact_id: '…' }
 *   { type: 'call-changed', call_id: '…' }
 *
 * Consumers should treat the broadcast as a pure invalidation hint:
 * react by refetching, not by trusting the payload.
 */

const CHANNEL_NAME = 'augmentum:connect:v1';

let _channel = null;
let _subscribers = new Set();
let _initialized = false;

// ── Public API ──────────────────────────────────────────────────

/** Initialize. Idempotent. Falls back to no-op if BroadcastChannel
    isn't available (very old browsers / Workers). */
export function initBroadcast() {
  if (_initialized) return;
  _initialized = true;
  if (typeof BroadcastChannel === 'undefined') {
    console.info('connect: BroadcastChannel unavailable — cross-tab sync disabled');
    return;
  }
  try {
    _channel = new BroadcastChannel(CHANNEL_NAME);
  } catch (err) {
    console.warn('connect: BroadcastChannel construction failed', err);
    _channel = null;
    return;
  }
  _channel.addEventListener('message', (evt) => {
    const msg = evt && evt.data;
    if (!msg || typeof msg !== 'object') return;
    for (const fn of _subscribers) {
      try { fn(msg); }
      catch (err) { console.warn('connect: broadcast subscriber failed', err); }
    }
  });
}

/**
 * Publish an invalidation to other tabs. Same-tab listeners do NOT
 * receive their own broadcasts — that's BroadcastChannel's native
 * behavior — so callers don't need to filter out echoes.
 */
export function broadcast(message) {
  if (!_channel || !message || typeof message !== 'object') return;
  try {
    _channel.postMessage(message);
  } catch (err) {
    // Likely a structured-clone failure — log but don't throw.
    console.warn('connect: broadcast postMessage failed', err);
  }
}

/** Subscribe to broadcasts. Returns an unsubscribe handle. */
export function onBroadcast(fn) {
  if (typeof fn !== 'function') return () => {};
  _subscribers.add(fn);
  return () => _subscribers.delete(fn);
}

// ── Domain-specific convenience helpers ─────────────────────────

/** A thread was mutated locally — let other tabs re-fetch it. */
export function broadcastThreadChanged(threadId, kind = 'unknown') {
  if (!threadId) return;
  broadcast({ type: 'thread-changed', thread_id: threadId, kind });
}

/** A contact (add/remove/block/tag) was mutated locally. */
export function broadcastContactChanged(contactId, kind = 'unknown') {
  if (!contactId) return;
  broadcast({ type: 'contact-changed', contact_id: contactId, kind });
}

/** A call (place/answer/decline/end/rate) was mutated locally. */
export function broadcastCallChanged(callId, kind = 'unknown') {
  if (!callId) return;
  broadcast({ type: 'call-changed', call_id: callId, kind });
}

// ── Test seam ────────────────────────────────────────────────────

export function _resetBroadcastForTest() {
  if (_channel) {
    try { _channel.close(); } catch (_) {}
    _channel = null;
  }
  _subscribers = new Set();
  _initialized = false;
}
