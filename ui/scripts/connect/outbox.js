/* connect/outbox.js — localStorage-backed outbox for Connect text sends.
 *
 * Why: messages.js::sendMessage() used to try WS, fall back to HTTP,
 * and re-throw on failure. A user composing a reply offline (or
 * mid-WS-flap) would see "Send failed" and lose the message. Phones
 * close tabs unpredictably. A power-blip drops anything in flight.
 *
 * The outbox is the durable buffer in front of the network. Sends
 * are persisted before they hit the wire and only popped once the
 * server confirms (HTTP 2xx or WS 'routed' ack). On reconnect, the
 * client.js EVENT_WELCOME hook flushes anything still queued.
 *
 * Scope is intentionally small: text-only. Typing indicators stay
 * fire-and-forget (no semantic value in replaying stale typing
 * state). Reactions / edits / deletes can adopt the same primitive
 * later — Phase 0 unblocks "messages don't get lost", which is the
 * UX-visible reliability win.
 *
 * Local format (JSON in localStorage):
 *   [
 *     {
 *       outbox_id: "<uuid>",       // local, used as React-style stable key
 *       message_id: "<24-char>",   // server-side id (also stable)
 *       thread_id: "<id>",
 *       peer_did: "user@instance",
 *       body, format, reply_to, attachment_ref,
 *       sent_at: "iso",            // when the user clicked Send
 *       enqueued_at: "iso",
 *       attempts: 0,
 *       last_error: "",            // populated when a send fails
 *       last_attempt_at: "iso"
 *     }, ...
 *   ]
 */

import { userScopedKey } from '../auth.js';

// Per-user keys (multi-tenant fix 2026-06): the outbox holds THIS user's
// unsent messages. Namespacing by user id stops a different tenant on a
// shared browser from inheriting — or flushing — someone else's queue.
const STORAGE_KEY = 'augmentum:connect:outbox:v1';
const FAILED_STORAGE_KEY = 'augmentum:connect:outbox:failed:v1';
const MAX_ATTEMPTS = 5;
// Cap on persisted queue size. If the user composes 1000 messages
// while offline, we keep the most recent N — beyond that, oldest
// gets dropped (and the UI surface that called enqueue can decide
// whether to scream).
const MAX_QUEUE = 200;
// Cap on the failed bucket — once we hit this, oldest failed entries
// fall off so a chronic problem doesn't unbounded-grow localStorage.
const MAX_FAILED = 50;

let _queue = null;             // lazy-loaded — pending sends
let _failed = null;             // lazy-loaded — sends that gave up
let _flushing = false;
const _listeners = new Set();  // change observers (UI re-render hook)

// ── Public ──────────────────────────────────────────────────────

/** Snapshot of the current queue. Treat returned objects as read-only. */
export function pending() {
  _ensureLoaded();
  return _queue.slice();
}

/** Snapshot of permanently-failed sends awaiting user decision. */
export function failed() {
  _ensureLoaded();
  return _failed.slice();
}

/** Look up one queued item by message_id (or outbox_id). */
export function find(id) {
  _ensureLoaded();
  return _queue.find((it) => it.message_id === id || it.outbox_id === id) || null;
}

/**
 * Look up a permanently-failed entry by message_id (or outbox_id).
 * Used by the bubble renderer to draw the failed state + retry row.
 */
export function findFailed(id) {
  _ensureLoaded();
  return _failed.find((it) => it.message_id === id || it.outbox_id === id) || null;
}

/** Whether anything is queued. */
export function hasPending() {
  _ensureLoaded();
  return _queue.length > 0;
}

/** Whether any failed-but-not-discarded sends exist. */
export function hasFailed() {
  _ensureLoaded();
  return _failed.length > 0;
}

/** Subscribe to "queue changed" notifications. Returns an unsubscriber. */
export function onChange(fn) {
  if (typeof fn !== 'function') return () => {};
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

/**
 * Add a message to the queue. Returns the persisted item.
 *
 * The caller (sendMessage) is expected to call attemptSend() right
 * after, but the queue is the source of truth — if the send fails
 * (or the page reloads mid-flight) the row stays here until a later
 * flush picks it up.
 */
export function enqueue(item) {
  _ensureLoaded();
  const now = new Date().toISOString();
  const stored = {
    outbox_id: _newOutboxId(),
    message_id: item.message_id,
    thread_id: item.thread_id || '',
    peer_did: item.peer_did,
    body: item.body || '',
    format: item.format || 'plain',
    reply_to: item.reply_to || '',
    attachment_ref: item.attachment_ref || '',
    sent_at: item.sent_at || now,
    enqueued_at: now,
    attempts: 0,
    last_error: '',
    last_attempt_at: '',
  };
  _queue.push(stored);
  // Honor the cap. Oldest-first eviction matches typical chat UX
  // expectations — a backlog of 200 stale messages is more likely
  // garbage than recent intent.
  while (_queue.length > MAX_QUEUE) _queue.shift();
  _persist();
  _notify();
  return stored;
}

/**
 * Mark a queued item as successfully sent. Removes it from the
 * queue. Returns whether something was actually removed.
 */
export function markSent(messageId) {
  _ensureLoaded();
  const idx = _queue.findIndex((it) => it.message_id === messageId);
  if (idx === -1) return false;
  _queue.splice(idx, 1);
  _persist();
  _notify();
  return true;
}

/**
 * Mark a send attempt as failed. Bumps attempt counter; on MAX
 * attempts the item is moved to the failed bucket and the function
 * returns 'gave_up' so the caller can surface a permanent failure.
 * Otherwise returns 'retry' (item stays in queue for the next flush).
 *
 * The failed bucket persists across reloads so the user can decide
 * whether to retry or discard after they've reconnected. Discarding
 * is the only way to silently drop a failed send.
 */
export function markFailed(messageId, error) {
  _ensureLoaded();
  const idx = _queue.findIndex((it) => it.message_id === messageId);
  if (idx === -1) return 'unknown';
  const it = _queue[idx];
  // An outage is not the message's fault. MAX_ATTEMPTS exists to retire a
  // send the SERVER keeps rejecting — burning all 5 tries during a wifi blip
  // would wrongly dump good messages into the failed bucket. While offline,
  // hold the item in the queue without advancing toward give-up; the 'online'
  // / WS-reconnect flush picks it up the moment the network returns.
  if (_isOffline()) {
    it.last_error = 'Waiting for network…';
    it.last_attempt_at = new Date().toISOString();
    _persist();
    _notify();
    return 'retry';
  }
  it.attempts += 1;
  it.last_error = String(error?.message || error || '');
  it.last_attempt_at = new Date().toISOString();
  if (it.attempts >= MAX_ATTEMPTS) {
    _queue.splice(idx, 1);
    _failed.push(it);
    // Honor the failed cap — drop oldest if it overflows. The user
    // hasn't seen these anyway by definition.
    while (_failed.length > MAX_FAILED) _failed.shift();
    _persist();
    _notify();
    return 'gave_up';
  }
  _persist();
  _notify();
  return 'retry';
}

/**
 * Move a permanently-failed item back to the queue, reset attempts,
 * and return the requeued item (or null if not found). The caller is
 * expected to trigger a flush right after — retry() doesn't dispatch
 * the network call itself.
 */
export function retry(messageId) {
  _ensureLoaded();
  const idx = _failed.findIndex((it) => it.message_id === messageId);
  if (idx === -1) return null;
  const [it] = _failed.splice(idx, 1);
  it.attempts = 0;
  it.last_error = '';
  it.last_attempt_at = '';
  _queue.push(it);
  while (_queue.length > MAX_QUEUE) _queue.shift();
  _persist();
  _notify();
  return it;
}

/**
 * Discard a failed item permanently. Used when the user gives up on
 * a stuck send (e.g. the recipient's contact info changed and the
 * message will never go through). Returns whether something was
 * removed.
 */
export function discard(messageId) {
  _ensureLoaded();
  const idx = _failed.findIndex((it) => it.message_id === messageId);
  if (idx === -1) return false;
  _failed.splice(idx, 1);
  _persist();
  _notify();
  return true;
}

/**
 * Attempt to send everything currently queued via the supplied
 * sender callback. The sender receives one queued item and resolves
 * with the server result (or throws on failure). The outbox owns
 * the markSent/markFailed bookkeeping.
 *
 * Flushes are guarded with a process-local flag so two concurrent
 * triggers (welcome + a manual nudge) don't double-send.
 *
 * Returns the count of items that were sent successfully.
 */
export async function flush(sender) {
  _ensureLoaded();
  if (_flushing) return 0;
  if (!_queue.length) return 0;
  if (typeof sender !== 'function') return 0;
  // Don't drain the queue into the network while the device is offline —
  // every send would fail instantly. Leave everything queued; the reconnect
  // / 'online' hook re-runs flush when connectivity is back.
  if (_isOffline()) return 0;
  _flushing = true;
  let sentCount = 0;
  try {
    // Snapshot — markSent mutates _queue, but we iterate the
    // snapshot so a concurrent enqueue doesn't interfere mid-flush.
    const snapshot = _queue.slice();
    for (const it of snapshot) {
      try {
        await sender(it);
        markSent(it.message_id);
        sentCount += 1;
      } catch (err) {
        const verdict = markFailed(it.message_id, err);
        // Stop the flush on the first non-recoverable burst —
        // chances are the network just dropped, and grinding
        // through 200 doomed attempts is worse than waiting for
        // the next reconnect.
        if (verdict === 'gave_up') continue;
        break;
      }
    }
  } finally {
    _flushing = false;
  }
  return sentCount;
}

/**
 * Clear the queue (NOT the failed bucket). Manual escape hatch for
 * cancelling an in-flight queue (e.g. user changed their mind about
 * a batch they composed offline). Use sparingly.
 */
export function clear() {
  _ensureLoaded();
  if (!_queue.length) return 0;
  const n = _queue.length;
  _queue = [];
  _persist();
  _notify();
  return n;
}

/** Clear the failed bucket. Equivalent to discarding every failed item. */
export function clearFailed() {
  _ensureLoaded();
  if (!_failed.length) return 0;
  const n = _failed.length;
  _failed = [];
  _persist();
  _notify();
  return n;
}

// ── Internals ───────────────────────────────────────────────────

// True only when the browser is confidently offline. `navigator.onLine`
// false-positives toward "online" (it can't see a dead upstream), so this is
// a cheap pre-filter, not a guarantee — a send that fails while onLine is true
// still counts normally.
function _isOffline() {
  return typeof navigator !== 'undefined' && navigator.onLine === false;
}

function _ensureLoaded() {
  if (_queue !== null && _failed !== null) return;
  _queue = _readArray(STORAGE_KEY);
  _failed = _readArray(FAILED_STORAGE_KEY);
}

function _readArray(key) {
  const k = userScopedKey(key);
  if (!k) return [];
  try {
    const raw = localStorage.getItem(k);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((it) => (
      it && typeof it === 'object' && it.message_id && it.peer_did
    )) : [];
  } catch (_) {
    // Corrupt storage shouldn't tank the app. Start empty.
    return [];
  }
}

function _persist() {
  const qk = userScopedKey(STORAGE_KEY);
  const fk = userScopedKey(FAILED_STORAGE_KEY);
  if (!qk || !fk) return;
  try {
    localStorage.setItem(qk, JSON.stringify(_queue));
    localStorage.setItem(fk, JSON.stringify(_failed));
  } catch (_) {
    // Quota errors or private-browsing mode — accept the loss; the
    // in-memory state still works for this session.
  }
}

function _notify() {
  for (const fn of _listeners) {
    try { fn(_queue.slice()); }
    catch (err) { console.warn('connect outbox listener failed', err); }
  }
}

function _newOutboxId() {
  try {
    if (crypto && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch (_) { /* fall through */ }
  return `obx-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
