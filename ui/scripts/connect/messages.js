/* connect/messages.js — Text messaging client (HTTP + WS event glue).
 *
 * Two surfaces:
 *
 *   1) HTTP client for thread + history fetches and HTTP fallback send.
 *      Matches the server routes added to connect_routes.py:
 *         GET    /api/connect/threads
 *         GET    /api/connect/threads/{thread_id}/messages
 *         POST   /api/connect/threads/{thread_id}/send
 *         POST   /api/connect/threads/{thread_id}/mark-read
 *
 *   2) WS event dispatcher — subscribes to EVENT_TEXT_RECEIVED /
 *      EVENT_TEXT_READ / EVENT_TEXT_EDIT / EVENT_TEXT_DELETE via the
 *      shared connect/client.js and re-emits as DOM events the
 *      thread-panel listens for:
 *         augmentum:connect-message-received
 *         augmentum:connect-message-read
 *         augmentum:connect-message-edit
 *         augmentum:connect-message-delete
 *
 * The panel module (thread-panel.js) renders against these events
 * so it doesn't have to import the WS client directly.
 *
 * sendMessage() prefers the WS path when the signaling socket is
 * already open (no extra HTTP round trip) and falls back to the
 * HTTP route otherwise (e.g. WS dropped mid-compose). Either path
 * resolves with the canonical thread_id + message_id the server
 * adopted.
 */

import { broadcastThreadChanged } from './broadcast.js';
import {
  ensureConnected,
  getConnectState,
  on as onConnectEvent,
  onStateChange as onConnectStateChange,
  seedPeerStatus,
  send as sendEnvelope,
  sendAndAwaitRouted,
} from './client.js';
import * as outbox from './outbox.js';

let _wired = false;
// Per-thread cursor: ISO sent_at of the newest message we've seen.
// Persisted under augmentum:connect:cursors so a tab restart can
// resume catch-up without re-pulling history.
const CURSOR_KEY = 'augmentum:connect:cursors:v1';
let _cursors = null;

// ── Public API ──────────────────────────────────────────────────

/** Subscribe to inbound text events. Safe to call multiple times. */
export function initConnectMessaging() {
  if (_wired) return;
  _wired = true;

  onConnectEvent('text_received', ({ data, peer }) => {
    // Update per-thread cursor BEFORE firing the DOM event so any
    // listener that triggers a catch-up doesn't refetch the message
    // we just got over the WS.
    const tid = data.thread_id;
    if (tid && data.sent_at) _bumpCursor(tid, data.sent_at);
    _fire('augmentum:connect-message-received', {
      ...data, sender_did: data.sender_did || peer,
    });
    // Send a delivery ack back to the sender — the server stamps
    // delivered_at on their row and fans EVENT_TEXT_DELIVERED back.
    // Best-effort; if the WS isn't open the catch-up endpoint will
    // do the same stamping when the sender (or this client) reconnects.
    _ackDelivered(peer, tid, [data.message_id]);
  });
  onConnectEvent('text_read', ({ data }) => {
    _fire('augmentum:connect-message-read', data);
  });
  onConnectEvent('text_delivered', ({ data }) => {
    _fire('augmentum:connect-message-delivered', data);
  });
  onConnectEvent('text_edit', ({ data }) => {
    _fire('augmentum:connect-message-edit', data);
  });
  onConnectEvent('text_delete', ({ data }) => {
    _fire('augmentum:connect-message-delete', data);
  });
  onConnectEvent('text_react', ({ data, peer }) => {
    _fire('augmentum:connect-message-react', {
      ...data, reactor_did: data.reactor_did || peer,
    });
  });
  onConnectEvent('typing_start', ({ data, peer }) => {
    _fire('augmentum:connect-typing-start', {
      ...data, sender_did: peer,
    });
  });
  onConnectEvent('typing_stop', ({ data, peer }) => {
    _fire('augmentum:connect-typing-stop', {
      ...data, sender_did: peer,
    });
  });

  // On every transition into 'open' (initial + every reconnect),
  // flush the outbox, refresh the display-name + presence cache
  // (a peer who came back online during our offline window won't
  // surface a fresh EVENT_PRESENCE_UPDATE for us; the next directory
  // pull is the canonical source), and fire a window event so panels
  // can run their own catch-up.
  onConnectStateChange((state) => {
    if (state !== 'open') return;
    // Defer slightly so EVENT_WELCOME has finished propagating and
    // panels have rebuilt their thread caches.
    setTimeout(() => {
      flushOutbox().catch((err) => {
        console.warn('connect outbox flush failed', err);
      });
      refreshDisplayNameCache().catch((err) => {
        console.warn('connect display-name refresh failed', err);
      });
      _fire('augmentum:connect-reconnected', { at: Date.now() });
    }, 50);
  });

  // The moment the OS reports the network is back, flush queued sends —
  // don't wait for the signaling WS to cycle through its reconnect backoff.
  // The outbox is idempotent (server keys on message_id) so a double-trigger
  // with the WS-reconnect flush above is harmless.
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('online', () => {
      flushOutbox().catch((err) => {
        console.warn('connect outbox flush (online) failed', err);
      });
    });
  }

  // Eager first-load hydration so the first incoming call has a real
  // name on the modal instead of falling back to a Title-Cased user_id.
  refreshDisplayNameCache().catch((err) => {
    console.warn('connect display-name initial hydrate failed', err);
  });
}

/**
 * Flush queued outbound messages via the HTTP send path. HTTP-first
 * (not WS) because flush happens right after reconnect and we want
 * the durable round-trip, not another speculative WS write that
 * might race a flap.
 */
export async function flushOutbox() {
  return outbox.flush(async (item) => {
    const url = `/api/connect/threads/${encodeURIComponent(item.thread_id || item.message_id)}/send`;
    const flushBody = {
      peer_did: item.peer_did,
      thread_id: item.thread_id,
      message_id: item.message_id,
      body: item.body,
      format: item.format,
      reply_to: item.reply_to,
      attachment_ref: item.attachment_ref,
      sent_at: item.sent_at,
    };
    if (item.attachment_ref) {
      if (item.attachment_name) flushBody.attachment_name = item.attachment_name;
      if (item.attachment_mime) flushBody.attachment_mime = item.attachment_mime;
      if (item.attachment_size) flushBody.attachment_size = item.attachment_size;
    }
    const resp = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(flushBody),
    });
    if (!resp.ok) {
      let detail = '';
      try { detail = (await resp.json()).detail || ''; } catch (_) {}
      throw new Error(
        `outbox_flush_http_${resp.status}: ${
          typeof detail === 'string' ? detail : JSON.stringify(detail)
        }`,
      );
    }
    return resp.json();
  });
}

/** Re-export outbox primitives so UI panels can render queue state. */
export const outboxPending = outbox.pending;
export const outboxFailed = outbox.failed;
export const outboxHasPending = outbox.hasPending;
export const outboxHasFailed = outbox.hasFailed;
export const outboxFind = outbox.find;
export const outboxFindFailed = outbox.findFailed;
export const onOutboxChange = outbox.onChange;
export const clearOutbox = outbox.clear;
export const clearOutboxFailed = outbox.clearFailed;
export const outboxDiscard = outbox.discard;

/**
 * Retry a permanently-failed send. Resets attempts, re-queues, and
 * immediately attempts a flush. Returns the requeued item (or null
 * if the id wasn't found in the failed bucket).
 */
export async function retryFailedSend(messageId) {
  const it = outbox.retry(messageId);
  if (!it) return null;
  // Don't await flush — fire and forget so the UI can update its
  // tick state to 'pending' immediately. Errors flow back to the
  // queue → failed cycle naturally.
  flushOutbox().catch((err) => {
    console.warn('connect: retry flush failed', err);
  });
  return it;
}

/**
 * Send a typing indicator. Cheap and ephemeral — the server doesn't
 * persist these, just fans them out to the peer. Caller is responsible
 * for debouncing (one start, one stop per ~3s typing window).
 */
export function sendTyping(peerDid, threadId, isTyping) {
  if (getConnectState() !== 'open') return;
  try {
    sendEnvelope({
      verb: isTyping ? 'typing_start' : 'typing_stop',
      peer: peerDid,
      data: { thread_id: threadId },
    });
  } catch (_) { /* WS dropped — drop the indicator silently */ }
}

// ── Directory ───────────────────────────────────────────────────

/**
 * GET /api/connect/directory
 *
 * Auto-discovered, mutual-consent peer directory. Returns same-instance
 * users (and Phase 2 fabric peers) the caller can see based on both
 * sides opting into discoverability. No need for a DID — each row
 * carries display_name + peer_did + online status. Empty list when
 * the substrate is disabled (503) so callers don't need to special-case.
 */
export async function listDirectory() {
  const resp = await fetch('/api/connect/directory', {
    credentials: 'same-origin',
  });
  if (resp.status === 503) {
    return {
      people: [],
      self_discoverable_same_instance: false,
      self_discoverable_fabric_peers: false,
    };
  }
  if (!resp.ok) throw new Error(`list_directory_http_${resp.status}`);
  return await resp.json();
}

// Live server-side directory search — finds any user on the machine by
// handle/display name (not just the locally-cached directory). Returns
// { people: [...] }; empty on a blank query or 503.
export async function searchPeers(query) {
  const q = (query || '').trim();
  if (!q) return { people: [] };
  let resp;
  try {
    resp = await fetch(`/api/connect/search?q=${encodeURIComponent(q)}`, {
      credentials: 'same-origin',
    });
  } catch {
    return { people: [] };
  }
  if (!resp.ok) return { people: [] };
  const data = await resp.json();
  // Hydrate the name cache so picker rows render with friendly names.
  for (const p of data.people || []) {
    if (p.peer_did && p.display_name) _displayNameByDid.set(p.peer_did, p.display_name);
  }
  return data;
}

// ── Display-name cache ──────────────────────────────────────────
//
// Connect surfaces (incoming-call modal, in-call header, thread
// list, picker) all need to turn ``user@instance`` DIDs into a
// human display name. The server has the names in users.display_name
// + connect_contacts.peer_display_name, but they only get sent down
// via the directory + contacts endpoints. Cache the union here so
// the call UI can render synchronously without an extra round trip.
//
// Two sources hydrate the cache:
//   * /api/connect/directory   — discoverable same-instance + fabric peers
//   * /api/connect/contacts    — user's saved contacts (covers blocked
//                                / non-discoverable peers the user
//                                still has saved)
//
// On a miss, `resolvePeerName(did)` returns the local-part of the
// DID Title-Cased (the previous default behavior). The miss is what
// produced the "USR + random string" the user saw when the cache
// hadn't been hydrated yet.
const _displayNameByDid = new Map();
let _displayNameHydrating = null;  // Promise, dedupes concurrent hits

// Auto-generated account ids look like ``usr_adfce8a89def0c60`` — there is
// no human name hiding in them, so the Title-Case path turns them into
// "Usr Adfce8a89def0c60", which is worse than the raw string: it looks
// like a name and isn't one. Recognise the shape and render a short,
// honest label that still distinguishes two unnamed peers from each other.
const _GENERATED_ID = /^usr[_-]?[0-9a-f]{8,}$/i;

function _titleCaseLocalPart(did) {
  const raw = String(did || '').trim();
  if (!raw) return '';
  const [user] = raw.split('@');
  if (!user) return raw;
  if (_GENERATED_ID.test(user)) {
    return `User ${user.replace(/^usr[_-]?/i, '').slice(0, 6)}`;
  }
  return user
    .replace(/[-_.]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Best-effort display name for a Connect DID.
 *
 * Looks up the in-memory cache first; falls back to a humanised
 * Title-Case of the DID's local part when uncached. Always returns
 * a non-empty string for a non-empty DID — never returns the raw
 * ``usr_<hash>`` form.
 */
export function resolvePeerName(did) {
  const raw = String(did || '').trim();
  if (!raw) return '';
  const cached = _displayNameByDid.get(raw);
  if (cached) return cached;
  return _titleCaseLocalPart(raw);
}

/**
 * The secondary line shown under a peer's name.
 *
 * NEVER the raw DID. ``usr_adfce8a89def0c60@this-instance`` is an internal
 * identifier that leaked into the in-call header, the end-of-call card, the
 * call picker, the people list and the contact picker — it tells the user
 * nothing and reads like a bug. For a peer on this instance there is no
 * useful subtitle at all (everyone is on it), so return empty and let the
 * caller omit the line. For a fabric peer the *instance* is the one genuinely
 * informative bit — it's how you know the call is crossing machines.
 *
 * Single definition on purpose: this was previously reimplemented inside
 * ui.js while five other surfaces just interpolated the DID directly, which
 * is how the same raw tag ended up in five different places.
 */
export function peerSubtitle(did) {
  const raw = String(did || '').trim();
  if (!raw) return '';
  const [, instance] = raw.split('@');
  if (!instance || instance === 'this-instance') return '';
  return `@${instance}`;
}

/**
 * Refresh the display-name cache from directory + contacts. Idempotent
 * and concurrency-safe (a second call while one is in flight returns
 * the same promise). Errors are absorbed — a stale cache is better
 * than a thrown promise on a render path.
 *
 * Called at app boot (eager hydration so the first incoming call lands
 * with a real name) and on every signaling-WS reconnect (so a peer
 * who just turned discoverability on, or a freshly-added contact,
 * resolves without a page reload).
 */
export function refreshDisplayNameCache() {
  if (_displayNameHydrating) return _displayNameHydrating;
  _displayNameHydrating = (async () => {
    try {
      const [dir, contacts, calls] = await Promise.all([
        listDirectory().catch(() => ({ people: [] })),
        listContacts({ includeBlocked: true }).catch(() => []),
        listCalls({ limit: 100 }).catch(() => []),
      ]);
      for (const p of (dir?.people || [])) {
        const did = String(p?.peer_did || '');
        const name = String(p?.display_name || '').trim();
        if (did && name) _displayNameByDid.set(did, name);
        // Authoritative online-snapshot from the server resolves any
        // stale 'offline' the local _peerStatus cache held during the
        // signaling reconnect window.
        if (did) seedPeerStatus(did, p?.online ? 'online' : 'offline');
      }
      for (const c of (contacts || [])) {
        const did = String(c?.peer_did || '');
        const name = String(
          c?.peer_display_name || c?.display_name || '',
        ).trim();
        if (did && name) _displayNameByDid.set(did, name);
      }
      // Call history covers peers who are neither discoverable nor saved as a
      // contact (e.g. a missed call from a guest) — the server resolves the
      // username into peer_display_name, so seed those too. Don't clobber an
      // existing (directory/contact) entry.
      for (const c of (calls || [])) {
        const did = String(c?.peer_did || '');
        const name = String(c?.peer_display_name || '').trim();
        if (did && name && !_displayNameByDid.has(did)) {
          _displayNameByDid.set(did, name);
        }
      }
    } finally {
      _displayNameHydrating = null;
    }
  })();
  return _displayNameHydrating;
}

// ── Contacts ────────────────────────────────────────────────────

/** GET /api/connect/contacts */
export async function listContacts({ includeBlocked = false, tag = null } = {}) {
  const url = new URL('/api/connect/contacts', location.origin);
  url.searchParams.set('include_blocked', includeBlocked ? 'true' : 'false');
  if (tag) url.searchParams.set('tag', tag);
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) throw new Error(`list_contacts_http_${resp.status}`);
  const data = await resp.json();
  return Array.isArray(data?.contacts) ? data.contacts : [];
}

/** POST /api/connect/contacts */
export async function addContact({ peerDid, peerDisplayName = '', tags = [] }) {
  if (!peerDid || !peerDid.includes('@')) {
    throw new Error('peerDid must be user@instance');
  }
  const resp = await fetch('/api/connect/contacts', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      peer_did: peerDid,
      peer_display_name: peerDisplayName,
      tags,
    }),
  });
  if (!resp.ok) throw new Error(`add_contact_http_${resp.status}`);
  return await resp.json();
}

/** DELETE /api/connect/contacts/{contact_id} */
export async function removeContact(contactId) {
  const enc = encodeURIComponent(contactId);
  const resp = await fetch(`/api/connect/contacts/${enc}`, {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  if (!resp.ok) throw new Error(`remove_contact_http_${resp.status}`);
  return await resp.json();
}

/** PATCH /api/connect/contacts/{contact_id} — blocked + tags */
export async function patchContact(contactId, patch) {
  const enc = encodeURIComponent(contactId);
  const resp = await fetch(`/api/connect/contacts/${enc}`, {
    method: 'PATCH',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!resp.ok) throw new Error(`patch_contact_http_${resp.status}`);
  return await resp.json();
}

/** POST /api/connect/contacts/block — block/unblock by DID. Auto-creates
 *  the contact row if one doesn't already exist, so the UI can block
 *  a peer that only exists as an open thread. */
export async function setPeerBlocked(peerDid, blocked) {
  const resp = await fetch('/api/connect/contacts/block', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ peer_did: peerDid, blocked: !!blocked }),
  });
  if (!resp.ok) throw new Error(`set_peer_blocked_http_${resp.status}`);
  return await resp.json();
}

/** DELETE /api/connect/threads/{thread_id}/messages — local Clear Chat
 *  History. Peer instance keeps their copy; only the caller's rows go. */
export async function clearThreadMessages(threadId) {
  const enc = encodeURIComponent(threadId);
  const resp = await fetch(`/api/connect/threads/${enc}/messages`, {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  if (!resp.ok) throw new Error(`clear_thread_http_${resp.status}`);
  return await resp.json();
}

/** PATCH /api/connect/threads/{thread_id} — persist per-thread prefs
 *  (pin / mute / archive) for the caller. `flags` is any subset of
 *  {pinned, muted, archived}. Peer's copy is unaffected. */
export async function setThreadFlags(threadId, flags) {
  const enc = encodeURIComponent(threadId);
  const resp = await fetch(`/api/connect/threads/${enc}`, {
    method: 'PATCH',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(flags || {}),
  });
  if (!resp.ok) throw new Error(`set_thread_flags_http_${resp.status}`);
  return await resp.json();
}

// ── Calls ───────────────────────────────────────────────────────

/** GET /api/connect/calls */
export async function listCalls({ limit = 100, state = null, before = null } = {}) {
  const url = new URL('/api/connect/calls', location.origin);
  url.searchParams.set('limit', String(limit));
  if (state) url.searchParams.set('state', state);
  if (before) url.searchParams.set('before', before);
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) throw new Error(`list_calls_http_${resp.status}`);
  const data = await resp.json();
  return Array.isArray(data?.calls) ? data.calls : [];
}

/** GET /api/connect/calls/{call_id} */
export async function getCallDetail(callId) {
  const enc = encodeURIComponent(callId);
  const resp = await fetch(`/api/connect/calls/${enc}`, {
    credentials: 'same-origin',
  });
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`get_call_http_${resp.status}`);
  return await resp.json();
}

/** POST /api/connect/calls/{call_id}/rate */
export async function rateCall(callId, rating, notes = '') {
  const enc = encodeURIComponent(callId);
  const resp = await fetch(`/api/connect/calls/${enc}/rate`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ rating, notes }),
  });
  if (!resp.ok) throw new Error(`rate_call_http_${resp.status}`);
  return await resp.json();
}

/** GET /api/connect/threads */
export async function listThreads({ includeArchived = false, limit = 100 } = {}) {
  const url = new URL('/api/connect/threads', location.origin);
  url.searchParams.set('include_archived', includeArchived ? 'true' : 'false');
  url.searchParams.set('limit', String(limit));
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) throw new Error(`list_threads_http_${resp.status}`);
  const data = await resp.json();
  return Array.isArray(data?.threads) ? data.threads : [];
}

/** GET /api/connect/threads/{thread_id}/messages */
export async function listMessages(threadId, { limit = 100, before = null } = {}) {
  const url = new URL(
    `/api/connect/threads/${encodeURIComponent(threadId)}/messages`,
    location.origin,
  );
  url.searchParams.set('limit', String(limit));
  if (before) url.searchParams.set('before', before);
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (resp.status === 404) return { thread: null, messages: [] };
  if (!resp.ok) throw new Error(`list_messages_http_${resp.status}`);
  return await resp.json();
}

/**
 * Send a message. Always enqueues to the outbox first; the network
 * call is best-effort on top of that durable record. Three outcomes:
 *
 *   - WS open + ack succeeds: outbox row removed, returns the ack
 *   - WS errored, HTTP fallback succeeds: same — outbox row removed
 *   - Both fail: outbox row stays. The call still resolves with
 *     ``{queued: true, message_id, thread_id}`` so the UI can render
 *     the message as "pending" without showing an error. The next
 *     reconnect or a manual ``flushOutbox()`` retries.
 *
 * Returns ``{thread_id, message_id, routed, notification_id, queued}``.
 * ``queued`` is true when the message landed in the outbox but never
 * reached the server.
 */
export async function sendMessage({
  peerDid, threadId = '', body, format = 'plain',
  replyTo = '', attachmentRef = '',
  attachmentName = '', attachmentMime = '', attachmentSize = 0,
}) {
  if (!peerDid) {
    throw new Error('sendMessage: peerDid required');
  }
  // Either body or attachment_ref must be present — attachment-only
  // messages (e.g. shared image) are first-class.
  if (!body && !attachmentRef) {
    throw new Error('sendMessage: body or attachmentRef required');
  }
  // ``tmp:<peer-did>`` is the placeholder thread_id minted by
  // thread-panel.js::_openOrCreateThreadForPeer for a conversation
  // we haven't started yet — pure client-side fiction. Don't
  // persist it on the outbox or send it on the wire; let the server
  // mint a real thread_id via new_thread_id() on first contact.
  if (threadId.startsWith('tmp:')) threadId = '';
  const messageId = _newMessageId();
  const sentAt = new Date().toISOString();
  const payload = {
    thread_id: threadId,
    message_id: messageId,
    body,
    format,
    reply_to: replyTo,
    attachment_ref: attachmentRef,
    sent_at: sentAt,
  };
  // Sender supplies the attachment metadata on the wire so the
  // recipient's UI can pick the right render widget without an
  // extra HEAD request. The server's stored row carries only
  // attachment_ref; the metadata fields ride along on the live
  // event but aren't persisted.
  if (attachmentRef) {
    if (attachmentName) payload.attachment_name = attachmentName;
    if (attachmentMime) payload.attachment_mime = attachmentMime;
    if (attachmentSize) payload.attachment_size = attachmentSize;
  }

  // Persist BEFORE attempting the network. If the user closes the
  // tab mid-send, the queue still has the message and the next
  // session resumes from there.
  outbox.enqueue({
    message_id: messageId,
    thread_id: threadId,
    peer_did: peerDid,
    body, format,
    reply_to: replyTo,
    attachment_ref: attachmentRef,
    attachment_name: attachmentName,
    attachment_mime: attachmentMime,
    attachment_size: attachmentSize,
    sent_at: sentAt,
  });

  if (getConnectState() === 'open') {
    try {
      const ack = await sendAndAwaitRouted({
        verb: 'text_send',
        peer: peerDid,
        data: payload,
      });
      outbox.markSent(messageId);
      broadcastThreadChanged(ack.thread_id || threadId, 'send');
      return {
        thread_id: ack.thread_id || threadId,
        message_id: ack.message_id || messageId,
        routed: ack.routed || 0,
        notification_id: ack.notification_id || '',
        queued: false,
      };
    } catch (err) {
      console.warn('connect: WS send failed, falling back to HTTP', err);
    }
  }

  // HTTP fallback.
  const httpThreadKey = encodeURIComponent(threadId || messageId);
  try {
    const httpResp = await fetch(`/api/connect/threads/${httpThreadKey}/send`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ peer_did: peerDid, ...payload }),
    });
    if (!httpResp.ok) {
      let detail = '';
      try { detail = (await httpResp.json()).detail || ''; } catch (_) {}
      throw new Error(`send_message_http_${httpResp.status}: ${
        typeof detail === 'string' ? detail : JSON.stringify(detail)
      }`);
    }
    const data = await httpResp.json();
    outbox.markSent(messageId);
    broadcastThreadChanged(data.thread_id || threadId, 'send');
    return { ...data, queued: false };
  } catch (err) {
    // Both paths failed. The message is durably queued; surface a
    // soft signal to the caller so it can render "pending" rather
    // than "failed".
    outbox.markFailed(messageId, err);
    return {
      thread_id: threadId,
      message_id: messageId,
      routed: 0,
      notification_id: '',
      queued: true,
      error: String(err?.message || err || ''),
    };
  }
}

/**
 * Catch-up fetch for one thread: pull messages newer than our
 * persisted cursor. Updates the cursor on success. Returns the new
 * messages (oldest-first for easy append) or an empty array if the
 * thread had nothing new.
 *
 * The server-side route auto-stamps delivered_at + fans
 * EVENT_TEXT_DELIVERED back to senders for any of the returned
 * inbound rows that hadn't been acked yet — so the catch-up
 * fetch IS the delivery ack when the WS was down.
 */
export async function catchUpThread(threadId, { limit = 200 } = {}) {
  if (!threadId) return [];
  const since = _getCursor(threadId);
  if (!since) {
    // No cursor yet means we've never loaded this thread — let the
    // first paint do the work, nothing to catch up on.
    return [];
  }
  const url = new URL(
    `/api/connect/threads/${encodeURIComponent(threadId)}/messages`,
    location.origin,
  );
  url.searchParams.set('since', since);
  url.searchParams.set('limit', String(limit));
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (resp.status === 404) return [];
  if (!resp.ok) throw new Error(`catch_up_http_${resp.status}`);
  const data = await resp.json();
  const msgs = Array.isArray(data?.messages) ? data.messages.slice() : [];
  if (!msgs.length) return [];
  // Server returns newest-first; flip for chronological append.
  msgs.reverse();
  // Bump the cursor past the newest we just received.
  const newest = msgs[msgs.length - 1];
  if (newest && newest.sent_at) _bumpCursor(threadId, newest.sent_at);
  return msgs;
}

/** Manually bump a thread's cursor — call after rendering existing history. */
export function setThreadCursor(threadId, sentAt) {
  if (!threadId || !sentAt) return;
  _bumpCursor(threadId, sentAt);
}

/** Read a thread's persisted cursor. Returns '' if none. */
export function getThreadCursor(threadId) {
  return _getCursor(threadId);
}

/**
 * POST /api/connect/threads/{thread_id}/messages/{message_id}/react
 *
 * Add or remove an emoji reaction on a message. Prefers the WS path
 * when the signaling socket is open (no extra HTTP round trip);
 * falls back to the HTTP route otherwise. Both paths land the row on
 * BOTH sides + route an EVENT_TEXT_REACT to the peer.
 */
export async function reactToMessage({
  peerDid, threadId, messageId, emoji, action = 'add',
}) {
  if (!peerDid || !messageId || !emoji) {
    throw new Error('reactToMessage: peerDid, messageId, emoji required');
  }
  if (getConnectState() === 'open') {
    try {
      sendEnvelope({
        verb: 'text_react',
        peer: peerDid,
        data: {
          thread_id: threadId || '',
          message_id: messageId,
          emoji,
          action,
        },
      });
      broadcastThreadChanged(threadId, 'react');
      return { routed: 1, via: 'ws' };
    } catch (_) { /* fall through to HTTP */ }
  }
  const tEnc = encodeURIComponent(threadId || '');
  const mEnc = encodeURIComponent(messageId);
  const resp = await fetch(`/api/connect/threads/${tEnc}/messages/${mEnc}/react`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ peer_did: peerDid, emoji, action }),
  });
  if (!resp.ok) throw new Error(`react_http_${resp.status}`);
  broadcastThreadChanged(threadId, 'react');
  return await resp.json();
}

/** POST /api/connect/threads/{thread_id}/mark-read */
export async function markThreadRead(threadId, lastReadMessageId = '') {
  // Inline template literal — see sendMessage above for the scanner
  // pattern this avoids breaking.
  const enc = encodeURIComponent(threadId);
  const resp = await fetch(`/api/connect/threads/${enc}/mark-read`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(
      lastReadMessageId ? { last_read_message_id: lastReadMessageId } : {},
    ),
  });
  if (!resp.ok) throw new Error(`mark_read_http_${resp.status}`);
  broadcastThreadChanged(threadId, 'read');
  return await resp.json();
}

/** Convenience — kicks the WS open if it isn't already. */
export async function ensureConnectReady() {
  return ensureConnected();
}

/**
 * Upload a File/Blob through the shared /api/files/upload pipeline
 * and return the upload metadata in a Connect-friendly shape:
 *
 *   { upload_id, filename, mime, size, deduped, raw }
 *
 * The caller then passes ``upload_id`` as ``attachmentRef`` to
 * ``sendMessage``. ``raw`` is the full response item for callers
 * that want to surface server-side warnings (mime_mismatch flag etc.).
 */
/**
 * Connect-specific per-attachment ceiling. The global files upload
 * pipeline allows up to ``files_upload_max_file_bytes`` (default
 * 100MB) but Connect attachments are surfaced inline in chat bubbles
 * and routed through a notification preview — 25MB is plenty for any
 * voice note / image / short video and keeps the recipient's mobile
 * data bill tame. Server-side cap still backstops this.
 */
export const CONNECT_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024;

function _formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export async function uploadAttachment(file, { onProgress = null } = {}) {
  if (!(file instanceof Blob)) {
    throw new Error('uploadAttachment: File or Blob required');
  }
  // Pre-flight size check — reject locally before consuming the
  // user's bandwidth on a doomed upload. The server enforces the
  // same cap (and a stricter per-user quota) but seeing the error
  // before the progress bar hits 100% is a strict UX improvement.
  if (file.size > CONNECT_ATTACHMENT_MAX_BYTES) {
    const err = new Error(
      `Attachment too large (${_formatBytes(file.size)}; `
      + `max ${_formatBytes(CONNECT_ATTACHMENT_MAX_BYTES)})`,
    );
    err.code = 'attachment_too_large';
    err.size = file.size;
    err.limit = CONNECT_ATTACHMENT_MAX_BYTES;
    throw err;
  }
  const fd = new FormData();
  fd.append('file', file, file.name || 'attachment');

  // Use XHR for progress events — fetch doesn't surface upload
  // progress on most browsers. Falls back to a no-progress XHR
  // when onProgress isn't supplied; the response shape is identical.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/files/upload');
    xhr.responseType = 'json';
    xhr.withCredentials = true;
    if (typeof onProgress === 'function' && xhr.upload) {
      xhr.upload.addEventListener('progress', (ev) => {
        if (ev.lengthComputable) {
          onProgress(ev.loaded / ev.total, ev.loaded, ev.total);
        }
      });
    }
    xhr.addEventListener('error', () => reject(new Error('upload_network_error')));
    xhr.addEventListener('abort', () => reject(new Error('upload_aborted')));
    xhr.addEventListener('load', () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`upload_http_${xhr.status}`));
        return;
      }
      const data = xhr.response || {};
      const uploaded = Array.isArray(data.uploaded) ? data.uploaded : [];
      const errors = Array.isArray(data.errors) ? data.errors : [];
      if (!uploaded.length) {
        const errMsg = errors[0]?.error || 'upload_no_result';
        reject(new Error(errMsg));
        return;
      }
      const u = uploaded[0];
      resolve({
        upload_id: u.id,
        filename: u.filename,
        mime: u.mime_type || u.mime_sniffed || '',
        size: u.size_bytes || 0,
        deduped: !!u.deduped,
        raw: u,
      });
    });
    xhr.send(fd);
  });
}

/**
 * Build the URL the bubble renderer should use for an attachment.
 * Uses GET form (inline display); call with download=true for the
 * "Save as…" pattern. The server enforces participant access — no
 * extra auth header needed; same-origin credentials carry the
 * session.
 */
export function attachmentUrl(threadId, messageId, opts = {}) {
  const { download = false, fetchUrl = '', fetchToken = '' } = opts;
  // Fabric-delivered messages carry a pre-signed URL + token pair
  // pointing at the SENDER's instance. Use it verbatim so the
  // recipient's browser fetches the blob directly from the sender
  // (no proxy through our own instance). Falls back to the local
  // route for same-instance messages.
  if (fetchUrl && fetchToken) {
    const sep = fetchUrl.includes('?') ? '&' : '?';
    const tok = encodeURIComponent(fetchToken);
    const dl = download ? '&download=1' : '';
    return `${fetchUrl}${sep}token=${tok}${dl}`;
  }
  const tid = encodeURIComponent(threadId);
  const mid = encodeURIComponent(messageId);
  const qs = download ? '?download=1' : '';
  return `/api/connect/threads/${tid}/messages/${mid}/attachment${qs}`;
}

// ── Internals ───────────────────────────────────────────────────

function _fire(name, detail) {
  try {
    window.dispatchEvent(new CustomEvent(name, { detail }));
  } catch (_) { /* listener errors are non-fatal — bus is best-effort */ }
}

function _newMessageId() {
  // 24-char hex matching the server-side default in message_store.py.
  // Crypto.randomUUID is well-supported in modern browsers; fall back
  // to Math.random for very old user agents (no-op in production).
  const uuid =
    (crypto && typeof crypto.randomUUID === 'function')
      ? crypto.randomUUID()
      : `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
  return uuid.replace(/-/g, '').slice(0, 24);
}

// ── Cursor persistence (per-thread "newest seen") ──────────────

function _loadCursors() {
  if (_cursors !== null) return _cursors;
  try {
    const raw = localStorage.getItem(CURSOR_KEY);
    _cursors = raw ? (JSON.parse(raw) || {}) : {};
  } catch (_) { _cursors = {}; }
  return _cursors;
}

function _persistCursors() {
  try {
    localStorage.setItem(CURSOR_KEY, JSON.stringify(_cursors || {}));
  } catch (_) { /* quota / private mode — accept the loss */ }
}

function _getCursor(threadId) {
  const c = _loadCursors();
  return c[threadId] || '';
}

function _bumpCursor(threadId, sentAt) {
  if (!threadId || !sentAt) return;
  const c = _loadCursors();
  // Lexicographic compare works for ISO-8601 UTC timestamps.
  if (!c[threadId] || sentAt > c[threadId]) {
    c[threadId] = sentAt;
    _persistCursors();
  }
}

// ── Delivery ack ───────────────────────────────────────────────

function _ackDelivered(peerDid, threadId, messageIds) {
  if (!peerDid || !threadId || !Array.isArray(messageIds) || !messageIds.length) {
    return;
  }
  if (getConnectState() !== 'open') {
    // WS is down. The catch-up endpoint stamps + fans EVENT_TEXT_DELIVERED
    // on behalf of the recipient when the sender (or this client)
    // reconnects, so dropping the ack here is recoverable.
    return;
  }
  try {
    sendEnvelope({
      verb: 'text_delivered',
      peer: peerDid,
      data: { thread_id: threadId, message_ids: messageIds },
    });
  } catch (_) {
    // Same recovery story — catch-up endpoint will handle it.
  }
}
