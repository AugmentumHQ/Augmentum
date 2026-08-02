/* connect/client.js — Connect signaling WebSocket client (singleton).
 *
 * One module-scoped client per page. Owns the WS lifecycle to
 * /api/connect/signaling, parses the envelope shape from
 * augmentum/connect/protocol.py, and dispatches events to listeners
 * registered via .on(verb, fn).
 *
 * Auth pattern matches notifications.js: POST /api/auth/ws-ticket →
 * { ticket } → open WS ?ticket=<t>. The middleware validates the
 * ticket once (single-use) and binds the user_id into the WS scope.
 *
 * Reconnect: capped exponential backoff with jitter. Lighter cap
 * than notifications because a stale signaling socket blocks new
 * calls — a half-minute retry ceiling is fine for the auxiliary
 * pinger, not for the thing that dials. Reset on a successful
 * EVENT_WELCOME (the auth handshake completed, not just TCP).
 *
 * Send API: .send({verb, data, peer, corr_id}) returns a corr_id so
 * the caller can correlate request → response (e.g. await the
 * 'routed' event for a placed invite).
 *
 * The dialer module is the only consumer today; future surfaces
 * (text threads, presence panel) plug in via the same .on() hook.
 */

import { getSettings } from '../settings.js';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

let _ws = null;
let _state = 'idle'; // idle | connecting | open | closed | disabled
let _reconnectMs = RECONNECT_BASE_MS;
let _reconnectTimer = null;
let _explicitlyClosed = false;
let _welcome = null;          // { user_did, party_id, turn, server_time }
let _onlineUserIds = new Set();
let _peerStatus = new Map();  // peer_did → 'online' | 'away' | 'dnd' | 'offline'
let _stateListeners = new Set();
let _eventListeners = new Map(); // verb → Set<fn>
let _corrCounter = 0;
let _pendingRouted = new Map(); // corr_id → resolver fn (for awaited sends)

// ── Public surface ──────────────────────────────────────────────

export function getConnectState() {
  return _state;
}

export function getWelcome() {
  return _welcome;
}

export function getOnlinePeers() {
  return Array.from(_onlineUserIds);
}

/**
 * Return a peer's presence status as a string:
 *   'online' | 'away' | 'dnd' | 'offline'
 *
 * Defaults to 'offline' for any peer we haven't received an
 * EVENT_PRESENCE_UPDATE for. UI consumers (thread list, header,
 * picker) read this for the 4-color dot.
 */
export function getPeerStatus(peerDid) {
  return _peerStatus.get(peerDid) || 'offline';
}

/**
 * Seed the presence cache from an authoritative snapshot (e.g. the
 * /directory response). Called after a signaling reconnect: any
 * EVENT_PRESENCE_UPDATE we missed during the offline window otherwise
 * leaves _peerStatus stuck on a stale 'offline'. Idempotent — no-op
 * when the value already matches.
 */
export function seedPeerStatus(peerDid, status) {
  const did = String(peerDid || '');
  const norm = String(status || 'offline').toLowerCase();
  if (!did) return;
  const valid = ['online', 'away', 'dnd', 'offline'];
  const next = valid.includes(norm) ? norm : 'offline';
  if (_peerStatus.get(did) === next) return;
  _peerStatus.set(did, next);
  if (next === 'offline') _onlineUserIds.delete(did);
  else _onlineUserIds.add(did);
}

/** Register a listener for an event verb. Returns an unsubscriber. */
export function on(verb, fn) {
  if (typeof fn !== 'function') return () => {};
  let set = _eventListeners.get(verb);
  if (!set) { set = new Set(); _eventListeners.set(verb, set); }
  set.add(fn);
  return () => set.delete(fn);
}

/** Subscribe to ('idle'|'connecting'|'open'|'closed'|'disabled') transitions. */
export function onStateChange(fn) {
  if (typeof fn !== 'function') return () => {};
  _stateListeners.add(fn);
  return () => _stateListeners.delete(fn);
}

/**
 * Ensure the WS is open (or opening). Idempotent. Resolves once
 * EVENT_WELCOME has been received (the handshake actually completed)
 * or rejects if the server reports the subsystem disabled.
 */
export async function ensureConnected() {
  if (!_isEnabledSetting()) {
    _setState('disabled');
    throw new Error('connect_disabled_setting');
  }
  if (_state === 'open' && _welcome) return _welcome;
  if (_state === 'connecting') {
    return _waitFor('welcome', 8000);
  }
  _explicitlyClosed = false;
  await _connect();
  return _waitFor('welcome', 8000);
}

/**
 * Send a Connect envelope. Returns the corr_id used so callers can
 * await a follow-up event keyed by it.
 */
export function send({ verb, data = {}, peer = '', corrId = '' }) {
  if (_state !== 'open' || !_ws) {
    throw new Error('connect_not_open');
  }
  const id = corrId || _nextCorrId();
  const env = { type: 'msg', msg: verb, id };
  if (peer) env.to = peer;
  if (data && Object.keys(data).length) env.data = data;
  _ws.send(JSON.stringify(env));
  return id;
}

/**
 * Send and wait for a server 'routed' ack matching corr_id.
 * Returns ack data ({routed, call_id?, notification_id?}) or throws
 * on EVENT_ERROR / timeout.
 */
export function sendAndAwaitRouted({ verb, data = {}, peer = '', timeoutMs = 8000 }) {
  return new Promise((resolve, reject) => {
    let corrId;
    try { corrId = send({ verb, data, peer }); }
    catch (err) { reject(err); return; }

    const timer = setTimeout(() => {
      _pendingRouted.delete(corrId);
      reject(new Error('connect_routed_timeout'));
    }, timeoutMs);

    _pendingRouted.set(corrId, (kind, payload) => {
      clearTimeout(timer);
      _pendingRouted.delete(corrId);
      if (kind === 'routed') resolve(payload || {});
      else reject(new Error(`connect_${kind}:${(payload && payload.code) || ''}`));
    });
  });
}

/** Explicit teardown (e.g. on logout). The reconnect loop stops. */
export function disconnect() {
  _explicitlyClosed = true;
  if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
  if (_ws) { try { _ws.close(); } catch (_) {} _ws = null; }
  _setState('closed');
}

// ── Internals ───────────────────────────────────────────────────

function _isEnabledSetting() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}

function _setState(next) {
  if (_state === next) return;
  _state = next;
  for (const fn of _stateListeners) {
    try { fn(next); } catch (err) { console.warn('connect state listener failed', err); }
  }
}

function _nextCorrId() {
  _corrCounter += 1;
  return `c${Date.now().toString(36)}-${_corrCounter}`;
}

async function _fetchTicket() {
  const resp = await fetch('/api/auth/ws-ticket', {
    method: 'POST',
    credentials: 'same-origin',
  });
  if (!resp.ok) throw new Error(`ws_ticket_http_${resp.status}`);
  const data = await resp.json();
  if (!data.ticket) throw new Error('ws_ticket_missing');
  return data.ticket;
}

async function _connect() {
  if (_state === 'connecting' || _state === 'open') return;
  _setState('connecting');

  let ticket;
  try {
    ticket = await _fetchTicket();
  } catch (err) {
    _setState('closed');
    _scheduleReconnect();
    throw err;
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/connect/signaling?ticket=${encodeURIComponent(ticket)}`;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch (err) {
    _setState('closed');
    _scheduleReconnect();
    throw err;
  }
  _ws = ws;

  ws.onopen = () => {
    // TCP/WS upgrade done — auth handshake (welcome) follows on first
    // frame. State stays 'connecting' until welcome arrives.
  };

  ws.onmessage = (evt) => {
    let parsed;
    try { parsed = JSON.parse(evt.data); }
    catch (_) { return; }
    if (!parsed || typeof parsed !== 'object') return;
    if (parsed.type !== 'event') return;

    const verb = parsed.event;
    const data = parsed.data || {};
    const peer = parsed.from || '';
    const corrId = parsed.id || '';

    if (verb === 'welcome') {
      _welcome = data;
      _setState('open');
      _reconnectMs = RECONNECT_BASE_MS;
    } else if (verb === 'presence_update') {
      const peerDid = data.peer_did;
      if (peerDid) {
        const status = String(data.status || 'offline').toLowerCase();
        const valid = ['online', 'away', 'dnd', 'offline'];
        const norm = valid.includes(status) ? status : 'offline';
        _peerStatus.set(peerDid, norm);
        if (norm === 'offline') _onlineUserIds.delete(peerDid);
        else _onlineUserIds.add(peerDid);
      }
    } else if (verb === 'routed' && _pendingRouted.has(corrId)) {
      _pendingRouted.get(corrId)('routed', data);
    } else if (verb === 'error' && _pendingRouted.has(corrId)) {
      _pendingRouted.get(corrId)('error', data);
    }

    // Fan out to listeners for this verb.
    const set = _eventListeners.get(verb);
    if (set) {
      for (const fn of set) {
        try { fn({ data, peer, corrId }); }
        catch (err) { console.warn(`connect listener (${verb}) failed`, err); }
      }
    }
  };

  ws.onerror = () => { /* close handler does the work */ };

  ws.onclose = () => {
    _ws = null;
    _welcome = null;
    _onlineUserIds.clear();
    _peerStatus.clear();
    _setState('closed');
    // Surface a synthetic "closed" event to anyone listening so the
    // dialer can tear down in-flight calls. We don't have a verb for
    // this on the wire — it's a local synthetic.
    const set = _eventListeners.get('__closed');
    if (set) {
      for (const fn of set) {
        try { fn({}); } catch (_) {}
      }
    }
    // Fail any pending routed waiters so awaiters don't hang.
    for (const resolver of _pendingRouted.values()) {
      resolver('error', { code: 'ws_closed', message: 'signaling socket closed' });
    }
    _pendingRouted.clear();

    _scheduleReconnect();
  };
}

function _scheduleReconnect() {
  if (_explicitlyClosed) return;
  if (!_isEnabledSetting()) return;
  if (_reconnectTimer) return;
  const jitter = Math.floor(Math.random() * 500);
  const wait = Math.min(_reconnectMs + jitter, RECONNECT_MAX_MS);
  _reconnectMs = Math.min(_reconnectMs * 2, RECONNECT_MAX_MS);
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    _connect().catch(() => {});
  }, wait);
}

function _waitFor(target, timeoutMs) {
  return new Promise((resolve, reject) => {
    if (target === 'welcome' && _welcome) { resolve(_welcome); return; }
    const timer = setTimeout(() => {
      offState();
      reject(new Error(`connect_${target}_timeout`));
    }, timeoutMs);
    const offState = onStateChange((next) => {
      if (target === 'welcome' && next === 'open' && _welcome) {
        clearTimeout(timer);
        offState();
        resolve(_welcome);
      } else if (next === 'closed' || next === 'disabled') {
        clearTimeout(timer);
        offState();
        reject(new Error(`connect_${next}`));
      }
    });
  });
}
