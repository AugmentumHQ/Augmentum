/**
 * cast-guest-join.js — guest-side flow for couch co-op join.
 *
 * The guest scans a QR on the TV and lands here with ``?token=wsi_*``.
 * This page:
 *
 *   1. Reads the token from the URL.
 *   2. Opens ``/api/cast/input/ws?join_token=<token>`` directly.
 *      The middleware resolves the token to the host's user_id,
 *      bypassing the WS-ticket auth path that every other phone
 *      surface uses. No Augmentum account required.
 *   3. Reuses the same controller-producer pattern cast-control.js
 *      uses for the host (60Hz Gamepad API polling, state-delta
 *      framing, rumble dispatch). The producer is configured to skip
 *      its own session-ownership preflight since this surface has no
 *      cookie session — it dials the WS straight away.
 *   4. Surfaces UI state through five card variants:
 *        connecting / needs-pad / claim-slot / playing / terminal.
 *
 * Routing model (this is important for understanding the multi-phone
 * input path):
 *
 *   Phone A (host)    Phone B (guest)   Phone C (guest)
 *        │                 │                 │
 *        └─ ws ──┐         └─ ws ──┐         └─ ws ──┐
 *                ▼                  ▼                 ▼
 *           ┌───────────────────────────────────────┐
 *           │  CastInputRegistry on augmentum proxy │
 *           │  - one attachment_id per phone WS     │
 *           │  - slot claim: index OR firstpress    │
 *           │  - stamps slot onto each forwarded    │
 *           │    frame so the container can route   │
 *           │    to the right UInput pad            │
 *           └────────────────┬──────────────────────┘
 *                            │ (one container WS, frames
 *                            │  interleaved with slot stamp)
 *                            ▼
 *                     cast-input-bridge.py
 *                            │
 *                ┌───────────┼───────────┐
 *                ▼           ▼           ▼
 *           UInput P1   UInput P2   UInput P3
 *                            │
 *                            ▼
 *                       Emulator
 *
 * Concurrent input: each phone's WS receives frames at 60Hz, the
 * registry's ``route_input`` runs in the proxy's single event loop
 * (no locks needed), each frame gets stamped with its phone's claimed
 * slot, and the container daemon's read loop dispatches them by slot
 * to per-pad UInput devices. Frame ordering across phones isn't
 * guaranteed at the millisecond level — but ordering WITHIN a phone
 * is, and that's all the emulator cares about per player.
 */

const STATES = [
  'connecting', 'who-are-you', 'welcome-back',
  'needs-pad', 'claim-slot', 'playing', 'terminal',
];

// localStorage key for the device-uuid (Phase 3 fingerprint input).
// Phase 2 already mints + sends it so Phase 3 lands as a pure
// substrate change without the frontend needing a re-edit.
const LS_DEVICE_UUID = 'augmentum.cast.guest.deviceUuid';

function _ensureDeviceUuid() {
  try {
    let id = localStorage.getItem(LS_DEVICE_UUID) || '';
    if (!id) {
      id = (crypto?.randomUUID ? crypto.randomUUID() : _fallbackUuid());
      localStorage.setItem(LS_DEVICE_UUID, id);
    }
    return id;
  } catch {
    return _fallbackUuid();
  }
}

function _fallbackUuid() {
  // Very-low-entropy fallback for environments without
  // crypto.randomUUID (insecure context). Good enough — collisions
  // within a household are vanishingly rare and worst case is "the
  // guest gets re-asked for their name."
  return 'fp-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function _uaHash() {
  // Cheap UA + viewport fingerprint. Adds defence against the case
  // where someone clears localStorage but otherwise hasn't changed
  // the device — Phase 3 uses this alongside device_uuid.
  try {
    const parts = [
      navigator.userAgent || '',
      window.screen?.width || 0,
      window.screen?.height || 0,
      window.devicePixelRatio || 1,
    ].join('|');
    let hash = 0;
    for (let i = 0; i < parts.length; i++) {
      hash = ((hash << 5) - hash) + parts.charCodeAt(i);
      hash |= 0;
    }
    return String(hash);
  } catch {
    return '';
  }
}

const DEVICE_UUID = _ensureDeviceUuid();
const UA_HASH = _uaHash();
let _resolvedToken = '';
let _resolvedProfileId = '';

function setState(name, detail = {}) {
  for (const s of STATES) {
    const card = document.querySelector(`[data-state="${s}"]`);
    if (card) card.hidden = (s !== name);
  }
  if (name === 'playing') {
    const label = document.querySelector('[data-slot-label]');
    if (label) {
      label.textContent = detail.slotNumber
        ? `Playing as Player ${detail.slotNumber}`
        : 'Playing';
    }
  }
  if (name === 'terminal') {
    const title = document.querySelector('[data-terminal-title]');
    const sub = document.querySelector('[data-terminal-sub]');
    if (title) title.textContent = detail.title || 'Disconnected';
    if (sub) sub.textContent = detail.sub || '';
  }
}

function readToken() {
  try {
    return new URLSearchParams(location.search).get('token') || '';
  } catch {
    return '';
  }
}

function hasGamepad() {
  try {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      if (p && p.connected) return true;
    }
  } catch {}
  return false;
}


/* ── Inline producer ────────────────────────────────────────────── */
/*
 * Almost the same shape as ui/cast-control/controller-producer.js,
 * but the auth path is different: the host opens with cookie auth
 * via ``?session_id=<id>``, the guest opens with token auth via
 * ``?join_token=<token>``. We can't share the host module verbatim
 * because it builds its URL from a session_id stored in cast-control
 * state — the guest has neither cookie nor session_id.
 */

const POLL_MS = 16;
const SEND_KEEPALIVE_MS = 2000;
const NUM_BUTTONS = 17;
const NUM_AXES = 4;
// Latency probe cadence — request an echo every Nth frame so we
// have a current RTT estimate without doubling the wire bandwidth.
// 1/30 → ~2Hz at 60fps which is ample for jitter tracking.
const ECHO_EVERY_N_FRAMES = 30;

let _ws = null;
let _seq = 0;
let _pollTimer = null;
let _lastSentState = null;
let _hasClaimedSlot = false;
// Most recent measured RTT in ms; the producer attaches this to each
// outbound frame so the server has the phone-local view. EMA-smoothed
// so a single spike doesn't pollute the running estimate.
let _rttMs = 0;
// Pending echo correlations: seq → t_send (perf-now ms).
const _pendingEchoes = new Map();

function _onMessage(ev) {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  if (!msg || typeof msg !== 'object') return;
  if (msg.kind === 'rumble') {
    _handleRumble(msg);
  } else if (msg.kind === 'echo') {
    _handleEcho(msg);
  }
}

function _handleEcho(msg) {
  // Server echoed back a frame we marked echo=true. RTT = now - t_send.
  // Smoothed via EMA so a single jitter spike doesn't yank the value.
  const seq = msg.seq;
  const sent = _pendingEchoes.get(seq);
  if (sent === undefined) return;
  _pendingEchoes.delete(seq);
  const rtt = performance.now() - sent;
  if (rtt < 0 || rtt > 5000) return;  // clamp obvious garbage
  if (_rttMs === 0) {
    _rttMs = rtt;
  } else {
    // α=0.3: responsive to changes but resistant to single-frame spikes.
    _rttMs = 0.7 * _rttMs + 0.3 * rtt;
  }
}

function _handleRumble({ slot, duration_ms, strong, weak }) {
  const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
  const target = pads.find(p => p && p.connected && p.vibrationActuator);
  const actuator = target?.vibrationActuator;
  if (!actuator || typeof actuator.playEffect !== 'function') return;
  const duration = Math.max(0, Math.min(5000, Number(duration_ms) || 0));
  const strongMagnitude = Math.max(0, Math.min(1, Number(strong) || 0));
  const weakMagnitude = Math.max(0, Math.min(1, Number(weak) || 0));
  try {
    actuator.playEffect('dual-rumble', {
      duration, strongMagnitude, weakMagnitude, startDelay: 0,
    });
  } catch {}
  // Slot is hinted server-side. Show "Playing as Player N" the first
  // time we get rumble back — the server claimed our slot at that
  // point so we know our pad assignment.
  if (!_hasClaimedSlot && typeof slot === 'number' && slot >= 0) {
    _hasClaimedSlot = true;
    setState('playing', { slotNumber: slot + 1 });
  }
}

function _tick() {
  if (!_ws || _ws.readyState !== 1) return;
  const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
  if (!pads.some(p => p && p.connected)) {
    if (_hasClaimedSlot) {
      // Active player just lost their controller — surface needs-pad
      // (but don't tear down the slot — server holds it until WS dies).
      setState('needs-pad');
    } else if (document.querySelector('[data-state="needs-pad"]')?.hidden !== false) {
      setState('needs-pad');
    }
    return;
  }
  const now = performance.now();
  for (const pad of pads) {
    if (!pad || !pad.connected) continue;
    const buttons = new Array(NUM_BUTTONS).fill(0);
    const axes = new Array(NUM_AXES).fill(0);
    for (let i = 0; i < Math.min(NUM_BUTTONS, pad.buttons.length); i++) {
      const b = pad.buttons[i];
      buttons[i] = (b && (b.pressed || b.value > 0.5)) ? 1 : 0;
    }
    for (let i = 0; i < Math.min(NUM_AXES, pad.axes.length); i++) {
      axes[i] = Math.round(Number(pad.axes[i] || 0) * 100) / 100;
    }
    const cache = _lastSentState || { buttons: new Array(NUM_BUTTONS).fill(0),
                                       axes: new Array(NUM_AXES).fill(0),
                                       lastSendMs: 0 };
    const changed = !_eqArr(buttons, cache.buttons) || !_eqArr(axes, cache.axes);
    const keepalive = now - cache.lastSendMs >= SEND_KEEPALIVE_MS;
    if (!changed && !keepalive) {
      // Even without a state delta — if the host's controller just
      // came back, transition out of the placeholder state.
      if (_hasClaimedSlot) setState('playing', {});
      continue;
    }
    _lastSentState = { buttons, axes, lastSendMs: now };
    _seq += 1;
    // Latency #4: request a server echo every Nth frame so we can
    // measure RTT, and piggyback our most recent measurement onto
    // every frame so the server has a phone-side view too.
    const wantEcho = (_seq % ECHO_EVERY_N_FRAMES) === 0;
    if (wantEcho) {
      _pendingEchoes.set(_seq, now);
      // GC pending-echo map if it grows (bad WS, missed echoes).
      if (_pendingEchoes.size > 16) {
        const oldestSeq = _pendingEchoes.keys().next().value;
        _pendingEchoes.delete(oldestSeq);
      }
    }
    try {
      _ws.send(JSON.stringify({
        seq: _seq,
        t_send: now,
        echo: wantEcho || undefined,
        rtt_ms: _rttMs > 0 ? Math.round(_rttMs) : undefined,
        event: {
          kind: 'gamepad_state',
          pad_index: pad.index,
          buttons, axes,
        },
      }));
    } catch {
      break;
    }
    // Phase 1: firstpress only — we ask the server to claim a slot
    // when ANY button is pressed. The server stamps the slot in
    // outbound frames it sends to the container, and we learn our
    // slot number when rumble routes back to us (see _handleRumble).
    // Until then, show "claim-slot" so the user knows to press.
    if (!_hasClaimedSlot && buttons.some(v => v === 1)) {
      setState('claim-slot');  // server will respond with slot stamp
    }
    break;  // only send one pad per tick — guest is one player
  }
}

function _eqArr(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}


/* ── Bootstrap ──────────────────────────────────────────────────── */


async function _identifyAndClaim() {
  setState('connecting');
  let identifyResult;
  try {
    const r = await fetch('/api/cast/guest/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token: _resolvedToken,
        device_uuid: DEVICE_UUID,
        ua_hash: UA_HASH,
      }),
    });
    if (!r.ok) {
      setState('terminal', {
        title: r.status === 404 ? 'Invite expired' : 'Could not check invite',
        sub: r.status === 404
          ? 'Ask the host to mint a new one.'
          : `Server returned ${r.status}.`,
      });
      return;
    }
    identifyResult = await r.json();
  } catch (err) {
    setState('terminal', {
      title: 'Network error',
      sub: 'Check your phone\'s connection and reload the page.',
    });
    return;
  }

  // Phase 3 welcome-back path — device matched a profile.
  if (identifyResult.matched && identifyResult.profile?.id) {
    const profile = identifyResult.profile;
    _showWelcomeBack(profile);
    // Slightly delayed open so the user sees who they're joining as.
    await new Promise(res => setTimeout(res, 600));
    if (document.querySelector('[data-state="welcome-back"]')?.hidden === false) {
      await _claimAndOpenWs(profile.id);
    }
    return;
  }

  // Phase 2 picker / name-entry.
  _showWhoAreYou(identifyResult.existing_profiles || []);
}

function _showWelcomeBack(profile) {
  setState('welcome-back');
  const title = document.querySelector('[data-welcome-title]');
  if (title) title.textContent = `Welcome back, ${profile.display_name}`;
}

function _showWhoAreYou(existing) {
  setState('who-are-you');
  const list = document.querySelector('[data-profile-list]');
  const divider = document.querySelector('[data-profile-divider]');
  if (!list) return;
  if (existing.length === 0) {
    list.hidden = true;
    if (divider) divider.hidden = true;
    return;
  }
  list.hidden = false;
  if (divider) divider.hidden = false;
  list.innerHTML = existing.map(p => `
    <button class="gj-profile-pill" data-profile-id="${_esc(p.id)}">
      <span class="gj-profile-dot" style="background:${_esc(p.color || '#888')}"></span>
      <span class="gj-profile-name">${_esc(p.display_name)}</span>
    </button>
  `).join('');
  list.querySelectorAll('[data-profile-id]').forEach(btn => {
    btn.addEventListener('click', () => {
      _claimAndOpenWs(btn.dataset.profileId);
    });
  });
}

function _esc(s) {
  return String(s || '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

async function _claimAndOpenWs(profileIdOrEmpty, newName = '') {
  const payload = {
    token: _resolvedToken,
    device_uuid: DEVICE_UUID,
    ua_hash: UA_HASH,
  };
  if (profileIdOrEmpty) payload.profile_id = profileIdOrEmpty;
  else if (newName) payload.new_name = newName;
  else return;

  setState('connecting');
  try {
    const r = await fetch('/api/cast/guest/claim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      _showNameError(`Server returned ${r.status}.`);
      _showWhoAreYou([]);
      return;
    }
    const body = await r.json();
    if (body.conflict && body.existing_profile?.id) {
      // Same-name collision. Surface "is that you?" inline.
      _resolvedProfileId = body.existing_profile.id;
      _showWelcomeBack(body.existing_profile);
      await new Promise(res => setTimeout(res, 800));
      await _openWsWithProfile(_resolvedProfileId);
      return;
    }
    if (!body.profile?.id) {
      _showNameError('Could not register profile.');
      return;
    }
    _resolvedProfileId = body.profile.id;
    await _openWsWithProfile(_resolvedProfileId);
  } catch (err) {
    _showNameError('Network error — try again.');
  }
}

// Resilience: track reconnect attempts so we back off on persistent
// failure but stay aggressive for transient drops (Wi-Fi handoff,
// phone backgrounded). Server holds the slot warm for ~30s, so we
// retry quickly during that window.
let _reconnectAttempts = 0;
const RECONNECT_MAX = 6;
const RECONNECT_BASE_MS = 500;

// Wake lock: kept while the controller WS is open so the phone's
// screen doesn't turn off mid-game and kill the WS as a side effect.
// Released on terminal states.
let _wakeLock = null;

async function _acquireWakeLock() {
  try {
    if (navigator?.wakeLock?.request) {
      _wakeLock = await navigator.wakeLock.request('screen');
      _wakeLock.addEventListener?.('release', () => { _wakeLock = null; });
    }
  } catch {
    // Wake Lock unsupported (older Safari) or denied — degrades
    // gracefully; the game will pause when the phone sleeps but
    // that's the user-visible existing behaviour.
  }
}

function _releaseWakeLock() {
  try { _wakeLock?.release?.(); } catch {}
  _wakeLock = null;
}

async function _openWsWithProfile(profileId) {
  // Replace the producer's _openWs call signature to include the
  // guest_profile_id query param.
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/cast/input/ws`
    + `?join_token=${encodeURIComponent(_resolvedToken)}`
    + `&guest_profile_id=${encodeURIComponent(profileId)}`;
  try {
    _ws = new WebSocket(url);
  } catch (err) {
    setState('terminal', {
      title: 'Could not connect',
      sub: 'The invite link looks malformed. Ask the host to mint a new one.',
    });
    return;
  }
  _ws.addEventListener('open', () => {
    _reconnectAttempts = 0;
    _acquireWakeLock();
    if (hasGamepad()) setState('claim-slot');
    else setState('needs-pad');
    _pollTimer = setInterval(_tick, POLL_MS);
  });
  _ws.addEventListener('message', _onMessage);
  _ws.addEventListener('close', (ev) => _onWsClose(ev, profileId));
  _ws.addEventListener('error', () => {});
}

function _onWsClose(ev, profileId) {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = null;
  _ws = null;

  // Terminal closes — no retry, surface the right message.
  if (ev.code === 4001) {
    _releaseWakeLock();
    setState('terminal', { title: 'Invite expired',
      sub: 'Ask the host to mint a fresh one.' });
    return;
  }
  if (ev.code === 1011) {
    _releaseWakeLock();
    setState('terminal', { title: 'Server unavailable',
      sub: 'The cast session may have ended.' });
    return;
  }
  if (ev.code === 1001) {
    _releaseWakeLock();
    setState('terminal', { title: 'Game ended',
      sub: 'The host stopped the session.' });
    return;
  }
  if (ev.code === 1000 && (ev.reason === 'guest_leave' || ev.reason === 'page_unload')) {
    // Intentional close — no retry.
    _releaseWakeLock();
    return;
  }

  // Transient closes — try to reconnect inside the server's
  // warm-slot window (~30s). The slot is held for our guest_profile_id
  // so we'll reclaim our slot rather than landing on a fresh one.
  if (_reconnectAttempts < RECONNECT_MAX && profileId) {
    _reconnectAttempts += 1;
    const delay = Math.min(
      RECONNECT_BASE_MS * (2 ** (_reconnectAttempts - 1)),
      5000,
    );
    setState('connecting');
    setTimeout(() => {
      if (!_ws) _openWsWithProfile(profileId);
    }, delay);
    return;
  }

  // Exhausted retries — give up.
  _releaseWakeLock();
  if (ev.code === 1008) {
    setState('terminal', { title: 'All slots taken',
      sub: 'The host can mint another invite if a player drops.' });
  } else {
    setState('terminal', {
      title: 'Disconnected',
      sub: ev.reason ? `Reason: ${ev.reason}` : 'Connection lost.',
    });
  }
}

// Re-acquire the wake lock when the user comes back to the tab
// (Wake Lock API auto-releases when the page loses visibility).
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && _ws && _ws.readyState === 1) {
    _acquireWakeLock();
  }
});

function _showNameError(msg) {
  const el = document.querySelector('[data-name-error]');
  if (el) {
    el.textContent = msg;
    el.hidden = false;
  }
}

document.querySelector('[data-name-form]')?.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const input = document.querySelector('[data-name-input]');
  const name = (input?.value || '').trim();
  if (!name) {
    _showNameError('Type a name first.');
    return;
  }
  _claimAndOpenWs('', name);
});

document.querySelector('[data-action="not-me"]')?.addEventListener('click', async () => {
  // Phase 3: forget this device + drop back to the picker.
  try {
    await fetch('/api/cast/guest/forget-device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token: _resolvedToken,
        device_uuid: DEVICE_UUID,
      }),
    });
  } catch {}
  try { localStorage.removeItem(LS_DEVICE_UUID); } catch {}
  // Re-identify without the device-uuid so the picker comes back.
  const r = await fetch('/api/cast/guest/identify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: _resolvedToken }),
  });
  const body = await r.json();
  _showWhoAreYou(body.existing_profiles || []);
});

const token = readToken();
if (!token) {
  setState('terminal', {
    title: 'Missing invite',
    sub: 'Scan the QR code on the TV again.',
  });
} else {
  _resolvedToken = token;
  _identifyAndClaim();
}

// Watch for pad connect/disconnect so the placeholder swaps in real time.
window.addEventListener('gamepadconnected', () => {
  if (_ws && _ws.readyState === 1 && !_hasClaimedSlot) {
    setState('claim-slot');
  }
});
window.addEventListener('gamepaddisconnected', () => {
  if (!hasGamepad()) {
    setState('needs-pad');
  }
});

// Stop the producer when the user leaves the page.
window.addEventListener('beforeunload', () => {
  try { _ws?.close(1000, 'page_unload'); } catch {}
});

document.querySelector('[data-action="leave"]')?.addEventListener('click', () => {
  try { _ws?.close(1000, 'guest_leave'); } catch {}
  setState('terminal', { title: 'Left the game', sub: 'You can rejoin from the same QR if slots are open.' });
});
