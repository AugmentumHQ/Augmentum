/* connect/ui.js — Connect picker dialog + in-call overlay.
 *
 * Two surfaces:
 *
 *   1) Picker dialog — modal with peer DID input + Place-call button.
 *      Phase-1 minimal: a single text input. Presence-aware listing
 *      ships when the contacts table is populated; the WS already
 *      broadcasts EVENT_PRESENCE_UPDATE so the data path is ready.
 *
 *   2) In-call overlay — fixed bottom-right, shows the call's named
 *      state (Connecting / Ringing / Connected), peer label, mute,
 *      and hangup. Hosts the <audio> element that plays the remote
 *      stream once .ontrack fires.
 *
 * Wired entry points:
 *   - Command palette: "Connect: Place a call" (id connect.placeCall)
 *   - window.augmentumConnect.placeCall(peerDid) for console / debug
 *
 * The top-right mic long-press from the design spec is intentionally
 * deferred — there isn't a single global mic launcher in the chrome
 * yet, so wedging long-press into a non-canonical button would
 * regress the discovery surface. The command-palette entry is the
 * discoverable launcher for now; the spec's mic-long-press will
 * replace it once the global mic button lands.
 */

import { escapeHtml, showToast } from '../app.js';
import {
  canShareScreen,
  getPreferredVideoDeviceId,
  listVideoDevices,
  onDeviceChange,
  openCameraStream,
  setPreferredVideoDeviceId,
  stopStream,
} from '../camera.js';
import { getSettings } from '../settings.js';
import { registerCommand } from '../command-palette.js';
import { getOnlinePeers, getPeerStatus, getWelcome } from './client.js';
import { CALL_STATES, placeCall } from './dialer.js';
import { icon } from './icons.js';
import { mountMintForm } from './invite-mint.js';
import { initIncomingCallConsumer } from './incoming.js';
import { initIncomingCallModal } from './incoming-modal.js';
import {
  listCalls, listContacts, listDirectory, peerSubtitle, resolvePeerName,
} from './messages.js';
import { openMessagingPanelForPeer } from './thread-panel.js';

let _picker = null;
let _overlay = null;
let _audioEl = null;
// document/window listeners survive overlay teardown — see _ensureOverlay.
let _globalCallListenersWired = false;
let _globalWindowListenersWired = false;
let _activeSession = null;
let _initialized = false;
let _deferredRetryArmed = false;

// Picker preview state. The self-preview <video> tile + camera picker
// only matter while the dialog is open with Video toggled on, so the
// stream is lazily started/stopped to avoid holding the camera open.
let _previewStream = null;
let _previewDeviceId = '';
let _previewDeviceChangeUnsub = null;

// Three-phase lobby copy per the FaceTime / Meet pattern:
//   * CONNECTING — we're acquiring mic + opening signaling. "Calling…"
//     reads better than "Connecting…" because the user just tapped Call.
//   * RINGING    — invite delivered, waiting for the receiver to accept.
//   * NEGOTIATING — they accepted; SDP / media handshake in flight.
//   * CONNECTED  — label is suppressed at render time so the duration
//     ticker takes the visual focus.
//
// Receiver-side states (pending / accepting from incoming.js) get
// their own labels so the same overlay is honest from both sides.
const QUALITY_LABEL = {
  // Shown only on degraded buckets — at excellent/good the pill is
  // hidden to keep the chrome quiet (Skype / Meet pattern: only
  // surface signal when it matters).
  measuring: '',
  excellent: '',
  good: '',
  weak: 'Weak connection',
  poor: 'Poor connection',
};

const STATE_LABEL = {
  [CALL_STATES.IDLE]: '',
  [CALL_STATES.CONNECTING]: 'Calling…',
  [CALL_STATES.RINGING]: 'Ringing…',
  [CALL_STATES.NEGOTIATING]: 'Connecting…',
  [CALL_STATES.CONNECTED]: '',
  [CALL_STATES.RECONNECTING]: 'Reconnecting…',
  [CALL_STATES.ENDED]: 'Ended',
  // Receiver-side (incoming.js uses a separate enum w/ string values
  // — these labels match by raw value so the same lookup works).
  pending: 'Incoming…',
  accepting: 'Connecting…',
  reconnecting: 'Reconnecting…',
};

export function initConnectUI() {
  // Re-entrant by design — boot calls this once, but settings.js
  // dispatches augmentum:settings-loaded after the server config
  // resolves AND augmentum:connect-enabled when the user toggles
  // the modal. Either re-call lands here.
  if (_initialized) return;
  if (!_isEnabled()) {
    _armDeferredRetry();
    return;
  }
  _initialized = true;

  _ensurePicker();
  _ensureOverlay();
  _registerCommands();
  _exposeGlobal();
  _wireGlobalMicLongPress();

  // Receiver-side WebRTC consumer. When an incoming call's Accept
  // button is clicked, this constructs the receiver session and
  // renders it in the same overlay as outgoing calls. Both shapes
  // expose ``on()`` / ``hangup()`` / ``setMicMuted()`` / ``peerDid``,
  // so the overlay glue is uniform.
  // Full-screen modal on EVENT_INVITE — accept/decline lives here
  // alongside the smaller notification banner. Both surfaces converge
  // on the same ``augmentum:notification-action`` event so the
  // gesture-capture path in incoming.js doesn't double-fire.
  initIncomingCallModal();

  initIncomingCallConsumer({
    onSession: ({ session }) => {
      if (_activeSession && _activeSession.state !== CALL_STATES.ENDED) {
        // A genuinely live call (connected / reconnecting) keeps the overlay —
        // the new incoming runs in the background. But a STALE pre-connected
        // _activeSession (e.g. a prior outgoing that never answered and didn't
        // clear) must not silently swallow a real incoming call, so replace it.
        const live = _activeSession.state === CALL_STATES.CONNECTED
          || _activeSession.state === CALL_STATES.RECONNECTING;
        if (live) return;
        try { _activeSession.hangup?.('superseded'); } catch (_) { /* best-effort */ }
        _activeSession = null;
      }
      _activeSession = session;
      _wireSessionToOverlay(session);
    },
  });
}

/**
 * Listen for the two events that can flip connectEnabled true after
 * boot's synchronous init pass already returned early:
 *   - augmentum:settings-loaded — server /api/config/ui resolved
 *   - augmentum:connect-enabled — user flipped the modal toggle
 *
 * Idempotent — only the first call attaches listeners.
 */
function _armDeferredRetry() {
  if (_deferredRetryArmed) return;
  _deferredRetryArmed = true;
  const retry = () => {
    if (_initialized || !_isEnabled()) return;
    try { initConnectUI(); } catch (e) { console.warn('[connect] deferred init failed', e); }
  };
  window.addEventListener('augmentum:settings-loaded', retry);
  window.addEventListener('augmentum:connect-enabled', retry);
}

// ── Public-ish API ──────────────────────────────────────────────

/** Open the picker dialog. Idempotent. */
export function openPicker() {
  if (!_picker) _ensurePicker();
  _picker.classList.remove('hidden');
  _setPickerError('');
  // Re-populate the lists every open — presence changes, recent-call
  // list shifts, contacts may have been added since last open.
  _refreshPickerContent().catch((err) => {
    console.warn('connect: picker refresh failed', err);
  });
  const search = _picker.querySelector('.connect-picker-search');
  if (search) {
    search.value = '';
    setTimeout(() => search.focus(), 0);
  }
}

/** Place a call programmatically. Returns the active session. */
export async function startCall(peerDid, { withVideo = false, videoDeviceId = '' } = {}) {
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return null;
  }
  if (_activeSession && _activeSession.state !== CALL_STATES.ENDED) {
    // Only a genuinely LIVE call (connected, or a connected call that's
    // briefly reconnecting) blocks placing a new one. A session stuck in a
    // pre-connection state — a call-back that's still ringing an unanswered
    // peer, or a half-failed attempt that didn't reach 'ended' — must NOT
    // wedge the dialer: the user is explicitly trying to call, so tear the
    // stale one down and proceed. This is what produced the "a call is
    // already in progress" ghost after a clean hangup+Return.
    const live = _activeSession.state === CALL_STATES.CONNECTED
      || _activeSession.state === CALL_STATES.RECONNECTING;
    if (live) {
      showToast('A call is already in progress', 'info');
      return _activeSession;
    }
    try { await _activeSession.hangup('superseded'); } catch (_) { /* best-effort */ }
    _activeSession = null;
  }
  if (!peerDid || !peerDid.includes('@')) {
    showToast('Enter an address like alice@home.alice.dev', 'warning');
    return null;
  }

  const session = placeCall({ peerDid, withVideo, videoDeviceId });
  _activeSession = session;
  _wireSessionToOverlay(session);

  try {
    await session.start();
  } catch (err) {
    // _wireSessionToOverlay listens for 'error' / 'ended' — the overlay's
    // error handler already showed the friendly toast. Defensively clear the
    // active-session slot here too: if start() threw a raw error that never
    // routed through _fail (so no 'ended' fired), the slot would otherwise
    // stay pinned and wedge the next call with a phantom "already in progress".
    if (_activeSession === session) _activeSession = null;
    console.warn('connect: start failed', err);
    throw err;
  }
  return session;
}

// ── Picker dialog ───────────────────────────────────────────────
//
// Per the Connect spec, the picker is contact-first: contacts list
// with presence indicators on top, recent calls underneath, and an
// expandable "Call by DID..." footer for unknown peers. Video toggle
// + a single-tap row immediately starts the call.

function _ensurePicker() {
  if (_picker) return _picker;
  const el = document.createElement('div');
  el.className = 'connect-picker-overlay hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', 'Place a Connect call');
  el.innerHTML = `
    <div class="connect-picker-card">
      <div class="connect-picker-head">
        <div class="connect-picker-headtext">
          <div class="connect-picker-title">Call someone</div>
          <div class="connect-picker-sub">Tap a contact, or call by DID below.</div>
        </div>
        <button class="connect-picker-close" type="button" aria-label="Close">&#x2715;</button>
      </div>
      <div class="connect-picker-toolbar">
        <input type="search" class="connect-picker-search"
               placeholder="Search contacts" autocomplete="off" spellcheck="false">
        <label class="connect-picker-video" title="Start call with video">
          <input type="checkbox" class="connect-picker-video-toggle">
          <span class="connect-picker-video-glyph" aria-hidden="true">${icon('video', { size: 14 })}</span>
          <span class="connect-picker-video-label">Video</span>
        </label>
      </div>
      <div class="connect-picker-preview" hidden>
        <div class="connect-picker-preview-frame">
          <video class="connect-picker-preview-video" autoplay playsinline muted></video>
          <div class="connect-picker-preview-placeholder" hidden>
            <span class="connect-picker-preview-placeholder-glyph" aria-hidden="true">${icon('video-off', { size: 22 })}</span>
            <span class="connect-picker-preview-placeholder-text">Camera unavailable</span>
          </div>
        </div>
        <div class="connect-picker-preview-controls">
          <label class="connect-picker-camera-label" for="connect-picker-camera-select">Camera</label>
          <select class="connect-picker-camera-select" id="connect-picker-camera-select"></select>
        </div>
      </div>
      <div class="connect-picker-visibility-hint" hidden>
        <span>You're invisible to housemates.</span>
        <button class="connect-picker-visibility-btn" type="button">Be discoverable</button>
      </div>
      <div class="connect-picker-hero" hidden>
        <div class="connect-picker-hero-glyph">${icon('phone', { size: 56 })}</div>
        <div class="connect-picker-hero-title">Connect with people</div>
        <div class="connect-picker-hero-sub"></div>
        <div class="connect-picker-hero-actions"></div>
        <div class="connect-picker-hero-tip">
          <span class="connect-picker-hero-tip-glyph">${icon('star', { size: 14 })}</span>
          <span class="connect-picker-hero-tip-text"></span>
        </div>
      </div>
      <div class="connect-picker-scroll">
        <div class="connect-picker-section connect-picker-directory-section">
          <div class="connect-picker-section-head">People here</div>
          <div class="connect-picker-list connect-picker-directory"></div>
        </div>
        <div class="connect-picker-section connect-picker-contacts-section">
          <div class="connect-picker-section-head">Saved contacts</div>
          <div class="connect-picker-list connect-picker-contacts"></div>
        </div>
        <div class="connect-picker-section connect-picker-recents-section">
          <div class="connect-picker-section-head">Recent calls</div>
          <div class="connect-picker-list connect-picker-recents"></div>
        </div>
      </div>
      <div class="connect-picker-error" hidden></div>
      <details class="connect-picker-byhandle">
        <summary>Call by DID</summary>
        <div class="connect-picker-byhandle-row">
          <input type="text" class="connect-picker-input" autocomplete="off"
                 spellcheck="false" placeholder="user@instance.host">
          <button class="connect-picker-call" type="button">Call</button>
        </div>
      </details>
    </div>
  `;
  document.body.appendChild(el);
  _picker = el;

  const card = el.querySelector('.connect-picker-card');
  const closeBtn = el.querySelector('.connect-picker-close');
  const search = el.querySelector('.connect-picker-search');
  const input = el.querySelector('.connect-picker-input');
  const callBtn = el.querySelector('.connect-picker-call');
  const videoToggle = el.querySelector('.connect-picker-video-toggle');
  const cameraSelect = el.querySelector('.connect-picker-camera-select');

  // Video toggle drives the preview on/off. Mirrors what real comms
  // apps do — toggle Video, see yourself in the lobby, pick the right
  // camera before dialing rather than the other side seeing your
  // ceiling for the first 5 seconds.
  videoToggle.addEventListener('change', () => {
    if (videoToggle.checked) _startPickerPreview();
    else _stopPickerPreview();
  });

  // Camera switching from the dropdown: persist the choice so future
  // calls + the receiver-side accept honor it, then reopen the preview
  // on the new device.
  cameraSelect.addEventListener('change', () => {
    const next = String(cameraSelect.value || '');
    if (!next) return;
    setPreferredVideoDeviceId(next);
    _previewDeviceId = next;
    if (videoToggle.checked) _startPickerPreview();
  });

  const submitByHandle = async () => {
    const peerDid = (input.value || '').trim();
    if (!peerDid) {
      _setPickerError('Enter an address first.');
      return;
    }
    if (!peerDid.includes('@')) {
      _setPickerError('Peer DID needs an @instance suffix.');
      return;
    }
    const withVideo = !!videoToggle?.checked;
    const videoDeviceId = withVideo ? (_previewDeviceId || getPreferredVideoDeviceId()) : '';
    callBtn.disabled = true;
    try {
      _closePicker();
      await startCall(peerDid, { withVideo, videoDeviceId });
    } catch (err) {
      // The wired session error handler already toasted the friendly reason;
      // re-open the picker so a mistyped address can be corrected, with a
      // calm inline note (never the raw browser/JS error).
      _setPickerError("Couldn't start the call. Check the address and try again.");
      _picker.classList.remove('hidden');
      input.focus();
      console.warn('connect: call-by-handle failed', err);
    } finally {
      callBtn.disabled = false;
    }
  };

  closeBtn.addEventListener('click', _closePicker);
  callBtn.addEventListener('click', submitByHandle);
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); submitByHandle(); }
    else if (ev.key === 'Escape') { ev.preventDefault(); _closePicker(); }
  });
  search.addEventListener('input', () => _filterPickerLists(search.value));
  search.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') { ev.preventDefault(); _closePicker(); }
  });
  el.addEventListener('click', (ev) => {
    if (ev.target === el) _closePicker();
  });
  card.addEventListener('click', (ev) => ev.stopPropagation());

  return el;
}

async function _refreshPickerContent() {
  if (!_picker) return;
  const directoryEl = _picker.querySelector('.connect-picker-directory');
  const contactsEl = _picker.querySelector('.connect-picker-contacts');
  const recentsEl = _picker.querySelector('.connect-picker-recents');
  if (!directoryEl || !contactsEl || !recentsEl) return;

  // Render skeleton row so the user sees something during load.
  directoryEl.innerHTML = '<div class="connect-picker-skeleton">Looking for people…</div>';
  contactsEl.innerHTML = '<div class="connect-picker-skeleton">Loading contacts…</div>';
  recentsEl.innerHTML = '<div class="connect-picker-skeleton">Loading recent calls…</div>';

  // All three fetches in parallel — independent endpoints.
  const [directory, contacts, recentCalls] = await Promise.all([
    listDirectory()
      .catch((err) => {
        console.warn('connect: listDirectory failed', err);
        return { people: [], self_discoverable_same_instance: false };
      }),
    listContacts({ includeBlocked: false })
      .catch((err) => { console.warn('connect: listContacts failed', err); return []; }),
    listCalls({ limit: 8 })
      .catch((err) => { console.warn('connect: listCalls failed', err); return []; }),
  ]);

  // Server-side presence (from ConnectHub) is the source of truth, but
  // we also fold in the WS-driven local cache so a peer who came online
  // since the last directory fetch shows up live.
  const onlinePeers = new Set(getOnlinePeers());
  for (const p of directory?.people || []) {
    if (p.online) onlinePeers.add(p.peer_did);
  }

  const people = directory?.people || [];
  const completelyEmpty = people.length === 0
    && contacts.length === 0
    && recentCalls.length === 0;

  if (completelyEmpty) {
    _renderPickerHero(directory);
  } else {
    _renderPickerDirectory(directoryEl, people);
    _renderPickerContacts(contactsEl, contacts, onlinePeers);
    _renderPickerRecents(recentsEl, recentCalls);
    _renderVisibilityHint(directory);
    _showPickerScroll();
  }

  _picker._directoryCache = people;
  _picker._contactsCache = contacts;
  _picker._recentsCache = recentCalls;
}

// ── Picker hero (unified empty state) ──────────────────────────
//
// When the picker has nothing to show in any of its three sections
// — no discoverable peers, no saved contacts, no recent calls — we
// collapse all three empty-state strings into a single hero block
// that pitches the feature rather than apologising for emptiness.
// Per the Meet pattern: treat empty as a normal first-run state,
// not a void.
//
// The CTA copy is context-aware:
//   * If the user has discoverability OFF, the primary action
//     becomes "Be discoverable" so they show up in housemates'
//     pickers as soon as they tick it.
//   * If discoverability is already ON, the primary action becomes
//     "Show me how to invite someone" (toast for now; will become
//     a real invite-onboarding step later).
//   * Either way, the secondary action is "Call by DID" which
//     expands the by-handle input below.

const PICKER_TIPS = [
  'Discoverable people on this Augmentum appear here automatically — no contact requests needed.',
  'Tap any avatar to start a call. Hold the mic button in the top bar for quick access.',
  'Both you and your housemate need "Discoverable" turned on to see each other.',
  'For cross-instance calls, you can paste their address below — like alice@home.alice.dev.',
];

function _renderPickerHero(directory) {
  if (!_picker) return;
  const hero = _picker.querySelector('.connect-picker-hero');
  const scroll = _picker.querySelector('.connect-picker-scroll');
  const visibility = _picker.querySelector('.connect-picker-visibility-hint');
  if (!hero || !scroll) return;

  scroll.hidden = true;
  if (visibility) visibility.hidden = true;
  hero.hidden = false;

  const titleEl = hero.querySelector('.connect-picker-hero-title');
  const subEl = hero.querySelector('.connect-picker-hero-sub');
  const actionsEl = hero.querySelector('.connect-picker-hero-actions');
  const tipTextEl = hero.querySelector('.connect-picker-hero-tip-text');

  const discoverable = !!directory?.self_discoverable_same_instance;

  if (discoverable) {
    titleEl.textContent = 'No one to call yet';
    subEl.textContent = 'When other people on this Augmentum turn on Connect, they\'ll show up here automatically.';
    actionsEl.innerHTML = `
      <button class="connect-picker-hero-primary" type="button" data-action="invite">
        ${icon('users', { size: 14 })}<span>Invite a housemate</span>
      </button>
      <button class="connect-picker-hero-secondary" type="button" data-action="bydid">
        ${icon('phone', { size: 14 })}<span>Call by DID</span>
      </button>
    `;
  } else {
    titleEl.textContent = 'Be findable';
    subEl.textContent = 'Turn on Discoverable in Settings → General → Connect so housemates on this Augmentum can find you.';
    actionsEl.innerHTML = `
      <button class="connect-picker-hero-primary" type="button" data-action="discover">
        ${icon('star', { size: 14 })}<span>Be discoverable</span>
      </button>
      <button class="connect-picker-hero-secondary" type="button" data-action="bydid">
        ${icon('phone', { size: 14 })}<span>Call by DID</span>
      </button>
    `;
  }

  // Pick a tip — random per-render so two consecutive opens don't show
  // the same hint. Stays static within one open so the user can read it.
  if (tipTextEl) {
    const idx = Math.floor(Math.random() * PICKER_TIPS.length);
    tipTextEl.textContent = PICKER_TIPS[idx];
  }

  // Wire action handlers.
  for (const btn of actionsEl.querySelectorAll('button')) {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'discover') {
        _onVisibilityHintClick();
      } else if (action === 'invite') {
        mintAndShowInvite();
      } else if (action === 'bydid') {
        const byhandle = _picker.querySelector('.connect-picker-byhandle');
        if (byhandle) {
          byhandle.open = true;
          setTimeout(() => byhandle.querySelector('.connect-picker-input')?.focus(), 50);
        }
      }
    });
  }
}

// Open the invite flow: pick how far the link needs to reach (the privacy
// ladder — same network / tailnet / anywhere), then mint a link scoped to the
// least-exposing reachable path. Invite creation is admin-only server-side, so
// a non-admin gets a clear explanation. See augmentum/connect/reachability.py
// and augmentum/proxy/auth_routes.py invite endpoints.
export function mintAndShowInvite() {
  _showInviteDialog();
}

function _showInviteDialog() {
  const prior = document.querySelector('.connect-invite-dialog');
  if (prior) prior.remove();

  const wrap = document.createElement('div');
  wrap.className = 'connect-invite-dialog';
  wrap.setAttribute('role', 'dialog');
  wrap.setAttribute('aria-label', 'Invite someone to Connect');
  wrap.innerHTML = `
    <div class="connect-invite-card">
      <div class="connect-invite-title">${icon('users', { size: 16 })}<span>Invite someone</span></div>
      <p class="connect-invite-sub">They’ll get a small app to text and call you — no full account. Revoke anytime from Guests.</p>
      <div class="connect-invite-mount" data-mount></div>
      <div class="connect-invite-actions">
        <button class="connect-invite-done" type="button">Close</button>
      </div>
    </div>`;
  const close = () => wrap.remove();
  wrap.addEventListener('click', (e) => { if (e.target === wrap) close(); });
  wrap.querySelector('.connect-invite-done').addEventListener('click', close);
  mountMintForm(wrap.querySelector('[data-mount]'), { role: 'guest' });
  document.body.appendChild(wrap);
  wrap.querySelector('.connect-invite-scope')?.focus();
}


/**
 * Render the invite generator inline (the Connect home's Invite section).
 * Delegates to the shared invite-mint module (scope-aware, QR, blocked
 * state) so all three mint sites behave identically.
 */
export function mountInviteInto(host) {
  if (!host) return;
  host.innerHTML = `
    <div class="connect-invite-inline">
      <div class="connect-invite-title">${icon('users', { size: 16 })}<span>Invite someone</span></div>
      <p class="connect-invite-sub">They’ll get a small app to text and call you — no full account. Revoke anytime from Guests.</p>
      <div class="connect-invite-mount" data-mount></div>
    </div>`;
  mountMintForm(host.querySelector('[data-mount]'), { role: 'guest' });
}

function _showPickerScroll() {
  if (!_picker) return;
  const hero = _picker.querySelector('.connect-picker-hero');
  const scroll = _picker.querySelector('.connect-picker-scroll');
  if (hero) hero.hidden = true;
  if (scroll) scroll.hidden = false;
}

function _renderPickerDirectory(host, people) {
  if (!people.length) {
    host.innerHTML = `
      <div class="connect-picker-empty">
        No one else is opted-in here yet. When other people on this
        Augmentum enable Connect &amp; discoverability, they'll appear.
      </div>
    `;
    return;
  }
  const PRESENCE_LABEL = {
    online: 'Online', away: 'Away', dnd: 'Do not disturb', offline: 'Offline',
  };
  host.innerHTML = people.map((p) => {
    const label = p.display_name || resolvePeerName(p.peer_did || '') || 'Unknown';
    const name = escapeHtml(label);
    // Raw DID stays in the SEARCH haystack (typing a full DID should still
    // find the row) but never reaches the rendered text.
    const did = escapeHtml(p.peer_did || '');
    const sub = escapeHtml(peerSubtitle(p.peer_did || ''));
    const initial = _initialFor(label);
    // WS-driven status takes precedence over the directory snapshot —
    // it's live, the snapshot is moments-old.
    const live = getPeerStatus(p.peer_did);
    const status = live !== 'offline' ? live : (p.online ? 'online' : 'offline');
    const statusLabel = PRESENCE_LABEL[status] || 'Offline';
    return `
      <div class="connect-picker-row directory"
           data-peer-did="${did}"
           data-search-haystack="${escapeHtml((name + ' ' + did).toLowerCase())}">
        <button class="connect-picker-row-body" type="button"
                data-action="call" title="Call ${name}" aria-label="Call ${name}">
          <span class="connect-picker-avatar">${escapeHtml(initial)}</span>
          <span class="connect-picker-row-text">
            <span class="connect-picker-row-name">${name}</span>
            <span class="connect-picker-row-sub">${escapeHtml(statusLabel)} · ${sub || 'on this Augmentum'}</span>
          </span>
          <span class="connect-picker-presence presence-${escapeHtml(status)}"
                title="${escapeHtml(statusLabel)}"></span>
        </button>
        <div class="connect-picker-row-actions">
          <button class="connect-picker-row-action" type="button"
                  data-action="message" title="Send a message"
                  aria-label="Message ${name}">
            ${icon('message', { size: 16 })}
          </button>
        </div>
      </div>
    `;
  }).join('');
  _wirePickerRowActions(host);
}

function _wirePickerRowActions(host) {
  for (const row of host.querySelectorAll('.connect-picker-row')) {
    const peerDid = row.dataset.peerDid;
    if (!peerDid) continue;
    // Default action — clicking the row body (avatar/name area) places
    // a call. Mirrors the prior single-action behavior so users who
    // had built muscle memory aren't surprised.
    const body = row.querySelector('.connect-picker-row-body');
    if (body) body.addEventListener('click', () => _pickerRowAction(peerDid));
    // Secondary action — message icon. Lives in its own sub-button so
    // the click doesn't bubble up to the call handler.
    for (const btn of row.querySelectorAll('[data-action="message"]')) {
      btn.addEventListener('click', (evt) => {
        evt.stopPropagation();
        _pickerRowMessageAction(peerDid);
      });
    }
  }
}

function _renderVisibilityHint(directory) {
  if (!_picker) return;
  const hint = _picker.querySelector('.connect-picker-visibility-hint');
  if (!hint) return;
  // Only show the hint if the user is NOT discoverable on the
  // same-instance scope (the most common case). When they ARE
  // discoverable but no one else has opted in, the empty-state row
  // in the directory list already explains the situation.
  const hide = !!directory?.self_discoverable_same_instance;
  hint.hidden = hide;
  if (hide) return;
  const btn = hint.querySelector('.connect-picker-visibility-btn');
  if (btn && !btn.dataset.bound) {
    btn.dataset.bound = '1';
    btn.addEventListener('click', _onVisibilityHintClick);
  }
}

function _onVisibilityHintClick() {
  // Surface the General settings tab with focus on the Connect
  // section. The user can flip the checkbox + close — next picker
  // open will re-fetch and the hint disappears.
  try {
    document.dispatchEvent(new CustomEvent('augmentum:open-settings', {
      detail: { tab: 'general' },
    }));
  } catch (err) {
    showToast('Open settings → General → Connect to be discoverable', 'info');
  }
  _closePicker();
}

function _renderPickerContacts(host, contacts, onlinePeers) {
  if (!contacts.length) {
    host.innerHTML = `
      <div class="connect-picker-empty">
        No contacts yet — call someone by DID below to add them.
      </div>
    `;
    return;
  }
  // Online first, then by name.
  const sorted = contacts.slice().sort((a, b) => {
    const aOn = onlinePeers.has(a.peer_did) ? 0 : 1;
    const bOn = onlinePeers.has(b.peer_did) ? 0 : 1;
    if (aOn !== bOn) return aOn - bOn;
    const an = (a.peer_display_name || a.peer_did || '').toLowerCase();
    const bn = (b.peer_display_name || b.peer_did || '').toLowerCase();
    return an.localeCompare(bn);
  });
  host.innerHTML = sorted.map((c) => {
    const label = (c.peer_display_name || '').trim()
      || resolvePeerName(c.peer_did || '') || 'Unknown';
    const name = escapeHtml(label);
    const did = escapeHtml(c.peer_did || '');
    const sub = escapeHtml(peerSubtitle(c.peer_did || ''));
    const initial = _initialFor(label);
    const status = getPeerStatus(c.peer_did);
    const PRESENCE_LABEL = {
      online: 'Online', away: 'Away', dnd: 'Do not disturb', offline: 'Offline',
    };
    const statusLabel = PRESENCE_LABEL[status] || 'Offline';
    return `
      <div class="connect-picker-row"
           data-peer-did="${did}"
           data-search-haystack="${escapeHtml((name + ' ' + did).toLowerCase())}">
        <button class="connect-picker-row-body" type="button"
                data-action="call" title="Call ${name}" aria-label="Call ${name}">
          <span class="connect-picker-avatar">${escapeHtml(initial)}</span>
          <span class="connect-picker-row-text">
            <span class="connect-picker-row-name">${name}</span>
            ${sub ? `<span class="connect-picker-row-sub">${sub}</span>` : ''}
          </span>
          <span class="connect-picker-presence presence-${escapeHtml(status)}"
                title="${escapeHtml(statusLabel)}"></span>
        </button>
        <div class="connect-picker-row-actions">
          <button class="connect-picker-row-action" type="button"
                  data-action="message" title="Send a message"
                  aria-label="Message ${name}">
            ${icon('message', { size: 16 })}
          </button>
        </div>
      </div>
    `;
  }).join('');
  _wirePickerRowActions(host);
}

function _renderPickerRecents(host, calls) {
  if (!calls.length) {
    host.innerHTML = '<div class="connect-picker-empty">No recent calls yet.</div>';
    return;
  }
  host.innerHTML = calls.map((c) => {
    const peerLabel = (c.peer_display_name || '').trim()
      || resolvePeerName(c.peer_did || '') || 'Unknown';
    const peer = escapeHtml(peerLabel);
    const initial = _initialFor(peerLabel);
    const arrow = c.state === 'missed'
      ? `<span class="connect-picker-row-arrow missed">${icon('phone-missed', { size: 14 })}</span>`
      : (c.direction === 'outgoing'
          ? `<span class="connect-picker-row-arrow out">${icon('arrow-up-right', { size: 14 })}</span>`
          : `<span class="connect-picker-row-arrow in">${icon('arrow-down-left', { size: 14 })}</span>`);
    const when = escapeHtml(_humaniseRelative(c.initiated_at));
    return `
      <button class="connect-picker-row recent" type="button"
              data-peer-did="${peer}"
              data-search-haystack="${escapeHtml(peer.toLowerCase())}">
        <span class="connect-picker-avatar">${escapeHtml(initial)}</span>
        <span class="connect-picker-row-text">
          <span class="connect-picker-row-name">${peer}</span>
          <span class="connect-picker-row-sub">${arrow} ${when}</span>
        </span>
      </button>
    `;
  }).join('');
  for (const row of host.querySelectorAll('.connect-picker-row')) {
    row.addEventListener('click', () => _pickerRowAction(row.dataset.peerDid));
  }
}

function _filterPickerLists(query) {
  if (!_picker) return;
  const q = (query || '').toLowerCase().trim();
  for (const row of _picker.querySelectorAll('.connect-picker-row')) {
    const hay = row.dataset.searchHaystack || '';
    row.style.display = (!q || hay.includes(q)) ? '' : 'none';
  }
}

async function _pickerRowAction(peerDid) {
  if (!peerDid) return;
  const videoToggle = _picker?.querySelector('.connect-picker-video-toggle');
  const withVideo = !!videoToggle?.checked;
  const videoDeviceId = withVideo ? (_previewDeviceId || getPreferredVideoDeviceId()) : '';
  _closePicker();
  try {
    await startCall(peerDid, { withVideo, videoDeviceId });
  } catch (err) {
    // The wired session error handler owns the user-facing toast (friendly,
    // reason-mapped). Don't double-toast a raw err.message here.
    console.warn('connect: call failed', err);
  }
}

async function _pickerRowMessageAction(peerDid) {
  if (!peerDid) return;
  _closePicker();
  try {
    await openMessagingPanelForPeer(peerDid);
  } catch (err) {
    showToast(
      `Could not open conversation: ${err?.message || 'unknown error'}`,
      'error',
    );
  }
}

function _closePicker() {
  if (_picker) _picker.classList.add('hidden');
  // Release the camera. Holding it open across picker close + reopen
  // bursts is wasteful (LEDs on, system permission indicator lit) and
  // dialer.start() needs it back anyway. Same teardown on Escape /
  // outside-click / call placement.
  _stopPickerPreview();
}

// ── Picker self-preview + camera select ───────────────────────────

async function _startPickerPreview() {
  if (!_picker) return;
  const previewBox = _picker.querySelector('.connect-picker-preview');
  const videoEl = _picker.querySelector('.connect-picker-preview-video');
  const placeholder = _picker.querySelector('.connect-picker-preview-placeholder');
  const select = _picker.querySelector('.connect-picker-camera-select');
  const controls = _picker.querySelector('.connect-picker-preview-controls');
  if (!previewBox || !videoEl || !select) return;

  previewBox.hidden = false;
  placeholder.hidden = true;
  // Hide the camera-select row until we successfully open a stream +
  // confirm 2+ devices. Without this, a failed preview leaves the
  // chrome up next to an empty <select>, which is exactly the bug a
  // phone user hits when the cert isn't trusted / permission was
  // declined — they see an empty dropdown and "nothing happens".
  if (controls) controls.hidden = true;

  // Stop the previous stream before opening a new one — Chrome will
  // happily give us TWO simultaneous handles to the same camera, which
  // shows up as two "in use" indicators and pegs the CPU.
  if (_previewStream) {
    stopStream(_previewStream);
    _previewStream = null;
  }

  // Early guards for failure modes that don't even reach getUserMedia.
  // The common phone failure: untrusted self-signed cert on iOS Safari
  // disables navigator.mediaDevices entirely; or the user landed on
  // an HTTP origin (LAN IP without HTTPS). Surface actionable text
  // instead of the generic "Camera unavailable" placeholder.
  if (!window.isSecureContext) {
    _setPreviewPlaceholder(previewBox, placeholder,
      'Camera needs a secure (https://) connection on this device.');
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    _setPreviewPlaceholder(previewBox, placeholder,
      'Camera unavailable in this browser. On iPhone, install Augmentum’s root certificate and reload.');
    return;
  }

  const deviceId = _previewDeviceId || getPreferredVideoDeviceId();
  try {
    _previewStream = await openCameraStream({
      deviceId, audio: false, video: true,
    });
  } catch (err) {
    console.warn('connect: preview failed', err);
    _setPreviewPlaceholder(previewBox, placeholder, _previewErrorText(err));
    return;
  }

  videoEl.srcObject = _previewStream;
  videoEl.play?.().catch(() => {});

  // Track which deviceId we landed on. getUserMedia may have honored
  // an `ideal` constraint to something other than the one we asked
  // for (camera unplugged after preference save); reflect reality.
  const track = _previewStream.getVideoTracks()[0];
  const settings = track?.getSettings?.();
  if (settings?.deviceId) {
    _previewDeviceId = String(settings.deviceId);
  }

  await _refreshCameraSelect();

  // Hot-update the device list when the user plugs/unplugs a camera
  // while the picker is open. One subscription per open; cleared on
  // _stopPickerPreview.
  if (!_previewDeviceChangeUnsub) {
    _previewDeviceChangeUnsub = onDeviceChange(() => {
      _refreshCameraSelect().catch(() => {});
    });
  }
}

function _stopPickerPreview() {
  if (_previewStream) {
    stopStream(_previewStream);
    _previewStream = null;
  }
  if (_previewDeviceChangeUnsub) {
    _previewDeviceChangeUnsub();
    _previewDeviceChangeUnsub = null;
  }
  if (!_picker) return;
  const previewBox = _picker.querySelector('.connect-picker-preview');
  const videoEl = _picker.querySelector('.connect-picker-preview-video');
  if (videoEl) {
    try { videoEl.srcObject = null; } catch (_) {}
  }
  if (previewBox) previewBox.hidden = true;
}

function _setPreviewPlaceholder(previewBox, placeholder, message) {
  placeholder.hidden = false;
  const text = previewBox.querySelector('.connect-picker-preview-placeholder-text');
  if (text) text.textContent = message;
}

// Map getUserMedia error names to actionable strings. Phone users hit
// NotAllowedError the most (permission denied or untrusted origin);
// the message has to explain how to recover from outside the page.
function _previewErrorText(err) {
  const name = String(err?.name || '');
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera blocked. Allow camera access in your browser’s site settings, then re-check Video.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No camera found on this device.';
  }
  if (name === 'NotReadableError') {
    return 'Camera in use by another app. Close it and try again.';
  }
  return 'Camera unavailable.';
}

async function _refreshCameraSelect() {
  if (!_picker) return;
  const select = _picker.querySelector('.connect-picker-camera-select');
  if (!select) return;

  let devices;
  try {
    devices = await listVideoDevices({ probeForLabels: false });
  } catch (_) { devices = []; }

  // Hide the select entirely when there's only one camera — picking
  // is meaningless and the row just adds chrome.
  const controls = _picker.querySelector('.connect-picker-preview-controls');
  if (controls) controls.hidden = devices.length <= 1;
  if (devices.length === 0) {
    select.innerHTML = '';
    return;
  }

  const current = _previewDeviceId || getPreferredVideoDeviceId();
  select.innerHTML = devices.map((d, i) => {
    const label = escapeHtml(d.label || `Camera ${i + 1}`);
    const selected = d.deviceId === current ? ' selected' : '';
    return `<option value="${escapeHtml(d.deviceId)}"${selected}>${label}</option>`;
  }).join('');
}

function _setPickerError(msg) {
  if (!_picker) return;
  const el = _picker.querySelector('.connect-picker-error');
  if (!el) return;
  if (msg) {
    el.textContent = msg;
    el.hidden = false;
  } else {
    el.textContent = '';
    el.hidden = true;
  }
}

function _initialFor(s) {
  const cleaned = String(s || '').trim();
  if (!cleaned) return '?';
  const beforeAt = cleaned.split('@')[0] || cleaned;
  const ch = beforeAt[0];
  return (ch || '?').toUpperCase();
}

/**
 * Display-friendly name for a peer DID.
 *
 * Thin alias over the shared resolver in messages.js. This used to be a
 * second, independent implementation living here -- which is exactly why
 * the raw ``usr_<hex>@this-instance`` form got fixed in the in-call header
 * and still leaked from the end-of-call card, the call pickers, the people
 * list and the post-call rating toast. One resolver, one behaviour.
 */
function _prettyPeerName(did) {
  return resolvePeerName(did);
}

/**
 * The DID line shown under the name. For local-instance peers
 * we hide it (it's "@this-instance" which is meaningless to a
 * non-developer). For fabric peers we show the @instance part so
 * the user knows it's a cross-instance call.
 */
function _peerSubtitle(did) {
  return peerSubtitle(did);
}

/**
 * Deterministic hue from a DID so each peer has a stable identity
 * color across calls. Maps to a hue (0-360) and pairs it with a
 * complementary hue for a two-color gradient on the avatar +
 * backdrop. Pure function of the string — no randomness.
 */
function _peerColor(did) {
  const raw = String(did || '?').trim();
  let h = 0;
  for (let i = 0; i < raw.length; i++) {
    h = (h * 31 + raw.charCodeAt(i)) >>> 0;
  }
  const hue = h % 360;
  // Stay away from theme-clashing greys; saturate enough to read
  // as identity but not so much it screams.
  return {
    hue,
    primary:   `hsl(${hue}, 62%, 56%)`,
    secondary: `hsl(${(hue + 40) % 360}, 58%, 48%)`,
    deep:      `hsl(${hue}, 50%, 18%)`,
  };
}

function _humaniseRelative(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (_) {
    return iso;
  }
}

// Suppress unused-import warning — getWelcome is exported for parity
// with thread-panel.js, may be used by future presence UI here.
void getWelcome;

// ── In-call overlay ─────────────────────────────────────────────

function _ensureOverlay() {
  if (_overlay) return _overlay;
  const el = document.createElement('div');
  el.className = 'connect-call-overlay hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', 'Active Connect call');
  el.innerHTML = `
    <div class="connect-call-backdrop"></div>
    <div class="connect-call-card">
      <div class="connect-call-videos" hidden>
        <video class="connect-call-remote-video" data-tile="remote" autoplay playsinline></video>
        <div class="connect-call-tile-off connect-call-remote-off" data-tile="remote" aria-live="polite">
          <div class="connect-call-tile-off-avatar" aria-hidden="true"></div>
          <div class="connect-call-tile-off-name"></div>
          <div class="connect-call-tile-off-glyph" aria-hidden="true">
            ${icon('video-off', { size: 18 })}
          </div>
        </div>
        <div class="connect-call-caption connect-call-remote-caption" hidden aria-hidden="true">
          <span class="connect-call-tile-name"></span>
        </div>
        <video class="connect-call-local-video" data-tile="local" autoplay playsinline muted></video>
        <div class="connect-call-tile-off connect-call-local-off" data-tile="local" aria-hidden="true">
          <div class="connect-call-tile-off-avatar"></div>
          <div class="connect-call-tile-off-name">You</div>
          <div class="connect-call-tile-off-glyph">
            ${icon('video-off', { size: 16 })}
          </div>
        </div>
        <div class="connect-call-caption connect-call-local-caption" hidden aria-hidden="true">
          <span class="connect-call-tile-name">You</span>
          <span class="connect-call-tile-suffix" hidden> · sharing</span>
        </div>
        <div class="connect-call-sharing-badge" hidden aria-live="polite">
          ${icon('monitor', { size: 14 })}
          <span>Sharing your screen</span>
        </div>
      </div>
      <div class="connect-call-headline">
        <div class="connect-call-avatar-wrap" aria-hidden="true">
          <div class="connect-call-pulse-ring"></div>
          <div class="connect-call-avatar"></div>
        </div>
        <div class="connect-call-peer-name"></div>
        <div class="connect-call-peer-subtitle"></div>
        <div class="connect-call-state"></div>
        <div class="connect-call-quality" data-bucket="measuring" hidden
             title="Connection quality"
             aria-label="Connection quality">
          <span class="connect-call-quality-dot"></span>
          <span class="connect-call-quality-label"></span>
        </div>
        <div class="connect-call-peer-muted" hidden
             aria-live="polite">
          ${icon('mic-off', { size: 14 })}
          <span>Peer muted</span>
        </div>
        <canvas class="connect-call-waveform" width="320" height="44"
                aria-hidden="true" hidden></canvas>
        <div class="connect-call-duration" hidden>00:00</div>
      </div>
      <div class="connect-call-controls">
        <button class="connect-call-mute connect-call-ctrl" type="button" aria-pressed="false" title="Mute" aria-label="Mute">
          ${icon('mic', { size: 22 })}
        </button>
        <button class="connect-call-video-toggle connect-call-ctrl" type="button" aria-pressed="true" hidden title="Camera off" aria-label="Camera off">
          ${icon('video', { size: 22 })}
        </button>
        <button class="connect-call-video-escalate connect-call-ctrl" type="button" hidden title="Add video" aria-label="Add video">
          ${icon('video-plus', { size: 22 })}
        </button>
        <button class="connect-call-camera-switch connect-call-ctrl" type="button" hidden title="Switch camera" aria-label="Switch camera">
          ${icon('video', { size: 22 })}
        </button>
        <button class="connect-call-screen-share connect-call-ctrl" type="button" hidden aria-pressed="false" title="Share screen" aria-label="Share screen">
          ${icon('monitor', { size: 22 })}
        </button>
        <button class="connect-call-fullscreen connect-call-ctrl" type="button" hidden aria-pressed="false" title="Fullscreen" aria-label="Enter fullscreen">
          ${icon('maximize', { size: 22 })}
        </button>
        <button class="connect-call-output connect-call-ctrl" type="button" hidden title="Switch output device" aria-label="Switch output device">
          ${icon('volume', { size: 22 })}
        </button>
        <button class="connect-call-hangup connect-call-ctrl danger" type="button" title="End call" aria-label="End call">
          ${icon('phone-off', { size: 22 })}
        </button>
      </div>
      <div class="connect-call-output-menu" hidden role="menu"
           aria-label="Audio output devices"></div>
      <div class="connect-call-camera-menu" hidden role="menu"
           aria-label="Camera devices"></div>
      <audio class="connect-call-audio" autoplay></audio>
    </div>
  `;
  document.body.appendChild(el);
  _overlay = el;
  _audioEl = el.querySelector('.connect-call-audio');

  const muteBtn = el.querySelector('.connect-call-mute');
  const hangupBtn = el.querySelector('.connect-call-hangup');
  const videoBtn = el.querySelector('.connect-call-video-toggle');
  const escalateBtn = el.querySelector('.connect-call-video-escalate');
  const cameraBtn = el.querySelector('.connect-call-camera-switch');
  const cameraMenu = el.querySelector('.connect-call-camera-menu');
  const shareBtn = el.querySelector('.connect-call-screen-share');
  const outputBtn = el.querySelector('.connect-call-output');
  const outputMenu = el.querySelector('.connect-call-output-menu');

  cameraBtn.addEventListener('click', () => {
    if (!_activeSession) return;
    _toggleCameraMenu(cameraMenu).catch((err) =>
      console.warn('connect: camera menu open failed', err),
    );
  });

  const fullscreenBtn = el.querySelector('.connect-call-fullscreen');
  // Hide on browsers that don't support the Fullscreen API. We check
  // once at construction time — support doesn't change at runtime.
  if (document.fullscreenEnabled || document.webkitFullscreenEnabled) {
    fullscreenBtn.hidden = false;
  }
  fullscreenBtn.addEventListener('click', async () => {
    const card = el.querySelector('.connect-call-card');
    if (!card) return;
    const inFullscreen = document.fullscreenElement === card
      || document.webkitFullscreenElement === card;
    try {
      if (inFullscreen) {
        await (document.exitFullscreen?.() || document.webkitExitFullscreen?.());
      } else {
        await (card.requestFullscreen?.() || card.webkitRequestFullscreen?.());
      }
    } catch (err) {
      // Fullscreen entry can fail outside a user gesture context or
      // when the element is detached. Quiet error — the button click
      // IS the gesture, so this is rare.
      console.warn('connect: fullscreen toggle failed', err);
    }
  });

  // Sync the button's pressed-state + icon on every fullscreen
  // transition. The Esc key + browser chrome can exit fullscreen
  // without going through our button, so we listen to the document.
  const onFsChange = () => {
    const card = el.querySelector('.connect-call-card');
    const active = (document.fullscreenElement === card)
      || (document.webkitFullscreenElement === card);
    fullscreenBtn.setAttribute('aria-pressed', active ? 'true' : 'false');
    fullscreenBtn.title = active ? 'Exit fullscreen' : 'Fullscreen';
    fullscreenBtn.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
    fullscreenBtn.innerHTML = icon(active ? 'minimize' : 'maximize', { size: 22 });
    if (_overlay) {
      if (active) _overlay.dataset.fullscreen = 'true';
      else delete _overlay.dataset.fullscreen;
      // Entering/leaving fullscreen flips whether chrome should auto-hide,
      // and resizes the video area out from under the PiP.
      _armIdleHide();
      _reclampPipCorner();
    }
  };
  // document/window listeners outlive the overlay element, which is now
  // rebuilt after any call that reached the end-of-call screen. Wire them
  // exactly once or every call stacks another copy. They read the module
  // ref rather than a captured element, so one set serves every overlay.
  if (!_globalCallListenersWired) {
    _globalCallListenersWired = true;
    document.addEventListener('fullscreenchange', onFsChange);
    document.addEventListener('webkitfullscreenchange', onFsChange);
  }

  shareBtn.addEventListener('click', async () => {
    if (!_activeSession) return;
    const session = _activeSession;
    if (typeof session.startScreenShare !== 'function') return;
    shareBtn.disabled = true;
    try {
      if (session.isScreenSharing) {
        await session.stopScreenShare();
      } else {
        await session.startScreenShare();
      }
    } catch (err) {
      // User-cancel of the browser picker is the expected no-op
      // path — silent rather than toasting an error.
      if (err?.message !== 'screen_share_cancelled') {
        const msg = err?.message === 'screen_share_requires_video'
          ? 'Add video first, then share your screen.'
          : err?.message === 'screen_share_failed'
            ? 'Screen share unavailable.'
            : err?.message === 'camera_restore_failed'
              ? 'Stopped sharing — camera could not be restored.'
              : err?.message || 'Screen share failed.';
        showToast(msg, 'error');
      }
    } finally {
      shareBtn.disabled = false;
      _syncShareUiForSession(session);
    }
  });
  outputBtn.addEventListener('click', () => {
    if (!_audioEl) return;
    _toggleOutputMenu(outputMenu);
  });
  // Show the output-picker button only when the browser supports
  // setSinkId; otherwise hide entirely (no half-broken affordance).
  if (_audioEl && typeof _audioEl.setSinkId === 'function') {
    outputBtn.hidden = false;
  }

  escalateBtn.addEventListener('click', async () => {
    if (!_activeSession) return;
    const session = _activeSession;
    const willAdd = !session.withVideo;
    if (willAdd && typeof session.addVideo !== 'function') return;
    if (!willAdd && typeof session.removeVideo !== 'function') return;
    escalateBtn.disabled = true;
    try {
      if (willAdd) {
        await session.addVideo();
      } else {
        await session.removeVideo();
      }
    } catch (err) {
      const reason = err?.message === 'camera_denied'
        ? 'camera access denied'
        : (err?.message || 'renegotiation failed');
      showToast(`Video: ${reason}`, 'error');
    } finally {
      escalateBtn.disabled = false;
      _syncVideoUiForSession(session);
    }
  });

  // Icon-only controls — labels are conveyed via title + aria-label
  // (tooltip + screen reader), not visual text under the icon.
  muteBtn.addEventListener('click', () => {
    if (!_activeSession) return;
    const next = muteBtn.getAttribute('aria-pressed') !== 'true';
    muteBtn.setAttribute('aria-pressed', next ? 'true' : 'false');
    muteBtn.setAttribute('title', next ? 'Unmute' : 'Mute');
    muteBtn.setAttribute('aria-label', next ? 'Unmute' : 'Mute');
    // Swap the icon to match the state (mic vs mic-off).
    muteBtn.innerHTML = icon(next ? 'mic-off' : 'mic', { size: 22 });
    _activeSession.setMicMuted(next);
  });

  videoBtn.addEventListener('click', () => {
    if (!_activeSession || !_activeSession.setVideoEnabled) return;
    // Derive the next state from the live track, not from the button's
    // own attribute — the attribute can be stale if anything re-synced
    // the chrome since the last click.
    const enabled = !_sessionVideoEnabled(_activeSession);
    _activeSession.setVideoEnabled(enabled);
    // Reflect immediately rather than waiting for frames to stop: the
    // placeholder should appear the instant the user asks for it.
    _syncVideoToggleButton(videoBtn, enabled);
  });

  // Rotation / window-resize: re-clamp the PiP into its saved corner and
  // re-evaluate whether chrome should auto-hide (a coarse-pointer tablet
  // can gain a pointer, and fullscreen changes the answer too). Layout is
  // otherwise pure CSS, so this is the only JS reflow hook the call needs.
  // Wired once for the same reason as the fullscreen listeners above —
  // the overlay is rebuilt per call, these are not.
  if (!_globalWindowListenersWired) {
    _globalWindowListenersWired = true;
    window.addEventListener('resize', () => {
      _reclampPipCorner();
      _armIdleHide();
    });
    window.addEventListener('orientationchange', () => {
      // Orientation fires before the viewport settles on some mobile
      // browsers — defer a frame so measurements are post-rotation.
      setTimeout(() => _reclampPipCorner(), 200);
    });
  }

  hangupBtn.addEventListener('click', async () => {
    if (!_activeSession) { _hideOverlay(); return; }
    hangupBtn.disabled = true;
    try {
      await _activeSession.hangup('local_hangup');
    } finally {
      hangupBtn.disabled = false;
    }
  });

  return el;
}

function _showOverlay(peerDid, state, { withVideo = false, localStream = null } = {}) {
  if (!_overlay) _ensureOverlay();
  _overlay.classList.remove('hidden');

  // Identity layer: pretty display name on the primary line, the
  // @instance subtitle only when it's an actual remote host (not
  // the local sentinel). DID is no longer rendered as a name.
  const nameEl = _overlay.querySelector('.connect-call-peer-name');
  if (nameEl) nameEl.textContent = _prettyPeerName(peerDid);
  const subEl = _overlay.querySelector('.connect-call-peer-subtitle');
  if (subEl) {
    const sub = _peerSubtitle(peerDid);
    subEl.textContent = sub;
    subEl.hidden = !sub;
  }
  const avEl = _overlay.querySelector('.connect-call-avatar');
  if (avEl) avEl.textContent = _initialFor(_prettyPeerName(peerDid) || peerDid);

  // Per-peer color identity — sets CSS variables the backdrop +
  // avatar + pulse ring all read so each contact has stable colors
  // across calls. Deterministic from the DID hash.
  const color = _peerColor(peerDid);
  _overlay.style.setProperty('--connect-call-peer-primary', color.primary);
  _overlay.style.setProperty('--connect-call-peer-secondary', color.secondary);
  _overlay.style.setProperty('--connect-call-peer-deep', color.deep);

  _setOverlayState(state);

  const videosWrap = _overlay.querySelector('.connect-call-videos');
  const videoBtn = _overlay.querySelector('.connect-call-video-toggle');
  if (withVideo) {
    videosWrap.hidden = false;
    videoBtn.hidden = false;
    videoBtn.setAttribute('aria-pressed', 'true');
    if (localStream) {
      const localVideo = _overlay.querySelector('.connect-call-local-video');
      _attachVideoStream(localVideo, localStream);
    }
  } else {
    videosWrap.hidden = true;
    videoBtn.hidden = true;
  }
}

function _hideOverlay() {
  if (!_overlay) return;
  _stopDurationTicker();
  _stopAudioPulse();
  _disarmIdleHide();
  // Exit fullscreen if we entered it — otherwise the user lands on a
  // black browser viewport after the call ends.
  const card = _overlay.querySelector('.connect-call-card');
  const inFs = (document.fullscreenElement === card)
    || (document.webkitFullscreenElement === card);
  if (inFs) {
    try {
      (document.exitFullscreen?.() || document.webkitExitFullscreen?.())
        ?.catch?.(() => {});
    } catch (_) { /* defensive */ }
  }
  _overlay.classList.add('hidden');
  if (_audioEl) {
    try { _audioEl.srcObject = null; } catch (_) {}
  }
  // The end-of-call screen replaces the card's ENTIRE innerHTML, so once
  // it has rendered the overlay holds none of the live-call DOM — no
  // video tiles, no controls, no headline. _ensureOverlay short-circuits
  // on an existing _overlay, so the next call reused this husk and
  // re-presented the stale "How did it sound? / Rejoin" screen instead of
  // connecting. Tear it down here so the next call builds a fresh one.
  if (card?.classList.contains('ended')) {
    try { _overlay.remove(); } catch (_) { /* already detached */ }
    _overlay = null;
    _audioEl = null;
    return;
  }
  const muteBtn = _overlay.querySelector('.connect-call-mute');
  if (muteBtn) {
    muteBtn.setAttribute('aria-pressed', 'false');
    const labelEl = muteBtn.querySelector('.connect-call-ctrl-label');
    if (labelEl) labelEl.textContent = 'Mute';
  }
  const remoteVideo = _overlay.querySelector('.connect-call-remote-video');
  const localVideo = _overlay.querySelector('.connect-call-local-video');
  if (remoteVideo) { try { remoteVideo.srcObject = null; } catch (_) {} }
  if (localVideo) { try { localVideo.srcObject = null; } catch (_) {} }
  // The overlay element is reused for every subsequent call, so per-call
  // video state has to be cleared here or the next call opens showing a
  // stale "camera off" card, a stale spotlight swap, or the previous
  // camera's frame shape.
  _setLocalCameraOff(false);
  _setPeerCameraOff(false);
  _setSpotlight('remote');
  // Aspect is element-scoped now, and the <video> elements survive a hide
  // when the call didn't reach the end-of-call screen — so clear it there,
  // or the next call's tiles open in the previous camera's shape.
  for (const el of _overlay.querySelectorAll(
    '.connect-call-local-video, .connect-call-remote-video, .connect-call-tile-off',
  )) {
    el.style.removeProperty('--tile-aspect');
    delete el.dataset.aspectWired;
  }
  const durEl = _overlay.querySelector('.connect-call-duration');
  if (durEl) { durEl.hidden = true; durEl.textContent = '00:00'; }
  // Reset quality pill so the next call doesn't inherit the prior
  // bucket on its first frame.
  _updateQualityPill({ bucket: 'measuring' });
  _setPeerMuted(false);
}

// ── Duration ticker ─────────────────────────────────────────────

let _durationTimer = null;
let _connectedSince = 0;

function _startDurationTicker() {
  if (_durationTimer) return;
  _connectedSince = Date.now();
  const tick = () => {
    if (!_overlay) return;
    const durEl = _overlay.querySelector('.connect-call-duration');
    if (!durEl) return;
    durEl.hidden = false;
    durEl.textContent = _formatHMS((Date.now() - _connectedSince) / 1000);
  };
  tick();
  _durationTimer = setInterval(tick, 1000);
}

function _stopDurationTicker() {
  if (_durationTimer) { clearInterval(_durationTimer); _durationTimer = null; }
  _connectedSince = 0;
}

// ── Idle-hide chrome + draggable PiP ──────────────────────────
//
// Per the FaceTime / Meet pattern: once we're in a connected call,
// the control bar fades after 3s of pointer inactivity and reasserts
// on any pointer move. Reduces visual clutter during the call,
// especially for video where the user wants to see the peer's face,
// not a row of buttons.
//
// The local PiP tile gets drag + corner-magnet behavior so the user
// can move it out of the way of important video content.

const IDLE_HIDE_MS = 3000;
let _idleTimer = null;

/**
 * Auto-hiding chrome is a touch / fullscreen idiom, not a desktop-windowed
 * one. With a mouse present in a normal window, every mature client keeps
 * the control bar up permanently — hiding it makes users hunt for the
 * hangup button. So auto-hide applies only when the pointer is coarse
 * (touch) or we're actually in fullscreen.
 */
function _shouldIdleHide() {
  if (!_overlay) return false;
  if (_overlay.dataset.fullscreen === 'true') return true;
  try {
    return window.matchMedia('(pointer: coarse)').matches;
  } catch (_) {
    return false;
  }
}

function _armIdleHide() {
  if (!_overlay) return;
  // Only auto-hide once CONNECTED — earlier states are short and the
  // user needs to see them to understand call status.
  if (_overlay.dataset.state !== CALL_STATES.CONNECTED) return;
  if (!_shouldIdleHide()) { _disarmIdleHide(); return; }
  _overlay.classList.remove('chrome-idle');
  if (_idleTimer) clearTimeout(_idleTimer);
  _idleTimer = setTimeout(() => {
    if (_overlay && _overlay.dataset.state === CALL_STATES.CONNECTED) {
      _overlay.classList.add('chrome-idle');
    }
  }, IDLE_HIDE_MS);
}

function _disarmIdleHide() {
  if (_idleTimer) { clearTimeout(_idleTimer); _idleTimer = null; }
  if (_overlay) _overlay.classList.remove('chrome-idle');
}

function _wireIdleHide() {
  if (!_overlay || _overlay.dataset.idleWired === '1') return;
  _overlay.dataset.idleWired = '1';
  const reset = () => _armIdleHide();
  _overlay.addEventListener('pointermove', reset);
  _overlay.addEventListener('pointerdown', reset);
  _overlay.addEventListener('keydown', reset);
}

// Spotlight: which tile is the hero (full-area) vs. the PiP corner.
// Default is remote-as-hero (the other person). Click the local tile
// while it's the PiP to spotlight yourself; click again to swap back.
// Useful most often paired with screen share — sender wants their own
// shared screen to be the hero so they can confirm what's transmitted.
function _wireSpotlightSwap() {
  if (!_overlay || _overlay.dataset.spotlightWired === '1') return;
  _overlay.dataset.spotlightWired = '1';
  const local = _overlay.querySelector('.connect-call-local-video');
  const remote = _overlay.querySelector('.connect-call-remote-video');
  if (!local || !remote) return;

  // Distinguish click from drag — the drag handler captures pointers
  // and adds .dragging during movement. Only swap when no drag occurred.
  let downPos = null;
  local.addEventListener('pointerdown', (ev) => {
    downPos = { x: ev.clientX, y: ev.clientY };
  });
  local.addEventListener('pointerup', (ev) => {
    if (!downPos) return;
    const dx = ev.clientX - downPos.x;
    const dy = ev.clientY - downPos.y;
    downPos = null;
    // Drag threshold mirrors the desktop click-vs-drag heuristic.
    if (Math.hypot(dx, dy) > 6) return;
    _setSpotlight(_overlay.dataset.spotlight === 'local' ? 'remote' : 'local');
  });

  // Clicking the remote (hero) tile while the local is spotlighted
  // brings the remote back to hero. While remote is already hero we
  // do nothing — clicking the hero shouldn't swap to PiP unexpectedly.
  remote.addEventListener('click', () => {
    if (_overlay.dataset.spotlight === 'local') _setSpotlight('remote');
  });
}

function _setSpotlight(which) {
  if (!_overlay) return;
  // Default remote-as-hero — represented by absence of the attribute
  // so the CSS selectors fall back to base rules. Local-as-hero is
  // explicit via data-spotlight="local".
  if (which === 'local') _overlay.dataset.spotlight = 'local';
  else delete _overlay.dataset.spotlight;
  // Roles just swapped — the outgoing hero must surrender the corner and
  // the incoming PiP must take it, or the stale inline offsets strand the
  // new hero as an edge-pinned strip.
  _syncTileRoles();
}

// PiP drag + corner-magnet. The local video tile is absolutely
// positioned inside the videos wrap; drag mutates `top` / `left`
// during the gesture and snaps to the nearest corner on release.
// The chosen corner is persisted to localStorage so the next call
// opens with the user's preferred PiP position (matches FaceTime /
// Meet expectation that PiP placement is stable across sessions).
const PIP_CORNER_KEY = 'augmentum:connect:pip-corner';

function _applyPipCorner(el, h, v) {
  // h: 'left' | 'right', v: 'top' | 'bottom'.
  //
  // A flat 12px bottom gutter used to be written here, which silently
  // overrode the clearance the stylesheet reserves above the controls
  // pill — so simply dragging the PiP parked it underneath the controls
  // and the bottom scrim. Mirror the CSS values instead: safe-area aware,
  // and clear of the chrome at the bottom edge.
  const CHROME_CLEARANCE = 'calc(max(env(safe-area-inset-bottom, 0px), 24px) + 100px)';
  const TOP_GUTTER = 'max(env(safe-area-inset-top, 0px), 12px)';
  el.style.left = h === 'left' ? '12px' : 'auto';
  el.style.right = h === 'right' ? '12px' : 'auto';
  el.style.top = v === 'top' ? TOP_GUTTER : 'auto';
  el.style.bottom = v === 'bottom' ? CHROME_CLEARANCE : 'auto';
}

/**
 * Drop every inline geometry offset from a tile, handing layout back to
 * the stylesheet.
 *
 * Inline styles beat ALL stylesheet rules, so a tile still carrying PiP
 * corner offsets cannot be laid out by the role rules that say
 * `inset: 0; width: 100%; height: 100%`. It stays pinned to one edge with
 * no opposing anchor and renders as a narrow full-height strip. Any tile
 * that is not currently the PiP must be stripped clean.
 */
function _clearTileGeometry(el) {
  if (!el) return;
  for (const prop of ['left', 'right', 'top', 'bottom']) {
    el.style.removeProperty(prop);
  }
}

/**
 * Single owner for which tile is the corner PiP.
 *
 * Exactly one tile may hold inline corner offsets, and only while it is
 * genuinely in the PiP role; every other tile's geometry belongs to CSS.
 * The corner used to be written to the local tile on connect and never
 * revoked, so entering the lobby phase or spotlighting the local tile
 * left the hero fighting its own stale offsets.
 */
function _syncTileRoles() {
  if (!_overlay) return;
  const local = _overlay.querySelector('.connect-call-local-video');
  const remote = _overlay.querySelector('.connect-call-remote-video');
  const localOff = _overlay.querySelector('.connect-call-local-off');
  const remoteOff = _overlay.querySelector('.connect-call-remote-off');

  // ── Resolve roles ONCE ──────────────────────────────────────────
  // Lobby is a single full-bleed tile (your own camera while dialling)
  // and has no PiP at all. Live is hero + PiP, with spotlight deciding
  // which way round. This is the only place that decision is made.
  const lobby = _overlay.dataset.phase === 'lobby';
  let localRole;
  let remoteRole;
  if (lobby) {
    localRole = 'hero';
    remoteRole = 'hidden';
  } else if (_overlay.dataset.spotlight === 'local') {
    localRole = 'hero';
    remoteRole = 'pip';
  } else {
    localRole = 'pip';
    remoteRole = 'hero';
  }

  // Stamp the role onto the video AND its camera-off placeholder, so the
  // card always inherits the same geometry as the tile it stands in for.
  // Previously each of hero/PiP/lobby geometry was re-declared per tile
  // per phase per spotlight state — six near-duplicate blocks that drifted
  // apart. Now CSS defines `hero` and `pip` once and both tiles use them.
  const localCap = _overlay.querySelector('.connect-call-local-caption');
  const remoteCap = _overlay.querySelector('.connect-call-remote-caption');
  for (const el of [local, localOff, localCap]) {
    if (el) el.dataset.role = localRole;
  }
  for (const el of [remote, remoteOff, remoteCap]) {
    if (el) el.dataset.role = remoteRole;
  }

  // ── Geometry ownership ──────────────────────────────────────────
  // Only the PiP may carry inline offsets, and only while it holds that
  // role. Inline styles outrank every stylesheet rule, so a tile that
  // keeps them after being promoted to hero cannot lay itself out.
  const pip = localRole === 'pip' ? local : (remoteRole === 'pip' ? remote : null);
  for (const el of [local, remote]) {
    if (el && el !== pip) _clearTileGeometry(el);
  }
  if (pip) _applySavedPipCorner(pip);
}

function _applySavedPipCorner(target) {
  if (!_overlay) return;
  const el = target || _overlay.querySelector('.connect-call-local-video');
  if (!el) return;
  let corner;
  try { corner = localStorage.getItem(PIP_CORNER_KEY); } catch (_) { return; }
  if (!corner || !/^[tb][lr]$/.test(corner)) return;
  const h = corner[1] === 'l' ? 'left' : 'right';
  const v = corner[0] === 't' ? 'top' : 'bottom';
  _applyPipCorner(el, h, v);
}

/**
 * Re-apply the saved corner after anything that changes the video area's
 * dimensions — rotation, window resize, fullscreen transitions. The drag
 * handler writes absolute pixel offsets during a gesture; without this a
 * corner chosen in landscape can land off-screen or under the chrome once
 * the device is rotated to portrait.
 */
function _reclampPipCorner() {
  if (!_overlay || _overlay.classList.contains('hidden')) return;
  _syncTileRoles();
}

function _wireLocalVideoDrag() {
  if (!_overlay) return;
  const local = _overlay.querySelector('.connect-call-local-video');
  if (!local || local.dataset.dragWired === '1') return;
  local.dataset.dragWired = '1';

  let pointerId = null;
  let offsetX = 0;
  let offsetY = 0;
  let parentRect = null;
  let tileRect = null;

  local.addEventListener('pointerdown', (ev) => {
    if (pointerId !== null) return;
    // Only the corner PiP is draggable. While the local tile is the hero
    // (lobby, or spotlighted) a drag would write inline offsets onto a
    // full-bleed tile and pin it to an edge as a strip.
    if (_overlay?.dataset.phase === 'lobby'
      || _overlay?.dataset.spotlight === 'local') return;
    const parent = local.parentElement;
    if (!parent) return;
    parentRect = parent.getBoundingClientRect();
    tileRect = local.getBoundingClientRect();
    offsetX = ev.clientX - tileRect.left;
    offsetY = ev.clientY - tileRect.top;
    pointerId = ev.pointerId;
    local.setPointerCapture(pointerId);
    local.classList.add('dragging');
  });

  local.addEventListener('pointermove', (ev) => {
    if (pointerId !== ev.pointerId || !parentRect || !tileRect) return;
    const x = ev.clientX - parentRect.left - offsetX;
    const y = ev.clientY - parentRect.top - offsetY;
    const maxX = parentRect.width - tileRect.width;
    const maxY = parentRect.height - tileRect.height;
    const clampedX = Math.min(Math.max(0, x), maxX);
    const clampedY = Math.min(Math.max(0, y), maxY);
    local.style.left = `${clampedX}px`;
    local.style.top = `${clampedY}px`;
    local.style.right = 'auto';
    local.style.bottom = 'auto';
  });

  const finish = (ev) => {
    if (pointerId !== ev.pointerId || !parentRect || !tileRect) return;
    try { local.releasePointerCapture(pointerId); } catch (_) {}
    pointerId = null;
    local.classList.remove('dragging');
    // Snap to the nearest corner: compute centre, decide L/R + T/B.
    const localRect = local.getBoundingClientRect();
    const cx = localRect.left + localRect.width / 2 - parentRect.left;
    const cy = localRect.top + localRect.height / 2 - parentRect.top;
    const goRight = cx > parentRect.width / 2;
    const goBottom = cy > parentRect.height / 2;
    local.style.transition = 'left 220ms var(--ease-out), top 220ms var(--ease-out), right 220ms var(--ease-out), bottom 220ms var(--ease-out)';
    _applyPipCorner(local, goRight ? 'right' : 'left', goBottom ? 'bottom' : 'top');
    setTimeout(() => { local.style.transition = ''; }, 240);
    // Persist so the next call opens with the same corner. Keep four
    // discrete corners (vs. arbitrary x/y) — re-applying an exact pixel
    // position is fragile across window-resize / different aspect
    // ratios between calls.
    const corner = `${goBottom ? 'b' : 't'}${goRight ? 'r' : 'l'}`;
    try { localStorage.setItem(PIP_CORNER_KEY, corner); } catch (_) {}
  };
  local.addEventListener('pointerup', finish);
  local.addEventListener('pointercancel', finish);
}

// ── Remote audio level → avatar pulse ring ────────────────────
//
// Per the Discord / Meet pattern: audio-only calls especially need a
// presence cue beyond a static avatar. We hook an AudioContext +
// AnalyserNode to the remote MediaStream and drive a CSS custom
// property (--connect-call-pulse-level, 0..1) on each frame. The
// pulse ring's scale + opacity track that variable.
//
// All resources get cleaned up in _stopAudioPulse on call end.

let _audioCtx = null;
let _audioAnalyser = null;
let _audioSource = null;
let _audioRafId = null;
let _audioBuffer = null;

function _startAudioPulse(stream) {
  _stopAudioPulse();
  if (!stream || !_overlay) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    _audioCtx = new Ctx();
    _audioSource = _audioCtx.createMediaStreamSource(stream);
    _audioAnalyser = _audioCtx.createAnalyser();
    _audioAnalyser.fftSize = 256;
    _audioAnalyser.smoothingTimeConstant = 0.7;
    _audioSource.connect(_audioAnalyser);
    // We don't connect to _audioCtx.destination — the audio is already
    // playing through the <audio> element. This analyser is read-only.
    _audioBuffer = new Uint8Array(_audioAnalyser.frequencyBinCount);
  } catch (err) {
    console.warn('connect: audio analyser unavailable', err);
    return;
  }

  // Waveform canvas (drawn during connected voice calls only).
  const waveCanvas = _overlay.querySelector('.connect-call-waveform');
  const waveCtx = waveCanvas?.getContext('2d') || null;

  const tick = () => {
    if (!_audioAnalyser || !_overlay) {
      _audioRafId = null;
      return;
    }
    _audioAnalyser.getByteFrequencyData(_audioBuffer);
    // Average bin energy in the speech band (~85Hz-3000Hz). With
    // fftSize=256 and 48kHz sample rate, that's roughly bins 0-16.
    let sum = 0;
    const bins = Math.min(16, _audioBuffer.length);
    for (let i = 0; i < bins; i++) sum += _audioBuffer[i];
    const avg = sum / bins / 255;
    // Light gamma curve so quiet speech still produces a visible ring.
    const level = Math.min(1, Math.pow(avg, 0.6) * 1.6);
    _overlay.style.setProperty('--connect-call-pulse-level', level.toFixed(3));

    // Draw the waveform — only when the canvas is visible (CSS
    // toggles it via overlay's data-phase + data-state).
    if (waveCtx && waveCanvas && !waveCanvas.hidden) {
      _drawWaveform(waveCtx, waveCanvas, _audioBuffer);
    }

    _audioRafId = requestAnimationFrame(tick);
  };
  _audioRafId = requestAnimationFrame(tick);
}

/**
 * Draw a centered audio bars visualisation onto the overlay's
 * waveform canvas. 28 bars across the canvas width, each height
 * driven by a band of the FFT buffer. Bars use the per-peer color
 * with a soft top-to-bottom fade so they read as speech, not equaliser.
 */
function _drawWaveform(ctx, canvas, buffer) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const bars = 28;
  const gap = 3;
  const barW = Math.max(2, (w - gap * (bars - 1)) / bars);
  const binsPerBar = Math.max(1, Math.floor(buffer.length / bars));

  // Sample the peer primary color from CSS so the waveform inherits
  // the per-peer identity already on the avatar + backdrop. Comes
  // back as hsl(N, S%, L%) from _peerColor; canvas accepts both
  // hsl() and hsla() so we just append an alpha channel for the
  // softer edge stops.
  const peerColor = getComputedStyle(canvas).getPropertyValue('--connect-call-peer-primary').trim()
    || 'hsl(214, 62%, 56%)';
  const hslMatch = peerColor.match(/hsl\(\s*(\d+)[,\s]+(\d+)%[,\s]+(\d+)%\s*\)/);
  const toHsla = (alpha) => hslMatch
    ? `hsla(${hslMatch[1]}, ${hslMatch[2]}%, ${hslMatch[3]}%, ${alpha})`
    : peerColor;

  for (let i = 0; i < bars; i++) {
    // Average a band of FFT bins for this bar.
    let sum = 0;
    const start = i * binsPerBar;
    for (let j = 0; j < binsPerBar; j++) sum += buffer[start + j] || 0;
    const v = sum / binsPerBar / 255;
    // Resting baseline so the bars never disappear entirely — keeps
    // the strip looking alive even at silence.
    const norm = 0.08 + Math.pow(v, 0.65) * 0.92;
    const barH = Math.max(3, norm * h);
    const x = i * (barW + gap);
    const y = (h - barH) / 2;

    // Vertical gradient: faded tips, bright midline. Reads as
    // speech, not an equaliser. Hsla lets us alpha the tips.
    const grad = ctx.createLinearGradient(0, y, 0, y + barH);
    grad.addColorStop(0,   toHsla(0.55));
    grad.addColorStop(0.5, toHsla(1));
    grad.addColorStop(1,   toHsla(0.55));
    ctx.fillStyle = grad;
    ctx.beginPath();
    const r = Math.min(barW / 2, 3);
    ctx.roundRect(x, y, barW, barH, r);
    ctx.fill();
  }
}

function _stopAudioPulse() {
  if (_audioRafId) { cancelAnimationFrame(_audioRafId); _audioRafId = null; }
  if (_audioSource) {
    try { _audioSource.disconnect(); } catch (_) {}
    _audioSource = null;
  }
  _audioAnalyser = null;
  if (_audioCtx) {
    try { _audioCtx.close(); } catch (_) {}
    _audioCtx = null;
  }
  _audioBuffer = null;
  if (_overlay) _overlay.style.removeProperty('--connect-call-pulse-level');
}

function _formatHMS(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(r).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}


// Last-picked output device (localStorage). When the call starts we
// try to honour it on the audio element via setSinkId. The browser
// returns an error when the device is no longer attached — we just
// fall back to default in that case.
const OUTPUT_PREF_KEY = 'augmentum:connect:output-device-id';

async function _toggleOutputMenu(menu) {
  if (!menu || !_audioEl) return;
  if (!menu.hidden) { menu.hidden = true; return; }
  let devices;
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    showToast('Could not list audio devices', 'error');
    return;
  }
  const outputs = (devices || [])
    .filter((d) => d.kind === 'audiooutput')
    .filter((d) => d.deviceId);  // Some browsers list a "" entry
  if (!outputs.length) {
    showToast('No audio output devices available', 'warning');
    return;
  }
  const currentSink = _audioEl.sinkId || 'default';
  menu.innerHTML = outputs.map((d) => {
    const isActive = d.deviceId === currentSink
      || (currentSink === 'default' && d.deviceId === 'default');
    const label = escapeHtml(d.label || `Audio output ${d.deviceId.slice(0, 6)}`);
    return `
      <button class="connect-call-output-item${isActive ? ' active' : ''}"
              type="button" role="menuitem"
              data-device-id="${escapeHtml(d.deviceId)}">
        ${label}${isActive ? ' ✓' : ''}
      </button>`;
  }).join('');
  menu.hidden = false;
  // One-shot delegated handler — re-bound each open so we don't
  // accumulate listeners over many opens.
  menu.onclick = async (evt) => {
    const btn = evt.target.closest('.connect-call-output-item');
    if (!btn) return;
    const id = btn.dataset.deviceId;
    if (!id) return;
    try {
      await _audioEl.setSinkId(id);
      try { localStorage.setItem(OUTPUT_PREF_KEY, id); } catch (_) {}
      showToast('Output switched', 'success');
    } catch (err) {
      showToast(`Could not switch output: ${err?.message || err}`, 'error');
    }
    menu.hidden = true;
  };
}

async function _toggleCameraMenu(menu) {
  if (!menu || !_activeSession) return;
  if (!menu.hidden) { menu.hidden = true; return; }

  let devices;
  try {
    devices = await listVideoDevices({ probeForLabels: false });
  } catch (err) {
    showToast('Could not list cameras', 'error');
    return;
  }
  if (!devices.length) {
    showToast('No cameras available', 'warning');
    return;
  }
  if (devices.length === 1) {
    // Don't open a single-item menu — it just adds clicks. Toast the
    // status instead so the user knows there's nothing to switch to.
    showToast('Only one camera available', 'info');
    return;
  }

  const current = String(_activeSession.videoDeviceId || getPreferredVideoDeviceId() || '');
  menu.innerHTML = devices.map((d, i) => {
    const isActive = d.deviceId === current;
    const label = escapeHtml(d.label || `Camera ${i + 1}`);
    return `
      <button class="connect-call-camera-item${isActive ? ' active' : ''}"
              type="button" role="menuitem"
              data-device-id="${escapeHtml(d.deviceId)}">
        ${label}${isActive ? ' ✓' : ''}
      </button>`;
  }).join('');
  menu.hidden = false;

  menu.onclick = async (evt) => {
    const btn = evt.target.closest('.connect-call-camera-item');
    if (!btn) return;
    const id = btn.dataset.deviceId;
    if (!id || !_activeSession) { menu.hidden = true; return; }
    menu.hidden = true;
    try {
      const result = await _activeSession.switchVideoDevice(id);
      // Persist so the next call defaults to this camera too.
      setPreferredVideoDeviceId(id);
      if (result?.swapped) {
        showToast('Camera switched', 'success');
        // Local-video tile picks up the new track via the
        // local-stream-updated emit — same MediaStream object, new
        // tracks. _syncVideoUiForSession handles the refresh.
      }
    } catch (err) {
      const reason = err?.message === 'camera_switch_failed'
        ? 'camera access denied'
        : (err?.message || 'switch failed');
      showToast(`Camera: ${reason}`, 'error');
    }
  };
}

async function _applyPreferredOutput() {
  if (!_audioEl || typeof _audioEl.setSinkId !== 'function') return;
  let preferred;
  try { preferred = localStorage.getItem(OUTPUT_PREF_KEY); } catch (_) {}
  if (!preferred) return;
  try {
    await _audioEl.setSinkId(preferred);
  } catch (_) {
    // Device no longer present — silently fall back to default.
    try { localStorage.removeItem(OUTPUT_PREF_KEY); } catch (_) {}
  }
}

function _setPeerMuted(muted) {
  if (!_overlay) return;
  const badge = _overlay.querySelector('.connect-call-peer-muted');
  if (!badge) return;
  badge.hidden = !muted;
}

function _updateQualityPill({ bucket = 'measuring', rttMs = null, lossPct = null } = {}) {
  if (!_overlay) return;
  const pill = _overlay.querySelector('.connect-call-quality');
  if (!pill) return;
  pill.dataset.bucket = bucket;
  const label = QUALITY_LABEL[bucket] ?? '';
  const labelEl = pill.querySelector('.connect-call-quality-label');
  if (labelEl) labelEl.textContent = label;
  // Pill stays hidden on excellent/good — only weak/poor surface
  // explicitly. The dot alone would be ambiguous; pairing it with a
  // short label avoids any "is that a notification?" confusion.
  pill.hidden = !label;
  // Detailed RTT/loss live on the title attribute so power users
  // hovering can see exactly why the bucket changed.
  const parts = [];
  if (rttMs != null) parts.push(`${rttMs} ms RTT`);
  if (lossPct != null) parts.push(`${lossPct}% loss`);
  pill.title = parts.length ? parts.join(' · ') : 'Connection quality';
}

function _setOverlayState(state) {
  if (!_overlay) return;
  // Empty-string label (CONNECTED) hides the state line so the
  // duration ticker isn't fighting it for vertical space.
  const label = STATE_LABEL[state] ?? state;
  const stateEl = _overlay.querySelector('.connect-call-state');
  if (stateEl) {
    stateEl.textContent = label || '';
    stateEl.hidden = !label;
  }
  _overlay.dataset.state = state;

  // Track lobby vs live as a separate attribute so CSS can swap
  // layouts without having to handle every individual state value.
  // Lobby = pre-CONNECTED (we're dialling / waiting / setting up
  // media); Live = post-CONNECTED (call is in progress).
  const lobbyStates = new Set([
    CALL_STATES.IDLE,
    CALL_STATES.CONNECTING,
    CALL_STATES.RINGING,
    CALL_STATES.NEGOTIATING,
    'pending',
    'accepting',
  ]);
  _overlay.dataset.phase = lobbyStates.has(state) ? 'lobby' : 'live';
  // Phase decides whether a PiP exists at all (lobby is single full-bleed
  // tile), so tile roles have to be re-resolved on every transition.
  _syncTileRoles();

  // Hangup button reads "Cancel" during lobby, "End call" in live —
  // small wording swap but it signals the user can still bail out
  // without "ending" something that hasn't started.
  const hangupBtn = _overlay.querySelector('.connect-call-hangup');
  if (hangupBtn) {
    const isLobby = _overlay.dataset.phase === 'lobby';
    hangupBtn.setAttribute('title', isLobby ? 'Cancel call' : 'End call');
    hangupBtn.setAttribute('aria-label', isLobby ? 'Cancel call' : 'End call');
  }

  // Surface the waveform canvas only when we're connected on a voice
  // call. The CSS rule that controls visibility uses :has() to gate
  // on video presence, but we still need to clear the [hidden]
  // attribute (set on the element at boot to keep it out of layout).
  const wave = _overlay.querySelector('.connect-call-waveform');
  if (wave) {
    const onVoiceCall = state === CALL_STATES.CONNECTED;
    wave.hidden = !onVoiceCall;
  }
}

// ── End-of-call screen ─────────────────────────────────────────
//
// Replaces the previous auto-dismiss-then-toast flow with a proper
// take-over inside the same overlay. Shows the peer + duration +
// modality summary, an inline rating row, and Rejoin / Return
// actions. The user dismisses explicitly — no timer.

import('./messages.js').catch(() => {});  // warm the chunk

async function _renderEndOfCall(session, { callId, peerDid, modalities, durationSeconds, reason }) {
  if (!_overlay) return;
  const card = _overlay.querySelector('.connect-call-card');
  if (!card) return;

  // Clear out the live-call children except the avatar wrap (kept as
  // visual anchor) and audio element (already silent at this point).
  // We replace the whole card contents with the end-of-call layout.
  const initial = _initialFor(peerDid || '?');
  const wasVideo = String(modalities || '').includes('video');
  const subtitle = wasVideo ? 'Video call' : 'Voice call';
  const durLabel = durationSeconds > 0 ? _formatHMS(durationSeconds) : '00:00';
  const reasonLabel = _humaniseEndReason(reason);

  card.classList.add('ended');
  card.innerHTML = `
    <div class="connect-call-end-headline">
      <div class="connect-call-avatar-wrap connect-call-avatar-wrap--end" aria-hidden="true">
        <div class="connect-call-avatar">${escapeHtml(initial)}</div>
      </div>
      <div class="connect-call-end-name">${escapeHtml(_prettyPeerName(peerDid) || 'Unknown')}</div>
      <div class="connect-call-end-meta">${escapeHtml(subtitle)} · ${escapeHtml(durLabel)}</div>
      ${reasonLabel ? `<div class="connect-call-end-reason">${escapeHtml(reasonLabel)}</div>` : ''}
    </div>
    <div class="connect-call-end-rating">
      <div class="connect-call-end-rating-title">How did it sound?</div>
      <div class="connect-call-end-rating-row">
        <button class="connect-call-end-rate-btn" type="button" data-rating="1" title="Good">
          <span class="connect-call-end-rate-glyph"></span>
          <span>Good</span>
        </button>
        <button class="connect-call-end-rate-btn" type="button" data-rating="0" title="OK">
          <span class="connect-call-end-rate-glyph"></span>
          <span>OK</span>
        </button>
        <button class="connect-call-end-rate-btn" type="button" data-rating="-1" title="Bad">
          <span class="connect-call-end-rate-glyph"></span>
          <span>Bad</span>
        </button>
      </div>
      <div class="connect-call-end-rating-saved" hidden>Thanks — saved.</div>
    </div>
    <div class="connect-call-end-actions">
      <button class="connect-call-end-action primary" type="button" data-action="rejoin">
        <span class="connect-call-end-action-glyph"></span>
        <span>Rejoin</span>
      </button>
      <button class="connect-call-end-action" type="button" data-action="dismiss">Return</button>
    </div>
  `;

  // Populate icons after innerHTML so JS-imported icon() composes
  // cleanly without nested-template-literal issues.
  card.querySelectorAll('.connect-call-end-rate-btn[data-rating="1"] .connect-call-end-rate-glyph')[0].innerHTML = icon('thumbs-up', { size: 18 });
  card.querySelectorAll('.connect-call-end-rate-btn[data-rating="0"] .connect-call-end-rate-glyph')[0].innerHTML = icon('minus', { size: 18 });
  card.querySelectorAll('.connect-call-end-rate-btn[data-rating="-1"] .connect-call-end-rate-glyph')[0].innerHTML = icon('thumbs-down', { size: 18 });
  card.querySelectorAll('.connect-call-end-action.primary .connect-call-end-action-glyph')[0].innerHTML = icon('phone', { size: 14 });

  // Wire rating buttons.
  const savedEl = card.querySelector('.connect-call-end-rating-saved');
  const rateBtns = card.querySelectorAll('.connect-call-end-rate-btn');
  for (const btn of rateBtns) {
    btn.addEventListener('click', async () => {
      const rating = parseInt(btn.dataset.rating, 10);
      for (const b of rateBtns) b.setAttribute('aria-pressed', 'false');
      btn.setAttribute('aria-pressed', 'true');
      try {
        const { rateCall } = await import('./messages.js');
        await rateCall(callId, rating, '');
        if (savedEl) { savedEl.hidden = false; }
      } catch (err) {
        if (savedEl) { savedEl.hidden = false; savedEl.textContent = 'Could not save — try again.'; }
      }
    });
  }

  // Wire actions.
  card.querySelector('[data-action="rejoin"]')?.addEventListener('click', async () => {
    if (_activeSession === session) _activeSession = null;
    _hideOverlay();
    try {
      await startCall(peerDid, { withVideo: wasVideo });
    } catch (err) {
      showToast(`Rejoin failed: ${err?.message || 'unknown'}`, 'error');
    }
  });
  card.querySelector('[data-action="dismiss"]')?.addEventListener('click', () => {
    if (_activeSession === session) _activeSession = null;
    _hideOverlay();
  });
}

// Single source of truth for turning a call failure / end reason into a
// calm, human sentence. Both the error toast (pre-connect failures) and the
// end-of-call summary (post-connect endings) delegate here so the wording
// never drifts and no raw reason code (`answer_failed`) or browser error
// ("InvalidStateError") ever reaches the user. Every reason emitted by
// dialer.js / incoming.js `_fail()` is covered; the fallback stays generic
// rather than leaking err.message.
const CALL_REASON_TEXT = {
  local_hangup: '',
  remote_hangup: 'They ended the call.',
  declined: 'They declined the call.',
  no_answer: 'No answer.',
  missed: 'Missed call.',
  timeout: 'The call timed out.',
  mic_denied: "Microphone access is blocked. Allow it in your browser's site settings, then try again.",
  camera_denied: "Camera access is blocked. Allow it in your browser's site settings, then try again.",
  signaling_unavailable: "Couldn't reach the server. Check your connection and try again.",
  signaling_lost: 'The connection dropped.',
  ice_failed: "Couldn't connect — one of you may be on a restricted network.",
  reconnect_timeout: "Lost the connection and couldn't reconnect.",
  connect_timeout: 'Took too long to connect. Try again.',
  negotiate_timeout: 'Took too long to connect. Try again.',
  accept_failed: "Couldn't connect the call. Try again.",
  answer_failed: "Couldn't connect the call. Try again.",
  offer_failed: "Couldn't connect the call. Try again.",
  candidates_failed: 'Hit a network snag connecting. Try again.',
  negotiate_failed: "Couldn't update the call. Try again.",
};

function _friendlyCallReason(reason, err) {
  // invite_failed carries a server reason code in err.message — map the
  // common ones to a specific line, else a soft "couldn't reach them".
  if (reason === 'invite_failed') {
    const code = String(err?.message || '').toLowerCase();
    if (code.includes('offline') || code.includes('unreachable')) return 'They appear to be offline.';
    if (code.includes('busy') || code.includes('in_call')) return "They're on another call.";
    if (code.includes('declin') || code.includes('reject')) return 'They declined the call.';
    return "Couldn't reach them — they may be offline.";
  }
  if (reason in CALL_REASON_TEXT) return CALL_REASON_TEXT[reason];
  return 'Something went wrong with the call. Try again.';
}

function _humaniseEndReason(reason) {
  return _friendlyCallReason(reason);
}

/**
 * Refresh the overlay's video-related affordances against the current
 * session shape. Called whenever modalities or state may have shifted
 * (state-change, negotiated, local-stream-updated, remote-stream).
 *
 *   - Video tile visibility tracks session.withVideo
 *   - "Camera off / on" toggles only when video is attached
 *   - "Add video / Drop video" escalate button is visible once the
 *     call is connected and either side can support negotiation
 */
function _syncVideoUiForSession(session) {
  if (!_overlay || !session) return;
  const videosWrap = _overlay.querySelector('.connect-call-videos');
  const videoBtn = _overlay.querySelector('.connect-call-video-toggle');
  const escalateBtn = _overlay.querySelector('.connect-call-video-escalate');
  const cameraBtn = _overlay.querySelector('.connect-call-camera-switch');
  const localVideo = _overlay.querySelector('.connect-call-local-video');
  const remoteVideo = _overlay.querySelector('.connect-call-remote-video');
  const withVideo = !!session.withVideo;

  if (videosWrap) videosWrap.hidden = !withVideo;
  if (videoBtn) {
    videoBtn.hidden = !withVideo;
    // Reflect the ACTUAL track state, never a hardcoded "on". This sync
    // runs on state-change / negotiated / local-stream-updated / remote
    // stream arrival, so forcing 'true' here silently desynced the
    // button from a camera the user had turned off — the next click then
    // computed enabled=false and re-disabled an already-disabled track,
    // making the toggle feel inverted and stuck.
    if (withVideo) _syncVideoToggleButton(videoBtn, _sessionVideoEnabled(session));
  }

  if (escalateBtn) {
    const canNegotiate = (typeof session.addVideo === 'function')
      && (typeof session.removeVideo === 'function');
    const connected = session.state === CALL_STATES.CONNECTED
      || session.state === CALL_STATES.NEGOTIATING;
    escalateBtn.hidden = !(canNegotiate && connected);
    // Label via title/aria-label only — assigning textContent here wiped
    // the SVG icon, leaving a bare text label in a row of icon buttons.
    _setCtrlLabel(escalateBtn, withVideo ? 'Drop video' : 'Add video',
      withVideo ? 'video-off' : 'video-plus');
  }

  if (cameraBtn) {
    // Switch button only matters when video is attached AND the
    // session can hot-swap. Camera count check happens lazily on
    // click — enumerateDevices is async and we don't want this
    // sync-path doing IO. Single-camera systems just show an empty
    // menu, which is fine.
    const canSwitch = typeof session.switchVideoDevice === 'function';
    cameraBtn.hidden = !(withVideo && canSwitch);
  }

  _syncShareUiForSession(session);

  // Tile captions — only meaningful when video is on. Reads from the
  // session's peerDid + the same prettyPeerName helper the headline
  // uses, so the on-tile chip matches the name in the headline.
  const remoteCap = _overlay.querySelector('.connect-call-remote-caption');
  const localCap = _overlay.querySelector('.connect-call-local-caption');
  if (remoteCap) {
    remoteCap.hidden = !withVideo;
    const nameEl = remoteCap.querySelector('.connect-call-tile-name');
    if (nameEl) nameEl.textContent = _prettyPeerName(session.peerDid) || 'Peer';
  }
  if (localCap) {
    localCap.hidden = !withVideo;
  }

  // No `!srcObject` guard: _attachVideoStream is idempotent, and the guard
  // actively harmed us — whenever another path had already set srcObject,
  // this branch skipped and took the tile's only reveal with it. It also
  // blocked re-attach after a renegotiation swapped the stream object.
  if (withVideo && session.localStream && localVideo) {
    _attachVideoStream(localVideo, session.localStream);
  }
  if (withVideo && session.remoteStream && remoteVideo) {
    _attachVideoStream(remoteVideo, session.remoteStream);
  }
  // Track each tile's real frame shape (rotate + camera-swap aware).
  if (withVideo) {
    _wireTileAspect(localVideo, _overlay.querySelector('.connect-call-local-off'));
    _wireTileAspect(remoteVideo, _overlay.querySelector('.connect-call-remote-off'));
    _syncTileOffIdentity(session);
    _syncTileRoles();
  }
  // If video was just dropped, clear the tiles to release the camera.
  if (!withVideo && localVideo && localVideo.srcObject) {
    try { localVideo.srcObject = null; } catch (_) {}
    localVideo.classList.remove('is-live');
  }
  if (!withVideo && remoteVideo && remoteVideo.srcObject) {
    try { remoteVideo.srcObject = null; } catch (_) {}
    remoteVideo.classList.remove('is-live');
  }
}

/**
 * Reveal a video tile only once it actually has frames, so the FaceTime-
 * style fullscreen doesn't flash an empty black tile during the ~hundreds of
 * ms between attaching srcObject and the first decoded frame. The CSS fades
 * `.is-live` in from opacity 0; a safety timeout guarantees the tile never
 * stays invisible if neither media event fires (some browsers/codecs).
 */
function _fadeInVideoTile(videoEl) {
  if (!videoEl) return;
  const reveal = () => videoEl.classList.add('is-live');
  // Already decoding — reveal immediately. Re-running the hide-then-wait
  // cycle on a live element would blank it for 1.5s, because
  // loadedmetadata/playing have already fired and will not fire again;
  // only the fallback timer would bring it back.
  if (videoEl.readyState >= 1) { reveal(); return; }
  videoEl.classList.remove('is-live');
  videoEl.addEventListener('loadedmetadata', reveal, { once: true });
  videoEl.addEventListener('playing', reveal, { once: true });
  setTimeout(reveal, 1500);
}

/**
 * The ONLY way a stream should reach a video tile.
 *
 * `.connect-call-remote-video` is `opacity: 0` until `.is-live` is added,
 * so attaching a stream without revealing the tile yields a tile that is
 * playing, decoding frames, and totally invisible. That is exactly what
 * happened: the 'remote-stream' handler set `srcObject` directly while
 * the reveal lived in the other attach path, which then skipped itself
 * because it is guarded on `!srcObject`. Two attach sites, one reveal —
 * so the peer's video never appeared. Route both through here.
 */
function _attachVideoStream(videoEl, stream) {
  if (!videoEl) return;
  if (videoEl.srcObject !== stream) videoEl.srcObject = stream;
  videoEl.play?.().catch(() => {});
  _fadeInVideoTile(videoEl);
}

/**
 * Set an icon-only control's label + glyph without destroying the other.
 * Every button in the controls pill conveys its label through
 * title + aria-label (tooltip + screen reader) and its state through the
 * SVG glyph. Assigning `.textContent` to one of these — which the video
 * and escalate buttons both used to do — replaces the SVG child with a
 * text node, degrading the button to bare text in a row of icons.
 */
function _setCtrlLabel(btn, label, iconName) {
  if (!btn) return;
  btn.setAttribute('title', label);
  btn.setAttribute('aria-label', label);
  if (iconName) btn.innerHTML = icon(iconName, { size: 22 });
}

/** True when the session is currently sending camera frames. */
function _sessionVideoEnabled(session) {
  if (typeof session?.isVideoEnabled === 'boolean') return session.isVideoEnabled;
  const tracks = session?.localStream?.getVideoTracks?.() || [];
  return tracks.length > 0 && tracks.some((t) => t.enabled);
}

/** Drive the camera toggle's pressed-state, glyph and label from one source. */
function _syncVideoToggleButton(btn, enabled) {
  if (!btn) return;
  btn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  _setCtrlLabel(btn, enabled ? 'Turn camera off' : 'Turn camera on',
    enabled ? 'video' : 'video-off');
  _setLocalCameraOff(!enabled);
}

/**
 * Camera-off placeholders. `track.enabled = false` emits BLACK frames
 * rather than stopping the stream, so without these a camera-off tile is
 * a black rectangle — indistinguishable from a frozen or dropped call.
 * The overlay data-attributes drive which placeholder shows; CSS owns the
 * geometry so the cards track their tiles through every role swap.
 */
function _setLocalCameraOff(off) {
  if (!_overlay) return;
  if (off) _overlay.dataset.localCamera = 'off';
  else delete _overlay.dataset.localCamera;
}

function _setPeerCameraOff(off) {
  if (!_overlay) return;
  if (off) _overlay.dataset.peerCamera = 'off';
  else delete _overlay.dataset.peerCamera;
}

/**
 * Fill the remote camera-off card with the peer's identity, reusing the
 * same name + initial helpers the headline and picker use so the card
 * can never disagree with the rest of the chrome. The local card needs
 * no identity — "You" plus the glyph is unambiguous.
 */
function _syncTileOffIdentity(session) {
  if (!_overlay || !session) return;
  const name = _prettyPeerName(session.peerDid) || 'Peer';
  const av = _overlay.querySelector('.connect-call-remote-off .connect-call-tile-off-avatar');
  const nm = _overlay.querySelector('.connect-call-remote-off .connect-call-tile-off-name');
  if (av) av.textContent = _initialFor(name);
  if (nm) nm.textContent = `${name}'s camera is off`;
}

/**
 * Keep a tile's box shape following its track's real dimensions.
 *
 * The `resize` event on a <video> fires whenever the intrinsic frame size
 * changes — which covers device rotation AND a front/back camera swap,
 * neither of which any viewport media query can observe. We clamp to a
 * sane band so an unusual camera can't render the PiP as a sliver, and
 * write a CSS custom property the tile rules consume.
 */
function _wireTileAspect(videoEl, offEl) {
  if (!videoEl || !_overlay) return;
  const apply = () => {
    const w = videoEl.videoWidth;
    const h = videoEl.videoHeight;
    if (!w || !h) return;
    const clamped = Math.min(Math.max(w / h, 3 / 4), 16 / 9);
    // Written on the ELEMENT, under one shared name. It used to be two
    // overlay-scoped variables (--tile-aspect / --remote-tile-aspect),
    // which meant every geometry rule had to know which tile it was
    // styling and pick the matching variable — the same coupling that
    // forced the per-tile rule duplication. Element-scoped, both tiles
    // read `var(--tile-aspect)` and each gets its own camera's shape.
    videoEl.style.setProperty('--tile-aspect', String(clamped));
    if (offEl) offEl.style.setProperty('--tile-aspect', String(clamped));
  };
  if (videoEl.dataset.aspectWired !== '1') {
    videoEl.dataset.aspectWired = '1';
    videoEl.addEventListener('resize', apply);
    videoEl.addEventListener('loadedmetadata', apply);
  }
  apply();
}

/**
 * Mirror the screen-share session state into the overlay chrome:
 * button pressed-state, badge visibility, and the self-view label.
 * Pulled out separately from the video sync so the screen-share-changed
 * emit doesn't re-run the full video pipeline (which could nuke
 * srcObject mid-replaceTrack).
 */
function _syncShareUiForSession(session) {
  if (!_overlay || !session) return;
  const shareBtn = _overlay.querySelector('.connect-call-screen-share');
  const badge = _overlay.querySelector('.connect-call-sharing-badge');
  const withVideo = !!session.withVideo;
  const supportsShare = canShareScreen()
    && typeof session.startScreenShare === 'function';
  const sharing = !!session.isScreenSharing;

  if (shareBtn) {
    // Visible only while video is on. We could show the button when
    // audio-only, but tapping it would either silently escalate to
    // video (surprising) or throw `screen_share_requires_video` and
    // toast. Hiding it is the honest signal.
    shareBtn.hidden = !(withVideo && supportsShare);
    shareBtn.setAttribute('aria-pressed', sharing ? 'true' : 'false');
    shareBtn.title = sharing ? 'Stop sharing' : 'Share screen';
    shareBtn.setAttribute('aria-label', sharing ? 'Stop sharing' : 'Share screen');
    if (sharing) shareBtn.classList.add('active');
    else shareBtn.classList.remove('active');
  }
  if (badge) badge.hidden = !sharing;
  // Local caption gets the " · sharing" suffix when sharing so the
  // user has a second persistent affordance besides the floating
  // badge (which they may have dragged or fullscreened past).
  const suffix = _overlay.querySelector('.connect-call-local-caption .connect-call-tile-suffix');
  if (suffix) suffix.hidden = !sharing;
  // Mirroring the self-view is right for a webcam (matches expectation
  // of looking in a mirror) but wrong for a screen share — the user
  // shouldn't see their own desktop reversed. CSS un-mirrors when
  // data-sharing="true". Lives at the overlay so other consumers
  // (e.g. spotlight) can key off it too.
  if (_overlay) {
    if (sharing) _overlay.dataset.sharing = 'true';
    else delete _overlay.dataset.sharing;
  }
}

function _wireSessionToOverlay(session) {
  let reachedConnected = false;
  _showOverlay(session.peerDid, session.state || CALL_STATES.CONNECTING, {
    withVideo: session.withVideo,
    localStream: session.localStream,
  });
  _syncVideoUiForSession(session);

  session.on('state-change', (next) => {
    _setOverlayState(next);
    if (next === CALL_STATES.CONNECTED) {
      reachedConnected = true;
      _startDurationTicker();
      _wireIdleHide();
      _wireLocalVideoDrag();
      _wireSpotlightSwap();
      _syncTileRoles();
      _armIdleHide();
    } else {
      _disarmIdleHide();
    }
    // Local stream may not be attached at first overlay-show (caller
    // case: localStream arrives during start()). Re-attach when the
    // state advances to negotiating or connected.
    if (session.withVideo && session.localStream && _overlay) {
      const localVideo = _overlay.querySelector('.connect-call-local-video');
      _attachVideoStream(localVideo, session.localStream);
    }
    _syncVideoUiForSession(session);
  });

  // Re-render the video tiles + escalate-button label when the
  // peer connection's modalities shift mid-call.
  session.on?.('negotiated', () => _syncVideoUiForSession(session));
  session.on?.('local-stream-updated', () => _syncVideoUiForSession(session));

  // Screen-share-only chrome refresh — keeps the badge + pressed
  // button state aligned with the session. Doesn't re-run the full
  // video sync, which could disturb srcObject during replaceTrack.
  session.on?.('screen-share-changed', () => _syncShareUiForSession(session));

  session.on('remote-stream', (stream) => {
    if (_audioEl) {
      _audioEl.srcObject = stream;
      // Some browsers don't autoplay reliably without an explicit play().
      _audioEl.play?.().catch(() => { /* user gesture already happened */ });
      // Apply the user's last-picked output device, if remembered
      // and still present. Fire-and-forget; errors fall back to default.
      _applyPreferredOutput();
    }
    // For video calls, also pipe the remote stream into the video tile.
    // The same MediaStream carries both audio and video tracks, so
    // assigning to both elements is fine — browsers route accordingly.
    // We also re-run for renegotiations that add a video track to an
    // audio-only call mid-stream.
    if (session.withVideo && _overlay) {
      const remoteVideo = _overlay.querySelector('.connect-call-remote-video');
      if (remoteVideo) _attachVideoStream(remoteVideo, stream);
    }
    // Drive the avatar pulse ring from the remote audio level so the
    // call feels "alive" — Discord/Meet pattern. Audio-only calls
    // especially need a presence cue beyond a static avatar.
    _startAudioPulse(stream);
    _syncVideoUiForSession(session);
  });

  session.on?.('quality-change', (q) => _updateQualityPill(q));
  session.on?.('peer-mute', ({ muted }) => _setPeerMuted(!!muted));
  // Camera-off is signalled explicitly rather than inferred: a disabled
  // track still sends black frames, so without this the remote tile is a
  // black rectangle the user can't distinguish from a frozen call.
  session.on?.('peer-video', ({ videoEnabled }) => _setPeerCameraOff(!videoEnabled));

  session.on('error', ({ reason, error }) => {
    const msg = _errorMessage(reason, error);
    // Empty message = a clean local hangup that surfaced as an error event;
    // nothing to tell the user. Otherwise show the self-contained sentence.
    if (msg) showToast(msg, 'error');
  });

  session.on('ended', ({ reason }) => {
    const callId = session.callId;
    const durationSeconds = _connectedSince
      ? Math.max(0, Math.floor((Date.now() - _connectedSince) / 1000))
      : 0;
    // Stop ticking + analyzing the moment media stops. Avatar pulse +
    // duration both halt right away so the end screen isn't fighting
    // them for rendering cycles.
    _stopDurationTicker();
    _stopAudioPulse();

    // Clear the active-session slot the moment the dialer reports
    // ENDED. The summary card's rejoin/dismiss buttons close over the
    // local `session` variable, not over _activeSession, so they
    // still work — and downstream startCall() will no longer be
    // blocked by an orphan session reference if the user closes the
    // overlay any other way (browser navigation, fullscreen exit,
    // tab close, async error before summary renders). Previously this
    // null-set lived only in the pre-media setTimeout and on the
    // summary card buttons, which is how the "A call is already in
    // progress" ghost could persist after a clean teardown.
    if (_activeSession === session) _activeSession = null;

    if (reachedConnected) {
      // Take over the overlay with the end-of-call screen. The user
      // dismisses or rejoins from there — no auto-fade timer.
      _renderEndOfCall(session, {
        callId,
        peerDid: session.peerDid,
        modalities: session.modalities,
        durationSeconds,
        reason,
      });
    } else {
      // Pre-media end (declined / missed / failed) — short hold so
      // the user reads the state, then dismiss. No rating prompt for
      // a call that never connected.
      const stateEl = _overlay?.querySelector('.connect-call-state');
      if (stateEl) {
        stateEl.hidden = false;
        stateEl.textContent = reason && reason !== 'local_hangup' ? `Ended — ${reason}` : 'Ended';
      }
      setTimeout(() => {
        _hideOverlay();
      }, 1800);
    }

    // Broadcast so the calls-history panel can refresh its cache.
    try {
      window.dispatchEvent(new CustomEvent('augmentum:connect-call-ended', {
        detail: { call_id: callId, peer_did: session.peerDid, reason },
      }));
    } catch (_) { /* listener errors are non-fatal — call already ended */ }
  });
}

function _errorMessage(reason, err) {
  return _friendlyCallReason(reason, err);
}

// ── Launcher registration ───────────────────────────────────────

function _registerCommands() {
  registerCommand({
    id: 'connect.placeCall',
    label: 'Connect: Place a call',
    hint: 'Find someone to message or call',
    group: 'Connect',
    keywords: 'call dial peer signaling webrtc',
    run: () => _openConnectHome('people'),
    when: () => _isEnabled(),
  });
  registerCommand({
    id: 'connect.invite',
    label: 'Connect: Invite someone',
    hint: 'Create an invite link so a new person can join and message you',
    group: 'Connect',
    keywords: 'invite link join onboard new person account',
    run: () => _openConnectHome('invite'),
    when: () => _isEnabled(),
  });
}

// ── Global mic long-press launcher ──────────────────────────────
//
// The header's #voice-call-btn is the chrome's canonical mic button.
// Tap toggles the Voice overlay (wired in voice.js). Hold opens the
// Connect picker — peer-to-peer is "calling another person" while
// tap-to-talk is "calling the assistant," so the gesture maps
// cleanly to user intent without cluttering the chrome with a
// sibling icon.
//
// Wiring lives here, not voice.js, so voice + Connect stay
// independently maintainable: future voice work can edit the click
// handler without thinking about Connect, and Connect can be
// disabled (connectEnabled=off) and the long-press disappears
// cleanly via _isEnabled().

const LONG_PRESS_MS = 500;

function _wireGlobalMicLongPress() {
  const btn = document.getElementById('voice-call-btn');
  if (!btn || btn.dataset.connectLongPressWired === '1') return;
  btn.dataset.connectLongPressWired = '1';

  let pressTimer = null;
  let longPressFired = false;
  let gestureActive = false;

  const clearGesture = () => {
    if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    gestureActive = false;
  };

  const fireLongPress = () => {
    pressTimer = null;
    longPressFired = true;
    // Hold opens a two-row chooser, NOT the home directly: "Chat with
    // assistant" (voice) vs "Connect" (the unified home). Keeping it a
    // menu means the disabled-Connect state can still surface a setup
    // entry rather than dead-ending.
    try {
      _openMicHoldMenu(btn);
    } catch (err) {
      console.warn('connect: long-press launcher failed', err);
    }
  };

  const onDown = (ev) => {
    if (!_isEnabled()) return;
    if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    gestureActive = true;
    longPressFired = false;
    // Touch path: suppress system long-press callout (text select /
    // link preview / context menu). Don't preventDefault for mouse —
    // it would block focus + the normal click path that voice.js owns.
    if (ev.cancelable && ev.pointerType === 'touch') {
      try { ev.preventDefault(); } catch (_) {}
    }
    pressTimer = setTimeout(fireLongPress, LONG_PRESS_MS);
  };

  const onUp = () => clearGesture();
  const onCancel = () => clearGesture();
  const onLeave = () => clearGesture();

  btn.addEventListener('pointerdown', onDown);
  btn.addEventListener('pointerup', onUp);
  btn.addEventListener('pointercancel', onCancel);
  btn.addEventListener('pointerleave', onLeave);

  // Fallback for embedded webviews where pointer events sometimes
  // don't fire even though touch events do. Same gesture machine.
  btn.addEventListener('touchstart', (ev) => {
    if (gestureActive) return;  // pointerdown already armed us
    if (!_isEnabled()) return;
    gestureActive = true;
    longPressFired = false;
    if (ev.cancelable) {
      try { ev.preventDefault(); } catch (_) {}
    }
    pressTimer = setTimeout(fireLongPress, LONG_PRESS_MS);
  }, { passive: false });
  btn.addEventListener('touchend', onUp);
  btn.addEventListener('touchcancel', onCancel);

  // iOS + Android both treat a sustained touch as a system callout
  // by default. CSS + contextmenu guards keep the gesture in our
  // hands long enough for the 500ms timer to fire.
  btn.style.touchAction = 'manipulation';
  btn.style.webkitTouchCallout = 'none';
  btn.style.webkitUserSelect = 'none';
  btn.style.userSelect = 'none';
  btn.addEventListener('contextmenu', (ev) => ev.preventDefault());

  // Capture-phase click suppressor — fires BEFORE the bubble-phase
  // click listener that voice.js wired for toggleVoiceCall. When a
  // long-press triggered the picker, swallow the synthetic click so
  // the voice overlay doesn't ALSO open underneath.
  btn.addEventListener('click', (ev) => {
    if (longPressFired) {
      longPressFired = false;
      ev.stopImmediatePropagation();
      ev.preventDefault();
    }
  }, { capture: true });

  // Tooltip hint — only update if the existing title doesn't already
  // mention the long-press affordance, so we don't fight other code.
  const existingTitle = btn.getAttribute('title') || '';
  if (!existingTitle.toLowerCase().includes('hold')) {
    btn.setAttribute('title', `${existingTitle} · Hold for Connect`.trim());
  }

  // Diagnostic seam — set a window flag so the user can verify
  // wiring from devtools:  > window.__augmentumConnectLongPressArmed
  window.__augmentumConnectLongPressArmed = true;
}

// ── Mic-hold discovery menu ────────────────────────────────────
//
// The long-press surfaces a two-row chooser rather than jumping
// straight into a surface:
//   • Chat with assistant — fires the existing voice-call-btn click path
//   • Connect             — opens the unified home (_openConnectHome)
//
// The two former Connect rows ("Call a friend" + "Open Connect")
// collapsed into the single Connect door; placing a call now lives
// inside the home's People section. The menu anchors to the mic
// button, dismisses on outside-click / Escape / option-select.

let _micMenu = null;

function _openMicHoldMenu(anchorBtn) {
  if (_micMenu) {
    _closeMicHoldMenu();
    return;
  }
  const rect = anchorBtn.getBoundingClientRect();
  const menu = document.createElement('div');
  menu.className = 'connect-mic-menu';
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', 'Voice & calling');
  menu.innerHTML = `
    <button class="connect-mic-menu-row" role="menuitem" data-action="becca">
      <span class="connect-mic-menu-icon">${icon('mic', { size: 18 })}</span>
      <span class="connect-mic-menu-text">
        <span class="connect-mic-menu-title">Chat with assistant</span>
        <span class="connect-mic-menu-sub">Voice chat with Becca</span>
      </span>
    </button>
    <button class="connect-mic-menu-row" role="menuitem" data-action="connect">
      <span class="connect-mic-menu-icon">${icon('phone', { size: 18 })}</span>
      <span class="connect-mic-menu-text">
        <span class="connect-mic-menu-title">Connect</span>
        <span class="connect-mic-menu-sub">Messages, calls &amp; people</span>
      </span>
    </button>
  `;
  document.body.appendChild(menu);
  _micMenu = menu;
  // Position under the button, right-aligned with the chrome header.
  const top = Math.round(rect.bottom + 8);
  const right = Math.round(window.innerWidth - rect.right);
  menu.style.top = `${top}px`;
  menu.style.right = `${right}px`;
  // Defer enter animation so the position lands before the transform.
  requestAnimationFrame(() => menu.classList.add('open'));

  const handle = (action) => {
    _closeMicHoldMenu();
    if (action === 'becca') {
      // Fire the existing click path. The capture-phase click
      // suppressor in _wireGlobalMicLongPress only swallows clicks
      // when longPressFired is still true — we cleared it on close,
      // so this synthetic click reaches voice.js' toggleVoiceCall.
      try { anchorBtn.click(); } catch (_) {}
    } else if (action === 'connect') {
      // Single Connect door → the unified home (calls + messages +
      // people + guests + invite + federation). Placing a call now
      // happens inside the home, so the old "Call a friend" row is
      // gone — its fast-path survives as the People section and the
      // `Connect: Place a call` palette command.
      _openConnectHome();
    }
  };

  for (const row of menu.querySelectorAll('.connect-mic-menu-row')) {
    row.addEventListener('click', () => handle(row.dataset.action));
  }

  // Dismiss on outside click + Escape.
  const onOutside = (ev) => {
    if (!_micMenu) return;
    if (_micMenu.contains(ev.target)) return;
    if (anchorBtn.contains(ev.target)) return;
    _closeMicHoldMenu();
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') _closeMicHoldMenu();
  };
  setTimeout(() => {
    // Defer attaching so the gesture that opened the menu doesn't
    // immediately close it. Capture phase so we win over child handlers.
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey);
  }, 0);
  menu._cleanup = () => {
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey);
  };
}

// ── Connect home launcher ──────────────────────────────────────
//
// The single destination behind the long-press "Connect" row and the
// six `Connect: …` palette commands. Phase 0 routes to the existing
// messaging master-detail panel (the closest thing to the home and
// the shell the unified home is built from); Phase 1 swaps the body
// to import ./home.js and route to a named section. Keeping the
// indirection here means the menu + palette callers never change as
// the home lands.
//
// `section` is a forward-looking hint ('chats'|'calls'|'people'|
// 'guests'|'invite'|'federation') ignored until the home shell exists.
function _openConnectHome(section = 'chats') {
  import('./home.js')
    .then((m) => m.openConnectHome?.(section))
    .catch((err) => console.warn('connect: open home failed', err));
}

function _closeMicHoldMenu() {
  if (!_micMenu) return;
  try { _micMenu._cleanup?.(); } catch (_) {}
  _micMenu.classList.remove('open');
  const m = _micMenu;
  _micMenu = null;
  setTimeout(() => { try { m.remove(); } catch (_) {} }, 140);
}


function _exposeGlobal() {
  // Console / debug surface. Lets you trial a call straight from
  // devtools without re-opening the picker:
  //   await augmentumConnect.placeCall('alice@home.alice.dev')
  window.augmentumConnect = {
    placeCall: (peerDid) => startCall(peerDid),
    openPicker,
  };
}

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}

// Escape helper kept for parity with the rest of the UI even though
// no innerHTML is built from user input above — peer DIDs go into
// textContent, not template literals.
void escapeHtml;
