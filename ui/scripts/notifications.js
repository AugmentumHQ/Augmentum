/* notifications.js — UI client for the notification substrate.
 *
 * Subscribes to GET /api/notify/subscribe (WS), renders each incoming
 * notification according to its importance + actions, and POSTs back
 * to the action callback endpoint when the user clicks an action
 * button or dismisses.
 *
 * Importance → presentation:
 *   * 0..2 (MIN/LOW/DEFAULT): transient toast via existing showToast()
 *   * 3 (HIGH) + actions: persistent banner with action buttons
 *   * 4 (CRITICAL): persistent banner, sound (when wired), pierces toasts
 *   * 3+ without actions: persistent banner with dismiss only
 *
 * The banner is a separate top-right container so toasts and banners
 * can coexist (a CRITICAL incoming-call shouldn't be obscured by a
 * stack of "knowledge pack imported" toasts).
 *
 * On boot the module:
 *   1. Checks the notificationsEnabled setting; if off, does nothing.
 *   2. POSTs /api/auth/ws-ticket to get an auth ticket (short-TTL).
 *   3. Opens WS /api/notify/subscribe?ticket=<t>.
 *   4. Fetches the existing feed GET /api/notify/feed?include_read=false
 *      to render missed-while-offline notifications.
 *   5. Reconnects with jittered backoff on close.
 *
 * Design discussion lives in docs/superpowers/specs/2026-06-01-
 * notification-substrate-design.md. The HTTP/WS contract is owned
 * by augmentum/proxy/notifications_routes.py.
 */

import { showToast, escapeHtml } from './app.js';
import { getSettings } from './settings.js';

const IMPORTANCE_HIGH = 3;
const IMPORTANCE_CRITICAL = 4;

// Reconnect with capped exponential backoff + jitter so a transient
// failure doesn't hammer the server. Mirrors the becca-presence
// pattern but capped tighter — notifications are auxiliary, not
// load-bearing for the chat UX, so a few extra seconds on retry is
// fine.
const RECONNECT_BASE_MS = 1500;
const RECONNECT_MAX_MS = 30000;

let _ws = null;
let _reconnectMs = RECONNECT_BASE_MS;
let _stopped = false;
// Track rendered banners keyed by notification_id so updates-in-place
// (dedupe_key republish) replace the banner instead of stacking.
const _banners = new Map();

// ── Public surface ───────────────────────────────────────────────

export async function initNotifications() {
  // The presence surface (live wallpaper + lock-screen avatar, both loaded
  // with ?presence=1) is PURELY VISUAL — the only interaction it affords is
  // summoning the assistant (wake word / power-button assist). It is not an
  // interactable surface: the wallpaper forwards no touch, so any banner/toast
  // we render there can never be dismissed and just stacks in front of the VRM,
  // hiding her. So never run the notification client on a presence surface —
  // briefings/notifications are delivered on the real app surfaces + the OS
  // notification shade instead. See presence-fullscreen.js.
  if (_isPresenceSurface()) return;
  if (!_isEnabled()) return;
  _ensureBannerContainer();
  await _initialFeedFetch();
  _connect();
}

/** True on the ambient companion presence surface (live wallpaper /
 *  lock-screen avatar). Marked by ?presence=1 — see presence-fullscreen.js. */
function _isPresenceSurface() {
  try {
    return new URLSearchParams(window.location.search).get('presence') === '1';
  } catch (_) {
    return false;
  }
}

export function stopNotifications() {
  _stopped = true;
  try { _ws?.close(); } catch (_) {}
  _ws = null;
}

// ── Setting + container helpers ──────────────────────────────────

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.notificationsEnabled);
}

// ── Audible cue ──────────────────────────────────────────────────

/** True when the wall clock is inside the user's quiet-hours window.
 *  Start '24:00' (the default) or start===end means "no quiet hours".
 *  Handles windows that wrap past midnight (e.g. 22:00–07:00). */
function _inQuietHours() {
  const s = getSettings?.() || {};
  const start = String(s.companionQuietHoursStart || '24:00');
  const end = String(s.companionQuietHoursEnd || '07:00');
  if (start === '24:00' || start === end) return false;
  const toMin = (hhmm) => {
    const [h, m] = hhmm.split(':').map((x) => parseInt(x, 10) || 0);
    return h * 60 + m;
  };
  const now = new Date();
  const cur = now.getHours() * 60 + now.getMinutes();
  const a = toMin(start);
  const b = toMin(end);
  return a < b ? (cur >= a && cur < b) : (cur >= a || cur < b);
}

/** Play the notification's cue, honoring the sound setting + quiet
 *  hours. CRITICAL pierces quiet hours (a tornado warning rings at 3am
 *  on purpose); everything else respects it. */
// Shared gate for both the chime and the spoken-briefing follow-on: sound
// must be enabled, and quiet hours silences everything below CRITICAL.
function _soundAllowed(n) {
  const s = getSettings?.() || {};
  if (s.notificationSoundEnabled === false) return false;
  const importance = Number(n.importance ?? 2);
  if (importance < IMPORTANCE_CRITICAL && _inQuietHours()) return false;
  return true;
}

function _maybePlaySound(n) {
  if (!_soundAllowed(n)) return;
  const s = getSettings?.() || {};
  const importance = Number(n.importance ?? 2);
  // User's chosen tone wins; 'auto' (or unset) defers to the channel's
  // catalog cue, then an importance-based fallback.
  const pref = String(s.notificationSound || 'auto');
  const name = (pref && pref !== 'auto')
    ? pref
    : (n.sound || (importance >= IMPORTANCE_HIGH ? 'chime' : 'ping'));
  import('./notification-sound.js')
    .then((m) => m.playNotificationSound(name))
    .catch(() => {});
}

// Per-briefing spoken delivery: chime → a beat → server-TTS of the briefing
// in the user's default voice. Gated on the same sound rules as the chime
// (quiet hours / sound-off silence it too) and only on live arrivals (stale
// catch-up replays don't narrate — handled by the !suppressBanner caller).
// The read-aloud pipeline synthesizes server-side via the user's configured
// voice provider; narrateNoteOnce dedupes against the drawer-open path.
function _maybeSpeakBriefing(n) {
  try {
    const p = n && n.payload;
    if (!p || !p.read_aloud || !p.speak_text) return;
    if (!_soundAllowed(n)) return;
    // One breath after the chime so the cue and the voice don't collide.
    setTimeout(() => {
      import('./read-aloud.js')
        .then((m) => m.narrateNoteOnce?.(
          p.note_id, p.speak_text, { title: n.title || 'Briefing' },
        ))
        .catch(() => {});
    }, 1000);
  } catch (_) { /* best effort */ }
}

function _ensureBannerContainer() {
  if (document.getElementById('notification-banner-container')) return;
  const div = document.createElement('div');
  div.id = 'notification-banner-container';
  div.setAttribute('role', 'region');
  div.setAttribute('aria-label', 'Notifications');
  document.body.appendChild(div);
}

// ── WS lifecycle ────────────────────────────────────────────────

async function _fetchTicket() {
  try {
    const resp = await fetch('/api/auth/ws-ticket', {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    return data.ticket || null;
  } catch (_) {
    return null;
  }
}

async function _connect() {
  if (_stopped) return;
  const ticket = await _fetchTicket();
  if (!ticket) {
    _scheduleReconnect();
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/notify/subscribe`
            + `?ticket=${encodeURIComponent(ticket)}`;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch (_) {
    _scheduleReconnect();
    return;
  }
  _ws = ws;

  ws.onopen = () => {
    // Reset backoff on a successful open. Don't reset until the
    // server-side handshake actually completed — the open event
    // fires on TCP connect, but auth happens on the next frame.
    _reconnectMs = RECONNECT_BASE_MS;
  };

  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch (_) { return; }
    if (!msg) return;
    if (msg.type === 'notification' && msg.notification) {
      _renderNotification(msg.notification);
    } else if (msg.type === 'notification_update' && msg.notification_id) {
      // Cross-client sync: it was read/dismissed/acted-on elsewhere (or
      // here). Clear the live banner so it doesn't linger on this client,
      // and let the bell/feed react. Idempotent — a no-op if we already
      // removed it optimistically on the originating client.
      _dismissBannerDom(msg.notification_id);
      try {
        window.dispatchEvent(new CustomEvent('augmentum:notification-update', {
          detail: { notificationId: msg.notification_id, state: msg.state || '' },
        }));
      } catch (_) { /* no CustomEvent support — banner removal already done */ }
    }
  };

  ws.onerror = () => { /* close handler does the work */ };

  ws.onclose = () => {
    _ws = null;
    _scheduleReconnect();
  };
}

function _scheduleReconnect() {
  if (_stopped) return;
  const jitter = Math.floor(Math.random() * 500);
  const wait = Math.min(_reconnectMs + jitter, RECONNECT_MAX_MS);
  _reconnectMs = Math.min(_reconnectMs * 2, RECONNECT_MAX_MS);
  setTimeout(_connect, wait);
}

// ── Initial feed (missed-while-offline) ──────────────────────────

// Catch-up replay grace window. Notifications older than this on the
// initial feed fetch (i.e. piled up while we were offline) skip the
// full banner overlay and only land in the bell. Without this an
// offline-then-online user opens the tab and gets every queued
// invite popping as a separate banner — 10 missed call rings = 10
// banners they have to dismiss one by one. Live WS pushes after the
// catch-up (i.e. genuinely new events) continue to banner normally.
const CATCHUP_BANNER_GRACE_MS = 30_000;

async function _initialFeedFetch() {
  try {
    const resp = await fetch(
      '/api/notify/feed?include_read=false&include_dismissed=false',
      { credentials: 'same-origin' },
    );
    if (!resp.ok) return;
    const data = await resp.json();
    const items = Array.isArray(data?.items) ? data.items : [];
    const now = Date.now();
    // Render oldest-first so the newest lands on top of the banner
    // stack (consistent with feed order DESC + LIFO stacking).
    items.slice().reverse().forEach((n) => {
      const ts = Date.parse(n.created_at || n.published_at || '');
      const isStale = !Number.isNaN(ts)
        && (now - ts) > CATCHUP_BANNER_GRACE_MS;
      _renderNotification(n, { suppressBanner: isStale });
    });
  } catch (_) {
    // The WS subscription is the primary path; missing the initial
    // fetch just means the user has to refresh to see backlog.
  }
}

// ── Render: banner vs toast ──────────────────────────────────────

function _renderNotification(n, { suppressBanner = false } = {}) {
  // Dedupe at the UI level — if the same notification_id arrives
  // twice (initial-fetch + WS push race), the second call replaces
  // the first banner.
  if (_banners.has(n.notification_id)) {
    _dismissBannerDom(n.notification_id);
  }

  // Channels owned by a dedicated UI surface — skip the generic
  // banner/toast so we don't double-prompt:
  //   * connect.call.incoming → full-screen modal (connect/incoming-modal.js)
  //   * system.offer          → inline offer chip in chat (chat/offer-chip.js)
  // The offer chip is the spec'd surface for gated-tool/install/switch
  // proposals (Accept/Not now/Never, persistent). Without this skip the
  // generic handler ALSO rendered each offer as a transient toast whose
  // primary action is labelled "Install" (the catalog's default action
  // label) and which auto-dismissed in ~3s — so e.g. an image-generation
  // proposal flashed a dead "Install" toast that vanished before it could
  // be clicked, shadowing the real chip.
  if (n.channel_id === 'connect.call.incoming') return;
  if (n.channel_id === 'system.offer') return;

  // Companion mood narrations (narrate_state_to_user verb) → becca-
  // presence status pill, reusing the idle/hosting slot rather than
  // firing a generic system toast. The title doubles as the PAD-
  // quadrant tag for ::before-dot colouring (energized/settled/
  // restless/subdued). Falls back to a quiet toast if the widget
  // isn't mounted yet.
  if (n.channel_id === 'companion.state') {
    const title = n.title || '';
    const body = n.body || '';
    const tag = title.trim().toLowerCase();
    if (typeof window !== 'undefined'
        && typeof window.beccaShowMood === 'function') {
      window.beccaShowMood(title, body, undefined, tag);
      return;
    }
    // Fallback path — pre-widget-mount or feature-disabled.
    showToast(title, 'info', undefined, body ? { description: body } : {});
    return;
  }

  // Audible cue — only for live arrivals (suppressBanner is the
  // offline catch-up replay; replaying a stack of cues would be noise).
  // Read-aloud briefings follow the chime with spoken delivery.
  if (!suppressBanner) {
    _maybePlaySound(n);
    _maybeSpeakBriefing(n);
  }

  const importance = Number(n.importance ?? 2);
  const hasActions = Array.isArray(n.actions) && n.actions.length > 0;

  // suppressBanner is set during catch-up replay for stale items —
  // they still need to land in the bell + feed (handled by the
  // notification client elsewhere), but skipping the overlay avoids
  // the offline-then-online banner spam.
  if (importance >= IMPORTANCE_HIGH && !suppressBanner) {
    _renderBanner(n);
    return;
  }
  if (suppressBanner) return;
  // Low / default: toast, optionally with a single click-through action.
  const opts = {};
  if (n.body) opts.description = n.body;
  if (hasActions) {
    const primary = n.actions.find(a => a.style === 'primary') || n.actions[0];
    opts.action = {
      label: primary.label || primary.id,
      onClick: () => {
        // Parity with the banner path: fire the synchronous DOM event
        // BEFORE the POST so subsystems that need user-gesture context
        // (navigation, getUserMedia) get a hook from toast clicks too —
        // previously only banner (HIGH-importance) actions dispatched
        // this, leaving DEFAULT-importance actions with no client hook.
        try {
          window.dispatchEvent(new CustomEvent('augmentum:notification-action', {
            detail: { notification: n, actionId: primary.id },
          }));
        } catch (_) { /* listeners are best-effort */ }
        _invokeAction(n.notification_id, primary.id);
      },
    };
  } else {
    // Quiet-tier task fires (watches, feed follows on the default
    // delivery) carry no server actions, but they DO carry the item —
    // give the toast the same one-tap Open/Play the banner has, or a
    // path to the full note in the drawer. A watch result the user
    // can't reach in one tap is a dead-end headline.
    const openUrl = (n.payload && typeof n.payload.open_url === 'string')
      ? n.payload.open_url : '';
    const isTaskNote = n.channel_id === 'companion.tasks';
    if (openUrl) {
      opts.action = {
        label: n.payload.open_kind === 'video' ? 'Play' : 'Open',
        onClick: () => _openNotificationMedia(
          openUrl, n.payload.open_kind, n.payload.open_title),
      };
    } else if (isTaskNote && typeof window.openCompanionNotes === 'function') {
      opts.action = {
        label: 'Open',
        onClick: () => { try { window.openCompanionNotes(); } catch (_) {} },
      };
    }
  }
  const type = importance >= IMPORTANCE_HIGH ? 'warning' : 'info';
  showToast(n.title || '', type, undefined, opts);
}

// Human-readable channel labels for the banner footer — raw dotted ids
// ("companion.tasks") read as debug output, not professional copy. Falls
// back to a prettified form of the id for channels not in the map.
const _CHANNEL_LABELS = {
  'companion.tasks': 'Scheduled tasks & briefings',
  'companion.state': 'Companion',
  'companion.observation': 'Companion',
  'time.timer': 'Timers & reminders',
};
function _friendlyChannel(id) {
  if (!id) return '';
  if (_CHANNEL_LABELS[id]) return _CHANNEL_LABELS[id];
  const seg = String(id).split('.').pop().replace(/[_-]+/g, ' ').trim();
  return seg ? seg.charAt(0).toUpperCase() + seg.slice(1) : '';
}

// Extract a YouTube video id from a watch / youtu.be url, or '' if not one.
function _youtubeId(url) {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\.|^m\./, '');
    if (host === 'youtube.com') return u.searchParams.get('v') || '';
    if (host === 'youtu.be') return u.pathname.slice(1).split('/')[0] || '';
  } catch (_) { /* not a parseable url */ }
  return '';
}

// Deep-link a briefing notification's primary media. YouTube links play in
// the in-app panel via the shared `media:play` event (youtube-panel.js
// registers the listener at module load, so import first); everything else
// opens in a new tab.
function _openNotificationMedia(url, kind, title) {
  if (!url) return;
  const vid = _youtubeId(url);
  if (vid) {
    import('./youtube-panel.js')
      .then(() => {
        window.dispatchEvent(new CustomEvent('media:play', {
          detail: { videoId: vid, title: title || '' },
        }));
      })
      .catch(() => { try { window.open(url, '_blank', 'noopener'); } catch (_) {} });
    return;
  }
  try { window.open(url, '_blank', 'noopener'); } catch (_) { /* blocked */ }
}

// ── Web Push click-through ───────────────────────────────────────
// The service worker (notification-sw.js) forwards a push-notification
// tap as an `augmentum:notification-click` window event (re-dispatched
// by push-subscribe.js). Route task fires the same way the in-app
// banner does: straight to the item (in-app player for video, new tab
// otherwise), else to the full note in the drawer. Installed at module
// scope — notifications.js is statically imported at boot, so the
// listener exists before any push tap can arrive.
if (typeof window !== 'undefined' && !window.__augNotifClickRouted) {
  window.__augNotifClickRouted = true;
  window.addEventListener('augmentum:notification-click', (ev) => {
    const d = (ev && ev.detail) || {};
    const p = d.payload || {};
    if (typeof p.open_url === 'string' && p.open_url) {
      _openNotificationMedia(p.open_url, p.open_kind, p.open_title);
    } else if (d.channel_id === 'companion.tasks') {
      try { window.openCompanionNotes?.(); } catch (_) { /* drawer absent */ }
    }
  });
}

function _renderBanner(n) {
  const container = document.getElementById('notification-banner-container');
  if (!container) return;

  const el = document.createElement('div');
  el.className = `notification-banner importance-${n.importance}`;
  el.dataset.notificationId = n.notification_id;
  el.dataset.channelId = n.channel_id || '';
  el.setAttribute('role', n.importance >= IMPORTANCE_CRITICAL ? 'alert' : 'status');
  el.setAttribute('aria-live',
    n.importance >= IMPORTANCE_CRITICAL ? 'assertive' : 'polite');

  // Visible content is escaped because notification title/body can
  // include peer-supplied strings (e.g. "Call from <user>"). The
  // payload field is opaque and only used for routing — never rendered.
  const titleHtml = escapeHtml(n.title || '');
  const bodyHtml = n.body ? escapeHtml(n.body) : '';
  const channelLabel = escapeHtml(_friendlyChannel(n.channel_id));
  const iconHtml = n.icon ? escapeHtml(n.icon) : '';

  let html = '';
  html += '<div class="notification-banner-row">';
  if (iconHtml) html += `<div class="notification-banner-icon">${iconHtml}</div>`;
  html += '<div class="notification-banner-text">';
  html += `<div class="notification-banner-title">${titleHtml}</div>`;
  if (bodyHtml) html += `<div class="notification-banner-body">${bodyHtml}</div>`;
  if (channelLabel) {
    html += `<div class="notification-banner-channel">${channelLabel}</div>`;
  }
  html += '</div>';
  html += '<button class="notification-banner-dismiss" aria-label="Dismiss">&#x2715;</button>';
  html += '</div>';

  const actions = Array.isArray(n.actions) ? n.actions : [];
  // Scheduled-task results carry no server actions, but the banner is
  // headline-only — give the user a one-tap path to the full rich note
  // (media, sources, full text) instead of a dead-end headline.
  const isTaskNote = n.channel_id === 'companion.tasks';
  // A media briefing carries a deep-link target (payload.open_url) so the
  // one-tap action lands ON the video/page; text-only briefings fall back to
  // opening the notes drawer to read the full note.
  const openUrl = (n.payload && typeof n.payload.open_url === 'string')
    ? n.payload.open_url : '';
  const canDrawer = isTaskNote && typeof window !== 'undefined'
    && typeof window.openCompanionNotes === 'function';
  if (actions.length > 0) {
    html += '<div class="notification-banner-actions">';
    for (const a of actions) {
      const style = a.style && /^[a-z0-9_-]+$/i.test(a.style) ? a.style : 'default';
      html += `<button class="notification-banner-action style-${escapeHtml(style)}"`
            + ` data-action-id="${escapeHtml(a.id)}">${escapeHtml(a.label)}</button>`;
    }
    html += '</div>';
  } else if (openUrl || canDrawer) {
    const openLabel = (openUrl && n.payload.open_kind === 'video') ? 'Play' : 'Open';
    html += '<div class="notification-banner-actions">'
          + '<button class="notification-banner-action style-primary"'
          + ` data-open-notes="1">${openLabel}</button></div>`;
  }
  el.innerHTML = html;

  // Wire the scheduled-task one-tap affordance. A media briefing deep-links
  // to the video/page (in-app player for video hosts, new tab otherwise);
  // a text-only briefing opens the companion notes drawer for the full note.
  el.querySelector('[data-open-notes]')?.addEventListener('click', () => {
    if (openUrl) {
      _openNotificationMedia(openUrl, n.payload.open_kind, n.payload.open_title);
    } else {
      try { window.openCompanionNotes?.(); } catch (_) { /* defensive */ }
    }
    _dismissBanner(n.notification_id);
  });

  // Wire action buttons. Scope to buttons that actually carry a server
  // action id — the task-note "Open" shortcut above shares the
  // .notification-banner-action class but has only [data-open-notes]
  // (no [data-action-id]); without this filter it would ALSO get the
  // POST handler and fire /api/notify/<id>/action/undefined (400).
  for (const btn of el.querySelectorAll('.notification-banner-action[data-action-id]')) {
    btn.addEventListener('click', async () => {
      const aid = btn.dataset.actionId;
      // Fire a synchronous DOM event BEFORE the POST so subsystems
      // that need user-gesture context (e.g. connect/incoming.js
      // calling getUserMedia) can kick off work without losing the
      // click-gesture permission. Listeners may stash promises on
      // event.detail to coordinate teardown if the action fails.
      try {
        window.dispatchEvent(new CustomEvent('augmentum:notification-action', {
          detail: { notification: n, actionId: aid },
        }));
      } catch (_) { /* defensive */ }
      btn.disabled = true;
      btn.classList.add('pending');
      // Harness agent bridge "Reply…": prompt for text BEFORE any POST —
      // the action route marks the notification read (clearing the banner
      // on every client), so prompting after it strands the request when
      // the user cancels. Cancel = banner stays, request stays pending.
      if (n.channel_id === 'harness.agent.request' && aid === 'reply') {
        const reqId = n.payload && n.payload.request_id;
        const text = window.prompt(
          `Reply to the agent:\n${n.title || ''}`.trim(), '');
        if (!text || !text.trim() || !reqId) {
          btn.disabled = false;
          btn.classList.remove('pending');
          return;
        }
        try {
          const r = await fetch('/api/harness/agent/reply', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ request_id: reqId, text: text.trim() }),
          });
          showToast(r.ok ? 'Reply sent to the agent.'
                         : 'Reply failed to send.', r.ok ? 'success' : 'error');
          if (!r.ok) {
            btn.disabled = false;
            btn.classList.remove('pending');
            return;
          }
        } catch (_) {
          showToast('Reply failed to send.', 'error');
          btn.disabled = false;
          btn.classList.remove('pending');
          return;
        }
        _dismissBanner(n.notification_id);
        return;
      }
      const result = await _invokeAction(n.notification_id, aid);
      if (result.ok) {
        _dismissBannerDom(n.notification_id);
      } else {
        btn.disabled = false;
        btn.classList.remove('pending');
        showToast(
          `Action failed: ${result.error || 'unknown error'}`,
          'error',
        );
      }
    });
  }
  // Wire dismiss button.
  el.querySelector('.notification-banner-dismiss')?.addEventListener(
    'click',
    () => _dismissBanner(n.notification_id),
  );

  container.appendChild(el);
  _banners.set(n.notification_id, el);
}

function _dismissBannerDom(notificationId) {
  const el = _banners.get(notificationId);
  if (el) {
    el.remove();
    _banners.delete(notificationId);
  }
}

// ── Action + dismiss calls ───────────────────────────────────────

async function _invokeAction(notificationId, actionId) {
  // String concat (not template literal) so the wiring scanner can
  // see both URL segments — its template-literal matcher trims
  // after the first `${...}`.
  const url = '/api/notify/' + encodeURIComponent(notificationId)
            + '/action/' + encodeURIComponent(actionId);
  try {
    const resp = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) {
      let detail = '';
      try { detail = (await resp.json()).detail || ''; } catch (_) {}
      return { ok: false, error: detail || `HTTP ${resp.status}` };
    }
    return { ok: true, data: await resp.json() };
  } catch (e) {
    return { ok: false, error: String(e?.message || e) };
  }
}

async function _dismissBanner(notificationId) {
  // Optimistically remove from the DOM. If the dismiss POST fails
  // we leave the row alone server-side; the next reconnect will
  // re-render it from the feed.
  _dismissBannerDom(notificationId);
  try {
    await fetch(
      `/api/notify/${encodeURIComponent(notificationId)}/dismiss`,
      { method: 'POST', credentials: 'same-origin' },
    );
  } catch (_) {
    // best-effort
  }
}
