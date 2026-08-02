/* connect/calls-panel.js — Connect call history surface.
 *
 * Docked panel (right side, parallel to thread-panel.js) with two
 * columns:
 *
 *   ┌────────────┬──────────────────────────┐
 *   │ Filter     │ Call detail              │
 *   │ All        │  peer + direction        │
 *   │ Missed     │  state pill              │
 *   │ Completed  │  duration / timestamps   │
 *   │ ── list    │  events timeline         │
 *   │ • peer     │  rating control          │
 *   │ • state    │                          │
 *   └────────────┴──────────────────────────┘
 *
 * Reads:
 *   GET /api/connect/calls            — list w/ ?state= + ?before=
 *   GET /api/connect/calls/{call_id}  — detail w/ events timeline
 *   POST /api/connect/calls/{id}/rate — set rating + notes
 *
 * The panel is created on-demand the first time it's opened (via
 * command palette: "Connect: Open call history") and hidden between
 * uses. State filter chips drive ?state= and paginate via the cursor
 * the API returns in the last row's initiated_at field.
 */

import { escapeHtml, showToast } from '../app.js';
import { getSettings } from '../settings.js';
import { registerCommand } from '../command-palette.js';
import { icon } from './icons.js';
import { getCallDetail, listCalls, rateCall, resolvePeerName } from './messages.js';
import { startCall } from './ui.js';

const PAGE_SIZE = 50;
const FILTER_LABELS = Object.freeze({
  all:       'All',
  missed:    'Missed',
  ended:     'Completed',
  declined:  'Declined',
});

let _panel = null;
let _activeFilter = 'all';
let _activeCallId = '';
let _callsCache = [];          // newest-first list
let _detailCache = new Map();   // call_id → {call, events}
let _initialized = false;
let _pageCursor = '';           // initiated_at of oldest loaded row
let _exhausted = false;         // true when API returned < PAGE_SIZE

// ── Init ────────────────────────────────────────────────────────

let _deferredRetryArmed = false;

export function initConnectCallsUI() {
  if (_initialized) return;
  if (!_isEnabled()) {
    if (!_deferredRetryArmed) {
      _deferredRetryArmed = true;
      const retry = () => {
        if (_initialized || !_isEnabled()) return;
        try { initConnectCallsUI(); }
        catch (e) { console.warn('[connect-calls] deferred init failed', e); }
      };
      window.addEventListener('augmentum:settings-loaded', retry);
      window.addEventListener('augmentum:connect-enabled', retry);
    }
    return;
  }
  _initialized = true;

  registerCommand({
    id: 'connect.openCalls',
    label: 'Connect: Open call history',
    hint: 'Show recent calls, missed-call list, and quality ratings',
    group: 'Connect',
    keywords: 'connect calls history missed dial recents log',
    run: () => import('./home.js').then((m) => m.openConnectHome('calls')),
    when: () => _isEnabled(),
  });

  // Inbound call events bump the list. We listen at the window so the
  // panel re-renders whether or not it's currently visible (cheap when
  // closed — the cache update happens regardless, render is no-op).
  window.addEventListener('augmentum:connect-call-ended', _onCallEnded);

  window.augmentumConnectCalls = {
    open: openCallsPanel,
    refresh: _refreshList,
  };
}

// ── Public-ish surface ──────────────────────────────────────────

export async function openCallsPanel(callId = '') {
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  if (!_panel) _ensurePanel();
  _panel.classList.remove('hidden');
  try {
    await _refreshList();
    if (callId) {
      await _openCallDetail(callId);
    } else if (_activeCallId && _callsCache.some((c) => c.call_id === _activeCallId)) {
      await _openCallDetail(_activeCallId);
    }
  } catch (err) {
    console.warn('connect: open calls failed', err);
  }
}

export function closeCallsPanel() {
  if (_panel) _panel.classList.add('hidden');
}

/**
 * Embed the call-history panel inside a host container (the Connect
 * home's Calls section) rather than floating it on <body>. Mirrors
 * thread-panel.js::mountMessagingInto — the whole `.connect-calls-panel`
 * subtree relocates into `host` + `.is-embedded`; CSS strips the
 * floating chrome so it fills the content region.
 */
export async function mountCallsInto(host) {
  if (!host) return;
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  if (!_panel) _ensurePanel();
  _panel.classList.add('is-embedded');
  _panel.classList.remove('hidden');
  if (_panel.parentElement !== host) host.appendChild(_panel);
  try {
    await _refreshList();
    if (_activeCallId && _callsCache.some((c) => c.call_id === _activeCallId)) {
      await _openCallDetail(_activeCallId);
    }
  } catch (err) {
    console.warn('connect: mount calls failed', err);
  }
}

// ── DOM construction ────────────────────────────────────────────

function _ensurePanel() {
  if (_panel) return _panel;
  const el = document.createElement('div');
  el.className = 'connect-calls-panel hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', 'Connect call history');
  el.innerHTML = `
    <div class="connect-calls-panel-card">
      <div class="connect-calls-panel-header">
        <div class="connect-calls-panel-title">Calls</div>
        <button class="connect-calls-panel-close" type="button" aria-label="Close">&#x2715;</button>
      </div>
      <div class="connect-calls-panel-body">
        <aside class="connect-calls-aside" aria-label="Call filters">
          <div class="connect-calls-filters" role="tablist"></div>
          <div class="connect-calls-list" aria-label="Recent calls"></div>
          <button class="connect-calls-more" type="button" hidden>Load older</button>
        </aside>
        <section class="connect-calls-detail">
          <div class="connect-calls-detail-empty">
            <div class="connect-calls-detail-empty-glyph">${icon('phone', { size: 48 })}</div>
            <div class="connect-calls-detail-empty-title">Pick a call</div>
            <div class="connect-calls-detail-empty-sub">Tap any row to see who called, how long it lasted, and the event timeline.</div>
          </div>
          <div class="connect-calls-detail-body" hidden></div>
        </section>
      </div>
    </div>
  `;
  document.body.appendChild(el);
  _panel = el;

  el.querySelector('.connect-calls-panel-close')
    .addEventListener('click', closeCallsPanel);

  _renderFilters();

  el.querySelector('.connect-calls-more')
    .addEventListener('click', _loadOlder);

  el.addEventListener('click', (ev) => {
    // Backdrop click closes the panel.
    if (ev.target === el) closeCallsPanel();
  });

  return el;
}

function _renderFilters() {
  if (!_panel) return;
  const wrap = _panel.querySelector('.connect-calls-filters');
  if (!wrap) return;
  wrap.innerHTML = Object.entries(FILTER_LABELS).map(([key, label]) => {
    const active = key === _activeFilter ? ' active' : '';
    return `
      <button class="connect-calls-filter${active}" data-filter="${escapeHtml(key)}"
              type="button" role="tab" aria-selected="${active ? 'true' : 'false'}">
        ${escapeHtml(label)}
      </button>
    `;
  }).join('');
  for (const btn of wrap.querySelectorAll('.connect-calls-filter')) {
    btn.addEventListener('click', () => {
      const next = btn.dataset.filter;
      if (next === _activeFilter) return;
      _activeFilter = next;
      _refreshList().catch((err) => console.warn('connect: filter refresh failed', err));
    });
  }
}

// ── List ────────────────────────────────────────────────────────

async function _refreshList() {
  _pageCursor = '';
  _exhausted = false;
  _callsCache = [];
  await _fetchPage();
  _renderFilters();
  _renderList();
}

async function _loadOlder() {
  if (_exhausted) return;
  await _fetchPage();
  _renderList();
}

async function _fetchPage() {
  const stateParam = _activeFilter === 'all' ? null : _activeFilter;
  try {
    const calls = await listCalls({
      limit: PAGE_SIZE,
      state: stateParam,
      before: _pageCursor || null,
    });
    if (!Array.isArray(calls)) return;
    _callsCache.push(...calls);
    if (calls.length < PAGE_SIZE) {
      _exhausted = true;
    } else {
      _pageCursor = calls[calls.length - 1]?.initiated_at || '';
    }
  } catch (err) {
    console.warn('connect: listCalls failed', err);
    showToast('Could not load calls', 'error');
  }
}

function _renderList() {
  if (!_panel) return;
  const list = _panel.querySelector('.connect-calls-list');
  const more = _panel.querySelector('.connect-calls-more');
  if (!list) return;
  if (_callsCache.length === 0) {
    const filterLabel = FILTER_LABELS[_activeFilter] || 'this filter';
    const hint = _activeFilter === 'all'
      ? 'Make your first call to see it here.'
      : `No ${filterLabel.toLowerCase()} calls on record.`;
    list.innerHTML = `
      <div class="connect-calls-list-empty">
        <div class="connect-calls-list-empty-glyph">${icon('history', { size: 32 })}</div>
        <div class="connect-calls-list-empty-text">No calls yet</div>
        <div class="connect-calls-list-empty-sub">${escapeHtml(hint)}</div>
      </div>
    `;
    if (more) more.hidden = true;
    return;
  }
  list.innerHTML = _callsCache.map((c) => {
    const peer = escapeHtml(c.peer_display_name || resolvePeerName(c.peer_did || ''));
    const isMissed = c.state === 'missed';
    const isOutgoing = c.direction === 'outgoing';
    const arrow = isMissed
      ? `<span class="connect-calls-row-arrow missed" aria-label="Missed">${icon('phone-missed', { size: 14 })}</span>`
      : (isOutgoing
        ? `<span class="connect-calls-row-arrow out" aria-label="Outgoing">${icon('arrow-up-right', { size: 14 })}</span>`
        : `<span class="connect-calls-row-arrow in" aria-label="Incoming">${icon('arrow-down-left', { size: 14 })}</span>`);
    const stamp = escapeHtml(_humaniseDate(c.initiated_at));
    const sub = _formatRowSubtitle(c);
    const activeCls = c.call_id === _activeCallId ? ' active' : '';
    return `
      <div class="connect-calls-row${activeCls}${isMissed ? ' missed' : ''}"
           data-call-id="${escapeHtml(c.call_id)}"
           role="button" tabindex="0">
        ${arrow}
        <div class="connect-calls-row-text">
          <div class="connect-calls-row-peer">${peer}</div>
          <div class="connect-calls-row-sub">${escapeHtml(sub)}</div>
        </div>
        <div class="connect-calls-row-time">${stamp}</div>
      </div>
    `;
  }).join('');
  for (const row of list.querySelectorAll('.connect-calls-row')) {
    const open = () => _openCallDetail(row.dataset.callId);
    row.addEventListener('click', open);
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        open();
      }
    });
  }
  if (more) more.hidden = _exhausted;
}

function _formatRowSubtitle(c) {
  if (c.state === 'missed') return 'Missed';
  if (c.state === 'declined') return 'Declined';
  if (c.state === 'failed') return 'Failed';
  if (typeof c.duration_seconds === 'number') {
    return `${c.modalities?.includes('video') ? 'Video' : 'Voice'} · ${_humaniseDuration(c.duration_seconds)}`;
  }
  if (c.state === 'connected' || c.state === 'ringing' || c.state === 'invited') {
    return 'In progress';
  }
  return c.state || '';
}

// ── Detail ──────────────────────────────────────────────────────

// Mobile single-pane: return from a call's detail to the calls list.
function _backToCallList() {
  _activeCallId = '';
  if (_panel) {
    _panel.classList.remove('show-detail');
    const emptyEl = _panel.querySelector('.connect-calls-detail-empty');
    const bodyEl = _panel.querySelector('.connect-calls-detail-body');
    if (emptyEl) emptyEl.hidden = false;
    if (bodyEl) bodyEl.hidden = true;
  }
  _renderList();
}

async function _openCallDetail(callId) {
  if (!_panel) _ensurePanel();
  _activeCallId = callId;
  // Mobile single-pane: reveal the detail pane (CSS hides the list at
  // ≤700px while this class is present). No-op on desktop.
  _panel.classList.add('show-detail');
  _renderList();
  const emptyEl = _panel.querySelector('.connect-calls-detail-empty');
  const bodyEl = _panel.querySelector('.connect-calls-detail-body');
  emptyEl.hidden = true;
  bodyEl.hidden = false;
  bodyEl.innerHTML = '<div class="connect-calls-detail-loading">Loading…</div>';

  let detail = _detailCache.get(callId);
  if (!detail) {
    try {
      detail = await getCallDetail(callId);
    } catch (err) {
      console.warn('connect: getCallDetail failed', err);
      bodyEl.innerHTML = '<div class="connect-calls-detail-error">Could not load this call.</div>';
      return;
    }
  }
  if (!detail || !detail.call) {
    bodyEl.innerHTML = '<div class="connect-calls-detail-error">Call not found.</div>';
    return;
  }
  _detailCache.set(callId, detail);
  _renderDetail(detail);
}

function _renderDetail(detail) {
  if (!_panel) return;
  const bodyEl = _panel.querySelector('.connect-calls-detail-body');
  if (!bodyEl) return;
  const call = detail.call;
  const events = Array.isArray(detail.events) ? detail.events : [];
  const peer = escapeHtml(call.peer_display_name || resolvePeerName(call.peer_did || ''));
  const directionLabel = call.state === 'missed'
    ? 'Missed call'
    : (call.direction === 'outgoing' ? 'Outgoing call' : 'Incoming call');
  const modality = (call.modalities || '').includes('video') ? 'Video' : 'Voice';
  const stateChip = `
    <span class="connect-calls-state-chip state-${escapeHtml(call.state || 'unknown')}">
      ${escapeHtml(_humaniseState(call.state))}
    </span>
  `;
  const ratingHtml = _renderRatingControl(call);
  const eventsHtml = _renderEventsTimeline(events);
  const initiated = escapeHtml(_humaniseDateTime(call.initiated_at));
  const connected = call.connected_at
    ? `<div>Connected ${escapeHtml(_humaniseDateTime(call.connected_at))}</div>`
    : '';
  const duration = (typeof call.duration_seconds === 'number')
    ? `<div>Duration ${escapeHtml(_humaniseDuration(call.duration_seconds))}</div>`
    : '';
  const reason = call.end_reason
    ? `<div>Ended · ${escapeHtml(_humaniseEndReason(call.end_reason))}</div>`
    : '';
  const canCallBack = call.peer_did && call.state !== 'connected' &&
                      call.state !== 'ringing' && call.state !== 'invited';
  const callBackBtn = canCallBack
    ? `<button class="connect-calls-detail-callback" type="button">Call back</button>`
    : '';

  bodyEl.innerHTML = `
    <div class="connect-calls-detail-head">
      <div class="connect-calls-detail-headtop">
        <button class="connect-calls-mobile-back" type="button" data-action="back"
                aria-label="Back to calls" title="Back">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <div class="connect-calls-detail-peer">${peer}</div>
        ${stateChip}
      </div>
      <div class="connect-calls-detail-meta">
        <div>${escapeHtml(directionLabel)} · ${escapeHtml(modality)}</div>
        <div>Started ${initiated}</div>
        ${connected}
        ${duration}
        ${reason}
      </div>
      <div class="connect-calls-detail-actions">${callBackBtn}</div>
    </div>
    <div class="connect-calls-detail-rating">${ratingHtml}</div>
    <div class="connect-calls-detail-timeline">
      <div class="connect-calls-detail-timeline-title">Timeline</div>
      ${eventsHtml}
    </div>
  `;

  const backEl = bodyEl.querySelector('.connect-calls-mobile-back');
  if (backEl) backEl.addEventListener('click', _backToCallList);

  if (canCallBack) {
    const btn = bodyEl.querySelector('.connect-calls-detail-callback');
    if (btn) {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        try {
          await startCall(call.peer_did, {
            withVideo: (call.modalities || '').includes('video'),
          });
        } catch (err) {
          showToast(`Call back failed: ${err?.message || 'unknown error'}`, 'error');
        } finally {
          btn.disabled = false;
        }
      });
    }
  }

  _wireRatingControl(bodyEl, call);
}

function _renderRatingControl(call) {
  const current = (typeof call.quality_rating === 'number')
    ? call.quality_rating : null;
  const notes = call.quality_notes || '';
  const states = [
    { val:  1, label: 'Good', iconName: 'thumbs-up' },
    { val:  0, label: 'OK',   iconName: 'minus' },
    { val: -1, label: 'Bad',  iconName: 'thumbs-down' },
  ];
  const btns = states.map((s) => {
    const pressed = current === s.val ? 'true' : 'false';
    return `
      <button class="connect-calls-rate-btn" type="button"
              data-rating="${s.val}" aria-pressed="${pressed}"
              title="${escapeHtml(s.label)}">
        <span class="connect-calls-rate-glyph">${icon(s.iconName, { size: 18 })}</span>
        <span class="connect-calls-rate-label">${escapeHtml(s.label)}</span>
      </button>
    `;
  }).join('');
  return `
    <div class="connect-calls-rate-title">How did this call go?</div>
    <div class="connect-calls-rate-buttons">${btns}</div>
    <textarea class="connect-calls-rate-notes" rows="2"
              placeholder="Optional note (echo, dropouts, you tell us)…"
              maxlength="2000">${escapeHtml(notes)}</textarea>
    <div class="connect-calls-rate-status" hidden></div>
  `;
}

function _wireRatingControl(bodyEl, call) {
  const buttons = bodyEl.querySelectorAll('.connect-calls-rate-btn');
  const notesEl = bodyEl.querySelector('.connect-calls-rate-notes');
  const statusEl = bodyEl.querySelector('.connect-calls-rate-status');
  let pending = null;
  let lastRating = (typeof call.quality_rating === 'number') ? call.quality_rating : null;

  const submit = async (rating) => {
    if (pending) clearTimeout(pending);
    pending = null;
    const notes = (notesEl?.value || '').slice(0, 2000);
    if (statusEl) {
      statusEl.hidden = false;
      statusEl.textContent = 'Saving…';
      statusEl.className = 'connect-calls-rate-status saving';
    }
    try {
      await rateCall(call.call_id, rating, notes);
      call.quality_rating = rating;
      call.quality_notes = notes;
      lastRating = rating;
      // Update cached list row + detail cache.
      const cached = _detailCache.get(call.call_id);
      if (cached) {
        cached.call.quality_rating = rating;
        cached.call.quality_notes = notes;
      }
      const listRow = _callsCache.find((c) => c.call_id === call.call_id);
      if (listRow) {
        listRow.quality_rating = rating;
        listRow.quality_notes = notes;
      }
      if (statusEl) {
        statusEl.textContent = 'Saved.';
        statusEl.className = 'connect-calls-rate-status saved';
        setTimeout(() => { if (statusEl) statusEl.hidden = true; }, 1500);
      }
    } catch (err) {
      console.warn('connect: rateCall failed', err);
      if (statusEl) {
        statusEl.textContent = 'Could not save. Try again.';
        statusEl.className = 'connect-calls-rate-status error';
      }
    }
  };

  for (const btn of buttons) {
    btn.addEventListener('click', () => {
      const rating = parseInt(btn.dataset.rating, 10);
      // Reset other buttons' aria-pressed; set this one.
      for (const b of buttons) b.setAttribute('aria-pressed', 'false');
      btn.setAttribute('aria-pressed', 'true');
      submit(rating);
    });
  }

  if (notesEl) {
    notesEl.addEventListener('input', () => {
      // Re-submit on debounced typing-stop so the note saves with the
      // current rating choice. Only meaningful once a rating is picked.
      if (lastRating === null) return;
      if (pending) clearTimeout(pending);
      pending = setTimeout(() => submit(lastRating), 700);
    });
  }
}

function _renderEventsTimeline(events) {
  if (!events.length) {
    return '<div class="connect-calls-detail-timeline-empty">No events recorded.</div>';
  }
  return events.map((ev) => {
    const t = escapeHtml(_humaniseTime(ev.created_at));
    const type = escapeHtml(_humaniseEventType(ev.event_type || ''));
    let body = '';
    if (ev.event_data && typeof ev.event_data === 'object') {
      const flat = Object.entries(ev.event_data)
        .filter(([, v]) => v !== '' && v !== null && v !== undefined)
        .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
        .join(' · ');
      if (flat) body = `<div class="connect-calls-tl-body">${escapeHtml(flat)}</div>`;
    }
    return `
      <div class="connect-calls-tl-row">
        <div class="connect-calls-tl-time">${t}</div>
        <div class="connect-calls-tl-type">${type}</div>
        ${body}
      </div>
    `;
  }).join('');
}

// ── Event reactors ──────────────────────────────────────────────

function _onCallEnded(evt) {
  const detail = evt.detail || {};
  const callId = detail.call_id || '';
  if (!callId) return;
  // Drop any cached detail so the next open re-fetches with the now-
  // terminal state. Cheap; the panel may be closed when this fires.
  _detailCache.delete(callId);
  // Refresh in the background if the panel is currently visible.
  if (_panel && !_panel.classList.contains('hidden')) {
    _refreshList().catch(() => {});
  }
}

// ── Helpers ────────────────────────────────────────────────────

const _END_REASON_LABELS = Object.freeze({
  peer_hangup:        'Other side hung up',
  local_hangup:       'You hung up',
  declined:           'Declined',
  missed:             'No answer',
  timeout:            'Timed out',
  failed:             'Connection failed',
  ice_failed:         'Network couldn’t connect',
  network_lost:       'Network dropped',
  cancelled:          'Cancelled before connect',
});
function _humaniseEndReason(reason) {
  if (!reason) return '';
  const key = String(reason).toLowerCase();
  return _END_REASON_LABELS[key] || reason.replace(/_/g, ' ');
}

const _EVENT_TYPE_LABELS = Object.freeze({
  'signaling.invite':           'Invite sent',
  'signaling.offer':            'Offer exchanged',
  'signaling.answer':           'Answer received',
  'signaling.candidates':       'ICE candidates',
  'signaling.accept':           'Accepted',
  'signaling.decline':          'Declined',
  'signaling.hangup':           'Hangup',
  'signaling.negotiate':        'Renegotiated',
  'signaling.select_answer':    'Answer selected',
  'signaling.mute_state':       'Mute state changed',
  'peer.connected':             'Peer connected',
  'peer.disconnected':          'Peer disconnected',
  'media.attached':             'Media stream attached',
  'media.failed':               'Media failed',
  'call.missed':                'Missed (no answer)',
  'call.ended':                 'Call ended',
});
function _humaniseEventType(type) {
  if (!type) return '';
  return _EVENT_TYPE_LABELS[type] || type.replace(/[._]/g, ' ');
}

function _humaniseState(state) {
  switch (state) {
    case 'connected': return 'Connected';
    case 'ringing':   return 'Ringing';
    case 'invited':   return 'Invited';
    case 'ended':     return 'Ended';
    case 'missed':    return 'Missed';
    case 'declined':  return 'Declined';
    case 'failed':    return 'Failed';
    default: return state || 'Unknown';
  }
}

function _humaniseDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const now = new Date();
    const sameDay = d.getFullYear() === now.getFullYear()
      && d.getMonth() === now.getMonth()
      && d.getDate() === now.getDate();
    if (sameDay) {
      return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
    }
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const sameYesterday = d.getFullYear() === yesterday.getFullYear()
      && d.getMonth() === yesterday.getMonth()
      && d.getDate() === yesterday.getDate();
    if (sameYesterday) return 'Yesterday';
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (_) {
    return iso;
  }
}

function _humaniseDateTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch (_) {
    return iso;
  }
}

function _humaniseTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', second: '2-digit' });
  } catch (_) {
    return iso;
  }
}

function _humaniseDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}
