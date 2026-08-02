/**
 * cast-control.js — phone/desktop remote.
 *
 * Polls /api/cast/receivers for live TVs + /api/cast/library/home for
 * the user's content (same endpoint cast-home uses), renders rails
 * and a receiver picker, and routes taps through /api/cast/send so
 * the TV mounts the selected surface_kind + URL.
 *
 * Active-cast state lives in localStorage so a phone refresh /
 * tab-switch / second-device picks up the same now-playing card.
 */

import {
  startProducer,
  stopProducer,
  stopAllProducers,
  hasConnectedGamepad,
} from './controller-producer.js';

const LS_CONTROLLER_NUDGE_PREFIX = 'augmentum.cast.control.ctrlNudge.';

function _nudgeDismissed(fileId) {
  if (!fileId) return false;
  try {
    return !!localStorage.getItem(LS_CONTROLLER_NUDGE_PREFIX + fileId);
  } catch { return false; }
}
function _markNudgeDismissed(fileId) {
  if (!fileId) return;
  try {
    localStorage.setItem(LS_CONTROLLER_NUDGE_PREFIX + fileId, '1');
  } catch {}
}

// Status pill plumbing. Cast-control renders a small "🎮 …" chip in
// the now-playing card while a game cast is active; the producer
// pushes state changes through this callback so the UI follows.
let _controllerState = '';
function _onControllerStatus(payload) {
  _controllerState = payload?.state || '';
  // Re-render now-playing so the pill picks up the new state. Cheap
  // — renderNowPlaying() is signature-cached.
  try { renderNowPlaying(); } catch {}
}

// Stop the input producer on tab close so the server doesn't leak
// the phone-side attachment. The container side detects the read-loop
// EOF and detaches on its own.
window.addEventListener('beforeunload', () => {
  try { stopAllProducers(); } catch {}
});

const RECEIVER_POLL_MS = 8000;
const LIBRARY_REFRESH_MS = 5 * 60 * 1000;
const PLAYBACK_POLL_MS = 3000;
const LS_LAST_RECEIVER = 'augmentum.cast.control.lastReceiver';
const LS_NOW_PLAYING = 'augmentum.cast.control.nowPlaying';
const LS_FOLLOW_MODE = 'augmentum.cast.control.followMode';
// Per-file A/V offset (ms). Mirrors floating-video.js's
// _SYNC_OFFSET_STORAGE_PREFIX so users get familiar behavior: a
// per-title sticky setting, default 0 for clean sources. Sent to
// cast-video.js as patch.audio_offset_ms.
const LS_SYNC_OFFSET_PREFIX = 'augmentum.cast.control.sync-offset.';
const SYNC_OFFSET_MAX_MS = 2000;

function loadSyncOffsetMs(fileId) {
  if (!fileId) return 0;
  try {
    const raw = localStorage.getItem(LS_SYNC_OFFSET_PREFIX + fileId);
    if (!raw) return 0;
    return Math.max(0, Math.min(SYNC_OFFSET_MAX_MS, Number(raw) || 0));
  } catch { return 0; }
}

function saveSyncOffsetMs(fileId, ms) {
  if (!fileId) return;
  try {
    const clamped = Math.max(0, Math.min(SYNC_OFFSET_MAX_MS, Math.round(Number(ms) || 0)));
    if (clamped > 0) {
      localStorage.setItem(LS_SYNC_OFFSET_PREFIX + fileId, String(clamped));
    } else {
      localStorage.removeItem(LS_SYNC_OFFSET_PREFIX + fileId);
    }
  } catch { /* quota / private mode — best-effort */ }
}

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  receivers: [],
  // Durable trusted-receiver rows (connected + offline). Sourced from
  // /api/cast/trusted-receivers; powers the Wake-on-LAN section of the
  // picker so the user can boot a TV that isn't currently connected.
  // Keyed by the persistent ``id`` (tr_...), distinct from the live
  // ``registration_id`` on state.receivers.
  trustedReceivers: [],
  selectedReceiverId: localStorage.getItem(LS_LAST_RECEIVER) || '',
  library: null,
  receiverPollTimer: null,
  libraryRefreshTimer: null,
  playbackPollTimer: null,
  // Last server-reported position. Updated on each playback-status
  // poll; consulted by tap-to-seek (so the user's tap lands at the
  // right ratio of the actual duration, not whatever we last rendered).
  playbackStatus: { current_time_s: 0, duration_s: 0, progress_pct: 0 },
  // Track list fetched from /api/media/details when a video casts.
  // Each entry: {index, label, language, is_default}. Cleared on stop.
  subtitleTracks: [],
  activeSubtitleIdx: -1,
  // Comic-reader local prefs. Sent to the TV as patches when the user
  // taps the relevant sheet control. Memory-only — survives a sheet
  // open/close cycle, resets on phone refresh. The TV reader has its
  // own auto-detect that picks sensible defaults on first mount, so
  // these only diverge when the user explicitly overrides.
  comicPrefs: {
    autoplayMs: 0,        // paged / dual — ms between page flips
    autoplayPxPerSec: 0,  // webtoon — continuous scroll velocity
    autoplayPaused: false,
    mode: 'auto',         // auto | single | dual | webtoon
    fit: 'smart',         // smart | width | height | native
    direction: 'ltr',     // ltr | rtl
    borderCrop: true,
    padDismissed: false,  // user closed the scroll pad this session
  },
  // Follow-on-TV — when true, every browsing transition (open section,
  // open series, close sheet → home) is mirrored onto the selected
  // receiver's cast-home idle surface so a group can see what the
  // phone-holder is picking. Persisted across reloads.
  followMode: localStorage.getItem(LS_FOLLOW_MODE) === '1',
};


/* ── Follow on TV ───────────────────────────────────────────────── */

async function sendNavToTv(payload, { force = false } = {}) {
  // Two callers:
  //   1. Follow-mode mirror — only fires when state.followMode is on.
  //   2. Trackpad / TV remote (scroll + tap) — fires regardless of
  //      followMode because the trackpad IS the user's deliberate
  //      remote-control action. Pass force=true for these.
  if (!force && !state.followMode) return;
  if (!state.selectedReceiverId) return;
  try {
    const r = await fetch('/api/cast/send/nav', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        receiver_id: state.selectedReceiverId,
        payload,
      }),
    });
    if (!r.ok) {
      console.warn('[follow] nav rejected', r.status, await r.text().catch(() => ''));
    } else {
      console.log('[follow] nav sent', payload);
    }
  } catch (err) {
    console.warn('[follow] nav network error', err);
  }
}

function setFollowMode(on) {
  state.followMode = !!on;
  localStorage.setItem(LS_FOLLOW_MODE, state.followMode ? '1' : '0');
  const btn = $('[data-cc-follow-toggle]');
  if (btn) btn.setAttribute('aria-pressed', state.followMode ? 'true' : 'false');
  if (state.followMode) {
    // On enable: immediately push the current view so the TV catches
    // up. If a sheet is open, mirror that; otherwise mirror home.
    if (_librarySheet.mode === 'series' && _librarySheet.seriesFileId) {
      sendNavToTv({ view: 'series', file_id: _librarySheet.seriesFileId });
    } else if (_librarySheet.mode === 'section' && _librarySheet.slug) {
      sendNavToTv({ view: 'section', slug: _librarySheet.slug });
    } else {
      sendNavToTv({ view: 'home' });
    }
  } else {
    // On disable: leave the TV on whatever it's showing. No yank-back —
    // the user toggling off probably wants the room's attention to
    // drift away from the TV, not snap to a different screen.
  }
  // Clearing × on a previously-dismissed pad — a fresh follow toggle
  // ON should reveal the trackpad, not leave it stuck hidden from
  // the last session's dismiss.
  if (state.followMode) state.comicPrefs.padDismissed = false;
  // Re-evaluate trackpad visibility immediately so the toggle change
  // reflects without waiting for the next poll. Call directly rather
  // than relying on renderNowPlaying's side effect — that function
  // early-returns on its dedup hash when np is empty, which would
  // skip the pad refresh on the no-media path (× on the remote-mode
  // pad). Derive isComic the same way renderNowPlaying does so a
  // follow-toggle while a comic is casting doesn't accidentally
  // close the comic pad.
  const np = loadNowPlaying();
  const sk = np?.item?.play?.surface_kind || '';
  const k = np?.item?.kind || '';
  const isComic = sk === 'html.generic' && k === 'comic';
  updateScrollPadVisibility(isComic);
  renderNowPlaying();
}

function updateFollowToggleVisibility() {
  // Only show the toggle once a receiver is selected — without one,
  // there's nothing to follow.
  const btn = $('[data-cc-follow-toggle]');
  if (!btn) return;
  btn.classList.toggle('hidden', !state.selectedReceiverId);
}

function wireFollowToggle() {
  const btn = $('[data-cc-follow-toggle]');
  if (!btn) return;
  btn.setAttribute('aria-pressed', state.followMode ? 'true' : 'false');
  btn.addEventListener('click', () => setFollowMode(!state.followMode));
}


/* ── TV preferences sheet ─────────────────────────────────────────
 *
 * Per-receiver display preferences (rail visibility, backdrop cycle,
 * subtitle default, follow-mode allowed). Reads /api/cast/trusted-
 * receivers/{trusted_id}/prefs on open, PUTs back the whole bag on
 * each change. Auto-save — the toggle IS the commit. Failures fall
 * back to local state so the user can keep tweaking; the next change
 * retries via a fresh PUT. */

// Rail catalog lives on the server (augmentum/cast/rail_catalog.py).
// Fetched on first prefs-sheet open and cached for the page lifetime
// so the prefs UI can't drift from the data layer — adding/removing a
// rail server-side automatically updates the toggles here.
let _railCatalog = null;  // [{slug, title, hint}, ...] once loaded
let _railCatalogPromise = null;

async function _loadRailCatalog() {
  if (_railCatalog) return _railCatalog;
  if (_railCatalogPromise) return _railCatalogPromise;
  _railCatalogPromise = (async () => {
    try {
      const r = await fetch('/api/cast/rails/catalog', {
        credentials: 'same-origin', cache: 'no-store',
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      _railCatalog = Array.isArray(body.rails) ? body.rails : [];
      return _railCatalog;
    } catch (err) {
      console.warn('[cast-control] rail catalog fetch failed', err);
      _railCatalog = [];
      return _railCatalog;
    } finally {
      _railCatalogPromise = null;
    }
  })();
  return _railCatalogPromise;
}

const BEHAVIOUR_META = [
  ['backdrop_cycle', 'Backdrop cycle',
   'Slowly cycle background art on the idle screen'],
  ['subtitle_default', 'Subtitles on by default',
   'Newly-cast videos start with subtitles on'],
  ['follow_mode_allowed', 'Allow Follow on TV',
   'Let the controller mirror its view onto this TV'],
];

const _prefsState = {
  trusted_id: '',
  receiver_label: '',
  prefs: null,
  saving: false,
};

function _trustedIdForSelected() {
  const r = state.receivers.find(
    (rcv) => rcv.registration_id === state.selectedReceiverId,
  );
  return r?.trusted_id || '';
}

function updatePrefsButtonVisibility() {
  const btn = $('[data-cc-prefs-button]');
  if (!btn) return;
  btn.classList.toggle(
    'hidden',
    !state.selectedReceiverId || !_trustedIdForSelected(),
  );
}

function openPrefsSheet() {
  const trusted_id = _trustedIdForSelected();
  if (!trusted_id) {
    toast('This TV doesn\'t have a persistent trust entry yet.');
    return;
  }
  const receiver = state.receivers.find(
    (r) => r.registration_id === state.selectedReceiverId,
  );
  _prefsState.trusted_id = trusted_id;
  _prefsState.receiver_label = receiver?.label || 'TV';
  _prefsState.prefs = null;

  const sheet = $('[data-cc-prefs-sheet]');
  const title = $('[data-cc-prefs-sheet-title]');
  if (title) title.textContent = _prefsState.receiver_label;
  if (sheet) sheet.classList.remove('hidden');

  const body = $('[data-cc-prefs-sheet-body]');
  if (body) body.innerHTML = '<div class="prefs-sheet-loading">Loading…</div>';
  _fetchPrefs();
}

function closePrefsSheet() {
  const sheet = $('[data-cc-prefs-sheet]');
  if (sheet) sheet.classList.add('hidden');
  _prefsState.trusted_id = '';
  _prefsState.prefs = null;
}

async function _fetchPrefs() {
  const id = _prefsState.trusted_id;
  if (!id) return;
  // Fetch the per-receiver prefs and the global rail catalog in
  // parallel. Catalog lives behind a module-level cache, so this is
  // a single round-trip on first open and a no-op on subsequent ones.
  try {
    const [prefsResp] = await Promise.all([
      fetch(
        `/api/cast/trusted-receivers/${encodeURIComponent(id)}/prefs`,
        { credentials: 'same-origin', cache: 'no-store' },
      ),
      _loadRailCatalog(),
    ]);
    if (!prefsResp.ok) {
      const body = $('[data-cc-prefs-sheet-body]');
      if (body) body.innerHTML = `<div class="prefs-sheet-loading">Couldn't load (HTTP ${prefsResp.status}).</div>`;
      return;
    }
    const body = await prefsResp.json();
    _prefsState.prefs = body.prefs || {};
    _renderPrefsBody();
  } catch {
    const body = $('[data-cc-prefs-sheet-body]');
    if (body) body.innerHTML = `<div class="prefs-sheet-loading">Network error.</div>`;
  }
}

function _renderPrefsBody() {
  const body = $('[data-cc-prefs-sheet-body]');
  if (!body || !_prefsState.prefs) return;
  const p = _prefsState.prefs;
  const railsVis = (p.rails_visible && typeof p.rails_visible === 'object')
    ? p.rails_visible : {};

  // Catalog populated by _loadRailCatalog before _renderPrefsBody is
  // called from _fetchPrefs. Defensive fallback to [] keeps the sheet
  // renderable in the (unlikely) event the catalog fetch failed —
  // user just sees no rail toggles rather than a broken sheet.
  const catalog = _railCatalog || [];
  const railRows = catalog.map(({ slug, title, hint }) => {
    const checked = railsVis[slug] !== false;
    return `
      <label class="prefs-toggle" data-cc-prefs-rail="${escapeHtml(slug)}">
        <span class="prefs-toggle-label">
          <span class="prefs-toggle-name">${escapeHtml(title)}</span>
          <span class="prefs-toggle-hint">${escapeHtml(hint)}</span>
        </span>
        <input type="checkbox" ${checked ? 'checked' : ''}>
      </label>
    `;
  }).join('');

  const behaviourRows = BEHAVIOUR_META.map(([key, name, hint]) => {
    const checked = p[key] !== false;
    return `
      <label class="prefs-toggle" data-cc-prefs-toggle="${escapeHtml(key)}">
        <span class="prefs-toggle-label">
          <span class="prefs-toggle-name">${escapeHtml(name)}</span>
          <span class="prefs-toggle-hint">${escapeHtml(hint)}</span>
        </span>
        <input type="checkbox" ${checked ? 'checked' : ''}>
      </label>
    `;
  }).join('');

  body.innerHTML = `
    <section class="prefs-sheet-section">
      <div class="prefs-sheet-section-label">Idle home rails</div>
      ${railRows}
    </section>
    <section class="prefs-sheet-section">
      <div class="prefs-sheet-section-label">Behaviour</div>
      ${behaviourRows}
    </section>
  `;

  body.querySelectorAll('[data-cc-prefs-rail] input').forEach((cb) => {
    cb.addEventListener('change', () => {
      const slug = cb.closest('[data-cc-prefs-rail]').dataset.ccPrefsRail;
      if (!_prefsState.prefs.rails_visible) _prefsState.prefs.rails_visible = {};
      _prefsState.prefs.rails_visible[slug] = cb.checked;
      _savePrefs();
    });
  });
  body.querySelectorAll('[data-cc-prefs-toggle] input').forEach((cb) => {
    cb.addEventListener('change', () => {
      const key = cb.closest('[data-cc-prefs-toggle]').dataset.ccPrefsToggle;
      _prefsState.prefs[key] = cb.checked;
      _savePrefs();
    });
  });
}

async function _savePrefs() {
  const id = _prefsState.trusted_id;
  if (!id || !_prefsState.prefs) return;
  if (_prefsState.saving) return;
  _prefsState.saving = true;
  try {
    const r = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(id)}/prefs`,
      {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ prefs: _prefsState.prefs }),
      },
    );
    if (!r.ok) {
      toast(`Couldn't save preferences (HTTP ${r.status}).`);
      return;
    }
    const body = await r.json();
    _prefsState.prefs = body.prefs || _prefsState.prefs;
    // Prefs changes only affect what the receiver (TV) renders, not
    // the controller's library view — no library re-fetch here.
    // The receiver itself re-fetches via the augmentum.prefs
    // postMessage path when its prefs bag updates.
  } catch {
    toast('Network error saving preferences.');
  } finally {
    _prefsState.saving = false;
  }
}

function wirePrefsButton() {
  const btn = $('[data-cc-prefs-button]');
  if (btn) btn.addEventListener('click', openPrefsSheet);
  $('[data-cc-prefs-sheet-close]')?.addEventListener('click', closePrefsSheet);
  $('[data-cc-prefs-sheet-scrim]')?.addEventListener('click', closePrefsSheet);
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    const sheet = $('[data-cc-prefs-sheet]');
    if (sheet && !sheet.classList.contains('hidden')) closePrefsSheet();
  });
}


/* ── Util ───────────────────────────────────────────────────────── */

function escapeHtml(s) {
  return String(s ?? '').replace(/[<>&"']/g, (c) => ({
    '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function toast(text, opts = {}) {
  const el = $('[data-cc-toast]');
  if (!el) return;
  el.textContent = text;
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), opts.duration || 2500);
}


/* ── Receivers ──────────────────────────────────────────────────── */

async function refreshReceivers() {
  try {
    const r = await fetch('/api/cast/receivers', { credentials: 'same-origin' });
    if (!r.ok) return;
    const body = await r.json();
    state.receivers = body.receivers || [];
  } catch {
    return;
  }

  // Pull durable rows in parallel so the picker's "Wake a sleeping TV"
  // section reflects every paired device, not just the live ones.
  // Failures are silent — wake is a nice-to-have, the live list is the
  // primary cast target and shouldn't be blocked on this.
  try {
    const r2 = await fetch('/api/cast/trusted-receivers', { credentials: 'same-origin' });
    if (r2.ok) {
      const b2 = await r2.json();
      state.trustedReceivers = b2.receivers || [];
    }
  } catch { /* nice-to-have only */ }

  // If our selected receiver is gone, fall back to the first connected
  // one (silent — picker dropdown reflects the new state).
  if (state.selectedReceiverId
      && !state.receivers.find((r) => r.registration_id === state.selectedReceiverId)) {
    state.selectedReceiverId = '';
  }
  if (!state.selectedReceiverId && state.receivers.length > 0) {
    state.selectedReceiverId = state.receivers[0].registration_id;
    localStorage.setItem(LS_LAST_RECEIVER, state.selectedReceiverId);
  }
  renderPicker();
  // Ghost cleanup: if the persisted now-playing references a receiver
  // we no longer see (re-pair killed it, container restart, etc.),
  // the saved surface_id is dead — no patch will reach it. Clear the
  // mini-player so taps on it don't silently fail.
  validateNowPlayingAgainstReceivers();
}

function validateNowPlayingAgainstReceivers() {
  const np = loadNowPlaying();
  if (!np || !np.receiver_id) return;
  // Quick check: the saved registration_id needs to STILL match a
  // currently-connected receiver. If the pair was redone, the new
  // session gets a fresh registration_id even when the trusted_id
  // is the same — so old surfaces always die with the old reg.
  const alive = state.receivers.some(
    (r) => r.registration_id === np.receiver_id,
  );
  if (!alive) {
    clearNowPlaying();
    renderNowPlaying();
  }
}

// Hash of (selectedId + receivers) used to skip the picker rebuild
// when nothing about the choices changed. Without this every 8-second
// poll re-innerHTMLs the dropdown list — visible as a flicker if the
// user has it open, and as wasted layout work either way.
let _pickerSig = '';

function renderPicker() {
  const buttonLabel = $('[data-cc-picker-label]');
  const buttonDot = $('[data-cc-picker-dot]');
  const empty = $('[data-cc-picker-empty]');
  const itemsHost = $('[data-cc-picker-items]');
  if (!buttonLabel || !itemsHost) return;

  const sig = JSON.stringify({
    sel: state.selectedReceiverId,
    rs: state.receivers.map((r) => ({
      id: r.registration_id,
      lbl: r.label,
      plat: r?.info?.platform || '',
      tr: r.trusted_id || '',
    })),
    // Include the offline trusted list — without this, the "Sleeping"
    // section never gains/loses rows between polls.
    tr: (state.trustedReceivers || []).map((t) => ({
      id: t.id,
      lbl: t.label,
      wol: !!t.wol_ready,
      rev: !!t.revoked,
    })),
  });
  if (sig === _pickerSig) {
    // Choices identical to last paint — the visibility hook for the
    // follow-toggle still needs to run since selectedReceiverId could
    // have flipped to/from empty between paints, but it's cheap and
    // idempotent.
    updateFollowToggleVisibility();
    updatePrefsButtonVisibility();
    return;
  }
  _pickerSig = sig;

  const selected = state.receivers.find((r) => r.registration_id === state.selectedReceiverId);
  if (selected) {
    buttonLabel.textContent = selected.label || 'TV';
    buttonDot?.classList.add('live');
  } else if (state.receivers.length > 0) {
    buttonLabel.textContent = 'Pick a TV';
    buttonDot?.classList.remove('live');
  } else {
    buttonLabel.textContent = 'No TVs paired';
    buttonDot?.classList.remove('live');
  }

  updateFollowToggleVisibility();
  updatePrefsButtonVisibility();

  // Connected trusted_ids — used to mark live rows in the durable
  // list, so we can render offline-only TVs without duplicating the
  // ones already visible above.
  const liveTrustedIds = new Set(
    state.receivers.map((r) => r.trusted_id).filter(Boolean),
  );
  const offlineTrusted = (state.trustedReceivers || []).filter(
    (tr) => !tr.revoked && !liveTrustedIds.has(tr.id),
  );

  const haveAnyTvs = state.receivers.length > 0 || offlineTrusted.length > 0;

  if (!haveAnyTvs) {
    empty?.removeAttribute('hidden');
    itemsHost.innerHTML = '';
  } else {
    empty?.setAttribute('hidden', '');
    const liveHtml = state.receivers.map((r) => {
      const platform = (r.info && r.info.platform) || '';
      const active = r.registration_id === state.selectedReceiverId;
      return `
        <div class="receiver-picker-item ${active ? 'active' : ''}" data-cc-receiver="${escapeHtml(r.registration_id)}" role="option">
          <span class="receiver-dot live"></span>
          <div class="receiver-picker-item-meta">
            <span>${escapeHtml(r.label || r.registration_id)}</span>
            ${platform ? `<span class="receiver-picker-item-sub">${escapeHtml(platform)}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');

    // Offline section. Wake button when MAC is set; otherwise a "Set
    // MAC" button that opens an inline editor. Listing offline TVs
    // is the entire point of the WoL UX — without this, the only way
    // to wake a sleeping TV would be to have a separate "saved
    // devices" surface.
    const offlineHtml = offlineTrusted.length === 0 ? '' : `
      <div class="receiver-picker-section">Sleeping</div>
      ${offlineTrusted.map((tr) => {
        const action = tr.wol_ready
          ? `<button class="receiver-picker-action" data-cc-wake="${escapeHtml(tr.id)}" title="Wake this TV">Wake</button>`
          : `<button class="receiver-picker-action ghost" data-cc-edit-mac="${escapeHtml(tr.id)}" title="Set MAC to enable wake">Set MAC</button>`;
        return `
          <div class="receiver-picker-item offline" role="option">
            <span class="receiver-dot"></span>
            <div class="receiver-picker-item-meta">
              <span>${escapeHtml(tr.label || tr.id)}</span>
              <span class="receiver-picker-item-sub">${escapeHtml(tr.platform || 'offline')}</span>
            </div>
            ${action}
          </div>
        `;
      }).join('')}
    `;

    itemsHost.innerHTML = liveHtml + offlineHtml;
    itemsHost.querySelectorAll('[data-cc-receiver]').forEach((el) => {
      el.addEventListener('click', () => {
        state.selectedReceiverId = el.dataset.ccReceiver;
        localStorage.setItem(LS_LAST_RECEIVER, state.selectedReceiverId);
        closePicker();
        renderPicker();
        // Receiver selection toggles the "TV remote" eligibility for
        // the scroll pad — re-evaluate visibility immediately so the
        // pad appears the first time a user picks a TV.
        renderNowPlaying();
        // The library view is receiver-agnostic — selecting a TV
        // only changes the cast target, not the browsable content.
        // No library re-fetch needed.
      });
    });
    itemsHost.querySelectorAll('[data-cc-wake]').forEach((el) => {
      el.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        await sendWake(el.dataset.ccWake);
      });
    });
    itemsHost.querySelectorAll('[data-cc-edit-mac]').forEach((el) => {
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        openMacEditor(el.dataset.ccEditMac);
      });
    });
  }
}


/* ── Wake-on-LAN ────────────────────────────────────────────────── */
//
// Wake offline TVs from the picker dropdown. The server sends a magic
// packet to the subnet broadcast derived from the last LAN IP the
// receiver self-reported (browser receivers won't have one — the user
// supplies the MAC via the "Set MAC" editor and optionally a broadcast
// override).

async function sendWake(trustedId) {
  if (!trustedId) return;
  try {
    const r = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(trustedId)}/wake`,
      { method: 'POST', credentials: 'same-origin' },
    );
    const body = await r.json().catch(() => ({}));
    if (!r.ok || !body.ok) {
      const reason = body.reason || body.detail || `HTTP ${r.status}`;
      toast(`Wake failed: ${reason}`);
      // 422 = no MAC set; surface the editor so the fix is one tap away.
      if (r.status === 422) openMacEditor(trustedId);
      return;
    }
    toast(`Magic packet sent (${body.broadcast || ''})`, { duration: 2000 });
  } catch (err) {
    toast('Wake failed: network error');
  }
}

function openMacEditor(trustedId) {
  const tr = (state.trustedReceivers || []).find((x) => x.id === trustedId);
  if (!tr) return;
  // Lightweight prompt-based editor — full settings UI lives elsewhere.
  // Two prompts in sequence: MAC (required) and broadcast override
  // (optional, '' clears).
  const macAnswer = window.prompt(
    `MAC address for "${tr.label || tr.id}"\nFormat: aa:bb:cc:dd:ee:ff (other separators OK)`,
    tr.mac_address || '',
  );
  if (macAnswer === null) return;  // cancelled
  const bcastAnswer = window.prompt(
    `Broadcast address override (optional)\nLeave blank to auto-derive from the TV's last LAN IP.`,
    tr.wol_broadcast_override || '',
  );
  if (bcastAnswer === null) return;  // cancelled
  saveWolFields(trustedId, macAnswer.trim(), bcastAnswer.trim());
}

async function saveWolFields(trustedId, macAddress, wolBroadcastOverride) {
  try {
    const r = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(trustedId)}`,
      {
        method: 'PATCH',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mac_address: macAddress,
          wol_broadcast_override: wolBroadcastOverride,
        }),
      },
    );
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      toast(body.detail || `Save failed (HTTP ${r.status})`);
      return;
    }
    const body = await r.json().catch(() => ({}));
    // Patch the in-memory list so the dropdown rerenders with Wake
    // showing immediately, without waiting for the next poll cycle.
    if (body.receiver) {
      const idx = state.trustedReceivers.findIndex((x) => x.id === trustedId);
      if (idx >= 0) state.trustedReceivers[idx] = body.receiver;
    }
    _pickerSig = '';  // force re-render
    renderPicker();
    if (macAddress) toast('Saved. Tap Wake to test.', { duration: 2000 });
  } catch {
    toast('Save failed: network error');
  }
}

function openPicker() { $('[data-cc-picker-menu]')?.classList.remove('hidden'); }
function closePicker() { $('[data-cc-picker-menu]')?.classList.add('hidden'); }

function wirePicker() {
  $('[data-cc-picker-button]')?.addEventListener('click', () => {
    const menu = $('[data-cc-picker-menu]');
    if (!menu) return;
    menu.classList.toggle('hidden');
  });
  document.addEventListener('click', (ev) => {
    if (!ev.target.closest('[data-cc-picker]')) closePicker();
  });

  // Pair-by-code form. The receiver shows a short pair_code on the TV
  // screen; the user types it here and POST /api/cast/pair/approve
  // claims it for this account. On success the receiver's own poll
  // (./cast-receiver/) picks up the ws_token and goes live within
  // ~1s; we kick refreshReceivers() so the picker reflects it without
  // waiting for the next poll tick.
  const form = $('[data-cc-pair-form]');
  if (form) {
    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const input = $('[data-cc-pair-input]');
      const submit = $('[data-cc-pair-submit]');
      const msg = $('[data-cc-pair-msg]');
      if (!input || !submit || !msg) return;
      const code = (input.value || '').trim().toUpperCase();
      if (!code) {
        msg.textContent = 'Enter the code shown on the TV.';
        msg.dataset.state = 'error';
        return;
      }
      submit.disabled = true;
      submit.textContent = 'Pairing…';
      msg.textContent = '';
      msg.dataset.state = '';
      let resp;
      try {
        resp = await fetch(`/api/cast/pair/approve/${encodeURIComponent(code)}`, {
          method: 'POST',
          credentials: 'same-origin',
        });
      } catch (err) {
        msg.textContent = `Network error: ${err.message || err}`;
        msg.dataset.state = 'error';
        submit.disabled = false;
        submit.textContent = 'Pair';
        return;
      }
      if (resp.status === 409) {
        msg.textContent = 'That code expired or was already claimed.';
        msg.dataset.state = 'error';
      } else if (!resp.ok) {
        msg.textContent = `Pairing failed (${resp.status}).`;
        msg.dataset.state = 'error';
      } else {
        msg.textContent = 'Paired. The TV will connect in a moment.';
        msg.dataset.state = 'success';
        input.value = '';
        refreshReceivers();
      }
      submit.disabled = false;
      submit.textContent = 'Pair';
    });
  }
}


/* ── Library ────────────────────────────────────────────────────── */

function renderTile(item) {
  // Aspect ratio is per-entity-kind, not per-kind. Movies + series
  // ship 2:3 poster art (just like audiobooks + comics), so they
  // should match those rails visually. Episodes + music videos are
  // the landscape (16:9) outliers — their thumbs are scene-frame
  // shaped. Treating every kind=video as landscape was what stretched
  // movie posters and made the home view feel hodge-podge.
  const ek = (item.entity_kind || '').toLowerCase();
  const landscape = item.kind === 'video'
    && (ek === 'episode' || ek === 'music_video');
  const cover = item.cover_url || '';
  const sub = item.subtitle || (item.source ? item.source : '');
  const pct = Math.max(0, Math.min(100, item.progress_pct || 0));
  // Game tiles get a "Controller required" chip so the user knows
  // they need a paired gamepad before tapping. The launch path
  // double-checks via hasConnectedGamepad() and surfaces a nudge
  // when nothing's paired — the chip is the cheap first signal.
  const isGame = item.kind === 'game' || (item.play && item.play.action === 'cast_game');
  return `
    <article class="tile" data-cc-item="${escapeHtml(item.file_id)}">
      <div class="tile-art ${landscape ? 'landscape' : ''}">
        ${cover
          ? `<img src="${escapeHtml(cover)}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'tile-art-placeholder\\'>${escapeHtml(item.kind || 'media')}</div>'">`
          : `<div class="tile-art-placeholder">${escapeHtml(item.kind || 'media')}</div>`
        }
        ${pct > 0 ? `<div class="tile-progress"><div class="tile-progress-fill" style="width:${pct}%"></div></div>` : ''}
        ${isGame ? `<div class="tile-chip" title="Pair a controller before tapping">🎮 Controller</div>` : ''}
      </div>
      <div class="tile-title">${escapeHtml(item.title)}</div>
      ${sub ? `<div class="tile-sub">${escapeHtml(sub)}</div>` : ''}
    </article>
  `;
}

function renderRail(section) {
  if (!section.items || section.items.length === 0) return '';
  // "See all →" appears on every section EXCEPT ``resume`` (continue
  // is already a short focused list — a paginated view would be
  // empty after the first page). All other rails accept paging
  // through /api/cast/library/section/<slug>.
  const showSeeAll = section.id !== 'resume';
  return `
    <section class="rail" data-cc-rail="${escapeHtml(section.id)}">
      <header class="rail-head">
        <h2 class="rail-title">${escapeHtml(section.title)}</h2>
        ${showSeeAll
          ? `<button class="rail-see-all" data-cc-see-all="${escapeHtml(section.id)}"
                     data-cc-see-all-title="${escapeHtml(section.title)}">
               See all →
             </button>`
          : ''}
      </header>
      <div class="rail-strip">
        ${section.items.map(renderTile).join('')}
      </div>
    </section>
  `;
}

function renderLibraryShortcuts(libraries) {
  if (!libraries || libraries.length === 0) return '';
  const chips = libraries.map((lib) => `
    <span class="library-chip">
      <span>${escapeHtml(lib.name || lib.id)}</span>
      <span class="library-chip-provider">${escapeHtml(lib.provider || '')}</span>
    </span>
  `).join('');
  return `
    <section class="rail" data-cc-rail="libraries">
      <header class="rail-head">
        <h2 class="rail-title">Your libraries</h2>
      </header>
      <div class="libraries">${chips}</div>
    </section>
  `;
}

// Signature of the last library payload that produced DOM. We rebuild
// only when the set of visible tiles or their progress moves — the
// 5-minute timer firing on an unchanged catalog should be a no-op, not
// a full wipe of every <img> in every rail.
let _librarySig = '';

function _signatureForLibrary(sections, libraries) {
  return JSON.stringify({
    s: (sections || []).map((s) => ({
      id: s.id,
      // Items keyed by file_id + progress + title. Progress is in the
      // mix because the resume rail's tile-progress fill width updates
      // even when the same tile is at the same position.
      it: (s.items || []).map((i) => [
        i.file_id || '',
        Math.round(i.progress_pct || 0),
        i.title || '',
      ]),
    })),
    l: (libraries || []).map((l) => l.id || l.name || ''),
  });
}

async function refreshLibrary() {
  const host = $('[data-cc-library]');
  if (!host) return;

  // Controller always shows the full library — the rails_visible
  // prefs are about what each physical TV surfaces, not what the
  // user can browse from the navigator. The user might want to cast
  // Comics to TV A even if Comics is hidden on TV B, so the
  // controller stays unfiltered regardless of which receiver is
  // selected. Per-receiver filtering happens on the cast-home (TV)
  // side via its own trusted_id.
  try {
    const r = await fetch(`/api/cast/library/home`, {
      credentials: 'same-origin', cache: 'no-store',
    });
    if (!r.ok) {
      host.innerHTML = `<div class="loading">Couldn't load library (HTTP ${r.status}).</div>`;
      _librarySig = '';
      return;
    }
    state.library = await r.json();
  } catch (err) {
    host.innerHTML = `<div class="loading">Network error loading library.</div>`;
    _librarySig = '';
    return;
  }

  const sections = state.library.sections || [];
  const libraries = state.library.libraries || [];

  if (sections.length === 0 && libraries.length === 0) {
    host.innerHTML = `
      <div class="loading">
        No castable content yet. Connect a media server in Augmentum
        settings → Media to see your libraries here.
      </div>
    `;
    _librarySig = '';
    return;
  }

  const sig = _signatureForLibrary(sections, libraries);
  if (sig === _librarySig) return;  // visible content unchanged
  _librarySig = sig;

  host.innerHTML = [
    ...sections.map(renderRail),
    renderLibraryShortcuts(libraries),
  ].filter(Boolean).join('');

  // Wire tile taps. We look up the item from state.library by file_id
  // to access the play block on send.
  host.querySelectorAll('[data-cc-item]').forEach((el) => {
    el.addEventListener('click', () => {
      const fid = el.dataset.ccItem;
      const item = findItem(fid);
      if (item) castItem(item);
    });
  });
  // "See all →" per-rail. Opens the paginated section browser.
  host.querySelectorAll('[data-cc-see-all]').forEach((el) => {
    el.addEventListener('click', () => {
      const slug = el.dataset.ccSeeAll;
      const title = el.dataset.ccSeeAllTitle;
      if (slug) openSectionBrowser(slug, title);
    });
  });
}


/* ── Library drill-in sheet ─────────────────────────────────────── */

const SECTION_PAGE_LIMIT = 60;
const _librarySheet = {
  mode: '',           // 'section' or 'series'
  slug: '',           // section slug if mode==='section'
  seriesFileId: '',   // series file_id if mode==='series'
  offset: 0,
  loading: false,
  // Comic-section sort/filter (only meaningful for slug==='comics').
  // Defaults match what the server applies when no params land — keep
  // them in sync so the initial chip highlight matches the actual
  // returned order.
  sort: 'name_asc',
  filter: 'all',
  // Watch/listen-status chip (movies/shows/audiobooks). 'all' = no
  // filter, otherwise matches server media_status predicates.
  status: 'all',
  // Recently-added kind chip. 'all' keeps the spec's exclude list;
  // 'movie' / 'series' / 'audio' / 'comic' narrows the rail to one
  // media kind.
  kindFilter: 'all',
  // In-sheet title search. Section mode → forwarded to the server as
  // ?q=; series mode → applied client-side over the cached drill-in
  // payload (episodes/chapters were already fetched in one shot).
  q: '',
  // Debounce handle for the section refetch — cleared on input,
  // closeLibrarySheet, and openSectionBrowser.
  qDebounce: null,
  // Cached drill-in payload so series-mode search can filter without
  // re-hitting the server. Set in openSeriesDrillIn.
  seriesCache: null,
};

// Per-section controls config. One entry per slug; the controls strip
// is rebuilt against this on every openSectionBrowser. Keeping the
// shape declarative means adding a new rail = one map entry, not five
// scattered conditionals.
//
//   sort         — array of {value, label} for the dropdown, or null
//                  to hide the sort wrap entirely (e.g. recently_added
//                  is sorted by added-time by definition)
//   sortDefault  — initial sort key set on open
//   chipGroup    — 'status' | 'kind' | 'comic' | 'gallery' | null
//   statusLabels — optional per-slug overrides for the status chips
//                  (audiobooks reads better with "Unstarted")
const SECTION_CONTROLS = {
  recently_added: {
    sort: null,
    sortDefault: '',
    chipGroup: 'kind',
  },
  audiobooks: {
    sort: [
      { value: 'newest', label: 'Recently added' },
      { value: 'name',   label: 'A → Z' },
      { value: 'author', label: 'Author' },
    ],
    sortDefault: 'newest',
    chipGroup: 'status',
    statusLabels: { not_started: 'Unstarted' },
  },
  movies: {
    sort: [
      { value: 'newest',    label: 'Recently added' },
      { value: 'name',      label: 'A → Z' },
      { value: 'year_desc', label: 'Year ↓' },
      { value: 'year_asc',  label: 'Year ↑' },
    ],
    sortDefault: 'newest',
    chipGroup: 'status',
  },
  shows: {
    sort: [
      { value: 'progress', label: 'Recently active' },
      { value: 'name',     label: 'A → Z' },
    ],
    sortDefault: 'progress',
    chipGroup: 'status',
  },
  music_videos: {
    sort: [
      { value: 'newest', label: 'Recently added' },
      { value: 'name',   label: 'A → Z' },
    ],
    sortDefault: 'newest',
    chipGroup: null,
  },
  comics: {
    sort: [
      { value: 'name_asc',  label: 'A → Z' },
      { value: 'name_desc', label: 'Z → A' },
      { value: 'active',    label: 'Recently active' },
      { value: 'chapters',  label: 'Most chapters' },
    ],
    sortDefault: 'name_asc',
    chipGroup: 'comic',
  },
  gallery: {
    sort: null,
    sortDefault: '',
    chipGroup: null,
  },
};

// Sections where the search box is meaningful. Resume is a short
// curated list (no see-all anyway), and gallery rows rarely carry
// title text that's worth FTS-matching.
const SEARCHABLE_SECTION_SLUGS = new Set([
  'recently_added',
  'audiobooks',
  'movies',
  'shows',
  'music_videos',
  'comics',
]);

function openLibrarySheet() {
  const sheet = $('[data-cc-lib-sheet]');
  if (sheet) sheet.classList.remove('hidden');
}

function closeLibrarySheet() {
  const sheet = $('[data-cc-lib-sheet]');
  if (sheet) sheet.classList.add('hidden');
  _librarySheet.mode = '';
  _librarySheet.slug = '';
  _librarySheet.seriesFileId = '';
  _librarySheet.offset = 0;
  _librarySheet.status = 'all';
  _librarySheet.kindFilter = 'all';
  _librarySheet.q = '';
  _librarySheet.seriesCache = null;
  if (_librarySheet.qDebounce) {
    clearTimeout(_librarySheet.qDebounce);
    _librarySheet.qDebounce = null;
  }
  const searchInput = $('[data-cc-lib-sheet-search]');
  if (searchInput) searchInput.value = '';
  // Back to TV idle home — phone returned to the rails view.
  sendNavToTv({ view: 'home' });
}

function setLibrarySheetHeader(overline, title) {
  const ovEl = $('[data-cc-lib-sheet-overline]');
  const titleEl = $('[data-cc-lib-sheet-title]');
  if (ovEl) ovEl.textContent = overline || '';
  if (titleEl) titleEl.textContent = title || '';
}

function setLibrarySheetBody(html) {
  const body = $('[data-cc-lib-sheet-body]');
  if (body) body.innerHTML = html;
}

function setLibrarySheetFoot({ visible, label, disabled }) {
  const foot = $('[data-cc-lib-sheet-foot]');
  const btn = $('[data-cc-lib-sheet-loadmore]');
  if (!foot || !btn) return;
  foot.classList.toggle('hidden', !visible);
  btn.textContent = label || 'Load more';
  btn.disabled = !!disabled;
}

function _renderSheetGrid(items) {
  if (!items.length) {
    if (_librarySheet.q) {
      return `<div class="library-sheet-loading">No titles match “${escapeHtml(_librarySheet.q)}”.</div>`;
    }
    return `<div class="library-sheet-loading">Nothing here yet.</div>`;
  }
  return `<div class="library-sheet-grid">${items.map(renderTile).join('')}</div>`;
}

function _wireSheetGridTiles(items) {
  const body = $('[data-cc-lib-sheet-body]');
  if (!body) return;
  body.querySelectorAll('[data-cc-item]').forEach((el) => {
    el.addEventListener('click', () => {
      const fid = el.dataset.ccItem;
      const item = items.find((it) => it.file_id === fid);
      if (item) castItem(item);
    });
  });
}

async function openSectionBrowser(slug, title) {
  _librarySheet.mode = 'section';
  _librarySheet.slug = slug;
  _librarySheet.offset = 0;
  sendNavToTv({ view: 'section', slug });
  // Reset all controls to defaults on each open so a previous
  // session's selection doesn't leak across sections. Sort default
  // comes from SECTION_CONTROLS so each rail lands on the order
  // that matches its content best (movies → recently added, shows →
  // recently active, comics → A→Z, etc).
  const cfg = SECTION_CONTROLS[slug] || {};
  _librarySheet.sort = cfg.sortDefault || '';
  _librarySheet.filter = 'all';
  _librarySheet.status = 'all';
  _librarySheet.kindFilter = 'all';
  _librarySheet.q = '';
  _librarySheet.seriesCache = null;
  if (_librarySheet.qDebounce) {
    clearTimeout(_librarySheet.qDebounce);
    _librarySheet.qDebounce = null;
  }
  const searchInput = $('[data-cc-lib-sheet-search]');
  if (searchInput) searchInput.value = '';
  _updateLibrarySheetControlsVisibility();
  _renderLibrarySheetControlsState();
  setLibrarySheetHeader('library', title || slug);
  setLibrarySheetBody(`<div class="library-sheet-loading">Loading…</div>`);
  setLibrarySheetFoot({ visible: false });
  openLibrarySheet();
  await _fetchSectionPage(/* append */ false);
}

function _updateLibrarySheetControlsVisibility() {
  const controls = $('[data-cc-lib-sheet-controls]');
  if (!controls) return;
  // Controls strip layout is driven by SECTION_CONTROLS for sections
  // and by hard-coded "search-only" rules for series drill-ins. We
  // resolve which sub-controls to show + populate the sort dropdown
  // here so the chips always match the slug the user just opened.
  const isSection = _librarySheet.mode === 'section';
  const isSeries  = _librarySheet.mode === 'series';
  const slug = _librarySheet.slug;
  const cfg  = isSection ? (SECTION_CONTROLS[slug] || null) : null;
  const searchable = (isSection && SEARCHABLE_SECTION_SLUGS.has(slug)) || isSeries;
  const showSort   = !!(cfg && cfg.sort && cfg.sort.length);
  const chipGroup  = cfg ? cfg.chipGroup : null;
  const anyControl = searchable || showSort || !!chipGroup;
  controls.classList.toggle('hidden', !anyControl);

  const searchWrap  = $('[data-cc-lib-sheet-search-wrap]');
  const sortWrap    = $('[data-cc-lib-sheet-sort-wrap]');
  const comicChips  = $('[data-cc-lib-sheet-filter-chips]');
  const statusChips = $('[data-cc-lib-sheet-status-chips]');
  const kindChips   = $('[data-cc-lib-sheet-kind-chips]');
  if (searchWrap)  searchWrap.hidden  = !searchable;
  if (sortWrap)    sortWrap.hidden    = !showSort;
  if (comicChips)  comicChips.hidden  = chipGroup !== 'comic';
  if (statusChips) statusChips.hidden = chipGroup !== 'status';
  if (kindChips)   kindChips.hidden   = chipGroup !== 'kind';

  // Repopulate the sort dropdown for the active section. Done here
  // (not in openSectionBrowser) so adding a new SECTION_CONTROLS entry
  // automatically picks up the right options without extra wiring.
  if (showSort) {
    const sortSel = $('[data-cc-lib-sheet-sort]');
    if (sortSel) {
      sortSel.innerHTML = cfg.sort.map(
        (o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`,
      ).join('');
    }
  }

  // Status chip labels — audiobooks override "Unwatched" to "Unstarted"
  // because nobody calls finishing-an-audiobook "watching". Reset to
  // defaults on any non-overriding slug so a previous open doesn't
  // leak its labels.
  if (statusChips) {
    const overrides = (cfg && cfg.statusLabels) || {};
    const DEFAULT_STATUS_LABELS = {
      all: 'All',
      not_started: 'Unwatched',
      in_progress: 'In progress',
      finished: 'Finished',
    };
    for (const btn of statusChips.querySelectorAll('.library-sheet-chip')) {
      const key = btn.dataset.ccLibSheetStatus;
      btn.textContent = overrides[key] || DEFAULT_STATUS_LABELS[key] || btn.textContent;
    }
  }
}

function _renderLibrarySheetControlsState() {
  const sortSel = $('[data-cc-lib-sheet-sort]');
  if (sortSel) sortSel.value = _librarySheet.sort;
  for (const c of $$('[data-cc-lib-sheet-filter-chips] .library-sheet-chip')) {
    c.classList.toggle('is-active', c.dataset.ccLibSheetFilter === _librarySheet.filter);
  }
  for (const c of $$('[data-cc-lib-sheet-status-chips] .library-sheet-chip')) {
    c.classList.toggle('is-active', c.dataset.ccLibSheetStatus === _librarySheet.status);
  }
  for (const c of $$('[data-cc-lib-sheet-kind-chips] .library-sheet-chip')) {
    c.classList.toggle('is-active', c.dataset.ccLibSheetKind === _librarySheet.kindFilter);
  }
}

async function _fetchSectionPage(append) {
  if (_librarySheet.loading) return;
  _librarySheet.loading = true;
  try {
    const qs = new URLSearchParams({
      offset: String(_librarySheet.offset),
      limit: String(SECTION_PAGE_LIMIT),
    });
    // Sort — only ships if the active section's SECTION_CONTROLS has
    // a sort list. Server falls back to the section's spec default
    // when omitted, so unsorted sections never get a misleading key.
    const slugCfg = SECTION_CONTROLS[_librarySheet.slug];
    if (slugCfg && slugCfg.sort && _librarySheet.sort) {
      qs.set('sort', _librarySheet.sort);
    }
    // Comics' separate ``filter`` semantic — series reading status
    // (ongoing/completed/unknown), not watch progress.
    if (_librarySheet.slug === 'comics') {
      qs.set('filter', _librarySheet.filter);
    }
    // Watch/listen status (movies/shows/audiobooks). "all" is the
    // default + means "no filter" — omit it from the URL so the
    // request matches the unfiltered baseline byte-for-byte.
    if (_librarySheet.status && _librarySheet.status !== 'all') {
      qs.set('status', _librarySheet.status);
    }
    // Recently-added kind chip. Same all-means-omit convention.
    if (_librarySheet.kindFilter && _librarySheet.kindFilter !== 'all') {
      qs.set('kind', _librarySheet.kindFilter);
    }
    // Title search — gallery + resume ignore it server-side, but the
    // UI hides the input for those slugs so it never ships from here.
    if (_librarySheet.q) qs.set('q', _librarySheet.q);
    const r = await fetch(
      `/api/cast/library/section/${encodeURIComponent(_librarySheet.slug)}?${qs}`,
      { credentials: 'same-origin', cache: 'no-store' },
    );
    if (!r.ok) {
      setLibrarySheetBody(`<div class="library-sheet-loading">Couldn't load (HTTP ${r.status}).</div>`);
      return;
    }
    const body = await r.json();
    const items = body.items || [];
    // Gallery's Private category may come back gated when the
    // cast_gallery_show_private setting is off — render the gating
    // hint instead of an empty grid so the user understands why
    // there's nothing to see.
    if (body.private_gated && !append) {
      setLibrarySheetBody(`
        <div class="library-sheet-loading">
          Private images are hidden. Enable
          <code>cast_gallery_show_private</code> in settings to
          surface them here.
        </div>`);
      setLibrarySheetFoot({ visible: false });
      return;
    }
    if (!append) {
      setLibrarySheetBody(_renderSheetGrid(items));
      _wireSheetGridTiles(items);
    } else {
      // Append to existing grid in place
      const grid = document.querySelector('[data-cc-lib-sheet-body] .library-sheet-grid');
      if (grid) {
        grid.insertAdjacentHTML('beforeend', items.map(renderTile).join(''));
        _wireSheetGridTiles(items);
      }
    }
    _librarySheet.offset += items.length;
    setLibrarySheetFoot({
      visible: !!body.has_more,
      label: 'Load more',
      disabled: false,
    });
  } finally {
    _librarySheet.loading = false;
  }
}

async function openSeriesDrillIn(seriesItem) {
  _librarySheet.mode = 'series';
  _librarySheet.seriesFileId = seriesItem.file_id;
  _librarySheet.q = '';
  _librarySheet.seriesCache = null;
  if (_librarySheet.qDebounce) {
    clearTimeout(_librarySheet.qDebounce);
    _librarySheet.qDebounce = null;
  }
  const searchInput = $('[data-cc-lib-sheet-search]');
  if (searchInput) searchInput.value = '';
  _updateLibrarySheetControlsVisibility();
  // Comic series tile carries entity_kind='comic_series'; video series
  // are entity_kind='series'. The right endpoint and render shape
  // differ — comics are chapter-numbered ungrouped, videos are
  // season-grouped. Both reuse the same overlay chrome.
  const isComic = seriesItem.kind === 'comic'
    || seriesItem.entity_kind === 'comic_series';
  // Include a kind hint so cast-home routes to the right drill-in
  // endpoint (chapters/ for comics, episodes/ for video). The cast-
  // home side falls back to trying both if no hint lands, but the
  // explicit hint skips the wasted 404.
  sendNavToTv({
    view: 'series',
    file_id: seriesItem.file_id,
    kind: seriesItem.kind || '',
    entity_kind: seriesItem.entity_kind || '',
  });
  setLibrarySheetHeader('series', seriesItem.title);
  setLibrarySheetBody(`<div class="library-sheet-loading">${isComic ? 'Loading chapters' : 'Loading episodes'}…</div>`);
  setLibrarySheetFoot({ visible: false });
  openLibrarySheet();

  const endpoint = isComic
    ? `/api/cast/library/chapters/${encodeURIComponent(seriesItem.file_id)}`
    : `/api/cast/library/episodes/${encodeURIComponent(seriesItem.file_id)}`;

  try {
    const r = await fetch(endpoint, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) {
      const label = isComic ? 'chapters' : 'episodes';
      setLibrarySheetBody(`<div class="library-sheet-loading">Couldn't load ${label} (HTTP ${r.status}).</div>`);
      return;
    }
    const body = await r.json();
    if (isComic) {
      const chapters = body.chapters || [];
      _librarySheet.seriesCache = { kind: 'comic', chapters };
      if (!chapters.length) {
        setLibrarySheetBody(`<div class="library-sheet-loading">No chapters indexed yet.</div>`);
        return;
      }
      _renderSeriesFromCache();
    } else {
      const seasons = body.seasons || [];
      _librarySheet.seriesCache = { kind: 'video', seasons };
      if (!seasons.length) {
        setLibrarySheetBody(`<div class="library-sheet-loading">No episodes available yet.</div>`);
        return;
      }
      _renderSeriesFromCache();
    }
  } catch (err) {
    setLibrarySheetBody(`<div class="library-sheet-loading">Network error.</div>`);
  }
}

// Renders the active series-drill-in payload, applying the search
// filter when one is set. Search runs client-side because the drill-in
// already pulled every chapter/episode in one shot — round-tripping
// would just add latency.
function _renderSeriesFromCache() {
  const cache = _librarySheet.seriesCache;
  if (!cache) return;
  const needle = (_librarySheet.q || '').toLowerCase();
  if (cache.kind === 'comic') {
    const all = cache.chapters || [];
    const matched = needle
      ? all.filter((c) => (c.title || '').toLowerCase().includes(needle))
      : all;
    if (!matched.length) {
      setLibrarySheetBody(
        `<div class="library-sheet-loading">No chapters match “${escapeHtml(_librarySheet.q)}”.</div>`,
      );
      return;
    }
    setLibrarySheetBody(matched.map(_renderChapterRow).join(''));
    _wireChapterRows(matched);
  } else {
    const all = cache.seasons || [];
    const filteredSeasons = needle
      ? all
          .map((s) => ({
            ...s,
            episodes: (s.episodes || []).filter(
              (e) => (e.title || '').toLowerCase().includes(needle),
            ),
          }))
          .filter((s) => (s.episodes || []).length > 0)
      : all;
    if (!filteredSeasons.length) {
      setLibrarySheetBody(
        `<div class="library-sheet-loading">No episodes match “${escapeHtml(_librarySheet.q)}”.</div>`,
      );
      return;
    }
    setLibrarySheetBody(filteredSeasons.map(_renderSeasonGroup).join(''));
    _wireSeasonEpisodes(filteredSeasons);
  }
}

/** One row per chapter in the comic drill-in. Chapter number prefix
 *  is taken from server-side metadata so half-chapters (e.g. "12.5")
 *  render correctly. Progress bar surfaces last-read position. */
function _renderChapterRow(ch) {
  const cover = ch.cover_url || '';
  const pct = Math.max(0, Math.min(100, ch.progress_pct || 0));
  const num = ch.chapter_number || 0;
  // Strip the trailing ".0" on whole chapters so "12.0" reads as "12".
  const numLabel = num
    ? `Ch. ${Number.isInteger(num) ? num : num.toFixed(1)}`
    : '';
  return `
    <div class="library-sheet-episode" data-cc-chapter-id="${escapeHtml(ch.file_id)}">
      <div class="library-sheet-episode-art">
        ${cover
          ? `<img src="${escapeHtml(cover)}" alt="" loading="lazy" onerror="this.style.display='none'">`
          : ''}
      </div>
      <div class="library-sheet-episode-meta">
        ${numLabel ? `<div class="library-sheet-episode-num">${escapeHtml(numLabel)}</div>` : ''}
        <div class="library-sheet-episode-title">${escapeHtml(ch.title)}</div>
        ${pct > 0
          ? `<div class="library-sheet-episode-progress"><div class="library-sheet-episode-progress-fill" style="width:${pct}%"></div></div>`
          : ''}
      </div>
    </div>
  `;
}

function _wireChapterRows(chapters) {
  const body = $('[data-cc-lib-sheet-body]');
  if (!body) return;
  body.querySelectorAll('[data-cc-chapter-id]').forEach((el) => {
    el.addEventListener('click', () => {
      const fid = el.dataset.ccChapterId;
      const ch = chapters.find((c) => c.file_id === fid);
      if (ch) {
        closeLibrarySheet();
        castItem(ch);
      }
    });
  });
}

function _renderSeasonGroup(season) {
  return `
    <section class="library-sheet-season">
      <header class="library-sheet-season-head">${escapeHtml(season.label || `Season ${season.season_number}`)}</header>
      ${(season.episodes || []).map(_renderEpisodeRow).join('')}
    </section>
  `;
}

function _renderEpisodeRow(ep) {
  const cover = ep.cover_url || '';
  const pct = Math.max(0, Math.min(100, ep.progress_pct || 0));
  const sn = ep.season_number || 0;
  const en = ep.episode_number || 0;
  const label = sn && en ? `S${sn}E${String(en).padStart(2, '0')}` : '';
  return `
    <div class="library-sheet-episode" data-cc-ep-id="${escapeHtml(ep.file_id)}">
      <div class="library-sheet-episode-art">
        ${cover
          ? `<img src="${escapeHtml(cover)}" alt="" loading="lazy" onerror="this.style.display='none'">`
          : ''}
      </div>
      <div class="library-sheet-episode-meta">
        ${label ? `<div class="library-sheet-episode-num">${escapeHtml(label)}</div>` : ''}
        <div class="library-sheet-episode-title">${escapeHtml(ep.title)}</div>
        ${pct > 0
          ? `<div class="library-sheet-episode-progress"><div class="library-sheet-episode-progress-fill" style="width:${pct}%"></div></div>`
          : ''}
      </div>
    </div>
  `;
}

function _wireSeasonEpisodes(seasons) {
  const flat = seasons.flatMap((s) => s.episodes || []);
  const body = $('[data-cc-lib-sheet-body]');
  if (!body) return;
  body.querySelectorAll('[data-cc-ep-id]').forEach((el) => {
    el.addEventListener('click', () => {
      const fid = el.dataset.ccEpId;
      const ep = flat.find((e) => e.file_id === fid);
      if (ep) {
        closeLibrarySheet();
        castItem(ep);
      }
    });
  });
}

function wireLibrarySheet() {
  $('[data-cc-lib-sheet-close]')?.addEventListener('click', closeLibrarySheet);
  $('[data-cc-lib-sheet-scrim]')?.addEventListener('click', closeLibrarySheet);
  $('[data-cc-lib-sheet-loadmore]')?.addEventListener('click', () => {
    if (_librarySheet.mode === 'section') {
      setLibrarySheetFoot({ visible: true, label: 'Loading…', disabled: true });
      _fetchSectionPage(/* append */ true);
    }
  });
  // Sort dropdown — change resets offset and re-fetches the whole list
  // so the user lands at the top of the new order rather than at a
  // random offset into it.
  $('[data-cc-lib-sheet-sort]')?.addEventListener('change', (ev) => {
    _librarySheet.sort = ev.target.value || 'name_asc';
    _librarySheet.offset = 0;
    setLibrarySheetBody(`<div class="library-sheet-loading">Sorting…</div>`);
    setLibrarySheetFoot({ visible: false });
    _fetchSectionPage(/* append */ false);
  });
  // Filter chips — same reset-then-refetch pattern.
  $$('[data-cc-lib-sheet-filter-chips] .library-sheet-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      _librarySheet.filter = chip.dataset.ccLibSheetFilter || 'all';
      _librarySheet.offset = 0;
      _renderLibrarySheetControlsState();
      setLibrarySheetBody(`<div class="library-sheet-loading">Filtering…</div>`);
      setLibrarySheetFoot({ visible: false });
      _fetchSectionPage(/* append */ false);
    });
  });
  // Status chips — watch/listen progress (movies/shows/audiobooks).
  // Same reset-and-refetch as the comic filter chips.
  $$('[data-cc-lib-sheet-status-chips] .library-sheet-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      _librarySheet.status = chip.dataset.ccLibSheetStatus || 'all';
      _librarySheet.offset = 0;
      _renderLibrarySheetControlsState();
      setLibrarySheetBody(`<div class="library-sheet-loading">Filtering…</div>`);
      setLibrarySheetFoot({ visible: false });
      _fetchSectionPage(/* append */ false);
    });
  });
  // Recently-added kind chips — Movies / Shows / Audiobooks / Comics.
  // "All" sends no ``?kind=``; specific chips narrow the spec's kind
  // filter via the server's _KIND_CHIP_TRANSLATIONS map.
  $$('[data-cc-lib-sheet-kind-chips] .library-sheet-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      _librarySheet.kindFilter = chip.dataset.ccLibSheetKind || 'all';
      _librarySheet.offset = 0;
      _renderLibrarySheetControlsState();
      setLibrarySheetBody(`<div class="library-sheet-loading">Filtering…</div>`);
      setLibrarySheetFoot({ visible: false });
      _fetchSectionPage(/* append */ false);
    });
  });
  // Title search — debounced so server load stays sane while the user
  // is still typing. Section mode round-trips through the server (FTS5
  // covers libraries that don't fit in memory); series mode filters
  // the already-fetched seasons/chapters cache locally.
  const searchInput = $('[data-cc-lib-sheet-search]');
  searchInput?.addEventListener('input', (ev) => {
    const raw = (ev.target.value || '').trim();
    _librarySheet.q = raw;
    if (_librarySheet.qDebounce) {
      clearTimeout(_librarySheet.qDebounce);
    }
    if (_librarySheet.mode === 'series') {
      // Local filter — no network round-trip, instant feedback.
      _renderSeriesFromCache();
      return;
    }
    _librarySheet.qDebounce = setTimeout(() => {
      _librarySheet.qDebounce = null;
      if (_librarySheet.mode !== 'section') return;
      _librarySheet.offset = 0;
      setLibrarySheetBody(`<div class="library-sheet-loading">Searching…</div>`);
      setLibrarySheetFoot({ visible: false });
      _fetchSectionPage(/* append */ false);
    }, 220);
  });
  // Pressing Escape inside the search input clears the query first;
  // a second Escape (or one when the input is empty) closes the sheet.
  searchInput?.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if (searchInput.value) {
      ev.stopPropagation();
      searchInput.value = '';
      searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
  // Escape key closes the sheet for keyboard users.
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    const sheet = $('[data-cc-lib-sheet]');
    if (sheet && !sheet.classList.contains('hidden')) closeLibrarySheet();
  });
}

function findItem(fileId) {
  if (!state.library) return null;
  for (const section of state.library.sections || []) {
    for (const item of section.items || []) {
      if (item.file_id === fileId) return item;
    }
  }
  return null;
}


/* ── Casting ────────────────────────────────────────────────────── */

async function fetchSubtitleTracksFor(fileId) {
  // /api/media/details returns the parsed media_sources[].subtitle_tracks
  // shape used by the main UI's player. Reusing it here means a future
  // refactor only has one place to change.
  state.subtitleTracks = [];
  state.activeSubtitleIdx = -1;
  if (!fileId) return;
  try {
    const r = await fetch(`/api/media/details/${encodeURIComponent(fileId)}`, {
      credentials: 'same-origin',
    });
    if (!r.ok) return;
    const body = await r.json();
    // /api/media/details wraps media_sources inside ``body.playback``.
    // Reading the top level (which was the previous implementation)
    // always got an empty list, so the picker was always empty even
    // when the show actually had subs available.
    const playback = body.playback || body || {};
    const sources = playback.media_sources || body.media_sources || [];
    const selectedId = playback.selected_media_source_id
      || body.selected_media_source_id
      || '';
    const src = sources.find((s) => s.id === selectedId) || sources[0] || {};
    state.subtitleTracks = (src.subtitle_tracks || [])
      .filter((t) => t && typeof t.index === 'number' && t.index >= 0)
      .map((t) => ({
        index: t.index,
        label: t.label || t.language || `Track ${t.index}`,
        language: t.language || '',
        language_code: t.language_code || '',
        is_default: !!t.is_default,
      }));
    renderSubtitlePicker();
  } catch (err) {
    console.warn('[cast-control] details fetch threw', err);
  }
}

function renderSubtitlePicker() {
  const host = $('[data-cc-np-cc-list]');
  const section = document.querySelector('.np-sheet-section-cc');
  if (!host || !section) return;
  const np = loadNowPlaying();
  const k = np?.item?.kind || '';
  // Subtitle picker only meaningful for video.
  if (k !== 'video') {
    section.hidden = true;
    return;
  }
  // Always render an "Off" item even if no tracks (so user knows
  // we tried). Server returns 401 on /api/media/subtitle/x w/o auth;
  // an empty list usually means "this title has no subs."
  const items = [
    { index: -1, label: 'Off', sub: '' },
    ...state.subtitleTracks.map((t) => ({
      index: t.index,
      label: t.label,
      sub: t.language_code || t.language,
    })),
  ];
  host.innerHTML = items.map((it) => {
    const active = state.activeSubtitleIdx === it.index;
    return `
      <button class="np-cc-item ${active ? 'is-active' : ''}" data-cc-cc-idx="${it.index}">
        <span>${escapeHtml(it.label)}</span>
        ${it.sub ? `<span class="np-cc-item-sub">${escapeHtml(it.sub)}</span>` : ''}
      </button>
    `;
  }).join('');
  host.querySelectorAll('[data-cc-cc-idx]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.ccCcIdx);
      state.activeSubtitleIdx = idx;
      // Update active class in place — cheaper than re-render.
      host.querySelectorAll('[data-cc-cc-idx]').forEach((b) => {
        b.classList.toggle('is-active', Number(b.dataset.ccCcIdx) === idx);
      });
      sendPatch({ subtitle_idx: idx });
    });
  });
  section.hidden = state.subtitleTracks.length === 0 && state.activeSubtitleIdx === -1;
  if (state.subtitleTracks.length > 0) section.hidden = false;
}


async function castItem(item) {
  const play = item.play || {};
  // Series tiles aren't directly castable — they need the user to
  // pick an episode. Open the season-grouped drill-in instead.
  // Comes BEFORE the receiver check because browsing doesn't need
  // a TV picked yet (user can pick after seeing what they want).
  if (play.action === 'browse_series') {
    openSeriesDrillIn(item);
    return;
  }
  if (!state.selectedReceiverId) {
    toast('Pick a TV first');
    openPicker();
    return;
  }
  // Games take a dedicated endpoint that composes AGSP launch + cast
  // surface in one call. Server-side keeps the title_runs telemetry +
  // BIOS validation honest; the controller just shows progress/errors.
  if (play.action === 'cast_game') {
    // Pre-play nudge: if no gamepad is paired the user will be
    // greeted by a black screen with no inputs working. Surface a
    // one-time per-game hint so the failure mode is obvious and
    // user-actionable. Dismissed entries are remembered per game.
    if (!hasConnectedGamepad() && !_nudgeDismissed(item.file_id)) {
      const ok = window.confirm(
        'No controller detected.\n\n'
        + 'Pair a controller in your device’s Bluetooth or USB '
        + 'settings, then come back here and tap again.\n\n'
        + 'OK to dismiss this reminder for this game.'
      );
      if (ok) _markNudgeDismissed(item.file_id);
      return;
    }
    toast(`Launching "${item.title}"…`);
    try {
      const r = await fetch(`/api/cast/games/${encodeURIComponent(item.file_id)}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          receiver_id: state.selectedReceiverId,
          slot: 'main',
        }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        // 409 → user-actionable (missing BIOS etc.). Surface the
        // server's hint verbatim. Other failures get a generic line.
        if (r.status === 409) {
          toast(body.detail || 'Cannot launch this game');
        } else {
          toast(`Launch failed: ${body.detail || r.status}`);
        }
        return;
      }
      const body = await r.json();
      // Thread max_players from the game-start response onto the
      // item so renderNowPlaying can decide whether to expose the
      // "+ Players" couch-coop button. Single-player profiles keep
      // it hidden.
      const itemWithCoop = {
        ...item,
        max_players: Number(body.max_players) || 1,
      };
      saveNowPlaying({
        surface_id: body.surface_id,
        receiver_id: body.receiver_id,
        item: itemWithCoop,
        stream_session_id: body.stream_session_id || '',
      });
      renderNowPlaying();
      pollPlayback();
      setTimeout(() => refreshLibrary().catch(() => {}), 1500);
      toast(`Playing on ${currentReceiverLabel()}`);
      // Open the input WS so the phone's gamepad drives the streamed
      // emulator. The producer self-detaches when the WS closes (cast
      // ended, network drop). hasConnectedGamepad() was checked above;
      // even if it becomes false later, the producer dials anyway and
      // sends zero-state frames so the server's session is alive when
      // the user re-pairs a controller mid-game.
      if (body.stream_session_id) {
        startProducer(body.stream_session_id, _onControllerStatus);
      }
    } catch (err) {
      toast(`Network error: ${err.message || err}`);
    }
    return;
  }
  if (!play.surface_kind || !play.surface_url) {
    toast(`Don't know how to cast this ${item.kind || 'item'} yet`);
    return;
  }
  toast(`Sending "${item.title}"…`);
  try {
    const r = await fetch('/api/cast/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        receiver_id: state.selectedReceiverId,
        surface_kind: play.surface_kind,
        surface_url: play.surface_url,
        slot: 'main',
      }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      toast(`Cast failed: ${body.detail || r.status}`);
      return;
    }
    const body = await r.json();
    saveNowPlaying({
      surface_id: body.surface_id,
      receiver_id: body.receiver_id,
      item,
    });
    renderNowPlaying();
    // Fetch the full subtitle track list for video casts so the
    // sheet's CC section has something to show beyond "Off".
    if (item.kind === 'video') {
      fetchSubtitleTracksFor(item.file_id);
    } else {
      state.subtitleTracks = [];
      state.activeSubtitleIdx = -1;
    }
    pollPlayback();
    // Refresh the library after a tiny grace window so the Continue
    // rail reorders to reflect the just-played item. Default poll
    // interval is 5 minutes; without this nudge the user never sees
    // the rail change in a single session.
    setTimeout(() => refreshLibrary().catch(() => {}), 1500);
    // Replay the saved A/V offset for this title so reopen-then-play
    // applies the user's previous fix automatically. Skipped at 0 to
    // avoid building the receiver's Web Audio graph for clean sources.
    const savedSync = loadSyncOffsetMs(item.file_id || '');
    if (savedSync > 0) {
      // Small delay so the receiver iframe has time to mount + reach
      // HAVE_CURRENT_DATA — the graph init guard rejects earlier.
      setTimeout(() => {
        sendPatch({ audio_offset_ms: savedSync }).catch(() => {});
      }, 1500);
    }
    toast(`Casting on ${currentReceiverLabel()}`);
  } catch (err) {
    toast(`Network error: ${err.message || err}`);
  }
}

function currentReceiverLabel() {
  const r = state.receivers.find((x) => x.registration_id === state.selectedReceiverId);
  return r?.label || 'TV';
}


/* ── Now-playing ────────────────────────────────────────────────── */

function saveNowPlaying(payload) {
  try {
    localStorage.setItem(LS_NOW_PLAYING, JSON.stringify(payload));
  } catch { /* localStorage unavailable */ }
}

function loadNowPlaying() {
  try {
    const raw = localStorage.getItem(LS_NOW_PLAYING);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function clearNowPlaying() {
  // Stop any active controller producer keyed to this now-playing.
  // Reading the row first lets us reach the stream_session_id even
  // though we're about to drop it. Safe to call when producer wasn't
  // started (stopProducer is a no-op for unknown ids).
  try {
    const cur = loadNowPlaying();
    if (cur?.stream_session_id) {
      stopProducer(cur.stream_session_id, 'now_playing_cleared');
    }
  } catch {}
  try { localStorage.removeItem(LS_NOW_PLAYING); } catch {}
}

// Signature of the last now-playing paint. Cleared (set to "") any time
// we want to force a redraw — e.g. when the user toggles a comic pref
// inside the sheet, the prefs hash changes naturally; but for transport
// taps that only update _paused locally, callers should reset this if
// they need the icon to swap.
let _npSig = '';

function renderNowPlaying() {
  const np = loadNowPlaying();
  const card = $('[data-cc-now-playing]');
  if (!card) return;
  if (!np || !np.item) {
    if (_npSig === 'EMPTY') return;
    _npSig = 'EMPTY';
    card.classList.add('hidden');
    updateScrollPadVisibility(false);
    return;
  }
  // Cheap hash of every field this function actually paints. If it
  // matches the last paint we skip — saves text-content writes,
  // background-image assignments, and the per-section hidden flips
  // that otherwise run on every transport tap and progress poll.
  const sk = (np.item.play?.surface_kind) || '';
  const k = np.item.kind || '';
  const receiver = state.receivers.find((r) => r.registration_id === np.receiver_id)
    || state.receivers.find((r) => r.registration_id === state.selectedReceiverId);
  const platform = (receiver?.info?.platform || '').toLowerCase();
  const sig = JSON.stringify({
    rcv: np.receiver_id,
    fid: np.item.file_id || '',
    title: np.item.title || '',
    sub: np.item.subtitle || np.item.source || '',
    cover: np.item.cover_url || '',
    sk, k,
    mp: np.item.max_players || 1,
    plat: platform,
    cp: state.comicPrefs,
    paused: _paused,
  });
  if (sig === _npSig) return;
  _npSig = sig;

  card.classList.remove('hidden');
  $('[data-cc-np-title]').textContent = np.item.title || '';
  $('[data-cc-np-sub]').textContent = np.item.subtitle || np.item.source || '';
  const art = $('[data-cc-np-art]');
  const desired = np.item.cover_url ? `url("${np.item.cover_url}")` : '';
  // Only reassign when the URL string actually differs — even the
  // cached-image identical-URL assignment triggers a paint.
  if (art && art.style.backgroundImage !== desired) {
    art.style.backgroundImage = desired;
  }

  // Wire transport set per kind. ``sk`` / ``k`` were already computed
  // at the top of the function for the signature hash; reuse them
  // here rather than redeclaring (SyntaxError otherwise).
  const back = $('[data-cc-np-back]');
  const toggle = $('[data-cc-np-toggle]');
  const fwd = $('[data-cc-np-fwd]');
  const more = $('[data-cc-np-more]');

  const isAudio = sk === 'html.generic' && k === 'audio';
  const isVideo = sk === 'html.generic' && k === 'video';
  const isComic = sk === 'html.generic' && k === 'comic';

  // Hide all, then re-enable what's relevant.
  for (const el of [back, toggle, fwd, more]) { el.hidden = true; }

  if (isAudio || isVideo) {
    back.hidden = false; back.textContent = '⏪';  back.dataset.action = 'skip_back';
    toggle.hidden = false;
    fwd.hidden = false; fwd.textContent = '⏩';   fwd.dataset.action = 'skip_fwd';
    more.hidden = false;
  } else if (isComic) {
    back.hidden = false; back.textContent = '◀';  back.dataset.action = 'page_prev';
    fwd.hidden = false; fwd.textContent = '▶';   fwd.dataset.action = 'page_next';
    // Always show ⏯ for comics — it doubles as "start autoplay" when
    // off (kicks the mode-appropriate default speed) and pause/resume
    // when on. State is derived from comicPrefs in the current mode.
    toggle.hidden = false;
    toggle.dataset.action = 'comic_autoplay_toggle';
    const active = _isComicAutoplayActive();
    toggle.textContent = !active
      ? '▶'
      : (state.comicPrefs.autoplayPaused ? '▶' : '⏸');
    more.hidden = false;
  }

  // Sheet sections — audio/video keep CC + speed + jump; comic gets
  // its own set (autoplay, mode, fit, direction, crop, jump).
  document.querySelector('.np-sheet-section-cc').hidden = !isVideo;
  document.querySelector('.np-sheet-section-speed').hidden = !(isAudio || isVideo);
  document.querySelector('.np-sheet-section-jump').hidden = !(isAudio || isVideo);
  document.querySelector('.np-sheet-section-volume').hidden = !(isAudio || isVideo);
  // A/V sync — video only (no video = no drift to compensate).
  // Populate the slider from the per-file saved offset so re-opening
  // the now-playing card on a known-troublesome title shows the user's
  // previous setting without them having to redrag.
  const syncSection = document.querySelector('.np-sheet-section-sync');
  if (syncSection) {
    syncSection.hidden = !isVideo;
    if (!syncSection.hidden) {
      const savedMs = loadSyncOffsetMs(np.item.file_id || '');
      const slider = $('[data-cc-np-sync]');
      const readout = $('[data-cc-np-sync-readout]');
      if (slider && Number(slider.value) !== savedMs) {
        slider.value = String(savedMs);
      }
      if (readout) readout.textContent = `${savedMs} ms`;
    }
  }
  // TV master volume: only meaningful on the Android TV APK receiver
  // (the only context with the AugmentumTV JS bridge). For audio /
  // video / comic casts alike — TV master adjusts the speaker even
  // when no audio is playing, e.g. while watching a silent comic.
  const tvVolumeSection = document.querySelector('.np-sheet-section-tv-volume');
  if (tvVolumeSection) {
    // ``platform`` was computed at the top of the function for the
    // signature hash — reuse it instead of recomputing.
    tvVolumeSection.hidden = platform !== 'android-tv';
  }
  document.querySelector('.np-sheet-section-comic-autoplay').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-jump').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-mode').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-fit').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-dir').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-crop').hidden = !isComic;
  document.querySelector('.np-sheet-section-comic-pad').hidden = !isComic;
  if (isComic) {
    renderComicSheetState();
  }
  // Couch co-op: surface the "+ Players" button when the casting game
  // supports >1 player. Hidden otherwise so single-player titles don't
  // pretend to offer invites.
  _updateInviteButtonVisibility(np.item);
  updateScrollPadVisibility(isComic);
}

/** Show the scroll pad in two distinct modes:
 *   - Comic mode: a comic is cast → pad drives the comic reader via
 *     patches. Auto-shows; user can dismiss for the session via ×.
 *   - Remote mode: TV is selected, no media casting, AND the user
 *     has explicitly enabled it via the "Follow on TV" toggle. The
 *     toggle IS the gate — without it the cast-control library stays
 *     the primary surface (no auto-show). × clears the toggle for a
 *     clean round-trip dismiss. */
function updateScrollPadVisibility(isComic) {
  const pad = $('[data-cc-scroll-pad]');
  if (!pad) return;
  const hasReceiver = !!state.selectedReceiverId;
  const np = loadNowPlaying();
  // Remote mode is opt-in: requires followMode to be explicitly on.
  // No auto-show — the library list remains primary.
  const remoteMode = hasReceiver && !np && state.followMode;
  const wantOpen = (isComic || remoteMode)
    && !state.comicPrefs.padDismissed;
  pad.classList.toggle('hidden', !wantOpen);
  pad.dataset.padMode = isComic ? 'comic' : (remoteMode ? 'remote' : '');
  // Sensitivity gradient — visible in any comic mode. Controller's
  // mode defaults to 'auto' and the TV resolves it, so gating on
  // 'webtoon' explicitly would never light up for the common case.
  pad.classList.toggle('comic-gain', isComic);
}


/* ── Comic sheet state sync ───────────────────────────────────── */

/** Reflect state.comicPrefs onto the comic sheet's active-button
 * styling. Called on each renderNowPlaying when a comic is cast and
 * after each pref change so the highlighted button is always the
 * source of truth. */
function renderComicSheetState() {
  const p = state.comicPrefs;
  // Swap which autoplay-row is visible based on current mode.
  // Webtoon mode wants continuous-velocity presets (px/sec); paged
  // and dual want discrete page-flip intervals (ms).
  const isWebtoon = p.mode === 'webtoon';
  const pagedRow = document.querySelector('[data-cc-np-comic-autoplay-row]');
  const pxRow    = document.querySelector('[data-cc-np-comic-autoplay-px-row]');
  if (pagedRow) pagedRow.hidden = isWebtoon;
  if (pxRow)    pxRow.hidden    = !isWebtoon;
  _setRowActive('[data-cc-np-comic-autoplay-row]',    'data-autoplay',    String(p.autoplayMs));
  _setRowActive('[data-cc-np-comic-autoplay-px-row]', 'data-autoplay-px', String(p.autoplayPxPerSec));
  _setRowActive('[data-cc-np-comic-mode-row]',        'data-mode',        p.mode);
  _setRowActive('[data-cc-np-comic-fit-row]',         'data-fit',         p.fit);
  _setToggleState('[data-cc-np-comic-dir]',  p.direction === 'rtl', 'RTL',    'LTR');
  _setToggleState('[data-cc-np-comic-crop]', p.borderCrop,          'On',     'Off');
  _setToggleState('[data-cc-np-comic-pad]',  !p.padDismissed,       'Shown',  'Hidden');
}

/** True when the current mode's autoplay engine is configured (not
 *  paused state — just whether a non-zero speed is set). Used to
 *  decide the ⏯ button's icon and the strip's visibility. */
function _isComicAutoplayActive() {
  if (state.comicPrefs.mode === 'webtoon') return state.comicPrefs.autoplayPxPerSec > 0;
  return state.comicPrefs.autoplayMs > 0;
}

function _setRowActive(rowSel, attr, value) {
  const row = document.querySelector(rowSel);
  if (!row) return;
  row.querySelectorAll('button').forEach((b) => {
    b.classList.toggle('is-active', b.getAttribute(attr) === value);
  });
}

function _setToggleState(sel, on, onLabel, offLabel) {
  const btn = document.querySelector(sel);
  if (!btn) return;
  btn.dataset.on = String(on);
  btn.textContent = on ? onLabel : offLabel;
}


/* ── Transport ──────────────────────────────────────────────── */

let _paused = false;  // local guess; the surface is the truth, but we don't read back

async function sendPatch(patch) {
  const np = loadNowPlaying();
  if (!np || !np.surface_id || !np.receiver_id) return;
  try {
    const r = await fetch('/api/cast/send/patch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        receiver_id: np.receiver_id,
        surface_id: np.surface_id,
        patch,
      }),
    });
    if (!r.ok) {
      // Dead surface — receiver was re-paired, container crashed, or
      // the user stopped from another device. Clear the mini-player
      // so the next tap doesn't fail the same way silently.
      console.warn('[cast-control] patch rejected', r.status);
      if (r.status === 404 || r.status === 502) {
        clearNowPlaying();
        renderNowPlaying();
        toast('Cast ended — pick something to play.');
      }
    }
  } catch (err) {
    console.warn('[cast-control] patch failed', err);
  }
}

/* ── Progress polling ──────────────────────────────────────────── */

function fmtClock(s) {
  if (!s || !isFinite(s)) return '—:—';
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

async function pollPlayback() {
  const np = loadNowPlaying();
  const wrap = $('[data-cc-np-progress]');
  if (!np || !np.item?.file_id) {
    if (wrap) wrap.hidden = true;
    return;
  }
  // Image surfaces have no progress to report.
  const kind = np.item.kind || '';
  if (kind === 'image') { if (wrap) wrap.hidden = true; return; }
  try {
    const r = await fetch(
      `/api/cast/playback-status/${encodeURIComponent(np.item.file_id)}`,
      { credentials: 'same-origin', cache: 'no-store' },
    );
    if (!r.ok) return;
    const body = await r.json();
    state.playbackStatus = body;
    renderProgress(body);
  } catch { /* swallow — next tick retries */ }
}

function renderProgress(p) {
  const wrap = $('[data-cc-np-progress]');
  if (!wrap || !p) return;
  const dur = Number(p.duration_s || 0);
  const pos = Number(p.current_time_s || 0);
  if (dur <= 0) {
    // Provider hasn't reported a duration yet — keep the strip
    // hidden so we don't show "0:00 / —:—".
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const pct = Math.max(0, Math.min(100, (pos / dur) * 100));
  $('[data-cc-np-bar-fill]').style.width = `${pct}%`;
  $('[data-cc-np-elapsed]').textContent = fmtClock(pos);
  $('[data-cc-np-duration]').textContent = fmtClock(dur);
}

function wireSeekBar() {
  const bar = $('[data-cc-np-bar]');
  if (!bar) return;
  bar.addEventListener('click', (ev) => {
    const dur = Number(state.playbackStatus.duration_s || 0);
    if (dur <= 0) return;
    const rect = bar.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const target = ratio * dur;
    sendPatch({ position_s: target });
    // Optimistic update so the user gets immediate feedback before
    // the next playback-status poll lands.
    state.playbackStatus.current_time_s = target;
    renderProgress(state.playbackStatus);
  });
}


function transportSkipBack()  { sendPatch({ seek_delta_s: -15 }); }
function transportSkipFwd()   { sendPatch({ seek_delta_s:  30 }); }
function transportToggle()    {
  _paused = !_paused;
  sendPatch({ paused: _paused });
  const t = $('[data-cc-np-toggle]');
  if (t) t.textContent = _paused ? '▶' : '⏸';
}
function transportPagePrev()  { sendPatch({ page_delta: -1 }); }
function transportPageNext()  { sendPatch({ page_delta:  1 }); }

/** Comic ⏯ — pauses or resumes the running autoplay. Off-state is
 *  hidden in renderNowPlaying, so this only runs when autoplayMs > 0. */
/** Comic ⏯ — three behaviors:
 *  1. Autoplay off → kick the mode-appropriate default speed and start
 *  2. Autoplay on, playing → pause
 *  3. Autoplay on, paused → resume
 *
 *  Default speeds: paged/dual = 15s/page, webtoon = 240 px/sec (Med). */
function transportComicAutoplayToggle() {
  const p = state.comicPrefs;
  const active = _isComicAutoplayActive();
  if (!active) {
    p.autoplayPaused = false;
    if (p.mode === 'webtoon') {
      p.autoplayPxPerSec = 240;
      sendPatch({ autoplay_px_per_sec: 240 });
    } else {
      p.autoplayMs = 15000;
      sendPatch({ autoplay_ms: 15000 });
    }
  } else {
    p.autoplayPaused = !p.autoplayPaused;
    sendPatch({ paused: p.autoplayPaused });
  }
  renderComicSheetState();
  renderNowPlaying();
}

function wireTransport() {
  $('[data-cc-np-back]')?.addEventListener('click', () => {
    const act = $('[data-cc-np-back]')?.dataset.action;
    if (act === 'skip_back') transportSkipBack();
    else if (act === 'page_prev') transportPagePrev();
  });
  $('[data-cc-np-fwd]')?.addEventListener('click', () => {
    const act = $('[data-cc-np-fwd]')?.dataset.action;
    if (act === 'skip_fwd') transportSkipFwd();
    else if (act === 'page_next') transportPageNext();
  });
  $('[data-cc-np-toggle]')?.addEventListener('click', () => {
    const act = $('[data-cc-np-toggle]')?.dataset.action;
    if (act === 'comic_autoplay_toggle') transportComicAutoplayToggle();
    else transportToggle();
  });

  // Sheet open/close
  $('[data-cc-np-more]')?.addEventListener('click', () => {
    const sheet = $('[data-cc-np-sheet]');
    if (sheet) sheet.hidden = !sheet.hidden;
  });
  $('[data-cc-np-sheet-close]')?.addEventListener('click', () => {
    const sheet = $('[data-cc-np-sheet]');
    if (sheet) sheet.hidden = true;
  });

  // CC picker buttons are wired by renderSubtitlePicker() each time
  // the tracks list refreshes — clicking a button sends the index
  // patch. Boot-time render happens in the boot block below.

  // Speed selector — single source of truth pinned to the controller's
  // local guess; the surface is authoritative but we don't read back
  // (yet). Future enhancement: surface_state echo from cast-video to
  // sync the highlight.
  document.querySelectorAll('[data-cc-np-speed-row] .np-speed').forEach((btn) => {
    btn.addEventListener('click', () => {
      const speed = Number(btn.dataset.speed || 1);
      document.querySelectorAll('[data-cc-np-speed-row] .np-speed').forEach(
        (b) => b.classList.toggle('is-active', b === btn),
      );
      sendPatch({ speed });
    });
  });

  // Jump buttons — audio/video only (data-skip = seconds). The comic
  // jump buttons live in a separate section with data-comic-* attrs
  // wired below so we don't cross the streams.
  document.querySelectorAll('.np-jump[data-skip]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const delta = Number(btn.dataset.skip || 0);
      if (delta) sendPatch({ seek_delta_s: delta });
    });
  });

  // ── Volume ────────────────────────────────────────────────────
  // Live drag on the slider sends a patch on every ``input`` event,
  // but throttled so the WS doesn't choke on a 0→100 sweep. The
  // mute button toggles independently of slider position (matches
  // native player behavior — mute remembers level).
  const volSlider = $('[data-cc-np-volume]');
  const volReadout = $('[data-cc-np-volume-readout]');
  const volMute = $('[data-cc-np-mute]');
  let _volSendAt = 0;
  let _volPendingPct = null;
  let _volMuted = false;

  function applyVolumeStyling(pct) {
    if (!volSlider) return;
    volSlider.style.setProperty('--cc-vol-pct', `${pct}%`);
    if (volReadout) volReadout.textContent = String(pct);
  }
  function flushVolume(pct) {
    sendPatch({ volume: Math.max(0, Math.min(1, pct / 100)) });
  }
  function handleVolumeInput() {
    const pct = Number(volSlider.value || 0);
    applyVolumeStyling(pct);
    // Throttle: at most one patch per 120ms during a drag, plus a
    // final flush in the trailing 200ms so the release lands.
    const now = Date.now();
    if (now - _volSendAt > 120) {
      _volSendAt = now;
      _volPendingPct = null;
      flushVolume(pct);
    } else {
      _volPendingPct = pct;
      clearTimeout(handleVolumeInput._t);
      handleVolumeInput._t = setTimeout(() => {
        if (_volPendingPct !== null) {
          flushVolume(_volPendingPct);
          _volPendingPct = null;
        }
      }, 200);
    }
  }
  function toggleMute() {
    _volMuted = !_volMuted;
    if (volMute) {
      volMute.classList.toggle('is-muted', _volMuted);
      volMute.textContent = _volMuted ? '🔇' : '🔊';
    }
    sendPatch({ muted: _volMuted });
  }
  volSlider?.addEventListener('input', handleVolumeInput);
  volMute?.addEventListener('click', toggleMute);
  // Initialise the visual styling so the slider shows its starting
  // fill (default value=100 → 100% accent fill).
  if (volSlider) applyVolumeStyling(Number(volSlider.value || 100));

  // ── A/V sync (lip-sync offset) ────────────────────────────────
  // Same debounced-input pattern as the volume slider. The receiver
  // applies the offset via a Web Audio DelayNode (see cast-video.js).
  // Persistence is keyed by file_id so a sticky source remembers its
  // setting across sessions; clean sources stay at 0.
  const syncSlider = $('[data-cc-np-sync]');
  const syncReadout = $('[data-cc-np-sync-readout]');
  const syncReset = $('[data-cc-np-sync-reset]');
  let _syncSendAt = 0;
  let _syncPendingMs = null;

  function applySyncStyling(ms) {
    if (syncSlider) {
      const pct = (ms / SYNC_OFFSET_MAX_MS) * 100;
      syncSlider.style.setProperty('--cc-vol-pct', `${pct}%`);
    }
    if (syncReadout) syncReadout.textContent = `${ms} ms`;
  }
  function flushSync(ms) {
    const fileId = loadNowPlaying()?.item?.file_id || '';
    saveSyncOffsetMs(fileId, ms);
    sendPatch({ audio_offset_ms: ms });
  }
  function handleSyncInput() {
    const ms = Math.max(0, Math.min(SYNC_OFFSET_MAX_MS, Number(syncSlider.value || 0)));
    applySyncStyling(ms);
    const now = Date.now();
    if (now - _syncSendAt > 120) {
      _syncSendAt = now;
      _syncPendingMs = null;
      flushSync(ms);
    } else {
      _syncPendingMs = ms;
      clearTimeout(handleSyncInput._t);
      handleSyncInput._t = setTimeout(() => {
        if (_syncPendingMs !== null) {
          flushSync(_syncPendingMs);
          _syncPendingMs = null;
        }
      }, 200);
    }
  }
  syncSlider?.addEventListener('input', handleSyncInput);
  syncReset?.addEventListener('click', () => {
    if (syncSlider) syncSlider.value = '0';
    applySyncStyling(0);
    flushSync(0);
  });
  if (syncSlider) applySyncStyling(Number(syncSlider.value || 0));

  // ── TV master volume ──────────────────────────────────────────
  // Sends to a different endpoint (/api/cast/send/system-volume)
  // that routes via WS to the AugmentumTV JS bridge on the
  // Android TV receiver. Same debouncing pattern.
  const tvSlider = $('[data-cc-np-tv-volume]');
  const tvReadout = $('[data-cc-np-tv-volume-readout]');
  const tvMute = $('[data-cc-np-tv-mute]');
  let _tvVolSendAt = 0;
  let _tvVolPendingPct = null;
  let _tvVolMuted = false;

  function applyTvVolumeStyling(pct) {
    if (!tvSlider) return;
    tvSlider.style.setProperty('--cc-vol-pct', `${pct}%`);
    if (tvReadout) tvReadout.textContent = String(pct);
  }
  async function sendSystemVolume(payload) {
    if (!state.selectedReceiverId) return;
    try {
      await fetch('/api/cast/send/system-volume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          receiver_id: state.selectedReceiverId,
          ...payload,
        }),
      });
    } catch (err) {
      console.warn('[cast-control] tv volume patch failed', err);
    }
  }
  function flushTvVolume(pct) {
    sendSystemVolume({ volume: Math.max(0, Math.min(1, pct / 100)) });
  }
  function handleTvVolumeInput() {
    const pct = Number(tvSlider.value || 0);
    applyTvVolumeStyling(pct);
    const now = Date.now();
    if (now - _tvVolSendAt > 150) {
      _tvVolSendAt = now;
      _tvVolPendingPct = null;
      flushTvVolume(pct);
    } else {
      _tvVolPendingPct = pct;
      clearTimeout(handleTvVolumeInput._t);
      handleTvVolumeInput._t = setTimeout(() => {
        if (_tvVolPendingPct !== null) {
          flushTvVolume(_tvVolPendingPct);
          _tvVolPendingPct = null;
        }
      }, 220);
    }
  }
  function toggleTvMute() {
    _tvVolMuted = !_tvVolMuted;
    if (tvMute) {
      tvMute.classList.toggle('is-muted', _tvVolMuted);
      tvMute.textContent = _tvVolMuted ? '🔇' : '📺';
    }
    sendSystemVolume({ muted: _tvVolMuted });
  }
  tvSlider?.addEventListener('input', handleTvVolumeInput);
  tvMute?.addEventListener('click', toggleTvMute);
  if (tvSlider) applyTvVolumeStyling(Number(tvSlider.value || 50));

  // ── Comic sheet wiring ────────────────────────────────────────
  // Each row updates state.comicPrefs (so the active-button highlight
  // survives sheet close/reopen) then patches the TV. All-in-one
  // delegation would be tighter but per-row is easier to reason about
  // and matches the existing audio/video pattern.

  document.querySelectorAll('[data-cc-np-comic-autoplay-row] .np-speed').forEach((btn) => {
    btn.addEventListener('click', () => {
      const ms = Number(btn.dataset.autoplay || 0);
      state.comicPrefs.autoplayMs = ms;
      state.comicPrefs.autoplayPaused = false;
      sendPatch({ autoplay_ms: ms });
      renderComicSheetState();
      renderNowPlaying();  // ⏯ icon depends on active state
    });
  });
  document.querySelectorAll('[data-cc-np-comic-autoplay-px-row] .np-speed').forEach((btn) => {
    btn.addEventListener('click', () => {
      const px = Number(btn.dataset.autoplayPx || 0);
      state.comicPrefs.autoplayPxPerSec = px;
      state.comicPrefs.autoplayPaused = false;
      sendPatch({ autoplay_px_per_sec: px });
      renderComicSheetState();
      renderNowPlaying();
    });
  });

  document.querySelectorAll('[data-cc-np-comic-mode-row] .np-speed').forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode || 'auto';
      state.comicPrefs.mode = mode;
      sendPatch({ mode });
      // Mode change swaps which autoplay row is shown and changes the
      // ⏯ button's source-of-truth, so re-render both.
      renderComicSheetState();
      renderNowPlaying();
    });
  });

  document.querySelectorAll('[data-cc-np-comic-fit-row] .np-speed').forEach((btn) => {
    btn.addEventListener('click', () => {
      const fit = btn.dataset.fit || 'smart';
      state.comicPrefs.fit = fit;
      sendPatch({ fit });
      renderComicSheetState();
    });
  });

  $('[data-cc-np-comic-dir]')?.addEventListener('click', () => {
    const next = state.comicPrefs.direction === 'ltr' ? 'rtl' : 'ltr';
    state.comicPrefs.direction = next;
    sendPatch({ reading_direction: next });
    renderComicSheetState();
  });

  $('[data-cc-np-comic-crop]')?.addEventListener('click', () => {
    const next = !state.comicPrefs.borderCrop;
    state.comicPrefs.borderCrop = next;
    sendPatch({ border_crop: next });
    renderComicSheetState();
  });

  document.querySelectorAll('.np-jump[data-comic-jump]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const where = btn.dataset.comicJump;
      if (where === 'first' || where === 'last') sendPatch({ jump: where });
    });
  });
  document.querySelectorAll('.np-jump[data-comic-delta]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const delta = Number(btn.dataset.comicDelta || 0);
      if (delta) sendPatch({ page_delta: delta });
    });
  });

  $('[data-cc-np-comic-pad]')?.addEventListener('click', () => {
    state.comicPrefs.padDismissed = !state.comicPrefs.padDismissed;
    renderComicSheetState();
    renderNowPlaying();
  });
}


/* ── Scroll pad ────────────────────────────────────────────────
 * Captures pointer drag, wheel, and arrow-key input and forwards
 * them to the TV as scroll_delta_px patches at ~30 Hz. The TV's
 * cast-comic surface scrolls the webtoon column (or accumulates
 * deltas into page-flips in paged/dual modes). Mouse momentum is
 * derived from the trailing pointer velocity. */

const SCROLL_PATCH_HZ = 30;                 // max patches per second
const SCROLL_PATCH_INTERVAL = 1000 / SCROLL_PATCH_HZ;
const SCROLL_FLASH_MS = 220;                // indicator pulse on wheel/key
const SCROLL_MOMENTUM_FRAMES = 14;          // tail length after fling
const SCROLL_MOMENTUM_DECAY = 0.84;         // per-frame velocity decay
const SCROLL_MIN_FLING_VEL = 0.4;           // px/ms threshold for momentum

// Webtoon-mode sensitivity gradient — left half = precise, right half =
// fast traversal. A 1:1 phone-px → TV-px mapping feels glacial because a
// webtoon strip is 8000+ px and a phone pad is ~300 px. Center sits at
// 3.5× so the median feels like native phone swiping; the gradient lets
// the user trade precision for speed by drifting horizontally.
const WEBTOON_GAIN_MIN = 1.5;
const WEBTOON_GAIN_MAX = 6.0;
const WEBTOON_GAIN_BASE = 3.5;              // keyboard / fallback
let _padLastGain = WEBTOON_GAIN_BASE;

let _padPointerActive = false;
let _padLastY = 0;
let _padPendingDelta = 0;                   // batched between throttled sends
let _padLastSendAt = 0;
let _padVelocitySamples = [];               // {t, y} for momentum calc
let _padMomentumRaf = 0;
let _padFlashTimer = 0;
let _padPausedByHold = false;               // hold-to-pause set this touch?
// Tap-vs-drag detection. A pointerdown→up cycle is a TAP if total
// movement stays below TAP_MOVE_PX and duration below TAP_TIME_MS.
// In remote mode this fires a {view:'tap'} nav to the TV at the
// proportional x/y of the trackpad.
const TAP_MOVE_PX = 8;
const TAP_TIME_MS = 280;
let _padStartX = 0, _padStartY = 0;
let _padLastX = 0;
let _padStartTs = 0;
let _padTotalMove = 0;

function wireScrollPad() {
  const pad     = $('[data-cc-scroll-pad]');
  const surface = $('[data-cc-scroll-pad-surface]');
  if (!pad || !surface) return;

  $('[data-cc-scroll-pad-close]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    // Dismiss semantics depend on which mode opened the pad:
    //   - Comic mode: pad is auto-shown for the comic surface, so
    //     mark it dismissed for the session (the More sheet's
    //     "Scroll pad: Shown" toggle brings it back).
    //   - Remote mode: the followMode toggle IS the gate, so flip
    //     it off — same way the topbar's Follow toggle would.
    //     User re-engages the trackpad by tapping the toggle on.
    const padMode = $('[data-cc-scroll-pad]')?.dataset.padMode || '';
    if (padMode === 'remote') {
      setFollowMode(false);
    } else {
      state.comicPrefs.padDismissed = true;
      renderComicSheetState();
    }
    renderNowPlaying();
  });

  // Pointer drag — covers touch on phones AND mouse-drag on laptops
  // through the unified Pointer Events API. touch-action: none on the
  // surface CSS prevents the browser from claiming the gesture as a
  // scroll. We invert the sign: drag UP on phone = comic moves UP
  // (i.e. user pulls content upward) which matches how a real scroll
  // would behave on the TV itself.
  surface.addEventListener('pointerdown', (e) => {
    if (e.button != null && e.button > 0) return;  // primary button only
    _padPointerActive = true;
    _padLastY = e.clientY;
    _padLastX = e.clientX;
    _padStartX = e.clientX;
    _padStartY = e.clientY;
    _padStartTs = performance.now();
    _padTotalMove = 0;
    _padVelocitySamples = [{ t: _padStartTs, y: e.clientY }];
    _stopMomentum();
    surface.setPointerCapture?.(e.pointerId);
    surface.classList.add('dragging');
    // Hold-to-pause: if autoplay is running, suspend it for the
    // duration of the touch. We only pause if it was actively
    // playing — if the user had already paused via ⏯, leave that
    // state alone (releasing the pad would unexpectedly resume it).
    if (_isComicAutoplayActive() && !state.comicPrefs.autoplayPaused) {
      _padPausedByHold = true;
      state.comicPrefs.autoplayPaused = true;
      sendPatch({ paused: true });
    }
  });
  surface.addEventListener('pointermove', (e) => {
    if (!_padPointerActive) return;
    const dy = e.clientY - _padLastY;
    const dx = e.clientX - _padLastX;
    _padTotalMove += Math.abs(dy) + Math.abs(dx);
    _padLastY = e.clientY;
    _padLastX = e.clientX;
    // Webtoon gain — recompute each move so drifting the finger
    // rightward speeds up mid-drag. Outside webtoon the helper returns
    // 1.0 so paged/dual/remote behavior is unchanged.
    const gain = _padGainAt(surface, e.clientX);
    _padPendingDelta -= dy * gain;  // drag down → comic scrolls up (natural)
    _trackVelocity(e.clientY);
    _maybeFlushScroll();
    _updateIndicator(surface, e.clientX, e.clientY);
  });
  const endDrag = (e) => {
    if (!_padPointerActive) return;
    _padPointerActive = false;
    surface.releasePointerCapture?.(e.pointerId);
    surface.classList.remove('dragging');
    // Tap detection BEFORE flushing scroll — a true tap shouldn't
    // generate a final scroll patch from the trailing velocity.
    const duration = performance.now() - _padStartTs;
    const isTap = _padTotalMove < TAP_MOVE_PX && duration < TAP_TIME_MS;
    if (isTap) {
      _padPendingDelta = 0;  // discard any pixel noise
      _emitTap(surface, e.clientX, e.clientY);
    } else {
      _flushScroll();
      _startMomentum();
    }
    // Release the hold-to-pause if we set it on pointerdown.
    if (_padPausedByHold) {
      _padPausedByHold = false;
      state.comicPrefs.autoplayPaused = false;
      sendPatch({ paused: false });
    }
  };
  surface.addEventListener('pointerup', endDrag);
  surface.addEventListener('pointercancel', endDrag);

  // Wheel — laptop scroll wheel / trackpad two-finger scroll. Wheel
  // events come pre-discretized; send each one with proper sign.
  // deltaY > 0 = scrolling down → TV scrolls down (same direction).
  surface.addEventListener('wheel', (e) => {
    e.preventDefault();
    // Some browsers (FF) report large LINE-unit deltas. Normalize to
    // pixels — modern Chromium reports PIXEL by default with ~100 per
    // notch, which is a sensible "one scroll click = ~viewport/8".
    let px = e.deltaY;
    if (e.deltaMode === 1) px *= 16;        // LINE
    else if (e.deltaMode === 2) px *= window.innerHeight;  // PAGE
    // Wheel events carry clientX so the gain still tracks horizontal
    // position (trackpad two-finger scroll on the right side = faster).
    _padPendingDelta += px * _padGainAt(surface, e.clientX);
    _maybeFlushScroll();
    _flashIndicator(surface);
  }, { passive: false });

  // Keyboard — arrow keys + PgUp/PgDn scroll when the pad has focus.
  // We don't capture globally because the user might have an input
  // focused elsewhere (search box etc).
  pad.addEventListener('keydown', (e) => {
    const big = e.shiftKey || e.key === 'PageDown' || e.key === 'PageUp';
    const step = big ? 0.85 * window.innerHeight : 80;
    let dy = 0;
    if (e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ') dy = step;
    else if (e.key === 'ArrowUp' || e.key === 'PageUp')                 dy = -step;
    else if (e.key === 'Home') { sendPatch({ jump: 'first' }); e.preventDefault(); return; }
    else if (e.key === 'End')  { sendPatch({ jump: 'last' });  e.preventDefault(); return; }
    else return;
    e.preventDefault();
    // Keyboard has no cursor position — use the base gain (center of
    // the gradient). In paged/dual/remote modes this is 1.0.
    _padPendingDelta += dy * _padGainBase();
    _maybeFlushScroll();
    _flashIndicator(surface);
  });
}

/** Comic-mode gain from horizontal position on the pad surface.
 *  Linear interpolation across the surface width — left edge =
 *  WEBTOON_GAIN_MIN (precise), right edge = WEBTOON_GAIN_MAX (fast).
 *  Applied in any comic mode: the controller's mode is 'auto' by
 *  default and the TV auto-detects webtoon, so a webtoon-only gate
 *  would no-op for typical users. Paged/dual accumulate into
 *  page-flips at a half-viewport threshold — a stronger swipe just
 *  flips faster, which matches native phone scroll semantics.
 *  Remote mode (no media) stays 1:1 — it drives a normal web page. */
function _padGainAt(surface, clientX) {
  if ($('[data-cc-scroll-pad]')?.dataset.padMode !== 'comic') return 1.0;
  const rect = surface.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const t = Math.max(0, Math.min(1, (clientX - rect.left) / width));
  const gain = WEBTOON_GAIN_MIN + (WEBTOON_GAIN_MAX - WEBTOON_GAIN_MIN) * t;
  _padLastGain = gain;
  return gain;
}

function _padGainBase() {
  if ($('[data-cc-scroll-pad]')?.dataset.padMode !== 'comic') return 1.0;
  return _padLastGain || WEBTOON_GAIN_BASE;
}

function _trackVelocity(y) {
  const now = performance.now();
  _padVelocitySamples.push({ t: now, y });
  // Keep only the last ~80 ms — enough for a stable velocity, short
  // enough that decelerating drags don't get over-flung.
  const cutoff = now - 80;
  while (_padVelocitySamples.length > 2 && _padVelocitySamples[0].t < cutoff) {
    _padVelocitySamples.shift();
  }
}

function _startMomentum() {
  if (_padVelocitySamples.length < 2) return;
  const first = _padVelocitySamples[0];
  const last = _padVelocitySamples[_padVelocitySamples.length - 1];
  const dt = last.t - first.t;
  if (dt <= 0) return;
  // px/ms — note Y INVERTED because drag-up = scroll-down
  let vel = -(last.y - first.y) / dt;
  if (Math.abs(vel) < SCROLL_MIN_FLING_VEL) return;
  let frames = 0;
  // Freeze the gain at fling-release time — momentum should honor the
  // sensitivity the user picked, not whatever happens to be under
  // _padLastGain by the time the rAF fires.
  const gain = _padGainBase();
  const tick = () => {
    frames += 1;
    if (frames > SCROLL_MOMENTUM_FRAMES) { _padMomentumRaf = 0; return; }
    // Emit one frame's worth of scroll (vel is px/ms × 16 ms/frame).
    _padPendingDelta += vel * 16 * gain;
    _maybeFlushScroll();
    vel *= SCROLL_MOMENTUM_DECAY;
    if (Math.abs(vel) < 0.05) { _padMomentumRaf = 0; return; }
    _padMomentumRaf = requestAnimationFrame(tick);
  };
  _padMomentumRaf = requestAnimationFrame(tick);
}

function _stopMomentum() {
  if (_padMomentumRaf) cancelAnimationFrame(_padMomentumRaf);
  _padMomentumRaf = 0;
}

function _maybeFlushScroll() {
  const now = performance.now();
  if (now - _padLastSendAt >= SCROLL_PATCH_INTERVAL) _flushScroll();
}

function _flushScroll() {
  if (_padPendingDelta === 0) return;
  const delta = Math.round(_padPendingDelta);
  if (delta === 0) return;
  _padPendingDelta -= delta;  // preserve the sub-pixel remainder
  _padLastSendAt = performance.now();
  // Dual-mode dispatch: when an active cast surface is mounted we
  // patch it (cast-comic etc. consume `scroll_delta_px`). When no
  // media is casting, the TV is showing cast-home — route the scroll
  // through the device-scoped nav channel instead so cast-home's
  // window.scrollBy handler picks it up.
  const padMode = $('[data-cc-scroll-pad]')?.dataset.padMode || '';
  if (padMode === 'remote') {
    sendNavToTv({ view: 'scroll', dy: delta }, { force: true });
  } else {
    sendPatch({ scroll_delta_px: delta });
  }
}

/** Tap on the trackpad — fires a TV-side click at the proportional
 *  position. Only meaningful in remote mode (cast-home is what's
 *  showing on the TV). Other modes (comic) ignore taps; the comic
 *  reader has its own page-flip semantics on the prev/next buttons. */
function _emitTap(surface, clientX, clientY) {
  // padMode is set on the outer scroll-pad element by
  // updateScrollPadVisibility, not on the inner pointer surface.
  const padMode = $('[data-cc-scroll-pad]')?.dataset.padMode || '';
  if (padMode !== 'remote') return;
  const rect = surface.getBoundingClientRect();
  const xPct = (clientX - rect.left) / Math.max(1, rect.width);
  const yPct = (clientY - rect.top) / Math.max(1, rect.height);
  sendNavToTv({
    view: 'tap',
    x_pct: Math.max(0, Math.min(1, xPct)),
    y_pct: Math.max(0, Math.min(1, yPct)),
  }, { force: true });
  _flashIndicator(surface);
}

function _flashIndicator(surface) {
  surface.classList.add('flash');
  clearTimeout(_padFlashTimer);
  _padFlashTimer = setTimeout(
    () => surface.classList.remove('flash'),
    SCROLL_FLASH_MS,
  );
}

function _updateIndicator(surface, x, y) {
  const indicator = $('[data-cc-scroll-pad-indicator]', surface);
  if (!indicator) return;
  const rect = surface.getBoundingClientRect();
  const lx = x - rect.left;
  const ly = y - rect.top;
  indicator.style.transform = `translate(${lx - rect.width / 2}px, ${ly - rect.height / 2}px)`;
}

async function stopNowPlaying() {
  const np = loadNowPlaying();
  if (!np) return;
  try {
    await fetch('/api/cast/send/close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        receiver_id: np.receiver_id,
        surface_id: np.surface_id,
      }),
    });
  } catch { /* swallow */ }
  clearNowPlaying();
  renderNowPlaying();
  toast('Stopped');
}


/* ── Couch co-op invite trigger ────────────────────────────────── */


// Cached active-invite token so we can revoke + know whether the
// waiting panel should be visible across renderNowPlaying signature
// invalidations.
let _activeInvite = null;

function wireInviteButton() {
  $('[data-cc-np-invite]')?.addEventListener('click', _onInviteTap);
  $('[data-cc-np-invite-revoke]')?.addEventListener('click', _onInviteRevokeTap);
}

async function _onInviteTap() {
  const np = loadNowPlaying();
  const sessionId = np?.stream_session_id || '';
  const receiverId = np?.receiver_id || state.selectedReceiverId || '';
  if (!sessionId || !receiverId) return;
  // Optimistic UI: show the waiting panel immediately, then settle
  // once the mint response lands. Failed mints clear it.
  _renderInvitePanel({ status: 'minting' });
  try {
    const r = await fetch(
      `/api/cast/games/session/${encodeURIComponent(sessionId)}/invite`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ receiver_id: receiverId, max_slots: 3 }),
      },
    );
    if (!r.ok) {
      const detail = await r.text().catch(() => '');
      _renderInvitePanel({
        status: 'error',
        message: r.status === 409 ? 'This game is single-player.' : `Failed (${r.status}).`,
        detail,
      });
      return;
    }
    const body = await r.json();
    _activeInvite = {
      token: body.token,
      sessionId,
      slotsRemaining: body.slots_remaining,
      slotsTotal: body.slots_total,
      expiresAt: body.expires_at,
    };
    _renderInvitePanel({ status: 'active', ..._activeInvite });
  } catch (err) {
    _renderInvitePanel({ status: 'error', message: 'Network error.', detail: String(err) });
  }
}

async function _onInviteRevokeTap() {
  if (!_activeInvite) {
    _renderInvitePanel({ status: 'idle' });
    return;
  }
  const { sessionId, token } = _activeInvite;
  try {
    await fetch(
      `/api/cast/games/session/${encodeURIComponent(sessionId)}/invite/${encodeURIComponent(token)}/revoke`,
      { method: 'POST' },
    );
  } catch (err) {
    console.warn('[cast-control] invite revoke failed', err);
  }
  _activeInvite = null;
  _renderInvitePanel({ status: 'idle' });
}

function _renderInvitePanel(view) {
  const panel = $('[data-cc-np-invite-panel]');
  const status = $('[data-cc-np-invite-status]');
  if (!panel || !status) return;
  if (view.status === 'idle') {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  if (view.status === 'minting') {
    status.textContent = 'Generating invite…';
  } else if (view.status === 'active') {
    const remaining = view.slotsRemaining;
    const total = view.slotsTotal;
    const slotLabel = remaining === 1 ? '1 slot open' : `${remaining} slots open`;
    status.textContent = `Scan the QR on the TV — ${slotLabel} of ${total}.`;
  } else if (view.status === 'error') {
    status.textContent = view.message || 'Invite failed.';
  }
}

/** Show or hide the "+ Players" button based on the current
 *  now-playing item. Called from renderNowPlaying. */
function _updateInviteButtonVisibility(item) {
  const btn = $('[data-cc-np-invite]');
  if (!btn) return;
  const isMultiplayerGame = (
    item?.kind === 'game'
    && (item?.max_players ?? 1) > 1
  );
  btn.hidden = !isMultiplayerGame;
  if (!isMultiplayerGame && _activeInvite) {
    // Game changed mid-invite — clean up the stale panel.
    _activeInvite = null;
    _renderInvitePanel({ status: 'idle' });
  }
}


/* ── Boot ───────────────────────────────────────────────────────── */

wirePicker();
wireTransport();
wireSeekBar();
wireLibrarySheet();
wireScrollPad();
wireFollowToggle();
wirePrefsButton();
$('[data-cc-np-stop]')?.addEventListener('click', stopNowPlaying);
wireInviteButton();
renderNowPlaying();

refreshReceivers();
refreshLibrary();
pollPlayback();  // initial draw of the progress bar if a cast is live
// If there's an active video cast from a previous session, populate
// the subtitle picker on boot so it's not empty until the next cast.
(() => {
  const np = loadNowPlaying();
  if (np?.item?.kind === 'video' && np.item.file_id) {
    fetchSubtitleTracksFor(np.item.file_id);
  } else {
    renderSubtitlePicker();
  }
})();

// If a game cast survived a page refresh, restart the input WS so
// the phone keeps driving the emulator after the user returned to
// cast-control. Guarded on stream_session_id because pre-this-change
// now-playing rows didn't carry it.
(() => {
  const np = loadNowPlaying();
  if (np?.stream_session_id) {
    startProducer(np.stream_session_id, _onControllerStatus);
  }
})();
state.receiverPollTimer = setInterval(refreshReceivers, RECEIVER_POLL_MS);
state.libraryRefreshTimer = setInterval(refreshLibrary, LIBRARY_REFRESH_MS);
state.playbackPollTimer = setInterval(pollPlayback, PLAYBACK_POLL_MS);

// Auto-pause an active game stream if the phone is backgrounded for
// more than IDLE_PAUSE_MS. Keeps container CPU/encoder load at zero
// when nobody's actually playing, while preserving game state in RAM
// for a sub-second resume when the user comes back. Coupled with the
// server-side credit budget — paused sessions free their active-credit
// slot for other users without losing the session reservation.
const IDLE_PAUSE_MS = 30000;
let _idlePauseTimer = null;

async function _pauseActiveStream(reason) {
  const np = loadNowPlaying();
  const sid = np?.stream_session_id;
  if (!sid) return;
  try {
    await fetch(`/api/cast/games/session/${encodeURIComponent(sid)}/pause`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    stopProducer(sid, reason || 'visibility_hidden');
  } catch {
    // Best-effort. If the network is gone the container will hit the
    // server-side idle/paused-stop sweeps as a fallback.
  }
}

async function _resumeActiveStream() {
  const np = loadNowPlaying();
  const sid = np?.stream_session_id;
  if (!sid) return;
  try {
    const r = await fetch(
      `/api/cast/games/session/${encodeURIComponent(sid)}/resume`,
      { method: 'POST', credentials: 'same-origin' },
    );
    if (r.ok) {
      // Re-open the producer once the server is back in CONNECTED.
      startProducer(sid, _onControllerStatus);
    }
  } catch {
    // Same fallback as pause — server will keep the session paused.
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    if (_idlePauseTimer) clearTimeout(_idlePauseTimer);
    _idlePauseTimer = setTimeout(
      () => _pauseActiveStream('visibility_hidden_timeout'),
      IDLE_PAUSE_MS,
    );
  } else {
    if (_idlePauseTimer) {
      clearTimeout(_idlePauseTimer);
      _idlePauseTimer = null;
    }
    refreshReceivers();
    refreshLibrary();
    pollPlayback();
    _resumeActiveStream();
  }
});
