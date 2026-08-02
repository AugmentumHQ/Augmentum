/**
 * cast-shelf.js — floating cast affordance for the main Augmentum UI.
 *
 * Responsibilities:
 *
 *   1. TV directory — bottom-right "Cast" button shows a popover with
 *      one row per connected receiver. Hidden when zero TVs are paired
 *      so it doesn't take real estate.
 *
 *   2. Active-cast transport — every receiver that has something
 *      casting gets inline playback controls (scrubber, pause, skip,
 *      volume, stop) in its row. Replaces the prior separate
 *      cast-remote pill which couldn't be dismissed and duplicated
 *      this surface.
 *
 *   3. Manage TVs link — opens /ui/cast-stage/, the editorial
 *      management surface for durable trust + audit.
 *
 * Cast-starting affordances live on the content surfaces themselves
 * (floating-video / media-mini-player call openCastPicker). The shelf
 * is a directory + transport, not a launch pad for arbitrary content.
 *
 * Polling:
 *   - /api/cast/receivers — fast when any cast is active (so the
 *     scrubber stays accurate), slow when all idle.
 *   - /api/cast/trusted-receivers — enriches with currently_showing
 *     so the transport surfaces in the same paint.
 */

import { escapeHtml } from './app.js';


const POLL_ACTIVE_MS = 3000;
const POLL_IDLE_MS = 8000;

const state = {
  receivers: [],            // last polled list of connected ConnectedReceiver dicts
  trustedById: new Map(),   // trusted_id → trusted receiver dict (with currently_showing)
  open: false,              // popover open?
  pollTimer: null,
  initialized: false,
  scrubDragging: false,     // suspend scrubber writes while user drags
};

let _trigger = null;
let _popover = null;


/* ── Public entry ─────────────────────────────────────────────── */


export function initCastShelf() {
  if (state.initialized) return;
  state.initialized = true;
  _ensureDom();
  _scheduleNextPoll(0);  // initial fetch
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) _refreshReceivers();
  });
}


/* ── DOM ──────────────────────────────────────────────────────── */


function _ensureDom() {
  _trigger = document.createElement('button');
  _trigger.className = 'cast-shelf-trigger hidden';
  _trigger.title = 'Cast to TV';
  _trigger.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M2 16.1A5 5 0 0 1 5.9 20"/>
      <path d="M2 12.05A9 9 0 0 1 9.95 20"/>
      <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
      <line x1="2" y1="20" x2="2.01" y2="20"/>
    </svg>
    <span class="cast-shelf-trigger-label">Cast</span>
    <span class="cast-shelf-count" data-cast-shelf-count></span>
  `;
  _trigger.addEventListener('click', () => _setOpen(!state.open));
  // Mount inline in the chat composer's toolbar row when it exists, so
  // the pill sits next to the summon-companion / tools / web-search
  // buttons rather than floating in the corner. The CSS keys off the
  // ``--inline`` class to drop position:fixed + the pill chrome.
  // Coder mode + non-chat surfaces (no #input-toolbar) fall back to
  // the floating bottom-right placement.
  _mountTriggerInContext();
  // Re-parent when modes swap — the input-toolbar disappears in coder
  // mode, reappears in chat. augmentum:mode-changed fires on every swap.
  document.addEventListener('augmentum:mode-changed', _mountTriggerInContext);

  _popover = document.createElement('div');
  _popover.className = 'cast-shelf-popover';
  _popover.innerHTML = `
    <div class="cast-shelf-head">
      <span>Cast to TV</span>
      <button class="cast-shelf-close" data-cast-shelf-close aria-label="Close">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 6L6 18M6 6l12 12"/>
        </svg>
      </button>
    </div>
    <div class="cast-shelf-body" data-cast-shelf-body></div>
    <div class="cast-shelf-foot">
      <a href="/ui/cast-control/" target="_blank" rel="noopener" class="cast-shelf-foot-link">
        Browse for TV →
      </a>
      <button class="cast-shelf-foot-link" data-cast-shelf-manage type="button">
        Manage TVs…
      </button>
      <a href="/ui/cast-receiver/" target="_blank" rel="noopener" class="cast-shelf-foot-link">
        Pair a new TV
      </a>
    </div>
  `;
  document.body.appendChild(_popover);

  _popover.querySelector('[data-cast-shelf-close]').addEventListener('click', () => _setOpen(false));
  _popover.querySelector('[data-cast-shelf-manage]').addEventListener('click', () => {
    _setOpen(false);
    // Manage TVs lives at its own editorial surface — open it in
    // a new tab so the user doesn't lose whatever they were doing
    // in the main app.
    window.open('/ui/cast-stage/', '_blank', 'noopener');
  });

  document.addEventListener('click', (e) => {
    if (!state.open) return;
    if (_popover.contains(e.target) || _trigger.contains(e.target)) return;
    _setOpen(false);
  });

  // Drop any FAB drag position from prior builds — the trigger is now
  // anchored by mode/companion context, not user drag.
  try { localStorage.removeItem('augmentum.cast.shelf.triggerPos'); } catch {}

  _trackCompanionPresence();
  window.addEventListener('resize', () => {
    _syncBottomExtra();
    if (state.open) _anchorPopover();
  });
}


/* ── Companion-aware vertical offset ──────────────────────────────
 *
 * Different mode UIs put different things in the bottom-right corner:
 *
 *   - passthrough / analytical / narrative / agentic: nothing fixed
 *     in the corner. Becca-presence may be mounted in the right
 *     gutter (default bottom: 24px; 360×480, user-resizable).
 *   - coder: `.coder-agent-btn` is pinned at bottom: 16px right: 16px.
 *
 * Rather than fight for the corner, the trigger stacks above whatever
 * is there. We publish a CSS var `--cast-shelf-bottom-extra` on
 * documentElement; the cast-shelf-trigger CSS adds it to its base
 * `bottom`. Recomputed on companion mount/unmount/resize/drag and on
 * window resize.
 *
 * Coder mode is handled in pure CSS via a `body[data-mode="coder"]`
 * selector since `.coder-agent-btn` has a fixed footprint.
 */
function _trackCompanionPresence() {
  let _beccaRO = null;
  let _beccaMO = null;
  let _watchedBecca = null;

  const sync = () => _syncBottomExtra();

  const watch = (el) => {
    if (_watchedBecca === el) return;
    if (_beccaRO) { try { _beccaRO.disconnect(); } catch {} }
    if (_beccaMO) { try { _beccaMO.disconnect(); } catch {} }
    _watchedBecca = el;
    if (!el) { sync(); return; }
    _beccaRO = new ResizeObserver(sync);
    _beccaRO.observe(el);
    // Becca's drag updates her inline `style` (top/left); a mutation
    // observer on her style attribute catches position changes that
    // ResizeObserver doesn't.
    _beccaMO = new MutationObserver(sync);
    _beccaMO.observe(el, { attributes: true, attributeFilter: ['style', 'class'] });
    sync();
  };

  // Watch body for Becca's mount/unmount.
  const bodyMO = new MutationObserver(() => {
    const el = document.querySelector('.becca-presence');
    if (el !== _watchedBecca) watch(el);
  });
  bodyMO.observe(document.body, { childList: true, subtree: false });
  watch(document.querySelector('.becca-presence'));
}


/* Inline (chat composer toolbar) is the primary placement: the pill
 * lives alongside the summon-companion / tools / web-search buttons and
 * therefore never moves regardless of viewport width or whether Becca
 * is summoned. Float-fallback handles the surfaces that don't have a
 * composer toolbar (today: coder mode).
 *
 * Becca lift logic only matters in float mode — when the pill is inline
 * inside the toolbar, the toolbar is part of the chat shell layout and
 * Becca can never paint over it (she's z-indexed below the input area).
 */
function _mountTriggerInContext() {
  if (!_trigger) return;
  const toolbar = document.getElementById('input-toolbar');
  if (toolbar) {
    if (_trigger.parentNode !== toolbar) {
      // Mount at the end so it sits after #becca-summon-btn — the user's
      // mental model is "another toolbar action, in the same row".
      toolbar.appendChild(_trigger);
    }
    _trigger.classList.add('cast-shelf-trigger--inline');
    // Inline placement doesn't need the Becca-aware vertical lift.
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
  } else {
    if (_trigger.parentNode !== document.body) {
      document.body.appendChild(_trigger);
    }
    _trigger.classList.remove('cast-shelf-trigger--inline');
    // Recompute the float-mode Becca lift now that we're floating again.
    _syncBottomExtra();
  }
}


/* Float-mode Becca lift: only used when the trigger is NOT inline
 * (coder mode, etc.). She can sit anywhere in the right gutter and is
 * tall enough to swallow the static base; lift the pill above her top
 * edge by publishing `--cast-shelf-bottom-extra`. Inline mode skips
 * this entirely (the toolbar Becca can't reach).
 */
function _syncBottomExtra() {
  const trigger = _trigger || document.querySelector('.cast-shelf-trigger');
  if (!trigger || trigger.classList.contains('cast-shelf-trigger--inline')) {
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
    return;
  }
  const becca = document.querySelector('.becca-presence');
  if (!becca) {
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
    return;
  }
  const rect = becca.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
    return;
  }
  const fromRight = window.innerWidth - rect.right;
  const trigWidth = trigger.getBoundingClientRect().width || 140;
  if (fromRight >= 24 + trigWidth + 12) {
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
    return;
  }
  const baseRaw = getComputedStyle(trigger).getPropertyValue('--cast-shelf-base-bottom');
  const base = parseFloat(baseRaw) || 24;
  const fromBottom = Math.max(0, window.innerHeight - rect.top + 12);
  const lift = Math.max(0, Math.round(fromBottom - base));
  if (lift <= 0) {
    document.documentElement.style.removeProperty('--cast-shelf-bottom-extra');
    return;
  }
  document.documentElement.style.setProperty('--cast-shelf-bottom-extra', `${lift}px`);
}


/**
 * Position the popover next to the trigger's current rect.
 *
 * The trigger can move (user drag) so the popover can't rely on the
 * static ``bottom: 150px; right: 16px`` it had in CSS — once the
 * trigger isn't at its default spot, those coords leave the menu
 * floating in the corner with no visual link to the button. We
 * compute a position that prefers above-and-aligned-with the trigger
 * and falls back to below-it when there isn't headroom.
 */
function _anchorPopover() {
  if (!_popover || !_trigger) return;
  const tRect = _trigger.getBoundingClientRect();
  const pRect = _popover.getBoundingClientRect();
  // If the popover hasn't been laid out yet (display: none), grab
  // its declared width from CSS via offsetWidth after a transient
  // visibility toggle. Faster path: trust the CSS max-width of
  // min(380px, 92vw) and use the smaller value as the estimate.
  const pw = pRect.width || Math.min(380, window.innerWidth * 0.92);
  const ph = pRect.height || 360;  // rough — refined on next open

  // Preferred: above the trigger, right-aligned with it.
  let left = tRect.right - pw;
  let top = tRect.top - ph - 10;

  // Not enough headroom → place below the trigger.
  if (top < 8) top = tRect.bottom + 10;

  // Keep inside the viewport horizontally + vertically.
  left = Math.max(8, Math.min(window.innerWidth - pw - 8, left));
  top  = Math.max(8, Math.min(window.innerHeight - ph - 8, top));

  _popover.style.left = `${left}px`;
  _popover.style.top = `${top}px`;
  _popover.style.right = 'auto';
  _popover.style.bottom = 'auto';
}


function _setOpen(open) {
  state.open = !!open;
  _popover.classList.toggle('open', state.open);
  if (state.open) {
    // Anchor twice: once before paint with the previous rect (cheap
    // estimate), then again after the body renders so the second
    // anchor lands with the popover's real height. Cheap.
    _anchorPopover();
    _refreshReceivers();
    _refreshTrusted();  // fire-and-forget: enriches once it lands
    _renderBody();
    requestAnimationFrame(_anchorPopover);
  }
}


/* ── Receiver polling ─────────────────────────────────────────── */


async function _refreshReceivers() {
  try {
    const r = await fetch('/api/cast/receivers', { credentials: 'same-origin' });
    if (!r.ok) {
      state.receivers = [];
    } else {
      const body = await r.json();
      state.receivers = Array.isArray(body.receivers) ? body.receivers : [];
    }
  } catch {
    state.receivers = [];
  }
  _renderTrigger();
  if (state.open) _renderBody();
}


async function _refreshTrusted() {
  try {
    const r = await fetch('/api/cast/trusted-receivers', {
      credentials: 'same-origin',
    });
    if (!r.ok) {
      state.trustedById = new Map();
    } else {
      const body = await r.json();
      const list = Array.isArray(body.receivers) ? body.receivers : [];
      state.trustedById = new Map(list.map(t => [t.id, t]));
    }
  } catch {
    state.trustedById = new Map();
  }
  if (state.open) _renderBody();
}


function _scheduleNextPoll(delay) {
  clearTimeout(state.pollTimer);
  if (typeof delay !== 'number') {
    delay = _anyReceiverActive() ? POLL_ACTIVE_MS : POLL_IDLE_MS;
  }
  state.pollTimer = setTimeout(async () => {
    await _refreshReceivers();
    // Always refresh trusted alongside receivers when a cast is in
    // flight — the scrubber needs fresh position_s every tick. When
    // idle we skip it to halve the request rate.
    if (_anyReceiverActive() || state.open) await _refreshTrusted();
    _scheduleNextPoll();
  }, delay);
}


function _anyReceiverActive() {
  return state.receivers.some((r) => !!_currentlyShowingFor(r));
}


/* ── Quick-cast popover rendering ─────────────────────────────── */


function _renderTrigger() {
  if (!_trigger) return;
  const count = state.receivers.length;
  const countEl = _trigger.querySelector('[data-cast-shelf-count]');
  if (count > 0) {
    _trigger.classList.remove('hidden');
    if (countEl) countEl.textContent = String(count);
  } else {
    _trigger.classList.add('hidden');
    if (state.open) _setOpen(false);
  }
  const hasActive = state.receivers.some(r => !!_currentlyShowingFor(r));
  _trigger.classList.toggle('has-active', hasActive);
}


function _currentlyShowingFor(receiver) {
  // Server's trusted-store currently_showing is the authoritative view.
  // We pick `[0]` (the most recent active event) — the orphan fix in
  // receiver_registry.record_event now closes prior events on
  // surface_closed, so this should be the only active row in practice.
  const trusted = receiver.trusted_id
    ? state.trustedById.get(receiver.trusted_id)
    : null;
  if (trusted && Array.isArray(trusted.currently_showing) && trusted.currently_showing.length) {
    const ev = trusted.currently_showing.find(e => e.active) || trusted.currently_showing[0];
    return ev || null;
  }
  return null;
}


function _renderBody() {
  if (!_popover) return;
  // Don't yank the scrubber out from under the user mid-drag. The
  // next poll after release catches up. scrubDragging is reset by
  // the slider's `change` handler.
  if (state.scrubDragging) return;
  const body = _popover.querySelector('[data-cast-shelf-body]');
  if (!body) return;

  if (!state.receivers.length) {
    body.innerHTML = `
      <div class="cast-shelf-empty">
        No TVs connected. Open
        <a href="/ui/cast-receiver/" target="_blank" rel="noopener">/ui/cast-receiver/</a>
        on a TV / phone / browser tab to pair, then it'll show up here.
      </div>`;
    return;
  }

  body.innerHTML = state.receivers.map(r => _receiverRowHtml(r)).join('');

  // Per-row transport bindings. Idle rows have no buttons so the
  // querySelectorAll skips them naturally.
  state.receivers.forEach((receiver) => {
    const rid = receiver.registration_id;
    const row = body.querySelector(`[data-cs-receiver-row="${escapeAttr(rid)}"]`);
    if (!row) return;
    const showing = _currentlyShowingFor(receiver);
    if (!showing) return;
    _wireTransportRow(row, receiver, showing);
  });
}


function _wireTransportRow(row, receiver, showing) {
  const rid = receiver.registration_id;
  const surfaceId = showing.surface_id;
  const isAudio = _capabilityFor(showing) === 'media.audio_play@1';
  const skipBack = isAudio ? 15 : 10;
  const skipFwd  = isAudio ? 30 : 30;

  row.querySelector('[data-cs-action="play"]')?.addEventListener('click', () => {
    _patch(rid, surfaceId, _isPausedOn(showing)
      ? { paused: false }
      : { paused: true });
  });

  row.querySelector('[data-cs-action="skip-back"]')?.addEventListener('click', () => {
    _patch(rid, surfaceId, { seek_delta_s: -skipBack });
  });

  row.querySelector('[data-cs-action="skip-forward"]')?.addEventListener('click', () => {
    _patch(rid, surfaceId, { seek_delta_s: skipFwd });
  });

  row.querySelector('[data-cs-action="stop"]')?.addEventListener('click', () => {
    _stop(rid, surfaceId);
  });

  row.querySelector('[data-cs-action="mute"]')?.addEventListener('click', () => {
    const muted = !!showing.surface_state?.muted;
    _patch(rid, surfaceId, { muted: !muted });
  });

  const volSlider = row.querySelector('[data-cs-volume]');
  if (volSlider) {
    volSlider.addEventListener('input', () => {
      const valEl = row.querySelector('[data-cs-volume-value]');
      if (valEl) valEl.textContent = `${volSlider.value}%`;
    });
    // Dispatch on release rather than every input — keeps the patch
    // stream sane while the user is dragging.
    volSlider.addEventListener('change', () => {
      const level = Math.max(0, Math.min(1, Number(volSlider.value) / 100));
      _patch(rid, surfaceId, { volume: level });
    });
  }

  const scrubber = row.querySelector('[data-cs-scrubber]');
  if (scrubber) {
    scrubber.addEventListener('mousedown', () => { state.scrubDragging = true; });
    scrubber.addEventListener('touchstart', () => { state.scrubDragging = true; });
    scrubber.addEventListener('change', () => {
      state.scrubDragging = false;
      const pos = Number(scrubber.value);
      if (Number.isFinite(pos)) _patch(rid, surfaceId, { position_s: pos });
    });
  }
}


function _receiverRowHtml(receiver) {
  const rid = receiver.registration_id;
  const name = receiver.label || rid;
  const platform = (receiver.info && receiver.info.platform) || '';
  const showing = _currentlyShowingFor(receiver);

  // Idle row — short header + status. No launch buttons (cast starts
  // from the content surface; the shelf is directory + transport).
  if (!showing) {
    return `
      <div class="cast-shelf-receiver" data-cs-receiver-row="${escapeAttr(rid)}">
        <div class="cast-shelf-receiver-head">
          <span class="cast-shelf-dot"></span>
          <div class="cast-shelf-receiver-meta">
            <div class="cast-shelf-receiver-name">${escapeHtml(name)}</div>
            <div class="cast-shelf-receiver-sub">${escapeHtml(platform || 'idle')}</div>
          </div>
        </div>
      </div>
    `;
  }

  // Active row — full transport card. Uses surface_state echoed back
  // from cast-{audio,video} (position_s / duration_s / paused / volume
  // / muted) when present; absent state collapses the scrubber + volume
  // gracefully.
  const surfaceState = showing.surface_state || {};
  const cur = Number(surfaceState.position_s ?? 0);
  const dur = Number(surfaceState.duration_s ?? 0);
  const paused = _isPausedOn(showing);
  const muted = !!surfaceState.muted;
  const volume = Number(surfaceState.volume);
  const hasVolume = Number.isFinite(volume) && volume >= 0;
  const volumePct = hasVolume ? Math.round(volume * 100) : 0;
  const cap = _capabilityFor(showing);
  const isAudio = cap === 'media.audio_play@1';
  const isImage = cap === 'display.image_show@1';
  const hasScrubber = dur > 0 && !isImage;
  const skipBack = isAudio ? 15 : 10;
  const skipFwd  = isAudio ? 30 : 30;
  const title = _showingTitle(showing, receiver);

  return `
    <div class="cast-shelf-receiver cast-shelf-receiver-active" data-cs-receiver-row="${escapeAttr(rid)}">
      <div class="cast-shelf-receiver-head">
        <span class="cast-shelf-dot active"></span>
        <div class="cast-shelf-receiver-meta">
          <div class="cast-shelf-receiver-name">${escapeHtml(name)}</div>
          <div class="cast-shelf-receiver-sub">${escapeHtml(title)}</div>
        </div>
      </div>
      ${hasScrubber ? `
        <div class="cast-shelf-scrubber-row">
          <span class="cast-shelf-time">${_fmtTime(cur)}</span>
          <input type="range" class="cast-shelf-scrubber" data-cs-scrubber
            min="0" max="${Math.max(1, Math.floor(dur))}" step="1" value="${Math.floor(cur)}">
          <span class="cast-shelf-time">${_fmtTime(dur)}</span>
        </div>
      ` : ''}
      <div class="cast-shelf-transport">
        ${!isImage ? `
          <button class="cast-shelf-tbtn" data-cs-action="skip-back" title="Back ${skipBack}s">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><polyline points="3 3 3 8 8 8"/></svg>
            <span class="cast-shelf-tbtn-label">${skipBack}</span>
          </button>` : ''}
        <button class="cast-shelf-tbtn cast-shelf-tbtn-primary" data-cs-action="play" title="${paused ? 'Resume' : 'Pause'}">
          ${paused
            ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`
            : `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>`}
        </button>
        ${!isImage ? `
          <button class="cast-shelf-tbtn" data-cs-action="skip-forward" title="Forward ${skipFwd}s">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><polyline points="21 3 21 8 16 8"/></svg>
            <span class="cast-shelf-tbtn-label">${skipFwd}</span>
          </button>` : ''}
        <button class="cast-shelf-tbtn cast-shelf-tbtn-stop" data-cs-action="stop" title="Stop cast">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
        </button>
      </div>
      ${hasVolume ? `
        <div class="cast-shelf-volume-row">
          <button class="cast-shelf-mute" data-cs-action="mute" title="${muted ? 'Unmute' : 'Mute'}">
            ${muted
              ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`
              : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>`}
          </button>
          <input type="range" class="cast-shelf-volume" data-cs-volume
            min="0" max="100" step="1" value="${volumePct}" ${muted ? 'disabled' : ''}>
          <span class="cast-shelf-volume-value" data-cs-volume-value>${volumePct}%</span>
        </div>
      ` : ''}
    </div>
  `;
}


/* ── Helpers (active-cast rendering) ──────────────────────────── */


function escapeAttr(s) {
  return String(s ?? '').replace(/[<>&"']/g, (c) => ({
    '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;',
  }[c]));
}


function _showingTitle(showing, receiver) {
  // Surface kind makes the friendly label; receiver name is in the
  // row header so we don't repeat "on TV" here.
  const kindLabel = {
    'media.audio': 'Audio',
    'media.video': 'Video',
    'media.image': 'Image',
    'html.generic': 'Streaming',
  }[showing.surface_kind] || 'Streaming';
  return kindLabel;
}


function _capabilityFor(showing) {
  const url = String(showing.surface_url || '');
  if (showing.surface_kind === 'media.audio' || url.includes('/cast-audio/')) {
    return 'media.audio_play@1';
  }
  if (showing.surface_kind === 'media.video' || url.includes('/cast-video/')) {
    return 'media.video_play@1';
  }
  if (showing.surface_kind === 'media.image') {
    return 'display.image_show@1';
  }
  return '';
}


function _isPausedOn(showing) {
  const ss = showing.surface_state || {};
  if (typeof ss.paused === 'boolean') return ss.paused;
  return false;
}


function _fmtTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}


/* ── Transport dispatch ───────────────────────────────────────── */


/**
 * POST a surface_state patch to /api/cast/send/patch. The cast-receiver
 * applies it locally (cast-receiver.js applyPatch maps paused/position/
 * volume/muted onto the iframe), so any surface that's already echoing
 * state via augmentum.surface_state honours these out of the box.
 */
async function _patch(receiverId, surfaceId, patch) {
  if (!receiverId || !surfaceId || !patch || !Object.keys(patch).length) return;
  try {
    const r = await fetch('/api/cast/send/patch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        receiver_id: receiverId,
        surface_id: surfaceId,
        patch,
      }),
    });
    if (!r.ok) {
      console.warn('[cast-shelf] patch failed', patch, r.status);
    }
  } catch (err) {
    console.warn('[cast-shelf] patch error', err);
  }
  // Quick re-poll so the UI reflects the new state without waiting
  // for the next interval tick.
  setTimeout(() => _refreshTrusted(), 250);
}


async function _stop(receiverId, surfaceId) {
  if (!surfaceId) return;
  try {
    await fetch('/api/cast/send/close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        receiver_id: receiverId,
        surface_id: surfaceId,
      }),
    });
  } catch {
    // Optimistic — re-poll reconciles.
  }
  setTimeout(() => _refreshTrusted(), 250);
}


/* ── Public hooks ─────────────────────────────────────────────── */


/**
 * Called by other modules (cast-or-play.js, cast-picker.js) right after
 * they kick off a cast, so the shelf can refresh immediately rather
 * than waiting for the next poll tick. Previously this lived in
 * cast-remote.js — the export name is preserved for caller stability.
 *
 * Pass ``{ openShelf: true }`` for handoff flows that tear down a
 * local player (e.g. floating-video → "Saved TVs and speakers"). The
 * pill opens automatically so the user lands directly on the new
 * transport controller instead of staring at a closed player.
 */
export function notifyCastStarted({ openShelf = false } = {}) {
  // Both polls — receivers may have just promoted from disconnected to
  // connected, and trusted-receivers has the new active surface.
  setTimeout(() => {
    _refreshReceivers();
    _refreshTrusted();
    if (openShelf) _setOpen(true);
  }, 150);
}
