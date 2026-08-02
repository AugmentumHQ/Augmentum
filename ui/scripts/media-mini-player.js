/**
 * media-mini-player.js — Bottom-docked persistent player bar.
 *
 * Rendered once at app init into ``#app-media-mini-player-host``. Hides
 * itself when nothing is playing. Shows whenever the media-player
 * singleton has a loaded item — across every navigation. Clicking the
 * cover or title opens the detail panel for that row; the close button
 * clears the player.
 *
 * Kept deliberately minimal — the full detail surface (chapter list,
 * sleep timer, speed) lives in the detail panel. The mini-player is
 * the "what's currently playing" ambient state, not a full UI.
 */

import { escapeHtml } from './app.js';
import {
  subscribe, getState, toggle, skipForward, skipBackward, seek, close,
  setSpeed, setSleepTimer, sleepTimerRemainingMs,
} from './media-player.js';
import { openCastPicker } from './cast-picker.js';

// Sleep timer preset menu — matches Audible / Libby muscle memory, plus
// "end of chapter" which is the most requested option in audiobook UX.
const SLEEP_PRESETS = [
  { label: 'Off',              value: 0 },
  { label: '15 minutes',       value: 15 },
  { label: '30 minutes',       value: 30 },
  { label: '45 minutes',       value: 45 },
  { label: '60 minutes',       value: 60 },
  { label: 'End of chapter',   value: 'end-of-chapter' },
];
// Speed presets mirror Audible: heavy-stepped below 1× (rare), fine-
// grained above (common listening range). Users can still slide in 0.05
// increments between presets.
const SPEED_PRESETS = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0];

const HOST_ID = 'media-mini-player-host';

let _root = null;
let _orb = null;
let _scrubber = null;
let _scrubberDragging = false;
let _unsubscribe = null;

// Session-level minimized preference so the orb persists across tab
// navigations but resets on a genuine reload (matches Grove / voice
// semantics — ambient state, not a persistent layout commitment).
const MIN_KEY = 'augmentum.mediaPlayer.minimized';
function _loadMinimized() { return sessionStorage.getItem(MIN_KEY) === '1'; }
function _saveMinimized(v) {
  try { sessionStorage.setItem(MIN_KEY, v ? '1' : '0'); } catch { /* private mode */ }
}

// Press-and-hold drag positions, persisted per-element so the bar and
// the orb remember independent placement. Session-scoped (resets on a
// real reload, like the minimize flag) — keeps the default "centered
// bottom" layout authoritative and treats moves as temporary.
const _POS_KEYS = { bar: 'augmentum.mediaPlayer.barPos', orb: 'augmentum.mediaPlayer.orbPos' };
function _loadPos(kind) {
  try {
    const raw = sessionStorage.getItem(_POS_KEYS[kind]);
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (typeof p?.x === 'number' && typeof p?.y === 'number') return p;
  } catch { /* */ }
  return null;
}
function _savePos(kind, pos) {
  try {
    if (pos) sessionStorage.setItem(_POS_KEYS[kind], JSON.stringify(pos));
    else sessionStorage.removeItem(_POS_KEYS[kind]);
  } catch { /* */ }
}

export function initMediaMiniPlayer() {
  const host = document.getElementById(HOST_ID) || _ensureHost();
  host.innerHTML = _htmlShell();
  _root = host.querySelector('.mini-player');
  _orb  = host.querySelector('.mini-player-orb');
  _wire();
  const _gripEl = _root.querySelector('.mini-player-grip');
  _wireDrag(_root, 'bar', _root.querySelectorAll('.mini-player-cover, .mini-player-meta'), {
    instantHandles: _gripEl ? [_gripEl] : [],
  });
  _wireDrag(_orb, 'orb', [_orb]);   // whole orb is a grab handle
  _restorePosition('bar', _root);
  _restorePosition('orb', _orb);
  // Re-clamp on resize so a stashed position doesn't end up offscreen
  // when the user rotates a phone or resizes the window.
  window.addEventListener('resize', () => {
    _restorePosition('bar', _root);
    _restorePosition('orb', _orb);
  });
  _unsubscribe = subscribe(_render);
}

// Press-and-hold drag. ``el`` is the element being moved; ``handles``
// is the list of regions that act as drag handles (so we don't hijack
// clicks on play/skip/close buttons). Press ≥350ms without lifting
// engages drag; release without engaging passes through as a click.
//
// During drag the element is positioned via top/left in pixels — we
// override the default bottom/right/margin layout so it can sit anywhere.
// Persisted to session storage per ``kind`` and re-clamped on resize.
const _PRESS_MS = 350;
const _MOVE_THRESHOLD_PX = 6;
function _wireDrag(el, kind, handles, opts = {}) {
  if (!el) return;
  const instantHandles = opts.instantHandles || [];
  let pressTimer = null;
  let pointerId = null;
  let startX = 0, startY = 0;
  let elStartX = 0, elStartY = 0;
  let engaged = false;
  let pendingHandle = null;

  function _isInstantHandle(target) {
    if (!target || !instantHandles.length) return false;
    for (const h of instantHandles) {
      if (h === target || h.contains(target)) return true;
    }
    return false;
  }

  function _isHandle(target) {
    if (!target) return false;
    // Block drag if the press started on a button / control. Those
    // need to stay tappable without a 350ms surprise drag.
    if (target.closest('button, input, [role="button"][data-action]:not(.mini-player-cover):not(.mini-player-meta)')) {
      return false;
    }
    for (const h of handles) {
      if (h === target || h.contains(target)) return true;
    }
    return false;
  }

  function _engage() {
    engaged = true;
    pressTimer = null;
    el.classList.add('mp-dragging');
    document.body.classList.add('mp-drag-active');
  }

  function _clamp(x, y) {
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    return {
      x: Math.max(8, Math.min(window.innerWidth - w - 8, x)),
      y: Math.max(8, Math.min(window.innerHeight - h - 8, y)),
    };
  }

  el.addEventListener('pointerdown', (e) => {
    if (e.button && e.button !== 0) return;   // ignore right/middle click
    const instant = _isInstantHandle(e.target);
    if (!instant && !_isHandle(e.target)) return;
    pendingHandle = e.target;
    pointerId = e.pointerId;
    startX = e.clientX;
    startY = e.clientY;
    const rect = el.getBoundingClientRect();
    elStartX = rect.left;
    elStartY = rect.top;
    engaged = false;
    // Capture immediately for both paths. The press-hold path needs it
    // too: without an early capture, Chrome on Android can hand the
    // pointer to its scroll machinery during the 350ms hold (slight
    // finger tremor counts as movement) and our pointermove never
    // fires when the timer eventually completes — the visible bug was
    // "drag works for a moment then dies". Capturing on down pins the
    // pointer to el from the start; the press-hold + move-threshold
    // logic in pointermove still distinguishes tap from drag, so the
    // click pathway is unaffected.
    try { el.setPointerCapture(e.pointerId); } catch { /* */ }
    if (instant) {
      // Visible grip pill — engage immediately so a single tap-and-drag
      // moves the player. No 350ms wait, no chance of a stray scroll
      // canceling the press timer.
      _engage();
      e.preventDefault();
    } else {
      pressTimer = setTimeout(_engage, _PRESS_MS);
    }
  });

  el.addEventListener('pointermove', (e) => {
    if (pointerId !== e.pointerId) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (!engaged) {
      // Cancel the press if the pointer moves before the hold fires —
      // treat as a scroll/swipe, not a drag.
      if (Math.abs(dx) > _MOVE_THRESHOLD_PX || Math.abs(dy) > _MOVE_THRESHOLD_PX) {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
        pointerId = null;
      }
      return;
    }
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch { /* */ }
    const pos = _clamp(elStartX + dx, elStartY + dy);
    _applyPos(el, pos);
  });

  function _end(e) {
    if (pointerId !== e.pointerId) return;
    if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    try { el.releasePointerCapture(e.pointerId); } catch { /* */ }
    pointerId = null;
    if (engaged) {
      el.classList.remove('mp-dragging');
      document.body.classList.remove('mp-drag-active');
      const rect = el.getBoundingClientRect();
      _savePos(kind, { x: rect.left, y: rect.top });
      // Suppress the synthetic click that follows pointerup — the
      // press-and-hold was intent to move, not to open.
      const blocker = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      el.addEventListener('click', blocker, { capture: true, once: true });
      engaged = false;
    }
  }
  el.addEventListener('pointerup', _end);
  el.addEventListener('pointercancel', _end);

  // Double-click on a handle resets to default position. Discoverable
  // enough for an escape hatch; not so common as to surprise.
  el.addEventListener('dblclick', (e) => {
    if (!_isHandle(e.target) && !_isInstantHandle(e.target)) return;
    _savePos(kind, null);
    _resetPosition(el);
  });
}

function _applyPos(el, pos) {
  el.style.left = `${pos.x}px`;
  el.style.top = `${pos.y}px`;
  el.style.right = 'auto';
  el.style.bottom = 'auto';
  el.style.margin = '0';
  el.classList.add('mp-moved');
}

function _resetPosition(el) {
  el.style.left = '';
  el.style.top = '';
  el.style.right = '';
  el.style.bottom = '';
  el.style.margin = '';
  el.classList.remove('mp-moved');
}

function _restorePosition(kind, el) {
  if (!el) return;
  const pos = _loadPos(kind);
  if (!pos) { _resetPosition(el); return; }
  // Element may not have measurable size yet if hidden; defer to next
  // frame so offsetWidth/Height are valid for clamping.
  requestAnimationFrame(() => {
    if (!el.offsetWidth || !el.offsetHeight) {
      _applyPos(el, pos);   // best-effort; will re-clamp on resize
      return;
    }
    const clamped = {
      x: Math.max(8, Math.min(window.innerWidth - el.offsetWidth - 8, pos.x)),
      y: Math.max(8, Math.min(window.innerHeight - el.offsetHeight - 8, pos.y)),
    };
    _applyPos(el, clamped);
  });
}

function _ensureHost() {
  // Fallback mount so the module survives pages that didn't pre-provision
  // the host div (tests, standalone demos). Real app puts the host in
  // index.html so we don't have to worry about layering.
  const h = document.createElement('div');
  h.id = HOST_ID;
  document.body.appendChild(h);
  return h;
}

function _htmlShell() {
  return `
    <div class="mini-player-orb hidden" role="button" tabindex="0"
         aria-label="Media playing — click to expand player">
      <div class="mp-orb-cover"></div>
      <div class="mp-orb-ring"></div>
      <div class="mp-orb-eq" aria-hidden="true">
        <span></span><span></span><span></span><span></span>
      </div>
    </div>
    <div class="mini-player hidden" role="region" aria-label="Media player">
      <button class="mini-player-grip" type="button" data-mp-grip
              title="Drag to move" aria-label="Drag to move player"></button>
      <button class="mini-player-cover" type="button" data-action="expand" title="Open details">
        <img alt="" class="mini-player-cover-img">
        <span class="mini-player-cover-fallback" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        </span>
      </button>
      <div class="mini-player-meta" role="button" tabindex="0" data-action="expand">
        <div class="mini-player-title"></div>
        <div class="mini-player-subtitle"></div>
      </div>
      <div class="mini-player-controls">
        <button type="button" class="mp-btn" data-action="skip-back" title="Back 15s" aria-label="Skip back 15 seconds">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><polyline points="3 3 3 8 8 8"/></svg>
          <span class="mp-btn-label">15</span>
        </button>
        <button type="button" class="mp-btn mp-play" data-action="toggle" aria-label="Play or pause">
          <svg class="mp-icon-play"  width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          <svg class="mp-icon-pause" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>
        </button>
        <button type="button" class="mp-btn" data-action="skip-forward" title="Forward 30s" aria-label="Skip forward 30 seconds">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><polyline points="21 3 21 8 16 8"/></svg>
          <span class="mp-btn-label">30</span>
        </button>
        <button type="button" class="mp-btn mp-cast-btn" data-action="cast" title="Cast to a device" aria-label="Cast to a device">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 16.1A5 5 0 0 1 5.9 20"/>
            <path d="M2 12.05A9 9 0 0 1 9.95 20"/>
            <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
            <line x1="2" y1="20" x2="2.01" y2="20"/>
          </svg>
        </button>
        <button type="button" class="mp-btn" data-action="playlist" title="Add to Grove playlist" aria-label="Add to Grove playlist">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>
        </button>
        <button type="button" class="mp-chip mp-chip-speed" data-action="toggle-speed" title="Playback speed" aria-haspopup="true">
          <span class="mp-chip-value">1.0×</span>
        </button>
        <button type="button" class="mp-chip mp-chip-sleep" data-action="toggle-sleep" title="Sleep timer" aria-haspopup="true">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          <span class="mp-chip-value mp-chip-sleep-value"></span>
        </button>
      </div>
      <div class="mini-player-scrubber">
        <span class="mini-player-time mp-time-current">0:00</span>
        <input type="range" class="mini-player-range" min="0" max="1000" step="1" value="0" aria-label="Seek">
        <span class="mini-player-time mp-time-total">0:00</span>
      </div>
      <button class="mini-player-minimize" type="button" data-action="minimize" title="Collapse to orb" aria-label="Collapse to orb">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14"/></svg>
      </button>
      <button class="mini-player-close" type="button" data-action="close" title="Close player" aria-label="Close player">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
      <div class="mp-popover" data-popover="speed" hidden>
        <div class="mp-popover-title">Playback speed</div>
        <div class="mp-popover-speed-grid" data-speed-grid></div>
        <input type="range" class="mp-popover-speed-range" min="50" max="300" step="5" value="100"
               data-speed-range aria-label="Playback speed">
        <div class="mp-popover-speed-value" data-speed-readout>1.00×</div>
      </div>
      <div class="mp-popover" data-popover="sleep" hidden>
        <div class="mp-popover-title">Sleep timer</div>
        <ul class="mp-popover-list">
          ${SLEEP_PRESETS.map(p => `
            <li>
              <button class="mp-popover-item" type="button" data-sleep-value="${escapeHtml(String(p.value))}">
                ${escapeHtml(p.label)}
              </button>
            </li>`).join('')}
        </ul>
        <div class="mp-popover-foot" data-sleep-status></div>
      </div>
    </div>
  `;
}

function _wire() {
  if (!_root) return;
  _scrubber = _root.querySelector('.mini-player-range');

  _root.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (btn) {
      const action = btn.dataset.action;
      if (action === 'toggle')              toggle();
      else if (action === 'skip-back')      skipBackward();
      else if (action === 'skip-forward')   skipForward();
      else if (action === 'expand')         _expandToDetail();
      else if (action === 'close')          close();
      else if (action === 'toggle-speed')   _togglePopover('speed');
      else if (action === 'toggle-sleep')   _togglePopover('sleep');
      else if (action === 'minimize')       _setMinimized(true);
      else if (action === 'cast')           _openCastPicker(btn);
      else if (action === 'playlist') {
        const st = getState();
        if (st?.fileId) {
          window.dispatchEvent(new CustomEvent('playlist:add-item', {
            detail: {
              type: 'file',
              fileId: st.fileId,
              name: st.title || '',
              kind: 'audio',
              thumbnail: st.coverUrl || '',
            },
          }));
        }
      }
      else if (action === 'filter-author') {
        // "More by this author" — opens Files at the audiobooks chip
        // with the author seeded into search. Stop propagation so the
        // outer meta's data-action="expand" doesn't ALSO fire (which
        // would yank the user to the file's detail row instead).
        e.stopPropagation();
        const author = btn.dataset.author || '';
        if (author) {
          window.dispatchEvent(new CustomEvent('files:open-with-filter', {
            detail: { chip: 'audiobooks', search: author },
          }));
        }
      }
      return;
    }
    const sleepBtn = e.target.closest('[data-sleep-value]');
    if (sleepBtn) {
      const v = sleepBtn.dataset.sleepValue;
      setSleepTimer(v === 'end-of-chapter' ? 'end-of-chapter' : Number(v));
      _closePopovers();
      return;
    }
    const speedBtn = e.target.closest('[data-speed-preset]');
    if (speedBtn) {
      setSpeed(Number(speedBtn.dataset.speedPreset));
    }
  });
  // Close popovers when clicking outside the player.
  document.addEventListener('click', (e) => {
    if (!_root || !e.target) return;
    if (!_root.contains(e.target)) _closePopovers();
  });

  // Orb click → restore the full bar. Keyboard: Enter/Space on the orb
  // (role=button) does the same so this stays accessible.
  const orbClick = () => _setMinimized(false);
  _orb.addEventListener('click', orbClick);
  _orb.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); orbClick(); }
  });

  // Speed grid (static, built once) and the slider.
  const speedGrid = _root.querySelector('[data-speed-grid]');
  if (speedGrid) {
    speedGrid.innerHTML = SPEED_PRESETS.map(s =>
      `<button type="button" class="mp-speed-preset" data-speed-preset="${s}">${s.toFixed(2).replace(/\.?0+$/, '')}×</button>`
    ).join('');
  }
  const speedRange = _root.querySelector('[data-speed-range]');
  speedRange?.addEventListener('input', () => {
    setSpeed(Number(speedRange.value) / 100);
  });
  _root.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const btn = e.target.closest('[data-action="expand"]');
    if (!btn) return;
    e.preventDefault();
    _expandToDetail();
  });

  // Scrubber interactions — pointer down/up for drag semantics so we
  // don't thrash seek while dragging across the bar. Progress updates
  // from timeupdate are suppressed during drag to prevent jitter.
  _scrubber.addEventListener('pointerdown', () => { _scrubberDragging = true; });
  _scrubber.addEventListener('pointerup',   () => { _scrubberDragging = false; _commitScrubber(); });
  _scrubber.addEventListener('change',       _commitScrubber);
  // Also accept keyboard seeking (tab + arrow keys on the range).
  _scrubber.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.stopPropagation();
    }
  });
}

function _commitScrubber() {
  const st = getState();
  if (!st.durationS) return;
  const frac = Number(_scrubber.value || 0) / 1000;
  seek(frac * st.durationS);
}

function _expandToDetail() {
  const st = getState();
  if (!st.fileId) return;
  // Detail panel opens via a lightweight custom event; the Files panel
  // listens and opens its detail view for the given id. Keeps this
  // module decoupled from the Files DOM / state store.
  window.dispatchEvent(new CustomEvent('media-player:expand', {
    detail: { fileId: st.fileId },
  }));
}

function _togglePopover(name) {
  const target = _root.querySelector(`.mp-popover[data-popover="${name}"]`);
  if (!target) return;
  const wasHidden = target.hasAttribute('hidden');
  _closePopovers();
  if (wasHidden) target.removeAttribute('hidden');
}


function _openCastPicker(anchor) {
  const st = getState();
  if (!st.fileId) return;
  // The mini-player handles audio (audiobooks, music, LibriVox). Filter
  // devices to those supporting audio playback. We pass the current
  // stream URL + position so the user can resume on the new device from
  // wherever they are locally.
  openCastPicker({
    anchor,
    capability: 'media.audio_play@1',
    content: {
      contentUrl: st.streamUrl,
      title: st.title || 'Audio',
      author: st.author || '',
      artist: st.author || st.narrator || '',
      posterUrl: st.coverUrl,
      startTimeS: Math.max(0, Number(st.currentTimeS) || 0),
      fileId: st.fileId,
      contentKey: st.fileId,
    },
  });
}

function _closePopovers() {
  _root.querySelectorAll('.mp-popover').forEach(p => p.setAttribute('hidden', ''));
}

// Minimize transitions between the full bar and a Grove-style orb.
// The orb has a circular cover, animated equalizer bars, and a glow
// ring tinted by the accent — matches the voice pipeline's ambient
// aesthetic so audio always has the same visual language in this app.
function _setMinimized(minimized) {
  _saveMinimized(minimized);
  _render(getState());  // re-drive visibility from the fresh flag
}

let _sleepCountdownHandle = null;

function _render(state) {
  if (!_root) return;
  const active = !!state.fileId;
  const minimized = active && _loadMinimized();

  // Top-level visibility: bar visible when active AND not minimized;
  // orb visible when active AND minimized; both hidden otherwise.
  _root.classList.toggle('hidden', !active || minimized);
  _orb?.classList.toggle('hidden', !active || !minimized);

  // Body class so sibling mini-docks (comic reader, future video
  // reader) can stack cleanly above the audio bar instead of
  // overlapping it. Only the full bar counts — the orb is a corner
  // surface and doesn't compete for horizontal space.
  document.body.classList.toggle('media-mini-active', active && !minimized);

  // Keep orb state-synced even when it's not the visible surface —
  // the cover + playing animation should reflect the same state so
  // the transition back to full bar is seamless.
  if (_orb && active) {
    _orb.classList.toggle('is-playing', state.isPlaying);
    _orb.classList.toggle('is-loading', state.isLoading);
    const coverEl = _orb.querySelector('.mp-orb-cover');
    if (state.coverUrl && coverEl) {
      coverEl.style.backgroundImage = `url("${state.coverUrl}")`;
    } else if (coverEl) {
      coverEl.style.backgroundImage = '';
    }
    _orb.setAttribute('title',
      state.title ? `${state.isPlaying ? 'Playing' : 'Paused'}: ${state.title}` : 'Media playing'
    );
  }

  if (!active || minimized) return;

  // --- Cover + title block ---
  const coverImg = _root.querySelector('.mini-player-cover-img');
  if (state.coverUrl && coverImg.src !== state.coverUrl) {
    coverImg.src = state.coverUrl;
  }
  _root.querySelector('.mini-player-title').textContent = state.title || 'Untitled';
  // Subtitle = author + current chapter, with the author surfaced as a
  // click-through to Files filtered by author. Chapter title stays
  // plain — it's a navigational *position* rather than a *concept*
  // worth filtering on. innerHTML is built from escaped pieces so
  // titles with HTML metacharacters can't break out of the structure.
  const subEl = _root.querySelector('.mini-player-subtitle');
  if (subEl) {
    const parts = [];
    if (state.author) {
      const a = escapeHtml(state.author);
      parts.push(`<button type="button" class="mini-player-author-link" data-action="filter-author" data-author="${a}" title="More by ${a}">${a}</button>`);
    }
    const ch = state.chapters[state.currentChapterIdx];
    if (ch?.title) parts.push(`<span class="mini-player-chapter">${escapeHtml(ch.title)}</span>`);
    subEl.innerHTML = parts.join(' <span class="mini-player-sep" aria-hidden="true">·</span> ');
  }

  // --- Play/pause icon swap ---
  _root.classList.toggle('is-playing', state.isPlaying);
  _root.classList.toggle('is-loading', state.isLoading);

  // --- Scrubber ---
  if (!_scrubberDragging) {
    const frac = state.durationS ? state.currentTimeS / state.durationS : 0;
    _scrubber.value = String(Math.round(frac * 1000));
  }
  _root.querySelector('.mp-time-current').textContent = _fmtTime(state.currentTimeS);
  _root.querySelector('.mp-time-total').textContent   = _fmtTime(state.durationS);

  // --- Speed chip + popover sync ---
  const speedVal = _root.querySelector('.mp-chip-speed .mp-chip-value');
  if (speedVal) speedVal.textContent = `${state.speed.toFixed(2).replace(/\.?0+$/, '')}×`;
  const speedRange = _root.querySelector('[data-speed-range]');
  if (speedRange) speedRange.value = String(Math.round(state.speed * 100));
  const speedReadout = _root.querySelector('[data-speed-readout]');
  if (speedReadout) speedReadout.textContent = `${state.speed.toFixed(2)}×`;
  _root.querySelectorAll('.mp-speed-preset').forEach(b => {
    b.classList.toggle('is-active', Math.abs(Number(b.dataset.speedPreset) - state.speed) < 0.001);
  });
  _root.querySelector('.mp-chip-speed')?.classList.toggle('is-active', Math.abs(state.speed - 1) > 0.001);

  // --- Sleep timer chip ---
  const sleepChip = _root.querySelector('.mp-chip-sleep');
  const sleepVal  = _root.querySelector('.mp-chip-sleep-value');
  const sleepStatus = _root.querySelector('[data-sleep-status]');
  const timerActive = state.sleepTimerEndOfChapter || state.sleepTimerMs > 0;
  sleepChip?.classList.toggle('is-active', timerActive);
  if (timerActive) {
    if (state.sleepTimerEndOfChapter) {
      if (sleepVal) sleepVal.textContent = 'EOC';
      if (sleepStatus) sleepStatus.textContent = 'Stopping at end of current chapter.';
    } else {
      const remainingMs = sleepTimerRemainingMs() ?? 0;
      const mins = Math.ceil(remainingMs / 60000);
      if (sleepVal) sleepVal.textContent = `${mins}m`;
      if (sleepStatus) sleepStatus.textContent = `Stops in ${mins} min${mins === 1 ? '' : 's'}.`;
    }
    // Light 5s-period refresh so the countdown stays honest even when
    // timeupdate events aren't firing (e.g. user paused).
    if (!_sleepCountdownHandle) {
      _sleepCountdownHandle = setInterval(() => _render(getState()), 5000);
    }
  } else {
    if (sleepVal) sleepVal.textContent = '';
    if (sleepStatus) sleepStatus.textContent = '';
    if (_sleepCountdownHandle) {
      clearInterval(_sleepCountdownHandle);
      _sleepCountdownHandle = null;
    }
  }
  _root.querySelectorAll('[data-sleep-value]').forEach(b => {
    const v = b.dataset.sleepValue;
    const matches =
      (v === '0' && !timerActive) ||
      (v === 'end-of-chapter' && state.sleepTimerEndOfChapter) ||
      (Number(v) > 0 && state.sleepTimerMs === Number(v) * 60_000);
    b.classList.toggle('is-active', matches);
  });
}

function _fmtTime(totalSec) {
  const s = Math.max(0, Math.floor(totalSec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h
    ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
    : `${m}:${String(r).padStart(2, '0')}`;
}
