/**
 * Grove Orb Detach — in-place floating-orb controller.
 *
 * Long-press on #grove-orb-container (≥280ms, ≤6px) promotes it to
 * position: fixed in place. NO DOM reparent — the YouTube iframe is
 * never touched, so playback, seek, and controls remain live.
 *
 * The grove panel hides via a "soft-close" class (visibility: hidden,
 * not display: none) so the orb inside stays in the render tree and
 * escapes via explicit visibility: visible. Because grove-panel has no
 * transform / filter on itself, position: fixed on the orb resolves
 * against the viewport.
 *
 * Exports: init, detach, redock, dismiss, isDetached
 */

import * as ambient from './grove-ambient.js';
import { getCurrentUser } from './auth.js';

// ── Tunables ────────────────────────────────────────────────────────────────
const LONG_PRESS_MS    = 280;
const PRESS_CANCEL_PX  = 6;
const EDGE_SNAP_PX     = 32;
const VIEWPORT_MARGIN  = 8;
const PLUCK_VIBRATE_MS = 15;
// Chrome (dock/close/size buttons) auto-hide timer. Mouse users get a
// short timer because pointermove keeps refreshing it as they navigate
// to a button — chrome effectively stays up the whole time the cursor
// is hovering. Touch users have no continuous hover signal, so a fast
// timer means: tap orb (chrome shows) → lift finger → reposition →
// chrome already faded out before they can land the second tap. The
// touch timer is 4× longer so the user can reach a 28-40px button at
// human speed without racing the fade-out.
const CHROME_HIDE_MS_MOUSE = 2000;
const CHROME_HIDE_MS_TOUCH = 8000;

const SIZES = ['compact', 'standard', 'focus'];
const SIZE_DIAMETER = { compact: 160, standard: 220, focus: 320 };
// Default size on first detach — user can minimize via the size button.
const DEFAULT_SIZE = 'focus';

// Per-user storage keys — orb float geometry (detached state, size, x/y)
// and the first-use hint flag. Global keys would carry Profile A's
// custom orb position into Profile B's session, which is a UI paper-cut
// rather than a privacy leak but worth fixing for parity with the
// other resume modules. `_key()` returns null pre-auth; callers treat
// null as "skip" (same contract as media-resume.js / grove-resume.js).
const STORAGE_KEY_BASE   = 'augmentum-grove-orb-float';
const FIRST_USE_KEY_BASE = 'augmentum-grove-orb-detach-hinted';

function _key(base) {
  const u = getCurrentUser();
  return u && u.id ? `${base}::u:${u.id}` : null;
}

// ── DOM ─────────────────────────────────────────────────────────────────────
let _orbContainer = null;
let _grovePanel   = null;
let _placeholder  = null;
let _placeholderTitle = null;
let _slotRedockBtn = null;
let _dockBtn = null;
let _closeBtn = null;
let _sizeBtn = null;

// ── State ───────────────────────────────────────────────────────────────────
let _detached = false;
let _currentSize = DEFAULT_SIZE;
let _pos = { x: null, y: null };
let _pressTimer = null;
let _pressAnchor = null;
let _dragActive = false;
let _dragOffset = { x: 0, y: 0 };
let _dragTarget = { x: 0, y: 0 };
let _rafId = null;
let _chromeHideTimer = null;
// Last pointer modality — used to choose the chrome-hide timer and to
// decide whether pointerleave should hide chrome instantly (mouse, yes;
// touch, no — leave always fires after a tap-lift on touch and would
// pull the chrome out from under the user's next button tap).
let _lastPointerType = 'mouse';

// ── Persistence ────────────────────────────────────────────────────────────
function _readPersisted() {
  const key = _key(STORAGE_KEY_BASE);
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function _persist() {
  const key = _key(STORAGE_KEY_BASE);
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify({
      detached: _detached,
      size: _currentSize,
      pos: _pos,
    }));
  } catch { /* quota/unavailable — silent */ }
}

// ── Geometry ───────────────────────────────────────────────────────────────
function _clampToViewport(x, y) {
  const d = SIZE_DIAMETER[_currentSize] || 220;
  const maxX = window.innerWidth  - d - VIEWPORT_MARGIN;
  const maxY = window.innerHeight - d - VIEWPORT_MARGIN;
  return {
    x: Math.max(VIEWPORT_MARGIN, Math.min(maxX, x)),
    y: Math.max(VIEWPORT_MARGIN, Math.min(maxY, y)),
  };
}

function _edgeSnap(x, y) {
  const d = SIZE_DIAMETER[_currentSize] || 220;
  const rightEdge  = window.innerWidth  - d - VIEWPORT_MARGIN;
  const bottomEdge = window.innerHeight - d - VIEWPORT_MARGIN;
  if (x - VIEWPORT_MARGIN < EDGE_SNAP_PX) x = VIEWPORT_MARGIN;
  else if (rightEdge - x < EDGE_SNAP_PX)  x = rightEdge;
  if (y - VIEWPORT_MARGIN < EDGE_SNAP_PX) y = VIEWPORT_MARGIN;
  else if (bottomEdge - y < EDGE_SNAP_PX) y = bottomEdge;
  return { x, y };
}

function _defaultPos() {
  const d = SIZE_DIAMETER[_currentSize] || 220;
  return {
    x: window.innerWidth  - d - VIEWPORT_MARGIN - 16,
    y: window.innerHeight - d - VIEWPORT_MARGIN - 16,
  };
}

// ── Positioning — via inline left/top, NOT transform.
//   transform on an iframe ancestor forces the iframe into its own
//   compositor layer, which Chromium re-rasters. YouTube's page-visibility
//   observer interprets that as "hidden" and pauses playback. left/top
//   positioning is slower but reflow-only and keeps the iframe live.
//   rAF batching keeps it smooth enough (capped to monitor refresh).
function _writePosition() {
  if (!_orbContainer) return;
  _orbContainer.style.left = (_pos.x || 0) + 'px';
  _orbContainer.style.top  = (_pos.y || 0) + 'px';
}

function _scheduleFrame() {
  if (_rafId) return;
  _rafId = requestAnimationFrame(() => {
    _rafId = null;
    const { x, y } = _clampToViewport(_dragTarget.x, _dragTarget.y);
    _pos = { x, y };
    _writePosition();
  });
}

// ── Size mode ──────────────────────────────────────────────────────────────
function _applySizeClass(size) {
  if (!SIZES.includes(size)) size = 'standard';
  _currentSize = size;
  if (!_orbContainer) return;
  _orbContainer.classList.remove('size-compact', 'size-standard', 'size-focus');
  if (_detached) {
    _orbContainer.classList.add('size-' + size);
    ambient.setQualityForSize?.(size);
  }
}

// ── Chrome reveal ──────────────────────────────────────────────────────────
function _showChrome() {
  if (!_detached || !_orbContainer) return;
  _orbContainer.classList.add('chrome-visible');
  clearTimeout(_chromeHideTimer);
  const hideMs = _lastPointerType === 'mouse'
    ? CHROME_HIDE_MS_MOUSE
    : CHROME_HIDE_MS_TOUCH;
  _chromeHideTimer = setTimeout(() => {
    _orbContainer?.classList.remove('chrome-visible');
  }, hideMs);
}

function _hideChrome() {
  clearTimeout(_chromeHideTimer);
  _orbContainer?.classList.remove('chrome-visible');
}

// ── Long-press detection ───────────────────────────────────────────────────
/**
 * Returns true if the pointerdown target is a genuinely interactive control
 * that must NOT be claimed for drag/long-press. We deliberately do NOT
 * return true for #grove-orb-hover (the backdrop overlay) — the backdrop
 * between its buttons should be draggable. Only the buttons inside are
 * interactive. Same for #grove-orb-empty (the container) vs its svg (the
 * + icon to open discover).
 */
function _isInteractiveTarget(target) {
  if (!target) return false;
  if (target.closest('#grove-orb-arc-hit'))    return true;  // seek
  if (target.closest('.grove-orb-ctrl'))        return true;  // play/prev/next
  if (target.closest('.grove-orb-float-btn'))   return true;  // chrome
  // Empty-state + button — only when actually clickable (no video yet)
  if (target.closest('#grove-orb-empty') && !_detached) return true;
  return false;
}

function _onPointerDown(e) {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  if (!_orbContainer) return;
  // Remember modality for chrome-hide policy (mouse vs touch/pen). Done
  // BEFORE the interactive-target early-return so a tap on a chrome
  // button still updates the modality — otherwise the first orb tap
  // sets `touch`, then any subsequent button tap leaves it stale.
  if (e.pointerType) _lastPointerType = e.pointerType;
  if (_isInteractiveTarget(e.target)) return;
  if (!_orbContainer.contains(e.target)) return;

  // Suppress native long-press callout, text selection, image drag.
  try { e.preventDefault(); } catch { /* non-cancelable */ }

  _pressAnchor = {
    x: e.clientX,
    y: e.clientY,
    pointerId: e.pointerId,
  };
  _orbContainer.classList.add('pressing');

  if (_detached) {
    // Already floating — ANY pointer-down on the body is the start of a
    // potential drag. We don't start dragging until movement exceeds the
    // cancel threshold (so taps don't jitter the orb). Show chrome as
    // hover feedback so the user sees the buttons.
    _showChrome();
    _dragOffset = {
      x: e.clientX - _pos.x,
      y: e.clientY - _pos.y,
    };
    // No long-press timer here — redock is the dock-button's job. A hold
    // without movement should feel like "I'm about to drag," not "oops,
    // it went home."
  } else {
    // Docked — long-press to detach.
    clearTimeout(_pressTimer);
    _pressTimer = setTimeout(_firePluck, LONG_PRESS_MS);
  }
}

function _onPointerMove(e) {
  if (_dragActive) {
    _dragTarget.x = e.clientX - _dragOffset.x;
    _dragTarget.y = e.clientY - _dragOffset.y;
    _scheduleFrame();
    return;
  }
  if (!_pressAnchor) return;
  const dx = e.clientX - _pressAnchor.x;
  const dy = e.clientY - _pressAnchor.y;
  if (Math.hypot(dx, dy) <= PRESS_CANCEL_PX) return;

  // Movement threshold crossed.
  if (_detached) {
    // Promote press → drag. Use the dragOffset captured in pointerdown so
    // the orb stays under the finger (no jump).
    clearTimeout(_pressTimer);
    _pressTimer = null;
    _orbContainer?.classList.remove('pressing');
    _pressAnchor = null;
    _dragActive = true;
    _dragTarget.x = e.clientX - _dragOffset.x;
    _dragTarget.y = e.clientY - _dragOffset.y;
    _scheduleFrame();
  } else {
    // Docked — movement cancels the long-press (user is sliding off).
    _cancelPress();
  }
}

function _onPointerUp(e) {
  if (_dragActive) return _dragEnd(e);
  _cancelPress();
}

function _cancelPress() {
  clearTimeout(_pressTimer);
  _pressTimer = null;
  _pressAnchor = null;
  _orbContainer?.classList.remove('pressing');
}

function _firePluck() {
  if (!_pressAnchor) return;
  // Only fired from the docked state (long-press to detach).
  try { navigator.vibrate?.(PLUCK_VIBRATE_MS); } catch { /* unsupported */ }
  _orbContainer?.classList.remove('pressing');
  const anchor = _pressAnchor;
  _pressAnchor = null;
  _detachAtPointer(anchor);
}

// ── Detach: no reparent, just promote to position:fixed ───────────────────
function _detachAtPointer(anchor) {
  if (!_orbContainer) return;

  // Measure where the orb is NOW (in the grove panel). That screen rect
  // becomes the orb's initial fixed-position coords, so visually it
  // doesn't jump at the moment of promotion.
  const rect = _orbContainer.getBoundingClientRect();

  // Record where the finger sits inside the orb — drag uses this so the
  // orb doesn't snap to the pointer's position, it stays under the finger.
  _dragOffset = {
    x: anchor.x - rect.left,
    y: anchor.y - rect.top,
  };

  _detached = true;
  _pos = { x: rect.left, y: rect.top };
  // Write inline left/top BEFORE the .detached class kicks in position:fixed,
  // so the orb doesn't flash at (0,0) for one frame.
  _writePosition();
  _applySizeClass(_currentSize);
  _orbContainer.classList.add('detached');
  _grovePanel?.classList.add('orb-is-detached');

  _showPlaceholder();
  _showChrome();
  _maybeMarkFirstUse();
  _persist();

  // Hand off to drag immediately — the user is still holding.
  _dragActive = true;
  _dragTarget = { x: _pos.x, y: _pos.y };

  document.dispatchEvent(new CustomEvent('grove:orb-detached'));
}

// ── Drag ────────────────────────────────────────────────────────────────────
function _dragEnd(e) {
  if (!_dragActive) return;
  _dragActive = false;
  if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
  _pos = _edgeSnap(_pos.x, _pos.y);
  _writePosition();
  _persist();
}

// ── Redock ─────────────────────────────────────────────────────────────────
export function redock() {
  if (!_detached || !_orbContainer) return;

  _orbContainer.classList.remove('detached', 'chrome-visible');
  _orbContainer.classList.remove('size-compact', 'size-standard', 'size-focus');
  _grovePanel?.classList.remove('orb-is-detached');
  // Clear inline position so the orb falls back to its grove-slot layout.
  _orbContainer.style.left = '';
  _orbContainer.style.top  = '';
  _orbContainer.style.transform = '';  // clear legacy transform if any

  _detached = false;
  _hidePlaceholder();
  _persist();

  document.dispatchEvent(new CustomEvent('grove:orb-redocked'));
}

export function dismiss() {
  if (!_detached) return;
  redock();
  document.dispatchEvent(new CustomEvent('grove:orb-dismiss-requested'));
}

// ── Programmatic detach ────────────────────────────────────────────────────
export function detach() {
  if (_detached || !_orbContainer) return;
  const rect = _orbContainer.getBoundingClientRect();
  _detachAtPointer({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
  _dragActive = false;
  requestAnimationFrame(() => {
    _pos = _defaultPos();
    _writePosition();
    _persist();
  });
}

export function isDetached() { return _detached; }

// ── Placeholder ────────────────────────────────────────────────────────────
function _showPlaceholder() {
  if (!_placeholder) return;
  _placeholder.hidden = false;
  const state = ambient.getState?.();
  const title = state?.currentVideo?.title || 'Ambient video';
  if (_placeholderTitle) _placeholderTitle.textContent = title;
}

function _hidePlaceholder() {
  if (_placeholder) _placeholder.hidden = true;
}

// ── First-use hint ─────────────────────────────────────────────────────────
function _maybeMarkFirstUse() {
  const key = _key(FIRST_USE_KEY_BASE);
  if (!key) return;
  try { localStorage.setItem(key, '1'); } catch { /* silent */ }
}

function _maybeShowPreDetachHint() {
  const key = _key(FIRST_USE_KEY_BASE);
  // No user yet → suppress the hint rather than risk showing it twice
  // to the same user (once anonymously, once after auth). It'll fire on
  // the next idle poll once auth has resolved.
  if (!key) return;
  try {
    if (localStorage.getItem(key) === '1') return;
  } catch { return; }
  const state = ambient.getState?.();
  if (!state?.currentVideo) return;
  const host = document.getElementById('grove-ambient-section');
  if (!host || host.querySelector('.grove-orb-hint-tip')) return;
  const tip = document.createElement('div');
  tip.className = 'grove-orb-hint-tip';
  tip.textContent = 'Hold the orb to pop it out';
  host.appendChild(tip);
  requestAnimationFrame(() => tip.classList.add('visible'));
  setTimeout(() => {
    tip.classList.remove('visible');
    setTimeout(() => tip.remove(), 400);
    try { localStorage.setItem(key, '1'); } catch { /* silent */ }
  }, 4000);
}

// ── Chrome actions ─────────────────────────────────────────────────────────
function _cycleSize() {
  const idx = SIZES.indexOf(_currentSize);
  const next = SIZES[(idx + 1) % SIZES.length];
  _applySizeClass(next);
  _pos = _clampToViewport(_pos.x, _pos.y);
  _writePosition();
  _persist();
  _showChrome();
}

function _wireChrome() {
  _dockBtn?.addEventListener('click',  e => { e.stopPropagation(); redock(); });
  _closeBtn?.addEventListener('click', e => { e.stopPropagation(); dismiss(); });
  _sizeBtn?.addEventListener('click',  e => { e.stopPropagation(); _cycleSize(); });
  _slotRedockBtn?.addEventListener('click', e => { e.stopPropagation(); redock(); });

  // Double-click orb body → size cycle
  let lastTap = 0;
  _orbContainer?.addEventListener('click', e => {
    if (!_detached) return;
    if (_isInteractiveTarget(e.target)) return;
    const now = Date.now();
    if (now - lastTap < 300) { _cycleSize(); lastTap = 0; }
    else lastTap = now;
  });

  // Reveal chrome on pointer activity. Also track pointerType from
  // hover/move events so chrome timing reflects the actual modality
  // even before a pointerdown lands.
  _orbContainer?.addEventListener('pointermove', (e) => {
    if (e.pointerType) _lastPointerType = e.pointerType;
    if (_detached) _showChrome();
  });
  // Mouse leaving the orb area is a deliberate signal that the user
  // moved on — hide chrome immediately. On touch/pen, ``pointerleave``
  // ALSO fires the moment the finger lifts after a tap, which would
  // pull the chrome out before the user can land their second tap on
  // a chrome button. So for non-mouse modalities we let the timer
  // (8s on touch) handle the hide instead.
  _orbContainer?.addEventListener('pointerleave', (e) => {
    if (!_detached) return;
    const type = e.pointerType || _lastPointerType;
    if (type === 'mouse') _hideChrome();
  });
  _orbContainer?.addEventListener('focusin',      () => { if (_detached) _showChrome(); });
}

function _wireKeyboard() {
  document.addEventListener('keydown', e => {
    if (!_detached) return;
    if (e.key === 'Escape' && _orbContainer?.contains(document.activeElement)) {
      e.preventDefault();
      dismiss();
    }
  });
}

function _wireResize() {
  window.addEventListener('resize', () => {
    if (!_detached) return;
    _pos = _clampToViewport(_pos.x ?? _defaultPos().x, _pos.y ?? _defaultPos().y);
    _writePosition();
  });
}

// ── Init ────────────────────────────────────────────────────────────────────
export function init() {
  _orbContainer = document.getElementById('grove-orb-container');
  _grovePanel   = document.getElementById('grove-panel');
  _placeholder  = document.getElementById('grove-orb-slot-placeholder');
  _placeholderTitle = document.getElementById('grove-orb-slot-title');
  _slotRedockBtn = document.getElementById('grove-orb-slot-redock');
  _dockBtn  = document.getElementById('grove-orb-float-dock');
  _closeBtn = document.getElementById('grove-orb-float-close');
  _sizeBtn  = document.getElementById('grove-orb-float-size');

  if (!_orbContainer) {
    console.warn('[grove-orb-detach] missing #grove-orb-container; disabled');
    return;
  }

  const saved = _readPersisted();
  if (saved?.size && SIZES.includes(saved.size)) _currentSize = saved.size;

  // pointerdown scoped to the orb (non-passive so we can preventDefault
  // and suppress iOS callout / text selection on long-press).
  _orbContainer.addEventListener('pointerdown', _onPointerDown, { passive: false });
  // Move/up on document so we keep receiving events during drag regardless
  // of where the pointer ends up.
  document.addEventListener('pointermove',   _onPointerMove, { passive: true });
  document.addEventListener('pointerup',     _onPointerUp,   { passive: true });
  document.addEventListener('pointercancel', _onPointerUp,   { passive: true });

  _wireChrome();
  _wireKeyboard();
  _wireResize();

  // Restore detached state on reload. Skip if no current video — a stranded
  // detached flag with no content is never a useful restoration.
  if (saved?.detached) {
    const st = ambient.getState?.();
    if (st?.currentVideo) {
      requestAnimationFrame(() => {
        _detached = true;
        _pos = saved.pos && typeof saved.pos.x === 'number'
          ? _clampToViewport(saved.pos.x, saved.pos.y)
          : _defaultPos();
        _applySizeClass(_currentSize);
        _orbContainer.classList.add('detached');
        _grovePanel?.classList.add('orb-is-detached');
        // On restore, the grove panel was closed at last save. Set it to
        // soft-closed so the orb (inside the panel's DOM) stays rendered.
        if (_grovePanel) {
          _grovePanel.classList.add('visible', 'soft-closed');
        }
        _writePosition();
        _showPlaceholder();
        document.dispatchEvent(new CustomEvent('grove:orb-detached', {
          detail: { restored: true },
        }));
      });
    } else {
      // Clear stale flag
      _persist();
    }
  }

  // First-use hint — once the user has loaded a video and left it idle for
  // a few seconds, surface the "Hold the orb to pop it out" tooltip.
  let polls = 0;
  const hintInterval = setInterval(() => {
    polls++;
    const st = ambient.getState?.();
    if (st?.currentVideo && !_detached) {
      _maybeShowPreDetachHint();
      clearInterval(hintInterval);
    } else if (polls > 20) {
      clearInterval(hintInterval);
    }
  }, 1500);
}
