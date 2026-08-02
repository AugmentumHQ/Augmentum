/**
 * draggable-fab.js — shared drag-anywhere behavior for floating
 * action buttons.
 *
 * Usage:
 *   import { makeDraggable } from './draggable-fab.js';
 *   makeDraggable(myButtonEl, {
 *     storageKey: 'augmentum.fab.myButton',
 *     onMove:     (rect) => { ... },   // optional, fires every frame during drag
 *     onDrop:     (rect) => { ... },   // optional, fires once on release
 *   });
 *
 * Pointer-events-based so touch and mouse take the same path. A tap
 * that moves less than ``dragThresholdPx`` falls through to the
 * element's native click handler; anything beyond it switches into
 * drag mode, captures the pointer, and suppresses the synthesised
 * click on release.
 *
 * Position persists to localStorage as ``{x: cssLeftPx, y: cssTopPx}``
 * keyed by ``storageKey``. The element is clamped to the viewport on
 * both drop AND on window resize so a previously-saved position that
 * lands off-screen (rotation, browser-window resize, smaller TV) gets
 * pulled back into view automatically.
 *
 * Important: the element is repositioned from its CSS-default
 * right/bottom anchors to left/top once a position is applied. CSS
 * authors can keep using right/bottom for the default — this module
 * only switches when there's a stored position or a real drag.
 */

const DEFAULT_DRAG_THRESHOLD_PX = 6;
// Touch + pen need a larger threshold than mouse — a finger tap on a
// small target easily wanders >6px even when the user means it as a
// click, which used to translate to "pip teleports a few pixels every
// tap". 12px is roughly the iOS / Material touch slop convention.
const DEFAULT_TOUCH_DRAG_THRESHOLD_PX = 12;

export function makeDraggable(el, opts = {}) {
  if (!el || !opts.storageKey) return;
  const storageKey = String(opts.storageKey);
  const mouseThreshold = opts.dragThresholdPx ?? DEFAULT_DRAG_THRESHOLD_PX;
  const touchThreshold = opts.touchDragThresholdPx ?? DEFAULT_TOUCH_DRAG_THRESHOLD_PX;

  // Apply persisted position if present. Deferred a frame so the
  // element has its computed size for clamping.
  _applyStored(el, storageKey);

  let pointerId = null;
  let activeThreshold = mouseThreshold;
  let startX = 0, startY = 0;     // pointer position at drag-start
  let elStartX = 0, elStartY = 0; // element top-left at drag-start
  let moved = false;
  // Tracks whether THIS pointerdown sequence ended in a real drag.
  // Sticks past pointerup so the synthesized click event (which fires
  // AFTER pointerup) can still see it and suppress itself.
  let lastDragWasReal = false;

  el.addEventListener('pointerdown', (ev) => {
    // Left mouse only; touch + pen pass through.
    if (ev.pointerType === 'mouse' && ev.button !== 0) return;
    pointerId = ev.pointerId;
    activeThreshold = (ev.pointerType === 'touch' || ev.pointerType === 'pen')
      ? touchThreshold
      : mouseThreshold;
    startX = ev.clientX;
    startY = ev.clientY;
    const rect = el.getBoundingClientRect();
    elStartX = rect.left;
    elStartY = rect.top;
    moved = false;
    try { el.setPointerCapture(pointerId); } catch { /* old browsers */ }
  });

  el.addEventListener('pointermove', (ev) => {
    if (pointerId === null || ev.pointerId !== pointerId) return;
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    if (!moved && Math.hypot(dx, dy) < activeThreshold) return;
    moved = true;
    el.classList.add('dragging');
    _setPosition(el, elStartX + dx, elStartY + dy);
    if (typeof opts.onMove === 'function') {
      try { opts.onMove(el.getBoundingClientRect()); } catch { /* swallow */ }
    }
  });

  function _release(ev) {
    if (pointerId === null || ev.pointerId !== pointerId) return;
    try { el.releasePointerCapture(pointerId); } catch { /* */ }
    pointerId = null;
    if (moved) {
      lastDragWasReal = true;
      const rect = el.getBoundingClientRect();
      _save(storageKey, { x: rect.left, y: rect.top });
      if (typeof opts.onDrop === 'function') {
        try { opts.onDrop(rect); } catch { /* swallow */ }
      }
    }
    el.classList.remove('dragging');
    moved = false;
  }
  el.addEventListener('pointerup', _release);
  el.addEventListener('pointercancel', _release);

  // Suppress the synthesized click that fires after a drag. The
  // capture-phase handler runs before any user-installed click
  // listener, so the click never reaches the FAB's own onClick.
  el.addEventListener('click', (ev) => {
    if (lastDragWasReal) {
      lastDragWasReal = false;
      ev.preventDefault();
      ev.stopImmediatePropagation();
    }
  }, true);

  // Re-clamp on viewport changes so a phone rotation or browser
  // resize doesn't strand the FAB outside the visible area. The clamp
  // re-saves when it actually moves the element so a smaller-viewport
  // session doesn't keep replaying the same out-of-bounds coords from
  // a wider session — the visible position IS the persisted state.
  const onResize = () => _clampToViewport(el, storageKey);
  window.addEventListener('resize', onResize);
  window.addEventListener('orientationchange', onResize);
}

/** Programmatic position reset — clears the stored position and
 *  re-applies the CSS default. Useful for a "reset layout" affordance
 *  the user can hit if they've dragged something out of reach. */
export function resetDraggablePosition(el, storageKey) {
  try { localStorage.removeItem(storageKey); } catch { /* */ }
  if (!el) return;
  el.style.left = '';
  el.style.top = '';
  el.style.right = '';
  el.style.bottom = '';
}


/* ── Internals ────────────────────────────────────────────────── */

function _setPosition(el, x, y) {
  const rect = el.getBoundingClientRect();
  const cx = Math.max(0, Math.min(window.innerWidth - rect.width, x));
  const cy = Math.max(0, Math.min(window.innerHeight - rect.height, y));
  // Switch from the CSS default right/bottom anchors to left/top.
  // We set both ``auto`` so a stylesheet ``right: 16px`` doesn't
  // continue to influence layout once the user has dragged.
  el.style.left = `${cx}px`;
  el.style.top = `${cy}px`;
  el.style.right = 'auto';
  el.style.bottom = 'auto';
}

function _applyStored(el, key) {
  let parsed;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return;
    parsed = JSON.parse(raw);
  } catch { return; }
  if (!parsed || typeof parsed.x !== 'number' || typeof parsed.y !== 'number') return;
  el.style.left = `${parsed.x}px`;
  el.style.top = `${parsed.y}px`;
  el.style.right = 'auto';
  el.style.bottom = 'auto';
  // Clamp on next frame once the element has its computed size. Pass
  // the storage key so a stored value that lands outside the current
  // viewport is normalized in-place and persisted, rather than
  // silently drifting every session.
  requestAnimationFrame(() => _clampToViewport(el, key));
}

function _save(key, pos) {
  try { localStorage.setItem(key, JSON.stringify(pos)); } catch { /* */ }
}

function _clampToViewport(el, key) {
  // Only clamp if the FAB has actually been moved (has explicit
  // left/top). The CSS-default right/bottom case is already inside
  // the viewport by definition.
  if (!el.style.left) return;
  const rect = el.getBoundingClientRect();
  const cx = Math.max(0, Math.min(window.innerWidth - rect.width, rect.left));
  const cy = Math.max(0, Math.min(window.innerHeight - rect.height, rect.top));
  // Skip the write + persist when nothing actually changed — avoids
  // localStorage churn on every resize tick.
  if (Math.abs(cx - rect.left) < 1 && Math.abs(cy - rect.top) < 1) return;
  el.style.left = `${cx}px`;
  el.style.top = `${cy}px`;
  if (key) _save(key, { x: cx, y: cy });
}
