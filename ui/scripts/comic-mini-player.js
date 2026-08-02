/**
 * comic-mini-player.js — Floating "paused comic" bubble.
 *
 * Replaces the previous bottom-docked horizontal strip with a
 * draggable circular bubble the user can place anywhere in the
 * viewport. Why the change:
 *   - On mobile portrait, the dock sat behind the orb nav bar
 *     (z-index ordering) — invisible and untappable.
 *   - The horizontal strip was 720px wide and visually heavy for
 *     a "set aside, come back later" affordance. A bubble reads
 *     as an unobtrusive sticker.
 *
 * Visual language matches the Grove orb's float-anywhere pattern,
 * intentional cross-modal cohesion: AI orb + reading bubble are
 * both circular, both draggable, both floating chrome.
 *
 * The exported API is unchanged from the previous version:
 *   - showComicMini(ctx)
 *   - hideComicMini()
 *   - getComicMiniState()
 *   - subscribeComicMini(fn)
 *   - initComicMiniPlayer()
 * so the comic-reader's `minimize()` doesn't have to change.
 *
 * Interactions:
 *   - Tap bubble body  → resume (open full reader at saved page)
 *   - Drag bubble       → reposition; release snaps to nearest
 *                         horizontal edge, persists in localStorage
 *   - Tap X badge       → dismiss (clears resume context entirely)
 * Always-visible X (no long-press, no auto-hiding chrome). Direct
 * user feedback on Grove orb's auto-hide chrome was that buttons
 * disappearing makes intent harder, not easier — so the X is
 * persistent here.
 */

const HOST_ID = 'comic-mini-player-host';
const POSITION_STORAGE_KEY = 'augmentum-comic-mini-pos-v1';
const FIRST_APPEARANCE_KEY = 'augmentum-comic-mini-seen-v1';

// --- Tunables -------------------------------------------------------
// Drag threshold matches Grove orb (6px) so the cross-app gesture
// vocabulary stays consistent.
const DRAG_THRESHOLD_PX = 6;
// Snap range: pointer-up within this distance of an edge snaps to it.
// 32px matches Grove orb.
const EDGE_SNAP_PX = 32;
// Inset from viewport edges so the bubble never crops the X badge
// or hits the safe-area inset.
const VIEWPORT_MARGIN = 12;
// Mobile breakpoint mirrored from CSS (700px) so JS sizing decisions
// match what users see.
const MOBILE_BREAKPOINT = 700;
const BUBBLE_SIZE_DESKTOP = 64;
const BUBBLE_SIZE_MOBILE  = 56;
// First-appearance label timing — long enough to read, short enough
// not to nag.
const LABEL_VISIBLE_MS = 2500;

// --- Module state ---------------------------------------------------

const _state = {
  active: false,
  fileId: '',
  coverUrl: '',
  title: '',
  subtitle: '',
  page: 1,
  pageCount: 0,
  resumeContext: null,
};
const _subscribers = new Set();

// Drag-control state lives outside _state — internal to the
// controller, not visible to subscribers.
const _drag = {
  pressing: false,
  pressAnchor: null,           // { x, y, pointerId }
  active: false,               // movement threshold crossed
  offset: { x: 0, y: 0 },      // pointer-to-bubble-corner offset
  target: { x: 0, y: 0 },      // current pointer-driven target
  rafId: null,
};

// Persisted bubble position. Null means "use the computed default
// based on current viewport + chrome state."
let _pos = null;

let _root = null;
let _initialized = false;
let _labelTimer = null;
let _onResize = null;

// --- Init -----------------------------------------------------------

export function initComicMiniPlayer() {
  if (_initialized) return;
  const host = document.getElementById(HOST_ID) || _ensureHost();
  host.innerHTML = _htmlShell();
  _root = host.querySelector('.comic-mini-bubble');
  _wire();
  _readPersistedPosition();
  _render();
  _initialized = true;
}

function _ensureHost() {
  const h = document.createElement('div');
  h.id = HOST_ID;
  document.body.appendChild(h);
  return h;
}

function _htmlShell() {
  return `
    <div class="comic-mini-bubble hidden" role="button" tabindex="0"
         aria-label="Resume reading" data-action="resume">
      <div class="comic-mini-cover-wrap">
        <img alt="" class="comic-mini-cover-img" draggable="false">
        <span class="comic-mini-cover-fallback" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.8"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
          </svg>
        </span>
      </div>
      <button type="button" class="comic-mini-x" data-action="dismiss"
              aria-label="Dismiss paused reader" title="Dismiss">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
      <div class="comic-mini-label" aria-hidden="true"></div>
    </div>
  `;
}

// --- Geometry -------------------------------------------------------

function _bubbleSize() {
  return window.innerWidth < MOBILE_BREAKPOINT
    ? BUBBLE_SIZE_MOBILE : BUBBLE_SIZE_DESKTOP;
}

/**
 * Default position: right edge, ~40% from top of viewport.
 *
 * Anchored from the chat UI's natural viewport — the user's primary
 * surface they return to after minimizing. 40% from top puts the
 * bubble:
 *   - Below the header (~56px)
 *   - Above the input toolbar + textarea (bottom ~140px on mobile)
 *   - Clear of the orb nav bar (mobile, ~64px + safe-area)
 *   - Clear of the audio mini-player (~88px when active)
 * One rule, holds across desktop and mobile.
 *
 * Right-edge anchored because Augmentum is LTR-first and the right
 * side is where chat metadata and secondary chrome already live.
 */
function _defaultPos() {
  const size = _bubbleSize();
  return {
    x: window.innerWidth - size - VIEWPORT_MARGIN,
    y: Math.round(window.innerHeight * 0.4),
  };
}

function _clampToViewport(x, y) {
  const size = _bubbleSize();
  const maxX = window.innerWidth  - size - VIEWPORT_MARGIN;
  const maxY = window.innerHeight - size - VIEWPORT_MARGIN;
  return {
    x: Math.max(VIEWPORT_MARGIN, Math.min(maxX, x)),
    y: Math.max(VIEWPORT_MARGIN, Math.min(maxY, y)),
  };
}

/**
 * Snap to the nearer horizontal edge if released within EDGE_SNAP_PX.
 * Vertical position is preserved — the user picks their own height
 * since "right side" vs "left side" is the only axis with a strong
 * default. iOS PiP behaves the same way.
 */
function _snapToEdge(x, y) {
  const size = _bubbleSize();
  const rightEdge = window.innerWidth - size - VIEWPORT_MARGIN;
  if (x - VIEWPORT_MARGIN < EDGE_SNAP_PX)        x = VIEWPORT_MARGIN;
  else if (rightEdge - x < EDGE_SNAP_PX)         x = rightEdge;
  return { x, y };
}

// --- Persistence ----------------------------------------------------

function _readPersistedPosition() {
  try {
    const raw = localStorage.getItem(POSITION_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (
      parsed && typeof parsed.x === 'number' && typeof parsed.y === 'number'
    ) {
      _pos = parsed;
    }
  } catch { /* corrupt or unavailable — fall through to default */ }
}

function _persistPosition() {
  if (!_pos) return;
  try {
    localStorage.setItem(POSITION_STORAGE_KEY, JSON.stringify(_pos));
  } catch { /* quota / private mode — silent */ }
}

// --- Position writing -----------------------------------------------

/**
 * Write inline left/top via rAF batching during drag, immediate during
 * snap / restore. Using left/top (not transform) so the bubble can be
 * absolutely positioned within the host without a transform-induced
 * compositor layer breaking anything underneath. Same approach as
 * Grove orb-detach.js for the same family of reasons.
 */
function _writePosition() {
  if (!_root || !_pos) return;
  _root.style.left = `${_pos.x}px`;
  _root.style.top  = `${_pos.y}px`;
  _root.style.right = 'auto';
  // Update label-side class so the first-appearance label slides out
  // toward viewport center, not off-screen. Threshold is a third from
  // the right edge — anything past that flips the label to the LEFT
  // side of the bubble.
  const flipRight = _pos.x < window.innerWidth / 3;
  _root.classList.toggle('label-right', flipRight);
}

function _scheduleFrame() {
  if (_drag.rafId) return;
  _drag.rafId = requestAnimationFrame(() => {
    _drag.rafId = null;
    const { x, y } = _clampToViewport(_drag.target.x, _drag.target.y);
    _pos = { x, y };
    _writePosition();
  });
}

// --- Pointer handling -----------------------------------------------

/**
 * Returns true when the pointerdown target is a separate interactive
 * surface (the X badge) that owns its own click handler. Skipping
 * those for press/drag avoids fighting their click listener — the
 * dismiss tap should fire dismiss, not start a bubble drag.
 */
function _isInteractiveTarget(target) {
  if (!target) return false;
  if (target.closest('.comic-mini-x')) return true;
  return false;
}

function _onPointerDown(e) {
  if (!_root) return;
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  if (_isInteractiveTarget(e.target)) return;
  if (!_root.contains(e.target)) return;

  // Suppress browser native long-press callout (image save menu),
  // text selection, image drag.
  try { e.preventDefault(); } catch { /* non-cancelable */ }

  _drag.pressing = true;
  _drag.pressAnchor = {
    x: e.clientX,
    y: e.clientY,
    pointerId: e.pointerId,
  };
  // Capture pointer-to-bubble-corner offset NOW so if the press
  // promotes to a drag mid-movement, the bubble stays under the
  // pointer (no jump on first move past threshold).
  if (_pos) {
    _drag.offset = {
      x: e.clientX - _pos.x,
      y: e.clientY - _pos.y,
    };
  }
  _root.classList.add('is-pressing');
  // Capture pointer so we keep getting move/up events even if the
  // pointer leaves the bubble (which it will during a drag).
  try { _root.setPointerCapture(e.pointerId); } catch { /* ignore */ }
}

function _onPointerMove(e) {
  if (!_drag.pressing) return;
  // Already dragging — just keep up.
  if (_drag.active) {
    _drag.target.x = e.clientX - _drag.offset.x;
    _drag.target.y = e.clientY - _drag.offset.y;
    _scheduleFrame();
    return;
  }
  // Not yet dragging — check threshold.
  const dx = e.clientX - _drag.pressAnchor.x;
  const dy = e.clientY - _drag.pressAnchor.y;
  if (Math.hypot(dx, dy) <= DRAG_THRESHOLD_PX) return;

  // Threshold crossed — promote to drag.
  _drag.active = true;
  _root.classList.remove('is-pressing');
  _root.classList.add('is-dragging');
  _drag.target.x = e.clientX - _drag.offset.x;
  _drag.target.y = e.clientY - _drag.offset.y;
  _scheduleFrame();
}

function _onPointerUp(e) {
  if (!_drag.pressing) return;
  const wasDragging = _drag.active;
  // Reset state regardless of branch.
  _drag.pressing = false;
  _drag.active = false;
  _drag.pressAnchor = null;
  if (_drag.rafId) {
    cancelAnimationFrame(_drag.rafId);
    _drag.rafId = null;
  }
  _root?.classList.remove('is-pressing', 'is-dragging');
  try { _root?.releasePointerCapture(e.pointerId); } catch { /* ignore */ }

  if (wasDragging) {
    // Drag release — snap to nearest horizontal edge, persist.
    if (_pos) {
      const snapped = _snapToEdge(_pos.x, _pos.y);
      const clamped = _clampToViewport(snapped.x, snapped.y);
      _pos = clamped;
      _writePosition();
      _persistPosition();
    }
    return;
  }
  // No drag — this was a tap. Resume.
  _resume();
}

function _onPointerCancel() {
  // Pointer cancel (touch-action interrupted, gesture stolen by
  // browser, etc.) — abandon the press without firing tap or drag.
  _drag.pressing = false;
  _drag.active = false;
  _drag.pressAnchor = null;
  if (_drag.rafId) {
    cancelAnimationFrame(_drag.rafId);
    _drag.rafId = null;
  }
  _root?.classList.remove('is-pressing', 'is-dragging');
}

// --- Resize handling ------------------------------------------------

/**
 * On viewport resize (orientation change, browser resize, mobile
 * keyboard appearing), re-clamp the saved position so the bubble
 * doesn't get stranded off-screen. Doesn't move it if already in
 * bounds — only rescues stuck cases.
 */
function _handleResize() {
  if (!_pos || !_state.active) return;
  const clamped = _clampToViewport(_pos.x, _pos.y);
  if (clamped.x !== _pos.x || clamped.y !== _pos.y) {
    _pos = clamped;
    _writePosition();
    _persistPosition();
  }
}

// --- Wiring ---------------------------------------------------------

function _wire() {
  if (!_root) return;
  // Pointer events drive both press / drag and tap-to-resume. The X
  // dismiss button has its own click listener that stopPropagation's
  // so it never falls through to a bubble tap.
  _root.addEventListener('pointerdown', _onPointerDown);
  _root.addEventListener('pointermove', _onPointerMove);
  _root.addEventListener('pointerup', _onPointerUp);
  _root.addEventListener('pointercancel', _onPointerCancel);

  // Dismiss button — separate hit target.
  const xBtn = _root.querySelector('.comic-mini-x');
  xBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    hideComicMini();
  });
  // Pointerdown on X also stops propagation so the bubble's
  // pointerdown handler (above) doesn't start a press on the X.
  // Belt-and-suspenders against the _isInteractiveTarget check.
  xBtn?.addEventListener('pointerdown', (e) => e.stopPropagation());

  // Keyboard — Enter/Space resume, Escape dismiss, mirroring focus
  // management from the previous dock. focus is on the bubble
  // itself (role=button, tabindex=0).
  _root.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      // Don't fire if focus is on the X badge — that has its own
      // Enter/Space handling via the native button element.
      if (document.activeElement?.classList.contains('comic-mini-x')) return;
      e.preventDefault();
      _resume();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      hideComicMini();
    }
  });

  // Re-clamp on viewport changes so a bubble dragged to the right
  // edge in landscape doesn't disappear off-screen in portrait.
  _onResize = () => _handleResize();
  window.addEventListener('resize', _onResize);
  window.addEventListener('orientationchange', _onResize);
}

// --- Resume / dismiss -----------------------------------------------

async function _resume() {
  if (!_state.active || !_state.resumeContext) return;
  const ctx = _state.resumeContext;
  // Hide before the reader mounts so there's no flicker of two
  // surfaces at once if the reader's mount is synchronous.
  _setActive(false);
  try {
    const { openComicReader } = await import('./comic-reader/index.js?v=surface-handoff-20260512a');
    openComicReader(ctx.file, { siblings: ctx.siblings, resume: ctx.uiState });
  } catch (err) {
    console.error('[comic-mini] resume failed:', err);
    // Restore the bubble so the user's state isn't silently lost.
    _setActive(true);
  }
}

// --- First-appearance label ------------------------------------------

function _showLabelOnce() {
  if (!_root) return;
  // Always show the label on every show, not just the first session.
  // Why: user might minimize from one comic, surf the chat for a
  // while, minimize a different comic — they need to know which one
  // is parked. The label is short-lived enough that showing it on
  // every show isn't intrusive.
  // The FIRST_APPEARANCE_KEY is reserved for future "long" hint copy
  // (e.g. "Tap to resume, drag to move, X to dismiss") if we ever
  // want a one-time onboarding pass.
  const labelEl = _root.querySelector('.comic-mini-label');
  if (!labelEl) return;
  const text = _state.title || 'Continue reading';
  labelEl.textContent = text;
  // Force a layout cycle before adding the visible class so the
  // transition fires reliably even on a freshly-shown bubble.
  void labelEl.offsetWidth;
  _root.classList.add('label-visible');
  clearTimeout(_labelTimer);
  _labelTimer = setTimeout(() => {
    _root?.classList.remove('label-visible');
  }, LABEL_VISIBLE_MS);
}

// --- Public API -----------------------------------------------------

/**
 * Show the bubble with a snapshot from the reader.
 *
 * ``ctx`` shape: { file, siblings, title, subtitle, page, pageCount,
 *                  coverUrl, uiState: { zoom, webtoonScrollY, ... } }
 * ``uiState`` is opaque to this module — ComicReader owns its
 * contents and reads it back in ``_restoreResumeState``.
 */
export function showComicMini(ctx) {
  if (!_initialized) initComicMiniPlayer();
  if (!ctx || !ctx.file || !ctx.file.id) return;
  _state.fileId = ctx.file.id;
  _state.title = ctx.title || ctx.file.name || 'Reading';
  _state.subtitle = ctx.subtitle || '';
  _state.page = Math.max(1, Number(ctx.page) || 1);
  _state.pageCount = Math.max(0, Number(ctx.pageCount) || 0);
  _state.coverUrl = ctx.coverUrl || '';
  _state.resumeContext = {
    file: ctx.file,
    siblings: ctx.siblings || null,
    uiState: ctx.uiState || null,
  };
  // Fall back to default position if no persisted position exists.
  // Persist nothing yet — first drag is what saves the user's
  // preferred location. Default-only sessions leave localStorage
  // untouched.
  //
  // Existing positions get re-clamped on every show: a position saved
  // on a wider viewport (window resized since, other monitor) would
  // otherwise put the bubble entirely off-screen — minimize appears to
  // do nothing. The resize handler can't rescue this case because it
  // early-returns while the bubble is inactive.
  _pos = _pos ? _clampToViewport(_pos.x, _pos.y) : _defaultPos();
  _setActive(true);
}

export function hideComicMini() {
  if (!_state.active) return;
  _state.resumeContext = null;
  clearTimeout(_labelTimer);
  _root?.classList.remove('label-visible');
  _setActive(false);
}

export function getComicMiniState() {
  return { ..._state };
}

export function subscribeComicMini(fn) {
  _subscribers.add(fn);
  try { fn(getComicMiniState()); } catch { /* ignore */ }
  return () => _subscribers.delete(fn);
}

// --- Internal render -------------------------------------------------

function _setActive(v) {
  _state.active = !!v;
  // Body class kept for backwards compat — older code paths checked
  // `document.body.classList.contains('comic-mini-active')` to know
  // whether a paused comic existed. The class no longer drives
  // layout (the bubble doesn't push other chrome), but removing
  // the class without checking callsites would be a silent break,
  // so leave it.
  document.body.classList.toggle('comic-mini-active', _state.active);
  _render();
  if (_state.active) {
    _showLabelOnce();
  }
  _subscribers.forEach(fn => {
    try { fn(getComicMiniState()); } catch (err) {
      console.warn('[comic-mini] subscriber error:', err);
    }
  });
}

function _render() {
  if (!_root) return;
  _root.classList.toggle('hidden', !_state.active);
  if (!_state.active) return;

  // Position
  if (_pos) _writePosition();

  // Cover
  const coverImg = _root.querySelector('.comic-mini-cover-img');
  if (coverImg) {
    if (_state.coverUrl && coverImg.getAttribute('src') !== _state.coverUrl) {
      coverImg.onerror = () => {
        // Cover proxy failed — drop src so the SVG fallback shows
        // instead of a broken-image icon. CSS sibling selector
        // handles the visibility swap automatically.
        coverImg.removeAttribute('src');
      };
      coverImg.setAttribute('src', _state.coverUrl);
    } else if (!_state.coverUrl) {
      coverImg.removeAttribute('src');
    }
  }

  // Aria — fold page progress in so screen-reader users know where
  // they are without having to navigate the (visually hidden)
  // label. Updated on every render so page-flip pushes from the
  // reader keep this fresh.
  const ariaParts = [`Resume reading ${_state.title}`];
  if (_state.pageCount) {
    ariaParts.push(`page ${_state.page} of ${_state.pageCount}`);
  }
  _root.setAttribute('aria-label', ariaParts.join(', '));

  // Progress ring — pure CSS via custom property. 0–100, the
  // conic-gradient in the stylesheet renders the visible arc.
  const pct = _state.pageCount
    ? Math.max(0, Math.min(100, (_state.page / _state.pageCount) * 100))
    : 0;
  _root.style.setProperty('--progress', String(pct));
}
