/**
 * tts-mini-player.js — Bottom-docked control for active read-aloud
 * sessions. Surfaces the same pause/stop affordances the browse-panel
 * Listen button offers, so closing the panel or navigating elsewhere
 * doesn't strand the user with audio they can't stop without going
 * back. Jump-back chip re-opens the source article.
 *
 * Subscribes to the read-aloud singleton; hides when no session is
 * active and renders when one starts. Lighter than media-mini-player
 * because TTS has no seekable timeline — pause/resume + stop + jump
 * back is the entire control surface.
 */

import {
  subscribeReadAloud,
  pauseReadAloud,
  resumeReadAloud,
  stopReadAloud,
} from './read-aloud.js';

const HOST_ID = 'tts-mini-player-host';
const POS_KEY = 'augmentum.ttsMiniPlayer.pos';
const PRESS_MS = 350;
const MOVE_THRESHOLD_PX = 6;

let _root = null;
let _unsubscribe = null;

export function initTtsMiniPlayer() {
  const host = document.getElementById(HOST_ID) || _ensureHost();
  host.innerHTML = _htmlShell();
  _root = host.querySelector('.tts-mini');
  _wire();
  _wireDrag();
  _restorePosition();
  window.addEventListener('resize', _restorePosition);
  _unsubscribe = subscribeReadAloud(_render);
}

function _ensureHost() {
  const h = document.createElement('div');
  h.id = HOST_ID;
  document.body.appendChild(h);
  return h;
}

function _htmlShell() {
  return `
    <div class="tts-mini hidden" role="region" aria-label="Article read-aloud">
      <div class="tts-mini-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <path d="M3 14h4l5-4v8l-5-4H3z"/>
          <path d="M15 8a5 5 0 0 1 0 8"/>
          <path d="M18 5a9 9 0 0 1 0 14"/>
        </svg>
      </div>
      <div class="tts-mini-meta">
        <div class="tts-mini-label">Reading</div>
        <div class="tts-mini-title" title=""></div>
      </div>
      <div class="tts-mini-controls">
        <button class="tts-mini-btn tts-mini-toggle" type="button" data-action="toggle" aria-label="Pause">
          <svg class="tts-icon-pause" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>
          <svg class="tts-icon-play" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M8 5v14l11-7z"/></svg>
        </button>
        <button class="tts-mini-btn tts-mini-jump" type="button" data-action="jump" aria-label="Open article" title="Open article">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="15" height="15"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
        </button>
        <button class="tts-mini-btn tts-mini-close" type="button" data-action="stop" aria-label="Stop reading" title="Stop">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
    </div>
  `;
}

function _wire() {
  if (!_root) return;
  _root.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === 'toggle') {
      if (_root.dataset.paused === 'true') resumeReadAloud();
      else pauseReadAloud();
    } else if (action === 'stop') {
      stopReadAloud();
    } else if (action === 'jump') {
      const url = _root.dataset.sourceUrl;
      if (url) {
        document.dispatchEvent(new CustomEvent('augmentum:browse-url', { detail: { url } }));
      }
    }
  });
}

// Press-and-hold drag. Mirrors media-mini-player's pattern: 350ms hold
// engages drag, taps under the threshold pass through as clicks so the
// pause/stop/jump buttons stay responsive. Position is session-scoped
// (resets on full reload) and re-clamped on resize so a stashed
// position doesn't end up off-screen after a rotation.
function _loadPos() {
  try {
    const raw = sessionStorage.getItem(POS_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (typeof p?.x === 'number' && typeof p?.y === 'number') return p;
  } catch { /* */ }
  return null;
}

function _savePos(pos) {
  try {
    if (pos) sessionStorage.setItem(POS_KEY, JSON.stringify(pos));
    else sessionStorage.removeItem(POS_KEY);
  } catch { /* private mode */ }
}

function _applyPos(pos) {
  if (!_root) return;
  _root.style.left = `${pos.x}px`;
  _root.style.top = `${pos.y}px`;
  _root.style.right = 'auto';
  _root.style.bottom = 'auto';
  _root.style.margin = '0';
  _root.classList.add('tts-moved');
}

function _resetPosition() {
  if (!_root) return;
  _root.style.left = '';
  _root.style.top = '';
  _root.style.right = '';
  _root.style.bottom = '';
  _root.style.margin = '';
  _root.classList.remove('tts-moved');
}

function _restorePosition() {
  if (!_root) return;
  const pos = _loadPos();
  if (!pos) { _resetPosition(); return; }
  requestAnimationFrame(() => {
    if (!_root.offsetWidth || !_root.offsetHeight) {
      _applyPos(pos);
      return;
    }
    const clamped = {
      x: Math.max(8, Math.min(window.innerWidth - _root.offsetWidth - 8, pos.x)),
      y: Math.max(8, Math.min(window.innerHeight - _root.offsetHeight - 8, pos.y)),
    };
    _applyPos(clamped);
  });
}

function _wireDrag() {
  if (!_root) return;
  let pressTimer = null;
  let pointerId = null;
  let startX = 0, startY = 0;
  let elStartX = 0, elStartY = 0;
  let engaged = false;

  // Buttons stay tappable; only the icon and title area act as grab
  // handles. The .tts-mini-controls subtree contains the action
  // buttons and is excluded from drag.
  const _isHandle = (target) => {
    if (!target) return false;
    if (target.closest('.tts-mini-controls')) return false;
    return _root.contains(target);
  };

  const _engage = () => {
    engaged = true;
    pressTimer = null;
    _root.classList.add('tts-dragging');
    document.body.classList.add('tts-drag-active');
  };

  const _clamp = (x, y) => {
    const w = _root.offsetWidth;
    const h = _root.offsetHeight;
    return {
      x: Math.max(8, Math.min(window.innerWidth - w - 8, x)),
      y: Math.max(8, Math.min(window.innerHeight - h - 8, y)),
    };
  };

  _root.addEventListener('pointerdown', (e) => {
    if (e.button && e.button !== 0) return;
    if (!_isHandle(e.target)) return;
    pointerId = e.pointerId;
    startX = e.clientX;
    startY = e.clientY;
    const rect = _root.getBoundingClientRect();
    elStartX = rect.left;
    elStartY = rect.top;
    engaged = false;
    pressTimer = setTimeout(_engage, PRESS_MS);
  });

  _root.addEventListener('pointermove', (e) => {
    if (pointerId !== e.pointerId) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (!engaged) {
      if (Math.abs(dx) > MOVE_THRESHOLD_PX || Math.abs(dy) > MOVE_THRESHOLD_PX) {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
        pointerId = null;
      }
      return;
    }
    e.preventDefault();
    try { _root.setPointerCapture(e.pointerId); } catch { /* */ }
    _applyPos(_clamp(elStartX + dx, elStartY + dy));
  });

  const _end = (e) => {
    if (pointerId !== e.pointerId) return;
    if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    try { _root.releasePointerCapture(e.pointerId); } catch { /* */ }
    pointerId = null;
    if (engaged) {
      _root.classList.remove('tts-dragging');
      document.body.classList.remove('tts-drag-active');
      const rect = _root.getBoundingClientRect();
      _savePos({ x: rect.left, y: rect.top });
      // Suppress the synthetic click that follows pointerup — the
      // press-and-hold was intent to move, not to activate a control.
      const blocker = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      _root.addEventListener('click', blocker, { capture: true, once: true });
      engaged = false;
    }
  };
  _root.addEventListener('pointerup', _end);
  _root.addEventListener('pointercancel', _end);

  // Double-click on a handle resets to the docked default position.
  _root.addEventListener('dblclick', (e) => {
    if (!_isHandle(e.target)) return;
    _savePos(null);
    _resetPosition();
  });
}

function _render(snap) {
  if (!_root) return;
  if (!snap.active) {
    _root.classList.add('hidden');
    return;
  }
  _root.classList.remove('hidden');
  _root.dataset.paused = snap.paused ? 'true' : 'false';
  _root.dataset.sourceUrl = snap.sourceUrl || '';
  _root.dataset.hasTitle = snap.title ? 'true' : 'false';

  const titleEl = _root.querySelector('.tts-mini-title');
  if (titleEl) {
    const text = snap.title || 'Reading aloud';
    titleEl.textContent = text;
    titleEl.title = text;
  }
  const toggleBtn = _root.querySelector('[data-action="toggle"]');
  if (toggleBtn) {
    // Streaming-WAV voices (Kokoro/PocketTTS) can't pause — hide the
    // control rather than show a Pause that silently stops with no
    // resume. Stop stays available. canPause is undefined for older
    // snapshots → default to showing it.
    const canPause = snap.canPause !== false;
    toggleBtn.style.display = canPause ? '' : 'none';
    toggleBtn.setAttribute('aria-label', snap.paused ? 'Resume' : 'Pause');
    toggleBtn.title = snap.paused ? 'Resume' : 'Pause';
  }
  const jumpBtn = _root.querySelector('[data-action="jump"]');
  if (jumpBtn) jumpBtn.style.display = snap.sourceUrl ? '' : 'none';
}
