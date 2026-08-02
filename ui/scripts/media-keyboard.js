/**
 * media-keyboard.js — Global hotkeys for the media-player singleton.
 *
 * Bindings (only active when an item is loaded AND the focus isn't in
 * a text input / textarea / contenteditable):
 *
 *   Space               play / pause
 *   ←  / →              skip back 15 / skip forward 30
 *   Shift + ←  / →      prev / next chapter
 *   [  / ]              slower / faster (0.05× steps)
 *   =                   reset speed to 1.0×
 *   M                   mute toggle
 *
 * Mounted via ``initMediaKeyboard()`` from app.js. Silent when no media
 * is active, so normal keyboard navigation is untouched everywhere else.
 */

import {
  getState, toggle, skip, skipChapterRelative, setSpeed,
} from './media-player.js';

const SKIP_FORWARD_S = 30;
const SKIP_BACK_S = 15;
const SPEED_STEP = 0.05;
const SPEED_MIN = 0.5;
const SPEED_MAX = 3.0;

let _audioEl = null;
let _mounted = false;

export function initMediaKeyboard() {
  if (_mounted) return;
  _mounted = true;
  document.addEventListener('keydown', _onKey, { capture: false });
}

function _isTextTarget(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    // Range inputs (volume/scrubber sliders) are NOT text — leaving
    // them non-blocking means arrow keys still nudge the slider while
    // a slider is focused, but that's the intuitive behavior.
    if (tag === 'INPUT' && el.type === 'range') return false;
    return true;
  }
  return false;
}

function _getAudio() {
  if (_audioEl && document.body.contains(_audioEl)) return _audioEl;
  _audioEl = document.body.querySelector('audio[style*="display: none"]');
  return _audioEl;
}

function _onKey(e) {
  if (!getState().fileId) return;           // no active media
  if (_isTextTarget(e.target)) return;       // don't hijack editors
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  switch (e.key) {
    case ' ':
    case 'k':  // YouTube-muscle-memory alternative
      e.preventDefault();
      toggle();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      if (e.shiftKey) skipChapterRelative(-1);
      else            skip(-SKIP_BACK_S);
      break;
    case 'ArrowRight':
      e.preventDefault();
      if (e.shiftKey) skipChapterRelative(1);
      else            skip(SKIP_FORWARD_S);
      break;
    case '[':
      e.preventDefault();
      setSpeed(Math.max(SPEED_MIN, _roundSpeed(getState().speed - SPEED_STEP)));
      break;
    case ']':
      e.preventDefault();
      setSpeed(Math.min(SPEED_MAX, _roundSpeed(getState().speed + SPEED_STEP)));
      break;
    case '=':
    case '0':
      e.preventDefault();
      setSpeed(1.0);
      break;
    case 'm':
    case 'M': {
      e.preventDefault();
      const audio = _getAudio();
      if (audio) audio.muted = !audio.muted;
      break;
    }
  }
}

function _roundSpeed(v) {
  // Avoid floating-point drift (1.0 + 0.05 + 0.05 ... != 1.15).
  return Math.round(v * 100) / 100;
}
