/**
 * adapters/keyboard.js — synthetic KeyboardEvent adapter.
 *
 * Translates gamepad input frames into ``keydown``/``keyup`` events on
 * each target realm's focused element (or document body). Edge-triggered:
 * a button held across N frames fires one keydown + one eventual keyup,
 * not N events per frame.
 *
 * Stick → arrow-key mapping uses a deadzone so a thumbed-rest stick
 * doesn't spam direction events. Sticks emit one keydown when they
 * cross the deadzone, one keyup when they return.
 *
 * Default keymap targets NES/retro UX (Z=A, X=B, arrows, Enter=start).
 * Per-game overrides arrive via ``ctx.keymap.keyboard``.
 */

export const id = 'keyboard';

// 17-button standard layout (matches gamepad_api wire shape).
const DEFAULT_KEYMAP = Object.freeze({
  buttons: {
    0: 'KeyZ',          // A → Z
    1: 'KeyX',          // B → X
    2: 'KeyA',          // X → A
    3: 'KeyS',          // Y → S
    4: 'KeyQ',          // LB → Q
    5: 'KeyW',          // RB → W
    6: 'KeyE',          // LT → E
    7: 'KeyR',          // RT → R
    8: 'ShiftRight',    // Select
    9: 'Enter',         // Start
    12: 'ArrowUp',
    13: 'ArrowDown',
    14: 'ArrowLeft',
    15: 'ArrowRight',
  },
  axes: {
    0: { negative: 'ArrowLeft', positive: 'ArrowRight' },
    1: { negative: 'ArrowUp', positive: 'ArrowDown' },
  },
  deadzone: 0.5,
});

// KeyboardEvent.key derived from .code — games gate on both. Minimal
// table; anything not in here falls back to the code string (which is
// not strictly correct but won't blow up).
const CODE_TO_KEY = Object.freeze({
  ArrowUp: 'ArrowUp', ArrowDown: 'ArrowDown',
  ArrowLeft: 'ArrowLeft', ArrowRight: 'ArrowRight',
  Enter: 'Enter', Space: ' ', Escape: 'Escape', Tab: 'Tab',
  ShiftLeft: 'Shift', ShiftRight: 'Shift',
  ControlLeft: 'Control', ControlRight: 'Control',
  AltLeft: 'Alt', AltRight: 'Alt',
  KeyA: 'a', KeyB: 'b', KeyC: 'c', KeyD: 'd', KeyE: 'e', KeyF: 'f',
  KeyG: 'g', KeyH: 'h', KeyI: 'i', KeyJ: 'j', KeyK: 'k', KeyL: 'l',
  KeyM: 'm', KeyN: 'n', KeyO: 'o', KeyP: 'p', KeyQ: 'q', KeyR: 'r',
  KeyS: 's', KeyT: 't', KeyU: 'u', KeyV: 'v', KeyW: 'w', KeyX: 'x',
  KeyY: 'y', KeyZ: 'z',
  Digit0: '0', Digit1: '1', Digit2: '2', Digit3: '3', Digit4: '4',
  Digit5: '5', Digit6: '6', Digit7: '7', Digit8: '8', Digit9: '9',
});

// ── Pure helpers (testable without DOM) ──────────────────────────

export function _codeToKey(code) {
  return CODE_TO_KEY[code] ?? code;
}

/**
 * Diff a new gamepad frame against previous state, returning the keys
 * that should fire keydown/keyup right now.
 *
 * @param {Object} prev — { buttons: bool[17], axes: {[idx]: {pos: bool, neg: bool}} }
 * @param {Object} frame — { buttons: number[17], axes: number[4] }
 * @param {Object} keymap — { buttons: {[i]: code}, axes: {[i]: {positive, negative}}, deadzone }
 * @returns {{ changes: Array<{code: string, down: boolean}>, next: Object }}
 */
export function _diffChanges(prev, frame, keymap) {
  const km = keymap || DEFAULT_KEYMAP;
  const buttons = frame?.buttons || [];
  const axes = frame?.axes || [];
  const dz = typeof km.deadzone === 'number' ? km.deadzone : 0.5;

  const prevBtns = prev?.buttons || [];
  const prevAxes = prev?.axes || {};
  const nextBtns = new Array(17).fill(false);
  const nextAxes = {};
  const changes = [];

  // Button edges
  for (const [idxStr, code] of Object.entries(km.buttons || {})) {
    const i = Number(idxStr);
    const raw = Number(buttons[i] || 0);
    const pressed = raw > 0.5;
    nextBtns[i] = pressed;
    const wasPressed = !!prevBtns[i];
    if (pressed !== wasPressed) {
      changes.push({ code, down: pressed });
    }
  }

  // Axis edges
  for (const [idxStr, mapping] of Object.entries(km.axes || {})) {
    const i = Number(idxStr);
    const v = Number(axes[i] || 0);
    const positive = v > dz;
    const negative = v < -dz;
    const prevState = prevAxes[i] || { positive: false, negative: false };
    if (positive !== prevState.positive && mapping.positive) {
      changes.push({ code: mapping.positive, down: positive });
    }
    if (negative !== prevState.negative && mapping.negative) {
      changes.push({ code: mapping.negative, down: negative });
    }
    nextAxes[i] = { positive, negative };
  }

  return {
    changes,
    next: { buttons: nextBtns, axes: nextAxes },
  };
}

export function _heldKeys(state, keymap) {
  // Returns codes currently held — used at deactivate time to release.
  const km = keymap || DEFAULT_KEYMAP;
  const held = [];
  for (const [idxStr, code] of Object.entries(km.buttons || {})) {
    if (state?.buttons?.[Number(idxStr)]) held.push(code);
  }
  for (const [idxStr, mapping] of Object.entries(km.axes || {})) {
    const axState = state?.axes?.[Number(idxStr)] || {};
    if (axState.positive && mapping.positive) held.push(mapping.positive);
    if (axState.negative && mapping.negative) held.push(mapping.negative);
  }
  return held;
}

// ── DOM-touching internals ───────────────────────────────────────

function _dispatchKeyTo(target, code, down) {
  try {
    const ev = new KeyboardEvent(down ? 'keydown' : 'keyup', {
      code,
      key: _codeToKey(code),
      bubbles: true,
      cancelable: true,
      composed: true,
    });
    target.dispatchEvent(ev);
    return true;
  } catch (_) {
    return false;
  }
}

function _resolveTarget(win) {
  try {
    return win?.document?.activeElement || win?.document?.body || win;
  } catch (_) { return null; }
}

// ── Adapter contract ─────────────────────────────────────────────

export async function probe(iframe) {
  // Phase 4 will instrument addEventListener('keydown') traps. For now
  // declare medium-low confidence so the classifier prefers a more
  // specific adapter when available.
  return { confidence: 0.3, evidence: [] };
}

export function activate(ctx) {
  const keymap = ctx.keymap?.keyboard || DEFAULT_KEYMAP;
  let state = { buttons: new Array(17).fill(false), axes: {} };

  const unsubRecv = ctx.recv((frame) => {
    const { changes, next } = _diffChanges(state, frame, keymap);
    state = next;
    if (!changes.length) return;
    for (const { code, down } of changes) {
      for (const win of ctx.targets()) {
        const target = _resolveTarget(win);
        if (target && _dispatchKeyTo(target, code, down)) {
          ctx.telemetry.dispatches += 1;
        }
      }
    }
  });

  return function deactivate() {
    unsubRecv();
    // Release everything we're holding so the game doesn't get stuck
    // with a phantom keydown when the adapter swaps out.
    const held = _heldKeys(state, keymap);
    for (const code of held) {
      for (const win of ctx.targets()) {
        const target = _resolveTarget(win);
        if (target) _dispatchKeyTo(target, code, false);
      }
    }
    state = { buttons: new Array(17).fill(false), axes: {} };
  };
}
