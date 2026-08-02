/**
 * adapters/gamepad_api.js — virtual Gamepad API adapter.
 *
 * Universal cast input adapter that translates the wire-format
 * ``CMD_INPUT_GAMEPAD`` frame into ``navigator.getGamepads()`` results
 * visible to same-realm code. This is the factored-out version of the
 * original ``iframe-gamepad-shim.js`` — kept the same wire shape +
 * walk-and-shim behavior so existing games (EmulatorJS) work unchanged.
 *
 * Loaded by ``universal-input-adapter.js`` when ``gamepad_api`` appears
 * in the active input chain. The loader passes a ``ctx`` with:
 *   - recv(cb)      — subscribe to input frames; returns unsub
 *   - targets()     — returns array of [window, ...same-origin iframes]
 *   - telemetry     — mutable telemetry counters
 *   - keymap        — (unused by this adapter)
 *
 * Cross-origin iframes are unreachable from here; the origin proxy
 * (Phase 3) solves that by serving cross-origin games through our
 * origin so they become same-realm.
 */

export const id = 'gamepad_api';

const _NUM_BUTTONS = 17;
const _NUM_AXES = 4;
const _MAX_PADS = 4;
const _DISCONNECT_TIMEOUT_MS = 5000;

// Per-adapter state. Module-level because activate/deactivate during
// a single page lifetime should preserve pad state — the receiver may
// reconfigure the input_chain (e.g. add keyboard adapter) without
// dropping the gamepad's connection.
const _pads = new Array(_MAX_PADS).fill(null);
const _lastFrameAt = new Array(_MAX_PADS).fill(0);
let _frameCounter = 0;
let _installedOn = new WeakSet();

function _emptyPad(slot) {
  return {
    id: `Augmentum Cast Pad ${slot + 1} (Standard Gamepad)`,
    index: slot,
    connected: true,
    timestamp: performance.now(),
    mapping: 'standard',
    buttons: new Array(_NUM_BUTTONS).fill(0).map(() => ({
      pressed: false, touched: false, value: 0,
    })),
    axes: new Array(_NUM_AXES).fill(0),
    vibrationActuator: null,
  };
}

// ── Pure helpers (testable without DOM) ──────────────────────────

export function _applyFrame(pad, buttons, axes) {
  // Mutates pad in place. Returns the same pad (for chaining).
  // Reuses existing GamepadButton objects so a game holding a
  // reference to .buttons[i] across frames sees its values update.
  pad.timestamp = performance.now();
  for (let i = 0; i < _NUM_BUTTONS; i += 1) {
    const raw = Number(buttons?.[i]) || 0;
    const btn = pad.buttons[i];
    const pressed = raw > 0.5;
    btn.pressed = pressed;
    btn.touched = pressed || raw > 0;
    btn.value = raw;
  }
  for (let i = 0; i < _NUM_AXES; i += 1) {
    pad.axes[i] = Number(axes?.[i]) || 0;
  }
  return pad;
}

export function _newPad(slot) {
  return _emptyPad(slot);
}

export function _clampSlot(raw) {
  const n = Number(raw) | 0;
  if (n < 0) return 0;
  if (n >= _MAX_PADS) return _MAX_PADS - 1;
  return n;
}

// ── DOM-touching internals ───────────────────────────────────────

function _updatePad(slot, buttons, axes) {
  let pad = _pads[slot];
  const wasConnected = !!pad;
  if (!pad) {
    pad = _emptyPad(slot);
    _pads[slot] = pad;
  }
  _applyFrame(pad, buttons, axes);
  _lastFrameAt[slot] = performance.now();
  _frameCounter += 1;
  if (!wasConnected) _dispatchConnection(slot, true);
}

function _reapStalePads() {
  const now = performance.now();
  for (let slot = 0; slot < _MAX_PADS; slot += 1) {
    if (_pads[slot] && now - _lastFrameAt[slot] > _DISCONNECT_TIMEOUT_MS) {
      _pads[slot] = null;
      _dispatchConnection(slot, false);
    }
  }
}

function _dispatchConnection(slot, connected) {
  // Synthetic gamepad{connected,disconnected} event on the host window
  // so games gating boot on it can wake. Best-effort — some realms
  // refuse to construct GamepadEvent.
  try {
    const ev = new Event(connected ? 'gamepadconnected' : 'gamepaddisconnected');
    Object.defineProperty(ev, 'gamepad', {
      value: _pads[slot] || _emptyPad(slot),
      enumerable: true,
    });
    window.dispatchEvent(ev);
  } catch (_) { /* event shape unsupported here */ }
}

function _installOnNavigator(nav) {
  if (_installedOn.has(nav)) return;
  const shim = () => _pads.slice();
  try {
    Object.defineProperty(nav, 'getGamepads', {
      value: shim,
      writable: true,
      configurable: true,
    });
    _installedOn.add(nav);
  } catch (_) {
    try { nav.getGamepads = shim; _installedOn.add(nav); } catch (_) { /* give up */ }
  }
}

function _walkAndInstall(targets) {
  for (const win of targets) {
    try {
      const nav = win?.navigator;
      if (nav) _installOnNavigator(nav);
    } catch (_) { /* cross-origin — skip */ }
  }
}

// ── Adapter contract ─────────────────────────────────────────────

export async function probe(_iframe) {
  // Gamepad shim is universally safe: it never fires events for games
  // that don't call getGamepads. Returns mid-confidence so explicit
  // probes (Phase 4) can prefer more specific adapters when warranted.
  return { confidence: 0.5, evidence: ['gamepad_api_default'] };
}

export function activate(ctx) {
  _walkAndInstall(ctx.targets());

  const unsubRecv = ctx.recv((frame) => {
    const slot = _clampSlot(frame.slot ?? frame.pad_index ?? 0);
    _updatePad(slot, frame.buttons, frame.axes);
    // Re-walk on every frame so newly-mounted same-origin iframes
    // (e.g. EmulatorJS' lazy-loaded inner iframe) pick up the shim
    // before they poll.
    _walkAndInstall(ctx.targets());
    ctx.telemetry.dispatches += 1;
  });

  const reapInterval = setInterval(_reapStalePads, 1000);

  return function deactivate() {
    clearInterval(reapInterval);
    unsubRecv();
    // We don't try to un-patch navigators — once a game has a reference
    // to the shim'd getGamepads it'd keep working anyway. Future
    // reactivation will hit _installedOn and no-op.
  };
}

// Diagnostic surface (also kept on the legacy global for smoke parity).
export function _diagnostics() {
  return {
    pads: _pads.slice(),
    frameCount: _frameCounter,
  };
}
