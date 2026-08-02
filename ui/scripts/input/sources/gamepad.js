// ui/scripts/input/sources/gamepad.js
//
// HTML5 Gamepad API source. Polls navigator.getGamepads() at 60Hz
// (rAF-driven) and diffs button/axis state to emit transition events.
// The Gamepad API is event-light by design — there is no "button up"
// event from the browser, so polling + diff is the standard pattern.
//
// Pad index is included in the raw code so multi-pad scenarios (local
// co-op) can target a specific pad: 'gp:0:button:0' vs 'gp:1:button:0'.
//
// Battery / vibration / lightbar APIs are out of scope here — they
// belong in a separate gamepad-effects module that consumes the same
// pad indices.

import { RawInputKind } from '../input-bus.js';

const AXIS_EPSILON = 0.01; // ignore micro-jitter

export class GamepadSource {
  constructor() {
    this._bus = null;
    this._raf = 0;
    // padIndex -> { buttons: number[], axes: number[] }
    this._state = new Map();
    this._connectedHandler = null;
    this._disconnectedHandler = null;
    this._running = false;
  }

  attach(bus) {
    this._bus = bus;
    this._connectedHandler = () => { /* no-op — getGamepads picks it up */ };
    this._disconnectedHandler = (e) => this._handleDisconnect(e);
    if (typeof window !== 'undefined') {
      window.addEventListener('gamepadconnected', this._connectedHandler);
      window.addEventListener('gamepaddisconnected', this._disconnectedHandler);
    }
    this._running = true;
    this._tick();
  }

  detach() {
    this._running = false;
    if (this._raf && typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(this._raf);
    }
    if (typeof window !== 'undefined') {
      if (this._connectedHandler) window.removeEventListener('gamepadconnected', this._connectedHandler);
      if (this._disconnectedHandler) window.removeEventListener('gamepaddisconnected', this._disconnectedHandler);
    }
    // Synthetic release for everything held.
    if (this._bus) {
      for (const [padIndex, st] of this._state) {
        st.buttons.forEach((pressed, idx) => {
          if (pressed) this._emitButton(padIndex, idx, false, true);
        });
      }
    }
    this._state.clear();
    this._bus = null;
  }

  _tick() {
    if (!this._running || !this._bus) return;
    if (typeof navigator === 'undefined' || !navigator.getGamepads) return;
    const pads = navigator.getGamepads();
    for (let i = 0; i < pads.length; i++) {
      const pad = pads[i];
      if (!pad) {
        if (this._state.has(i)) this._releasePad(i);
        continue;
      }
      this._diffPad(pad);
    }
    if (typeof requestAnimationFrame !== 'undefined') {
      this._raf = requestAnimationFrame(() => this._tick());
    }
  }

  _diffPad(pad) {
    let st = this._state.get(pad.index);
    if (!st) {
      st = { buttons: pad.buttons.map(() => false), axes: pad.axes.map(() => 0) };
      this._state.set(pad.index, st);
    }
    // Buttons
    for (let b = 0; b < pad.buttons.length; b++) {
      const pressed = !!pad.buttons[b].pressed;
      if (pressed !== st.buttons[b]) {
        st.buttons[b] = pressed;
        this._emitButton(pad.index, b, pressed, false);
      }
    }
    // Axes
    for (let a = 0; a < pad.axes.length; a++) {
      const v = pad.axes[a];
      if (Math.abs(v - st.axes[a]) > AXIS_EPSILON) {
        st.axes[a] = v;
        this._emitAxis(pad.index, a, v);
      }
    }
  }

  _emitButton(padIndex, buttonIndex, pressed, synthetic) {
    this._bus._dispatchRaw({
      kind: pressed ? RawInputKind.BUTTON_DOWN : RawInputKind.BUTTON_UP,
      source: 'gamepad',
      code: `gp:${padIndex}:button:${buttonIndex}`,
      padIndex,
      buttonIndex,
      value: pressed ? 1 : 0,
      synthetic: !!synthetic,
    });
  }

  _emitAxis(padIndex, axisIndex, value) {
    this._bus._dispatchRaw({
      kind: RawInputKind.AXIS,
      source: 'gamepad',
      code: `gp:${padIndex}:axis:${axisIndex}`,
      padIndex,
      axisIndex,
      value,
    });
  }

  _releasePad(padIndex) {
    const st = this._state.get(padIndex);
    if (!st) return;
    st.buttons.forEach((pressed, idx) => {
      if (pressed) this._emitButton(padIndex, idx, false, true);
    });
    st.axes.forEach((v, idx) => {
      if (Math.abs(v) > AXIS_EPSILON) this._emitAxis(padIndex, idx, 0);
    });
    this._state.delete(padIndex);
  }

  _handleDisconnect(e) {
    if (e && e.gamepad) this._releasePad(e.gamepad.index);
  }
}
