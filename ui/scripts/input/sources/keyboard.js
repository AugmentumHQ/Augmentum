// ui/scripts/input/sources/keyboard.js
//
// Keyboard input source. Listens to window-level keydown/keyup and
// emits raw events through the bus. Repeat events are suppressed
// (the bus already handles "stuck key" logic per binding).
//
// Targets: any element/scope; default is window. Use a tighter
// target when the bus is scoped (e.g. an embedded canvas in a game
// stage that should only react when focused).

import { RawInputKind } from '../input-bus.js';

export class KeyboardSource {
  constructor(target = (typeof window !== 'undefined' ? window : null)) {
    this._target = target;
    this._bus = null;
    this._onDown = null;
    this._onUp = null;
    this._onBlur = null;
    this._held = new Set();
  }

  attach(bus) {
    if (!this._target) return;
    this._bus = bus;
    this._onDown = (e) => this._handleDown(e);
    this._onUp = (e) => this._handleUp(e);
    this._onBlur = () => this._releaseAll();
    this._target.addEventListener('keydown', this._onDown, { passive: true });
    this._target.addEventListener('keyup', this._onUp, { passive: true });
    this._target.addEventListener('blur', this._onBlur);
  }

  detach() {
    if (!this._target) return;
    if (this._onDown) this._target.removeEventListener('keydown', this._onDown);
    if (this._onUp) this._target.removeEventListener('keyup', this._onUp);
    if (this._onBlur) this._target.removeEventListener('blur', this._onBlur);
    this._releaseAll();
    this._bus = null;
  }

  _handleDown(e) {
    if (e.repeat) return;
    const code = e.code || e.key;
    if (!code) return;
    this._held.add(code);
    this._bus._dispatchRaw({
      kind: RawInputKind.BUTTON_DOWN,
      source: 'keyboard',
      code,
      value: 1,
    });
  }

  _handleUp(e) {
    const code = e.code || e.key;
    if (!code) return;
    if (!this._held.has(code)) return;
    this._held.delete(code);
    this._bus._dispatchRaw({
      kind: RawInputKind.BUTTON_UP,
      source: 'keyboard',
      code,
      value: 0,
    });
  }

  _releaseAll() {
    if (!this._bus) return;
    for (const code of this._held) {
      this._bus._dispatchRaw({
        kind: RawInputKind.BUTTON_UP,
        source: 'keyboard',
        code,
        value: 0,
        synthetic: true,
      });
    }
    this._held.clear();
  }
}
