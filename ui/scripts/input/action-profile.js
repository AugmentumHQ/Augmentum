// ui/scripts/input/action-profile.js
//
// Maps physical input events (key codes, gamepad buttons/axes) to
// logical actions ("jump", "next_track", "menu_up"). One profile is
// active on the InputBus at a time; switching profiles is the way to
// re-skin the controller for a different surface (game vs gallery vs
// avatar control).
//
// A binding entry is one of:
//   { source: 'keyboard', code: 'Space' }
//   { source: 'gamepad', button: 0 }
//   { source: 'gamepad', axis: 0, sign: +1, deadzone: 0.15 }   (axis -> button)
//   { source: 'gamepad', axis: 0 }                              (axis -> axis)
//   { source: 'mouse', button: 2 }
//   { source: 'touch', tag: 'jump' }                            (virtual gamepad)
//
// Profiles are JSON-friendly so users can save/share them.

import { InputEventKind, RawInputKind } from './input-bus.js';

export class ActionProfile {
  constructor(id, bindings = {}) {
    this.id = String(id || 'default');
    this._bindings = new Map();
    this.replaceBindings(bindings);
  }

  replaceBindings(bindings) {
    this._bindings.clear();
    for (const [action, list] of Object.entries(bindings || {})) {
      this.bind(action, list);
    }
  }

  bind(action, bindingOrList) {
    if (!action) return;
    const list = Array.isArray(bindingOrList) ? bindingOrList : [bindingOrList];
    this._bindings.set(action, list.filter(Boolean));
  }

  unbind(action) { this._bindings.delete(action); }

  bindingsFor(action) { return this._bindings.get(action) || []; }

  toJSON() {
    const out = {};
    for (const [k, v] of this._bindings) out[k] = v;
    return { id: this.id, bindings: out };
  }

  static fromJSON(obj) {
    if (!obj) return new ActionProfile('default');
    return new ActionProfile(obj.id || 'default', obj.bindings || {});
  }

  // ── Resolution: raw event -> action event (called by InputBus) ──

  resolve(rawEvent) {
    if (!rawEvent || !rawEvent.source) return null;
    // Check every binding once — small profiles, this is fine.
    for (const [action, bindings] of this._bindings) {
      for (const b of bindings) {
        if (b.source !== rawEvent.source) continue;
        const match = this._matchBinding(b, rawEvent);
        if (match) return { ...match, action };
      }
    }
    return null;
  }

  _matchBinding(b, raw) {
    // Keyboard: code-by-code match
    if (b.source === 'keyboard' && b.code === raw.code) {
      if (raw.kind === RawInputKind.BUTTON_DOWN) {
        return { kind: InputEventKind.ACTION_DOWN, source: raw.source, code: raw.code, value: 1 };
      }
      if (raw.kind === RawInputKind.BUTTON_UP) {
        return { kind: InputEventKind.ACTION_UP, source: raw.source, code: raw.code, value: 0 };
      }
    }
    // Gamepad button
    if (b.source === 'gamepad' && b.button != null && raw.code === `gp:${raw.padIndex ?? 0}:button:${b.button}`) {
      if (raw.kind === RawInputKind.BUTTON_DOWN) {
        return { kind: InputEventKind.ACTION_DOWN, source: raw.source, code: raw.code, value: 1 };
      }
      if (raw.kind === RawInputKind.BUTTON_UP) {
        return { kind: InputEventKind.ACTION_UP, source: raw.source, code: raw.code, value: 0 };
      }
    }
    // Gamepad axis
    if (b.source === 'gamepad' && b.axis != null && raw.kind === RawInputKind.AXIS && raw.code === `gp:${raw.padIndex ?? 0}:axis:${b.axis}`) {
      const dz = b.deadzone ?? 0.1;
      const v = Math.abs(raw.value) < dz ? 0 : raw.value;
      // If the binding has a sign, treat as a button (e.g. "left stick up = jump")
      if (b.sign != null) {
        const triggered = (b.sign > 0 ? v > 0.5 : v < -0.5);
        return {
          kind: triggered ? InputEventKind.ACTION_DOWN : InputEventKind.ACTION_UP,
          source: raw.source,
          code: raw.code,
          value: triggered ? 1 : 0,
        };
      }
      return { kind: InputEventKind.AXIS, source: raw.source, code: raw.code, value: v };
    }
    // Mouse / touch fall through: not implemented in this phase
    return null;
  }
}

// ── Convenience presets ───────────────────────────────────────────
// Two starter profiles; both intentionally minimal. The full game-
// stream profile (with all 30+ Luanti bindings) lives alongside the
// stage UI in a later phase. These are here so the bus is useful for
// non-game consumers from day 1.

export function navigationProfile() {
  // Designed for surfaces like the gallery, settings menu, etc.
  return new ActionProfile('navigation', {
    'menu_up':     [{ source: 'keyboard', code: 'ArrowUp' },    { source: 'gamepad', axis: 1, sign: -1 }, { source: 'gamepad', button: 12 }],
    'menu_down':   [{ source: 'keyboard', code: 'ArrowDown' },  { source: 'gamepad', axis: 1, sign: +1 }, { source: 'gamepad', button: 13 }],
    'menu_left':   [{ source: 'keyboard', code: 'ArrowLeft' },  { source: 'gamepad', axis: 0, sign: -1 }, { source: 'gamepad', button: 14 }],
    'menu_right':  [{ source: 'keyboard', code: 'ArrowRight' }, { source: 'gamepad', axis: 0, sign: +1 }, { source: 'gamepad', button: 15 }],
    'menu_select': [{ source: 'keyboard', code: 'Enter' },      { source: 'gamepad', button: 0 }],
    'menu_back':   [{ source: 'keyboard', code: 'Escape' },     { source: 'gamepad', button: 1 }],
  });
}

export function mediaProfile() {
  // Shoulder buttons for prev/next, X to play/pause. Designed to
  // co-exist with other consumers on the same bus.
  return new ActionProfile('media', {
    'media_prev':       [{ source: 'gamepad', button: 4 }, { source: 'keyboard', code: 'MediaTrackPrevious' }],
    'media_next':       [{ source: 'gamepad', button: 5 }, { source: 'keyboard', code: 'MediaTrackNext' }],
    'media_toggle':     [{ source: 'gamepad', button: 2 }, { source: 'keyboard', code: 'MediaPlayPause' }],
    'push_to_talk':     [{ source: 'gamepad', button: 6 }, { source: 'keyboard', code: 'KeyT' }],
  });
}
