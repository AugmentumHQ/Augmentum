/**
 * keymap-defaults.js — built-in keymap presets per game genre.
 *
 * The classifier (Phase 2) picks one of these by ``CastProfile.keymap.preset``
 * when no explicit per-game mapping exists. Authors can override any
 * field; missing fields fall back to the adapter's own DEFAULT_*.
 *
 * Adding a preset: drop a new entry and document the target genre. The
 * registry is open — unknown preset names just fall back to retro_dpad.
 */

export const KEYMAP_PRESETS = Object.freeze({
  // NES/SNES era — 2-button + dpad; A/B map to Z/X (EmulatorJS-flavored
  // defaults so existing muscle memory transfers).
  retro_dpad: {
    keyboard: {
      buttons: {
        0: 'KeyZ',          // A
        1: 'KeyX',          // B
        2: 'KeyA',          // X (extra face button on SNES+)
        3: 'KeyS',          // Y
        8: 'ShiftRight',    // Select
        9: 'Enter',         // Start
        12: 'ArrowUp', 13: 'ArrowDown',
        14: 'ArrowLeft', 15: 'ArrowRight',
      },
      axes: {
        0: { negative: 'ArrowLeft', positive: 'ArrowRight' },
        1: { negative: 'ArrowUp', positive: 'ArrowDown' },
      },
      deadzone: 0.5,
    },
  },

  // Roguelike / text-adventure / vi-keys games.
  roguelike: {
    keyboard: {
      buttons: {
        0: 'Enter',         // A → enter/confirm
        1: 'Escape',        // B → back
        2: 'KeyI',          // X → inventory
        3: 'KeyM',          // Y → map
        9: 'KeyQ',          // Start → quit/menu
        12: 'KeyK', 13: 'KeyJ', 14: 'KeyH', 15: 'KeyL',  // vi-keys
      },
      axes: {
        0: { negative: 'KeyH', positive: 'KeyL' },
        1: { negative: 'KeyK', positive: 'KeyJ' },
      },
      deadzone: 0.5,
    },
  },

  // WASD platformer / shmup — most jam games + js13k entries.
  wasd_action: {
    keyboard: {
      buttons: {
        0: 'Space',         // A → jump/action
        1: 'ShiftLeft',     // B → run/shoot
        2: 'KeyE',          // X → use
        3: 'KeyR',          // Y → reload/secondary
        9: 'Escape',        // Start → menu/pause
        12: 'KeyW', 13: 'KeyS', 14: 'KeyA', 15: 'KeyD',
      },
      axes: {
        0: { negative: 'KeyA', positive: 'KeyD' },
        1: { negative: 'KeyW', positive: 'KeyS' },
      },
      deadzone: 0.4,
    },
  },

  // Mobile-first touch games: stick = single touch point, A/B = taps.
  touch_default: {
    touch: {
      center: { x: 0.5, y: 0.5 },
      radius: 0.3,
      tap_buttons: [0, 1],
      hold_threshold: 0.3,
    },
  },

  // Pointer-lock FPS / mouse-look webgl. Left stick = look, A = fire.
  pointer_fps: {
    pointer: {
      sensitivity: 12,
      click_buttons: [0],          // A → primary fire
      right_click_buttons: [1],    // B → aim
      middle_click_buttons: [2],   // X → middle
      prefer_pointerlock: true,
      deadzone: 0.15,
    },
  },
});

export function presetFor(name) {
  return KEYMAP_PRESETS[name] || KEYMAP_PRESETS.retro_dpad;
}
