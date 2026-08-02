// ui/scripts/input/index.js
//
// Public surface for the app-wide input layer. Import from here, not
// the inner files, so call sites don't churn when internals move.
//
// Typical wiring at app start:
//
//   import {
//     inputBus, ActionProfile, navigationProfile,
//     KeyboardSource, GamepadSource,
//   } from './input/index.js';
//
//   inputBus.attachSource('keyboard', new KeyboardSource());
//   inputBus.attachSource('gamepad', new GamepadSource());
//   inputBus.setProfile(navigationProfile());
//
//   inputBus.on('menu_select', () => openCurrent());
//
// Surfaces that own their own profile (e.g. the game stage) push their
// profile via inputBus.setProfile() while active and restore the
// previous one on close.

export {
  InputBus,
  inputBus,
  InputEventKind,
  RawInputKind,
} from './input-bus.js';

export {
  ActionProfile,
  navigationProfile,
  mediaProfile,
} from './action-profile.js';

export { KeyboardSource } from './sources/keyboard.js';
export { GamepadSource } from './sources/gamepad.js';
