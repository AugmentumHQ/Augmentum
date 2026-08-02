// tests/test_cast_input_adapters_js.mjs
//
// Pure-Node tests for the universal cast input adapter pipeline.
// We exercise the *pure* helpers exported from each adapter:
//   - _applyFrame / _clampSlot (gamepad_api)
//   - _diffChanges / _heldKeys / _codeToKey (keyboard)
//   - _resolveTouchPos / _diffTouch (touch)
//   - _resolveMovement / _diffPointer (pointer)
//
// The DOM-touching ``activate()`` paths can't be exercised without a
// browser realm, but every adapter's interesting logic lives in the
// pure helpers above — keep that property intentional.
//
// Run with:
//   node tests/test_cast_input_adapters_js.mjs

import {
  _applyFrame,
  _newPad,
  _clampSlot,
} from '../ui/scripts/cast-input/adapters/gamepad_api.js';

import {
  _diffChanges as kb_diffChanges,
  _heldKeys as kb_heldKeys,
  _codeToKey as kb_codeToKey,
} from '../ui/scripts/cast-input/adapters/keyboard.js';

import {
  _resolveTouchPos,
  _diffTouch,
} from '../ui/scripts/cast-input/adapters/touch.js';

import {
  _resolveMovement,
  _diffPointer,
} from '../ui/scripts/cast-input/adapters/pointer.js';

import { KEYMAP_PRESETS, presetFor } from '../ui/scripts/cast-input/keymap-defaults.js';

let _failed = 0;
let _ran = 0;

function assert(cond, label) {
  _ran++;
  if (cond) { console.log(`PASS ${label}`); }
  else { _failed++; console.error(`FAIL ${label}`); }
}

function assertEq(actual, expected, label) {
  assert(
    JSON.stringify(actual) === JSON.stringify(expected),
    `${label}\n  expected: ${JSON.stringify(expected)}\n    actual: ${JSON.stringify(actual)}`,
  );
}

// performance.now is needed by gamepad_api's _applyFrame; Node ≥16 has it,
// older / non-perf builds need a shim.
if (typeof performance === 'undefined') {
  // eslint-disable-next-line no-global-assign
  globalThis.performance = { now: () => Date.now() };
}


// ── gamepad_api ───────────────────────────────────────────────────


(function gpClampSlot() {
  assertEq(_clampSlot(0), 0, 'clampSlot keeps 0');
  assertEq(_clampSlot(3), 3, 'clampSlot keeps 3');
  assertEq(_clampSlot(-1), 0, 'clampSlot floors negatives to 0');
  assertEq(_clampSlot(7), 3, 'clampSlot caps above-max to 3');
  assertEq(_clampSlot('2'), 2, 'clampSlot coerces strings');
  assertEq(_clampSlot('foo'), 0, 'clampSlot treats NaN as 0');
})();


(function gpApplyFrame() {
  const pad = _newPad(0);
  _applyFrame(pad, [1, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.7, -0.5, 0, 0]);
  assert(pad.buttons[0].pressed === true, 'A button pressed');
  assert(pad.buttons[0].value === 1, 'A button value=1');
  assert(pad.buttons[6].pressed === true, 'LT pressed (analog > 0.5)');
  assert(pad.buttons[6].value === 0.6, 'LT analog value preserved');
  assert(pad.buttons[1].pressed === false, 'B not pressed');
  assertEq(pad.axes[0], 0.7, 'axis 0 = 0.7');
  assertEq(pad.axes[1], -0.5, 'axis 1 = -0.5');

  // Frame update reuses the same button objects (identity-preserving)
  const btnRef = pad.buttons[0];
  _applyFrame(pad, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0]);
  assert(pad.buttons[0] === btnRef, 'button object identity preserved across frames');
  assert(pad.buttons[0].pressed === false, 'A button released');
})();


(function gpApplyFrameNullsCoerce() {
  const pad = _newPad(1);
  _applyFrame(pad, null, null);
  assert(pad.buttons.every(b => b.value === 0), 'null buttons → zeros');
  assert(pad.axes.every(v => v === 0), 'null axes → zeros');
})();


// ── keyboard ──────────────────────────────────────────────────────


(function kbCodeToKey() {
  assertEq(kb_codeToKey('KeyZ'), 'z', 'KeyZ → z');
  assertEq(kb_codeToKey('ArrowUp'), 'ArrowUp', 'ArrowUp passes through');
  assertEq(kb_codeToKey('Enter'), 'Enter', 'Enter passes through');
  assertEq(kb_codeToKey('Space'), ' ', 'Space → space char');
  assertEq(kb_codeToKey('ShiftLeft'), 'Shift', 'ShiftLeft → Shift');
  assertEq(kb_codeToKey('UnknownCode'), 'UnknownCode', 'unknown codes pass through');
})();


(function kbDiffButtonEdge() {
  const buttons = new Array(17).fill(0); buttons[0] = 1; // A pressed
  const frame = { buttons, axes: [0, 0, 0, 0] };
  const prev = { buttons: new Array(17).fill(false), axes: {} };
  const { changes, next } = kb_diffChanges(prev, frame, null);
  assertEq(changes.length, 1, 'one change on rising edge');
  assertEq(changes[0], { code: 'KeyZ', down: true }, 'A → KeyZ down');
  assert(next.buttons[0] === true, 'next state tracks A held');

  // Idempotent across same-state frames
  const r2 = kb_diffChanges(next, frame, null);
  assertEq(r2.changes.length, 0, 'no change when state unchanged');

  // Release edge
  const releaseFrame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  const r3 = kb_diffChanges(next, releaseFrame, null);
  assertEq(r3.changes, [{ code: 'KeyZ', down: false }], 'release fires keyup');
})();


(function kbDiffAxisEdge() {
  const frame = { buttons: new Array(17).fill(0), axes: [0.9, 0, 0, 0] };  // hard-right
  const prev = { buttons: new Array(17).fill(false), axes: {} };
  const { changes, next } = kb_diffChanges(prev, frame, null);
  assertEq(changes, [{ code: 'ArrowRight', down: true }], 'positive axis → ArrowRight down');
  assertEq(next.axes[0], { positive: true, negative: false }, 'axis 0 marked positive');

  // Releasing past deadzone fires keyup
  const restFrame = { buttons: new Array(17).fill(0), axes: [0.1, 0, 0, 0] };
  const r2 = kb_diffChanges(next, restFrame, null);
  assertEq(r2.changes, [{ code: 'ArrowRight', down: false }], 'inside deadzone → keyup');

  // Cross deadzone the other way
  const leftFrame = { buttons: new Array(17).fill(0), axes: [-0.8, 0, 0, 0] };
  const r3 = kb_diffChanges(r2.next, leftFrame, null);
  assertEq(r3.changes, [{ code: 'ArrowLeft', down: true }], 'negative axis → ArrowLeft down');
})();


(function kbDiffWithCustomKeymap() {
  const wasd = KEYMAP_PRESETS.wasd_action.keyboard;
  const frame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  frame.buttons[0] = 1;  // A
  const prev = { buttons: new Array(17).fill(false), axes: {} };
  const { changes } = kb_diffChanges(prev, frame, wasd);
  assertEq(changes, [{ code: 'Space', down: true }], 'A → Space under wasd preset');
})();


(function kbHeldKeysReleases() {
  const state = {
    buttons: new Array(17).fill(false),
    axes: { 0: { positive: true, negative: false } },
  };
  state.buttons[0] = true; state.buttons[9] = true;
  const held = kb_heldKeys(state, null);
  assert(held.includes('KeyZ'), 'KeyZ in held set (A button)');
  assert(held.includes('Enter'), 'Enter in held set (Start)');
  assert(held.includes('ArrowRight'), 'ArrowRight in held set (axis 0 positive)');
})();


(function kbButtonChangeWhilstSameFrame() {
  // Test that simultaneous button + axis edges all surface in one diff
  const frame = { buttons: new Array(17).fill(0), axes: [-0.9, 0.9, 0, 0] };
  frame.buttons[9] = 1; frame.buttons[0] = 1;  // Start + A
  const { changes } = kb_diffChanges(
    { buttons: new Array(17).fill(false), axes: {} },
    frame,
    null,
  );
  const codes = changes.map(c => c.code).sort();
  assertEq(codes, ['ArrowDown', 'ArrowLeft', 'Enter', 'KeyZ'],
    'simultaneous button + axis edges all fire');
})();


// ── touch ─────────────────────────────────────────────────────────


(function touchResolvePos() {
  const pos = _resolveTouchPos(
    [1, 0, 0, 0], { x: 0.5, y: 0.5 }, 0.25, 1000, 1000,
  );
  // Stick all the way right, radius=0.25 of min(1000,1000) = 250 → center + 250 = 750
  assertEq(pos.x, 750, 'stick right → x = 750');
  assertEq(pos.y, 500, 'no y deflection → y = 500');
  assertEq(pos.magnitude, 1, 'magnitude 1 at full deflection');
})();


(function touchResolvePosClampsMagnitude() {
  // Magnitude > 1 (shouldn't happen but be defensive)
  const pos = _resolveTouchPos([2, 2, 0, 0], { x: 0.5, y: 0.5 }, 0.25, 1000, 1000);
  assertEq(pos.magnitude, 1, 'magnitude clamped to 1');
})();


(function touchDiffLifecycle() {
  const viewport = { vw: 1000, vh: 1000 };
  const frame1 = { buttons: new Array(17).fill(0), axes: [0.8, 0, 0, 0] };
  const r1 = _diffTouch(null, frame1, null, viewport);
  assertEq(r1.events.length, 1, 'first hard-deflection emits one event');
  assertEq(r1.events[0].type, 'touchstart', 'first event is touchstart');
  assert(r1.next.isDown === true, 'state: touch down');

  // Same frame, no change → no events (touch holds at same pos)
  const r2 = _diffTouch(r1.next, frame1, null, viewport);
  assertEq(r2.events.length, 0, 'no event when stick stays in place');

  // Move the stick
  const frame2 = { buttons: new Array(17).fill(0), axes: [0.5, 0.5, 0, 0] };
  const r3 = _diffTouch(r1.next, frame2, null, viewport);
  assert(r3.events.some(e => e.type === 'touchmove'), 'movement emits touchmove');

  // Release stick
  const frame3 = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  const r4 = _diffTouch(r3.next, frame3, null, viewport);
  assertEq(r4.events[0].type, 'touchend', 'releasing stick fires touchend');
  assert(r4.next.isDown === false, 'state: no touch');
})();


(function touchTapButtonsEdge() {
  const viewport = { vw: 1000, vh: 1000 };
  const frame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  frame.buttons[0] = 1;
  const r = _diffTouch(null, frame, null, viewport);
  const tapStarts = r.events.filter(e => e.synthetic_tap && e.type === 'touchstart');
  assertEq(tapStarts.length, 1, 'tap button fires touchstart');

  // Release
  const releaseFrame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  const r2 = _diffTouch(r.next, releaseFrame, null, viewport);
  const tapEnds = r2.events.filter(e => e.synthetic_tap && e.type === 'touchend');
  assertEq(tapEnds.length, 1, 'tap release fires touchend');
})();


// ── pointer ───────────────────────────────────────────────────────


(function pointerResolveMovement() {
  const m1 = _resolveMovement([1, 0, 0, 0], 10, 0.15);
  assertEq(m1.dx, 10, 'stick right → dx=10');
  assertEq(m1.dy, 0, 'stick right → dy=0');

  const m2 = _resolveMovement([0.1, 0, 0, 0], 10, 0.15);
  assertEq(m2.dx, 0, 'inside deadzone → dx=0');

  const m3 = _resolveMovement([-0.5, 0.5, 0, 0], 8, 0.15);
  assertEq(m3.dx, -4, 'stick left → dx=-4');
  assertEq(m3.dy, 4, 'stick down → dy=4');
})();


(function pointerDiffMovementAccumulates() {
  const viewport = { vw: 1000, vh: 1000 };
  const frame = { buttons: new Array(17).fill(0), axes: [0.5, 0, 0, 0] };
  const r1 = _diffPointer(null, frame, null, viewport);
  assert(r1.events.some(e => e.type === 'mousemove'), 'movement emits mousemove');
  assert(r1.next.pos.x > 500, 'pos.x moved right of center');

  // Movement clamped to viewport
  let state = r1.next;
  const farFrame = { buttons: new Array(17).fill(0), axes: [1, 0, 0, 0] };
  for (let i = 0; i < 1000; i++) {
    const r = _diffPointer(state, farFrame, null, viewport);
    state = r.next;
  }
  assertEq(state.pos.x, viewport.vw, 'cursor pinned at viewport width');
})();


(function pointerDiffClickLifecycle() {
  const viewport = { vw: 1000, vh: 1000 };
  const frame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  frame.buttons[0] = 1;  // A → left click
  const r = _diffPointer(null, frame, null, viewport);
  const types = r.events.map(e => e.type);
  assert(types.includes('mousedown'), 'A press → mousedown');
  assert(!types.includes('click'), 'no click on press');
  assert(!types.includes('mouseup'), 'no mouseup on press');

  // Release
  const releaseFrame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  const r2 = _diffPointer(r.next, releaseFrame, null, viewport);
  const types2 = r2.events.map(e => e.type);
  assert(types2.includes('mouseup'), 'A release → mouseup');
  assert(types2.includes('click'), 'A release also fires click');
})();


(function pointerRightClick() {
  const viewport = { vw: 1000, vh: 1000 };
  const frame = { buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
  frame.buttons[1] = 1;  // B → right click
  const r = _diffPointer(null, frame, null, viewport);
  const mouseDown = r.events.find(e => e.type === 'mousedown');
  assertEq(mouseDown.button, 2, 'B → button 2 (right click)');
})();


// ── keymap-defaults ───────────────────────────────────────────────


(function presetsExist() {
  assert(KEYMAP_PRESETS.retro_dpad, 'retro_dpad preset exists');
  assert(KEYMAP_PRESETS.roguelike, 'roguelike preset exists');
  assert(KEYMAP_PRESETS.wasd_action, 'wasd_action preset exists');
  assert(KEYMAP_PRESETS.touch_default, 'touch_default preset exists');
  assert(KEYMAP_PRESETS.pointer_fps, 'pointer_fps preset exists');
})();


(function presetForFallback() {
  const p = presetFor('not_a_real_preset');
  assertEq(p, KEYMAP_PRESETS.retro_dpad, 'unknown name → retro_dpad fallback');
})();


(function presetSanityCheck() {
  // Every keyboard preset must map every dpad direction.
  const presets = ['retro_dpad', 'roguelike', 'wasd_action'];
  for (const name of presets) {
    const kb = KEYMAP_PRESETS[name].keyboard;
    assert(!!kb.buttons[12], `${name} maps dpad up`);
    assert(!!kb.buttons[13], `${name} maps dpad down`);
    assert(!!kb.buttons[14], `${name} maps dpad left`);
    assert(!!kb.buttons[15], `${name} maps dpad right`);
    assert(!!kb.axes[0]?.negative && !!kb.axes[0]?.positive,
      `${name} maps stick X axis`);
    assert(!!kb.axes[1]?.negative && !!kb.axes[1]?.positive,
      `${name} maps stick Y axis`);
  }
})();


// ── Summary ───────────────────────────────────────────────────────


console.log(`\n${_ran} ran, ${_failed} failed`);
process.exit(_failed === 0 ? 0 : 1);
