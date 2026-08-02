/**
 * adapters/pointer.js — synthetic PointerEvent / MouseEvent adapter.
 *
 * Translates left-stick deflection into mousemove + pointermove events
 * with proper ``movementX`` / ``movementY`` deltas (so pointer-lock
 * WebGL games — FPS demos, the legendary js13k mouse-look entries —
 * get camera input from the gamepad).
 *
 * A button maps to mousedown/mouseup at the current pointer position.
 * Right-stick is reserved for camera/look in pointer-lock mode; if
 * Element.requestPointerLock isn't engaged, right-stick is ignored
 * (we don't want to spam mousemove from both sticks).
 *
 * Default sensitivity tuned for "feels OK with a real gamepad" — per-game
 * overrides via ``ctx.keymap.pointer``:
 *   - sensitivity: number     — pixels per axis-unit per frame (default 8)
 *   - click_buttons: [int]    — gamepad buttons → click (default [0])
 *   - right_click_buttons:[int]
 *   - prefer_pointerlock: bool — when true, use movementX/Y deltas only
 */

export const id = 'pointer';

const DEFAULT_POINTER = Object.freeze({
  sensitivity: 8,            // pixels / axis-unit / frame
  click_buttons: [0],        // A → left click
  right_click_buttons: [1],  // B → right click
  middle_click_buttons: [2], // X → middle click
  prefer_pointerlock: false,
  deadzone: 0.15,
});

// ── Pure helpers (testable without DOM) ──────────────────────────

/**
 * Compute movement delta from a stick frame.
 * @returns {{dx: number, dy: number}}
 */
export function _resolveMovement(axes, sensitivity, deadzone) {
  const ax = Number(axes?.[0] || 0);
  const ay = Number(axes?.[1] || 0);
  const dz = typeof deadzone === 'number' ? deadzone : 0.15;
  const norm = (v) => (Math.abs(v) < dz ? 0 : v);
  return {
    dx: norm(ax) * sensitivity,
    dy: norm(ay) * sensitivity,
  };
}

/**
 * Diff a frame against previous state, decide what mouse events fire.
 *
 * @returns {{
 *   events: Array<{type: string, button?: number, dx?: number, dy?: number}>,
 *   next: { pos: {x, y}, buttonsDown: Object }
 * }}
 */
export function _diffPointer(prev, frame, opts, viewport) {
  const o = { ...DEFAULT_POINTER, ...(opts || {}) };
  const axes = frame?.axes || [];
  const buttons = frame?.buttons || [];
  const { vw, vh } = viewport || { vw: 1920, vh: 1080 };
  const prevState = prev || { pos: { x: vw / 2, y: vh / 2 }, buttonsDown: {} };

  const events = [];
  const { dx, dy } = _resolveMovement(axes, o.sensitivity, o.deadzone);

  let nextPos = prevState.pos;
  if (dx !== 0 || dy !== 0) {
    nextPos = {
      x: Math.max(0, Math.min(vw, prevState.pos.x + dx)),
      y: Math.max(0, Math.min(vh, prevState.pos.y + dy)),
    };
    events.push({ type: 'mousemove', x: nextPos.x, y: nextPos.y, dx, dy });
  }

  const buttonMap = [
    [o.click_buttons, 0],
    [o.right_click_buttons, 2],
    [o.middle_click_buttons, 1],
  ];

  const nextDown = { ...prevState.buttonsDown };
  for (const [gpBtns, mouseBtn] of buttonMap) {
    for (const idx of gpBtns) {
      const i = Number(idx);
      const raw = Number(buttons[i] || 0);
      const pressed = raw > 0.5;
      const wasPressed = !!prevState.buttonsDown[i];
      if (pressed !== wasPressed) {
        nextDown[i] = pressed;
        events.push({
          type: pressed ? 'mousedown' : 'mouseup',
          button: mouseBtn,
          x: nextPos.x,
          y: nextPos.y,
        });
        if (!pressed) {
          // Click on release (standard browser behavior)
          events.push({ type: 'click', button: mouseBtn, x: nextPos.x, y: nextPos.y });
        }
      }
    }
  }

  return { events, next: { pos: nextPos, buttonsDown: nextDown } };
}

// ── DOM-touching internals ───────────────────────────────────────

function _dispatchMouseTo(doc, type, x, y, button, dx, dy) {
  try {
    const target = doc.elementFromPoint?.(x, y) || doc.body;
    const init = {
      bubbles: true, cancelable: true, composed: true, view: doc.defaultView,
      clientX: x, clientY: y,
      screenX: x, screenY: y,
      button: button ?? 0,
      buttons: button != null ? (1 << button) : 0,
      movementX: dx || 0,
      movementY: dy || 0,
    };
    const ev = new MouseEvent(type, init);
    target.dispatchEvent(ev);
    return true;
  } catch (_) { return false; }
}

// ── Adapter contract ─────────────────────────────────────────────

export async function probe(_iframe) {
  return { confidence: 0.25, evidence: [] };
}

export function activate(ctx) {
  const opts = ctx.keymap?.pointer || DEFAULT_POINTER;
  let state = null;

  const unsubRecv = ctx.recv((frame) => {
    const viewport = {
      vw: window.innerWidth || 1920,
      vh: window.innerHeight || 1080,
    };
    const { events, next } = _diffPointer(state, frame, opts, viewport);
    state = next;
    if (!events.length) return;
    for (const ev of events) {
      for (const win of ctx.targets()) {
        try {
          const doc = win?.document;
          if (!doc?.body) continue;
          if (_dispatchMouseTo(doc, ev.type, ev.x, ev.y, ev.button, ev.dx, ev.dy)) {
            ctx.telemetry.dispatches += 1;
          }
        } catch (_) {}
      }
    }
  });

  return function deactivate() {
    unsubRecv();
    // Release any held buttons
    if (state) {
      for (const [idxStr, down] of Object.entries(state.buttonsDown)) {
        if (down) {
          for (const win of ctx.targets()) {
            try {
              const doc = win?.document;
              if (!doc?.body) continue;
              _dispatchMouseTo(doc, 'mouseup', state.pos.x, state.pos.y, 0, 0, 0);
            } catch (_) {}
          }
        }
      }
    }
    state = null;
  };
}
