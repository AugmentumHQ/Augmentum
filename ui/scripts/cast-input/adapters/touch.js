/**
 * adapters/touch.js — synthetic TouchEvent adapter.
 *
 * Translates left-stick deflection into a virtual touch point that
 * follows the stick around a notional radius from a fixed center on
 * screen. A/B buttons fire tap events at the current touch position.
 *
 * Phase 1 ships a sensible default: stick controls touch position,
 * trigger buttons fire taps. Touch-game-shaped (rhythm, drag-and-drop,
 * mobile platformers) work for free as long as they listen for the
 * standard ``touchstart``/``touchmove``/``touchend`` events on the
 * document or a known canvas.
 *
 * Per-game customisation via ``ctx.keymap.touch``:
 *   - center: {x, y}     — touch origin in viewport coords
 *   - radius: number     — pixels at full stick deflection
 *   - tap_buttons: [int] — gamepad buttons that fire taps (default [0])
 *   - hold_threshold: number — stick magnitude above which touch is "down"
 *
 * Implementation note: TouchEvent + Touch are constructible in modern
 * browsers; we fall back to a no-op when not. Pointer events as a
 * cross-cutting fallback are handled by adapters/pointer.js.
 */

export const id = 'touch';

const DEFAULT_TOUCH = Object.freeze({
  center: { x: 0.5, y: 0.5 },    // fractions of viewport (resolved at dispatch)
  radius: 0.25,                    // fraction of min(width, height)
  tap_buttons: [0, 1],             // A + B
  hold_threshold: 0.3,             // stick magnitude
});

// ── Pure helpers (testable without DOM) ──────────────────────────

/**
 * Compute a touch position from stick axes.
 * @param {number[]} axes   — full gamepad axes array
 * @param {{x: number, y: number}} centerFrac — center as fraction of viewport
 * @param {number} radiusFrac — radius as fraction of min(width, height)
 * @param {number} vw — viewport width
 * @param {number} vh — viewport height
 * @returns {{x: number, y: number, magnitude: number}}
 */
export function _resolveTouchPos(axes, centerFrac, radiusFrac, vw, vh) {
  const ax = Number(axes?.[0] || 0);
  const ay = Number(axes?.[1] || 0);
  const magnitude = Math.min(1, Math.sqrt(ax * ax + ay * ay));
  const radPx = radiusFrac * Math.min(vw, vh);
  const x = centerFrac.x * vw + ax * radPx;
  const y = centerFrac.y * vh + ay * radPx;
  return { x, y, magnitude };
}

/**
 * Diff frame against previous state, decide which touch lifecycle
 * events should fire. Edge-triggered on hold_threshold + tap buttons.
 *
 * @returns {{
 *   events: Array<{type: 'touchstart'|'touchmove'|'touchend'|'tap', x?: number, y?: number}>,
 *   next: { isDown: boolean, lastPos: {x, y}|null, tapButtonsHeld: Object }
 * }}
 */
export function _diffTouch(prev, frame, opts, viewport) {
  const o = { ...DEFAULT_TOUCH, ...(opts || {}) };
  const axes = frame?.axes || [];
  const buttons = frame?.buttons || [];
  const { vw, vh } = viewport || { vw: 1920, vh: 1080 };
  const pos = _resolveTouchPos(axes, o.center, o.radius, vw, vh);

  const prevState = prev || { isDown: false, lastPos: null, tapButtonsHeld: {} };
  const events = [];
  const next = {
    isDown: prevState.isDown,
    lastPos: prevState.lastPos,
    tapButtonsHeld: { ...prevState.tapButtonsHeld },
  };

  // Stick-driven touch lifecycle
  const wantDown = pos.magnitude >= o.hold_threshold;
  if (wantDown && !prevState.isDown) {
    events.push({ type: 'touchstart', x: pos.x, y: pos.y });
    next.isDown = true;
    next.lastPos = { x: pos.x, y: pos.y };
  } else if (!wantDown && prevState.isDown) {
    const endPos = prevState.lastPos || { x: pos.x, y: pos.y };
    events.push({ type: 'touchend', x: endPos.x, y: endPos.y });
    next.isDown = false;
    next.lastPos = null;
  } else if (wantDown) {
    // touchmove only when actually moved (avoid 60Hz spam from a held stick)
    const last = prevState.lastPos;
    const dx = Math.abs(pos.x - (last?.x ?? pos.x));
    const dy = Math.abs(pos.y - (last?.y ?? pos.y));
    if (dx > 1 || dy > 1) {
      events.push({ type: 'touchmove', x: pos.x, y: pos.y });
      next.lastPos = { x: pos.x, y: pos.y };
    }
  }

  // Tap-button edges → tap at center (or current touch pos if held)
  for (const idx of o.tap_buttons) {
    const i = Number(idx);
    const raw = Number(buttons[i] || 0);
    const pressed = raw > 0.5;
    const wasPressed = !!prevState.tapButtonsHeld[i];
    if (pressed !== wasPressed) {
      next.tapButtonsHeld[i] = pressed;
      const tapX = next.lastPos?.x ?? o.center.x * vw;
      const tapY = next.lastPos?.y ?? o.center.y * vh;
      events.push({
        type: pressed ? 'touchstart' : 'touchend',
        x: tapX,
        y: tapY,
        synthetic_tap: true,
      });
    }
  }

  return { events, next };
}

// ── DOM-touching internals ───────────────────────────────────────

let _identifierCounter = 1000;

function _newIdentifier() {
  _identifierCounter += 1;
  return _identifierCounter;
}

function _constructTouchEvent(type, doc, x, y, identifier) {
  // Touch / TouchEvent constructors exist in Chrome on Android,
  // Firefox, Safari — and notably NOT in stock desktop Chrome unless
  // touch events are explicitly enabled. We feature-detect once and
  // fall back to a synthetic Event with the touches list attached.
  try {
    const touch = new Touch({
      identifier,
      target: doc.body,
      clientX: x, clientY: y,
      screenX: x, screenY: y,
      pageX: x, pageY: y,
      radiusX: 8, radiusY: 8,
      rotationAngle: 0,
      force: 1,
    });
    const touches = type === 'touchend' ? [] : [touch];
    const targetTouches = type === 'touchend' ? [] : [touch];
    const changedTouches = [touch];
    return new TouchEvent(type, {
      touches, targetTouches, changedTouches,
      bubbles: true, cancelable: true, composed: true,
    });
  } catch (_) {
    // Fallback: plain Event with .touches attached. Most games will
    // ignore this; that's the correct outcome on a platform without
    // real touch support.
    const ev = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(ev, 'touches', {
      value: type === 'touchend' ? [] : [{ clientX: x, clientY: y, identifier }],
    });
    return ev;
  }
}

// ── Adapter contract ─────────────────────────────────────────────

export async function probe(_iframe) {
  return { confidence: 0.2, evidence: [] };
}

export function activate(ctx) {
  const opts = ctx.keymap?.touch || DEFAULT_TOUCH;
  let state = null;
  let identifier = _newIdentifier();

  const unsubRecv = ctx.recv((frame) => {
    const viewport = {
      vw: window.innerWidth || 1920,
      vh: window.innerHeight || 1080,
    };
    const { events, next } = _diffTouch(state, frame, opts, viewport);
    state = next;
    if (!events.length) return;
    for (const ev of events) {
      // Rotate identifier on lifecycle start so each touch sequence
      // gets a unique id (some games key tracking by identifier).
      if (ev.type === 'touchstart') identifier = _newIdentifier();
      for (const win of ctx.targets()) {
        try {
          const doc = win?.document;
          if (!doc?.body) continue;
          const target = doc.elementFromPoint?.(ev.x, ev.y) || doc.body;
          const touchEv = _constructTouchEvent(ev.type, doc, ev.x, ev.y, identifier);
          target.dispatchEvent(touchEv);
          ctx.telemetry.dispatches += 1;
        } catch (_) { /* cross-realm or no Touch ctor */ }
      }
    }
  });

  return function deactivate() {
    unsubRecv();
    // Release any in-flight touch
    if (state?.isDown) {
      const last = state.lastPos || { x: 0, y: 0 };
      for (const win of ctx.targets()) {
        try {
          const doc = win?.document;
          if (!doc?.body) continue;
          const ev = _constructTouchEvent('touchend', doc, last.x, last.y, identifier);
          doc.body.dispatchEvent(ev);
        } catch (_) {}
      }
    }
    state = null;
  };
}
