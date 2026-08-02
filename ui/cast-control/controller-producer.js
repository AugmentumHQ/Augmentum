/**
 * controller-producer.js — phone-side WS producer for cast game input.
 *
 * Opens a WebSocket against /api/cast/input/ws?session_id=...&pad_index=N
 * when a streamed game has been cast, polls navigator.getGamepads() at
 * 60Hz, sends state-delta frames, and dispatches inbound rumble events
 * to the corresponding pad's vibrationActuator.
 *
 * Lifecycle is owned by cast-control.js — call start() after a cast_game
 * launch succeeds, stop() when the cast ends or the user navigates away.
 *
 * Wire protocol matches augmentum/cast/input_bridge.py:
 *   client → server: { seq, t_send, event: { kind: 'gamepad_state',
 *                                            pad_index, buttons, axes } }
 *   server → client: { kind: 'rumble', slot, duration_ms, strong, weak }
 */

const POLL_MS = 16;  // ~60Hz; below this the kernel input poller may
                     // miss a button press that's only down for one tick.
const SEND_KEEPALIVE_MS = 2000;  // floor so the server's "phone alive"
                                 // signal doesn't lapse during idle play

// Standard Gamepad API maps the first 17 buttons; we tolerate >17 by
// silently dropping the tail since the container daemon also caps at 17.
const NUM_BUTTONS = 17;
const NUM_AXES = 4;

const _producers = new Map();   // session_id → ControllerProducer

class ControllerProducer {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.ws = null;
    this.seq = 0;
    this.pollTimer = null;
    // Per-(pad_index) cache of last-sent state so we only ship deltas.
    // pad_index → { buttons:[], axes:[], lastSendMs }
    this._padState = new Map();
    this._stopped = false;
    this._onStatusChange = null;
    this._wsReadyAt = 0;
  }

  setStatusListener(fn) { this._onStatusChange = fn; }

  start() {
    if (this.ws) return;  // already connected
    // Defer open until at least one gamepad is connected. If none is
    // connected yet, listen for the connect event and open then. This
    // matches the "controller required" pre-play nudge: the user pairs
    // their controller before the producer comes alive.
    if (this._anyGamepadConnected()) {
      this._openWs(0);
    } else {
      window.addEventListener('gamepadconnected', this._onConnect, { once: true });
    }
  }

  stop(reason = 'stop') {
    if (this._stopped) return;
    this._stopped = true;
    window.removeEventListener('gamepadconnected', this._onConnect);
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.ws) {
      try { this.ws.close(1000, reason); } catch {}
      this.ws = null;
    }
    _producers.delete(this.sessionId);
    this._notify({ state: 'disconnected', reason });
  }

  _onConnect = (ev) => {
    if (this._stopped) return;
    const pad = ev.gamepad;
    this._openWs(pad?.index || 0);
  };

  _anyGamepadConnected() {
    try {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      for (const p of pads) {
        if (p && p.connected) return true;
      }
    } catch {}
    return false;
  }

  _openWs(padIndex) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/api/cast/input/ws`
      + `?session_id=${encodeURIComponent(this.sessionId)}`
      + `&pad_index=${encodeURIComponent(String(padIndex))}`;
    this._notify({ state: 'connecting' });
    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      this._notify({ state: 'error', message: String(err) });
      return;
    }
    this.ws.addEventListener('open', () => {
      this._wsReadyAt = performance.now();
      this._notify({ state: 'connected' });
      this.pollTimer = setInterval(() => this._tick(), POLL_MS);
    });
    this.ws.addEventListener('message', (ev) => this._onMessage(ev));
    this.ws.addEventListener('close', (ev) => {
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
      this._notify({ state: 'disconnected', code: ev.code, reason: ev.reason });
      // Auto-reconnect once if we weren't stopped intentionally and
      // the disconnect looks transient (1006 = abnormal closure).
      if (!this._stopped && (ev.code === 1006 || ev.code === 1001)) {
        setTimeout(() => {
          if (!this._stopped) this._openWs(padIndex);
        }, 1000);
      }
    });
    this.ws.addEventListener('error', () => {
      // close handler runs right after; no separate state update needed
    });
  }

  _onMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg || typeof msg !== 'object') return;
    if (msg.kind === 'rumble') {
      this._handleRumble(msg);
    }
  }

  _handleRumble({ slot, duration_ms, strong, weak }) {
    // Find the gamepad whose pad_index matches the slot we claimed for
    // it. Cheapest approach: try the Gamepad API's index === slot first
    // (the index strategy maps 1:1), fall back to the first connected
    // pad with a vibrationActuator (covers firstpress slots that may
    // not match index).
    const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
    let target = pads[slot];
    if (!target || !target.connected) {
      target = pads.find(p => p && p.connected && p.vibrationActuator);
    }
    const actuator = target?.vibrationActuator;
    if (!actuator || typeof actuator.playEffect !== 'function') return;
    // Clamp inputs to spec-safe ranges. Some browsers reject NaN or
    // out-of-band values by throwing inside the promise.
    const duration = Math.max(0, Math.min(5000, Number(duration_ms) || 0));
    const strongMagnitude = Math.max(0, Math.min(1, Number(strong) || 0));
    const weakMagnitude = Math.max(0, Math.min(1, Number(weak) || 0));
    try {
      actuator.playEffect('dual-rumble', {
        duration, strongMagnitude, weakMagnitude, startDelay: 0,
      });
    } catch {}
  }

  _tick() {
    if (this._stopped || !this.ws || this.ws.readyState !== 1) return;
    const now = performance.now();
    const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
    for (const pad of pads) {
      if (!pad || !pad.connected) continue;
      const buttons = new Array(NUM_BUTTONS).fill(0);
      const axes = new Array(NUM_AXES).fill(0);
      for (let i = 0; i < Math.min(NUM_BUTTONS, pad.buttons.length); i++) {
        const b = pad.buttons[i];
        buttons[i] = (b && (b.pressed || b.value > 0.5)) ? 1 : 0;
      }
      for (let i = 0; i < Math.min(NUM_AXES, pad.axes.length); i++) {
        const v = pad.axes[i];
        // Round to 2 decimals to suppress stick-jitter frames.
        axes[i] = Math.round(Number(v || 0) * 100) / 100;
      }
      const cache = this._padState.get(pad.index) || {
        buttons: new Array(NUM_BUTTONS).fill(0),
        axes: new Array(NUM_AXES).fill(0),
        lastSendMs: 0,
      };
      const changed = !_eqArr(buttons, cache.buttons)
        || !_eqArr(axes, cache.axes);
      const keepalive = now - cache.lastSendMs >= SEND_KEEPALIVE_MS;
      if (!changed && !keepalive) continue;
      cache.buttons = buttons;
      cache.axes = axes;
      cache.lastSendMs = now;
      this._padState.set(pad.index, cache);
      this.seq += 1;
      try {
        this.ws.send(JSON.stringify({
          seq: this.seq,
          t_send: now,
          event: {
            kind: 'gamepad_state',
            pad_index: pad.index,
            buttons,
            axes,
          },
        }));
      } catch {
        // WS dead — close handler will fire and we'll auto-reconnect.
        break;
      }
    }
  }

  _notify(payload) {
    if (typeof this._onStatusChange === 'function') {
      try { this._onStatusChange({ sessionId: this.sessionId, ...payload }); }
      catch {}
    }
  }
}

function _eqArr(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/* ── Public API ─────────────────────────────────────────────────────── */

/**
 * Start streaming this device's gamepad state to the named game-stream
 * session. Idempotent — calling twice for the same session is a no-op.
 *
 * @param {string} sessionId  AGSP session id returned by /api/cast/games/.../start
 * @param {(status: object) => void} [onStatus]  Lifecycle callback.
 * @returns {ControllerProducer}
 */
export function startProducer(sessionId, onStatus) {
  if (!sessionId) return null;
  const existing = _producers.get(sessionId);
  if (existing) {
    if (onStatus) existing.setStatusListener(onStatus);
    return existing;
  }
  const p = new ControllerProducer(sessionId);
  if (onStatus) p.setStatusListener(onStatus);
  _producers.set(sessionId, p);
  p.start();
  return p;
}

/**
 * Stop any active producer for this session. No-op if none running.
 */
export function stopProducer(sessionId, reason) {
  const p = _producers.get(sessionId);
  if (p) p.stop(reason || 'stop');
}

/**
 * Stop every active producer. Called on phone tab-close / cast-ended.
 */
export function stopAllProducers() {
  for (const p of [..._producers.values()]) p.stop('shutdown');
}

/**
 * True iff at least one Gamepad API controller is currently connected.
 * Used by the pre-play nudge to decide whether to show "pair a
 * controller in Bluetooth settings".
 */
export function hasConnectedGamepad() {
  try {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      if (p && p.connected) return true;
    }
  } catch {}
  return false;
}
