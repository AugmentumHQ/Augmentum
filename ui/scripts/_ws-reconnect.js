/**
 * _ws-reconnect.js — framework-free WebSocket reconnect scheduler.
 *
 * Replaces the recurring ad-hoc pattern where each WS consumer hand-rolled
 * a flat `setTimeout(reconnect, 4000)` in its `onclose`. A fixed delay has
 * two problems: it never backs off (so a server that's down hammers it
 * every 4s forever), and when many tabs/clients reconnect on the same
 * fixed interval they synchronize into a thundering herd. This module
 * applies **full-jitter exponential backoff** (the AWS-recommended shape):
 *
 *     ceiling = min(cap, base * 2^attempt)
 *     delay   = random(0, ceiling)
 *
 * — the ceiling doubles each failed attempt up to `cap`, and the actual
 * delay is a uniform random point in [0, ceiling] so independent clients
 * spread out instead of retrying in lockstep.
 *
 * Contract:
 *   - `connect()` opens the socket and returns it (or a Promise of it, or
 *     null/throw on failure). The reconnector attaches a one-shot `open`
 *     listener that resets the backoff, so the caller only has to wire
 *     `ws.onclose = () => r.schedule()`.
 *   - single in-flight: `schedule()` is a no-op while a retry is pending.
 *   - `stop()` (call on detach/unmount) cancels any pending retry and makes
 *     all further `schedule()`/`start()` calls no-ops; a connection that
 *     resolves after stop() is closed immediately.
 *
 * Usage:
 *
 *   import { createReconnector } from './_ws-reconnect.js';
 *
 *   const r = createReconnector({
 *     connect: () => openMyWebSocket(),   // returns ws | Promise<ws> | null
 *     base: 1000, cap: 30000, name: 'presence',
 *   });
 *   r.start();          // open now
 *   // inside openMyWebSocket: ws.onclose = () => r.schedule();
 *   // on unmount: r.stop();
 */

/**
 * Pre-jitter backoff ceiling for a given attempt (0-based). Pure — exported
 * for unit testing the formula deterministically.
 */
export function backoffCeiling(attempt, { base = 1000, cap = 30000 } = {}) {
  const a = attempt < 0 ? 0 : attempt;
  // 2^a can overflow for absurd attempt counts; clamp via the cap anyway.
  const expo = base * Math.pow(2, a);
  return Math.min(cap, expo);
}

/**
 * Full-jitter delay for a given attempt: a uniform random point in
 * [0, backoffCeiling(attempt)]. `rng` is injectable for tests.
 */
export function fullJitterDelay(attempt, { base = 1000, cap = 30000, rng = Math.random } = {}) {
  return rng() * backoffCeiling(attempt, { base, cap });
}

/**
 * Create a reconnect scheduler. Returns { start, schedule, reset, stop,
 * attempts }.
 */
export function createReconnector({
  connect,
  base = 1000,
  cap = 30000,
  name = 'ws',
  onError = null,
  rng = Math.random,
} = {}) {
  if (typeof connect !== 'function') {
    throw new TypeError('createReconnector requires a connect() function');
  }

  let attempt = 0;        // failed-attempt counter; reset to 0 on open
  let timer = null;       // pending retry timer (single in-flight)
  let stopped = false;

  async function _run() {
    timer = null;
    if (stopped) return;
    let ws = null;
    try {
      ws = await connect();
    } catch (e) {
      if (onError) { try { onError(e); } catch (_) { /* ignore */ } }
      ws = null;
    }
    if (stopped) {
      // Detached while the connect was in flight — don't leak the socket.
      if (ws && typeof ws.close === 'function') {
        try { ws.close(1000, 'detached'); } catch (_) { /* ignore */ }
      }
      return;
    }
    if (ws && typeof ws.addEventListener === 'function') {
      // Backoff resets only once the socket actually reaches OPEN, not
      // merely because connect() returned — a socket can construct and
      // then immediately error.
      ws.addEventListener('open', () => { if (!stopped) attempt = 0; }, { once: true });
    } else if (!ws) {
      // connect() failed (null/throw) without ever giving us a socket whose
      // close event would drive the next retry — so back off and try again.
      schedule();
    }
  }

  function schedule() {
    if (stopped || timer !== null) return;  // single in-flight
    const delay = fullJitterDelay(attempt, { base, cap, rng });
    attempt += 1;
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[ws-reconnect:${name}] retry #${attempt} in ${Math.round(delay)}ms`);
    }
    timer = setTimeout(_run, delay);
  }

  function start() {
    if (stopped) return;
    attempt = 0;
    if (timer !== null) { clearTimeout(timer); timer = null; }
    _run();
  }

  function reset() { attempt = 0; }

  function stop() {
    stopped = true;
    if (timer !== null) { clearTimeout(timer); timer = null; }
  }

  return {
    start,
    schedule,
    reset,
    stop,
    get attempts() { return attempt; },
  };
}
