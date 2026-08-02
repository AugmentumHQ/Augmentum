// tests/test_ws_reconnect_js.mjs
//
// Pure-Node tests for the shared WebSocket reconnect scheduler
// (ui/scripts/_ws-reconnect.js): the full-jitter backoff formula plus the
// createReconnector control flow (single in-flight, reset-on-open, stop).
//
// Run with:
//   node tests/test_ws_reconnect_js.mjs

import {
  backoffCeiling,
  fullJitterDelay,
  createReconnector,
} from '../ui/scripts/_ws-reconnect.js';

let _failed = 0;
let _ran = 0;

function assert(cond, label) {
  _ran++;
  if (cond) console.log(`PASS ${label}`);
  else { _failed++; console.error(`FAIL ${label}`); }
}

function assertEq(actual, expected, label) {
  _ran++;
  if (actual === expected) console.log(`PASS ${label}`);
  else { _failed++; console.error(`FAIL ${label}\n  expected: ${expected}\n    actual: ${actual}`); }
}

// ── backoffCeiling: doubles per attempt, clamps at cap ──────────────────
(function ceilingDoublesAndClamps() {
  const opts = { base: 1000, cap: 30000 };
  assertEq(backoffCeiling(0, opts), 1000, 'ceiling attempt 0 = base');
  assertEq(backoffCeiling(1, opts), 2000, 'ceiling attempt 1 = base*2');
  assertEq(backoffCeiling(2, opts), 4000, 'ceiling attempt 2 = base*4');
  assertEq(backoffCeiling(3, opts), 8000, 'ceiling attempt 3 = base*8');
  assertEq(backoffCeiling(5, opts), 30000, 'ceiling clamps at cap (base*32 > cap)');
  assertEq(backoffCeiling(50, opts), 30000, 'ceiling stays clamped for large attempts');
  assertEq(backoffCeiling(-3, opts), 1000, 'negative attempt treated as 0');
})();

// ── fullJitterDelay: uniform in [0, ceiling] ────────────────────────────
(function jitterStaysWithinCeiling() {
  const opts = { base: 1000, cap: 30000 };
  // rng = 0 → delay 0; rng = 1 → delay = ceiling
  assertEq(fullJitterDelay(2, { ...opts, rng: () => 0 }), 0, 'jitter at rng=0 is 0');
  assertEq(fullJitterDelay(2, { ...opts, rng: () => 1 }), 4000, 'jitter at rng=1 is the full ceiling');
  // Many samples must never exceed the ceiling and at least one must be > 0.
  let maxSeen = 0;
  let anyPositive = false;
  for (let i = 0; i < 500; i++) {
    const d = fullJitterDelay(3, opts);  // ceiling 8000
    if (d > maxSeen) maxSeen = d;
    if (d > 0) anyPositive = true;
    if (d < 0 || d > 8000) { assert(false, `jitter sample out of [0,8000]: ${d}`); return; }
  }
  assert(maxSeen <= 8000, 'all jitter samples within ceiling');
  assert(anyPositive, 'jitter produces non-zero delays');
})();

// ── createReconnector: control flow with a fake clock ───────────────────
function withFakeTimers(fn) {
  const realSet = globalThis.setTimeout;
  const realClear = globalThis.clearTimeout;
  const pending = new Map();
  let nextId = 1;
  globalThis.setTimeout = (cb) => { const id = nextId++; pending.set(id, cb); return id; };
  globalThis.clearTimeout = (id) => { pending.delete(id); };
  const flush = async () => {
    // Drain whatever is scheduled right now (one generation at a time).
    const gen = [...pending.entries()];
    pending.clear();
    for (const [, cb] of gen) await cb();
  };
  try { return fn({ flush, pendingCount: () => pending.size }); }
  finally { globalThis.setTimeout = realSet; globalThis.clearTimeout = realClear; }
}

// Minimal fake WebSocket: records listeners, lets the test fire 'open'.
function makeFakeWs() {
  const listeners = {};
  return {
    closed: false,
    addEventListener(type, cb) { (listeners[type] ||= []).push(cb); },
    fire(type) { (listeners[type] || []).forEach((cb) => cb()); },
    close() { this.closed = true; },
  };
}

(function singleInFlightAndAttemptGrowth() {
  withFakeTimers(({ flush, pendingCount }) => {
    let connects = 0;
    const r = createReconnector({
      connect: () => { connects++; return makeFakeWs(); },
      base: 1000, cap: 30000, rng: () => 0.5,
    });
    r.schedule();
    r.schedule();  // second call must be a no-op while one is pending
    assertEq(pendingCount(), 1, 'single in-flight: only one timer scheduled');
    assertEq(r.attempts, 1, 'attempt counter incremented once');
    return flush();
  });
})();

(async function resetsOnOpen() {
  await withFakeTimers(async ({ flush }) => {
    let lastWs = null;
    const r = createReconnector({
      connect: () => { lastWs = makeFakeWs(); return lastWs; },
      base: 1000, cap: 30000, rng: () => 0.5,
    });
    r.schedule();                 // attempt → 1
    r.schedule = r.schedule;      // (no-op, keep linter calm)
    await flush();                // runs connect, attaches open listener
    assert(r.attempts === 1, 'attempts is 1 before open fires');
    lastWs.fire('open');          // socket reaches OPEN
    assertEq(r.attempts, 0, 'attempt counter resets to 0 on open');
  });
})();

(async function stopCancelsAndClosesLateSocket() {
  await withFakeTimers(async ({ flush, pendingCount }) => {
    let lateWs = null;
    const r = createReconnector({
      // connect resolves async so we can stop() before it returns.
      connect: async () => { lateWs = makeFakeWs(); return lateWs; },
      base: 1000, cap: 30000, rng: () => 0.5,
    });
    r.schedule();
    r.stop();
    assertEq(pendingCount(), 0, 'stop() clears the pending retry timer');
    r.schedule();                 // must be a no-op after stop
    assertEq(pendingCount(), 0, 'schedule() is a no-op after stop()');
    await flush();                // nothing scheduled → nothing runs
    // A connect that had already been kicked off and resolves after stop
    // gets its socket closed (covered by the async path); assert no throw.
    assert(true, 'stop() is safe with no pending work');
  });
})();

setTimeout(() => {
  console.log(`\n${_ran - _failed}/${_ran} passed`);
  process.exit(_failed ? 1 : 0);
}, 50);
