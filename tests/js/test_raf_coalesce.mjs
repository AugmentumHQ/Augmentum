/**
 * Node regression for the rAF coalescer (run by hand:
 *   node tests/js/test_raf_coalesce.mjs).
 *
 * Locks in the contract every streaming surface relies on: many calls within
 * one frame collapse to ONE invocation on the next frame, with the latest
 * args. A fake window.requestAnimationFrame drives the frame boundary.
 */

let pass = 0, fail = 0;
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { pass++; }
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${e}\n   got      ${a}`); }
}

// Install a manual rAF queue so we control frame boundaries.
const rafQueue = [];
globalThis.window = { requestAnimationFrame: (cb) => { rafQueue.push(cb); return rafQueue.length; } };
const tick = () => { const q = rafQueue.splice(0); q.forEach(cb => cb()); };

const url = new URL('../../ui/scripts/raf-coalesce.js', import.meta.url);
const { rafCoalesce } = await import(url);

// ── coalescing: N calls in a frame → 1 run, latest args ───────────────────
{
  const seen = [];
  const sched = rafCoalesce((x) => seen.push(x));
  sched(1); sched(2); sched(3);
  eq(seen, [], 'nothing runs until the frame boundary');
  tick();
  eq(seen, [3], 'one run on next frame, with the LATEST args');
  tick();
  eq(seen, [3], 'no extra run when nothing was scheduled');
}

// ── a fresh burst after a frame schedules again ───────────────────────────
{
  const seen = [];
  const sched = rafCoalesce((x) => seen.push(x));
  sched('a'); tick();
  sched('b'); sched('c'); tick();
  eq(seen, ['a', 'c'], 'each frame yields exactly one run with its latest args');
}

// ── stress: 1000 calls across 10 frames → at most 10 runs ─────────────────
{
  let runs = 0;
  const sched = rafCoalesce(() => { runs++; });
  for (let f = 0; f < 10; f++) {
    for (let i = 0; i < 100; i++) sched();
    tick();
  }
  eq(runs, 10, '1000 calls over 10 frames → 10 runs (not 1000)');
}

// ── cancel() drops a pending run ──────────────────────────────────────────
{
  const seen = [];
  const sched = rafCoalesce((x) => seen.push(x));
  sched(7); sched.cancel(); tick();
  eq(seen, [], 'cancel() drops the pending run');
}

// ── flush() runs a pending run immediately ────────────────────────────────
{
  const seen = [];
  const sched = rafCoalesce((x) => seen.push(x));
  sched(9); sched.flush();
  eq(seen, [9], 'flush() runs the pending work now');
  tick();
  eq(seen, [9], 'the frame run is a no-op after flush()');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
