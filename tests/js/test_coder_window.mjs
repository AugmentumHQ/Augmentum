/**
 * Node regression for the coder conversation DOM-windowing math (run by hand:
 *   node tests/js/test_coder_window.mjs).
 *
 * Covers the two pure decisions that keep a long coder session's live DOM
 * bounded — how many nodes to detach when over the cap, and how many to
 * re-attach on scroll-up. They live in coder-window.js precisely so this test
 * needs no DOM; the DOM wiring (_trimWindowIfNeeded / _rehydrateOlder in
 * coder-conversation.js) is verified in-browser.
 */

const url = new URL('../../ui/scripts/coder-window.js', import.meta.url);
const { nodesToDetach, nodesToRehydrate } = await import(url);

let pass = 0, fail = 0;
function eq(actual, expected, msg) {
  if (actual === expected) { pass++; }
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${expected}\n   got      ${actual}`); }
}

// ── nodesToDetach ─────────────────────────────────────────────────────────
const D = (liveCount, scrolledUp = false) =>
  nodesToDetach({ liveCount, maxLive: 80, batch: 20, scrolledUp });

eq(D(50), 0, 'under the cap → detach nothing');
eq(D(80), 0, 'exactly at the cap → detach nothing');
eq(D(95), 0, 'over the cap but under the batch hysteresis (excess 15 < 20) → wait');
eq(D(99), 0, 'just under hysteresis (excess 19) → still wait');
eq(D(100), 20, 'hits hysteresis (excess 20) → trim all the way back to the cap');
eq(D(130), 50, 'well over → trim the full excess in one pass (back to 80)');
// Never trim under a scrolled-up viewport, no matter how far over.
eq(D(500, true), 0, 'scrolled up → never detach (would shift what they read)');
eq(D(100, true), 0, 'scrolled up at hysteresis → still nothing');

// Trimming converges: after one trim of the excess we are exactly at the cap.
{
  let live = 130;
  const n = D(live);
  live -= n;
  eq(live, 80, 'after a trim pass the live count equals the cap');
  eq(D(live), 0, 'a freshly-trimmed window needs no further trim');
}

// Boundary on a different cap/batch.
eq(nodesToDetach({ liveCount: 12, maxLive: 10, batch: 5, scrolledUp: false }), 0,
   'excess 2 < batch 5 → wait');
eq(nodesToDetach({ liveCount: 15, maxLive: 10, batch: 5, scrolledUp: false }), 5,
   'excess 5 == batch 5 → trim to cap');

// ── nodesToRehydrate ──────────────────────────────────────────────────────
const R = (detachedCount) => nodesToRehydrate({ detachedCount, batch: 30 });
eq(R(0), 0, 'nothing detached → re-attach nothing');
eq(R(5), 5, 'fewer detached than a batch → re-attach all of them');
eq(R(30), 30, 'exactly a batch → re-attach the batch');
eq(R(200), 30, 'many detached → re-attach at most one batch');
eq(nodesToRehydrate({ detachedCount: -3, batch: 30 }), 0, 'defensive: negative → 0');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
