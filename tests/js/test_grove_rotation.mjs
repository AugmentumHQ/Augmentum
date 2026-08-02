/**
 * Node regression for grove-rotation.js — the recently-played rotation that
 * stops repeat genre asks ("jazz" again and again) from replaying one video.
 *
 * Run by hand:  node tests/js/test_grove_rotation.mjs
 */

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) pass++; else { fail++; console.error(`FAIL: ${msg}`); } }

const url = new URL('../../ui/scripts/grove-rotation.js', import.meta.url);
const { createRotation } = await import(url);

const id = (x) => x;

// ── empty / basic note+isRecent ─────────────────────────────────────────────
{
  const r = createRotation(6);
  ok(r.preferFresh([], id) === null, 'empty list → null');
  ok(!r.isRecent('a'), 'nothing recent initially');
  r.note('a');
  ok(r.isRecent('a'), 'note() marks recent');
  r.note('');
  ok(r.snapshot().length === 1, 'note("") is a no-op');
}

// ── preferFresh excludes recent when a fresh option exists ──────────────────
{
  const r = createRotation(6);
  const items = ['v1', 'v2', 'v3', 'v4'];
  r.note('v1'); r.note('v2'); r.note('v3');
  // Only v4 is fresh among the top 4 → must return it deterministically.
  ok(r.preferFresh(items, id) === 'v4', 'returns the only fresh item');
}

// ── all-recent → falls back to the top slice (never null) ───────────────────
{
  const r = createRotation(6);
  const items = ['v1', 'v2', 'v3', 'v4'];
  items.forEach((x) => r.note(x));
  const pick = r.preferFresh(items, id);
  ok(items.includes(pick), 'all-recent falls back to a top item (not null)');
}

// ── THE property: repeat asks rotate — first N picks over an N-list are all
//    distinct (each pick is noted, excluding it next time). ──────────────────
{
  const r = createRotation(6);
  const items = ['j1', 'j2', 'j3', 'j4'];
  const picks = [];
  for (let i = 0; i < 4; i++) {
    const p = r.preferFresh(items, id);
    r.note(p);
    picks.push(p);
  }
  ok(new Set(picks).size === 4, `4 repeat asks cycle through all 4 (got ${JSON.stringify(picks)})`);
}

// ── topN bias: a fresh item OUTSIDE the top slice is not reachable ───────────
{
  const r = createRotation(6);
  const items = ['a', 'b', 'c', 'd', 'e'];   // 'e' is rank 5, outside top 4
  r.note('a'); r.note('b'); r.note('c'); r.note('d');
  // Top-4 are all recent → fall back to top-4 (never reaches 'e').
  const pick = r.preferFresh(items, id, 4);
  ok(['a', 'b', 'c', 'd'].includes(pick) && pick !== 'e', 'topN bound respected');
  ok(r.preferFresh(items, id, 5) === 'e', 'widening topN reaches the fresh 5th');
}

// ── cap: only the last `cap` ids are retained ───────────────────────────────
{
  const r = createRotation(3);
  ['1', '2', '3', '4', '5'].forEach((x) => r.note(x));
  ok(r.snapshot().join(',') === '3,4,5', 'cap keeps newest 3');
  ok(!r.isRecent('1') && r.isRecent('5'), 'evicted ids fall out of the window');
}

// ── re-noting an id moves it to newest (no duplicates) ──────────────────────
{
  const r = createRotation(6);
  r.note('x'); r.note('y'); r.note('x');
  ok(r.snapshot().join(',') === 'y,x', 're-note dedups and refreshes recency');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
