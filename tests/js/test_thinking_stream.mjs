/**
 * Node regression for the reasoning/thinking streaming decision logic (no CI
 * harness for JS yet — run by hand:  node tests/js/test_thinking_stream.mjs).
 *
 * Locks in the O(delta) invariant: while reasoning streams in, every frame
 * must APPEND only the new tail (never rewrite the whole accumulated string),
 * which is what keeps long CoT off the O(n²) freeze path. The DOM wiring in
 * renderer.js (_flushModelThinking) is verified in-browser after a refresh.
 */

const url = new URL('../../ui/scripts/chat/thinking-stream.js', import.meta.url);
const { planThinkingAppend } = await import(url);

let pass = 0, fail = 0;
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { pass++; }
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${e}\n   got      ${a}`); }
}

// ── steady state: append only the tail ────────────────────────────────────
eq(planThinkingAppend(0, 5), { action: 'append', from: 0 }, 'first content → append from 0');
eq(planThinkingAppend(5, 9), { action: 'append', from: 5 }, 'grew by 4 → append from prior length');
eq(planThinkingAppend(9, 9), { action: 'noop', from: 9 }, 'no growth → noop (e.g. coalesced empty frame)');

// ── desync: resync wholesale exactly once ─────────────────────────────────
eq(planThinkingAppend(-1, 12), { action: 'resync', from: 0 }, 'unknown cursor (foreign block) → resync');
eq(planThinkingAppend(20, 12), { action: 'resync', from: 0 }, 'cursor > full (accumulator reset) → resync');
eq(planThinkingAppend(NaN, 12), { action: 'resync', from: 0 }, 'non-finite cursor → resync');

// ── the core invariant: a full token-by-token stream is O(delta) ──────────
// Simulate reasoning arriving one char at a time and assert we NEVER rewrite
// the whole string (no 'append' from 0 after the first char, no 'resync').
{
  let rendered = 0;
  let appends = 0, rewrites = 0;
  const N = 2000;
  for (let len = 1; len <= N; len++) {
    const plan = planThinkingAppend(rendered, len);
    if (plan.action === 'resync') rewrites++;
    if (plan.action === 'append') {
      appends++;
      // tail length appended this step is exactly the delta (1 char), not N.
      if (len - plan.from !== 1) { fail++; console.error(`FAIL: step ${len} appended ${len - plan.from} chars, expected 1`); }
    }
    rendered = len; // DOM now holds `len` chars
  }
  eq(rewrites, 0, 'no wholesale rewrites across a full stream');
  eq(appends, N, 'every growth step was an O(delta) append');
}

// ── coalesced frame: multiple tokens land, one append covers them all ─────
{
  // Frame 1 rendered 3 chars; by frame 2 the accumulator jumped to 50.
  const plan = planThinkingAppend(3, 50);
  eq(plan, { action: 'append', from: 3 }, 'coalesced burst → single append of the whole tail');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
