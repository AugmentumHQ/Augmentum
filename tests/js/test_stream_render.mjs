/**
 * Node regression for the streaming-render core logic (no CI harness exists
 * for JS yet — run by hand:  node tests/js/test_stream_render.mjs).
 *
 * Covers the two pure functions that make the incremental renderer correct
 * and O(delta) instead of O(n²): the stable-boundary promotion and the
 * open-fence detector. They live in stream-fence.js precisely so this test
 * needs no DOM. The DOM wiring (renderStreamSplit / makeStreamRenderer) is
 * verified in-browser after a hard refresh.
 */

const url = new URL('../../ui/scripts/chat/stream-fence.js', import.meta.url);
const { findStableBoundary, findOpenFence, findForcedBoundary, looksAllProse } = await import(url);

let pass = 0, fail = 0;
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { pass++; }
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${e}\n   got      ${a}`); }
}

// ── findStableBoundary ────────────────────────────────────────────────────
// Promotes to the last \n\n where fences are balanced.
eq(findStableBoundary('hello world', 0), 0, 'no paragraph break → no promotion');
eq(findStableBoundary('para one\n\npara two', 0), 10, 'single \\n\\n promotes to after it');
eq(findStableBoundary('a\n\nb\n\nc', 0), 6, 'advances to the LAST balanced break');
// An OPEN fence blocks promotion past it (the bug we fixed lived here).
eq(findStableBoundary('intro\n\n```py\ncode here\n\nstill in block', 0), 7,
   'open fence blocks promotion past the fence');
// A CLOSED fence is balanced → promotion can cross it (boundary = after \n\n).
eq(findStableBoundary('```py\nx=1\n```\n\nafter', 0), 15,
   'closed fence is balanced → promote past it');
// Respects the starting boundary (incremental calls).
eq(findStableBoundary('a\n\nb\n\nc', 3), 6, 'starts scanning from currentBoundary');

// ── findOpenFence ─────────────────────────────────────────────────────────
eq(findOpenFence('just prose, no fences'), null, 'no fence → null');
eq(findOpenFence('```py\nx=1\n```'), null, 'balanced fence → null');
{
  const r = findOpenFence('intro\n```python\nprint(1)');
  eq(!!r, true, 'open fence detected');
  eq(r.lang, 'python', 'lang parsed from opener');
  eq(r.openIdx, 6, 'openIdx at the ``` line');
  // body begins after the "```python\n" line
  eq('intro\n```python\nprint(1)'.slice(r.bodyStart), 'print(1)', 'bodyStart after opener newline');
}
{
  // Opener line still streaming (no newline yet) → no body to append.
  const r = findOpenFence('text\n```pyth');
  eq(!!r, true, 'partial opener still counts as open');
  eq('text\n```pyth'.slice(r.bodyStart), '', 'bodyStart == end while opener line incomplete');
}
{
  // A closed block followed by a freshly-opened one (each opener line-start):
  // last unmatched opener wins.
  const t = 'one\n```js\na\n```\nthen\n```py\nb';
  const r = findOpenFence(t);
  eq(!!r, true, 'second (open) fence detected after a closed one');
  eq(r.lang, 'py', 'lang from the last unmatched opener');
}
{
  // Inline ```py mid-line is NOT a fence (must be line-start).
  eq(findOpenFence('see ```py here'), null, 'mid-line backticks are not a fence opener');
}
{
  // Lang with trailing spaces / info string.
  const r = findOpenFence('```js  \ncode');
  eq(r.lang, 'js', 'lang trims trailing whitespace');
}

// ── looksAllProse ─────────────────────────────────────────────────────────
// The safety gate: any markdown BLOCK structure disables the active-tail cap.
eq(looksAllProse('just a wall of plain prose with words and words'), true, 'flat prose → true');
eq(looksAllProse('line one\nline two\nline three'), true, 'soft-wrapped prose → true');
eq(looksAllProse('intro\n- a list item\nmore'), false, 'unordered list → false');
eq(looksAllProse('intro\n1. an ordered item\nmore'), false, 'ordered list → false');
eq(looksAllProse('intro\n> a quote'), false, 'blockquote → false');
eq(looksAllProse('# heading\ntext'), false, 'ATX heading → false');
eq(looksAllProse('col a | col b\n--- | ---'), false, 'table (pipes) → false');
eq(looksAllProse('text\n```\ncode'), false, 'fence → false');
eq(looksAllProse('text\n    indented code'), false, 'indented code → false');
eq(looksAllProse('text\n***\nmore'), false, 'thematic break → false');

// ── findForcedBoundary ────────────────────────────────────────────────────
// Below the cap → never promote.
eq(findForcedBoundary('short prose', 6000), -1, 'under softCap → no forced promotion');
eq(findForcedBoundary('x'.repeat(5999), 6000), -1, 'just under softCap → none');
// Structured content over the cap → still refuse (never corrupt a list/table).
{
  const bigList = ('- item number ' + 'x'.repeat(40) + '\n').repeat(400); // >6k, all list
  eq(findForcedBoundary(bigList, 6000), -1, 'huge list → refuse (no corruption)');
  const bigTable = ('a | b | c\n').repeat(1000);
  eq(findForcedBoundary(bigTable, 6000), -1, 'huge table → refuse (no corruption)');
}
// All-prose over the cap → promote at a line boundary at/under target.
{
  // 200 lines of ~50 chars = ~10k chars, plain prose.
  const line = 'the quick brown fox jumped over the lazy dog again. ';
  const active = Array.from({ length: 200 }, () => line).join('\n');
  const cut = findForcedBoundary(active, 6000);
  eq(cut > 0, true, 'all-prose over cap → a forced boundary is returned');
  eq(active[cut - 1], '\n', 'promotes THROUGH a newline (line boundary)');
  const target = active.length - 3000;
  eq(cut <= target + 1, true, 'boundary is at/under the target (keeps a tail)');
  eq(active.length - cut >= 2999, true, 'leaves ~softCap/2 streaming tail');
}
// Single unbroken giant paragraph (no newline) → sentence-boundary fallback.
{
  const sentence = 'This is a complete sentence about nothing in particular here. ';
  const active = sentence.repeat(200); // ~12k chars, no newlines
  eq(active.includes('\n'), false, 'precondition: no newlines');
  const cut = findForcedBoundary(active, 6000);
  eq(cut > 0, true, 'giant single paragraph → sentence fallback promotes');
  eq(active[cut - 1], ' ', 'sentence cut lands just past the terminator space');
  eq('.!?'.includes(active[cut - 2]), true, 'cut follows a sentence terminator');
}
// Convergence: repeatedly promoting bounds the active region to < softCap.
{
  let active = ('word '.repeat(20) + '\n').repeat(500); // ~50k of plain prose
  let guard = 0;
  while (active.length > 6000 && guard < 1000) {
    const cut = findForcedBoundary(active, 6000);
    if (cut <= 0) break;
    active = active.slice(cut);
    guard++;
  }
  eq(active.length <= 6000, true, 'iterated promotion converges the tail under softCap');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
