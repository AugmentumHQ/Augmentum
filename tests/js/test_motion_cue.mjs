/**
 * Node regression for motion-cue.js — extracting/stripping the model's inline
 * [motion:xxx] avatar tag so it never reaches the rendered/saved chat text.
 *
 * Run by hand:  node tests/js/test_motion_cue.mjs
 */

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) pass++; else { fail++; console.error(`FAIL: ${msg}`); } }

const url = new URL('../../ui/scripts/motion-cue.js', import.meta.url);
const { readMotionCue, stripMotionCue, stripMotionCueStreaming, extractMotionCue } = await import(url);

// ── read ────────────────────────────────────────────────────────────────────
ok(readMotionCue('Hi there! [motion:happy]') === 'happy', 'reads a trailing cue');
ok(readMotionCue('text') === null, 'no tag → null');
ok(readMotionCue('[MOTION:Bow]') === 'bow', 'case-insensitive + lowercased');
ok(readMotionCue('') === null, 'empty → null');

// ── strip (save/final) ───────────────────────────────────────────────────────
ok(stripMotionCue('Hello [motion:wave]') === 'Hello', 'strips the tag (+ its space)');
ok(stripMotionCue('no tag here') === 'no tag here', 'leaves clean text alone');
ok(!/motion/.test(stripMotionCue('Line one.\n[motion:happy]')), 'strips an own-line tag');
ok(stripMotionCue('A [motion:sad] B') === 'A B', 'strips an inline tag, keeps prose');

// ── streaming strip (no flash of a half-typed tag) ───────────────────────────
ok(stripMotionCueStreaming('Hi [motion:hap') === 'Hi ', 'strips a partial trailing tag');
ok(stripMotionCueStreaming('Hi [motion:happy] done') === 'Hi  done', 'strips a complete tag mid-stream');
ok(stripMotionCueStreaming('cost is [5') === 'cost is [5', 'leaves a non-motion bracket alone');
ok(stripMotionCueStreaming('Hi [mot') === 'Hi ', 'strips an early-partial tag');

// ── combined extract ─────────────────────────────────────────────────────────
{
  const r = extractMotionCue('That made my day! [motion:laugh]');
  ok(r.cue === 'laugh' && r.text === 'That made my day!', 'extract returns {cue, clean text}');
}
{
  const r = extractMotionCue('just words');
  ok(r.cue === null && r.text === 'just words', 'extract with no tag');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
