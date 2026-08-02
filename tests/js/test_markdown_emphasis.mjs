/**
 * Node regression for the inline-emphasis pass (chat/emphasis.js).
 * Run by hand: node tests/js/test_markdown_emphasis.mjs
 * (Same convention as test_stream_render.mjs — no CI harness for JS yet.)
 *
 * Two jobs:
 *   1. Correctness parity with the regexes this module replaced — the
 *      guards that keep math / snake_case / list markers literal, and the
 *      narrative `*action*` styling that RP mode depends on.
 *   2. The performance contract that motivated the extraction: the old
 *      `[\s\S]+?` bodies were quadratic on unmatched delimiters (measured
 *      2.3s on 500KB of code-ish prose). The bounded bodies must stay
 *      linear — the perf gate below fails the run if 500KB takes >250ms.
 */

const url = new URL('../../ui/scripts/chat/emphasis.js', import.meta.url);
const { applyInlineEmphasis } = await import(url);

let pass = 0, fail = 0;
function eq(actual, expected, msg) {
  if (actual === expected) { pass++; }
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${JSON.stringify(expected)}\n   got      ${JSON.stringify(actual)}`); }
}

// ── basic forms ───────────────────────────────────────────────────────────
eq(applyInlineEmphasis('**bold**'), '<strong>bold</strong>', 'asterisk bold');
eq(applyInlineEmphasis('__bold__'), '<strong>bold</strong>', 'underscore bold');
eq(applyInlineEmphasis('*ital*'), '<em>ital</em>', 'asterisk italic');
eq(applyInlineEmphasis('_ital_'), '<em>ital</em>', 'underscore italic');
eq(applyInlineEmphasis('**bold** and *ital*'),
   '<strong>bold</strong> and <em>ital</em>', 'mixed forms in one line');

// ── guards that must stay literal ─────────────────────────────────────────
eq(applyInlineEmphasis('2 * 3 = 6 and 4 * 5 = 20'),
   '2 * 3 = 6 and 4 * 5 = 20', 'spaced math stars never emphasize');
eq(applyInlineEmphasis('my__var__name'), 'my__var__name',
   'mid-word double underscore stays literal');
eq(applyInlineEmphasis('use snake_case_names here'),
   'use snake_case_names here', 'snake_case stays literal');
eq(applyInlineEmphasis('* hello *'), '* hello *',
   'space-hugging delimiters stay literal');

// ── shapes carried over from the old regexes ──────────────────────────────
eq(applyInlineEmphasis('**a * b**'), '<strong>a * b</strong>',
   'lone star allowed inside bold');
eq(applyInlineEmphasis('__has_snake_case__'),
   '<strong>has_snake_case</strong>', 'mid-word underscore inside bold-underscore');
eq(applyInlineEmphasis('_foo_bar_'), '<em>foo_bar</em>',
   'snake_case inside underscore italic (closer = final _)');
// Changed from the old regexes ON PURPOSE: glued-star runs stay literal.
// Allowing `*` inside italic bodies is what let failed matches scan
// through star-dense soup (the quadratic case). `*a*b*` as literal text
// is the safer read; proper `*a b*` spans are unaffected.
eq(applyInlineEmphasis('*a*b*'), '*a*b*',
   'glued-star run stays literal (perf-motivated, documented change)');

// ── narrative styling ─────────────────────────────────────────────────────
eq(applyInlineEmphasis('*she walks\nslowly*'),
   '<em>she walks\nslowly</em>', 'action span crosses a soft line break');
eq(applyInlineEmphasis('*orphan\n\nnew para*'), '*orphan\n\nnew para*',
   'emphasis never crosses a blank line (paragraph pass would split the tags)');
eq(applyInlineEmphasis('**bold\n\nacross**'), '**bold\n\nacross**',
   'bold never crosses a blank line either');

// ── unmatched-delimiter content stays untouched ──────────────────────────
eq(applyInlineEmphasis('SELECT * FROM t WHERE x LIKE "%*"'),
   'SELECT * FROM t WHERE x LIKE "%*"', 'SQL wildcard soup unchanged');
{
  const log = 'result = a * b; // fails *sometimes\n';
  eq(applyInlineEmphasis(log), log, 'code-ish log line unchanged');
}

// ── perf gate: linear on adversarial content ─────────────────────────────
{
  const line = 'result = a * b * c; // 5 * 3 fails *sometimes\n';
  const big = line.repeat(Math.ceil(500 * 1024 / line.length)); // ~500KB
  const t0 = performance.now();
  applyInlineEmphasis(big);
  const ms = performance.now() - t0;
  if (ms > 250) { fail++; console.error(`FAIL: 500KB code-ish prose took ${ms.toFixed(0)}ms (budget 250ms — quadratic regression?)`); }
  else { pass++; console.log(`perf: 500KB code-ish prose in ${ms.toFixed(1)}ms`); }
}
{
  // Single giant paragraph (no blank lines at all) — the worst case for
  // the blank-line bound; the {1,1000} char cap must hold it linear.
  const big = ('x *y '.repeat(200) + '\n').repeat(500); // ~500KB, one \n per line
  const t0 = performance.now();
  applyInlineEmphasis(big);
  const ms = performance.now() - t0;
  if (ms > 400) { fail++; console.error(`FAIL: 500KB single-paragraph soup took ${ms.toFixed(0)}ms (budget 400ms)`); }
  else { pass++; console.log(`perf: 500KB unbroken-paragraph soup in ${ms.toFixed(1)}ms`); }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
