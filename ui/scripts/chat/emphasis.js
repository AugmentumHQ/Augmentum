/**
 * emphasis.js — inline emphasis pass for the markdown renderer.
 *
 * Extracted from markdown.js (same pattern as stream-fence.js: pure
 * string logic in a DOM-free module so tests/js/test_markdown_emphasis.mjs
 * can exercise the REAL code in node).
 *
 * Why this exists as its own module — the previous in-place regexes used
 * `[\s\S]+?` inner bodies, which are quadratic on content with unmatched
 * delimiters: every stray `*` (ubiquitous in logs, shell output, and
 * un-fenced code) made the engine lazily expand to the END OF THE STRING
 * before giving up. Measured on the real patterns: 100KB of code-ish
 * prose = ~90ms, 500KB = ~2.3s — one synchronous main-thread block per
 * render. These bodies bound the scan instead:
 *
 *   - No crossing blank lines (CommonMark forbids emphasis containing a
 *     blank line anyway, and the paragraph pass would split the tag pair
 *     into broken HTML). A failed match now stops at the paragraph edge.
 *   - No unrestricted delimiter chars inside the body — only the
 *     specific "internal" shapes we actually want (`*` inside `**bold**`,
 *     snake_case `_` inside underscore emphasis), so a failed match stops
 *     at the next delimiter instead of scanning past it.
 *   - A hard 1000-char body cap as the backstop for giant single-
 *     paragraph blobs. No real emphasis run is anywhere near that long.
 *
 * Behavior preserved from the originals:
 *   - `**X**` / `__X__` bold, `*X*` / `_X_` italic, `~~X~~` strike.
 *   - `2 * 3` math and `* ` list markers never emphasize (whitespace
 *     guards on both ends).
 *   - `my__var__name` / `snake_case` stay literal (word-boundary guards).
 *   - Narrative action spans survive a soft line break: `*she walks\nslowly*`.
 *   - Bold may contain a lone `*`; underscore forms may contain
 *     mid-word `_`.
 */

// Inner-body building blocks. Each alternative consumes exactly one char,
// so the {1,1000} cap is a character cap.
//   [^*\n]      — anything but a delimiter or newline
//   \n(?!\n)    — a SOFT line break only (never a blank line)
//   \*(?!\*)    — a lone star inside bold (not a `**` closer)
//   _(?!_) / _(?=\w) — mid-word underscores for the underscore forms
//
// The italic-asterisk body deliberately excludes `*` entirely: a failed
// match then stops at the very next star, which is what keeps star-dense
// soup (SQL wildcards, C pointers, multiplication chains) linear. The
// cost is that `*a*b*` no longer italicizes as `a*b` — glued-star runs
// stay literal, which is the safer read of that input anyway.
const _BOLD_AST = /\*\*(?!\s)((?:[^*\n]|\n(?!\n)|\*(?!\*)){1,1000}?)(?<!\s)\*\*/g;
const _BOLD_UND = /(^|[^\w])__(?!\s)((?:[^_\n]|\n(?!\n)|_(?!_)){1,1000}?)(?<!\s)__(?!\w)/g;
const _ITAL_AST = /(?<![*\w])\*(?!\s|\*)((?:[^*\n]|\n(?!\n)){1,1000}?)(?<!\s)\*(?![*\w])/g;
const _ITAL_UND = /(^|[^\w])_(?!\s)((?:[^_\n]|\n(?!\n)|_(?=\w)){1,1000}?)(?<!\s)_(?!\w)/g;

/**
 * Apply bold / italic to an (already HTML-escaped) markdown string.
 * Order matters: bold first so `**` runs are consumed before the italic
 * pass sees them. Strikethrough deliberately stays in markdown.js — it
 * must run AFTER the narrative dialogue pass (a `<del>` inserted first
 * would break the dialogue regex's `[^&<>]` body on quoted text that
 * contains `~~strike~~`).
 *
 * @param {string} html
 * @returns {string}
 */
export function applyInlineEmphasis(html) {
  html = html.replace(_BOLD_AST, '<strong>$1</strong>');
  html = html.replace(_BOLD_UND, '$1<strong>$2</strong>');
  html = html.replace(_ITAL_AST, '<em>$1</em>');
  html = html.replace(_ITAL_UND, '$1<em>$2</em>');
  return html;
}
