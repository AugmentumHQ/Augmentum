/**
 * stream-fence.js — pure, dependency-free string logic for the incremental
 * streaming renderer. Kept separate from stream-render.js (which imports the
 * DOM/markdown chain) so this can be unit-tested under node without loading
 * the whole UI bundle. See tests/js/test_stream_render.mjs.
 */

/**
 * Advance the stable boundary: the last ``\n\n`` paragraph break at which all
 * ``` fences are balanced. Content before it is settled — safe to render once
 * and never touch again. An open (unbalanced) fence blocks promotion past it.
 *
 * @param {string} raw - full accumulated text
 * @param {number} currentBoundary - last known stable index
 * @returns {number} new stable boundary (>= currentBoundary)
 */
export function findStableBoundary(raw, currentBoundary) {
  let last = currentBoundary;
  let idx = raw.indexOf('\n\n', currentBoundary);
  while (idx !== -1) {
    const candidate = raw.slice(0, idx);
    const fences = candidate.match(/^```/gm);
    if (!fences || fences.length % 2 === 0) {
      last = idx + 2;
    }
    idx = raw.indexOf('\n\n', idx + 2);
  }
  return last;
}

/**
 * Locate the last UNMATCHED ``` opener in ``text`` (the active suffix, i.e.
 * ``raw.slice(stableEnd)``). Returns ``{ openIdx, lang, bodyStart }`` or null
 * when fences are balanced.
 *
 * ``bodyStart`` is where the code body begins — just after the opener line's
 * newline. While the opener line is still streaming (no newline yet)
 * ``bodyStart === text.length`` so there's nothing to append until it
 * completes.
 *
 * @param {string} text
 * @returns {{openIdx:number, lang:string, bodyStart:number}|null}
 */
export function findOpenFence(text) {
  const re = /^```([^\n]*)$/gm;
  const openers = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    openers.push({ index: m.index, info: m[1].trim(), lineEnd: m.index + m[0].length });
    if (re.lastIndex === m.index) re.lastIndex++; // guard against zero-width
  }
  if (openers.length % 2 === 0) return null; // balanced (incl. zero)
  const open = openers[openers.length - 1];
  const lang = (open.info.match(/^\S+/) || [''])[0];
  let bodyStart = open.lineEnd;
  if (text[bodyStart] === '\n') bodyStart += 1;
  return { openIdx: open.index, lang, bodyStart };
}

/**
 * Is ``text`` entirely plain paragraph prose — no markdown BLOCK structure?
 *
 * This is the safety gate for the active-tail cap. The cap force-promotes a
 * prefix of the streaming suffix into the settled region to bound per-frame
 * re-render cost. Doing that across a list/table/blockquote/heading boundary
 * would CORRUPT rendering (renumber an ordered list, split a table, escape a
 * lazy continuation out of its list item). So we only ever force-promote when
 * the whole active region is flat prose, where the worst that can happen is a
 * soft line-break becoming a paragraph break — cosmetic, never wrong.
 *
 * Conservative by design: any whiff of structure (even a stray ``|``) returns
 * false, which just disables the cap for that message — safe, never corrupting.
 */
export function looksAllProse(text) {
  return !(
    /^\s*(\d+[.)]|[-*+])\s/m.test(text) ||  // ordered / unordered list item
    /^\s*>/m.test(text) ||                   // blockquote
    /^\s*#{1,6}\s/m.test(text) ||            // ATX heading
    /^\s*`{3,}/m.test(text) ||               // ``` fence (a closed one can sit here)
    /^\s*~{3,}/m.test(text) ||               // ~~~ fence
    /\|/.test(text) ||                        // table pipe (anywhere)
    /^(\t| {4})/m.test(text) ||              // indented code / list continuation
    /^\s*([-*_])\s*\1\s*\1/m.test(text) ||   // thematic break (---, ***, ___)
    /^\s*=+\s*$/m.test(text) ||              // setext underline
    /^\s*</m.test(text)                       // raw HTML block start
  );
}

/** Last sentence terminator (``.!?`` followed by space/newline) at or before
 *  ``target``. Returns the index just past the whitespace (so the promoted
 *  chunk ends cleanly and the tail starts at the next sentence), or -1. */
function _lastSentenceEnd(text, target) {
  const start = Math.min(target, text.length - 1);
  for (let i = start; i > 0; i--) {
    const c = text[i];
    if ((c === ' ' || c === '\n')) {
      const p = text[i - 1];
      if (p === '.' || p === '!' || p === '?') return i + 1;
    }
  }
  return -1;
}

/**
 * Active-tail cap. When the active (no-open-fence) suffix grows past ``softCap``
 * with no ``\n\n`` to settle it, return a SAFE index to force-promote so the
 * per-frame wholesale re-render stays bounded (otherwise: O(n²) over a giant
 * paragraph at fast tok/s — the last unhardened corner of the streaming class).
 *
 * Safety: only fires on all-prose content (``looksAllProse``); there, EVERY
 * newline is a safe break (no list/table to corrupt). Prefers a line boundary
 * (least visible), falling back to a sentence end for a single unbroken wall of
 * text. Leaves a ``softCap/2`` streaming tail so it doesn't re-promote every
 * frame. Returns -1 when nothing should be promoted.
 *
 * @param {string} active - the active suffix (``raw.slice(stableEnd)``)
 * @param {number} [softCap=6000] - promote once the suffix exceeds this
 * @returns {number} index (exclusive) within ``active`` to promote up to, or -1
 */
export function findForcedBoundary(active, softCap = 6000) {
  if (active.length <= softCap) return -1;
  if (!looksAllProse(active)) return -1; // never split structured content
  const target = active.length - Math.floor(softCap / 2);
  const nl = active.lastIndexOf('\n', target);
  if (nl > 0) return nl + 1;                 // promote through the line break
  const s = _lastSentenceEnd(active, target);
  return s > 0 ? s : -1;                      // else split a giant single paragraph
}
