/**
 * motion-cue.js — extract the chat model's inline avatar-motion tag.
 *
 * When her avatar is on screen, the model may end a reply with a single hidden
 * tag like [motion:happy] (see the becca_direct prompt directive). We strip it
 * from the rendered + saved text and map the cue word to animation roles so her
 * avatar reacts — one model call, no separate tool round-trip. The cue routes
 * through the conductor's role-based select(), so the user's ratings / disables /
 * uploaded clips govern what actually plays.
 *
 * Pure + dependency-free so it's unit-testable apart from the DOM.
 */

// The tag the model emits, e.g. [motion:happy]. Single square brackets + the
// `motion:` prefix won't render as a markdown link and is unlikely to collide
// with prose. Vocabulary lives in the backend directive + MOTION_CUE_INTENT.
const MOTION_CUE_RE = /\[motion:([a-z_]+)\]/i;
const MOTION_CUE_RE_G = /\[motion:[a-z_]+\]/gi;
// A half-streamed tag still arriving — any trailing prefix of "[motion:WORD",
// e.g. "[", "[mot", "[motion:", "[motion:ha" — so it never flashes. A trailing
// non-motion bracket like "[5" is left alone.
const MOTION_CUE_PARTIAL_RE = /\[(?:m(?:o(?:t(?:i(?:o(?:n(?::[a-z_]*)?)?)?)?)?)?)?$/i;

/** First cue word in the text (lowercased), or null. */
export function readMotionCue(text) {
  if (!text) return null;
  const m = MOTION_CUE_RE.exec(text);
  return m ? m[1].toLowerCase() : null;
}

/** Text with all complete motion tags removed (for save + final render). */
export function stripMotionCue(text) {
  if (!text) return text || '';
  return text
    .replace(MOTION_CUE_RE_G, '')   // remove the tag itself
    .replace(/[ \t]{2,}/g, ' ')     // collapse a double-space left inline (A  B → A B)
    .replace(/[ \t]+\n/g, '\n')     // drop trailing spaces before a newline
    .replace(/\n{3,}/g, '\n\n')     // collapse extra blank lines
    .trim();
}

/** Streaming-safe strip: complete tags + any half-typed trailing tag. */
export function stripMotionCueStreaming(text) {
  if (!text) return text || '';
  return text.replace(MOTION_CUE_RE_G, '').replace(MOTION_CUE_PARTIAL_RE, '');
}

/** {cue, text} — read the cue and return the cleaned text in one pass. */
export function extractMotionCue(text) {
  return { cue: readMotionCue(text), text: stripMotionCue(text) };
}
