// Textarea auto-resize, deferred + rAF-batched.
//
// The naive pattern (`ta.style.height='auto'` → read `scrollHeight` →
// write computed height) forces synchronous layout on every keystroke
// and shows up in INP traces as 32-150ms per character with streaming
// content active. This helper batches the read/write into the next
// animation frame via rAF, so multiple input events in the same
// frame coalesce into one layout flush — the load-bearing perf win.
//
// Replaces inline `autoResize(textarea)` definitions in:
//   - chat/input.js (was _autoResize() method)
//   - app.js (was autoResize() top-level)
//   - narrative/index.js (was autoResize() top-level)
//   - coder.js (was an inline rAF wrapper)
//
// One source of truth; future textareas just import this.
//
// 2026-05-28: removed the height-cache "fast path" that previously
// short-circuited when scrollHeight matched the last-written height.
// The fast path was a real bug:
//
// When a textarea has an explicit pixel height set, `scrollHeight`
// returns `max(contentHeight, clientHeight)` — i.e. the visible box
// height when content fits inside the box. So when the user shrinks
// the content (cut, backspace many lines, send-then-type-less),
// scrollHeight stayed at the previous box height, equaled the cache,
// and the fast path returned without shrinking. Across many messages
// this accreted: every spike set a new floor the box couldn't fall
// back from. Users reported "the textbox gets pretty large overtime
// becoming multiple times taller than normal" — same root cause.
//
// The fix: always reset to 'auto' before measuring so scrollHeight
// reflects content size rather than the existing box size. rAF
// batching alone gives us 60 layouts/sec maximum (matching display
// refresh), which is what the original perf optimization was after.

const _rafScheduled = new WeakSet();

/**
 * Schedule an auto-resize of the textarea on the next animation frame.
 * Safe to call from input/keydown handlers — multiple calls in the
 * same frame coalesce into one layout flush.
 *
 * @param {HTMLTextAreaElement} ta - the textarea to resize
 * @param {number} [maxHeight] - pixel cap; default 50% of viewport
 */
export function scheduleAutosize(ta, maxHeight) {
  if (!ta) return;
  if (_rafScheduled.has(ta)) return;
  _rafScheduled.add(ta);
  requestAnimationFrame(() => {
    _rafScheduled.delete(ta);
    const cap = maxHeight ?? window.innerHeight * 0.5;
    // Always reset to 'auto' BEFORE reading scrollHeight. Without
    // this, an explicit pixel height pins scrollHeight to the box
    // size and we can't detect content shrinkage. See the file
    // header for the bug history.
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, cap) + 'px';
  });
}

/**
 * Force an immediate auto-resize, bypassing the rAF defer. Use for
 * the rare cases that need synchronous layout (e.g. just-mounted
 * textarea, or programmatic value set followed by an animation
 * that needs the final size before the next frame).
 */
export function resizeNow(ta, maxHeight) {
  if (!ta) return;
  const cap = maxHeight ?? window.innerHeight * 0.5;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, cap) + 'px';
}
