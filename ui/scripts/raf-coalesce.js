/**
 * raf-coalesce.js — one tiny primitive for the whole streaming-render class.
 *
 * Streamed sources (chat content/thinking, coder shell, cardsmith deltas,
 * bug-finder events, …) arrive at network cadence — often many per frame.
 * Doing DOM work (re-paint, scroll, reflow read) per message is what pins the
 * main thread: O(n²) rewrites, layout thrash, compositor churn. The fix is
 * always the same shape — accumulate cheaply, then do the expensive DOM work
 * AT MOST ONCE per animation frame. This wraps that shape so every surface
 * shares it instead of re-deriving (and re-breaking) it.
 *
 * Pairs with the O(delta) append helpers (chat/thinking-stream.js,
 * chat/stream-render.js): coalescing bounds HOW OFTEN you paint; append-only
 * bounds how much each paint costs.
 */

/**
 * Wrap ``fn`` so repeated calls within one frame collapse into a single
 * invocation on the next animation frame, using the MOST RECENT arguments.
 *
 * Example:
 *   const repaint = rafCoalesce(() => rebuildDashboard());
 *   ws.onmessage = () => repaint();        // 200 msgs/s → ~60 rebuilds/s max
 *
 *   const scrollToEnd = rafCoalesce((el) => { el.scrollTop = el.scrollHeight; });
 *   onDelta = (el) => scrollToEnd(el);     // one reflow/frame, latest el wins
 *
 * @param {Function} fn - the expensive DOM work to coalesce.
 * @returns {Function} a scheduler with the same call signature as ``fn``.
 *   Has ``.cancel()`` to drop a pending run and ``.flush()`` to run it now.
 */
export function rafCoalesce(fn) {
  let scheduled = false;
  let lastArgs = null;
  const run = () => {
    if (!scheduled) return; // cancelled or already flushed this frame
    scheduled = false;
    const args = lastArgs || [];
    lastArgs = null;
    fn(...args);
  };
  const schedule = function (...args) {
    lastArgs = args; // latest call wins — intermediate frames are stale anyway
    if (scheduled) return;
    scheduled = true;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(run);
    } else {
      run(); // test / non-browser host: run synchronously
    }
  };
  schedule.cancel = () => { scheduled = false; lastArgs = null; };
  schedule.flush = () => { if (scheduled) run(); };
  return schedule;
}
