/**
 * coder-window.js — pure windowing math for the coder conversation DOM
 * virtualizer.
 *
 * Kept deliberately DOM-free so it's unit-testable in node (see
 * tests/js/test_coder_window.mjs); the DOM manipulation that consumes these
 * decisions lives in coder-conversation.js (_trimWindowIfNeeded /
 * _rehydrateOlder) and is verified in-browser.
 *
 * WHY THIS EXISTS: a coder conversation grows without bound as the user keeps
 * prompting — every message and tool card ever shown used to stay live in the
 * DOM. With thousands of nodes, EVERY per-frame operation during a stream
 * (markdown flush, auto-scroll, running-banner reflow) pays O(entire-session)
 * layout/style-recalc cost, which is the multi-minute "unusable during
 * generation on a long session" freeze. Capping the number of LIVE nodes keeps
 * that cost O(window) regardless of how long the session runs. Detached nodes
 * are retained (not destroyed) so re-attaching them on scroll-up is exact —
 * identical rendering, handlers, and coalesce grouping, with no re-render.
 */

/**
 * How many of the oldest live nodes to detach so the live DOM stays bounded.
 *
 * Hysteresis: we only start trimming once we're at least `batch` over
 * `maxLive`, and then we trim all the way back down to `maxLive` in one pass.
 * The hysteresis avoids detaching a single node on every append (churn); the
 * trim-to-cap means a long-history load reaches the bound in one call.
 *
 * Never trims while the user has scrolled up — detaching nodes above their
 * viewport would shift what they're reading. We re-trim once they return to
 * the bottom (the streaming common case, where far-above nodes are invisible).
 *
 * @param {{liveCount:number, maxLive:number, batch:number, scrolledUp:boolean}} o
 * @returns {number} count of oldest nodes to detach (0 = leave the DOM alone)
 */
export function nodesToDetach({ liveCount, maxLive, batch, scrolledUp }) {
  if (scrolledUp) return 0;
  const excess = liveCount - maxLive;
  if (excess < batch) return 0;
  return excess;
}

/**
 * How many detached nodes to re-attach when the user scrolls back up to read
 * earlier history. Bounded by what's actually been detached.
 *
 * @param {{detachedCount:number, batch:number}} o
 * @returns {number}
 */
export function nodesToRehydrate({ detachedCount, batch }) {
  return Math.max(0, Math.min(batch, detachedCount));
}
