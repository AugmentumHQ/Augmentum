/**
 * thinking-stream.js — pure, dependency-free decision logic for the
 * reasoning/thinking streaming path. Split out of renderer.js (which imports
 * the whole UI chain) so the O(delta) invariant can be unit-tested under node
 * without a DOM — same rationale as stream-fence.js for the content path.
 *
 * The reasoning stream is large (DeepSeek V4 Pro CoT ~4× legacy volume) and
 * arrives at full token rate. The renderer must append ONLY the new tail to
 * the live <pre> each frame (O(delta)); rewriting the whole accumulated string
 * per token is O(n²) and was the observed freeze. The one exception is
 * desync — a block adopted from another path (stored rebuild / resume) or the
 * accumulator reset under it — which resyncs wholesale exactly once.
 */

/**
 * Decide how to reflect ``fullLen`` chars of accumulated thinking into a <pre>
 * that already holds ``renderedLen`` chars.
 *
 * @param {number} renderedLen - chars already in the DOM (block-local cursor;
 *   pass a negative number when the cursor is unknown / block is foreign).
 * @param {number} fullLen - length of the full accumulated thinking string.
 * @returns {{action: 'append'|'resync'|'noop', from: number}}
 *   - 'append': append ``full.slice(from)`` as a text node (the steady state).
 *   - 'resync': set ``pre.textContent = full`` once (cursor lost / shrank).
 *   - 'noop':   nothing new to render this frame.
 */
export function planThinkingAppend(renderedLen, fullLen) {
  if (!Number.isFinite(renderedLen) || renderedLen < 0 || renderedLen > fullLen) {
    return { action: 'resync', from: 0 };
  }
  if (fullLen > renderedLen) {
    return { action: 'append', from: renderedLen };
  }
  return { action: 'noop', from: renderedLen };
}
