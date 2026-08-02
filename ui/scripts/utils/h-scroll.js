/* ==========================================================================
   Universal Horizontal-Scroll Helper
   Translates a mouse-wheel vertical delta into horizontal scrollLeft for
   containers that scroll horizontally but not vertically (tab strips,
   chip rails, model pickers, etc.). Single document-level listener walks
   the closest scrollable ancestor on each event — zero per-surface
   wiring required.

   Behavior contract:
     - Mouse wheel (deltaY only) → scrollLeft, when ancestor has
       horizontal-only overflow (overflow-x:auto/scroll AND
       scrollHeight <= clientHeight).
     - Trackpad horizontal two-finger gestures (deltaX dominant) → left
       to native (do NOT intercept).
     - Touchscreen pan → native (the listener only fires on `wheel`).
     - Ctrl+wheel (pinch-zoom on trackpad) → left to native.
     - Surfaces with their own wheel handler that calls preventDefault
       win (`e.defaultPrevented` short-circuits us), so the existing
       grove-ambient and settings-nav handlers continue to drive
       themselves without double-scroll.
     - Opt-out per subtree via `data-h-scroll="off"`.

   ========================================================================== */

const _MAX_ANCESTOR_WALK = 12;

function _isHorizontalOnlyScroller(el) {
  if (!el || el.nodeType !== 1) return false;
  if (el.scrollWidth <= el.clientWidth) return false;
  // Allow 1px slop for sub-pixel rounding; anything beyond means the
  // container scrolls vertically too and we should leave the wheel alone.
  if (el.scrollHeight - el.clientHeight > 1) return false;
  const cs = getComputedStyle(el);
  if (cs.overflowX !== 'auto' && cs.overflowX !== 'scroll') return false;
  return true;
}

function _findHorizontalAncestor(target) {
  let el = target;
  for (let i = 0; i < _MAX_ANCESTOR_WALK && el && el !== document.body; i++) {
    if (el.dataset && el.dataset.hScroll === 'off') return null;
    if (_isHorizontalOnlyScroller(el)) return el;
    el = el.parentElement;
  }
  return null;
}

let _installed = false;
export function initHorizontalScroll() {
  if (_installed) return;
  _installed = true;
  document.addEventListener('wheel', (e) => {
    if (e.defaultPrevented) return;
    if (e.ctrlKey) return;
    // Trackpad horizontal gestures already do the right thing natively;
    // only intercept when the wheel is primarily vertical.
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
    const el = _findHorizontalAncestor(e.target);
    if (!el) return;
    const max = el.scrollWidth - el.clientWidth;
    if (max <= 0) return;
    const canAdvance = (e.deltaY > 0 && el.scrollLeft < max)
                    || (e.deltaY < 0 && el.scrollLeft > 0);
    if (!canAdvance) return;
    e.preventDefault();
    el.scrollLeft = Math.max(0, Math.min(max, el.scrollLeft + e.deltaY));
  }, { passive: false });
}
