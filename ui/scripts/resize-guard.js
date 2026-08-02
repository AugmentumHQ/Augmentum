/* ==========================================================================
   Resize Guard — flags active window-resize on <html> so global CSS can
   suspend GPU-expensive effects (backdrop-filter especially) during the
   drag, then restore them after a short idle. Mitigates a Chromium/Edge
   compositor flash where stacked backdrop-filter layers go transparent
   for a few frames during rapid resize events.
   ========================================================================== */

const ROOT = document.documentElement;
const IDLE_MS = 180;         // post-drag debounce — lets Chromium finish repainting
const MAX_ACTIVE_MS = 3000;  // safety cap if the idle timer never fires

let idleTimer = 0;
let safetyTimer = 0;
let active = false;
let rafClear = 0;

function setActive() {
  if (rafClear) {
    cancelAnimationFrame(rafClear);
    rafClear = 0;
  }
  if (!active) {
    active = true;
    ROOT.dataset.resizing = 'true';
    safetyTimer = window.setTimeout(clearActive, MAX_ACTIVE_MS);
  }
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = window.setTimeout(clearActive, IDLE_MS);
}

function clearActive() {
  if (!active && !ROOT.dataset.resizing) return;
  if (rafClear) return;
  // Clear on the next frame, not directly inside the last resize task.
  // That gives layout a chance to settle before expensive effects come
  // back, which avoids the intermittent black/gray compositor surface.
  rafClear = requestAnimationFrame(() => {
    rafClear = 0;
    active = false;
    delete ROOT.dataset.resizing;
    // Compositor recovery nudge — when a Chromium minimize/maximize
    // or rapid resize drops a GPU layer for the active overlay panel,
    // simply removing data-resizing isn't enough: the dropped layer
    // sometimes never re-rasterizes and the user is left with a black
    // or grey rectangle until the next interaction forces a repaint.
    // Reading offsetHeight on each visible overlay forces a synchronous
    // layout, which in turn makes Chromium re-promote the layer. Cheap
    // (it's a layout read, no style write), and only runs once per
    // resize-settled event.
    _nudgePanelRecomposite();
    window.dispatchEvent(new CustomEvent('augmentum:resize-settled'));
  });
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = 0; }
  if (safetyTimer) { clearTimeout(safetyTimer); safetyTimer = 0; }
}

// Panels that overlay the entire viewport with their own stacking
// context — exactly the surfaces vulnerable to the compositor-loss
// failure mode. Reading offsetHeight forces layout/paint on each that
// is actually visible (skipping `.hidden` skips work for closed ones).
const _NUDGE_SELECTORS = [
  '.browse-panel:not(.hidden)',
  '.files-panel:not(.hidden)',
  '.image-panel:not(.hidden)',
  '.studio-overlay:not(.hidden)',
  '.voice-overlay:not(.hidden)',
  '.flow-editor-overlay:not(.hidden)',
  '.lib-overlay:not(.hidden)',
];
function _nudgePanelRecomposite() {
  try {
    for (const sel of _NUDGE_SELECTORS) {
      const el = document.querySelector(sel);
      if (!el) continue;
      // The void cast is so a future minifier doesn't drop the
      // expression as dead code — its side-effect (forced layout) is
      // the whole point.
      void el.offsetHeight;
    }
  } catch { /* never let a query failure break the resize cleanup */ }
}

window.addEventListener('resize', setActive, { passive: true });
window.addEventListener('orientationchange', setActive, { passive: true });
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', setActive, { passive: true });
}
// Tab visibility change can leave us stuck if a resize finished while
// hidden — clear on return so we don't keep effects suppressed.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) clearActive();
});
