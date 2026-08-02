/* ==========================================================================
   Toolbar utilities — shared by per-control modules under chat/toolbar/.

   Extracted from app.js as part of the surface-owned composer migration
   (Step 2). Each per-control module imports from here instead of reaching
   into app.js internals.
   ========================================================================== */

/**
 * Flash a toolbar button to give visual confirmation that a click registered.
 * The CSS animation lives in components.css under `.toolbar-flash`.
 */
export function flashToolbarBtn(btn) {
  if (!btn) return;
  btn.classList.remove('toolbar-flash');
  // Force reflow so re-adding the class triggers the animation
  void btn.offsetWidth;
  btn.classList.add('toolbar-flash');
}

/**
 * Find a toolbar control by its canonical id key.
 *
 * The primary (singleton) toolbar keeps real `id` attributes. Per-surface
 * toolbar clones re-key every `id` to `data-tid` so `document.getElementById`
 * across the app keeps resolving to the primary's elements (no duplicate-id
 * ambiguity). Wire modules use this helper so one code path serves both.
 */
export function tbFind(rootEl, key) {
  if (!rootEl) return null;
  return rootEl.querySelector(`#${key}`) || rootEl.querySelector(`[data-tid="${key}"]`) || null;
}

/**
 * Sync a single boolean toggle to the backend config API.
 * Used by toolbar buttons whose state lives in user settings.
 */
export async function syncToggleToBackend(key, value) {
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
  } catch { /* best effort — UI state already updated */ }
}
