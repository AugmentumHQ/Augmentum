/* ==========================================================================
   Toolbar control — Auto-search button

   Toggles `settings.autoSearch` and syncs to the backend `uarf_auto_search`
   tool flag. Visible in analytical + agentic modes (visibility still gated by
   app.js::applyMode for now — moves per-surface in Step 4).

   Step 2 of the surface-owned composer migration:
   - Function signature accepts (toolbarEl, surface) so future steps can scope
     the query to the surface's own toolbar instead of the global document.
   - The `surface` parameter is currently unused (Step 3 will start reading
     from it via surface.state.*).
   ========================================================================== */

import { getSettings, save as saveSettings } from '../../settings.js';
import { showToast } from '../../app.js';
import { flashToolbarBtn, syncToggleToBackend, tbFind } from './util.js';

/**
 * Wire the auto-search button inside the given toolbar root. No-op if the
 * button is not present. Safe to call multiple times only on different
 * toolbar elements — the click handler is registered once per call.
 *
 * @param {HTMLElement|null} toolbarEl  The composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireAutoSearch(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'auto-search-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const s = getSettings();
    s.autoSearch = !s.autoSearch;
    btn.dataset.active = s.autoSearch ? 'true' : 'false';
    flashToolbarBtn(btn);
    saveSettings();
    syncToggleToBackend('uarf_auto_search', s.autoSearch);
    showToast(s.autoSearch ? 'Auto-search enabled' : 'Auto-search disabled', 'info');
  });
}
