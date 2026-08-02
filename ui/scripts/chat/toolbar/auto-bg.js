/* ==========================================================================
   Toolbar control — Auto-background button (narrative mode)

   3-state cycle:
     off    → config (dropdowns visible)
     config → armed (dropdowns hidden, feature active)
     armed  → off

   State transitions delegate to app.js::_setAutoBgState which owns the
   side effects (backend sync, scene clear, toast, dropdown population).

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { _setAutoBgState } from '../../app.js';

/**
 * Wire the auto-background button inside the given toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireAutoBg(toolbarEl, surface) {
  const btn = toolbarEl?.querySelector('#auto-bg-btn') ?? null;
  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = btn.dataset.state || 'off';
    if (current === 'off') {
      _setAutoBgState('config');
    } else if (current === 'config') {
      _setAutoBgState('armed');
    } else {
      _setAutoBgState('off');
    }
  });
}
