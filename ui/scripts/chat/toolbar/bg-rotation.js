/* ==========================================================================
   Toolbar control — Background rotation button

   3-state cycle: off → config (dropdowns visible) → active → off

   Sub-elements live inside the same toolbar:
     #bg-rotation-config    — config dropdown wrapper
     #bg-rotation-interval  — rotation interval select
     #bg-rotation-scope     — scope select (narrative-only / all-modes)
     #bg-rotation-frost     — frost overlay checkbox (live — applies even
                              while rotation is active)

   Side effects (state mutation, polling timer) live behind
   window.toggleBgRotation / setBgRotationInterval / setBgRotationScope /
   setBgRotationFrosted in grove-ambient.js. We call those globals; we don't
   own the rotation engine.

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { showToast, applyMode, app } from '../../app.js';
import { flashToolbarBtn, tbFind } from './util.js';

/**
 * Wire the bg-rotation button + its config sub-elements inside the given
 * toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireBgRotation(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'bg-rotation-btn');
  if (!btn) return;

  const config = tbFind(toolbarEl, 'bg-rotation-config');
  const intSel = tbFind(toolbarEl, 'bg-rotation-interval');
  const scopeSel = tbFind(toolbarEl, 'bg-rotation-scope');
  const frostCheck = tbFind(toolbarEl, 'bg-rotation-frost');

  btn.addEventListener('click', () => {
    const isActive = btn.dataset.active === 'true';
    const configShown = config && !config.classList.contains('hidden');

    if (!isActive && !configShown) {
      // off → show config
      if (config) config.classList.remove('hidden');
      btn.title = 'Choose settings, then click to activate';
      const rotState = window._bgRotationState;
      if (intSel && rotState) intSel.value = String(rotState.interval);
      if (scopeSel && rotState) {
        // If configuring from outside Narrative, default the scope to
        // "All modes" — picking "Narrative only" here would activate to
        // no visible effect. Doesn't persist until the user activates.
        scopeSel.value = (rotState.scope === 'narrative' && app.state.mode !== 'narrative')
          ? 'all'
          : rotState.scope;
      }
      if (frostCheck && rotState) frostCheck.checked = rotState.frosted !== false;
    } else if (!isActive && configShown) {
      // config → activate
      if (config) config.classList.add('hidden');
      if (intSel && window.setBgRotationInterval) {
        window.setBgRotationInterval(parseInt(intSel.value, 10) || 30);
      }
      if (scopeSel && window.setBgRotationScope) {
        window.setBgRotationScope(scopeSel.value);
      }
      if (frostCheck && window.setBgRotationFrosted) {
        window.setBgRotationFrosted(frostCheck.checked);
      }
      btn.dataset.active = 'true';
      flashToolbarBtn(btn);
      if (window.toggleBgRotation) window.toggleBgRotation(true);
      showToast('Background rotation enabled', 'success');
      applyMode();
    } else {
      // active → off
      if (config) config.classList.add('hidden');
      btn.dataset.active = 'false';
      flashToolbarBtn(btn);
      if (window.toggleBgRotation) window.toggleBgRotation(false);
      showToast('Background rotation disabled', 'info');
      applyMode();
    }
  });

  // Live frost toggle — updates immediately even while rotation is active
  if (frostCheck) {
    frostCheck.addEventListener('change', () => {
      if (window.setBgRotationFrosted) window.setBgRotationFrosted(frostCheck.checked);
    });
  }
}
