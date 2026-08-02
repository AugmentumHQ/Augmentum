/* ==========================================================================
   Toolbar control — Reading-room button (narrative mode)

   Toggles the reading-room rendering style. The actual mode change is
   delegated to `window.toggleReadingRoom` (set by narrative/index.js); the
   button is a UI surface for it. Also syncs the inspector checkbox state
   if the inspector is open.

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { flashToolbarBtn, tbFind } from './util.js';

/**
 * Wire the reading-room button inside the given toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireReadingRoom(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'reading-room-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const isActive = btn.dataset.active === 'true';
    const newState = !isActive;
    btn.dataset.active = newState ? 'true' : 'false';
    flashToolbarBtn(btn);
    if (window.toggleReadingRoom) window.toggleReadingRoom(newState);
    const rrCheck = document.getElementById('reading-room-check');
    if (rrCheck) rrCheck.checked = newState;
  });
}
