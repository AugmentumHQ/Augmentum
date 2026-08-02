/* ==========================================================================
   Toolbar control — Narrative chat-bubbles button (narrative mode)

   Toggles bubble-style rendering for narrative messages. Delegates to
   `window.toggleNarrativeBubbles` (set by narrative/index.js).

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { showToast } from '../../app.js';
import { flashToolbarBtn, tbFind } from './util.js';

/**
 * Wire the narrative-bubbles button inside the given toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireNarrativeBubbles(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'narrative-bubbles-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const isActive = btn.dataset.active === 'true';
    const newState = !isActive;
    btn.dataset.active = newState ? 'true' : 'false';
    flashToolbarBtn(btn);
    if (window.toggleNarrativeBubbles) window.toggleNarrativeBubbles(newState);
    showToast(newState ? 'Chat bubbles on' : 'Chat bubbles off', 'info');
  });
}
