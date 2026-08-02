/* ==========================================================================
   Toolbar control — Instant scene generation (narrative mode)

   Click → fire the same scene-generate flow the header's `scene-gen-go` button
   uses. The heavy lifting lives in app.js::_fireSceneGenerate (HTTP roundtrip,
   loader rendering, error toasts); we just trigger it.

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { _fireSceneGenerate } from '../../app.js';
import { tbFind } from './util.js';

/**
 * Wire the instant-scene button inside the given toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireInstantScene(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'instant-scene-btn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    _fireSceneGenerate();
  });
}
