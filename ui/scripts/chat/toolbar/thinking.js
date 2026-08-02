/* ==========================================================================
   Toolbar control — Thinking toggle (per-surface clone path)

   The PRIMARY's thinking button is owned by settings.js
   (updateThinkingToggleUI) — model detection, effort badges, the OpenAI
   effort picker, and the Qwen preserve popover all render into the
   singleton `#thinking-toggle` + `#thinking-config`. This module is the
   reduced per-surface sibling for secondary tabs:

   - `toggleable` families → full on/off toggle (global setting; the wire
     value already rides every tab's requests via getThinkingOverrideForModel
     in chat/stream.js — this button just makes the state visible/flippable
     from the tab).
   - `effort_select` + offSelectable (DeepSeek V3.2/V4) → the button reads
     as the thinking state and click flips Off ↔ default-on, matching the
     primary's semantics.
   - Everything else (always-on/off, OpenAI 5-level picker) → hidden here;
     the effort picker anchors to the primary's DOM and stays primary-only.

   Model changes have no DOM event — settings.js mutates the primary button
   directly — so re-sync rides a MutationObserver on the primary's button.
   The setting is user-global until spec Step 5, so all tabs mirror the same
   on/off state by design.
   ========================================================================== */

import {
  detectThinkingSupport,
  getThinkingOverrideForModel,
  setThinkingEnabledPreference,
  setReasoningEffortPreference,
  getSettings,
} from '../../settings.js';
import { app } from '../../app.js';
import { flashToolbarBtn, tbFind } from './util.js';

/**
 * Wire the thinking toggle inside a per-surface toolbar clone.
 * Returns a cleanup fn (disconnects the primary-button observer).
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root (clone).
 * @param {object|null}      surface    Owning surface (unused — Step 3).
 */
export function wireThinking(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'thinking-toggle');
  if (!btn) return undefined;

  const sync = () => {
    const model = app?.state?.currentModel || '';
    const support = detectThinkingSupport(model);
    const binary = support.mode === 'toggleable';
    const offSelectable = support.mode === 'effort_select' && !!support.offSelectable;
    if (!binary && !offSelectable) {
      btn.classList.add('hidden');
      return;
    }
    const on = getThinkingOverrideForModel(model) !== false;
    btn.classList.remove('hidden');
    btn.dataset.active = on ? 'true' : 'false';
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.title = on ? 'Thinking on' : 'Thinking off';
    btn.setAttribute('aria-label', btn.title);
  };
  sync();

  btn.addEventListener('click', () => {
    const model = app?.state?.currentModel || '';
    const support = detectThinkingSupport(model);
    if (support.mode === 'toggleable') {
      setThinkingEnabledPreference(!(getThinkingOverrideForModel(model) !== false));
    } else if (support.mode === 'effort_select' && support.offSelectable) {
      setReasoningEffortPreference(getSettings().reasoningEffort === 'off' ? '' : 'off');
    } else {
      return;
    }
    flashToolbarBtn(btn);
    sync();
  });

  // Mirror model changes: updateThinkingToggleUI rewrites the primary
  // button's attributes on every model swap; observing them is the only
  // signal available (no model-changed DOM event exists).
  const primaryBtn = document.getElementById('thinking-toggle');
  let obs = null;
  if (primaryBtn && primaryBtn !== btn) {
    obs = new MutationObserver(sync);
    obs.observe(primaryBtn, { attributes: true, attributeFilter: ['class', 'data-active', 'title'] });
  }
  return () => obs?.disconnect();
}
