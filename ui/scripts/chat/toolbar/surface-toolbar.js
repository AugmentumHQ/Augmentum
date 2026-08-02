/* ==========================================================================
   Per-surface composer toolbar — secondary (non-primary) chat/story tabs.

   Journey: ctrl+click / long-press "open alongside" tabs shipped with a
   bare composer (textarea + send only) because the toolbar is a singleton
   in index.html wired once against the primary surface — the exact gap
   documented in docs/superpowers/specs/2026-05-31-surface-owned-composer-
   design.md §F. This module is the incremental Step 4 of that spec, scoped
   to SECONDARY surfaces only: the primary keeps its singleton DOM (no
   risky cutover), while every independent tab gets a real toolbar built
   from the singleton's markup as a template.

   How it works:
   - Clone `#input-toolbar`, strip the controls that can't be double-wired
     yet (see _STRIP below, each with its reason), re-key every `id` to
     `data-tid` so `document.getElementById` still uniquely resolves to the
     primary's elements, then wire the surviving controls with the same
     chat/toolbar/* modules the primary uses (they query via tbFind, which
     matches both `#id` and `[data-tid]`).
   - Visibility is gated by the OWNING SURFACE's mode via
     updateSurfaceToolbarMode — not by the global applyMode. A story tab
     shows story controls even while the global mode is chat, which is the
     "toolbar gets janky with multiple tabs" class fix.
   - Controls that delegate to global handlers (instant-scene, reading-room,
     tuning) operate on the ACTIVE session/mode — correct for the focused
     tab because Surface.activate() promotes its session to active. That is
     the same contract the primary's controls have.

   Stripped for now (Step 4 full-parity is a later pass):
   flow-bar, web search sheet, thinking toggle, knowledge-pack bar,
   canvas dock, companion summon, auto-bg (its 3-state machine lives in
   app.js::_setAutoBgState against the primary's DOM), mobile overflow.
   ========================================================================== */

import { tbFind } from './util.js';
import { wireTools } from './tools.js';
import { wireChatTuning } from './tuning.js';
import { wireAutoSearch } from './auto-search.js';
import { wireAutoRead } from './auto-read.js';
import { wireReadingRoom } from './reading-room.js';
import { wireNarrativeBubbles } from './narrative-bubbles.js';
import { wireInstantScene } from './instant-scene.js';
import { wireBgRotation } from './bg-rotation.js';
import { wireThinking } from './thinking.js';
import { wireDocContext } from './doc-context.js';

/** Controls removed from the clone — each is a singleton that can't be
 *  double-wired yet. Keep the reasons; they are the remaining work list. */
const _STRIP = [
  '#toolbar-overflow-btn',    // overflow.js is module-singleton (mobile parking)
  '#toolbar-overflow-popover',
  '#canvas-toggle-btn',       // canvas-dock singleton
  '#web-search-wrap',         // needs the #composer-search sheet sibling
  '#thinking-config',         // preserve popover + effort picker anchor to primary DOM
  '#becca-summon-btn',        // becca-bootstrap wires by id
  '#auto-bg-btn',             // _setAutoBgState drives the PRIMARY's button/config
  '#auto-bg-config',
  '#flow-bar',                // flow-bar.js singleton (injected into the template)
  '.flow-bar',
  // Runtime-injected into the primary toolbar; cloneNode drops their
  // listeners, so a copy would be a dead button.
  '.toolbar-multi-model-wrap', // chat/multi-model.js compare-models wrap
  '.cast-shelf-trigger',       // cast-shelf.js cast trigger
];

const _CHAT_FAMILY = ['passthrough', 'analytical', 'agentic'];

/**
 * Gate control visibility by the owning surface's mode. Mirrors the
 * applyMode() toolbar branch (spec §D table), scoped to one toolbar element.
 * Exported so surfaces can re-gate when their mode changes in place
 * (left-panel session click flipping a chat tab analytical↔passthrough).
 */
export function updateSurfaceToolbarMode(toolbarEl, mode) {
  if (!toolbarEl) return;
  const show = (key, on) => {
    const el = tbFind(toolbarEl, key);
    if (el) el.classList.toggle('hidden', !on);
  };
  const isNarrative = mode === 'narrative';
  const isChatFamily = _CHAT_FAMILY.includes(mode);
  show('attach-btn', true);
  show('chat-tuning-btn', true);
  show('auto-read-btn', true);
  show('bg-rotation-btn', mode !== 'coder');
  show('tools-toggle-wrap', isChatFamily);
  show('auto-search-btn', mode === 'analytical' || mode === 'agentic');
  // Knowledge context bar is chat-family only (matches
  // app.js::_modeSupportsDocContext); its renderer re-gates by mode too.
  show('doc-context-bar', isChatFamily);
  // NOTE: thinking-toggle is deliberately absent here — it's model-gated,
  // not mode-gated (wireThinking owns its visibility).
  show('instant-scene-btn', isNarrative);
  show('reading-room-btn', isNarrative);
  show('narrative-bubbles-btn', isNarrative);
}

function _wireAttach(toolbarEl, chatInput) {
  const btn = tbFind(toolbarEl, 'attach-btn');
  if (!btn || !chatInput) return;
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.multiple = true;
  fileInput.style.display = 'none';
  toolbarEl.appendChild(fileInput);
  btn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    for (const file of fileInput.files || []) {
      if (!file.type.startsWith('image/')) continue;
      const reader = new FileReader();
      reader.onload = () => chatInput.addImage(reader.result);
      reader.readAsDataURL(file);
    }
    fileInput.value = '';
  });
}

/**
 * Build and wire a per-surface toolbar inside a secondary surface's
 * `.input-area`. Returns `{ el, updateMode, cleanup }` or null when the
 * singleton template isn't in the document (tests, stripped builds).
 *
 * @param {object} surface    Owning surface (ChatSurface | NarrativeSurface)
 * @param {object} chatInput  The surface's ChatInput instance
 * @param {HTMLElement} areaEl  The surface's `.input-area` element
 */
export function buildSurfaceToolbar(surface, chatInput, areaEl) {
  const template = document.getElementById('input-toolbar');
  if (!template || !areaEl) return null;

  const clone = template.cloneNode(true);
  clone.removeAttribute('id');
  for (const sel of _STRIP) clone.querySelector(sel)?.remove();

  // Re-key ids → data-tid. document.getElementById keeps resolving to the
  // primary's controls; wire modules find the clone's via tbFind.
  for (const el of clone.querySelectorAll('[id]')) {
    el.dataset.tid = el.id;
    el.removeAttribute('id');
  }
  clone.classList.remove('hidden');

  const mode = surface?._mode || surface?.getContext?.().mode || 'passthrough';
  updateSurfaceToolbarMode(clone, mode);

  // Toolbar sits above the attachment strip + input row, same as primary.
  areaEl.insertBefore(clone, areaEl.firstChild);

  const cleanups = [];
  const push = (fn) => { if (typeof fn === 'function') cleanups.push(fn); };

  _wireAttach(clone, chatInput);
  push(wireTools(clone, surface));
  push(wireChatTuning(clone, surface));
  push(wireThinking(clone, surface));
  push(wireDocContext(clone, surface));
  wireAutoSearch(clone, surface);
  wireAutoRead(clone, surface);
  wireReadingRoom(clone, surface);
  wireNarrativeBubbles(clone, surface);
  wireInstantScene(clone, surface);
  wireBgRotation(clone, surface);

  return {
    el: clone,
    updateMode: (m) => updateSurfaceToolbarMode(clone, m),
    cleanup: () => {
      for (const fn of cleanups.splice(0)) {
        try { fn(); } catch { /* teardown is best-effort */ }
      }
      clone.remove();
    },
  };
}
