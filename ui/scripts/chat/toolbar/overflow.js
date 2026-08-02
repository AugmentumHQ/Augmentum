/* ==========================================================================
   Toolbar control — Mobile overflow ("More") menu

   On phones the per-mode composer buttons crammed the row above the textbox.
   This collapses the secondary buttons that are VISIBLE FOR THE CURRENT MODE
   into a labeled popover menu, keeping every control reachable and named (no
   accessibility loss) and the row uncluttered.

   Mode-aware by construction: applyMode already toggles `hidden` on the
   per-mode buttons, so we simply park whichever eligible buttons are visible
   right now. Re-running on `augmentum:mode-changed` means chat / analyze /
   build / story each overflow their own relevant buttons.

   Safe by design:
     - Eligible buttons are marked `data-overflow` in index.html. Most are
       simple toggles that close the stack when picked. ONE popover-OWNING wrap
       (compare-models) is also eligible — it travels as a unit and its inner
       popover is re-anchored to fixed bottom-left positioning while parked
       (see `.toolbar-overflow-popover .mm-popover` in layout.css) so it doesn't
       shoot off-screen. The click handler below special-cases it so picking it
       doesn't auto-close the stack out from under its just-opened popover.
       (Web search is a plain toggle here — it expands an inline sheet from the
       composer, not a popover, so it SHOULD close the stack like any toggle.)
       Thinking / bg-rotation / tools stay inline (no `data-overflow`).
       auto-bg IS eligible: its model-picker dropdowns (#auto-bg-config) are a
       separate toolbar sibling, so parking only the button is fine — tapping it
       cycles off→config and the stack auto-closes (not the multi-model
       exception), revealing the dropdowns in their natural toolbar slot.
       auto-read (TTS) is deliberately NOT eligible — voice is cross-cutting, so
       it stays inline in every mode.
     - Popover-owners must have their handlers wired BEFORE this module runs
       (see wireWebSearch ordering in app.js): once relayout() re-parents the
       wrap to the body-mounted popover, a toolbar-scoped querySelector for the
       button would miss it and never attach a handler.
     - Buttons are *moved* (re-parented), so their existing event handlers and
       ids survive untouched. A placeholder comment marks each one's inline
       slot so desktop restores exact order.
     - DESKTOP IS UNTOUCHED: every move is gated behind matchMedia(mobile).
       On desktop nothing re-parents and the ⋯ button stays hidden, so a bug
       in the mobile path can't affect the desktop composer.
   ========================================================================== */

const _MOBILE_MQ = '(max-width: 767px)';

let _toolbar = null;
let _btn = null;
let _popover = null;
let _backdrop = null;
let _observer = null;            // watches the toolbar for injected eligibles
let _vvHandler = null;           // keyboard/viewport re-snap while the stack is open
const _placeholders = new Map(); // button.id -> comment node marking inline slot

function _isMobile() {
  try { return window.matchMedia(_MOBILE_MQ).matches; } catch { return false; }
}

// All data-overflow buttons, wherever they currently live (toolbar or popover).
function _eligible() {
  const out = [];
  if (_toolbar) out.push(..._toolbar.querySelectorAll('[data-overflow]'));
  if (_popover) out.push(..._popover.querySelectorAll('[data-overflow]'));
  return out;
}

// Height of the on-screen keyboard, in CSS px (0 when closed). This app runs
// the keyboard in OVERLAY mode (index.html: navigator.virtualKeyboard
// .overlaysContent = true / interactive-widget=resizes-content), so
// `window.innerHeight` does NOT shrink when the keyboard is up and neither
// `window.resize` nor `visualViewport.resize` reliably fires — the only true
// signals are `env(keyboard-inset-bottom)` (CSS) and the VirtualKeyboard API's
// boundingRect / geometrychange (JS). We read the inset here so positioning can
// account for it. visualViewport is a fallback for Safari (no VirtualKeyboard
// API — there the keyboard uses resizes-content, so vv.height DOES shrink).
function _keyboardInset() {
  try {
    const vk = navigator.virtualKeyboard;
    if (vk && vk.boundingRect && vk.boundingRect.height) return vk.boundingRect.height;
  } catch { /* no VirtualKeyboard API */ }
  const vv = window.visualViewport;
  if (vv && vv.height) {
    return Math.max(0, window.innerHeight - vv.height - (vv.offsetTop || 0));
  }
  return 0;
}

// Anchor the chip column straight up from the ⋯ button, targeting the composer's
// RESTING (keyboard-closed) position. Subtracting the keyboard inset is what
// makes that work: the button's gap above the visible-area bottom (top of the
// keyboard) is constant across keyboard states, so `innerHeight - inset -
// rect.top` yields the same bottom offset whether the keyboard is up or down.
// Without the inset, measuring at tap-time (keyboard up, composer floated
// mid-screen, innerHeight still full) stranded the stack near the top with no
// event to correct it once the keyboard dismissed. Re-run on geometrychange /
// vv resize so it also tracks any post-open viewport shift.
function _positionPopover() {
  if (!_popover || !_btn || _popover.classList.contains('hidden')) return;
  const rect = _btn.getBoundingClientRect();
  const inset = _keyboardInset();
  _popover.style.bottom = (window.innerHeight - inset - rect.top + 8) + 'px';
  _popover.style.left = Math.max(8, rect.left + rect.width / 2 - 21) + 'px';
}

// Re-anchor the open stack as the keyboard opens/closes or the viewport shifts.
// Registered across every signal because platforms differ: Chrome/Android in
// overlay mode fires ONLY geometrychange; Safari fires vv resize; desktop
// resize covers the rest. Idempotent add/remove keyed off _vvHandler.
function _addSnapListeners() {
  if (_vvHandler) return;
  _vvHandler = () => _positionPopover();
  try { navigator.virtualKeyboard?.addEventListener('geometrychange', _vvHandler); } catch { /* unsupported */ }
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', _vvHandler);
    window.visualViewport.addEventListener('scroll', _vvHandler);
  }
  window.addEventListener('resize', _vvHandler);
}

function _removeSnapListeners() {
  if (!_vvHandler) return;
  try { navigator.virtualKeyboard?.removeEventListener('geometrychange', _vvHandler); } catch { /* unsupported */ }
  if (window.visualViewport) {
    window.visualViewport.removeEventListener('resize', _vvHandler);
    window.visualViewport.removeEventListener('scroll', _vvHandler);
  }
  window.removeEventListener('resize', _vvHandler);
  _vvHandler = null;
}

function _closePopover() {
  if (_popover) _popover.classList.add('hidden');
  if (_btn) _btn.setAttribute('aria-expanded', 'false');
  if (_backdrop) { _backdrop.remove(); _backdrop = null; }
  _removeSnapListeners();
}

function _openPopover() {
  if (!_popover || !_btn) return;
  // Tapping ⋯ dismisses the keyboard; do it explicitly so it starts retracting
  // immediately. We anchor to the resting position from the start (inset-aware),
  // so the stack lands where the composer will settle and doesn't jump when the
  // keyboard finishes closing.
  const focused = document.activeElement;
  if (focused && typeof focused.blur === 'function' &&
      (focused.tagName === 'INPUT' || focused.tagName === 'TEXTAREA' ||
       focused.isContentEditable)) {
    focused.blur();
  }
  _popover.classList.remove('hidden');
  _btn.setAttribute('aria-expanded', 'true');
  _positionPopover();
  if (!_backdrop) {
    _backdrop = document.createElement('div');
    _backdrop.className = 'toolbar-overflow-backdrop';
    _backdrop.addEventListener('click', _closePopover);
    document.body.appendChild(_backdrop);
  }
  // Re-anchor as the keyboard animation settles (geometrychange) or the viewport
  // otherwise shifts, so the stack tracks the composer instead of freezing.
  _addSnapListeners();
}

// Record an eligible element's inline slot with a placeholder comment so we
// can restore exact order on desktop. Lazy + idempotent — this is how
// dynamically-injected eligibles (e.g. the compare-models wrap, mounted by
// chat/multi-model.js after init) get picked up.
function _register(el) {
  if (!el.id || _placeholders.has(el.id)) return;
  const ph = document.createComment(`overflow-slot:${el.id}`);
  el.parentNode.insertBefore(ph, el);
  _placeholders.set(el.id, ph);
}

/**
 * Reconcile button placement with the current viewport + mode. Idempotent.
 * Disconnects the toolbar observer around its own DOM moves so re-parenting
 * doesn't re-trigger itself.
 */
function relayout() {
  if (!_toolbar || !_btn || !_popover) return;
  // Pick up any newly-injected eligibles sitting in the toolbar.
  for (const el of _toolbar.querySelectorAll('[data-overflow]')) _register(el);

  if (_observer) _observer.disconnect();
  const mobile = _isMobile();
  let parkedVisible = 0;

  for (const btn of _eligible()) {
    const ph = _placeholders.get(btn.id);
    if (!ph) continue;
    // applyMode toggles `hidden` for the active mode — it's our "is this
    // button relevant right now?" signal, and it keeps working in the popover
    // (a mode-hidden button stays display:none there too).
    const modeVisible = !btn.classList.contains('hidden');
    if (mobile && modeVisible) {
      if (btn.parentElement !== _popover) _popover.appendChild(btn);
      parkedVisible++;
    } else if (btn.parentElement !== _toolbar) {
      // Restore to its exact inline slot (before its placeholder).
      ph.parentNode.insertBefore(btn, ph);
    }
  }

  const show = mobile && parkedVisible > 0;
  _btn.classList.toggle('hidden', !show);
  if (!show) _closePopover();

  if (_observer && _toolbar) _observer.observe(_toolbar, { childList: true });
}

/**
 * Wire the overflow button + menu inside the composer toolbar.
 * @param {HTMLElement|null} toolbarEl  The #input-toolbar element.
 */
export function wireOverflow(toolbarEl) {
  if (!toolbarEl) return;
  _toolbar = toolbarEl;
  _btn = toolbarEl.querySelector('#toolbar-overflow-btn');
  _popover = toolbarEl.querySelector('#toolbar-overflow-popover');
  if (!_btn || !_popover) return;

  // Body-mount the popover so it escapes the input-area's backdrop-filter
  // stacking context (same reason the tools dropdown is body-mounted).
  document.body.appendChild(_popover);

  // Watch the toolbar for late-injected eligibles (the compare-models wrap is
  // mounted by chat/multi-model.js after init). relayout() disconnects this
  // around its own moves, so re-parenting never re-triggers the observer.
  // NOT observed yet — the first (deferred) relayout starts it at its end, so
  // no buttons move during synchronous init.
  _observer = new MutationObserver(() => relayout());

  _btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_popover.classList.contains('hidden')) _openPopover();
    else _closePopover();
  });

  // Picking a one-shot button closes the stack (deferred a frame so the
  // button's own handler runs first). The compare-models wrap is the lone
  // exception — its popover lives inside the wrap inside the stack, so closing
  // would hide what it just opened. (Web search is NOT excepted: it opens an
  // inline composer sheet, so we WANT the stack to close behind it — that's why
  // its toggle handler in web-search.js deliberately lets the click bubble.)
  _popover.addEventListener('click', (e) => {
    const hit = e.target.closest('[data-overflow]');
    if (!hit) return;
    if (hit.matches('.toolbar-multi-model-wrap')) return;
    setTimeout(_closePopover, 0);
  });

  document.addEventListener('click', (e) => {
    if (_popover.classList.contains('hidden')) return;
    if (!_popover.contains(e.target) && !_btn.contains(e.target)) _closePopover();
  });

  // Re-run on mode change (per-mode contents) and viewport change. Deferred a
  // macrotask so every other mode-changed handler (applyMode hiding buttons,
  // multi-model toggling its wrap's `hidden`) has finished first — relayout
  // reads those `hidden` states to decide what to park.
  document.addEventListener('augmentum:mode-changed', () => setTimeout(relayout, 0));
  try {
    window.matchMedia(_MOBILE_MQ).addEventListener('change', relayout);
  } catch {
    // Safari < 14 — fall back to resize only.
  }
  window.addEventListener('resize', relayout);

  // Defer the FIRST layout to the next macrotask. wireOverflow runs before
  // other toolbar controls are wired (e.g. wireWebSearch, which queries
  // #input-toolbar for its button) — if we moved buttons out now, those
  // wire-time querySelector('#id') calls would miss the moved element and the
  // button would never get its handler (the "browse never worked" bug). By the
  // time this fires, all synchronous init wiring is done. relayout() starts the
  // MutationObserver at its end, so late injections (compare) are still caught.
  setTimeout(relayout, 0);
}
