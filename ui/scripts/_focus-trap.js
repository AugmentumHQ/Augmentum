/**
 * _focus-trap.js — framework-free modal-dialog a11y primitive.
 *
 * Promotes the focus-trap that lived privately inside sheet.js into a shared
 * helper so every hand-rolled modal (companion self-portrait, reset confirm,
 * topics editor, …) gets the same keyboard contract instead of each
 * re-implementing a partial version:
 *
 *   - role="dialog" + aria-modal="true" on the root (screen-reader modality)
 *   - focus moves into the dialog on open (initialFocus or first focusable)
 *   - Tab / Shift+Tab wrap within the dialog (focus can't escape to the page)
 *   - Escape invokes onClose
 *   - focus is restored to the previously-focused element on release
 *
 * Usage:
 *
 *   import { installDialog } from './_focus-trap.js';
 *
 *   const dlg = installDialog(rootEl, {
 *     onClose: () => closeMyModal(),   // also wire your × button / backdrop to closeMyModal
 *     initialFocus: '.primary-input',
 *     label: 'Edit topics',
 *   });
 *   // in closeMyModal(), after removing the DOM:
 *   dlg.release();
 *
 * installDialog does NOT create or remove DOM and does NOT own the backdrop —
 * it only manages keyboard + focus + ARIA for an element you mount/unmount.
 * Keydown is captured (capture phase) so it wins over page-level handlers.
 */

export const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]):not([type="hidden"]), ' +
  'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function _isVisible(el) {
  // offsetParent is null for display:none or detached; good-enough heuristic.
  return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
}

export function firstFocusable(root) {
  for (const el of root.querySelectorAll(FOCUSABLE_SELECTOR)) {
    if (_isVisible(el)) return el;
  }
  return null;
}

/** Wrap Tab focus within `root` for a Tab keydown event. */
export function trapFocus(root, e) {
  const list = Array.from(root.querySelectorAll(FOCUSABLE_SELECTOR)).filter(_isVisible);
  if (!list.length) { e.preventDefault(); return; }
  const first = list[0];
  const last = list[list.length - 1];
  const active = document.activeElement;
  if (e.shiftKey && active === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  }
}

/**
 * Install dialog keyboard + focus management on a mounted element.
 *
 * @param {HTMLElement} root
 * @param {object} [opts]
 * @param {function} [opts.onClose]      invoked on Escape
 * @param {string|Element} [opts.initialFocus]  selector/Element to focus on open
 * @param {boolean} [opts.escapeCloses=true]
 * @param {boolean} [opts.setAria=true]  set role/aria-modal/aria-label
 * @param {string} [opts.label]          aria-label (only if not already set)
 * @param {boolean} [opts.restoreFocus=true]
 * @returns {{ release: () => void }}
 */
export function installDialog(root, {
  onClose = null,
  initialFocus = null,
  escapeCloses = true,
  setAria = true,
  label = '',
  restoreFocus = true,
} = {}) {
  if (!root) return { release() {} };

  const previousFocus = restoreFocus ? document.activeElement : null;

  if (setAria) {
    if (!root.getAttribute('role')) root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    if (label && !root.getAttribute('aria-label')) root.setAttribute('aria-label', label);
  }

  let released = false;

  const keyHandler = (e) => {
    if (escapeCloses && e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      if (typeof onClose === 'function') {
        try { onClose(); } catch (err) { console.warn('[focus-trap] onClose threw:', err?.message || err); }
      }
      return;
    }
    if (e.key === 'Tab') trapFocus(root, e);
  };

  document.addEventListener('keydown', keyHandler, true);

  // Defer initial focus so layout settles (matches the prior sheet.js timing).
  requestAnimationFrame(() => {
    if (released) return;
    let target = null;
    if (initialFocus) {
      target = typeof initialFocus === 'string' ? root.querySelector(initialFocus) : initialFocus;
    }
    if (!target) target = firstFocusable(root);
    if (target) { try { target.focus({ preventScroll: true }); } catch { /* noop */ } }
  });

  function release() {
    if (released) return;
    released = true;
    document.removeEventListener('keydown', keyHandler, true);
    if (restoreFocus && previousFocus && document.contains(previousFocus)) {
      try { previousFocus.focus({ preventScroll: true }); } catch { /* noop */ }
    }
  }

  return { release };
}
