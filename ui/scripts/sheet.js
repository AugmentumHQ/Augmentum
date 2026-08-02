/* ==========================================================================
   Sheet — reusable centered modal (desktop) / bottom sheet (mobile).

   Minimal, dependency-free dialog primitive.  Handles the scaffolding every
   creation/confirmation flow needs: backdrop, close button, Escape key,
   focus trap, focus restoration, and a bottom-sheet layout on narrow screens.

   Usage:
     import { openSheet } from './sheet.js';

     const { close, bodyEl, footerEl } = openSheet({
       title: 'Create a Thing',
       body: htmlStringOrElement,
       footer: htmlStringOrElement,
       className: 'my-sheet',            // extra class on the root
       accent: 'var(--mode-analytical)', // sets --sheet-accent inside the sheet
       onClose: () => {},
       initialFocus: '.my-input',        // selector or Element
       closeOnBackdrop: true,
     });

   Only one sheet may be open at a time.  Opening a second will close the first.
   ========================================================================== */

import { escapeHtml } from './app.js';
import { installDialog } from './_focus-trap.js';

let _activeSheet = null;

/**
 * Open a modal sheet. Returns control handles.
 *
 * @param {object} opts
 * @param {string} [opts.title]
 * @param {string|Node} [opts.body]
 * @param {string|Node} [opts.footer]
 * @param {string} [opts.className]
 * @param {string} [opts.accent]
 * @param {function} [opts.onClose]
 * @param {string|Element} [opts.initialFocus]
 * @param {boolean} [opts.closeOnBackdrop=true]
 * @returns {{close: () => void, root: HTMLElement, bodyEl: HTMLElement, footerEl: HTMLElement|null}}
 */
export function openSheet({
  title = '',
  body = '',
  footer = null,
  className = '',
  accent = '',
  onClose = null,
  initialFocus = null,
  closeOnBackdrop = true,
} = {}) {
  // Enforce singleton — second open replaces first.
  if (_activeSheet) _activeSheet.close();

  // --- Backdrop ----------------------------------------------------------
  const backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';

  // --- Sheet root --------------------------------------------------------
  const root = document.createElement('div');
  root.className = 'sheet' + (className ? ` ${className}` : '');
  root.setAttribute('role', 'dialog');
  root.setAttribute('aria-modal', 'true');
  if (title) root.setAttribute('aria-label', title);
  if (accent) root.style.setProperty('--sheet-accent', accent);

  // --- Header ------------------------------------------------------------
  const header = document.createElement('div');
  header.className = 'sheet__header';
  header.innerHTML = `
    <h2 class="sheet__title">${escapeHtml(title || '')}</h2>
    <button class="sheet__close" type="button" aria-label="Close">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="3" y1="3" x2="13" y2="13"/>
        <line x1="13" y1="3" x2="3" y2="13"/>
      </svg>
    </button>
  `;

  // --- Body --------------------------------------------------------------
  const bodyEl = document.createElement('div');
  bodyEl.className = 'sheet__body';
  _mount(bodyEl, body);

  // --- Footer (optional) -------------------------------------------------
  let footerEl = null;
  if (footer != null && footer !== '') {
    footerEl = document.createElement('div');
    footerEl.className = 'sheet__footer';
    _mount(footerEl, footer);
  }

  root.appendChild(header);
  root.appendChild(bodyEl);
  if (footerEl) root.appendChild(footerEl);

  document.body.appendChild(backdrop);
  document.body.appendChild(root);

  // --- Handlers ----------------------------------------------------------
  let closed = false;

  // Keyboard (Tab wrap + Escape), ARIA modality, initial focus, and
  // focus-restore are owned by the shared dialog primitive. Escape routes
  // back through close() so DOM teardown + onClose run exactly once.
  const dialog = installDialog(root, {
    onClose: () => close(),
    initialFocus,
    label: title,
    setAria: false,   // openSheet already set role/aria-modal/aria-label
  });

  const backdropHandler = () => { if (closeOnBackdrop) close(); };
  const closeBtn = header.querySelector('.sheet__close');

  backdrop.addEventListener('click', backdropHandler);
  closeBtn?.addEventListener('click', () => close());

  function close() {
    if (closed) return;
    closed = true;
    _activeSheet = null;
    dialog.release();   // detaches keydown + restores previous focus
    backdrop.remove();
    root.remove();
    if (typeof onClose === 'function') {
      try { onClose(); } catch (e) {
        console.warn('[sheet] onClose threw:', e.message || e);
      }
    }
  }

  _activeSheet = { close };
  return { close, root, bodyEl, footerEl };
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

function _mount(target, content) {
  if (typeof content === 'string') {
    target.innerHTML = content;
  } else if (content instanceof Node) {
    target.appendChild(content);
  }
}
