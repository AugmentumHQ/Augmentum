/**
 * note-mobile-toolbar.js — Mobile keyboard toolbar for the notes editor.
 *
 * Appears above the soft keyboard when the CodeMirror 6 editor is
 * focused and the visual viewport shrinks (keyboard open). Buttons
 * dispatch CM6 transactions rather than `document.execCommand`, which
 * is the right API for CM6's editing model and survives the Live
 * Preview decoration layer.
 *
 * Usage (from note-editor.js):
 *   import * as MobileToolbar from './note-mobile-toolbar.js';
 *   MobileToolbar.init({ getView: () => state.milkdownEditor?.codemirror });
 *
 * The toolbar is desktop-safe: if `window.visualViewport` is unavailable
 * (desktop browsers mostly expose it but never produce a keyboard
 * shrink) it simply never shows. A matching `@media (min-width: 768px)`
 * CSS rule hides it outright on wider viewports.
 */

import { toggleWrap, toggleLinePrefix } from './notes-editor.js';

let toolbar = null;
let viewportHandler = null;
let _getView = null;

const BUTTONS = [
  { label: 'B',     cmd: 'bold',     title: 'Bold',     style: 'font-weight:700' },
  { label: 'I',     cmd: 'italic',   title: 'Italic',   style: 'font-style:italic' },
  { label: 'S',     cmd: 'strike',   title: 'Strikethrough', style: 'text-decoration:line-through' },
  { label: 'H',     cmd: 'heading',  title: 'Heading' },
  'sep',
  { label: '•',     cmd: 'bullet',   title: 'Bullet list' },
  { label: '1.',    cmd: 'numbered', title: 'Numbered list' },
  { label: '☐',     cmd: 'checkbox', title: 'Checkbox' },
  'sep',
  { label: '🔗',    cmd: 'link',     title: 'Insert link' },
  { label: '✨ AI', cmd: 'ai',       title: 'Ask AI',   className: 'note-mobile-ai' },
];

function buildToolbar() {
  toolbar = document.createElement('div');
  toolbar.className = 'note-mobile-toolbar hidden';
  toolbar.setAttribute('role', 'toolbar');
  toolbar.setAttribute('aria-label', 'Formatting toolbar');

  for (const item of BUTTONS) {
    if (item === 'sep') {
      const sep = document.createElement('span');
      sep.className = 'note-mobile-sep';
      sep.setAttribute('aria-hidden', 'true');
      toolbar.appendChild(sep);
      continue;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = item.label;
    if (item.title) btn.title = item.title;
    if (item.title) btn.setAttribute('aria-label', item.title);
    if (item.style) btn.setAttribute('style', item.style);
    if (item.className) btn.classList.add(item.className);
    // mousedown + touchstart are preventDefault'd so the on-screen
    // keyboard and the CM6 selection both stay put while we act.
    const fire = (e) => { e.preventDefault(); runCommand(item.cmd); };
    btn.addEventListener('mousedown', fire);
    btn.addEventListener('touchstart', fire, { passive: false });
    toolbar.appendChild(btn);
  }
  return toolbar;
}

function runCommand(cmd) {
  const view = _getView?.();
  if (!view) return;

  switch (cmd) {
    // Wrap-toggles — uses the shared toggle helper so hitting the
    // same button twice on a word strips the markers instead of
    // stacking `****` around `****hi****`.
    case 'bold':     toggleWrap(view, '**'); break;
    case 'italic':   toggleWrap(view, '*'); break;
    case 'strike':   toggleWrap(view, '~~'); break;
    case 'code':     toggleWrap(view, '`'); break;
    // Line-prefix toggles — recognise variants (e.g. `## `, `1. `,
    // `* `, `- [x]`) so pressing the button on an already-formatted
    // line removes the formatting.
    case 'heading':  toggleLinePrefix(view, '# ',     /^#{1,6}\s+/); break;
    case 'bullet':   toggleLinePrefix(view, '- ',     /^[-*]\s+/); break;
    case 'numbered': toggleLinePrefix(view, '1. ',    /^\d+\.\s+/); break;
    case 'checkbox': toggleLinePrefix(view, '- [ ] ', /^[-*]\s\[[ xX]\]\s+/); break;
    case 'link': {
      // Intentionally lightweight — surface a browser prompt for the
      // URL. A richer flow (bubble menu, link editor) can replace this
      // without changing the toolbar.
      const url = window.prompt('Link URL:');
      if (!url) return;
      const { state } = view;
      const r = state.selection.main;
      const label = r.empty ? 'link text' : state.doc.sliceString(r.from, r.to);
      const insert = `[${label}](${url})`;
      view.dispatch({
        changes: { from: r.from, to: r.to, insert },
        selection: r.empty
          ? { anchor: r.from + 1, head: r.from + 1 + label.length }
          : undefined,
        userEvent: 'input.format',
      });
      view.focus();
      break;
    }
    case 'ai': {
      const selectedText = view.state.sliceDoc(
        view.state.selection.main.from,
        view.state.selection.main.to,
      );
      document.dispatchEvent(new CustomEvent('note-ai-action', {
        detail: { action: 'ai-menu', selectedText },
      }));
      break;
    }
  }
}

function isEditorFocused() {
  const el = document.activeElement;
  if (!el) return false;
  const editorBody = document.getElementById('note-editor-body');
  return !!editorBody && (editorBody === el || editorBody.contains(el));
}

function onViewportResize() {
  if (!toolbar || !window.visualViewport) return;
  const vvHeight = window.visualViewport.height;
  const windowHeight = window.innerHeight;
  const keyboardHeight = windowHeight - vvHeight;

  if (keyboardHeight > 100 && isEditorFocused()) {
    toolbar.classList.remove('hidden');
  } else {
    toolbar.classList.add('hidden');
  }
}

/* ---- Public API ---- */

export function init(opts = {}) {
  // Only meaningful on mobile viewports; visualViewport is universal
  // but the resize→keyboard signal only fires when a soft keyboard
  // appears. The CSS media query at >=768px also hides this.
  if (!window.visualViewport) return;
  if (typeof opts.getView === 'function') _getView = opts.getView;

  if (!toolbar) {
    buildToolbar();
    document.body.appendChild(toolbar);
    viewportHandler = onViewportResize;
    window.visualViewport.addEventListener('resize', viewportHandler);
    window.visualViewport.addEventListener('scroll', viewportHandler);
  }
}

export function setView(view) {
  _getView = view ? () => view : null;
}

export function destroy() {
  if (viewportHandler && window.visualViewport) {
    window.visualViewport.removeEventListener('resize', viewportHandler);
    window.visualViewport.removeEventListener('scroll', viewportHandler);
  }
  if (toolbar && toolbar.parentNode) toolbar.parentNode.removeChild(toolbar);
  toolbar = null;
  viewportHandler = null;
  _getView = null;
}
