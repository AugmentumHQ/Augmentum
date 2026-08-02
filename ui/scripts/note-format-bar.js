/**
 * note-format-bar.js — Floating format bar with AI dropdown
 *
 * Appears above selected text in #note-editor-body.
 * Export: init(editorContainer), destroy()
 */

const FORMAT_ACTIONS = [
  { label: 'B',  cmd: 'bold',          title: 'Bold' },
  { label: 'I',  cmd: 'italic',        title: 'Italic', style: 'font-style:italic' },
  { label: 'S',  cmd: 'strikethrough', title: 'Strikethrough', style: 'text-decoration:line-through' },
  { label: '<>', cmd: 'code',          title: 'Inline code', style: 'font-family:monospace' },
  { label: '🔗', cmd: 'link',          title: 'Link' },
  'sep',
  { label: 'H',  cmd: 'heading',       title: 'Heading', style: 'font-weight:700' },
  { label: '❝',  cmd: 'blockquote',    title: 'Quote' },
  'sep',
];

const AI_WRITING = [
  { icon: '✏️', label: 'Rewrite',      action: 'rewrite' },
  { icon: '📝', label: 'Expand',       action: 'expand' },
  { icon: '📐', label: 'Compress',     action: 'compress' },
  { icon: '🎭', label: 'Change tone…', action: 'tone' },
];

const AI_RESEARCH = [
  { icon: '🔍', label: 'Research this', action: 'research' },
  { icon: '📖', label: 'Define',        action: 'define' },
  { icon: '🌐', label: 'Translate…',    action: 'translate' },
];

let bar = null;
let dropdown = null;
let editorEl = null;
let _bound = {};

/* ------------------------------------------------------------------ */
/*  Build DOM                                                          */
/* ------------------------------------------------------------------ */

function createBar() {
  const el = document.createElement('div');
  el.className = 'note-format-bar';

  for (const item of FORMAT_ACTIONS) {
    if (item === 'sep') {
      const sep = document.createElement('span');
      sep.className = 'note-format-bar-sep';
      el.appendChild(sep);
      continue;
    }
    const btn = document.createElement('button');
    btn.title = item.title;
    btn.textContent = item.label;
    if (item.style) btn.setAttribute('style', item.style);
    btn.dataset.cmd = item.cmd;
    btn.addEventListener('mousedown', onFormatClick);
    el.appendChild(btn);
  }

  // AI button
  const aiBtn = document.createElement('button');
  aiBtn.className = 'note-format-bar-ai';
  aiBtn.textContent = '★ AI';
  aiBtn.title = 'AI writing tools';
  aiBtn.addEventListener('mousedown', onAIClick);
  el.appendChild(aiBtn);

  document.body.appendChild(el);
  return el;
}

function createDropdown() {
  const el = document.createElement('div');
  el.className = 'note-ai-dropdown hidden';

  const writingSec = document.createElement('div');
  writingSec.className = 'note-ai-dropdown-section';
  writingSec.textContent = 'Writing';
  el.appendChild(writingSec);

  for (const item of AI_WRITING) {
    el.appendChild(makeDropdownBtn(item));
  }

  const divider = document.createElement('div');
  divider.className = 'note-ai-dropdown-divider';
  el.appendChild(divider);

  const researchSec = document.createElement('div');
  researchSec.className = 'note-ai-dropdown-section';
  researchSec.textContent = 'Research';
  el.appendChild(researchSec);

  for (const item of AI_RESEARCH) {
    el.appendChild(makeDropdownBtn(item));
  }

  document.body.appendChild(el);
  return el;
}

function makeDropdownBtn({ icon, label, action }) {
  const btn = document.createElement('button');
  const iconSpan = document.createElement('span');
  iconSpan.textContent = icon;
  const labelSpan = document.createElement('span');
  labelSpan.textContent = label;
  btn.appendChild(iconSpan);
  btn.appendChild(labelSpan);
  btn.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    document.dispatchEvent(new CustomEvent('note-ai-action', {
      detail: { action, selectedText: text }
    }));
    hideBar();
  });
  return btn;
}

/* ------------------------------------------------------------------ */
/*  Positioning                                                        */
/* ------------------------------------------------------------------ */

function positionBar(range) {
  const rect = range.getBoundingClientRect();
  const barRect = bar.getBoundingClientRect();
  const GAP = 8;

  let top = rect.top - barRect.height - GAP + window.scrollY;
  let flipBelow = false;

  // Flip below if too close to viewport top
  if (rect.top - barRect.height - GAP < 8) {
    top = rect.bottom + GAP + window.scrollY;
    flipBelow = true;
  }

  let left = rect.left + (rect.width / 2) - (barRect.width / 2) + window.scrollX;
  // Clamp horizontal
  left = Math.max(8, Math.min(left, window.innerWidth - barRect.width - 8));

  bar.style.top = `${top}px`;
  bar.style.left = `${left}px`;

  return flipBelow;
}

function positionDropdown(flipBelow) {
  if (!dropdown || dropdown.classList.contains('hidden')) return;
  const barRect = bar.getBoundingClientRect();
  const ddWidth = 200;

  let left = barRect.right - ddWidth + window.scrollX;
  left = Math.max(8, Math.min(left, window.innerWidth - ddWidth - 8));

  let top;
  if (flipBelow) {
    // bar is below selection, dropdown below bar
    top = barRect.bottom + 4 + window.scrollY;
  } else {
    top = barRect.bottom + 4 + window.scrollY;
  }

  dropdown.style.top = `${top}px`;
  dropdown.style.left = `${left}px`;
}

/* ------------------------------------------------------------------ */
/*  Show / Hide                                                        */
/* ------------------------------------------------------------------ */

let _lastFlip = false;

function showBar(range) {
  if (!bar) return;
  // Make visible so we can measure
  bar.classList.add('visible');
  _lastFlip = positionBar(range);
}

function hideBar() {
  if (!bar) return;
  bar.classList.remove('visible');
  if (dropdown) dropdown.classList.add('hidden');
}

/* ------------------------------------------------------------------ */
/*  Event Handlers                                                     */
/* ------------------------------------------------------------------ */

function onSelectionChange() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) { hideBar(); return; }

  const range = sel.getRangeAt(0);
  if (!editorEl || !editorEl.contains(range.commonAncestorContainer)) {
    hideBar();
    return;
  }

  const text = sel.toString().trim();
  if (!text) { hideBar(); return; }

  showBar(range);
}

function onMouseUp() {
  // Small delay so selection is finalised
  requestAnimationFrame(onSelectionChange);
}

function onFormatClick(e) {
  e.preventDefault();
  e.stopPropagation();
  const cmd = e.currentTarget.dataset.cmd;

  if (cmd === 'link') {
    const url = prompt('URL:');
    if (url) document.execCommand('createLink', false, url);
  } else if (cmd === 'code') {
    // Wrap in <code>
    const sel = window.getSelection();
    if (sel && sel.rangeCount) {
      const range = sel.getRangeAt(0);
      const code = document.createElement('code');
      try {
        range.surroundContents(code);
      } catch (_) {
        document.execCommand('insertHTML', false,
          `<code>${sel.toString()}</code>`);
      }
    }
  } else if (cmd === 'heading') {
    document.execCommand('formatBlock', false, 'h2');
  } else if (cmd === 'blockquote') {
    document.execCommand('formatBlock', false, 'blockquote');
  } else {
    document.execCommand(cmd, false, null);
  }
}

function onAIClick(e) {
  e.preventDefault();
  e.stopPropagation();
  if (!dropdown) return;
  const wasHidden = dropdown.classList.contains('hidden');
  dropdown.classList.toggle('hidden');
  if (wasHidden) positionDropdown(_lastFlip);
}

function onDocClick(e) {
  if (bar && !bar.contains(e.target) && dropdown && !dropdown.contains(e.target)) {
    hideBar();
  }
}

function onKeyDown(e) {
  if (e.key === 'Escape') hideBar();
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

export function init(editorContainer) {
  editorEl = editorContainer || document.getElementById('note-editor-body');
  if (!editorEl) return;

  bar = createBar();
  dropdown = createDropdown();

  _bound.selectionChange = onSelectionChange;
  _bound.mouseUp = onMouseUp;
  _bound.docClick = onDocClick;
  _bound.keyDown = onKeyDown;

  document.addEventListener('selectionchange', _bound.selectionChange);
  editorEl.addEventListener('mouseup', _bound.mouseUp);
  document.addEventListener('mousedown', _bound.docClick);
  document.addEventListener('keydown', _bound.keyDown);
}

export function destroy() {
  document.removeEventListener('selectionchange', _bound.selectionChange);
  if (editorEl) editorEl.removeEventListener('mouseup', _bound.mouseUp);
  document.removeEventListener('mousedown', _bound.docClick);
  document.removeEventListener('keydown', _bound.keyDown);

  if (bar) { bar.remove(); bar = null; }
  if (dropdown) { dropdown.remove(); dropdown = null; }
  editorEl = null;
  _bound = {};
}
