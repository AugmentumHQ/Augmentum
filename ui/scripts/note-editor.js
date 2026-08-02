/* ==========================================================================
   Note Editor — Orchestrator Module
   Manages inline title, metadata line, whisper bar, save coordination,
   format state, and Milkdown editor loading.
   ========================================================================== */

import { escapeHtml } from './app.js';
import * as SlashMenu from './note-slash-menu.js';
import * as MobileToolbar from './note-mobile-toolbar.js';

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
const _state = {
  noteId: null,
  title: '',
  format: 'note',       // 'note' | 'article' | 'journal'
  wordCount: 0,
  saveTimer: null,
  saveStatusTimer: null,
  saveRetryTimer: null,
  saveRetries: 0,
  milkdownEditor: null,
  note: null,            // full note object from API
};

let _dom = {};
let _onSave = null;      // callback: (noteId, { title, content, tags, format }) => Promise
let _slashCallbacks = {};

// ---------------------------------------------------------------------------
// Format labels
// ---------------------------------------------------------------------------
const FORMAT_LABELS = {
  note: 'Note',
  article: 'Article',
  journal: 'Journal',
};
const FORMAT_CYCLE = ['note', 'article', 'journal'];

// ---------------------------------------------------------------------------
// init — called once from browse.js
// ---------------------------------------------------------------------------
export function init({ dom, onSave, slashCallbacks }) {
  _dom = dom;
  _onSave = onSave;
  _slashCallbacks = slashCallbacks || {};

  // Wire slash menu into the editor body — callbacks for generate
  // items (image / AI / knowledge). Actual DOM hookup happens when
  // notes-editor.js registers the CM6 view via SlashMenu.setView.
  if (_dom.editorBody) {
    SlashMenu.attach(_dom.editorBody, _slashCallbacks);
  }

  // Mobile keyboard toolbar — soft-keyboard-aware formatting strip.
  // getView() returns the live CM6 EditorView so the toolbar always
  // targets the note currently being edited. Desktop viewports are
  // hidden via CSS (@media min-width: 768px) and by the visualViewport
  // keyboard-height heuristic in the module.
  MobileToolbar.init({ getView: () => _state.milkdownEditor?.codemirror });

  // Global keyboard shortcuts (find, zen mode, save)
  document.addEventListener('keydown', _handleShortcut);

  // Inline title: Enter → focus the editor body. The current editor is
  // CodeMirror 6 (EditorView); it exposes both .focus() on the wrapper
  // API and a raw view at .codemirror. Fallbacks cover prior editors
  // (EasyMDE CM5, Milkdown/ProseMirror) in case a refresh hasn't
  // happened yet.
  _dom.inlineTitle?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const ed = _state.milkdownEditor;
      if (ed && typeof ed.focus === 'function') { ed.focus(); return; }
      if (ed?.codemirror?.focus) { ed.codemirror.focus(); return; }
      const editable = _dom.editorBody?.querySelector('.cm-content, .CodeMirror, .ProseMirror, [contenteditable]');
      if (editable?.CodeMirror?.focus) { editable.CodeMirror.focus(); return; }
      if (editable) editable.focus();
    }
  });

  // Inline title: input → sync state + debounce save
  _dom.inlineTitle?.addEventListener('input', () => {
    _state.title = (_dom.inlineTitle.textContent || '').trim();
    _debounceSave();
  });

  // Inline title: paste as plain text
  _dom.inlineTitle?.addEventListener('paste', (e) => {
    e.preventDefault();
    const text = (e.clipboardData || window.clipboardData).getData('text/plain');
    document.execCommand('insertText', false, text);
  });

  // Editor body paste: sanitize HTML at capture phase before Milkdown/Crepe
  // sees it. ProseMirror's schema already drops unknown tags/attrs, so this
  // is a defense-in-depth pass — strips on* handlers, style attributes, and
  // forbidden URI schemes (javascript:, data: for scripts) in case a future
  // Milkdown upgrade loosens its schema. Only intercepts when HTML is in
  // the clipboard; plain-text and markdown pastes are untouched and follow
  // Crepe's fast path.
  _dom.editorBody?.addEventListener('paste', (e) => {
    if (typeof DOMPurify === 'undefined') return;
    const cd = e.clipboardData || window.clipboardData;
    if (!cd) return;
    const html = cd.getData('text/html');
    if (!html) return; // plain-text / markdown paste — let Crepe handle
    const cleaned = DOMPurify.sanitize(html, {
      FORBID_ATTR: ['onload','onerror','onclick','onmouseover','onmouseout',
        'onfocus','onblur','onsubmit','onchange','onkeydown','onkeyup',
        'onkeypress','ontouchstart','ontouchend','onpointerdown','onpointerup',
        'formaction','style'],
      FORBID_TAGS: ['script','style','noscript','iframe','object','embed','form','input'],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel):|[^a-z]|[a-z+.-]+(?:[^a-z+.:-]|$))/i,
    });
    if (cleaned === html) return; // Nothing was stripped — let Crepe proceed
    // Replace the clipboard payload for downstream handlers. Chrome/Firefox
    // allow mutating the event's DataTransfer before the default handler
    // consumes it; if setData is unavailable (Safari ~pre-16.4), fall back
    // to preventing default + executing an insertHTML with the clean
    // payload directly, which Crepe's HTML paste rules then parse.
    try {
      cd.setData('text/html', cleaned);
      cd.setData('text/plain', cleaned.replace(/<[^>]+>/g, ''));
    } catch {
      e.preventDefault();
      try { document.execCommand('insertHTML', false, cleaned); } catch { /* best effort */ }
    }
  }, true);

  // Click anywhere in scroll area → focus editor (full-page writing surface feel)
  _dom.scroll?.addEventListener('click', (e) => {
    // Only if click target is the scroll container itself or the writing surface (not a button/input/link)
    const tag = e.target.tagName?.toLowerCase();
    if (tag === 'button' || tag === 'input' || tag === 'a' || tag === 'textarea') return;
    if (e.target.closest('button, input, a, textarea, .note-ask-bar, .note-ai-blocks')) return;
    // If click is below editor content, focus editor at end
    const editable = _dom.editorBody?.querySelector('.cm-content, .ProseMirror, [contenteditable], textarea');
    if (editable && !editable.contains(e.target) && !_dom.inlineTitle?.contains(e.target)) {
      editable.focus();
    }
  });

  // Format button — cycle through formats
  _dom.formatBtn?.addEventListener('click', () => {
    const idx = FORMAT_CYCLE.indexOf(_state.format);
    const next = FORMAT_CYCLE[(idx + 1) % FORMAT_CYCLE.length];
    setFormat(next);
    _debounceSave();
  });

  // Tag add button — show input
  _dom.addTagBtn?.addEventListener('click', () => {
    _dom.tagInput?.classList.remove('hidden');
    _dom.tagInput?.focus();
  });

  // Tag input — Enter/comma to add tag
  _dom.tagInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const tag = (_dom.tagInput.value || '').trim().replace(/^#/, '');
      if (tag && _state.note) {
        if (!_state.note.tags) _state.note.tags = [];
        if (!_state.note.tags.includes(tag)) {
          _state.note.tags.push(tag);
          _renderWhisperTags();
          _debounceSave();
        }
        _dom.tagInput.value = '';
      }
    }
    if (e.key === 'Escape') {
      _dom.tagInput.value = '';
      _dom.tagInput.classList.add('hidden');
    }
  });

  // Tag input — hide on blur if empty
  _dom.tagInput?.addEventListener('blur', () => {
    if (!_dom.tagInput.value.trim()) {
      _dom.tagInput.classList.add('hidden');
    }
  });
}

// ---------------------------------------------------------------------------
// openNote — populate all editor UI from a note object
// ---------------------------------------------------------------------------
export async function openNote(note) {
  // Flush any pending save for previous note
  _flushSave();

  _state.noteId = note.id;
  _state.note = note;
  _state.title = note.title || '';
  _state.format = note.format || 'note';
  _state.wordCount = 0;

  // Inline title
  if (_dom.inlineTitle) {
    _dom.inlineTitle.textContent = _state.title || '';
  }

  // Format label
  setFormat(_state.format);

  // Whisper tags
  _renderWhisperTags();

  // Metadata — source attribution
  _renderSource(note);

  // Show whisper bar
  _dom.whisperBar?.classList.remove('hidden');

  // Load editor via the bridge function (provided by browse.js)
  const markdown = note.content || '';
  if (typeof window.__loadNoteEditor === 'function') {
    _state.milkdownEditor = await window.__loadNoteEditor(
      _dom.editorBody,
      markdown,
      _onContentChange,
    );
  }

  // Compute initial word count
  _updateWordCount(markdown);

  // Clear save status
  _setSaveStatus('');
}

// ---------------------------------------------------------------------------
// closeNote — flush and tear down
// ---------------------------------------------------------------------------
export function closeNote() {
  _flushSave();

  // Cancel any pending failure-retry — clearing noteId below already guards
  // the in-flight flush, but drop the timer too so nothing fires after close.
  if (_state.saveRetryTimer) { clearTimeout(_state.saveRetryTimer); _state.saveRetryTimer = null; }
  _state.saveRetries = 0;
  _state.noteId = null;
  _state.note = null;
  _state.title = '';
  _state.format = 'note';
  _state.wordCount = 0;

  // Destroy milkdown. Also clear the shared window handle used by the
  // loader chain in browse.js so the next open doesn't see a stale
  // instance and try to destroy it twice.
  if (_state.milkdownEditor) {
    try {
      if (_state.milkdownEditor.destroy) _state.milkdownEditor.destroy();
    } catch { /* ignore */ }
    _state.milkdownEditor = null;
  }
  if (typeof window !== 'undefined' && window.__activeCrepeInstance) {
    window.__activeCrepeInstance = null;
  }

  // Clear UI
  if (_dom.inlineTitle) _dom.inlineTitle.textContent = '';
  if (_dom.whisperTags) _dom.whisperTags.innerHTML = '';
  if (_dom.editorBody) _dom.editorBody.innerHTML = '';
  if (_dom.metaWords) _dom.metaWords.textContent = '';
  if (_dom.metaTime) _dom.metaTime.textContent = '';
  if (_dom.metaSource) _dom.metaSource.innerHTML = '';

  // Hide whisper bar
  _dom.whisperBar?.classList.add('hidden');

  // Clear save status
  _setSaveStatus('');
}

// ---------------------------------------------------------------------------
// Getters / setters
// ---------------------------------------------------------------------------
export function getTitle() {
  return _state.title || (_dom.inlineTitle?.textContent || '').trim();
}

export function getFormat() {
  return _state.format;
}

export function setFormat(fmt) {
  _state.format = fmt || 'note';
  if (_dom.formatBtn) {
    _dom.formatBtn.textContent = FORMAT_LABELS[_state.format] || 'Note';
  }
}

export function getMarkdown() {
  if (!_state.milkdownEditor) return _state.note?.content || '';

  // Textarea fallback
  if (_state.milkdownEditor._textarea) {
    return _state.milkdownEditor._textarea.value;
  }

  // Crepe API
  try {
    if (typeof _state.milkdownEditor.getMarkdown === 'function') {
      return _state.milkdownEditor.getMarkdown();
    }
  } catch { /* ignore */ }

  // DOM fallback
  const editorEl = _dom.editorBody?.querySelector('.editor, .ProseMirror, [contenteditable]');
  return editorEl?.textContent || _state.note?.content || '';
}

export function getNoteId() {
  return _state.noteId;
}

export function getNote() {
  return _state.note;
}

// ---------------------------------------------------------------------------
// Zen mode toggle
// ---------------------------------------------------------------------------
let _zenMode = false;

export function toggleZenMode() {
  _toggleZenMode();
}

function _toggleZenMode() {
  _zenMode = !_zenMode;
  document.querySelector('.browse-notes-view')?.classList.toggle('zen-mode', _zenMode);
  if (_zenMode) {
    // CM6 renders `.cm-content`; prefer the editor's own focus() so we
    // don't depend on a specific internal node. (.ProseMirror was the
    // old editor and never matches now.)
    if (_state.milkdownEditor?.focus) _state.milkdownEditor.focus();
    else _dom.editorBody?.querySelector('.cm-content')?.focus();
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
function _handleShortcut(e) {
  const inEditor = document.activeElement?.closest('#note-editor-body, .note-inline-title, #note-writing-surface');
  if (!inEditor) return;
  const mod = e.ctrlKey || e.metaKey;

  if (mod && e.shiftKey && e.key === 'F') { e.preventDefault(); _toggleZenMode(); return; }
  // Find \u2014 route to the CM6 editor's native search panel. The editor also
  // binds Mod-f itself (searchKeymap), and openSearchPanel is idempotent,
  // so intercepting here is safe; it makes Find work even when the inline
  // title is focused and stops the browser's own find from stealing it.
  // (Replaces the old DOM find bar, which queried a `.ProseMirror` node
  // that the CM6 editor never renders, so it always said "No matches".)
  if (mod && !e.shiftKey && e.key === 'f') {
    if (_state.milkdownEditor?.find) { e.preventDefault(); _state.milkdownEditor.find(); }
    return;
  }
  if (mod && e.key === 's') { e.preventDefault(); _flushSave(); return; }
  if (e.key === 'Escape' && _zenMode) { _toggleZenMode(); }
}

// ---------------------------------------------------------------------------
// Content change handler — called by Milkdown on markdownUpdated
// ---------------------------------------------------------------------------
function _onContentChange() {
  const md = getMarkdown();
  _updateWordCount(md);
  _debounceSave();
}

// ---------------------------------------------------------------------------
// Word count + reading time
// ---------------------------------------------------------------------------
function _updateWordCount(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  _state.wordCount = words;

  if (_dom.metaWords) {
    _dom.metaWords.textContent = `${words} ${words === 1 ? 'word' : 'words'}`;
  }
  if (_dom.metaTime) {
    const mins = Math.ceil(words / 200);
    _dom.metaTime.textContent = `${mins} min read`;
  }
}

// ---------------------------------------------------------------------------
// Source attribution
// ---------------------------------------------------------------------------
function _renderSource(note) {
  if (!_dom.metaSource) return;

  if (note.source_url) {
    let hostname = '';
    try { hostname = new URL(note.source_url).hostname; } catch { /* ignore */ }
    _dom.metaSource.innerHTML = `<span class="note-meta-sep">&middot;</span> Clipped from <a href="${escapeHtml(note.source_url)}" target="_blank" rel="noopener">${escapeHtml(note.source_title || hostname)}</a>`;
  } else {
    _dom.metaSource.innerHTML = '';
  }
}

// ---------------------------------------------------------------------------
// Whisper tag rendering
// ---------------------------------------------------------------------------
function _renderWhisperTags() {
  if (!_dom.whisperTags || !_state.note) return;

  _dom.whisperTags.innerHTML = '';

  const tags = _state.note.tags || [];
  tags.forEach(tag => {
    const btn = document.createElement('button');
    btn.className = 'note-whisper-tag';
    btn.textContent = `#${tag}`;
    btn.title = `Remove #${tag}`;
    btn.addEventListener('click', () => {
      _state.note.tags = _state.note.tags.filter(t => t !== tag);
      _renderWhisperTags();
      _debounceSave();
    });
    _dom.whisperTags.appendChild(btn);
  });
}

// ---------------------------------------------------------------------------
// Save coordination — debounced
// ---------------------------------------------------------------------------
function _debounceSave() {
  if (_state.saveTimer) clearTimeout(_state.saveTimer);
  // A fresh edit supersedes any pending failure retry — the upcoming
  // flush will carry the latest content anyway.
  if (_state.saveRetryTimer) { clearTimeout(_state.saveRetryTimer); _state.saveRetryTimer = null; }
  _setSaveStatus('saving');
  _state.saveTimer = setTimeout(() => _flushSave(), 1000);
}

async function _flushSave() {
  if (_state.saveTimer) {
    clearTimeout(_state.saveTimer);
    _state.saveTimer = null;
  }
  if (_state.saveRetryTimer) {
    clearTimeout(_state.saveRetryTimer);
    _state.saveRetryTimer = null;
  }

  if (!_state.noteId || !_state.note || !_onSave) return;

  // Capture the note id we're saving so an async failure that lands after
  // the user has switched notes doesn't retry against the wrong note.
  const noteId = _state.noteId;
  const content = getMarkdown();
  const title = getTitle() || 'Untitled';
  const words = _state.wordCount || 0;

  try {
    await _onSave(noteId, {
      title,
      content,
      tags: _state.note.tags || [],
      format: _state.format,
      word_count: words,
      reading_time_min: Math.max(1, Math.ceil(words / 200)),
    });

    _state.saveRetries = 0;
    _setSaveStatus('saved');
  } catch {
    // Never fail silently — a dropped save means unsynced edits the user
    // can't see. Surface it and retry with backoff until the server
    // recovers or the note is closed. (Any new keystroke also re-arms the
    // debounced save via _debounceSave.)
    if (_state.noteId !== noteId) { _setSaveStatus(''); return; }  // switched notes mid-flight
    _setSaveStatus('error');
    const delay = Math.min(30000, 4000 * 2 ** _state.saveRetries);
    _state.saveRetries += 1;
    _state.saveRetryTimer = setTimeout(() => {
      _state.saveRetryTimer = null;
      if (_state.noteId === noteId) _flushSave();
    }, delay);
  }
}

// ---------------------------------------------------------------------------
// Save status indicator
// ---------------------------------------------------------------------------
function _setSaveStatus(status) {
  const el = _dom.saveStatus;
  if (!el) return;

  if (_state.saveStatusTimer) {
    clearTimeout(_state.saveStatusTimer);
    _state.saveStatusTimer = null;
  }
  el.removeAttribute('title');  // only the error state carries a tooltip

  if (status === 'saving') {
    el.textContent = 'Saving\u2026';
    el.dataset.status = 'saving';
  } else if (status === 'saved') {
    el.textContent = '\u2713 Saved';
    el.dataset.status = 'saved';
    _state.saveStatusTimer = setTimeout(() => {
      el.textContent = '';
      delete el.dataset.status;
    }, 2000);
  } else if (status === 'error') {
    // Persistent (no auto-clear) \u2014 clears on the next successful save.
    el.textContent = 'Couldn\u2019t save \u2014 retrying\u2026';
    el.dataset.status = 'error';
    el.title = 'Your edits are still in the editor. Retrying automatically.';
  } else {
    el.textContent = '';
    delete el.dataset.status;
  }
}

// ---------------------------------------------------------------------------
// Expose debounce for external callers (e.g. AI block changes)
// ---------------------------------------------------------------------------
export function debounceSave() {
  _debounceSave();
}

export function flushSave() {
  return _flushSave();
}
