/* ==========================================================================
   Storybook flow renderer

   Owns the ``#task-flow-body`` slot during a storybook generation run.

   Layout (top → bottom):
     - Cover slot  (cover image once available)
     - Style chip   (the planner's one-line visual brief)
     - Character chips (named characters from the planner)
     - Chapter strip (one card per chapter — heading, status dot, thumbnail)

   Driven by these meta events:
     - ``ebook_plan``           one-shot: title, author, style, characters, chapters
     - ``chapter_illustration`` per-chapter: index, status (rendering|complete|failed),
                                url (when complete), prompt, error
   ========================================================================== */

import { escapeHtml } from '../app.js';

// Flow id is the canonical id from reasoning/templates.py — this MUST match
// or the registry won't pick this renderer.
const FLOW_ID = 'flow_research_illustrate';

// Per-task local state. Wiped on reset().
const _state = {
  title: '',
  author: '',
  style: '',
  characters: {},     // {name: description}
  chapters: [],       // [{index, heading, prompt, status, url, error}]
  rendered: false,    // true once initial scaffold is in DOM
};

function _wipe() {
  _state.title = '';
  _state.author = '';
  _state.style = '';
  _state.characters = {};
  _state.chapters = [];
  _state.rendered = false;
}

function _scaffoldHtml() {
  return `
    <div class="storybook-panel">
      <div class="storybook-header">
        <div class="storybook-title" data-role="title"></div>
        <div class="storybook-author" data-role="author"></div>
      </div>
      <div class="storybook-style" data-role="style" hidden></div>
      <div class="storybook-characters" data-role="characters" hidden></div>
      <div class="storybook-chapters" data-role="chapters"></div>
    </div>
  `;
}

function _ensureScaffold(slot) {
  if (_state.rendered) return;
  slot.innerHTML = _scaffoldHtml();
  _state.rendered = true;
}

function _paintHeader(slot) {
  const titleEl = slot.querySelector('[data-role="title"]');
  const authorEl = slot.querySelector('[data-role="author"]');
  if (titleEl) titleEl.textContent = _state.title || 'Untitled';
  if (authorEl) authorEl.textContent = _state.author ? `by ${_state.author}` : '';
}

function _paintStyle(slot) {
  const el = slot.querySelector('[data-role="style"]');
  if (!el) return;
  if (_state.style) {
    el.textContent = _state.style;
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

function _paintCharacters(slot) {
  const el = slot.querySelector('[data-role="characters"]');
  if (!el) return;
  const names = Object.keys(_state.characters);
  if (!names.length) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = names.map(name => {
    const desc = _state.characters[name] || '';
    const safeName = escapeHtml(name);
    const safeDesc = escapeHtml(desc);
    return `<span class="storybook-char-chip" title="${safeDesc}">${safeName}</span>`;
  }).join('');
}

function _chapterCardHtml(ch) {
  const idx = Number(ch.index || 0);
  const heading = escapeHtml(ch.heading || `Chapter ${idx + 1}`);
  const status = (ch.status || 'pending');
  const url = ch.url || '';
  const error = escapeHtml(ch.error || '');

  let media = '';
  if (status === 'complete' && url) {
    media = `<img class="storybook-chapter-img" src="${escapeHtml(url)}"
                  alt="${heading}" loading="lazy" decoding="async" />`;
  } else if (status === 'rendering') {
    media = '<div class="storybook-chapter-thumb pending"><div class="storybook-spinner"></div></div>';
  } else if (status === 'failed') {
    media = `<div class="storybook-chapter-thumb failed" title="${error}">×</div>`;
  } else {
    media = '<div class="storybook-chapter-thumb idle"></div>';
  }

  return `<div class="storybook-chapter" data-status="${status}" data-idx="${idx}">
    <div class="storybook-chapter-media">${media}</div>
    <div class="storybook-chapter-body">
      <div class="storybook-chapter-num">Ch. ${idx + 1}</div>
      <div class="storybook-chapter-heading">${heading}</div>
    </div>
  </div>`;
}

function _paintChapters(slot) {
  const el = slot.querySelector('[data-role="chapters"]');
  if (!el) return;
  if (!_state.chapters.length) { el.innerHTML = ''; return; }
  el.innerHTML = _state.chapters.map(_chapterCardHtml).join('');
}

function _onPlan(slot, plan) {
  _state.title = plan.title || _state.title;
  _state.author = plan.author || _state.author;
  _state.style = plan.style || '';
  _state.characters = plan.characters || {};
  // Initialize chapter rows from the plan, preserving any existing status
  // (illustration events can land before the plan if the planner failed
  // and we fell back to legacy heuristics).
  const incoming = Array.isArray(plan.chapters) ? plan.chapters : [];
  const byIdx = new Map(_state.chapters.map(ch => [ch.index, ch]));
  _state.chapters = incoming.map(ch => {
    const existing = byIdx.get(ch.index) || {};
    return {
      index: ch.index,
      heading: ch.heading || existing.heading || '',
      prompt: ch.prompt || existing.prompt || '',
      status: existing.status || 'pending',
      url: existing.url || '',
      error: existing.error || '',
    };
  });
  _ensureScaffold(slot);
  _paintHeader(slot);
  _paintStyle(slot);
  _paintCharacters(slot);
  _paintChapters(slot);
}

function _onChapter(slot, ev) {
  const idx = Number(ev.index || 0);
  // Find or create the row. The planner event normally seeds these but
  // we tolerate out-of-order arrivals (early image events before plan).
  let row = _state.chapters.find(c => c.index === idx);
  if (!row) {
    row = { index: idx, heading: '', prompt: '', status: 'pending', url: '', error: '' };
    _state.chapters.push(row);
    _state.chapters.sort((a, b) => a.index - b.index);
  }
  if (ev.heading) row.heading = ev.heading;
  if (ev.prompt) row.prompt = ev.prompt;
  if (ev.status) row.status = ev.status;
  if (ev.url) row.url = ev.url;
  if (ev.error) row.error = ev.error;

  _ensureScaffold(slot);
  // Targeted DOM update — replace just the affected card so completed
  // images don't flash on every event.
  const container = slot.querySelector('[data-role="chapters"]');
  if (!container) return;
  const existing = container.querySelector(`.storybook-chapter[data-idx="${idx}"]`);
  const html = _chapterCardHtml(row).trim();
  const tmp = document.createElement('template');
  tmp.innerHTML = html;
  const node = tmp.content.firstElementChild;
  if (existing && node) {
    container.replaceChild(node, existing);
  } else if (node) {
    // Insert in index order so out-of-order arrivals don't reverse.
    let inserted = false;
    for (const child of Array.from(container.children)) {
      if (Number(child.dataset.idx) > idx) {
        container.insertBefore(node, child);
        inserted = true;
        break;
      }
    }
    if (!inserted) container.appendChild(node);
  }
}

export const storybookRenderer = {
  id: FLOW_ID,
  reset(_slot) {
    _wipe();
  },
  handle(slot, meta) {
    if (meta && meta.ebook_plan) _onPlan(slot, meta.ebook_plan);
    if (meta && meta.chapter_illustration) _onChapter(slot, meta.chapter_illustration);
  },
};
