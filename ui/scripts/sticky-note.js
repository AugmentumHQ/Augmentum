/**
 * sticky-note.js — Floating sticky-note overlay.
 *
 * Renders one (or more) compact note tiles anchored to the screen,
 * driven by ``intent_action`` channels:
 *
 *   note.open_sticky    {note_id, title, content}  — surface a sticky
 *   note.update_sticky  {note_id, content}         — live-update body
 *   note.capture_started {note_id}                 — capture-mode UI
 *   note.capture_ended  {note_id}                  — exit capture
 *
 * Each sticky is ~280px wide, draggable (350ms press-and-hold), and has:
 *   - Editable title row + body textarea (auto-saves to /api/notes/:id)
 *   - Expand button → opens the full notes editor in the browse panel
 *   - Close button → hides this sticky (note persists in the DB)
 *
 * Multiple notes can coexist — keyed by note_id in a Map.
 */

const HOST_ID = 'sticky-notes-host';
const MOVE_THRESHOLD_PX = 4;
const AUTOSAVE_DEBOUNCE_MS = 600;

let _host = null;
let _resizeWired = false;
const _stickies = new Map(); // note_id -> { root, autosaveTimer, lastSaved }

export function initStickyNotes() {
  _host = document.getElementById(HOST_ID) || _ensureHost();
  if (!_resizeWired) {
    _resizeWired = true;
    // Stale saved positions (window resized, monitor changed) are why
    // stickies came back half off-screen — re-clamp moved notes.
    window.addEventListener('resize', () => {
      for (const entry of _stickies.values()) {
        if (!entry.root.classList.contains('sticky-moved')) continue;
        const r = entry.root.getBoundingClientRect();
        _applyPos(entry, _clampToViewport(entry, { x: r.left, y: r.top }));
      }
    });
  }
}

function _ensureHost() {
  const h = document.createElement('div');
  h.id = HOST_ID;
  document.body.appendChild(h);
  return h;
}

/** Render or update a sticky for the given note. */
export function showSticky({ note_id, title, content }) {
  if (!_host) initStickyNotes();
  if (!note_id) return;
  let entry = _stickies.get(note_id);
  if (!entry) {
    entry = _buildSticky({ note_id, title, content });
    _stickies.set(note_id, entry);
    _host.appendChild(entry.root);
    _restorePosition(entry);
    requestAnimationFrame(() => entry.root.classList.add('sticky-visible'));
    _refreshAgentLiveness();
  } else {
    // Update in place — preserve focus/selection.
    _updateContent(entry, { title, content });
  }
  return entry;
}

// App menu: "put the note away" — closes the sticky overlay(s) without
// touching the note itself (it persists in the DB; note.show_sticky
// brings it back). The companion presses this via app.act; a dedicated
// verb would be overkill for an arg-less, on-screen, reversible act.
import('./command-palette.js').then(({ registerCommand }) => {
  registerCommand({
    id: 'note.put-away-sticky',
    label: 'Put the sticky note away',
    group: 'Notes',
    keywords: 'close hide put away dismiss sticky note overlay',
    when: () => _stickies.size > 0,
    agent: {
      description: 'Close the sticky note overlay on screen (the note itself is kept)',
      speak: 'Tucked the note away — it\'s saved.',
    },
    run: () => {
      for (const id of [..._stickies.keys()]) closeSticky(id);
    },
  });
}).catch(() => {});

function _refreshAgentLiveness() {
  import('./command-palette.js')
    .then((m) => m.refreshAgentCatalog())
    .catch(() => {});
}

export function updateSticky({ note_id, title, content }) {
  const entry = _stickies.get(note_id);
  if (!entry) {
    // No sticky open for this note — surface one. An incoming update
    // means she's actively writing to it; in the co-author register
    // that's exactly when the user wants it on screen.
    return !!showSticky({ note_id, title, content });
  }
  _updateContent(entry, { title, content });
  return true;
}

export function closeSticky(note_id) {
  const entry = _stickies.get(note_id);
  if (!entry) return;
  if (entry.autosaveTimer) clearTimeout(entry.autosaveTimer);
  entry.root.classList.remove('sticky-visible');
  setTimeout(() => entry.root.remove(), 200);
  _stickies.delete(note_id);
  _refreshAgentLiveness();
}

export function setCaptureState(note_id, capturing) {
  const entry = _stickies.get(note_id);
  if (!entry) return;
  entry.root.dataset.capturing = capturing ? 'true' : 'false';
}

function _buildSticky({ note_id, title, content }) {
  const root = document.createElement('div');
  root.className = 'sticky-note';
  root.dataset.noteId = note_id;
  root.innerHTML = `
    <div class="sticky-header">
      <input class="sticky-title" type="text" placeholder="Untitled" maxlength="200">
      <button class="sticky-btn sticky-expand" type="button" title="Open in editor" aria-label="Open in editor">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
          <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
          <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
      </button>
      <button class="sticky-btn sticky-close" type="button" title="Close" aria-label="Close">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13">
          <path d="M18 6L6 18M6 6l12 12"/>
        </svg>
      </button>
    </div>
    <div class="sticky-capture-indicator" aria-hidden="true">
      <span class="sticky-cap-dot"></span><span class="sticky-cap-label">Listening</span>
    </div>
    <textarea class="sticky-body" placeholder="Start typing or speak…" rows="6"></textarea>
    <div class="sticky-attachments" hidden></div>
  `;
  const entry = {
    root,
    titleInput: root.querySelector('.sticky-title'),
    body: root.querySelector('.sticky-body'),
    attachStrip: root.querySelector('.sticky-attachments'),
    autosaveTimer: null,
    lastSaved: { title, content },
  };
  entry.titleInput.value = title || '';
  entry.body.value = content || '';
  _renderAttachments(entry);

  entry.titleInput.addEventListener('input', () => _scheduleSave(entry));
  entry.body.addEventListener('input', () => {
    _scheduleSave(entry);
    _renderAttachments(entry);
  });
  entry.attachStrip.addEventListener('click', (e) => {
    const img = e.target.closest('.sticky-attach-thumb');
    if (img?.src) window.open(img.src, '_blank', 'noopener');
  });

  root.querySelector('.sticky-expand').addEventListener('click', () => {
    _openInEditor(note_id);
  });
  root.querySelector('.sticky-close').addEventListener('click', () => {
    closeSticky(note_id);
  });

  _wireDrag(entry);
  return entry;
}

function _updateContent(entry, { title, content }) {
  if (typeof title === 'string' && document.activeElement !== entry.titleInput) {
    entry.titleInput.value = title;
  }
  if (typeof content === 'string' && document.activeElement !== entry.body) {
    // Preserve scroll position when content updates from the server side
    // (e.g., thought-capture appending new lines).
    const wasAtBottom =
      entry.body.scrollTop + entry.body.clientHeight >= entry.body.scrollHeight - 4;
    entry.body.value = content;
    if (wasAtBottom) entry.body.scrollTop = entry.body.scrollHeight;
    _renderAttachments(entry);
  }
  entry.lastSaved = {
    title: entry.titleInput.value,
    content: entry.body.value,
  };
}

// Markdown image lines (``![caption](url)``) are the canonical image
// representation in note content — the textarea shows the raw line
// (editable, deletable), this strip renders the actual images. Only
// gallery-relative (/api/...) and http(s) urls render; anything else
// stays inert text.
const _IMG_MD_RE = /!\[([^\]]*)\]\(([^)\s]+)\)/g;

function _renderAttachments(entry) {
  const strip = entry.attachStrip;
  if (!strip) return;
  const matches = [...(entry.body.value || '').matchAll(_IMG_MD_RE)]
    .filter(([, , url]) =>
      url.startsWith('/') || url.startsWith('http://') || url.startsWith('https://'));
  if (!matches.length) {
    strip.hidden = true;
    strip.innerHTML = '';
    return;
  }
  const esc = (s) => String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  strip.hidden = false;
  strip.innerHTML = matches.map(([, cap, url]) =>
    `<img class="sticky-attach-thumb" src="${esc(url)}" alt="${esc(cap || 'attachment')}" title="${esc(cap || '')}" loading="lazy">`,
  ).join('');
}

function _scheduleSave(entry) {
  if (entry.autosaveTimer) clearTimeout(entry.autosaveTimer);
  entry.autosaveTimer = setTimeout(() => {
    entry.autosaveTimer = null;
    _saveNote(entry).catch((err) =>
      console.warn('[sticky] save failed', err),
    );
  }, AUTOSAVE_DEBOUNCE_MS);
}

async function _saveNote(entry) {
  const note_id = entry.root.dataset.noteId;
  if (!note_id) return;
  const payload = {
    title: entry.titleInput.value,
    content: entry.body.value,
  };
  if (
    payload.title === entry.lastSaved.title &&
    payload.content === entry.lastSaved.content
  ) {
    return;
  }
  // Wrap the fetch so a transient network error doesn't reject the
  // autosave promise into a noisy unhandled rejection. We schedule
  // a debounced retry on failure so the user's edit isn't lost when
  // the WS flips around (the chain's debounce already coalesces).
  let resp;
  try {
    // Notes CRUD lives under /api/browse (notes_routes.py). The old
    // /api/notes prefix belongs to note-intelligence and has no PUT —
    // every user edit in the sticky 404'd silently (2026-06-11), which
    // broke the co-author loop: her appends re-read the stored note,
    // so the user's unsaved typing never existed server-side.
    resp = await fetch(`/api/browse/notes/${encodeURIComponent(note_id)}`, {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.warn('[sticky] autosave network error', err);
    _scheduleSave(entry);
    return;
  }
  if (resp.ok) {
    entry.lastSaved = payload;
  } else {
    console.warn('[sticky] autosave failed', resp.status);
    // 401 = auth gone, no point retrying. Other transient codes
    // (502/503/504) get one rescheduled attempt.
    if (resp.status >= 500) _scheduleSave(entry);
  }
}

async function _openInEditor(note_id) {
  // Defer to the existing browse-panel notes editor.
  try {
    const browse = await import('./browse.js');
    browse.openBrowsePanel?.();
  } catch (err) {
    console.warn('[sticky] could not open browse panel', err);
    return;
  }
  document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab', {
    detail: { tab: 'notes' },
  }));
  // The notes panel needs a beat to mount, then we ask it to open this note.
  setTimeout(() => {
    document.dispatchEvent(new CustomEvent('augmentum:open-note', {
      detail: { note_id },
    }));
  }, 200);
}

// ---------------------------------------------------------------------------
// Drag handling — press-and-hold, persisted per note_id in sessionStorage.
// ---------------------------------------------------------------------------

function _posKey(note_id) {
  return `augmentum.sticky.pos.${note_id}`;
}

function _loadPos(note_id) {
  try {
    const raw = sessionStorage.getItem(_posKey(note_id));
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (typeof p?.x === 'number' && typeof p?.y === 'number') return p;
  } catch { /* */ }
  return null;
}

function _savePos(note_id, pos) {
  try {
    if (pos) sessionStorage.setItem(_posKey(note_id), JSON.stringify(pos));
    else sessionStorage.removeItem(_posKey(note_id));
  } catch { /* */ }
}

function _applyPos(entry, pos) {
  const root = entry.root;
  root.style.left = `${pos.x}px`;
  root.style.top = `${pos.y}px`;
  root.style.right = 'auto';
  root.style.bottom = 'auto';
  root.classList.add('sticky-moved');
}

function _clampToViewport(entry, pos) {
  const w = entry.root.offsetWidth || 300;
  const h = entry.root.offsetHeight || 180;
  return {
    x: Math.max(8, Math.min(window.innerWidth - w - 8, pos.x)),
    y: Math.max(8, Math.min(window.innerHeight - h - 8, pos.y)),
  };
}

function _restorePosition(entry) {
  const pos = _loadPos(entry.root.dataset.noteId);
  if (!pos) return;
  // Clamp — the saved position may be from a different viewport.
  requestAnimationFrame(() => _applyPos(entry, _clampToViewport(entry, pos)));
}

function _wireDrag(entry) {
  // Header = drag handle, instant engagement past a small threshold.
  // The old gesture was press-and-hold 350ms; moving during the hold
  // CANCELLED the drag (so natural quick drags never engaged), and
  // pointer capture wasn't taken until after engage — a release in
  // that gap never reached the element, leaving the note glued to an
  // unclicked cursor (2026-06-11, "sticky even when unclicked").
  const root = entry.root;
  const header = root.querySelector('.sticky-header');
  let pointerId = null;
  let startX = 0, startY = 0;
  let elStartX = 0, elStartY = 0;
  let engaged = false;

  const _end = (e) => {
    if (pointerId !== e.pointerId) return;
    try { header.releasePointerCapture(e.pointerId); } catch { /* */ }
    pointerId = null;
    if (engaged) {
      engaged = false;
      root.classList.remove('sticky-dragging');
      const rect = root.getBoundingClientRect();
      _savePos(entry.root.dataset.noteId, { x: rect.left, y: rect.top });
    }
  };

  header.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    if (e.target.closest('.sticky-btn, .sticky-title')) return;
    pointerId = e.pointerId;
    startX = e.clientX;
    startY = e.clientY;
    const rect = root.getBoundingClientRect();
    elStartX = rect.left;
    elStartY = rect.top;
    engaged = false;
    // Capture IMMEDIATELY — guarantees pointerup/move reach us even
    // when the cursor leaves the note mid-gesture.
    try { header.setPointerCapture(e.pointerId); } catch { /* */ }
  });

  header.addEventListener('pointermove', (e) => {
    if (pointerId !== e.pointerId) return;
    if (e.buttons === 0) { _end(e); return; }  // missed-pointerup safety
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (!engaged) {
      if (Math.abs(dx) < MOVE_THRESHOLD_PX && Math.abs(dy) < MOVE_THRESHOLD_PX) {
        return;
      }
      engaged = true;
      root.classList.add('sticky-dragging');
    }
    e.preventDefault();
    _applyPos(entry, _clampToViewport(entry, {
      x: elStartX + dx,
      y: elStartY + dy,
    }));
  });

  header.addEventListener('pointerup', _end);
  header.addEventListener('pointercancel', _end);
}
