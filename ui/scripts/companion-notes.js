/**
 * companion-notes.js — the note pip + drawer + cards (Sprint 3 Piece 13).
 *
 * Mounts a soft glow dot near the chat input. Polls /api/companion/notes.
 * Click opens a drawer that slides up from the bottom of the chat area.
 * Each note renders as a card with 3 actions: Pull it together,
 * Good to know, Mute this topic.
 *
 * Self-contained — does not depend on becca-presence.js. The avatar
 * widget is HER BODY (presence + voice). This pip is HER NOTE (the
 * "she left a sticky note" surface). Two different relational registers.
 *
 * Wire-up: chat.js or the main app bootstrap calls
 * ``CompanionNotes.mount()`` once after auth completes.
 *
 * Resource-conscious:
 *  - Poll interval: 60s default (configurable via setting); pauses
 *    polling when the page is hidden (visibility API).
 *  - Drawer DOM is built once, hidden/shown via class toggle.
 *  - Card list re-rendered only when /notes response changes.
 */

import { createLifetime } from './_lifecycle.js';
import { showToast } from './app.js';
import { COMPANION_STRINGS } from './_companion-strings.js';
import { readAloud, narrateNoteOnce } from './read-aloud.js';

const POLL_INTERVAL_MS = 60000;          // 60s — gentle by design
const HIDDEN_POLL_INTERVAL_MS = 5 * 60 * 1000;  // 5 min when tab hidden

const PRESENCE_MODE_STORAGE_KEY = 'companion.presence.mode';
const PRESENCE_MODE_DEFAULT = 'gentle';

let _pollTimer = null;
let _lastNotes = [];
let _mounted = false;
let _isOpen = false;
// Mount-scoped lifetime tracker — every DOM listener, observer, and
// timer registered through it is torn down on unmount(). becca-bootstrap
// calls mount() on every settings-reconcile (page load, tab refocus,
// reconnect, post-save), so before this was added the same anonymous
// outside-click + visibilitychange handlers piled up indefinitely.
let _lifetime = null;

function _escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;{');
}

// Only allow http(s) in href — neutralizes stored javascript:/data:/
// vbscript: schemes that survive _escapeHtml (curator picks are
// model/scrape-derived, a stored-XSS vector). Returns '#' for anything
// that isn't a parseable http(s) URL. (audit 2026-06-17)
function _safeHref(u) {
  try {
    const p = new URL(String(u), window.location.origin);
    return (p.protocol === 'http:' || p.protocol === 'https:') ? p.href : '#';
  } catch (_) {
    return '#';
  }
}

function _presenceMode() {
  try {
    return localStorage.getItem(PRESENCE_MODE_STORAGE_KEY) || PRESENCE_MODE_DEFAULT;
  } catch (_) {
    return PRESENCE_MODE_DEFAULT;
  }
}

async function _fetchNotes() {
  try {
    const resp = await fetch('/api/companion/notes', { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data.notes) ? data.notes : [];
  } catch (_) {
    return [];
  }
}

async function _postAction(noteId, action) {
  try {
    const resp = await fetch(`/api/companion/notes/${noteId}/${action}`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    return resp.ok;
  } catch (_) {
    return false;
  }
}

// Generic feedback signal — routes to /feedback with the verb in the body
// (server maps it to the bias-function kind). Needed for "dismiss", whose
// negative signal the dedicated action endpoints don't carry. Returns ok.
async function _postFeedback(noteId, verb) {
  try {
    const resp = await fetch(`/api/companion/notes/${noteId}/feedback`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: verb }),
    });
    return resp.ok;
  } catch (_) {
    return false;
  }
}

// Re-parent the dot so it lives inside .becca-presence when the widget
// is mounted, falling back to <body> only so the element is in the DOM
// at all (the CSS rule body.becca-dismissed .companion-note-dot
// {display:none} keeps it invisible in that state).
function _attachDotToWidget(dotEl) {
  if (!dotEl) return;
  const widget = document.querySelector('.becca-presence');
  const target = widget || document.body;
  if (dotEl.parentElement !== target) target.appendChild(dotEl);
}

// Watches document.body for the avatar widget showing up after the
// drawer mounts so we can re-parent the dot into .becca-presence. The
// observer is registered against the per-mount lifetime; previously it
// lived in a module-scope ref that never got disconnected on unmount.
function _watchWidgetMount() {
  if (!_lifetime) return;
  const observer = new MutationObserver(() => {
    const dot = document.getElementById('companion-note-dot');
    if (dot) _attachDotToWidget(dot);
  });
  observer.observe(document.body, { childList: true });
  _lifetime.addObserver(observer);
}

// Outside-click closer. Extracted from the inline handler in
// _buildShell so it can be registered against the lifetime (and so
// unmount actually detaches it). The drawer + dot are looked up at
// call time rather than captured in closure — they may be re-built on
// a remount and stale refs would fail .contains() checks silently.
function _onDocumentClick(e) {
  if (!_isOpen) return;
  const drawer = document.getElementById('companion-note-drawer');
  const dot = document.getElementById('companion-note-dot');
  if (drawer && drawer.contains(e.target)) return;
  if (dot && dot.contains(e.target)) return;
  _closeDrawer();
}

// Tab-visibility refresh — kicks an immediate poll when the user
// returns to the tab so the dot reflects whatever Becca generated
// while we were hidden, then reschedules the regular poll cadence.
function _onVisibilityChange() {
  if (!document.hidden) _refreshNow();
  _schedulePoll();
}

// Backend-driven presence-mode flip — dispatched by becca-bootstrap
// after every /api/config/tools refresh. The drawer's dot visibility
// depends on _presenceMode(), so we re-evaluate immediately instead
// of letting the 60s poll catch up.
function _onPresenceModeChanged() {
  if (!_mounted) return;
  _updateDot(_lastNotes, _lastToday);
}

function _buildShell() {
  // Dot — lives ON the companion widget when it's mounted, hidden
  // otherwise. Earlier design was a standalone fixed-position pip
  // near the chat input, but that read as free-floating chat chrome
  // and the user couldn't tell which surface owned it. Re-parenting
  // it into .becca-presence makes the relational register obvious:
  // "she left a sticky note on her presence", not on the chat.
  const dot = document.createElement('div');
  dot.id = 'companion-note-dot';
  dot.className = 'companion-note-dot hidden';
  dot.setAttribute('role', 'button');
  dot.setAttribute('aria-label', 'Companion has notes');
  dot.tabIndex = 0;

  // Drawer — slides up from bottom of chat area. Two zones:
  //   .companion-today-zone — daily in-her-voice reflection (Today entry)
  //   .companion-note-drawer-body — list of note cards
  //
  // Plus a slide-over history panel (.companion-note-history) that
  // covers the drawer body when the user clicks the history toggle in
  // the header. The history fetch is lazy — only fires on first open —
  // so the drawer's first paint stays cheap.
  const drawer = document.createElement('div');
  drawer.id = 'companion-note-drawer';
  drawer.className = 'companion-note-drawer hidden';
  drawer.innerHTML = `
    <div class="companion-note-drawer-header">
      <div class="companion-note-drawer-titlewrap">
        <span class="companion-note-drawer-title">Notes</span>
        <span class="companion-note-drawer-subtitle">${COMPANION_STRINGS.notesSubtitle}</span>
      </div>
      <div class="companion-note-drawer-controls">
        <button type="button" class="companion-note-drawer-topics"
                aria-label="${COMPANION_STRINGS.watchListCta}" title="${COMPANION_STRINGS.watchListCta}">
          <span aria-hidden="true">⚙</span>
        </button>
        <button type="button" class="companion-note-drawer-history"
                aria-label="Show earlier notes" title="Earlier notes" aria-pressed="false">
          <span class="companion-note-drawer-history-icon" aria-hidden="true">⟨</span>
        </button>
        <button type="button" class="companion-note-drawer-close" aria-label="Close">×</button>
      </div>
    </div>
    <div class="companion-note-coachmark" hidden>
      <div class="companion-note-coachmark-body">
        <p>This drawer is a window into what your companion has been doing in the background — articles it found for you, things it noticed, threads it's wondering about.</p>
        <p>Use <strong>⚙</strong> to set what it watches, and react to each note so it learns what's useful.</p>
      </div>
      <button type="button" class="companion-note-coachmark-dismiss" aria-label="Got it">Got it</button>
    </div>
    <section class="companion-today-zone" data-state="loading">
      <header class="companion-today-header">
        <span class="companion-today-label">Today</span>
        <span class="companion-today-meta"></span>
      </header>
      <div class="companion-today-prose"></div>
      <footer class="companion-today-footer">
        <a href="#" class="companion-today-archive-link" data-action="archive">See archive</a>
      </footer>
    </section>
    <div class="companion-note-drawer-body"></div>
    <aside class="companion-note-history" aria-hidden="true">
      <div class="companion-note-history-header">
        <button type="button" class="companion-note-history-back" aria-label="Back to notes">
          <span class="companion-note-history-back-icon" aria-hidden="true">←</span>
          <span class="companion-note-history-back-text">back</span>
        </button>
        <span class="companion-note-history-title">earlier</span>
        <span class="companion-note-history-spacer" aria-hidden="true"></span>
      </div>
      <div class="companion-note-history-body" data-state="loading">
        <div class="companion-note-history-empty">loading…</div>
      </div>
    </aside>
  `;

  // Drawer stays viewport-fixed so it can slide up from the bottom
  // regardless of where the widget is parked; only the dot moves.
  _attachDotToWidget(dot);
  document.body.appendChild(drawer);
  _watchWidgetMount();

  // Listeners on elements that are CHILDREN of dot/drawer don't need
  // lifetime tracking — they're discarded when the elements are
  // .remove()'d in unmount(). The document-level outside-click is the
  // one that has to be lifetime-tracked.
  dot.addEventListener('click', _openDrawer);
  dot.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') _openDrawer();
  });
  // Expose the drawer-open so other surfaces (e.g. a scheduled-task
  // notification's "Open" affordance) can deep-link the user to the
  // full rich note rather than leaving them with a headline-only banner.
  if (typeof window !== 'undefined') window.openCompanionNotes = _openDrawer;
  drawer.querySelector('.companion-note-drawer-close')
    .addEventListener('click', _closeDrawer);
  drawer.querySelector('.companion-note-drawer-history')
    ?.addEventListener('click', _openHistoryPanel);
  drawer.querySelector('.companion-note-drawer-topics')
    ?.addEventListener('click', _openTopicsModal);
  drawer.querySelector('.companion-note-history-back')
    ?.addEventListener('click', _closeHistoryPanel);
  drawer.querySelector('.companion-note-coachmark-dismiss')
    ?.addEventListener('click', _dismissCoachmark);

  // Delegated card-action handler. Replaces the previous per-card
  // attach pattern in _renderCards — with 20 active notes and a poll
  // cadence that re-renders on every change, that path was binding
  // 100+ listeners per minute. One listener on the body covers every
  // action and survives re-renders.
  const body = drawer.querySelector('.companion-note-drawer-body');
  if (body) body.addEventListener('click', _onCardAction);

  // Document-level outside-click — must be lifetime-tracked or it
  // leaks on every settings-reconcile remount.
  if (_lifetime) {
    _lifetime.addEventListener(document, 'click', _onDocumentClick);
  } else {
    // Defensive: should never happen because mount() creates the
    // lifetime before calling _buildShell. Warn loudly and fall back
    // to a direct attach so the drawer at least functions — the leak
    // it produces is the lesser bug.
    console.warn('[companion-notes] _buildShell with no lifetime — outside-click will leak');
    document.addEventListener('click', _onDocumentClick);
  }
}

// Delegated card-action dispatcher. Routes clicks under
// .companion-note-drawer-body to the appropriate handler based on
// data-action + the enclosing data-note-id. Anchor tags (open-link)
// fall through to their own navigation; we just record the surfaced
// signal and let the browser open the URL.
function _onCardAction(e) {
  const target = e.target.closest('[data-action]');
  if (!target) return;
  const action = target.dataset.action;

  if (action === 'open-topics') {
    _openTopicsModal();
    return;
  }

  const card = target.closest('[data-note-id]');
  if (!card) return;
  const noteId = card.getAttribute('data-note-id');
  if (!noteId) return;

  switch (action) {
    case 'talk':         _handleTalk(noteId, card); break;
    case 'acknowledge':  _handleAcknowledge(noteId, card); break;
    case 'dismiss':      _handleDismiss(noteId, card); break;
    case 'mute-topic':   _handleMute(noteId, card); break;
    case 'save-later':   _handleSaveLater(noteId, card); break;
    case 'open-link':    _handleOpenLink(noteId, card); break;
    case 'read-aloud':   _handleReadAloud(noteId, card, target); break;
    // Unknown action — no-op. Future data-action values render
    // without code changes; we'll surface them when a handler lands.
    default: break;
  }
}

// ── Read-aloud ──────────────────────────────────────────────────────
// Server-TTS playback of a task-result note. Browser autoplay rules mean
// playback must ride a user gesture, so the manual Listen button is the
// reliable path; auto-start (for read_aloud-toggled briefings) is a
// best-effort attempt off the drawer-open gesture.

// Strip bare URLs (sources live in the chips, not the spoken prose) and
// collapse runs of whitespace so the TTS engine gets clean sentences.
function _proseForReadAloud(text) {
  return String(text || '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function _handleReadAloud(noteId, card, btn) {
  const note = _lastNotes.find((n) => String(n.id) === String(noteId));
  const prose = _proseForReadAloud(note ? note.content : '');
  if (!prose) {
    showToast('Nothing to read aloud here.', 'info');
    return;
  }
  // readAloud toggles stop if the same button is clicked while playing.
  readAloud(prose, btn || undefined, {
    title: (note && note.entry_type === 'standing_task') ? 'Briefing' : '',
  });
}

// Best-effort auto-narration when the drawer opens (a user gesture) for the
// freshest read_aloud-toggled briefing not yet narrated this session. Gated
// on a configured TTS voice so users without TTS don't get a warning toast
// on every open; the manual Listen button stays available regardless.
async function _maybeAutoReadAloud() {
  let note;
  try {
    note = (_lastNotes || []).find(
      (n) => n && n.origin && n.origin.read_aloud
        && (n.entry_type || '').toLowerCase() === 'standing_task',
    );
  } catch (_) { return; }
  if (!note) return;
  try {
    const { getSettings } = await import('./settings.js');
    const s = getSettings?.() || {};
    if (!s.voiceDefaultVoice) return;  // no TTS configured → silent skip
  } catch (_) { return; }
  let btn = null;
  try {
    btn = document.querySelector(
      `[data-note-id="${(window.CSS && CSS.escape) ? CSS.escape(String(note.id)) : note.id}"] [data-action="read-aloud"]`,
    );
  } catch (_) { /* selector edge — fall back to no button */ }
  // narrateNoteOnce dedupes by note id against the live-notification path,
  // so a briefing heard live isn't re-spoken when the drawer opens.
  narrateNoteOnce(note.id, note.content, { button: btn || undefined, title: 'Briefing' });
}

// Card rendering is innerHTML-only. Click handling lives on the
// delegated _onCardAction listener attached once in _buildShell, so
// re-rendering cards on every poll tick no longer churns listeners.
//
// Action semantics, for reference:
//   talk         — seed the composer + mark surfaced; closes drawer.
//   acknowledge  — "Yes, that's me / good to know / mark done".
//   dismiss      — soft no (no 90-day topic mute; only the explicit
//                  "mute-topic" action does that).
//   mute-topic   — curator-pick only; 90-day topic suppression.
//   save-later   — wondering-card only; acknowledge + flag for history.
//   open-link    — external link; anchor handles navigation, we just
//                  mark surfaced.
function _renderCards(notes) {
  const body = document.querySelector('#companion-note-drawer .companion-note-drawer-body');
  if (!body) return;
  if (!notes.length) {
    body.innerHTML = _emptyStateHtml();
    return;
  }
  body.innerHTML = notes.map(_cardHtml).join('\n');
}

// ── Card dispatcher ─────────────────────────────────────────────────
//
// Three layouts, one per intent:
//   - curator_note  → pick card    (article preview + open/talk/dismiss/mute)
//   - wondering     → question card (loose-end + pick up / save / drop)
//   - default       → observation card (her noticings about you)
//
// The default branch covers `noticing`, `conversation_moment`,
// `standing_task`, and any future type — the contract is "we have prose
// to display and don't otherwise know what to do with it" rather than
// per-type ceremony for every variant.

function _cardHtml(note) {
  const type = (note.entry_type || '').toLowerCase();
  // Task results carry their own source refs now, but they're a delivered
  // answer (full prose + sources), NOT a single-link curator pick — keep
  // them on the observation layout even though _hasUrlRef is true.
  if (type === 'standing_task') return _observationCardHtml(note);
  if (type === 'curator_note' || _hasUrlRef(note)) return _pickCardHtml(note);
  if (type === 'wondering') return _wonderingCardHtml(note);
  return _observationCardHtml(note);
}

function _hasUrlRef(note) {
  return (note.content_refs || []).some((r) => r && r.kind === 'url' && r.url);
}

// ── Provenance chip ─────────────────────────────────────────────────
//
// Every note answers "why am I seeing this" in its header. `origin` is
// the decoded origin_json from the API (notes v2): which pipeline wrote
// the note, which device's signals fed it, how many, over what window.
// Notes written before migration 257 have no origin and render no chip.
// Always on — provenance is not a setting.

function _provenanceChipHtml(note, suppressLabel) {
  const origin = note && note.origin;
  if (!origin || typeof origin !== 'object' || !origin.source) return '';
  const client = String(origin.client || '').trim();
  const count = parseInt(origin.signal_count, 10) || 0;
  const clientPart = client ? ` (${client})` : '';
  let label = '';
  switch (String(origin.source)) {
    case 'attention':
      label = `from browsing${clientPart}` + (count ? ` · ${count} visit${count === 1 ? '' : 's'}` : '');
      break;
    case 'revisit':
      label = `looped back${clientPart}`;
      break;
    case 'curator':
      label = 'found for you';
      break;
    case 'task':
      label = 'task result';
      break;
    case 'commitment':
      label = 'from our chat';
      break;
    default:
      label = String(origin.source);
  }
  // The observation card's type badge already says "from our chat" /
  // "task result" — don't render the same words twice in one header.
  if (suppressLabel && label === suppressLabel) return '';
  const detailParts = [];
  if (origin.detail) detailParts.push(String(origin.detail));
  if (origin.window) detailParts.push(String(origin.window));
  const title = detailParts.join(' · ');
  return `<span class="companion-note-card-origin"${title ? ` title="${_escapeHtml(title)}"` : ''}>${_escapeHtml(label)}</span>`;
}

function _findUrlRef(note) {
  return (note.content_refs || []).find((r) => r && r.kind === 'url' && r.url) || null;
}

// Parse the curator's composed prose into {topicLine, title, snippet}
// so older notes (written before the rich-refs change) still render
// with structure. Format produced by compose_note_from_rec is:
//
//   On <cluster>\n<title> — <snippet>
//
// Rich refs (post-fix) carry title/snippet directly and skip this path.
function _splitComposedBody(content) {
  const text = String(content || '').trim();
  const nl = text.indexOf('\n');
  let topicLine = nl >= 0 ? text.slice(0, nl).trim() : '';
  // Legacy notes (pre-sanitization) often end the topic line on a stray
  // em-dash, colon, or comma left over from mid-phrase truncation
  // ("On Grok 4.20 First Look –"). Strip trailing punctuation so they
  // read as cleanly as new notes.
  topicLine = topicLine.replace(/[\s\-—–:;,.]+$/u, '').trim();
  // If the topic line collapsed to something too short to read as a
  // phrase ("On –"), hide it entirely rather than showing a fragment.
  if (topicLine.length < 4) topicLine = '';
  const rest = nl >= 0 ? text.slice(nl + 1).trim() : text;
  const dashIdx = rest.indexOf(' — ');
  const title = dashIdx >= 0 ? rest.slice(0, dashIdx).trim() : rest;
  const snippet = dashIdx >= 0 ? rest.slice(dashIdx + 3).trim() : '';
  return { topicLine, title, snippet };
}

function _pickCardHtml(note) {
  const ref = _findUrlRef(note) || {};
  const parsed = _splitComposedBody(note.content);
  // Prefer rich-ref fields when present; fall back to parsed prose so
  // notes written before the curator backend was extended still render
  // with structure rather than as a wall of text.
  const url = ref.url || '';
  const title = (ref.title || parsed.title || '').trim();
  const snippet = (ref.snippet || parsed.snippet || '').trim();
  const domain = (ref.domain || _domainFromUrl(url) || '').trim();
  const topicLine = (parsed.topicLine || '').trim();
  const affect = _escapeHtml(note.affect_tag || '');
  const when = _escapeHtml(_historyRelativeWhen(note.created_at));
  const safeId = _escapeHtml(String(note.id));
  const safeUrl = _escapeHtml(_safeHref(url));
  const safeTitle = _escapeHtml(title);
  const safeSnippet = _escapeHtml(_truncateSentenceAware(snippet, 220));
  const safeDomain = _escapeHtml(domain);
  const safeTopic = _escapeHtml(topicLine);

  const titleHtml = url
    ? `<a class="companion-pick-title" href="${safeUrl}" target="_blank" rel="noopener noreferrer" data-action="open-link">${safeTitle || safeDomain || safeUrl}</a>`
    : `<span class="companion-pick-title">${safeTitle || '(no title)'}</span>`;

  return `
    <article class="companion-note-card companion-pick-card" data-note-id="${safeId}" data-affect="${affect}" data-type="pick">
      <header class="companion-note-card-meta">
        ${when ? `<span class="companion-note-card-when">${when}</span>` : ''}
        ${safeDomain ? `<span class="companion-pick-domain">${safeDomain}</span>` : ''}
        ${_provenanceChipHtml(note)}
        ${affect ? `<span class="companion-note-card-affect">${affect}</span>` : ''}
      </header>
      ${safeTopic ? `<p class="companion-pick-topic">${safeTopic}</p>` : ''}
      <div class="companion-pick-body">
        ${titleHtml}
        ${safeSnippet ? `<p class="companion-pick-snippet">${safeSnippet}</p>` : ''}
      </div>
      <div class="companion-note-card-actions">
        ${url ? `<a class="companion-note-action primary" href="${safeUrl}" target="_blank" rel="noopener noreferrer" data-action="open-link">Open ↗</a>` : ''}
        <button type="button" data-action="talk" class="companion-note-action">Talk about this</button>
        <button type="button" data-action="dismiss" class="companion-note-action">Not for me</button>
        <button type="button" data-action="mute-topic" class="companion-note-action danger" title="Stop surfacing this topic for ~90 days">Mute topic</button>
      </div>
    </article>
  `;
}

function _wonderingCardHtml(note) {
  const affect = _escapeHtml(note.affect_tag || '');
  const when = _escapeHtml(_historyRelativeWhen(note.created_at));
  const safeId = _escapeHtml(String(note.id));
  const body = _escapeHtml(_truncateSentenceAware(note.content || '', 320));
  return `
    <article class="companion-note-card companion-wondering-card" data-note-id="${safeId}" data-affect="${affect}" data-type="wondering">
      <header class="companion-note-card-meta">
        ${when ? `<span class="companion-note-card-when">${when}</span>` : ''}
        <span class="companion-wondering-badge" aria-label="wondering">wondering</span>
        ${_provenanceChipHtml(note)}
        ${affect ? `<span class="companion-note-card-affect">${affect}</span>` : ''}
      </header>
      <div class="companion-note-card-body companion-wondering-body">${body}</div>
      <div class="companion-note-card-actions">
        <button type="button" data-action="talk" class="companion-note-action primary">Let's pick this up</button>
        <button type="button" data-action="save-later" class="companion-note-action">Save for later</button>
        <button type="button" data-action="dismiss" class="companion-note-action">Drop it</button>
      </div>
    </article>
  `;
}

function _observationCardHtml(note) {
  const affect = _escapeHtml(note.affect_tag || '');
  const when = _escapeHtml(_historyRelativeWhen(note.created_at));
  const safeId = _escapeHtml(String(note.id));
  const type = (note.entry_type || '').toLowerCase();
  // A task result is a deliverable the user explicitly asked for — show it
  // in full (don't chop a multi-section answer at 320 chars). Her own
  // noticings stay capped so the drawer doesn't become a wall of text.
  const isTask = type === 'standing_task';
  // Task results are a deliverable the user asked for — render the FULL
  // body (the synthesis token cap already bounds it); only her own
  // noticings stay capped so the drawer doesn't become a wall of text.
  const body = isTask
    ? _escapeHtml(note.content || '')
    : _escapeHtml(_truncateSentenceAware(note.content || '', 320));
  // Conversation moments + standing-task results get a small badge so
  // users can tell "this is a recap" vs "this is something she noticed."
  const badge =
    type === 'conversation_moment' ? 'from our chat' :
    type === 'standing_task' ? 'task result' :
    type === 'noticing' ? 'noticing' : type;
  // Sources for fast research access: prefer the model-curated citations
  // (titled — the body's key references) and fall back to the raw gathered
  // url refs. Both ride in content_refs; citations were previously dropped
  // before render, same as the media below.
  const allRefs = note.content_refs || [];
  const citationRefs = allRefs.filter((r) => r && r.kind === 'citation' && r.url);
  const urlRefs = allRefs.filter((r) => r && r.kind === 'url' && r.url);
  const sourceRefs = citationRefs.length ? citationRefs : urlRefs;
  const sourcesHtml = sourceRefs.length
    ? `<div class="companion-note-sources">
         <div class="companion-note-sources-label">Sources</div>
         <ol class="companion-note-sources-list">${sourceRefs.map((r) => {
            const dom = _escapeHtml(_domainFromUrl(r.url) || r.domain || '');
            const title = _escapeHtml((r.title || '').trim());
            const label = title || dom || _escapeHtml(r.url);
            return `<li><a class="companion-note-source" href="${_escapeHtml(_safeHref(r.url))}" target="_blank" rel="noopener noreferrer" data-action="open-link"><span class="companion-note-source-title">${label}</span>${dom && title ? `<span class="companion-note-source-dom">${dom}</span>` : ''}</a></li>`;
         }).join('')}</ol>
       </div>`
    : '';
  // Rich media a briefing gathered (hero image, video). Previously these
  // were fetched + synthesized over, then dropped before render; now they
  // ride in content_refs as kind:'image'/'video' and surface here — the
  // hero opens full-size, the video opens its source.
  const imgRef = allRefs.find((r) => r && r.kind === 'image' && r.url);
  const heroHtml = imgRef
    ? `<a class="companion-note-hero-link" href="${_escapeHtml(_safeHref(imgRef.url))}" target="_blank" rel="noopener noreferrer" data-action="open-link"><img class="companion-note-hero" src="${_escapeHtml(_safeHref(imgRef.url))}" alt="${_escapeHtml(imgRef.alt || badge || '')}" loading="lazy" decoding="async" /></a>`
    : '';
  const vidRef = allRefs.find((r) => r && r.kind === 'video' && r.url);
  const videoHtml = vidRef
    ? `<a class="companion-note-video" href="${_escapeHtml(_safeHref(vidRef.url))}" target="_blank" rel="noopener noreferrer" data-action="open-link">
         <span class="companion-note-video-play" aria-hidden="true">▶</span>
         <span class="companion-note-video-label">${_escapeHtml(vidRef.summary || 'Watch the clip')}</span>
       </a>`
    : '';
  // Read-aloud: every task result can be spoken (server TTS). When the
  // briefing carries the per-briefing read_aloud toggle (origin.read_aloud)
  // the button is emphasized and the note auto-starts on drawer open.
  const autoRead = !!(note.origin && note.origin.read_aloud);
  const listenBtn = isTask
    ? `<button type="button" data-action="read-aloud" class="companion-note-action companion-note-readaloud${autoRead ? ' is-auto' : ''}" title="Read aloud">
         <span class="companion-readaloud-icon" aria-hidden="true">▶</span> Listen
       </button>`
    : '';
  // Task results aren't "noticings about you" — the yes/no-that's-me framing
  // doesn't fit. Give them result-appropriate actions.
  const actions = isTask
    ? `<button type="button" data-action="talk" class="companion-note-action primary">Go deeper</button>
       ${listenBtn}
       <button type="button" data-action="acknowledge" class="companion-note-action">Got it</button>`
    : `<button type="button" data-action="acknowledge" class="companion-note-action primary">Yes, that's me</button>
       <button type="button" data-action="talk" class="companion-note-action">Tell me more</button>
       <button type="button" data-action="dismiss" class="companion-note-action">Not quite</button>`;
  return `
    <article class="companion-note-card companion-observation-card${isTask ? ' companion-task-card' : ''}" data-note-id="${safeId}" data-affect="${affect}" data-type="observation">
      <header class="companion-note-card-meta">
        ${when ? `<span class="companion-note-card-when">${when}</span>` : ''}
        ${badge ? `<span class="companion-observation-badge">${_escapeHtml(badge)}</span>` : ''}
        ${_provenanceChipHtml(note, badge)}
        ${affect ? `<span class="companion-note-card-affect">${affect}</span>` : ''}
      </header>
      ${heroHtml}
      <div class="companion-note-card-body">${body}</div>
      ${videoHtml}
      ${sourcesHtml}
      <div class="companion-note-card-actions">
        ${actions}
      </div>
    </article>
  `;
}

// ── Utilities ───────────────────────────────────────────────────────

function _domainFromUrl(url) {
  if (!url) return '';
  try {
    const u = new URL(url);
    let host = u.hostname || '';
    if (host.startsWith('www.')) host = host.slice(4);
    return host;
  } catch (_) { return ''; }
}

// Cut at the nearest sentence boundary at or below `max` chars, falling
// back to a word boundary, then to a hard slice. Prevents the current
// `.slice(0, 400)` from producing fragments like "…full-length scen…".
function _truncateSentenceAware(text, max) {
  const s = String(text || '');
  if (s.length <= max) return s;
  const window = s.slice(0, max);
  const sentenceEnd = Math.max(
    window.lastIndexOf('. '),
    window.lastIndexOf('! '),
    window.lastIndexOf('? '),
    window.lastIndexOf('\n'),
  );
  if (sentenceEnd > max * 0.4) return window.slice(0, sentenceEnd + 1).trim();
  const wordEnd = window.lastIndexOf(' ');
  if (wordEnd > max * 0.5) return window.slice(0, wordEnd).trim() + '…';
  return window.trim() + '…';
}

function _emptyStateHtml() {
  return `
    <div class="companion-note-empty">
      <p class="companion-note-empty-prose">Nothing on your companion's mind right now.</p>
      <p class="companion-note-empty-hint">It'll leave a note here when something catches its attention — articles it found, observations about your work, threads it wants to come back to.</p>
      <button type="button" class="companion-note-empty-cta" data-action="open-topics">${COMPANION_STRINGS.watchListCta} ⚙</button>
    </div>
  `;
}

// "Talk about this" / "Tell me more" / "Let's pick this up" — all funnel
// into the same path: pre-seed the chat composer with type-appropriate
// invite text, mark surfaced, close the drawer so the user lands on chat.
async function _handleTalk(noteId, cardEl) {
  const note = _lastNotes.find((n) => String(n.id) === String(noteId));
  const type = (note && (note.entry_type || '')).toLowerCase();
  const when = (note && note.created_at) ? _historyRelativeWhen(note.created_at) : '';
  let seed;
  if (type === 'standing_task') {
    // A scheduled result. The follow-up MUST carry the actual content, or
    // chat has no idea what "go deeper" refers to and just talks about the
    // scheduling itself. Quote an excerpt so the model can genuinely
    // expand on the topic. (Checked before the url-ref branch because task
    // results now carry source refs too.)
    const content = ((note && note.content) || '').trim();
    const excerpt = content.length > 320
      ? content.slice(0, 320).trim() + '…'
      : content;
    seed = excerpt
      ? `Earlier my scheduled task gave me this:\n\n"${excerpt}"\n\nGo deeper — add detail, context, and anything new since.`
      : `that scheduled update from ${when || 'earlier'} — go deeper and tell me more.`;
  } else if (type === 'curator_note' || _hasUrlRef(note || {})) {
    const ref = note ? _findUrlRef(note) : null;
    const title = (ref && ref.title) ? `"${ref.title}"` : 'that link';
    seed = `${title} you sent me — what made you think of it?`;
  } else if (type === 'wondering') {
    seed = when
      ? `that thread you were wondering about ${when} — let's pick it up.`
      : `that thread you were wondering about — let's pick it up.`;
  } else {
    seed = when
      ? `that note you left ${when} — tell me what you were noticing.`
      : `that note you left — tell me what you were noticing.`;
  }
  const composer = document.querySelector('#chat-input, .chat-composer textarea');
  if (composer) {
    composer.value = seed;
    composer.focus();
    composer.dispatchEvent(new Event('input', { bubbles: true }));
  }
  await _postAction(noteId, 'surfaced');
  cardEl.classList.add('dismissing');
  setTimeout(() => _refreshNow(), 200);
  _invalidateHistory();
  _closeDrawer();
}

// Apply the result of an awaited note action where dismissing the card IS
// the visible effect. On success the card animates out and the list
// refreshes; on FAILURE the card stays put and we surface a toast — the
// old code animated the card away unconditionally, so an offline/500 POST
// silently lost the signal AND lied to the user that it had landed (audit
// 2026-06-17). Re-enables any buttons disabled during the in-flight POST.
function _settleCardAction(ok, cardEl, failMsg) {
  if (ok) {
    cardEl.classList.add('dismissing');
    setTimeout(() => _refreshNow(), 200);
    _invalidateHistory();
    return true;
  }
  cardEl.classList.remove('dismissing', 'is-busy');
  cardEl.querySelectorAll('button[disabled]').forEach((b) => { b.disabled = false; });
  showToast(failMsg || "Couldn't save that just now — still here, try again.", 'error');
  return false;
}

async function _handleAcknowledge(noteId, cardEl) {
  const ok = await _postAction(noteId, 'acknowledged');
  _settleCardAction(ok, cardEl);
}

// Soft "no" — same backend semantics as acknowledged (note moves to
// history) but the verb is "not for me / not quite / drop it" which
// signals Becca to weight the topic down without nuking it from orbit.
async function _handleDismiss(noteId, cardEl) {
  // Was _postAction(.,'acknowledged') — which recorded the POSITIVE +0.2
  // signal, so "not for me" BOOSTED the topic (audit 2026-06-17). Route
  // the real negative signal via /feedback {kind:dismiss} → "dismissed".
  const ok = await _postFeedback(noteId, 'dismiss');
  _settleCardAction(ok, cardEl);
}

// Hard "no on this topic for a while" — explicit 90-day mute. Curator
// honors the mute via topic-mutes filtering. Only present on
// curator-pick cards.
async function _handleMute(noteId, cardEl) {
  if (!window.confirm("Stop surfacing this topic for ~90 days?")) return;
  const ok = await _postAction(noteId, 'muted_topic');
  _settleCardAction(ok, cardEl, "Couldn't mute that just now — try again.");
}

// "Save for later" — wondering cards only. Acknowledges (clears from
// the active drawer) so the user can scan it later from history.
async function _handleSaveLater(noteId, cardEl) {
  const ok = await _postAction(noteId, 'acknowledged');
  _settleCardAction(ok, cardEl);
}

// Clicking the title link or "Open ↗" also surfaces the note so it
// stops re-appearing as unread after the user has clearly engaged.
// The link itself opens in a new tab (target="_blank") via the anchor's
// own behavior; this handler only posts the surfaced signal.
async function _handleOpenLink(noteId, cardEl) {
  await _postAction(noteId, 'surfaced');
  cardEl.classList.add('dismissing');
  setTimeout(() => _refreshNow(), 200);
  _invalidateHistory();
}

// Note actions move a note from active → history. Refresh now if the
// history panel is open; otherwise just drop the cached flag so the
// next open re-fetches instead of showing a stale list.
function _invalidateHistory() {
  if (_historyOpen) {
    void _refreshHistory();
  } else {
    _historyFetched = false;
  }
}

function _firstSentence(text) {
  if (!text) return '';
  const m = String(text).match(/^[^.!?\n]+[.!?]?/);
  return (m ? m[0] : text).slice(0, 200);
}

const COACHMARK_DISMISSED_KEY = 'companion.notes.coachmark.dismissed.v1';

function _shouldShowCoachmark() {
  try {
    return localStorage.getItem(COACHMARK_DISMISSED_KEY) !== '1';
  } catch (_) {
    return false;
  }
}

function _dismissCoachmark() {
  const coach = document.querySelector('.companion-note-coachmark');
  if (coach) coach.hidden = true;
  try { localStorage.setItem(COACHMARK_DISMISSED_KEY, '1'); } catch (_) { /* ignore */ }
}

function _openDrawer() {
  if (_presenceMode() === 'silent') return;
  _isOpen = true;
  const drawer = document.querySelector('#companion-note-drawer');
  if (drawer) {
    drawer.classList.remove('hidden');
    drawer.classList.add('open');
    // First-open coachmark: explains the mirror metaphor + points at
    // the gear. Hidden forever after one dismiss; gated by localStorage
    // so opening on a different surface doesn't re-trigger it.
    const coach = drawer.querySelector('.companion-note-coachmark');
    if (coach) coach.hidden = !_shouldShowCoachmark();
  }
  // Per-briefing read-aloud: best-effort auto-start off this open gesture.
  _maybeAutoReadAloud();
}

function _closeDrawer() {
  _isOpen = false;
  const drawer = document.querySelector('#companion-note-drawer');
  if (drawer) {
    drawer.classList.add('hidden');
    drawer.classList.remove('open');
  }
  // Reset the slide-over so reopening the drawer starts on active
  // notes, not on whatever history page the user was looking at last.
  _closeHistoryPanel();
}

// ── Slide-over history panel ────────────────────────────────────────

let _historyOpen = false;
let _historyFetched = false;

function _openHistoryPanel() {
  const drawer = document.querySelector('#companion-note-drawer');
  if (!drawer) return;
  _historyOpen = true;
  drawer.classList.add('history-open');
  const panel = drawer.querySelector('.companion-note-history');
  if (panel) panel.setAttribute('aria-hidden', 'false');
  const btn = drawer.querySelector('.companion-note-drawer-history');
  if (btn) btn.setAttribute('aria-pressed', 'true');
  // Lazy fetch — only the first open hits the network.
  if (!_historyFetched) {
    _historyFetched = true;
    void _refreshHistory();
  }
}

function _closeHistoryPanel() {
  const drawer = document.querySelector('#companion-note-drawer');
  if (!drawer) return;
  _historyOpen = false;
  drawer.classList.remove('history-open');
  const panel = drawer.querySelector('.companion-note-history');
  if (panel) panel.setAttribute('aria-hidden', 'true');
  const btn = drawer.querySelector('.companion-note-drawer-history');
  if (btn) btn.setAttribute('aria-pressed', 'false');
}

async function _refreshHistory() {
  const body = document.querySelector('.companion-note-history-body');
  if (!body) return;
  body.dataset.state = 'loading';
  body.innerHTML = `<div class="companion-note-history-empty">loading…</div>`;
  let notes = [];
  try {
    const resp = await fetch('/api/companion/notes/history?limit=50', {
      credentials: 'same-origin',
    });
    if (resp.ok) {
      const data = await resp.json();
      notes = Array.isArray(data.notes) ? data.notes : [];
    }
  } catch (_) { /* fall through to empty */ }
  body.dataset.state = notes.length ? 'ready' : 'empty';
  if (!notes.length) {
    body.innerHTML = `<div class="companion-note-history-empty">Nothing here yet.</div>`;
    return;
  }
  body.innerHTML = notes.map(_historyCardHtml).join('\n');
}

// Per-type history rendering — same dispatch shape as the active drawer
// but at lower density (no actions, smaller type, faded). Re-opening
// a curator pick from history still works: the title is a clickable
// link to the original URL.
function _historyCardHtml(note) {
  const type = (note.entry_type || '').toLowerCase();
  // Task results carry source refs but are a delivered answer, not a
  // single-link curator pick — keep them on the observation layout.
  if (type === 'standing_task') return _observationHistoryCardHtml(note);
  if (type === 'curator_note' || _hasUrlRef(note)) return _pickHistoryCardHtml(note);
  if (type === 'wondering') return _wonderingHistoryCardHtml(note);
  return _observationHistoryCardHtml(note);
}

function _historyMetaFooter(note) {
  const when = _escapeHtml(_historyRelativeWhen(note.surfaced_at));
  const outcome = note.outcome === 'muted' ? 'muted' : 'seen';
  return `
    <div class="companion-note-history-card-meta">
      <span class="companion-note-history-outcome">${outcome}</span>
      <span class="companion-note-history-when">${when}</span>
    </div>
  `;
}

function _pickHistoryCardHtml(note) {
  const ref = _findUrlRef(note) || {};
  const parsed = _splitComposedBody(note.content);
  const url = ref.url || '';
  const title = (ref.title || parsed.title || '').trim();
  const domain = (ref.domain || _domainFromUrl(url) || '').trim();
  const affect = _escapeHtml(note.affect_tag || '');
  const outcome = note.outcome === 'muted' ? 'muted' : 'seen';
  const safeUrl = _escapeHtml(_safeHref(url));
  const safeTitle = _escapeHtml(title || domain || '(no title)');
  const safeDomain = _escapeHtml(domain);

  const titleHtml = url
    ? `<a class="companion-history-pick-title" href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeTitle}</a>`
    : `<span class="companion-history-pick-title">${safeTitle}</span>`;

  return `
    <article class="companion-note-history-card companion-history-pick" data-affect="${affect}" data-outcome="${outcome}" data-type="pick">
      <div class="companion-history-pick-row">
        ${safeDomain ? `<span class="companion-history-pick-domain">${safeDomain}</span>` : ''}
        ${titleHtml}
      </div>
      ${_historyMetaFooter(note)}
    </article>
  `;
}

function _wonderingHistoryCardHtml(note) {
  const affect = _escapeHtml(note.affect_tag || '');
  const outcome = note.outcome === 'muted' ? 'muted' : 'seen';
  const body = _escapeHtml(_truncateSentenceAware(note.content || '', 220));
  return `
    <article class="companion-note-history-card companion-history-wondering" data-affect="${affect}" data-outcome="${outcome}" data-type="wondering">
      <div class="companion-note-history-card-body">${body}</div>
      ${_historyMetaFooter(note)}
    </article>
  `;
}

function _observationHistoryCardHtml(note) {
  const affect = _escapeHtml(note.affect_tag || '');
  const outcome = note.outcome === 'muted' ? 'muted' : 'seen';
  const body = _escapeHtml(_truncateSentenceAware(note.content || '', 220));
  const imgRef = (note.content_refs || []).find(
    (r) => r && r.kind === 'image' && r.url,
  );
  const heroHtml = imgRef
    ? `<img class="companion-note-hero companion-note-hero-sm" src="${_escapeHtml(_safeHref(imgRef.url))}" alt="" loading="lazy" decoding="async" />`
    : '';
  return `
    <article class="companion-note-history-card companion-history-observation" data-affect="${affect}" data-outcome="${outcome}" data-type="observation">
      ${heroHtml}
      <div class="companion-note-history-card-body">${body}</div>
      ${_historyMetaFooter(note)}
    </article>
  `;
}

function _historyRelativeWhen(isoStr) {
  if (!isoStr) return '';
  // SQLite datetime() returns 'YYYY-MM-DD HH:MM:SS' in UTC; normalize
  // to an ISO string the browser can parse.
  const norm = isoStr.includes('T') ? isoStr : isoStr.replace(' ', 'T') + 'Z';
  const t = Date.parse(norm);
  if (!Number.isFinite(t)) return '';
  const secs = Math.max(0, (Date.now() - t) / 1000);
  if (secs < 60) return 'just now';
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(t).toLocaleDateString();
}

function _updateDot(notes, today) {
  const dot = document.querySelector('#companion-note-dot');
  if (!dot) return;
  const mode = _presenceMode();
  if (mode === 'silent') {
    dot.classList.add('hidden');
    return;
  }
  // Dot shows when there are notes OR a fresh unread Today reflection.
  const hasNotes = notes.length > 0;
  const hasFreshToday = !!(today && today.today && !today.today.quarantined);
  if (!hasNotes && !hasFreshToday) {
    dot.classList.add('hidden');
  } else {
    dot.classList.remove('hidden');
    // Affect-tinted accent. Notes win if any; otherwise neutral.
    const affect = hasNotes
      ? (notes[0].affect_tag || '').toLowerCase()
      : '';
    dot.setAttribute('data-affect', affect);
  }
}

async function _refreshNow() {
  if (!_mounted) return;
  const [notes, today] = await Promise.all([
    _fetchNotes(),
    _fetchToday(),
  ]);
  // Only re-render notes if the set changed (by id list)
  const sig = notes.map((n) => n.id).join(',');
  const prevSig = _lastNotes.map((n) => n.id).join(',');
  _lastNotes = notes;
  if (sig !== prevSig) {
    _renderCards(notes);
  }
  _renderToday(today);
  _updateDot(notes, today);
}

// ── Today entry ─────────────────────────────────────────────────────

let _lastToday = null;  // cached for the action-popover handlers

async function _fetchToday() {
  try {
    const resp = await fetch('/api/companion/today', { credentials: 'same-origin' });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) { return null; }
}

function _renderToday(data) {
  const zone = document.querySelector('.companion-today-zone');
  if (!zone) return;
  _lastToday = data;
  const proseEl = zone.querySelector('.companion-today-prose');
  const metaEl = zone.querySelector('.companion-today-meta');
  if (!proseEl || !metaEl) return;

  // Silent presence: explicit hint, not blank.
  if (data && data.presence_mode === 'silent') {
    zone.dataset.state = 'silent';
    proseEl.innerHTML = `<p class="companion-today-empty">Presence mode is silent. No reflection is being generated.</p>`;
    metaEl.textContent = '';
    return;
  }
  // Not yet generated.
  if (!data || !data.today) {
    zone.dataset.state = 'pending';
    const hint = (data && data.hint) || 'Not yet written. Comes back later in the day.';
    proseEl.innerHTML = `<p class="companion-today-empty">${_escapeHtml(hint)}</p>`;
    metaEl.textContent = '';
    return;
  }
  zone.dataset.state = data.today.quarantined ? 'quarantined' : 'ready';
  proseEl.innerHTML = _renderReflectionProse(data.today);
  metaEl.textContent = _formatUpdated(data.today.last_updated_at);
  _bindCitationActions(zone);
}

function _renderReflectionProse(today) {
  const text = String(today.content || '');
  if (!text.trim()) {
    return `<p class="companion-today-empty">Stayed in the background today.</p>`;
  }
  const realRefs = new Set(
    (today.source_refs || [])
      .filter((r) => r && r.kind && Number.isInteger(r.id))
      .map((r) => `${r.kind}:${r.id}`),
  );
  // Rewrite [kind:N] citations to clickable spans. Escape the rest.
  // Split-and-reassemble preserves prose around citations.
  const parts = [];
  const re = /\[(note|wondering|journal):(\d+)\]/g;
  let cursor = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > cursor) {
      parts.push(_escapeHtml(text.slice(cursor, m.index)));
    }
    const kind = m[1];
    const id = m[2];
    const key = `${kind}:${id}`;
    if (realRefs.has(key)) {
      parts.push(
        `<span class="companion-today-citation" tabindex="0" `
        + `role="button" data-kind="${_escapeHtml(kind)}" `
        + `data-id="${_escapeHtml(id)}">${_escapeHtml(`[${kind}:${id}]`)}</span>`,
      );
    } else {
      // Hallucinated id — strip the citation markup, keep nothing
      // (the validator should have caught these, but defense in depth).
    }
    cursor = m.index + m[0].length;
  }
  if (cursor < text.length) parts.push(_escapeHtml(text.slice(cursor)));
  return `<p>${parts.join('')}</p>`;
}

function _formatUpdated(isoStr) {
  if (!isoStr) return '';
  const t = Date.parse(isoStr.includes('T') ? isoStr : isoStr.replace(' ', 'T') + 'Z');
  if (!Number.isFinite(t)) return '';
  const secs = Math.max(0, (Date.now() - t) / 1000);
  if (secs < 60) return 'Updated just now';
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `Updated ${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `Updated ${hrs}h ago`;
  return `Updated ${Math.floor(hrs / 24)}d ago`;
}

// ── Citation action popover (Ask / Mute / Forget) ───────────────────

function _bindCitationActions(zone) {
  zone.querySelectorAll('.companion-today-citation').forEach((el) => {
    if (el._bound) return;
    el._bound = true;
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      _showCitationPopover(el);
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        _showCitationPopover(el);
      }
    });
  });
  const archiveLink = zone.querySelector('.companion-today-archive-link');
  if (archiveLink && !archiveLink._bound) {
    archiveLink._bound = true;
    archiveLink.addEventListener('click', (e) => {
      e.preventDefault();
      _openArchive();
    });
  }
}

let _activePopover = null;

function _closePopover() {
  if (_activePopover) {
    try { _activePopover.remove(); } catch (_) {}
    _activePopover = null;
  }
}

function _showCitationPopover(anchorEl) {
  _closePopover();
  const kind = anchorEl.dataset.kind;
  const id = parseInt(anchorEl.dataset.id, 10);
  if (!kind || !Number.isInteger(id)) return;

  const pop = document.createElement('div');
  pop.className = 'companion-today-popover';
  pop.innerHTML = `
    <button type="button" data-pop-action="ask">Ask about it</button>
    <button type="button" data-pop-action="mute">Mute this topic</button>
    <button type="button" data-pop-action="forget">Forget</button>
  `;
  const rect = anchorEl.getBoundingClientRect();
  pop.style.position = 'fixed';
  pop.style.left = `${Math.max(8, rect.left)}px`;
  pop.style.top = `${rect.bottom + 4}px`;
  document.body.appendChild(pop);
  _activePopover = pop;

  pop.querySelector('[data-pop-action="ask"]').addEventListener('click', async () => {
    _closePopover();
    await _askAboutCitation(kind, id);
  });
  pop.querySelector('[data-pop-action="mute"]').addEventListener('click', async () => {
    _closePopover();
    if (!window.confirm('Mute this topic for ~90 days?')) return;
    // Reuse the existing note-muted endpoint when kind=note; journals
    // don't have a per-id mute endpoint yet, so use forget for them.
    if (kind === 'note') {
      await _postAction(id, 'muted_topic');
    } else {
      await _postForget([{ kind, id }]);
    }
    await _refreshNow();
  });
  pop.querySelector('[data-pop-action="forget"]').addEventListener('click', async () => {
    _closePopover();
    if (!window.confirm("Forget this from your companion's interior? It won't reappear in future reflections.")) return;
    await _postForget([{ kind, id }]);
    await _refreshNow();
  });

  // Dismiss on outside click / escape.
  setTimeout(() => {
    const off = (e) => {
      if (!pop.contains(e.target)) {
        document.removeEventListener('click', off);
        _closePopover();
      }
    };
    document.addEventListener('click', off);
  }, 0);
}

async function _askAboutCitation(kind, id) {
  // Seed the chat composer with a prompt that references the entry.
  // Mirrors the "Pull it together" pattern on note cards.
  const composer = document.querySelector('#chat-input, .chat-composer textarea');
  if (!composer) return;
  let seed = '';
  const today = _lastToday && _lastToday.today;
  if (today && today.content) {
    seed = `tell me more about what you wrote in today's reflection (${kind}:${id}).`;
  } else {
    seed = `tell me more about ${kind}:${id}.`;
  }
  composer.value = seed;
  composer.focus();
  composer.dispatchEvent(new Event('input', { bubbles: true }));
  _closeDrawer();
}

async function _postForget(refs) {
  try {
    const resp = await fetch('/api/companion/today/forget', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ refs }),
    });
    return resp.ok;
  } catch (_) { return false; }
}

function _openArchive() {
  // Lazy-load the archive modal to keep the drawer's footprint small.
  import('./companion-today-archive.js').then((mod) => {
    if (mod && typeof mod.open === 'function') mod.open();
  }).catch((e) => {
    console.warn('[companion-today] archive open failed', e);
  });
}

function _openTopicsModal() {
  import('./companion-topics-modal.js').then((mod) => {
    if (mod && typeof mod.open === 'function') mod.open();
  }).catch((e) => {
    console.warn('[companion-topics] modal open failed', e);
  });
}

function _schedulePoll() {
  if (_pollTimer) clearTimeout(_pollTimer);
  const interval = document.hidden ? HIDDEN_POLL_INTERVAL_MS : POLL_INTERVAL_MS;
  _pollTimer = setTimeout(async () => {
    await _refreshNow();
    _schedulePoll();
  }, interval);
}

async function mount() {
  if (_mounted) return;
  _mounted = true;
  // Per-mount lifetime — every listener / observer / timer that lives
  // on document or window is registered here so unmount() can clean
  // up symmetrically. Mount/unmount cycles previously leaked the
  // outside-click + visibility + presence-mode handlers each time the
  // bootstrap reconciler ran (page load, tab refocus, post-save, …).
  _lifetime = createLifetime();
  _buildShell();
  _registerScheduleCommand();
  // Personalize the drawer title with the companion's display_name
  // when the user has set one. Defaults to "Notes" (neutral chrome);
  // becomes "{Name}'s notes" once a non-default name is in place.
  // Fire-and-forget — if /status is slow or offline, the default
  // title already renders.
  _refreshDrawerTitleFromIdentity().catch(() => {});
  await _refreshNow();
  _schedulePoll();
  _lifetime.addEventListener(document, 'visibilitychange', _onVisibilityChange);
  // Bootstrap dispatches this whenever /api/config/tools reports a
  // new presence_mode — we re-evaluate the dot immediately rather
  // than waiting up to 60s for the next poll to surface the change.
  _lifetime.addEventListener(window, 'companion:presence-mode-changed', _onPresenceModeChanged);
}

// Surface the schedule (topics + standing tasks) as a top-level entry
// so it's reachable from the command palette (Ctrl/Cmd+K) and the
// companion's app menu — not only via the ⚙ inside this drawer.
let _scheduleCommandRegistered = false;
function _registerScheduleCommand() {
  if (_scheduleCommandRegistered) return;
  _scheduleCommandRegistered = true;
  import('./command-palette.js').then(({ registerCommand }) => {
    // The calendar is the top-level Schedule surface now — events, your
    // synced device calendar, and companion tasks on one grid.
    registerCommand({
      id: 'calendar.open',
      label: 'Open Calendar',
      hint: 'Events, appointments & briefings',
      group: 'Companion',
      keywords: 'calendar schedule event appointment agenda month week day briefing reminder caldav sync',
      agent: {
        description: 'Open the calendar — events, appointments, briefings, and watches',
        speak: 'Opening your calendar.',
      },
      run: () => import('./calendar/index.js').then((m) => m.open?.()).catch(() => {}),
    });
    // The companion task manager stays directly reachable for creating new
    // briefings / watches / scheduled requests (the calendar edits existing
    // ones; this is where new ones are authored).
    registerCommand({
      id: 'companion.tasks.open',
      label: 'Open Companion Tasks',
      hint: 'Briefings, reminders & watches',
      group: 'Companion',
      keywords: 'companion task briefing reminder watch standing cron daily alarm notify digest',
      agent: {
        description: 'Open the companion task manager — briefings, reminders, and watches',
        speak: 'Opening your companion tasks.',
      },
      run: () => _openTopicsModal(),
    });
  }).catch(() => {});
}

async function _refreshDrawerTitleFromIdentity() {
  let identity = null;
  try {
    const resp = await fetch('/api/companion/status', { credentials: 'same-origin' });
    if (!resp.ok) return;
    const data = await resp.json();
    identity = data && data.identity;
  } catch (_) {
    return;
  }
  if (!identity || identity.is_default_name) return;
  const name = (identity.display_name || '').trim();
  if (!name) return;
  const titleEl = document.querySelector('#companion-note-drawer .companion-note-drawer-title');
  if (titleEl) titleEl.textContent = `${name}'s notes`;
  const dotEl = document.querySelector('#companion-note-dot');
  if (dotEl) dotEl.setAttribute('aria-label', `${name} has notes`);
}

function unmount() {
  _mounted = false;
  if (_pollTimer) {
    clearTimeout(_pollTimer);
    _pollTimer = null;
  }
  // Reset cross-render state flags so a fresh mount() doesn't inherit
  // ghosts (e.g., _historyOpen=true causing the next mount to flash
  // the slide-over before the user clicks anything).
  _isOpen = false;
  _historyOpen = false;
  _historyFetched = false;
  _closePopover();
  _lastNotes = [];
  _lastToday = null;
  // Drop every lifetime-tracked listener / observer in one call.
  if (_lifetime) {
    try { _lifetime.dispose(); } catch (_) { /* per-step errors already logged */ }
    _lifetime = null;
  }
  document.querySelector('#companion-note-dot')?.remove();
  document.querySelector('#companion-note-drawer')?.remove();
}

export const CompanionNotes = { mount, unmount, refresh: _refreshNow };

// Convenience: auto-expose on window so chat.js / app bootstrap can
// call mount() without an explicit import in the legacy bundle path.
if (typeof window !== 'undefined') {
  window.CompanionNotes = CompanionNotes;
}
