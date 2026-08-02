// Cardsmith — co-design pipeline for new character cards.
//
// Public surface:
//   openCardsmithLauncher({ onCardSaved, onBlankRequested })
//
// Two visual surfaces:
//   1. Launcher modal — type radio (Single/Ensemble/World-RPG) + 3 lanes
//      (Describe / Wiki / Blank). Mirrors .persona-modal styling.
//   2. Conversation modal — split layout: chat thread on the left, live card
//      preview on the right. Streams /api/characters/cardsmith/turn (SSE).
//
// Phase 1 ships Single + AI-Describe + Blank. Wiki, Ensemble, World/RPG land
// in subsequent phases (visible-but-disabled in the launcher today).

import { escapeHtml, showToast } from '../app.js';
import { renderMarkdown } from '../chat/markdown.js';
import { rafCoalesce } from '../raf-coalesce.js';

let _activeOverlay = null;
// Track the element that had focus when the modal opened so we can restore
// focus on close (a11y best practice — keeps keyboard users oriented).
let _focusReturnEl = null;

// ─── Resume token (localStorage) ───────────────────────────────────────────
//
// The server persists in-flight cardsmith sessions to disk (cardsmith_sessions
// table). This client-side token lets the launcher find that disk row again
// after a browser refresh — without it, state.sessionId is in JS memory only
// and a reload orphans the persisted session.
//
// Single-session model: last write wins. If a user starts two cards back-to-
// back without finishing either, only the most recent one shows in the resume
// banner. The older session still survives server-side until TTL evicts it
// (4h), but the user has to start a new session to talk to it. Acceptable
// trade-off for a much simpler resume UX than a session-picker list.

const RESUME_KEY = 'cardsmith.resume';

function _saveResumeToken({ sessionId, cardType, source }) {
  try {
    localStorage.setItem(RESUME_KEY, JSON.stringify({
      sessionId, cardType, source, startedAt: Date.now(),
    }));
  } catch { /* quota or private mode — non-fatal */ }
}

function _loadResumeToken() {
  try {
    const raw = localStorage.getItem(RESUME_KEY);
    if (!raw) return null;
    const t = JSON.parse(raw);
    if (!t?.sessionId) return null;
    return t;
  } catch { return null; }
}

function _clearResumeToken() {
  try { localStorage.removeItem(RESUME_KEY); } catch { /* */ }
}

async function _fetchResumePreview(sessionId) {
  if (!sessionId) return null;
  try {
    const resp = await fetch(`/api/characters/cardsmith/session/${encodeURIComponent(sessionId)}`);
    if (resp.status === 404) {
      // Server has no such session — token is stale (TTL elapsed, server
      // dropped, or different user). Clean up so the banner doesn't keep
      // re-asking on every launcher open.
      _clearResumeToken();
      return null;
    }
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

// ─── Public entry ──────────────────────────────────────────────────────────

export function openCardsmithLauncher(opts) {
  _closeAny();
  _focusReturnEl = (typeof document !== 'undefined') ? document.activeElement : null;
  const overlay = _buildLauncher(opts || {});
  document.body.appendChild(overlay);
  _activeOverlay = overlay;
  _attachFocusTrap(overlay);
  setTimeout(() => {
    const seed = overlay.querySelector('.cs-seed-input');
    if (seed) seed.focus();
  }, 60);
  // Async: fetch all in-progress drafts for this user and inject the
  // drafts list section. Doesn't block the launcher render — section pops
  // in if/when the fetch completes and the user is still on the launcher.
  _maybeShowDraftsList(overlay, opts);
}

async function _maybeShowDraftsList(overlay, opts) {
  // Fetch every in-progress draft for this user from the server. The
  // localStorage resume token only tracks the most recent — switching to
  // the server-backed list means a user with several drafts (e.g. one
  // ensemble in progress plus a side character started later) sees all
  // of them, not just the last one they touched.
  let drafts = [];
  try {
    const resp = await fetch('/api/characters/cardsmith/sessions');
    if (!resp.ok) return;
    const data = await resp.json();
    drafts = Array.isArray(data.sessions) ? data.sessions : [];
  } catch {
    // Server unavailable / offline — silent. Launcher renders without
    // the drafts section, same as if the user has no drafts.
    return;
  }
  if (!drafts.length) {
    _clearResumeToken();
    return;
  }
  if (overlay !== _activeOverlay || !overlay.isConnected) return;
  // Highlight the most recent (per localStorage token) when present.
  const token = _loadResumeToken();
  const recentId = token?.sessionId || drafts[0].session_id;
  _injectDraftsList(overlay, opts, drafts, recentId);
}

function _injectDraftsList(overlay, opts, drafts, recentId) {
  const body = overlay.querySelector('.cardsmith-launcher-body');
  if (!body) return;
  const section = document.createElement('div');
  section.className = 'cs-drafts-section';
  const typeLabel = (t) => ({
    single: 'Single',
    ensemble: 'Ensemble',
    world_rpg: 'World / RPG',
  })[t] || 'Card';
  const rows = drafts.slice(0, 5).map(d => {
    const updated = d.last_active_at ? new Date(d.last_active_at * 1000) : null;
    const when = updated ? _relativeTimeString(updated) : 'just now';
    const isRecent = d.session_id === recentId;
    const universeBadge = d.has_universe
      ? '<span class="cs-draft-badge">wiki</span>'
      : '';
    return `
      <div class="cs-draft-row${isRecent ? ' cs-draft-row-recent' : ''}"
           data-session-id="${escapeHtml(d.session_id)}"
           data-card-type="${escapeHtml(d.card_type)}">
        <div class="cs-draft-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">
            <path d="M12 8v4l3 2"/><circle cx="12" cy="12" r="10"/>
          </svg>
        </div>
        <div class="cs-draft-body">
          <div class="cs-draft-label">${escapeHtml(d.friendly_label || 'Draft')}</div>
          <div class="cs-draft-meta">${escapeHtml(typeLabel(d.card_type))} · ${escapeHtml(when)} ${universeBadge}</div>
        </div>
        <div class="cs-draft-actions">
          <button class="btn btn-sm btn-primary cs-draft-resume" aria-label="Resume draft">Resume</button>
          <button class="icon-btn small cs-draft-discard" title="Discard this draft" aria-label="Discard draft">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>
    `;
  }).join('');
  const moreNote = drafts.length > 5
    ? `<div class="cs-drafts-more">+${drafts.length - 5} older draft${drafts.length - 5 === 1 ? '' : 's'}</div>`
    : '';
  section.innerHTML = `
    <div class="cs-drafts-header">
      <span class="cs-drafts-title">In-progress drafts</span>
      <span class="cs-drafts-count">${drafts.length}</span>
    </div>
    <div class="cs-drafts-list">${rows}</div>
    ${moreNote}
  `;
  body.insertBefore(section, body.firstChild);

  section.querySelectorAll('.cs-draft-row').forEach(rowEl => {
    const sid = rowEl.dataset.sessionId;
    const cardType = rowEl.dataset.cardType;
    rowEl.querySelector('.cs-draft-resume')?.addEventListener('click', async () => {
      const preview = await _fetchResumePreview(sid);
      if (!preview) {
        showToast('That draft has expired', 'info');
        rowEl.remove();
        return;
      }
      _openConversation(opts, {
        sessionId: preview.session_id,
        cardType: preview.card_type || cardType,
        seedPrompt: '',
        resumeFrom: preview,
      });
    });
    rowEl.querySelector('.cs-draft-discard')?.addEventListener('click', () => {
      void _discardResumeSession(sid);
      rowEl.remove();
      // If we removed the last visible row, drop the whole section so the
      // empty header doesn't linger.
      if (!section.querySelector('.cs-draft-row')) {
        section.remove();
      }
    });
  });
}

async function _discardResumeSession(sessionId) {
  _clearResumeToken();
  try {
    await fetch('/api/characters/cardsmith/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch { /* server cleanup is best-effort */ }
}

function _relativeTimeString(date) {
  const diffS = (Date.now() - date.getTime()) / 1000;
  if (diffS < 60) return 'just now';
  if (diffS < 3600) return `${Math.floor(diffS / 60)}m ago`;
  if (diffS < 86400) return `${Math.floor(diffS / 3600)}h ago`;
  return `${Math.floor(diffS / 86400)}d ago`;
}

function _closeAny() {
  if (_activeOverlay && _activeOverlay.parentNode) {
    _activeOverlay.classList.add('cs-closing');
    const node = _activeOverlay;
    setTimeout(() => node.remove(), 180);
  }
  _activeOverlay = null;
  // Restore focus to whatever the user was on before opening the modal.
  if (_focusReturnEl && typeof _focusReturnEl.focus === 'function') {
    try { _focusReturnEl.focus(); } catch { /* element may have been removed */ }
  }
  _focusReturnEl = null;
}

// ─── Focus trap ────────────────────────────────────────────────────────────

const _FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

function _attachFocusTrap(overlay) {
  overlay.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;
    const focusables = Array.from(overlay.querySelectorAll(_FOCUSABLE_SELECTOR))
      .filter(el => el.offsetParent !== null);  // visible only
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && active === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  });
}

// ─── Launcher modal ────────────────────────────────────────────────────────

function _buildLauncher(opts) {
  const overlay = document.createElement('div');
  overlay.className = 'persona-modal-overlay cardsmith-launcher-overlay';
  overlay.innerHTML = `
    <div class="persona-modal cardsmith-launcher" role="dialog" aria-modal="true" aria-labelledby="cs-launcher-title">
      <div class="persona-modal-header">
        <span class="persona-modal-title" id="cs-launcher-title">Create a Character</span>
        <button class="icon-btn small cs-close-btn" title="Close" aria-label="Close character creation">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>

      <div class="persona-modal-body cardsmith-launcher-body">

        <div class="cs-type-row">
          <span class="field-label cs-type-label">Card type</span>
          <div class="cs-type-pills">
            <label class="cs-type-pill cs-type-active">
              <input type="radio" name="cs-type" value="single" checked>
              <span class="cs-type-pill-text">Single character</span>
            </label>
            <label class="cs-type-pill">
              <input type="radio" name="cs-type" value="ensemble">
              <span class="cs-type-pill-text">Ensemble</span>
            </label>
            <label class="cs-type-pill">
              <input type="radio" name="cs-type" value="world_rpg">
              <span class="cs-type-pill-text">World / RPG</span>
            </label>
          </div>
        </div>

        <div class="cs-lanes">

          <div class="cs-lane cs-lane-describe">
            <div class="cs-lane-header">
              <div class="cs-lane-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2l1.6 4.6L18 8l-4.4 1.4L12 14l-1.6-4.6L6 8l4.4-1.4L12 2z"/>
                  <path d="M5 17l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7L5 17z"/>
                  <path d="M19 14l.5 1.5 1.5.5-1.5.5L19 18l-.5-1.5-1.5-.5 1.5-.5L19 14z"/>
                </svg>
              </div>
              <span class="cs-lane-title">Describe with AI</span>
            </div>
            <div class="cs-lane-body">
              <textarea class="field-textarea cs-seed-input"
                        rows="3"
                        placeholder="A reclusive cyberpunk medic hiding from her old crew. Soft for strays. Deadly with a scalpel."></textarea>
              <div class="cs-style-row">
                <label class="field-label cs-style-label">Style hint</label>
                <select class="field-input cs-style-select">
                  <option value="">No preset</option>
                  <option value="anime">Anime / Manga</option>
                  <option value="painterly">Painterly / Concept</option>
                  <option value="photorealistic">Photorealistic</option>
                  <option value="watercolor">Watercolor</option>
                  <option value="pixel">Pixel Art</option>
                  <option value="comic">Comic Book</option>
                  <option value="dark">Dark / Gothic</option>
                  <option value="fantasy">High Fantasy</option>
                  <option value="scifi">Sci-Fi / Cyberpunk</option>
                  <option value="ukiyoe">Ukiyo-e</option>
                  <option value="noir">Film Noir</option>
                  <option value="cozy">Cozy / Slice of Life</option>
                </select>
              </div>
            </div>
            <div class="cs-lane-footer">
              <button class="btn btn-primary btn-sm cs-describe-go-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13">
                  <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
                Begin
              </button>
            </div>
          </div>

          <div class="cs-lane cs-lane-wiki">
            <div class="cs-lane-header">
              <div class="cs-lane-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                </svg>
              </div>
              <span class="cs-lane-title">From a Wiki</span>
            </div>
            <div class="cs-lane-body">
              <p class="cs-lane-blurb">Paste a Fandom or Wikipedia URL and the Cardsmith pulls canonical traits, then asks how you want to twist them.</p>
              <input type="url"
                     class="field-input cs-wiki-url-input"
                     placeholder="https://naruto.fandom.com/wiki/Sasuke_Uchiha"
                     aria-label="Wiki URL">
              <div class="cs-wiki-error" role="alert" hidden></div>
            </div>
            <div class="cs-lane-footer">
              <button class="btn btn-primary btn-sm cs-wiki-preview-btn" disabled>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13" aria-hidden="true">
                  <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                Preview
              </button>
            </div>
          </div>

          <div class="cs-lane cs-lane-blank">
            <div class="cs-lane-header">
              <div class="cs-lane-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
              </div>
              <span class="cs-lane-title">Start Blank</span>
            </div>
            <div class="cs-lane-body">
              <p class="cs-lane-blurb">Skip the Cardsmith and open a fresh editor. Best when you already know exactly what you want.</p>
            </div>
            <div class="cs-lane-footer">
              <button class="btn btn-sm cs-blank-go-btn">Open editor</button>
            </div>
          </div>

        </div>
      </div>
    </div>
  `;

  // Wiring
  overlay.querySelector('.cs-close-btn')?.addEventListener('click', () => _closeAny());
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) _closeAny();
  });
  document.addEventListener('keydown', _escHandlerOnce, { once: true });

  // Type pill toggling (visual only — only Single is enabled in Phase 1)
  overlay.querySelectorAll('.cs-type-pill input[type="radio"]').forEach(r => {
    r.addEventListener('change', () => {
      overlay.querySelectorAll('.cs-type-pill').forEach(p => p.classList.remove('cs-type-active'));
      const label = r.closest('.cs-type-pill');
      if (label) label.classList.add('cs-type-active');
    });
  });

  // Describe lane: ⌘+Enter / Ctrl+Enter to submit
  const seed = overlay.querySelector('.cs-seed-input');
  if (seed) {
    seed.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        overlay.querySelector('.cs-describe-go-btn')?.click();
      }
    });
  }

  // Describe go
  overlay.querySelector('.cs-describe-go-btn')?.addEventListener('click', async () => {
    const seedText = (overlay.querySelector('.cs-seed-input')?.value || '').trim();
    const style = overlay.querySelector('.cs-style-select')?.value || '';
    const cardType = overlay.querySelector('input[name="cs-type"]:checked')?.value || 'single';
    const fullSeed = style ? `${seedText}\n\n(Style preference: ${style})` : seedText;
    await _startDescribeFlow(opts, cardType, fullSeed);
  });

  // Blank go
  overlay.querySelector('.cs-blank-go-btn')?.addEventListener('click', () => {
    _closeAny();
    if (typeof opts.onBlankRequested === 'function') {
      opts.onBlankRequested();
    }
  });

  // ── Wiki lane wiring ─────────────────────────────────────────────────────
  const wikiInput = overlay.querySelector('.cs-wiki-url-input');
  const wikiPreviewBtn = overlay.querySelector('.cs-wiki-preview-btn');
  const wikiErr = overlay.querySelector('.cs-wiki-error');

  function _validateWikiUrl(value) {
    const v = (value || '').trim();
    if (!v) return false;
    try {
      const u = new URL(v);
      return u.protocol === 'http:' || u.protocol === 'https:';
    } catch {
      return false;
    }
  }

  wikiInput?.addEventListener('input', () => {
    wikiPreviewBtn.disabled = !_validateWikiUrl(wikiInput.value);
    if (wikiErr) wikiErr.hidden = true;
  });
  wikiInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && _validateWikiUrl(wikiInput.value)) {
      e.preventDefault();
      wikiPreviewBtn?.click();
    }
  });

  wikiPreviewBtn?.addEventListener('click', async () => {
    const url = (wikiInput?.value || '').trim();
    if (!_validateWikiUrl(url)) return;
    const cardType = overlay.querySelector('input[name="cs-type"]:checked')?.value || 'single';
    wikiPreviewBtn.disabled = true;
    wikiPreviewBtn.classList.add('cs-loading');
    if (wikiErr) wikiErr.hidden = true;
    try {
      const resp = await fetch('/api/characters/cardsmith/wiki-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (!resp.ok) {
        const j = await resp.json().catch(() => ({}));
        throw new Error(j.error || `preview failed: ${resp.status}`);
      }
      const preview = await resp.json();
      _showWikiConfirmation(opts, overlay, { url, cardType, preview });
    } catch (err) {
      if (wikiErr) {
        wikiErr.textContent = err.message || String(err);
        wikiErr.hidden = false;
      } else {
        showToast(err.message || String(err), 'error');
      }
    } finally {
      wikiPreviewBtn.disabled = !_validateWikiUrl(wikiInput?.value);
      wikiPreviewBtn.classList.remove('cs-loading');
    }
  });

  return overlay;
}

// ─── Wiki confirmation step ────────────────────────────────────────────────

function _showWikiConfirmation(opts, overlay, ctx) {
  // Swap the launcher body to a confirmation view. Keep the same modal
  // shell so visual continuity is preserved (animation, focus trap,
  // dimensions all stay the same).
  const body = overlay.querySelector('.cardsmith-launcher-body');
  if (!body) return;

  const { preview, url, cardType } = ctx;
  // referrerpolicy="no-referrer" is load-bearing for Fandom/Wikia images:
  // their CDN (static.wikia.nocookie.net) 404s any request whose Referer
  // isn't a Fandom host — hotlink protection. Stripping the Referer
  // entirely turns the request into an anonymous direct fetch which the
  // CDN serves with 200. Also a generally safer default for third-party
  // image hosts that do similar gating.
  const thumbnail = preview.thumbnail_url
    ? `<img class="cs-confirm-thumb" src="${escapeHtml(preview.thumbnail_url)}" alt="" loading="lazy" referrerpolicy="no-referrer">`
    : `<div class="cs-confirm-thumb cs-confirm-thumb-placeholder" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" width="24" height="24">
           <rect x="4" y="6" width="16" height="14" rx="2"/><line x1="8" y1="11" x2="16" y2="11"/><line x1="8" y1="15" x2="14" y2="15"/>
         </svg>
       </div>`;

  const hostLabel = {
    fandom: 'Fandom',
    wikipedia: 'Wikipedia',
    generic: 'Web',
  }[preview.host_kind] || preview.host_kind || 'Web';

  const typeLabelMap = {
    single: 'Single character',
    ensemble: 'Ensemble',
    world_rpg: 'World / RPG',
  };
  // Show the user's CHOSEN type as the headline — never override their
  // selection with the classifier's guess. Surface the classifier's guess
  // only when it differs AND has meaningful confidence (>=0.7), as a hint.
  const userTypeLabel = typeLabelMap[cardType] || cardType;
  const detected = preview.detected_type;
  const detectedConfidence = preview.confidence || 0;
  const showMismatchHint = detected
    && detected !== cardType
    && detectedConfidence >= 0.7;
  const sourceLine = showMismatchHint
    ? `${escapeHtml(hostLabel)} · ${escapeHtml(userTypeLabel)} <span class="cs-confirm-source-hint">(wiki shape suggests ${escapeHtml(typeLabelMap[detected] || detected)})</span>`
    : `${escapeHtml(hostLabel)} · ${escapeHtml(userTypeLabel)}`;

  const sectionsHtml = (preview.section_headings || []).slice(0, 6)
    .map(h => `<span class="cs-confirm-tag">${escapeHtml(h)}</span>`)
    .join('');

  const warningHtml = preview.warning
    ? `<div class="cs-confirm-warning">${escapeHtml(preview.warning)}</div>`
    : '';

  body.innerHTML = `
    <div class="cs-confirm-step">
      <div class="cs-confirm-header">
        ${thumbnail}
        <div class="cs-confirm-meta">
          <div class="cs-confirm-title">${escapeHtml(preview.title || 'Untitled')}</div>
          <div class="cs-confirm-source">${sourceLine}</div>
          <div class="cs-confirm-summary">${escapeHtml((preview.summary || '').slice(0, 280))}${(preview.summary || '').length > 280 ? '…' : ''}</div>
        </div>
      </div>

      ${warningHtml}

      ${sectionsHtml ? `
        <div class="cs-confirm-row">
          <span class="field-label cs-confirm-row-label">Sections found</span>
          <div class="cs-confirm-tags">${sectionsHtml}</div>
        </div>
      ` : ''}

      <div class="cs-confirm-row">
        <label class="field-label cs-confirm-row-label" for="cs-confirm-twist">
          Twist (optional)
        </label>
        <textarea class="field-textarea cs-confirm-twist"
                  id="cs-confirm-twist"
                  rows="2"
                  placeholder="Anything you want different from canon?"></textarea>
      </div>

      <div class="cs-confirm-actions">
        <button class="btn btn-sm cs-confirm-back-btn">← Back</button>
        <button class="btn btn-primary btn-sm cs-confirm-begin-btn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
          Begin
        </button>
      </div>
    </div>
  `;

  setTimeout(() => overlay.querySelector('.cs-confirm-twist')?.focus(), 60);

  body.querySelector('.cs-confirm-back-btn')?.addEventListener('click', () => {
    // Re-render the launcher body to its original 3-lane layout. Easiest way:
    // close + reopen the launcher with the same opts.
    _closeAny();
    openCardsmithLauncher(opts);
  });

  body.querySelector('.cs-confirm-begin-btn')?.addEventListener('click', async () => {
    const twist = (body.querySelector('.cs-confirm-twist')?.value || '').trim();
    await _startWikiFlow(opts, { url, cardType, twist });
  });
}

async function _startWikiFlow(opts, ctx) {
  const beginBtn = _activeOverlay?.querySelector('.cs-confirm-begin-btn');
  if (beginBtn) {
    beginBtn.disabled = true;
    beginBtn.classList.add('cs-loading');
  }
  try {
    const resp = await fetch('/api/characters/cardsmith/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        card_type: ctx.cardType,
        source: 'wiki',
        wiki_url: ctx.url,
        seed_prompt: ctx.twist || '',
      }),
    });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.error || `start failed: ${resp.status}`);
    }
    const { session_id } = await resp.json();
    _saveResumeToken({ sessionId: session_id, cardType: ctx.cardType, source: 'wiki' });
    _openConversation(opts, {
      sessionId: session_id,
      cardType: ctx.cardType,
      seedPrompt: ctx.twist || '',
    });
  } catch (err) {
    showToast(`Couldn't start Cardsmith: ${err.message || err}`, 'error');
    if (beginBtn) {
      beginBtn.disabled = false;
      beginBtn.classList.remove('cs-loading');
    }
  }
}

function _escHandlerOnce(e) {
  if (e.key === 'Escape' && _activeOverlay) {
    e.preventDefault();
    _closeAny();
  }
}

// ─── Describe flow: start + open conversation modal ────────────────────────

async function _startDescribeFlow(opts, cardType, seedPrompt) {
  const btn = _activeOverlay?.querySelector('.cs-describe-go-btn');
  if (btn) {
    btn.disabled = true;
    btn.classList.add('cs-loading');
  }
  try {
    const resp = await fetch('/api/characters/cardsmith/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        card_type: cardType,
        source: 'describe',
        seed_prompt: seedPrompt || '',
      }),
    });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.error || `start failed: ${resp.status}`);
    }
    const { session_id } = await resp.json();
    _saveResumeToken({ sessionId: session_id, cardType, source: 'describe' });
    _openConversation(opts, {
      sessionId: session_id,
      cardType,
      seedPrompt,
    });
  } catch (err) {
    showToast(`Couldn't start Cardsmith: ${err.message || err}`, 'error');
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('cs-loading');
    }
  }
}

// ─── Conversation modal ────────────────────────────────────────────────────

function _openConversation(opts, ctx) {
  _closeAny();

  const overlay = document.createElement('div');
  overlay.className = 'persona-modal-overlay cardsmith-conversation-overlay';
  overlay.dataset.sessionId = ctx.sessionId;
  overlay.innerHTML = `
    <div class="persona-modal cardsmith-conversation" role="dialog" aria-modal="true" aria-labelledby="cs-conv-title">
      <div class="persona-modal-header">
        <span class="persona-modal-title" id="cs-conv-title">
          <span class="cs-conv-title-tag">Cardsmith</span>
          <span class="cs-conv-title-sep">·</span>
          <span class="cs-conv-title-type">${escapeHtml({
            single: 'Single character',
            ensemble: 'Ensemble',
            world_rpg: 'World / RPG',
          }[ctx.cardType] || 'Single character')}</span>
        </span>
        <div class="cs-conv-header-actions">
          <button class="btn btn-sm btn-ghost cs-drop-editor-btn" title="Save what we have so far and open it in the editor" aria-label="Save current draft and open in editor">
            Drop to editor
          </button>
          <button class="icon-btn small cs-close-btn" title="Close (will discard)" aria-label="Close without saving">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16" aria-hidden="true">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>

      <div class="persona-modal-body cardsmith-conversation-body">

        <div class="cs-conv-error-banner" role="alert" hidden>
          <div class="cs-conv-error-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
          </div>
          <div class="cs-conv-error-body">
            <div class="cs-conv-error-message"></div>
            <div class="cs-conv-error-actions">
              <button class="btn btn-sm cs-conv-error-retry">Retry</button>
              <button class="btn btn-sm btn-ghost cs-conv-error-dismiss">Dismiss</button>
            </div>
          </div>
        </div>

        <div class="cs-conv-mobile-tabs">
          <button class="cs-conv-mobile-tab cs-conv-mobile-tab-active" data-tab="chat">Conversation</button>
          <button class="cs-conv-mobile-tab" data-tab="preview">Card</button>
        </div>

        <div class="cs-conv-split">

          <div class="cs-conv-pane cs-conv-chat-pane" data-pane="chat">
            <div class="cs-chat-thread" role="log" aria-live="polite" aria-relevant="additions" aria-label="Cardsmith conversation"></div>
            <div class="cs-chat-status" role="status" aria-live="polite"></div>
          </div>

          <div class="cs-conv-pane cs-conv-preview-pane" data-pane="preview">
            <div class="cs-preview-card">
              <div class="cs-preview-empty">
                <span class="cs-preview-empty-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="6" width="16" height="14" rx="2"/>
                    <line x1="8" y1="11" x2="16" y2="11"/>
                    <line x1="8" y1="15" x2="14" y2="15"/>
                  </svg>
                </span>
                <span class="cs-preview-empty-text">Card fields fill in here as you go.</span>
              </div>
              <div class="cs-preview-fields"></div>
            </div>
          </div>

        </div>
      </div>

      <div class="cardsmith-conversation-footer">
        <textarea class="field-textarea cs-reply-input"
                  rows="2"
                  aria-label="Reply to the Cardsmith"
                  placeholder="Reply to the Cardsmith..."></textarea>
        <button class="btn btn-primary btn-sm cs-send-btn" aria-label="Send reply" disabled>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
          Send
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  _activeOverlay = overlay;
  _attachFocusTrap(overlay);

  const state = {
    sessionId: ctx.sessionId,
    cardType: ctx.cardType,
    seedPrompt: ctx.seedPrompt,
    fields: {},
    streaming: false,
    finalized: false,
    abortController: null,
    overlay,
    opts,
  };

  // Wire interactions
  _wireConversation(state);

  if (ctx.resumeFrom) {
    // Resume path: hydrate the thread + preview from the persisted server
    // snapshot. No initial /turn call — the prior conversation is the
    // starting point, and the user types the next message themselves.
    _hydrateConversation(state, ctx.resumeFrom);
  } else {
    if (ctx.chainedFromName) {
      // Continuation marker — visual hand-off from the just-saved card.
      // Stays in the thread as the user's first scrollback anchor.
      const thread = overlay.querySelector('.cs-chat-thread');
      if (thread) {
        const divider = document.createElement('div');
        divider.className = 'cs-chain-divider';
        divider.innerHTML = `
          <span class="cs-chain-divider-line" aria-hidden="true"></span>
          <span class="cs-chain-divider-label">
            Continuing from <strong>${escapeHtml(ctx.chainedFromName)}</strong>
          </span>
          <span class="cs-chain-divider-line" aria-hidden="true"></span>
        `;
        thread.appendChild(divider);
      }
    }
    // Kick off the first turn (Cardsmith speaks first, primed by seed prompt).
    setTimeout(() => _sendTurn(state, ''), 50);
  }
}

function _hydrateConversation(state, snapshot) {
  const thread = state.overlay.querySelector('.cs-chat-thread');
  if (!thread) return;
  // Replay each saved message as a fully-formed bubble. We skip the
  // streaming animation for hydrated bubbles — the user already saw them
  // stream the first time, and replaying the animation on resume would
  // feel like the model is re-thinking text it already said.
  for (const msg of (snapshot.messages || [])) {
    if (!msg || !msg.content) continue;
    if (msg.role === 'user') {
      _appendUserBubble(thread, msg.content);
    } else if (msg.role === 'assistant') {
      const bubble = _appendAssistantBubble(thread);
      bubble.classList.remove('cs-bubble-pending');
      bubble._rawText = msg.content;
      // Use the same markdown path as the end-of-stream finalize so
      // resumed bubbles match the look of live-streamed ones.
      _finalizeAssistantBubble(bubble);
    }
  }
  // Restore accumulated field state so the preview pane reflects every
  // field the model committed before the disconnect.
  state.fields = snapshot.fields || {};
  _renderPreview(state);
  // If the session was mid-turn when it dropped (last message is from the
  // user, no assistant reply followed) the user can either re-send to
  // prompt a fresh reply or just type their next message. The Send button
  // is enabled either way once the input has content — same as a fresh
  // session that just got its first cardsmith turn.
}

function _wireConversation(state) {
  const { overlay } = state;

  overlay.querySelector('.cs-close-btn')?.addEventListener('click', () => {
    _confirmCancel(state);
  });
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) _confirmCancel(state);
  });
  document.addEventListener('keydown', function escClose(e) {
    if (e.key === 'Escape' && _activeOverlay === overlay) {
      e.preventDefault();
      _confirmCancel(state);
      document.removeEventListener('keydown', escClose);
    }
  });

  // Mobile tab toggle
  overlay.querySelectorAll('.cs-conv-mobile-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      overlay.querySelectorAll('.cs-conv-mobile-tab').forEach(t => t.classList.remove('cs-conv-mobile-tab-active'));
      tab.classList.add('cs-conv-mobile-tab-active');
      overlay.querySelectorAll('.cs-conv-pane').forEach(p => {
        p.classList.toggle('cs-pane-active', p.dataset.pane === target);
      });
    });
  });

  // Reply send
  const input = overlay.querySelector('.cs-reply-input');
  const sendBtn = overlay.querySelector('.cs-send-btn');
  input?.addEventListener('input', () => {
    sendBtn.disabled = state.streaming || !input.value.trim();
  });
  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      sendBtn?.click();
    }
  });
  sendBtn?.addEventListener('click', () => {
    const text = (input.value || '').trim();
    if (!text || state.streaming) return;
    _sendTurn(state, text);
    input.value = '';
    sendBtn.disabled = true;
  });

  // Drop to editor
  overlay.querySelector('.cs-drop-editor-btn')?.addEventListener('click', async () => {
    if (state.streaming) {
      showToast('Wait for the Cardsmith to finish writing', 'info');
      return;
    }
    await _finalizeNow(state);
  });
}

// ─── Turn streaming (SSE) ──────────────────────────────────────────────────

async function _sendTurn(state, userMessage) {
  if (state.streaming || state.finalized) return;
  state.streaming = true;
  state.abortController = new AbortController();
  // Auto-clear any persistent error banner from a previous failed attempt —
  // either we'll succeed now (and it stays gone) or fail again (and the new
  // banner replaces it).
  _hideModalError(state);

  const thread = state.overlay.querySelector('.cs-chat-thread');
  const status = state.overlay.querySelector('.cs-chat-status');

  if (userMessage) {
    _appendUserBubble(thread, userMessage);
  }
  const assistantBubble = _appendAssistantBubble(thread);
  _setStatus(status, 'awaiting');

  try {
    const resp = await fetch('/api/characters/cardsmith/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        user_message: userMessage,
      }),
      signal: state.abortController.signal,
    });
    if (!resp.ok || !resp.body) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.error || `turn failed: ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let firstDeltaSeen = false;
    let firstThinkingSeen = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split('\n\n');
      buf = events.pop() || '';
      for (const ev of events) {
        const line = ev.replace(/^data:\s*/, '');
        if (line === '[DONE]') continue;
        if (!line) continue;
        let payload = null;
        try { payload = JSON.parse(line); } catch { continue; }
        if (payload.type === 'delta') {
          if (!firstDeltaSeen) {
            firstDeltaSeen = true;
            _setStatus(status, 'responding');
            assistantBubble.classList.remove('cs-bubble-pending');
            // Visible content arrived — collapse any reasoning block so the
            // user's eye goes to the actual reply rather than the (often
            // long) reasoning trace.
            _collapseThinkingBlock(assistantBubble);
          }
          _appendDelta(assistantBubble, payload.text || '');
          _scrollThreadToBottom(thread);
        } else if (payload.type === 'thinking') {
          // Reasoning-capable models (GLM-4.x, EXAONE, DeepSeek V3.2/V4,
          // Qwen 3.x in thinking mode) emit reasoning before any visible
          // content. Without surfacing this, turn 2+ looks hung while
          // tokens are quietly being burned. We show the thinking text in
          // a dim collapsible block and switch the status to "reasoning".
          if (!firstThinkingSeen && !firstDeltaSeen) {
            firstThinkingSeen = true;
            _setStatus(status, 'reasoning');
            assistantBubble.classList.remove('cs-bubble-pending');
          }
          _appendThinkingDelta(assistantBubble, payload.text || '');
          _scrollThreadToBottom(thread);
        } else if (payload.type === 'field') {
          _commitField(state, payload.path, payload.value);
        } else if (payload.type === 'fetching') {
          // Cardsmith committed fetch_targets[] — server is pulling docs
          // before the next turn. Show a subtle inline status so the user
          // sees the agentic loop working.
          _appendFetchProgress(thread, payload);
          _scrollThreadToBottom(thread);
        } else if (payload.type === 'fetched') {
          _completeFetchProgress(thread, payload.count);
        } else if (payload.type === 'finalized') {
          state.finalized = true;
          state.streaming = false;
          _setStatus(status, '');
          await _handleFinalized(
            state, payload.char_id, payload.name,
            {
              hasUniverse: !!payload.has_universe,
              sessionId: payload.session_id || state.sessionId,
            },
          );
          return;
        } else if (payload.type === 'error') {
          throw new Error(payload.error || 'Cardsmith error');
        }
      }
    }
  } catch (err) {
    if (err && err.name === 'AbortError') {
      // User-initiated abort — clean up the placeholder bubble silently.
      assistantBubble.remove();
    } else {
      assistantBubble.classList.remove('cs-bubble-pending');
      assistantBubble.classList.add('cs-bubble-error');
      assistantBubble.querySelector('.cs-bubble-content').textContent =
        `(error — ${err.message || err})`;
      // Persistent banner instead of an ephemeral toast — user can read it
      // even if they Alt-Tabbed away during the failure.
      _showModalError(state, err.message || String(err), () => _sendTurn(state, userMessage));
    }
  } finally {
    // Re-render the streamed plain text as markdown now that the model has
    // finished. Skips errored / aborted bubbles (those have no _rawText
    // worth rendering, or were already removed).
    if (!state.finalized && assistantBubble && assistantBubble.isConnected) {
      _collapseThinkingBlock(assistantBubble);
      _finalizeAssistantBubble(assistantBubble);
    }
    state.streaming = false;
    state.abortController = null;
    _setStatus(state.overlay.querySelector('.cs-chat-status'), '');
    const sendBtn = state.overlay.querySelector('.cs-send-btn');
    const input = state.overlay.querySelector('.cs-reply-input');
    if (sendBtn) sendBtn.disabled = !input?.value?.trim();
    input?.focus();
  }
}

// ─── DOM helpers ───────────────────────────────────────────────────────────

function _appendUserBubble(thread, text) {
  const el = document.createElement('div');
  el.className = 'cs-bubble cs-bubble-user';
  el.innerHTML = `<div class="cs-bubble-content">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
  thread.appendChild(el);
  _scrollThreadToBottom(thread);
  return el;
}

function _appendAssistantBubble(thread) {
  const el = document.createElement('div');
  el.className = 'cs-bubble cs-bubble-cardsmith cs-bubble-pending';
  el.innerHTML = `
    <div class="cs-bubble-author">Cardsmith</div>
    <div class="cs-bubble-content"><span class="cs-pending-dots"><span></span><span></span><span></span></span></div>
  `;
  el._rawText = '';  // accumulate raw deltas for end-of-stream markdown rerender
  thread.appendChild(el);
  _scrollThreadToBottom(thread);
  return el;
}

function _appendDelta(bubble, text) {
  const content = bubble.querySelector('.cs-bubble-content');
  if (!content) return;
  // First delta? clear placeholder.
  if (!bubble.classList.contains('cs-bubble-just-cleared') &&
      content.querySelector('.cs-pending-dots')) {
    content.innerHTML = '';
    bubble.classList.add('cs-bubble-just-cleared');
  }
  // During streaming, append as plain text — partial markdown (e.g. an
  // unclosed `*italic`) renders ugly. We re-render to markdown after the
  // turn completes via _finalizeAssistantBubble.
  bubble._rawText = (bubble._rawText || '') + text;
  const span = document.createElement('span');
  span.textContent = text;
  content.appendChild(span);
}

function _appendThinkingDelta(bubble, text) {
  if (!text) return;
  // Lazily create a thinking sub-block above the visible content. The
  // bubble layout becomes: [thinking block (dim, italic)] then [content].
  let thinkBlock = bubble.querySelector('.cs-bubble-thinking');
  if (!thinkBlock) {
    // Strip the pending dots in the content area so the bubble stops
    // looking idle (we now have actual reasoning streaming).
    const content = bubble.querySelector('.cs-bubble-content');
    if (content && !bubble.classList.contains('cs-bubble-just-cleared') &&
        content.querySelector('.cs-pending-dots')) {
      content.innerHTML = '';
      bubble.classList.add('cs-bubble-just-cleared');
    }
    thinkBlock = document.createElement('details');
    thinkBlock.className = 'cs-bubble-thinking';
    thinkBlock.open = true;
    thinkBlock.innerHTML = `
      <summary class="cs-bubble-thinking-summary">
        <span class="cs-bubble-thinking-dot" aria-hidden="true"></span>
        Reasoning…
      </summary>
      <div class="cs-bubble-thinking-body"></div>
    `;
    // Insert before the visible content div so it appears above.
    const contentEl = bubble.querySelector('.cs-bubble-content');
    if (contentEl) bubble.insertBefore(thinkBlock, contentEl);
    else bubble.appendChild(thinkBlock);
  }
  const body = thinkBlock.querySelector('.cs-bubble-thinking-body');
  if (!body) return;
  const span = document.createElement('span');
  span.textContent = text;
  body.appendChild(span);
}

function _collapseThinkingBlock(bubble) {
  // Once visible content starts (or the turn ends), collapse the reasoning
  // block so the user's eye lands on the actual reply.
  const thinkBlock = bubble?.querySelector('.cs-bubble-thinking');
  if (thinkBlock) {
    thinkBlock.open = false;
    const summary = thinkBlock.querySelector('.cs-bubble-thinking-summary');
    if (summary) {
      summary.innerHTML = `
        <span class="cs-bubble-thinking-dot cs-bubble-thinking-dot-done" aria-hidden="true"></span>
        Reasoning (click to expand)
      `;
    }
  }
}

function _finalizeAssistantBubble(bubble) {
  if (!bubble) return;
  const content = bubble.querySelector('.cs-bubble-content');
  if (!content) return;
  const raw = (bubble._rawText || '').trim();
  if (!raw) return;
  try {
    content.innerHTML = renderMarkdown(raw, { mode: 'narrative', narrativePanelsCollapsed: true });
  } catch {
    // Markdown rendering should never throw, but if it does we keep the
    // already-rendered plain-text view that streamed in.
  }
}

// Coalesced to one reflow/frame (was a synchronous scrollHeight read on every
// SSE delta — 3× per llm_delta — which thrashed layout). The near-bottom gate
// also stops auto-scroll from yanking a user who scrolled up, matching the
// chat/voice behavior.
const _scrollThreadToBottom = rafCoalesce((thread) => {
  if (!thread) return;
  const dist = thread.scrollHeight - thread.scrollTop - thread.clientHeight;
  if (dist > 80) return; // user scrolled up — leave them be
  thread.scrollTop = thread.scrollHeight;
});

// ─── Fetch progress (agentic loop) ─────────────────────────────────────────

function _appendFetchProgress(thread, payload) {
  const el = document.createElement('div');
  el.className = 'cs-fetch-progress cs-fetch-progress-running';
  const targets = (payload.targets || []).filter(Boolean).slice(0, 4);
  const targetList = targets.length
    ? `<span class="cs-fetch-progress-targets">${targets.map(t => escapeHtml(String(t))).join(', ')}${(payload.targets || []).length > targets.length ? '…' : ''}</span>`
    : '';
  el.innerHTML = `
    <span class="cs-fetch-progress-spinner" aria-hidden="true"></span>
    <span class="cs-fetch-progress-label">
      Pulling ${payload.count} reference${payload.count === 1 ? '' : 's'}
      ${targetList}
    </span>
  `;
  thread.appendChild(el);
}

function _completeFetchProgress(thread, count) {
  // Find the most recent in-flight progress element and mark it done.
  const el = thread.querySelector('.cs-fetch-progress-running:last-child');
  if (!el) return;
  el.classList.remove('cs-fetch-progress-running');
  el.classList.add('cs-fetch-progress-done');
  const c = Number.isFinite(count) ? count : 0;
  el.innerHTML = `
    <span class="cs-fetch-progress-check" aria-hidden="true">✓</span>
    <span class="cs-fetch-progress-label">Pulled ${c} reference${c === 1 ? '' : 's'}</span>
  `;
}

function _setStatus(node, kind) {
  if (!node) return;
  if (!kind) {
    node.textContent = '';
    node.className = 'cs-chat-status';
    return;
  }
  if (kind === 'awaiting') {
    node.textContent = 'Cardsmith is thinking…';
    node.className = 'cs-chat-status cs-status-awaiting';
  } else if (kind === 'reasoning') {
    node.textContent = 'Cardsmith is reasoning…';
    node.className = 'cs-chat-status cs-status-reasoning';
  } else if (kind === 'responding') {
    node.textContent = 'Cardsmith is writing…';
    node.className = 'cs-chat-status cs-status-responding';
  }
}

// ─── Live preview ──────────────────────────────────────────────────────────

// Base labels for each commit key. Per-type overrides let the same field
// surface as e.g. "Description" for Single and "Setting" for World/RPG.
const _PREVIEW_LABELS = {
  name: 'Name',
  description: 'Description',
  personality: 'Personality',
  scenario: 'Scenario',
  greeting: 'Greeting',
  examples: 'Examples',
  visualTraits: 'Visual traits',
  imageStyle: 'Image style',
  voice: 'Voice (TTS)',
  tags: 'Tags',
  alternateGreetings: 'Alternate greetings',
  lorebook: 'Lorebook entries',
  regex_scripts: 'Regex scripts',
  avatar_prompt: 'Background image',
  // Ensemble-specific
  group_dynamic: 'Group dynamic',
  members: 'Members',
  relationships: 'Relationships',
  generation_mode: 'Speaker selection',
};

const _PREVIEW_LABEL_OVERRIDES = {
  world_rpg: {
    description: 'Setting',
    personality: 'Narrator voice',
    scenario: 'World state',
    greeting: 'Opening scene',
    alternateGreetings: 'Alt openings',
    visualTraits: 'World aesthetic',
  },
};

function _previewLabel(key, cardType) {
  const overrides = _PREVIEW_LABEL_OVERRIDES[cardType] || {};
  return overrides[key] || _PREVIEW_LABELS[key] || key;
}

const _PREVIEW_ORDER_SINGLE = [
  'name', 'description', 'personality', 'scenario', 'greeting', 'examples',
  'visualTraits', 'imageStyle', 'voice', 'tags', 'alternateGreetings',
  'lorebook', 'regex_scripts', 'avatar_prompt',
];

const _PREVIEW_ORDER_ENSEMBLE = [
  'name', 'group_dynamic', 'members', 'relationships', 'generation_mode',
  'scenario', 'greeting', 'examples', 'imageStyle', 'tags',
  'alternateGreetings', 'lorebook', 'regex_scripts', 'avatar_prompt',
];

// World/RPG: setting + lorebook are the centerpieces; surface them early.
const _PREVIEW_ORDER_WORLD_RPG = [
  'name', 'description', 'personality', 'scenario', 'greeting',
  'lorebook', 'alternateGreetings', 'imageStyle', 'tags',
  'examples', 'voice', 'visualTraits', 'avatar_prompt', 'regex_scripts',
];

function _previewOrder(state) {
  if (state.cardType === 'ensemble') return _PREVIEW_ORDER_ENSEMBLE;
  if (state.cardType === 'world_rpg') return _PREVIEW_ORDER_WORLD_RPG;
  return _PREVIEW_ORDER_SINGLE;
}

function _commitField(state, path, value) {
  if (path.endsWith('[]')) {
    const base = path.slice(0, -2);
    if (!Array.isArray(state.fields[base])) state.fields[base] = [];
    state.fields[base].push(_tryParseJson(value));
  } else {
    state.fields[path] = value;
  }
  // Description paragraph slots (desc_physical / desc_personality / desc_depth)
  // compose into the unified description preview. The slots themselves never
  // render as separate keys (not listed in _PREVIEW_ORDER), but the user sees
  // the description grow paragraph-by-paragraph as each slot lands.
  if (path === 'desc_physical' || path === 'desc_personality' || path === 'desc_depth') {
    const parts = [];
    for (const k of ['desc_physical', 'desc_personality', 'desc_depth']) {
      const v = state.fields[k];
      if (typeof v === 'string' && v.trim()) parts.push(v.trim());
    }
    if (parts.length) state.fields.description = parts.join('\n\n');
  }
  _renderPreview(state);
}

function _tryParseJson(s) {
  if (typeof s !== 'string') return s;
  const trimmed = s.trim();
  if (trimmed && (trimmed[0] === '{' || trimmed[0] === '[')) {
    try { return JSON.parse(trimmed); } catch { /* fall through */ }
  }
  return s;
}

function _renderPreview(state) {
  const fieldsEl = state.overlay.querySelector('.cs-preview-fields');
  const emptyEl = state.overlay.querySelector('.cs-preview-empty');
  if (!fieldsEl) return;

  const order = _previewOrder(state);
  const populated = order.filter(k => _hasValue(state.fields[k]));
  if (populated.length === 0) {
    if (emptyEl) emptyEl.style.display = '';
    fieldsEl.innerHTML = '';
    return;
  }
  if (emptyEl) emptyEl.style.display = 'none';

  fieldsEl.innerHTML = populated.map(k => _renderPreviewField(k, state.fields[k], state.cardType)).join('');
}

function _hasValue(v) {
  if (v === undefined || v === null) return false;
  if (typeof v === 'string') return v.trim().length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return true;
}

// Fields whose value reads as prose — render with markdown so asterisks,
// emphasis, and macros display nicely. Other fields render as escaped text
// because they're either lists, code-shaped, or short tags.
const _MARKDOWN_FIELDS = new Set([
  'description', 'personality', 'scenario', 'greeting', 'examples',
]);

function _renderProse(text) {
  try {
    return renderMarkdown(text, { mode: 'narrative', narrativePanelsCollapsed: true });
  } catch {
    return escapeHtml(text).replace(/\n/g, '<br>');
  }
}

function _renderPreviewField(key, value, cardType) {
  const label = _previewLabel(key, cardType);
  let body = '';
  if (Array.isArray(value)) {
    if (key === 'tags') {
      body = `<div class="cs-preview-chips">${
        value.map(t => `<span class="cs-preview-chip">${escapeHtml(String(t))}</span>`).join('')
      }</div>`;
    } else if (key === 'lorebook') {
      body = `<div class="cs-preview-list">${
        value.map(entry => {
          const keys = Array.isArray(entry?.keys) ? entry.keys.join(', ') : '';
          const content = entry?.content || '';
          return `<div class="cs-preview-list-item"><strong>${escapeHtml(keys)}</strong> — ${escapeHtml(_clip(content, 140))}</div>`;
        }).join('')
      }</div>`;
    } else if (key === 'regex_scripts') {
      body = `<div class="cs-preview-list">${
        value.map(r => `<div class="cs-preview-list-item"><code>${escapeHtml(_clip(r?.find || '', 60))}</code> → <code>${escapeHtml(_clip(r?.replace || '', 60))}</code> <span class="cs-preview-chip cs-preview-chip-muted">${escapeHtml(r?.placement || 'output')}</span></div>`).join('')
      }</div>`;
    } else if (key === 'alternateGreetings') {
      body = `<div class="cs-preview-list">${
        value.map((g, i) => `<div class="cs-preview-list-item"><strong>Alt ${i + 1}</strong><div class="cs-preview-text">${_renderProse(String(g))}</div></div>`).join('')
      }</div>`;
    } else if (key === 'members') {
      body = `<div class="cs-preview-list">${
        value.map(m => {
          const name = escapeHtml(m?.name || '?');
          const role = m?.role ? `<span class="cs-preview-chip cs-preview-chip-muted">${escapeHtml(m.role)}</span>` : '';
          const summary = m?.summary ? escapeHtml(_clip(m.summary, 140)) : '<em style="opacity:0.6">(pending)</em>';
          const physical = m?.physical ? `<div class="cs-preview-member-physical">${escapeHtml(_clip(m.physical, 160))}</div>` : '';
          return `<div class="cs-preview-list-item"><div class="cs-preview-member-head"><strong>${name}</strong> ${role}</div><div class="cs-preview-text">${summary}</div>${physical}</div>`;
        }).join('')
      }</div>`;
    } else if (key === 'relationships') {
      body = `<div class="cs-preview-list">${
        value.map(r => {
          const src = escapeHtml(r?.source || '?');
          const tgt = escapeHtml(r?.target || '?');
          const label = r?.label ? `<span class="cs-preview-chip cs-preview-chip-muted">${escapeHtml(r.label)}</span>` : '';
          const trust = typeof r?.trust === 'number' ? r.trust.toFixed(2) : '0.00';
          const aff = typeof r?.affection === 'number' ? r.affection.toFixed(2) : '0.00';
          const ten = typeof r?.tension === 'number' ? r.tension.toFixed(2) : '0.00';
          return `<div class="cs-preview-list-item"><div><strong>${src}</strong> → <strong>${tgt}</strong> ${label}</div><div class="cs-preview-text" style="font-family:var(--font-mono);font-size:11px">trust ${trust} · affection ${aff} · tension ${ten}</div></div>`;
        }).join('')
      }</div>`;
    } else {
      body = `<div class="cs-preview-text">${escapeHtml(value.map(v => typeof v === 'string' ? v : JSON.stringify(v)).join(', '))}</div>`;
    }
  } else if (typeof value === 'string') {
    if (_MARKDOWN_FIELDS.has(key)) {
      body = `<div class="cs-preview-text cs-preview-text-prose">${_renderProse(value)}</div>`;
    } else {
      body = `<div class="cs-preview-text">${escapeHtml(value).replace(/\n/g, '<br>')}</div>`;
    }
  } else {
    body = `<div class="cs-preview-text"><code>${escapeHtml(JSON.stringify(value))}</code></div>`;
  }

  return `
    <div class="cs-preview-field">
      <div class="cs-preview-field-label">${escapeHtml(label)}</div>
      ${body}
    </div>
  `;
}

function _clip(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

// ─── Finalize / cancel ─────────────────────────────────────────────────────

async function _finalizeNow(state) {
  if (state.finalized) return;
  state.finalized = true;
  // Recovery extraction can run up to ~10s on the server when the model
  // bypassed the inline-tag protocol (see _recover_fields_from_conversation
  // in cardsmith_routes.py). Disable the drop button + show a spinner so
  // the user doesn't think the click was lost and double-fire.
  const dropBtn = state.overlay?.querySelector('.cs-drop-editor-btn');
  if (dropBtn) {
    dropBtn.disabled = true;
    dropBtn.classList.add('cs-loading');
  }
  showToast('Saving card…', 'loading', 0);
  try {
    const resp = await fetch('/api/characters/cardsmith/finalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.error || `finalize failed: ${resp.status}`);
    }
    const data = await resp.json();
    await _handleFinalized(
      state, data.char_id, data.name,
      {
        hasUniverse: !!data.has_universe,
        sessionId: data.session_id || state.sessionId,
      },
    );
  } catch (err) {
    state.finalized = false;
    if (dropBtn) {
      dropBtn.disabled = false;
      dropBtn.classList.remove('cs-loading');
    }
    showToast(`Save failed: ${err.message || err}`, 'error');
  }
}

async function _handleFinalized(state, charId, name, opts = {}) {
  // The card is durably saved in ui_characters. Two finalize-time outcomes
  // depending on universe context:
  //  - When the session has wiki/scratchpad context, offer the user a
  //    chain-continuation: build another character in the same setting
  //    without re-pasting the wiki URL. The modal stays open with a save
  //    confirmation + chain prompt; the chat thread is preserved as
  //    backdrop until the user picks an action.
  //  - When there's no universe context, close immediately as before.
  _clearResumeToken();
  const hasUniverse = !!opts.hasUniverse;
  const parentSessionId = opts.sessionId || state.sessionId;
  if (hasUniverse && state.overlay?.isConnected) {
    _showUniverseChainPrompt(state, charId, name, parentSessionId);
    // Trigger the onCardSaved callback now so the inspector + character
    // grid update behind the modal — user sees the card already present
    // when they eventually close.
    if (typeof state.opts.onCardSaved === 'function') {
      try {
        await state.opts.onCardSaved(charId, name);
      } catch (err) {
        console.warn('cardsmith onCardSaved handler failed', err);
      }
    }
    return;
  }
  _closeAny();
  showToast(`Saved ${name || 'character'}`, 'success');
  if (typeof state.opts.onCardSaved === 'function') {
    try {
      await state.opts.onCardSaved(charId, name);
    } catch (err) {
      console.warn('cardsmith onCardSaved handler failed', err);
    }
  }
}

function _showUniverseChainPrompt(state, charId, name, parentSessionId) {
  // Render an inline post-save panel overlaying the conversation modal.
  // Chat thread + preview stay visible (as backdrop) so the user can scroll
  // back to what they just made before deciding whether to chain.
  const body = state.overlay?.querySelector('.cardsmith-conversation-body');
  if (!body) return;
  // If a prompt is already present (defensive: double-finalize race), don't
  // stack a second one.
  if (body.querySelector('.cs-chain-prompt')) return;

  const panel = document.createElement('div');
  panel.className = 'cs-chain-prompt';
  panel.innerHTML = `
    <div class="cs-chain-card">
      <div class="cs-chain-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="22" height="22">
          <path d="M20 6L9 17l-5-5"/>
        </svg>
      </div>
      <div class="cs-chain-body">
        <div class="cs-chain-title">Saved ${escapeHtml(name || 'character')}</div>
        <div class="cs-chain-subtitle">Build another character in this universe?</div>
      </div>
      <div class="cs-chain-type-row">
        <label class="cs-chain-type-pill cs-chain-type-active">
          <input type="radio" name="cs-chain-type" value="single" checked>
          <span>Single</span>
        </label>
        <label class="cs-chain-type-pill">
          <input type="radio" name="cs-chain-type" value="ensemble">
          <span>Ensemble</span>
        </label>
        <label class="cs-chain-type-pill">
          <input type="radio" name="cs-chain-type" value="world_rpg">
          <span>World / RPG</span>
        </label>
      </div>
      <div class="cs-chain-actions">
        <button class="btn btn-primary btn-sm cs-chain-continue">+ Add another</button>
        <button class="btn btn-sm btn-ghost cs-chain-done">Done</button>
      </div>
    </div>
  `;
  body.appendChild(panel);

  panel.querySelectorAll('.cs-chain-type-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      panel.querySelectorAll('.cs-chain-type-pill').forEach(p =>
        p.classList.remove('cs-chain-type-active')
      );
      pill.classList.add('cs-chain-type-active');
    });
  });

  panel.querySelector('.cs-chain-done')?.addEventListener('click', () => {
    _closeAny();
    showToast(`Saved ${name || 'character'}`, 'success');
  });

  panel.querySelector('.cs-chain-continue')?.addEventListener('click', async () => {
    const btn = panel.querySelector('.cs-chain-continue');
    if (btn) {
      btn.disabled = true;
      btn.classList.add('cs-loading');
    }
    const chosenType = panel.querySelector(
      'input[name="cs-chain-type"]:checked',
    )?.value || 'single';
    try {
      const resp = await fetch('/api/characters/cardsmith/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          card_type: chosenType,
          // Source is inherited from parent server-side; pass describe as
          // a safe default since this isn't a wiki-URL flow.
          source: 'describe',
          parent_session_id: parentSessionId,
        }),
      });
      if (!resp.ok) {
        const j = await resp.json().catch(() => ({}));
        throw new Error(j.error || `start failed: ${resp.status}`);
      }
      const { session_id } = await resp.json();
      _saveResumeToken({
        sessionId: session_id, cardType: chosenType, source: 'describe',
      });
      // Swap the conversation overlay to the new session. The previous
      // chat thread isn't preserved across sessions — each session owns
      // its own messages — but the scratchpad/universe context carries.
      _openConversation(state.opts, {
        sessionId: session_id,
        cardType: chosenType,
        seedPrompt: '',
        chainedFromName: name,
      });
    } catch (err) {
      if (btn) {
        btn.disabled = false;
        btn.classList.remove('cs-loading');
      }
      showToast(`Couldn't continue: ${err.message || err}`, 'error');
    }
  });
}

function _confirmCancel(state) {
  if (state.finalized) {
    _closeAny();
    return;
  }
  const hasProgress = state.fields && Object.keys(state.fields).length > 0;
  if (!hasProgress) {
    _cancelSession(state);
    return;
  }
  // Quick inline confirm — full modal would feel heavy here.
  if (window.confirm('Discard this draft? You can also "Drop to editor" to keep what you have.')) {
    _cancelSession(state);
  }
}

// ─── Persistent error banner ───────────────────────────────────────────────

function _showModalError(state, message, retryFn) {
  const banner = state.overlay?.querySelector('.cs-conv-error-banner');
  if (!banner) return;
  const msgEl = banner.querySelector('.cs-conv-error-message');
  const retryBtn = banner.querySelector('.cs-conv-error-retry');
  const dismissBtn = banner.querySelector('.cs-conv-error-dismiss');
  if (msgEl) msgEl.textContent = String(message || 'Something went wrong.');
  banner.hidden = false;

  // Re-bind handlers each time so closures over the latest state stay correct.
  if (retryBtn) {
    retryBtn.onclick = () => {
      banner.hidden = true;
      if (typeof retryFn === 'function') retryFn();
    };
    retryBtn.style.display = (typeof retryFn === 'function') ? '' : 'none';
  }
  if (dismissBtn) {
    dismissBtn.onclick = () => { banner.hidden = true; };
  }
}

function _hideModalError(state) {
  const banner = state.overlay?.querySelector('.cs-conv-error-banner');
  if (banner) banner.hidden = true;
}

function _cancelSession(state) {
  // Abort any in-flight /turn fetch so the backend can short-circuit and stop
  // wasting tokens on a reply the user will never see.
  if (state.abortController) {
    try { state.abortController.abort(); } catch { /* noop */ }
    state.abortController = null;
  }
  fetch('/api/characters/cardsmith/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).catch(() => {});
  // Drop the localStorage resume token — cancel is the user explicitly
  // throwing this session away, the banner should not pop again next time.
  _clearResumeToken();
  _closeAny();
}
