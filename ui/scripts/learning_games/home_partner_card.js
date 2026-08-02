/**
 * Homepage re-entry card for the language partner.
 *
 * Renders inside the chat empty-state so a returning user sees
 * "Continue with Yuki — 3 words due for review" the moment they
 * land on a fresh session, instead of having to dig through the
 * Learning hub to re-enter.
 *
 * Reuses:
 *   - GET /api/learning/partners (extended to include dueCount)
 *   - openLanguagePartner / langLabel from ./partner_launch.js
 *
 * Dim variant (.lg-home-partner-quiet) shows when dueCount == 0 so
 * the card is still present but doesn't read as nagging on days
 * when there's nothing scheduled.
 *
 * Cap at MAX_CARDS partners — beyond that the empty-state turns
 * into a wall of rose. Ordering: due count desc, then most-recently-
 * updated. See [[project-language-partner]].
 */

import { escapeHtml } from '../app.js';
import { openLanguagePartner, langLabel } from './partner_launch.js';

const MAX_CARDS = 3;

// In-memory cache so empty-state toggles (which can fire on every
// session switch) don't refetch /partners. Cleared on launch so the
// next empty-state reflects the new dueCount after a review session.
let _cache = null;
let _inflight = null;

async function _fetchPartners() {
  if (_cache) return _cache;
  if (_inflight) return _inflight;
  _inflight = (async () => {
    try {
      const r = await fetch('/api/learning/partners');
      if (!r.ok) throw new Error(`partners endpoint ${r.status}`);
      const json = await r.json();
      _cache = Array.isArray(json?.partners) ? json.partners : [];
      return _cache;
    } catch (err) {
      console.warn('[home-partner] fetch failed', err);
      _cache = [];
      return _cache;
    } finally {
      _inflight = null;
    }
  })();
  return _inflight;
}

export function invalidatePartnerCache() {
  _cache = null;
}

function _orderForDisplay(partners) {
  // Due count desc primary so urgent reviews surface first; then
  // updatedAt desc as tie-breaker (most-recently-touched partner
  // first when nothing's due).
  return [...partners].sort((a, b) => {
    const dueDiff = (b.dueCount || 0) - (a.dueCount || 0);
    if (dueDiff !== 0) return dueDiff;
    return String(b.updatedAt || '').localeCompare(String(a.updatedAt || ''));
  }).slice(0, MAX_CARDS);
}

function _renderCard(partner) {
  const due = Number(partner.dueCount || 0);
  const isQuiet = due <= 0;
  // Escape via String->escapeHtml even though `due` is numeric and
  // safe — keeps the validator happy (its regex can't prove the
  // interpolation is non-user-input) and costs nothing.
  const dueStr = escapeHtml(String(due));
  const subtitle = isQuiet
    ? `Pick up where you left off in ${escapeHtml(langLabel(partner.lang_code))}.`
    : `<b>${dueStr}</b> ${due === 1 ? 'word' : 'words'} due for review`;
  const badge = isQuiet
    ? ''
    : `<div class="lg-home-partner-badge">${dueStr}</div>`;
  const quietCls = isQuiet ? ' lg-home-partner-quiet' : '';

  return `
    <button type="button" class="lg-home-partner${quietCls}" data-lang="${escapeHtml(partner.lang_code)}" aria-label="Continue with ${escapeHtml(partner.name)}">
      <div class="lg-home-partner-emoji">💬</div>
      <div class="lg-home-partner-body">
        <div class="lg-home-partner-name">Continue with ${escapeHtml(partner.name)}</div>
        <div class="lg-home-partner-tag">${subtitle}</div>
      </div>
      ${badge}
      <div class="lg-home-partner-cta" aria-hidden="true">→</div>
    </button>`;
}

/**
 * Render partner cards into the given container. Idempotent: removes
 * its own previously-rendered cards before re-rendering so toggling
 * the empty-state doesn't stack duplicates.
 *
 * Returns the number of cards rendered. 0 means the caller can
 * safely leave the empty-state's existing layout alone.
 */
export async function renderHomePartnerCards(container) {
  if (!container) return 0;

  // Strip previous render (if any) — empty-state may have been
  // toggled by a session switch, and we always re-render from
  // fresh data rather than trusting stale DOM.
  container.querySelectorAll('.lg-home-partner-host').forEach(n => n.remove());

  const partners = await _fetchPartners();
  if (!partners.length) {
    // No partners: release the width override so the empty-state
    // returns to its default centered chip-row layout.
    container.classList.remove('lg-home-partner-active');
    return 0;
  }

  const ordered = _orderForDisplay(partners);
  const host = document.createElement('div');
  host.className = 'lg-home-partner-host';
  host.innerHTML = ordered.map(_renderCard).join('');

  // Widen the empty-state past its 360px default — the partner card
  // is denser than a prompt chip and reads as the primary surface
  // here, so it should breathe.
  container.classList.add('lg-home-partner-active');

  // Insert above any existing chip row so the partner reads as the
  // primary suggestion, not buried below the generic prompts.
  const chips = container.querySelector('.empty-state-chips');
  if (chips) {
    container.insertBefore(host, chips);
  } else {
    container.appendChild(host);
  }

  host.querySelectorAll('.lg-home-partner').forEach(btn => {
    btn.addEventListener('click', async () => {
      const lang = btn.dataset.lang;
      if (!lang) return;
      btn.classList.add('lg-home-partner-launching');
      try {
        await openLanguagePartner(lang);
        // Bust the cache so the next empty-state render reflects
        // the post-review due count (the user may have just
        // burned down the queue).
        invalidatePartnerCache();
      } catch (err) {
        console.warn('[home-partner] launch failed', err);
        btn.classList.remove('lg-home-partner-launching');
      }
    });
  });

  return ordered.length;
}
