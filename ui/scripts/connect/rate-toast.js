/* connect/rate-toast.js — Post-call summary card.
 *
 * Replaces the original three-button toast with a richer summary:
 * peer name + avatar, duration + modality + outcome line, rating row,
 * optional note, and action row (Call back / Open chat / Dismiss).
 *
 * Takes the same showPostCallRating({callId, peerDid, durationSeconds,
 * modalities}) entry point so the ui.js wiring doesn't change. Extra
 * fields are optional — older callers continue to work.
 *
 * Visible only for calls that actually connected; ui.js gates on
 * reachedConnected before invoking us.
 */

import { escapeHtml, showToast } from '../app.js';
import { icon } from './icons.js';
import { rateCall, resolvePeerName } from './messages.js';

let _card = null;
let _activeCallId = '';
let _activePeerDid = '';
let _activeModalities = '';
let _dismissTimer = null;

const AUTO_DISMISS_MS = 45_000;
const SAVED_HOLD_MS = 1_400;

/**
 * Show the post-call summary card for a finished call.
 *
 *   showPostCallRating({
 *     callId, peerDid,
 *     durationSeconds, modalities,
 *   });
 *
 * Idempotent on call_id — calling twice in a row updates the existing
 * card in place rather than stacking. Auto-dismissed after
 * AUTO_DISMISS_MS so an unrated call doesn't clutter the UI.
 */
export function showPostCallRating({
  callId, peerDid = '',
  durationSeconds = null, modalities = '',
}) {
  if (!callId) return;
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  _activeCallId = callId;
  _activePeerDid = peerDid;
  _activeModalities = modalities;
  _ensureCard();
  _populate({ peerDid, durationSeconds, modalities });
  _resetButtons();
  _card.classList.remove('hidden');
  _dismissTimer = setTimeout(hidePostCallRating, AUTO_DISMISS_MS);
}

/** Hide and clear the card. Safe to call any time. */
export function hidePostCallRating() {
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  if (_card) _card.classList.add('hidden');
  _activeCallId = '';
  _activePeerDid = '';
  _activeModalities = '';
}

// ── Internal ────────────────────────────────────────────────────

function _ensureCard() {
  if (_card) return _card;
  const el = document.createElement('div');
  el.className = 'connect-summary-card hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-modal', 'true');
  el.setAttribute('aria-label', 'Call summary');
  el.innerHTML = `
    <div class="connect-summary-inner">
    <div class="connect-summary-head">
      <div class="connect-summary-avatar" aria-hidden="true"></div>
      <div class="connect-summary-headtext">
        <div class="connect-summary-peer"></div>
        <div class="connect-summary-meta"></div>
      </div>
      <button class="connect-summary-close" type="button" aria-label="Dismiss summary">&#x2715;</button>
    </div>
    <div class="connect-summary-rate-section">
      <div class="connect-summary-rate-title">How was the call?</div>
      <div class="connect-summary-rate-row">
        <button class="connect-summary-rate-btn" type="button" data-rating="1" title="Good">
          <span class="connect-summary-rate-glyph">${icon('thumbs-up', { size: 18 })}</span>
          <span class="connect-summary-rate-label">Good</span>
        </button>
        <button class="connect-summary-rate-btn" type="button" data-rating="0" title="OK">
          <span class="connect-summary-rate-glyph">${icon('minus', { size: 18 })}</span>
          <span class="connect-summary-rate-label">OK</span>
        </button>
        <button class="connect-summary-rate-btn" type="button" data-rating="-1" title="Bad">
          <span class="connect-summary-rate-glyph">${icon('thumbs-down', { size: 18 })}</span>
          <span class="connect-summary-rate-label">Bad</span>
        </button>
      </div>
      <textarea class="connect-summary-notes" rows="2"
                placeholder="Optional note (echo, dropouts, anything notable)…"
                maxlength="2000"></textarea>
      <div class="connect-summary-saved" hidden>Thanks — saved.</div>
    </div>
    <div class="connect-summary-actions">
      <button class="connect-summary-action callback" type="button">
        ${icon('phone', { size: 15 })}<span>Call back</span>
      </button>
      <button class="connect-summary-action openchat" type="button">
        ${icon('message', { size: 15 })}<span>Open chat</span>
      </button>
      <button class="connect-summary-action dismiss" type="button">Dismiss</button>
    </div>
    </div>
  `;
  document.body.appendChild(el);
  _card = el;

  el.querySelector('.connect-summary-close')
    .addEventListener('click', hidePostCallRating);
  el.querySelector('.connect-summary-action.dismiss')
    .addEventListener('click', hidePostCallRating);

  // Rating buttons.
  let lastRating = null;
  let notesTimer = null;
  const notesEl = el.querySelector('.connect-summary-notes');
  const savedEl = el.querySelector('.connect-summary-saved');

  const submit = async (rating) => {
    if (!_activeCallId) return;
    const notes = (notesEl?.value || '').slice(0, 2000);
    _disableButtons(true);
    try {
      await rateCall(_activeCallId, rating, notes);
      lastRating = rating;
      if (savedEl) {
        savedEl.hidden = false;
        savedEl.textContent = 'Thanks — saved.';
      }
      // Don't auto-hide after rating — let the user decide. Just
      // reset the dismiss timer so the card stays visible longer.
      if (_dismissTimer) { clearTimeout(_dismissTimer); }
      _dismissTimer = setTimeout(hidePostCallRating, AUTO_DISMISS_MS);
    } catch (err) {
      console.warn('connect: post-call rate failed', err);
      if (savedEl) {
        savedEl.hidden = false;
        savedEl.textContent = 'Could not save — try again.';
      }
    } finally {
      _disableButtons(false);
    }
  };

  for (const btn of el.querySelectorAll('.connect-summary-rate-btn')) {
    btn.addEventListener('click', () => {
      const rating = parseInt(btn.dataset.rating, 10);
      if (Number.isNaN(rating)) return;
      for (const b of el.querySelectorAll('.connect-summary-rate-btn')) {
        b.setAttribute('aria-pressed', 'false');
      }
      btn.setAttribute('aria-pressed', 'true');
      submit(rating);
    });
  }

  if (notesEl) {
    notesEl.addEventListener('input', () => {
      // Re-save with whatever rating's already been picked once the
      // user pauses typing. No-op if no rating has been chosen yet.
      if (lastRating === null) return;
      if (notesTimer) clearTimeout(notesTimer);
      notesTimer = setTimeout(() => submit(lastRating), 800);
    });
  }

  // Action buttons.
  el.querySelector('.connect-summary-action.callback')
    .addEventListener('click', async () => {
      const peerDid = _activePeerDid;
      const withVideo = String(_activeModalities || '').includes('video');
      hidePostCallRating();
      if (!peerDid) return;
      try {
        const mod = await import('./ui.js');
        await mod.startCall?.(peerDid, { withVideo });
      } catch (err) {
        showToast(`Call back failed: ${err?.message || 'unknown'}`, 'error');
      }
    });

  el.querySelector('.connect-summary-action.openchat')
    .addEventListener('click', async () => {
      const peerDid = _activePeerDid;
      hidePostCallRating();
      if (!peerDid) return;
      try {
        const mod = await import('./thread-panel.js');
        // The messaging panel doesn't take a peer DID directly —
        // it works in thread_ids. The cleanest path: just open the
        // panel; the contact picker inside it handles new threads.
        await mod.openMessagingPanel?.();
      } catch (err) {
        showToast(`Open chat failed: ${err?.message || 'unknown'}`, 'error');
      }
    });

  return el;
}

function _populate({ peerDid, durationSeconds, modalities }) {
  if (!_card) return;
  const peerEl = _card.querySelector('.connect-summary-peer');
  // Was the raw DID — the post-call card is the surface the user stares at
  // longest, so it was the most visible place the internal id leaked.
  const peerLabel = resolvePeerName(peerDid || '') || 'Call ended';
  if (peerEl) peerEl.textContent = peerLabel;
  const avEl = _card.querySelector('.connect-summary-avatar');
  if (avEl) avEl.textContent = _initialFor(peerLabel);
  const metaEl = _card.querySelector('.connect-summary-meta');
  if (metaEl) {
    const parts = [];
    if (String(modalities || '').includes('video')) {
      parts.push('Video call');
    } else {
      parts.push('Voice call');
    }
    if (typeof durationSeconds === 'number' && durationSeconds >= 0) {
      parts.push(_formatDuration(durationSeconds));
    }
    metaEl.textContent = parts.join(' · ');
  }
}

function _resetButtons() {
  if (!_card) return;
  _disableButtons(false);
  const saved = _card.querySelector('.connect-summary-saved');
  if (saved) { saved.hidden = true; saved.textContent = 'Thanks — saved.'; }
  for (const b of _card.querySelectorAll('.connect-summary-rate-btn')) {
    b.setAttribute('aria-pressed', 'false');
  }
  const notes = _card.querySelector('.connect-summary-notes');
  if (notes) notes.value = '';
}

function _disableButtons(disabled) {
  if (!_card) return;
  for (const btn of _card.querySelectorAll('.connect-summary-rate-btn')) {
    btn.disabled = disabled;
  }
}

function _initialFor(s) {
  const cleaned = String(s || '').trim();
  if (!cleaned) return '?';
  const beforeAt = cleaned.split('@')[0] || cleaned;
  return (beforeAt[0] || '?').toUpperCase();
}

function _formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

// Keep escapeHtml import alive — peerDid goes through textContent so we
// don't need it for unsafe interpolation, but app.js exports it from a
// shared module and tree-shake-less imports keep parity tests cleaner.
void escapeHtml;
