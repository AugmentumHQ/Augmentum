/**
 * Shared sentence-breakdown popover.
 *
 * Lifted out of companion_dialogue.js so any chat surface can open it
 * for any text. The companion popover is doing more (contextual word-
 * level meaning via LLM + add-to-queue + speak); this one is the
 * lightweight version: it tokenises the text against the dictionary,
 * shows surface + reading + first gloss for each matched token, and
 * lets you tap any token to add it to your queue.
 *
 * Why a separate popover instead of lifting companion's directly?
 *   - Companion's version is closure-coupled to lang/voice/pool/etc.
 *     Lifting cleanly = a real refactor.
 *   - This is the safer first step toward a unified breakdown surface:
 *     ships value now, leaves the deeper "click a single word for
 *     contextual meaning" path open as Phase 4-proper later.
 *
 * Open via `openBreakdownPopover({text, lang, anchor})`. The anchor
 * argument is the DOM element clicked, used to position the popover.
 */

import { escapeHtml } from '../app.js';

const _POP_ID = 'augmentum-bd-pop';

export async function openBreakdownPopover({ text, lang, anchor }) {
  if (!text || !lang) return;
  // Cap to the breakdown endpoint's hard limit so a 5KB bubble doesn't
  // 400 — and the popover stays usable rather than scrolling forever.
  const q = String(text).trim().slice(0, 200);

  document.getElementById(_POP_ID)?.remove();
  const pop = document.createElement('div');
  pop.id = _POP_ID;
  pop.className = 'augmentum-bd-pop';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Sentence breakdown');
  pop.innerHTML = `<div class="augmentum-bd-loading">Breaking down…</div>`;
  document.body.appendChild(pop);
  _position(pop, anchor);

  // Dismiss on outside click — registered next tick so the click that
  // opened the popover doesn't immediately close it.
  setTimeout(() => {
    function off(e) {
      if (!pop.contains(e.target)) {
        pop.remove();
        document.removeEventListener('mousedown', off);
        document.removeEventListener('keydown', onKey);
      }
    }
    function onKey(e) {
      if (e.key === 'Escape') {
        pop.remove();
        document.removeEventListener('mousedown', off);
        document.removeEventListener('keydown', onKey);
      }
    }
    document.addEventListener('mousedown', off);
    document.addEventListener('keydown', onKey);
  }, 0);

  let tokens = [];
  try {
    const r = await fetch(`/api/learning/breakdown/${encodeURIComponent(lang)}?q=${encodeURIComponent(q)}`);
    if (r.ok) {
      const j = await r.json();
      tokens = Array.isArray(j.tokens) ? j.tokens : [];
    }
  } catch { /* fall through to empty state */ }

  if (!tokens.length) {
    pop.innerHTML = `
      <div class="augmentum-bd-empty">
        Couldn't break this down — no dictionary matches.
        <button type="button" class="augmentum-bd-close" aria-label="Close">×</button>
      </div>`;
    pop.querySelector('.augmentum-bd-close')?.addEventListener('click', () => pop.remove());
    return;
  }

  const matched = tokens.filter(t => t && t.matched);
  const rows = matched.map((t) => {
    const surface = t.surface || t.text || '';
    const reading = t.reading && t.reading !== surface ? t.reading : '';
    const gloss = Array.isArray(t.glosses) && t.glosses.length ? t.glosses[0] : '';
    const wordId = t.word_id || '';
    return `
      <li class="augmentum-bd-row" data-word-id="${escapeHtml(wordId)}" data-surface="${escapeHtml(surface)}">
        <span class="augmentum-bd-surface">${escapeHtml(surface)}</span>
        ${reading ? `<span class="augmentum-bd-reading">${escapeHtml(reading)}</span>` : ''}
        <span class="augmentum-bd-gloss">${escapeHtml(gloss)}</span>
        <button type="button" class="augmentum-bd-add" title="Save to my words">+</button>
      </li>`;
  }).join('');

  pop.innerHTML = `
    <div class="augmentum-bd-head">
      <div class="augmentum-bd-title">Sentence breakdown</div>
      <button type="button" class="augmentum-bd-close" aria-label="Close">×</button>
    </div>
    <ul class="augmentum-bd-list">${rows || '<li class="augmentum-bd-empty-row">No matched tokens.</li>'}</ul>`;

  pop.querySelector('.augmentum-bd-close')?.addEventListener('click', () => pop.remove());

  // Delegated click — each + button saves the row's word to the user's
  // SRS queue. Idempotent (re-adding returns added=false), so we use a
  // visual flag rather than disabling so retries on transient failures
  // still work.
  pop.querySelector('.augmentum-bd-list')?.addEventListener('click', async (e) => {
    const btn = e.target.closest('.augmentum-bd-add');
    if (!btn) return;
    const row = btn.closest('.augmentum-bd-row');
    if (!row) return;
    const wordId = row.dataset.wordId;
    if (!wordId) {
      btn.textContent = '?';
      btn.title = 'No word_id resolved';
      return;
    }
    btn.textContent = '…';
    try {
      const r = await fetch('/api/learning/vocab/add', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang, word_id: wordId, source_surface: 'chat_breakdown' }),
      });
      if (r.ok) {
        const j = await r.json();
        btn.textContent = '✓';
        btn.classList.add('augmentum-bd-added');
        btn.title = j.added ? 'Saved to your queue' : 'Already in your queue';
      } else {
        btn.textContent = '!';
        btn.title = `Save failed (${r.status})`;
      }
    } catch {
      btn.textContent = '!';
      btn.title = 'Save failed (network)';
    }
  });
}

function _position(pop, anchor) {
  // Anchor-relative placement, falling back to viewport-center when
  // the anchor isn't a valid element (e.g. event from a stale node).
  const POP_MAX_W = 380;
  if (anchor && typeof anchor.getBoundingClientRect === 'function') {
    const rect = anchor.getBoundingClientRect();
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - POP_MAX_W - 8));
    const top = rect.bottom + 6;
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
  } else {
    pop.style.left = `calc(50% - ${POP_MAX_W / 2}px)`;
    pop.style.top = '20%';
  }
}
