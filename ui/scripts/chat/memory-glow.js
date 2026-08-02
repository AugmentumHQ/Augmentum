/* ==========================================================================
   Chat Module — Memory Constellation Glow + Notification Polling
   Drives the memory glow indicator, margin marks, and memory notifications
   ========================================================================== */

import { escapeHtml } from '../app.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _memPollTimer = null;
let _memPollCount = 0;
let _memNoticeTimer = null;
let _memCurrentNotice = null;
const _memNoticeQueue = [];
const _MEM_POLL_MAX = 40;
const _MEM_POLL_INTERVAL = 3000;
const _MEM_NOTICE_DURATION = 9000;

// ---------------------------------------------------------------------------
// Glow indicator
// ---------------------------------------------------------------------------

/** Show constellation glow if memory is enabled. Called on init + mode change. */
export async function memGlowInit(mode) {
  const glow = document.getElementById('memory-glow');
  if (!glow) return;
  if (mode === 'narrative') { glow.classList.add('hidden'); return; }
  try {
    const resp = await fetch('/v1/memory/context-preview');
    if (!resp.ok) { glow.classList.add('hidden'); return; }
    const data = await resp.json();
    if (!data.enabled || data.total_memories === 0) { glow.classList.add('hidden'); return; }
    glow.classList.remove('hidden');
  } catch { glow.classList.add('hidden'); }
}

/** Set glow to recalling state (before sending message). */
export function memGlowRecalling() {
  const glow = document.getElementById('memory-glow');
  if (glow) { glow.classList.add('recalling'); glow.classList.remove('learned'); }
}

/** Clear recalling state (after response starts streaming). */
export function memGlowIdle() {
  const glow = document.getElementById('memory-glow');
  if (glow) glow.classList.remove('recalling');
}

/** Flash learned state (new memory stored). */
export function memGlowLearned() {
  const glow = document.getElementById('memory-glow');
  if (!glow) return;
  glow.classList.remove('hidden');
  glow.classList.remove('recalling');
  glow.classList.add('learned');
  setTimeout(() => glow.classList.remove('learned'), 2200);
}

/** Wire click handler — open memory settings page. */
export function memGlowClick() {
  const glow = document.getElementById('memory-glow');
  if (glow) glow.addEventListener('click', () => {
    document.dispatchEvent(new CustomEvent('augmentum:open-settings', { detail: { tab: 'memory' } }));
  });
}

// ---------------------------------------------------------------------------
// Notification polling (drives glow + margin marks + toast)
// ---------------------------------------------------------------------------

/** Start short-lived polling for memory notifications. */
export function memStartPolling() {
  _memPollCount = 0;
  if (_memPollTimer) clearInterval(_memPollTimer);
  _memPoll();
  _memPollTimer = setInterval(_memPoll, _MEM_POLL_INTERVAL);
}

/** Stop polling. */
export function memStopPolling() {
  if (_memPollTimer) {
    clearInterval(_memPollTimer);
    _memPollTimer = null;
  }
}

async function _memPoll() {
  _memPollCount++;
  if (_memPollCount > _MEM_POLL_MAX) {
    clearInterval(_memPollTimer);
    _memPollTimer = null;
    // Remove extracting marks that never confirmed
    document.querySelectorAll('[data-mem-extracting]').forEach(el => {
      el.removeAttribute('data-mem-extracting');
    });
    return;
  }

  try {
    const resp = await fetch('/v1/memory/notifications');
    if (!resp.ok) return;
    const data = await resp.json();
    const items = data.notifications || [];

    if (items.length > 0) {
      // Flash glow
      memGlowLearned();

      // Upgrade margin marks from extracting -> learned
      document.querySelectorAll('[data-mem-extracting]').forEach(el => {
        el.removeAttribute('data-mem-extracting');
        el.setAttribute('data-mem-learned', items[0].content || '');
      });

      _enqueueMemoryNotices(items);
    }
  } catch { /* silent */ }
}

/** Add extracting mark to the most recent user message. */
export function memMarkExtracting() {
  const msgs = document.querySelectorAll('.response-block.user-message');
  const last = msgs[msgs.length - 1];
  if (last && !last.hasAttribute('data-mem-learned')) {
    last.setAttribute('data-mem-extracting', '');
  }
}

function _enqueueMemoryNotices(items) {
  for (const item of items) {
    if (!item || !item.id) continue;
    if (_memCurrentNotice?.id === item.id) continue;
    if (_memNoticeQueue.some(queued => queued.id === item.id)) continue;
    _memNoticeQueue.push(item);
  }
  _showNextMemoryNotice();
}

function _showNextMemoryNotice() {
  if (_memCurrentNotice || _memNoticeQueue.length === 0) return;
  _memCurrentNotice = _memNoticeQueue.shift();
  _renderMemoryNotice(_memCurrentNotice);
}

function _dismissMemoryNotice() {
  const toast = document.getElementById('memory-toast');
  const glow = document.getElementById('memory-glow');
  if (_memNoticeTimer) {
    clearTimeout(_memNoticeTimer);
    _memNoticeTimer = null;
  }
  if (toast) toast.classList.add('hidden');
  _memCurrentNotice = null;
  if (_memNoticeQueue.length === 0 && glow) glow.classList.remove('notice');
  setTimeout(_showNextMemoryNotice, 140);
}

/** Show a dismissable memory notification anchored to the composer. */
async function _resolveNotice(id, action) {
  // action: 'approve' (keep → ACTIVE) | 'dismiss' (forget). Reuses the
  // existing review endpoints so a tap decides it in place — no navigation.
  try {
    await fetch(`/v1/memory/notifications/${id}/${action}`, { method: 'POST' });
    if (action === 'approve') memGlowLearned();
  } catch { /* best-effort — the panel still has it */ }
}

function _renderMemoryNotice(item) {
  const toast = document.getElementById('memory-toast');
  if (!toast) return;
  const glow = document.getElementById('memory-glow');
  const isProvisional = item.tier === 'provisional';
  const content = String(item.content || '').trim();
  // "Why" line — present on belief OFFERS (she noticed something and is
  // asking). Reads as her observation ("You made a playlist …").
  const why = String(item.evidence || '').trim();

  if (glow) {
    glow.classList.remove('hidden');
    glow.classList.add('notice');
  }

  const icon = `
    <div class="memory-toast-icon" aria-hidden="true">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">
        <circle cx="4" cy="4" r="1.8" opacity="0.9"/>
        <circle cx="12" cy="6" r="1.5" opacity="0.6"/>
        <circle cx="8" cy="13" r="1.5" opacity="0.7"/>
        <line x1="4" y1="4" x2="12" y2="6" stroke="currentColor" stroke-width="1" opacity="0.35" fill="none"/>
        <line x1="4" y1="4" x2="8" y2="13" stroke="currentColor" stroke-width="1" opacity="0.35" fill="none"/>
      </svg>
    </div>`;
  const closeBtn = `
    <button class="memory-toast-close" type="button" data-mem-close aria-label="Dismiss">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
        <path d="M4 4l8 8"/><path d="M12 4l-8 8"/>
      </svg>
    </button>`;

  if (isProvisional) {
    // A decision she wants from you — her voice, the why, and one yes/no.
    const whyLine = why ? `<div class="memory-toast-why">${escapeHtml(why.slice(0, 100))}</div>` : '';
    toast.innerHTML = `
      ${icon}
      <div class="memory-toast-copy">
        <div class="memory-toast-title">Want me to remember this?</div>
        <div class="memory-toast-content">${escapeHtml(content.slice(0, 120))}</div>
        ${whyLine}
        <div class="memory-toast-choices">
          <button class="memory-toast-action" type="button" data-mem-keep>Keep</button>
          <button class="memory-toast-action ghost" type="button" data-mem-skip>Not now</button>
        </div>
      </div>
      ${closeBtn}`;
    toast.querySelector('[data-mem-keep]')?.addEventListener('click', () => {
      _resolveNotice(item.id, 'approve');
      _dismissMemoryNotice();
    });
    toast.querySelector('[data-mem-skip]')?.addEventListener('click', () => {
      _resolveNotice(item.id, 'dismiss');
      _dismissMemoryNotice();
    });
  } else {
    // Routine FYI — quiet, with a way to look if curious.
    toast.innerHTML = `
      ${icon}
      <div class="memory-toast-copy">
        <div class="memory-toast-title">Saved to memory</div>
        <div class="memory-toast-content">${escapeHtml(content.slice(0, 120))}</div>
      </div>
      <button class="memory-toast-action" type="button" data-mem-open>View</button>
      ${closeBtn}`;
    toast.querySelector('[data-mem-open]')?.addEventListener('click', () => {
      _dismissMemoryNotice();
      document.dispatchEvent(new CustomEvent('augmentum:open-settings', { detail: { tab: 'memory' } }));
    });
  }
  toast.querySelector('[data-mem-close]')?.addEventListener('click', _dismissMemoryNotice);
  toast.classList.remove('hidden');
  // Remove and re-add to restart animation
  toast.style.animation = 'none';
  toast.offsetHeight; // trigger reflow
  toast.style.animation = '';
  if (_memNoticeTimer) clearTimeout(_memNoticeTimer);
  _memNoticeTimer = setTimeout(_dismissMemoryNotice, _MEM_NOTICE_DURATION);
}
