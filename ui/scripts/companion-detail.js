/**
 * companion-detail.js — a generic, agnostic detail overlay the companion
 * opens to SHOW the user something in depth, over whatever surface they're
 * on (chat, avatar, anywhere). Not media-specific: any companion action
 * that wants to present an item with art + facts + contextual actions
 * routes through here.
 *
 * Design intent (Matt, 2026-07-16): a companion recommendation shouldn't
 * fire-and-forget into playback. Clicking a pick opens a right-side panel
 * with the item's info (like the Files detail view, but surface-agnostic
 * and overlaid), from which the user can act (Play, …) or go BACK to the
 * options without the whole thing vanishing. This is the reusable shell;
 * callers supply a descriptor.
 *
 * Descriptor:
 *   {
 *     title:       string,               // required
 *     subtitle?:   string,
 *     badge?:      string,               // small kind chip (e.g. "Movie")
 *     coverUrl?:   string,               // poster
 *     backdropUrl?:string,               // wide hero art
 *     description?:string,
 *     fields?:     [{ label, value }],   // meta chips (runtime, year, …)
 *     note?:       string,               // honest status line (e.g. unresolved)
 *     actions?:    [{ label, primary?, onClick, disabled? }],
 *     onBack?:     () => void,           // shows a Back button when set
 *     mount?:      ({hero, body, root}) => void,
 *                  //  optional — called after render with the hero region,
 *                  //  body region, and panel root so a caller can inject
 *                  //  custom content (a live preview iframe, a code diff,
 *                  //  citation dropdowns). Keeps this shell agnostic: the
 *                  //  media path ignores it; the coder-brief path uses it.
 *   }
 *
 * Re-calling openCompanionDetail() replaces the panel in place — callers
 * open with basic info immediately, then re-open enriched once async data
 * lands (no flash, no second panel).
 */

import { escapeHtml } from './app.js';

const HOST_ID = 'companion-detail-panel';

function _ensureStyles() {
  if (document.getElementById('companion-detail-style')) return;
  const style = document.createElement('style');
  style.id = 'companion-detail-style';
  style.textContent = `
    #${HOST_ID} {
      position: fixed;
      top: 0;
      right: 0;
      height: 100vh;
      width: min(380px, 100vw);
      z-index: 900;
      display: flex;
      flex-direction: column;
      background: var(--bg-elevated, var(--bg-secondary, #16181d));
      border-left: 1px solid var(--border-color, rgba(255,255,255,0.08));
      box-shadow: -8px 0 32px rgba(0, 0, 0, 0.35);
      animation: cd-slide 220ms ease-out;
      overflow: hidden;
    }
    @keyframes cd-slide {
      from { transform: translateX(24px); opacity: 0; }
      to   { transform: translateX(0); opacity: 1; }
    }
    #${HOST_ID}.cd-leaving {
      transition: transform 180ms ease, opacity 180ms ease;
      transform: translateX(24px);
      opacity: 0;
      pointer-events: none;
    }
    .cd-scroll { overflow-y: auto; flex: 1; }
    .cd-hero {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: color-mix(in srgb, currentColor 8%, transparent);
      overflow: hidden;
    }
    .cd-hero img.cd-backdrop {
      width: 100%; height: 100%; object-fit: cover; display: block;
    }
    .cd-hero::after {
      content: ''; position: absolute; inset: 0;
      background: linear-gradient(180deg, transparent 40%, var(--bg-elevated, #16181d) 100%);
    }
    .cd-topbar {
      position: absolute; top: 0; left: 0; right: 0;
      display: flex; justify-content: space-between; padding: 10px;
      z-index: 2;
    }
    .cd-iconbtn {
      background: rgba(0,0,0,0.45);
      border: none; color: #fff; cursor: pointer;
      width: 32px; height: 32px; border-radius: 50%;
      font-size: 15px; line-height: 1;
      display: flex; align-items: center; justify-content: center;
      backdrop-filter: blur(4px);
    }
    .cd-iconbtn:hover { background: rgba(0,0,0,0.65); }
    .cd-body { padding: 0 16px 16px; margin-top: -34px; position: relative; z-index: 1; }
    .cd-poster-row { display: flex; gap: 12px; align-items: flex-end; }
    .cd-poster {
      flex: none; width: 76px; height: 110px; border-radius: 8px;
      object-fit: cover; box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      background: color-mix(in srgb, currentColor 10%, transparent);
    }
    .cd-heading { min-width: 0; padding-bottom: 4px; }
    .cd-badge {
      display: inline-block; font-size: 10px; font-weight: 600;
      letter-spacing: 0.4px; text-transform: uppercase;
      padding: 3px 7px; border-radius: 6px; margin-bottom: 6px;
      color: var(--accent-color, #7aa2f7);
      background: color-mix(in srgb, var(--accent-color, #7aa2f7) 14%, transparent);
    }
    .cd-title { font-size: 17px; font-weight: 700; color: var(--text-primary, #e8eaed); line-height: 1.2; }
    .cd-subtitle { font-size: 12.5px; color: var(--text-secondary, #9aa0a6); margin-top: 3px; }
    .cd-fields { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 0; }
    .cd-field {
      font-size: 11px; color: var(--text-secondary, #9aa0a6);
      padding: 4px 8px; border-radius: 6px;
      background: color-mix(in srgb, currentColor 8%, transparent);
    }
    .cd-desc { font-size: 13px; line-height: 1.5; color: var(--text-primary, #e8eaed); margin-top: 14px; white-space: pre-wrap; }
    .cd-note { font-size: 12px; color: var(--text-secondary, #9aa0a6); margin-top: 12px; font-style: italic; }
    .cd-actions {
      display: flex; flex-direction: column; gap: 8px;
      padding: 14px 16px; border-top: 1px solid var(--border-color, rgba(255,255,255,0.07));
    }
    .cd-actionrow { display: flex; gap: 8px; }
    .cd-btn {
      flex: 1; padding: 11px 12px; border-radius: 10px; cursor: pointer;
      font-size: 13.5px; font-weight: 600; border: 1px solid transparent;
      background: color-mix(in srgb, currentColor 10%, transparent);
      color: var(--text-primary, #e8eaed);
      transition: transform 100ms ease, background 120ms ease;
    }
    .cd-btn:hover:not(:disabled) { transform: translateY(-1px); }
    .cd-btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .cd-btn.cd-primary {
      background: var(--accent-color, #7aa2f7);
      color: var(--on-accent, #10131a);
      border-color: transparent;
    }
    .cd-btn.cd-back { flex: none; min-width: 96px; }
  `;
  document.head.appendChild(style);
}

let _actionHandlers = [];

/** Open (or replace) the companion detail panel with ``descriptor``. */
export function openCompanionDetail(descriptor) {
  if (!descriptor || !descriptor.title) return;
  _ensureStyles();

  const {
    title, subtitle = '', badge = '', coverUrl = '', backdropUrl = '',
    description = '', fields = [], note = '', actions = [], onBack = null,
    mount = null,
  } = descriptor;

  let host = document.getElementById(HOST_ID);
  const fresh = !host;
  if (fresh) {
    host = document.createElement('div');
    host.id = HOST_ID;
    host.setAttribute('role', 'dialog');
    host.setAttribute('aria-label', 'Details');
  }

  const backdrop = backdropUrl
    ? `<img class="cd-backdrop" src="${escapeHtml(backdropUrl)}" alt="" loading="lazy">`
    : '';
  const poster = coverUrl
    ? `<img class="cd-poster" src="${escapeHtml(coverUrl)}" alt="" loading="lazy">`
    : '';
  const fieldChips = (fields || [])
    .filter(f => f && f.value)
    .map(f => `<span class="cd-field">${escapeHtml(String(f.label ? f.label + ': ' : ''))}${escapeHtml(String(f.value))}</span>`)
    .join('');
  const descHtml = description ? `<div class="cd-desc">${escapeHtml(description)}</div>` : '';
  const noteHtml = note ? `<div class="cd-note">${escapeHtml(note)}</div>` : '';

  // Action buttons — handlers bound after render by index (CSP-safe).
  _actionHandlers = [];
  const backBtn = onBack
    ? `<button class="cd-btn cd-back" data-cd-back="1">← Back</button>`
    : '';
  const actionBtns = (actions || []).map((a, i) => {
    _actionHandlers[i] = a.onClick;
    const cls = a.primary ? 'cd-btn cd-primary' : 'cd-btn';
    const dis = a.disabled ? 'disabled' : '';
    return `<button class="${cls}" data-cd-action="${i}" ${dis}>${escapeHtml(a.label || '')}</button>`;
  }).join('');

  host.innerHTML = `
    <div class="cd-scroll">
      <div class="cd-hero">
        ${backdrop}
        <div class="cd-topbar">
          <span></span>
          <button class="cd-iconbtn" data-cd-close="1" aria-label="Close">✕</button>
        </div>
      </div>
      <div class="cd-body">
        <div class="cd-poster-row">
          ${poster}
          <div class="cd-heading">
            ${badge ? `<div class="cd-badge">${escapeHtml(badge)}</div>` : ''}
            <div class="cd-title">${escapeHtml(title)}</div>
            ${subtitle ? `<div class="cd-subtitle">${escapeHtml(subtitle)}</div>` : ''}
          </div>
        </div>
        ${fieldChips ? `<div class="cd-fields">${fieldChips}</div>` : ''}
        ${descHtml}
        ${noteHtml}
      </div>
    </div>
    <div class="cd-actions">
      <div class="cd-actionrow">
        ${backBtn}
        ${actionBtns}
      </div>
    </div>
  `;

  // Hide broken/absent art in JS (CSP-safe — no inline handlers).
  host.querySelectorAll('img.cd-backdrop, img.cd-poster').forEach(img => {
    img.addEventListener('error', () => { img.style.display = 'none'; }, { once: true });
  });
  host.querySelector('[data-cd-close]')?.addEventListener('click', () => closeCompanionDetail());
  host.querySelector('[data-cd-back]')?.addEventListener('click', () => {
    closeCompanionDetail({ instant: true });
    try { onBack?.(); } catch (err) { console.warn('[companion-detail] back failed', err); }
  });
  host.querySelectorAll('[data-cd-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      const fn = _actionHandlers[Number(btn.dataset.cdAction)];
      if (typeof fn === 'function') {
        try { fn(); } catch (err) { console.warn('[companion-detail] action failed', err); }
      }
    });
  });

  if (fresh) document.body.appendChild(host);

  // Optional body/hero injection — the coder-brief path mounts a live preview
  // + diff here; media callers omit it. Runs after append so the caller can
  // measure/observe the mounted nodes.
  if (typeof mount === 'function') {
    try {
      mount({
        hero: host.querySelector('.cd-hero'),
        body: host.querySelector('.cd-body'),
        root: host,
      });
    } catch (err) {
      console.warn('[companion-detail] mount hook failed', err);
    }
  }
}

/** Close the panel (soft exit unless instant). */
export function closeCompanionDetail({ instant = false } = {}) {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  if (instant) { host.remove(); return; }
  host.classList.add('cd-leaving');
  setTimeout(() => host.remove(), 190);
}

export function isCompanionDetailOpen() {
  return !!document.getElementById(HOST_ID);
}
