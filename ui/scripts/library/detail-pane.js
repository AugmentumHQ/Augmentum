/**
 * library/detail-pane.js — right column.
 *
 * Shows one item at a time. Layout, top to bottom:
 *   • Hero cover (3:4)
 *   • Editorial-serif title + meta
 *   • Primary action (brass button) — Open / Play / Read / View
 *   • Secondary icon row — Cast / Pin / Edit tags / Delete
 *   • Activity timeline (last N events)
 *   • Tags chip list with inline-edit
 *
 * No item selected → soft empty state with a hint. We DON'T render a
 * dashboard here; the main pane owns the "nothing-selected" dashboard.
 *
 * Mutations write through the API helpers and refresh the local state
 * directly — no full reload of the home/items query — so the surface
 * stays responsive even on slower devices.
 */

import { escapeHtml, showToast } from '../app.js';
import {
  listActivity, recordActivity, setPin, setTags,
} from './api.js';
import { castGame, isCastable } from './cast-launch.js';
import { renderCover } from './cover.js';
import { friendlyFormat, labelForFormat } from './types.js';

// Formats the server renders into a self-contained preview at
// /api/artifacts/{id}/preview — PDF served natively, the rest as inline-
// styled HTML (docx/epub/pptx/xlsx via python libs, md/txt/csv/json as
// styled text, html as-is). Safe to iframe same-origin under an
// allow-scripts sandbox: no cross-origin isolation dance, no authed
// external subresources for the text formats. Multi-file ZIP apps are
// deliberately NOT here — their sibling CSS/JS need authenticated
// subresource requests that break under the sandbox (that was the
// "waking / unauthorized / CSS-less" bug), so they use a screenshot hero.
const _PREVIEW_IFRAME_FORMATS = new Set([
  'pdf', 'docx', 'epub', 'pptx', 'xlsx', 'csv',
  'md', 'txt', 'json', 'rst', 'log', 'html', 'htm',
]);
const _STATIC_IMAGE_FORMATS = new Set(['png', 'jpg', 'jpeg', 'webp', 'svg', 'gif']);


// Action key picked from artifact format. The brass label is the
// strongest UX signal on this surface; pick a verb that matches the
// content type.
const PRIMARY_VERBS = {
  Apps:      'Open app',
  Games:     'Play game',
  Documents: 'Read',
  Books:     'Read',
  Notes:     'Open',
  Slides:    'Open slides',
  Sheets:    'Open sheet',
  Images:    'View',
  Other:     'Open',
};


export class DetailPane {
  constructor(host, { onChange, onBack } = {}) {
    this.host = host;
    this.onChange = onChange || (() => {});
    this.onBack = onBack || (() => {});
    this.item = null;
    this.activity = [];
    this.buildEmpty();
  }

  buildEmpty() {
    if (this._visHandler) {
      document.removeEventListener('visibilitychange', this._visHandler);
      this._visHandler = null;
    }
    this._previewIframe = null;
    this.host.classList.add('lib-detail');
    // Workbench illustration: a single accent-tinted lamp over a bench.
    // Pure line art using currentColor for the chrome and --accent /
    // --accent-subtle for the lamp so it tracks the user's theme.
    this.host.innerHTML = `
      <div class="lib-detail-empty">
        <svg class="lib-detail-empty-art" viewBox="0 0 160 160" aria-hidden="true">
          <!-- bench -->
          <path d="M20 120 L140 120" stroke="currentColor" stroke-width="1.2" fill="none" stroke-linecap="round"/>
          <path d="M28 120 L28 138 M132 120 L132 138" stroke="currentColor" stroke-width="1.2" fill="none" stroke-linecap="round"/>
          <!-- bench top edge -->
          <path d="M20 118 L140 118" stroke="currentColor" stroke-width="0.6" fill="none" opacity="0.5"/>
          <!-- lamp arm -->
          <path d="M44 120 L44 78 L78 66 L78 56" stroke="var(--accent)" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
          <!-- lamp shade -->
          <path d="M68 56 L88 56 L84 44 L72 44 Z" stroke="var(--accent)" stroke-width="1.4" fill="var(--accent-subtle)" stroke-linejoin="round"/>
          <!-- lamp base bolt -->
          <circle cx="44" cy="120" r="2.5" fill="var(--accent)" stroke="none"/>
          <!-- bench items: stack of pages -->
          <rect x="96" y="106" width="28" height="12" rx="1" stroke="currentColor" stroke-width="1" fill="none" opacity="0.7"/>
          <rect x="100" y="100" width="28" height="12" rx="1" stroke="currentColor" stroke-width="1" fill="none" opacity="0.55"/>
          <!-- light pool -->
          <ellipse cx="78" cy="118" rx="48" ry="6" fill="var(--accent-subtle)" opacity="0.5"/>
        </svg>
        <div class="lib-detail-empty-eyebrow">An empty workbench</div>
        <div class="lib-detail-empty-hint">
          Pick something from the list to see covers, actions, and history.
        </div>
      </div>
    `;
  }

  async show(item) {
    if (!item) {
      this.item = null;
      this.activity = [];
      this.buildEmpty();
      return;
    }
    this.item = item;
    this._renderShell();
    await this._loadActivity();
  }

  // ── Loading ──────────────────────────────────────────────────────

  async _loadActivity() {
    if (!this.item) return;
    // Publications (pub_*) have no per-artifact activity timeline — the
    // activity routes operate on the ``artifacts`` table and 404 for
    // pub_* ids by design (see library_routes.py). Launch tracking for
    // publications goes through /publications/{id}/launch instead. Skip
    // the call so we don't spam the console with expected 404s.
    if (this.item._isPublication) {
      this.activity = [];
      const tl0 = this.host.querySelector('.lib-activity-list');
      if (tl0) tl0.replaceWith(this._renderActivityList());
      return;
    }
    try {
      const body = await listActivity(this.item.id);
      this.activity = body.events || [];
    } catch (err) {
      console.warn('[library] activity fetch failed', err);
      this.activity = [];
    }
    const tl = this.host.querySelector('.lib-activity-list');
    if (tl) tl.replaceWith(this._renderActivityList());
  }

  // ── Rendering ────────────────────────────────────────────────────

  _populateHero(host, item) {
    const fmt = (item.format || '').toLowerCase();
    const isPub = item._isPublication
      || (typeof item.id === 'string' && item.id.startsWith('pub_'));
    // Reset variant class — _renderShell builds a fresh hero element
    // per item but this defends against re-entry on the same host.
    host.classList.remove('lib-detail-hero-cover');

    // Publications + multi-file ZIP apps → captured screenshot, not a live
    // iframe. Running a multi-file app in the hero means cross-origin /
    // authed-subresource loads that break under the sandbox; the screenshot
    // is the reliable preview and "Open" runs the real thing.
    if (isPub) {
      this._screenshotHero(
        host, item,
        `/api/library/publications/${encodeURIComponent(item.id)}`
          + `/assets/__screenshot.png`,
      );
      return;
    }
    if (fmt === 'zip' || item._type === 'app') {
      this._screenshotHero(
        host, item,
        `/api/artifacts/${encodeURIComponent(item.id)}/preview-image`,
      );
      return;
    }

    if (_STATIC_IMAGE_FORMATS.has(fmt)) {
      const img = document.createElement('img');
      img.className = 'lib-detail-preview-img';
      img.alt = item.display_name || item.filename || '';
      img.loading = 'lazy';
      img.decoding = 'async';
      img.src = `/api/artifacts/${encodeURIComponent(item.id)}/download`;
      host.appendChild(img);
      return;
    }

    // Imported raw HTML is served as a download by /preview now (untrusted —
    // see the XSS hardening), so an iframe would just blank. Route it to the
    // cover card. pdf/docx/etc are server-converted to safe output even when
    // imported, so they still render in the iframe.
    const rawImportedHtml = (fmt === 'html' || fmt === 'htm')
      && item.metadata?.imported;
    if (_PREVIEW_IFRAME_FORMATS.has(fmt) && !rawImportedHtml) {
      // Same-origin, allow-scripts sandbox. The server renders a
      // self-contained document (native PDF / inline-styled HTML), so no
      // isolated origin or token is needed — that machinery only mattered
      // for live multi-file apps, which now use the screenshot path above.
      const iframe = document.createElement('iframe');
      iframe.className = 'lib-detail-preview-iframe';
      iframe.setAttribute('sandbox', 'allow-scripts');
      iframe.loading = 'lazy';
      iframe.title = `${item.display_name || item.filename || 'Preview'} preview`;
      // Pause by clearing src when hidden; restore on show so a
      // backgrounded preview doesn't keep decoding.
      this._wirePreviewVisibility(iframe);
      host.appendChild(iframe);
      iframe.src = `/api/artifacts/${encodeURIComponent(item.id)}/preview`;
      return;
    }

    // Fallback: hand off to the shared cover renderer so the hero gets
    // the format-specific mini representation (paper card, slide
    // thumbnail, sheet grid, procedural game cover) rather than a bare
    // initial. Wrapper class lets CSS style the hero variant.
    host.classList.add('lib-detail-hero-cover');
    host.innerHTML = renderCover(item, { size: 'hero' });
  }

  _screenshotHero(host, item, url) {
    // Captured screenshot with a graceful fall-through to the shared cover
    // renderer when there's no screenshot yet (publication saved without
    // one, app not yet captured). Never paints a broken-image glyph.
    const img = document.createElement('img');
    img.className = 'lib-detail-preview-img';
    img.alt = item.display_name || item.filename || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.addEventListener('error', () => {
      if (!img.isConnected) return;
      host.classList.add('lib-detail-hero-cover');
      host.innerHTML = renderCover(item, { size: 'hero' });
    });
    img.src = url;
    host.appendChild(img);
  }

  _wirePreviewVisibility(iframe) {
    // One listener per pane lifetime. The previous handler is torn
    // down when buildEmpty() / _renderShell() blow away the iframe
    // reference; the global listener stays installed but no-ops when
    // there's nothing to pause.
    if (this._visHandler) {
      document.removeEventListener('visibilitychange', this._visHandler);
    }
    this._previewIframe = iframe;
    const visHandler = () => {
      const live = this._previewIframe;
      if (!live || !live.isConnected) return;
      if (document.hidden) {
        live.dataset.savedSrc = live.src;
        live.src = 'about:blank';
      } else if (live.dataset.savedSrc) {
        live.src = live.dataset.savedSrc;
        delete live.dataset.savedSrc;
      }
    };
    this._visHandler = visHandler;
    document.addEventListener('visibilitychange', visHandler);
  }

  _renderShell() {
    const it = this.item;
    const label = labelForFormat(it.format);
    const verb = PRIMARY_VERBS[label] || PRIMARY_VERBS.Other;
    const meta = _metaLine(it);

    this.host.innerHTML = '';
    this.host.classList.add('lib-detail');

    const back = document.createElement('button');
    back.type = 'button';
    back.className = 'lib-back-btn lib-back-to-main';
    back.setAttribute('aria-label', 'Back to list');
    back.title = 'Back';
    back.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
           stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M15 6l-6 6 6 6"/>
      </svg>
    `;
    back.addEventListener('click', () => this.onBack());
    this.host.appendChild(back);

    const hero = document.createElement('div');
    hero.className = 'lib-detail-hero';
    this._populateHero(hero, it);
    this.host.appendChild(hero);

    const head = document.createElement('div');
    head.className = 'lib-detail-head';
    head.innerHTML = `
      <h2 class="lib-detail-title">${escapeHtml(it.display_name || it.filename || 'Untitled')}</h2>
      <div class="lib-detail-meta">${escapeHtml(meta)}</div>
    `;
    this.host.appendChild(head);

    const actions = document.createElement('div');
    actions.className = 'lib-detail-actions';
    const hasSource = !!it.metadata?.source_url;
    actions.innerHTML = `
      <button type="button" class="lib-action-primary" data-action="open">
        ${escapeHtml(verb)}
      </button>
      <div class="lib-action-secondary" role="group" aria-label="Secondary actions">
        <button type="button" class="lib-icon-btn" data-action="cast" title="Cast">
          ${_icons.cast}
        </button>
        <button type="button" class="lib-icon-btn ${it.pinned ? 'pinned' : ''}"
                data-action="pin" title="${it.pinned ? 'Unpin' : 'Pin'}">
          ${_icons.pin}
        </button>
        <button type="button" class="lib-icon-btn" data-action="edit-tags" title="Edit tags">
          ${_icons.tag}
        </button>
        <div class="lib-overflow-wrap">
          <button type="button" class="lib-icon-btn" data-action="more"
                  aria-haspopup="menu" aria-expanded="false" title="More actions">
            ${_icons.more}
          </button>
          <div class="lib-overflow-menu hidden" role="menu">
            <button type="button" class="lib-overflow-item" data-action="download" role="menuitem">
              ${_icons.download}<span>Download</span>
            </button>
            <button type="button" class="lib-overflow-item" data-action="studio" role="menuitem">
              ${_icons.studio}<span>Open in Studio</span>
            </button>
            <button type="button" class="lib-overflow-item" data-action="ai" role="menuitem">
              ${_icons.ai}<span>Edit with AI</span>
            </button>
            <button type="button" class="lib-overflow-item" data-action="newtab" role="menuitem">
              ${_icons.newtab}<span>Open in new tab</span>
            </button>
            ${hasSource ? `
              <button type="button" class="lib-overflow-item" data-action="source" role="menuitem">
                ${_icons.source}<span>Visit source page</span>
              </button>
            ` : ''}
          </div>
        </div>
        <button type="button" class="lib-icon-btn destructive" data-action="delete" title="Delete">
          ${_icons.trash}
        </button>
      </div>
    `;
    this.host.appendChild(actions);
    actions.addEventListener('click', this._onActionClick);

    const tagsWrap = document.createElement('div');
    tagsWrap.className = 'lib-detail-tags-wrap';
    tagsWrap.innerHTML = `<h3 class="lib-detail-section-title">Tags</h3>`;
    tagsWrap.appendChild(this._renderTagChips());
    this.host.appendChild(tagsWrap);

    const activityWrap = document.createElement('div');
    activityWrap.className = 'lib-detail-activity-wrap';
    activityWrap.innerHTML = `<h3 class="lib-detail-section-title">Activity</h3>`;
    activityWrap.appendChild(this._renderActivityList());
    this.host.appendChild(activityWrap);
  }

  _renderActivityList() {
    const list = document.createElement('ul');
    list.className = 'lib-activity-list';
    if (!this.activity.length) {
      const li = document.createElement('li');
      li.className = 'lib-activity-empty';
      li.textContent = 'Nothing recorded yet.';
      list.appendChild(li);
      return list;
    }
    for (const ev of this.activity.slice(0, 12)) {
      const li = document.createElement('li');
      li.className = 'lib-activity-row';
      li.innerHTML = `
        <span class="lib-activity-dot" aria-hidden="true"></span>
        <span class="lib-activity-verb">${escapeHtml(_humanAction(ev.action))}</span>
        <span class="lib-activity-when">${escapeHtml(_relativeTime(ev.occurred_at))}</span>
      `;
      list.appendChild(li);
    }
    return list;
  }

  _renderTagChips() {
    const wrap = document.createElement('div');
    wrap.className = 'lib-tag-chips';
    const tags = Array.isArray(this.item.tags) ? this.item.tags : [];

    for (const t of tags) {
      const chip = document.createElement('span');
      chip.className = 'lib-tag-chip';
      chip.innerHTML = `
        <span class="lib-tag-text">${escapeHtml(t)}</span>
        <button type="button" class="lib-tag-remove"
                data-tag="${escapeHtml(t)}" aria-label="Remove tag ${escapeHtml(t)}">×</button>
      `;
      wrap.appendChild(chip);
    }

    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'lib-tag-add';
    addBtn.textContent = '+ Add tag';
    wrap.appendChild(addBtn);

    wrap.addEventListener('click', (ev) => {
      const remove = ev.target.closest('.lib-tag-remove');
      if (remove) {
        this._removeTag(remove.dataset.tag);
        return;
      }
      if (ev.target === addBtn) {
        this._openTagInput(wrap, addBtn);
      }
    });

    return wrap;
  }

  _openTagInput(wrap, addBtn) {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'lib-tag-input';
    input.placeholder = 'New tag…';
    input.maxLength = 32;
    wrap.insertBefore(input, addBtn);
    addBtn.style.display = 'none';
    input.focus();

    const commit = () => {
      const v = input.value.trim();
      input.remove();
      addBtn.style.display = '';
      if (v) this._addTag(v);
    };
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); commit(); }
      else if (ev.key === 'Escape') {
        input.remove();
        addBtn.style.display = '';
      }
    });
    input.addEventListener('blur', commit);
  }

  // ── Mutations ────────────────────────────────────────────────────

  async _addTag(tag) {
    const next = [...(this.item.tags || []), tag];
    await this._saveTags(next);
  }

  async _removeTag(tag) {
    const next = (this.item.tags || []).filter(t => t !== tag);
    await this._saveTags(next);
  }

  async _saveTags(next) {
    try {
      const body = await setTags(this.item.id, next);
      this.item.tags = body.tags;
      // Re-render only the tag chips; everything else stays put.
      const wrap = this.host.querySelector('.lib-tag-chips');
      if (wrap) wrap.replaceWith(this._renderTagChips());
      this.onChange({ kind: 'tags', item: this.item });
    } catch (err) {
      console.warn('[library] save tags failed', err);
    }
  }

  async _togglePin() {
    const next = !this.item.pinned;
    const btn = this.host.querySelector('[data-action="pin"]');
    try {
      const body = await setPin(this.item.id, next);
      this.item.pinned = body.pinned;
      btn?.classList.toggle('pinned', this.item.pinned);
      btn?.setAttribute('title', this.item.pinned ? 'Unpin' : 'Pin');
      // Promotion confirmation: brass ripple radiates from the button.
      // 480ms total; class self-clears so repeated pins re-trigger.
      if (this.item.pinned && btn) {
        btn.classList.remove('pulse');
        // Force reflow so the next class add re-runs the animation.
        void btn.offsetWidth;
        btn.classList.add('pulse');
        setTimeout(() => btn.classList.remove('pulse'), 600);
      }
      this.onChange({ kind: 'pin', item: this.item });
    } catch (err) {
      console.warn('[library] toggle pin failed', err);
    }
  }

  async _openItem() {
    const item = this.item;
    if (!item) return;
    // Dispatch (tracking + kind routing) lives in library/open-item.js
    // — shared with the companion's game.launch channel and candidate
    // cards, so every surface opens items through one implementation.
    const m = await import('./open-item.js');
    await m.openLibraryItem(item);
  }

  _onActionClick = (ev) => {
    const btn = ev.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    const item = this.item;
    if (!item) return;

    // Most actions live inline; the "more" button toggles an overflow
    // popover that hosts the rest. When an overflow item fires it
    // bubbles back here as data-action="<verb>" so the dispatch table
    // stays flat. Close the menu after dispatch.
    const closeOverflow = () => {
      const menu = this.host.querySelector('.lib-overflow-menu');
      const trigger = this.host.querySelector('[data-action="more"]');
      menu?.classList.add('hidden');
      trigger?.setAttribute('aria-expanded', 'false');
    };

    if (action === 'open')       { this._openItem();   return; }
    if (action === 'pin')        { this._togglePin();  return; }
    if (action === 'edit-tags') {
      const addBtn = this.host.querySelector('.lib-tag-add');
      const wrap   = this.host.querySelector('.lib-tag-chips');
      if (addBtn && wrap) this._openTagInput(wrap, addBtn);
      return;
    }
    if (action === 'more') {
      const menu = this.host.querySelector('.lib-overflow-menu');
      const willOpen = menu?.classList.contains('hidden');
      menu?.classList.toggle('hidden');
      btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
      if (willOpen) {
        // Close on outside-click. One-shot — the dispatch above re-attaches.
        const onDocClick = (e) => {
          if (!e.target.closest('.lib-overflow-wrap')) {
            closeOverflow();
            document.removeEventListener('click', onDocClick, true);
          }
        };
        // Defer so this click doesn't immediately match the listener.
        setTimeout(() => document.addEventListener('click', onDocClick, true), 0);
      }
      return;
    }

    if (action === 'cast') {
      // Cast dispatch: pick a receiver, POST /api/cast/send with a
      // play.kiosk URL. Activity row only lands on success so the
      // timeline doesn't lie about a failed cast.
      if (!isCastable(item)) {
        // Non-castable kinds fall through to a "not supported" toast
        // inside castGame. Future: cast PDFs through a viewer surface.
      }
      castGame(item)
        .then((resp) => {
          recordActivity(item.id, 'cast', {
            surface: 'tv', payload: { receiver_id: resp.receiver_id },
          })
            .then(() => this._loadActivity())
            .catch(() => {});
        })
        .catch((err) => {
          if (err && err.message === 'cancelled') return;
          // showToast already happened inside castGame on real errors.
        });
      return;
    }

    // ── Overflow menu items ───────────────────────────────────────────
    if (action === 'download') {
      closeOverflow();
      this._downloadItem();
      return;
    }
    if (action === 'studio') {
      closeOverflow();
      import('../studio.js').then(m => m.openStudio(item.id, { fromLibrary: true }))
        .catch(err => {
          console.error('[library] studio open failed', err);
          showToast(`Failed to open in Studio: ${err?.message || 'Unknown error'}`, 'error');
        });
      return;
    }
    if (action === 'ai') {
      closeOverflow();
      // Workspace "work" mode is the agentic editor — same module as the
      // Open dispatcher's app path, just a different mode flag.
      import('../workspace.js').then(m => m.openWorkspace(item, 'work'))
        .catch(err => {
          console.error('[library] workspace open failed', err);
          showToast(`Failed to open workspace: ${err?.message || 'Unknown error'}`, 'error');
        });
      return;
    }
    if (action === 'newtab') {
      closeOverflow();
      this._openInNewTab();
      return;
    }
    if (action === 'source') {
      closeOverflow();
      const url = item.metadata?.source_url;
      if (url) window.open(url, '_blank', 'noopener,noreferrer');
      else showToast('No source URL for this item', 'warning');
      return;
    }

    if (action === 'delete') {
      this._confirmDelete();
    }
  };

  _downloadItem() {
    const it = this.item;
    if (!it?.id) return;
    // Publications download their bundle.zip from the publications route;
    // artifacts from the artifact download route.
    const a = document.createElement('a');
    a.href = _isPub(it)
      ? `/api/library/publications/${encodeURIComponent(it.id)}/download`
      : `/api/artifacts/${encodeURIComponent(it.id)}/download`;
    a.download = it.filename || 'artifact';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async _openInNewTab() {
    const it = this.item;
    if (!it?.id) return;
    // Publications open through their sandboxed launcher.
    if (_isPub(it)) {
      window.open(`/api/library/play/${encodeURIComponent(it.id)}`, '_blank');
      return;
    }
    // For apps we prefer an assembled HTML blob so multi-file ZIPs
    // render correctly in a standalone tab. Single-file apps and every
    // non-app type fall through to the server-rendered /preview URL.
    if (it._type === 'app') {
      try {
        if (!it.source_json) {
          const r = await fetch(`/api/artifacts/${encodeURIComponent(it.id)}`);
          if (r.ok) it.source_json = (await r.json()).source_json;
        }
        if (it.source_json) {
          const src = typeof it.source_json === 'string'
            ? JSON.parse(it.source_json)
            : it.source_json;
          if (src?.files?.length) {
            const { assembleProject } = await import('../assemble.js');
            const html = assembleProject(src.files);
            if (html) {
              const blob = new Blob([html], { type: 'text/html' });
              window.open(URL.createObjectURL(blob), '_blank');
              return;
            }
          }
        }
      } catch { /* fall through to preview URL */ }
    }
    window.open(`/api/artifacts/${encodeURIComponent(it.id)}/preview`, '_blank');
  }

  async _confirmDelete() {
    const it = this.item;
    if (!it?.id) return;
    // Pinned games go through the games unpin endpoint so user_settings
    // (saved progress) gets cleaned up alongside the artifact. Generic
    // delete leaves ``game_save:{id}`` orphaned.
    const isGame = it._type === 'game' && !_isPub(it);
    const isPub = _isPub(it);
    const name = it.display_name || it.filename || 'item';
    const promptText = isGame
      ? `Unpin "${name}" from your library?\n\nThis removes saved progress. You can re-pin it from the Game Portal.`
      : `Delete "${name}"?\n\nThis cannot be undone.`;
    if (!confirm(promptText)) return;
    try {
      // Three delete namespaces: games unpin (cleans save state),
      // publications (pub_ ids), and generic artifacts.
      const url = isGame
        ? `/api/games/pin/${encodeURIComponent(it.id)}`
        : isPub
          ? `/api/library/publications/${encodeURIComponent(it.id)}`
          : `/api/artifacts/${encodeURIComponent(it.id)}`;
      const resp = await fetch(url, { method: 'DELETE', credentials: 'same-origin' });
      if (!resp.ok && resp.status !== 404) {
        throw new Error(`HTTP ${resp.status}`);
      }
      showToast(isGame ? `Unpinned ${name}` : `Deleted ${name}`, 'success');
      // Tell the orchestrator so the main pane + sidebar counts refresh
      // and the active item is dropped from the detail pane.
      this.onChange({ kind: 'delete', item: it });
      this.item = null;
      this.buildEmpty();
    } catch (err) {
      console.error('[library] delete failed', err);
      showToast(`Failed to delete: ${err?.message || 'Unknown error'}`, 'error');
    }
  }
}


// ── Helpers ────────────────────────────────────────────────────────

// A library item is a publication (coder "Save to Library") when its id
// carries the pub_ prefix. classifyItem stamps ``_isPublication`` at the
// API layer, but fall back to the prefix so this is robust on any path.
function _isPub(item) {
  return !!(item && (item._isPublication
    || (typeof item.id === 'string' && item.id.startsWith('pub_'))));
}

const _icons = {
  cast: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 8V6a1 1 0 0 1 1-1h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1h-6"/><path d="M3 14a4 4 0 0 1 4 4"/><path d="M3 11a7 7 0 0 1 7 7"/><circle cx="4" cy="20" r="0.7" fill="currentColor"/></svg>',
  pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2l2.5 6.5L21 9.7l-5 4.6L17.3 21 12 17.7 6.7 21 8 14.3 3 9.7l6.5-1.2z"/></svg>',
  tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.6 13.6L13 21.2a2 2 0 0 1-2.8 0L2.8 13.8a2 2 0 0 1 0-2.8L10.4 3.4A2 2 0 0 1 11.8 3H20a1 1 0 0 1 1 1v8.2a2 2 0 0 1-.4 1.4z"/><circle cx="16" cy="8" r="1.2" fill="currentColor"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/><path d="M5 7l1 13a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-13"/></svg>',
  more: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/></svg>',
  studio: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 17.5L13 7.5l3.5 3.5L6.5 21H3v-3.5z"/><path d="M14.5 6l2-2 3.5 3.5-2 2z"/></svg>',
  ai: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l1.7 4.3L18 9l-4.3 1.7L12 15l-1.7-4.3L6 9l4.3-1.7z"/><path d="M19 14l.85 2.15L22 17l-2.15.85L19 20l-.85-2.15L16 17l2.15-.85z"/></svg>',
  newtab: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></svg>',
  source: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a13 13 0 0 1 0 18a13 13 0 0 1 0-18"/></svg>',
};


function _metaLine(item) {
  const bits = [];
  const fmtLabel = friendlyFormat(item);
  if (fmtLabel) bits.push(fmtLabel);
  if (item.size_bytes) bits.push(_formatBytes(item.size_bytes));
  if (item.last_opened_at) bits.push(`opened ${_relativeTime(item.last_opened_at)}`);
  else if (item.created_at) bits.push(`added ${_relativeTime(item.created_at)}`);
  return bits.join(' · ');
}

function _formatBytes(n) {
  if (!n) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = Number(n); let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return v >= 10 || i === 0 ? `${v.toFixed(0)} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`;
}

function _relativeTime(iso) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const diffSec = Math.max(1, (Date.now() - t) / 1000);
  if (diffSec < 90) return 'just now';
  const diffMin = diffSec / 60;
  if (diffMin < 90) return `${Math.round(diffMin)}m ago`;
  const diffHr = diffMin / 60;
  if (diffHr < 36) return `${Math.round(diffHr)}h ago`;
  const diffDay = diffHr / 24;
  if (diffDay < 14) return `${Math.round(diffDay)}d ago`;
  const diffWk = diffDay / 7;
  if (diffWk < 12) return `${Math.round(diffWk)}w ago`;
  return new Date(t).toLocaleDateString();
}

function _humanAction(action) {
  switch (action) {
    case 'open':  return 'Opened';
    case 'cast':  return 'Cast';
    case 'edit':  return 'Edited';
    case 'pin':   return 'Pinned';
    case 'unpin': return 'Unpinned';
    case 'tag':   return 'Tagged';
    default:      return action || 'Activity';
  }
}
