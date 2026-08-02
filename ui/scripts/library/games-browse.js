/**
 * library/games-browse.js — Game Portal browser.
 *
 * Reuses the existing source registry (``library-game-sources.js``) so
 * library doesn't fork the 2,300-line js13k / marketplace / emulator
 * fetch+launch logic. We override the shared browse-card renderer via
 * ``setSharedBrowseCardRenderer`` with a library-flavored card markup
 * — paper chip + brass primary action — so the visual register stays
 * consistent with the rest of the surface.
 *
 * Lifecycle:
 *   show(host, settings)  — fetch enabled sources, render source pills,
 *                            load first page of the default source.
 *   destroy()             — abort any in-flight fetch.
 *
 * Pin path: each card carries ``data-pin-src`` + ``data-pin-sid``; a
 * delegated click on the host POSTs to ``/api/games/pin`` and emits
 * ``onPinned`` so the parent can refresh the home payload.
 */

import { escapeHtml } from '../app.js';
import {
  GAME_SOURCES,
  enabledSources,
  ensurePreviewHTML,
  fetchEmulatorBiosStatus,
  getSource,
  importEmulatorRomsFromEntries,
  openBiosVault,
  openTitleRename,
  openCoverPicker,
  openSystemPicker,
  removeEmulatorTitle,
  setSharedBrowseCardRenderer,
} from '../library-game-sources.js';


// Install the library card renderer for any source that delegates to
// the shared one (js13k, marketplace). Streamed + emulator carry their
// own renderCard so are untouched here.
setSharedBrowseCardRenderer(_renderCard);


export class GamesBrowse {
  constructor(host, { onPinned, getSettings } = {}) {
    this.host = host;
    this.onPinned = onPinned || (() => {});
    this.getSettings = getSettings || (() => ({}));
    this.activeSourceId = '';
    this.pageData = { items: [], hasMore: false };
    this.loading = false;
    this._abort = null;
    // Per-source filter state. Today only the emulator source consumes
    // this; future sources slot their own filter shapes in by id.
    this._filters = {
      emulator: { systemId: 'all', biosReadyOnly: false, sort: 'recent' },
    };
    this._biosStatusBySystem = null;   // cached after first emulator load
    this._buildShell();
  }

  _buildShell() {
    this.host.innerHTML = '';
    this.host.classList.add('lib-games-browse');

    this._tabs = document.createElement('div');
    this._tabs.className = 'lib-games-tabs';
    this.host.appendChild(this._tabs);

    this._subtitle = document.createElement('div');
    this._subtitle.className = 'lib-games-subtitle';
    this.host.appendChild(this._subtitle);

    this._filterBar = document.createElement('div');
    this._filterBar.className = 'lib-games-filterbar';
    this.host.appendChild(this._filterBar);

    this._grid = document.createElement('div');
    this._grid.className = 'lib-games-grid';
    this.host.appendChild(this._grid);

    this._loadMoreWrap = document.createElement('div');
    this._loadMoreWrap.className = 'lib-games-loadmore-wrap';
    this.host.appendChild(this._loadMoreWrap);

    this._grid.addEventListener('click', this._onGridClick);
    this._tabs.addEventListener('click', (ev) => {
      const pill = ev.target.closest('[data-source]');
      if (!pill) return;
      this.setActiveSource(pill.dataset.source);
    });
  }

  async show() {
    const settings = await this._resolveSettings();
    const sources = enabledSources(settings);
    if (!sources.length) {
      this.host.classList.add('empty');
      this._tabs.innerHTML = '';
      this._subtitle.innerHTML = '';
      this._grid.innerHTML = `
        <div class="lib-main-state lib-state-empty">
          <div class="lib-empty-line">No game sources enabled.</div>
          <div class="lib-empty-hint">Turn on Game Portal in Settings to browse.</div>
        </div>
      `;
      this._loadMoreWrap.innerHTML = '';
      return;
    }
    this.host.classList.remove('empty');

    this._tabs.innerHTML = sources.map(src => `
      <button type="button" class="lib-games-pill ${src.id === this.activeSourceId ? 'active' : ''}"
              data-source="${escapeHtml(src.id)}">
        <span class="lib-games-pill-label">${escapeHtml(src.label)}</span>
        ${src.hint
          ? `<span class="lib-games-pill-hint">${escapeHtml(src.hint)}</span>`
          : ''}
      </button>
    `).join('');

    // Default to first source if no active selection, or if the
    // previously-active source got disabled.
    if (!sources.find(s => s.id === this.activeSourceId)) {
      await this.setActiveSource(sources[0].id);
    } else {
      await this._loadFirstPage();
    }
  }

  async setActiveSource(id) {
    if (this.activeSourceId === id && this.pageData.items.length) return;
    this.activeSourceId = id;
    this._tabs.querySelectorAll('[data-source]').forEach(el => {
      el.classList.toggle('active', el.dataset.source === id);
    });
    const src = getSource(id);
    this._subtitle.textContent = src?.subtitle || '';
    // Emulator source warms BIOS status once so the filter chips and
    // per-row "needs BIOS" badges render correctly.
    if (id === 'emulator' && this._biosStatusBySystem === null) {
      try {
        this._biosStatusBySystem = await fetchEmulatorBiosStatus();
      } catch (err) {
        console.warn('[library] BIOS status fetch failed', err);
        this._biosStatusBySystem = {};
      }
    }
    await this._loadFirstPage();
  }

  async _loadFirstPage() {
    const src = getSource(this.activeSourceId);
    if (!src) return;
    this.pageData = { items: [], hasMore: false };
    this._page = 1;
    this._renderLoading();
    try {
      const result = await src.fetch({ sort: 'newest', page: 1 });
      this.pageData = result || { items: [], hasMore: false };
      this._renderGrid();
    } catch (err) {
      console.warn('[library] games fetch failed', err);
      this._renderError(err.message || 'Fetch failed.');
    }
  }

  async _loadNextPage() {
    const src = getSource(this.activeSourceId);
    if (!src || !this.pageData.hasMore) return;
    this._page = (this._page || 1) + 1;
    try {
      const more = await src.fetch({ sort: 'newest', page: this._page });
      this.pageData = {
        items: [...this.pageData.items, ...(more.items || [])],
        hasMore: !!more.hasMore,
      };
      this._renderGrid();
    } catch (err) {
      console.warn('[library] games next-page failed', err);
    }
  }

  _renderLoading() {
    this._grid.innerHTML = `<div class="lib-main-state lib-state-loading">Loading…</div>`;
    this._loadMoreWrap.innerHTML = '';
  }

  _renderError(msg) {
    this._grid.innerHTML = `
      <div class="lib-main-state lib-state-empty">
        <div class="lib-empty-line">Couldn't load games.</div>
        <div class="lib-empty-hint">${escapeHtml(msg)}</div>
      </div>
    `;
    this._loadMoreWrap.innerHTML = '';
  }

  _renderGrid() {
    const src = getSource(this.activeSourceId);
    let items = this.pageData.items || [];

    // Source-defined filter bar (today: emulator only). Rendered above
    // the grid; we re-attach handlers each render so the chips can
    // re-flow if BIOS status changes.
    if (src.renderFilters && src.applyFilters) {
      const fState = this._filters[this.activeSourceId];
      const filtersHTML = src.renderFilters(items, fState, this._biosStatusBySystem);
      this._filterBar.innerHTML = filtersHTML || '';
      this._wireFilterHandlers();
      items = src.applyFilters(items, fState, this._biosStatusBySystem);
    } else {
      this._filterBar.innerHTML = '';
    }

    if (!items.length) {
      this._renderError('No games match these filters.');
      return;
    }

    this._grid.innerHTML = items.map(it => {
      const html = (src.renderCard && src.renderCard(it)) || _renderCard(it);
      return html;
    }).join('');

    // Emulator gets a "BIOS Vault" affordance below the grid since its
    // ROMs are the only ones that need outside BIOS to actually run.
    if (this.activeSourceId === 'emulator') {
      // "Scrape covers" sits beside the BIOS Vault because both are
      // library-wide maintenance, not per-card actions. Without it the
      // only way to get art onto an imported folder of ROMs is to open
      // the cover picker once per title.
      this._loadMoreWrap.innerHTML = `
        <button type="button" class="lib-games-bios-btn">Open BIOS Vault</button>
        <button type="button" class="lib-games-scrape-btn">Scrape covers</button>
        <div class="lib-games-scrape-status" hidden></div>
        ${this.pageData.hasMore ? `<button type="button" class="lib-games-loadmore">Load more</button>` : ''}
      `;
      this._loadMoreWrap.querySelector('.lib-games-bios-btn')
        ?.addEventListener('click', () => openBiosVault({}));
      this._loadMoreWrap.querySelector('.lib-games-scrape-btn')
        ?.addEventListener('click', () => this._startCoverScrape());
    } else {
      this._loadMoreWrap.innerHTML = this.pageData.hasMore
        ? `<button type="button" class="lib-games-loadmore">Load more</button>`
        : '';
    }
    this._loadMoreWrap.querySelector('.lib-games-loadmore')
      ?.addEventListener('click', () => this._loadNextPage());

    this._wireRomDropzones();
  }

  async _startCoverScrape() {
    // Guard against a second click while one is in flight — the button
    // stays in the DOM across the re-render that finishing triggers.
    if (this._scrapeJobId) return;
    // Re-query on every write instead of caching the nodes. A filter
    // click or BIOS refresh re-renders the grid mid-scrape, which
    // replaces _loadMoreWrap's children — a cached reference would then
    // be writing progress into a detached node and silently showing
    // nothing.
    const el = (sel) => this._loadMoreWrap.querySelector(sel);
    const setBtnDisabled = (v) => {
      const b = el('.lib-games-scrape-btn');
      if (b) b.disabled = v;
    };
    const say = (msg) => {
      const status = el('.lib-games-scrape-status');
      if (!status) return;
      status.hidden = false;
      status.textContent = msg;
    };

    try {
      const res = await fetch('/api/titles/_/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // only_missing: a cover the user picked by hand outranks
        // anything we can guess, so it is never overwritten.
        body: JSON.stringify({ only_missing: true }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { job_id: jobId, total } = await res.json();
      if (!total) {
        say('Every ROM already has cover art.');
        return;
      }
      this._scrapeJobId = jobId;
      setBtnDisabled(true);
      say(`Scraping 0 / ${total}…`);
      await this._pollCoverScrape(jobId, say);
    } catch (err) {
      console.warn('[library] cover scrape failed', err);
      say('Cover scrape failed — see console.');
    } finally {
      this._scrapeJobId = null;
      setBtnDisabled(false);
    }
  }

  async _pollCoverScrape(jobId, say) {
    // Poll rather than stream: a scrape is a handful of seconds of HEAD
    // requests, and this avoids standing up a WS channel for it.
    for (;;) {
      await new Promise(r => setTimeout(r, 700));
      const res = await fetch(`/api/titles/_/scrape/${encodeURIComponent(jobId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = await res.json();
      if (job.state === 'running') {
        const now = job.current ? ` — ${job.current}` : '';
        say(`Scraping ${job.done} / ${job.total}${now}`);
        continue;
      }
      // Terminal. Report what actually landed, and be explicit about
      // the misses instead of implying full coverage.
      const results = job.results || [];
      const applied = results.filter(r => r.status === 'applied').length;
      const review = results.filter(r => r.status === 'needs_review').length;
      const parts = [`Found art for ${applied} of ${job.total}`];
      if (review) parts.push(`${review} need a manual pick`);
      if (job.state === 'cancelled') parts.push('(cancelled)');
      if (job.state === 'failed') parts.push(`(failed: ${job.error || 'unknown'})`);
      say(`${parts.join(' · ')}.`);
      if (applied) {
        // Covers are persisted on the titles now; a refetch is what
        // makes them appear without a page reload.
        // WINDOW, not document — library.js listens on window (see its
        // subscription at library.js:177), and every other dispatcher in
        // library-game-sources.js uses window too. Dispatching on
        // document here would be silently inert.
        window.dispatchEvent(new CustomEvent('library:games-source-refresh'));
      }
      return;
    }
  }

  _wireRomDropzones() {
    // The "Add ROMs" card advertises itself as a dropzone in markup
    // (`data-emulator-dropzone`), but markup alone does nothing —
    // without these listeners a dropped folder is handed to the
    // browser, which navigates away from the app. Re-attached on every
    // render because innerHTML replaced the nodes.
    this._grid.querySelectorAll('[data-emulator-dropzone]').forEach((zone) => {
      zone.addEventListener('dragover', (ev) => {
        // dragover MUST preventDefault or 'drop' never fires at all.
        ev.preventDefault();
        zone.classList.add('is-drag-over');
      });
      zone.addEventListener('dragleave', () => zone.classList.remove('is-drag-over'));
      zone.addEventListener('drop', (ev) => {
        ev.preventDefault();
        zone.classList.remove('is-drag-over');
        // DataTransfer goes inert the moment this handler returns, so
        // snapshot entries AND files synchronously before any await —
        // the importer's contract (see importEmulatorRomsFromEntries)
        // is that the caller has already captured them.
        const items = Array.from(ev.dataTransfer?.items || []);
        const entries = items
          .map(it => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null))
          .filter(Boolean);
        const files = Array.from(ev.dataTransfer?.files || []);
        if (!entries.length && !files.length) return;
        Promise.resolve(importEmulatorRomsFromEntries(entries, files))
          .catch(err => console.warn('[library] ROM drop import failed', err));
      });
    });
  }

  _wireFilterHandlers() {
    if (this.activeSourceId !== 'emulator') return;
    const fs = this._filters.emulator;
    const bar = this._filterBar;

    // The data attributes here match what _renderEmulatorFilters emits
    // in library-game-sources.js — DON'T rename them on the library
    // side or filter clicks become no-ops.
    bar.querySelectorAll('[data-emu-system]').forEach((pill) => {
      pill.addEventListener('click', () => {
        const sid = pill.dataset.emuSystem;
        if (fs.systemId === sid) return;
        fs.systemId = sid;
        this._renderGrid();
      });
    });

    const biosCheckbox = bar.querySelector('[data-emu-bios-ready]');
    biosCheckbox?.addEventListener('change', (ev) => {
      fs.biosReadyOnly = !!ev.target.checked;
      this._renderGrid();
    });

    const sortSelect = bar.querySelector('[data-emu-sort]');
    sortSelect?.addEventListener('change', (ev) => {
      fs.sort = ev.target.value || 'recent';
      this._renderGrid();
    });

    // BIOS Vault manage button.
    bar.querySelectorAll('[data-emu-bios-manage]').forEach((btn) => {
      btn.addEventListener('click', () => {
        openBiosVault({
          onChange: async () => {
            try {
              this._biosStatusBySystem = await fetchEmulatorBiosStatus();
            } catch (err) {
              console.warn('[library] BIOS status refresh failed', err);
            }
            this._renderGrid();
          },
        });
      });
    });
  }

  _onGridClick = async (ev) => {
    const pinBtn = ev.target.closest('[data-pin-src]');
    if (pinBtn) {
      await this._pin(pinBtn);
      return;
    }
    // Synthetic "Add ROMs" card. It is the ONLY card that carries no
    // real title id, so it emits `data-emulator-action` instead of
    // `data-launch-emulator` and needs its own branch — without this
    // the card and its "Pick folder" button are inert, which is
    // exactly how the folder picker got orphaned when the grid moved
    // into this module. Route it through the same onLaunch the real
    // cards use; the source already dispatches on `_action`.
    const emuAction = ev.target.closest('[data-emulator-action]');
    if (emuAction) {
      const action = emuAction.dataset.emulatorAction;
      const item = this.pageData.items.find(i => i._action === action);
      const src = getSource('emulator');
      if (src?.onLaunch && item) src.onLaunch(item);
      return;
    }
    // Per-card overlay buttons on the ROM cards. All three are rendered
    // by _renderEmulatorCard but none of them were handled here — the
    // same orphaning that hit "Pick folder" when the grid moved into
    // this module. Each helper dispatches `library:games-source-refresh`
    // on success, which library.js already listens for, so there is
    // nothing to re-render by hand.
    //
    // Cover art: this is the "search the web for box art" button. It
    // opens the picker, which hits /api/titles/{id}/cover-candidates and
    // offers the results plus a custom upload and a reset-to-auto.
    const coverEdit = ev.target.closest('[data-cover-edit]');
    if (coverEdit) {
      // Stop the click reaching any card-level launch handler — the
      // button sits inside the preview slot.
      ev.preventDefault();
      ev.stopPropagation();
      const id = coverEdit.dataset.coverEdit;
      const item = this.pageData.items.find(i => String(i.id) === String(id));
      await openCoverPicker(id, item?.title || '');
      return;
    }
    // Card title doubles as the rename affordance. Renaming was already
    // supported by the store (a `title` in the metadata patch updates
    // display_name) with no UI in front of it.
    const renameTitle = ev.target.closest('[data-rename-title]');
    if (renameTitle) {
      ev.preventDefault();
      ev.stopPropagation();
      const id = renameTitle.dataset.renameTitle;
      const item = this.pageData.items.find(i => String(i.id) === String(id));
      await openTitleRename(id, item?.title || '');
      return;
    }
    // System badge: rescues a mis-detected ROM (a PS2 ISO read as PSX,
    // etc.) without a re-upload. Also the only way to set a system on a
    // ROM that detected as nothing, which is what "SET SYSTEM" means.
    const changeSystem = ev.target.closest('[data-change-system]');
    if (changeSystem) {
      ev.preventDefault();
      ev.stopPropagation();
      await openSystemPicker(
        changeSystem.dataset.changeSystem,
        changeSystem.dataset.currentSystem || '',
      );
      return;
    }
    const removeEmu = ev.target.closest('[data-remove-emulator]');
    if (removeEmu) {
      ev.preventDefault();
      ev.stopPropagation();
      const id = removeEmu.dataset.removeEmulator;
      const item = this.pageData.items.find(i => String(i.id) === String(id));
      await removeEmulatorTitle(id, item?.title || '');
      return;
    }
    // Launch buttons: emulator emits data-launch-emulator, streamed
    // emits data-launch-streamed. Each carries the source-specific id
    // we use to look the item up in pageData.
    const emuLaunch = ev.target.closest('[data-launch-emulator]');
    if (emuLaunch) {
      const id = emuLaunch.dataset.launchEmulator;
      const item = this.pageData.items.find(i => String(i.id) === String(id));
      const src = getSource('emulator');
      if (src?.onLaunch && item) src.onLaunch(item);
      return;
    }
    const streamedLaunch = ev.target.closest('[data-launch-streamed]');
    if (streamedLaunch) {
      const id = streamedLaunch.dataset.launchStreamed;
      const item = this.pageData.items.find(i => String(i.id) === String(id));
      const src = getSource('streamed');
      if (src?.onLaunch && item) src.onLaunch(item);
      return;
    }
  };

  async _pin(btn) {
    const source = btn.dataset.pinSrc;
    const source_id = btn.dataset.pinSid;
    if (!source || !source_id) return;
    // GamePinRequest REQUIRES `name` (plus optional metadata). Sending
    // only {source, source_id} 422s every pin. The full browse item is in
    // pageData — look it up and forward the real metadata so the pinned
    // artifact carries its title/cover/links instead of failing.
    const item = (this.pageData.items || []).find(
      (i) => String(i.source_id) === String(source_id)
        && String(i.source || source) === String(source),
    ) || {};
    const body = {
      source,
      source_id,
      name: item.name || item.title || source_id,
      author: item.author || '',
      tagline: item.tagline || '',
      thumbnail_url: item.thumbnail_url || item.image || '',
      source_url: item.source_url || item.url || '',
      embed_url: item.embed_url || '',
      play_mode: item.play_mode || 'embed',
      genre: Array.isArray(item.genre) ? item.genre : [],
      size_bytes: Number(item.size_bytes) || 0,
      load_estimate_ms: Number(item.load_estimate_ms) || 0,
      extra: (item.extra && typeof item.extra === 'object') ? item.extra : {},
    };
    btn.disabled = true;
    btn.textContent = 'Pinning…';
    try {
      const resp = await fetch('/api/games/pin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`status ${resp.status}`);
      btn.textContent = 'Pinned';
      btn.classList.add('done');
      this.onPinned({ source, source_id });
    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Pin';
      console.warn('[library] pin failed', err);
    }
  }

  async _resolveSettings() {
    try {
      const settings = await this.getSettings();
      return settings || {};
    } catch {
      return {};
    }
  }

  destroy() {
    // No long-running fetches to abort yet — the AbortController hook
    // is here so a future "fetch with progress" path can wire it.
    if (this._abort) this._abort.abort();
  }
}


// Library2-flavored shared card — atelier register, brass primary
// action. Stays compatible with the existing data-pin-src / data-pin-sid
// contract so /api/games/pin still works without any backend change.
function _renderCard(item) {
  const title = item.name || item.title || 'Untitled';
  const subtitle = item.tagline || item.author || '';
  const thumb = item.thumbnail_url || item.image || '';
  const preview = ensurePreviewHTML(thumb, title);

  return `
    <article class="lib-game-card">
      <div class="lib-game-card-cover">${preview}</div>
      <div class="lib-game-card-body">
        <div class="lib-game-card-title">${escapeHtml(title)}</div>
        ${subtitle
          ? `<div class="lib-game-card-sub">${escapeHtml(subtitle)}</div>`
          : ''}
        <div class="lib-game-card-actions">
          <button type="button" class="lib-game-pin"
                  data-pin-src="${escapeHtml(item.source || '')}"
                  data-pin-sid="${escapeHtml(item.source_id || '')}">
            Pin
          </button>
        </div>
      </div>
    </article>
  `;
}
