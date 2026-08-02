// Declarative action substrate — registry-driven settings UI (Phase 2).
//
// Renders /api/settings/registry into a search-first, section-grouped
// surface. Self-contained: no edits to settings.js beyond a single
// nav button + pane shell. Lazy-mounts on first reveal of the
// settings-tab-registry pane.
//
// Search operators:
//   @modified           — only settings whose current value != default
//   @advanced           — include settings tagged advanced (hidden by default)
//   @tag:<name>         — restrict to a tag
//   @section:<dotted>   — restrict to a section prefix
//   @restart            — restart-required only
//   @deprecated         — include deprecated
//   anything else       — free-text matches label / description / key / tags

(function () {
  'use strict';

  const ROOT_ID = 'settings-registry-root';
  const PANE_ID = 'settings-tab-registry';
  let _mounted = false;
  let _settings = [];      // all registered settings
  let _showAdvanced = false;
  let _filterText = '';

  // Shared control renderer (settings-render.js) — the single source of truth
  // for row/control markup + wiring + persistence, shared with the curated
  // settings tabs so every knob looks and behaves identically.
  const R = window.augmentumSettingsRender;
  const escapeHtml = R.escapeHtml;

  // ----- fetch -----

  async function fetchRegistry() {
    const params = new URLSearchParams();
    if (_showAdvanced) params.set('show_advanced', 'true');
    const url = '/api/settings/registry/?' + params.toString();
    const r = await fetch(url);
    if (!r.ok) throw new Error('Failed to load registry: ' + r.status);
    const body = await r.json();
    return body.settings || [];
  }

  // ----- search parsing -----

  function parseQuery(raw) {
    const tokens = raw.trim().split(/\s+/).filter(Boolean);
    const flags = {
      modified: false,
      advanced: false,
      restart: false,
      deprecated: false,
    };
    const tags = [];
    const sections = [];
    const text = [];
    for (const t of tokens) {
      if (t === '@modified') flags.modified = true;
      else if (t === '@advanced') flags.advanced = true;
      else if (t === '@restart') flags.restart = true;
      else if (t === '@deprecated') flags.deprecated = true;
      else if (t.startsWith('@tag:')) tags.push(t.slice(5).toLowerCase());
      else if (t.startsWith('@section:')) sections.push(t.slice(9).toLowerCase());
      else text.push(t.toLowerCase());
    }
    return { flags, tags, sections, text };
  }

  function matches(setting, parsed) {
    const f = parsed.flags;
    if (f.modified && !setting.modified) return false;
    if (f.restart && !setting.restart_required) return false;
    if (!f.deprecated && setting.deprecated) return false;
    for (const tag of parsed.tags) {
      if (!setting.tags.map((t) => t.toLowerCase()).includes(tag)) return false;
    }
    for (const sec of parsed.sections) {
      const ssec = setting.section.toLowerCase();
      if (!(ssec === sec || ssec.startsWith(sec + '.'))) return false;
    }
    if (parsed.text.length) {
      const hay = (
        setting.key +
        ' ' +
        setting.label +
        ' ' +
        setting.description +
        ' ' +
        setting.section +
        ' ' +
        (setting.tags || []).join(' ') +
        ' ' +
        (setting.voice_aliases || []).join(' ')
      ).toLowerCase();
      for (const w of parsed.text) {
        if (!hay.includes(w)) return false;
      }
    }
    return true;
  }

  // ----- render -----

  function shellHtml(total, visible) {
    return `
      <div class="settings-registry-toolbar">
        <input class="settings-registry-search field-input" id="settings-registry-search"
               type="search" placeholder="Search settings — try @modified, @advanced, @tag:voice"
               value="${escapeHtml(_filterText)}" autocomplete="off">
        <label class="settings-registry-advanced-toggle">
          <input type="checkbox" id="settings-registry-show-advanced" ${_showAdvanced ? 'checked' : ''}>
          <span>Show advanced</span>
        </label>
      </div>
      <div class="settings-registry-count" id="settings-registry-count">
        Showing ${visible} of ${total} registered setting${total === 1 ? '' : 's'}
      </div>
      <div class="settings-registry-list" id="settings-registry-list"></div>
    `;
  }

  // Row rendering delegates to the shared component (settings-render.js) so the
  // registry list and the curated tabs stay pixel-identical. The registry shows
  // the meta line (key · section · tags); curated tabs will suppress it.
  const settingRowHtml = (setting) => R.rowHtml(setting, { showMeta: true });

  function sectionHtml(section, items) {
    const rows = items.map(settingRowHtml).join('');
    return `
      <details class="settings-registry-section" open>
        <summary class="settings-registry-section-head">
          <span class="sr-section-name">${escapeHtml(section)}</span>
          <span class="sr-section-count">${items.length}</span>
        </summary>
        <div class="settings-registry-section-body">${rows}</div>
      </details>
    `;
  }

  // Build the shell (toolbar + search input) ONCE, then delegate to
  // renderList for the dynamic part. Rebuilding the whole shell on every
  // keystroke was destroying the search <input> and stealing focus/cursor
  // mid-type — re-render only the list so the input survives.
  function render(root) {
    root.innerHTML = shellHtml(_settings.length, _settings.length);
    wireToolbar(root);
    renderList(root);
  }

  function renderList(root) {
    const parsed = parseQuery(_filterText);
    // The @advanced operator widens the filter without needing the
    // checkbox; treat it as a temporary include.
    const effectiveShowAdvanced = _showAdvanced || parsed.flags.advanced;
    const filtered = _settings.filter((s) => {
      if (!effectiveShowAdvanced && s.advanced) return false;
      return matches(s, parsed);
    });

    const countEl = root.querySelector('#settings-registry-count');
    if (countEl) {
      countEl.textContent =
        `Showing ${filtered.length} of ${_settings.length} registered setting${_settings.length === 1 ? '' : 's'}`;
    }

    const list = root.querySelector('#settings-registry-list');
    if (!list) return;

    // Group by section.
    const bySection = new Map();
    for (const s of filtered) {
      if (!bySection.has(s.section)) bySection.set(s.section, []);
      bySection.get(s.section).push(s);
    }
    const sectionsSorted = Array.from(bySection.keys()).sort();
    list.innerHTML = sectionsSorted.map((sec) => sectionHtml(sec, bySection.get(sec))).join('');

    wireRows(root);
  }

  // ----- wiring -----

  function wireToolbar(root) {
    const search = root.querySelector('#settings-registry-search');
    if (search) {
      let t = null;
      search.addEventListener('input', () => {
        clearTimeout(t);
        t = setTimeout(() => {
          _filterText = search.value;
          renderList(root);  // list only — keep the search input (focus/cursor) intact
        }, 120);
      });
    }
    const adv = root.querySelector('#settings-registry-show-advanced');
    if (adv) {
      adv.addEventListener('change', async () => {
        _showAdvanced = adv.checked;
        await reload();
      });
    }
  }

  async function reload() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    try {
      _settings = await fetchRegistry();
      render(root);
    } catch (exc) {
      root.innerHTML = `<div class="settings-registry-error">Failed to load: ${escapeHtml(exc.message)}</div>`;
    }
  }

  function wireRows(root) {
    R.wireControls(root, (key, value) => applyChange(key, value));
  }

  async function applyChange(key, value) {
    const status = document.getElementById('set-status-' + key);
    if (status) status.textContent = 'Saving…';
    try {
      await R.saveSetting(key, value);
      if (status) {
        status.textContent = 'Saved';
        status.classList.add('setting-status-ok');
        setTimeout(() => {
          status.textContent = '';
          status.classList.remove('setting-status-ok');
        }, 2000);
      }
      // Refresh the underlying setting state so the modified badge stays accurate.
      try {
        const fresh = await (await fetch('/api/settings/registry/' + encodeURIComponent(key))).json();
        const idx = _settings.findIndex((s) => s.key === key);
        if (idx >= 0) _settings[idx] = fresh;
      } catch (_) { /* tolerate; UI is fine even if refresh fails */ }
    } catch (exc) {
      if (status) {
        status.textContent = 'Save failed: ' + exc.message;
        status.classList.add('setting-status-err');
        setTimeout(() => status.classList.remove('setting-status-err'), 5000);
      }
    }
  }

  // ----- mount on tab reveal -----

  async function maybeMount() {
    if (_mounted) return;
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    _mounted = true;
    await reload();
  }

  function observePane() {
    const pane = document.getElementById(PANE_ID);
    if (!pane) {
      // Settings modal not constructed yet; wait for next tick.
      setTimeout(observePane, 250);
      return;
    }
    // Reveal triggers mount.
    const obs = new MutationObserver(() => {
      if (!pane.classList.contains('hidden')) maybeMount();
    });
    obs.observe(pane, { attributes: true, attributeFilter: ['class'] });
    if (!pane.classList.contains('hidden')) maybeMount();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observePane);
  } else {
    observePane();
  }

  window.augmentumSettingsRegistry = {
    reload,
    refresh: () => {
      _mounted = false;
      maybeMount();
    },
  };
})();
