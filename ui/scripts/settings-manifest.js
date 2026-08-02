// Curated-tab manifest engine (settings-modal convergence, Phase 1).
//
// A *manifest* is a thin, opinionated view over the settings registry: it names
// an ordered set of groups, each group a human title + the registry section(s)
// whose knobs belong under it. The engine fetches the registry once, filters to
// the declared sections, and renders each group with the shared control
// renderer (settings-render.js) — so a curated group looks and saves exactly
// like the "All Settings" pane, with zero hand-authored control HTML.
//
// This is how curated settings tabs stop being 700-inline-style walls: they
// become data (a manifest) + one renderer. Bespoke non-knob UI (dropzones,
// status panels, browsers) stays hand-authored and is simply mounted around
// the manifest group — the engine only owns the plain knobs.
//
// Manifest shape:
//   {
//     id: 'search-retrieval',           // unique; used for idempotent mount
//     mount: 'search-retrieval-host',   // id of an empty <div> in the pane
//     intro?: 'Optional lead paragraph for the whole block.',
//     groups: [
//       { title: 'Web search quality',
//         intro?: '...',
//         sections: ['search.pipeline'] },
//     ],
//   }
//
// Advanced-tagged settings are omitted from curated groups by design (curated
// tabs stay un-overwhelming); they remain reachable in the All Settings pane.

(function () {
  'use strict';

  const R = window.augmentumSettingsRender;
  const escapeHtml = R.escapeHtml;

  const _manifests = [];        // registered manifests
  const _mounted = new Set();   // manifest ids already rendered
  let _registryCache = null;    // settings[] fetched once, shared across manifests

  // ----- registry fetch (shared, cached) -----

  async function fetchRegistry() {
    if (_registryCache) return _registryCache;
    // include advanced so a group that explicitly lists an advanced-heavy
    // section still resolves; per-setting advanced filtering happens below.
    const r = await fetch('/api/settings/registry/?show_advanced=true');
    if (!r.ok) throw new Error('registry ' + r.status);
    const body = await r.json();
    _registryCache = body.settings || [];
    return _registryCache;
  }

  function inSections(setting, sections) {
    const s = setting.section || '';
    return sections.some((sec) => s === sec || s.startsWith(sec + '.'));
  }

  // ----- render -----

  function groupHtml(group, settings) {
    const rows = settings.map((s) => R.rowHtml(s, { showMeta: false })).join('');
    const intro = group.intro
      ? `<div class="setting-group-intro">${escapeHtml(group.intro)}</div>`
      : '';
    return `
      <section class="setting-group">
        <div class="setting-group-head">${escapeHtml(group.title)}</div>
        ${intro}
        <div class="setting-group-body">${rows}</div>
      </section>
    `;
  }

  function renderManifest(manifest, allSettings) {
    const host = document.getElementById(manifest.mount);
    if (!host) return;

    const intro = manifest.intro
      ? `<div class="setting-manifest-intro">${escapeHtml(manifest.intro)}</div>`
      : '';

    const groupsHtml = manifest.groups
      .map((group) => {
        const picked = allSettings
          .filter((s) => inSections(s, group.sections) && !s.advanced && !s.deprecated)
          .sort((a, b) => a.section.localeCompare(b.section) || a.label.localeCompare(b.label));
        if (!picked.length) return '';
        return groupHtml(group, picked);
      })
      .filter(Boolean)
      .join('');

    host.innerHTML = intro + groupsHtml;
    R.wireControls(host, applyChange);
  }

  // ----- persistence (same status UX as the All Settings pane) -----

  async function applyChange(key, value) {
    const status = document.getElementById('set-status-' + key);
    if (status) status.textContent = 'Saving…';
    try {
      await R.saveSetting(key, value);
      // keep the cache honest so a later re-mount shows the saved value
      if (_registryCache) {
        const row = _registryCache.find((s) => s.key === key);
        if (row) row.current = value;
      }
      if (status) {
        status.textContent = 'Saved';
        status.classList.add('setting-status-ok');
        setTimeout(() => {
          status.textContent = '';
          status.classList.remove('setting-status-ok');
        }, 2000);
      }
    } catch (exc) {
      if (status) {
        status.textContent = 'Save failed: ' + exc.message;
        status.classList.add('setting-status-err');
        setTimeout(() => status.classList.remove('setting-status-err'), 5000);
      }
    }
  }

  // ----- mount on pane reveal -----

  async function mount(manifest) {
    if (_mounted.has(manifest.id)) return;
    const host = document.getElementById(manifest.mount);
    if (!host) return;
    _mounted.add(manifest.id);
    try {
      const settings = await fetchRegistry();
      renderManifest(manifest, settings);
    } catch (exc) {
      host.innerHTML = `<div class="settings-registry-error">Failed to load settings: ${escapeHtml(exc.message)}</div>`;
      _mounted.delete(manifest.id);  // allow a retry on next reveal
    }
  }

  function observe(manifest) {
    const host = document.getElementById(manifest.mount);
    if (!host) {
      setTimeout(() => observe(manifest), 250);  // pane not built yet
      return;
    }
    const pane = host.closest('.settings-pane') || host;
    const isVisible = () => !pane.classList.contains('hidden');
    const obs = new MutationObserver(() => {
      if (isVisible()) mount(manifest);
    });
    obs.observe(pane, { attributes: true, attributeFilter: ['class'] });
    if (isVisible()) mount(manifest);
  }

  function register(manifest) {
    _manifests.push(manifest);
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => observe(manifest));
    } else {
      observe(manifest);
    }
  }

  window.augmentumSettingsManifest = {
    register,
    // expose for tests / manual refresh
    _invalidate: () => {
      _registryCache = null;
      _mounted.clear();
    },
  };

  // ---- Phase 1 pilot manifest: web-search quality knobs ----
  // These 8 registry settings (section `search.pipeline`) had NO curated UI —
  // they were reachable only via the flat All Settings search. Mounted into the
  // admin-only Search pane, they get a proper home for the first time.
  register({
    id: 'search-retrieval',
    mount: 'search-retrieval-host',
    intro:
      'Tuning for how web-search results are expanded, scored, and filtered ' +
      'before they reach the model. Advanced knobs live in All Settings.',
    groups: [
      { title: 'Web search quality', sections: ['search.pipeline'] },
    ],
  });
})();
