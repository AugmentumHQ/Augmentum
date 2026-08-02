// Shared settings-control renderer (settings-modal convergence, Phase 0).
//
// The single source of truth for turning a *setting descriptor* (the shape
// returned by /api/settings/registry) into a rendered row + wired control.
// Both the registry "All Settings" pane (settings-registry.js) and — in later
// phases — the curated settings tabs consume this, so every knob looks and
// behaves identically no matter which surface shows it.
//
// A descriptor is the registry JSON:
//   { key, kind, current, label, description, section, tags,
//     enum_values?, min_value?, max_value?, max_length?,
//     modified?, advanced?, restart_required?, deprecated?,
//     trust_tier?, voice_aliases? }
//
// This module is intentionally persistence-agnostic: it renders + wires
// controls and reports value changes via a callback. Saving lives with the
// caller (registry pane owns its status/refresh; curated tabs own theirs).
//
// Markup uses neutral `.setting-*` classes styled by settings-controls.css —
// NOT the registry-specific `.settings-registry-*` list chrome.

(function () {
  'use strict';

  // ----- safe HTML -----

  function escapeHtml(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/`/g, '&#96;')
      .replace(/\$\{/g, '&#36;{');
  }

  // ----- control markup -----

  function controlHtml(setting) {
    const cur = setting.current;
    const id = `set-input-${setting.key}`;
    const keyAttr = escapeHtml(setting.key);

    if (setting.kind === 'bool') {
      const checked = cur ? 'checked' : '';
      return `
        <label class="setting-bool">
          <input type="checkbox" id="${id}" data-key="${keyAttr}" ${checked}>
          <span class="setting-bool-track" aria-hidden="true"></span>
        </label>
      `;
    }
    if (setting.kind === 'enum') {
      const opts = (setting.enum_values || [])
        .map(
          (v) =>
            `<option value="${escapeHtml(v)}" ${v === cur ? 'selected' : ''}>${escapeHtml(v || '(empty)')}</option>`,
        )
        .join('');
      return `
        <select class="field-input setting-enum" id="${id}" data-key="${keyAttr}">
          ${opts}
        </select>
      `;
    }
    if (setting.kind === 'tristate') {
      const sel = cur === null || cur === undefined ? 'auto' : cur ? 'true' : 'false';
      return `
        <select class="field-input setting-tristate" id="${id}" data-key="${keyAttr}">
          <option value="auto" ${sel === 'auto' ? 'selected' : ''}>auto (codebase default)</option>
          <option value="true" ${sel === 'true' ? 'selected' : ''}>on</option>
          <option value="false" ${sel === 'false' ? 'selected' : ''}>off</option>
        </select>
      `;
    }
    if (setting.kind === 'int' || setting.kind === 'float') {
      const step = setting.kind === 'float' ? 'any' : '1';
      const min = setting.min_value !== null && setting.min_value !== undefined ? ` min="${setting.min_value}"` : '';
      const max = setting.max_value !== null && setting.max_value !== undefined ? ` max="${setting.max_value}"` : '';
      return `
        <input type="number" class="field-input setting-numeric" id="${id}"
               data-key="${keyAttr}" data-kind="${setting.kind}"
               value="${escapeHtml(cur ?? '')}" step="${step}"${min}${max}>
      `;
    }
    // str
    const maxlen = setting.max_length ? ` maxlength="${setting.max_length}"` : '';
    return `
      <input type="text" class="field-input setting-text" id="${id}"
             data-key="${keyAttr}"
             value="${escapeHtml(cur ?? '')}"${maxlen}>
    `;
  }

  // ----- badges -----

  function badgesHtml(setting) {
    const b = [];
    if (setting.modified) b.push('<span class="setting-badge setting-badge-modified">modified</span>');
    if (setting.advanced) b.push('<span class="setting-badge setting-badge-advanced">advanced</span>');
    if (setting.restart_required) b.push('<span class="setting-badge setting-badge-restart">restart</span>');
    if (setting.deprecated) b.push(`<span class="setting-badge setting-badge-deprecated" title="${escapeHtml(setting.deprecated)}">deprecated</span>`);
    if (setting.trust_tier === 'admin_only') b.push('<span class="setting-badge setting-badge-admin">admin</span>');
    if (setting.trust_tier === 'external') b.push('<span class="setting-badge setting-badge-external">external</span>');
    return b.join(' ');
  }

  // ----- full row -----
  //
  // opts.showMeta (default true) — the key · section · tags · aliases line.
  //   Curated tabs pass false: the tab already establishes the section, so
  //   the raw key/section line would be noise there.

  function rowHtml(setting, opts) {
    const showMeta = !opts || opts.showMeta !== false;
    const tags = setting.tags || [];
    const aliases = setting.voice_aliases || [];
    const metaHtml = showMeta
      ? `
        <div class="setting-row-meta">
          <span class="setting-meta-key">${escapeHtml(setting.key)}</span>
          <span class="setting-meta-sep">·</span>
          <span class="setting-meta-section">${escapeHtml(setting.section)}</span>
          ${tags.length ? `<span class="setting-meta-sep">·</span><span class="setting-meta-tags">${escapeHtml(tags.join(', '))}</span>` : ''}
          ${aliases.length ? `<span class="setting-meta-sep">·</span><span class="setting-meta-aliases" title="Voice aliases">🎙 ${escapeHtml(aliases.join(', '))}</span>` : ''}
        </div>`
      : '';
    return `
      <div class="setting-row" data-key="${escapeHtml(setting.key)}">
        <div class="setting-row-head">
          <div class="setting-row-label">
            ${escapeHtml(setting.label)}
            ${badgesHtml(setting)}
          </div>
          <div class="setting-row-control">${controlHtml(setting)}</div>
        </div>
        <div class="setting-row-desc">${escapeHtml(setting.description)}</div>
        ${metaHtml}
        <div class="setting-row-status" id="set-status-${escapeHtml(setting.key)}"></div>
      </div>
    `;
  }

  // ----- read a control's current value in the right JS type -----

  function readControlValue(el) {
    if (el.type === 'checkbox') return el.checked;
    if (el.dataset.kind === 'int') return parseInt(el.value, 10);
    if (el.dataset.kind === 'float') return parseFloat(el.value);
    // tristate <select> yields "auto" | "true" | "false"
    if (el.classList.contains('setting-tristate')) {
      if (el.value === 'auto') return null;
      return el.value === 'true';
    }
    return el.value;
  }

  // ----- wire every control under `root` to `onChange(key, value, el)` -----

  function wireControls(root, onChange) {
    root.querySelectorAll('input[data-key], select[data-key]').forEach((el) => {
      el.addEventListener('change', () => {
        onChange(el.dataset.key, readControlValue(el), el);
      });
    });
  }

  // ----- generic persistence (shared by every surface) -----
  //
  // Saves a single key via the same endpoint the legacy UI uses. Returns the
  // parsed error text on failure (throws), resolves on success. Callers own
  // their own status/refresh UX.

  async function saveSetting(key, value) {
    const r = await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
    if (!r.ok) {
      const body = await r.text();
      throw new Error('HTTP ' + r.status + ': ' + body.slice(0, 200));
    }
  }

  window.augmentumSettingsRender = {
    escapeHtml,
    controlHtml,
    badgesHtml,
    rowHtml,
    readControlValue,
    wireControls,
    saveSetting,
  };
})();
