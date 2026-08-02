/* ==========================================================================
   Studio Design Tool — theme + typography + density + accent
   --------------------------------------------------------------------------
   Mounts into the Tool Palette drawer. Renders five control groups in a
   single vertical column:
     1. Theme swatches (5 presets — slate / corporate / modern / emerald / rose)
     2. Font family (system / sans / serif / mono / dyslexic)
     3. Font size scale (4 presets — 0.85 / 1.0 / 1.15 / 1.3)
     4. Line height (tight / comfortable / airy)
     5. Density (compact / default / spacious)
     6. Accent override (color input + reset button)

   Backend contract:
     ctx.getDesign()          → { theme, font_family, font_size_scale,
                                  line_height, density, accent_override }
     ctx.onDesignChange(d)    → mutate source.design + schedule save
     ctx.themes               → optional [{name, accent, ...}] list (passed
                                from studio.js; falls back to FALLBACK_THEMES
                                if missing so the tool still renders before
                                the first /themes/list response lands)
   ========================================================================== */

import { escapeHtml } from '../../app.js';

// Default design used when ctx.getDesign() returns falsy. Mirrors the Python
// DEFAULT_DESIGN; kept in JS so the tool can render before any save round-trip.
const DEFAULT_DESIGN = {
  theme: '',
  font_family: 'system',
  font_size_scale: 1.0,
  line_height: 'comfortable',
  density: 'default',
  accent_override: null,
};

// Fallback theme set if ctx.themes wasn't seeded — gives the tool something to
// show on first paint even when the /themes/list response is in-flight. The
// real palette overrides this once ctx.themes lands.
const FALLBACK_THEMES = [
  { name: 'slate',     accent: '#2563EB', label: 'Slate' },
  { name: 'corporate', accent: '#1E3A8A', label: 'Corporate' },
  { name: 'modern',    accent: '#4F46E5', label: 'Modern' },
  { name: 'emerald',   accent: '#059669', label: 'Emerald' },
  { name: 'rose',      accent: '#E11D48', label: 'Rose' },
];

const FONT_FAMILY_OPTIONS = [
  { value: 'system',   label: 'System',    note: 'Native default' },
  { value: 'sans',     label: 'Sans',      note: 'Arial / Helvetica' },
  { value: 'serif',    label: 'Serif',     note: 'Georgia / Times' },
  { value: 'mono',     label: 'Mono',      note: 'Consolas / Courier' },
  { value: 'dyslexic', label: 'Dyslexic',  note: 'Comic Sans MS proxy' },
];

const FONT_SIZE_OPTIONS = [
  { value: 0.85, label: 'Small' },
  { value: 1.0,  label: 'Standard' },
  { value: 1.15, label: 'Large' },
  { value: 1.3,  label: 'X-Large' },
];

const LINE_HEIGHT_OPTIONS = [
  { value: 'tight',       label: 'Tight' },
  { value: 'comfortable', label: 'Comfortable' },
  { value: 'airy',        label: 'Airy' },
];

const DENSITY_OPTIONS = [
  { value: 'compact',  label: 'Compact' },
  { value: 'default',  label: 'Default' },
  { value: 'spacious', label: 'Spacious' },
];

export function createDesignTool() {
  let mountEl = null;
  let ctx = null;
  // Latest design snapshot the tool is showing. Kept in-memory so segment
  // clicks repaint instantly without an extra ctx round-trip.
  let design = { ...DEFAULT_DESIGN };

  function mount(el, toolCtx) {
    mountEl = el;
    ctx = toolCtx;
    el.classList.add('studio-tool-design');
    _hydrate();
    _render();
    el.addEventListener('click', _onClick);
    el.addEventListener('input', _onInput);
    el.addEventListener('change', _onChange);
  }

  function unmount(el) {
    el?.removeEventListener('click', _onClick);
    el?.removeEventListener('input', _onInput);
    el?.removeEventListener('change', _onChange);
    if (el) {
      el.classList.remove('studio-tool-design');
      el.innerHTML = '';
    }
    mountEl = null;
    ctx = null;
  }

  function onCtxChange(newCtx) {
    ctx = newCtx || ctx;
    _hydrate();
    _render();
  }

  function _hydrate() {
    const fromCtx = ctx?.getDesign?.();
    if (fromCtx && typeof fromCtx === 'object') {
      design = { ...DEFAULT_DESIGN, ...fromCtx };
    }
  }

  function _commit(partial) {
    design = { ...design, ...partial };
    try {
      ctx?.onDesignChange?.(design);
    } catch (err) {
      console.warn('design tool onDesignChange threw', err);
    }
    _render();
  }

  // --- DOM ------------------------------------------------------------------

  function _render() {
    if (!mountEl) return;
    const themes = (ctx?.themes && ctx.themes.length ? ctx.themes : FALLBACK_THEMES);
    const activeTheme = design.theme || (themes[0]?.name || '');

    const inheritsNote = ctx?.inheritsNote
      ? `<div class="studio-tool-design-inherits">${escapeHtml(ctx.inheritsNote)}</div>`
      : '';

    mountEl.innerHTML = `
      ${inheritsNote}
      <section class="studio-tool-design-group" data-group="theme">
        <header class="studio-tool-design-group-head">Theme</header>
        <div class="studio-tool-design-theme-grid">
          ${themes.map(t => _themeSwatchHtml(t, activeTheme)).join('')}
        </div>
      </section>

      <section class="studio-tool-design-group" data-group="font-family">
        <header class="studio-tool-design-group-head">Font</header>
        <div class="studio-tool-design-seg" role="radiogroup" aria-label="Font family">
          ${FONT_FAMILY_OPTIONS.map(o => _segBtnHtml('font_family', o.value, o.label, design.font_family, o.note)).join('')}
        </div>
      </section>

      <section class="studio-tool-design-group" data-group="font-size">
        <header class="studio-tool-design-group-head">Size</header>
        <div class="studio-tool-design-seg" role="radiogroup" aria-label="Font size">
          ${FONT_SIZE_OPTIONS.map(o => _segBtnHtml('font_size_scale', o.value, o.label, design.font_size_scale)).join('')}
        </div>
      </section>

      <section class="studio-tool-design-group" data-group="line-height">
        <header class="studio-tool-design-group-head">Line height</header>
        <div class="studio-tool-design-seg" role="radiogroup" aria-label="Line height">
          ${LINE_HEIGHT_OPTIONS.map(o => _segBtnHtml('line_height', o.value, o.label, design.line_height)).join('')}
        </div>
      </section>

      <section class="studio-tool-design-group" data-group="density">
        <header class="studio-tool-design-group-head">Density</header>
        <div class="studio-tool-design-seg" role="radiogroup" aria-label="Density">
          ${DENSITY_OPTIONS.map(o => _segBtnHtml('density', o.value, o.label, design.density)).join('')}
        </div>
      </section>

      <section class="studio-tool-design-group" data-group="accent">
        <header class="studio-tool-design-group-head">Accent override</header>
        <div class="studio-tool-design-accent-row">
          <input type="color" class="studio-tool-design-accent-input"
                 value="${escapeHtml(design.accent_override || _activeThemeAccent(themes, activeTheme))}"
                 aria-label="Accent color"
                 data-design-key="accent_override">
          <span class="studio-tool-design-accent-value">${escapeHtml(design.accent_override || 'Theme default')}</span>
          ${design.accent_override
            ? `<button type="button" class="studio-tool-design-reset" data-reset-accent>Reset</button>`
            : ''}
        </div>
        <p class="studio-tool-design-hint">Overrides only the accent color — text + neutrals stay on the theme palette.</p>
      </section>
    `;
  }

  function _themeSwatchHtml(t, active) {
    const accent = t.accent || '#999';
    const accentDark = t.accent_dark || accent;
    const isActive = t.name === active;
    const label = t.label || t.name;
    return `
      <button type="button"
              class="studio-tool-design-theme-swatch${isActive ? ' active' : ''}"
              data-theme="${escapeHtml(t.name)}"
              aria-pressed="${isActive ? 'true' : 'false'}"
              title="${escapeHtml(label)}">
        <span class="studio-tool-design-theme-swatch-color"
              style="background:linear-gradient(135deg, ${escapeHtml(accent)} 0%, ${escapeHtml(accentDark)} 100%)"></span>
        <span class="studio-tool-design-theme-swatch-label">${escapeHtml(label)}</span>
      </button>
    `;
  }

  function _segBtnHtml(key, value, label, active, note) {
    const isActive = String(value) === String(active);
    const noteHtml = note ? `<span class="studio-tool-design-seg-note">${escapeHtml(note)}</span>` : '';
    return `
      <button type="button"
              class="studio-tool-design-seg-btn${isActive ? ' active' : ''}"
              data-design-key="${escapeHtml(key)}"
              data-design-value="${escapeHtml(String(value))}"
              role="radio"
              aria-checked="${isActive ? 'true' : 'false'}">
        <span class="studio-tool-design-seg-label">${escapeHtml(label)}</span>
        ${noteHtml}
      </button>
    `;
  }

  function _activeThemeAccent(themes, activeName) {
    const t = themes.find(x => x.name === activeName) || themes[0];
    return t?.accent || '#2563EB';
  }

  // --- Event handlers -------------------------------------------------------

  function _onClick(e) {
    const swatch = e.target.closest('[data-theme]');
    if (swatch) {
      _commit({ theme: swatch.dataset.theme });
      return;
    }
    const seg = e.target.closest('[data-design-key][data-design-value]');
    if (seg) {
      const key = seg.dataset.designKey;
      let value = seg.dataset.designValue;
      // font_size_scale arrives as string from dataset — coerce.
      if (key === 'font_size_scale') value = parseFloat(value);
      _commit({ [key]: value });
      return;
    }
    if (e.target.closest('[data-reset-accent]')) {
      _commit({ accent_override: null });
    }
  }

  function _onInput(e) {
    // Live-preview color drag — fires on every value change while picking.
    const colorInput = e.target.closest('input[type="color"][data-design-key="accent_override"]');
    if (!colorInput) return;
    // Reflect the value text immediately without writing back to ctx every
    // pixel of drag — _onChange (below) does the actual commit.
    const valueLabel = mountEl?.querySelector('.studio-tool-design-accent-value');
    if (valueLabel) valueLabel.textContent = colorInput.value.toUpperCase();
  }

  function _onChange(e) {
    // Color input "change" fires once the picker closes — that's our commit.
    const colorInput = e.target.closest('input[type="color"][data-design-key="accent_override"]');
    if (!colorInput) return;
    _commit({ accent_override: colorInput.value.toUpperCase() });
  }

  return {
    id: 'design',
    label: 'Design',
    mount,
    unmount,
    onCtxChange,
  };
}
