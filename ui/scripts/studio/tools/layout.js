/* ==========================================================================
   Studio Layout Tool — per-artifact-type structural picker
   --------------------------------------------------------------------------
   The Layout tool is editor-shape-specific. Same icon, same palette slot,
   but the controls inside depend on what "layout" means for the focused
   element of that artifact type:

     Presentation: which slide layout (title / content / section / 2-col / blank)
     Document:     section header level for new sections (H1-H4)
     Spreadsheet:  freeze header toggle + summary row mode (per active sheet)
     Chart:        chart type (bar / line / pie / scatter / area / stacked)
     Ebook:        chapter break style + page break before (per focused chapter)

   The tool itself is artifact-agnostic — it renders whatever `ctx.getLayoutOptions()`
   returns. studio.js owns the per-type builder that returns option groups, so
   adding a new artifact type or a new control is just two changes there. The
   tool stays a 200-LOC view.

   Backend contract:
     ctx.getLayoutOptions()                    → [LayoutGroup, ...]
     ctx.onLayoutChange(groupId, value)        → apply change + schedule save

   LayoutGroup shape:
     {
       id: string,                            // 'slide_layout' / 'chart_type' / ...
       label: string,                         // 'Slide layout'
       type: 'segmented' | 'toggle' | 'select',
       options: [{ value, label, iconSvg?, note? }, ...],  // for segmented/select
       activeValue: any,                      // currently selected value
       help?: string,                         // optional hint below the group
     }
   ========================================================================== */

import { escapeHtml } from '../../app.js';

export function createLayoutTool() {
  let mountEl = null;
  let ctx = null;

  function mount(el, toolCtx) {
    mountEl = el;
    ctx = toolCtx;
    el.classList.add('studio-tool-layout');
    _render();
    el.addEventListener('click', _onClick);
    el.addEventListener('change', _onChange);
  }

  function unmount(el) {
    el?.removeEventListener('click', _onClick);
    el?.removeEventListener('change', _onChange);
    if (el) {
      el.classList.remove('studio-tool-layout');
      el.innerHTML = '';
    }
    mountEl = null;
    ctx = null;
  }

  function onCtxChange(newCtx) {
    ctx = newCtx || ctx;
    _render();
  }

  function _render() {
    if (!mountEl) return;
    let groups = [];
    try {
      const supplied = ctx?.getLayoutOptions?.();
      if (Array.isArray(supplied)) groups = supplied;
    } catch (err) {
      console.warn('Layout tool getLayoutOptions threw', err);
    }

    if (!groups.length) {
      mountEl.innerHTML = `
        <div class="studio-tool-layout-empty">
          No layout options for this element.<br>
          Switch to a slide, chart, or sheet to see its layout controls.
        </div>
      `;
      return;
    }

    mountEl.innerHTML = groups.map(_groupHtml).join('');
  }

  function _groupHtml(g) {
    const help = g.help ? `<p class="studio-tool-layout-help">${escapeHtml(g.help)}</p>` : '';
    if (g.type === 'toggle') {
      const checked = g.activeValue ? 'checked' : '';
      return `
        <section class="studio-tool-layout-group" data-group="${escapeHtml(g.id)}">
          <header class="studio-tool-layout-group-head">${escapeHtml(g.label)}</header>
          <label class="studio-tool-layout-toggle">
            <input type="checkbox"
                   data-layout-group="${escapeHtml(g.id)}"
                   data-layout-toggle
                   ${checked}>
            <span>${escapeHtml(g.toggleLabel || 'Enabled')}</span>
          </label>
          ${help}
        </section>
      `;
    }
    if (g.type === 'select') {
      const opts = (g.options || []).map(o => `
        <option value="${escapeHtml(String(o.value))}"
                ${String(o.value) === String(g.activeValue) ? 'selected' : ''}>
          ${escapeHtml(o.label)}
        </option>
      `).join('');
      return `
        <section class="studio-tool-layout-group" data-group="${escapeHtml(g.id)}">
          <header class="studio-tool-layout-group-head">${escapeHtml(g.label)}</header>
          <select class="studio-tool-layout-select"
                  data-layout-group="${escapeHtml(g.id)}">
            ${opts}
          </select>
          ${help}
        </section>
      `;
    }
    // Default: segmented
    const cells = (g.options || []).map(o => {
      const active = String(o.value) === String(g.activeValue);
      const icon = o.iconSvg ? `<span class="studio-tool-layout-cell-icon" aria-hidden="true">${o.iconSvg}</span>` : '';
      const note = o.note ? `<span class="studio-tool-layout-cell-note">${escapeHtml(o.note)}</span>` : '';
      return `
        <button type="button"
                class="studio-tool-layout-cell${active ? ' active' : ''}"
                data-layout-group="${escapeHtml(g.id)}"
                data-layout-value="${escapeHtml(String(o.value))}"
                aria-pressed="${active ? 'true' : 'false'}"
                title="${escapeHtml(o.label)}">
          ${icon}
          <span class="studio-tool-layout-cell-label">${escapeHtml(o.label)}</span>
          ${note}
        </button>
      `;
    }).join('');
    return `
      <section class="studio-tool-layout-group" data-group="${escapeHtml(g.id)}">
        <header class="studio-tool-layout-group-head">${escapeHtml(g.label)}</header>
        <div class="studio-tool-layout-cells">${cells}</div>
        ${help}
      </section>
    `;
  }

  function _onClick(e) {
    const cell = e.target.closest('[data-layout-group][data-layout-value]');
    if (!cell) return;
    const id = cell.dataset.layoutGroup;
    const value = cell.dataset.layoutValue;
    _apply(id, value);
  }

  function _onChange(e) {
    const sel = e.target.closest('select[data-layout-group]');
    if (sel) {
      _apply(sel.dataset.layoutGroup, sel.value);
      return;
    }
    const toggle = e.target.closest('input[type="checkbox"][data-layout-toggle]');
    if (toggle) {
      _apply(toggle.dataset.layoutGroup, toggle.checked);
    }
  }

  function _apply(groupId, value) {
    try {
      ctx?.onLayoutChange?.(groupId, value);
    } catch (err) {
      console.warn('Layout tool onLayoutChange threw', err);
    }
    // Re-render so the active-state highlight reflects the new value
    // without waiting on the editor to re-issue ctx.
    _render();
  }

  return {
    id: 'layout',
    label: 'Layout',
    mount,
    unmount,
    onCtxChange,
  };
}
