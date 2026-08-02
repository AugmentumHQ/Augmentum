/* ==========================================================================
   Studio Structure Tool — outline / sorter / TOC / sheet tabs as one surface
   --------------------------------------------------------------------------
   Lists the artifact's structural items (document sections, slide order,
   sheet tabs, chart datasets, ebook chapters) in the palette drawer. Click
   to jump to that element in the editor. For presentations the tool also
   surfaces "Open slide sorter" since the full-screen sorter remains the
   primary multi-select / reorder surface.

   Backend contract (studio.js wires these per artifact type):
     ctx.getStructureItems()   → [{ id, label, kind?, level?, badge?, active? }, ...]
     ctx.onStructureJump(id)   → focus that element in the editor
     ctx.getStructureActions() → [{ id, label }, ...] (optional, e.g. "Open sorter")
     ctx.onStructureAction(id) → run that action
   ========================================================================== */

import { escapeHtml } from '../../app.js';

export function createStructureTool() {
  let mountEl = null;
  let ctx = null;
  let pollTimer = null;

  function mount(el, toolCtx) {
    mountEl = el;
    ctx = toolCtx;
    el.classList.add('studio-tool-structure');
    _render();
    el.addEventListener('click', _onClick);
    // Live-refresh every 2s while the tool is open — headings/slides/chapters
    // can change as the user edits without firing a ctx update, so a cheap
    // poll keeps the list in sync. Cleared on unmount.
    pollTimer = setInterval(_render, 2000);
  }

  function unmount(el) {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    el?.removeEventListener('click', _onClick);
    if (el) {
      el.classList.remove('studio-tool-structure');
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
    let items = [];
    let actions = [];
    try {
      const a = ctx?.getStructureItems?.();
      if (Array.isArray(a)) items = a;
      const b = ctx?.getStructureActions?.();
      if (Array.isArray(b)) actions = b;
    } catch (err) {
      console.warn('Structure tool getter threw', err);
    }

    const actionsHtml = actions.length ? `
      <div class="studio-tool-structure-actions">
        ${actions.map(a => `
          <button type="button"
                  class="studio-tool-structure-action"
                  data-action="${escapeHtml(a.id)}">
            ${escapeHtml(a.label)}
          </button>
        `).join('')}
      </div>
    ` : '';

    if (!items.length) {
      mountEl.innerHTML = `
        ${actionsHtml}
        <div class="studio-tool-structure-empty">
          No structure yet. Add a heading, slide, sheet, or chapter to populate this list.
        </div>
      `;
      return;
    }

    const listHtml = items.map(it => `
      <button type="button"
              class="studio-tool-structure-item${it.active ? ' active' : ''}"
              data-jump="${escapeHtml(String(it.id))}"
              data-level="${escapeHtml(String(it.level || 1))}"
              title="${escapeHtml(it.label || '')}">
        <span class="studio-tool-structure-item-marker" aria-hidden="true">
          ${it.kind === 'slide' ? '▦' : it.kind === 'sheet' ? '▭' : it.kind === 'chapter' ? '▤' : '▸'}
        </span>
        <span class="studio-tool-structure-item-label">${escapeHtml(it.label || '(untitled)')}</span>
        ${it.badge ? `<span class="studio-tool-structure-item-badge">${escapeHtml(it.badge)}</span>` : ''}
      </button>
    `).join('');

    mountEl.innerHTML = `
      ${actionsHtml}
      <div class="studio-tool-structure-list" role="listbox">
        ${listHtml}
      </div>
    `;
  }

  function _onClick(e) {
    const item = e.target.closest('[data-jump]');
    if (item) {
      try { ctx?.onStructureJump?.(item.dataset.jump); }
      catch (err) { console.warn('Structure jump threw', err); }
      return;
    }
    const act = e.target.closest('[data-action]');
    if (act) {
      try { ctx?.onStructureAction?.(act.dataset.action); }
      catch (err) { console.warn('Structure action threw', err); }
    }
  }

  return {
    id: 'structure',
    label: 'Structure',
    mount,
    unmount,
    onCtxChange,
  };
}
