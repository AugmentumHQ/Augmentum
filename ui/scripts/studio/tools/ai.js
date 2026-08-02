/* ==========================================================================
   Studio AI Assist Tool — palette-resident action menu
   --------------------------------------------------------------------------
   Rehomes the existing AI popover (Edit / Analyze / Create groups) into the
   palette drawer. Action names and dispatch behavior are unchanged — clicking
   an item calls ctx.runAi(action, prompt?). The header keeps a small
   sparkle-icon shortcut that activates this tool in one click.

   Translate is the one action that needs an inline prompt (target language);
   we render a tiny input row below the action when it's chosen, instead of
   firing a window.prompt() — pro tools never break flow to ask a question.

   Backend contract:
     ctx.aiActionGroups  → [{ group, items: [{ action, name, desc }] }, ...]
     ctx.runAi(action, extraArg?)  → dispatch (studio.js owns the network call)
   ========================================================================== */

import { escapeHtml } from '../../app.js';

export function createAiTool() {
  let mountEl = null;
  let ctx = null;
  let translatePending = false;   // toggled when user clicks Translate

  function mount(el, toolCtx) {
    mountEl = el;
    ctx = toolCtx;
    el.classList.add('studio-tool-ai');
    _render();
    el.addEventListener('click', _onClick);
    el.addEventListener('keydown', _onKeydown);
  }

  function unmount(el) {
    el?.removeEventListener('click', _onClick);
    el?.removeEventListener('keydown', _onKeydown);
    if (el) {
      el.classList.remove('studio-tool-ai');
      el.innerHTML = '';
    }
    mountEl = null;
    ctx = null;
    translatePending = false;
  }

  function onCtxChange(newCtx) {
    ctx = newCtx || ctx;
    _render();
  }

  function _render() {
    if (!mountEl) return;
    const groups = Array.isArray(ctx?.aiActionGroups) ? ctx.aiActionGroups : [];
    if (!groups.length) {
      mountEl.innerHTML = `
        <div class="studio-tool-ai-empty">
          AI actions aren't wired for this editor yet.
        </div>
      `;
      return;
    }

    const groupHtml = groups.map(g => {
      const items = (g.items || []).map(it => `
        <button type="button"
                class="studio-tool-ai-item"
                data-ai-action="${escapeHtml(it.action)}">
          <span class="studio-tool-ai-item-name">${escapeHtml(it.name)}</span>
          ${it.desc ? `<span class="studio-tool-ai-item-desc">${escapeHtml(it.desc)}</span>` : ''}
        </button>
      `).join('');
      return `
        <section class="studio-tool-ai-group">
          <header class="studio-tool-ai-group-head">${escapeHtml(g.group || '')}</header>
          ${items}
        </section>
      `;
    }).join('');

    const translateRow = translatePending ? `
      <div class="studio-tool-ai-translate-row">
        <label for="studio-tool-ai-translate-input">Translate to</label>
        <input id="studio-tool-ai-translate-input"
               type="text"
               class="studio-tool-ai-translate-input"
               placeholder="e.g. Spanish, German, Japanese"
               autocomplete="off">
        <button type="button"
                class="studio-tool-ai-translate-go"
                data-ai-translate-go>Translate</button>
        <button type="button"
                class="studio-tool-ai-translate-cancel"
                data-ai-translate-cancel>Cancel</button>
      </div>
    ` : '';

    mountEl.innerHTML = groupHtml + translateRow;

    if (translatePending) {
      mountEl.querySelector('#studio-tool-ai-translate-input')?.focus();
    }
  }

  function _onClick(e) {
    if (e.target.closest('[data-ai-translate-go]')) {
      const inp = mountEl.querySelector('#studio-tool-ai-translate-input');
      const lang = (inp?.value || '').trim();
      if (!lang) { inp?.focus(); return; }
      translatePending = false;
      _render();
      try { ctx?.runAi?.('translate', lang); }
      catch (err) { console.warn('AI translate threw', err); }
      return;
    }
    if (e.target.closest('[data-ai-translate-cancel]')) {
      translatePending = false;
      _render();
      return;
    }
    const item = e.target.closest('[data-ai-action]');
    if (!item) return;
    const action = item.dataset.aiAction;
    if (action === 'translate') {
      translatePending = true;
      _render();
      return;
    }
    try { ctx?.runAi?.(action); }
    catch (err) { console.warn('AI runAi threw', err); }
  }

  function _onKeydown(e) {
    if (e.key !== 'Enter') return;
    if (e.target.id === 'studio-tool-ai-translate-input') {
      e.preventDefault();
      mountEl.querySelector('[data-ai-translate-go]')?.click();
    }
  }

  return {
    id: 'ai',
    label: 'AI Assist',
    mount,
    unmount,
    onCtxChange,
  };
}
