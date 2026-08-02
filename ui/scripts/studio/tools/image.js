/* ==========================================================================
   Studio Image Tool — Library / Search / Generate tabs
   --------------------------------------------------------------------------
   Mounts into the Tool Palette drawer. Three tabs share a focus-slot
   contract: when the user clicks a tile / commits a generation, the tool
   calls ctx.onSlotChange(url) which the editor wires to the focused
   image slot (slide.image_url, slide.additional_images[i], chapter.image_url,
   etc.).

   Backend contract:
     ctx.api.searchImages({query, count, prefer_charts})
       → { candidates: [{candidate_id, embed_url, thumb_url, source, title}] }
     ctx.api.generateImage({prompt, style, aspect})
       → { gen_id, embed_url, prompt_used, staged_until }
     ctx.api.commitStaged(gen_id)        → { committed, url }
     ctx.api.discardStaged(gen_id)       → { deleted, gen_id }
     ctx.api.listLibrary()               → [{id, download_url, display_name}]
   ========================================================================== */

import { escapeHtml } from '../../app.js';

const STYLE_PRESETS = [
  { value: '',             label: 'Default' },
  { value: 'photorealism', label: 'Photorealistic' },
  { value: 'illustration', label: 'Illustration' },
  { value: 'vector',       label: 'Vector' },
  { value: 'painterly',    label: 'Painterly' },
  { value: 'diagram',      label: 'Diagram' },
];
const ASPECT_PRESETS = [
  { value: 'square',    label: 'Square' },
  { value: 'landscape', label: 'Landscape' },
  { value: 'portrait',  label: 'Portrait' },
];

export function createImageTool() {
  // Per-mount state. Cleared by unmount().
  let mountEl = null;
  let ctx = null;
  let activeTab = 'library';
  let staged = null;   // {gen_id, embed_url, prompt_used, staged_until}
  let busy = false;

  function mount(el, toolCtx) {
    mountEl = el;
    ctx = toolCtx;
    el.classList.add('studio-tool-image');
    el.innerHTML = `
      <div class="studio-tool-image-tabs" role="tablist">
        <button type="button" role="tab" class="studio-tool-image-tab" data-tab="library" aria-selected="true">Library</button>
        <button type="button" role="tab" class="studio-tool-image-tab" data-tab="search" aria-selected="false">Search</button>
        <button type="button" role="tab" class="studio-tool-image-tab" data-tab="generate" aria-selected="false">Generate</button>
      </div>
      <div class="studio-tool-image-panel"></div>
    `;
    el.addEventListener('click', _onClick);
    el.addEventListener('keydown', _onKeydown);
    _renderTab('library');
  }

  function unmount(el) {
    // If the user navigated away mid-staging without acting, discard.
    // The sweep would catch it eventually, but discarding now reclaims
    // the library slot immediately.
    if (staged?.gen_id && ctx?.api?.discardStaged) {
      ctx.api.discardStaged(staged.gen_id).catch(() => {});
    }
    staged = null;
    el?.removeEventListener('click', _onClick);
    el?.removeEventListener('keydown', _onKeydown);
    if (el) {
      el.classList.remove('studio-tool-image');
      el.innerHTML = '';
    }
    mountEl = null;
    ctx = null;
    busy = false;
    activeTab = 'library';
  }

  function onCtxChange(nextCtx) {
    ctx = nextCtx;
    // If the focused slot changed, re-render the active tab so e.g. Search
    // pre-fills the new title.
    if (mountEl) _renderTab(activeTab);
  }

  // ----- Tab routing ---------------------------------------------------------
  function _onClick(e) {
    const tabBtn = e.target.closest('[data-tab]');
    if (tabBtn) {
      _switchTab(tabBtn.dataset.tab);
      return;
    }
    const tile = e.target.closest('[data-pick-url]');
    if (tile) {
      const appendBtn = e.target.closest('[data-append]');
      if (appendBtn) {
        _pickUrl(tile.dataset.pickUrl, /*append=*/true);
        e.stopPropagation();
        return;
      }
      _pickUrl(tile.dataset.pickUrl, /*append=*/false);
      return;
    }
    const action = e.target.closest('[data-action]');
    if (action) {
      _handleAction(action.dataset.action, action);
    }
  }

  function _onKeydown(e) {
    if (e.key === 'Enter' && e.target.matches('[data-tab]')) {
      _switchTab(e.target.dataset.tab);
    }
  }

  function _switchTab(tab) {
    if (busy) return;
    if (!['library', 'search', 'generate'].includes(tab)) return;
    activeTab = tab;
    mountEl.querySelectorAll('[data-tab]').forEach((btn) => {
      btn.setAttribute('aria-selected', btn.dataset.tab === tab ? 'true' : 'false');
    });
    _renderTab(tab);
  }

  function _panel() { return mountEl.querySelector('.studio-tool-image-panel'); }

  function _renderTab(tab) {
    if (tab === 'library') return _renderLibrary();
    if (tab === 'search') return _renderSearch();
    if (tab === 'generate') return _renderGenerate();
  }

  function _pickUrl(url, append) {
    if (!url || !ctx?.onSlotChange) return;
    ctx.onSlotChange(url, { append });
  }

  function _handleAction(name, btn) {
    if (name === 'search-go') return _doSearch();
    if (name === 'generate-go') return _doGenerate();
    if (name === 'use-staged') return _commitStaged(/*place=*/true);
    if (name === 'save-staged') return _commitStaged(/*place=*/false);
    if (name === 'regenerate-staged') return _regenerateStaged();
  }

  // ----- Library tab ---------------------------------------------------------
  async function _renderLibrary() {
    const panel = _panel();
    panel.innerHTML = `<div class="studio-tool-image-loading">Loading library…</div>`;
    let images = [];
    try {
      images = (await ctx.api.listLibrary?.()) || [];
    } catch (err) {
      panel.innerHTML = `<div class="studio-tool-image-empty">Could not load image library.</div>`;
      return;
    }
    if (!images.length) {
      panel.innerHTML = `
        <div class="studio-tool-image-empty">
          No images in your library yet.<br>
          Generate one in the Generate tab, or search the web in Search.
        </div>`;
      return;
    }
    panel.innerHTML = `
      <div class="studio-tool-image-grid">
        ${images.map((img) => _tileHtml({
          url: img.download_url || img.url || '',
          thumb: img.thumb_url || img.download_url || img.url || '',
          caption: img.display_name || img.filename || '',
          allowAppend: ctx?.supportsAppend === true,
        })).join('')}
      </div>
    `;
  }

  // ----- Search tab ----------------------------------------------------------
  function _renderSearch() {
    const panel = _panel();
    const preFill = _focusedTitle();
    panel.innerHTML = `
      <div class="studio-tool-image-form">
        <label>Search the web</label>
        <input type="text" data-search-input value="${escapeHtml(preFill)}" placeholder="e.g. solar cost decline chart IRENA">
        <div class="studio-tool-image-form-row">
          <label style="display:flex;align-items:center;gap:6px;font-size:12px;text-transform:none;">
            <input type="checkbox" data-prefer-charts>
            Prefer charts &amp; diagrams
          </label>
          <button type="button" class="studio-tool-image-action" data-action="search-go">Search</button>
        </div>
      </div>
      <div class="studio-tool-image-results"></div>
    `;
    // Pressing Enter in the search box fires Search.
    const input = panel.querySelector('[data-search-input]');
    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') _doSearch();
    });
    if (input && !preFill) input.focus();
  }

  async function _doSearch() {
    const panel = _panel();
    const input = panel.querySelector('[data-search-input]');
    const charts = panel.querySelector('[data-prefer-charts]');
    const results = panel.querySelector('.studio-tool-image-results');
    const query = (input?.value || '').trim();
    if (!query) { input?.focus(); return; }
    if (busy) return;
    busy = true;
    results.innerHTML = `<div class="studio-tool-image-loading">Searching…</div>`;
    try {
      const data = await ctx.api.searchImages({
        query,
        count: 4,
        prefer_charts: !!charts?.checked,
      });
      const cands = data?.candidates || [];
      if (!cands.length) {
        results.innerHTML = `<div class="studio-tool-image-empty">No results for "${escapeHtml(query)}". Try different terms.</div>`;
      } else {
        results.innerHTML = `
          <div class="studio-tool-image-grid">
            ${cands.map((c) => _tileHtml({
              url: c.embed_url,
              thumb: c.thumb_url,
              caption: c.source || c.title || '',
              allowAppend: ctx?.supportsAppend === true,
            })).join('')}
          </div>
        `;
      }
    } catch (err) {
      results.innerHTML = `<div class="studio-tool-image-empty">Search failed: ${escapeHtml(String(err))}</div>`;
    } finally {
      busy = false;
    }
  }

  // ----- Generate tab --------------------------------------------------------
  function _renderGenerate() {
    const panel = _panel();
    if (staged) { _renderStaged(); return; }
    const preFill = _focusedTitle();
    const styleOpts = STYLE_PRESETS.map((s) => `<option value="${escapeHtml(s.value)}">${escapeHtml(s.label)}</option>`).join('');
    const aspectOpts = ASPECT_PRESETS.map((s) => `<option value="${escapeHtml(s.value)}">${escapeHtml(s.label)}</option>`).join('');
    panel.innerHTML = `
      <div class="studio-tool-image-form">
        <label>Generate an image</label>
        <textarea data-gen-prompt placeholder="Describe what you want to see — subject, setting, mood">${escapeHtml(preFill)}</textarea>
        <div class="studio-tool-image-form-row">
          <select data-gen-style aria-label="Style">${styleOpts}</select>
          <select data-gen-aspect aria-label="Aspect">${aspectOpts}</select>
        </div>
        <button type="button" class="studio-tool-image-action" data-action="generate-go">Generate</button>
      </div>
    `;
    const ta = panel.querySelector('[data-gen-prompt]');
    if (ta && !preFill) ta.focus();
  }

  async function _doGenerate() {
    if (busy) return;
    const panel = _panel();
    const prompt = (panel.querySelector('[data-gen-prompt]')?.value || '').trim();
    const style = panel.querySelector('[data-gen-style]')?.value || '';
    const aspect = panel.querySelector('[data-gen-aspect]')?.value || 'square';
    if (!prompt) {
      panel.querySelector('[data-gen-prompt]')?.focus();
      return;
    }
    busy = true;
    panel.innerHTML = `<div class="studio-tool-image-loading">Generating…</div>`;
    try {
      const data = await ctx.api.generateImage({ prompt, style, aspect });
      staged = {
        gen_id: data.gen_id,
        embed_url: data.embed_url,
        prompt_used: data.prompt_used || prompt,
        style, aspect,
      };
      _renderStaged();
    } catch (err) {
      panel.innerHTML = `
        <div class="studio-tool-image-empty">Generation failed: ${escapeHtml(String(err))}</div>
        <button type="button" class="studio-tool-image-action secondary" data-action="regenerate-staged">Try again</button>
      `;
    } finally {
      busy = false;
    }
  }

  function _renderStaged() {
    const panel = _panel();
    panel.innerHTML = `
      <div class="studio-tool-image-staged">
        <div class="studio-tool-image-staged-preview">
          <img src="${escapeHtml(staged.embed_url)}" alt="Generated preview">
        </div>
        <div class="studio-tool-image-staged-meta">
          ${escapeHtml(staged.prompt_used)}
        </div>
        <div class="studio-tool-image-staged-actions">
          <button type="button" class="studio-tool-image-action" data-action="use-staged">Use it</button>
          <button type="button" class="studio-tool-image-action secondary" data-action="save-staged">Save to library</button>
          <button type="button" class="studio-tool-image-action secondary" data-action="regenerate-staged">Regenerate</button>
        </div>
      </div>
    `;
  }

  async function _commitStaged(place) {
    if (!staged?.gen_id) return;
    const url = staged.embed_url;
    try {
      await ctx.api.commitStaged(staged.gen_id);
    } catch (err) {
      console.warn('commit staged failed', err);
    }
    if (place) _pickUrl(url, false);
    staged = null;
    // After Use: show the search tab so the user can try alternatives.
    // After Save: stay on Generate with a fresh form so they can iterate.
    if (place) _switchTab('search'); else _renderGenerate();
  }

  async function _regenerateStaged() {
    if (!staged?.gen_id) return _renderGenerate();
    const prevPrompt = staged.prompt_used;
    const style = staged.style;
    const aspect = staged.aspect;
    try {
      await ctx.api.discardStaged(staged.gen_id);
    } catch (err) {
      console.warn('discard staged failed', err);
    }
    staged = null;
    // Replay the generation with the same prompt — fresh seed at the engine.
    const panel = _panel();
    panel.innerHTML = `<div class="studio-tool-image-loading">Regenerating…</div>`;
    busy = true;
    try {
      const data = await ctx.api.generateImage({
        prompt: prevPrompt, style, aspect,
      });
      staged = {
        gen_id: data.gen_id, embed_url: data.embed_url,
        prompt_used: data.prompt_used || prevPrompt, style, aspect,
      };
      _renderStaged();
    } catch (err) {
      panel.innerHTML = `
        <div class="studio-tool-image-empty">Generation failed: ${escapeHtml(String(err))}</div>
        <button type="button" class="studio-tool-image-action secondary" data-action="regenerate-staged">Try again</button>
      `;
    } finally {
      busy = false;
    }
  }

  // ----- Helpers -------------------------------------------------------------
  function _focusedTitle() {
    const slot = ctx?.getFocusSlot?.();
    return slot?.suggestedQuery || '';
  }

  function _tileHtml({ url, thumb, caption, allowAppend }) {
    const safeUrl = escapeHtml(url);
    const safeThumb = escapeHtml(thumb || url);
    const safeCaption = escapeHtml(caption || '');
    return `
      <button type="button" class="studio-tool-image-tile" data-pick-url="${safeUrl}">
        <img src="${safeThumb}" alt="${safeCaption}" loading="lazy">
        ${safeCaption ? `<span class="studio-tool-image-tile-caption">${safeCaption}</span>` : ''}
        ${allowAppend ? `<button type="button" class="studio-tool-image-tile-append" data-append title="Add as additional image">+</button>` : ''}
      </button>
    `;
  }

  return {
    id: 'image',
    label: 'Image',
    mount,
    unmount,
    onCtxChange,
  };
}
