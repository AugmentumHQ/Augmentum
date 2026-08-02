/* ==========================================================================
   Flow Create Sheet — dedicated creation UI for reasoning flows.

   Replaces the overlay's in-place "+ New from Template" button and the
   raw native prompt() fallback.  One entry point, one styled sheet, one
   atomic POST.  Shared by the flow-bar picker and the inspector
   FlowEditor so both paths land in the same experience.

   Flow:
     1. User types a name (required, trimmed, <= MAX_NAME_LEN)
     2. User picks a starting template (Blank by default)
     3. Create:
        - Blank   → POST /api/reasoning/flows
        - Template → POST /api/reasoning/templates/{name}/create
                     then PUT /api/reasoning/flows/{id} to apply the user's name
     4. On success: showToast → close → onCreated(flow)
        Default onCreated opens the flow-editor overlay with the new flow selected.
   ========================================================================== */

import { escapeHtml, showToast } from './app.js';
import { openSheet } from './sheet.js';

// Icon map — keyed by template.name returned by /api/reasoning/templates.
// These are built-in names; user-created flows never appear here.
// Unicode escapes (not raw glyphs) to keep the file ASCII-safe across editors.
const TEMPLATE_ICONS = {
  auto_routing:            '\u2699',           // gear
  quick_answer:            '\u26A1',           // bolt
  research:                '\uD83D\uDD2C',     // microscope
  code_review:             '\uD83E\uDDE9',     // puzzle piece
  math:                    '\u03A3',           // sigma
  creative:                '\u2728',           // sparkles
  debate:                  '\u2696',           // scales
  explainer:               '\uD83D\uDCA1',     // light bulb
  live_lookup:             '\uD83D\uDCE1',     // satellite antenna
  summarize:               '\uD83D\uDCDD',     // memo
  agentic_report:          '\uD83D\uDCCB',     // clipboard
  agentic_presentation:    '\uD83D\uDCCA',     // bar chart
  agentic_storybook:       '\uD83D\uDCDA',     // books
  agentic_data_comparison: '\uD83D\uDD0D',     // magnifier
  agentic_fact_checker:    '\u2713\u2713',     // double check
  agentic_tutorial:        '\uD83D\uDCD6',     // open book
};

const MAX_NAME_LEN = 80;
const FALLBACK_ICON = '\u25C6';                // diamond

/**
 * Open the "create a reasoning flow" sheet.
 *
 * @param {object} opts
 * @param {'analytical'|'agentic'} [opts.mode='analytical']
 * @param {function(object):void} [opts.onCreated] - called with the new flow;
 *   defaults to opening the flow-editor overlay pinned to the new flow.
 */
export function openFlowCreateSheet({ mode = 'analytical', onCreated } = {}) {
  const isAgentic = mode === 'agentic';
  const accent = isAgentic ? 'var(--mode-agentic)' : 'var(--mode-analytical)';
  const modeLabel = isAgentic ? 'Creator' : 'Thinker';
  const subtitle = isAgentic
    ? 'Design a workflow that plans, creates, and delivers artifacts.'
    : 'Design a multi-step reasoning pipeline that classifies, researches, and verifies.';

  // --- Body --------------------------------------------------------------
  const body = document.createElement('div');
  body.className = 'flow-create-sheet__body';
  body.innerHTML = `
    <p class="flow-create-sheet__subtitle">${escapeHtml(subtitle)}</p>

    <label class="flow-create-sheet__field">
      <span class="flow-create-sheet__label">Name</span>
      <input class="flow-create-sheet__name-input" type="text"
             placeholder="My ${escapeHtml(modeLabel)} Flow"
             maxlength="${MAX_NAME_LEN}"
             autocomplete="off" spellcheck="false"
             aria-describedby="flow-create-sheet-hint">
      <span class="flow-create-sheet__hint" id="flow-create-sheet-hint"></span>
    </label>

    <div class="flow-create-sheet__section-label" id="flow-create-sheet-templates-label">
      Start from
    </div>
    <div class="flow-create-sheet__templates"
         role="radiogroup"
         aria-labelledby="flow-create-sheet-templates-label">
      <div class="flow-create-sheet__templates-status">Loading templates\u2026</div>
    </div>
  `;

  // --- Footer ------------------------------------------------------------
  const footer = document.createElement('div');
  footer.className = 'flow-create-sheet__actions';
  footer.innerHTML = `
    <button class="flow-create-sheet__btn flow-create-sheet__btn--ghost" type="button"
            data-role="cancel">Cancel</button>
    <button class="flow-create-sheet__btn flow-create-sheet__btn--primary" type="button"
            data-role="create" disabled>Create</button>
  `;

  const { close, root } = openSheet({
    title: `New ${modeLabel} Flow`,
    body,
    footer,
    className: `flow-create-sheet flow-create-sheet--${mode}`,
    accent,
    initialFocus: '.flow-create-sheet__name-input',
  });

  // --- Element refs ------------------------------------------------------
  const nameInput  = root.querySelector('.flow-create-sheet__name-input');
  const hintEl     = root.querySelector('.flow-create-sheet__hint');
  const templatesEl = root.querySelector('.flow-create-sheet__templates');
  const cancelBtn  = root.querySelector('[data-role="cancel"]');
  const createBtn  = root.querySelector('[data-role="create"]');

  // --- State -------------------------------------------------------------
  let selectedTemplate = 'blank';   // 'blank' or a template name
  let creating = false;
  let templatesLoaded = [];         // populated after fetch resolves

  // --- Helpers -----------------------------------------------------------
  function updateCreateEnabled() {
    const ok = !!nameInput.value.trim() && !creating;
    createBtn.disabled = !ok;
  }

  function setHint(msg, isError = false) {
    hintEl.textContent = msg || '';
    hintEl.classList.toggle('flow-create-sheet__hint--error', !!isError);
  }

  function renderTemplates(templates) {
    templatesLoaded = templates;

    const blankCard = `
      <label class="flow-create-sheet__template flow-create-sheet__template--blank${
        selectedTemplate === 'blank' ? ' is-selected' : ''
      }">
        <input type="radio" name="flow-template" value="blank"
               ${selectedTemplate === 'blank' ? 'checked' : ''}>
        <span class="flow-create-sheet__template-icon" aria-hidden="true">+</span>
        <span class="flow-create-sheet__template-name">Blank Flow</span>
        <span class="flow-create-sheet__template-desc">Start from scratch. No steps preloaded.</span>
      </label>
    `;

    const cards = templates.map(t => {
      const icon = TEMPLATE_ICONS[t.name] || FALLBACK_ICON;
      const isSel = selectedTemplate === t.name ? ' is-selected' : '';
      const stepCount = Number.isFinite(t.step_count) ? t.step_count : 0;
      return `
        <label class="flow-create-sheet__template${isSel}">
          <input type="radio" name="flow-template" value="${escapeHtml(t.name)}"
                 ${selectedTemplate === t.name ? 'checked' : ''}>
          <span class="flow-create-sheet__template-icon" aria-hidden="true">${icon}</span>
          <span class="flow-create-sheet__template-name">${escapeHtml(t.display_name || t.name)}</span>
          <span class="flow-create-sheet__template-desc">${escapeHtml(t.description || '')}</span>
          <span class="flow-create-sheet__template-meta">${stepCount} step${stepCount === 1 ? '' : 's'}</span>
        </label>
      `;
    }).join('');

    templatesEl.innerHTML = blankCard + cards;

    templatesEl.querySelectorAll('input[name="flow-template"]').forEach(input => {
      input.addEventListener('change', () => {
        if (!input.checked) return;
        selectedTemplate = input.value;
        templatesEl.querySelectorAll('.flow-create-sheet__template').forEach(el => {
          const radio = el.querySelector('input[name="flow-template"]');
          el.classList.toggle('is-selected', !!radio && radio.value === selectedTemplate);
        });
      });
    });
  }

  // --- Load templates ----------------------------------------------------
  (async () => {
    try {
      const resp = await fetch('/api/reasoning/templates');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const all = await resp.json();
      const filtered = isAgentic
        ? all.filter(t => typeof t.name === 'string' && t.name.startsWith('agentic'))
        : all.filter(t => typeof t.name === 'string' && !t.name.startsWith('agentic'));
      renderTemplates(filtered);
    } catch (e) {
      console.warn('[flow-create-sheet] Failed to load templates:', e.message || e);
      // Still let the user proceed with a Blank flow.
      templatesEl.innerHTML = `
        <div class="flow-create-sheet__templates-status flow-create-sheet__templates-status--error">
          Couldn\u2019t load templates. You can still create a blank flow.
        </div>
      `;
      renderTemplates([]);
    }
  })();

  // --- Bindings ----------------------------------------------------------
  nameInput.addEventListener('input', () => {
    setHint('');
    updateCreateEnabled();
  });

  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!createBtn.disabled) handleCreate();
    }
  });

  cancelBtn.addEventListener('click', () => close());
  createBtn.addEventListener('click', () => handleCreate());

  // --- Create flow -------------------------------------------------------
  async function handleCreate() {
    const name = nameInput.value.trim();
    if (!name) {
      setHint('Name is required', true);
      nameInput.focus();
      return;
    }
    if (creating) return;
    creating = true;
    updateCreateEnabled();
    createBtn.classList.add('is-loading');
    createBtn.textContent = 'Creating\u2026';

    try {
      const flow = selectedTemplate === 'blank'
        ? await _createBlank(name, isAgentic)
        : await _createFromTemplate(selectedTemplate, name);

      showToast(`Created: ${flow.name}`, 'success');
      close();

      // Broadcast so any cached flow lists (flow-bar, other surfaces) can refresh.
      document.dispatchEvent(new CustomEvent('augmentum:flow-created', {
        detail: { flow, mode },
      }));

      if (typeof onCreated === 'function') {
        onCreated(flow);
      } else if (typeof window !== 'undefined' && window.openFlowEditorOverlay) {
        window.openFlowEditorOverlay(mode, flow.id);
      }
    } catch (e) {
      const msg = e && e.message ? e.message : 'Unknown error';
      console.warn('[flow-create-sheet] Creation failed:', msg);
      setHint(`Couldn\u2019t create flow: ${msg}`, true);
      creating = false;
      createBtn.classList.remove('is-loading');
      createBtn.textContent = 'Create';
      updateCreateEnabled();
    }
  }

  updateCreateEnabled();
}

// ---------------------------------------------------------------------------
// Internal API helpers
// ---------------------------------------------------------------------------

async function _createBlank(name, isAgentic) {
  const resp = await fetch('/api/reasoning/flows', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name,
      description: '',
      steps: [],
      trigger_domains: isAgentic ? ['agentic'] : [],
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function _createFromTemplate(templateName, name) {
  const createResp = await fetch(
    `/api/reasoning/templates/${encodeURIComponent(templateName)}/create`,
    { method: 'POST' }
  );
  if (!createResp.ok) {
    const err = await createResp.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${createResp.status}`);
  }
  const created = await createResp.json();

  // Backend names it "{template} (copy)" — rename to the user's chosen name.
  if (created.name === name) return created;

  const renameResp = await fetch(
    `/api/reasoning/flows/${encodeURIComponent(created.id)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }
  );
  if (renameResp.ok) {
    return renameResp.json();
  }
  // Rename failed — surface a non-fatal hint but keep the created flow.
  console.warn('[flow-create-sheet] Created but rename failed:', renameResp.status);
  return created;
}
