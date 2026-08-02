/* ==========================================================================
   Toolbar control — Passthrough tools dropdown

   Drives the wrench/tools button in the composer (#tools-toggle-btn). Hosts
   two strata:
     1. Auto-tools toggle  — heuristic SSOS detection (zero LLM overhead).
                            Persisted server-side via `ui.autoTools`.
     2. Per-tool overrides — explicit opt-in by tool name. Persisted in
                            localStorage under `augmentum-passthrough-tools`.
                            Mirrored at runtime on `app.state.passthroughTools`
                            so chat/index.js + chat-surface.js can read it
                            when building the per-turn `tools=` parameter.

   Visibility is gated by app.js::applyMode for now (chat/analyze/build only;
   not narrative/coder). Step 4 will move that per-surface.

   Step 2 of the surface-owned composer migration.
   ========================================================================== */

import { app } from '../../app.js';
import { tbFind } from './util.js';

const PT_TOOLS_KEY = 'augmentum-passthrough-tools';
const PT_TOOLS_ENABLED_KEY = 'augmentum-passthrough-tools-enabled';

// Friendly display metadata for each tool shown in the dropdown.
// Utility tools (calculator, datetime, etc.) are auto-included by the
// backend when any tool is active, so they don't appear here.
const _TOOL_DISPLAY = {
  web:                  { icon: '🌐', label: 'Web Search',          desc: 'Search the web and read pages' },
  python_exec:          { icon: '🐍', label: 'Python',               desc: 'Run code in a sandbox' },
  image_generation:     { icon: '🎨', label: 'Image Generation',     desc: 'Create images from text prompts' },
  build_application:    { icon: '🏗️', label: 'App Builder',          desc: 'Build complete web apps from a description' },
  create_ebook:         { icon: '📕', label: 'Ebook Creator',        desc: 'Write and illustrate complete ebooks' },
  create_chart:         { icon: '📈', label: 'Charts',               desc: 'Draw charts and graphs from data' },
  create_spreadsheet:   { icon: '🧮', label: 'Spreadsheets',         desc: 'Build .xlsx spreadsheets from data' },
  wikipedia:            { icon: '📖', label: 'Wikipedia',            desc: 'Look up facts and articles' },
  youtube:              { icon: '▶️', label: 'YouTube',              desc: 'Find and watch videos with transcripts' },
  file_ops:             { icon: '📂', label: 'File Manager',          desc: 'Read, write, and organize files' },
  document_parse:       { icon: '📄', label: 'Document Reader',      desc: 'Extract text from PDF, DOCX, XLSX' },
  image_search:         { icon: '🖼️', label: 'Image Search',         desc: 'Find images from the web' },
  search_files:         { icon: '🔍', label: 'Search Files',          desc: 'Find your artifacts, documents, and uploads' },
  export_markdown:      { icon: '📝', label: 'Export Markdown',      desc: 'Save content as .md files' },
  export_csv:           { icon: '📊', label: 'Export CSV',            desc: 'Save data as .csv files' },
  export_code:          { icon: '💾', label: 'Export Code',           desc: 'Save code as files' },
};

// Tools that should not appear in the passthrough tools menu.
// Internal tools, utilities auto-included, or tools that only make sense in analytical/agentic.
const _TOOL_HIDDEN_FROM_MENU = new Set([
  'hash', 'json_tool', 'text_analysis', 'consistency_check', 'draft_section',
  // Backend plumbing — auto-available when needed, not user-toggled
  'file_ops', 'document_parse', 'search_files',
  'export_markdown', 'export_csv', 'export_code',
]);

// The single allowlist of user-selectable tools: tools we've explicitly
// curated with a dedicated icon + label (_TOOL_DISPLAY), MINUS the
// backend-plumbing tools that are auto-included rather than user-toggled
// (_TOOL_HIDDEN_FROM_MENU). The /api/config/passthrough-tools endpoint also
// returns companion / agentic / internal tools that are NOT meant for
// user-side selection; without an allowlist those leaked into the composer
// as a bare ⚙ + raw tool name. This is now the SINGLE source of truth for
// what the menu shows and what may be active — any future backend tool is
// excluded by default until it's given a dedicated _TOOL_DISPLAY entry.
const _USER_SELECTABLE_TOOLS = new Set(
  Object.keys(_TOOL_DISPLAY).filter((name) => !_TOOL_HIDDEN_FROM_MENU.has(name)),
);

// Presentation order for tool categories (human-friendly group names).
// Most-used categories first: search/code/images are top-of-menu.
const _CATEGORY_ORDER = ['search', 'execute', 'image', 'artifact', 'fetch', 'file', 'verify'];
const _CATEGORY_LABELS = {
  search:   'Search & Lookup',
  execute:  'Code & Build',
  image:    'Images',
  artifact: 'Create & Export',
  fetch:    'Content',
  file:     'Files',
  verify:   'Utilities',
};

/**
 * Hydrate `app.state.passthroughTools` from localStorage on first call.
 * Safe to re-call: subsequent calls overwrite from storage, which matches
 * the previous app.js::initPassthroughTools semantics.
 */
function _hydratePassthroughToolsState() {
  const saved = localStorage.getItem(PT_TOOLS_KEY);
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      // Filter out the legacy "all" marker AND any tool that isn't user-
      // selectable — drops stale/leaked companion-tool selections so they
      // can never be sent on a turn, not just hidden from the menu.
      app.state.passthroughTools = Array.isArray(parsed)
        ? parsed.filter(t => t !== 'all' && _USER_SELECTABLE_TOOLS.has(t))
        : [];
    } catch {
      app.state.passthroughTools = [];
    }
  } else {
    // First use — no individual tools enabled, Auto handles common queries
    app.state.passthroughTools = [];
  }
}

function _toggleTool(name, enabled, btn) {
  const list = app.state.passthroughTools || [];
  if (enabled) {
    if (!list.includes(name)) list.push(name);
  } else {
    app.state.passthroughTools = list.filter(t => t !== name);
  }
  localStorage.setItem(PT_TOOLS_KEY, JSON.stringify(app.state.passthroughTools));
  localStorage.setItem(
    PT_TOOLS_ENABLED_KEY,
    app.state.passthroughTools.length > 0 ? 'true' : 'false',
  );
  if (btn) btn.dataset.active = app.state.passthroughTools.length > 0 ? 'true' : 'false';
}

async function _loadToolsList(listEl, toolsBtn = null) {
  if (!listEl) return;

  try {
    const resp = await fetch('/api/config/passthrough-tools');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    // Fetch current auto-tools state from per-user UI settings.
    // Stored under `ui.autoTools` in user_settings; default is off.
    let autoToolsEnabled = false;
    try {
      const uiResp = await fetch('/api/config/ui');
      if (uiResp.ok) {
        const ui = await uiResp.json();
        autoToolsEnabled = ui.autoTools === 'true' || ui.autoTools === true;
      }
    } catch { /* ignore */ }

    listEl.innerHTML = '';

    // --- Auto-tools toggle (always first, default off) ---
    // Auto handles search, calc, URL fetch, code execution via heuristic
    // detection — zero LLM overhead, no tool schemas injected. Per-user.
    const autoDiv = document.createElement('div');
    autoDiv.className = 'tool-auto-toggle';
    const autoLabel = document.createElement('label');
    autoLabel.className = 'tool-option tool-auto-option' + (autoToolsEnabled ? ' checked' : '');
    autoLabel.title = 'Math, unit conversions, and URLs run instantly. For web search, images, videos, and Wikipedia, the model decides when they help — no tool-calling overhead when they don\'t. Works in all chats.';
    const autoCb = document.createElement('input');
    autoCb.type = 'checkbox';
    autoCb.checked = autoToolsEnabled;
    autoCb.addEventListener('change', async () => {
      const desired = autoCb.checked;
      autoLabel.classList.toggle('checked', desired);
      let saved = false;
      try {
        const r = await fetch('/api/config/ui', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ autoTools: desired ? 'true' : 'false' }),
        });
        saved = r.ok;
      } catch { saved = false; }
      if (!saved) {
        // Revert UI to match server state on failure (auth lost, etc.)
        autoCb.checked = !desired;
        autoLabel.classList.toggle('checked', !desired);
      }
    });
    const autoIcon = document.createElement('span');
    autoIcon.className = 'tool-option-icon';
    autoIcon.textContent = '⚡';
    const autoText = document.createElement('span');
    autoText.className = 'tool-option-text';
    const autoName = document.createElement('span');
    autoName.className = 'tool-option-label';
    autoName.textContent = 'Auto';
    const autoDesc = document.createElement('span');
    autoDesc.className = 'tool-option-desc';
    autoDesc.textContent = 'Math & URLs instant; web, images, video when they help';
    autoText.appendChild(autoName);
    autoText.appendChild(autoDesc);
    autoLabel.appendChild(autoCb);
    autoLabel.appendChild(autoIcon);
    autoLabel.appendChild(autoText);
    autoDiv.appendChild(autoLabel);
    listEl.appendChild(autoDiv);

    // --- Tool chain limit ---
    // How many tool round-trips the model may take in one turn. The
    // user's call, not ours: a frontier model with a long context can
    // chain far more than a local 12B, and when the cap is hit mid-plan
    // the model has its tools stripped and can leak a tool call into its
    // prose. Blank = install default. 0 = unlimited (a 150 backstop still
    // applies so a genuinely stuck turn can't run forever).
    let chainLimit = '';
    try {
      const uiResp2 = await fetch('/api/config/ui');
      if (uiResp2.ok) {
        const ui2 = await uiResp2.json();
        chainLimit = ui2.toolChainLimit == null ? '' : String(ui2.toolChainLimit);
      }
    } catch { /* leave blank — falls back to the install default */ }

    const chainDiv = document.createElement('div');
    chainDiv.className = 'tool-chain-limit';
    const chainSummary = document.createElement('button');
    chainSummary.type = 'button';
    chainSummary.className = 'tool-chain-summary';
    chainSummary.innerHTML =
      '<span class="tool-option-icon">\u{1F517}</span>'
      + '<span class="tool-option-text">'
      + '<span class="tool-option-label">Tool chain limit</span>'
      + '<span class="tool-option-desc tool-chain-current"></span>'
      + '</span>';
    const chainBody = document.createElement('div');
    chainBody.className = 'tool-chain-body';
    chainBody.hidden = true;
    const chainInput = document.createElement('input');
    chainInput.type = 'number';
    chainInput.min = '0';
    chainInput.max = '150';
    chainInput.className = 'tool-chain-input';
    chainInput.placeholder = 'default';
    chainInput.value = chainLimit;
    const chainHint = document.createElement('span');
    chainHint.className = 'tool-chain-hint';
    chainHint.textContent = 'Blank = default. 0 = unlimited.';
    const chainStatus = document.createElement('span');
    chainStatus.className = 'tool-chain-status';

    const renderCurrent = () => {
      const v = (chainInput.value || '').trim();
      const cur = chainSummary.querySelector('.tool-chain-current');
      if (!cur) return;
      if (v === '') cur.textContent = 'Using the default';
      else if (v === '0') cur.textContent = 'Unlimited (150 max)';
      else cur.textContent = `${v} round-trip${v === '1' ? '' : 's'} per turn`;
    };
    renderCurrent();

    chainSummary.addEventListener('click', () => {
      chainBody.hidden = !chainBody.hidden;
      chainSummary.classList.toggle('open', !chainBody.hidden);
      if (!chainBody.hidden) chainInput.focus();
    });

    let chainTimer = null;
    const saveChain = async () => {
      let v = (chainInput.value || '').trim();
      if (v !== '') {
        const n = Number(v);
        // Reject rather than silently clamp — a value the user typed that
        // quietly becomes a different one is worse than being told no.
        if (!Number.isInteger(n) || n < 0 || n > 150) {
          chainStatus.textContent = 'Enter a whole number from 0 to 150.';
          chainStatus.className = 'tool-chain-status error';
          return;
        }
        v = String(n);
      }
      chainStatus.textContent = 'Saving…';
      chainStatus.className = 'tool-chain-status';
      try {
        const r = await fetch('/api/config/ui', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ toolChainLimit: v }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        chainStatus.textContent = 'Saved';
        chainStatus.className = 'tool-chain-status ok';
        renderCurrent();
      } catch (e) {
        chainStatus.textContent = `Couldn't save: ${e.message}`;
        chainStatus.className = 'tool-chain-status error';
      }
    };
    chainInput.addEventListener('input', () => {
      renderCurrent();
      clearTimeout(chainTimer);
      chainTimer = setTimeout(saveChain, 600);
    });

    chainBody.appendChild(chainInput);
    chainBody.appendChild(chainHint);
    chainBody.appendChild(chainStatus);
    chainDiv.appendChild(chainSummary);
    chainDiv.appendChild(chainBody);
    listEl.appendChild(chainDiv);

    if (!data.tools || data.tools.length === 0) {
      return;
    }

    // Section header for individual tool overrides
    const overrideHeader = document.createElement('div');
    overrideHeader.className = 'tool-group-header';
    overrideHeader.textContent = 'Override — enable for specific requests';
    listEl.appendChild(overrideHeader);

    // Group by category in presentation order. ALLOWLIST: only tools we've
    // curated with a dedicated icon/label are user-selectable; everything
    // else the backend returns (companion/agentic/internal/flow_* tools) is
    // excluded so it can't leak into the composer as a bare ⚙ + raw name.
    const groups = {};
    for (const tool of data.tools) {
      if (!_USER_SELECTABLE_TOOLS.has(tool.name)) continue;
      const cat = tool.category || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(tool);
    }

    const orderedCats = _CATEGORY_ORDER.filter(c => groups[c]);
    // Append any unknown categories at the end
    for (const cat of Object.keys(groups)) {
      if (!orderedCats.includes(cat)) orderedCats.push(cat);
    }

    for (const cat of orderedCats) {
      const tools = groups[cat];
      if (!tools) continue;

      // Category section header
      const header = document.createElement('div');
      header.className = 'tool-group-header';
      header.textContent = _CATEGORY_LABELS[cat] || cat;
      listEl.appendChild(header);

      for (const tool of tools) {
        const isChecked = (app.state.passthroughTools || []).includes(tool.name);
        const meta = _TOOL_DISPLAY[tool.name] || {};
        const isHealthy = tool.healthy !== false;

        const opt = document.createElement('label');
        opt.className = 'tool-option' + (isChecked ? ' checked' : '') + (!isHealthy ? ' tool-unhealthy' : '');
        opt.title = isHealthy ? tool.description : `${tool.description}\n⚠ Service unavailable — ${(tool.requires || []).join(', ') || 'dependency'} not running`;

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = isChecked;
        cb.dataset.toolName = tool.name;
        cb.addEventListener('change', () => {
          _toggleTool(tool.name, cb.checked, toolsBtn);
          opt.classList.toggle('checked', cb.checked);
        });

        const icon = document.createElement('span');
        icon.className = 'tool-option-icon';
        icon.textContent = meta.icon || '⚙';

        const textWrap = document.createElement('span');
        textWrap.className = 'tool-option-text';

        const label = document.createElement('span');
        label.className = 'tool-option-label';
        label.textContent = meta.label || tool.name.replace(/_/g, ' ');
        if (!isHealthy) {
          const warn = document.createElement('span');
          warn.className = 'tool-health-warn';
          warn.textContent = ' (offline)';
          label.appendChild(warn);
        }

        const desc = document.createElement('span');
        desc.className = 'tool-option-desc';
        desc.textContent = meta.desc || '';

        textWrap.appendChild(label);
        if (meta.desc) textWrap.appendChild(desc);
        opt.appendChild(cb);
        opt.appendChild(icon);
        opt.appendChild(textWrap);
        listEl.appendChild(opt);
      }
    }
  } catch {
    listEl.innerHTML = '<div class="tools-dropdown-loading">Failed to load tools</div>';
  }
}

/**
 * Wire the tools toggle button + dropdown inside the given toolbar root.
 * Safe to call once per DISTINCT toolbar root (primary + per-surface clones);
 * all state is closure-scoped. Returns a cleanup function that removes the
 * body-mounted dropdown/backdrop and the document click listener — required
 * for per-surface toolbars so closing a tab doesn't orphan them.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireTools(toolbarEl, surface) {
  _hydratePassthroughToolsState();

  const btn = tbFind(toolbarEl, 'tools-toggle-btn');
  const dropdown = tbFind(toolbarEl, 'tools-dropdown');
  const wrap = tbFind(toolbarEl, 'tools-toggle-wrap');
  if (!btn || !dropdown || !wrap) return undefined;

  // Active state reflects whether any individual tools are checked
  btn.dataset.active = (app.state.passthroughTools || []).length > 0 ? 'true' : 'false';

  let toolsBackdrop = null;
  const _isMobile = () => window.innerWidth < 768;

  // Move dropdown to document.body so it escapes the input-area's
  // backdrop-filter stacking context (fixed children are trapped inside it).
  document.body.appendChild(dropdown);

  const listEl = tbFind(dropdown, 'tools-dropdown-list');

  function _closeToolsDropdown() {
    dropdown.classList.add('hidden');
    if (toolsBackdrop) { toolsBackdrop.remove(); toolsBackdrop = null; }
  }

  function _openToolsDropdown() {
    dropdown.classList.remove('hidden');
    _loadToolsList(listEl, btn);

    // Position the dropdown above the toggle button on desktop.
    // On mobile the CSS bottom-sheet rules take over (bottom:0, full width).
    if (!_isMobile()) {
      const rect = btn.getBoundingClientRect();
      const dropW = 280; // matches CSS width
      const pad = 8;     // minimum gap from viewport edge
      let rightPx = window.innerWidth - rect.right;
      // Clamp so the dropdown's left edge doesn't go offscreen
      const leftEdge = window.innerWidth - rightPx - dropW;
      if (leftEdge < pad) {
        rightPx = window.innerWidth - dropW - pad;
      }
      dropdown.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
      dropdown.style.right = Math.max(pad, rightPx) + 'px';
      dropdown.style.left = '';
    } else {
      dropdown.style.bottom = '';
      dropdown.style.right = '';
    }

    if (!toolsBackdrop) {
      toolsBackdrop = document.createElement('div');
      toolsBackdrop.className = 'tools-backdrop';
      toolsBackdrop.addEventListener('click', _closeToolsDropdown);
      document.body.appendChild(toolsBackdrop);
    }
  }

  // Toggle dropdown on button click
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = !dropdown.classList.contains('hidden');
    if (isOpen) _closeToolsDropdown(); else _openToolsDropdown();
  });

  // Close dropdown on outside click
  const _outsideClick = (e) => {
    if (!dropdown.contains(e.target) && !btn.contains(e.target)) {
      _closeToolsDropdown();
    }
  };
  document.addEventListener('click', _outsideClick);

  return () => {
    _closeToolsDropdown();
    document.removeEventListener('click', _outsideClick);
    dropdown.remove();
  };
}
