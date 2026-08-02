/**
 * marketplace.js — Provider marketplace.
 *
 * A hub, not a list. Shows what's currently powering Augmentum,
 * offers opinionated quick-start recipes, and presents providers
 * on per-modality rails with a friendly "bring your own" drawer.
 */
import { escapeHtml } from './app.js';

let _overlay = null;
let _catalog = [];
let _services = {};
let _hardware = {};
let _busy = new Set();
let _customOpen = false;
let _customMode = 'quick';

const CATEGORY_META = {
  llm:   { label: 'Language',  verb: 'thinks',   hint: 'Runs text through a model',        accent: 'var(--mp-llm)',   order: 1 },
  tts:   { label: 'Voice Out', verb: 'speaks',   hint: 'Reads answers aloud',              accent: 'var(--mp-tts)',   order: 2 },
  stt:   { label: 'Voice In',  verb: 'listens',  hint: 'Hears what you say',               accent: 'var(--mp-stt)',   order: 3 },
  image: { label: 'Vision',    verb: 'imagines', hint: 'Generates and understands images', accent: 'var(--mp-image)', order: 4 },
};

// Recipes add VOICE to the AI you already run. Augmentum ships its own
// LLM engine (llama.cpp, with session restore) and auto-detects any external
// OpenAI-compatible endpoint — so the marketplace focuses on the pieces it
// doesn't already provide: speaking (TTS), listening (STT), and vision.
const RECIPES = [
  {
    id: 'voice-assistant',
    name: 'Voice Assistant',
    tag: 'Conversational',
    blurb: 'Kokoro TTS + Speaches STT — give your built-in AI a voice. It speaks and listens; the model is already running.',
    gpu_stack: ['kokoro-server', 'speaches-stt'],
    cpu_stack: ['kokoro-server', 'speaches-stt'],
    accent: 'var(--mp-tts)',
  },
  {
    id: 'expressive-voice',
    name: 'Expressive Voice',
    tag: 'Lifelike',
    blurb: 'Chatterbox voice cloning + Speaches STT. Paralinguistic tags ([laugh], [cough]), 23 languages.',
    gpu_stack: ['chatterbox-turbo', 'speaches-stt'],
    cpu_stack: ['chatterbox-tts', 'speaches-stt'],
    accent: 'var(--mp-stt)',
  },
];

// Service-ID prefix → { initials, tint } for faux-logo chips.
const BRANDS = {
  'chatterbox':  { initials: 'CB', tint: '#f97316', fg: '#ffffff' },
  'fish':        { initials: 'Fi', tint: '#0891b2', fg: '#ffffff' },
  'kokoro':      { initials: 'Ko', tint: '#ec4899', fg: '#ffffff' },
  'speaches':    { initials: 'Sp', tint: '#0ea5e9', fg: '#ffffff' },
  'custom':      { initials: '+',  tint: 'var(--bg-tertiary, var(--bg-secondary))', fg: 'var(--text-primary)' },
};

function _brandFor(serviceId) {
  for (const key of Object.keys(BRANDS)) {
    if (serviceId.startsWith(key)) return BRANDS[key];
  }
  return BRANDS.custom;
}

export async function openMarketplace() {
  if (!_overlay) _buildPanel();
  _overlay.classList.add('visible');
  document.body.classList.add('mp-lock-scroll');
  await _refresh();
}

export function closeMarketplace() {
  if (!_overlay) return;
  _overlay.classList.remove('visible');
  document.body.classList.remove('mp-lock-scroll');
}

async function _refresh() {
  try {
    const [catalogRes, servicesRes, hwRes] = await Promise.all([
      fetch('/api/marketplace/catalog'),
      fetch('/api/marketplace/services'),
      fetch('/api/marketplace/hardware'),
    ]);
    if (catalogRes.ok) _catalog = await catalogRes.json();
    if (servicesRes.ok) {
      const list = await servicesRes.json();
      _services = {};
      for (const s of list) _services[s.id] = s;
    }
    if (hwRes.ok) _hardware = await hwRes.json();
  } catch (err) {
    console.error('Marketplace refresh failed:', err);
  }
  _render();
}

function _buildPanel() {
  _overlay = document.createElement('div');
  _overlay.className = 'marketplace-overlay';
  _overlay.innerHTML = `
    <div class="marketplace-panel" role="dialog" aria-labelledby="mp-title" aria-modal="true">
      <header class="mp-header">
        <div class="mp-header-text">
          <h2 id="mp-title">Provider Marketplace</h2>
          <p class="mp-subtitle">Attach the things you want Augmentum to do.</p>
        </div>
        <button class="mp-icon-btn" data-action="close" aria-label="Close">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>
        </button>
      </header>
      <div class="mp-body" id="mp-body"></div>
    </div>
  `;
  document.body.appendChild(_overlay);

  _overlay.addEventListener('click', _onClick);
  _overlay.addEventListener('keydown', _onKey);
}

function _onKey(e) {
  if (e.key === 'Escape') closeMarketplace();
}

function _onClick(e) {
  if (e.target === _overlay) { closeMarketplace(); return; }
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  const id = btn.dataset.id;

  if (action === 'close')          closeMarketplace();
  else if (action === 'enable')    _enable(id);
  else if (action === 'disable')   _disable(id);
  else if (action === 'recipe')    _installRecipe(id);
  else if (action === 'toggle-custom') _toggleCustom();
  else if (action === 'custom-mode')   _setCustomMode(btn.dataset.mode);
  else if (action === 'custom-submit') _submitCustom();
  else if (action === 'custom-parse')  _parseDockerRun();
}

function _render() {
  const body = _overlay.querySelector('#mp-body');
  body.innerHTML = `
    ${_renderPowering()}
    ${_renderRecipes()}
    ${_renderRails()}
    ${_renderCustomSection()}
  `;
}

/* ---------- Currently Powering strip ---------- */

function _renderPowering() {
  const cats = Object.keys(CATEGORY_META).sort((a, b) => CATEGORY_META[a].order - CATEGORY_META[b].order);
  const running = {};
  for (const id of Object.keys(_services)) {
    const s = _services[id];
    if (s.enabled && s.status === 'running') {
      if (!running[s.category]) running[s.category] = [];
      running[s.category].push(s);
    }
  }
  const anyRunning = Object.keys(running).length > 0;

  const chips = cats.map(cat => {
    const meta = CATEGORY_META[cat];
    const list = running[cat] || [];
    if (list.length === 0) {
      return `
        <div class="mp-power-chip empty" style="--accent:${meta.accent}">
          <span class="mp-power-label">${escapeHtml(meta.label)}</span>
          <span class="mp-power-value muted">—</span>
        </div>
      `;
    }
    const name = list.map(s => s.name).join(', ');
    return `
      <div class="mp-power-chip" style="--accent:${meta.accent}">
        <span class="mp-power-label">${escapeHtml(meta.label)}</span>
        <span class="mp-power-value">${escapeHtml(name)}</span>
      </div>
    `;
  }).join('');

  return `
    <section class="mp-powering ${anyRunning ? '' : 'empty'}">
      <div class="mp-powering-header">
        <span class="mp-powering-pulse"></span>
        <span>${anyRunning ? 'Currently powering' : 'Nothing connected yet'}</span>
      </div>
      <div class="mp-powering-grid">${chips}</div>
    </section>
  `;
}

/* ---------- Quick Start Recipes ---------- */

function _renderRecipes() {
  const hasGpu = !!_hardware.gpu_available;
  const cards = RECIPES.map(r => {
    const stack = hasGpu ? r.gpu_stack : r.cpu_stack;
    const resolved = stack.filter(id => _catalog.find(c => c.id === id));
    const unresolved = stack.filter(id => !_catalog.find(c => c.id === id));
    const allEnabled = resolved.length > 0 && resolved.every(id => _services[id]?.enabled);
    const installing = _busy.has(`recipe:${r.id}`);

    const serviceChips = resolved.map(id => {
      const p = _catalog.find(c => c.id === id);
      return `<span class="mp-recipe-chip">${escapeHtml(p?.name || id)}</span>`;
    }).join('');

    return `
      <article class="mp-recipe" style="--accent:${r.accent}" data-installing="${installing}">
        <div class="mp-recipe-tag">${escapeHtml(r.tag)}</div>
        <h3 class="mp-recipe-name">${escapeHtml(r.name)}</h3>
        <p class="mp-recipe-blurb">${escapeHtml(r.blurb)}</p>
        <div class="mp-recipe-stack">${serviceChips}</div>
        ${unresolved.length ? `<div class="mp-recipe-warn">Not available in your catalog: ${escapeHtml(unresolved.join(', '))}</div>` : ''}
        <button
          class="mp-recipe-btn ${allEnabled ? 'done' : ''}"
          data-action="recipe"
          data-id="${escapeHtml(r.id)}"
          ${resolved.length === 0 || installing ? 'disabled' : ''}
        >
          ${installing ? '<span class="mp-spinner"></span> Installing…' : allEnabled ? 'All set ✓' : 'Install stack'}
        </button>
      </article>
    `;
  }).join('');

  return `
    <section class="mp-section mp-recipes-section">
      <header class="mp-section-head">
        <h3>Quick Start</h3>
        <p class="mp-section-sub">Opinionated stacks — pick one, we'll wire the rest.</p>
      </header>
      <div class="mp-recipes">${cards}</div>
    </section>
  `;
}

/* ---------- Modality rails ---------- */

function _renderRails() {
  const cats = Object.keys(CATEGORY_META).sort((a, b) => CATEGORY_META[a].order - CATEGORY_META[b].order);
  return cats.map(cat => {
    const meta = CATEGORY_META[cat];
    const list = _catalog.filter(p => p.category === cat);
    const recommendedId = _recommendedFor(cat, list);
    return `
      <section class="mp-rail" style="--accent:${meta.accent}">
        <header class="mp-rail-head">
          <div>
            <h3><span class="mp-rail-dot"></span>${escapeHtml(meta.label)}</h3>
            <p class="mp-rail-sub">Augmentum ${escapeHtml(meta.verb)}.</p>
          </div>
          <span class="mp-rail-count">${list.length} option${list.length === 1 ? '' : 's'}</span>
        </header>
        ${list.length === 0 ? _renderEmptyRail(cat) : `
          <div class="mp-rail-cards">
            ${list.map(p => _renderCard(p, p.id === recommendedId)).join('')}
          </div>
        `}
      </section>
    `;
  }).join('');
}

function _recommendedFor(cat, list) {
  if (!list.length) return null;
  const hasGpu = !!_hardware.gpu_available;
  const vram = _hardware.gpu_vram_mb || 0;
  // Picks: first entry whose GPU requirement matches current hardware.
  const fits = list.find(p => {
    if (p.gpu.required && !hasGpu) return false;
    if (p.gpu.vram_mb && vram < p.gpu.vram_mb) return false;
    return true;
  });
  return fits?.id || list[0].id;
}

function _renderEmptyRail(cat) {
  const copy = {
    image: 'No image providers in your catalog yet. Add one via "Bring your own" below, or configure cloud providers in Settings → Providers.',
  }[cat] || 'Nothing here yet. Add a custom provider below.';
  return `<div class="mp-rail-empty">${escapeHtml(copy)}</div>`;
}

function _renderCard(p, recommended) {
  const svc = _services[p.id];
  const enabled = svc?.enabled || p.enabled;
  const status = svc?.status || p.status || 'stopped';
  const busy = _busy.has(p.id);
  const gpuOk = !p.gpu.required || _hardware.gpu_available;
  const vramOk = !p.gpu.vram_mb || (_hardware.gpu_vram_mb || 0) >= p.gpu.vram_mb;
  const compat = _compatPill(p, gpuOk, vramOk);

  const brand = _brandFor(p.id);
  const features = (p.features || []).slice(0, 3).map(f =>
    `<span class="mp-feature">${escapeHtml(f.replace(/_/g, ' '))}</span>`
  ).join('');

  let btnLabel, btnClass, btnAction;
  if (busy) {
    btnLabel = '<span class="mp-spinner"></span>';
    btnClass = 'busy';
    btnAction = '';
  } else if (enabled) {
    btnLabel = 'Disable';
    btnClass = 'disable';
    btnAction = 'disable';
  } else {
    btnLabel = 'Enable';
    btnClass = 'enable';
    btnAction = 'enable';
  }

  return `
    <article class="mp-card ${enabled ? 'enabled' : ''} ${busy ? 'busy' : ''}" data-id="${escapeHtml(p.id)}">
      ${recommended ? '<span class="mp-reco" title="Best fit for your hardware">★ Recommended</span>' : ''}
      <div class="mp-card-head">
        <div class="mp-brand" style="background:${brand.tint};color:${brand.fg}">${escapeHtml(brand.initials)}</div>
        <div class="mp-card-title">
          <h4>${escapeHtml(p.name)}</h4>
          <div class="mp-card-status ${status}">
            <span class="mp-status-dot"></span>
            <span>${escapeHtml(status)}</span>
          </div>
        </div>
      </div>
      <p class="mp-card-desc">${escapeHtml(p.description)}</p>
      <div class="mp-card-meta">
        ${compat}
        ${features}
      </div>
      <button
        class="mp-card-btn ${btnClass}"
        ${btnAction ? `data-action="${btnAction}"` : ''}
        data-id="${escapeHtml(p.id)}"
        ${busy || (!enabled && !gpuOk) ? 'disabled' : ''}
        ${!gpuOk ? 'title="GPU required but not detected"' : ''}
      >${btnLabel}</button>
    </article>
  `;
}

function _compatPill(p, gpuOk, vramOk) {
  if (!p.gpu.required && !p.gpu.vram_mb) {
    return '<span class="mp-compat ok">✓ Runs anywhere</span>';
  }
  if (!gpuOk) return '<span class="mp-compat bad">Needs GPU</span>';
  if (!vramOk) {
    const need = Math.round((p.gpu.vram_mb || 0) / 1024);
    return `<span class="mp-compat warn">Needs ~${need} GB VRAM</span>`;
  }
  const vram = p.gpu.vram_mb ? Math.round(p.gpu.vram_mb / 1024) + ' GB VRAM' : 'GPU';
  return `<span class="mp-compat ok">✓ Ready · ${escapeHtml(vram)}</span>`;
}

/* ---------- Custom attachment drawer ---------- */

function _renderCustomSection() {
  return `
    <section class="mp-custom ${_customOpen ? 'open' : ''}">
      <button class="mp-custom-toggle" data-action="toggle-custom" aria-expanded="${_customOpen}">
        <div>
          <h3>Bring your own</h3>
          <p>Point Augmentum at a container you've built, or paste a <code>docker run</code> command.</p>
        </div>
        <svg class="mp-custom-caret" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      ${_customOpen ? `
        <div class="mp-custom-body">
          <div class="mp-custom-modes" role="tablist">
            <button class="${_customMode === 'quick' ? 'active' : ''}" data-action="custom-mode" data-mode="quick" role="tab">Paste command</button>
            <button class="${_customMode === 'full' ? 'active' : ''}" data-action="custom-mode" data-mode="full" role="tab">Full form</button>
          </div>
          ${_customMode === 'quick' ? _renderCustomQuick() : _renderCustomFull()}
        </div>
      ` : ''}
    </section>
  `;
}

function _renderCustomQuick() {
  return `
    <div class="mp-custom-form" data-mode="quick">
      <label>Paste your <code>docker run</code> command</label>
      <textarea id="mp-docker-run" placeholder="docker run -d -p 8080:8080 myorg/my-inference-server:latest" spellcheck="false" rows="3"></textarea>
      <div class="mp-custom-row">
        <label class="grow">Give it a name
          <input type="text" id="mp-quick-name" placeholder="My LLM Server">
        </label>
        <label>Category
          <select id="mp-quick-category">
            <option value="llm">Language</option>
            <option value="tts">Voice Out</option>
            <option value="stt">Voice In</option>
            <option value="image">Vision</option>
          </select>
        </label>
      </div>
      <div class="mp-custom-actions">
        <button class="mp-secondary-btn" data-action="custom-parse">Parse command</button>
        <button class="mp-primary-btn" data-action="custom-submit" data-mode="quick">Attach</button>
      </div>
      <p class="mp-custom-hint">We'll read the image and the first <code>-p HOST:CONTAINER</code> mapping. Adjust API type afterwards if needed.</p>
    </div>
  `;
}

function _renderCustomFull() {
  return `
    <div class="mp-custom-form" data-mode="full">
      <div class="mp-custom-row">
        <label class="grow">Name
          <input type="text" id="mp-full-name" placeholder="My Service">
        </label>
        <label>Category
          <select id="mp-full-category">
            <option value="llm">Language</option>
            <option value="tts">Voice Out</option>
            <option value="stt">Voice In</option>
            <option value="image">Vision</option>
          </select>
        </label>
      </div>
      <label>Docker image
        <input type="text" id="mp-full-image" placeholder="vllm/vllm-openai:latest">
      </label>
      <div class="mp-custom-row">
        <label class="grow">Container port
          <input type="number" id="mp-full-port" value="8080">
        </label>
        <label class="grow">Host port
          <input type="number" id="mp-full-host-port" value="6700">
        </label>
      </div>
      <div class="mp-custom-row">
        <label class="grow">API type
          <select id="mp-full-api">
            <option value="openai_llm">OpenAI-compatible LLM</option>
            <option value="ollama">Ollama</option>
            <option value="openai_tts">OpenAI-compatible TTS</option>
            <option value="openai_stt">OpenAI-compatible STT</option>
            <option value="openai_image">OpenAI-compatible Image</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label class="grow">Health endpoint
          <input type="text" id="mp-full-health" value="/health">
        </label>
      </div>
      <div class="mp-custom-actions">
        <button class="mp-primary-btn" data-action="custom-submit" data-mode="full">Attach</button>
      </div>
    </div>
  `;
}

function _toggleCustom() {
  _customOpen = !_customOpen;
  _render();
}

function _setCustomMode(mode) {
  _customMode = mode === 'full' ? 'full' : 'quick';
  _render();
}

function _parseDockerRun() {
  const raw = /** @type {HTMLTextAreaElement} */(document.getElementById('mp-docker-run'))?.value || '';
  // Pull the image (last token that isn't a flag value) and the first -p HOST:CONTAINER mapping.
  const portMatch = raw.match(/-p\s+(\d+):(\d+)/);
  const tokens = raw.trim().split(/\s+/);
  let image = '';
  for (let i = tokens.length - 1; i >= 0; i--) {
    const t = tokens[i];
    if (!t || t.startsWith('-')) continue;
    if (/^(docker|run|--\S+|-\S+)$/.test(t)) continue;
    image = t;
    break;
  }
  if (image) {
    const input = document.getElementById('mp-full-image');
    if (input) /** @type {HTMLInputElement} */(input).value = image;
  }
  if (portMatch) {
    const [, hostP, intP] = portMatch;
    const p = document.getElementById('mp-full-port');
    const hp = document.getElementById('mp-full-host-port');
    if (p)  /** @type {HTMLInputElement} */(p).value = intP;
    if (hp) /** @type {HTMLInputElement} */(hp).value = hostP;
  }
  // Ease the user into full view if we extracted anything.
  if (image || portMatch) {
    _customMode = 'full';
    _render();
  } else {
    _flash('Couldn\'t read the command — try Full form.');
  }
}

async function _submitCustom() {
  const mode = _customMode;
  let payload;
  if (mode === 'quick') {
    const raw = /** @type {HTMLTextAreaElement} */(document.getElementById('mp-docker-run'))?.value || '';
    const portMatch = raw.match(/-p\s+(\d+):(\d+)/);
    const tokens = raw.trim().split(/\s+/);
    let image = '';
    for (let i = tokens.length - 1; i >= 0; i--) {
      const t = tokens[i];
      if (!t || t.startsWith('-')) continue;
      if (/^(docker|run|--\S+|-\S+)$/.test(t)) continue;
      image = t;
      break;
    }
    const name = /** @type {HTMLInputElement} */(document.getElementById('mp-quick-name'))?.value?.trim();
    const category = /** @type {HTMLSelectElement} */(document.getElementById('mp-quick-category'))?.value || 'llm';
    if (!image || !name) { _flash('Name and a parseable image are required.'); return; }
    payload = {
      name, category, image,
      internal_port: portMatch ? parseInt(portMatch[2], 10) : 8080,
      host_port:     portMatch ? parseInt(portMatch[1], 10) : 6700,
      api_type: _defaultApiFor(category),
      health_endpoint: '/health',
    };
  } else {
    const name = /** @type {HTMLInputElement} */(document.getElementById('mp-full-name'))?.value?.trim();
    const image = /** @type {HTMLInputElement} */(document.getElementById('mp-full-image'))?.value?.trim();
    if (!name || !image) { _flash('Name and image are required.'); return; }
    payload = {
      name,
      category: /** @type {HTMLSelectElement} */(document.getElementById('mp-full-category'))?.value || 'llm',
      image,
      internal_port: parseInt(/** @type {HTMLInputElement} */(document.getElementById('mp-full-port'))?.value, 10) || 8080,
      host_port:     parseInt(/** @type {HTMLInputElement} */(document.getElementById('mp-full-host-port'))?.value, 10) || 6700,
      api_type: /** @type {HTMLSelectElement} */(document.getElementById('mp-full-api'))?.value || 'openai_llm',
      health_endpoint: /** @type {HTMLInputElement} */(document.getElementById('mp-full-health'))?.value || '/health',
    };
  }

  try {
    const res = await fetch('/api/marketplace/services/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      _flash(`Failed: ${err.error || res.statusText}`);
    } else {
      _customOpen = false;
    }
  } catch (err) {
    _flash(`Error: ${err.message}`);
  }
  await _refresh();
}

function _defaultApiFor(category) {
  return { llm: 'openai_llm', tts: 'openai_tts', stt: 'openai_stt', image: 'openai_image' }[category] || 'openai_llm';
}

/* ---------- Actions ---------- */

async function _enable(id) {
  _busy.add(id); _render();
  try {
    const res = await fetch(`/api/marketplace/services/${id}/enable`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      _flash(`Failed: ${err.error || 'Unknown error'}`);
    }
  } catch (err) {
    _flash(`Error: ${err.message}`);
  }
  _busy.delete(id);
  await _refresh();
}

async function _disable(id) {
  if (!confirm('Disable this service? The container will be removed.')) return;
  _busy.add(id); _render();
  try {
    const res = await fetch(`/api/marketplace/services/${id}/disable`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      _flash(`Failed: ${err.error || 'Unknown error'}`);
    }
  } catch (err) {
    _flash(`Error: ${err.message}`);
  }
  _busy.delete(id);
  await _refresh();
}

async function _installRecipe(recipeId) {
  const recipe = RECIPES.find(r => r.id === recipeId);
  if (!recipe) return;
  const stack = (_hardware.gpu_available ? recipe.gpu_stack : recipe.cpu_stack)
    .filter(sid => _catalog.find(c => c.id === sid));
  if (!stack.length) { _flash('No services in this recipe are available.'); return; }

  _busy.add(`recipe:${recipeId}`); _render();
  for (const sid of stack) {
    if (_services[sid]?.enabled) continue;
    _busy.add(sid); _render();
    try {
      const res = await fetch(`/api/marketplace/services/${sid}/enable`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        _flash(`${sid}: ${err.error || 'failed'}`);
      }
    } catch (err) {
      _flash(`${sid}: ${err.message}`);
    }
    _busy.delete(sid);
    await _refresh();
  }
  _busy.delete(`recipe:${recipeId}`);
  _render();
}

/* ---------- Toast ---------- */

function _flash(msg) {
  const toast = document.createElement('div');
  toast.className = 'mp-toast';
  toast.textContent = msg;
  _overlay.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('visible'));
  setTimeout(() => {
    toast.classList.remove('visible');
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}
