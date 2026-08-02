/* ==========================================================================
   Augmentum — Image Module
   Image generation panel, gallery, lightbox, model management, image library
   Ported from old monolithic app.js, adapted for modular ES module system
   ========================================================================== */

import { app, escapeHtml, extractErrorMessage, showToast } from './app.js';
import { getModels, getImageModels, getToolSettings, getCloudImageModels } from './model-cache.js';
import { ViewStack } from './view-stack.js';
import { copyToClipboard } from './clipboard.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_IMG_SETTINGS = 'augmentum_img_settings';

const IMG_RESOLUTION_PRESETS = {
  sd15: [
    { label: '512\u00d7512', w: 512, h: 512 },
    { label: '512\u00d7768', w: 512, h: 768 },
    { label: '768\u00d7512', w: 768, h: 512 },
    { label: '768\u00d7768', w: 768, h: 768 },
  ],
  sdxl: [
    { label: '1024\u00d71024', w: 1024, h: 1024 },
    { label: '832\u00d71216', w: 832, h: 1216 },
    { label: '1216\u00d7832', w: 1216, h: 832 },
    { label: '768\u00d71344', w: 768, h: 1344 },
    { label: '1344\u00d7768', w: 1344, h: 768 },
  ],
  flux: [
    { label: '1024\u00d71024', w: 1024, h: 1024 },
    { label: '832\u00d71216', w: 832, h: 1216 },
    { label: '1216\u00d7832', w: 1216, h: 832 },
    { label: '768\u00d71344', w: 768, h: 1344 },
    { label: '1344\u00d7768', w: 1344, h: 768 },
  ],
  cloud: [
    { label: '1024\u00d71024', w: 1024, h: 1024 },
    { label: '1024\u00d71792', w: 1024, h: 1792 },
    { label: '1792\u00d71024', w: 1792, h: 1024 },
    { label: '1280\u00d7720', w: 1280, h: 720 },
    { label: '720\u00d71280', w: 720, h: 1280 },
  ],
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let imageGenerating = false;
let imageAbortController = null;
let _stagePollingInterval = null;
// One-shot provenance flag — armed by generateFromArchitect() before it
// clicks the canonical generate button, consumed by the submit handler
// (body.origin = 'companion'). Companion-initiated generations land in
// the same gallery, filterable by the origin chip.
let _architectOriginPending = false;
let imageHardwareInfo = null;
let imageModelsData = [];
let currentImageMode = 'txt2img';
let currentInpaintMode = 'default';
let sourceImageBase64 = '';
// Mask editor state
const maskEditor = {
  // Canvas contexts
  bgCtx: null,       // Layer 0: source image (static)
  maskCtx: null,     // Layer 1: red overlay / B&W mask display
  uiCtx: null,       // Layer 2: cursor preview
  dataCtx: null,     // Offscreen: black/white mask for export

  // Brush
  tool: 'brush',
  brushSize: 30,
  brushOpacity: 0.5,
  brushBlur: 4,
  painting: false,
  lastPoint: null,

  // Undo/redo
  undoStack: [],
  redoStack: [],
  maxHistory: 20,

  // Zoom/pan
  zoom: 1.0,
  panX: 0,
  panY: 0,
  panning: false,
  panStart: null,
  spaceHeld: false,

  // View
  maskViewMode: 'overlay',

  // Lasso / rect selection
  lassoPoints: [],   // Array of {x, y}
  rectStart: null,   // {x, y}

  // Source
  sourceImg: null,
  naturalW: 0,
  naturalH: 0,
  baseDisplayW: 0,
  baseDisplayH: 0,
};
let currentLightboxEntry = null;

// Image settings persisted to localStorage
let imgSettings = {
  width: null, height: null, steps: null, cfg: null, seed: null,
  sampler: '', model: '', preset: '', negative: '', condenseModel: '',
  galleryCollapsed: false, advancedCollapsed: true, panelOpen: false,
  // Per-generation quality
  cfgRescale: 0.0, hiresFix: false, hiresScale: 1.5, hiresDenoise: 0.5,
  // Quality section collapse state
  qualityCollapsed: true,
};

// Image library state
const imgLibState = {
  view: 'grid',
  bulkMode: false,
  selectedIds: new Set(),
  offset: 0,
  total: 0,
  currentEntry: null,
  searchTimer: null,
  loading: false,
  entries: [],
  privateMode: false,  // true = viewing private gallery
  backgroundMode: false, // true = viewing backgrounds collection
  // Tab-aware pagination: keep each tab's loaded pages + scroll position
  // so switching tabs doesn't yank the user back to the top or force
  // a full refetch. Restore is filter-signature-gated so stale results
  // don't bleed across filter changes.
  activeTab: 'gallery',
  tabCache: { gallery: null, private: null, backgrounds: null },
  io: null, // IntersectionObserver for scroll-triggered auto-load
};

// Track active catalog downloads
const _activeCatalogDownloads = {};

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Settings Persistence
// ---------------------------------------------------------------------------

function loadImgSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_IMG_SETTINGS);
    if (raw) imgSettings = { ...imgSettings, ...JSON.parse(raw) };
  } catch { /* ignore */ }

  // If localStorage had no saved model/steps, try loading from server
  // (covers fresh browser, cleared cache, or different device)
  if (!imgSettings.model && !imgSettings.steps) {
    fetch('/api/image/active-settings')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data || !Object.keys(data).length) return;
        if (data.model && !imgSettings.model) imgSettings.model = data.model;
        if (data.steps && !imgSettings.steps) imgSettings.steps = data.steps;
        if (data.cfg_scale != null && !imgSettings.cfg) imgSettings.cfg = data.cfg_scale;
        if (data.width && !imgSettings.width) imgSettings.width = data.width;
        if (data.height && !imgSettings.height) imgSettings.height = data.height;
        if (data.seed != null && imgSettings.seed == null) imgSettings.seed = data.seed;
        if (data.sampler && !imgSettings.sampler) imgSettings.sampler = data.sampler;
        if (data.preset && !imgSettings.preset) imgSettings.preset = data.preset;
        if (data.negative_prompt && !imgSettings.negative) imgSettings.negative = data.negative_prompt;
        saveImgSettings();
        restoreImageFormFromSettings();
      })
      .catch(() => { /* server not reachable */ });
  }
}

function saveImgSettings() {
  localStorage.setItem(STORAGE_IMG_SETTINGS, JSON.stringify(imgSettings));
}

/** Push current image panel settings to the server so the image generation
 *  tool uses the same model/steps/cfg/sampler the user has configured. */
function _pushActiveSettings() {
  const body = {};
  if (imgSettings.model) body.model = imgSettings.model;
  if (imgSettings.steps != null) body.steps = imgSettings.steps;
  if (imgSettings.cfg != null) body.cfg_scale = imgSettings.cfg;
  if (imgSettings.width != null) body.width = imgSettings.width;
  if (imgSettings.height != null) body.height = imgSettings.height;
  if (imgSettings.sampler) body.sampler = imgSettings.sampler;
  if (imgSettings.preset) body.preset = imgSettings.preset;
  if (imgSettings.negative) body.negative_prompt = imgSettings.negative;
  if (imgSettings.seed != null) body.seed = imgSettings.seed;
  if (imgSettings.cfgRescale > 0) body.guidance_rescale = imgSettings.cfgRescale;
  if (imgSettings.hiresFix) {
    body.hires_fix = true;
    body.hires_scale = imgSettings.hiresScale;
    body.hires_denoise = imgSettings.hiresDenoise;
  }
  // Include cloud provider info so the image_generation tool can route correctly
  if (isSelectedModelCloud()) {
    body.cloud_provider_id = getSelectedCloudProviderId();
    const cq = $('img-cloud-quality');
    if (cq && cq.value) body.cloud_quality = cq.value;
  } else {
    body.cloud_provider_id = '';
  }
  fetch('/api/image/active-settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => { /* best-effort */ });
}

function saveImageFormToSettings() {
  const w = $('img-width'), h = $('img-height'), s = $('img-steps'), c = $('img-cfg');
  const sd = $('img-seed'), sm = $('img-sampler'), m = $('img-model'), p = $('img-preset');
  const n = $('img-negative'), cm = $('img-condense-model');
  if (w && w.value) imgSettings.width = parseInt(w.value) || imgSettings.width;
  if (h && h.value) imgSettings.height = parseInt(h.value) || imgSettings.height;
  if (s && s.value) imgSettings.steps = parseInt(s.value) || imgSettings.steps;
  if (c && c.value !== '') imgSettings.cfg = parseFloat(c.value) ?? imgSettings.cfg;
  if (sd) imgSettings.seed = parseInt(sd.value);
  // Only update dropdowns if populated AND the value is non-empty.
  // Prevents overwriting a saved model with "" when the dropdown fails to
  // restore (e.g. cloud model temporarily unreachable).
  if (m && m.options.length > 1 && m.value) imgSettings.model = m.value;
  if (sm && sm.options.length > 1 && sm.value) imgSettings.sampler = sm.value;
  if (p) imgSettings.preset = p.value;
  if (cm && cm.options.length > 1 && cm.value) imgSettings.condenseModel = cm.value;
  imgSettings.negative = n ? n.value : '';
  // Quality & speed per-gen settings
  const cfgRescaleEl = $('img-cfg-rescale');
  if (cfgRescaleEl) imgSettings.cfgRescale = parseFloat(cfgRescaleEl.value) || 0;
  const hiresFixEl = $('img-hires-fix');
  if (hiresFixEl) imgSettings.hiresFix = hiresFixEl.checked;
  const hiresScaleEl = $('img-hires-scale');
  if (hiresScaleEl) imgSettings.hiresScale = parseFloat(hiresScaleEl.value) || 1.5;
  const hiresDenoiseEl = $('img-hires-denoise');
  if (hiresDenoiseEl) imgSettings.hiresDenoise = parseFloat(hiresDenoiseEl.value) || 0.5;
  const clipSkipEl = $('img-clip-skip');
  if (clipSkipEl) imgSettings.clipSkip = clipSkipEl.value;
  saveImgSettings();
  _pushActiveSettings();
}

function restoreImageFormFromSettings() {
  const s = imgSettings;
  if (s.width != null) { const el = $('img-width'); if (el) el.value = s.width; }
  if (s.height != null) { const el = $('img-height'); if (el) el.value = s.height; }
  if (s.steps != null) { const el = $('img-steps'); if (el) el.value = s.steps; }
  if (s.cfg != null) { const el = $('img-cfg'); if (el) el.value = s.cfg; }
  if (s.seed != null) { const el = $('img-seed'); if (el) el.value = s.seed; }
  if (s.negative) { const el = $('img-negative'); if (el) el.value = s.negative; }
  if (s.preset) { const el = $('img-preset'); if (el) el.value = s.preset; }
  // Quality & speed per-gen settings
  const cfgRescaleEl = $('img-cfg-rescale');
  if (cfgRescaleEl) { cfgRescaleEl.value = s.cfgRescale || 0; const valEl = $('img-cfg-rescale-val'); if (valEl) valEl.textContent = parseFloat(cfgRescaleEl.value).toFixed(2); }
  const hiresFixEl = $('img-hires-fix');
  if (hiresFixEl) { hiresFixEl.checked = !!s.hiresFix; const opts = $('img-hires-opts'); if (opts) opts.classList.toggle('hidden', !s.hiresFix); }
  const hiresScaleEl = $('img-hires-scale');
  if (hiresScaleEl) hiresScaleEl.value = s.hiresScale || 1.5;
  const hiresDenoiseEl = $('img-hires-denoise');
  if (hiresDenoiseEl) { hiresDenoiseEl.value = s.hiresDenoise || 0.5; const valEl = $('img-hires-denoise-val'); if (valEl) valEl.textContent = parseFloat(hiresDenoiseEl.value).toFixed(2); }
  const clipSkipEl = $('img-clip-skip');
  if (clipSkipEl && s.clipSkip) clipSkipEl.value = s.clipSkip;
  // Advanced section collapse
  const advSec = $('img-advanced-section');
  const advToggle = $('img-advanced-toggle');
  if (advSec && s.advancedCollapsed === false) { advSec.classList.remove('collapsed'); if (advToggle) advToggle.classList.remove('collapsed'); }
  // Quality section collapse
  const qualSec = $('img-quality-section');
  const qualToggle = $('img-quality-toggle');
  if (qualSec && !s.qualityCollapsed) { qualSec.classList.remove('collapsed'); if (qualToggle) qualToggle.classList.remove('collapsed'); }
  // Model/sampler/condenseModel are restored after their async fetches populate the dropdowns
  // (see refreshImageModels, fetchImageSamplers, refreshCondenseModels)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

function getSelectedPipelineType() {
  const modelEl = $('img-model');
  const modelName = modelEl ? modelEl.value : '';
  if (modelName && imageModelsData.length) {
    const found = imageModelsData.find(m => m.name === modelName);
    if (found) {
      // Cloud models should use cloud resolution presets
      if (found.source === 'cloud' || (found.path && found.path.startsWith('cloud:'))) {
        return 'cloud';
      }
      return found.pipeline_type;
    }
  }
  if (imageHardwareInfo && imageHardwareInfo.recommended_pipeline) {
    return imageHardwareInfo.recommended_pipeline;
  }
  return 'sd15';
}

function isSelectedModelCloud() {
  const modelEl = $('img-model');
  if (!modelEl) return false;
  const opt = modelEl.selectedOptions[0];
  if (!opt) return false;
  if (opt.dataset.providerId) return true;
  const modelName = modelEl.value;
  const found = imageModelsData.find(m => m.name === modelName);
  return found && (found.source === 'cloud' || (found.path && found.path.startsWith('cloud:')));
}

function getSelectedCloudProviderId() {
  const modelEl = $('img-model');
  if (!modelEl) return '';
  const opt = modelEl.selectedOptions[0];
  if (opt && opt.dataset.providerId) return opt.dataset.providerId;
  const modelName = modelEl.value;
  const found = imageModelsData.find(m => m.name === modelName);
  if (found && found.path) return found.path.replace('cloud:', '');
  return '';
}

function updateCloudLocalFields() {
  const isCloud = isSelectedModelCloud();
  // Local-only fields: steps, CFG, sampler, preset, unload/rename buttons
  const localEls = ['img-steps', 'img-cfg', 'img-sampler', 'img-preset', 'img-unload-btn', 'img-rename-btn'];
  for (const id of localEls) {
    const el = $(id);
    if (el) {
      const parent = el.closest('.img-form-row') || el.closest('div');
      if (parent && (id === 'img-steps' || id === 'img-cfg')) {
        // Steps/CFG share a row with seed — hide just their wrappers
        el.parentElement.style.display = isCloud ? 'none' : '';
      } else if (id === 'img-unload-btn' || id === 'img-rename-btn') {
        el.style.display = isCloud ? 'none' : '';
      } else if (el.tagName === 'SELECT') {
        const wrapper = el.closest('div[style*="flex"]') || el.parentElement;
        if (wrapper) wrapper.style.display = isCloud ? 'none' : '';
      }
    }
  }

  // Cloud quality dropdown
  let qualityEl = $('img-cloud-quality');
  if (isCloud && !qualityEl) {
    // Insert quality dropdown next to seed
    const seedEl = $('img-seed');
    if (seedEl && seedEl.parentElement) {
      const wrapper = document.createElement('div');
      wrapper.style.flex = '1';
      wrapper.innerHTML = '<label class="field-label">Quality</label><select class="field-input" id="img-cloud-quality"><option value="standard">Standard</option><option value="hd">HD</option><option value="high">High</option><option value="low">Low</option></select>';
      seedEl.parentElement.parentElement.appendChild(wrapper);
    }
  } else if (!isCloud && qualityEl) {
    qualityEl.parentElement.remove();
  }
}

// ---------------------------------------------------------------------------
// Model-Recommended Defaults
// ---------------------------------------------------------------------------

function applyModelDefaults() {
  const modelEl = $('img-model');
  if (!modelEl) return;
  const modelName = modelEl.value;
  const found = imageModelsData.find(m => m.name === modelName);
  if (!found) return;

  const stepsEl = $('img-steps');
  const cfgEl = $('img-cfg');
  if (found.recommended_steps != null && stepsEl) {
    stepsEl.value = found.recommended_steps;
    // Update range display if it has one
    const stepsDisplay = stepsEl.parentElement?.querySelector('.img-range-value');
    if (stepsDisplay) stepsDisplay.textContent = found.recommended_steps;
  }
  if (found.recommended_cfg != null && cfgEl) {
    cfgEl.value = found.recommended_cfg;
    const cfgDisplay = cfgEl.parentElement?.querySelector('.img-range-value');
    if (cfgDisplay) cfgDisplay.textContent = found.recommended_cfg;
  }
}

// ---------------------------------------------------------------------------
// Resolution Presets
// ---------------------------------------------------------------------------

function renderResolutionPresets() {
  const container = $('img-resolution-presets');
  if (!container) return;
  const ptype = getSelectedPipelineType();
  const presets = IMG_RESOLUTION_PRESETS[ptype] || IMG_RESOLUTION_PRESETS.sd15;
  const widthEl = $('img-width');
  const heightEl = $('img-height');
  const curW = widthEl ? parseInt(widthEl.value) || 0 : 0;
  const curH = heightEl ? parseInt(heightEl.value) || 0 : 0;

  container.innerHTML = '';
  presets.forEach(p => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'img-resolution-preset';
    btn.textContent = p.label;
    if (p.w === curW && p.h === curH) btn.classList.add('active');
    btn.addEventListener('click', () => {
      if (widthEl) widthEl.value = p.w;
      if (heightEl) heightEl.value = p.h;
      container.querySelectorAll('.img-resolution-preset').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      saveImageFormToSettings();
    });
    container.appendChild(btn);
  });
}

// ---------------------------------------------------------------------------
// Hardware Info
// ---------------------------------------------------------------------------

async function fetchImageHardware() {
  const container = $('img-hardware-info');
  if (!container) return;

  try {
    const resp = await fetch('/api/image/hardware');
    if (resp.status === 503 || !resp.ok) return;

    const hw = await resp.json();
    imageHardwareInfo = hw;

    const deviceEl = $('img-hw-device');
    if (deviceEl) deviceEl.textContent = hw.device_name || hw.device || 'Unknown';

    const tierEl = $('img-hw-tier');
    if (tierEl) {
      const tier = (hw.tier || 'cpu').toLowerCase();
      tierEl.textContent = tier.toUpperCase();
      tierEl.className = 'img-hw-tier-badge tier-' + tier;
    }

    const vramRow = $('img-hw-vram-row');
    const vramFill = $('img-hw-vram-fill');
    const vramText = $('img-hw-vram-text');
    if (hw.vram_total_mb > 0) {
      const usedMb = hw.vram_total_mb - hw.vram_free_mb;
      const pct = Math.min(100, Math.round((usedMb / hw.vram_total_mb) * 100));
      if (vramFill) vramFill.style.width = pct + '%';
      if (vramText) vramText.textContent =
        (usedMb / 1024).toFixed(1) + ' / ' + (hw.vram_total_mb / 1024).toFixed(1) + ' GB';
      if (vramRow) vramRow.style.display = '';
    } else {
      if (vramRow) vramRow.style.display = 'none';
    }

    const recEl = $('img-hw-rec');
    if (recEl && hw.recommended_pipeline) {
      recEl.textContent = 'Recommended: ' + hw.recommended_pipeline.toUpperCase();
    }

    container.classList.remove('hidden');
    renderResolutionPresets();
  } catch { /* silently ignore */ }
}

// ---------------------------------------------------------------------------
// Model List
// ---------------------------------------------------------------------------

async function refreshImageModels() {
  const select = $('img-model');
  if (!select) return;

  try {
    let models = [];

    // Try the main models endpoint (returns local + cloud)
    const resp = await fetch('/api/image/models');
    if (resp.ok) {
      models = await resp.json();
    }

    // If no cloud models came back, try the dedicated cloud endpoint
    const hasCloud = models.some(m => m.source === 'cloud');
    if (!hasCloud) {
      try {
        const cloudResp = await fetch('/api/image/cloud/models');
        if (cloudResp.ok) {
          const cloudModels = await cloudResp.json();
          for (const cm of cloudModels) {
            models.push({
              name: cm.name,
              pipeline_type: cm.pipeline_type || 'cloud',
              source: 'cloud',
              path: cm.provider_id ? `cloud:${cm.provider_id}` : '',
              is_loaded: false,
              capabilities: cm.capabilities || { txt2img: 'yes', img2img: 'yes', inpaint: 'fallback' },
            });
          }
        }
      } catch { /* cloud endpoint not available */ }
    }

    imageModelsData = models;
    _populateModelDropdown(select, models);
  } catch { /* image models endpoint not available */ }
}

// Live-refresh the image-model dropdown when a model finishes pulling/
// importing or is uploaded/deleted (image_routes.py emits image.models.changed
// over the SSE bus). Only re-fetch while the image panel is open — the
// installed PWA has no manual refresh. refreshImageModels() hits the endpoint
// directly (not the model-cache), so this listener drives it explicitly.
window.addEventListener('system-event:image.models.changed', () => {
  const panel = document.getElementById('image-panel');
  if (panel && !panel.classList.contains('hidden')) refreshImageModels();
});

/**
 * Populate the model dropdown, filtering and sorting by current image mode.
 * Models that natively support the current mode appear first with a green dot;
 * fallback models show a yellow dot; unsupported models are dimmed.
 */
function _populateModelDropdown(select, models) {
  const mode = currentImageMode; // 'txt2img' | 'img2img' | 'inpaint'
  const capKey = mode === 'txt2img' ? 'txt2img' : mode === 'img2img' ? 'img2img' : 'inpaint';

  // Sort: native > fallback > no; loaded first within each tier
  const sorted = [...models].sort((a, b) => {
    const capA = (a.capabilities || {})[capKey] || 'fallback';
    const capB = (b.capabilities || {})[capKey] || 'fallback';
    const rankMap = { yes: 0, fallback: 1, no: 2 };
    const rankDiff = (rankMap[capA] ?? 1) - (rankMap[capB] ?? 1);
    if (rankDiff !== 0) return rankDiff;
    // Loaded model first within same rank
    if (a.is_loaded && !b.is_loaded) return -1;
    if (!a.is_loaded && b.is_loaded) return 1;
    return 0;
  });

  select.innerHTML = '<option value="">Default</option>';
  for (const m of sorted) {
    const opt = document.createElement('option');
    opt.value = m.name;
    const isCloud = m.source === 'cloud' || (m.path && m.path.startsWith('cloud:'));
    const isPeer = m.source === 'peer' || (m.path && m.path.startsWith('peer:'));
    const caps = m.capabilities || {};
    const support = caps[capKey] || 'fallback';

    // Capability dot: ● native, ◐ fallback, ○ unsupported
    const dot = support === 'yes' ? '\u25CF' : support === 'fallback' ? '\u25D0' : '\u25CB';
    // Source icon: \u2601 cloud, peer-chosen emoji (\u{1F517} fallback) for
    // peer, \u{1F5A5} local. Phase 8 adds the peer branch alongside the
    // existing cloud / local distinction.
    let sourceIcon;
    if (isCloud) sourceIcon = '\u2601';
    else if (isPeer) sourceIcon = m.peer_icon || '\u{1F517}';
    else sourceIcon = '\u{1F5A5}';
    const typeLabel = isCloud
      ? 'cloud'
      : (isPeer ? (m.peer_hostname || 'peer') : m.pipeline_type);
    opt.textContent = dot + ' ' + sourceIcon + ' ' + m.name + ' (' + typeLabel + ')';
    if (m.is_loaded) opt.textContent += ' \u2605'; // ★ loaded

    if (isCloud) opt.dataset.providerId = (m.path || '').replace('cloud:', '');
    if (isPeer) opt.dataset.peerNodeId = (m.path || '').replace('peer:', '');

    // Dim unsupported models
    if (support === 'no') opt.style.opacity = '0.45';

    select.appendChild(opt);
  }
  if (imgSettings.model) select.value = imgSettings.model;
  renderResolutionPresets();
  updateCloudLocalFields();
}

async function fetchImageSamplers() {
  try {
    const resp = await fetch('/api/image/samplers');
    if (!resp.ok) return;
    const samplers = await resp.json();
    const select = $('img-sampler');
    if (!select) return;

    // Group by category
    const categories = {
      recommended: { label: 'Recommended', items: [] },
      quality: { label: 'Quality', items: [] },
      fast: { label: 'Fast', items: [] },
      specialized: { label: 'Specialized', items: [] },
    };
    for (const s of samplers) {
      const cat = categories[s.category] || categories.specialized;
      cat.items.push(s);
    }

    select.innerHTML = '<option value="">Default (model decides)</option>';
    for (const [, cat] of Object.entries(categories)) {
      if (!cat.items.length) continue;
      const group = document.createElement('optgroup');
      group.label = cat.label;
      for (const s of cat.items) {
        const opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = s.display_name;
        if (s.description) opt.title = s.description;
        group.appendChild(opt);
      }
      select.appendChild(group);
    }
    if (imgSettings.sampler) select.value = imgSettings.sampler;
  } catch { /* samplers endpoint not available */ }
}

async function refreshCondenseModels() {
  const select = $('img-condense-model');
  if (!select) return;
  const saved = imgSettings.condenseModel || select.value;
  try {
    const resp = await fetch('/api/tags');
    if (!resp.ok) return;
    const data = await resp.json();
    const models = (data.models || []).filter(m =>
      !m.name.startsWith('a/') && !m.name.startsWith('n/') && !m.name.startsWith('p/')
    );
    select.innerHTML = '<option value="">Default (backend default)</option>';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      select.appendChild(opt);
    });
    if (saved) select.value = saved;
  } catch { /* backend not reachable */ }
}

// ---------------------------------------------------------------------------
// Image Mode Switching (txt2img / img2img / inpaint)
// ---------------------------------------------------------------------------

function setImageMode(mode) {
  currentImageMode = mode;
  document.querySelectorAll('.img-mode-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });
  const srcSection = $('img-source-section');
  const maskSection = $('img-mask-section');
  if (srcSection) srcSection.classList.toggle('hidden', mode === 'txt2img');
  if (maskSection) maskSection.classList.toggle('hidden', mode !== 'inpaint');

  const genBtn = $('img-generate-btn');
  if (genBtn) {
    if (mode === 'txt2img') genBtn.textContent = 'Generate';
    else if (mode === 'img2img') genBtn.textContent = 'Transform';
    else genBtn.textContent = 'Inpaint';
  }

  // Show/hide inpaint-specific rows
  const fullresRow = $('img-inpaint-fullres-row');
  if (fullresRow) fullresRow.classList.toggle('hidden', mode !== 'inpaint');
  const inpaintModeRow = $('img-inpaint-mode-row');
  if (inpaintModeRow) inpaintModeRow.classList.toggle('hidden', mode !== 'inpaint');

  // Reset inpaint mode chips when switching modes
  if (mode === 'inpaint') {
    currentInpaintMode = 'default';
    document.querySelectorAll('.img-inpaint-mode').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === 'default');
    });
  }

  const strengthEl = $('img-strength');
  const strengthValEl = $('img-strength-value');
  if (strengthEl) {
    strengthEl.value = mode === 'inpaint' ? '1.0' : '0.75';
    if (strengthValEl) strengthValEl.textContent = strengthEl.value;
  }

  // Re-sort model dropdown for the new mode's capabilities
  const select = $('img-model');
  if (select && imageModelsData.length) {
    _populateModelDropdown(select, imageModelsData);
  }

  // Re-sort catalog for the new mode's capabilities
  const catalogContainer = $('img-catalog');
  if (catalogContainer && _catalogData.length) {
    _renderCatalog(catalogContainer, _catalogData);
  }
}

// ---------------------------------------------------------------------------
// Source Image Upload
// ---------------------------------------------------------------------------

function loadSourceImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    sourceImageBase64 = dataUrl.split(',')[1];
    const preview = $('img-source-preview');
    const placeholder = $('img-source-placeholder');
    const clearBtn = $('img-source-clear');
    if (preview) { preview.src = dataUrl; preview.classList.remove('hidden'); }
    if (placeholder) placeholder.classList.add('hidden');
    if (clearBtn) clearBtn.classList.remove('hidden');
    if (currentImageMode === 'inpaint') setupMaskCanvas(dataUrl);
  };
  reader.readAsDataURL(file);
}

function loadSourceImageFromUrl(url) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    const dataUrl = canvas.toDataURL('image/png');
    sourceImageBase64 = dataUrl.split(',')[1];
    const preview = $('img-source-preview');
    const placeholder = $('img-source-placeholder');
    const clearBtn = $('img-source-clear');
    if (preview) { preview.src = dataUrl; preview.classList.remove('hidden'); }
    if (placeholder) placeholder.classList.add('hidden');
    if (clearBtn) clearBtn.classList.remove('hidden');
    if (currentImageMode === 'inpaint') setupMaskCanvas(dataUrl);
  };
  img.src = url;
}

// ---------------------------------------------------------------------------
// Inpaint Mask Editor
// ---------------------------------------------------------------------------

function setupMaskCanvas(imageDataUrl) {
  const wrapEl = $('img-mask-canvas-wrap');
  if (!wrapEl) return;

  const img = new Image();
  img.onload = () => {
    const me = maskEditor;
    me.sourceImg = img;
    me.naturalW = img.naturalWidth;
    me.naturalH = img.naturalHeight;

    // Responsive: fill container width
    const displayW = wrapEl.clientWidth || 300;
    const scale = displayW / me.naturalW;
    const displayH = Math.round(me.naturalH * scale);
    me.baseDisplayW = displayW;
    me.baseDisplayH = displayH;

    // Set container aspect ratio
    wrapEl.style.height = displayH + 'px';

    // Setup 3 DOM canvases
    const bgCanvas = $('img-mask-bg-canvas');
    const maskLayer = $('img-mask-layer');
    const uiCanvas = $('img-mask-ui-canvas');

    [bgCanvas, maskLayer, uiCanvas].forEach(c => {
      if (!c) return;
      c.width = me.naturalW;
      c.height = me.naturalH;
      c.style.width = displayW + 'px';
      c.style.height = displayH + 'px';
    });

    // Draw source image on background layer
    me.bgCtx = bgCanvas.getContext('2d');
    me.bgCtx.drawImage(img, 0, 0);

    // Mask display layer (red overlay or B&W)
    me.maskCtx = maskLayer.getContext('2d');

    // UI layer (cursor preview)
    me.uiCtx = uiCanvas.getContext('2d');

    // Offscreen data canvas (black/white mask for export)
    const offscreen = document.createElement('canvas');
    offscreen.width = me.naturalW;
    offscreen.height = me.naturalH;
    me.dataCtx = offscreen.getContext('2d');
    me.dataCtx.fillStyle = 'black';
    me.dataCtx.fillRect(0, 0, me.naturalW, me.naturalH);

    // Reset state
    me.undoStack = [];
    me.redoStack = [];
    me.zoom = 1.0;
    me.panX = 0;
    me.panY = 0;
    me.lastPoint = null;
    me.painting = false;
    me.maskViewMode = 'overlay';

    const wrap = $('img-mask-canvas-wrap');
    if (wrap) wrap.classList.remove('mask-bw-view');

    // Push initial blank state for undo
    maskEditorPushUndo();
    maskEditorRender();
    maskEditorUpdateZoom();
  };
  img.src = imageDataUrl;
}

// --- Render the mask overlay on the display canvas ---
let _maskTintCanvas = null; // Cached temp canvas for overlay compositing
function maskEditorRender() {
  const me = maskEditor;
  if (!me.maskCtx || !me.dataCtx) return;

  const w = me.naturalW;
  const h = me.naturalH;
  me.maskCtx.clearRect(0, 0, w, h);

  if (me.maskViewMode === 'overlay') {
    // Reuse temp canvas for red tint compositing (avoid per-frame allocation)
    if (!_maskTintCanvas || _maskTintCanvas.width !== w || _maskTintCanvas.height !== h) {
      _maskTintCanvas = document.createElement('canvas');
      _maskTintCanvas.width = w;
      _maskTintCanvas.height = h;
    }
    const tempCtx = _maskTintCanvas.getContext('2d');
    tempCtx.clearRect(0, 0, w, h);

    // Draw mask data
    tempCtx.drawImage(me.dataCtx.canvas, 0, 0);

    // Use compositing: only keep red where mask is white
    tempCtx.globalCompositeOperation = 'source-in';
    tempCtx.fillStyle = 'rgba(255, 60, 60, 1)';
    tempCtx.fillRect(0, 0, w, h);
    tempCtx.globalCompositeOperation = 'source-over';

    // Draw the tinted mask onto the display with opacity
    me.maskCtx.globalAlpha = me.brushOpacity;
    me.maskCtx.drawImage(_maskTintCanvas, 0, 0);
    me.maskCtx.globalAlpha = 1.0;
  } else {
    // B&W mask view: draw data canvas directly
    me.maskCtx.drawImage(me.dataCtx.canvas, 0, 0);
  }
}

// --- Paint on the mask ---
function maskEditorPaint(e) {
  const me = maskEditor;
  if (!me.dataCtx) return;

  const pos = maskEditorPointerToCanvas(e);
  if (!pos) return;

  const { x, y } = pos;
  const radius = me.brushSize * (me.naturalW / me.baseDisplayW);

  me.dataCtx.lineCap = 'round';
  me.dataCtx.lineJoin = 'round';
  me.dataCtx.lineWidth = radius * 2;
  me.dataCtx.strokeStyle = me.tool === 'brush' ? 'white' : 'black';
  me.dataCtx.fillStyle = me.tool === 'brush' ? 'white' : 'black';

  if (me.lastPoint) {
    // Stroke interpolation: connect last point to current with a line
    me.dataCtx.beginPath();
    me.dataCtx.moveTo(me.lastPoint.x, me.lastPoint.y);
    me.dataCtx.lineTo(x, y);
    me.dataCtx.stroke();
  } else {
    // Single dot for click without drag
    me.dataCtx.beginPath();
    me.dataCtx.arc(x, y, radius, 0, Math.PI * 2);
    me.dataCtx.fill();
  }

  me.lastPoint = { x, y };
  maskEditorRender();
}

// --- Brush cursor preview ---
function maskEditorDrawCursor(e) {
  const me = maskEditor;
  if (!me.uiCtx) return;

  me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);

  const pos = maskEditorPointerToCanvas(e);
  if (!pos) return;

  const radius = me.brushSize * (me.naturalW / me.baseDisplayW);

  // Outer ring (white)
  me.uiCtx.beginPath();
  me.uiCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
  me.uiCtx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  me.uiCtx.lineWidth = 2 * (me.naturalW / me.baseDisplayW);
  me.uiCtx.stroke();

  // Inner ring (dark, for contrast)
  me.uiCtx.beginPath();
  me.uiCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
  me.uiCtx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
  me.uiCtx.lineWidth = 1 * (me.naturalW / me.baseDisplayW);
  me.uiCtx.stroke();

  // Crosshair center
  const ch = 4 * (me.naturalW / me.baseDisplayW);
  me.uiCtx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  me.uiCtx.lineWidth = 1 * (me.naturalW / me.baseDisplayW);
  me.uiCtx.beginPath();
  me.uiCtx.moveTo(pos.x - ch, pos.y);
  me.uiCtx.lineTo(pos.x + ch, pos.y);
  me.uiCtx.moveTo(pos.x, pos.y - ch);
  me.uiCtx.lineTo(pos.x, pos.y + ch);
  me.uiCtx.stroke();
}

// --- Convert pointer event to canvas-space coordinates ---
function maskEditorPointerToCanvas(e) {
  const me = maskEditor;
  // Use the container rect (unaffected by CSS transforms on children)
  const wrapEl = $('img-mask-canvas-wrap');
  if (!wrapEl || !me.baseDisplayW) return null;

  const rect = wrapEl.getBoundingClientRect();
  const displayX = e.clientX - rect.left;
  const displayY = e.clientY - rect.top;

  // Convert from container space → canvas natural space, accounting for zoom/pan
  const scaleX = me.naturalW / me.baseDisplayW;
  const scaleY = me.naturalH / me.baseDisplayH;
  const x = ((displayX - me.panX) / me.zoom) * scaleX;
  const y = ((displayY - me.panY) / me.zoom) * scaleY;

  return { x, y };
}

// --- Pointer event handlers ---
function maskEditorPointerDown(e) {
  e.preventDefault();
  const me = maskEditor;

  if (me.spaceHeld) {
    // Start panning
    me.panning = true;
    me.panStart = { x: e.clientX, y: e.clientY, panX: me.panX, panY: me.panY };
    return;
  }

  if (me.tool === 'lasso') { lassoPointerDown(e); return; }
  if (me.tool === 'rect') { rectPointerDown(e); return; }

  me.painting = true;
  me.lastPoint = null;
  // Visual feedback: canvas glow while painting
  const wrap = $('img-mask-canvas-wrap');
  if (wrap) wrap.classList.add('is-painting');
  maskEditorPaint(e);
}

function maskEditorPointerMove(e) {
  const me = maskEditor;
  if (me.tool === 'lasso') { lassoPointerMove(e); return; }
  if (me.tool === 'rect') { rectPointerMove(e); return; }
  maskEditorDrawCursor(e);

  if (me.panning && me.panStart) {
    me.panX = me.panStart.panX + (e.clientX - me.panStart.x);
    me.panY = me.panStart.panY + (e.clientY - me.panStart.y);
    maskEditorApplyTransform();
    return;
  }

  if (me.painting) {
    maskEditorPaint(e);
  }
}

function maskEditorPointerUp(e) {
  const me = maskEditor;
  if (me.tool === 'lasso') { lassoPointerUp(); return; }
  if (me.tool === 'rect') { rectPointerUp(e); return; }
  if (me.painting) {
    me.painting = false;
    me.lastPoint = null;
    maskEditorPushUndo();
    const wrap = $('img-mask-canvas-wrap');
    if (wrap) wrap.classList.remove('is-painting');
  }
  if (me.panning) {
    me.panning = false;
    me.panStart = null;
  }
}

function maskEditorPointerLeave() {
  const me = maskEditor;
  // Clear cursor
  if (me.uiCtx) me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
  if (me.painting) {
    me.painting = false;
    me.lastPoint = null;
    maskEditorPushUndo();
    const wrap = $('img-mask-canvas-wrap');
    if (wrap) wrap.classList.remove('is-painting');
  }
  me.panning = false;
  me.panStart = null;
}

// --- Zoom ---
function maskEditorWheel(e) {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();

  const me = maskEditor;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newZoom = Math.max(0.5, Math.min(5.0, me.zoom * delta));

  // Zoom toward cursor
  const rect = e.target.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  me.panX = mx - (mx - me.panX) * (newZoom / me.zoom);
  me.panY = my - (my - me.panY) * (newZoom / me.zoom);
  me.zoom = newZoom;

  maskEditorApplyTransform();
  maskEditorUpdateZoom();
}

function maskEditorApplyTransform() {
  const me = maskEditor;
  const canvases = [
    $('img-mask-bg-canvas'),
    $('img-mask-layer'),
    $('img-mask-ui-canvas'),
  ];
  const transform = `translate(${me.panX}px, ${me.panY}px) scale(${me.zoom})`;
  canvases.forEach(c => {
    if (c) {
      c.style.transformOrigin = '0 0';
      c.style.transform = transform;
    }
  });
}

function maskEditorFitToView() {
  const me = maskEditor;
  me.zoom = 1.0;
  me.panX = 0;
  me.panY = 0;
  maskEditorApplyTransform();
  maskEditorUpdateZoom();
}

function maskEditorUpdateZoom() {
  const el = $('img-mask-zoom-level');
  if (el) el.textContent = Math.round(maskEditor.zoom * 100) + '%';
}

// --- Undo / Redo ---
function maskEditorPushUndo() {
  const me = maskEditor;
  if (!me.dataCtx) return;
  const data = me.dataCtx.getImageData(0, 0, me.naturalW, me.naturalH);
  me.undoStack.push(data);
  if (me.undoStack.length > me.maxHistory) me.undoStack.shift();
  me.redoStack = [];
}

function maskEditorUndo() {
  const me = maskEditor;
  if (me.undoStack.length <= 1) return; // keep at least initial state
  // Save current to redo
  const current = me.dataCtx.getImageData(0, 0, me.naturalW, me.naturalH);
  me.redoStack.push(current);
  // Pop and discard current, restore previous
  me.undoStack.pop();
  const prev = me.undoStack[me.undoStack.length - 1];
  me.dataCtx.putImageData(prev, 0, 0);
  maskEditorRender();
}

function maskEditorRedo() {
  const me = maskEditor;
  if (!me.redoStack.length) return;
  const next = me.redoStack.pop();
  me.undoStack.push(next);
  me.dataCtx.putImageData(next, 0, 0);
  maskEditorRender();
}

// --- Tool helpers ---
function setMaskTool(tool) {
  maskEditor.tool = tool;
  document.querySelectorAll('.img-mask-tool').forEach(b =>
    b.classList.toggle('active', b.dataset.tool === tool)
  );
}

function adjustBrushSize(delta) {
  const me = maskEditor;
  me.brushSize = Math.max(5, Math.min(100, me.brushSize + delta));
  const el = $('img-mask-brush-size');
  const valEl = $('img-mask-brush-value');
  if (el) el.value = me.brushSize;
  if (valEl) valEl.textContent = me.brushSize;
}

function toggleMaskView() {
  const me = maskEditor;
  me.maskViewMode = me.maskViewMode === 'overlay' ? 'mask' : 'overlay';
  const wrap = $('img-mask-canvas-wrap');
  if (wrap) wrap.classList.toggle('mask-bw-view', me.maskViewMode === 'mask');
  const btn = $('img-mask-view-toggle');
  if (btn) btn.classList.toggle('active', me.maskViewMode === 'mask');
  maskEditorRender();
}

function clearMask() {
  const me = maskEditor;
  if (!me.dataCtx) return;
  me.dataCtx.fillStyle = 'black';
  me.dataCtx.fillRect(0, 0, me.naturalW, me.naturalH);
  me.undoStack = [];
  me.redoStack = [];
  maskEditorPushUndo();
  maskEditorRender();
}

function invertMask() {
  const me = maskEditor;
  if (!me.dataCtx) return;
  maskEditorPushUndo();
  const imageData = me.dataCtx.getImageData(0, 0, me.naturalW, me.naturalH);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = 255 - d[i];
    d[i + 1] = 255 - d[i + 1];
    d[i + 2] = 255 - d[i + 2];
  }
  me.dataCtx.putImageData(imageData, 0, 0);
  maskEditorRender();
}

function getMaskBase64() {
  if (!maskEditor.dataCtx) return '';
  return maskEditor.dataCtx.canvas.toDataURL('image/png').split(',')[1];
}

// --- Lasso tool ---
function lassoPointerDown(e) {
  const pt = maskEditorPointerToCanvas(e);
  maskEditor.lassoPoints = [pt];
  const me = maskEditor;
  if (me.uiCtx) {
    me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
    me.uiCtx.setLineDash([4, 4]);
    me.uiCtx.strokeStyle = 'rgba(255,255,255,0.9)';
    me.uiCtx.lineWidth = 1.5;
  }
}

function lassoPointerMove(e) {
  const me = maskEditor;
  if (!me.lassoPoints.length) return;
  const pt = maskEditorPointerToCanvas(e);
  me.lassoPoints.push(pt);
  if (!me.uiCtx) return;
  me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
  me.uiCtx.beginPath();
  me.uiCtx.setLineDash([4, 4]);
  me.uiCtx.strokeStyle = 'rgba(255,255,255,0.9)';
  me.uiCtx.lineWidth = 1.5;
  me.lassoPoints.forEach((p, i) => {
    if (i === 0) me.uiCtx.moveTo(p.x, p.y);
    else me.uiCtx.lineTo(p.x, p.y);
  });
  me.uiCtx.stroke();
}

function lassoPointerUp() {
  const me = maskEditor;
  if (me.lassoPoints.length < 3) {
    me.lassoPoints = [];
    if (me.uiCtx) me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
    return;
  }
  maskEditorPushUndo();
  me.dataCtx.save();
  me.dataCtx.beginPath();
  me.lassoPoints.forEach((p, i) => {
    if (i === 0) me.dataCtx.moveTo(p.x, p.y);
    else me.dataCtx.lineTo(p.x, p.y);
  });
  me.dataCtx.closePath();
  me.dataCtx.fillStyle = 'white';
  me.dataCtx.fill();
  me.dataCtx.restore();
  me.lassoPoints = [];
  if (me.uiCtx) me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
  maskEditorRender();
}

// --- Rectangle tool ---
function rectPointerDown(e) {
  maskEditor.rectStart = maskEditorPointerToCanvas(e);
}

function rectPointerMove(e) {
  const me = maskEditor;
  if (!me.rectStart) return;
  const cur = maskEditorPointerToCanvas(e);
  if (!me.uiCtx) return;
  me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
  me.uiCtx.beginPath();
  me.uiCtx.setLineDash([4, 4]);
  me.uiCtx.strokeStyle = 'rgba(255,255,255,0.9)';
  me.uiCtx.lineWidth = 1.5;
  me.uiCtx.strokeRect(
    me.rectStart.x, me.rectStart.y,
    cur.x - me.rectStart.x, cur.y - me.rectStart.y
  );
}

function rectPointerUp(e) {
  const me = maskEditor;
  if (!me.rectStart) return;
  const cur = maskEditorPointerToCanvas(e);
  maskEditorPushUndo();
  me.dataCtx.save();
  me.dataCtx.fillStyle = 'white';
  me.dataCtx.fillRect(
    Math.min(me.rectStart.x, cur.x),
    Math.min(me.rectStart.y, cur.y),
    Math.abs(cur.x - me.rectStart.x),
    Math.abs(cur.y - me.rectStart.y)
  );
  me.dataCtx.restore();
  me.rectStart = null;
  if (me.uiCtx) me.uiCtx.clearRect(0, 0, me.naturalW, me.naturalH);
  maskEditorRender();
}

// --- AI Mask Actions ---

function aiExpandMask(pixels) {
  const me = maskEditor;
  if (!me.dataCtx) return;
  maskEditorPushUndo();
  const W = me.naturalW, H = me.naturalH;
  const src = me.dataCtx.getImageData(0, 0, W, H);
  const dst = me.dataCtx.createImageData(W, H);
  const s = src.data, d = dst.data;
  const r2 = pixels * pixels;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      if (s[idx] > 128) {
        d[idx] = d[idx + 1] = d[idx + 2] = 255;
        d[idx + 3] = 255;
      } else {
        let found = false;
        const x0 = Math.max(0, x - pixels), x1 = Math.min(W - 1, x + pixels);
        const y0 = Math.max(0, y - pixels), y1 = Math.min(H - 1, y + pixels);
        outer: for (let ny = y0; ny <= y1; ny++) {
          for (let nx = x0; nx <= x1; nx++) {
            const dx = nx - x, dy = ny - y;
            if (dx * dx + dy * dy <= r2 && s[(ny * W + nx) * 4] > 128) {
              found = true; break outer;
            }
          }
        }
        if (found) {
          d[idx] = d[idx + 1] = d[idx + 2] = 255;
          d[idx + 3] = 255;
        } else {
          d[idx] = d[idx + 1] = d[idx + 2] = 0;
          d[idx + 3] = 255;
        }
      }
    }
  }
  me.dataCtx.putImageData(dst, 0, 0);
  maskEditorRender();
  showToast('Mask expanded', 'success');
}

function aiContractMask(pixels) {
  const me = maskEditor;
  if (!me.dataCtx) return;
  maskEditorPushUndo();
  const W = me.naturalW, H = me.naturalH;
  const src = me.dataCtx.getImageData(0, 0, W, H);
  const dst = me.dataCtx.createImageData(W, H);
  const s = src.data, d = dst.data;
  const r2 = pixels * pixels;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      if (s[idx] <= 128) {
        d[idx] = d[idx + 1] = d[idx + 2] = 0;
        d[idx + 3] = 255;
      } else {
        let edged = false;
        const x0 = Math.max(0, x - pixels), x1 = Math.min(W - 1, x + pixels);
        const y0 = Math.max(0, y - pixels), y1 = Math.min(H - 1, y + pixels);
        outer: for (let ny = y0; ny <= y1; ny++) {
          for (let nx = x0; nx <= x1; nx++) {
            const dx = nx - x, dy = ny - y;
            if (dx * dx + dy * dy <= r2 && s[(ny * W + nx) * 4] <= 128) {
              edged = true; break outer;
            }
          }
        }
        if (edged) {
          d[idx] = d[idx + 1] = d[idx + 2] = 0;
          d[idx + 3] = 255;
        } else {
          d[idx] = d[idx + 1] = d[idx + 2] = 255;
          d[idx + 3] = 255;
        }
      }
    }
  }
  me.dataCtx.putImageData(dst, 0, 0);
  maskEditorRender();
  showToast('Mask contracted', 'success');
}

function aiEdgeDetect() {
  const me = maskEditor;
  if (!me.dataCtx || !me.sourceImg) return;
  maskEditorPushUndo();
  const W = me.naturalW, H = me.naturalH;

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = W; tmpCanvas.height = H;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.drawImage(me.sourceImg, 0, 0, W, H);
  const srcData = tmpCtx.getImageData(0, 0, W, H).data;

  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = srcData[i * 4], g = srcData[i * 4 + 1], b = srcData[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  const edges = new Float32Array(W * H);
  const Kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const Ky = [-1, -2, -1,  0,  0,  0,  1,  2,  1];
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const v = gray[(y + ky) * W + (x + kx)];
          const ki = (ky + 1) * 3 + (kx + 1);
          gx += Kx[ki] * v;
          gy += Ky[ki] * v;
        }
      }
      edges[y * W + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }

  const maskData = me.dataCtx.getImageData(0, 0, W, H);
  const d = maskData.data;
  const threshold = 50;
  for (let i = 0; i < W * H; i++) {
    if (edges[i] > threshold) {
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 255;
      d[i * 4 + 3] = 255;
    }
  }
  me.dataCtx.putImageData(maskData, 0, 0);
  maskEditorRender();
  showToast('Edge detection applied', 'success');
}

// --- Clear source image ---
function clearSourceImage() {
  sourceImageBase64 = '';
  const preview = $('img-source-preview');
  const placeholder = $('img-source-placeholder');
  const clearBtn = $('img-source-clear');
  if (preview) { preview.src = ''; preview.classList.add('hidden'); }
  if (placeholder) placeholder.classList.remove('hidden');
  if (clearBtn) clearBtn.classList.add('hidden');

  // Reset mask editor
  const me = maskEditor;
  me.sourceImg = null;
  me.undoStack = [];
  me.redoStack = [];
  me.dataCtx = null;
  me.bgCtx = null;
  me.maskCtx = null;
  me.uiCtx = null;

  // Clear canvases
  ['img-mask-bg-canvas', 'img-mask-layer', 'img-mask-ui-canvas'].forEach(id => {
    const c = $(id);
    if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); }
  });
}

// --- Keyboard shortcuts for mask editor ---
function maskEditorKeyDown(e) {
  const maskSection = $('img-mask-section');
  if (!maskSection || maskSection.classList.contains('hidden')) return;
  const imagePanel = document.getElementById('image-panel');
  if (!imagePanel || imagePanel.classList.contains('hidden')) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

  switch (e.key) {
    case 'b': case 'B':
      if (!e.ctrlKey && !e.metaKey) { setMaskTool('brush'); e.preventDefault(); }
      break;
    case 'e': case 'E':
      if (!e.ctrlKey && !e.metaKey) { setMaskTool('eraser'); e.preventDefault(); }
      break;
    case '[':
      adjustBrushSize(-5); e.preventDefault();
      break;
    case ']':
      adjustBrushSize(5); e.preventDefault();
      break;
    case 'l': case 'L':
      if (!e.ctrlKey && !e.metaKey) { setMaskTool('lasso'); e.preventDefault(); }
      break;
    case 'r': case 'R':
      if (!e.ctrlKey && !e.metaKey) { setMaskTool('rect'); e.preventDefault(); }
      break;
    case 'i': case 'I':
      if (!e.ctrlKey && !e.metaKey) { invertMask(); e.preventDefault(); }
      break;
    case 'x': case 'X':
      if (!e.ctrlKey && !e.metaKey) { clearMask(); e.preventDefault(); }
      break;
    case 'm': case 'M':
      if (!e.ctrlKey && !e.metaKey) { toggleMaskView(); e.preventDefault(); }
      break;
    case 'z': case 'Z':
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        if (e.shiftKey) maskEditorRedo();
        else maskEditorUndo();
      }
      break;
    case ' ':
      e.preventDefault();
      maskEditor.spaceHeld = true;
      break;
  }
}

function maskEditorKeyUp(e) {
  if (e.key === ' ') maskEditor.spaceHeld = false;
}

// ---------------------------------------------------------------------------
// Gallery (sidebar mini-gallery)
// ---------------------------------------------------------------------------

// Cached availability result — refreshed each panel open. Used to swap the
// gallery empty state for a setup-card when no image-generation path exists
// (no local pipeline, no cloud provider, no fabric peer with image cap).
// The whole-empty-state replacement is gated on `entries.length === 0` so a
// returning user with existing history still sees their gallery if they
// later remove every path (e.g. reverted GPU compose → CPU compose).
let imageAvailability = null;

async function fetchImageAvailability() {
  try {
    const resp = await fetch('/api/image/availability');
    if (!resp.ok) {
      imageAvailability = null;
      return;
    }
    imageAvailability = await resp.json();
  } catch {
    imageAvailability = null;
  }
}

function _renderImageSetupCard(gallery) {
  const a = imageAvailability || {};
  const peersConnected = a.fabric_peers_connected || 0;
  // Tailor the peer chip to whether peers exist but lack image capability,
  // vs. no peers connected at all. Both flow to the same fabric tab — the
  // difference is what the user will see when they land there.
  const peerSubtitle = peersConnected > 0
    ? `${peersConnected} peer${peersConnected === 1 ? '' : 's'} connected, none advertise image`
    : 'Pair another Augmentum instance that has a GPU';
  // Only surface the GPU-variant chip when the backend's torch.cuda probe
  // actually reports a CUDA device. Suggesting compose.gpu.yaml to a
  // Mac/AMD user is wrong — the NVIDIA runtime isn't available. CPU-
  // container users with NVIDIA on the host won't see this chip either
  // (the container can't see the host GPU); README/onboarding remains
  // their path. Better to hide than to mislead.
  const gpuChip = a.host_gpu_detected
    ? '<button type="button" class="img-setup-chip" data-img-setup="gpu">' +
        '<span class="img-setup-chip-title">Run the GPU variant</span>' +
        '<span class="img-setup-chip-sub">compose.gpu.yaml ships DreamShaper 8 pre-baked</span>' +
      '</button>'
    : '';
  gallery.innerHTML =
    '<div class="img-empty-state img-empty-setup">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="32" height="32"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
    '<p><strong>Image generation isn’t available yet.</strong></p>' +
    '<p class="img-empty-sub">Pick a path to start generating:</p>' +
    '<div class="img-setup-actions">' +
      '<button type="button" class="img-setup-chip" data-img-setup="fabric">' +
        '<span class="img-setup-chip-title">Pair a fabric peer</span>' +
        `<span class="img-setup-chip-sub">${escapeHtml(peerSubtitle)}</span>` +
      '</button>' +
      '<button type="button" class="img-setup-chip" data-img-setup="cloud">' +
        '<span class="img-setup-chip-title">Add a cloud provider</span>' +
        '<span class="img-setup-chip-sub">OpenAI, Stability, or any compatible API</span>' +
      '</button>' +
      gpuChip +
    '</div></div>';

  gallery.querySelectorAll('[data-img-setup]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.imgSetup;
      if (target === 'cloud') {
        document.dispatchEvent(new CustomEvent('augmentum:open-settings', {
          detail: { tab: 'providers' },
        }));
      } else if (target === 'fabric') {
        import('./models.js').then((mod) => {
          if (typeof mod.openModelManager === 'function') {
            mod.openModelManager('fabric');
          }
        }).catch(() => showToast('Could not open the model manager', 'error'));
      } else if (target === 'gpu') {
        showToast(
          'Stop the stack and restart with: start.bat (with compose.gpu.yaml in .augmentum.conf)',
          'info',
          8000,
        );
      }
    });
  });
}

async function refreshImageGallery() {
  const gallery = $('img-gallery');
  if (!gallery) return;

  try {
    const resp = await fetch('/api/image/history?limit=20&private=false');
    if (!resp.ok) return;
    const data = await resp.json();
    const entries = Array.isArray(data) ? data : (data.entries || []);

    if (!entries.length) {
      if (imageAvailability && imageAvailability.any_path_available === false) {
        _renderImageSetupCard(gallery);
        return;
      }
      gallery.innerHTML =
        '<div class="img-empty-state">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="32" height="32"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>' +
        '<p>Generated images will appear here.</p></div>';
      return;
    }

    const countEl = $('img-gallery-count');
    if (countEl) countEl.textContent = '(' + entries.length + ')';

    gallery.innerHTML = '';
    for (const entry of entries) {
      // Full image — used by the lightbox for full-quality view, and
      // as the <img onerror> fallback when no thumbnail has been
      // produced yet (e.g. the 7 generations currently drifted out
      // of file_index — those 404 the by-source thumb route).
      const fullUrl = entry.url || '/api/image/' + encodeURIComponent(entry.image_id);
      // 300px WebP thumb via the cacheable route. ~30KB vs ~2MB for
      // the source PNG; for a 20-image gallery that's 600KB vs 40MB
      // per refresh, plus the year-long cache keeps follow-ups free.
      const thumbSrc = '/api/files/thumb/by-source/images/'
        + encodeURIComponent(entry.image_id) + '?size=300';
      const thumb = document.createElement('div');
      thumb.className = 'img-thumb';
      const img = document.createElement('img');
      img.src = thumbSrc;
      img.alt = entry.prompt || '';
      img.loading = 'lazy';
      img.dataset.fallbackSrc = fullUrl;
      img.addEventListener('error', function onerr() {
        // Try the full PNG once before giving up. Setting onerror to
        // null on the same node prevents an infinite loop if both
        // sources fail.
        if (img.src !== img.dataset.fallbackSrc) {
          img.removeEventListener('error', onerr);
          img.src = img.dataset.fallbackSrc;
        }
      });
      thumb.appendChild(img);
      // Lightbox always opens the full asset — thumbnails are only
      // for the grid affordance.
      thumb.addEventListener('click', () => openLightbox(entry, fullUrl));
      gallery.appendChild(thumb);
    }
  } catch { /* silently fail */ }
}

// ---------------------------------------------------------------------------
// Lightbox
// ---------------------------------------------------------------------------

function openLightbox(entry, urlOverride) {
  const modal = $('image-lightbox-modal');
  const imgEl = $('lightbox-img');
  const meta = $('lightbox-meta');
  if (!modal || !imgEl) return;

  const url = urlOverride || entry.url || '';
  currentLightboxEntry = entry;

  const fullUrl = url.startsWith('http') ? url : url;
  imgEl.src = fullUrl;
  if (meta) {
    const parts = [];
    if (entry.prompt) parts.push(entry.prompt.substring(0, 200));
    if (entry.seed && entry.seed !== -1) parts.push('Seed: ' + entry.seed);
    if (entry.width && entry.height) parts.push(entry.width + 'x' + entry.height);
    if (entry.model) parts.push(entry.model);
    if (entry.steps) parts.push(entry.steps + ' steps');
    meta.textContent = parts.join(' \u2022 ');
  }
  modal.classList.add('visible');
}

// --- Lightbox inpaint state ---
let _lbInpaintActive = false;
let _lbMaskEditor = null;

function closeLightbox() {
  if (_lbInpaintActive) closeLightboxInpaint();
  const modal = $('image-lightbox-modal');
  if (modal) {
    modal.classList.remove('visible');
    const actions = modal.querySelector('.lightbox-actions');
    if (actions) actions.style.display = '';
  }
  currentLightboxEntry = null;
}

// ---------------------------------------------------------------------------
// Quality & Speed Section
// ---------------------------------------------------------------------------

function _initQualitySection() {
  // Section collapse toggle
  const qualToggle = $('img-quality-toggle');
  const qualSec = $('img-quality-section');
  if (qualToggle && qualSec) {
    qualToggle.addEventListener('click', () => {
      const collapsed = !qualSec.classList.contains('collapsed');
      qualSec.classList.toggle('collapsed', collapsed);
      qualToggle.classList.toggle('collapsed', collapsed);
      imgSettings.qualityCollapsed = collapsed;
      saveImgSettings();
    });
  }

  // --- Load-time settings (persist to /api/config/tools) ---
  function _saveLoadTimeSetting(key, value) {
    fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    }).catch(() => { /* best effort */ });
  }

  // Load load-time settings from server on init
  fetch('/api/config/tools', { credentials: 'same-origin' }).then(r => r.ok ? r.json() : null).then(data => {
    if (!data) return;
    const freeuEl = $('img-freeu');
    if (freeuEl && data.image_freeu_enabled != null) freeuEl.checked = !!data.image_freeu_enabled;
    const compileEl = $('img-torch-compile');
    if (compileEl && data.image_torch_compile != null) {
      // Support legacy bool values from before the auto/on/off migration
      const val = data.image_torch_compile;
      if (typeof val === 'boolean' || val === 0 || val === 1) {
        compileEl.value = val ? 'on' : 'off';
      } else {
        compileEl.value = String(val);
      }
    }
    const tomeEl = $('img-tome');
    if (tomeEl && data.image_tome_enabled != null) {
      tomeEl.checked = !!data.image_tome_enabled;
      const opts = $('img-tome-opts');
      if (opts) opts.classList.toggle('hidden', !tomeEl.checked);
    }
    const tomeRatioEl = $('img-tome-ratio');
    if (tomeRatioEl && data.image_tome_ratio != null) {
      tomeRatioEl.value = data.image_tome_ratio;
      const valEl = $('img-tome-ratio-val');
      if (valEl) valEl.textContent = parseFloat(tomeRatioEl.value).toFixed(2);
    }
    // Per-gen defaults from config
    const cfgRescaleEl = $('img-cfg-rescale');
    if (cfgRescaleEl && data.image_cfg_rescale != null && !imgSettings.cfgRescale) {
      cfgRescaleEl.value = data.image_cfg_rescale;
      imgSettings.cfgRescale = data.image_cfg_rescale;
      const valEl = $('img-cfg-rescale-val');
      if (valEl) valEl.textContent = parseFloat(cfgRescaleEl.value).toFixed(2);
    }
    const hiresFixEl = $('img-hires-fix');
    if (hiresFixEl && data.image_hires_fix != null && !imgSettings.hiresFix) {
      hiresFixEl.checked = !!data.image_hires_fix;
      imgSettings.hiresFix = !!data.image_hires_fix;
      const opts = $('img-hires-opts');
      if (opts) opts.classList.toggle('hidden', !hiresFixEl.checked);
    }
  }).catch(() => {});

  // FreeU
  const freeuEl = $('img-freeu');
  if (freeuEl) freeuEl.addEventListener('change', () => _saveLoadTimeSetting('image_freeu_enabled', freeuEl.checked));

  // torch.compile
  const compileEl = $('img-torch-compile');
  if (compileEl) compileEl.addEventListener('change', () => {
    _saveLoadTimeSetting('image_torch_compile', compileEl.value);
  });

  // Token Merging
  const tomeEl = $('img-tome');
  if (tomeEl) tomeEl.addEventListener('change', () => {
    _saveLoadTimeSetting('image_tome_enabled', tomeEl.checked);
    const opts = $('img-tome-opts');
    if (opts) opts.classList.toggle('hidden', !tomeEl.checked);
  });
  const tomeRatioEl = $('img-tome-ratio');
  if (tomeRatioEl) tomeRatioEl.addEventListener('input', () => {
    const valEl = $('img-tome-ratio-val');
    if (valEl) valEl.textContent = parseFloat(tomeRatioEl.value).toFixed(2);
    _saveLoadTimeSetting('image_tome_ratio', parseFloat(tomeRatioEl.value));
  });

  // Reload pipeline button
  const reloadBtn = $('img-reload-pipeline-btn');
  if (reloadBtn) reloadBtn.addEventListener('click', async () => {
    reloadBtn.disabled = true;
    reloadBtn.textContent = 'Reloading...';
    try {
      const resp = await fetch('/api/image/reload-pipeline', { method: 'POST' });
      if (resp.ok) {
        showToast('Pipeline reloaded with new settings', 'success');
      } else {
        const err = await resp.json().catch(() => ({}));
        showToast(extractErrorMessage(err, 'Reload failed'), 'error');
      }
    } catch { showToast('Reload failed', 'error'); }
    finally { reloadBtn.disabled = false; reloadBtn.textContent = 'Reload pipeline now'; }
  });

  // --- Per-generation settings ---
  const cfgRescaleEl = $('img-cfg-rescale');
  if (cfgRescaleEl) cfgRescaleEl.addEventListener('input', () => {
    const valEl = $('img-cfg-rescale-val');
    if (valEl) valEl.textContent = parseFloat(cfgRescaleEl.value).toFixed(2);
    saveImageFormToSettings();
  });

  const hiresFixEl = $('img-hires-fix');
  if (hiresFixEl) hiresFixEl.addEventListener('change', () => {
    const opts = $('img-hires-opts');
    if (opts) opts.classList.toggle('hidden', !hiresFixEl.checked);
    saveImageFormToSettings();
  });
  const hiresScaleEl = $('img-hires-scale');
  if (hiresScaleEl) hiresScaleEl.addEventListener('change', () => saveImageFormToSettings());
  const hiresDenoiseEl = $('img-hires-denoise');
  if (hiresDenoiseEl) hiresDenoiseEl.addEventListener('input', () => {
    const valEl = $('img-hires-denoise-val');
    if (valEl) valEl.textContent = parseFloat(hiresDenoiseEl.value).toFixed(2);
    saveImageFormToSettings();
  });
}

// ---------------------------------------------------------------------------
// Generation Stage Polling
// ---------------------------------------------------------------------------

function _startStagePolling() {
  _stopStagePolling();
  _stagePollingInterval = setInterval(async () => {
    try {
      const resp = await fetch('/api/image/generation-status');
      if (!resp.ok) return;
      const data = await resp.json();
      const label = $('img-progress-label');
      if (!label) return;
      // Build the same "stage · step N/M · elapsed" label the shared
      // loader uses, so the Studio panel matches every other surface.
      const stageText = data.pre_queue?.stage || data.stage || (data.active ? 'Generating' : '');
      const parts = [];
      if (stageText) parts.push(stageText);
      if (data.steps_total > 0) parts.push(`step ${data.steps_done}/${data.steps_total}`);
      if (data.queue_size > 0 && !data.active) parts.push(`${data.queue_size} ahead`);
      if (data.elapsed_s > 0) parts.push(`${Math.round(data.elapsed_s)}s`);
      if (parts.length > 0) {
        label.textContent = parts.join(' · ') + '...';
      } else if (data.queue_size > 0) {
        label.textContent = 'Queued...';
      }
    } catch { /* poll failure is non-fatal */ }
  }, 800);
}

function _stopStagePolling() {
  if (_stagePollingInterval) {
    clearInterval(_stagePollingInterval);
    _stagePollingInterval = null;
  }
}

// ---------------------------------------------------------------------------
// Image Generation
// ---------------------------------------------------------------------------

async function handleImageGenerate() {
  if (imageGenerating) return;

  const promptEl = $('img-prompt');
  const prompt = promptEl ? promptEl.value : '';
  if (!prompt.trim()) return;

  const negative = ($('img-negative') || {}).value || '';
  const preset = ($('img-preset') || {}).value || '';
  const width = parseInt(($('img-width') || {}).value) || 512;
  const height = parseInt(($('img-height') || {}).value) || 512;
  const steps = parseInt(($('img-steps') || {}).value) || 20;
  const cfg = parseFloat(($('img-cfg') || {}).value) || 7.0;
  const seed = parseInt(($('img-seed') || {}).value) || -1;
  const sampler = ($('img-sampler') || {}).value || '';
  const model = ($('img-model') || {}).value || '';

  // VRAM courtesy check (skip for cloud models and already-loaded models)
  if (imageHardwareInfo && model && !isSelectedModelCloud()) {
    const selectedModel = imageModelsData.find(m => m.name === model);
    if (selectedModel && !selectedModel.is_loaded) {
      const vramNeeded = { sd15: 2000, sdxl: 5500, flux: 10000 };
      const needed = vramNeeded[selectedModel.pipeline_type] || 0;
      const freeMb = imageHardwareInfo.vram_free_mb || 0;
      const isCpu = imageHardwareInfo.device === 'cpu';
      // When swapping models, current model's VRAM will be freed first
      const loadedModel = imageModelsData.find(m => m.is_loaded);
      const swapFreeMb = loadedModel
        ? freeMb + (vramNeeded[loadedModel.pipeline_type] || 0)
        : freeMb;
      let warn = false;
      if (isCpu && selectedModel.pipeline_type !== 'sd15') warn = true;
      else if (!isCpu && needed > 0 && swapFreeMb < needed) warn = true;
      if (warn && !confirm('This model may exceed your GPU memory. Continue anyway?')) return;
    }
  }

  // Read batch count from UI
  const batchCount = parseInt(document.querySelector('.img-batch-btn.active')?.dataset.count || '1');

  imageGenerating = true;
  imageAbortController = new AbortController();
  const generateBtn = $('img-generate-btn');
  const cancelBtn = $('img-cancel-btn');
  const progress = $('img-progress');
  if (generateBtn) generateBtn.classList.add('hidden');
  if (cancelBtn) cancelBtn.classList.remove('hidden');
  if (progress) progress.classList.remove('hidden');
  const progressLabel = $('img-progress-label');
  if (progressLabel) progressLabel.textContent = batchCount > 1 ? 'Generating 1/' + batchCount + '...' : 'Queued...';
  _startStagePolling();

  try {
    let endpoint, bodyObj;
    const cloudModel = isSelectedModelCloud();
    const cloudProviderId = getSelectedCloudProviderId();
    const cloudQuality = ($('img-cloud-quality') || {}).value || 'standard';

    if (cloudModel) {
      // Route to cloud endpoints
      if (currentImageMode === 'img2img' || currentImageMode === 'inpaint') {
        if (!sourceImageBase64) { showToast('Please upload a source image first', 'error'); return; }
        const maskB64 = currentImageMode === 'inpaint' ? getMaskBase64() : '';
        if (currentImageMode === 'inpaint' && !maskB64) { showToast('Please paint a mask on the image', 'error'); return; }
        const strength = parseFloat(($('img-strength') || {}).value) || 0.75;
        endpoint = '/api/image/cloud/edit';
        bodyObj = { prompt, provider_id: cloudProviderId, model, source_image: sourceImageBase64, mask_image: maskB64, strength, width, height, quality: cloudQuality, n: batchCount };
      } else {
        endpoint = '/api/image/cloud/generate';
        bodyObj = { prompt, negative_prompt: negative, provider_id: cloudProviderId, model, width, height, quality: cloudQuality, seed, n: batchCount };
      }
    } else if (currentImageMode === 'img2img') {
      if (!sourceImageBase64) { showToast('Please upload a source image first', 'error'); return; }
      const strength = parseFloat(($('img-strength') || {}).value) || 0.75;
      endpoint = '/api/image/img2img';
      bodyObj = { prompt, negative_prompt: negative, model, source_image: sourceImageBase64, strength, width, height, steps, cfg_scale: cfg, seed, sampler: sampler || undefined, preset };
    } else if (currentImageMode === 'inpaint') {
      if (!sourceImageBase64) { showToast('Please upload a source image first', 'error'); return; }
      const maskB64 = getMaskBase64();
      if (!maskB64) { showToast('Please paint a mask on the image', 'error'); return; }
      const strength = parseFloat(($('img-strength') || {}).value) || 1.0;
      endpoint = '/api/image/inpaint';
      bodyObj = { prompt, negative_prompt: negative, model, source_image: sourceImageBase64, mask_image: maskB64, strength, width, height, steps, cfg_scale: cfg, seed, sampler: sampler || undefined, preset, inpaint_mode: currentInpaintMode };
      // Full-resolution inpaint
      const fullresEl = $('img-fullres');
      if (fullresEl && fullresEl.checked) {
        bodyObj.full_res = true;
        bodyObj.full_res_padding = parseInt(($('img-fullres-padding') || {}).value) || 32;
      }
    } else {
      endpoint = '/api/image/generate';
      bodyObj = { prompt, negative_prompt: negative, preset, width, height, steps, cfg_scale: cfg, seed, sampler: sampler || undefined, model };
      // Provenance: the architect-dispatched path armed a one-shot
      // flag before clicking the canonical button — mark this
      // generation companion-created so the gallery can filter it.
      if (_architectOriginPending) {
        bodyObj.origin = 'companion';
        _architectOriginPending = false;
      }
      // Per-gen quality fields
      if (imgSettings.cfgRescale > 0) bodyObj.guidance_rescale = imgSettings.cfgRescale;
      if (imgSettings.hiresFix) {
        bodyObj.hires_fix = true;
        bodyObj.hires_scale = imgSettings.hiresScale;
        bodyObj.hires_denoise = imgSettings.hiresDenoise;
      }
      // IP-Adapter reference image(s) — single or multiple
      const _ipRef = initImagePanel._getIpRef ? initImagePanel._getIpRef() : null;
      if (_ipRef && _ipRef.data) {
        bodyObj.ip_adapter_image = _ipRef.data;  // string or array of strings
        bodyObj.ip_adapter_scale = _ipRef.scale;
      }
      // LoRA adapters
      const loras = getActiveLoras();
      if (loras.length) bodyObj.loras = loras;
    }

    const condenseEl = $('img-condense-model');
    if (!cloudModel && condenseEl && condenseEl.value) bodyObj.condense_model = condenseEl.value;

    // CLIP skip (local only, SD1.5/SDXL)
    const clipSkipEl = $('img-clip-skip');
    if (!cloudModel && clipSkipEl && clipSkipEl.value) {
      bodyObj.clip_skip = parseInt(clipSkipEl.value);
    }

    // Determine iteration count: cloud uses n param natively, local loops
    const isLocalBatch = !cloudModel && batchCount > 1;
    const iterations = isLocalBatch ? batchCount : 1;
    let lastData = null;

    for (let i = 0; i < iterations; i++) {
      if (imageAbortController?.signal.aborted) break;

      // For local batch: update progress and increment seed
      if (isLocalBatch) {
        if (progressLabel) progressLabel.textContent = `Generating ${i + 1}/${batchCount}...`;
        if (i > 0 && seed !== -1) bodyObj.seed = seed + i;
      }

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyObj),
        signal: imageAbortController.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Generation failed' }));
        showToast(extractErrorMessage(err, 'Image generation failed'), 'error');
        if (isLocalBatch && i > 0) break; // partial batch — keep what we have
        return;
      }

      const data = await resp.json();
      refreshImageGallery();

      // Cloud batch: backend may return { images: [...] } for n>1
      if (data.images && Array.isArray(data.images)) {
        lastData = data.images[data.images.length - 1];
      } else {
        lastData = data;
      }
    }

    // Open lightbox on the last generated image
    if (lastData && lastData.url) {
      openLightbox({
        image_id: lastData.image_id || '',
        prompt: lastData.prompt || prompt,
        negative_prompt: lastData.negative_prompt || negative,
        seed: lastData.seed || -1,
        width: lastData.width || width,
        height: lastData.height || height,
        steps: lastData.steps || steps,
        cfg_scale: lastData.cfg_scale || cfg,
        model: lastData.model || model,
      }, lastData.url);
      // Notify SurfaceFlows subscribers so chat surfaces that opted in
      // to image events can pick up the latest render.
      document.dispatchEvent(new CustomEvent('augmentum:image-generated', {
        detail: { url: lastData.url, prompt: lastData.prompt || prompt },
      }));
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      showToast('Image generation cancelled', 'warning');
    } else {
      showToast('Image generation failed: ' + err.message, 'error');
    }
  } finally {
    _stopStagePolling();
    imageGenerating = false;
    imageAbortController = null;
    if (generateBtn) generateBtn.classList.remove('hidden');
    if (cancelBtn) cancelBtn.classList.add('hidden');
    if (progress) progress.classList.add('hidden');
    // Refresh hardware info + model list (VRAM changed after load/generate)
    fetchImageHardware();
    refreshImageModels();
  }
}

// ---------------------------------------------------------------------------
// Model Rename
// ---------------------------------------------------------------------------

async function handleImageModelRename() {
  const modelEl = $('img-model');
  const oldName = modelEl ? modelEl.value : '';
  if (!oldName) { showToast('Select a model to rename', 'warning'); return; }

  const newName = prompt('Enter new name for "' + oldName + '":', oldName);
  if (!newName || newName.trim() === '' || newName.trim() === oldName) return;

  try {
    const resp = await fetch('/api/image/models/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_name: oldName, new_name: newName.trim() }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Rename failed'), 'error');
      return;
    }
    showToast('Renamed to "' + newName.trim() + '"', 'success');
    refreshImageModels();
  } catch (err) {
    showToast('Rename failed: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Poll-based Model Downloads
// ---------------------------------------------------------------------------

function pollPullTask(taskId, onProgress, onComplete, onError) {
  let consecutiveErrors = 0;
  const maxRetries = 10;
  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/api/image/models/pull/' + taskId);
      if (!resp.ok) {
        if (resp.status === 404) {
          clearInterval(interval);
          onError('Download task not found (may have completed)');
          return;
        }
        consecutiveErrors++;
        if (consecutiveErrors >= maxRetries) {
          clearInterval(interval);
          onError('Lost connection after ' + maxRetries + ' retries');
        }
        return;
      }
      const data = await resp.json();
      consecutiveErrors = 0;

      if (data.status === 'running') onProgress(data);
      else if (data.status === 'complete' || data.status === 'exists') {
        clearInterval(interval);
        onComplete(data);
      } else if (data.status === 'error') {
        clearInterval(interval);
        onError(data.error || 'Unknown error');
      }
    } catch (err) {
      consecutiveErrors++;
      if (consecutiveErrors >= maxRetries) {
        clearInterval(interval);
        onError('Polling failed: ' + err.message);
      }
    }
  }, 2000);
  return interval;
}

// ---------------------------------------------------------------------------
// Two-step pull: Detect → select variant → Download
// ---------------------------------------------------------------------------

let _pullDetected = null; // cached detect result
let _pullState = 'detect'; // 'detect' | 'download'

function _resetPullState() {
  _pullDetected = null;
  _pullState = 'detect';
  const btn = $('img-pull-btn');
  const variantsArea = $('img-pull-variants');
  if (btn) btn.textContent = 'Detect';
  if (variantsArea) variantsArea.classList.add('hidden');
}

async function handleImagePull() {
  const input = $('img-pull-input');
  const btn = $('img-pull-btn');
  if (!input || !input.value.trim()) return;

  if (_pullState === 'detect') {
    await _handleDetect(input, btn);
  } else {
    await _handleDownload(input, btn);
  }
}

async function _handleDetect(input, btn) {
  const source = input.value.trim();
  btn.disabled = true;
  btn.textContent = 'Detecting...';

  try {
    const resp = await fetch('/api/image/models/detect?source=' + encodeURIComponent(source));
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(err.detail || err.error || 'Detection failed', 'error');
      btn.disabled = false;
      btn.textContent = 'Detect';
      return;
    }

    const data = await resp.json();
    if (data.error) {
      showToast(data.error, 'error');
      btn.disabled = false;
      btn.textContent = 'Detect';
      return;
    }

    _pullDetected = data;
    const variants = data.variants || [];
    if (variants.length === 0) {
      showToast('No downloadable variants found', 'error');
      btn.disabled = false;
      btn.textContent = 'Detect';
      return;
    }

    // Show variant selector
    const variantsArea = $('img-pull-variants');
    const modelName = $('img-pull-model-name');
    const select = $('img-pull-variant-select');
    const typeLabel = data.model_type === 'lora' ? ' [LoRA]' : '';
    const baseLabel = data.base_model_raw ? ' (' + data.base_model_raw + ')' : '';
    if (modelName) modelName.textContent = (data.name || source) + typeLabel + baseLabel;
    if (select) {
      select.innerHTML = variants.map((v, i) => {
        const sizeText = v.size_gb ? ` (${v.size_gb} GB)` : '';
        const label = (v.label || v.file_name || 'Variant ' + (i + 1)) + sizeText;
        const primary = v.primary ? ' \u2605' : '';
        return '<option value="' + i + '">' + escapeHtml(label + primary) + '</option>';
      }).join('');
    }
    if (variantsArea) variantsArea.classList.remove('hidden');

    _pullState = 'download';
    btn.textContent = 'Download';
    btn.disabled = false;
  } catch (err) {
    showToast('Detection failed: ' + err.message, 'error');
    btn.disabled = false;
    btn.textContent = 'Detect';
  }
}

async function _handleDownload(input, btn) {
  if (!_pullDetected) { _resetPullState(); return; }

  const select = $('img-pull-variant-select');
  const idx = select ? parseInt(select.value) : 0;
  const variant = (_pullDetected.variants || [])[idx];
  if (!variant) { _resetPullState(); return; }

  const progressArea = $('img-pull-progress');
  const fill = $('img-pull-fill');
  const statusEl = $('img-pull-status');
  if (progressArea) progressArea.classList.remove('hidden');
  if (statusEl) statusEl.textContent = 'Starting download...';
  if (fill) fill.style.width = '0%';
  btn.disabled = true;
  btn.textContent = 'Downloading...';

  // Build pull request based on source type
  const body = { source: input.value.trim() };
  if (_pullDetected.source_type === 'civitai' && variant.download_url) {
    body.source = variant.download_url;
  } else if (_pullDetected.source_type === 'huggingface') {
    if (variant.variant) body.variant = variant.variant;
    if (variant.allow_patterns) body.allow_patterns = variant.allow_patterns;
  }
  // Route LoRAs to the loras/ directory with metadata
  if (_pullDetected.model_type === 'lora') {
    body.asset_type = 'lora';
    if (_pullDetected.trigger_words?.length) {
      body.trigger_words = _pullDetected.trigger_words;
    }
    if (_pullDetected.base_model) {
      body.base_model = _pullDetected.base_model;
    }
  }

  try {
    const resp = await fetch('/api/image/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      if (statusEl) statusEl.textContent = 'Error: ' + (errBody.detail || errBody.error || 'HTTP ' + resp.status);
      setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); _resetPullState(); }, 4000);
      return;
    }

    const task = await resp.json();
    if (statusEl) statusEl.textContent = 'Downloading...';

    pollPullTask(task.task_id,
      (data) => {
        if (data.percent !== undefined) {
          const pct = Math.round(data.percent);
          if (fill) fill.style.width = Math.max(pct, 2) + '%';
          let pctText = pct + '%';
          if (data.downloaded && data.total) pctText += ' \u2014 ' + formatBytes(data.downloaded) + ' / ' + formatBytes(data.total);
          if (statusEl) statusEl.textContent = pctText;
        }
      },
      (data) => {
        if (fill) fill.style.width = '100%';
        if (statusEl) statusEl.textContent = data.status === 'exists' ? 'Already exists.' : 'Complete!';
        refreshImageModels();
        refreshImageCatalog();
        refreshLoraList();
        setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); _resetPullState(); }, 3000);
      },
      (errMsg) => {
        if (statusEl) statusEl.textContent = 'Error: ' + errMsg;
        setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); _resetPullState(); }, 6000);
      }
    );
  } catch (err) {
    if (statusEl) statusEl.textContent = 'Download failed: ' + err.message;
    setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); _resetPullState(); }, 4000);
  }
}

// ---------------------------------------------------------------------------
// Custom Import (file upload tab)
// Mirrors the URL importer's UX rhythm: pick → confirm → progress.
// Backend endpoints: GET /api/image/gguf-families, POST /api/image/models/import
// ---------------------------------------------------------------------------

let _customImportFile = null;
let _customImportFamilies = null;

function _switchImportTab(tab) {
  document.querySelectorAll('.img-import-tab').forEach(t => {
    const active = t.dataset.tab === tab;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  document.querySelectorAll('.img-import-pane').forEach(p => p.classList.add('hidden'));
  const pane = $('img-import-pane-' + tab);
  if (pane) pane.classList.remove('hidden');
}

function _resetImportState() {
  _customImportFile = null;
  const drop = $('img-import-drop');
  if (drop) { drop.classList.remove('hidden', 'drag-over'); }
  const confirm = $('img-import-confirm');
  if (confirm) confirm.classList.add('hidden');
  const progress = $('img-import-progress');
  if (progress) progress.classList.add('hidden');
  const fileInput = $('img-import-file');
  if (fileInput) fileInput.value = '';
  ['img-import-name', 'img-import-base-repo', 'img-import-pipeline-class', 'img-import-transformer-class'].forEach(id => {
    const el = $(id);
    if (el) el.value = '';
  });
  const advBar = $('img-import-advanced-bar');
  if (advBar) advBar.classList.add('hidden');
  const advanced = $('img-import-advanced');
  if (advanced) advanced.classList.add('hidden');
  const familyRow = $('img-import-family-row');
  if (familyRow) familyRow.classList.add('hidden');
}

async function _loadGgufFamilies() {
  if (_customImportFamilies) return _customImportFamilies;
  try {
    const resp = await fetch('/api/image/gguf-families');
    if (!resp.ok) { _customImportFamilies = []; return _customImportFamilies; }
    const data = await resp.json();
    _customImportFamilies = data.families || [];
  } catch {
    _customImportFamilies = [];
  }
  return _customImportFamilies;
}

async function _readGgufMagic(file) {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const bytes = new Uint8Array(reader.result);
        resolve(String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]));
      } catch { resolve(''); }
    };
    reader.onerror = () => resolve('');
    reader.readAsArrayBuffer(file.slice(0, 4));
  });
}

function _inferKindFromName(name) {
  const lower = name.toLowerCase();
  if (lower.endsWith('.gguf')) return 'gguf';
  if (lower.endsWith('.zip')) return 'diffusers-zip';
  if (lower.endsWith('.safetensors')) return 'safetensors-single';
  return '';
}

async function _stageImportFile(file) {
  if (!file) return;
  _customImportFile = file;

  const drop = $('img-import-drop');
  if (drop) drop.classList.add('hidden');
  const confirm = $('img-import-confirm');
  if (confirm) confirm.classList.remove('hidden');

  const summary = $('img-import-file-summary');
  if (summary) summary.textContent = file.name + ' — ' + formatBytes(file.size);

  const nameEl = $('img-import-name');
  if (nameEl) {
    const stem = file.name.replace(/\.[^.]+$/, '');
    nameEl.value = stem.replace(/[^A-Za-z0-9._-]+/g, '-').slice(0, 64);
  }

  const kind = _inferKindFromName(file.name);
  // Verify GGUF magic header — filename can lie
  if (kind === 'gguf') {
    const magic = await _readGgufMagic(file);
    const status = $('img-import-status');
    if (magic && magic !== 'GGUF' && status) {
      status.textContent = 'Warning: filename says .gguf but header is "' + magic + '"';
    }
  }

  // Populate family dropdown for GGUFs; auto-suggest by filename pattern
  const familyRow = $('img-import-family-row');
  const advBar = $('img-import-advanced-bar');
  if (kind === 'gguf') {
    const families = await _loadGgufFamilies();
    const select = $('img-import-family');
    if (select) {
      select.innerHTML = families.map(f =>
        '<option value="' + escapeHtml(f.family) + '">' +
        escapeHtml(f.family) + ' — ' + escapeHtml(f.base_repo) + '</option>'
      ).join('');
      const lowerName = file.name.toLowerCase();
      const match = families.find(f => (f.name_patterns || []).some(p => p && lowerName.includes(p)));
      if (match) select.value = match.family;
    }
    if (familyRow) familyRow.classList.remove('hidden');
    if (advBar) advBar.classList.remove('hidden');
  } else {
    if (familyRow) familyRow.classList.add('hidden');
    if (advBar) advBar.classList.add('hidden');
  }
}

function _handleImportInstall() {
  if (!_customImportFile) return;
  const file = _customImportFile;
  const name = ($('img-import-name')?.value || '').trim();
  if (!name) {
    showToast('Please enter a model name', 'error');
    return;
  }
  const kind = _inferKindFromName(file.name);
  if (!kind) {
    showToast('Unsupported file type', 'error');
    return;
  }

  const confirm = $('img-import-confirm');
  if (confirm) confirm.classList.add('hidden');
  const progressArea = $('img-import-progress');
  const fill = $('img-import-fill');
  const status = $('img-import-status');
  if (progressArea) progressArea.classList.remove('hidden');
  if (fill) fill.style.width = '0%';
  if (status) status.textContent = 'Uploading...';

  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  formData.append('kind', kind);
  const familyEl = $('img-import-family');
  const advanced = $('img-import-advanced');
  const useAdvanced = advanced && !advanced.classList.contains('hidden');
  if (useAdvanced) {
    formData.append('gguf_base_repo', ($('img-import-base-repo')?.value || '').trim());
    formData.append('gguf_pipeline_class', ($('img-import-pipeline-class')?.value || '').trim());
    formData.append('gguf_transformer_class', ($('img-import-transformer-class')?.value || '').trim());
  } else if (kind === 'gguf' && familyEl) {
    formData.append('gguf_family', familyEl.value);
  }
  const prefetchEl = $('img-import-prefetch');
  formData.append('prefetch_base_components', prefetchEl?.checked ? 'true' : 'false');

  // XHR — gives us request-side upload progress that fetch() can't expose
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/image/models/import');
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      if (fill) fill.style.width = pct + '%';
      if (status) status.textContent = pct + '% — ' + formatBytes(e.loaded) + ' / ' + formatBytes(e.total);
    }
  };
  xhr.onload = () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      let errMsg = 'Install failed';
      try {
        const errBody = JSON.parse(xhr.responseText);
        errMsg = errBody.detail || errBody.error || errMsg;
      } catch { /* ignore */ }
      if (status) status.textContent = 'Error: ' + errMsg;
      setTimeout(_resetImportState, 6000);
      return;
    }
    let resp;
    try { resp = JSON.parse(xhr.responseText); } catch { resp = {}; }
    if (resp.status === 'complete') {
      if (fill) fill.style.width = '100%';
      if (status) status.textContent = 'Complete!';
      refreshImageModels();
      refreshImageCatalog();
      setTimeout(_resetImportState, 3000);
    } else if (resp.task_id) {
      if (status) status.textContent = 'Pre-fetching base components...';
      pollPullTask(resp.task_id,
        (data) => {
          if (data.percent !== undefined) {
            const pct = Math.round(data.percent);
            if (fill) fill.style.width = pct + '%';
            const phase = data.phase || 'Pre-fetching...';
            if (status) status.textContent = pct + '% — ' + phase;
          }
        },
        () => {
          if (fill) fill.style.width = '100%';
          if (status) status.textContent = 'Complete!';
          refreshImageModels();
          refreshImageCatalog();
          setTimeout(_resetImportState, 3000);
        },
        (errMsg) => {
          if (status) status.textContent = 'Prefetch error: ' + errMsg;
          setTimeout(_resetImportState, 6000);
        }
      );
    } else {
      if (status) status.textContent = 'Unknown response';
      setTimeout(_resetImportState, 4000);
    }
  };
  xhr.onerror = () => {
    if (status) status.textContent = 'Network error during upload';
    setTimeout(_resetImportState, 4000);
  };
  xhr.send(formData);
}

function _initCustomImportPanel() {
  document.querySelectorAll('.img-import-tab').forEach(tab => {
    tab.addEventListener('click', () => _switchImportTab(tab.dataset.tab));
  });

  const drop = $('img-import-drop');
  const fileInput = $('img-import-file');
  const pickBtn = $('img-import-pick');

  if (pickBtn && fileInput) {
    pickBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
  }
  if (drop && fileInput) {
    drop.addEventListener('click', (e) => {
      if (e.target === pickBtn) return;
      fileInput.click();
    });
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
      drop.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(evt => {
      drop.addEventListener(evt, () => drop.classList.add('drag-over'));
    });
    ['dragleave', 'drop'].forEach(evt => {
      drop.addEventListener(evt, () => drop.classList.remove('drag-over'));
    });
    drop.addEventListener('drop', e => {
      const f = e.dataTransfer?.files?.[0];
      if (f) _stageImportFile(f);
    });
  }
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files?.[0]) _stageImportFile(fileInput.files[0]);
    });
  }

  const advToggle = $('img-import-advanced-toggle');
  const advanced = $('img-import-advanced');
  const familyRow = $('img-import-family-row');
  if (advToggle && advanced) {
    advToggle.addEventListener('click', () => {
      const opening = advanced.classList.contains('hidden');
      advanced.classList.toggle('hidden');
      // Advanced overrides family — hide family row when advanced is open
      if (familyRow) familyRow.classList.toggle('hidden', opening);
    });
  }

  const installBtn = $('img-import-install');
  const cancelBtn = $('img-import-cancel');
  if (installBtn) installBtn.addEventListener('click', _handleImportInstall);
  if (cancelBtn) cancelBtn.addEventListener('click', _resetImportState);
}

// ---------------------------------------------------------------------------
// LoRA Adapters
// ---------------------------------------------------------------------------

let _activeLoras = [];  // [{name, weight}] — sent with each generation request

async function refreshLoraList() {
  const container = $('img-lora-list');
  if (!container) return;
  try {
    const resp = await fetch('/api/image/loras');
    if (!resp.ok) return;
    const loras = await resp.json();
    if (!loras.length) {
      container.innerHTML = '<div class="img-empty-state"><p>No LoRAs installed. Download one below.</p></div>';
      return;
    }
    container.innerHTML = loras.map(l => {
      const active = _activeLoras.find(a => a.name === l.name);
      const checked = active ? 'checked' : '';
      const weight = active ? active.weight : 0.75;
      const triggers = l.trigger_words?.length
        ? '<div class="img-lora-triggers">Triggers: ' + l.trigger_words.map(w => '<code>' + escapeHtml(w) + '</code>').join(', ') + '</div>'
        : '';
      const baseLabel = l.base_model
        ? '<span class="img-lora-base">' + escapeHtml(l.base_model.toUpperCase()) + '</span>'
        : '';
      return (
        '<div class="img-lora-card" data-lora-name="' + escapeHtml(l.name) + '" data-base-model="' + escapeHtml(l.base_model || '') + '">'
        + '<div class="img-lora-header">'
        +   '<label class="img-lora-toggle">'
        +     '<input type="checkbox" class="img-lora-check" ' + checked + '>'
        +     '<span class="img-lora-name">' + escapeHtml(l.name) + '</span>'
        +   '</label>'
        +   baseLabel
        +   '<span class="img-lora-size">' + l.size_mb + ' MB</span>'
        + '</div>'
        + '<div class="img-lora-weight' + (active ? '' : ' hidden') + '">'
        +   '<input type="range" class="img-lora-slider" min="0" max="1" step="0.05" value="' + weight + '">'
        +   '<span class="img-lora-weight-val">' + weight.toFixed(2) + '</span>'
        + '</div>'
        + triggers
        + '</div>'
      );
    }).join('');

    // Wire toggle + slider handlers
    container.querySelectorAll('.img-lora-check').forEach(cb => {
      cb.addEventListener('change', () => _onLoraToggle(cb));
    });
    container.querySelectorAll('.img-lora-slider').forEach(slider => {
      slider.addEventListener('input', () => {
        const card = slider.closest('.img-lora-card');
        const name = card.dataset.loraName;
        const val = parseFloat(slider.value);
        card.querySelector('.img-lora-weight-val').textContent = val.toFixed(2);
        const entry = _activeLoras.find(a => a.name === name);
        if (entry) entry.weight = val;
      });
    });
  } catch { /* silently fail */ }
}

function _onLoraToggle(cb) {
  const card = cb.closest('.img-lora-card');
  const name = card.dataset.loraName;
  const weightRow = card.querySelector('.img-lora-weight');
  if (cb.checked) {
    const slider = card.querySelector('.img-lora-slider');
    const weight = parseFloat(slider.value);
    _activeLoras.push({ name, weight });
    weightRow.classList.remove('hidden');
  } else {
    _activeLoras = _activeLoras.filter(a => a.name !== name);
    weightRow.classList.add('hidden');
  }
}

/** Get active LoRAs for inclusion in generation requests. */
function getActiveLoras() {
  return _activeLoras.map(a => ({ name: a.name, weight: a.weight }));
}

// ---------------------------------------------------------------------------
// Catalog Toggle (Models / LoRAs)
// ---------------------------------------------------------------------------

function initCatalogToggle() {
  const toggle = $('img-catalog-toggle');
  if (!toggle) return;
  toggle.querySelectorAll('.img-catalog-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      toggle.querySelectorAll('.img-catalog-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const isLoras = tab.dataset.tab === 'loras';
      const modelCatalog = $('img-catalog');
      const loraCatalog = $('img-lora-catalog');
      if (modelCatalog) modelCatalog.classList.toggle('hidden', isLoras);
      if (loraCatalog) loraCatalog.classList.toggle('hidden', !isLoras);
      if (isLoras && !_loraCatalogLoaded) refreshLoraCatalog();
    });
  });
}

// ---------------------------------------------------------------------------
// LoRA Catalog (curated recommendations)
// ---------------------------------------------------------------------------

let _loraCatalogLoaded = false;

async function refreshLoraCatalog() {
  const container = $('img-lora-catalog');
  if (!container) return;
  try {
    const resp = await fetch('/api/image/loras/catalog');
    if (!resp.ok) {
      container.innerHTML = '<div class="img-empty-state"><p>LoRA catalog unavailable</p></div>';
      return;
    }
    const catalog = await resp.json();
    if (!catalog.length) {
      container.innerHTML = '<div class="img-empty-state"><p>No curated LoRAs available</p></div>';
      return;
    }
    _loraCatalogLoaded = true;

    container.innerHTML = catalog.map(l => {
      const baseColor = l.base_model === 'sd15' ? 'var(--success)' : l.base_model === 'sdxl' ? 'var(--accent)' : 'var(--warning)';
      const compatBadge = l.compatible
        ? '<span class="catalog-badge installed">Compatible</span>'
        : '';
      const installedBadge = l.installed
        ? '<span class="catalog-badge installed">Installed</span>'
        : '';
      return (
        '<div class="catalog-card' + (l.compatible ? '' : ' incompatible') + '" data-civitai-id="' + escapeHtml(l.civitai_id) + '">'
        + '<div class="catalog-card-header">'
        +   '<span class="catalog-card-name">' + escapeHtml(l.name) + '</span>'
        +   '<div class="catalog-card-badges">'
        +     '<span class="catalog-badge" style="background:' + baseColor + '">' + escapeHtml(l.base_model.toUpperCase()) + '</span>'
        +     '<span class="catalog-badge">' + escapeHtml(l.category) + '</span>'
        +     compatBadge + installedBadge
        +   '</div>'
        + '</div>'
        + '<div class="catalog-card-desc">' + escapeHtml(l.description) + '</div>'
        + '<div class="catalog-card-meta">'
        +   '<div class="catalog-card-specs">'
        +     '<span class="catalog-spec">' + l.size_mb + ' MB</span>'
        +     (l.trigger_words?.length ? '<span class="catalog-spec">Trigger: ' + l.trigger_words.map(w => escapeHtml(w)).join(', ') + '</span>' : '')
        +   '</div>'
        +   '<div class="catalog-card-actions">'
        +     (l.installed
              ? '<button class="btn btn-sm" disabled>Installed</button>'
              : '<button class="btn btn-sm btn-primary img-lora-dl-btn" data-civitai-id="' + escapeHtml(l.civitai_id) + '" data-base="' + escapeHtml(l.base_model) + '" data-triggers=\'' + escapeHtml(JSON.stringify(l.trigger_words || [])) + '\'>Download</button>')
        +   '</div>'
        + '</div>'
        + '</div>'
      );
    }).join('');

    // Wire download buttons
    container.querySelectorAll('.img-lora-dl-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const civitaiId = btn.dataset.civitaiId;
        const baseModel = btn.dataset.base;
        let triggers = [];
        try { triggers = JSON.parse(btn.dataset.triggers); } catch {}
        btn.disabled = true;
        btn.textContent = 'Downloading...';
        try {
          const resp = await fetch('/api/image/models/pull', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              source: 'https://civitai.com/models/' + civitaiId,
              asset_type: 'lora',
              base_model: baseModel,
              trigger_words: triggers,
            }),
          });
          if (resp.ok) {
            const task = await resp.json();
            pollPullTask(task.task_id,
              () => {},
              () => { btn.textContent = 'Installed'; refreshLoraList(); refreshLoraCatalog(); },
              (err) => { btn.textContent = 'Error'; btn.disabled = false; }
            );
          } else {
            btn.textContent = 'Error';
            btn.disabled = false;
          }
        } catch {
          btn.textContent = 'Error';
          btn.disabled = false;
        }
      });
    });
  } catch {
    container.innerHTML = '<div class="img-empty-state"><p>Could not load LoRA catalog</p></div>';
  }
}

// ---------------------------------------------------------------------------
// Model Catalog
// ---------------------------------------------------------------------------

let _catalogData = [];
async function refreshImageCatalog() {
  const container = $('img-catalog');
  if (!container) return;

  try {
    const resp = await fetch('/api/image/models/catalog');
    if (!resp.ok) {
      container.innerHTML = '<div class="img-empty-state"><p>Catalog unavailable</p></div>';
      return;
    }
    const catalog = await resp.json();
    if (!catalog || !catalog.length) {
      container.innerHTML = '<div class="img-empty-state"><p>No models in catalog</p></div>';
      return;
    }

    _catalogData = catalog;
    _renderCatalog(container, catalog);
  } catch {
    container.innerHTML = '<div class="img-empty-state"><p>Could not load catalog</p></div>';
  }
}

function _renderCatalog(container, catalog) {
  const mode = currentImageMode;
  const capKey = mode === 'txt2img' ? 'txt2img' : mode === 'img2img' ? 'img2img' : 'inpaint';

  // Sort: native first, then fallback, then unsupported; installed first within each tier
  const sorted = [...catalog].sort((a, b) => {
    const capA = (a.capabilities || {})[capKey] || 'fallback';
    const capB = (b.capabilities || {})[capKey] || 'fallback';
    const rankMap = { yes: 0, fallback: 1, no: 2 };
    const rankDiff = (rankMap[capA] ?? 1) - (rankMap[capB] ?? 1);
    if (rankDiff !== 0) return rankDiff;
    if (a.installed && !b.installed) return -1;
    if (!a.installed && b.installed) return 1;
    return 0;
  });

  container.innerHTML = '';
  for (const m of sorted) container.appendChild(buildCatalogCard(m));
  reconnectCatalogDownloads();
}

function buildCatalogCard(model) {
  const card = document.createElement('div');
  card.className = 'catalog-card' + (model.compatible ? '' : ' incompatible');
  const isGguf = !!(model.allow_patterns && model.allow_patterns.length);

  const vramText = model.min_vram_mb > 0
    ? (model.min_vram_mb >= 1000 ? (model.min_vram_mb / 1000) + 'GB VRAM' : model.min_vram_mb + 'MB VRAM')
    : 'No GPU required';

  // Capability badge for current mode
  const capKey = currentImageMode === 'txt2img' ? 'txt2img' : currentImageMode === 'img2img' ? 'img2img' : 'inpaint';
  const caps = model.capabilities || {};
  const support = caps[capKey] || 'fallback';
  const capLabel = support === 'yes' ? '\u25CF Native' : support === 'fallback' ? '\u25D0 Fallback' : '\u25CB Unsupported';
  const capClass = support === 'yes' ? 'cap-native' : support === 'fallback' ? 'cap-fallback' : 'cap-none';

  let badges = '<span class="catalog-badge ' + capClass + '">' + capLabel + '</span>';
  badges += '<span class="catalog-badge pipeline-' + model.pipeline_type + '">' + model.pipeline_type.toUpperCase() + '</span>';
  if (isGguf) badges += '<span class="catalog-badge gguf-tag">GGUF</span>';
  // Show precision badge for single-variant models (no dropdown shown)
  if (!isGguf && model.precision_variants && model.precision_variants.length === 1) {
    badges += '<span class="catalog-badge precision-tag">' + model.precision_variants[0].variant.toUpperCase() + '</span>';
  }
  if (model.cpu_friendly) badges += '<span class="catalog-badge cpu-ok">CPU OK</span>';
  if (model.installed) badges += '<span class="catalog-badge installed">Installed</span>';

  // Dim unsupported models
  if (support === 'no') card.style.opacity = '0.5';

  let btnClass = 'catalog-download-btn';
  let btnText = 'Download';
  let btnDisabled = '';
  if (model.installed) {
    btnClass += ' installed-btn';
    btnText = 'Installed';
    btnDisabled = ' disabled';
  }

  const deleteBtn = model.installed
    ? '<button class="catalog-delete-btn" data-name="' + escapeHtml(model.installed_name || model.name) + '" title="Delete model from disk">' +
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
          '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>' +
        '</svg>' +
      '</button>'
    : '';

  // For GGUF models, add a quant selector row
  const quantRow = isGguf && !model.installed
    ? '<div class="catalog-quant-row">' +
        '<select class="catalog-quant-select" data-repo="' + escapeHtml(model.repo_id) + '">' +
          '<option value="" disabled selected>Select quantization…</option>' +
        '</select>' +
        '<span class="catalog-quant-size"></span>' +
      '</div>'
    : '';

  // For transformers models with precision variants, add a precision selector
  const hasPrecision = !isGguf && model.precision_variants && model.precision_variants.length > 1 && !model.installed;
  const precisionRow = hasPrecision
    ? '<div class="catalog-quant-row">' +
        '<select class="catalog-precision-select" data-repo="' + escapeHtml(model.repo_id) + '">' +
          model.precision_variants.map(function(pv) {
            const rec = pv.variant === 'fp16';
            return '<option value="' + escapeHtml(pv.variant) + '" data-size-gb="' + pv.size_gb + '"' +
              (rec ? ' selected' : '') + '>' +
              escapeHtml(pv.label) + ' (' + pv.size_gb + ' GB)</option>';
          }).join('') +
        '</select>' +
      '</div>'
    : '';

  card.innerHTML =
    '<div class="catalog-card-header">' +
      '<span class="catalog-card-name">' + escapeHtml(model.name) + '</span>' +
      '<div class="catalog-card-badges">' + badges + '</div>' +
    '</div>' +
    '<div class="catalog-card-desc">' + escapeHtml(model.description) + '</div>' +
    quantRow +
    precisionRow +
    '<div class="catalog-card-meta">' +
      '<div class="catalog-card-specs">' +
        '<span class="catalog-spec">' + escapeHtml(vramText) + '</span>' +
        '<span class="catalog-spec catalog-size-label">' + model.size_gb + ' GB</span>' +
        (model.speed_note ? '<span class="catalog-spec">' + escapeHtml(model.speed_note) + '</span>' : '') +
      '</div>' +
      '<div class="catalog-card-actions">' +
        '<button class="' + btnClass + '"' + btnDisabled + ' data-repo="' + escapeHtml(model.repo_id) + '">' + btnText + '</button>' +
        deleteBtn +
      '</div>' +
    '</div>';

  // Wire delete button for installed models
  if (model.installed) {
    const delBtn = card.querySelector('.catalog-delete-btn');
    if (delBtn) {
      delBtn.addEventListener('click', async () => {
        const modelName = delBtn.dataset.name;
        if (!confirm('Delete "' + modelName + '" from disk? This cannot be undone.')) return;
        delBtn.disabled = true;
        try {
          const resp = await fetch('/api/image/models/' + encodeURIComponent(modelName), { method: 'DELETE' });
          if (resp.ok) {
            showToast('"' + modelName + '" deleted', 'success');
            refreshImageModels();
            refreshImageCatalog();
          } else {
            const err = await resp.json().catch(() => ({}));
            showToast('Delete failed: ' + (err.detail || err.error || resp.status), 'error');
            delBtn.disabled = false;
          }
        } catch (e) {
          showToast('Delete failed: ' + e.message, 'error');
          delBtn.disabled = false;
        }
      });
    }
  }

  if (!model.installed) {
    const btn = card.querySelector('.catalog-download-btn');

    if (isGguf) {
      const select = card.querySelector('.catalog-quant-select');
      const sizeLabel = card.querySelector('.catalog-quant-size');
      const cardSizeLabel = card.querySelector('.catalog-size-label');
      let variantsLoaded = false;
      let variants = [];

      // Lazy-load variants on first interaction
      async function loadVariants() {
        if (variantsLoaded) return;
        variantsLoaded = true;
        select.innerHTML = '<option value="" disabled selected>Loading…</option>';
        try {
          const resp = await fetch('/api/image/models/variants?repo_id=' + encodeURIComponent(model.repo_id));
          if (!resp.ok) throw new Error('Failed to fetch');
          const data = await resp.json();
          variants = data.variants || [];
          select.innerHTML = '';
          if (!variants.length) {
            select.innerHTML = '<option value="" disabled selected>No variants found</option>';
            return;
          }
          // Find the default (Q4_K_M) or first variant
          const defaultQuant = model.allow_patterns[0].replace(/\*/g, '');
          for (const v of variants) {
            const opt = document.createElement('option');
            opt.value = v.pattern;
            opt.textContent = v.quant + (v.size_gb ? ' (' + v.size_gb + ' GB)' : '');
            opt.dataset.sizeGb = v.size_gb;
            if (v.quant === defaultQuant) opt.selected = true;
            select.appendChild(opt);
          }
          // If default was selected, update size display
          if (select.value) {
            const sel = variants.find(v => v.pattern === select.value);
            if (sel) {
              sizeLabel.textContent = sel.size_gb + ' GB';
              cardSizeLabel.textContent = sel.size_gb + ' GB';
            }
          }
        } catch {
          select.innerHTML = '<option value="" disabled selected>Failed to load</option>';
        }
      }

      select.addEventListener('focus', loadVariants);
      select.addEventListener('mousedown', loadVariants);

      select.addEventListener('change', () => {
        const sel = variants.find(v => v.pattern === select.value);
        if (sel) {
          sizeLabel.textContent = sel.size_gb + ' GB';
          cardSizeLabel.textContent = sel.size_gb + ' GB';
        }
      });

      btn.addEventListener('click', () => {
        const pattern = select.value;
        if (!pattern) {
          loadVariants();
          select.focus();
          return;
        }
        handleCatalogDownload(model.repo_id, btn, [pattern]);
      });
    } else if (hasPrecision) {
      const precisionSelect = card.querySelector('.catalog-precision-select');
      const cardSizeLabel = card.querySelector('.catalog-size-label');

      // Update displayed size when precision changes
      precisionSelect.addEventListener('change', () => {
        const opt = precisionSelect.selectedOptions[0];
        if (opt && opt.dataset.sizeGb && cardSizeLabel) {
          cardSizeLabel.textContent = opt.dataset.sizeGb + ' GB';
        }
      });
      // Set initial size from selected variant
      if (precisionSelect.selectedOptions[0] && precisionSelect.selectedOptions[0].dataset.sizeGb && cardSizeLabel) {
        cardSizeLabel.textContent = precisionSelect.selectedOptions[0].dataset.sizeGb + ' GB';
      }

      btn.addEventListener('click', () => {
        const variant = precisionSelect.value;
        handleCatalogDownload(model.repo_id, btn, null, variant);
      });
    } else {
      btn.addEventListener('click', () => handleCatalogDownload(model.repo_id, btn, model.allow_patterns || null));
    }
  }

  return card;
}

async function handleCatalogDownload(repoId, btn, allowPatterns, variant) {
  if (btn.disabled) return;
  btn.disabled = true;
  btn.classList.add('downloading');
  const originalText = btn.textContent;
  btn.textContent = 'Starting...';

  function resetBtn(text, delay) {
    btn.textContent = text;
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
      btn.classList.remove('downloading');
    }, delay);
    delete _activeCatalogDownloads[repoId];
  }

  try {
    const body = { source: repoId };
    if (allowPatterns) body.allow_patterns = allowPatterns;
    if (variant) body.variant = variant;
    const resp = await fetch('/api/image/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      resetBtn('Error: ' + (errBody.detail || errBody.error || resp.status), 6000);
      return;
    }

    const task = await resp.json();
    btn.textContent = 'Preparing...';
    _activeCatalogDownloads[repoId] = { taskId: task.task_id, btn };

    const interval = pollPullTask(task.task_id,
      (data) => {
        if (data.percent !== undefined) {
          const pct = Math.round(data.percent);
          let detail = pct + '%';
          if (data.downloaded && data.total) detail += ' ' + formatBytes(data.downloaded) + '/' + formatBytes(data.total);
          btn.textContent = detail;
        }
      },
      () => {
        btn.textContent = 'Installed';
        btn.classList.remove('downloading');
        btn.classList.add('installed-btn');
        delete _activeCatalogDownloads[repoId];
        refreshImageModels();
        refreshImageCatalog();
      },
      (errMsg) => resetBtn('Error: ' + errMsg.substring(0, 40), 6000)
    );
    _activeCatalogDownloads[repoId].interval = interval;
  } catch (err) {
    resetBtn('Failed: ' + err.message.substring(0, 40), 6000);
  }
}

async function reconnectCatalogDownloads() {
  try {
    const resp = await fetch('/api/image/models/pull');
    if (!resp.ok) return;
    const tasks = await resp.json();
    if (!tasks || !tasks.length) return;

    for (const t of tasks) {
      if (t.status !== 'running') continue;
      const source = t.source || '';
      if (!source || _activeCatalogDownloads[source]) continue;

      const catalogBtn = document.querySelector('.catalog-download-btn[data-repo="' + source + '"]');
      if (!catalogBtn) continue;

      catalogBtn.disabled = true;
      catalogBtn.classList.add('downloading');
      catalogBtn.textContent = t.percent ? Math.round(t.percent) + '%' : 'Downloading...';
      _activeCatalogDownloads[source] = { taskId: t.task_id, btn: catalogBtn };

      const interval = pollPullTask(t.task_id,
        (data) => {
          if (data.percent !== undefined) {
            const pct = Math.round(data.percent);
            let detail = pct + '%';
            if (data.downloaded && data.total) detail += ' ' + formatBytes(data.downloaded) + '/' + formatBytes(data.total);
            catalogBtn.textContent = detail;
          }
        },
        () => {
          catalogBtn.textContent = 'Installed';
          catalogBtn.classList.remove('downloading');
          catalogBtn.classList.add('installed-btn');
          delete _activeCatalogDownloads[source];
          refreshImageModels();
          refreshImageCatalog();
        },
        () => {
          catalogBtn.textContent = 'Download';
          catalogBtn.disabled = false;
          catalogBtn.classList.remove('downloading');
          delete _activeCatalogDownloads[source];
        }
      );
      _activeCatalogDownloads[source].interval = interval;
    }
  } catch { /* not critical */ }
}

// ---------------------------------------------------------------------------
// Image Library Modal
// ---------------------------------------------------------------------------

let libraryModalEl = null;

function createLibraryModal() {
  if (libraryModalEl) return;

  libraryModalEl = document.createElement('div');
  libraryModalEl.className = 'modal-overlay img-lib-modal hidden';
  libraryModalEl.id = 'img-library-modal';
  libraryModalEl.innerHTML = `
    <div class="modal" style="width:min(900px,95vw);max-height:90dvh">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:var(--space-md)">
          <span class="modal-title">Image Library</span>
          <div class="img-lib-tabs">
            <button class="img-lib-tab active" id="img-lib-tab-gallery" data-tab="gallery">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
              Gallery
            </button>
            <button class="img-lib-tab" id="img-lib-tab-private" data-tab="private">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              Private
            </button>
            <button class="img-lib-tab" id="img-lib-tab-backgrounds" data-tab="backgrounds">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M2 16l5-5 4 4 3-3 6 6"/><circle cx="15.5" cy="7.5" r="1.5"/></svg>
              Backgrounds
            </button>
          </div>
        </div>
        <div class="img-lib-toolbar">
          <span id="img-lib-count" class="img-lib-count"></span>
          <button class="icon-btn small" id="img-lib-view-grid" title="Grid view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          </button>
          <button class="icon-btn small" id="img-lib-view-list" title="List view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
          </button>
          <button class="btn btn-sm" id="img-lib-select-btn">Select</button>
          <button class="btn btn-sm hidden" id="img-lib-bulk-delete-btn" style="color:var(--error)">Delete Selected</button>
          <button class="btn btn-sm hidden" id="img-lib-bulk-privacy-btn">Move to Private</button>
          <button class="icon-btn small" id="img-library-close" title="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>

      <!-- Filters -->
      <div class="img-lib-filters">
        <input type="text" class="field-input" id="img-lib-search" placeholder="Search prompts...">
        <select class="field-input" id="img-lib-model-filter"><option value="">All Models</option></select>
        <select class="field-input" id="img-lib-preset-filter"><option value="">All Presets</option></select>
        <select class="field-input" id="img-lib-sort">
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
        </select>
        <select class="field-input" id="img-lib-origin-filter" title="Who created these">
          <option value="">Anyone</option>
          <option value="companion">Companion-created</option>
        </select>
      </div>

      <!-- Content -->
      <div class="img-lib-body">
        <div id="img-lib-grid" class="img-lib-grid"></div>
        <div id="img-lib-empty" class="img-empty-state" style="display:none">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="40" height="40"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          <p>No images yet. Generate something!</p>
        </div>
        <div id="img-lib-load-more" class="img-lib-load-more hidden">
          <button class="btn" id="img-lib-load-more-btn">Load More</button>
        </div>

        <!-- Detail Panel -->
        <div id="img-lib-detail" class="img-lib-detail hidden">
          <div class="img-lib-detail-header">
            <span class="img-lib-detail-title">Image Details</span>
            <button class="icon-btn small" id="img-lib-detail-close" title="Close">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <img id="img-lib-detail-img" class="img-lib-detail-preview" alt="">
          <div id="img-lib-detail-meta" class="img-lib-meta-grid"></div>
          <div class="img-lib-detail-actions">
            <button class="btn btn-sm" id="img-lib-use-prompt">Reuse Prompt</button>
            <button class="btn btn-sm" id="img-lib-cast" title="Cast to a paired TV">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12" style="vertical-align:-2px;margin-right:4px"><path d="M2 16.1A5 5 0 0 1 5.9 20"/><path d="M2 12.05A9 9 0 0 1 9.95 20"/><path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/><line x1="2" y1="20" x2="2.01" y2="20"/></svg>Cast to TV
            </button>
            <button class="btn btn-sm" id="img-lib-download">Download</button>
            <button class="btn btn-sm" id="img-lib-copy-seed">Copy Seed</button>
            <button class="btn btn-sm" id="img-lib-toggle-background">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12" style="vertical-align:-2px;margin-right:2px"><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M2 16l5-5 4 4 3-3 6 6"/></svg>
              <span id="img-lib-background-label">Add to Backgrounds</span>
            </button>
            <button class="btn btn-sm" id="img-lib-toggle-private">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12" style="vertical-align:-2px;margin-right:2px"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              <span id="img-lib-privacy-label">Move to Private</span>
            </button>
            <button class="btn btn-sm" id="img-lib-delete" style="color:var(--error)">Delete</button>
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(libraryModalEl);
  bindLibraryEvents();
}

function bindLibraryEvents() {
  const modal = libraryModalEl;
  modal.querySelector('#img-library-close').addEventListener('click', closeImageLibrary);
  modal.addEventListener('click', e => { if (e.target === modal) closeImageLibrary(); });

  modal.querySelector('#img-lib-search').addEventListener('input', () => {
    clearTimeout(imgLibState.searchTimer);
    imgLibState.searchTimer = setTimeout(imgLibResetAndLoad, 350);
  });

  modal.querySelector('#img-lib-model-filter').addEventListener('change', imgLibResetAndLoad);
  modal.querySelector('#img-lib-preset-filter').addEventListener('change', imgLibResetAndLoad);
  modal.querySelector('#img-lib-sort').addEventListener('change', imgLibResetAndLoad);
  modal.querySelector('#img-lib-origin-filter').addEventListener('change', imgLibResetAndLoad);

  modal.querySelector('#img-lib-view-grid').addEventListener('click', () => setLibraryView('grid'));
  modal.querySelector('#img-lib-view-list').addEventListener('click', () => setLibraryView('list'));

  modal.querySelector('#img-lib-load-more-btn').addEventListener('click', imgLibLoadMore);
  modal.querySelector('#img-lib-select-btn').addEventListener('click', toggleBulkMode);
  modal.querySelector('#img-lib-bulk-delete-btn').addEventListener('click', handleBulkDelete);
  modal.querySelector('#img-lib-bulk-privacy-btn').addEventListener('click', handleBulkPrivacyToggle);
  modal.querySelector('#img-lib-detail-close').addEventListener('click', closeLibraryDetail);
  modal.querySelector('#img-lib-use-prompt').addEventListener('click', handleLibraryReusePrompt);
  modal.querySelector('#img-lib-download').addEventListener('click', handleLibraryDownload);
  modal.querySelector('#img-lib-cast').addEventListener('click', (e) => handleLibraryCast(e.currentTarget));
  modal.querySelector('#img-lib-copy-seed').addEventListener('click', handleLibraryCopySeed);
  modal.querySelector('#img-lib-toggle-private').addEventListener('click', handleDetailPrivacyToggle);
  modal.querySelector('#img-lib-delete').addEventListener('click', () => handleLibraryDelete());

  // Tab switching (gallery / private / backgrounds).
  // Save the leaving tab's loaded entries + scroll position, then try to
  // restore the new tab from its cache. Falls through to a fresh fetch
  // when no compatible cache exists (first visit, or filter changed).
  modal.querySelectorAll('.img-lib-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const tabName = tab.dataset.tab;
      if (tabName === imgLibState.activeTab) return;
      imgLibSaveTabCache(imgLibState.activeTab);
      imgLibState.activeTab = tabName;
      imgLibState.privateMode = tabName === 'private';
      imgLibState.backgroundMode = tabName === 'backgrounds';
      modal.querySelectorAll('.img-lib-tab').forEach(t => t.classList.toggle('active', t === tab));
      updateBulkPrivacyLabel();
      if (!imgLibRestoreTabCache(tabName)) {
        imgLibResetAndLoad();
      }
    });
  });

  modal.querySelector('#img-lib-toggle-background').addEventListener('click', handleDetailBackgroundToggle);

  // Grid delegation
  modal.querySelector('#img-lib-grid').addEventListener('click', e => {
    const actionBtn = e.target.closest('.img-lib-card-action');
    if (actionBtn) {
      e.stopPropagation();
      const action = actionBtn.dataset.action;
      const id = actionBtn.closest('[data-image-id]').dataset.imageId;
      const entry = imgLibState.entries.find(en => en.image_id === id);
      if (!entry) return;
      if (action === 'download') { imgLibState.currentEntry = entry; handleLibraryDownload(); }
      else if (action === 'delete') handleLibraryDelete(id);
      else if (action === 'privacy') handleCardPrivacyToggle(entry);
      return;
    }

    const checkEl = e.target.closest('.img-lib-card-check, .img-lib-row-check');
    if (checkEl && imgLibState.bulkMode) {
      e.stopPropagation();
      const card = checkEl.closest('[data-image-id]');
      const imgId = card.dataset.imageId;
      if (imgLibState.selectedIds.has(imgId)) {
        imgLibState.selectedIds.delete(imgId);
        card.classList.remove('selected');
        checkEl.innerHTML = '';
      } else {
        imgLibState.selectedIds.add(imgId);
        card.classList.add('selected');
        checkEl.innerHTML = (window.icons && window.icons.checkSmall) || '\u2713';
      }
      updateBulkDeleteLabel();
      return;
    }

    const cardEl = e.target.closest('[data-image-id]');
    if (cardEl) {
      if (imgLibState.bulkMode) {
        const chk = cardEl.querySelector('.img-lib-card-check, .img-lib-row-check');
        if (chk) chk.click();
        return;
      }
      const entryId = cardEl.dataset.imageId;
      const entry = imgLibState.entries.find(en => en.image_id === entryId);
      if (entry) imgLibSelectEntry(entry);
    }
  });

  document.addEventListener('keydown', e => {
    if (modal.classList.contains('hidden')) return;

    // Lightbox navigation
    const lb = document.getElementById('img-lib-lightbox');
    if (lb && lb.classList.contains('visible')) {
      if (e.key === 'Escape') { _closeLibraryLightbox(); e.stopPropagation(); return; }
      if (e.key === 'ArrowLeft') { _libLightboxNav(-1); return; }
      if (e.key === 'ArrowRight') { _libLightboxNav(1); return; }
      return;
    }

    if (e.key === 'Escape') closeImageLibrary();
  });
}

function openImageLibrary() {
  createLibraryModal();
  libraryModalEl.classList.remove('hidden');
  imgLibState.bulkMode = false;
  imgLibState.selectedIds.clear();
  imgLibState.currentEntry = null;
  imgLibState.entries = [];
  imgLibState.privateMode = false;
  imgLibState.backgroundMode = false;
  imgLibState.activeTab = 'gallery';
  imgLibState.tabCache = { gallery: null, private: null, backgrounds: null };

  // Reset tab state
  libraryModalEl.querySelectorAll('.img-lib-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === 'gallery');
  });

  const grid = libraryModalEl.querySelector('#img-lib-grid');
  grid.classList.remove('bulk-mode');
  libraryModalEl.querySelector('#img-lib-bulk-delete-btn').classList.add('hidden');
  libraryModalEl.querySelector('#img-lib-select-btn').textContent = 'Select';
  closeLibraryDetail();

  // Reset filters
  libraryModalEl.querySelector('#img-lib-model-filter').innerHTML = '<option value="">All Models</option>';
  libraryModalEl.querySelector('#img-lib-preset-filter').innerHTML = '<option value="">All Presets</option>';

  imgLibResetAndLoad();
  imgLibSetupAutoLoad();
}

function closeImageLibrary() {
  if (libraryModalEl) libraryModalEl.classList.add('hidden');
  clearTimeout(imgLibState.searchTimer);
  if (imgLibState.io) { imgLibState.io.disconnect(); imgLibState.io = null; }
}

function imgLibFilterSignature() {
  if (!libraryModalEl) return '';
  const q = (libraryModalEl.querySelector('#img-lib-search') || {}).value || '';
  const model = (libraryModalEl.querySelector('#img-lib-model-filter') || {}).value || '';
  const preset = (libraryModalEl.querySelector('#img-lib-preset-filter') || {}).value || '';
  const sort = (libraryModalEl.querySelector('#img-lib-sort') || {}).value || 'newest';
  return q + '\u0001' + model + '\u0001' + preset + '\u0001' + sort;
}

function imgLibSaveTabCache(tab) {
  if (!libraryModalEl || !tab) return;
  const body = libraryModalEl.querySelector('.img-lib-body');
  imgLibState.tabCache[tab] = {
    entries: imgLibState.entries.slice(),
    offset: imgLibState.offset,
    total: imgLibState.total,
    scrollTop: body ? body.scrollTop : 0,
    filterSig: imgLibFilterSignature(),
  };
}

function imgLibRestoreTabCache(tab) {
  const cache = imgLibState.tabCache[tab];
  if (!cache || cache.filterSig !== imgLibFilterSignature()) return false;
  if (!cache.entries.length) return false;
  imgLibState.entries = cache.entries.slice();
  imgLibState.offset = cache.offset;
  imgLibState.total = cache.total;
  const grid = libraryModalEl.querySelector('#img-lib-grid');
  grid.innerHTML = '';
  imgLibRenderEntries(cache.entries, true);
  libraryModalEl.querySelector('#img-lib-count').textContent =
    imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
  const empty = libraryModalEl.querySelector('#img-lib-empty');
  empty.style.display = imgLibState.total === 0 ? '' : 'none';
  const loadMore = libraryModalEl.querySelector('#img-lib-load-more');
  if (loadMore) loadMore.classList.toggle('hidden', imgLibState.entries.length >= imgLibState.total);
  updateFilterDropdowns();
  // Scroll restore has to wait for layout so flex-growth + image intrinsic
  // sizing settle before we assign scrollTop, otherwise the container is
  // still too short and the clamp stomps our value.
  requestAnimationFrame(() => {
    const body = libraryModalEl.querySelector('.img-lib-body');
    if (body) body.scrollTop = cache.scrollTop;
  });
  return true;
}

// Invalidate caches for tabs other than the active one so an optimistic
// mutation (delete, privacy toggle, background toggle) can't leave a
// stale snapshot behind. The active tab stays in sync via direct edits
// to imgLibState.entries in each mutation handler.
function imgLibInvalidateOtherCaches() {
  const cur = imgLibState.activeTab;
  Object.keys(imgLibState.tabCache).forEach(k => {
    if (k !== cur) imgLibState.tabCache[k] = null;
  });
}

function imgLibSetupAutoLoad() {
  if (imgLibState.io) { imgLibState.io.disconnect(); imgLibState.io = null; }
  if (!libraryModalEl || typeof IntersectionObserver === 'undefined') return;
  const body = libraryModalEl.querySelector('.img-lib-body');
  const sentinel = libraryModalEl.querySelector('#img-lib-load-more');
  if (!body || !sentinel) return;
  imgLibState.io = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      if (imgLibState.loading) continue;
      if (imgLibState.entries.length >= imgLibState.total) continue;
      imgLibLoadMore();
    }
  }, { root: body, rootMargin: '400px', threshold: 0 });
  imgLibState.io.observe(sentinel);
}

function imgLibResetAndLoad() {
  imgLibState.offset = 0;
  imgLibState.entries = [];
  const grid = libraryModalEl.querySelector('#img-lib-grid');
  grid.innerHTML = '';
  for (let i = 0; i < 12; i++) {
    const skel = document.createElement('div');
    skel.className = 'img-lib-skeleton';
    grid.appendChild(skel);
  }
  libraryModalEl.querySelector('#img-lib-empty').style.display = 'none';
  const body = libraryModalEl.querySelector('.img-lib-body');
  if (body) body.scrollTop = 0;
  imgLibFetchPage(false);
}

async function imgLibFetchPage(append) {
  if (imgLibState.loading) return;
  imgLibState.loading = true;
  const moreBtn = libraryModalEl && libraryModalEl.querySelector('#img-lib-load-more-btn');
  if (moreBtn && append) {
    moreBtn.disabled = true;
    moreBtn.textContent = 'Loading\u2026';
  }

  const q = (libraryModalEl.querySelector('#img-lib-search') || {}).value || '';
  const model = (libraryModalEl.querySelector('#img-lib-model-filter') || {}).value || '';
  const preset = (libraryModalEl.querySelector('#img-lib-preset-filter') || {}).value || '';
  const sort = (libraryModalEl.querySelector('#img-lib-sort') || {}).value || 'newest';
  const limit = 48;

  const privateParam = imgLibState.privateMode ? 'true' : 'false';
  const backgroundParam = imgLibState.backgroundMode ? 'true' : '';
  // Provenance filter — companion-created images live in this same
  // gallery; the chip narrows to hers.
  const origin = (libraryModalEl.querySelector('#img-lib-origin-filter') || {}).value || '';
  const url = '/api/image/history?limit=' + limit +
    '&offset=' + imgLibState.offset +
    '&q=' + encodeURIComponent(q) +
    '&model=' + encodeURIComponent(model) +
    '&preset=' + encodeURIComponent(preset) +
    '&sort=' + encodeURIComponent(sort) +
    '&private=' + privateParam +
    '&background=' + backgroundParam +
    '&origin=' + encodeURIComponent(origin);

  try {
    const resp = await fetch(url);
    if (!resp.ok) { imgLibState.loading = false; return; }
    const data = await resp.json();
    const entries = data.entries || [];
    imgLibState.total = data.total || 0;

    entries.forEach(e => { e.url = e.url || '/api/image/' + e.image_id; });

    if (!append) {
      const grid = libraryModalEl.querySelector('#img-lib-grid');
      grid.innerHTML = '';
      imgLibState.entries = entries;
    } else {
      imgLibState.entries = imgLibState.entries.concat(entries);
    }

    imgLibRenderEntries(entries, append);
    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');

    const loaded = imgLibState.entries.length;
    const loadMore = libraryModalEl.querySelector('#img-lib-load-more');
    loadMore.classList.toggle('hidden', loaded >= imgLibState.total);

    const empty = libraryModalEl.querySelector('#img-lib-empty');
    empty.style.display = imgLibState.total === 0 ? '' : 'none';
    if (imgLibState.total === 0) {
      const emptyP = empty.querySelector('p');
      if (emptyP) {
        emptyP.textContent = imgLibState.backgroundMode
          ? 'No backgrounds yet. Use "Add to Backgrounds" on any image.'
          : 'No images yet. Generate something!';
      }
    }

    updateFilterDropdowns();
  } catch { /* silently fail */ }
  imgLibState.loading = false;
  if (moreBtn) {
    moreBtn.disabled = false;
    moreBtn.textContent = 'Load More';
  }
}

function imgLibRenderEntries(entries, append) {
  const grid = libraryModalEl.querySelector('#img-lib-grid');
  if (!append) grid.innerHTML = '';

  for (const entry of entries) {
    if (imgLibState.view === 'list') grid.appendChild(imgLibBuildRow(entry));
    else grid.appendChild(imgLibBuildCard(entry));
  }
}

function imgLibBuildCard(entry) {
  const card = document.createElement('div');
  card.className = 'img-lib-card';
  card.dataset.imageId = entry.image_id;
  if (imgLibState.selectedIds.has(entry.image_id)) card.classList.add('selected');

  const img = document.createElement('img');
  // Cards render a 300 px thumb, not the full-res PNG — a 4x-upscaled
  // image is often 20+ MB, and we were transferring + decoding that just
  // to paint a ~150 px tile. The unified thumb endpoint caches WebPs at
  // ~15-30 KB each. Falls back to the full URL if the thumb 404s (stale
  // entry, thumb service unavailable, etc.) so pre-existing images keep
  // rendering during rollout.
  const thumbUrl = '/api/files/thumb/by-source/images/' + entry.image_id + '?size=300';
  img.src = thumbUrl;
  img.alt = entry.prompt || '';
  img.loading = 'lazy';
  img.decoding = 'async';
  let retries = 0;
  let fellBackToFull = false;
  img.onerror = () => {
    if (!fellBackToFull) {
      fellBackToFull = true;
      img.src = entry.url;
      return;
    }
    if (retries < 2) {
      retries++;
      setTimeout(() => {
        img.src = entry.url + (entry.url.includes('?') ? '&' : '?') + '_r=' + retries;
      }, retries * 800);
    }
  };
  card.appendChild(img);

  const overlay = document.createElement('div');
  overlay.className = 'img-lib-card-overlay';
  const promptEl = document.createElement('div');
  promptEl.className = 'img-lib-card-prompt';
  promptEl.textContent = entry.prompt || '';
  overlay.appendChild(promptEl);

  const actions = document.createElement('div');
  actions.className = 'img-lib-card-actions';
  const _ic = window.icons || {};
  const privacyIcon = entry.is_private ? (_ic.globe || 'Pub') : (_ic.lock || 'Priv');
  const dlIcon = _ic.download || 'DL';
  const delIcon = _ic.trash || 'Del';
  actions.innerHTML = '<button class="img-lib-card-action" data-action="privacy" title="' + (entry.is_private ? 'Make Public' : 'Move to Private') + '">' + privacyIcon + '</button><button class="img-lib-card-action" data-action="download" title="Download">' + dlIcon + '</button><button class="img-lib-card-action" data-action="delete" title="Delete">' + delIcon + '</button>';
  overlay.appendChild(actions);
  card.appendChild(overlay);

  // Private badge
  if (entry.is_private && !imgLibState.privateMode) {
    const badge = document.createElement('div');
    badge.className = 'img-lib-private-badge';
    badge.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="10" height="10"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>';
    card.appendChild(badge);
  }

  const check = document.createElement('div');
  check.className = 'img-lib-card-check';
  if (imgLibState.selectedIds.has(entry.image_id)) check.innerHTML = (window.icons && window.icons.checkSmall) || '\u2713';
  card.appendChild(check);

  return card;
}

function imgLibBuildRow(entry) {
  const row = document.createElement('div');
  row.className = 'img-lib-row';
  row.dataset.imageId = entry.image_id;
  if (imgLibState.selectedIds.has(entry.image_id)) row.classList.add('selected');

  const thumb = document.createElement('img');
  thumb.className = 'img-lib-row-thumb';
  // List rows render a tiny thumb — size=150 is plenty. Fall through to
  // full res if the thumb endpoint 404s for any reason.
  const thumbUrl = '/api/files/thumb/by-source/images/' + entry.image_id + '?size=150';
  thumb.src = thumbUrl;
  thumb.alt = entry.prompt || '';
  thumb.loading = 'lazy';
  thumb.decoding = 'async';
  let thumbRetries = 0;
  let thumbFellBack = false;
  thumb.onerror = () => {
    if (!thumbFellBack) {
      thumbFellBack = true;
      thumb.src = entry.url;
      return;
    }
    if (thumbRetries < 2) {
      thumbRetries++;
      setTimeout(() => {
        thumb.src = entry.url + (entry.url.includes('?') ? '&' : '?') + '_r=' + thumbRetries;
      }, thumbRetries * 800);
    }
  };
  row.appendChild(thumb);

  const info = document.createElement('div');
  info.className = 'img-lib-row-info';
  const p = document.createElement('div');
  p.className = 'img-lib-row-prompt';
  p.textContent = entry.prompt || '';
  info.appendChild(p);

  const meta = document.createElement('div');
  meta.className = 'img-lib-row-meta';
  if (entry.model) meta.innerHTML += '<span class="img-lib-row-chip">' + escapeHtml(entry.model) + '</span>';
  meta.innerHTML += '<span class="img-lib-row-chip">' + entry.width + 'x' + entry.height + '</span>';
  if (entry.seed != null && entry.seed !== -1) meta.innerHTML += '<span class="img-lib-row-chip">seed: ' + entry.seed + '</span>';
  if (entry.is_private) meta.innerHTML += '<span class="img-lib-row-chip private-chip">Private</span>';
  info.appendChild(meta);
  row.appendChild(info);

  const actions = document.createElement('div');
  actions.className = 'img-lib-row-actions';
  const _icr = window.icons || {};
  const rowPrivIcon = entry.is_private ? (_icr.globe || 'Pub') : (_icr.lock || 'Priv');
  const rowDlIcon = _icr.download || 'DL';
  const rowDelIcon = _icr.trash || 'Del';
  actions.innerHTML = '<button class="img-lib-card-action" data-action="privacy" title="' + (entry.is_private ? 'Make Public' : 'Move to Private') + '">' + rowPrivIcon + '</button><button class="img-lib-card-action" data-action="download" title="Download">' + rowDlIcon + '</button><button class="img-lib-card-action" data-action="delete" title="Delete">' + rowDelIcon + '</button>';
  row.appendChild(actions);

  const check = document.createElement('div');
  check.className = 'img-lib-row-check';
  if (imgLibState.selectedIds.has(entry.image_id)) check.innerHTML = (window.icons && window.icons.checkSmall) || '\u2713';
  row.appendChild(check);

  return row;
}

function imgLibLoadMore() {
  imgLibState.offset += 48;
  imgLibFetchPage(true);
}

function imgLibSelectEntry(entry) {
  imgLibState.currentEntry = entry;
  _openLibraryLightbox(entry);
}

function _openLibraryLightbox(entry) {
  let overlay = document.getElementById('img-lib-lightbox');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'img-lib-lightbox';
    overlay.className = 'img-lib-lightbox';
    overlay.innerHTML =
      '<button class="img-lib-lb-nav img-lib-lb-prev" title="Previous">' +
        '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>' +
      '</button>' +
      '<button class="img-lib-lb-nav img-lib-lb-next" title="Next">' +
        '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"/></svg>' +
      '</button>' +
      '<button class="lightbox-close img-lib-lb-close" title="Close">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
      '</button>' +
      '<div class="img-lib-lb-content">' +
        '<img class="img-lib-lb-img" alt="">' +
        '<div class="img-lib-lb-meta"></div>' +
        '<div class="img-lib-lb-actions"></div>' +
      '</div>';
    document.body.appendChild(overlay);

    // Close on backdrop click
    overlay.addEventListener('click', e => {
      if (e.target === overlay) _closeLibraryLightbox();
    });
    overlay.querySelector('.img-lib-lb-close').addEventListener('click', _closeLibraryLightbox);
    overlay.querySelector('.img-lib-lb-prev').addEventListener('click', () => _libLightboxNav(-1));
    overlay.querySelector('.img-lib-lb-next').addEventListener('click', () => _libLightboxNav(1));
  }

  // Populate image
  const imgEl = overlay.querySelector('.img-lib-lb-img');
  imgEl.src = entry.url;

  // Populate meta
  const metaEl = overlay.querySelector('.img-lib-lb-meta');
  const parts = [];
  if (entry.prompt) parts.push(entry.prompt.substring(0, 200));
  if (entry.model) parts.push(entry.model);
  if (entry.seed != null && entry.seed !== -1) parts.push('Seed: ' + entry.seed);
  if (entry.width && entry.height) parts.push(entry.width + 'x' + entry.height);
  if (entry.steps) parts.push(entry.steps + ' steps');
  if (entry.cfg_scale) parts.push('CFG: ' + entry.cfg_scale);
  if (entry.preset) parts.push(entry.preset);
  metaEl.textContent = parts.join(' \u2022 ');

  // Populate action buttons
  const actionsEl = overlay.querySelector('.img-lib-lb-actions');
  const isPrivate = entry.is_private;
  const isBg = entry.is_background;
  actionsEl.innerHTML =
    '<button class="lightbox-action-btn" data-act="prompt">Reuse Prompt</button>' +
    '<button class="lightbox-action-btn" data-act="seed">Copy Seed</button>' +
    '<button class="lightbox-action-btn" data-act="download">Download</button>' +
    '<button class="lightbox-action-btn" data-act="background">' + (isBg ? 'Remove from Backgrounds' : 'Add to Backgrounds') + '</button>' +
    '<button class="lightbox-action-btn" data-act="privacy">' + (isPrivate ? 'Make Public' : 'Move to Private') + '</button>' +
    '<button class="lightbox-action-btn" data-act="delete" style="color:#ff5252">Delete</button>';

  // Wire action buttons
  actionsEl.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', () => {
      const act = btn.dataset.act;
      if (act === 'prompt') {
        imgLibState.currentEntry = entry;
        handleLibraryReusePrompt();
      } else if (act === 'seed') {
        if (entry.seed != null) copyToClipboard(String(entry.seed)).then((ok) => { if (ok) showToast('Seed copied', 'success'); });
      } else if (act === 'download') {
        imgLibState.currentEntry = entry;
        handleLibraryDownload();
      } else if (act === 'background') {
        handleDetailBackgroundToggle();
        // Update button text
        const newBg = !entry.is_background;
        entry.is_background = newBg;
        btn.textContent = newBg ? 'Remove from Backgrounds' : 'Add to Backgrounds';
      } else if (act === 'privacy') {
        handleDetailPrivacyToggle();
        const newPriv = !entry.is_private;
        entry.is_private = newPriv;
        btn.textContent = newPriv ? 'Make Public' : 'Move to Private';
      } else if (act === 'delete') {
        handleLibraryDelete(entry.image_id);
        _closeLibraryLightbox();
      }
    });
  });

  // Show/hide nav arrows based on position
  const idx = imgLibState.entries.findIndex(e => e.image_id === entry.image_id);
  overlay.querySelector('.img-lib-lb-prev').style.visibility = idx > 0 ? 'visible' : 'hidden';
  overlay.querySelector('.img-lib-lb-next').style.visibility = idx < imgLibState.entries.length - 1 ? 'visible' : 'hidden';

  overlay.classList.add('visible');
}

function _closeLibraryLightbox() {
  const overlay = document.getElementById('img-lib-lightbox');
  if (overlay) overlay.classList.remove('visible');
  imgLibState.currentEntry = null;
}

function _libLightboxNav(dir) {
  const entries = imgLibState.entries;
  if (!entries.length || !imgLibState.currentEntry) return;
  const idx = entries.findIndex(e => e.image_id === imgLibState.currentEntry.image_id);
  const newIdx = idx + dir;
  if (newIdx < 0 || newIdx >= entries.length) return;
  const entry = entries[newIdx];
  imgLibState.currentEntry = entry;
  _openLibraryLightbox(entry);
}

function closeLibraryDetail() {
  _closeLibraryLightbox();
}

function setLibraryView(mode) {
  imgLibState.view = mode;
  const grid = libraryModalEl.querySelector('#img-lib-grid');
  const gridBtn = libraryModalEl.querySelector('#img-lib-view-grid');
  const listBtn = libraryModalEl.querySelector('#img-lib-view-list');

  if (mode === 'list') {
    grid.classList.add('list-view');
    listBtn.classList.add('active');
    gridBtn.classList.remove('active');
  } else {
    grid.classList.remove('list-view');
    gridBtn.classList.add('active');
    listBtn.classList.remove('active');
  }
  imgLibRenderEntries(imgLibState.entries, false);
}

function toggleBulkMode() {
  imgLibState.bulkMode = !imgLibState.bulkMode;
  const grid = libraryModalEl.querySelector('#img-lib-grid');
  const selectBtn = libraryModalEl.querySelector('#img-lib-select-btn');
  const bulkDeleteBtn = libraryModalEl.querySelector('#img-lib-bulk-delete-btn');
  const bulkPrivacyBtn = libraryModalEl.querySelector('#img-lib-bulk-privacy-btn');

  if (imgLibState.bulkMode) {
    grid.classList.add('bulk-mode');
    selectBtn.textContent = 'Cancel';
    bulkDeleteBtn.classList.remove('hidden');
    bulkPrivacyBtn.classList.remove('hidden');
    updateBulkPrivacyLabel();
  } else {
    grid.classList.remove('bulk-mode');
    selectBtn.textContent = 'Select';
    bulkDeleteBtn.classList.add('hidden');
    bulkPrivacyBtn.classList.add('hidden');
    imgLibState.selectedIds.clear();
    grid.querySelectorAll('.selected').forEach(el => {
      el.classList.remove('selected');
      const chk = el.querySelector('.img-lib-card-check, .img-lib-row-check');
      if (chk) chk.innerHTML = '';
    });
  }
  updateBulkDeleteLabel();
}

function updateBulkDeleteLabel() {
  const btn = libraryModalEl.querySelector('#img-lib-bulk-delete-btn');
  const count = imgLibState.selectedIds.size;
  btn.textContent = count > 0 ? 'Delete Selected (' + count + ')' : 'Delete Selected';
}

function updateFilterDropdowns() {
  const modelSel = libraryModalEl.querySelector('#img-lib-model-filter');
  const presetSel = libraryModalEl.querySelector('#img-lib-preset-filter');
  const models = new Set();
  const presets = new Set();
  imgLibState.entries.forEach(e => {
    if (e.model) models.add(e.model);
    if (e.preset) presets.add(e.preset);
  });
  const currentModel = modelSel.value;
  const currentPreset = presetSel.value;
  modelSel.innerHTML = '<option value="">All Models</option>';
  models.forEach(m => { const opt = document.createElement('option'); opt.value = m; opt.textContent = m; modelSel.appendChild(opt); });
  presetSel.innerHTML = '<option value="">All Presets</option>';
  presets.forEach(p => { const opt = document.createElement('option'); opt.value = p; opt.textContent = p; presetSel.appendChild(opt); });
  modelSel.value = currentModel;
  presetSel.value = currentPreset;
}

async function handleLibraryDelete(imageId) {
  const id = imageId || (imgLibState.currentEntry && imgLibState.currentEntry.image_id);
  if (!id) return;
  if (!confirm('Delete this image permanently?')) return;

  try {
    const resp = await fetch('/api/image/' + id, { method: 'DELETE' });
    if (!resp.ok) { showToast('Failed to delete image', 'error'); return; }

    imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== id);
    imgLibState.total = Math.max(0, imgLibState.total - 1);
    imgLibState.selectedIds.delete(id);

    const card = libraryModalEl.querySelector('[data-image-id="' + id + '"]');
    if (card) card.remove();

    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');

    if (imgLibState.currentEntry && imgLibState.currentEntry.image_id === id) closeLibraryDetail();
    if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';

    imgLibInvalidateOtherCaches();
    refreshImageGallery();
  } catch { showToast('Error deleting image', 'error'); }
}

async function handleBulkDelete() {
  const ids = Array.from(imgLibState.selectedIds);
  if (!ids.length) return;
  if (!confirm('Delete ' + ids.length + ' image' + (ids.length > 1 ? 's' : '') + ' permanently?')) return;

  try {
    const resp = await fetch('/api/image/batch', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: ids }),
    });
    if (!resp.ok) { showToast('Failed to delete images', 'error'); return; }
    const result = await resp.json();
    const deleted = result.deleted || [];

    deleted.forEach(delId => {
      imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== delId);
      imgLibState.selectedIds.delete(delId);
      const card = libraryModalEl.querySelector('[data-image-id="' + delId + '"]');
      if (card) card.remove();
    });

    imgLibState.total = Math.max(0, imgLibState.total - deleted.length);
    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
    updateBulkDeleteLabel();

    if (imgLibState.currentEntry && deleted.includes(imgLibState.currentEntry.image_id)) closeLibraryDetail();
    if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';

    imgLibInvalidateOtherCaches();
    refreshImageGallery();
  } catch { showToast('Error deleting images', 'error'); }
}

// --- Privacy toggle helpers ---

async function togglePrivacy(imageIds, makePrivate) {
  try {
    const resp = await fetch('/api/image/privacy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds, is_private: makePrivate }),
    });
    if (!resp.ok) { showToast('Failed to update privacy', 'error'); return false; }
    return true;
  } catch { showToast('Error updating privacy', 'error'); return false; }
}

async function handleCardPrivacyToggle(entry) {
  const makePrivate = !entry.is_private;
  if (await togglePrivacy([entry.image_id], makePrivate)) {
    entry.is_private = makePrivate;
    // Remove from view if it no longer matches the current tab
    const card = libraryModalEl.querySelector('[data-image-id="' + entry.image_id + '"]');
    if (card) card.remove();
    imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== entry.image_id);
    imgLibState.total = Math.max(0, imgLibState.total - 1);
    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
    if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';
    showToast(makePrivate ? 'Moved to private' : 'Moved to gallery', 'success');
    imgLibInvalidateOtherCaches();
    refreshImageGallery();
  }
}

async function handleDetailPrivacyToggle() {
  const entry = imgLibState.currentEntry;
  if (!entry) return;
  const makePrivate = !entry.is_private;
  if (await togglePrivacy([entry.image_id], makePrivate)) {
    entry.is_private = makePrivate;
    // Update label
    const label = libraryModalEl.querySelector('#img-lib-privacy-label');
    if (label) label.textContent = makePrivate ? 'Make Public' : 'Move to Private';
    // Remove from grid since it switched sections
    const card = libraryModalEl.querySelector('[data-image-id="' + entry.image_id + '"]');
    if (card) card.remove();
    imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== entry.image_id);
    imgLibState.total = Math.max(0, imgLibState.total - 1);
    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
    closeLibraryDetail();
    if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';
    showToast(makePrivate ? 'Moved to private' : 'Moved to gallery', 'success');
    imgLibInvalidateOtherCaches();
    refreshImageGallery();
  }
}

async function handleBulkPrivacyToggle() {
  const ids = Array.from(imgLibState.selectedIds);
  if (!ids.length) return;
  const makePrivate = !imgLibState.privateMode;
  if (await togglePrivacy(ids, makePrivate)) {
    ids.forEach(id => {
      const card = libraryModalEl.querySelector('[data-image-id="' + id + '"]');
      if (card) card.remove();
      imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== id);
      imgLibState.selectedIds.delete(id);
    });
    imgLibState.total = Math.max(0, imgLibState.total - ids.length);
    libraryModalEl.querySelector('#img-lib-count').textContent =
      imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
    updateBulkDeleteLabel();
    if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';
    showToast(
      (makePrivate ? 'Moved ' : 'Published ') + ids.length + ' image' + (ids.length > 1 ? 's' : ''),
      'success',
    );
    imgLibInvalidateOtherCaches();
    refreshImageGallery();
  }
}

function updateBulkPrivacyLabel() {
  const btn = libraryModalEl?.querySelector('#img-lib-bulk-privacy-btn');
  if (btn) {
    btn.textContent = imgLibState.privateMode ? 'Make Public' : 'Move to Private';
  }
}

async function handleDetailBackgroundToggle() {
  const entry = imgLibState.currentEntry;
  if (!entry) return;
  const addToBg = !entry.is_background;
  try {
    const resp = await fetch('/api/image/backgrounds/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: [entry.image_id], is_background: addToBg }),
    });
    if (!resp.ok) { showToast('Failed to update background collection', 'error'); return; }
    entry.is_background = addToBg;
    const label = libraryModalEl.querySelector('#img-lib-background-label');
    if (label) label.textContent = addToBg ? 'Remove from Backgrounds' : 'Add to Backgrounds';
    // If viewing backgrounds tab and removing, pull it from the grid
    if (imgLibState.backgroundMode && !addToBg) {
      const card = libraryModalEl.querySelector('[data-image-id="' + entry.image_id + '"]');
      if (card) card.remove();
      imgLibState.entries = imgLibState.entries.filter(e => e.image_id !== entry.image_id);
      imgLibState.total = Math.max(0, imgLibState.total - 1);
      libraryModalEl.querySelector('#img-lib-count').textContent =
        imgLibState.total + ' image' + (imgLibState.total !== 1 ? 's' : '');
      closeLibraryDetail();
      if (imgLibState.total === 0) libraryModalEl.querySelector('#img-lib-empty').style.display = '';
    }
    showToast(addToBg ? 'Added to backgrounds' : 'Removed from backgrounds', 'success');
    imgLibInvalidateOtherCaches();
    // Notify rotation engine
    if (window._bgRotationRefresh) window._bgRotationRefresh();
  } catch {
    showToast('Failed to update background collection', 'error');
  }
}

function handleLibraryDownload() {
  if (!imgLibState.currentEntry) return;
  const a = document.createElement('a');
  a.href = imgLibState.currentEntry.url;
  a.download = imgLibState.currentEntry.image_id + '.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/**
 * Cast the currently-viewed image to a paired TV. Routes through the
 * shared cast picker so the picker UX matches every other surface.
 * Image URL is the standard /api/image/{id} endpoint which the cast-
 * receiver mounts as a media.image surface.
 */
async function handleLibraryCast(anchor) {
  const entry = imgLibState.currentEntry;
  if (!entry) return;
  const { openCastPicker } = await import('./cast-picker.js');
  openCastPicker({
    anchor: anchor || libraryModalEl?.querySelector('#img-lib-cast'),
    capability: 'display.image_show@1',
    content: {
      contentUrl: entry.url || `/api/image/${encodeURIComponent(entry.image_id)}`,
      title: (entry.prompt || '').slice(0, 80) || 'Generated image',
      contentKey: `image:${entry.image_id}`,
      fileId: entry.image_id,
      metadata: {
        model: entry.model || '',
        width: entry.width || 0,
        height: entry.height || 0,
        source: 'image-library',
      },
    },
  });
}

function handleLibraryCopySeed() {
  if (!imgLibState.currentEntry) return;
  copyToClipboard(String(imgLibState.currentEntry.seed)).then((ok) => {
    if (!ok) return;
    const btn = libraryModalEl.querySelector('#img-lib-copy-seed');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

function handleLibraryReusePrompt() {
  if (!imgLibState.currentEntry) return;
  const entry = imgLibState.currentEntry;
  const promptEl = $('img-prompt');
  const negEl = $('img-negative');
  const seedEl = $('img-seed');
  const widthEl = $('img-width');
  const heightEl = $('img-height');
  const stepsEl = $('img-steps');
  const cfgEl = $('img-cfg');

  if (promptEl) promptEl.value = entry.prompt || '';
  if (negEl) negEl.value = entry.negative_prompt || '';
  if (seedEl) seedEl.value = -1;
  if (widthEl && entry.width) widthEl.value = entry.width;
  if (heightEl && entry.height) heightEl.value = entry.height;
  if (stepsEl && entry.steps) stepsEl.value = entry.steps;
  if (cfgEl && entry.cfg_scale) cfgEl.value = entry.cfg_scale;

  renderResolutionPresets();
  closeImageLibrary();

  const panel = $('image-panel');
  if (panel && panel.classList.contains('hidden')) {
    app.closeInspector();
    if (window.innerWidth < 1024) app.closePanel();
    panel.classList.remove('hidden');
    imgSettings.panelOpen = true;
    saveImgSettings();
  }
}

// ---------------------------------------------------------------------------
// Lightbox Actions (from image panel lightbox)
// ---------------------------------------------------------------------------

function initLightboxActions() {
  const lightboxModal = $('image-lightbox-modal');
  const lightboxClose = $('lightbox-close');
  const lbCopyPrompt = $('lightbox-copy-prompt');
  const lbCopySeed = $('lightbox-copy-seed');
  const lbDownload = $('lightbox-download');
  const lbUsePrompt = $('lightbox-use-prompt');
  const lbDelete = $('lightbox-delete');
  const editImg2img = $('lightbox-edit-img2img');
  const editInpaint = $('lightbox-edit-inpaint');

  if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);
  if (lightboxModal) lightboxModal.addEventListener('click', e => { if (e.target === lightboxModal) closeLightbox(); });

  function sendToEdit(mode) {
    const imgEl = $('lightbox-img');
    if (!imgEl || !imgEl.src) return;
    closeLightbox();
    setImageMode(mode);
    app.closeInspector();
    if (window.innerWidth < 1024) app.closePanel();
    const panel = $('image-panel');
    if (panel) panel.classList.remove('hidden');
    loadSourceImageFromUrl(imgEl.src);
  }

  if (editImg2img) editImg2img.addEventListener('click', () => sendToEdit('img2img'));
  if (editInpaint) editInpaint.addEventListener('click', () => openLightboxInpaint());

  if (lbCopyPrompt) lbCopyPrompt.addEventListener('click', () => {
    if (!currentLightboxEntry || !currentLightboxEntry.prompt) return;
    copyToClipboard(currentLightboxEntry.prompt).then((ok) => { if (ok) showToast('Prompt copied', 'success'); });
  });

  if (lbCopySeed) lbCopySeed.addEventListener('click', () => {
    if (!currentLightboxEntry || currentLightboxEntry.seed == null) return;
    copyToClipboard(String(currentLightboxEntry.seed)).then((ok) => { if (ok) showToast('Seed copied: ' + currentLightboxEntry.seed, 'success'); });
  });

  if (lbDownload) lbDownload.addEventListener('click', () => {
    const imgEl = $('lightbox-img');
    if (!imgEl || !imgEl.src) return;
    const a = document.createElement('a');
    a.href = imgEl.src;
    a.download = (currentLightboxEntry?.image_id || 'image') + '.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  if (lbUsePrompt) lbUsePrompt.addEventListener('click', () => {
    if (!currentLightboxEntry) return;
    const e = currentLightboxEntry;
    const promptEl = $('img-prompt');
    if (promptEl && e.prompt) promptEl.value = e.prompt;
    const negEl = $('img-negative');
    if (negEl && e.negative_prompt) negEl.value = e.negative_prompt;
    const seedEl = $('img-seed');
    if (seedEl && e.seed != null) seedEl.value = e.seed;
    if (e.width) { const w = $('img-width'); if (w) w.value = e.width; }
    if (e.height) { const h = $('img-height'); if (h) h.value = e.height; }
    if (e.steps) { const s = $('img-steps'); if (s) s.value = e.steps; }
    if (e.cfg_scale) { const c = $('img-cfg'); if (c) c.value = e.cfg_scale; }
    saveImageFormToSettings();
    closeLightbox();
    app.closeInspector();
    if (window.innerWidth < 1024) app.closePanel();
    const panel = $('image-panel');
    if (panel) panel.classList.remove('hidden');
    showToast('Settings loaded from image', 'success');
  });

  if (lbDelete) lbDelete.addEventListener('click', async () => {
    if (!currentLightboxEntry || !currentLightboxEntry.image_id) return;
    if (!confirm('Delete this image?')) return;
    try {
      const resp = await fetch('/api/image/' + currentLightboxEntry.image_id, { method: 'DELETE' });
      if (resp.ok) { showToast('Image deleted', 'success'); closeLightbox(); refreshImageGallery(); }
      else showToast('Failed to delete image', 'error');
    } catch (err) { showToast('Delete failed: ' + err.message, 'error'); }
  });

  // Background toggle (add/remove from backgrounds collection)
  const lbToggleBg = $('lightbox-toggle-background');
  if (lbToggleBg) lbToggleBg.addEventListener('click', async () => {
    const entry = currentLightboxEntry;
    if (!entry) return;
    const addToBg = !entry.is_background;
    try {
      const resp = await fetch('/api/image/backgrounds/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_ids: [entry.image_id], is_background: addToBg }),
      });
      if (!resp.ok) { showToast('Failed to update backgrounds', 'error'); return; }
      entry.is_background = addToBg;
      lbToggleBg.textContent = addToBg ? 'Remove from Backgrounds' : 'Add to Backgrounds';
      showToast(addToBg ? 'Added to backgrounds' : 'Removed from backgrounds', 'success');
    } catch { showToast('Error updating backgrounds', 'error'); }
  });

  // Privacy toggle (public/private)
  const lbTogglePriv = $('lightbox-toggle-private');
  if (lbTogglePriv) lbTogglePriv.addEventListener('click', async () => {
    const entry = currentLightboxEntry;
    if (!entry) return;
    const makePrivate = !entry.is_private;
    if (await togglePrivacy([entry.image_id], makePrivate)) {
      entry.is_private = makePrivate;
      lbTogglePriv.textContent = makePrivate ? 'Make Public' : 'Move to Private';
      showToast(makePrivate ? 'Moved to private' : 'Moved to gallery', 'success');
      refreshImageGallery();
    }
  });

  // Upscale 4x
  const lbUpscale = $('lightbox-upscale');
  if (lbUpscale) lbUpscale.addEventListener('click', async () => {
    const entry = currentLightboxEntry;
    if (!entry || !entry.image_id) return;
    lbUpscale.disabled = true;
    lbUpscale.textContent = 'Upscaling...';
    try {
      const resp = await fetch(`/api/image/${entry.image_id}/upscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scale: 4 }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        showToast(err.detail || 'Upscale failed', 'error');
        return;
      }
      const data = await resp.json();
      refreshImageGallery();
      showToast(`Upscaled to ${data.width}x${data.height}`, 'success');
      openLightbox({
        image_id: data.image_id,
        prompt: entry.prompt || '',
        negative_prompt: entry.negative_prompt || '',
        seed: entry.seed || -1,
        width: data.width,
        height: data.height,
        steps: 0,
        cfg_scale: 0,
        model: entry.model || '',
      }, data.url);
    } catch (err) {
      showToast('Upscale failed: ' + err.message, 'error');
    } finally {
      lbUpscale.disabled = false;
      lbUpscale.textContent = 'Upscale 4x';
    }
  });

  // Remove Background
  const lbRemoveBg = $('lightbox-remove-bg');
  if (lbRemoveBg) lbRemoveBg.addEventListener('click', async () => {
    const entry = currentLightboxEntry;
    if (!entry || !entry.image_id) return;
    lbRemoveBg.disabled = true;
    lbRemoveBg.textContent = 'Removing BG...';
    try {
      const resp = await fetch(`/api/image/${entry.image_id}/remove-bg`, { method: 'POST' });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        showToast(err.detail || 'Background removal failed', 'error');
        return;
      }
      const data = await resp.json();
      refreshImageGallery();
      showToast('Background removed', 'success');
      openLightbox({
        image_id: data.image_id,
        prompt: entry.prompt || '',
        negative_prompt: entry.negative_prompt || '',
        seed: entry.seed || -1,
        width: data.width,
        height: data.height,
        steps: 0,
        cfg_scale: 0,
        model: entry.model || '',
      }, data.url);
    } catch (err) {
      showToast('Background removal failed: ' + err.message, 'error');
    } finally {
      lbRemoveBg.disabled = false;
      lbRemoveBg.textContent = 'Remove BG';
    }
  });
}

// ---------------------------------------------------------------------------
// Lightbox Inpaint Editor — thin shim over ui/scripts/mask-editor.js.
//
// Before this refactor the lightbox had its own mask editor with a clone
// of the sidebar's paint logic (~250 lines) and Generate worked by closing
// itself, opening the sidebar, copying its mask data over, and synthetically
// clicking the sidebar's Generate. Now we mount the extracted mask-editor
// module and POST directly to /api/image/inpaint — one fewer surface hop,
// and the same module powers Studio's image viewer.
// ---------------------------------------------------------------------------

async function openLightboxInpaint() {
  const container = $('lightbox-inpaint');
  const imgEl = $('lightbox-img');
  const meta = $('lightbox-meta');
  const actions = document.querySelector('.lightbox-actions');
  if (!container || !imgEl) return;

  imgEl.style.display = 'none';
  if (meta) meta.style.display = 'none';
  if (actions) actions.style.display = 'none';
  container.classList.remove('hidden');
  container.innerHTML = '';
  _lbInpaintActive = true;

  // The <img> in the lightbox is usually already-loaded (same-origin). If
  // not, wait so the module has naturalWidth/Height available before it
  // tries to size its canvases.
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error('Failed to load source image'));
    img.src = imgEl.src;
  }).catch(err => {
    showToast(`Inpaint: ${err.message}`, 'error');
    closeLightboxInpaint();
    throw err;
  });

  const { createMaskEditor } = await import('./mask-editor.js');
  _lbMaskEditor = createMaskEditor({
    container,
    sourceImg: img,
    variant: 'lightbox',
    showPromptStrip: true,
    initialPrompt: (currentLightboxEntry?.prompt || '').trim(),
    onCancel: closeLightboxInpaint,
    onGenerate: _lbInpaintGenerate,
    generateLabel: 'Generate',
  });
}

// Direct POST to /api/image/inpaint. No sidebar roundtrip — this is the
// whole point of the refactor. Reads the currently-selected sidebar model
// (when the sidebar has been initialized at least once) so the user gets
// their chosen inpaint-capable model; otherwise falls through to the
// server's default model.
async function _lbInpaintGenerate(payload) {
  if (!_lbMaskEditor) return;
  _lbMaskEditor.setBusy(true);

  // Source image: the sidebar's base64-encoded source is handy when the
  // lightbox image is the same one the sidebar just loaded; otherwise we
  // build one from the `<img>` element. Base64 is what `/api/image/inpaint`
  // accepts on the `source_image` field (plus image_id, but that's the
  // lightbox-bound gallery id — we'd have to resolve it either way).
  let sourceB64 = sourceImageBase64 || '';
  if (!sourceB64) {
    const imgEl = $('lightbox-img');
    if (imgEl) sourceB64 = await _imgElToBase64(imgEl);
  }
  if (!sourceB64) {
    showToast('Could not read source image', 'error');
    _lbMaskEditor.setBusy(false);
    return;
  }

  const modelEl = $('img-model-select');
  const body = {
    prompt: payload.prompt || '',
    negative_prompt: payload.negativePrompt || '',
    model: modelEl?.value || '',
    source_image: sourceB64,
    mask_image: payload.maskBase64,
    strength: payload.strength,
    inpaint_mode: payload.mode || 'default',
  };

  try {
    const resp = await fetch('/api/image/inpaint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || `Failed (${resp.status})`);
    showToast(`Inpainted (${data.width}×${data.height})`, 'success');
    closeLightboxInpaint();
    closeLightbox();
    refreshImageGallery();
    // Open the freshly inpainted image in the lightbox so the user sees it.
    openLightbox(
      {
        image_id: data.image_id,
        prompt: body.prompt,
        negative_prompt: body.negative_prompt,
        seed: data.seed ?? -1,
        width: data.width,
        height: data.height,
        steps: 0,
        cfg_scale: 0,
        model: body.model,
      },
      data.url,
    );
  } catch (err) {
    showToast(`Inpaint failed: ${err.message}`, 'error');
    _lbMaskEditor?.setBusy(false);
  }
}

// Rasterize an <img> to a base64 PNG (header-stripped). Used when the
// sidebar hasn't cached a base64 copy of the current lightbox image —
// e.g., the user opened a gallery item directly without going through the
// sidebar's load path first.
async function _imgElToBase64(imgEl) {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    canvas.getContext('2d').drawImage(imgEl, 0, 0);
    return canvas.toDataURL('image/png').split(',')[1];
  } catch {
    return '';
  }
}

function closeLightboxInpaint() {
  const container = $('lightbox-inpaint');
  const imgEl = $('lightbox-img');
  const meta = $('lightbox-meta');
  const actions = document.querySelector('.lightbox-actions');

  if (container) {
    container.classList.add('hidden');
    container.innerHTML = '';
  }
  if (imgEl) imgEl.style.display = '';
  if (meta) meta.style.display = '';
  if (actions) actions.style.display = '';
  _lbInpaintActive = false;
  if (_lbMaskEditor) {
    try { _lbMaskEditor.destroy(); } catch { /* already torn down */ }
    _lbMaskEditor = null;
  }
}

// ---------------------------------------------------------------------------
// Panel Init
// ---------------------------------------------------------------------------

function initImagePanel() {
  const toggleBtn = $('toggle-image-btn');
  const panel = $('image-panel');
  const closeBtn = $('close-image-btn');
  const generateBtn = $('img-generate-btn');
  const pullBtn = $('img-pull-btn');
  const openLibraryBtn = $('open-image-library-btn');

  if (!toggleBtn || !panel) return;

  restoreImageFormFromSettings();
  _pushActiveSettings();

  // Mode tabs
  document.querySelectorAll('.img-mode-tab').forEach(tab => {
    tab.addEventListener('click', () => setImageMode(tab.dataset.mode));
  });
  setImageMode('txt2img');

  // Source image upload
  const sourceDrop = $('img-source-drop');
  const sourceFile = $('img-source-file');
  if (sourceDrop) {
    sourceDrop.addEventListener('click', () => { if (sourceFile) sourceFile.click(); });
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
      sourceDrop.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(evt => {
      sourceDrop.addEventListener(evt, () => sourceDrop.classList.add('drag-over'));
    });
    ['dragleave', 'drop'].forEach(evt => {
      sourceDrop.addEventListener(evt, () => sourceDrop.classList.remove('drag-over'));
    });
    sourceDrop.addEventListener('drop', e => {
      if (e.dataTransfer.files.length > 0) loadSourceImage(e.dataTransfer.files[0]);
    });
  }
  if (sourceFile) {
    sourceFile.addEventListener('change', () => {
      if (sourceFile.files.length > 0) { loadSourceImage(sourceFile.files[0]); sourceFile.value = ''; }
    });
  }

  // Strength slider
  const strengthSlider = $('img-strength');
  const strengthVal = $('img-strength-value');
  if (strengthSlider) {
    strengthSlider.addEventListener('input', () => { if (strengthVal) strengthVal.textContent = strengthSlider.value; });
  }

  // Clear source image button
  const clearSourceBtn = $('img-source-clear');
  if (clearSourceBtn) clearSourceBtn.addEventListener('click', clearSourceImage);

  // Mask tools
  document.querySelectorAll('.img-mask-tool').forEach(btn => {
    btn.addEventListener('click', () => setMaskTool(btn.dataset.tool));
  });

  // Brush size slider
  const brushSizeEl = $('img-mask-brush-size');
  const brushValEl = $('img-mask-brush-value');
  if (brushSizeEl) {
    brushSizeEl.addEventListener('input', () => {
      maskEditor.brushSize = parseInt(brushSizeEl.value) || 30;
      if (brushValEl) brushValEl.textContent = maskEditor.brushSize;
    });
  }

  // Brush opacity slider
  const opacityEl = $('img-mask-opacity');
  const opacityValEl = $('img-mask-opacity-value');
  if (opacityEl) {
    opacityEl.addEventListener('input', () => {
      maskEditor.brushOpacity = parseInt(opacityEl.value) / 100;
      if (opacityValEl) opacityValEl.textContent = opacityEl.value + '%';
      maskEditorRender();
    });
  }

  // Undo / Redo buttons
  const undoBtn = $('img-mask-undo');
  const redoBtn = $('img-mask-redo');
  if (undoBtn) undoBtn.addEventListener('click', maskEditorUndo);
  if (redoBtn) redoBtn.addEventListener('click', maskEditorRedo);

  // Mask view toggle
  const viewToggle = $('img-mask-view-toggle');
  if (viewToggle) viewToggle.addEventListener('click', toggleMaskView);

  // Clear mask
  const clearMaskBtn = $('img-mask-clear');
  if (clearMaskBtn) clearMaskBtn.addEventListener('click', clearMask);

  // Invert mask
  const invertMaskBtn = $('img-mask-invert');
  if (invertMaskBtn) invertMaskBtn.addEventListener('click', invertMask);

  // Expand mask (opens lightbox inpaint)
  const maskExpandBtn = $('img-mask-expand');
  if (maskExpandBtn) maskExpandBtn.addEventListener('click', openLightboxInpaint);

  // Full-resolution inpaint checkbox
  const fullresCheckbox = $('img-fullres');
  const fullresPaddingRow = $('img-fullres-padding-row');
  if (fullresCheckbox) {
    fullresCheckbox.addEventListener('change', () => {
      if (fullresPaddingRow) fullresPaddingRow.classList.toggle('hidden', !fullresCheckbox.checked);
    });
  }

  // Inpaint mode buttons
  document.querySelectorAll('.img-inpaint-mode').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.img-inpaint-mode').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentInpaintMode = btn.dataset.mode || 'default';
      // Adjust strength based on mode
      const strengthEl = $('img-strength');
      const strengthValEl = $('img-strength-value');
      if (strengthEl) {
        if (currentInpaintMode === 'fill') { strengthEl.value = '1.0'; }
        else if (currentInpaintMode === 'original') { strengthEl.value = '0.5'; }
        else { strengthEl.value = '1.0'; }
        if (strengthValEl) strengthValEl.textContent = strengthEl.value;
      }
    });
  });

  // Fit to view
  const fitBtn = $('img-mask-fit');
  if (fitBtn) fitBtn.addEventListener('click', maskEditorFitToView);

  // Canvas pointer events (on the UI layer — top canvas)
  const uiCanvas = $('img-mask-ui-canvas');
  if (uiCanvas) {
    uiCanvas.addEventListener('pointerdown', maskEditorPointerDown);
    uiCanvas.addEventListener('pointermove', maskEditorPointerMove);
    uiCanvas.addEventListener('pointerup', maskEditorPointerUp);
    uiCanvas.addEventListener('pointerleave', maskEditorPointerLeave);
    uiCanvas.addEventListener('wheel', maskEditorWheel, { passive: false });
  }

  // Keyboard shortcuts for mask editor
  document.addEventListener('keydown', maskEditorKeyDown);
  document.addEventListener('keyup', maskEditorKeyUp);

  // Advanced section collapse
  const advToggle = $('img-advanced-toggle');
  const advSec = $('img-advanced-section');
  if (advToggle && advSec) {
    if (!imgSettings.advancedCollapsed && imgSettings.advancedCollapsed !== false) {
      imgSettings.advancedCollapsed = true; // default collapsed
    }
    if (!imgSettings.advancedCollapsed) {
      advSec.classList.remove('collapsed');
      advToggle.classList.remove('collapsed');
    }
    advToggle.addEventListener('click', () => {
      const isCollapsed = !advSec.classList.contains('collapsed');
      advSec.classList.toggle('collapsed', isCollapsed);
      advToggle.classList.toggle('collapsed', isCollapsed);
      imgSettings.advancedCollapsed = isCollapsed;
      saveImgSettings();
    });
  }

  // Gallery collapse
  const galleryToggle = $('img-gallery-toggle');
  const galleryEl = $('img-gallery');
  if (galleryToggle && galleryEl) {
    if (imgSettings.galleryCollapsed) {
      galleryToggle.classList.add('collapsed');
      galleryEl.classList.add('collapsed');
    }
    galleryToggle.addEventListener('click', () => {
      const isCollapsed = galleryToggle.classList.toggle('collapsed');
      galleryEl.classList.toggle('collapsed', isCollapsed);
      imgSettings.galleryCollapsed = isCollapsed;
      saveImgSettings();
    });
  }

  // Toggle panel — close other right-side panels to avoid overlap
  toggleBtn.addEventListener('click', () => {
    const opening = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    imgSettings.panelOpen = !panel.classList.contains('hidden');
    saveImgSettings();
    if (imgSettings.panelOpen) {
      app.closeInspector();
      if (window.innerWidth < 1024) app.closePanel();
      fetchImageHardware();
      refreshImageModels();
      // Resolve availability BEFORE the first gallery render so the
      // empty-state setup card can take over when no path exists. The
      // gallery render falls through to the normal placeholder otherwise.
      fetchImageAvailability().then(() => refreshImageGallery());
      refreshImageCatalog();
      fetchImageSamplers();
      refreshCondenseModels();
      renderResolutionPresets();
      if (opening) ViewStack.pushOverlay('image', { onClose: closeImagePanel });
    } else {
      // Panel closed — stop any in-flight stage polling
      _stopStagePolling();
      if (ViewStack.hasOverlay('image')) ViewStack.popOverlay('image');
    }
  });

  // Model change updates resolution presets
  const modelSelect = $('img-model');
  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      const ptype = getSelectedPipelineType();
      const presets = IMG_RESOLUTION_PRESETS[ptype] || IMG_RESOLUTION_PRESETS.sd15;
      const widthEl = $('img-width');
      const heightEl = $('img-height');
      if (widthEl) widthEl.value = presets[0].w;
      if (heightEl) heightEl.value = presets[0].h;
      renderResolutionPresets();
      updateCloudLocalFields();
      applyModelDefaults();
    });
  }

  // Clear preset highlight on manual edit
  const widthEl = $('img-width');
  const heightEl = $('img-height');
  function clearPresetHighlight() {
    const container = $('img-resolution-presets');
    if (container) container.querySelectorAll('.img-resolution-preset').forEach(b => b.classList.remove('active'));
  }
  if (widthEl) widthEl.addEventListener('input', clearPresetHighlight);
  if (heightEl) heightEl.addEventListener('input', clearPresetHighlight);

  // Persist on change
  ['img-width', 'img-height', 'img-steps', 'img-cfg', 'img-seed', 'img-negative'].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('input', saveImageFormToSettings);
  });
  ['img-sampler', 'img-model', 'img-preset', 'img-condense-model'].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('change', saveImageFormToSettings);
  });

  // Unload model
  const unloadBtn = $('img-unload-btn');
  if (unloadBtn) {
    unloadBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/image/unload', { method: 'POST' });
        const data = await resp.json();
        if (data.unloaded) showToast('Model unloaded: ' + data.model, 'success');
        else showToast(data.reason || 'No model loaded', 'info');
      } catch (err) { showToast('Unload failed: ' + err.message, 'error'); }
    });
  }

  // Rename model
  const renameBtn = $('img-rename-btn');
  if (renameBtn) renameBtn.addEventListener('click', handleImageModelRename);

  // Expand prompt
  const expandBtn = $('img-expand-btn');
  if (expandBtn) {
    expandBtn.addEventListener('click', () => {
      const promptEl = $('img-prompt');
      if (!promptEl) return;
      const expanded = promptEl.classList.toggle('expanded');
      expandBtn.classList.toggle('active', expanded);
      promptEl.rows = expanded ? 12 : 3;
    });
  }

  // Enhance prompt
  const enhanceBtn = $('img-enhance-btn');
  if (enhanceBtn) {
    enhanceBtn.addEventListener('click', async () => {
      const promptEl = $('img-prompt');
      if (!promptEl || !promptEl.value.trim()) { showToast('Enter a prompt first', 'warning'); return; }
      enhanceBtn.classList.add('enhancing');
      enhanceBtn.disabled = true;
      try {
        const bodyObj = { prompt: promptEl.value };
        const condenseEl = $('img-condense-model');
        if (condenseEl && condenseEl.value) bodyObj.model = condenseEl.value;
        const resp = await fetch('/api/image/enhance-prompt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bodyObj),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          showToast(err.detail || 'Enhancement failed', 'error');
          return;
        }
        const data = await resp.json();
        if (data.prompt) { promptEl.value = data.prompt; showToast('Prompt enhanced', 'success'); }
      } catch { showToast('Enhancement failed', 'error'); }
      finally { enhanceBtn.classList.remove('enhancing'); enhanceBtn.disabled = false; }
    });
  }

  // Generate negative prompt
  const negGenBtn = $('img-negative-gen-btn');
  if (negGenBtn) {
    negGenBtn.addEventListener('click', async () => {
      const promptEl = $('img-prompt');
      if (!promptEl || !promptEl.value.trim()) { showToast('Enter a positive prompt first', 'warning'); return; }
      negGenBtn.classList.add('generating');
      negGenBtn.disabled = true;
      try {
        const bodyObj = { prompt: promptEl.value };
        const condenseEl = $('img-condense-model');
        if (condenseEl && condenseEl.value) bodyObj.model = condenseEl.value;
        const resp = await fetch('/api/image/generate-negative', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bodyObj),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          showToast(err.detail || 'Negative prompt generation failed', 'error');
          return;
        }
        const data = await resp.json();
        if (data.negative_prompt) {
          const negEl = $('img-negative');
          if (negEl) { negEl.value = data.negative_prompt; saveImageFormToSettings(); }
          showToast('Negative prompt generated', 'success');
        }
      } catch { showToast('Negative prompt generation failed', 'error'); }
      finally { negGenBtn.classList.remove('generating'); negGenBtn.disabled = false; }
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      panel.classList.add('hidden');
      imgSettings.panelOpen = false;
      saveImgSettings();
    });
  }

  if (generateBtn) generateBtn.addEventListener('click', handleImageGenerate);

  // Batch count button group
  const batchGroup = $('img-batch-group');
  if (batchGroup) {
    batchGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('.img-batch-btn');
      if (!btn) return;
      batchGroup.querySelectorAll('.img-batch-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }

  const cancelBtn = $('img-cancel-btn');
  if (cancelBtn) {
    cancelBtn.addEventListener('click', async () => {
      if (imageAbortController) imageAbortController.abort();
      try { await fetch('/api/image/cancel', { method: 'POST' }); } catch { /* best effort */ }
    });
  }

  // --- Quality & Speed event listeners ---
  _initQualitySection();

  if (pullBtn) pullBtn.addEventListener('click', handleImagePull);
  // Reset detect state when input changes
  const pullInput = $('img-pull-input');
  if (pullInput) pullInput.addEventListener('input', _resetPullState);
  // Custom import panel (upload tab alongside URL tab)
  _initCustomImportPanel();
  if (openLibraryBtn) openLibraryBtn.addEventListener('click', openImageLibrary);

  // Init lightbox actions
  initLightboxActions();

  // Restore panel visibility — close inspector if image panel was left open
  if (imgSettings.panelOpen) {
    app.closeInspector();
    panel.classList.remove('hidden');
    fetchImageHardware();
    refreshImageModels();
    refreshImageGallery();
    refreshImageCatalog();
    refreshLoraList();
    initCatalogToggle();
    fetchImageSamplers();
    refreshCondenseModels();
    renderResolutionPresets();
  }

  // Reconnect any in-progress downloads
  reconnectPullTasks();

  // --- IP-Adapter Reference Images (supports multiple) ---
  const _ipRefImages = [];  // array of base64 data URLs

  const refThumb = $('img-ref-thumb');
  const refUpload = $('img-ref-upload');
  const refPreview = $('img-ref-preview');
  const refPlaceholder = $('img-ref-placeholder');
  const refScale = $('img-ref-scale');
  const refScaleVal = $('img-ref-scale-val');
  const refClear = $('img-ref-clear');

  function _addRefImage(dataUrl) {
    _ipRefImages.push(dataUrl);
    _renderRefThumbs();
  }

  function _removeRefImage(index) {
    _ipRefImages.splice(index, 1);
    _renderRefThumbs();
  }

  function _clearAllRefImages() {
    _ipRefImages.length = 0;
    _renderRefThumbs();
  }

  function _renderRefThumbs() {
    if (!refPreview) return;
    // Clear existing thumbnails
    const container = refPreview.parentElement;
    container.querySelectorAll('.img-ref-multi-thumb').forEach(el => el.remove());

    if (_ipRefImages.length === 0) {
      refPreview.src = '';
      refPreview.classList.add('hidden');
      if (refPlaceholder) refPlaceholder.classList.remove('hidden');
      if (refClear) refClear.classList.add('hidden');
      return;
    }

    if (refPlaceholder) refPlaceholder.classList.add('hidden');
    if (refClear) refClear.classList.remove('hidden');

    // Show first image in the main preview
    refPreview.src = _ipRefImages[0];
    refPreview.classList.remove('hidden');

    // Add additional thumbnails for multi-ref
    if (_ipRefImages.length > 1) {
      const thumbRow = document.createElement('div');
      thumbRow.className = 'img-ref-multi-thumb';
      thumbRow.style.cssText = 'display:flex;gap:4px;margin-top:4px;flex-wrap:wrap';
      _ipRefImages.forEach((url, i) => {
        const thumb = document.createElement('div');
        thumb.style.cssText = 'position:relative;width:36px;height:36px;border-radius:4px;overflow:hidden;border:1px solid var(--border-light);cursor:pointer';
        const img = document.createElement('img');
        img.src = url;
        img.style.cssText = 'width:100%;height:100%;object-fit:cover';
        thumb.appendChild(img);
        // Remove button
        const rm = document.createElement('div');
        rm.textContent = '\u00D7';
        rm.style.cssText = 'position:absolute;top:-2px;right:1px;font-size:12px;color:white;text-shadow:0 1px 2px rgba(0,0,0,0.8);cursor:pointer;line-height:1';
        rm.addEventListener('click', (e) => { e.stopPropagation(); _removeRefImage(i); });
        thumb.appendChild(rm);
        thumbRow.appendChild(thumb);
      });
      container.appendChild(thumbRow);
    }
  }

  if (refThumb && refUpload) {
    refThumb.addEventListener('click', () => refUpload.click());

    // Drag and drop — supports multiple files
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
      refThumb.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(evt => {
      refThumb.addEventListener(evt, () => refThumb.classList.add('drag-over'));
    });
    ['dragleave', 'drop'].forEach(evt => {
      refThumb.addEventListener(evt, () => refThumb.classList.remove('drag-over'));
    });
    refThumb.addEventListener('drop', e => {
      const files = [...(e.dataTransfer.files || [])].filter(f => f.type.startsWith('image/'));
      files.forEach(file => {
        const reader = new FileReader();
        reader.onload = (ev) => _addRefImage(ev.target.result);
        reader.readAsDataURL(file);
      });
    });

    refUpload.addEventListener('change', (e) => {
      const files = [...(e.target.files || [])].filter(f => f.type.startsWith('image/'));
      files.forEach(file => {
        const reader = new FileReader();
        reader.onload = (ev) => _addRefImage(ev.target.result);
        reader.readAsDataURL(file);
      });
      refUpload.value = "";
    });

    // Allow multiple file selection
    refUpload.setAttribute('multiple', '');

    if (refScale) {
      refScale.addEventListener('input', () => {
        if (refScaleVal) refScaleVal.textContent = (refScale.value / 100).toFixed(2);
      });
    }

    if (refClear) {
      refClear.addEventListener('click', _clearAllRefImages);
    }
  }

  // Expose IP-Adapter data for the generate handler (closure bridge)
  // Returns single string for 1 image, array for multiple, empty string for none
  initImagePanel._getIpRef = () => {
    const scale = refScale ? refScale.value / 100 : 0.55;
    if (_ipRefImages.length === 0) return { data: '', scale };
    if (_ipRefImages.length === 1) return { data: _ipRefImages[0], scale };
    return { data: [..._ipRefImages], scale };
  };
}

async function reconnectPullTasks() {
  try {
    const resp = await fetch('/api/image/models/pull');
    if (!resp.ok) return;
    const tasks = await resp.json();
    if (!tasks || !tasks.length) return;

    for (const t of tasks) {
      if (t.status !== 'running') continue;
      const progressArea = $('img-pull-progress');
      const fill = $('img-pull-fill');
      const statusEl = $('img-pull-status');
      if (progressArea) progressArea.classList.remove('hidden');
      if (statusEl) statusEl.textContent = 'Resuming download...';
      if (fill) fill.style.width = (t.percent || 0) + '%';

      pollPullTask(t.task_id,
        data => {
          if (data.percent !== undefined) {
            if (fill) fill.style.width = data.percent + '%';
            let pctText = Math.round(data.percent) + '%';
            if (data.files_done && data.files_total) pctText += ' (' + data.files_done + '/' + data.files_total + ' files)';
            if (statusEl) statusEl.textContent = pctText;
          }
        },
        () => {
          if (fill) fill.style.width = '100%';
          if (statusEl) statusEl.textContent = 'Complete!';
          refreshImageModels();
          refreshImageCatalog();
          setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); }, 3000);
        },
        errMsg => {
          if (statusEl) statusEl.textContent = 'Error: ' + errMsg;
          setTimeout(() => { if (progressArea) progressArea.classList.add('hidden'); }, 4000);
        }
      );
      break; // Only show one active download
    }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Export: called by chat module for inline image generation
// ---------------------------------------------------------------------------

export function getImageSettings() {
  return {
    negative: ($('img-negative') || {}).value || '',
    preset: ($('img-preset') || {}).value || '',
    width: parseInt(($('img-width') || {}).value) || 512,
    height: parseInt(($('img-height') || {}).value) || 512,
    steps: parseInt(($('img-steps') || {}).value) || 20,
    cfg: parseFloat(($('img-cfg') || {}).value) || 7.0,
    seed: parseInt(($('img-seed') || {}).value) || -1,
    sampler: ($('img-sampler') || {}).value || '',
    model: ($('img-model') || {}).value || '',
  };
}

/** Close the image panel (used by other modules to avoid overlap on mobile). */
export function closeImagePanel() {
  const panel = document.getElementById('image-panel');
  if (panel && !panel.classList.contains('hidden')) {
    panel.classList.add('hidden');
    imgSettings.panelOpen = false;
    saveImgSettings();
  }
  // Stop stage polling — generation may have been in-flight when panel closed
  _stopStagePolling();
  // Pop from ViewStack if we were tracked there (idempotent; fine if not).
  if (ViewStack.hasOverlay('image')) ViewStack.popOverlay('image');
}

export { openLightbox, refreshImageGallery };

/**
 * Architect handoff — open the image panel, fill the form from the
 * architect's inferred settings, and trigger generation. Called from
 * intent-action-router on a ``image.generate`` channel event.
 *
 * Payload shape mirrors image_generations columns:
 *   { prompt, model?, negative_prompt?, width?, height?, steps?,
 *     cfg_scale?, preset? }
 *
 * Missing fields fall back to whatever the form currently shows
 * (which itself fell back to the loaded imgSettings on init). The
 * architect intentionally leaves them unset when the user has no
 * image history yet — the form's own defaults take over cleanly.
 */
export function generateFromArchitect(payload) {
  if (!payload || typeof payload !== 'object') return false;

  // Open the panel if it's hidden so the user sees what's happening.
  // The toggle handler attaches inside initImagePanel — calling click
  // here would be hacky, so flip the class directly + replicate the
  // necessary side-effects (fetch hardware + refresh models) so the
  // first generation has the data it needs.
  const panel = document.getElementById('image-panel');
  if (panel && panel.classList.contains('hidden')) {
    panel.classList.remove('hidden');
    try { imgSettings.panelOpen = true; saveImgSettings(); } catch (_) {}
    try { fetchImageHardware(); } catch (_) {}
    try { refreshImageModels(); } catch (_) {}
  }

  // Fill form fields when the architect supplied them. Keep existing
  // values when not (caller's intent: "use what I had").
  const set = (id, value) => {
    if (value === undefined || value === null || value === '') return;
    const el = document.getElementById(id);
    if (el) el.value = String(value);
  };

  // Model name resolution — image_generations stores the full path
  // ("/data/image_models/Lumina") but the dropdown's option.value is
  // the short name ("Lumina"). When the architect's inferrer pulls
  // model from the DB row, it sends the path; without this lookup
  // the dropdown silently rejects the value and keeps the previous
  // selection. Match by path OR by name endsWith, against the live
  // ``imageModelsData`` already loaded for the dropdown.
  let resolvedModel = payload.model;
  if (resolvedModel && Array.isArray(imageModelsData) && imageModelsData.length) {
    const raw = String(resolvedModel);
    // 1. Exact path match
    let hit = imageModelsData.find(m => m && m.path === raw);
    // 2. Endswith /<name>  for "/data/image_models/Lumina" -> "Lumina"
    if (!hit) {
      hit = imageModelsData.find(m => {
        if (!m || !m.name) return false;
        return raw.endsWith('/' + m.name) || raw.endsWith('\\' + m.name);
      });
    }
    // 3. Direct name match (already short — passthrough)
    if (!hit) {
      hit = imageModelsData.find(m => m && m.name === raw);
    }
    if (hit && hit.name) {
      if (hit.name !== resolvedModel) {
        console.debug('[image] architect path resolved', raw, '->', hit.name);
      }
      resolvedModel = hit.name;
    } else {
      // No match — clear so the form keeps its default rather than
      // submitting an empty path the backend can't dispatch.
      console.warn('[image] architect model not in dropdown, falling back to form default:', raw);
      resolvedModel = undefined;
    }
  }

  set('img-prompt', payload.prompt);
  set('img-negative', payload.negative_prompt);
  set('img-model', resolvedModel);
  set('img-preset', payload.preset);
  set('img-width', payload.width);
  set('img-height', payload.height);
  set('img-steps', payload.steps);
  set('img-cfg', payload.cfg_scale);

  // Trigger generation by clicking the existing button — keeps the
  // VRAM-courtesy check + abort wiring + UI state on the canonical path.
  // Arm the one-shot provenance flag first: this generation is
  // companion-initiated and the submit handler marks the body so.
  const btn = document.getElementById('img-generate-btn');
  if (btn) {
    _architectOriginPending = true;
    btn.click();
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Init — called from app.js
// ---------------------------------------------------------------------------

export function initImage() {
  loadImgSettings();
  initImagePanel();

  // Command composer / voice → open the image panel and populate the
  // prompt. We don't auto-fire generation: the user may want to tweak
  // style, aspect, or model before running. The Generate button is
  // right there — one extra click, but no runaway GPU.
  document.addEventListener('augmentum:generate-image', (e) => {
    const prompt = (e.detail?.prompt || '').trim();
    if (!prompt) return;
    const panel = document.getElementById('image-panel');
    if (panel && panel.classList.contains('hidden')) {
      app.closeInspector();
      if (window.innerWidth < 1024) app.closePanel();
      panel.classList.remove('hidden');
      imgSettings.panelOpen = true;
      saveImgSettings();
    }
    const promptEl = document.getElementById('img-prompt');
    if (promptEl) {
      promptEl.value = prompt;
      promptEl.focus();
      promptEl.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
}
